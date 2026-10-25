from pathlib import Path
import random


def seed_everything(seed, torch=None):
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def choose_device(requested, torch):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelSession:
    def __init__(self, config, adapter_path=None, trainable=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.path = config["model"]["path"]
        self.revision = str(config["model"].get("revision", "unspecified"))
        self.device = choose_device(config["model"].get("device", "auto"), torch)
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config["model"]["dtype"]]
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, revision=None if self.revision == "local" else self.revision)
        base = AutoModelForCausalLM.from_pretrained(
            self.path,
            revision=None if self.revision == "local" else self.revision,
            torch_dtype=dtype,
        )

        if adapter_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(base, adapter_path, is_trainable=trainable)
        elif trainable:
            from peft import LoraConfig, get_peft_model

            lora = config["model"]["lora"]
            self.model = get_peft_model(
                base,
                LoraConfig(
                    r=lora["r"],
                    lora_alpha=lora["alpha"],
                    lora_dropout=lora["dropout"],
                    target_modules=lora["target_modules"],
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
        else:
            self.model = base

        if trainable and config["training"].get("gradient_checkpointing", False):
            base.gradient_checkpointing_enable()
            base.config.use_cache = False
            base.enable_input_require_grads()

        self.model.to(self.device)
        self.model_revision = self.revision
        self.adapter_version = str(adapter_path) if adapter_path else ("trainable-new" if trainable else "base")

    def encode_messages(self, messages):
        encoded = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=False
        )
        return encoded.to(self.device)

    def generate(self, messages, max_new_tokens, temperature, top_p):
        torch = self.torch
        prompt_ids = self.encode_messages(messages)
        self.model.eval()
        sampling = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "top_k": 0,
            "top_p": top_p,
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        if temperature > 0:
            sampling["temperature"] = temperature
        with torch.no_grad():
            output = self.model.generate(prompt_ids, **sampling)
        generated_ids = output.sequences[:, prompt_ids.shape[1]:]
        transition = self.model.compute_transition_scores(
            output.sequences, output.scores, normalize_logits=True
        )
        if generated_ids.shape[1] != transition.shape[1]:
            raise RuntimeError("generated token and behavior log-probability lengths differ")
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return {
            "prompt_ids": prompt_ids[0].detach().cpu().tolist(),
            "generated_ids": generated_ids[0].detach().cpu().tolist(),
            "generated_text": text,
            "behavior_logprobs": transition[0].detach().float().cpu().tolist(),
            "model_revision": self.model_revision,
            "adapter_version": self.adapter_version,
            "sampling_parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": 0,
                "top_p": top_p,
            },
        }

    def response_logprobs(self, prompt_ids, generated_ids, sampling_parameters):
        torch = self.torch
        self.model.eval()
        prompt = torch.tensor([prompt_ids], device=self.device)
        completion = torch.tensor([generated_ids], device=self.device)
        input_ids = torch.cat((prompt, completion), dim=1)
        logits = self.model(input_ids=input_ids).logits[0, prompt.shape[1] - 1:-1]
        temperature = sampling_parameters["temperature"]
        top_p = sampling_parameters["top_p"]
        if temperature > 0:
            logits = logits / temperature
        if sampling_parameters["temperature"] > 0 and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove = cumulative_probs > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            remove_mask = torch.zeros_like(remove).scatter(1, sorted_indices, remove)
            logits = logits.masked_fill(remove_mask, float("-inf"))
        token_logprobs = torch.log_softmax(logits, dim=-1).gather(1, completion[0].unsqueeze(1))
        return token_logprobs.squeeze(1)

    def sequence_logits(self, prompt_ids, generated_ids):
        prompt = self.torch.tensor([prompt_ids], device=self.device)
        completion = self.torch.tensor([generated_ids], device=self.device)
        input_ids = self.torch.cat((prompt, completion), dim=1)
        self.model.eval()
        return self.model(input_ids=input_ids).logits[0, prompt.shape[1] - 1:-1]

    def save_adapter(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
