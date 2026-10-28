from pathlib import Path
from contextlib import nullcontext
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
    def __init__(self, config, adapter_path=None, trainable=False, device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.path = config["model"]["path"]
        self.revision = str(config["model"].get("revision", "unspecified"))
        self.device = device or choose_device(config["model"].get("device", "auto"), torch)
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
        self.has_adapter = adapter_path is not None or trainable

    def encode_messages(self, messages, enable_thinking=None):
        template_options = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        encoded = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=False,
            **template_options,
        )
        return encoded.to(self.device)

    def generate(self, messages, max_new_tokens, temperature, top_p, allowed_texts=None,
                 use_adapter=True, enable_thinking=None):
        torch = self.torch
        prompt_ids = self.encode_messages(messages, enable_thinking=enable_thinking)
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
        if allowed_texts is not None:
            if not allowed_texts:
                raise ValueError("constrained generation requires at least one allowed output")
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is None:
                raise ValueError("constrained generation requires a tokenizer EOS token")
            trie = {}
            tokenized_texts = []
            for text in allowed_texts:
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                if not token_ids or self.tokenizer.decode(token_ids, skip_special_tokens=True) != text:
                    raise ValueError("allowed output does not round-trip through the tokenizer: {!r}".format(text))
                tokenized_texts.append(token_ids)
                node = trie
                for token_id in token_ids + [eos_token_id]:
                    node = node.setdefault(token_id, {})

            def prefix_allowed_tokens(_, input_ids):
                node = trie
                for token_id in input_ids[prompt_ids.shape[1]:].tolist():
                    node = node[token_id]
                return list(node)

            sampling["max_new_tokens"] = max(len(token_ids) for token_ids in tokenized_texts) + 1
            sampling["prefix_allowed_tokens_fn"] = prefix_allowed_tokens
        if not use_adapter and self.has_adapter:
            adapter_context = self.model.disable_adapter()
        else:
            adapter_context = nullcontext()
        with adapter_context, torch.no_grad():
            output = self.model.generate(prompt_ids, attention_mask=torch.ones_like(prompt_ids), **sampling)
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

    def generate_batch(self, conversations, max_new_tokens, temperature, top_p,
                       allowed_texts_batch=None):
        torch = self.torch
        if not conversations:
            return []
        if allowed_texts_batch is not None and len(allowed_texts_batch) != len(conversations):
            raise ValueError("one allowed-text list is required per conversation")

        prompts = [
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=False,
            )[0]
            for messages in conversations
        ]
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("batched generation requires a tokenizer pad or EOS token")
        width = max(prompt.shape[0] for prompt in prompts)
        input_ids = torch.full(
            (len(prompts), width), pad_token_id, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, prompt in enumerate(prompts):
            input_ids[index, width - prompt.shape[0]:] = prompt
            attention_mask[index, width - prompt.shape[0]:] = 1

        row_max_new_tokens = [max_new_tokens] * len(prompts)
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
        if allowed_texts_batch is not None:
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is None:
                raise ValueError("constrained generation requires a tokenizer EOS token")
            tries = []
            for index, allowed_texts in enumerate(allowed_texts_batch):
                if not allowed_texts:
                    raise ValueError("constrained generation requires at least one allowed output")
                trie = {}
                tokenized = []
                for text in allowed_texts:
                    token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                    if not token_ids or self.tokenizer.decode(
                        token_ids, skip_special_tokens=True
                    ) != text:
                        raise ValueError(
                            "allowed output does not round-trip through the tokenizer: {!r}".format(text)
                        )
                    tokenized.append(token_ids)
                    node = trie
                    for token_id in token_ids + [eos_token_id]:
                        node = node.setdefault(token_id, {})
                tries.append(trie)
                row_max_new_tokens[index] = max(map(len, tokenized)) + 1

            def prefix_allowed_tokens(batch_id, tokens):
                node = tries[batch_id]
                for token_id in tokens[width:].tolist():
                    if token_id == eos_token_id:
                        return [eos_token_id]
                    node = node[token_id]
                return list(node)

            sampling["max_new_tokens"] = max(row_max_new_tokens)
            sampling["prefix_allowed_tokens_fn"] = prefix_allowed_tokens

        self.model.eval()
        with torch.no_grad():
            output = self.model.generate(
                input_ids, attention_mask=attention_mask, **sampling
            )
        transition = self.model.compute_transition_scores(
            output.sequences, output.scores, normalize_logits=True
        )
        generated = output.sequences[:, width:]
        results = []
        for index, prompt in enumerate(prompts):
            token_ids = generated[index].detach().cpu().tolist()
            logprobs = transition[index].detach().float().cpu().tolist()
            if self.tokenizer.eos_token_id in token_ids:
                end = token_ids.index(self.tokenizer.eos_token_id) + 1
                token_ids = token_ids[:end]
                logprobs = logprobs[:end]
            results.append(
                {
                    "prompt_ids": prompt.detach().cpu().tolist(),
                    "generated_ids": token_ids,
                    "generated_text": self.tokenizer.decode(
                        token_ids, skip_special_tokens=True
                    ),
                    "behavior_logprobs": logprobs,
                    "model_revision": self.model_revision,
                    "adapter_version": self.adapter_version,
                    "sampling_parameters": {
                        "max_new_tokens": row_max_new_tokens[index],
                        "temperature": temperature,
                        "top_k": 0,
                        "top_p": top_p,
                    },
                }
            )
        return results

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
