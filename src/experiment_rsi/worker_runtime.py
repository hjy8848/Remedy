import contextlib
import io
import json
import random
import sys
import traceback


protocol = sys.__stdout__
source_path = sys.argv[1]
random.seed(int(sys.argv[2]))
program = None
MAX_RESULT_BYTES = 1024 * 1024
program_error = None


def send(value):
    protocol.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    protocol.flush()


send({"ready": True})

for line in sys.stdin:
    request = json.loads(line)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            if program is None:
                with open(source_path, "r", encoding="utf-8") as stream:
                    source = stream.read()
                namespace = {"__name__": "candidate_policy"}
                exec(compile(source, source_path, "exec"), namespace)
                program = namespace["step"]
                if not callable(program):
                    raise TypeError("candidate step must be callable")
            result = program(request["view"], request["state"])
        if not isinstance(result, dict) or "kind" not in result or "state" not in result:
            raise TypeError("step must return a mapping with kind and state")
        if not isinstance(result["state"], dict):
            raise TypeError("step state must be a JSON object")
        if result["kind"] == "action":
            if set(result) != {"kind", "action", "state"} or not isinstance(result["action"], str):
                raise TypeError("action result must contain kind, action, and state text")
        elif result["kind"] == "model":
            if (
                set(result) != {"kind", "messages", "state"}
                or not isinstance(result["messages"], list)
                or not result["messages"]
            ):
                raise TypeError("model result must contain kind, messages, and state")
            for message in result["messages"]:
                if (
                    not isinstance(message, dict)
                    or message.get("role") not in ("system", "user", "assistant")
                    or not isinstance(message.get("content"), str)
                ):
                    raise TypeError("each model message needs a supported role and text content")
        else:
            raise ValueError("step kind must be action or model")
        serialized = json.dumps(result, ensure_ascii=False, allow_nan=False)
        if len(serialized.encode("utf-8")) > MAX_RESULT_BYTES:
            raise ValueError("candidate result exceeds 1 MiB")
        send({"result": result})
    except BaseException as error:
        message = "{}: {}".format(type(error).__name__, error)
        send({"candidate_error": message, "traceback": traceback.format_exc(limit=4)})
