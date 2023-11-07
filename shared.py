import json
from typing import Dict, Optional
from joblib import Memory
import dotenv
import logging

dotenv.load_dotenv()

FUNCTION_CALL_TAG = "<function>"
FUNCTION_ARGS_TAG = "<arguments>"

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def cache_model(model_name):
    import os

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "true"

    from dotenv import load_dotenv

    load_dotenv()

    from huggingface_hub import snapshot_download

    snapshot_download(model_name)
    print(f"Cached {model_name}")


def serialize_input(inputs: dict) -> str:
    # print("Serializing input", inputs)
    functions = None

    function_call = inputs.get("function_call")
    input_functions = inputs.get("functions")

    if function_call == "none":
        functions = None
    elif isinstance(function_call, dict) and "name" in function_call:
        functions = [function_call["name"]]
    elif input_functions:
        functions = [fn["name"] for fn in input_functions]

    # Reorder the keys within each message to be `role`, `content`, `function_call`
    def correct_key_order(message):
        message = {
            "role": message["role"],
            "content": message["content"],
        }
        if "function_call" in message:
            message["function_call"] = message["function_call"]
        return message

    corrected_messages = [correct_key_order(m) for m in inputs["messages"]]

    to_serialize = {"messages": corrected_messages}

    if functions:
        to_serialize["functions"] = functions

    serialized = json.dumps(to_serialize, separators=(",", ":"))

    return f"### Instruction:\n{serialized}\n\n### Response:\n"


# Can't just use the polars `str.json_extract` because it aggressively
# drops null columns https://github.com/jorgecarleitao/arrow2/issues/1459
def load_json_preserve_nulls(x: str) -> dict:
    try:
        obj = json.loads(x)
        assert obj is not None
        assert isinstance(obj, dict)
        return obj
    except json.JSONDecodeError:
        print(f"Failed to parse {x}")
        return {}


def serialize_chat_output(output: dict) -> str:
    formatted = ""
    if "function_call" in output:
        formatted = FUNCTION_CALL_TAG + output["function_call"]["name"]
        if "arguments" in output["function_call"]:
            formatted += FUNCTION_ARGS_TAG + output["function_call"]["arguments"]
    else:
        formatted = output.get("content", "")
    return formatted


def deserialize_chat_output(output: str):
    if not output.strip().startswith(FUNCTION_CALL_TAG):
        return {"role": "assistant", "content": output}

    fn_call_name = output.split(FUNCTION_CALL_TAG)[1].split(FUNCTION_ARGS_TAG)[0]
    args = output.split(FUNCTION_ARGS_TAG)[1]

    return {
        "role": "assistant",
        "function_call": {"name": fn_call_name, "arguments": args},
    }


def calculate_accuracy(
    row: Dict[str, Optional[Dict[str, Optional[Dict[str, str]]]]]
) -> Optional[float]:
    # print(row)
    if row["gold"] is None or row["gold"]["function_call"] is None:
        return None

    if row["prediction"] is None or row["prediction"]["function_call"] is None:
        return 0

    try:
        gold_args = json.loads(row["gold"]["function_call"]["arguments"])
    except json.JSONDecodeError:
        print("Error, gold args not valid JSON")
        print(row["gold"]["function_call"]["arguments"])
        return None
    try:
        pred_args = json.loads(row["prediction"]["function_call"]["arguments"])
    except json.JSONDecodeError:
        print("Error, prediction args not valid JSON")
        print(row["prediction"]["function_call"]["arguments"])
        return 0

    total_fields = len(gold_args)
    correct_fields = len([f for f in gold_args if gold_args[f] == pred_args.get(f)])

    if total_fields == 0:
        return 1
    else:
        return correct_fields / total_fields


memory = Memory("/workspace/cache/get_completions", verbose=0)


@memory.cache
def get_completions(model_id, inputs):
    from vllm import LLM, SamplingParams

    model = LLM(model=model_id)

    outputs = model.generate(
        inputs,
        SamplingParams(temperature=0, max_tokens=1000, logprobs=1),
    )

    return outputs
