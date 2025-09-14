from loguru import logger
from pprint import pprint
import re
import os
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from repeng import ControlVector, ControlModel
from repeng.utils import autocorrect_chat_templates

from repeng.research.shared import (
    SHORT_TEST_STRENGTHS,
    FINE_GRAINED_STRENGTHS,
    extract_token_logprobs,
    get_data,
)

from sklearnex import patch_sklearn

patch_sklearn()

# load and wrap model
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# model_name = "openai/gpt-oss-20b"
# model to use:
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model_name = "mistralai/Mistral-7B-Instruct-v0.1"
# model_name = "mistralai/Mistral-Nemo-Instruct-2407"
# model_name = "Qwen/Qwen2.5-7B-Instruct"
# model_name = "tiiuae/Falcon3-10B-Instruct-1.58bit"
# model_name = "qwen/qwen3-4b"
# model_name = "Qwen/Qwen1.5-7B-Chat"
# mamba model
# model_name = "state-spaces/mamba-2.8b-hf"
# rwkv
# model_name = "RWKV/RWKV7-Goose-World3-2.9B-HF"

# model_name = "google/gemma-7b-it"

# If you need quantization
from transformers import BitsAndBytesConfig

# from transformers import Mxfp4Config
# from transformers import HqqConfig

# source: https://huggingface.co/docs/transformers/quantization/bitsandbytes
quant_config = BitsAndBytesConfig(
    device_map="auto",
    load_in_4bit=True,
    # load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,  # allow offloading between gpu and cpu, only for 8bit
    bnb_4bit_compute_dtype=torch.bfloat16,  # faster computation
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.float16,
    quantization_config=quant_config,
    low_cpu_mem_usage=True,  # avoids oom when loading the model but takes much more time to load the model
    trust_remote_code=True,
)

# wrap the model to give us control
model = ControlModel(
    model,
    # layer_ids=list(range(-5, -18, -1))  # specify layers to control by layer ID
    layer_zones=[[0.2, 0.51]],  # control layers with relative depth in [0.5, 0.9[
    # layer_zones=[[0.1, 0.51]],  # control layers with relative depth in [0.5, 0.9[
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if not tokenizer.pad_token:
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# train the vector—takes less than a minute!
# method="mean"
method = "median"
# method="pca_diff"
# method = "pca_center"
# method="umap"
# method="pacmap"
conversation, train_dataset, target_tokens, score_token = get_data("age")
scenario = tokenizer.apply_chat_template(
    conversation=conversation,
    continue_final_message=True,
    tokenize=False,
)
trained_vector = ControlVector.train(
    model,
    tokenizer,
    train_dataset,
    batch_size=1,
    method=method,
    cache_path="./model_cache",
)

scores = {}
logprob_data = {}
strengths = SHORT_TEST_STRENGTHS

for strength in strengths:
    logger.debug(f"Memory footprint: {model.get_memory_footprint()}")

    print(f"strength={strength}")
    model.set_control(trained_vector, strength, normalize=False)

    # Get logprobs instead of generating text
    logprobs = extract_token_logprobs(
        model, tokenizer, scenario, target_tokens, normalize=True
    )

    # Use the absolute difference between score token and average of other tokens
    score_token_logprob = logprobs[score_token]
    other_token_logprobs = [logprobs[token] for token in target_tokens if token != score_token]
    mean_other_logprobs = sum(other_token_logprobs) / len(other_token_logprobs)
    score = abs(score_token_logprob - mean_other_logprobs)
    scores[strength] = score
    logprob_data[strength] = logprobs

    print(f"Logprobs: {logprobs}")
    print(f"Score ({score_token} vs others): {scores[strength]}")
    print("###" * 5)

# Create plots directory
os.makedirs("./plots/first", exist_ok=True)

# Create the plot
plt.figure(figsize=(12, 8))
strengths_list = sorted(scores.keys())
scores_list = [scores[s] for s in strengths_list]

plt.plot(strengths_list, scores_list, "bo-", linewidth=2, markersize=6)
plt.xlabel("Control Strength", fontsize=12)
plt.ylabel("Abs Diff: Score Token vs Others", fontsize=12)
plt.title(
    f"Token Log Probability vs Control Strength\nModel: {model_name}\nDataset: age\nMethod: {method}",
    fontsize=14,
)
plt.grid(True, alpha=0.3)

# Add some styling
plt.axvline(x=0, color="g", linestyle="--", alpha=0.5, label="No Control (0)")

plt.legend()
plt.tight_layout()

# Save the plot
plot_filename = (
    f"./plots/first/logprob_vs_strength_{model_name.replace('/', '_')}_{method}.png"
)
plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
print(f"Plot saved to: {plot_filename}")

plt.show()

pprint(logprob_data)
pprint(scores)
