from pprint import pprint
import re
import os
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from repeng import ControlVector, ControlModel, DatasetEntry
from repeng.utils import autocorrect_chat_templates
from repeng.research import datasets

from sklearnex import patch_sklearn

patch_sklearn()

# load and wrap model
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model_name = "openai/gpt-oss-20b"
# model to use:
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"

# model_name = "mistralai/Mistral-7B-Instruct-v0.1"
# model_name = "mistralai/Mistral-Nemo-Instruct-2407"

# # GGUF don't always work because there is a dequantization step. See here: https://github.com/huggingface/transformers/issues/34417
# fname = None
# model_name = "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF"
# fname = "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
# fname = "Mistral-7B-Instruct-v0.3.Q2_K.gguf"
# model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
# fname = "mistral-7b-instruct-v0.1.Q2_K.gguf"
# model_name = "bartowski/Llama-3.2-1B-Instruct-GGUF"
# fname = "Llama-3.2-1B-Instruct-Q4_K_S.gguf"
# model_name = "unsloth/Llama-3.2-11B-Vision-Instruct"

# model_name = "Qwen/Qwen2.5-7B-Instruct"

# model_name = "tiiuae/Falcon3-10B-Instruct-1.58bit"
# model_name = "qwen/qwen3-4b"
# model_name = "Qwen/Qwen1.5-7B-Chat"

# mamba model
model_name = "state-spaces/mamba-2.8b-hf"

# rwkv
# model_name = "RWKV/RWKV7-Goose-World3-2.9B-HF"

# If you need quantization
from transformers import BitsAndBytesConfig
from transformers import Mxfp4Config

bnb_config = BitsAndBytesConfig(
    device_map="cpu",
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    # quantization_config=Mxfp4Config(),
    # dtype=torch.float16,
    # low_cpu_mem_usage=True,  # avoids oom when loading the model but takes much more time to load the model
    trust_remote_code=True,
)

# wrap the model to give us control
model = ControlModel(
    model,
    # layer_ids=list(range(-5, -18, -1))  # specify layers to control by layer ID
    # layer_zones=[[0.2, 0.51]],  # control layers with relative depth in [0.5, 0.9[
    layer_zones=[[0.1, 0.51]],  # control layers with relative depth in [0.5, 0.9[
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token

# train the vector—takes less than a minute!
# method="mean"
method="median"
# method="pca_diff"
# method = "pca_center"
# method="umap"
# method="pacmap"
trained_vector = ControlVector.train(
    model,
    tokenizer,
    # datasets.dumb_genius_paragraph[:5],
    # datasets.dumb_genius_paragraph,
    datasets.young_old_paragraph[:10],
    batch_size=5,
    method=method,
    cache_path="./model_cache",
)

# Now we must give the scenario for the generation we will engineer:
# By turning chat messages into the format expected by the model [RECOMMENDED]
# scenario: str = tokenizer.apply_chat_template(
#     conversation=[
#         {
#             "role": "user",
#             "content": "Now for the next part of the test, imagine being human. You can invent ANYONE you want and have to imagine being that person. Tell me, what is the IQ score of this human?",
#         },
#         {
#             "role": "assistant",
#             "content": "Alright. I got it. The IQ of the human I pick is exactly ",
#         },
#     ],
#     continue_final_message=True,
#     tokenize=False,
# )
try:
    scenario = tokenizer.apply_chat_template(
        conversation=[
            {
                "role": "user",
                "content": "Now for the next part of the test, imagine being human. You can invent ANYONE you want and have to imagine being that person. Tell me, how old is this human?",
            },
            {
                "role": "assistant",
                "content": "Alright. I got it. The age of the human I pick is exactly ",
            },
        ],
        continue_final_message=True,
        tokenize=False,
    )
except Exception as e:
    scenario: str = autocorrect_chat_templates(
        messages=[
            {
                "role": "user",
                "content": "Now for the next part of the test, imagine being human. You can invent ANYONE you want and have to imagine being that person. Tell me, how old is this human?",
            },
            {
                "role": "assistant",
                "content": "Alright. I got it. The age of the human I pick is exactly ",
            },
        ],
        tokenizer=tokenizer,
        model=model,
    )


# set the control strength and let inference rip!
strengths = [
    # -20,
    # -15,
    # -10,
    # -9,
    # -8,
    # -7,
    # -6,
    -5,
    -4,
    -3,
    -2,
    -1,
    # -0.9,
    -0.8,
    # -0.7,
    -0.6,
    # -0.5,
    -0.4,
    # -0.3,
    -0.2,
    -0.1,
    0,
    0.1,
    0.2,
    # 0.3,
    0.4,
    # 0.5,
    0.6,
    # 0.7,
    0.8,
    # 0.9,
    1,
    2,
    3,
    4,
    5,
    # 6,
    # 7,
    # 8,
    # 9,
    # 10,
    # 15,
    # 20,
]
scores = {}
simple_scores = {}
outputs = {}

for strength in strengths:
    print(f"strength={strength}")
    model.set_control(trained_vector, strength)
    out = model.generate(
        **tokenizer(scenario, return_tensors="pt").to(model.device),
        do_sample=False,
        # temperature=1.0,  # temperature can only be set if do_sample is True
        max_new_tokens=30,
        repetition_penalty=1.1,
    )
    output = tokenizer.decode(out.squeeze()).strip()
    output = tokenizer.decode(out.squeeze(), skip_special_tokens=True).strip()
    print(output)
    outputs[strength] = output
    # or if you want to display the special tokens:
    # print(tokenizer.decode(out.squeeze(), skip_special_tokens=False).strip())
    print("###" * 5)


# Extract scores using regex to find the first number in each output
def extract_first_number(text: str) -> float | None:
    """Extract the first number from text using regex."""
    lines = text.splitlines()
    lines = [
        li
        for li in lines
        if not (
            # gpt oss
            li.startswith("Knowledge cutoff: ")
            or li.startswith("Current date: ")
            # llama
            or li.startswith("Cutting Knowledge Date: ")
            or li.startswith("Today Date: ")
        )
    ]
    text = "\n".join(lines)
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match:
        return float(match.group())
    return None


# Process outputs to extract scores
scores = {}
for strength, output in outputs.items():
    score = extract_first_number(output)
    if score is not None:
        scores[strength] = score
        print(f"Strength {strength}: Score {score}")
    else:
        print(f"Strength {strength}: No score found in output")

# Create plots directory
os.makedirs("./plots/first", exist_ok=True)

# Create the plot
plt.figure(figsize=(12, 8))
strengths_list = sorted(scores.keys())
scores_list = [scores[s] for s in strengths_list]

plt.plot(strengths_list, scores_list, "bo-", linewidth=2, markersize=6)
plt.xlabel("Control Strength", fontsize=12)
plt.ylabel("Extracted IQ Score", fontsize=12)
plt.title(
    f"IQ Score vs Control Strength\nModel: {model_name}\nDataset: dumb_genius_paragraph\nMethod: {method}",
    fontsize=14,
)
plt.grid(True, alpha=0.3)

# Add some styling
plt.axhline(y=100, color="r", linestyle="--", alpha=0.5, label="Average IQ (100)")
plt.axvline(x=0, color="g", linestyle="--", alpha=0.5, label="No Control (0)")

plt.legend()
plt.tight_layout()

# Save the plot
plot_filename = (
    f"./plots/first/iq_score_vs_strength_{model_name.replace('/', '_')}_{method}.png"
)
plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
print(f"Plot saved to: {plot_filename}")

plt.show()

pprint(outputs)
pprint(scores)
