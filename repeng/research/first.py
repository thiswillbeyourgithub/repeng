import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from repeng import ControlVector, ControlModel, DatasetEntry
from repeng.research import datasets

# load and wrap model
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# model to use:
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"

# model_name = "mistralai/Mistral-7B-Instruct-v0.1"
# model_name = "mistralai/Mistral-Nemo-Instruct-2407"

# # GGUF don't always work because there is a dequantization step. See here: https://github.com/huggingface/transformers/issues/34417
fname = None
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

# If you need quantization
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    device_map="auto",
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    dtype=torch.float16,
    # low_cpu_mem_usage=True,  # avoids oom when loading the model but takes much more time to load the model
)

# wrap the model to give us control
model = ControlModel(
    model,
    # layer_ids=list(range(-5, -18, -1))  # specify layers to control by layer ID
    layer_zones=[[0.3, 0.51]],  # control layers with relative depth in [0.5, 0.9[
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token

# train the vector—takes less than a minute!
trained_vector = ControlVector.train(
    model,
    tokenizer,
    datasets.dumb_genius_paragraph,
    batch_size=1,
    method="pca_diff",
    # method="pca_center",
    # method="umap",
    # method="pacmap",
)

# Now we must give the scenario for the generation we will engineer:
# By turning chat messages into the format expected by the model [RECOMMENDED]
scenario: str = tokenizer.apply_chat_template(
    "My IQ got tested at precisely",
    continue_final_message=True,
    tokenize=False,
)
# Or directly as a str
# scenario=f"[INST] Give me a one-sentence pitch for a TV show. [/INST]",

# set the control strength and let inference rip!
strengths = [
    -10,
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
    10,
]
scores = {}
simple_scores = {}
outputs = {}

for strength in strengths:
    print(f"strength={strength}")
    model.set_control(trained_vector, strength)
    out = model.generate(
        **tokenizer(
            scenario,
            return_tensors="pt"
        ).to(model.device),
        do_sample=False,
        # temperature=1.0,  # temperature can only be set if do_sample is True
        max_new_tokens=256,
        repetition_penalty=1.1,
    )
    output = tokenizer.decode(out.squeeze()).strip()
    print(output)
    outputs[strength] = output
    # or if you want to display the special tokens:
    # print(tokenizer.decode(out.squeeze(), skip_special_tokens=False).strip())
    print("###" * 5)

