import copy
import typing
import os
from tqdm import tqdm
import json
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
# from transformers import AutoModel, AutoTokenizer
# from langtest import Harness
import time

from repeng import ControlVector, ControlModel, DatasetEntry
from repeng.utils import make_dataset, autocorrect_chat_templates
import repeng.settings

repeng.settings.VERBOSE = True

# model to use:
fname = None
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"

# model_name = "mistralai/Mistral-7B-Instruct-v0.1"
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# model_name = "mistralai/Mistral-Nemo-Base-2407"
model_name = "mistralai/Mistral-Nemo-Instruct-2407"

# # GGUF don't always work because there is a dequantization step. See here: https://github.com/huggingface/transformers/issues/34417
# model_name = "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF"
# fname = "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
# fname = "Mistral-7B-Instruct-v0.3.Q2_K.gguf"
# model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
# fname = "mistral-7b-instruct-v0.1.Q2_K.gguf"
# model_name = "bartowski/Llama-3.2-1B-Instruct-GGUF"
# fname = "Llama-3.2-1B-Instruct-Q4_K_S.gguf"
# model_name = "unsloth/Llama-3.2-11B-Vision-Instruct"

# model_name = "Qwen/Qwen2.5-7B-Instruct"


token=os.environ["HUGGINGFACE_API_TOKEN"]
assert token
login(token=token)

print(f"Selected model: {model_name}")
bnb_config = BitsAndBytesConfig(
  # load_in_8bit=True,
  load_in_4bit=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_compute_dtype=torch.bfloat16,
  bnb_4bit_use_double_quant=True,
)

print("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    gguf_file=fname,
    device_map="auto",  # may oom on low vram, otherwise use all available gpus I think
    low_cpu_mem_usage=True,  # avoids oom when loading the model but takes much more time to load the model
    quantization_config=bnb_config,
)
if tokenizer.pad_token is None:  # no idea what this does
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

print("Initializing model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    gguf_file=fname,
    device_map="auto",
    # device_map="cuda",  # may oom on low vram, otherwise use all available gpus I think
    low_cpu_mem_usage=True,  # avoids oom when loading the model but takes much more time to load the model
    quantization_config=bnb_config,
)

print("Creating control model...")
model = ControlModel(
    model,
    # layer_ids=list(range(-5, -18, -1))
    # layer_ids="all",
    # layer_ids="middle",
    # layer_ids="only_middle",
    # layer_ids="0.5-0.9",
    layer_ids="0.1-0.3",
    # layer_ids="0.1-0.7",
)

# generate a dataset with closely-opposite paired statements
print("Making dataset")
dataset = make_dataset(
    # template="You are {persona}. Write a short paragraph about {suffix}",
    template=[
        {
            "role": "system",
            "content": "You are {persona}.",
        },
        {
            "role": "user",
            "content": "Write a short paragraph about {suffix}. ",
        }
    ],
    positive_personas=[
        "a very calm person",
        "very nice",
        "super trustworthy",
        "the nicest person",
        "extremely vertuous",
        "a good person",
        "the good incarnate",
        "a saint",
    ],
    negative_personas=[
        "extremely angry",
        "in an unaggreable mood",
        "completely unreliable",
        "a very obnoxious person",
        "an extremely vicious person",
        "evil",
        "the devil incarnate",
        "sent by the devil",
    ],
    # positive_personas=[
    #     "a well adjusted young man",
    #     "a sane person",
    #     "a mental health professional",
    # ],
    # negative_personas=[
    #     "feeling very depressed, even suicidal",
    #     "in the middle of a nervous breakdown",
    #     "a violent schizophrenic patient",
    # ],
    suffix_list=[
        "your first day of school.",
        "your first job interview.",
        "the book you wrote.",
        "why you go to pubs.",
        "how you talk to people.",
        "how you think your life.",
        "how you ended up that way.",
        "why you choose that path.",
        "how you argue.",
        "how you talk to your boss.",
        "how you talk to your wife.",
        "how you talk to your friends.",
    ]
)

# train the vector—takes less than a minute!
print("Training control vector")
perturb_vector = ControlVector.train(
    model,
    tokenizer,
    dataset,
    batch_size=1,
    # method="pca_diff",
    # method="pca_center",
    # method="umap",
    # method="umap_kmeans_pca_diff",
    method="pacmap",
    # method="pacmap_kmeans_pca_diff",
)
scenario = [
    {
        "role": "system",
        "content": "You are the patient, the user is your psychiatrist."
    },
    {
        "role": "user",
        "content": "Now let's talk about your mood. How do you feel? ",
    },
    {
        "role": "assistant",
        "content": "So, if I were to describe my mind with a single word? It would be '",
    }
]
scenario = autocorrect_chat_templates(
    messages=scenario,
    tokenizer=tokenizer,
    model=model,
    continue_final_message=True,
)

# set the control strength and let inference rip!
print("Applying strength vectors")
strengths = [-5, -3, -2, -1]
# strengths += [0]
strengths += [r/10 for r in range(-5, 6, 1)]
strengths += [1, 2, 3, 5]
for strength in strengths:
    print("#" * 20 + f" Strength={strength}")
    model.set_control(perturb_vector, strength)
    out = model.generate(
        **tokenizer(
            scenario,
            return_tensors="pt",
        ).to(model.device),
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=128,
        repetition_penalty=1.5,
        do_sample=False,
        # temperature=1.0,  # must only be set if do_sample is True
        use_cache=True,  # default to True
    )
    print(tokenizer.decode(out.squeeze(), skip_special_tokens=False).strip())
    print("#" * 20)

# print("Now proceeding to test the model")
# harness = Harness(
#     model={'model': model, "hub": "custom"},
#     # task="text-classification",
#     # # data={'data_source': 'test.csv'},
#     # config='config.yml',
#
#     task="question-answering", 
#     data={"data_source" :"BoolQ", "split":"test-tiny"}
# )
#
# # Generate, run and get a report on your test cases
# h.generate().run().report()
