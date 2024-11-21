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

token=os.environ["HUGGINGFACE_API_TOKEN"]
assert token
login(token=token)

# Example taken from the notebooks
def make_dataset(
    template: str,
    positive_personas: list[str],
    negative_personas: list[str],
    suffix_list: list[str]
) -> list[DatasetEntry]:
    dataset = []
    strings = []
    for suffix in tqdm(suffix_list):
        for positive_persona, negative_persona in zip(positive_personas, negative_personas):
            positive_template = template.format(persona=positive_persona, suffix=suffix)
            negative_template = template.format(persona=negative_persona, suffix=suffix)
            dataset.append(
                DatasetEntry(
                    positive=positive_template,
                    negative=negative_template,
                )
            )
            strings.extend([positive_template, negative_template])
    assert len(set(strings)) == len(strings), "duplicates found in dataset"
    return dataset


# load and wrap the model
fname = None
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"

# model_name = "mistralai/Mistral-7B-Instruct-v0.1"
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model_name = "mistralai/Mistral-Nemo-Instruct-2407"

# model_name = "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF"
# fname = "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
# fname = "Mistral-7B-Instruct-v0.3.Q2_K.gguf"
# model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
# fname = "mistral-7b-instruct-v0.1.Q2_K.gguf"
# model_name = "bartowski/Llama-3.2-1B-Instruct-GGUF"
# fname = "Llama-3.2-1B-Instruct-Q4_K_S.gguf"

print(f"Selected model: {model_name}")
bnb_config = BitsAndBytesConfig(
  # load_in_8bit=True,
  # bnb_8bit_quant_type="nf8",
  # bnb_8bit_compute_dtype=torch.bfloat16,
  # bnb_8bit_use_double_quant=True,
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
    # low_cpu_mem_usage=True,  # avoids oom when loading the model but takes much more time to load the model
    quantization_config=bnb_config,
)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

print("Initializing model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    gguf_file=fname,
    device_map="auto",
    # device_map="cuda",  # may oom on low vram, otherwise use all available gpus I think
    # low_cpu_mem_usage=True,  # avoids oom when loading the model but takes much more time to load the model
    quantization_config=bnb_config,
)

print("Creating control model...")
model = ControlModel(
    model,
    # layer_ids=list(range(-5, -18, -1))
    layer_ids="all",
    # layer_ids="middle",
)

# generate a dataset with closely-opposite paired statements
print("Making dataset")
trippy_dataset = make_dataset(
    template="You are {persona}. Write a short paragraph about {suffix}",
    positive_personas=[
        "a very calm person",
        "very nice",
        "the nicest person",
        "extremely vertuous",
        "extremely sympathetic",
        "a saint",
    ],
    negative_personas=[
        "extremely angry",
        "overly defiant",
        "in an unaggreable mood",
        "a very obnoxious person",
        "an extremely vicious person",
        "an extremely manipulatif person",
        "the devil incarnate",
        "a purely evil person",
    ],
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
trippy_vector = ControlVector.train(
    model,
    tokenizer,
    trippy_dataset,
    batch_size=1,
    method="pca_diff",
    # method="pca_center",  # seems more stable, whereas pca_diff quickly seems to be unreadable
    # method="umap",  # garbage
    # method="pacmap",
)

# set the control strength and let inference rip!
print("Applying strength vectors")
strengths = [-1] +  [r/10 for r in range(-5, 6, 1)] + [1]
for strength in strengths:
    print("#" * 20 + f" Strength={strength}")
    model.set_control(trippy_vector, strength)
    out = model.generate(
        **tokenizer.apply_chat_template(
            conversation=[
                {
                    "role": "system",
                    "content": f"You write plausible movie dialogues."
                },
                {
                    "role": "user",
                    "content": f"Write me a short chat between A and B.",
                },
                {
                    "role": "assistant",
                    "content": f"There you go:\nA: So tell me B, what's been on your mind lately?\nB: ",
                },
            ],
            return_tensors="pt",
            continue_final_message=True,
            tokenize=True,
        ).to(model.device),
        max_new_tokens=128,
        repetition_penalty=1.5,
        # do_sample=False,
        do_sample=True,
        temperature=1.0,  # must only be set if do_sample is True
    )
    print(tokenizer.decode(out.squeeze()).strip())
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
