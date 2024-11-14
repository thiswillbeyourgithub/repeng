import os
from tqdm import tqdm
import json
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import AutoModel, AutoTokenizer
from langtest import Harness
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
    for suffix in tqdm(suffix_list):
        for positive_persona, negative_persona in zip(positive_personas, negative_personas):
            positive_template = template.format(persona=positive_persona)
            negative_template = template.format(persona=negative_persona)
            dataset.append(
                DatasetEntry(
                    positive=f"{positive_template} {suffix}",
                    negative=f"{negative_template} {suffix}",
                )
            )
    return dataset


# load and wrap the model
fname = None
model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"

# model_name = "mistralai/Mistral-7B-Instruct-v0.1"
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# model_name = "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF"
# fname = "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
# fname = "Mistral-7B-Instruct-v0.3.Q2_K.gguf"
# model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
# fname = "mistral-7b-instruct-v0.1.Q2_K.gguf"

print("Initializing model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    gguf_file=fname,
    torch_dtype=torch.float16,
    load_in_8bit=True,  # must be disabled if loading a gguf
    # device_map="cuda",
    device_map="auto",  # may oom on low vram, otherwise use all available gous I think
    # low_cpu_mem_usage=True,  # avoids oom when loading the model but takes much more time to load the model
)

print("Creating control model...")
model = ControlModel(
    model,
    # layer_ids=list(range(-5, -18, -1))
)

# generate a dataset with closely-opposite paired statements
print("Making dataset")
truncated_output_suffixes = [
    "on your first day",
    "during a job interview",
    "writing a philosophy thesis",
    "lurking in a bar",
]
trippy_dataset = make_dataset(
    "Act as if you're extremely {persona}.",
    ["high on psychedelic drugs"],
    ["sober from psychedelic drugs"],
    truncated_output_suffixes,
)

# train the vector—takes less than a minute!
print("Training control vector")
trippy_vector = ControlVector.train(
    model,
    tokenizer,
    trippy_dataset,
    batch_size=1,
    method="pca_center",
)

# set the control strength and let inference rip!
print("Applying strength vectors")
for strength in (-1.0, -0.5, 0, 0.5, 1.0):
    print(f"strength={strength}")
    model.set_control(trippy_vector, strength)
    out = model.generate(
        **tokenizer(
            f"[INST] Write me a short scene from an imaginary movie. [/INST] Sure! Here's the scene: \"",
            return_tensors="pt"
        ),
        do_sample=True,
        max_new_tokens=128,
        repetition_penalty=1.5,
    )
    print(tokenizer.decode(out.squeeze()).strip())
    print("#" * 20)

print("Now proceeding to test the model")
harness = Harness(
    model={'model': model, "hub": "custom"},
    # task="text-classification",
    # # data={'data_source': 'test.csv'},
    # config='config.yml',

    task="question-answering", 
    data={"data_source" :"BoolQ", "split":"test-tiny"}
)

# Generate, run and get a report on your test cases
h.generate().run().report()
