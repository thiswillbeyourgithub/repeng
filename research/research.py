import os
from tqdm import tqdm
import json
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import AutoModel, AutoTokenizer
import time

from repeng import ControlVector, ControlModel, DatasetEntry

start_time = time.time()
def printer(message: str):
    global start_time
    endtime = time.time()
    elapsed = endtime - start_time
    elapsedmin = int(elapsed // 60)
    print(f"T+{elapsedmin}M: {message}")
    start_time = time.time()

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


# load and wrap Mistral-7B
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# model_name = "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF"
# fname = "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
# fname = "Mistral-7B-Instruct-v0.3.Q2_K.gguf"
# model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
# fname = "mistral-7b-instruct-v0.1.Q2_K.gguf"

printer("Initializing model...")
token=os.environ["HUGGINGFACE_API_TOKEN"]
assert token
login(token=token)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=token,
    # gguf_file=fname,
    # torch_dtype="int8",
    load_in_8bit=True,  # must be disabled if loading a gguf
    # device_map="cuda",
    device_map="auto",  # may oom on low vram, otherwise use all available gous I think
    low_cpu_mem_usage=True,
)

printer("Creating control model...")
model = ControlModel(model, list(range(-5, -18, -1)))

# generate a dataset with closely-opposite paired statements
printer("Making dataset")
truncated_output_suffixes = [""]
trippy_dataset = make_dataset(
    "Act as if you're extremely {persona}.",
    ["high on psychedelic drugs"],
    ["sober from psychedelic drugs"],
    truncated_output_suffixes,
)

# train the vector—takes less than a minute!
printer("Training control vector")
trippy_vector = ControlVector.train(model, tokenizer, trippy_dataset)

# set the control strength and let inference rip!
printer("Applying strength vectors")
for strength in (-2.2, 1, 2.2):
    printer(f"strength={strength}")
    model.set_control(trippy_vector, strength)
    out = model.generate(
        **tokenizer(
            f"[INST] Give me a one-sentence pitch for a TV show. [/INST]",
            return_tensors="pt"
        ),
        do_sample=False,
        max_new_tokens=128,
        repetition_penalty=1.1,
    )
    printer(tokenizer.decode(out.squeeze()).strip())
    print()

printer("Now proceeding to test the model")
from langtest import Harness
# Create test Harness
harness = Harness(task="text-classification",
                  model={'model': model, "hub": "custom"}, 
                  data={'data_source': 'test.csv'},
                  config='config.yml')	

# Generate, run and get a report on your test cases
h.generate().run().report()
