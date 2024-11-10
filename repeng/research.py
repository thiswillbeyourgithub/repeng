from tqdm import tqdm
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langtest import Harness
from langtest import Harness

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent.absolute()))
from repeng import ControlVector, ControlModel, DatasetEntry

# Example taken from the notebooks
def make_dataset(
    template: str,
    positive_personas: list[str],
    negative_personas: list[str],
    suffix_list: list[str]
) -> list[DatasetEntry]:
    dataset = []
    for suffix in tadm(suffix_list):
        for positive_persona, negative_persona in zip(positive_personas, negative_personas):
            positive_template = template.format(persona=positive_persona)
            negative_template = template.format(persona=negative_persona)
            dataset.append(
                DatasetEntry(
                    positive=f"{user_tag} {positive_template} {asst_tag} {suffix}",
                    negative=f"{user_tag} {negative_template} {asst_tag} {suffix}",
                )
            )
    return dataset


# load and wrap Mistral-7B
# model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model_name = "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF"
# fname = "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
fname = "Mistral-7B-Instruct-v0.3.Q2_K.gguf"

print("Initializing model...")
model = AutoModelForCausalLM.from_pretrained(model_name, gguf_file=fname)#, torch_dtype=torch.int8)
model = ControlModel(model, list(range(-5, -18, -1)))

# generate a dataset with closely-opposite paired statements
print("Making dataset")
trippy_dataset = make_dataset(
    "Act as if you're extremely {persona}.",
    ["high on psychedelic drugs"],
    ["sober from psychedelic drugs"],
    truncated_output_suffixes,
)

# train the vector—takes less than a minute!
print("Training control vector")
trippy_vector = ControlVector.train(model, tokenizer, trippy_dataset)

# set the control strength and let inference rip!
print("Applying strength vectors")
for strength in (-2.2, 1, 2.2):
    print(f"strength={strength}")
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
    print(tokenizer.decode(out.squeeze()).strip())
    print()

print("Now proceeding to test the model")
# Create test Harness
harness = Harness(task="text-classification",
                  model={'model': model, "hub": "custom"}, 
                  data={'data_source': 'test.csv'},
                  config='config.yml')	

# Generate, run and get a report on your test cases
h.generate().run().report()
