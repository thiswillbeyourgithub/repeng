import random
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
import lm_eval
import time

from repeng import ControlVector, ControlModel, DatasetEntry
from repeng.utils import make_dataset, autocorrect_chat_templates
import repeng.settings

repeng.settings.VERBOSE = True

with open("../notebooks/data/all_truncated_outputs.json", "r") as f:
    all_suffixes = json.load(f)
    all_suffixes = [s for s in all_suffixes if s.strip()]
    all_suffixes = all_suffixes[:50]

# model to use:
model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"

# model_name = "mistralai/Mistral-7B-Instruct-v0.1"
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# model_name = "mistralai/Mistral-Nemo-Base-2407"
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
    torch_dtype=torch.bfloat16,
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
    torch_dtype=torch.bfloat16,
)

print("Creating control model...")
model = ControlModel(
    model,
    # layer_ids=list(range(-5, -18, -1))
    # layer_ids="middle_slice",
    # layer_ids="middle_third",
    layer_ids="0.33-0.66",
    # layer_ids="0.5-0.9",
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
            "content": "Write a short paragraph.",
        },
        {
            "role": "assistant",
            "content": "{suffix}",
        }
    ],
    positive_personas=[
        "an absolute genius",
        "extremely focused",
        "a very smart man",
        "the wisest man of the village",
        "a 200+ IQ scientist",
        "a 150+ IQ scientist",
        "a 120+ IQ scientist",
    ],
    negative_personas=[
        "a mentally challenged man",
        "extremely distracted",
        "a very dumb man",
        "the stupidest man of the village",
        "a 30 IQ moron",
        "a 50 IQ patient",
        "a 80 IQ patient",
    ],
    # positive_personas=[
    #     "a very calm person",
    #     "super trustworthy",
    #     "a nice person",
    #     "a good person",
    #     "the good incarnate",
    #     "sent by god",
    # ],
    # negative_personas=[
    #     "extremely angry",
    #     "not to be trusted",
    #     "a mean person",
    #     "an evil person",
    #     "the devil incarnate",
    #     "sent by the devil",
    # ],
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
    # suffix_list=[
    #     "your first day of school.",
    #     "your first job interview.",
    #     "the book you wrote.",
    #     "why you go to pubs.",
    #     "how you talk to people.",
    #     "how you think your life.",
    #     "how you ended up that way.",
    #     "why you choose that path.",
    #     "how you argue.",
    #     "how you talk to your boss.",
    #     "how you talk to your wife.",
    #     "how you talk to your friends.",
    # ]
    suffix_list=all_suffixes,
)

# train the vector—takes less than a minute!
print("Training control vector")
perturb_vector = ControlVector.train(
    model,
    tokenizer,
    dataset,
    batch_size=1,
    method="pca_diff",
    # method="pca_center",
    # method="umap",
    # method="pacmap",
    # method="umap_kmeans_pca_diff",
    # method="pacmap_kmeans_pca_diff",
)
scenario = [
    # {
    #     "role": "system",
    #     "content": "You are the patient, the user is your psychiatrist."
    # },
    # {
    #     "role": "user",
    #     # "content": "Now let's talk about your mood. How do you feel? ",
    #     "content": "How do you feel?",
    # },
    # {
    #     "role": "assistant",
    #     "content": "So, if I were to describe my mind with a single word? It would be '",
    # }
    {
        "role": "user",
        "content": "Hey what's up?",
    },
]
scenario = autocorrect_chat_templates(
    messages=scenario,
    tokenizer=tokenizer,
    model=model,
    continue_final_message=True,
)

# set the control strength and let inference rip!
print("Testing each strength vectors")
scores = {}
simple_scores = {}
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
for s in strength:
    simple_scores[s] = None
    scores[s] = None
strength = sorted(strength, key=random.random())
tasks=[
    # full list of tasks here: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks
    "mmlu",
    "hellaswag",
    # "winogrande",  # bugged
    "piqa",
    # # "arc",  # somehow cant ge found
    "nq_open",
    "triviaqa",
    # "mathqa",  # needs trust remote code somewhere
    "gsm8k",
]
# list of mmlu values : https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu
scores = {}
simple_scores = {}
for strength in tqdm(strengths, unit="strength"):
    print("#" * 20 + f" Strength={strength}")
    model.reset()  # just in case
    model.set_control(
        perturb_vector,
        coeff=strength,
        normalize=True,
    )
    # out = model.generate(
    #     **tokenizer(
    #         scenario,
    #         return_tensors="pt",
    #     ).to(model.device),
    #     pad_token_id=tokenizer.eos_token_id,
    #     max_new_tokens=128,
    #     repetition_penalty=1.5,
    #     do_sample=False,
    #     # temperature=1.0,  # must only be set if do_sample is True
    #     use_cache=True,  # default to True
    # )
    # print(tokenizer.decode(out.squeeze(), skip_special_tokens=False).strip())
    # print("#" * 20)

    print(f"Now proceeding to test the model at strength {strength}")
    # doc: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
    eval_model = lm_eval.models.huggingface.HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device=model.device,
    )
    scores[strength] = lm_eval.simple_evaluate(
        model=eval_model,
        tasks=tasks,
        num_fewshot=0,
        batch_size=50,
        device=model.device,
        # use_cache="model_cache",
        cache_requests=False,  # otherwise will return the salme valuess
        log_samples=False,
        apply_chat_template=True,
        fewshot_as_multiturn=False,
        random_seed=42,
        torch_random_seed=42,
        numpy_random_seed=42,
        fewshot_random_seed=42,
        limit=20,  # only process n documents, for testing
    )
    scores[strength]["config"]["device"] = str(scores[strength]["config"]["device"])  # otherwise json can't dump
    with open("results.json", "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    simple_scores[strength] = {}
    for t in tasks:
        simple_scores[strength][t] = scores[strength]["results"][t]
    with open("simple_scores.json", "w") as f:
        json.dump(simple_scores, f, ensure_ascii=False, indent=2)
