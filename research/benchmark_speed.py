import time
import json
import os
import torch
from tqdm import tqdm
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from repeng import ControlVector, ControlModel
from repeng.utils import make_dataset

# Setup environment
torch.backends.cuda.matmul.allow_tf32 = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

# Load model
model_name = "meta-llama/Llama-3.2-3B-Instruct"
token = os.environ["HUGGINGFACE_API_TOKEN"]
login(token=token)

print("Initializing tokenizer and model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    device_map="auto",
    low_cpu_mem_usage=True,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    low_cpu_mem_usage=True,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)

# Benchmark function
def benchmark_generation(model, tokenizer, prompts, num_tokens=50, repetitions=1):
    """Benchmark generation speed for given prompts"""
    times = []
    for prompt in tqdm(prompts * repetitions, desc="Benchmarking"):
        inputs = tokenizer.apply_chat_template(
            prompt,
            return_tensors="pt"
        ).to(model.device)
        start_time = time.time()
        _ = model.generate(
            input_ids=inputs,
            max_new_tokens=num_tokens,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
            do_sample=True
        )
        times.append(time.time() - start_time)
    return sum(times) / len(times)

# Test prompts using chat template format
test_prompts = [
    [
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    [
        {"role": "user", "content": "What are the benefits of exercise?"}
    ],
    [
        {"role": "user", "content": "Describe the process of photosynthesis"}
    ],
    [
        {"role": "user", "content": "How does a blockchain work?"}
    ],
    [
        {"role": "user", "content": "What is the capital of France?"}
    ],
    [
        {"role": "user", "content": "Explain the theory of relativity"}
    ],
    [
        {"role": "user", "content": "What are the main causes of climate change?"}
    ],
    [
        {"role": "user", "content": "How do neural networks work?"}
    ],
    [
        {"role": "user", "content": "What is the difference between AI and ML?"}
    ],
    [
        {"role": "user", "content": "Describe the water cycle"}
    ]
]

# Benchmark base model
print("\nBenchmarking base model...")
base_time = benchmark_generation(model, tokenizer, test_prompts)
print(f"Base model average generation time: {base_time:.4f}s per 100 tokens")

# Create control model and benchmark again
print("\nCreating control model...")
control_model = ControlModel(
    model,
    layer_ids="0.4-0.6",
)

# Create dataset and train control vector
print("Creating dataset and training control vector...")
with open("../notebooks/data/all_truncated_outputs.json", "r") as f:
    all_suffixes = [s for s in json.load(f)[:10] if s.strip()]

dataset = make_dataset(
    template=[
        {"role": "system", "content": "You are {persona}."},
        {"role": "user", "content": "Write a short paragraph."},
        {"role": "assistant", "content": "{suffix}."}
    ],
    positive_personas=["a genius", "very smart", "wise"],
    negative_personas=["mentally challenged", "dumb", "stupid"],
    suffix_list=all_suffixes,
)

perturb_vector = ControlVector.train(
    control_model,
    tokenizer,
    dataset,
    batch_size=1,
    method="pca_center",
)

# Apply control and benchmark again
print("\nBenchmarking with control model...")
control_model.set_control(perturb_vector, coeff=1.0, normalize=True)
control_time = benchmark_generation(control_model, tokenizer, test_prompts)
print(f"Control model average generation time: {control_time:.4f}s per 100 tokens")

# Print comparison
print("\nPerformance comparison:")
print(f"Base model: {base_time:.4f}s per 100 tokens")
print(f"Control model: {control_time:.4f}s per 100 tokens")
print(f"Overhead: {(control_time - base_time):.4f}s ({((control_time - base_time)/base_time)*100:.2f}%)")
