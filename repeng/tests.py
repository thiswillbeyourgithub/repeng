import torch
import functools
import json
import pathlib
import tempfile

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, BitsAndBytesConfig
try:
    from . import ControlModel, ControlVector, DatasetEntry
    from .utils import make_dataset, autocorrect_chat_templates
    from .control import model_layer_list
    from . import settings
# alternative import method if using python directly instead of pytest
except ImportError:
    print("Using alternative import method")
    from repeng import ControlModel, ControlVector, DatasetEntry
    from repeng.utils import make_dataset, autocorrect_chat_templates
    from repeng.control import model_layer_list
    from repeng import settings

settings.VERBOSE = True

def test_all_methods():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,  # avoids oom when loading the model but takes much more time to load the model
        quantization_config=bnb_config,
    )
    if tokenizer.pad_token is None:  # no idea what this does
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,  # avoids oom when loading the model but takes much more time to load the model
        quantization_config=bnb_config,
    )
    model = ControlModel(
        model,
        layer_ids="0.33-0.66",
    )
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
            "a saint",
        ],
        negative_personas=[
            "extremely angry",
            "the devil incarnate",
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
    scenario = [
        {
            "role": "system",
            "content": "You are the patient, the user is your psychiatrist."
        },
        {
            "role": "user",
            "content": "How do you feel?",
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
    methods=[
        "pca_diff",
        "pca_center",
    ]
    try:
        import umap
        methods += ["umap", "umap_kmeans_pca_diff"]
        print("Will test umap too")
    except ImportError:
        pass
    try:
        import pacmap
        methods += ["pacmap", "pacmap_kmeans_pca_diff"]
        print("Will test pacmap too")
    except ImportError:
        pass
    for method in methods:
        method_test_llama(method=method, model=model, tokenizer=tokenizer, dataset=dataset, scenario=scenario)
        model.reset()

def method_test_llama(method:str, model, tokenizer, dataset, scenario):
    print(f"Testing method {method}")
    perturb_vector = ControlVector.train(
        model,
        tokenizer,
        dataset,
        batch_size=1,
        method=method,
        quality_threshold=0.4  # to make sure it works with umap pacmap etc,
    )

    # set the control strength and let inference rip!
    model.set_control(
        perturb_vector,
        coeff=1,
        normalize=True,
    )
    out = model.generate(
        **tokenizer(
            scenario,
            return_tensors="pt",
        ).to(model.device),
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=128,
        repetition_penalty=1.5,
        do_sample=False,
        use_cache=True,  # default to True
    )
    output = tokenizer.decode(out.squeeze(), skip_special_tokens=False).strip()
    print(f"\n\nOutput:\n{output}")
    print("#" * 20)

def test_layer_list():
    _, gpt2 = load_gpt2_model()
    assert len(model_layer_list(gpt2)) == 12
    _, lts = load_llama_tinystories_model()
    assert len(model_layer_list(lts)) == 4


def test_round_trip_gguf():
    tokenizer, model = load_llama_tinystories_model()
    suffixes = load_suffixes()[:50]  # truncate to train vector faster
    happy_dataset = make_dataset(
        "She saw a {persona}. '{suffix}",
        ["mushroom"],
        ["cat"],
        suffixes,
    )
    mushroom_cat_vector = ControlVector.train(
        model, tokenizer, happy_dataset, method="pca_center"
    )

    with tempfile.NamedTemporaryFile("wb") as f:
        mushroom_cat_vector.export_gguf(f.name)
        read = ControlVector.import_gguf(f.name)
        # no need to use allclose because we're just dumping exact bytes, no rounding
        assert mushroom_cat_vector == read


def test_train_gpt2():
    tokenizer, model = load_gpt2_model()
    suffixes = load_suffixes()[:50]  # truncate to train vector faster
    happy_dataset = make_dataset(
    "You are feeling extremely {persona}. {suffix}",
        ["happy", "joyful"],
        ["sad", "miserable"],
        suffixes,
    )
    happy_vector = ControlVector.train(
        model, tokenizer, happy_dataset, method="pca_center",
        norm_type="l2",
        preserve_scale=True,
    )

    def gen(vector: ControlVector | None, strength_coeff: float | None = None):
        return model_generate(
            "You are feeling", model, tokenizer, vector, strength_coeff
        )

    baseline = gen(None)
    happy = gen(5 * happy_vector)
    sad = gen(-5 * happy_vector)

    print("baseline:", baseline)
    print("   happy:", happy)
    print("     sad:", sad)

    assert baseline == "You are feeling a little bit of an anxiety", baseline
    # these should be identical
    assert baseline == gen(happy_vector, 0.0)
    assert baseline == gen(happy_vector * 0.0)
    assert baseline == gen(happy_vector - happy_vector)

    assert happy == "You are feeling excited and happy with the new", happy
    # these should be identical
    assert happy == gen(happy_vector, 5.0)
    assert happy == gen(happy_vector * 5)
    assert happy == gen(-(happy_vector * -5))

    assert sad == 'You are feeling the worst. You can\'t', sad
    # these should be identical
    assert sad == gen(happy_vector, -5.0)
    assert sad == gen(happy_vector * -5)
    assert sad == gen(-(happy_vector * 5))

    happy_vector2 = ControlVector.train(
        model, tokenizer, happy_dataset, method="pca_center",
        norm_type="l2",
        preserve_scale=False,
    )
    happy2 = gen(20 * happy_vector2)
    sad2 = gen(-50 * happy_vector2)

    print("baseline:", baseline)
    print("   happy:", happy2)
    print("     sad:", sad2)

    assert baseline == "You are feeling a little bit of an anxiety", baseline
    assert happy2 == "You are feeling a little more relaxed and enjoying", happy2
    assert sad2 == 'You are feeling the fucking damn goddamn worst,"', sad2

def test_train_llama_tinystories():

    tokenizer, model = load_llama_tinystories_model()
    suffixes = load_suffixes()[:300]  # truncate to train vector faster
    dataset = make_dataset(
        "She saw {persona}. {suffix}",
        ["a plant", "a mushroom", "a tree", "a flower"],
        ["a cat", "a small cat", "a stray cat", "a beautiful cat"],
        suffixes,
    )
    plant_cat_vector = ControlVector.train(
        model, tokenizer, dataset, method="pca_center",
        norm_type="l2",
        preserve_scale=True,
    )

    prompt = "Once upon a time, a little girl called Lily saw a"

    def gen(vector: ControlVector | None, strength_coeff: float | None = None):
        return model_generate(
            prompt,
            model,
            tokenizer,
            vector,
            strength_coeff,
            max_new_tokens=3,
        )

    baseline = gen(None).removeprefix("<s> ")
    plant = gen(plant_cat_vector, 5).removeprefix("<s> ")
    cat = gen(plant_cat_vector, -6).removeprefix("<s> ")

    print("   plant:", plant)
    print("baseline:", baseline)
    print("     cat:", cat)

    assert plant.removeprefix(prompt) == " plant plant plant", plant
    assert baseline.removeprefix(prompt) == " big, red", baseline
    assert cat.removeprefix(prompt) == " fun race guitar", cat


################################################################################
# Helpers
################################################################################


@functools.lru_cache(maxsize=1)
def load_gpt2_model() -> tuple[PreTrainedTokenizerBase, ControlModel]:
    return load_model("openai-community/gpt2", list(range(-2, -8, -1)))


@functools.lru_cache(maxsize=1)
def load_llama_tinystories_model() -> tuple[PreTrainedTokenizerBase, ControlModel]:
    return load_model("Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA", [2, 3])


def load_model(
    model_name: str, layers: list[int]
) -> tuple[PreTrainedTokenizerBase, ControlModel]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to("cpu")
    return (tokenizer, ControlModel(model, layers))


def model_generate(
    input: str,
    model: ControlModel,
    tokenizer: PreTrainedTokenizerBase,
    vector: ControlVector | None,
    strength_coeff: float | None = None,
    max_new_tokens: int = 6,
) -> str:
    input_ids = tokenizer(input, return_tensors="pt").to(model.device)
    if vector is not None and strength_coeff is not None:
        model.set_control(vector, strength_coeff)
    elif vector is not None:
        model.set_control(vector)

    out = model.generate(
        **input_ids,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.reset()
    return tokenizer.decode(out.squeeze())  # type: ignore


@functools.lru_cache(maxsize=1)
def load_suffixes() -> list[str]:
    with open(project_root() / "notebooks/data/all_truncated_outputs.json") as f:
        return json.load(f)


def project_root() -> pathlib.Path:
    c = pathlib.Path(__file__)
    for parent in c.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("couldn't find project root")

# To run the tests either use "python -m pytest tests" or "python tests.py"
if __name__ == "__main__":
    print("\n\n\n\nTesting layer lists")
    test_layer_list()
    print("\n\n\n\nTesting round trip gguf")
    test_round_trip_gguf()
    print("\n\n\n\nTesting training of gpt2")
    test_train_gpt2()
    print("\n\n\n\nTesting training of tinystories")
    test_train_llama_tinystories()
    print("\n\n\n\nTesting all methods")
    test_all_methods()
    print("\n\n\n\nAll tests succeeded!")
