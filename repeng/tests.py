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
    print("\n\n\n\nAll tests succeeded!")
