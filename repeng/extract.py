import dataclasses
import hashlib
import os
import typing
import warnings
from datetime import datetime

import gguf
import h5py
import numpy as np
from sklearn.decomposition import PCA
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import tqdm

from .control import ControlModel, model_layer_list
from .saes import Sae
from .utils import DatasetEntry, autocorrect_chat_templates, get_model_name

__VERSION__ = "0.4.0"


@dataclasses.dataclass
class ControlVector:
    model_type: str
    directions: dict[int, np.ndarray]

    @classmethod
    def train(
        cls,
        model: "PreTrainedModel | ControlModel",
        tokenizer: PreTrainedTokenizerBase,
        dataset: list[DatasetEntry],
        cache_path: os.PathLike[str] | str | None = None,
        **kwargs,
    ) -> "ControlVector":
        """
        Train a ControlVector for a given model and tokenizer using the provided dataset.

        Args:
            model (PreTrainedModel | ControlModel): The model to train against.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to tokenize the dataset.
            dataset (list[DatasetEntry]): The dataset used for training.
            cache_path (os.PathLike[str] | str | None, optional): Path to directory for h5py caching.
                If None, activations are computed and stored in memory. If provided, activations
                are cached to disk to allow for better memory scaling. Defaults to None.
            **kwargs: Additional keyword arguments. See help(repeng.extract.read_representations) for details.

        Returns:
            ControlVector: The trained vector.
        """
        with torch.inference_mode():
            dirs = read_representations(
                model,
                tokenizer,
                dataset,
                cache_path=cache_path,
                **kwargs,
            )
        return cls(model_type=model.config.model_type, directions=dirs)

    @classmethod
    def train_with_sae(
        cls,
        model: "PreTrainedModel | ControlModel",
        tokenizer: PreTrainedTokenizerBase,
        sae: Sae,
        dataset: list[DatasetEntry],
        *,
        decode: bool = True,
        method: typing.Literal["pca_diff", "pca_center", "umap"] = "pca_center",
        cache_path: os.PathLike[str] | str | None = None,
        **kwargs,
    ) -> "ControlVector":
        """
        Like ControlVector.train, but using an SAE. It's better! WIP.


        Args:
            model (PreTrainedModel | ControlModel): The model to train against.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to tokenize the dataset.
            sae (saes.Sae): See the `saes` module for how to load this.
            dataset (list[DatasetEntry]): The dataset used for training.
            **kwargs: Additional keyword arguments.
                decode (bool, optional): Whether to decode the vector to make it immediately usable.
                    If not, keeps it as monosemantic SAE features for introspection, but you will need to decode it manually
                    to use it. Defaults to True.
                cache_path (os.PathLike[str] | str | None, optional): Path to directory for h5py caching.
                    If None, activations are computed and stored in memory. If provided, activations
                    are cached to disk to allow for better memory scaling. Defaults to None.
                max_batch_size (int, optional): The maximum batch size for training.
                    Defaults to 32. Try reducing this if you're running out of memory.
                method (str, optional): The training method to use. Can be either
                    "pca_diff" or "pca_center". Defaults to "pca_center"! This is different
                    than ControlVector.train, which defaults to "pca_diff".

        Returns:
            ControlVector: The trained vector.
        """

        def transform_hiddens(hiddens: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
            sae_hiddens = {}
            for k, v in tqdm.tqdm(hiddens.items(), desc="sae encoding"):
                sae_hiddens[k] = sae.layers[k].encode(v)
            return sae_hiddens

        with torch.inference_mode():
            dirs = read_representations(
                model,
                tokenizer,
                dataset,
                transform_hiddens=transform_hiddens,
                method=method,
                cache_path=cache_path,
                **kwargs,
            )

            final_dirs = {}
            if decode:
                for k, v in tqdm.tqdm(dirs.items(), desc="sae decoding"):
                    final_dirs[k] = sae.layers[k].decode(v)
            else:
                final_dirs = dirs

        return cls(model_type=model.config.model_type, directions=final_dirs)

    def export_gguf(self, path: os.PathLike[str] | str):
        """
        Export a trained ControlVector to a llama.cpp .gguf file.
        Note: This file can't be used with llama.cpp yet. WIP!

        ```python
        vector = ControlVector.train(...)
        vector.export_gguf("path/to/write/vector.gguf")
        ```
        ```
        """

        arch = "controlvector"
        writer = gguf.GGUFWriter(path, arch)
        writer.add_string(f"{arch}.model_hint", self.model_type)
        writer.add_uint32(f"{arch}.layer_count", len(self.directions))
        for layer in self.directions.keys():
            writer.add_tensor(f"direction.{layer}", self.directions[layer])
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    @classmethod
    def import_gguf(cls, path: os.PathLike[str] | str) -> "ControlVector":
        reader = gguf.GGUFReader(path)

        archf = reader.get_field("general.architecture")
        if not archf or not len(archf.parts):
            warnings.warn(".gguf file missing architecture field")
        else:
            arch = str(bytes(archf.parts[-1]), encoding="utf-8", errors="replace")
            if arch != "controlvector":
                warnings.warn(
                    f".gguf file with architecture {arch!r} does not appear to be a control vector!"
                )

        modelf = reader.get_field("controlvector.model_hint")
        if not modelf or not len(modelf.parts):
            raise ValueError(".gguf file missing controlvector.model_hint field")
        model_hint = str(bytes(modelf.parts[-1]), encoding="utf-8")

        directions = {}
        for tensor in reader.tensors:
            if not tensor.name.startswith("direction."):
                continue
            try:
                layer = int(tensor.name.split(".")[1])
            except (IndexError, ValueError):
                raise ValueError(
                    f".gguf file has invalid direction field name: {tensor.name}"
                )
            directions[layer] = tensor.data

        return cls(model_type=model_hint, directions=directions)

    def _helper_combine(
        self, other: "ControlVector", other_coeff: float
    ) -> "ControlVector":
        if self.model_type != other.model_type:
            warnings.warn(
                "Trying to add vectors with mismatched model_types together, this may produce unexpected results."
            )

        model_type = self.model_type
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = self.directions[layer]
        for layer in other.directions:
            other_layer = other_coeff * other.directions[layer]
            if layer in directions:
                directions[layer] = directions[layer] + other_layer
            else:
                directions[layer] = other_layer
        return ControlVector(model_type=model_type, directions=directions)

    def __eq__(self, other: "ControlVector") -> bool:
        if self is other:
            return True

        if self.model_type != other.model_type:
            return False
        if self.directions.keys() != other.directions.keys():
            return False
        for k in self.directions.keys():
            if (self.directions[k] != other.directions[k]).any():
                return False
        return True

    def __add__(self, other: "ControlVector") -> "ControlVector":
        if not isinstance(other, ControlVector):
            raise TypeError(
                f"Unsupported operand type(s) for +: 'ControlVector' and '{type(other).__name__}'"
            )
        return self._helper_combine(other, 1)

    def __sub__(self, other: "ControlVector") -> "ControlVector":
        if not isinstance(other, ControlVector):
            raise TypeError(
                f"Unsupported operand type(s) for -: 'ControlVector' and '{type(other).__name__}'"
            )
        return self._helper_combine(other, -1)

    def __neg__(self) -> "ControlVector":
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = -self.directions[layer]
        return ControlVector(model_type=self.model_type, directions=directions)

    def __mul__(self, other: int | float | np.number) -> "ControlVector":
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = other * self.directions[layer]
        return ControlVector(model_type=self.model_type, directions=directions)

    def __rmul__(self, other: int | float | np.number) -> "ControlVector":
        return self.__mul__(other)

    def __truediv__(self, other: int | float | np.number) -> "ControlVector":
        return self.__mul__(1 / other)


def compute_direction(
    hidden_states: np.ndarray,
    method: typing.Union[
        typing.Literal["pca_diff", "pca_center", "umap"],
        typing.Callable[[np.ndarray], np.ndarray],
    ],
) -> np.ndarray:
    """
    Compute a direction vector from hidden states using the specified method.

    Args:
        hidden_states (np.ndarray): Hidden states array of shape (n_samples, hidden_dim).
            For contrast methods, should have even number of samples where pairs represent
            [positive, negative, positive, negative, ...] examples.
        method: The method to use for computing the direction. Can be "pca_diff",
            "pca_center", "umap", or a callable that takes hidden states and returns
            a direction vector.

    Returns:
        np.ndarray: Direction vector of shape (hidden_dim,).
    """
    if callable(method):
        # Custom method: directly compute direction from hidden states
        return method(hidden_states).astype(np.float32)
    elif method == "pca_diff":
        train = hidden_states[::2] - hidden_states[1::2]
        # shape (1, n_features)
        pca_model = PCA(n_components=1, whiten=False).fit(train)
        # shape (n_features,)
        return pca_model.components_.astype(np.float32).squeeze(axis=0)
    elif method == "pca_center":
        center = (hidden_states[::2] + hidden_states[1::2]) / 2
        train = hidden_states.copy()
        train[::2] -= center
        train[1::2] -= center
        # shape (1, n_features)
        pca_model = PCA(n_components=1, whiten=False).fit(train)
        # shape (n_features,)
        return pca_model.components_.astype(np.float32).squeeze(axis=0)
    elif method == "umap":
        train = hidden_states
        # still experimental so don't want to add this as a real dependency yet
        import umap  # type: ignore

        umap_model = umap.UMAP(n_components=1)
        embedding = umap_model.fit_transform(train).astype(np.float32)
        return np.sum(train * embedding, axis=0) / np.sum(embedding)
    else:
        raise ValueError(f"unknown method {method}")


def read_representations(
    model: "PreTrainedModel | ControlModel",
    tokenizer: PreTrainedTokenizerBase,
    inputs: list[DatasetEntry],
    hidden_layers: typing.Iterable[int] | None = None,
    batch_size: int = 32,
    method: typing.Union[
        typing.Literal["pca_diff", "pca_center", "umap"],
        typing.Callable[[np.ndarray], np.ndarray],
    ] = "pca_diff",
    transform_hiddens: (
        typing.Callable[[dict[int, np.ndarray]], dict[int, np.ndarray]] | None
    ) = None,
    cache_path: os.PathLike[str] | str | None = None,
) -> dict[int, np.ndarray]:
    """
    Extract the representations based on the contrast dataset.
    Called by ControlVector.train

    Args:
        hidden_layers (Iterable[int] | None, optional): Which model layers to extract
            representations from. If None, defaults to all transformer layers in reverse
            order. Negative indices are supported (e.g., -1 for last layer).
            Defaults to None.
        batch_size (int, optional): The maximum batch size for training.
            Defaults to 32. Try reducing this if you're running out of memory.
        method (str | Callable, optional): The training method to use. Can be either
            "pca_diff", "pca_center", "umap", or a callable that takes hidden states
            array of shape (n_samples, hidden_dim) and returns a direction vector
            of shape (hidden_dim,). Defaults to "pca_diff".
        transform_hiddens (Callable[[dict[int, np.ndarray]], dict[int, np.ndarray]] | None, optional):
            Optional function to transform the extracted hidden states before computing
            directions. Takes a dict mapping layer indices to hidden state arrays and
            returns a transformed dict with the same structure. Used for SAE encoding.
            Defaults to None.
        cache_path (os.PathLike[str] | str | None, optional): Path to directory for h5py caching.
            If None, activations are computed and stored in memory. If provided, activations
            are cached to disk to allow for better memory scaling. Defaults to None.
    """
    if not hidden_layers:
        hidden_layers = range(-1, -model.config.num_hidden_layers, -1)

    # normalize the layer indexes if they're negative
    n_layers = len(model_layer_list(model))
    hidden_layers = [i if i >= 0 else n_layers + i for i in hidden_layers]

    # the order is [positive, negative, positive, negative, ...]
    train_strs: list[str] = autocorrect_chat_templates(
        messages=[s for ex in inputs for s in (ex.positive, ex.negative)],
        tokenizer=tokenizer,
        model=model,
    )

    if cache_path is None:
        # Original behavior - store all activation layers in memory
        layer_hiddens = batched_get_hiddens(
            model=model,
            tokenizer=tokenizer,
            inputs=train_strs,
            hidden_layers=hidden_layers,
            batch_size=batch_size,
        )

        if transform_hiddens is not None:
            layer_hiddens = transform_hiddens(layer_hiddens)
    else:
        # Use h5py caching for better memory scaling
        cache_file = batched_get_hiddens_cached(
            model=model,
            tokenizer=tokenizer,
            inputs=train_strs,
            hidden_layers=hidden_layers,
            batch_size=batch_size,
            cache_path=cache_path,
        )
        model_args = _get_model_args_string(model)
        train_strs_hash = _hash_train_strs(train_strs)
        group_path = f"{model_args}/{train_strs_hash}"

    # get directions for each layer using PCA
    directions: dict[int, np.ndarray] = {}
    for layer in tqdm.tqdm(hidden_layers, desc="Altering directions"):
        # Load hidden states either from memory dict or h5py cache
        if cache_path is None:
            # Load from memory
            h = layer_hiddens[layer]
        else:
            # Load from h5py cache only a single layer's activations
            with h5py.File(cache_file, "r") as f:
                h = f[group_path][f"layer_{layer}"][:]
            assert not np.all(h == 0)  # failed to get populated

        assert h.shape[0] == len(inputs) * 2

        directions[layer] = compute_direction(h, method)

        # calculate sign
        projected_hiddens = project_onto_direction(h, directions[layer])

        # order is [positive, negative, positive, negative, ...]
        positive_smaller_mean = np.mean(
            [
                projected_hiddens[i] < projected_hiddens[i + 1]
                for i in range(0, len(inputs) * 2, 2)
            ]
        )
        positive_larger_mean = np.mean(
            [
                projected_hiddens[i] > projected_hiddens[i + 1]
                for i in range(0, len(inputs) * 2, 2)
            ]
        )

        if positive_smaller_mean > positive_larger_mean:  # type: ignore
            directions[layer] *= -1

    return directions


def batched_get_hiddens(
    model,
    tokenizer,
    inputs: list[str],
    hidden_layers: list[int],
    batch_size: int,
) -> dict[int, np.ndarray]:
    """
    Using the given model and tokenizer, pass the inputs through the model and get the hidden
    states for each layer in `hidden_layers` for the last token.

    Returns a dictionary from `hidden_layers` layer id to an numpy array of shape `(n_inputs, hidden_dim)`
    """
    batched_inputs = [
        inputs[p : p + batch_size] for p in range(0, len(inputs), batch_size)
    ]
    hidden_states = {layer: [] for layer in hidden_layers}
    with torch.no_grad():
        for batch in tqdm.tqdm(batched_inputs):
            # get the last token, handling right padding if present
            encoded_batch = tokenizer(batch, padding=True, return_tensors="pt").to(
                model.device
            )
            out = model(**encoded_batch, output_hidden_states=True)
            attention_mask = encoded_batch["attention_mask"]
            for i in range(len(batch)):
                last_non_padding_index = (
                    attention_mask[i].nonzero(as_tuple=True)[0][-1].item()
                )
                for layer in hidden_layers:
                    hidden_idx = layer + 1 if layer >= 0 else layer
                    hidden_state = (
                        out.hidden_states[hidden_idx][i][last_non_padding_index]
                        .cpu()
                        .float()
                        .numpy()
                    )
                    hidden_states[layer].append(hidden_state)
            del out

    return {k: np.vstack(v) for k, v in hidden_states.items()}


def batched_get_hiddens_cached(
    model,
    tokenizer,
    inputs: list[str],
    hidden_layers: list[int],
    batch_size: int,
    cache_path: os.PathLike[str] | str,
) -> str:
    """
    Cached version of batched_get_hiddens using h5py for disk storage.
    The memory requirements should be approximately the same as the batch
    itself as we don't have to hold all passed activations.

    It does not return the dict of activations but the path to the h5 cache.
    """
    os.makedirs(cache_path, exist_ok=True)

    model_name = get_model_name(model)
    cache_file = os.path.join(cache_path, f"{model_name}.h5")
    model_args = _get_model_args_string(model)
    train_strs_hash = _hash_train_strs(inputs)

    # Check if cache exists and is complete
    group_path = f"{model_args}/{train_strs_hash}"
    try:
        with h5py.File(cache_file, "r") as f:
            if (
                group_path in f
                and "done" in f[group_path]
                and f[group_path]["done"][()]
            ):
                # Cache exists and is complete, load and return
                return cache_file
    except (OSError, KeyError):
        # Cache doesn't exist or is incomplete, proceed with computation
        pass

    # Compute hidden states and cache them
    # First, we need to get one batch to determine the hidden dimension
    with torch.no_grad():
        sample_batch = inputs[: min(batch_size, len(inputs))]
        encoded_sample = tokenizer(sample_batch, padding=True, return_tensors="pt").to(
            model.device
        )
        sample_out = model(**encoded_sample, output_hidden_states=True)
        hidden_dim = sample_out.hidden_states[0].shape[-1]
        del sample_out

    # Initialize h5py datasets
    with h5py.File(cache_file, "a") as f:
        # Create group hierarchy if it doesn't exist
        if group_path not in f:
            group = f.create_group(group_path)
            # Add metadata
            group.attrs["creation_date"] = datetime.now().isoformat()
            group.attrs["repeng_version"] = __VERSION__
            group.attrs["model_name"] = model_name
            group.attrs["num_inputs"] = len(inputs)

            # Initialize datasets for each layer with zeros
            for layer in hidden_layers:
                f.create_dataset(
                    f"{group_path}/layer_{layer}",
                    shape=(len(inputs), hidden_dim),
                    dtype=np.float32,
                    compression="lzf",
                    shuffle=True,
                    chunks=True,
                    fillvalue=0.0,
                )

            # Initialize done flag as False
            f.create_dataset(f"{group_path}/done", data=False)

    # Process batches and store results
    batched_inputs = [
        inputs[p : p + batch_size] for p in range(0, len(inputs), batch_size)
    ]

    batch_start_idx = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(batched_inputs, desc="Computing and caching hiddens"):
            batch_size_actual = len(batch)

            # get the last token, handling right padding if present
            encoded_batch = tokenizer(batch, padding=True, return_tensors="pt").to(
                model.device
            )
            out = model(**encoded_batch, output_hidden_states=True)
            attention_mask = encoded_batch["attention_mask"]

            # Collect batch results by layer
            batch_hiddens = {layer: [] for layer in hidden_layers}

            for i in range(batch_size_actual):
                last_non_padding_index = (
                    attention_mask[i].nonzero(as_tuple=True)[0][-1].item()
                )
                for layer in hidden_layers:
                    hidden_idx = layer + 1 if layer >= 0 else layer
                    hidden_state = (
                        out.hidden_states[hidden_idx][i][last_non_padding_index]
                        .cpu()
                        .float()
                        .numpy()
                    )
                    batch_hiddens[layer].append(hidden_state)
            del out

            # Write batch results to h5py
            with h5py.File(cache_file, "a") as f:
                for layer in hidden_layers:
                    layer_data = np.vstack(batch_hiddens[layer])
                    f[f"{group_path}/layer_{layer}"][
                        batch_start_idx : batch_start_idx + batch_size_actual
                    ] = layer_data

            batch_start_idx += batch_size_actual

    # Mark as complete
    with h5py.File(cache_file, "a") as f:
        f[f"{group_path}/done"][()] = True

    return cache_file


def project_onto_direction(H, direction):
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    mag = np.linalg.norm(direction)
    assert not np.isinf(mag)
    return (H @ direction) / mag


def _get_model_args_string(model) -> str:
    """Generate a human-readable string for model arguments used in cache hierarchy."""
    config = model.config
    # Include key model parameters that would affect hidden states
    args = [
        f"layers_{config.num_hidden_layers}",
        f"hidden_{config.hidden_size}",
        f"type_{config.model_type}",
    ]
    if hasattr(config, "num_attention_heads"):
        args.append(f"heads_{config.num_attention_heads}")
    if hasattr(config, "intermediate_size"):
        args.append(f"inter_{config.intermediate_size}")

    return "_".join(args)


def _hash_train_strs(train_strs: list[str]) -> str:
    """Generate a hash of the training strings for cache key."""
    # Create a deterministic hash of all training strings
    hash_sum = sum(hash(elem) for elem in train_strs)
    return str(abs(hash_sum))
