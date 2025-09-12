import json
import dataclasses
import hashlib
import os
import typing
import warnings
from datetime import datetime

import gguf
import h5py
import numpy as np
from sklearn.decomposition import PCA, FastICA, DictionaryLearning
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import tqdm
from loguru import logger

from .control import ControlModel, model_layer_list
from .saes import Sae
from .utils import (
    DatasetEntry,
    get_model_name,
    autocorrect_chat_templates,
    get_num_hidden_layer,
)

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
        rescaling: str | None = "layer_magnitude",
        enable_thinking: bool = False,
        output_training_avg_logprob: bool = False,
        **kwargs,
    ) -> "ControlVector | tuple[ControlVector, list[float]]":
        """
        Train a ControlVector for a given model and tokenizer using the provided dataset.

        Args:
            model (PreTrainedModel | ControlModel): The model to train against.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to tokenize the dataset.
            dataset (list[DatasetEntry]): The dataset used for training.
            cache_path (os.PathLike[str] | str | None, optional): Path to directory for h5py caching.
                If None, activations are computed and stored in memory. If provided, activations
                are cached to disk to allow for better memory scaling. Defaults to None.
            rescaling (str | None, optional): How to rescale the direction vectors. If None,
                uses original scaling. If "layer_magnitude", rescales to match typical activation
                magnitude in each layer. Defaults to "layer_magnitude".
            output_training_avg_logprob (bool, optional): If True, also return average log
                probabilities for each training example. Defaults to False.
            **kwargs: Additional keyword arguments. See help(repeng.extract.read_representations) for details.

        Returns:
            ControlVector | tuple[ControlVector, list[float]]: The trained vector, and optionally
                the average log probabilities for each training example if output_training_avg_logprob is True.
        """
        with torch.inference_mode():
            result = read_representations(
                model,
                tokenizer,
                dataset,
                cache_path=cache_path,
                rescaling=rescaling,
                enable_thinking=enable_thinking,
                output_training_avg_logprob=output_training_avg_logprob,
                **kwargs,
            )

            if output_training_avg_logprob:
                dirs, avg_logprobs = result
                control_vector = cls(
                    model_type=model.config.model_type, directions=dirs
                )
                return control_vector, avg_logprobs
            else:
                dirs = result
                return cls(model_type=model.config.model_type, directions=dirs)

    @classmethod
    def train_with_sae(
        cls,
        model: "PreTrainedModel | ControlModel",
        tokenizer: PreTrainedTokenizerBase,
        dataset: typing.List[DatasetEntry],
        sae: Sae,
        decode: bool = True,
        cache_path: os.PathLike[str] | str | None = None,
        rescaling: str | None = "layer_magnitude",
        enable_thinking: bool = False,
        output_training_avg_logprob: bool = False,
        **kwargs,
    ) -> "ControlVector | tuple[ControlVector, list[float]]":
        """
        Like ControlVector.train, but using an SAE. It's better! WIP.


        Args:
            model (PreTrainedModel | ControlModel): The model to train against.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to tokenize the dataset.
            dataset (list[DatasetEntry]): The dataset used for training.
            sae (saes.Sae): See the `saes` module for how to load this.
            decode (bool, optional): Whether to decode the vector to make it immediately usable.
                If not, keeps it as monosemantic SAE features for introspection, but you will need to decode it manually
                to use it. Defaults to True.
            cache_path (os.PathLike[str] | str | None, optional): Path to directory for h5py caching.
                If None, activations are computed and stored in memory. If provided, activations
                are cached to disk to allow for better memory scaling. Defaults to None.
            rescaling (str | None, optional): How to rescale the direction vectors. If None,
                uses original scaling. If "layer_magnitude", rescales to match typical activation
                magnitude in each layer. Defaults to "layer_magnitude".
            output_training_avg_logprob (bool, optional): If True, also return average log
                probabilities for each training example. Defaults to False.
            **kwargs: Additional keyword arguments. See help(repeng.extract.read_representations) for details.

        Returns:
            ControlVector | tuple[ControlVector, list[float]]: The trained vector, and optionally
                the average log probabilities for each training example if output_training_avg_logprob is True.
        """
        with torch.inference_mode():
            result = read_representations(
                model,
                tokenizer,
                dataset,
                sae=sae,
                sae_decode=decode,
                rescaling=rescaling,
                cache_path=cache_path,
                enable_thinking=enable_thinking,
                output_training_avg_logprob=output_training_avg_logprob,
                **kwargs,
            )

            if output_training_avg_logprob:
                dirs, avg_logprobs = result
                control_vector = cls(
                    model_type=model.config.model_type, directions=dirs
                )
                return control_vector, avg_logprobs
            else:
                dirs = result
                return cls(model_type=model.config.model_type, directions=dirs)

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
        typing.Literal[
            "pca_diff",
            "pca_center",
            "mean",
            "median",
            "umap",
            "umap_densmap",
            "ica_diff",
            "ica_center",
            "dict_diff",
            "dict_center",
        ],
        typing.Callable[[np.ndarray], np.ndarray],
    ],
    rescaling: str | None,
) -> np.ndarray:
    """
    Compute a direction vector from hidden states using the specified method.

    Args:
        hidden_states (np.ndarray): Hidden states array of shape (n_samples, hidden_dim).
            For contrast methods, should have even number of samples where pairs represent
            [positive, negative, positive, negative, ...] examples.
        method: The method to use for computing the direction. Can be "pca_diff",
            "pca_center", "mean", "median", "umap", "umap_densmap", "ica_diff", "ica_center", "dict_diff", "dict_center", or a callable that takes hidden states and returns
            a direction vector.
        rescaling (str | None, optional): How to rescale the direction vector. If None,
            uses original scaling. If "layer_magnitude", rescales to match typical activation
            magnitude. If interesting in rescaling, you might want to look
            at the `normalize` argument of `control.py:ControlModel.set_control`.py`.

    Returns:
        np.ndarray: Direction vector of shape (hidden_dim,).
    """
    # Compute the direction based on the method
    if callable(method):
        # Custom method: directly compute direction from hidden states
        direction = method(hidden_states).astype(np.float32)
    elif method == "pca_diff":
        train = hidden_states[::2] - hidden_states[1::2]
        # shape (1, n_features)
        pca_model = PCA(n_components=1, whiten=False).fit(train)
        # shape (n_features,)
        direction = pca_model.components_.astype(np.float32).squeeze(axis=0)
    elif method == "pca_center":
        center = (hidden_states[::2] + hidden_states[1::2]) / 2
        train = hidden_states.copy()
        train[::2] -= center
        train[1::2] -= center
        # shape (1, n_features)
        pca_model = PCA(n_components=1, whiten=False).fit(train)
        # shape (n_features,)
        direction = pca_model.components_.astype(np.float32).squeeze(axis=0)
    elif method == "mean":
        # Compute direction as difference between mean of positive and negative samples
        # Order is [positive, negative, positive, negative, ...]
        positive_states = hidden_states[::2]  # Every even index
        negative_states = hidden_states[1::2]  # Every odd index

        mean_positive = np.mean(positive_states, axis=0)
        mean_negative = np.mean(negative_states, axis=0)

        # Direction points from negative to positive
        direction = (mean_positive - mean_negative).astype(np.float32)
    elif method == "median":
        # Compute direction as difference between median of positive and negative samples
        # Order is [positive, negative, positive, negative, ...]
        positive_states = hidden_states[::2]  # Every even index
        negative_states = hidden_states[1::2]  # Every odd index

        median_positive = np.median(positive_states, axis=0)
        median_negative = np.median(negative_states, axis=0)

        # Direction points from negative to positive
        direction = (median_positive - median_negative).astype(np.float32)
    elif method == "umap":
        train = hidden_states
        # still experimental so don't want to add this as a real dependency yet
        import umap  # type: ignore

        umap_model = umap.UMAP(n_components=1, low_memory=True)
        embedding = umap_model.fit_transform(train).astype(np.float32)
        direction = np.sum(train * embedding, axis=0) / np.sum(embedding)
    elif method == "umap_densmap":
        train = hidden_states
        # still experimental so don't want to add this as a real dependency yet
        import umap  # type: ignore

        umap_model = umap.UMAP(n_components=1, densmap=True, low_memory=True)
        embedding = umap_model.fit_transform(train).astype(np.float32)
        direction = np.sum(train * embedding, axis=0) / np.sum(embedding)
    elif method == "ica_diff":
        # Use difference between positive and negative examples for ICA
        train = hidden_states[::2] - hidden_states[1::2]
        # Fit ICA with 1 component to extract the most independent direction
        ica_model = FastICA(n_components=1, whiten="unit-variance", random_state=42)
        ica_model.fit(train)
        # Return the first (and only) component, shape (n_features,)
        direction = ica_model.components_.astype(np.float32).squeeze(axis=0)
    elif method == "ica_center":
        # Use centered data for ICA (like pca_center)
        center = (hidden_states[::2] + hidden_states[1::2]) / 2
        train = hidden_states.copy()
        train[::2] -= center
        train[1::2] -= center
        # Fit ICA with 1 component to extract the most independent direction
        ica_model = FastICA(n_components=1, whiten="unit-variance", random_state=42)
        ica_model.fit(train)
        # Return the first (and only) component, shape (n_features,)
        direction = ica_model.components_.astype(np.float32).squeeze(axis=0)
    elif method == "dict_diff":
        # Use difference between positive and negative examples for Dictionary Learning
        train = hidden_states[::2] - hidden_states[1::2]
        # Fit Dictionary Learning with 1 component to extract the most representative atom
        dict_model = DictionaryLearning(n_components=1, random_state=42, max_iter=100)
        dict_model.fit(train)
        # Return the first (and only) dictionary atom, shape (n_features,)
        direction = dict_model.components_.astype(np.float32).squeeze(axis=0)
    elif method == "dict_center":
        # Use centered data for Dictionary Learning (like pca_center)
        center = (hidden_states[::2] + hidden_states[1::2]) / 2
        train = hidden_states.copy()
        train[::2] -= center
        train[1::2] -= center
        # Fit Dictionary Learning with 1 component to extract the most representative atom
        dict_model = DictionaryLearning(n_components=1, random_state=42, max_iter=100)
        dict_model.fit(train)
        # Return the first (and only) dictionary atom, shape (n_features,)
        direction = dict_model.components_.astype(np.float32).squeeze(axis=0)
    else:
        raise ValueError(f"unknown method {method}")

    # Apply rescaling if requested
    if rescaling == "layer_magnitude":
        # Calculate typical magnitude of activations in this layer
        activation_magnitudes = np.linalg.norm(hidden_states, axis=1, ord=2)
        typical_magnitude = np.mean(activation_magnitudes)

        # Normalize direction to unit length, then scale by typical magnitude
        direction_norm = np.linalg.norm(direction, ord=2)
        if direction_norm != 0:  # avoid the rare division by 0
            direction = (direction / direction_norm) * typical_magnitude
    elif rescaling is not None:
        raise ValueError(f"unknown rescaling method {rescaling}")

    return direction


def read_representations(
    model: "PreTrainedModel | ControlModel",
    tokenizer: PreTrainedTokenizerBase,
    inputs: list[DatasetEntry],
    hidden_layers: typing.Iterable[int] | None = None,
    batch_size: int = 32,
    method: typing.Union[
        typing.Literal[
            "pca_diff",
            "pca_center",
            "mean",
            "median",
            "umap",
            "umap_densmap",
            "ica_diff",
            "ica_center",
            "dict_diff",
            "dict_center",
        ],
        typing.Callable[[np.ndarray], np.ndarray],
    ] = "pca_diff",
    sae: Sae | None = None,
    sae_decode: bool = True,
    rescaling: str | None = None,
    cache_path: os.PathLike[str] | str | None = None,
    enable_thinking: bool = False,
    output_training_avg_logprob: bool = False,
) -> "dict[int, np.ndarray] | tuple[dict[int, np.ndarray], list[float]]":
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
            "pca_diff", "pca_center", "mean", "median", "umap", "umap_densmap", "ica_diff", "ica_center", "dict_diff", "dict_center", or a callable that takes hidden states
            array of shape (n_samples, hidden_dim) and returns a direction vector
            of shape (hidden_dim,). Defaults to "pca_diff".
        sae (Sae | None, optional): Optional SAE to use for transforming hidden states
            before computing directions. If provided, hidden states will be encoded
            through the SAE. Defaults to None.
        sae_decode (bool, optional): If using SAE, whether to decode the direction vectors
            back to the original space. If False, returns directions in SAE feature space.
            Defaults to True.
        rescaling (str | None, optional): How to rescale the direction vectors. If None,
            uses original scaling. If "layer_magnitude", rescales to match typical activation
            magnitude in each layer. Defaults to None.
        cache_path (os.PathLike[str] | str | None, optional): Path to directory for h5py caching.
            If None, activations are computed and stored in memory. If provided, activations
            are cached to disk to allow for better memory scaling. Defaults to None.
        enable_thinking (bool, optional): Whether to enable thinking tokens when applying
            chat templates to the dataset entries. This controls the `enable_thinking`
            parameter passed to `tokenizer.apply_chat_template()`. Defaults to False.
        output_training_avg_logprob (bool, optional): If True, also return average log
            probabilities for each training example. Defaults to False.
    """
    if not hidden_layers:
        hidden_layers = list(range(get_num_hidden_layer(model)))

    n_layers = len(model_layer_list(model))

    # the order is [positive, negative, positive, negative, ...]
    train_list = []
    [train_list.extend([ex.positive, ex.negative]) for ex in inputs]
    assert len(train_list) == len(
        set([json.dumps(ex) for ex in train_list])
    ), "There are duplicates in the training dataset"
    try:
        train_strs: list[str] = [
            tokenizer.apply_chat_template(
                conversation=chat,
                tokenize=False,
                enable_thinking=enable_thinking,
            )
            for chat in train_list
        ]
        # check that there are no duplicates, because some chat template silently
        # drop the system prompt so we instead use the autocorrect function
        assert len(train_strs) == len(
            set(train_strs)
        ), "There are duplicates in the training dataset"
    except Exception as e:
        warnings.warn(
            f"Error when applying chat template: '{e}'\nTrying to autocorrect the template anyway."
        )
        train_strs: list[str] = [
            autocorrect_chat_templates(
                messages=chat,
                tokenizer=tokenizer,
                model=model,
            )
            for chat in train_list
        ]

    assert len(train_strs) == len(
        set(train_strs)
    ), "There are duplicates in the training dataset"

    if cache_path is None:
        # Original behavior - store all activation layers in memory
        logger.debug("No cache path provided, computing activations in memory")
        layer_hiddens, log_probs = batched_get_hiddens(
            model=model,
            tokenizer=tokenizer,
            inputs=train_strs,
            hidden_layers=hidden_layers,
            batch_size=batch_size,
        )

        if sae is not None:
            logger.debug("Applying SAE transformation in memory")
            sae_hiddens = {}
            for k, v in tqdm.tqdm(layer_hiddens.items(), desc="sae encoding"):
                sae_hiddens[k] = sae.layers[k].encode(v)
            layer_hiddens = sae_hiddens
    else:
        logger.debug(f"Using h5py cache at {cache_path}")
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

        # Load log probabilities from cache if needed
        if output_training_avg_logprob:
            with h5py.File(cache_file, "r") as f:
                log_probs = f[group_path]["log_probs"][:]

        # SAE transformation with caching
        if sae is not None:
            sae_group_path = f"{group_path}_sae"
            # Check if SAE cache exists and is complete
            sae_cache_exists = False
            try:
                with h5py.File(cache_file, "r") as f:
                    if (
                        sae_group_path in f
                        and "done_sae" in f[sae_group_path]
                        and f[sae_group_path]["done_sae"][()]
                    ):
                        sae_cache_exists = True
            except (OSError, KeyError):
                pass

            # Apply SAE transformation if cache doesn't exist
            if not sae_cache_exists:
                logger.debug(
                    f"SAE cache not found, computing SAE transformations for {sae_group_path}"
                )
                apply_sae_transform_cached(
                    cache_file=cache_file,
                    group_path=group_path,
                    sae=sae,
                    hidden_layers=hidden_layers,
                )
            else:
                logger.debug(
                    f"Found existing SAE cache at {sae_group_path}, skipping SAE computation"
                )

    # Compute average log probabilities if requested
    avg_logprobs = None
    if output_training_avg_logprob:
        # log_probs shape: (n_inputs, max_seq_len)
        # We need to compute average for each input, excluding padding tokens
        avg_logprobs = []
        for i, input_str in enumerate(train_strs):
            # Get the actual length of this input (non-padding tokens)
            tokens = tokenizer(input_str, return_tensors="pt")
            actual_length = tokens["attention_mask"].sum().item()
            # Average log prob for this input, excluding padding
            avg_logprob = np.mean(
                log_probs[i][: actual_length - 1]
            )  # -1 because log_probs exclude first token
            avg_logprobs.append(float(avg_logprob))

    # get directions for each layer using the specified method
    directions: dict[int, np.ndarray] = {}
    for layer in tqdm.tqdm(hidden_layers, desc="Altering directions"):
        # Load hidden states either from memory dict or h5py cache
        if cache_path is None:
            # Load from memory
            h = layer_hiddens[layer]
        else:
            # Load from h5py cache only a single layer's activations
            source_group = f"{group_path}_sae" if sae is not None else group_path
            with h5py.File(cache_file, "r") as f:
                h = f[source_group][f"layer_{layer}"][:]
            assert not np.all(h == 0)  # failed to get populated

        assert h.shape[0] == len(inputs) * 2

        # check that there are no duplicates
        unique_rows, counts = np.unique(h, axis=0, return_counts=True)
        assert not np.any(
            counts > 1
        ), f"Duplicates hidden layer activation found. Counts: {counts}"

        directions[layer] = compute_direction(h, method, rescaling)

        if method not in ["mean", "median"]:
            # calculate sign as pca can return a direction vector that points
            # either way along the principal component. There's no inherent
            # orientation - PCA just finds the direction of maximum variance,
            # but it could be pointing towards the positive concept or towards
            # the negative concept
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
                logger.debug(f"Direction of layer {layer} had to be flipped")

        # Decode SAE directions back to original space if requested
        # TODO: should the decoding take place after the flip or before? Does it even matter?
        if sae is not None and sae_decode:
            directions[layer] = sae.layers[layer].decode(directions[layer])

    if output_training_avg_logprob:
        return directions, avg_logprobs
    else:
        return directions


def batched_get_hiddens(
    model,
    tokenizer,
    inputs: list[str],
    hidden_layers: list[int],
    batch_size: int,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """
    Using the given model and tokenizer, pass the inputs through the model and get the hidden
    states for each layer in `hidden_layers` for the last token, plus log probabilities for all tokens.

    Returns a tuple of:
    - Dictionary from `hidden_layers` layer id to numpy array of shape `(n_inputs, hidden_dim)`
    - Log probabilities array of shape `(n_inputs, max_seq_len)` for all tokens
    """
    batched_inputs = [
        inputs[p : p + batch_size] for p in range(0, len(inputs), batch_size)
    ]
    hidden_states = {layer: [] for layer in hidden_layers}
    all_log_probs = []

    with torch.no_grad():
        for batch in tqdm.tqdm(batched_inputs, desc="Computing activations"):
            # get the last token, handling right padding if present
            encoded_batch = tokenizer(batch, padding=True, return_tensors="pt").to(
                model.device
            )
            out = model(**encoded_batch, output_hidden_states=True)
            attention_mask = encoded_batch["attention_mask"]

            # Compute log probabilities for all tokens
            logits = out.logits  # shape: (batch_size, seq_len, vocab_size)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            input_ids = encoded_batch["input_ids"]

            # Get log prob for each actual token (excluding padding)
            batch_log_probs = []
            for i in range(len(batch)):
                # Get log probs for this sequence's actual tokens
                seq_len = attention_mask[i].sum().item()
                # Get log prob of each actual token in the sequence
                token_log_probs = (
                    log_probs[i, : seq_len - 1, :]
                    .gather(dim=1, index=input_ids[i, 1:seq_len].unsqueeze(1))
                    .squeeze(1)
                )

                # Pad to max sequence length for consistent storage
                max_len = (
                    logits.shape[1] - 1
                )  # -1 because we skip first token for log prob
                padded_log_probs = torch.zeros(max_len, device=token_log_probs.device)
                padded_log_probs[: len(token_log_probs)] = token_log_probs
                batch_log_probs.append(padded_log_probs.cpu().float().numpy())

                # Get hidden states for last non-padding token
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

            all_log_probs.extend(batch_log_probs)
            del out

    return {k: np.vstack(v) for k, v in hidden_states.items()}, np.vstack(all_log_probs)


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
    Always computes and caches log probabilities along with hidden states.

    Returns the path to the h5 cache file.
    """
    model_name = get_model_name(model)
    cache_file = os.path.join(cache_path, f"{model_name}.h5")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
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
                logger.debug(
                    f"Found existing activation cache at {group_path}, skipping computation"
                )
                return cache_file
    except (OSError, KeyError):
        # Cache doesn't exist or is incomplete, proceed with computation
        pass

    logger.debug(f"Activation cache not found, computing activations for {group_path}")

    # Compute hidden states and cache them
    # First, we need to get one batch to determine dimensions
    with torch.no_grad():
        sample_batch = inputs[: min(batch_size, len(inputs))]
        encoded_sample = tokenizer(sample_batch, padding=True, return_tensors="pt").to(
            model.device
        )
        sample_out = model(**encoded_sample, output_hidden_states=True)
        hidden_dim = sample_out.hidden_states[0].shape[-1]
        max_seq_len = sample_out.logits.shape[1] - 1  # -1 for log prob computation
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

            # Initialize log probabilities dataset
            f.create_dataset(
                f"{group_path}/log_probs",
                shape=(len(inputs), max_seq_len),
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

            # Compute log probabilities for all tokens
            logits = out.logits  # shape: (batch_size, seq_len, vocab_size)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            input_ids = encoded_batch["input_ids"]

            # Collect batch results by layer and log probs
            batch_hiddens = {layer: [] for layer in hidden_layers}
            batch_log_probs = []

            for i in range(batch_size_actual):
                last_non_padding_index = (
                    attention_mask[i].nonzero(as_tuple=True)[0][-1].item()
                )

                # Collect hidden states
                for layer in hidden_layers:
                    hidden_idx = layer + 1 if layer >= 0 else layer
                    hidden_state = (
                        out.hidden_states[hidden_idx][i][last_non_padding_index]
                        .cpu()
                        .float()
                        .numpy()
                    )
                    batch_hiddens[layer].append(hidden_state)

                # Collect log probabilities
                seq_len = attention_mask[i].sum().item()
                # Get log prob of each actual token in the sequence
                token_log_probs = (
                    log_probs[i, : seq_len - 1, :]
                    .gather(dim=1, index=input_ids[i, 1:seq_len].unsqueeze(1))
                    .squeeze(1)
                )

                # Pad to max sequence length for consistent storage
                padded_log_probs = torch.zeros(
                    max_seq_len, device=token_log_probs.device
                )
                padded_log_probs[: len(token_log_probs)] = token_log_probs
                batch_log_probs.append(padded_log_probs.cpu().float().numpy())

            del out

            # Write batch results to h5py
            with h5py.File(cache_file, "a") as f:
                # Store hidden states
                for layer in hidden_layers:
                    layer_data = np.vstack(batch_hiddens[layer])
                    f[f"{group_path}/layer_{layer}"][
                        batch_start_idx : batch_start_idx + batch_size_actual
                    ] = layer_data

                # Store log probabilities
                log_prob_data = np.vstack(batch_log_probs)
                f[f"{group_path}/log_probs"][
                    batch_start_idx : batch_start_idx + batch_size_actual
                ] = log_prob_data

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
    if hasattr(config, "text_config"):  # gemma 3 config
        args = [
            f"layers_{config.text_config.num_hidden_layers}",
            f"hidden_{config.text_config.hidden_size}",
            f"type_{config.text_config.model_type}",
        ]
        if hasattr(config.text_config, "num_attention_heads"):
            args.append(f"heads_{config.text_config.num_attention_heads}")
        if hasattr(config.text_config, "intermediate_size"):
            args.append(f"inter_{config.text_config.intermediate_size}")
    else:
        args = [
            f"layers_{config.num_hidden_layers}",
            f"hidden_{config.hidden_size}",
            f"type_{config.model_type}",
        ]
        if hasattr(config, "num_attention_heads"):
            args.append(f"heads_{config.num_attention_heads}")
        if hasattr(config, "intermediate_size"):
            args.append(f"inter_{config.intermediate_size}")

    # Include dtype information - critical for cache correctness
    if hasattr(model, "dtype") and model.dtype is not None:
        args.append(f"dtype_{str(model.dtype).replace('torch.', '')}")
    else:
        # Fallback: check dtype of first parameter
        try:
            first_param = next(iter(model.parameters()))
            args.append(f"dtype_{str(first_param.dtype).replace('torch.', '')}")
        except (StopIteration, AttributeError):
            args.append("dtype_unknown")

    # Include quantization config - different quantization affects hidden states
    if (
        hasattr(config, "quantization_config")
        and config.quantization_config is not None
    ):
        quant_config = config.quantization_config
        if hasattr(quant_config, "load_in_4bit") and quant_config.load_in_4bit:
            args.append("quant_4bit")
            if hasattr(quant_config, "bnb_4bit_quant_type"):
                args.append(f"qtype_{quant_config.bnb_4bit_quant_type}")
        elif hasattr(quant_config, "load_in_8bit") and quant_config.load_in_8bit:
            args.append("quant_8bit")
        elif hasattr(quant_config, "quant_method"):
            args.append(f"quant_{quant_config.quant_method}")
        else:
            args.append("quant_other")
    else:
        args.append("quant_none")

    return "_".join(args)


def apply_sae_transform_cached(
    cache_file: str,
    group_path: str,
    sae: Sae,
    hidden_layers: list[int],
) -> None:
    """
    Apply SAE transformation to cached hidden states with caching of transformed values.
    This function stores the transformed layers to the cache.

    Args:
        cache_file (str): Path to the h5py cache file.
        group_path (str): Base group path in the cache file.
        sae (Sae): SAE model to apply transformation.
        hidden_layers (list[int]): List of layer indices to transform.
    """

    sae_group_path = f"{group_path}_sae"

    with h5py.File(cache_file, "a") as f:
        # Create SAE group if it doesn't exist
        if sae_group_path not in f:
            sae_group = f.create_group(sae_group_path)
            sae_group.attrs["creation_date"] = datetime.now().isoformat()
            sae_group.attrs["repeng_version"] = __VERSION__
            sae_group.attrs["sae_applied"] = True

        for layer in tqdm.tqdm(hidden_layers, desc="SAE encoding and caching"):
            # Load original hidden states
            original_hiddens = f[group_path][f"layer_{layer}"][:]

            # Apply SAE encoding
            transformed_hiddens = sae.layers[layer].encode(original_hiddens)

            # Cache the transformed hiddens
            if f"layer_{layer}" not in f[sae_group_path]:
                f.create_dataset(
                    f"{sae_group_path}/layer_{layer}",
                    data=transformed_hiddens,
                    compression="lzf",
                    shuffle=True,
                    chunks=True,
                )
            else:
                f[f"{sae_group_path}/layer_{layer}"][:] = transformed_hiddens

        # Mark SAE transformation as complete
        if "done_sae" not in f[sae_group_path]:
            f.create_dataset(f"{sae_group_path}/done_sae", data=True)
        else:
            f[f"{sae_group_path}/done_sae"][()] = True


def _hash_train_strs(train_strs: list[str]) -> str:
    """Generate a hash of the training strings for cache key."""
    # Create a deterministic hash of all training strings
    content = "\n".join(train_strs).encode("utf-8")
    hash_hex = hashlib.sha256(content).hexdigest()
    return hash_hex[:12]  # Use first 12 characters for readability
