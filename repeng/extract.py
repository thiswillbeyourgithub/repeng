import dataclasses
import os
import typing
import warnings

import gguf
import numpy as np
from sklearn.decomposition import PCA
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import tqdm

from .control import ControlModel, model_layer_list
from .saes import Sae

if not hasattr(np, "float_"):
    np.float_ = np.float64

VERBOSE = False
LOW_MEMORY = True

@dataclasses.dataclass
class DatasetEntry:
    positive: typing.Union[str, typing.List]
    negative: typing.Union[str, typing.List]


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
        **kwargs,
    ) -> "ControlVector":
        """
        Train a ControlVector for a given model and tokenizer using the provided dataset.

        Args:
            model (PreTrainedModel | ControlModel): The model to train against.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to tokenize the dataset.
            dataset (list[DatasetEntry]): The dataset used for training.
            **kwargs: Additional keyword arguments.
                max_batch_size (int, optional): The maximum batch size for training.
                    Defaults to 32. Try reducing this if you're running out of memory.
                method (str, optional): The training method to use. Can be either
                    "pca_diff" or "pca_center". Defaults to "pca_diff".

        Returns:
            ControlVector: The trained vector.
        """
        with torch.inference_mode():
            dirs = read_representations(
                model,
                tokenizer,
                dataset,
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
        method: typing.Literal["pca_diff", "pca_center", "umap", "pacmap"] = "pca_center",
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
                max_batch_size (int, optional): The maximum batch size for training.
                    Defaults to 32. Try reducing this if you're running out of memory.
                method (str, optional): The training method to use. Can be either
                    "pca_diff", "pca_center", "umap" or "pacmap". Defaults to "pca_center"! This is different
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
            except:
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

    def __mul__(self, other: int | float | np.int_ | np.float_) -> "ControlVector":
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = other * self.directions[layer]
        return ControlVector(model_type=self.model_type, directions=directions)

    def __rmul__(self, other: int | float | np.int_ | np.float_) -> "ControlVector":
        return self.__mul__(other)

    def __truediv__(self, other: int | float | np.int_ | np.float_) -> "ControlVector":
        return self.__mul__(1 / other)


def read_representations(
    model: "PreTrainedModel | ControlModel",
    tokenizer: PreTrainedTokenizerBase,
    inputs: list[DatasetEntry],
    hidden_layers: typing.Iterable[int] | None = None,
    batch_size: int = 32,
    method: typing.Literal["pca_diff", "pca_center", "umap", "pacmap", "umap_kmeans_pca_diff"] = "pca_diff",
    transform_hiddens: (
        typing.Callable[[dict[int, np.ndarray]], dict[int, np.ndarray]] | None
    ) = None,
) -> dict[int, np.ndarray]:
    """
    Extract the representations based on the contrast dataset.
    """
    if not hidden_layers:
        hidden_layers = range(-1, -model.config.num_hidden_layers, -1)

    # normalize the layer indexes if they're negative
    n_layers = len(model_layer_list(model))
    hidden_layers = [i if i >= 0 else n_layers + i for i in hidden_layers]

    # the order is [positive, negative, positive, negative, ...]
    train_strs = [s for ex in inputs for s in (ex.positive, ex.negative)]

    # apply chat template or not, depending on the type
    for iex, ex in enumerate(train_strs):
        if isinstance(ex, list):
            train_strs[iex] = tokenizer.apply_chat_template(ex, tokenize=False)

    layer_hiddens = batched_get_hiddens(
        model, tokenizer, train_strs, hidden_layers, batch_size
    )

    if transform_hiddens is not None:
        layer_hiddens = transform_hiddens(layer_hiddens)

    # get directions for each layer using PCA
    directions: dict[int, np.ndarray] = {}
    for layer in tqdm.tqdm(hidden_layers, desc="Altering direction"):
        h = layer_hiddens[layer]
        assert h.shape[0] == len(inputs) * 2

        if np.isnan(h.ravel()).all():
            warnings.warn(f"Skipping layer {layer} because the vector is full of nan")
            continue
        elif np.isnan(h.ravel()).any():
            warnings.warn(f"Skipping layer {layer} because the vector contains at least one nan")
            continue

        if method == "pca_diff":
            train = h[::2] - h[1::2]
        elif method == "pca_center":
            center = (h[::2] + h[1::2]) / 2
            train = h
            train[::2] -= center
            train[1::2] -= center
        elif method in ["umap", "pacmap", "umap_kmeans_pca_diff"]:
            train = h
        else:
            raise ValueError("unknown method " + method)

        if method in ["pca_center", "pca_diff"]:
            # shape (1, n_features)
            pca_model = PCA(n_components=1, whiten=False).fit(train)
            # shape (n_features,)
            newlayer = pca_model.components_.astype(np.float32).squeeze(axis=0)

        elif method == "umap":
            # still experimental so don't want to add this as a real dependency yet
            import umap  # type: ignore

            umap_model = umap.UMAP(
                n_components=1,
                # low_memory=True,
                # random_state=42,
                # transform_seed=42,
                # densmap=True,
            )
            embedding = umap_model.fit_transform(train).astype(np.float32)
            # embedding = umap_model.fit_transform(train.T).astype(np.float32)
            embedding /= np.abs(embedding.ravel()).max()
            if VERBOSE:
                print("Embedding:")
                print(embedding)
            # newlayer = embedding.squeeze()
            newlayer = np.sum(train * embedding, axis=0) / np.sum(embedding)
            # newlayer = (train.T @ embedding).squeeze()

        elif method == "umap_kmeans_pca_diff":
            # still experimental so don't want to add this as a real dependency yet
            import umap
            from sklearn.cluster import KMeans

            n_clusters = 10

            # First reduce to 2D with UMAP
            umap_model = umap.UMAP(n_components=2, random_state=42, transform_seed=42, densmap=True)
            umap_embedding = umap_model.fit_transform(train)

            # Run KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(umap_embedding)

            # For each cluster, run PCA on the differences between positive and negative samples
            newlayer = np.zeros_like(train[0])
            for cluster_idx in range(n_clusters):
                cluster_mask = clusters == cluster_idx
                if np.sum(cluster_mask) > 1:  # Only process clusters with >1 sample
                    # Get original indices for this cluster
                    cluster_indices = np.where(cluster_mask)[0]
                    # Map back to original pairs (even=positive, odd=negative)
                    pairs = []
                    for idx in cluster_indices:
                        pair_idx = idx // 2 * 2  # Get the even index for this pair
                        if pair_idx + 1 < len(train):  # Make sure we have both positive and negative
                            pairs.append((pair_idx, pair_idx + 1))
                    
                    if pairs:  # Only process if we have complete pairs
                        # Calculate differences between positive and negative samples
                        differences = np.array([train[pos] - train[neg] for pos, neg in pairs])
                        if len(differences) > 1:  # Need at least 2 samples for PCA
                            pca_model = PCA(n_components=1, whiten=False).fit(differences)
                            cluster_direction = pca_model.components_.squeeze()
                            # Weight by number of pairs in cluster
                            newlayer += cluster_direction * len(pairs)
                    else:
                        raise Exception("missing pair")
            
            # Normalize
            newlayer = newlayer.astype(np.float32)
            newlayer /= np.linalg.norm(newlayer)  # L2 norm
            # newlayer /= np.linalg.norm(newlayer, 1)  # L1 norm

        elif method == "pacmap":
            # still experimental so don't want to add this as a real dependency yet
            import pacmap  # type: ignore

            pacmap_model = pacmap.PaCMAP(
                n_components=1,
                verbose=False,
                apply_pca=True,  # wether to start by a pca or not, not the same as 'init'
            )
            pm_embedding = pacmap_model.fit_transform(train.T, init="pca").T.astype(np.float32)
            # pm_embedding = pacmap_model.fit_transform(train.T, init="random").T.astype(np.float32)
            pm_embedding /= np.abs(pm_embedding.ravel()).max()
            if VERBOSE:
                print("Embedding:")
                print(pm_embedding)

            newlayer = np.sum(train * pm_embedding, axis=0) / np.sum(pm_embedding)

        directions[layer] = newlayer

        if VERBOSE:
            print("Direction of layer:")
            print(directions[layer])

        # calculate sign
        projected_hiddens = project_onto_direction(h, directions[layer])
        if VERBOSE:
            print("Projection:")
            print(projected_hiddens)


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
        for batch in tqdm.tqdm(batched_inputs, desc="Getting activations"):
            # get the last token, handling right padding if present
            encoded_batch = tokenizer(batch, padding=True, return_tensors="pt").to(model.device)
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
                        # .cpu()
                        # .float()
                        # .numpy()
                    )
                    if LOW_MEMORY:
                        if len(hidden_states[layer]):
                            hidden_states[layer] = np.vstack((hidden_states[layer], hidden_state.cpu().float().numpy()))
                        else:
                            hidden_states[layer].append(hidden_state.cpu().float().numpy())
                    else:
                        hidden_states[layer].append(hidden_state)
            del out

    if LOW_MEMORY:
        return hidden_states
    else:
        return {k: torch.vstack(v).cpu().float().numpy() for k, v in hidden_states.items()}


def project_onto_direction(H, direction):
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    mag = np.linalg.norm(direction)
    assert not np.isinf(mag)
    return (H @ direction) / mag
