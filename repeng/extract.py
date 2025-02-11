import dataclasses
import os
import typing
import warnings
from joblib import Memory
from pathlib import Path

import gguf
import numpy as np
from sklearn.decomposition import PCA
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import tqdm

from .control import ControlModel, model_layer_list
from .saes import Sae
from .settings import VERBOSE, LOW_MEMORY
from .utils import autocorrect_chat_templates, DatasetEntry, get_model_name, detect_norm_type

if not hasattr(np, "float_"):
    np.float_ = np.float64

# Setup cache
cache_dir = Path.home() / ".cache" / "repeng"
memory = Memory(cache_dir, verbose=0)

@memory.cache(ignore=["model", "encoded_batch"])
def cached_forward(model, encoded_batch, model_name: str, device: str, batch: typing.List[str], **kwargs):
    if VERBOSE:
        print("cache bypassed")
    return model(**encoded_batch, output_hidden_states=True, **kwargs)

def _model_forward(model, encoded_batch, batch, use_cache=True):
    """Model forward pass with optional caching"""
    if use_cache:
        # the joblib cache can't handle pickling models etc so we just take the string of the dict
        return cached_forward(
            model=model,
            encoded_batch=encoded_batch,
            model_name=get_model_name(model),
            device=str(model.device),
            batch=batch,
        )
    else:
        return model(**encoded_batch, output_hidden_states=True)

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
        method: typing.Literal["pca_diff", "pca_center", "umap", "pacmap", "umap_kmeans_pca_diff", "pacmap_kmeans_pca_diff"] = "pca_center",
        **kwargs,
    ) -> "ControlVector":
        """
        Train a ControlVector for a given model and tokenizer using the provided dataset.

        Args:
            model (PreTrainedModel | ControlModel): The model to train against.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to tokenize the dataset.
            dataset (list[DatasetEntry]): The dataset used for training.
            method (str, optional): The training method to use. Can be either
                "pca_diff", "pca_center", "umap", "umap_kmeans_pca_diff", "pacmap_kmeans_pca_diff" or "pacmap". Defaults to "pca_center"! This is different
                than ControlVector.train, which defaults to "pca_diff".
            **kwargs: Additional keyword arguments.
                max_batch_size (int, optional): The maximum batch size for training.
                    Defaults to 32. Try reducing this if you're running out of memory.
                quality_threshold (float, optional): Minimum quality score (0-1) for keeping 
                    a layer's direction. Layers below this threshold will have zero 
                    directions. Defaults to 0.6.
                quality_filter_n (int | None, optional): If set, only keep the top N highest 
                    quality layers. Other layers will be set to None. Set to None to 
                    keep all layers above quality_threshold. Defaults to 10.

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
        method: typing.Literal["pca_diff", "pca_center", "umap", "pacmap", "umap_kmeans_pca_diff", "pacmap_kmeans_pca_diff"] = "pca_center",
        **kwargs,
    ) -> "ControlVector":
        """
        Like ControlVector.train, but using an SAE. It's better!

        The only args that differ from ControlVector.train are:
            sae (saes.Sae): See the `saes` module for how to load this.
            decode (bool, optional): Whether to decode the vector to make it immediately usable.
                If not, keeps it as monosemantic SAE features for introspection, but you will need to decode it manually
                to use it. Defaults to True.

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
    use_cache: bool = True,
    method: typing.Literal["pca_diff", "pca_center", "umap", "pacmap", "umap_kmeans_pca_diff", "pacmap_kmeans_pca_diff"] = "pca_diff",
    transform_hiddens: (
        typing.Callable[[dict[int, np.ndarray]], dict[int, np.ndarray]] | None
    ) = None,
    norm_type: typing.Literal["l1", "l2", "auto"] = "auto",
    quality_threshold: float = 0.6,
    quality_filter_n: typing.Optional[int] = 10,
    ) -> dict[int, np.ndarray]:
    """
    Extract the representations based on the contrast dataset.
    """
    if not hidden_layers:
        hidden_layers = range(-1, -model.config.num_hidden_layers, -1)

    # normalize the layer indexes if they're negative
    n_layers = len(model_layer_list(model))
    hidden_layers = [i if i >= 0 else n_layers + i for i in hidden_layers]

    # the order of example is [positive valence, negative, positive, negative, ...]
    train_strs = autocorrect_chat_templates(
        messages=[s for ex in inputs for s in (ex.positive, ex.negative)],
        tokenizer=tokenizer,
        model=model,
    )

    layer_hiddens = batched_get_hiddens(
        model, tokenizer, train_strs, hidden_layers, batch_size, use_cache=use_cache
    )

    if transform_hiddens is not None:
        layer_hiddens = transform_hiddens(layer_hiddens)

    n_sample = len(train_strs)

    # get directions for each layer using PCA
    directions: dict[int, np.ndarray] = {}
    qualities: dict[int, int] = {}
    for layer in tqdm.tqdm(hidden_layers, desc="Computing direction for samples", unit="layer"):
        h = layer_hiddens[layer]
        assert h.shape[0] == len(inputs) * 2

        assert not np.isnan(h.ravel()).any(), f"the activation layer '{layer}' contains at least one nan"

        if method == "pca_diff":
            train = h[::2] - h[1::2]
        elif method == "pca_center":
            center = (h[::2] + h[1::2]) / 2
            train = h
            train[::2] -= center
            train[1::2] -= center
        elif method in ["umap", "pacmap", "umap_kmeans_pca_diff", "pacmap_kmeans_pca_diff"]:
            train = h
        else:
            raise ValueError("unknown method " + method)

        if method in ["pca_center", "pca_diff"]:
            # shape (1, n_features)
            pca_model = PCA(n_components=1, whiten=False).fit(train)
            newlayer = pca_model.components_.squeeze(axis=0)

        elif method == "umap":
            # compute pca diff too to compare
            ref_train = h[::2] - h[1::2]
            ref_pca_model = PCA(n_components=1, whiten=False).fit(ref_train)
            ref_layer = ref_pca_model.components_.squeeze(axis=0)

            # still experimental so don't want to add this as a real dependency yet
            import umap  # type: ignore

            # documentation: https://github.com/lmcinnes/umap
            umap_model = umap.UMAP(
                n_components=1,
                low_memory=True,
                random_state=42,
                transform_seed=42,
                densmap=True,
                n_jobs=1,
                n_neighbors=max(5, min(50, train.shape[0] // 4)),
                min_dist=0.3,
            )

            # method 1: use umap output directly
            newlayer = umap_model.fit_transform(train.T).squeeze()

            # # method 2: use umap and train:
            # # note: in my tests, it appears that using this second method makes
            # # the strength a lot stronger (meaning +5 makes it output
            # # gibberish) and not necessarily better
            # embedding = umap_model.fit_transform(train).squeeze()
            # newlayer = np.sum(train * embedding.reshape(-1, 1), axis=0) / np.sum(embedding)

        elif method == "pacmap":
            # compute pca diff too to compare
            ref_train = h[::2] - h[1::2]
            ref_pca_model = PCA(n_components=1, whiten=False).fit(ref_train)
            ref_layer = ref_pca_model.components_.squeeze(axis=0)

            import pacmap  # type: ignore

            # documentation: https://github.com/YingfanWang/PaCMAP
            pacmap_model = pacmap.PaCMAP(
                n_components=1,
                n_neighbors=max(10, min(50, train.shape[0] // 4)),  # defaults to 10
                MN_ratio=1,  # default 0.5
                FP_ratio=4,  # default 2
                verbose=False,
                apply_pca=True,  # wether to start by a pca or not, not the same as 'init'
            )
            newlayer = pacmap_model.fit_transform(train.T, init="pca").squeeze()

        elif method == "umap_kmeans_pca_diff" or method == "pacmap_kmeans_pca_diff":
            # compute pca diff too to compare
            ref_train = h[::2] - h[1::2]
            ref_pca_model = PCA(n_components=1, whiten=False).fit(ref_train)
            ref_layer = ref_pca_model.components_.squeeze(axis=0)

            if method == "umap_kmeans_pca_diff":
                import umap
                from sklearn.cluster import KMeans

                # First reduce to 2D with UMAP by reducing the features, not the samples
                umap_model = umap.UMAP(
                    n_components=3,
                    low_memory=True,
                    random_state=42,
                    transform_seed=42,
                    densmap=True,
                    n_jobs=1,
                    n_neighbors=max(5, min(50, train.shape[0] // 4)),
                    min_dist=0.3,
                )
                embedding = umap_model.fit_transform(train).squeeze()

            elif method == "pacmap_kmeans_pca_diff":
                import pacmap  # type: ignore
                from sklearn.cluster import KMeans

                pacmap_model = pacmap.PaCMAP(
                    n_components=3,
                    n_neighbors=max(10, min(50, train.shape[0] // 4)),  # defaults to 10
                    MN_ratio=1,  # default 0.5
                    FP_ratio=4,  # default 2
                    verbose=False,
                    apply_pca=True,  # wether to start by a pca or not, not the same as 'init'
                )
                embedding = pacmap_model.fit_transform(train, init="pca").squeeze()

            else:
                raise ValueError(method)

            # Run KMeans clustering
            kmeans = KMeans(n_clusters=2, random_state=42)
            clusters = kmeans.fit_predict(embedding)

            # Remap clusters to match positive/negative samples
            # Remember samples are ordered: positive, negative, positive, negative...
            positive_indices = np.arange(0, len(clusters), 2)
            negative_indices = np.arange(1, len(clusters), 2)

            # Count matches for current mapping vs flipped mapping
            current_matches = (clusters[positive_indices] == 1).sum() + (clusters[negative_indices] == 0).sum()
            flipped_matches = (clusters[positive_indices] == 0).sum() + (clusters[negative_indices] == 1).sum()

            # Flip cluster labels if it improves matching
            if flipped_matches > current_matches:
                clusters = 1 - clusters

            # Calculate and print agreement percentage
            best_matches = max(current_matches, flipped_matches)
            agreement_percentage = (best_matches / len(clusters)) * 100
            print(f"UMAP Clustering agreement with pos/neg labels: {agreement_percentage:.1f}%")

            # Filter clusters based on even/odd index condition
            kept = len(clusters)
            for i in range(len(clusters)):
                if (i % 2 == 0 and clusters[i] != 1) or (i % 2 == 1 and clusters[i] != 0):
                    clusters[i] -= 2  # turn 1 into -1 and 0 into -2
                    kept -= 1

            qual = kept / len(clusters)
            qualities[layer] = qual
            if qual < quality_threshold:
                print(f"Ignored layer because only {qual*100:.2f}% label matched the truth (<{quality_threshold})")
                directions[layer] = None
                continue
            else:
                print(f"Kept {qual * 100:.2f}% samples because their label matched the truth ({kept}/{len(clusters)})")

                p1_mu = np.median(h[clusters == 1, :], axis=0)
                diffs = h.copy()
                diffs[clusters == 0] -= p1_mu
                pca_model = PCA(n_components=1, whiten=False).fit(diffs)
                newlayer = pca_model.components_.squeeze(axis=0)
        else:
            raise ValueError(method)

        newlayer = newlayer.astype(np.float32)
        assert not np.isclose(np.abs(newlayer.ravel()).sum(), 0), f"Computed direction is mostly zero {newlayer}"

        # calculate sign
        projected_hiddens = project_onto_direction(h, newlayer)

        # order of examples is [positive valence, negative, positive, negative, ...]
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
            newlayer *= -1
            if VERBOSE:
                print(f"Reversed the direction of layer {layer}")

        if "ref_layer" in locals():
            import scipy
            cc = np.corrcoef(newlayer, ref_layer)[0, 1]
            spearman = scipy.stats.spearmanr(newlayer, ref_layer)[0]
            cossim = np.dot(newlayer, ref_layer) / (np.linalg.norm(newlayer) * np.linalg.norm(ref_layer))
            ang = np.arccos(cossim) * 180 / np.pi
            print(f"Comparison between {method} and pca_diff: CC={cc:.3f}  Spearman={spearman:.3f} Cosim={cossim:.3f} Angle={ang:.3f}")

        # Shapes reminder:
        # train: shape is (n_samples, n_features)
        # each direction is stored in newlayer and must be of shape (n_features,)
        newlayer = newlayer.squeeze()
        assert len(newlayer.shape) == 1 and newlayer.shape[0] == train.shape[1], f"newlayer is of shape {newlayer.shape} but should be ({train.shape[1]},)"

        assert layer not in directions, layer
        directions[layer] = newlayer

    if qualities and quality_filter_n:
        vals = sorted(qualities.values())
        vals = vals[-quality_filter_n:]
        for layer, q in qualities.items():
            if q <= quality_threshold:
                # this layer was already ignored
                continue
            if q not in vals:
                directions[layer] = None
                print(f"Removing layer {layer} after filtering")
            else:
                print(f"Kept layer {layer} after filtering")

    assert not all(
            np.isclose(np.abs(lay.ravel()).sum(), 0)
            for lay in directions.values()
            if lay is not None
    ), "All computed direction are mostly zero"
    return directions


def batched_get_hiddens(
    model,
    tokenizer,
    inputs: list[str],
    hidden_layers: list[int],
    batch_size: int,
    use_cache: bool = True,
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
            out = _model_forward(model, encoded_batch, batch, use_cache=use_cache)

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
