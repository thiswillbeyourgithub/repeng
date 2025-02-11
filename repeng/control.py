import dataclasses
import typing
import warnings

import torch
from transformers import PretrainedConfig, PreTrainedModel

if typing.TYPE_CHECKING:
    from .extract import ControlVector


class ControlModel(torch.nn.Module):
    """
    **This mutates the wrapped `model`! Be careful using `model` after passing it to this class.**

    A wrapped language model that can have controls set on its layers with `self.set_control`.
    """

    def __init__(
            self,
            model: PreTrainedModel,
            layer_ids: typing.Union[typing.Iterable[int], typing.Literal['all', 'middle_third', 'middle_slice'], str] = "middle_third",
    ):
        """
        **This mutates the wrapped `model`! Be careful using `model` after passing it to this class.**

        Build a new ControlModel around a model instance, initializing control on
        the layers specified in `layer_ids`.

        `layer_ids` can be 'all' to  use all layers ( not recommended as it
        often corrupts the tokens), 'middle_third' to use the middle third layers
        (the default), 'middle_slice' to only use a single layer at the middle,
        and also be a string in the format "start-end" where start and end
        are floats between 0 and 1, indicating the percentage range of layers
        to select for example "0.33-0.66" is equivalent to middle_third.
        """
        super().__init__()
        self.model = model
        num_layers = model.config.num_hidden_layers

        assert num_layers > 2, f"Suspiciously low number of layers: {num_layers}"

        if not layer_ids or layer_ids == "all":
            layer_ids = range(-1, -num_layers, -1)
        elif layer_ids == "middle_third":  # keep only the middle third
            layer_ids = [li for li in range(-1, -num_layers, -1)]
            third = len(layer_ids)//3
            layer_ids = layer_ids[third:-third]
        elif layer_ids == "middle_slice":  # keep only the middle layer
            layer_ids = [li for li in range(-1, -num_layers, -1)]
            layer_ids = [layer_ids[len(layer_ids)//2]]
        elif isinstance(layer_ids, str) and '-' in layer_ids:
            start, end = map(float, layer_ids.split('-'))
            if not (0 <= start < end <= 1):
                raise ValueError("Invalid percentage range. Must be 0 <= start < end <= 1")
            start_idx = max(0, min(num_layers - 1, int(start * num_layers)))
            end_idx = max(0, min(num_layers - 1, int(end * num_layers)))
            if start_idx == end_idx:
                layer_ids = [start_idx]
            else:
                layer_ids = list(range(start_idx, end_idx + 1))
            if not layer_ids:
                raise ValueError("The specified range doesn't include any layers")
        else:
            assert isinstance(layer_ids, list) and all(isinstance(item, int) for item in layer_ids), "unexpected value for layer_ids"

        layers = model_layer_list(model)
        self.layer_ids = [i if i >= 0 else len(layers) + i for i in layer_ids]

        for layer_id in self.layer_ids:
            layer = layers[layer_id]
            if not isinstance(layer, ControlModule):
                layers[layer_id] = ControlModule(layer)
            else:
                warnings.warn(
                    "Trying to rewrap a wrapped model! Probably not what you want! Try calling .unwrap first."
                )

        # needed by some abstractions like lm_eval
        self.name_or_path = str(model.name_or_path) + "_repeng"
        self.tie_weights = model.model.tie_weights

    @property
    def config(self) -> PretrainedConfig:
        return self.model.config

    @property
    def device(self) -> torch.device:
        return self.model.device

    def unwrap(self) -> PreTrainedModel:
        """
        Removes the mutations done to the wrapped model and returns it.
        After using this method, `set_control` and `reset` will not work.
        """

        layers = model_layer_list(self.model)
        for layer_id in self.layer_ids:
            layers[layer_id] = layers[layer_id].block
        return self.model

    def set_control(
        self, control: "ControlVector", coeff: float = 1.0, **kwargs
    ) -> None:
        """
        Set a `ControlVector` for the layers this ControlModel handles, with a strength given
        by `coeff`. (Negative `coeff` values invert the control vector, e.g. happiness→sadness.)
        `coeff` defaults to `1.0`.

        Additional kwargs:
        - `normalize: bool`: track the magnitude of the non-modified activation, and rescale the
          activation to that magnitude after control (default: `False`)
        - `operator: Union[str, Callable[[Tensor, Tensor], Tensor]]`: how to combine the base output and control
          (default: +)
        """

        raw_control = {}
        for layer_id in self.layer_ids:
            if layer_id not in control.directions:
                print(f"Missing control layer with id '{layer_id}', skipping it.")
                continue
            if control.directions[layer_id] is None:
                if VERBOSE:
                    print(f"Skipped layer {layer_id} because None")
            raw_control[layer_id] = torch.tensor(
                coeff * control.directions[layer_id]
            ).to(self.model.device, dtype=self.model.dtype)
        self.set_raw_control(raw_control, **kwargs)

    def reset(self) -> None:
        """
        Resets the control for all layer_ids, returning the model to base behavior.
        """
        self.set_raw_control(None)

    def set_raw_control(
        self, control: dict[int, torch.Tensor] | None, **kwargs
    ) -> None:
        """
        Set or remove control parameters to the layers this ControlModel handles.
        The keys of `control` should be equal to or a superset of the `layer_ids` passed to __init__.
        Only those layers will be controlled, any others in `control` will be ignored.

        Passing `control=None` will reset the control tensor for all layer_ids, making the model act
        like a non-control model.

        Additional kwargs:
        - `normalize: bool`: track the magnitude of the non-modified activation, and rescale the
          activation to that magnitude after control (default: `True`)
        - `operator: Union[str, Callable[[Tensor, Tensor], Tensor]]`: how to combine the base output and control
          (default: +)
        """

        layers = model_layer_list(self.model)
        for layer_id in self.layer_ids:
            layer: ControlModule = layers[layer_id]  # type: ignore
            if control is None:
                layer.reset()
            elif layer_id not in control:
                print(f"Missing control layer with id '{layer_id}', skipping it.")
            else:
                layer.set_control(BlockControlParams(control[layer_id], **kwargs))

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


@dataclasses.dataclass
class BlockControlParams:
    control: torch.Tensor | None = None
    normalize: bool = True
    operator: typing.Union[str, typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "+"

    @classmethod
    def default(cls) -> "BlockControlParams":
        return cls()


class ControlModule(torch.nn.Module):
    def __init__(self, block: torch.nn.Module) -> None:
        super().__init__()
        self.block: torch.nn.Module = block
        self.params: BlockControlParams = BlockControlParams.default()

    def set_control(self, params: BlockControlParams) -> None:
        self.params = params

    def reset(self) -> None:
        self.set_control(BlockControlParams.default())

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)

        control = self.params.control

        if control is None:
            return output
        elif len(control.shape) == 1:
            control = control.reshape(1, 1, -1)

        if isinstance(output, tuple):
            modified = output[0]
        else:
            modified = output

        assert len(control.shape) == len(modified.shape)
        control = control.to(modified.device)

        if self.params.normalize:
            norm_pre = torch.norm(modified, dim=-1, keepdim=True)

        # we should ignore the padding tokens when doing the activation addition
        # mask has ones for non padding tokens and zeros at padding tokens.
        # only tested this on left padding
        if "position_ids" in kwargs:
            pos = kwargs["position_ids"]
            zero_indices = (pos == 0).cumsum(1).argmax(1, keepdim=True)
            col_indices = torch.arange(pos.size(1), device=pos.device).unsqueeze(0)
            target_shape = modified.shape
            mask = (
                (col_indices >= zero_indices)
                .float()
                .reshape(target_shape[0], target_shape[1], 1)
            )
            mask = mask.to(modified.dtype).to(modified.device)

            if isinstance(self.params.operator, str) and self.params.operator == "+":
                modified = modified + control * mask
            else:
                modified = self.params.operator(modified, control * mask)
        else:
            if isinstance(self.params.operator, str) and self.params.operator == "+":
                modified = modified + control
            else:
                modified = self.params.operator(modified, control)


        if self.params.normalize:
            norm_post = torch.norm(modified, dim=-1, keepdim=True)
            modified = modified / norm_post * norm_pre

        if isinstance(output, tuple):
            output = (modified,) + output[1:]
        else:
            output = modified

        return output


def model_layer_list(model: ControlModel | PreTrainedModel) -> torch.nn.ModuleList:
    if isinstance(model, ControlModel):
        model = model.model

    assert not (hasattr(model, "model") and hasattr(model, "transformer")), "ambiguous model, not sure where to find the layers"
    if hasattr(model, "model"):  # mistral-like
        return model.model.layers
    elif hasattr(model, "transformer"):  # gpt-2-like
        return model.transformer.h
    else:
        raise ValueError(f"don't know how to get layer list for {type(model)}")
