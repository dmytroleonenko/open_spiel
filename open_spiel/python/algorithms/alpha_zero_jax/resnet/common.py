from functools import partial
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, Union, Literal

import flax
import flax.linen as nn

ModuleDef = Callable[..., Callable]
# InitFn = Callable[[PRNGKey, Shape, DType], Array]
InitFn = Callable[[Any, Iterable[int], Any], Any]


class ConvBlock(nn.Module):
    n_filters: int
    kernel_size: Union[int, Tuple[int, int]]
    strides: Union[int, Tuple[int, int]] = 1
    dim: Literal[1, 2] = 2
    activation: Callable = nn.relu
    padding: Union[str, Iterable[Tuple[int, int]], Iterable[Tuple[int]]] = "SAME"
    is_last: bool = False
    groups: int = 1
    kernel_init: InitFn = nn.initializers.kaiming_normal()
    bias_init: InitFn = nn.initializers.zeros

    conv_cls: ModuleDef = nn.Conv
    norm_cls: Optional[ModuleDef] = partial(nn.BatchNorm, momentum=0.95)

    force_conv_bias: bool = False

    @nn.compact
    def __call__(self, x, training: bool):
        _kernel_size = (self.kernel_size,) if self.dim == 1 and isinstance(self.kernel_size, int) else self.kernel_size
        _strides = (self.strides,) if self.dim == 1 and isinstance(self.strides, int) else self.strides
        
        _padding = self.padding
        if self.dim == 1:
            if isinstance(self.padding, str):
                # 'SAME' or 'VALID' are fine as is for 1D
                _padding = self.padding
            elif isinstance(self.padding, tuple) and len(self.padding) == 1 and \
                 isinstance(self.padding[0], tuple) and len(self.padding[0]) == 2 and \
                 isinstance(self.padding[0][0], int) and isinstance(self.padding[0][1], int):
                # Already in correct format like ((low, high),)
                _padding = self.padding
            elif isinstance(self.padding, list) and len(self.padding) == 1 and \
                 isinstance(self.padding[0], tuple) and len(self.padding[0]) == 2 and \
                 isinstance(self.padding[0][0], int) and isinstance(self.padding[0][1], int):
                # Convert [(low, high)] to ((low, high),)
                _padding = tuple(self.padding)
            else:
                raise ValueError(
                    f"Unsupported padding format for 1D convolution in ConvBlock: {self.padding}. "
                    f"Expected 'SAME', 'VALID', or a sequence like ((pad_lo, pad_hi),) or [(pad_lo, pad_hi)]."
                )

        # For self.dim == 2, self.padding is assumed to be correctly formatted (string or sequence of 2 tuples)

        x = self.conv_cls(
            features=self.n_filters,
            kernel_size=_kernel_size,
            strides=_strides,
            use_bias=(not self.norm_cls or self.force_conv_bias),
            padding=_padding,
            feature_group_count=self.groups,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        if self.norm_cls:
            scale_init = (nn.initializers.zeros
                          if self.is_last else nn.initializers.ones)
            x = self.norm_cls(use_running_average=not training, scale_init=scale_init)(x)

        if not self.is_last:
            x = self.activation(x)
        return x


def slice_variables(variables: Mapping[str, Any],
                    start: int = 0,
                    end: Optional[int] = None) -> flax.core.FrozenDict:
    """Returns variables dict correspond to a sliced model.

    You can retrieve the model corresponding to the slices variables via
    `Sequential(model.layers[start:end])`.

    The variables mapping should have the same structure as a Sequential
    model's variable dict (based on Flax):

        variables = {
            'group1': ['layers_a', 'layers_b', ...]
            'group2': ['layers_a', 'layers_b', ...]
            ...,
        }

    Typically, 'group1' and 'group2' would be 'params' and 'batch_stats', but
    they don't have to be. 'a, b, ...' correspond to the integer indices of the
    layers.

    Args:
        variables: A dict (typically a flax.core.FrozenDict) containing the
            model parameters and state.
        start: integer indicating the first layer to keep.
        end: integer indicating the first layer to exclude (can be negative,
            has the same semantics as negative list indexing).

    Returns:
        A flax.core.FrozenDict with the subset of parameters/state requested.
    """
    last_ind = max(int(s.split('_')[-1]) for s in variables['params'])
    if end is None:
        end = last_ind + 1
    elif end < 0:
        end += last_ind + 1

    sliced_variables: Dict[str, Any] = {}
    for k, var_dict in variables.items():  # usually params and batch_stats
        sliced_variables[k] = {
            f'layers_{i-start}': var_dict[f'layers_{i}']
            for i in range(start, end)
            if f'layers_{i}' in var_dict
        }

    return flax.core.freeze(sliced_variables)
