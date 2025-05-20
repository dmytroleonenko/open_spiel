from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union, Literal

import jax.numpy as jnp
from flax import linen as nn
import jax
import inspect
import functools

from .common import ConvBlock, ModuleDef
from .splat import SplAtConv2d
from .splat1d import SplAtConv1d

STAGE_SIZES = {
    18: [2, 2, 2, 2],
    20: [3, 3, 3, 3],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3],
    269: [3, 30, 48, 8],
}


class ResNetStem(nn.Module):
    conv_block_cls: ModuleDef = ConvBlock
    n_hidden: int = 64
    dim: Literal[1, 2] = 2

    @nn.compact
    def __call__(self, x, training=False):
        kernel_size = (7, 7) if self.dim == 2 else (7,)
        strides = (2, 2) if self.dim == 2 else (2,)
        padding = ((3, 3), (3, 3)) if self.dim == 2 else ((3, 3),)
        return self.conv_block_cls(
            n_filters=self.n_hidden,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dim=self.dim)(x, training=training)


class ResNetDStem(nn.Module):
    conv_block_cls: ModuleDef = ConvBlock
    stem_width: int = 32
    dim: Literal[1, 2] = 2

    adaptive_first_width: bool = False

    @nn.compact
    def __call__(self, x, training=False):
        kernel_size = (3,3) if self.dim == 2 else (3,)
        padding = ((1,1),(1,1)) if self.dim == 2 else ((1,1),)

        cls = partial(self.conv_block_cls, kernel_size=kernel_size, padding=padding, dim=self.dim)
        first_width = (8 * (x.shape[-1] + 1)
                       if self.adaptive_first_width else self.stem_width)
        
        strides_conv1 = (2,2) if self.dim == 2 else (2,)
        strides_conv23 = (1,1) if self.dim == 2 else (1,)

        x = cls(n_filters=first_width, strides=strides_conv1)(x, training=training)
        x = cls(n_filters=self.stem_width, strides=strides_conv23)(x, training=training)
        x = cls(n_filters=self.stem_width * 2, strides=strides_conv23)(x, training=training)
        return x


class ResNetSkipConnection(nn.Module):
    strides: Union[Tuple[int, int], int]
    conv_block_cls: ModuleDef = ConvBlock
    dim: Literal[1, 2] = 2

    @nn.compact
    def __call__(self, x, out_shape, training: bool = False):
        if x.shape != out_shape:
            _kernel_size = (1,1) if self.dim == 2 else (1,)
            _strides = self.strides
            x = self.conv_block_cls(n_filters=out_shape[-1],
                                    kernel_size=_kernel_size,
                                    strides=_strides,
                                    activation=lambda y: y,
                                    dim=self.dim)(x, training=training)
        return x


class ResNetDSkipConnection(nn.Module):
    strides: Union[Tuple[int, int], int]
    conv_block_cls: ModuleDef = ConvBlock
    dim: Literal[1, 2] = 2

    @nn.compact
    def __call__(self, x, out_shape, training: bool = False):
        perform_pooling = False
        if self.dim == 2 and self.strides != (1,1):
            perform_pooling = True
            pool_kernel_size = (2,2)
            pool_strides = self.strides
            pool_padding = 'SAME'
        elif self.dim == 1 and self.strides != 1:
            perform_pooling = True
            pool_kernel_size = (2,)
            pool_strides = (self.strides,) if isinstance(self.strides, int) else self.strides
            pool_padding = 'SAME'

        if perform_pooling:
            x = nn.avg_pool(x, pool_kernel_size, strides=pool_strides, padding=pool_padding)

        if x.shape[-1] != out_shape[-1]:
            _kernel_size = (1,1) if self.dim == 2 else (1,)
            _strides_conv = (1,1) if self.dim == 2 else (1,)
            x = self.conv_block_cls(n_filters=out_shape[-1],
                                    kernel_size=_kernel_size,
                                    strides=_strides_conv,
                                    activation=lambda y: y,
                                    dim=self.dim)(x, training=training)
        return x


class ResNeStSkipConnection(ResNetDSkipConnection):
    pass


# Specialized 1D Skip Connection for ResNeSt1D blocks
class ResNeSt1DSkipConnection(ResNetDSkipConnection):
    dim: Literal[1] = 1 # type: ignore # Force dim to 1

    @nn.compact
    def __call__(self, x, out_shape, training: bool = False):
        # ResNetDSkipConnection is already written to use self.dim to adapt its behavior.
        return super().__call__(x, out_shape, training=training)


class ResNetBlock(nn.Module):
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)
    radix: int = 2
    groups: int = 1
    base_width: int = 64
    dim: Literal[1,2] = 2

    activation: Callable = nn.relu
    conv_block_cls: ModuleDef = ConvBlock
    splat_cls: ModuleDef = SplAtConv1d
    skip_cls: ModuleDef = ResNeSt1DSkipConnection

    @nn.compact
    def __call__(self, x, training=False):
        expected_ndim = 3 if self.dim == 1 else 4
        assert x.ndim == expected_ndim, f"Input x.ndim={x.ndim}D, inconsistent with ResNetBlock's expected {expected_ndim}D input for dim={self.dim}."
        
        current_skip_cls = partial(ResNetSkipConnection, conv_block_cls=self.conv_block_cls, dim=self.dim)
        skip_conn_instance = current_skip_cls(strides=self.strides)
        
        # Pass kernel_size as a tuple for 2D, or int for 1D (ConvBlock handles tuple conversion for 1D if int)
        conv_kernel_size = (3,3) if self.dim == 2 else 3

        y = self.conv_block_cls(self.n_hidden, kernel_size=conv_kernel_size, strides=self.strides, padding=((1,1),(1,1)) if self.dim==2 else ((1,1),), dim=self.dim, name="conv1")(x, training=training)
        y = self.conv_block_cls(self.n_hidden, kernel_size=conv_kernel_size, strides=1, padding=((1,1),(1,1)) if self.dim==2 else ((1,1),), dim=self.dim, is_last=True, name="conv2")(y, training=training)
        
        shortcut = skip_conn_instance(x, out_shape=y.shape, training=training)
        return self.activation(y + shortcut)


class ResNetBottleneckBlock(nn.Module):
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)
    expansion: int = 4
    groups: int = 1
    base_width: int = 64

    activation: Callable = nn.relu
    conv_block_cls: ModuleDef = ConvBlock
    skip_cls: ModuleDef = ResNetSkipConnection

    @nn.compact
    def __call__(self, x, training=False):
        skip_connection_constructor = partial(self.skip_cls, conv_block_cls=self.conv_block_cls)
        
        group_width = int(self.n_hidden * (self.base_width / 64.)) * self.groups

        y = self.conv_block_cls(group_width, kernel_size=(1, 1))(x, training=training)
        y = self.conv_block_cls(group_width,
                                strides=self.strides,
                                groups=self.groups,
                                padding=((1, 1), (1, 1)))(y, training=training)
        y = self.conv_block_cls(self.n_hidden * self.expansion,
                                kernel_size=(1, 1),
                                is_last=True)(y, training=training)
        shortcut = skip_connection_constructor(strides=self.strides)(x, out_shape=y.shape, training=training)
        return self.activation(y + shortcut)


class ResNetDBlock(nn.Module):
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)

    activation: Callable = nn.relu
    conv_block_cls: ModuleDef = ConvBlock
    skip_cls: ModuleDef = ResNetDSkipConnection

    @nn.compact
    def __call__(self, x, training=False):
        skip_connection_constructor = partial(self.skip_cls, conv_block_cls=self.conv_block_cls)
        
        y = self.conv_block_cls(self.n_hidden,
                                padding=[(1, 1), (1, 1)],
                                strides=self.strides)(x, training=training)
        y = self.conv_block_cls(self.n_hidden, padding=[(1, 1), (1, 1)],
                                is_last=True)(y, training=training)
        shortcut = skip_connection_constructor(strides=self.strides)(x, out_shape=y.shape, training=training)
        return self.activation(y + shortcut)


class ResNetDBottleneckBlock(nn.Module):
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)
    expansion: int = 4
    groups: int = 1
    base_width: int = 64

    activation: Callable = nn.relu
    conv_block_cls: ModuleDef = ConvBlock
    skip_cls: ModuleDef = ResNetDSkipConnection

    @nn.compact
    def __call__(self, x, training=False):
        skip_connection_constructor = partial(self.skip_cls, conv_block_cls=self.conv_block_cls)
        
        group_width = int(self.n_hidden * (self.base_width / 64.)) * self.groups

        y = self.conv_block_cls(group_width, kernel_size=(1, 1))(x, training=training)
        y = self.conv_block_cls(group_width,
                                strides=self.strides,
                                groups=self.groups,
                                padding=[(1, 1), (1, 1)])(y, training=training)
        y = self.conv_block_cls(self.n_hidden * self.expansion,
                                kernel_size=(1, 1),
                                is_last=True)(y, training=training)
        shortcut = skip_connection_constructor(strides=self.strides)(x, out_shape=y.shape, training=training)
        return self.activation(y + shortcut)


class ResNeStBottleneckBlock(ResNetBottleneckBlock):
    skip_cls: ModuleDef = ResNeStSkipConnection
    avg_pool_first: bool = False
    radix: int = 2
    splat_cls: ModuleDef = SplAtConv2d

    @nn.compact
    def __call__(self, x, training=False):
        skip_connection_constructor = partial(self.skip_cls, conv_block_cls=self.conv_block_cls)

        group_width = int(self.n_hidden * (self.base_width / 64.)) * self.groups
        y = self.conv_block_cls(group_width, kernel_size=(1, 1))(x, training=training)

        if self.strides != (1, 1) and self.avg_pool_first:
            y = nn.avg_pool(y, (3, 3), strides=self.strides, padding='SAME')

        y = self.splat_cls(group_width,
                           kernel_size=(3, 3),
                           strides=(1, 1),
                           padding=[(1, 1), (1, 1)],
                           groups=self.groups,
                           radix=self.radix,
                           conv_block_cls=self.conv_block_cls)(y, training=training)

        if self.strides != (1, 1) and not self.avg_pool_first:
            y = nn.avg_pool(y, (3, 3), strides=self.strides, padding='SAME')

        y = self.conv_block_cls(self.n_hidden * self.expansion,
                                kernel_size=(1, 1),
                                is_last=True)(y, training=training)
        
        shortcut = skip_connection_constructor(strides=self.strides)(x, out_shape=y.shape, training=training)
        return self.activation(y + shortcut)


class ResNeSt1DBottleneckBlock(nn.Module):
    n_hidden: int
    strides: int = 1
    expansion: int = 4
    groups: int = 1
    base_width: int = 64
    radix: int = 2

    activation: Callable = nn.relu
    conv_block_cls: ModuleDef = ConvBlock
    splat_cls: ModuleDef = SplAtConv1d
    skip_cls: ModuleDef = ResNeSt1DSkipConnection

    @nn.compact
    def __call__(self, x, training=False):
        assert x.ndim == 3, f"Input to ResNeSt1DBottleneckBlock must be 3D (Batch, Length, Channels), got {x.ndim}D"
        skip_connection_constructor = partial(self.skip_cls, conv_block_cls=self.conv_block_cls, dim=1)
        
        group_width = int(self.n_hidden * (self.base_width / 64.)) * self.groups

        y = self.conv_block_cls(n_filters=group_width, kernel_size=1, dim=1)(x, training=training)
        y = self.splat_cls(
            channels=group_width,
            kernel_size=3,
            strides=self.strides,
            groups=self.groups,
            radix=self.radix,
            conv_block_cls=self.conv_block_cls
        )(y, training=training)
        y = self.conv_block_cls(
            n_filters=self.n_hidden * self.expansion,
            kernel_size=1,
            is_last=True,
            dim=1
        )(y, training=training)
        
        shortcut = skip_connection_constructor(strides=self.strides)(x, out_shape=y.shape, training=training)
        return self.activation(y + shortcut)


class ResNet(nn.Module):
    stem_cls: ModuleDef
    block_cls: ModuleDef
    stage_sizes: Sequence[int]
    hidden_sizes: Sequence[int]
    output_size: int
    dim: Literal[1,2] = 2

    block_kwargs: Optional[dict] = None
    stage_strides_config: Optional[Sequence[Union[int, Tuple[int,int]]]] = None

    az_head_hidden_width: int = 256
    az_value_use_conv_head: bool = True
    az_policy_use_conv_head: bool = True

    # --- New Transformer Head Config (only active if use_transformer_head is True) ---
    use_transformer_head: bool = False
    # is_backgammon_resnet1d: bool = False # REMOVED - This logic will be in a wrapper model
    transformer_model_dim: int = 256
    transformer_num_heads: int = 4
    transformer_num_layers: int = 2
    transformer_mlp_dim: int = 512  # Added (common default: 2*model_dim or 4*model_dim)
    transformer_dropout_rate: float = 0.1  # Added
    attention_dropout_rate: float = 0.1 # Passed to SelfAttention within TransformerEncoderLayer

    @nn.compact
    def __call__(self, x: jax.Array, training: bool, legals_mask: Optional[jax.Array] = None, global_features: Optional[jax.Array] = None):
        # x is (Batch, Length, Channels) for 1D or (Batch, Height, Width, Channels) for 2D (spatial features)
        # global_features (if provided and use_transformer_head is True) is (Batch, NumGlobalFeatures)
        
        # Stem
        # Ensure stem_cls is partialled with dim or is dim-aware
        current_stem_cls = self.stem_cls
        if not isinstance(self.stem_cls, functools.partial) or 'dim' not in self.stem_cls.keywords:
            stem_instance = current_stem_cls(dim=self.dim) # Call with dim if it's a class
        else:
            stem_instance = current_stem_cls() # Call if it's a partial already configured with dim
        
        x = stem_instance(x, training=training)

        # Max pooling after stem (common in ResNets)
        if self.dim == 2:
            x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
        else:
            x = nn.max_pool(x, window_shape=(3,), strides=(2,), padding='SAME')

        current_n_hidden = self.hidden_sizes[0]
        effective_block_kwargs = self.block_kwargs if self.block_kwargs is not None else {}
        for i, num_blocks_in_stage in enumerate(self.stage_sizes):
            stage_base_channels = self.hidden_sizes[i]
            
            if self.stage_strides_config:
                current_stage_stride = self.stage_strides_config[i]
            else:
                if self.dim == 2:
                    current_stage_stride = (1,1) if i == 0 else (2,2)
                else:
                    current_stage_stride = 1 if i == 0 else 2
            
            for block_idx in range(num_blocks_in_stage):
                block_stride = current_stage_stride if block_idx == 0 else ( (1,1) if self.dim == 2 else 1 )
                
                block_params = {
                    'n_hidden': stage_base_channels,
                    'strides': block_stride,
                    **effective_block_kwargs
                }
                actual_callable_for_signature = self.block_cls
                if isinstance(self.block_cls, functools.partial):
                    actual_callable_for_signature = self.block_cls.func
                
                sig = inspect.signature(actual_callable_for_signature)
                if 'dim' in sig.parameters:
                    block_params['dim'] = self.dim
                
                current_block = self.block_cls(**block_params)
                x = current_block(x, training=training)

        if self.dim == 2:
            x_pooled = jnp.mean(x, axis=(1, 2))
        else:
            x_pooled = jnp.mean(x, axis=1)

        # --- AZ Heads: Potentially with Transformer --- 
        final_features_for_heads = x_pooled

        if self.use_transformer_head:
            if global_features is None:
                raise ValueError("ResNet: use_transformer_head is True, but global_features were not provided. This may lead to incorrect behavior (e.g., missing dice/bar info for Backgammon).")
            
            current_transformer_input_features = x_pooled # Default to pooled spatial features
            if global_features is not None: # This check is now slightly redundant due to the above raise, but kept for explicitness
                # Concatenate pooled spatial features with raw global features if available
                fused_features = jnp.concatenate([x_pooled, global_features], axis=-1)
                current_transformer_input_features = fused_features
            # else: if global_features is None, current_transformer_input_features remains x_pooled
            
            # Project to Transformer model dimension
            projected_to_transformer_dim = nn.Dense(
                features=self.transformer_model_dim, 
                name="transformer_projection"
            )(current_transformer_input_features) # Use current_transformer_input_features
            
            # Reshape for Transformer: (B, SeqLen=1, transformer_model_dim)
            transformer_input = jnp.expand_dims(projected_to_transformer_dim, axis=1)

            # Transformer Encoder
            transformer_output = TransformerEncoder(
                num_layers=self.transformer_num_layers,
                model_dim=self.transformer_model_dim,
                num_heads=self.transformer_num_heads,
                mlp_dim=self.transformer_mlp_dim,
                dropout_rate=self.transformer_dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                name="transformer_encoder"
            )(transformer_input, training=training) # No padding_mask needed for SeqLen=1

            final_features_for_heads = jnp.squeeze(transformer_output, axis=1)

        # --- Policy and Value Heads (now primarily MLP on final_features_for_heads) ---
        # If Transformer was used, final_features_for_heads is its output.
        # If not, final_features_for_heads is x_pooled.
        # The conv-based heads are generally not compatible with Transformer output unless x (pre-pool) is used.
        # For simplicity, if use_transformer_head is True, we use MLP heads on its output.
        # If use_transformer_head is False, the original logic for az_policy/value_use_conv_head applies to x (pre-pool) or x_pooled.

        if self.use_transformer_head: # Simplified MLP heads if transformer is used
            policy_hidden = nn.Dense(features=self.az_head_hidden_width, name="policy_head_hidden")(final_features_for_heads)
            policy_hidden_relu = nn.relu(policy_hidden)
            policy_logits = nn.Dense(features=self.output_size, name="policy_head_dense")(policy_hidden_relu)

            value_hidden = nn.Dense(features=self.az_head_hidden_width, name="value_head_hidden")(final_features_for_heads)
            value_hidden_relu = nn.relu(value_hidden)
            value_output = nn.Dense(features=1, name="value_head_output")(value_hidden_relu)
            value_output = jnp.tanh(value_output)
        else: # Original head logic (conv or MLP from x_pooled) when transformer is not used
            # Policy Head
            if self.az_policy_use_conv_head:
                ph_channels = 2 
                ph_conv_kernel = (1,1) if self.dim == 2 else (1,)
                # IMPORTANT: conv head operates on `x` (pre-pooling), not `final_features_for_heads` (which is x_pooled here)
                ph = ConvBlock(n_filters=ph_channels, kernel_size=ph_conv_kernel, strides=1, padding='SAME', dim=self.dim, name="policy_head_conv_block")(x, training=training)
                ph_flat = ph.reshape((ph.shape[0], -1))
                policy_logits = nn.Dense(features=self.output_size, name="policy_head_dense")(ph_flat)
            else: 
                policy_hidden = nn.Dense(features=self.az_head_hidden_width, name="policy_head_hidden")(final_features_for_heads) # final_features_for_heads is x_pooled here
                policy_hidden_relu = nn.relu(policy_hidden)
                policy_logits = nn.Dense(features=self.output_size, name="policy_head_dense")(policy_hidden_relu)

            # Value Head
            if self.az_value_use_conv_head:
                vh_channels = 1 
                vh_conv_kernel = (1,1) if self.dim == 2 else (1,)
                # IMPORTANT: conv head operates on `x` (pre-pooling)
                vh_conv = ConvBlock(n_filters=vh_channels, kernel_size=vh_conv_kernel, strides=1, padding='SAME', dim=self.dim, name="value_head_conv_block")(x, training=training)
                vh_flat = vh_conv.reshape((vh_conv.shape[0], -1))
                vh_dense1 = nn.Dense(features=self.az_head_hidden_width, name="value_head_dense1")(vh_flat)
                vh_relu = nn.relu(vh_dense1)
                value_output = nn.Dense(features=1, name="value_head_dense2")(vh_relu)
            else: 
                value_hidden = nn.Dense(features=self.az_head_hidden_width, name="value_head_hidden")(final_features_for_heads) # final_features_for_heads is x_pooled here
                value_hidden_relu = nn.relu(value_hidden)
                value_output = nn.Dense(features=1, name="value_head_dense2")(value_hidden_relu)
            
            value_output = jnp.tanh(value_output)

        if legals_mask is not None:
            policy_logits = jnp.where(legals_mask, policy_logits, -jnp.inf)

        return policy_logits, value_output


ResNeSt1D50_AZ = partial(
    ResNet,
    stem_cls=partial(ResNetDStem, stem_width=32, dim=1),
    block_cls=ResNeSt1DBottleneckBlock,
    stage_sizes=STAGE_SIZES[50],
    hidden_sizes=(64, 128, 256, 512),
    dim=1,
    block_kwargs={"expansion": 4, "groups": 1, "radix": 2, "base_width": 64},
    stage_strides_config = (1, 2, 2, 1),
    # --- Args for Transformer Head (active if use_transformer_head = True) ---
    use_transformer_head=True, 
    transformer_model_dim=256,
    transformer_num_heads=4,
    transformer_num_layers=2, # Explicitly set
    transformer_mlp_dim=512,  # Explicitly set (2 * model_dim)
    transformer_dropout_rate=0.1, # Explicitly set
    attention_dropout_rate=0.1 # Explicitly set
)

ResNet50_AZ_2D = partial(
    ResNet,
    stem_cls=partial(ResNetStem, dim=2),
    block_cls=partial(ResNetBottleneckBlock, conv_block_cls=ConvBlock, skip_cls=ResNetSkipConnection, dim=2),
    stage_sizes=STAGE_SIZES[50],
    hidden_sizes=(64, 128, 256, 512),
    dim=2,
    block_kwargs={"expansion": 4, "groups": 1, "base_width": 64},
    stage_strides_config = ((1,1), (2,2), (2,2), (2,2)),
)

# yapf: disable
# ResNet18 = partial(ResNet, stage_sizes=STAGE_SIZES[18],
# stem_cls=ResNetStem, block_cls=ResNetBlock)
# ... (many other old partials for nn.Sequential based ResNets) ...
# yapf: enable

# --- Transformer Components --- START ---
class TransformerEncoderLayer(nn.Module):
    model_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, *, training: bool, padding_mask: Optional[jax.Array] = None):
        # x shape: (batch_size, seq_len, model_dim)
        # padding_mask shape: (batch_size, seq_len) - True where padded

        # Self-attention sublayer
        attn_output = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.model_dim,
            out_features=self.model_dim,
            dropout_rate=self.attention_dropout_rate,
            deterministic=not training,
            name="self_attention"
        )(x, mask=padding_mask) # SelfAttention uses mask where True indicates non-padding
        
        x = nn.LayerNorm(name="layernorm_1")(x + attn_output)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # Feed-forward sublayer (MLP)
        mlp_output = nn.Dense(features=self.mlp_dim, name="mlp_dense_1")(x)
        mlp_output = nn.relu(mlp_output)
        mlp_output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(mlp_output) # Dropout after activation too
        mlp_output = nn.Dense(features=self.model_dim, name="mlp_dense_2")(mlp_output)
        
        x = nn.LayerNorm(name="layernorm_2")(x + mlp_output)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        return x

class TransformerEncoder(nn.Module):
    num_layers: int
    model_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, *, training: bool, padding_mask: Optional[jax.Array] = None):
        # x shape: (batch_size, seq_len, model_dim)
        # padding_mask shape: (batch_size, seq_len)
        
        for i in range(self.num_layers):
            x = TransformerEncoderLayer(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                name=f"encoder_layer_{i}"
            )(x, training=training, padding_mask=padding_mask)
            
        return x
# --- Transformer Components --- END ---
