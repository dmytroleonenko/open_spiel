from typing import Iterable, Tuple, Union, Literal # Added Literal

import jax.numpy as jnp
from flax import linen as nn

from .common import ConvBlock, ModuleDef # Assuming common.py is in the same directory


def rsoftmax(x, radix, cardinality):
    # x shape: (batch_size, features)
    # features = output_channels_splat_conv * radix
    # cardinality = self.groups (groups for the main splat conv logic)
    batch = x.shape[0]
    if radix > 1:
        # This assertion should hold if features = (output_splat_channels / cardinality_groups) * radix_splits * cardinality_groups
        # i.e. features = output_splat_channels * radix
        # And output_splat_channels must be divisible by cardinality for the reshape.
        # So, features must be divisible by (radix * cardinality)
        assert x.shape[-1] % (radix * cardinality) == 0, \
            f"Input features {x.shape[-1]} not divisible by radix*cardinality ({radix*cardinality})"
        
        # channels_per_cardinality_group_output = x.shape[-1] // (radix * cardinality)
        # This is equivalent to (self.channels / cardinality)
        x = x.reshape((batch, cardinality, radix, -1)) # (B, card, R, C_out/card)
        x = x.swapaxes(1, 2) # (B, R, card, C_out/card)
        return nn.softmax(x, axis=1).reshape((batch, -1)) # Softmax over R, then flatten
    else: # radix == 1
        return nn.sigmoid(x)


class SplAtConv1d(nn.Module):
    channels: int
    kernel_size: int # Kernel size is int for 1D
    strides: int = 1
    padding: Union[str, Iterable[Tuple[int]]] = "SAME"
    groups: int = 1
    radix: int = 2
    reduction_factor: int = 4

    conv_block_cls: ModuleDef = ConvBlock

    # Match extra bias here:
    # github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/splat.py#L39
    match_reference: bool = False

    @nn.compact
    def __call__(self, x, training: bool):
        assert x.ndim == 3, f"Input to SplAtConv1d must be 3D (Batch, Length, Channels), got {x.ndim}D"
        cardinality = self.groups
        # Number of filters for the initial grouped convolution
        # Each of the `radix` splits will have `channels` output channels from the main conv.
        # So, the grouped conv needs to output `channels * radix` total filters.
        initial_conv_filters = self.channels * self.radix

        # This ConvBlock performs the initial grouped convolution + BN + ReLU
        # It splits the input channels into `groups * radix` groups.
        conv_block = self.conv_block_cls(
            n_filters=initial_conv_filters,
            kernel_size=self.kernel_size, # Uses self.kernel_size (int)
            strides=self.strides,
            groups=cardinality * self.radix, # Number of groups for the conv layer
            padding=self.padding,
            dim=1 # Specify 1D operation
        )
        
        # x shape: (Batch, Length, InputChannels)
        x = conv_block(x, training=training)
        # x shape after conv_block: (Batch, NewLength, initial_conv_filters)
        # initial_conv_filters = self.channels * self.radix

        assert x.shape[1] >= 1, f"Length dimension became < 1 after initial convolution in SplAtConv1d. Input Length: {x.shape[1]}, Strides: {self.strides}, Padding: {self.padding}"

        if self.radix > 1:
            # Split the output of the initial convolution along the channel axis into `radix` parts.
            # Each part corresponds to one branch of the split-attention mechanism.
            # x shape: (Batch, NewLength, self.channels * self.radix)
            # After split: list of `radix` tensors, each (Batch, NewLength, self.channels)
            split_tensors = jnp.split(x, self.radix, axis=-1)
            gap_input = sum(split_tensors) # Element-wise sum of splits
            # gap_input shape: (Batch, NewLength, self.channels)
        else:
            gap_input = x
            # gap_input shape: (Batch, NewLength, self.channels)

        # Global Average Pooling over the Length dimension
        # gap_input shape: (Batch, NewLength, self.channels_after_sum_or_identity)
        # For ResNeSt, the number of channels in `gap_input` is `self.channels` (output channels of SplAtConv1d)
        # if radix > 1, or `self.channels * self.radix` if radix == 1 (which means initial_conv_filters)
        # This should be self.channels if radix > 1, because each split had self.channels filters.
        # If radix == 1, x has self.channels filters.
        pooled_features = gap_input.mean((1,), keepdims=True) # Mean over axis 1 (Length)
        # pooled_features shape: (Batch, 1, self.channels)

        # pooled_features has `self.channels` channels.
        # inter_channels = max(pooled_features.shape[-1] // self.reduction_factor, 32)
        # If self.channels refers to the final output channels of the SplAtConv1D (after weighting & sum), then this is fine.
        # Let's reconsider the SplAtConv2d in pytorch: `inter_channels = channels*radix // reduction`.
        # Here, their `channels` argument is the output channels of the entire SplAtConv. So it should be:
        # inter_channels_fc = max(self.channels * self.radix // self.reduction_factor, 32) if self.radix > 1 else max(self.channels // self.reduction_factor, 32)
        # if self.radix == 0: inter_channels_fc = max(self.channels // self.reduction_factor, 32) # fallback for radix=0? radix should be >=1
        # MODIFIED: Simplify inter_channels calculation for attention FCs
        inter_channels_attention = max(self.channels * self.radix // self.reduction_factor, 32)

        # Attention mechanism: two 1x1 ConvBlocks (effectively FC layers on channels)
        attn_fc1 = self.conv_block_cls(
            n_filters=inter_channels_attention, # MODIFIED: Use consistent name
            kernel_size=1, # 1x1 Conv for 1D is kernel_size=(1,)
            strides=1,
            padding="VALID", # Or SAME, for 1x1 it does not matter much if followed by another 1x1
            groups=cardinality, # Apply FC per cardinality group
            force_conv_bias=self.match_reference,
            dim=1
        )
        # attn_fc1 output: (Batch, 1, inter_channels_attention)
        attention_squeeze = attn_fc1(pooled_features, training=training)
        # attention_squeeze is (Batch, 1, inter_channels_attention)
        # No explicit activation here in original ResNeSt before the second FC for attention, only after BN.

        # Second FC for attention logits
        # This layer outputs `self.channels * self.radix` channels because these are the weights for each of the `radix` splits,
        # and each split contributes to the final `self.channels` output features.
        # attention_logits_conv = self.conv_block_cls.conv_cls # Direct nn.Conv
        attention_logits = nn.Conv( # Use explicit nn.Conv for clarity
            features=self.channels * self.radix, # Output `channels * radix` for softmax over radix splits for each of the `channels` features
            kernel_size=(1,), # 1D convolution
            strides=(1,),     # 1D stride
            padding="VALID",
            feature_group_count=cardinality, # Groups for the conv
            use_bias=True, # Typically attention logits layer has bias
            name="attn_fc2_conv"
        )(attention_squeeze)
        # attention_logits shape: (Batch, 1, self.channels * self.radix)

        # Reshape for rsoftmax: (Batch, self.channels * self.radix)
        attention_logits_flat = attention_logits.reshape((x.shape[0], -1))
        # Apply rsoftmax (softmax over radix for each group of channels)
        attention_weights = rsoftmax(attention_logits_flat, self.radix, self.groups)
        # attention_weights shape: (Batch, self.channels * self.radix)
        
        # Reshape attention_weights to (Batch, 1, self.channels * self.radix) to multiply with splits
        attention_weights_reshaped = attention_weights.reshape((x.shape[0], 1, -1))

        if self.radix > 1:
            # Apply attention_weights to the splits
            # split_tensors is a list of `radix` tensors, each (Batch, NewLength, self.channels)
            # attention_weights_reshaped is (Batch, 1, self.channels * self.radix)
            # We need to split attention_weights_reshaped into `radix` parts, each (Batch, 1, self.channels)
            attention_weights_splits = jnp.split(attention_weights_reshaped, self.radix, axis=-1)
            
            weighted_splits = []
            for i in range(self.radix):
                weighted_splits.append(attention_weights_splits[i] * split_tensors[i])
            
            out = sum(weighted_splits) # Summing up the weighted splits
            # out shape: (Batch, NewLength, self.channels)
        else: # radix == 1, effectively a Squeeze-and-Excitation block
            # x has shape (Batch, NewLength, self.channels)
            # attention_weights_reshaped has shape (Batch, 1, self.channels)
            out = attention_weights_reshaped * x # Element-wise multiplication
            # out shape: (Batch, NewLength, self.channels)

        return out 