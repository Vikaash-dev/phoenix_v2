import tensorflow as tf
from tensorflow.keras import layers, models
from src.models.context_aggregator import SliceContextAggregator
from src.models.scout import PriorityScout
from src.models.kan_ssm import LiquidKANS6Cell
from src.models.spectral_snake import MultiSpectralConcordanceGating
from src.models.spatial_mixer import SpatialMixer

class DynamicReshape(layers.Layer):
    """
    Reshapes tensors dynamically based on runtime shape.
    Used to switch between Spatial (H, W, C) and Sequence (L, C) formats
    without hardcoding dimensions.
    """
    def __init__(self, mode='flatten', **kwargs):
        super(DynamicReshape, self).__init__(**kwargs)
        self.mode = mode # 'flatten' or 'spatial'

    def call(self, x, spatial_shape=None):
        shape = tf.shape(x)
        batch_size = shape[0]

        if self.mode == 'flatten':
            # (B, H, W, C) -> (B, H*W, C)
            h = shape[1]
            w = shape[2]
            c = shape[3]
            return tf.reshape(x, [batch_size, h * w, c])

        elif self.mode == 'spatial':
            # (B, L, C) -> (B, H, W, C)
            # Requires spatial_shape to be passed or inferred
            if spatial_shape is None:
                raise ValueError("spatial_shape must be provided for spatial restore")
            h, w = spatial_shape
            c = shape[2]
            return tf.reshape(x, [batch_size, h, w, c])

    def compute_output_shape(self, input_shape):
        if self.mode == 'flatten':
            # We can't know H*W static shape usually if H,W are None
            return (input_shape[0], None, input_shape[-1])
        return (input_shape[0], None, None, input_shape[-1])

def residual_conv_block(x, filters, stride=1, name="res_block"):
    """
    Standard ResNet-style Convolutional Block.
    Used for high-resolution stages to save memory compared to SSM.
    """
    shortcut = x

    # Pre-activation BN? Or Post? Using standard Post-activation here
    x = layers.Conv2D(filters, 3, strides=stride, padding='same', name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation('silu', name=f"{name}_act1")(x)

    x = layers.Conv2D(filters, 3, strides=1, padding='same', name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)

    # Projection shortcut if dimensions change
    if stride > 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', name=f"{name}_shortcut")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_shortcut_bn")(shortcut)

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.Activation('silu', name=f"{name}_out_act")(x)
    return x

def create_phoenix_v3_1(input_shape=(3, None, None, 3), num_classes=4):
    """
    PHOENIX-v3.1 'Hybrid Pyramid' Assembly.

    OPTIMIZED for Production/Clinical Deployment.

    Changes from v3.0:
    1. Hybrid Architecture: Uses Conv2D for Stages 1 & 2 (High Res) to prevent SSM OOM.
       Uses Liquid-SSM for Stages 3 & 4 (Low Res) for global context.
    2. Dynamic Shapes: Supports variable input resolutions.
    3. Parameter Efficiency: Maintains < 5M params while improving throughput.
    """
    # Note: input_shape uses None for H/W to support dynamic shapes
    inputs = layers.Input(shape=input_shape, name="mri_2.5d_input")

    # --- Phase 1: Volumetric Context (Preserved) ---
    x = SliceContextAggregator(embed_dim=64, name="volumetric_context")(inputs)

    # --- Phase 2: Priority Scout (Preserved) ---
    priority_map = PriorityScout(name="diagnostic_scout")(x)

    # --- Phase 3: Hybrid Backbone ---
    # Channels: [32, 64, 128, 256]

    # === Stage 1: Conv2D (High Res) ===
    # Resolution: Input (e.g. 224x224)
    # SSM here causes OOM (L=50k). Conv2D is O(L).
    x = layers.Conv2D(32, 1, padding='same', name="stage1_proj")(x)
    x = residual_conv_block(x, 32, stride=1, name="stage1_block")

    # === Stage 2: Conv2D (Med Res) ===
    # Resolution: Input/2 (e.g. 112x112)
    # L=12.5k. SSM is possible but Conv is faster/lighter here.
    x = residual_conv_block(x, 64, stride=2, name="stage2_block")

    # Downsample priority map for later stages
    p2 = layers.AveragePooling2D(pool_size=2, strides=2, name="priority_down_2")(priority_map)

    # === Stage 3: Liquid-SSM (Low Res) ===
    # Resolution: Input/4 (e.g. 56x56)
    # L=3136. SSM is very efficient here and captures global context.

    # Downsample features
    x = layers.Conv2D(128, 3, strides=2, padding='same', name="stage3_downsample")(x)
    # Downsample priority
    p3 = layers.AveragePooling2D(pool_size=2, strides=2, name="priority_down_3")(p2)

    # [TOPOLOGY FIX] Mix spatial context before flattening
    # Prevents "Zombie Topology" - enriches each pixel with 3×3 neighborhood info
    # Before: Each token = isolated pixel
    # After: Each token = pixel + 8 adjacent neighbors
    x = SpatialMixer(128, name="stage3_spatial_mix")(x)

    # Save spatial reference for shape restoration later
    x_spatial_s3 = x

    # Dynamic Reshape for SSM (Keras 3 compatible)
    # Flatten: (B, H, W, C) -> (B, H*W, C)
    x_flat = DynamicReshape(mode='flatten', name="stage3_flatten")(x)
    p_flat = DynamicReshape(mode='flatten', name="stage3_p_flatten")(p3)

    # Liquid KAN S6 Cell
    x_flat = LiquidKANS6Cell(
        d_model=128,
        d_state=16,
        name="stage3_liquid_mamba"
    )(x_flat, priority_map=p_flat)

    # Restore Spatial: (B, L, C) -> (B, H, W, C)
    # Use Lambda to compute spatial shape dynamically at runtime from saved reference
    x = layers.Lambda(
        lambda inputs: tf.reshape(
            inputs[0],  # x_flat to reshape
            [tf.shape(inputs[1])[0],  # batch from spatial reference
             tf.shape(inputs[1])[1],  # height from spatial reference
             tf.shape(inputs[1])[2],  # width from spatial reference
             128]  # channels
        ),
        name="stage3_restore"
    )([x_flat, x_spatial_s3])

    # Spectral Gating (Efficient at this resolution)
    # Note: SpectralSnake needs h_param/w_param. For dynamic, we might need adjustments.
    # The original layer creates weights based on init params.
    # If we want fully dynamic, we'd need to resize weights dynamically in call().
    # The current implementation does resize weights in call(), so it SHOULD support dynamic inputs
    # as long as we pass reasonable defaults for weight creation.
    x = MultiSpectralConcordanceGating(
        channels=128,
        h_param=56, # Nominal size for weight init
        w_param=56,
        name="stage3_spectral_gate"
    )(x)

    x = layers.BatchNormalization(name="stage3_bn")(x)

    # === Stage 4: Liquid-SSM (Tiny Res) ===
    # Resolution: Input/8 (e.g. 28x28)
    # L=784. Extremely fast.

    x = layers.Conv2D(256, 3, strides=2, padding='same', name="stage4_downsample")(x)
    p4 = layers.AveragePooling2D(pool_size=2, strides=2, name="priority_down_4")(p3)

    # [TOPOLOGY FIX] Mix spatial context before flattening
    # Stage 4 operates at 28×28, even smaller sequence length (L=784)
    # SpatialMixer ensures semantic tokens, not isolated pixels
    x = SpatialMixer(256, name="stage4_spatial_mix")(x)

    # Save spatial reference for shape restoration
    x_spatial_s4 = x

    # Dynamic Reshape for SSM (Keras 3 compatible)
    x_flat = DynamicReshape(mode='flatten', name="stage4_flatten")(x)
    p_flat = DynamicReshape(mode='flatten', name="stage4_p_flatten")(p4)

    x_flat = LiquidKANS6Cell(
        d_model=256,
        d_state=16,
        name="stage4_liquid_mamba"
    )(x_flat, priority_map=p_flat)

    # Restore Spatial using Lambda layer
    x = layers.Lambda(
        lambda inputs: tf.reshape(
            inputs[0],
            [tf.shape(inputs[1])[0],  # batch
             tf.shape(inputs[1])[1],  # height
             tf.shape(inputs[1])[2],  # width
             256]
        ),
        name="stage4_restore"
    )([x_flat, x_spatial_s4])

    x = MultiSpectralConcordanceGating(
        channels=256,
        h_param=28,
        w_param=28,
        name="stage4_spectral_gate"
    )(x)

    x = layers.BatchNormalization(name="stage4_bn")(x)

    # --- Phase 4: Classification Head ---
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="uncertainty_dropout")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="tumor_grading_head")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="PHOENIX_v3_1_Hybrid_Pyramid")
    return model
