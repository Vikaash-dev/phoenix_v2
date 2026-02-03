import tensorflow as tf
from tensorflow.keras import layers, Model
from phoenix_mamba_v2.models.ssm import S6Layer
from phoenix_mamba_v2.models.aggregator import SliceAggregator25D
from phoenix_mamba_v2.models.attention import HeterogeneityAttention
from phoenix_mamba_v2.core.config import ModelConfig, Config

class MambaBlock(layers.Layer):
    def __init__(self, d_model, d_state=16, **kwargs):
        super(MambaBlock, self).__init__(**kwargs)
        self.d_model = int(d_model) # Ensure int
        self.d_state = int(d_state)
        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.s6 = S6Layer(self.d_model, d_state=self.d_state)

    def build(self, input_shape):
        # Explicitly build sublayers to ensure shapes are defined
        # input_shape is typically (Batch, Length, Channels)
        # We ensure Channels is fixed to d_model

        # LayerNormalization needs the shape of the axis it normalizes.
        # Default is axis=-1.
        self.norm.build((None, None, self.d_model))

        # S6Layer build
        self.s6.build((None, None, self.d_model))

        super(MambaBlock, self).build(input_shape)

    def call(self, x, return_internals=False):
        # x input shape: (B, L, C)
        # Ensure channel dimension is statically defined for safety
        x.set_shape([None, None, self.d_model])

        residual = x
        x = self.norm(x)
        if return_internals:
            x, internals = self.s6(x, return_internals=True)
            return residual + x, internals
        else:
            x = self.s6(x)
            return residual + x

class PhoenixMambaV2(Model):
    """
    PHOENIX-MAMBA v2: Hierarchical Selective State-Space Model.
    """
    def __init__(self, config: ModelConfig = None, **kwargs):
        # Handle custom args before passing to super
        num_classes_arg = kwargs.pop('num_classes', None)

        super(PhoenixMambaV2, self).__init__(**kwargs)

        if config is None:
            # Fallback to default configuration if not provided
            config = Config().model

        # Override num_classes if passed via kwargs (for backward compatibility)
        if num_classes_arg is not None:
             config.num_classes = num_classes_arg

        self.config = config

        # 1. 2.5D Aggregator
        # First stage output size determines the aggregator output size
        self.aggregator = SliceAggregator25D(filters=self.config.d_model_stages[0])

        # 2. Hierarchical Backbone
        # Stage 1
        self.stage1_conv = layers.Conv2D(self.config.d_model_stages[0], 3, strides=1, padding='same')
        self.stage1_mamba = MambaBlock(self.config.d_model_stages[0], d_state=self.config.d_state)
        self.stage1_down = layers.MaxPool2D(2)

        # Stage 2
        self.stage2_conv = layers.Conv2D(self.config.d_model_stages[1], 3, strides=1, padding='same')
        self.stage2_mamba = MambaBlock(self.config.d_model_stages[1], d_state=self.config.d_state)
        self.stage2_down = layers.MaxPool2D(2)

        # Stage 3
        self.stage3_conv = layers.Conv2D(self.config.d_model_stages[2], 3, strides=1, padding='same')
        self.stage3_mamba = MambaBlock(self.config.d_model_stages[2], d_state=self.config.d_state)
        self.stage3_down = layers.MaxPool2D(2)

        # Stage 4
        self.stage4_conv = layers.Conv2D(self.config.d_model_stages[3], 3, strides=1, padding='same')
        self.stage4_mamba = MambaBlock(self.config.d_model_stages[3], d_state=self.config.d_state)
        # No downsampling at end of stage 4 for bottleneck

        # 3. Heterogeneity Attention
        self.attn_norm = layers.LayerNormalization()
        self.attention = HeterogeneityAttention(d_model=self.config.d_model_stages[3], num_heads=self.config.num_heads_attention)

        # 4. Heads
        self.gap = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(self.config.dropout_rate)
        self.classifier = layers.Dense(self.config.num_classes, activation='softmax', name="classifier")

    def call(self, inputs, training=None, mask=None):
        # inputs: (Batch, Depth, H, W, C)

        x = self.aggregator(inputs) # (B, H, W, C_stage1)

        internals_dict = {}

        # Helper for applying 1D Mamba to 2D features
        def apply_mamba_2d(layer, tensor, stage_name):
            # tensor: (B, H, W, C)
            B_shape = tf.shape(tensor)[0]
            H_shape = tf.shape(tensor)[1]
            W_shape = tf.shape(tensor)[2]

            # Use layer.d_model as the ground truth for channels
            C_static = layer.d_model

            # Reshape to (B, H*W, C)
            # Ensure C_static is an int for reshape
            tensor_flat = tf.reshape(tensor, (B_shape, H_shape * W_shape, C_static))

            # Explicitly set shape for downstream layers
            tensor_flat.set_shape([None, None, C_static])

            # Apply Mamba block
            # We check if we are in a mode where we want internals?
            # For standard call, we don't return internals to keep signature simple.
            # But we can't easily pass a flag to call() in standard Keras fit().
            # However, for explainability, we usually call model(inputs, training=False) directly or use a custom method.
            # Here we will check a dynamic attribute or just handle it if possible.
            # But Keras call signature is strict.
            # Let's assume we use a separate method for explainability or we inspect the model properties.
            # But we can just use the standard call and if the user wants internals, they might have to access layers directly?
            # No, Functional API or Subclassing makes it hard to get intermediate outputs easily without rebuilding.
            # Let's add an optional argument to call, but standard Keras might ignore it.
            # Actually, we can return a dictionary if we are not in training mode? No, that breaks compatibility.

            # Implementation Strategy:
            # We will return normal output.
            # If the user wants internals, they should use `get_explainability_maps`.

            tensor_out = layer(tensor_flat)

            # Reshape back to (B, H, W, C)
            return tf.reshape(tensor_out, (B_shape, H_shape, W_shape, C_static))

        # Stage 1
        x = self.stage1_conv(x)
        x = apply_mamba_2d(self.stage1_mamba, x, "stage1")
        x = self.stage1_down(x)

        # Stage 2
        x = self.stage2_conv(x)
        x = apply_mamba_2d(self.stage2_mamba, x, "stage2")
        x = self.stage2_down(x)

        # Stage 3
        x = self.stage3_conv(x)
        x = apply_mamba_2d(self.stage3_mamba, x, "stage3")
        x = self.stage3_down(x)

        # Stage 4
        x = self.stage4_conv(x)
        x = apply_mamba_2d(self.stage4_mamba, x, "stage4")

        # Attention
        # Flatten for attention
        B_shape = tf.shape(x)[0]
        final_dim = self.config.d_model_stages[3]
        x_flat = tf.reshape(x, (B_shape, -1, final_dim))
        x_flat.set_shape([None, None, final_dim]) # Ensure static shape

        x_attn = self.attn_norm(x_flat)
        x_attn = self.attention(x_attn, x_attn, x_attn) # Self-attention
        x = x_flat + x_attn # Residual

        # Classifier
        x_pool = tf.reduce_mean(x, axis=1) # GAP over sequence dimension (L) -> (B, 512)

        x_drop = self.dropout(x_pool)
        logits = self.classifier(x_drop)

        return logits

    def get_explainability_maps(self, inputs):
        """
        Runs the forward pass and returns the logits along with Mamba internals for all stages.
        """
        x = self.aggregator(inputs) # (B, H, W, C_stage1)

        internals_dict = {}

        def apply_mamba_2d_with_internals(layer, tensor, stage_name):
            B_shape = tf.shape(tensor)[0]
            H_shape = tf.shape(tensor)[1]
            W_shape = tf.shape(tensor)[2]
            C_static = layer.d_model
            tensor_flat = tf.reshape(tensor, (B_shape, H_shape * W_shape, C_static))
            tensor_flat.set_shape([None, None, C_static])

            # Call with return_internals=True
            tensor_out, internals = layer(tensor_flat, return_internals=True)

            # Store internals
            # Reshape internals that are spatial
            # z: (B, L, d_mamba) -> (B, H, W, d_mamba)
            if 'z' in internals:
                d_mamba = tf.shape(internals['z'])[-1]
                internals['z'] = tf.reshape(internals['z'], (B_shape, H_shape, W_shape, d_mamba))

            internals_dict[stage_name] = internals

            return tf.reshape(tensor_out, (B_shape, H_shape, W_shape, C_static))

        # Stage 1
        x = self.stage1_conv(x)
        x = apply_mamba_2d_with_internals(self.stage1_mamba, x, "stage1")
        x = self.stage1_down(x)

        # Stage 2
        x = self.stage2_conv(x)
        x = apply_mamba_2d_with_internals(self.stage2_mamba, x, "stage2")
        x = self.stage2_down(x)

        # Stage 3
        x = self.stage3_conv(x)
        x = apply_mamba_2d_with_internals(self.stage3_mamba, x, "stage3")
        x = self.stage3_down(x)

        # Stage 4
        x = self.stage4_conv(x)
        x = apply_mamba_2d_with_internals(self.stage4_mamba, x, "stage4")

        # Rest of the network
        B_shape = tf.shape(x)[0]
        final_dim = self.config.d_model_stages[3]
        x_flat = tf.reshape(x, (B_shape, -1, final_dim))
        x_flat.set_shape([None, None, final_dim])

        x_attn = self.attn_norm(x_flat)
        x_attn = self.attention(x_attn, x_attn, x_attn)
        x = x_flat + x_attn

        x_pool = tf.reduce_mean(x, axis=1)
        x_drop = self.dropout(x_pool)
        logits = self.classifier(x_drop)

        return logits, internals_dict
