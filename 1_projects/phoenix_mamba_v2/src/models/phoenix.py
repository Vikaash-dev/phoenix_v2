import tensorflow as tf
from tensorflow.keras import layers, Model
from src.models.ssm import S6Layer
from src.models.aggregator import SliceAggregator25D
from src.models.attention import HeterogeneityAttention

class MambaBlock(layers.Layer):
    def __init__(self, d_model, d_state=16, **kwargs):
        super(MambaBlock, self).__init__(**kwargs)
        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.s6 = S6Layer(d_model, d_state=d_state)
        # Skip connection handling is inside S6 for D, but usually there's a residual block structure
        # Block: x + S6(Norm(x))
        
    def call(self, x):
        residual = x
        x = self.norm(x)
        x = self.s6(x)
        return residual + x

class PhoenixMambaV2(Model):
    """
    PHOENIX-MAMBA v2: Hierarchical Selective State-Space Model.
    """
    def __init__(self, num_classes=4, **kwargs):
        super(PhoenixMambaV2, self).__init__(**kwargs)
        
        # 1. 2.5D Aggregator
        self.aggregator = SliceAggregator25D(filters=64)
        
        # 2. Hierarchical Backbone
        # Stage 1
        self.stage1_conv = layers.Conv2D(64, 3, strides=1, padding='same')
        self.stage1_mamba = MambaBlock(64)
        self.stage1_down = layers.MaxPool2D(2)
        
        # Stage 2
        self.stage2_conv = layers.Conv2D(128, 3, strides=1, padding='same')
        self.stage2_mamba = MambaBlock(128)
        self.stage2_down = layers.MaxPool2D(2)
        
        # Stage 3
        self.stage3_conv = layers.Conv2D(256, 3, strides=1, padding='same')
        self.stage3_mamba = MambaBlock(256)
        self.stage3_down = layers.MaxPool2D(2)
        
        # Stage 4
        self.stage4_conv = layers.Conv2D(512, 3, strides=1, padding='same')
        self.stage4_mamba = MambaBlock(512)
        # No downsampling at end of stage 4 for bottleneck
        
        # 3. Heterogeneity Attention
        self.attn_norm = layers.LayerNormalization()
        self.attention = HeterogeneityAttention(d_model=512, num_heads=4) # Adjusted to 4 to divide 512
        
        # 4. Heads
        self.gap = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(0.2)
        self.classifier = layers.Dense(num_classes, activation='softmax', name="classifier")
        
        # Optional: Segmentation Decoder could be added here (U-Net style)
        # For description "Brain tumor detection", classification/detection is primary.
        # Spec mentions "Segmentation/classification". Let's assume classification for "Detection".
        
    def call(self, inputs):
        # inputs: (Batch, 3, 224, 224, 3)
        
        x = self.aggregator(inputs) # (B, 224, 224, 64)
        
        # Stage 1
        x = self.stage1_conv(x)
        # S6 requires sequence input (B, L, C)
        # Flatten spatial for Mamba? Or scan over H, W?
        # Standard Vision Mamba flattens or scans 2D.
        # Let's flatten (H*W) -> L
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        def apply_mamba_2d(layer, tensor, h, w):
            # Reshape to (B, H*W, C)
            tensor_flat = tf.reshape(tensor, (B, h * w, -1))
            tensor_out = layer(tensor_flat)
            # Reshape back
            return tf.reshape(tensor_out, (B, h, w, -1))

        x = apply_mamba_2d(self.stage1_mamba, x, H, W)
        x = self.stage1_down(x)
        
        # Stage 2
        x = self.stage2_conv(x)
        B, H, W = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        x = apply_mamba_2d(self.stage2_mamba, x, H, W)
        x = self.stage2_down(x)
        
        # Stage 3
        x = self.stage3_conv(x)
        B, H, W = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        x = apply_mamba_2d(self.stage3_mamba, x, H, W)
        x = self.stage3_down(x)
        
        # Stage 4
        x = self.stage4_conv(x)
        B, H, W = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        x = apply_mamba_2d(self.stage4_mamba, x, H, W)
        
        # Attention
        # Flatten for attention
        x_flat = tf.reshape(x, (B, -1, 512))
        x_attn = self.attn_norm(x_flat)
        x_attn = self.attention(x_attn, x_attn, x_attn) # Self-attention
        x = x_flat + x_attn # Residual
        
        # Classifier
        # Reshape back to 2D for GAP or just use sequence
        x_2d = tf.reshape(x, (B, H, W, 512))
        x_pool = self.gap(x_2d)
        x_drop = self.dropout(x_pool)
        logits = self.classifier(x_drop)
        
        return logits
