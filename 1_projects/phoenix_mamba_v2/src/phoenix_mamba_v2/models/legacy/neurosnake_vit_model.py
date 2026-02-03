"""
NeuroSnake-ViT: A hybrid architecture combining the geometric adaptability of
Dynamic Snake Convolutions with the global context awareness of Vision Transformers.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from v1.models.dynamic_snake_conv import DynamicSnakeConv2D, SnakeConvBlock
from v1.models.neurosnake_model import MobileViTBlock
import v1.config as config

class NeuroSnakeViTModel:
    """
    NeuroSnake-ViT: Hybrid architecture for brain tumor detection.
    """

    @staticmethod
    def create_model(
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
        num_classes=config.NUM_CLASSES,
        num_transformer_blocks=4,
        num_heads=8,
        mlp_dim=512,
        dropout_rate=0.3,
    ):
        """
        Create NeuroSnake-ViT model.

        Args:
            input_shape: Input image shape
            num_classes: Number of output classes
            num_transformer_blocks: Number of transformer blocks in the ViT
            num_heads: Number of attention heads in the ViT
            mlp_dim: Dimension of the MLP in the ViT
            dropout_rate: Dropout rate for regularization

        Returns:
            Keras model
        """
        inputs = layers.Input(shape=input_shape, name="input")

        # Stage 1: Initial processing with standard conv
        x = layers.Conv2D(32, 3, strides=2, padding="same", name="stem_conv")(inputs)
        x = layers.BatchNormalization(name="stem_bn")(x)
        x = layers.Activation("relu", name="stem_act")(x)

        # Stage 2: Snake Conv Block 1
        x = SnakeConvBlock(
            64, kernel_size=3, strides=1, dropout_rate=dropout_rate, name="snake_block1"
        )(x)
        x = layers.MaxPooling2D(2, name="pool1")(x)

        # Stage 3: Snake Conv Block 2
        x = SnakeConvBlock(
            128,
            kernel_size=3,
            strides=1,
            dropout_rate=dropout_rate,
            name="snake_block2",
        )(x)
        x = layers.MaxPooling2D(2, name="pool2")(x)

        # Stage 4: Vision Transformer
        # Reshape for ViT
        patch_size = 2
        num_patches = (x.shape[1] // patch_size) * (x.shape[2] // patch_size)
        x = layers.Reshape((num_patches, x.shape[3] * patch_size * patch_size))(x)

        # Transformer blocks
        for i in range(num_transformer_blocks):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(x)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=x.shape[-1] // num_heads, dropout=dropout_rate
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, x])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = layers.Dense(mlp_dim, activation=tf.nn.gelu)(x3)
            x3 = layers.Dropout(dropout_rate)(x3)
            x3 = layers.Dense(x.shape[-1])(x3)
            x3 = layers.Dropout(dropout_rate)(x3)
            # Skip connection 2.
            x = layers.Add()([x3, x2])

        # Reshape back to image format
        x = layers.Reshape((int(tf.sqrt(float(num_patches))), int(tf.sqrt(float(num_patches))), x.shape[-1]))(x)


        # Global pooling
        x = layers.GlobalAveragePooling2D(name="global_pool")(x)

        # Classification head
        x = layers.Dense(256, activation="relu", name="fc1")(x)
        x = layers.BatchNormalization(name="fc_bn1")(x)
        x = layers.Dropout(0.5, name="fc_dropout1")(x)

        x = layers.Dense(128, activation="relu", name="fc2")(x)
        x = layers.BatchNormalization(name="fc_bn2")(x)
        x = layers.Dropout(0.5, name="fc_dropout2")(x)

        # Output layer
        outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name="NeuroSnakeViT")

        return model

if __name__ == "__main__":
    print("=" * 80)
    print("NeuroSnake-ViT Model Architecture")
    print("=" * 80)

    # Create NeuroSnake-ViT model
    print("\n1. Creating NeuroSnake-ViT model...")
    neurosnake_vit = NeuroSnakeViTModel.create_model()
    neurosnake_vit.summary()

    print(f"\nNeuroSnake-ViT Parameters: {neurosnake_vit.count_params():,}")

    # Test forward pass
    print("\n" + "=" * 80)
    print("2. Testing forward pass...")
    test_input = tf.random.normal(
        [1, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS]
    )

    neurosnake_vit_output = neurosnake_vit(test_input, training=False)

    print(f"NeuroSnake-ViT output shape: {neurosnake_vit_output.shape}")
    print("\n✓ NeuroSnake-ViT test passed!")
