"""
Fallback imports for Phoenix Protocol models
Provides graceful fallbacks when optional attention modules are unavailable.
"""


class FallbackAttention:
    def create_identity_block(filters: int, name: str = "identity_attention"):
        import tensorflow as tf
        from tensorflow.keras import layers

        return tf.keras.Sequential(
            [
                layers.Conv2D(filters, 1, 1, padding="same", name=f"{name}_conv"),
                layers.BatchNormalization(name=f"{name}_bn"),
                layers.ReLU(name=f"{name}_relu"),
                layers.Conv2D(filters, 3, 1, padding="same", name=f"{name}_conv2"),
                layers.BatchNormalization(name=f"{name}_bn2"),
                layers.ReLU(name=f"{name}_relu2"),
            ],
            name=name,
        )

    def create_simple_se_block(filters: int, name: str = "simple_se"):
        import tensorflow as tf
        from tensorflow.keras import layers

        def se_block(input_tensor):
            se = layers.GlobalAveragePooling2D()(input_tensor)
            se = layers.Dense(filters // 16, activation="relu", name=f"{name}_fc1")(se)
            se = layers.Dense(filters, activation="sigmoid", name=f"{name}_fc2")(se)
            se = layers.Reshape((1, 1, filters))(se)
            return layers.Multiply()([input_tensor, se])

        return se_block
