import tensorflow as tf
from tensorflow.keras import layers

class SliceAggregator25D(layers.Layer):
    """
    2.5D Slice Aggregator.
    Aggregates information across adjacent slices (Depth) to provide volumetric context.
    Uses an attention-based mechanism to weight the importance of center vs neighbor slices.
    
    Args:
        filters (int): Number of output filters.
    """
    def __init__(self, filters, **kwargs):
        super(SliceAggregator25D, self).__init__(**kwargs)
        self.filters = filters
        
    def build(self, input_shape):
        # input_shape: (Batch, Depth, Height, Width, Channels)
        # We want to learn weights for the depth dimension
        # Simple attention: 1x1 conv to squeeze channels -> softmax over depth -> weight
        
        self.attention_conv = layers.Conv3D(
            filters=1,
            kernel_size=(1, 3, 3),
            padding='same',
            activation=None,
            name="attention_score_conv"
        )
        
        self.out_conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=3,
            padding='same',
            activation='relu',
            name="out_conv"
        )
        
        super(SliceAggregator25D, self).build(input_shape)
        
    def call(self, inputs):
        # inputs: (Batch, Depth, Height, Width, Channels) or (Batch, Height, Width, Channels)
        
        # Auto-adapt to 2D inputs by treating them as 3D with depth=1
        is_expanded = False
        if len(inputs.shape) == 4:
            inputs = tf.expand_dims(inputs, axis=1) # (B, 1, H, W, C)
            is_expanded = True
            
        # Calculate attention scores per spatial location
        # (Batch, Depth, H, W, 1)
        scores = self.attention_conv(inputs)
        weights = tf.nn.softmax(scores, axis=1) # Softmax over Depth
        
        # Weighted sum across depth
        # (Batch, Depth, H, W, C) * (Batch, Depth, H, W, 1)
        weighted_inputs = inputs * weights
        
        # Sum over Depth -> (Batch, H, W, C)
        aggregated = tf.reduce_sum(weighted_inputs, axis=1)
        
        # Refine features
        output = self.out_conv(aggregated)
        
        return output
