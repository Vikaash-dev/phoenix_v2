import tensorflow as tf
import numpy as np
import cv2

class MambaCAM:
    """
    MambaCAM: Visualization tool for State-Space Model internals.

    This class helps visualize the internal dynamics of Mamba/S6 layers,
    specifically focusing on:
    1. Gating mechanism (z): Determines how much information flows through.
    2. State contributions (B * x): Shows what information is being written to the state.
    3. Time-scale (dt): Shows the granularity/persistence of memory.
    """

    def __init__(self, model):
        """
        Args:
            model: The PhoenixMambaV2 model instance.
        """
        self.model = model

    def normalize_heatmap(self, heatmap):
        """Normalizes a heatmap to range [0, 1]."""
        heatmap = tf.maximum(heatmap, 0) # ReLU
        if tf.reduce_max(heatmap) > 0:
            heatmap /= tf.reduce_max(heatmap)
        return heatmap.numpy()

    def generate_map(self, input_image, target_class_index=None, layer_name='stage4', focus='gating'):
        """
        Generates a heatmap based on the specified focus.

        Args:
            input_image: Input tensor of shape (1, D, H, W, C)
            target_class_index: (Optional) Index of the target class for gradient-based weighting.
                                If None, returns the raw activation magnitude.
            layer_name: The stage to visualize (e.g., 'stage4').
            focus: 'gating' (z), 'state_write' (dB * u), or 'timescale' (dt).

        Returns:
            heatmap: 2D numpy array of shape (H, W)
        """

        # We use a GradientTape to capture gradients if we want class-discriminative maps
        # But since the user specifically asked for B*x or z, we might just look at activations.
        # However, to be a true "CAM", we typically weight channel activations by the gradient of the class.
        # Let's support both. If target_class_index is provided, we use Grad-CAM like weighting.
        # If not, we just average the activations (like Activation Maximization or simple feature viz).

        with tf.GradientTape() as tape:
            # We need to watch the input if we were doing input gradients,
            # but here we need gradients w.r.t internals.
            # Since get_explainability_maps returns tensors, we can't easily watch 'internals'
            # because they are created inside the function.
            #
            # Standard GradCAM requires access to the feature map tensor *within* the graph.
            # The current implementation of `get_explainability_maps` returns numpy values or eager tensors?
            # It returns the tensors.

            logits, internals_dict = self.model.get_explainability_maps(input_image)

            if layer_name not in internals_dict:
                available = list(internals_dict.keys())
                raise ValueError(f"Layer {layer_name} not found. Available: {available}")

            internals = internals_dict[layer_name]

            # Derive shapes from 'z' which is (B, H, W, d_mamba)
            B_shape = tf.shape(input_image)[0]
            z_shape = tf.shape(internals['z'])
            H, W = z_shape[1], z_shape[2]

            # Extract the tensor of interest
            if focus == 'gating':
                # z is (B, H, W, d_mamba)
                # We typically want silu(z) as that's the actual gate
                target_tensor = tf.nn.silu(internals['z'])
            elif focus == 'state_write':
                # dB: (B, L, d_mamba, d_state)
                # u/x_mamba: (B, L, d_mamba)
                # We want to see the magnitude of the update: dB * u
                # But shapes are different.
                # dB is derived from dt and B.
                # Let's approximate 'state write' as |dt * B * u|.
                # But internals has dB and x_mamba.
                # dB: (B, L, d_mamba, d_state) flattened spatial
                # x_mamba: (B, L, d_mamba) flattened spatial

                # Reshape back to spatial if they are flattened
                # The model code flattens them. get_explainability_maps reshapes 'z' but maybe not others?
                # Let's check `get_explainability_maps` in `phoenix.py`.
                # It only reshapes 'z'.

                # We need to reshape others.

                x_mamba = internals['x_mamba'] # (B, L, d_mamba)
                dB = internals['dB'] # (B, L, d_mamba, d_state)

                # Reshape x_mamba
                d_mamba = tf.shape(x_mamba)[-1]
                x_mamba_spatial = tf.reshape(x_mamba, (B_shape, H, W, d_mamba))

                # Compute magnitude of write.
                # Term is dB * u. u is x_mamba.
                # dB is (B, L, d_mamba, d_state).
                # Sum over d_state to get "total write strength per channel"?
                # Or L2 norm over d_state.
                dB_norm = tf.reduce_mean(tf.abs(dB), axis=-1) # (B, L, d_mamba)

                # Reshape dB_norm
                dB_norm_spatial = tf.reshape(dB_norm, (B_shape, H, W, d_mamba))

                # Element-wise mult
                target_tensor = dB_norm_spatial * tf.abs(x_mamba_spatial)

            elif focus == 'timescale':
                dt = internals['dt'] # (B, L, d_mamba)
                d_mamba = tf.shape(dt)[-1]
                target_tensor = tf.reshape(dt, (B_shape, H, W, d_mamba))
            else:
                raise ValueError(f"Unknown focus: {focus}")

            if target_class_index is not None:
                score = logits[:, target_class_index]
            else:
                score = None

        if target_class_index is not None and score is not None:
            # Gradient-weighted (Grad-CAM style)
            # Warning: This requires tape.gradient to work through the model graph.
            # But we already ran the model forward pass inside tape? Yes.
            # So we can compute gradients of score w.r.t target_tensor.
            grads = tape.gradient(score, target_tensor)

            # Global Average Pooling of gradients
            weights = tf.reduce_mean(grads, axis=(0, 1, 2))

            # Weighted combination
            cam = tf.reduce_sum(tf.multiply(weights, target_tensor), axis=-1)

        else:
            # Simple magnitude (activation sum)
            cam = tf.reduce_mean(target_tensor, axis=-1)

        # Process the heatmap
        cam = cam[0] # Take first sample
        cam = self.normalize_heatmap(cam)

        # Resize to input image resolution if needed
        # (Assuming input_image is (B, D, H, W, C), we want (H, W))
        # But we can leave it at feature map resolution or resize.
        # Let's resize to a standard size or leave it to the user.
        # Usually CAM is overlaid on image.

        return cam

    def overlay_heatmap(self, image, heatmap, alpha=0.4):
        """
        Overlays heatmap on an image.

        Args:
            image: Original image (H, W, 3) or (H, W) in range [0, 1] or [0, 255]
            heatmap: Heatmap (H, W) range [0, 1]
        """
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        if image.max() <= 1.0:
            image = np.uint8(255 * image)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        return overlay
