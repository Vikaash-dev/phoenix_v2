"""
Kolmogorov-Arnold Network (KAN) Layer for TensorFlow.
Based on the principle that multivariate functions can be represented as sums of univariate non-linear functions.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class KANLinear(layers.Layer):
    """
    KAN Linear Layer.
    Replaces matrix multiplication w*x with learnable non-linear functions phi(x).
    phi(x) is modeled as a basis of B-splines.
    """
    
    def __init__(
        self,
        units,
        grid_size=5,
        spline_order=3,
        base_activation='silu',
        grid_range=[-1, 1],
        **kwargs
    ):
        super(KANLinear, self).__init__(**kwargs)
        self.units = units
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = keras.activations.get(base_activation)
        self.grid_range = grid_range

    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Grid initialization
        # h = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        # Grid points: (grid_size + 1)
        # We need extended grid for B-splines
        
        # Base weight (like residual connection w*x)
        self.base_weight = self.add_weight(
            name='base_weight',
            shape=(input_dim, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Spline weights
        # Number of coefficients per input-output pair = grid_size + spline_order
        num_coeffs = self.grid_size + self.spline_order
        
        self.spline_weight = self.add_weight(
            name='spline_weight',
            shape=(input_dim, self.units, num_coeffs),
            initializer=tf.random_normal_initializer(stddev=0.1),
            trainable=True
        )
        
        # Spline scaler (w_s in original paper)
        self.spline_scaler = self.add_weight(
            name='spline_scaler',
            shape=(input_dim, self.units),
            initializer='ones',
            trainable=True
        )
        
        # Fixed grid for B-spline calculation
        # Range [-1, 1] extended by spline_order on both sides
        step = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        grid_start = self.grid_range[0] - self.spline_order * step
        grid_end = self.grid_range[1] + self.spline_order * step + step # +step for inclusive
        
        # Create grid vector
        # Using tf.constant to keep it fixed
        self.grid = tf.linspace(grid_start, grid_end, self.grid_size + 2 * self.spline_order + 1)
        self.grid = tf.cast(self.grid, self.dtype)
        
        super(KANLinear, self).build(input_shape)

    def call(self, x):
        # x shape: (batch, input_dim)
        
        # 1. Base activation (silu(x) * w_b)
        base_output = tf.matmul(self.base_activation(x), self.base_weight)
        
        # 2. B-spline interpolation
        # Compute B-spline bases
        # x shape: (batch, input_dim)
        # We need to map x to splines. 
        # For efficiency, we can assume x is in [-1, 1] (inputs should be normalized!)
        # Or we use a grid that covers the input distribution.
        
        # Expand x for spline computation
        # (batch, input_dim, 1)
        x_expanded = tf.expand_dims(x, -1)
        
        # Compute spline bases B_i(x)
        # This is the tricky part in TF without specialized ops.
        # We implement a recursive De Boor's algorithm or efficient approximate.
        # For speed, we use a simplified basis expansion if possible, but let's try a vectorized implementation.
        
        # Normalized x for grid
        # grid shape: (G,)
        # We want to find which interval x falls into.
        
        # Recursive B-spline Implementation (Order 0 to k)
        
        # Flatten grid to (1, 1, G)
        grid = tf.reshape(self.grid, (1, 1, -1))
        
        # Bases order 0: 1 if grid[i] <= x < grid[i+1], else 0
        # Check against all intervals
        # (batch, input_dim, G-1)
        
        # Note: We need (grid_size + spline_order) basis functions
        # The grid has (grid_size + 2*order + 1) points
        
        # Let's start with order 0
        # bases[i, k] = 1 if t[i] <= x < t[i+1]
        
        # We perform this calculation iteratively
        
        # grid: [t_0, t_1, ..., t_{G+2k}]
        
        bases = []
        
        # Order 0
        # Check all intervals t_i <= x < t_{i+1}
        # grid has T points. We have T-1 intervals.
        
        # x_expanded: (B, D, 1)
        # grid: (1, 1, T)
        
        # mask: (B, D, T-1)
        grid_lower = grid[:, :, :-1]
        grid_upper = grid[:, :, 1:]
        
        # Order 0 bases
        b0 = tf.cast((x_expanded >= grid_lower) & (x_expanded < grid_upper), self.dtype)
        
        bases_current = b0 # (B, D, T-1)
        
        # Recursion for higher orders
        for k in range(1, self.spline_order + 1):
            # B_{i,k}(x) = term1 + term2
            # term1 = (x - t_i) / (t_{i+k} - t_i) * B_{i, k-1}(x)
            # term2 = (t_{i+k+1} - x) / (t_{i+k+1} - t_{i+1}) * B_{i+1, k-1}(x)
            
            # We need to act on bases_current which has shape (..., M)
            # New bases will have shape (..., M-1)
            
            b_prev = bases_current
            
            # Shifted grids
            # t_i
            t_i = grid[:, :, :-(k+1)]
            # t_{i+k}
            t_ik = grid[:, :, k:-1]
            # t_{i+1}
            t_i1 = grid[:, :, 1:-k]
            # t_{i+k+1}
            t_ik1 = grid[:, :, (k+1):]
            
            # Safe division (avoid div by zero if grid points are same)
            eps = 1e-7
            
            term1 = (x_expanded - t_i) / (t_ik - t_i + eps) * b_prev[:, :, :-1]
            term2 = (t_ik1 - x_expanded) / (t_ik1 - t_i1 + eps) * b_prev[:, :, 1:]
            
            bases_current = term1 + term2
            
        # Final bases: bases_current
        # Shape: (B, D, grid_size + spline_order)
        # We kept extra grid points to ensure we end up with enough bases
        
        spline_basis = bases_current
        
        # 3. Spline activation: sum(c_i * B_i(x))
        # spline_basis: (B, in_dim, num_coeffs)
        # spline_weight: (in_dim, out_dim, num_coeffs)
        
        # We need to contract over in_dim and num_coeffs... no wait.
        # Original KAN: for each input-output pair (i, j), we have a function phi_{ij}(x_i)
        # output_j = sum_i phi_{ij}(x_i)
        
        # So we calculate y_ij = sum_k (c_{ijk} * B_k(x_i))
        # Then output_j = sum_i y_ij
        
        # Einsum:
        # B: batch
        # I: input_dim
        # O: output_dim
        # C: num_coeffs
        
        # basis: (B, I, C)
        # weight: (I, O, C)
        # result: (B, O)
        
        # First compute y_{ij} (contribution of input i to output j)
        # But we also have spline_scaler (I, O)
        
        # Let's do it efficiently:
        # spline_output = sum_i sum_k (weight_{i,j,k} * basis_{b,i,k})
        
        spline_output = tf.einsum('bic,ioc->bo', spline_basis, self.spline_weight)
        
        # Apply normalization/scaling if needed?
        # Original paper suggests scaling.
        # But for this implementation, let's keep it simple.
        
        # Wait, the scaler is important.
        # Scaler is per activation function phi_{ij}.
        # Wait, usually scaler is just a scalar per function.
        # Actually my einsum implicitly sums over 'i'.
        # If we want to apply scaler, we should do:
        # y_ij_raw = sum_k (weight_{i,j,k} * basis_{b,i,k}) -> shape (B, I, O)
        # y_ij_scaled = y_ij_raw * scaler_{i,j}
        # output_j = sum_i y_ij_scaled
        
        # Re-doing einsum to keep (B, I, O) intermediate
        # But that's memory heavy (Batch * In * Out).
        # Optimization: (weight * scaler) can be precomputed if scaler is static?
        # Scaler is trainable.
        
        # weighted_weights = weight * scaler[:, :, None]
        # output = einsum('bic,ioc->bo', basis, weighted_weights)
        
        weighted_spline_weights = self.spline_weight * self.spline_scaler[:, :, tf.newaxis]
        spline_output = tf.einsum('bic,ioc->bo', spline_basis, weighted_spline_weights)
        
        return base_output + spline_output

if __name__ == "__main__":
    # Test KAN Layer
    layer = KANLinear(32)
    x = tf.random.normal((10, 16)) # Batch 10, Input 16
    y = layer(x)
    print("Input:", x.shape)
    print("Output:", y.shape)
