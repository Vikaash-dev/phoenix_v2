import tensorflow as tf
import numpy as np
import os

class True25DLoader:
    """
    Implements the 'True 2.5D' loading protocol defined by Agent 3 (Empirical Architect).
    
    Replaces the previous 'fake stacking' (repeating the same slice 3 times) with 
    genuine volumetric context: [z-1, z, z+1].
    """
    
    def __init__(self, volume_dims=(240, 240, 155), modalities=4):
        self.H, self.W, self.D = volume_dims
        self.C = modalities
        
    def extract_2_5d_stack(self, volume, slice_idx):
        """
        Extracts a 2.5D stack for a given slice index from a 3D volume.
        
        Args:
            volume: Numpy array of shape (H, W, D, C) or (H, W, D)
            slice_idx: Integer index of the target slice (z)
            
        Returns:
            stack: Tensor of shape (H, W, 3, C)
        """
        # Ensure volume has channel dim
        if len(volume.shape) == 3:
            volume = volume[..., np.newaxis]
            
        z = slice_idx
        max_z = volume.shape[2] - 1
        
        # Determine indices with boundary handling (clamping)
        # Agent 3 Spec: Pad z=0 with [z, z, z+1], z=D with [z-1, z, z]
        idx_prev = max(0, z - 1)
        idx_curr = z
        idx_next = min(max_z, z + 1)
        
        # Extract slices: (H, W, 1, C)
        slice_prev = volume[:, :, idx_prev, :]
        slice_curr = volume[:, :, idx_curr, :]
        slice_next = volume[:, :, idx_next, :]
        
        # Stack depthwise: (H, W, 3, C)
        # Note: tf.stack acts on a new axis. providing list of (H, W, C) -> (3, H, W, C)
        # We want (Slices, H, W, C) for the PHOENIX input of (B, 3, H, W, C)
        
        stack = np.stack([slice_prev, slice_curr, slice_next], axis=0)
        
        return stack

    def create_mock_batch(self, batch_size=2):
        """
        Creates a mock batch with TRUE volumetric relationships for testing.
        Slice z contains signal S. Slice z-1 contains S - delta.
        This validates that the network can see the delta.
        """
        # Create a synthetic volume where intensity varies by depth
        # Volume: (64, 64, 10, 4)
        vol_shape = (64, 64, 10, 4)
        volume = np.zeros(vol_shape, dtype=np.float32)
        
        # Fill with gradient along Z
        for z in range(10):
            volume[:, :, z, :] = z * 0.1  # Distinct signal per slice
            
        X_batch = []
        for _ in range(batch_size):
            z = np.random.randint(1, 9) # Safe middle slice
            stack = self.extract_2_5d_stack(volume, z)
            X_batch.append(stack)
            
        return np.array(X_batch, dtype=np.float32)

if __name__ == "__main__":
    # Self-test logic
    loader = True25DLoader(volume_dims=(64, 64, 10))
    batch = loader.create_mock_batch(batch_size=1)
    
    print(f"Batch shape: {batch.shape}")
    # Verify gradient: Slice 0 should be < Slice 1 < Slice 2
    s_prev = batch[0, 0, 32, 32, 0]
    s_curr = batch[0, 1, 32, 32, 0] 
    s_next = batch[0, 2, 32, 32, 0]
    
    print(f"Values at center pixel: Prev={s_prev:.2f}, Curr={s_curr:.2f}, Next={s_next:.2f}")
    
    if s_prev < s_curr < s_next:
        print("✅ SUCCESS: True volumetric context preserved.")
    else:
        print("❌ FAILURE: Context not strictly ordered.")
