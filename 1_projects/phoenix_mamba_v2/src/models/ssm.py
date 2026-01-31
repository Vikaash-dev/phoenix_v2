import tensorflow as tf
from tensorflow.keras import layers, initializers

class S6Layer(layers.Layer):
    """
    Selective State-Space Model (S6) Layer.
    Implements a simplified selective scan mechanism using TensorFlow.
    
    Args:
        d_model (int): Dimension of the model (input channels).
        d_state (int): Dimension of the state (hidden size of SSM).
        d_mamba (int): Dimension of the inner projection (usually expand * d_model).
        dt_rank (int): Rank of the delta projection.
    """
    def __init__(self, d_model, d_state=16, d_mamba=None, dt_rank=None, **kwargs):
        super(S6Layer, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.d_mamba = d_mamba if d_mamba is not None else 2 * d_model
        self.dt_rank = dt_rank if dt_rank is not None else (d_model + 15) // 16

    def build(self, input_shape):
        # Projects input to hidden mamba dimension
        self.in_proj = layers.Dense(self.d_mamba * 2, use_bias=False)
        
        # Conv1D logic (simulated with Conv1D layer for local context)
        self.conv1d = layers.Conv1D(
            filters=self.d_mamba, 
            kernel_size=4, 
            groups=self.d_mamba, 
            padding='same', 
            use_bias=True
        )
        
        # S6 Parameters
        # A: (d_mamba, d_state) of trainable parameters
        self.A_log = self.add_weight(
            name='A_log',
            shape=(self.d_mamba, self.d_state),
            initializer=initializers.RandomUniform(minval=-3.0, maxval=1.0),
            trainable=True
        )
        
        # D: (d_mamba) skip connection
        self.D = self.add_weight(
            name='D',
            shape=(self.d_mamba,),
            initializer='ones',
            trainable=True
        )

        # Selection Projects: x -> B, C, dt
        # Projecting to (dt + B + C)
        # dt: d_mamba
        # B: d_state
        # C: d_state
        # But commonly B and C are broadcasted or have rank.
        # Simplified: x -> dt (d_mamba), B (d_state), C (d_state)
        # In full Mamba, B and C are (d_mamba, d_state) but derived from (Batch, L, d_state)
        # Let's project to combined features.
        
        self.x_proj = layers.Dense(self.dt_rank + self.d_state * 2, use_bias=False)
        
        self.dt_proj = layers.Dense(self.d_mamba, use_bias=True)
        
        self.out_proj = layers.Dense(self.d_model, use_bias=False)
        
        super(S6Layer, self).build(input_shape)

    def call(self, x):
        # x: (Batch, L, d_model)
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        # 1. Project to mamba dim * 2 (for x and z/gate)
        xz = self.in_proj(x) # (B, L, 2 * d_mamba)
        x_mamba, z = tf.split(xz, 2, axis=-1)
        
        # 2. Conv1D
        x_mamba = self.conv1d(x_mamba)
        x_mamba = tf.nn.silu(x_mamba)
        
        # 3. SSM
        # Project input to time-dependent parameters
        x_dbl = self.x_proj(x_mamba) # (B, L, dt_rank + 2*d_state)
        dt_rank_val, B, C = tf.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], axis=-1)
        
        dt = self.dt_proj(dt_rank_val) # (B, L, d_mamba)
        dt = tf.nn.softplus(dt) # Softplus to ensure positive timescale
        
        A = -tf.exp(self.A_log) # Ensure A is negative for stability
        
        # Discretization
        # A_bar = exp(dt * A)
        # B_bar = dt * B (simplified, really (exp(dt*A)-1)/A * B but approx dt*B works for small dt)
        
        # A: (d_mamba, d_state)
        # dt: (B, L, d_mamba)
        # We need broadcast for element-wise muiltiplication
        # A_bar = exp(dt.unsqueeze(-1) * A)
         
        dt_expanded = tf.expand_dims(dt, axis=-1) # (B, L, d_mamba, 1)
        A_expanded = tf.reshape(A, (1, 1, self.d_mamba, self.d_state))
        
        dA = tf.exp(dt_expanded * A_expanded) # (B, L, d_mamba, d_state)
        
        # B: (B, L, d_state)
        # B needs to be (B, L, d_mamba, d_state) or effectively projected
        # In Mamba paper: B is (B, L, N), and affects all D channels?
        # Standard implementation: Matrix broadcasting
        # Let's treat B as shared across d_mamba or apply a scalar mix.
        # Actually x_mamba is (B, L, d_mamba).
        # We want to integrate: h_t = dA * h_{t-1} + dB * x_t
        # where x_t is a scalar per channel? No, x_mamba is vector.
        # This part is tricky in pure TF without scan overhead. 
        # Let's use simplified scan.
        
        B_expanded = tf.expand_dims(B, axis=-2) # (B, L, 1, d_state)
        dB = dt_expanded * B_expanded # (B, L, d_mamba, d_state)
        
        # Input for recurrence
        # u = x_mamba (B, L, d_mamba)
        u_expanded = tf.expand_dims(x_mamba, axis=-1) # (B, L, d_mamba, 1)
        
        # Recurrence func
        # state: (B, d_mamba, d_state)
        # step input: (dA_t, dB_t, u_t)
        
        # Rearrange for scan: (L, B, ...)
        dA_T = tf.transpose(dA, [1, 0, 2, 3])
        dB_T = tf.transpose(dB, [1, 0, 2, 3])
        u_T = tf.transpose(u_expanded, [1, 0, 2, 3])
        
        initial_state = tf.zeros((batch_size, self.d_mamba, self.d_state))
        
        def scan_step(prev_state, inputs):
            da_t, db_t, u_t = inputs
            # prev_state: (B, d_mamba, d_state)
            # da_t: (B, d_mamba, d_state)
            # db_t: (B, d_mamba, d_state)
            # u_t: (B, d_mamba, 1)
            
            # h_t = da_t * h_{t-1} + db_t * u_t
            # element-wise mul
            return da_t * prev_state + db_t * u_t

        states = tf.scan(
            scan_step,
            elems=(dA_T, dB_T, u_T),
            initializer=initial_state
        )
        
        # states: (L, B, d_mamba, d_state)
        states = tf.transpose(states, [1, 0, 2, 3]) # (B, L, d_mamba, d_state)
        
        # C: (B, L, d_state)
        C_expanded = tf.expand_dims(C, axis=-2) # (B, L, 1, d_state)
        
        # y = C * h
        # dot product over d_state
        y = tf.reduce_sum(states * C_expanded, axis=-1) # (B, L, d_mamba)
        
        # Add D skip
        y = y + x_mamba * self.D
        
        # 4. Gating
        y = y * tf.nn.silu(z)
        
        # 5. Out projection
        out = self.out_proj(y)
        
        return out
