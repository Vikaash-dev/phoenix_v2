# PHOENIX-MAMBA v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the next-generation Phoenix-Mamba v2 architecture featuring Selective State-Space Models and 2.5D volumetric context.

**Architecture:** A hierarchical hybrid model that replaces self-attention with Selective SSM (S6) blocks. It uses a 2.5D slice aggregator for volumetric context and clinical-specific heterogeneity attention heads.

**Tech Stack:** TensorFlow 2.15+, Keras, UV (environment manager), Pytest.

---

### Task 1: SliceContextAggregator (2.5D)

**Files:**
- Create: `1_projects/phoenix_mamba_v2/src/models/context_aggregator.py`
- Test: `1_projects/phoenix_mamba_v2/tests/test_context_aggregator.py`

**Step 1: Write the failing test**

```python
import tensorflow as tf
import pytest
from src.models.context_aggregator import SliceContextAggregator

def test_aggregator_output_shape():
    batch, h, w, c = 2, 224, 224, 3
    aggregator = SliceContextAggregator(embed_dim=64)
    # Input is 3 adjacent slices
    dummy_input = tf.random.normal((batch, 3, h, w, c))
    output = aggregator(dummy_input)
    assert output.shape == (batch, h, w, 64)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest 1_projects/phoenix_mamba_v2/tests/test_context_aggregator.py -v`
Expected: FAIL (ModuleNotFoundError or AttributeError)

**Step 3: Write minimal implementation**

```python
import tensorflow as tf
from tensorflow.keras import layers

class SliceContextAggregator(layers.Layer):
    def __init__(self, embed_dim=64, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def build(self, input_shape):
        self.slice_encoder = layers.Conv2D(self.embed_dim, 3, padding='same')
        self.temporal_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads
        )
        self.fusion = layers.Conv2D(self.embed_dim, 1, padding='same')
        self.norm = layers.LayerNormalization()

    def call(self, slices, training=None):
        batch_size = tf.shape(slices)[0]
        h, w = tf.shape(slices)[2], tf.shape(slices)[3]

        # Encode current slice (index 1) and neighbors
        s1 = self.slice_encoder(slices[:, 1])
        s0 = self.slice_encoder(slices[:, 0])
        s2 = self.slice_encoder(slices[:, 2])

        # Simple temporal aggregation via attention
        stacked = tf.stack([s0, s1, s2], axis=1)
        reshaped = tf.reshape(stacked, [batch_size, 3, -1, self.embed_dim])

        query = reshaped[:, 1:2]
        attended = self.temporal_attention(query, reshaped, reshaped)

        fused = tf.reshape(attended, [batch_size, h, w, self.embed_dim])
        return self.norm(self.fusion(fused) + s1)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest 1_projects/phoenix_mamba_v2/tests/test_context_aggregator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add 1_projects/phoenix_mamba_v2/src/models/context_aggregator.py 1_projects/phoenix_mamba_v2/tests/test_context_aggregator.py
git commit -m "feat: implement 2.5D SliceContextAggregator"
```

---

### Task 2: SelectiveSSM (S6 Core)

**Files:**
- Create: `1_projects/phoenix_mamba_v2/src/models/mamba_ssm.py`
- Test: `1_projects/phoenix_mamba_v2/tests/test_mamba_ssm.py`

**Step 1: Write the failing test**

```python
import tensorflow as tf
import pytest
from src.models.mamba_ssm import SelectiveSSM

def test_ssm_linearity():
    batch, seq, d_model = 2, 64, 32
    ssm = SelectiveSSM(d_model=d_model, d_state=16)
    x = tf.random.normal((batch, seq, d_model))
    output = ssm(x)
    assert output.shape == (batch, seq, d_model)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest 1_projects/phoenix_mamba_v2/tests/test_mamba_ssm.py -v`
Expected: FAIL

**Step 3: Write minimal implementation (Sequential Approximation)**

```python
import tensorflow as tf
from tensorflow.keras import layers

class SelectiveSSM(layers.Layer):
    def __init__(self, d_model, d_state=16, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.d_state = d_model, d_state

    def build(self, input_shape):
        self.proj_in = layers.Dense(self.d_model * 2)
        self.A_log = self.add_weight("A_log", (self.d_model, self.d_state), initializer='random_normal')
        self.D = self.add_weight("D", (self.d_model,), initializer='ones')
        self.proj_delta = layers.Dense(self.d_model)
        self.proj_B = layers.Dense(self.d_state)
        self.proj_C = layers.Dense(self.d_state)
        self.proj_out = layers.Dense(self.d_model)

    def call(self, x):
        xz = self.proj_in(x)
        x_proj, z = tf.split(xz, 2, axis=-1)

        delta = tf.nn.softplus(self.proj_delta(x_proj))
        B, C = self.proj_B(x_proj), self.proj_C(x_proj)
        A = -tf.exp(self.A_log)

        # Simple scan
        h = tf.zeros((tf.shape(x)[0], self.d_model, self.d_state))
        outputs = []
        for t in range(tf.shape(x)[1]):
            xt, dt, bt, ct = x_proj[:, t], delta[:, t], B[:, t], C[:, t]
            dA = tf.exp(tf.einsum('bd,ds->bds', dt, A))
            dB = tf.einsum('bd,bs->bds', dt * xt, bt)
            h = dA * h + dB
            outputs.append(tf.einsum('bds,bs->bd', h, ct) + self.D * xt)

        y = tf.stack(outputs, axis=1) * tf.nn.silu(z)
        return self.proj_out(y)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest 1_projects/phoenix_mamba_v2/tests/test_mamba_ssm.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add 1_projects/phoenix_mamba_v2/src/models/mamba_ssm.py 1_projects/phoenix_mamba_v2/tests/test_mamba_ssm.py
git commit -m "feat: implement SelectiveSSM (S6) core"
```

---

### Task 3: HeterogeneityAttention

**Files:**
- Create: `1_projects/phoenix_mamba_v2/src/models/clinical_attention.py`
- Test: `1_projects/phoenix_mamba_v2/tests/test_clinical_attention.py`

**Step 1: Write the failing test**

```python
import tensorflow as tf
from src.models.clinical_attention import HeterogeneityAttention

def test_heterogeneity_attention():
    batch, h, w, c = 2, 16, 16, 64
    layer = HeterogeneityAttention(channels=c, num_regions=3)
    x = tf.random.normal((batch, h, w, c))
    output = layer(x)
    assert output.shape == (batch, h, w, c)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest 1_projects/phoenix_mamba_v2/tests/test_clinical_attention.py -v`

**Step 3: Write minimal implementation**

```python
import tensorflow as tf
from tensorflow.keras import layers

class HeterogeneityAttention(layers.Layer):
    def __init__(self, channels, num_regions=3, **kwargs):
        super().__init__(**kwargs)
        self.channels, self.num_regions = channels, num_regions

    def build(self, input_shape):
        self.heads = [layers.MultiHeadAttention(num_heads=1, key_dim=self.channels//self.num_regions) for _ in range(self.num_regions)]
        self.region_weights = self.add_weight("weights", (self.num_regions,), initializer=tf.constant_initializer([0.4, 0.4, 0.2]))
        self.fusion = layers.Dense(self.channels)

    def call(self, x):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x_flat = tf.reshape(x, [tf.shape(x)[0], -1, self.channels])

        outputs = [head(x_flat, x_flat) for head in self.heads]
        weights = tf.nn.softmax(self.region_weights)
        fused = sum(w * out for w, out in zip(tf.unstack(weights), outputs))

        fused = tf.reshape(fused, [tf.shape(x)[0], h, w, -1])
        return self.fusion(fused) + x
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest 1_projects/phoenix_mamba_v2/tests/test_clinical_attention.py -v`

**Step 5: Commit**

```bash
git add 1_projects/phoenix_mamba_v2/src/models/clinical_attention.py 1_projects/phoenix_mamba_v2/tests/test_clinical_attention.py
git commit -m "feat: implement HeterogeneityAttention"
```

---

### Task 4: Model Assembly (PHOENIX-MAMBA v2)

**Files:**
- Create: `1_projects/phoenix_mamba_v2/src/models/phoenix_v2.py`
- Test: `1_projects/phoenix_mamba_v2/tests/test_phoenix_v2.py`

**Step 1-4: Standard Assembly**
Assemble the `create_phoenix_mamba_v2` function using the components from Tasks 1-3. Verify with a full-model summary and shape test.

**Step 5: Commit**
```bash
git add 1_projects/phoenix_mamba_v2/src/models/phoenix_v2.py
git commit -m "feat: assemble complete PHOENIX-MAMBA v2 model"
```
