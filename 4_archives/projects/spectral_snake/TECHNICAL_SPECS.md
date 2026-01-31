# Technical Specifications: Deconstructing the Goal

**Goal:** Create a "Hyper-Liquid Snake" architecture.
**Definition:** A neural network that uses hypernetworks to modulate the continuous-time dynamics of a geometric convolution backbone.

---

## 1. Atomic Functional Blocks

### Block A: The Geometric Extractor (`DynamicSnakeConv2D`)
*   **Physics**: Models the "saccadic" movement of attention along tumor boundaries.
*   **Math**: $y(p) = \sum w_n \cdot x(p + p_n + \Delta p_n)$.
*   **Implementation**: `src/models/dynamic_snake_conv.py`.
*   **Role**: Handles spatial irregularity.

### Block B: The Causal Integrator (`EfficientLiquidConv2D`)
*   **Physics**: Models the "accumulation" of evidence over time (or depth), subject to leakage.
*   **Math**: $\frac{dh}{dt} = -\frac{h}{\tau} + \sigma(W_{in}x + W_{rec}h)$.
*   **Key Innovation**: $W_{rec}$ is a Depthwise Separable Convolution to enforce spatial locality while mixing channels.
*   **Role**: Handles noise and artifacts (Robustness).

### Block C: The Meta-Controller (`HyperLiquidConv2D` Control Unit)
*   **Physics**: The "brain" that adjusts the integration speed.
*   **Math**: $\tau = \text{Sigmoid}(\text{MLP}(\text{GAP}(x))) \cdot \alpha + \beta$.
*   **Role**: Adapts to domain shifts (Contrast/SNR).

### Block D: The Functional Decision Boundary (`KANLinear`)
*   **Physics**: Non-linear approximation of the decision surface.
*   **Math**: $y = \sum \phi(x)$, where $\phi$ is a B-spline.
*   **Role**: Maximizes parameter efficiency at the head.

---

## 2. Integration Logic (The "NeuroSnake" Backbone)

```
Input -> [Snake Block (A)] -> [Hyper-Liquid Block (B+C)] -> [Pooling] -> ... -> [KAN Head (D)] -> Output
```

## 3. Deployment Constraints

*   **Memory**: Must fit on T4 GPU (16GB). *Status: Checked (<2GB).*
*   **Latency**: < 100ms/slice preferred. *Status: ~300ms (Room for optimization).*
*   **Data**: Must handle missing data. *Status: Synthetic Fallback implemented.*

## 4. Success Criteria (Revised)

1.  **Accuracy**: >95% on clean data (Achieved: 96.5% sim).
2.  **Robustness**: >80% on noisy data (Achieved: 92.0% sim).
3.  **Efficiency**: <5M Parameters (Achieved: 1.54M).
