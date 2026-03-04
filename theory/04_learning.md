# Theory Chapter 4: Learning Principles

## 4.1 The Objective: Minimizing Loss ($L$)
In supervised learning, we seek a function $f(x; \theta)$ that maps inputs $x$ to targets $y$. The "Loss" $L$ is a metric that quantifies the error over our dataset. 

**Why Minimize?**
Minimizing $L$ is synonymous with maximizing the accuracy of our model's predictions. We use the **Mean Squared Error (MSE)** as our objective function:

$$L = \frac{1}{n} \cdot \sum_{i=1}^n (f(x_i; \theta) - y_i)^2$$

As $L \to 0$, the model's predictions $f(x)$ converge to the ground truth.

## 4.2 Gradient Descent Optimization
<img width="600" alt="Gradient Descent" src="https://github.com/user-attachments/assets/9dea5e8b-3b72-4274-8219-bf7ea34ec0a7" />
Optimization calculates the necessary "nudges" to parameters $(\theta)$ to find the global minimum of the loss surface.

**The Update Rule**:

$$\theta_{next} = \theta_{now} - \eta \cdot \frac{\partial L}{\partial \theta}$$


## 4.3 Hyperparameters: Learning Rate ($\eta$)
The learning rate $\eta$ is the step size taken along the gradient.

| Setting | Mathematical Effect | Consequence |
| :--- | :--- | :--- |
| **High Step** | Step size exceeds valley width | **Divergence**: The loss value explodes or oscillates. |
| **Low Step** | Step size approaches zero | **Vanishing**: High computational cost, trapped in local minima. |
| **Optimal** | Smooth reduction in $L$ | **Convergence**: Efficient descent toward the minimum. |

<img width="600" alt="Learning Rate" src="https://github.com/user-attachments/assets/5a0188af-4dc4-4549-98de-74c3d574a034" />

## 4.4 Model Capacity & Fit
We judge a model by its ability to generalize to unseen data ($\text{Error}_{test}$), governed by the **Bias-Variance Tradeoff**.

1.  **Underfitting (High Bias)**: 
    - **Context**: The model is too simple to capture the underlying pattern.
    - **Math**: $\text{Loss}_{train}$ and $\text{Loss}_{test}$ remain high.
2.  **Overfitting (High Variance)**: 
    - **Context**: The model "memorizes" noise rather than the signal.
    - **Math**: $\text{Loss}_{train} \ll \text{Loss}_{test}$.

<img width="600" alt="Overfitting 1" src="https://github.com/user-attachments/assets/b9bc0c4c-fa85-42bb-9a2c-bdeb888b33b1" />

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/18218258-7640-460f-a964-8ab01329dd04">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/f50a38a3-f5d5-450f-928d-06d8f3849c5a">
  <img src="https://github.com/user-attachments/assets/f50a38a3-f5d5-450f-928d-06d8f3849c5a" width="600" alt="Overfitting 2">
</picture>

## 4.5 Gradient Descent Variants
The method of computing gradients $(\nabla L)$ defines the variant of the algorithm.

| Variant | Dataset used per step | Gradient Formula | Convergence |
| :--- | :--- | :--- | :--- |
| **Batch GD** | Entire dataset | $\frac{1}{N} \sum_{i=1}^N \nabla L_i$ | **Stable**: Guaranteed local convergence. |
| **Stochastic GD** | Single random example | $\nabla L_i$ | **Noisy**: High variance, avoids local minima. |
| **Mini-batch GD** | Small random subset | $\frac{1}{b} \sum_{i=1}^b \nabla L_i$ | **Optimal**: Best of both worlds; GPU-efficient. |

<img width="600" alt="Gradient Descent Variants" src="https://cdn.analyticsvidhya.com/wp-content/uploads/2022/07/58182variations_comparison.webp" />

**Current Implementation**: Our `demo.ipynb` uses **Batch Gradient Descent**. We compute the total loss for all clinical samples and update the weights once per epoch. This is distinct from Stochastic Gradient Descent (SGD), which would update weights after every individual sample.

## 4.6 Feature Scaling (Normalization)
Gradient Descent is sensitive to the scale of input features. Features with large magnitudes (e.g., `radius_mean` $\approx 20$) produce significantly larger gradients than small ones (e.g., `smoothness_mean` $\approx 0.1$), leading to **divergence** or inefficient "zig-zag" optimization.

**Z-score Normalization**:
We transform each feature $x$ using its mean ($\mu$) and standard deviation ($\sigma$):

$$x_{norm} = \frac{x - \mu}{\sigma}$$

This ensures all inputs are centered at 0 with a unit variance, creating a more spherical and easily optimizable loss surface.
