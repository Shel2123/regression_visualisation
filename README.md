# Linear Regression from Scratch (2 Features, 1 Target, 3D vizualization)

This project demonstrates a minimal linear regression trained with **gradient descent** using two intuitive features from [the Ames Housing dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques): `GrLivArea` (living area) and `OverallQual` (overall quality).

We minimize the  **Mean Squared Error (MSE)** evaluate performance with the  **Root Mean Squared Error (RMSE)** and visualize the fitted regression plane along with the training dynamics over epochs.

---

## 📐 Used maths

**Linear model**

```math
\hat{Y} = XW + b
```

**Standardization**

```math
X_{\text{std}} = \frac{X - \mu}{\sigma}
```

**Loss functions**

```math
\text{MSE}(Y,\hat{Y}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
```

```math
\text{RMSE}(Y,\hat{Y}) = \sqrt{\text{MSE}(Y,\hat{Y})}
```

**Gradients**

```math
\frac{\partial L}{\partial W} = \frac{2}{n} X^\top (\hat{Y} - Y), 
\quad
\frac{\partial L}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)
```

**Gradient descent update**

```math
W = W - \eta \frac{\partial L}{\partial W}

\quad 
b = b - \eta \frac{\partial L}{\partial b}
```

**Early stopping criterion**

```math
\Delta \text{RMSE} < tol \;\; \text{for } p \text{ consecutive epochs } \;\; \Rightarrow \;\; stop
```

---

## ⚙️ How it works

- **Model:** Linear regression with standardized features
- **Loss:** MSE and RMSE (root mean squared error)
- **Optimizer:** Gradient descent with early stopping
- **Visualization:** Interactive Plotly plots + 3D GIF of the regression plane on Matplotlib

---

## 🚀 Quickstart

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the notebook:
   ```bash
   jupyter lab notebooks/main.ipynb
   ```
3. Outputs (GIF, figures) will appear in the `assets/` folder.

---

## 📊 Results (sample)

* Gradient descent


<p align="center">
  <img src="assets/gradient_descent_good.gif" width="600">
</p>


- Final RMSE: ~42,000($)
- 3D visualization of the regression plane:

<p align="center">
  <img src="assets/regression_animation.gif" width="600">
</p>

---

## 📜 License

![MIT License](LICENSE)
