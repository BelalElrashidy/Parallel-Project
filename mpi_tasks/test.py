import numpy as np
from sklearn.preprocessing import StandardScaler

# Synthetic data: features without intercept
X_raw = np.array([[1, 2, 1],
                  [2, 3, 4],
                  [3, 2, 2],
                  [4, 5, 5]])
y = np.array([5, 15, 10, 20])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X = np.hstack((np.ones((X_scaled.shape[0],1)), X_scaled))

# Normal equation with ridge
lambda_reg = 1e-5
XtX = X.T @ X
Xty = X.T @ y
theta = np.linalg.solve(XtX + lambda_reg * np.eye(X.shape[1]), Xty)

intercept = theta[0]
coeffs = theta[1:]

print("Intercept:", intercept)
print("Coefficients:", coeffs)

def predict(X_raw, intercept, coeffs, scaler):
    X_scaled = scaler.transform(X_raw)
    X_with_intercept = np.hstack((np.ones((X_scaled.shape[0],1)), X_scaled))
    return X_with_intercept @ np.hstack((intercept, coeffs))

# Prediction on training data
preds = predict(X_raw, intercept, coeffs, scaler)
print("Predictions:", preds)
print("True y:", y)
