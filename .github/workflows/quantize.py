import joblib
import numpy as np
import torch

# Load sklearn model
model = joblib.load("linear_model.joblib")
coef = model.coef_
intercept = model.intercept_

unquant_params = {"coef": coef, "intercept": intercept}
joblib.dump(unquant_params, "unquant_params.joblib")

# Quantization (Manual)
scale = 255 / (np.max(np.abs(coef)))
quant_coef = np.round(coef * scale).astype(np.uint8)
quant_intercept = np.round(intercept * scale).astype(np.uint8)

quant_params = {"coef": quant_coef, "intercept": quant_intercept, "scale": scale}
joblib.dump(quant_params, "quant_params.joblib")

# Dequantization for inference
dequant_coef = quant_coef / scale
dequant_intercept = quant_intercept / scale

X = torch.tensor(np.random.rand(1, len(coef)), dtype=torch.float32)
y_pred = torch.matmul(X, torch.tensor(dequant_coef, dtype=torch.float32)) + dequant_intercept
print("Sample inference:", y_pred)
