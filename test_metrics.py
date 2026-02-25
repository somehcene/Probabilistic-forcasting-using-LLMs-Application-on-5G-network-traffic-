import numpy as np
from src.evaluate import calculate_metrics

actual = np.random.rand(12, 1) # This is what scaler.inverse_transform(test_labels[idx].numpy()) produces
chronos_pred_unscaled = np.random.rand(12, 1)

print("actual shape:", actual.shape)
print("pred shape:", chronos_pred_unscaled.shape)
print("Metrics 1st try:", calculate_metrics(actual, chronos_pred_unscaled))

# Now let's see what happens if chronos_pred_unscaled has shape (12, )
chronos_pred_unscaled_flat = np.random.rand(12)
try:
    print("Metrics flat:", calculate_metrics(actual, chronos_pred_unscaled_flat))
except Exception as e:
    print("Error with flat pred:", e)

# Now, wait, the error is [12, 1]... meaning one is len 12 and one is len 1?
try:
    calculate_metrics(np.random.rand(12, 1), np.random.rand(1, 1))
except Exception as e:
    print("Error 12 vs 1:", e)
