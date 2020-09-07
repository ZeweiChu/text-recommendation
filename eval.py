import pandas as pd
import sys

label_file = sys.argv[1]
pred_file = sys.argv[2]
constant_pred = None

try:
    constant_pred = float(pred_file)
except ValueError:
    pass

label_data = pd.read_csv(label_file)
labels = label_data["ratings"].values

if constant_pred is None:
    preds_data = pd.read_csv(pred_file, sep="\t")
    preds = preds_data["prediction"].values
    assert labels.shape[0] == preds.shape[0]
else:
    preds = constant_pred

mse = ((labels - preds)**2).mean() 
print("mse: ", mse)
