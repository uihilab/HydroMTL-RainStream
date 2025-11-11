import numpy as np
from scipy.stats import pearsonr
from math import sqrt
from sklearn.metrics import mean_squared_error as mse
import torch

def nse(y_true, y_pred):
    """Nash-Sutcliffe Efficiency"""
    return 1 - np.sum((y_pred - y_true)**2) / np.sum((y_true - np.mean(y_true))**2)

def pearson_r_mean(y_true, y_pred):
    """Mean Pearson R across all forecast horizons."""
    correlations = []
    for i in range(y_true.shape[1]):
        # Calculate Pearson R for each timestep (column)
        r = pearsonr(y_true[:, i], y_pred[:, i]).statistic
        correlations.append(r)
    return np.mean(correlations)

def nrmse_calculation(y_true, y_pred):
    """Normalized Root Mean Squared Error."""
    return sqrt(mse(y_true, y_pred)) / np.mean(y_true)


def evaluate_streamflow_with_custom_metrics(model, test_loader, scaler_y, device):
    """
    Evaluates the streamflow task using your specific logic and metrics.
    """
    print("Starting evaluation for streamflow task...")
    model.eval()
    
    # --- 1. Gather all predictions and targets ---
    # This loop collects all tensors. Can be memory intensive for very large test sets.
    with torch.no_grad():
        for k, (x, y) in enumerate(test_loader):
            x = x.to(device)
            output = model(x)
            
            if k == 0:
                result_pred_test = output.cpu()
                result_test = y.cpu()
            else:
                result_pred_test = torch.cat((result_pred_test, output.cpu()), 0)
                result_test = torch.cat((result_test, y), 0)

    print("All predictions gathered.")

    # --- 2. Inverse transform the scaled data to its original range ---
    print("Inverse transforming predictions and targets...")
    result_pred_test = scaler_y.inverse_transform(result_pred_test.numpy())
    result_test = scaler_y.inverse_transform(result_test.numpy())

    # --- 3. Calculate metrics using your custom functions ---
    print("Calculating metrics...")
    
    # Calculate overall metrics
    nrmse = nrmse_calculation(result_test, result_pred_test)
    pearson = pearson_r_mean(result_test, result_pred_test)

    # Calculate NSE for each of the 24 forecast horizons
    nses_test = []
    for i in range(result_test.shape[1]): # Iterate through the 24 columns
        nses_test.append(nse(result_test[:, i], result_pred_test[:, i]))
    
    # Calculate the average NSE
    nse_mean = np.mean(nses_test)
    print(f"NSEs for each horizon: {nses_test} Mean NSE: {np.mean(nses_test)} Median NSE: {nse_median}")
    results = {
        "NRMSE": nrmse,
        "Pearson_R_(Mean)": pearson,
        "NSE_(Mean)": nse_mean,
    }
    return results