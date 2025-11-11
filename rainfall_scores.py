import torch
from torch.nn import functional as F

def hits(a, b):
    return torch.sum(torch.logical_and(a, b))

def misses(true, pred):
    return torch.sum(torch.logical_xor(pred, true) * true.bool())

def false_alarm(true, pred):
    return torch.sum(torch.logical_xor(pred, true) * pred.bool())

eps=1e-10

def pod(true, pred):
    h = hits(true, pred)
    m = misses(true, pred)
    return h/(h+m+eps)

def far(true, pred):
    f = false_alarm(true, pred)
    h = hits(true, pred)
    return f/(h+f+eps)

def csi(true, pred):
    h = hits(true, pred)
    m = misses(true, pred)
    f = false_alarm(true, pred)
    return h/(h+f+m+eps)


def evaluate_rainfall_with_custom_metrics(model, test_loader, device):
    """
    Evaluates the rainfall task using your specific logic and categorical metrics.
    """
    print("Starting evaluation for rainfall task...")
    model.eval()
    
    # Initialize loss function and accumulators for each metric
    loss_function = torch.nn.L1Loss()
    total_mae, total_pod, total_far, total_csi = 0.0, 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # --- 1. Get model prediction and apply post-processing ---
            predictions = model(inputs) * 105.0
            
            # Scale and threshold the data as per your logic
            predictions_processed = F.threshold(predictions, 0.0001, 0)
            targets_processed = targets * 105.0

            # --- 2. Calculate metrics for the current batch ---
            total_mae += loss_function(predictions_processed, targets_processed).item()
            total_pod += pod(targets_processed, predictions_processed).item()
            total_far += far(targets_processed, predictions_processed).item()
            total_csi += csi(targets_processed, predictions_processed).item()

    # --- 3. Average the metrics over all batches ---
    num_batches = len(test_loader)
    results = {
        "Rainfall_MAE": total_mae / num_batches,
        "Rainfall_POD": total_pod / num_batches,
        "Rainfall_FAR": total_far / num_batches,
        "Rainfall_CSI": total_csi / num_batches,
    }
    return results