import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
import csv
import random
import numpy as np
import pandas as pd

# Import the final hybrid model
from hybrid_model import MultiModalTransformer
from datasets import BenchMarkDataset, RainDataset
from rainfall_scores import evaluate_rainfall_with_custom_metrics
from streamflow_scores import evaluate_streamflow_with_custom_metrics

# Function to set random seeds
def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set random seed
SEED = 42
set_random_seeds(SEED)

# ======================================================================================
# 1. DESIRABLE HYPERPARAMETERS (CONFIG)
# ======================================================================================
CONFIG = {
    # --- Training Setup ---
    "EPOCHS": 150,
    "BATCH_SIZE": 64,
    "LEARNING_RATE": 3e-4,
    "WEIGHT_DECAY": 0.01,

    # --- Loss Weighting ---
    "LOSS_WEIGHT_STREAMFLOW": 1.0,
    "LOSS_WEIGHT_RAINFALL": 1.0,
    
    # --- Scheduler ---
    "WARMUP_STEPS": 250,
    "PATIENCE": 15, # Early stopping patience
    
    # --- Model Architecture ---
    "EMBEDDING_DIM": 128,
    "N_HEADS": 4,
    "N_LAYERS": 2,
    "DROPOUT": 0.2,
    
    # --- Data Specific ---
    "PATCH_SIZE": 16,
    "IMAGE_SIZE": 128,
    "RAINFALL_CHANNELS": 2,
    "STREAMFLOW_INPUT_DIM": 3,
    "STREAMFLOW_OUTPUT_DIM": 24,

    # --- Explicit Dataset Sizes around ---
    "NUM_TRAIN_SAMPLES_STREAMFLOW": 40000,
    "NUM_TRAIN_SAMPLES_RAINFALL": 20000,
    "NUM_VAL_SAMPLES_STREAMFLOW": 14000,
    "NUM_VAL_SAMPLES_RAINFALL": 7000,

    # --- Script Control & SAVING ---
    "MODEL_SAVE_PATH": "Results/best_multitask_model.pth",
    "EPOCH_LOG_PATH": "Results/epoch_log.csv",
    "STEP_LOG_PATH": "Results/step_log.csv",  
    "LOG_FREQUENCY": 50, # Log every 50 steps
}

# ======================================================================================
# 2. SETUP & DATASET PLACEHOLDER
# ======================================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======================================================================================
# 3. ROBUST TRAINING AND VALIDATION FUNCTIONS FOR IMBALANCED DATA
# ======================================================================================
def train_one_epoch(model, s_loader, r_loader, optimizer, scheduler, criterion, epoch, config, step_writer, step_log_file):
    model.train()
    criterion.train()
    running_loss = 0.0
    num_steps_per_epoch = max(len(s_loader), len(r_loader))
    s_iter, r_iter = iter(s_loader), iter(r_loader)
    
    print(f"\n--- Starting Epoch {epoch+1}/{config['EPOCHS']} ({num_steps_per_epoch} steps) ---")
    
    for i in range(num_steps_per_epoch):
        try: s_input, s_target = next(s_iter)
        except StopIteration: s_iter = iter(s_loader); s_input, s_target = next(s_iter)
        try: r_input, r_target = next(r_iter)
        except StopIteration: r_iter = iter(r_loader); r_input, r_target = next(r_iter)
        s_input, s_target = s_input.to(device), s_target.to(device)
        r_input, r_target = r_input.to(device), r_target.to(device)
        
        optimizer.zero_grad()
        
        # --- Make predictions ---
        s_pred = model(s_input)
        r_pred = model(r_input)
        
        # --- Calculate loss using the custom criterion ---
        # It returns the weighted total loss, and the raw losses for logging
        total_loss, raw_loss_s, raw_loss_r = criterion(s_pred, s_target, r_pred, r_target)
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += total_loss.item()
        if (i + 1) % config["LOG_FREQUENCY"] == 0 or (i + 1) == num_steps_per_epoch:
            # The learned weights are exp(-log_sigma)
            s_weight = torch.exp(-criterion.log_sigmas[0]).item()
            r_weight = torch.exp(-criterion.log_sigmas[1]).item()
            
            step_log_row = [
                epoch + 1, i + 1, total_loss.item(), 
                raw_loss_s.item(), raw_loss_r.item(), # Log the raw, unweighted losses
                s_weight, r_weight, # Log the learned weights
                scheduler.get_last_lr()[0]
            ]
            step_writer.writerow(step_log_row)
            step_log_file.flush()

    return running_loss / num_steps_per_epoch


def validate(model, s_loader, r_loader):
    model.eval()
    s_loss, r_loss = 0.0, 0.0

    # We use a standard L1 loss for validation, as we only care about the raw performance
    validation_criterion = nn.L1Loss()
    
    with torch.no_grad():
        for s_input, s_target in s_loader:
            s_input, s_target = s_input.to(device), s_target.to(device)
            s_pred = model(s_input)
            s_loss += validation_criterion(s_pred, s_target).item()
            
        for r_input, r_target in r_loader:
            r_input, r_target = r_input.to(device), r_target.to(device)
            r_pred = model(r_input)
            r_loss += validation_criterion(r_pred, r_target).item()

    return s_loss / len(s_loader), r_loss / len(r_loader)


class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super(UncertaintyWeightedLoss, self).__init__()
        # Initialize the log(sigma^2) parameters, one for each task
        self.log_sigmas = nn.Parameter(torch.zeros(num_tasks))
        self.mse = nn.L1Loss()

    def forward(self, pred_s, target_s, pred_r, target_r):
        # Calculate the individual MSE losses
        loss_s = self.mse(pred_s, target_s)
        loss_r = self.mse(pred_r, target_r)
        
        
        weighted_loss_s = 0.5 * torch.exp(-self.log_sigmas[0]) * loss_s + 0.5 * self.log_sigmas[0]
        weighted_loss_r = 0.5 * torch.exp(-self.log_sigmas[1]) * loss_r + 0.5 * self.log_sigmas[1]
        
        return weighted_loss_s + weighted_loss_r, loss_s.detach(), loss_r.detach()

# ======================================================================================
# 4. MAIN EXECUTION BLOCK
# ======================================================================================
if __name__ == '__main__':
    model = MultiModalTransformer(config=CONFIG).to(device)
    criterion = UncertaintyWeightedLoss().to(device)
    optimizer = optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=CONFIG["LEARNING_RATE"], weight_decay=CONFIG["WEIGHT_DECAY"])

    train_rain_dataset = RainDataset(type='train')
    test_rain_dataset = RainDataset(type='test')
    val_rain_dataset = RainDataset(type='val')

    train_streamflow_dataset = BenchMarkDataset(split='train')
    test_streamflow_dataset = BenchMarkDataset(split='test')
    val_streamflow_dataset = BenchMarkDataset(split='valid')
    streamflow_scalerY = test_streamflow_dataset.scalerY
    
    # Use num_workers > 0 and pin_memory=True for faster data loading
    train_rain_loader = DataLoader(train_rain_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)
    val_rain_loader = DataLoader(val_rain_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=4, pin_memory=True)
    test_rain_loader = DataLoader(test_rain_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=4, pin_memory=True)

    train_streamflow_loader = DataLoader(train_streamflow_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)
    val_streamflow_loader = DataLoader(val_streamflow_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=4, pin_memory=True)
    test_streamflow_loader = DataLoader(test_streamflow_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=4, pin_memory=True)

    num_training_steps = CONFIG["EPOCHS"] * max(len(train_streamflow_loader), len(train_rain_loader))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=CONFIG["WARMUP_STEPS"], num_training_steps=num_training_steps)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    with open(CONFIG["EPOCH_LOG_PATH"], 'w', newline='') as epoch_log_file, \
        open(CONFIG["STEP_LOG_PATH"], 'w', newline='') as step_log_file:
        epoch_writer = csv.writer(epoch_log_file)
        step_writer = csv.writer(step_log_file)

        epoch_header = ["epoch", "avg_train_loss", "val_loss_streamflow", "val_loss_rainfall", "combined_val_loss"]
        epoch_writer.writerow(epoch_header)
        epoch_log_file.flush() # Flush after writing header
        step_header = ["epoch", "step", "total_loss", "streamflow_loss", "rainfall_loss", "learning_rate"]
        step_writer.writerow(step_header)
        step_log_file.flush() # Flush after writing header

        print("Starting training with the final Hybrid CNN-Transformer model...")
        for epoch in range(CONFIG["EPOCHS"]):
            train_loss = train_one_epoch(model, train_streamflow_loader, train_rain_loader, optimizer, scheduler, criterion, epoch, CONFIG, step_writer, step_log_file)
            val_streamflow_loss, val_rain_loss = validate(model, val_streamflow_loader, val_rain_loader)
            current_val_loss = val_streamflow_loss + val_rain_loss
            epoch_log_row = [epoch + 1, train_loss, val_streamflow_loss, val_rain_loss, current_val_loss]
            epoch_writer.writerow(epoch_log_row)
            epoch_log_file.flush() # Flush after each epoch
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                epochs_no_improve = 0 # Reset patience
                torch.save(model.state_dict(), CONFIG["MODEL_SAVE_PATH"])
                print(f"  Validation loss improved. Saving model to {CONFIG['MODEL_SAVE_PATH']}\n")
            else:
                epochs_no_improve += 1
                print(f"  No improvement in validation loss for {epochs_no_improve} epoch(s).\n")
            if epochs_no_improve >= CONFIG["PATIENCE"]:
                print(f"Early stopping triggered after {CONFIG['PATIENCE']} epochs with no improvement.")
                break
    model.load_state_dict(torch.load(CONFIG['MODEL_SAVE_PATH'], map_location=device, weights_only=True))
    rainfall_scores = evaluate_rainfall_with_custom_metrics(model, test_rain_loader, device)
    streamflow_scores = evaluate_streamflow_with_custom_metrics(model, test_streamflow_loader, streamflow_scalerY, device)
    df_rainfall= pd.DataFrame([rainfall_scores])
    df_streamflow= pd.DataFrame([streamflow_scores])
    df_rainfall.to_csv('Results/rainfall_scores.csv', index=False)
    df_streamflow.to_csv('Results/streamflow_scores.csv', index=False)
    print("\nTraining finished.")
    print(f"Best validation loss achieved: {best_val_loss:.8f}")