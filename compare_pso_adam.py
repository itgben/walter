import os
import math
import time
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from collections import deque

import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import KFold

import optuna
from optuna.storages import RDBStorage
from optuna.pruners import MedianPruner


import seaborn
import plotly.io as pio


print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)



transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])




class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), # 3 input channels for RGB, 16 output channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Pooling layer with 2x2 filter
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512), # After 3 max-pooling layers, the size is 4x4 with 64 channels
            nn.ReLU(),
            nn.Dropout(0.5),  # Adding dropout with a rate of 50%
            nn.Linear(512, 10), # 10 classes in CIFAR10
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x



model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

if device == "cuda":
    torch.cuda.empty_cache()

model = model.to(device)
print("Model on:", next(model.parameters()).device)

criterion = nn.CrossEntropyLoss()



train_data_full = CIFAR10(root='./data', train=True, download=True, transform=transform)

# Split into training and validation (80-20%)
train_size = int(0.8 * len(train_data_full))
val_size = len(train_data_full) - train_size
train_data, val_data = random_split(train_data_full, [train_size, val_size])

# Further split validation into validation and test sets (50-50%)
val_size = test_size = val_size // 2
val_data, test_data = random_split(val_data, [val_size, test_size])

# Create DataLoaders for the train, validation, and test sets
batch_size = 32 # Experiment with values like 32, 64, 128, 256, etc.
num_workers = os.cpu_count() - 1
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)




class PhaseShiftOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr, threshold, shift_boundary, min_gradient_magnitude, optimal_weight, A, f, phi, C, phase_shift_loss_weight, transition_slope, optimizer_name):

        # Initialize the optimizer with phase shift parameters.
        defaults = dict(lr=lr, threshold=threshold, shift_boundary=shift_boundary, min_gradient_magnitude=min_gradient_magnitude, optimal_weight=optimal_weight, A=A, f=f, phi=phi, C=C, phase_shift_loss_weight=phase_shift_loss_weight, transition_slope=transition_slope, optimizer_name=optimizer_name)
        
        print("Initializing \U0001F4E1  PhaseShiftOptimizer \U0001F4E1  with parameters:\n", defaults)
        print("\n")

        super(PhaseShiftOptimizer, self).__init__(params, defaults)
        self.gradient_history = deque(maxlen=2)

    def step(self, closure=None):
        # Perform an optimization step, applying phase shift logic if conditions are met.
        #print("Starting optimizer step...")  # Debug
        loss = None
        if closure is not None:
            loss = closure()

        phase_shift_details = []

        for group in self.param_groups:
            # Extract parameters
            lr, threshold, shift_boundary, min_gradient_magnitude, optimal_weight, A, f, phi, C, phase_shift_loss_weight, transition_slope, optimizer_name = self.extract_parameters(group)
            #print(f"Extracted parameters: lr={lr}, threshold={threshold}, shift_boundary={shift_boundary}, optimal_weight={optimal_weight}, A={A}, f={f}, phi={phi}, C={C}, transition_slope={transition_slope}")  # Debug
            most_significant_weight = None
            max_gradient_change = 0

            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                p.data -= lr * grad
                most_significant_weight, max_gradient_change = self.apply_phase_shift_logic(idx, grad, max_gradient_change, min_gradient_magnitude, p, phase_shift_details)

            # Additional logic for phase shift
            if most_significant_weight is not None:
                phase_shift_direction = torch.sign(most_significant_weight.grad)
                #print(f"Applying phase shift: phase_shift_direction={phase_shift_direction}")  # Debug
                self.apply_phase_shift(most_significant_weight, phase_shift_direction, threshold, shift_boundary)


        # Update gradient history
        self.gradient_history.append([p.grad.clone() for p in self.param_groups[0]['params']])

        return loss, phase_shift_details



    def phase_shift_loss(self, weight, optimal_weight, A, f, phi, C, transition_slope):
        # Compute the phase shift loss, a combination of sinusoidal and linear parts.
        #print(f"Calculating phase shift loss: optimal_weight={optimal_weight}, A={A}, f={f}, phi={phi}, C={C}, transition_slope={transition_slope}")  # Debug
        sinusoidal_part = A * torch.sin(2 * np.pi * f * weight + phi)
        phase_shift_part = C + transition_slope * (weight - optimal_weight)
        loss_values = torch.where(weight < optimal_weight, sinusoidal_part, phase_shift_part)
        return loss_values.mean()


    def extract_parameters(self, group):
        # Extract parameters for the current optimization step.
        #print("Extracting parameters...")  # Debug statement
        lr, threshold, shift_boundary, min_gradient_magnitude, optimal_weight, A, f, phi, C, phase_shift_loss_weight, transition_slope, optimizer_name = group['lr'], group['threshold'], group['shift_boundary'], group['min_gradient_magnitude'], group['optimal_weight'], group['A'], group['f'], group['phi'], group['C'], group['phase_shift_loss_weight'], group['transition_slope'], group['optimizer_name']
        #print(f"Extracted phase_shift_loss_weight: {phase_shift_loss_weight}")  # Debug statement
        return lr, threshold, shift_boundary, min_gradient_magnitude, optimal_weight, A, f, phi, C, phase_shift_loss_weight, transition_slope, optimizer_name


    def apply_phase_shift_logic(self, idx, grad, max_gradient_change, min_gradient_magnitude, p, phase_shift_details):
        # Determine if a phase shift should be applied based on gradient changes.
        #print(f"Applying phase shift logic for param {idx}")  # Debug
        previous_grad = self.gradient_history[-1][idx] if len(self.gradient_history) > 0 else torch.zeros_like(grad)
        gradient_change = torch.abs(grad - previous_grad).max()
        most_significant_weight = None  # Initialize variable

        if gradient_change > max_gradient_change and gradient_change > min_gradient_magnitude:
            max_gradient_change = gradient_change
            most_significant_weight = p
            phase_shift_details.append({
                'gradient_change': max_gradient_change.item(),
                'phase_shift_value': gradient_change.cpu().numpy().tolist()
            })
        return most_significant_weight, max_gradient_change



    def apply_phase_shift(self, most_significant_weight, phase_shift_direction, threshold, shift_boundary):
        # Apply the phase shift to the most significant weight.
        #print(f"Applying phase shift: most_significant_weight={most_significant_weight}, phase_shift_direction={phase_shift_direction}, threshold={threshold}, shift_boundary={shift_boundary}")  # Debug
        with torch.no_grad():
            phase_shift_value = phase_shift_direction * threshold
            padded_shift_value = torch.clamp(phase_shift_value, -shift_boundary, shift_boundary)
            most_significant_weight += padded_shift_value






def train(optimizer, train_loader, val_loader, test_loader):
    training_loss_history = []
    validation_loss_history = []


    lr = optimizer.param_groups[0]['lr']
    optimizer_name = optimizer.param_groups[0].get('optimizer_name', None)

    if optimizer_name == "PSO":
        writer = SummaryWriter('runs/simulate_pso_cifar')

        threshold = optimizer.param_groups[0]['threshold']
        shift_boundary = optimizer.param_groups[0]['shift_boundary']
        min_gradient_magnitude = optimizer.param_groups[0]['min_gradient_magnitude']
        optimal_weight = optimizer.param_groups[0]['optimal_weight']
        A = optimizer.param_groups[0]['A']
        f = optimizer.param_groups[0]['f']
        phi = optimizer.param_groups[0]['phi']
        C = optimizer.param_groups[0]['C']
        phase_shift_loss_weight = optimizer.param_groups[0]['phase_shift_loss_weight']
        transition_slope = optimizer.param_groups[0]['transition_slope']
    else:
        writer = SummaryWriter('runs/simulate_adam_cifar')


    start_time = time.time()

    best_val_loss = float('inf')
    global_step = 0
    num_epochs = 10
    early_stopping_patience = 3
    epochs_without_improvement = 0

    convergence_threshold = 0.005
    convergence_patience = 4
    convergence_counter = 0

    training_accuracy_metric = Accuracy(task='multiclass', num_classes=10).to(device)


    for epoch in range(num_epochs):
        model.train()
        training_loss_epoch = 0
        training_accuracy_epoch = 0
        training_accuracy_metric.reset()

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            def closure():
                optimizer.zero_grad()

                outputs = model(batch_x)
                main_loss = criterion(outputs, batch_y)

                # Calculate the predicted labels
                predicted_labels = torch.argmax(outputs, dim=1)

                # Update the training accuracy metric
                training_accuracy_metric.update(predicted_labels, batch_y)


                if optimizer_name == "PSO":
                    weights = [param.data for param in model.parameters()]

                    # Calculate the phase shift loss for each weight
                    phase_shift_loss_values = [
                        optimizer.phase_shift_loss(weight, optimal_weight, A, f, phi, C, transition_slope)
                        for weight in weights
                    ]

                    # Combine the phase shift loss values
                    phase_shift_loss_value = sum(phase_shift_loss_values)

                    # Combine losses
                    total_loss = main_loss + phase_shift_loss_weight * phase_shift_loss_value


                else:
                    # No phase shift loss for other optimizers
                    phase_shift_loss_value = 0
                    total_loss = main_loss


                total_loss.backward()

                return total_loss, main_loss, phase_shift_loss_value



            result = optimizer.step(closure=closure)

            if optimizer_name == "PSO":
                loss, phase_shift_details = result
            else:
                loss = result
                phase_shift_details = None

            # Unpack the values from the loss tuple
            total_loss_value, main_loss_value, phase_shift_loss_value = loss


            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


            if optimizer_name == "PSO":
                for step_num, details in enumerate(phase_shift_details):
                    writer.add_scalar(f'Gradient Change/Step {step_num}', details['gradient_change'], global_step)
                    writer.add_scalar(f'Phase Shift Value/Step {step_num}', details['phase_shift_value'], global_step)
                    writer.add_scalar('Phase Shift Loss', phase_shift_loss_value.item(), global_step)

            writer.add_scalar('Main Loss', main_loss_value.item(), global_step)



            # Accumulate training loss for the epoch
            training_loss_epoch += total_loss_value.item()

            training_accuracy_epoch = training_accuracy_metric.compute()

            # Log the training accuracy
            writer.add_scalar('Training Accuracy', training_accuracy_epoch, epoch)

            # Reset the training accuracy metric for the next epoch
            training_accuracy_metric.reset()

            global_step += 1

        # Compute average training loss for the epoch
        training_loss_epoch /= len(train_loader)
        training_loss_history.append(training_loss_epoch)


        # Log histograms of weights and gradients
        for name, weight in model.named_parameters():
            writer.add_histogram(name, weight, epoch)
            writer.add_histogram(f'{name}/grad', weight.grad, epoch)

        # Evaluate on validation set
        accuracy_metric = Accuracy(task='multiclass', num_classes=10).to(device)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_outputs = model(val_x)
                val_loss += criterion(val_outputs, val_y).item()
                val_acc = accuracy_metric(val_outputs, val_y)
        val_loss /= len(val_loader)  # Compute average validation loss
        validation_accuracy_text = f"Validation Accuracy/Multiclass: {val_acc:.6f}"
        print(highlight(validation_accuracy_text))
        

        # Compute relative change in validation loss
        if epoch > 0:
            previous_val_loss = validation_loss_history[-1]
            relative_change = abs(val_loss - previous_val_loss) / (previous_val_loss + 1e-7)

            if relative_change < convergence_threshold:
                convergence_counter += 1
                print(f'Convergence Warning: {convergence_counter}/{convergence_patience} relative change: {relative_change:.6f}') # New debug statement
                if convergence_counter >= convergence_patience:
                    print("RELAX. CONVERGENCE HAS HAPPENED.")
                    break
            else:
                convergence_counter = 0


        validation_loss_history.append(val_loss)



        writer.add_scalar('Training Loss', total_loss_value.item(), epoch)
        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Validation Accuracy', val_acc, epoch)



        if epoch % 2 == 0:
            images = torchvision.utils.make_grid(batch_x)
            writer.add_image('Images', images, epoch)

            print(f'Epoch [{epoch+1}/{num_epochs}], Time: {time.time() - start_time:.2f} seconds, Main Loss: {main_loss_value:.6f}, Training Loss: {training_loss_epoch:.6f}, Training Accuracy: {training_accuracy_epoch:.6f}, Phase Shift Loss: {phase_shift_loss_value:.6f}, Validation Loss: {val_loss:.6f}, Best Validation Loss: {best_val_loss:.6f}')

        

        # Save the model if it has the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

            if optimizer_name == "PSO":
                torch.save(best_model_state, "cifar_best_pso_model_weights.pth")
            else:
                torch.save(best_model_state, "cifar_best_adam_model_weights.pth")


            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"Warning: No improvement for {epochs_without_improvement}/{early_stopping_patience} epochs.") # New debug statement
            if epochs_without_improvement >= early_stopping_patience:
                print("Early stopping triggered. Restoring best model.")
                model.load_state_dict(best_model_state)
                break


    if optimizer_name == "PSO":

        hparams = {
            'lr': lr,
            'threshold': threshold,
            'shift_boundary': shift_boundary,
            'min_gradient_magnitude': min_gradient_magnitude,
            'optimal_weight': optimal_weight,
            'A': A,
            'f': f,
            'phi': phi,
            'C': C,
            'phase_shift_loss_weight': phase_shift_loss_weight,
            'transition_slope': transition_slope,
            'optimizer_name': optimizer_name,
        }
    else:
        hparams = {
            'lr': lr,
            'optimizer_name': optimizer_name
        }

    final_validation_loss = best_val_loss
    final_validation_accuracy = val_acc  # Assuming val_acc has the final validation accuracy
    metrics = {'final_validation_loss': final_validation_loss, 'final_validation_accuracy': final_validation_accuracy}
    writer.add_hparams(hparams, metrics)


    writer.close()

    return training_loss_history, validation_loss_history



def train_and_evaluate_fold(train_loader_fold, val_loader_fold, test_loader_fold, optimizer_params, fold_number, trial):
    print("TRAIN FOLD  \U0001F3CB")

    # Create a new model instance for the fold
    model_fold = model

    optimizer_name = optimizer_params['optimizer_name']

    if optimizer_name == "PSO":
        optimizer_fold = PhaseShiftOptimizer(model_fold.parameters(), **optimizer_params) # Unpack parameters
    elif optimizer_name == "Adam":
        optimizer_fold = torch.optim.Adam(model_fold.parameters(), lr=optimizer_params['lr'])
    elif optimizer_name == "SGD":
        optimizer_fold = torch.optim.SGD(model_fold.parameters(), lr=optimizer_params['lr'])
    elif optimizer_name == "RMSprop":
        optimizer_fold = torch.optim.RMSprop(model_fold.parameters(), lr=optimizer_params['lr'])
    else:
        raise ValueError(f"Unknown optimizer name: {optimizer_name}")


    
    # Train the model on the fold
    training_loss_history, validation_loss_history = train(optimizer_fold, train_loader_fold, val_loader_fold, test_loader_fold)
    

    # Print separator for better output readability
    print("\n" + "-" * 40)
    val_accuracy, val_class_report, val_conf_matrix, val_f1_score_macro = evaluate_metrics(model_fold, val_loader_fold)
    print("\n" + "-" * 40)


    # Compute a custom validation metric, if needed
    val_metric = compute_validation_metric(validation_loss_history)

    trial.report(val_metric, step=fold_number)  # Report the metric with the fold_number as the step

    # Check if the trial should be pruned based on the reported value
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    return val_metric, val_f1_score_macro # Return custom validation metric and F1-score


def cross_validate(train_loader, val_loader, test_loader, optimizer_params, k_folds, trial):
    validation_metrics = []
    f1_scores = [] # To store F1-scores
    
    for fold_number in range(k_folds):
        print(f"\n\nFold: {fold_number}")
        val_metric, f1_score = train_and_evaluate_fold(train_loader, val_loader, test_loader, optimizer_params, fold_number, trial)
        validation_metrics.append(val_metric)  # Store the custom validation metric
        f1_scores.append(f1_score)  # Store the F1-score

    average_val_metric = sum(validation_metrics) / len(validation_metrics)
    average_f1_score = sum(f1_scores) / len(f1_scores)

    return average_val_metric, average_f1_score # Return both averages






def compute_validation_metric(validation_loss_history):
    # Here you can define how to compute a validation metric from the validation loss history
    # For example, you might return the final validation loss:
    fvl = validation_loss_history[-1]
    fvl_text = f"Compute Final Validation Loss: {fvl})"

    print(highlight(fvl_text))
    return fvl



def evaluate_metrics(model, loader_fold):
    print("Evaluating Model Metrics \U0001F4CA")

    # Capture warnings to handle specific issues
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # Enable all warnings

        model.eval()
        y_pred_list = []
        y_true_list = []

        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in loader_fold:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device) # Move to device
                y_pred = model(x_batch)
                
                # Compute predicted labels from the output
                _, predicted = torch.max(y_pred.data, 1)
                
                # Update counts for total samples and correct predictions
                total_samples += y_batch.size(0)
                correct_predictions += (predicted == y_batch).sum().item()
                
                # Store predicted and true labels for analysis
                y_pred_list.append(predicted.cpu().numpy())
                y_true_list.append(y_batch.cpu().numpy())

        # Handle specific warnings
        for warning in w:
            if issubclass(warning.category, UndefinedMetricWarning):
                print("Warning: Undefined metric detected - some classes may have no predicted samples, leading to ill-defined precision and F-score.")
            else:
                warnings.warn(warning.message)

        y_pred_all = np.concatenate(y_pred_list, axis=0)
        y_true_all = np.concatenate(y_true_list, axis=0)
        
        # Compute overall accuracy
        accuracy = correct_predictions / total_samples

        # Compute classification metrics (precision, recall, F1-score)
        class_report = classification_report(y_true_all, y_pred_all, output_dict=True)
        f1_score_macro = class_report['macro avg']['f1-score']  # or 'micro avg' or 'weighted avg', depending on your needs

        # Compute confusion matrix for detailed analysis
        conf_matrix = confusion_matrix(y_true_all, y_pred_all)

        # Print results
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        print('Detailed Classification Report:')
        print(class_report)
        print('Confusion Matrix (True vs Predicted Labels):')
        print(conf_matrix)

    return accuracy, class_report, conf_matrix, f1_score_macro




def plot_results(x_test, y_test, y_pred, loss_history, validation_loss_history, weight_history, gradient_history, residuals, title_suffix):
    fig, axs = plt.subplots(3, 2, figsize=(15, 18))

    # Scatter Plot for Actual vs Predicted
    axs[0, 0].scatter(x_test.cpu().numpy(), y_test.cpu().numpy(), c='b', label='Actual')
    axs[0, 0].scatter(x_test.cpu().numpy(), y_pred.cpu().numpy(), c='r', label='Predicted')
    axs[0, 0].set_xlabel('Input')
    axs[0, 0].set_ylabel('Output')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    axs[0, 0].set_title('Actual vs Predicted ' + title_suffix)

    # Plot for the validation loss curve
    axs[0, 1].plot(loss_history, 'b-', label='Training Loss')
    axs[0, 1].plot(validation_loss_history, 'r-', label='Validation Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    axs[0, 1].set_yscale('log')  # Uncomment if a log scale is desired
    axs[0, 1].set_title('Loss Curve ' + title_suffix)

    # Plotting Weight Changes
    axs[1, 0].plot([weights[0][0].item() for weights in weight_history], 'g-')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Weight Value')
    axs[1, 0].grid(True)
    axs[1, 0].set_title('Weight Changes of the First Weight in the First Layer ' + title_suffix)

    # Plotting Gradient Magnitudes
    axs[1, 1].plot([grad[0][0].item() for grad in gradient_history], 'm-')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Gradient Magnitude')
    axs[1, 1].grid(True)
    axs[1, 1].set_title('Gradient Magnitudes of the First Weight in the First Layer')

    # Plotting Residuals
    axs[2, 0].scatter(x_test.cpu().numpy(), residuals, c='b')
    axs[2, 0].set_xlabel('Actual Output')
    axs[2, 0].set_ylabel('Residual')
    axs[2, 0].grid(True)
    axs[2, 0].set_title('Residual Plot ' + title_suffix)

    # Histogram for Residuals
    axs[2, 1].hist(residuals, bins=20, color='c')
    axs[2, 1].set_xlabel('Residual')
    axs[2, 1].set_ylabel('Frequency')
    axs[2, 1].set_title('Histogram of Residuals ' + title_suffix)
    axs[2, 1].grid(True)

    plt.tight_layout()





def highlight(text, width=80, character='*'):
    border = character * width
    centered_text = textwrap.fill(text, width=width - 4)
    centered_text = centered_text.center(width - 4)
    highlighted_text = f"{border}\n* {centered_text} *\n{border}"
    return highlighted_text



def objective(trial,optimizer_name):


    '''
    # GOOD - wide open training params    
    
    k_folds = 5
    lr = trial.suggest_float('lr', 0.0001, 0.01)  # Learning rate
    threshold = trial.suggest_float('threshold', 0.1, 0.2, step=0.01)  # Threshold for phase shift
    shift_boundary = trial.suggest_float('shift_boundary', 0.1, 0.3)  # shift_boundary for shift variation
    min_gradient_magnitude = trial.suggest_float('min_gradient_magnitude', 0.05, 0.2)  # Sensitivity to gradient changes
    optimal_weight = trial.suggest_float('optimal_weight', 0.7, 0.85)  # Transition point
    A = trial.suggest_float('A', 0.1, 1.0)  # Amplitude for oscillations
    f = trial.suggest_float('f', 0.75, 1.0)  # Frequency for oscillations
    phi = trial.suggest_float('phi', 0.001, math.pi)  # Phase shift
    C = trial.suggest_float('C', 0.5, 2.0)  # Constant for vapor phase
    phase_shift_loss_weight = trial.suggest_float('phase_shift_loss_weight', 0.01, 0.5)  # Weight for phase shift loss
    transition_slope = trial.suggest_float('transition_slope', 0.1, 0.2)  # Transition slope



    # BETTER - more focused training params    
    
    k_folds = 5
    lr = trial.suggest_float('lr', 0.0003, 0.001)  # Learning rate
    threshold = trial.suggest_float('threshold', 0.23, 0.25, step=0.01)  # Threshold for phase shift
    shift_boundary = trial.suggest_float('shift_boundary', 0.27, 0.3)  # shift_boundary for shift variation
    min_gradient_magnitude = trial.suggest_float('min_gradient_magnitude', 0.16, 0.18)  # Sensitivity to gradient changes
    optimal_weight = trial.suggest_float('optimal_weight', 0.74, 0.77)  # Transition point
    A = trial.suggest_float('A', 0.13, 0.15)  # Amplitude for oscillations
    f = trial.suggest_float('f', 0.88, .95)  # Frequency for oscillations
    phi = trial.suggest_float('phi', 1.42, math.pi)  # Phase shift
    C = trial.suggest_float('C', 1.07, 1.98)  # Constant for vapor phase
    phase_shift_loss_weight = trial.suggest_float('phase_shift_loss_weight', 0.31, 0.36)  # Weight for phase shift loss
    transition_slope = trial.suggest_float('transition_slope', 0.16, 0.18)  # Transition slope





    # BEST - focused and tuned training params    
    
    k_folds = 4
    lr = trial.suggest_float('lr', 0.005, 0.01)  # Learning rate
    threshold = trial.suggest_float('threshold', 0.18, 0.3, step=0.005)  # Threshold for phase shift
    shift_boundary = trial.suggest_float('shift_boundary', 0.1, 0.3)  # shift_boundary for shift variation
    min_gradient_magnitude = trial.suggest_float('min_gradient_magnitude', 0.1, 0.3)  # Sensitivity to gradient changes
    optimal_weight = trial.suggest_float('optimal_weight', 0.7, 0.85)  # Transition point
    A = trial.suggest_float('A', 0.1, 1.0)  # Amplitude for oscillations
    f = trial.suggest_float('f', 0.75, 1.0)  # Frequency for oscillations
    phi = trial.suggest_float('phi', 0.001, math.pi)  # Phase shift
    C = trial.suggest_float('C', 0.5, 2.0)  # Constant for vapor phase
    phase_shift_loss_weight = trial.suggest_float('phase_shift_loss_weight', 0.01, 0.5)  # Weight for phase shift loss
    transition_slope = trial.suggest_float('transition_slope', 0.1, 0.2)  # Transition slope


     '''





    
    k_folds = 5
    lr = trial.suggest_float('lr', 0.0001, 0.001)  # Learning rate
    threshold = trial.suggest_float('threshold', 0.23, 0.25, step=0.01)  # Threshold for phase shift
    shift_boundary = trial.suggest_float('shift_boundary', 0.25, 0.3)  # shift_boundary for shift variation
    min_gradient_magnitude = trial.suggest_float('min_gradient_magnitude', 0.16, 0.18)  # Sensitivity to gradient changes
    optimal_weight = trial.suggest_float('optimal_weight', 0.7, 0.8)  # Transition point
    A = trial.suggest_float('A', 0.1, 0.9)  # Amplitude for oscillations
    f = trial.suggest_float('f', 0.65, .95)  # Frequency for oscillations
    phi = trial.suggest_float('phi', 1.42, math.pi)  # Phase shift
    C = trial.suggest_float('C', 1.07, 1.98)  # Constant for vapor phase
    phase_shift_loss_weight = trial.suggest_float('phase_shift_loss_weight', 0.30, 0.40)  # Weight for phase shift loss
    transition_slope = trial.suggest_float('transition_slope', 0.16, 0.18)  # Transition slope

    # Create optimizer with hyperparameters
    if optimizer_name == "PSO":
        optimizer_params = {
            'lr': lr,
            'threshold': threshold,
            'shift_boundary': shift_boundary,
            'min_gradient_magnitude': min_gradient_magnitude,
            'optimal_weight': optimal_weight,
            'A': A,
            'f': f,
            'phi': phi,
            'C': C,
            'phase_shift_loss_weight': phase_shift_loss_weight,
            'transition_slope': transition_slope,
            'optimizer_name': optimizer_name
        }
    else:
        optimizer_params = {
            'lr': lr,
            'optimizer_name': optimizer_name
        }

    # Training and Evaluation function
    average_val_metric, average_f1_score = cross_validate(train_loader, val_loader, test_loader, optimizer_params, k_folds, trial)

    # Print the current trial's result (optional, for monitoring progress)
    average_val_metric_text = f"Trial result: Average validation metric = {average_val_metric}"
    print(highlight(average_val_metric_text))

    average_f1_score_text = f"Trial result: Average f1-score = {average_f1_score}"
    print(highlight(average_f1_score_text))

    # Combining both metrics (minimizing loss and maximizing F1-score)
    combined_metric = average_val_metric - average_f1_score  # Or you can use a weighted combination

    return combined_metric









# EXECUTE THE SCRIPT
# Create a unique folder based on timestamp
pso_folder_name = f"cifar_pso_results_{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(pso_folder_name)
pso_study_file_path = os.path.join(pso_folder_name, "study.db")
pso_study_name = "cifar_pso_study"



adam_folder_name = f"cifar_adam_results_{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(adam_folder_name)
adam_study_file_path = os.path.join(adam_folder_name, "study.db")
adam_study_name = "cifar_adam_study"

pruner = MedianPruner(n_startup_trials=1, n_warmup_steps=1, interval_steps=1)

study_adam = optuna.create_study(direction='minimize', pruner=pruner, study_name=adam_study_name, storage=f"sqlite:///{adam_study_file_path}")
study_adam.optimize(lambda trial: objective(trial, 'Adam'), n_trials=50)


study_pso = optuna.create_study(direction='minimize', pruner=pruner, study_name=pso_study_name, storage=f"sqlite:///{pso_study_file_path}")
study_pso.optimize(lambda trial: objective(trial, 'PSO'), n_trials=50)


# Print best parameters
print(study_pso.best_params)
print(study_adam.best_params)


# Save the trials data frame as a CSV file
study_pso_df = study_pso.trials_dataframe()
study_pso_df.to_csv(os.path.join(pso_folder_name, "trials.csv"), index=False)

study_adam_df = study_pso.trials_dataframe()
study_adam_df.to_csv(os.path.join(adam_folder_name, "trials.csv"), index=False)

print(study_pso_df)
print(study_adam_df)


# Save the optimization history plot
fig = optuna.visualization.plot_optimization_history(study_pso)
pio.write_image(fig, os.path.join(pso_folder_name, "optimization_history.png"))

# Save the parallel coordinate plot
fig = optuna.visualization.plot_parallel_coordinate(study_pso)
pio.write_image(fig, os.path.join(pso_folder_name, "parallel_coordinate.png"))

# Save the contour plot
fig = optuna.visualization.plot_contour(study_pso, params=['lr', 'threshold'])
pio.write_image(fig, os.path.join(pso_folder_name, "contour.png"))

# Save the slice plot
fig = optuna.visualization.plot_slice(study_pso)
pio.write_image(fig, os.path.join(pso_folder_name, "slice.png"))


# Importance of individual hyperparameters, which can help you understand which hyperparameters are most influential in affecting the objective function.
fig = optuna.visualization.plot_param_importances(study_pso)
pio.write_image(fig, os.path.join(pso_folder_name, "param_importances.png"))


#  Specific hyperparameters you want to investigate further, like "f" and "transition_slope."
fig = optuna.visualization.plot_contour(study_pso, params=['f', 'transition_slope'])
pio.write_image(fig, os.path.join(pso_folder_name, "custom_contour.png"))

# Individual hyperparameter's effects by "slicing" other hyperparameters' ranges.
fig = optuna.visualization.plot_slice(study_pso, params=['f', 'transition_slope'])
pio.write_image(fig, os.path.join(pso_folder_name, "custom_slice.png"))

# Save the Edwards Plot
fig = optuna.visualization.plot_edf(study_pso)
pio.write_image(fig, os.path.join(pso_folder_name, "edf.png"))

# Save the Intermediate Values Plot
fig = optuna.visualization.plot_intermediate_values(study_pso)
pio.write_image(fig, os.path.join(pso_folder_name, "intermediate_values.png"))


# Save the parallel coordinate plot for optimizer comparison
#fig = optuna.visualization.plot_parallel_coordinate(study_pso, params=['optimizer_name', 'value'])
#pio.write_image(fig, os.path.join(pso_folder_name, "optimizer_parallel_coordinate.png"))


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

# Save the optimization history plot
fig = optuna.visualization.plot_optimization_history(study_adam)
pio.write_image(fig, os.path.join(adam_folder_name, "optimization_history.png"))

# Save the parallel coordinate plot
fig = optuna.visualization.plot_parallel_coordinate(study_adam)
pio.write_image(fig, os.path.join(adam_folder_name, "parallel_coordinate.png"))


# Save the slice plot
fig = optuna.visualization.plot_slice(study_adam)
pio.write_image(fig, os.path.join(adam_folder_name, "slice.png"))


# Importance of individual hyperparameters, which can help you understand which hyperparameters are most influential in affecting the objective function.
fig = optuna.visualization.plot_param_importances(study_adam)
pio.write_image(fig, os.path.join(adam_folder_name, "param_importances.png"))


# Save the Edwards Plot
fig = optuna.visualization.plot_edf(study_adam)
pio.write_image(fig, os.path.join(adam_folder_name, "edf.png"))

# Save the Intermediate Values Plot
fig = optuna.visualization.plot_intermediate_values(study_adam)
pio.write_image(fig, os.path.join(adam_folder_name, "intermediate_values.png"))


# Save the parallel coordinate plot for optimizer comparison
#fig = optuna.visualization.plot_parallel_coordinate(study_adam, params=['optimizer_name', 'value'])
#pio.write_image(fig, os.path.join(adam_folder_name, "optimizer_parallel_coordinate.png"))

# Show the figures if needed (optional)
#fig.show()




