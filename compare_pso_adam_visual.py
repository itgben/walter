import os
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import optuna
from optuna.visualization import plot_param_importances, plot_contour
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from tensorboard.backend.event_processing import event_accumulator
'''
# Check CUDA availability
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


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


# Instantiate the model and move to device
model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create CIFAR10 dataset and split into train, validation, and test sets
train_data_full = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(train_data_full))
val_size = test_size = (len(train_data_full) - train_size) // 2
_, _, test_data = random_split(train_data_full, [train_size, val_size, test_size])

# Create DataLoader for the test set
batch_size = 64
test_loader = DataLoader(test_data, batch_size=batch_size)


# Load model's saved weights if available
adam_model_path = "cinic_best_adam_model_weights.pth"
pso_model_path = "cinic_best_pso_model_weights.pth"



if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()

# Get a batch of test data
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

# Get predictions
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


def display_images(images, labels, predictions, classes):
    fig = plt.figure(figsize=(10, 10))
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1)
        plt.imshow((images[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2)
        true_label = classes[labels[i].item()]
        pred_label = classes[predictions[i].item()]
        ax.set_title(f"True: {true_label}\nPred: {pred_label}")
    plt.tight_layout()
    plt.show()
    
display_images(images, labels, predicted, classes)

'''




# Path to your study database
adam_database_path = "cifar_adam_results_20230826_180611/study.db"
pso_database_path = "cifar_pso_results_20230826_180611/study.db"


adam_study_name = "cifar_adam_study"
adam_storage_name = f"sqlite:///{adam_database_path}"
adam_study = optuna.load_study(study_name=adam_study_name, storage=adam_storage_name)


pso_study_name = "cifar_pso_study"
pso_storage_name = f"sqlite:///{pso_database_path}"
pso_study = optuna.load_study(study_name=pso_study_name, storage=pso_storage_name)





print("Adam study statistics:")
print("Number of finished trials: ", len(adam_study.trials))
print("Number of pruned trials: ", sum([trial.state == optuna.trial.TrialState.PRUNED for trial in adam_study.trials]))
print("Best trial:")
trial = adam_study.best_trial
print(" Value: ", trial.value)
print(" Params: ", trial.params)

print("\nPSO study statistics:")
print("Number of finished trials: ", len(pso_study.trials))
print("Number of pruned trials: ", sum([trial.state == optuna.trial.TrialState.PRUNED for trial in pso_study.trials]))
print("Best trial:")
trial = pso_study.best_trial
print(" Value: ", trial.value)
print(" Params: ", trial.params)


# Extract the loss values (objective values) for both optimizers
adam_losses = [trial.value for trial in adam_study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
pso_losses = [trial.value for trial in pso_study.trials if trial.state == optuna.trial.TrialState.COMPLETE]

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the losses for the Adam optimizer
ax.plot(adam_losses, label='Adam', color='blue')

# Plot the losses for the PSO optimizer
ax.plot(pso_losses, label='PSO', color='red')

# Add labels, title, and a legend
ax.set_xlabel('Trial')
ax.set_ylabel('Loss')
ax.set_title('Comparison of Adam and PSO Optimizers')
ax.legend()

# Show the plot
plt.show()





def plot_padding_optimal_weight(study):
    trials = study.trials_dataframe(attrs=('params', 'value'))

    print(trials.columns)

    fig = px.scatter_3d(
        trials,
        x=trials['params_shift_boundary'],
        y=trials['params_optimal_weight'],
        z=trials['value'],
        title='Validation Loss as a function of Padding and Optimal Weight',
        labels={
            'x': 'Padding',
            'y': 'Optimal Weight',
            'z': 'Validation Loss'
        }
    )
    fig.show()

plot_padding_optimal_weight(pso_study)






def prepare_data(study, optimizer_name):
    trials = study.trials_dataframe(attrs=('params', 'value'))
    trials['Optimizer'] = optimizer_name
    return trials[['params_lr', 'value', 'Optimizer']]

# Prepare the data for both studies
adam_data = prepare_data(adam_study, 'Adam')
pso_data = prepare_data(pso_study, 'PSO')

# Combine the data
combined_data = pd.concat([adam_data, pso_data])

# Create the scatter plot
fig = px.scatter(
    combined_data,
    x='params_lr',
    y='value',
    color='Optimizer',
    title='Validation Loss as a function of Learning Rate',
    labels={
        'params_lr': 'Learning Rate',
        'value': 'Validation Loss'
    }
)
fig.show()



fig = px.histogram(
    combined_data,
    x='params_lr',
    color='Optimizer',
    title='Distribution of Learning Rates',
    labels={
        'params_lr': 'Learning Rate',
    }
)
fig.show()



# Sorting the losses to show progression
adam_losses_sorted = sorted(adam_losses)
pso_losses_sorted = sorted(pso_losses)

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the sorted losses for the Adam optimizer
ax.plot(adam_losses_sorted, label='Adam', color='blue')

# Plot the sorted losses for the PSO optimizer
ax.plot(pso_losses_sorted, label='PSO', color='red')

# Add labels, title, and a legend
ax.set_xlabel('Sorted Trial')
ax.set_ylabel('Loss')
ax.set_title('Comparison of Adam and PSO Optimizers (Sorted Losses)')
ax.legend()

# Show the plot
plt.show()




# Combine losses into a DataFrame for visualization
loss_data = pd.DataFrame({
    'Optimizer': ['Adam'] * len(adam_losses) + ['PSO'] * len(pso_losses),
    'Loss': adam_losses + pso_losses
})

# Create a box plot
sns.boxplot(x='Optimizer', y='Loss', data=loss_data)
plt.title('Distribution of Losses for Adam and PSO Optimizers')
plt.show()







'''







# Print existing trial parameters
#for trial in study.trials:
    #print(trial.params)





# Path to the folder containing the TensorBoard event files
path = 'runs/simulate_cifar'
#print(os.listdir(path))

# Create an accumulator and load the events
ea = event_accumulator.EventAccumulator(path)


# List all the tags available, find the one corresponding to class-wise accuracies
print(ea.Tags())

# Assuming 'class_accuracies' is the correct tag for your data
#class_accuracies = ea.Scalars('class_accuracies')


def plot_parameter_importance(study):
    importance = optuna.importance.get_param_importances(study)
    fig = px.bar(x=list(importance.keys()), y=list(importance.values()), title='Parameter Importances')
    fig.show()

plot_parameter_importance(study)



def plot_contour_plots(study):
    trials = study.trials_dataframe(attrs=('params', 'value'))

    for param1, param2 in [('params_lr', 'params_threshold'), ('params_lr', 'params_A')]:
        fig = go.Figure(data =
            go.Contour(
                z=trials['value'],
                x=trials[f'{param1}'],
                y=trials[f'{param2}']
            )
        )
        fig.update_layout(
            title=f'Contour Plot of {param1[7:]} vs {param2[7:]}',
            xaxis_title=param1[7:],
            yaxis_title=param2[7:],
            autosize=False,
            width=500,
            height=500,
        )
        fig.show()



def plot_padding_optimal_weight(study):
    trials = study.trials_dataframe(attrs=('params', 'value'))

    print(trials.columns)

    fig = px.scatter_3d(
        trials,
        x=trials['params_shift_boundary'],
        y=trials['params_optimal_weight'],
        z=trials['value'],
        title='Validation Loss as a function of Padding and Optimal Weight',
        labels={
            'x': 'Padding',
            'y': 'Optimal Weight',
            'z': 'Validation Loss'
        }
    )
    fig.show()

plot_padding_optimal_weight(study)




def plot_custom_contour(study, param1='f', param2='transition_slope'):
    trials = study.trials_dataframe(attrs=('params', 'value'))
    
    print(trials.columns)

    fig = go.Figure(data =
        go.Contour(
            z=trials['value'],           # Validation loss
            x=trials[f'params_f'], # Parameter "f"
            y=trials[f'params_transition_slope']  # Parameter "transition_slope"
        )
    )
    fig.update_layout(
        title=f'Contour Plot of {param1} vs {param2}',
        xaxis_title=param1,
        yaxis_title=param2,
        autosize=False,
        width=500,
        height=500,
    )
    fig.show()

plot_custom_contour(study)



def plot_C_phi_contour(study):
    trials = study.trials_dataframe(attrs=('params', 'value'))
    fig = go.Figure(data =
        go.Contour(
            z=trials['value'],
            x=trials['params_C'],
            y=trials['params_phi']
        )
    )
    fig.update_layout(
        title='Contour Plot of C vs Phi',
        xaxis_title='C',
        yaxis_title='Phi',
        autosize=False,
        width=500,
        height=500,
    )
    fig.show()

plot_C_phi_contour(study)







# Compute class-wise accuracy
class_accuracies = [accuracy_score((labels == i).cpu().numpy(), (predicted == i).cpu().numpy()) for i in range(10)]

# Plot class-wise accuracy
fig = px.bar(x=classes, y=class_accuracies, title='Class-wise Accuracies')
fig.show()


def plot_loss_curve_from_file(fold_number):
    # Define the path to the CSV file
    fine_tuned_model_path = f"cifar_fine_tuned_models/fold_{fold_number}"
    loss_history_csv_path = os.path.join(fine_tuned_model_path, "loss_history.csv")

    # Read the CSV file into a DataFrame
    loss_history_df = pd.read_csv(loss_history_csv_path)

    # Extract the training and validation loss histories
    training_loss_history = loss_history_df['training_loss']
    validation_loss_history = loss_history_df['validation_loss']

    # Plot the loss histories
    plot_loss_curve(training_loss_history, validation_loss_history)


def plot_loss_curve(training_loss_history, validation_loss_history):
    plt.plot(training_loss_history, label="Training Loss")
    plt.plot(validation_loss_history, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss Over Time")
    plt.show()

# Example usage for fold 1
plot_loss_curve_from_file(fold_number=1)





def plot_optimizer_comparison_line_chart(study):
    trials = study.trials_dataframe(attrs=('params', 'value'))
    optimizer_trials = trials[['params_optimizer_name', 'value']]
    
    plt.figure(figsize=[10, 6])
    sns.lineplot(x=optimizer_trials.index, y='value', hue='params_optimizer_name', data=optimizer_trials)
    plt.title('Comparison of Different Optimizers')
    plt.xlabel('Trial')
    plt.ylabel('Objective Value')
    plt.legend(title='Optimizer')
    plt.show()

def plot_optimizer_comparison_box_plot(study):
    trials = study.trials_dataframe(attrs=('params', 'value'))
    optimizer_trials = trials[['params_optimizer_name', 'value']]
    
    plt.figure(figsize=[10, 6])
    sns.boxplot(x='params_optimizer_name', y='value', data=optimizer_trials)
    plt.title('Box Plot Comparison of Different Optimizers')
    plt.xlabel('Optimizer')
    plt.ylabel('Objective Value')
    plt.show()

# Calling the functions to plot the visuals
plot_optimizer_comparison_line_chart(study)
plot_optimizer_comparison_box_plot(study)


'''
