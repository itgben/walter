import os
import math
import matplotlib.pyplot as plt
import numpy as np
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
from scipy.interpolate import griddata
from scipy.optimize import least_squares


from tensorboard.backend.event_processing import event_accumulator


# Path to your study database
adam_database_path = "/home/ben/Desktop/trials/cinic_adam_results_20230831_100902/study.db"
pso_database_path = "/home/ben/Desktop/trials/cinic_pso_results_20230831_100902/study.db"


adam_study_name = "cinic_adam_study"
adam_storage_name = f"sqlite:///{adam_database_path}"
adam_study = optuna.load_study(study_name=adam_study_name, storage=adam_storage_name)


pso_study_name = "cinic_pso_study"
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





def plot_with_lines(trials):
    # Define your 3D scatter plot
    scatter = go.Scatter3d(
        x=trials['params_shift_boundary'],
        y=trials['params_optimal_weight'],
        z=trials['value'],
        mode='markers',
        marker=dict(
            size=6,
            color=trials['value'],  # set color to an array/list of desired values
            colorscale='Viridis',
        )
    )

    # Sorting values to form a line
    trials_sorted = trials.sort_values(by='value')

    # Define your lines
    line = go.Scatter3d(
        x=trials_sorted['params_shift_boundary'],
        y=trials_sorted['params_optimal_weight'],
        z=trials_sorted['value'],
        mode='lines',
        line=dict(
            width=3,
            color='red',
        )
    )

    # Combine scatter plot and line
    fig = go.Figure(data=[scatter, line])

    # Set labels and title
    fig.update_layout(
        title='Validation Loss as a function of Padding and Optimal Weight',
        scene=dict(
            xaxis_title='Padding',
            yaxis_title='Optimal Weight',
            zaxis_title='Validation Loss'
        )
    )
    
    # Show the plot
    fig.show()

# Fetch the trials dataframe from the study
pso_trials = pso_study.trials_dataframe(attrs=('params', 'value'))

# Plot the 3D scatter points and lines
plot_with_lines(pso_trials)









def plot_parallel_coordinates(trials):
    # Creating the DataFrame for plotting
    df = trials[['params_shift_boundary', 'params_optimal_weight', 'value']]
    df.columns = ['Padding', 'Optimal Weight', 'Validation Loss']  # Renaming columns for better readability
    
    # Create the 3D parallel coordinates plot
    fig = px.parallel_coordinates(
        df,
        labels={
            'Padding': 'Padding',
            'Optimal Weight': 'Optimal Weight',
            'Validation Loss': 'Validation Loss'
        },
        color='Validation Loss',
        color_continuous_scale=px.colors.diverging.Tealrose,
        color_continuous_midpoint=2
    )
    
    # Show the plot
    fig.show()

# Fetch the trials dataframe from the study (PSO for example)
pso_trials = pso_study.trials_dataframe(attrs=('params', 'value'))

# Create the 3D parallel coordinates plot
plot_parallel_coordinates(pso_trials)

















def plot_surface(trials):
    # Define the coordinate points for the grid
    x = trials['params_shift_boundary']
    y = trials['params_optimal_weight']
    z = trials['value']
    
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    
    # Interpolate the points to fit to a grid
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    
    # Create the surface plot
    fig = go.Figure(data=[go.Surface(z=zi, x=xi, y=yi)])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Padding',
            yaxis_title='Optimal Weight',
            zaxis_title='Validation Loss'
        ),
        title='3D Surface Plot for Validation Loss'
    )
    
    # Show the plot
    fig.show()

# Fetch the trials dataframe from the study (PSO for example)
pso_trials = pso_study.trials_dataframe(attrs=('params', 'value'))

# Create the 3D surface plot
plot_surface(pso_trials)














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

















def plot_multidimensional(trials):
    # Create a list of trial numbers based on the length of the trials DataFrame
    trial_numbers = list(range(len(trials)))

    scatter = go.Scatter3d(
        x=trial_numbers,
        y=trials['params_optimal_weight'],
        z=trials['params_transition_slope'],
        mode='markers',
        marker=dict(
            size=6,
            color=trials['value'],  # using validation loss as the color scale
            colorscale='Viridis',
        )
    )

    fig = go.Figure(data=[scatter])
    
    fig.update_layout(
        title='3D Scatter Plot for Trial Number, Optimal Weight, and Transition Slope',
        scene=dict(
            xaxis_title='Trial Number',
            yaxis_title='Optimal Weight',
            zaxis_title='Transition Slope'
        )
    )
    
    fig.show()

# Fetch the trials dataframe from the study
multidimensional_trials = pso_study.trials_dataframe(attrs=('number', 'params', 'value'))

# Create the 3D scatter plot
plot_multidimensional(multidimensional_trials)














threshold_value = 0.1  # Define your threshold
optimal_trials = pso_trials[pso_trials['value'] < threshold_value]


x = optimal_trials['params_optimal_weight'].to_numpy()
y = optimal_trials['params_transition_slope'].to_numpy()
z = optimal_trials.index.to_numpy()  # trial number can serve as the third dimension





# Check if x, y, z are empty
if len(x) == 0 or len(y) == 0 or len(z) == 0:
    print("Data arrays are empty. Skipping the plotting.")
else:
    def residuals(params, x, y, z):
        x0, y0, z0, r = params
        return (x - x0)**2 + (y - y0)**2 + (z - z0)**2 - r**2

    initial_guess = [0, 0, 0, 1]  # x0, y0, z0, r
    result = least_squares(residuals, initial_guess, args=(x, y, z))
    x0, y0, z0, r = result.x





    phi = np.linspace(0, np.pi, 30)
    theta = np.linspace(0, 2 * np.pi, 30)
    phi, theta = np.meshgrid(phi, theta)

    # The Cartesian coordinates of the sphere
    X = x0 + r * np.sin(phi) * np.cos(theta)
    Y = y0 + r * np.sin(phi) * np.sin(theta)
    Z = z0 + r * np.cos(phi)

    # Create a scatter plot for the optimal points and a mesh plot for the sphere
    fig = go.Figure(data=[
        go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=6, color='blue')),
        go.Mesh3d(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), opacity=0.5, color='red')
    ])

    fig.show()








optimal_trials_filtered = pso_trials[pso_trials['value'] < threshold_value]

# Extract the relevant parameters for the x, y, and z axes
x = optimal_trials_filtered['params_optimal_weight'].to_numpy()
y = optimal_trials_filtered['params_phase_shift_loss_weight'].to_numpy()
z = optimal_trials_filtered['params_transition_slope'].to_numpy()


# Check if x, y, z are empty
if len(x) == 0 or len(y) == 0 or len(z) == 0:
    print("Data arrays are empty. Skipping the plotting.")
else:

    # Create the figure object
    fig = go.Figure()

    # Add a scatter plot for the points
    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+lines',
            marker=dict(size=6, color='blue'),
            line=dict(color='blue', width=2)
        )
    )

    # Update the layout to include axis titles and overall title
    fig.update_layout(
        scene=dict(
            xaxis_title='Optimal Weight',
            yaxis_title='Phase Shift Loss Weight',
            zaxis_title='Transition Slope',
            xaxis=dict(range=[min(x) - 0.5, max(x) + 0.5]),
            yaxis=dict(range=[min(y) - 0.5, max(y) + 0.5]),
            zaxis=dict(range=[min(z) - 0.5, max(z) + 0.5])
        ),
        title='3D Scatter Plot of Optimal Parameters'
    )

    # Show the figure
    fig.show()






    # Define the residual function
    def residuals(params, x, y, z):
        x0, y0, z0, r = params
        return (x - x0)**2 + (y - y0)**2 + (z - z0)**2 - r**2

    # Initial guess for the sphere's center and radius
    initial_guess = [0, 0, 0, 1]
    result = least_squares(residuals, initial_guess, args=(x, y, z))
    x0, y0, z0, r = result.x

    # Generate points for the sphere surface
    phi = np.linspace(0, np.pi, 30)
    theta = np.linspace(0, 2 * np.pi, 30)
    phi, theta = np.meshgrid(phi, theta)
    X_sphere = x0 + r * np.sin(phi) * np.cos(theta)
    Y_sphere = y0 + r * np.sin(phi) * np.sin(theta)
    Z_sphere = z0 + r * np.cos(phi)

    # Create the figure
    fig = go.Figure()

    # Scatter plot for the data points
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=6, color='blue')))

    # Mesh plot for the sphere
    fig.add_trace(go.Mesh3d(x=X_sphere.flatten(), y=Y_sphere.flatten(), z=Z_sphere.flatten(), opacity=0.5, color='red'))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='Optimal Weight',
            yaxis_title='Phase Shift Loss Weight',
            zaxis_title='Transition Slope',
            xaxis=dict(range=[min(x) - 0.5, max(x) + 0.5]),
            yaxis=dict(range=[min(y) - 0.5, max(y) + 0.5]),
            zaxis=dict(range=[min(z) - 0.5, max(z) + 0.5])
        ),
        title='3D Scatter Plot of Optimal Parameters with Fitted Sphere'
    )

    # Show the figure
    fig.show()














#---------------




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
