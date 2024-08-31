import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# Load data from CSV
df_wofs_rf_cm = pd.read_csv('./predictions.csv')

# Extract the Y_test and Y_pred columns
data_wofs_rf_cm = [df_wofs_rf_cm['Y_test'].values, df_wofs_rf_cm['Y_pred'].values]

# Calculate confusion matrix and ROC AUC score
conf_matrix = confusion_matrix(data_wofs_rf_cm[0], data_wofs_rf_cm[1])
roc_auc = roc_auc_score(data_wofs_rf_cm[0], data_wofs_rf_cm[1])

# Sample cross-validation scores
cv_scores = [0.93333333, 0.8, 0.85714286, 0.85714286, 0.92857143]

# Create a figure and axes for the plots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Adjusted figure size for better spacing

# Add a button to the figure
button_ax = plt.axes([0.4, 0.01, 0.2, 0.05])  # Button position (left, bottom, width, height)
button = Button(button_ax, 'Without FS', color='lightblue', hovercolor='lightgreen')

def update_plots(event):
    # Clear existing plots
    for ax in axs:
        ax.clear()
    
    # Recalculate confusion matrix and ROC AUC score
    conf_matrix = confusion_matrix(data_wofs_rf_cm[0], data_wofs_rf_cm[1])
    roc_auc = roc_auc_score(data_wofs_rf_cm[0], data_wofs_rf_cm[1])
    
    # Plot the Confusion Matrix
    im = axs[0].imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    axs[0].figure.colorbar(im, ax=axs[0])
    axs[0].set(xticks=np.arange(conf_matrix.shape[1]),
               yticks=np.arange(conf_matrix.shape[0]),
               xticklabels=np.arange(conf_matrix.shape[1]),
               yticklabels=np.arange(conf_matrix.shape[0]),
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
    axs[0].set_ylim(conf_matrix.shape[0] - 0.5, -0.5)  # Reverse y-axis to have (0, 0) in top-left
    
    # Plot the ROC Curve
    fpr, tpr, _ = roc_curve(data_wofs_rf_cm[0], data_wofs_rf_cm[1])
    axs[1].plot(fpr, tpr, color='red', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    axs[1].plot([0, 1], [0, 1], color='green', linestyle='--')
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].set_title('Receiver Operating Characteristic (ROC) Curve')
    axs[1].legend(loc="lower right")
    
    # Plot the Cross-Validation Scores
    bar_width = 0.4
    index = np.arange(len(cv_scores))
    axs[2].bar(index, cv_scores, bar_width, color='skyblue')
    axs[2].set_xlabel('Fold')
    axs[2].set_ylabel('Accuracy')
    axs[2].set_title('Cross-validation Scores')
    axs[2].set_xticks(index)
    axs[2].set_xticklabels([f'Fold {i+1}' for i in index])
    axs[2].set_ylim(0, 1.1)  # Limit y-axis for better visualization
    axs[2].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    fig.canvas.draw()  # Redraw the figure to update plots

# Connect the button to the update_plots function
button.on_clicked(update_plots)

# Show the button and plot
plt.show()
