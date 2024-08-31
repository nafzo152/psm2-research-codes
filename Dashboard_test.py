import numpy as np
import pandas as pd
import tkinter as tk
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix



# Function to show the RF options in the top right frame
def show_rf_options():
    reset_bottom_right_frame()
    # Clear any existing widgets in the top right frame
    for widget in top_right_frame.winfo_children():
        widget.destroy()
    
    # Add the FS_RF and No_FS_RF buttons
    fs_rf_button = tk.Button(top_right_frame, text="RF Feature Selection", font=("Arial", 14), command=show_rffs_graph)
    no_fs_rf_button = tk.Button(top_right_frame, text="Without Feature Selection", font=("Arial", 14))
    
    fs_rf_button.pack(side=tk.LEFT, padx=10, pady=10)
    no_fs_rf_button.pack(side=tk.LEFT, padx=10, pady=10)

# Function to show the RF options in the top right frame
def show_svm_options():
    reset_bottom_right_frame()
    # Clear any existing widgets in the top right frame
    for widget in top_right_frame.winfo_children():
        widget.destroy()
    
    # Add the FS_RF and No_FS_RF buttons
    fs_rf_button = tk.Button(top_right_frame, text="SVM Feature Selection", font=("Arial", 14), command=show_svmfs_graph)
    no_fs_rf_button = tk.Button(top_right_frame, text="Without Feature Selection", font=("Arial", 14))
    
    fs_rf_button.pack(side=tk.LEFT, padx=10, pady=10)
    no_fs_rf_button.pack(side=tk.LEFT, padx=10, pady=10)


# Function to reset the window to its initial state
def reset_window():
    # Clear any existing widgets in the right frame
    for widget in right_frame.winfo_children():
        widget.destroy()

    # Recreate the top and bottom frames
    global top_right_frame, bottom_right_frame
    top_right_frame = tk.Frame(right_frame, bg='lightblue')
    bottom_right_frame = tk.Frame(right_frame, bg='lightgreen')

    top_right_frame.grid(row=0, column=0, sticky='ew')
    bottom_right_frame.grid(row=1, column=0, sticky='nsew')

    # Add the default labels back
    top_label = tk.Label(top_right_frame, text="Top Right Frame", font=("Arial", 16), bg='lightblue')
    top_label.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
    bottom_label = tk.Label(bottom_right_frame, text="Bottom Right Frame", font=("Arial", 16), bg='lightgreen')
    bottom_label.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

# Function to reset the window to its initial state
def reset_bottom_right_frame():
    # Clear any existing widgets in the bottom right frame
    for widget in bottom_right_frame.winfo_children():
        widget.destroy()
    
    # Recreate the default label in the bottom right frame
    bottom_label = tk.Label(bottom_right_frame, text="Bottom Right Frame", font=("Arial", 16), bg='lightgreen')
    bottom_label.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
    
def show_rffs_graph():
    reset_bottom_right_frame()
    
    # Calculate confusion matrix and ROC AUC score
    conf_matrix = [[10,  0], [ 0,  5]]
    roc_auc = 0.7778

    # Sample cross-validation scores
    cv_scores = [0.93333333, 0.8, 0.85714286, 0.85714286, 0.92857143]
    folds = [1,2,3,4,5]

    # Clear any existing widgets in the bottom right frame
    for widget in bottom_right_frame.winfo_children():
        widget.destroy()
    
    # Create a figure with three subplots
    fig = Figure(figsize=(7, 6), dpi=100)
    ax1 = fig.add_subplot(131)  # 3 rows, 1 column, 1st subplot
    ax2 = fig.add_subplot(132)  # 3 rows, 1 column, 2nd subplot
    ax3 = fig.add_subplot(133)  # 3 rows, 1 column, 3rd subplot
    
    # Plot the ROC Curve
    fpr, tpr= [0.0 ,0.0 ,1.0],[0.0   ,0.75 ,1.0  ]
    ax1.plot(fpr, tpr, color='red', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='green', linestyle='--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax1.legend(loc="lower right")
    ax1.legend()
    ax1.set_facecolor('#f0f0f0')
    
    ax2.bar(folds, cv_scores, color='red', edgecolor='black')
    ax2.set_title("Cross-Validation Results")
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Score")
    #ax2.legend() #commented out because of warning : "No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument."
    ax2.set_facecolor('#f0f0f0')
    
    #confusion matrix
    print(conf_matrix)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=False)
    ax3.set_title("Confusion Matrix")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("True")
    ax3.set_facecolor('#f0f0f0')
    
    # Adjust layout
    fig.tight_layout()

    # Embed the figure in the Tkinter frame
    canvas = FigureCanvasTkAgg(fig, master=bottom_right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
def show_svmfs_graph():
    reset_bottom_right_frame()
    
    # Calculate confusion matrix and ROC AUC score
    conf_matrix = [[10,0], [ 3,  2]]
    roc_auc = 0.8125

    # Sample cross-validation scores
    cv_scores = [0.77777778, 0.72222222]
    folds = [1,2]

    # Clear any existing widgets in the bottom right frame
    for widget in bottom_right_frame.winfo_children():
        widget.destroy()
    
    # Create a figure with three subplots
    fig = Figure(figsize=(7, 6), dpi=100)
    ax1 = fig.add_subplot(131)  # 3 rows, 1 column, 1st subplot
    ax2 = fig.add_subplot(132)  # 3 rows, 1 column, 2nd subplot
    ax3 = fig.add_subplot(133)  # 3 rows, 1 column, 3rd subplot
    
    # Plot the ROC Curve
    fpr, tpr= [0.0, 0.0, 1.0],[0.0,    0.625, 1.0   ]
    ax1.plot(fpr, tpr, color='red', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='green', linestyle='--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax1.legend(loc="lower right")
    ax1.legend()
    ax1.set_facecolor('#f0f0f0')
    
    ax2.bar(folds, cv_scores, color='red', edgecolor='black')
    ax2.set_title("Cross-Validation Results")
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Score")
    #ax2.legend() #commented out because of warning : "No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument."
    ax2.set_facecolor('#f0f0f0')
    
    #confusion matrix
    print(conf_matrix)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=False)
    ax3.set_title("Confusion Matrix")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("True")
    ax3.set_facecolor('#f0f0f0')
    
    # Adjust layout
    fig.tight_layout()

    # Embed the figure in the Tkinter frame
    canvas = FigureCanvasTkAgg(fig, master=bottom_right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


# Create the main window
root = tk.Tk()
root.title("Dashboard")

# Set the fixed size of the window
root.geometry("800x600")  # Width x Height

# Create the left frame for menu buttons
left_frame = tk.Frame(root, bg='lightgrey')
left_frame.grid(row=0, column=0, sticky='ns', padx=0, pady=0)

# Create the right frame for display
right_frame = tk.Frame(root, bg='white')
right_frame.grid(row=0, column=1, sticky='nsew', pady=3, padx=5,)

# Create two subframes within the right frame
top_right_frame = tk.Frame(right_frame, bg='lightblue')
bottom_right_frame = tk.Frame(right_frame, bg='lightgreen')

top_right_frame.grid(row=0, column=0, sticky='ew')
bottom_right_frame.grid(row=1, column=0, sticky='nsew')

# Configure the row weights of the right_frame
right_frame.grid_rowconfigure(1, weight=1)
right_frame.grid_columnconfigure(0, weight=1)

# Configure the left_frame to not expand
left_frame.grid_rowconfigure(0, weight=0)
left_frame.grid_columnconfigure(0, weight=0)

# Configure the main window grid to handle resizing
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)


# Create two buttons in the left frame
RF = tk.Button(left_frame, text="RF", font=("Arial", 14), command= show_rf_options)
RF.pack(pady=5, padx=10, fill=tk.BOTH, anchor="n")

SVM = tk.Button(left_frame, text="SVM", font=("Arial", 14), command=show_svm_options)
SVM.pack(pady=5, padx=10, fill=tk.BOTH, anchor="n")

RESET = tk.Button(left_frame, text="Reset", font=("Arial", 14), command=reset_window)
RESET.pack(pady=5, padx=10, fill=tk.BOTH, anchor="n")

# Create labels in the subframes
top_label = tk.Label(top_right_frame, text="Top Right Frame", font=("Arial", 16), bg='lightblue')
top_label.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
bottom_label = tk.Label(bottom_right_frame, text="Bottom Right Frame", font=("Arial", 16), bg='lightgreen')
bottom_label.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

# Start the Tkinter main loop
root.mainloop()
