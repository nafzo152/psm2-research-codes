import numpy as np
import pandas as pd
import tkinter as tk
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix

data = {
        "RF": {
            "no_feature_selection": {
                "conf_matrix": [[8, 2], [1, 4]], 
                "roc_auc": 0.72, 
                "cv_scores": [0.85, 0.9], 
                "folds": [1, 2]
                },
            "mi": {
                "conf_matrix" : [[11 , 1], [ 0,  3]],
                "roc_auc" : 1.0000,
                "cv_scores" : [1.0, 1.0, 0.90909091, 1.0, 0.90909091],
                "folds" :[1,2,3,4,5]
                },
            "chi": {
                "conf_matrix" : [[12 , 0], [ 0,  3]],
                "roc_auc" : 1.0000,
                "cv_scores" : [1.0, 1.0, 0.90909091, 1.0, 0.90909091],
                "folds" :[1,2,3,4,5]          
            },
            "fs":{
                "conf_matrix" : [[12 , 0], [ 0,  3]],
                "roc_auc" : 1.0000,
                "cv_scores" : [1.0, 1.0, 0.90909091, 1.0, 0.90909091],
                "folds" :[1,2,3,4,5]          
            },
            "ref": {
                "conf_matrix" : [[11 , 0],[ 2,  2]],
                "roc_auc" : 0.9318,
                "cv_scores" : [0.75, 0.5, 0.72727273, 0.90909091, 0.72727273],
                "folds" :[1,2,3,4,5]          
            },
            "ga": {
                "conf_matrix" : [[12 , 0],[ 0,  3]],
                "roc_auc" : 1.0000,
                "cv_scores" : [1.0, 1.0, 0.90909091, 1.0, 0.81818182],
                "folds" :[1,2,3,4,5]          
            }
               },
        "SVM": {
            "no_feature_selection": {
                "conf_matrix": [[7, 3], [2, 3]], 
                "roc_auc": 0.76, 
                "cv_scores": [0.8, 0.78], 
                "folds": [1, 2]
                },
            "mi": {
                "conf_matrix" : [[11 , 1], [ 0,  3]],
                "roc_auc" : 1.0000,
                "cv_scores" :  [0.96551724, 0.96428571],
                "folds" :[1,2]
                },
            "chi": {
                "conf_matrix" : [[12 , 0],[ 0,  3]],
                "roc_auc" : 1.0000,
                "cv_scores" : [0.96551724, 0.96428571],
                "folds" :[1,2]
            },
            "fs":{
                "conf_matrix" : [[11 , 1],[ 0,  3]],
                "roc_auc" : 1.0000,
                "cv_scores" : [0.96551724, 0.96428571],
                "folds" :[1,2]          
            },
            "ref": {
                "conf_matrix" : [[11 , 1],[ 0,  3]],
                "roc_auc" : 1.0000,
                "cv_scores" : [1,1,1,1,1],
                "folds" :[1,2,3,4,5]          
            },
            "ga": {
                "conf_matrix" : [[12 , 0],[ 0,  3]],
                "roc_auc" : 1.0000,
                "cv_scores" : [1.0, 1.0, 0.90909091, 1.0, 1.0],
                "folds" :[1,2,3,4,5]          
            }
        }
    }

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
    top_label = tk.Label(top_right_frame, font=("Arial", 16), bg='lightblue')
    top_label.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
    bottom_label = tk.Label(bottom_right_frame, font=("Arial", 16), bg='lightgreen')
    bottom_label.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)


# Function to reset only the bottom right frame
def reset_bottom_right_frame():
    # Clear any existing widgets in the bottom right frame
    for widget in bottom_right_frame.winfo_children():
        widget.destroy()
    
    # Recreate the default label in the bottom right frame
    bottom_label = tk.Label(bottom_right_frame, font=("Arial", 16), bg='lightgreen')
    bottom_label.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)


# Generalized function to show options for RF and SVM
def show_options(algorithm):
    button_data = {
        "RF": {"fs": "RF Feature Selection", "no_fs": "Without Feature Selection"},
        "SVM": {"fs": "SVM Feature Selection", "no_fs": "Without Feature Selection"}
    }
    
    reset_bottom_right_frame()
    
    # Clear any existing widgets in the top right frame
    for widget in top_right_frame.winfo_children():
        widget.destroy()

    # Create feature selection and no feature selection buttons based on the selected algorithm
    if algorithm in button_data:
        fs_button = tk.Button(top_right_frame, text=button_data[algorithm]["fs"], font=("Arial", 14), 
                              command=lambda: show_btn_fs(algorithm))
        fs_button.pack(side=tk.LEFT, padx=5, pady=10)
        
        no_fs_button = tk.Button(top_right_frame, text=button_data[algorithm]["no_fs"], font=("Arial", 14), 
                                 command=lambda: show_graph(algorithm, "no_feature_selection"))
        no_fs_button.pack(side=tk.LEFT, padx=5, pady=10)
        
        performance_button = tk.Button(top_right_frame, text="Performance Graph", font=("Arial", 14), 
                                       command=lambda: show_performance_graph(algorithm))
        performance_button.pack(side=tk.LEFT, padx=5, pady=10)

def show_graph(algotirhm,feature):
    
    reset_bottom_right_frame()
    # Create a figure with three subplots
    fig = Figure(figsize=(7, 6), dpi=100)
    ax1 = fig.add_subplot(131)  # 3 rows, 1 column, 1st subplot
    ax2 = fig.add_subplot(132)  # 3 rows, 1 column, 2nd subplot
    ax3 = fig.add_subplot(133)  # 3 rows, 1 column, 3rd subplot
    
    # Plot the ROC Curve
    fpr, tpr= [0.0, 0.0, 0.0, 1.0], [0.0, 0.33333333, 1.0, 1.0]
    ax1.plot(fpr, tpr, color='red', lw=2, label=f'ROC Curve (AUC = {data[algotirhm][feature]["roc_auc"]:.2f})')
    ax1.plot([0, 1], [0, 1], color='green', linestyle='--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax1.legend(loc="lower right")
    ax1.legend()
    ax1.set_facecolor('#f0f0f0')
    
    ax2.bar(data[algotirhm][feature]["folds"],data[algotirhm][feature]["cv_scores"], color='red', edgecolor='black')
    ax2.set_title("Cross-Validation Results")
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Score")
    #ax2.legend() #commented out because of warning : "No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument."
    ax2.set_facecolor('#f0f0f0')
    
    #confusion matrix
    print(data[algotirhm][feature]["conf_matrix"])
    sns.heatmap(data[algotirhm][feature]["conf_matrix"], annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=False)
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

def show_btn_fs(algorithm):
    reset_bottom_right_frame()
    
    # Clear any existing widgets in the top right frame
    for widget in top_right_frame.winfo_children():
        widget.destroy()
    
    # Create five buttons for different feature selection techniques
    mi_button = tk.Button(top_right_frame, text="Mutual Information", font=("Arial", 14), command=lambda: show_graph(algorithm,"mi"))
    chi_square_button = tk.Button(top_right_frame, text="Chi-Square", font=("Arial", 14), command=lambda: show_graph(algorithm,"chi"))
    f_score_button = tk.Button(top_right_frame, text="F-Score", font=("Arial", 14), command=lambda: show_graph(algorithm,"fs"))
    rfe_button = tk.Button(top_right_frame, text="RFE", font=("Arial", 14), command=lambda: show_graph(algorithm,"ref"))
    ga_button = tk.Button(top_right_frame, text="Genetic Algorithm", font=("Arial", 14), command=lambda: show_graph(algorithm,"ga"))
    
    # Pack the buttons into the top right frame
    mi_button.pack(side=tk.LEFT, padx=5, pady=10)
    chi_square_button.pack(side=tk.LEFT, padx=5, pady=10)
    f_score_button.pack(side=tk.LEFT, padx=5, pady=10)
    rfe_button.pack(side=tk.LEFT, padx=5, pady=10)
    ga_button.pack(side=tk.LEFT, padx=5, pady=10)

    
# General function to show performance graphs based on the algorithm
def show_performance_graph(algorithm):
    reset_bottom_right_frame()
    
    # Example accuracy data
    techniques = ["All Features", "Mutual Information", "Chi-Square", "F-Score", "RFE", "Genetic Algorithm"]
    performance_data = {
        "RF": [86.67, 93.33, 100, 93.33, 86.67, 100],
        "SVM": [93.33, 100, 93.33, 93.33, 93.33, 100]
    }
    
    accuracies = performance_data[algorithm]
    
    # Create a figure for performance graph
    fig = Figure(figsize=(7, 5), dpi=100)
    ax = fig.add_subplot(111)
    
    # Plotting the accuracies
    ax.plot(techniques, accuracies, marker='o', color='blue', linewidth=2, markersize=8)
    
    # Set the title and labels
    ax.set_title(f"{algorithm} Performance with Different Feature Selection Techniques")
    ax.set_xlabel("Techniques")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(80, 105)  # Adjust y-axis range
    
    # Add accuracy values on top of the points
    for i, acc in enumerate(accuracies):
        ax.text(i, acc + 0.5, f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
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
RF = tk.Button(left_frame, text="RF", font=("Arial", 14), command= lambda: show_options("RF"))
RF.pack(pady=5, padx=10, fill=tk.BOTH, anchor="n")

SVM = tk.Button(left_frame, text="SVM", font=("Arial", 14), command=lambda: show_options("SVM"))
SVM.pack(pady=5, padx=10, fill=tk.BOTH, anchor="n")

RESET = tk.Button(left_frame, text="Reset", font=("Arial", 14), command=reset_window)
RESET.pack(pady=5, padx=10, fill=tk.BOTH, anchor="n")

# Create labels in the subframes
top_label = tk.Label(top_right_frame, text="", font=("Arial", 16), bg='lightblue')
top_label.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
bottom_label = tk.Label(bottom_right_frame, text="", font=("Arial", 16), bg='lightgreen')
bottom_label.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

# Start the Tkinter main loop
root.mainloop()