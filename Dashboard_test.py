import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Function to show the RF options in the top right frame
def show_rf_options():
    # Clear any existing widgets in the top right frame
    for widget in top_right_frame.winfo_children():
        widget.destroy()
    
    # Add the FS_RF and No_FS_RF buttons
    fs_rf_button = tk.Button(top_right_frame, text="RF Feature Selection", font=("Arial", 14), command=show_rf_graph)
    no_fs_rf_button = tk.Button(top_right_frame, text="Without Feature Selection", font=("Arial", 14))
    
    fs_rf_button.pack(side=tk.LEFT, padx=10, pady=10)
    no_fs_rf_button.pack(side=tk.LEFT, padx=10, pady=10)

# Function to show the RF options in the top right frame
def show_svm_options():
    # Clear any existing widgets in the top right frame
    for widget in top_right_frame.winfo_children():
        widget.destroy()
    
    # Add the FS_RF and No_FS_RF buttons
    fs_rf_button = tk.Button(top_right_frame, text="SVM Feature Selection", font=("Arial", 14))
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

def show_rf_graph():
    # Clear any existing widgets in the bottom right frame
    for widget in bottom_right_frame.winfo_children():
        widget.destroy()
    
    # Create a sample graph using Matplotlib
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot([1, 2, 3, 4], [10, 20, 25, 30], label="RF Data")
    ax.set_title("RF Feature Selection Graph")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.legend()

    # Embed the graph in the Tkinter frame
    canvas = FigureCanvasTkAgg(fig, master=bottom_right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Create the main window
root = tk.Tk()
root.title("Frame Example")

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
