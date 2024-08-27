import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

# Sample data sets
data1 = [['hamid', 'nadim', 'neo', 'ana', 'david'], [10, 25, 55.5, 91.0, 18.5]]
data2 = [['hamida', 'nilima', 'lux', 'Loid', 'dona'], [20, 24, 95.6, 51.02, 28.9]]
data3 = [['alpha', 'bravo', 'charlie', 'delta', 'echo'], [55, 45, 85, 75, 35]]

# Create initial data
x = data1[0]
y = data1[1]

# Create a simple plot
fig, ax = plt.subplots()
line, = ax.plot(x, y)

# Function to update plot data
def update_plot_data(new_x, new_y):
    line.set_xdata(new_x)
    line.set_ydata(new_y)
    ax.relim()  # Recalculate limits
    ax.autoscale_view()  # Autoscale the view
    fig.canvas.draw_idle()

# Callback functions for each button
def on_button1_click(event):
    update_plot_data(data1[0], data1[1])
    print("Data 1 Button clicked!")

def on_button2_click(event):
    update_plot_data(data2[0], data2[1])
    print("Data 2 Button clicked!")

def on_button3_click(event):
    update_plot_data(data3[0], data3[1])
    print("Data 3 Button clicked!")

# Create button axes and buttons
button1_ax = plt.axes([0.1, 0.05, 0.2, 0.05])  # (left, bottom, width, height)
button2_ax = plt.axes([0.4, 0.05, 0.2, 0.05])
button3_ax = plt.axes([0.7, 0.05, 0.2, 0.05])

button1 = Button(button1_ax, 'Data 1', color='lightblue', hovercolor='lightgreen')
button2 = Button(button2_ax, 'Data 2', color='lightblue', hovercolor='lightgreen')
button3 = Button(button3_ax, 'Data 3', color='lightblue', hovercolor='lightgreen')

# Connect the buttons to their callback functions
button1.on_clicked(on_button1_click)
button2.on_clicked(on_button2_click)
button3.on_clicked(on_button3_click)

# Adjust the layout manually to avoid overlapping
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)  # Leave space at the bottom for the buttons

plt.show()

