import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Generating x and y-values
x = np.arange(0, 1, 0.02)
y = x

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.3)
p, = plt.plot(x, y, color='red')
ax.title.set_text('Graph for y = x')

# Home button
axButn1 = plt.axes([0.1, 0.1, 0.1, 0.1])
btn1 = Button(
    axButn1, label="Home", color='pink', hovercolor='tomato')


# To plot a graph for y = x
def plot1(event):
    p.set_xdata(x)
    p.set_ydata(x)
    ax.title.set_text('Graph for y = x')
    plt.draw()


btn1.on_clicked(plot1)

# Previous button
axButn2 = plt.axes([0.3, 0.1, 0.1, 0.1])
btn2 = Button(
    axButn2, label="Prev", color='pink', hovercolor='tomato')


# To plot a graph for y = x**2
def plot2(event):
    p.set_xdata(x)
    p.set_ydata(x ** 2)
    ax.title.set_text('Graph for y = x**2')
    plt.draw()


btn2.on_clicked(plot2)

# Next button
axButn3 = plt.axes([0.5, 0.1, 0.1, 0.1])
btn3 = Button(
    axButn3, label="Next", color='pink', hovercolor='tomato')


# To plot a graph for y = 2x
def plot3(event):
    p.set_xdata(x)
    p.set_ydata(2 * x)
    ax.title.set_text('Graph for y = 2x')
    plt.draw()


btn3.on_clicked(plot3)
plt.show()