import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time

# Function to plot a single frame
def plot_frame(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y, color="blue", lw=2)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("Animated Sine Wave")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    return fig

# Main Streamlit app
st.title("Animated Plot with Streamlit")

# Generate Data
x = np.linspace(0, 2 * np.pi, 100)
frames = 100  # Number of animation frames

# Create a placeholder for the plot
placeholder = st.empty()

# Animation loop
for i in range(1, frames + 1):
    y = np.sin(x[:i])  # Generate sine wave for current frame
    fig = plot_frame(x[:i], y)  # Plot the current frame
    placeholder.pyplot(fig)  # Display the plot in Streamlit
    time.sleep(0.1)  # Pause to simulate animation
