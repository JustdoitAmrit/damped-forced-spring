import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Define the function for the system of ODEs
def f(y, t, m, k, c, omega, P):
    func = [y[1], P * np.cos(omega * t) + (-k / m) * y[0] - (c / m) * y[1]]
    return np.array(func)

# Runge-Kutta 4th order method
def forward_euler(f, T, N, y0, m, k, c, omega, P):
    dt = T / N
    t = np.zeros(N + 1)
    y = np.zeros((N + 1, 2))

    y[0] = y0
    for n in range(N):
        k1 = f(y[n], t[n], m, k, c, omega, P)
        k2 = f(y[n] + (dt / 2) * k1, t[n] + dt / 2, m, k, c, omega, P)
        k3 = f(y[n] + (dt / 2) * k2, t[n] + dt / 2, m, k, c, omega, P)
        k4 = f(y[n] + dt * k3, t[n] + dt, m, k, c, omega, P)
        t[n + 1] = t[n] + dt
        y[n + 1] = y[n] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, y

# Streamlit app
st.title("Spring-Mass System with Variable Parameters")

# Create two columns: one for sliders and one for the graph
col1, col2 = st.columns([1, 2])  # Adjust the proportions as needed

# User inputs in the first column
with col1:
    st.header("Parameters")
    
    # Mass input
    m = st.slider('Mass (m)', 0.1, 100.0, 1.0, format="%.1f", key='mass')
    m_input = st.text_input('Enter Mass (m)', value=str(m), key='mass_input')
    m = float(m_input) if m_input else m  # Update mass if input is provided
    
    # Spring constant input
    k = st.slider('Spring Constant (k)', 1, 100000, 1, format="%d", key='spring_constant')
    k_input = st.text_input('Enter Spring Constant (k)', value=str(k), key='spring_constant_input')
    k = int(k_input) if k_input else k  # Update spring constant if input is provided
    
    # Damping coefficient input
    c = st.slider('Damping Coefficient (c)', 0.0, 10.0, 0.0, step=0.05, format="%.2f", key='damping_coefficient')
    c_input = st.text_input('Enter Damping Coefficient (c)', value=str(c), key='damping_coefficient_input')
    c = float(c_input) if c_input else c  # Update damping coefficient if input is provided
    
    # Driving frequency input
    omega = st.slider('Driving Frequency (ω)', 0.1, 10.0, 0.1, format="%.1f", key='frequency')
    omega_input = st.text_input('Enter Driving Frequency (ω)', value=str(omega), key='frequency_input')
    omega = float(omega_input) if omega_input else omega  # Update frequency if input is provided
    
    # Driving force input
    P = st.slider('Driving Force (P)', 0, 500, 100, format="%d", key='driving_force')
    P_input = st.text_input('Enter Driving Force (P)', value=str(P), key='driving_force_input')
    P = int(P_input) if P_input else P  # Update driving force if input is provided

# Initial conditions
y0 = [0, 0]  # Starting displacement (maximum amplitude)

# Solve using forward Euler method with initial values
t, y = forward_euler(f, 80, 20000, y0, m, k, c, omega, P)

# Plotting in the second column
with col2:
    fig, ax = plt.subplots()
    ax.plot(t, y[:, 0], label='Displacement')
    ax.set_xlabel('Time')
    ax.set_ylabel('Displacement')
    ax.set_title('Displacement vs Time')
    ax.grid()
    st.pyplot(fig)

