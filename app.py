
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

# User inputs
m = st.slider('Mass (m)', 0.1, 100.0, 1.0)
k = st.slider('Spring Constant (k)', 1, 100000, 1)
c = st.slider('Damping Coefficient (c)', 0, 500, 0)
omega = st.slider('Driving Frequency (ω)', 0.1, np.sqrt(k/m), 0.1)
P = st.slider('Driving Force (P)', 0, 500, 100)

# Initial conditions
y0 = [0, 0]  # Starting displacement (maximum amplitude)

# Solve using forward Euler method with initial values
t, y = forward_euler(f, 80, 20000, y0, m, k, c, omega, P)

# Plotting
fig, ax = plt.subplots()
ax.plot(t, y[:, 0], label='Displacement')
ax.set_xlabel('Time')
ax.set_ylabel('Displacement')
ax.set_title('Displacement vs Time')
ax.grid()
st.pyplot(fig)
