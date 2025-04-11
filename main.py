# %%
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %%
# Simulate noisy data from two sensors
np.random.seed(42)
true_position = np.linspace(0, 10, 100)  # True position in 1D

# Sensor 1: Low frequency, high accuracy
s1_time = np.linspace(0, 10, 20)  # Low frequency
s1_measurements = np.interp(s1_time, true_position, true_position) + np.random.normal(0, 0.1, len(s1_time))

# Sensor 2: High frequency, low accuracy
s2_time = np.linspace(0, 10, 100)  # High frequency
s2_measurements = true_position + np.random.normal(0, 0.5, len(s2_time))

# %%
# Simple averaging to denoise sensor measurements
average_position = np.interp(s2_time, s1_time, s1_measurements) * 0.5 + s2_measurements * 0.5

# %%
# Kalman Filter implementation
class KalmanFilter:
    def __init__(self, process_variance, measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0
        self.error_covariance = 1

    def update(self, measurement):
        kalman_gain = self.error_covariance / (self.error_covariance + self.measurement_variance)
        self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
        self.error_covariance = (1 - kalman_gain) * self.error_covariance

    def predict(self):
        self.error_covariance += self.process_variance
        return self.estimate

kf = KalmanFilter(process_variance=0.1, measurement_variance=0.5)
kalman_positions = []
for measurement in s2_measurements:
    kf.predict()
    kf.update(measurement)
    kalman_positions.append(kf.estimate)

# %%
# Calculate errors
kalman_error = np.array(kalman_positions) - true_position
s1_error = np.interp(s2_time, s1_time, s1_measurements) - true_position
s2_error = s2_measurements - true_position

# %%
# Calculate Mean Squared Errors (MSE)
kalman_mse = np.mean(kalman_error**2)
s1_mse = np.mean(s1_error**2)
s2_mse = np.mean(s2_error**2)

# Create subplots
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                     subplot_titles=("Position Estimation", "Estimation Error"))

# Add position estimation traces
fig.add_trace(go.Scatter(x=np.linspace(0, 10, 100), y=true_position, mode='lines', name='True Position', line=dict(dash='dash')),
              row=1, col=1)
fig.add_trace(go.Scatter(x=s1_time, y=s1_measurements, mode='markers', name='Sensor 1 (Low Freq, High Acc)', marker=dict(color='red')),
              row=1, col=1)
fig.add_trace(go.Scatter(x=s2_time, y=s2_measurements, mode='markers', name='Sensor 2 (High Freq, Low Acc)', marker=dict(color='blue', opacity=0.5)),
              row=1, col=1)
fig.add_trace(go.Scatter(x=s2_time, y=average_position, mode='lines', name='Simple Average', line=dict(color='green')),
              row=1, col=1)
fig.add_trace(go.Scatter(x=s2_time, y=kalman_positions, mode='lines', name='Kalman Filter', line=dict(color='orange')),
              row=1, col=1)

# Add error traces with MSE in legend labels
fig.add_trace(go.Scatter(x=np.linspace(0, 10, 100), y=kalman_error, mode='lines', name=f'Kalman Error (MSE={kalman_mse:.2f})', line=dict(color='purple')),
              row=2, col=1)
fig.add_trace(go.Scatter(x=s2_time, y=s1_error, mode='lines', name=f'Sensor 1 Error (MSE={s1_mse:.2f})', line=dict(color='red', dash='dot')),
              row=2, col=1)
fig.add_trace(go.Scatter(x=s2_time, y=s2_error, mode='lines', name=f'Sensor 2 Error (MSE={s2_mse:.2f})', line=dict(color='blue', dash='dot')),
              row=2, col=1)

# Update layout
fig.update_layout(
    title="State Estimation in 1D",
    xaxis_title="Time",
    yaxis_title="Position",
    xaxis2_title="Time",
    yaxis2_title="Error",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    template="plotly_white"
)

fig.show()


# %%
# Calculate velocity for each sensor and Kalman filter
true_velocity = np.gradient(true_position, s2_time)
s1_velocity = np.gradient(np.interp(s2_time, s1_time, s1_measurements), s2_time)
s2_velocity = np.gradient(s2_measurements, s2_time)
kalman_velocity = np.gradient(kalman_positions, s2_time)

# Calculate velocity errors
s1_velocity_error = s1_velocity - true_velocity
s2_velocity_error = s2_velocity - true_velocity
kalman_velocity_error = kalman_velocity - true_velocity

# Calculate Mean Squared Errors (MSE) for velocity
s1_velocity_mse = np.mean(s1_velocity_error**2)
s2_velocity_mse = np.mean(s2_velocity_error**2)
kalman_velocity_mse = np.mean(kalman_velocity_error**2)

# Create subplots for velocity
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                     subplot_titles=("Velocity Estimation", "Velocity Estimation Error"))

# Add velocity estimation traces
fig.add_trace(go.Scatter(x=s2_time, y=true_velocity, mode='lines', name='True Velocity', line=dict(dash='dash')),
              row=1, col=1)
fig.add_trace(go.Scatter(x=s2_time, y=s1_velocity, mode='lines', name='Sensor 1 Velocity', line=dict(color='red')),
              row=1, col=1)
fig.add_trace(go.Scatter(x=s2_time, y=s2_velocity, mode='lines', name='Sensor 2 Velocity', line=dict(color='blue')),
              row=1, col=1)
fig.add_trace(go.Scatter(x=s2_time, y=kalman_velocity, mode='lines', name='Kalman Velocity', line=dict(color='orange')),
              row=1, col=1)

# Add velocity error traces
fig.add_trace(go.Scatter(x=s2_time, y=s1_velocity_error, mode='lines', name=f'Sensor 1 Velocity Error (MSE={s1_velocity_mse:.2f})', line=dict(color='red', dash='dot')),
              row=2, col=1)
fig.add_trace(go.Scatter(x=s2_time, y=s2_velocity_error, mode='lines', name=f'Sensor 2 Velocity Error (MSE={s2_velocity_mse:.2f})', line=dict(color='blue', dash='dot')),
              row=2, col=1)
fig.add_trace(go.Scatter(x=s2_time, y=kalman_velocity_error, mode='lines', name=f'Kalman Velocity Error (MSE={kalman_velocity_mse:.2f})', line=dict(color='purple')),
              row=2, col=1)

# Update layout
fig.update_layout(
    title="Velocity Estimation in 1D",
    xaxis_title="Time",
    yaxis_title="Velocity",
    xaxis2_title="Time",
    yaxis2_title="Error",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    template="plotly_white"
)

fig.show()
