# %%
import numpy as np
from utils import plot_position_estimation, plot_velocity_estimation


# Simulate noisy data from two sensors
np.random.seed(42)
true_position = np.linspace(0, 10, 100)  # True position in 1D

# Sensor 1: Low frequency, high accuracy
s1_time = np.linspace(0, 10, 20)  # Low frequency
s1_measurements = np.interp(s1_time, true_position, true_position) + np.random.normal(0, 0.1, len(s1_time))

# Sensor 2: High frequency, low accuracy
s2_time = np.linspace(0, 10, 100)  # High frequency
s2_measurements = true_position + np.random.normal(0, 0.5, len(s2_time))

# Simple averaging to denoise sensor measurements
average_position = np.interp(s2_time, s1_time, s1_measurements) * 0.5 + s2_measurements * 0.5

# Calculate errors
s1_error = np.interp(s2_time, s1_time, s1_measurements) - true_position
s2_error = s2_measurements - true_position

# Calculate Mean Squared Errors (MSE)
s1_mse = np.mean(s1_error**2)
s2_mse = np.mean(s2_error**2)

# Calculate velocity for each sensor and Kalman filter
true_velocity = np.gradient(true_position, s2_time)
s1_velocity = np.gradient(np.interp(s2_time, s1_time, s1_measurements), s2_time)
s2_velocity = np.gradient(s2_measurements, s2_time)

# Calculate velocity errors
s1_velocity_error = s1_velocity - true_velocity
s2_velocity_error = s2_velocity - true_velocity

# Calculate Mean Squared Errors (MSE) for velocity
s1_velocity_mse = np.mean(s1_velocity_error**2)
s2_velocity_mse = np.mean(s2_velocity_error**2)


# %% Update and Predict (The right way)
# Kalman Filter implementation with velocity input for prediction
class KalmanFilter3:
    def __init__(self, process_variance, measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate_pos = 0
        self.estimate_vel = 0
        self.error_covariance = 1

    def update(self, measurement):
        kalman_gain = self.error_covariance / (self.error_covariance + self.measurement_variance)
        estimate_pos_new = self.estimate_pos + kalman_gain * (measurement - self.estimate_pos)
        self.estimate_vel = (estimate_pos_new - self.estimate_pos)
        self.estimate_pos = estimate_pos_new
        self.error_covariance = (1 - kalman_gain) * self.error_covariance

    def predict(self):
        self.estimate_pos = self.estimate_pos + self.estimate_vel
        self.error_covariance += self.process_variance
        return self.estimate_pos

# %%
kf = KalmanFilter3(process_variance=0.1, measurement_variance=0.5)
kalman_positions4 = []
kalman_velocity4 = []

for measurement in s2_measurements:
    kf.update(measurement)
    kf.predict()
    kalman_positions4.append(kf.estimate_pos)
    kalman_velocity4.append(kf.estimate_vel)

# Calculate errors
kalman_error4 = np.array(kalman_positions4) - true_position

# Calculate Mean Squared Errors (MSE)
kalman_mse = np.mean(kalman_error4**2)

# Plot position estimation and errors
plot_position_estimation(
    true_position, s1_time, s1_measurements, s2_time, s2_measurements,
    average_position, kalman_positions4, kalman_error4, s1_error, s2_error,
    kalman_mse, s1_mse, s2_mse
)

# Calculate velocity errors
kalman_velocity_error = kalman_velocity4 - true_velocity

# Calculate Mean Squared Errors (MSE) for velocity
kalman_velocity_mse = np.mean(kalman_velocity_error**2)

# Plot velocity estimation and errors
plot_velocity_estimation(
    true_velocity, s1_velocity, s2_velocity, kalman_velocity4,
    s1_velocity_error, s2_velocity_error, kalman_velocity_error,
    s1_velocity_mse, s2_velocity_mse, kalman_velocity_mse, s2_time
)

# %% Update and Predict (The ??? way)
# Kalman Filter implementation with velocity input for prediction
class KalmanFilter4:
    def __init__(self, process_variance, measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate_pos = 0
        self.error_covariance = 1

    def update(self, measurement):
        kalman_gain = self.error_covariance / (self.error_covariance + self.measurement_variance)
        estimate_pos_new = self.estimate_pos + kalman_gain * (measurement - self.estimate_pos)
        self.estimate_pos = estimate_pos_new
        self.error_covariance = (1 - kalman_gain) * self.error_covariance

    def predict(self, velocity_measurement):
        self.estimate_pos = self.estimate_pos + velocity_measurement
        self.error_covariance += self.process_variance
        return self.estimate_pos

# %%
kf = KalmanFilter4(process_variance=0.1, measurement_variance=0.5)
kalman_positions5 = []
kalman_velocity5 = []

for measurement in s2_measurements:
    old_pos = kf.estimate_pos
    kf.predict(kalman_velocity5[-1] if kalman_velocity5 else 0)
    kf.update(measurement)
    kalman_positions5.append(kf.estimate_pos)
    kalman_velocity5.append(kf.estimate_pos - old_pos)

# Calculate errors
kalman_error5 = np.array(kalman_positions5) - true_position

# Calculate Mean Squared Errors (MSE)
kalman_mse = np.mean(kalman_error5**2)

# Plot position estimation and errors
plot_position_estimation(
    true_position, s1_time, s1_measurements, s2_time, s2_measurements,
    average_position, kalman_positions4, kalman_error5, s1_error, s2_error,
    kalman_mse, s1_mse, s2_mse
)

# Calculate velocity errors
kalman_velocity_error = kalman_velocity5 - true_velocity

# Calculate Mean Squared Errors (MSE) for velocity
kalman_velocity_mse = np.mean(kalman_velocity_error**2)

# Plot velocity estimation and errors
plot_velocity_estimation(
    true_velocity, s1_velocity, s2_velocity, kalman_velocity5,
    s1_velocity_error, s2_velocity_error, kalman_velocity_error,
    s1_velocity_mse, s2_velocity_mse, kalman_velocity_mse, s2_time
)

# %% Update and Predict (The right way)
class ImprovedKalmanFilter:
    def __init__(self, process_variance, measurement_variance):
        # State vector [position, velocity]
        self.x = np.array([0.0, 0.0])
        
        # Error covariance matrix
        self.P = np.array([[1.0, 0.0],
                           [0.0, 1.0]])
        
        # State transition matrix
        self.F = np.array([[1.0, 1.0],
                           [0.0, 1.0]])
        
        # Measurement matrix (we only measure position)
        self.H = np.array([[1.0, 0.0]])
        
        # Process noise covariance
        self.Q = np.array([[0.25*process_variance, 0.5*process_variance],
                           [0.5*process_variance, process_variance]])
        
        # Measurement noise
        self.R = np.array([[measurement_variance]])
    
    def predict(self, dt=1.0):
        # Update transition matrix for current time step
        self.F[0, 1] = dt
        
        # Predict state
        self.x = self.F @ self.x
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x[0]  # Return position estimate
    
    def update(self, measurement):
        # Calculate Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state estimate
        y = np.array([measurement]) - self.H @ self.x
        self.x = self.x + K @ y
        
        # Update error covariance
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x[0]  # Return updated position
    

# Calculate time steps
dt = s2_time[1] - s2_time[0]  # Assuming uniform time steps

kf = ImprovedKalmanFilter(process_variance=0.1, measurement_variance=0.5)
kalman_positions9 = []
kalman_velocities9 = []

for measurement in s2_measurements:
    kf.predict(dt)  # First predict
    kf.update(measurement)  # Then update with measurement
    kalman_positions9.append(kf.x[0])  # Store position
    kalman_velocities9.append(kf.x[1])  # Store velocity

# Calculate errors
kalman_error9 = np.array(kalman_positions9) - true_position

# Calculate Mean Squared Errors (MSE)
kalman_mse = np.mean(kalman_error9**2)

# Plot position estimation and errors
plot_position_estimation(
    true_position, s1_time, s1_measurements, s2_time, s2_measurements,
    average_position, kalman_positions4, kalman_error9, s1_error, s2_error,
    kalman_mse, s1_mse, s2_mse
)

# Calculate velocity errors
kalman_velocity_error = kalman_velocities9 - true_velocity

# Calculate Mean Squared Errors (MSE) for velocity
kalman_velocity_mse = np.mean(kalman_velocity_error**2)

# Plot velocity estimation and errors
plot_velocity_estimation(
    true_velocity, s1_velocity, s2_velocity, kalman_velocities9,
    s1_velocity_error, s2_velocity_error, kalman_velocity_error,
    s1_velocity_mse, s2_velocity_mse, kalman_velocity_mse, s2_time
)