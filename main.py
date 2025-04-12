# %%
import numpy as np
from utils import plot_position_estimation, plot_velocity_estimation


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

# %% Using Update only
# Simple averaging to denoise sensor measurements
average_position = np.interp(s2_time, s1_time, s1_measurements) * 0.5 + s2_measurements * 0.5

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

# Plot position estimation and errors
plot_position_estimation(
    true_position, s1_time, s1_measurements, s2_time, s2_measurements,
    average_position, kalman_positions, kalman_error, s1_error, s2_error,
    kalman_mse, s1_mse, s2_mse
)

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

# Plot velocity estimation and errors
plot_velocity_estimation(
    true_velocity, s1_velocity, s2_velocity, kalman_velocity,
    s1_velocity_error, s2_velocity_error, kalman_velocity_error,
    s1_velocity_mse, s2_velocity_mse, kalman_velocity_mse, s2_time
)

# %% Using Predict Only
# Kalman Filter implementation with velocity input for prediction
class KalmanFilter2:
    def __init__(self, process_variance, measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0
        self.error_covariance = 1

    def update(self, measurement):
        kalman_gain = self.error_covariance / (self.error_covariance + self.measurement_variance)
        self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
        self.error_covariance = (1 - kalman_gain) * self.error_covariance

    def predict(self, velocity_measurement):
        self.estimate += velocity_measurement  # Incorporate velocity into position prediction
        self.error_covariance += self.process_variance
        return self.estimate
    
kf = KalmanFilter2(process_variance=0.1, measurement_variance=0.5)
kalman_positions2 = []

for velocity in s2_velocity:
    kf.predict(velocity)
    kalman_positions2.append(kf.estimate)

# Calculate errors
kalman_error2 = np.array(kalman_positions2) - true_position

# %%
# Calculate Mean Squared Errors (MSE)
kalman_mse = np.mean(kalman_error2**2)

# %%
# Plot position estimation and errors
plot_position_estimation(
    true_position, s1_time, s1_measurements, s2_time, s2_measurements,
    average_position, kalman_positions2, kalman_error2, s1_error, s2_error,
    kalman_mse, s1_mse, s2_mse
)

# %%
# Calculate velocity for each sensor and Kalman filter
kalman_velocity2 = np.gradient(kalman_positions2, s2_time)

# Calculate velocity errors
kalman_velocity_error = kalman_velocity2 - true_velocity

# Calculate Mean Squared Errors (MSE) for velocity
kalman_velocity_mse = np.mean(kalman_velocity_error**2)

# Plot velocity estimation and errors
plot_velocity_estimation(
    true_velocity, s1_velocity, s2_velocity, kalman_velocity2,
    s1_velocity_error, s2_velocity_error, kalman_velocity_error,
    s1_velocity_mse, s2_velocity_mse, kalman_velocity_mse, s2_time
)

# %% Update and Predict (The wrong way)
# Kalman Filter implementation with velocity input for prediction
class KalmanFilter3:
    def __init__(self, process_variance, measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0
        self.error_covariance = 1

    def update(self, measurement):
        kalman_gain = self.error_covariance / (self.error_covariance + self.measurement_variance)
        self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
        self.error_covariance = (1 - kalman_gain) * self.error_covariance

    def predict(self, velocity_measurement):
        self.estimate += velocity_measurement  # Incorporate velocity into position prediction
        self.error_covariance += self.process_variance
        return self.estimate
    
kf = KalmanFilter3(process_variance=0.1, measurement_variance=0.5)
kalman_positions3 = []

for measurement, velocity in zip(s2_measurements, s2_velocity):
    kf.update(measurement)
    kf.predict(velocity)
    kalman_positions3.append(kf.estimate)

# %%
# Calculate errors
kalman_error3 = np.array(kalman_positions3) - true_position


# %%
# Calculate Mean Squared Errors (MSE)
kalman_mse = np.mean(kalman_error3**2)

# %%
# Plot position estimation and errors
plot_position_estimation(
    true_position, s1_time, s1_measurements, s2_time, s2_measurements,
    average_position, kalman_positions3, kalman_error3, s1_error, s2_error,
    kalman_mse, s1_mse, s2_mse
)

# %%
# Calculate velocity for each sensor and Kalman filter
kalman_velocity3 = np.gradient(kalman_positions3, s2_time)

# Calculate velocity errors
kalman_velocity_error = kalman_velocity3 - true_velocity

# Calculate Mean Squared Errors (MSE) for velocity
kalman_velocity_mse = np.mean(kalman_velocity_error**2)

# Plot velocity estimation and errors
plot_velocity_estimation(
    true_velocity, s1_velocity, s2_velocity, kalman_velocity3,
    s1_velocity_error, s2_velocity_error, kalman_velocity_error,
    s1_velocity_mse, s2_velocity_mse, kalman_velocity_mse, s2_time
)


# %% Update and Predict (The right way)
# Kalman Filter implementation with velocity input for prediction
class KalmanFilter4:
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
        self.error_covariance += self.process_variance
        return self.estimate_pos
    
kf = KalmanFilter4(process_variance=0.1, measurement_variance=0.5)
kalman_positions4 = []
kalman_velocity4 = []

for measurement in s2_measurements:
    kf.update(measurement)
    kf.predict()
    kalman_positions4.append(kf.estimate_pos)
    kalman_velocity4.append(kf.estimate_vel)

# %%
# Calculate errors
kalman_error4 = np.array(kalman_positions4) - true_position

# %%
# Calculate Mean Squared Errors (MSE)
kalman_mse = np.mean(kalman_error4**2)

# %%
# Plot position estimation and errors
plot_position_estimation(
    true_position, s1_time, s1_measurements, s2_time, s2_measurements,
    average_position, kalman_positions4, kalman_error4, s1_error, s2_error,
    kalman_mse, s1_mse, s2_mse
)

# %%
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
