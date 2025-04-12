import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # State Estimation Educational Project

        This project demonstrates state estimation concepts using Python. It includes examples of tracking a 1D position using noisy sensor data and comparing different estimation techniques, such as simple averaging and Kalman filtering. The project is designed to be accessible to non-technical audiences and provides interactive visualizations to build intuition about state estimation.

        In this project I want to talk about the following:

        1. A simple average can beat a poorly implemented Kalman Filter
        1. Introducing new kind of Bugs: The Circular Dependency Bug in this case

        How to use this project:

        1. Try to find all the issues in the first two Kalman Filter classes/implementations
        1. Try to fix all the issues
        1. See AI fix at the end and compare it to your solution
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def plot_position_estimation(true_position, s1_time, s1_measurements, s2_time, s2_measurements, average_position, kalman_positions, kalman_error, s1_error, s2_error, kalman_mse, s1_mse, s2_mse, simple_average_error, simple_average_error_mse, title="Position Estimation in 1D"):
        """Plots position estimation and errors."""
        # Create subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                             subplot_titles=("Position Estimation", "Estimation Error"))

        # Add position estimation traces
        fig.add_trace(go.Scatter(x=s2_time, y=true_position, mode='lines', name='True Position', line=dict(dash='dash')),
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
        fig.add_trace(go.Scatter(x=s2_time, y=s1_error, mode='lines', name=f'Sensor 1 Error (MSE={s1_mse:.2f})', line=dict(color='red', dash='dot')),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=s2_time, y=s2_error, mode='lines', name=f'Sensor 2 Error (MSE={s2_mse:.2f})', line=dict(color='blue', dash='dot')),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=s2_time, y=simple_average_error, mode='lines', name=f'Simple Average Error (MSE={simple_average_error_mse:.2f})', line=dict(color='green', dash='dot')),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=s2_time, y=kalman_error, mode='lines', name=f'Kalman Error (MSE={kalman_mse:.2f})', line=dict(color='purple')),
                      row=2, col=1)

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Position",
            xaxis2_title="Time",
            yaxis2_title="Error",
            # legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            template="plotly_white"
        )

        fig.show()

    def plot_velocity_estimation(true_velocity, s1_velocity, s2_velocity, kalman_velocity, s1_velocity_error, s2_velocity_error, kalman_velocity_error, s1_velocity_mse, s2_velocity_mse, kalman_velocity_mse, s2_time, title="Velocity Estimation in 1D"):
        """Plots velocity estimation and errors."""
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
        fig.add_trace(go.Scatter(x=s2_time, y=kalman_velocity_error, mode='lines', name=f'Kalman Velocity Error (MSE={kalman_velocity_mse:.2f})', line=dict(color='purple')),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=s2_time, y=s1_velocity_error, mode='lines', name=f'Sensor 1 Velocity Error (MSE={s1_velocity_mse:.2f})', line=dict(color='red', dash='dot')),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=s2_time, y=s2_velocity_error, mode='lines', name=f'Sensor 2 Velocity Error (MSE={s2_velocity_mse:.2f})', line=dict(color='blue', dash='dot')),
                      row=2, col=1)

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Velocity",
            xaxis2_title="Time",
            yaxis2_title="Error",
            # legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            template="plotly_white"
        )

        fig.show()
    return (
        go,
        make_subplots,
        plot_position_estimation,
        plot_velocity_estimation,
    )


@app.cell(hide_code=True)
def _(plot_position_estimation, plot_velocity_estimation):
    import numpy as np


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
    simple_average_error = average_position - true_position

    # Calculate Mean Squared Errors (MSE)
    s1_mse = np.mean(s1_error**2)
    s2_mse = np.mean(s2_error**2)
    simple_average_error_mse = np.mean(simple_average_error**2)

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
    class KalmanFilter1:
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

    kf = KalmanFilter1(process_variance=0.1, measurement_variance=0.5)
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
        kalman_mse, s1_mse, s2_mse, simple_average_error, simple_average_error_mse, "Position Estimation in 1D (First Implementation)"
    )

    # Calculate velocity errors
    kalman_velocity_error = kalman_velocity4 - true_velocity

    # Calculate Mean Squared Errors (MSE) for velocity
    kalman_velocity_mse = np.mean(kalman_velocity_error**2)

    # Plot velocity estimation and errors
    plot_velocity_estimation(
        true_velocity, s1_velocity, s2_velocity, kalman_velocity4,
        s1_velocity_error, s2_velocity_error, kalman_velocity_error,
        s1_velocity_mse, s2_velocity_mse, kalman_velocity_mse, s2_time, "Velocity Estimation in 1D (First Implementation)"
    )

    # %% Update and Predict (The ??? way)
    # Kalman Filter implementation with velocity input for prediction
    class KalmanFilter2:
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

    kf = KalmanFilter2(process_variance=0.1, measurement_variance=0.5)
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
        kalman_mse, s1_mse, s2_mse, simple_average_error, simple_average_error_mse, "Position Estimation in 1D (Second Implementation)"
    )

    # Calculate velocity errors
    kalman_velocity_error = kalman_velocity5 - true_velocity

    # Calculate Mean Squared Errors (MSE) for velocity
    kalman_velocity_mse = np.mean(kalman_velocity_error**2)

    # Plot velocity estimation and errors
    plot_velocity_estimation(
        true_velocity, s1_velocity, s2_velocity, kalman_velocity5,
        s1_velocity_error, s2_velocity_error, kalman_velocity_error,
        s1_velocity_mse, s2_velocity_mse, kalman_velocity_mse, s2_time, "Velocity Estimation in 1D (Second Implementation)"
    )
    return (
        KalmanFilter1,
        KalmanFilter2,
        average_position,
        kalman_error4,
        kalman_error5,
        kalman_mse,
        kalman_positions4,
        kalman_positions5,
        kalman_velocity4,
        kalman_velocity5,
        kalman_velocity_error,
        kalman_velocity_mse,
        kf,
        measurement,
        np,
        old_pos,
        s1_error,
        s1_measurements,
        s1_mse,
        s1_time,
        s1_velocity,
        s1_velocity_error,
        s1_velocity_mse,
        s2_error,
        s2_measurements,
        s2_mse,
        s2_time,
        s2_velocity,
        s2_velocity_error,
        s2_velocity_mse,
        simple_average_error,
        simple_average_error_mse,
        true_position,
        true_velocity,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## AI Solution""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""See Conversation here: https://claude.ai/share/ca7ce754-f8f7-4b5f-9330-10cf09ad7d29""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        > The following section was generated using the following prompt and attaching the two Kalman Filter class implementations into a ma∈.pymain.py file to the conversation

        Human: attached there are two kalman filter classes. please read them carefully and let me know any issues with both. i.e. which one is correct if any? also, how to fix the problems?
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Analysis of Kalman Filter Implementations

        After examining the two Kalman filter implementations, I've found several issues with both versions. Let me explain the problems and how to fix them.

        ## KalmanFilter3 Analysis

        `python
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
                self.estimate_vel = (estimate_pos_new - self.estimate_pos)  # Issue here
                self.estimate_pos = estimate_pos_new
                self.error_covariance = (1 - kalman_gain) * self.error_covariance

            def predict(self):
                self.estimate_pos = self.estimate_pos + self.estimate_vel
                self.error_covariance += self.process_variance
                return self.estimate_pos
        ```

        **Issues with KalmanFilter3:**

        1. Velocity calculation doesn't account for time steps - it assumes dt=1
        2. Uses scalar error covariance instead of a proper 2×2 matrix for position/velocity
        3. The velocity is calculated as a byproduct of the update step rather than being properly estimated

        ## KalmanFilter4 Analysis

        ```python
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

            def predict(self, velocity_measurement):  # Issue here - taking velocity as input
                self.estimate_pos = self.estimate_pos + velocity_measurement
                self.error_covariance += self.process_variance
                return self.estimate_pos
        ```

        **Issues with KalmanFilter4:**

        1. It doesn't actually estimate velocity - it takes it as an external input
        2. <mark>The circular dependency in usage: velocity is calculated outside the filter, then fed back in</mark>
        3. Also uses scalar error covariance instead of a matrix

        ## Which One Is More Correct?

        Neither implementation is fully correct for a proper Kalman filter. KalmanFilter3 is closer to the right approach because it at least attempts to estimate velocity internally, but it has significant limitations.

        ## Corrected Implementation

        Here's a proper implementation that fixes these issues:

        ```python
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
        ```

        **Key improvements:**

        1. Properly tracks a state vector with both position and velocity
        2. Uses appropriate matrix operations for state transition
        3. Handles the error covariance as a matrix, not a scalar
        4. Accounts for time steps between measurements
        5. Follows the standard predict-then-update cycle

        To use this improved filter:

        ```python
        # Calculate time steps
        dt = s2_time[1] - s2_time[0]  # Assuming uniform time steps

        kf = ImprovedKalmanFilter(process_variance=0.1, measurement_variance=0.5)
        kalman_positions = []
        kalman_velocities = []

        for measurement in s2_measurements:
            kf.predict(dt)  # First predict
            kf.update(measurement)  # Then update with measurement
            kalman_positions.append(kf.x[0])  # Store position
            kalman_velocities.append(kf.x[1])  # Store velocity
        ```

        This implementation follows proper Kalman filter theory and should give you more accurate results for both position and velocity estimation.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    ## AI Solution Results
    return


@app.cell(hide_code=True)
def _(
    average_position,
    kalman_positions4,
    np,
    plot_position_estimation,
    plot_velocity_estimation,
    s1_error,
    s1_measurements,
    s1_mse,
    s1_time,
    s1_velocity,
    s1_velocity_error,
    s1_velocity_mse,
    s2_error,
    s2_measurements,
    s2_mse,
    s2_time,
    s2_velocity,
    s2_velocity_error,
    s2_velocity_mse,
    simple_average_error,
    simple_average_error_mse,
    true_position,
    true_velocity,
):
    def _():
        # %% The AI Solution
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
            kalman_mse, s1_mse, s2_mse, simple_average_error, simple_average_error_mse, "Position Estimation in 1D (AI Implementation)"
        )

        # Calculate velocity errors
        kalman_velocity_error = kalman_velocities9 - true_velocity

        # Calculate Mean Squared Errors (MSE) for velocity
        kalman_velocity_mse = np.mean(kalman_velocity_error**2)

        # Plot velocity estimation and errors
        return plot_velocity_estimation(
            true_velocity, s1_velocity, s2_velocity, kalman_velocities9,
            s1_velocity_error, s2_velocity_error, kalman_velocity_error,
            s1_velocity_mse, s2_velocity_mse, kalman_velocity_mse, s2_time, "Velocity Estimation in 1D (AI Implementation)"
        )


    _()
    return


if __name__ == "__main__":
    app.run()
