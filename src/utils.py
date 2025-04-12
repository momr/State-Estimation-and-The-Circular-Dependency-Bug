import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_position_estimation(true_position, s1_time, s1_measurements, s2_time, s2_measurements, average_position, kalman_positions, kalman_error, s1_error, s2_error, kalman_mse, s1_mse, s2_mse, simple_average_error, simple_average_error_mse):
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
        title="Position Estimation in 1D",
        xaxis_title="Time",
        yaxis_title="Position",
        xaxis2_title="Time",
        yaxis2_title="Error",
        # legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        template="plotly_white"
    )

    fig.show()

def plot_velocity_estimation(true_velocity, s1_velocity, s2_velocity, kalman_velocity, s1_velocity_error, s2_velocity_error, kalman_velocity_error, s1_velocity_mse, s2_velocity_mse, kalman_velocity_mse, s2_time):
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
        title="Velocity Estimation in 1D",
        xaxis_title="Time",
        yaxis_title="Velocity",
        xaxis2_title="Time",
        yaxis2_title="Error",
        # legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        template="plotly_white"
    )

    fig.show()
