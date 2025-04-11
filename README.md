# State Estimation Educational Project

This project demonstrates state estimation concepts using Python. It includes examples of tracking a 1D position using noisy sensor data and comparing different estimation techniques, such as simple averaging and Kalman filtering. The project is designed to be accessible to non-technical audiences and provides interactive visualizations to build intuition about state estimation.

## Features
- Simulates noisy data from two sensors with different characteristics.
- Implements simple averaging and Kalman filtering for state estimation.
- Visualizes results using Plotly for interactive exploration.

## Requirements
- Python 3.8+
- Required Python libraries:
  - `numpy`
  - `plotly`
  - `uv`

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies using the `uv` package manager:
   ```bash
   uv install
   ```

## Usage
1. Run the main script to simulate and visualize state estimation:
   ```bash
   python main.py
   ```

2. Explore the interactive Plotly visualization to compare the true position, sensor measurements, and estimation results.

## Project Structure
```
state_estimation_edu/
├── main.py        # Main script for state estimation example
├── spec.md        # Project specification and goals
└── README.md      # Project overview and instructions
```

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests to improve the project.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
