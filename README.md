# Intersection Project

## Overview
Intersection is a simulation project designed to model a traffic intersection using a dynamic bicycle model for vehicle behavior. The project utilizes Pygame for rendering the environment and vehicles, and implements the Intelligent Driver Model (IDM) for longitudinal control of vehicles.

## Project Structure
The project consists of the following files:


- **vehicle.py**: Defines the `Vehicle` class that implements the dynamic bicycle model. It includes methods for longitudinal control using the Intelligent Driver Model (IDM) and lateral control to follow the lane center defined in the `Road` class.

- **road.py**: Intended to contain the `Road` class and related functionalities.

- **utils/__init__.py**: Contains utility functions that may be used across the project, including helper functions for calculations or data processing.

- **main.py**: Serves as the entry point for the application. It initializes the Pygame environment, creates instances of the `Road` and `Vehicle` classes, and manages the main loop for rendering and updating the simulation.

- **requirements.txt**: Lists the dependencies required for the project, such as Pygame and any other libraries needed for the simulation.

## Setup Instructions
1. Clone the repository or download the project files.
2. Navigate to the project directory.
3. Install the required dependencies using the following command:
   ```
   pip install -r requirements.txt
   ```
4. Run the application using:
   ```
   python main.py
   ```

## Usage
Once the application is running, you will see a graphical representation of the intersection with vehicles following the defined road layout. The vehicles will utilize the dynamic bicycle model for movement and will adhere to the lane markings.

## Contributing
Contributions to the project are welcome. Please feel free to submit issues or pull requests for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.