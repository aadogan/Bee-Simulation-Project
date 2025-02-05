**Project Objective**

This project aims to create an environment that simulates the environmental interactions of bees. The bee simulation provides a platform to understand bees' interactions with flowers, energy management, and environmental conditions. Additionally, this simulation, integrated with a language model (LLM), adds an interactive narrative layer that describes the bees' conditions and environment. This enables users to better understand the behaviors and environmental interactions of bees.

----------------------------------------

How It Works

The project has been developed using the Python programming language and various libraries. Below are the main components and functionality of the project:

Custom Environment (Gymnasium): The project creates a bee simulation environment using the OpenAI Gym library. This environment includes a grid area where the bee can move and interact with flowers. The bee can move up, down, left, right, and express the surrounding conditions with the "dance" action.

Action and Observation Spaces: The bee interacts with surrounding flowers by taking specific actions (moving or dancing). The observation space includes the bee's position and the position of the nearest flower. This information is used in the model's decision-making process.

PPO Model (Stable-Baselines3): The project uses the PPO (Proximal Policy Optimization) algorithm from the Stable-Baselines3 library to enable the bee to learn its behaviors. The model learns the best actions by receiving rewards throughout the simulation.

LLM Integration: The project is integrated with a language model (LLM). When the bee performs specific actions, the LLM generates texts describing the bee's condition and environment. This adds an interactive narrative layer to the simulation.

Visualization (Pygame and OpenGL): The simulation is visualized using Pygame and OpenGL. The bee, flowers, and grid area are represented visually. Additionally, the right panel displays statistics about the bee's condition and performance.


------------------------------------------

Installation

You can use the following commands to install the required libraries:

pip install gymnasium stable-baselines3 pygame ctransformers PyOpenGL PyOpenGL_accelerate


-------------------------------------------
**Statistic**

Steps: Represents the total number of movements or update steps performed in the simulation. Each action (a decision made by the model) is counted as one step. This value can be used to show how long the simulation has been running and how many times the model has been updated.

Total Reward: Shows the total sum of all rewards obtained by the model during the simulation. At each step, a certain reward point is received based on the action taken in the environment; these rewards are accumulated to give the total reward value. A high total reward indicates that the model has learned the task well.

Energy: Shows the amount of energy the bee has in the simulation. Usually, energy decreases with each movement or specific actions. If the energy falls below a certain threshold, the simulation may end or the bee may need to take other actions. This value can be considered a metric that reflects the bee's current "condition."

Efficiency: Typically used as a percentage to evaluate the model's performance. This metric is found by calculating the ratio of the average reward received per step to the ideal maximum reward. For example, if the maximum expected reward per step in an ideal situation is 10, and the model earns an average of 5 rewards, efficiency is approximately 50%. This value can be used as an indicator of how "efficiently" the model is working.
