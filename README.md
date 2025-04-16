# RHINO XR3 Robotic Arm Inverse Kinematics using Neural Networks #

This project implements an inverse kinematics solution for the RHINO XR3 robotic arm using deep neural networks. Instead of relying on traditional analytical methods, this approach uses machine learning to model the complex relationship between end-effector positions/orientations and the corresponding joint angles.

## Overview ##

The inverse kinematics problem involves determining the joint angles required to position a robot's end-effector at a desired location with a specific orientation. This project tackles this challenge by:

1. Generating a comprehensive dataset of joint angles and their corresponding end-effector positions and orientations
2. Training a neural network to learn the mapping from end-effector state to joint angles
3. Providing a model that can predict the required joint angles for new desired positions

## Features ##

* Complete implementation of the RHINO XR3 forward kinematics using Denavit-Hartenberg parameters
* Dataset generation across the robot's full joint space
* Neural network architecture optimized for inverse kinematics learning
* Normalization for improved training stability
* Model saving and loading functionality for practical deployment
* Utility function for easy prediction of joint angles from end-effector states

## Results ##

The neural network demonstrates promising performance in predicting joint angles for the RHINO XR3 robotic arm. The comparison between target and predicted angles shows relatively small errors across all five joints. The first joint angle prediction (-101.36° vs -105°) has about a 3.5% error, while the second through fifth joints show errors of approximately 1.8°, 1.9°, 3.6°, and 3.3° respectively.
These small deviations indicate that the neural network has successfully captured the inverse kinematics relationship between end-effector position/orientation and joint angles. For robotics applications, these prediction errors are within an acceptable range for many tasks, though further refinement could improve precision for applications requiring higher accuracy. Overall, the neural network approach provides an effective alternative to traditional analytical inverse kinematics solutions for the RHINO XR3 robotic arm.
