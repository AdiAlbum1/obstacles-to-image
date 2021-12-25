# Robotic Motion Planning - Final Project

## Table of Contents

- [Task](#task)
- [Installation](#installation)
- [Project Description](#project_description)
- [Usage](#usage)

## Task
Sampling based approaches for robotic motion planning suffer from the ability to find a valid path in scenes with critical narrow passageways. My project proposal is a machine-learning based solution recommending a sample point in the scene’s narrow passageway.
A convolutional neural network which reads as input the image describing the scene: including its source, target and obstacles, and outputs two parameters describing the recommended sample point in the narrow passageway.
To keep the project contained we’ll focus on a single disc robot translating in the plane.

## Installation
```sh
git clone https://github.com/AdiAlbum1/robotic-motion-planning-final-project
cd robotic-motion-planning-final-project
pip install -r requirments.txt
```

## Project Description
- ```python stage_1.py ``` : Reads as input the JSON files from ```input_json_obstacles/``` contatining the scene's description, and generates binary images of the scene's obstacles
- ```python augment_image.py ``` : Generates augmentations of the obstacle images

## Usage
TBD