# Robotic Motion Planning - Final Project

## Table of Contents

- [Task](#task)
- [Installation](#installation)
- [Project Description](#project_description)
- [Training Procedure](#training_procedure)
- [Usage](#usage)

## Task
Sampling based approaches for robotic motion planning suffer from the ability to find a valid path in scenes with critical narrow passageways. My project proposal is a machine-learning based solution recommending a sample point in the scene’s narrow passageway.
A deep Convolutional Neural Network which reads as input the image describing the scene: including its source, target and obstacles, and outputs four parameters determening the bounding box describing the narrow passageway.
To keep the project contained we’ll focus on a single disc robot translating in the plane, but we'll supply solutions for both multiple discs and also a rod translating and rotating in the plane.

## Installation
```sh
git clone https://github.com/AdiAlbum1/robotic-motion-planning-final-project
cd robotic-motion-planning-final-project
pip install pipenv
pipenv install
pipenv shell
```

## Project Description
- ```python generate_base_obstacle_images.py ``` : Reads as input the JSON files from ```input_json_obstacles/``` contatining the scene's description, and generates binary images of the scene's obstacles
- ``` python train.py``` : Trains a Deep Neural Network for the above regression task
- ``` DiscoPygal/mrmp/solvers/dlprm_discs.py``` and ``` DiscoPygal/rod/solvers/dlprm_rod.py``` : Our product. An improved PRM implementation where points are sampled using the narrow passageway outputed by the CNN. These are to be used by running CGAL's simulators ```python DiscoPygal/mrmp/mrmp_main.py``` and ```python DiscoPygal/rod/rod_main.py``` respectiverly and loading the solvers.

## Training Procedure
### Data
1. Randomly select a base obstacle map with a single critical passageway.
   <br>We manually generated multiple base obstacles maps with single critical passageways using CGAL's scene designer.
   The JSON files describing the scenes were tranformed to images using ```generate_base_obstacle_images.py```
   The generated scenes contain various obstacle maps with vertical passageways centered at positions (0,0), (1,0), ..., (9,0)
   as in the following image
    * (4,0):
    <br>![(4,0) - Example](samples/base_(4,0).png)

2. Randomly translate the obstacle map along the y-axis
    * A slight translation up along the y-axis
    <br>![(4,0) - Example translated](samples/base_(4,0)\_translated_1.png)
    * A slight translation down along the y-axis
    <br>![(4,0) - Example translated](samples/base_(4,0)\_translated_2.png)

3. Randomly rotate the image around its center with {0, 90, 180, 270} degrees
    * A rotation by 90 degrees
    <br>![(4,0) - Example rotated](samples/base_(4,0)\_rotated_1.png)
    * A rotation by 180 degrees
    <br>![(4,0) - Example rotated](samples/base_(4,0)\_rotated_2.png)

4. Randomly scattered additional obstacles
    * ![(4,0) - With scattered obstacles](samples/base_(4,0)\_obstacles_1.png)
    * ![(4,0) - With scattered obstacles](samples/base_(4,0)\_obstacles_2.png)

5. Ground truth passageways are maintained with above augmentations
    * ![(4,0) - With ground truth](samples/base_(4,0)\_gt_1.png)
    * ![(4,0) - With ground truth](samples/base_(4,0)\_gt_2.png)

### Model

## Usage
TBD