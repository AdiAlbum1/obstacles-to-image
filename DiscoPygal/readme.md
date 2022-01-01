# Python RMP Framework
A GUI framework for robot motion planning written in Python.

Includes projects for simulation and verification of RMP under various scenarios.

## Setup
This project uses pipenv to create a virtual environment and install required dependencies.

Windows 10/11:
* Install Python 3.9 for Windows (with pip)
* Install pipenv by running `pip install pipenv` in the command line.
* cd to the project's directory
* Run: `pipenv install`
* Run: `pipenv shell`
* Run: `python <project>.py`

Ubuntu (tested with Ubuntu 20.04.3):
* Run: `sudo apt-get install libxcb-xinerama0`
* Install python3.9-dev (`sudo apt-get install python3.9-dev`)
* Install pip
* Install pipenv
* cd to the project's directory
* Run: `pipenv install`
* Run: `pipenv shell`.
* Run: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pipenv --venv)/lib/python3.9/site-packages`
* Run: `python <project>.py`

## Projects overview
Below is a short overview of the projects:

### Interface
Provides a minimal interface for passing inputs to an external script and
drawing 2D graphics from that script.

### Multi Robot Motion Planning - Disc
Provides simulation and verification for simultaneous motion of multiple
disc robots in scenes consisting of polygonal obstacles.

### Multi Robot Motion Planning - Polygon
Provides simulation and verification for simultaneous motion of multiple
polygonal robots in scenes consisting of polygonal obstacles.

### Robot Motion Planning Workshop
Provides an environment for running a turn based robot motion planning game in
which two resource limited teams need to plan motions for multiple disc robots
among polygonal scenes in order to reach some goals.

### Rod
Provides simulation and verification for the motion of a rod robot involving
translation and rotation in scenes consisting of polygonal obstacles.

### Scene Designer
Provides a graphical tool to assist with designing polygonal scenes.

### Minkowski Sum Star
Allows the user to experiment with loading a polygon (or union of polygons) and computing its
minkwoski sum start (sum with itself n times, divided by n).
