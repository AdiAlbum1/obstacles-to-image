# Rod
Provides simulation and verification for the motion of a rod robot involving
translation and rotation in scenes consisting of polygonal obstacles.

Served as an environment for HW3 in Fall 2019/2020 Algorithmic Robotics
and Motion Planning course at Tel Aviv University.

## Scene structure
Scenes are JSON files with the following fields:

* `length` - The length of the robot
* `origin` - A list where the first two values specify the pivot's starting
position and the third the initial angle (in radians)
* `destination` - A list where the first two values specify the pivot's goal
position and the third the goal's angle (in radians)
* `obstacles` - A list of obstacles, each of which is a list of pairs of
points

An example scene is provided in `scenes/example_scene.json` .

## Generating a path
Clicking on `Generate Path` will call the `generate_path` function of the
provided planner script with the following parameters:

* `length` - An integral value representing the length of the robot.
* `obstacles` - a list of lists of tuples, where the i-th list represents the i-
th obstacle
* `origin` - A tuple (x, y, angle) of the origin position of the robot's
reference point
* `destination` - A tuple (x, y, angle) of the destination position of the
robot's reference point
* `argument` - An argument provided to the solver (i.e number of samples)
* `writer` - can be passed to the `file` named argument of the print function
to print the output to the application's GUI
* `isRunning` - A list holding one bool value, can be checked during the solver's
execution to determine if execution should be terminated.

The function should return a list of tuples. Each tuple represents a single configuration:
its first 3 indices represent the x, y, angle values of the corresponding configuration
as rational numbers. The last item in every tuple is either True or False,
stating whether the transition from the previous configuration should be in
a clockwise manner or not. Note that for the first configuration,
representing the robot in its initial position,  this value is meaningless.  
Here is an example of a possible implementation:

```
def generate_path(length, obstacles, origin, destination, argument, writer, isRunning):
  path = []
  path.append((FT(Gmpq(230)), FT(Gmpq(500)), FT(Gmpq("0/3")), True))
  path.append((FT(Gmpq(300)), FT(Gmpq(1000)), FT(Gmpq("2/1")), True))
  path.append((FT(Gmpq(230)), FT(Gmpq(700)), FT(Gmpq("1/5")), False))
  return path
```

An example solver `prm_basic.py` is provided under the `solvers` folder.

## Loading a Path
A path can also be loaded from a file. The file's content should include one
line for each part of the movement specifying the robot's position at the
end of the part, separated by spaces like so:

` x y angle`

## Verification and Animation
After a path is generated or loaded it can be verified and an animation of
the robot moving along the generated path can be displayed. If the path is
not valid the program will report the reason for failure.
