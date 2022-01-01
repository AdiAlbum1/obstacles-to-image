# Multi Robot Motion Planning
Provides simulation and verification for simultaneous motion of multiple
disc/polygonal robots in scenes consisting of polygonal obstacles.

## Prerequisites

* Install Python 3.9 64 bit
* Install PyQt5 - ``pip install PyQt5``
* Install networkx - ``pip install networkx``

# Modes
Two modes of operation are available, Disc (for handling disc robots) and
 Polygon (for handling polygonal robots). The mode should be selected before
 the scene is loaded.

## Scene structure
Scenes are JSON files with the following fields:

* `sources` - A list of starting positions for the robots. The length of the
list also determines the number of robots
* `targets` - A list of (labeled) targets for the robots - the i'th robot's
reference point has to reach the i'th target
* `obstacles` - A list of obstacles, each of which is a list of pairs of
points
* `disc_obstacles` - A list of objects with `center` and `radius` describing disc
* obstacles in the scene
* `radii` - A list of radii where the i'th entry is the radius of the i'th disc robot
* `robots` - A list of polygons describing the shapes of the polygonal robots.
Each polygon will be translated such that its first coordinate will coincide
with its respective source.


Example scenes are provided under `scenes/` .

## Generating a Solution
Clicking on `Generate Path` will call the `generate_path_disc` (in disc mode)
or `generate_path_polygon` function of the provided planner script with the
following parameters:

### generate_path_polygon arguments:
* `robots` - A list of the robots as Pol2.Polygon_2 objects
* `obstacles` - A list of the obstacles as Pol2.Polygon_2 objects
* `destinations` - A list of Ker.Point_2 objects of the robots' destinations
* `argument` - an optional argument passed to the solver (suggestion, taken from the GUI)
* `writer` - An object that when written into redirects the input into the
standard output and the GUI output (can be used as the `file` argument of
  `print`)
* `isRunning` - A boolean that changes to false when the solver is requested to
abort its operation and return (should be checked every once in a while)


### generate_path_disc arguments:
* `robots` - A list of the robots as a list of dicts with
entries `center` (pair of `Ker.FT` coordinates) and `radius` (of type `FT`)
* `obstacles` - A list of the obstacles as `Pol2.Polygon_2` objects
* `disc_obstacles` - A list of the disc_obstacles as a list of dicts with
entries `center` (pair of `Ker.FT` coordinates) and `radius` (of type `FT`)
* `destinations` - A list of `Ker.Point_2` objects of the robots' destinations
* `argument` - an optional argument passed to the solver (suggestion, taken from the GUI)
* `writer` - An object that when written into redirects the input into the
standard output and the GUI output (can be used as the `file` argument of
  `print`)
* `isRunning` - A boolean that changes to false when the solver is requested to
abort its operation and return (should be checked every once in a while)


The solver is expected to return a list of tuples of `Ker.Point_2` objects for the
robots to interpolate between, and optinally a `networkx.Graph` object. The projection of
the graph on the 2-d plane of a specific robot can be displayed on the scene.

An example prm based solver is provided under the `solvers` folder


## Loading a Solution
A solution can also be loaded from a file. The file's content should include one
line for each part of the movement specifying where the robots will end up at,
separated by spaces like so:

` x1 y1 x2 y2 x3 y3 ... `

Where each coordinate is represented as a rational number, namely
`integer/integer`.

## Exporting a path
Providing a path to a file and clicking on `Export Solution` will export The
solution with the format described above to a file.

## Verification and Animation
After a path is generated or loaded it can be verified and an animation of
the robots moving along the generated path can be displayed. If the path is
not valid the program will report the reason for failure.
