# Scene Designer
Provides a graphical tool to assist with designing polygonal scenes.

## Prerequisites

* Install Python 3.9 64 bit
* Install PyQt5 - ``pip install PyQt5``

## How to Use

* Right click anywhere to start a new polygonal chain.
* Keep right clicking to add new vertices to the chain.
* Close the chain to form a polygon obstacle by right clicking on some existing vertex
of the chain.
* Closing a chain of one point adds a disc obstacle.
* Closing a chain of two points adds a robot and its corresponding target to
the scene at the segment's endpoints.
* Clicking on `Add Circular Room` will add to the scene a circular room with
 the given parameters, centered around the last point of the current chain.
* Press ctrl+Z to remove the most recently added vertex from the chain.
* To select an existing polygon, left click on one of its vertices.
* Press DELETE on your keyboard to delete the selected obstacle or source/destination (when a source is selected).
* If multiple polygons share the same vertex, use selection by index to
specify
the polygon to be selected.
* New vertices will snap to the grid (default resolution of 1.0 units).
* The grid's resolution can set if provided as a command line argument and
can also be changed from the GUI.
* The scene can be exported to a JSON file with a format
corresponding to the expected format for a list of polygons in the other
projects.
* A scene can also be loaded from a JSON file into the scene designer for
 inspection or modification.
