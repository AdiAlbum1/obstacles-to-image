# Scene Designer - Rod
Provides a graphical tool to assist with designing polygonal scenes with a rod polygon.

## Prerequisites

* Install Python 3.9 64 bit
* Install PyQt5 - ``pip install PyQt5``

## How to Use

* Use the mode buttons in the left panel to select what to insert
* For polygonal obstacles: Right click anywhere to start a new polygonal chain, 
    keep right clicking to add new vertices to the chain. 
    To close the chain to form a polygon obstacle, right clicking on some existing vertex of the chain.
* For rod start/end: Right click once to change location of reference point
* Press ctrl+Z to remove the most recently added vertex from the chain (in polyon mode).
* To select an existing polygon, left click on one of its vertices.
* Press DELETE on your keyboard to delete the selected obstacle.
* If multiple polygons share the same vertex, use selection by index to specify the polygon to be selected.
* New vertices will snap to the grid (default resolution of 1.0 units).
* The grid's resolution can be set from the GUI.
* The scene can be exported to a JSON file with a format corresponding to the expected format.
* A scene can also be loaded from a JSON file into the scene designer for inspection or modification.
