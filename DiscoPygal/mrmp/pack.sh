#!/bin/sh
project=mrmp
rm -r ./package
mkdir package
mkdir package/$project
cp -r ../gui package/$project/gui

cp readme.md package/$project/readme.md
cp bindings.py package/$project/bindings.py
cp conversions.py package/$project/conversions.py
cp Collision_detection.py package/$project/Collision_detection.py
cp gui_mrmp.py package/$project/gui_mrmp.py
cp $project.py package/$project/$project.py
cp ms_polygon_segment.py package/$project/ms_polygon_segment.py
cp linear_path_intersection_test.py package/$project/linear_path_intersection_test.py
cp SceneBase.py package/$project/SceneBase.py
cp ScenePolygon.py package/$project/ScenePolygon.py
cp SceneDisc.py package/$project/SceneDisc.py
cp read_input.py package/$project/read_input.py
cp -r scenes package/$project/scenes
cp -r solvers package/$project/solvers

curl https://www.cs.tau.ac.il/~cgl/python_rmp/libgmp-10.dll -o package/$project/libgmp-10.dll
curl https://www.cs.tau.ac.il/~cgl/python_rmp/libmpfr-4.dll -o package/$project/libmpfr-4.dll
CGALPY=CGALPY.pyd
curl https://www.cs.tau.ac.il/~cgl/python_rmp/$CGALPY -o package/$project/$CGALPY
