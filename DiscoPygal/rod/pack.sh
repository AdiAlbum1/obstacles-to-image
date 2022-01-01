#!/bin/sh
project=rod
rm -r ./package
mkdir package
mkdir package/$project
cp -r ../gui package/$project/gui

cp readme.md package/$project/readme.md
cp bindings.py package/$project/bindings.py
cp $project.py package/$project/$project.py
cp conversions.py package/$project/conversions.py
cp gui_rod.py package/$project/gui_rod.py
cp read_input.py package/$project/read_input.py
cp sample_polygons.py package/$project/sample_polygons.py
cp Polygons_scene.py package/$project/Polygons_scene.py
cp -r scenes package/$project/scenes
cp -r solvers package/$project/solvers


curl https://www.cs.tau.ac.il/~cgl/python_rmp/libgmp-10.dll -o package/$project/libgmp-10.dll
curl https://www.cs.tau.ac.il/~cgl/python_rmp/libmpfr-4.dll -o package/$project/libmpfr-4.dll
curl https://www.cs.tau.ac.il/~cgl/python_rmp/CGALPY.pyd -o package/$project/CGALPY.pyd
