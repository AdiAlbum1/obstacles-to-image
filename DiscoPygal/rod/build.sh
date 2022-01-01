#!/bin/sh
project=rod
rm -rf ./build
rm -rf ./dist
pyinstaller $project.py
cp readme.md dist/$project/readme.md
cp -r ../gui ./dist/$project/gui
cp -r scenes dist/$project/scenes
cp -r solvers dist/$project/solvers
curl https://www.cs.tau.ac.il/~cgl/python_rmp/libgmp-10.dll -o dist/$project/libgmp-10.dll
curl https://www.cs.tau.ac.il/~cgl/python_rmp/libmpfr-4.dll -o dist/$project/libmpfr-4.dll
CGALPY=CGALPY.pyd
curl https://www.cs.tau.ac.il/~cgl/python_rmp/$CGALPY -o dist/$project/$CGALPY
rm ./$project.spec
rm -r ./build
