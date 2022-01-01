#/bin/sh
project=scene_designer
rm -rf ./build
rm -rf ./dist
pyinstaller $project.py
cp -r ../gui ./dist/$project/gui
cp readme.md dist/$project/readme.md
rm $project.spec
rm -r ./build
