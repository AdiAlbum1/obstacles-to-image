#!/bin/sh
project=scene_designer
rm -r package
mkdir package
mkdir package/$project
cp -r ../gui package/$project/gui
cp scene_designer.py package/$project/scene_designer.py
cp gui_scene_designer.py package/$project/gui_scene_designer.py
cp circular_room.py package/$project/circular_room.py
cp readme.md package/$project/readme.md
