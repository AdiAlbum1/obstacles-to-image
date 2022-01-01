
import sys
import argparse


from importlib import util

from SceneDisc import SceneDisc
from ScenePolygon import ScenePolygon
import read_input

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, required=True, help='path to the scene file (.json extension)')
    parser.add_argument('--solver', type=str, required=True, help='path to the solver file (.py extension)')
    parser.add_argument('--argument', type=str, default="", help='argument provided to the solver')
    parser.add_argument('--output', type=str, default="", help='export the path to the file if provided')
    parser.add_argument('--input', type=str, default="", help='loads the path from file if provided')
    parser.add_argument('--mode', type=str, default="disc", help='mode of operation (either disc or polygon)')
    args = parser.parse_args()
    mode = args.mode
    scene = args.scene
    solver = args.solver
    argument = args.argument
    input = args.input
    output = args.output
    if mode == "disc":
        ps = SceneDisc(None, sys.stdout)
    elif mode == "polygon":
        ps = ScenePolygon(None, sys.stdout)
    else:
        print("Unsupported mode:", mode)
        exit()
        
    ps.load_scene(scene)
    
    if input:
        read_input.load_path(ps.path, input, ps.number_of_robots)
    else:
        spec = util.spec_from_file_location(solver, solver)
        gp = util.module_from_spec(spec)
        spec.loader.exec_module(gp)
        if mode == "disc":
            ps.path, _ = gp.generate_path_disc(ps.robots, ps.obstacles, ps.disc_obstacles, ps.destinations,
                                        argument, sys.stdout, [ True ])
        else:
            ps.path, _ = gp.generate_path_polygon(ps.robots, ps.obstacles, ps.destinations,
                                        argument, sys.stdout, [ True ])
    ps.is_path_valid()
    if output:
        with open(output, 'w') as f:
            for i in range(len(ps.path)):
                p = ps.path[i]
                for robot in p:
                    x = robot.x().exact()
                    y = robot.y().exact()
                    f.write(str(x.numerator()) + "/" + str(x.denominator()) + " " + str(y.numerator()) + "/" + str(
                        y.denominator()) + " ")
                if i != len(ps.path) - 1:
                    f.write('\n')