from full_grasp_backend import FullGraspBackend
import numpy as np

be = FullGraspBackend(render=True, gain=3.0)
spots = [(1.30, 0.72), (1.38, 0.80), (1.32, 0.68)]
try:
    i = 0
    while True:
        be.set_object([*spots[i % 3], 0.425])
        print(be.pick_place("rubiks_cube", ["box_1", "box_2", "box_3"][i % 3]))
        i += 1
except KeyboardInterrupt:
    pass
finally:
    be.close()