import numpy as np
from sim_backend import SimBackend

be = SimBackend(render=False)
be.move_ee_to(be.resolve("table_left"))
c = be.resolve("table_center"); r = 0.04
be.add_obstacle(c, r)

start = be.ee_pos(); tgt = be.resolve("table_right")
wps = [np.array([start[0], start[1], 0.55]),
       np.array([tgt[0], tgt[1], 0.55]), tgt]
mc = 1e9
for wp in wps:
    for _ in range(200):
        mc = min(mc, float(np.linalg.norm(be.ee_pos() - c)) - r)
        e = wp - be.ee_pos()
        if np.linalg.norm(e) < 0.02:
            break
        be._step(np.concatenate([be.gain * e, [0]]))
print(f"min clearance: {mc*100:.1f} cm  (positive = never touched the 4 cm obstacle)")
be.close()