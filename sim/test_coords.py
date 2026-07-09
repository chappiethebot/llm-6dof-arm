from sim_backend import SimBackend, COORDS

be = SimBackend(render=False)
for name in COORDS:
    reached, steps, err = be.move_ee_to(be.resolve(name))
    print(f"{'OK ' if reached else 'FAIL'} {name:13s} err {err*100:4.1f} cm")
be.close()