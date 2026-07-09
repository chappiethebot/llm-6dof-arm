from sim_backend import SimBackend

be = SimBackend(render=False)
print("valid   :", be.move_to("table_right"))
try:
    be.move_to("teleporter_bay")
    print("invalid : NO ERROR (bad)")
except KeyError as e:
    print("invalid : KeyError ->", e)
be.move_to("table_left")
be.add_obstacle(be.resolve("table_center"), radius=0.04)
print("blocked :", be.move_to("table_right"))
be.close()