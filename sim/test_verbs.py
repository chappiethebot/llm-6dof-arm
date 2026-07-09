from sim_backend import SimBackend

be = SimBackend(render=False)

# all seven verbs
for r in [be.move_to("table_right"),
          be.pick_place("bottle", "box_2"),
          be.fetch("rubiks_cube"),
          be.hold("toy_car"),
          be.inspect("bottle"),
          be.stop(),
          be.reject()]:
    print(f"{r['verb']:11s} ok={r['ok']} steps={r['steps']}")

# invalid name on a target verb and a location verb
for bad in [lambda: be.hold("dragon"), lambda: be.move_to("narnia")]:
    try:
        bad(); print("MISSED invalid (bad)")
    except KeyError as e:
        print("invalid ->", e)

# obstacle avoidance still works through a verb
be.move_to("table_left")
be.add_obstacle(be.resolve("table_center"), radius=0.04)
r = be.move_to("table_right")
print(f"blocked move_to ok={r['ok']} steps={r['steps']} (higher steps = detoured)")

be.close()