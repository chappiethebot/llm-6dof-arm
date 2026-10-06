"""GPU load generator for diag_energy.py. Runs in the thesis2gpu env (CUDA torch).

    python diag_load.py steady 60     fp16 4096x4096 matmul in a loop for 60 s
    python diag_load.py pulsed 60     100 ms matmul, 100 ms sleep, for 60 s
"""
import sys
import time

import torch

mode, seconds = sys.argv[1], float(sys.argv[2])
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
torch.cuda.synchronize()
end = time.perf_counter() + seconds
while time.perf_counter() < end:
    burst_end = time.perf_counter() + 0.1
    while time.perf_counter() < burst_end:
        c = a @ b
        torch.cuda.synchronize()
    if mode == "pulsed":
        time.sleep(0.1)
print("load done", mode)
