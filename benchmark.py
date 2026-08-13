from ultralytics import YOLO
import numpy as np
import time

model = YOLO("yolov8n.pt")
dummy = np.zeros((480, 640, 3), dtype="uint8")

# warm up
model(dummy, device="cpu", classes=[0], verbose=False)

times = []
for _ in range(10):
    t0 = time.perf_counter()
    model(dummy, device="cpu", classes=[0], verbose=False)
    times.append(time.perf_counter() - t0)

avg = sum(times) / len(times)
print(f"Avg inference latency : {avg*1000:.1f} ms")
print(f"Inference-only FPS    : {1/avg:.1f}")
print("Smoke test PASSED")
