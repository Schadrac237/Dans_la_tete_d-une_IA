import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

from backend.gradcam import gradcam_processor

img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

print("Running compute_live...")
out = gradcam_processor.compute_live(img, None, "model.model[21]")

if np.array_equal(out, img):
    print("FALLBACK: output equals input")
else:
    print("SUCCESS: output differs")

# Let's also check target_class
out2 = gradcam_processor.compute_live(img, 0, "model.model[21]")
if np.array_equal(out2, img):
    print("FALLBACK target_class 0")
else:
    print("SUCCESS target_class 0")

