import cv2
import numpy as np
from backend.gradcam import gradcam_processor

img = np.zeros((480, 640, 3), dtype=np.uint8)
out = gradcam_processor.compute_live(img, None, "model.model[21]")
if np.array_equal(out, img):
    print("FALLBACK: compute_live returned original image")
else:
    print("SUCCESS: compute_live returned modified image")
