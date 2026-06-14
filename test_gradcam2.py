import sys
import numpy as np

sys.path.append("/home/schadrac/IA_avec_nous/backend")
from gradcam import gradcam_processor

img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

print("Running compute_live...")
out = gradcam_processor.compute_live(img, None, "model.model[21]")

if np.array_equal(out, img):
    print("FALLBACK: output equals input")
else:
    print("SUCCESS: output differs")
