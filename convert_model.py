"""
convert_model.py
----------------
Phase 1 utility: Converts hybrid_model_final_best.keras to a
post-training-quantized TFLite model (YogaMaster_Hybrid_Production.tflite).

The hybrid model has two input layers:
  - image_input : (1, 224, 224, 3)  float32
  - pose_input  : (1, 34)           float32

Run from the project root:
    python convert_model.py
"""

import os
import sys
import numpy as np
import tensorflow as tf

# ── Paths ──────────────────────────────────────────────────────────────────────
KERAS_MODEL_PATH  = "hybrid_model_final_best.keras"
TFLITE_MODEL_PATH = "YogaMaster_Hybrid_Production.tflite"

# ── Sanity check ───────────────────────────────────────────────────────────────
if not os.path.exists(KERAS_MODEL_PATH):
    print(f"[ERROR] Keras model not found at '{KERAS_MODEL_PATH}'.")
    print("        Please ensure you are running this script from the project root.")
    sys.exit(1)

# ── 1. Load the Keras model ────────────────────────────────────────────────────
print(f"[1/4] Loading Keras model from '{KERAS_MODEL_PATH}' …")
model = tf.keras.models.load_model(KERAS_MODEL_PATH)
print(f"      Model loaded.  Inputs : {[i.name for i in model.inputs]}")
print(f"                     Outputs: {[o.name for o in model.outputs]}")

# ── 2. Build the TFLite converter ─────────────────────────────────────────────
print("[2/4] Building TFLite converter with DEFAULT optimizations …")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset for full-integer calibration (optional but recommended).
# Here we generate random data that matches both inputs so the quantizer can
# estimate activation ranges.  Swap in real yoga images for better calibration.
def representative_dataset():
    num_samples = 100
    for _ in range(num_samples):
        image_sample = np.random.rand(1, 224, 224, 3).astype(np.float32)
        pose_sample  = np.random.rand(1, 34).astype(np.float32)
        # The converter expects a list whose order matches model.inputs.
        # We detect the correct order by inspecting input names.
        input_order = []
        for inp in model.inputs:
            if "image" in inp.name.lower():
                input_order.append(image_sample)
            elif "pose" in inp.name.lower():
                input_order.append(pose_sample)
            else:
                # Fallback: use shape to decide
                if len(inp.shape) == 4:
                    input_order.append(image_sample)
                else:
                    input_order.append(pose_sample)
        yield input_order

converter.representative_dataset = representative_dataset

# ── 3. Convert ────────────────────────────────────────────────────────────────
print("[3/4] Converting … (this may take a minute)")
try:
    tflite_model = converter.convert()
except Exception as exc:
    print(f"\n[WARNING] Full-integer quantization failed: {exc}")
    print("          Falling back to dynamic-range quantization …\n")
    converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter2.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter2.convert()

# ── 4. Save & validate ────────────────────────────────────────────────────────
print(f"[4/4] Saving TFLite model to '{TFLITE_MODEL_PATH}' …")
with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(TFLITE_MODEL_PATH) / (1024 ** 2)
print(f"      Saved  ({size_mb:.2f} MB)")

# Quick smoke-test: load the TFLite model and print its input/output details.
print("\n── Smoke test ────────────────────────────────────────────────────────────")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"  TFLite inputs  ({len(input_details)}):")
for d in input_details:
    print(f"    index={d['index']}  name='{d['name']}'  shape={d['shape']}  dtype={d['dtype'].__name__}")

print(f"  TFLite outputs ({len(output_details)}):")
for d in output_details:
    print(f"    index={d['index']}  name='{d['name']}'  shape={d['shape']}  dtype={d['dtype'].__name__}")

print("\n[DONE] Quantization complete. ✓")
print(f"       Output file: {os.path.abspath(TFLITE_MODEL_PATH)}")
