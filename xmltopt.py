from openvino.runtime import Core
import numpy as np

# Load OpenVINO model
core = Core()
model = core.read_model("./best_openvino_model/best_openvino_int8_model/best_with_preprocess.xml")
compiled_model = core.compile_model(model, "CPU")

# Get input/output layers
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

print("Model details extracted.")
