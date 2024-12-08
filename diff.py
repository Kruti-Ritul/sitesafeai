from openvino.runtime import Core
import cv2
# Load OpenVINO model
ie = Core()
model_xml = './best_openvino_model/best.xml'
model_bin = model_xml.replace('.xml', '.bin')

# Compile the OpenVINO model
compiled_model = ie.compile_model(model=model_xml, device_name='CPU')

# Run inference
image_path = './data/img1.jpg'
input_image = cv2.imread(image_path)

# Preprocess the image
input_blob = compiled_model.input(0)
preprocessed_image = preprocess_image(input_image, input_blob.shape)

# Perform inference
results = compiled_model([preprocessed_image])

# Post-process the results and visualize
detections = postprocess_results(results, input_image.shape)
visualize_detections(input_image, detections)

