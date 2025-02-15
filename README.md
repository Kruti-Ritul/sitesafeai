# Site Safe AI

## Overview

Site Safe AI is an AI-driven system designed to enhance safety compliance on construction sites. Using advanced computer vision techniques and deep learning models, the project detects essential safety equipment like helmets and vests, ensuring that workers adhere to safety standards. The solution provides real-time monitoring to reduce risks, prevent accidents, and improve compliance with safety regulations.

## Problem Statement

Construction sites are high-risk environments where failure to use proper safety equipment, such as helmets and vests, can result in serious injuries or fatalities. Manual safety monitoring is labor-intensive, error-prone, and often inefficient. Site Safe AI addresses these challenges by automating safety equipment detection, ensuring worker compliance in real-time, and promoting a safer work environment.

## Solution

Site Safe AI leverages the YOLOv8 (You Only Look Once) model for object detection and OpenVINO for optimization. It processes video footage or images to identify safety equipment and alerts relevant personnel in case of non-compliance. The solution integrates:

- **Computer Vision**: To detect and classify workers' safety gear.
- **Real-Time Inference**: To monitor compliance on-site without manual intervention.
- **Performance Optimization**: Using IPEX and OpenVINO for maximum efficiency.
- **Post-Processing Alerts**: Notifications via email or SMS when safety violations are detected.

## Features

- Real-time detection of safety equipment (helmets, vests, etc.).
- Optimized inference using Intel OpenVINO for faster processing.
- Enhanced PyTorch performance with Intel IPEX optimization.
- Optimized inference using Intel OpenVINO and IPEX or faster processing. 
- Alerts for safety violations via email or SMS.
- Scalable architecture for deployment on edge devices or cloud platforms.

## Dataset

The dataset is sourced from Kaggle: [Construction Site Safety Dataset](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)

## Technologies Used

- **AI Models and Frameworks**:
  - YOLOv8: Deep learning model for object detection.
  - OpenVINO: Model optimization for faster inference.
  - Intel Extension for PyTorch (IPEX): Performance optimization for PyTorch operations.

- **Programming Languages**:
  - Python
- **Tools and Platforms**:
  - Kaggle for model training and experimentation.
  - OpenVINO Toolkit: For optimizing the YOLOv8 model, enhancing inference performance on Intel hardware.
   - Intel IPEX: For accelerating PyTorch workloads on Intel hardware.

## How to use

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Pree46/sitesafeai.git
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Train the model**:
- Use the `./sitesafeai.ipynb` notebook to train the model on the provided dataset.
4. **Optimize the model**:
- Convert the trained model using OpenVINO(`./OpenVino Optimization.ipynb`) for faster inference.
- Apply IPEX Optimization (`./IPEX Optimization.ipynb`)
- Use the `./sitesafeai.ipynb` (YOLOv8 notebook) to train the model on the provided dataset.
- Add your email id to get generate reports in `./app.py`.

4. **Optimize the model**:

- Convert the trained model using OpenVINO (`./OpenVino Optimization.ipynb`) for faster inference.
- To optimize Yolov8 using IPEX(`./IPEX Optimization.ipynb`) 

5. **Run the application**:
   ```bash
   python app.py
   ```

## Demo

[Watch the Demo Video](https://drive.google.com/file/d/1311zScdP6FhMhBqcy5lCJLJ8fiYMlEFk/view?usp=sharing)

<p align="center">
  <img src="./data/demo.jpg" alt="Immediate SMS Alerts" width="400" height="700" />
</p>

## Acknowledgements

- Kaggle for providing the Construction Site Safety Dataset.
- Intel OpenVINO for optimization tools.
- Intel IPEX for PyTorch optimization support. 
- YOLOv8 for object detection.


