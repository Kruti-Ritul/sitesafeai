# Site Safe AI

## Overview
Site Safe AI is an AI-driven system designed to enhance safety compliance on construction sites. Using advanced computer vision techniques and deep learning models, the project detects essential safety equipment like helmets and vests, ensuring that workers adhere to safety standards. The solution provides real-time monitoring to reduce risks, prevent accidents, and improve compliance with safety regulations.

## Problem Statement
Construction sites are high-risk environments where failure to use proper safety equipment, such as helmets and vests, can result in serious injuries or fatalities. Manual safety monitoring is labor-intensive, error-prone, and often inefficient. Site Safe AI addresses these challenges by automating safety equipment detection, ensuring worker compliance in real-time, and promoting a safer work environment.

## Solution
Site Safe AI leverages the YOLOv8 (You Only Look Once) model for object detection and OpenVINO for optimization. It processes video footage or images to identify safety equipment and alerts relevant personnel in case of non-compliance. The solution integrates:

- **Computer Vision**: To detect and classify workers' safety gear.
- **Real-Time Inference**: To monitor compliance on-site without manual intervention.
- **Post-Processing Alerts**: Notifications via email or SMS when safety violations are detected.

## Features
- Real-time detection of safety equipment (helmets, vests, etc.).
- Optimized inference using Intel OpenVINO for faster processing.
- Alerts for safety violations via email or SMS.
- Scalable architecture for deployment on edge devices or cloud platforms.

## Dataset
The dataset is sourced from Kaggle: [Construction Site Safety Dataset](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)

## Technologies Used
- **AI Models and Frameworks**:
  - YOLOv8: Deep learning model for object detection.
  - OpenVINO: Model optimization for faster inference.
  
- **Programming Languages**:
  - Python
  
- **Tools and Platforms**:
  - Kaggle for model training and experimentation.
  - OpenVINO Toolkit: For optimizing the YOLOv8 model, enhancing inference performance on Intel hardware.

## Acknowledgements
- Kaggle for providing the Construction Site Safety Dataset.
- Intel OpenVINO for optimization tools.
- YOLOv8 for object detection.
