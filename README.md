Provisional Diagnosis and Prognosis of Burn Skin Using CNN
This project utilizes Convolutional Neural Networks (CNNs) to improve the diagnosis and prognosis of burn injuries. The model not only classifies the severity of burns but also offers treatment recommendations for effective clinical decision-making in plastic surgery.

Overview
Burn injuries require precise diagnosis and prognosis for successful treatment. Traditional diagnostic methods often fall short in accuracy. This project leverages the power of deep learning and CNNs to:

Classify burn injuries into four categories: First-degree, Second-degree, Third-degree, and Healthy skin.
Provide prognostic insights by predicting the healing process.
Offer treatment recommendations based on burn severity, assisting medical professionals.

Key Features:
Model Architecture: Custom CNN with multiple convolutional layers and dropout to prevent overfitting.
Dataset: 1,100 labeled images representing diverse burn conditions.
Performance: Accurate burn classification and actionable treatment suggestions.

Dataset
The dataset includes images categorized into:
First-degree burns: Mild burns with tan.
Second-degree burns: Moderate burns with blisters.
Third-degree burns: Severe burns requiring surgical intervention.
Healthy skin: Reference images for baseline.
Dataset Source:
Images sourced from Google, ensuring diversity and robustness.
Methodology
CNN Architecture
Input Layer: Accepts images of size 128x128 with RGB channels.
Convolutional Layers:
32–96 filters with ReLU activation.
MaxPooling2D after each convolution for dimensionality reduction.
Dropout: Applied to prevent overfitting.
Dense Layers: Includes a final Softmax layer for multi-class classification.
Prediction Capabilities
First-degree burns: Basic first-aid suggestions.
Second-degree burns: Recommends medical consultation.
Third-degree burns: Urgent surgical care guidance.
Healthy skin: Suggestions for skin health maintenance.
