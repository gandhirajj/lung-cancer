Lung Cancer Classification using CNN + Vision Transformer (DeiT).

🧠 CNN + Transformer (DeiT) Architecture
This project combines the Convolutional Neural Network (CNN) and the Vision Transformer (DeiT) to leverage the strengths of both architectures for lung cancer classification.

1. CNN Module
Purpose: Extracts low-level spatial features such as edges, textures, and simple patterns from the CT scan images.

How it works:

Applies convolutional layers to scan the image with small kernels.

Uses pooling layers to reduce dimensionality while preserving important features.

Generates a compact feature map that represents the important visual details.

2. Vision Transformer (DeiT) Module
Purpose: Captures global relationships and context between different parts of the image.

How it works:

Splits the CNN-extracted feature map into patches.

Embeds each patch into a vector representation.

Uses multi-head self-attention to learn how different image regions relate to each other.

DeiT (Data-efficient Image Transformer) uses distillation to train efficiently on smaller datasets.

3. Why CNN + DeiT?
CNN excels at learning local features (textures, edges).

Transformers excel at learning global dependencies (long-range relationships in the image).

Combining both:

CNN acts as a feature extractor.

Transformer acts as a context learner.

This hybrid approach helps in medical imaging tasks where fine local details and overall structural patterns are both important.

Pipeline in This Project
Input Image → Preprocessing (Resize, Normalize)

CNN Layers → Extract local features

Flatten + Patch Embedding → Prepare for Transformer

DeiT Transformer Layers → Learn global context

Fully Connected Layer → Predict one of the 4 lung cancer classes

Dataset Description
The dataset consists of 1006 CT scan images categorized into four classes:

Class Name	         Train Images	    Test Images	          Validation Images

Adenocarcinoma	        195	            120	                    24

Large Cell Carcinoma	  115	            51	                    21

Normal	                142	            54	                    13

Squamous Cell Carcinoma	160	            90	                    15

This well-structured dataset is split into training, testing, and validation sets, ensuring balanced model development, evaluation, and validation.

Tech Stack
Python
PyTorch
TorchVision
Transformers (Hugging Face)
Streamlit 
PIL (Pillow)

Project Structure
bash
Copy
Edit
.
├── ct_scan.ipynb                # Model training code

├── app2.py                        # Deployment script (Streamlit)

├── best_model_cnn_diet.pth       # Trained model weights

├── requirements.txt              # Dependencies

├── README.md                     # Project documentation

└── dataset/                      # CT scan images (train/test/val)

Model Accuracy
Based on the training with the CNN + DeiT (Vision Transformer) hybrid architecture, the project achieved:
Training Accuracy: ~98%

Validation Accuracy: ~94%

Test Accuracy: ~93%
These results indicate that the model generalizes well, with only a small gap between training and validation accuracy, meaning it’s not heavily overfitting and is able to handle unseen CT scan images effectively.

🏁 Conclusion
This project successfully demonstrates the potential of combining Convolutional Neural Networks (CNNs) with Vision Transformers (DeiT) for medical image classification tasks, specifically lung cancer detection from CT scans.
CNN layers excelled at extracting local patterns such as tissue textures, edges, and small anomalies.
DeiT layers captured the global relationships within lung structures, improving classification accuracy and robustness.
The hybrid approach outperformed using CNN or Transformer alone, especially on small and imbalanced medical datasets.
The model can be used as a decision-support tool for radiologists, reducing diagnosis time and improving early cancer detection rates.
With further fine-tuning, larger datasets, and integration into a clinical pipeline, this system could contribute to AI-assisted lung cancer screening in real-world scenarios.



