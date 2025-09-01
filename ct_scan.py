import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.swin_transformer import swin_t, Swin_T_Weights
import bitsandbytes as bnb
from PIL import Image
import numpy as np

# --------------------------
# QLoRA Wrapper
# --------------------------
class QLoRALinear(nn.Module):
    def __init__(self, orig_linear, r=8, alpha=16):
        super().__init__()
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        self.lora_A = bnb.nn.Linear4bit(self.in_features, r, bias=False, quant_type='nf4')
        self.lora_B = bnb.nn.Linear4bit(r, self.out_features, bias=False, quant_type='nf4')

        self.weight = orig_linear.weight
        self.bias = orig_linear.bias
        self.orig_linear = orig_linear

        self.requires_grad_(False)
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias) + self.lora_B(self.lora_A(x)) * self.scale

def inject_qlora_into_swin(swin_model):
    for name, module in swin_model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features == 96:
            parent = dict(swin_model.named_modules())[name.rsplit('.', 1)[0]]
            attr = name.split('.')[-1]
            setattr(parent, attr, QLoRALinear(module))
    return swin_model

# --------------------------
# Hybrid Model Definition
# --------------------------
class CNN_Swin_QLoRA(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.cnn = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.cnn.classifier = nn.Identity()
        self.cnn_out_dim = 1280

        self.swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.swin.head = nn.Identity()
        self.swin = inject_qlora_into_swin(self.swin)

        self.cnn_project = nn.Linear(self.cnn_out_dim, 768)

        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        cnn_feat = self.cnn.features(x)
        cnn_feat = cnn_feat.mean(dim=[2, 3])
        cnn_feat = self.cnn_project(cnn_feat)

        swin_feat = self.swin(x)
        combined = torch.cat([cnn_feat, swin_feat], dim=1)
        return self.classifier(combined)

# --------------------------
# Streamlit App
# --------------------------
st.set_page_config(page_title="CT Scan Lung Nodule Classifier", layout="centered")
st.title("🫁 CT Scan Lung Nodule Classifier")
st.write("Upload a CT scan image to classify the type of lung nodule.")

uploaded_file = st.file_uploader("Upload CT scan image (JPG/PNG)", type=["jpg", "jpeg", "png"])

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load model
@st.cache_resource
def load_model():
    model = CNN_Swin_QLoRA(num_classes=4)
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()
class_labels = ['Benign', 'Malignant', 'Non-nodule', 'Other']  # Adjust as per your class names

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded CT Scan", use_column_width=True)

    with st.spinner("Classifying..."):
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            predicted = probs.argmax(dim=1).item()
            confidence = probs[0][predicted].item() * 100

        st.success(f"**Prediction:** {class_labels[predicted]} ({confidence:.2f}%)")

        st.subheader("Confidence Scores")
        for idx, label in enumerate(class_labels):
            st.write(f"{label}: {probs[0][idx]*100:.2f}%")
