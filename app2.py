import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from transformers import DeiTModel, DeiTConfig
from PIL import Image

# -----------------------------
# Model Definition (must match training)
# -----------------------------
class CNN_DeiT_Hybrid(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.cnn.classifier = nn.Identity()
        self.cnn_out_channels = 1280

        deit_config = DeiTConfig.from_pretrained("facebook/deit-base-patch16-224")
        self.deit = DeiTModel(deit_config)
        self.deit_dim = deit_config.hidden_size

        self.patch_proj = nn.Linear(self.cnn_out_channels, self.deit_dim)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(self.deit_dim),
            nn.Linear(self.deit_dim, num_classes)
        )

    def forward(self, x):
        feats = self.cnn.features(x)  # (B, 1280, H, W)
        B, C, H, W = feats.shape
        feats = feats.flatten(2).transpose(1, 2)  # (B, N, C)
        tokens = self.patch_proj(feats)  # (B, N, deit_dim)

        cls_token = self.deit.embeddings.cls_token.expand(B, -1, -1)
        input_tokens = torch.cat((cls_token, tokens), dim=1)

        pos_embed = self.deit.embeddings.position_embeddings[:, :input_tokens.size(1), :]
        input_tokens = input_tokens + pos_embed

        x = self.deit.encoder(input_tokens)
        cls_output = x.last_hidden_state[:, 0]  # CLS token
        return self.cls_head(cls_output)

# -----------------------------
# Streamlit App Config
# -----------------------------
st.set_page_config(page_title="Lung CT Scan Classifier", layout="centered")
st.title("🩻 Lung CT Scan Classification")
st.write("Upload a CT scan image to classify it into one of four lung cancer types or normal.")

# Class names in correct order
class_names = [
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Normal",
    "Squamous Cell Carcinoma"
]

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
@st.cache_resource
def load_model():
    model = CNN_DeiT_Hybrid(num_classes=len(class_names)).to(device)
    state_dict = torch.load("best_model_cnn_deit1.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# Image transform (same as validation)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# File uploader
uploaded_file = st.file_uploader("Choose a CT scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_class = class_names[pred_idx]
        confidence = probs[pred_idx].item() * 100

    # Display result
    st.markdown(f"### Prediction: **{pred_class}**")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
    st.write("#### Class Probabilities:")
    for i, cls in enumerate(class_names):
        st.write(f"{cls}: {probs[i].item()*100:.2f}%")
