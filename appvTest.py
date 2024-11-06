import streamlit as st
# import torch
# from torchvision import transforms, models
# from PIL import Image
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Define the path to the saved model
# MODEL_PATH = "models/buildings_deepfake_detector_mobilenet_v3_small_ep_10.pth"

# Define the image size and transforms
# image_size = (240, 240)
# image_transforms = transforms.Compose([
#     transforms.Resize(image_size),
#     transforms.ToTensor(),
# ])

# Load the model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("going to load the model")
# model = models.mobilenet_v3_small(pretrained=False)
# num_features = model.classifier[3].in_features
# model.classifier[3] = torch.nn.Linear(num_features, 2)  # Adjust the final layer to match your classes
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
print("Loaded the model")
# model = model.to(device)
# model.eval()

# Define class names
class_names = ["fake", "real"]

# Streamlit app
st.set_page_config(page_title="DeepFake Detector: Buildings", page_icon="🏢", layout="centered")

# Placeholder for logo (add your logo file path)
st.image("logo.png", use_column_width=True)

# App header
st.title("Building Authenticity Classifier")
st.write("""
Welcome to the Building Authenticity Classifier! This platform allows you to upload images of buildings and
determine whether they are real or fake. Our model has been trained to provide accurate classifications based on your uploaded images.
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # # Transform the image and prepare it for prediction
    # image_tensor = image_transforms(image).unsqueeze(0).to(device)

    # # Predict the class
    # with torch.no_grad():
    #     outputs = model(image_tensor)
    #     _, predicted = torch.max(outputs, 1)
    #     predicted_class = class_names[predicted.item()]

    # # Display the prediction
    # st.write(f"Prediction: **{predicted_class.capitalize()}**")
