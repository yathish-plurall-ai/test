import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from supabase import create_client, Client



# st.set_page_config(page_title="DeepFake Detector: Buildings", page_icon="🏢", layout="centered")
st.set_page_config(page_title="DeepFake Detector: Buildings", page_icon="Light_Icon.png", layout="centered")


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)



# Initialize Supabase client
SUPABASE_URL = "https://kwmpxpodmpwplujkpwjn.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt3bXB4cG9kbXB3cGx1amtwd2puIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcxOTMzMzMwNywiZXhwIjoyMDM0OTA5MzA3fQ.ZZM2pFJeBi3Tl6xI1NNzv3f4ETs2iIcS1EqkIowuO-M"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)



# Define the path to the saved model
MODEL_PATH = "models/buildings_deepfake_detector_mobilenet_v3_small_ep_10.pth"

# Define the image size and transforms
image_size = (240, 240)
image_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("going to load the model")
# model = models.mobilenet_v3_small(pretrained=False)
# num_features = model.classifier[3].in_features
# model.classifier[3] = torch.nn.Linear(num_features, 2)  # Adjust the final layer to match your classes
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# print("Loaded the model")
# model = model.to(device)
# model.eval()


@st.cache_resource
def load_model():
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading the model")
    model = models.mobilenet_v3_small(pretrained=False)
    num_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(num_features, 2)  # Adjust the final layer to match your classes
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Loaded the model")
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# Define class names
class_names = ["fake", "real"]

# Streamlit app


# Placeholder for logo (add your logo file path)
# st.image("logo-banner.png", use_column_width='auto')

# st.columns(3)[1].image("logo-banner2.png", use_column_width='auto')
# st.columns(3)[1].image("logo-banner2.png")
st.image("logo-banner2.png")


# App header
st.subheader("GaussMass 1.0")

st.markdown(
"""
- AI Detection Tool that uses in-house fundamental infrastructure
- 93% accuracy within seconds of reaction
- Help analyze files including photos and videos
- Safeguard against deepfake content
"""
)

# st.write("""
# GaussMass 1.0, a powerful AI detection tool that uses in-house fundamental infrastructure to detect deepfakes content. It works by first analyzing and detecting files uploaded, pushing detection result back to the user with a 93% accuracy within seconds of reaction. GaussMass 1.0 is a safeguard against potential malicious deepfakes content on the Internet for people to get closer to truth.
# """)
# st.write("""
# Aiming for detection image deepfakes content, GaussMass 1.0 combines the advantages of high speed, controlling latency, maintaining scalability, reassuring safety to ensure good quality detection delivery as the most unique AI detection tools anti-deepfakes. Audio detection and video section will be the next mission for GaussMass.
# """)

# Initialize session state for email and feedback
if 'email' not in st.session_state:
    st.session_state.email = ""
if 'prediction' not in st.session_state:
    st.session_state.prediction = ""

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Email input
    st.session_state.email = st.text_input("Enter your email to see whether its fake:", value=st.session_state.email)
    
    if st.session_state.email:
        if st.button("Submit"):
            # Transform the image and prepare it for prediction
            image_tensor = image_transforms(image).unsqueeze(0).to(device)
            
            # Predict the class
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                st.session_state.prediction = class_names[predicted.item()]
            
            # Display the prediction
            st.write(f"Prediction: This is **{st.session_state.prediction.capitalize()}**")

if st.session_state.prediction:
    # Thumbs up/down feedback
    feedback = st.radio("Was this prediction helpful?", ("", "Yes", "No"))
    if feedback:
        with open("feedback.txt", "a") as f:
            f.write(f"Email: {st.session_state.email}, Prediction: {st.session_state.prediction}, Feedback: {feedback}\n")
        response = supabase.table('UserFeedback').insert({
            'email_id': st.session_state.email,            
            'feedback': feedback
        }).execute()
        
        # print(response.__dir__())

        if response:  # Check for an error in the response
            st.success("Thank you for your feedback!")
        else:
            st.error("There was an error submitting your feedback. Please try again.")