import streamlit as st
from PIL import Image
import os
from utils import show_bbox
from model import CoinExtractor, CoinClassifier
from PIL import Image

coin_extractor = CoinExtractor('models/fasterRCNN_coin_detector.pth')
classifier = CoinClassifier(model_path='models/FastViT_coin_classifier.pth',class_names_path='data/class_names.txt')

def apply_transformation(image):
    # Progress bar streamlit
    progress_bar = st.progress(0)  # Initialize the progress bar
    coins = coin_extractor.extract(image)
    progress_bar.progress(0.2)
    labels = []
    total_coins = len(coins)
    _, bboxes, scores = coin_extractor.extract_bboxes(image, return_class=True)
    for i, coin in enumerate(coins):
        prediction = classifier.classify(coin)
        p = 0.8 * ((i + 1) / total_coins)
        progress_bar.progress(p)
        labels.append(prediction)
    output_image = show_bbox(image, labels, bboxes, scores)
    progress_bar.progress(100)
    st.success("🎉 All coins have been processed!")  # Celebrate completion with an emoji
    return output_image

# Title of the app
st.title("🪙 Coin Counter")

# Sidebar for user options
st.sidebar.title("⚙️ Options")

# Allow the user to either upload a photo or choose one from the carousel
upload_option = st.sidebar.radio(
    "How would you like to input your photo?",
    ('📤 Upload Photo', '🎠 Choose from Carousel')
)

image = None

# Option 1: Upload Photo
if upload_option == '📤 Upload Photo':
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

# Option 2: Choose from Carousel (predefined images)
else:
    img_dir = "data/carousel"  # Directory with images
    img_list = os.listdir(img_dir)
    img_selected = st.sidebar.selectbox("Select an image", img_list)
    image = Image.open(os.path.join(img_dir, img_selected))
    
# If an image is available, show the options to transform
if image is not None:
    st.image(image, caption="🖼️ Original Image", use_column_width=True)

    # Apply the transformation
    transformed_image = apply_transformation(image)

    # Display the transformed image
    st.image(transformed_image, caption=f"🔍 Coins Detected", use_column_width=True)
else:
    st.write("📷 Upload an image or select one from the carousel to begin!")
