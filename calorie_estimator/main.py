# type: ignore
import streamlit as st
import os
import time
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
import threading
import cv2
import sys

# Check Python version
if sys.version_info < (3, 12):
    st.error(f"This application requires Python 3.12 or newer. You're running Python {'.'.join(map(str, sys.version_info[:3]))}")
    sys.exit(1)

# Import our modules
from utils import preprocess_image, load_image_for_display, capture_image_from_webcam, save_uploaded_image
from classifier import FoodClassifier
from calorie_estimator import CalorieEstimator
from config import get_nutrition_api_key

# Set page configuration
st.set_page_config(
    page_title="Food Calorie Estimator",
    page_icon="🍔",
    layout="wide"
)

# Set app title and description
st.title("🍔 Food Calorie Estimator")
st.subheader("Upload a food image or use webcam to estimate calories")

# Sidebar for settings
st.sidebar.title("Settings")

# Load API key from .env file
default_api_key = get_nutrition_api_key()

# Option to select model
model_path = st.sidebar.selectbox(
    "Select Model",
    ["Default (MobileNetV2)", "Custom (if available)"],
    help="Select the model to use for food classification"
)

# Option to adjust serving size
serving_size = st.sidebar.slider(
    "Serving Size (grams)",
    min_value=50,
    max_value=500,
    value=100,
    step=10,
    help="Adjust the serving size for calorie calculation"
)

# Initialize session state
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'captured_image_path' not in st.session_state:
    st.session_state.captured_image_path = None
if 'trigger_path' not in st.session_state:
    st.session_state.trigger_path = None

# Initialize classifier
@st.cache_resource
def load_classifier(model_selection):
    """Load and cache the food classifier model"""
    if model_selection == "Default (MobileNetV2)":
        return FoodClassifier(model_path=None)
    else:
        # Try to load custom model if available
        custom_model_path = os.path.join("model", "food_model.pth")
        if os.path.exists(custom_model_path):
            return FoodClassifier(model_path=custom_model_path)
        else:
            st.warning("Custom model not found, using default model instead")
            return FoodClassifier(model_path=None)

# Initialize calorie estimator
@st.cache_resource
def load_calorie_estimator(api_key):
    """Load and cache the calorie estimator"""
    return CalorieEstimator(api_key=api_key)

# Load the models
with st.spinner("Loading models..."):
    classifier = load_classifier(model_path)
    calorie_estimator = load_calorie_estimator(default_api_key)
    st.success("Models loaded successfully!")

# Function to predict and display results
def predict_and_display(image_path):
    """
    Process the image, predict food class, and display results
    """
    # Load and display the image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded/Captured Image")
        display_image = load_image_for_display(image_path)
        st.image(display_image, use_column_width=True)
    
    # Preprocess the image for the model
    try:
        image_tensor = preprocess_image(image_path)
        
        # Get top predictions
        with st.spinner("Identifying food..."):
            # Add a small delay to show the spinner
            time.sleep(1)
            top_predictions = classifier.get_top_predictions(image_tensor)
        
        # Get the top prediction
        top_food, confidence = top_predictions[0]
        
        # Get calorie information
        with st.spinner("Estimating calories..."):
            # Add a small delay to show the spinner
            time.sleep(1)
            nutrition_info = calorie_estimator.get_calories(top_food, serving_size)
        
        # Display results
        with col2:
            st.subheader("Results")
            st.write(f"**Identified Food:** {nutrition_info['food_item'].title()}")
            st.write(f"**Confidence:** {confidence*100:.1f}%")
            st.write(f"**Serving Size:** {nutrition_info['serving_size']}g")
            st.write(f"**Estimated Calories:** {nutrition_info['calories']} kcal")
            
            # Display nutrition facts if available
            if nutrition_info['protein'] is not None:
                st.subheader("Nutrition Facts")
                st.write(f"**Protein:** {nutrition_info['protein']}g")
                st.write(f"**Fat:** {nutrition_info['fat']}g")
                st.write(f"**Carbs:** {nutrition_info['carbs']}g")
                
                # Create a pie chart for macronutrients
                fig, ax = plt.subplots(figsize=(4, 4))
                labels = ['Protein', 'Fat', 'Carbs']
                sizes = [nutrition_info['protein'], nutrition_info['fat'], nutrition_info['carbs']]
                colors = ['#ff9999', '#66b3ff', '#99ff99']
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
        
        # Display all predictions
        st.subheader("Other Possible Matches")
        for food, prob in top_predictions[1:]:
            st.write(f"- {food.replace('_', ' ').title()}: {prob*100:.1f}%")
            
        # Show nutrition data source
        st.caption(f"Nutrition data source: {nutrition_info['source']}")
        
    except Exception as e:
        st.error(f"Error processing image: {e}")

# Main interface - Tabs for different input methods
tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Image", "📷 Use Webcam", "🔍 Example Images", "💬 Text Chat"])

with tab1:
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file
        tmp_file_path = save_uploaded_image(uploaded_file)
        
        # Process the image and display results
        predict_and_display(tmp_file_path)
        
        # Clean up the temporary file
        os.unlink(tmp_file_path)

with tab2:
    st.header("Capture from Webcam")
    st.write("Use your webcam to take a picture of your food")
    
    # Create placeholders for webcam feed and controls
    webcam_placeholder = st.empty()
    
    # Controls in two columns
    col1, col2 = st.columns(2)
    
    # Button to start/stop camera
    with col1:
        if not st.session_state.webcam_active:
            if st.button("Start Camera"):
                st.session_state.webcam_active = True
                st.experimental_rerun()
        else:
            if st.button("Stop Camera"):
                st.session_state.webcam_active = False
                # Clean up any trigger files
                if st.session_state.trigger_path and os.path.exists(st.session_state.trigger_path):
                    try:
                        os.remove(st.session_state.trigger_path)
                    except:
                        pass
                st.session_state.trigger_path = None
                st.experimental_rerun()
    
    # Button to capture image
    with col2:
        capture_button = st.button("Capture Image", disabled=not st.session_state.webcam_active)
    
    # Handle webcam capture
    if st.session_state.webcam_active and not st.session_state.captured_image_path:
        # Create a trigger file for the webcam capture function
        if capture_button:
            # Create a trigger file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.trigger')
            trigger_path = temp_file.name
            temp_file.close()
            
            # Save the trigger path in session state
            st.session_state.trigger_path = trigger_path
            
            # Create a trigger file that the webcam function will detect
            with open(trigger_path, 'w') as f:
                f.write('capture')
            
            # Start webcam capture in a separate thread to avoid blocking the UI
            def webcam_capture_thread():
                try:
                    # Capture the image
                    image_path = capture_image_from_webcam(webcam_placeholder)
                    
                    # Store the path in session state
                    st.session_state.captured_image_path = image_path
                    st.session_state.webcam_active = False
                    
                    # Force UI update
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error capturing image: {e}")
                    st.session_state.webcam_active = False
            
            # Start the capture thread
            thread = threading.Thread(target=webcam_capture_thread)
            thread.daemon = True
            thread.start()
        else:
            # Just display the webcam feed without capturing
            try:
                # Show webcam feed in placeholder
                webcam_placeholder.info("Camera is active. Click 'Capture Image' when ready.")
                
                # Create a static webcam display
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    webcam_placeholder.error("Could not open webcam. Please check your camera connection.")
                    st.session_state.webcam_active = False
                    st.experimental_rerun()
                
                # Display a single frame to show the webcam is working
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    webcam_placeholder.image(frame_rgb, caption="Webcam Feed", use_column_width=True)
                
                # Release the camera to avoid blocking it
                cap.release()
            except Exception as e:
                webcam_placeholder.error(f"Error accessing webcam: {e}")
                st.session_state.webcam_active = False
    
    # Display and process captured image if available
    if st.session_state.captured_image_path and os.path.exists(st.session_state.captured_image_path):
        # Process and display the captured image
        predict_and_display(st.session_state.captured_image_path)
        
        # Add button to retake photo
        if st.button("Retake Photo"):
            # Clean up
            if os.path.exists(st.session_state.captured_image_path):
                os.unlink(st.session_state.captured_image_path)
            
            # Reset the state
            st.session_state.captured_image_path = None
            st.session_state.webcam_active = True
            
            # Force a rerun
            st.experimental_rerun()

with tab3:
    st.header("Example Images")
    st.write("Try one of these example images to see how the app works")
    
    # Example images
    example_images = {
        "Pizza": "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?auto=format&fit=crop&w=600&q=80",
        "Salad": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?auto=format&fit=crop&w=600&q=80",
        "Burger": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?auto=format&fit=crop&w=600&q=80"
    }
    
    # Display example images in a row
    cols = st.columns(len(example_images))
    
    for i, (label, url) in enumerate(example_images.items()):
        with cols[i]:
            st.image(url, caption=label, use_column_width=True)
            if st.button(f"Try {label}", key=f"example_{i}"):
                # Download the example image
                with st.spinner(f"Processing {label} image..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        response = requests.get(url)
                        tmp_file.write(response.content)
                        tmp_file_path = tmp_file.name
                
                # Process the example image
                predict_and_display(tmp_file_path)
                
                # Clean up
                os.unlink(tmp_file_path)

with tab4:
    st.header("Ask for Calorie Information")
    st.write("Type in the name of a food item to get calorie information")
    
    # Text input for food query
    food_query = st.text_input("What food would you like to know about?", placeholder="E.g., butter chicken, sushi, pizza")
    
    # Serving size slider for text queries
    text_serving_size = st.slider(
        "Serving Size (grams)",
        min_value=50,
        max_value=500,
        value=100,
        step=10,
        help="Adjust the serving size for calorie calculation",
        key="text_serving_size"
    )
    
    # Process the query when the user submits it
    if food_query:
        with st.spinner(f"Looking up nutrition information for {food_query}..."):
            # Clean up the query (replace spaces with underscores for the API)
            food_item = food_query.lower().strip()
            
            # Get calorie information
            nutrition_info = calorie_estimator.get_calories(food_item, text_serving_size)
            
            # Create columns for display
            col1, col2 = st.columns([1, 1])
            
            # Display the results
            with col1:
                st.subheader("Nutrition Information")
                st.write(f"**Food:** {nutrition_info['food_item'].title()}")
                st.write(f"**Serving Size:** {nutrition_info['serving_size']}g")
                st.write(f"**Calories:** {nutrition_info['calories']} kcal")
                
                # Display nutrition facts if available
                if nutrition_info['protein'] is not None:
                    st.write(f"**Protein:** {nutrition_info['protein']}g")
                    st.write(f"**Fat:** {nutrition_info['fat']}g")
                    st.write(f"**Carbs:** {nutrition_info['carbs']}g")
                    
                    # Create a pie chart for macronutrients
                    with col2:
                        st.subheader("Macronutrient Breakdown")
                        fig, ax = plt.subplots(figsize=(4, 4))
                        labels = ['Protein', 'Fat', 'Carbs']
                        sizes = [nutrition_info['protein'], nutrition_info['fat'], nutrition_info['carbs']]
                        colors = ['#ff9999', '#66b3ff', '#99ff99']
                        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                        ax.axis('equal')
                        st.pyplot(fig)
                else:
                    with col2:
                        st.info("Detailed nutrition information (protein, fat, carbs) is not available for this food item in the fallback database.")
            
            # Show data source
            st.caption(f"Nutrition data source: {nutrition_info['source']}")
            
            # Suggest some related foods from our database
            st.subheader("You might also be interested in:")
            similar_foods = []
            query_words = set(food_item.split())
            
            # Find foods with similar words in the name
            for key in calorie_estimator.fallback_data.keys():
                key_words = set(key.replace('_', ' ').split())
                if query_words.intersection(key_words) and key.replace('_', ' ') != food_item:
                    similar_foods.append(key)
                if len(similar_foods) >= 5:  # Limit to 5 suggestions
                    break
            
            # If no similar foods found, suggest random ones from the same cuisine type
            if not similar_foods:
                cuisines = {
                    "indian": ["butter_chicken", "samosa", "biryani", "naan"],
                    "chinese": ["dim_sum", "fried_rice", "kung_pao_chicken"],
                    "japanese": ["sushi", "ramen", "tempura"],
                    "italian": ["pizza", "pasta", "lasagna"],
                    "mexican": ["tacos", "enchiladas"],
                    "american": ["burger", "french_fries", "apple_pie"]
                }
                
                # Try to determine the cuisine from the query
                detected_cuisine = None
                for cuisine, foods in cuisines.items():
                    for food in foods:
                        if food.replace('_', ' ') in food_item or food_item in food.replace('_', ' '):
                            detected_cuisine = cuisine
                            break
                    if detected_cuisine:
                        break
                
                # If cuisine detected, suggest foods from that cuisine
                if detected_cuisine:
                    similar_foods = [food for food in cuisines[detected_cuisine] 
                                    if food.replace('_', ' ') != food_item][:5]
                else:
                    # Otherwise, suggest some popular foods
                    similar_foods = ["pizza", "burger", "sushi", "pasta", "chicken"]
            
            # Display suggestions
            for food in similar_foods:
                calories = calorie_estimator.fallback_data.get(food, 150)
                st.write(f"- {food.replace('_', ' ').title()}: ~{calories} calories per 100g")

# Information section
with st.expander("How it works"):
    st.markdown("""
    This app uses computer vision and machine learning to identify food items and estimate their calorie content.
    
    **How it works:**
    1. Upload an image or capture with webcam
    2. A pre-trained Convolutional Neural Network (CNN) identifies the food item
    3. The app estimates calories using nutritional data
    
    **Limitations:**
    - The model can recognize about 100 common food items
    - Calorie estimates are approximate and based on standard serving sizes
    - Mixed dishes may be identified by their most prominent ingredient
    
    **Tips for best results:**
    - Use clear, well-lit photos
    - Center the food item in the image
    - Photograph from above for best recognition
    """)

# Footer
st.markdown("---")
st.caption("© 2025 Food Calorie Estimator | Built with Streamlit, PyTorch, and OpenCV") 