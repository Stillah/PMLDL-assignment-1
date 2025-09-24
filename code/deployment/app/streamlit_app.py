import streamlit as st
import requests
import io
from PIL import Image


FASTAPI_URL = "http://127.0.0.1:8000/predict"

def main():
    st.set_page_config(page_title="Image Classifier", page_icon="📷", layout="centered")
    
    st.title("🖼️ Cloths and accessories classification")
    st.markdown("Upload an image to test the classification model")
    
    st.markdown("---")
    

    st.subheader("Upload Your Image From The Dataset")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Supported formats: JPG, JPEG, PNG, BMP"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width='stretch')
        
        if st.button("🔍 Classify Uploaded Image", width='stretch'):
            classify_uploaded_image(uploaded_file)


def classify_uploaded_image(uploaded_file):
    """Classify the uploaded image"""
    try:
        # Prepare file for upload
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        
        with st.spinner('🔬 Analyzing image...'):
            response = requests.post(FASTAPI_URL, files=files)
        
        if response.status_code == 200:
            display_results(response.json())
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")

def display_results(result):
    """Display the prediction results in a nice format"""
    st.success("✅ Classification Complete!")
    
    # Main results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Prediction Results")
        st.metric(
            label="Predicted Class", 
            value=result.get('predicted_class', 'Unknown'),
            delta=f"{result.get('confidence', 0) * 100:.2f}% confidence"
        )
    
    with col2:
        confidence = result.get('confidence', 0) * 100
        st.write(f"**Confidence:** {confidence:.2f}%")



if __name__ == "__main__":
    main()