import streamlit as st
import Data
import Results
import Playground

# Set page configuration
st.set_page_config(
    page_title="Deepfake Faces Detection Project",
    page_icon="🤖",
    layout="wide",
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ("Overview", "Data", "Results", "Playground"),
)

# Load corresponding page based on selection
if page == "Overview":
    # Main title
    st.title("Deepfake Faces Detection Project")
    st.subheader("Multi-Face Forgery Detection")

    # Introduction section
    with st.container():
        st.header("Introduction")
        st.write(
            """
            The rapid advancement of artificial intelligence and computer vision technologies has led to the creation of hyper-realistic manipulated media, commonly known as deepfakes. 
            These synthetic media pose significant challenges in identifying truth from deception. 
            This project utilizes the OpenForensics dataset and ConvNeXtV2 model to enhance deepfake detection capabilities, addressing the growing risks associated with manipulated content.
            """
        )

    # Create tabs for organization
    tabs = st.tabs(["Problem Statement", "Objectives", "Dataset Details"])

    # Problem Statement tab
    with tabs[0]:
        st.header("Problem Statement")
        st.write(
            """
            1. The hyper-realistic quality of deepfakes makes it challenging to distinguish authentic from manipulated content.
            2. The volume of media shared online overwhelms manual review processes, allowing deepfake content to spread before detection.
            """
        )

    # Objectives tab
    with tabs[1]:
        st.header("Objectives")
        st.write(
            """
            1. To **develop** a system for face detection and classification to distinguish between authentic and deepfake faces in an image.
            1. To **evaluate** the performance of of the deepfake face detection model
            3. To **present** a web application for deepfake face detection
            """
        )

    # Dataset Details tab
    with tabs[2]:
        st.header("Dataset Details")
        st.write(
            """
            The OpenForensics dataset, introduced in ICCV 2021, is designed for multi-face forgery detection and segmentation in real-world scenarios. 
            It consists of:
            - **115,325 high-quality images** with over 330,000 labeled faces.
            - Both real and GAN-manipulated faces, offering a comprehensive resource for training and evaluation.
            """
        )
        # st.image("https://via.placeholder.com/600x300", caption="Sample Image from OpenForensics Dataset", use_column_width=True)

elif page == "Data":
    
    Data.show()  # Ensure `data.show()` is defined in the corresponding module

elif page == "Results":
    
    Results.show()  # Ensure `results.show()` is defined in the corresponding module

elif page == "Playground":
    
    Playground.show()  # Ensure `playground.show()` is defined in the corresponding module
