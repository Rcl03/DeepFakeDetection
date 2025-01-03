import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import os

# Streamlit page configuration
# st.set_page_config(page_title="Face Analysis Dashboard", layout="wide")
def show():
    # Title
    st.title("Dataset")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Training Dataset", "Test Dataset","Sample Image"])

    # Training Dataset Tab
    with tab1:
        # Path to the CSV file
        CSV_PATH_train = os.path.join(os.path.dirname(__file__), "Metadata", "Metadata_train.csv")

        try:
            # Load the CSV file
            df_train = pd.read_csv(CSV_PATH_train)

            # Calculate Summary Data
            total_images = df_train.shape[0]
            total_faces = df_train["No. of Faces"].sum()
            total_real_faces = df_train["Real Face Count"].sum()
            total_fake_faces = df_train["Fake Face Count"].sum()

            # Categorize images
            all_real = df_train[(df_train["Real Face Count"] > 0) & (df_train["Fake Face Count"] == 0)].shape[0]
            all_fake = df_train[(df_train["Fake Face Count"] > 0) & (df_train["Real Face Count"] == 0)].shape[0]
            mixed_faces = df_train[(df_train["Real Face Count"] > 0) & (df_train["Fake Face Count"] > 0)].shape[0]

            # Layout: Top Cards
            st.write("### Summary Statistics")
            col1, col2 = st.columns(2)

            with col1:
                st.metric(label="Total Number of Images", value=f"{total_images:,}")
                st.metric(label="Total Number of Faces", value=f"{total_faces:,}")

            with col2:
                st.metric(label="Total Real Faces", value=f"{total_real_faces:,}")
                st.metric(label="Total Fake Faces", value=f"{total_fake_faces:,}")

            # Layout: Side-by-Side Pie Charts
            st.write("---")  # Separator
            st.write("### Distribution of Faces")
            col1, col2 = st.columns(2)

            with col1:
                st.write("#### Images by Face Type")
                fig1, ax1 = plt.subplots(figsize=(6, 6))
                labels1 = ["All Real Faces", "All Fake Faces", "Mixed Faces"]
                sizes1 = [all_real, all_fake, mixed_faces]
                colors1 = ["#8dd3c7", "#fb8072", "#80b1d3"]
                ax1.pie(sizes1, labels=labels1, autopct="%1.1f%%", colors=colors1, startangle=140, textprops={'fontsize': 10})
                ax1.axis('equal')  # Ensures the pie is circular
                fig1.tight_layout()  # Adjusts spacing automatically]
                ax1.set_facecolor('none')
                fig1.patch.set_facecolor('none')
                ax1.set_title("Images by Face Type")
                st.pyplot(fig1)

            with col2:
                st.write("#### Real vs Fake Faces")
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                labels2 = ["Real Faces", "Fake Faces"]
                sizes2 = [total_real_faces, total_fake_faces]
                colors2 = ["#8dd3c7", "#fb8072"]
                ax2.pie(sizes2, labels=labels2, autopct="%1.1f%%", colors=colors2, startangle=140, textprops={'fontsize': 10})
                ax2.axis('equal')  # Ensures the pie is circular
                ax2.set_facecolor('none')
                fig2.tight_layout() 
                fig2.patch.set_facecolor('none')
                ax2.set_title("Total Real vs Fake Faces")
                st.pyplot(fig2)


            # Layout: Bar Chart
            st.write("---")  # Separator
            st.write("### Number of Images by Face Count")

            # Bar Chart Data
            face_count_data = df_train["No. of Faces"].value_counts().sort_index()

            # Plot Bar Chart
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.bar(face_count_data.index, face_count_data.values, color="#80b1d3")
            ax3.set_xlabel("Number of Faces in an Image")
            ax3.set_ylabel("Number of Images")
            ax3.set_title("Images Grouped by Face Count")
            ax3.set_xticks(face_count_data.index)

            # Display the bar chart
            st.pyplot(fig3)

        except FileNotFoundError:
            st.error(f"File not found at path: {CSV_PATH_train}")
        except pd.errors.EmptyDataError:
            st.error("The CSV file is empty. Please ensure it contains data.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    # Test Dataset Tab
    with tab2:
        # Path to the CSV file
        CSV_PATH_test=os.path.join(os.path.dirname(__file__), "Metadata", "Metadata_test.csv")

        try:
            # Load the CSV file
            df_test = pd.read_csv(CSV_PATH_test)

            # Calculate Summary Data
            total_images = df_test.shape[0]
            total_faces = df_test["No. of Faces"].sum()
            total_real_faces = df_test["Real Face Count"].sum()
            total_fake_faces = df_test["Fake Face Count"].sum()

            # Categorize images
            all_real = df_test[(df_test["Real Face Count"] > 0) & (df_test["Fake Face Count"] == 0)].shape[0]
            all_fake = df_test[(df_test["Fake Face Count"] > 0) & (df_test["Real Face Count"] == 0)].shape[0]
            mixed_faces = df_test[(df_test["Real Face Count"] > 0) & (df_test["Fake Face Count"] > 0)].shape[0]

            # Layout: Top Cards
            st.write("### Summary Statistics")
            col1, col2 = st.columns(2)

            with col1:
                st.metric(label="Total Number of Images", value=f"{total_images:,}")
                st.metric(label="Total Number of Faces", value=f"{total_faces:,}")

            with col2:
                st.metric(label="Total Real Faces", value=f"{total_real_faces:,}")
                st.metric(label="Total Fake Faces", value=f"{total_fake_faces:,}")

            # Layout: Side-by-Side Pie Charts
            st.write("---")  # Separator
            st.write("### Distribution of Faces")
            col1, col2 = st.columns(2)

            with col1:
                # Pie Chart: Images by Face Type
                st.write("#### Images by Face Type")
                fig1, ax1 = plt.subplots(figsize=(6, 6))
                labels1 = ["All Real Faces", "All Fake Faces", "Mixed Faces"]
                sizes1 = [all_real, all_fake, mixed_faces]
                colors1 = ["#8dd3c7", "#fb8072", "#80b1d3"]
                ax1.pie(sizes1, labels=labels1, autopct="%1.1f%%", colors=colors1, startangle=140)
                ax1.set_title("Images by Face Type")
                st.pyplot(fig1)

            with col2:
                # Pie Chart: Total Real vs Fake Faces
                st.write("#### Real vs Fake Faces")
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                labels2 = ["Real Faces", "Fake Faces"]
                sizes2 = [total_real_faces, total_fake_faces]
                colors2 = ["#8dd3c7", "#fb8072"]
                ax2.pie(sizes2, labels=labels2, autopct="%1.1f%%", colors=colors2, startangle=140)
                ax2.set_title("Total Real vs Fake Faces")
                st.pyplot(fig2)

            # Layout: Bar Chart
            st.write("---")  # Separator
            st.write("### Number of Images by Face Count")

            # Bar Chart Data
            face_count_data = df_test["No. of Faces"].value_counts().sort_index()

            # Plot Bar Chart
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.bar(face_count_data.index, face_count_data.values, color="#80b1d3")
            ax3.set_xlabel("Number of Faces in an Image")
            ax3.set_ylabel("Number of Images")
            ax3.set_title("Images Grouped by Face Count")
            ax3.set_xticks(face_count_data.index)

            # Display the bar chart
            st.pyplot(fig3)

        except FileNotFoundError:
            st.error(f"File not found at path: {CSV_PATH_test}")
        except pd.errors.EmptyDataError:
            st.error("The CSV file is empty. Please ensure it contains data.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    with tab3:
        st.title("Sample Images")

        # Create tabs for Sample Image 1 and Sample Image 2
        sample_tab1, sample_tab2 = st.tabs(["Least Face", "Most face"])

        with sample_tab1:
            # Display Sample Image 1 and its annotated version side by side
            col1, col2 = st.columns(2)
            with col1:
                img1_path = os.path.join(os.path.dirname(__file__), "10ddfd6652.jpg")
                img1 = Image.open(img1_path)
                st.image(img1, caption="Sample Image 1 (Without Annotation)", use_container_width=True)

            with col2:
                img3_path = os.path.join(os.path.dirname(__file__), "annotated_image_10ddfd6652.jpg")
                img3 = Image.open(img3_path)
                st.image(img3, caption="Sample Image 1 (With Annotation)", use_container_width=True)

        with sample_tab2:
            # Display Sample Image 2 and its annotated version side by side
            col3, col4 = st.columns(2)
            with col3:
                img2_path = os.path.join(os.path.dirname(__file__), "138cb1e248.jpg")
                img2 = Image.open(img2_path)
                st.image(img2, caption="Sample Image 2 (Without Annotation)", use_container_width=True)

            with col4:
                img4_path = os.path.join(os.path.dirname(__file__), "annotated_image_138cb1e248.jpg")
                img4 = Image.open(img4_path)
                st.image(img4, caption="Sample Image 2 (With Annotation)", use_container_width=True)


