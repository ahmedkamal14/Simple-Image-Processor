import streamlit as st
import cv2
import numpy as np
from streamlit_image_comparison import image_comparison

st.set_page_config(
    page_title="Image Processor",
    layout="centered",
)
st.title("📸 Simple Image Processing with OpenCV")

st.markdown("""This app allows you to upload an image and apply basic OpenCV filters.""")

st.header("Upload an Image")

img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

with st.expander("Preview Image"):
    if img is not None:
        img_bytes = img.read()  # Read once
        file_bytes = np.asarray(bytearray(img_bytes), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, channels="BGR", caption="Uploaded Image")

if img is not None:
    st.header("Apply Filters")
    # reuse the already read img_bytes
    file_bytes = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    display_img = cv2.imdecode(file_bytes, 1)
    filtered_img = display_img.copy()
    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)


    col1, col2, col3 = st.columns(3)

    # Checkboxes for different filters
    convert_to_gray = col1.checkbox("Convert to Grayscale")
    apply_blur = col2.checkbox("Apply Gaussian Blur")
    invert_colors = col3.checkbox("Invert Colors")
    rotate_image = st.checkbox("Rotate Image")

    # Apply filters based on user selection
    if convert_to_gray:
        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2GRAY)
    if apply_blur:
        k_size = st.slider("Select Gaussian Blur Kernel Size", 1, 51, 15, step=2)
        filtered_img = cv2.GaussianBlur(filtered_img, (k_size, k_size), 0)
    if invert_colors:
        filtered_img = cv2.bitwise_not(filtered_img)
    if rotate_image:
        angle = st.slider("Select Rotation Angle", 0, 360, 0)
        h, w = filtered_img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        filtered_img = cv2.warpAffine(filtered_img, rotation_matrix, (w, h))

    image_comparison(
        img1=display_img,
        img2=filtered_img,
        label1="Original",
        label2="Filtered",
    )

    st.header("Contrast and Brightness Adjustment")

    brightness = st.slider("Adjust Brightness", -100, 100, 0)
    contrast = st.slider("Adjust Contrast", 0.5, 3.0, 1.0)

    with st.expander("Adjusted Image"):
        adjusted_img = cv2.convertScaleAbs(filtered_img, alpha=contrast, beta=brightness)
        st.image(adjusted_img, channels="RGB", caption="Adjusted Image", use_container_width=True)
        st.download_button(
            label="Download Adjusted Image",
            data=cv2.imencode('.png', adjusted_img)[1].tobytes(),
            file_name='adjusted_image.png',
            mime='image/png'
        )
        
    st.header("Extra Filters")

    col4, col5 = st.columns(2)

    # Checkboxes for additional filters
    apply_canny = col4.checkbox("Edge Detection (Canny)")
    apply_thresh = col5.checkbox("Apply Thresholding")

    st.info("Please note that these filters are applied to the filtered image Without Contrast and Brightness.")
    
    if apply_canny:
        with st.expander("Canny Edge Detection Parameters"):
            low_threshold = st.slider("Low Threshold", 0, 255, 100)
            high_threshold = st.slider("High Threshold", 0, 255, 200)
        canny_edges = cv2.Canny(filtered_img, low_threshold, high_threshold)

        with st.expander("Canny Edge Detection Result"):
            st.image(canny_edges, channels="GRAY", caption="Canny Edges", use_container_width=True)
            st.download_button(
                label="Download Canny Edges",
                data=cv2.imencode('.png', canny_edges)[1].tobytes(),
                file_name='canny_edges.png',
                mime='image/png'
            )
    if apply_thresh:
        with st.expander("Thresholding Parameters"):
            thresh_value = st.slider("Threshold Value", 0, 255, 127)
        _, thresh_img = cv2.threshold(filtered_img, thresh_value, 255, cv2.THRESH_BINARY)

        with st.expander("Thresholding Result"):
            st.image(thresh_img, channels="GRAY", caption="Thresholded Image", use_container_width=True)
            st.download_button(
                label="Download Thresholded Image",
                data=cv2.imencode('.png', thresh_img)[1].tobytes(),
                file_name='thresholded_image.png',
                mime='image/png'
            )
        