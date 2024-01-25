import streamlit as st
import requests
from PIL import Image, ImageDraw
import os
from base64 import decodebytes
import numpy as np

##########
##### Set up sidebar.
##########

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Leprosy Detector",
    page_icon = ":health:",
    initial_sidebar_state = 'auto'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) # hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML

def prediction_cls(prediction): # predict the class of the images based on the model results
    for key, clss in class_names.items(): # create a dictionary of the output classes
        if np.argmax(prediction)==clss: # check the class
            
            return key

with st.sidebar:
        st.title("Leprosy")
        st.subheader("Accessible deep learning model to predict whether or not you have Leprosy.")

st.write("""
         # LEPROSY DETECTOR
         """
         )

uploaded_file = st.sidebar.file_uploader('',
                                         type=['png', 'jpg', 'jpeg'],
                                         accept_multiple_files=False)

## Pull in default image or user-selected image.
if uploaded_file is None:
    # Default image.
    url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)

else:
    ## Construct the URL to retrieve image.
    url = "https://api.ultralytics.com/v1/predict/uGSF6K8CcSiY0sWOK0JC"
    headers = {"x-api-key": "8584421e457423aff1f44eb39986b002dfeef914bb"}
    data = {"size": 640, "confidence": 0.25, "iou": 0.45}

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Save the uploaded file locally
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    files2 = {"image": (uploaded_file.name, open(file_path, "rb"))}
      

    ## POST to the API.
    response = requests.post(url, headers=headers, data=data, files=files2)

    response_json = response.json()
    if 'data' in response_json and len(response_json['data']) > 0:
            # Open the uploaded image using PIL
            pil_image = Image.open(uploaded_file)

            # Get image dimensions
            image_width, image_height = pil_image.size

            # Create a drawing object to draw on the image
            draw = ImageDraw.Draw(pil_image)

            # Draw bounding boxes on the image
            for detection in response_json['data']:
                x_center = int(detection['xcenter'] * image_width)
                y_center = int(detection['ycenter'] * image_height)
                box_width = int(detection['width'] * image_width)
                box_height = int(detection['height'] * image_height)

                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)
                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)

                # Draw the bounding box
                draw.rectangle([x1, y1, x2, y2], outline="green", width=2)

                # Display class name and confidence
                class_name = detection['name']
                confidence = detection['confidence']
                label = f"{class_name} {confidence:.2f}"
                draw.text((x1, y1 - 15), label, fill="green")

            # Display the image with Streamlit
            st.image(pil_image, caption="...oh dear", use_column_width=True)
            st.write('Unfortunately, we detected signs of leprosy. Perhaps you should go to the doctor.')

    else:
      st.write("# Thank God, we don't detect any leprosy...")
