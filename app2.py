# import streamlit as st
# from PIL import Image
# from torchvision.transforms import functional as F
# from yolov5.models.experimental import attempt_load
# from yolov5.utils.general import non_max_suppression

# # Load the YOLO model
# model = attempt_load("./weights/best.pt")

# # Function to perform object detection on the image
# def perform_object_detection(image):
#     try:
#         # Preprocess the image
#         image = F.to_tensor(image)
#         image = image.unsqueeze(0)

#         # Perform object detection
#         detections = model(image)

#         # Apply non-maximum suppression to get the most confident predictions
#         detections = non_max_suppression(detections, conf_thres=0.5, iou_thres=0.5)

#         return detections
#     except Exception as e:
#         st.error(f"Error during object detection: {e}")
#         return None

# # Function to check if the image is dyslexic or not
# def check_dyslexia(image):
#     try:
#         # Perform object detection on the image
#         detections = perform_object_detection(image)

#         # Handle empty or invalid detections
#         if detections and len(detections) > 0 and detections[0] is not None and len(detections[0]) > 0:
#             return True
#         else:
#             return False
#     except Exception as e:
#         st.error(f"Error during dyslexia detection: {e}")
#         return False

# # Streamlit app
# def main():
#     st.title("Dyslexia Detection App")
#     uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

#     if uploaded_image is not None:
#         image = Image.open(uploaded_image)
#         st.image(image, caption="Uploaded Image", use_column_width=True)

#         # Resize the image to a specific size
#         resized_image = image.resize((224, 224))  # You can change the dimensions as per your requirement

#         if st.button("Detect Dyslexia"):
#             is_dyslexic = check_dyslexia(resized_image)
#             if is_dyslexic:
#                 st.write("**The image is dyslexic.**")
#             else:
#                 st.write("**The image is not dyslexic.**")

# if __name__ == "__main__":
#     main()


import streamlit as st
from PIL import Image
from torchvision.transforms import functional as F
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression

# Load the YOLO model
model = attempt_load("./weights/best10.pt")

# Function to perform object detection on the image
def perform_object_detection(image):
    try:
        # Preprocess the image
        image = F.to_tensor(image)
        image = image.unsqueeze(0)

        # Perform object detection
        detections = model(image)

        # Apply non-maximum suppression to get the most confident predictions
        detections = non_max_suppression(detections, conf_thres=0.3, iou_thres=0.5)
        print(detections)

        return detections
    except Exception as e:
        st.error(f"Error during object detection: {e}")
        return None

# Function to check if the image is dyslexic or not
def check_dyslexia(image):
    try:
        # Perform object detection on the image
        detections = perform_object_detection(image)

        # Handle empty or invalid detections
        if detections and len(detections) > 0 and detections[0] is not None and len(detections[0]) > 0:
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Error during dyslexia detection: {e}")
        return False

# Streamlit app
def main():
    st.title("Dyslexia Detection App")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Resize the image to a specific size
        resized_image = image.resize((224, 224))  # You can change the dimensions as per your requirement

        if st.button("Detect Dyslexia"):
            is_dyslexic = check_dyslexia(resized_image)
            if is_dyslexic:
                st.write("**The image is dyslexic.**")
            else:
                st.write("**The image is not dyslexic.**")

if __name__ == "__main__":
    main()
