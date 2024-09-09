import streamlit as st
from PIL import Image
#from inference_sdk import InferenceHTTPClient
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

def side_bar_nails():
    # Sidebar: Allows the user to upload an image file (PNG, JPG, JPEG)
    st.sidebar.write('#### Select an image to upload.')
    uploaded_file = st.sidebar.file_uploader('', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False,key="sidebar")

    # Sidebar: Confidence threshold slider allows the user to set a minimum confidence level for predictions
    confidence_threshold: float = st.sidebar.slider(
        'Confidence threshold: What is the acceptable confidence level?',
        0.0, 1.0, 0.5, 0.01
    )
    # Sidebar: Display logos or images as visual elements
    st.sidebar.image(
        'https://cdn.prod.website-files.com/5f6bc60e665f54545a1e52a5/612ceede647190109abb0541_full-logo-p-500.png',
        use_column_width=True
    )
    st.sidebar.image(
        'https://miro.medium.com/v2/resize:fit:1400/0*5yINw4AB2CojpC0X.png',
        use_column_width=True
    )
    return uploaded_file,confidence_threshold
def initalize_variables():
    # Define types for better understanding of the data structures
    Prediction = Dict[str, Any]  # Predictions from the API could contain many fields, so using Any for flexibility
    PolygonPoints = List[Dict[str, float]]  # Each point in the polygon is a dictionary with 'x' and 'y' keys

    # Initialize the variable `nails` as None to hold the image (if uploaded)
    nails: Optional[Image.Image] = None
    return Prediction, PolygonPoints, nails

def print_info():
    # Main section: Title of the app and a subheading indicating the output will show an inferenced image
    st.write('# Nails Detection')
    st.write('### Inferenced Image')



# Function to display the uploaded image or a default image
def load_image(uploaded_file: Optional[st.runtime.uploaded_file_manager.UploadedFile]) -> Optional[Image.Image]:
    if uploaded_file is None:
        # Load and display a default image if no file is uploaded
        st.image('https://i8.amplience.net/i/Cosnova/3207001')
        st.write("## Waiting for an image to be uploaded.")
        return None
    else:
        # Open and display the uploaded image using PIL
        nails = Image.open(uploaded_file)
        st.image(nails, use_column_width=True)
        return nails

def treatment_nails(uploaded_file,confidence_threshold,Prediction,PolygonPoints):
    # Load the image (uploaded or default)
    nails = load_image(uploaded_file)
    # Proceed only if an image is loaded (either uploaded or default)
    if nails is not None:
        # Initialize the inference client to communicate with the Roboflow API
        CLIENT: InferenceHTTPClient = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="yVnoBqLgjl2tRxWIWMvx"
        )

        try:
            # Perform inference on the uploaded image using the model with ID 'laurent/1'
            result: Dict[str, Any] = CLIENT.infer(nails, model_id="laurent/1")
            output_dict: Dict[str, Any] = result  # Assume the result is already a dictionary
        except Exception as e:
            # If there's an error in inference, display an error message and stop execution
            st.error(f"Failed to obtain inference results: {e}")
            st.stop()

        # Filter predictions based on the confidence threshold set by the user
        def filter_predictions(predictions: List[Prediction], confidence_threshold: float) -> List[Prediction]:
            return [
                pred for pred in predictions
                if pred.get('confidence', 0) >= confidence_threshold
            ]

        # Get the filtered predictions
        filtered_predictions: List[Prediction] = filter_predictions(output_dict.get('predictions', []), confidence_threshold)

        # Display the filtered results (if any predictions meet the confidence threshold)
        if len(filtered_predictions) > 0:
            st.write('## Results')
            st.write(f"Detected nails: {len(filtered_predictions)}")
        else:
            st.warning("No nails detected with the specified confidence threshold.")

        # Function to draw polygons on the image
        def draw_polygons(ax: plt.Axes, predictions: List[Prediction], image: Image.Image) -> None:
            """Draws polygons on the given image based on predictions."""
            for prediction in predictions:
                if 'points' in prediction:
                    points: PolygonPoints = prediction['points']  # Extract points from the prediction
                    label: str = prediction.get('class', 'Unknown')  # Extract the class label
                    confidence: float = prediction.get('confidence', 0)  # Extract the confidence score

                    # Extract the x, y coordinates of each point in the polygon
                    polygon_points: List[Tuple[float, float]] = [(point['x'], point['y']) for point in points]

                    # Ensure that there are at least 3 points to form a valid polygon
                    if len(polygon_points) >= 3:
                        # Create a polygon patch (an outline) to overlay on the image
                        polygon_patch = patches.Polygon(
                            polygon_points,
                            closed=True,  # Ensure the polygon is closed
                            edgecolor='red',  # Color of the polygon's edge
                            fill=False,  # Do not fill the polygon with color
                            linewidth=2  # Set the thickness of the polygon's edge
                        )
                        ax.add_patch(polygon_patch)  # Add the polygon to the image

                        # Calculate the centroid (center) of the polygon to place text
                        centroid_x: float = np.mean([p[0] for p in polygon_points])
                        centroid_y: float = np.mean([p[1] for p in polygon_points])

                        # Add the class label and confidence score at the centroid of the polygon
                        ax.text(
                            centroid_x, centroid_y,  # Position of the text
                            f'{label} ({confidence:.2f})',  # Text content (class and confidence)
                            color='red',  # Color of the text
                            fontsize=12,  # Size of the font
                            ha='center'  # Horizontal alignment
                        )
                    else:
                        # If there are fewer than 3 points, display a warning
                        st.warning("Prediction with less than 3 points, unable to draw a polygon.")
                else:
                    # If no points are found in the prediction, display a warning
                    st.warning("No 'points' key found in a prediction.")

        # Create a matplotlib figure and axis for plotting the image and polygons
        fig, ax = plt.subplots()
        ax.imshow(nails)  # Display the image

        # Draw the polygons on the image
        draw_polygons(ax, filtered_predictions, nails)

        # Display the image with annotations (polygons and labels)
        st.write('### Annotated Image')
        st.pyplot(fig)  # Show the annotated image in the Streamlit app

def nails_page():
    st.write("ok")
    return None