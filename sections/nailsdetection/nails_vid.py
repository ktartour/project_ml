import streamlit as st
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any
from inference_sdk import InferenceHTTPClient
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer


class VideoTransformer(VideoTransformerBase):
    def __init__(self, model_id: str, confidence_threshold: float):
        self.model_id = model_id
        self.confidence_threshold = confidence_threshold
        self.client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="yVnoBqLgjl2tRxWIWMvx"
        )

    def transform(self, frame: np.ndarray) -> np.ndarray:
        try:
            # Convert the frame to a PIL Image for compatibility with the inference model
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Run inference
            result: Dict[str, Any] = self.client.infer(frame_pil, model_id=self.model_id)
            output_dict: Dict[str, Any] = result  # Assume the result is already a dictionary

            # Filter predictions by confidence
            def filter_predictions(predictions: List[Dict[str, Any]], confidence_threshold: float) -> List[
                Dict[str, Any]]:
                return [
                    pred for pred in predictions
                    if pred.get('confidence', 0) >= confidence_threshold
                ]

            filtered_predictions = filter_predictions(output_dict.get('predictions', []), self.confidence_threshold)
            # If detections are found, annotate the frame
            if filtered_predictions:
                frame = self.draw_polygons_on_frame(frame, filtered_predictions)
        except Exception as e:
            st.error(f"Error processing frame: {e}")
        return frame

    def draw_polygons_on_frame(self, frame: np.ndarray, predictions: List[Dict[str, Any]]) -> np.ndarray:
        """Draw bounding boxes and polygons on a video frame."""
        for prediction in predictions:
            if 'points' in prediction:
                points = prediction['points']
                polygon_points = [(int(p['x']), int(p['y'])) for p in points]
                if len(polygon_points) >= 3:
                    cv2.polylines(frame, [np.array(polygon_points)], isClosed=True, color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, f"{prediction['class']} ({prediction['confidence']:.2f})",
                                (polygon_points[0][0], polygon_points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)
        return frame


def side_bar_nails():
    st.write('#### Set detection confidence threshold.')

    confidence_threshold: float = st.slider(
        'Confidence threshold: What is the acceptable confidence level?',
        0.0, 1.0, 0.5, 0.01, key="nailsvid"
    )
    st.write(f"Confidence threshold set to: {confidence_threshold}")
    return confidence_threshold


def nails_page():
    confidence_threshold = side_bar_nails()
    model_id = "laurent/1"
    # Initialize session state for the webcam
    if "run_webcam" not in st.session_state:
        st.session_state["run_webcam"] = False

    # Display both buttons: "Run" and "Stop"
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Run"):
            st.session_state["run_webcam"] = True  # Start the webcam

    with col2:
        if st.button("Stop"):
            st.session_state["run_webcam"] = False  # Stop the webcam

    # Run the webcam processing if "Run" was pressed
    if st.session_state["run_webcam"]:
        webrtc_streamer(
            key="nails-detection",
            video_transformer_factory=lambda: VideoTransformer(model_id, confidence_threshold)
        )
#