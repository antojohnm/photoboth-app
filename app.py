# Import required libraries
import cv2  # OpenCV for computer vision tasks
import numpy as np  # Numerical computing library
import streamlit as st  # Web application framework
import time  # Time-related functions
import base64  # Base64 encoding for image downloads

# Load Haar Cascade for smile detection using OpenCV's pre-trained model
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Create Streamlit web application
st.title("Photobooth 📸")  # Set main page title

# Initialize session states to preserve values between app reruns
if 'last_capture' not in st.session_state:
    st.session_state.update({
        'last_capture': 0,  # Timestamp of last photo capture
        'download_ready': None,  # Status of downloadable image
        'capture_count': 0  # Counter for total captures
    })

# Set up webcam capture (0 = default camera)
cap = cv2.VideoCapture(0)

# Create Streamlit UI elements for dynamic content
frame_window = st.image([], channels="BGR", use_container_width=True)  # Webcam feed display
download_placeholder = st.empty()  # Placeholder for download link

# Photobooth configuration parameters
COOLDOWN = 3  # Minimum seconds between captures
DETECTION_THRESHOLD = 5  # Required consecutive smile detection frames
detection_counter = 0  # Counter for consecutive smile detections

# Create exit button (placed outside main loop for persistent visibility)
exit_button = st.button("🚪 Exit Photobooth", key="unique_exit_button_12345")

# Main application loop
while True:
    if exit_button:  # Check if exit button was clicked
        break

    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:  # Error handling for failed frame capture
        st.error("Failed to grab frame")
        break

    # Convert frame to grayscale for facial detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smiles = smile_cascade.detectMultiScale(gray, 1.8, 20)  # Detect smiles in frame
    
    # Calculate time remaining until next allowed capture
    current_time = time.time()
    cooldown_remaining = COOLDOWN - (current_time - st.session_state.last_capture)

    # Smile detection and capture logic
    if len(smiles) > 0:  # If smiles are detected
        detection_counter += 1  # Increment consecutive detection counter
        
        # Check if conditions met for capture (threshold reached and cooldown passed)
        if detection_counter >= DETECTION_THRESHOLD and cooldown_remaining <= 0:
            st.session_state.capture_count += 1  # Update total capture counter
            
            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            img_path = f"smile_{timestamp}.jpg"
            
            # Process image for download
            _, buffer = cv2.imencode('.jpg', frame)  # Convert frame to JPEG bytes
            img_bytes = base64.b64encode(buffer).decode()  # Encode in base64 for HTML download
            
            # Create styled download link with HTML
            download_html = f'''
                <a href="data:image/jpg;base64,{img_bytes}" 
                   download="{img_path}"
                   style="
                       display: inline-block;
                       padding: 0.5em 1em;
                       color: white;
                       background-color: #FF4B4B;
                       border-radius: 0.5em;
                       text-decoration: none;
                   ">
                   ⬇️ Download Latest Photo
                </a>
            '''
            
            # Update UI elements
            download_placeholder.markdown(download_html, unsafe_allow_html=True)
            st.session_state.last_capture = current_time  # Reset cooldown timer
            detection_counter = 0  # Reset detection counter

            # Show success message
            st.success(f"📸 Captured! Click the link above to download '{img_path}'")

    else:  # If no smiles detected
        detection_counter = max(0, detection_counter - 1)  # Gradually decrease counter

    # Add visual overlays to webcam feed
    for (x, y, w, h) in smiles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green boxes around smiles
    
    # Display cooldown timer if active
    if cooldown_remaining > 0:
        timer_text = f"Next capture available in: {max(0, int(cooldown_remaining))}s"
        cv2.putText(frame, timer_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Update webcam feed in Streamlit interface
    frame_window.image(frame, channels="BGR")

    # Double-check exit condition
    if exit_button:
        break

# Cleanup resources when exiting
cap.release()  # Release webcam
cv2.destroyAllWindows()  # Close any OpenCV windows
