import cv2
import numpy as np
import streamlit as st
import time
import base64

# Load Haar Cascade for smile detection
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Create Streamlit app
st.title("Photobooth 📸")

# Initialize session states
if 'last_capture' not in st.session_state:
    st.session_state.update({
        'last_capture': 0,
        'download_ready': None,
        'capture_count': 0
    })

# Set up the webcam capture
cap = cv2.VideoCapture(0)

# Create placeholders for dynamic elements
frame_window = st.image([], channels="BGR", use_container_width=True)
download_placeholder = st.empty()

# Configuration
COOLDOWN = 3  # Seconds between captures
DETECTION_THRESHOLD = 5  # Minimum consecutive smile frames required
detection_counter = 0

# Create exit button outside the main loop
exit_button = st.button("🚪 Exit Photobooth", key="unique_exit_button_12345")

while True:
    if exit_button:
        break

    ret, frame = cap.read()
    if not ret:
        st.error("Failed to grab frame")
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smiles = smile_cascade.detectMultiScale(gray, 1.8, 20)
    
    current_time = time.time()
    cooldown_remaining = COOLDOWN - (current_time - st.session_state.last_capture)

    # Smile detection logic
    if len(smiles) > 0:
        detection_counter += 1
        if detection_counter >= DETECTION_THRESHOLD and cooldown_remaining <= 0:
            # Update capture counter for unique keys
            st.session_state.capture_count += 1
            
            # Capture and process image
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            img_path = f"smile_{timestamp}.jpg"
            
            # Convert frame to bytes and base64
            _, buffer = cv2.imencode('.jpg', frame)
            img_bytes = base64.b64encode(buffer).decode()
            
            # Create download link
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
            
            # Update download components
            download_placeholder.markdown(download_html, unsafe_allow_html=True)
            st.session_state.last_capture = current_time
            detection_counter = 0

            # Display success message
            st.success(f"📸 Captured! Click the link above to download '{img_path}'")

    else:
        detection_counter = max(0, detection_counter - 1)

    # Draw UI overlays
    for (x, y, w, h) in smiles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    if cooldown_remaining > 0:
        timer_text = f"Next capture available in: {max(0, int(cooldown_remaining))}s"
        cv2.putText(frame, timer_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Update live feed
    frame_window.image(frame, channels="BGR")

    # Check if exit button was pressed again
    if exit_button:
        break

# Cleanup resources
cap.release()
cv2.destroyAllWindows()