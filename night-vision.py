import cv2
import numpy as np

def enhance_night_vision(frame):
    """
    Enhance the frame to improve night vision.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization
    enhanced = cv2.equalizeHist(gray)
    
    # Convert back to BGR format
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced_bgr

def main():
    # Open the default camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Enhance the frame for night vision
        enhanced_frame = enhance_night_vision(frame)
        
        # Display the resulting frame
        cv2.imshow('Night Vision Glasses', enhanced_frame)
        
        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
