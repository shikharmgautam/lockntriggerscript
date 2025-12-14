import cv2
import time
from detector import HumanDetector
from roi_manager import ROIManager
import numpy as np

def main(video_source=0):
    """
    Main loop for the CV Alert System.
    video_source: Path to video file or camera index (default 0 for webcam).
    """
    # 1. Initialize Detector with optimized settings for long-distance detection
    print("Initializing YOLO detector with enhanced settings...")
    detector = HumanDetector(
        model_path='yolov8m-seg.pt',  # Larger model for better accuracy
        confidence=0.15,               # Lower threshold for distant objects
        imgsz=1024,                    # Higher resolution for small objects
        use_preprocessing=True         # CLAHE contrast enhancement
    )
    
    # 2. Initialize Video Capture
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    # Read the first frame to determine video size for ROI setting
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return
    
    height, width = frame.shape[:2]
    
    # 3. Define ROI (hardcoded generic polygon for demo - center area)
    # Adjust these coordinates based on your specific camera view
    # Format: [(x, y), ...]
    polygon_points = [
        (int(width * 0.2), int(height * 0.2)),
        (int(width * 0.8), int(height * 0.2)),
        (int(width * 0.8), int(height * 0.8)),
        (int(width * 0.2), int(height * 0.8))
    ]
    roi_manager = ROIManager(polygon_points)

    print("Starting video loop using source:", video_source)
    print("Press 'q' to quit.")

def process_frame(frame, detector, roi_manager):
    """
    Process a single frame: Detect humans, check ROI, draw annotations.
    Returns: processed_frame, alert_triggered (bool)
    """
    # 4. Detect Humans
    detections = detector.detect(frame)
    
    alert_triggered = False

    # 5. Process Detections
    # 5. Process Detections
    for det in detections:
        x1, y1, x2, y2 = det['box']
        mask = det['mask']
        conf = det['conf']
        
        # Check Mask Intersection
        is_inside = roi_manager.is_mask_intersecting(mask)
        
        # Visualization
        color = (0, 255, 0) # Green if safe
        if is_inside:
            color = (0, 0, 255) # Red if alerting
            alert_triggered = True
            
            # Draw alert indicator
            cv2.putText(frame, "ALERT!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Overlay Mask
        # Create a colored overlay
        mask_overlay = np.zeros_like(frame)
        mask_overlay[mask > 0.5] = color
        
        # Blend overlay (alpha=0.4)
        cv2.addWeighted(mask_overlay, 0.4, frame, 1, 0, frame)

    # 6. Global Alert (if any person is inside)
    roi_color = (0, 255, 0)
    if alert_triggered:
        roi_color = (0, 0, 255)
        # Display big alert on screen
        cv2.putText(frame, "INTRUSION DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Draw ROI
    roi_manager.draw_roi(frame, color=roi_color)
    
    return frame, alert_triggered

def main(video_source=0):
    """
    Main loop for the CV Alert System.
    video_source: Path to video file or camera index (default 0 for webcam).
    """
    # 1. Initialize Detector with MAXIMUM ACCURACY settings (for GPU/Colab)
    print("Initializing YOLO detector with maximum accuracy settings...")
    detector = HumanDetector(
        model_path='yolov8x-seg.pt',  # Largest model - highest accuracy
        confidence=0.10,               # Very low threshold - catch everything
        imgsz=1280,                    # Maximum resolution
        use_preprocessing=True         # CLAHE contrast enhancement enabled
    )
    
    # 2. Initialize Video Capture
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    # Read the first frame to determine video size for ROI setting
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return
    
    height, width = frame.shape[:2]
    
    # 3. Define ROI (hardcoded generic polygon for demo - center area)
    # Adjust these coordinates based on your specific camera view
    # Format: [(x, y), ...]
    # 3. Define ROI (hardcoded generic polygon for demo - center area)
    # Adjust these coordinates based on your specific camera view
    # Format: [(x, y), ...]
    # 3. Define ROI - TINY BOX (0.49 to 0.51) to verify change
    polygon_points = [
        (int(width * 0.49), int(height * 0.49)),
        (int(width * 0.51), int(height * 0.49)),
        (int(width * 0.51), int(height * 0.51)),
        (int(width * 0.49), int(height * 0.51))
    ]
    # Pass dimensions to ROIManager
    roi_manager = ROIManager(polygon_points, width, height)

    print("Starting video loop using source:", video_source)
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video or error

        frame, alert_triggered = process_frame(frame, detector, roi_manager)
        
        if alert_triggered:
            # In a real app, you might send an email, play a sound, etc. here.
            print("!!! ALERT: Human inside ROI !!!") 

        # Show Frame
        cv2.imshow("CV Human Alert System", frame)

        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    # Allow passing video path as argument
    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]
    
    try:
        # Check if source is digit (camera index)
        source = int(source)
    except ValueError:
        pass # It's a file path
        
    main(source)
