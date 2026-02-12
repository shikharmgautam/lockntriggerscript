import cv2
import time
from detector import HumanDetector
from roi_manager import ROIManager
import numpy as np


def process_frame(frame, detector, roi_manager):
    """
    Process a single frame: Detect humans, check ROI, draw annotations.
    Returns: processed_frame, alert_triggered (bool)
    """
    detections = detector.detect(frame)
    alert_triggered = False

    for det in detections:
        x1, y1, x2, y2 = det['box']
        mask = det['mask']
        conf = det['conf']
        
        # Check Mask Intersection
        is_inside = roi_manager.is_mask_intersecting(mask)
        
        # Visualization
        color = (0, 255, 0)  # Green if safe
        if is_inside:
            color = (0, 0, 255)  # Red if alerting
            alert_triggered = True
            
            # Draw alert indicator with confidence
            cv2.putText(frame, f"ALERT! ({conf:.2f})", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Overlay Mask
        mask_overlay = np.zeros_like(frame)
        # Resize mask to frame dimensions if needed
        if mask.shape[:2] != frame.shape[:2]:
            resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        else:
            resized_mask = mask
        mask_overlay[resized_mask > 0.5] = color
        
        # Blend overlay (alpha=0.4)
        cv2.addWeighted(mask_overlay, 0.4, frame, 1, 0, frame)

    # Global Alert (if any person is inside)
    roi_color = (0, 255, 0)
    if alert_triggered:
        roi_color = (0, 0, 255)
        # Display big alert on screen
        cv2.putText(frame, "INTRUSION DETECTED!", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Draw ROI
    roi_manager.draw_roi(frame, color=roi_color)
    
    return frame, alert_triggered


def main(video_source=0, preset='balanced'):
    """
    Main loop for the CV Alert System.
    
    Args:
        video_source: Path to video file or camera index (default 0 for webcam)
        preset: Detection preset - 'fast', 'balanced', or 'accurate'
    """
    # 1. Initialize Detector with CPU-optimized preset
    print(f"Initializing detector with '{preset}' preset...")
    detector = HumanDetector(preset=preset)
    
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
    
    # 3. Define ROI (center 60% area - adjust for your use case)
    polygon_points = [
        (int(width * 0.49), int(height * 0.49)),
        (int(width * 0.51), int(height * 0.49)),
        (int(width * 0.51), int(height * 0.51)),
        (int(width * 0.49), int(height * 0.51))
    ]
    roi_manager = ROIManager(polygon_points, width, height)

    print(f"Video: {width}x{height}")
    print(f"ROI: Center 60% area")
    print("Starting video loop... Press 'q' to quit.")

    # Performance tracking
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, alert_triggered = process_frame(frame, detector, roi_manager)
        frame_count += 1
        
        # Calculate and display FPS every 30 frames
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"FPS: {fps:.1f}")
        
        if alert_triggered:
            print("!!! ALERT: Human inside ROI !!!")

        # Show Frame
        cv2.imshow("CV Human Alert System", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Final stats
    elapsed = time.time() - start_time
    print(f"\nSession: {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='CV Human Alert System')
    parser.add_argument('source', nargs='?', default='0', 
                       help='Video source (camera index or file path)')
    parser.add_argument('--preset', '-p', default='balanced',
                       choices=['fast', 'balanced'],
                       help='Detection preset (default: balanced)')
    
    args = parser.parse_args()
    
    # Parse source
    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass  # It's a file path
    
    main(source, preset=args.preset)
