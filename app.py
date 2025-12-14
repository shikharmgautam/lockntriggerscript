"""
Flask Backend for CV Human Alert System
Wraps the existing Python detection logic with a web interface.
"""
from flask import Flask, Response, render_template, jsonify
import cv2
import threading
import time
from detector import HumanDetector
from roi_manager import ROIManager
import numpy as np

app = Flask(__name__)

# Global state
alert_status = {"triggered": False, "timestamp": None}
alert_lock = threading.Lock()
camera = None
detector = None
roi_manager = None
frame_width = 640
frame_height = 480

def init_camera():
    """Initialize camera and detector."""
    global camera, detector, roi_manager, frame_width, frame_height
    
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Read first frame to get dimensions
        ret, frame = camera.read()
        if ret:
            frame_height, frame_width = frame.shape[:2]
        
        # Initialize detector
        detector = HumanDetector()
        
        # Initialize ROI (center area - same as main.py)
        polygon_points = [
            (int(frame_width * 0.2), int(frame_height * 0.2)),
            (int(frame_width * 0.8), int(frame_height * 0.2)),
            (int(frame_width * 0.8), int(frame_height * 0.8)),
            (int(frame_width * 0.2), int(frame_height * 0.8))
        ]
        roi_manager = ROIManager(polygon_points, frame_width, frame_height)

def process_frame(frame):
    """Process a frame: detect humans, check ROI, draw annotations."""
    global alert_status
    
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
            cv2.putText(frame, "ALERT!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Overlay Mask
        mask_overlay = np.zeros_like(frame)
        mask_overlay[mask > 0.5] = color
        cv2.addWeighted(mask_overlay, 0.4, frame, 1, 0, frame)

    # Global Alert
    roi_color = (0, 255, 0)
    if alert_triggered:
        roi_color = (0, 0, 255)
        cv2.putText(frame, "INTRUSION DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Draw ROI
    roi_manager.draw_roi(frame, color=roi_color)
    
    # Update global alert status
    with alert_lock:
        alert_status["triggered"] = alert_triggered
        alert_status["timestamp"] = time.time()
    
    return frame

def generate_frames():
    """Generator for video streaming."""
    init_camera()
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        # Process frame with detection
        processed_frame = process_frame(frame)
        
        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/events')
def events():
    """Server-Sent Events for real-time alert status."""
    def generate():
        last_status = None
        while True:
            with alert_lock:
                current_status = alert_status["triggered"]
            
            if current_status != last_status:
                last_status = current_status
                yield f"data: {{'alert': {str(current_status).lower()}}}\n\n"
            
            time.sleep(0.1)
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/status')
def status():
    """Get current alert status."""
    with alert_lock:
        return jsonify(alert_status)

if __name__ == '__main__':
    print("Starting CV Alert System Web Server...")
    print("Open http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
