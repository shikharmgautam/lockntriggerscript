# ============================================================
# CV HUMAN ALERT SYSTEM - GOOGLE COLAB VERSION
# ============================================================
# Copy this entire file into a Google Colab cell and run it!
# Make sure to enable GPU: Runtime > Change runtime type > GPU
# ============================================================

# CELL 1: Install dependencies (run this first)
# !pip install ultralytics opencv-python-headless

# ============================================================
# IMPORTS
# ============================================================
import cv2
import numpy as np
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
from google.colab import files
import IPython.display as display
from PIL import Image
import io

# ============================================================
# HUMAN DETECTOR (Maximum Accuracy Configuration)
# ============================================================
class HumanDetector:
    """
    Human detector optimized for MAXIMUM ACCURACY.
    Uses YOLOv8x-seg (largest model) with high resolution.
    """
    
    def __init__(self, 
                 model_path='yolov8x-seg.pt',  # Largest model
                 confidence=0.10,               # Very low threshold
                 imgsz=1280,                    # Maximum resolution
                 use_preprocessing=True):
        
        print(f"Loading model: {model_path} (this may take a moment...)")
        self.model = YOLO(model_path)
        self.target_class_id = 0  # 'person' in COCO
        self.confidence = confidence
        self.imgsz = imgsz
        self.use_preprocessing = use_preprocessing
        
        # CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Confidence threshold: {confidence}")
        print(f"✓ Input resolution: {imgsz}")
        print(f"✓ Preprocessing (CLAHE): {use_preprocessing}")

    def preprocess(self, frame):
        if not self.use_preprocessing:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def detect(self, frame):
        processed_frame = self.preprocess(frame)
        results = self.model(
            processed_frame, 
            verbose=False, 
            classes=[self.target_class_id], 
            retina_masks=True,
            conf=self.confidence,
            imgsz=self.imgsz
        )
        
        detections = []
        for result in results:
            if result.masks is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            masks = result.masks.data.cpu().numpy()
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                if i < len(masks):
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "mask": masks[i],
                        "conf": float(confs[i])
                    })
        return detections


# ============================================================
# ROI MANAGER
# ============================================================
class ROIManager:
    """Manages the Region of Interest polygon."""
    
    def __init__(self, polygon_points, frame_width, frame_height):
        self.polygon = np.array(polygon_points, dtype=np.int32)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.roi_mask = self._create_roi_mask()
    
    def _create_roi_mask(self):
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        cv2.fillPoly(mask, [self.polygon], 255)
        return mask
    
    def is_mask_intersecting(self, human_mask):
        # Resize human mask to frame size if needed
        if human_mask.shape[:2] != (self.frame_height, self.frame_width):
            human_mask = cv2.resize(human_mask, (self.frame_width, self.frame_height))
        
        # Binarize the mask
        binary_mask = (human_mask > 0.5).astype(np.uint8) * 255
        
        # Check intersection
        intersection = cv2.bitwise_and(binary_mask, self.roi_mask)
        return np.any(intersection > 0)
    
    def draw_roi(self, frame, color=(0, 255, 0), thickness=2):
        cv2.polylines(frame, [self.polygon], True, color, thickness)
        # Semi-transparent fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.polygon], color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)


# ============================================================
# FRAME PROCESSOR
# ============================================================
def process_frame(frame, detector, roi_manager):
    """Process a single frame: detect humans, check ROI, draw annotations."""
    detections = detector.detect(frame)
    alert_triggered = False

    for det in detections:
        x1, y1, x2, y2 = det['box']
        mask = det['mask']
        conf = det['conf']
        
        is_inside = roi_manager.is_mask_intersecting(mask)
        color = (0, 255, 0)  # Green = safe
        
        if is_inside:
            color = (0, 0, 255)  # Red = alert
            alert_triggered = True
            cv2.putText(frame, f"ALERT! ({conf:.2f})", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Overlay mask
        mask_overlay = np.zeros_like(frame)
        resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        mask_overlay[resized_mask > 0.5] = color
        cv2.addWeighted(mask_overlay, 0.3, frame, 1, 0, frame)

    # Draw ROI
    roi_color = (0, 0, 255) if alert_triggered else (0, 255, 0)
    roi_manager.draw_roi(frame, color=roi_color)
    
    if alert_triggered:
        cv2.putText(frame, "⚠ INTRUSION DETECTED!", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    return frame, alert_triggered


# ============================================================
# MAIN FUNCTION - PROCESS VIDEO
# ============================================================
def process_video(video_path, output_path='output.mp4', show_every_n_frames=30):
    """
    Process a video file and detect humans in ROI.
    
    Args:
        video_path: Path to input video
        output_path: Path for output video with annotations
        show_every_n_frames: Display frame in notebook every N frames
    """
    print("\n" + "="*60)
    print("CV HUMAN ALERT SYSTEM - MAXIMUM ACCURACY MODE")
    print("="*60)
    
    # Initialize detector
    detector = HumanDetector(
        model_path='yolov8x-seg.pt',
        confidence=0.10,
        imgsz=1280,
        use_preprocessing=True
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Define ROI (center 60% of frame - adjust as needed)
    polygon_points = [
        (int(width * 0.2), int(height * 0.2)),
        (int(width * 0.8), int(height * 0.2)),
        (int(width * 0.8), int(height * 0.8)),
        (int(width * 0.2), int(height * 0.8))
    ]
    roi_manager = ROIManager(polygon_points, width, height)
    print(f"📍 ROI defined: {polygon_points}")
    
    # Video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    alert_count = 0
    
    print(f"\n🔄 Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        processed_frame, alert = process_frame(frame, detector, roi_manager)
        
        if alert:
            alert_count += 1
        
        # Write to output video
        out.write(processed_frame)
        
        # Show progress and frame periodically
        if frame_count % show_every_n_frames == 0:
            print(f"  Frame {frame_count}/{total_frames} ({100*frame_count/total_frames:.1f}%)")
            # Convert BGR to RGB for display
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            display.display(Image.fromarray(rgb_frame))
            display.clear_output(wait=True)
    
    cap.release()
    out.release()
    
    print(f"\n" + "="*60)
    print(f"✅ PROCESSING COMPLETE!")
    print(f"="*60)
    print(f"📊 Total frames processed: {frame_count}")
    print(f"⚠️  Frames with alerts: {alert_count}")
    print(f"💾 Output saved to: {output_path}")
    
    # Show final frame
    print(f"\n📸 Final processed frame:")
    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    display.display(Image.fromarray(rgb_frame))
    
    return output_path


# ============================================================
# RUN THIS TO START
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("📤 UPLOAD YOUR VIDEO FILE")
    print("="*60)
    
    # Upload video
    uploaded = files.upload()
    video_filename = list(uploaded.keys())[0]
    print(f"\n✓ Uploaded: {video_filename}")
    
    # Process video
    output_file = process_video(video_filename)
    
    # Download processed video
    print(f"\n📥 Downloading processed video...")
    files.download(output_file)
