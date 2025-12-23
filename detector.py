from ultralytics import YOLO
import cv2
import numpy as np

# Presets optimized for CPU-only systems (AMD Ryzen 3 5300U, no GPU)
# Note: yolov8m-seg and larger models crash on this system
PRESETS = {
    'fast': {
        'model': 'yolov8n-seg.pt',
        'imgsz': 480,
        'confidence': 0.20,
        'description': 'Fastest - lower accuracy'
    },
    'balanced': {
        'model': 'yolov8s-seg.pt',
        'imgsz': 640,
        'confidence': 0.15,
        'description': 'Best working model for this system'
    }
}


class HumanDetector:
    """
    Human detector optimized for CPU-only systems.
    Full polygon mask matching (instance segmentation) is preserved.
    
    Presets:
    - 'fast': yolov8n-seg @ 480px
    - 'balanced': yolov8s-seg @ 640px [RECOMMENDED - works on your system]
    """
    
    def __init__(self, 
                 preset='balanced',
                 model_path=None,
                 confidence=None,
                 imgsz=None,
                 use_preprocessing=True):
        """
        Initialize the human detector.
        
        Args:
            preset: One of 'fast', 'balanced', or 'accurate'
            model_path: Override model (optional)
            confidence: Override confidence threshold (optional)
            imgsz: Override input resolution (optional)
            use_preprocessing: Enable CLAHE contrast enhancement
        """
        # Load preset defaults
        if preset not in PRESETS:
            print(f"[Warning] Unknown preset '{preset}', using 'balanced'")
            preset = 'balanced'
        
        config = PRESETS[preset]
        
        # Allow overrides
        self.model_path = model_path or config['model']
        self.confidence = confidence if confidence is not None else config['confidence']
        self.imgsz = imgsz or config['imgsz']
        self.use_preprocessing = use_preprocessing
        
        # Load model with CPU optimization
        print(f"[Detector] Loading {self.model_path} ({preset} preset)...")
        self.model = YOLO(self.model_path)
        self.target_class_id = 0  # 'person' in COCO dataset
        
        # Initialize CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        print(f"[Detector] ✓ Model: {self.model_path}")
        print(f"[Detector] ✓ Preset: {preset} - {config['description']}")
        print(f"[Detector] ✓ Confidence: {self.confidence}")
        print(f"[Detector] ✓ Resolution: {self.imgsz}px")
        print(f"[Detector] ✓ Preprocessing (CLAHE): {use_preprocessing}")

    def preprocess(self, frame):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Improves detection of humans in low-contrast or poorly lit scenes.
        """
        if not self.use_preprocessing:
            return frame
            
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel (luminance)
        l = self.clahe.apply(l)
        
        # Merge and convert back to BGR
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def detect(self, frame):
        """
        Run inference and return boxes and masks for detected humans.
        
        Returns: List of dicts with keys:
            - box: [x1, y1, x2, y2]
            - mask: float32 mask (0-1) at original frame resolution
            - conf: confidence score
        """
        # Apply preprocessing if enabled
        processed_frame = self.preprocess(frame)
        
        # Run YOLO inference with CPU-optimized parameters
        results = self.model(
            processed_frame, 
            verbose=False, 
            classes=[self.target_class_id], 
            retina_masks=True,
            conf=self.confidence,
            imgsz=self.imgsz,
            device='cpu',              # Explicit CPU
            half=False,                # Full precision (FP16 needs GPU)
            agnostic_nms=True          # Better NMS for overlapping detections
        )
        
        detections = []
        
        for result in results:
            if result.masks is None:
                continue
                
            # Extract boxes and masks
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            # Masks with retina_masks=True are scaled to original image size
            masks = result.masks.data.cpu().numpy()
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                conf = confs[i]
                
                # Check for segmentation mask
                if i < len(masks):
                    mask = masks[i]
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "mask": mask,  # float32 mask 0-1
                        "conf": float(conf)
                    })
        
        return detections
