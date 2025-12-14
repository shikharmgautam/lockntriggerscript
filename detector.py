from ultralytics import YOLO
import cv2
import numpy as np

class HumanDetector:
    """
    Human detector optimized for long-distance detection.
    
    Improvements over default:
    - Uses larger model (yolov8m-seg) for better accuracy
    - Lower confidence threshold to catch distant/small humans
    - Higher input resolution for better detail preservation
    - Optional CLAHE preprocessing for improved contrast
    """
    
    def __init__(self, 
                 model_path='yolov8x-seg.pt',  # Largest model for maximum accuracy
                 confidence=0.10,               # Very low threshold for distant objects
                 imgsz=1280,                    # Maximum resolution
                 use_preprocessing=True):       # Enable CLAHE preprocessing
        """
        Initialize the human detector.
        
        Args:
            model_path: YOLO model to use. Options:
                - 'yolov8n-seg.pt' (fastest, least accurate)
                - 'yolov8s-seg.pt' (fast, moderate accuracy)
                - 'yolov8m-seg.pt' (balanced - recommended)
                - 'yolov8l-seg.pt' (slower, high accuracy)
                - 'yolov8x-seg.pt' (slowest, highest accuracy)
            confidence: Confidence threshold (0.0-1.0). Lower = more detections.
            imgsz: Input image size. Higher = better for small objects but slower.
            use_preprocessing: Whether to apply CLAHE contrast enhancement.
        """
        self.model = YOLO(model_path)
        self.target_class_id = 0  # 'person' in COCO dataset
        self.confidence = confidence
        self.imgsz = imgsz
        self.use_preprocessing = use_preprocessing
        
        # Initialize CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        print(f"[Detector] Model: {model_path}")
        print(f"[Detector] Confidence threshold: {confidence}")
        print(f"[Detector] Input resolution: {imgsz}")
        print(f"[Detector] Preprocessing (CLAHE): {use_preprocessing}")

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
        
        # Run YOLO inference with optimized parameters
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
                
            # Extract boxes and masks
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            # Masks are returned as an object. We want the data.
            # retina_masks=True ensures masks are scaled to original image size
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
