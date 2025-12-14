import cv2
import numpy as np

class ROIManager:
    def __init__(self, polygon_coords, frame_width, frame_height):
        """
        Initialize with polygon coords and frame dimensions to create a mask.
        """
        self.polygon_coords = np.array(polygon_coords, np.int32)
        self.polygon_coords = self.polygon_coords.reshape((-1, 1, 2))
        
        # Create a binary mask for the ROI
        self.mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        cv2.fillPoly(self.mask, [self.polygon_coords], 1) # Fill ROI with 1s

    def is_box_intersecting(self, box):
        """
        Check if a bounding box (x1, y1, x2, y2) intersects with the ROI.
        """
        x1, y1, x2, y2 = box
        
        # Ensure coordinates are within frame bounds
        h, w = self.mask.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x1 >= x2 or y1 >= y2:
            return False

        # Slice the mask using the box coordinates
        roi_section = self.mask[y1:y2, x1:x2]
        
        # If any pixel in this section is 1, there is an intersection
        return np.any(roi_section)

    def is_mask_intersecting(self, person_mask):
        """
        Check if the person's segmentation mask intersects with the ROI mask.
        person_mask: Binary mask of the person (same shape as frame ideally, or needs resizing).
                     YOLOv8 'retina_masks=True' returns masks in original img shape.
        """
        # Ensure mask is binary uint8
        if person_mask.dtype != np.uint8:
            person_mask = (person_mask > 0.5).astype(np.uint8)
        
        # Ensure shapes match (resize if necessary - safeguard)
        if person_mask.shape != self.mask.shape:
             person_mask = cv2.resize(person_mask, (self.mask.shape[1], self.mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Bitwise AND to find overlap
        overlap = cv2.bitwise_and(self.mask, person_mask)
        
        # Check if any pixel overlaps
        return cv2.countNonZero(overlap) > 0

    def draw_roi(self, frame, color=(0, 255, 0), thickness=2):
        """
        Draw the ROI polygon on the frame.
        """
        cv2.polylines(frame, [self.polygon_coords], isClosed=True, color=color, thickness=thickness)
