import unittest
import numpy as np
from roi_manager import ROIManager
from detector import HumanDetector
import cv2
import os

class TestCVSystem(unittest.TestCase):
    def test_roi_logic(self):
        # Define a square 100x100 polygon at (0,0) to (100,100)
        polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
        roi = ROIManager(polygon)
        
        # Test inside
        self.assertTrue(roi.is_point_inside((50, 50)), "Point (50, 50) should be inside")
        
        # Test outside
        self.assertFalse(roi.is_point_inside((150, 150)), "Point (150, 150) should be outside")
        
        # Test edge (might return True or False based on implementation details, usually True/0 is >=0)
        # cv2.pointPolygonTest returns 0 on edge. Logic is >= 0
        self.assertTrue(roi.is_point_inside((0, 0)), "Point (0, 0) should be on edge/inside")

    def test_model_loading(self):
        print("Testing model loading... (this might download weights)")
        try:
            detector = HumanDetector()
            self.assertIsNotNone(detector.model)
            print("Model loaded successfully.")
        except Exception as e:
            self.fail(f"Model failed to load: {e}")

    def test_detection_on_dummy_frame(self):
        # Create a black image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector = HumanDetector()
        
        # Should detect nothing on black image
        detections = detector.detect(frame)
        self.assertEqual(len(detections), 0)
        print("Detection on empty frame verified.")

if __name__ == '__main__':
    unittest.main()
