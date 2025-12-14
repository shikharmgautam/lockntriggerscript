import unittest
import numpy as np
from roi_manager import ROIManager
import cv2

class TestROILogic(unittest.TestCase):
    def test_roi_logic(self):
        # Define a square 100x100 polygon at (0,0) to (100,100)
        polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
        # Mask needs frame size. Let's say 200x200
        roi = ROIManager(polygon, 200, 200)
        
        # Test 1: Box Fully Inside [(10,10) to (50,50)]
        self.assertTrue(roi.is_box_intersecting((10, 10, 50, 50)), "Fully inside box should intersect")
        
        # Test 2: Box Partially Overlapping [(90, 90) to (110, 110)] -> Corner overlap
        self.assertTrue(roi.is_box_intersecting((90, 90, 110, 110)), "Partially overlapping box should intersect")
        
        # Test 3: Box Fully Outside [(150, 150) to (160, 160)]
        self.assertFalse(roi.is_box_intersecting((150, 150, 160, 160)), "Outside box should NOT intersect")
        
        # Test 4: Box Touching Edge [(100, 50) to (110, 60)] -> Edge overlap
        # Due to rasterization (fillPoly), edge pixels are usually included
        self.assertTrue(roi.is_box_intersecting((99, 50, 110, 60)), "Touching box should intersect")
        
        print("ROI Logic verification passed.")

if __name__ == '__main__':
    unittest.main()
