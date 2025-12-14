import unittest
import sys
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

# MOCK ultralytics BEFORE importing detector/main
# This allows testing the logic even if dependencies aren't installed
mock_ultralytics = MagicMock()
sys.modules["ultralytics"] = mock_ultralytics

# Now we can import our modules
# We also need to patch HumanDetector class in detector.py to avoid it trying to instantiate YOLO
with patch("detector.HumanDetector") as MockDetector:
    from main import process_frame
    from roi_manager import ROIManager

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # ROI: Square 100x100 at (0,0)
        self.polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
        # Add dimensions 200x200
        self.roi_manager = ROIManager(self.polygon, 200, 200)
        
        # Mock Detector Instance
        self.mock_detector = MagicMock()
        
        # Blank Frame 200x200
        self.frame = np.zeros((200, 200, 3), dtype=np.uint8)

    def test_alert_triggered_when_human_inside(self):
        # Setup: Mock detect to return one person INSIDE the ROI
        # Box: [20, 20, 40, 40] is fully inside 0-100 zone
        self.mock_detector.detect.return_value = [[20, 20, 40, 40, 0.9]]
        
        # Execute
        processed_frame, alert = process_frame(self.frame, self.mock_detector, self.roi_manager)
        
        # Verify
        self.assertTrue(alert, "Alert SHOULD be triggered when person is inside ROI")
        # Check if text "INTRUSION DETECTED" is possibly in the frame (hard to check pixels, but logic bool is sufficient)

    def test_no_alert_when_human_outside(self):
        # Setup: Mock detect to return one person OUTSIDE the ROI
        # Box: x1=120, y1=120, x2=140, y2=140. Foot is at (130, 140) which is OUTSIDE
        self.mock_detector.detect.return_value = [[120, 120, 140, 140, 0.9]]
        
        # Execute
        processed_frame, alert = process_frame(self.frame, self.mock_detector, self.roi_manager)
        
        # Verify
        self.assertFalse(alert, "Alert should NOT be triggered when person is outside ROI")

    def test_no_alert_when_no_human(self):
        # Setup: No detections
        self.mock_detector.detect.return_value = []
        
        # Execute
        processed_frame, alert = process_frame(self.frame, self.mock_detector, self.roi_manager)
        
        # Verify
        self.assertFalse(alert, "Alert should NOT be triggered when no humans are detected")

if __name__ == '__main__':
    unittest.main()
