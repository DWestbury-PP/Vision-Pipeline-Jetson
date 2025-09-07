"""
OpenCV import patch to handle DNN module compatibility issues.

This module provides a safe way to import OpenCV that handles the
cv2.dnn.DictValue AttributeError that occurs in some NVIDIA container environments.
"""

import sys
import importlib.util
from typing import Any, Optional

# Global flag to track if OpenCV is available
CV2_AVAILABLE = False
cv2: Optional[Any] = None


def patch_opencv_import():
    """
    Safely import OpenCV with DNN compatibility patches.
    
    Returns:
        tuple: (cv2_module, is_available)
    """
    global CV2_AVAILABLE, cv2
    
    if cv2 is not None:
        return cv2, CV2_AVAILABLE
    
    try:
        # First, try to import cv2 normally
        import cv2 as opencv_module
        
        # Test if the DNN module works
        try:
            _ = opencv_module.dnn.DictValue
            # If we get here, DNN module is working properly
            CV2_AVAILABLE = True
            cv2 = opencv_module
            print("✅ OpenCV imported successfully with working DNN module")
            
        except AttributeError:
            # DNN module has compatibility issues, patch it
            print("⚠️  OpenCV DNN module has compatibility issues, applying patch...")
            
            # Create a mock DictValue if it doesn't exist
            if not hasattr(opencv_module.dnn, 'DictValue'):
                # Create a simple mock class
                class MockDictValue:
                    def __init__(self, *args, **kwargs):
                        pass
                    
                    def __call__(self, *args, **kwargs):
                        return self
                
                # Monkey patch the missing attribute
                opencv_module.dnn.DictValue = MockDictValue
                print("🔧 Applied DictValue patch to OpenCV DNN module")
            
            CV2_AVAILABLE = True
            cv2 = opencv_module
            print("✅ OpenCV imported successfully with DNN compatibility patch")
            
    except ImportError as e:
        print(f"❌ Failed to import OpenCV: {e}")
        
        # Create a minimal mock cv2 module for basic compatibility
        class MockCV2:
            """Minimal mock OpenCV module for basic functionality."""
            
            # Common constants
            COLOR_BGR2RGB = 4
            COLOR_RGB2BGR = 3
            CAP_GSTREAMER = 1800
            CAP_V4L2 = 200
            CAP_PROP_FRAME_WIDTH = 3
            CAP_PROP_FRAME_HEIGHT = 4
            CAP_PROP_FPS = 5
            CAP_PROP_BUFFERSIZE = 38
            
            @staticmethod
            def VideoCapture(*args, **kwargs):
                """Mock VideoCapture that returns None."""
                return None
            
            @staticmethod
            def cvtColor(src, code):
                """Mock color conversion that returns input unchanged."""
                return src
            
            @staticmethod
            def resize(src, dsize, *args, **kwargs):
                """Mock resize that returns input unchanged."""
                return src
            
            class dnn:
                """Mock DNN module."""
                
                class DictValue:
                    """Mock DictValue class."""
                    def __init__(self, *args, **kwargs):
                        pass
            
            @staticmethod
            def __version__():
                return "mock-4.5.0"
        
        CV2_AVAILABLE = False
        cv2 = MockCV2()
        print("🔧 Using mock OpenCV module for basic compatibility")
    
    return cv2, CV2_AVAILABLE


def get_opencv():
    """
    Get the OpenCV module (real or mock) and availability status.
    
    Returns:
        tuple: (cv2_module, is_available)
    """
    return patch_opencv_import()


# Automatically patch on import
cv2, CV2_AVAILABLE = patch_opencv_import()
