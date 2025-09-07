"""
OpenCV import patch to handle DNN module compatibility issues.

This module provides a safe way to import OpenCV that handles the
cv2.dnn.DictValue AttributeError that occurs in some NVIDIA container environments.

The patch works by intercepting the module loading process and monkey-patching
the missing DictValue attribute before the typing module tries to access it.
"""

import sys
import importlib
import importlib.util
from typing import Any, Optional

# Global flag to track if OpenCV is available
CV2_AVAILABLE = False
cv2: Optional[Any] = None


def pre_patch_opencv_typing():
    """
    Pre-patch OpenCV typing module to prevent DictValue AttributeError.
    
    This function monkey-patches the cv2.dnn module before the typing
    module tries to access the missing DictValue attribute.
    """
    try:
        # Check if cv2 is already in sys.modules
        if 'cv2' in sys.modules:
            opencv_module = sys.modules['cv2']
        else:
            # Import cv2 core module without triggering full initialization
            import cv2 as opencv_module
        
        # Check if dnn module exists and patch if needed
        if hasattr(opencv_module, 'dnn'):
            if not hasattr(opencv_module.dnn, 'DictValue'):
                print("🔧 Pre-patching OpenCV DNN module with missing DictValue...")
                
                # Create a mock DictValue class
                class MockDictValue:
                    """Mock DictValue for compatibility."""
                    def __init__(self, *args, **kwargs):
                        pass
                    
                    def __call__(self, *args, **kwargs):
                        return self
                    
                    def __str__(self):
                        return "MockDictValue"
                    
                    def __repr__(self):
                        return "MockDictValue()"
                
                # Monkey patch the missing attribute
                opencv_module.dnn.DictValue = MockDictValue
                print("✅ Successfully patched cv2.dnn.DictValue")
                
        return True
        
    except Exception as e:
        print(f"⚠️  Pre-patch failed: {e}")
        return False


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
        # Pre-patch OpenCV typing issues
        pre_patch_opencv_typing()
        
        # Now try to import cv2 normally
        import cv2 as opencv_module
        
        # Verify the patch worked
        try:
            _ = opencv_module.dnn.DictValue
            print("✅ OpenCV imported successfully with working DNN module")
        except AttributeError:
            print("⚠️  DictValue still missing, applying backup patch...")
            # Apply backup patch
            class MockDictValue:
                def __init__(self, *args, **kwargs):
                    pass
            opencv_module.dnn.DictValue = MockDictValue
            print("🔧 Applied backup DictValue patch")
        
        CV2_AVAILABLE = True
        cv2 = opencv_module
        print(f"📸 OpenCV version: {opencv_module.__version__}")
        
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
            
            @staticmethod
            def __version__():
                return "mock-4.5.0"
            
            class dnn:
                """Mock DNN module."""
                
                class DictValue:
                    """Mock DictValue class."""
                    def __init__(self, *args, **kwargs):
                        pass
        
        CV2_AVAILABLE = False
        cv2 = MockCV2()
        print("🔧 Using mock OpenCV module for basic compatibility")
    
    except Exception as e:
        print(f"❌ Unexpected error during OpenCV import: {e}")
        # Fallback to mock module
        class MockCV2:
            @staticmethod
            def __version__():
                return "mock-error-4.5.0"
            class dnn:
                class DictValue:
                    def __init__(self, *args, **kwargs):
                        pass
        
        CV2_AVAILABLE = False
        cv2 = MockCV2()
        print("🔧 Using error fallback mock OpenCV module")
    
    return cv2, CV2_AVAILABLE


def get_opencv():
    """
    Get the OpenCV module (real or mock) and availability status.
    
    Returns:
        tuple: (cv2_module, is_available)
    """
    return patch_opencv_import()


# Automatically patch on import
try:
    cv2, CV2_AVAILABLE = patch_opencv_import()
except Exception as e:
    print(f"❌ Critical error in opencv_patch initialization: {e}")
    # Emergency fallback
    class EmergencyMockCV2:
        @staticmethod
        def __version__():
            return "emergency-mock-4.5.0"
        class dnn:
            class DictValue:
                def __init__(self, *args, **kwargs):
                    pass
    
    CV2_AVAILABLE = False
    cv2 = EmergencyMockCV2()
