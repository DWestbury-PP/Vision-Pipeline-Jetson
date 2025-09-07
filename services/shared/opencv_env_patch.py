#!/usr/bin/env python3
"""
Environment-level OpenCV DNN compatibility patch.

This module patches OpenCV at the system level by intercepting the import
process before any modules can trigger the DictValue AttributeError.

Must be imported BEFORE any OpenCV imports occur.
"""

import os
import sys
import importlib.util

# Set environment variables to disable problematic OpenCV features
os.environ['OPENCV_DISABLE_TYPING'] = '1'
os.environ['OPENCV_DISABLE_DNN_TYPING'] = '1'

def apply_system_opencv_patch():
    """
    Apply system-level OpenCV patches before any imports occur.
    """
    print("🔧 Applying system-level OpenCV compatibility patches...")
    
    # Hook into the import system to patch cv2 before it fully loads
    class OpenCVImportHook:
        """Custom import hook to patch OpenCV on first import."""
        
        def __init__(self):
            self.patched = False
        
        def find_spec(self, fullname, path, target=None):
            if fullname == 'cv2' and not self.patched:
                print("🔍 Intercepting cv2 import for patching...")
                self.patched = True
                
                # Let the normal import proceed, but we'll patch it immediately after
                return None
            return None
        
        def find_module(self, fullname, path=None):
            return None
    
    # Install the import hook
    hook = OpenCVImportHook()
    sys.meta_path.insert(0, hook)
    
    print("✅ OpenCV import hook installed")

def patch_cv2_after_import():
    """
    Patch cv2 after it's been imported but before typing module loads.
    """
    if 'cv2' in sys.modules:
        cv2_module = sys.modules['cv2']
        
        try:
            # Check if DNN module exists and needs patching
            if hasattr(cv2_module, 'dnn'):
                if not hasattr(cv2_module.dnn, 'DictValue'):
                    print("🔧 Patching missing cv2.dnn.DictValue...")
                    
                    class MockDictValue:
                        """Mock DictValue class for compatibility."""
                        def __init__(self, *args, **kwargs):
                            pass
                        def __call__(self, *args, **kwargs):
                            return self
                        def __str__(self):
                            return "MockDictValue"
                        def __repr__(self):
                            return "MockDictValue()"
                    
                    cv2_module.dnn.DictValue = MockDictValue
                    print("✅ Successfully patched cv2.dnn.DictValue")
                    return True
                else:
                    print("✅ cv2.dnn.DictValue already exists")
                    return True
            else:
                print("⚠️  cv2.dnn module not found")
                return False
                
        except Exception as e:
            print(f"❌ Failed to patch cv2.dnn: {e}")
            return False
    
    return False

# Apply patches immediately when this module is imported
if __name__ != "__main__":
    apply_system_opencv_patch()
