#!/usr/bin/env python3
"""
Pre-execution OpenCV DNN compatibility patch.

This script must be run BEFORE any Python process that imports OpenCV.
It patches the OpenCV installation at the file system level to prevent
the DictValue AttributeError from occurring.
"""

import os
import sys
import tempfile
import shutil

def patch_opencv_typing():
    """
    Patch the OpenCV typing module to prevent DictValue errors.
    """
    try:
        # Find OpenCV installation
        import cv2
        opencv_path = os.path.dirname(cv2.__file__)
        typing_file = os.path.join(opencv_path, 'typing', '__init__.py')
        
        if os.path.exists(typing_file):
            print(f"🔍 Found OpenCV typing file: {typing_file}")
            
            # Read the original file
            with open(typing_file, 'r') as f:
                content = f.read()
            
            # Check if it contains the problematic line
            if 'LayerId = cv2.dnn.DictValue' in content:
                print("🔧 Patching problematic DictValue line...")
                
                # Create a backup
                backup_file = typing_file + '.backup'
                if not os.path.exists(backup_file):
                    shutil.copy2(typing_file, backup_file)
                    print(f"📋 Created backup: {backup_file}")
                
                # Replace the problematic line with a try-catch
                patched_content = content.replace(
                    'LayerId = cv2.dnn.DictValue',
                    '''try:
    LayerId = cv2.dnn.DictValue
except AttributeError:
    # Compatibility patch for missing DictValue
    class MockDictValue:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return self
    LayerId = MockDictValue'''
                )
                
                # Write the patched content
                with open(typing_file, 'w') as f:
                    f.write(patched_content)
                
                print("✅ Successfully patched OpenCV typing module")
                return True
            else:
                print("ℹ️  OpenCV typing module doesn't contain problematic line")
                return True
        else:
            print("⚠️  OpenCV typing file not found")
            return False
            
    except Exception as e:
        print(f"❌ Failed to patch OpenCV typing: {e}")
        return False

def verify_patch():
    """
    Verify that the patch works by importing OpenCV.
    """
    try:
        print("🧪 Testing OpenCV import after patch...")
        import cv2
        
        # Test DNN module
        if hasattr(cv2, 'dnn'):
            try:
                _ = cv2.dnn.DictValue
                print("✅ cv2.dnn.DictValue accessible")
            except AttributeError:
                print("⚠️  cv2.dnn.DictValue still missing, but import succeeded")
        
        print(f"📸 OpenCV version: {cv2.__version__}")
        print("✅ OpenCV import test successful")
        return True
        
    except Exception as e:
        print(f"❌ OpenCV import test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting OpenCV DNN compatibility patch...")
    
    if patch_opencv_typing():
        if verify_patch():
            print("🎉 OpenCV patch completed successfully!")
            sys.exit(0)
        else:
            print("❌ Patch verification failed")
            sys.exit(1)
    else:
        print("❌ Patch application failed")
        sys.exit(1)
