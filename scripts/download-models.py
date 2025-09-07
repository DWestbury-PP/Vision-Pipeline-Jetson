#!/usr/bin/env python3

"""
Model Downloader for Vision Pipeline
This script downloads required models using Python libraries
"""

import os
import sys
from pathlib import Path

def setup_directories():
    """Create model directories"""
    models_dir = Path("models")
    yolo_dir = models_dir / "yolo"
    moondream_dir = models_dir / "moondream"
    
    yolo_dir.mkdir(parents=True, exist_ok=True)
    moondream_dir.mkdir(parents=True, exist_ok=True)
    
    print("📁 Created model directories")
    return yolo_dir, moondream_dir

def download_yolo_model(yolo_dir):
    """Download YOLO model using ultralytics"""
    yolo_model_path = yolo_dir / "yolo11n.pt"
    
    if yolo_model_path.exists():
        print("✅ YOLO model already exists")
        return True
    
    print("⬇️  Downloading YOLO11n model using ultralytics...")
    
    try:
        # Try to import and use ultralytics
        from ultralytics import YOLO
        
        # This will download the model to the current directory
        model = YOLO('yolo11n.pt')
        
        # Move it to the correct location
        if Path('yolo11n.pt').exists():
            Path('yolo11n.pt').rename(yolo_model_path)
            print("✅ YOLO model downloaded successfully")
            return True
        else:
            print("⚠️  YOLO model download failed")
            return False
            
    except ImportError:
        print("❌ ultralytics not installed. Install with: pip install ultralytics")
        print("💡 Alternative: Download manually from:")
        print("   https://github.com/ultralytics/assets/releases/")
        return False
    except Exception as e:
        print(f"❌ Error downloading YOLO model: {e}")
        return False

def download_moondream_model(moondream_dir):
    """Download Moondream model using transformers"""
    moondream_model_dir = moondream_dir / "moondream2"
    
    if moondream_model_dir.exists() and (moondream_model_dir / "model.safetensors").exists():
        print("✅ Moondream model already exists")
        return True
    
    print("⬇️  Downloading Moondream2 model using transformers...")
    
    try:
        # Try to use transformers to download the model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("🔄 Downloading Moondream2 model files...")
        
        # Download model and tokenizer (this will cache them locally)
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            cache_dir=str(moondream_dir),
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            "vikhyatk/moondream2",
            cache_dir=str(moondream_dir)
        )
        
        print("✅ Moondream model downloaded successfully")
        return True
        
    except ImportError:
        print("❌ transformers not installed. Install with: pip install transformers")
        print("💡 Alternative: Use git-lfs to clone:")
        print("   git clone https://huggingface.co/vikhyatk/moondream2 models/moondream/moondream2")
        return False
    except Exception as e:
        print(f"❌ Error downloading Moondream model: {e}")
        print("💡 Alternative: Use git-lfs to clone:")
        print("   git clone https://huggingface.co/vikhyatk/moondream2 models/moondream/moondream2")
        return False

def main():
    print("🚀 Python Model Downloader for Vision Pipeline")
    print("=" * 50)
    
    # Setup directories
    yolo_dir, moondream_dir = setup_directories()
    
    # Download models
    yolo_success = download_yolo_model(yolo_dir)
    moondream_success = download_moondream_model(moondream_dir)
    
    # Report status
    print("\n📊 Model Status:")
    yolo_status = "✅ Available" if (yolo_dir / "yolo11n.pt").exists() else "❌ Missing"
    moondream_status = "✅ Available" if (moondream_dir / "moondream2").exists() else "❌ Missing"
    
    print(f"   YOLO11n: {yolo_status}")
    print(f"   Moondream2: {moondream_status}")
    
    print("\n🎯 Download complete! You can now run:")
    print("   docker compose build --no-cache")
    print("   docker compose up")
    
    if not (yolo_success and moondream_success):
        print("\n⚠️  Some models failed to download. Services will run in mock mode.")
        sys.exit(1)

if __name__ == "__main__":
    main()
