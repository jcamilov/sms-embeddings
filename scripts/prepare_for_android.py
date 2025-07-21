#!/usr/bin/env python3
"""
Main script to prepare everything for Android use.
This script will:
1. Convert the SentenceTransformers model to TensorFlow Lite
2. Prepare embeddings in JSON format for Android
3. Create a complete Android-ready package
"""

import os
import sys
import subprocess
import shutil

# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_NAME, CLASS_SMISHING, CLASS_BENIGN

def run_script(script_name, description):
    """
    Run a Python script and handle errors.
    """
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        print(f"Error: Script {script_name} not found!")
        return False
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def create_android_package():
    """
    Create a complete Android package with all necessary files.
    """
    print(f"\n{'='*60}")
    print("CREATING ANDROID PACKAGE")
    print(f"{'='*60}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    android_assets_dir = os.path.join(base_dir, "..", "android_assets")
    
    if not os.path.exists(android_assets_dir):
        print("Error: Android assets directory not found!")
        print("Please run the conversion scripts first.")
        return False
    
    # Create a README for Android developers
    readme_content = f"""# Android SMS Semantic Search Assets

This directory contains all the files needed for SMS semantic search in your Android app.

## Files Included:

### Model Files:
- `sms_embedding_model.tflite` - TensorFlow Lite model for generating embeddings
- Model: {MODEL_NAME}

### Embedding Files:
- `smishing_embeddings.json` - Pre-computed embeddings for smishing SMS
- `benign_embeddings.json` - Pre-computed embeddings for benign SMS
- `embeddings_metadata.json` - Metadata about the embeddings

## How to Use:

1. Copy all files to your Android app's `assets` folder
2. Use TensorFlow Lite to load the model
3. Load the JSON files to get pre-computed embeddings
4. Implement cosine similarity search

## Model Information:
- Model: {MODEL_NAME}
- Classes: {CLASS_SMISHING}, {CLASS_BENIGN}
- Embedding dimension: Check metadata file

## Android Implementation Steps:

1. Add TensorFlow Lite dependency to your `build.gradle`:
```gradle
implementation 'org.tensorflow:tensorflow-lite:2.9.0'
implementation 'org.tensorflow:tensorflow-lite-support:0.4.2'
```

2. Load the model:
```kotlin
val model = Interpreter(loadModelFile(context, "sms_embedding_model.tflite"))
```

3. Load embeddings from JSON files
4. Implement semantic search using cosine similarity

## File Sizes:
"""
    
    # Add file sizes to README
    for filename in os.listdir(android_assets_dir):
        file_path = os.path.join(android_assets_dir, filename)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024*1024)
            readme_content += f"- {filename}: {size_mb:.2f} MB\n"
    
    readme_path = os.path.join(android_assets_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"Created README: {readme_path}")
    
    # Create a simple test script
    test_script_content = f"""#!/usr/bin/env python3
\"\"\"
Test script to verify Android assets work correctly.
\"\"\"

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def test_android_assets():
    \"\"\"
    Test the prepared Android assets.
    \"\"\"
    print("Testing Android assets...")
    
    # Test embeddings loading
    for class_name in ['{CLASS_SMISHING}', '{CLASS_BENIGN}']:
        json_file = f'{{class_name}}_embeddings.json'
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            embeddings = np.array(data['embeddings'])
            texts = data['texts']
            
            print(f"✓ {{class_name}}: {{len(embeddings)}} embeddings loaded")
            print(f"  - Dimension: {{embeddings.shape[1]}}")
            print(f"  - Sample: '{{texts[0][:50]}}...'")
            
        except Exception as e:
            print(f"✗ {{class_name}}: Error - {{e}}")
    
    print("\\nTest completed!")

if __name__ == "__main__":
    test_android_assets()
"""
    
    test_script_path = os.path.join(android_assets_dir, "test_assets.py")
    with open(test_script_path, 'w', encoding='utf-8') as f:
        f.write(test_script_content)
    
    # Make it executable
    os.chmod(test_script_path, 0o755)
    
    print(f"Created test script: {test_script_path}")
    
    return True

def main():
    """
    Main function to orchestrate the entire Android preparation process.
    """
    print("="*60)
    print("ANDROID PREPARATION TOOL")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"Classes: {CLASS_SMISHING}, {CLASS_BENIGN}")
    print("="*60)
    
    # Step 1: Convert model to TensorFlow Lite
    if not run_script("convert_model_to_tflite.py", "Converting model to TensorFlow Lite"):
        print("Model conversion failed!")
        return False
    
    # Step 2: Prepare embeddings for Android
    if not run_script("prepare_embeddings_for_android.py", "Preparing embeddings for Android"):
        print("Embeddings preparation failed!")
        return False
    
    # Step 3: Create Android package
    if not create_android_package():
        print("Android package creation failed!")
        return False
    
    print("\n" + "="*60)
    print("ANDROID PREPARATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("All files are ready for your Android app!")
    print("\nNext steps:")
    print("1. Copy the contents of 'android_assets' folder to your Android app")
    print("2. Add TensorFlow Lite dependency to your Android project")
    print("3. Implement the semantic search functionality")
    print("4. Test with the provided test script")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 