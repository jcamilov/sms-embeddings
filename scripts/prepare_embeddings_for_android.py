#!/usr/bin/env python3
"""
Script to prepare embeddings for Android use.
Converts embeddings to JSON format for easy loading in Android.
"""

import os
import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Add the scripts directory to the path to import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_NAME, CLASS_SMISHING, CLASS_BENIGN

def prepare_embeddings_for_android():
    """
    Prepare embeddings in JSON format for Android use.
    """
    print("Preparing embeddings for Android...")
    
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_dir = os.path.join(base_dir, "..", "embeddings")
    android_assets_dir = os.path.join(base_dir, "..", "android_assets")
    
    # Create android_assets directory if it doesn't exist
    os.makedirs(android_assets_dir, exist_ok=True)
    
    # Load the model to verify embeddings
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    # Process each class
    classes = [CLASS_SMISHING, CLASS_BENIGN]
    
    for class_name in classes:
        print(f"\nProcessing {class_name} embeddings...")
        
        # Define file paths
        embeddings_path = os.path.join(embeddings_dir, f'{class_name}_embeddings.npy')
        texts_path = os.path.join(embeddings_dir, f'{class_name}_texts.npy')
        ids_path = os.path.join(embeddings_dir, f'{class_name}_ids.npy')
        
        # Check if files exist
        if not all(os.path.exists(path) for path in [embeddings_path, texts_path, ids_path]):
            print(f"Warning: {class_name} embedding files not found.")
            print("Run first: python scripts/generate_embeddings.py")
            continue
        
        # Load embeddings
        embeddings = np.load(embeddings_path)
        texts = np.load(texts_path, allow_pickle=True)
        ids = np.load(ids_path, allow_pickle=True)
        
        print(f"Loaded {len(embeddings)} {class_name} embeddings")
        
        # Convert to JSON format
        android_data = {
            "class": class_name,
            "model_name": MODEL_NAME,
            "embedding_dimension": embeddings.shape[1],
            "total_embeddings": len(embeddings),
            "embeddings": embeddings.tolist(),
            "texts": texts.tolist(),
            "ids": ids.tolist()
        }
        
        # Save to JSON file
        output_path = os.path.join(android_assets_dir, f'{class_name}_embeddings.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(android_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {class_name} embeddings to: {output_path}")
        print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    # Create a metadata file
    metadata = {
        "model_name": MODEL_NAME,
        "classes": classes,
        "generated_at": str(np.datetime64('now')),
        "description": "Embeddings for SMS semantic search in Android app"
    }
    
    metadata_path = os.path.join(android_assets_dir, "embeddings_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\nMetadata saved to: {metadata_path}")
    
    # Create a summary
    print("\n" + "="*60)
    print("EMBEDDINGS PREPARATION COMPLETED")
    print("="*60)
    print(f"Output directory: {android_assets_dir}")
    print("Files created:")
    
    for class_name in classes:
        json_path = os.path.join(android_assets_dir, f'{class_name}_embeddings.json')
        if os.path.exists(json_path):
            size_mb = os.path.getsize(json_path) / (1024*1024)
            print(f"  - {class_name}_embeddings.json ({size_mb:.2f} MB)")
    
    print(f"  - embeddings_metadata.json")
    print("\nNext steps:")
    print("1. Copy the JSON files to your Android app's assets folder")
    print("2. Use the metadata to load embeddings in your Android app")
    print("3. Test semantic search functionality")

def verify_embeddings():
    """
    Verify that the prepared embeddings work correctly.
    """
    print("\nVerifying embeddings...")
    
    android_assets_dir = os.path.join(os.path.dirname(__file__), "..", "android_assets")
    
    for class_name in [CLASS_SMISHING, CLASS_BENIGN]:
        json_path = os.path.join(android_assets_dir, f'{class_name}_embeddings.json')
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            embeddings = np.array(data['embeddings'])
            texts = data['texts']
            ids = data['ids']
            
            print(f"[OK] {class_name}: {len(embeddings)} embeddings loaded")
            print(f"  - Embedding dimension: {embeddings.shape[1]}")
            print(f"  - Sample text: '{texts[0][:50]}...'")
        else:
            print(f"[ERROR] {class_name}: File not found")

if __name__ == "__main__":
    print("="*60)
    print("ANDROID EMBEDDINGS PREPARATION TOOL")
    print("="*60)
    
    # Check if embeddings exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_dir = os.path.join(base_dir, "..", "embeddings")
    
    if not os.path.exists(embeddings_dir):
        print("Error: Embeddings directory not found.")
        print("Please run first: python scripts/generate_embeddings.py")
        sys.exit(1)
    
    # Prepare embeddings
    prepare_embeddings_for_android()
    
    # Verify the results
    verify_embeddings()
    
    print("\nPreparation completed successfully!") 