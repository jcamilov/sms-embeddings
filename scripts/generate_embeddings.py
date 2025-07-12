import os
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import time
from config import MODEL_NAME, INPUT_FILE_PATH, CLASS_SMISHING, CLASS_BENIGN

# --- Configuration ---
SMS_FILE = INPUT_FILE_PATH  # Path to the .CSV or .py file
EXTENSION = SMS_FILE.split('.')[-1]
SMS_TEXT_COLUMN = 'sms_text'  # Name of the column with SMS messages
SMS_ID_COLUMN = 'sms_id'  # Name of the column with SMS IDs
EMBEDDING_MODEL = MODEL_NAME

print(f"Loading embedding model: {EMBEDDING_MODEL}...")
model = SentenceTransformer(EMBEDDING_MODEL)
print("Model loaded.")

# --- Load SMS from file ---
print(f"Loading SMS data from: {SMS_FILE}")

# if the file is .py, load the SMS file from the data folder
if SMS_FILE.endswith('.py'):
    sms_collection = pd.read_py(SMS_FILE)
    sms_collection = sms_collection[SMS_TEXT_COLUMN].dropna().tolist()
    print(f"Loaded {len(sms_collection)} SMS from the collection.")
else:
    df_sms = pd.read_csv(SMS_FILE)
    print(f"Loaded {len(df_sms)} total SMS records.")
    
    # Separate SMS by class
    smishing_data = df_sms[df_sms['class'] == CLASS_SMISHING][[SMS_ID_COLUMN, SMS_TEXT_COLUMN]].dropna()
    benign_data = df_sms[df_sms['class'] == CLASS_BENIGN][[SMS_ID_COLUMN, SMS_TEXT_COLUMN]].dropna()
    
    smishing_sms = smishing_data[SMS_TEXT_COLUMN].tolist()
    smishing_ids = smishing_data[SMS_ID_COLUMN].tolist()
    benign_sms = benign_data[SMS_TEXT_COLUMN].tolist()
    benign_ids = benign_data[SMS_ID_COLUMN].tolist()
    
    print(f"Found {len(smishing_sms)} {CLASS_SMISHING} SMS")
    print(f"Found {len(benign_sms)} {CLASS_BENIGN} SMS")

# --- Generate Embeddings for each class ---
def generate_embeddings_for_class(sms_list, sms_ids, class_name):
    """Generate embeddings for a specific class of SMS"""
    if not sms_list:
        print(f"No {class_name} SMS found. Skipping...")
        return
    
    print(f"\nGenerating embeddings for {class_name} SMS...")
    print(f"Total {class_name} SMS to process: {len(sms_list)}")
    
    # Start timer
    start_time = time.time()
    
    # Process in batches to show detailed progress
    batch_size = 100
    total_sms = len(sms_list)
    sms_embeddings = []
    
    for i in range(0, total_sms, batch_size):
        batch = sms_list[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        sms_embeddings.extend(batch_embeddings)
        
        # Calculate progress and ETA
        processed = min(i + batch_size, total_sms)
        progress = (processed / total_sms) * 100
        
        print(f"Processed: {processed}/{total_sms} {class_name} SMS ({progress:.1f}%)")
        
        # Simple ETA based on average time per batch
        if i > 0:
            elapsed_time = time.time() - start_time
            avg_time_per_batch = elapsed_time / (i // batch_size + 1)
            remaining_batches = (total_sms - processed) // batch_size + (1 if total_sms % batch_size > 0 else 0)
            eta_seconds = remaining_batches * avg_time_per_batch
            eta_minutes = eta_seconds / 60
            print(f"Estimated time remaining: {eta_minutes:.1f} minutes")
    
    sms_embeddings = np.array(sms_embeddings)
    print(f"{class_name} embeddings generated.")
    
    # Create embeddings folder if it doesn't exist
    os.makedirs('embeddings', exist_ok=True)
    
    # Save embeddings, texts, and IDs for this class
    embeddings_path = os.path.join('embeddings', f'{class_name}_embeddings.npy')
    texts_path = os.path.join('embeddings', f'{class_name}_texts.npy')
    ids_path = os.path.join('embeddings', f'{class_name}_ids.npy')
    
    np.save(embeddings_path, sms_embeddings)
    np.save(texts_path, np.array(sms_list))
    np.save(ids_path, np.array(sms_ids))
    
    print(f"{class_name} embeddings, texts, and IDs saved in the 'embeddings/' folder.")
    return embeddings_path, texts_path, ids_path

# Generate embeddings for each class
if SMS_FILE.endswith('.py'):
    # For .py files, process all SMS together (legacy behavior)
    sms_collection = pd.read_py(SMS_FILE)
    sms_collection = sms_collection[SMS_TEXT_COLUMN].dropna().tolist()
    print(f"Loaded {len(sms_collection)} SMS from the collection.")
    
    # Process all SMS together
    print("Generating embeddings for the SMS collection...")
    print(f"Total SMS to process: {len(sms_collection)}")
    
    # Start timer
    start_time = time.time()
    
    # Process in batches to show detailed progress
    batch_size = 100
    total_sms = len(sms_collection)
    sms_embeddings = []
    
    for i in range(0, total_sms, batch_size):
        batch = sms_collection[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        sms_embeddings.extend(batch_embeddings)
        
        # Calculate progress and ETA
        processed = min(i + batch_size, total_sms)
        progress = (processed / total_sms) * 100
        
        print(f"Processed: {processed}/{total_sms} SMS ({progress:.1f}%)")
        
        # Simple ETA based on average time per batch
        if i > 0:
            elapsed_time = time.time() - start_time
            avg_time_per_batch = elapsed_time / (i // batch_size + 1)
            remaining_batches = (total_sms - processed) // batch_size + (1 if total_sms % batch_size > 0 else 0)
            eta_seconds = remaining_batches * avg_time_per_batch
            eta_minutes = eta_seconds / 60
            print(f"Estimated time remaining: {eta_minutes:.1f} minutes")
    
    sms_embeddings = np.array(sms_embeddings)
    print("Embeddings generated.")
    
    # Create embeddings folder if it doesn't exist
    os.makedirs('embeddings', exist_ok=True)
    
    # Save embeddings and texts
    embeddings_path = os.path.join('embeddings', SMS_FILE.replace('.' + EXTENSION, '_embeddings.npy'))
    texts_path = os.path.join('embeddings', SMS_FILE.replace('.' + EXTENSION, '_texts.npy'))
    
    np.save(embeddings_path, sms_embeddings)
    np.save(texts_path, np.array(sms_collection))
    
    print(f"Embeddings and texts saved in the 'embeddings/' folder.")
else:
    # For CSV files, process each class separately
    generate_embeddings_for_class(smishing_sms, smishing_ids, CLASS_SMISHING)
    generate_embeddings_for_class(benign_sms, benign_ids, CLASS_BENIGN)

print("\nEmbedding generation process completed.") 