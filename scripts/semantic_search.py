import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from config import *


# --- Configuration ---
SMS_FILE = os.path.join('data', INPUT_FILE_PATH)
EXTENSION = SMS_FILE.split('.')[-1]
SMS_TEXT_COLUMN = 'sms_text'
EMBEDDING_MODEL = MODEL_NAME

# Class-specific embeddings file paths
SMISHING_EMBEDDINGS_PATH = os.path.join('embeddings', f'{CLASS_SMISHING}_embeddings.npy')
SMISHING_TEXTS_PATH = os.path.join('embeddings', f'{CLASS_SMISHING}_texts.npy')
SMISHING_IDS_PATH = os.path.join('embeddings', f'{CLASS_SMISHING}_ids.npy')
BENIGN_EMBEDDINGS_PATH = os.path.join('embeddings', f'{CLASS_BENIGN}_embeddings.npy')
BENIGN_TEXTS_PATH = os.path.join('embeddings', f'{CLASS_BENIGN}_texts.npy')
BENIGN_IDS_PATH = os.path.join('embeddings', f'{CLASS_BENIGN}_ids.npy')

def load_embeddings_and_texts_for_class(class_name):
    """Load embeddings, texts, and IDs for a specific class."""
    if class_name == CLASS_SMISHING:
        embeddings_path = SMISHING_EMBEDDINGS_PATH
        texts_path = SMISHING_TEXTS_PATH
        ids_path = SMISHING_IDS_PATH
    elif class_name == CLASS_BENIGN:
        embeddings_path = BENIGN_EMBEDDINGS_PATH
        texts_path = BENIGN_TEXTS_PATH
        ids_path = BENIGN_IDS_PATH
    else:
        print(f"Error: Unknown class '{class_name}'")
        return None, None, None
    
    try:
        embeddings = np.load(embeddings_path)
        texts = np.load(texts_path, allow_pickle=True)
        ids = np.load(ids_path, allow_pickle=True)
        return embeddings, texts, ids
    except FileNotFoundError:
        print(f"Error: {class_name} embeddings files not found.")
        print("Run first: python scripts/generate_embeddings.py")
        return None, None, None

def search_similar_sms(query, embeddings, texts, ids, model, top_k=3):
    """
    Search for SMS most similar to the query using cosine similarity.
    
    Args:
        query: Search text
        embeddings: Array of embeddings from the collection
        texts: List of original texts
        ids: List of SMS IDs
        model: SentenceTransformer model
        top_k: Number of results to return
    
    Returns:
        List of dictionaries with text, similarity, and sms_id
    """
    # Generate embedding for the query
    query_embedding = model.encode([query])
    
    # Calculate cosine similarity with all embeddings
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get indices of top_k most similar
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Create results list
    results = []
    for idx in top_indices:
        result = {
            'text': texts[idx],
            'similarity': similarities[idx],
            'sms_id': ids[idx]
        }
        results.append(result)
    
    return results

def semantic_search_sms(sms_text, top_k=3):
    """
    Perform semantic search for a given SMS text across both classes.
    
    Args:
        sms_text (str): The SMS text to search for similar messages
        top_k (int): Number of top results to return for each class
    
    Returns:
        dict: Dictionary containing search results for both classes
            {
                'smishing': [list of smishing results],
                'benign': [list of benign results]
            }
    """
    # Load the embedding model
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Load embeddings, texts, and IDs for both classes
    smishing_embeddings, smishing_texts, smishing_ids = load_embeddings_and_texts_for_class(CLASS_SMISHING)
    benign_embeddings, benign_texts, benign_ids = load_embeddings_and_texts_for_class(CLASS_BENIGN)
    
    results = {
        'smishing': [],
        'benign': []
    }
    
    # Search in smishing class
    if smishing_embeddings is not None and smishing_texts is not None and smishing_ids is not None:
        smishing_results = search_similar_sms(sms_text, smishing_embeddings, smishing_texts, smishing_ids, model, top_k)
        results['smishing'] = smishing_results
    
    # Search in benign class
    if benign_embeddings is not None and benign_texts is not None and benign_ids is not None:
        benign_results = search_similar_sms(sms_text, benign_embeddings, benign_texts, benign_ids, model, top_k)
        results['benign'] = benign_results
    
    return results

def show_results(results):
    """Display search results in a clear format."""
    print("\n" + "="*60)
    print("SEMANTIC SEARCH RESULTS")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. SMS (Similarity: {result['similarity']:.3f})")
        print(f"   SMS ID: {result['sms_id']}")
        print(f"   Text: {result['text']}")
        print("-" * 40)

def interactive_search():
    """Interactive search interface for command line use."""
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded.")
    
    # Load embeddings, texts, and IDs for all classes
    smishing_embeddings, smishing_texts, smishing_ids = load_embeddings_and_texts_for_class(CLASS_SMISHING)
    benign_embeddings, benign_texts, benign_ids = load_embeddings_and_texts_for_class(CLASS_BENIGN)
    
    if smishing_embeddings is None and benign_embeddings is None:
        print("Error: No embeddings found for any class.")
        return
    
    print("\n" + "="*60)
    print("SMS SEMANTIC SEARCH")
    print("="*60)
    print("Type 'exit' or 'q' to quit.")
    print("Type 'help' to see available commands.")
    print("Type 'class <class_name>' to search within a specific class.")
    print(f"Available classes: {CLASS_SMISHING}, {CLASS_BENIGN}")
    
    while True:
        print("\n" + "-"*40)
        query = input("Enter your SMS query: ").strip()
        
        if query.lower() == 'exit' or query.lower() == 'q':
            print("Goodbye!")
            break
        elif query.lower() == 'help':
            print("\nAvailable commands:")
            print("- 'exit' or 'q': Quit the program")
            print("- 'help': Show this help")
            print(f"- 'class {CLASS_SMISHING}': Search only in {CLASS_SMISHING} SMS")
            print(f"- 'class {CLASS_BENIGN}': Search only in {CLASS_BENIGN} SMS")
            print("- Any text: Search across all SMS")
            continue
        elif query.lower().startswith('class '):
            # Search within specific class
            class_name = query[6:].strip().lower()
            if class_name == CLASS_SMISHING and smishing_embeddings is not None:
                search_query = input(f"Enter your query for {class_name} SMS: ").strip()
                if search_query:
                    print(f"\nSearching for {class_name} SMS similar to: '{search_query}'")
                    results = search_similar_sms(search_query, smishing_embeddings, smishing_texts, smishing_ids, model, top_k=3)
                    show_results(results)
            elif class_name == CLASS_BENIGN and benign_embeddings is not None:
                search_query = input(f"Enter your query for {class_name} SMS: ").strip()
                if search_query:
                    print(f"\nSearching for {class_name} SMS similar to: '{search_query}'")
                    results = search_similar_sms(search_query, benign_embeddings, benign_texts, benign_ids, model, top_k=3)
                    show_results(results)
            else:
                print(f"Unknown class '{class_name}' or embeddings not available. Available classes: {CLASS_SMISHING}, {CLASS_BENIGN}")
            continue
        elif not query:
            print("Please enter a valid query.")
            continue
        
        # Perform search using the main function
        print(f"\nSearching for SMS similar to: '{query}'")
        search_results = semantic_search_sms(query, top_k=3)
        
        # Display results for each class
        if search_results['smishing']:
            print(f"\n--- {CLASS_SMISHING.upper()} RESULTS ---")
            show_results(search_results['smishing'])
        
        if search_results['benign']:
            print(f"\n--- {CLASS_BENIGN.upper()} RESULTS ---")
            show_results(search_results['benign'])

if __name__ == "__main__":
    interactive_search() 