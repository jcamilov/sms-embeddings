# SMS Semantic Search with Embeddings

A Python project for performing semantic search on SMS messages using sentence embeddings. This tool can classify SMS messages into "smishing" (phishing via SMS) and "benign" categories and find similar messages within each class.

## Features

- **Semantic Search**: Find similar SMS messages using cosine similarity
- **Class-based Search**: Search separately within smishing and benign SMS classes
- **Configurable**: Easy configuration through `config.py`
- **SMS ID Tracking**: Results include original SMS IDs for traceability

## Quick Start

### Environment setup

1. **Create a virtual environment** (recommended):
```bash
python -m venv venv

# Activate the virtual environment:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

2. **Install required packages**:
```bash
pip install -r requirements.txt
```

### 1. Prepare Your Data

Create a CSV file with the following columns:
- `sms_id`: Unique identifier for each SMS
- `sms_text`: The SMS message content
- `class`: Either "smishing" or "benign"

Example CSV structure:
```csv
sms_id,sms_text,class
SMS_001,Your bank account is suspended. Click here: http://fake-bank.com,smishing
SMS_002,Hi mom, can you pick me up?,benign
SMS_003,URGENT: Package delivery failed. Call now!,smishing
```

Place your CSV file in the `data/` folder.

### 2. Configure the Project

Edit `scripts/config.py` to match your setup:

```python
# Model configuration
MODEL_NAME = "all-MiniLM-L6-v2"  # Choose your preferred model

# Input file configuration
INPUT_FILE_PATH = "data/your_file.csv"  # Path to your CSV file
CLASS_SMISHING = "smishing"  # Your smishing class label
CLASS_BENIGN = "benign"      # Your benign class label
```

### 3. Generate Embeddings

Run the embedding generation script:

```bash
python scripts/generate_embeddings.py
```

This will:
- Load your CSV file
- Separate SMS by class (smishing/benign)
- Generate embeddings for each class
- Save files in `embeddings/` folder:
  - `smishing_embeddings.npy`
  - `smishing_texts.npy`
  - `smishing_ids.npy`
  - `benign_embeddings.npy`
  - `benign_texts.npy`
  - `benign_ids.npy`

### 4. Use Semantic Search
In your code: (also see below for examples)

```python
from scripts.semantic_search import semantic_search_sms

# Search for similar SMS
results = semantic_search_sms("Your SMS text here", top_k=3)

# Access results
smishing_results = results['smishing']
benign_results = results['benign']

# Each result contains:
# - text: The SMS message
# - similarity: Similarity score (0-1)
# - sms_id: Original SMS ID
```

#### Option C: Run Tests
```bash
python scripts/test_semantic_search.py
```

## Example Usage

### Search Results Format

```python
results = semantic_search_sms("bank account suspended", top_k=3)

# Results structure:
{
    'smishing': [
        {
            'text': 'Your account has been locked. Verify here: http://bank-secure.com',
            'similarity': 0.892,
            'sms_id': 'SMS_000123'
        },
        # ... more results
    ],
    'benign': [
        {
            'text': 'Hi, how are you doing today?',
            'similarity': 0.234,
            'sms_id': 'SMS_000789'
        },
        # ... more results
    ]
}
```
