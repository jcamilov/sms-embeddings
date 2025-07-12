# Configuration parameters for SMS classifier embeddings

# Model configuration
# Model                                    Size     Speed       Quality
# all-MiniLM-L3-v2                         ~60MB    ⚡⚡⚡     ⭐⭐
# all-MiniLM-L6-v2                         ~80MB    ⚡⚡        ⭐⭐⭐
# paraphrase-multilingual-MiniLM-L12-v2    ~117MB   ⚡          ⭐⭐⭐⭐
MODEL_NAME = "all-MiniLM-L6-v2"  

# Input file configuration
# this is the file that contains the dataset with the sms messages to be used to get embeddings for the semantic search
# IMPORTANT: 
# Please make sure that this the headers contain at least "class" ,"sms_id", "sms_text"
# Also, the class column should be either "smishing" or "benign" (or change it here if different)
INPUT_FILE_PATH = "data/combined_limited.csv"  # Full path to the CSV file for generating embeddings
CLASS_SMISHING = "smishing"
CLASS_BENIGN = "benign"

