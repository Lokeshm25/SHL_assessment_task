import json
import pickle
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def build_data_pipeline():
    print("Loading catalog...")
    with open('shl_product_catalog.json', 'r', encoding='utf-8') as f:
        # Using strict=False to handle unescaped control characters in the JSON
        data = json.loads(f.read(), strict=False)
        
    print(f"Loaded {len(data)} items. Filtering and processing...")
    
    valid_items = []
    documents = []
    metadata = []
    exact_match_lookup = {}
    
    # Simple set to check for pre-packaged job solutions as requested
    job_solution_keywords = ['pre-packaged', 'job solution']
    
    for idx, item in enumerate(data):
        name = item.get('name', '')
        desc = item.get('description', '')
        
        name_lower = name.lower()
        desc_lower = desc.lower()
        
        # 1. Filter out pre-packaged job solutions
        if any(kw in name_lower or kw in desc_lower for kw in job_solution_keywords):
            continue
            
        # 2. Extract test_type
        keys = item.get('keys', [])
        test_type = "U" # Unknown
        if keys:
            first_key = keys[0]
            # Map standard categories or just use first letter
            if "Knowledge" in first_key:
                test_type = "K"
            elif "Ability" in first_key:
                test_type = "A"
            elif "Personality" in first_key:
                test_type = "P"
            else:
                test_type = first_key[0].upper()
                
        # 3. Create rich text document for embedding
        # We heavily weight the name by repeating it
        doc = f"Name: {name}. Name: {name}. Type: {keys}. Description: {desc}. Level: {item.get('job_levels_raw', '')}"
        
        # 4. Save metadata
        item_meta = {
            "name": name,
            "url": item.get('link', ''),
            "test_type": test_type,
            "description": desc,
            "keys": keys,
            "duration": item.get('duration', '')
        }
        
        curr_idx = len(valid_items)
        valid_items.append(item)
        documents.append(doc)
        metadata.append(item_meta)
        
        # 5. Populate exact match lookup
        # Full name exact match
        exact_match_lookup[name_lower] = curr_idx
        
        # Extract potential acronyms (e.g., OPQ32r from Occupational Personality Questionnaire OPQ32r)
        tokens = name.split()
        for token in tokens:
            # Strip punctuation
            clean_token = re.sub(r'[^\w\s]', '', token)
            # If it's mostly uppercase or contains numbers (like OPQ32r, GSA)
            if re.match(r'^[A-Z0-9]+[a-z]?$', clean_token) and len(clean_token) > 1:
                exact_match_lookup[clean_token.lower()] = curr_idx
                
    print(f"Retained {len(valid_items)} valid individual test solutions.")
    print("Generating embeddings. This may take a moment...")
    
    # 6. Generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
    
    # 7. Build FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # 8. Save artifacts
    print("Saving index and metadata...")
    faiss.write_index(index, 'shl_catalog.index')
    
    with open('catalog_metadata.pkl', 'wb') as f:
        pickle.dump({
            'metadata': metadata,
            'exact_match_lookup': exact_match_lookup
        }, f)
        
    print("Data pipeline execution complete. Artifacts saved.")

if __name__ == "__main__":
    build_data_pipeline()
