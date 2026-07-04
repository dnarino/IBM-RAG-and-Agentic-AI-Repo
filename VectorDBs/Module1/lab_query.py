#EXERCISE

"""
In the above examples, we calculated similarity between 4 documents:

```python
documents = [
    'Bugs introduced by the intern had to be squashed by the lead developer.',
    'Bugs found by the quality assurance engineer were difficult to debug.',
    'Bugs are common throughout the warm summer months, according to the entomologist.',
    'Bugs, in particular spiders, are extensively studied by arachnologists.'
]
```

Now, your task is to find which of these 4 documents is most similar to the query 
`Who is responsible for a coding project and fixing others' mistakes?` using cosine similarity. 
You can reuse the `documents` and `normalized_embeddings_manual` arrays in your answer:

"""

from torch import embedding
import numpy as np
from sentence_transformers import SentenceTransformer

documents = [
    'Bugs introduced by the intern had to be squashed by the lead developer.',
    'Bugs found by the quality assurance engineer were difficult to debug.',
    'Bugs are common throughout the warm summer months, according to the entomologist.',
    'Bugs, in particular spiders, are extensively studied by arachnologists.',
    'Bugs are made by all developers and the senior developer is responsible for fixing them.'
]


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def retrieve_best_match(query_text, document_list, embedding_model):
    # 1. Generate embeddings
    docs_emb = embedding_model.encode(document_list)
    query_emb = embedding_model.encode([query_text]) # Wrap query in a list
    
    # 2. Normalize vectors (for Cosine Similarity)
    # np.linalg.norm is a faster, built-in way to calculate L2 norms
    norm_docs = docs_emb / np.linalg.norm(docs_emb, axis=1, keepdims=True)
    norm_query = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    
    # 3. Calculate similarity scores (vectorized dot product)
    scores = (norm_docs @ norm_query.T).flatten()
    print(scores)
    
    # 4. Automatically find the best index and score
    best_idx = np.argmax(scores)
    
    return document_list[best_idx], scores[best_idx]

# Try a new query
user_query = "why are to many bugs in this hot december?"

best_doc, best_score = retrieve_best_match(user_query, documents, model)

print(f"🏆 Best Score: {best_score:.4f}")
print(f"🏆 Best Matching Document: '{best_doc}'")
