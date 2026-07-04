import math
import numpy as np 
import scipy
import torch
from sentence_transformers import SentenceTransformer

# Example documents
documents = [
    'Bugs introduced by the intern had to be squashed by the lead developer.',
    'Bugs found by the quality assurance engineer were difficult to debug.',
    'Bugs are common throughout the warm summer months, according to the entomologist.',
    'Bugs, in particular spiders, are extensively studied by arachnologists.'
]

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(documents)

print("=" * 80)
print("🎓 SIMILARITY SEARCH: UNDERSTANDING EMBEDDINGS")
print("=" * 80)

print("\n📝 INPUT DOCUMENTS:")
for idx, doc in enumerate(documents):
    print(f"  [{idx}] '{doc}'")

print("\n⚙️ MODEL INFO:")
print("  • Model ID: paraphrase-MiniLM-L6-v2")
print("  • Purpose:  Transforms text into a 384-dimensional semantic space")

print(f"\n📊 EMBEDDINGS MATRIX SHAPE: {embeddings.shape}")
print("  • 4   -> Total number of documents (rows)")
print("  • 384 -> Dimensionality of the vectors (columns)")

print("\n🔍 VECTOR REPRESENTATION (Truncated by NumPy for readability):")
print(embeddings)
print("=" * 80)

### Manual implementation of L2 distance calculation

def euclidean_distance_fn(vector1, vector2):
    squared_sum = sum((x - y) ** 2 for x, y in zip(vector1, vector2))
    return math.sqrt(squared_sum)

# Calculate distance matrix for Model 1 (paraphrase-MiniLM-L6-v2)
l2_dist_manual = np.zeros([4,4])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        l2_dist_manual[i,j] = euclidean_distance_fn(embeddings[i], embeddings[j])

# Helper function to print matrices in a clear educational grid
def print_labeled_matrix(matrix, title):
    print(f"\n📏 {title}:")
    print("        [Doc 0]  [Doc 1]  [Doc 2]  [Doc 3]")
    for idx, row in enumerate(matrix):
        formatted_row = "  ".join(f"{val:7.2f}" for val in row)
        print(f"Doc {idx} |  {formatted_row}")

print_labeled_matrix(l2_dist_manual, "L2 DISTANCES FOR Model 1: paraphrase-MiniLM-L6-v2")

print("\n" + "=" * 80)
print("⚙️ LOADING SECOND MODEL: 'all-MiniLM-L6-v2' (General-Purpose choice)...")
print("=" * 80)

model_alt = SentenceTransformer('all-MiniLM-L6-v2')
embeddings_alt = model_alt.encode(documents)

# Calculate distance matrix for Model 2 (all-MiniLM-L6-v2)
l2_dist_alt = np.zeros([4,4])
for i in range(embeddings_alt.shape[0]):
    for j in range(embeddings_alt.shape[0]):
        # 2. Only calculate the upper triangle (where column index j is greater than row index i)
        if j > i: 
            l2_dist_alt[i,j] = euclidean_distance_fn(embeddings_alt[i], embeddings_alt[j])
        # 3. If we are below the diagonal (row index i is greater than column j), copy the value
        elif i > j: 
            l2_dist_alt[i,j] = l2_dist_alt[j,i]
print_labeled_matrix(l2_dist_alt, "L2 DISTANCES FOR Model 2: all-MiniLM-L6-v2")
print("=" * 80)

### Calculate L2 distance using `scipy`

l2_dist_scipy_model1 = scipy.spatial.distance.cdist(embeddings, embeddings, 'euclidean')
l2_dist_scipy_model2 = scipy.spatial.distance.cdist(embeddings_alt, embeddings_alt, 'euclidean')

print("\n" + "=" * 80)
print("🔬 STEP 3: CROSS-DISTANCE CALCULATION WITH SCIPY (`scipy.spatial.distance.cdist`)")
print("=" * 80)

print_labeled_matrix(l2_dist_scipy_model1, "L2 DISTANCES Model 1 (Scipy 'euclidean')")
print_labeled_matrix(l2_dist_scipy_model2, "L2 DISTANCES Model 2 (Scipy 'euclidean')")

# Check identity using np.allclose to verify the manual math matches library implementations
match_model1 = np.allclose(l2_dist_manual, l2_dist_scipy_model1)
match_model2 = np.allclose(l2_dist_alt, l2_dist_scipy_model2)

print("\n🔍 VERIFYING EQUIVALENCE (Manual Nested Loop vs. Scipy C-Library):")
print(f"  • Model 1 matrices are identical? {match_model1} ✅" if match_model1 else "  • Model 1 mismatch ❌")
print(f"  • Model 2 matrices are identical? {match_model2} ✅" if match_model2 else "  • Model 2 mismatch ❌")
print("=" * 80)

## Dot Product Similarity and Distance

### Manual implementation of dot product calculation

def dot_product_fn(vector1, vector2):
    return sum(x * y for x, y in zip(vector1, vector2))

# Generate manual dot product matrix
dot_product_manual = np.empty([4,4])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        dot_product_manual[i,j] = dot_product_fn(embeddings[i], embeddings[j])

# Matrix multiplication operator
dot_product_operator = embeddings @ embeddings.T

print("\n" + "=" * 80)
print("🎓 SIMILARITY SEARCH: DOT PRODUCT SIMILARITY & DISTANCE")
print("=" * 80)

print_labeled_matrix(dot_product_manual, "DOT PRODUCT SIMILARITY (Manual Loop)")
print_labeled_matrix(dot_product_operator, "DOT PRODUCT SIMILARITY (Matrix multiplication `embeddings @ embeddings.T`)")

# Check identity using np.allclose
match_dot = np.allclose(dot_product_manual, dot_product_operator, atol=1e-05)

print("\n🔍 VERIFYING EQUIVALENCE (Manual Loop vs. Matrix Multiplication):")
print(f"  • Dot product matrices are identical? {match_dot} ✅" if match_dot else "  • Dot product mismatch ❌")

# Calculate dot product distance
dot_product_distance = -dot_product_manual
print_labeled_matrix(dot_product_distance, "DOT PRODUCT DISTANCE (Negative of Dot Product Similarity)")
print("  💡 Note: Similar documents (like Doc 0 and 1) have smaller/more negative values (-18.54 vs. -8.57).")
print("=" * 80)

## Cosine Similarity and Distance

# L2 norms
l2_norms = np.sqrt(np.sum(embeddings**2, axis=1))
l2_norms_reshaped = l2_norms.reshape(-1,1)

normalized_embeddings_manual = embeddings/l2_norms_reshaped

#### Normalize embeddings using PyTorch
normalized_embeddings_torch = torch.nn.functional.normalize(
    torch.from_numpy(embeddings)
).numpy()

### Calculate cosine similarity using matrix multiplication
cosine_similarity_operator = normalized_embeddings_manual @ normalized_embeddings_manual.T

print("\n" + "=" * 80)
print("🎓 SIMILARITY SEARCH: COSINE SIMILARITY")
print("=" * 80)

print(f"\n📊 L2 NORMS (Vector Magnitudes):")
for idx, norm in enumerate(l2_norms):
    print(f"  • Doc {idx}: {norm:.4f}")

# Check equivalence of PyTorch vs Manual Normalization
match_normalization = np.allclose(normalized_embeddings_manual, normalized_embeddings_torch)
print(f"\n🔍 VERIFYING NORMALIZATION EQUIVALENCE (Manual division vs. PyTorch normalize):")
print(f"  • Normalized embeddings are identical? {match_normalization} ✅" if match_normalization else "  • Mismatch ❌")

print_labeled_matrix(cosine_similarity_operator, "COSINE SIMILARITY MATRIX")
print("  💡 Note: Values lie between -1 and 1. The diagonal is exactly 1.0 (perfect self-similarity).")
print("=" * 80)





