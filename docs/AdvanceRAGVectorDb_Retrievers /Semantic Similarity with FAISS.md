<p style="text-align:center">
    <a href="https://skills.network" target="_blank">
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo">
    </a>
</p>


# Semantic Similarity with FAISS

Estimated time needed: **60** minutes

Welcome to a hands-on exploration of semantic search, where we unravel the intricacies of finding meaning in text. This lab is a beginner's journey into the realm of advanced information retrieval. You'll start by learning the essentials of text preprocessing to enhance data quality. Next, you'll dive into the world of vector spaces, using the Universal Sentence Encoder to convert text into a format that machines understand. Finally, you'll harness the efficiency of FAISS, a library built for rapid similarity search, to compare and retrieve information. By the end of our session, you'll have a functional semantic search engine that not only understands the subtleties of human language but also fetches information that truly matters.

<p style='color: red'>Embark on this learning adventure to build a search engine that sees beyond the obvious, leveraging context and semantics to satisfy the quest for information.</p>


# __Table of Contents__

<ol>
    <li><a href="#Objectives">Objectives</a></li>
    <li>
        <a href="#Setup">Setup</a>
        <ol>
            <li><a href="#Installing-Required-Libraries">Installing Required Libraries</a></li>
            <li><a href="#Importing-Required-Libraries">Importing Required Libraries</a></li>
        </ol>
    </li>
    <li>
        <a href="#Understanding-Semantic-Search">Understanding Semantic Search</a>
    </li>
    <li><a href="#Understanding-Vectorization-and-Indexing">Understanding Vectorization and Indexing</a></li>
    <li><a href="#The-20-Newsgroups-Dataset">The 20 Newsgroups Dataset</a></li>
    <li><a href="#Pre-processing-Data">Pre-processing Data</a></li>
    <li><a href="#Universal-Sentence-Encoder">Universal Sentence Encoder</a></li>
    <li><a href="#Indexing-with-FAISS">Indexing with FAISS</a></li>
</ol>


---


# Objectives

In this lab, our objectives are to:

- Understand the fundamentals of semantic search and its advantages over traditional search methods.
- Familiarize with the process of preparing text data for semantic analysis, including cleaning and standardization techniques.
- Learn how to utilize the Universal Sentence Encoder to convert text into high-dimensional vector space representations.
- Gain practical experience with FAISS (Facebook AI Similarity Search), an efficient library for indexing and searching high-dimensional vectors.
- Apply these techniques to build a fully functioning semantic search engine that can interpret and respond to natural language queries.

By accomplishing these objectives, you will acquire a comprehensive skill set that underpins advanced search functionalities in modern AI-driven systems, preparing you for further exploration and development in the field of natural language processing and information retrieval.


---


# Setup

To ensure a smooth experience throughout this lab, we need to set up our environment properly. This includes installing necessary libraries, importing them, and preparing helper functions that will be used later in the lab.

## Installing Required Libraries

Before we start, you need to install the following libraries if you haven't already:

- `tensorflow`: The core library for TensorFlow, required for working with the Universal Sentence Encoder.
- `tensorflow-hub`: A library that makes it easy to download and deploy pre-trained TensorFlow models, including the Universal Sentence Encoder.
- `faiss-cpu`: A library for efficient similarity search and clustering of dense vectors.
- `numpy`: A library for numerical computing, which we will use to handle arrays and matrices.
- `scikit-learn`: A machine learning library that provides various tools for data mining and data analysis, useful for additional tasks like data splitting and evaluation metrics.

You can install these libraries using `pip` with the following commands:


The following required libraries are __not__ pre-installed in the Skills Network Labs environment. __You will need to run the following cell__ to install them:



```python
!pip install faiss-cpu numpy scikit-learn
!pip install "tensorflow>=2.0.0"
!pip install --upgrade tensorflow-hub
```

### Importing Required Libraries

_We recommend you import all required libraries in one place (here):_



```python
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import faiss
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

# Suppressing warnings
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')
```

---


## Understanding Semantic Search

When we're looking to build a semantic search engine, it's important to start with the basics. Let's break down what semantic search is and why it's a game-changer in finding information.

### What is Semantic Search?

Semantic search transcends the limitations of traditional keyword searches by understanding the context and nuances of language in user queries. At its core, semantic search:

- Enhances the search experience by interpreting the intent and contextual meaning behind search queries.
- Delivers more accurate and relevant search results by analyzing the relationships between words and phrases within the search context.
- Adapts to user behavior and preferences, refining search results for better user satisfaction.

### How Semantic Search Works - The Simple Version

Now, how does this smart assistant do its job? It uses some clever tricks from a field called Natural Language Processing, or NLP for short. Here’s the simple version of the process:

- **Getting the Gist**: First up, the search engine listens to your query and tries to get the gist of it. Instead of just spotting keywords, it digs deeper to find the real meaning.
- **Making Connections**: Next, it thinks about all the different ways words can be related (like "doctor" and "physician" meaning the same thing). This helps it get a better sense of what you're asking for.
- **Picking the Best**: Finally, it acts like a librarian who knows every book in the library. It sorts through tons of information to pick what matches your query best, considering what you probably mean.

### The Technical Side of Semantic Search

After understanding the basics, let's peek under the hood at the technical engine powering semantic search. This part is a bit like math class, where we learn about vectors — no, not the ones you learned in physics, but something similar that we use in search engines.

#### Vectors: The Language of Semantic Search

In the world of semantic search, a vector is a list of numbers that a computer uses to represent the meaning of words or sentences. Imagine each word or sentence as a point in space. The closer two points are, the more similar their meanings.

- **Creating Vectors**: We start by turning words or sentences into vectors using models like the Universal Sentence Encoder. It's like giving each piece of text its unique numerical fingerprint.
- **Calculating Similarity**: To find out how similar two pieces of text are, we measure how close their vectors are in space. This is done using mathematical formulas, such as cosine similarity, which tells us how similar or different two text fingerprints are.
- **Using Vectors for Search**: When you search for something, the search engine looks for the vectors closest to the vector of your query. The closest vectors represent the most relevant results to what you're asking.

#### How Vectors Power Our Search

Vectors are powerful because they can capture the subtle meanings of language that go beyond the surface of words. Here's what happens in a semantic search engine:

1. **Vectorization**: When we type in a search query, the engine immediately turns our words into a vector.
2. **Indexing**: It then quickly scans through a massive index of other vectors, each representing different pieces of information.
3. **Retrieval**: By finding the closest matching vectors, the engine retrieves information that's not just textually similar but semantically related.

By the end of this guide, you'll understand how to create a search engine that does all of this and more. We'll start simple and build up step by step. Ready? Let's get started!


---


## Understanding Vectorization and Indexing

Vectorization and indexing are key components of building a semantic search engine. Let's explore how they work using the Universal Sentence Encoder (USE) and FAISS.

### What does the Universal Sentence Encoder do?

The Universal Sentence Encoder (USE) takes sentences, no matter how complex, and turns them into vectors. These vectors are arrays of numbers that capture the essence of sentences. Here's why it's amazing:

- **Language Comprehension**: USE understands the meaning of sentences by considering the context in which each word is used.
- **Versatility**: It's trained on a variety of data sources, enabling it to handle a wide range of topics and sentence structures.
- **Speed**: Once trained, USE can quickly convert sentences to vectors, making it highly efficient.

### How does the Universal Sentence Encoder work?

The magic of USE lies in its training. It uses deep learning models to digest vast amounts of text. Here’s what it does:

1. **Analyzes Words**: It looks at each word in a sentence and the words around it to get a full picture of their meaning.
2. **Understands Context**: It pays attention to the order of words and how they're used together to grasp the sentence's intent.
3. **Creates Vectors**: It converts all this understanding into a numeric vector that represents the sentence.

### What is FAISS and what does it do?

FAISS, developed by Facebook AI, is a library for efficient similarity search. After we have vectors from USE, we need a way to search through them quickly to find the most relevant ones to a query. FAISS does just that:

- **Efficient Searching**: It uses optimized algorithms to rapidly search through large collections of vectors.
- **Scalability**: It can handle databases of vectors that are too large to fit in memory, making it suitable for big data applications.
- **Accuracy**: It provides highly accurate search results, thanks to its advanced indexing strategies.

### How does FAISS work?

FAISS creates an index of all the vectors, which allows it to search through them efficiently. Here's a simplified version of its process:

1. **Index Building**: It organizes vectors in a way that similar ones are near each other, making it faster to find matches.
2. **Searching**: When you search with a new vector, FAISS quickly identifies which part of the index to look at for the closest matches.
3. **Retrieving Results**: It then retrieves the most similar vectors, which correspond to the most relevant search results.

Putting it all together:

With USE and FAISS, we have a powerful duo. USE helps us understand language in numerical terms, and FAISS lets us search through these numbers to find meaningful connections. Combining them, we create a semantic search engine that's both smart and swift.

<!-- Insert a diagram that visually represents the flow from text input to vectorization with USE to searching and indexing with FAISS -->


---


## The 20 Newsgroups Dataset

In this project, we'll be using the 20 Newsgroups dataset, a collection of approximately 20,000 newsgroup documents, partitioned across 20 different newsgroups. It's a go-to dataset in the NLP community because it presents real-world challenges:

### What is the 20 Newsgroups Dataset?

- **Diverse Topics**: The dataset spans 20 different topics, from sports and science to politics and religion, reflecting the diverse interests of newsgroup members.
- **Natural Language**: It contains actual discussions, with all the nuances of human language, making it ideal for semantic search.
- **Prevalence of Context**: The conversations within it require understanding of context to differentiate between the topics effectively.

### How are we using the 20 Newsgroups Dataset?

1. **Exploring Data**: We'll start by loading the dataset and exploring its structure to understand the kind of information it holds.
2. **Preprocessing**: We'll clean the text data, removing any unwanted noise that could affect our semantic analysis.
3. **Vectorization**: We'll then use the Universal Sentence Encoder to transform this text into numerical vectors that capture the essence of each document.
4. **Semantic Search Implementation**: Finally, we'll use FAISS to index these vectors, allowing us to perform fast and efficient semantic searches across the dataset.

By working with the 20 Newsgroups dataset, you'll gain hands-on experience with real-world data and the end-to-end process of building a semantic search engine.

<!-- An image of a sample newsgroup post or a chart showing the distribution of topics within the dataset can be helpful here -->



```python
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')
```


```python
pprint(list(newsgroups_train.target_names))
```


```python
# Display the first 3 posts from the dataset
for i in range(3):
    print(f"Sample post {i+1}:\n")
    pprint(newsgroups_train.data[i])
    print("\n" + "-"*80 + "\n")
```

---


# Pre-processing Data

In this section, we focus on preparing the text data from the 20 Newsgroups dataset for our semantic search engine. Preprocessing is a critical step to ensure the quality and consistency of the data before it's fed into the Universal Sentence Encoder.

## Steps in Preprocessing:

1. **Fetching Data**: 
   - We load the complete 20 Newsgroups dataset using `fetch_20newsgroups` from `sklearn.datasets`. 
   - `documents = newsgroups.data` stores all the newsgroup documents in a list.

2. **Defining the Preprocessing Function**:
   - The `preprocess_text` function is designed to clean each text document. Here's what it does to every piece of text:
     - **Removes Email Headers**: Strips off lines that start with 'From:' as they usually contain metadata like email addresses.
     - **Eliminates Email Addresses**: Finds patterns resembling email addresses and removes them.
     - **Strips Punctuations and Numbers**: Removes all characters except alphabets, aiding in focusing on textual data.
     - **Converts to Lowercase**: Standardizes the text by converting all characters to lowercase, ensuring uniformity.
     - **Trims Excess Whitespace**: Cleans up any extra spaces, tabs, or line breaks.

3. **Applying Preprocessing**:
   - We iterate over each document in the `documents` list and apply our `preprocess_text` function.
   - The cleaned documents are stored in `processed_documents`, ready for further processing.

By preprocessing the text data in this way, we reduce noise and standardize the text, which is essential for achieving meaningful semantic analysis in later steps.



```python
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Basic preprocessing of text data
def preprocess_text(text):
    # Remove email headers
    text = re.sub(r'^From:.*\n?', '', text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    # Remove punctuations and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Preprocess each document
processed_documents = [preprocess_text(doc) for doc in documents]
```


```python
# Choose a sample post to display
sample_index = 0  # for example, the first post in the dataset

# Print the original post
print("Original post:\n")
print(newsgroups_train.data[sample_index])
print("\n" + "-"*80 + "\n")

# Print the preprocessed post
print("Preprocessed post:\n")
print(preprocess_text(newsgroups_train.data[sample_index]))
print("\n" + "-"*80 + "\n")
```

---


# Universal Sentence Encoder

After preprocessing the text data, the next step is to transform this cleaned text into numerical vectors using the Universal Sentence Encoder (USE). These vectors capture the semantic essence of the text.

### Loading the USE Module:

- We use TensorFlow Hub (`hub`) to load the pre-trained Universal Sentence Encoder.
- `embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")` fetches the USE module, making it ready for vectorization.

### Defining the Embedding Function:

- The `embed_text` function is defined to take a piece of text as input and return its vector representation.
- Inside the function, `embed(text)` converts the text into a high-dimensional vector, capturing the nuanced semantic meaning.
- `.numpy()` is used to convert the result from a TensorFlow tensor to a NumPy array, which is a more versatile format for subsequent operations.

### Vectorizing Preprocessed Documents:

- We then apply the `embed_text` function to each document in our preprocessed dataset, `processed_documents`.
- `np.vstack([...])` stacks the vectors vertically to create a 2D array, where each row represents a document.
- The resulting array `X_use` holds the vectorized representations of all the preprocessed documents, ready to be used for semantic search indexing and querying.

By vectorizing the text with USE, we've now converted our textual data into a format that can be efficiently processed by machine learning algorithms, setting the stage for the next step: indexing with FAISS.



```python
# Load the Universal Sentence Encoder's TF Hub module
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Function to generate embeddings
def embed_text(text):
    return embed(text).numpy()

# Generate embeddings for each preprocessed document
X_use = np.vstack([embed_text([doc]) for doc in processed_documents])
```

---


# Indexing with FAISS

With our documents now represented as vectors using the Universal Sentence Encoder, the next step is to use FAISS (Facebook AI Similarity Search) for efficient similarity searching.

## Creating a FAISS Index:

- We first determine the dimension of our vectors from `X_use` using `X_use.shape[1]`.
- A FAISS index (`index`) is created specifically for L2 distance (Euclidean distance) using `faiss.IndexFlatL2(dimension)`.
- We add our document vectors to this index with `index.add(X_use)`. This step effectively creates a searchable space for our document vectors.

### Choosing the Right Index:

- In this project, we use `IndexFlatL2` for its simplicity and effectiveness in handling small to medium-sized datasets.
- FAISS offers a variety of indexes tailored for different use cases and dataset sizes. Depending on your specific needs and the complexity of your data, you might consider other indexes for more efficient searching.
- For larger datasets or more advanced use cases, indexes like `IndexIVFFlat`, `IndexIVFPQ`, and others can provide faster search times and reduced memory usage. Explore more at [FAISS indexes wiki](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).



```python
dimension = X_use.shape[1]
index = faiss.IndexFlatL2(dimension)  # Creating a FAISS index
index.add(X_use)  # Adding the document vectors to the index
```

##  Quering with FAISS
### Defining the Search Function:

- The `search` function is designed to find documents that are semantically similar to a given query.
- It preprocesses the query text using the `preprocess_text` function to ensure consistency.
- The query text is then converted to a vector using `embed_text`.
- FAISS performs a search for the nearest neighbors (`k`) to this query vector in our index.
- It returns the distances and indices of these nearest neighbors.

### Executing a Query and Displaying Results:

- We test our search engine with an example query (e.g., "motorcycle").
- The `search` function returns the indices of the documents in the index that are most similar to the query.
- For each result, we display:
   - The ranking of the result (based on distance).
   - The distance value itself, indicating how close the document is to the query.
   - The actual text of the document. We display both the preprocessed and original versions of each document for comparison.

This functionality showcases the practical application of semantic search: retrieving information that is contextually relevant to the query, not just based on keyword matching. The displayed results will give a clear idea of how our semantic search engine interprets and responds to natural language queries.



```python
# Function to perform a query using the Faiss index
def search(query_text, k=5):
    # Preprocess the query text
    preprocessed_query = preprocess_text(query_text)
    # Generate the query vector
    query_vector = embed_text([preprocessed_query])
    # Perform the search
    distances, indices = index.search(query_vector.astype('float32'), k)
    return distances, indices

# Example Query
query_text = "motorcycle"
distances, indices = search(query_text)

# Display the results
for i, idx in enumerate(indices[0]):
    # Ensure that the displayed document is the preprocessed one
    print(f"Rank {i+1}: (Distance: {distances[0][i]})\n{processed_documents[idx]}\n")
```


```python
# Display the results
for i, idx in enumerate(indices[0]):
    # Displaying the original (unprocessed) document corresponding to the search result
    print(f"Rank {i+1}: (Distance: {distances[0][i]})\n{documents[idx]}\n")
```

---


# Congratulations! You have completed the lab


## Authors


[Ashutosh Sagar](https://www.linkedin.com/in/ashutoshsagar/) is completing his MS in CS from Dalhousie University. He has previous experience working with Natural Language Processing and as a Data Scientist.


## Change Log

<details>
    <summary>Click here for the changelog</summary>

|Date (YYYY-MM-DD)|Version|Changed By|Change Description|
|-|-|-|-|
|2024-01-08|0.1|Ashutosh Sagar|SME initial creation|
|2025-07-17|0.2|Steve Ryan|ID review and format fixes|
|2025-07-25|0.3|Steve Ryan|ID fixed TOC and lab title|

</detials>


Copyright © IBM Corporation. All rights reserved.

