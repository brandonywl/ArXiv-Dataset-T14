import numpy as np
import pandas as pd
from umap import UMAP
from datetime import datetime
from bertopic import BERTopic
# from cuml.cluster import HDBSCAN
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired

def main(input_filepath):
    "Loads input data file, runs BERTopic pipeline and saves model"
    df = pd.read_csv(input_filepath)
    abstract_list = df['abstract'].to_list()

    # BERTopic Pipeline
    # Step 1 - Extract embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 2 - Reduce dimensionality
    umap_model = UMAP(low_memory=True)

    # Step 3 - Cluster reduced embeddings
    # hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True) # default
    # hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True, prediction_data=True)
    hdbscan_model = KMeans(n_clusters=50)

    # Step 4 - Tokenize topics
    vectorizer_model = CountVectorizer(stop_words="english")

    # Step 5 - Create topic representation
    ctfidf_model = ClassTfidfTransformer()

    # Step 6 - (Optional) Fine-tune topic representations with a `bertopic.representation` model
    representation_model = KeyBERTInspired()

    topic_model = BERTopic(
        language="english", 
        calculate_probabilities=True,
        verbose=True,
        embedding_model=embedding_model,          # Step 1 - Extract embeddings
        umap_model=umap_model,                    # Step 2 - Reduce dimensionality
        hdbscan_model=hdbscan_model,              # Step 3 - Cluster reduced embeddings
        vectorizer_model=vectorizer_model,        # Step 4 - Tokenize topics
        ctfidf_model=ctfidf_model,                # Step 5 - Extract topic words
        representation_model=representation_model # Step 6 - (Optional) Fine-tune topic represenations
    )

    topics, probs = topic_model.fit_transform(abstract_list)

    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    topic_model.save(f"models/bertopic - {date_time}")
    return

if __name__ == "__main__":
    main('data/arxiv-cs-papers.csv')