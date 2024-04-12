import numpy as np
import pandas as pd
from umap import UMAP
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def get_intertopic_dist(model, topics = None, top_n_topics = None, custom_labels = False):

    # Select topics based on top_n and topics args
    freq_df = model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    # Extract topic words and their frequencies
    topic_list = sorted(topics)
    frequencies = [model.topic_sizes_[topic] for topic in topic_list]
    if isinstance(custom_labels, str):
        words = [[[str(topic), None]] + model.topic_aspects_[custom_labels][topic] for topic in topic_list]
        words = ["_".join([label[0] for label in labels[:4]]) for labels in words]
        words = [label if len(label) < 30 else label[:27] + "..." for label in words]
    elif custom_labels and model.custom_labels_ is not None:
        words = [model.custom_labels_[topic + model._outliers] for topic in topic_list]
    else:
        words = [" | ".join([word[0] for word in model.get_topic(topic)[:5]]) for topic in topic_list]

    # Embed c-TF-IDF into 2D
    all_topics = sorted(list(model.get_topics().keys()))
    indices = np.array([all_topics.index(topic) for topic in topics])

    if model.topic_embeddings_ is not None:
        embeddings = model.topic_embeddings_[indices]
        embeddings = UMAP(n_neighbors=2, n_components=2, metric='cosine', random_state=42).fit_transform(embeddings)
    else:
        embeddings = model.c_tf_idf_.toarray()[indices]
        embeddings = MinMaxScaler().fit_transform(embeddings)
        embeddings = UMAP(n_neighbors=2, n_components=2, metric='hellinger', random_state=42).fit_transform(embeddings)

    dist_df = pd.DataFrame({"x": embeddings[:, 0], "y": embeddings[:, 1],
                        "Topic": topic_list, "Words": words, "Size": frequencies})

    return dist_df

    

def get_cosine_sim(model, n_clusters=20, topics=None, top_n_topics=None):

    # Select topic embeddings
    if model.topic_embeddings_ is not None:
        embeddings = np.array(model.topic_embeddings_)[model._outliers:]
    else:
        embeddings = model.c_tf_idf_[model._outliers:]

    # Select topics based on top_n and topics args
    freq_df = model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    # Order heatmap by similar clusters of topics
    sorted_topics = topics
    if n_clusters:
        if n_clusters >= len(set(topics)):
            raise ValueError("Make sure to set `n_clusters` lower than "
                                "the total number of unique topics.")

        distance_matrix = cosine_similarity(embeddings[topics])
        Z = linkage(distance_matrix, 'ward')
        clusters = fcluster(Z, t=n_clusters, criterion='maxclust')

        # Extract new order of topics
        mapping = {cluster: [] for cluster in clusters}
        for topic, cluster in zip(topics, clusters):
            mapping[cluster].append(topic)
        mapping = [cluster for cluster in mapping.values()]
        sorted_topics = [topic for cluster in mapping for topic in cluster]

    # Select embeddings
    indices = np.array([topics.index(topic) for topic in sorted_topics])
    embeddings = embeddings[indices]
    sim_matrix = cosine_similarity(embeddings)

    return sim_matrix