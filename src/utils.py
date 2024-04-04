import numpy as np


def vector_norm(X):
    """
    Applies L2 norm onto each vector in the matrix.
    """
    return X / np.sqrt(np.einsum('ij, ij -> i', X, X))[:, None]

def cosine_sim(a, b=None):
    """
    Obtains pairwise cosine sim between vectors by taking the dot product between unit vectors.

    If need to find pairwise cosine sim between vectors of the same matrix, leave b empty.

    Expects a: [batch_size_1, embed_dim], b: [batch_size_2, embed_dim] where the embed_dim is the same.
    """
    a = vector_norm(a)
    b = a if b is None else vector_norm(b)
    return np.einsum('ik, jk -> ij', a, b)

def euclidean_dist(a, b=None):
    """
    Obtains a pairwise euclidean distance between vectors by projecting the difference into a third dimension before using dot product to square and sum the difference.
    """
    b = a if b is None else b
    # Broadcasting to learn the diff for each (X, Y) pairing
    diff = a[:,:,None] - b.T
    # Take L2 norm to learn np.sqrt(np.sum(sqr(diff))).
    distances = np.sqrt(np.einsum('ijk, ijk->ik', diff, diff))
    return distances

def top_k_retrieval(pairwise_sim, top_k=None, return_score=False, ignore_self_similarity=True):
    """
    Performs retrieval based on a pairwise similarity matrix. Expects the similarity to be higher -> more similar.
    Returns a list of indexes where the list of indexes are indexes to the papers most similar. The index of the returned list is the index of the paper itself.
    """
    top_papers_idx = []
    pairwise_sim = pairwise_sim.copy()
    for idx, paper in enumerate(pairwise_sim):
        if ignore_self_similarity:
            paper[idx] = np.Inf
        top_paper_idx = np.argsort(-paper)
        if top_k is not None:
            top_paper_idx = top_paper_idx[:top_k]
        if return_score:
            sim_scores = paper[top_paper_idx]
            top_paper_idx = np.concatenate((top_paper_idx[:,None], sim_scores[:,None]))
        
        top_papers_idx.append(top_paper_idx)
    return top_papers_idx

if __name__ == "__main__":
    a, b = np.random.randn(64, 512), np.random.randn(64, 512)
    print(cosine_sim(a, b))
    print(cosine_sim(a))