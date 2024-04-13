import networkx as nx
import numpy as np

from src.citations.utils import filter_citations_to_subset, get_paper_cited, get_paper_citations_made


def get_paper_weights_and_edges(paper_subset, citations, weight_type='total'):
    """
    Given a set of papers, we return a dictionary computing the weight of each nodes convenient for ranking and visualization algorithms.

    total - Includes citations outside the subset and even outside the sub-field (Computer Science)
    subset - Only includes citations within the subset specified
    pagerank - Applies subset first. Citations are treated as unweighted directed edges in the graph and PageRank is ran to retrieve the weight of nodes.
    """

    # weight_type = 'total' # Subset, PaperRank
    assert weight_type in ('total', 'subset', 'paperrank'), f"Only accepting 'total', 'subset', and 'paperrank'. You keyed in '{weight_type}'."
    
    weights = []

    # Filter citations to get weight by count in the subset / pagerank
    if weight_type != 'total':
        citations = filter_citations_to_subset(paper_subset, citations)
    
    references_of_paper = get_paper_cited(citations)

    # For paperrank, just use networkx to compute the weights
    if weight_type != 'paperrank':
        for row_idx, row_series in paper_subset.iterrows():
            if row_series['id'] in references_of_paper:
                weight = len(references_of_paper[row_series['id']])
            else:
                weight = 0
            weights.append(weight)

    incoming_edges = [references_of_paper[id] if id in references_of_paper else [] for id in paper_subset.id]
    
    # Filter citations to get relevant edges
    if weight_type == 'total':
        citations = filter_citations_to_subset(paper_subset, citations)
    
    paper_references = get_paper_citations_made(citations)
    outgoing_edges = [paper_references[id] if id in paper_references else [] for id in paper_subset.id]

    if weight_type != 'paperrank':
        edge_weights = [1 for _ in paper_subset.id]
    else:
        dg = nx.DiGraph()

        paper_to_idx = {paper: idx for idx, paper in enumerate(paper_subset.id)}
        idx_to_paper = {idx: paper for paper, idx in paper_to_idx.items()}
        
        graph_nodes = paper_subset.id
        graph_edges = [(idx_to_paper[paper], citation) for paper, citations in enumerate(outgoing_edges) for citation in citations]

        dg.add_nodes_from(graph_nodes)
        dg.add_edges_from(graph_edges)

        pagerank = nx.algorithms.link_analysis.pagerank(dg)

        weights = [pagerank[paper] for paper in graph_nodes]
        edge_weights = [pagerank[paper] / len(outgoing_edges[paper_to_idx[paper]]) if len(outgoing_edges[paper_to_idx[paper]]) > 0 else 0 for paper in graph_nodes]
        

    return {
        id:
        {
            'weight': weights[idx],
            'incoming_edges': incoming_edges[idx],
            'weight_type': weight_type,
            'edges': outgoing_edges[idx],
            'edge_weight': edge_weights[idx]
        }
        for idx, id in enumerate(paper_subset.id)
    }


def sort_paper_importance(paper_weights_edges, top_k: int=None, return_weights:bool =True, descending:bool =True):
    """
    Performs retrieval on a list of inputs. Expect the format to be [(id, ..., weight), ...], where weight is the last index in one sample. 
    Can be used for similarity-based retrieval also. Just input in the same format.
    """
    weight_data = [(paper, data['weight']) for paper, data in paper_weights_edges.items()]
    sort_dir = -1 if descending else 1
    sorted_weight_data = sorted(weight_data, key=lambda x: sort_dir * x[-1])
    if top_k is not None:
        sorted_weight_data = sorted_weight_data[:top_k]

    if not return_weights:
        return list(zip(*sorted_weight_data))[0]
    return sorted_weight_data


def visualize_subset(paper_weight_edges, size_range=[100, 200], scaler='standard'):
    weights =  np.array([data['weight'] for paper, data in paper_weight_edges.items()])
    min_weight = np.min(weights)
    max_weight = np.max(weights)
    range_weight = max_weight - min_weight
    range_size = size_range[1] - size_range[0]

    if range_weight == 0 :
        size_to_put = np.mean(size_range)
        rescaled_sizes = size_to_put

    else:
        # Run Standard Scaling and apply size range to get paper node size
        scaled_weights = (weights - min_weight) / range_weight
        rescaled_sizes = scaled_weights * range_size + size_range[0]

    paper_weight = {
        paper: {"size": rescaled_sizes[idx]}
        for idx, paper in enumerate(paper_weight_edges.keys())
    }
    # Automatically use edge weights as weights
    edge_with_weights = [
        (paper, out_paper, data['edge_weight'])
        for paper, data in paper_weight_edges.items()
        for out_paper in data['edges']
    ]
    g = nx.DiGraph()

    g.add_nodes_from(paper_weight)
    g.add_weighted_edges_from(edge_with_weights)

    return g