def get_paper_citations_made(citations, count=False):
    """
    A dictionary is returned where the keys are the id of the paper and the values are a list of the papers that it cites.
    """
    return {paper_id: len([id for id in df_papers_cited.id_reference]) if count else [id for id in df_papers_cited.id_reference] for paper_id, df_papers_cited in citations.groupby('id')}

def get_paper_cited(citations, count=False):
    """
    A dictionary is returned where the keys are the id of the paper and the values are a list of the papers that cites it.
    """
    return {id_referenced: len([id for id in df.id]) if count else [id for id in df.id] for id_referenced, df in citations.groupby('id_reference')}

def filter_citations_to_subset(paper_subset, citations):
    """
    Filters citations only to include those that papers in the subset are in both the paper citing and the paper cited column.

    Used for counting when 
    """
    
    return citations.merge(paper_subset, how='inner', on='id').merge(paper_subset, how='inner', left_on='id_reference', right_on='id', suffixes=("", "_y")).iloc[:,:2]