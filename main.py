import os
import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
from bertopic import BERTopic
import plotly.graph_objects as go
from sklearn.metrics.pairwise import pairwise_distances
from src.topic_similarity import get_intertopic_dist
import matplotlib.pyplot as plt
from src.citations.analytics import get_paper_weights_and_edges, sort_paper_importance, visualize_subset
from src.data_handler.dataloader import load_citations, load_raw_papers, load_taxonomy, load_category, load_raw_cs_papers, load_version
from get_clean_text import load_cs_papers
from src.citations.utils import filter_citations_to_subset

st.header('Google Scholar (at home)')

model_set = set(os.listdir('models'))

target_columns = [
            'id',
            'group_name',            
            'year',
            'title',
            'authors',
            'citation_count',
            'journal-ref',
            'category_name',
            'abstract',
            'Name'
        ]

@st.cache_data
def prep_df() -> pd.DataFrame:
    # category_file = 'data/arxiv-metadata-ext-category.csv'
    # citation_file = 'data/arxiv-metadata-ext-citation.csv'
    # taxonomy_file = 'data/arxiv-metadata-ext-taxonomy.csv'
    # version_file = 'data/arxiv-metadata-ext-version.csv'

    # category_df = pd.read_csv(category_file, dtype={"id":object})
    # citation_df = pd.read_csv(citation_file, dtype={"id":object})
    # citation_count_df = citation_df.groupby('id_reference')\
    #                                 .count().reset_index()\
    #                                 .rename(columns={'id_reference':'id', 'id':'citation_count'})\
    #                                 .sort_values(by='citation_count', ascending=False)
    # taxonomy_df = pd.read_csv(taxonomy_file)
    # version_df = pd.read_csv(version_file, dtype={"id":object})
    # abstract_df = pd.read_csv('data/arxiv-cs-papers-clean_abbv_casefold_punct.csv', dtype={"id":object})

    # df = citation_df.merge(category_df,on="id")\
    #             .drop(columns=['id_reference'], axis=1)\
    #             .merge(taxonomy_df,on="category_id").drop_duplicates(["id","group_name"])\
    #             .query('`group_name`=="Computer Science"')\
    #             .merge(version_df[["id","year"]], on ="id")\
    #             .merge(abstract_df, on="id")\
    #             .merge(citation_count_df, how='left', on='id')
                
    # df['citation_count'] = df['citation_count'].fillna(0)    
    df = pd.read_pickle('data/df.pickle')
    return df

def prep_citation_network():
    papers = load_cs_papers("clean_abbv_casefold_punct", run_preprocessor=False)
    citations = load_citations()
    return papers,citations

def add_topic(df:pd.DataFrame, model) -> pd.DataFrame:
    df['Topic'] = model.topics_
    freq = model.get_topic_info()
    df = pd.merge(df, freq[['Topic', 'Name']], how='left', on='Topic')
    return df

base_df = prep_df()
papers, citations = prep_citation_network()

with st.form("input_form"):
    model_name = st.selectbox(
        'Select saved model to load:',
        model_set
    )

    paper_name = st.selectbox(
        'Select arXiv paper to analyse:',
        set(base_df['title'].to_list()),
        index=None,
        placeholder='Enter arXiv research paper name'
    )

    pagerank_method = st.selectbox(
        'Select PageRank method to use:',
        ('paperrank', 'total'),
        index=None
    )

    submitted = st.form_submit_button("Submit")

## App layout
tab_list = [
    "Overview", 
    "Similar Topics", 
    "PageRank"
]
overview_tab, sim_topic_tab, pagerank_tab = st.tabs(tab_list)

if submitted:
    # model = BERTopic.load(f"models/bertopic - 2024:03:25:00:43:23")
    model = BERTopic.load(f"models/{model_name}")
    df = add_topic(base_df, model)

with overview_tab:
    if submitted:
        query = df.query(f'`title`=="{paper_name}"')
        topic_num = query['Topic'].iloc[0]
        st.subheader(f'Topic identified:')
        st.write(query["Name"])
        query = query[target_columns]
        st.write(query)

        st.subheader(f'Top 5 papers recommendations from identified topic:')
        paper_rank_df = df.query(f'`title`!="{paper_name}"')
        paper_rank_df = paper_rank_df.query(f'`Topic`=={topic_num}')\
                            .sort_values(['citation_count'], ascending=False).head()
        paper_rank_df = paper_rank_df[target_columns]
        
        st.write(paper_rank_df)

with sim_topic_tab:
    if submitted:
        st.subheader(f'Similar topics & papers you might be interested in:')
        intertopic_dist = get_intertopic_dist(model, topics = None, top_n_topics = None, custom_labels = False)
        
        pairwise_dist = pairwise_distances(intertopic_dist[['x','y']].to_numpy())
        ind = np.argpartition(pairwise_dist[0], 6)[:6] # top 6 argument of smallest distance
        other_topic = np.delete(ind, [0]).tolist() # remove 0-th element as it is the pairwise distance of topic against itself

        sim_paper_rank = df.query('Topic.isin(@other_topic)', engine="python")\
                            .sort_values(['citation_count'], ascending=False).head()
        sim_paper_rank = sim_paper_rank[target_columns]
        
        st.write(sim_paper_rank)

with pagerank_tab:
    if submitted:
        paper_weight_edges = get_paper_weights_and_edges(papers, citations, pagerank_method)
        top_k = sort_paper_importance(paper_weight_edges, 10)

        id_cache = []
        score_cache = []
        for paper_id, paper_score in top_k:
            id_cache.append(paper_id)
            score_cache.append(paper_score)

        d = {'Paper ID':id_cache, 'Score':score_cache}
        st.write(pd.DataFrame(data=d))