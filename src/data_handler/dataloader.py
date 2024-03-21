import os
import pandas as pd
import pickle

from src.data_handler.file_utils import get_data_path, get_cache_path
from src.data_handler.data_downloader import process_raw_downloads


def ensure_data_file_exists(data_file):
    if not os.path.exists(data_file):
        print("Processing raw downloads")
        process_raw_downloads()
        print("Raw Downloads processed")
    return True


def load_raw_papers():
    data_file = get_data_path("arxiv-metadata-ext-paper.csv")
    ensure_data_file_exists(data_file)
    df_papers = pd.read_csv(data_file, dtype={'id': str})
    return df_papers


def load_category():
    data_file = get_data_path("arxiv-metadata-ext-category.csv")
    ensure_data_file_exists(data_file)
    df_categories = pd.read_csv(data_file, dtype={"id": object, "category_id": object})
    return df_categories


def load_version():
    data_file = get_data_path("arxiv-metadata-ext-version.csv")
    ensure_data_file_exists(data_file)
    df_versions = pd.read_csv(data_file, dtype={'id': object})
    return df_versions


def load_taxonomy():
    data_file = get_data_path("arxiv-metadata-ext-taxonomy.csv")
    ensure_data_file_exists(data_file)
    df_taxonomy = pd.read_csv(data_file)
    return df_taxonomy


def load_citations():
    data_file = get_data_path("arxiv-metadata-ext-citation.csv")
    ensure_data_file_exists(data_file)
    df_citations = pd.read_csv(data_file, dtype={"id": object, "id_reference": object})
    return df_citations


def load_raw_cs_papers(clean_suffix=""):
    if clean_suffix:
        clean_suffix = f"-{clean_suffix}"
    data_file = get_data_path(f"arxiv-cs-papers{clean_suffix}.csv")

    if os.path.exists(data_file):
        print(f"Loading cs papers")
        cs_papers = pd.read_csv(data_file, dtype={"id": str})

    else:
        print("Loading raw papers")
        df_papers = load_raw_papers()
        print("Loading categories")
        df_categories = load_category()
        print("Extracting cs papers")
        cs_papers = extract_cs_papers(df_papers, df_categories)
        print("Storing cs papers")
        cs_papers.to_csv(data_file, index=False)

    return cs_papers


def load_cached_tokens(suffix="", base_file="arxiv-cs-papers"):
    base_file = base_file if suffix == "" else f"{base_file}-{suffix}.pickle"
    base_file = get_cache_path(base_file)
    with open(base_file, "rb") as f:
        file = pickle.load(f)
    return file


def extract_cs_papers(df_papers, df_categories):
    cs_papers_id = pd.Series(df_categories[df_categories["category_id"].str.contains(r"\bcs\.[A-Z]{2}\b")]["id"].unique(), name="id")
    # Not necessary if loading with dtype. Trailing cleaning is also not necessary
    # dirty_idx = cs_papers_id.str.contains(r'^0.+$')
    # cs_papers_id[dirty_idx] = cs_papers_id[dirty_idx].str[1:]
    cs_papers = df_papers.merge(cs_papers_id, on="id", how='right')

    return clean_id_trailing_zeros(df_papers, cs_papers, target_col='abstract')


def clean_id_trailing_zeros(full_papers, target_papers, target_col='abstract'):
    new_df = target_papers
    dirty_idx = new_df[target_col].isnull() & new_df['id'].str.contains(r'^[0-9]{3,4}.[0-9]+0$')

    if dirty_idx.sum() == 0:
        return target_papers

    list_of_dfs = [new_df[~dirty_idx]]

    new_df = full_papers.merge(target_papers[dirty_idx].id.str[:-1], on='id', how='right')
    new_df.index = target_papers[dirty_idx].index

    # Target indexes that got parsed to have trailing zeros
    while new_df['abstract'].isnull().sum() > 0:
        dirty_idx =  new_df[target_col].isnull() & new_df.id.str.contains(r'^[0-9]{3,4}.[0-9]+0$')
        if dirty_idx.sum() == 0:
            break
        list_of_dfs.append(new_df[~dirty_idx])
        idxs = new_df[dirty_idx].index
        new_df = full_papers.merge(new_df[dirty_idx].id.str[:-1], on='id', how='right')
        new_df.index = idxs

    list_of_dfs.append(new_df)
    num_dirty = target_papers[target_col].isnull().sum()
    num_cleaned = sum([len(n) for n in list_of_dfs[1:]])

    assert num_dirty == num_cleaned

    return pd.concat(list_of_dfs).sort_index()


if __name__ == "__main__":
    pass
    # print(load_raw_cs_papers())
