import pandas as pd
from tqdm import tqdm

from src import text_cleaning

class Preprocessor:
    def __init__(self, to_exclude=[], to_include=None, **pipeline_kwargs_dict):
        self.pipeline_fn = {
            "clean": self.clean_input,
            "abbv": self.remove_abbv,
            "case-fold": self.case_folding,
            "tokenize": self.tokenize
        }

        self.preprocessor_pipeline = ["clean", "abbv", "case-fold", "tokenize"]

        if to_include is not None:
            to_exclude = self.preprocessor_pipeline
            self.preprocessor_pipeline = to_include
        else:
            self.preprocessor_pipeline = [preprocess for preprocess in self.preprocessor_pipeline if
                                          preprocess not in to_exclude]

        self.pipeline_kwargs_dict = pipeline_kwargs_dict

    def execute(self, df, columns_interested=['abstract']):
        for preprocess in self.preprocessor_pipeline:
            if preprocess in self.pipeline_kwargs_dict:
                kwargs = self.pipeline_kwargs_dict[preprocess]
            else:
                kwargs = {}
            preprocess = self.pipeline_fn[preprocess]

            df = preprocess(df, columns_interested, **kwargs)

        return df

    def clean_input(self, df, columns_interested=['abstract']):
        df = df.copy(deep=True)
        for col in columns_interested:
            df[col] = text_cleaning.preprocess(df[col])
        return df

    def remove_abbv(self, df, columns_interested=['abstract'], **kwargs):
        df = df.copy(deep=True)

        nlp = text_cleaning.prepare_spacy_nlp()
        pipes_to_disable = ['tagger', 'parser', 'ner', 'lemmatizer', 'textcat', 'tok2vec', 'attribute_ruler']

        for col in columns_interested:
            tokenized_abstracts = nlp.pipe(df[col], disable=pipes_to_disable)

            disambiguated_text = []

            for doc in tqdm(tokenized_abstracts, total=df.shape[0]):

                text = text_cleaning.abbreviation_expansion_disambiguation(doc)
                disambiguated_text.append(text)

            df[col] = pd.Series(disambiguated_text)

        return df

    def case_folding(self, df, columns_interested=['abstract'], **kwargs):
        df = df.copy(deep=True)
        for col in columns_interested:
            df[col] = df[col].str.lower()
        return df

    def tokenize(self, df, columns_interested=['abstract'], **kwargs):
        nlp = text_cleaning.prepare_spacy_nlp()
        pipes_to_disable = ['tagger', 'parser', 'ner', 'lemmatizer', 'textcat', 'tok2vec', 'attribute_ruler', 'abbreviation_detector']
        output_tokens = {}
        for col in columns_interested:
            tokenized_abstracts = nlp.pipe(df[col], disable=pipes_to_disable)
            tokens = []
            
            for doc in tqdm(tokenized_abstracts, total=df.shape[0]):
                tokens_ = [token.text for token in doc]
                tokens.append(tokens_)

            output_tokens[col] = tokens

        return output_tokens