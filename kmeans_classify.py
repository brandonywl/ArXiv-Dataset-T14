import nltk, pickle, string, re, os 
from nltk.tokenize import word_tokenize #Used to extract words from documents
from nltk.stem import WordNetLemmatizer #Used to lemmatize words
from nltk.corpus import stopwords
import matplotlib.pyplot as plt 
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn import metrics
from wordcloud import WordCloud
import numpy as np


# The function clean_text is used to clean up the user's input, remove stop words, etc.
def clean_text(text, lemmatizer, tokenizer, stopwords):
    text = str(text).lower()  # Lowercase words
    text = re.sub(r"\[(.*?)\]", "", text)  # Remove [+XYZ chars] in content
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
    text = re.sub(r"\w+…|…", "", text)  # Remove ellipsis (and last word)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Replace dash between words
    text = re.sub(
        f"[{re.escape(string.punctuation)}]", "", text
    )  # Remove punctuation

    tokens = tokenizer(text)  # Get tokens from text
    tokens = [t for t in tokens if not t in stopwords]  # Remove stopwords
    tokens = ["" if t.isdigit() else t for t in tokens]  # Remove digits
    tokens = [t for t in tokens if len(t) > 1]  # Remove short tokens

    # lemmatization
    lemmatized_doc = ""
    for token in tokens:
      if token not in stopwords:
        lemmatized_doc = lemmatized_doc + " " + lemmatizer.lemmatize(token)
    return lemmatized_doc

# save model or vectorizer or any processed data into pickle file
def save_model_to_pickle(model_name, file_name):
    project_path = os.getcwd()
    model_data_path = project_path + "/model_data"

    if not os.path.exists(model_data_path):
        os.makedirs(model_data_path)
    pickle.dump(model_name, open(os.getcwd() + "/" + file_name, "wb"))

# load any model or pre-trained info from pickle file
def load_from_pickle(file_name):
    file_name = os.getcwd() + "/" + file_name

    if not os.path.exists(file_name):
        print("file {} does not exist".format(file_name))
        return None
    else:
        file = open(file_name, "rb")
        data = pickle.load(file)
        file.close()
        return data

# concatenate data_df, df_categories and df_taxonomy
# add field "year" based on arXiv naming convension
# remove stop words to faciliate extracting key info. if input parameter remove_stop_words = False, then skip removing_stop_words, the function could run mush faster
def concatenate_data(data_df, df_categories, df_taxonomy, remove_stop_words = False):
    data = data_df.merge(df_categories, how = "left", on="id")
    data = data.merge(df_taxonomy, how='left', on='category_id')

    def get_year(x):
        # based on the rules on arXiv website: https://info.arxiv.org/help/arxiv_identifier.html
        match = re.search(r"\w+(\.|-)\w+/(\d{2})\d+", x)
        if match:
            if match[2][0] == "9":
                return int("19" + match[2])
            else:
                return int("20" + match[2])
        else:
            match =  re.search(r"(\d{2})\d{2}\.\d+", x)
            if match:
                return int("20" + match[1])
            else:
                # could not find the matching pattern, return the original format
                return 0
    # removing stop words need 10+ mins.
    if remove_stop_words:    
        nltk.download('stopwords')
        stop_words = stopwords.words('english')
        def remove_stopwords(x):
            tokens = word_tokenize(x)  # Get tokens from text
            tokens = [t for t in tokens if not t in stop_words]  
            return ' '.join(tokens)

        data['abstract'] = data['abstract'].apply(remove_stopwords)

    data['year'] = data['id'].apply(get_year)
    data.year = data.year.astype("int")
    # some paper's category_name is NA
    data.category_name.fillna("NA", inplace=True)

    data.to_csv(os.getcwd() + "/model_data/data_df.csv")
    return data

# get average of each word vector to represent the doc
def vectorize_single_doc(tokenized_doc, model):

  vectors = []
  for token in tokenized_doc:
      # if this token belongs to the word dimension of the model
      if token in model.wv:
          try:
              # get the word vector name
              vectors.append(model.wv[token])
          except KeyError:
              continue
  # if the doc has one or more tokens being found in the dimension of word features, convert them to narray, and calculate the mean
  if vectors:
      vectors = np.asarray(vectors)
      avg_vec = vectors.mean(axis=0)
      return avg_vec
  else:
    # the dimension of word2vec in the model
      return np.zeros(model.vector_size)

# vectorize all the docs
def vectorize(list_of_docs, model):
    """Generate vectors for list of documents using a Word Embedding"""
    features = []
    for index, doc in enumerate(list_of_docs):
        if index % 5000 == 0:
          print("Starting vectorizing docs {}-{}".format(index, index+5000))
        # if the doc has one or more tokens being found in the dimension of word features, convert them to narray, and calculate the mean
        avg_vec = vectorize_single_doc(doc, model)
        features.append(avg_vec)
    print("Finish vectorizing all the docs.")
    return features


# clean up user's input and vectorize the input doc as a vector
def generate_vector_for_user_input(user_input, vectorizer):
  processed_token = set(clean_text(user_input, WordNetLemmatizer(), word_tokenize, stopwords.words('english')).split())
  new_vector = vectorize_single_doc(processed_token, vectorizer)
  return new_vector

# This is a wrap function to get top cited papers, keywords from a specified Kmeans cluster
# km_model  - KMeans model
# data_df   - the dataframe contains the paper info
# word_vectorizer - here we use word2vec vectorizer
# vectorized_doc  - a list of list. Each sub list represents a vectorized doc
# cluster_num - the Kmeans cluster number
# centroid_paper_num - how many papers to pick up
# top_key_words - how many keywords to pick up
# return: the selected paper ids, the selected keywords
def centroids_papers_keywords_in_clusters(km_model, data_df, word_vectorizer, vectorized_docs, cluster_num, centroid_paper_num = 10, top_key_words = 10):

    # find the most centric word vectors in each K-Means cluster as the most representative of the cluster
    centroids_tokens = []
    most_representative = word_vectorizer.wv.most_similar(positive =[km_model.cluster_centers_[cluster_num]], topn = top_key_words)
    for t in most_representative:
        centroids_tokens.append(t[0])

    # sorting based on the vector distance toward to the center of the cluster
    most_representative_docs_index = np.argsort(
        np.linalg.norm(vectorized_docs - km_model.cluster_centers_[cluster_num], axis=1)
    )

    # find the papers closest to the center of the the cluster
    print("\n-------------------------------------------")
    print("Top {} papers closest to the center of Cluster {}".format(centroid_paper_num, cluster_num))
    centroid_paper_ids = most_representative_docs_index[: centroid_paper_num]

    records = data_df.iloc[most_representative_docs_index[:centroid_paper_num], :][['id', 'title']].reset_index(drop=True)
    print(records)

    print("Top {} keywords closest to the center of Cluster {}\n{}".format(top_key_words, cluster_num, centroids_tokens))
    return records.id, centroids_tokens

# since paper might cover multiple categories, when visualizing the data,
# show this limitation. 
def view_top_category_paper_amount(data_df, top_category_num, begin_year=1991, end_year=2021):
    if begin_year == 2000 or end_year == 2021:
        stat_data = data_df.groupby("category_name").size().sort_values(ascending=False).reset_index(name='counts').head(top_category_num)
    else:
        stat_data = data_df[(data_df['year'] >=begin_year) & (data_df['year']<=end_year)].groupby("category_name").size().sort_values(ascending=False).reset_index(name='counts').head(top_category_num)

    category_name = list(stat_data['category_name'])

    # draw the diagram
    my_cmap = plt.get_cmap("plasma")
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
    fig, ax = plt.subplots()
    ax.barh(stat_data.category_name, stat_data.counts, color=my_cmap(rescale(stat_data.counts)))

    ax.invert_yaxis()
    ax.set_ylabel("Paper Count")
    ax.set_title("Top {} Paper Amount from year {} to {}".format(top_category_num, begin_year, end_year))
    plt.xticks(rotation=90)


    multi_category_info = data_df.groupby("id")['category_name'].size().reset_index(name='counts')
    multi_category_info = multi_category_info.groupby("counts").size().reset_index()
    multi_category_info.columns = ['category_per_paper', "count"]
    
    plt.figure()
    plt.bar(multi_category_info['category_per_paper'], multi_category_info['count'], color='#6890F0')
    plt.xlabel("Multi-categories")
    plt.ylabel("Paper Amount")
    plt.title("Papers Have Multiple Categories")
    plt.show()

    return category_name

# this function wraps up the operations of word vectorization and doc vectorization. Return values:
# word_vectorizer - word2Vec object
# vectorized_doc - list of list, each sub list represents an abstract's vector
def generate_word_and_doc_vectors(data_df):
    # use word2vec to tokenize the whole abstract field and save to pickle
    # Note: running this step needs 30+ mins.
    tokenized_docs = data_df.abstract.str.split()
    word_vectorizer = Word2Vec(sentences=tokenized_docs, vector_size=300, workers=1, seed=42)
    print("Generating word vectors... Take 30+ mins.")
    save_model_to_pickle(word_vectorizer, "model_data/word2vec_model.pkl")

    # vectorize each document into ONE representative vector and save to pickle
    # this step needs 10+ mins.
    vectorized_docs = vectorize(tokenized_docs, model=word_vectorizer)
    print("vectorized_docs shape: samples - {}, each sample dimension - {}".format(len(vectorized_docs), len(vectorized_docs[0])))
    save_model_to_pickle(vectorized_docs, "model_data/vectorized_doc.pkl")

    return word_vectorizer, vectorized_docs 

# Use KMeans to classify the vectorized data. Save the model to pickle
def fit_kmean_model(cluster_num, vectorized_docs):
    km = KMeans(n_clusters=cluster_num, init='k-means++', max_iter=200)
    print("Start fit into Kmeans model. Make take several minutes.")
    km.fit(np.array(vectorized_docs))
    save_model_to_pickle(km, "model_data/km_model.pkl")
    return km

# Measure KMean result. Please note this result is not accurate, since each
# paper cover one or several categories.
def check_kmean_efficiency(km_model, data_df, vectorized_docs):
    ground_truth_category = data_df.category_name
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(ground_truth_category, km_model.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(ground_truth_category, km_model.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(ground_truth_category, km_model.labels_))
    print("Adjusted Rand-Index: %.3f"
        % metrics.adjusted_rand_score(ground_truth_category, km_model.labels_))
    print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(vectorized_docs, km_model.labels_, sample_size=5000))

# based on citation number, get the top influential papers in specified cluster
def top_influencial_in_cluster(data_df, citation_df, km_model, cluster_num, top_influential_paper_num):

    cluster_paper_ids = list(data_df[km_model.labels_ == cluster_num]['id'])
    df_papers = citation_df.groupby("id_reference").count()
    df_papers = df_papers.reset_index()
    df_papers.columns = ['id', 'citation_count']
    df_result = df_papers[df_papers.id.isin(cluster_paper_ids)]

    if len(df_result) == 0:
        return "no reference for any of those papers in this cluster."
    else:
        papers_id = df_result.sort_values("citation_count", ascending = False).id.values[:top_influential_paper_num]
        print("\nCluster {}'s top {} most cited papers:".format(cluster_num, top_influential_paper_num))
        
        top_cited_papers = data_df[data_df.id.isin(papers_id)][['id', 'title']].reset_index(drop=True)
        print(top_cited_papers)

# create word cloud for the keywords most centric to the cluster
# km_model - the Kmean model object
# word_vectorizer - the word2vec object, or most of time, the program loads the pre_trained word2vec model from pickle file
# cluster_index: the cluster number in the KMeans model
# top_words: the word number to include in the word cloud
def centric_words_cloud(km_model, word_vectorizer, cluster_index, top_words=20):
    if cluster_index >= len(set(km_model.labels_)):
        print("cluster_index is beyond the boundary.")
        return 
    
    centric_words = {}
    most_representative = word_vectorizer.wv.most_similar(topn=top_words, positive=[km_model.cluster_centers_[cluster_index]])
    # build the word cloud dictionary
    for t in most_representative:
        centric_words[t[0]] = t[1]

    makeImage(centric_words)
    return centric_words

# this function is called by centric_words_cloud function to show the image
def makeImage(word_dict):
    wc = WordCloud(background_color="white", max_words=50)
    wc.generate_from_frequencies(word_dict)
    # show
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# this function is a wrap function, which calls several other functions to show one specific cluster's information, including top centric keywords, centric papers, top cited papers, and word cloud for the top keywords
def report_cluster_info(cluster_num, word_vectorizer, km_model, data_df, df_citations, vectorized_docs):
    if cluster_num >= len(set(km_model.labels_)):
        print("cluster_num is beyond the boundary.")
        return
    
    centroids_papers_keywords_in_clusters(km_model, data_df, word_vectorizer, vectorized_docs, cluster_num, centroid_paper_num=5, top_key_words = 10)
    top_influencial_in_cluster(data_df, df_citations, km_model, cluster_num, 10)
    centric_words_cloud(km_model, word_vectorizer, cluster_num, top_words=10)

# this function vectorize the user input and then utilize KMeans.predict to
# find the closest cluster to the user query. Then it calls report_cluster_info to show the chosen cluster's information
def predict_user_query_cluster(user_input, word_vectorizer, km_model, data_df, df_citations, vectorized_docs):
    # get user unput vector
    user_input_vector = generate_vector_for_user_input(user_input, word_vectorizer)
    
    label_ = km_model.predict(np.array(user_input_vector).reshape(1, -1).astype('float'))
    cluster_num = label_[0]
    print("this input is closest to this cluster {}".format(cluster_num))
    report_cluster_info(cluster_num, word_vectorizer, km_model, data_df, df_citations, vectorized_docs)
