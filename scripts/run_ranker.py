import torch
from transformers import AutoTokenizer, AutoModel
import json
import numpy as np
import sent2vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial import distance
import pandas as pd

from datasets import load_dataset
from transformers import DataCollatorWithPadding
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from trectools import TrecEval, TrecQrel, TrecRun

import matplotlib.pyplot as plt
import numpy as np

from llm2vec import LLM2Vec

path_to_article_embeddings = "path_to_article_embeddings"
path_to_sentence_embeddings = "path_to_sentence_embeddings"
path_to_sentence_metadata = "path_to_sentence_metadata"
save_path_for_rankings = "save_path_for_rankings"

def load_articles(embeddings_db):
    with open(embeddings_db, 'r') as f:
        line = f.readline()
        data = json.loads(line)
        embed_dim = len(data['embedding'])
    
    embeddings_art = np.empty((255749, embed_dim))
    pmids_art = np.empty((255749,1))
    pmid_inds = dict()

    print('Loading articles\n*******************\n')
    curr_line = 0
    with open(embeddings_db, 'r') as f:
        for line in f:
            data = json.loads(line)
            embeddings_art[curr_line,:] = data['embedding']
            pmids_art[curr_line,:] = data['pmid']
            pmid_inds[data['pmid']] = curr_line
            curr_line += 1

    return embeddings_art, pmids_art, pmid_inds

def get_rankings(embeddings_db, embeddings_st, preload = None):
    with open(embeddings_st, 'r') as f:
        line = f.readline()
        data = json.loads(line)
        embed_dim = len(data['embedding'])
        line_count = sum(1 for _ in enumerate(f, 1))+1

    if preload:
        embeddings_art, pmids_art, pmid_inds = preload
    else:
        embeddings_art, pmids_art, pmid_inds = load_articles(embeddings_db)

    embeddings_sents = np.empty((line_count, embed_dim))
    
    print('Loading sentences\n*******************\n')
    curr_line = 0
    with open(embeddings_st, 'r') as f:
        for line in f:
            if curr_line % 5000 == 0:
                print(curr_line)
            embeddings_sents[curr_line,:] = json.loads(line)['embedding']
            curr_line += 1
    
    print('Computing cosines\n*******************\n')
    cosines = embeddings_sents @ embeddings_art.T

    print('Generating rankings\n*******************\n')
    
    rankings = np.argsort(-cosines)
    ranked_cosines = -np.sort(-cosines)
    pmid_rankings = np.array(np.asmatrix(pmids_art[rankings]).astype(int))
    top20_pmids = np.array(np.asmatrix(pmids_art[rankings[:,:20]]).astype(int))
    
    sentences = pd.read_csv(path_to_sentence_metadata, index_col = 0)

    ## Newer code
    sentences['refs'] = sentences.pmids.apply(lambda x: np.fromstring(x.strip('[]'), dtype=int, sep=' '))
    sentences = sentences.drop(columns = 'pmids')
    
    sentences['rankings'] = None
    sentences['gold_cosines'] = None
    for index, row in sentences.iterrows():
        temp_rankings = []
        temp_cosines = []
        for pmid in row.refs:
            indices = np.where(rankings[index,:] == pmid_inds[pmid])[0][0]
            temp_rankings.append(indices + 1)
            temp_cosines.append(ranked_cosines[index, indices])
        temp_rankings.sort()
        sentences.at[index, 'rankings'] = np.array(temp_rankings) if len(temp_rankings) > 1 else [temp_rankings[0]]
        sentences.at[index, 'gold_cosines'] = -np.sort(-np.array(temp_cosines)) if len(temp_cosines) > 1 else [temp_cosines[0]]
    
    sentences['first_hit'] = sentences.rankings.apply(lambda x: x[0])
    
    sentences['top_20'] = None
    for index, row in sentences.iterrows():
        sentences.at[index, 'top_20'] = top20_pmids[index,:]

    sentences['top_cosine'] = ranked_cosines[:,0]
    
    sentences.to_json(save_path_for_rankings, orient = 'records')

    return ranked_cosines, sentences, pmid_rankings
    
for model_name in models:
    path_to_article_embeddings = f'/gpfs/gibbs/project/xu_hua/shared/citationgpt/citationgpt_datasets/embeddings_db/articles/{model_name}_article_embeddings.jsonl'
    embeddings_art, pmids_art, pmid_inds = load_articles(path_to_article_embeddings)
    
    for sent_count in [1,2,3]:
        path_to_sentence_embeddings = f'/gpfs/gibbs/project/xu_hua/shared/citationgpt/citationgpt_datasets/embeddings_db/sentences/{model_name}/sentence_embeddings_{sent_count}.jsonl'
        save_path_for_rankings = f'/gpfs/gibbs/project/xu_hua/shared/citationgpt/citationgpt_datasets/rankings/{model_name}_rankings_{sent_count}.json'
        path_to_sentence_metadata = f'/gpfs/gibbs/project/xu_hua/shared/citationgpt/citationgpt_datasets/test_sentences/test_sentences_{sent_count}.csv'
    
        cosines, sentences, pmid_rankings = get_rankings(path_to_article_embeddings, path_to_sentence_embeddings, preload = (embeddings_art, pmids_art, pmid_inds))
