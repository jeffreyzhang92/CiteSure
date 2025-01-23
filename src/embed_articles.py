articles_dir = '../data'

import torch
from transformers import 
import json
import numpy as np
import sent2vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial import distance
import pandas as pd

from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModel, AutoModelForSequenceClassification, TrainingArguments, Trainer
from llm2vec import LLM2Vec

import argparse


class LLM:
    def __init__(self, normalize=True):
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/llm-embedder')
        self.model = AutoModel.from_pretrained('BAAI/llm-embedder')
        self.model.eval()
        self.normalize = normalize

    def embed(self, sentences):
            encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                # Perform pooling. In this case, cls pooling.
                sentence_embeddings = model_output[0][:, 0]
            if self.normalize:
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings.numpy().tolist()

    def embed_sentences(self, sentences):
        queries = ['Encode this query for searching relevant passages:\n' + sentence for sentence in sentences]
        return self.embed(queries)

    def embed_articles(self, articles):
        queries = ['Represent this document for retrieval:\n' + article for article in articles]
        return self.embed('queries')

class BS2V:
    def __init__(self, normalize=True):
        self.normalize = normalize
        model_path = '***************/BioSentVec_PubMed_MIMICIII-bigram_d700.bin' # change to path with BioSent2Vec weights
        self.model = sent2vec.Sent2vecModel()
        try:
            self.model.load_model(model_path)
        except Exception as e:
            print(e)
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_sentence(self, text):
        text = text.replace('/', ' / ')
        text = text.replace('.-', ' .- ')
        text = text.replace('.', ' . ')
        text = text.replace('\'', ' \' ')
        text = text.lower()
    
        tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in self.stop_words]
        return ' '.join(tokens)

    def embed(self, sentence):
        embedding = self.model.embed_sentence(self.preprocess_sentence(sentence))[0]
        if self.normalize:
            embedding = embedding/np.linalg.norm(embedding)
        return embedding.tolist()

    def embed_sentences(self, sentences):
        ret = []
        for sentence in sentences:
            ret.append(self.embed(sentence))
        return ret

    def embed_articles(self, articles):
        ret = []
        for article in articles:
            ret.append(self.embed(article))
        return ret

class MCPT:
    def __init__(self, normalize=True, mode='aq'):
        self.normalize = normalize
        if 'q' in mode:
            self.qmodel = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
            self.qtokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        if 'a' in mode:
            self.amodel = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")
            self.atokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

    def embed_sentences(self, sentence):
        with torch.no_grad():
            encoded = self.qtokenizer(
                sentence, 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=64,
            )
            embedding = self.qmodel(**encoded).last_hidden_state[:, 0, :]
        if self.normalize:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.numpy().tolist()

    def embed_articles(self, articles):
        articles = [article.split('\n',1) for article in articles]
        with torch.no_grad():
            encoded = self.atokenizer(
                articles, 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=512,
            )
            embedding = self.amodel(**encoded).last_hidden_state[:, 0, :]
        if self.normalize:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.detach().numpy().tolist()

class LLAMA:
    def __init__(self, normalize=True, path = "YBXL/GPTVec"):
        self.normalize = normalize
        self.l2v = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            peft_model_name_or_path= path ,#"McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
        )

    def embed(self, sentences):
        return self.l2v.encode(sentences)

    def embed_sentences(self, sentences):
        instruction = 'Given a sentence, retrieve biomedical papers cited by the sentence.'
        queries = [instruction + sentence for sentence in sentences]
        embedding = self.l2v.encode(queries)
        if self.normalize:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.numpy().tolist()

    def embed_articles(self, articles):
        embedding = self.l2v.encode(articles)
        if self.normalize:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.numpy().tolist()
        
def embed_articles(model, references_DB, art_embeddings, overwrite = False):
    print('Embedding articles\n*******************\n')
    total_lines = 0

    mode = 'w' if overwrite else 'a'
    
    with open(references_DB, 'r') as f:
        with open(art_embeddings, mode) as w:
            for line in f:
                if total_lines % 10000 == 0:
                    print(f'Embedding articles: {total_lines}/255761')
                data = json.loads(line)
                if 'title' not in data or 'abstract' not in data:
                    continue
                text = data['title'] + '\n' + data['abstract']
                w.write(json.dumps({'pmid': data['pmid'], 'embedding': model.embed_articles([text])[0]}))
                w.write('\n')
                total_lines += 1

def main():
    parser = argparse.ArgumentParser(description='--model: select embedding model.\n--articles: name of file with article titles and abstracts.\n--path: For use with LLAMA/LLM2Vec models.')
    parser.add_argument('--model', type=str)
    parser.add_argument('--articles', type=str)
    parser.add_argument('--path', type=str, nargs='?', default='')
    args = parser.parse_args()
    
    references_DB = articles_dir + args.articles
    
    if args.model == 'LLM':
        model = LLM()
    elif 'LLAMA' in args.model:
        if args.path:
            model = LLAMA(path=args.path)
        else:
            model = LLAMA()
    elif args.model == 'MCPT':
        model = MCPT(mode='a')
    elif args.model == 'BS2V':
        model = BS2V()

    art_embeddings = f'../data/{args.model}_{args.articles}.jsonl'
    embed_articles(model, references_DB, art_embeddings, overwrite = True)

if __name__ == '__main__':
    main()
