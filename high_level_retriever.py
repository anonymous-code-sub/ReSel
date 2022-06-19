import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import STOPWORDS
import math

class BM25:
    def __init__(self, document) -> None:
        self.paragraphs = document.parts

    def remove_stopwords(self, paragraph):
        pruned_paragraph = []
        tokens_without_sw = [word for word in paragraph if not word in STOPWORDS]
        return tokens_without_sw

    def get_top_n(self, n, query):
        tokenized_paragraphs = []
        for paragraph in self.paragraphs:
            tokenized_paragraph = word_tokenize(paragraph)
            tokenized_paragraphs.append(self.remove_stopwords(tokenized_paragraph))
        bm25 = BM25Okapi(tokenized_paragraphs, k1=1.5, b=0.75)
        tokenized_query = word_tokenize(query)
        return bm25.get_top_n(tokenized_query, self.paragraphs, n=n)


class BM25Okapi():
    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.epsilon = epsilon
        self.k1 = k1
        self.b = b
        nd = self._initialize(corpus)
        self._calc_idf(nd)
        

    def _initialize(self, corpus):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word]+=1
                except KeyError:
                    nd[word] = 1
        self.avgdl = num_doc / self.corpus_size
        return nd

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]

class TfIdf():
    def __init__(self, n_to_select=None):
        self.n_to_select = n_to_select
    
    def dists(self, question, paragraphs):
        tfidf = TfidfVectorizer(strip_accents="unicode")
        text = []
        for para in paragraphs:
            text.append(para)
        try:
            para_features = tfidf.fit_transform(text)
            q_features = tfidf.transform([question])
        except ValueError:
            return []
        
        dists = pairwise_distances(q_features, para_features, "cosine").ravel()
        sorted_ix = np.lexsort(([x for x in range(len(paragraphs))], dists))

        if self.n_to_select == None:
            return dists
        else:
            return dists

