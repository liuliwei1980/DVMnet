import math
import pickle

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from itertools import product
from collections import defaultdict
def Gaussian(x):
    return math.exp(-0.5*(x*x))
def paired(x,y):
    if x == 'A' and y == 'U':
        return 2
    elif x == 'G' and y == 'C':
        return 3
    elif x == "G"and y == 'U':
        return 0.8
    elif x == 'U' and y == 'A':
        return 2
    elif x == 'C' and y == 'G':
        return 3
    elif x == "U"and y == 'G':
        return 0.8
    else:
        return 0
def generate_kmers(k, bases=['A', 'C', 'G', 'T']):
    return sorted(''.join(kmer) for kmer in product(bases, repeat=k))
def build_de_bruijn_graph(sequence, k):
    kmers = generate_kmers(k)
    kmer_to_index = {kmer: index for index, kmer in enumerate(kmers)}
    adj_matrix = np.zeros((len(kmers), len(kmers)), dtype=int)
    edges = defaultdict(list)
    for i in range(len(sequence) - k + 1):
        source_kmer = sequence[i:i + k]
        for j in range(len(sequence) - k + 1):
            target_kmer = sequence[j:j + k]
            coefficient = 0
            for add in range(30):
                if i - add >= 0 and j + add < len(sequence):
                    score = paired(sequence[i - add].replace('T', 'U'), sequence[j + add].replace('T', 'U'))
                    if score == 0:
                        break
                    else:
                        coefficient = coefficient + score * Gaussian(add)
                        edges[source_kmer].append((target_kmer, coefficient))
                else:
                    break
    for source_kmer, targets_weights in edges.items():
        source_index = kmer_to_index[source_kmer]
        for target_kmer, weight in targets_weights:
            target_index = kmer_to_index[target_kmer]
            adj_matrix[source_index, target_index] += weight

    return adj_matrix, kmer_to_index, kmers
def create_tagged_documents(seq, name):
    tagged_docs = [TaggedDocument(seq[i], name[i])
                   for i in range(len(name))]

    return tagged_docs

def de_bruijn_graph(seq,name,k):
    tagged_docs_lnc = create_tagged_documents(seq, name)
    vectors = {}
    i=0
    for sequence in tagged_docs_lnc:
        print(i)
        i=i+1
        seq = sequence.words
        if len(seq) > 3000:
             seq = seq[:3000]
        seq = seq.replace('U', 'T')
        adj_matrix, kmer_to_index, kmers = build_de_bruijn_graph(seq, k)
        out_degree = np.sum(adj_matrix, axis=1)
        out_degree_inv = 1.0 / out_degree
        out_degree_inv[np.isinf(out_degree_inv)] = 0
        out_degree_matrix_inv = np.diag(out_degree_inv)
        norm_adj_matrix_left = np.matmul(out_degree_matrix_inv, adj_matrix)
        vectors[sequence.tags] = norm_adj_matrix_left
    with open('vectors_lnc.pkl', 'wb') as f:
        pickle.dump(vectors, f)
    return vectors
