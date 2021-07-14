import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

# Read the file
with open('sentences.txt', 'r') as file:
    file = file.read().lower()
    file = re.sub(r'[^\w\s]','',file)

# Break into sentences
corpus = file.split('\n')
print(corpus[:10])

# TFIDF vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,10), analyzer='char_wb')

# Fit the model
model = vectorizer.fit_transform(corpus)
#print(vectorizer.get_feature_names()[:100])
#print(model.shape), print((model.A)), print((model.A).shape)

# K-best suggestions for given query
def k_best(query,model=model, corpus=corpus, k=5):
    # Check if k out of bound of corpus size
    if k+1 > model.shape[0]:
        return f"Index Error: index {k+1} out of bound for corpus size {model.shape[0]}"

    # Preprocess the query
    query = re.sub(r'[^\w\s]','',query)

    # Convert query to embedding
    test = vectorizer.transform([query])

    # Distance of input string with all documents <--- (Cosine Similarity)
    D = np.matmul(model.A, (test.A).T).ravel()

    # Indexes of K highest distances
    k_best_id = D.argsort()[:(-k-1):-1]
    k_best = []
    k_best_simi = []

    # Doc list with high to low distances
    for idx in k_best_id:
        k_best.append(corpus[idx])
        k_best_simi.append(D[idx])

    # Slice first K
    if k > len(k_best):
        best = k_best
        best_prob = k_best_simi
    else:
        best = k_best[:k]
        best_prob = k_best_simi[:k]

    return best, best_prob


best, best_prob = k_best(input('Enter Query:'))
print(best)