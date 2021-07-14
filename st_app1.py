import numpy as np
import pandas as pd
import re
import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_pkl = open("vectorizer_model.pkl","rb")
vectorizer = pickle.load(vectorizer_pkl)
model_pkl = open('model.pkl', 'rb')
model = pickle.load(model_pkl)

# Read the file
with open('sentences.txt', 'r') as file:
    file = file.read().lower()
    file = re.sub(r'[^\w\s]','',file)

# Break into sentences
corpus = file.split('\n')

# def train(corpus):
#   vectorizer = TfidfVectorizer(ngram_range=(1,10), analyzer='char_wb')
#   model = vectorizer.fit_transform(corpus)
#   return vectorizer, model

# K-best suggestions for given query
def k_best(query, corpus=corpus, vectorizer=vectorizer, model=model, k=5):
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

  return best #, best_prob



def main():

  st.title("Search Query Prediction")
  html_temp = """
  <div style="background-color:tomato;padding:5px">
  <h2 style="color:white;text-align:center;"> Search Query Prediction App</h2>
  </div>
  """
  st.markdown(html_temp,unsafe_allow_html=True)

  # Add a selectbox to the sidebar:
  domain_selectbox = st.selectbox('Select Domain', ('Finance', 'Politics', 'Education'))
  sub_domain = st.text_input("Sub-domain",value="Type Here")
  query = st.text_input("Search", value="Enter Search Query")

  best_k = []
  if query:
    best_k = k_best(query)
    # best_k, best_probs = k_best(query)

  st.success(best_k)
  # st.multiselect("output",options=best_k)

if __name__=='__main__':
  main()