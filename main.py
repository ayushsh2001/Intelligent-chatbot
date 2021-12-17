
#Import the libraries
import nltk
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import bs4 as bs
import requests
import re
import warnings
warnings.filterwarnings('ignore')

#Download the punkt package
nltk.download('punkt')
nltk.download('wordnet')

r=requests.get('https://en.wikipedia.org/wiki/Climate_change')
raw_html=r.text

corpus_html=bs.BeautifulSoup(raw_html)
corpus_paras = corpus_html.find_all('p')
corpus_text = ' '

for para in corpus_paras:
  corpus_text +=para.text

  corpus_text = corpus_text.lower()


corpus_text = re.sub(r'\[[0-9]*\]',' ',corpus_text)
corpus_text = re.sub(r'\s+',' ',corpus_text)
[17]
corpus_sentence = nltk.sent_tokenize(corpus_text)
corpus_words = nltk.word_tokenize(corpus_text)


greeting_inputs = ("hey","good morning","good evening","morning","evening","hi","whatsup")
greeting_response = ["hey","hey hows you","*nods*","hello,how you doing","hello","Welcome, I am doing good"]

def greeting_response(greeting):
  for token in greeting.split():
    if token.lower() in greeting_inputs:
      return random.choice(greeting_responses)


wn_lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_corpus(tokens):
  return [wn_lemmatizer.lemmatize(token) for token in tokens]

punct_removal_dict = dict((ord(punctuation),None) for punctuation in string.punctuation)

def get_processed_text(document):
  return lemmatize_corpus(nltk.word_tokenize(document.lower().translate(punct_removal_dict)))

def respond(user_input):
  bot_response = ''
  corpus_sentence.append(user_input)

  word_vectorizer = TfidfVectorizer(tokenizer=get_processed_text, stop_words='english')
  corpus_word_vector = word_vectorizer.fit_transform(corpus_sentence)

  cos_sim_vectors = cosine_similarity(corpus_word_vector[-1], corpus_word_vector)
  similar_response_idx = cos_sim_vectors.argsort()[0][-2]

  matched_vectors = cos_sim_vectors.flatten()
  matched_vectors.sort()
  vector_matched = matched_vectors[-2]

  if vector_matched == 0:
    bot_response = bot_response + "I am sorry, what is it, again ?"
    return bot_response

  else :
    bot_response = bot_response + corpus_sentence[similar_response_idx]
    return bot_response


chat = True
print("Hello, What do you want to learn about climate today ?")
while(chat == True):
  user_query = input()
  user_query = user_query.lower()
  if user_query != 'quit':
    if user_query == 'thanks' or user_query == 'thank you':
      chat = False
      print("Climatebot: You are welcome!")
    else:
      if greeting_response(user_query) != None:
        print("Climatebot: " + greet_response(user_query))
      else:
        print("Climatebot:", end=" ")
        print(respond(user_query))
        corpus_sentence.remove(user_query)
  else:
    chat = False
    print("Climatebot: Good Bye!")
