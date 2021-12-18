
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

#Download the punkt and wordnet packages used for textual processing
nltk.download('punkt')
nltk.download('wordnet')

#collecting data...
r=requests.get('https://en.wikipedia.org/wiki/Season')
raw_html=r.text

#cleaning the raw data
corpus_html=bs.BeautifulSoup(raw_html)

#extracting paragraph from html..
corpus_paras = corpus_html.find_all('p')
corpus_text = ' '

#concatenating all the paragraphs..
for para in corpus_paras:
  corpus_text +=para.text

#converting the text into lower case..
  corpus_text = corpus_text.lower()

#removing all the special charaters from the text..
corpus_text = re.sub(r'\[[0-9]*\]',' ',corpus_text)
corpus_text = re.sub(r'\s+',' ',corpus_text)

#converting text into sentences and words..
corpus_sentence = nltk.sent_tokenize(corpus_text)
corpus_words = nltk.word_tokenize(corpus_text)

#generating greeting responses..
greeting_inputs = ("hey","good morning","good evening","morning","evening","hi","whatsup")
greeting_response = ["hey","hey hows you","*nods*","hello,how you doing","hello","Welcome, I am doing good"]

def greeting_response(greeting):
  for token in greeting.split():
    if token.lower() in greeting_inputs:
      return random.choice(greeting_response)

#preprocessing with punctuation removal and lemmatizing..
wn_lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_corpus(tokens):
  return [wn_lemmatizer.lemmatize(token) for token in tokens]

punct_removal_dict = dict((ord(punctuation),None) for punctuation in string.punctuation)

#language modeling with tf-idf..
def get_processed_text(document):
  return lemmatize_corpus(nltk.word_tokenize(document.lower().translate(punct_removal_dict)))

def respond(user_input):
  bot_response = ''
  corpus_sentence.append(user_input)

#vectorizing the processed text..
  word_vectorizer = TfidfVectorizer(tokenizer=get_processed_text, stop_words='english')
  corpus_word_vector = word_vectorizer.fit_transform(corpus_sentence)

#finding similarity between user query and entire corpus
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

#simply starting the chatbot..
chat = True
print("Hello, What do you want to learn about seasons today ?")
while(chat == True):
  user_query = input()
  user_query = user_query.lower()
  if user_query != 'quit':
    if user_query != 'thanks' and user_query != 'thank you' and user_query != 'bye':
      if greeting_response(user_query) != None:
        print("Seasonbot: " + greet_response(user_query))
      else:
        print("Seasonbot:", end=" ")
        print(respond(user_query))
        corpus_sentence.remove(user_query)
    else:
      chat = False
      print("Seasonbot: You are welcome!")
  else:
    chat = False
    print("Seasonbot: Good Bye!")  #ending message..
