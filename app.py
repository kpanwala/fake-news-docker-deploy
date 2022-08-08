import pickle
import numpy as np
import nltk 
import string 
import re
import num2words
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import base64
from flask import Flask, render_template
print(dir(Flask))
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline
import joblib 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from empath import Empath
from nltk.stem.porter import PorterStemmer 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
import time
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import zipfile
from flask import request
import gunicorn

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__, template_folder='template')

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route("/")
def home():
    '''nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')'''
    
    zip_ref = zipfile.ZipFile("pickles.zip", 'r')
    zip_ref.extractall("./zipFolder")
    
    return render_template("index.html")


# Convert to lowercase
def text_lowercase(text): 
	return text.lower() 
  
# convert number into words 
def convert_number(text): 
    # split string into list of words 
    temp_str = text.split() 
    # initialise empty list 
    new_string = [] 
  
    for word in temp_str: 
        # if word is a digit, convert the digit 
        # to numbers and append into the new_string list 
        if word.isdigit(): 
            temp = num2words(word) 
            new_string.append(temp) 
  
        # append the word as it is 
        else: 
            new_string.append(word) 
  
    # join the words of new_string to form a string 
    temp_str = ' '.join(new_string) 
    return temp_str 

# remove punctuation 
def remove_punctuation(text): 
    translator = str.maketrans('', '', string.punctuation) 
    return text.translate(translator) 

# remove whitespace from text 
def remove_whitespace(text): 
    return  " ".join(text.split()) 
  
# remove stopwords function 
def remove_stopwords(text): 
    stop_words = set(stopwords.words("english")) 
    word_tokens = word_tokenize(text) 
    filtered_text = [word for word in word_tokens if word not in stop_words] 
    return ' '.join(filtered_text)

# stem words in the list of tokenised words 
def stem_words(text):
    stemmer = PorterStemmer() 
    word_tokens = word_tokenize(text) 
    stems = [stemmer.stem(word) for word in word_tokens] 
    return ' '.join(stems)

# lemmatize string 
def lemmatize_word(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text) 
    # provide context i.e. part-of-speech 
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens] 
    return ' '.join(lemmas)

def empath_feature_calc(text):
    try:
        lexicon = Empath()
        return lexicon.analyze(text,normalize=True)
    except:
        return None

def empath_feature_calc(text):
    try:
        lexicon = Empath()
        return lexicon.analyze(text,normalize=True)
    except:
        print("gajab bejjati he")
        return None

def fun(res):
  print(res)
  if res == 0:
    return "Fake"
  return "True"

def predictEmpath(text, algorithm):
  df = pd.DataFrame([[text]],
     index=[0],
     columns=['text'])
  df['text']=df['text'].apply(text_lowercase)
  df['text']=df['text'].apply(convert_number)
  df['text']=df['text'].apply(remove_punctuation)
  df['text']=df['text'].apply(remove_whitespace)
  df['text']=df['text'].apply(remove_stopwords)
  df['text']=df['text'].apply(stem_words)
  df['text']=df['text'].apply(lemmatize_word)

  df['empath_features'] = df['text'].apply(empath_feature_calc)
  print(df.empath_features)
  newDF = pd.DataFrame(columns = list(df.empath_features[0].keys()))
  print(df.empath_features[0].keys())
  exclude_idx = []
  for i in range(df.shape[0]):
      dict_new_dataframe = df.empath_features[i]
      try:
          new_df = pd.DataFrame.from_dict(dict_new_dataframe.items()).transpose()
          header_new_dataframe = new_df.iloc[0]  # 0th index is header
          new_df = new_df[1:]  # fetch data except header
          new_df.columns = header_new_dataframe #set header as dataframe header in pandas
          newDF = newDF.append(new_df, ignore_index=True) # populate new dataframe with new data
      except:
          exclude_idx.append(i)
          pass

  columns=newDF.columns[:]
  x1=newDF[columns]
  print(x1)
  if algorithm == "Regressor":
    regE = joblib.load('./zipFolder/regressorEmpath.pkl')
    return fun(regE.predict(x1)[0])


  gbcE = joblib.load('./zipFolder/gbcEmpath.pkl')  
  return fun(gbcE.predict(x1)[0])

def predictTS20(text, algorithm):
  df = pd.DataFrame([[text]],
     index=[0],
     columns=['text'])
  df['text']=df['text'].apply(text_lowercase)
  df['text']=df['text'].apply(convert_number)
  df['text']=df['text'].apply(remove_punctuation)
  df['text']=df['text'].apply(remove_whitespace)
  df['text']=df['text'].apply(remove_stopwords)
  df['text']=df['text'].apply(stem_words)
  df['text']=df['text'].apply(lemmatize_word)
  vectorizer = TfidfVectorizer(stop_words="english", use_idf=True, ngram_range=(1,1))
  print(df['text'])
  tfidf = vectorizer.fit_transform(df['text'])
  n_topics = 20
  nmf = NMF(n_components=n_topics,random_state=0)
  topics = nmf.fit_transform(tfidf)
  top_n_words = 5
  t_words, word_strengths = {}, {}
  for t_id, t in enumerate(nmf.components_):
      t_words[t_id] = [vectorizer.get_feature_names()[i] for i in t.argsort()[:-top_n_words - 1:-1]]
      word_strengths[t_id] = t[t.argsort()[:-top_n_words - 1:-1]]
  pipe = Pipeline([
    ('tfidf', vectorizer),
    ('nmf', nmf)
  ])
  t = pipe.transform(df['text']) 
  t = pd.DataFrame(t, columns=[str(t_words[i]) for i in range(0, n_topics)])

  if algorithm == "Regressor":
    regTS = joblib.load('./zipFolder/regressorTS20.pkl')
    return fun(regTS.predict(t)[0])
  gbcTS = joblib.load('./zipFolder/gbcTS20.pkl')
  return fun(gbcTS.predict(t)[0])

def predictHybrid(text, algorithm):
  df = pd.DataFrame([[text]],
     index=[0],
     columns=['text'])
  df['text']=df['text'].apply(text_lowercase)
  df['text']=df['text'].apply(convert_number)
  df['text']=df['text'].apply(remove_punctuation)
  df['text']=df['text'].apply(remove_whitespace)
  df['text']=df['text'].apply(remove_stopwords)
  df['text']=df['text'].apply(stem_words)
  df['text']=df['text'].apply(lemmatize_word)

  df['empath_features'] = df['text'].apply(empath_feature_calc)

  newDF = pd.DataFrame(columns = list(df.empath_features[0].keys()))
  exclude_idx = []
  for i in range(df.shape[0]):
      dict_new_dataframe = df.empath_features[i]
      try:
          new_df = pd.DataFrame.from_dict(dict_new_dataframe.items()).transpose()
          header_new_dataframe = new_df.iloc[0]  # 0th index is header
          new_df = new_df[1:]  # fetch data except header
          new_df.columns = header_new_dataframe #set header as dataframe header in pandas
          newDF = newDF.append(new_df, ignore_index=True) # populate new dataframe with new data
      except:
          exclude_idx.append(i)
          pass
  
  vectorizer = TfidfVectorizer(stop_words="english", use_idf=True, ngram_range=(1,1))
  tfidf = vectorizer.fit_transform(df['text'])
  n_topics = 20
  nmf = NMF(n_components=n_topics,random_state=0)
  topics = nmf.fit_transform(tfidf)
  top_n_words = 5
  t_words, word_strengths = {}, {}
  for t_id, t in enumerate(nmf.components_):
      t_words[t_id] = [vectorizer.get_feature_names()[i] for i in t.argsort()[:-top_n_words - 1:-1]]
      word_strengths[t_id] = t[t.argsort()[:-top_n_words - 1:-1]]
  pipe = Pipeline([
    ('tfidf', vectorizer),
    ('nmf', nmf)
  ])
  t = pipe.transform(df['text']) 
  t = pd.DataFrame(t, columns=[str(t_words[i]) for i in range(0, n_topics)])
  merged=pd.concat([t, newDF], axis=1)

  columns=merged.columns[:]
  x0=merged[columns]
  if algorithm == "Regressor":
    regHy = joblib.load('./zipFolder/regressorHybrid.pkl')
    return fun(regHy.predict(x0)[0])
  gbcHy = joblib.load('./zipFolder/gbcHybrid.pkl')
  return fun(gbcHy.predict(x0)[0])

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    input_data = list(request.form.values())
    text = input_data[0]
    algo = input_data[1]
    method = input_data[2]
    print(text,algo,method)
    
    if method=="Hybrid":
        return render_template('index.html', prediction_text=" The news is {}".format(predictHybrid(text,algo)))
    elif method=="Fine":
        return render_template('index.html', prediction_text=" The news is {}".format(predictEmpath(text,algo)))
    else:
        return render_template('index.html', prediction_text=" The news is {}".format(predictTS20(text,algo)))
    
    
import base64
def stringToBase64(s):
    return base64.b64encode(s.encode('utf-8'))

def base64ToString(b):
    return base64.b64decode(b).decode('utf-8')
 
# main driver function
if __name__ == '__main__':
    app.run()