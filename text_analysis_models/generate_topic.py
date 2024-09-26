# %%
from typing import Union, List
import gzip

import pandas as pd
import numpy as np
import json

from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
stops.remove('not')
stops.remove('to')
stops.remove('no')
stops.remove('again')
stops.remove('off')

from nltk import ngrams
import re, string
import spacy
# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vectorizer = CountVectorizer(lowercase=True, 
                             stop_words=list(stops), token_pattern='[A-Za-z0-9í]+')
       
vectorizer2 = TfidfVectorizer(lowercase=True, 
                             stop_words=list(stops), token_pattern='[A-Za-z0-9í]+')

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
import umap
from sklearn.metrics.pairwise import cosine_distances

from tqdm import tqdm
tqdm.pandas()
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer, util
import time
kw_model = SentenceTransformer('all-distilroberta-v1')#('all-MiniLM-L6-v2')

from groq import Groq




def generate_llm_topics(text:list[str], model_name:str='llama-3.1-70b-versatile', 
                    API_KEY:str='', number_of_topics:int=10):
    """
        generate topics for the given list of text using llama model
        
        Args:
            text (list[str]): input text to generate topic for
            model_name (str): name of the llama model to utilize
            API_KEY (str): API key for the groq module
            number_of_topics (int): Number of topics to generate for each text

        Returns:
            list: list of lists of topics for given text list
    """
    client = Groq(
                  api_key=API_KEY,
                )
    
    topics = []
    for review in tqdm(text):
        try:
            chat_completion = client.chat.completions.create(
            messages = [
            {"role": "system", "content": "You are a helpful assitant who can generate topics for the review text given by user."},
            {"role": "user", "content": f"""
            Find the critical top {number_of_topics} topics about the product features only in the provided review that is delimited by triple backticks.
            Strictly follow the below guidelines-
            - Make sure that topics consist of words that are mentioned in the review and strictly no additional words should be used.
            - each topic consists of maximum 5 words.
            - include only topics that showcase user sentiments towards product or its components.
            - include topics that give sentiment towards product quality and usage.
            - include topics which show positive or negative aspects of product only.
            - topics should be interpretable.
            - extract unique topics only.
            review text: ```{review}``` 
            Format the output as JSON with the following keys:
            topic_1
            topic_2
            and so on
            In the output response, return the above JSON data only and no other text.
            If no topic found then return none. 
            """},
            ],
            model=model_name, #"llama-3.1-8b-instant", #"llama-3.1-70b-versatile"
            temperature=0,
            )
            response = chat_completion.choices[0].message.content
            if '```json' in response:
                topic = eval(response[7:-3].strip().replace('\n', '').replace('null', "'None'"))
            elif '```' in response:
                topic = eval(response[3:-3].strip().replace('\n', '').replace('null', "'None'"))
            else:
                topic = eval(response.strip().replace('\n', '').replace('null', "'None'"))
            topics.append(list(topic.values()))

        except:
            time.sleep(5)
            chat_completion = client.chat.completions.create(
            messages = [
            {"role": "system", "content": "You are a helpful assitant who can generate topics for the review text given by user."},
            {"role": "user", "content": f"""
            Find the critical top {number_of_topics} topics about the product features only in the provided review that is delimited by triple backticks.
            Strictly follow the below guidelines-
            - Make sure that topics consist of words that are mentioned in the review and strictly no additional words should be used.
            - each topic consists of maximum 5 words.
            - include only topics that showcase user sentiments towards product or its components.
            - include topics that give sentiment towards product quality and usage.
            - include topics which show positive or negative aspects of product only.
            - topics should be interpretable.
            - extract unique topics only.
            review text: ```{review}``` 
            Format the output as JSON with the following keys:
            topic_1
            topic_2
            and so on
            In the output response, return the above JSON data only and no other text.
            If no topic found then return none. 
            """},
            ],
            model=model_name, #"llama-3.1-8b-instant", #"llama-3.1-70b-versatile"
            temperature=0,
            )
            response = chat_completion.choices[0].message.content
            if '```json' in response:
                topic = eval(response[7:-3].strip().replace('\n', '').replace('null', "'None'"))
            elif '```' in response:
                topic = eval(response[3:-3].strip().replace('\n', '').replace('null', "'None'"))
            else:
                topic = eval(response.strip().replace('\n', '').replace('null', "'None'"))
                
            topics.append(list(topic.values()))
    
    return topics

 
def clean_text(text:str, lemma:bool=True):
    """
    1. converts the input text to lower string, removes punctuation, double whitespace, \ and / slash
    2. lemmatize text
    3. generate pos tags
    4. filter on unwanted pos tags

    Args:
        text (str): input text

    Returns:
        str: clean text
    """
    # lower encoding the text, some text formatting
    text = text.replace("'t", "t").lower().replace('no ', 'not ').replace('-', '').replace('’', "'")
    # remove punctuations
    # text = re.sub('['+string.punctuation+']+', ' ', text)     
    text = re.sub('['+string.punctuation+']+', ' ', text)  
    
    # Remove double whitespace                       
    text = re.sub('\s+\s+', ' ', text)      
    # Remove \ slash
    text = re.sub(r'\\', ' ', text)         
    # Remove / slash
    text = re.sub(r'\/', ' ', text)   

    # Process the text using spaCy
    doc = nlp(text)

    # Extract lemmatized tokens
    lemmatized_tokens = [(token, token.lemma_, token.pos_) for token in doc]

    # Join the lemmatized tokens into a sentence
    processed_text = ' '.join([token[1] for token in lemmatized_tokens])

    # filter text to remove unwanted pos tags
    pos_text = ' '.join([str(token[0])for token in lemmatized_tokens if token[2] not in [
                                                'AUX', 'PUNCT', 'SCONJ', 'CCONJ', # 'ADP',
                                                'DET', 'INTJ', 'CONJ', 'PRON', 'SYM', 'X', 'NUM',
                                                ]])

    # get pos tags of text
    pos_tags = {}
    for token in doc:
        if token.pos_ not in ['AUX', 'PUNCT', 'SCONJ', 'CCONJ', 'DET', #'ADP',
                                                'INTJ', 'CONJ', 'PRON', 'SYM', 'X', 'NUM']:
            pos_tags[str(token).lower()] = token.pos_

    return processed_text.lower(), pos_text.lower(), pos_tags


def filter_keywords(docs_pos)->set:
    """
    1. remove stop words
    2. generates n grams of text
    3. filter on specified combination of pos tags

    Args:
        keywords (list): input [(keyword, score)] format
        pos_tags (dict): pos tag for all words in document

    Returns:
        list: _description_
    """
    # list to store topics
    topics = []
    # topics_pos_tags = []

    for doc_pos in docs_pos:
        processed_text, pos_text, pos = doc_pos
        
        # remove stop words
        # processed_text = ' '.join([token for token in processed_text.split() if token not in stops])
        pos_text = ' '.join([token for token in pos_text.split() if token not in stops])

        # generate n grams
        tokens = []
        for i in range(1, 4):
            tokens = tokens + list(set([' '.join(tokens) for tokens in ngrams(pos_text.split(), i)]))

        # keep only n grams that are present in actual sentence
        # tokens = [token for token in tokens if token in processed_text]

        for token in tokens:
            pos_tag = ' '.join([pos[word] for word in token.split()])
            # print(token, pos_tag)
            if pos_tag not in [
                'PART', 'VERB', 'ADV', 'PROPN', 'NUM', 'ADP', 'NUM',
                
                'NUM NOUN', 'NUM ADJ', 'NUM ADV', 'NUM ADJ ADJ', 'NUM NOUN ADJ', 'NUM NOUN NUM', 'NUM NUM VERB',
                'NUM ADJ ADJ NOUN', 'NUM NOUN ADJ VERB', 'NUM VERB', 'NUM VERB ADJ', 'NUM VERB ADP', 'NUM ADP ADJ',
                'NUM ADJ VERB', 'NUM NUM', 'NUM VERB VERB', 'NUM NUM NOUN', 'NUM NOUN PART', 'NUM NUM NUM',
                'NUM VERB NUM', 'NUM NOUN ADV', 'NUM VERB NOUN', 'NUM ADJ NOUN', 'NUM PROPN', 'NUM NOUN PROPN',
                'NUM PROPN NOUN', 'NUM ADP', 'NUM ADV VERB', 'NUM PROPN PROPN', 'NUM PART', 'NUM PART ADV',
                'NUM NUM PART', 'NUM VERB PART', 'NUM NOUN NOUN', 'NUM NOUN VERB', 'NUM PART NUM', 
                
                'PART ADV', 'PART VERB', 'PART VERB NOUN', 'PART VERB ADJ', 'PART VERB PART', 'PART NUM NOUN',
                'PART VERB NOUN NOUN', 'PART VERB PART VERB', 'PART VERB ADJ NOUN', 'PART PART VERB', 'PART PART',
                'PART VERB VERB',  'PART NOUN VERB', 'PART VERB NUM', 'PART ADJ VERB', 'PART ADV NUM',
                'PART ADV NOUN', 'PART ADJ PART', 'PART NOUN NOUN', 'PART NUM', 'PART ADV ADP', 'PART NOUN ADJ',
                'PART NUM ADJ', 'PART ADP', 'PART ADP NOUN', 'PART PROPN', 'PART ADJ NUM', 'PART NOUN',
                'PART ADJ ADP', 'PART NOUN PART', 'PART VERB ADP', 'PART VERB PROPN', 'PART NOUN PROPN', 
                'PART ADP VERB',

                'NOUN PART', 'NOUN NUM', 'NOUN PROPN', 'NOUN ADV', 'NOUN PART ADV', 'NOUN PROPN ADV',
                'NOUN NOUN VERB', 'NOUN NOUN NOUN', 'NOUN VERB ADV', 'NOUN ADV VERB',  'NOUN ADV PART',
                'NOUN NOUN NUM', 'NOUN ADJ VERB', 'NOUN NOUN ADV', 'NOUN ADJ NOUN', 'NOUN NUM ADP', 'NOUN ADV ADV', 
                'NOUN NUM NOUN', 'NOUN NOUN PART', 'NOUN VERB ADJ', 'NOUN PART VERB ADJ', 'NOUN VERB PROPN',
                'NOUN NOUN VERB ADV', 'NOUN ADV ADJ PART', 'NOUN NOUN NUM NOUN', 'NOUN NUM ADV', 'NOUN PROPN ADJ',
                'NOUN NOUN NOUN PART', 'NOUN ADJ NOUN NOUN', 'NOUN NOUN PART VERB', 'NOUN ADV NUM', 'NOUN PROPN PART',
                'NOUN VERB ADP', 'NOUN NUM ADJ', 'NOUN PART NOUN', 'NOUN ADP VERB', 'NOUN ADV ADP', 'NOUN ADJ ADP',
                'NOUN VERB NUM', 'NOUN ADJ PART', 'NOUN ADP', 'NOUN ADP NOUN', 'NOUN NOUN ADP', 'NOUN ADP PROPN',
                'NOUN ADV NOUN', 'NOUN PROPN NOUN', 'NOUN ADJ NUM', 'NOUN NUM NUM', 'NOUN ADJ ADV', 'NOUN NUM VERB', 
                'NOUN NOUN PROPN', 'NOUN VERB PART', 'NOUN PROPN VERB', 'NOUN PART PROPN', 'NUM VERB PROPN',
                'NOUN PART NUM', 'NOUN ADV PROPN', 'NOUN ADP ADJ', 'NOUN ADP PART', 'NOUN ADV ADJ', 'NOUN PROPN PROPN',
                'NOUN PART ADJ', 'NOUN ADP ADV',  'NOUN ADP ADP',
                
                'VERB ADJ', 'VERB PRON', 'VERB PART', 'VERB NUM', 'VERB VERB NOUN', 'VERB NOUN PART', 'VERB NUM ADJ',
                'VERB PART VERB', 'VERB ADV NOUN', 'VERB NUM ADV', 'VERB NOUN VERB', 'VERB VERB VERB', 'VERB ADP NUM',
                'VERB ADV ADV ADV', 'VERB ADJ ADJ NOUN', 'VERB ADJ NOUN PART', 'VERB ADJ ADJ', 'VERB PROPN ADJ',
                'VERB VERB ADJ', 'VERB ADP', 'VERB NUM VERB', 'VERB ADJ VERB', 'VERB NOUN NUM', 'VERB NOUN PROPN',
                'VERB NOUN ADP', 'VERB NUM NOUN', 'VERB NUM NUM', 'VERB ADV VERB', 'VERB PROPN', 'VERB PROPN ADP',
                'VERB ADV PART', 'VERB PROPN NOUN', 'VERB VERB PROPN', 'VERB ADP VERB', 'VERB PROPN PROPN',
                'VERB VERB ADV', 'VERB ADJ ADV', 'VERB ADJ PROPN', 'VERB ADV NUM', 'VERB PROPN VERB', 'VERB ADJ NUM',
                'VERB PART ADJ', 'VERB VERB ADP', 'VERB ADJ PART', 'VERB VERB NUM', 'VERB VERB PART', 'VERB NUM PROPN',
                'VERB PART ADV', 'VERB PROPN NUM', 'VERB NUM PART', 'VERB ADV ADP', 'VERB PART PART', 'VERB ADJ ADP',
                
                'PROPN NOUN PART', 'PROPN PART', 'PROPN PART ADJ', 'PROPN VERB PROPN', 'PROPN ADP', 'PROPN ADP NOUN', 
                'PROPN NOUN VERB', 'PROPN NOUN ADJ', 'PROPN VERB ADJ', 'PROPN NOUN ADP', 'PROPN ADJ',
                'PROPN NUM', 'PROPN VERB NOUN', 'PROPN NUM NOUN', 'PROPN PROPN NUM', 'PROPN PROPN ADJ',
                'PROPN NOUN ADV', 'PROPN NOUN PROPN', 'PROPN ADV', 'PROPN ADV PROPN', 'PROPN ADJ PART',
                'PROPN ADJ NOUN', 'PROPN PROPN NOUN', 'PROPN NOUN NUM', 'PROPN PROPN VERB', 'PROPN VERB VERB',
                'PROPN VERB NUM', 'PROPN NUM NUM', 'PROPN PROPN', 'PROPN PROPN PROPN', 'PROPN VERB PART',
                'PROPN NUM PROPN', 'PROPN PROPN PART', 'PROPN PART NOUN', 'PROPN ADV VERB', 'PROPN PART ADV',
                
                'ADJ PART', 'ADJ NOUN VERB', 'ADJ NOUN ADJ', 'ADJ PROPN PROPN',
                'ADJ ADV', 'ADJ NOUN ADV','ADJ NOUN NUM', 'ADJ PART ADV', 
                'ADJ NOUN NOUN ADV', 'ADJ NOUN NOUN VERB', 'ADJ NOUN ADJ NOUN', 'ADJ NOUN PART', 'ADJ ADJ ADV',
                'ADJ ADJ VERB', 'ADJ ADP', 'ADJ VERB ADJ', 'ADJ PART NOUN', 'ADJ ADV VERB', 'ADJ ADJ NUM',
                'ADJ VERB NUM', 'ADJ ADP VERB', 'ADJ NUM', 'ADJ NUM VERB', 'ADJ NUM NOUN', 'ADJ NUM NUM',
                'ADJ PART ADJ', 'ADJ ADV PART', 'ADJ ADJ ADJ', 'ADJ VERB ADV', 'ADJ VERB VERB', 'ADJ ADJ PART',
                'ADJ VERB PART', 'ADJ VERB ADP', 'ADJ PROPN PART', 'ADJ PROPN', ' ADJ ADV NOUN', 'ADJ PART ADP',
                'ADJ ADV NOUN', 'ADJ NOUN ADP', 'ADJ ADV ADV', 'ADJ ADP NUM', 'ADJ ADV ADP', 'ADJ ADV ADJ',
                'ADJ ADP PROPN', 'ADJ NOUN PROPN',
                
                'ADV ADJ', 'ADV ADV', 'ADV VERB PART', 'ADV VERB PART VERB', 'ADV ADJ PART', 'ADV ADV VERB',
                'ADV ADJ PART VERB', 'ADV NOUN VERB', 'ADV VERB ADJ', 'ADV ADV ADV', 'ADV NUM ADJ', 'ADV PROPN',
                'ADV VERB ADV', 'ADV NOUN ADJ', 'ADV NOUN ADJ', 'ADV ADJ NUM', 'ADV PART', 'ADV PART PART',
                'ADV PART VERB', 'ADV ADV ADP', 'ADV ADP', 'ADV ADP NOUN', 'ADV VERB ADP', 'ADV PROPN NOUN',
                'ADV NOUN NOUN', 'ADV ADV NOUN', 'ADV ADJ ADJ', 'ADV VERB VERB', 'ADV VERB PROPN', 'ADV ADJ ADV',
                'ADV ADJ VERB', 'ADV NOUN PART', 'ADV NUM', 'ADV NUM NOUN', 'ADV NOUN ADV', 'ADV ADV PART',
                'ADV PART ADV', 'ADV ADV ADJ', 'ADV NUM VERB', 'ADV PROPN PROPN', 'ADV PART NOUN', 'ADV VERB NUM',
                'ADV PROPN VERB', 'ADV ADV NUM', 'ADV NOUN ADP', 'ADV ADP PART', 'ADV ADJ ADP', 'ADV NOUN PROPN',
                'ADV NOUN', 'ADV ADP VERB', 'ADV PART ADP',

                'ADP NOUN', 'ADP NOUN NOUN', 'ADP VERB', 'ADP VERB VERB', 'ADP PART', 'ADP PART VERB', 'ADP PROPN',
                'ADP NOUN VERB', 'ADP ADJ', 'ADP ADJ NOUN', 'ADP VERB ADJ', 'ADP NUM', 'ADP NOUN ADV',
                'ADP NUM NOUN', 'ADP PROPN NOUN', 'ADP VERB NOUN', 'ADP NOUN ADJ', 'ADP PROPN PROPN',
                'ADP ADJ ADJ', 'ADP NOUN PART', 'ADP VERB ADP', 'ADP NUM NUM', 'ADP PROPN PART',
                'ADP ADV', 'ADP ADV ADV', 'ADP ADP', 'ADP ADP ADJ',
                
                'PROPN VERB', 'NOUN ADJ PROPN', 'VERB VERB', 'VERB NOUN ADV', 'VERB ADP ADJ', 'ADJ ADJ',# RETHINK
                'ADJ VERB NOUN', 'VERB NOUN', 'ADJ VERB', 'PART ADV VERB', 'ADV ADP ADJ',# RETHINK

                # 'NOUN', 'ADJ', # KEEP
                # 'PROPN NOUN NOUN', 'PROPN NOUN', # KEEP
                # 'NOUN VERB VERB', 'NOUN VERB', 'NOUN VERB NOUN', 'NOUN PART VERB', # KEEP
                # 'NOUN NOUN', 'NOUN NOUN ADJ', 'NOUN ADJ ADJ', 'NOUN ADJ',#KEEP
                # 'VERB ADP NOUN', 'VERB NOUN ADJ', 'VERB ADJ NOUN', 'VERB PART NOUN', 'VERB NOUN NOUN', #KEEP
                # 'VERB ADV', 'VERB ADV ADJ', 'VERB ADV ADV', #KEEP
                # # 'VERB PART VERB NOUN',
                # 'ADJ PART VERB', 'ADJ ADP NOUN', 'ADJ PROPN NOUN', 'ADJ NOUN', 'ADJ ADJ NOUN', #KEEP
                # 'ADJ NOUN NOUN', 'ADJ ADJ PROPN', #KEEP
                # # 'ADJ NOUN PART VERB',
                # 'ADV ADJ NOUN', 'ADV VERB NOUN', 'ADV VERB', 'ADV ADP VERB',# KEEP
                # 'PART ADJ', 'PART ADJ NOUN', 'PART ADJ ADV', 'PART ADV ADV', 'PART ADV ADJ', 'PART VERB ADV',  # KEEP
                # # 'PART VERB ADV ADV', 
                ]:
                topics.append(token)
                # topics_pos_tags.append(pos_tag)

    return topics#, topics_pos_tags


def get_unique_topics(df, train_data, number_of_topics, topic_method):
    if len(df)>1:
        # get embedding for the topics
        df = pd.merge(df, train_data, on='topics_lemma')
        # get distance between topics
        sample = df[[column for column in df.columns if column in range(0, 768)]]
        df['distance'] = cosine_distances(sample, sample).mean(axis=1)

        # get final score = mean value of above two, sort as per that score and return topics.
        df['final_score'] = pd.DataFrame(MinMaxScaler().fit_transform(df[[topic_method, 'distance']])).mean(axis=1)
        # print(df.sort_values('final_score', ascending=False)[['topics', 'count_score', 'count_score', 'distance', 'distance_score', 'final_score']].head(15))
        tpics = df.sort_values('final_score', ascending=False)[:number_of_topics]
        tpics = pd.DataFrame.from_dict({'topics':[tpics.topics.tolist()], 'cluster_head':[tpics.cluster_head.tolist()], 'final_score':[tpics.final_score.tolist()]})
        # print(tpics)
    else:
        tpics = df
    return tpics


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def load_data(category:str='gift_cards')->pd.core.frame.DataFrame:
  
  if category=='gift_cards':
    # gift cards
    reviews = getDF('Gift_Cards.json.gz')
    
  elif category=='luxury_beauty':
    # beauty products
    reviews = getDF('Luxury_Beauty.json.gz')
    
  reviews = reviews[['asin', 'overall', 'reviewText']]

  # dropna
  reviews.dropna(inplace=True)

  # extract random sample of 1000 reviews
  reviews = reviews.sample(1000).reset_index(drop=True)

  return reviews

#category = 'luxury_beauty'
# load data by category
#reviews = load_data(category)

def get_score(x, count_data):
  temp = count_data[count_data['processed text']==x['processed text']].iloc[0].to_dict()
  # give some keywords higher importance
  if 'not' in temp.keys():
      temp['not'] = temp['not']*1.75
  if 'to' in temp.keys():
      temp['to'] = temp['to']*0.70
  # print(temp)
  score = np.sum([temp[word] for word in x['topics_lemma'].split() if word in [item for item in temp.keys()]])
  return score


def lemmatization(x):
    doc = nlp(x)
    # Extract lemmatized tokens
    lemmatized_tokens = [token.lemma_ for token in doc]
    return ' '.join(lemmatized_tokens).lower()


def generateTopic(original_text:Union[str, List[str]], topic_method:str='tfidf_score', 
                  generate_dendogram:bool=False, useLLM:bool=False,
                  distance_threshold:float=0.5, number_of_topics:int=5, API_KEY:str=''):
  """
      Function to generate topics from the given list of text or single document.
      Steps Involved-
        1. generate topics/keywords from the given documents.
        2. Filter keywords for appropriate POS combinations. (IF NOT LLM)
        3. Generate embedding for all the topics identified.
        4. Cluster them using Hierarchical Clustering to reduce topic redundancy and improve quality of topics.
        5. Score each and every topic using tfidf vectorizer. 
        6. Identify one topic from each cluster that has highest score.
        7. Generate topics for each document based on their scores and return original top topics and their cluster representations.
        8. To generate overall topics for all the documents-
          a. sort topics on count value (sum of count of each word in topic)
          b. sort topics on tfidf value
          c. sort topics on modified tfidf score based on importance of POS combinations (importance dictionary)
        9. Get average cosine distance between each and every topic.
        10. Using scoring method above (8) and cosine distance, generate final score as average of two called final_score.
        11. Generate top topics based on this final score.
        12. Apply step 9 and 10 on the topics from each document and return top number_of_topics.

      count_score: Step 8(a)
      tfidf_score: Step 8(b)
      i_score: Step 8(c)
      method3: method 3 is similar to method 2, instead of clustering
              the tokens, it clusters the sentences separated by '.'.
              Instead of using cosine similarity it generates an importance 
              score for each token using below formula-
              importance = (word count in current cluster/total word count in current cluster)*
                            log(total number of clusters/word count across clusters)
              In short a token appearing across clusters reduces its importance,
              while its appearance in current cluster increases its importance.
              Now for each token in each sentence an importance score is available
              and accordingly topic is generated. Also called c-tfidf

  Args:
      original_text (Union[str, list[str]]): input text to generate topic
      topic_method (str, optional): Choose from count_score, tfidf_score, i_score. Defaults to 'tfidf_score'. See summary for details- point 8.
      generate_dendogram (bool, optional): To generate dendogram for the purpose of deciding appropriate distance_threshold. Defaults to False.
      useLLM (bool, optional): Whether to use LLM to generate topics. Defaults to False.
      distance_threshold (float, optional): distance threshold value to cluster data. Defaults to 0.5.
      clusters (int, optional): number of clusters to group keywords, you can also decide based on the elbow method as generated automatically. Defaults to 10.
      number_of_topics (int, optional): number of topics to return. Defaults to 5.
      API_KEY (str, optional): Provide your api key for groq.

  Returns:
      pd.DataFrame, List: returns dataframe with original text, top topics and topic representation along with a list of overall topics.
  """

  # generate a list if string
  if type(original_text)==str:
    original_text = [original_text]

  ################################################# get topics ##############################
  topics = []
  processed_text = []

  if useLLM:
    # get topics using llama
    topics = generate_llm_topics(original_text, API_KEY=API_KEY)
    
    for doc in original_text:
      # break into sentences and clean text
      doc_pos = [clean_text(sentence) for sentence in doc.split('.')]
      processed_text.append(' '.join([i[0] for i in doc_pos]))
      
  else:
    for doc in original_text:
      # break into sentences and clean text
      doc_pos = [clean_text(sentence) for sentence in doc.split('.')]
      processed_text.append(' '.join([i[0] for i in doc_pos]))
      # filter keywords using pos tags
      filtered_topics = filter_keywords(doc_pos)
      # get topics based on pos tags
      topics.append(filtered_topics)

  topics_data = pd.DataFrame()
  topics_data['text'] = original_text
  topics_data['processed text'] = processed_text
  topics_data['topics'] = topics
  
  # expand each topic as a separate row
  topics_data = topics_data.explode(['topics'])
  
  # remove null values
  topics_data = topics_data.fillna('')

  # lemmatize topics for scoring and clustering
  topics_data['topics_lemma'] = topics_data['topics'].apply(lemmatization)

  # get all unique topics into a list
  topics_lemma = topics_data.topics_lemma.unique().tolist()
  ###############################################################################################

  #################################### generate scores for each topic ###########################
  ''' count vector '''
  # generate count matrix of data
  matrix = vectorizer.fit_transform(topics_data['processed text'].unique())
  # create dataframe of count data
  count_data = pd.DataFrame(data=matrix.toarray(), columns=vectorizer.get_feature_names_out())
  # generate overall score across documents for each word
  count_score = count_data.sum().to_dict()
  # sorted(count_score.items(), key=lambda x:x[1], reverse=True)

  for i,j in count_score.items():
    if j>=matrix.shape[0]*0.35:
        count_score[i] = j*0.35
    else:
        count_score[i] = j*0.9

  scores = []
  for i, doc in enumerate(topics_lemma):
    scores.append(np.sum([count_score[word] for word in doc.split() if word in [item for item in count_score.keys()]]))
      
  # create empty dataframe to store score and topics
  topics_df = pd.DataFrame()

  # combine all topics and scores
  topics_df['topics_lemma'] = topics_lemma
  topics_df['count_score'] = scores

  # add score column in topics_data
  topics_data['count_score'] = topics_data['topics_lemma'].apply(lambda x: topics_df[topics_df.topics_lemma==x]['count_score'].iloc[0])


  ''' tfidf '''
  # generate tfidf matrix of data
  matrix = vectorizer2.fit_transform(topics_data['processed text'].unique())
  
  # create dataframe of count data
  count_data = pd.DataFrame(data=matrix.toarray(), columns=vectorizer2.get_feature_names_out())
  count_data['processed text'] = topics_data['processed text'].unique()
    
  #print(matrix.toarray())
  #print(vectorizer.get_feature_names_out())
  #print(matrix.shape)

  topics_data['tfidf_score'] = topics_data[['processed text', 'topics_lemma']].apply(
     lambda x: get_score(x, count_data), axis=1)

  ###############################################################################################

  ################################## combine similar topics into clusters #######################
  # generate embedding for documents and tokens
  token_emb = kw_model.encode(topics_lemma)
  df_embedding = pd.DataFrame(token_emb)
  df_embedding['topics_lemma'] = topics_lemma
  train_data = df_embedding.drop_duplicates(['topics_lemma'])

  # prepare data for model training
  reducer = umap.UMAP(random_state=42, n_components=2)
  embeddings = reducer.fit_transform(train_data[list(range(0, token_emb.shape[1]))])

  # hierarchical clustering
  model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, 
                                  metric='euclidean', linkage='ward')
  y = model.fit_predict(embeddings)
  train_data['clusters'] = y
  train_data = train_data.reset_index(drop=True)
  print('number of clusters-', len(train_data['clusters'].unique()))

  if generate_dendogram:
    print('top 10 clusters and their count-', "\n", train_data.clusters.value_counts()[:10])
    clusters = train_data.clusters.value_counts()[:10].index[:3].tolist()

    plt.figure(figsize = (10, 12))
    # cluster 1
    labels = train_data.iloc[train_data[train_data.clusters.isin(clusters[:1])]['topics_lemma'].index.values.tolist()]['topics_lemma'].tolist()
    dendrogram(linkage(embeddings[train_data[train_data.clusters.isin(clusters[:1])]['topics_lemma'].index.values.tolist()], method='ward'), 
                        labels=labels, leaf_rotation=0, leaf_font_size=8, orientation='right')
    plt.show()

    # cluster 2
    labels = train_data.iloc[train_data[train_data.clusters.isin(clusters[1:2])]['topics_lemma'].index.values.tolist()]['topics_lemma'].tolist()
    dendrogram(linkage(embeddings[train_data[train_data.clusters.isin(clusters[1:2])]['topics_lemma'].index.values.tolist()], method='ward'), 
                        labels=labels, leaf_rotation=0, leaf_font_size=8, orientation='right')
    plt.show()

    # cluster 3
    labels = train_data.iloc[train_data[train_data.clusters.isin(clusters[2:3])]['topics_lemma'].index.values.tolist()]['topics_lemma'].tolist()
    dendrogram(linkage(embeddings[train_data[train_data.clusters.isin(clusters[2:3])]['topics_lemma'].index.values.tolist()], method='ward'), 
                        labels=labels, leaf_rotation=0, leaf_font_size=8, orientation='right')
    plt.show()

  # sort all topics based on score and drop duplicate topics
  cluster_head = topics_data.sort_values(['tfidf_score'], ascending=False).drop_duplicates()

  # assign cluster to all topics
  clusters = []
  for topic in cluster_head.topics_lemma:
      clusters.append(train_data[train_data['topics_lemma']==topic]['clusters'].iloc[0])
  cluster_head['clusters'] = clusters

  # keep only one value from each cluster, as they are sorted by score already hence 
  # top keyword from each cluster will remain in the dataframe
  cluster_head = cluster_head.drop_duplicates(['clusters'])

  # add cluster head to each topic
  topics_data['cluster_head'] = topics_data['topics_lemma'].apply(
     lambda x: cluster_head[cluster_head.clusters==train_data[train_data['topics_lemma']==x
                                                              ]['clusters'].iloc[0]]['topics_lemma'].iloc[0])
  ###############################################################################################  
  
  ################# get topics that are far from each other in terms of context #################

  # get top scoring cluster head from each document
  top_topics = topics_data.sort_values(topic_method, ascending=False).drop_duplicates(
      'text').drop_duplicates('cluster_head')[['text', 'topics', 'topics_lemma', 'cluster_head', 
                                               'count_score', 'tfidf_score']]

  # get embedding for the topics
  df = pd.merge(top_topics, train_data, on='topics_lemma')

  # get distance between topics
  sample = df[[column for column in df.columns if column in range(0, 768)]]
  df['distance'] = cosine_distances(sample, sample).mean(axis=1)

  # get final score = mean value of above two after normalizing, sort as per that score and return topics.
  df['final_score'] = pd.DataFrame(MinMaxScaler().fit_transform(df[['tfidf_score', 'distance']])).mean(axis=1)
  top_topics = df.sort_values('final_score', ascending=False)[['text', 'topics', 'final_score']]
  top_topics = top_topics.rename(columns={'topics':'final_topic', 'final_score':'final_topic_score'})

  ###############################################################################################

  ########################## get overall top topics #############################################
  # get top topics for each document
  df1 = topics_data.groupby('text')[['topics_lemma', 'topics', topic_method, 'cluster_head']].apply(
     lambda x: get_unique_topics(x, train_data, number_of_topics, topic_method)).reset_index()[['text',	'topics',	'cluster_head',	'final_score']]

  # merge with final topic
  df1 = pd.merge(df1, top_topics, on='text')
  # df1 = df1.rename(columns={0:'topics'})
  ###############################################################################################
  
  return df1, top_topics['final_topic'].values.tolist()[:number_of_topics]