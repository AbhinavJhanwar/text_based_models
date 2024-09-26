# %%
import pandas as pd
import gzip
import json

from transformers import pipeline

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

# %%
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

# %%
def generate_sentiment(doc:str)->dict:
  """generates star rating from 1 to 5, 1 being lowest
    and 5 being highest, along with confidence of model.

  Args:
      doc (str): input string to generate sentiment for.

  Returns:
      dict: dictionary with rating and model confidence score.
  """

  # define tokenizer+classifier pipeline
  sentiment_pipeline = pipeline(model="nlptown/bert-base-multilingual-uncased-sentiment")
  doc= doc[:1400]
  
  review_generated=False
  while not review_generated:
      try:
          # run sentiment prediction model
          sentiment_op = sentiment_pipeline(doc)[0]
          review_generated=True
      except:
          doc = doc[:len(doc)-100]

  # separate sentiment label and score
  sentiment_label = int(sentiment_op['label'].split()[0])
  sentiment_score = sentiment_op['score']

  return {'Star Rating': sentiment_label,
            'Confidence Score': sentiment_score}

