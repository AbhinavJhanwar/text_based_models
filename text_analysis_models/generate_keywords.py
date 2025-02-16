# %%
import yake
import re, string

import nltk
from nltk.corpus import stopwords
from nltk import ngrams
nltk.download('stopwords')
stops = set(stopwords.words('english'))

import spacy
# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')

from sentence_transformers import util

# %%
def clean_text(text:str)->str:
      """converts the input text to lower string, removes punctuation, double whitespace, \ and / slash

      Args:
          text (str): input text

      Returns:
          str: clean text
      """
      # lower encoding the text
      text = text.replace("'t", "t").lower()
      # remove punctuations
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
      lemmatized_tokens = [token.lemma_ for token in doc]

      # Join the lemmatized tokens into a sentence
      text = ' '.join(lemmatized_tokens)
      
      return text.lower()


def generate_keywords(doc:str, kw_model, keyphrase_ngram_range:tuple=(1, 3), n_keywords:int=10,
                      stop_words:str='english', method:str='simple', seed_keywords:list=[], highlight:bool=False):
      """Function to generate keywords.
      
      Method- simple:
            General keyword extractor using given BERT model
      Method- candidate:
            In some cases, one might want to be using candidate keywords 
            generated by other keyword algorithms or retrieved from a 
            select list of possible keywords/keyphrases. 
            In KeyBERT, you can easily use those candidate keywords to 
            perform keyword extraction.
      Method- guided:
            Guided KeyBERT is similar to Guided Topic Modeling in that 
            it tries to steer the training towards a set of seeded terms. 
            When applying KeyBERT it automatically extracts the most 
            related keywords to a specific document. However, there are
            times when stakeholders and users are looking for specific 
            types of keywords. For example, when publishing an article 
            on your website through contentful, you typically already 
            know the global keywords related to the article. However, 
            there might be a specific topic in the article that you would 
            like to be extracted through the keywords. To achieve this, 
            we simply give KeyBERT a set of related seeded keywords 
            (it can also be a single one!) and search for keywords that 
            are similar to both the document and the seeded keywords.
      Method- embedding:
            This method utilizes BERT model to generate embeddings for 
            the document and the words, then generates the keywords from
            those embeddings using cosine similarity and provides a score
            

      Args:
          doc (str): input document or text to extract keywords from
          kw_model (_type_): text model like KeyBert to process text and extract keywords
          keyphrase_ngram_range (tuple, optional): Maximum number of words in keyword. Defaults to (1, 3).
          n_keywords (int): maximum number of keywords to generate
          stop_words (str, optional): Choose from language in nltk. Defaults to 'english'.
          method (str, optional): _description_. Defaults to 'simple'.
          seed_keywords (list, optional): _description_. Defaults to [].
          highlight (bool, optional): Whether to hightlight keywords in the provided document or not. Defaults to False.

      Returns:
          _type_: list of keywords or no keywords
      """

      # clean text
      doc = clean_text(doc)

      if stop_words=='nltk':
            stop_words = list(stops)
      
      # method 1
      if method == 'simple':
            # directly generate keywords from bert model
            keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=keyphrase_ngram_range, top_n=max(50, n_keywords), stop_words=stop_words, highlight=highlight)
            return keywords
                        
      # method 2
      elif method == 'candidate':
            # generate keywords using candidates from other algorithm
            # Create candidates
            kw_extractor = yake.KeywordExtractor(top=max(100, n_keywords))
            candidates = kw_extractor.extract_keywords(doc)
            candidates = [candidate[0] for candidate in candidates]

            # Pass candidates to KeyBERT
            keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=keyphrase_ngram_range, top_n=max(50, n_keywords), stop_words=stop_words, highlight=highlight, candidates=candidates)
            
            return keywords
      
      # method 3
      elif method == 'guided':
            keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=keyphrase_ngram_range, top_n=max(50, n_keywords), stop_words=stop_words, highlight=highlight, seed_keywords=seed_keywords)
            return keywords

      # method 4
      elif method == 'embedding':
            # doc, pos_tags = clean_text(doc)

            # remove stop words
            text1 = ' '.join([token for token in doc.split() if token not in stops])

            if len(text1)>2:
                  # generate n grams
                  tokens = []
                  for i in range(1, 4):
                        tokens = tokens + list(set([' '.join(tokens) for tokens in ngrams(text1.split(), i)]))

                  # keep only n grams that are present in actual sentence
                  tokens = [token for token in tokens if token in doc]

                  # generate embedding
                  doc_emb = kw_model.encode(doc)
                  token_emb = kw_model.encode(tokens)

                  # Compute cosine similarity score between tokens and document embeddings
                  scores = util.cos_sim(doc_emb, token_emb)[0].cpu().tolist()

                  # Combine tokens & scores
                  token_score_pairs = list(zip(tokens, scores))

                  # Sort by decreasing score
                  token_score_pairs = sorted(token_score_pairs, key=lambda x: x[1], reverse=True)
                  return token_score_pairs[:n_keywords]
            else:
                  return []
