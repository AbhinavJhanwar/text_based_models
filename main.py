# %%
from keybert import KeyBERT
from text_analysis_models.generate_keywords import generate_keywords

doc = """
         Family. The fact is, there is no foundation, no secure ground, 
         upon which people may stand today if it isn't the family. It's 
         become quite clear to Morrie as he's been sick. If you don't 
         have the support and love and caring and concern that you get from
         family, you don't have much at all. Love is so supremely important. 
         As our great poet Auden said, 'Love each other or perish.'
         Say you are divorced, or living alone, or had no childern. This
         disease-what Morrie is going through-would be so much harder. He is
         not sure he could do it. Sure, people would come visit, friends, 
         associates, but it's not the same as having someone who will not 
         leave. It's not the same as having someone whom you know has an eye
         on you, is watching you the whole time.
         This is part of what a family is about, not just love, but letting 
         others know there's someone who is watching out for them. It's what 
         Morrie missed so much when his mother died-what he calls your 'spiritual
         security'-knowing that your family will be there watching out for you. 
         Nothing else will give you that. Not money. Not fame. Not work.
      """

# %%
######################################################
############## generate keywords #####################
######################################################
kw_model = KeyBERT('all-distilroberta-v1')

################# method 1
keywords = generate_keywords(doc.lower(), kw_model, keyphrase_ngram_range=(1, 3), stop_words='nltk', method='simple', highlight=True) 
print("simple method", keywords, sep='\n') 

################# method 2
keywords = generate_keywords(doc.lower(), kw_model, keyphrase_ngram_range=(1, 3), stop_words='nltk', method='candidate', highlight=True) 
print("candidate method", keywords, sep='\n')

################# method 3
# Define seeded terms
seed_keywords = ["family", "knowledge", "children"]
keywords = generate_keywords(doc.lower(), kw_model, keyphrase_ngram_range=(1, 3), stop_words='nltk', method='guided', seed_keywords=seed_keywords, highlight=True) 
print("guided method", keywords, sep='\n')

################ method 4
# generate token and score
from sentence_transformers import SentenceTransformer
kw_model = SentenceTransformer('all-distilroberta-v1')#('all-MiniLM-L6-v2')
token_score_pairs = generate_keywords(doc.lower(), kw_model, method='embedding')
print("embedding method", token_score_pairs, sep='\n')

# %%
##################################################
############## plot keywords #####################
##################################################
from text_analysis_models.plot_keywords import plot_data
from sentence_transformers import SentenceTransformer
from text_analysis_models.generate_keywords import generate_keywords
import pandas as pd

'''# generate token and score
kw_model = SentenceTransformer('all-distilroberta-v1')#('all-MiniLM-L6-v2')
token_score_pairs = generate_keywords(doc.lower(), kw_model, method='embedding')
df = pd.DataFrame(token_score_pairs, columns=['keyword', 'weights'])

# plot data
plot_data(data=df.copy(), 
    keyword_column='keyword', 
    weights_column='weights',
    plot_type='phrase', 
    x_axis='count', 
    y_axis='importance', 
    bubble_size='count', 
    title_text='Keyword Importance', 
    save_file='bubble_plot', 
    sort_data='importance', min_size=0.005, 
    number_of_keywords_to_plot=20)'''

############# plot keywords for multiple docs
from text_analysis_models.generate_keywords import clean_text
from tqdm import tqdm
kw_model = SentenceTransformer('all-distilroberta-v1')#('all-MiniLM-L6-v2')

# read reviews
reviews = pd.read_pickle('data/reviews_gift_cards.pkl')
# clean review text
reviews['clean_reviewText'] = reviews['reviewText'].apply(clean_text)

# generate token and score
df = pd.DataFrame()
for i, doc in tqdm(enumerate(reviews.clean_reviewText)):
      token_score_pairs = generate_keywords(doc.lower(), kw_model, method='embedding')
      if type(token_score_pairs)!=str:
            sentiment = reviews['sentiment'].iloc[i]
            temp = pd.DataFrame(token_score_pairs, columns=['keyword', 'weights'])
            temp['sentiment'] = sentiment
            df = pd.concat([df, temp])

# plot data
plot_data(data=df.copy(), 
    keyword_column='keyword', 
    weights_column='weights',
    plot_type='phrase', 
    x_axis='count', 
    y_axis='importance', 
    bubble_size='sentiment', 
    title_text='Keyword Importance', 
    save_file='bubble_plot_reviews', 
    sort_data='count', 
    min_size=0.005, 
    number_of_keywords_to_plot=20)


# %%
#########################################################
################## sentiment generation #################
#########################################################
from text_analysis_models.sentiment_analysis import generate_sentiment
doc = """
      The Table looks better than the pics. 
      It is very Sturdy. The seller contacted me to ask 
      my colour preferences for the stool tapestry and
      what polish I want for my table. He did a fabulous 
      job and my table looks just the way I wanted it to! 
      Total value for money. 5 stars to the product, 
      seller and Amazon
      """
sentiment = generate_sentiment(doc)
print(sentiment)


# %%
#########################################################
################## topic modeling #######################
#########################################################
from text_analysis_models.generate_topic import generateTopic
import pandas as pd
doc = """
      The Table looks better than the pics. 
      It is very Sturdy. The seller contacted me to ask 
      my colour preferences for the stool tapestry and
      what polish I want for my table. He did a fabulous 
      job and my table looks just the way I wanted it to! 
      Total value for money. 5 stars to the product, 
      seller and Amazon
      """ 
# print('document:', doc)
reviews = pd.read_pickle('data/reviews_gift_cards.pkl')
reviews = reviews.drop_duplicates(['reviewText'])
texts = reviews.reviewText.tolist()

topics_data, top_topics = generateTopic(texts, topic_method='tfidf_score', generate_dendogram=False,
                  useLLM=False, distance_threshold=0.1, number_of_topics=10)

print('Overall topics:', top_topics)