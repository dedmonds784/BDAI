
# coding: utf-8

# In[ ]:


## Create Api Url
import requests as rq
import pandas as pd
import numpy
import sys
import newspaper as np 
from nltk.tokenize import RegexpTokenizer
from textblob import TextBlob
def extract_articles(company, source, key= "79613011d94745a89fbd3067a42c9c0f"):
    url = ['https://newsapi.org/v2/everything?', # organizes the url to the api
               'q=%s&'%(company),
               'sources=%s&'%(source),
               'apiKey=%s'%(key)]
    url = "".join(url)
    response = rq.get(url) 
    xDF = pd.DataFrame.from_dict(pd.DataFrame(response.json()).articles)
    return xDF
#x = extract_articles('apple', 'cnn')



# In[ ]:


def clean_df(df):    
    # cleans the dates of the published articles
    df = pd.DataFrame.from_dict(list(df.articles))
    df['publishedAt'] = pd.to_datetime(df['publishedAt'].str.extract('(\d{4}-\d{2}-\d{2})'))
    df['url'] = df['url'].apply(np.Article) # performs 
    return df
#x = clean_df(x)  



# In[ ]:


def get_text(x): 
    try:
        x.download()
        x.parse()
        tokenizer = RegexpTokenizer('\w+')
        return tokenizer.tokenize(x.text)
    except:
        pass


# In[ ]:


def get_text2(x): 
    try:
        x.download()
        x.parse()
        return x.text
    except:
        pass


# In[ ]:


def organize_tokens_by_date(corpus):
    corp = pd.DataFrame(columns = ['Date', 'text']) # pulls the requested columns into a specific df
    corp.text = corpus['url'].apply(get_text)
    corp.Date = corpus.publishedAt 
    corp.set_index('Date', inplace = True)
    corp = (corp.text.apply(pd.Series)
                .stack()
                .reset_index(level = 1, drop = True)
                .to_frame('text'))
    corp.rename(columns ={'text' : 'word' }, inplace = True)
    return corp
#x = organize_tokens_by_date(x)



# In[ ]:


def organize_articles_by_date(corpus):
    corp = pd.DataFrame(columns = ['Date', 'text']) # pulls the requested columns into a specific df
    corp['text'] = corpus['url'].apply(get_text2)
    corp['Date'] = corpus.publishedAt 
    corp.set_index('Date', inplace = True)
    corp = (corp.text.apply(pd.Series)
                .stack()
                .reset_index(level = 1, drop = True)
                .to_frame('text'))
    corp.rename(columns ={'text' : 'word' }, inplace = True)
    corp['Polarity'] = corp['word'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
    corp['Subjectivity'] = corp['word'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
    return corp
#y = organize_articles_by_date(x)


# In[ ]:


#import textblob as tb
#pol = tb.TextBlob(y.iloc[0][0])
#pol
#y.keys()


# In[ ]:


def apply_sentiments(tokens, method):
    if method == 'bing':
        bing = pd.read_csv("/Users/dylanedmonds/Documents/data/BDAI/Data sets/bing_sents.csv")
        corp_sents = tokens.merge(bing, on = 'word',how  = 'inner', right_index = True)
        corp_sents.drop('Unnamed: 0', axis = 1, inplace = True)
    if method == 'afinn':
        afinn = pd.read_csv("/Users/dylanedmonds/Documents/data/BDAI/Data sets/afinn_sents.csv")
        corp_sents = tokens.merge(afinn, on = 'word',how  = 'inner', right_index = True)
        corp_sents.drop('Unnamed: 0', axis = 1, inplace = True)
    if method == 'loughran':
        loughran = pd.read_csv("/Users/dylanedmonds/Documents/data/BDAI/Data sets/loughran_sentiments.csv")
        corp_sents = corp.merge(loughran, on = 'word',how  = 'inner', right_index = True)
        corp_sents.drop('Unnamed: 0', axis = 1, inplace = True)
    return corp_sents
#x = apply_sentiments(x, method = 'bing')
#y = apply_sentiments(x, method = 'afinn')
#x


# In[ ]:


def create_sent_score(df, method):
    if method == 'bing':
        corp_sents_count = df.groupby([df.index, df.sentiment]).count()
        corp_sents_count.reset_index(inplace = True)
        corp_sents_count = corp_sents_count.pivot(index = 'Date',columns='sentiment', values='word')
        corp_sents_count.fillna(value = 0, inplace = True)
        corp_sents_count['bing_Score'] = corp_sents_count.groupby(['positive', 'negative'], group_keys = False).apply(lambda g: (g['positive'] - g['negative'])/ (g['positive']+g['negative']))
    if method == 'afinn':
        corp_sents_count= df.groupby([df.index, df.score]).count()
        corp_sents_count.reset_index(inplace = True)
        corp_sents_count.fillna(value = 0, inplace = True)
        corp_sents_count['score*word'] = corp_sents_count.groupby(['score', 'word'],group_keys=False).apply(lambda g: (g.score* g.word))
        corp_sents_count = corp_sents_count.groupby(corp_sents_count['Date'])['score*word'].sum()
        corp_sents_count = pd.DataFrame(corp_sents_count)
        corp_sents_count.rename(columns = {'score*word':"afinn_Score"}, inplace = True)
    if method =='loughran':
        corp_sents_count = df.groupby([df.index, df.sentiment]).count()
        corp_sents_count.reset_index(inplace = True)
        corp_sents_count = corp_sents_count.pivot(index = 'Date',columns='sentiment', values='word')
        corp_sents_count.fillna(value = 0, inplace = True)
        corp_sents_count['loughran_Score'] = corp_sents_count.groupby(['positive', 'negative'], group_keys = False).apply(lambda g: (g['positive'] - g['negative'])/ (g['positive']+g['negative']))
    return corp_sents_count
#x = create_sent_score(df = x,  method = 'bing')
#y = create_sent_score(df = y,  method = 'afinn')



# In[ ]:


def create_sentiments(company, source, method = 'bing', key = "79613011d94745a89fbd3067a42c9c0f"):
    try:
        df = extract_articles(company, source)
        df_clean = clean_df(df)
        if method == 'textblob':
            df_tokens = organize_articles_by_date(df_clean)
        else:
            df_tokens = organize_tokens_by_date(df_clean)
        if (method == 'bing' or method == 'afinn'):
            df_sents = apply_sentiments(df_tokens, method)
            df_score = create_sent_score(df_sents,  method)
            return df_score
        else:
            return df_tokens
    except: 
        return print('Error: Could not find sentiments. Try another general search term.')
#create_sentiments(company = 'Tesla', source = 'cnn', method = 'bing')


# In[ ]:


def merge_sentiment_sources_sm(company, method):
    x_cnn = create_sentiments(company, 'cnn', method)
    x_bi = create_sentiments(company, 'business-insider', method)
    x_bbc = create_sentiments(company, 'bbc-news', method)
    x_bloom = create_sentiments(company, 'bloomberg', method)
    frames = [x_cnn, x_bi, x_bbc, x_bloom]   
    x = pd.concat(frames)
    return x
#merge_sentiment_sources_sm('Tesla', 'bing')


# In[ ]:


def merge_sentiment_sources_lg(company, method):
    x_cnn = create_sentiments(company, 'cnn', method)
    x_bi = create_sentiments(company, 'business-insider', method)
    x_fox = create_sentiments(company, 'fox-news', method)
    x_bloom = create_sentiments(company, 'bloomberg', method)
    x_bbc = create_sentiments(company, 'bbc-news', method)
    x_cbc = create_sentiments(company, 'cbc-news', method)
    x_cnbc = create_sentiments(company, 'cnbc', method)
    x_cbs = create_sentiments(company, 'cbs-news', method)
    x_ft = create_sentiments(company, 'financial-times', method)
    x_fp = create_sentiments(company, 'financial-post', method)
    x_mir = create_sentiments(company, 'mirror', method)
    x_reu = create_sentiments(company, 'reuters', method)
    x_eco = create_sentiments(company, 'the-economist', method)
    frames = [x_cnn, x_bi, x_fox, x_bloom, x_bbc, x_cnbc, x_cbs, x_ft, x_fp, x_mir, x_reu, x_eco]
    x = pd.concat(frames)
    return x
#merge_sentiment_sources_lg('Tesla', 'bing')

