
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as mlp
import pandas as pd
import os
#os.chdir('desktop') #set this to the file that sentiment_finder file is located
import sentiment_finder as sf
x = sf.create_sentiments(company = 'Toyota', source = 'cnn', method = 'bing')
y = sf.create_sentiments(company = 'Toyota', source = 'cnn', method = 'afinn')
edit = pd.concat([x.bing_Score, y.afinn_Score], axis=1)
edit = (edit - edit.mean()) / (edit.max() - edit.min())
edit.bing_Score.corr(edit.afinn_Score)
mlp.plot(edit.index,edit.afinn_Score)
mlp.plot(edit.index, edit.bing_Score)


from datetime import datetime
import quandl
quandl.ApiConfig.api_key = "<insert key>"
gm = sf.merge_sentiment_sources('Post-holdings', 'bing')
gm1= sf.merge_sentiment_sources('Post-holdings', 'afinn')


edit2 = pd.concat([gm.bing_Score, gm1.afinn_Score], axis = 1)
edit2 = edit2.reset_index().sort_values('Date').set_index('Date')



def normalise_df(df):
    df = (df - df.mean()) / (df.max() - df.min())
    return df


norm = normalise_df(edit2)


mlp.plot(norm.index,norm.bing_Score)
mlp.plot(norm.index, norm.afinn_Score)
norm.bing_Score.corr(norm.afinn_Score)


import quandl
post = quandl.get('WIKI/POST', start_date = '2017-11-28', end_date = '2018-02-24')


post_join = post.join(edit2,how = 'left')[['Adj. Close','bing_Score', 'afinn_Score']]
bing_interpolate = pd.DataFrame(post_join.bing_Score.interpolate())
afinn_interpolate = pd.DataFrame(post_join.afinn_Score.interpolate())


data = bing_interpolate.join(afinn_interpolate,how='left')
def normalise_df(df):
    df = (df - df.mean()) / (df.max() - df.min())
    return df
data = normalise_df(data).shift(-2)

data = data.join(post, how = 'left')[['bing_Score', 'afinn_Score', 'Adj. Close']]

data.afinn_Score.corr(data['Adj. Close'])

