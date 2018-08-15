library(tm.plugin.webmining)
library(purrr)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(stringr)
library(tidyr)
library(tidytext)

# function created to download files on the companies from the Yahoo AOU
download_articles <- function(Symbol) {
  WebCorpus(YahooFinanceSource(Symbol, params = list(s = Symbol, region = "US", 
                                                     lang = "en-US", startdate = '2015-01-01', end = '2015-01-28')))
}

# read in data set containing companies of interest
real_state <- read.csv("~/documents/data/BDAI/Data sets/realestate_landlords.csv", header = TRUE)
# use the data to download articles of interest
stock_articles <- real_state[real_state$Symbol != "",c("Company", "Symbol")] %>%
  mutate(corpus = map(Symbol, download_articles))

# uneest those tokens 
tokens <- stock_articles %>%
  unnest(map(corpus, tidy)) %>%
  unnest_tokens(word, text) %>%
  select(Company, datetimestamp, word, id, heading)

# tf-idf gives weights of importance to words based on how common they are
article_tf_idf <- tokens %>%
  count(Company, word) %>%
  filter(!str_detect(word, "\\d+")) %>%
  bind_tf_idf(word, Company, n) %>%
  arrange(-tf_idf)
article_tf_idf

plot_article <- article_tf_idf %>%
  arrange(desc(tf_idf)) %>%
  mutate(word = factor(word, levels = rev(unique(word))))

# plot the words that have the most impact on a company based on a term frequency inverse document frequency
plot_article %>% 
  top_n(10) %>%
  ggplot(aes(word, tf_idf, fill = Company)) +
  geom_col() +
  labs(x = NULL, y = "tf-idf") +
  coord_flip() + theme_minimal() + ggtitle('Highest tf_idf Words for Each Company')+
  scale_fill_brewer(palette = "Spectral")

# show if a word contributed positively or negativley and by how much
tokens %>%
  anti_join(stop_words, by = "word") %>%
  count(word, id, sort = TRUE) %>%
  inner_join(get_sentiments("afinn"), by = "word") %>%
  group_by(word) %>%
  summarize(contribution = sum(n * score)) %>%
  top_n(15, abs(contribution)) %>%
  mutate(word = reorder(word, contribution)) %>%
  ggplot(aes(word, contribution, fill = contribution > 0)) +
  geom_col() +
  coord_flip() +
  ggtitle('Frequency of Words AFINN Score') + theme_minimal()   

t1 <- tokens %>%
  count(word) %>%
  inner_join(get_sentiments("loughran"), by = "word") %>%
  group_by(sentiment) %>%
  top_n(5, n) %>%
  ungroup() %>%
  mutate(word = reorder(word, n)) 
  
# most common words in all the articles compared the sentiment they are associated with
ggplot(data = t1, aes(word, n, fill = sentiment)) +
  geom_col() +
  coord_flip() +
  facet_wrap(~sentiment, scales = "free") +
  ggtitle("Frequency of This Word in Google Finance Articles") + theme_minimal() + 
  theme(legend.position = "none") +  scale_fill_brewer(palette = "Spectral")

# inner-join our company term list with the sentiment list based on the 'word' column 
# and count how man of each type of sentiment the 
 sentiment_fre <- tokens %>%
  inner_join(get_sentiments("loughran"), by = "word") %>%
  count(sentiment, Company) %>%
  spread(sentiment, n, fill = 0)
sentiment_fre

# take the 'sentiment_fre' object and give it a sentiment score based on the ratio of positive to 
# negative words
sentiment_fre %>%
  mutate(score = (positive - negative) / (positive + negative)) %>%
  mutate(company = reorder(Company, score)) %>%
  ggplot(aes(company, score, fill = score > 0)) +
  geom_col(show.legend = FALSE) + coord_flip() + 
  theme_minimal() + ggtitle('Positive or Negative Scores Among Recent Google Finance Articles')

  
       