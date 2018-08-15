library(ngram)
library(RWeka)
library(tm)
library(qdap)
library(tm.plugin.webmining)
library(ggplot2)
library(dplyr)
library(tidytext)

# download a corpus of data on the General Electric company from the Yahoo API
ge <- WebCorpus(YahooFinanceSource("GE"))

# Function made to clean a corupus using the tm map and several regex matching functions 
current_corpClean <- function(corp){
  # remove words like "can't"
  corp_clean <- tm_map(corp, replace_contraction) 
  # remove words like "Yr", "hr", "ft", "sec"
  corp_clean <- tm_map(corp, replace_abbreviation) 
  # remove all punctuation !! should be done after removing contractions
  corp_clean <- tm_map(corp, removePunctuation) 
  # remove unnecessary white space
  corp_clean <- tm_map(corp, stripWhitespace)
  # put all characters into lowe case form
  corp_clean <- tm_map(corp, content_transformer(tolower))
  return(corp_clean)
}

ge_clean <- current_corpClean(ge)

# create annonymous function to create unigrams
unagramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 1))
#
ge_terms <- ge_clean %>%
        TermDocumentMatrix(control = list(tokenize = unagramTokenizer)) %>%
        Terms() %>% # unlist tokens into a list
        as.data.frame() # convert that list to a  data frame for easy manipulation


names(ge_terms) <- "word"
ge_terms$word <- as.character(ge_terms$word)

# plot the most common words for each sentiment
ge_terms %>%
  count(word) %>%
  inner_join(get_sentiments("loughran"), by = "word") %>%
  group_by(sentiment) %>%
  top_n(5, n) %>%
  ungroup() %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_col() +
  coord_flip() +
  facet_wrap(~ sentiment, scales = "free") +
  ggtitle("Frequency of This Word in GE Articles") + theme_minimal()
# not useful because each word only occured one time in this case

# count the # of of occurences for each sentiment 
sentiment_fre <- ge_terms %>%
  inner_join(get_sentiments("loughran"), by = "word") %>%
  count(sentiment) 
sentiment_fre

# create a plot that will show the amount of occurences for each sentiment
ggplot(data = sentiment_fre)  + 
  geom_col(aes(sentiment, n, fill = sentiment)) +
  scale_fill_brewer(palette = "Paired")

## Process repeated for Tesla
TSLA_corpus <- WebCorpus(YahooFinanceSource("TSLA"))
tsla_clean <- current_corpClean(TSLA_corpus)

unagramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 1))
tdm.tsla <- TermDocumentMatrix(TSLA_corpus, control = list(tokenize = unagramTokenizer))
TSLA_Terms <- Terms(tdm.tsla); TSLA_Terms <- as.data.frame(TSLA_Terms); names(TSLA_Terms) <- "word"

sentiment_fre <- TSLA_Terms %>%
  inner_join(get_sentiments("loughran"), by = "word") %>%
  count(sentiment) 
sentiment_fre

sentiment_word_fre <- TSLA_Terms %>%
  inner_join(get_sentiments("loughran"), by = "word") %>%
  count(sentiment, word) %>% arrange(desc(n))
sentiment_word_fre

ggplot(data = sentiment_fre, aes(x = sentiment, y = n, fill = sentiment)) + 
  geom_col() + scale_fill_brewer(palette = "Set3")


