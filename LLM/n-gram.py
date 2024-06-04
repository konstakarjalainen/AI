import re
from nltk import FreqDist
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

phrase = ["might", "be", "long"]
STOPWORDS = ["a", "an", "and", "as", "at", "for", "from", "in", "into", "of", "on", "or", "the", "to"]
with open("TheStoryofAnHour-KateChopin.txt", 'r') as file:
    file_text = file.read()
    text = file_text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words_without_stopwords = [word for word in word_tokenize(text) if word not in STOPWORDS]
    
trimmed_list = []
for word in words_without_stopwords:
    if len(word) > 1:
        trimmed_list.append(word)
print(trimmed_list)
print(len(trimmed_list))
trigrams = list(ngrams(trimmed_list, 3))
freq_dist = FreqDist(trigrams)

# Count the total number of trigrams
total_trigrams = len(trigrams)
print(total_trigrams)

# Calculate the probability of the trigram
trigram = ("might", "be", "long")
print(freq_dist[trigram])
probability = freq_dist[trigram] / (total_trigrams - 2)
print(probability)