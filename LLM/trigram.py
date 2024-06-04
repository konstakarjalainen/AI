import re
from nltk import FreqDist
from nltk.util import ngrams

# Preprocess the text
text = "Your text goes here. It can be as long as you want."
text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
text = text.lower()  # Convert to lowercase
tokens = text.split()  # Tokenize into word
# Count the frequency of each trigram
n = 3  # Trigrams
trigrams = ngrams(tokens, n)
freq_dist = FreqDist(trigrams)

# Count the total number of trigrams
total_trigrams = len(list(trigrams))

# Calculate the probability of the trigram
trigram = ("might", "be", "long")
probability = freq_dist[trigram] / total_trigrams

print(probability)