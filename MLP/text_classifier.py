from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


def size_mb(docs):
    return sum(len(s.encode("utf-8")) for s in docs) / 1e6


# Trains nearest neighbor classifier without using stop words and with for N features.
# Returns accuracies for both.
def train_nnc(features, train_data, test_data, y_train, y_test):
    vectorizer = CountVectorizer(max_features=features)
    vectorizer_stopwords = CountVectorizer(stop_words="english", max_features=features)
    X_train = vectorizer.fit_transform(train_data)
    X_train_stopwords = vectorizer_stopwords.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    X_test_stopwords = vectorizer_stopwords.transform(test_data)

    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=features)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_data).toarray()
    X_test_tfidf = tfidf_vectorizer.transform(test_data).toarray()

    nn_clf = NearestNeighbors(n_neighbors=1)
    nn_clf_stopwords = NearestNeighbors(n_neighbors=1)
    nn_clf_tfidf = NearestNeighbors(n_neighbors=1)

    nn_clf.fit(X_train, y_train)
    sklearn_nn_pred = nn_clf.kneighbors(X_test, n_neighbors=1, return_distance=False)
    acc = accuracy_score(y_test, y_train[sklearn_nn_pred])

    nn_clf_stopwords.fit(X_train_stopwords, y_train)
    sklearn_nn_pred_stopwords = nn_clf_stopwords.kneighbors(X_test_stopwords, n_neighbors=1, return_distance=False)
    acc_stopwords = accuracy_score(y_test, y_train[sklearn_nn_pred_stopwords])

    nn_clf_tfidf.fit(X_train_tfidf, y_train)
    sklearn_nn_pred_tfidf = nn_clf_tfidf.kneighbors(X_test_tfidf, n_neighbors=1, return_distance=False)
    acc_tfidf = accuracy_score(y_test, y_train[sklearn_nn_pred_tfidf])
    return acc, acc_stopwords, acc_tfidf


data_train = fetch_20newsgroups(subset='train')
data_test = fetch_20newsgroups(subset='test')

print(f'Total of {len(data_train.data)} posts in the dataset and the total size is {size_mb(data_train.data):.2f}MB')

y_train, y_test = data_train.target, data_test.target

accuracies = []
accuracies_stopwords = []
accuracies_tfidf = []
vocabularies = [10, 100, 1000, 1500, 2000, 3000]

for n in vocabularies:
    accuracy, stopwords_accuracy, tfidf_accuracy = train_nnc(n, data_train.data, data_test.data, y_train, y_test)
    accuracies.append(accuracy)
    accuracies_stopwords.append(stopwords_accuracy)
    accuracies_tfidf.append(tfidf_accuracy)

print("Accuracies without stop_words attribute", accuracies)
print("Accuracies with stop_words attribute", accuracies_stopwords)
print("Accuracies with stop_words attribute and TF-IDF", accuracies_tfidf)


plt.figure(figsize=(10, 6))

plt.bar([size - 20 for size in vocabularies], accuracies, width=20, color='blue', label='Without stopwords')
plt.bar([size for size in vocabularies], accuracies_stopwords, width=20, color='red', label='With stopwords')
plt.bar([size + 20 for size in vocabularies], accuracies_tfidf, width=20, color='green', label='With stopwords and tf-idf')

plt.xlabel('Vocabulary Size')
plt.ylabel('Vocabulary Accuracy')
plt.xticks(vocabularies, vocabularies)
plt.legend()

plt.tight_layout()
plt.show()

