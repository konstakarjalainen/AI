from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
import timeit


def size_mb(docs):
    return sum(len(s.encode("utf-8")) for s in docs) / 1e6


# a
data_train = fetch_20newsgroups(subset='train')
data_test = fetch_20newsgroups(subset='test')
vectorizer = CountVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(data_train.data)
print(f'Size of the vocabulary is {len(vectorizer.get_feature_names_out())}')

X_test = vectorizer.transform(data_test.data)
y_train, y_test = data_train.target, data_test.target
print(X_train.shape)
print(y_train.shape)

# b Simple baseline
dummy_clf = DummyClassifier(strategy="most_frequent")
start_time = timeit.default_timer()
dummy_clf.fit(X_train, y_train)
dummy_clf_pred = dummy_clf.predict(X_test)
dummy_acc = accuracy_score(y_test, dummy_clf_pred)
stop_time = timeit.default_timer()
print("Simple baseline classification accuracy:", dummy_acc)
print("Computation time:", stop_time - start_time)


# g-h SKlearn Nearest Neighbor classifier
nn_clf = NearestNeighbors(n_neighbors=1)
start_time = timeit.default_timer()
nn_clf.fit(X_train, y_train)
sklearn_nn_pred = nn_clf.kneighbors(X_test, n_neighbors=1, return_distance=False)
sklearn_nn_accuracy = accuracy_score(y_test, y_train[sklearn_nn_pred])
sklearn_nn_time = timeit.default_timer() - start_time

print("Sklearn Nearest Neighbor Classifier Accuracy:", sklearn_nn_accuracy)
print("Sklearn Nearest Neighbor Classifier Computation Time:", sklearn_nn_time)


# e-f Custom Nearest neighbor classifier
def nearest_neighbor_classifier(X_train, X_test, y_train):
    predictions = []
    for test_sample in X_test:
        distances = []
        for train_sample in X_train:
            distance = ((train_sample - test_sample)**2).sum()
            distances.append(distance)
        best_index = distances.index(min(distances))
        predictions.append(y_train[best_index])
    return predictions


X_train_arr = X_train.toarray()
X_test_arr = X_test.toarray()
samples_five_percent = int(len(X_test_arr) * 0.001)
start_time = timeit.default_timer()
nn_clf_predictions = nearest_neighbor_classifier(X_train_arr, X_test_arr[:samples_five_percent], y_train)
nn_acc = accuracy_score(y_test[:samples_five_percent], nn_clf_predictions)
stop_time = timeit.default_timer()
print("Number of samples used:", samples_five_percent)
print("Custom nn classification accuracy:", nn_acc)
print("Estimated computation time:", (stop_time - start_time) / 0.001)




