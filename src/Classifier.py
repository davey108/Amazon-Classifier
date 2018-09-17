import numpy
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import threading

result_arr = []
# numpy.seterr(invalid='ignore')
# def cosine_similarity(vector1, vector2):
#    dot_product = numpy.dot(vector1.toarray(), vector2.toarray())
#    # calculate the norm aka magnitude
#    vect1_magnitude = numpy.linalg.norm(vector1)
#    vect2_magnitude = numpy.linalg.norm(vector2)
#    return dot_product/(vect1_magnitude*vect2_magnitude)


def cosine_similarity(vector_1, vector_2):
    vector_1 = vector_1.toarray()
    vector_2 = vector_2.toarray()
    return numpy.dot(vector_1, vector_2.T) / \
        (numpy.sqrt(numpy.dot(vector_1, vector_1.T)) * numpy.sqrt(numpy.dot(vector_2, vector_2.T)))


# test_review is each row of the test_matrix
def knn(train_data, test_review, rating_arr, k):
    collected_distances = []
    train_rows = numpy.shape(train_matrix)[0]
    # get the cosine similarity of each review to each of the review in train_set
    for num_row in range(train_rows):
        similarity_dist = cosine_similarity(train_data.getrow(num_row), test_review)
        collected_distances.append((rating_arr[num_row], similarity_dist))
    collected_distances = sorted(collected_distances, key=distance, reverse=True)
    limited_dist_list = collected_distances[0:k]
    res = Counter(elem[0] for elem in limited_dist_list)
    print(res.most_common(2))
    return res.most_common(1)[0][0]
    # get values of k neighbors
    # count_neg = 0
    # count_pos = 0
    # for i in range(k):
    #     if collected_distances[i][0] == "+1":
    #         count_pos += 1
    #     else:
    #         count_neg += 1
    # if count_pos >= count_neg:
    #     return "+1"
    # else:
    #     return "-1"


def distance(element):
    return element[1]


# Should return an array of knn results along with the thread id
def start_knn(train_m, test_m, ratings_ls, k, begin, end):
    rating_res = []
    for i in range(begin, end):
        rating = knn(train_m, test_m.getrow(i), ratings_ls, k)
        rating_res.append(rating)
    result_arr.append((threading.current_thread().name, result_arr))


def start_threads(train_m, test_m, rating_ls, k):
    global result_arr
    rows_test = numpy.shape(test_matrix)[0]
    threads = []
    n = 4
    range_1 = rows_test//n
    range_2 = 2*(rows_test//n)
    range_3 = 3*(rows_test//n)
    ranges = [range_1, range_2, range_3, rows_test]
    for i in range(n):
        start_range = 0
        if i > 0:
            start_range = ranges[i-1]
        thread_n = threading.Thread(target=start_knn, args=(train_m, test_m, rating_ls, k, start_range, ranges[i]))
        # set a thread name for identifying
        thread_n.name = i
        thread_n.start()
        threads.append(thread_n)

    for t in threads:
        t.join()
    result_file = open("result.txt", "w")
    result_arr = sorted(result_arr, key=lambda x: x[0])
    # layout of result arr = [(thread_id,knn_result),(thread_id,knn_result),...]
    for item in result_arr:
        for rating in item[1]:
            result_file.write(rating)
    result_file.close()


train_file = open("train/train.data", "r")
train_corpus = []
rating_labels = []
for review in train_file:
    text_line = review.split("\t")
    # review
    review = text_line[1]
    # rid of all the numbers and special symbols
    new_review = re.sub("[^a-zA-Z_' -]", "", review)
    # for empty reviews
    if new_review != "":
        rating_labels.append(text_line[0])
        train_corpus.append(new_review)
train_file.close()

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
# Learn vocabulary and idf and return term document matrix
train_matrix = tfidf_vectorizer.fit_transform(train_corpus)

# parse the test data
test_file = open("test/test.data", "r")
test_corpus = []
for review in test_file:
    new_review = re.sub("[^a-zA-Z_' -]", "", review)
    # for empty reviews
    if new_review != "":
        test_corpus.append(new_review)
test_file.close()

# Now transform the test data into a term document matrix using the learned vocabulary and idf from the train_data
test_matrix = tfidf_vectorizer.transform(test_corpus)
test_rows = numpy.shape(test_matrix)[0]

# Make the ranges for n threads

# result_file = open("result.txt", "w")
# for row in range(test_rows):
#     rating = knn(train_matrix, test_matrix.getrow(row), rating_labels, 130)
#     print(str(row) + ":" + rating)
#     result_file.write(rating)
# result_file.close()

start_threads(train_matrix, test_matrix, rating_labels, 100)





