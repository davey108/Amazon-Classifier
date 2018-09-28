import numpy
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import threading

result_arr = []


# test_review is each row of the test_matrix
def knn(distance_list, rating_arr, k):
    # threadLock.acquire()
    collected_distances = []
    i = 0
    for item in distance_list:
        collected_distances.append((rating_arr[i], item))
        i += 1
    collected_distances = sorted(collected_distances, key=lambda x: x[1], reverse=True)
    limited_dist_list = collected_distances[0:k]
    res = Counter(elem[0] for elem in limited_dist_list)
    # threadLock.release()
    return res.most_common(1)[0][0]


# Should return an array of knn results along with the thread id
def start_knn(train_m, test_m, ratings_ls, k, begin, end):
    rating_res = []
    for i in range(begin, end):
        print("Review: " + str(i) + "\n")
        similarity_list = cosine_similarity(test_m.getrow(i), train_m).tolist()
        rating = knn(similarity_list[0], ratings_ls, k)
        rating_res.append(rating)
    result_arr.append((threading.current_thread().name, rating_res))


def start_threads(train_m, test_m, rating_ls, k):
    global result_arr
    global empty_review_index
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

    result_arr = sorted(result_arr, key=lambda x: x[0])
    # layout of result arr = [(thread_id,knn_result),(thread_id,knn_result),...]
    result_list = []
    for item in result_arr:
        for rating in item[1]:
            # result_file.write(rating + "\n")
            result_list.append(rating)
    for index in range(len(empty_review_index)):
        result_list.insert(empty_review_index[index], "-1")
    result_file = open("result.txt", "w")
    for element in result_list:
        result_file.write(element + "\n")
    result_file.close()


start_time = time.time()
train_file = open("train/train.data", "r")
train_corpus = []
rating_labels = []
for review in train_file:
    text_line = review.split("\t")
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
# uses to stores index of empty review which is to be negative
empty_review_index = []
count_index = 0
for review in test_file:
    new_review = re.sub("[^a-zA-Z_' -]", "", review)
    # for empty reviews
    if new_review != "":
        test_corpus.append(new_review)
    else:
        empty_review_index.append(count_index)
    count_index += 1
test_file.close()

# Now transform the test data into a term document matrix using the learned vocabulary and idf from the train_data
test_matrix = tfidf_vectorizer.transform(test_corpus)
start_threads(train_matrix, test_matrix, rating_labels, 130)
# test_rows = numpy.shape(test_matrix)[0]
print("--------------%s minutes----------------" % ((time.time() - start_time)//60))


