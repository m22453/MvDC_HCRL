# -*- coding: utf-8 -*-
# @Time : 2021/10/20 14:31
# @Author : ruinabai_TEXTCCI
# @FileName: preprocessing.py
# @Email : m15661362714@163.com
# @Software: PyCharm

# @Blog ：https://www.jianshu.com/u/3a5783818e3a

# Data Preprocessing
import re, os
import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter


# Data Cleaning
def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", str(text))
    text = ' '.join(text.split())
    return text

# function to remove stopwords
def remove_stopwords(text, words_type='english'):
    if words_type == 'english':
        text = text.lower()
        stop_words = set(stopwords.words(words_type))
        no_stopword_text = [w for w in text.split() if w not in stop_words]

    elif words_type == 'chinese':
        cn_path = os.path.expanduser('./data/baidu_stopwords.txt')
        with open(cn_path, 'r', encoding='UTF-8') as f:
            stop_words = f.readlines()
        stop_words = [c.strip() for c in stop_words]
        no_stopword_text = [w for w in jieba.cut(text) if w not in stop_words]

    elif words_type == 'name':
        # for name in no-semantic view
        return text

    return ' '.join(no_stopword_text)


# word Stemming
def stemming(text):
    snowball_stemmer = SnowballStemmer('english')
    stem = [snowball_stemmer.stem(w) for w in text.split()]
    return ' '.join(stem)

# Word lemmatization
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    lem = [lemmatizer.lemmatize(w) for w in text.split()]
    return ' '.join(lem)

def filter_low_term(view_texts):
    # new corpus by filtering low terms
    #bbc mini_news 4 chinese_news 3 news 5 ag_news7.6k 4  ag_news65k 10 ag_news12w 15
    res = []

    text = ' '.join(view_texts)
    word2frequency = dict(Counter(text.split()))
    low_term_set = set()
    for w, f in word2frequency.items():
        if f < 3:
            low_term_set.add(w)
    for sub_text in view_texts:
        n_text = [w for w in sub_text.split() if w not in low_term_set]
        res.append(' '.join(n_text))
    return res


def vectorize_text(corpus):
    # transformer for texts
    vectorizer = CountVectorizer()
    # vectorize by tf
    X = vectorizer.fit_transform(corpus)
    # keywords
    word = vectorizer.get_feature_names()
    print('corpus length:', len(word))

    # vectorize by tf-idf
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    tfidf_arr = tfidf.toarray()

    return tfidf_arr

def preprocessing(text_list, type):
    """
    Approach for text preprocessing

    lower casing
    removal of punctuations and numbers
    remove white space
    remove stops
    lemmitization word
    """

    if type == 'english':

        text_list = [clean_text(text) for text in text_list]

    text_list = [remove_stopwords(text, words_type=type) for text in text_list]

    if type == 'english':

        text_list = [stemming(text) for text in text_list]

        text_list = [lemmatization(text) for text in text_list]

    text_list = filter_low_term(text_list)

    tfidf_arr = vectorize_text(text_list)

    return tfidf_arr

def save_mat(output_path, save_dict):
    import scipy.io as scio
    scio.savemat(output_path, save_dict)


# if __name__ == "__main__":
#     from processor import *
#     processor = AirlineProcessor()
#     dataset = 'ag_news_all'
#     data_dir = './data/ag_news/train'
#     label_list = processor.get_labels()
#     print(label_list)

#     examples_views = processor.get_train_examples(data_dir)
#     x0, y = preprocessing(examples_views[0])
#     x1, _ = preprocessing(examples_views[1])
#     print('num of samples', len(x0))
#     y = [label_list.index(l) for l in y]
#     path ='./data/MAT/'+ dataset +'.mat'
#     dic = {'x0': x0, 'x1': x1, 'y': y}
#     save_mat(path, dic)
#     pass

