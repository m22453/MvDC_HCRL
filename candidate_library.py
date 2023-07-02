# -*- coding: utf-8 -*-
# @Time : 2021/9/19 17:22
# @Author : ruinabai_TEXTCCI
# @FileName: candidate_library.py
# @Email : m15661362714@163.com
# @Software: PyCharm

# @Blog ：https://www.jianshu.com/u/3a5783818e3a

from gensim import corpora, models
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import json


def no_semantic_view_chinese_news(dic_lines, input_file=None, encoding='utf-8'):
    """Reads a json file."""
    if input_file:
        dic_lines = []
        with open(input_file, 'r', encoding=encoding) as f:
            while True:
                line = f.readline()
                if not line:  # 到 EOF，返回空字符串，则终止循环
                    break
                js = json.loads(line)
                dic_lines.append(js)

    view_lst, view_text_lst, view_title_lst, labels = [], [], [], []
    # random.seed(123)
    # random.shuffle(dic_lines)
    lib_dic = dict()
    edit_dic = dict()
    # use title or text for converting
    for (_, line) in enumerate(dic_lines):
        title = line['title']
        label = line['label']
        text = line['text']
        edit = line['edit']
        for e in edit:
            if e not in edit_dic.keys():
                edit_dic[e] = [label]
            else:
                if label not in edit_dic[e]:
                    edit_dic[e].append(label)
        view_lst.append(edit)
        view_text_lst.append(text)
        view_title_lst.append(title)
        labels.append(label)
        for ed in edit:
            tmp_text = text.split("。")[0]+"。"
            if ed not in lib_dic.keys():
                lib_dic[ed] = [title]
                # lib_dic[ed] = [tmp_text]
            else:
                lib_dic[ed].append(title)
                # lib_dic[ed].append(tmp_text)

    arr = []
    for v in lib_dic.values():
        arr.append(len(v))
    print('max legth:', max(arr))
    # target view, labels, library, _, text view, title view
    return view_lst, labels, lib_dic, edit_dic, view_text_lst, view_title_lst


def modeling_discrete_view(length, view_lst, labels, lib, show=True, create=True, output_file='./data/chinese_news/convert_view.txt'):
    """ length of convertion"""
    if create == False:
        # get the semantic view representation already
        import jieba
        words_ls = []
        for text in view_lst:
            words = [w for w in jieba.cut(text)]
            words_ls.append(words)
        view_lst = words_ls


    # Build a corpus dictionary with text data of cutted words
    dictionary = corpora.Dictionary(view_lst)
    feature_cnt = len(dictionary.token2id.keys())
    print('features cnt', feature_cnt)
    corpus = [dictionary.doc2bow(text) for text in view_lst]

    # tfidf transformer
    tfidf = models.TfidfModel(corpus, normalize=False)



    if create:
        # trans from mini library
        view_list_trans = []


        for lst in tfidf[corpus]:
            sorted_l = sorted(lst, key=lambda t: t[1], reverse=True)
            # choose the top 2 in no semantic view / can regard as a Prior Knowledge
            sorted_cropus = [item[0] for item in sorted_l[:2]]
            # print(sorted_cropus,)

            view_str = ''
            for top_c in sorted_cropus:
                n_view = dictionary[top_c]
                # print(n_view)
                # --------------------------------- set the anchor max number
                max_n = int(np.floor(np.sqrt(len(labels) / len(set(labels)))))
                max_n = min(max_n, length)
                v = lib[n_view]
                if len(v) > max_n:
                    s = np.random.choice(len(v), size=max_n, replace=False)
                    v_arr = np.array(v)
                    v_ = list(v_arr[s])
                    # print(v_)

                else:
                    v_ = v
                    # print(v_)

                view_str = ' '.join(v_)
            view_list_trans.append(view_str)



    if create:
        filename = output_file
        with open(filename, 'w', encoding='utf8') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
            for tm in view_list_trans:
                f.write(tm+'\n')
        return view_list_trans, labels



# a, b, c, _, text, title = read_json('./data/chinese_news/fileOf.json')
# c, d = modeling_discrete_view(a, b, c, True, True)
# print('====================================')
# modeling_discrete_view(title, d, c,  True, False)
#
# filename = 'write_data.txt'
# with open(filename, 'r', encoding='utf8') as f:
#     new_title = f.readlines()
#
#
# import jieba
# def jieba_tokenize(text):
#     return jieba.lcut(text)
#
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# tfidf_vectorizer = TfidfVectorizer(tokenizer=jieba_tokenize, lowercase=False)
# '''
# tokenizer: 指定分词函数
# lowercase: 在分词之前将所有的文本转换成小写，因为涉及到中文文本处理，
# 所以最好是False
# '''
#
#
# # 需要进行聚类的文本集
#
# tfidf_matrix = tfidf_vectorizer.fit_transform(new_title)
#
# num_clusters = 4
# km_cluster = KMeans(n_clusters=num_clusters, random_state=44)
#
# result = km_cluster.fit_predict(tfidf_matrix)
# print(normalized_mutual_info_score(np.array(b), result))

# np.random.seed(0)
# p = np.array([0.1, 0.0, 0.7, 0.2])
# indexs = [np.random.choice([0, 1, 2, 3], p=p.ravel()) for _ in range(10)]
# print(indexs)

