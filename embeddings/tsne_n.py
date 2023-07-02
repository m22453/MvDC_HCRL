# -*- coding: utf-8 -*-
# @Time : 2022/12/16 16:14
# @Author : ruinabai_TEXTCCI
# @FileName: tsne_n.py
# @Email : m15661362714@163.com
# @Software: PyCharm

# @Blog ：https://www.jianshu.com/u/3a5783818e3a


from sklearn.manifold import TSNE
import numpy as np
import torch
import os
root_path = './bbc/'

title_dicts = {
    'cl_data_1': 'v1-cl-data',
    'cl_data_2': 'v2-cl-data',
    'cl_task_1': 'v1-cl-task',
    'cl_task_2': 'v2-cl-data',
    'input_1': 'v1-input',
    'input_2': 'v2-input',
    'cl_final': 'v12-final'
}

target = np.load(root_path + 'all_y.npy').reshape((-1, 1))
print(target.shape)

for file_name in os.listdir(root_path):
    if file_name == 'all_y.npy':
        continue

    print(file_name)

    data = np.load(root_path+file_name, allow_pickle=True)

    print(data.shape)

    # X_tsne = TSNE(n_components=2, random_state=44).fit_transform(data.reshape(-1, 1))
    # print(set(target))
    #
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    #
    # title_str = title_dicts[file_name.split('.')[0]]
    # plt.figure(figsize=(10, 8))
    # sns.scatterplot(x=X_tsne[:, 0],
    #                 y=X_tsne[:, 1],
    #                 hue=target,
    #                 # style=target,
    #                 palette="Set2",
    #                 legend=False
    #                 )
    #
    # plt.title(title_str)
    # plt.tight_layout()  # 调整整体空白
    # plt.savefig(root_path + title_str + ".png", format='png', dpi=150)
