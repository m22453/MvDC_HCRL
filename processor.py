# -*- coding: utf-8 -*-
# @Time : 2021/8/31 15:16
# @Author : ruinabai_TEXTCCI
# @FileName: Processor.py
# @Email : m15661362714@163.com
# @Software: PyCharm

# @Blog ：https://www.jianshu.com/u/3a5783818e3a

import csv
import logging
import json, os
import random
from candidate_library import *
from preprocessing import preprocessing, save_mat

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
random.seed(123)

GLOBAL_LENGTH = 5

# for convert view length
print('GLOBAL_LENGTH ', GLOBAL_LENGTH)

def warn(*args, **kwargs):
    pass
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file, encoding='utf-8'):
        """Reads a json file."""
        dic_lines = []
        with open(input_file, 'r', encoding=encoding) as f:
            while True:
                line = f.readline()
                if not line:  # 到 EOF，返回空字符串，则终止循环
                    break
                js = json.loads(line)
                dic_lines.append(js)
        return dic_lines

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a , separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=",", quotechar=quotechar)
            lines = []
            for idx, line in enumerate(reader):
                if idx == 0:
                    continue
                line = [line[0], line[1], line[2]]
                lines.append(line)
            return lines

    @classmethod
    def _read_bbc_file(cls, path='./data/bbc'):
        " :return {label,title,text} dic"
        res_lst = []
        dir_lst = os.listdir(path)
        for dir in dir_lst:
            sub_path = path + '/' + dir
            sub_dir_lst = os.listdir(sub_path)
            for sub_dir in sub_dir_lst:
                full_path = sub_path + '/' + sub_dir
                write_line = dict()
                write_line['label'] = dir
                with open(full_path, 'r', encoding='unicode_escape') as f:
                    text_str = ''
                    for i, line in enumerate(f.readlines()):
                        if i == 0 and line != '\n':
                            write_line['title'] = line.replace('\n', '')
                        elif line == '\n':
                            continue
                        else:
                            text_str += line.replace('\n', '') + ' '
                    write_line['text'] = text_str
                    res_lst.append(write_line)
        return res_lst

class BBCProcessor(DataProcessor):
    """Processor for the News data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples_for_mv(
            self._read_bbc_file(), 'train'
        )

    def get_labels(self):
        """See base class."""
        return ['entertainment', 'business', 'politics', 'sport', 'tech']

    def _create_examples_for_mv(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = [[], []]
        texts = [[], []]
        y = []
        random.shuffle(lines)
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_headline = line['title']
            text_description = line['text']
            label = line['label']
            y_label = self.get_labels().index(label)

            headline_example = InputExample(guid=guid, text_a=text_headline, text_b=None, label=label)
            description_example = InputExample(guid=guid, text_a=text_description, text_b=None, label=label)
            examples[0].append(headline_example)
            examples[1].append(description_example)

            texts[0].append(text_headline)
            texts[1].append(text_description) # main view
            y.append(y_label)

        x0 = preprocessing(texts[0], 'english')
        x1 = preprocessing(texts[1], 'english')
        dic = {'views':[x0, x1], 'label':y}

        tmp = {'x0':x0, 'x1': x1, 'y': y}
        path ='./data/MAT/bbc.mat'
        save_mat(path, tmp)

        return examples, dic


class MiniNewsProcessor(DataProcessor):
    """Processor for the News data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples_for_mv(
            self._read_json(data_dir + ".json"), 'train'
        )

    def get_labels(self):
        """See base class."""
        return ['LATINO VOICES', 'ENVIRONMENT', 'EDUCATION']

    def _create_examples_for_mv(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = [[], []]
        texts = [[], []]
        y = []
        random.shuffle(lines)
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_headline = line['headline']
            text_description = line['short_description']
            label = line['category']
            y_label = self.get_labels().index(label)

            headline_example = InputExample(guid=guid, text_a=text_headline, text_b=None, label=label)
            description_example = InputExample(guid=guid, text_a=text_description, text_b=None, label=label)
            examples[0].append(headline_example)
            examples[1].append(description_example)

            texts[0].append(text_headline)
            texts[1].append(text_description) # main view
            y.append(y_label)

        x0 = preprocessing(texts[0], 'english')
        x1 = preprocessing(texts[1], 'english')
        dic = {'views':[x0, x1], 'label':y}

        # tmp = {'x0':x0, 'x1': x1, 'y': y}
        # path ='./data/MAT/mini_news.mat'
        # save_mat(path, tmp)

        return examples, dic


class ChineseNewsProcessor(DataProcessor):
    """Processor for the News data set."""

    def get_train_examples(self, data_dir, trans=True):
        if trans:
            """ for view convert"""
            return self._create_examples_for_mv_enchanced(
                self._read_json(data_dir + ".json"), 'train'
            )
        else:
            """for view feature [label, text]"""
            return self._create_examples_for_mv(
                self._read_json(data_dir + ".json"), 'train'
            )

    def get_labels(self):
        """See base class."""
        return ['文化', '军事', '科技', '体育']

    def _create_examples_for_mv(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = [[], []]
        texts = [[], []]
        y = []
        random.shuffle(lines)
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_headline_o = line['title']
            text_description = line['text']
            label = line['label']
            y_label = self.get_labels().index(label)

            headline_example = InputExample(guid=guid, text_a=text_headline_o, text_b=None, label=label)
            description_example = InputExample(guid=guid, text_a=text_description, text_b=None, label=label)
            examples[0].append(headline_example)
            examples[1].append(description_example)

            texts[0].append(text_headline_o)
            texts[1].append(text_description) # main view
            y.append(y_label)

        x0 = preprocessing(texts[0], 'chinese')
        x1 = preprocessing(texts[1], 'chinese')
        dic = {'views':[x0, x1], 'label':y}

        # tmp = {'x0':x0, 'x1': x1, 'y': y}
        # path ='./data/MAT/chinese_news.mat'
        # save_mat(path, tmp)
        return examples, dic

    def _create_examples_for_mv_enchanced(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = [[], []]
        texts = [[], []]
        y = []
        random.shuffle(lines)
        # transform part from no semantic view
        view_lst, labels, lib_dic, edit_dic, view_text_lst, view_title_lst = \
            no_semantic_view_chinese_news(dic_lines=lines)
        view_list_trans, _ = modeling_discrete_view(GLOBAL_LENGTH, view_lst, labels, lib_dic, False, True, './data/chinese_news/convert_view.txt')
        logging.info('!!convert success!!')
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            #text_headline = line['title']
            text_headline = view_list_trans[i] # trans use the edit + context (text)
            text_edits = line['edit']
            text_edits = ' '.join(text_edits)
            
            text_description = line['text'] # context
            label = line['label']
            y_label = self.get_labels().index(label)

            headline_example = InputExample(guid=guid, text_a=text_headline, text_b=None, label=label)
            description_example = InputExample(guid=guid, text_a=text_description, text_b=None, label=label)
            examples[0].append(headline_example)
            examples[1].append(description_example)

            texts[0].append(text_edits) # non-contextual view
            texts[1].append(text_description) # main view
            y.append(y_label)

        x0 = preprocessing(texts[0], 'name')
        x1 = preprocessing(texts[1], 'chinese')
        dic = {'views':[x0, x1], 'label':y}

        # tmp = {'x0':x0, 'x1': x1, 'y': y}
        # path ='./data/MAT/chinese_news_edit.mat'
        # save_mat(path, tmp)
        return examples, dic

class ChineseNewsTripleProcessor(DataProcessor):
    """Processor for the News data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples_for_mv(
            self._read_json(data_dir + ".json"), 'train'
        )

    def get_labels(self):
        """See base class."""
        return ['文化', '军事', '科技', '体育']

    def _create_examples_for_mv(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = [[], [], []]
        texts = [[], [], []]
        y = []
        
        random.shuffle(lines)
        # transform part from no semantic view
        view_lst, labels, lib_dic, edit_dic, view_text_lst, view_title_lst = \
            no_semantic_view_chinese_news(dic_lines=lines)
        view_list_trans, _ = modeling_discrete_view(GLOBAL_LENGTH,view_lst, labels, lib_dic, False, True, './data/chinese_news/convert_view.txt')
        logging.info('!!convert success!!')
        

        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_headline_o = line['title'] # headline
            text_headline = view_list_trans[i] # trans use the edit + context (text)

            # for data generating
            text_edits = line['edit']
            text_edits = ' '.join(text_edits)

            text_description = line['text'] # context
            label = line['label']
            y_label = self.get_labels().index(label)

            headline_example_o = InputExample(guid=guid, text_a=text_headline_o, text_b=None, label=label)
            headline_example = InputExample(guid=guid, text_a=text_headline, text_b=None, label=label)
            description_example = InputExample(guid=guid, text_a=text_description, text_b=None, label=label)

            examples[0].append(headline_example_o)
            examples[1].append(headline_example)
            examples[2].append(description_example)

            texts[0].append(text_headline_o)
            texts[1].append(text_edits)
            texts[2].append(text_description)
            y.append(y_label)

        x0 = preprocessing(texts[0], 'chinese')
        x1 = preprocessing(texts[1], 'name')
        x2 = preprocessing(texts[2], 'chinese')
        dic = {'views':[x0, x1, x2], 'label':y}

        # tmp = {'x0':x0, 'x1': x1, 'x2': x2, 'y': y}
        # path ='./data/MAT/chinese_news_3.mat'
        # save_mat(path, tmp)

        return examples, dic

class AgProcessor(DataProcessor):
    """Processor for the ag_news data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples_for_mv(
            self._read_csv(data_dir + ".csv"), 'test'
        )

    def get_labels(self):
        """See base class."""
        return ['1', '2', '3', '4']

    def _create_examples_for_mv(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = [[], []]
        texts = [[], []]
        y = []
        random.shuffle(lines)
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_v1 = line[1]
            text_v2 = line[-1]
            label = line[0]
            y_label = self.get_labels().index(label)

            headline_example = InputExample(guid=guid, text_a=text_v1, text_b=None, label=label)
            description_example = InputExample(guid=guid, text_a=text_v2, text_b=None, label=label)
            examples[0].append(headline_example)
            examples[1].append(description_example)

            texts[0].append(text_v1)
            texts[1].append(text_v2) # main view
            y.append(y_label)

        x0 = preprocessing(texts[0], 'english')
        x1 = preprocessing(texts[1], 'english')
        dic = {'views':[x0, x1], 'label':y}

        tmp = {'x0':x0, 'x1': x1, 'y': y}
        if len(y) > 10000:
            path ='./data/MAT/ag_news_large.mat'
        else:
            path ='./data/MAT/ag_news.mat'
        # save_mat(path, tmp)
        return examples, dic


class AminerProcessor(DataProcessor):
    """Processor for the News data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples_for_mv(
            self._read_json(data_dir + ".json"), 'train'
        )

    def get_labels(self):
        """See base class."""
        return ['database', 'graphics', 'infocoms']

    def _create_examples_for_mv(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = [[], []]
        random.shuffle(lines)
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_headline = line['authors']
            text_description = line['abstract']
            label = line['cluster']
            headline_example = InputExample(guid=guid, text_a=text_headline, text_b=None, label=label)
            description_example = InputExample(guid=guid, text_a=text_description, text_b=None, label=label)
            examples[0].append(headline_example)
            examples[1].append(description_example)
        return examples


class NewsProcessor(DataProcessor):
    """Processor for the News data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples_for_mv(
            self._read_json(data_dir + ".json"), 'train'
        )

    def get_labels(self):
        """See base class."""

        return ['SCIENCE', 'LATINO VOICES', 'ENVIRONMENT', 'EDUCATION', 'COLLEGE', 'MONEY', 'RELIGION', 'DIVORCE', 'CRIME', 'SPORTS']

    def _create_examples_for_mv(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = [[], []]
        texts = [[], []]
        y = []
        random.shuffle(lines)
        targets = ['SCIENCE', 'LATINO VOICES', 'ENVIRONMENT', 'EDUCATION', 'COLLEGE', 'MONEY', 'RELIGION', 'DIVORCE', 'CRIME', 'SPORTS']

        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_headline = line['headline']
            text_description = line['short_description']
            label = line['category']
            if label not in targets:
                continue
            y_label = self.get_labels().index(label)
            headline_example = InputExample(guid=guid, text_a=text_headline, text_b=None, label=label)
            description_example = InputExample(guid=guid, text_a=text_description, text_b=None, label=label)
            examples[0].append(headline_example)
            examples[1].append(description_example)

            texts[0].append(text_headline) 
            texts[1].append(text_description) # main view
            y.append(y_label)

        x0 = preprocessing(texts[0], 'name')
        x1 = preprocessing(texts[1], 'chinese')
        dic = {'views':[x0, x1], 'label':y}

        tmp = {'x0':x0, 'x1': x1, 'y': y}
        path ='./data/MAT/news.mat'
        # save_mat(path, tmp)
        return examples, dic
        


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        tokens_c = None
        if example.text_c:
            tokens_b = tokenizer.tokenize(example.text_b)
            tokens_c = tokenizer.tokenize(example.text_c)
            # Account for [CLS], [SEP], [SEP], [SEP] with "- 4"
            _truncate_seq_context(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        elif example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        if tokens_c:
            tokens += tokens_c + ["[SEP]"]
            segment_ids += [0] * (len(tokens_c) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop()


def _truncate_seq_context(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_c) or len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        elif len(tokens_c) > len(tokens_a) or len(tokens_c) > len(tokens_b):
            tokens_c.pop()
        else:
            tokens_b.pop()


