import os
import re
import hashlib
from random import shuffle
import h5py
import numpy as np
import scipy.sparse as sparse


def tokenize(t):
    t2 = t.lower()
    t3 = re.findall(r'\b[a-z\'-]{2,}\b', t2)
    return t3


def hash_trick(s):
    m = hashlib.md5()
    m.update(s)
    num = int(m.hexdigest()[:8], 16)
    num = num % (2**23)
    return num


def process_bigrams(l):
    bigrams_id = []
    for start_i in range(len(l)-1):
        s = l[start_i] + ' ' + l[start_i+1]
        bigrams_id.append(hash_trick(s))
    return bigrams_id


def init_dict(tpath):
    tok_acc = {}
    with open(tpath, 'r') as f:
        for l in f:
            tokens = tokenize(l)
            tokens = set(tokens)
            for t in tokens:
                if t in tok_acc:
                    tok_acc[t] += 1
                else:
                    tok_acc[t] = 1
    print 'Unique tokens found: ', len(tok_acc)

    selected_tok = []
    for t, n in tok_acc.iteritems():
        if n > 100:
            selected_tok.append(t)
    print 'Selected tokens:', len(selected_tok)

    tok_dict = {t: i for i, t in enumerate(selected_tok)}
    return tok_dict


def init_label_dict(lfile):
    labels = []
    with open(lfile) as f:
        for l in f:
            labels.append(l[:-1])
    unique_labels = set(labels)
    label_dict = {l: i for i, l in enumerate(unique_labels)}

    return label_dict


def init_model(opath, tpath, lpath):
    assert os.path.exists(tpath), 'Download data and run init_dataset.py first'

    if not os.path.exists(opath):
        os.mkdir(opath)

    tok_dict = init_dict(tpath)
    label_dict = init_label_dict(lpath)
    with h5py.File(os.path.join(opath, 'metadata.h5')) as hf:
        hf['token_dict_keys'] = tok_dict.keys()
        hf['token_dict_values'] = tok_dict.values()

        hf['label_dict_keys'] = label_dict.keys()
        hf['label_dict_values'] = label_dict.values()


def preprocess_data(tpath, opath, filename='train_data.h5'):
    with h5py.File(os.path.join(opath, 'metadata.h5')) as hf:
        tok_dict = dict(zip(hf['token_dict_keys'][:],
                            hf['token_dict_values'][:]))

    with open(tpath) as f:
        M_words_indptr = [0]
        M_words_indices = []
        M_bigram_indptr = [0]
        M_bigram_indices = []

        for l in f:
            toks = tokenize(l)
            toks_selected = [t for t in toks if t in tok_dict]

            toks_id = [tok_dict[t] for t in toks_selected]
            toks_id = list(set(toks_id))
            M_words_indptr.append(M_words_indptr[-1]+len(toks_id))
            M_words_indices += list(toks_id)

            bigrams_id = process_bigrams(toks_selected)
            bigrams_id = list(set(bigrams_id))
            M_bigram_indptr.append(M_bigram_indptr[-1]+len(bigrams_id))
            M_bigram_indices += list(bigrams_id)

    with h5py.File(os.path.join(opath, filename)) as hf:
        hf.create_dataset('M_words_indptr',
                          data=M_words_indptr, dtype='int64')
        hf.create_dataset('M_words_indices',
                          data=M_words_indices, dtype='int32')
        hf.create_dataset('M_bigram_indptr',
                          data=M_bigram_indptr, dtype='int64')
        hf.create_dataset('M_bigram_indices',
                          data=M_bigram_indices, dtype='int32')


def collect_train_data(lpath,
                       mpath,
                       opath=None,
                       filename='train_data.h5'):

    if opath is None:
        opath = mpath

    train_p = 0.9
    test_p = 0.05
    val_p = 1-train_p-test_p

    with h5py.File(os.path.join(mpath, 'metadata.h5')) as hf:
        label_dict_keys = hf['label_dict_keys'][:]
        label_dict_values = hf['label_dict_values'][:]
    label_dict = dict(zip(label_dict_keys, label_dict_values))

    label = []
    label_list = {}
    with open(lpath) as lfile:
        i = 0
        for l in lfile:
            labeli = label_dict[l[:-1]]
            label.append(labeli)
            if labeli in label_list:
                label_list[labeli].append(i)
            else:
                label_list[labeli] = [i]
            i += 1

    train_ids = []
    val_ids = []
    test_ids = []

    for labeli, ids in label_list.iteritems():
        ntrain = int(train_p*len(ids))
        nval = int(test_p*len(ids))

        shuffle(ids)

        train_ids += ids[:ntrain]
        val_ids += ids[ntrain:ntrain+nval]
        test_ids += ids[ntrain+nval:]

    with h5py.File(os.path.join(opath, filename)) as hf:
        hf['train_ids'] = train_ids
        hf['val_ids'] = val_ids
        hf['test_ids'] = test_ids

        hf['label'] = label


class dataLoader:
    train_start_id = 0
    test_start_id = 0
    val_start_id = 0

    def __init__(self, data='../cache/model/train_data.h5',
                 meta='../cache/model/metadata.h5', batch_size=500):
        np.random.seed(0)
        self.batch_size = batch_size
        self.bigram_width = 2**23

        with h5py.File(meta) as hf:
            self.word_width = hf['token_dict_keys'].shape[0]
            self.label_n = hf['label_dict_keys'].shape[0]

        with h5py.File(data) as hf:
            M_bigram_indices = hf['M_bigram_indices'][:]
            M_bigram_indptr = hf['M_bigram_indptr'][:]
            M_word_indices = hf['M_words_indices'][:]
            M_word_indptr = hf['M_words_indptr'][:]

            self.total_n = len(M_word_indptr)-1

            self.M_word = sparse.csr_matrix(
                (np.ones(len(M_word_indices), dtype=np.float32),
                 M_word_indices, M_word_indptr),
                shape=(self.total_n, self.word_width))

            self.M_bigram = sparse.csr_matrix(
                (np.ones(len(M_bigram_indices), dtype=np.float32),
                 M_bigram_indices, M_bigram_indptr),
                shape=(self.total_n, self.bigram_width))

            if 'label' in hf:
                self.label = hf['label'][:]

            if 'train_ids' in hf:
                self.train_ids = hf['train_ids'][:]
                self.test_ids = hf['test_ids'][:]
                self.val_ids = hf['val_ids'][:]

                self.train_n = len(self.train_ids)
                self.test_n = len(self.test_ids)
                self.val_n = len(self.val_ids)

                tmp = np.random.permutation(self.train_n)
                self.train_ids_shuf = self.train_ids[tmp]

    def get_train_batch(self):
        end_id = self.train_start_id + self.batch_size
        if end_id > self.train_n:
            self.train_start_id = 0
            end_id = self.batch_size
            tmp = np.random.permutation(self.train_n)
            self.train_ids_shuf = self.train_ids[tmp]
        sample_ids = self.train_ids_shuf[self.train_start_id:end_id]

        M_bigram_sample = self.M_bigram[sample_ids]
        M_word_sample = self.M_word[sample_ids]

        label_sample = self.label[sample_ids]

        self.train_start_id = end_id
        return M_word_sample, M_bigram_sample, label_sample

    def get_test_batch(self):
        end_id = self.test_start_id + self.batch_size
        if end_id > self.test_n:
            sample_ids = self.test_ids[self.test_start_id:]
            self.test_start_id = 0
            end_id = self.batch_size
        else:
            sample_ids = self.test_ids[self.test_start_id:end_id]

        M_bigram_sample = self.M_bigram[sample_ids]
        M_word_sample = self.M_word[sample_ids]

        label_sample = self.label[sample_ids]

        self.test_start_id = end_id
        return M_word_sample, M_bigram_sample, label_sample

    def get_val_batch(self):
        end_id = self.val_start_id + self.batch_size
        if end_id > self.val_n:
            sample_ids = self.val_ids[self.val_start_id:]
            self.val_start_id = 0
            end_id = self.batch_size
        else:
            sample_ids = self.val_ids[self.val_start_id:end_id]

        M_bigram_sample = self.M_bigram[sample_ids]
        M_word_sample = self.M_word[sample_ids]

        label_sample = self.label[sample_ids]

        self.val_start_id = end_id
        return M_word_sample, M_bigram_sample, label_sample
