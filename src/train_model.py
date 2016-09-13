# preprocess
import h5py
import os
import numpy as np
import scipy.sparse as sparse
import time

import util

TPATH = '../cache/text_selected.txt'
LPATH = '../cache/cat_selected.txt'
MPATH = '../cache/model/'


class model:
    def __init__(self, word_width, bigram_width, h_width, label_n,
                 label_weight):
        np.random.seed(0)

        self.word_width = word_width
        self.bigram_width = bigram_width
        self.h_width = h_width
        self.label_n = label_n
        self.label_weight = label_weight

        # init nn state
        self.w00 = np.random.rand(word_width, h_width)
        self.w00 = np.float32((self.w00-0.5)*2 / (word_width**0.5))
        self.w00_acc = np.zeros((word_width, h_width), dtype=np.float32)

        self.w01 = np.random.rand(bigram_width, h_width)
        self.w01 = np.float32((self.w01-0.5)*2 / (bigram_width**0.5))
        self.w01_acc = np.zeros((bigram_width, h_width), dtype=np.float32)

        self.h_width = h_width
        self.w1 = np.random.rand(h_width+1, label_n)
        self.w1 = np.float32((self.w1-0.5)*2/h_width**0.5)
        # initializing output layer bias for softmax approximation
        # (NCE creiterion)
        self.w1[0, :] += -np.log(label_n)
        self.w1_acc = np.zeros((h_width+1, label_n), dtype=np.float32)

    def relu(s, x):
        return np.maximum(x, 0)

    def train_batch(self, M_word, M_bigram, lbl,
                    lr=0.01, rho=0.95, l2=0.0001, eps=0.000001):
        batch_size = M_word.shape[0]
        odat0 = np.zeros((batch_size, self.h_width), dtype=np.float32)

        odat0 = M_word.dot(self.w00)
        odat0 += M_bigram.dot(self.w01)

        odat0s = self.relu(odat0)
        odat0s = np.concatenate((np.ones((batch_size, 1), dtype=np.float32),
                                 odat0s), axis=1)

        """ Softmax
        odat1 = odat0s.dot(self.w1)
        exp_scores = np.exp(odat1)
        probs = exp_scores / (np.sum(exp_scores, 1) + eps)
        """

        # NCE
        lbl_unique = list(set(lbl))
        tmp_d = {l: i for i, l in enumerate(lbl_unique)}
        lbl_ = np.array([tmp_d[l] for l in lbl])

        w1_s = self.w1[:, lbl_unique]

        odat1 = odat0s.dot(w1_s)  # pos and shared neg samples
        maxv = odat1.max(1).reshape(-1, 1)
        maxv[maxv < 20] = 0
        odat1 = odat1-maxv
        # simple approach with constant scale
        # probs = np.exp(odat1) # / 1 : approximation
        # plus heuristic that helps to scale batch probabilities
        exp_scores = np.exp(odat1)
        maxv = exp_scores.max(1)
        scale = (exp_scores.sum(1)-maxv)/(odat1.shape[1]-1)
        scale = scale*(self.label_n-1)+maxv
        scale[maxv < 1] = 1
        probs = exp_scores/scale.reshape(-1, 1)

        # loss eval
        ids_ = range(batch_size)
        logprobs = -np.log(probs[ids_, lbl_] + eps)
        loss = np.mean(logprobs*self.label_weight[lbl])
        # loss += (self.w1**2).sum()*l2
        # loss += (self.w01**2).sum()*l2
        # loss += (self.w00**2).sum()*l2
        # not including l2 loss to save eval time

        # backprop
        err = probs.copy()
        err[ids_, lbl_] = err[ids_, lbl_] - 1
        err = err * self.label_weight[lbl].reshape(-1, 1)

        grad1 = odat0s.T.dot(err)
        grad1_p = err.dot(w1_s[1:].T)
        grad1_p = (odat0 > 0)*grad1_p  # ReLU diff

        nz_word = M_word.nonzero()[1]
        nz_word = list(set(nz_word))
        nz_bigram = M_bigram.nonzero()[1]
        nz_bigram = list(set(nz_bigram))

        M_word_ = M_word[:, nz_word]
        M_bigram_ = M_bigram[:, nz_bigram]

        grad00 = M_word_.T.dot(grad1_p)
        grad01 = M_bigram_.T.dot(grad1_p)

        # normalize gradients in embedding layer (under test)
        nz_word_w = np.sqrt(M_word_.sum(0))
        nz_bigram_w = np.sqrt(M_bigram_.sum(0))
        grad00 = np.multiply(grad00, 1/nz_word_w.T)
        grad01 = np.multiply(grad01, 1/nz_bigram_w.T)

        # l2 reqularization
        if l2 > 0:
            grad00 += self.w00[nz_word]*l2
            grad01 += self.w01[nz_bigram]*l2
            grad1 += w1_s*l2

        """ SGD
        self.w00[nz_word] -= grad00*lr
        self.w01[nz_bigram] -= grad01*lr
        self.w1[:,lbl_unique] = w1_s - grad1*lr
        """

        # RMSProp

        w00_acc_s = self.w00_acc[nz_word]  # sampling once saves a few ms
        w00_acc_s *= rho
        w00_acc_s += (1-rho)*np.multiply(grad00, grad00)

        w01_acc_s = self.w01_acc[nz_bigram]
        w01_acc_s *= rho
        w01_acc_s += (1-rho)*np.multiply(grad01, grad01)

        w1_acc_s = self.w1_acc[:, lbl_unique]
        w1_acc_s *= rho
        w1_acc_s += (1-rho)*grad1**2

        self.w00[nz_word] -= grad00*lr/np.sqrt(w00_acc_s + eps)
        self.w01[nz_bigram] -= grad01*lr/np.sqrt(w01_acc_s + eps)
        self.w1[:, lbl_unique] = w1_s - grad1*lr/np.sqrt(w1_acc_s + eps)

        self.w00_acc[nz_word] = w00_acc_s
        self.w01_acc[nz_bigram] = w01_acc_s
        self.w1_acc[:, lbl_unique] = w1_acc_s

        true_pos_match = lbl[np.argmax(probs, 1) == lbl_]

        return loss, true_pos_match

    def test_batch(self, M_word, M_bigram, lbl, eps=0.000001):
        batch_size = M_word.shape[0]
        odat0 = np.zeros((batch_size, self.h_width), dtype=np.float32)

        odat0 = M_word.dot(self.w00)
        odat0 += M_bigram.dot(self.w01)

        odat0s = self.relu(odat0)
        odat0s = np.concatenate((np.ones((batch_size, 1), dtype=np.float32),
                                 odat0s), axis=1)

        odat1 = odat0s.dot(self.w1)  # shared negative samples
        odat1 = odat1-odat1.max(1).reshape(-1, 1)
        exp_scores = np.exp(odat1)
        probs = exp_scores/exp_scores.sum(1).reshape(-1, 1)

        # loss eval
        ids_ = range(batch_size)
        logprobs = -np.log(probs[ids_, lbl] + eps)
        loss = np.mean(logprobs*m.label_weight[lbl])

        true_pos_match = lbl[np.argmax(probs, 1) == lbl]

        return loss, true_pos_match


def main(h_width=64, lr=0.03, rho=0.95, l2=0.0,
         decay_after=100, decay_rate=0.95,
         batch_size=500, nepochs=50, print_every=100,
         log_file='train.log'):
    # Init model and data
    if (not os.path.exists(MPATH) or
            'train_data.h5' not in os.listdir(MPATH)):
        util.init_model(MPATH, TPATH, LPATH)
        util.preprocess_data(TPATH, MPATH, 'train_data.h5')
        util.collect_train_data(LPATH, MPATH)

    dl = util.dataLoader(batch_size=batch_size)

    label_acc = np.zeros(dl.label_n)
    for l in dl.label:
        label_acc[l] += 1
    label_weight = (1/label_acc)
    label_weight = label_weight/max(label_weight)

    m = model(dl.word_width, dl.bigram_width,
              h_width, dl.label_n, label_weight)

    n_train_batches = int(dl.train_n/batch_size)
    n_val_batches = int(np.ceil(dl.val_n/batch_size))

    train_accuracy = []
    test_accuracy = []
    train_loss = []
    test_loss = []

    log_file = open(log_file, 'w')

    decay = 1
    for epochi in range(nepochs):
        loss_acc = 0

        acc_true_pos = np.zeros(dl.label_n)
        acc_n = np.zeros(dl.label_n)
        t0 = time.time()

        if epochi >= decay_after:
            decay *= decay_rate

        print 'Epoch :', epochi
        print '-- Train --'
        for batchi in range(n_train_batches):
            M_word, M_bigram, lbl = dl.get_train_batch()
            loss, true_pos = m.train_batch(M_word, M_bigram, lbl,
                                           lr=lr*decay, rho=rho, l2=l2)
            loss_acc += loss
            for l in lbl:
                acc_n[l] += 1
            for l in true_pos:
                acc_true_pos[l] += 1

            if (batchi+1) % print_every == 0:
                accuracy = (acc_true_pos/(acc_n+1)).mean()

                print 'Batch: ', batchi
                print 'Accuracy: ', accuracy
                print 'Loss: ', loss_acc/float(batchi)
                print 'Time: ', (time.time()-t0)/float(print_every), ' s/batch'
                t0 = time.time()

        train_loss.append(loss_acc/float(n_train_batches))
        train_accuracy.append((acc_true_pos/(acc_n+1)).mean())

        loss_acc = 0
        acc_true_pos[:] = 0
        acc_n[:] = 0

        for batchi in range(n_val_batches):
            M_word, M_bigram, lbl = dl.get_val_batch()
            loss, true_pos = m.test_batch(M_word, M_bigram, lbl)

            loss_acc += loss
            for l in lbl:
                acc_n[l] += 1
            for l in true_pos:
                acc_true_pos[l] += 1

        val_loss.append(loss_acc/float(n_val_batches))
        val_accuracy.append((acc_true_pos/(acc_n+1)).mean())

        print '-- Validation --'
        print 'Accuracy: ', val_accuracy[-1]
        print 'Loss: ', val_loss[-1]

        log_file.write('Epoch: '+str(epochi)+'\n')
        log_file.write('train loss: '+str(train_loss[-1])+'\n')
        log_file.write('val loss: '+str(test_loss[-1])+'\n')
        log_file.write('train acc: '+str(train_accuracy[-1])+'\n')
        log_file.write('val acc: '+str(test_accuracy[-1])+'\n')
        log_file.flush()

        print '-- Validation --'

        np.savez(
            MPATH+model_file_head+'npz',
            test_loss=test_loss,
            test_accuracy=test_accuracy,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            param=('lr:'+str(lr) +
                   ' ,rho:'+str(rho) +
                   ' ,l2:'+str(l2) +
                   ' ,hwidth:'+str(hwidth)),
            w1=m.w1,
            w01=m.w01,
            w00=m.w00)
        print '\n\n'

if __name__ == '__main__':
    main()
