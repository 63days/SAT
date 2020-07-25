from nltk.translate.bleu_score import corpus_bleu
import torch
import numpy as np
import time


def test_func(pred, itow):  # pred: list, yb: list of tensor
    # pred: [B/5, L], yb: [B, L]
    # pred <end>까지 커팅해야됨
    def map_func(x):
        if x in itow:
            return itow[x]
        else:
            return '<unk>'

    with open('./dataset/test_dataset.txt') as f:
        df = [line.strip().split('\t') for line in f.readlines()]
    df = np.array(df)
    img_name = df[:, 0][::5]
    refs = df[:, 1].tolist()

    refs = [list(filter(lambda x: x != '<start>' and x != '<end>', line.split())) for line in refs]
    R = []
    for i in range(0, len(refs), 5):
        tmp = [refs[i], refs[i + 1], refs[i + 2], refs[i + 3], refs[i + 4]]
        R.append(tmp)

    C = []
    for line in pred:
        line = list(map(map_func, line))  # ['I', 'am' ...]
        for i, x in enumerate(line):
            if x == '<end>':
                line = line[1:i]
                C.append(line)
                break
    # refs: [1000, 5]
    bleu_1 = corpus_bleu(R, C, weights=(1., 0, 0, 0)) * 100
    bleu_2 = corpus_bleu(R, C, weights=(1. / 2., 1. / 2., 0, 0)) * 100
    bleu_3 = corpus_bleu(R, C, weights=(1. / 3., 1. / 3., 1. / 3., 0)) * 100
    bleu_4 = corpus_bleu(R, C) * 100

    print('| BLEU-1:{:.2f} | BLEU-2:{:.2f} | BLEU-3:{:.2f} | BLEU-4:{:.2f} |'.format(bleu_1, bleu_2, bleu_3, bleu_4))

    with open('./result.txt', 'at', encoding='utf-8') as f:
        tic = time.strftime('%y%m%d %H:%M', time.localtime(time.time()))
        f.write('| BLEU-1:{:.2f} | BLEU-2:{:.2f} | BLEU-3:{:.2f} | BLEU-4:{:.2f} |\n'.format(bleu_1, bleu_2, bleu_3,
                                                                                             bleu_4))
        f.write('{} result\n'.format(tic))
        for i in range(len(img_name)):
            c = ' '.join(C[i])
            line = '{}\t{}\n'.format(img_name[i], c)
            f.write(line)
