import os
import torch
import numpy as np
import pickle
from skimage import io, transform
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class CaptionDataset(Dataset):

    def __init__(self, mode, path='./dataset', transform=None):
        super(CaptionDataset, self).__init__()
        self.img_dir = os.path.join(path, 'Images')
        self.transform = transform
        if mode == 'train':
            with open(os.path.join(path, 'train_dataset.txt')) as f:
                self.data_frame = [line.strip().split('\t') for line in f.readlines()]

        elif mode == 'val':
            with open(os.path.join(path, 'val_dataset.txt')) as f:
                self.data_frame = [line.strip().split('\t') for line in f.readlines()]
        elif mode == 'test':
            with open(os.path.join(path, 'test_dataset.txt')) as f:
                self.data_frame = [line.strip().split('\t') for line in f.readlines()]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.img_dir, self.data_frame[idx][0])
        img = io.imread(img_name)  # type:numpy shape: (H, W, C)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        cap = self.data_frame[idx][1]
        sample = (img, cap)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, sample):  # sample: (img, cap)
        img, cap = sample
        h, w = img.shape[:2]
        new_h, new_w = self.output_size, self.output_size

        img = transform.resize(img, (new_h, new_w))  # shape:[224,224,3]
        return (img, cap)


class Numericalize(object):

    def __init__(self):
        with open('./dataset/wtoi.pkl', 'rb') as f:
            mapping = pickle.load(f)
            self.mapping = mapping

    def __call__(self, sample):
        img, cap = sample
        cap = cap.split()

        def map_func(x):
            if x in self.mapping:
                return self.mapping[x]
            else:
                return self.mapping['<unk>']

        new_cap = list(map(map_func, cap))

        return (img, new_cap)


class ToTensor(object):

    def __call__(self, sample):
        img, cap = sample
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img)
        cap = torch.tensor(cap)
        return (img, cap)


# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
class Normalize(object):

    def __init__(self, mean, std):
        assert isinstance(mean, (tuple, list)) and isinstance(std, (tuple, list))
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, sample):
        img, cap = sample
        img = self.normalize(img)
        return img, cap


def get_loader(mode, path, batch_size):
    tsfm = transforms.Compose([Rescale(224), Numericalize(), ToTensor(),
                               Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    ds = CaptionDataset(mode=mode, path=path, transform=tsfm)
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_padding, pin_memory=True)

    return dl


def makeTxtFile(path='./dataset/Flickr_TextData'):
    img = {}
    txt = {}  # filename: caption
    with open(os.path.join(path, 'Flickr_8k.trainImages.txt'), 'r') as f:
        img['train'] = [line.strip() for line in f.readlines()]  # filename
    with open(os.path.join(path, 'Flickr_8k.testImages.txt'), 'r') as f:
        img['test'] = [line.strip() for line in f.readlines()]
    with open(os.path.join(path, 'Flickr_8k.devImages.txt'), 'r') as f:
        img['val'] = [line.strip() for line in f.readlines()]
    with open(os.path.join(path, 'Flickr8k.token.txt'), 'r') as f:
        for line in f.readlines():
            filename, caption = line.strip().split('\t')  # [ID, caption]
            filename = filename[:-2]
            if filename not in txt:
                txt[filename] = [caption]
            else:
                txt[filename].append(caption)

    print(txt)  # {filename: [seq1, seq2, ...]}
    if os.path.exists('./dataset/train_dataset.txt'):
        print('train_dataset already exists')
    else:
        with open('./dataset/train_dataset.txt', 'w') as f:
            for key in img['train']:
                seqs = txt[key]
                for i in range(len(seqs)):
                    seqs[i] = '<start> ' + seqs[i] + ' <end>'
                    line = '{}\t{}\n'.format(key, seqs[i])
                    f.write(line)
    if os.path.exists('./dataset/test_dataset.txt'):
        print('test_datset already exists')
    else:
        with open('./dataset/test_dataset.txt', 'w') as f:
            for key in img['test']:
                seqs = txt[key]
                for i in range(len(seqs)):
                    seqs[i] = '<start> ' + seqs[i] + ' <end>'
                    line = '{}\t{}\n'.format(key, seqs[i])
                    f.write(line)


def collate_padding(batch):
    def pad_func(vec, max_len):
        p = torch.empty(max_len, dtype=torch.int).fill_(2)
        l = len(vec)
        p[:l] = vec[:l]
        return p

    batch_size = len(batch)
    max_len = -1
    I = []
    C = []
    for item in batch:
        if max_len == -1 or max_len < len(item[1]):
            max_len = len(item[1])
        I.append(item[0])
        C.append(item[1])
    # padding shape: [batch_size, max_len]

    P = [pad_func(vec, max_len) for vec in C]

    return [I, P]


def make_vocab(path='./dataset/Flickr_TextData', thres=2):
    count = {}
    wtoi = {}
    itow = {}
    wtoi['<start>'] = 0
    itow[0] = '<start>'

    wtoi['<end>'] = 1
    itow[1] = '<end>'

    wtoi['<pad>'] = 2
    itow[2] = '<pad>'

    wtoi['<unk>'] = 3
    itow[3] = '<unk>'

    index = 4
    with open(os.path.join(path, 'Flickr8k.token.txt'), 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split('\t')[1].split()
            for token in tokens:
                if token not in count:
                    count[token] = 1
                else:
                    count[token] += 1
    for k, v in count.items():
        if v > 2:
            wtoi[k] = index
            itow[index] = k
            index += 1

    assert len(wtoi) == len(itow)
    #print(len(wtoi))  # 4244


    if not os.path.exists('./dataset/wtoi.pkl'):
        with open('./dataset/wtoi.pkl', 'wb') as f:
            pickle.dump(wtoi, f)

    if not os.path.exists('./dataset/itow.pkl'):
        with open('./dataset/itow.pkl', 'wb') as f:
            pickle.dump(itow, f)

#make_vocab()
# makeTxtFile()
# transform=transforms.Compose([Rescale(256), Numericalize(), ToTensor()])
