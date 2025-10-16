from torchvision import transforms
from PIL import Image
import numpy as np
import random
from torch.utils.data import Dataset
import os
import json

WORKDIR = ''
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

class BaseHashDataset(Dataset):
    def __init__(self, label_file, img_root, img_transform, split_list, split, sample_size, contrastive=False) -> None:
        super().__init__()
        self.img_root = img_root
        self.img_transform =img_transform
        self.contrastive = contrastive
        img_labels = open(label_file, 'r').readlines()
        self.img_label_list = [(
            os.path.join(img_root, val.split()[0]),
            np.array([int(la) for la in val.split()[1:]])
            ) for val in img_labels]

        if sample_size is None:
            self.global_index = np.array(split_list[split])
        else:
            _size = sum([len(split_list[key]) for key in ['query', 'db']])
            sample_index = random.sample(list(range(_size)), sample_size)
            self.global_index = np.array(sample_index)

    def get_image(self, imgpath):
        with open(imgpath, 'rb') as imgf:
            img = Image.open(imgf).convert('RGB')
        return img
    def get_image(self, imgpath):
        try:
            # img = Image.open(img_path).convert('RGB')
            with open(imgpath, 'rb') as imgf:
                img = Image.open(imgf).convert('RGB')
            return img
        except Exception as e:
            print(f"bad: {imgpath}") 
            raise e  

    def __getitem__(self, index):
        img_path, label = self.img_label_list[self.global_index[index]]
        image = self.get_image(img_path)
        if self.contrastive and (self.img_transform == train_transform) :
            image1 = self.img_transform(image)
            image2 = self.img_transform(image)
            return index, image1, label, image2
        elif self.img_transform is not None:
            image = self.img_transform(image)
            return index, image, label
    
    def __len__(self):
        return len(self.global_index)

class MIRFlickrHashDataset(BaseHashDataset):
    def __init__(
            self,
            img_transform, split,
            sample_size = None,
            label_file = './data/flickr25k/allannots.txt',
            img_root = 'dir to mirflickr',
            contrastive = False
        ):
        assert split in ['train', 'db', 'query']
        if os.path.exists('./data/hash_split_for_mirflickr.json'):
            split_list = json.load(open('./data/hash_split_for_mirflickr.json', 'r'))
        else:
            all_index = list(range(len(open(label_file, 'r').readlines())))
            query_index = random.sample(all_index, 2000) 
            train_index = random.sample(list(set(all_index)-set(query_index)), 5000)
            db_index = list(set(all_index)-set(query_index)) 
            split_list = {
                'train': train_index,
                'db': db_index,
                'query': query_index
            }
            json.dump(split_list, open('./data/hash_split_for_mirflickr.json', 'w'))
        super().__init__(label_file, img_root, img_transform, split_list, split, sample_size, contrastive)

class COCO14HashDataset(BaseHashDataset):
    def __init__(
            self,
            img_transform, split,
            sample_size = None,
            label_file = './data/coco2014/allannots.txt',
            img_root = 'dir to coco2014',
            contrastive = False
        ):
        assert split in ['train', 'db', 'query']
        if os.path.exists('./data/hash_split_for_coco14.json'):
            split_list = json.load(open('./data/hash_split_for_coco14.json', 'r'))
        else:
            all_index = list(range(len(open(label_file, 'r').readlines())))
            query_index = random.sample(all_index, 5000)
            train_index = random.sample(list(set(all_index)-set(query_index)), 10000)
            db_index = list(set(all_index)-set(query_index))
            split_list = {
                'train': train_index,
                'db': db_index,
                'query': query_index
            }
            json.dump(split_list, open('./data/hash_split_for_coco14.json', 'w'))
        super().__init__(label_file, img_root, img_transform, split_list, split, sample_size, contrastive)

class NUSWideHashDataset1(Dataset):
    def __init__(
            self,
            img_transform,
            split,
            sample_size=None,
            img_root='dir to nuswide',
            contrastive=False
        ):
        assert split in ['train', 'db', 'query'], "split must be one of ['train', 'db', 'query']"
        
        train_file = './data/nuswide/train.txt'
        db_file = './data/nuswide/database.txt'
        query_file = './data/nuswide/test.txt'

        self.img_transform = img_transform
        self.img_root = img_root
        self.contrastive = contrastive

        if split == 'train':
            self.split_file = train_file
        elif split == 'db':
            self.split_file = db_file
        elif split == 'query':
            self.split_file = query_file

        self.img_label_list = self._load_split(self.split_file)
        
        self.global_index = np.array(range(len(self.img_label_list)))
        if sample_size is not None:
            sample_index = random.sample(range(len(self.img_label_list)), sample_size)
            self.global_index = np.array(sample_index)

    def _load_split(self, split_file):
        with open(split_file, 'r') as f:
            lines = f.readlines()
        
        img_label_list = []
        for line in lines:
            parts = line.strip().split()
            img_path = os.path.join(self.img_root, parts[0]) 
            label = np.array([int(la) for la in parts[1:]]) 
            img_label_list.append((img_path, label))
        
        return img_label_list

    def get_image(self, imgpath):
        img_path = os.path.join(self.img_root, imgpath)
        with open(img_path, 'rb') as imgf:
            img = Image.open(imgf).convert('RGB')
        return img

    def __getitem__(self, index):
        img_path, label = self.img_label_list[self.global_index[index]]
        image = self.get_image(img_path)
        
        if self.contrastive and (self.img_transform == train_transform):
            image1 = self.img_transform(image)
            image2 = self.img_transform(image)
            return index, image1, label, image2
        elif self.img_transform is not None:
            image = self.img_transform(image)
            return index, image, label

    def __len__(self):
        return len(self.global_index)

