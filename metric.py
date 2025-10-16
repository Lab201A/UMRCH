import torch
import numpy as np
import numba as nb

import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import BaseIRModel
from dataset import test_transform, MIRFlickrHashDataset, NUSWideHashDataset1, COCO14HashDataset

@nb.njit('int32[:, ::1](float32[:,::1])', parallel=True)
def _argsort(a):
    b = np.empty(a.shape, dtype=np.int32)
    for i in nb.prange(a.shape[0]):
        b[i,:] = np.argsort(a[i,:])
    return b

def generate_code(
    model: BaseIRModel,
    db_dataloader: DataLoader,
    query_dataloader: DataLoader,
    is_code=True,
    device='cuda:0'
):
    
    db_continues_img = []
    db_binary_img = []
    db_label = []
    db_feature_img = []
    query_continues_img = []
    query_binary_img = []
    query_label = []
    query_feature_img = []

    num_query_images = len(query_dataloader.dataset)
    num_db_images = len(db_dataloader.dataset)
    total_images = num_query_images + num_db_images

    encoding_start_time = time.time()

    with torch.no_grad():
        for _, img, label in tqdm(query_dataloader):
            img = img.to(device, non_blocking=True)
            _image_f_reps, _image_con_reps, _image_reps = model.get_code(img)
            if is_code:
                query_binary_img.extend(torch.sign(_image_reps).cpu().tolist())
            else:
                query_binary_img.extend(_image_reps.cpu().tolist())
            query_label.extend(label.tolist())

        for _, img, label in tqdm(db_dataloader):
            img = img.to(device, non_blocking=True)
            _image_f_reps, _image_con_reps, _image_reps = model.get_code(img)
            if is_code:
                db_binary_img.extend(torch.sign(_image_reps).cpu().tolist())
            else:
                db_binary_img.extend(_image_reps.cpu().tolist())
            db_label.extend(label.tolist())

    encoding_end_time = time.time()
    encoding_duration = encoding_end_time - encoding_start_time

    if total_images > 0 and encoding_duration > 0:
        time_per_image_ms = (encoding_duration / total_images) * 1000
        images_per_second =  total_images / encoding_duration
    else:
        time_per_image_ms = 0
        images_per_second = 0
    
    db_binary_img = np.array(db_binary_img, dtype=np.float32)
    db_label = np.array(db_label, dtype=np.float32)
    query_binary_img = np.array(query_binary_img, dtype=np.float32)
    query_label = np.array(query_label, dtype=np.float32)
    
    db_continues_img = 0
    query_continues_img = 0
    db_feature_img = 0
    query_feature_img = 0

    return db_binary_img, db_label, query_binary_img, query_label, db_continues_img, query_continues_img, db_feature_img, query_feature_img, time_per_image_ms, images_per_second

def map_topk(inner_dot_neg, relevant_mask, topk=None):
    AP = []
    relevant_mask = (relevant_mask > 0).astype(np.bool8)
    topkindex = _argsort(inner_dot_neg)[:, :topk].astype(np.int32)
    relevant_topk_mask = np.take_along_axis(relevant_mask, topkindex, axis=1)
    cumsum = np.cumsum(relevant_topk_mask, axis=1)
    precision = cumsum / np.arange(1, topkindex.shape[1]+1)

    for query in range(relevant_mask.shape[0]):
        if np.sum(relevant_topk_mask[query]) == 0:
            continue
        AP.append(np.sum(precision[query]*relevant_topk_mask[query]) / np.sum(relevant_topk_mask[query]))
    return float(np.mean(AP))

def DCG(rel, dist, topk=None):

    rank_index = _argsort(dist)[:, :topk]
    rel_rank = np.take_along_axis(rel, rank_index, axis=1)
    return np.mean(np.sum(np.divide(np.power(2, rel_rank) - 1, np.log2(np.arange(rel_rank.shape[1], dtype=np.float32) + 2)), axis=1))

def NDCG(rel, dist, topk=None):
    dcg = DCG(rel, dist, topk)
    idcg = DCG(rel, -rel, topk)

    if dcg == 0.0:
        return 0.0
    ndcg = dcg / idcg
    return float(ndcg)

def map_test(model, args):
    print('computing map for retrieval...')
    query_transform = test_transform

    model.eval()
    if args.dataset == 'flickr':
        query_dataset = MIRFlickrHashDataset(query_transform, 'query')
        db_dataset = MIRFlickrHashDataset(test_transform, 'db') 
    elif args.dataset == 'coco2014':
        query_dataset = COCO14HashDataset(query_transform, 'query')
        db_dataset = COCO14HashDataset(test_transform, 'db')
    elif args.dataset == 'nuswide' or args.dataset == 'nuswide1':
        query_dataset = NUSWideHashDataset1(query_transform, 'query')
        db_dataset = NUSWideHashDataset1(test_transform, 'db')
    else:
        raise NotImplementedError
    
    query_dataloader = DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    db_dataloader = DataLoader(db_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    db_binary_img, db_label, query_binary_img, query_label, db_continues_img, query_continues_img \
        , db_feature_img, query_feature_img, time_per_image_ms, images_per_second\
        = generate_code(model, db_dataloader, query_dataloader, args.iscode, args.device)
    
    inner_dot_neg_i2i = -np.dot(query_binary_img, db_binary_img.T) # 负 二值化后的内积=K*余弦相似度
    relevant_mask = np.dot(query_label, db_label.T) # 共同标签数量，交叉相似度，不是iou相似度

    print(relevant_mask.shape)
    topk = 5000
    
    map = map_topk(inner_dot_neg_i2i, relevant_mask, topk)
    ndcg = NDCG(relevant_mask, inner_dot_neg_i2i, topk)

    model.train()
    return {'map': map, 'ndcg': ndcg, 'time_per_image_ms':time_per_image_ms, 'images_per_second':images_per_second}

if __name__ == "__main__":
    import argparse
    from rich import print
    from models import register_models
    import logging
    import json
    
    log_filename = 'evaluation_log.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s', 
        handlers=[
            logging.FileHandler(log_filename, mode='a'),
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='flickr')
    parser.add_argument('--hash_bit', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--method', type=str, default='orthocos')
    parser.add_argument('--ckpt', type=str, required=True)

    parser.add_argument('--iscode', action='store_true')
    parser.add_argument('--backbone', type=str, default='clip')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--backbone_frozen', action='store_true')
    args = parser.parse_args()

    logging.info(f"Starting evaluation with parameters: {json.dumps(vars(args), indent=4)}")

    print(args)
    Model = register_models
    model:BaseIRModel = Model[args.method](args)
    model.load_state_dict(torch.load(args.ckpt))
    model.to('cuda:0', non_blocking=True)
    model.eval()
    metric = map_test(model, args)

    logging.info(f"Evaluation finished. Metrics: {json.dumps(metric, indent=4)}")
    logging.info("-" * 80) 

    

