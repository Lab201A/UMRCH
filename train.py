import torch
import os, time
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
import numpy as np
from rich import print
from models import BaseIRModel
from metric import map_test
from models import register_models
from dataset import MIRFlickrHashDataset, train_transform, NUSWideHashDataset1, COCO14HashDataset
from torch.utils.tensorboard import SummaryWriter
from munch import Munch
from config import ex
from copy import deepcopy
import json

@ex.automain
def main(_config):
    args = Munch.fromDict(deepcopy(_config))

    comment = '_'.join(list(map(str, [
        args.comment, args.method, args.epochs, args.dataset, args.backbone, 
        args.backbone_frozen, args.hash_bit, args.batch_size, args.lr, args.T, args.th,
        args.alpha, args.beta, args.lr_strategy, args.aggregation, args.neg_th
    ])))
    print(args)
    print(comment)
    metric_fun = map_test
    os.makedirs(os.path.join(args.logfile_path, comment), exist_ok=True)
    if args.checkpoint_path is not None:
        os.makedirs(os.path.join(args.checkpoint_path, comment), exist_ok=True)

    if args.proxyinfo_path is not None:
        os.makedirs(os.path.join(args.proxyinfo_path, comment), exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(args.logfile_path, comment), flush_secs=10)
    if args.dataset == 'flickr':
        train_dataset = MIRFlickrHashDataset(train_transform, 'train', contrastive=args.contrastive)
    elif args.dataset == 'coco2014':
        train_dataset = COCO14HashDataset(train_transform, 'train', contrastive=args.contrastive)
    elif args.dataset == 'nuswide' or args.dataset == 'nuswide1':
        train_dataset = NUSWideHashDataset1(train_transform, 'train', contrastive=args.contrastive)     
    else:
        raise NotImplementedError
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

    global_step = 0
    best_metric = 0.0
    mean_metric = 0.0
    best_file = ''
    best_proxy = ''
    best_metric_dict = {'flag': comment}
    steps_perepoch = len(train_dataloader)
    Model = register_models
    args.steps_perepoch = steps_perepoch
    model:BaseIRModel = Model[args.method](args)
    model.to(args.device, non_blocking=True)
    summary(model)
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters->Total:{total_params/1e6:.2f}M|Trainable:{trainable_params/1e6:.2f}M")
        best_metric_dict['total_params_M']=f"{total_params/1e6:.2f}"
        best_metric_dict['trainable_params_M']=f"{trainable_params/1e6:.2f}"
    except Exception as e:
        print(f"Could not calculate model parameters:{e}")
    
    total_train_time_all_epochs = 0
    total_eval_time_all_epochs = 0
    total_images_per_second = 0
    total_time_per_image_ms = 0
    evaluation_count = 0

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        epoch_eval_time = 0
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        model.set_epoch(epoch)
        if args.backbone_frozen:
            model.cnn_model.eval()
        model.train_epoch_start()

        for i, data in pbar:
            loss = model.train_step(data, writer)
            global_step += 1
            model.global_step += 1
            if isinstance(loss, dict):
                pbar.set_description("epoch: %3d "%(epoch))
                for k, v in loss.items():
                    writer.add_scalar(k, v.item(), global_step=global_step, walltime=time.time())
            else:
                pbar.set_description("epoch: %3d, loss: %7.3f"%(epoch, loss.item()))
                writer.add_scalar('loss', loss.item(), global_step=global_step, walltime=time.time())
            writer.add_scalar('epoch', epoch, global_step=global_step, walltime=time.time())
            writer.add_scalar('lr', model.optim.param_groups[0]['lr'], global_step=global_step, walltime=time.time())
            
            if global_step % int(steps_perepoch*args.eval_interval) == 0:
                eval_start_time = time.time()
                metric = metric_fun(model, args)
                if args.backbone_frozen:
                    model.cnn_model.eval()
                # mean_metric = np.mean([metric['map'], metric['ndcg'], metric['ndcg_iou']])
                mean_metric = np.mean([metric['map'], metric['ndcg']])
                writer.add_scalar('mean_metric', mean_metric, global_step=global_step, walltime=time.time())
                writer.add_scalar('map', metric['map'], global_step=global_step, walltime=time.time())
                writer.add_scalar('ndcg', metric['ndcg'], global_step=global_step, walltime=time.time())

                time_per_image = metric.get('time_per_image_ms', 0)
                images_per_second = metric.get('images_per_second', 0)

                total_time_per_image_ms += time_per_image
                total_images_per_second += images_per_second
                evaluation_count += 1

                print(metric)
                print("mean metric: %.2f"%mean_metric)
                if mean_metric > best_metric:
                    best_metric = mean_metric
                    best_metric_dict.update(metric)
                    best_metric_dict.update({"epoch": epoch})
                    best_metric_dict.update({"mean_metric": mean_metric})
                    if args.checkpoint_path is not None and best_file != '':
                        os.system(f'rm {best_file}')
                    if args.checkpoint_path is not None:
                        torch.save(model.required_state_dict(), os.path.join(args.checkpoint_path, comment, '%d_%d_%.2f_best.pth'%(epoch, global_step, mean_metric)))
                        best_file = os.path.join(args.checkpoint_path, comment, '%d_%d_%.2f_best.pth'%(epoch, global_step, mean_metric))
                        print("Best model get %.2f mean metric, saved to %s"%(best_metric, best_file))
                    if args.proxyinfo_path is not None and best_proxy != '':
                        os.system(f'rm {best_proxy}')
                    if args.proxyinfo_path is not None:
                        pinfo = model.get_pinfo()
                        pinfo.to_excel(os.path.join(args.proxyinfo_path, comment, '%d_%d_%.2f_best.xlsx'%(epoch, global_step, mean_metric)), index=False)
                        best_proxy = os.path.join(args.proxyinfo_path, comment, '%d_%d_%.2f_best.xlsx'%(epoch, global_step, mean_metric))
                        print("Best model get proxy_info, saved to %s"%(best_proxy))
                        
                print(best_metric_dict) 
                eval_end_time = time.time()
                epoch_eval_time += (eval_end_time - eval_start_time)            

        model.train_epoch_end()

        if args.checkpoint_path is not None:
            torch.save(model.required_state_dict(), os.path.join(args.checkpoint_path, comment, 'last.pth'))
            print("Last model get %.2f mean metric, saved to %s"%(mean_metric, os.path.join(args.checkpoint_path, comment, 'last.pth')))
        epoch_end_time = time.time()
        epoch_total_duration = epoch_end_time - epoch_start_time
        epoch_train_duration = epoch_total_duration - epoch_eval_time

        print(f"Epoch {epoch} Done. Train time:{epoch_train_duration:.2f}s, Eval time:{epoch_eval_time:.2f}s, Total time:{epoch_total_duration:.2f}s")
        total_train_time_all_epochs += epoch_train_duration
        total_eval_time_all_epochs += epoch_eval_time

    writer.close()
    
    if args.epochs > 0:
        avg_train_time = total_train_time_all_epochs / args.epochs
        avg_eval_time = total_eval_time_all_epochs / evaluation_count
        best_metric_dict['avg_train_time_per_epochs_s'] = f"{avg_train_time:.2f}"
        best_metric_dict['avg_eval_time_per_epoch_s'] = f"{avg_eval_time:.2f}"
    
        overall_avg_time_per_image = total_time_per_image_ms / evaluation_count
        overall_avg_images_per_second = total_images_per_second /evaluation_count
        best_metric_dict['overall_avg_time_per_image_ms'] = f"{overall_avg_time_per_image:.2f}"
        best_metric_dict['overall_avg_images_per_second'] = f"{overall_avg_images_per_second:.2f}"

    if args.save_best_log is not None:
        with open(args.save_best_log, 'a+') as f:
            f.write(json.dumps(best_metric_dict, indent=4))

