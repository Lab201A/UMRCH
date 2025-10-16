from sacred import Experiment

ex = Experiment('Image Hashing')

@ex.config
def config():
    dataset = 'flickr'
    nclass = 24
    hash_bit = 32
    batch_size = 32
    logfile_path = './tensorboard_logs'
    checkpoint_path = './checkpoints'
    save_best_log = './best_log/best.log'
    proxyinfo_path = None

    method = 'umrch'
    backbone = 'clip'
    backbone_frozen = True

    comment = 'default'
    device = 'cuda:0'
    epochs = 40
    lr = 0.0001
    eval_interval = 1

    contrastive = False
    #+++++++++

    T = 0.01
    th = 0.4 # <= 0.4
    temperature = 1
    alpha = 1
    beta = 1
    lr_strategy = None
    aggregation = 0
    neg_th = 0

    #+++++++++++++++++++++++++++++++++++
    iscode = True

@ex.named_config
def no_ckpt():
    checkpoint_path = None

@ex.named_config
def flickr():
    dataset ='flickr'
    nclass = 24

@ex.named_config
def coco2014():
    dataset = 'coco2014'
    eval_interval = 2
    nclass = 80

@ex.named_config
def nuswide():
    dataset = 'nuswide1'
    eval_interval = 2
    nclass = 21

@ex.named_config
def nuswide1():
    dataset = 'nuswide1'
    eval_interval = 2
    nclass = 21
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

@ex.named_config
def umrch():
    method = 'umrch'
    backbone = 'clip'
    backbone_frozen = True
    lr = 0.005
    batch_size = 32
    epochs = 100
    contrastive = True
    save_best_log = './best_log/umrch.log'
    logfile_path = './tensorboard_logs/umrch'
    checkpoint_path = './checkpoints/umrch'
    T = 0.01
    th = 0.3 # <= 0.4
    temperature = 1
    alpha = 1
    beta = 150
    lr_strategy = "onecyclelr"
    aggregation = 0
    neg_th = 0

@ex.named_config
def cliph():
    method = 'cliph'
    backbone = 'clip'
    backbone_frozen = True
    lr = 0.1
    batch_size = 32
    epochs = 70
    save_best_log = './best_log/cliph.log'
    logfile_path = './tensorboard_logs/cliph'
    checkpoint_path = './checkpoints/cliph'
    T = 0.01
    th = 0.4 # <= 0.4
    temperature = 1
    alpha = 1
    beta = 1
    lr_strategy = "onecyclelr"
    aggregation = 0
    neg_th = 0.2
