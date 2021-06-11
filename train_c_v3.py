import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *

mixed_precision = True
try:
    from apex import amp
except:
    print("x")
    mixed_precision = False

wdir = 'weights' + os.sep
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'

# hp
hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.0005,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}
f = glob.glob('hyp*.txt')  # 返回所匹配的文件路径列表
if f:
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v


def train(hyp):
    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
    weights = opt.weights  # initial training weights
    imgsz_min, imgsz_max, imgsz_test = opt.img_size  # img sizes (min, max, test)

    gs = 32
    assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)
    opt.multi_scale |= imgsz_min != imgsz_max
    if opt.multi_scale:
        if imgsz_min == imgsz_max:
            imgsz_min //= 1.5
            imgsz_max //= 0.667
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
    img_size = imgsz_max

    init_seeds()
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    nc = 1 if opt.single_cls else int(data_dict['classes'])  # number of classes
    hyp['cls'] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset

    # Remove previous results
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
        os.remove(f)

    model = Darknet(cfg).to(device)

    # optimizer
    pg0, pg1, pg2 = [], [], []
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    # AdGrad or 动量随机梯度下降：SGD+Momentum+Nesterov
    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    # 建立一个渐进的网络，不同层石永红不同的lr和weight decay

    # learning rate decay
    # def adjust_learing_rate(optimizer, epoch):
    #     lr = args.lr * (0.1 ** (epoch // 20))
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    # optimizer通过param_group来管理参数组
    # optimizer.SGD(model.parameters(), lr=1e-2, momentum=.9)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    start_epoch = 0
    best_fitness = 0.0
    # attempt_download(weights)
    if weights.endswith('.pt'):
        pass
    elif len(weights) > 0:
        pass

    if opt.freeze_layers:
        # parmaeter.requires_grad_(False)
        pass

    # Trick1: mixed_presion 加速训练
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # scheduler ?
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1

    # 分布式训练
    # 可以多机分布式训练

    # 流程：
    # 1.在使用distributed包的其它任何函数之前，需要使用init_process_group初始化进程组，
    # 同时初始化distributed包
    # 2.如果需要进行小组内集体通信，用new_group创建子分组
    # 3.创建分布式并行模型DDP（model,device_ids=device_ids)
    # 4.为数据创建Sampler
    # 5.启动工具torch.distrubuted.launch在每个主机上执行一次脚本，开始徐那里拿
    # 6.destory_process_group()销毁进程组
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='tcp://127.0.0.1:9999',
                                             world_size=1,
                                             rank=0)
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers

    # Dataset
    dataset = LoadImageAndLabels(train_path, img_size, batch_size,
                                 augment=True,
                                 hyp=hyp,
                                 rect=opt.rect,
                                 cache_images=opt.cache_images,
                                 single_cls=opt.single_cls)

    # Dataloader
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not opt.rect,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn())

    # Testloader相同

    #
    model.nc = nc
    model.hyp = hyp
    model.gr = 1.0
    # Trick2:类别权重处理不平衡问题
    # 类别多权重就低
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)

    # Trick2:EMA移动指数平均对模型的参数做平均，以求提高测试指标并增加模型鲁棒性
    ema = torch_utils.ModelEMA(model)

    nb = len(dataloader)
    n_burn = max(3 * nb, 500)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)
    t0 = time.time()

    # 开始训练
    for epoch in range(start_epoch, epochs):
        model.train()

        # Update image weights ？
        if dataset.imge_weights:
            pass

        mloss = torch.zeros(4, device=device)

        pbar = tqdm(enumerate(dataloader), total=nb)
        for i, (imags, targets, paths, _) in pbar:
            ni = i + nb * epoch

            imgs = imags.to(device, non_blocking=True).float() / 255.0

            # Warup
            if ni <= n_burn:
                xi = [0, n_burn]
                model.gr = np.interp(ni, xi, [0.0, 1.0])
                accumulate = max(1, np.interp(ni, xi, [1, 64 / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])

            # Multi-scale
            if opt.multi_scale:
                pass

            # Forward
            pred = model(imgs)

            # Loss
            loss, loss_items = compute_loss(pred, targets, model)
            if not torch.isfinite(loss):
                return results

            # Backward
            loss *= batch_size / 64
            loss.backward()

            # Optimize
            optimizer.step()

            # Update scheduler
            scheduler.step()

        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2017.data', help='*.data path')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', nargs='+', type=int, default=[320, 640], help='[min_train, max-train, test]')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--freeze-layers', action='store_true', help='Freeze non-output layers')
    opt = parser.parse_args()
    opt.weights = last if opt.resume and not opt.weights else opt.weights
    check_git_status()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.data = check_file(opt.data)  # check file
    print(opt)
    opt.img_size.extend([opt.img_size[-1]] * (3 - len(opt.img_size)))
    # 选择设备
    device = torch_utils.select_device(opt.device, apex=mixed_orecision, batch_size=opt.batch_size)
    if device.type == 'cpu':
        mixed_orecision = False

    tb_writer = None
    if not opt.evolve:
        # 可视化训练
        tb_writer = SummaryWriter(comment=opt.name)
        train(hyp)
    else:
        if opt.bucket:
            # 用命令行下载谷歌存储文件
            os.system('')

        for _ in range(1):
            if os.path.exists('evolve.txt'):
                parent = 'single'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))
                x = x[np.argsort(-fitness(x))][:n]
                # 减去最小的适应度是为了防止适应度出现负数
                # 这是后代表村下来的概率，而概率是不能为负的
                w = fitness(x) - fitness(x).min() + 1e-3# 权重
                # parent决定如何选择上一代
                # single就选择上一代中最好的那个
                # weighted代表得分前5个加权平均结果作为下一代
                if parent == 'single' or len(x) == 1:
                    # weights代表成员出现的概率
                    x = x[random.choices(range(n), weights=w)[0]]
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum() # new parent

                # Mutate 交叉&变异
                method, mp, s = 3, 0.9, 0.2 # 20% sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([]) # 加权
                ng = len(g)
                # All parameters are mutated simultaneously
                # based on a normal distribution with about 20% 1-sigma:
                if method == 1:
                    v = (npr.randn(ng) * npr.randn() * g * s + 1) ** 2.0
                elif method == 2:
                    v = (npr.randn(ng) * npr.randn(ng) * g * s + 1) ** 2.0
                for i, k in enumerate(hyp.keys()):
                    hyp[k] = x[i + 7] * v[i] # mutate

    # 参数超过不合理范围，剪去
    keys = ['lr0', 'iou_t', 'momentum',
            'weight_decay', 'hsv_s',
            'hsv_v', 'translate',
            'scale', 'fl_gamma']
    limits = [(1e-5, 1e-2), (0.00, 0.70),
          (0.60, 0.98), (0, 0.001),
          (0, .9), (0, .9), (0, .9),
          (0, .9), (0, 3)]

    for k, v in zip(keys, limits):
        hyp[k] = np.clip(hyp[k], v[0], v[1])

    results = train(hyp=copy())







