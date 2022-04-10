
 
#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Trains a Neural Message Passing Model on various datasets. Methodologi defined in:
    Gilmer, J., Schoenholz S.S., Riley, P.F., Vinyals, O., Dahl, G.E. (2017)
    Neural Message Passing for Quantum Chemistry.
    arXiv preprint arXiv:1704.01212 [cs.LG]

    First run version for 1000 Epochs on scc cluster.
    Log: tensorboard
    Plot: heatmap, 2d histrogram
    Metrics: 
    Lr: linear weight decay

    Ref: https://github.com/optuna/optuna-examples/edit/main/pytorch/pytorch_simple.py
    Visualization: https://tigeraus.gitee.io/doc-optuna-chinese-build/reference/visualization.html
"""

# Torch
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import time
import argparse
import os
import numpy as np
from infer import scatter_histogram_bestfit, heatmap, write_csv
# Our Modules
import datasets
from datasets import utils
from models.MPNN import MPNN
from LogMetric import AverageMeter, Logger, RunningAverage, RunningAverageDict, compute_errors
from pathlib import Path, PurePath, PurePosixPath
import os 
import uuid 
import wandb
from datetime import datetime as dt
import sys
import argparse
import torch.optim as optim
from tqdm import tqdm 
import optuna 
from optuna.trial import TrialState
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt 

__author__ = "chen shao"
__email__ = "chen.shao@student.kit.edu"


# os.environ['WANDB_MODE'] = 'dryrun'

global args, best_er1
PROJECT = "MPNN-Displace-Reaction-Training-Tuning"
logging = True
EPOCHS = 20
LOG_INTERVAL = 10

def restricted_float(x, inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]"%(x,))
    return x

def main_worker(gpu, ngpus_per_node, args, trial):

    hidden_state_size = trial.suggest_int("hidden_state_size", 60, 100, step=10)
    message_size = trial.suggest_int("mesage_size", 60, 100, step=10)
    n_layers = trial.suggest_int("n_layers", 1, 5)
    lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    # optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    global PROJECT
    args.gpu = gpu 
   
    ############################# Dataloader & Preprocessing ########################
    # main worker
    print('Prepare files')
    data_train = datasets.PICKLE(args,'train', args.datasetPath)
    data_valid = datasets.PICKLE(args,'eval', args.datasetPath)

    # Select one graph
    g_tuple, l = data_train[0]
    g, h_t, e = g_tuple

    # Data Loader
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size, shuffle=True,
                                               collate_fn=datasets.utils.collate_g,
                                               num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid,
                                               batch_size=args.batch_size, collate_fn=datasets.utils.collate_g,
                                               num_workers=args.prefetch, pin_memory=True)
    ################################ Model  ##########################################
    print('\tCreate model')
    in_n = [len(h_t[0]), len(list(e.values())[0])]
    hidden_state_size = trial.suggest_int("hidden_state_size", 60, 100, step=10)
    message_size = trial.suggest_int("message_size", 60, 100, step=10)
    n_layers = trial.suggest_int("n_layers", 1, 5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    l_target = 1
    type ='regression'
    model = MPNN(in_n, hidden_state_size, message_size, n_layers, l_target, type=type)
    del in_n, hidden_state_size, message_size, n_layers, l_target, type
    ##################################################################################

    if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    args.multigpu = False
    if args.distributed:
        # Use DDP
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        # args.batch_size = 8
        args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print(f"gpu: {args.gpu}, rank: {args.rank}, batch size: {args.batch_size}, works :{args.workers}")
        torch.cuda.set_device(args.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu,
                                                          find_unused_parameters=True)
        print(f"rank is {args.rank}")
    elif args.gpu is None:
        # Use DP
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    args.epoch = 0
    args.last_epoch = -1

    val_mse = train(trial, model, train_loader, valid_loader, args, epochs=args.epochs, lr=lr, device=args.gpu)

    return val_mse

def train(trial, model, train_loader, valid_loader, args, epochs, lr=0.0001, device=None):
    
    global PROJECT, best_er1, step, logging

    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f"Training {args.name}.")

    if args.rank == 0:
        run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-bs{args.batch_size}-tep{args.epochs}-{uuid.uuid4()}"
        args.directory = f'./run/{args.pathname}/run_{run_id}/xyz_displace/mpnn/checkpoint' if args.resume is None else args.resume
    should_write = ((not args.distributed) or args.rank == 0)
    should_log = should_write and logging

    if should_log:
        print('Logger')
        args.logPath = f'./run/{args.pathname}/run_{run_id}/xyz_displace/mpnn/log'
        if Path(args.directory).exists is False:
            Path(args.directory).mkdir()
        if Path(args.logPath).exists is False:
            Path(args.logPath).mkdir()
        writer = SummaryWriter(args.logPath)
  
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    ##################################### Loss:MSE ############################################
    print('Optimizer')
    criterion_mse = nn.MSELoss()
    ##################################### Evaluation ##########################################
    evaluation = lambda output, target: torch.mean(torch.abs(output - target) / torch.abs(target))
    ###################################### Logg ################################################


    ###########################################################################################
    # leanring rate change strategy
    lr_step = (lr-lr*args.lr_decay)/(args.epochs*args.schedule[1] - args.epochs*args.schedule[0])    

    if args.resume:
        checkpoint_dir = args.resume
        best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if os.path.isfile(best_model_file):
            print("=> loading best model '{}'".format(best_model_file))
            checkpoint = torch.load(best_model_file)
            args.start_epoch = checkpoint['epoch']
            best_er1 = checkpoint['best_er1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])    
            print("=> loaded best model '{}' (epoch {})".format(best_model_file, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no best model found at '{}'".format(best_model_file))

    # if args.same_lr:
    #     print("Using same LR")
    #     params = model.parameters()
    # else:
    #     print("Using diff LR")
    #     params = [{"params": model.get_1x_lr_params(), "lr": lr / 10},
    #               {"params": model.get_10x_lr_params(), "lr": lr}]

    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr_step, epochs=epochs, steps_per_epoch=len(train_loader),
    #                                         cycle_momentum=True,
    #                                         base_momentum=0.85, max_momentum=0.95, last_epoch=args.last_epoch,
    #                                         div_factor=args.div_factor,
    #                                         final_div_factor=args.final_div_factor)

    # if args.resume != '' and scheduler is not None:
    #     scheduler.step(args.epoch + 1)

    iters = len(train_loader)
    step = args.epoch * iters
    best_er1 = np.inf
    # switch to train mode
    model.train()

    # Train loop
    for epoch in range(args.last_epoch+1, epochs):
        if epoch > args.epochs * args.schedule[0] and epoch < args.epochs * args.schedule[1]:
            lr -= lr_step
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if should_log: writer.add_scalar('learning_rate', lr, step)
        # switch to train mode
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        error_ratio = AverageMeter()
        metrics = RunningAverageDict()
        model.train()

        end_load = time.time()
        ########################################## Loop for train #####################################################
        # for i, (g, h, e, target) in tqdm(enumerate(train_loader), desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train\n",\
        #                      total=len(train_loader)):
        for i, (g, h, e, target) in enumerate(train_loader):
            if i >= 100:
                break
            data_time.update(time.time() - end_load)
            # Prepare input data

            g, h, e, target = g.to(device), h.to(device), e.to(device), target.to(device)
            
            # if epoch == 0:
            #     writer.add_graph(model, (g, h, e))
            optimizer.zero_grad()

            # Compute output
            output = model(g, h, e)
            # Loss 
            # Train_loss = criterion_mse(output.to(device=args.gpu), target)
            Train_loss = criterion_mse(output, target)
            # compute gradient and do SGD step
            Train_loss.backward()
            # cut big gradient
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            batch_time.update(time.time() - end_load)

            # Logs for average losses and error ratio for all training samples
            losses.update(Train_loss.data.item(), g.size(0))
            # monitor most important metrics in training process
            error_ratio.update(evaluation(output, target).data, g.size(0))
            metrics.update(compute_errors(output, target), g.size(0))

            if should_log and step% 5 ==0:
                writer.add_scalar('train_epoch_loss',  Train_loss.item(), step)
                writer.add_scalar('train_epoch_error_ratio', error_ratio.avg.item(), step)
                writer.add_scalars('train/metrics', metrics.get_value(), step)

            # log in terminal 
            step += 1
            # if i % args.log_interval == 0:
            #     print('\nEpoch / Train: [{0}][{1}/{2}]\t'
            #         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #         'Error Ratio {err.val:.4f} ({err.avg:.4f})'
            #         .format(epoch+1, i, len(train_loader), batch_time=batch_time,
            #                 data_time=data_time, loss=losses, err=error_ratio))
            ########################################################################################################
            #################################### Loop for validate ################################################
            if should_write and step % args.validate_every == 0 and step > 0:
                model.eval()
                metrics_eval, val_mse, val_err = validate( args, model, valid_loader, criterion_mse, evaluation, epoch, epochs, device)
                print("Validated: {}".format(metrics_eval))

                # log into tensorboard
                if should_log:
                    writer.add_scalar('Validation_epoch_loss', losses.avg, step)
                    writer.add_scalar('Validation_epoch_error_ratio', error_ratio.avg, step)
                    writer.add_scalars('Validation/metrics', metrics_eval, step)
                # get the best checkpoint and test it with test set

                er1 = metrics_eval["mae"] 
                if er1 < best_er1 and should_write:
                    is_best = er1 < best_er1
                    best_er1 = min(er1, best_er1)
                    utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_er1': best_er1,
                                'optimizer': optimizer.state_dict(), }, is_best=is_best, directory=args.directory)
                    del er1, metrics_eval,is_best
                model.train()
            #######################################################################################################
            end_load = time.time()
        
        # report val_mse, quit if not needed.
        trial.report(error_ratio.get_value(), epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    # print('\nEpoch: [{0}] Avg Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
    #       .format(epoch, err=error_ratio, loss=losses, b_time=batch_time))
    if should_log:
        writer.close()
    return error_ratio.get_value()
    
def validate(args, model, val_loader, criterion, evaluation, epoch, epochs, device='cpu'):
    with torch.no_grad():
        metrics = RunningAverageDict()

        batch_time = AverageMeter()
        val_mse = AverageMeter()
        val_error_ratio = AverageMeter()
    
        for i, (g, h, e, target) in enumerate(val_loader):
            if i > 100:
                break
            # Prepare input data

            g, h, e, target = g.to(device), h.to(device), e.to(device), target.to(device)
            
            # Compute output
            output = model(g, h, e)

            # Logs
            val_mse.update(criterion(output, target).data.item(), g.size(0))
            val_error_ratio.update(evaluation(output, target).data.item(), g.size(0))
            metrics.update(compute_errors(output, target), g.size(0))
            # if i % args.log_interval == 0 and i > 0:
            #     print('\nValidation: [{0}/{1}]\t'
            #         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #         'Error Ratio {err.val:.4f} ({err.avg:.4f})'
            #         .format(i, len(val_loader), batch_time=batch_time,
            #                 loss=val_mse, err=val_error_ratio))
            

        # print('Validation * Average Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}'
        #     .format(err=val_error_ratio, loss=val_mse))

    return metrics.get_value(), val_mse.get_value(), val_error_ratio


def objective(trial):

    # Argument parser
    parser = argparse.ArgumentParser(description='Neural message passing')

    parser.add_argument('--dataset', default='Displace-Reaction-03302021', help='dataset ID')
    parser.add_argument('--datasetPath', default='data/data/data-b3d25f30-aebb-4e5d-bebd-71b1bf999c02.pickle', help='dataset path')
    parser.add_argument('--logPath', default='./log/qm9/mpnn/', help='log path')
    parser.add_argument('--plotLr', default=False, help='allow plotting the data')
    parser.add_argument('--plotPath', default='./plot/qm9/mpnn/', help='plot path')
    parser.add_argument('--resume', default=None,
                        help='path to latest checkpoint')
    parser.add_argument("--root", default=".", type=str,
                            help="Root folder to save data in")
    parser.add_argument("--name", default="MPNN-Displace-Reaction")

    # Optimization Options
    parser.add_argument('--bs', '--batch-size', type=int, default=20, metavar='N',
                        help='Input batch size for training (default: 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Enables CUDA training')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='Number of epochs for param tuning.')
    # parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-5, 1e-2]), default=1e-4, metavar='LR',
    #                     help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
    parser.add_argument('--lr-decay', type=lambda x: restricted_float(x, [.01, 1]), default=0.6, metavar='LR-DECAY',
                        help='Learning rate decay factor [.01, 1] (default: 0.6)')

    parser.add_argument('--schedule', type=list, default=[0.1, 0.9], metavar='S',
                        help='Percentage of epochs to start the learning rate decay [0, 1] (default: [0.1, 0.9])')
    # parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
    #                     help='SGD momentum (default: 0.9)')
    parser.add_argument("--same-lr", "--same_lr", default=False, action="store_true",
                        help="Use same LR for all param groups")
    # parser.add_argument('--div-factor', '--div_factor', default=25, type=float, help="Initial div factor for lr")
    # parser.add_argument('--final-div-factor', '--final_div_factor', default=100, type=float,
    #                     help="final div factor for lr")


    # i/o
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='How many batches to wait before logging training status')
    # Accelerating
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
# train 
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    parser.add_argument("--workers", default=20, type=int, help="Number of workers for data loading")
    parser.add_argument("--tags", default="tuning dataset splition", type=str, help="Wandb tags.")
    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument('--validate-every', '--validate_every', default=200, type=int, help='validation period')
    parser.add_argument('--split', '--split', default=0.8, type=float, help='fraction')
    parser.add_argument("--distributed", default=False, action="store_true", help="Use DDP if set")
    parser.add_argument('--pathname', '--pn', default=dt.now().strftime('%d-%h-%H-%M'), type=str, help='name for multigpu training')


    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        print(arg_filename_with_prefix)
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    # standardize variable
    args.batch_size = args.bs
    args.num_threads = args.workers
    args.mode = 'train'

    # Folder to save $ new parameters for documenting on W&B

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace('[', '').replace(']', '')
        nodes = node_str.split(',')

        args.world_size = len(nodes)
        args.rank = int(os.environ['SLURM_PROCID'])

    except KeyError as e:
        # We are NOT using SLURM
        args.world_size = 1
        args.rank = 0
        nodes = ["127.0.0.1"]

    if args.distributed:
        mp.set_start_method('forkserver')

        print(f"rank: {args.rank}")
        port = np.random.randint(15000, 15025)
        args.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        print(f"dist_url: {args.dist_url}")
        args.dist_backend = 'nccl'
        args.gpu = None

    # Configurate gpu
    ngpus_per_node = torch.cuda.device_count()
    print(f"Get {ngpus_per_node} devices...")
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mse = mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, trial))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        mse = main_worker(args.gpu, ngpus_per_node, args, trial)
    return mse

if __name__ == '__main__':
    study = optuna.create_study(direction="minimize", storage='sqlite:///example.db', load_if_exists=True)
    study.optimize(objective, n_trials=5)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    df = study.trials_dataframe(attrs=('number', 'params', 'value', 'state'))   
    print(df)
    optuna.visualization.plot_contour(study, params=['hidden_state_size', 'mesage_size', 'n_layers'])
    fig.show()

    fig = optuna.visualization.plot_contour(study, params=['hidden_state_size', 'mesage_size', 'n_layers'])
    # install plotly then you can print out the in website.
    fig.show()
    fig = optuna.visualization.plot_contour(study, params=['hidden_state_size', 'mesage_size', 'n_layers'])
    fig.show()
    fig = optuna.visualization.plot_parallel_coordinate(study, params=['hidden_state_size', 'mesage_size', 'n_layers'])
    fig.show()
    fig.show()
