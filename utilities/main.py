
 
#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Trains a Neural Message Passing Model on various datasets. Methodologi defined in:
    Gilmer, J., Schoenholz S.S., Riley, P.F., Vinyals, O., Dahl, G.E. (2017)
    Neural Message Passing for Quantum Chemistry.
    arXiv preprint arXiv:1704.01212 [cs.LG]
"""

# Torch
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import time
import argparse
import os
import numpy as np

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
from tqdm import tqdm 

__author__ = "chen shao"
__email__ = "chen.shao@student.kit.edu"


# os.environ['WANDB_MODE'] = 'dryrun'

global args, best_er1
PROJECT = "MPNN-Displace-Reaction-Training-Tuning"
logging = True

def main_worker(gpu, ngpus_per_node, args):
    # state gloabl variable for wandb and train 
    global PROJECT
    args.gpu = gpu 
    # Check if CUDA is enabled
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    # Load data
    root = args.datasetPath

    if ngpus_per_node == 1:
        args.gpu = 0

    args.last_epoch = -1
    args.epoch = 0
    args.rank = 0

    ############################# Dataloader & Preprocessing ########################
    # main worker
    print('Prepare files')

    data_train = datasets.PICKLE(args,'train', args.datasetPath)
    data_valid = datasets.PICKLE(args,'eval', args.datasetPath)
    data_test = datasets.PICKLE(args,'test', args.datasetPath)    # Define model and optimizer
    
        # Select one graph
    g_tuple, l = data_train[0]
    g, h_t, e = g_tuple
    
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
    hidden_state_size = 73
    message_size = 73
    n_layers = 3
    l_target = 1
    type ='regression'
    model = MPNN(in_n, hidden_state_size, message_size, n_layers, l_target, type=type)
    del in_n, hidden_state_size, message_size, n_layers, l_target, type

    args.multigpu = False

    print('Check cuda')
    if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif args.gpu is None:
        # Use DP
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    args.epoch = 0
    args.last_epoch = -1
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # train for one epoch
    train(model, train_loader, valid_loader, args, optimizer, epochs=args.epochs, lr=args.lr, device=args.gpu, root=args.root, experiment_name=args.name)


def train(model, train_loader, valid_loader, args, optimizer, epochs, lr=0.0001, device=None, experiment_name="Hyperparameter_tuning", root="."):
    
    global PROJECT, best_er1, step
    root = args.datasetPath
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f"Training {args.name}.")

    # new parameters for documenting on W&B
    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-bs{args.batch_size}-tep{args.epochs}-{uuid.uuid4()}"
    name = f"{args.name}_{run_id}"
    should_write = (args.rank == 0)
    
    should_log = should_write and logging
    if should_log:
        tags = args.tags.split(',') if args.tags != '' else None
        if args.dataset != 'Displace-Reaction': # first change it into 'Displace-Reaction'
            PROJECT = PROJECT + f"-{args.dataset}"
            wandb.init(project=PROJECT, name=name, config=args, dir=args.root, tags=tags, notes=args.notes, id=run_id)
            wandb.watch(model)

    ##################################### Loss:MSE ############################################
    print('Optimizer')
    criterion_mse = nn.MSELoss()
    ##################################### Evaluation ##########################################
    evaluation = lambda output, target: torch.mean(torch.abs(output - target) / torch.abs(target))
    ###########################################################################################
    print('Logger')
    logger = Logger(args.logPath)
    ###########################################################################################
    # leanring rate change strategy
    lr_step = (args.lr-args.lr*args.lr_decay)/(args.epochs*args.schedule[1] - args.epochs*args.schedule[0])


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

    # switch to train mode
    model.train()

    # Train loop
    for epoch in range(args.last_epoch+1, epochs):
        if should_log: wandb.log({"Epoch": epoch}, step=step)
        if epoch > args.epochs * args.schedule[0] and epoch < args.epochs * args.schedule[1]:
            args.lr -= lr_step
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        wandb.log({f"Train/learning_rate": args.lr}, step=step)

        # switch to train mode
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        error_ratio = AverageMeter()
        model.train()
        for i, (g, h, e, target) in tqdm(enumerate(train_loader), desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train\n",\
                             total=len(train_loader)):

            # Prepare input data
            if args.cuda:
                g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
            g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)

            optimizer.zero_grad()

            # Compute output
            output = model(g, h, e)

            # Loss 
            Train_loss = criterion_mse(output.to(device=args.gpu), target)

            # compute gradient and do SGD step
            Train_loss.backward()
            # cut big gradient
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            # Logs for average losses and error ratio for all training samples
            losses.update(Train_loss.data.item(), g.size(0))
            # monitor most important metrics in training process
            error_ratio.update(evaluation(output, target).data, g.size(0))

            # log in wandb
            if should_log and step% 5 == 0:
                wandb.log({f"Train/train_loss_current": Train_loss.item()}, step=step)
                wandb.log({f"Train/error_ratio_avg per epoch": error_ratio.avg.item()}, step=step)
            step += 1
            # logger.log_value('train_epoch_loss', val_mse.avg)
            # logger.log_value('train_epoch_error_ratio', error_ratio.avg.item())

            # log in terminal 
            if i % args.log_interval == 0 and i > 0:
                print('\nEpoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Error Ratio {err.val:.4f} ({err.avg:.4f})'
                    .format(epoch, i, len(train_loader), batch_time=batch_time,
                            data_time=data_time, loss=losses, err=error_ratio))
            # validate 
            if should_write and step % args.validate_every == 0:
                model.eval()
                metrics, val_mse, val_err = validate( args, model, valid_loader, criterion_mse, evaluation, epoch, epochs)
                print("Validated: {}".format(metrics))

                # log into wandb
                if should_log:
                    wandb.log({
                       f"Validate/validation_loss": val_mse.avg}, step=step)
                    wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=step)
                    wandb.log({f"Validate/error_ratio_per_epoch": error_ratio.avg.item()})
                # get the best checkpoint and test it with test set
                model.train()
                er1 = metrics["error_ratio"] 
                is_best = er1 > best_er1
                best_er1 = min(er1, best_er1)
                utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_er1': best_er1,
                                'optimizer': optimizer.state_dict(), }, is_best=is_best, directory=args.resume)
                del er1, metrics, val_mse, is_best
    print('\nEpoch: [{0}] Avg Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
          .format(epoch, err=error_ratio, loss=losses, b_time=batch_time))

    return model

def validate(args, model, val_loader, criterion, evaluation, epoch, epochs, device='cpu'):
    with torch.no_grad():
        metrics = RunningAverageDict()

        batch_time = AverageMeter()
        val_mse = AverageMeter()
        val_error_ratio = AverageMeter()
    
        for i, (g, h, e, target) in tqdm(enumerate(val_loader), desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation\n", total=len(val_loader)):

            # Prepare input data
            if args.cuda:
                g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
            g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)

            # Compute output
            output = model(g, h, e)

            # Logs
            val_mse.update(criterion(output, target).data.item(), g.size(0))
            val_error_ratio.update(evaluation(output, target).data.item(), g.size(0))

            if i % args.log_interval == 0 and i > 0:
                
                print('\nValidation: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Error Ratio {err.val:.4f} ({err.avg:.4f})'
                    .format(i, len(val_loader), batch_time=batch_time,
                            loss=val_mse, err=val_error_ratio))
            metrics.update(compute_errors(output, target), g.size(0))

        print(' * Average Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}'
            .format(err=val_error_ratio, loss=val_mse))

    return metrics.get_value(), val_mse, val_error_ratio

    
if __name__ == '__main__':

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
    parser.add_argument('--bs', '--batch-size', type=int, default=10, metavar='N',
                        help='Input batch size for training (default: 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Enables CUDA training')
    parser.add_argument('--epochs', type=int, default=360, metavar='N',
                        help='Number of epochs to train (default: 360)')
    parser.add_argument('--lr', type=lambda x: utils.restricted_float(x, [1e-5, 1e-2]), default=1e-4, metavar='LR',
                        help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
    parser.add_argument('--lr-decay', type=lambda x: utils.restricted_float(x, [.01, 1]), default=0.6, metavar='LR-DECAY',
                        help='Learning rate decay factor [.01, 1] (default: 0.6)')
    parser.add_argument('--schedule', type=list, default=[0.1, 0.9], metavar='S',
                        help='Percentage of epochs to start the learning rate decay [0, 1] (default: [0.1, 0.9])')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument("--same-lr", "--same_lr", default=False, action="store_true",
                        help="Use same LR for all param groups")
    parser.add_argument('--div-factor', '--div_factor', default=25, type=float, help="Initial div factor for lr")
    parser.add_argument('--final-div-factor', '--final_div_factor', default=100, type=float,
                        help="final div factor for lr")


    # i/o
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='How many batches to wait before logging training status')
    # Accelerating
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')

    # train 
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    parser.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")
    parser.add_argument("--tags", default="tuning dataset splition", type=str, help="Wandb tags.")
    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument('--validate-every', '--validate_every', default=100, type=int, help='validation period')
    parser.add_argument('--split', '--split', default=0.8, type=float, help='fraction')

    best_er1 = 0

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

    # Folder to save 
    if args.root != "." and not os.path.isdir(args.root):
        os.makedirs(args.root)

    # Configurate gpu
    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    if ngpus_per_node == 1:
        args.gpu = 0

    main_worker(args.gpu, ngpus_per_node, args)
