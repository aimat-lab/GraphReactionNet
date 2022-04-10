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
"""
# TODO readout from global node
# TODO change nonlinear function 
# TODO concatenate nodes of all layers
# TODO change loss function 
# TODO learning rate update strategy
# TODO evaluate two lastest and best model
import json
import IPython
import torch
import wandb
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from LogMetric import scatter_histogram_bestfit, heatmap, write_csv
from ignite.contrib.metrics.regression import R2Score as R2
import datasets
from datasets import xyz_utils
from models import custom_loss
from models.MPNN import MPNN
from models.MPNN_GlobalReadout import MPNN_GlobalReadout
from models.MPNN_concat import MPNN_concat
from LogMetric import AverageMeter, RunningAverageDict, compute_errors, csv_result_file, save_args
import os
import uuid
from torch.utils.data import DataLoader
from pdb import set_trace
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.utils.data.distributed
from datetime import datetime as dt
import sys
import argparse
from tqdm import tqdm
from datasets import PICKLE
from pathlib import Path, PurePath, PurePosixPath

__author__ = "chen shao"
__email__ = "chen.shao@student.kit.edu"

global args, best_er1
PROJECT = "MPNN-Displace-Reaction-Training-Optimization"

# os.environ['WANDB_MODE'] = 'dryrun'

def restricted_float(x, inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]" % (x,))
    return x


def main_worker(gpu, ngpus_per_node, args):
    # state gloabl variable for wandb and train
    global PROJECT
    args.gpu = gpu

    # Dataloader & Preprocessing
    print('Prepare files')

    data_train = datasets.PICKLE(args, 'train', args.datasetPath)
    data_valid = datasets.PICKLE(args, 'eval', args.datasetPath, index_list=data_train.valid_id)
    data_test = datasets.PICKLE(args, 'test', args.datasetPath, index_list=data_train.test_id)  # Define model and optimizer

    # data_train = datasets.PICKLE(args, 'train', args.datasetPath)
    # data_valid = datasets.PICKLE(args, 'eval', args.datasetPath)
    # data_test = datasets.PICKLE(args, 'test', args.datasetPath)  # Define model and optimizer

    # Select one graph
    _, g_tuple, _, _ = data_train[0]
    _, h_t, e = g_tuple

    # Data Loader
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size, shuffle=True,
                                               collate_fn=datasets.xyz_utils.collate_g,
                                               num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid,
                                               batch_size=args.batch_size, collate_fn=datasets.xyz_utils.collate_g,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=args.batch_size, collate_fn=datasets.xyz_utils.collate_g,
                                              num_workers=args.prefetch, pin_memory=True)

    ################################ Model  ##########################################
    print('\tCreate model')

    in_n = [len(h_t[0]), len(list(e.values())[0])]
    hidden_state_size = 97
    message_size = 65 
    n_layers = 3
    l_target = 1

    type = 'regression'

    config_model = dict()
    config_model.update(
        {"in_n": in_n, "hidden_state_size": hidden_state_size, "message_size": message_size, "n_layers": n_layers,
         "l_target": 1, "type": type})


    if args.model_name == "mpnn":
        model = MPNN(in_n, hidden_state_size, message_size, n_layers, l_target, args.model_name, type=type)
    elif args.model_name == "mpnn_global_readout":
        model = MPNN_GlobalReadout(in_n, hidden_state_size, message_size, n_layers, l_target, args.model_name, type=type)
    elif args.model_name == "mpnn_concat":
        model = MPNN_concat(in_n, hidden_state_size, message_size, n_layers, l_target,  args.model_name, type=type)

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
        print(
            f"Config_Distributed: gpu: {args.gpu}, rank: {args.rank}, batch size: {args.batch_size}, works :{args.workers}")
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

    train(model, train_loader, valid_loader, args, epochs=args.epochs, lr=args.lr, device=args.gpu,
          config_dict=config_model)

    ################################################### Evalution ##################################################

    if args.rank == 0:

        model.eval()
        validate_save(args, 'lastest', 'valid', model, valid_loader, data_train, config_model, device=args.gpu)
        validate_save(args, 'lastest', 'test', model, test_loader, data_train, config_model, device=args.gpu)
        validate_save(args, 'lastest', 'train', model, train_loader, data_train, config_model, device=args.gpu)

        best_model_file = os.path.join(args.directory, 'model_best.pth')

        if Path(args.directory).exists is False:
            os.makedirs(args.directory)

        if os.path.isfile(best_model_file):
            print("=> loading best model '{}'".format(best_model_file))
            checkpoint = torch.load(best_model_file)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded best model '{}' (epoch {})".format(best_model_file, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no best model found at '{}'".format(best_model_file))

        model.eval()
        validate_save(args, 'best', 'valid', model, valid_loader, data_train, config_model, device=args.gpu)
        validate_save(args, 'best', 'test', model, test_loader, data_train, config_model, device=args.gpu)
        validate_save(args, 'best', 'train', model, train_loader, data_train, config_model, device=args.gpu)

    return


def train(model, train_loader, valid_loader, args, epochs, lr=0.0001, device=None, config_dict=None):
    global PROJECT, best_er1, step

    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f"Training {args.name}.")

    if args.rank == 0:
        run_id = f"{args.model_name}-bs{args.batch_size}-epoch{args.epochs}-{uuid.uuid4()}"
        args.directory = f'./{args.run_mode}/{args.pathname}/run_{run_id}/{PurePosixPath(Path(args.datasetPath).parts[-1]).stem}/checkpoint' if args.resume is None else args.resume
        print(f"Record starts: saved in {args.directory}.")
        name = f"{args.name}_{run_id}"
    should_write = ((not args.distributed) or args.rank == 0)
    should_log = should_write and args.wandb
    if should_log:
        tags = args.tags.split(',') if args.tags != '' else None
        if args.dataset != 'dataset2_no_overlapping':
            PROJECT = PROJECT + f"-{args.dataset}"
            wandb.init(project=PROJECT, name=name, config=args, dir=args.root, tags=tags, notes=args.notes, id=run_id)
            wandb.watch(model)


    if should_log:
        print('Logger')
        args.logPath = f'./{args.run_mode}/{args.pathname}/run_{run_id}/{PurePosixPath(Path(args.datasetPath).parts[-1]).stem}/log'
        if Path(args.directory).exists is False:
            Path(args.directory).mkdir()
        if Path(args.logPath).exists is False:
            Path(args.logPath).mkdir()
        writer = SummaryWriter(args.logPath)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ##################################### Loss:MSE ############################################
    print('Optimizer')
    if args.loss == "MSE":
        criterion = nn.MSELoss()
    elif args.loss == "L1Loss":
        criterion = nn.L1Loss()
    elif args.loss == "SmoothL1Loss":
        criterion = nn.SmoothL1Loss()
    elif args.loss == "HuberLoss":
        criterion = nn.HuberLoss()
    ##################################### Evaluation ##########################################
    evaluation = lambda output, target: torch.mean(torch.abs(output - target) / torch.abs(target))
    ###################################### Logg ################################################
    if args.decay_mode:
        lr_step = (args.lr - args.lr * args.lr_decay) / (args.epochs * args.schedule[1] - args.epochs * args.schedule[0])
    else:
        lr_step = 0

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


    iters = len(train_loader)
    step = args.epoch * iters
    best_er1 = np.inf

    if args.pytorch_scheduler is True:

        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs+1, steps_per_epoch=len(train_loader),
                                                cycle_momentum=True,
                                                base_momentum=0.85, max_momentum=0.95, last_epoch=args.last_epoch,
                                                div_factor=args.div_factor,
                                                final_div_factor=args.final_div_factor)

        if args.resume != '' and scheduler is not None:
            scheduler.step(args.epoch + 1)

    # switch to train mode
    model.train()
    print(f"args.epoch, {args.epoch}, epochs, {epochs}, args.last_epoch, {args.last_epoch}")

    for epoch in range(args.last_epoch + 1, epochs):
        if args.decay_mode is True:
            if epoch > args.epochs * args.schedule[0] and epoch < args.epochs * args.schedule[1]:
                args.lr -= lr_step
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr

        if should_log:
            writer.add_scalar('learning_rate', args.lr, step)

        # switch to train mode
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        error_ratio = AverageMeter()
        metrics = RunningAverageDict()
        r2_score = R2(device=args.gpu)
        model.train()

        end_load = time.time()
        ########################################## Loop for train #####################################################
        for i, (_, g, h, e, target, _) in enumerate(train_loader):

            data_time.update(time.time() - end_load)
            # Prepare input data
            g, h, e, target = g.to(device), h.to(device), e.to(device), target.to(device)

            optimizer.zero_grad()

            # Compute output
            output = model(g, h, e)
            # Loss

            Train_loss = criterion(output, target)
            # compute gradient and do SGD step
            Train_loss.backward()
            # cut big gradient
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            if args.pytorch_scheduler is True:
                scheduler.step()

            batch_time.update(time.time() - end_load)

            # Logs for average losses and error ratio for all training samples
            losses.update(Train_loss.data.item(), g.size(0))
            r2_score.update((output, target))
            # monitor most important metrics in training process
            error_ratio.update(evaluation(output, target).data, g.size(0))
            metrics.update(compute_errors(output, target), g.size(0))

            if (should_log) and step % 5 == 0:
                writer.add_scalar('Train/loss', Train_loss.item(), step)
                writer.add_scalar('Train/error_ratio', error_ratio.avg.item(), step)
                writer.add_scalars('Train/metrics', metrics.get_value(), step)
                
            
            if should_log and step % 5 == 0:
                wandb.log({f"Train/{args.loss}": Train_loss.item()}, step=step)
                wandb.log({"Train/error_ratio": error_ratio.avg.item()}, step=step)
                if args.pytorch_scheduler is True:
                    wandb.log({"Train/lr": scheduler.get_last_lr()[0]}, step=step)
                elif args.decay_mode is True:
                    wandb.log({"Train/lr": args.lr}, step=step)
                wandb.log({f"Train_/Metrics_{k}": v for k, v in metrics.get_value().items()}, step=step)
                wandb.log({f"Train/r2": r2_score.compute()}, step=step)
            step += 1

            if i % args.log_interval == 0:
                print('\nEpoch / Train: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'R2 {r2_score:.3f}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Error Ratio {err.val:.4f} ({err.avg:.4f})'
                      .format(epoch + 1, i, len(train_loader), batch_time=batch_time,
                              r2_score=r2_score.compute(), loss=losses, err=error_ratio))
            ########################################################################################################
            #################################### Loop for validate ################################################
            if should_write and step % args.validate_every == 0 and step > 0:
                model.eval()
                metrics_eval, val_mse, val_err, val_r2 = validate(args, model, valid_loader, criterion, evaluation, epoch,
                                                          epochs, device)
                print("Validated: {}".format(metrics_eval))

                # log into tensorboard
                if should_log:
                    writer.add_scalar('Valid/loss', losses.avg, step)
                    writer.add_scalar('Valid/error_ratio', error_ratio.avg, step)
                    writer.add_scalars('Valid_/metrics', metrics_eval, step)
                # get the best checkpoint and test it with test set
                if should_log:
                    wandb.log({
                       f"Valid/loss": losses.avg}, step=step)
                    wandb.log({
                       f"Valid/r2": val_r2}, step=step)
                    wandb.log({f"Valid_/Metrics_{k}": v for k, v in metrics_eval.items()}, step=step)

                er1 = metrics_eval["mae"]
                if er1 < best_er1 and should_write:
                    is_best = er1 < best_er1
                    best_er1 = min(er1, best_er1)
                    xyz_utils.save_checkpoint(
                        {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_er1': best_er1,
                         'optimizer': optimizer.state_dict(), }, is_best=is_best, directory=args.directory)
                    del er1, metrics_eval, val_mse, is_best
                model.train()
            #######################################################################################################
            end_load = time.time()
    print('\nEpoch: [{0}] Avg Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}; Avg MAE x Batch {best_er1:.3f}'
          .format(epoch, err=error_ratio, loss=losses, best_er1=best_er1))
    if should_log:
        writer.close()
    return


def validate(args, model, val_loader, criterion, evaluation, epoch, epochs, device='cpu'):
    with torch.no_grad():
        metrics = RunningAverageDict()
        r2_score = R2(device=args.gpu)
        batch_time = AverageMeter()
        val_mse = AverageMeter()
        val_error_ratio = AverageMeter()

        for i, (_, g, h, e, target, _) in tqdm(enumerate(val_loader),
                                               desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation\n",
                                               total=len(val_loader)):

            # Prepare input data

            g, h, e, target = g.to(device), h.to(device), e.to(device), target.to(device)
            # Compute output
            output = model(g, h, e)
            # Logs
            val_mse.update(criterion(output, target).data.item(), g.size(0))
            val_error_ratio.update(evaluation(output, target).data.item(), g.size(0))
            metrics.update(compute_errors(output, target), g.size(0))
            r2_score.update((output, target))
            if i % args.log_interval == 0 and i > 0:
                print('\nValidation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Error Ratio {err.val:.4f} ({err.avg:.4f})'
                      .format(i, len(val_loader), batch_time=batch_time,
                              loss=val_mse, err=val_error_ratio))
        print('Validation * Average Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}'
              .format(err=val_error_ratio, loss=val_mse))

    return metrics.get_value(), val_mse.get_value(), val_error_ratio, r2_score.compute()


def validate_eval(args, model, val_loader, device='cpu'):
    with torch.no_grad():
        metrics = RunningAverageDict()
        val_mse = AverageMeter()
        val_error_ratio = AverageMeter()
        r2_score = R2(device=args.gpu)
        output_array, target_array, id_array = np.array([]), np.array([]), np.array([])

        if args.loss == "MSE":
            criterion = nn.MSELoss()
        elif args.loss == "L1Loss":
            criterion = nn.L1Loss()
        elif args.loss == "SmoothL1Loss":
            criterion = nn.SmoothL1Loss()
        elif args.loss == "HuberLoss":
            criterion = nn.HuberLoss()

        ##################################### Evaluation ##########################################
        evaluation = lambda output, target: torch.mean(torch.abs(output - target) / torch.abs(target))

        for i, (id, g, h, e, target, _) in tqdm(enumerate(val_loader), desc=f"Loop: Evaluation\n",
                                                     total=len(val_loader)):

            # Prepare input data
            g, h, e, target = g.to(device), h.to(device), e.to(device), target.to(device)

            # Compute output
            output = model(g, h, e)
            # Logs
            try:
                # TODO change evaluate size to 1
                target_array = np.concatenate((target_array, target.squeeze().cpu().data.numpy()), axis=0)
                output_array = np.concatenate((output_array, output.squeeze().cpu().data.numpy()), axis=0)
                id_array = np.concatenate((id_array, id.squeeze().cpu().data.numpy()), axis=0)
                val_mse.update(criterion(output, target).data.item(), g.size(0))
                val_error_ratio.update(evaluation(output, target).data.item(), g.size(0))
                r2_score.update((output, target))
                metrics.update(compute_errors(output, target), g.size(0))
                
            except:
                print(f"shape of target array is {target_array.shape}")
                print(f"shape of target shape is {target.shape}")
                if target.shape == torch.Size([1, 1]):
                    target_array = np.concatenate((target_array[..., None], target.cpu().data.numpy()))
                    output_array = np.concatenate((output_array[..., None], output.cpu().data.numpy()), axis=0)
                    id_array = np.concatenate((id_array[..., None], id.cpu().data.numpy()), axis=0)
                    r2_score.update((output, target))
                    val_mse.update(criterion(output, target).data.item(), g.size(0))
                    val_error_ratio.update(evaluation(output, target).data.item(), g.size(0))
                    metrics.update(compute_errors(output, target), g.size(0))
                pass
        metrics = metrics.get_value()
        metrics.update({'r2': r2_score.compute()})
    return metrics, val_mse.get_value(), val_error_ratio.get_value(), output_array, target_array, id_array


def validate_save(args, mode: str, data:str, model:MPNN, loader:PICKLE, data_train:DataLoader, config_model:dict, device='cpu') -> None:
    metrics, val_mse, val_err, output, target, id = validate_eval(args, model, loader, device=args.gpu)

    output = output * data_train.pickle["target_std"] + data_train.pickle["target_mean"]
    target = target * data_train.pickle["target_std"] + data_train.pickle["target_mean"]

    dataname = PurePosixPath(Path(loader.dataset.pickle_file).parts[-1]).stem

    args.directory = Path(args.directory).parent.parent.joinpath(f"Evaluation_{mode}").joinpath(f"{data}_{dataname}")
    evaldir = Path(args.directory).joinpath("plots")
    if Path(evaldir).exists() is False:
        evaldir.mkdir(parents=True, exist_ok=True)
    scatter_histogram_bestfit((output, target), evaldir.joinpath("scatter.png"))
    heatmap((output, target), evaldir)

    csvdir = Path(args.directory).joinpath("eval.csv")
    write_csv(metrics, val_mse, val_err, csvdir)
    print("Validated: {}".format(metrics))

    resultFile = Path(args.directory).joinpath("result.csv")
    csv_result_file(id, target, output, resultFile)
    argsdir = Path(args.directory).joinpath("config.json")
    save_args(args, argsdir)

    modeldir = Path(args.directory).joinpath("model_config.json")
    with open(modeldir, "w") as jsonFile:
        json.dump(config_model, jsonFile, indent=2, sort_keys=True)

    print(f"sort of absolute error is {np.sort(np.abs(target - output))}")
    delete = list(np.abs(target - output).argsort()[-20:])
    outlier_index = []
    for i in delete:
        outlier_index.append(id[i])
    print(f"error should be deleted: {outlier_index}")
    del metrics, val_mse, val_err, output, target, id

if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(description='Neural message passing')

    # dataset
    parser.add_argument('--dataset', default='Dataset2_Modelvariant', help='dataset ID')

    parser.add_argument('--datasetPath',
                        default="data/CSV/clean_good_luck-a34110ac-5d88-4b0c-aec7-88d8d3097b47_reactant-graphsupernode_withSMILES_normALL.pickle")
    parser.add_argument('--split_mode', '--split-mode', default= "no_overlapping",
                        choices=["overlapping", "one_molecule_overlapping", "no_overlapping"], help='dataset path')

    # load model
    parser.add_argument('--resume', default=None,
                        help='path to latest checkpoint')
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")
    parser.add_argument("--name", default="MPNN-Displace-Reaction")

    # Optimization Options
    parser.add_argument('--model_name', '--model-name', type=str, default= "mpnn", choices=["mpnn", "mpnn_global_readout", "mpnn_concat"], metavar='N',
                        help='which model is to be traint.')
    parser.add_argument('--bs', '--batch-size', type=int, default=20, metavar='N',
                        help='Input batch size for training (default: 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Enables CUDA training')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='Number of epochs to train (default: 360)')
    parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-5, 1e-2]), default=5.7976e-05, metavar='LR',
                        help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
    parser.add_argument('--lr-decay', type=lambda x: restricted_float(x, [.01, 1]), default=0.2, metavar='LR-DECAY',
                        help='Learning rate decay factor [.01, 1] (default: 0.6)')
    parser.add_argument('--decay-mode', type=bool, default=True, help='update learning rate mode')
    parser.add_argument('--schedule', type=list, default=[0.1, 0.9], metavar='S',
                        help='Percentage of epochs to start the learning rate decay [0, 1] (default: [0.1, 0.9])')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument("--same-lr", "--same_lr", default=False, action="store_true",
                        help="Use same LR for all param groups")
    parser.add_argument('--div-factor', '--div_factor', default=25, type=float, help="Initial div factor for lr")
    parser.add_argument('--final-div-factor', '--final_div_factor', default=100, type=float,
                        help="final div factor for lr")
    parser.add_argument('--pytorch_scheduler', type=bool, default=False, help='update learning rate mode using pytorch schedule')  

    parser.add_argument("--distributed", default=False, action="store_true", help="Use DDP if set")
    # i/o
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='How many batches to wait before logging training status')
    # Accelerating
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')

    # train
    parser.add_argument('--gpu', default=0, type=int, help='Which gpu to use')
    parser.add_argument("--workers", default=12, type=int, help="Number of workers for data loading")
    parser.add_argument('--validate-every', '--validate_every', default=500, type=int, help='validation period')
    # parser.add_argument('--split', '--split', default=0.8, type=float, help='fraction')
    parser.add_argument('--pathname', '--pn', default=dt.now().strftime('%d-%h-%H-%M'), type=str,
                        help='name for multigpu training')
    parser.add_argument('--run_mode', '--run-mode', default="debug", type=str, help="name of the save folder.")
    parser.add_argument('--tags', default="tune_visualization", type=str, help="name of the tags.")
    parser.add_argument('--notes', default='', type=str, help="wandb notes")
    parser.add_argument('--loss', default='MSE', type=str, choices=["MSE", "L1Loss", "SmoothL1Loss", "HuberLoss"], help="wandb notes")
    parser.add_argument('--wandb', type=bool, default=True, help='enable wandb monitoring process or not.')


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
        print(f"Get nodes list {nodes}.")
        args.world_size = len(nodes)
        args.rank = int(os.environ['SLURM_PROCID'])
    except KeyError as e:
        # We are NOT using SLURM
        args.world_size = 1
        args.rank = 0
        nodes = ["127.0.0.1"]

    if args.distributed:
        mp.set_start_method('forkserver')
        port = np.random.randint(15000, 15025)
        args.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        args.dist_backend = 'nccl'
        args.gpu = None
        print(f"Configuration of distributed training is started.")

    # Configurate gpu
    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)
