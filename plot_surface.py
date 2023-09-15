"""
    Calculate and visualize the loss surface.
    Usage example:
    >>  python plot_surface.py --x=-1:1:101 --y=-1:1:101 --model resnet56
"""
import argparse
import copy
import h5py
import torch
import time
import os
import numpy as np
import torchvision
import torch.nn as nn
from torch.backends import cudnn

import evaluation
import h52vtp
import projection as proj
import net_plotter
import plot_2D
import plot_1D
import model_loader
import scheduler
import mpi4pytorch as mpi
import utils
from dataloader import load_dataset
from utils import log_info as log_fn
from utils import str2bool
from utils_diffusion import get_alphas_and_betas

# mpirun -n 4 pyshifeng -u ./plot_surface.py --mpi True
rank2gpu_map = {
    0: [4],
    1: [5],
    2: [6],
    3: [7],
    4: [2],
    5: [3],
}

def name_surface_file(args, dir_file):
    # skip if surf_file is specified in args
    if args.surf_file:
        return args.surf_file

    # use args.dir_file as the prefix
    surf_file = dir_file

    # resolution
    surf_file += '_[%s,%s,%d]' % (str(args.xmin), str(args.xmax), int(args.xnum))
    if args.y:
        surf_file += 'x[%s,%s,%d]' % (str(args.ymin), str(args.ymax), int(args.ynum))

    # data loder parameters
    if args.raw_data: # without data normalization
        surf_file += '_rawdata'
    if args.data_split > 1:
        surf_file += '_datasplit=' + str(args.data_split) + '_splitidx=' + str(args.split_idx)

    return surf_file + ".h5"

def setup_surface_file(args, surf_file, dir_file):
    # skip if the direction file already exists
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, 'r')
        if (args.y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
            f.close()
            log_fn("%s is already set up" % surf_file)
            return

    f = h5py.File(surf_file, 'a')
    f['dir_file'] = dir_file

    # Create the coordinates(resolutions) at which the function is evaluated
    xcoordinates = np.linspace(args.xmin, args.xmax, num=args.xnum)
    f['xcoordinates'] = xcoordinates

    if args.y:
        ycoordinates = np.linspace(args.ymin, args.ymax, num=args.ynum)
        f['ycoordinates'] = ycoordinates
    f.close()

    return surf_file

def crunch(surf_file, net, w, s, direction, data_loader, loss_key, acc_key, comm, rank, args):
    """
        Calculate the loss values and accuracies of modified models in parallel
        using MPI reduce.
    """

    log_fn(f"crunch(rank={rank})...")
    log_fn(f"  surf_file: {surf_file}") if rank == 0 else None
    log_fn(f"  loss_key : {loss_key}") if rank == 0 else None
    log_fn(f"  acc_key  : {acc_key}") if rank == 0 else None
    fptr = h5py.File(surf_file, 'r')
    xcoordinates = fptr['xcoordinates'][:]
    ycoordinates = fptr['ycoordinates'][:] if 'ycoordinates' in fptr.keys() else None

    if loss_key not in fptr.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates), len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
    else:
        losses = fptr[loss_key][:]
        accuracies = fptr[acc_key][:]

    # close file ASAP. when rank 0 write it, the non-zero rank will not be affected.
    fptr.close()

    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    inds, coords, inds_nums = scheduler.get_job_indices(losses, xcoordinates, ycoordinates, comm)

    msg = f"Rank{rank}: inds:{len(inds)}"
    if len(inds) > 0: msg += f"({inds[0]}~{inds[-1]})"
    msg += f", coords:{len(coords)}, inds_nums:{inds_nums}"
    log_fn(msg)
    start_time = time.time()
    total_sync = 0.0

    criterion = nn.CrossEntropyLoss()
    if args.loss_name == 'mse':
        criterion = nn.MSELoss()

    if args.model == 'diffusion_model':
        beta_schedule = args.diffusion_beta_schedule
        alpha_arr, ab_arr, beta_arr = get_alphas_and_betas(beta_schedule=beta_schedule)
        ab_arr = ab_arr.to(args.device)
        if rank == 0:
            log_fn(f"diffusion_t         : {args.diffusion_t}")
            log_fn(f"beta_schedule       : {beta_schedule}")
            log_fn(f"diffusion_ab_arr    : {len(ab_arr)}")
            log_fn(f"diffusion_ab_arr[0] : {ab_arr[0]:.6f}")
            log_fn(f"diffusion_ab_arr[-1]: {ab_arr[-1]:.6f}")
    else:
        ab_arr = None
    # Loop over all un-calculated loss values
    ind_cnt = len(inds)
    for idx, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        coord = coords[idx]

        # Load the weights corresponding to those coordinates into the net
        real_net = net.module if isinstance(net, torch.nn.DataParallel) else net
        if args.dir_type == 'weights':
            net_plotter.set_weights(real_net, w, direction, coord)
        elif args.dir_type == 'states':
            net_plotter.set_states(real_net, s, direction, coord)
        else:
            raise ValueError(f"Invalid args.dir_type {args.dir_type}")

        if args.model == 'diffusion_model':
            t = args.diffusion_t
            loss, acc = evaluation.eval_loss(net, criterion, data_loader, args.device, t, ab_arr)
        else:
            loss, acc = evaluation.eval_loss(net, criterion, data_loader, args.device)

        # Record the result in the local array
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc

        # Send updated plot data to the master node
        syc_start = time.time()
        losses     = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)
        total_sync += time.time() - syc_start

        # Only the master node writes to the file - this avoids write conflicts
        if rank == 0:
            with h5py.File(surf_file, 'r+') as fptr:
                if loss_key not in fptr.keys():
                    fptr[loss_key] = losses
                    fptr[acc_key] = accuracies
                else:
                    fptr[loss_key][:] = losses
                    fptr[acc_key][:] = accuracies
                fptr.flush()
            # with
        # if rank == 0

        msg = f"R{rank} {idx:3d}/{ind_cnt}: coord=[{coord[0]:5.2f},{coord[1]:5.2f}], " \
              f"loss={loss:.6f}, accu={acc:5.2f}%"
        if rank == 0:
            elp, eta = utils.get_time_ttl_and_eta(start_time, idx, ind_cnt)
            msg += f", elp:{elp}, eta:{eta}"
        log_fn(msg)
    # for

    # This is only needed to make MPI run smoothly. If this process has less work than
    # the rank0 process, then we need to keep calling reduce so the rank0 process doesn't block
    for i in range(max(inds_nums) - len(inds)):
        losses = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)

    total_time = time.time() - start_time
    log_fn('Rank %d done!  Total time: %.2f Sync: %.2f' % (rank, total_time, total_sync))

def parse_args():
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--mpi', '-m', type=str2bool, default=False, help='use mpi')
    parser.add_argument('--cuda', '-c', type=str2bool, default=True, help='use cuda')
    parser.add_argument('--threads', default=2, type=int, help='number of threads')
    parser.add_argument('--batch_size', default=1500, type=int, help='minibatch size')
    parser.add_argument('--seed', default=123, type=int)

    # data parameters
    parser.add_argument('--dataset', default='cifar10', help='cifar10 | imagenet')
    parser.add_argument('--datapath', default='cifar10/data', metavar='DIR', help='path to the dataset')
    parser.add_argument('--raw_data', action='store_true', default=False, help='no data preprocessing')
    parser.add_argument('--data_split', default=1, type=int, help='the number of splits for the dataloader')
    parser.add_argument('--split_idx', default=0, type=int, help='the index of data splits for the dataloader')
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')
    parser.add_argument('--rank0gpu_ids', nargs='*', type=int, default=[])
    parser.add_argument('--rank1gpu_ids', nargs='*', type=int, default=[])
    parser.add_argument('--rank2gpu_ids', nargs='*', type=int, default=[])
    parser.add_argument('--rank3gpu_ids', nargs='*', type=int, default=[])
    parser.add_argument('--rank4gpu_ids', nargs='*', type=int, default=[])
    parser.add_argument('--rank5gpu_ids', nargs='*', type=int, default=[])

    # model parameters
    parser.add_argument('--model', default='diffusion_model', help='model name')
    parser.add_argument('--model_file', default='./checkpoint/cifar10/diffusion_model/ema-cifar10-model-790000.ckpt')
    # parser.add_argument('--model', default='resnet56', help='model name')
    # parser.add_argument('--model_file', default='./checkpoint/temp/resnet56_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7')
    parser.add_argument('--model_file2', default='', help='use (model_file2 - model_file) as the xdirection')
    parser.add_argument('--model_file3', default='', help='use (model_file3 - model_file) as the ydirection')
    parser.add_argument('--model_folder', default='', help='the common folder that contains model_file and model_file2')
    parser.add_argument('--loss_name', '-l', default='crossentropy', help='loss functions: crossentropy | mse')
    parser.add_argument('--diffusion_t', type=int, default=900)
    parser.add_argument('--diffusion_beta_schedule', type=str, default='linear')

    # direction parameters
    parser.add_argument('--dir_file', default='', help='specify the name or path of direction file')
    parser.add_argument('--dir_type', default='weights', help="direction type: weights | states (including BN's running_mean/var)")
    parser.add_argument('--x', default='-1:1:51', help='A string with format xmin:xmax:xnum')
    parser.add_argument('--y', default='-1:1:51', help='A string with format ymin:ymax:ynum')
    parser.add_argument('--xnorm', default='filter', help='direction normalization: filter | layer | weight')
    parser.add_argument('--ynorm', default='filter', help='direction normalization: filter | layer | weight')
    parser.add_argument('--xignore', default='biasbn', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--yignore', default='biasbn', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--same_dir', type=str2bool, default=False, help='use the same random direction for x y axis')
    parser.add_argument('--idx', default=0, type=int, help='the index for the repeatness experiment')
    parser.add_argument('--surf_file', default='', help='the name of surface file, could be an existing file.')

    # plot parameters
    parser.add_argument('--proj_file', default='', help='the .h5 file contains projected optimization trajectory.')
    parser.add_argument('--loss_max', default=5, type=float, help='Maximum value to show in 1D plot')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Minimum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    parser.add_argument('--show', action='store_true', default=False, help='show plotted figures')
    parser.add_argument('--log', action='store_true', default=False, help='use log scale for loss values')
    parser.add_argument('--plot', type=str2bool, default=True, help='plot figures after computation')

    args = parser.parse_args()
    cudnn.benchmark = True

    # --------------------------------------------------------------------------
    # Check plotting resolution
    # --------------------------------------------------------------------------
    try:
        arr = args.x.split(':')
        args.xmin, args.xmax, args.xnum = float(arr[0]), float(arr[1]), int(arr[2])
        args.ymin, args.ymax, args.ynum = (None, None, None)
        if args.y:
            arr = args.y.split(':')
            args.ymin, args.ymax, args.ynum = float(arr[0]), float(arr[1]), int(arr[2])
            assert args.ymin and args.ymax and args.ynum, \
                'You specified some arguments for the y axis, but not all'
    except RuntimeError:
        raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')

    return args

def main():
    args = parse_args()
    # --------------------------------------------------------------------------
    # Environment setup
    # --------------------------------------------------------------------------
    rank2gpu_map[0] = args.rank0gpu_ids or rank2gpu_map[0]
    rank2gpu_map[1] = args.rank1gpu_ids or rank2gpu_map[1]
    rank2gpu_map[2] = args.rank2gpu_ids or rank2gpu_map[2]
    rank2gpu_map[3] = args.rank3gpu_ids or rank2gpu_map[3]
    rank2gpu_map[4] = args.rank4gpu_ids or rank2gpu_map[4]
    rank2gpu_map[5] = args.rank5gpu_ids or rank2gpu_map[5]
    if args.mpi:
        comm = mpi.setup_MPI()
        rank, nproc = comm.Get_rank(), comm.Get_size()
    else:
        comm, rank, nproc = None, 0, 1
    if rank in rank2gpu_map:
        gpu_ids = rank2gpu_map[rank]
        args.gpu_ids = gpu_ids
        args.device = f"cuda:{gpu_ids[0]}"
        args.ngpu = len(gpu_ids)
    else:
        raise ValueError(f"Not found rank {rank} in rank2gpu_map.")
    if rank == 0:
        log_fn(f"======================== args.mpi: {args.mpi} ========================")
    if rank == 0:
        log_fn(f"Rank{rank} cwd : {os.getcwd()}")
        log_fn(f"Rank{rank} args: {args}")
    log_fn(f"Rank{rank} gpu_ids:{args.gpu_ids}. device:{args.device}. ngpu:{args.ngpu}. pid:{os.getpid()}")

    seed = args.seed  # if seed is 0. then ignore it.
    log_fn(f"args.seed : {seed}") if rank == 0 else None
    if seed:
        log_fn(f"  torch.manual_seed({seed})") if rank == 0 else None
        log_fn(f"  np.random.seed({seed})") if rank == 0 else None
        torch.manual_seed(seed)
        np.random.seed(seed)
    if seed and torch.cuda.is_available():
        log_fn(f"  torch.cuda.manual_seed_all({seed})") if rank == 0 else None
        torch.cuda.manual_seed_all(seed)
    log_fn(f"final seed: torch.initial_seed(): {torch.initial_seed()}") if rank == 0 else None

    # --------------------------------------------------------------------------
    # Load models and extract parameters
    # --------------------------------------------------------------------------
    net = model_loader.load(args.dataset, args.model, args.model_file)
    w = net_plotter.get_weights(net) # initial parameters
    s = copy.deepcopy(net.state_dict()) # deepcopy since state_dict are references
    if args.ngpu > 1:
        # data parallel with multiple GPUs on a single node
        net = nn.DataParallel(net, device_ids=args.gpu_ids)
        log_fn(f"Rank {rank}: net = nn.DataParallel(net, device_ids={args.gpu_ids})")

    # --------------------------------------------------------------------------
    # Setup the direction file and the surface file
    # --------------------------------------------------------------------------
    dir_file = net_plotter.name_direction_file(args) # name the direction file
    if rank == 0:
        net_plotter.setup_direction(args, dir_file, net)

    surf_file = name_surface_file(args, dir_file)
    if rank == 0:
        setup_surface_file(args, surf_file, dir_file)

    # wait until master has set up the direction file and surface file
    mpi.barrier(comm)

    # load directions
    d = net_plotter.load_directions(dir_file)
    log_fn(f"Rank{rank}: dir_file: {dir_file}")
    # calculate the cosine similarity of the two directions
    if len(d) == 2 and rank == 0:
        similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
        log_fn('cosine similarity between x-axis and y-axis: %f' % similarity)

    # --------------------------------------------------------------------------
    # Setup dataloader
    # --------------------------------------------------------------------------
    # download CIFAR10 if it does not exit
    if rank == 0 and args.dataset == 'cifar10':
        torchvision.datasets.CIFAR10(root=args.dataset + '/data', train=True, download=True)

    mpi.barrier(comm)

    if rank == 0:
        log_fn(f"load_dataset()...")
        log_fn(f"  args.dataset   : {args.dataset}")
        log_fn(f"  args.datapath  : {args.datapath}")
        log_fn(f"  args.batch_size: {args.batch_size}")
        log_fn(f"  args.threads   : {args.threads}")
    trainloader, testloader = load_dataset(args.dataset, args.datapath,
                                           args.batch_size, args.threads, args.raw_data,
                                           args.data_split, args.split_idx,
                                           args.trainloader, args.testloader)

    # --------------------------------------------------------------------------
    # Start the computation
    # --------------------------------------------------------------------------
    crunch(surf_file, net, w, s, d, trainloader, 'train_loss', 'train_acc', comm, rank, args)
    # crunch(surf_file, net, w, s, d, testloader, 'test_loss', 'test_acc', comm, rank, args)

    # --------------------------------------------------------------------------
    # Plot figures
    # --------------------------------------------------------------------------
    if args.plot and rank == 0:
        if args.y and args.proj_file:
            plot_2D.plot_contour_trajectory(surf_file, dir_file, args.proj_file, 'train_loss', args.show)
        elif args.y:
            plot_2D.plot_2d_contour(surf_file, 'train_loss', args.vmin, args.vmax, args.vlevel, args.show)
            h52vtp.h5_to_vtp(surf_file, 'train_loss', log=True, zmax=-1, interp=-1)
        else:
            plot_1D.plot_1d_loss_err(surf_file, args.xmin, args.xmax, args.loss_max, args.log, args.show)
# main()

if __name__ == '__main__':
    main()
