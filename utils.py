import argparse
import datetime
import math
import sys
import time
import torch

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def log_info(*args):
    dtstr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    # MPI is multiple-process, so need to flush every time.
    # If use print(...), and there are 4 processes, the output will be messy,
    # and different lines mixed together in output.
    sys.stdout.write(f"[{dtstr}] {args[0]}\r\n")

def get_time_ttl_and_eta(time_start, elapsed_iter, total_iter):
    """
    Get estimated total time and ETA time.
    :param time_start:
    :param elapsed_iter:
    :param total_iter:
    :return: string of elapsed time, string of ETA
    """

    def sec_to_str(sec):
        val = int(sec)  # seconds in int type
        s = val % 60
        val = val // 60  # minutes
        m = val % 60
        val = val // 60  # hours
        h = val % 24
        d = val // 24  # days
        return f"{d}-{h:02d}:{m:02d}:{s:02d}"

    elapsed_time = time.time() - time_start  # seconds elapsed
    elp = sec_to_str(elapsed_time)
    if elapsed_iter == 0:
        eta = 'NA'
    else:
        # seconds
        eta = elapsed_time * (total_iter - elapsed_iter) / elapsed_iter
        eta = sec_to_str(eta)
    return elp, eta

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def count_parameters(model: torch.nn.Module, log_fn=None):
    def prt(x):
        if log_fn: log_fn(x)

    def convert_size_str(size_bytes):
        if size_bytes == 0:
            return "0B"
        size_name = ("", "K", "M", "G", "T", "P", "E", "Z", "Y")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return "%s %s" % (s, size_name[i])

    prt(f"count_parameters({type(model)}) ------------")
    prt('  requires_grad  name  count  size')
    counter = 0
    for name, param in model.named_parameters():
        s_list = list(param.size())
        prt(f"  {param.requires_grad} {name} {param.numel()} = {s_list}")
        c = param.numel()
        counter += c
    # for
    str_size = convert_size_str(counter)
    prt(f"  total  : {counter} {str_size}")
    return counter, str_size

def read_lines(f_path):
    # read dir list from file, where line is a dir path.
    # remove trailing \r\n from each line, and ignore lines starting with '#'
    with open(f_path) as fptr: lines = fptr.readlines()
    new_lst = []
    for line in lines:
        line = line.strip()
        if line == "": continue
        if line.startswith('#'): continue
        new_lst.append(line)
    # for
    return new_lst
