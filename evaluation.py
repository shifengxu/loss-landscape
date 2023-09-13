"""
    The calculation to be performed at each point (modified model), evaluating
    the loss value, accuracy and eigen values of the hessian matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable

from cifar10.models.diffusion_model import DiffusionModel


def eval_loss(net, criterion, loader, device=None, diffusion_t=500, diffusion_ab_arr=None):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        device: GPU device or None
        diffusion_t: timestep t of diffusion model
        diffusion_ab_arr: diffusion alpha_bar array
    Returns:
        loss value and accuracy
    """
    def is_diffusion_model(mdl):
        if isinstance(mdl, torch.nn.DataParallel):
            mdl = mdl.module
        return isinstance(mdl, DiffusionModel)

    correct = 0
    total_loss = 0
    total = 0 # number of samples

    if device:
        net = net.to(device)
    net.eval()

    with torch.no_grad():
        if is_diffusion_model(net): # diffusion model
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                if device:
                    inputs, targets = inputs.to(device), targets.to(device)
                ts = (torch.ones(batch_size) * diffusion_t).long().to(device)   # timestep expand to batch_size
                eps = torch.randn_like(inputs).to(device)                       # epsilon. same shape with inputs
                ab_t = diffusion_ab_arr.index_select(0, ts).view(-1, 1, 1, 1)   # alpha_bar_t
                x_t = inputs * ab_t.sqrt() + eps * (1 - ab_t).sqrt()            # x_t
                outputs = net(x_t, ts)
                loss = (eps - outputs).square().mean()
                total_loss += loss.item() * batch_size

        elif isinstance(criterion, nn.CrossEntropyLoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)
                targets = Variable(targets)
                if device:
                    inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets).sum().item()

        elif isinstance(criterion, nn.MSELoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)

                one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
                one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
                one_hot_targets = one_hot_targets.float()
                one_hot_targets = Variable(one_hot_targets)
                if device:
                    inputs, one_hot_targets = inputs.to(device), one_hot_targets.to(device)
                outputs = F.softmax(net(inputs))
                loss = criterion(outputs, one_hot_targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.cpu().eq(targets).sum().item()

    return total_loss/total, 100.*correct/total
