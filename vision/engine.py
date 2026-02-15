# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
from base_linear import QuantizedLinear

from contextlib import nullcontext
from typing import Callable
import wandb


def train_one_epoch(model: Callable, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler: torch.cuda.amp.GradScaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None, not_compiled_model: torch.nn.Module=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    i = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
        
        ctx = None
        if args.dtype == "float16":
            ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        elif args.dtype == "bfloat16":
            ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
            ctx = nullcontext()
        with ctx:
            outputs = model(samples)
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        if args.dtype in ["float16"]:
            loss_scaler.scale(loss).backward()
            loss_scaler.unscale_(optimizer)
            if max_norm is not None and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            loss_scaler.step(optimizer)
            loss_scaler.update()
        else:
            loss.backward()
            if max_norm is not None and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

        all_sxs = {'epoch_': epoch + i / len(data_loader)}
        with torch.no_grad():
            sum_grad_square = 0
            for name, parameter in not_compiled_model.named_parameters():
                if parameter.requires_grad:
                    sum_grad_square += parameter.grad.nan_to_num().square().sum()
            metric_logger.update(gn=sum_grad_square.sqrt().item())

            sum_sw = 0
            sum_sw2 = 0
            min_sw = torch.tensor(math.inf)
            min_sw2 = torch.tensor(math.inf)
            sumentw = 0
            num_w = 0

            sum_sx = 0
            sum_sx2 = 0
            min_sx = torch.tensor(math.inf)
            min_sx2 = torch.tensor(math.inf)
            sumentx = 0
            num_x = 0

            sum_sa = 0
            sum_sa2 = 0
            min_sa = torch.tensor(math.inf)
            min_sa2 = torch.tensor(math.inf)
            sumenta = 0
            num_a = 0
            
            for name, module in not_compiled_model.named_modules():
                if name.endswith("_quantizer"):
                    if name.endswith("weight_quantizer"):

                        all_sxs[f"wmul/{name}"] = module.mul_step.abs().mean()
                        all_sxs[f"wdiv/{name}"] = module.div_step.abs().mean()
                        my_sw = module.mul_step.abs()
                        min_sw = torch.minimum(min_sw, my_sw.min())
                        sum_sw += my_sw.mean()

                        my_sw2 = module.div_step.abs()
                        min_sw2 = torch.minimum(min_sw2, my_sw2.min())
                        sum_sw2 += my_sw2.mean()

                        sumentw += module.ent

                        num_w += 1
                    elif name.endswith("activation_quantizer"):

                        all_sxs[f"xmul/{name}"] = module.mul_step.abs().mean()
                        all_sxs[f"xdiv/{name}"] = module.div_step.abs().mean()
                        my_sx = module.mul_step.abs()
                        min_sx = torch.minimum(min_sx, my_sx.min())
                        sum_sx += my_sx.mean()

                        my_sx2 = module.div_step.abs()
                        min_sx2 = torch.minimum(min_sx2, my_sx2.min())
                        sum_sx2 += my_sx2.mean()

                        sumentx += module.ent

                        num_x += 1
                    else:

                        all_sxs[f"amul/{name}"] = module.mul_step.abs().mean()
                        all_sxs[f"adiv/{name}"] = module.div_step.abs().mean()
                        my_sa = module.mul_step.abs()
                        min_sa = torch.minimum(min_sa, my_sa.min())
                        sum_sa += my_sa.mean()

                        my_sa2 = module.div_step.abs()
                        min_sa2 = torch.minimum(min_sa2, my_sa2.min())
                        sum_sa2 += my_sa2.mean()

                        sumenta += module.ent

                        num_a += 1
            
            if not not args.log_every:
                wandb.log(all_sxs)
            metric_logger.update(msx=min_sx.item())
            metric_logger.update(asx=(sum_sx / num_x).item())
            metric_logger.update(msx2=min_sx2.item())
            metric_logger.update(asx2=(sum_sx2 / num_x).item())
            metric_logger.update(entx=(sumentx / num_x).item())
            metric_logger.update(msw=min_sw.item())
            metric_logger.update(asw=(sum_sw / num_w).item())
            metric_logger.update(msw2=min_sw2.item())
            metric_logger.update(asw2=(sum_sw2 / num_w).item())
            metric_logger.update(entw=(sumentw / num_w).item())
            if num_a > 0:
                metric_logger.update(msa=min_sa.item())
                metric_logger.update(asa=(sum_sa / num_a).item())
                metric_logger.update(msa2=min_sa2.item())
                metric_logger.update(asa2=(sum_sa2 / num_a).item())
                metric_logger.update(enta=(sumenta / num_a).item())

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        i += 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        ctx = None
        if args.dtype == "float16":
            ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        elif args.dtype == "bfloat16":
            ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
            ctx = nullcontext()
        with ctx:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
