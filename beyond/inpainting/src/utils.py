import os
import cv2
import sys
import time
import random
import numpy as np
import scipy
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from skimage.color import rgb2gray, gray2rgb

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def same_padding(images, ksizes, strides, rates):   
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images

def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x

def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x

def extract_image_patches(images, ksizes, strides, padding='same'):

    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, [1, 1])
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
            Only "same" or "valid" are supported.'.format(padding))
    batch_size, channel, height, width = images.size()

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                            padding=0,
                            stride=strides)
    patches = unfold(images)
    return patches


def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    height, width = inputs[0][:, :, 0].shape
    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img

# def random_crop(npdata, crop_size, datatype):
    
#     height, width = npdata.shape[0:2]
#     mask = np.ones((height, width))

#     if datatype == 1:
#         h = random.randint(0, height - crop_size)
#         w = random.randint(0, width - crop_size)
#         mask[h: h+crop_size, w: w+crop_size] = 0
#         crop_image = npdata[h: h+crop_size, w: w+crop_size]
    
#     if datatype == 2:
#         h = 0
#         w = random.randint(0, width - crop_size)
#         mask[:, w: w+crop_size] = 0
#         crop_image = npdata[:, w: w+crop_size] 
#     return crop_image, (w, h), mask

def random_crop(npdata, crop_size, datatype, count, pos, known_mask=None):
    
    height, width = npdata.shape[0:2]
    mask = np.ones((height, width))

    if datatype == 1:
        if count == 0 and not known_mask:
            h = random.randint(0, height - crop_size)
            w = random.randint(0, width - crop_size)
        else:
            w, h = pos[0], pos[1]
        mask[h: h+crop_size, w: w+crop_size] = 0
        crop_image = npdata[h: h+crop_size, w: w+crop_size]
    
    if datatype == 2:
        h = 0
        w = (width - crop_size) // 2
        mask[:, w: w+crop_size] = 0
        crop_image = npdata[:, w: w+crop_size] 
    return crop_image, (w, h), mask

def center_crop(npdata, crop_size):
    height, width = npdata.shape[0:2]
    mask = np.ones((height, width))
    w = 64
    h = 64
    mask[h: h+crop_size, w: w+crop_size] = 0

    crop_image = npdata[h: h+crop_size, w: w+crop_size]
    return crop_image, (w, h), mask
    
def side_crop(data, crop_size):
    height, width = data.shape[0:2]
    mask = np.ones((height, width))
    
    w = (width - crop_size) // 2
    h = 0
    mask[:, 0: w] = 0.
    mask[:, w+crop_size:] = 0.
    
    return (w, h), mask

def imshow(img, title=''):
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('off')
    plt.imshow(img, interpolation='none')
    plt.show()


def imsave(img, path):
    im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    im.save(path)

def savetxt(arr, path):
    np.savetxt(path, arr.cpu().numpy().squeeze(), fmt='%.2f')
    
def template_match(target, source):
    locs = []
    _src = []
    for i in range(target.shape[0]):
        src = source[i].detach().cpu().permute(1, 2, 0).numpy()
        tar = target[i].detach().cpu().permute(1, 2, 0).numpy()
        
        src_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        tar_gray = cv2.cvtColor(tar, cv2.COLOR_RGB2GRAY)
        w, h = tar_gray.shape[::-1]

        res = cv2.matchTemplate(src_gray, tar_gray, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, loc = cv2.minMaxLoc(res)
        locs.append(loc)
        
        src = src * 255
        im = Image.fromarray(src.astype(np.uint8).squeeze())
        draw = ImageDraw.Draw(im)
        draw.rectangle([loc, (loc[0] + w, loc[1] + h)], outline=0)
        im = np.array(im)
        _src.append(im)
        
    return torch.Tensor(_src), locs    


def make_mask(data, pdata, pos, device):
    
    crop_size = pdata.shape[3]
    mask_with_pdata = torch.zeros(data.shape).to(device)
    mask_with_ones = torch.ones(data.shape).to(device)

    for po in range(len(pos)):
        w, h = pos[po][0], pos[po][1]
        mask_with_pdata[po, :, h: h+crop_size, w: w+crop_size] = pdata[po]
        mask_with_ones[po, :, h: h+crop_size, w: w+crop_size] = 0

    return mask_with_pdata, mask_with_ones
    

class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)

import math
from torch.optim.optimizer import Optimizer

class Adam16(Optimizer):
  def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    
    defaults = dict(lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay)
    params = list(params)
    super(Adam16, self).__init__(params, defaults)
      
  # Safety modification to make sure we floatify our state
  def load_state_dict(self, state_dict):
    super(Adam16, self).load_state_dict(state_dict)
    for group in self.param_groups:
      for p in group['params']:
        
        self.state[p]['exp_avg'] = self.state[p]['exp_avg'].float()
        self.state[p]['exp_avg_sq'] = self.state[p]['exp_avg_sq'].float()
        self.state[p]['fp32_p'] = self.state[p]['fp32_p'].float()

  def step(self, closure=None):
    """Performs a single optimization step.
    Arguments:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    """
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
          
        grad = p.grad.data.float()
        state = self.state[p]

        # State initialization
        if len(state) == 0:
          state['step'] = 0
          # Exponential moving average of gradient values
          state['exp_avg'] = grad.new().resize_as_(grad).zero_()
          # Exponential moving average of squared gradient values
          state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()
          # Fp32 copy of the weights
          state['fp32_p'] = p.data.float()

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        if group['weight_decay'] != 0:
          grad = grad.add(group['weight_decay'], state['fp32_p'])

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
      
        state['fp32_p'].addcdiv_(-step_size, exp_avg, denom)
        p.data = state['fp32_p'].float()

    return loss

def PositionEmbeddingSine():
    temperature = 10000
    feature_h = 256 // 2**2
    num_pos_feats = 64 * (2**2) // 2
    mask = torch.ones((feature_h, feature_h))
    y_embed = mask.cumsum(0, dtype=torch.float32)
    x_embed = mask.cumsum(1, dtype=torch.float32)

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
    return pos

def PatchPositionEmbeddingSine():
    temperature = 10000
    feature_h = 256//8
    num_pos_feats = 8*8*3//2
    mask = torch.ones((feature_h, feature_h))
    y_embed = mask.cumsum(0, dtype=torch.float32)
    x_embed = mask.cumsum(1, dtype=torch.float32)
    # if self.normalize:
    #     eps = 1e-6
    #     y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
    #     x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
    return pos