import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .dataset import Dataset
from .model.model import ImagineModel
from .utils import Progbar, create_dir, stitch_images, imsave, template_match, PositionEmbeddingSine, PatchPositionEmbeddingSine
from PIL import Image
# from tensorboardX import SummaryWriter
import torch.nn.functional as F

from .metrics import PSNR

class ImagineGAN():
    def __init__(self, config):
        self.config = config
        self.model_name = 'ImagineModel'
        self.Model = ImagineModel(config).to(config.DEVICE)

        postion_embedding = PositionEmbeddingSine()
        patch_pos = PatchPositionEmbeddingSine()
        self.Model.set_position(postion_embedding, patch_pos=patch_pos, batch_size=self.config.BATCH_SIZE)
        
        self.psnr = PSNR(255.0).to(config.DEVICE)

        self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.MASK_FLIST, augment=False, training=True)
        self.val_dataset = Dataset(config, config.VAL_FLIST, config.MASK_VAL_FLIST, augment=False, training=False)
        self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)
        
        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')
        
        self.log_file = os.path.join(config.PATH, 'log-' + self.model_name + '.txt')
        
        # self.writer = SummaryWriter(os.path.join(config.PATH, 'runs'))
        
    def load(self):
        self.Model.load()

    def save(self, ite):
        self.Model.save(ite)

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=False,
            shuffle=True,
            pin_memory=True
        )

        epoch = 0
        keep_training = True

        max_iter = int(self.config.MAX_ITERS)
        total = len(self.train_dataset)

        while(keep_training):
            epoch += 1
            
            probar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter', 'mean_gate', 'max_gate', 'min_gate'])
            
            ite = self.Model.iteration

            for it in train_loader:
                self.Model.train()
                data, pdata, fmask_data, mask = self.cuda(*it)

                outputs, g_loss, logs = self.Model.process(data, pdata, fmask_data, ite)
  
                self.Model.backward(g_loss)
                
                psnr = self.psnr(self.postprocess(data), self.postprocess(outputs))
                mae = (torch.sum(torch.abs(data - outputs)) / torch.sum(data)).float()
                
                ite = self.Model.iteration
                
                # ------------------------------------------------------------------------------------
                # end training
                
                if ite > max_iter:
                    keep_training = False
                    break

                # ------------------------------------------------------------------------------------
                # save log & sample & eval & save model
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))
                    
                logs = [("epoch", epoch), ("iter", ite)] + logs
                # self.writer.add_scalars('Discriminator', {'domaink': d_loss}, epoch)
                # self.writer.add_scalars('Generator', {'domaink': g_loss}, epoch)
                # self.writer.add_scalars('Detail', self.log2dict(logs), epoch)
                
                # progbar
                probar.add(len(data), values=[x for x in logs])

                if self.config.INTERVAL and ite % self.config.INTERVAL == 0:
                    self.log(logs)
                    self.sample()

                if self.config.SAVE_INTERAL and ite % self.config.SAVE_INTERAL == 0:
                    self.save(ite)

        print('\nEnd trainging...')
        # self.writer.close()
    
    def log2dict(self, logs):
        dict = {}
        for i in range(2, len(logs)):
            dict[logs[i][0]] = logs[i][1]
        return dict
    
    def test(self):
        test_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=1
        )

        create_dir(self.results_path)
        
        input_data = os.path.join(self.results_path, 'input')
        gt_data = os.path.join(self.results_path, 'gt')
        masks = os.path.join(self.results_path, 'mask_gt')
        state_1_results = os.path.join(self.results_path, 'output')
        state_rec_results = os.path.join(self.results_path, 'output_blackbox')
        
        create_dir(input_data)
        create_dir(gt_data)
        create_dir(masks)
        create_dir(state_1_results)
        create_dir(state_rec_results)
        
        total = len(self.val_dataset)

        index = 0

        progbar = Progbar(total, width=20, stateful_metrics=['it'])

        postion_embedding = PositionEmbeddingSine()
        patch_pos = PatchPositionEmbeddingSine()
        self.Model.set_position(postion_embedding, patch_pos=patch_pos, batch_size=1)

        for it in test_loader:
            
            # file name
            # name = str(index) + '_' + self.val_dataset.load_name(index)
            # name = self.val_dataset.load_name(index) + '_' + str(index) + '_' + self.val_dataset.load_mask_name(index)
            # name = self.val_dataset.load_name(index)
            
            # name = self.val_dataset.load_name(index) + '_' + self.val_dataset.load_mask_name(index)
            index += 1
            name = 'img_{:d}.png'.format(index)

            data, pdata, fmask_data, mask = self.cuda(*it)
            o = self.Model(pdata, fmask_data)

            o = fmask_data * o + data * (1 - fmask_data)
            
            data = self.postprocess(data)[0]
            pdata = self.postprocess(pdata)[0]
            mask = self.postprocess(mask)[0]
            o = self.postprocess(o)[0]
  
            imsave(pdata, os.path.join(input_data, name))
            imsave(data, os.path.join(gt_data, name))
            imsave(mask, os.path.join(masks, name))
            imsave(o, os.path.join(state_1_results, name))

            print(index, name)

        print('\nEnd test....')
        
    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\r\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()      

    def sample(self):

        ite = self.Model.iteration
        its = next(self.sample_iterator)
        
        data, pdata, fmask_data, mask = self.cuda(*its)
        
        # input = torch.cat((pdata, fmask_data), dim=1)
        postion_embedding = PositionEmbeddingSine()
        patch_pos = PatchPositionEmbeddingSine()
        self.Model.set_position(postion_embedding, patch_pos=patch_pos, batch_size=1)
        o_1s = self.Model(pdata, fmask_data)

        up = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)

        o = o_1s * fmask_data + data * (1 - fmask_data)

        # draw sample image
        image_per_row = 1
        images = stitch_images(
            self.postprocess(mask),
            self.postprocess(o_1s),
            self.postprocess(o),
            self.postprocess(data),
            img_per_row = image_per_row
        )

        path = os.path.join(self.samples_path)
        name = os.path.join(path, str(ite).zfill(5) + '.png')
        create_dir(path)

        print('\nSaving sample images...' + name)
        images.save(name)