import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .dataset import Dataset
from .model.model import INNModel
from .utils import Progbar, create_dir, stitch_images, imsave, template_match
from PIL import Image
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from .model.networks import ImagineNet

from .metrics import PSNR


class INN():
    def __init__(self, config):
        self.config = config
        self.model_name = 'INN'
        self.Model = INNModel(config).to(config.DEVICE)
        
        self.psnr = PSNR(255.0).to(config.DEVICE)

        self.train_dataset = Dataset(config, config.TRAIN_FLIST, augment=False, training=True)
        self.val_dataset = Dataset(config, config.VAL_FLIST, augment=False, training=True)
        self.sample_iterator = self.train_dataset.create_iterator(config.SAMPLE_SIZE)
        
        self.samples_path = os.path.join(config.PATH, 'samples_inn')
        if config.CENTER == 1:
            self.results_path = os.path.join(config.PATH, 'result_inn_c')
        else:
            self.results_path = os.path.join(config.PATH, 'result_inn')
        
        self.log_file = os.path.join(config.PATH, 'log-' + self.model_name + '.txt')
        
        self.writer = SummaryWriter(os.path.join(config.PATH, 'runs'))

        # load imagine
        imagine_g_weights_path = os.path.join(config.PATH, 'imagine_g.pth')
        g_data = torch.load(imagine_g_weights_path)
        
        if self.config.CATMASK:
            self.imagine_g = ImagineNet(in_channels=7).to(config.DEVICE)
        else:
            self.imagine_g = ImagineNet(in_channels=3).to(config.DEVICE)
        
        self.imagine_g.load_state_dict(g_data['params'])

    def load(self):
        self.Model.load()

    def save(self):
        self.Model.save()

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
            
            ite = 0
            for it in train_loader:
                self.Model.train()
                data, pdata, pos, fmask_data, half_fmask, mask, z = self.cuda(*it)
                
                if self.config.CATMASK:
                    sf_data = self.imagine_g(torch.cat((pdata, half_fmask, z), dim=1))
                else:
                    sf_data = self.imagine_g(pdata)

                if self.config.DATATYPE == 1:
                    up = nn.Upsample(scale_factor=2, mode='nearest')                
                if self.config.DATATYPE == 2:
                    up = nn.Upsample(size=(256,512), mode='nearest')
      
                sf_data = up(sf_data)

                outputs, d_loss, dp_loss, g_loss, logs = self.Model.process(data, pdata, pos, fmask_data, mask, sf_data)

                self.Model.backward(d_loss, dp_loss, g_loss)
                
                psnr = self.psnr(self.postprocess(data), self.postprocess(outputs))
                mae = (torch.sum(torch.abs(data - outputs)) / torch.sum(data)).float()
                
                iteration = self.Model.iteration
                
                ite = self.Model.iteration
                # ------------------------------------------------------------------------------------
                # end training
                
                if ite >= max_iter:
                    keep_training = False
                    break
                # ------------------------------------------------------------------------------------
                # save log & sample & eval & save model
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))
                    
                logs = [("epoch", epoch), ("iter", ite)] + logs
                self.writer.add_scalars('Discriminator', {'domaink': d_loss}, epoch)
                self.writer.add_scalars('Generator', {'domaink': g_loss}, epoch)
                self.writer.add_scalars('Detail', self.log2dict(logs), epoch)
                
                # progbar
                probar.add(len(data), values=[x for x in logs])
                # log & sample & eval & save model
                if self.config.INTERVAL and ite % self.config.INTERVAL == 0:
                    self.log(logs)
                    self.sample()
                    self.save()

        print('\nEnd trainging...')
        self.writer.close()
    
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
        
        final_results = os.path.join(self.results_path, 'r_know_mask')
        final_results_2 = os.path.join(self.results_path, 'r_unknow_mask')
        masks = os.path.join(self.results_path, 'input')
        state_1_results = os.path.join(self.results_path, 'concept_graph')
        state_rec_results = os.path.join(self.results_path, 'gt')
        
        create_dir(final_results)
        create_dir(final_results_2)
        create_dir(masks)
        create_dir(state_1_results)
        create_dir(state_rec_results)
        
        total = len(self.val_dataset)

        index = 0

        progbar = Progbar(total, width=20, stateful_metrics=['it'])

        for it in test_loader:
            
            # file name
            name = self.val_dataset.load_name(index)
            index += 1

            data, pdata, pos, fmask_data, half_fmask, mask, z = self.cuda(*it)
            
            if self.config.CATMASK:
                sf_data = self.imagine_g(torch.cat((pdata, half_fmask, z), dim=1))
            else:
                sf_data = self.imagine_g(pdata)
                
            if self.config.DATATYPE == 1:
                up = nn.Upsample(scale_factor=2, mode='nearest')                    
            if self.config.DATATYPE == 2:
                up = nn.Upsample(size=(256,512), mode='nearest')

            sf_data = up(sf_data)
    
            o, _ = self.Model(sf_data, pdata, fmask_data, mask, pos)
            final = o * fmask_data + mask
            
            # o2, cal_mask, ones_mask = self.Model(sf_data, pdata)
            # final_2 = o2 * ones_mask + cal_mask
            
            # print(path)
            mask = self.postprocess(mask)[0]
            o_1 = self.postprocess(sf_data)[0]
            final = self.postprocess(final)[0]
            # final_2 = self.postprocess(final_2)[0]
            data = self.postprocess(data)[0]
            
            imsave(mask, os.path.join(masks, name))
            imsave(o_1, os.path.join(state_1_results, name))
            imsave(data, os.path.join(state_rec_results, name))
            imsave(final, os.path.join(final_results, name))
            
            # imsave(final_2, os.path.join(final_results_2, name))

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
        data, pdata, pos, fmask_data, half_fmask, mask, z = self.cuda(*its)
        
        if self.config.CATMASK:
            sf_data = self.imagine_g(torch.cat((pdata, half_fmask, z), dim=1))
        else:
            sf_data = self.imagine_g(pdata)

        if self.config.DATATYPE == 1:
            up = nn.Upsample(scale_factor=2, mode='nearest')
                    
        if self.config.DATATYPE == 2:
            up = nn.Upsample(size=(256,512), mode='nearest')
            
        sf_data = up(sf_data)
        
        o, gate_map = self.Model(sf_data, pdata, fmask_data, mask, pos)
        gate_map = torch.mean(gate_map, 1, True)
        print(gate_map.shape)
        # print(gate_map.shape)
        up_map = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        gate_map = up_map(gate_map)
        final = o * fmask_data + mask
        #_o_1s, _pos = template_match(pdata, sf_data)
        
        # draw sample image
        image_per_row = 2
        images = stitch_images(
            self.postprocess(mask),
            self.postprocess(sf_data),
            self.postprocess(o),
            self.postprocess(gate_map),
            self.postprocess(final),
            self.postprocess(data),
            img_per_row = image_per_row
        )

        path = os.path.join(self.samples_path)
        name = os.path.join(path, str(ite).zfill(5) + '.png')
        create_dir(path)

        print('\nSaving sample images...' + name)
        images.save(name)
