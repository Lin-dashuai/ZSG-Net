"""ZSG single-image SR cycle training, adapted from the supplied SR experiment. Active names use ZSG; see release notes and upstream notices."""
import itertools
import os
import time
import cv2
import numpy as np
import torch
from torch import nn
from PIL import Image
from N_Shape04 import N_Unet
from sr_config import Config
from sr_resize import imresize
from sr_utils import ReplayBuffer, preprocess_kernels, back_projection
from sr_models import degradation, UNetDiscriminatorAesrgan
from sr_vgg import Vgg19
from sr_degradation import degradation_bsrgan
from zsg_utils import resolve_device

class ZSG:

    def __init__(self, input_img, conf=None, ground_truth=None, kernels=None):
        conf = Config() if conf is None else conf
        self.conf = conf
        self.sf = None
        self.output_shape = None
        self.base_sf = 1.0
        self.sf_ind = 0
        if conf.scale_factors != [[4.0, 4.0]] or conf.base_change_sfs:
            raise ValueError('The supplied experiment supports a single fourfold scale with fixed 192x192 training geometry.')
        self.device = resolve_device(conf.device)
        self.cuda = self.device.type == 'cuda'
        self.input = self.read_image(input_img)
        self.input = cv2.resize(self.input, (192, 192))
        self.Y = False
        self.gt = None if ground_truth is None else self.read_image(ground_truth)
        if self.gt is not None and self.gt.shape != (768, 768, 3):
            raise ValueError('Ground truth must be aligned RGB at the final 768x768 resolution; it is never resized for scoring.')
        self.kernels = preprocess_kernels(kernels, conf)
        self.model_L2H = N_Unet(3, 3, encoder_weights=conf.encoder_weights).to(self.device)
        self.model_H2L = degradation(encoder_weights=conf.encoder_weights).to(self.device)
        self.net_vgg = Vgg19(weights=conf.vgg_weights).to(self.device)
        self.Discriminator_L = UNetDiscriminatorAesrgan().to(self.device)
        self.Discriminator_H = UNetDiscriminatorAesrgan().to(self.device)
        self.fake_L_buffer = ReplayBuffer()
        self.fake_H_buffer = ReplayBuffer()
        self.history = []
        self.timings = {}
        self.init_parameters()
        self.hr_fathers_sources = [self.input]
        self.file_name = str(input_img) if isinstance(input_img, (str, os.PathLike)) else conf.name

    @staticmethod
    def read_image(value):
        if isinstance(value, (str, os.PathLike)):
            with Image.open(value) as image:
                return np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
        image = np.asarray(value)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError('Expected a grayscale, RGB, or RGBA image.')
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        image = image[:, :, :3].astype(np.float32)
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError('Floating-point image arrays must be finite and in [0, 1].')
        return image

    def run(self):
        for self.sf_ind, (sf, self.kernel) in enumerate(zip(self.conf.scale_factors, self.kernels)):
            print('self_kernels:', self.kernels)
            print('self.sf_ind', self.sf_ind)
            print('*** Start training for sf=', sf, ' ***')
            train_start = time.perf_counter()
            if np.isscalar(sf):
                sf = [sf, sf]
            self.sf = np.array(sf) / np.array(self.base_sf)
            self.output_shape = np.ceil(np.array(self.input.shape[0:2]) * sf).astype(np.int64)
            print('input shape', self.input.shape)
            self.init_parameters()
            self.train()
            train_time_ms = (time.perf_counter() - train_start) * 1000
            self.timings['training_seconds'] = train_time_ms / 1000
            print(f'Training time: {train_time_ms:.3f} ms')
            test_start = time.perf_counter()
            post_processed_output = self.final_test()
            test_time_ms = (time.perf_counter() - test_start) * 1000
            self.timings['reconstruction_seconds'] = test_time_ms / 1000
            print(f'Testing time: {test_time_ms:.3f} ms')
            self.hr_fathers_sources.append(post_processed_output)
            if self.conf.save_results:
                sf_str = ''.join(('X%.2f' % s for s in self.conf.scale_factors[self.sf_ind]))
                os.makedirs(self.conf.result_path, exist_ok=True)
                self.output_file = '%s/%s_zsg_%s.png' % (self.conf.result_path, os.path.splitext(os.path.basename(self.file_name))[0], sf_str)
                Image.fromarray(np.round(np.clip(post_processed_output, 0, 1) * 255).astype(np.uint8)).save(self.output_file)
            if self.conf.save_checkpoint:
                path = os.path.join(self.conf.result_path, os.path.splitext(os.path.basename(self.file_name))[0] + '_zsg.pth')
                torch.save({'model_L2H': self.model_L2H.state_dict(), 'model_H2L': self.model_H2L.state_dict(), 'Discriminator_L': self.Discriminator_L.state_dict(), 'Discriminator_H': self.Discriminator_H.state_dict()}, path)
            print('** Done training for sf=', sf, ' **')
        return post_processed_output

    def init_parameters(self):
        self.loss = [None] * self.conf.max_iters
        self.mse, self.mse_rec, self.interp_mse, self.interp_rec_mse, self.mse_steps = ([], [], [], [], [])
        self.iter = 0
        self.learning_rate = self.conf.learning_rate
        self.learning_rate_change_iter_nums = [0]
        self.gt_per_sf = imresize(self.gt, scale_factor=self.sf / self.conf.scale_factors[-1], output_shape=self.output_shape, kernel=self.conf.downscale_gt_method) if self.gt is not None and self.sf is not None and np.any(np.abs(self.sf - self.conf.scale_factors[-1]) > 0.01) else self.gt

    def forward_backward_pass(self, lr_son, hr_father, criterion, optimizers):
        interpolated_lr_son = imresize(lr_son, self.sf, hr_father.shape, self.conf.upscale_method)
        if self.Y == True:
            lr_son_input = torch.Tensor(interpolated_lr_son).unsqueeze_(0).unsqueeze_(0)
            hr_father = torch.Tensor(hr_father).unsqueeze_(0).unsqueeze_(0)
        else:
            lr_son_input = torch.Tensor(interpolated_lr_son).permute(2, 0, 1).unsqueeze_(0)
            hr_father = torch.Tensor(hr_father).permute(2, 0, 1).unsqueeze_(0)
        lr_son_input = lr_son_input.requires_grad_()
        target_real = torch.ones((1, 1, 192, 192), device=self.device)
        target_fake = torch.zeros_like(target_real)
        z = np.random.randn(1, 1, 192).astype(np.float32)
        hr_father = hr_father.to(self.device)
        lr_son_input = lr_son_input.to(self.device)
        z = torch.from_numpy(z).to(self.device)
        Gen_hr = self.model_L2H(lr_son_input)
        Gen_lr = self.model_H2L(hr_father, z)
        cycle_hr = self.model_L2H(Gen_lr)
        cycle_lr = self.model_H2L(Gen_hr, z)
        fake_H_224 = torch.nn.functional.interpolate(Gen_hr, 224, mode='bicubic', align_corners=True)
        real_H_224 = torch.nn.functional.interpolate(hr_father, 224, mode='bicubic', align_corners=True)
        vgg_recon_emb = self.net_vgg((fake_H_224 + 1) / 2)
        vgg_target_emb = self.net_vgg((real_H_224 + 1) / 2)
        loss_feature_H = nn.MSELoss()(vgg_recon_emb, vgg_target_emb)
        fake_L_224 = torch.nn.functional.interpolate(Gen_lr, 224, mode='bicubic', align_corners=True)
        real_L_224 = torch.nn.functional.interpolate(lr_son_input, 224, mode='bicubic', align_corners=True)
        vgg_recon_emb = self.net_vgg((fake_L_224 + 1) / 2)
        vgg_target_emb = self.net_vgg((real_L_224 + 1) / 2)
        loss_feature_L = nn.MSELoss()(vgg_recon_emb, vgg_target_emb)
        pred_hr = self.Discriminator_H(Gen_hr)
        loss_GAN_L2H = nn.MSELoss()(pred_hr, target_real)
        pred_lr = self.Discriminator_L(Gen_lr)
        loss_GAN_H2L = nn.MSELoss()(pred_lr, target_real)
        loss_GAN = loss_GAN_L2H + loss_GAN_H2L
        loss_feature = loss_feature_L + loss_feature_H
        l1_loss = criterion(hr_father, Gen_hr) + criterion(lr_son_input, Gen_lr)
        cycle_loss = criterion(hr_father, cycle_hr) + criterion(lr_son_input, cycle_lr)
        loss = cycle_loss + l1_loss + loss_feature + loss_GAN * 0.15
        optimizers[0].zero_grad()
        loss.backward()
        optimizers[0].step()
        self.loss[self.iter] = float(loss.detach().item())
        optimizers[1].zero_grad()
        pred_real = self.Discriminator_L(lr_son_input)
        loss_D_real = nn.MSELoss()(pred_real, target_real)
        fake_L = self.fake_L_buffer.push_and_pop(Gen_lr)
        pred_fake = self.Discriminator_L(fake_L.detach())
        loss_D_fake = nn.MSELoss()(pred_fake, target_fake)
        loss_D_L = loss_D_fake + loss_D_real
        loss_D_L.backward()
        optimizers[1].step()
        optimizers[2].zero_grad()
        pred_real = self.Discriminator_H(hr_father)
        loss_D_real = nn.MSELoss()(pred_real, target_real)
        fake_H = self.fake_H_buffer.push_and_pop(Gen_hr)
        pred_fake = self.Discriminator_H(fake_H.detach())
        loss_D_fake = nn.MSELoss()(pred_fake, target_fake)
        loss_D_H = loss_D_fake + loss_D_real
        loss_D_H.backward()
        optimizers[2].step()
        self.history.append(dict(iteration=self.iter, learning_rate=optimizers[0].param_groups[0]['lr'], total=self.loss[self.iter], cycle=float(cycle_loss.detach().item()), l1=float(l1_loss.detach().item()), feature=float(loss_feature.detach().item()), gan=float(loss_GAN.detach().item()), discriminator_l=float(loss_D_L.detach().item()), discriminator_h=float(loss_D_H.detach().item())))
        return [value.detach() for value in (lr_son_input, hr_father, Gen_lr, Gen_hr)]

    def forward_pass(self, lr_son, hr_father_shape=None):
        interpolated_lr_son = imresize(lr_son, self.sf, hr_father_shape, self.conf.upscale_method)
        if self.Y == True:
            interpolated_lr_son = torch.Tensor(interpolated_lr_son).unsqueeze_(0).unsqueeze_(0)
        else:
            interpolated_lr_son = torch.Tensor(interpolated_lr_son).permute(2, 0, 1).unsqueeze_(0)
        interpolated_lr_son = interpolated_lr_son.to(self.device)
        was_training = self.model_L2H.training
        self.model_L2H.eval()
        try:
            with torch.inference_mode():
                output = self.model_L2H(interpolated_lr_son)
            return np.clip(output.cpu().permute(0, 2, 3, 1).numpy()[0], 0, 1)
        finally:
            self.model_L2H.train(was_training)

    def learning_rate_policy(self):
        if not (1 + self.iter) % self.conf.learning_rate_policy_check_every and self.iter - self.learning_rate_change_iter_nums[-1] > self.conf.min_iters:
            count = self.conf.learning_rate_slope_range // self.conf.run_test_every
            if len(self.mse_steps[-count:]) < 3:
                return
            [slope, _], [[var, _], _] = np.polyfit(self.mse_steps[-(self.conf.learning_rate_slope_range // self.conf.run_test_every):], self.mse_rec[-(self.conf.learning_rate_slope_range // self.conf.run_test_every):], 1, cov=True)
            std = np.sqrt(var)
            print('slope: ', slope, 'STD: ', std)
            if -self.conf.learning_rate_change_ratio * slope < std:
                self.learning_rate /= 10
                print('learning rate updated: ', self.learning_rate)
                self.learning_rate_change_iter_nums.append(self.iter)

    def quick_test(self):
        self.sr = self.forward_pass(self.input)
        self.mse = self.mse + [np.mean(np.ndarray.flatten(np.square(self.gt_per_sf - self.sr)))] if self.gt_per_sf is not None else None
        self.reconstruct_output = self.forward_pass(self.father_to_son(self.input), self.input.shape)
        self.mse_rec.append(np.mean(np.ndarray.flatten(np.square(self.input - self.reconstruct_output))))
        interp_sr = imresize(self.input, self.sf, self.output_shape, self.conf.upscale_method)
        self.interp_mse = self.interp_mse + [np.mean(np.ndarray.flatten(np.square(self.gt_per_sf - interp_sr)))] if self.gt_per_sf is not None else None
        interp_rec = imresize(self.father_to_son(self.input), self.sf, self.input.shape[0:2], self.conf.upscale_method)
        self.interp_rec_mse.append(np.mean(np.ndarray.flatten(np.square(self.input - interp_rec))))
        self.mse_steps.append(self.iter)
        if self.conf.display_test_results:
            print('iteration: ', self.iter, 'reconstruct mse:', self.mse_rec[-1], ', true mse:', self.mse[-1] if self.mse else None)

    def train(self):
        criterion = nn.L1Loss()
        optimizer_G = torch.optim.Adam(itertools.chain(self.model_L2H.parameters(), self.model_H2L.parameters()), lr=self.learning_rate)
        optimizer_D_L = torch.optim.Adam(self.Discriminator_L.parameters(), lr=self.learning_rate)
        optimizer_D_H = torch.optim.Adam(self.Discriminator_H.parameters(), lr=self.learning_rate)
        optimizers = [optimizer_G, optimizer_D_L, optimizer_D_H]
        for self.iter in range(self.conf.max_iters):
            self.hr_father = self.input
            self.lr_son = self.father_to_son(self.hr_father)
            self.train_output = self.forward_backward_pass(self.lr_son, self.hr_father, criterion, optimizers)
            if not self.iter % self.conf.display_every:
                print('sf:', self.sf * self.base_sf, ', iteration: ', self.iter, ', loss: ', self.loss[self.iter])
            if self.conf.run_test and (not self.iter % self.conf.run_test_every):
                self.quick_test()
            self.learning_rate_policy()
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group['lr'] = self.learning_rate
            if self.learning_rate < self.conf.min_learning_rate:
                break

    def father_to_son(self, hr_father):
        lr_son, hq = degradation_bsrgan(hr_father)
        return np.clip(lr_son + np.random.randn(*lr_son.shape) * self.conf.noise_std, 0, 1)

    def final_test(self):
        outputs = []
        for k in range(0, 1 + 7 * self.conf.output_flip, 1 + int(self.sf[0] != self.sf[1])):
            test_input = np.rot90(self.input, k) if k < 4 else np.fliplr(np.rot90(self.input, k))
            tmp_output = self.forward_pass(test_input)
            tmp_output = np.rot90(tmp_output, -k) if k < 4 else np.rot90(np.fliplr(tmp_output), -k)
            for bp_iter in range(self.conf.back_projection_iters[self.sf_ind]):
                tmp_output = back_projection(tmp_output, self.input, down_kernel=self.kernel, up_kernel=self.conf.upscale_method, sf=self.sf)
            outputs.append(tmp_output)
        almost_final_sr = np.median(outputs, 0)
        for bp_iter in range(self.conf.back_projection_iters[self.sf_ind]):
            almost_final_sr = back_projection(almost_final_sr, self.input, down_kernel=self.kernel, up_kernel=self.conf.upscale_method, sf=self.sf)
        self.final_sr = almost_final_sr
        return self.final_sr
