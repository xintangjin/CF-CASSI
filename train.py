import os
from option import opt
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

from architecture import *
from utils import *
import torch
import scipy.io as scio
import time
import numpy as np
from torch.autograd import Variable
import datetime

import torch.nn.functional as F
import losses
import torch.nn as nn

# MULTI_GPU = True
# device_ids = [0, 1]
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

nm_cave = np.float32([453.3, 457.6, 462.1, 466.8, 471.6, 476.5, 481.6, 486.9, 492.4, 498.0, 503.9, 509.9, 516.2, 522.7, 529.5, 536.5, 543.8,
551.4, 558.6, 567.5, 575.3, 584.3, 594.4, 604.2, 614.4, 625.1, 636.3, 648.1])
k = 11
# init mask
mask3d_batch_train, input_mask_train = init_mask(opt.mask_path, opt.input_mask, opt.batch_size)
mask3d_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, 10)

# dataset
train_set, train_curve = LoadTraining(opt.data_path)
test_data, test_curve = LoadTest(opt.test_path)
test_curve = test_curve.cuda().float()
# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + date_time + '/result/'
model_path = opt.outf + date_time + '/model/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# model
if opt.two_stages:
    if opt.method == 'hdnet':
        model, FDL_loss = model_generator(opt.method, opt.pretrained_model_path).cuda()
    else:
        model = model_generator(opt.method, opt.pretrained_model_path).cuda()


model_hyper = model_hyper_generator(opt.multi2hyper, opt.pretrained_hypermodel_path, k=k).cuda()

# optimizing
if opt.two_stages:
    if opt.train_method:
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
        if opt.scheduler=='MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
        elif opt.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)
    else:
        model.eval()
else:
    pass
if opt.multi2hyper.find('Tradition') >= 0:
    pass
else:
    optimizer_hyper = torch.optim.Adam(model_hyper.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
    if opt.scheduler=='MultiStepLR':
        scheduler_lines = torch.optim.lr_scheduler.MultiStepLR(optimizer_hyper, milestones=opt.milestones, gamma=opt.gamma)
    elif opt.scheduler=='CosineAnnealingLR':
        scheduler_lines = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_hyper, opt.max_epoch, eta_min=1e-6)

mse = torch.nn.MSELoss().cuda()
if opt.loss_type=='MSE':
    get_loss = torch.nn.MSELoss().cuda()
elif opt.loss_type=='CharbonnierLoss':
    get_loss = losses.CharbonnierLoss().cuda()
get_point_loss = losses.Curve_Loss().cuda()
get_loss_L0 = losses.Curve_L0_Loss().cuda()
get_loss_L1 = losses.Curve_L1_Loss().cuda()
get_loss_L2 = losses.Curve_L2_Loss().cuda()


xi = torch.from_numpy(nm_cave).cuda().float() #28
xdiff = torch.diff(xi) #27
delt_x = make_interpolation_x(xdiff, 28,opt.batch_size, k=1) # 1,256,310,28,4
x_ki = make_interpolation_x(xdiff, 28, opt.batch_size, k=k) # 1,256,256,27,11,4
x_ki_test = make_interpolation_x(xdiff, 28, 10, k=k) # 10,256,256,27,11,4
delt_x_test = make_interpolation_x(xdiff, 28, 10, k=1) #10,256,310,28,4
# if MULTI_GPU:
#     xdiff = xdiff.repeat(2,1)
#     xdiff = xdiff.cuda().float()

def train(epoch, logger):
    epoch_loss = 0
    begin = time.time()
    batch_num = int(np.floor(opt.epoch_sam_num / opt.batch_size))
    for i in range(batch_num):
        gt, gt_curve = shuffle_crop(train_set, train_curve, opt.batch_size) #[b, c, w, h]   [b, c-1, w,h ,4]
        gt = Variable(gt).cuda().float()
        gt_curve = Variable(gt_curve).cuda().float()
        gt_curve = gt_curve.permute(0, 2, 3, 1, 4)
        input_meas = init_meas(gt, mask3d_batch_train, opt.input_setting)
        if opt.two_stages:
            if opt.train_method:
                optimizer.zero_grad()
            else:
                model.eval()
        if opt.multi2hyper.find('Tradition') >= 0:
            pass
        else:
            optimizer_hyper.zero_grad()
        if opt.two_stages:
            if opt.method in ['cst_s', 'cst_m', 'cst_l']:
                z, diff_pred = model(input_meas, input_mask_train)
                # loss = torch.sqrt(mse(model_out, gt))
                # diff_gt = torch.mean(torch.abs(model_out.detach() - gt),dim=1, keepdim=True)  # [b,1,h,w]
                # loss_sparsity = F.mse_loss(diff_gt, diff_pred)
                # loss = loss + 2 * loss_sparsity

            else:
                z = model(input_meas, input_mask_train)
                # loss = torch.sqrt(mse(model_out, gt))
            # if opt.method=='hdnet':
            #     fdl_loss = FDL_loss(model_out, gt)
            #     loss = loss + 0.7 * fdl_loss
            if opt.multi2hyper.find('Tradition') >= 0:
                z, curve_pred = model_hyper(z, xdiff=xdiff, delt_x=delt_x)
                curve_pred = curve_interpolation(curve_pred, k=k)
                value_pred = curve_pred * x_ki
                value_pred = torch.sum(value_pred, -1)
                value_pred = rearrange(value_pred, 'b w h c d-> b (c d) w h ')
            else:
                value_pred = model_hyper(z)
        else:
            if opt.multi2hyper.find('CoSF') >= 0:
                z, curve_pred = model_hyper(input_meas, input_mask_train, xdiff, delt_x)
                # curve_pred = m2abcd(z, curve_pred_m, xdiff)
                curve_pred = curve_interpolation(curve_pred, k=k)
                value_pred = curve_pred * x_ki
                value_pred = torch.sum(value_pred, -1)
                value_pred = rearrange(value_pred, 'b w h c d-> b (c d) w h ')

                loss_L0 = get_loss_L0(curve_pred, x_ki)
                loss_L1 = get_loss_L1(curve_pred, x_ki)
                loss_L2 = get_loss_L2(curve_pred, x_ki)

            elif opt.multi2hyper.find('NeSR') >= 0:
                return 0
        gt_curve = curve_interpolation(gt_curve, k=k)
        loss_pred = get_loss(z, gt)


        gt_curve_value = gt_curve * x_ki
        gt_curve_value = torch.sum(gt_curve_value, -1)
        gt_curve_value = rearrange(gt_curve_value, 'b w h c d-> b (c d) w h ')

        loss_point = get_point_loss(gt_curve_value, value_pred)

        if opt.multi2hyper.find('CoSF') >= 0:
            loss = loss_pred + loss_point + loss_L0 + loss_L1 + loss_L2
            if i % 2000 == 0:
                print(str(loss_pred) + '_' + str(loss_point) + '_' + str(loss_L0) + '_' + str(loss_L1) + '_' + str(
                    loss_L2))
        else:
            loss = loss_pred + loss_point
            if i % 2000 == 0:
                print(str(loss_pred) + '_' + str(loss_point))

        epoch_loss += loss.data
        loss.backward()
        if opt.multi2hyper.find('Tradition') >= 0:
            pass
        else:
            optimizer_hyper.step()
        if opt.two_stages:
            if opt.train_method:
                optimizer.step()

    end = time.time()
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".
                format(epoch, epoch_loss / batch_num, (end - begin)))
    return 0

def test(epoch, logger, test_curve, k):
    psnr_list, ssim_list = [], []
    test_gt = test_data.cuda().float()
    test_curve = curve_interpolation(test_curve, k=k)
    test_curve_value = test_curve * x_ki_test
    test_curve_value = torch.sum(test_curve_value, -1)
    test_curve_value = rearrange(test_curve_value, 'b w h c d-> b (c d) w h ')

    input_meas = init_meas(test_gt, mask3d_batch_test, opt.input_setting)
    if opt.two_stages:
        model.eval()
    model_hyper.eval()
    begin = time.time()
    with torch.no_grad():
        if opt.two_stages:
            if opt.method in ['cst_s', 'cst_m', 'cst_l']:
                z, _ = model(input_meas, input_mask_test)
            else:
                z = model(input_meas, input_mask_test)
            if opt.multi2hyper.find('Tradition') >= 0:
                model_out, curve_pred = model_hyper(z, xdiff=xdiff, delt_x=delt_x_test)
                curve_pred = curve_interpolation(curve_pred, k=k)
                value_pred = curve_pred * x_ki_test
                value_pred = torch.sum(value_pred, -1)
                value_pred = rearrange(value_pred, 'b w h c d-> b (c d) w h ')
            else:
                value_pred = model_hyper(z)

        else:
            if opt.multi2hyper.find('CoSF') >= 0:
                model_out, curve_pred = model_hyper(input_meas, input_mask_test, xdiff, delt_x_test)
                curve_pred = curve_interpolation(curve_pred, k=k)
                value_pred = curve_pred * x_ki_test
                value_pred = torch.sum(value_pred, -1)
                value_pred = rearrange(value_pred, 'b w h c d-> b (c d) w h ')


    end = time.time()
    for k in range(test_curve_value.shape[0]):
        psnr_val = torch_psnr(value_pred[k, :, :, :], test_curve_value[k, :, :, :])
        ssim_val = torch_ssim(value_pred[k, :, :, :], test_curve_value[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
    pred = np.transpose(value_pred.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_curve_value.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    logger.info('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'
                .format(epoch, psnr_mean, ssim_mean,(end - begin)))
    if opt.two_stages:
        model.train()
    model_hyper.train()
    return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean

def main():
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))
    psnr_max = 0
    for epoch in range(1, opt.max_epoch + 1):
        train(epoch, logger)
        (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = test(epoch, logger, test_curve, k)
        if opt.multi2hyper.find('Tradition') >= 0:
            pass
        else:
            scheduler_lines.step()
        if opt.train_method & opt.two_stages:
            scheduler.step()
        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            if psnr_mean > 28:
                name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch, psnr_max, ssim_mean) + '.mat'
                scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all})
                if opt.two_stages:
                    checkpoint(model, epoch, model_path, logger)
                checkpoint2(model_hyper, epoch, model_path, logger)

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()


