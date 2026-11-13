from functools import partial
import time
from copy import deepcopy
import einops
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
import random
import numpy as np
import torch
import torchvision as tv
from scipy.io import savemat
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset.pan_dataset import PanDataset
from diffusion.diffusion_ddpm_pan import make_beta_schedule
from utils.logger import TensorboardLogger
from utils.lr_scheduler import get_lr_from_optimizer, StepsAll
from utils.metric import AnalysisPanAcc, NonAnalysisPanAcc
from utils.misc import compute_iters, exist, grad_clip, model_load, path_legal_checker
from utils.optim_utils import EmaUpdater

import argparse


def show_distribution(hr):
    plt.figure()
    gt = hr[:2].detach().cpu().flatten().numpy()
    sns.displot(data=gt)
    plt.show()


def norm(x):
    # x range is [0, 1]
    return x * 2 - 1


def unorm(x):
    # x range is [-1, 1]
    return (x + 1) / 2


def clamp_fn(g_sr):
    def inner(x):
        x = x + g_sr
        x = x.clamp(0, 1.0)
        return x - g_sr

    return inner


def engine_google(
    # dataset
     train_dataset_path,
    valid_dataset_path,
    dataset_name=None,
    # image settings
    image_n_channel=31,
    image_size=64,
    # diffusion settings
    schedule_type="cosine",
    n_steps=3_000,
    max_iterations=400_000,
    # device setting
    device="cuda:0",
    # optimizer settings
    batch_size=128,
    lr_d=1e-4,
    show_recon=False,
    # pretrain settings
    pretrain_weight_s1=None,
    pretrain_weight_s2=None,
    pretrain_iterations_s1=None,
    pretrain_iterations_s2=None,
    # others
    constrain_channel=None,
    recon_display_interval=None,
    valid_interval=None,
    train_mode="train",
    pred_mode="x_start",
    delta_end=1.8e-3,
    flexible_val = None,
    fuser = None,
    n_feats = None,
    embed_dim=None,
    latent_encoder = None,
    latent_encoder_2 = None,
    stage = 1,
    denoiser_scale=5,
    fus_en_scale=3,
    lr_opt=None,
    wd_opt=None,
):
    
    from diffusion.ddpm_low_rank import GaussianDiffusion as LowRankDiffusion
    from models.denoiser import mid_denoiser
    from models.fus_encoder import (PLFE_Stage1, PLFE_Stage2)
    from models.FusUneXt_scale import FusionNetConvNeXtScale
    from build_model import LeFuser, LDMFuser

    # init logger
    stf_time = time.strftime("%m-%d_%H-%M", time.localtime())
    logger = TensorboardLogger(file_logger_name="{}-{}-stage{}".format(stf_time, dataset_name, stage))

    config = {
            # Dataset parameters
            'train_dataset_path': train_dataset_path,
            'valid_dataset_path': valid_dataset_path,
            'dataset_name': dataset_name,
            
            # Image settings
            'image_n_channel': image_n_channel,
            'image_size': image_size,
            
            # Diffusion settings
            'schedule_type': schedule_type,
            'n_steps': n_steps,
            'max_iterations': max_iterations,
            
            # Device setting
            'device': device,
            
            # Optimizer settings
            'batch_size': batch_size,
            'lr_d': lr_d,
            'show_recon': show_recon,
            
            # Pretrain settings
            'pretrain_weight_s1': pretrain_weight_s1,
            'pretrain_iterations_s1': pretrain_iterations_s1,
            'pretrain_weight_s2': pretrain_weight_s2,
            'pretrain_iterations_s2': pretrain_iterations_s2,
            
            # Debugging parameters
            'constrain_channel': constrain_channel,
            'recon_display_interval': recon_display_interval,
            'valid_interval': valid_interval,
            'train_mode': train_mode,
            'pred_mode': pred_mode,
            'delta_end': delta_end,
            'flexible_val': flexible_val,
            'fuser': fuser,
            'n_feats': n_feats,
            'embed_dim': embed_dim,
            'latent_encoder': latent_encoder,
            'latent_encoder_2': latent_encoder_2,
            'stage': stage,
            'denoiser_scale': denoiser_scale, 
            'fus_en_scale' : fus_en_scale, 
            'lr_opt' : lr_opt,
            'wd_opt' : wd_opt,
        }
    logger.print(config)

    dataset_name = (
        train_dataset_path.strip(".h5").split("_")[-1]
        if not exist(dataset_name)
        else dataset_name
    )
    logger.print(f"dataset name: {dataset_name}")
    division_dict = {"wv3": 2047.0, "gf2": 1023.0, "qb": 2047.0}
    logger.print(f"dataset norm division: {division_dict[dataset_name]}")
    rgb_channel = {
        "wv3": [4, 2, 0],
        "gf2": [0, 1, 2],
        "qb": [0, 1, 2],
    }
    logger.print(f"rgb channel: {rgb_channel[dataset_name]}")

    # initialize models
    
    #torch.cuda.set_device(device)
    
    
        
    if dataset_name in ['wv3', 'gf2', 'qb']:
        image_n_channel = 8 if dataset_name == 'wv3' else 4
        pan_channel = 1
    
   
    if stage == 1 or stage == 2:
        # set latent encoder
        
        le1_in_chans = image_n_channel*2+pan_channel
        
        latent_encoder == 'fus_encoder_s1'
        LE_model = PLFE_Stage1(pan_c=pan_channel, lms_c=image_n_channel, gt_c=image_n_channel, 
                                        base_dim=embed_dim, group=4, num_stages=fus_en_scale).to(device)


        # set fuser
        if fuser == "FusUneXt_scale":
            fuse_model = FusionNetConvNeXtScale(image_n_channel, base_dim=64, depths=[1, 2, 2, 4], 
                                            layer_scale_init=1e-6, embed_dim=embed_dim).to(device)
        
        
        # integrate latent encoder and fuser
        le_fuser = LeFuser(LE_model=LE_model, fuse_model=fuse_model).to(device)  

        if pretrain_weight_s1 is not None:
            if isinstance(pretrain_weight_s1, (list, tuple)):
                model_load(pretrain_weight_s1[0], le_fuser, strict=True, device=device)
            else:
                model_load(pretrain_weight_s1, le_fuser, strict=False, device=device)
            print("load pretrain weight (stage 1) from {}".format(pretrain_weight_s1)) 

    if stage == 2:
        # freeze pretrained latent encoder 

        fuse_model = le_fuser.fuse_model.to(device)
        LE_model_1 = le_fuser.LE_model.to(device)
        for param in LE_model_1.parameters():
            param.requires_grad = False
        
        # latent encoder for stage 2
        
        le2_in_chans = image_n_channel + pan_channel
        LE_model_2 = PLFE_Stage2(pan_c=pan_channel, lms_c=image_n_channel, 
                                        base_dim=embed_dim, group=4, num_stages=fus_en_scale).to(device)
        

         # set denoiser
        denoise_fn = mid_denoiser( in_channel=embed_dim*4, 
                                    out_channel=embed_dim*4, 
                                    inner_channel=embed_dim*4*2,
                                    block_num=denoiser_scale,
                                    n_steps=n_steps,
                                    group=4, 
                                    patch_expansion=0.5, 
                                    channel_expansion=2).to(device)


        # integrate all modules to be trained
        ldm_fuser = LDMFuser(denoise_fn=denoise_fn, LE_model_2=LE_model_2, fuse_model=fuse_model)

        if pretrain_weight_s2 is not None:
            if isinstance(pretrain_weight_s2, (list, tuple)):
                model_load(pretrain_weight_s2[0], ldm_fuser, strict=True, device=device)
            else:
                model_load(pretrain_weight_s2, ldm_fuser, strict=False, device=device)
            print("load pretrain weight (stage 2) from {}".format(pretrain_weight_s2)) 
        


    # get dataset
    d_train = h5py.File(train_dataset_path)
    d_valid = h5py.File(valid_dataset_path)
    if dataset_name in ["wv3", "gf2", "qb"]:
        DatasetUsed = partial(
            PanDataset,
            full_res=False,
            norm_range=False,
            constrain_channel=constrain_channel,
            division=division_dict[dataset_name],
            aug_prob=0,
            wavelets=True,
        )
    else:
        raise NotImplementedError("dataset {} not supported".format(dataset_name))

    ds_train = DatasetUsed(
        d_train,
    )
    ds_valid = DatasetUsed(
        d_valid,
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )

    # diffusion
    if stage == 2:
        
        diffusion = LowRankDiffusion(
            ldm_fuser.denoise_fn,
            channels=n_feats,
            # num_sample_steps=n_steps,
            pred_mode=pred_mode,
            loss_type="l1",
            device=device,
            clamp_range=(0, 1),
            )

        diffusion.set_new_noise_schedule(
            betas=make_beta_schedule(schedule=schedule_type, n_timestep=n_steps, cosine_s=8e-3), trunc=False
        )
        diffusion = diffusion.to(device)
    
        diffusion_dp = (
        diffusion
        )
    
        ema_updater = EmaUpdater(
            ldm_fuser, deepcopy(ldm_fuser), decay=0.995, start_iter=20_000, diff_flag=False,
        )
       
        if lr_opt is not None:
            optimizer_params = [
                                {"params": ldm_fuser.denoise_fn.parameters(), "lr": lr_opt['denoise_fn'], "weight_decay": wd_opt['denoise_fn']},
                                {"params": ldm_fuser.LE_model_2.parameters(), "lr": lr_opt['LE_model_2'], "weight_decay": wd_opt['LE_model_2']},
                                {"params": ldm_fuser.fuse_model.parameters(), "lr": lr_opt['fuse_model'], "weight_decay": wd_opt['fuse_model']},
                            ]
            opt_d = torch.optim.AdamW(optimizer_params)
        else:
            opt_d = torch.optim.AdamW(ldm_fuser.parameters(), lr=lr_d, weight_decay=1e-4)
        scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
        opt_d, milestones=[100_000, 200_000, 350_000], gamma=0.2
        )
        schedulers = StepsAll(scheduler_d)

    if stage == 1:
        ema_updater = EmaUpdater(
            le_fuser, deepcopy(le_fuser), decay=0.995, start_iter=20_000, diff_flag=False,
        )
        opt_d = torch.optim.AdamW(le_fuser.parameters(), lr=lr_d, weight_decay=1e-4)
        scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
        opt_d, milestones=[100_000, 200_000, 350_000], gamma=0.2
        )
        schedulers = StepsAll(scheduler_d)
            

    # training
    if pretrain_iterations_s1 is not None and stage == 1:
        iterations = pretrain_iterations_s1
        logger.print("load previous training (stage 1) with {} iterations".format(iterations))
    elif pretrain_iterations_s1 is not None and stage == 2:
        if pretrain_iterations_s2 is None:
            iterations = 0
            logger.print("load previous training (stage 1) with {} iterations for STAGE-2 training from Scratch".format(pretrain_iterations_s1))
        else :
            iterations = pretrain_iterations_s2
            logger.print("load LE_Model_1 and Trained LDMfuse previous training (stage 2) with {} iterations".format(iterations))
    else:
        iterations = 0
    while iterations <= max_iterations:
        for i, (pan, lms, hr, wavelets) in enumerate(dl_train, 1): # here, lms represents hsi_up
            pan, lms, hr, wavelets = map(lambda x: x.to(device), 
                                         (pan.float(), lms.float(), hr.float(), wavelets.float()))

            opt_d.zero_grad()
            

            if stage == 1:
                output, prior = le_fuser(lms=lms, gt=hr, pan=pan)
                recon_x = output
                loss = F.l1_loss(output, hr)

            if stage == 2:
                cond = ldm_fuser.LE_model_2(lms=lms, pan=pan) # stage2 prior
                prior_0 = LE_model_1(lms=lms, gt=hr, pan=pan) # stage1 prior

                diff_loss, recon_prior = diffusion_dp(prior_0, mode=train_mode, cond=cond) 
                output = ldm_fuser.fuse_model(lms, pan=pan, prior=recon_prior)

                loss = diff_loss + F.l1_loss(output, hr)

            loss.backward()
            
            if stage == 1:
                grad_clip(le_fuser.parameters(), mode="norm", value=0.003) 
            elif stage == 2:
                grad_clip(ldm_fuser.parameters(), mode="norm", value=0.003) 

           
            opt_d.step()
            ema_updater.update(iterations)
            schedulers.step()

            iterations += 1
            logger.print(
                f"[iter {iterations}/{max_iterations}: "
                + f"d_lr {get_lr_from_optimizer(opt_d): .6f}] - "
                + f"loss {loss:.6f} " #
                #+ f"diff loss {diff_loss:.6f}"
            )
            
            # test predicted sr
            if show_recon and iterations % recon_display_interval == 0:
                # NOTE: only used to validate code
                recon_x = recon_x[:64]
                

                x = tv.utils.make_grid(recon_x, nrow=8, padding=0).cpu()
                x = x.clip(0, 1)  # for no warning
                fig, ax = plt.subplots(figsize=(x.shape[-1] // 100, x.shape[-2] // 100))
                x_show = (
                    x.permute(1, 2, 0).detach().numpy()[..., rgb_channel[dataset_name]]
                )
                ax.imshow(x_show)
                ax.set_axis_off()
                plt.tight_layout(pad=0)
                # plt.show()
                fig.savefig(
                    f"./samples/recon_x/{dataset_name}_iter_{iterations}.png",
                    dpi=200,
                    bbox_inches="tight",
                    pad_inches=0,
                )

            # do some sampling to check quality
            if iterations in flexible_val:
                # diffusion_dp.model.eval()
                # ema_updater.ema_model.model.eval()
                if stage == 1:
                    le_fuser.eval()
                    ema_updater.ema_model.eval()
                elif stage == 2:
                    ldm_fuser.eval()
                    ema_updater.ema_model.eval()

                analysis_d = AnalysisPanAcc()
                with torch.no_grad():
                    for i, (pan, lms, hr, wavelets) in enumerate(dl_valid, 1):
                        torch.cuda.empty_cache()
                        pan, lms, hr, wavelets = map(
                            lambda x: x.to(device), 
                            (pan.float(), lms.float(), hr.float(), wavelets.float()) #x.cuda()
                        )

                        if stage == 1:
                            sr, _ = ema_updater.ema_model(lms=lms, gt=hr, pan=pan)
                        elif stage == 2:
                            diffusion_eval = LowRankDiffusion(
                                    ema_updater.ema_model.denoise_fn,
                                    channels=n_feats,
                                    # num_sample_steps=n_steps,
                                    pred_mode=pred_mode,
                                    loss_type="l1",
                                    device=device,
                                    clamp_range=(0, 1),
                                    )
                            diffusion_eval.set_new_noise_schedule(
                                betas=make_beta_schedule(schedule=schedule_type, n_timestep=n_steps, cosine_s=8e-3), trunc=False
                            )
                            diffusion_eval = diffusion_eval.to(device)

                            cond = ema_updater.ema_model.LE_model_2(lms=lms, pan=pan).to(device)

                            recon_prior = diffusion_eval(cond, mode="ddim_sample", section_counts="ddim25")
                            sr = ema_updater.ema_model.fuse_model(lms, pan=pan, prior=recon_prior)

                

                        sr = sr.clip(0, 1)
                        hr = hr.to(sr.device)
                        
                        analysis_d(hr, sr)
                        
                        hr = tv.utils.make_grid(hr, nrow=4, padding=0).cpu()
                        x = tv.utils.make_grid(sr, nrow=4, padding=0).detach().cpu()
                        x = x.clip(0, 1)

                        s = torch.cat([hr, x], dim=-1)  # [b, c, h, 2*w]

                        logger.print("---diffusion result---")
                        logger.print(analysis_d.last_acc)
                    if i != 1:
                        logger.print("---diffusion result---")
                        logger.print(analysis_d.print_str())
                
                if stage == 1:
                    le_fuser.train()
                elif stage == 2:
                    ldm_fuser.train()
                setattr(ema_updater.model, "image_size", 64)

                torch.save(
                    ema_updater.ema_model_state_dict,
                    f"./weights/ema_{dataset_name}_iter_{iterations}_stage{stage}.pth",
                )
                logger.print("save model")

                logger.log_scalars("diffusion_perf", analysis_d.acc_ave, iterations)
                logger.print("saved performances")

            # log loss
            if iterations % 50 == 0:
                logger.log_scalar("denoised_loss", loss.item(), iterations)


@torch.no_grad()
def test_fn(
    test_data_path,
    weight_path,
    schedule_type="cosine",
    batch_size=320,
    n_steps=1500,
    show=False,
    device="cuda:0",
    full_res=False,
    dataset_name="gf2",
    division=1023,
    pred_mode = 'x_start',
    fuser = None,
    n_feats = 8, 
    embed_dim = 8, 
    latent_encoder = None,
    latent_encoder_2 = None,
    stage = 2,
    denoiser_scale = 5,
    fus_en_scale=3,
):

    from diffusion.ddpm_low_rank import GaussianDiffusion as LowRankDiffusion
    from models.denoiser import mid_denoiser
    from models.fus_encoder import PLFE_Stage1, PLFE_Stage2
    from models.FusUneXt_scale import FusionNetConvNeXtScale
    

    from build_model import LeFuser, LDMFuser
    division_dict = {"wv3": 2047.0, "gf2": 1023.0, "qb": 2047.0}
    division = division_dict[dataset_name]
    #torch.cuda.set_device(device)

    # load model
    if dataset_name in ['wv3', 'gf2', 'qb']:
        image_size = 512 if full_res else 256
        image_n_channel = 8 if dataset_name == 'wv3' else 4
        pan_channel = 1
        rgb_channels = [4, 2, 0] if dataset_name == 'wv3' else [2, 1, 0]

    LE_model = PLFE_Stage1(pan_c=pan_channel, lms_c=image_n_channel, gt_c=image_n_channel, 
                                                base_dim=embed_dim, group=4, num_stages=fus_en_scale).to(device)
    
    if fuser == "FusUneXt_scale":
        fuse_model = FusionNetConvNeXtScale(image_n_channel, base_dim=64, depths=[1, 2, 2, 4], 
                                        layer_scale_init=1e-6, embed_dim=embed_dim).to(device)
       
    # integrate latent encoder and fuser
    le_fuser = LeFuser(LE_model=LE_model, fuse_model=fuse_model).to(device)  

    fuse_model = le_fuser.fuse_model.to(device)
    LE_model_1 = le_fuser.LE_model.to(device)

    # latent encoder
    le2_in_chans = image_n_channel + pan_channel
    LE_model_2 = PLFE_Stage2(pan_c=pan_channel, lms_c=image_n_channel,
                                        base_dim=embed_dim, group=4, num_stages=fus_en_scale).to(device)

     # set denoiser
    
    denoise_fn = mid_denoiser( in_channel=embed_dim*4, 
                                out_channel=embed_dim*4, 
                                inner_channel=embed_dim*4*2,
                                block_num=denoiser_scale,
                                n_steps=n_steps,
                                group=4, 
                                patch_expansion=0.5, 
                                channel_expansion=2)

    # integrate all modules to be trained
    ldm_fuser = LDMFuser(denoise_fn=denoise_fn, LE_model_2=LE_model_2, fuse_model=fuse_model)

    ldm_fuser = model_load(weight_path, ldm_fuser, device=device)

    ldm_fuser.eval()
    print(f"load weight {weight_path}")
   
        
    diffusion = LowRankDiffusion(
        ldm_fuser.denoise_fn,
        channels=n_feats,
        # num_sample_steps=n_steps,
        pred_mode=pred_mode,
        loss_type="l1",
        device=device,
        clamp_range=(0, 1),
        )

    diffusion.set_new_noise_schedule(
        betas=make_beta_schedule(schedule=schedule_type, n_timestep=n_steps, cosine_s=8e-3), trunc=False
    )
    diffusion = diffusion.to(device)
        
        
            

    # load dataset
    d_test = h5py.File(test_data_path)
    
    if dataset_name in ["wv3", "gf2", "qb"]:
        ds_test = PanDataset(
            d_test, full_res=full_res, norm_range=False, division=division, wavelets=True
        )
        
    dl_test = DataLoader(
            ds_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0
        )

    saved_name = "reduced" if not full_res else "full"

    # do sampling
    preds = []
    sample_times = len(dl_test)
    analysis = AnalysisPanAcc() if not full_res else NonAnalysisPanAcc()
    total_time = 0
    for i, batch in enumerate(dl_test):
        if full_res:
            pan, lms, wavelets = batch
            gt = None
        else:
            pan, lms, gt, wavelets = batch
        print(f"sampling [{i}/{sample_times}]")
        pan, lms, wavelets = map(lambda x: x.to(device).clip(0, 1), (pan, lms, wavelets))

        cond = ldm_fuser.LE_model_2(lms, pan=pan, cond=None).to(device)
        recon_prior = diffusion(cond, mode="ddim_sample", section_counts="ddim25")
        
       
        sr = ldm_fuser.fuse_model(lms, pan=pan, prior=recon_prior)

        sr = sr.clip(0, 1)
        
        if not full_res:
            analysis(sr.detach().cpu(), gt.detach().cpu())
        else:
            analysis(sr.detach())

        print(analysis.print_str(analysis.last_acc))

        if show:
            s = tv.utils.make_grid(sr[:64], nrow=4, padding=0).cpu()
            s.clip_(0, 1)
            channel_select = rgb_channels
            if not full_res:
                fig, axes = plt.subplots(ncols=4, figsize=(20, 5))
                axes[0].imshow(lms[0].cpu().permute(1, 2, 0).numpy()[..., channel_select])
                axes[1].imshow(sr[0].cpu().permute(1, 2, 0).numpy()[..., channel_select])
                axes[0].imshow(gt[0].cpu().permute(1, 2, 0).numpy()[..., channel_select])
                axes[1].imshow(recon_prior[0].cpu(), cmap='viridis', aspect='auto')
                
            else:
                fig, axes = plt.subplots(ncols=3, figsize=(20, 5))
                axes[0].imshow(lms[0].cpu().permute(1, 2, 0).numpy()[..., channel_select])
                axes[1].imshow(sr[0].cpu().permute(1, 2, 0).numpy()[..., channel_select])
                axes[2].imshow(recon_prior[0].cpu(), cmap='viridis', aspect='auto')

            plt.tight_layout(pad=0)
            #plt.show()
            fig.savefig(
                path_legal_checker(
                f"./samples/pandiff_{dataset_name}_test/test_sample_iter_{n_steps}_{saved_name}_part_{i}.png"
                ), 
                dpi=200,
                bbox_inches="tight",
                pad_inches=0,)

        sr = sr.detach().cpu().numpy()  # [b, c, h, w]
        sr = sr * division  # [0, 2047]
        preds.append(sr.clip(0, division))

        print(f"over all test acc:\n {analysis.print_str()}")
        if i > 0:
            print(f"average inference time: {total_time/(i)}s")

    # save results
    with h5py.File(test_data_path, 'r') as hf:
        hf.visit(print)
    if not full_res:
        if dataset_name == 'cave':
            d = dict(  # [b, c, h, w], wv3 [0, 2047]
                gt=d_test["GT"][:],
                hsi_up=d_test["HSI_up"][:],
                lr_hsi=d_test["LRHSI"][:],
                rgb=d_test["RGB"][:],
                sr=np.concatenate(preds, axis=0),
            )
        else:
            d = dict(  # [b, c, h, w], wv3 [0, 2047]
                gt=d_test["gt"][:],
                ms=d_test["ms"][:],
                lms=d_test["lms"][:],
                pan=d_test["pan"][:],
                sr=np.concatenate(preds, axis=0),
            )
    else:
        d = dict(
            ms=d_test["ms"][:],
            lms=d_test["lms"][:],
            pan=d_test["pan"][:],
            sr=np.concatenate(preds, axis=0),
        )
    # model_iterations = weight_path.split("_")[-1].strip(".pth")
    savemat(
        path_legal_checker(f"./samples/mat/test_{fuser}_{saved_name}_{dataset_name}.mat"), 
        d
    )
    print("save result")

def set_seed(SEED):
    # initialize random seed
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="KSDiff Train or Test")

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="train or test"
    )

    args = parser.parse_args()

    set_seed(42)

    if args.mode == "train":

        engine_google(
            train_dataset_path="/workspace/data/train_wv3.h5",
            valid_dataset_path="/workspace/data/valid_wv3.h5",
            dataset_name="wv3",
            pretrain_weight_s1=None,

            max_iterations=300010,
            show_recon=False,
            n_steps=25,
            schedule_type="cosine",
            device="cuda",
            recon_display_interval=50000,
            valid_interval=100000,

            # Training configs
            lr_d=1e-4,
            flexible_val=[10, 50, 100, 1000, 5000, 10000,
                          20000, 40000, 80000, 100000],

            # Model configs
            batch_size=64,
            denoiser_scale=5,
            pred_mode="x_start",
            fuser="FusUneXt_scale",
            n_feats=64,
            embed_dim=64,
            latent_encoder="fus_encoder_s1",
            latent_encoder_2="fus_encoder_s2",
            fus_en_scale=3,
            stage=2,
        )

    elif args.mode == "test":

        test_fn(
            test_data_path="/workspace/data/test_wv3_multiExm1.h5",
            weight_path="",
            dataset_name="wv3",

            batch_size=1,
            n_steps=25,
            show=False,
            full_res=False,
            device="cuda:0",
            pred_mode="x_start",

            denoiser_scale=5,
            fuser="FusUneXt_scale",
            n_feats=64,
            embed_dim=64,
            latent_encoder="fus_encoder_s1",
            latent_encoder_2="fus_encoder_s2",
            fus_en_scale=3,
            stage=2,
        )

  




