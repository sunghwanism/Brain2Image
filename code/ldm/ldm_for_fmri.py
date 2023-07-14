import numpy as np
import wandb
import torch
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch.nn as nn
import os
from ldm.models.diffusion.plms import PLMSSampler
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sc_mbm.fmri_vit_model import fmri_encoder # 태성이랑 혜원이한테 받아야됨

def create_model_from_config(config, num_voxels, global_pool):
    model = fmri_encoder(num_voxels=num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,
                depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, global_pool=global_pool) 
    return model

class cond_stage_model(nn.Module): # 나중에 어떻게 사용되는지 확인하면서 값 맞춰줄 것*
    def __init__(self, metafile, num_voxels, cond_dim=1280, global_pool=True): # cond_dim:  1024 -> 1280? *, # metafile은 아마 Config_MBM_fMRI 이것일 것으로 추정. 태성이한테 pth 파일 받고 확인해야됨 *
        super().__init__()
        # prepare pretrained fmri mae 
        model = create_model_from_config(metafile['config'], num_voxels, global_pool)
        model.load_checkpoint(metafile['model'])
        self.mae = model
        self.fmri_seq_len = model.num_patches # 351
        self.fmri_latent_dim = model.embed_dim # 1024
        if global_pool == False:
            self.channel_mapper = nn.Sequential(
                nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 2, 1, bias=True), # 351 -> 175
                nn.Conv1d(self.fmri_seq_len // 2, 77, 1, bias=True) # 175 -> 77. 77의 의미는? * 
            )
        self.dim_mapper = nn.Linear(self.fmri_latent_dim, cond_dim, bias=True)
        self.global_pool = global_pool # global pool이 어디서 사용되는지 확인할 것 *
    
    def forward(self, x):
        if len(x.shape) == 4:
            x = torch.squeeze(x, dim=1)

        latent_crossattn = self.mae(x)
        if self.global_pool == False:
            latent_crossattn = self.channel_mapper(latent_crossattn)
        latent_crossattn = self.dim_mapper(latent_crossattn) # 1280
        out = latent_crossattn
        return out
    
class fLDM:
    def __init__(self, metafile, num_voxels, device=torch.device('cpu'),
                 pretrain_root='../pretrains/ldm/label2img',
                 logger=None, ddim_steps=250, global_pool=True, use_time_cond=True):
        self.ckp_path = os.path.join(pretrain_root, 'model.ckpt') # ldm 모델 파일 같음 -> label2image pretrained 모델 파일임
        self.config_path = os.path.join(pretrain_root, 'config.yaml')
        config = OmegaConf.load(self.config_path)
        config.model.params.unet_config.params.use_time_cond = use_time_cond # true
        config.model.params.unet_config.params.global_pool = global_pool # false

        self.cond_dim = config.model.params.unet_config.params.context_dim # 512. 이게 머임? *

        model = instantiate_from_config(config.model)
        pl_sd = torch.load(self.ckp_path, map_location = 'cpu')['state_dict']

        m, u = model.load_state_dict(pl_sd, strict=False) # shape 안 맞아서 일단 잠깐 지웠음
        model.cond_stage_trainable = True
        model.cond_stage_model = cond_stage_model(metafile, num_voxels, self.cond_dim, global_pool=global_pool)
        model.ddim_steps = ddim_steps
        model.re_init_ema() # 이게 뭔지 확인 *
        if logger is not None:
            logger.watch(model, log='all', log_graph=False)

        model.p_channels = config.model.params.channels # 3
        model.p_image_size = config.model.params.image_size # 64
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult # [1,2,4]

        self.device = device    
        self.model = model
        self.ldm_config = config
        self.pretrain_root = pretrain_root
        self.fmri_latent_dim = model.cond_stage_model.fmri_latent_dim
        self.metafile = metafile

    def finetune(self, trainers, dataset, test_dataset, bs1, lr1,
                 output_path, config=None):
        # bs1: batchsize
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        self.model.run_full_validation_threshold = 0.15
        # stage one: train the cond encoder with the pretrained one
      
        # # stage one: only optimize conditional encoders
        print('\n##### Stage One: only optimize conditional encoders #####')
        dataloader = DataLoader(dataset, batch_size = bs1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        self.model.unfreeze_whole_model()
        #self.model.freeze_first_stage() # ldm freeze. train only cond stage model(encoder)

        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = True
        self.model.eval_avg = config.eval_avg

            
        wandb.unwatch() # TorchHistory 관련된 오류 때문에 붙인 거임 *

        print("trainers.fit!!!!!!!!!!!!!!!!!!!")
        trainers.fit(self.model, dataloader, val_dataloaders=test_loader)

        self.model.unfreeze_whole_model()
        print("torch.save model_state at checkpoint!!!!!!!!!!!!!!!!!!!!!")
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'config': config,
                'state': torch.random.get_rng_state()

            },
            os.path.join(output_path, 'checkpoint.pth')
        )

    @torch.no_grad()
    def generate(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None): # limit이 뭐임? *
        # fmri_embedding: n, seq_len, embed_dim 371, 1024 (53*7 -> 7개 roi 합쳐서 371)
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels, 
            self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size) # (3, 64, 64) -> vqvae의 latent space? *
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model) # sampler 확인할 것! *

        if state is not None:
            torch.cuda.set_rng_state(state)

        with model.ema_scope():
            model.eval()
            for count, item in enumerate(fmri_embedding):
                # print("generating in generate method!!!!!!!!!!!!!!!!")
                if limit is not None:
                    if count >= limit:
                        break
                latent = item['fMRI']
                gt_image = rearrange(item['Image'], 'h w c -> 1 c h w')
                gt_image = torch.from_numpy(gt_image)
                print(f"rendering {num_samples} examples in {ddim_steps} steps.")
                # c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                c = model.get_learned_conditioning(torch.from_numpy(repeat(latent, 'h w -> c h w', c=num_samples)).type('torch.FloatTensor').to(self.device))
                
                samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                conditioning=c,
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False)
                
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)
                print("gt_image shape:", gt_image.shape, "!!!!!!!!!!!!!!!!!!!!!!!!!")
                
                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first [2, 3, 256, 256]
        
        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to('cpu')
        
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)
    