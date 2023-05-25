import os, yaml
import datetime
from pathlib import Path
from torch import distributed as dist
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from denoising_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer
from src.utils import *

def main():

    ### User input ###
    # define run name, if run name already exists and is not 'pretrained', load_model_step must be provided
    run_name = 'pretrained'
    
    # number of predictions to generate for each conditioning
    num_preds = 1

    # change to your '<your_wandb_username>' if you want to log to wandb
    wandb_username = None
    ###

    # initialize distributed training
    dist.init_process_group(backend='gloo')
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    ip_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=60*10))  # required to prevent timeout error when nonequal distribution of samplings to GPUs
    accelerator = Accelerator(mixed_precision='fp16', kwargs_handlers=[ddp_kwargs, ip_kwargs], log_with='wandb' if wandb_username is not None else None)

    # store data in given directory
    cur_dir = './'  # local path to repo
    run_dir = cur_dir + '/runs/' + run_name + '/'

    # path to target stress-strain curves
    target_labels_dir = cur_dir + 'data/target_responses.csv'

    # check if directory exists
    if os.path.exists(run_dir):
        if run_name == 'pretrained':
            load_model_step = 200000  # pretrained model was trained for 200k steps
        if load_model_step is None:
            accelerator.print('Directory already exists, please change run_name to train new model or provide load_model_step')
            return
        # extract model parameters from given yaml
        config = yaml.safe_load(Path(run_dir + 'model/model.yaml').read_text())
    else:
        # extract model parameters from created yaml
        config = yaml.safe_load(Path('model/model.yaml').read_text())
        accelerator.wait_for_everyone()
        # save model parameters in run_dir    
        if accelerator.is_main_process:
            # create folder for all saved files
            os.makedirs(run_dir + '/training')
            with open(run_dir + 'model/model.yaml', 'w') as file:
                yaml.dump(config, file)

    model = Unet3D(
        dim = config['unet_dim'],
        dim_mults = (1, 2, 4, 8),
        channels = len(config['selected_channels']),
        attn_heads = config['unet_attn_heads'],
        attn_dim_head = config['unet_attn_dim_head'],
        init_dim = None,
        init_kernel_size = 7,
        use_sparse_linear_attn = config['unet_use_sparse_linear_attn'],
        resnet_groups = config['unet_resnet_groups'],
        cond_bias = True,
        cond_attention = config['unet_cond_attention'],
        cond_attention_tokens = config['unet_cond_attention_tokens'],
        cond_att_GRU = config['unet_cond_att_GRU'],
        use_temporal_attention_cond = config['unet_temporal_att_cond'],
        cond_to_time = config['unet_cond_to_time'],
        per_frame_cond = config['per_frame_cond'],
        padding_mode = config['padding_mode'],
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 96,
        channels = len(config['selected_channels']),
        num_frames = 11,
        timesteps = config['train_timesteps'],
        loss_type = 'l1',
        use_dynamic_thres = config['use_dynamic_thres'],
        sampling_timesteps = config['sampling_timesteps'],
    )

    data_dir = cur_dir + 'data/' + config['reference_frame'] + '/training/'
    data_dir_validation = cur_dir + 'data/' + config['reference_frame'] + '/validation/'

    trainer = Trainer(
        diffusion,
        folder = data_dir,
        validation_folder = data_dir_validation,
        results_folder = run_dir,
        selected_channels = config['selected_channels'],
        train_batch_size = config['batch_size'],
        test_batch_size = config['batch_size'],
        train_lr = config['learning_rate'],
        save_and_sample_every = 10000,
        train_num_steps = 200000,
        ema_decay = 0.995,
        log = True,
        null_cond_prob = 0.1,
        per_frame_cond = config['per_frame_cond'],
        reference_frame = config['reference_frame'],
        run_name = run_name,
        accelerator = accelerator,
        wandb_username = wandb_username
    )

    trainer.train(load_model_step=load_model_step, num_samples=3, num_preds=num_preds)
    trainer.eval_target(target_labels_dir, guidance_scale=5., num_preds=num_preds)

if __name__ == '__main__':
    main()