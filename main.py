import os, yaml
from pathlib import Path
from accelerate import Accelerator, DistributedDataParallelKwargs
from denoising_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer
from src.utils import *

def main():


    # provide wandb username if you want to log to wandb
    wandb_username = 'jbastek' # None

    # select pretrained run
    run_name = 'run_1'
    load_model_step = None


    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision='fp16', kwargs_handlers=[ddp_kwargs], log_with='wandb' if wandb_username is not None else None)

    # store data in given directory
    scratch_dir = './'
    run_dir = scratch_dir + 'runs/' + run_name + '/'

    # check if directory exists
    if os.path.exists(run_dir):
        if load_model_step is None:
            accelerator.print('Directory already exists, please change run_name to train new model or provide load_model_step')
            return
        # extract model parameters from given yaml
        config = yaml.safe_load(Path(run_dir + 'model.yaml').read_text())
    else:
        # extract model parameters from created yaml
        config = yaml.safe_load(Path('model.yaml').read_text())
        accelerator.wait_for_everyone()
        # save model parameters in run_dir    
        if accelerator.is_main_process:
            # create folder for all saved files
            os.makedirs(run_dir + '/training')
            with open(run_dir + 'model.yaml', 'w') as file:
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
        padding_mode= config['padding_mode'],
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 96,
        channels = len(config['selected_channels']),
        num_frames = 11,
        timesteps = config['train_timesteps'],
        loss_type = 'l1',  # l1 or l2
        use_dynamic_thres = config['use_dynamic_thres'],
        sampling_timesteps = config['sampling_timesteps'],
    )

    data_dir = scratch_dir + 'data/' + config['reference_frame'] + '/training/'
    data_dir_validation = scratch_dir + 'data/' + config['reference_frame'] + '/validation/'

    trainer = Trainer(
        diffusion,
        folder = data_dir,
        validation_folder = data_dir_validation,
        results_folder = run_dir,
        selected_channels = config['selected_channels'],
        train_batch_size = config['batch_size'],
        test_batch_size = config['batch_size'],
        train_lr = config['learning_rate'],
        save_and_sample_every = 20,
        train_num_steps = 40,
        ema_decay = 0.995,
        preds_per_sample = 4,
        log = True,
        null_cond_prob = 0.1,
        per_frame_cond = config['per_frame_cond'],
        reference_frame = config['reference_frame'],
        run_name=run_name,
        accelerator = accelerator,
        wandb_username = wandb_username,
    )
    trainer.accelerator.print('Number of parameters: ', count_parameters(diffusion))
    trainer.train(load_model_step=load_model_step)

if __name__ == '__main__':
    main()