import os, yaml
from pathlib import Path
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
import datetime
from denoising_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer
from src.utils import *

def main():

    # # fix all seeds for reproducibility
    # seed = 42
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)

    # select sweep
    # run_name = 'run_1'
    run_name = 'run_5'
    load_model_step = 200000

    # # change this manually if required
    # config['guidance_scale'] = 2.0
    # # config['sampling_timesteps'] = 16
    # # config['use_dynamic_thres'] = False

    # guidance_scales = [1.0]
    # guidance_scales = [5.0]
    guidance_scales = [5.0]
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    ip_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=60*600))
    accelerator = Accelerator(mixed_precision='fp16', gradient_accumulation_steps=1, kwargs_handlers=[ddp_kwargs, ip_kwargs], log_with='wandb')
    # dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=5400))

    # store data in Euler scratch directory
    scratch_dir = '../../../../scratch/jbastek/ml/video_diffusion/'

    run_dir = scratch_dir + 'final_runs/' + run_name + '/'

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
            os.makedirs(run_dir  + '/prediction')
            with open(run_dir + 'model.yaml', 'w') as file:
                yaml.dump(config, file)

    batch_size = config['batch_size']
    selected_channels = config['selected_channels']    
    reference_frame = config['reference_frame']

    model = Unet3D(
        dim = config['unet_dim'],
        dim_mults = (1, 2, 4, 8),
        channels = len(selected_channels),
        attn_heads = config['unet_attn_heads'],
        attn_dim_head = config['unet_attn_dim_head'],
        init_dim = None,
        init_kernel_size = 7,
        use_sparse_linear_attn = config['unet_use_sparse_linear_attn'],
        resnet_groups = config['unet_resnet_groups'],
        cond_bias = True,
        cond_attention = config['unet_cond_attention'], # 'none', 'self-stacked', 'cross-attention', 'self-cross/spatial'
        cond_attention_tokens = config['unet_cond_attention_tokens'],
        cond_att_GRU = config['unet_cond_att_GRU'],
        use_temporal_attention_cond = config['unet_temporal_att_cond'], # should probably be activated
        cond_to_time = config['unet_cond_to_time'],
        # guidance_scale = config['guidance_scale'],
        per_frame_cond = config['per_frame_cond'],
        padding_mode= config['padding_mode'],
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 96,
        channels = len(selected_channels),
        # num_frames = 11,
        num_frames = 1,
        timesteps = config['train_timesteps'],  # number of steps
        loss_type = 'l1',   # L1 or L2
        use_dynamic_thres = config['use_dynamic_thres'],
        sampling_timesteps = config['sampling_timesteps'], # number of steps for sampling, if lower as training DDIM is activated
    )

    data_dir = scratch_dir + 'data/training/' + reference_frame + '/'
    data_dir_validation = scratch_dir + 'data/validation/' + reference_frame + '/'

    trainer = Trainer(
        diffusion,
        # num_frames = 11,
        num_frames = 1,
        folder = data_dir,
        validation_folder = data_dir_validation,
        results_folder = run_dir,
        selected_channels = selected_channels,
        train_batch_size = batch_size,
        train_lr = config['learning_rate'],
        save_and_sample_every = 20000,
        log_freq = 1,
        eval_abq_only_final = False,
        train_num_steps = 200000,         # total training steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # EMA decay
        amp = True,                       # turn on mixed precision
        num_samples = 4,
        preds_per_sample = 4,
        log = True,
        null_cond_prob = 0.1,
        per_frame_cond = config['per_frame_cond'],
        reference_frame = config['reference_frame'],
        run_name=run_name,
        accelerator = accelerator,
    )

    trainer.accelerator.print('Number of parameters: ', count_parameters(diffusion))
    # target_labels_dir = data_dir + '../../random_targets_100.csv'
    target_labels_dir = data_dir + '../../random_targets_100_full.csv'
    # target_labels_dir = data_dir + '../../min_max.csv'
    # target_labels_dir = data_dir + '../../worst_matches_100.csv'
    # target_labels_dir = data_dir + '../../worst_matches_50.csv'

    for guidance_scale in guidance_scales:
        trainer.accelerator.wait_for_everyone()
        trainer.eval(target_labels_dir=target_labels_dir, load_model_step=load_model_step, guidance_scale=guidance_scale, no_cores=10)
    # end training
    trainer.accelerator.end_training()

if __name__ == '__main__':
    main()