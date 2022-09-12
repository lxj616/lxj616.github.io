---
layout: post
title:  "Finetune stable diffusion under 24gb vram in hours"
date:   2022-09-12 13:15:31 +0800
categories: jekyll update
---

# Finetune Stable Diffusion Under 24GB VRAM In Hours

Compared to textual inversion stable diffusion (which needs 10GB+), resume training the original model itself needs more resources, but I have managed to do it using one single RTX 3090Ti, in hours

![tomandjerry_finetune.jpg](/assets/tomandjerry_finetune.jpg)

## Why not textual inversion

**Assume stable diffusion has capabilities of generating all distributions, then textual inversion is the same with resume training**

[An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://arxiv.org/abs/2208.01618)

And due to its strong capabilities, everything you wish to finetune on could be expressed as one embedding, for further explanations, please see the original paper

For most of the case, it works perfectly

But stable diffusion model does not work on outlier distributions it has never seen, for example, my mom

No matter how close the embedding leads to the original image, it is not my mom, a lady of the same age and similar head shape and expression is not good enough

**The embedding to distribution loss is too high on this case, can not be ignored, similar cases include highly detailed anime hands and arms which stable diffusion have difficulties in the first place**

On this case, we are going further to get my mom being recognized by stable diffusion, not as a embedding, but a new distribution

However, let's respect my mom's privacy and use Tom and Jerry screenshots as a example instead

## Pre-encode the CLIP and f8 embedding to free more vrams

The original training/inference config encode text/image pair on the fly, which loads CLIP model into vram, we can not afford it

And if you are really tight in vram, you can remove the first stage model as well, but totally not recommended, because logging images regularly is important for spoting bugs early

pre-encode f8:

```
posterior = first_stage_model.encode(img_tensor)
```

pre-encode CLIP:
```
txt_embed = cond_stage_model.encode(text)
```

And a example config with the pre-encodings instead of CLIP model
```
model:
  base_learning_rate: 6.666e-08
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.00085
    cond_stage_key: t5 #actually CLIP for stable-diffusion, pre-encoded, lazy not changing this
    linear_end: 0.012
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: f8
    conditioning_key: crossattn
    image_size: 64
    channels: 4
    monitor: val/loss_simple_ema
    scale_factor: 0.18215
    use_ema: False

    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 32
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions:
        # note: this isn\t actually the resolution but
        # the downsampling factor, i.e. this corresnponds to
        # attention on spatial resolution 8,16,32, as the
        # spatial reolution of the latents is 64 for f4
        - 4
        - 2
        - 1
        num_res_blocks: 2
        channel_mult:
        - 1
        - 2
        - 4
        - 4
        num_heads: 8
        use_spatial_transformer: true
        transformer_depth: 1
        use_checkpoint: True
        context_dim: 768
        legacy: False
    first_stage_config:
      target: ldm.models.autoencoder.AutoencoderKL
      params:
        #ckpt_path: /workdir/latent-diffusion/models/first_stage_models/checkpoints/last.ckpt
        embed_dim: 4
        monitor: val/rec_loss
        ddconfig:
          double_z: true
          z_channels: 4
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity
    cond_stage_config:
      target: ldm.modules.encoders.modules.DummyEncoder
      params:
          key: t5 # CLIP, lazy not changing this, works the same
data:
  target: main.WebDataNpyModuleFromConfig
  params:
    batch_size: 1
    num_workers: 8
    training_urls: /workdir/datasets/windows_storage/tomandjerry_finetune.tar
    val_urls: /workdir/datasets/windows_storage/tomandjerry_val.tar
    test_urls: /workdir/datasets/windows_storage/tomandjerry_val.tar
    #null_cond_dropout: 0.2
```

The t5 in config file is actually CLIP, did not change it after first experiment, they are all pre-encodings, so just a lazy typo

## Hack the pretrained stable-diffusion weights to training checkpoint

If we are to resume training, we need a training checkpoint to resume on, but the released checkpoint are not for training, so we need to hack it

```
CUDA_VISIBLE_DEVICES=0 python main.py --no-test --base configs/latent-diffusion/finetune_stable_diffusion.yaml -t --gpus 0,
```

This will start a new training from scrach, and we only need the training checkpoint, so we abort training when finishing the first epoch (to spot bugs early, does not need the epoch to finish, abort when you please)

The checkpoint will be located in logs folder

Now let's hack the weights

```
model_train_dict = torch.load("/workdir/dev/latent-diffusion.dev/logs/2022-09-06T05-24-48_finetune_stable_diffusion/checkpoints/last.ckpt", map_location="cpu")

tmp_dict = model.state_dict()
keys_list = tmp_dict.keys()
for i in keys_list:
    if "cond_stage_model" not in i:
        model_train_dict['state_dict'][i] = tmp_dict[i]
torch.save(model_train_dict, "/tmp/test_merged.ckpt")
```

If you remember, we removed the CLIP model to free more vrams, so we should skip copying the cond_stage_model

Now put this checkpoint back into the logs folder, we are good to resume training now

## Resume training as finetuning

Replace your previous checkpoint folder in command

```
CUDA_VISIBLE_DEVICES=0 python main.py --resume logs/2022-09-06T05-24-48_finetune_stable_diffusion--base configs/latent-diffusion/finetune_stable_diffusion.yaml -t --gpus 0,
```

## Merge back to release checkpoint (optional)

After training, you should get a 10GB checkpoint, it would be better if we merge it to the original 4GB checkpoint so that everything is faster

```
for k in list(model_train_dict.keys()):
    if k != "state_dict":
        model_train_dict.pop(k, None)
patch_dict = model.state_dict()
new_patch_dict = {}
for i in patch_dict.keys():
    if i not in model_train_dict['state_dict'].keys():
        model_train_dict['state_dict'][i] = patch_dict[i]
torch.save(model_train_dict, "/tmp/test_merge.ckpt")
```

Now you get a 4GB checkpoint, well done

## Limits and weak points

During the finetuning process, the stable diffusion model starts to forget other objects in the same catagory, and everything will be biased to match the new finetuning dataset

After finetuning on Tom And Jerry images, the model starts to draw cat & kittens as tom, and cartoon bears as jerry, even without any prompt related to Tom And Jerry

![tom_biased.jpg](/assets/tom_biased.jpg)

## Possible improvements in the future

[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)

In their research, a class-specific prior preservation loss is suggested to improve the weak points mentioned above, however, their approach requires more vram to host a original model to gather the loss

I don't have the resource to do that, if you got more vram to spare, do try the class-specific prior preservation loss as optional improvement

## Citations

```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```
@misc{gal2022textual,
      doi = {10.48550/ARXIV.2208.01618},
      url = {https://arxiv.org/abs/2208.01618},
      author = {Gal, Rinon and Alaluf, Yuval and Atzmon, Yuval and Patashnik, Or and Bermano, Amit H. and Chechik, Gal and Cohen-Or, Daniel},
      title = {An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion},
      publisher = {arXiv},
      year = {2022},
      primaryClass={cs.CV}
}
```

```
@article{ruiz2022dreambooth,
  title={DreamBooth: Fine Tuning Text-to-image Diffusion Models for Subject-Driven Generation},
  author={Ruiz, Nataniel and Li, Yuanzhen and Jampani, Varun and Pritch, Yael and Rubinstein, Michael and Aberman, Kfir},
  booktitle={arXiv preprint arxiv:2208.12242},
  year={2022}
}
```
