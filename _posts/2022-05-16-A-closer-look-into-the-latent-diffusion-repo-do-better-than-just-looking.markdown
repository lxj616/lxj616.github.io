---
layout: post
title:  "A Closer Look Into The latent-diffusion Repo, Do Better Than Just Looking"
date:   2022-05-16 14:16:02 +0800
categories: jekyll update
---

# A Closer Look Into The latent-diffusion Repo, Do Better Than Just Looking

The two stage compress-then-diffusion boosts training efficiency dramatically, which made low computing art creations possible

For those readers who aren't familiar with [https://github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) , please see [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

I would assume that you already tried or is going to try out this model-combo for your own artistic datasets

The authors already released some pretrained vq/kl regularized autoencoder models, **If your dataset looks like openimages dataset, or mathematically speaking your dataset have a similar visual feature distribution with the openimages dataset, then you're in luck, just grab one pretrained autoencoder and train your ldm with demo configs**

But what if your own dataset is visually very not similar from openimages, for example, danbooru anime dataset ?

## The pretrained autoencoder is not a silver bullet

Let's give the pretrained vq-f4 a reconstruction shot at danbooru images

![151115](/assets/151115.jpg)
![151115_rec](/assets/151115_rec.jpg)
![80100](/assets/80100.jpg)
![80100_rec](/assets/80100_rec.jpg)

Hmmm, the eyes are a little off, but it looks kinda fine

Then the pretrained vq-f8:

![151115](/assets/151115.jpg)
![151115_rec_f8](/assets/151115_rec_f8.jpg)
![80100](/assets/80100.jpg)
![80100_rec_f8](/assets/80100_rec_f8.jpg)

Oh no, this is giving me nightmare

So there is no need to test out the vq-f16, vq-f8 is compressing way too much

**The question is: is the pretrained vq-f4 on openimages good enough for danbooru dataset ?**

**Well, considering I already found that re-train a vq-f4 only takes one or two epochs, it's really not necessery to endure with the eye detail gliches, as well as the biased codebook distribution**

But even if the autoencoder training takes long, I still wouldn't chose to use the pretrained vq-f4 on danbooru dataset, not only because the 'best reconstruction' is not good enough, the distribution of the codebook entries are very different than the danbooru dataset as well, it means that somewhere between a dress fiber texture code and hair strand texture code, there is a squirrel fur texture code should not be used, but will be optimized during later diffusion training, I have no idea what consequence it shall make but definitely not favoring it

## Selecting a proper autoencoder config

I have tried vq-f16/vq-f8/vq-f4 with the default config and the trainable parameter count are near identical due to gpu limit

**Generally speaking, if you are creating 256x256 images, vq-f4 leads to a 64x64 diffusion model training, after my experiments, this is the best combo config that works really well**

And after trying vq-f8, I found it hard to sustain the reconstrution quality without ramping up model abilities, with the same trainable parameters, it does't do details well, another problem is if 64x64 diffusion model is training as fast as in 5 days on my case, spend more time on autoencoders in exchange for a easier diffusion training does not seem to be worth it

I also tried vq-f16 on my first attempt, nah... , it's working, but after two weeks time, it doesn't seem to be more impressive, all the gloomy details drives me mad, compared with vq-f4 + 64x64 diffusion, it's totally not worth it, unless you wanna try 1024x1024 high resolution image generation, which makes vq-f16 + 64x64 diffusion seems proper, but that shall be a different story then

## Conditioning on keypoints

[https://github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) already give examples about how to use cross attention to do conditional training, such as conditioning on image/class_label/object_center_point/depth_map etc, and I just made the 17 keypoints into a [1, 17] tensor, just as the object center point hack

```
    def tokenize_coordinates(self, x: float, y: float, dim:int) -> int:
        """
        Express 2d coordinates with one number.
        Example: assume self.no_tokens = 16, then no_sections = 4:
        0  0  0  0
        0  0  #  0
        0  0  0  0
        0  0  0  x
        Then the # position corresponds to token 6, the x position to token 15.
        @param x: float in [0, 1]
        @param y: float in [0, 1]
        @return: discrete tokenized coordinate
        """
        #x_discrete = int(round(x * (self.512 - 1)))
        #y_discrete = int(round(y * (self.512 - 1)))
        if x > (dim - 1):
            x = (dim - 1)
        if y > (dim - 1):
            y = (dim - 1)
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        return int(round(x)) + int(round(y) * dim)

    def coordinates_from_token(self, token: int, dim: int) -> (float, float):
        x = token % dim
        y = token // dim
        return x, y

    def build_tensor_from_kps(self, kps, dim):
        kps_names = ['nose',
                     'eye_left',
                     'eye_right',
                     'ear_left',
                     'ear_right',
                     'shoulder_left',
                     'shoulder_right',
                     'elbow_left',
                     'elbow_right',
                     'wrist_left',
                     'wrist_right',
                     'hip_left',
                     'hip_right',
                     'knee_left',
                     'knee_right',
                     'ankle_left',
                     'ankle_right']
        tokens = []
        for name in kps_names:
           x = kps[name][0]
           y = kps[name][1]
           if dim != 512:
               x = x // (512/dim)
               y = y // (512/dim)
           _token = self.tokenize_coordinates(x, y, dim)
           tokens.append(_token)
        #return LongTensor(tokens)
        return Tensor(tokens)
```

1. A custom dataloader to load the json keypoints file
2. A DummyEncoder to construct the [1, 17] tensor from keypoints
3. Add keypoints option to the conditioning key for the model, modify log_images function to log conditioning keypoints as well
4. Adapt the config to use keypoints conditioning and enable spartial transformer cross attention options

The complete code can be found at [https://github.com/lxj616/latent-diffusion](https://github.com/lxj616/latent-diffusion)

## Training the diffusion model

I decreased the model_channels to 160 (from 256) to save vram for larger batches, everything else is more or less copied from example config

This config file can be found at configs/latent-diffusion/danbooru-keypoints-ldm-vq-4.yaml in my latent-diffusion fork repo

```
model:
  base_learning_rate: 2.0e-06
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    cond_stage_key: keypoints
    linear_end: 0.0195
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: image
    conditioning_key: crossattn
    image_size: 64
    channels: 3
    monitor: val/loss_simple_ema
    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 64
        in_channels: 3
        out_channels: 3
        model_channels: 160
        attention_resolutions:
        - 8
        - 4
        - 2
        num_res_blocks: 2
        channel_mult:
        - 1
        - 2
        - 3
        - 4
        num_heads: 8
        use_spatial_transformer: true
        transformer_depth: 1
        context_dim: 17
        legacy: false
    first_stage_config:
      target: ldm.models.autoencoder.VQModelInterface
      params:
        embed_dim: 3
        n_embed: 8192
        ckpt_path: /workdir/taming-transformers/logs/2022-04-25T12-37-06_custom_vqgan/checkpoints/last.ckpt
        ddconfig:
          double_z: false
          z_channels: 3
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity
    cond_stage_config:
      target: ldm.modules.encoders.modules.DummyEncoder
      params:
        key: keypoints
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 32
    num_workers: 8
    wrap: false
    train:
      target: ldm.data.custom_616.CustomTrain
      params:
        training_images_list_file: /workdir/datasets/danbooru_tiny.list
        size: 256
    validation:
      target: ldm.data.custom_616.CustomTest
      params:
        test_images_list_file: /workdir/datasets/danbooru_tiny_test.list
        size: 256

```

After 20 epochs of training, I suspect there is some overfitting, and keep observing until epoch 33, it became more and more obvious

![overfitting](/assets/progressive_row_gs-380000_e-000033_b-004988_cropped.jpg)

The model is trying to denoise some tiny details in the last steps, only to make the details worse, this is very dataset specific, too much noise in the danbooru dataset make details impossible to refine, as well as lack of data in total

**So by calculation 20 epochs is good enough, which is 2022-04-30 ~ 2022-05-04 (5 days), the vq-f4 training took a day, so this is within a week on a single 3090 Ti**

Maybe I should not have filtered the dataset too heavy losing too much slightly noisy data ...

## Sample using pose keypoints

In short, parse the conditioning to ddim sampler, and you'll get the conditioned output

There is a demo script at scripts/sample_pose_diffusion.py in my latent-diffusion fork repo

```
samples, intermediates = ddim.sample(steps, conditioning=cond, batch_size=bs, shape=shape, eta=eta, verbose=False,)
```

And there are many hidden tricks that ddim already have, such as inpainting, priming on a different noise latent ...

When I tried to illustrate the final output quality, **I chose a real world pose example from a short video**, instead of using poses from the training/validation set, this is more fun and fair to demonstrate the model capabilities

cherry-picked:

![vq-f4 ddpm](/assets/000068_1.jpg)

random sample:

![random sample](/assets/total_grid.jpg)

However if you really need to generate better quality images, you can also consider **the “truncation trick”** similar to stylegan/biggan, but in our case is to select **"the most common top N poses"**

Ha, gotcha, there is no common pose in the dataset, the keypoints are way too scattered to be in common, there are total 17 points everywhere, how could they possible accidentally be the same ? The 'pose available space' is 87112285931760246646623899502532662132736 large(I regularized the coordinate to be within 16x16 grids each, math.pow(256, 17)), good luck finding a most common pose to get the trick done

The following pose is from the top row image in the training set, the generated output image is with similar pose but not identical with the training set image, there must be a similar image in the dataset though, if you don't mind, selecting poses from the training set can give you better results, but not as fun as selecting real world poses to challenge the model, and to further improve the output quality by cheating, select a top common pose rather than random training set pose

![VQGAN f4 With latent diffusion](/assets/ldm_example.jpg)

## Failure attempts

I tried to further clean up the danbooru dataset subset, reducing the noisy images and try vq-f8 to get details right and blazing fast, ended up worse output quality due to lack of data, details see my last post about datasets

I tried to clean up anime image backgrounds, it wasn't accurate enough, introduces new random image feature noises, not working

I forgot to set spartial transformer in the config, find that out after many days when the log image can be clearly distinguished, my heart is broken, especially when I see the code comment after carefully debug

```
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)
```

## Citations

latent diffusion models trained using [https://github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) , modifications on keypoints conditioning

vq regularized models trained using [https://github.com/CompVis/taming-transformers](https://github.com/CompVis/taming-transformers) , no modifications

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
@misc{esser2020taming,
      title={Taming Transformers for High-Resolution Image Synthesis}, 
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2020},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
