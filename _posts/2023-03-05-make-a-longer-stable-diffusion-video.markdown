---
layout: post
title:  "Make a longer stable diffusion video on home computers"
date:   2023-03-05 09:37:50 +0800
categories: jekyll update
---

# Make A Longer Stable Diffusion Video On Home Computers

![trump_small.gif](/assets/trump_small.gif)
![spiderman3_small.gif](/assets/spiderman3_small.gif)
![ironman_small.gif](/assets/ironman_small.gif)
![spiderman2_small.gif](/assets/spiderman2_small.gif)

The gif is manually resized to 256x256 and heavy lossy compressed using `gifsicle -O3 --lossy=200 --resize 256x256` for better blog network loading speed, originally trained at 512x512 and 64 frames

However these gifs are more than 1MB each, so if you have trouble loading the gif, you may need to go and download from github blog repo yourself, I can't compress the gifs any further

**The dataset and model heavy resemble real life human with personal identity such as faces and bodys, thus can not go opensource for legal concerns**

## Longer Problem For Longer Video

1. It simply just won't work if you set frames_length to higher value and press enter harder with your finger

2. My previous timelapse toy model used timelapse video dataset, although they have adequate clip length for longer video experiment, it doesn't make sense training longer sequence when shorter is good enough

3. As in tradition, one single RTX 3090Ti (24 GB vram) is what I got, and all the fancy longer video generation stuff, I mean it to get it done with the exact same home computer computation limitations

## The Missing Make A Video Technique

Well, I already give out make-a-stable-diffusion-video github repo to demonstrate how to make it work, especially on home computers

And I stated that in my last blog post: 'Oh, I can not afford that with 24gb vram, let's just pretend there isn't a whole paragraph in make-a-video paper explaining video frame interpolation'

Now that I'm gonna try more frames and wish to get coherent results for a long range of frames, and probably finish training myself instead of leaving a letter to my grandson

**Video frame interpolation is what I need, the missing piece**

[https://paperswithcode.com/task/video-frame-interpolation](https://paperswithcode.com/task/video-frame-interpolation)

So I made a hack to my code, using the inpainting model special feature to implement fast and incorrect interpolation

    hint_latents = latents[:,:,0::4,:,:]
    hint_latents_expand = hint_latents.repeat_interleave(4,2)
    hint_latents_expand = hint_latents_expand[:,:,:args.frames_length,:,:]
    latent_model_input = torch.cat([noisy_latents, masks_input, hint_latents_expand], dim=1).to(accelerator.device)

Well, for every 4 frames, set the original frame to inpainting condition input to generate exact frame image since I masked nothing

Good thing is this hack is almost one line without custom attention module modification, bad thing is this is mathematically wrong because I really should set the static frame without model backbone inference on it

And as for duplicating the hint input to every subordinate frames, I didn't do research on its effects, I can't answer it because I have no clue myself

So, here is the plan:

1. generate 5 frames
2. interpolate to 17 frames, (5-1)x4+1=17
3. interpolate to 65 frames, (17-1)x4+1=65

## Attention Is All I Can Not Afford

Surely I wouldn't meet many trouble dealing with 5 frames, whatever attention I use

But 65 frames leads to a huge problem, especially when we do interpolation

Normal attention has O(n^2) complexity

**For 5 frames, n is 5, so that would be 5^2=25 units of complexity**
**For 65 frames, n is 65, so that would be 65^2=4225 units of complexity**

So it is obvious I need to train 65 frames model 169x times than the 5 frames model, this could be a problem

I actually tried that for comparison, the 65 frames model is a total mess visually and by loss curve, even already trained for 20 epochs, I should have saved the screenshot, but I get too frustrated and forgot

Here comes a better(?) attention mechanism for long sequences

[https://paperswithcode.com/method/sliding-window-attention](https://paperswithcode.com/method/sliding-window-attention)

With sliding window attention such as local attention, you only need O(n x w) complexity, for 65 frames and windowsize 5, its 65x5=325 units of complexity, compared to 4225 it almost seem like a silver bullet

But, does it ?

By using [https://github.com/facebookresearch/xformers/blob/main/xformers/components/attention/local.py](https://github.com/facebookresearch/xformers/blob/main/xformers/components/attention/local.py)

**I found the shortcomings of local attention, which is too narrow minded on adjacent frames, the most notable effect is the identity of the main subject changes rapidly across frames**

Good thing is local attention's shortcomings are trival to our interpolation network, we always have a reference frame every 4th frames, so no need to worry about identity change

Note: I have no idea what windowsize is the best, I just used 5 which is the default value

It turned out by using local attention, I am being able to train the 65 frames interpolation network for 20 epochs and seems to be good enough (my fancy way of saying I ran out of patience and stopped)

## A Not Too Short And Not Too Small Dataset For Testing

Many would ask: what's wrong with the timelapse video dataset, it's long enough and you already have that, why bother making another dataset ?

1. timelapse video dataset is too small (286 videos), it does not generalize well to make creativity art, a cat standing idle with clouds moving is almost the only thing it does
2. stable diffusion is good at generating landscape images, and human eyes are not sensitive about nature scenes, nature landscape change very little across time, a timelapse dataset can cover hidden technical problems, but this time I shall face the real challenge

So, this time I made a 'fashion model walking on stage' video dataset, it has the following features:

1. 2848 videos, all above 100 frames, almost 10x the size of timelapse video dataset
2. contains human, a forbidden area for stable diffusion both for poor generation quality and for legal obligations
3. human changes scale across time, far to near, not the same size

**Speak of legal obligations, I am not being able to opensource the dataset or the model, because it is trained on human, it has to more or less contain personal identity information such as faces and bodys**

**All I could publish is where I got the raw videos:**

[https://www.youtube.com/@yaersfashiontv](https://www.youtube.com/@yaersfashiontv)

And I used blender to preprocess the videos, manually

## Unverified experimental hacks

Except for what I did and tested above, I actually experimented custom attention patterns like always attend to first frame no matter what window size

But I can not tell the difference, so without proof I can just say I am not able to confirm whether they work or not

## Recommended Opensource Implementations

I have noticed that there is a cleaner and easier to use implementation other than my make-a-stable-diffusion-video repo

**If you got enough vram and wish not to use my hacks (which mainly focus on running under 24GB vram), you can check this work in progress implementation by chavinlo**

[https://github.com/chavinlo/TempoFunk](https://github.com/chavinlo/TempoFunk)

## Citations

Thanks to the opensource repos made by [https://github.com/lucidrains](https://github.com/lucidrains) , including but not limited to:

https://github.com/lucidrains/make-a-video-pytorch

https://github.com/lucidrains/video-diffusion-pytorch

And my code is based on [https://github.com/huggingface/diffusers](https://github.com/huggingface/diffusers), especially most of the speed up tricks are bundled within the original repository

```
@misc{Singer2022,
    author  = {Uriel Singer},
    url     = {https://makeavideo.studio/Make-A-Video.pdf}
}
```

```
@misc{ho2022video,
  title   = {Video Diffusion Models}, 
  author  = {Jonathan Ho and Tim Salimans and Alexey Gritsenko and William Chan and Mohammad Norouzi and David J. Fleet},
  year    = {2022},
  eprint  = {2204.03458},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV}
}
```

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
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```
