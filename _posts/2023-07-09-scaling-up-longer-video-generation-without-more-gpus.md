---
layout: post
title:  "Scaling up longer video generation model training without more gpus"
date:   2023-07-09 21:46:14 +0800
categories: jekyll update
---

# Scaling Up Longer Video Generation Model Training Without More GPUs

![darth_vader_final.gif](/assets/darth_vader_final.gif)
![harley_final.gif](/assets/harley_final.gif)

The gif is manually resized to 256x256 and heavy lossy compressed using `gifsicle -O3 --lossy=200 --resize 256x256` for better blog network loading speed, originally trained at 512x512 and generates more than 181 frames

However these gifs are more than 1MB each, so if you have trouble loading the gif, you may need to go and download from github blog repo yourself, I can't compress the gifs any further

**The dataset and model heavy resemble real life human with personal identity such as faces and bodys, thus can not go opensource for legal concerns**

## TL;DR

I used my RTX 3090Ti and created a 24370 clips dataset and trained a model under 24GB vram limitation that is capable of generating hundreds of frames with some consistency to the first frame, but during this experiment I changed every possible thing mid-training so there is no solid proof of what I learnt except for it more or less works this way

## Scaling up the dataset

Last time I hand crafted a walking on stage video dataset containing 2848 clips, and I trained on each first 65 frames

Which is bigger than the far previous 286 timelapse video dataset, but still too small for some real challenge

So I gathered a human dancing dataset from various internet sources, containing 24370 video clips and has 181 frames each

It is the most difficult subject for image generation and video generation: human and rapid motion

1. The clips are aligned using pose detection, and resized to 512x512
2. Each clip contains at maximum 2 alternative augmentation, so there are more than 24370 actual clips when training
3. Contains some "bad" clips which contains heavy camera motion, or the human ran out of screen

## Scaling up video duration by interpolation and extrapolation

Last time I did video interpolation on the whole clip, which contains two interpolation stages: 5 frames --> 17 frames --> 65 frames

And using local attention to crop down computational requirements

Although it is working at least, but generating 65 frames already consumed 24GB vram even with accelerate/deepspeed optimization and gradient checkpointing

If to generate as long as 181 frames, I decided to train the model in a autoregressive way

1. a base model generating 4 frames, and with some hacky inference technique, can generate 7 frames, as called "starter model"
2. a extrapolation model generate new 3 frames from the previous frames, but with a step of every 4th frames
3. a interpolation model fills the previously newly generated 3 frames with a total of 9 frames (fill in 3 frames into the two gaps)

The frame number generated in the following way (newly generated frame ends with a !):

- 1, 2!, 3!, 4!, 5!, 6!, 7!
- 1, 2, 3, 7, 11!, 15!, 19!
- 1, 2, 6, 7, 8!, 9!, 10!, 11 ...

I know it's vague, don't get too serious about it, it is a rough hack by myself and does not work too well, for now

Good thing is that by this method, I don't need to do gradient checkpointing and cpu offloading, which speeds up training further, not to mention 7 frames iters far quicker every step than 65 frames

However, when handling dataset this large, I need to further speed it up, not only on the training side

## Dataset hack & cheating

**Well, if you are doing academical reserch, don't do anything like this**

I got inspired by [https://arxiv.org/abs/2206.07137](https://arxiv.org/abs/2206.07137), as the title suggests:

> Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt

I decided to train the model on the full dataset first (and never ending), for a few epochs, then determine which data points are worth learning

I don't want to talk about it in detail, changing the dataset itself mid-training is a forbidden method because the result of the training became unreproducible and fragile 

However since I never finish training or writing papers, so this is not a problem to me

I deleted 10% of the dataset which seems to be hard to learn in the first starter stage

And also 10% in the expolation skipper stage, too, but not the same 10% (yeah, maybe it would be better if treated the same)

The training loss drop like crazy, and the test generation is improving faster after since

However, I can't give out any proof of what I felt, this experiment is totally inaccurate because of the following reasons

## Noise augmentation to keep the long time consistency

When doing autoregressive generation, error gathers every iteration, so say we generate 120 frames in my method, we need to generate around 10 loops, each loop using previously generated content as hint, and that could be very inaccurate

So firstly I employ signal noise ratio 0.33 at generated frames (but not the first hint frame), it seems to be good when testing with very few test clips

Then I found it wasn't enough, then I changed the augmentation noise, from signal noise ratio 0.33 to 0.66, it gets better, I feel better, no proof of any kind however ...

And this means I changed the augmentation mid-training, I would be fired if I am a scientist LOL

## half-way fix of half-way attention

When I coded the first version of this experiment, I used a half-way attention to split the sequence in half then combine after calculation

Which yields max 0.06 error every time and the average error is 0.01, I thought that was acceptable, much better than out of vram doing nothing

But yet I forgot about it, and didn't revert the half-way attention hack, when I realized about this, I decided to revert to 'correct as a whole' attention mid-training

Okay, this is to say, I changed the model structure at mid training, this is not good, very not good, but neccesary

## Power failure and forgot to dump adam optimizer state

Em... yeah, I forgot to dump adam optimizer state at first, then my apartment got power failure mid-training

So, the training does not need to restart from the beginning but the training loss went crazy for days before it talks sense

## What I learned from the experiment

So much for confession, despite all the bad things I hacked and fixed, I actually learned something as follows

1. Always dump optimizer states when training with adam something
2. A hack can be helpful at first when testing, if you forget about it when scaling up, it could be a disaster
3. Noise augmentation is very cool, but determine how much noise to add, is a total pain in the (beep)
4. Autoregressive is good, saves vram, saves time, if you code it right, it will crash later than sooner
5. I realized I have to redo the experiment again with smarter generation schedule to make sure the quality won't drop significantly across time, not to mention everything I did wrong

## Not Really a Conclusion

I changed model structure, augmentation, dataset, and optimizer state mid-training, these are unforgivable mistakes that should be avoided, but

At least it works, barely works, but it works

And hey, it's under 24GB vram, and capable of generating hundreds of frames

I am so eager to share with everyone what I did good, but currently the quality is poor, that is to say I am not doing good for now

So at it's current state, if to claim that the model works, it would be a false claim, sharing non-working code would be irresponsible and thus I won't update my github repo this time, but hopefully not for long

## Limitations

1. Every time the generated illustrated figure tries to turn their heads left or right, it creates artifacts, stable diffusion v1.5 cannot handle these circumstances well
2. The generated figure tends to become female in the autoregressive pipeline, due to the dataset bias
3. Although in theory it can generate unlimited length of clips, human rapid actions always reach a status that the generation is broken, such as too far or too close to the camera etc
4. If the generated figure not moving fast, there is overfitting on background

## Citations

Thanks to the opensource repos made by [https://github.com/lucidrains](https://github.com/lucidrains) , including but not limited to:

https://github.com/lucidrains/make-a-video-pytorch

https://github.com/lucidrains/video-diffusion-pytorch

And my code is based on [https://github.com/huggingface/diffusers](https://github.com/huggingface/diffusers), especially most of the speed up tricks are bundled within the original repository

```
@misc{mindermann2022prioritized,
      title={Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt}, 
      author={Sören Mindermann and Jan Brauner and Muhammed Razzak and Mrinank Sharma and Andreas Kirsch and Winnie Xu and Benedikt Höltgen and Aidan N. Gomez and Adrien Morisot and Sebastian Farquhar and Yarin Gal},
      year={2022},
      eprint={2206.07137},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```
@misc{Singer2022,
    author  = {Uriel Singer},
    url     = {https://makeavideo.studio/Make-A-Video.pdf}
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
