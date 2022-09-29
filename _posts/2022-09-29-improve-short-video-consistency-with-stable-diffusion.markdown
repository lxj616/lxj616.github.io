---
layout: post
title:  "Improve short video consistency with stable diffusion"
date:   2022-09-29 19:04:25 +0800
categories: jekyll update
---

# Improve Short Video Consistency With Stable Diffusion

Stable diffusion has a built-in example for img2img generation and thus we could easily adopt it for vid2vid, however, it does not seem to be good enough keeping the video frames consistent and smooth

![video_consistency.gif](/assets/video_consistency.gif)

In case you have doubts, I already used fixed noise and fixed seed(s) for all frames, now we can focus on the obvious problems

The gif above is resized and compressed for better webpage loading, not the original length and quality

1. One problem is that if we select a 'noising strength' too low as right-top-corner (--strength 0.45), the model seems doing trival edits which does not do anything but for adding jumping artifacts across frames
2. Another problem is that if we select a higher 'noising strength' as left-bottom-corner (--strength 0.75), the model ignores the obvious object across frames and makes the car disappear, and I still feel it not artistic enough

Here I adopt a idea from paper [Deep Video Prior for Video Consistency and Propagation](https://arxiv.org/abs/2201.11632), and make it like right-bottom-corner achieving better video consistency for short videos

## Not the old content style balance problem

If you remember the old neural style stuff, you could recall something named content style balance, there is a magic ratio to be tuned manually so as to find better trade-off on content fidelity against style

Here we have a parameter 'noising strength', you put a 0.01 and got near exact the original content, and you put a 0.99 for total imagination with prompt, could there be a satisfying value in the middle ?

Well, I couldn't find one, and even with my hack done, the video is still kind of jumpy, the improvement is limited

**You have to increase content fidelity by using a lower noising strength for video frames consistency, but how are you going to make notable text prompt edits on such low noising strength ?**

Now we got a problem to solve

## Short video as the unconditional dataset

We hope the stable diffusion model to generate video frames according to the reference video, at some degree, we do not wish to generate something far from all frames

So we could finetune the stable diffusion model to reconstruct better video frames if not given any text prompts, then use text prompts to edit them

A fun fact is that after many experiments, I found 30 frames is good enough to deal with a 300 frames short video, not really need to finetune on them all, unless your video got sudden subject twists

## Text to image as the conditional dataset

Select a frame as a example, do txt2img until you are satisfied, with a rather large noising strength, don't worry about the content may inconsistent with the original frame yet, we have more steps further down

1. It is okay that the edited frame has obviously changed too much in color space, for example black shirts to red dress, you may use (--strength 0.75) and even more
2. It is NOT okay if the subject changed composition too much, for example human arm position may change a lot, generate more images to select the nearest one, or decrease the noising strength, frames are going jumpy otherwise
3. Remember the text prompt

## Finetune on combined dataset

Now, we got a unconditional dataset which consists 30 frames, with empty text embedding, a conditional dataset consists maybe 2 different text prompts on 1 frame

So we have a dataset of 32 frames in total

Let's resume training on stable diffusion as finetuning, if you have not read my previous post about how to finetune the model, it is time to go for it now

I have also employed some techniques I discovered earlier, including only finetune on late steps to speed up training

And make sure text conditional dataset start denoising from its paired original frame as starting point

Due to the small amount of frames (32 in this case), the whole process is within hours, for one single video, but the output quality still needs to be improved

## Further details

The original video is from youtube 'https://www.youtube.com/channel/UCBcVQr-07MH-p9e2kRTdB3A', author J Utah, cropped 10 seconds (from 1.5 hour) and to 512x512

The text prompt is "a abstract painting of a cyberpunk city night, tron robotic, trending on artstation"

Strength parameter for clockwise: original, 0.45, 0.45, 0.75 (after finetuning, you can lower the strength parameter to get more fidelity, I use 0.45 for comparison, for human 0.325 is good enough)

Finetuned for 1000 iters (for human only need around 400 iters), 1e-5 lr, late steps 500

Generation using 50 steps (lazy, nah)

Using blender for linux to combine the image to videos

## Citations

```
@inproceedings{lei2020dvp,
  title={Blind Video Temporal Consistency via Deep Video Prior},
  author={Lei, Chenyang and Xing, Yazhou and Chen, Qifeng},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}    
```

```
@article{DVP_lei,
  author    = {Chenyang Lei and
               Yazhou Xing and
               Hao Ouyang and
               Qifeng Chen},
  title     = {Deep Video Prior for Video Consistency and Propagation},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year      = {To Appear}
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
