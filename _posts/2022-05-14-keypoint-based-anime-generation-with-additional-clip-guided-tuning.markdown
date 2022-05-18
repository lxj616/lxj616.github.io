---
layout: post
title:  "Keypoint Based Anime Generation With Additional CLIP Guided Tuning"
date:   2022-05-14 18:36:10 +0800
categories: jekyll update
---

# Keypoint Based Anime Generation With Additional CLIP Guided Tuning

Make anime drawings with a selective pose and text captioning, is now possible

Well... to be precisely, It has been possible in theory for many years, but getting such a task done requires so much computing power

Or, does it ?

vq-f16 cond_transformer (real world pose keypoints):

![VQGAN f16 With Conditional Transformer](/assets/00019.jpg)

![VQGAN f16 With Conditional Transformer 2](/assets/00055.jpg)

vq-f4 latent-diffusion (pose from the training set, keypoints, input, reconstruction, ddim progressive row):

![VQGAN f4 With latent diffusion](/assets/ldm_example.jpg)

To avoid possible confusion, the image above is cherrypicked for better illustration of diffusion models, and is from the training set, result from testing the vq-f4 ldm model with real world poses are shown in below sections

## Possible for low computing scenarios

I have been testing different models & configs to get this done with my single RTX 3090 Ti and very limited human lifespan

**Finally I made it possible to train within a week (1 day for vq-f4 and 4 days for latent-diffusion as optimal, however I continue to observe the overfitting for another 3 days), and I also tried some extreme settings such as vq-f8 + 1/3 dataset with no luck**

Here are the specs and models I experiments on:

| Model            | Spec        | Dataset | Time    | Note                            |
| ---------------- | ----------- | ------- | ------- | ------------------------------- |
| cond_transformer | vq-f16      | 363k    | 20d     | vanilla first attempt           |
| ddpm             | vq-f4       | 363k    | 8d      | overfitting at day 5, 20 epochs |
| ddpm             | vq-f8       | 101k    | 4d      | too less data harms training    |

Please do remember these experiments are not with best optimal settings, some training time are redundant, dataset cleaning a little overhead, and will discuss what went wrong in the following blog entries

A glance at the cherry-picked results for each spec

1. vq-f16 ViT ![vq-f16 cond_transformer](/assets/00019_1_40.jpg)
2. vq-f4 ddpm ![vq-f4 ddpm](/assets/000068_1.jpg)
3. vq-f8 ddpm ![vq-f8 ddpm](/assets/000034_1.jpg)

vq-f4 / vq-f8 are easily converge in several epochs, and vq-f16 using 256 channels reconstruction image seems to be not improving after two weeks, and I used a small batch of 4 so vq-f16 trains much longer

diffusion model for 64x64 spartial size I trained for 7 days, around 32 epochs, overfitting at around day 4, 20 epochs

I found that the vq-f4 ddpm model outperforms vq-f16 cond_transformer, even with half the training time altogether, suggesting the vq-f16 hit its limit long before reaching 20 days, the loss is still decreasing, weird

Without cherry-picking, vq-f4 ddpm generates semantically consistent poses for the given condition, while vq-f16 cond_transformer could sometimes generate a total mess

But when using CLIP Guided tuning, vq-f16 is semantically better than vq-f8 and CLIP almost doesn't work with vq-f4, tests as below

## CLIP guided tuning with model generated anime images

A glance at the CLIP guided tuning example for each specs (using "colorful" as the magic word)

1. vq-f16 ViT ![vq-f16 cond_transformer](/assets/colorful_f16.jpg)
2. vq-f4 ddpm ![vq-f4 ddpm](/assets/000010_1629_50.jpg)
3. vq-f8 ddpm ![vq-f8 ddpm](/assets/000034_1_100.jpg)

It's pretty clear that vq-f8 is struggling to give different hair strands different color, but the semantic shape isn't consistent around the large skirt region

And as for vq-f4, CLIP with it seems to be operating pure pixel-wise, everywhere except the hair seems weird and without meaning, and the hair itself is not colorful, only partially getting vibrant

It seems that the CLIP guidance alone does not composite the image semantically, to get better results, even with pre-composition from other models, the optimizing target lantent space is better style-based than composition-based

## How do I make this happen under low computing restrictions

In short, I learned it the hard way

I'm gonna start writing a serie of blogs explaining the whole process, by timeline order as follows

1. Rethinking the Danbooru 2021 dataset
2. A closer look into the latent-diffusion repo, do better than just looking
3. The speed and quality trade-off for low computing scenarios

## Related pose keypoints dataset, code and model release

Keypoints tar ball(more details in coming up posts):

[https://drive.google.com/file/d/1KqdDfUJQkY-8MoQhnCCTXq-YpDciZlco/view?usp=sharing](https://drive.google.com/file/d/1KqdDfUJQkY-8MoQhnCCTXq-YpDciZlco/view?usp=sharing)

Code of the latent-diffusion fork repo with keypoints conditioning (pretrained vq/ldm models included in repo README):

[https://github.com/lxj616/latent-diffusion](https://github.com/lxj616/latent-diffusion)

## Citations

danbooru 2021 dataset originally contains 4.9m+ images, here I filtered out 363k subset, then further made a 101k tiny subset for further testing, https://www.gwern.net/Danbooru2021

latent diffusion models trained using [https://github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) , modifications on keypoints conditioning

vq regularized models trained using [https://github.com/CompVis/taming-transformers](https://github.com/CompVis/taming-transformers) , no modifications

CLIP guided tuning using [https://twitter.com/RiversHaveWings/status/1516582795438567424](https://twitter.com/RiversHaveWings/status/1516582795438567424) , directly on vq regularized model latents, not reranking the composition stage

```
@misc{danbooru2021,
    author = {Anonymous and Danbooru community and Gwern Branwen},
    title = {Danbooru2021: A Large-Scale Crowdsourced and Tagged Anime Illustration Dataset},
    howpublished = {\url{https://www.gwern.net/Danbooru2021}},
    url = {https://www.gwern.net/Danbooru2021},
    type = {dataset},
    year = {2022},
    month = {January},
    timestamp = {2022-01-21},
    note = {Accessed: DATE} }
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
@misc{esser2020taming,
      title={Taming Transformers for High-Resolution Image Synthesis}, 
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2020},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
```
@misc{https://doi.org/10.48550/arxiv.2204.08583,
  doi = {10.48550/ARXIV.2204.08583}, 
  url = {https://arxiv.org/abs/2204.08583},
  author = {Crowson, Katherine and Biderman, Stella and Kornis, Daniel and Stander, Dashiell and Hallahan, Eric and Castricato, Louis and Raff, Edward},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {VQGAN-CLIP: Open Domain Image Generation and Editing with Natural Language Guidance},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution Share Alike 4.0 International}
}
```
