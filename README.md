## DiffDGSSv2: Towards Semantically Faithful Diffusion Representation for Generalizable Retinal Image Segmentation (TMI 2025)

[![Stars](https://img.shields.io/github/stars/Xyporz/DiffDGSSv2?style=flat-square)](https://github.com/Xyporz/DiffDGSSv2)
[![Forks](https://img.shields.io/github/forks/Xyporz/DiffDGSSv2?style=flat-square)](https://github.com/Xyporz/DiffDGSSv2/fork)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-%20-orange?style=flat-square)](https://pytorch.org)

### 📄 Abstract
Retinal image segmentation is essential for analyzing retinal structures like vessels and diagnosing retinopathy. However, the inherent intricacy of the retina, along with annotation scarcity and data heterogeneity, presents prevalent challenges in creating accurate and generalizable deep learning models. Diffusion models, while initially developed for image generation, have recently shown great promise for visual perception by leveraging the learned internal representations. However, these diffusion representations, which spread across network blocks (space) and diffusion timesteps (time), potentially suffer from issues like stochastic semantic distortion and cumulative structural blurring, compromising their semantic fidelity to the source image. In this paper, by delving into the generalization property of diffusion models, we propose a novel anchoring inversion strategy to derive diffusion representations that are semantically faithful to the source image from the deterministic trajectory. Furthermore, we introduce a time-space frequency-aware aggregation interpreter (T&S-FreqAgg) to aggregate the multi-scale and multi-timestep diffusion representations in a frequency-aware way for Domain Generalizable Semantic Segmentation (DGSS). Extensive experiments on nine public retinal image datasets demonstrate the superiority of our proposed framework, DiffDGSSv2, over state-of-the-art methods. Our code will be available at: https://github.com/Xyporz/DiffDGSSv2.

This codebase is largely based on and adapted from [yandex-research/ddpm-segmentation](https://github.com/yandex-research/ddpm-segmentation).

### 🚀 Quick Start
- Datasets: place images and same-named `.npy` masks side by side (example):
```
datasets/
  horse_21/
    real/
      train/
        xxx.jpg, xxx.npy
      test/
        yyy.jpg, yyy.npy
```
- Packaged download: [datasets.tar.gz](https://storage.yandexcloud.net/yandex-research/ddpm-segmentation/datasets.tar.gz) (~47 MB)
 
- Train & evaluate (example):
```
bash scripts/ddpm/train_interpreter.sh horse_21
```

### 🧠 Pretrained Diffusion Model

- Set pretrained diffusion weights via `model_path` in `experiments/<DATASET>/ddpm.json`.
- For Horse-21, download the pretrained checkpoint:
  - LSUN-Horse: [lsun_horse.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_horse.pt)
- For retinal images, users may train their own diffusion model on EyePACS with [guided-diffusion](https://github.com/openai/guided-diffusion); for reference, our configuration is:
```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 4000 --image_size 512 --learn_sigma False --noise_schedule cosine --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --predict_xstart True"
```

### 📚 Citation
```
@article{xie2025towards,
  title={Towards Semantically Faithful Diffusion Representation for Generalizable Retinal Image Segmentation},
  author={Xie, Yingpeng and Chen, Hao and Qin, Jing and Zhang, Yongtao and Dong, Lei and Du, Jie and Wang, Tianfu and Lei, Baiying},
  journal={IEEE Transactions on Medical Imaging},
  year={2025},
  publisher={IEEE}
}
```

