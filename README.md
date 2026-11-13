
<h1 align="center">Fast Kernel-Space Diffusion for Remote Sensing Pansharpening</h1>


<div align="center">

**Hancong Jin, Zihan Cao, Liang-Jian Deng, Jingjing Li**

University of Electronic Science and Technology of China (UESTC)


[![Paper](https://img.shields.io/badge/Paper-CVPR%202026%20Findings-blue)](https://openaccess.thecvf.com/content/CVPR2026F/papers/Jin_Fast_Kernel-Space_Diffusion_for_Remote_Sensing_Pansharpening_CVPRF_2026_paper.pdf)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Hugging%20Face-yellow)](https://huggingface.co/datasets/elsting/PanCollection)

</div>

---





## 🚀 Abstract
KSDiff is a fast kernel-space diffusion framework for remote sensing pansharpening, aiming to fuse high-resolution panchromatic (PAN) and low-resolution multispectral (LRMS) images into high-quality multispectral images. Unlike conventional diffusion-based approaches that directly generate images through iterative sampling, KSDiff leverages diffusion models to generate convolutional kernels enriched with global contextual priors. Specifically, KSDiff integrates a low-rank core tensor generator and a unified factor generator through a structure-aware multi-head attention mechanism. A two-stage training strategy further enables seamless integration into existing pansharpening architectures. Extensive experiments demonstrate that KSDiff achieves superior pansharpening performance while delivering over 500× faster inference than diffusion-based pansharpening baselines.


## 🛠️ Inplementation



### Environment Setup

Clone this repository:

```bash
git clone https://github.com/0rzBen/KSDiff.git
cd KSDiff
```

Create a virtual environment:

```bash
conda create -n ksdiff python=3.10
conda activate ksdiff
```

Install the dependencies:

```bash
pip install -r requirements.txt
```


### Dataset

We conduct experiments on three widely used remote sensing pansharpening datasets:

- **WorldView-3 (WV3)**
- **QuickBird (QB)**
- **GaoFen-2 (GF2)**


All datasets are available through **[PanCollection (GitHub)](https://github.com/liangjiandeng/PanCollection)** and **[PanCollection (Hugging Face)](https://huggingface.co/datasets/elsting/PanCollection)**.






### Training

KSDiff adopts a two-stage training strategy. Before training, please configure the dataset paths and training parameters in `engine.py`.

#### Stage 1: Pre-training

In the first stage, set the following parameters in the `engine_google()` function call within `engine.py`:

```python
stage=1,
pretrain_weight_s1=None,
```

Then, run:

```bash
python engine.py --mode train
```

The pretrained model weights will be saved in the `weights/` directory.

#### Stage 2: Diffusion Model Training

In the second stage, initialize the model using the pretrained weights obtained from Stage 1.

Modify the corresponding parameters in `engine.py` as follows:

```python
stage=2,
pretrain_weight_s1="./weights/your_stage1_checkpoint.pth",
```

Then, run:

```bash
python engine.py --mode train
```

The trained model weights will also be saved in the `weights/` directory.



### Testing

To evaluate the trained model, run:

```bash
python engine.py --mode test
```

The test results will be saved in the `samples/mat` directory.



## 📝 Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{jin2026fast,
  title     = {Fast Kernel-Space Diffusion for Remote Sensing Pansharpening},
  author    = {Jin, Hancong and Cao, Zihan and Deng, Liang-Jian and Li, Jingjing},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  pages     = {6291--6301},
  year      = {2026}
}
```


