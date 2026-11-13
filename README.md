
<h1 align="center">Fast Kernel-Space Diffusion for Remote Sensing Pansharpening</h1>

<div align="center">

**Hancong Jin, Zihan Cao, Liang-Jian Deng, Jingjing Li**

[![Paper](https://img.shields.io/badge/Paper-CVPR%202026%20Findings-blue)](https://openaccess.thecvf.com/content/CVPR2026F/papers/Jin_Fast_Kernel-Space_Diffusion_for_Remote_Sensing_Pansharpening_CVPRF_2026_paper.pdf)

</div>

---





## 🚀 Abstract
KSDiff is a fast kernel-space diffusion framework for remote sensing pansharpening, aiming to fuse high-resolution panchromatic (PAN) and low-resolution multispectral (LRMS) images into high-quality multispectral images. Unlike conventional diffusion-based approaches that directly generate images through iterative sampling, KSDiff leverages diffusion models to generate convolutional kernels enriched with global contextual priors. Specifically, KSDiff integrates a low-rank core tensor generator and a unified factor generator through a structure-aware multi-head attention mechanism. A two-stage training strategy further enables seamless integration into existing pansharpening architectures. Extensive experiments demonstrate that KSDiff achieves superior pansharpening performance while delivering over 500× faster inference than diffusion-based pansharpening baselines.


## 🛠️ Inplementation



### Environment Setup

Clone this repository:

```bash
git clone https://github.com/USERNAME/REPOSITORY.git
cd REPOSITORY
```

Create a virtual environment:

```bash
conda create -n FKSD python=TODO
conda activate FKSD
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Dataset

<!-- TODO: Describe the datasets used in the experiments. -->

The datasets used in this work include:

- **WorldView-3 (WV3)**
- **QuickBird (QB)**
- **GaoFen-2 (GF2)**
- **TODO:** Other datasets, if applicable.



### Training

<!-- TODO: Provide training instructions and configuration details. -->

To train the proposed model, run:

```bash
python train.py --config configs/XXX.yaml
```



### Testing

To evaluate the trained model, run:

```bash
python test.py --config configs/XXX.yaml --checkpoint checkpoints/XXX.pth
```



## 📝 Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{jinTODO,
  title   = {Fast Kernel-Space Diffusion for Remote Sensing Pansharpening},
  author  = {Jin, Hancong and Cao, Zihan and Deng, Liang-Jian and Li, Jingjing},
  journal = {TODO},
  year    = {TODO}
}
```


