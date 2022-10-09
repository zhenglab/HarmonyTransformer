<base target="_blank"/>


# Transformer for Image Inpainting<br>

Here we provide PyTorch implementation and the pre-trained model of our latest version.

## Prerequisites

- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Train/Test
- Download Paris StreetView dataset, and put it in the directory `dataset/images/`.

- Train our HT+ model (FC-TRE-DeCNN):
```bash
python train.py --path=$configpath$
```
- Test our HT+ model
```bash
python test.py --path=$configpath$
```

## Apply a pre-trained model
- Download the pre-trained model from [Google Drive](https://drive.google.com/file/d/1pRDpYZrRd6iR314skgX086lmYvUMExIx/view?usp=sharing) or [BaiduCloud](https://pan.baidu.com/s/10xrromZdpnD5lB_nF1MO2Q) (access code: cb09); 

- Put `g.pth` in the directory `checkpoints/paris-fcin-deconvout-2H6L` and modify `MODE: 1` to `MODE: 2` in the `checkpoints/paris-fcin-deconvout-2H6L/config.yml`;
- Run:
```bash
# Our HT model
python test.py --path=checkpoints/paris-fcin-deconvout-2H6L
```

# Bibtex
If you use this code for your research, please cite our papers.


```
@article{guo2022transformer,
  title={Transformer for Image Harmonization and Beyond},
  author={Guo, Zonghui and Gu, Zhaorui and Zheng, Bing and Dong, Junyu and Zheng, Haiyong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```
