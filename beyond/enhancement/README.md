<base target="_blank"/>


# Transformer for Image Enhancement<br>

Here we provide PyTorch implementation and the pre-trained model of our latest version.

## Prerequisites

- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Train/Test
- Download MIT-Adobe-5K-UPE dataset.

- Train our **HT+** model:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model ht --tr_r_enc_head x --tr_r_enc_layers x --name experiment_name --dataset_root <dataset_dir> --batch_size xx --init_port xxxx
```
- Test our **HT+** model
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model ht --tr_r_enc_head x --tr_r_enc_layers x --name experiment_name --dataset_root <dataset_dir> --batch_size xx --init_port xxxx
```

## Apply a pre-trained model
- Download the pre-trained model from [Google Drive](https://drive.google.com/file/d/1vEvs-ddc6ZcJDd0Q8XD2_zZxgZ8CSBht/view?usp=sharing) or [BaiduCloud](https://pan.baidu.com/s/1nz9MKTox6196NLfw_Zlv7Q) (access code: mknj), and put `latest_net_G.pth` in the directory `checkpoints/enhancement_experiment`. Run:
```bash
# Our HT model
CUDA_VISIBLE_DEVICES=0 python test.py --model ht --tr_r_enc_head 2 --tr_r_enc_layers 6 --name enhancement_experiment --dataset_root <dataset_dir> --batch_size xx --init_port xxxx
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
