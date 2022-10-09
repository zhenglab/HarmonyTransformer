<base target="_blank"/>


# Transformer for Portrait Relighting<br>

Here we provide PyTorch implementation and the pre-trained model of our latest version.

## Prerequisites

- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Train/Test
- Download DPR dataset.

- Train our **DHT+** model:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model dht --tr_r_enc_head 2 --tr_r_enc_layers 9  --tr_i_dec_head 2 --tr_i_dec_layers 9 --tr_l_dec_head 2 --tr_l_dec_layers 9 --name experiment_name --dataset_root <dataset_dir> --batch_size xx --init_port xxxx
```
- Test our **DHT+** model
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model dht --tr_r_enc_head 2 --tr_r_enc_layers 9  --tr_i_dec_head 2 --tr_i_dec_layers 9 --tr_l_dec_head 2 --tr_l_dec_layers 9 --name experiment_name --dataset_root <dataset_dir> --batch_size xx --init_port xxxx
```

## Apply a pre-trained model
- Download the pre-trained model from [Google Drive](https://drive.google.com/file/d/1ISXB7l71ox2efAwRrg3on8Oe_f_Zmapz/view?usp=sharing) or [BaiduCloud](https://pan.baidu.com/s/1eFKMOrqvnQv0An_ztl1Nng) (access code: 1o5g), and put `latest_net_G.pth` in the directory `checkpoints/relighting_experiment`. Run:
```bash
# SH-based relighting
CUDA_VISIBLE_DEVICES=0 python test.py --model dht --name relighting_experiment  --relighting_action relighting  --dataset_root <dataset_dir> --dataset_name DPR --dataset_mode dpr  --batch_size 1 --init_port xxxx
#Image-based relighting
CUDA_VISIBLE_DEVICES=0 python test.py --model dht --name relighting_experiment  --relighting_action transfer  --dataset_root <dataset_dir> --dataset_name DPR --dataset_mode dprtransfer  --batch_size 1 --init_port xxxx
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
