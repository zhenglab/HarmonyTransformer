<base target="_blank"/>

# Transformer for White-Balance Editing<br>

Here we provide PyTorch implementation and the pre-trained model of our latest version.

## Prerequisites

- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Train/Test
- Download [Rendered WB dataset](https://cvil.eecs.yorku.ca/projects/public_html/sRGB_WB_correction/dataset.html) dataset.

- Train our WB-HT+ model (**WB-HT+**):
```bash
 CUDA_VISIBLE_DEVICES=0 python train.py --model ht --tr_r_enc_head 2 --tr_r_enc_layers 6 --tr_r_dec_head 2 --tr_r_dec_layers 6  --use_patch --ksize 4 --stride 4 --name experimane_name --dataset_root <dataset_dir> --batch_size xx --init_port xxxx
```
- Test our WB-HT+ model (**WB-HT+**):
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model ht --tr_r_enc_head 2 --tr_r_enc_layers 6 --tr_r_dec_head 2 --tr_r_dec_layers 6  --use_patch --ksize 4 --stride 4 --name experiment_name --dataset_root <dataset_dir> --batch_size xx --init_port xxxx
```

## Apply a pre-trained model
- Download pre-trained models from [Google Drive](https://drive.google.com/file/d/1wMQBjh5q6XyWnsYik3cH1UjEEpGsSfdh/view?usp=sharing) or [BaiduCloud](https://pan.baidu.com/s/17NFJDP_zCkcEl7LH_r74yQ) (access code: 46wg), and put `latest_net_G.pth` in the directory `checkpoints/HT_2H6L`. Run:
```bash
# Our HT model
CUDA_VISIBLE_DEVICES=0 python test.py --model ht --tr_r_enc_head 2 --tr_r_enc_layers 6 --tr_r_dec_head 2 --tr_r_dec_layers 6  --use_patch --ksize 4 --stride 4 --name HT_2H6L --dataset_root <dataset_dir> --batch_size xx --init_port xxxx
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

# Acknowledgement
For some of the data modules and model functions used in this source code, we need to acknowledge the repositories of [Deep_White_Balance](https://github.com/mahmoudnafifi/Deep_White_Balance). 
