<base target="_blank"/>

# Transformer for Image Harmonization and Beyond **[[Paper](https://ieeexplore.ieee.org/abstract/document/9893399)]**<br>
Zonghui Guo, Zhaorui Gu, Bing Zheng, Junyu Dong, Haiyong Zheng<br>
IEEE Transactions on Pattern Analysis and Machine Intelligence<br>

Here we provide the PyTorch implementation and pre-trained model of our latest version, if you require the code of our previous ICCV version (**["Image Harmonization With Transformer"](https://openaccess.thecvf.com/content/ICCV2021/papers/Guo_Image_Harmonization_With_Transformer_ICCV_2021_paper.pdf)**), please click the **[released version](https://github.com/zhenglab/HarmonyTransformer/releases/tag/v1.0)**.

## Prerequisites

- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Train/Test
- Download [iHarmony4](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4) dataset.

- Train our HT+ model (**FC-TRE-DeCNN**):
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model ht --tr_r_enc_head x --tr_r_enc_layers x --name experiment_name --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```
- Test our HT+ model (**FC-TRE-DeCNN**):
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model ht --tr_r_enc_head x --tr_r_enc_layers x --name experiment_name --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```

- Train our **DHT+** model:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model dht --light_use_mask --tr_r_enc_head x --tr_r_enc_layers x  --tr_i_dec_head x --tr_i_dec_layers x --tr_l_dec_head x --tr_l_dec_layers x --name DHT_experiment_name --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```
- Test our **DHT+** model:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model dht --light_use_mask --tr_r_enc_head x --tr_r_enc_layers x  --tr_i_dec_head x --tr_i_dec_layers x --tr_l_dec_head x --tr_l_dec_layers x --name DHT_experiment_name --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```

## Apply a pre-trained model
- Download pre-trained models from [Google Drive](https://drive.google.com/file/d/1uQqveBSUfTmvA4FEWC_stAyf2oMS4UHC/view?usp=sharing) or [BaiduCloud](https://pan.baidu.com/s/1KxN0WYwaLBP1THatuzhq1A) (access code: vmrg), and put `latest_net_G.pth` in the directory `checkpoints/HT_2H9L_allihd` or `checkpoints/DHT_2H9L_allihd`. Run:
```bash
# Our HT model
CUDA_VISIBLE_DEVICES=0 python test.py --model ht --tr_r_enc_head 2 --tr_r_enc_layers 9 --name HT_2H9L_allihd --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
# Our CNN-DHT model
CUDA_VISIBLE_DEVICES=0 python test.py --model dht --light_use_mask --tr_r_enc_head 2 --tr_r_enc_layers 9  --tr_i_dec_head 2 --tr_i_dec_layers 9 --tr_l_dec_head 2 --tr_l_dec_layers 9 --name DHT_2H9L_allihd --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```
## Evaluation
We provide the code in `ih_evaluation.py`. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/ih_evaluation.py --dataroot <dataset_dir> --result_root  results/experiment_name/test_latest/images/ --evaluation_type our --dataset_name ALL
```

## Real composite image harmonnization
More compared results can be found at [Google Drive](https://drive.google.com/file/d/1qkLdvS8rTng4bxWKSFjtfPgQ5SK2OvGa/view?usp=sharing) or [BaduCloud](https://pan.baidu.com/s/1mf4h4jOrVO9jFEthYsHyzw) (access code: n37b).


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

@InProceedings{Guo_2021_ICCV,
    author    = {Guo, Zonghui and Guo, Dongsheng and Zheng, Haiyong and Gu, Zhaorui and Zheng, Bing and Dong, Junyu},
    title     = {Image Harmonization With Transformer},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {14870-14879}
}
```

# Acknowledgement
For some of the data modules and model functions used in this source code, we need to acknowledge the repositories of [DoveNet](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4/tree/master/DoveNet), [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [
SpiralNet](https://github.com/zhenglab/spiralnet) and [IntrinsicHarmony](https://github.com/zhenglab/IntrinsicHarmony). 
