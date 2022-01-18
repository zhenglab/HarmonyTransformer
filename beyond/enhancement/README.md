<base target="_blank"/>


# Image Harmonization with Transformer **[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Guo_Image_Harmonization_With_Transformer_ICCV_2021_paper.pdf)]**<br>
Zonghui Guo, Dongsheng Guo, Haiyong Zheng, Zhaorui Gu, Bing Zheng<br>

Here we provide PyTorch implementation and the trained model of image enhancement framework.

## Prerequisites

- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Train/Test
- Download MIT-Adobe-5K-UPE dataset.

- Train our HT model (FC-TRE-DeCNN):
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model enhancement --tr_r_enc_head x --tr_r_enc_layers x --name experiment_name --dataset_root <dataset_dir> --batch_size xx --init_port xxxx
```
- Test our HT model
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model enhancement --tr_r_enc_head x --tr_r_enc_layers x --name experiment_name --dataset_root <dataset_dir> --batch_size xx --init_port xxxx
```

## Apply a pre-trained model
- Download the pretrained model from [Google Drive](https://drive.google.com/file/d/1rJhObsXP_cQVE4XOPWBzwT6nh1gnMDrR/view?usp=sharing) or [BaiduCloud](https://pan.baidu.com/s/130FikTVedUP_Eu0pYiMc3w) (access code: v56v), and put latest_net_G.pth in the directory checkpoints/enhancement_pretrained. Run:
```bash
# Our FCHT model
CUDA_VISIBLE_DEVICES=0 python test.py --model enhancement --tr_r_enc_head 2 --tr_r_enc_layers 6 --name enhancement_pretrained --dataset_root <dataset_dir> --batch_size xx --init_port xxxx
```

# Bibtex
If you use this code for your research, please cite our papers.


```
@InProceedings{Guo_2021_ICCV,
    author    = {Guo, Zonghui and Guo, Dongsheng and Zheng, Haiyong and Gu, Zhaorui and Zheng, Bing and Dong, Junyu},
    title     = {Image Harmonization With Transformer},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {14870-14879}
}
```
