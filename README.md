<base target="_blank"/>


# Image Harmonization with Transformer **[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Guo_Image_Harmonization_With_Transformer_ICCV_2021_paper.pdf)]**<br>
Zonghui Guo, Dongsheng Guo, Haiyong Zheng, Zhaorui Gu, Bing Zheng<br>


Here we provide PyTorch implementation and the trained model of our framework.

## Prerequisites

- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Train/Test
- Download [iHarmony4](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4) dataset.

- Train our HT model:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model ht --tr_r_enc_head x --tr_r_enc_layers x --name HT_2H9L_allihd --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```
- Test our HT model
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model ht --tr_r_enc_head x --tr_r_enc_layers x --name HT_2H9L_allihd --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```

- Train our D-HT model:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model dht --light_use_mask --tr_r_enc_head 2 --tr_r_enc_layers 9  --tr_i_dec_head 2 --tr_i_dec_layers 9 --tr_l_enc_head 2 --tr_l_enc_layers 9 --tr_l_dec_head 2 --tr_l_dec_layers 9 --name DHT_2H9L_allihd --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```
- Test our D-HT model
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model dht --light_use_mask --tr_r_enc_head 2 --tr_r_enc_layers 9  --tr_i_dec_head 2 --tr_i_dec_layers 9 --tr_l_enc_head 2 --tr_l_enc_layers 9 --tr_l_dec_head 2 --tr_l_dec_layers 9 --name DHT_2H9L_allihd --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```

## Apply a pre-trained model
- Download the pretrained model from [Google Drive](https://drive.google.com/file/d/1rJhObsXP_cQVE4XOPWBzwT6nh1gnMDrR/view?usp=sharing) or [BaiduCloud](https://pan.baidu.com/s/130FikTVedUP_Eu0pYiMc3w) (access code: v56v), and put latest_net_G.pth in the directory checkpoints/HT_2H9L_allihd or checkpoints/DHT_2H9L_allihd . Run:
```bash
# Our HT model
CUDA_VISIBLE_DEVICES=0 python test.py --model ht --tr_r_enc_head x --tr_r_enc_layers x --name HT_2H9L_allihd --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
# Our D-HT model
CUDA_VISIBLE_DEVICES=0 python test.py --model dht --light_use_mask --tr_r_enc_head 2 --tr_r_enc_layers 9  --tr_i_dec_head 2 --tr_i_dec_layers 9 --tr_l_enc_head 2 --tr_l_enc_layers 9 --tr_l_dec_head 2 --tr_l_dec_layers 9 --name DHT_2H9L_allihd --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```
## Evaluation
We provide the code in ih_evaluation.py. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/ih_evaluation.py --dataroot <dataset_dir> --result_root  results/experiment/test_latest/images/ --evaluation_type our --dataset_name ALL
```
## Quantitative Result

<table class="tg">
  <tr>
    <th class="tg-0pky" align="center">Dataset</th>
    <th class="tg-0pky" align="center">Metrics</th>
    <th class="tg-0pky" align="center">Composite</th>
    <th class="tg-0pky" align="center">Ours<br>(HT: 2H9L)</th>
    <th class="tg-0pky" align="center">Ours<br>(D-HT: 2H9L)</th>
  </tr>
  <tr>
    <td class="tg-0pky" align="center">HCOCO</td>
    <td class="tg-0pky" align="center">
        PSNR</br>
        fPSNR</br>
        MSE</br>
        fMSE
    </td>
    <td class="tg-0pky" align="right">
        33.99</br>
        19.86</br>
        69.37</br>
        996.59
    </td>
    <td class="tg-0pky" align="right">
        37.81</br>
        24.18</br>
        21.57</br>
        382.62
    </td>
    <td class="tg-0pky" align="right">
        38.99</br>
        25.52</br>
        15.65</br>
        286.59
    </td>
  </tr>
  <tr>
    <td class="tg-0pky" align="center">HAdobe5k</td>
    <td class="tg-0pky" align="center">
        PSNR</br>
        fPSNR</br>
        MSE</br>
        fMSE
    </td>
    <td class="tg-0pky" align="right">
        28.52</br>
        17.52</br>
        345.54</br>
        2051.61
    </td>
    <td class="tg-0pky" align="right">
        35.95</br>
        25.61</br>
        44.50</br>
        328.33
    </td>
    <td class="tg-0pky" align="right">
        37.05</br>
        26.96</br>
        34.78</br>
        243.02
    </td>
  </tr>
  <tr>
    <td class="tg-0pky" align="center">HFlickr</td>
    <td class="tg-0pky" align="center">
        PSNR</br>
        fPSNR</br>
        MSE</br>
        fMSE
    </td>
    <td class="tg-0pky" align="right">
        28.43</br>
        18.09</br>
        264.35</br>
        1574.37
    </td>
    <td class="tg-0pky" align="right">
        32.16</br>
        22.02</br>
        86.70</br>
        619.50
    </td>
    <td class="tg-0pky" align="right">
        33.55</br>
        23.51</br>
        64.97</br>
        464.72
    </td>
  </tr>
  <tr>
    <td class="tg-0pky" align="center">Hday2night</td>
    <td class="tg-0pky" align="center">
        PSNR</br>
        fPSNR</br>
        MSE</br>
        fMSE
    </td>
    <td class="tg-0pky" align="right">
        34.36</br>
        19.14</br>
        109.65</br>
        1409.98
    </td>
    <td class="tg-0pky" align="right">
        35.97</br>
        21.22</br>
        61.97</br>
        878.24
    </td>
    <td class="tg-0pky" align="right">
        37.03</br>
        22.34</br>
        46.60</br>
        627.14
    </td>
  </tr>
  
  <tr>
    <td class="tg-0pky" align="center">ALL</td>
    <td class="tg-0pky" align="center">
        PSNR</br>
        fPSNR</br>
        MSE</br>
        fMSE
    </td>
    <td class="tg-0pky" align="right">
        31.78</br>
        18.97</br>
        172.47</br>
        1376.42
    </td>
    <td class="tg-0pky" align="right">
        36.60</br>
        24.30</br>
        36.27</br>
        402.17
    </td>
    <td class="tg-0pky" align="right">
        37.78</br>
        25.66</br>
        27.30</br>
        299.92
    </td>
  </tr>

</table>


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

# Acknowledgement
For some of the data modules and model functions used in this source code, we need to acknowledge the repo of [DoveNet](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4/tree/master/DoveNet), [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [IntrinsicHarmony](https://github.com/zhenglab/IntrinsicHarmony). 
