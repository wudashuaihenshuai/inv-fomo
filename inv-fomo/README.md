# Open World Object Detection via Causal Attribute Learning

## Requirements and Installation

```bash
conda create --name inv-fomo python==3.7.16
conda activate inv-fomo
pip install -r requirements.txt
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

## Dataset Setup
Dataset setup instruction is in [DATASET_SETUP.md](DATASET_SETUP.md).



### RUN
bash cit_setup.sh             

bash run_rwd_test.sh

bash run_rwd_test_t2.sh

bash viz_test.sh  


### For other detailed settings, please refer to FOMO:

```
@InProceedings{Zohar_2023_CVPR,
    author    = {Zohar, Orr and Wang, Kuan-Chieh and Yeung, Serena},
    title     = {PROB: Probabilistic Objectness for Open World Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {11444-11453}
}
```

code: https://github.com/orrzohar/FOMO.git
