### Basic Information:
This code is released for the paper: 

Haifeng Liu, Hongfei Lin, Chen Shen, et al. Self-Supervised Learning with Heterogeneous Graph Neural Network for COVID-19 Drug Recommendation. Accepted by BIBM2021. 

You can also preview our paper from the ieee(https://ieeexplore.ieee.org/document/9669340).

### Usage:
1. Environment: I have tested this code with python3.6, tensorflow-gpu-1.12.0 
2. Download the covid19 data from this [link](https://github.com/liuhaifeng0212/Drug2Cov), and unzip the directories in covid data to the dataset directory of your local clone repository.
3. cd the drug2cov directory and execute the command `python entry.py --data_name=<data_name> --model_name=<model_name> --gpu=<gpu id>` then you can see it works, if you have any available gpu device, you can specify the gpu id, or you can just ignore the gpu id. 

Following is an example:
`python entry.py --data_name=covid --model_name=drug2cov`

### Citation:
```
The dataset covid19 we use from this paper:
@inproceedings{liu2020drug,
  title={Drug Repositioning for SARS-CoV-2 Based on Graph Neural Network},
  author={Liu, Haifeng and Lin, Hongfei and Shen, Chen and Yang, Liang and Lin, Yuan and Xu, Bo and Yang, Zhihao and Wang, Jian and Sun, Yuanyuan},
  booktitle={2020 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={319--322},
  year={2020},
  organization={IEEE}
}

algorithms of paper
@inproceedings{liu2021self,
  title={Self-Supervised Learning with Heterogeneous Graph Neural Network for COVID-19 Drug Recommendation},
  author={Liu, Haifeng and Lin, Hongfei and Shen, Chen and Yang, Zhihao and Wang, Jian and Yang, Liang},
  booktitle={2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={1412--1417},
  year={2021},
  organization={IEEE}
}
 ```

### Author contact:
Email: liuhaifeng0212@qq.com
