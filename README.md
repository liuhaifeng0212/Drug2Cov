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
  author    = {Haifeng Liu and
               Hongfei Lin and
               Chen Shen and
               Liang Yang and
               Yuan Lin and
               Bo Xu and
               Zhihao Yang and
               Jian Wang and
               Yuanyuan Sun},
  title     = {Drug Repositioning for SARS-CoV-2 Based on Graph Neural Network},
  booktitle = {{IEEE} International Conference on Bioinformatics and Biomedicine,
               {BIBM} 2020, Virtual Event, South Korea, December 16-19, 2020},
  pages     = {319--322},
  publisher = {{IEEE}},
  year      = {2020},
  url       = {https://doi.org/10.1109/BIBM49941.2020.9313236},
  doi       = {10.1109/BIBM49941.2020.9313236},
  timestamp = {Thu, 02 Sep 2021 17:53:02 +0200},
  biburl    = {https://dblp.org/rec/conf/bibm/LiuLSYLXYWS20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

algorithms of paper
@inproceedings{liu2021self,
  author    = {Haifeng Liu and
               Hongfei Lin and
               Chen Shen and
               Zhihao Yang and
               Jian Wang and
               Liang Yang},
  title     = {Self-Supervised Learning with Heterogeneous Graph Neural Network for
               {COVID-19} Drug Recommendation},
  booktitle = {{IEEE} International Conference on Bioinformatics and Biomedicine,
               {BIBM} 2021, Houston, TX, USA, December 9-12, 2021},
  pages     = {1412--1417},
  publisher = {{IEEE}},
  year      = {2021},
  url       = {https://doi.org/10.1109/BIBM52615.2021.9669340},
  doi       = {10.1109/BIBM52615.2021.9669340},
  timestamp = {Tue, 18 Jan 2022 13:10:12 +0100},
  biburl    = {https://dblp.org/rec/conf/bibm/LiuLSYWY21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
 ```

### Author contact:
Email: liuhaifeng0212@qq.com
