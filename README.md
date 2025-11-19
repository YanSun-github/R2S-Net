# R^2S: Recognize-then-Refine-Spot network


![model_overview](./figures/framework.png)



## Results
We compare our method against others on two benchmark datasets, i.e., [CAS(ME)<sup>2</sup>](http://fu.psych.ac.cn/CASME/cas(me)2-en.php) and [SAMM-LV](http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php) in terms of STRS、F1-Score Spotting、F1-Score Analysis:

![model_results](./figures/result.png)

## Experiment environment 
OS: Ubuntu 20.04.4 LTS 

Python: 3.8

Pytorch: 1.10.1

CUDA: 10.2, cudnn: 7.6.5

GPU: NVIDIA GeForce RTX 2080 Ti

## Getting started
1. Clone this repository
```shell
$ git clone git@github.com:YanSun-github/R2S-Net.git
$ cd RRSN-MEA
```

2. Prepare environment

```shell
$ conda create -n env_name python=3.8
$ conda activate env_name
$ pip install -r requirements.txt
```

3. Download features

 

4. Training and Inference

Set `SUB_LIST`, 
`OUTPUT` (dir for saving ckpts, log and results)
and `DATASET` ( ["samm" | "cas(me)^2"] )  in [pipeline.sh], then run:
```shell
$ bash pipeline.sh
```

**We also provide ckpts, logs, etc.** to reproduce the results in the paper, please download [ckpt.tar.gz]().

 

 
## Citation
If you feel this project helpful to your research, please cite our work.
```
 
```

##### You may open an issue or email me at xxx if you have any inquiries or issues.
