# OneRing: A Simple Method for Source-free Open-partial Domain Adaptation  	

### _Shiqi Yang, Yaxing Wang, Kai Wang, Shangling Jui and Joost van de Weijer_

_**Keyworkds: Open-set Recognition; Open-set Single Domain Generalization; Source-free Universal/Open-partial Domain Adaptation**_

------------
Code for our paper **'OneRing: A Simple Method for Source-free Open-partial Domain Adaptation'** 

[[project]](https://sites.google.com/view/one-ring)[[arxiv]](https://arxiv.org/abs/2206.03600)


## Demo for our **OneRing** classifier

**Trained on 3 known categories.**

![](./toy_CE_loss.gif)

--------------
### Training for open-set single domain generalization and source-free open-partial domain adaptation

(**Attention**: Codes are based on **pytorch 1.3** with cuda 10.0, **please ensure the same pytorch version for reproducing**)

1. Download datasets and change the corresponding path in /data/*.txt
2. Training

- Run the following command for the whole training on **Office-31** under ***open-set single domain generalization***, the model will be only trained on source and directly evaluated on target domain.
> python train_office31_ossdg.py --gpu 0

- Run the following command for the whole training on **Office-Home** and **VisDA** under ***source-free open-partial domain adaptation***, the model will be first trained on source and then adapt to target domain without source data. (*lpa refers to our [AaD](https://openreview.net/forum?id=ZlCpRiZN7n).*)
> sh officehome_sfunda.sh
> 
> python train_visda_sfunda.py --gpu 0 --save_model --lpa


(**We provide the model weight after adaptation for SF-OPDA on Office-Home and VisDA, check the [link](https://drive.google.com/drive/folders/1_Kf5NivEspZ4THMx2KTT4EVhzl2HY68J?usp=sharing) here.**)


**You can check the old arxiv version for results of other tasks.**

### **Reference**

> @article{yang2022one,\
  title={One Ring to Bring Them All: Towards Open-Set Recognition under Domain Shift},\
  author={Yang, Shiqi and Wang, Yaxing and Wang, Kai and Jui, Shangling and van de Weijer, Joost},\
  journal={arXiv preprint arXiv:2206.03600},\
  year={2022}\
}
