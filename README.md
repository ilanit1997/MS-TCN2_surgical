# MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation (TPAMI 2020)

This repository provides a PyTorch implementation of the paper [MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation](https://arxiv.org/pdf/2006.09220.pdf).

## Environment
Python3, pytorch

## Tradeoff exploration:

#### How can we use learned weights to control the tradeoff between global and local history? 

## Difference from original model:
* We added additive attention to model.py using einstein summation
* Combine train, val, predict and eval into one module named - train_predict_eval.py
* The different experiments can be seen in - train_predict_eval.sh
* Adding logs to ClearML
* train-test-val split is unique to our dataset 


## Cite:
```BibTeX
@article{li2020ms,
   author={Shi-Jie Li and Yazan AbuFarha and Yun Liu and Ming-Ming Cheng and Juergen Gall},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
    title={MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation}, 
    year={2020},
    volume={},
    number={},
    pages={1-1},
    doi={10.1109/TPAMI.2020.3021756},
}

@inproceedings{farha2019ms,
  title={Ms-tcn: Multi-stage temporal convolutional network for action segmentation},
  author={Farha, Yazan Abu and Gall, Jurgen},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3575--3584},
  year={2019}
}

```
