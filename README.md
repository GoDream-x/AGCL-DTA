# AGCL-DTA 基于注意力和图对比学习的药物-靶标结合亲和力预测研究
The AGCL-DTA is a novel graph neural network model for drug-target interaction prediction.This method utilizes an enhanced graph convolutional neural network (TopoGCN) and an external attention mechanism to capture rich feature information and key internal structures of drug and target molecules. By employing graph contrastive learning, the model learns mutual information in an unsupervised or semi-supervised manner, enhancing generalization and prediction performance. Experiments on two benchmark datasets demonstrate that our model outperforms existing methods.
# Dependency
    python 3.10.6
    numpy 1.26.0
    torch 2.3.0
    torch-geometric 2.3.1
    rdkit 2023.3.3
# Data preparation
1. Unpacking data.zip.
2. The target molecule graphs data is downloaded from https://drive.google.com/open?id=1rqAopf_IaH3jzFkwXObQ4i-6bUUwizCv. Move the downloaded folders to the directory of each dataset.

    * /data/davis/aln
    * /data/davis/pconsc4
    * /data/kiba/aln
    * /data/kiba/pconsc4
   # Running
    python main.py --cuda 0
