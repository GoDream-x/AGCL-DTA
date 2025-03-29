# MVHGNN 基于多视图对比和同源特征的药物靶标结合亲和力预测研究
The MVHGNN is a novel graph neural network model for drug-target interaction prediction. MVHGNN employs a multi-view contrastive learning framework, using enhanced subgraph topology graph convolutional network (ESTGCN) and graph isomorphism network (GIN) as encoders to capture drug topological structures and protein hierarchical structures. Homology features integrate drug-drug and protein-protein multi-level features, enhancing intra-view feature utilization. Additionally, GCN extracts global topology in the drug-target affinity view. A cross-contrastive learning strategy maximizes mutual information across views, improving representation consistency and cross-view synergy. Experiments on benchmark datasets, particularly Davis, demonstrate MVHGNN’s superior performance (MES: 0.166, Corrected R²: 0.794), out-performing state-of-the-art methods. 

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
