# Optimal Transport Kernel

The repository implements the Optimal Transport Kernel (OTK) described in the following paper

>Grégoire Mialon*, Dexiong Chen*, Alexandre d'Aspremont, Julien Mairal.
[An Optimal Transport Kernel for Feature Aggregation and its Relationship to Attention][1]. preprint arXiv. 2020.
<br/>*Equal contribution

## A short description about the module

The principal module is implemented in `otk/layers.py` as `OTKernel`. It takes a sequence or image tensor as input, and performs an adaptive pooling (attention + pooling) based on optimal transport. Here is an example
```python
import torch
from otk.layers import OTKernel

n_dim = 128
# create an OTK layer with single reference and number of supports=10
otk_layer = OTKernel(in_dim=n_dim, out_size=10, heads=1)
# create 2 batches of sequences of L=100 and dim=128
input = torch.rand(2, 100, n_dim)
# each output sequence has L=10 and dim=128
output = otk_layer(input) # 2 x 10 x 128
```
The implemented layer can be trained in either unsupervised (with K-means) or supervised (like the multi-head self-attention module) fashions. See more details in our [paper][1].

## Installation

We strongly recommend users to use [miniconda][2] to install the following packages (link to [pytorch][3])
```
python=3.6
numpy
scipy
scikit-learn
pytorch=1.4.0
pandas
```
Then run
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Experiments

We provide here the commands to reproduce a part of the results in our paper.

#### Reproducing results for SCOP 1.75

To reproduce the results in Table 2, run the following commands.

* **Data preparation**
    ```bash
    cd data
    bash get_scop.sh
    ```

* **Unsupervised learning with OTK for SCOP 1.75**
    ```bash
    cd experiments
    python scop175_unsup.py --n-filters 512 --out-size 100 --eps 0.5
    ```

* **Supervised learning with OTK for SCOP 1.75**
    ```bash
    cd experiments
    # OTK with one reference
    python scop175_sup.py --n-filters 512 --heads 1 --out-size 50 --alternating
    # OTK with multiple references
    python scop175_sup.py --n-filters 512 --heads 5 --out-size 10 --alternating
    ```

#### Reproducing results for DeepSEA

To reproduce the results (auROC=0.936, auPRC=0.360) in Table 3, run the following commands.

* **Data preparation**
    ```bash
    cd data
    bash get_deepsea.sh
    ```

* **Evaluating our pretrained model**

    Download our [pretrained model][4] to `./logs_deepsea` and then run
    ```bash
    cd experiments
    python eval_deepsea.py --sigma 1.0 --heads 1 --out-size 64 --hidden-layer --position-encoding gaussian --weight-decay 1e-06 --position-sigma 0.1 --outdir ../logs_deepsea --max-iter 30 --filter-size 16 --hidden-size 1536
    ```

* **Training and Evaluating a new model**

    First train a model with the following commands
    ```bash
    cd experiments
    python train_deepsea.py --sigma 1.0 --heads 1 --out-size 64 --hidden-layer --position-encoding gaussian --weight-decay 1e-06 --position-sigma 0.1 --outdir ../logs_deepsea --max-iter 30 --filter-size 16 --hidden-size 1536
    ```
    Once the model is trained, run
    ```bash
    python eval_deepsea.py --sigma 1.0 --heads 1 --out-size 64 --hidden-layer --position-encoding gaussian --weight-decay 1e-06 --position-sigma 0.1 --outdir ../logs_deepsea --max-iter 30 --filter-size 16 --hidden-size 1536
    ```


[1]: http://arxiv.org/abs/2006
[2]: https://docs.conda.io/en/latest/miniconda.html
[3]: https://pytorch.org
[4]: http://pascal.inrialpes.fr/data2/dchen/pretrained/otk_checkpoint.zip
