# Variational Interaction Information Maximization for Cross-domain Disentanglement

<img src="imgs/IIAE.PNG" alt="IIAE" style="zoom:100%;" />



#### 1. Resolve dependencies

We recommend to use conda environment. Please run

```sh
conda env create --file IIAE.yml
```

#### 2. Prepare datasets and pre-trained models

Download [MNIST-CDCB](https://github.com/agonzgarc/cross-domain-disen), [Facades](https://github.com/phillipi/pix2pix), and [Maps](https://github.com/phillipi/pix2pix) datasets.

Download **Sketchy Extended** dataset by following [SEM-PCYC](https://github.com/AnjanDutta/sem-pcyc).

Lastly, download **Cars** dataset and  **VGG features** of Skethcy Extended dataset from this [link](http://doi.org/10.5281/zenodo.3890097).

Note that we extracted VGG features from the pretrained model provided by [SEM-PCYC](https://github.com/AnjanDutta/sem-pcyc).

Once you download all those datasets, please make sure you locate them under <root_path>/dataset/.

EX) <root_path>/dataset/MNIST-CDCB

​      <root_path>/dataset/cars

​      <root_path>/dataset/Sketchy

​      <root_path>/dataset/SketchyVGG

Now you are ready to train Interaction Information AutoEncoders (IIAEs).

You can also find the pretrained  IIAEs for those five different datasets from this [link](http://doi.org/10.5281/zenodo.3890097).

#### 3. Train and evaluate IIAE on Sketchy dataset

Train IIAE with Skethcy dataset.

```sh
python zs_cross_vae.py --mode train --root_path <root_path> --output_dir <root_path>/SUBMISSION/Sketchy
```

Evaluate IIAE with Skethcy dataset.

```sh
python zs_cross_vae.py --mode test --root_path <root_path> --output_dir <root_path>/SUBMISSION/Sketchy
```

#### 4. Train and evaluate IIAE on MNIST-CDCB, Cars, Facades, and Maps

Train IIAE with any of MNIST-CDCB, Cars, Facades, or Maps.

```sh
python run_cross_vae.py --mode train --dataset_name <dataset_name> --root_path <root_path> --output_dir <root_path>/SUBMISSION/<dataset_name>
```

Extract features and evaluate cross-domain retrieval (for MNIST-CDCB, Facades, and Maps).

```sh
python run_cross_vae.py --mode features --dataset_name <dataset_name> --root_path <root_path> --output_dir <root_path>/SUBMISSION/<dataset_name>
python feat_dist_full.py --dataset_name <dataset_name> --root_path <root_path> --output_dir <root_path>/SUBMISSION/<dataset_name>
```

Generate samples of cross-domain image translation. (for MNIST-CDCB and Cars)

```sh
python run_cross_vae.py --mode test --dataset_name <dataset_name> --root_path <root_path> --output_dir <root_path>/SUBMISSION/<dataset_name>
```



#### 5. Results

<img src="imgs/table1.PNG" alt="table1" style="zoom:50%;" />



<img src="imgs/table2.PNG" alt="table2" style="zoom:50%;" />



#### 6. Notes

Our code is based on [Cross-domain Disentantanglement Network](https://github.com/agonzgarc/cross-domain-disen).

The VGG features of Sketchy dataset are extracted using pretrained VGG model provided by [SEM-PCYC](https://github.com/AnjanDutta/sem-pcyc).

Raw files of Cars dataset can be found [here](http://www-personal.umich.edu/~reedscot/files/).

#### 7. References

If this repository helps you in your academic research, you are encouraged to cite our paper. Here is an example bibtex:

```bibtex
@inproceedings{HwangEtal.NeurIPS2020,
    author = {Hwang, HyeongJoo and Kim, Geon-Hyeong and Hong, Seunghoon and Kim, Kee-Eung},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
    pages = {22479--22491},
    publisher = {Curran Associates, Inc.},
    title = {Variational Interaction Information Maximization for Cross-domain Disentanglement},
    url = {https://proceedings.neurips.cc/paper/2020/file/fe663a72b27bdc613873fbbb512f6f67-Paper.pdf},
    volume = {33},
    year = {2020}
}
```
