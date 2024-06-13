# TorchSpatial: A Location Encoding Framework and Benchmark for Spatial Representation Learning

![TorchSpatial Overall Framework](figs/overall_framework.png)



🚧 Constructing...


## Data Download Instructions
The data can be downloaded from the following DOI link:
[Download Data](https://doi.org/10.6084/m9.figshare.26026798)

Data should be organized following the ..

## Code Execution
The example bash files for running the codes can be found in main/run_bash folder



## Image Datasets

### NABirds
Another image dataset about North American bird species constructed by Mac Aodha et al. (2019) based on the NABirds dataset (Van Horn et al., 2015) in which the location metadata were also simulated from the eBrid dataset (Sullivan et al., 2009).

**Download**: [NABirds Dataset](https://dl.allaboutbirds.org/nabirds)

### iNat2017
The species recognition dataset used in the iNaturalist 2017 challenges (Van Horn et al., 2018) with 5089 unique categories.

**Validation Split**: [iNat2017 Validation Split](https://github.com/visipedia/inat_comp/blob/master/2017/README.md)

### iNat2018
The species recognition dataset used in the iNaturalist 2018 challenges (Van Horn et al., 2018) with 8142 unique categories.

**Validation Split**: [iNat2018 Validation Split](https://github.com/visipedia/inat_comp/tree/master/2018)

### YFCC
Yahoo Flickr Creative Commons 100M dataset (YFCC100M-GEO100 dataset), which is a set of geo-tagged Flickr photos collected by Yahoo! Research. Here, we denote this dataset as YFCC. YFCC has been used in Tang et al. (2015); Mac Aodha et al. (2019) for geo-aware image classification.

**Download**: [YFCC Dataset](https://github.com/visipedia/fg_geo)

### fMoW
We use the Functional Map of the World dataset (denoted as fMoW) (Klocek et al., 2019) as one representative remote sensing (RS) image classification dataset. The fMoW dataset contains about 363K training and 53K validation remote sensing images which are classified into 62 different land use types. They are 4-band or 8-band multispectral remote sensing images. 4-band images are collected from the QuickBird2 or GeoEye-1 satellite systems while 8-band images are from WorldView-2 or WorldView-3. We use the fMoWrgb version of fMoW dataset which are JPEG compressed version of these remote sensing images with only the RGB bands. The reason we pick fMoM is that 1) the fMoW dataset contains RS images with diverse land use types collected all over the world (see Figure 8c and 8d); 2) it is a large RS image dataset with location metadata available. In contrast, the UC Merced dataset (Yang and Newsam, 2010) consist of RS images collected from only 20 US cities. The EuroSAT dataset (Helber et al., 2019) contained RS images collected from 30 European countries. And the location metadata of the RS images from these two datasets are not publicly available. Global coverage of the RS images is important in our experiment since we focus on studying how the map projection distortion problem and sphericalto-Euclidean distance approximation error can be solved by Sphere2Vec on a global scale geospatial problem. The reason we use the RGB version is that this dataset version has an existing pretrained image encoder – MoCo-V2+TP (Ayush et al., 2020) available to use. We do not need to train our own remote sensing image encoder.

**Download**: [fMoW Dataset](https://github.com/fMoW/dataset)



### Reference
If you find our work useful in your research please consider citing [our ISPRS PHOTO 2023 paper](https://www.researchgate.net/publication/371964548_Sphere2Vec_A_General-Purpose_Location_Representation_Learning_over_a_Spherical_Surface_for_Large-Scale_Geospatial_Predictions).  
```
@article{mai2023sphere2vec,
  title={Sphere2Vec: A General-Purpose Location Representation Learning over a Spherical Surface for Large-Scale Geospatial Predictions},
  author={Mai, Gengchen and Xuan, Yao and Zuo, Wenyun and He, Yutong and Song, Jiaming and Ermon, Stefano and Janowicz, Krzysztof and Lao, Ni},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  year={2023},
  vol={202},
  pages={439-462},
  publisher={Elsevier}
}
```
If you use grid location encoder, please also cite [our ICLR 2020 paper](https://openreview.net/forum?id=rJljdh4KDH) and [our IJGIS 2022 paper](https://www.tandfonline.com/doi/full/10.1080/13658816.2021.2004602):
```
@inproceedings{mai2020space2vec,
  title={Multi-Scale Representation Learning for Spatial Feature Distributions using Grid Cells},
  author={Mai, Gengchen and Janowicz, Krzysztof and Yan, Bo and Zhu, Rui and Cai, Ling and Lao, Ni},
  booktitle={International Conference on Learning Representations},
  year={2020},
  organization={openreview}
}

@article{mai2022review,
  title={A review of location encoding for GeoAI: methods and applications},
  author={Mai, Gengchen and Janowicz, Krzysztof and Hu, Yingjie and Gao, Song and Yan, Bo and Zhu, Rui and Cai, Ling and Lao, Ni},
  journal={International Journal of Geographical Information Science},
  volume={36},
  number={4},
  pages={639--673},
  year={2022},
  publisher={Taylor \& Francis}
}
```
If you use the unsupervised learning function, please also cite [our ICML 2023 paper](https://gengchenmai.github.io/csp-website/). Please refer to [our CSP webite](https://gengchenmai.github.io/csp-website/) for more detailed information.  
```
@inproceedings{mai2023csp,
  title={CSP: Self-Supervised Contrastive Spatial Pre-Training for Geospatial-Visual Representations},
  author={Mai, Gengchen and Lao, Ni and He, Yutong and Song, Jiaming and Ermon, Stefano},
  booktitle={International Conference on Machine Learning},
  year={2023},
  organization={PMLR}
}
```
