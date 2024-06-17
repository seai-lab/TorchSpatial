Image Classification Datasets
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The data can be downloaded from the following DOI link:
[Download Data](https://doi.org/10.6084/m9.figshare.26026798)

BirdSnap
==========
An image dataset about bird species based on BirdSnap dataset (Berg et al., 2014) with location annotations by (Aodha et al., 2019). It consists of 19576 images of 500 bird species that are commonly found in North America.  This dataset and the other two following are widely used by multiple studies (Aodha et al., 2019; Mai et al., 2020, 2023) to demonstrate location encoder's capacity to significantly increase the fine-grained species classification accuracy. 

BirdSnap†
==========
An enriched BirdSnap dataset constructed by (Aodha et al., 2019) by simulating locations, dates, and photographers from the eBrid dataset (Sullivan et al., 2009), containing 43470 images of 500 categories. 

NABirds†
==========
Another image dataset about North American bird species constructed by (Aodha et al., 2019) 
based on the NABirds dataset (Van Horn et al., 2015), the location metadata were also simulated from the eBrid dataset (Sullivan et al., 2009). It contains 23699 images of 555 bird species categories. 

iNat2017
==========
The worldwide species recognition dataset used in the iNaturalist 2017 challenges (Van Horn et al., 2018) with 675170 images and 5089 unique categories. We add the location information retroactively provided by iNaturalist 2021. Although its spatial distribution focuses on North America and Europe, it still covers the entire globe, which makes it one of the most spatially extensive and species-rich image dataset known to us.

iNat2018
==========
The worldwide species recognition dataset used in the iNaturalist 2018 challenges (Van Horn et al., 2018) with 461939 images and 8142 unique categories. Although the original competition didn't provide coordinates, we add them to our benchmark as additional information from the same data source of iNaturalist 2021. It has a similar spatial distribution with iNat2017, covering all continents. We choose these two datasets to evaluate location encoder's capacity to improve fine-grained species classification performance at the global level.

YFCC
==========
YFCC100M-GEO100 dataset, an image dataset derived from Yahoo Flickr Creative Commons 100M dataset and was annotated by (Tang et al., 2015), containing 88986 images over 100 everyday object categories with location annotations. Here, we denote this dataset as YFCC. YFCC is a comprehensive public dataset with images across the United States. Despite the relatively limited geographic coverage, we employ this dataset to measure location encoder's capacity for multifaceted image classification in addition to domain-specific image classification.

fMoW
==========
Functional Map of the World dataset (denoted as fMoW) (Christie et al., 2018) is a remote sensing (RS) image classification dataset, containing RS images with diverse land use types collected all over the world. It is composed of about 363K training and 53K validation remote sensing images which are classified into 62 different land use types. We use the fMoWrgb version of the fMoW dataset which are JPEG compressed version of these RS images with only the RGB bands.