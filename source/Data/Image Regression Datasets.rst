Image Regression Datasets
++++++++++++++++++++++++++++++++++++++++++++++++++

The datasets for image regression tasks are originally from the MOSAIKS dataset (Rolf et al., 2021) and the SustainBench benchmark (Yeh et al., 2021). We preprocess the data to fit the image regression task settings.
The data can be downloaded from the following DOI link:
`Download Data <https://doi.org/10.6084/m9.figshare.26026798>`_

MOSAIKS Population Density
=============================
This dataset uses daytime remote sensing images as covariables to predict population density at the corresponding locations. The observations were geographically sampled with the uniformly-at-random (UAR) strategy on the earth's surface. The MOSAIKS originally contains 100K population density records with coordinates, but less than half of them can be matched to remote sensing images on the dataset. We apply a log transformation of the labels and add 1 beforehand to avoid dropping zero-valued labels. 
After data cleaning, we get 425637 observations uniformly distributed across the world. 

MOSAIKS Forest Cover
=============================
According to (Rolf et al., 2021), forest in this dataset is defined as vegetation greater than 5 meters in height, 
and measurements of forest cover are given at a raw resolution of roughly 30m by 30m. 
The estimation of forest cover rate was achieved by analysis of multiple spectral bands of remote sensing imagery, 
other than RGB bands used in this dataset. After similar data cleaning and preprocessing step, 
we get 498,106 observations at the global level. 

MOSAIKS Nightlight Luminosity
=============================
Like forest cover rate, nightlight luminosity is also derived from satellite imagery, but not the RGB bands that most computer vision models work on, nor daytime remote sensing images we use as inputs in our benchmark. Specifically, luminosity in this dataset refers to the average radiance at night in 2015, provided by the Visible Infrared Imaging Radiometer Suite (VIIRS). Following the same data preprocess step, we offer 492226 observations of nightlight luminosity with corresponding satellite images.

MOSAIKS Elevation
=============================
Similarly, Satellite RGB bands are used to predict the elevation at the corresponding location. Following the same data preprocess step, we offer 498,115 elevation observations. To align with the settings of MOSAIKS, we did not apply a log transformation on elevation labels. The underlying data behind this dataset mainly comes from the Shuttle Radar Topography Mission (SRTM) at NASA's Jet Propulsion Laboratory (JPL), in addition to other open data projects.

SustainBench Series
=============================
The SustainBench series including 6 datasets: Asset Index, Women BMI, Water Index, Child Mortality Rate, Sanitation Index, and Women Edu.
They were derived from survey data from the Demographic and Health Surveys (DHS) program. 