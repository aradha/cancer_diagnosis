# cancer_diagnosis

This repository contains code for the paper "Machine Learning for Nuclear Mechano-Morphometric Biomarkers in Cancer Diagnosis" appearing in Nature: Scientific Reports.


# Segmentation

The file refined_segmentation.py contains code for our technique to segment out nuclei from images.  The technique is very conservative and may be throwing away too many nuclei depending on your needs.  You can run the file with:

python refined_segmentation.py [fname]

The resulting output will be two files in the same directory with prefix [fname] with one followed by _crops.hdf5 ad one followed by _patch_crops.hdf5.  The former contains 128 x 128 nuclei crops and the latter will contain 32 x 32 patch crops.

# Classification

The files linear_main.py and main.py can be used to respectively run the linear and neural models for cancer classification system.  Two neural models are provided for classification (vgg_encoder_model.py and patch_vgg_encoder_model.py).  The latter should be used with patches due to the smaller image size of patches (you can swap out the model in the main.py file).

The options_parser.py contains the possible options and default options for running the models.  Namely, a train and test directory can be included as well as a log directory for tensorboard functionality.  To store additional features for T-SNE visualizations, we write to a separate directory titled features.
