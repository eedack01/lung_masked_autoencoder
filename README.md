# 📘 Code for: *Unmasking Interstitial Lung Diseases: Leveraging Masked Autoencoders for Diagnosis*

This repository contains the code used in our paper:

### 🔗 [Unmasking Interstitial Lung Diseases: Leveraging Masked Autoencoders for Diagnosis](https://arxiv.org/)

---

## 🔍 Key Contributions

The core utilities provided here are designed to support researchers working with training masked autoencoders with CT scans. The **main contributions** include:

- **MAE model** training pipeline with and without our new proposed loss.
- **Comprehensive preprocessing utilities** to standardize, process, and analyze CT images.
- **Baseline files provided** to compare against. 

---

## 📁 Repository Structure

- **`mae_train.py, mae_train_with_new_loss, mae_train_with_new_loss_clean`**  
  This is the main file for training done in Pytorch lightning for easy scaling and setting config, with_new_loss was our first attempt at getting the loss to focus on certain areas and we cleaned this up later in the other file, to run the pipeline you can run 'python3 /mae_main_new.py fit --config /config/mae_config_new.json'

- **`mae_eval.py`**  
  We predict reconstructed patches with the models predict function and evaluate the mean absolute error and SSIM.

- **`ild_finetune.py`**  
  Our finetuning the encoder from the MAE and using the downstream model for the classification tasks.

- **`ild_walsh_baseline.py and ild_walsh_baseline_vit.py`**  
  Training pipeline for Walsh et al. baseline with InceptionResNetv3 and ViT.

- **`ild_radiomics_baseline.py`**  
  Modified version from Fontanellaz et al. using nnUNet (provided below) and radiomic features.

### `Utils/` Directory

This directory includes modular scripts and helper functions to streamline data preparation and visualization:

- **`build_walsh_baseline_db.py`**  
  Code to create the montages in Walsh et al., uses the LungMask library to get the lung masks.

- **`class_weights.py`**  
  Get class weights in imbalanced problems to pass into the loss function.

- **`preprocess.py`**  
  Saves niftis as npz files, first resamples, crops the image based on the lung mask, scale values to between 0 and 1.

- **`visualise.py and visualise_patches.py`**  
  Visualise.py will reconstruct patches based on the masked tokens and replace the masked tokens in an image, you can save this then visualise it with a chosen tool e.g. slicer3d.
  Visualise patches will create a plot of reconstructed patch and ground truth patch beneath each other in a matplotlib fashion.

---

## 🛠️ Notes on Dataset Classes & Transforms

You will need to update the dataset class to match your chosen structure.

---

## 📖 Citing Our Work

If you use this code or find it helpful, please consider citing:

```
@misc{dack2025understandingdatasetbiasmedical,
      title={Understanding Dataset Bias in Medical Imaging: A Case Study on Chest X-rays}, 
      author={Ethan Dack and Chengliang Dai},
      year={2025},
      eprint={2507.07722},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.07722}, 
}
```

For setting up the baseline we made use of a pretrained nnUNet on common interstitial lung patterns, we have made this available below,⚠️ *Important*: This was trained using the first version of nnUNet. The model is available here: 

If you make use of this, please cite the following paper: 

```
@ARTICLE{29,
  author={Fontanellaz, M. and Christe, A. and Christodoulidis, et al.},
  journal={IEEE Access}, 
  title={Computer-aided Diagnosis System for Lung Fibrosis: from the Effect of Radiomic Features and Multi-layer-perceptron Mixers to Pre-clinical Evaluation}, 
  year={2024}}
```

---

## 🙏 Acknowledgments

This work would not have been possible without the release of the following datasets:

- [OSIC Pulmonary Fibrosis Progression](https://www.kaggle.com/competitions/osic-pulmonary-fibrosis-progression/data)
- [Multimedia database of interstitial lung diseases](https://medgift.hevs.ch/wordpress/databases/ild-database/)
- [CT Images in COVID-19 ](https://www.cancerimagingarchive.net/collection/ct-images-in-covid-19/)
- [COVID-19-AR](https://www.cancerimagingarchive.net/collection/covid-19-ar/)
- [PleThora](https://www.cancerimagingarchive.net/analysis-result/plethora/)
- [MP-COVID-19-SegBenchmark](https://zenodo.org/records/3757476)
- [MosMed](https://www.kaggle.com/datasets/andrewmvd/mosmed-covid19-ct-scans)
- [COVID-CT-MD](https://springernature.figshare.com/articles/dataset/COVID-CT-MD_COVID-19_Computed_Tomography_Scan_Dataset_Applicable_in_Machine_Learning_and_Deep_Learning/12991592?file=26069987)
- [STOIC2021](https://zenodo.org/records/7969800)
- [COVID-19 for Radiomics](https://www.imagenglab.com/newsite/covid-19/?eeFolder=Zipped_patients&eeListID=1)


We also rely heavily on the [LungMask](https://github.com/JoHof/lungmask) library — if you find our code useful, please consider citing their work as well.


---

Thank you for exploring our code!
