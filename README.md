# 📘 Code for: *Unmasking Interstitial Lung Diseases: Leveraging Masked Autoencoders for Diagnosis*

This repository contains the code used in our paper:

### 🔗 [Unmasking Interstitial Lung Diseases: Leveraging Masked Autoencoders for Diagnosis](https://arxiv.org/)

---

## 🔍 Key Contributions

This repository supports training and evaluation pipelines for masked autoencoders (MAEs) applied to CT scans, with the following **main contributions**:

- ⚙️ A full **MAE training pipeline** with and without our proposed custom loss.
- 🧼 **Comprehensive preprocessing tools** to standardize and prepare CT images.
- 🧪 **Baselines provided** for comparison, including radiomics and ViT-based classifiers.

📦 **Pretrained MAE models** are available [here](https://www.dropbox.com/scl/fo/u9t4jb7edzkdcd5wpmz64/AMERnJD0Gk8A1FMnDpVigzw?rlkey=ywa18ok6pfjyrh09u3qat6sn6&st=ksxyjbam&dl=0).

---

## 📁 Repository Structure

- **`mae_train.py`, `mae_train_with_new_loss.py`, `mae_train_with_new_loss_clean.py`**  
  PyTorch Lightning-based training scripts. The `with_new_loss` files implement our region-focused loss function. To launch training, run:

  ```bash
  python3 mae_main_new.py fit --config /config/mae_config_new.json
  ```

- **`mae_eval.py`**  
  Evaluates reconstruction quality using MAE and SSIM metrics.

- **`ild_finetune.py`**  
  Finetunes the MAE encoder on downstream classification tasks.

- **`ild_walsh_baseline.py` / `ild_walsh_baseline_vit.py`**  
  Baseline training using Walsh et al.’s montage-style pipeline with InceptionResNetv3 and ViT.

- **`ild_radiomics_baseline.py`**  
  Implements the Fontanellaz et al. radiomics pipeline using nnUNet segmentations.

---

## 🧰 `Utils/` Directory

Modular tools for preprocessing and visualization:

- **`build_walsh_baseline_db.py`**  
  Builds image montages following Walsh et al., uses LungMask for segmentation.

- **`class_weights.py`**  
  Computes class weights to help balance imbalanced classification problems.

- **`preprocess.py`**  
  Converts and preprocesses CT scans: resampling, cropping via lung masks, scaling to [0, 1], and saving as `.npz`.

- **`visualise.py` & `visualise_patches.py`**  
  `visualise.py` reconstructs and visualizes masked areas for inspection.  
  `visualise_patches.py` compares predicted and ground-truth patches using matplotlib.

---

## 🛠️ Notes on Dataset Classes & Transforms

You will need to update the dataset class to match your chosen structure.

---

## 📖 Citing Our Work

If you use this code or find it helpful, please cite our work:

```bibtex
@misc{dack2025understandingdatasetbiasmedical,
  title={Understanding Dataset Bias in Medical Imaging: A Case Study on Chest X-rays}, 
  author={Ethan Dack and Chengliang Dai},
  year={2025},
  eprint={2507.07722},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2507.07722}
}
```

To replicate the baseline in Fontanellaz et al., we use a pretraiend nnUNet (version 1 of nnUNet!) to segment common interstitial lung patterns, the preprocessing steps are similar to the masked auto encoder, you will need to use the resampling, cropping to the lung area (with margin) and then scaling the values with the HU units, you don't need to resize your image here to (128, 128, 128) just use the image size after the cropping, if you are getting odd results try flipping the image array with something like np.flip(axis=1), model link:

📦 [Download the nnUNet model here](https://www.dropbox.com/scl/fo/4bd86pe7q54u8nx21laer/AC7Yyg3ir8GXTx7KvTG8j8o?rlkey=95rzlk4v6xyw6g2o5gzqq6kg8&st=0zmh68ak&dl=0)

If you use this model, please cite:

```bibtex
@ARTICLE{10381702,
  author={Fontanellaz, M. and Christe, A. and Christodoulidis, S. and Dack, E. and Roos, J. and Drakopoulos, D. and Sieron, D. and Peters, A. and Geiser, T. and Funke-Chambour, M. and Heverhagen, J. and Hoppe, H. and Exadaktylos, A. K. and Ebner, L. and Mougiakakou, S.},
  journal={IEEE Access}, 
  title={Computer-Aided Diagnosis System for Lung Fibrosis: From the Effect of Radiomic Features and Multi-Layer-Perceptron Mixers to Pre-Clinical Evaluation}, 
  year={2024},
  volume={12},
  pages={25642-25656},
  doi={10.1109/ACCESS.2024.3350430}}
```

---

## 🙏 Acknowledgments

This project would not have been possible without the release of the following datasets:

- [OSIC Pulmonary Fibrosis Progression](https://www.kaggle.com/competitions/osic-pulmonary-fibrosis-progression/data)
- [Multimedia database of interstitial lung diseases](https://medgift.hevs.ch/wordpress/databases/ild-database/)
- [CT Images in COVID-19](https://www.cancerimagingarchive.net/collection/ct-images-in-covid-19/)
- [COVID-19-AR](https://www.cancerimagingarchive.net/collection/covid-19-ar/)
- [PleThora](https://www.cancerimagingarchive.net/analysis-result/plethora/)
- [MP-COVID-19-SegBenchmark](https://zenodo.org/records/3757476)
- [MosMed](https://www.kaggle.com/datasets/andrewmvd/mosmed-covid19-ct-scans)
- [COVID-CT-MD](https://springernature.figshare.com/articles/dataset/COVID-CT-MD_COVID-19_Computed_Tomography_Scan_Dataset_Applicable_in_Machine_Learning_and_Deep_Learning/12991592?file=26069987)
- [STOIC2021](https://zenodo.org/records/7969800)
- [COVID-19 for Radiomics](https://www.imagenglab.com/newsite/covid-19/?eeFolder=Zipped_patients&eeListID=1)

We also rely heavily on the [LungMask](https://github.com/JoHof/lungmask) library — please consider citing their work if you use it.

---

Thank you for exploring our code!
