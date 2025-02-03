##### README.md #####
# CryoVisionNet - CryoET Object Identification

## 🏆 Kaggle Competition: CZII - CryoET Object Identification

This repository contains our solution for the [CZII - CryoET Object Identification](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/) Kaggle competition, organized by the **CZ Imaging Institute**, a **Chan Zuckerberg Initiative (CZI) Institute**.

### 🧑‍🔬 **Competition Overview**
Cryo-electron tomography (CryoET) enables near-atomic-resolution imaging of proteins within their natural environments, providing unprecedented insights into cellular biology. However, identifying individual protein complexes in 3D tomograms remains a major challenge.

In this competition, participants develop ML models to automatically detect and classify **five types of protein complexes** from **3D tomograms**. The ultimate goal is to accelerate biomedical discoveries and disease treatment by improving protein annotation techniques.

### 🔍 **Evaluation Metric**
Submissions are evaluated using the **F-beta metric (β=4)**, which emphasizes **recall over precision**. This approach penalizes missed detections (false negatives) more heavily than false positives, ensuring that difficult-to-detect particles are accurately identified.

## 📌 **Project Structure**
```
cryoet_project/
│── config.py          # Configuration settings (paths, hyperparameters, etc.)
│── data_loading.py    # Data loading and preprocessing
│── transforms.py      # Data augmentation and transformation functions
│── patching.py        # Patch extraction and reconstruction
│── model.py           # UNet model and training logic
│── training.py        # Training script using Lightning
│── utils.py           # Utility functions (e.g., logging, visualization)
│── main.py            # Main script to run training and validation
│── requirements.txt   # Dependencies
│── README.md          # Documentation
```

## 🚀 **Solution Approach**
We designed a **3D UNet-based deep learning pipeline** for CryoET object identification:
1. **Preprocessing & Augmentation**: Tomogram normalization, random cropping, rotation, and flipping.
2. **Patch-Based Training**: Breaking large tomograms into **96×96×96 patches** to improve detection.
3. **Deep Learning Model**: A **3D UNet** trained using **Tversky loss**, which balances false positives and false negatives.
4. **Postprocessing & Prediction**: Reconstructing predictions from patches and generating final 3D coordinates.

## 🔧 **Installation & Usage**
### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train the Model
```bash
python training.py
```

### Run Inference
```bash
python main.py
```

## 📜 **Citations & References**
If you use this repository, please cite the competition:

> Harrington, K., Paraan, M., Cheng, A., Ermel, U. H., Kandel, S., Kimanius, D., et al. (2024). "CZII - CryoET Object Identification." Kaggle. [https://www.kaggle.com/competitions/czii-cryo-et-object-identification](https://www.kaggle.com/competitions/czii-cryo-et-object-identification)

## 🤝 **Acknowledgments**
This work was made possible by:
- **CZ Imaging Institute** and **Chan Zuckerberg Initiative (CZI)**
- The Kaggle community and organizers
- The research community advancing CryoET object detection

---
🔬 **CryoVisionNet - Unlocking Molecular Biology with AI!** 🚀

