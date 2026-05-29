# MGMT Methylation Status Prediction in Glioblastoma using Multiparametric MRI

A deep learning project for non-invasive prediction of **MGMT promoter methylation status** in glioblastoma patients using multiparametric MRI, without the need for surgical biopsy.

---

## What is MGMT and Why Does It Matter?

**MGMT (O6-methylguanine-DNA methyltransferase)** is a DNA repair enzyme. When its promoter region is methylated, the gene is silenced — meaning the tumor loses its ability to repair DNA damage caused by chemotherapy drugs like temozolomide (TMZ). This makes MGMT-methylated tumors significantly more responsive to treatment.

Clinically, knowing a patient's MGMT status before surgery helps oncologists plan the most effective treatment strategy. Currently, this requires a tissue biopsy followed by lab analysis — an invasive procedure that carries surgical risks and can be affected by tumor heterogeneity. This project explores whether **MRI alone can predict MGMT status**, enabling a fully non-invasive preoperative assessment.

---

## Dataset

**UCSF Preoperative Diffuse Glioma MRI (PDGM)**

- Only **IDH-wildtype** patients were included, corresponding to true glioblastoma per the current WHO classification
- **354 patients** total
- Four MRI sequences per patient: **T1W, T1CE, T2W, FLAIR**
- Dataset available at: https://doi.org/10.7937/tcia.bdgf-8v37

> IDH-wildtype filtering was applied to ensure a clinically homogeneous cohort, as IDH-mutant tumors have a fundamentally different biology and prognosis, and mixing them would confound the MGMT prediction task.

---

## Approach

### The Core Problem with Standard Approaches

Most deep learning approaches feed the entire MRI slice to the model. The vast majority of pixels in a brain MRI, however, represent normal healthy tissue — essentially noise for a tumor classification task. This makes it harder for the model to learn the subtle signal differences that distinguish methylated from unmethylated tumors.

### Cross-Modality Mask Fusion

The central idea of this project is that **pathology detected in one MRI sequence should inform the region of interest in all other sequences**, even if those sequences appear normal in that region.

For example, a lesion visible only on FLAIR but not on T1W still harbors pathology in T1W at the same location. By combining segmentation masks from all four sequences into a single **fusion mask**, we extract tumor-only voxels from every sequence — eliminating surrounding normal anatomy entirely.

```
T1W mask  ──┐
T1CE mask ──┼──► Combined Fusion Mask ──► Applied to ALL 4 sequences
T2W mask  ──┤
FLAIR mask──┘
```

This produces clean, noise-reduced 3D tumor patches that are fed into the classifier.

### Model

A custom **3D CNN** was built to process volumetric MRI data, preserving spatial context across axial slices — something slice-by-slice 2D approaches lose.

- **Input:** 128×128×128 masked tumor patches from all four sequences
- **Architecture:** 4 downsampling blocks (Conv3D + ReLU + MaxPool3D)
- **Head:** Global Average Pooling → FC(512) → FC(256) → Sigmoid output
- **Optimizer:** Adam (lr=1e-3) | **Loss:** Binary Cross-Entropy
- **Training:** 100 epochs, batch size 16

---

## Results

| Cohort | Sequences | Patients | Accuracy |
|--------|-----------|----------|----------|
| UCSF PDGM (IDH-wildtype) | T1W + T1CE + T2W + FLAIR | 354 | **0.80** |

---

## Explainability

A **Streamlit application** was built alongside the model to make predictions interpretable and accessible:

| Module | Description |
|--------|-------------|
| Image Explorer | Browse T1CE/FLAIR volumes and overlaid segmentation masks |
| Crop Viewer | Inspect fusion-masked tumor patches fed into the model |
| Classifier | Get MGMT prediction with confidence score |
| Feature Maps | Visualize activations from each convolutional layer |
| GradCAM | Heatmap overlay showing which regions drove the prediction |

The app can be deployed locally via Docker and integrates with hospital PACS/RIS/HIS systems, enabling radiologists to use it directly in their workflow.

---

---

---

## Requirements

```
Python 3.7.0
PyTorch
SciPy 1.4.0
Panda
NumPy
```
