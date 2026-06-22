# Cancer Site Detection CNN — CT-Scan Imaging with Grad-CAM Explainability

A deep learning project for **lung cancer subtype classification from CT scans**, comparing three modeling strategies and deploying the winner as a live, interactive demo with explainability built in.

**🔗 [Live demo on Hugging Face](https://huggingface.co/spaces/ablackwellb/chest-ct-cancerscan)** · **▶️ [Project walkthrough on YouTube](https://youtu.be/qPS9zHlJQLY)**

---

## Overview

This project classifies chest CT-scan images into lung cancer subtypes (adenocarcinoma, large-cell carcinoma, squamous-cell carcinoma) versus normal tissue. Rather than building a single model, I benchmarked three distinct approaches to determine the most accurate *and* generalizable strategy on a small, class-imbalanced medical imaging dataset — then deployed the strongest model with Grad-CAM explainability so its predictions can be visually interrogated, not just trusted blindly.

## Dataset

A public, four-class chest CT-scan dataset from Kaggle (three cancer subtypes + normal), with meaningful class imbalance — a realistic constraint that shaped every modeling decision below.
Dataset source: (https://www.kaggle.com/datasets/hanyhossam/chest-ctscan-images-dataset)

## Approach — Three-Model Comparison

| Model | Test Accuracy | Notes |
|---|---|---|
| Baseline CNN (from scratch) | ~30.6% | Collapsed under class imbalance — defaulted to the dominant class. Served as the benchmark that justified transfer learning. |
| AutoKeras (AutoML / neural architecture search) | ~49% | Improved over baseline but lacked stability across runs. |
| **EfficientNetB0 (transfer learning)** | **85.71%** | **Selected model** — fine-tuned via transfer learning. |

The baseline failure was informative, not wasted: it demonstrated that a small imbalanced dataset couldn't support a from-scratch network, which is precisely what motivated the transfer-learning approach.

## Results — Selected Model (EfficientNetB0)

| Split | Accuracy | Macro-F1 | Macro-AUC |
|---|---|---|---|
| Training | 83.33% | 0.8459 | 0.968 |
| Test | 85.71% | 0.8551 | 0.969 |

A macro-AUC of 0.969 on a multi-class clinical imaging task indicates strong separability across subtypes. The confusion matrix showed the model's main difficulty was distinguishing adenocarcinoma from large-cell carcinoma — an overlap that mirrors genuine diagnostic ambiguity radiologists themselves encounter, rather than an arbitrary model error.

## Explainability — Grad-CAM

To move beyond a black-box prediction, I implemented Grad-CAM (gradient-weighted class activation mapping). The resulting heatmaps confirm the model concentrates attention on clinically relevant lung regions rather than background artifacts — supporting interpretability and trust.

![Grad-CAM example] (gradcam_example.png)

## Responsible Use

This model is a **decision-support prototype, not a diagnostic tool.** It is not a substitute for a trained radiologist and must not be used for clinical diagnosis. It was developed on a public, de-identified dataset with an emphasis on transparency, reproducibility, and honest reporting of its limitations — including the small dataset size and class imbalance noted above.

## Tech Stack

Python · TensorFlow / Keras · EfficientNetB0 (transfer learning) · AutoKeras · Grad-CAM · Gradio · Hugging Face Spaces · scikit-learn · NumPy · Matplotlib

## Repository Contents

- `cancer_site_detection_cnn.ipynb` — full training, evaluation, and Grad-CAM notebook
- `gradcam_example.png` — sample explainability heatmap
- `README.md` — this file

## References

- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML.
- Topol, E. (2019). *High-performance medicine: the convergence of human and artificial intelligence.* Nature Medicine.
- Shen, D., Wu, G., & Suk, H.-I. (2017). *Deep Learning in Medical Image Analysis.* Annual Review of Biomedical Engineering.
- Hany, M. (2020). Chest CT-Scan Images [Dataset]. Kaggle.


---

**Ashley Blackwell** — Applied Data Scientist
[LinkedIn](https://linkedin.com/in/ablackwellb) · [Hugging Face](https://huggingface.co/ablackwellb) · [Tableau Public](https://public.tableau.com/app/profile/ablackwellb)
