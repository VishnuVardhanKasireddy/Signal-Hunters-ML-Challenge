# Hack4Hack 2026 — Round 2: Signal Hunters

## Project Overview

**Signal Hunters** is a machine learning challenge from Hack4Hack 2026 Round 2 focused on classifying narrowband radio signals and identifying potential extraterrestrial intelligence (ETI) candidates from radio telescope observations.

The task involved analyzing six-panel observation cadences containing time-frequency signal data and classifying each observation into one of nine categories. A key challenge was distinguishing genuine astronomical signals from radio frequency interference (RFI), noise, and other signal patterns.

The dataset introduced a significant **domain shift**: the training observations came from the Green Bank Telescope (GBT), while the unseen test observations came from the Parkes telescope (PKS). This required developing models that could generalize across different observation environments rather than simply perform well on a random validation split.

### Challenge Objective

Build a machine learning classifier that maximizes **Macro F1-score** across the nine signal categories, with particular attention to detecting rare ETI candidates while maintaining reliable classification across all classes.

### Best Result

- **Best experiment:** E9 — ON/OFF Comparison CNN
- **Public leaderboard Macro F1-score:** 0.7675
- **Previous best public score:** 0.4784 (E5 — Original CNN)
- **Improvement:** +0.0510 absolute Macro F1

## Problem Statement & Dataset

### Problem Statement

The objective of Hack4Hack 2026 Round 2 was to classify radio telescope observations into nine categories using machine learning. The dataset contains time-frequency representations of radio signals organized into six-panel observation cadences.

The challenge was inspired by the search for extraterrestrial intelligence (SETI), where potential signals of interest must be distinguished from natural astronomical phenomena, background noise, and radio frequency interference (RFI).

### Dataset Overview

The dataset was provided as NumPy arrays containing the observation cadences, accompanied by CSV metadata files.

| Dataset | Number of Samples | Data Shape |
|---|---:|---|
| Training | 18,000 | `(18000, 6, 16, 128)` |
| Test | 10,000 | `(10000, 6, 16, 128)` |

Each sample contains six panels, with each panel represented as a `16 × 128` time-frequency matrix.

- **Rows:** Time bins.
- **Columns:** Frequency channels.
- **Zero-valued pixels:** Missing or unobserved measurements that require careful handling during preprocessing.
- **Training telescope:** Green Bank Telescope (GBT).
- **Test telescope:** Parkes Telescope (PKS).

### Signal Classes

The model predicts one of the following nine classes:

| Class | Description |
|---|---|
| `eti_candidate` | Candidate signal of potential extraterrestrial origin |
| `frb` | Fast radio burst |
| `noise` | Background noise |
| `pulsar` | Signal associated with a pulsar |
| `radar_chirp` | Frequency-drifting radar signal |
| `rfi_intermittent` | Intermittent radio frequency interference |
| `rfi_persistent` | Persistent radio frequency interference |
| `satellite_pass` | Signal associated with a satellite pass |
| `scintillating_star` | Signal exhibiting scintillation-related variability |

### Key Challenges

1. **Class imbalance:** The nine classes have different sample counts, making Macro F1-score more informative than accuracy alone.
2. **Missing measurements:** Zero-valued pixels must be handled carefully to avoid treating missing data as genuine signal measurements.
3. **ON/OFF observation cadence:** Comparing observations taken during ON and OFF periods can help distinguish astronomical signals from interference.
4. **Domain shift:** The training and test sets come from different telescopes, with differences in sampling intervals and frequency-channel widths. Models must generalize across these differences.
5. **Rare ETI candidates:** Detecting the minority `eti_candidate` class is particularly difficult because of its limited representation in the training data and its similarity to interference patterns.

## Approach & Experiment Evolution

The solution was developed through a series of controlled experiments, starting with handcrafted signal features and progressing to deep learning architectures designed to exploit the ON/OFF observation cadence.

### Experiment Summary

| Experiment | Approach | Validation Macro F1 | Public Macro F1 |
|---|---|---:|---:|
| E0 | Baseline handcrafted features with HistGradientBoosting | 0.5461 | 0.2040 |
| E1 | Bandpass correction and improved handcrafted features | 0.5722 | 0.2145 |
| E2 | Straight-line signal drift features | — | — |
| E3 | Drift features with ON/OFF and global-time features | 0.6141 | 0.2401 |
| E4 | Additional curved-drift features | 0.6089 | — |
| E5 | Original convolutional neural network (CNN) | 0.6559 | 0.4784 |
| E6 | Smaller CNN architecture | 0.5720 | — |
| E7 | CNN with additional median-corrected signal channels | 0.5345 | — |
| E8 | CNN with cadence-wise normalization | 0.5208 | — |
| **E9** | **ON/OFF Comparison CNN** | **0.7675** | **0.5294** |

*Validation scores are from the local stratified validation split. Public leaderboard scores are reported only for experiments with recorded submissions. A dash indicates that no corresponding score is recorded.*

### Feature Engineering

The initial experiments focused on extracting meaningful signal characteristics from the six observation panels.

Key techniques included:

- Bandpass correction and robust signal normalization.
- Straight-line frequency-drift searches.
- Curved-drift feature exploration.
- ON/OFF signal-strength comparisons.
- Cross-panel consistency and approximate global-time drift features.

Although these methods improved local validation performance, their public leaderboard results remained limited, suggesting that handcrafted features alone did not generalize well enough to the unseen telescope data.

### Deep Learning Experiments

The project then moved to convolutional neural networks (CNNs), which learn signal representations directly from time-frequency matrices.

The initial CNN (E5) achieved a public Macro F1-score of **0.4784**, substantially outperforming the handcrafted-feature models.

Several follow-up experiments tested smaller architectures and alternative normalization strategies. These changes did not improve on E5, so the original CNN remained the baseline for further development.

### E9 — ON/OFF Comparison CNN

The final and best-performing experiment introduced an architecture explicitly designed around the observation cadence.

Instead of encoding all six panels together from the beginning, E9 used a shared CNN encoder to process each panel independently. The resulting representations were grouped into ON and OFF observations and combined using:

- Mean ON representation.
- Mean OFF representation.
- ON-minus-OFF representation.
- Absolute ON/OFF difference.

These representations were concatenated and passed through a classification head to predict the signal class.

This architecture achieved a validation Macro F1-score of **0.7675** and a public leaderboard score of **0.5294**, making E9 the strongest evaluated model in the project.


## Model Architecture & Training Details

### E9 — ON/OFF Comparison CNN

E9 was the best-performing model developed during the challenge. Its architecture explicitly models the relationship between ON and OFF observations rather than treating all six panels as an undifferentiated input.

#### Architecture

1. **Panel-wise encoding:** A shared CNN encoder processes each of the six observation panels independently. Each panel has two input channels: normalized signal intensity and a missing-data mask.
2. **Feature extraction:** Three convolutional blocks learn increasingly complex time-frequency patterns.
3. **ON/OFF aggregation:** Encoded representations from panels 0, 2, and 4 are averaged to form the ON representation. Panels 1, 3, and 5 are averaged to form the OFF representation.
4. **Feature comparison:** Four representations are concatenated:
   - Mean ON features
   - Mean OFF features
   - ON-minus-OFF features
   - Absolute ON/OFF feature differences
5. **Classification head:** The combined representation is passed through fully connected layers to predict one of the nine signal classes.

#### Network Configuration

| Component | Configuration |
|---|---|
| Input | Six panels, each with signal and mask channels |
| Shared encoder | 3 convolutional blocks |
| Convolutional channels | 32, 64, 128 |
| Convolutional layers | 3 × 3 kernels with padding |
| Normalization | Batch Normalization |
| Activation | ReLU |
| Downsampling | Max Pooling |
| Feature pooling | Adaptive Average Pooling to `2 × 8` |
| Feature aggregation | ON, OFF, ON − OFF, absolute difference |
| Classification head | Linear → ReLU → Dropout → Linear |
| Hidden dimension | 256 |
| Output classes | 9 |
| Total parameters | Approximately 2.19 million |

### Input Preprocessing

The E9 model uses the same preprocessing strategy as the original CNN (E5).

- Convert input arrays to floating-point values.
- Identify valid measurements using a mask derived from nonzero pixels.
- Normalize each panel using its observed mean and standard deviation.
- Exclude missing pixels from normalization statistics.
- Preserve missing-data locations as explicit mask channels.

The resulting input has 12 channels: six normalized signal channels and six corresponding mask channels.

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | `0.001` |
| Weight decay | `0.0001` |
| Batch size | 64 |
| Validation batch size | 128 |
| Maximum epochs | 20 |
| Early stopping patience | 4 epochs |
| Loss function | Weighted Cross-Entropy Loss |
| Class weighting | Inverse square root of class frequency |
| Validation split | 80% training / 20% validation |
| Random seed | 42 |
| Model selection metric | Validation Macro F1-score |

Class weights were used to reduce the influence of class imbalance during training. The best checkpoint was selected using validation Macro F1-score rather than training loss alone.

### Validation Results

The best E9 checkpoint achieved:

- **Validation Macro F1-score:** 0.7675
- **Validation accuracy:** 82.4%
- **Validation samples:** 3,600

The model performed particularly well on pulsar, radar chirp, FRB, and noise classes. ETI candidate and scintillating star classification remained more challenging, highlighting opportunities for further improvement.

## Results & Evaluation

### Public Leaderboard Performance

The final submitted model, E9 (ON/OFF Comparison CNN), achieved a public leaderboard Macro F1-score of **0.5294**.

| Model | Validation Macro F1 | Public Macro F1 |
|---|---:|---:|
| E0 — Baseline | 0.5461 | 0.2040 |
| E1 — Bandpass-Corrected Features | 0.5722 | 0.2145 |
| E3 — ON/OFF Handcrafted Features | 0.6141 | 0.2401 |
| E5 — Original CNN | 0.6559 | 0.4784 |
| **E9 — ON/OFF Comparison CNN** | **0.7675** | **0.5294** |

E9 improved the public Macro F1-score by **0.0510** over E5, representing an approximately 10.7% relative improvement.

### E9 Per-Class Validation Performance

The classification report below shows the performance of the best E9 checkpoint on the local validation set.

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| ETI candidate | 0.373 | 0.260 | 0.307 |
| FRB | 0.859 | 0.933 | 0.894 |
| Noise | 0.878 | 0.867 | 0.873 |
| Pulsar | 0.902 | 0.991 | 0.944 |
| Radar chirp | 0.885 | 0.957 | 0.919 |
| Intermittent RFI | 0.767 | 0.742 | 0.755 |
| Persistent RFI | 0.833 | 0.881 | 0.856 |
| Satellite pass | 0.780 | 0.837 | 0.807 |
| Scintillating star | 0.662 | 0.472 | 0.551 |

### Error Analysis

The confusion matrix revealed several recurring classification challenges:

- **ETI candidates:** Many actual ETI candidates were misclassified as intermittent RFI, indicating difficulty distinguishing potential extraterrestrial signals from interference.
- **Scintillating stars:** Some examples were classified as noise or persistent RFI, suggesting overlap in their learned signal representations.
- **RFI categories:** Intermittent and persistent RFI were frequently confused with each other.
- **Strong-performing classes:** Pulsar, radar chirp, and FRB achieved high F1-scores, indicating that the model learned useful representations for these signal types.

### Generalization to Unseen Telescope Data

The validation Macro F1-score of 0.7675 was considerably higher than the public leaderboard score of 0.5294.

This gap highlights the challenge of **cross-telescope domain generalization**. The training data came from GBT, while the test data came from PKS, with differences in sampling intervals and frequency-channel widths.

The results demonstrate that strong local validation performance does not necessarily translate directly to unseen observational environments. Evaluating models on data collected under different conditions is therefore essential for realistic performance assessment.


## Repository Structure

The repository is organized to separate datasets, experimental notebooks, reusable feature extraction code, trained model checkpoints, and generated submissions.

```
Signal-Hunters/
├── data/
│   ├── train_cadences.npy
│   ├── test_cadences.npy
│   ├── train_meta.csv
│   └── test_meta.csv
├── notebooks/
│   ├── E0_baseline.ipynb
│   ├── E1_bandpass_correction.ipynb
│   ├── E2_drift_features.ipynb
│   ├── E3_on_off_features.ipynb
│   ├── E4_curved_drift.ipynb
│   ├── E5_original_cnn.ipynb
│   ├── E6_small_cnn.ipynb
│   ├── E7_corrected_channels.ipynb
│   ├── E8_cadence_normalization.ipynb
│   └── E9_on_off_comparison_cnn.ipynb
├── src/
│   └── drift_features.py
├── outputs/
│   ├── features/
│   ├── models/
│   └── submissions/
├── requirements.txt
└── README.md
```

## Setup & Reproduction

### Prerequisites

- Python 3.10 or later
- pip
- Git
- Visual Studio Code or Jupyter Notebook
- CPU or CUDA-compatible GPU (GPU recommended for faster CNN training)

### 1. Clone the Repository

```
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Signal-Hunters
```

2. Create a Virtual Environment
Windows
```
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS
```
python3 -m venv .venv
source .venv/bin/activate
```

3. Install Dependencies
Install the required Python packages:
```
pip install -r requirements.txt
```

Core libraries used during experimentation included:
- NumPy — numerical operations and array processing
- pandas — metadata and submission handling
- scikit-learn — data splitting, metrics, and baseline models
- SciPy — signal processing and feature extraction
- PyTorch — CNN implementation and training
- Matplotlib — visualization and confusion matrices
4. Prepare the Dataset
Place the competition-provided dataset files in the data/ directory:
```
data/
├── train_cadences.npy
├── test_cadences.npy
├── train_meta.csv
└── test_meta.csv
```

The dataset files were provided as part of the Hack4Hack 2026 Round 2 challenge and may not be publicly redistributable. Obtain them through the official competition resources.
5. Run the Experiments
Open the repository in Visual Studio Code or launch Jupyter Notebook:
jupyter notebook


The notebooks/ directory contains the experiments developed throughout the challenge, progressing from handcrafted features to CNN-based approaches.

Run the notebooks individually to explore the development process:
1. E0–E4: Baseline models, signal feature engineering, drift detection, and ON/OFF features.
2. E5–E8: CNN experiments and alternative preprocessing strategies.
3. E9: ON/OFF Comparison CNN, the best-performing experiment.
6. Generate Predictions
To reproduce the best-performing approach, open the E9 notebook and execute its preprocessing, model definition, training, and prediction cells.

The model generates one predicted class for each test observation. Use the IDs from test_meta.csv to construct the submission file.

## Key Learnings & Future Improvements

### Key Learnings

- **ON/OFF cadence modeling:** Explicitly comparing ON and OFF observations improved classification performance over treating all six panels as a single input.
- **Experiment-driven development:** Evaluating multiple feature-engineering and CNN approaches helped identify which changes produced measurable improvements.
- **Class imbalance:** Macro F1-score and per-class evaluation revealed weaknesses that overall accuracy alone would have hidden.
- **Domain generalization:** Strong local validation performance did not fully transfer to the unseen telescope, highlighting the difficulty of cross-instrument generalization.
- **Missing-data handling:** Explicit masks helped preserve information about unobserved pixels instead of treating them as ordinary signal measurements.
- **Error analysis:** Confusion matrices identified difficult class pairs, particularly ETI candidates versus intermittent RFI and scintillating stars versus noise or interference.

### Future Improvements

#### 1. Improve Cross-Telescope Generalization
Investigate normalization and calibration strategies that account for differences in frequency-channel widths, sampling intervals, and telescope characteristics.

#### 2. Improve Rare-Class Detection
Explore carefully tuned class weights, focal loss, and targeted data augmentation to improve ETI candidate recall without significantly increasing false positives.

#### 3. Refine ON/OFF Feature Learning
Experiment with alternative feature-fusion methods, attention mechanisms, and explicit comparisons between corresponding ON and OFF panels.

#### 4. Investigate Signal-Specific Features
Explore frequency-drift trajectories, temporal consistency, signal bandwidth, and persistence patterns to help distinguish astronomical signals from radio frequency interference.

#### 5. Strengthen Evaluation
Use repeated stratified validation experiments and per-class metrics to assess model stability. Where possible, evaluate on data from different observing conditions to better estimate generalization performance.

#### 6. Improve Training Efficiency
Optimize panel-wise CNN processing, batch preparation, and GPU utilization to reduce training time and enable more controlled experiments.

### Conclusion

Hack4Hack 2026 Round 2 provided practical experience in radio-signal classification, feature engineering, convolutional neural networks, class-imbalance handling, and domain generalization.

The ON/OFF Comparison CNN (E9) achieved the project's best public leaderboard Macro F1-score of **0.5294**, improving on the original CNN's score of 0.4784.

The most important outcome was learning to translate the structure of the observation process into a model architecture, validate hypotheses through controlled experiments, and analyze the gap between local validation and real-world generalization.


## Technologies Used

- **Programming Language:** Python
- **Deep Learning:** PyTorch
- **Machine Learning:** Scikit-learn
- **Numerical Computing:** NumPy
- **Data Processing:** pandas, SciPy
- **Data Visualization:** Matplotlib
- **Development Environment:** Visual Studio Code, Jupyter Notebook
- **Version Control:** Git, GitHub

### Core Techniques

- Convolutional Neural Networks (CNNs)
- ON/OFF observation comparison and feature fusion
- Signal normalization and missing-data masking
- Frequency-drift feature engineering
- Class-weighted cross-entropy loss
- Macro F1-score evaluation
- Confusion matrix analysis


## Acknowledgements

- Thanks to the **Hack4Hack 2026** organizers for designing the Signal Hunters challenge.
- The challenge provided an opportunity to explore signal classification, radio-frequency interference (RFI), and machine learning techniques inspired by SETI-style signal detection.
- Thanks to the open-source Python ecosystem, particularly NumPy, pandas, scikit-learn, and PyTorch, for the tools used to develop and evaluate the models.

---