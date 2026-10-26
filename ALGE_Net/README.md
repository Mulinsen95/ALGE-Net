# ALGE-Net

ALGE-Net (Adaptive Layer-wise Gated Edge-Fusion Network) is a deep learning framework for fetal gestational age (FGA) estimation from fetal magnetic resonance imaging (MRI).

This repository provides the model implementation, data loading modules, training and inference scripts, evaluation utilities, and supporting functions used in the ALGE-Net pipeline.

## 1. Project Structure

```text
ALGE_Net/
├── data/
│   ├── my_dataset.py
│   └── my_dataset_seq.py
├── models/
│   ├── Dual_branch_model.py
│   ├── Edge_branch.py
│   ├── KAN.py
│   └── KANResnet.py
├── utils/
│   ├── reg_R2.py
│   ├── utils1.py
│   └── utils2.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

## 2. Code Organization

The codebase is organized into three main modules: `data/`, `models/`, and `utils/`. The top-level scripts `train.py` and `predict.py` orchestrate model training and evaluation, respectively.

### 2.1 Entry-point Scripts

| File | Responsibility |
|---|---|
| `train.py` | Main training entry point. Loads the dataset, initializes the ALGE-Net model, generates Sobel-based structural targets, computes the training losses, updates model parameters, and manages the training loop. |
| `predict.py` | Main prediction and evaluation entry point. Loads trained checkpoints, performs slice-level inference and horizontal-flip test-time augmentation (TTA), aggregates predictions at the MRI-sequence level, calculates evaluation metrics, and generates result visualizations. |

### 2.2 Data Loading: `data/`

| File | Responsibility |
|---|---|
| `data/my_dataset.py` | Implements `MyDataSet`, the image-level dataset used during training. It reads image paths and ROI coordinates from the dataset records, loads DICOM images, extracts the specified ROI, converts the image to grayscale, and applies the configured preprocessing and transformations. |
| `data/my_dataset_seq.py` | Implements `MyDataSetSeq` for prediction-time slice loading. It organizes image samples by sequence using sequence-level metadata, retains the association between slices and their sequence identifiers, and prepares samples for sequence-level evaluation. It also contains dataset visualization utilities. |

The training and prediction datasets serve different purposes. `MyDataSet` supplies image-level samples for model optimization, whereas `MyDataSetSeq` preserves sequence identifiers so that predictions from multiple slices can be aggregated into a sequence-level estimate.

### 2.3 Model Components: `models/`

| File | Responsibility |
|---|---|
| `models/Dual_branch_model.py` | Defines `DualBranchModel`, the main ALGE-Net architecture. It integrates the semantic backbone, edge branch, gated feature-fusion modules, auxiliary structural supervision, and the final gestational-age regression head. |
| `models/Edge_branch.py` | Implements `EdgeBranch`, a convolutional encoder–decoder that extracts multi-scale structural features and produces an edge reconstruction. It also defines edge-related loss functions, including Dice loss and a combined edge loss. |
| `models/KANResnet.py` | Provides the ResNet-style backbone implementation used to extract hierarchical semantic features for age estimation. The module exposes the `resnet50` constructor. |
| `models/KAN.py` | Implements KAN-related layers, including `KANLinear`, which uses learnable spline-based transformations. These components are used by the regression head defined in the main model. |

The main architecture combines semantic representations from the ResNet backbone with structural representations from the edge branch. Within `Dual_branch_model.py`, gated fusion modules use edge features to modulate semantic features at multiple stages. The fused representations are then passed to the KAN-based regression head to predict gestational age.

The edge reconstruction task provides auxiliary structural supervision during training. It is not a separate clinical prediction output.

### 2.4 Utility Functions: `utils/`

| File | Responsibility |
|---|---|
| `utils/utils1.py` | Provides shared utilities for random-seed control, dataset splitting, sequence grouping, evaluation metrics, and result visualization. It includes functions used by both training and prediction scripts, such as `set_seed`, `read_csv_random_split_data`, `group_by_sequence`, and sequence-level evaluation and plotting functions. |
| `utils/utils2.py` | Contains training and experiment-management helpers, including `AverageMeter` for tracking running statistics, `ProgressMeter` for progress reporting, checkpoint and output-directory utilities, and learning-rate adjustment functions. |
| `utils/reg_R2.py` | Implements the coefficient of determination (\(R^2\)) calculation from the reference and predicted values. |

`utils/utils1.py` is a central shared module: it connects the data-splitting and sequence-grouping procedures with evaluation and visualization. `utils/utils2.py` provides supporting functions for training and experiment management, while `utils/reg_R2.py` contains a focused metric implementation.

## 3. How the Modules Work Together

The overall pipeline consists of data preparation, model training, checkpoint loading, slice-level inference, and sequence-level evaluation.

### 3.1 Training Pipeline

1. **Load and split the data.** `train.py` reads the dataset records and calls `read_csv_random_split_data()` from `utils/utils1.py` to prepare the training and test subsets.
2. **Load MRI images.** `MyDataSet` in `data/my_dataset.py` reads the image paths and ROI coordinates, loads the DICOM images, and prepares the input tensors.
3. **Generate structural targets.** `train.py` generates Sobel-based edge targets to provide auxiliary structural supervision.
4. **Forward pass through ALGE-Net.** `DualBranchModel` in `models/Dual_branch_model.py` processes the input through the semantic backbone and edge branch, combines their representations through gated fusion, and predicts gestational age.
5. **Compute losses and optimize.** The training script computes the regression and auxiliary structural losses, updates the model using the configured optimizer, and manages training progress and checkpoints using the available utilities.

### 3.2 Prediction and Evaluation Pipeline

1. **Load the evaluation data.** `predict.py` reads the dataset and uses shared utilities from `utils/utils1.py` to prepare the evaluation subset and group records by MRI sequence.
2. **Prepare slice-level samples.** `MyDataSetSeq` in `data/my_dataset_seq.py` creates samples while preserving their sequence identifiers.
3. **Load the trained model.** `predict.py` initializes `DualBranchModel` and loads the corresponding checkpoint.
4. **Perform inference.** Each slice is evaluated using the trained network. When enabled, horizontal-flip TTA averages predictions from the original and flipped inputs.
5. **Aggregate predictions.** Slice-level predictions are grouped by sequence and averaged to obtain sequence-level gestational-age estimates.
6. **Evaluate and visualize results.** Evaluation functions in `utils/utils1.py` calculate the relevant metrics and generate result plots.

The high-level dependency flow is summarized below:

```text
                         train.py
                            |
             +--------------+--------------+
             |              |              |
       utils/utils1.py  data/my_dataset.py  models/Dual_branch_model.py
             |              |              |
       Data splitting   MRI/ROI loading     +------------------+
                                            |                  |
                                    models/KANResnet.py  models/Edge_branch.py
                                            |                  |
                                            +--------+---------+
                                                     |
                                              models/KAN.py
                                                     |
                                           Age prediction and loss
                                                     |
                                           utils/utils2.py
                                      (training support and checkpoints)


                         predict.py
                            |
                 +----------+-----------+
                 |                      |
          utils/utils1.py      data/my_dataset_seq.py
                 |                      |
       Sequence grouping and       Slice-level samples
       evaluation functions              |
                 |                models/Dual_branch_model.py
                 |                      |
                 +----------+-----------+
                            |
                Sequence-level predictions,
                  metrics, and visualizations
```

This diagram describes the logical relationships among the modules. Individual functions may also call additional helpers within the same files.

## 4. Environment

- Python: 3.12
- PyTorch: 2.11.0+cu128
- torchvision: 0.26.0+cu128
- CUDA runtime: 12.8
- GPU acceleration: Available in the current development environment

### Installation

Install a compatible PyTorch and torchvision build for your operating system and CUDA environment:

https://pytorch.org/get-started/locally/

Install the remaining dependencies from the project directory:

```bash
pip install -r requirements.txt
```

## 5. Data Preparation

The original MRI dataset is not included in this repository. All data must be handled in accordance with the applicable ethics approval, institutional requirements, and data-use permissions.

The current dataset CSV is named `newdataWindowU1.csv`. Its records include participant identifiers, image paths, ROI coordinates, gestational-age fields, and a dataset-split field.

| Field | Description |
|---|---|
| `fs` | Participant identifier |
| `image` | MRI image path or identifier |
| `XMIN`, `YMIN`, `XMAX`, `YMAX` | ROI bounding-box coordinate fields |
| `week` | Gestational age in weeks |
| `day` | Gestational age in days |
| `split` | Dataset-split field |

An example record is shown below:

```csv
fs,image,XMIN,YMIN,XMAX,YMAX,week,day,split
100009662,E:\...\I1900000,187,339,304,234,35,251,
```

The image path is abbreviated for illustration. Replace it with a valid local path when preparing the dataset.

The study uses participant-level data partitioning, with 164 participants assigned to training and 43 to testing. No participant should appear in both subsets. Verify the split-generation procedure before running the scripts, particularly because the supplied CSV's `split` column may be empty.

The preprocessing pipeline includes DICOM loading, ROI extraction, grayscale conversion, and image resizing to \(96 \times 96\) pixels. The exact processing behavior follows the implementation in the dataset and training modules.

## 6. Model Training

The training entry point is `train.py`.

The main settings reported in the paper are:

- Epochs: 350
- Optimizer: Adam
- Initial learning rate: 0.001
- Batch size: 32
- Weight decay: 0.00001
- Learning-rate scheduler: StepLR, reducing the learning rate by a factor of 0.1 after epoch 300
- Random seeds: 1–10

Example command:

```bash
python train.py --data_dir "/path/to/data" --dataset "newdataWindowU1" --epochs 350 --batch_size 32 --lr 0.001
```

Replace the example data path with the local dataset directory. Check the checkpoint output path in the training script before execution.

## 7. Prediction and Evaluation

The prediction and evaluation entry point is `predict.py`.

Example command:

```bash
python predict.py --data_dir "/path/to/data" --dataset "newdataWindowU1" --weight_dir "/path/to/checkpoints" --output_dir "results" --device auto --seeds 1,2,3,4,5,6,7,8,9,10
```

Update the data and checkpoint paths to match your environment. Confirm the supported arguments and checkpoint naming convention in the source code.

The evaluation pipeline uses horizontal-flip test-time augmentation when enabled, aggregates slice-level predictions into sequence-level estimates, and calculates evaluation metrics. The paper reports the main results as the mean and standard deviation across 10 runs.

## 8. Reproducibility

The reported experiments use random seeds 1–10. The implementation controls the random states of Python, NumPy, PyTorch CPU, and CUDA and configures CuDNN for deterministic behavior.

Model complexity profiling was performed using an NVIDIA RTX 4090 GPU with an input size of \(96 \times 96\) pixels and a batch size of 1.

For reproducibility, use the same data preprocessing, participant-level splitting, checkpoint selection, and evaluation protocol as the study. Exact results may vary with hardware, software versions, and data configuration.

## 9. Contact

Email: 1545931532@qq.com
