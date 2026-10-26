# ALGE-Net

ALGE-Net (Adaptive Layer-wise Gated Edge-Fusion Network) is a deep learning framework for fetal gestational age estimation (FGA) from magnetic resonance imaging (MRI).

This repository contains the model implementation, training scripts, dataset loading code, and prediction and evaluation scripts.

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

## 2. Environment

- Python: 3.12
- PyTorch: 2.11.0+cu128
- torchvision: 0.26.0+cu128
- CUDA runtime: 12.8
- GPU acceleration: NVIDIA CUDA (available)

### Installation

Install PyTorch and torchvision compatible with your Python and CUDA environment. The versions listed above correspond to the current development environment.

For installation instructions, refer to the official PyTorch website:

https://pytorch.org/get-started/locally/

Install the remaining dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## 3. Data Preparation

The original MRI dataset is not included in this repository. All data must be handled in accordance with the applicable ethics approval, data-use permissions, and institutional requirements. Do not upload restricted participant data to this public repository.

The current CSV file is named `newdataWindowU1.csv` and contains 1,630 image records associated with 207 unique participant identifiers.

The CSV file contains the following fields:

| Field | Description |
|---|---|
| `fs` | Participant identifier |
| `image` | MRI image path or image identifier |
| `XMIN` | X-coordinate field of the ROI bounding box |
| `YMIN` | Y-coordinate field of the ROI bounding box |
| `XMAX` | X-coordinate field of the ROI bounding box |
| `YMAX` | Y-coordinate field of the ROI bounding box |
| `week` | Gestational age in weeks |
| `day` | Gestational age in days |
| `split` | Dataset split field |

An example CSV record is shown below:

```csv
fs,image,XMIN,YMIN,XMAX,YMAX,week,day,split
100009662,E:\...\I1900000,187,339,304,234,35,251,
```

The image path shown above has been abbreviated and should be replaced with the actual path accessible on your local system. The ROI coordinate fields must be consistent with the cropping conventions implemented in the data loading code.

**Note:** The `split` column in the provided CSV file is currently empty. According to the study protocol, the dataset is split at the participant level, with 164 participants assigned to the training set and 43 assigned to the test set. No participant should appear in both sets. Before running the scripts, verify how the data split is generated or loaded to prevent participant-level data leakage.

Data preprocessing includes DICOM image loading, fetal brain ROI cropping, grayscale conversion, and image resizing. The model input size is 96 × 96 pixels. The exact preprocessing procedure is determined by the implementation in the source code.

## 4. Model Training

The training entry point is `train.py`.

The main training settings reported in the paper are as follows:

- Number of epochs: 350
- Optimizer: Adam
- Initial learning rate: 0.001
- Batch size: 32
- Weight decay: 0.00001
- Learning rate scheduler: StepLR, reducing the learning rate to 10% of its previous value after epoch 300
- Random seeds: 1–10

Example command:

```bash
python train.py --data_dir "/path/to/data" --dataset "newdataWindowU1" --epochs 350 --batch_size 32 --lr 0.001
```

Replace the example path with the actual local data directory. Before training, verify that the model checkpoint output path is valid and that the destination directory exists.

## 5. Prediction and Evaluation

The prediction and evaluation entry point is `predict.py`.

Example command:

```bash
python predict.py --data_dir "/path/to/data" --dataset "newdataWindowU1" --weight_dir "/path/to/checkpoints" --output_dir "results" --device auto --seeds 1,2,3,4,5,6,7,8,9,10
```

Adjust the data directory and checkpoint paths according to your local environment. The actual command-line arguments and checkpoint naming conventions should be verified against the source code.

The paper uses horizontal-flip test-time augmentation (TTA), averaging predictions from the original and horizontally flipped images. Slice-level predictions are then aggregated to obtain sequence-level gestational age estimates. The main experimental results are reported as the mean and standard deviation across 10 runs.

## 6. Reproducibility

The experiments reported in the paper use random seeds 1–10. Random states for Python, NumPy, PyTorch CPU, and CUDA are controlled. CuDNN deterministic mode is enabled, and benchmarking is disabled.

Model complexity profiling is performed using an NVIDIA RTX 4090 GPU, with an input size of 96 × 96 pixels and a batch size of 1.

For reproducibility, ensure that data preprocessing, participant-level dataset splitting, checkpoint selection, and evaluation procedures are consistent with those used in the paper.

## 7. Contact

Email: 1545931532@qq.com