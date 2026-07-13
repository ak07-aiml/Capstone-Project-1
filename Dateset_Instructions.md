# Dataset

This project uses the **India Global Migration Dataset** available on Kaggle.

Dataset:
https://www.kaggle.com/datasets/aryanmdev/india-global-migration-dataset

The dataset is downloaded automatically using the `kagglehub` package.

Install the required package:

```bash
pip install kagglehub

The dataset can be downloaded using:

import kagglehub

# Download dataset
path = kagglehub.dataset_download(
    "aryanmdev/india-global-migration-dataset"
)

print(path)

The script automatically locates the CSV file inside the downloaded dataset folder.


---

## Also include Installation Instructions

```markdown
# Installation

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn kagglehub