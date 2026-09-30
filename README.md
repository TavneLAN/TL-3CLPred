# TL-3CLPred

TL-3CLPred is a protease cleavage site prediction workflow for dataset inspection, background model training, transfer learning, and ensemble prediction.

## Environment

This project targets a legacy deep-learning environment:

- Python 3.6
- Keras 2.1.6
- TensorFlow-GPU 1.13.1
- NumPy 1.19.2
- pandas 0.25.1
- matplotlib 3.2.2
- CUDA Toolkit 10.0.130
- cuDNN 7.6.5

Create the Conda environment:

```powershell
conda env create -f environment-legacy.yml
conda activate cleavage-keras216
python check_environment.py
```

Dataset inspection and syntax checks can run without TensorFlow. Training, prediction, and `smoke-test` require the legacy Keras/TensorFlow environment.

## Input Format

Input files are FASTA-like text files. A positive cleavage site is marked with `#` immediately after the residue before the cleavage site:

```text
>protein_1
MABCDEQ#FGHIK
```

In this example, `Q` is labeled as a positive site. Candidate residues without `#` are treated as negative sites.

## Basic Check

```powershell
python -m compileall -q .
python run.py inspect `
  --input pre_train_lable.txt `
  --focus Q `
  --radius 5 `
  --encoding physchem-v1 `
  --preview 5
```

## Train Background Model

```powershell
python run.py train-base `
  --input pre_train_lable.txt `
  --output-prefix runs\3cl_base `
  --focus Q `
  --radius 5 `
  --encoding physchem-v1 `
  --models 1 `
  --epochs 100 `
  --batch-size 256 `
  --patience 15 `
  --seed 42
```

## Transfer Learning

```powershell
python run.py train-transfer `
  --input 3CL_training_lable.txt `
  --background-prefix runs\3cl_base `
  --output-prefix runs\3cl_transfer `
  --focus Q `
  --models 1 `
  --unfreeze-last 4 `
  --epochs 60 `
  --batch-size 256 `
  --patience 10 `
  --seed 42
```

## Prediction

```powershell
python run.py predict `
  --input 3CL_independent_test_lable.txt `
  --model-prefix runs\3cl_transfer `
  --output runs\3cl_test_predictions.tsv `
  --threshold 0.5 `
  --batch-size 512
```

## Reference

This project was developed with reference to the DeepCleave study:

F. Li et al., "DeepCleave: a deep learning predictor for caspase and matrix metalloprotease substrates and cleavage sites," Bioinformatics, 36(4), 1057-1065, 2020. https://doi.org/10.1093/bioinformatics/btz721

## License

This repository does not currently include a `LICENSE` file.
