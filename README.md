# ECGR 4106 Homework 5

**Name:** Samantha Gonzalez  
**Student ID:** 801353957

**Course:** ECGR 4106 - Introduction to Deep Learning  
**Assignment:** Homework 5

## Vision Transformers for Image Classification

This repository contains my Homework 5 code and results for ECGR 4106. The assignment focuses on Vision Transformers for CIFAR-100 image classification.

## Contents

- `ECGR_4106_Homework_5.ipynb`  
  Main Jupyter notebook with all cells executed.

- `results/`  
  CSV result tables and generated plots from the experiments.

- `code/homework5_colab_solution.py`  
  Clean Python script version of the notebook code.

## Problems Covered

### Problem 1
Vision Transformer from scratch compared against ResNet-18 on CIFAR-100.

### Problem 2
Fine-tuning pretrained Swin-Tiny and Swin-Small models compared against a Swin Transformer trained from scratch.

## How to Run

The notebook was designed to run in Google Colab using a GPU runtime.

Recommended runtime:

- Python 3
- L4 GPU, A100 GPU, or T4 GPU

Run the notebook from top to bottom. The code downloads CIFAR-100 automatically using `torchvision.datasets.CIFAR100`.

## Results

The result CSV files and plots are saved in the `results/` folder. These outputs are included so the reported results can be reviewed and compared with the notebook output.
