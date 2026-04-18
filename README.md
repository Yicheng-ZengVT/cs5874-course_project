# CS 5874 

## Starting up the codes

to clone this repo:

```bash
git clone https://github.com/Yicheng-ZengVT/cs5874-course_project
```

Please check the [original repo](https://github.com/tientrandinh/revisiting-reverse-distillation)
original repo if you run into any trouble.

## Setup dependencies

Run this to install requirements: `geomloss` and `numba`:

```python
pip install -r requirements.txt
```
## Handling data

Please visit the [dataset website](https://www.mvtec.com/research-teaching/datasets) for the raw dataset, and put it in this folder: `dataset/datasets`, it will be ignore by git so you don't need to refresh it everytime.

## Quick Start:

Use [google colab](https://colab.research.google.com/github/tientrandinh/Revisiting-Reverse-Distillation/blob/main/main.ipynb) to run the code(don't need the repo to do so).

Or run `main.ipynb` in the environment, if you have virtual environment enabled, remember to run something like:

```bash
python -m venv .cs5874_venv
source .cs5874_venv/bin/activate
```

Such that your global python env is not polluted.