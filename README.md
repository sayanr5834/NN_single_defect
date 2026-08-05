# Data & Code — *A spectral viewpoint on the single defect tight-binding chain*


This repository contains the data and code used to generate the data and figures for:

> **A spectral viewpoint on the single defect tight-binding chain**  
> *Sayan Roy*  

---

## Repository Contents

- `functions.py` — Core functions needed for simulations.
- `build.py` — Main script containing functions for data generation and visualization.
- `plotting.ipynb` — Demonstration notebook that loads pre-generated data and reproduces the paper’s figure using `build.py`.
- `Data/` — CSV data for all figures (`Figure2a.csv` … `FigureS10.csv`).
- `Figures/` — Final figure PDFs (`Figure2a.pdf` … `FigureS10.pdf`).

---

## Installation

Tested with **Python 3.12.3**.

### Create and activate a virtual environment (recommended)

**Linux/macOS**
```bash
python3 -m venv venv
source venv/bin/activate

#Install packages manually
pip install numpy scipy matplotlib pandas joblib tqdm jupyterlab ipykernel
```
### Demo Notebook 

The notebook `plotting.ipynb` demonstrates how to load, analyze, and visualize the data corresponding to each figure in the article.

The data for the published parameter values is already provided in the `Data/` directory.

Each figure is plotted in a separate cell for clarity.

Users may optionally regenerate data by calling the appropriate functions from build.py (instructions are included inside the notebook).

To open the notebook:

```
jupyter notebook plotting.ipynb
```

### Data Availability

All data and code are permanently archived on Zenodo:

### License
- Code: MIT License (see `LICENSE`)
- Data: [![CC BY 4.0][cc-by-shield]][cc-by] (see `LICENSE-DATA`)


[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

### Contact
For questions or issues, please contact **Sayan Roy**: <Sayan Roy> sayan.roy@physik.uni-saarland.de or **Emma King**:<Emma King> emma.king@physik.uni-saarland.de
