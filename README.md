### Introduction

This repository contains a multimodal machine learning model for classifying left ventricular end-diastolic pressure (LVEDP), as described in *"A Multimodal Machine Learning Approach for Identifying Elevated Left Ventricular End-Diastolic Pressure" by Mathilde Verlyck, Debbie Zhao, Edward Ferdian, Stephen Creamer, Gina Quill, Katrina Poppe, Thiranja Prasad Babarenda Gamage, Alistair Young, and Martyn Nash*, presented at the 13th International Conference on Functional Imaging and Modeling of the Heart (FIMH) in Dallas, TX, USA, June 1–5, 2025.

Please cite the following paper if using this work:
> Verlyck MA, Zhao D, Ferdian E, Creamer SA, Quill GM, Poppe KK, Babarenda Gamage TP, Young AA, Nash MP. A Multimodal Machine Learning Approach for Identifying Elevated Left Ventricular End-Diastolic Pressure. Lect Notes Comput Sci (Including Subser Lect Notes Artif Intell Lect Notes Bioinformatics) 2025;15672:231–242. https://doi.org/10.1007/978-3-031-94559-5_21.

Pretrained model weights from the published version are available upon request by contacting [mathilde.verlyck@auckland.ac.nz](mailto:mathilde.verlyck@auckland.ac.nz).

---

### Setup and Requirements

To set up the environment, run:
```bash
conda create -n lvedp_env python=3.10
conda activate lvedp_env
```

Install pytorch, torch geometric and torch geometric temporal following the guides based on your machine specifications:

- [PyTorch installation guide](https://pytorch.org/get-started/locally/)  
- [PyTorch Geometric installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

**Tested with:** CUDA 11.7, PyTorch 2.0.1, torch_geometric 2.6.1
```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch_geometric==2.6.1
pip install torch_geometric_temporal -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

Finally, run:
```bash
pip install -r requirements.txt
```

---

### Training and Inference

**Training:**
```bash
python main.py --name <run_name> --num_epochs <epochs> --batch_size <size> --transform_prob <prob> --train [--cross_val] --modalities <modalities> --input_dir <dataset_path>
```

**Inference:**
```bash
python main.py --name <run_name> [--cross_val] --modalities <modalities> --input_dir <dataset_path>
```

**Arguments:**

- `--name <run_name>`: Name for the run (models saved under `runs/<run_name>`)  
- `--num_epochs <int>`: Number of training epochs (default: 35)  
- `--batch_size <int>`: Batch size (default: 6; adjust based on resources)  
- `--transform_prob <float>`: Probability of data augmentation during training (default: 0.5)  
- `--train`: Include to train the model  
- `--cross_val`: Use if training or evaluating with cross-validation (affects folder structure)  
- `--modalities <list>`: Modalities to use (default: `A2C A4C mesh clinical`): any combination or subset of these modalities  
- `--input_dir <path>`: Path to dataset folder (see below)

---

### Dataset Structure

**Without Cross-Validation:**

```bash
dataset_folder/

├── train/

│   ├── XX003.h5

│   └── ...

├── val/

│   ├── XX007.h5

│   └── ...

└── test/

    ├── XX012.h5

    └── ...
```

**With Cross-Validation:**
```bash
dataset_folder/

├── folds/

│   ├── fold_0/

│   │   ├── train/

│   │   └── val/

│   ├── fold_1/

│   │   ├── train/

│   │   └── val/

│   └── ...

└── test/

    ├── XX012.h5

    └── ...
```

**Each `.h5` file (one per patient) contains:**

- `'A2C_images'`: A2C ultrasound video(s) of the ultrasound cone region adjusted for pixel spacing on a black square background for one cardiac cycle (dict: series → np.array [N, 224, 224, 1])  
- `'A4C_images'`: A4C ultrasound video(s) of the ultrasound cone region adjusted for pixel spacing on a black square background for one cardiac cycle (dict: series → np.array [N, 224, 224, 1])  
- `'meshes'`: 3D mesh of the left ventricle over time (dict: series → np.array [N, 1570, 3])  
- `'edge_index'`: Edge indices for meshes (np.array [2, 9408])  
- `'edge_attr'`: Edge attributes/weights (np.array [9408,])  
- `'measurements'`: Clinical data: gender (0=male, 1=female), age, weight (kg), height (cm), systolic BP (mmHg), diastolic BP (mmHg) (np.array [6,])  
- `'label'`: Ground truth (0=normal, 1=elevated) (np.array [1,])

**Note 1:** All time series are interpolated on the images to have the same number of frames (N) across patients. For meshes with a different number of nodes, update `config/config.yaml` accordingly.

**Note 2:** One A2C series, A4C series, and mesh will be selected randomly (seeded) if multiple are available.

---

### Important Note

This model is experimental and was developed to study the contribution of different input types to LVEDP classification. It does **not** outperform current clinical guidelines and should be used for research purposes only.

---

### Acknowledgements

Pretrained models are available via PyTorch Hub and PyTorch Geometric.  
