# CoinageClusterML
Machine Learning framework for predicting binding energies and magic numbers in Cu, Ag, and Au nanoclusters (N ≤ 55) using the Open Quantum Cluster Database (OQCD). Features 7 ensemble models, SHAP interpretability, and 9 geometric descriptors.
Machine Learning Prediction of Coinage Metal Nanocluster Stability
Cu–Ag–Au Clusters (N ≤ 55)
________________________________________
Overview
This repository contains the full computational framework developed for the research study:
Machine Learning Prediction of Coinage Metal Nanocluster Stability and Magic Numbers
The project integrates Density Functional Theory (DFT) datasets, geometric descriptors, and advanced machine learning models to predict the binding energy per atom and identify magic-number stability patterns in coinage metal nanoclusters.
The investigated systems include clusters composed of:
•	Copper (Cu) 
•	Silver (Ag) 
•	Gold (Au) 
for cluster sizes:
N ≤ 55 atoms
The workflow provides a scalable approach for predicting nanocluster stability while significantly reducing the computational cost associated with large-scale DFT calculations.
________________________________________
Scientific Motivation
Metal nanoclusters exhibit strong size-dependent stability patterns, often referred to as magic numbers. These clusters play an important role in:
•	heterogeneous catalysis 
•	plasmonic nanomaterials 
•	quantum nanostructures 
•	energy conversion systems 
•	nanoscale electronic devices 
Traditional approaches rely on exhaustive Density Functional Theory calculations, which become computationally prohibitive for large datasets.
This project introduces a machine learning assisted framework capable of:
•	learning stability trends from DFT data 
•	predicting binding energies with high accuracy 
•	identifying stability peaks (magic numbers) 
•	analyzing structure–energy relationships 
________________________________________
Methodology
The computational workflow consists of five major stages.
________________________________________
1 Dataset Construction
The dataset contains DFT-computed structures and properties for Cu, Ag, and Au clusters.
Each entry includes:
•	atomic coordinates 
•	cluster size (N) 
•	metal type 
•	HOMO–LUMO gap 
•	magnetic moment 
•	number of valence electrons 
•	total DFT energy 
The input dataset is provided as:
CuAgAu.xlsx
________________________________________
2 Feature Engineering
Atomic coordinates are converted into physically meaningful geometric descriptors that capture cluster morphology.
Computed descriptors include:
•	mean radial distance 
•	distance standard deviation 
•	maximum radial distance 
•	radius of gyration 
•	asphericity index 
•	bounding box dimensions 
•	compactness index 
These descriptors quantify:
•	structural compactness 
•	geometric anisotropy 
•	cluster spatial distribution 
Such features are critical for linking cluster geometry with thermodynamic stability.
________________________________________
3 Target Variable
The machine learning models predict the binding energy per atom, defined as:
BE_per_atom = E_DFT / N_atoms
where:
•	E_DFT is the total DFT energy 
•	N_atoms is the number of atoms in the cluster 
This normalization enables direct comparison across clusters of different sizes.
________________________________________
4 Machine Learning Models
Seven machine learning algorithms are evaluated and compared.
Model	Category
ExtraTrees	Ensemble Learning
RandomForest	Ensemble Learning
XGBoost	Gradient Boosting
LightGBM	Gradient Boosting
CatBoost	Gradient Boosting
GradientBoosting	Ensemble Boosting
Neural Network (MLP)	Deep Learning
The dataset is divided using a stratified train-test split:
Training: 85%
Testing: 15%
Model performance is evaluated using:
•	Mean Absolute Error (MAE) 
•	Coefficient of Determination (R²) 
•	Training time 
The best-performing model is automatically selected.
________________________________________
5 Model Interpretability
To understand the physical drivers of stability, the framework includes:
•	global feature importance analysis 
•	SHAP (SHapley Additive exPlanations) 
•	learning curves 
•	residual analysis 
These tools reveal the contribution of geometric and electronic descriptors to cluster stability.
________________________________________
6 Magic Number Detection
Cluster stability is further analyzed using the second energy difference criterion:
Δ²E stability metric.
Clusters satisfying the condition:
Δ²E > threshold
are identified as magic-number clusters, corresponding to enhanced thermodynamic stability.
The framework compares:
•	literature-reported magic numbers 
•	newly discovered stability peaks 
________________________________________
Generated Outputs
Running the analysis automatically generates a complete research dataset including:
________________________________________
Figures
The pipeline produces 26 publication-quality figures, including:
•	ML vs DFT parity plot 
•	cluster stability vs size 
•	feature importance ranking 
•	SHAP interpretability analysis 
•	learning curves 
•	magic number stability peaks 
•	PCA projection 
•	t-SNE manifold visualization 
•	residual distribution 
•	energy density distributions 
•	geometric descriptor evolution 
•	3D stability landscape 
All figures are saved in:
/Figures
________________________________________
Tables
Three manuscript-ready tables are generated:
1.	Machine learning model performance comparison 
2.	ML predictions vs DFT energies 
3.	Magic number stability comparison 
Tables are stored in:
/Tables
________________________________________
Automatic Manuscript Report
The script generates a formatted report:
Full_Paper_Report_N55_DATE.docx
The report contains:
•	model performance summary 
•	ML vs DFT comparisons 
•	magic number table 
This file can be directly used in manuscript preparation.
________________________________________
Repository Structure
IMPERIAL_FINAL_2026_N55/

│
├── Figures/
│   ├── Fig_01_Parity_Plot.png
│   ├── Fig_02_Stability_vs_Size.png
│   ├── ...
│   └── Fig_26_Full_Correlation_Matrix.png
│
├── Tables/
│
├── Full_Paper_Report_N55_DATE.docx
│
├── CuAgAu.xlsx
│
└── atomic_analysis_suite.py
________________________________________
Installation
Python version:
Python ≥ 3.9
Install dependencies:
pip install numpy pandas matplotlib seaborn scikit-learn
pip install xgboost lightgbm catboost shap statsmodels python-docx
________________________________________
Running the Code
Execute the analysis pipeline:
python atomic_analysis_suite.py
The script will automatically generate:
•	figures 
•	tables 
•	manuscript report 
•	compressed ZIP archive of results 
________________________________________
Scientific Impact
This work demonstrates that machine learning models trained on DFT data can:
•	accurately reproduce nanocluster stability trends 
•	identify size-dependent stability patterns 
•	reveal structure–energy relationships 
•	detect magic-number clusters 
The framework provides a scalable alternative to expensive high-throughput DFT screening.
________________________________________
Applications
The methodology can be extended to:
•	alloy nanoclusters 
•	catalytic nanoparticles 
•	metallic nanoalloys 
•	high-throughput materials discovery 
•	AI-driven nanomaterials design 
________________________________________
Reproducibility
All data preprocessing, feature engineering, model training, and visualization steps are fully automated within the provided codebase to ensure complete reproducibility of results.
________________________________________
Author
Researcher in:
•	Computational Chemistry 
•	Machine Learning for Materials 
•	Nanocluster Physics 
•	Nanostructured Materials Modeling 
________________________________________
License
This repository is released under the MIT License.
________________________________________
إ

