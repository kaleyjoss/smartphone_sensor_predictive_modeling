# Smartphone Sensor Predictive Modeling

This repository contains code and resources for building predictive models using smartphone sensor data. The project encompasses data preprocessing, exploratory data analysis (EDA), feature selection, clustering, and various modeling techniques.

## Folder Structure

The repository is organized into the following directories:

- `/scripts`: Contains Python scripts for various stages of the analysis.
  - `feature_selection.py`: Performs feature selection using linear regression.
  - `modeling.py`: General modeling functions and utilities.
  - `variables.py`: Defines and manages variables used throughout the project.
  - `clustering.py`: Handles clustering of demographic data.
  - `preprocessing.py`: Manages data preprocessing tasks such as merging datasets and handling missing values.
  - `visualization.py`: Provides functions for visualizing data and results.

- `/notebooks`: Contains Jupyter notebooks documenting the analysis process.
  - `01_preprocessing.ipynb`: Data preprocessing steps, including merging datasets and imputing missing values.
  - `01_preprocessing_DT.ipynb`: Decision Tree-specific preprocessing steps.
  - `02_var_clustering.ipynb`: Variable clustering analysis.
  - `02_feature_pca.ipynb`: Principal Component Analysis (PCA) for feature extraction.
  - `02_feature_pca_with_missingness.ipynb`: PCA considering missing data patterns.
  - `02_feature_selection.ipynb`: Feature selection techniques and results.
  - `03_demographic_clustering.ipynb`: Clustering analysis based on demographic data.
  - `04_modeling_DT.ipynb`: Decision Tree modeling.
  - `04_modeling_LR.ipynb`: Logistic Regression modeling.
  - `04_modeling_RF.ipynb`: Random Forest modeling.
  - `05_results.ipynb`: Compilation and discussion of modeling results.
 

## Analysis Workflow

The project follows these main steps:

1. **Preprocessing**
   - Merging datasets.
   - Imputing missing values.
   - Encoding missingness based on defined thresholds.

2. **Exploratory Data Analysis (EDA)**
   - Visualizing participant data across different weeks.
   - Assessing variable distributions and missingness.
   - Feature selection using linear regression techniques.

3. **Clustering**
   - Variable clustering to identify related features.
   - Demographic data clustering to segment participants.

4. **Modeling**
   - Applying Decision Trees, Logistic Regression, Random Forests, and Support Vector Machines to build predictive models.
   - Evaluating model performance and comparing results.

5. **Results**
   - Compiling results from various models.
   - Interpreting findings and discussing implications.

## Dependencies

To replicate the analysis, pip install the requirements.tct:

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preprocessing**: Execute the scripts in the `/scripts` directory or run the `01_preprocessing.ipynb` notebook to preprocess the data.

2. **Exploratory Data Analysis**: Use the EDA notebooks (`02_var_clustering.ipynb`, `02_feature_pca.ipynb`, etc.) to explore and visualize the data.

3. **Feature Selection and Clustering**: Apply feature selection methods and perform clustering analyses using the corresponding notebooks.

4. **Modeling**: Run the modeling notebooks (`04_modeling_DT.ipynb`, `04_modeling_LR.ipynb`, etc.) to build and evaluate predictive models.

5. **Results Analysis**: Review the `05_results.ipynb` notebook for a comprehensive analysis of model performance and findings.


---

*Note: The content above is based on the repository's structure and available information. For detailed guidance, refer to the individual scripts and notebooks within the repository.* 
