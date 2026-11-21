# Prediction of EV-Battery Health Using Machine Learning

Predict lithium-ion battery **State of Health (SoH)** from discharge-cycle data using a **stacked machine-learning ensemble** and explain the predictions with **SHAP**. Includes an interactive **Gradio** demo.

## Tech Stack

`Python` · `pandas` · `NumPy` · `SciPy` · `scikit-learn` · `XGBoost` · `SHAP` · `Matplotlib` · `Gradio`

## How It Works

1. Load NASA battery-aging discharge records.
2. Denoise voltage, current and temperature signals using a Savitzky–Golay filter.
3. Extract five features: mean voltage, minimum voltage, mean current, maximum temperature, and discharge duration.
4. Train **SVR**, **Random Forest** and **XGBoost**; combine their predictions with a **Ridge regression** meta-learner.
5. Estimate capacity (Ah) and calculate battery SoH:

   ```text
   SoH (%) = predicted capacity (Ah) / 2.0 Ah × 100
   ```

6. Explain predictions with SHAP and display them in Gradio.

## Getting Started

### 1. Download the notebook

This project runs from the original Google Colab notebook:

```text
notebooks/EV_battery_camera_ready_experiments.ipynb
```

> **Required:** Add the actual `.ipynb` notebook to this repository. A PDF or screenshot of notebook cells cannot execute.

### 2. Install dependencies

In a Colab code cell:

```python
%pip install numpy pandas scipy scikit-learn xgboost shap matplotlib gradio
```

Or in a terminal (for a local Python environment):

```bash
python -m pip install -r requirements.txt
```

### 3. Prepare the dataset

Download the **NASA Ames Li-ion Battery Aging Dataset** and create this folder in your Google Drive:

```text
MyDrive/
└── NASA_Battery_Data/
    ├── metadata.csv
    └── data.zip
```

The notebook expects these exact paths:

```python
drive_folder_name = 'NASA_Battery_Data'
zip_path = f'/content/drive/MyDrive/{drive_folder_name}/data.zip'
metadata_path = f'/content/drive/MyDrive/{drive_folder_name}/metadata.csv'
```

The ZIP must contain the cycle CSV files in the expected `data/` subfolder. Data is **not** included in this repository.

### 4. Run the project

1. Open the `.ipynb` file in [Google Colab](https://colab.research.google.com/).
2. Choose **Runtime → Run all**.
3. Allow Google Drive access when asked.
4. Let the notebook extract data, train the models, evaluate the predictions and generate SHAP plots.
5. Run the **Gradio dashboard** cell. Open the URL printed by `dashboard.launch()` to view the interface.

**No compilation is needed**—this is a Python notebook. Execute its cells in order.

## Results

Results shown in the provided Colab run for the **B0005 chronological split** (134 training cycles; 34 test cycles):

| Metric | Reported result |
|---|---:|
| R² | **0.9841** |
| RMSE | **0.0028 Ah** |

The notebook also evaluates **B0005, B0006, B0007, and B0018** using leave-one-battery-out validation. These are results of the documented run, not newly reproduced benchmarks.

### Expected Outputs

- **Predicted vs. measured capacity plot** for the B0005 test cycles.
- **Model comparison table** with MAE, RMSE and R².
- **Cross-battery evaluation table** for B0005/B0006/B0007/B0018.
- **SHAP summary and local explanation plots** showing feature contributions.
- **Gradio dashboard** displaying predicted capacity, SoH percentage, a status label and a SHAP chart.

Example shown in the supplied notebook screenshot: **1.7673 Ah predicted capacity** and **88.36% SoH** for manually entered features. Outputs vary with the entered values and trained model.

## Repository Structure

```text
EV-Battery-Health-ML/
├── README.md
├── requirements.txt
├── notebooks/
│   └── EV_battery_camera_ready_experiments.ipynb
├── figures/
└── results/
```

## Notes

- The Gradio UI is a **manual-input software demonstrator**, not a live vehicle/BMS connection.
- The features are calculated from **complete discharge cycles**; partial-cycle or real-world use requires further validation.
- The Colab notebook uses Google Drive and `/content/` paths, so local execution requires adapting those paths.
- Do not upload private Google Drive credentials or dataset archives without checking redistribution permissions.

## Project Team

**Anurag University — Department of Computer Science and Engineering**  
**Guide:** Dr. G. Bala Krishna

Kandagatla Anoop · Nallamothu Venkata Charan · Thogiti Manideep · Kotakonda Jai Raj Maharishi · Yedula Vijay Eshwar
