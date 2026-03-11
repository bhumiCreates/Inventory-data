# Invoice Intelligence

## Overview
**Invoice Intelligence** is a machine learning project that predicts **freight cost** using key invoice attributes such as **purchase quantity** and **invoice dollars**.  
The goal of this project is to analyze invoice data and build a predictive model that estimates freight cost based on purchasing patterns.

The project also includes an **interactive web application built with Streamlit**, allowing users to input invoice details and instantly receive freight cost predictions.

---

## Features
- Freight cost prediction using machine learning
- Interactive web interface built with Streamlit
- Data analysis using Jupyter notebooks
- Model training and evaluation using Scikit-learn
- Organized and modular project structure

---

## Tech Stack
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

---

## Dataset
The project uses a **free dataset containing invoice and purchasing information**, including features such as:

- Purchase quantity
- Invoice dollar value
- Freight cost
- Item and order related attributes

This dataset is used to train a **regression model** to predict freight costs.

---

## Project Structure

```
Invoice-Intelligence/
│
├── app/
│   └── app.py                         # Streamlit application
│
├── Data/
│   ├── F1.ipynb
│   └── Invoice_flagging.ipynb
│
├── Freight_cost_prediction/
│   ├── Data_preprocessing.py
│   ├── model_evaluation.py
│   └── train.py
│
├── inference/
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
│
├── Invoice_flagging/
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   └── train.py
│
├── inventory.db                       
├── dump.sql                           
├── .gitignore
└── README.md
```


---

## Installation

### Clone the repository

```bash
git clone https://github.com/bhumiCreates/invoice-intelligence.git
cd invoice-intelligence
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

Run the Streamlit application locally:

```bash
python -m streamlit run app/app.py
```

After running the command, the app will open in your browser where you can enter invoice details and get **freight cost predictions**.

---

## Model Workflow

1. Data cleaning and preprocessing
2. Exploratory data analysis
3. Feature selection
4. Model training using Scikit-learn
5. Model evaluation using regression metrics
6. Deployment with Streamlit

---

## Future Improvements

- Improve model accuracy using advanced algorithms
- Deploy the application to the cloud
- Add data visualization dashboards
- Implement automated model retraining

---

## Author

**Bhumi**  
Machine Learning & Data Science Enthusiast
