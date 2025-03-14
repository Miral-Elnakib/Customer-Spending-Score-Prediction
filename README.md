# Customer-Spending-Score-Prediction
# Customer Spending Score Prediction

Predicting customer spending behavior using machine learning regression techniques.

## Overview
This project implements a **Customer Spending Score Prediction System** using regression models to analyze and predict customer spending behavior based on demographic and financial features. The dataset includes attributes such as **Gender, Age, and Annual Income**, with the target variable being **Spending Score (1-100)**. Two approaches are explored:

- **Neural Network** built with TensorFlow
- **Linear Regression** model from scikit-learn

Performance is evaluated using metrics like **Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE)**.

The project is implemented in **Python** using a **Jupyter Notebook**, leveraging libraries like **pandas, NumPy, Matplotlib, and scikit-learn** for data processing, visualization, and modeling.

## Features
### Data Preprocessing
- Loads and cleans the dataset (`Project_Idea_5.csv`).
- Encodes categorical variables (e.g., **Gender: Male=1, Female=0**).
- Drops unnecessary columns (e.g., **CustomerID**).

### Exploratory Data Analysis (EDA)
- Visualizes data distributions and relationships.

### Modeling
- **Neural Network**: A custom-built model using TensorFlow/Keras for regression.
- **Linear Regression**: A baseline model using scikit-learn.

### Evaluation
- Compares model performance using **MSE, MAE, and RMSE**.
- Visualizes **training/validation loss** and **predictions vs. true values**.

### Visualization
- Plots **model loss over epochs**.
- Displays **scatter plots of predictions**.

## Dataset
The dataset (`Project_Idea_5.csv`) contains **200 customer records** with the following columns:

- **CustomerID**: Unique identifier (dropped during preprocessing).
- **Gender**: Male or Female (encoded as **1 or 0**).
- **Age**: Customer age (18–70).
- **Annual Income (k$)**: Yearly income in thousands (15–137).
- **Spending Score (1-100)**: Target variable, representing spending behavior (1–99).

📌 **Ensure the dataset file is placed in the project directory before running the notebook.**

## Requirements
To run this project, install the following dependencies:
```bash
pip install -r requirements.txt
```

### Key Libraries
- `pandas`: Data manipulation and analysis.
- `numpy`: Numerical computations.
- `matplotlib & seaborn`: Data visualization.
- `scikit-learn`: Linear regression and evaluation metrics.
- `tensorflow`: Neural network modeling.
- `plotly`: Interactive visualizations.

## Installation
### Clone the Repository:
```bash
git clone https://github.com/yourusername/spending-score-prediction.git
cd spending-score-prediction
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```

### Add the Dataset:
Place `Project_Idea_5.csv` in the project root directory.

### Run the Notebook:
```bash
jupyter notebook spending_score_prediction.ipynb
```

## Usage
### Open the Notebook:
Launch `spending_score_prediction.ipynb` in **Jupyter Notebook**.

### Execute Cells:
Run the cells sequentially to:
- Load and preprocess the data.
- Perform **EDA**.
- Train and evaluate the **neural network and linear regression models**.
- Visualize results.

### View Results:
- **Loss curves** for the neural network (**train vs. test**).
- **Scatter plots** comparing true vs. predicted spending scores.

## Example Output
### Neural Network Metrics:
```bash
Loss: 0.2441, MAE: 0.3602, MSE: 0.2441
```

### Linear Regression Metrics:
```bash
MSE: 2.60e-28, MAE: 1.25e-14, RMSE: 1.61e-14
```

## Project Structure
```
spending-score-prediction/
│
├── spending_score_prediction.ipynb  # Main Jupyter Notebook
├── Project_Idea_5.csv               # Dataset (add manually)
├── requirements.txt                 # Dependencies
├── README.md                        # Project documentation
└── images/                          # Output plots (optional)
    ├── model_loss.png               # Loss curve plot
    ├── predictions_scatter.png      # Predictions scatter plot
```

## Results
### Neural Network:
✅ Achieves a **reasonable fit** with an **MSE of 0.2441** and **MAE of 0.3602** on the test set.
✅ Loss decreases steadily over epochs (**see model_loss.png**).

### Linear Regression:
✅ **Outperforms** the neural network with **near-perfect predictions** (**MSE ≈ 2.6e-28, MAE ≈ 1.25e-14**).
✅ Suggests a **strong linear relationship** in the data.

### Visualizations:
📈 **Loss Curve**: Shows training and validation loss convergence.
📊 **Scatter Plot**: Compares true vs. predicted spending scores for both models (**red = linear, blue = neural network**).

## Future Enhancements
🔹 Add **feature engineering** (e.g., interaction terms, polynomial features).
🔹 Experiment with **more complex neural network architectures** (e.g., deeper layers, dropout).
🔹 Incorporate **cross-validation** for robust evaluation.
🔹 Develop a **web interface** for interactive predictions using **Flask or Streamlit**.

## Contributing
Contributions are welcome! To contribute:
1. **Fork** the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a **Pull Request**.

## 📜 License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## 🙌 Acknowledgments
- **Dataset**: Inspired by common customer segmentation datasets (e.g., Mall Customer Dataset).
- **Libraries**: Thanks to the open-source communities behind **TensorFlow, scikit-learn, and Matplotlib**.
- **Inspiration**: Machine learning regression tutorials and Kaggle kernels.
