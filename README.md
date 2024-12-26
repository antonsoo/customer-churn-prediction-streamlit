# Customer Churn Prediction with Machine Learning

This project focuses on predicting customer churn using machine learning techniques. The project utilizes a Telco customer churn dataset to build a predictive model and deploys it as an interactive web application using Streamlit.

## Project Description

Customer churn, the rate at which customers stop doing business with an entity, is a critical metric for many businesses. This project aims to predict customer churn using machine learning, enabling businesses to take proactive steps to retain customers.

The project follows an end-to-end machine learning workflow:

1. **Data Loading and Preprocessing:** The dataset is loaded, cleaned, and preprocessed to handle missing values and prepare the data for modeling.
2. **Feature Engineering:** New features are created and categorical features are encoded to improve the model's performance.
3. **Model Selection and Training:** Several machine learning models are trained and evaluated, including Logistic Regression, Random Forest, and XGBoost. The best-performing model is selected based on the evaluation metrics.
4. **Model Evaluation:** The selected model is evaluated using a held-out test set, and performance metrics such as accuracy, precision, recall, F1-score, and AUC are reported.
5. **Web Application Development:** An interactive web application is built using Streamlit, allowing users to input customer data and receive churn predictions in real time.
6. **Deployment:** The Streamlit application is deployed on Hugging Face Spaces for easy access and demonstration.

## Demo

Try out the live demo of the churn prediction app on Hugging Face Spaces: \[Link to your Hugging Face Spaces app] (Replace with the actual link when you deploy it)

## Dataset

The project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle. The dataset contains information about customer demographics, services used, contract details, and whether or not the customer churned.

### Data Download

This repository does not include the full dataset due to its size. To download the dataset, please follow the instructions in the `data/HOW_TO_DOWNLOAD_DATA.md` file. This typically involves:

1. Creating a Kaggle account.
2. Downloading a Kaggle API token (`kaggle.json`).
3. Using the `download_data.py` script provided in the `src` directory.

## Installation

1. Clone the repository:

    ```bash
    git clone [Your GitHub repository URL]
    ```

2. Navigate to the project directory:

    ```bash
    cd customer-churn-prediction
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

**To run the Streamlit app locally:**

1. Make sure you have the trained model (`churn_prediction_model.pkl`) and the preprocessor (`preprocessor.pkl`) in the `model` directory. You can obtain these by running the training script (`train.py`) or downloading them from the project's releases (if available).
2. Run:

    ```bash
    streamlit run src/app.py
    ```

**To train the model:**

1. Download the dataset using the instructions in `data/HOW_TO_DOWNLOAD_DATA.md` and place it in the `data` directory.
2. Run the training script:

    ```bash
    python src/train.py
    ```

    This will train the model and save the model weights and preprocessor to the `model` directory.

## Model Performance

The best-performing model (XGBoost in this case) achieved the following results on the test set:

*   **Accuracy:** \[Insert your test accuracy]
*   **Precision:** \[Insert your test precision]
*   **Recall:** \[Insert your test recall]
*   **F1-score:** \[Insert your test F1-score]
*   **AUC:** \[Insert your test AUC]

**Confusion Matrix:**

\[Insert your confusion matrix visualization here]

**Training and Validation Curves:**

\[Insert your training/validation loss and accuracy plots here]

## Project Structure

```
customer-churn-prediction/
├── data/
│ └── HOW_TO_DOWNLOAD_DATA.md (Instructions for downloading the dataset)
├── model/
│ └── churn_prediction_model.pkl (Saved model weights)
│ └── preprocessor.pkl (Saved preprocessor object)
├── notebooks/
│ └── customer_churn_prediction.ipynb (Colab notebook for exploration and development)
├── src/
│ ├── train.py (Training script)
│ ├── evaluate.py (Evaluation script - if applicable)
│ ├── app.py (Streamlit application script)
│ └── download_data.py (Script to download data from Kaggle)
│ └── utils.py (Utility functions - if applicable)
├── README.md
├── requirements.txt
└── LICENSE
```

## Contributing

Contributions to this project are welcome! Feel free to submit pull requests or open issues to suggest improvements or report bugs.

## License

This project is licensed under the \[Your Chosen License] - see the [LICENSE](LICENSE) file for details.

## Contact

[Your Name] - [Your Email or Upwork Profile URL]

## Acknowledgements

*   [Kaggle](https://www.kaggle.com/) for providing the Telco Customer Churn dataset.
*   [Streamlit](https://streamlit.io/) for the easy-to-use web app framework.
*   [Hugging Face](https://huggingface.co/) for the Transformers library and the Spaces platform.
*   [XGBoost](https://xgboost.readthedocs.io/en/stable/) for the efficient gradient boosting algorithm.

