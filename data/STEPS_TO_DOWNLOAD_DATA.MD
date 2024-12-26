# How to Download the Dataset

This project uses the Telco Customer Churn dataset from Kaggle. You need to download it manually and place it in the `data` directory.

**Steps:**

1. **Kaggle Account:** Create a free account on [https://www.kaggle.com/](https://www.kaggle.com/).
2. **API Token:**
    *   Go to your Kaggle account settings (click on your profile picture, then "Account").
    *   Scroll down to the "API" section and click "Create New API Token."
    *   This will download a `kaggle.json` file.
3. **Place `kaggle.json`:**
    *   Move the downloaded `kaggle.json` file to the `~/.kaggle/` directory on your local machine (create the directory if it doesn't exist).
    *   On Linux/macOS, make sure it has the correct permissions: `chmod 600 ~/.kaggle/kaggle.json`
4. **Run `download_data.py`:**
    *   Navigate to the `src` directory in your project.
    *   Run the following command:

        ```bash
        python download_data.py paultimothymooney/telco-customer-churn ../data
        ```

    *   This will download the dataset and unzip it into the `data` directory.

**Alternative: Direct Download:**

If you prefer not to use the Kaggle API, you can download the dataset directly from the Kaggle website:

1. Go to [https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
2. Click the "Download" button.
3. Unzip the downloaded file and place the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file into the `data` directory of your project.
