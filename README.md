# Project Title

Mobile Price Classification using Machine Learning Models

## Description

This project involves using various machine learning models to classify mobile phones into different price ranges based on their features. We utilize popular libraries such as pandas for data manipulation and scikit-learn for model training and evaluation. The models used include SVC, DecisionTreeClassifier, RandomForestClassifier, and GradientBoostingClassifier. The dataset used is provided in 'train.csv'.

## Getting Started

### Dependencies

* Python 3.x
* pandas
* matplotlib
* scikit-learn

### Installing

1. Clone the repository or download the project files.
2. Ensure you have the required libraries installed. You can install them using pip:
    ```
    pip install pandas matplotlib scikit-learn
    ```

### Executing program

1. Load the dataset:
    ```python
    import pandas as pd
    df = pd.read_csv('train.csv')
    ```

2. Print out the first 5 rows of the dataset:
    ```python
    df.head()
    ```

3. Find the unique values in the price range:
    ```python
    df['price_range'].unique()
    ```

4. Check for null values:
    ```python
    print(df.isnull().sum())
    ```

5. Data Preprocessing:
    ```python
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=['price_range'])
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    ```

6. Get the description of each column in the training data:
    ```python
    X_train.describe()
    ```

7. Train and evaluate models:
    ```python
    from sklearn.metrics import classification_report
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    import matplotlib.pyplot as plt

    def evaluationTest(model):
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

        fig, ax = plt.subplots(figsize=(8, 5))
        cmp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["class_0", "class_1", "class_2", "class_3"])
        cmp.plot(ax=ax)
        plt.show()

    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Support Vector Machine': SVC(kernel='linear', gamma=0.5, C=1.0),
        'Gradient Boosting': GradientBoostingClassifier(),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        print(name)
        evaluationTest(model)
    ```
