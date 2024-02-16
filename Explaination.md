

# Naive Bayes Classifier Project 🌿📊

## Project Overview:

- **Objective:** Implement a Naive Bayes classifier using the Iris dataset.
- **Key Steps:**
  - Data loading
  - Exploratory Data Analysis (EDA)
  - Data splitting for training and testing
  - Training a Gaussian Naive Bayes classifier
  - Performance evaluation
  - Decision boundary visualization

1. **Data Gathering:**
   - The Iris dataset is loaded using the `load_iris` function from the `sklearn.datasets` module.
   - The features (X) and target labels (y) are extracted from the dataset.

2. **Exploratory Data Analysis (EDA):**
   - A pairplot is created using seaborn to visualize the relationships between different pairs of features in the Iris dataset. The pairplot is color-coded based on the species of Iris flowers (setosa, versicolor, virginica).
   - The pairplot provides insights into the distribution and relationships of features.
   - ![download](https://github.com/Rutuja-Salunke/Naive-Bayes/assets/102023809/7356a86a-5185-4d55-8df9-0619c27ee0aa)


3. **Data Splitting:**
   - The dataset is split into training and testing sets using the `train_test_split` function from `sklearn.model_selection`.
   - 80% of the data is used for training, and 20% is reserved for testing.

4. **Model Training:**
   - A Gaussian Naive Bayes classifier is employed for training using the `GaussianNB` class from `sklearn.naive_bayes`.
   - The model is trained on the training data.

5. **Model Evaluation:**
   - The trained model is used to predict the target labels for the test set.
   - The accuracy of the model is calculated using the `accuracy_score` function from `sklearn.metrics`.
   - A classification report and confusion matrix are generated using `classification_report` and `confusion_matrix` functions, respectively.
   - The results provide insights into the model's performance on the test data.

6. **Visualization:**
   - A heatmap of the confusion matrix is created using seaborn to visually represent the classification results.
   - Additionally, a decision boundary plot is generated to illustrate how the Naive Bayes classifier separates different classes based on feature values.

7. **Decision Boundary Visualization:**
   - A function (`plot_decision_boundary`) is defined to plot the decision boundary of the Naive Bayes classifier.
   - The decision boundary plot includes a contour plot of the classifier's predictions and a scatter plot of the data points with class labels.
   - ![Screenshot 2024-02-12 100654](https://github.com/Rutuja-Salunke/Naive-Bayes/assets/102023809/6d91a832-ae9c-4eb2-9710-f3265dad600b)


8. **Alternative Decision Boundary Plot:**
   - An alternative decision boundary plot is provided, focusing on the first and fourth features of the Iris dataset for simplicity.
   -  ![Screenshot 2024-02-16 201447](https://github.com/Rutuja-Salunke/Naive-Bayes/assets/102023809/1026afc9-63c2-4fd3-afcf-71d9dc90539e)


## Instructions:

1. **Install Necessary Packages:**
   - Run `pip install scikit-learn matplotlib seaborn` to install required packages.

2. **Run the Code Cells:**
   - Execute provided code cells in the notebook to load data, train the classifier, and visualize performance.
