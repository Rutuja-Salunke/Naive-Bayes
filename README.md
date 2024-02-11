# Naive Bayes Classifier Project 🌿📊

## Project Overview:
This project implements a Naive Bayes classifier on the Iris dataset, including data loading, exploratory data analysis (EDA), model training, performance evaluation, and decision boundary visualization.

## Code Explanation:
1. **Data Loading:**
   - The Iris dataset is loaded using scikit-learn's `load_iris()` function.

2. **Exploratory Data Analysis (EDA):**
   - A pairplot is created using Seaborn to visualize relationships between features for each iris species.

3. **Data Splitting:**
   - The dataset is split into training and testing sets using `train_test_split()`.

4. **Naive Bayes Classifier:**
   - A Gaussian Naive Bayes classifier is instantiated and trained on the training set.

5. **Performance Evaluation:**
   - Model accuracy is calculated using `accuracy_score`, and additional metrics such as the classification report and confusion matrix are displayed.

6. **Decision Boundary Visualization:**
   - A function `plot_decision_boundary` is defined to visualize the decision boundaries of the Naive Bayes classifier.

7. **Visualizing Pairplot:**
   - The pairplot is displayed and saved as an image.

8. **Visualizing Decision Boundaries:**
   - The decision boundaries are visualized using the first and fourth features.

## Instructions:
1. Install necessary packages: `pip install scikit-learn matplotlib seaborn`.

2. Run the code cells in the provided notebook.

3. Explore the EDA visualizations, model performance, and decision boundaries.


