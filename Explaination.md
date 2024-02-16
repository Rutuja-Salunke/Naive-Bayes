

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

## Code Explanation:

### 1. Data Loading:
- Utilizes the `load_iris()` function from scikit-learn to load the Iris dataset.

### 2. Exploratory Data Analysis (EDA):
- Generates a Seaborn pairplot to visually explore relationships between different features for each iris species.

### 3. Data Splitting:
- Splits the dataset into training and testing sets using `train_test_split()` for model evaluation.

### 4. Naive Bayes Classifier:
- Chooses a Gaussian Naive Bayes classifier for its simplicity and effectiveness with continuous features.
- Trains the classifier on the training set to learn underlying patterns.

### 5. Performance Evaluation:
- Calculates model accuracy using `accuracy_score`.
- Displays a classification report and confusion matrix for a comprehensive performance overview on the test set.

### 6. Decision Boundary Visualization:
- Defines a custom function, `plot_decision_boundary`, to visualize the decision boundaries of the Naive Bayes classifier.
- Allows for a clear understanding of how the model separates different classes in the feature space.
- ![Screenshot 2024-02-16 201447](https://github.com/Rutuja-Salunke/Naive-Bayes/assets/102023809/1026afc9-63c2-4fd3-afcf-71d9dc90539e)



### 7. Visualizing Pairplot:
- Displays a Seaborn pairplot for an in-depth visualization of relationships between various features for each iris species.
- Saves the pairplot as an image for reference.
- ![download](https://github.com/Rutuja-Salunke/Naive-Bayes/assets/102023809/7356a86a-5185-4d55-8df9-0619c27ee0aa)

### 8. Visualizing Decision Boundaries:
- Visualizes decision boundaries using the first and fourth features of the Iris dataset.
- Provides insights into how the model distinguishes between different classes.

## Instructions:

1. **Install Necessary Packages:**
   - Run `pip install scikit-learn matplotlib seaborn` to install required packages.

2. **Run the Code Cells:**
   - Execute provided code cells in the notebook to load data, train the classifier, and visualize performance.
