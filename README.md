# Diabetes Dataset Linear Regression

The diabetes_dataset branch exemplifies linear regression using the Diabetes dataset, focusing on predicting disease progression based on various physiological factors.

## Data Analysis

The analysis begins with exploring correlations among the dataset's features. Using Pandas and Matplotlib, a correlation matrix and scatter plots are generated to visualize relationships between different physiological attributes.

The most notable correlations are:
- Between 'total serum cholesterol' (s1) and 'low-density lipoproteins' (s2)
- Between 'body mass index' (bmi) and 'high-density lipoproteins' (s3)

## Linear Regression Insights

### Basic Regression
The dataset is split into training and test sets for implementing linear regression. On average, the model displays a mean absolute percentage error of 2.26 without excluding outliers.

### Outlier Handling
Another regression model is built after removing outliers, resulting in a reduced mean absolute percentage error of 1.70. Additionally, substituting outliers with mean values yields an error rate of 1.99.

### Feature Engineering
Further analysis involves creating new feature sets derived from the existing data, enhancing the predictive capacity of the model. These new features help improve the model's accuracy, reducing the error to 1.75.

## Launching the Project

To run this project, utilize Google Colab via the provided [Gist link](https://gist.github.com/RobertNeat/edf519473806a86a31fb4996310d8518). Alternatively, download the repository branch and execute the project locally using DataSpell, PyCharm, or Spyder IDE.

Make sure to have the dataset files within your project directory to ensure smooth execution.

## Conclusion

Through comprehensive data analysis and iterative model enhancements, the diabetes dataset branch showcases the iterative process of refining linear regression models for predicting disease progression. The insights gained from correlation analysis and feature engineering contribute to optimizing the model's predictive accuracy.

The Jupyter notebook or Python script in this branch serves as a practical demonstration of linear regression techniques applied to real-world healthcare datasets.
