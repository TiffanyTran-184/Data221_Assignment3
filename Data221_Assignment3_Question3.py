import pandas as pd

from sklearn.model_selection import train_test_split

kidney_data = pd.read_csv("kidney_disease.csv")
feature_matrix_x = kidney_data.loc[:, kidney_data.columns != "classification"]
vector_y = kidney_data.loc[:, "classification"]
feature_train, feature_test, y_train, y_test = train_test_split(feature_matrix_x,vector_y,test_size=0.3)
print(feature_train, feature_test, y_train, y_test )


#Why we should not train and test a model on the same data?
#We should not train and test a model on the same data because it can lead to overfitting, where the model memorizes the training data and performs poorly on new or unseen data.

# What the purpose of the testing set is?
# The testing set provides an unbiased evaluation of the model’s performance on data it hasn't seen before,
# helping to estimate how well the model will generalize to real-world data.