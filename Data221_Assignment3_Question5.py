import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

kidney_data = pd.read_csv("kidney_disease.csv")
feature_matrix_x = kidney_data.loc[:, kidney_data.columns != "classification"]
label_y = kidney_data.loc[:, "classification"]

feature_x_train, feature_x_test, label_y_train, label_y_test = train_test_split(feature_matrix_x, label_y, test_size=0.3)
k_values = [1, 3, 5, 7, 9]
accuracy_results = []

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(feature_x_train, label_y_train)
    label_y_predict = knn_model.predict(feature_x_test)
    test_accuracy = accuracy_score(label_y_test, label_y_predict)
    accuracy_results.append(test_accuracy)

accuracy_table = pd.DataFrame({"k": k_values, "Test Accuracy": accuracy_results})
print(accuracy_table)

# Identify the k with the highest test accuracy
best_index = accuracy_results.index(max(accuracy_results))
best_k = k_values[best_index]
print(f"Highest test accuracy is achieved at k = {best_k}")

# Changing k affects the smoothness of the decision boundary: smaller k produces more flexible boundaries, while larger k produces smoother boundaries.
# Very small values of k (like k=1) can lead to overfitting because the model closely follows the training data and may misclassify new points due to noise.
# Very large values of k may cause underfitting because the model becomes too simple, averaging over too many neighbors and ignoring local patterns.