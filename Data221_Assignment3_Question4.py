import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from Data221_Assignment3.Data221_Assignment3_Question3 import feature_matrix_x

kidney_data = pd.read_csv("kidney_disease.csv")
feature_matrix_x = kidney_data.loc[:, kidney_data.columns != "classification"]
label_y = kidney_data.loc[:, "classification"]
feature_x_train, feature_x_test, label_y_train, label_y_test = train_test_split(feature_matrix_x, label_y, test_size=0.3)

# ------------------ Train KNN ------------------
knn_model = KNeighborsClassifier(n_neighbors=5)  # k = 5
knn_model.fit(feature_x_train, label_y_train)
label_y_predict = knn_model.predict(feature_x_test)

# ------------------ Compute Metrics ------------------

accuracy_list = []
cm = confusion_matrix(label_y_test, label_y_predict)
accuracy = accuracy_score(label_y_test, label_y_predict)
accuracy_list.append(accuracy)
precision = precision_score(label_y, label_y_predict, pos_label="CKD")  # assuming "CKD" = positive class
recall = recall_score(label_y_test, label_y_predict, pos_label="CKD")
f1 = f1_score(label_y_test, label_y_predict, pos_label="CKD")

print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)