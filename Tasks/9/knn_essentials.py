import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

data = load_iris(as_frame = True) #Load dataset
predictors = data.data #Prediction parameters
target = data.target #Iris types
target_names = data.target_names #Iris names

x_train, x_test, y_train, y_test = train_test_split(predictors, target, train_size = .8, shuffle = True)

number_of_neighbors = np.arange(3, 10, 25) # кол-во соседей для перебора
model = KNeighborsClassifier()
params = { 'n_neighbors': number_of_neighbors }

grid_search = GridSearchCV(estimator=model, param_grid=params, cv=6)
grid_search.fit(x_train, y_train)

print("Macro-average best: ", grid_search.best_score_)

best_model = grid_search.best_estimator_
print(best_model)

knn_preds = best_model.predict(x_test)

print(classification_report(knn_preds, y_test))
