from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

data = load_iris(as_frame = True) #Load dataset
predictors = data.data #Prediction parameters
target = data.target #Iris types
target_names = data.target_names #Iris names

x_train, x_test, y_train, y_test = train_test_split(predictors, target, train_size = .8, shuffle = True)

param_kernel = ('linear', 'rbf', 'poly', 'sigmoid') # для переборя ядер
parameters = { 'kernel': param_kernel }
model = SVC()

grid_search_svm = GridSearchCV(estimator=model, param_grid=parameters, cv=6) # сетка для перебора параметров
grid_search_svm.fit(x_train, y_train) # обучаем модель с разными параметрами

best_model = grid_search_svm.best_estimator_

print('Лучшее ядро: ', best_model.kernel)

svm_preds = best_model.predict(x_test)

print(classification_report(svm_preds, y_test))