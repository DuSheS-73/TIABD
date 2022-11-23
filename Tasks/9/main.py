import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# # f = open('train_data.txt')
# #f.readline() # Пропустить заголовок
# data = np.genfromtxt('./assets/poker-hand-training.csv', skip_header=1, delimiter=',')
# # data = np.genfromtxt('./train_data.csv', dtype=None, usecols=range(4))

# print('TRAIN DATA:\t\t', data[0])

# x_train = data[:, :-1] # С первого до предпоследного столбца включительно
# y_train = data[:, -1] # Последний столбец
# print('X TRAIN:\t\t', x_train[0])
# print('Y TRAIN:\t\t', y_train[0])

# # scaler = StandardScaler().fit(x_train)
# # x_train_scaled = scaler.transform(x_train)
# # y_train_scaled = scaler.transform(y_train)

# # print('X TRAIN SCALED:\t', x_train_scaled[0])
# # print('Y TRAIN SCALED:\t', y_train_scaled[0])

# pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=271))
# pipe.fit(x_train, y_train)
# # pipe.score()

# # from sklearn.datasets import load_iris

# # data = load_iris(as_frame = True) #Load dataset
# # predictors = data.data #Prediction parameters
# # target = data.target #Iris types
# # target_names = data.target_names #Iris names

# # fig = px.histogram(predictors, target)
# # fig.update_layout(bargap=0.2)
# # fig.show()
# #print(data)
# # print(predictors.head(5))
# # print('Target variable:\n', target.head(5))
# #print('Names:\n', target_names)



# # import numpy as np
# # import pandas as pd

# # input_file = "mydata.csv"


# # # comma delimited is the default
# # df = pd.read_csv(input_file, header = 0)

# # # for space delimited use:
# # # df = pd.read_csv(input_file, header = 0, delimiter = " ")

# # # for tab delimited use:
# # # df = pd.read_csv(input_file, header = 0, delimiter = "\t")

# # # put the original column names in a python list
# # original_headers = list(df.columns.values)

# # # remove the non-numeric columns
# # df = df._get_numeric_data()

# # # put the numeric column names in a python list
# # numeric_headers = list(df.columns.values)

# # # create a numpy array with the numeric values for input into scikit-learn
# # numpy_array = df.as_matrix()

# # # reverse the order of the columns
# # numeric_headers.reverse()
# # reverse_df = df[numeric_headers]

# # # write the reverse_df to an excel spreadsheet
# # reverse_df.to_excel('path_to_file.xls')





data = pd.read_csv('./assets/poker-hand-training.csv', sep=',')

# Task 1
def task1():
    print('Задание 1.')
    print('Процент нечисловых значений в датасете:')
    for column in data.columns:
        missing = np.mean(data[column].isna() * 100)
        print(column, ": ", round(missing, 1), "%", sep='')

    print('\nПервые 5 строк датасета:\n', data.head())

# Task 2
def task2():
    data.hist()
    plt.tight_layout()
    plt.show()
    print('\nЗадание 2.')
    print('Кол-во комбинаций каждого ранга:\n', data["Poker Hand"].value_counts(), sep='')

# Разбиение выборки на тренировочную и тестовую для 3-5 заданий
x = data.drop(["Poker Hand"], axis=1)
y = data["Poker Hand"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=88, shuffle=True)

# Task 3
def task3():
    print('\nЗадание 3.')
    print('Размерность набора данных x_train: ', x_train.shape,
          '\nРазмерность набора данных x_test: ', x_test.shape,
          '\nРазмерность набора данных y_train: ', y_train.shape,
          '\nРазмерность набора данных y_test: ', y_test.shape)

# Вспомогательная функция построения графической матрицы ошибок
def plot_confusion_matrix(data_confusion, title='Матрица ошибок', cmap=plt.cm.bone):
    plt.matshow(data_confusion, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(data_confusion.columns))
    plt.xticks(tick_marks, data_confusion.columns)
    plt.yticks(tick_marks, data_confusion.index)
    plt.xlabel(data_confusion.columns.name)
    plt.ylabel(data_confusion.index.name)
    plt.show()


def task4and5():
    print('\nЗадания 4 и 5.')
    print('Применение метода логистической регрессии:\n')
    start = time.perf_counter()

    # Логистической регрессии
    pipe = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=1000, random_state=271))
    # logisticReg = LogisticRegression(solver='lbfgs', max_iter=10000)
    # logisticReg.fit(x_train, y_train.ravel())
    pipe.fit(x_train, y_train.ravel())
    prediction = pipe.predict(x_test)
    print(classification_report(y_test, prediction))

    elapsed = time.perf_counter() - start

    print('Время выполнения:', round(elapsed, 3),
          'сек\nКоэффициент точности (accuracy score):',
          round(accuracy_score(y_test, prediction) * 100),
          '%\nЧисловое представление матрицы ошибок:\n',
          confusion_matrix(y_test, prediction))

    data_confusion = pd.crosstab(y_test, prediction)
    plot_confusion_matrix(data_confusion)

    print('\nПрименение метода SVM (метод опорных векторов):\n')
    start = time.perf_counter()

    # SVM
    param_kernel = ('linear', 'rbf', 'poly', 'sigmoid')
    parameters = { 'kernel': param_kernel }
    model = SVC()

    grid_search_svm = GridSearchCV(estimator=model, param_grid=parameters, cv=6) # сетка для перебора параметров
    grid_search_svm.fit(x_train, y_train) # обучаем модель с разными параметрами

    best_model = grid_search_svm.best_estimator_

    print('Лучшее ядро: ', best_model.kernel)

    svm_preds = best_model.predict(x_test)
    print(classification_report(y_test, svm_preds))

    elapsed = time.perf_counter() - start

    print('Время выполнения:', round(elapsed, 3),
          'сек\nКоэффициент точности (accuracy score):',
          round(accuracy_score(y_test, svm_preds) * 100),
          '%\nЧисловое представление матрицы ошибок:\n',
          confusion_matrix(y_test, svm_preds))

    data_confusion = pd.crosstab(y_test, svm_preds)
    plot_confusion_matrix(data_confusion)

    # KNN
    print('\nПрименение метода KNN (кол-во ближайших соседей):\n')
    start = time.perf_counter()

    knn_model = KNeighborsClassifier(n_neighbors=25)
    knn_model.fit(x_train, y_train)
    knn_predict = knn_model.predict(x_test)
    print(classification_report(y_test, knn_predict))

    elapsed = time.perf_counter() - start

    print('Время выполнения:', round(elapsed, 3),
          'сек\nКоэффициент точности (accuracy score):',
          round(accuracy_score(y_test, knn_predict) * 100),
          '%\nЧисловое представление матрицы ошибок:\n',
          confusion_matrix(y_test, knn_predict))

    data_confusion = pd.crosstab(y_test, knn_predict)
    plot_confusion_matrix(data_confusion)

task1()
task2()
task3()
task4and5()