# import numpy as np
# import pandas as pd
#import sklearn
#import plotly.express as px

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data = load_iris(as_frame = True) #Load dataset
predictors = data.data #Prediction parameters
target = data.target #Iris types
target_names = data.target_names #Iris names

# print(predictors.head(5))
# print('Target variable:\n', target.head(5))
# print('Names:\n', target_names)

x_train, x_test, y_train, y_test = train_test_split(predictors, target, train_size = .8, shuffle = True)

# print('Размер для признаков обучающей выборки', x_train.shape, '\n',
#       'Размер для признаков тестовой выборки', x_test.shape, '\n',
#       'Размер для целевого показателя обучающей выборки', y_train.shape, '\n',
#       'Размер для показателя тестовой выборки', y_test.shape)

model = LogisticRegression(random_state = 271)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

# print ('Исходные значения:\n', np.array(y_test), '\n',
#        'Предсказанные значения:\n', y_predict)

# fig = px.imshow(confusion_matrix(y_test, y_predict), text_auto = True)
# fig.update_layout(xaxis_title = 'Target', yaxis_title = 'Prediction')
# fig.show()


# Support: количество наблюдений для каждого класса
# Macro avg: среднее арифметическое показателя между классами
# weighted avg: средневзвешенное значение рассчитывается путем 
# произведения оценки показателя каждого класса на его количество наблюдений, 
# последующее суммирование результата и деление результата на сумму наблюдений.
# Пример с recall:
# Macro avg = (1.00 + 0.94 + 0.90) / 3 = 2.84 / 3 = 0.947
# Weighted avg = (1.00 * 12 + 0.94 * 18 + 0.90 * 20) / (12+18+20) = = (12 + 16,92 + 18) / 50 = 0,94
print(classification_report(y_test, y_predict))

