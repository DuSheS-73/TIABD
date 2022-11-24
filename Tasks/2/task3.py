import numpy as np

rowAmount = int(input('Введите кол-во строк матрицы: '))
columnAmount = int(input('Введите кол-во столбцов матрицы: '))
print(rowAmount, columnAmount)
myMatrix = np.random.rand(rowAmount, columnAmount)
myMatrix.flatten()
print('Полученный вектор:', *myMatrix)