a = float(input("Введите первое число: "))
b = float(input("Введите второе число: "))
sum = a + b
sumsquared = a ** 2 + b ** 2
while sum != 0:
    temp = float(input("Введите очередное число: "))
    sum += temp
    sumsquared += temp ** 2

print ("Результат выполнения программы равен %d" %(sumsquared))