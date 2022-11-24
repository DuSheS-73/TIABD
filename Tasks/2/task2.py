n = int(input("Введите число N: "))
myList = []
for i in range(n):
    count = 0
    while count < i + 1:
        myList.append(i + 1)
        count += 1
        if n == len(myList):
            print("Результат работы программы:", *myList)
            break