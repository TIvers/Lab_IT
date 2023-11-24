s = int(input("Введите стаж работы в годах: \n"))
o = int(input("Введите оклад: \n"))
if (s < 1 or o < 20000):
    o+= o*0.05
    print(o)
elif ( (s >= 1 and s < 3) or (o >= 20000 and o < 30000)):
    o= o * 0.07
    print("Премия: ", o)
elif ( (s >= 3 and s < 5) or (o >= 30000 and o < 50000)):
    o= o*0.12
    print("Премия: ", o)
elif (s >= 5 or o >= 50000):
    o= o*0.15
    print("Премия: ", o)
