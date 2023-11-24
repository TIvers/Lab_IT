range = float(input("Введите дальность полета "))

try:
    res = 1 / range
except:
    print("Дальность не может быть 0")
    exit(0)
    
age = int(input("Введите возраст "))
try:
    res = 1 / age
except:
    print("Возраст не может быть 0")
    exit(0)   
    
class_ = int(input("Введите класс (1 - Эконом; 2 - Бизнес) "))

if (class_ == 1):
    coast = 5000*(range/100)*(age/2)
elif (class_ == 2):
    coast = 10000*(range/100)*(age/2)
else:
    print("Нет такого класса")
    exit(0)
print("Стоимость билета равна ")
print(coast)
