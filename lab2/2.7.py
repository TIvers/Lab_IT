from datetime import datetime
import time

start_time = datetime.now()

# Стоимость яблок
apple = 150

# Стоимость малины
m = 600

# Стоимость винограда
v = 220

# Стоимость хлеба
bread = 45

# Стоимость сыра
cheese = 700

name = "Расторгуев В.А."

print("Чек о продаже товаров в магазине \n")
print("Дата и время продажи ", start_time)
print("Яблоки (3 кг.) - ", apple)
print("Малина (1 кг.) - ", m)
print("Виноград (4 кг.) - ", v)
print("Хлеб (2 кг.) - ", bread)
print("Сыр (1 кг.) - ", cheese)

summ = apple + m + v + bread + cheese
print("\nИтого: ", summ)

discount = (summ / 100 * 13)
print("\nВам предоставлен купон на скидку")
print("В размере 13% от стоимости вашей покупки", discount)
print("Спасибо за покупку!")
print("\nПродавец ", name)