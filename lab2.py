import numpy as np
arr1 = np.array([1,2,3,4,5])
arr1*=2
print('1)',arr1)
arr2 = np.random.rand(3,3)
arr2_2 = np.random.rand(3,3)
arr2_pr = arr2 * arr2_2
print(arr2_pr)
arr5 = np.random.randint(1, 10, 10)
print(arr5)
print(np.mean(arr5), np.std(arr5).round(1), np.max(arr5), np.min(arr5))

import numpy as np

def neuralNetwork(inp, weights):
    # Первый скрытый слой
    prediction_h1 = inp.dot(weights[0])
    # Второй скрытый слой
    prediction_h2 = prediction_h1.dot(weights[1])
    # Выходной слой
    prediction_out = prediction_h2.dot(weights[2])
    return prediction_out


# Входные данные
inp = np.array([23, 45])

# Веса для первого скрытого слоя (2 входа → 3 нейрона)
weight_h_1 = [0.4, 0.1]
weight_h_2 = [0.3, 0.2]
weight_h_3 = [0.6, 0.2]

# Веса для второго скрытого слоя (3 входа → 3 нейрона)
weight_h2_1 = [0.4, 0.1, 0.2]
weight_h2_2 = [0.3, 0.1, 0.3]
weight_h2_3 = [0.7, 0.4, 0.5]

# Веса для выходного слоя (3 входа → 2 выхода)
weight_out_1 = [0.4, 0.1, 0.2]
weight_out_2 = [0.3, 0.1, 0.4]

# Преобразуем в матрицы и транспонируем
weights_h1 = np.array([weight_h_1, weight_h_2, weight_h_3]).T
weights_h2 = np.array([weight_h2_1, weight_h2_2, weight_h2_3]).T
weights_out = np.array([weight_out_1, weight_out_2]).T

# Собираем все веса в список
weights = [weights_h1, weights_h2, weights_out]

# Проверим результат
print(neuralNetwork(inp, weights))

inp = np.array([23, 45])

# Генерируем случайные веса
# 1. Первый скрытый слой: 2 входа → 3 нейрона
weights_h1 = np.random.rand(2, 3)

# 2. Второй скрытый слой: 3 входа → 3 нейрона
weights_h2 = np.random.rand(3, 3)

# 3. Выходной слой: 3 входа → 2 выхода
weights_out = np.random.rand(3, 2)

# Объединяем всё в список весов
weights = [weights_h1, weights_h2, weights_out]

print("\nВыход нейросети:", neuralNetwork(inp, weights))
