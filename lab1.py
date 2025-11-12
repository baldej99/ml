def neuralNetwork(inp, weight):
     prediction = inp * weight
     return prediction

out_1 = neuralNetwork(120, 0.4)
out_2 = neuralNetwork(110, 0.5)

print('1)Измените входные данные и вес нейросети в коде. Запустите программу с новыми значениями и опишите, как это повлияло на выходные данные. Объясните, почему это произошло с точки зрения работы нейронной сети')
print(out_1)
print(out_2)
#2
print('2)Создайте список входных данных (например, inputs = [150, 160, 170, 180, 190]) и используйте цикл for для вычисления выходных данных нейросети для каждого значения в списке. Распечатайте выходные данные для каждого входного значения.')
weight = 0.8
inputs = [125, 135, 140, 150, 160]
for input in inputs:
    print(neuralNetwork(input, weight))
#3
print('3)Модифицируйте функцию neural_network так, чтобы она принимала два входных параметра: inp и bias. Результат будет задан как inp * weight + bias. Запустите функцию с новыми значениями inp, weight и bias. Как изменится выходная переменная? Почему?')
def neuralNetwork_mod(inp, weight, bias):
    prediction = inp*weight + bias
    return prediction
print(neuralNetwork_mod(54, 0.7, -5))
#4
print('4)Измените функцию neural_network так, чтобы она возвращала не только предсказание (prediction), но и весь список промежуточных значений (произведение каждого элемента входных данных на соответствующий вес). Выведите обе переменные (prediction и список промежуточных значений) на экран.')
def neuralNetwork2(inps, weights):

    prediction = 0
    ii = []

    for i in range(len(weights)):

        prediction += inps[i]*weights[i]
        ii.append(inps[i]*weights[i])

    return prediction, ii

#5
print('5)Измените веса нейросети в коде и определите, при каких значениях весов выходные данные для каждого элемента становятся больше 0.5. Решите это методом проб и ошибок, меняя веса с небольшими шагами.')
out_1 = neuralNetwork2([150, 40], [0.3, 0.4])
print(out_1[1], out_1[0])
out_1 = neuralNetwork2([150, 40], [0.300666666666666666, 0.41])

out_2 = neuralNetwork2([80, 60], [0.2, 0.4])

print(out_1[1], out_1[0])

print(out_2[1], out_2[0])
print('6) Напишите код с циклом, где значение веса будет увеличиваться до тех пор, пока выходное значение меньше 0.5. Как только один выход стал больше 0.5, то изменение его веса останавливается. Как только второй выход стал больше 0.5, то изменение его веса также останавливается, а цикл завершается. Выведите получившиеся веса.')
def neuralNetwork3(inp, weights):
    prediction = [0, 0]
    for i in range(len(weights)):
        prediction[i] = inp * weights[i]
    return prediction


inp = 4
weights = [0.2, 0.5]

while True:
    prediction = neuralNetwork3(inp, weights)
    print(f"Выходы: {prediction}, веса: {weights}")

    if prediction[0] >= 0.5 and prediction[1] >= 0.5:
        break
    for i in range(len(weights)):
        if prediction[i] < 0.5:
            weights[i] += 0.01

print("\nИтоговые веса:", weights)

def neuralNetwork4(inp, weights):

    prediction = [0] * len(weights)

    for i in range(len(weights)):
        ws=0
        for j in range(len(inp)):
            ws += inp[j] * weights[i][j]
        prediction[i] = ws
    return prediction

#7,8
print('7)Добавьте еще один набор весов (weights_4 = [0.4, 0.2, 0.1]) и добавьте его в список weights. Запустите функцию с этим новым набором весов. Как это повлияло на предсказанные значения? Объясните, почему.')
inp = [50, 165, 45]



weights_1 = [0.19, 0.1, 0.6]

weights_2 = [0.1, 0.1, 0.7]

weights_3 = [0.5, 0.4, 0.34]

#weights_4 = [0.1, 0.7, 0.9]

print('т. к. 4 элемента в списке inp нет, weights 4 умножается на 0 и значение не меняется')
print('8)Измените веса нейросети таким образом, чтобы выходные данные для первого и второго нейрона стали равными. Используйте метод проб и ошибок. Входные значения менять нельзя.')
print('9)Выполните предыдущее задание, но с помощью цикла. После цикла выведите получившиеся веса. (приравняем оставшийся к остальным)')
weights = [weights_1, weights_2, weights_3]

prediction123 = (neuralNetwork4(inp, weights))
print(prediction123)
weights_3[0] -= 0.006
while not(neuralNetwork4(inp, weights)[0] >= neuralNetwork4(inp, weights)[2]):
        weights_3[0] -= 0.01
print(neuralNetwork4(inp,weights))
print('10) Измените веса нейросети так, чтобы предсказанные значения для второго слоя (prediction_h) стали больше 5. Напишите код, который это сделает. Выведите получившиеся веса. Само собой, входные данные менять нельзя.')
def neuralNetwork_h(inps, weights):
    prediction_h = [0] * len(weights[0])
    for i in range(len(weights)):
        ws = 0
        for j in range(len(inps)):
            while ws + inps[j] * weights[0][i][j] >=5:
                weights[0][i][j] -= 0.1
                print(weights[0][i][j])
            ws += inps[j] * weights[0][i][j]
        prediction_h[i] = ws
        print("\n pred[",i,"]", prediction_h[i], '\n т.к значения на скрытом слое изначально были больше 5 сделаем их меньше 5')
    prediction_out = [0] * len(weights[1])
    for i in range(len(weights)):
        ws = 0
        for j in range(len(prediction_h)):
            ws += prediction_h[j] * weights[1][i][j]
        prediction_out[i] = ws
    return prediction_out


inp = [23, 45]
weight_h_1 = [0.4, 0.1]
weight_h_2 = [0.3, 0.2]

weight_out_1 = [0.4, 0.1]
weight_out_2 = [0.3, 0.1]

weight_h = [weight_h_1, weight_h_2]
weight_out = [weight_out_1, weight_out_2]

weights = [weight_h, weight_out]

out_1 = neuralNetwork_h(inp, weights)

print(out_1)
