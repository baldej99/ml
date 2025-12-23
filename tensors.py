import numpy as np

class Tensor(object): #создаем класс Tensor, наследуем от object
    _next_id = 0
    def __init__(self, data, creators = None, operation_on_creation = None, autograd=False, id=None):
        self.data = np.array(data)
        self.creators = creators
        self.autograd = autograd
        self.operation_on_creation = operation_on_creation
        self.grad = None  # предварительное объявление поля градиента
        # нам нужна информация о всех дочерних элементах тензора, т.е. элементах в создании которых участвовал данный тензор
        self.children = {}  # создаем словарь
        # если id пустой, то мы его генерируем
        if id is None:
            self.id = Tensor._next_id
            Tensor._next_id += 1
        else:
            self.id = id

        # проверяем есть ли для нашего тензора создатели и если есть, то сообщаем им о себе, добавляя в словарь дочерних элементах
        if creators is not None:
            for creator in creators:
                if self.id not in creator.children:
                    creator.children[self.id] = 1
                else:
                    creator.children[
                    self.id] += 1  # если мы уже сообщади о себе, то просто увеличиваем количество детей от этого id

    def __add__(self, other):

    # если у складываемых тензоров включен автоградиент, то мы изменяем результат возвращаемый функцией. если автоградиент не включен, то информацию о создателях передавать не нужно.
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, [self, other], "+", True)
        return Tensor(self.data + other.data)

    def __str__(self):# функция нужна для удобного вывода в консоль
        return str(self.data.__str__())

    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
        # если градиент пустой, то мы создаем тензор, состоящий из одних единиц с размерностью data
            if grad is None:
                grad = Tensor(np.ones_like(self.data))
            if grad_origin is not None:
                if (self.children[grad_origin.id]) > 0:
                    self.children[grad_origin.id] -= 1
            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad  # суммируем старое и новое значения градиентов
        # теперь мы должны выполнить проверки и произвести вычисления
            if self.creators is not None and (self.check_grads_from_children() or grad_origin is None):
                if self.operation_on_creation == "+":  #
                    self.creators[0].backward(grad, grad_origin=self)
                    self.creators[1].backward(grad, grad_origin=self)

    def check_grads_from_children(self):  # функция возвращает true когда все градиенты детей получены
        for id in self.children:
            if self.children[id] != 0:
                return False
        return True



a_1 = Tensor([1,2,3], autograd=True)
a_2 = Tensor([1,2,3], autograd=True)
a_3 = Tensor([1,2,3], autograd=True)

a_add_1 = a_1 + a_2
a_add_2 = a_2 + a_3
a_add_3 = a_add_1 + a_add_2
a_add_3.backward(Tensor([4,5,3]))
print(a_2.grad)