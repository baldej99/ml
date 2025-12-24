import numpy as np

class Tensor(object): 
    _next_id = 0
    def __init__(self, data, creators = None, operation_on_creation = None, autograd=False, id=None):
        self.data = np.array(data)
        self.creators = creators
        self.autograd = autograd
        self.operation_on_creation = operation_on_creation
        self.grad = None  
        self.children = {} 
        if id is None:
            self.id = Tensor._next_id
            Tensor._next_id += 1
        else:
            self.id = id
        if creators is not None:
            for creator in creators:
                if self.id not in creator.children:
                    creator.children[self.id] = 1
                else:
                    creator.children[
                    self.id] += 1 

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, [self, other], "+", True)
        return Tensor(self.data + other.data)

    def __str__(self):
        return str(self.data.__str__())

    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            if grad is None:
                grad = Tensor(np.ones_like(self.data))
            if grad_origin is not None:
                if (self.children[grad_origin.id]) > 0:
                    self.children[grad_origin.id] -= 1
            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad  
            if self.creators is not None and (self.check_grads_from_children() or grad_origin is None):
                if self.operation_on_creation == "+":  #
                    self.creators[0].backward(grad, grad_origin=self)
                    self.creators[1].backward(grad, grad_origin=self)
                elif self.operation_on_creation == "-1":  
                    self.creators[0].backward(self.grad.__neg__(), self)  
                elif self.operation_on_creation == "-":  
                    self.creators[0].backward(self.grad, self)  
                    self.creators[1].backward(self.grad.__neg__(), self)  
                elif self.operation_on_creation == "*":  
                    self.creators[0].backward(self.grad * self.creators[1], self)
                    self.creators[1].backward(self.grad * self.creators[0], self) 
                elif self.operation_on_creation.startswith("sum_"):
                    axis = int(self.operation_on_creation.split("_")[1])  
                    self.creators[0].backward(self.grad.expand(axis, self.creators[0].data.shape[axis]), self)
                elif "expand" in self.operation_on_creation:
                    axis = int(self.operation_on_creation.split("_")[1]) 
                    self.creators[0].backward(self.grad.sum(axis), self)

    def check_grads_from_children(self):  
        for id in self.children:
            if self.children[id] != 0:
                return False
        return True

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1, [self], "-1", True)
        return Tensor(self.data * -1)

    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data, [self, other], "-", True)
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data, [self, other], "*", True)
        return Tensor(self.data * other.data)

    def sum(self, axis):  
        if self.autograd:
            return Tensor(self.data.sum(axis), [self], "sum_"+str(axis),True) 
        return Tensor(self.data.sum(axis))

    def expand(self, axis, count_copies): 
        transpose = list(range(0, len(self.data.shape))) 

        transpose.insert(axis, len(self.data.shape))
        expand_shape = list(self.data.shape) + [count_copies]  
        expand_data = (self.data.repeat(count_copies).reshape(expand_shape)) 
        expand_data = expand_data.transpose(transpose) 
        if self.autograd:
            return Tensor(expand_data, [self], "expand_" + str(axis), True)  
        return Tensor(expand_data)
        return Tensor(self.data.sum(axis), [self], "sum_" + str(axis), True)


a_1 = Tensor([1,2,3], autograd=True)
a_2 = Tensor([4,5,6], autograd=True)
a_3 = Tensor([9,8,7], autograd=True)
a_add_1 = a_1 + (-a_2)
a_add_2 = a_2 + a_3
a_add_3 = a_add_1 + a_add_2
a_add_3.backward(Tensor([1,2,3]))
print('neg')
print(a_3.grad)

a_1 = Tensor([1,2,3], autograd=True)
a_2 = Tensor([4,5,6], autograd=True)
a_3 = Tensor([9,8,7], autograd=True)
a_sub_1 = a_1 -a_2
a_sub_2 = a_2 - a_3
a_add_3 = a_sub_1 + a_sub_2
a_add_3.backward(Tensor([4,5,3]))
print('\n' + 'sub')
print(a_1.grad)
print(a_2.grad)
print(a_3.grad)

a_1 = Tensor([1,2,3], autograd=True)
a_2 = Tensor([4,5,6], autograd=True)
a_mult = a_1 * a_2
a_mult.backward(Tensor([4,5,3]))
print('\n' + 'mul')
print(a_mult.grad)
print(a_1.grad)
print(a_2.grad)

a_1 = Tensor([[1,2,3], [4,5,6]], autograd=True)
a_2 = a_1.sum(0)
a_3 = a_1.sum(1)
print('\n' + 'expand')
print(a_2)
print(a_2.expand(0,2))

a_2.backward(Tensor([5,10,20]))
print(a_1.grad)

a = Tensor([[1, 2, 3],
            [4, 5, 6]], autograd=True)
print('\n' + 'sum')
s = a.sum(0)
s.backward(Tensor([5, 10, 20]))
print(a.grad)

print(s.data, a.grad)
