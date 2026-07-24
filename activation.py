# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 08:19:54 2022

@author: ar2806
"""
import tensorflow as tf 


BINARY_EXPRESSIONS = {
    0: lambda a, b: f"({a} + {b})",
    1: lambda a, b: f"({a} * {b})",
    2: lambda a, b: f"max(max({a}, 0), {b})",
    3: lambda a, b: f"({a} * erf({b}))",
    4: lambda a, b: f"max({a}, {b})",
    5: lambda a, b: f"({a} * exp({b}))",
    6: lambda a, b: f"(sigmoid({a}) * {b})",
}


UNARY_EXPRESSIONS = {
    0: lambda x: x,
    1: lambda x: f"exp({x})",
    2: lambda x: f"abs({x})",
    3: lambda x: f"-({x})",
    4: lambda x: f"exp(-({x}))",
    5: lambda x: f"min({x}, 0)",
    6: lambda x: f"erf({x})",
    7: lambda x: f"sigmoid({x})",
    8: lambda x: f"({x} * erf({x}))",
    9: lambda x: f"max({x}, 0)",
    10: lambda x: f"sin({x})",
    11: lambda x: f"atan({x})",
    12: lambda x: f"tanh({x})",
    13: lambda x: f"log(1 + exp({x}))",
    14: lambda x: f"sinh({x})",
    15: lambda x: f"log(1 + exp({x}))",
    16: lambda x: f"atanh({x})",
    17: lambda x: f"asinh({x})",
}


class Activations(object):
    def b0(self,a,b):
        return a+b
    def b1(self,a,b):
        return a*b
    def b2(self,a,b):
        return tf.math.maximum(tf.math.maximum(a,0),b)
    def b3(self,a,b):
        return a*tf.math.erf(b)
    def b4(self,a,b):
        return tf.math.maximum(a,b)
    def b5(self,a,b):
        return a*tf.math.exp(b)
    def b6(self,a,b):
        return tf.math.sigmoid(a)*b
    
    def f0(self,x):
        return x
    def f1(self,x):
        return tf.math.exp(x)
    def f2(self,x):
        return tf.math.abs(x)
    def f3(self,x):
        return -x
    def f4(self,x):
        return tf.math.exp(-x)
    def f5(self,x):
        return tf.math.minimum(x,0)
    def f6(self,x):
        return tf.math.erf(x)
    def f7(self,x):
        return tf.math.sigmoid(x)
    def f8(self,x):
        return x*tf.math.erf(x)
    def f9(self,x):
        return tf.math.maximum(x,0)
    def f10(self,x):
        return tf.math.sin(x)
    def f11(self,x):
        return tf.math.atan(x)
    def f12(self,x):
        return tf.math.tanh(x)
    def f13(self,x):
        return tf.math.log(1+tf.math.exp(x))
    def f14(self,x):
        return tf.math.sinh(x)
    def f15(self,x):
        return tf.math.log(1+tf.math.exp(x))
    def f16(self,x):
        return tf.math.atanh(x)
    def f17(self,x):
        return tf.math.asinh(x)
    
def get_activation(pop,value):
    pop_len =len(pop)
    functions=Activations()
    if pop_len==3:
        x_1=getattr(functions, 'f'+str(pop[1]))(value)
        x_2=getattr(functions, 'f'+str(pop[2]))(value)
        y=getattr(functions, 'b'+str(pop[0]))(x_1,x_2)
    elif pop_len==5:
        x_1=getattr(functions, 'f'+str(pop[2]))(value)
        x_2=getattr(functions, 'f'+str(pop[3]))(value)
        x_3=getattr(functions, 'f'+str(pop[4]))(value)
        y_1=getattr(functions, 'b'+str(pop[1]))(x_1,x_2)
        y=getattr(functions, 'b'+str(pop[0]))(y_1,x_3)
    else:
        x_1=getattr(functions, 'f'+str(pop[3]))(value)
        x_2=getattr(functions, 'f'+str(pop[4]))(value)
        x_3=getattr(functions, 'f'+str(pop[5]))(value)
        x_4=getattr(functions, 'f'+str(pop[6]))(value)
        y_1=getattr(functions, 'b'+str(pop[1]))(x_1,x_2)
        y_2=getattr(functions, 'b'+str(pop[2]))(x_3,x_4)
        y=getattr(functions, 'b'+str(pop[0]))(y_1,y_2)
    return y


def chromosome_to_expression(pop, variable="x"):
    pop_len = len(pop)
    if pop_len == 3:
        x_1 = UNARY_EXPRESSIONS[pop[1]](variable)
        x_2 = UNARY_EXPRESSIONS[pop[2]](variable)
        return BINARY_EXPRESSIONS[pop[0]](x_1, x_2)
    elif pop_len == 5:
        x_1 = UNARY_EXPRESSIONS[pop[2]](variable)
        x_2 = UNARY_EXPRESSIONS[pop[3]](variable)
        x_3 = UNARY_EXPRESSIONS[pop[4]](variable)
        y_1 = BINARY_EXPRESSIONS[pop[1]](x_1, x_2)
        return BINARY_EXPRESSIONS[pop[0]](y_1, x_3)
    elif pop_len == 7:
        x_1 = UNARY_EXPRESSIONS[pop[3]](variable)
        x_2 = UNARY_EXPRESSIONS[pop[4]](variable)
        x_3 = UNARY_EXPRESSIONS[pop[5]](variable)
        x_4 = UNARY_EXPRESSIONS[pop[6]](variable)
        y_1 = BINARY_EXPRESSIONS[pop[1]](x_1, x_2)
        y_2 = BINARY_EXPRESSIONS[pop[2]](x_3, x_4)
        return BINARY_EXPRESSIONS[pop[0]](y_1, y_2)
    raise ValueError(f"Unsupported chromosome length: {pop_len}")
