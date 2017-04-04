import IPython
import random

def func1():
    IPython.embed()
    print('Run func1')
    
def func2():
    IPython.embed()
    print('Run func2')
    func1()
    
def func3():
    IPython.embed()
    print('Run func3')
    func2()

rand1 = random.random()
IPython.embed(header='Embed 1')
func3()