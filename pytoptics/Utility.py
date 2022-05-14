import sys
original_stdout = sys.stdout
import torchviz
import torch
import os

def getBack(var_grad_fn, f, counter=0):
    sys.stdout = f
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print('stack '+str(counter))
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0], f, counter+1)
    


def debugTracker(tensor, name="log"):

    if not os.path.exists('log'):
        os.makedirs('log')    
    try:
        torchviz.make_dot(tensor).render('log/'+name, format="jpg")
        loss = torch.std(tensor, unbiased=False)
        loss.backward(retain_graph=True)
        f = open('log/'+name+'.txt', 'w')
        getBack(loss.grad_fn, f)
        sys.stdout = original_stdout 
    except Exception as e: 
        sys.stdout = original_stdout 
        print(e)

#from .Utility import debugTracker; debugTracker(,"")