import functools
import torch

class ConstraintManager():
    def __init__(self, constraints:list) -> None:
        self.constraints = constraints
        self.losses = []

        for i in range(len(self.constraints)):
            if self.constraints[i][1] == "<":
                self.losses.append(construct_not_larger_loss(torch.tensor(self.constraints[i][2])))
            if self.constraints[i][1] == ">":
                self.losses.append(construct_not_smaller_loss(torch.tensor(self.constraints[i][2])))
            if self.constraints[i][1] == "=":
                self.losses.append(construct_target_loss(torch.tensor(self.constraints[i][2]), torch.tensor(self.constraints[i][3])))

    def print_constrained_variable(self):
        for i in range(len(self.constraints)):
            print(self.constraints[i][0])


def construct_target_loss(tgt, half_width, exponent=8, wt=1):
    return functools.partial(target_loss, tgt = tgt, half_width = half_width, exponent = exponent, wt = wt)

def construct_not_smaller_loss(tgt, exponent=20):
    return functools.partial(smaller_get_loss, tgt = tgt, exponent = exponent)

def construct_not_larger_loss(tgt, exponent=20):
    return functools.partial(larger_get_loss, tgt = tgt, exponent = exponent)

def target_loss(value, tgt, half_width, exponent=8, wt=1):
    return wt*((value - tgt)/half_width)**(exponent/2)

def larger_get_loss(value, tgt, exponent=20):
    return (torch.nn.functional.relu(value - tgt)/torch.abs(tgt) + 1.0)**(exponent/2) - 1.0

def smaller_get_loss(value, tgt, exponent=20):
    return (torch.nn.functional.relu(tgt - value)/torch.abs(tgt) + 1.0)**(exponent/2) - 1.0


if __name__ == "__main__":
    a = construct_target_loss(1,8)
    print(a(2))
    b = construct_not_smaller_loss(torch.tensor(4.0))
    print(b(torch.tensor(5.0)),b(torch.tensor(3.0)))