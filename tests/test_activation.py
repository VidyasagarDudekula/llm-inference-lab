import torch
from model.activations import SiLuCustome

class TestActivation:
    def __init__(self):
        pass
    
    def check(self, result):
        if result:
            return "Gradcheck passed!"
        else:
            return "Gradcheck Failed :("
    
    def test_grad(self, custome_class, *inputs):
        result = torch.autograd.gradcheck(custome_class.apply, *inputs)
        return self.check(result)
    

if __name__ == '__main__':
    inputs = torch.randn((2, 10), dtype=torch.float64, requires_grad=True)
    m = TestActivation()
    print(m.test_grad(SiLuCustome, inputs))