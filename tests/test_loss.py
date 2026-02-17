import torch
from model.loss import CrossEntropyLossCustome


class TestCrossEntropy:
    def __init__(self):
        pass
    
    def check(self, result):
        if result:
            return "Gradcheck passed!"
        else:
            return "Gradcheck Failed :("
    
    def compare_default(self, custome_class, default_class, logits: torch.Tensor, target_ids: torch.Tensor):
        B, T, C = logits.shape
        result_custome = custome_class(*inputs)
        result_default = default_class(*inputs)
        print(result_custome, result_default)
        return result_default == result_custome
        
        

if __name__ == '__main__':
    gen = torch.manual_seed(42)
    inputs = torch.randn((2, 5, 50), dtype=torch.float64, requires_grad=True, generator=gen)
    target_ids = torch.randint(0, 50, (2, 5), generator=gen)
    m = TestCrossEntropy()
    # print(m.compare_default(CrossEntropyLossCustome, torch.nn.CrossEntropyLoss, inputs))