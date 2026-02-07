import torch
from bench.timer import Timer
from bench.logger import log_metric

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')


def is_contiguous_fast(x: torch.Tensor):
    stride_values = x.stride()
    shape_values = x.shape
    i = len(stride_values)-1
    current = 1
    while i>=0:
        if stride_values[i] != current:
            return False
        current *= shape_values[i]
        i-=1
    return True

def change_view(x: torch.Tensor):
    y = x.view(-1)

def change_cont_view(x: torch.Tensor):
    y = x.contiguous().view(-1)


def main():
    x = torch.randn((1024*10, 512, 256), device=device)
    y = x.transpose(-2, -1)
    print(f"x:- shape: {x.shape}, stride: {x.stride()}, data_ptr: {x.data_ptr()}, data_type: {x.dtype}, element_size: {x.element_size()}")
    assert int(x[0, 0, 1].data_ptr()) - int(x[0, 0, 0].data_ptr()) == x.element_size()
    print(f"y:- shape: {y.shape}, stride: {y.stride()}, data_ptr: {y.data_ptr()}, data_type: {y.dtype}, element_size: {y.element_size()}")
    
    print(f"x is contiguous in memory: {is_contiguous_fast(x)}, default lib function: {x.is_contiguous()}")
    print(f"y is contiguous in memory: {is_contiguous_fast(y)}, default lib function: {y.is_contiguous()}")
    
    timer_obj = Timer()
    print("change view (no mem copy):- ", timer_obj(change_view, x))
    print("change view (mem copy):- ", timer_obj(change_cont_view, y))
    


if __name__ == '__main__':
    main()
    
    
    
    