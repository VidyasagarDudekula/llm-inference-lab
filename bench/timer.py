import time
import torch
from bench.logger import log_metric


class Timer:
    def __init__(self, warmup_steps=10, iter_steps=100):
        self.start_time = torch.cuda.Event(enable_timing=True)
        self.end_time = torch.cuda.Event(enable_timing=True)
        self.warmup_steps = max(warmup_steps, 0)
        self.iter_steps = max(iter_steps, 1)
    
    def __call__(self, *args):
        func = args[0]
        
        
        # warup steps
        for _ in range(self.warmup_steps):
            func(*args[1:])
        
        torch.cuda.synchronize()
        self.start_time.record()
        for _ in range(self.iter_steps):
            func(*args[1:])
        self.end_time.record()
        torch.cuda.synchronize()
        self.end_time.synchronize()
        total_time = self.start_time.elapsed_time(self.end_time)/self.iter_steps
        return total_time
    
    def start(self):
        torch.cuda.synchronize()
        self.start_time.record()
    
    def end(self):
        self.end_time.record()
        self.end_time.synchronize()
        torch.cuda.synchronize()
    
    def elapsed_timer(self):
        total_time = self.start_time.elapsed_time(self.end_time)
        return total_time
    

if __name__ == '__main__':
    log_metric(
        metric_name="matmul_f32",
        value=0.009,
        unit="ms",
        config={"batch": 2, "k": 100},
    )