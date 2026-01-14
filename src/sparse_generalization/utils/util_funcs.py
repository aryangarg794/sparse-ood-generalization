import numpy as np

def noise_scheduler(start_eta: float, step: int, gamma: float = 0.55):
    return start_eta / (1+step)**gamma