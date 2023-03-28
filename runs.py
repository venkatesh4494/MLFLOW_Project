import numpy as np
import os

alpha=np.linspace(0.1,1.0,5)
l1_ratio=np.linspace(0.1,1.0,5)

for alpha_ in alpha:
    for l1_ in l1_ratio:
        os.system(f'python main.py -a {alpha_} -l1 {l1_}')