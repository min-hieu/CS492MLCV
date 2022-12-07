import os
from multiprocessing import Pool

gpus    = [1,2,3,4,5,6,7,8]
use_all = True
gan     = 'DCGAN' # ['DCGAN', 'CGAN'] 
synth   = ['0.1','0.2','0.5','1.0']

def run(i):
    if use_all:
        all_gpu = ",".join(list(map(str, gpus)))
        gpu     = f"CUDA_VISIBLE_DEVICES={all_gpu}"
        ngpu    = len(gpus)
    else: 
        gpu     = f"CUDA_VISIBLE_DEVICES={gpus[i]}"
        ngpu    = 1

    if gan == 'DCGAN':
        vbn     = '_vbn' if i % 2 == 0 else ''
        cfg     = f"--config config/DCGAN{vbn}.yml --gan DCGAN"
    elif gan == 'CGAN':
        cfg     = f"--config config/CGAN_{synth[i]}.yml --gan CGAN"
    else:
        raise Exception

    quiet   = "> /dev/null 2> /dev/null"
    cmd     = f"{gpu} python train.py {cfg} --ngpu {ngpu}"
    print(cmd)
    os.system(cmd)

for i in range(4):
    run(i)
'''
with Pool(2) as p:
    p.map(run, [0])
'''
