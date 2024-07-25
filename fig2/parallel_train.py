import os
os.system('export MKL_NUM_THREADS=1')
from os.path import exists
from time import sleep, time
import numpy as np
import pickle
import subprocess
import sys
#sys.path.append('/storage/home/hcoda1/8/zmobille3/scratch/reviseCellTypeCircuit/CellTypeCircuit/newCode/code')
# from help_funcs import *

cmd = ['squeue', '-u', 'zmobille3', '-n', 'job', '-h', '-o', '%i']

# Execute the squeue command and capture the output
output = subprocess.check_output(cmd).decode().strip() # get the running jobs
num_jobs = len(output.splitlines())

c = os.getcwd()

def run_sim(Nh, freq, pace_queue, run_name, seed):
    
    try:
        os.mkdir('qfiles')
    except FileExistsError:
        pass

    try:
        os.mkdir('reports')
    except FileExistsError:
        pass

    sname = f'qfiles/train_Nh{Nh}_f{int(freq)}_seed{seed}.sbatch'
    
    simname = f'Nh{Nh}_f{int(freq)}_seed{seed}'

    f = open(sname,'w')
    f.write('''#!/bin/bash\n''')
    f.write('''#SBATCH -Jjob\n''')
    f.write('''#SBATCH --account=gts-hchoi387-math\n''')
    f.write('''#SBATCH -N1 -n4\n''')
    f.write('''#SBATCH --mem-per-cpu=1G\n''')
    f.write('''#SBATCH -t2:00:00\n''')
    f.write('''#SBATCH -q%s\n'''%pace_queue)
    f.write(f'#SBATCH -oreports/{simname}-%j.out\n')
    #f.write('''cd %s\n'''%simdir)
    f.write('''module load anaconda3\n''')
    f.write('''conda activate cuda_trainsnn\n''')
    f.write('''python %s.py %s %s %s\n'''%(run_name, Nh, seed, freq))
    f.close()

    os.system(f'sbatch {sname}')

#contrast_values = [0.02, 0.05,0.1,0.18, 0.33]
#conditions =  [['Spont',0]] +[  ['PV', c] for c in contrast_values] + [ ['SOM', c] for c in contrast_values]
#
def sim_results_exist(Nh, f, seed):
    simdir = c + '/trainedModels/Nh%s/f%s'%(Nh,int(f))
    # simdir = '/storage/home/hcoda1/8/zmobille3/scratch/SpikeCoding/FLAP_2024/0325/trainedModels/Nh%s/pcon%s'%(Nh,pcon)
    full_path = simdir+'/seed%s.pth'%(seed)
    if os.path.isfile(full_path):
        return True
    return False

run_name = 'seed_freq_train'
pace_queue = 'inferno'
numseeds = int(sys.argv[1])

seedlist = [ii for ii in range(numseeds)]
Nh_list = [int(x) for x in np.logspace(1,3,3)]
#Nh_list = [int(10000)]
f_list = [float(f) for f in np.arange(5,55,5)]
#f_list = [5,10,15,18,20,22,25,30,35,40,45,50]
#f_list = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
maxjobs = 300

notdone = False
while True:
#
    output = subprocess.check_output(cmd).decode().strip()
    num_jobs = len(output.splitlines())
    print(f"Number of jobs running: {num_jobs}. Submit {maxjobs-num_jobs} jobs.")
    notdone=False
    for seed in seedlist:
        for Nh in Nh_list:
            for freq in f_list:
                simname = f'Nh{Nh}_f{int(freq)}_seed{seed}'
                ex = sim_results_exist(Nh, freq, seed)
                if ex:
                    print(f'{simname} exists')
                    continue
                else:
                    print(f'Nh{Nh}_f{int(freq)}_seed{seed} doesnt exist, adding...')
                
                run_sim(Nh, freq, pace_queue, run_name, seed)
                num_jobs += 1
                
                if num_jobs >= maxjobs:
                    notdone = True
                    break
            
            if num_jobs >= maxjobs:
                notdone = True
                break
                
        if num_jobs >= maxjobs:
            notdone = True
            break
            
    while notdone:
        print('sleeping...', end='')
        sleep(10)
        try:
            output = subprocess.check_output(cmd).decode().strip()
            num_jobs = len(output.splitlines())
            print(f"Number of jobs running: {num_jobs}.")
        except:
            print("squeue didn't work, trying again.")
            continue
        if num_jobs < 10:
            print('squeue empty, submit jobs.')
            break

    if notdone: continue
    else:
        print("All simulations completed.")
        break
        


