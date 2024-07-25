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

def run_sim(Nh, stimseed, pace_queue, run_name, modelseed):
    
    try:
        os.mkdir('qfiles')
    except FileExistsError:
        pass

    try:
        os.mkdir('reports')
    except FileExistsError:
        pass

    sname = f'qfiles/train_Nh{Nh}_stimseed{stimseed}_modelseed{modelseed}.sbatch'
    
    simname = f'Nh{Nh}_stimseed{stimseed}_modelseed{modelseed}'

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
    f.write('''module load anaconda3/2022.05\n''')
    # f.write('''source /storage/home/hcoda1/8/zmobille3/.bashrc\n''')
    f.write('''conda activate cuda_trainsnn\n''')
    f.write('''python %s.py %s %s %s\n'''%(run_name, Nh, stimseed, modelseed))
    f.close()

    os.system(f'sbatch {sname}')


def sim_results_exist(Nh, stimseed, modelseed):
    simdir = c + '/trainedModels/Nh%s'%(Nh)
    full_path = simdir+'/modelseed%s_stimseed%s.pth'%(modelseed,stimseed)
    if os.path.isfile(full_path):
        return True
    return False

run_name = 'seed_train'
pace_queue = 'inferno'
numseeds = int(sys.argv[1])

stimseedlist = [ii for ii in range(numseeds)]
Nh_list = [int(x) for x in np.logspace(1,3,3)]
modelseed_list = [int(0)]
#modelseed_list = [ii for ii in range(numseeds)]
maxjobs = 300

notdone = False
while True:
#
    output = subprocess.check_output(cmd).decode().strip()
    num_jobs = len(output.splitlines())
    print(f"Number of jobs running: {num_jobs}. Submit {maxjobs-num_jobs} jobs.")
    notdone=False
    for stimseed in stimseedlist:
        for Nh in Nh_list:
            for modelseed in modelseed_list:
                simname = f'Nh{Nh}_stimseed{stimseed}_modelseed{modelseed}'
                ex = sim_results_exist(Nh, stimseed, modelseed)
                if ex:
                    print(f'{simname} exists')
                    continue
                else:
                    print(f'Nh{Nh}_stimseed{stimseed}_modelseed{modelseed} doesnt exist, adding...')
                
                run_sim(Nh, stimseed, pace_queue, run_name, modelseed)
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
        if num_jobs < 6:
            print('squeue empty, submit jobs.')
            break

    if notdone: continue
    else:
        print("All simulations completed.")
        break
        


