import os
import numpy as np
os.system('export MKL_NUM_THREADS=1')
from os.path import exists
from time import sleep, time
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

    sname = f'qfiles/run_Nh{Nh}_stimseed{stimseed}_modelseed{modelseed}.sbatch'
    
    simname = f'Nh{Nh}_stimseed{stimseed}_modelseed{modelseed}'

    f = open(sname,'w')
    f.write('''#!/bin/bash\n''')
    f.write('''#SBATCH -Jjob\n''')
    f.write('''#SBATCH --account=gts-hchoi387-math\n''')
    f.write('''#SBATCH -N1 -n4\n''')
    f.write('''#SBATCH --mem-per-cpu=1G\n''')
    f.write('''#SBATCH -t1:00:00\n''')
    f.write('''#SBATCH -q%s\n'''%pace_queue)
    f.write(f'#SBATCH -oreports/{simname}-%j.out\n')
    #f.write('''cd %s\n'''%simdir)
    f.write('''module load anaconda3\n''')
    f.write('''conda activate trainsnn\n''')
    f.write('''python %s.py %s %s %s\n'''%(run_name, Nh, stimseed, modelseed))
    f.close()

    os.system(f'sbatch {sname}')

#contrast_values = [0.02, 0.05,0.1,0.18, 0.33]
#conditions =  [['Spont',0]] +[  ['PV', c] for c in contrast_values] + [ ['SOM', c] for c in contrast_values]
#
def sim_results_exist(Nh, stimseed, modelseed):
    simdir = c + '/rawData/Nh%s/stimseed%s/modelseed%s'%(Nh,stimseed,modelseed)
    in_path = simdir+'/in_spks.npy'
    h_path = simdir+'/h_spks.npy'
    out_path = simdir+'/out_spks.npy'
    r_path = simdir+'/readout.npy'
    if os.path.isfile(in_path) and os.path.isfile(h_path) and os.path.isfile(out_path) and os.path.isfile(r_path):
        return True
    return False

run_name = 'seed_run'
pace_queue = 'inferno'
numseeds = int(sys.argv[1])

stimseedlist = [ii for ii in range(numseeds)]
Nh_list = [int(x) for x in np.logspace(1,3,3)]
modelseedlist = [int(0)]
#modelseedlist = [ii for ii in range(numseeds)]
maxjobs = 300

#modelseedlist=[int(0)]

notdone = False
while True:
#
    output = subprocess.check_output(cmd).decode().strip()
    num_jobs = len(output.splitlines())
    print(f"Number of jobs running: {num_jobs}. Submit {maxjobs-num_jobs} jobs.")
    notdone=False
    for stimseed in stimseedlist:
        for Nh in Nh_list:
            for modelseed in modelseedlist:
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
        

