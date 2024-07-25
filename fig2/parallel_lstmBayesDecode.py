import os
os.system('export MKL_NUM_THREADS=1')
from os.path import exists
from time import sleep, time
import numpy as np
import pickle
import subprocess
import sys

cmd = ['squeue', '-u', 'zmobille3', '-n', 'job', '-h', '-o', '%i']

# Execute the squeue command and capture the output
output = subprocess.check_output(cmd).decode().strip() # get the running jobs
num_jobs = len(output.splitlines())

c = os.getcwd()
def run_sim(Nh, freq, deltat, pace_queue, run_name, seed):
    
    try:
        os.mkdir('qfiles')
    except FileExistsError:
        pass

    try:
        os.mkdir('reports')
    except FileExistsError:
        pass

    sname = f'qfiles/decode_Nh{Nh}_freq{freq}_seed{seed}_deltat{deltat}.sbatch'
    
    simname = f'Nh{Nh}_freq{freq}_seed{seed}_deltat{deltat}'

    f = open(sname,'w')
    f.write('''#!/bin/bash\n''')
    f.write('''#SBATCH -Jjob\n''')
    f.write('''#SBATCH --account=gts-hchoi387-math\n''')
    f.write('''#SBATCH -N1 --ntasks-per-node=4\n''')
    f.write('''#SBATCH --mem-per-cpu=16G\n''')
    f.write('''#SBATCH -t4:00:00\n''')
    f.write('''#SBATCH -q%s\n'''%pace_queue)
    f.write(f'#SBATCH -oreports/{simname}-%j.out\n')
    f.write('''module load keras\n''')
    f.write('''conda activate trainsnn\n''')
    f.write('''python %s.py %s %s %s %s\n'''%(run_name, Nh, seed, freq, deltat))
    f.close()

    os.system(f'sbatch {sname}')


def sim_results_exist(Nh, freq, deltat, seed):
    simdir = c + '/LSTMbayesdecodeData/f%s/Nh%s/seed%s/deltat%s'%(int(freq),Nh,seed,deltat)
    #simdir = '/storage/home/hcoda1/8/zmobille3/scratch/SpikeCoding/FLAP_2024/0325/decodeData/Nh%s/pcon%s/seed%s/deltat%s'%(Nh,pcon,seed,deltat)
    path_in = simdir+'/ypred_in.npy'
    path_h = simdir+'/ypred_h.npy'
    path_out = simdir+'/ypred_out.npy'
    path_y = simdir+'/ytest.npy'
    if os.path.isfile(path_in) and os.path.isfile(path_h) and os.path.isfile(path_out) and os.path.isfile(path_y):
        return True
    return False

run_name = 'seed_freq_deltat_lstmBayesDecode'
pace_queue = 'inferno'
numseeds = int(sys.argv[1])

seedlist = [ii for ii in range(numseeds)]
Nh_list = [int(x) for x in np.logspace(1,3,3)]
#deltat_list = np.arange(10,60,10)
#deltat_list = [2,5,10,20,30,40,50]
#deltat_list = np.arange(5,55,5)
deltat_list = [2,5,10,25,50]
# deltat_list = np.arange(10,110,10)
#deltat_list = [5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250]
f_list = [int(f) for f in np.arange(5,55,5)]
#f_list = [18,22]
#f_list = [10,11,12,13,14,15,16,17,18,19,20]
#f_list = [10,20,30,40,50]
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
                for deltat in deltat_list:
                    simname = f'Nh{Nh}_freq{freq}_deltat{deltat}_seed{seed}'
                    ex = sim_results_exist(Nh, freq, deltat, seed)
                    if ex:
                        print(f'{simname} exists')
                        continue
                    else:
                        print(f'{simname} doesnt exist, adding...')

                    run_sim(Nh, freq, deltat, pace_queue, run_name, seed)
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
        if num_jobs < 20:
            print('squeue empty, submit jobs.')
            break

    if notdone: continue
    else:
        print("All simulations completed.")
        break
        

