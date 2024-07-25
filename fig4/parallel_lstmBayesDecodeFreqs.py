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
def run_sim(Nh, pcon, deltat, pace_queue, run_name, seed):
    
    try:
        os.mkdir('qfiles')
    except FileExistsError:
        pass

    try:
        os.mkdir('reports')
    except FileExistsError:
        pass

    sname = f'qfiles/decode_Nh{Nh}_pcon{pcon}_seed{seed}.sbatch'
    
    simname = f'Nh{Nh}_pcon{pcon}_seed{seed}'

    f = open(sname,'w')
    f.write('''#!/bin/bash\n''')
    f.write('''#SBATCH -Jjob\n''')
    f.write('''#SBATCH --account=gts-hchoi387-math\n''')
    f.write('''#SBATCH -N1 -n4\n''')
    f.write('''#SBATCH --mem-per-cpu=1G\n''')
    f.write('''#SBATCH -t4:00:00\n''')
    f.write('''#SBATCH -q%s\n'''%pace_queue)
    f.write(f'#SBATCH -oreports/{simname}-%j.out\n')
    f.write('''module load keras\n''')
    f.write('''conda activate trainsnn\n''')
    f.write('''python %s.py %s %s %s\n'''%(run_name, Nh, seed, deltat))
    f.close()

    os.system(f'sbatch {sname}')


def sim_results_exist(Nh, pcon, deltat, seed):
    simdir = c + '/LSTMbayesdecodeData/Nh%s/pcon%s/seed%s/deltat%s'%(Nh,pcon,seed,deltat)
    path_in4 = simdir+'/y4pred_in.npy'
    path_h4 = simdir+'/y4pred_h.npy'
    path_out4 = simdir+'/y4pred_out.npy'
    path_y4 = simdir+'/y4test.npy'
    path_in20 = simdir+'/y20pred_in.npy'
    path_h20 = simdir+'/y20pred_h.npy'
    path_out20 = simdir+'/y20pred_out.npy'
    path_y20 = simdir+'/y20test.npy'
    if (os.path.isfile(path_in4) and os.path.isfile(path_h4) and os.path.isfile(path_out4) and os.path.isfile(path_y4) and
        os.path.isfile(path_in20) and os.path.isfile(path_h20) and os.path.isfile(path_out20) and os.path.isfile(path_y20)):
        return True
    return False

run_name = 'seed_deltat_lstmDecodeBayesFreqs'
pace_queue = 'inferno'
numseeds = int(sys.argv[1])

seedlist = [ii for ii in range(numseeds)]
Nh_list = [int(x) for x in np.logspace(1,3,3)]
#deltat_list = np.arange(10,110,10)
deltat_list=[2,5,10,25,50]
pcon_list = [0.3]
maxjobs = 200

notdone = False
while True:
#
    output = subprocess.check_output(cmd).decode().strip()
    num_jobs = len(output.splitlines())
    print(f"Number of jobs running: {num_jobs}. Submit {maxjobs-num_jobs} jobs.")
    notdone=False
    for seed in seedlist:
        for Nh in Nh_list:
            for pcon in pcon_list:
                for deltat in deltat_list:
                    simname = f'Nh{Nh}_pcon{pcon}_deltat{deltat}_seed{seed}'
                    ex = sim_results_exist(Nh, pcon, deltat, seed)
                    if ex:
                        print(f'{simname} exists')
                        continue
                    else:
                        print(f'{simname} doesnt exist, adding...')

                    run_sim(Nh, pcon, deltat, pace_queue, run_name, seed)
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
        if num_jobs < 4:
            print('squeue empty, submit jobs.')
            break

    if notdone: continue
    else:
        print("All simulations completed.")
        break
        

