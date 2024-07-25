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
def run_sim(Nh, stimseed, modelseed, deltat, pace_queue, run_name):
    
    try:
        os.mkdir('qfiles')
    except FileExistsError:
        pass

    try:
        os.mkdir('reports')
    except FileExistsError:
        pass

    sname = f'qfiles/decode_Nh{Nh}_stimseed{stimseed}_modelseed{modelseed}_deltat{deltat}.sbatch'
    
    simname = f'decode_Nh{Nh}_stimseed{stimseed}_modelseed{modelseed}_deltat{deltat}'

    f = open(sname,'w')
    f.write('''#!/bin/bash\n''')
    f.write('''#SBATCH -Jjob\n''')
    f.write('''#SBATCH --account=gts-hchoi387-math\n''')
    f.write('''#SBATCH -N1 -n4\n''')
    f.write('''#SBATCH --mem-per-cpu=1G\n''')
    f.write('''#SBATCH -t8:00:00\n''')
    f.write('''#SBATCH -q%s\n'''%pace_queue)
    f.write(f'#SBATCH -oreports/{simname}-%j.out\n')
    f.write('''module load keras\n''')
    f.write('''conda activate trainsnn\n''')
    f.write('''python %s.py %s %s %s %s\n'''%(run_name, Nh, stimseed, modelseed, deltat))
    f.close()

    os.system(f'sbatch {sname}')


def sim_results_exist(Nh, stimseed, modelseed, deltat):
    simdir = c + '/LSTMbayesdecodeDataShuffle/Nh%s/stimseed%s/modelseed%s/deltat%s'%(Nh,stimseed,modelseed,deltat)
    #simdir = '/storage/home/hcoda1/8/zmobille3/scratch/SpikeCoding/FLAP_2024/0325/decodeData/Nh%s/pcon%s/seed%s/deltat%s'%(Nh,pcon,seed,deltat)
    path1 = simdir+'/ytest_pred_in.npy'
    path2 = simdir+'/ytest_pred_h.npy'
    path3 = simdir+'/ytest_pred_out.npy'
    path4 = simdir+'/ytest_in.npy'	
    path5 = simdir+'/ytest_h.npy'
    path6 = simdir+'/ytest_out.npy'
    if os.path.isfile(path1) and os.path.isfile(path2) and os.path.isfile(path3) and os.path.isfile(path4) and os.path.isfile(path5) and os.path.isfile(path6):
        return True
    return False

run_name = 'seed_deltat_lstmDecodeShuffle'
pace_queue = 'inferno'
numseeds = int(sys.argv[1])
#modelseed = 0
stimseedlist = [ii for ii in range(numseeds)]
Nh_list = [int(x) for x in np.logspace(1,3,3)]
modelseedlist = [int(0)]
# modelseedlist = [ii for ii in range(numseeds)]
#deltat_list = np.arange(10,110,10)
deltat_list=[2,5,10,25,50]
pcon_list = [0.3]
maxjobs = 300

notdone = False
while True:
#
    output = subprocess.check_output(cmd).decode().strip()
    num_jobs = len(output.splitlines())
    print(f"Number of jobs running: {num_jobs}. Submit {maxjobs-num_jobs} jobs.")
    notdone=False
    for Nh in Nh_list:
        for stimseed in stimseedlist:
            for modelseed in modelseedlist:
                for deltat in deltat_list:
                    simname = f'Nh{Nh}_stimseed{stimseed}_modelseed{modelseed}_deltat{deltat}'
                    ex = sim_results_exist(Nh, stimseed, modelseed, deltat)
                    if ex:
                        print(f'{simname} exists')
                        continue
                    else:
                        print(f'{simname} doesnt exist, adding...')

                    run_sim(Nh, stimseed, modelseed, deltat, pace_queue, run_name)
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
        

