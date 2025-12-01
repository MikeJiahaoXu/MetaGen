"""
Simulation Manager
Processes HFSS project files and runs electromagnetic simulations
"""
import os
from tqdm import tqdm
from utils import analyze
import argparse


def get_pending_simulations(simualtion_dir, result_dir):
    """Find incomplete simulations by checking result files"""
    return [f for f in os.listdir(simualtion_dir) 
           if f.endswith('.aedt') 
           and not os.path.exists(f"{result_dir}S21_re/{f[:-5]}.csv")]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--simualtion_dir', help='path to hfss aedt dataset', type=str, default="../output/")
    parser.add_argument('--result_dir', help='path to save csv reesults of S-params', type=str, default="../results/")
    args = parser.parse_args()

    while True:
        pending = get_pending_simulations(args.simualtion_dir, args.result_dir)
        if not pending:
            print("All simulations completed")
            break
            
        for project in tqdm(pending, desc="Processing Simulations"):
            if not os.path.exists(f"{args.simualtion_dir}{project}.lock"):
                analyze(project, args, face=1, ver="2022.2")  # Run HFSS simulation, the version is 2022r2, it's ok to change it to the version you have