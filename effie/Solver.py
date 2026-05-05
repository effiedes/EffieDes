# -*- coding: utf-8 -*-
"""

"""
import math
import torch
import numpy as np
import pytoulbar2
import subprocess
import os

def add_hints(problem, nb_var, hints, top=999999):

    for i in range(nb_var):
        costs = np.ones(20) * top

        for p in hints[i]:
            costs[p] = 0 

        problem.AddFunction([i], costs)


def make_CFN(W, idx = None, var_names = None, domains=None, unary_costs = None, top=999999999, resolution=3):
    
    """
    Create a CFN object described by the W function.
    Input: - the matrix (numpy array W) of size (nb_var, nb_var, nb_aa*nb_aa)
           - a Boolean matrix idx (shape nb_var, nb_var) whose value is True for the constraint to consider 
           (default is None: all constraints are written)
           - int top (default 999999999)
           - int resolution (default 3)
           - int backtrack (default 9999999999)
    """

    Problem = pytoulbar2.CFN(top, resolution, vac=True)
    nb_var = W.shape[1]
    nb_aa = math.isqrt(int(W.shape[-1]))
    if idx is None:
        idx = np.ones((nb_var, nb_var))
        
    #Defining variables & domains
    for i in range(nb_var):
        var_name = ("x" + str(i+1) if var_names is None else var_names[i]) 
        Problem.AddVariable(var_name, range(0, nb_aa) if domains is None else domains)
        
    #Defining cost functions
    for i in range(nb_var):
    
        # unary costs
        if unary_costs is not None:
            Problem.AddFunction([i], unary_costs[i])
        else:
            Problem.AddFunction([i], np.diag(W[i, i].reshape(nb_aa, nb_aa)) * idx[i, i])

        for j in range(i + 1, nb_var):
            #binary costs
            Problem.AddFunction([i, j], W[i, j] * idx[i, j])
                
    return Problem

                       
def LR_BCD(W, y, missing = None, hint=None, nb_pred_seq = 20, filename = 'NSR_Cb', return_all_pred = False):
    
    path = './LR-BCD/build/'
    filename = 'lrbcd'
    binary = os.path.join(path, filename)
    if not os.path.isfile(binary):
        print("\n  LR-BCD binary not found at ", binary)
        print("  Please clone LR-BCD: 'git clone https://github.com/ValDurante/LR-BCD.git'")
        print("  Install the eigen3 C++ library: 'sudo apt install libeigen3-dev' or 'brew install eigen'")
        print("  and run: 'cd LR-BCD && mkdir build && cd build && cmake .. && make && cd ../..'\n")
        raise FileNotFoundError(f"LR-BCD binary not found at {binary}")
    
    sol_file = os.path.join(path,'sol_' + filename + '.txt')
    instance = os.path.join(path,filename + '.wcsp')
    cmd = f"{binary} {instance} 2 -it=5 -k=-4 -nbR={nb_pred_seq} -f={sol_file}"
    print(cmd)
    try:
        y=y.flatten().detach().cpu().numpy()
        W=W.detach().cpu().numpy()
    except:
        pass
    nb_var = len(y)
    W = W.reshape(nb_var,nb_var,-1)
    Problem = make_CFN(W, idx = None, resolution=3, domains=amino_acids1)
    if hint is not None:
        add_hints(Problem, nb_var, hint)
    Problem.Dump(instance)

    ### Run convex relaxation ###
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
               shell=True, preexec_fn=os.setsid) 
    p.communicate(timeout = 900)
    if p.poll() is None: # p.subprocess is alive
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    #.communicate to wait until the file is written before keep going

    file = open(sol_file, 'r')
    L = file.readlines()
    file.close()

    predictions = []
    for line in L[:-1]:
        line = line.strip().split(' ')
        line = [int(l) for l in line]
        line = np.array(line).reshape(nb_var, 20)
        predictions.append(np.argmax(line, axis = 1)) 

    NSR = []
    for i in range(len(predictions)):
        if missing is not None:
            NSR.append(np.sum((y-predictions[i] == 0)[~missing.cpu()])/torch.sum(~missing).item())
        else:
            NSR.append(np.sum((y-predictions[i] == 0))/nb_var)
    E = np.array([float(l) for l in L[-1].strip().split(' ')])
        
    if return_all_pred:
        return NSR, predictions
    else:
        return NSR[np.argmin(E)], predictions[np.argmin(E)]
    
