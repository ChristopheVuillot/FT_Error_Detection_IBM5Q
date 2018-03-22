import ast
import os
import re
import statistics
import random
import numpy as np
from scipy.stats import t, norm, gaussian_kde
import matplotlib.pyplot as plt

from tools.Experiment_tools import CIRCUIT_NAMES


def plot_everything_raw(folder):
    list_file = os.listdir(folder)
    n_circuit = len(list_file)
    n_skipped = 0
    n_keapt = 0
    cmap = plt.cm.get_cmap('gist_ncar')
    plt.figure(figsize=(20, 20))
    for j, circuit_filename in enumerate(list_file):
        with open(folder+circuit_filename, 'r') as circuit_file:
            expe_list = circuit_file.readlines()
        stat_dist = []
        qasm_count = []
        for expe_data_string in expe_list:
            try:
                expe_data = ast.literal_eval(expe_data_string)
                qasm_count.append(expe_data['qasm_count'])
                stat_dist.append(expe_data['stat_dist'])
                n_keapt += 1
            except SyntaxError:
                n_skipped += 1
        plt.scatter(qasm_count, stat_dist, label=circuit_filename, marker='x', c=cmap(j/n_circuit))
    plt.title('all experiments')
    plt.legend(loc='lower left', bbox_to_anchor=(1, 0))
    plt.yscale('log')
    plt.grid()
    plt.show()
    print(n_skipped, n_keapt)

PLOT_LABELS = ['bare[1, 0]',
               'bare[2, 0]',
               'bare[2, 1]',
               'bare[2, 4]',
               'bare[3, 2]',
               'bare[3, 4]',
               'encoded|00>ftv1',
               'encoded|00>ftv2',
               'encoded|00>nftv1',
               'encoded|0+>',
               'encoded|00>+|11>']

RE_LABELS = [re.compile('[\\S]*\\[1, 0\\].txt'),
             re.compile('[\\S]*\\[2, 0\\].txt'),
             re.compile('[\\S]*\\[2, 1\\].txt'),
             re.compile('[\\S]*\\[2, 4\\].txt'),
             re.compile('[\\S]*\\[3, 2\\].txt'),
             re.compile('[\\S]*\\[3, 4\\].txt'),
             re.compile('[\\S]*>ftv1.txt'),
             re.compile('[\\S]*>ftv2.txt'),
             re.compile('[\\S]*nftv1.txt'),
             re.compile('e[\\S]*\\|0\\+>.txt'),
             re.compile('e[\\S]*\\|00>\\+\\|11>.txt')]

def plot_everything_averaged(folder, logscaley=True, sublabels=PLOT_LABELS, ci=.99, save_data_folder_pref=None):
    list_file = os.listdir(folder)
    n_skipped = 0
    n_kept = 0
    cmap = plt.cm.get_cmap('Paired')
    colors = [cmap(j/12) for j in [1,5,10,11,4,0,8,9,6,2,3]]
    qasm_counts = [[] for j in range(0, 12)]
    circuit_indices = [[] for j in range(0, 12)]
    stat_dists = [[] for j in range(0, 12)]
    post_select_r = [[] for j in range(0, 12)]
    stdevs = [[] for j in range(0, 12)]
    conf_ints = [[] for j in range(0, 12)]
    fig, ax = plt.subplots(figsize=(20, 20))
    for j, circuit_filename in enumerate(list_file):
        total = 0
        stat_dist_avg = 0
        values = []
        with open(folder+circuit_filename, 'r') as circuit_file:
            expe_list = circuit_file.readlines()
        for k, reg_ex in enumerate(RE_LABELS):
            if reg_ex.match(circuit_filename):
                index = k
                break
        for expe_data_string in expe_list:
            try:
                expe_data = ast.literal_eval(expe_data_string)
                total += 1
                stat_dist_avg += expe_data['stat_dist']
                post_select_r[index].append(expe_data['post_selection_ratio'])
                values.append(expe_data['stat_dist'])
                n_kept += 1
            except SyntaxError:
                n_skipped += 1
        stat_dists[index].append(stat_dist_avg/total)
        qasm_counts[index].append(expe_data['qasm_count'])
        for l, cn in enumerate(CIRCUIT_NAMES):
            if cn in circuit_filename:
                circuit_index = l+1
        circuit_indices[index].append(circuit_index)
        stdevs[index].append(statistics.stdev(values))
        ct = t.interval(ci, len(values)-1, loc=0, scale=1)[1]
        conf_ints[index].append(ct*statistics.stdev(values)/np.sqrt(len(values)))
    indices_to_plot = [PLOT_LABELS.index(pl) for pl in sublabels]
    for j in indices_to_plot:
        l1, l2, l3 = zip(*sorted(zip(circuit_indices[j], stat_dists[j], conf_ints[j])))
        ax.errorbar(l1, l2, yerr=l3, markersize=15, mew=3, fmt='x', label=PLOT_LABELS[j], c=colors[j])
        if save_data_folder_pref:
            with open(save_data_folder_pref + PLOT_LABELS[j] + '.dat', 'w') as data_file:
                data_file.write('index stat_dist conf_int99\n')
                for tup in zip(l1, l2, l3):
                    data_file.write('{} {} {}\n'.format(*tup))
    l1, l2 = zip(*sorted(zip(circuit_indices[0], qasm_counts[0])))
    ax2 = ax.twinx()
    ax2.plot(l1, l2, 'k-')
    if save_data_folder_pref:
        with open(save_data_folder_pref + 'bare_qasm_count.dat', 'w') as data_file:
            data_file.write('index qasm_count\n')
            for tup in zip(l1, l2):
                data_file.write('{} {}\n'.format(*tup))
    handles, labs = ax.get_legend_handles_labels()
    ax.set_title('all experiments')
    #ax.legend([h[0] for h in handles], labels, loc='lower left', bbox_to_anchor=(1, 0))
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0))
    if logscaley:
        ax.set_yscale('log')
    ax.grid(True)
    fig.tight_layout()
    plt.sca(ax)
    plt.xticks(range(1,21), [c[1:] for c in CIRCUIT_NAMES], rotation=60, horizontalalignment='right')
    plt.show()
    print(n_skipped, n_kept)
    print('\nAverage performance:\n')
    for k in range(0, 11):
        print(PLOT_LABELS[k], statistics.mean(stat_dists[k]))
    print('\nPost selection ratios:\n')
    for k in range(6, 10):
        print(PLOT_LABELS[k], statistics.mean(post_select_r[k]+post_select_r[10]+post_select_r[11]))

def plot_everything_averaged_diff(folder, logscaley=True, bareindex=1, ci=.99, plot_qasm_count=False, save_data_folder_pref=None):
    list_file = os.listdir(folder)
    n_skipped = 0
    n_kept = 0
    cmap = plt.cm.get_cmap('Paired')
    colors = [cmap(j/12) for j in [1,5,10,11,4,0,8,9,6,2,3]]
    qasm_counts = [[] for j in range(0, 12)]
    circuit_indices = [[] for j in range(0, 12)]
    stat_dists = [[] for j in range(0, 12)]
    stdevs = [[] for j in range(0, 12)]
    conf_ints = [[] for j in range(0, 12)]
    fig, ax = plt.subplots(figsize=(20, 20))
    for j, circuit_filename in enumerate(list_file):
        total = 0
        stat_dist_avg = 0
        values = []
        with open(folder+circuit_filename, 'r') as circuit_file:
            expe_list = circuit_file.readlines()
        for k, reg_ex in enumerate(RE_LABELS):
            if reg_ex.match(circuit_filename):
                index = k
                break
        for expe_data_string in expe_list:
            try:
                expe_data = ast.literal_eval(expe_data_string)
                total += 1
                stat_dist_avg += expe_data['stat_dist']
                values.append(expe_data['stat_dist'])
                n_kept += 1
            except SyntaxError:
                n_skipped += 1
        stat_dists[index].append(stat_dist_avg/total)
        qasm_counts[index].append(expe_data['qasm_count'])
        for l,cn in enumerate(CIRCUIT_NAMES):
            if cn in circuit_filename:
                circuit_index = l+1
        circuit_indices[index].append(circuit_index)
        stdevs[index].append(statistics.stdev(values))
        ct = t.interval(ci, len(values)-1, loc=0, scale=1)[1]
        conf_ints[index].append(ct*statistics.stdev(values)/np.sqrt(len(values)))
    if plot_qasm_count:
        ax2 = ax.twinx()
    ax.plot([j for j in range(-1,22)], [0 for j in range(-1,22)], '-r')
    indices_to_plot = [pl for pl in range(6,12)]
    for j in indices_to_plot:
        if plot_qasm_count:
            l1, l2 = zip(*sorted(zip(circuit_indices[j], qasm_counts[j])))
            ax2.plot(l1, l2, label=PLOT_LABELS[j], c=colors[j]) 
        for k in range(0,len(stat_dists[j])):
            bare_ref = stat_dists[bareindex][circuit_indices[bareindex].index(circuit_indices[j][k])]
            stat_dists[j][k] -= bare_ref
        ax.errorbar(np.array(circuit_indices[j]), np.array(stat_dists[j]), yerr=np.array(conf_ints[j]), markersize=15, mew=3, fmt='x', label=PLOT_LABELS[j], c=colors[j])
        if save_data_folder_pref:
            with open(save_data_folder_pref + PLOT_LABELS[j] + '-' + PLOT_LABELS[bareindex] + '.dat', 'w') as data_file:
                data_file.write('index stat_dist_diff conf_int99\n')
                for tup in sorted(zip(circuit_indices[j], stat_dists[j], conf_ints[j])):
                    data_file.write('{} {} {}\n'.format(*tup))
    handles, labs = ax.get_legend_handles_labels()
    ax.set_title('Encoded circuits compared to bare qubit pair '+PLOT_LABELS[bareindex][4:])
    if plot_qasm_count:
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 0))
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0))
    if logscaley:
        ax.set_yscale('log')
    ax.grid(True)
    ax.set_xlim([0,21])
    plt.sca(ax)
    plt.xticks(range(1,21), [c[1:] for c in CIRCUIT_NAMES], rotation=60, horizontalalignment='right')
    fig.tight_layout()
    plt.show()
    print(n_skipped, n_kept)
    for k in range(0,11):
        print(PLOT_LABELS[k],statistics.mean(stat_dists[k]))


def save_everything_calib_data_avg(folder, save_data_folder_pref):
    list_file = os.listdir(folder)
    n_skipped = 0
    n_kept = 0
    single_q_param_names = ['name', 'T1', 'T2', 'gateError', 'readoutError']
    #single_q_parameters = [dict.fromkeys(single_q_param_names) for j in range(0, 5)]
    single_q_parameters = [{} for j in range(0, 5)]
    multi_q_param_names = ['qubits', 'gateError']
    #multi_q_parameters = [dict.fromkeys(multi_q_param_names) for j in range(0, 6)]
    multi_q_parameters = [{} for j in range(0, 6)]
    fridge_T = []
    for j, circuit_filename in enumerate(list_file):
        with open(folder+circuit_filename, 'r') as circuit_file:
            expe_list = circuit_file.readlines()
        for expe_data_string in expe_list:
            try:
                expe_data = ast.literal_eval(expe_data_string)
                for i,param in enumerate(expe_data['calibration']['multiQubitGates']):
                    multi_q_parameters[i]['qubits'] = param['qubits']
                    for name in multi_q_param_names[1:]:
                        multi_q_parameters[i].setdefault(name, []).append(convert_parameter(param[name]))
                for i,param in enumerate(expe_data['calibration']['qubits']):
                    single_q_parameters[i]['name'] = param['name']
                    for name in single_q_param_names[1:]:
                        single_q_parameters[i].setdefault(name, []).append(convert_parameter(param[name]))
                fridge_T.append(expe_data['calibration']['fridgeParameters']['Temperature']['value'])
                n_kept += 1
            except SyntaxError:
                n_skipped += 1
    with open(save_data_folder_pref + 'multi_q.dat', 'w') as data_file:
        data_file.write('qubits gateError sigma(gateError)\n')
        for line in multi_q_parameters:
            data_file.write('{} {} {}\n'.format('-'.join([str(k) for k in line['qubits']]), statistics.mean(line['gateError']), statistics.stdev(line['gateError'])))
    with open(save_data_folder_pref + 'single_q.dat', 'w') as data_file:
        data_file.write('name T1 sigma(T1) T2 sigma(T2) gateError sigma(gateError) readoutError sigma(readoutError)\n')
        for line in single_q_parameters:
            data_file.write('{0} {1} {5} {2} {6} {3} {7} {4} {8}\n'.format(line['name'], *[statistics.mean(line[name]) for name in single_q_param_names[1:]], *[statistics.stdev(line[name]) for name in single_q_param_names[1:]])) 
    with open(save_data_folder_pref + 'temp.dat', 'w') as data_file:
        data_file.write('T sigma(T)\n')
        data_file.write('{} {}\n'.format(statistics.mean(fridge_T), statistics.stdev(fridge_T)))
    print(n_skipped, n_kept)
    

# Plotting one bare run next to one encoded run with the expected output distribution
def plot_one_random_expe(data_folder, circuit_name, deselect_labels=range(0,12), ci=.99):
    list_file = [filename for filename in os.listdir(data_folder) if circuit_name in filename]
    n_type = len(list_file)
    cmap = plt.cm.get_cmap('Paired')
    colors = [cmap(j/12) for j in [1,5,10,11,4,0,8,9,6,2,3]]
    N = 4;
    ind = np.arange(N)
    width = 1/(n_type+1)

    fig, ax = plt.subplots()
    hist = []

    for j, circuit_filename in enumerate(list_file):
        with open(data_folder+circuit_filename, 'r') as circuit_file:
            expe_string = random.choice(circuit_file.readlines())
        for k, reg_ex in enumerate(RE_LABELS):
            if reg_ex.match(circuit_filename):
                index = k
                break
        if index in deselect_labels:
            continue
        else:
            expe_data = ast.literal_eval(expe_string)
            hist.append(ax.bar(ind+j*width,
                               np.array(expe_data['experimental_distribution_array']),
                               width,
                               color=colors[index],
                               yerr=np.array(expe_data['stand_dev'])*norm.ppf(1/2+ci/2),
                               label=PLOT_LABELS[index]+'(stat dist: {} - r: {})'.format(expe_data['stat_dist'], expe_data['post_selection_ratio'])))

    ax.set_ylabel('Frequencies')
    ax.set_title('Performance on the circuit : ' + circuit_name)
    ax.set_xticks(ind + (n_type-1)*width/2)
    ax.set_xticklabels(['00', '01', '10', '11'])

    ax.legend(loc='lower left', bbox_to_anchor=(1, 0))

    plt.show()
