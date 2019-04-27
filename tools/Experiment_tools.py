###########################################################################################
#            Tools for demonstrating fault-tolerance on the IBM 5Q chip
#
#   contributor : Christophe Vuillot
#   affiliations : - JARA Institute for Quantum Information, RWTH Aachen university
#                  - QuTech, TU Delft
#
###########################################################################################

import ast
import time
import numpy as np
from qiskit import QISKitError

# Functions that create all the circuits inside a given QuantumProgram module
#############################################################################


# Misc aux circuits
###################


def swap_circuit(pair, quantump, qri=0):
    '''swap_circuit(pair, quantump, qri=0)
    Creates the swap circuit in the QuantumProgram quantump
    between the two qubits in the pair, the CNOTs used are ctrl:0 -> targ:1
    '''
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuitswap = quantump.create_circuit("SWAP" + str(pair), qrs, crs)
    qcircuitswap.cx(qrs[qri][pair[0]], qrs[qri][pair[1]])
    qcircuitswap.h(qrs[qri][pair[0]])
    qcircuitswap.h(qrs[qri][pair[1]])
    qcircuitswap.cx(qrs[qri][pair[0]], qrs[qri][pair[1]])
    qcircuitswap.h(qrs[qri][pair[0]])
    qcircuitswap.h(qrs[qri][pair[1]])
    qcircuitswap.cx(qrs[qri][pair[0]], qrs[qri][pair[1]])
    return qcircuitswap


def measure_all(quantump, qri=0, cri=0):
    '''measure_all(quantump, qri=0, cri=0)
    Creates the circuit measuring all outputs.
    '''
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuitmeasure = quantump.create_circuit("Measure all", qrs, crs)
    qcircuitmeasure.measure(qrs[qri][0], crs[cri][0])
    qcircuitmeasure.measure(qrs[qri][1], crs[cri][1])
    qcircuitmeasure.measure(qrs[qri][2], crs[cri][2])
    qcircuitmeasure.measure(qrs[qri][3], crs[cri][3])
    qcircuitmeasure.measure(qrs[qri][4], crs[cri][4])
    return qcircuitmeasure


# The encoded preparations
##########################


def encoded_00_prep_ftv1(quantump, qri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qc_ftv1 = quantump.create_circuit("e|00>ftv1", qrs, crs)
    qc_ftv1.h(qrs[qri][2])
    qc_ftv1.cx(qrs[qri][2], qrs[qri][0])
    qc_ftv1.cx(qrs[qri][2], qrs[qri][1])
    qc_ftv1.h(qrs[qri][2])
    qc_ftv1.h(qrs[qri][3])
    qc_ftv1.cx(qrs[qri][3], qrs[qri][2])
    qc_ftv1.h(qrs[qri][2])
    qc_ftv1.h(qrs[qri][3])
    qc_ftv1.cx(qrs[qri][2], qrs[qri][4])
    qc_ftv1.cx(qrs[qri][2], qrs[qri][0])
    return qc_ftv1


def encoded_00_prep_nftv1(quantump, qri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qc_nftv1 = quantump.create_circuit("e|00>nftv1", qrs, crs)
    qc_nftv1.h(qrs[qri][3])
    qc_nftv1.cx(qrs[qri][3], qrs[qri][4])
    qc_nftv1.cx(qrs[qri][3], qrs[qri][2])
    qc_nftv1.cx(qrs[qri][2], qrs[qri][1])
    return qc_nftv1


def encoded_00_prep_ftv2(quantump, qri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qc_ftv2 = quantump.create_circuit("e|00>ftv2", qrs, crs)
    qc_ftv2.h(qrs[qri][3])
    qc_ftv2.cx(qrs[qri][3], qrs[qri][2])
    qc_ftv2.h(qrs[qri][2])
    qc_ftv2.h(qrs[qri][3])
    qc_ftv2.cx(qrs[qri][2], qrs[qri][1])
    qc_ftv2.cx(qrs[qri][3], qrs[qri][4])
    qc_ftv2.h(qrs[qri][4])
    qc_ftv2.extend(swap_circuit([2, 4], quantump))
    qc_ftv2.cx(qrs[qri][2], qrs[qri][0])
    qc_ftv2.cx(qrs[qri][1], qrs[qri][0])
    qc_ftv2.h(qrs[qri][4])
    return qc_ftv2


def encoded_0p_prep(quantump, qri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qc_0p = quantump.create_circuit("e|0+>", qrs, crs)
    qc_0p.h(qrs[qri][1])
    qc_0p.h(qrs[qri][3])
    qc_0p.cx(qrs[qri][3], qrs[qri][2])
    qc_0p.extend(swap_circuit([2, 1], quantump))
    qc_0p.cx(qrs[qri][2], qrs[qri][4])
    return qc_0p


def encoded_2cat_prep(quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qc_2cat = quantump.create_circuit("e|00>+|11>", qrs, crs)
    qc_2cat.h(qrs[qri][2])
    qc_2cat.h(qrs[qri][3])
    qc_2cat.cx(qrs[qri][2], qrs[qri][1])
    qc_2cat.cx(qrs[qri][3], qrs[qri][4])
    return qc_2cat


# The bare preparations
#######################


def bare_00_prep(pair, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_bare_00 = quantump.create_circuit("b|00>" + str(pair), qrs, crs)
    return qcircuit_bare_00


def bare_0p_prep(pair, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_bare_0p = quantump.create_circuit("b|0+>" + str(pair), qrs, crs)
    qcircuit_bare_0p.h(qrs[qri][pair[1]])
    return qcircuit_bare_0p


def bare_2cat_prep(pair, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_bare_2cat = quantump.create_circuit("b|00>+|11>" + str(pair), qrs, crs)
    qcircuit_bare_2cat.h(qrs[qri][pair[0]])
    qcircuit_bare_2cat.cx(qrs[qri][pair[0]], qrs[qri][pair[1]])
    return qcircuit_bare_2cat


# The encoded gates
###################


def encoded_X1_circuit(mapping, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_encoded_X1 = quantump.create_circuit("eX1", qrs, crs)
    qcircuit_encoded_X1.x(qrs[qri][mapping[0]])
    qcircuit_encoded_X1.x(qrs[qri][mapping[1]])
    return qcircuit_encoded_X1


def encoded_X2_circuit(mapping, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_encoded_X2 = quantump.create_circuit("eX2", qrs, crs)
    qcircuit_encoded_X2.x(qrs[qri][mapping[0]])
    qcircuit_encoded_X2.x(qrs[qri][mapping[2]])
    return qcircuit_encoded_X2


def encoded_Z1_circuit(mapping, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_encoded_Z1 = quantump.create_circuit("eZ1", qrs, crs)
    qcircuit_encoded_Z1.z(qrs[qri][mapping[1]])
    qcircuit_encoded_Z1.z(qrs[qri][mapping[3]])
    return qcircuit_encoded_Z1


def encoded_Z2_circuit(mapping, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_encoded_Z2 = quantump.create_circuit("eZ2", qrs, crs)
    qcircuit_encoded_Z2.z(qrs[qri][mapping[2]])
    qcircuit_encoded_Z2.z(qrs[qri][mapping[3]])
    return qcircuit_encoded_Z2


def encoded_CZ_circuit(mapping, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_encoded_CZ = quantump.create_circuit("eCZ", qrs, crs)
    qcircuit_encoded_CZ.s(qrs[qri][mapping[0]])
    qcircuit_encoded_CZ.s(qrs[qri][mapping[1]])
    qcircuit_encoded_CZ.s(qrs[qri][mapping[2]])
    qcircuit_encoded_CZ.s(qrs[qri][mapping[3]])
    return qcircuit_encoded_CZ


def encoded_HHS_circuit(mapping, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_encoded_HHS = quantump.create_circuit("eHHS", qrs, crs)
    qcircuit_encoded_HHS.h(qrs[qri][mapping[0]])
    qcircuit_encoded_HHS.h(qrs[qri][mapping[1]])
    qcircuit_encoded_HHS.h(qrs[qri][mapping[2]])
    qcircuit_encoded_HHS.h(qrs[qri][mapping[3]])
    return qcircuit_encoded_HHS


# The bare gates
################


def bare_X1_circuit(pair, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_bare_X1 = quantump.create_circuit("bX1" + str(pair), qrs, crs)
    qcircuit_bare_X1.x(qrs[qri][pair[0]])
    return qcircuit_bare_X1


def bare_X2_circuit(pair, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_bare_X2 = quantump.create_circuit("bX2" + str(pair), qrs, crs)
    qcircuit_bare_X2.x(qrs[qri][pair[1]])
    return qcircuit_bare_X2


def bare_Z1_circuit(pair, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_bare_Z1 = quantump.create_circuit("bZ1" + str(pair), qrs, crs)
    qcircuit_bare_Z1.z(qrs[qri][pair[1]])
    return qcircuit_bare_Z1


def bare_Z2_circuit(pair, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_bare_Z2 = quantump.create_circuit("bZ2" + str(pair), qrs, crs)
    qcircuit_bare_Z2.z(qrs[qri][pair[1]])
    return qcircuit_bare_Z2


def bare_CZ_circuit(pair, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_bare_CZ = quantump.create_circuit("bCZ" + str(pair), qrs, crs)
    qcircuit_bare_CZ.h(qrs[qri][pair[1]])
    qcircuit_bare_CZ.cx(qrs[qri][pair[0]], qrs[0][pair[1]])
    qcircuit_bare_CZ.h(qrs[qri][pair[1]])
    return qcircuit_bare_CZ


def bare_HHS_circuit(pair, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_bare_HHS = quantump.create_circuit("bHHS" + str(pair), qrs, crs)
    qcircuit_bare_HHS.h(qrs[qri][pair[0]])
    qcircuit_bare_HHS.h(qrs[qri][pair[1]])
    return qcircuit_bare_HHS


# The dictionaries for all circuits
###################################


DICT_ENCODED = dict(zip(['eX1', 'eX2', 'eZ1', 'eZ2', 'eHHS', 'eCZ', 'e|00>ftv1',
                         'e|00>ftv2', 'e|00>nftv1', 'e|0+>', 'e|00>+|11>'],
                        [encoded_X1_circuit,
                         encoded_X2_circuit,
                         encoded_Z1_circuit,
                         encoded_Z2_circuit,
                         encoded_HHS_circuit,
                         encoded_CZ_circuit,
                         encoded_00_prep_ftv1,
                         encoded_00_prep_ftv2,
                         encoded_00_prep_nftv1,
                         encoded_0p_prep,
                         encoded_2cat_prep]))


DICT_BARE = dict(zip(['bX1', 'bX2', 'bZ1', 'bZ2', 'bHHS', 'bCZ', 'b|00>', 'b|0+>', 'b|00>+|11>'],
                     [bare_X1_circuit,
                      bare_X2_circuit,
                      bare_Z1_circuit,
                      bare_Z2_circuit,
                      bare_HHS_circuit,
                      bare_CZ_circuit,
                      bare_00_prep,
                      bare_0p_prep,
                      bare_2cat_prep]))


# The circuits for the experiment with input state and output distribution
##########################################################################


CIRCUITS = [[['X1', 'HHS', 'CZ', 'X2'], '|00>', [0.25, 0.25, 0.25, 0.25]],
            [['HHS', 'Z1', 'CZ'], '|00>', [0.25, 0.25, 0.25, 0.25]],
            [['HHS', 'Z1', 'Z2'], '|00>', [0.25, 0.25, 0.25, 0.25]],
            [['HHS', 'Z2', 'CZ'], '|00>', [0.25, 0.25, 0.25, 0.25]],
            [['Z2', 'X2'], '|00>+|11>', [0, .5, .5, 0]],
            [['X1', 'Z2'], '|0+>', [0, .5, 0, .5]],
            [['HHS', 'Z1'], '|00>', [0.25, 0.25, 0.25, 0.25]],
            [['HHS', 'CZ'], '|00>', [0.25, 0.25, 0.25, 0.25]],
            [['X1', 'X2'], '|00>', [0, 0, 0, 1]],
            [['HHS', 'Z2'], '|00>', [0.25, 0.25, 0.25, 0.25]],
            [['X1'], '|00>+|11>', [0, .5, .5, 0]],
            [['X1'], '|0+>', [0, .5, 0, .5]],
            [['HHS'], '|00>', [0.25, 0.25, 0.25, 0.25]],
            [['Z2'], '|00>+|11>', [.5, 0, 0, .5]],
            [['Z2'], '|0+>', [.5, 0, .5, 0]],
            [['X1'], '|00>', [0, 1, 0, 0]],
            [['X2'], '|00>', [0, 0, 1, 0]],
            [[], '|00>+|11>', [.5, 0, 0, .5]],
            [[], '|0+>', [.5, 0, .5, 0]],
            [[], '|00>', [1, 0, 0, 0]]]


CIRCUIT_NAMES = ['M' + '-'.join(reversed(c[0])) + c[1] for c in reversed(CIRCUITS)]
QASM_BARE_ORDER = [0, 1, 3, 4, 2, 5, 7, 8, 11, 6, 9, 10, 13, 14, 15, 17, 12, 16, 18, 19]
CIRCUIT_NAMES = [CIRCUIT_NAMES[j] for j in QASM_BARE_ORDER]


# The names of the different versions for encoding |00> and the chosen mapping
##############################################################################


ENCODED_VERSION_LIST = ['ftv1', 'ftv2', 'nftv1']
MAPPING = [3, 2, 1, 4]
CODEWORDS = [['0000', '1111'], ['1100', '0011'], ['1010', '0101'], ['1001', '0110']]
MAPPED_CODEWORDS = [[], [], [], []]
for i, cl in enumerate(CODEWORDS):
    for c in cl:
        MAPPED_CODEWORDS[i].append(''.join(list(reversed([c[j - 1] for j in MAPPING]))) + '0')


# Function that assembles all circuits within a given QuantumProgram module
###########################################################################


def all_circuits(quantump,
                 possible_pairs,
                 mapping=MAPPING,
                 circuits=CIRCUITS,
                 dict_bare=DICT_BARE,
                 dict_encoded=DICT_ENCODED,
                 encoded_version_list=ENCODED_VERSION_LIST):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    circuit_names = []
    for lc in circuits:
        for pair in possible_pairs:
            qcirc = quantump.create_circuit('bM' + '-'.join(reversed(lc[0])) + lc[1] + str(pair), qrs, crs)
            circuit_names.append('bM' + '-'.join(reversed(lc[0])) + lc[1] + str(pair))
            qcirc.extend(dict_bare['b' + lc[1]](pair, quantump))
            number_swap = 0
            for g in lc[0]:
                if g[0] == 'X' or g[0] == 'Z':
                    key = 'b' + g[0] + str(((int(g[1]) - 1 + number_swap) % 2) + 1)
                elif g[0] == 'H':
                    number_swap += 1
                    key = 'b' + g
                else:
                    key = 'b' + g
                qcirc.extend(dict_bare[key](pair, quantump))
            qcirc.extend(measure_all(quantump))
        if lc[1] == '|00>':
            for v in encoded_version_list:
                qcirc = quantump.create_circuit('eM' + '-'.join(reversed(lc[0])) + lc[1] + v, qrs, crs)
                circuit_names.append('eM' + '-'.join(reversed(lc[0])) + lc[1] + v)
                qcirc.extend(dict_encoded['e' + lc[1] + v](quantump))
                for g in lc[0]:
                    qcirc.extend(dict_encoded['e' + g](mapping, quantump))
                qcirc.extend(measure_all(quantump))
        else:
            qcirc = quantump.create_circuit('eM' + '-'.join(reversed(lc[0])) + lc[1], qrs, crs)
            circuit_names.append('eM' + '-'.join(reversed(lc[0])) + lc[1])
            qcirc.extend(dict_encoded['e' + lc[1]](quantump))
            for gate in lc[0]:
                qcirc.extend(dict_encoded['e' + gate](mapping, quantump))
            qcirc.extend(measure_all(quantump))
    return circuit_names


# Callback function for the run circuits
########################################


def post_treatment(res):
    '''Callback function to write the results into a file after the jobs are finished.
    '''
    with open('data/callback.log', 'a') as logfile:
        logfile.write(str(time.asctime(time.localtime(time.time()))) + ':' + res.get_status() + ' - id: ' + res.get_job_id() + '\n')
    circuit_names = res.get_names()
    try:
        for circuit_name in circuit_names:
            circuit_data = res.get_data(circuit_name)
            filename = 'data/Raw_counts/' + circuit_name + '_' + circuit_data['date'] + '.txt'
            with open(filename, 'w') as data_file:
                data_file.write(str(circuit_data['counts']))
        with open('data/completed.txt', 'a') as completed_file:
            completed_file.write(res.get_job_id() + '\n')
    except QISKitError as qiskit_err:
        print(qiskit_err)
        if str(qiskit_err) == '\'Time Out\'':
            with open('data/timed_out.txt', 'a') as timed_out_file:
                timed_out_file.write(res.get_job_id() + '\n')


def post_treatment_list(results):
    '''Callback function to write the results into a file after the jobs are finished.
    '''
    for res in results:
        with open('data/callback.log', 'a') as logfile:
            logfile.write(str(time.asctime(time.localtime(time.time()))) + ':' + res.get_status() + ' - id: ' + res.get_job_id() + '\n')
        circuit_names = res.get_names()
        try:
            for circuit_name in circuit_names:
                circuit_data = res.get_data(circuit_name)
                filename = 'data/Raw_counts/' + circuit_name + '_' + circuit_data['date'] + '.txt'
                with open(filename, 'w') as data_file:
                    data_file.write(str(circuit_data['counts']))
            with open('data/completed.txt', 'a') as completed_file:
                completed_file.write(res.get_job_id() + '\n')
        except QISKitError as qiskit_err:
            if str(qiskit_err) == '\'Time Out\'':
                with open('data/timed_out.txt', 'a') as timed_out_file:
                    timed_out_file.write(res.get_job_id() + '\n')


# Function to fetch previously timed out results


def fetch_previous(filename, api):
    '''Function that fetch previously ran experiements whose ids are stored in data/filename
    '''
    new = 0
    with open('data/' + filename, 'r') as ids_file_read:
        id_lines = ids_file_read.readlines()
    with open('data/' + filename, 'w') as ids_file_write:
        for id_line in id_lines:
            id_string = id_line.rstrip()
            job_result = api.get_job(id_string)
            if not job_result['status'] == 'COMPLETED':
                ids_file_write.write(id_line)
            else:
                new += 1
                with open('data/completed_' + filename, 'a') as comp_file:
                    comp_file.write(id_line)
                with open('data/API_dumps/api_dump_' + id_string + '.txt', 'w') as data_file:
                    data_file.write(str(job_result))
    return new


# Functions to analyse gathered data
####################################

# Function to get the dictionary of qasm vs circuit name


def get_qasm_name_dict(compiled_qobj_list):
    dictionary = {}
    for n, v in zip(sum([[circuit['compiled_circuit_qasm'] for circuit in batch['circuits']] for batch in compiled_qobj_list], []),
                    sum([[circuit['name'] for circuit in batch['circuits']] for batch in compiled_qobj_list], [])):
        dictionary.setdefault(n, []).append(v)
    return dictionary


def api_data_to_dict(res, name):
    data_dict = {'name': name}
    data_dict.setdefault('raw_counts', {}).update(res['data']['counts'])
    data_dict['counts'] = {'00': 0, '01': 0, '10': 0, '11': 0, 'err': 0, 'total_valid': 0}
    data_dict['qasm_count'] = len([q_instr for q_instr in res['qasm'].split('\n') if len(q_instr) > 0]) - 3
    n = len(name)
    number_H = name.count('H') / 2

    if name[0] == 'b':
        circuit_info = [c for c in CIRCUITS if '-'.join(reversed(c[0])) + c[1] == name[2:n - 6]][0]
        data_dict['expected_distribution_array'] = np.array(circuit_info[2], dtype=float)
        pair = ast.literal_eval(name[n - 6:n])
        data_dict['version'] = 'bare'
        if number_H % 2 == 1:
            pair.reverse()
        for key in res['data']['counts']:
            data_dict['counts'][''.join([key[4 - j] for j in reversed(pair)])] += res['data']['counts'][key]
            data_dict['counts']['total_valid'] += res['data']['counts'][key]

    elif name[0] == 'e':
        if 'nftv' in name[n - 5:n]:
            circuit_info = [c for c in CIRCUITS if '-'.join(reversed(c[0])) + c[1] == name[2:n - 5]][0]
        elif 'ftv' in name[n - 5:n]:
            circuit_info = [c for c in CIRCUITS if '-'.join(reversed(c[0])) + c[1] == name[2:n - 4]][0]
        else:
            circuit_info = [c for c in CIRCUITS if '-'.join(reversed(c[0])) + c[1] == name[2:n]][0]
        data_dict['expected_distribution_array'] = np.array(circuit_info[2], dtype=float)
        data_dict['version'] = 'encoded'
        for key in res['data']['counts']:
            found = False
            for i, codeword_list in enumerate(MAPPED_CODEWORDS):
                if key in codeword_list:
                    data_dict['counts']["{0:02b}".format(i)] += res['data']['counts'][key]
                    data_dict['counts']['total_valid'] += res['data']['counts'][key]
                    found = True
                    break
            if not found:
                data_dict['counts']['err'] += res['data']['counts'][key]

    data_dict['experimental_distribution_array'] = np.array([data_dict['counts'][s] / data_dict['counts']['total_valid']
                                                             for s in ['00', '01', '10', '11']], dtype=float)
    data_dict['post_selection_ratio'] = data_dict['counts']['total_valid'] / (data_dict['counts']['err'] + data_dict['counts']['total_valid'])
    data_dict['stat_dist'] = .5 * sum(np.abs(data_dict['experimental_distribution_array'] - data_dict['expected_distribution_array']))
    data_dict['stand_dev'] = np.sqrt(data_dict['experimental_distribution_array'] * (1 - data_dict['experimental_distribution_array']) / data_dict['counts']['total_valid'])
    stat_dist_stand_dev = 0
    for j in range(0, 4):
        stat_dist_stand_dev += data_dict['experimental_distribution_array'][j] * (1 - data_dict['experimental_distribution_array'][j]) / (4 * data_dict['counts']['total_valid'])
    for i in range(0, 4):
        for j in range(0, 4):
            if i != j:
                stat_dist_stand_dev += data_dict['experimental_distribution_array'][i] * data_dict['experimental_distribution_array'][j] / (4 * data_dict['counts']['total_valid'])
    stat_dist_stand_dev = np.sqrt(stat_dist_stand_dev)
    data_dict['stat_dist_stand_dev'] = stat_dist_stand_dev
    data_dict['stand_dev'] = list(data_dict['stand_dev'])
    data_dict['experimental_distribution_array'] = list(data_dict['experimental_distribution_array'])
    data_dict['expected_distribution_array'] = list(data_dict['expected_distribution_array'])
    return data_dict


def process_api_dump(filename, dict_qasm_name, dict_res={}):
    with open(filename, 'r') as api_dump_file:
        job_results = ast.literal_eval(api_dump_file.read())
    for res in job_results['qasms']:
        names = dict_qasm_name['OPENQASM 2.0;' + res['qasm']]
        for name in names:
            res_entry = api_data_to_dict(res, name)
            res_entry['calibration'] = job_results['calibration']
            dict_res.setdefault(name, []).append(res_entry)
            with open('data/Processed_data/' + name + '.txt', 'a') as circuit_file:
                circuit_file.write(str(res_entry) + '\n')
    return dict_res


def process_all_api_dumps(file_of_files_to_process, file_of_already_processed_files, dict_qasm_name):
    n_processed = 0
    with open(file_of_already_processed_files, 'r') as file_processed:
        processed = file_processed.readlines()
    with open(file_of_files_to_process, 'r') as file_to_process:
        to_process = file_to_process.readlines()
    with open(file_of_already_processed_files, 'a') as file_processed:
        for filename in to_process:
            if filename not in processed:
                n_processed += 1
                process_api_dump('data/API_dumps/api_dump_' + filename.rstrip() + '.txt', dict_qasm_name)
                file_processed.write(filename)
    return n_processed
