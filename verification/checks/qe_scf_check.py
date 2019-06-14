import itertools
import os
import json

import reframe as rfm
import reframe.utility.sanity as sn

test_folders = ['test1', 'test2', 'test3', 'test4', 'test5', 'test6', 'test7', 'test8',
    'test9', 'test10', 'test11', 'test12', 'test13', 'test14', 'test15', 'test16', 'test17']


#@sn.sanity_function
#def load_json(filename):
#    '''This will load a json data from a file.'''
#    raw_data = sn.extractsingle(r'(?s).+', filename).evaluate()
#    try:
#        return json.loads(raw_data)
#    except json.JSONDecodeError as e:
#        raise SanityError('failed to parse JSON file') from e
#
#@sn.sanity_function
#def energy_diff(filename, data_ref):
#    ''' Return the difference between obtained and reference total energies'''
#    parsed_output = load_json(filename)
#    return sn.abs(parsed_output['ground_state']['energy']['total'] -
#                       data_ref['ground_state']['energy']['total'])
#
#@sn.sanity_function
#def stress_diff(filename, data_ref):
#    ''' Return the difference between obtained and reference stress tensor components'''
#    parsed_output = load_json(filename)
#    if 'stress' in parsed_output['ground_state'] and 'stress' in data_ref['ground_state']:
#        return sn.sum(sn.abs(parsed_output['ground_state']['stress'][i][j] -
#                             data_ref['ground_state']['stress'][i][j]) for i in [0, 1, 2] for j in [0, 1, 2])
#    else:
#        return sn.abs(0)
#
#@sn.sanity_function
#def forces_diff(filename, data_ref):
#    ''' Return the difference between obtained and reference atomic forces'''
#    parsed_output = load_json(filename)
#    if 'forces' in parsed_output['ground_state'] and 'forces' in data_ref['ground_state']:
#        na = parsed_output['ground_state']['num_atoms'].evaluate()
#        return sn.sum(sn.abs(parsed_output['ground_state']['forces'][i][j] -
#                             data_ref['ground_state']['forces'][i][j]) for i in range(na) for j in [0, 1, 2])
#    else:
#        return sn.abs(0)
#
class qe_scf_base_test(rfm.RunOnlyRegressionTest):
    def __init__(self, num_ranks, test_folder, input_file_name, use_sirius, etot):
        super().__init__()
        self.descr = 'SCF check'
        self.valid_systems = ['osx', 'daint']
        self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-intel']

        self.num_tasks = num_ranks
        if self.current_system.name == 'daint':
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 12
            self.variables = {
                'OMP_NUM_THREADS': str(self.num_cpus_per_task),
                'MKL_NUM_THREADS': str(self.num_cpus_per_task)
            }

        self.executable = 'pw.x'
        self.sourcesdir = '../' + test_folder

        #self.sanity_patterns = sn.all([
        #    sn.assert_found(r'JOB DONE', self.stdout, msg="Calculation didn't converge"),
        #    #sn.assert_lt(energy_diff(fout, data_ref), 1e-5, msg="Total energy is different"),
        #    #sn.assert_lt(stress_diff(fout, data_ref), 1e-5, msg="Stress tensor is different"),
        #    #sn.assert_lt(forces_diff(fout, data_ref), 1e-5, msg="Atomic forces are different")
        #])

        self.executable_opts = ["-i %s"%input_file_name]
        if use_sirius:
            self.executable_opts.append('-sirius')

        #self.reference = {
        #    'osx:cpu': {
        #        'time': (0.67, None, 0.05, 'sec')
        #    }
        #}

        #self.reference = {
        #    'dom:mc': {
        #        'time': (159.0, None, 0.05, 's'),
        #    },
        #    'daint:mc': {
        #        'time': (151.6, None, 0.05, 's')
        #    },
        #}

        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }

        energy_all = sn.extractall(r'!\s+total energy\s+=\s+(?P<energy>\S+) Ry',
                                  self.stdout, 'energy', float)
        energy = energy_all[-1]

        self.sanity_patterns = sn.all([
            sn.assert_found(r'convergence has been achieved', self.stdout),
            sn.assert_reference(energy, etot, -1e-9, 1e-9)
        ])

        #self.perf_patterns = {
        #    'time': sn.extractsingle(r'PWSCF        :\s+(?P<cpu>\S+)s CPU \s+(?P<wall>\S+)s WALL',
        #                             self.stdout, 'wall', float)
        #}

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        if self.current_system.name in ['daint']:
            self.job.launcher.options = ["-c %i"%self.num_cpus_per_task, "-n %i"%self.num_tasks, '--hint=nomultithread']

@rfm.simple_test
class qe_Si(qe_scf_base_test):
    def __init__(self):
        super().__init__(1, 'Si', 'pw.in', False, -19.31899683)
        self.tags = {'serial'}

@rfm.simple_test
class qe_sirius_Si(qe_scf_base_test):
    def __init__(self):
        super().__init__(1, 'Si', 'pw.in', True, -19.31899883)
        self.tags = {'serial'}

@rfm.simple_test
class qe_Si_vc_relax(qe_scf_base_test):
    def __init__(self):
        super().__init__(1, 'Si-vc-relax', 'pw.in', False, -19.31468808)
        self.tags = {'serial'}

@rfm.simple_test
class qe_sirius_Si_vc_relax(qe_scf_base_test):
    def __init__(self):
        super().__init__(1, 'Si-vc-relax', 'pw.in', True, -19.31469019)
        self.tags = {'serial'}

