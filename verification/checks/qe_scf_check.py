import itertools
import os
import re
import json

import reframe as rfm
import reframe.utility.sanity as sn

@sn.sanity_function
def energy_diff(ostream, energy_ref):
    ''' Return the difference between obtained and reference total energies'''

    # take last energy value (is case of vc-relax)
    energy = sn.extractsingle(r'!\s+total energy\s+=\s+(?P<energy>\S+) Ry',
                              ostream, 'energy', float, item=-1)
    return sn.abs(energy - energy_ref)

@sn.sanity_function
def pressure_diff(ostream, pressure_ref):
    ''' Return the difference between obtained and reference total energies'''

    # take last pressure value (is case of vc-relax)
    pressure = sn.extractsingle(r'\s+total\s+stress.+\(kbar\)\s+P\=\s*(?P<pressure>\S+)',
                                ostream, 'pressure', float, item=-1)
    return sn.abs(pressure - pressure_ref)

@sn.sanity_function
def stress_diff(ostream, stress_ref):
    ''' Return the difference between obtained and reference stress tensor components'''

    # take last match (is case of vc-relax)
    raw_data = sn.extractsingle(r'total\s+stress.+\(kbar\)\s+P\=.+\s+.+\s+.+\s+.+', ostream, item=-1).evaluate()
    lines = raw_data.splitlines()

    stress = []
    for i in range(2):
        vals = lines[i + 1].split()
        stress.append([float(vals[j]) for j in range(2)])

    return sn.sum(sn.abs(stress_ref[i][j] - stress[i][j]) for i in range(2) for j in range(2))


@sn.sanity_function
def forces_diff(ostream, forces_ref):
    ''' Return the difference between obtained and reference atomic forces'''

    # get number of atoms
    natoms = sn.extractsingle(r'\s+number\sof\satoms\/cell\s+\=\s+(?P<natoms>\S+)', ostream, 'natoms', int).evaluate()

    fmt = r'     Forces acting on atoms \(cartesian axes, Ry\/au\):\s+'
    for i in range(natoms):
        fmt += r'.+\s+'

    # take last match (is case of vc-relax)
    raw_data = sn.extractsingle(fmt, ostream, item=-1).evaluate()

    lines = raw_data.splitlines()[2:]
    forces = []
    ia = 1
    for line in lines:
        result = re.search(r'\s+atom\s+(?P<atom>\S+)\s+type\s+\S+\s+force\s+\=\s+(?P<force>.+)', line)
        if result:
            sn.assert_eq(ia, int(result.group('atom')), msg='Wrong index of atom: {0} != {1}').evaluate()
            vals = result.group('force').split()
            forces.append([float(vals[i]) for i in range(2)])
            ia += 1

    return sn.sum(sn.abs(forces[i][j] - forces_ref[i][j]) for i in range(natoms) for j in range(2))

class qe_scf_base_test(rfm.RunOnlyRegressionTest):
    def __init__(self, num_ranks_k, num_ranks_d, test_folder, input_file_name, variant, energy_ref, P_ref, stress_ref, forces_ref):
        super().__init__()
        self.descr = 'SCF check'
        self.valid_systems = ['osx', 'daint']
        self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-intel']

        self.num_tasks = num_ranks_k * num_ranks_d
        if self.current_system.name == 'daint':
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 12
            self.variables = {
                'OMP_NUM_THREADS': str(self.num_cpus_per_task),
                'MKL_NUM_THREADS': str(self.num_cpus_per_task)
            }

        self.executable = 'pw.x'
        self.sourcesdir = '../' + test_folder

        self.executable_opts = ["-i %s"%input_file_name, "-npool %i"%num_ranks_k, "-ndiag %i"%num_ranks_d]
        if variant == 'sirius':
            self.executable_opts.append('-sirius')

        patterns = [
            sn.assert_found(r'convergence has been achieved', self.stdout),
            sn.assert_lt(energy_diff(self.stdout, energy_ref), 1e-8, msg="Total energy is different"),
            sn.assert_lt(pressure_diff(self.stdout, P_ref), 1e-2, msg="Pressure is different"),
            sn.assert_lt(stress_diff(self.stdout, stress_ref), 1e-5, msg="Stress tensor is different"),
            sn.assert_lt(forces_diff(self.stdout, forces_ref), 1e-5, msg="Atomic forces are different")
        ]
        if variant == 'sirius':
            patterns.append(sn.assert_found(r'SIRIUS.+git\shash', self.stdout))

        self.sanity_patterns = sn.all(patterns)

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        if self.current_system.name in ['daint']:
            self.job.launcher.options = ["-c %i"%self.num_cpus_per_task, "-n %i"%self.num_tasks, '--hint=nomultithread']

#--------------------------#
# Example of a simple test #
#--------------------------#
#@rfm.simple_test
#class qe_Si(qe_scf_base_test):
#    def __init__(self):
#        super().__init__(1, 'Si', 'pw.in', use_sirius=False,
#            energy_ref=-19.31881789,
#            P_ref=-55.34,
#            stress_ref=[[-0.00033422, -7.453e-05, -7.453e-05], [-7.453e-05, -0.00039715, 2.31e-06], [-7.453e-05, 2.31e-06, -0.00039715]],
#            forces_ref=[[-0.00021611, 0.00516719, 0.00516719], [0.00021611, -0.00516719, -0.00516719]])
#        self.tags = {'serial', 'qe-native'}


@rfm.parameterized_test(['native'], ['sirius'])
class qe_Si_scf(qe_scf_base_test):
    def __init__(self, variant):
        super().__init__(1, 1, 'Si', 'pw.in', variant,
            energy_ref=-19.31881789,
            P_ref=-55.34,
            stress_ref=[[-0.00033422, -7.453e-05, -7.453e-05], [-7.453e-05, -0.00039715, 2.31e-06], [-7.453e-05, 2.31e-06, -0.00039715]],
            forces_ref=[[-0.00021611, 0.00516719, 0.00516719], [0.00021611, -0.00516719, -0.00516719]])
        self.tags = {'qe-%s'%variant, 'serial'}


@rfm.parameterized_test(['native'], ['sirius'])
class qe_Si_vc_relax(qe_scf_base_test):
    def __init__(self, variant):
        super().__init__(1, 1, 'Si-vc-relax', 'pw.in', variant,
            energy_ref=-19.31469883,
            P_ref=-4.44,
            stress_ref=[[-3.164e-05, -5.7e-07, -5.7e-07], [-5.7e-07, -2.949e-05, 1.19e-06], [-5.7e-07, 1.19e-06, -2.949e-05]],
            forces_ref=[[-7.916e-05, 0.00014243, 0.00014243], [7.916e-05, -0.00014243, -0.00014243]])
        self.tags = {'qe-%s'%variant, 'serial'}


@rfm.parameterized_test(['native'], ['sirius'])
class qe_LiF_nc_scf(qe_scf_base_test):
    def __init__(self, variant):
        super().__init__(1, 1, 'LiF-nc', 'pw.in', variant,
            energy_ref=-47.18970921,
            P_ref=-3310.56,
            stress_ref=[[-0.02250471, -1.96e-06, -1.96e-06], [-1.96e-06, -0.02250471, -1.96e-06], [-1.96e-06, -1.96e-06, -0.02250471]],
            forces_ref=[[0.00351991, 0.00351991, 0.00351991], [-0.00351991, -0.00351991, -0.00351991]])
        self.tags = {'qe-%s'%variant, 'serial'}

@rfm.parameterized_test(*([variant, ranks] for variant in ['native', 'sirius'] for ranks in [(4,1), (1,4)]))
class qe_Si63Ge_scf(qe_scf_base_test):
    def __init__(self, variant, ranks):
        super().__init__(ranks[0], ranks[1], 'Si63Ge', 'pw.in', variant,
            energy_ref=-933.05993665,
            P_ref=19.15,
            stress_ref=[[0.0001302, 0.0, 0.0], [0.0, 0.0001302, 0.0], [0.0, 0.0, 0.0001302]],
            forces_ref=[[0.0, -0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -0.0, 0.0],
                        [0.00046938, -2.646e-05, -2.646e-05], [-0.00046938, -2.646e-05, 2.646e-05],
                        [-0.00046938, 2.646e-05, -2.646e-05], [0.00046938, 2.646e-05, 2.646e-05],
                        [-3.631e-05, 3.022e-05, 3.022e-05], [3.631e-05, 3.022e-05, -3.022e-05],
                        [3.631e-05, -3.022e-05, 3.022e-05], [-3.631e-05, -3.022e-05, -3.022e-05],
                        [-2.646e-05, 0.00046938, -2.646e-05], [-2.646e-05, -0.00046938, 2.646e-05],
                        [3.022e-05, -3.631e-05, 3.022e-05], [3.022e-05, 3.631e-05, -3.022e-05],
                        [2.646e-05, -0.00046938, -2.646e-05], [2.646e-05, 0.00046938, 2.646e-05],
                        [-3.022e-05, 3.631e-05, 3.022e-05], [-3.022e-05, -3.631e-05, -3.022e-05],
                        [-2.646e-05, -2.646e-05, 0.00046938], [3.022e-05, 3.022e-05, -3.631e-05],
                        [-2.646e-05, 2.646e-05, -0.00046938], [3.022e-05, -3.022e-05, 3.631e-05],
                        [2.646e-05, -2.646e-05, -0.00046938], [-3.022e-05, 3.022e-05, 3.631e-05],
                        [2.646e-05, 2.646e-05, 0.00046938], [-3.022e-05, -3.022e-05, -3.631e-05],
                        [3.331e-05, 3.331e-05, -6.763e-05], [-2.024e-05, -2.024e-05, 2.024e-05],
                        [1.52e-06, 1e-05, -1e-05], [3.331e-05, 6.763e-05, -3.331e-05],
                        [1e-05, 1.52e-06, -1e-05], [6.763e-05, 3.331e-05, -3.331e-05],
                        [-0.01100845, -0.01100845, 0.01100845], [1e-05, 1e-05, -1.52e-06],
                        [3.331e-05, -6.763e-05, 3.331e-05], [1.52e-06, -1e-05, 1e-05],
                        [-2.024e-05, 2.024e-05, -2.024e-05], [3.331e-05, -3.331e-05, 6.763e-05],
                        [1e-05, -1e-05, 1.52e-06], [-0.01100845, 0.01100845, -0.01100845],
                        [6.763e-05, -3.331e-05, 3.331e-05], [1e-05, -1.52e-06, 1e-05],
                        [-6.763e-05, 3.331e-05, 3.331e-05], [-1e-05, 1.52e-06, 1e-05],
                        [-1e-05, 1e-05, 1.52e-06], [0.01100845, -0.01100845, -0.01100845],
                        [2.024e-05, -2.024e-05, -2.024e-05], [-3.331e-05, 3.331e-05, 6.763e-05],
                        [-3.331e-05, 6.763e-05, 3.331e-05], [-1.52e-06, 1e-05, 1e-05],
                        [0.01100845, 0.01100845, 0.01100845], [-1e-05, -1e-05, -1.52e-06],
                        [-1e-05, -1.52e-06, -1e-05], [-6.763e-05, -3.331e-05, -3.331e-05],
                        [-1.52e-06, -1e-05, -1e-05], [-3.331e-05, -6.763e-05, -3.331e-05],
                        [-3.331e-05, -3.331e-05, -6.763e-05], [2.024e-05, 2.024e-05, 2.024e-05]])
        self.tags = {'qe-%s'%variant, 'parallel'}

#@rfm.simple_test
#class qe_LiF_nc_vc_relax(qe_scf_base_test):
#    def __init__(self):
#        super().__init__(1, 'LiF-nc', 'pw.in', False, -49.19540885)
#        self.tags = {'serial'}
#
#@rfm.simple_test
#class qe_sirius_LiF_nc_vc_relax(qe_scf_base_test):
#    def __init__(self):
#        super().__init__(1, 'LiF-nc', 'pw.in', True, -49.19541319)
#        self.tags = {'serial'}
#
#@rfm.simple_test
#class qe_LiF_paw_vc_relax(qe_scf_base_test):
#    def __init__(self):
#        super().__init__(1, 'LiF-paw', 'pw.in', False, -73.36174370)
#        self.tags = {'serial'}
#
#@rfm.simple_test
#class qe_sirius_LiF_paw_vc_relax(qe_scf_base_test):
#    def __init__(self):
#        super().__init__(1, 'LiF-paw', 'pw.in', True, -73.36174696)
#        self.tags = {'serial'}
#
#@rfm.simple_test
#class qe_LiF_uspp_vc_relax(qe_scf_base_test):
#    def __init__(self):
#        super().__init__(1, 'LiF-uspp', 'pw.in', False, -63.36417257)
#        self.tags = {'serial'}
#
#@rfm.simple_test
#class qe_sirius_LiF_uspp_vc_relax(qe_scf_base_test):
#    def __init__(self):
#        super().__init__(1, 'LiF-uspp', 'pw.in', True, -63.36416794)
#        self.tags = {'serial'}
#
#@rfm.simple_test
#class qe_MnO_ldapu(qe_scf_base_test):
#    def __init__(self):
#        super().__init__(1, 'MnO-LDA+U', 'pw.in', False, -491.69850186)
#        self.tags = {'serial'}
#
#@rfm.simple_test
#class qe_sirius_MnO_ldapu(qe_scf_base_test):
#    def __init__(self):
#        super().__init__(1, 'MnO-LDA+U', 'pw.in', True, -491.62465892)
#        self.tags = {'serial'}
#
