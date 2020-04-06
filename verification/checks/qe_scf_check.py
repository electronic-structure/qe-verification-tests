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

    return sn.max(sn.abs(stress_ref[i][j] - stress[i][j]) for i in range(2) for j in range(2))


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

    return sn.max(sn.abs(forces[i][j] - forces_ref[i][j]) for i in range(natoms) for j in range(2))

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
            sn.assert_lt(energy_diff(self.stdout, energy_ref), 1e-7, msg="Total energy is different"),
            sn.assert_lt(pressure_diff(self.stdout, P_ref), 2e-2, msg="Pressure is different"),
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

@rfm.parameterized_test(*([variant, ranks] for variant in ['native', 'sirius'] for ranks in [(2,4), (2,8)]))
class qe_Au_surf_scf(qe_scf_base_test):
    def __init__(self, variant, ranks):
        super().__init__(ranks[0], ranks[1], 'Au-surf', 'ausurf.in', variant,
            energy_ref=-11427.09402177,
            P_ref=-7.53,
            stress_ref=[[-8.82e-05, -0.0, -0.0], [-0.0, -8.37e-05, -1.64e-06], [-0.0, -1.64e-06, 1.832e-05]],
            forces_ref=[[-0.00099221, 0.00015991, 0.00836862], [-0.0009868, -0.0002161, 0.00808835],
                        [-0.00013254, 0.00022145, 0.00898978], [-0.00013204, -0.00016171, 0.0087064],
                        [6.109e-05, 0.00025217, 0.00904117], [6.094e-05, -0.00013157, 0.00875724],
                        [-9.624e-05, 0.00023366, 0.0089881], [-9.543e-05, -0.00014956, 0.00870461],
                        [0.0001045, 0.00023297, 0.00898803], [0.00010412, -0.00015002, 0.0087044],
                        [-9.048e-05, 0.00024575, 0.00904663], [-9.037e-05, -0.00013792, 0.00876264],
                        [0.0007742, 0.00026831, 0.00884048], [0.00077252, -0.00011435, 0.00855723],
                        [-0.00128616, 0.00041673, -0.0053258], [-0.00128208, -7.246e-05, -0.00543419],
                        [-0.00015674, 0.00038861, -0.00535447], [-0.00015625, -0.00010777, -0.00546311],
                        [7.385e-05, 0.0003942, -0.00534195], [7.359e-05, -0.00010326, -0.0054507],
                        [-0.00012538, 0.00039773, -0.00534011], [-0.00012479, -9.911e-05, -0.00544892],
                        [0.00013251, 0.0003978, -0.00533975], [0.00013202, -9.893e-05, -0.0054483],
                        [-0.00012529, 0.00039142, -0.00535197], [-0.00012506, -0.00010606, -0.00546059],
                        [0.00099157, 0.00039248, -0.00532452], [0.00099011, -0.00010373, -0.00543327],
                        [-0.00057523, -0.00010159, 0.00857694], [-0.00057612, -0.0005046, 0.00862758],
                        [-4.304e-05, -0.00012209, 0.00878225], [-4.305e-05, -0.00052597, 0.00883336],
                        [9.682e-05, -0.00013407, 0.00872741], [9.766e-05, -0.00053744, 0.008778],
                        [-4.46e-06, -0.00012095, 0.00876189], [-4.7e-06, -0.00052479, 0.00881328],
                        [-9.98e-05, -0.00012765, 0.00872775], [-0.00010051, -0.00053094, 0.00877846],
                        [9.694e-05, -0.00013373, 0.00877668], [9.698e-05, -0.00053772, 0.00882773],
                        [0.0008957, -0.00021827, 0.00807533], [0.00089956, -0.00061433, 0.00812601],
                        [-0.0007317, -0.00010732, -0.00536234], [-0.0007324, -0.00062356, -0.00533342],
                        [-5.038e-05, -0.00010894, -0.00539045], [-5.062e-05, -0.00062619, -0.00536122],
                        [0.00012905, -0.00010084, -0.00537966], [0.00012968, -0.00061723, -0.00535036],
                        [-6.07e-06, -0.00010374, -0.00537499], [-6.09e-06, -0.00062132, -0.00534602],
                        [-0.0001208, -0.00010266, -0.00538037], [-0.0001214, -0.00061907, -0.00535095],
                        [0.00011234, -0.00011421, -0.0053922], [0.00011232, -0.00063156, -0.00536323],
                        [0.0011606, -7.272e-05, -0.00536831], [0.00116593, -0.00058134, -0.00533767],
                        [-0.00099173, 0.0006908, -0.00840993], [-0.00098723, 0.00023394, -0.00808191],
                        [-0.00013265, 0.0006439, -0.00903037], [-0.00013214, 0.00017983, -0.00870158],
                        [6.119e-05, 0.00061408, -0.00908059], [6.107e-05, 0.00014918, -0.0087519],
                        [-9.547e-05, 0.00063121, -0.00902893], [-9.511e-05, 0.00016678, -0.00869969],
                        [0.00010407, 0.00063171, -0.00902974], [0.00010369, 0.00016759, -0.00870038],
                        [-9.1e-05, 0.00061989, -0.00908728], [-9.079e-05, 0.00015529, -0.00875832],
                        [0.00077548, 0.00059582, -0.00888228], [0.00077365, 0.00013231, -0.00855365],
                        [-0.000733, 0.00076195, 0.00519843], [-0.00073203, 0.0002231, 0.00536507],
                        [-5.084e-05, 0.00076505, 0.00522773], [-5.057e-05, 0.00022513, 0.0053943],
                        [0.00013044, 0.00075608, 0.00521601], [0.0001296, 0.00021655, 0.00538277],
                        [-5.95e-06, 0.00076075, 0.00521135], [-5.93e-06, 0.00022081, 0.00537784],
                        [-0.00012227, 0.00075798, 0.00521789], [-0.00012163, 0.00021866, 0.00538437],
                        [0.00011305, 0.00077013, 0.00523024], [0.00011287, 0.00023012, 0.00539693],
                        [0.00116505, 0.00071815, 0.00520432], [0.00115937, 0.0001868, 0.00537061],
                        [-0.00128241, 5.381e-05, 0.0053614], [-0.00128688, -0.00075071, 0.00553817],
                        [-0.00015701, 8.984e-05, 0.00539091], [-0.00015723, -0.00072576, 0.00556788],
                        [7.363e-05, 8.534e-05, 0.00537682], [7.357e-05, -0.00073164, 0.00555384],
                        [-0.00012435, 7.951e-05, 0.00537632], [-0.00012486, -0.00073604, 0.00555278],
                        [0.00013134, 7.992e-05, 0.00537622], [0.00013181, -0.00073587, 0.00555291],
                        [-0.00012475, 8.85e-05, 0.00538686], [-0.00012477, -0.00072812, 0.00556366],
                        [0.00099099, 8.585e-05, 0.00536073], [0.00099235, -0.00072908, 0.00553734],
                        [-0.0005743, 0.00011667, -0.00855742], [-0.00057481, -0.00029374, -0.00861953],
                        [-4.247e-05, 0.00014012, -0.00876535], [-4.247e-05, -0.00027084, -0.00882736],
                        [9.555e-05, 0.0001504, -0.00870798], [9.597e-05, -0.00026026, -0.00877011],
                        [-4.62e-06, 0.00013527, -0.00873955], [-4.56e-06, -0.00027586, -0.00880201],
                        [-9.76e-05, 0.00014469, -0.00871022], [-9.808e-05, -0.0002657, -0.00877231],
                        [9.538e-05, 0.00015098, -0.00875939], [9.545e-05, -0.00026018, -0.00882133],
                        [0.0008951, 0.00023327, -0.00805499], [0.00089848, -0.00016973, -0.00811716]])
        self.tags = {'qe-%s'%variant, 'parallel', 'Au-surf'}
        self.time_limit = '20m'

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
