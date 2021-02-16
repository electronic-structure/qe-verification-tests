import re

import reframe as rfm
import reframe.utility.sanity as sn

@sn.sanity_function
def get_energy(ostream):
    ''' Get energy from the output stream'''

    # take last energy value (is case of vc-relax)
    return sn.extractsingle(r'!\s+total energy\s+=\s+(?P<energy>\S+) Ry',
                            ostream, 'energy', float, item=-1)

@sn.sanity_function
def get_pressure(ostream):
    ''' Get pressure from the output stream'''

    # take last pressure value (is case of vc-relax)
    return sn.extractsingle(r'\s+total\s+stress.+\(kbar\)\s+P\=\s*(?P<pressure>\S+)',
                            ostream, 'pressure', float, item=-1)

@sn.sanity_function
def get_stress(ostream):
    ''' Get stress tensor from the output stream'''

    # take last match (is case of vc-relax)
    raw_data = sn.extractsingle(r'total\s+stress.+\(kbar\)\s+P\=.+\s+.+\s+.+\s+.+', ostream, item=-1).evaluate()
    lines = raw_data.splitlines()

    stress = []
    for i in range(2):
        vals = lines[i + 1].split()
        stress.append([float(vals[j]) for j in range(2)])

    return stress

@sn.sanity_function
def get_forces(ostream):
    ''' Get forces from the output stream'''

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

    return forces

@sn.sanity_function
def energy_diff(ostream, ostream_ref):
    ''' Return the difference between obtained and reference total energies'''
    return sn.abs(get_energy(ostream) - get_energy(ostream_ref))

@sn.sanity_function
def pressure_diff(ostream, ostream_ref):
    ''' Return the difference between obtained and reference total energies'''
    return sn.abs(get_pressure(ostream) - get_pressure(ostream_ref))

@sn.sanity_function
def stress_diff(ostream, ostream_ref):
    ''' Return the difference between obtained and reference stress tensor components'''

    stress = get_stress(ostream)
    stress_ref = get_stress(ostream_ref)
    return sn.max(sn.abs(stress_ref[i][j] - stress[i][j]) for i in range(2) for j in range(2))

@sn.sanity_function
def forces_diff(ostream, ostream_ref):
    ''' Return the difference between obtained and reference atomic forces'''

    forces = get_forces(ostream)
    forces_ref = get_forces(ostream_ref)

    na = 0
    for e in forces: na += 1
    na_ref = 0
    for e in forces_ref: na_ref += 1

    sn.assert_eq(na, na_ref, msg='Wrong length of forces array: {0} != {1}').evaluate()

    return sn.max(sn.abs(forces[i][j] - forces_ref[i][j]) for i in range(na) for j in range(2))

class qe_scf_base_test(rfm.RunOnlyRegressionTest):
    def __init__(self, num_ranks_k, num_ranks_d, test_folder, variant):
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

        self.executable_opts = ["-i pw.in", "-npool %i"%num_ranks_k, "-ndiag %i"%num_ranks_d]
        if variant == 'sirius':
            self.executable_opts.append('-sirius_scf')

        patterns = [
            sn.assert_found(r'convergence has been achieved', self.stdout),
            sn.assert_lt(energy_diff(self.stdout, 'out.txt'), 1e-5, msg="Total energy is different"),
            sn.assert_lt(pressure_diff(self.stdout, 'out.txt'), 1e-1, msg="Pressure is different"),
            sn.assert_lt(stress_diff(self.stdout, 'out.txt'), 1e-4, msg="Stress tensor is different"),
            sn.assert_lt(forces_diff(self.stdout, 'out.txt'), 1e-4, msg="Atomic forces are different")
        ]
        if variant == 'sirius':
            patterns.append(sn.assert_found(r'SIRIUS.+git\shash', self.stdout))

        self.sanity_patterns = sn.all(patterns)

    @rfm.run_after('setup')
    def set_launcher_options(self):
        if self.current_system.name in ['daint']:
            self.job.launcher.options = ["-c %i"%self.num_cpus_per_task, "-n %i"%self.num_tasks, '--hint=nomultithread']

#--------------------------#
# Example of a simple test #
#--------------------------#
#@rfm.simple_test
#class qe_Si(qe_scf_base_test):
#    def __init__(self):
#        super().__init__(1, 1, 'Si', 'native')
#        self.tags = {'serial', 'qe-native'}

@rfm.parameterized_test(['native'], ['sirius'])
class qe_Si_scf(qe_scf_base_test):
    def __init__(self, variant):
        super().__init__(1, 1, 'Si', variant)
        self.tags = {'qe-%s'%variant, 'serial'}

@rfm.parameterized_test(['native'], ['sirius'])
class qe_Si_vc_relax(qe_scf_base_test):
    def __init__(self, variant):
        super().__init__(1, 1, 'Si-vc-relax', variant)
        self.tags = {'qe-%s'%variant, 'serial'}

@rfm.parameterized_test(['native'], ['sirius'])
class qe_LiF_gga_nc_scf(qe_scf_base_test):
    def __init__(self, variant):
        super().__init__(1, 1, 'LiF-gga-nc', variant)
        self.tags = {'qe-%s'%variant, 'serial'}

@rfm.parameterized_test(['native'], ['sirius'])
class qe_LiF_lda_nc_scf(qe_scf_base_test):
    def __init__(self, variant):
        super().__init__(1, 1, 'LiF-lda-nc', variant)
        self.tags = {'qe-%s'%variant, 'serial'}

@rfm.parameterized_test(['native'], ['sirius'])
class qe_LiF_lda_uspp_scf(qe_scf_base_test):
    def __init__(self, variant):
        super().__init__(1, 1, 'LiF-lda-uspp', variant)
        self.tags = {'qe-%s'%variant, 'serial'}

@rfm.parameterized_test(['native'], ['sirius'])
class qe_LiF_esm_scf(qe_scf_base_test):
    def __init__(self, variant):
        super().__init__(1, 1, 'LiF-esm', variant)
        self.tags = {'qe-%s'%variant, 'serial'}

@rfm.parameterized_test(*([variant, ranks] for variant in ['native', 'sirius'] for ranks in [(1,1), (2,1), (4,1)]))
class qe_CdCO3_scf(qe_scf_base_test):
    def __init__(self, variant, ranks):
        super().__init__(ranks[0], ranks[1], 'CdCO3', variant)
        self.tags = {'qe-%s'%variant, 'parallel'}

@rfm.parameterized_test(*([variant, ranks] for variant in ['native', 'sirius'] for ranks in [(4,1), (1,4)]))
class qe_Si63Ge_scf(qe_scf_base_test):
    def __init__(self, variant, ranks):
        super().__init__(ranks[0], ranks[1], 'Si63Ge', variant)
        self.tags = {'qe-%s'%variant, 'parallel'}

@rfm.parameterized_test(*([variant, ranks] for variant in ['native', 'sirius'] for ranks in [(2,4), (2,8)]))
class qe_Au_surf_scf(qe_scf_base_test):
    def __init__(self, variant, ranks):
        super().__init__(ranks[0], ranks[1], 'Au-surf', variant)
        self.tags = {'qe-%s'%variant, 'parallel', 'Au-surf'}
        self.time_limit = '20m'

@rfm.parameterized_test(*([variant, ranks] for variant in ['native', 'sirius'] for ranks in [(4,1), (8,1)]))
class qe_HfNi5_scf(qe_scf_base_test):
    def __init__(self, variant, ranks):
        super().__init__(ranks[0], ranks[1], 'HfNi5', variant)
        self.tags = {'qe-%s'%variant, 'parallel', 'hfni5'}

@rfm.parameterized_test(*([variant, ranks] for variant in ['native', 'sirius'] for ranks in [(3,1), (5,1)]))
class qe_NiO_afm_scf(qe_scf_base_test):
    def __init__(self, variant, ranks):
        super().__init__(ranks[0], ranks[1], 'NiO-afm', variant)
        self.tags = {'qe-%s'%variant, 'parallel'}

@rfm.parameterized_test(*([variant, ranks] for variant in ['native', 'sirius'] for ranks in [(4,1), (8,1)]))
class qe_FeSe2_2D_scf(qe_scf_base_test):
    def __init__(self, variant, ranks):
        super().__init__(ranks[0], ranks[1], 'FeSe2-2D', variant)
        self.tags = {'qe-%s'%variant, 'parallel'}

@rfm.parameterized_test(*([variant, ranks] for variant in ['native', 'sirius'] for ranks in [(1,1), (2,1), (3,1)]))
class qe_Ni_ldapu_scf(qe_scf_base_test):
    def __init__(self, variant, ranks):
        super().__init__(ranks[0], ranks[1], 'Ni-ldapu', variant)
        self.tags = {'qe-%s'%variant, 'parallel'}

@rfm.parameterized_test(*([variant, ranks] for variant in ['native', 'sirius'] for ranks in [(1,1), (2,1), (3,1)]))
class qe_PrNiO_ldapu_scf(qe_scf_base_test):
    def __init__(self, variant, ranks):
        super().__init__(ranks[0], ranks[1], 'PrNiO-LDA+U', variant)
        self.tags = {'qe-%s'%variant, 'parallel'}

@rfm.parameterized_test(*([variant, ranks] for variant in ['native', 'sirius'] for ranks in [(1,1), (2,1), (3,1)]))
class qe_MnO_ldapu_scf(qe_scf_base_test):
    def __init__(self, variant, ranks):
        super().__init__(ranks[0], ranks[1], 'MnO-LDA+U', variant)
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
