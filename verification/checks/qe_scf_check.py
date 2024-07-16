import re

import reframe as rfm
import reframe.utility.sanity as sn

@sn.deferrable
def get_energy(ostream):
    ''' Get energy from the output stream'''

    # take last energy value (is case of vc-relax)
    return sn.extractsingle(r'!\s+total energy\s+=\s+(?P<energy>\S+) Ry',
                            ostream, 'energy', float, item=-1)

@sn.deferrable
def get_pressure(ostream):
    ''' Get pressure from the output stream'''

    # take last pressure value (is case of vc-relax)
    return sn.extractsingle(r'\s+total\s+stress.+\(kbar\)\s+P\=\s*(?P<pressure>\S+)',
                            ostream, 'pressure', float, item=-1)

@sn.deferrable
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

@sn.deferrable
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

@sn.deferrable
def energy_diff(ostream, ostream_ref):
    ''' Return the difference between obtained and reference total energies'''
    return sn.abs(get_energy(ostream) - get_energy(ostream_ref))

@sn.deferrable
def pressure_diff(ostream, ostream_ref):
    ''' Return the difference between obtained and reference total energies'''
    return sn.abs(get_pressure(ostream) - get_pressure(ostream_ref))

@sn.deferrable
def stress_diff(ostream, ostream_ref):
    ''' Return the difference between obtained and reference stress tensor components'''

    stress = get_stress(ostream)
    stress_ref = get_stress(ostream_ref)
    return sn.max(sn.abs(stress_ref[i][j] - stress[i][j]) for i in range(2) for j in range(2))

@sn.deferrable
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

class qe_scf_test_base(rfm.RunOnlyRegressionTest):
    valid_systems = ['todi', 'localhost']
    valid_prog_environs = ['builtin']
    executable = '$SCRATCH/mps-wrapper.sh pw.x'
    energy_tol = 1e-6
    pressure_tol = 3e-1
    stress_tol = 1e-4
    forces_tol = 1e-4

    @run_after('init')
    def set_description(self):
        self.descr = f'QuantumESPRESSO {self.test_folder} check'
        self.num_ranks_k = self.ranks[0]
        self.num_ranks_d = self.ranks[1]

        self.executable_opts = ['-in', 'pw.in', '-npool', f"{self.num_ranks_k}", '-ndiag', f"{self.num_ranks_d}"]
        if self.variant == 'native':
            self.executable_opts.append('-use_qe_scf')

        self.sourcesdir = f"../{self.test_folder}"
        self.tags.add(f"qe-{self.variant}")
        if self.ranks[0] == 1 and self.ranks[1] == 1:
            self.tags.add('serial')
        else:
            self.tags.add('parallel')



    @run_before('run')
    def set_parallel_resources(self):
        #self.skip_if_no_procinfo()
        #processor_info = self.current_partition.processor
        #self.node_label = processor_info.arch
        #if self.current_partition.devices:
        #    # device label, for example 4x-gpu-sm_80
        #    dev_label = ''
        #    for e in self.current_partition.devices:
        #        dev_label = f"-{dev_label}{e.num_devices}x-{e.type}-{e.arch}"

        #    self.node_label = self.node_label + dev_label

        ## number of physical cores
        #num_cores = int(
        #    processor_info.num_cpus / processor_info.num_cpus_per_core)
        num_cores = 72 * 4

        # total number of ranks
        self.num_tasks = self.num_ranks_k * self.num_ranks_d
        # threads / rank
        self.num_cpus_per_task = 8
        # ranks per node
        self.num_tasks_per_node = int(num_cores / self.num_cpus_per_task)
        if not self.env_vars:
            self.env_vars = {}
        self.env_vars['OMP_NUM_THREADS'] = '$SLURM_CPUS_PER_TASK'
        #self.env_vars['OMP_PLACES'] = 'cores'
        #self.env_vars['OMP_PROC_BIND'] = 'close'

    @sanity_function
    def assert_simulation_success(self):
        patterns = [
            sn.assert_found(r'convergence has been achieved', self.stdout),
            sn.assert_lt(energy_diff(self.stdout, 'out.txt'), self.energy_tol,
                msg="Total energy is different"),
            sn.assert_lt(pressure_diff(self.stdout, 'out.txt'), self.pressure_tol, msg="Pressure is different"),
            sn.assert_lt(stress_diff(self.stdout, 'out.txt'), self.stress_tol, msg="Stress tensor is different"),
            sn.assert_lt(forces_diff(self.stdout, 'out.txt'), self.forces_tol, msg="Atomic forces are different")
        ]
        if self.variant == 'sirius':
            patterns.append(sn.assert_found(r'SIRIUS.+git\shash', self.stdout))
        return sn.all(patterns)

    @run_after('setup')
    def set_launcher_options(self):
        if self.current_system.name in ['daint']:
            self.job.launcher.options = ["-c %i"%self.num_cpus_per_task, "-n %i"%self.num_tasks, '--hint=nomultithread']

#---------------#
# List of tests #
#---------------#
@rfm.simple_test
class qe_Si_scf(qe_scf_test_base):
    variant = parameter(['native', 'sirius'])
    tags = {'nc', 'lda'}
    ranks = parameter([(1, 1), (2, 1)])
    test_folder = 'Si'

@rfm.simple_test
class qe_Si_vc_relax(qe_scf_test_base):
    variant = parameter(['native', 'sirius'])
    tags = {'nc', 'lda'}
    ranks = parameter([(1, 1), (2, 1), (3, 1)])
    test_folder = 'Si-vc-relax'

@rfm.simple_test
class qe_LiF_gga_nc_scf(qe_scf_test_base):
    variant = parameter(['native', 'sirius'])
    tags = {'nc', 'gga'}
    ranks = parameter([(1, 1), (2, 1)])
    test_folder = 'LiF-gga-nc'

@rfm.simple_test
class qe_LiF_lda_nc_scf(qe_scf_test_base):
    variant = parameter(['native', 'sirius'])
    tags = {'nc', 'lda'}
    ranks = parameter([(1, 1), (2, 1)])
    test_folder = 'LiF-lda-nc'

@rfm.simple_test
class qe_LiF_lda_uspp_scf(qe_scf_test_base):
    variant = parameter(['native', 'sirius'])
    tags = {'uspp', 'lda'}
    ranks = parameter([(1, 1), (2, 1)])
    test_folder = 'LiF-lda-uspp'

@rfm.simple_test
class qe_LiF_lda_paw_scf(qe_scf_test_base):
    variant = parameter(['native', 'sirius'])
    tags = {'paw', 'lda'}
    ranks = parameter([(1, 1), (2, 1)])
    test_folder = 'LiF-lda-paw'

@rfm.simple_test
class qe_Fe_lda_uspp_scf(qe_scf_test_base):
    variant = parameter(['native', 'sirius'])
    tags = {'magn', 'lda', 'uspp'}
    ranks = parameter([(1, 1), (2, 1)])
    test_folder = 'Fe-lda-uspp'

@rfm.simple_test
class qe_NiO_afm_scf(qe_scf_test_base):
    variant = parameter(['native', 'sirius'])
    tags = {'magn', 'gga', 'uspp'}
    ranks = parameter([(1, 1), (2, 1)])
    test_folder = 'NiO-afm'

@rfm.simple_test
class qe_NiO_ldapu_scf(qe_scf_test_base):
    variant = parameter(['native', 'sirius'])
    tags = {'magn', 'gga', 'uspp', 'ldapu'}
    ranks = parameter([(1, 1), (2, 1)])
    test_folder = 'NiO-ldapu'

@rfm.simple_test
class qe_NiO_ldapuv_scf(qe_scf_test_base):
    variant = 'native'
    tags = {'magn', 'gga', 'uspp', 'ldapu'}
    ranks = parameter([(1, 1), (2, 1)])
    test_folder = 'NiO-ldapuv'

@rfm.simple_test
class qe_NiO_ldapuv_sirius_scf(qe_scf_test_base):
    variant = 'sirius'
    tags = {'magn', 'gga', 'uspp', 'ldapu'}
    ranks = parameter([(1, 1), (2, 1)])
    test_folder = 'NiO-ldapuv'
    # QE enforces occupation matrix to be real, this causes some larger energy difference
    energy_tol = 2e-5

#@rfm.simple_test
#class qe_NiO_ldapuv_ortho_scf(qe_scf_base_test):
#    variant = parameter(['native', 'sirius'])
#    def __init__(self):
#        # QE enforces occupation matrix to be real, this causes some larger energy difference
#        etol = 1e-6 if self.variant == "native" else 2e-5
#        super().__init__(1, 1, 'NiO-ldapuv-ortho', self.variant, energy_tol=etol)
#        self.tags = {f'qe-{self.variant}', 'serial', 'magn', 'gga', 'uspp', 'ldapu'}
#
##@rfm.simple_test
##class qe_LiF_esm_scf(qe_scf_base_test):
##    variant = parameter(['native', 'sirius'])
##    def __init__(self):
##        super().__init__(1, 1, 'LiF-esm', self.variant)
##        self.tags = {f'qe-{self.variant}', 'serial', 'esm'}
#
#@rfm.simple_test
#class qe_CdCO3_gga_paw_scf(qe_scf_base_test):
#    variant = parameter(['native', 'sirius'])
#    ranks = parameter([(1,1), (2,1), (4,1)])
#    def __init__(self):
#        #
#        # This is GGA PAW test. Right now PAW XC part is done by SIRIUS and for GGA libxc gives a different
#        # result comparing with QE implementation. Also, for PAW radial integrals are computed differently.
#        #
#        etol = 1e-6 if self.variant == "native" else 5e-3
#        super().__init__(self.ranks[0], self.ranks[1], 'CdCO3-gga-paw', self.variant, energy_tol=etol)
#        self.tags = {f'qe-{self.variant}', 'parallel', 'paw', 'gga'}
#
#@rfm.simple_test
#class qe_CdCO3_lda_paw_scf(qe_scf_base_test):
#    variant = parameter(['native', 'sirius'])
#    ranks = parameter([(1,1), (2,1), (4,1)])
#    def __init__(self):
#        #
#        # For PAW the radial integrals are computed differently in QE and SIRIUS and also the spherical coverage
#        # to compute XC potential is different, so we can't expect the full numerical reproducibility here
#        #
#        etol = 1e-6 if self.variant == "native" else 1e-4
#        ftol = 1e-4 if self.variant == "native" else 1e-3
#        super().__init__(self.ranks[0], self.ranks[1], 'CdCO3-lda-paw', self.variant, energy_tol=etol, forces_tol=ftol)
#        self.tags = {f'qe-{self.variant}', 'parallel', 'paw', 'lda'}
#
#@rfm.simple_test
#class qe_CdCO3_lda_uspp_scf(qe_scf_base_test):
#    variant = parameter(['native', 'sirius'])
#    ranks = parameter([(1,1), (2,1), (4,1)])
#    def __init__(self):
#        super().__init__(self.ranks[0], self.ranks[1], 'CdCO3-lda-uspp', self.variant)
#        self.tags = {f'qe-{self.variant}', 'parallel', 'lda', 'uspp'}
#
#@rfm.simple_test
#class qe_Si63Ge_scf(qe_scf_base_test):
#    variant = parameter(['native', 'sirius'])
#    ranks = parameter([(4,1), (1,4)])
#    def __init__(self):
#        #etol = 1e-6 if variant == "native" else 1e-3
#        #super().__init__(ranks[0], ranks[1], 'Si63Ge', variant, energy_tol=etol)
#        super().__init__(self.ranks[0], self.ranks[1], 'Si63Ge', self.variant)
#        self.tags = {f'qe-{self.variant}', 'parallel', 'uspp', 'gga'}
#
##= @rfm.parameterized_test(*([variant, ranks] for variant in ['native', 'sirius'] for ranks in [(2,4), (2,8)]))
##= class qe_Au_surf_scf(qe_scf_base_test):
##=     def __init__(self, variant, ranks):
##=         # GGA of libxc is slighly different
##=         etol = 1e-6 if variant == "native" else 1e-3
##=         super().__init__(ranks[0], ranks[1], 'Au-surf', variant, energy_tol=etol)
##=         self.tags = {'qe-%s'%variant, 'parallel', 'Au-surf', 'gga'}
##=         self.time_limit = '20m'
##= 
##= @rfm.parameterized_test(*([variant, ranks] for variant in ['native', 'sirius'] for ranks in [(4,1), (8,1)]))
##= class qe_HfNi5_scf(qe_scf_base_test):
##=     def __init__(self, variant, ranks):
##=         # GGA of libxc is slighly different
##=         etol = 1e-6 if variant == "native" else 1e-3
##=         super().__init__(ranks[0], ranks[1], 'HfNi5', variant, energy_tol=etol)
##=         self.tags = {'qe-%s'%variant, 'parallel', 'hfni5', 'gga'}
##= 
##= @rfm.parameterized_test(*([variant, ranks] for variant in ['native', 'sirius'] for ranks in [(3,1), (5,1)]))
##= class qe_NiO_afm_scf(qe_scf_base_test):
##=     def __init__(self, variant, ranks):
##=         etol = 1e-6 if variant == "native" else 1e-3
##=         super().__init__(ranks[0], ranks[1], 'NiO-afm', variant, energy_tol=etol)
##=         self.tags = {'qe-%s'%variant, 'parallel', 'magn', 'gga', 'paw'}
##= 
##= @rfm.parameterized_test(*([variant, ranks] for variant in ['native', 'sirius'] for ranks in [(1,1), (2,1)]))
##= class qe_NiO_lda_uspp_scf(qe_scf_base_test):
##=     def __init__(self, variant, ranks):
##=         super().__init__(ranks[0], ranks[1], 'NiO-lda-uspp', variant)
##=         self.tags = {'qe-%s'%variant, 'magn', 'uspp'}
##= 
##= @rfm.parameterized_test(*([variant, ranks] for variant in ['native', 'sirius'] for ranks in [(4,1), (8,1)]))
##= class qe_FeSe2_2D_scf(qe_scf_base_test):
##=     def __init__(self, variant, ranks):
##=         super().__init__(ranks[0], ranks[1], 'FeSe2-2D', variant)
##=         self.tags = {'qe-%s'%variant, 'parallel'}
##= 
##= @rfm.parameterized_test(*([variant, ranks] for variant in ['native', 'sirius'] for ranks in [(4,1), (8,1)]))
##= class qe_FeSe2_2D_lda_uspp_scf(qe_scf_base_test):
##=     def __init__(self, variant, ranks):
##=         super().__init__(ranks[0], ranks[1], 'FeSe2-2D-lda-uspp', variant)
##=         self.tags = {'qe-%s'%variant, 'parallel', 'uspp', 'magn'}
##= 
##= @rfm.simple_test
##= class qe_FeSe2_2D_lda_nc_scf(qe_scf_base_test):
##=     variant = parameter(['native', 'sirius'])
##=     ranks = parameter([(4,1), (8,1)])
##=     def __init__(self):
##=         super().__init__(self.ranks[0], self.ranks[1], 'FeSe2-2D-lda-nc', self.variant)
##=         self.tags = {f'qe-{self.variant}', 'parallel', 'nc', 'magn'}
##= 
##= @rfm.parameterized_test(*([variant, ranks] for variant in ['native', 'sirius'] for ranks in [(1,1), (2,1), (3,1)]))
##= class qe_Ni_ldapu_scf(qe_scf_base_test):
##=     def __init__(self, variant, ranks):
##=         super().__init__(ranks[0], ranks[1], 'Ni-ldapu', variant)
##=         self.tags = {'qe-%s'%variant, 'parallel'}
##= 
##= @rfm.parameterized_test(*([variant, ranks] for variant in ['native', 'sirius'] for ranks in [(1,1), (2,1), (3,1)]))
##= class qe_PrNiO_ldapu_scf(qe_scf_base_test):
##=     def __init__(self, variant, ranks):
##=         super().__init__(ranks[0], ranks[1], 'PrNiO-LDA+U', variant)
##=         self.tags = {'qe-%s'%variant, 'parallel'}
##= 
##= @rfm.parameterized_test(*([variant, ranks] for variant in ['native', 'sirius'] for ranks in [(1,1), (2,1), (3,1)]))
##= class qe_MnO_ldapu_scf(qe_scf_base_test):
##=     def __init__(self, variant, ranks):
##=         super().__init__(ranks[0], ranks[1], 'MnO-LDA+U', variant)
##=         self.tags = {'qe-%s'%variant, 'parallel'}
##= 
#
#
#
#
#
#
#
#
##@rfm.simple_test
##class qe_LiF_nc_vc_relax(qe_scf_base_test):
##    def __init__(self):
##        super().__init__(1, 'LiF-nc', 'pw.in', False, -49.19540885)
##        self.tags = {'serial'}
##
##@rfm.simple_test
##class qe_sirius_LiF_nc_vc_relax(qe_scf_base_test):
##    def __init__(self):
##        super().__init__(1, 'LiF-nc', 'pw.in', True, -49.19541319)
##        self.tags = {'serial'}
##
##@rfm.simple_test
##class qe_LiF_paw_vc_relax(qe_scf_base_test):
##    def __init__(self):
##        super().__init__(1, 'LiF-paw', 'pw.in', False, -73.36174370)
##        self.tags = {'serial'}
##
##@rfm.simple_test
##class qe_sirius_LiF_paw_vc_relax(qe_scf_base_test):
##    def __init__(self):
##        super().__init__(1, 'LiF-paw', 'pw.in', True, -73.36174696)
##        self.tags = {'serial'}
##
##@rfm.simple_test
##class qe_LiF_uspp_vc_relax(qe_scf_base_test):
##    def __init__(self):
##        super().__init__(1, 'LiF-uspp', 'pw.in', False, -63.36417257)
##        self.tags = {'serial'}
##
##@rfm.simple_test
##class qe_sirius_LiF_uspp_vc_relax(qe_scf_base_test):
##    def __init__(self):
##        super().__init__(1, 'LiF-uspp', 'pw.in', True, -63.36416794)
##        self.tags = {'serial'}
#
