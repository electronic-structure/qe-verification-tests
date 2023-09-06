# Regression tests for QE code

## How to run tests
We use ReFrame to describe and run regression tests. To run tests on Daint use the following command:
```bash
cd ./verification
~/reframe/bin/reframe -C ~/reframe/config/cscs.py --system=daint:gpu -c ./checks -p PrgEnv-gnu -R -r --exec-policy=async --prefix $SCRATCH/reframe
```
For the local runs use this command:
```bash
~/src/reframe/bin/reframe -C ~/src/github/electronic-structure/SIRIUS/reframe/config.py --system=osx -c ./checks -p PrgEnv-gnu -R -r --tag serial --prefix /tmp/reframe -v
```
Please make sure that `pw.x` binary is located in `$PATH`.


## How to add new tests
 * create a new test folder in ./verification
 * use `pw.in` for input file
 * run native QE and save output to `out.txt`
 * add new test description in `checks/qe_scf_check.py`


