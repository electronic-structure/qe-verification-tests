# Regression tests for QE+SIRIUS code

The following command is used:
```
~/reframe/bin/reframe -C ~/reframe/config/cscs.py --system=daint:gpu -c ./checks -p PrgEnv-intel -R -r --tag serial --exec-policy=async --prefix $SCRATCH/reframe
```
