# Regression tests for QE+SIRIUS code

The following command is used:
```bash
~/reframe/bin/reframe -C ~/reframe/config/cscs.py --system=daint:gpu -c ./checks -p PrgEnv-intel -R -r --tag serial --exec-policy=async --prefix $SCRATCH/reframe
```

For local runs:
```bash
~/src/reframe/bin/reframe -C ~/src/github/electronic-structure/SIRIUS/reframe/config.py --system=osx -c ./checks -p PrgEnv-gnu -R -r --tag serial --prefix /tmp/reframe -v
```
