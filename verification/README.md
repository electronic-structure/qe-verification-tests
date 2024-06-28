Example of ReFrame run:

```
python -m venv ~/rfm-venv
source ~/rfm-venv/bin/activate
pip install reframe-hpc
OMP_NUM_THREADS=4 PATH=/path/to/qe/bin:$PATH reframe -c ./checks/ -r  --system=localhost -C ./checks/config.py  --tag serial --tag ldapu
```
