set -e
pip install -e /tinyrl\[mkl\]
python3 /tinyrl/examples/pendulum_sac.py