set -e
export TINYRL_FULL_RUN=
for i in {15..99}; do echo $i; done | parallel -j 4 --halt now,fail=1 python3 train.py --config {} --output-dir results