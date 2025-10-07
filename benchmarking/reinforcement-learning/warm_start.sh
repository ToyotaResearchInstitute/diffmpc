
python benchmark_diffmpc.py --batch_size 64 --pcg_eps 1e-12 --num_repeats 100
python benchmark_diffmpc.py --batch_size 64 --pcg_eps 1e-8 --num_repeats 100
python benchmark_diffmpc.py --batch_size 64 --pcg_eps 1e-4 --num_repeats 100
python benchmark_diffmpc.py --batch_size 64 --pcg_eps 1e-12 --cold_start --num_repeats 100
python benchmark_diffmpc.py --batch_size 64 --pcg_eps 1e-8 --cold_start --num_repeats 100
python benchmark_diffmpc.py --batch_size 64 --pcg_eps 1e-4 --cold_start --num_repeats 100
