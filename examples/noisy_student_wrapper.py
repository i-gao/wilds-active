"""
Helper code to run multiple iterations of Noisy Student, using the same hyperparameters between iterations.

Normally, to run 2 iterations, one would run a sequence of commands:
    python examples/run_expt.py --root_dir $HOME --log_dir ./teacher --dataset DATASET --algorithm NoisyStudent --target_split test
    python examples/run_expt.py --root_dir $HOME --log_dir ./student1 --dataset DATASET --algorithm NoisyStudent --target_split test --teacher_model_path ./teacher/model.pth
    python examples/run_expt.py --root_dir $HOME --log_dir ./student2 --dataset DATASET --algorithm NoisyStudent --target_split test --teacher_model_path ./student1/model.pth

With this script, to run 2 iterations:
    python examples/noisy_student_wrapper.py 2 --root_dir $HOME --log_dir . --dataset DATASET --target_split test

i.e. usage:
    python examples/noisy_student_wrapper.py [NUM_ITERS] [REST OF RUN_EXPT COMMAND STRING]

Notes:
    - This command will use the FIRST occurrence of --log_dir (instead of the last).
"""
import subprocess
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("num_iters", type=int)
parser.add_argument("cmd", nargs=argparse.REMAINDER)
args = parser.parse_args()

import pathlib
prefix = pathlib.Path(__file__).parent.resolve()

# Parse out a few args that we need
try:
    idx = args.cmd.index('--log_dir')
    log_dir = args.cmd[idx + 1]
    args.cmd = args.cmd[:idx] + args.cmd[idx+2:] # will need to modify this between iters, so remove from args.cmd
except:
    log_dir = './logs' # default in run_expt.py

idx = args.cmd.index('--dataset')
dataset = args.cmd[idx + 1]

try:
    idx = args.cmd.index('--seed')
    seed = args.cmd[idx + 1]
except:
    seed = 0 # default in run_expt.py

idx = args.cmd.index('--target_split')
target_split = args.cmd[idx + 1]
args.cmd = args.cmd[:idx] + args.cmd[idx+2:] # will need to modify this between iters, so remove from args.cmd

# Run teacher
cmd = f"python {prefix}/run_expt.py --algorithm NoisyStudent {' '.join(args.cmd)} --log_dir {log_dir}/teacher"
print(f">>> Running {cmd}")
subprocess.Popen(cmd, shell=True).wait()

# Run student iters
for i in range(1, args.num_iters + 1):
    cmd = (
        f"python {prefix}/run_expt.py --algorithm NoisyStudent {' '.join(args.cmd)} --target_split {target_split} --log_dir {log_dir}/student{i}" 
        + f" --teacher_model_path {log_dir}/" + ('teacher' if i == 1 else f'student{i-1}') + f"/{dataset}_seed:{seed}_epoch:best_model.pth"
    )
    print(f">>> Running {cmd}")
    subprocess.Popen(cmd, shell=True).wait()

print(">>> Done!")