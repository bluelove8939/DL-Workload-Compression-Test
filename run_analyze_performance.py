import os
import subprocess


options = [
    ['python', os.path.join(os.curdir, 'analyze_eval_performance.py'), '--filename', 'eval_performance'],
    ['python', os.path.join(os.curdir, 'analyze_eval_performance.py'), '--filename', 'eval_performance_pruned'],
    ['python', os.path.join(os.curdir, 'analyze_eval_performance.py'), '--filename', 'eval_performance_quant'],
    ['python', os.path.join(os.curdir, 'analyze_eval_performance.py'), '--filename', 'eval_performance_quant_pruned'],
]

log_dirname = os.path.join(os.curdir, 'logs')


if __name__ == '__main__':
    for op in options:
        print(f"Running script: {op}")
        ec = subprocess.run(op)

        if ec.returncode != 0:
            print(f"- error occurred on running the script ({ec})")
        else:
            print(f"- succeed on running the script")


