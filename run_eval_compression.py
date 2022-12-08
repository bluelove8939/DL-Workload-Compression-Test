import os
import subprocess


script_paths = [
    os.path.join(os.curdir, 'eval_activation_compression.py'),
    os.path.join(os.curdir, 'eval_activation_compression_pruned.py'),
    os.path.join(os.curdir, 'eval_activation_compression_quant.py'),
    os.path.join(os.curdir, 'eval_activation_compression_quant_pruned.py'),

    os.path.join(os.curdir, 'eval_weight_compression.py'),
    os.path.join(os.curdir, 'eval_weight_compression_pruned.py'),
    os.path.join(os.curdir, 'eval_weight_compression_quant.py'),
    os.path.join(os.curdir, 'eval_weight_compression_quant_pruned.py'),
]

log_dirname = os.path.join(os.curdir, 'logs')


if __name__ == '__main__':
    for sp in script_paths:
        log_filename = os.path.split(sp)[1].split('.')[0] + '.log'
        res_filename = os.path.split(sp)[1].split('.')[0] + '.csv'

        with open(os.path.join(log_dirname, log_filename), 'wt') as logfile:
            print(f"Running script: {sp}")
            ec = subprocess.run(['python', sp], stdout=logfile, stderr=logfile)

        if ec.returncode != 0:
            print(f"- error occurred on running the script ({ec})")
            print(f"- you can see the details of the error at the logfile: {os.path.join(log_dirname, log_filename)}")
        else:
            print(f"- succeed on running the script")
            print(f"- results: {os.path.join(log_dirname, res_filename)}")
            os.remove(os.path.join(log_dirname, log_filename))

