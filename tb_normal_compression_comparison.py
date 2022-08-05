import os
import subprocess
import platform
import argparse


parser = argparse.ArgumentParser(description='Comparison Test Configs')
parser.add_argument('-cs', '--csize', default=128, help='Cache line size (int)', dest='csize')
parser.add_argument('-mi', '--maxiter', default=10000, help='Max iteration of the file fetch (int)', dest='maxiter')
parser.add_argument('-dir', '--directory', default=os.path.join(os.curdir, '../extractions_quant'), help='Directory of model extraction files', dest='extdir')
comp_args, _ = parser.parse_known_args()

tb_codename = 'tb_file'
tb_name = f'{tb_codename}.exe'
if 'linux' in platform.platform().lower():
    tb_name = f'./{tb_codename}'

print(f"gcc -o {tb_codename} ./{tb_codename}.c ./compression.c -lm -Wformat=0")
subprocess.run(f"gcc -o {tb_codename} ./{tb_codename}.c ./compression.c -lm -Wformat=0", shell=True, check=True)


for model_name in os.listdir(comp_args.extdir):
    filelist_path = os.path.join(comp_args.extdir, model_name, 'filelist.txt')
    result_path = os.path.join(comp_args.extdir, model_name, 'comparison_results.csv')
    print(f"\n{tb_name} {filelist_path} {comp_args.csize} {comp_args.maxiter} {result_path}")
    subprocess.run(f"{tb_name} {filelist_path} {comp_args.csize} {comp_args.maxiter} {result_path}")