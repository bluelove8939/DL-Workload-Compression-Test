import os

dirname = os.path.join(os.curdir, '../extractions_quant_wfile')
for modelname in os.listdir(dirname):
    filelist = []
    for filename in os.listdir(os.path.join(dirname, modelname)):
        if 'comparison' in filename or 'filelist' in filename:
            continue
        filelist.append(os.path.join(dirname, modelname, filename))

    with open(os.path.join(dirname, modelname, 'filelist.txt'), 'wt') as file:
        file.write('\n'.join(filelist))