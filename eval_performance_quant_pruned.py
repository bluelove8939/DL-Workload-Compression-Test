import os
import torch
from torch.utils.data import DataLoader

from models.model_presets import imagenet_pretrained, generate_from_quant_chkpoint
from models.tools.imagenet_utils.dataset_loader import val_dataset, val_sampler
from simulation.accelerator_sim import PerformanceSim
from simulation.testbenches import testbench_filter


if __name__ == '__main__':
    log_dirname = os.path.join(os.curdir, 'logs')
    log_filename = f"{os.path.split(__file__)[1].split('.')[0]}.csv"
    filepath_fmt = os.path.join(os.curdir, 'model_output', "{name}_quantized_tuned_citer_10_pruned_pamt_0.5.pth")

    perf_tb = PerformanceSim(quant=True, device='cpu')

    for name, config in imagenet_pretrained.items():
        if not os.path.isfile(filepath_fmt.format(name=name)):
            continue

        # Generate model
        model = generate_from_quant_chkpoint(
            model_primitive=config.generate(),
            chkpoint_path=filepath_fmt.format(name=name),)
        perf_tb.register_model(model=model, model_name=name, testbench_filter=testbench_filter)

        # Inference one batch
        val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, sampler=val_sampler)
        images, _ = next(iter(val_loader))

        model = model.to('cpu')
        images = images.to('cpu')

        model.eval()
        model(images)

    print(perf_tb.result)

    with open(os.path.join(log_dirname, log_filename), 'wt') as log_file:
        content = ['model name,layer name,total,valid,ratio']
        for key, val in perf_tb.result.items():
            content.append(f"{','.join(key)},{','.join(list(map(lambda x:f'{x:.4f}', val)))}")
        log_file.write('\n'.join(content))