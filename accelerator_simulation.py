import os
import numpy as np
from torch.utils.data import DataLoader

from simulation.cycle_sim import CompressedAcceleratorCycleSim
from simulation.testbenches import testbench_filter

from models.model_presets import generate_from_quant_chkpoint, imagenet_pretrained
from models.tools.imagenet_utils.dataset_loader import val_dataset, val_sampler


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)

    log_dirname = os.path.join(os.curdir, 'logs')
    log_filename = f"{os.path.split(__file__)[1].split('.')[0]}.csv"
    filepath_fmt = os.path.join(os.curdir, 'model_output', "{name}_quantized_tuned_citer_10_pruned_pamt_0.5.pth")

    sim = CompressedAcceleratorCycleSim(mult_num=2, pe_num=16, chunk_size=4, fifo_capacity=8, sa_shape=(8, 8),
                                        tile_shape=(32, 32), sampling_factor=10, quant=True)

    for name, config in imagenet_pretrained.items():
        if not os.path.isfile(filepath_fmt.format(name=name)):
            print(f"file '{filepath_fmt.format(name=name)}.pth' not found")
            continue

        model = generate_from_quant_chkpoint(
            model_primitive=config.generate(),
            chkpoint_path=filepath_fmt.format(name=name))

        sim.register_model(model=model, model_name=name, testbench_filter=testbench_filter)

        # Inference one batch
        val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, sampler=val_sampler)
        images, _ = next(iter(val_loader))

        model = model.to('cpu')
        images = images.to('cpu')

        model.eval()
        model(images)

    for (model_name, layer_name), (ca_cycle, sa_cycle) in sim.result.items():
        print(
            f"cycle simulation: {model_name:15s} {layer_name:30s}  "
            f"compressed accelerator: {ca_cycle}  "
            f"systolic array: {sa_cycle}  "
            f"performance gain: {sa_cycle / ca_cycle:.6f}"
        )

    with open(os.path.join(log_dirname, log_filename), 'wt') as log_file:
        content = ['model name,layer name,compressed accelerator,systolic array,performance gain']
        for key, (ca_cycle, sa_cycle) in sim.result.items():
            content.append(f"{','.join(key)},{ca_cycle},{sa_cycle},{sa_cycle / ca_cycle:.6f}")
        log_file.write('\n'.join(content))