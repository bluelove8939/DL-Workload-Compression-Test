import os
import argparse
import numpy as np
from torch.utils.data import DataLoader

from simulation.cycle_sim import CompressedAcceleratorCycleSim
from simulation.testbenches import testbench_filter

from models.model_presets import generate_from_quant_chkpoint, imagenet_pretrained
from models.tools.imagenet_utils.dataset_loader import val_dataset, val_sampler


parser = argparse.ArgumentParser(description='Compressed Accelerator Performance Simulation')
parser.add_argument('-en', '--engine-num',      default=4,        help='Number of compressed accelerator engines',               dest='engine_num',      type=int)
parser.add_argument('-pn', '--pe-num',          default=8,        help='Number of processing element within an engine',          dest='pe_num',          type=int)
parser.add_argument('-mn', '--multiplier-num',  default=2,        help='Number of multipliers within a processing element',      dest='mult_num',        type=int)
parser.add_argument('-cs', '--chunk-size',      default=4,        help='Size of a chunk',                                        dest='chunk_size',      type=int)
parser.add_argument('-fc', '--fifo-capacity',   default=8,        help='Capacity of the fifo',                                   dest='fifo_capacity',   type=int)
parser.add_argument('-ss', '--sa-shape',        default=(8, 8),   help='Shape of the systolic array',                            dest='sa_shape',        type=int, nargs=2)
parser.add_argument('-wts', '--wgt-tile-shape', default=(32, 32), help='Shape of a weight tile',                                 dest='wgt_tile_shape',  type=int, nargs=2)
parser.add_argument('-ats', '--act-tile-shape', default=(32, 32), help='Shape of an input activation tile',                      dest='act_tile_shape',  type=int, nargs=2)
parser.add_argument('-sf', '--sampling-factor', default=500,      help='Number of tile multiplication samples (0 for infinity)', dest='sampling_factor', type=int)
sim_args = parser.parse_args()


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)

    # Compressed Accelerator Configs
    engine_num    = sim_args.engine_num
    pe_num        = sim_args.pe_num
    mult_num      = sim_args.mult_num
    chunk_size    = sim_args.chunk_size
    fifo_capacity = sim_args.fifo_capacity

    # Simulation Environment Settings
    sa_shape        = tuple(sim_args.sa_shape)
    wgt_tile_shape  = tuple(sim_args.wgt_tile_shape)
    act_tile_shape  = tuple(sim_args.act_tile_shape)
    sampling_factor = sim_args.sampling_factor

    # Log file and model path
    log_dirname = os.path.join(os.curdir, 'logs')
    log_filename = f"{os.path.split(__file__)[1].split('.')[0]}_en{engine_num}_pn{pe_num}_mc{mult_num}_cs{chunk_size}_fc{fifo_capacity}_sf{sampling_factor}.csv"
    filepath_fmt = os.path.join(os.curdir, 'model_output', "{name}_quantized_tuned_citer_10_pruned_pamt_0.5.pth")

    # Start Simulation
    sim = CompressedAcceleratorCycleSim(engine_num=2, pe_num=pe_num, mult_num=mult_num, chunk_size=chunk_size,
                                        fifo_capacity=fifo_capacity, sa_shape=sa_shape,
                                        wgt_tile_shape=wgt_tile_shape, act_tile_shape=act_tile_shape,
                                        sampling_factor=sampling_factor, quant=True)

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
        content = []

        sim_info = {
            "number of engines": engine_num,
            "number of PEs": pe_num,
            "number of multipliers per PE": mult_num,
            "size of a chunk": chunk_size,
            "capacity of FIFOs": fifo_capacity,
            "shape of systolic array": sa_shape,
            "shape of a weight tile": wgt_tile_shape,
            "shape of a activation tile": act_tile_shape,
            "sampling factor": sampling_factor,
        }

        for key, val in sim_info.items():
            content.append(f"# {key}: {val}")
        content.append('')

        content.append('model name,layer name,compressed accelerator,systolic array,performance gain')
        for key, (ca_cycle, sa_cycle) in sim.result.items():
            content.append(f"{','.join(key)},{ca_cycle},{sa_cycle},{sa_cycle / ca_cycle:.6f}")
        log_file.write('\n'.join(content))