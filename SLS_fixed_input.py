from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import itertools
import numpy as np
import random
from caffe2.python import core, workspace, dyndep
from caffe2.proto import caffe2_pb2

init_net = core.Net("init")
net = core.Net("bench")


def benchSparseSegmentSum():
    size = args.table_size
    isize_input = args.batch_size * args.pooling
    for isize, bs, engine, dtype, itype in itertools.product(
        [isize_input],
        [args.column],
        [None],
        [core.DataType.FLOAT],
        [core.DataType.INT64],
    ):  
        for device in [workspace.GpuDeviceType]:
            if device != caffe2_pb2.CPU:
                if not workspace.has_gpu_support:
                    continue
                if engine is not None:
                    continue  # no fp16 support yet
                device_name = "gpu"
            else:
                device_name = "cpu"

            with core.DeviceScope(core.DeviceOption(device)):
                #d = init_net.UniformFill([], shape=[size, bs])
                #if dtype == core.DataType.FLOAT16:
                    #d = init_net.FloatToHalf([d])
                name = \
                    "SparseLengthsSum_{}_bs_{}_p_{}_eng_{}_dt_{}_ind_{}".format(
                        device_name, bs, isize, engine, dtype, itype)
                d = workspace.FeedBlob("weights", np.load("weights.npy"))
                #i = init_net.UniformIntFill([], shape=[isize], max=size - 1)
                #i = init_net.Cast([i], to=itype)
                i = workspace.FeedBlob("ind", np.load('ind.npy'))
                l = init_net.ConstantFill(
                    [],
                    shape=[isize // args.pooling],
                    value=args.pooling,
                    dtype=core.DataType.INT32,
                )
                #net.SparseLengthsSum([d, i, l], name=name, engine=engine)
                net.SparseLengthsSum(["weights", "ind", l], name=name, engine=engine)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--table-size', type=int, default=10**7,
                        help='embedding table size')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--pooling', type=int, default=20,
                        help='pooling')
    parser.add_argument('--column', type=int, default=64,
                        help='number of columns in the embedding table')
    args, extra_args = parser.parse_known_args()

    benchSparseSegmentSum()

    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'] + extra_args)
    workspace.RunNetOnce(init_net)
    workspace.CreateNet(net)
    workspace.BenchmarkNet(net.Proto().name, 100, 10000, True)
