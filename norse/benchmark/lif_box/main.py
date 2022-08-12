from argparse import ArgumentParser
import torch


def init_parser(parser: ArgumentParser):
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of data points per batch",
    )
    parser.add_argument(
        "--start", type=int, default=250, help="Start of the number of inputs to sweep"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=250,
        help="Steps in which to sweep over the number of inputs",
    )
    parser.add_argument(
        "--stop", type=int, default=10001, help="Number of inputs to sweep to"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=1000,
        help="Number of timesteps to simulate",
    )
    parser.add_argument("--dt", type=float, default=0.001, help="Simulation timestep")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use [cpu, cuda]",
    )
    parser.add_argument(
        "--runs", type=int, default=5, help="Number of runs per simulation step"
    )
    parser.add_argument(
        "--profile",
        default=False,
        action="store_true",
        help="Profile Norse benchmark? (Only works for Norse)",
    )
    parser.add_argument(
        "--jelly",
        default=False,
        action="store_true",
        help="Benchmark Spiking Jelly?",
    )
    parser.add_argument(
        "--norse",
        default=False,
        action="store_true",
        help="Benchmark Norse?",
    )
    parser.add_argument(
        "--bindsnet",
        default=False,
        action="store_true",
        help="Benchmark BindsNet?",
    )
    parser.set_defaults(func=execute_args)


def execute_args(args, continuation):
    # pytype: disable=import-error
    if args.jelly:
        import norse.benchmark.lif_box.jelly_lif_box as jelly_lif_box

        continuation(
            args,
            jelly_lif_box.lif_box_feed_forward_benchmark,
            label="SpikingJelly_lif_box",
        )
    if args.bindsnet:
        import norse.benchmark.lif_box.bindsnet_lif_box as bindsnet_lif_box

        continuation(
            args,
            bindsnet_lif_box.lif_box_feed_forward_benchmark,
            label="BindsNet_lif_box",
        )
    if args.norse:
        import norse.benchmark.lif_box.norse_lif_box as norse_lif_box

        if args.profile:
            import torch.autograd.profiler as profiler

            with profiler.profile(
                profile_memory=True, use_cuda=(args.device == "cuda")
            ) as prof:
                continuation(
                    args,
                    norse_lif_box.lif_box_feed_forward_benchmark,
                    label="Norse_lif_box",
                )
            prof.export_chrome_trace("trace.json")
        else:
            continuation(
                args,
                norse_lif_box.lif_box_feed_forward_benchmark,
                label="Norse_lif_box",
            )
    # pytype: enable=import-error
