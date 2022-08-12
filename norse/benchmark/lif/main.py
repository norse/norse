from argparse import ArgumentParser


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
        "--bindsnet",
        default=False,
        action="store_true",
        help="Benchmark Bindsnet?",
    )
    parser.add_argument(
        "--genn", default=False, action="store_true", help="Benchmark GeNN?"
    )
    parser.add_argument(
        "--norse",
        default=False,
        action="store_true",
        help="Benchmark Norse?",
    )
    parser.add_argument(
        "--rockpool", default=False, action="store_true", help="Benchmark Rockpool?"
    )
    parser.add_argument(
        "--lava", default=False, action="store_true", help="Benchmark Lava?"
    )
    parser.set_defaults(func=execute_args)


def execute_args(args, continuation):
    # pytype: disable=import-error
    if args.bindsnet:
        import norse.benchmark.lif.bindsnet_lif as bindsnet_lif

        continuation(
            args, bindsnet_lif.lif_feed_forward_benchmark, label="BindsNET_LIF"
        )
    if args.genn:
        import norse.benchmark.lif.genn_lif as genn_lif

        continuation(args, genn_lif.lif_feed_forward_benchmark, label="GeNN_LIF")
    if args.lava:
        import norse.benchmark.lif.lava_lif as lava_lif

        continuation(args, lava_lif.lif_feed_forward_benchmark, label="Lava_LIF")
    if args.norse:
        import norse.benchmark.lif.norse_lif as norse_lif

        if args.profile:
            import torch.autograd.profiler as profiler

            with profiler.profile(
                profile_memory=True, use_cuda=(args.device == "cuda")
            ) as prof:
                continuation(
                    args, norse_lif.lif_feed_forward_benchmark, label="Norse_LIF"
                )
            prof.export_chrome_trace("trace.json")
        else:
            continuation(args, norse_lif.lif_feed_forward_benchmark, label="Norse_LIF")
    if args.rockpool:
        import norse.benchmark.lif.rockpool_lif as rockpool_lif

        continuation(
            args, rockpool_lif.lif_feed_forward_benchmark, label="Rockpool_LIF"
        )
    # pytype: enable=import-error
