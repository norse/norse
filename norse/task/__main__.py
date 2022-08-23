if __name__ == "__main__":  # pragma: no cover
    print(
        """\

.:::     .::    .::::     .:::::::      .:: ::  .::::::::         .:       .::
.: .::   .::  .::    .::  .::    .::  .::    .::.::              .: ::     .::
.:: .::  .::.::        .::.::    .::   .::      .::             .:  .::    .::
.::  .:: .::.::        .::.: .::         .::    .::::::        .::   .::   .::
.::   .: .::.::        .::.::  .::          .:: .::           .:::::: .::  .::
.::    .: ::  .::     .:: .::    .::  .::    .::.::          .::       .:: .::
.::      .::    .::::     .::      .::  .:: ::  .::::::::.::.::         .::.::

  Available tasks (executed with `python -m norse.task.<>`):

    - cartpole
    - cifar10
    - correlation_experiment
    - memory
    - mnist
    - mnist_pl (PyTorch Lightning version)

  NOTE: To execute the above tasks, you may need additional dependencies.

  Please refer to the documentation for usage information: https://norse.ai
"""
    )
