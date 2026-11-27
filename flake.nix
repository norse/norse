{
  description = "Norse - a library for deep learning with spiking neural networks";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f:
        nixpkgs.lib.genAttrs systems (system:
          f (import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          }));

      # Nix evaluation is pure and cannot probe the GPU directly, but the
      # driver's control device exists iff the NVIDIA kernel module is loaded,
      # and builtins.pathExists on it is permitted under pure eval. This picks
      # the matching torch wheel below: a CUDA build on a machine that can use
      # it, a CPU build (~200 MB instead of ~2.5 GB) on one that cannot.
      cudaAvailable = builtins.pathExists /dev/nvidiactl;
    in
    {
      devShells = forAllSystems (pkgs:
        let
          pythonPackages = pkgs.python312Packages;
          torchIndex =
            if cudaAvailable && pkgs.stdenv.hostPlatform.isLinux
            then "https://download.pytorch.org/whl/cu128"
            else "https://download.pytorch.org/whl/cpu";
        in
        {
          default = pkgs.mkShell {
            name = "impurePythonEnv";
            venvDir = "./.venv";

            buildInputs = [
              # A Python interpreter including the 'venv' module is required to
              # bootstrap the environment.
              pythonPackages.python

              # Executes shell code to initialize a venv in $venvDir before
              # dropping into the shell
              pythonPackages.venvShellHook

              # Dependencies we would like to use from nixpkgs, which will be
              # added to PYTHONPATH and thus made accessible from the venv.
              pythonPackages.pybind11

              pkgs.cmake
              pkgs.ninja

              # torch.compile's inductor backend shells out to `openssl` when it
              # hashes generated kernel sources; without it every *_compile test
              # fails with InductorError: FileNotFoundError: 'openssl'.
              pkgs.openssl
            ];

            # Run this command, only after creating the virtual environment.
            # torch is installed by pip rather than taken from nixpkgs: the
            # nixpkgs torch-bin wheel is CUDA-only on x86_64-linux (see
            # pkgs/development/python-modules/torch/bin/binary-hashes.nix), and
            # the CPU torch there is a multi-hour source build.
            postVenvCreation = ''
              unset SOURCE_DATE_EPOCH
              pip install --upgrade pip
              pip install torch torchvision --index-url ${torchIndex}
              pip install nir nirtorch
              # Test tooling comes from pyproject's [dependency-groups] test,
              # the same set CI installs, rather than a second hand-kept list.
              pip install --group test
              pip install -e . --no-deps
            '';

            # Now we can execute any commands within the virtual environment.
            postShellHook = ''
              # Wheels installed by pip link against these at runtime (numpy
              # needs libz, for instance). Deliberately *not* glibc: see
              # nixpkgs issue of mixing ld.so and libc.so across versions.
              export LD_LIBRARY_PATH=${
                pkgs.lib.makeLibraryPath [
                  pkgs.stdenv.cc.cc.lib
                  pkgs.zlib
                  pkgs.libffi
                  pkgs.openssl
                ]
              }:/run/opengl-driver/lib/
              # allow pip to install wheels
              unset SOURCE_DATE_EPOCH
            '';
          };
        });

      formatter = forAllSystems (pkgs: pkgs.nixpkgs-fmt);
    };
}
