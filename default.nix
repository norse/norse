with import <nixpkgs> { };
let
  pythonPackages = python311Packages;
in
pkgs.mkShell rec {
  name = "impurePythonEnv";
  venvDir = "./.venv";
  buildInputs = [
    # A Python interpreter including the 'venv' module is required to bootstrap
    # the environment.
    pythonPackages.python

    # This execute some shell code to initialize a venv in $venvDir before
    # dropping into the shell
    pythonPackages.venvShellHook

    # Those are dependencies that we would like to use from nixpkgs, which will
    # add them to PYTHONPATH and thus make them accessible from within the venv.
    pythonPackages.matplotlib
    pythonPackages.pybind11
    # pythonPackages.pytest

    pythonPackages.torch-bin

    cmake
    ninja
  ];

  # Run this command, only after creating the virtual environment
  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install --upgrade pip
    pip install nir nirtorch
    pip install -e . --no-deps
  '';

  # Now we can execute any commands within the virtual environment.
  # This is optional and can be left out to run pip manually.
  postShellHook = ''
    export LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib/:/run/opengl-driver/lib/
    # allow pip to install wheels
    unset SOURCE_DATE_EPOCH
  '';

}
