## Release checklist

This checklist exists to ensure releases are correctly packaged and published.

* Bump the version number
  * `setup.py`
  * `publish/conda/meta.yaml`
  * `norse/__init__.py`
  * `docs/_config.yml`
* Deploy to PyPi
  * Ensure the action is properly deployed
  * If it fails, reproduce manually with `manylinux` (`quay.io/pypa/manylinux2014_x86_64`)
  * Remember to validate the content of the `sdist` package wrt. `MANIFEST.in`: are we including all the necessary files? Is `requirements.txt` in the package?