import torchvision


def local_state_dict(name):
    """torchvision's real key names and shapes, without the network download.

    `vgg._vgg` and `mobilenet.mobilenet_v2` rewrite every state-dict key to
    insert `lifted_module` and then call `load_state_dict` strictly, so the key
    names and tensor shapes are what the pretrained path actually exercises;
    the trained values are never asserted on. Building the architecture locally
    reproduces both without fetching several hundred MB.
    """
    return getattr(torchvision.models, name)(weights=None).state_dict()
