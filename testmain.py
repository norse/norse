import norse

from norse.torch.functional.leaky_integrator import LIParameters

from norse.torch.functional.lif_ex import LIFExParameters

from norse.torch.functional.lif_adex import LIFAdExParameters

from norse.torch.functional.lsnn import LSNNParameters

from norse.torch.functional.lif import LIFParameters

from norse.torch.functional.iaf import IAFParameters
import torch
from norse.torch.functional.parameter import default_bio_parameters, DEFAULT_BIO_PARAMS
import copy
import unittest

#OPTION 1

params = \
    {
        'v_th' : torch.tensor(0),
        'v_reset' : torch.tensor(-1)
    }
params = IAFParameters(**default_bio_parameters('iaf', **params))

#OPTION 2

params = IAFParameters(**default_bio_parameters('iaf', v_th=torch.tensor(-1)))

#OPTION 3

a = default_bio_parameters('iaf')
a['v_th'] = torch.tensor(4)
params = IAFParameters(**a)

#print(params)
#print(default_bio_parameters('iaf'))

#print(LIFParameters.bio_default())
#print(LSNNParameters.bio_default())
#print(IAFParameters.bio_default())
#print(LIFAdExParameters.bio_default())
print(LIFExParameters.bio_default())
LIFParameters.bio_default()

default_bio_parameters('lif', v_th=torch.tensor(2.0))

class TestIAFParameters(unittest.TestCase):
    def test_tensor_copy(self):
        copy_a = default_bio_parameters('iaf')
        copy_a['v_th'] = torch.tensor(4)

        self.assertNotEqual(id(copy_a), id(DEFAULT_BIO_PARAMS['iaf']))
        self.assertNotEqual(id(copy_a['v_th']), id(DEFAULT_BIO_PARAMS['iaf']['v_th']))
        self.assertNotEqual(copy_a['v_th'].item(), DEFAULT_BIO_PARAMS['iaf']['v_th'].item())


unittest.main()