# Event-based datasets

These datasets exist to provide faster access to native-spiking datasets.
Contributions are welcome!

## IBM DVS gesture dataset (vision)

> This dataset was used to build the real-time, gesture recognition system described in the CVPR 2017 paper titled “A Low Power, Fully Event-Based Gesture Recognition System.” The data was recorded using a DVS128. The dataset contains 11 hand gestures from 29 subjects under 3 illumination conditions and is released under a Creative Commons Attribution 4.0 license. 

Source: https://www.research.ibm.com/dvsgesture/

Note: Outputs **sparse** tensors

## Bit pattern STORE/RECALL memory dataset (memory)

A memory dataset that generates random patterns of 4-bit data, and
a 2-bit command pattern (store and recall).

Source: https://arxiv.org/abs/1901.09049

## Speech Commands (audio)

Speech Commands dataset as described in 
"[Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition](https://arxiv.org/abs/1804.03209)".
This is meant as a wrapper around the corresponding SPEECHCOMMANDS dataset defined in torchaudio.

Source: https://arxiv.org/abs/1804.03209

## Spiking Heidelberg Digits (SHD) audio dataset

Spiking audio datasets with 8332 training and 2088 testing samples.

Source: https://compneuro.net/posts/2019-spiking-heidelberg-digits/