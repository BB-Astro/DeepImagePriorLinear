# Third-party code

- `models/` (network architectures: skip, UNet, ResNet, texture nets,
  downsampler) and parts of `utils/common_utils.py` are adapted from
  [DmitryUlyanov/deep-image-prior](https://github.com/DmitryUlyanov/deep-image-prior),
  Copyright Dmitry Ulyanov, released under the Apache License 2.0.
  The original author asks to be contacted before any commercial use of
  his software.

- The ES-WMV early-stopping criterion implemented in
  `utils/early_stopping.py` follows the method described in
  ["Early Stopping for Deep Image Prior"](https://arxiv.org/abs/2112.06074)
  (H. Wang, T. Li, Z. Zhuang, T. Chen, H. Liang, J. Sun, TMLR 2023);
  the implementation here is original.

Everything else: MIT License, see `LICENSE`.
