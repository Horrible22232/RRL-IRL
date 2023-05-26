# RRL-IRL

# Expert model
Based on https://github.com/MarcoMeter/neroRL

Download the expert model file via `python receive_expert.py`, from the provided source (not available on GitHub due to its large size).

# Jax
To enable CUDA functionality, it is recommended to install JAX using the following command:

```
conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

This command ensures that JAX is installed with CUDA support. Without this installation, CUDA may not function properly.