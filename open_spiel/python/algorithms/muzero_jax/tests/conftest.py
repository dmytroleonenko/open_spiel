import os
import jax

os.environ["JAX_DISABLE_JIT"] = "1"   # must precede the first JAX import
jax.config.update("jax_disable_jit", True) 