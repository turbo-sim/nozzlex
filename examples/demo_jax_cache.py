
import time
import jax
import jax.numpy as jnp

# Enable JAX caching
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "all")


@jax.jit
def heavy_compute(x):
    for _ in range(400):
        x = jnp.sin(x) ** 2 + jnp.log1p(x)
    return x.sum()

def run_once():
    x = jnp.ones((1000, 1000))
    start = time.time()
    y = heavy_compute(x).block_until_ready()
    end = time.time()
    print(f"Result: {y:.3f}, time: {end - start:.2f} s")

if __name__ == "__main__":
    run_once()
    run_once()
    run_once()
    run_once()
