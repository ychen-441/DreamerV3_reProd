import jax
import jax.numpy as jnp

def where(condition, x, y):
    assert condition.dtype == bool, f"Condition must be bool, got {condition.dtype}"
    return jnp.where(condition, x, y)

def mask_fn(x, mask):
    return where(mask, x, jax.tree.map(jnp.zeros_like, x))

xs = [
    jnp.array([2.0, 1.0, -jnp.inf]),        # floating
    jnp.array([2, 1, -1]),                  # signed int
    jnp.array([2, 1, 2], dtype=jnp.uint32), # unsigned int
    jnp.array([True, False, True])          # bool
]

bdims = None  # or set to an integer to test slicing

for x in xs:
    if jnp.issubdtype(x.dtype, jnp.floating):
        mask = (x != -jnp.inf)
    elif jnp.issubdtype(x.dtype, jnp.signedinteger):
        mask = (x != -1)
    elif (
        jnp.issubdtype(x.dtype, jnp.unsignedinteger) or
        jnp.issubdtype(x.dtype, bool)
    ):
        shape = x.shape if bdims is None else x.shape[:bdims]
        mask = jnp.full(shape, True, dtype=bool)
    else:
        raise NotImplementedError(x.dtype)

    assert mask.dtype == bool

    masked = mask_fn(x, mask)

    print(f"x: {x}")
    print(f"mask: {mask}")
    print(f"masked: {masked}")
    print("-" * 30)


# Create an example tensor
x = jnp.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6)
print("Original shape:", x.shape)

# Parameters
bdims = 2
fdims = 3

# Compute how many dims to keep
keep = bdims + fdims - 3

# Reshape: keep leading dims, flatten the rest
flattened = x.reshape(*x.shape[:keep], -1)

print("New shape:", flattened.shape)

x = jnp.array([[0, 2],
               [1, 3]])
classes = 4

y = jax.nn.one_hot(x, classes)

print(y)


