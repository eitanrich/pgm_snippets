import numpy as np

# PMAI 2017/18
# Factor product example

na=2
nb=2
nc=3

phi_a = np.random.uniform(size=[na])
phi_b = np.random.uniform(size=[nb])
phi_c = np.random.uniform(size=[nc])
phi_ab = np.random.uniform(size=[na, nb])
phi_ac = np.random.uniform(size=[na, nc])

# Option 1 - using loops
phi_prod_1 = np.zeros([na, nb, nc])
for (a, b, c),_ in np.ndenumerate(phi_prod_1):
    phi_prod_1[a, b, c] = phi_a[a] * phi_b[b] * phi_c[c] * phi_ab[a, b] * phi_ac[a, c]

# Option 2 - using reshape and broadcast operations (faster...)
phi_prod_2 = phi_a.reshape([na, 1, 1]) * phi_b.reshape([1, nb, 1]) * phi_c.reshape([1, 1, nc]) * \
             phi_ab.reshape([na, nb, 1]) * phi_ac.reshape([na, 1, nc])


print(np.all(phi_prod_1 == phi_prod_2))
