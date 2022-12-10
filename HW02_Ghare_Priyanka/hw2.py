import sympy

x, y, z = sympy.symbols('x y z')
f = y * sympy.sin(5 * x) + sympy.exp(y * z) + sympy.log(z)

df_x = sympy.diff(f, x)
df_y = sympy.diff(f, y)
df_z = sympy.diff(f, z)
print(f"df/dx = {df_x}")
print(f"df/dy = {df_y}")
print(f"df/dz = {df_z}")
