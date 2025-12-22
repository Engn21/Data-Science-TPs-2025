import numpy as np

# Here for the u vector we are creating a vector which has a random length
u = np.random.randn(7)

# But for this one we are creating an another vector which is random also
v0 = np.random.randn(7)


#    proj_u(v0) = (u·v0 / u·u) * u
# here the in the code the v vector was created to be perpendicular to u
# to create this vector the sense of the code line used below:
# w = v0 - proj_u(v0) also we know that:
# proj_u(v0) = (u·v0 / u·u) * u
v = v0 - (np.dot(u, v0) / np.dot(u, u)) * u

# And now the code satisfied the perpendicularity of the vectors but 
# they should also have same length so we should satisfy:
#  ||v|| = ||u||
# np.linalg.norm(u) represents the ||u||
#   so we should rearrange the norms as below code by using the ratio
#  between ||u||/||v|| multiplied with v to get the same length with u.
v = v * (np.linalg.norm(u) / np.linalg.norm(v))

# here the code was checked using print functions:
print("u · v = 0", np.allclose(np.dot(u,v), 0)) # all close function checks whether 
#the dot product equals to zero or not that demonstrates perpendicularity.
print("||u|| =", np.linalg.norm(u))   # length of u
print("||v|| =", np.linalg.norm(v))   # length of v (the results should be equal to get the proper answer)
