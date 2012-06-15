"""
technique used in Sequential Quadratic Programming whereby if the approximation of a function at a point produces an indefinite quadratic (i.e., Hessian has negative eigenvalues), then the the approximation is coerced into being positive semi-definite.
"""
import sympy
import numpy as np


Q = sympy.Matrix( [[1, 0 ],
                   [0, -2] ] )

def force_definite(Q):
    eigvectors, eigvalues = Q.diagonalize()
    #Q = eigvectors * eigvalues * eigvectors.inv()
    #fixed_eigvalues = [eigvalues[i,i] if eigvalues[i,i] > 0 else 0 for i in range(Q.shape[0])] #make negative eigvalues zero
    fixed_eigvalues = [eigvalues[i,i] - min(eigvalues) for i in range(Q.shape[0])] #increase all eigenvalues by same amount such that the smallest is zero.
    fixed_eigvalues = sympy.diag(*fixed_eigvalues)
    return eigvectors* fixed_eigvalues * eigvectors.inv()

Qd = force_definite(Q)

sympy.var(['x','y'])

u = sympy.Matrix([x,y])

f = u.T * Qd * u
f = f[0,0]

sympy.Plot(f)
