
import numpy as np
from numpy.polynomial import Polynomial



# defect Hamiltonian needed for exact diagonalization
def H_defect(gamma,N,q,nd):

    '''
    defect hamiltonian H

    gamma = hopping parameter
    N = number of sites
    q = defect strength
    nd = defect site

    '''

    H = np.zeros((N,N))

    for i in range(1,N):
        H[i,i-1] = -gamma
        H[i-1,i] = -gamma
    
    #defect site
    H[nd,nd] = -q
    
    #periodic boundary condition
    H[0,N-1] = -gamma
    H[N-1,0] = -gamma

    return H



# defing the self consistency condition 1 - qF(E) [Equation 7]
def SCE(E, gamma, N, q):

    '''
    self consistency condition: 1 - qF(E) = 0
    gamma = hopping parameter
    N = number of sites
    q = defect strength

    '''

    a = 0
    for r in range(N):
        a = a + (q/N)* (1.0/(E + 2*gamma*np.cos(2*np.pi*r/N)))

    return a + 1 


#defining the chebyshev polynomials of order n
def chebyt_poly(n):
    """
    generate Chebyshev polynomial T_n(x) by recurrence relation.
    T_0(x) = 1
    T_1(x) = x
    T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
    """

    T0 = Polynomial([1.0])

    T1 = Polynomial([0.0, 1.0])

    if n == 0:
        return T0

    if n == 1:
        return T1

    for _ in range(2, n + 1):
        T0, T1 = T1, Polynomial([0.0,2.0])*T1 - T0

    return T1

def chebyu_poly(n):
    """
    generate Chebyshev polynomial U_n(x) by recurrence relation.
    U_0(x) = 1
    U_1(x) = 2x
    U_{n+1}(x) = 2x U_n(x) - U_{n-1}(x)
    """

    U0 = Polynomial([1.0])

    U1 = Polynomial([0.0, 2.0])

    if n == 0:
        return U0

    if n == 1:
        return U1

    for _ in range(2, n + 1):
        U0, U1 = U1, Polynomial([0.0,2.0])*U1 - U0

    return U1

# Writing the self consistency equation in terms of Chebyshev polynomial
def Q_B_poly(N, q, gamma):
    """
    Bright-sector characteristic polynomial (See Appendix A)
    """
    x = Polynomial([0.0, 1.0])
    a = q / (2.0*gamma)

    if N % 2 == 0:
        m = N // 2

        return (Polynomial([-1.0,0.0,1.0])*chebyu_poly(m - 1)- a*chebyt_poly(m))

    else:
        m = (N - 1) // 2

        return ( Polynomial([-1.0,1.0])*(chebyu_poly(m) + chebyu_poly(m - 1))- a* (chebyu_poly(m) - chebyu_poly(m - 1)))


# getting the eigenvalues from the roots of Chebyshev polynomial equation Q_B_poly
def bright_roots(N, q, gamma):

    '''
    Bright eigenvalues
    '''

    roots = np.real(Q_B_poly(N, q, gamma).roots())

    return -2*gamma*np.sort(roots)



#computing mean squared displacement [Equation 30]
def MSD_(N, q, l, gamma=1.0, n0=2):
    '''
    n0 = initial site
    l = distance from the initial site to the target site
    '''

    nd = n0 + l
    energies, eigenvectors = np.linalg.eigh(H_defect(gamma,N,q, nd))
    
    initial_weights = np.abs(eigenvectors[n0, :])**2

    # Infinite-time-averaged probability for q > 0
    P_bar = np.abs(eigenvectors)**2 @ initial_weights

    sites = np.arange(N)
    separation = np.abs(sites - n0)
    distance = np.minimum(separation, N - separation)

    MSD = np.sum((distance**2)*P_bar)


    return MSD

# computing critical q_* from the theory [Equation 28]
def q_star(l, gamma=1.0):
     return 2 * gamma * np.sinh(0.5 * np.arcsinh(1.0 / l) )