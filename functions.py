
import numpy as np
from numpy.polynomial import Polynomial


def H_defect(gamma,N,q,nd):

    """
    Generate the Hamiltonian for NN graph with a defect at defect site.
    
    Args:
        N (int): Number of nodes (system size).
        gamma (float): controls the 'hopping strength'.
        nd (int): Defect site location.
        q (float): Controls defect strength.
        
    Returns:
        np.ndarray: The resulting Hamiltonian in matrix form.

    """

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


def SCE(E, gamma, N, q):

    """
    Defining the self consistency condition 1 - qF(E) [Equation 11]
    
    Args:
        E (float): energy variable 
        N (int): Number of nodes (system size).
        gamma (float): controls the 'hopping strength'.
        q (float): Controls defect strength.
        
    Returns:
        np.float: The resulting functional value of 1 - qF(E) for a given E.

    """

    a = 0
    for r in range(N):
        a = a + (q/N)* (1.0/(E + 2*gamma*np.cos(2*np.pi*r/N)))

    return a + 1 


def chebyt_poly(n):

    """
    Generating Chebyshev polynomial of first kind T_n by recurrence relation.
    T_0(x) = 1
    T_1(x) = x
    T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
    
    Args:
        n (float): polynomial order
        
    Returns:
        np.polynomial.Polynomial: The Chebyshev polynomial T_n(x) represented in the standard power basis.

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
    Generating Chebyshev polynomial of second kind U_n by recurrence relation.
    U_0(x) = 1
    U_1(x) = 2x
    U_{n+1}(x) = 2x U_n(x) - U_{n-1}(x)


    Args:
        n (float): polynomial order
        
    Returns:
         np.polynomial.Polynomial: The Chebyshev polynomial U_n(x) represented in the standard power basis.

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
    Writing the factorized bright-sector polynomial corresponding to the self-consistency equation (See Appendix A)

    Args:
        N (int): Number of nodes (system size).
        gamma (float): controls the 'hopping strength'.
        q (float): Controls defect strength.

    Returns:
        np.polynomial.Polynomial

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

    """
    Computing bright eigenvalues from the roots of  Chebyshev polynomial equation Q_B_poly

    Args:
        N (int): Number of nodes (system size).
        gamma (float): controls the 'hopping strength'.
        q (float): Controls defect strength.

    Returns:
        np.float: bright eigenvalues

    """

    roots = np.real(Q_B_poly(N, q, gamma).roots())

    return -2*gamma*np.sort(roots)



#computing m
def MSD_(N, q, l, gamma=1.0, n0=2):

    """
    Computing Mean squared displacement (MSD) [Equation 15]
    Args:
        N (int): Number of nodes (system size).
        gamma (float): controls the 'hopping strength'.
        q (float): Controls defect strength.
        l (int): distance of the defect site from the initial site
        n0 (int): initial site.

    Returns:
        np.float: MSD

    """

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

# computing 
def q_star(l, gamma=1.0):

    """
    Computing critical q_* from the theory [Equation 33]

    Args:
        gamma (float): controls the 'hopping strength'.
        l (int): distance of the defect site from the initial site

    Returns:
        np.float: critical q_*

    """

    return 2 * gamma * np.sinh(0.5 * np.arcsinh(1.0 / l) )



def MSD_sector_wise(N, q, l, gamma=1.0, n0=2):
    '''
    Computing Mean squared displacement (MSD) [Equation 34]
    Args:
        N (int): Number of nodes (system size).
        gamma (float): controls the 'hopping strength'.
        q (float): Controls defect strength.
        l (int): distance of the defect site from the initial site
        n0 (int): initial site.

    Returns:
        Tuple: (sector wise initial weight list, sector wise second moments list)
    '''

    nd = (n0 + l) % N
    energies, eigenvectors = np.linalg.eigh(H_defect(gamma,N,q,nd))

    #spectral weights
    initial_weights = np.abs(eigenvectors[n0, :])**2

    # Infinite-time-occupation probability for each eigenvectors
    P_bar = np.abs(eigenvectors)**2 

    sites = np.arange(N)
    separation = np.abs(sites - n0)
    distance = np.minimum(separation, N - separation)

    moments = (distance**2)@ P_bar

    #classifying into bright and dark eigenvalues from the overlap with the defect site
    overlap_arr = np.array([np.abs(eigenvectors[:,i][nd])**2 for i in range(N)])
    mask_bright = overlap_arr > 1e-16
    dark_weights = initial_weights[np.invert(mask_bright)].sum()
    dark_MSD = (initial_weights[np.invert(mask_bright)]*moments[np.invert(mask_bright)]).sum()

    #Bright localized state
    mask_localized = energies < -2 * gamma
    localized_weights = initial_weights[mask_localized].sum()
    localized_MSD = (initial_weights[mask_localized]*moments[mask_localized]).sum()


    #bright extended state
    mask_extended = ~(np.invert(mask_bright) | mask_localized)
    extended_weights = initial_weights[mask_extended].sum()
    extended_MSD = (initial_weights[mask_extended]*moments[mask_extended]).sum()

    
    W = np.array([dark_weights, extended_weights,  localized_weights,])
    M = np.array([dark_MSD/dark_weights, extended_MSD/extended_weights,  localized_MSD/localized_weights])

    return W, M