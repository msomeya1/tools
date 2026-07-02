import numpy as np

def Maruyama1964(x1, x2, xi1, xi2, xi3, M11, M22, M33, M12, M23, M31, lam, mu):
    """
    [Surface displacement due to moment tensor]

    In an isotropic, homogeneous, elastic half-space (x3 ≤ 0) with Lamé's constants lam and mu,
    the surface displacement u_i due to the moment tensor density m_pq 
    distributed on a surface S is given by:
    u_i = \int_S G_ip_q * m_pq * dS.
    The expression for G_ip_q is found in Appendix A of Yabuki & Matsu'ura (1992), 
    with its origins back to Maruyama (1964).
    This function Maruyama1964 is its implementation in python.

    [Input]
    - x1, x2: coordinates of the observation point (x3=0).
    - xi1, xi2, xi3: coordinates of the source
    - M11, M22, M33, M12, M23, M31: six independent components of the moment tensor Mij (= mij * dS)
    - lam, mu: Lamé's constants

    [Output]
    - u1, u2, u3: surface displacement
    
    Note that the z-coordinate is defined as positive in the vertically downward direction.

    [References]
    - T. Yabuki, M. Matsu'ura, 
      Geodetic data inversion using a Bayesian information criterion for spatial distribution of fault slip, 
      Geophysical Journal International, Volume 109, Issue 2, May 1992, Pages 363-375, 
      https://doi.org/10.1111/j.1365-246X.1992.tb00102.x

    - Maruyama T.,
      Statical elastic dislocations in an infinite and semi-infinite medium, 
      Bulletin of Earthquake Research Institute, The University of Tokyo, 1964. 42, 289-368.
    """

    gamma = mu / (lam + mu)
    pimu = 1 / (np.pi * mu)

    X = x1 - xi1
    Y = x2 - xi2
    Z =    - xi3 # assume that observation at x3=0
    R = np.sqrt(X**2 + Y**2 + Z**2)
    z = Z / R
    
    F1 = 3     - gamma * (z+3) / (z+1)**3
    F2 = 3 * z - gamma * (z+2) / (z+1)**2

    # x (eqs. A2--A7 in YM1992)
    G11_1 = 0.25 * pimu * (X / R**3) * (F1 * X**2 / R**2 + 3 * gamma / (z+1)**2 - 1)
    G12_2 = 0.25 * pimu * (X / R**3) * (F1 * Y**2 / R**2 +     gamma / (z+1)**2 - 1)
    G13_3 = 0.25 * pimu * (X / R**3) * (3 * z**2         +     gamma            - 1)

    G11_2__G12_1 = 0.5 * pimu * (Y     / R**3) * (F1 * X**2 / R**2 + gamma / (z+1)**2)
    G11_3__G13_1 = 1.5 * pimu * (X**2  / R**4) * z
    G12_3__G13_2 = 1.5 * pimu * (X * Y / R**4) * z

    # y (eqs. A8--A13)
    G21_1 = 0.25 * pimu * (Y / R**3) * (F1 * X**2 / R**2 +     gamma / (z+1)**2 - 1)
    G22_2 = 0.25 * pimu * (Y / R**3) * (F1 * Y**2 / R**2 + 3 * gamma / (z+1)**2 - 1)
    G23_3 = 0.25 * pimu * (Y / R**3) * (3 * z**2         +     gamma            - 1)

    G21_2__G22_1 = 0.5 * pimu * (X     / R**3) * (F1 * Y**2 / R**2 + gamma / (z+1)**2)
    G21_3__G23_1 = 1.5 * pimu * (X * Y / R**4) * z
    G22_3__G23_2 = 1.5 * pimu * (Y**2  / R**4) * z

    # z (eqs. A14--A19)
    G31_1 = 0.25 * pimu * (1 / R**2) * (F2 * X**2 / R**2 - z + gamma / (z+1))
    G32_2 = 0.25 * pimu * (1 / R**2) * (F2 * Y**2 / R**2 - z + gamma / (z+1))
    G33_3 = 0.25 * pimu * (1 / R**2) * (3 * z**3 + (gamma-1) * z)

    G31_2__G32_1 = 0.5 * pimu * (X * Y / R**4) * F2
    G31_3__G33_1 = 1.5 * pimu * (X     / R**3) * z**2
    G32_3__G33_2 = 1.5 * pimu * (Y     / R**3) * z**2
    
    
    # sum
    u1 = (G11_1        * M11 + G12_2        * M22 + G13_3        * M33 \
        + G11_2__G12_1 * M12 + G12_3__G13_2 * M23 + G11_3__G13_1 * M31)
    u2 = (G21_1        * M11 + G22_2        * M22 + G23_3        * M33 \
        + G21_2__G22_1 * M12 + G22_3__G23_2 * M23 + G21_3__G23_1 * M31)
    u3 = (G31_1        * M11 + G32_2        * M22 + G33_3        * M33 \
        + G31_2__G32_1 * M12 + G32_3__G33_2 * M23 + G31_3__G33_1 * M31)
    
    return u1, u2, u3
