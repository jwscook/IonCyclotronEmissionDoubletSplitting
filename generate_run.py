
import numpy as np
## constants
me = 9.1093829E-31		# kg #
me_to_mp = 1836.2
me_to_malpha = 7294.3
me_to_mHe3 = 5497.885
me_to_mD = 3671.5
me_to_mT = 5497.93
mu0 = 1.2566371E-06		# Hm^[-1] #
qe = 1.6021766E-19		# C #

## plasma parameters
Z1 = 1 ; Z2 = 2 ; Z3 = 1
m1 = me_to_mD*me; m2 = me_to_mHe3*me; m3 = me_to_mp*me
B0 = 3.7 # T
n0 = 5e19 # 1/m^3
xi3 = 1e-3 #
n3 = xi3*n0 # 1/m^3
E3 = 14.68e6 # eV
v0 = np.sqrt(2*E3*qe/m3) # m/s
vperp_vA = 0.9 #

## formatting 
name = 'D_He3_p_0'
btext = '../julia-1.9.3/bin/julia --proj LMV.jl '
prcnt_conc = np.arange(0,110,5) # concentrations as %
th_spread_beam = ' 0.01'   # with spacing before
th_spread_ring = ' 0.001 ' # "			" and after
th_spreads = th_spread_beam+th_spread_ring
with open('trun.txt', 'a') as file:
    # loop over each concentration
    for xi2 in prcnt_conc:
        print(xi2)
        xi2 /= 100 # decimal
        n2 = xi2*n0
        n1 = (n0/Z1)*(1-xi2*Z2-xi3*Z3)
        xi1 = n1/n0
        if not (n0 == (n1*Z1+n2*Z2+n3*Z3)): #  @assert quasi-neutrality
            raise SystemExit
        rho0 = xi1*m1 + xi2*m2 + xi3*m3
        vA = B0/np.sqrt(mu0*n0*rho0)
        vperp = vA*vperp_vA
        vpara = np.sqrt(v0**2 - vperp**2)
        vpara_v0 = vpara/v0
        print(vpara_v0)
        file.write(btext+str(vpara_v0)+th_spreads+str(xi2)+' '+name+str(int(100*xi2))+'\n')

