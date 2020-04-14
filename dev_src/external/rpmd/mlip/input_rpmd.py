# encoding: utf-8 
  
from PES import get_potential, initialize_potential, finalize_potential
   
################################################################################
 
label = 'CN + CH4 -> HCN + CH3'
  
reactants( 
    atoms = ['H', 'C', 'H', 'H', 'H', 'C', 'N'], 
    reactant1Atoms = [1,2,3,4,5], 
    reactant2Atoms = [6,7], 
    Rinf = (15 * 0.52918,"angstrom"), 
) 
 
transitionState(
    geometry = ( 
       [[2.676712E-01,    3.020144E-02,   -6.721314E-03], 
       [4.670000E-02 ,   8.080000E-02 ,  -2.228300E+00], 
       [-1.209103E+00,   -1.455762E+00 ,  -2.807696E+00], 
       [-7.594269E-01,    1.903999E+00 ,  -2.775901E+00], 
       [1.910968E+00 ,  -1.605758E-01,   -3.088537E+00], 
       [5.940022E-01 ,  -4.452269E-02,    3.274114E+00], 
       [8.131560E-01 ,  -9.470499E-02,    5.477420E+00]], 
        "bohr", 
    ), 
    formingBonds = [(1,6)],  
    breakingBonds = [(2,1)], 
) 
 
equivalentTransitionState( 
    formingBonds=[(3,6)], 
    breakingBonds=[(2,3)], 
) 
equivalentTransitionState( 
    formingBonds=[(4,6)], 
    breakingBonds=[(2,4)], 
) 
equivalentTransitionState( 
    formingBonds=[(5,6)],  
    breakingBonds=[(2,5)], 
) 
 
thermostat('Andersen') 
  
################################################################################ 
 
xi_list = numpy.arange(-0.00, 1.25, 1.25) 
#xi_list = [-0.00] 

generateUmbrellaConfigurations( 
    dt = (0.0001,"ps"), 
    evolutionTime = (0.5,"ps"), 
    xi_list = xi_list, 
    kforce = 0.1 * T, 
) 

# xi_list = numpy.arange(-0.05, 1.05, 0.01) 
# windows = [] 
# for xi in xi_list: 
#     window = Window(xi=xi, kforce=0.1*T, trajectories=20, equilibrationTime=(20,"ps"), evolutionTime=(100,"ps")) 
#     windows.append(window) 
#  
# conductUmbrellaSampling( 
#     dt = (0.0001,"ps"), 
#     windows = windows, 
# ) 
#  
# computePotentialOfMeanForce(windows=windows, xi_min=-0.02, xi_max=1.02, bins=5000) 
#  
# computeRecrossingFactor( 
#     dt = (0.0001,"ps"), 
#     equilibrationTime = (20,"ps"), 
#     childTrajectories = 100000, 
#     childSamplingTime = (2,"ps"), 
#     childrenPerSampling = 100, 
#     childEvolutionTime = (0.3,"ps"),
# ) 
#  
# computeRateCoefficient() 
#  
