from numba import njit
    
@njit
def f_jacob(x,p,exog=[0]):
    
    ### This code was generated by Python.
    ### Eichenbaum, Rebelo and Trabandt Model with Resistant Virus Strain.
    
    from sympy import DiracDelta
    from snowdrop.src.preprocessor.condition import IfThenElse,IfThen,Derivative,Subs,Positive,Negative,myzif
    from snowdrop.src.preprocessor.functions import Heaviside

    import numpy as np
    from numpy import log,exp,sin,cos,tan,sqrt,sign
    from numpy import maximum as Max, min as Min, abs as Abs
     
    # Initialize variables
    _xi_1 = 0 
    _xi_2 = 0 
    _xi_3 = 0 
    y__ = x[0]
    k__ = x[1]
    n__ = x[2]
    w__ = x[3]
    rk__ = x[4]
    x__ = x[5]
    c__ = x[6]
    s1__ = x[8]
    s2__ = x[9]
    i__ = x[10]
    i1__ = x[11]
    i2__ = x[12]
    r__ = x[13]
    r1__ = x[14]
    r2__ = x[15]
    v__ = x[16]
    ns__ = x[17]
    ni__ = x[18]
    nr__ = x[19]
    cs__ = x[20]
    ci__ = x[21]
    cr__ = x[22]
    tau__ = x[23]
    tau1__ = x[24]
    tau2__ = x[25]
    lambtilde__ = x[26]
    lamtau__ = x[27]
    lami__ = x[28]
    lams__ = x[29]
    lamr__ = x[30]
    Rb__ = x[33]
    pie__ = x[34]
    mc__ = x[35]
    F__ = x[36]
    Kf__ = x[37]
    rr__ = x[38]
    pbreve__ = x[39]
    yF__ = x[40]
    kF__ = x[41]
    nF__ = x[42]
    wF__ = x[43]
    rkF__ = x[44]
    xF__ = x[45]
    cF__ = x[46]
    sF1__ = x[48]
    sF2__ = x[49]
    iF__ = x[50]
    iF1__ = x[51]
    iF2__ = x[52]
    rF__ = x[53]
    rF1__ = x[54]
    rF2__ = x[55]
    vF__ = x[56]
    nsF__ = x[57]
    niF__ = x[58]
    nrF__ = x[59]
    csF__ = x[60]
    ciF__ = x[61]
    crF__ = x[62]
    tauF__ = x[63]
    tauF1__ = x[64]
    tauF2__ = x[65]
    lambtildeF__ = x[66]
    lamtauF__ = x[67]
    lamiF__ = x[68]
    lamsF__ = x[69]
    lamrF__ = x[70]
    RbF__ = x[73]
    pieF__ = x[74]
    mcF__ = x[75]
    FF__ = x[76]
    KfF__ = x[77]
    rrF__ = x[78]
    pbreveF__ = x[79]
    ei1__ = x[80]
    ei2__ = x[81]
    ed__ = x[82]


    # Set parameters
    xi = p[0]
    rpi = p[1]
    rx = p[2]
    gam = p[3]
    pi1 = p[4]
    pi2 = p[5]
    pi3 = p[6]
    mult = p[7]
    mult2 = p[8]
    pir = p[9]
    pid = p[10]
    betta = p[11]
    i_ini = p[12]
    d_ini = p[13]
    A = p[14]
    theta = p[15]
    alfa = p[16]
    inc_target = p[17]
    n_target = p[18]
    delta = p[19]
    g_ss = p[20]
    eta = p[21]
    xi_flex = p[22]
    pie_ss = p[23]
    rr_ss = p[24]
    Rb_ss = p[25]
    lockdown_policy = p[26]
    sigma = p[27]
    theta_lockdown = p[28]
    vaccination_policy = p[29]
    vaccination_rate = p[30]
    virus_resistant_strain = p[31]
    virus_variant_start = p[32]
    ex = p[33]

    # Jacobian: 
    jacobian = np.zeros((80,83))
    jacobian[0,0] = 1
    jacobian[0,2] = -A*alfa*k__**(1 - alfa)*n__**alfa*pbreve__/n__
    jacobian[0,39] = -A*k__**(1 - alfa)*n__**alfa
    jacobian[1,3] = -alfa*rk__**(1 - alfa)*w__**alfa*(1 - alfa)**(alfa - 1)/(A*alfa**alfa*w__)
    jacobian[1,4] = -rk__**(1 - alfa)*w__**alfa*(1 - alfa)*(1 - alfa)**(alfa - 1)/(A*alfa**alfa*rk__)
    jacobian[1,35] = 1
    jacobian[2,2] = -A*alfa*k__**(1 - alfa)*mc__*n__**(alfa - 1)*(alfa - 1)/n__
    jacobian[2,3] = 1
    jacobian[2,35] = -A*alfa*k__**(1 - alfa)*n__**(alfa - 1)
    jacobian[3,1] = delta
    jacobian[3,5] = -1
    jacobian[4,0] = 1
    jacobian[4,5] = -1
    jacobian[4,6] = -1
    jacobian[5,2] = 1
    jacobian[5,17] = -s1__
    jacobian[5,18] = -i1__
    jacobian[5,19] = -r1__
    jacobian[6,6] = 1
    jacobian[6,20] = -s1__
    jacobian[6,21] = -i1__
    jacobian[6,22] = -r1__
    jacobian[7,17] = -i1__*ni__*pi2*s1__*(-lockdown_policy*theta_lockdown + 1)**2
    jacobian[7,18] = -i1__*ns__*pi2*s1__*(-lockdown_policy*theta_lockdown + 1)**2
    jacobian[7,20] = -ci__*i1__*pi1*s1__*(-lockdown_policy*theta_lockdown + 1)**2
    jacobian[7,21] = -cs__*i1__*pi1*s1__*(-lockdown_policy*theta_lockdown + 1)**2
    jacobian[7,24] = 1
    jacobian[8,17] = -i2__*ni__*pi2*s2__*virus_resistant_strain*(-lockdown_policy*theta_lockdown + 1)**2
    jacobian[8,18] = -i2__*ns__*pi2*s2__*virus_resistant_strain*(-lockdown_policy*theta_lockdown + 1)**2
    jacobian[8,20] = -ci__*i2__*pi1*s2__*virus_resistant_strain*(-lockdown_policy*theta_lockdown + 1)**2
    jacobian[8,21] = -cs__*i2__*pi1*s2__*virus_resistant_strain*(-lockdown_policy*theta_lockdown + 1)**2
    jacobian[8,25] = 1
    jacobian[9,23] = 1
    jacobian[9,24] = -1
    jacobian[9,25] = -1
    jacobian[10,16] = 1
    jacobian[10,24] = 1
    jacobian[11,25] = 1
    jacobian[12,16] = 1
    jacobian[12,23] = 1
    jacobian[13,11] = pid + pir
    jacobian[13,24] = -1
    jacobian[13,80] = -1
    jacobian[14,12] = -virus_resistant_strain*(-pir - pid/mult2)
    jacobian[14,25] = -virus_resistant_strain
    jacobian[14,81] = -1
    jacobian[15,10] = 1
    jacobian[15,11] = -1
    jacobian[15,12] = -ex
    jacobian[16,16] = -1
    jacobian[18,13] = 1
    jacobian[18,14] = -1
    jacobian[18,15] = -1
    jacobian[19,16] = 1
    jacobian[20,82] = -1
    jacobian[22,20] = -1/cs__**2
    jacobian[22,21] = i__*lamtau__*pi1
    jacobian[22,26] = -1
    jacobian[22,27] = ci__*i__*pi1
    jacobian[23,21] = -1/ci__**2
    jacobian[23,26] = -1
    jacobian[24,22] = -1/cr__**2
    jacobian[24,26] = -1
    jacobian[25,3] = -lambtilde__*(-lockdown_policy*theta_lockdown + 1)
    jacobian[25,17] = theta
    jacobian[25,18] = -i__*lamtau__*pi2*(-lockdown_policy*theta_lockdown + 1)
    jacobian[25,26] = -w__*(-lockdown_policy*theta_lockdown + 1)
    jacobian[25,27] = -i__*ni__*pi2*(-lockdown_policy*theta_lockdown + 1)
    jacobian[26,3] = -lambtilde__*(-lockdown_policy*theta_lockdown + 1)
    jacobian[26,18] = theta
    jacobian[26,26] = -w__*(-lockdown_policy*theta_lockdown + 1)
    jacobian[27,3] = -lambtilde__
    jacobian[27,19] = theta
    jacobian[27,26] = -w__
    jacobian[28,26] = -betta*(-delta + rk__ + 1) + 1
    jacobian[29,27] = -1
    jacobian[29,28] = 1
    jacobian[29,29] = -1
    jacobian[30,10] = lamtau__*(ci__*cs__*pi1 + ni__*ns__*pi2 + pi3)
    jacobian[30,29] = 1 - 1/betta
    jacobian[31,28] = -pid - pir + 1 - 1/betta
    jacobian[32,30] = 1 - 1/betta
    jacobian[33,26] = -Rb__*betta/pie__ + 1
    jacobian[33,33] = -betta*lambtilde__/pie__
    jacobian[34,33] = -1/pie__
    jacobian[34,38] = 1
    jacobian[35,0] = -gam*lambtilde__*mc__
    jacobian[35,26] = -gam*mc__*y__
    jacobian[35,35] = -gam*lambtilde__*y__
    jacobian[35,37] = -betta*pie__**(gam/(gam - 1))*xi + 1
    jacobian[36,0] = -lambtilde__
    jacobian[36,26] = -y__
    jacobian[36,36] = -betta*pie__**(1/(gam - 1))*xi + 1
    jacobian[37,34] = F__*pie__**(1/(gam - 1))*xi*((-pie__**(1/(gam - 1))*xi + 1)/(1 - xi))**(1 - gam)*(1 - gam)/(pie__*(gam - 1)*(-pie__**(1/(gam - 1))*xi + 1))
    jacobian[37,36] = -((-pie__**(1/(gam - 1))*xi + 1)/(1 - xi))**(1 - gam)
    jacobian[37,37] = 1
    jacobian[38,34] = gam*pie__**(1/(gam - 1))*xi*((-pie__**(1/(gam - 1))*xi + 1)/(1 - xi))**gam*(1 - xi)/(pie__*(gam - 1)*(-pie__**(1/(gam - 1))*xi + 1)) - gam*pie__**(gam/(gam - 1))*xi/(pbreve__*pie__*(gam - 1))
    jacobian[38,39] = pie__**(gam/(gam - 1))*xi/pbreve__**2 - 1/pbreve__**2
    jacobian[39,0] = -rx/y__
    jacobian[39,33] = 1
    jacobian[39,34] = -rpi/pie__
    jacobian[39,40] = rx/yF__
    jacobian[40,40] = 1
    jacobian[40,42] = -A*alfa*kF__**(1 - alfa)*nF__**alfa*pbreveF__/nF__
    jacobian[40,79] = -A*kF__**(1 - alfa)*nF__**alfa
    jacobian[41,43] = -alfa*rkF__**(1 - alfa)*wF__**alfa*(1 - alfa)**(alfa - 1)/(A*alfa**alfa*wF__)
    jacobian[41,44] = -rkF__**(1 - alfa)*wF__**alfa*(1 - alfa)*(1 - alfa)**(alfa - 1)/(A*alfa**alfa*rkF__)
    jacobian[41,75] = 1
    jacobian[42,42] = -A*alfa*kF__**(1 - alfa)*mcF__*nF__**(alfa - 1)*(alfa - 1)/nF__
    jacobian[42,43] = 1
    jacobian[42,75] = -A*alfa*kF__**(1 - alfa)*nF__**(alfa - 1)
    jacobian[43,41] = delta
    jacobian[43,45] = -1
    jacobian[44,40] = 1
    jacobian[44,45] = -1
    jacobian[44,46] = -1
    jacobian[45,42] = 1
    jacobian[45,57] = -sF1__
    jacobian[45,58] = -iF1__
    jacobian[45,59] = -rF1__
    jacobian[46,46] = 1
    jacobian[46,60] = -sF1__
    jacobian[46,61] = -iF1__
    jacobian[46,62] = -rF1__
    jacobian[47,57] = -iF1__*niF__*pi2*sF1__*(-lockdown_policy*theta_lockdown + 1)**2
    jacobian[47,58] = -iF1__*nsF__*pi2*sF1__*(-lockdown_policy*theta_lockdown + 1)**2
    jacobian[47,60] = -ciF__*iF1__*pi1*sF1__*(-lockdown_policy*theta_lockdown + 1)**2
    jacobian[47,61] = -csF__*iF1__*pi1*sF1__*(-lockdown_policy*theta_lockdown + 1)**2
    jacobian[47,64] = 1
    jacobian[48,57] = -iF2__*niF__*pi2*sF2__*virus_resistant_strain*(-lockdown_policy*theta_lockdown + 1)**2
    jacobian[48,58] = -iF2__*nsF__*pi2*sF2__*virus_resistant_strain*(-lockdown_policy*theta_lockdown + 1)**2
    jacobian[48,60] = -ciF__*iF2__*pi1*sF2__*virus_resistant_strain*(-lockdown_policy*theta_lockdown + 1)**2
    jacobian[48,61] = -csF__*iF2__*pi1*sF2__*virus_resistant_strain*(-lockdown_policy*theta_lockdown + 1)**2
    jacobian[48,65] = 1
    jacobian[49,63] = 1
    jacobian[49,64] = -1
    jacobian[49,65] = -1
    jacobian[50,56] = 1
    jacobian[50,64] = 1
    jacobian[51,65] = 1
    jacobian[52,56] = 1
    jacobian[52,63] = 1
    jacobian[53,51] = pid + pir
    jacobian[53,64] = -1
    jacobian[53,80] = -1
    jacobian[54,52] = -virus_resistant_strain*(-pir - pid/mult2)
    jacobian[54,65] = -virus_resistant_strain
    jacobian[54,81] = -1
    jacobian[55,50] = 1
    jacobian[55,51] = -1
    jacobian[55,52] = -ex
    jacobian[56,56] = -1
    jacobian[58,53] = 1
    jacobian[58,54] = -1
    jacobian[58,55] = -1
    jacobian[59,56] = 1
    jacobian[60,82] = -1
    jacobian[62,60] = -1/csF__**2
    jacobian[62,61] = iF__*lamtauF__*pi1
    jacobian[62,66] = -1
    jacobian[62,67] = ciF__*iF__*pi1
    jacobian[63,61] = -1/ciF__**2
    jacobian[63,66] = -1
    jacobian[64,62] = -1/crF__**2
    jacobian[64,66] = -1
    jacobian[65,43] = -lambtildeF__*(-lockdown_policy*theta_lockdown + 1)
    jacobian[65,57] = theta
    jacobian[65,58] = -iF__*lamtauF__*pi2*(-lockdown_policy*theta_lockdown + 1)
    jacobian[65,66] = -wF__*(-lockdown_policy*theta_lockdown + 1)
    jacobian[65,67] = -iF__*niF__*pi2*(-lockdown_policy*theta_lockdown + 1)
    jacobian[66,43] = -lambtildeF__*(-lockdown_policy*theta_lockdown + 1)
    jacobian[66,58] = theta
    jacobian[66,66] = -wF__*(-lockdown_policy*theta_lockdown + 1)
    jacobian[67,43] = -lambtildeF__
    jacobian[67,59] = theta
    jacobian[67,66] = -wF__
    jacobian[68,66] = -betta*(-delta + rkF__ + 1) + 1
    jacobian[69,67] = -1
    jacobian[69,68] = 1
    jacobian[69,69] = -1
    jacobian[70,50] = lamtauF__*(ciF__*csF__*pi1 + niF__*nsF__*pi2 + pi3)
    jacobian[70,69] = 1 - 1/betta
    jacobian[71,68] = -pid - pir + 1 - 1/betta
    jacobian[72,70] = 1 - 1/betta
    jacobian[73,66] = -RbF__*betta/pieF__ + 1
    jacobian[73,73] = -betta*lambtildeF__/pieF__
    jacobian[74,73] = -1/pieF__
    jacobian[74,78] = 1
    jacobian[75,40] = -gam*lambtildeF__*mcF__
    jacobian[75,66] = -gam*mcF__*yF__
    jacobian[75,75] = -gam*lambtildeF__*yF__
    jacobian[75,77] = -betta*pieF__**(gam/(gam - 1))*xi_flex + 1
    jacobian[76,40] = -lambtildeF__
    jacobian[76,66] = -yF__
    jacobian[76,76] = -betta*pieF__**(1/(gam - 1))*xi_flex + 1
    jacobian[77,74] = FF__*pieF__**(1/(gam - 1))*xi_flex*((-pieF__**(1/(gam - 1))*xi_flex + 1)/(1 - xi_flex))**(1 - gam)*(1 - gam)/(pieF__*(gam - 1)*(-pieF__**(1/(gam - 1))*xi_flex + 1))
    jacobian[77,76] = -((-pieF__**(1/(gam - 1))*xi_flex + 1)/(1 - xi_flex))**(1 - gam)
    jacobian[77,77] = 1
    jacobian[78,74] = gam*pieF__**(1/(gam - 1))*xi_flex*((-pieF__**(1/(gam - 1))*xi_flex + 1)/(1 - xi_flex))**gam*(1 - xi_flex)/(pieF__*(gam - 1)*(-pieF__**(1/(gam - 1))*xi_flex + 1)) - gam*pieF__**(gam/(gam - 1))*xi_flex/(pbreveF__*pieF__*(gam - 1))
    jacobian[78,79] = pieF__**(gam/(gam - 1))*xi_flex/pbreveF__**2 - 1/pbreveF__**2
    jacobian[79,73] = 1
    jacobian[79,74] = -rpi/pieF__

    return jacobian 
