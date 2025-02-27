// ======================================
// Basic Quarterly Projection Model (QPM)
// ======================================

//transition_variables
var  L_GDP  L_GDP_BAR  L_GDP_GAP DLA_GDP D4L_GDP DLA_GDP_BAR  MCI
     L_CPI DLA_CPI  E_DLA_CPI D4L_CPI D4L_CPI_TAR  RMC L_S  DLA_S
     D4L_S  PREM  RS  RR RR_BAR RR_GAP RSNEUTRAL L_Z L_Z_BAR L_Z_GAP
     DLA_Z DLA_Z_BAR  L_GDP_RW_GAP RS_RW RR_RW RR_RW_BAR  RR_RW_GAP L_CPI_RW DLA_CPI_RW
     OBS_L_GDP OBS_L_CPI OBS_RS OBS_L_S OBS_D4L_CPI_TAR OBS_L_GDP_RW_GAP
     OBS_DLA_CPI_RW OBS_RS_RW

//transition_shocks
varexo SHK_L_GDP_GAP SHK_DLA_CPI SHK_L_S SHK_RS SHK_D4L_CPI_TAR
       SHK_RR_BAR SHK_DLA_Z_BAR SHK_DLA_GDP_BAR SHK_L_GDP_RW_GAP 
       SHK_RS_RW SHK_DLA_CPI_RW SHK_RR_RW_BAR;

//parameters
parameters b1 b2 b3 b4 a1 a2 a3 e1 g1 g2 g3
	   rho_D4L_CPI_TAR rho_DLA_Z_BAR rho_RR_BAR rho_DLA_GDP_BAR
	   rho_L_GDP_RW_GAP rho_RS_RW rho_DLA_CPI_RW rho_RR_RW_BAR
	   ss_D4L_CPI_TAR ss_DLA_Z_BAR ss_RR_BAR ss_DLA_GDP_BAR
	   ss_DLA_CPI_RW ss_RR_RW_BAR;


     #Potential output growth
     ss_DLA_GDP_BAR = 2.5

     #Domestic inflation target
     ss_D4L_CPI_TAR = 2 

     #Domestic real interest rate 
     ss_RR_BAR = 0.5 

     #Change in the real ER (negative number - real appreciation)
     ss_DLA_Z_BAR = -1.5 

     #Foreign inflation or inflation target
     ss_DLA_CPI_RW = 2

     #Level of foreign real interest rate
     ss_RR_RW_BAR = 0.75

     # Typical and specific parameter values be used in calibrations    
     #-------- 1. Aggregate demand equation (the IS curve)
     #L_GDP_GAP =      b1*L_GDP_GAP(-1) - b2*MCI + b3*L_GDP_RW_GAP + SHK_L_GDP_GAP
     #MCI =      b4*RR_GAP + (1-b4)*(- L_Z_GAP)

     #output persistence
     b1 = 0.8      #b1 varies between 0.1 (extremely flexible) and 0.95(extremely persistent)

     #policy passthrough (the impact of monetary policy on real economy) 
     b2 = 0.3      #b2 varies between 0.1 (low impact) to 0.5 (strong impact)

     #the impact of external demand on domestic output 
     b3 = 0.5      #b3 varies between 0.1 and 0.7

     #the weight of the real interest rate and real exchange rate gaps in Monetary Conditions Index
     b4 = 0.7      #b4 varies from 0.3 to 0.8

     #-------- 2. Inflation equation (the Phillips curve)
     #DLA_CPI =      a1*DLA_CPI(-1) + (1-a1)*DLA_CPI(+1) + a2*RMC + SHK_DLA_CPI
     #RMC   =      a3*L_GDP_GAP + (1-a3)*L_Z_GAP

     #inflation persistence 
     a1 = 0.7      #a1 varies between 0.4 (low persistence) to 0.9 (high persistence)

     #passthrough of marginal costs to inflation (the impact of rmc on inflation) 
     a2 = 0.2      #a2 varies between 0.1 (a flat Phillips curve and a high sacrifice ratio) to 0.5 (a steep Phillips curve and a low sacrifice ratio)

     #the ratio of domestic costs in firms' aggregate costs
     a3 = 0.7      #a3 varies between 0.9 (for a relatively more closed economy) to 0.5 (for a relatively more open economy)

     #-------- 3. Monetary policy reaction function (a forward-looking Taylor rule)
     #RS = g1*RS(-1) + (1-g1)*(RSNEUTRAL + g2*(D4L_CPI(+4) - D4L_CPI_TAR(+4)) + g3*L_GDP_GAP) + SHK_RS

     #policy persistence 
     g1 = 0.7      #g1 varies from 0 (no persistence) to 0.8 ("wait and see" policy)

     #policy reactiveness= the weight put on inflation by the policy-makers 
     g2 = 0.5      #g2 has no upper limit but must be always higher than 0 (the Taylor principle)

     #policy reactiveness= the weight put on the output gap by the policy-makers 
     g3 = 0.5      #g3 has no upper limit but must be always higher than 0

     #-------- 4. Uncovered Interest Rate Parity (UIP)
     #L_S = (1-e1)*L_S(+1) + e1*(L_S(-1) + 2/4*(D4L_CPI_TAR - ss_DLA_CPI_RW + DLA_Z_BAR)) + (- RS + RS_RW + PREM)/4 + SHK_L_S

     #the weight of the backward-looking component
     e1 = 0.4      #setting e1 equal to 0 reduces the equation to the simple UIP 

     #-------- 5. Speed of convergence of selected variables to their trend values.
     #Used for inflation target, trends, and foreign variables 

     #persistence of inflation target adjustment to the medium-term target (higher values mean slower adjustment)
     #D4L_CPI_TAR  = rho_D4L_CPI_TAR*D4L_CPI_TAR(-1) + (1-rho_D4L_CPI_TAR)*ss_D4L_CPI_TAR + SHK_D4L_CPI_TAR
     rho_D4L_CPI_TAR = 0.5 

     #persistence in convergence of trend variables to their steady-state levels
     #applies for   DLA_GDP_BAR, DLA_Z_BAR, RR_BAR and RR_RW_BAR
     #example
     #DLA_Z_BAR = rho_DLA_Z_BAR*DLA_Z_BAR(-1) + (1-rho_DLA_Z_BAR)*ss_DLA_Z_BAR + SHK_DLA_Z_BAR
     rho_DLA_Z_BAR   = 0.8
     rho_DLA_GDP_BAR = 0.8
     rho_RR_BAR      = 0.8
     rho_RR_RW_BAR   = 0.8

     #persistence in foreign output gap 
     #L_GDP_RW_GAP = rho_L_GDP_RW_GAP*L_GDP_RW_GAP(-1) + SHK_L_GDP_RW_GAP
     rho_L_GDP_RW_GAP = 0.8

     #persistence in foreign interest rates and inflation
     #RS_RW = rho_RS_RW*RS_RW(-1) + (1-rho_RS_RW)*(RR_BAR + DLA_CPI_RW) + SHK_RS_RW
     rho_RS_RW      = 0.8
     rho_DLA_CPI_RW = 0.8

//transition_equations
model;
	//// === Aggregate demand (the IS curve) ===
	L_GDP_GAP = b1*L_GDP_GAP{-1} - b2*MCI + b3*L_GDP_RW_GAP + SHK_L_GDP_GAP;

	//-- Real monetary conditions index
	MCI = b4*RR_GAP + (1-b4)*(- L_Z_GAP);

	//// === Inflation (the Phillips curve) ===
	DLA_CPI =  a1*DLA_CPI{-1} + (1-a1)*DLA_CPI{+1} + a2*RMC + SHK_DLA_CPI;

	//-- Real marginal cost
	RMC = a3*L_GDP_GAP + (1-a3)*L_Z_GAP;

	//- expected inflation
	E_DLA_CPI = DLA_CPI{+1};

	//// === Monetary policy reaction function (a forward-looking Taylor-type Rule) ===
	RS = g1*RS{-1} + (1-g1)*(RSNEUTRAL + g2*(D4L_CPI{+4} - D4L_CPI_TAR{+4}) + g3*L_GDP_GAP) + SHK_RS;

	//- Neutral nominal policy interest rate
	RSNEUTRAL = RR_BAR + D4L_CPI{+1};

	//// === Modified Uncovered Interest Rate Parity (UIP) condition ===
	L_S = (1-e1)*L_S{+1} + e1*(L_S{-1} + 2/4*(D4L_CPI_TAR - ss_DLA_CPI_RW + DLA_Z_BAR)) + (- RS + RS_RW + PREM)/4 + SHK_L_S;

	//// === Definitions ===

	//- Fisher equation (RIR)
	RR = RS - D4L_CPI{+1};

	//- Real exchange rate (RER)
	L_Z = L_S + L_CPI_RW - L_CPI;

	//- Long-term version of UIP (consistency of trends)
	DLA_Z_BAR{+1} = RR_BAR - RR_RW_BAR - PREM;

	//// === Identities ===
	DLA_GDP_BAR = 4*(L_GDP_BAR - L_GDP_BAR{-1});
	DLA_Z_BAR   = 4*(L_Z_BAR - L_Z_BAR{-1});
	DLA_Z       = 4*(L_Z - L_Z{-1});
	DLA_GDP     = 4*(L_GDP - L_GDP{-1});
	DLA_CPI     = 4*(L_CPI - L_CPI{-1});
	DLA_S       = 4*(L_S - L_S{-1});

	D4L_GDP     = L_GDP - L_GDP{-4};
	D4L_CPI     = L_CPI - L_CPI{-4};
	D4L_S       = L_S - L_S{-4};

	//// === Gaps ===
	RR_GAP    = RR - RR_BAR;
	L_Z_GAP   = L_Z - L_Z_BAR;
	L_GDP_GAP = L_GDP - L_GDP_BAR;

	//// === Trends ===
	D4L_CPI_TAR = rho_D4L_CPI_TAR*D4L_CPI_TAR{-1} + (1-rho_D4L_CPI_TAR)*ss_D4L_CPI_TAR + SHK_D4L_CPI_TAR;
	DLA_Z_BAR   = rho_DLA_Z_BAR*DLA_Z_BAR{-1} + (1-rho_DLA_Z_BAR)*ss_DLA_Z_BAR + SHK_DLA_Z_BAR;
	RR_BAR      = rho_RR_BAR*RR_BAR{-1} + (1-rho_RR_BAR)*ss_RR_BAR + SHK_RR_BAR;
	DLA_GDP_BAR = rho_DLA_GDP_BAR*DLA_GDP_BAR{-1} + (1-rho_DLA_GDP_BAR)*ss_DLA_GDP_BAR + SHK_DLA_GDP_BAR;

	//// === Foreign Sector Equations ===
	L_GDP_RW_GAP = rho_L_GDP_RW_GAP*L_GDP_RW_GAP{-1} + SHK_L_GDP_RW_GAP;
	DLA_CPI_RW   = rho_DLA_CPI_RW*DLA_CPI_RW{-1} + (1-rho_DLA_CPI_RW)*ss_DLA_CPI_RW + SHK_DLA_CPI_RW;
	RS_RW        = rho_RS_RW*RS_RW{-1} + (1-rho_RS_RW)*(RR_RW_BAR + DLA_CPI_RW) + SHK_RS_RW;
	RR_RW_BAR    = rho_RR_RW_BAR*RR_RW_BAR{-1} + (1-rho_RR_RW_BAR)*ss_RR_RW_BAR + SHK_RR_RW_BAR;
	RR_RW        = RS_RW - DLA_CPI_RW;
	RR_RW_GAP    = RR_RW - RR_RW_BAR;

	DLA_CPI_RW   = 4*(L_CPI_RW - L_CPI_RW{-1});


	//measurement_equations
	OBS_L_GDP = L_GDP;
	OBS_L_CPI = L_CPI;
	OBS_RS    = RS;
	OBS_L_S   = L_S;
	OBS_D4L_CPI_TAR = D4L_CPI_TAR;

	OBS_L_GDP_RW_GAP = L_GDP_RW_GAP;
	OBS_DLA_CPI_RW   = DLA_CPI_RW;
	OBS_RS_RW        = RS_RW;
end;

initval;
     # Starting values for endogenous variables
     L_GDP=           1332.1627
     L_GDP_BAR=       L_GDP
     L_GDP_GAP=       0
     DLA_GDP=         2.5
     D4L_GDP=         DLA_GDP
     DLA_GDP_BAR=     DLA_GDP
     MCI=             0
     L_CPI=           417.4363
     DLA_CPI=         2
     E_DLA_CPI=       2
     D4L_CPI=         2
     D4L_CPI_TAR=     2
     RMC=             0
     L_S=             352.2236
     DLA_S=           -1.5
     D4L_S=           DLA_S
     PREM=            1.25
     RS=              2.5
     RR=              0.5
     RR_BAR=          0.5
     RR_GAP=          0
     RSNEUTRAL=       2.5
     L_Z=             -21.8209
     L_Z_BAR=         L_Z
     L_Z_GAP=          0
     DLA_Z=            -1.5
     DLA_Z_BAR=        DLA_Z
     L_GDP_RW_GAP=     0
     RS_RW=            2.75
     RR_RW=            0.75
     RR_RW_BAR=        RR_RW
     RR_RW_GAP=        0
     L_CPI_RW=         43.3918
     DLA_CPI_RW=       2
end;

shocks;
     # Standard deviation of endogenous variables shocks
     var SHK_L_GDP_GAP; stderr 1.0;
     std_SHK_DLA_GDP_BAR  = 0.5
     std_SHK_DLA_CPI      = 0.75
     std_SHK_D4L_CPI_TAR  = 2
     std_SHK_L_S          = 3
     std_SHK_RS           = 1
     std_SHK_RR_BAR       = 0.5
     std_SHK_DLA_Z_BAR    = 0.5
     std_SHK_L_GDP_RW_GAP = 1
     std_SHK_RS_RW        = 1
     std_SHK_DLA_CPI_RW   = 2
     std_SHK_RR_RW_BAR    = 0.5
end;

steady(solve_algo=1,maxit=1000);

//model_diagnostics;
model_info;
//check;

stoch_simul(order=1,irf=20) L_GDP L_GDP_BAR L_GDP_GAP DLA_GDP;

varobs OBS_L_GDP OBS_L_CPI OBS_RS OBS_L_S OBS_D4L_CPI_TAR OBS_L_GDP_RW_GAP
     OBS_DLA_CPI_RW OBS_RS_RW;

calib_smoother(diffuse_filter, datafile='data/data.csv');


