// Declaration of variables 

// Model constants   
      
real An, Bn;   
  
// Temperature scaling variables   
     
real Tk, Trk, tN, Tki, Tamb;    
real Vt, Vtr, VtINV, VtrINV, VdtINV;   
real Vdt;
   
real UdeT, VDE_T, UdcT, VDC_T;   
real CJE_T, CJC_T, XP_T;    
real CJCscale, CJCscaleINV;   
   
real RE_T, RBV_T, RBC_T, RCC_T, RCV_T;     
real BF_T, BRI_T;     
   
real IS_T, IK_T, IBF_T, IBR_T, VEF_T, VER_T;    
   
real TAUE_T, TAUB_T, TEPI_T, TAUR_T;    
real BnT, DEG_T, Tk300;   
  
`ifdef SELFHEATING
real RTH_Tamb; 
`endif

`ifdef SUBSTRATE
real UdsT, VDS_T, CJS_T, ISS_T, IKS_T; 
`endif

// MULT - scaling variables    
   
real invMULT;   
real IS_TM, IK_TM, IBF_TM, IBR_TM, IHC_M;   
real CJE_TM, CJC_TM;     
   
real RE_TM, RBC_TM, RBV_TM, RCC_TM, RCV_TM, SCRCV_M;   
real KF_M, KFN_M;   

`ifdef SELFHEATING
real RTH_Tamb_M, CTH_M;
`endif
  
`ifdef SUBSTRATE
real ISS_TM, IKS_TM, CJS_TM;
`endif


// Epilayer model variables   
   
real K0, Kw, pW, Ec, Ic1c2;    
real Vqs_th, Vqs, Iqs;    
real alpha, vyi, yi, xi_w, xi_w1;   
real gp0, gp02, p0star, eVb2c2star;    
real B1, B2, Vxi0, Vch, Icap, pav;     
  
// Effective emitter and collector junction bias variables    
   
real Vfe, Vje, Vte;     
real Vjunc, bjc, Vfc, Vjc, fI, Vcv, Vtc;   
  
// Transfer current variables   
   
real If0, f1, f2, n0, nB;   
real q0I, q1I, qBI, Ir, If, In;   
  
// Base and substrate current(s) variables    
   
real Xext1;
real Ib1, Ib1_s, Ib2, Ib3;                 
real Ibf0, Iex, Isub;                         
real g1, g2, pWex, nBex;   
real Xg1, XnBex, XIMex, XIMsub, Vex, VBex, Fex, XIex;   

`ifdef SUBSTRATE
real XIsub, Isf;
`endif
  
// Distributed base effects variables    
   
real q0Q, q1Q, qBQ, Rb2, Ib1b2;   
real dVteVb2e1, dVteVje, dVjeVb2e1;   
real dQteVb2e1, dQbeVb2e1, dQeVb2e1;  
real dn0Vb2e1;  
  
// Weak-avalanche current variables    
   
real dEdx0, xd, Weff, Wd, Eav, E0, Em, SHw, Efi, Ew;    
real lambda, Gem, Gmax, Iavl;   
real Icap_IHC;  
  
`ifdef SELFHEATING
real Vb2c2star, power;   
`endif
 
// Charges and capacitances variables   
   
real Qte, Vje_s, Qte_s;     
real Qtc;    
real Qb0, Qbe, Qbc, Qb1b2;    
real Qbe_qs, Qbc_qs;
real Vjcex, Vtexv, Qtex, XVjcex, XVtexv, XQtex;      

`ifdef SUBSTRATE
real Vfs, Vjs, Qts;      
`endif

real Qe0, Qe;    
real Qepi0, Qepi, Xg2, XpWex, XQex;   
real Qex;  
real CBEO_M, CBCO_M;  
  
// Biases and exponential terms variables   
  
real Vb2c1, Vb2c2, Vb2e1, Vb1e1, Vb1b2, Vb1c1, Vc1c2;   
real Vsc1, Vee1, Vbb1, Vbc1, Vcc1, Vbe, Vbc;    
real eVb2c2, eVb2e1, eVb1e1, eVb1b2, eVb1c1, eVbc1, eVsc1;    
real eVb1c1VDC, eVb2c2VDC, eVbc1VDC, eVb2c1VDC;  
  
// Help variables  
  
real dxa, sqr_arg;  
real eps2, x2;  
real alpha1, vdif, Ic1c2_Iqs, gp0_help;  
real EmEav_Em, Vb2e1Vfe, termE, termC;  
real Vex_bias;  
real eps_VDC, a_VDE, a_VDC; 

real expl, tmpExp, tmpV;


`ifdef  SUBSTRATE
real a_VDS;  
`endif
 
// Noise variables 
 
real common; 
real powerREC, powerRBC, powerRCC, powerRBV; 
real powerCCS;  
real powerFBCS, powerFBC1fB1, exponentFBC1fB2, powerFBC1fB2; 
real powerEBSCS, powerEBSC1f; 
real powerRBCS, powerRBC1f; 
real powerExCS, powerExCSMOD, powerExC1f, powerExC1fMOD; 

`ifdef SUBSTRATE
real powerSubsCS_B1S, powerSubsCS_BS; 
`endif

//real twoqIavl, powerCCS_A, powerFBCS_A, powerAVL_B2C2; 
real twoqIavl, cor_exp_1, cor_exp_2, powerCCS_A; 
 
