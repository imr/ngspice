/***********************************************************************
 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2003 STARC

 VERSION : HiSIM 1.2.0
 FILE : hisim.h of HiSIM 1.2.0

 April 9, 2003 : released by STARC Physical Design Group
***********************************************************************/


#ifndef _HiSIM_H
#define _HiSIM_H

/*#define HiSIM_TIME   0*/

/* return value */
#ifndef OK
#define HiSIM_OK        0
#define HiSIM_ERROR     1
#else
#define HiSIM_OK        OK
#define HiSIM_ERROR     E_PANIC
#endif

/* MOS type */
#ifndef NMOS
#define NMOS     1
#define PMOS    -1
#endif

/* device working mode */
#ifndef CMI_NORMAL_MODE
#define HiSIM_NORMAL_MODE    1
#define HiSIM_REVERSE_MODE  -1
#else
#define HiSIM_NORMAL_MODE  CMI_NORMAL_MODE
#define HiSIM_REVERSE_MODE CMI_REVERSE_MODE
#endif

/* others */
#ifndef NULL
#define NULL            0
#endif

#define HiSIM_FALSE     0
#define HiSIM_TRUE      1


/*-------------------------------------------------------------------*
* Structure for inputs of hisim.
*----------------*/
typedef struct sHiSIM_input {

  /* Flags that must be set in parent routines. */
  int     type ;  /* MOS type (1:NMOS, -1:PMOS) */
  int     mode ;  /* MOS mode (1:normal, -1:reverse) */
  int     qflag ; /* qflag (toggle for charge calc. Unused!) */
  int     has_prv ; /* 1 if previous values are held. */
  
  /* Bias conditions */
  double  vbs ;   /* Vbs [V] */
  double  vds ;   /* Vds [V] */
  double  vgs ;   /* Vgs [V] */

  /* frequency [Hz] */
  double  freq ;

  /* version */
  double  version ;
  
  /* Control options that can be set in a model parameter set. */
  int     info ;   /* information level (for debug, etc.) */
  int     corsrd ; /* solve equations accounting Rs and Rd. */
  int     coiprv ; /* use ids_prv as initial guess of Ids */
  int     copprv ; /* use ps{0/l}_prv as initial guess of Ps{0/l} */
  int     cocgso ; /* calculate cgso */
  int     cocgdo ; /* calculate cgdo */ 
  int     cocgbo ; /* calculate cgbo */
  int     coadov ; /* add overlap to intrisic */
  int     coxx08 ; /* spare */
  int     coxx09 ; /* spare */
  int     coisub ; /* calculate isub */
  int     coiigs ; /* calculate igate */
  int     cogidl ; /* calculate igidl */
  int     cogisl ; /* calculate igisl */
  int     coovlp ; /* calculate overlap charge */
  int     conois ; /* calculate 1/f noise */
  int     coisti ; /* calculate STI */
  int     cosmbi ; /* bias smoothing in dVth */

   /* Previous values that may be used as initial guesses */
  /* - derivatives are ones w.r.t. internal biases. */
  double  vbsc_prv ;
  double  vdsc_prv ;
  double  vgsc_prv ;
  double  ps0_prv ;
  double  ps0_dvbs_prv ;
  double  ps0_dvds_prv ;
  double  ps0_dvgs_prv ;
  double  pds_prv ;
  double  pds_dvbs_prv ;
  double  pds_dvds_prv ;
  double  pds_dvgs_prv ;
  double  ids_prv ;
  double  ids_dvbs_prv ;
  double  ids_dvds_prv ;
  double  ids_dvgs_prv ;

  /* for noise calc. */
  double  nfc ;

  /* Device instances */
  double  xl ;    /* channel length [m] (L=xl-xld) */
  double  xw ;    /* channel width [m] (W=xw-xwd) */
  double  ad ;    /* area of drain diffusion [m^2] */
  double  as ;    /* area of source diffusion [m^2] */
  double  pd ;    /* perimeter of drain junction [m] */
  double  ps ;    /* perimeter of source junction [m] */
  double  nrd ;   /* equivalent num of squares of drain [-] (unused) */
  double  nrs ;   /* equivalent num of squares of source [-] (unused) */
  double  temp ;  /* lattice temperature [K] */
  double  m ;     /* multiplier */
  
  /* Model parameters */
  double  vmax ;  /* saturation velocity [cm/s] */
  double  bgtmp1 ;    /* first order temp. coeff. for band gap [V/K] */
  double  bgtmp2 ;    /* second order temp. coeff. for band gap [V/K^2] */
  double  tox ;   /* oxide thickness [m] */
  double  xld ;    /* lateral diffusion of S/D under the gate [m] */
  double  xwd ;    /* lateral diffusion along the width dir. [m] */
  double  xj ;    /*  HiSIM1.0 [m] */
  double  xqy ;   /*  HiSIM1.1 [m] */
  /*--*/
  double  rd ; /* drain contact resistance  [ohm m] */
  double  rs ; /* source contact resistance [ohm m] */
/**/
  double  vfbc ;  /* constant part of Vfb [V] */
  double  nsubc ;    /* constant part of Nsub [1/cm^3] */
  double  parl1 ; /* factor for L dependency of dVthSC [-] */
  double  parl2 ; /* under diffusion [m]  */
  double  lp ; /* length of pocket potential [m] */
  double  nsubp ; /* [1/cm^3] */
  double  scp1 ;  /* parameter for pocket [1/V] */
  double  scp2 ;  /* parameter for pocket [1/V^2] */
  double  scp3 ;  /* parameter for pocket [m/V^2] */
  double  sc1 ;   /* parameter for SCE [1/V] */
  double  sc2 ;   /* parameter for SCE [1/V^2] */
  double  sc3 ;   /* parameter for SCE [m/V^2] */
  double  pgd1 ;  /* parameter for gate-poly depletion [V] */ 
  double  pgd2 ;  /* parameter for gate-poly depletion [V] */
  double  pgd3 ;  /* parameter for gate-poly depletion [-] */
/**/
  double  ndep ;  /* coeff. of Qbm for Eeff [-]  */
  double  ninv ;  /* coeff. of Qnm for Eeff [-]  */
  double  ninvd ;  /* parameter for universal mobility [1/V] */
  double  muecb0 ; /* const. part of coulomb scattering [cm^2/Vs] */
  double  muecb1 ;  /* coeff. for coulomb scattering [cm^2/Vs] */
  double  mueph0 ;  /* power of Eeff for phonon scattering [-] */
  double  mueph1 ;
  double  mueph2 ;
  double  w0 ;
  double  muesr0 ;  /* power of Eeff for S.R. scattering [-] */
  double  muesr1 ;  /* coeff. for S.R. scattering [-] */
  double  muetmp ;  /* parameter for mobility [-] */
  double  bb ;    /* empirical mobility model coefficient [-] */
/**/
  double  sub1 ;  /* parameter for Isub [1/V] */
  double  sub2 ;  /* parameter for Isub [V] */
  double  sub3 ;  /* parameter for Isub [-] */
/**/
  double  wvthsc ;  /* parameter for STI [-] HiSIM1.1 */
  double  nsti ;    /* parameter for STI [1/cm^3] HiSIM1.1 */
  double  wsti ;    /* parameter for STI [m] HiSIM1.1 */
/**/
  double  cgso ;  /* G-S overlap capacitance per unit W [F/m] */
  double  cgdo ;  /* G-D overlap capacitance per unit W [F/m] */
  double  cgbo ;  /* G-B overlap capacitance per unit L [F/m] */
/**/
  double  tpoly ; /* hight of poly gate [m] */
/**/
  double  js0 ;   /* Saturation current density [A/m^2] */
  double  js0sw ; /* Side wall saturation current density [A/m] */
  double  nj ;    /* Emission coefficient */
  double  njsw ;    /* Emission coefficient (sidewall) */  
  double  xti ;   /* Junction current temparature exponent coefficient */
  double  cj ;    /* Bottom junction capacitance per unit area 
		     at zero bias [F/m^2]*/
  double  cjsw ;  /* Source/drain sidewall junction capacitance grading 
		     coefficient per unit length at zero bias [F/m] */
  double  cjswg ; /* Source/drain gate sidewall junction capacitance 
		     per unit length at zero bias [F/m] */
  double  mj ;    /* Bottom junction capacitance grading coefficient  */
  double  mjsw ;  /* Source/drain sidewall junction capacitance grading 
		     coefficient */
  double  mjswg ; /* Source/drain gate sidewall junction capacitance grading 
		     coefficient */
  double  pb ;    /* Bottom junction build-in potential  [V] */
  double  pbsw ;  /* Source/drain sidewall junction build-in potential [V] */
  double  pbswg ; /* Source/drain gate sidewall junction build-in potential [V] */
  double  xpolyd ; /* parameter for Cov [m] */
/**/
  double  clm1 ;  /* parameter for CLM [-] */
  double  clm2 ;  /* parameter for CLM [1/m] */
  double  clm3 ;  /* parameter for CLM [-] */
/**/
  double  rpock1 ; /* parameter for Ids [V] */
  double  rpock2 ; /* parameter for Ids [V^2 sqrt(m)/A] */
  double  rpocp1 ; /* parameter for Ids [-] HiSIM1.1 */
  double  rpocp2 ; /* parameter for Ids [-] HiSIM1.1 */

/**/
  double  vover ;  /* parameter for overshoot [m^{voverp}]*/
  double  voverp ;  /* parameter for overshoot [-] */
  double  wfc ;  /* parameter for narrow channel effect [m*F/(cm^2)]*/
  double  qme1 ;  /* parameter for quantum effect [mV]*/
  double  qme2 ;  /* parameter for quantum effect [V]*/
  double  qme3 ;  /* parameter for quantum effect [m]*/
  double  gidl1 ;  /* parameter for GIDL [?] */
  double  gidl2 ;  /* parameter for GIDL [?] */
  double  gidl3 ;  /* parameter for GIDL [?] */
  double  gleak1 ;  /* parameter for gate current [?] */
  double  gleak2 ;  /* parameter for gate current [?] */
  double  gleak3 ;  /* parameter for gate current [?] */
/**/
  double  vzadd0 ;  /* Vzadd at Vds=0  [V] */
  double  pzadd0 ;  /* Pzadd at Vds=0  [V] */

  double  nftrp ;
  double  nfalp ;
  double  cit ;

  double  gmin ; /* gmin = minimum conductance of SPICE3 */
 /**/
  double glpart1 ; /* partition of gate leackage current */
  double glpart2 ;
  double kappa ;  /* */
  double xdiffd ; /* */
  double pthrou ;  /* */
  double vdiffj ; /* */

} HiSIM_input ;

/*-------------------------------------------------------------------*
* structure for outputs of hisim.
*----------------*/
typedef struct sHiSIM_output {
  double ids ;    /* channel current [A] */
  double gds ;    /* channel conductance (dIds/dVds) [S] */
  double gm ;     /* trans conductance (dIds/dVgs) [S] */
  double gmbs ;   /* substrate trans conductance (dIds/dVbs) [S] */
/**/
  double gd ;     /* parasitic drain conductance [S] */
  double gs ;     /* parasitic source conductance [S] */
/**/
  double cgso ;   /* G-S overlap capacitance [F] */
  double cgdo ;   /* G-D overlap capacitance [F] */
  double cgbo ;   /* G-B overlap capacitance [F] */
/**/
  double von ;    /* Vth [V] */
  double vdsat ;  /* saturation voltage [V] */
/**/
  double ibs ;    /* substrate source leakage current [A] */
  double ibd ;    /* substrate drain leakage current [A] */
  double gbs ;    /* substrate source conductance [S] */
  double gbd ;    /* substrate drain conductance [S] */
/**/
  double capbs ;  /* substrate source capacitance [F] */
  double capbd ;  /* substrate drain capacitance [F] */
  double qbs ;    /* substrate source charge [C] */
  double qbd ;    /* substrate drain charge [C] */
/**/
  double isub ;   /* substrate impact ionization current [A] */
  double gbgs ;   /* substrate trans conductance (dIsub/dVgs) [S] */
  double gbds ;   /* substrate trans conductance (dIsub/dVds) [S] */
  double gbbs ;   /* substrate trans conductance (dIsub/dVbs) [S] */
/**/
  double qg ;     /* intrinsic gate charge [C] */
  double qd ;     /* intrinsic drain charge [C] */
  double qs ;     /* intrinsic source charge [C] */
/**/
  double cggb ;   /* intrinsic gate capacitance w.r.t. gate [F] */
  double cgdb ;   /* intrinsic gate capacitance w.r.t. drain [F] */
  double cgsb ;   /* intrinsic gate capacitance w.r.t. source [F] */
  double cbgb ;   /* intrinsic bulk capacitance w.r.t. gate [F] */
  double cbdb ;   /* intrinsic bulk capacitance w.r.t. drain [F] */
  double cbsb ;   /* intrinsic bulk capacitance w.r.t. source [F] */
  double cdgb ;   /* intrinsic drain capacitance w.r.t. gate [F] */
  double cddb ;   /* intrinsic drain capacitance w.r.t. drain [F] */
  double cdsb ;   /* intrinsic drain capacitance w.r.t. source [F] */
/**/
  double igate ;  /* gate current due to tunneling [A] */
  double gggs ;   /* trans conductance (dIgate/dVgs) [S] */
  double ggds ;   /* trans conductance (dIgate/dVds) [S] */
  double ggbs ;   /* trans conductance (dIgate/dVbs) [S] */
/**/
  double igateb ;  /* gate current due to tunneling [A] (G->B)*/
  double ggbgs ;   /* trans conductance (dIgateb/dVgs) [S] */
  double ggbds ;   /* trans conductance (dIgateb/dVds) [S] */
  double ggbbs ;   /* trans conductance (dIgateb/dVbs) [S] */
/**/
  double igates ;  /* gate current due to tunneling [A] (G->S)*/
  double ggsgs ;   /* trans conductance (dIgates/dVgs) [S] */
  double ggsds ;   /* trans conductance (dIgates/dVds) [S] */
  double ggsbs ;   /* trans conductance (dIgates/dVbs) [S] */
/**/
  double igated ;  /* gate current due to tunneling [A] (G->D)*/
  double ggdgs ;   /* trans conductance (dIgated/dVgs) [S] */
  double ggdds ;   /* trans conductance (dIgated/dVds) [S] */
  double ggdbs ;   /* trans conductance (dIgated/dVbs) [S] */
/**/
  double igidl ;    /* gate induced drain leakage [A] */
  double ggidlgs ;   /* trans conductance (dIgidl/dVgs) [S] */
  double ggidlds ;   /* trans conductance (dIgidl/dVds) [S] */
  double ggidlbs ;   /* trans conductance (dIgidl/dVbs) [S] */
/**/
  double igisl ;    /* gate induced source leakage [A] */
  double ggislgd ;   /* trans conductance (dIgisl/dVgs) [S] */
  double ggislsd ;   /* trans conductance (dIgisl/dVds) [S] */
  double ggislbd ;   /* trans conductance (dIgisl/dVbs) [S] */
/**/
  double nois_idsfl ;
  double nois_ird ;
  double nois_irs ;
  double nois_idsth ;
/**/
  /* Outputs that may be used as initial guesses in the next calling */
  double  vbsc ;
  double  vdsc ;
  double  vgsc ;
  double  ps0 ;
  double  ps0_dvbs ;
  double  ps0_dvds ;
  double  ps0_dvgs ;
  double  pds ;
  double  pds_dvbs ;
  double  pds_dvds ;
  double  pds_dvgs ;
  double  ids_dvbs ;
  double  ids_dvds ;
  double  ids_dvgs ;

  /* for noise calc. */
  double  nf ;

  /* mobility added by K.M. */
  double  mu ;

  /* intrinsic charges */
  double qg_int ;
  double qd_int ;
  double qs_int ;
  double qb_int ;

} HiSIM_output ;

/*-------------------------------------------------------------------*
* structure for messengers to/from hisim.
*----------------*/
typedef struct sHiSIM_messenger {
  int     ims[20] ;
  double  dms[50] ;
  /* Control options for alpha versions */
  int     opt_ntn ;
  int     opt_psl ;
  int     opt_rsc ;
  int     opt_sce ;
  int     opt_mbl ;
  int     opt_dp0 ;
  int     opt_inv ;
  int     opt_bas ;
  int     opt_01 ;
  int     opt_02 ;
  int     opt_03 ;
  int     opt_04 ;
  int     opt_05 ;
} HiSIM_messenger ;
/* note: -----------------------------------
* if HiSIM_TEST is defined.
*   ims[ 1 ] :(in)  =1: output physical capacitances instead of ones 
*                       referenced to bulk.
*   ims[11-12] , dms[11-39] :(ot) additional outputs.
* if CMI_OK is defined.
*   dms[ 1 ] :(in) timepoint. 
-------------------------------------------*/


extern int HSM1evaluate102( 
 HiSIM_input     sIN,
 HiSIM_output    *pOT,
 HiSIM_messenger *pMS 
 ) ;
extern int HSM1evaluate112
(
 HiSIM_input     sIN,
 HiSIM_output    *pOT,
 HiSIM_messenger *pMS 
 ) ;
extern int HSM1evaluate120
(
 HiSIM_input     sIN,
 HiSIM_output    *pOT,
 HiSIM_messenger *pMS 
 ) ;

#endif /* _HiSIM_H */
