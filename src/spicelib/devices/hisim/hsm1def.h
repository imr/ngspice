/***********************************************************************
 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2003 STARC

 VERSION : HiSIM 1.2.0
 FILE : hsm1def.h of HiSIM 1.2.0

 April 9, 2003 : released by STARC Physical Design Group
***********************************************************************/

#ifndef HSM1
#define HSM1

#include "ifsim.h"
#include "gendefs.h"
#include "cktdefs.h"
#include "complex.h"
#include "noisedef.h"

/* declarations for HiSIM1 MOSFETs */

/* information needed for each instance */
typedef struct sHSM1instance {
  struct sHSM1model *HSM1modPtr;           /* pointer to model */
  struct sHSM1instance *HSM1nextInstance;  /* pointer to next instance of 
                                              current model*/
  IFuid HSM1name; /* pointer to character string naming this instance */
/*DW next is additional in spice */
  int HSM1owner;  /* number of owner process */
  int HSM1states; /* index into state table for this device */

  int HSM1dNode;      /* number of the drain node of the mosfet */
  int HSM1gNode;      /* number of the gate node of the mosfet */
  int HSM1sNode;      /* number of the source node of the mosfet */
  int HSM1bNode;      /* number of the bulk node of the mosfet */
  int HSM1dNodePrime; /* number od the inner drain node */
  int HSM1sNodePrime; /* number od the inner source node */

  char HSM1_called[4]; /* string to check the first call */

  /* previous values to evaluate initial guess */
  double HSM1_vbsc_prv;
  double HSM1_vdsc_prv;
  double HSM1_vgsc_prv;
  double HSM1_ps0_prv;
  double HSM1_ps0_dvbs_prv;
  double HSM1_ps0_dvds_prv;
  double HSM1_ps0_dvgs_prv;
  double HSM1_pds_prv;
  double HSM1_pds_dvbs_prv;
  double HSM1_pds_dvds_prv;
  double HSM1_pds_dvgs_prv;
  double HSM1_ids_prv;
  double HSM1_ids_dvbs_prv;
  double HSM1_ids_dvds_prv;
  double HSM1_ids_dvgs_prv;
  
  double HSM1_nfc; /* for noise calc. */

  /* instance */
  double HSM1_l;    /* the length of the channel region */
  double HSM1_w;    /* the width of the channel region */
  double HSM1_m;    /* Parallel multiplier */
  double HSM1_ad;   /* the area of the drain diffusion */
  double HSM1_as;   /* the area of the source diffusion */
  double HSM1_pd;   /* perimeter of drain junction [m] */
  double HSM1_ps;   /* perimeter of source junction [m] */
  double HSM1_nrd;  /* equivalent num of squares of drain [-] (unused) */
  double HSM1_nrs;  /* equivalent num of squares of source [-] (unused) */
  double HSM1_temp; /* lattice temperature [K] */
  double HSM1_dtemp;

  /* added by K.M. */
  double HSM1_weff; /* the effective width of the channel region */
  double HSM1_leff; /* the effective length of the channel region */

  /* output */
  int    HSM1_capop;
  double HSM1_gd;
  double HSM1_gs;
  double HSM1_cgso;
  double HSM1_cgdo;
  double HSM1_cgbo;
  double HSM1_von; /* vth */
  double HSM1_vdsat;
  double HSM1_ids; /* cdrain, HSM1_cd */
  double HSM1_gds;
  double HSM1_gm;
  double HSM1_gmbs;
  double HSM1_ibs; /* HSM1_cbs */
  double HSM1_ibd; /* HSM1_cbd */
  double HSM1_gbs;
  double HSM1_gbd;
  double HSM1_capbs;
  double HSM1_capbd;
  /*
  double HSM1_qbs;
  double HSM1_qbd;
  */
  double HSM1_capgs;
  double HSM1_capgd;
  double HSM1_capgb;
  double HSM1_isub; /* HSM1_csub */
  double HSM1_gbgs;
  double HSM1_gbds;
  double HSM1_gbbs;
  double HSM1_qg;
  double HSM1_qd;
  /*  double HSM1_qs; */
  double HSM1_qb;  /* bulk charge qb = -(qg + qd + qs) */
  double HSM1_cggb;
  double HSM1_cgdb;
  double HSM1_cgsb;
  double HSM1_cbgb;
  double HSM1_cbdb;
  double HSM1_cbsb;
  double HSM1_cdgb;
  double HSM1_cddb;
  double HSM1_cdsb;
  /* no use in SPICE3f5
  double HSM1_nois_irs;
  double HSM1_nois_ird;
  double HSM1_nois_idsth;
  double HSM1_nois_idsfl;
  double HSM1_freq;
  */

  double HSM1_mu; /* mobility */
  double HSM1_igidl; /* gate induced drain leakage */
  double HSM1_gigidlgs;
  double HSM1_gigidlds;
  double HSM1_gigidlbs;
  double HSM1_igisl; /* gate induced source leakage */
  double HSM1_gigislgd;
  double HSM1_gigislsd;
  double HSM1_gigislbd;
  double HSM1_igb; /* gate tunneling current (gate to bulk) */
  double HSM1_gigbg;
  double HSM1_gigbd;
  double HSM1_gigbb;
  double HSM1_gigbs;
  double HSM1_igs; /* gate tunneling current (gate to source) */
  double HSM1_gigsg;
  double HSM1_gigsd;
  double HSM1_gigsb;
  double HSM1_gigss;
  double HSM1_igd; /* gate tunneling current (gate to drain) */
  double HSM1_gigdg;
  double HSM1_gigdd;
  double HSM1_gigdb;
  double HSM1_gigds;
  double HSM1_qg_int ;
  double HSM1_qd_int ;
  double HSM1_qs_int ;
  double HSM1_qb_int ;

  /* no use in SPICE3f5
      double HSM1drainSquares;       the length of the drain in squares
      double HSM1sourceSquares;      the length of the source in squares */
  double HSM1sourceConductance; /* cond. of source (or 0): set in setup */
  double HSM1drainConductance;  /* cond. of drain (or 0): set in setup */

  double HSM1_icVBS; /* initial condition B-S voltage */
  double HSM1_icVDS; /* initial condition D-S voltage */
  double HSM1_icVGS; /* initial condition G-S voltage */
  int HSM1_off;      /* non-zero to indicate device is off for dc analysis */
  int HSM1_mode;     /* device mode : 1 = normal, -1 = inverse */

  unsigned HSM1_l_Given :1;
  unsigned HSM1_w_Given :1;
  unsigned HSM1_m_Given :1;
  unsigned HSM1_ad_Given :1;
  unsigned HSM1_as_Given    :1;
  /*  unsigned HSM1drainSquaresGiven  :1;
      unsigned HSM1sourceSquaresGiven :1;*/
  unsigned HSM1_pd_Given    :1;
  unsigned HSM1_ps_Given   :1;
  unsigned HSM1_nrd_Given  :1;
  unsigned HSM1_nrs_Given  :1;
  unsigned HSM1_temp_Given  :1;
  unsigned HSM1_dtemp_Given  :1;
  unsigned HSM1dNodePrimeSet  :1;
  unsigned HSM1sNodePrimeSet  :1;
  unsigned HSM1_icVBS_Given :1;
  unsigned HSM1_icVDS_Given :1;
  unsigned HSM1_icVGS_Given :1;
  
  /* pointer to sparse matrix */
  double *HSM1DdPtr;      /* pointer to sparse matrix element at 
			     (Drain node,drain node) */
  double *HSM1GgPtr;      /* pointer to sparse matrix element at
			     (gate node,gate node) */
  double *HSM1SsPtr;      /* pointer to sparse matrix element at
			     (source node,source node) */
  double *HSM1BbPtr;      /* pointer to sparse matrix element at
			     (bulk node,bulk node) */
  double *HSM1DPdpPtr;    /* pointer to sparse matrix element at
			     (drain prime node,drain prime node) */
  double *HSM1SPspPtr;    /* pointer to sparse matrix element at
			     (source prime node,source prime node) */
  double *HSM1DdpPtr;     /* pointer to sparse matrix element at
			     (drain node,drain prime node) */
  double *HSM1GbPtr;      /* pointer to sparse matrix element at
			     (gate node,bulk node) */
  double *HSM1GdpPtr;     /* pointer to sparse matrix element at
			     (gate node,drain prime node) */
  double *HSM1GspPtr;     /* pointer to sparse matrix element at
			     (gate node,source prime node) */
  double *HSM1SspPtr;     /* pointer to sparse matrix element at
			     (source node,source prime node) */
  double *HSM1BdpPtr;     /* pointer to sparse matrix element at
			     (bulk node,drain prime node) */
  double *HSM1BspPtr;     /* pointer to sparse matrix element at
			     (bulk node,source prime node) */
  double *HSM1DPspPtr;    /* pointer to sparse matrix element at
			     (drain prime node,source prime node) */
  double *HSM1DPdPtr;     /* pointer to sparse matrix element at
			     (drain prime node,drain node) */
  double *HSM1BgPtr;      /* pointer to sparse matrix element at
			     (bulk node,gate node) */
  double *HSM1DPgPtr;     /* pointer to sparse matrix element at
			     (drain prime node,gate node) */
  double *HSM1SPgPtr;     /* pointer to sparse matrix element at
			     (source prime node,gate node) */
  double *HSM1SPsPtr;     /* pointer to sparse matrix element at
			     (source prime node,source node) */
  double *HSM1DPbPtr;     /* pointer to sparse matrix element at
			     (drain prime node,bulk node) */
  double *HSM1SPbPtr;     /* pointer to sparse matrix element at
			     (source prime node,bulk node) */
  double *HSM1SPdpPtr;    /* pointer to sparse matrix element at
			     (source prime node,drain prime node) */

  /* common state values in hisim1 module */
#define HSM1vbd HSM1states+ 0
#define HSM1vbs HSM1states+ 1
#define HSM1vgs HSM1states+ 2
#define HSM1vds HSM1states+ 3

#define HSM1qb  HSM1states+ 4
#define HSM1cqb HSM1states+ 5
#define HSM1qg  HSM1states+ 6
#define HSM1cqg HSM1states+ 7
#define HSM1qd  HSM1states+ 8
#define HSM1cqd HSM1states+ 9 

#define HSM1qbs HSM1states+ 10
#define HSM1qbd HSM1states+ 11

#define HSM1numStates 12

  /* indices to the array of HiSIM1 NOISE SOURCES (the same as BSIM3) */
#define HSM1RDNOIZ       0
#define HSM1RSNOIZ       1
#define HSM1IDNOIZ       2
#define HSM1FLNOIZ       3
#define HSM1TOTNOIZ      4

#define HSM1NSRCS        5  /* the number of HiSIM1 MOSFET noise sources */

#ifndef NONOISE
  double HSM1nVar[NSTATVARS][HSM1NSRCS];
#else /* NONOISE */
  double **HSM1nVar;
#endif /* NONOISE */

} HSM1instance ;


/* per model data */

typedef struct sHiSIM1model {       	/* model structure for a resistor */
  int HSM1modType;    		/* type index of this device type */
  struct sHiSIM1model *HSM1nextModel; /* pointer to next possible model 
					 in linked list */
  HSM1instance * HSM1instances;	/* pointer to list of instances 
				   that have this model */
  IFuid HSM1modName;       	/* pointer to the name of this model */
  int HSM1_type;      		/* device type: 1 = nmos,  -1 = pmos */
  int HSM1_level;               /* level */
  int HSM1_info;                /* information */
  int HSM1_noise;               /* noise model selecter see hsm1noi.c */
  int HSM1_version;             /* model version 101/111/120 */
  int HSM1_show;                /* show physical value 1, 2, ... , 11 */

  /* flags for initial guess */
  int HSM1_corsrd ;
  int HSM1_coiprv ;
  int HSM1_copprv ;
  int HSM1_cocgso ;
  int HSM1_cocgdo ;
  int HSM1_cocgbo ;
  int HSM1_coadov ;
  int HSM1_coxx08 ;
  int HSM1_coxx09 ;
  int HSM1_coisub ;
  int HSM1_coiigs ;
  int HSM1_cogidl ;
  int HSM1_cogisl ;
  int HSM1_coovlp ;
  int HSM1_conois ;
  int HSM1_coisti ; /* HiSIM1.1 */
  int HSM1_cosmbi ; /* HiSIM1.2 */

  /* HiSIM original */
  double HSM1_vmax ;
  double HSM1_bgtmp1 ;
  double HSM1_bgtmp2 ;
  double HSM1_tox ;
  double HSM1_xld ;
  double HSM1_xwd ;
  double HSM1_xj ;   /* HiSIM1.0 */
  double HSM1_xqy ;  /* HiSIM1.1 */
  double HSM1_rs;     /* source contact resistance */
  double HSM1_rd;     /* drain contact resistance */
  double HSM1_vfbc ;
  double HSM1_nsubc ;
  double HSM1_parl1 ;
  double HSM1_parl2 ;
  double HSM1_lp ;
  double HSM1_nsubp ;
  double HSM1_scp1 ;
  double HSM1_scp2 ;
  double HSM1_scp3 ;
  double HSM1_sc1 ;
  double HSM1_sc2 ;
  double HSM1_sc3 ;
  double HSM1_pgd1 ;
  double HSM1_pgd2 ;
  double HSM1_pgd3 ;
  double HSM1_ndep ;
  double HSM1_ninv ;
  double HSM1_ninvd ;
  double HSM1_muecb0 ;
  double HSM1_muecb1 ;
  double HSM1_mueph1 ;
  double HSM1_mueph0 ;
  double HSM1_mueph2 ;
  double HSM1_w0 ;
  double HSM1_muesr1 ;
  double HSM1_muesr0 ;
  double HSM1_bb ;
  double HSM1_sub1 ;
  double HSM1_sub2 ;
  double HSM1_sub3 ;
  double HSM1_wvthsc ; /* HiSIM1.1 */
  double HSM1_nsti ;   /* HiSIM1.1 */
  double HSM1_wsti ;   /* HiSIM1.1 */
  double HSM1_cgso ;
  double HSM1_cgdo ;
  double HSM1_cgbo ;
  double HSM1_tpoly ;
  double HSM1_js0 ;
  double HSM1_js0sw ;
  double HSM1_nj ;
  double HSM1_njsw ;
  double HSM1_xti ;
  double HSM1_cj ;
  double HSM1_cjsw ;
  double HSM1_cjswg ;
  double HSM1_mj ;
  double HSM1_mjsw ;
  double HSM1_mjswg ;
  double HSM1_pb ;
  double HSM1_pbsw ;
  double HSM1_pbswg ;
  double HSM1_xpolyd ;
  double HSM1_clm1 ;
  double HSM1_clm2 ;
  double HSM1_clm3 ;
  double HSM1_muetmp ;
  double HSM1_rpock1 ;
  double HSM1_rpock2 ;
  double HSM1_rpocp1 ; /* HiSIM 1.1 */
  double HSM1_rpocp2 ; /* HiSIM 1.1 */
  double HSM1_vover ;
  double HSM1_voverp ;
  double HSM1_wfc ;
  double HSM1_qme1 ;
  double HSM1_qme2 ;
  double HSM1_qme3 ;
  double HSM1_gidl1 ;
  double HSM1_gidl2 ;
  double HSM1_gidl3 ;
  double HSM1_gleak1 ;
  double HSM1_gleak2 ;
  double HSM1_gleak3 ;
  double HSM1_vzadd0 ;
  double HSM1_pzadd0 ;
  double HSM1_nftrp ;
  double HSM1_nfalp ;
  double HSM1_cit ;
  double HSM1_glpart1 ; /* HiSIM1.2 */
  double HSM1_glpart2 ; /* HiSIM1.2 */
  double HSM1_kappa ;   /* HiSIM1.2 */
  double HSM1_xdiffd ;  /* HiSIM1.2 */
  double HSM1_pthrou ;   /* HiSIM1.2 */
  double HSM1_vdiffj ;  /* HiSIM1.2 */

  /* for flicker noise of SPICE3 added by K.M. */
  double HSM1_ef;
  double HSM1_af;
  double HSM1_kf;

  /* flag for model */
  unsigned HSM1_type_Given  :1;
  unsigned HSM1_level_Given  :1;
  unsigned HSM1_info_Given  :1;
  unsigned HSM1_noise_Given :1;
  unsigned HSM1_version_Given :1;
  unsigned HSM1_show_Given :1;
  unsigned HSM1_corsrd_Given  :1;
  unsigned HSM1_coiprv_Given  :1;
  unsigned HSM1_copprv_Given  :1;
  unsigned HSM1_cocgso_Given  :1;
  unsigned HSM1_cocgdo_Given  :1;
  unsigned HSM1_cocgbo_Given  :1;
  unsigned HSM1_coadov_Given  :1;
  unsigned HSM1_coxx08_Given  :1;
  unsigned HSM1_coxx09_Given  :1;
  unsigned HSM1_coisub_Given  :1;
  unsigned HSM1_coiigs_Given  :1;
  unsigned HSM1_cogidl_Given  :1;
  unsigned HSM1_cogisl_Given  :1;
  unsigned HSM1_coovlp_Given  :1;
  unsigned HSM1_conois_Given  :1;
  unsigned HSM1_coisti_Given  :1; /* HiSIM1.1 */
  unsigned HSM1_cosmbi_Given  :1; /* HiSIM1.2 */
  unsigned HSM1_glpart1_Given :1; /* HiSIM1.2 */
  unsigned HSM1_glpart2_Given :1; /* HiSIM1.2 */
  unsigned HSM1_kappa_Given :1;   /* HiSIM1.2 */
  unsigned HSM1_xdiffd_Given :1; /* HiSIM1.2 */
  unsigned HSM1_pthrou_Given :1; /* HiSIM1.2 */
  unsigned HSM1_vdiffj_Given :1; /* HiSIM1.2 */
  unsigned HSM1_vmax_Given  :1;
  unsigned HSM1_bgtmp1_Given  :1;
  unsigned HSM1_bgtmp2_Given  :1;
  unsigned HSM1_tox_Given  :1;
  unsigned HSM1_xld_Given  :1;
  unsigned HSM1_xwd_Given  :1; 
  unsigned HSM1_xj_Given  :1;    /* HiSIM1.0 */
  unsigned HSM1_xqy_Given  :1;   /* HiSIM1.1 */
  unsigned HSM1_rs_Given  :1;
  unsigned HSM1_rd_Given  :1;
  unsigned HSM1_vfbc_Given  :1;
  unsigned HSM1_nsubc_Given  :1;
  unsigned HSM1_parl1_Given  :1;
  unsigned HSM1_parl2_Given  :1;
  unsigned HSM1_lp_Given  :1;
  unsigned HSM1_nsubp_Given  :1;
  unsigned HSM1_scp1_Given  :1;
  unsigned HSM1_scp2_Given  :1;
  unsigned HSM1_scp3_Given  :1;
  unsigned HSM1_sc1_Given  :1;
  unsigned HSM1_sc2_Given  :1;
  unsigned HSM1_sc3_Given  :1;
  unsigned HSM1_pgd1_Given  :1;
  unsigned HSM1_pgd2_Given  :1;
  unsigned HSM1_pgd3_Given  :1;
  unsigned HSM1_ndep_Given  :1;
  unsigned HSM1_ninv_Given  :1;
  unsigned HSM1_ninvd_Given  :1;
  unsigned HSM1_muecb0_Given  :1;
  unsigned HSM1_muecb1_Given  :1;
  unsigned HSM1_mueph1_Given  :1;
  unsigned HSM1_mueph0_Given  :1;
  unsigned HSM1_mueph2_Given  :1;
  unsigned HSM1_w0_Given  :1;
  unsigned HSM1_muesr1_Given  :1;
  unsigned HSM1_muesr0_Given  :1;
  unsigned HSM1_bb_Given  :1;
  unsigned HSM1_sub1_Given  :1;
  unsigned HSM1_sub2_Given  :1;
  unsigned HSM1_sub3_Given  :1;
  unsigned HSM1_wvthsc_Given  :1; /* HiSIM1.1 */
  unsigned HSM1_nsti_Given  :1;   /* HiSIM1.1 */
  unsigned HSM1_wsti_Given  :1;   /* HiSIM1.1 */
  unsigned HSM1_cgso_Given  :1;
  unsigned HSM1_cgdo_Given  :1;
  unsigned HSM1_cgbo_Given  :1;
  unsigned HSM1_tpoly_Given  :1;
  unsigned HSM1_js0_Given  :1;
  unsigned HSM1_js0sw_Given  :1;
  unsigned HSM1_nj_Given  :1;
  unsigned HSM1_njsw_Given  :1;  
  unsigned HSM1_xti_Given  :1;
  unsigned HSM1_cj_Given  :1;
  unsigned HSM1_cjsw_Given  :1;
  unsigned HSM1_cjswg_Given  :1;
  unsigned HSM1_mj_Given  :1;
  unsigned HSM1_mjsw_Given  :1;
  unsigned HSM1_mjswg_Given  :1;
  unsigned HSM1_pb_Given  :1;
  unsigned HSM1_pbsw_Given  :1;
  unsigned HSM1_pbswg_Given  :1;
  unsigned HSM1_xpolyd_Given  :1;
  unsigned HSM1_clm1_Given  :1;
  unsigned HSM1_clm2_Given  :1;
  unsigned HSM1_clm3_Given  :1;
  unsigned HSM1_muetmp_Given  :1;
  unsigned HSM1_rpock1_Given  :1;
  unsigned HSM1_rpock2_Given  :1;
  unsigned HSM1_rpocp1_Given  :1; /* HiSIM1.1 */
  unsigned HSM1_rpocp2_Given  :1; /* HiSIM1.1 */
  unsigned HSM1_vover_Given  :1;
  unsigned HSM1_voverp_Given  :1;
  unsigned HSM1_wfc_Given  :1;
  unsigned HSM1_qme1_Given  :1;
  unsigned HSM1_qme2_Given  :1;
  unsigned HSM1_qme3_Given  :1;
  unsigned HSM1_gidl1_Given  :1;
  unsigned HSM1_gidl2_Given  :1;
  unsigned HSM1_gidl3_Given  :1;
  unsigned HSM1_gleak1_Given  :1;
  unsigned HSM1_gleak2_Given  :1;
  unsigned HSM1_gleak3_Given  :1;
  unsigned HSM1_vzadd0_Given  :1;
  unsigned HSM1_pzadd0_Given  :1;
  unsigned HSM1_nftrp_Given  :1;
  unsigned HSM1_nfalp_Given  :1;
  unsigned HSM1_cit_Given  :1;

  unsigned HSM1_ef_Given :1;
  unsigned HSM1_af_Given :1;
  unsigned HSM1_kf_Given :1;

} HSM1model;

#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/

#define HSM1_BAD_PARAM -1

/* flags */
#define HSM1_MOD_NMOS     1
#define HSM1_MOD_PMOS     2
#define HSM1_MOD_LEVEL    3
#define HSM1_MOD_INFO     4
#define HSM1_MOD_NOISE    5
#define HSM1_MOD_VERSION  6
#define HSM1_MOD_SHOW     7
#define HSM1_MOD_CORSRD  11
#define HSM1_MOD_COIPRV  12
#define HSM1_MOD_COPPRV  13
#define HSM1_MOD_COCGSO  14
#define HSM1_MOD_COCGDO  15
#define HSM1_MOD_COCGBO  16
#define HSM1_MOD_COADOV  17
#define HSM1_MOD_COXX08  18
#define HSM1_MOD_COXX09  19
#define HSM1_MOD_COISUB  21
#define HSM1_MOD_COIIGS  22
#define HSM1_MOD_COGIDL  23
#define HSM1_MOD_COOVLP  24
#define HSM1_MOD_CONOIS  25
#define HSM1_MOD_COISTI  26 /* HiSIM1.1 */
#define HSM1_MOD_COSMBI  27 /* HiSIM1.2 */
#define HSM1_MOD_COGISL  28 /* HiSIM1.2 */
/* device parameters */
#define HSM1_L 51
#define HSM1_W 52
#define HSM1_AD 53
#define HSM1_AS 54
#define HSM1_PD 55
#define HSM1_PS 56
#define HSM1_NRD 57
#define HSM1_NRS 58
#define HSM1_TEMP 59
#define HSM1_DTEMP 60
#define HSM1_OFF 61
#define HSM1_IC_VBS 62
#define HSM1_IC_VDS 63
#define HSM1_IC_VGS 64
#define HSM1_IC 65
#define HSM1_M 66

/* model parameters */
#define HSM1_MOD_VMAX   101
#define HSM1_MOD_BGTMP1 103
#define HSM1_MOD_BGTMP2 104
#define HSM1_MOD_TOX    105
#define HSM1_MOD_XLD    106
#define HSM1_MOD_XWD    107
#define HSM1_MOD_XJ     996 /* HiSIM1.0 */
#define HSM1_MOD_XQY    997 /* HiSIM1.1 */
#define HSM1_MOD_RS     108
#define HSM1_MOD_RD     109
#define HSM1_MOD_VFBC   110
#define HSM1_MOD_NSUBC  113
#define HSM1_MOD_PARL1  122
#define HSM1_MOD_PARL2  123
#define HSM1_MOD_SC1    124
#define HSM1_MOD_SC2    125
#define HSM1_MOD_SC3    126
#define HSM1_MOD_NDEP   129
#define HSM1_MOD_NINV   130
#define HSM1_MOD_MUECB0 131
#define HSM1_MOD_MUECB1 132
#define HSM1_MOD_MUEPH1 133
#define HSM1_MOD_MUEPH0 134
#define HSM1_MOD_MUEPH2 999
#define HSM1_MOD_W0 	998
#define HSM1_MOD_MUESR1 135
#define HSM1_MOD_MUESR0 136
#define HSM1_MOD_BB     137
#define HSM1_MOD_SUB1   141
#define HSM1_MOD_SUB2   142
#define HSM1_MOD_SUB3   143
#define HSM1_MOD_CGSO   144
#define HSM1_MOD_CGDO   145
#define HSM1_MOD_CGBO   146
#define HSM1_MOD_JS0    147 
#define HSM1_MOD_JS0SW  148
#define HSM1_MOD_NJ     149
#define HSM1_MOD_NJSW   150
#define HSM1_MOD_XTI    151
#define HSM1_MOD_CJ     152
#define HSM1_MOD_CJSW   156
#define HSM1_MOD_CJSWG  157
#define HSM1_MOD_MJ     160
#define HSM1_MOD_MJSW   161
#define HSM1_MOD_MJSWG  163
#define HSM1_MOD_PB     166
#define HSM1_MOD_PBSW   168
#define HSM1_MOD_PBSWG  169
#define HSM1_MOD_XPOLYD 170
#define HSM1_MOD_TPOLY  171
#define HSM1_MOD_LP     172
#define HSM1_MOD_NSUBP  173
#define HSM1_MOD_SCP1   174
#define HSM1_MOD_SCP2   175
#define HSM1_MOD_SCP3   176
#define HSM1_MOD_PGD1   177
#define HSM1_MOD_PGD2   178
#define HSM1_MOD_PGD3   179
#define HSM1_MOD_CLM1   180
#define HSM1_MOD_CLM2   181
#define HSM1_MOD_CLM3   182
#define HSM1_MOD_NINVD  183
#define HSM1_MOD_MUETMP 190
#define HSM1_MOD_RPOCK1 191
#define HSM1_MOD_RPOCK2 192
#define HSM1_MOD_VOVER  193
#define HSM1_MOD_VOVERP 194
#define HSM1_MOD_WFC    195
#define HSM1_MOD_QME1   196
#define HSM1_MOD_QME2   197
#define HSM1_MOD_QME3   198
#define HSM1_MOD_GIDL1  199
#define HSM1_MOD_GIDL2  200
#define HSM1_MOD_GIDL3  201
#define HSM1_MOD_GLEAK1 202
#define HSM1_MOD_GLEAK2 203
#define HSM1_MOD_GLEAK3 204
#define HSM1_MOD_VZADD0 205
#define HSM1_MOD_PZADD0 206
#define HSM1_MOD_WVTHSC 207 /* HiSIM1.1 */
#define HSM1_MOD_NSTI   208 /* HiSIM1.1 */
#define HSM1_MOD_WSTI   209 /* HiSIM1.1 */
#define HSM1_MOD_RPOCP1 210 /* HiSIM1.1 */
#define HSM1_MOD_RPOCP2 211 /* HiSIM1.1 */
#define HSM1_MOD_GLPART1 212 /* HiSIM1.2 */
#define HSM1_MOD_GLPART2 213 /* HiSIM1.2 */
#define HSM1_MOD_KAPPA  214 /* HiSIM1.2 */
#define HSM1_MOD_XDIFFD 215 /* HiSIM1.2 */
#define HSM1_MOD_PTHROU  216 /* HiSIM1.2 */
#define HSM1_MOD_VDIFFJ 217 /* HiSIM1.2 */
#define HSM1_MOD_NFTRP  401
#define HSM1_MOD_NFALP  402
#define HSM1_MOD_CIT    403
#define HSM1_MOD_EF     500
#define HSM1_MOD_AF     501
#define HSM1_MOD_KF     502

/* device questions */
#define HSM1_DNODE      341
#define HSM1_GNODE      342
#define HSM1_SNODE      343
#define HSM1_BNODE      344
#define HSM1_DNODEPRIME 345
#define HSM1_SNODEPRIME 346
#define HSM1_VBD        347
#define HSM1_VBS        348
#define HSM1_VGS        349
#define HSM1_VDS        350
#define HSM1_CD         351
#define HSM1_CBS        352
#define HSM1_CBD        353
#define HSM1_GM         354
#define HSM1_GDS        355
#define HSM1_GMBS       356
#define HSM1_GBD        357
#define HSM1_GBS        358
#define HSM1_QB         359
#define HSM1_CQB        360
#define HSM1_QG         361
#define HSM1_CQG        362
#define HSM1_QD         363
#define HSM1_CQD        364
#define HSM1_CGG        365
#define HSM1_CGD        366
#define HSM1_CGS        367
#define HSM1_CBG        368
#define HSM1_CAPBD      369
#define HSM1_CQBD       370
#define HSM1_CAPBS      371
#define HSM1_CQBS       372
#define HSM1_CDG        373
#define HSM1_CDD        374
#define HSM1_CDS        375
#define HSM1_VON        376
#define HSM1_VDSAT      377
#define HSM1_QBS        378
#define HSM1_QBD        379
#define HSM1_SOURCECONDUCT      380
#define HSM1_DRAINCONDUCT       381
#define HSM1_CBDB               382
#define HSM1_CBSB               383

#include "hsm1ext.h"

/*
extern void HSM1evaluate(double,double,double,HSM1instance*,HSM1model*,
        double*,double*,double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, CKTcircuit*);
*/

#endif /*HSM1*/

