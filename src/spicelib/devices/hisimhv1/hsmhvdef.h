/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 1  SUBVERSION : 2  REVISION : 4 )
 Model Parameter VERSION : 1.23
 FILE : hsmhvdef

 DATE : 2013.04.30

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#ifndef HSMHV
#define HSMHV

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

/* declarations for HiSIMHV MOSFETs */

/* unit-converted model parameters */
typedef struct sHSMHVmodelMKSParam {
  double HSMHV_npext ;
  double HSMHV_vmax ;
  double HSMHV_ll ;
  double HSMHV_wl ;
  double HSMHV_svgsl ;
  double HSMHV_svgsw ;
  double HSMHV_svbsl ;
  double HSMHV_slgl ;
  double HSMHV_sub1l ;
  double HSMHV_slg ;
  double HSMHV_sub2l ;
  double HSMHV_subld2 ;
  double HSMHV_rd22 ;
  double HSMHV_rd23 ;
  double HSMHV_rd24 ;
  double HSMHV_rdtemp1 ;
  double HSMHV_rdtemp2 ;
  double HSMHV_rdvd ;
  double HSMHV_rdvdtemp1 ;
  double HSMHV_rdvdtemp2 ;
//double HSMHV_muecb0 ;
//double HSMHV_muecb1 ;
//double HSMHV_muesr1 ;
//double HSMHV_mueph1 ;
  double HSMHV_nsubsub ;
  double HSMHV_nsubpsti1 ;
  double HSMHV_muesti1 ;
  double HSMHV_wfc ;
  double HSMHV_glksd1 ;
  double HSMHV_glksd2 ;
  double HSMHV_glksd3 ;
  double HSMHV_gleak2 ;
  double HSMHV_gleak4 ;
  double HSMHV_gleak5 ;
  double HSMHV_gleak7 ;
  double HSMHV_glkb2 ;
  double HSMHV_fn2 ;
  double HSMHV_gidl1 ;
  double HSMHV_gidl2 ;
  double HSMHV_nfalp ;
  double HSMHV_nftrp ;
  double HSMHV_cit ;
  double HSMHV_ovslp ;
  double HSMHV_dly3 ;
  double HSMHV_rth0 ;
  double HSMHV_cth0 ;
} HSMHVmodelMKSParam ;


/* binning parameters */
typedef struct sHSMHVbinningParam {
  double HSMHV_vmax ;
  double HSMHV_bgtmp1 ;
  double HSMHV_bgtmp2 ;
  double HSMHV_eg0 ;
  double HSMHV_vfbover ;
  double HSMHV_nover ;
  double HSMHV_novers ;
  double HSMHV_wl2 ;
  double HSMHV_vfbc ;
  double HSMHV_nsubc ;
  double HSMHV_nsubp ;
  double HSMHV_scp1 ;
  double HSMHV_scp2 ;
  double HSMHV_scp3 ;
  double HSMHV_sc1 ;
  double HSMHV_sc2 ;
  double HSMHV_sc3 ;
  double HSMHV_pgd1 ;
  double HSMHV_pgd3 ;
  double HSMHV_ndep ;
  double HSMHV_ninv ;
  double HSMHV_muecb0 ;
  double HSMHV_muecb1 ;
  double HSMHV_mueph1 ;
  double HSMHV_vtmp ;
  double HSMHV_wvth0 ;
  double HSMHV_muesr1 ;
  double HSMHV_muetmp ;
  double HSMHV_sub1 ;
  double HSMHV_sub2 ;
  double HSMHV_svds ;
  double HSMHV_svbs ;
  double HSMHV_svgs ;
  double HSMHV_fn1 ;
  double HSMHV_fn2 ;
  double HSMHV_fn3 ;
  double HSMHV_fvbs ;
  double HSMHV_nsti ;
  double HSMHV_wsti ;
  double HSMHV_scsti1 ;
  double HSMHV_scsti2 ;
  double HSMHV_vthsti ;
  double HSMHV_muesti1 ;
  double HSMHV_muesti2 ;
  double HSMHV_muesti3 ;
  double HSMHV_nsubpsti1 ;
  double HSMHV_nsubpsti2 ;
  double HSMHV_nsubpsti3 ;
  double HSMHV_cgso ;
  double HSMHV_cgdo ;
  double HSMHV_js0 ;
  double HSMHV_js0sw ;
  double HSMHV_nj ;
  double HSMHV_cisbk ;
  double HSMHV_clm1 ;
  double HSMHV_clm2 ;
  double HSMHV_clm3 ;
  double HSMHV_wfc ;
  double HSMHV_gidl1 ;
  double HSMHV_gidl2 ;
  double HSMHV_gleak1 ;
  double HSMHV_gleak2 ;
  double HSMHV_gleak3 ;
  double HSMHV_gleak6 ;
  double HSMHV_glksd1 ;
  double HSMHV_glksd2 ;
  double HSMHV_glkb1 ;
  double HSMHV_glkb2 ;
  double HSMHV_nftrp ;
  double HSMHV_nfalp ;
  double HSMHV_pthrou ;
  double HSMHV_vdiffj ;
  double HSMHV_ibpc1 ;
  double HSMHV_ibpc2 ;
  double HSMHV_cgbo ;
  double HSMHV_cvdsover ;
  double HSMHV_falph ;
  double HSMHV_npext ;
  double HSMHV_powrat ;
  double HSMHV_rd ;
  double HSMHV_rd22 ;
  double HSMHV_rd23 ;
  double HSMHV_rd24 ;
  double HSMHV_rdict1 ;
  double HSMHV_rdov13 ;
  double HSMHV_rdslp1 ;
  double HSMHV_rdvb ;
  double HSMHV_rdvd ;
  double HSMHV_rdvg11 ;
  double HSMHV_rs ;
  double HSMHV_rth0 ;
  double HSMHV_vover ;
  /*-----------SHE--------------*/
  double HSMHV_rth ;
  double HSMHV_cth ;
  /*-----------------------------*/

} HSMHVbinningParam ;

 
/* unit-converted parameters for each instance */
typedef struct sHSMHVhereMKSParam {
  double HSMHV_vmax ;
  double HSMHV_subld2 ;
//double HSMHV_muecb0 ;
//double HSMHV_muecb1 ;
//double HSMHV_muesr1 ;
//double HSMHV_mueph1 ;
  double HSMHV_ndep ;
  double HSMHV_ninv ;
  double HSMHV_nsubc ;
  double HSMHV_nsubcdfm ;
  double HSMHV_nsubp ;
  double HSMHV_nsubpsti1 ;
  double HSMHV_muesti1 ;
  double HSMHV_nsti ;
  double HSMHV_npext ;
  double HSMHV_nover ;
  double HSMHV_novers ;
  double HSMHV_wfc ;
  double HSMHV_glksd1 ;
  double HSMHV_glksd2 ;
  double HSMHV_gleak2 ;
  double HSMHV_glkb2 ;
  double HSMHV_fn2 ;
  double HSMHV_gidl1 ;
  double HSMHV_gidl2 ;
  double HSMHV_nfalp ;
  double HSMHV_nftrp ;
} HSMHVhereMKSParam ;


/* information needed for each instance */
typedef struct sHSMHVinstance {

  struct GENinstance gen;

#define HSMHVmodPtr(inst) ((struct sHSMHVmodel *)((inst)->gen.GENmodPtr))
#define HSMHVnextInstance(inst) ((struct sHSMHVinstance *)((inst)->gen.GENnextInstance))
#define HSMHVname gen.GENname
#define HSMHVstates gen.GENstate

  const int HSMHVdNode;      /* number of the drain node of the mosfet */
  const int HSMHVgNode;      /* number of the gate node of the mosfet */
  const int HSMHVsNode;      /* number of the source node of the mosfet */
  const int HSMHVbNode;      /* number of the bulk node of the mosfet */
  const int HSMHVsubNodeExt; /* number of the substrate node */
  const int HSMHVtempNodeExt;/* number of the temp node----------SHE--------*/
  int HSMHVsubNode;    /* number of the substrate node */
  int HSMHVtempNode;   /* number of the temp node */
  int HSMHVdNodePrime; /* number od the inner drain node */
  int HSMHVgNodePrime; /* number of the inner gate node */
  int HSMHVsNodePrime; /* number od the inner source node */
  int HSMHVbNodePrime;
  int HSMHVdbNode;
  int HSMHVsbNode;
  int HSMHVqiNode;     /* number of the qi node in case of NQS */
  int HSMHVqbNode;     /* number of the qb node in case of NQS */

  double HSMHV_noiflick; /* for 1/f noise calc. */
  double HSMHV_noithrml; /* for thrmal noise calc. */
  double HSMHV_noiigate; /* for induced gate noise */
  double HSMHV_noicross; /* for induced gate noise */

  /* instance */
  int HSMHV_coselfheat; /* Self-heating model */
  int HSMHV_cosubnode;  /* switch tempNode to subNode */
  double HSMHV_l;    /* the length of the channel region */
  double HSMHV_w;    /* the width of the channel region */
  double HSMHV_ad;   /* the area of the drain diffusion */
  double HSMHV_as;   /* the area of the source diffusion */
  double HSMHV_pd;   /* perimeter of drain junction [m] */
  double HSMHV_ps;   /* perimeter of source junction [m] */
  double HSMHV_nrd;  /* equivalent num of squares of drain [-] (unused) */
  double HSMHV_nrs;  /* equivalent num of squares of source [-] (unused) */
  double HSMHV_dtemp;

  double HSMHV_weff;    /* the effective width of the channel region */
  double HSMHV_weff_ld; /* the effective width of the drift region */
  double HSMHV_weff_cv;  /* the effective width of the drift region for capacitance */
  double HSMHV_weff_nf; /* Weff * NF */
  double HSMHV_weffcv_nf;  /* Weffcv * NF */
  double HSMHV_leff;    /* the effective length of the channel region */

  int HSMHV_corbnet  ;
  double HSMHV_rbpb ;
  double HSMHV_rbpd ;
  double HSMHV_rbps ;
  double HSMHV_rbdb ;
  double HSMHV_rbsb ;

  int HSMHV_corg ;
  double HSMHV_ngcon;
  double HSMHV_xgw;
  double HSMHV_xgl;
  double HSMHV_nf;

  double HSMHV_sa;
  double HSMHV_sb;
  double HSMHV_sd;
  double HSMHV_nsubcdfm;
  double HSMHV_m;
  double HSMHV_subld1;
  double HSMHV_subld2;
  double HSMHV_lover;  
  double HSMHV_lovers;  
  double HSMHV_loverld;
  double HSMHV_ldrift1;
  double HSMHV_ldrift2;
  double HSMHV_ldrift1s;
  double HSMHV_ldrift2s;

  int HSMHV_called; /* flag to check the first call */
  /* previous values to evaluate initial guess */
  double HSMHV_mode_prv;
  double HSMHV_vbsc_prv;
  double HSMHV_vdsc_prv;
  double HSMHV_vgsc_prv;
  double HSMHV_ps0_prv;
  double HSMHV_ps0_dvbs_prv;
  double HSMHV_ps0_dvds_prv;
  double HSMHV_ps0_dvgs_prv;
  double HSMHV_ps0_dtemp_prv;
  double HSMHV_pds_prv;
  double HSMHV_pds_dvbs_prv;
  double HSMHV_pds_dvds_prv;
  double HSMHV_pds_dvgs_prv;
  double HSMHV_pds_dtemp_prv;
  /* double HSMHV_ids_prv;		not used */
  /* double HSMHV_ids_dvbs_prv;		not used */
  /* double HSMHV_ids_dvds_prv;		not used */
  /* double HSMHV_ids_dvgs_prv;		not used */
  /* double HSMHV_ids_dtemp_prv;	not used */
  double HSMHV_mode_prv2;
  double HSMHV_vbsc_prv2;
  double HSMHV_vdsc_prv2;
  double HSMHV_vgsc_prv2;
  double HSMHV_ps0_prv2;	/* assigned but not used */
  double HSMHV_ps0_dvbs_prv2;
  double HSMHV_ps0_dvds_prv2;
  double HSMHV_ps0_dvgs_prv2;
  double HSMHV_pds_prv2;	/* assigned but not used */
  double HSMHV_pds_dvbs_prv2;
  double HSMHV_pds_dvds_prv2;
  double HSMHV_pds_dvgs_prv2;
  double HSMHV_temp_prv;
/*   double HSMHV_time; /\* for debug print *\/ */

  /* output */
  /* int    HSMHV_capop;	not used */
  /* double HSMHV_gd;		not used */
  /* double HSMHV_gs;		not used */
  double HSMHV_cgso;		/* can be made local */
  double HSMHV_cgdo;
  double HSMHV_cgbo;
  double HSMHV_cggo;
  double HSMHV_cdso;		/* can be made local */
  double HSMHV_cddo;
  double HSMHV_cdgo;
  double HSMHV_cdbo;
  double HSMHV_csso;		/* can be made local */
  double HSMHV_csdo;		/* can be made local */
  double HSMHV_csgo;		/* can be made local */
  double HSMHV_csbo;		/* can be made local */
  double HSMHV_cbdo;
  double HSMHV_cbgo;
  double HSMHV_cbbo;
  /* double HSMHV_cqyd;		not used */
  /* double HSMHV_cqyg;		not used */
  /* double HSMHV_cqyb;		not used */
  double HSMHV_von; /* vth */
  double HSMHV_vdsat;
  /* double HSMHV_capgs;	not used */
  /* double HSMHV_capgd;	not used */
  /* double HSMHV_capgb;	not used */

  /* double HSMHV_rth0;		not used */
  /* double HSMHV_cth0;		not used */
  /* double HSMHV_cth;		not used */


#define XDIM 14
  double HSMHV_ydc_d[XDIM],  HSMHV_ydc_dP[XDIM], HSMHV_ydc_g[XDIM],  HSMHV_ydc_gP[XDIM], HSMHV_ydc_s[XDIM], HSMHV_ydc_sP[XDIM], 
         HSMHV_ydc_bP[XDIM], HSMHV_ydc_b[XDIM],  HSMHV_ydc_db[XDIM], HSMHV_ydc_sb[XDIM], HSMHV_ydc_t[XDIM], HSMHV_ydc_qi[XDIM],
         HSMHV_ydc_qb[XDIM];
  double HSMHV_ydyn_d[XDIM],  HSMHV_ydyn_dP[XDIM], HSMHV_ydyn_g[XDIM],  HSMHV_ydyn_gP[XDIM], HSMHV_ydyn_s[XDIM], HSMHV_ydyn_sP[XDIM], 
         HSMHV_ydyn_bP[XDIM], HSMHV_ydyn_b[XDIM],  HSMHV_ydyn_db[XDIM], HSMHV_ydyn_sb[XDIM], HSMHV_ydyn_t[XDIM], HSMHV_ydyn_qi[XDIM],
         HSMHV_ydyn_qb[XDIM];

  /* resistances */
  double HSMHV_Rd ; /* different from HSMHV_rd */
  double HSMHV_dRd_dVdse  ;
  double HSMHV_dRd_dVgse  ;
  double HSMHV_dRd_dVbse  ;
  double HSMHV_dRd_dVsubs ;
  double HSMHV_dRd_dTi    ;
  double HSMHV_Rs ; /* different from HSMHV_rs */
  double HSMHV_dRs_dVdse  ;
  double HSMHV_dRs_dVgse  ;
  double HSMHV_dRs_dVbse  ;
  double HSMHV_dRs_dVsubs ;
  double HSMHV_dRs_dTi    ;
  /* drain current */
  double HSMHV_ids;
  double HSMHV_gds;		/* used for printout, but not loaded */
  double HSMHV_gm;		/* used for printout, but not loaded */
  double HSMHV_gmbs;		/* used for printout, but not loaded */
  double HSMHV_dIds_dVdse ;
  double HSMHV_dIds_dVgse ;
  double HSMHV_dIds_dVbse ;
  double HSMHV_dIds_dVdsi ;
  double HSMHV_dIds_dVgsi ;
  double HSMHV_dIds_dVbsi ;
  double HSMHV_dIds_dTi   ;
  /* substrate current */
  double HSMHV_isub;
  /* double HSMHV_gbgs;		not used */
  /* double HSMHV_gbds;		not used */
  /* double HSMHV_gbbs;		not used */
  double HSMHV_dIsub_dVdsi ;
  double HSMHV_dIsub_dVgsi ;
  double HSMHV_dIsub_dVbsi ;
  double HSMHV_dIsub_dTi   ;
  double HSMHV_dIsub_dVdse ;
  /* gidl and gisl current */
  double HSMHV_igidl; /* gate induced drain leakage */
  /* double HSMHV_gigidlgs;	not used */
  /* double HSMHV_gigidlds;	not used */
  /* double HSMHV_gigidlbs;	not used */
  double HSMHV_dIgidl_dVdsi ;
  double HSMHV_dIgidl_dVgsi ;
  double HSMHV_dIgidl_dVbsi ;
  double HSMHV_dIgidl_dTi   ;
  double HSMHV_igisl; /* gate induced source leakage */
  /* double HSMHV_gigislgd;	not used */
  /* double HSMHV_gigislsd;	not used */
  /* double HSMHV_gigislbd;	not used */
  double HSMHV_dIgisl_dVdsi ;
  double HSMHV_dIgisl_dVgsi ;
  double HSMHV_dIgisl_dVbsi ;
  double HSMHV_dIgisl_dTi   ;
  /* gate leakage currents */
  double HSMHV_igb; /* gate tunneling current (gate to bulk) */
  /* double HSMHV_gigbg;	not used */
  /* double HSMHV_gigbd;	not used */
  /* double HSMHV_gigbb;	not used */
  /* double HSMHV_gigbs;	not used */
  double HSMHV_dIgb_dVdsi ;
  double HSMHV_dIgb_dVgsi ;
  double HSMHV_dIgb_dVbsi ;
  double HSMHV_dIgb_dTi   ;
  double HSMHV_igd; /* gate tunneling current (gate to drain) */
  /* double HSMHV_gigdg;	not used */
  /* double HSMHV_gigdd;	not used */
  /* double HSMHV_gigdb;	not used */
  /* double HSMHV_gigds;	not used */
  double HSMHV_dIgd_dVdsi ;
  double HSMHV_dIgd_dVgsi ;
  double HSMHV_dIgd_dVbsi ;
  double HSMHV_dIgd_dTi   ;
  double HSMHV_igs; /* gate tunneling current (gate to source) */
  /* double HSMHV_gigsg;	not used */
  /* double HSMHV_gigsd;	not used */
  /* double HSMHV_gigsb;	not used */
  /* double HSMHV_gigss;	not used */
  double HSMHV_dIgs_dVdsi ;
  double HSMHV_dIgs_dVgsi ;
  double HSMHV_dIgs_dVbsi ;
  double HSMHV_dIgs_dTi   ;
  /* charges */
  double HSMHV_qd;
  double HSMHV_cdgb;		/* used for printout, but not loaded */
  /* double HSMHV_cddb;		not used */
  /* double HSMHV_cdsb;         not used */
  /* double HSMHVcdT;		not used */
  double HSMHV_dQdi_dVdsi ;
  double HSMHV_dQdi_dVgsi ;
  double HSMHV_dQdi_dVbsi ;
  double HSMHV_dQdi_dTi   ;
  double HSMHV_qg;
  double HSMHV_cggb;		/* used for printout, but not loaded */
  double HSMHV_cgdb;		/* used for printout, but not loaded */
  double HSMHV_cgsb;		/* used for printout, but not loaded */
  /* double HSMHVcgT;		not used */
  double HSMHV_dQg_dVdsi ;
  double HSMHV_dQg_dVgsi ;
  double HSMHV_dQg_dVbsi ;
  double HSMHV_dQg_dTi   ;
  double HSMHV_qs;
  double HSMHV_dQsi_dVdsi ;
  double HSMHV_dQsi_dVgsi ;
  double HSMHV_dQsi_dVbsi ;
  double HSMHV_dQsi_dTi   ;
  double HSMHV_qb;  /* bulk charge qb = -(qg + qd + qs) */
  double HSMHV_cbgb;		/* used for printout, but not loaded */
  /* double HSMHV_cbdb;		not used */
  /* double HSMHV_cbsb;		not used */
  /* double HSMHVcbT;		not used */
  double HSMHV_dQb_dVdsi ;     /* Qb: bulk charge inclusive overlaps, Qbulk: bulk charge without overlaps (see above) */
  double HSMHV_dQb_dVgsi ;
  double HSMHV_dQb_dVbsi ;
  double HSMHV_dQb_dTi   ;
  /* outer charges (fringing etc.) */
  double HSMHV_qdp ;
  double HSMHV_dqdp_dVdse ;
  double HSMHV_dqdp_dVgse ;
  double HSMHV_dqdp_dVbse ;
  double HSMHV_dqdp_dTi   ;
  double HSMHV_qsp ;
  double HSMHV_dqsp_dVdse ;
  double HSMHV_dqsp_dVgse ;
  double HSMHV_dqsp_dVbse ;
  double HSMHV_dqsp_dTi   ;
  double HSMHV_qgext ;
  double HSMHV_dQgext_dVdse ;
  double HSMHV_dQgext_dVgse ;
  double HSMHV_dQgext_dVbse ;
  double HSMHV_dQgext_dTi   ;
  double HSMHV_qdext ;
  double HSMHV_dQdext_dVdse ;
  double HSMHV_dQdext_dVgse ;
  double HSMHV_dQdext_dVbse ;
  double HSMHV_dQdext_dTi   ;
  double HSMHV_qbext ;
  double HSMHV_dQbext_dVdse ;
  double HSMHV_dQbext_dVgse ;
  double HSMHV_dQbext_dVbse ;
  double HSMHV_dQbext_dTi   ;
  double HSMHV_qsext ;
  double HSMHV_dQsext_dVdse ;
  double HSMHV_dQsext_dVgse ;
  double HSMHV_dQsext_dVbse ;
  double HSMHV_dQsext_dTi   ;
  /* junctions */
  double HSMHV_ibd;
  double HSMHV_gbd;
  double HSMHV_gbdT;
  double HSMHV_ibs;
  double HSMHV_gbs;
  double HSMHV_gbsT;
  double HSMHV_qbd;
  double HSMHV_capbd;
  double HSMHV_gcbdT;
  double HSMHV_qbs;
  double HSMHV_capbs;
  double HSMHV_gcbsT;

  /* double HSMHV_gtempg;	not used */
  /* double HSMHV_gtempt;	not used */
  /* double HSMHV_gtempd;	not used */
  /* double HSMHV_gtempb;	not used */

  /* double HSMHV_gmt;		not used */
  /* double HSMHV_isubt;	not used */


  /* NQS */
  double HSMHV_tau ;
  double HSMHV_tau_dVgsi ;
  double HSMHV_tau_dVdsi ;
  double HSMHV_tau_dVbsi ;
  double HSMHV_tau_dTi   ;
  double HSMHV_Xd  ;
  double HSMHV_Xd_dVgsi  ;
  double HSMHV_Xd_dVdsi  ;
  double HSMHV_Xd_dVbsi  ;
  double HSMHV_Xd_dTi    ;
  double HSMHV_Qi  ;
  double HSMHV_Qi_dVgsi  ;
  double HSMHV_Qi_dVdsi  ;
  double HSMHV_Qi_dVbsi  ;
  double HSMHV_Qi_dTi    ;
  double HSMHV_taub  ;
  double HSMHV_taub_dVgsi  ;
  double HSMHV_taub_dVdsi  ;
  double HSMHV_taub_dVbsi  ;
  double HSMHV_taub_dTi    ;
  double HSMHV_Qbulk  ;                          /* Qbulk: without overlaps, Qb: inclusive overlaps (see below) */
  double HSMHV_Qbulk_dVgsi  ;
  double HSMHV_Qbulk_dVdsi  ;
  double HSMHV_Qbulk_dVbsi  ;
  double HSMHV_Qbulk_dTi    ;




  /* internal variables */
  double HSMHV_eg ;
  double HSMHV_beta ;
  double HSMHV_beta_inv ;
  double HSMHV_beta2 ;
  double HSMHV_betatnom ;
  double HSMHV_nin ;
  double HSMHV_egp12 ;
  double HSMHV_egp32 ;
  double HSMHV_lgate ;
  double HSMHV_wg ;
  double HSMHV_mueph ;
  double HSMHV_mphn0 ;
  double HSMHV_mphn1 ;
  double HSMHV_muesr ;
  double HSMHV_rdvd ;
  double HSMHV_rsvd ; /* for the reverse mode */
  double HSMHV_rd23 ;

  double HSMHV_ninvd ;
  double HSMHV_ninvd0 ;

  double HSMHV_nsub ;
  double HSMHV_qnsub ;
  double HSMHV_qnsub_esi ;
  double HSMHV_2qnsub_esi ;
  double HSMHV_ptovr0 ;
  double HSMHV_ptovr ;
  double HSMHV_vmax0 ;
  double HSMHV_vmax ;
  double HSMHV_pb2 ;
  double HSMHV_pb20 ;
  double HSMHV_pb2c ;
  double HSMHV_cnst0 ;
  double HSMHV_cnst1 ;
  double HSMHV_isbd ;
  double HSMHV_isbd2 ;
  double HSMHV_isbs ;
  double HSMHV_isbs2 ;
  double HSMHV_vbdt ;
  double HSMHV_vbst ;
  double HSMHV_exptemp ;
  double HSMHV_wsti ;
  double HSMHV_cnstpgd ;
  /* double HSMHV_ninvp0 ;	not used */
  /* double HSMHV_ninv0 ;	not used */
  double HSMHV_grbpb ;
  double HSMHV_grbpd ;
  double HSMHV_grbps ;
  double HSMHV_grg ;
  double HSMHV_rs ;
  double HSMHV_rs0 ;
  double HSMHV_rd ;
  double HSMHV_rd0 ;
  double HSMHV_rdtemp0 ;
  double HSMHV_clmmod ;
  double HSMHV_lgatesm ;
  double HSMHV_dVthsm ;
  double HSMHV_ddlt ;
  double HSMHV_xsub1 ;
  double HSMHV_xsub2 ;
  double HSMHV_xgate ;
  double HSMHV_xvbs ;
  double HSMHV_vg2const ;
  double HSMHV_wdpl ;
  double HSMHV_wdplp ;
  double HSMHV_cfrng ;
  double HSMHV_jd_nvtm_inv ;
  double HSMHV_jd_expcd ;
  double HSMHV_jd_expcs ;
  double HSMHV_sqrt_eg ;

  double HSMHV_egtnom ;
  double HSMHV_cecox ;
  double HSMHV_msc ;
  int HSMHV_flg_pgd ;
  double HSMHV_ndep_o_esi ;
  double HSMHV_ninv_o_esi ;
  double HSMHV_cqyb0 ;
  double HSMHV_cnst0over ;
  double HSMHV_cnst0overs ;
  double HSMHV_costi00 ;
  double HSMHV_nsti_p2 ;
  double HSMHV_costi0 ;
  double HSMHV_costi0_p2 ;
  double HSMHV_costi1 ;
  double HSMHV_ps0ldinib ;
  double HSMHV_ps0ldinibs ;
  double HSMHV_rdvdtemp0 ;
  double HSMHV_rthtemp0 ;
  double HSMHV_powratio ;

  double HSMHV_mueph1 ;
  double HSMHV_nsubp;
  double HSMHV_nsubc;


  HSMHVhereMKSParam hereMKS ; /* unit-converted parameters */

  HSMHVbinningParam pParam ; /* binning parameters */
  
  /* no use in SPICE3f5
      double HSMHVdrainSquares;       the length of the drain in squares
      double HSMHVsourceSquares;      the length of the source in squares */
  double HSMHVsourceConductance; /* cond. of source (or 0): set in setup */
  double HSMHVdrainConductance;  /* cond. of drain (or 0): set in setup */

  double HSMHV_icVBS; /* initial condition B-S voltage */
  double HSMHV_icVDS; /* initial condition D-S voltage */
  double HSMHV_icVGS; /* initial condition G-S voltage */
  int HSMHV_off;      /* non-zero to indicate device is off for dc analysis */
  int HSMHV_mode;     /* device mode : 1 = normal, -1 = inverse */

  unsigned HSMHV_coselfheat_Given :1;
  unsigned HSMHV_cosubnode_Given :1;
  unsigned HSMHV_l_Given :1;
  unsigned HSMHV_w_Given :1;
  unsigned HSMHV_ad_Given :1;
  unsigned HSMHV_as_Given    :1;
  unsigned HSMHV_pd_Given    :1;
  unsigned HSMHV_ps_Given   :1;
  unsigned HSMHV_nrd_Given  :1;
  unsigned HSMHV_nrs_Given  :1;
  unsigned HSMHV_dtemp_Given  :1;
  unsigned HSMHV_icVBS_Given :1;
  unsigned HSMHV_icVDS_Given :1;
  unsigned HSMHV_icVGS_Given :1;
  unsigned HSMHV_corbnet_Given  :1;
  unsigned HSMHV_rbpb_Given :1;
  unsigned HSMHV_rbpd_Given :1;
  unsigned HSMHV_rbps_Given :1;
  unsigned HSMHV_rbdb_Given :1;
  unsigned HSMHV_rbsb_Given :1;
  unsigned HSMHV_corg_Given  :1;
  unsigned HSMHV_ngcon_Given  :1;
  unsigned HSMHV_xgw_Given  :1;
  unsigned HSMHV_xgl_Given  :1;
  unsigned HSMHV_nf_Given  :1;
  unsigned HSMHV_sa_Given  :1;
  unsigned HSMHV_sb_Given  :1;
  unsigned HSMHV_sd_Given  :1;
  unsigned HSMHV_nsubcdfm_Given  :1;
  unsigned HSMHV_m_Given  :1;
  unsigned HSMHV_subld1_Given  :1;
  unsigned HSMHV_subld2_Given  :1;
  unsigned HSMHV_lover_Given  :1;
  unsigned HSMHV_lovers_Given  :1;
  unsigned HSMHV_loverld_Given  :1;
  unsigned HSMHV_ldrift1_Given  :1;
  unsigned HSMHV_ldrift2_Given  :1;
  unsigned HSMHV_ldrift1s_Given :1;
  unsigned HSMHV_ldrift2s_Given :1;

  /* unsigned HSMHV_rth0_Given :1;	not used */
  /* unsigned HSMHV_cth0_Given :1;	not used */

  

  /* pointers to sparse matrix */

  double *HSMHVGgPtr;   /* pointer to sparse matrix element at (gate node,gate node) */
  double *HSMHVGgpPtr;  /* pointer to sparse matrix element at (gate node,gate prime node) */
  /* double *HSMHVGdpPtr;	not used */
  /* double *HSMHVGspPtr;	not used */
  /* double *HSMHVGbpPtr;	not used */

  double *HSMHVGPgPtr;  /* pointer to sparse matrix element at (gate prime node,gate node) */
  double *HSMHVGPgpPtr; /* pointer to sparse matrix element at (gate prime node,gate prime node) */
  double *HSMHVGPdpPtr; /* pointer to sparse matrix element at (gate prime node,drain prime node) */
  double *HSMHVGPspPtr; /* pointer to sparse matrix element at (gate prime node,source prime node) */
  double *HSMHVGPbpPtr; /* pointer to sparse matrix element at (gate prime node,bulk prime node) */

  double *HSMHVDPdPtr;  /* pointer to sparse matrix element at (drain prime node,drain node) */
  double *HSMHVDPdpPtr; /* pointer to sparse matrix element at (drain prime node,drain prime node) */
  double *HSMHVDPgpPtr; /* pointer to sparse matrix element at (drain prime node,gate prime node) */
  double *HSMHVDPspPtr; /* pointer to sparse matrix element at (drain prime node,source prime node) */
  double *HSMHVDPbpPtr; /* pointer to sparse matrix element at (drain prime node,bulk prime node) */

  double *HSMHVDdPtr;   /* pointer to sparse matrix element at (Drain node,drain node) */
  double *HSMHVDdpPtr;  /* pointer to sparse matrix element at (drain node,drain prime node) */
  double *HSMHVDspPtr;  /* pointer to sparse matrix element at (drain node,source prime node) */
  double *HSMHVDdbPtr;  /* pointer to sparse matrix element at (drain node,drain body node) */

  double *HSMHVSPsPtr;  /* pointer to sparse matrix element at (source prime node,source node) */
  double *HSMHVSPspPtr; /* pointer to sparse matrix element at (source prime node,source prime node) */
  double *HSMHVSPgpPtr; /* pointer to sparse matrix element at (source prime node,gate prime node) */
  double *HSMHVSPdpPtr; /* pointer to sparse matrix element at (source prime node,drain prime node) */
  double *HSMHVSPbpPtr; /* pointer to sparse matrix element at (source prime node,bulk prime node) */

  double *HSMHVSsPtr;   /* pointer to sparse matrix element at (source node,source node) */
  double *HSMHVSspPtr;  /* pointer to sparse matrix element at (source node,source prime node) */
  double *HSMHVSdpPtr;  /* pointer to sparse matrix element at (source node,drain prime node) */
  double *HSMHVSsbPtr;  /* pointer to sparse matrix element at (source node,source body node) */

  double *HSMHVBPgpPtr; /* pointer to sparse matrix element at (bulk prime node,gate prime node) */
  double *HSMHVBPbpPtr; /* pointer to sparse matrix element at (bulk prime node,bulk prime node) */
  double *HSMHVBPdPtr;  /* pointer to sparse matrix element at (bulk prime node,drain node) */
  double *HSMHVBPdpPtr; /* pointer to sparse matrix element at (bulk prime node,drain prime node) */
  double *HSMHVBPspPtr; /* pointer to sparse matrix element at (bulk prime node,source prime node) */
  double *HSMHVBPsPtr;  /* pointer to sparse matrix element at (bulk prime node,source node) */
  double *HSMHVBPbPtr;  /* pointer to sparse matrix element at (bulk prime node,bulk node) */
  double *HSMHVBPdbPtr; /* pointer to sparse matrix element at (bulk prime node,source body node) */
  double *HSMHVBPsbPtr; /* pointer to sparse matrix element at (bulk prime node,source body node) */

  double *HSMHVDBdPtr;  /* pointer to sparse matrix element at (drain body node,drain node) */
  double *HSMHVDBdbPtr; /* pointer to sparse matrix element at (drain body node,drain body node) */
  double *HSMHVDBbpPtr; /* pointer to sparse matrix element at (drain body node,bulk prime node) */
  /* double *HSMHVDBbPtr;	not used */
  
  double *HSMHVSBsPtr;  /* pointer to sparse matrix element at (source body node,source node) */
  double *HSMHVSBbpPtr; /* pointer to sparse matrix element at (source body node,bulk prime node) */
  /* double *HSMHVSBbPtr;	not used */
  double *HSMHVSBsbPtr; /* pointer to sparse matrix element at (source body node,source body node) */

  /* double *HSMHVBsbPtr;	not used */
  double *HSMHVBbpPtr;  /* pointer to sparse matrix element at (bulk node,bulk prime node) */
  /* double *HSMHVBdbPtr;	not used */
  double *HSMHVBbPtr;   /* pointer to sparse matrix element at (bulk node,bulk node) */
  
  double *HSMHVTemptempPtr; /* pointer to sparse matrix element at (temp node, temp node) */
  double *HSMHVTempdPtr;    /* pointer to sparse matrix element at (temp node, drain node) */
  double *HSMHVTempdpPtr;   /* pointer to sparse matrix element at (temp node, drain prime node) */
  double *HSMHVTempsPtr;    /* pointer to sparse matrix element at (temp node, source node) */
  double *HSMHVTempspPtr;   /* pointer to sparse matrix element at (temp node, source prime node) */
  /* double *HSMHVTempgPtr;	not used */
  double *HSMHVTempgpPtr;   /* pointer to sparse matrix element at (temp node, gate prime node) */
  /* double *HSMHVTempbPtr;	not used */
  double *HSMHVTempbpPtr;   /* pointer to sparse matrix element at (temp node, bulk prime node) */
  /* double *HSMHVGtempPtr;	not used */
  double *HSMHVGPtempPtr;   /* pointer to sparse matrix element at (gate prime node, temp node) */
  double *HSMHVDPtempPtr;   /* pointer to sparse matrix element at (drain prime node, temp node) */
  double *HSMHVSPtempPtr;   /* pointer to sparse matrix element at (source prime node, temp node) */
  /* double *HSMHVBtempPtr;	not used */
  double *HSMHVBPtempPtr;   /* pointer to sparse matrix element at (bulk prime node, temp node) */
  double *HSMHVDBtempPtr;   /* pointer to sparse matrix element at (drain bulk node, temp node) */
  double *HSMHVSBtempPtr;   /* pointer to sparse matrix element at (source bulk node, temp node) */

  double *HSMHVDgpPtr;      /* pointer to sparse matrix element at (drain node, gate prime node) */
  double *HSMHVDsPtr;       /* pointer to sparse matrix element at (drain node, source node) */
  double *HSMHVDbpPtr;      /* pointer to sparse matrix element at (drain node, bulk prime node) */
  double *HSMHVDtempPtr;    /* pointer to sparse matrix element at (drain node, temp node) */
  double *HSMHVDPsPtr;      /* pointer to sparse matrix element at (drain prime node, source node) */
  double *HSMHVGPdPtr;      /* pointer to sparse matrix element at (gate prime node, drain node) */
  double *HSMHVGPsPtr;      /* pointer to sparse matrix element at (gate prime node, source node) */
  double *HSMHVSdPtr;       /* pointer to sparse matrix element at (source node, drain node) */
  double *HSMHVSgpPtr;      /* pointer to sparse matrix element at (source node, gate prime node) */
  double *HSMHVSbpPtr;      /* pointer to sparse matrix element at (source node, bulk prime node) */
  double *HSMHVStempPtr;    /* pointer to sparse matrix element at (source node, temp node) */
  double *HSMHVSPdPtr;      /* pointer to sparse matrix element at (source prime node, drain node) */

  /* nqs related pointers */
  double *HSMHVDPqiPtr;     /* pointer to sparse matrix element at (drain prime node, qi_nqs node) */
  double *HSMHVGPqiPtr;     /* pointer to sparse matrix element at (gate prime node, qi_nqs node) */
  double *HSMHVGPqbPtr;     /* pointer to sparse matrix element at (gate prime node, qb_nqs node) */
  double *HSMHVSPqiPtr;     /* pointer to sparse matrix element at (source prime node, qi_nqs node) */
  double *HSMHVBPqbPtr;     /* pointer to sparse matrix element at (bulk prime node, qb_nqs node) */
  double *HSMHVQIdpPtr;     /* pointer to sparse matrix element at (qi_nqs node, drain prime node) */
  double *HSMHVQIgpPtr;     /* pointer to sparse matrix element at (qi_nqs node, gate prime node) */
  double *HSMHVQIspPtr;     /* pointer to sparse matrix element at (qi_nqs node, source prime node) */
  double *HSMHVQIbpPtr;     /* pointer to sparse matrix element at (qi_nqs node, bulk prime node) */
  double *HSMHVQIqiPtr;     /* pointer to sparse matrix element at (qi_nqs node, qi_nqs node) */
  double *HSMHVQBdpPtr;     /* pointer to sparse matrix element at (qb_nqs node, drain prime node) */
  double *HSMHVQBgpPtr;     /* pointer to sparse matrix element at (qb_nqs node, gate prime node) */
  double *HSMHVQBspPtr;     /* pointer to sparse matrix element at (qb_nqs node, source prime node) */
  double *HSMHVQBbpPtr;     /* pointer to sparse matrix element at (qb_nqs node, bulk prime node) */
  double *HSMHVQBqbPtr;     /* pointer to sparse matrix element at (qb_nqs node, qb_nqs node) */
  double *HSMHVQItempPtr;   /* pointer to sparse matrix element at (qi_nqs node, temp node) */
  double *HSMHVQBtempPtr;   /* pointer to sparse matrix element at (qb_nqs node, temp node) */

  /* Substrate effect related pointers */
  double *HSMHVDsubPtr;     /* pointer to sparse matrix element at (drain node, substrate node) */
  double *HSMHVDPsubPtr;    /* pointer to sparse matrix element at (drain prime node, substrate node) */
  double *HSMHVSsubPtr;     /* pointer to sparse matrix element at (source node, substrate node) */
  double *HSMHVSPsubPtr;    /* pointer to sparse matrix element at (source prime node, substrate node) */


  /* common state values in hisim module */
#define HSMHVvbd HSMHVstates+ 0
#define HSMHVvbs HSMHVstates+ 1
#define HSMHVvgs HSMHVstates+ 2
#define HSMHVvds HSMHVstates+ 3
#define HSMHVvdbs HSMHVstates+ 4
#define HSMHVvdbd HSMHVstates+ 5
#define HSMHVvsbs HSMHVstates+ 6
#define HSMHVvges HSMHVstates+ 7
#define HSMHVvsubs HSMHVstates+ 8 /* substrate bias */
#define HSMHVdeltemp HSMHVstates+ 9
#define HSMHVvdse HSMHVstates+ 10
#define HSMHVvgse HSMHVstates+ 11
#define HSMHVvbse HSMHVstates+ 12

#define HSMHVqb  HSMHVstates+ 13
#define HSMHVcqb HSMHVstates+ 14
#define HSMHVqg  HSMHVstates+ 15
#define HSMHVcqg HSMHVstates+ 16
#define HSMHVqd  HSMHVstates+ 17
#define HSMHVcqd HSMHVstates+ 18

#define HSMHVqbs HSMHVstates+ 19
#define HSMHVcqbs HSMHVstates+ 20
#define HSMHVqbd HSMHVstates+ 21
#define HSMHVcqbd HSMHVstates+ 22

#define HSMHVqth HSMHVstates+ 23
#define HSMHVcqth HSMHVstates+ 24

/*add fringing capacitance*/
#define HSMHVqfd HSMHVstates+ 25
#define HSMHVcqfd HSMHVstates+ 26
#define HSMHVqfs HSMHVstates+ 27
#define HSMHVcqfs HSMHVstates+ 28

/*add external drain capacitance*/
#define HSMHVqdE HSMHVstates+ 29
#define HSMHVcqdE HSMHVstates+ 30

#define HSMHVnumStates 31

/* nqs charges */
#define HSMHVqi_nqs HSMHVstates+ 32
#define HSMHVdotqi_nqs HSMHVstates + 33
#define HSMHVqb_nqs HSMHVstates+ 34
#define HSMHVdotqb_nqs HSMHVstates + 35

#define HSMHVnumStatesNqs 36

/* indices to the array of HiSIMHV NOISE SOURCES */
#define HSMHVRDNOIZ       0
#define HSMHVRSNOIZ       1
#define HSMHVIDNOIZ       2
#define HSMHVFLNOIZ       3
#define HSMHVIGNOIZ       4
#define HSMHVTOTNOIZ      5

#define HSMHVNSRCS        6  /* the number of HiSIMHV MOSFET noise sources */

#ifndef NONOISE
  double HSMHVnVar[NSTATVARS][HSMHVNSRCS];
#else /* NONOISE */
  double **HSMHVnVar;
#endif /* NONOISE */

} HSMHVinstance ;


/* per model data */

typedef struct sHSMHVmodel {     /* model structure for a resistor */

  struct GENmodel gen;

#define HSMHVmodType gen.GENmodType
#define HSMHVnextModel(inst) ((struct sHSMHVmodel *)((inst)->gen.GENnextModel))
#define HSMHVinstances(inst) ((HSMHVinstance *)((inst)->gen.GENinstances))
#define HSMHVmodName gen.GENmodName

  int HSMHV_type;      		/* device type: 1 = nmos,  -1 = pmos */
  int HSMHV_level;               /* level */
  int HSMHV_info;                /* information */
  int HSMHV_noise;               /* noise model selecter see hsmhvnoi.c */
  char *HSMHV_version;           /* model version */
  int HSMHV_show;                /* show physical value 1, 2, ... , 11 */


  int HSMHV_corsrd ;
  int HSMHV_corg   ;
  int HSMHV_coiprv ;
  int HSMHV_copprv ;
  int HSMHV_coadov ;
  int HSMHV_coisub ;
  int HSMHV_coiigs ;
  int HSMHV_cogidl ;
  int HSMHV_coovlp ;
  int HSMHV_coovlps ;
  int HSMHV_coflick ;
  int HSMHV_coisti ;
  int HSMHV_conqs  ;
  int HSMHV_corbnet ;
  int HSMHV_cothrml;
  int HSMHV_coign;      /* Induced gate noise */
  int HSMHV_codfm;      /* DFM */
  int HSMHV_coqovsm ;
  int HSMHV_coselfheat; /* Self-heating model */
  int HSMHV_cosubnode;  /* switch tempNode to subNode */
  int HSMHV_cosym;      /* Symmetry model for HV */
  int HSMHV_cotemp;
  int HSMHV_coldrift;


  double HSMHV_vmax ;
  double HSMHV_vmaxt1 ;
  double HSMHV_vmaxt2 ;
  double HSMHV_bgtmp1 ;
  double HSMHV_bgtmp2 ;
  double HSMHV_eg0 ;
  double HSMHV_tox ;
  double HSMHV_xld ;
  double HSMHV_xldld ;
  double HSMHV_xwdld ;
  double HSMHV_lover ;
  double HSMHV_lovers ;
  double HSMHV_rdov11 ;
  double HSMHV_rdov12 ;
  double HSMHV_rdov13 ;
  double HSMHV_rdslp1 ;
  double HSMHV_rdict1 ;
  double HSMHV_rdslp2 ;
  double HSMHV_rdict2 ;
  double HSMHV_loverld ;
  double HSMHV_ldrift1 ;
  double HSMHV_ldrift2 ;
  double HSMHV_ldrift1s ;
  double HSMHV_ldrift2s ;
  double HSMHV_subld1 ;
  double HSMHV_subld2 ;
  double HSMHV_ddltmax ;
  double HSMHV_ddltslp ;
  double HSMHV_ddltict ;
  double HSMHV_vfbover ;
  double HSMHV_nover ;
  double HSMHV_novers ;
  double HSMHV_xwd ;
  double HSMHV_xwdc ;
  double HSMHV_xl ;
  double HSMHV_xw ;
  double HSMHV_saref ;
  double HSMHV_sbref ;
  double HSMHV_ll ;
  double HSMHV_lld ;
  double HSMHV_lln ;
  double HSMHV_wl ;
  double HSMHV_wl1 ;
  double HSMHV_wl1p ;
  double HSMHV_wl2 ;
  double HSMHV_wl2p ;
  double HSMHV_wld ;
  double HSMHV_wln ;
  double HSMHV_xqy ;
  double HSMHV_xqy1 ;
  double HSMHV_xqy2 ;
  double HSMHV_rs;     /* source contact resistance */
  double HSMHV_rd;     /* drain contact resistance */
  double HSMHV_rsh;    /* source/drain diffusion sheet resistance */
  double HSMHV_rshg;
/*   double HSMHV_ngcon; */
/*   double HSMHV_xgw; */
/*   double HSMHV_xgl; */
/*   double HSMHV_nf; */
  double HSMHV_vfbc ;
  double HSMHV_vbi ;
  double HSMHV_nsubc ;
  double HSMHV_qdftvd ;
  double HSMHV_parl2 ;
  double HSMHV_lp ;
  double HSMHV_nsubp ;
  double HSMHV_nsubp0 ;
  double HSMHV_nsubwp ;
  double HSMHV_scp1 ;
  double HSMHV_scp2 ;
  double HSMHV_scp3 ;
  double HSMHV_sc1 ;
  double HSMHV_sc2 ;
  double HSMHV_sc3 ;
  double HSMHV_sc4 ;
  double HSMHV_pgd1 ;
  double HSMHV_pgd2 ;
  double HSMHV_pgd3 ;
  double HSMHV_pgd4 ;
  double HSMHV_ndep ;
  double HSMHV_ndepl ;
  double HSMHV_ndeplp ;
  double HSMHV_ninv ;
  double HSMHV_ninvd ;
  double HSMHV_ninvdw ;
  double HSMHV_ninvdwp ;
  double HSMHV_ninvdt1 ;
  double HSMHV_ninvdt2 ;
  double HSMHV_muecb0 ;
  double HSMHV_muecb1 ;
  double HSMHV_mueph1 ;
  double HSMHV_mueph0 ;
  double HSMHV_muephw ;
  double HSMHV_muepwp ;
  double HSMHV_muephl ;
  double HSMHV_mueplp ;
  double HSMHV_muephs ;
  double HSMHV_muepsp ;
  double HSMHV_vtmp ;
  double HSMHV_wvth0 ;
  double HSMHV_muesr1 ;
  double HSMHV_muesr0 ;
  double HSMHV_muesrw ;
  double HSMHV_mueswp ;
  double HSMHV_muesrl ;
  double HSMHV_mueslp ;
  double HSMHV_bb ;
  double HSMHV_sub1 ;
  double HSMHV_sub2 ;
  double HSMHV_svgs ;
  double HSMHV_svbs ;
  double HSMHV_svbsl ;
  double HSMHV_svds ;
  double HSMHV_slg ;
  double HSMHV_sub1l ;
  double HSMHV_sub2l ;
  double HSMHV_fn1 ;
  double HSMHV_fn2 ;
  double HSMHV_fn3 ;
  double HSMHV_fvbs ;
  double HSMHV_svgsl ;
  double HSMHV_svgslp ;
  double HSMHV_svgswp ;
  double HSMHV_svgsw ;
  double HSMHV_svbslp ;
  double HSMHV_slgl ;
  double HSMHV_slglp ;
  double HSMHV_sub1lp ;
  double HSMHV_nsti ;  
  double HSMHV_wsti ;
  double HSMHV_wstil ;
  double HSMHV_wstilp ;
  double HSMHV_wstiw ;
  double HSMHV_wstiwp ;
  double HSMHV_scsti1 ;
  double HSMHV_scsti2 ;
  double HSMHV_vthsti ;
  double HSMHV_vdsti ;
  double HSMHV_muesti1 ;
  double HSMHV_muesti2 ;
  double HSMHV_muesti3 ;
  double HSMHV_nsubpsti1 ;
  double HSMHV_nsubpsti2 ;
  double HSMHV_nsubpsti3 ;
  double HSMHV_lpext ;
  double HSMHV_npext ;
  double HSMHV_scp22 ;
  double HSMHV_scp21 ;
  double HSMHV_bs1 ;
  double HSMHV_bs2 ;
  double HSMHV_cgso ;
  double HSMHV_cgdo ;
  double HSMHV_cgbo ;
  double HSMHV_tpoly ;
  double HSMHV_js0 ;
  double HSMHV_js0sw ;
  double HSMHV_nj ;
  double HSMHV_njsw ;
  double HSMHV_xti ;
  double HSMHV_cj ;
  double HSMHV_cjsw ;
  double HSMHV_cjswg ;
  double HSMHV_mj ;
  double HSMHV_mjsw ;
  double HSMHV_mjswg ;
  double HSMHV_xti2 ;
  double HSMHV_cisb ;
  double HSMHV_cvb ;
  double HSMHV_ctemp ;
  double HSMHV_cisbk ;
  double HSMHV_cvbk ;
  double HSMHV_divx ;
  double HSMHV_pb ;
  double HSMHV_pbsw ;
  double HSMHV_pbswg ;
  double HSMHV_clm1 ;
  double HSMHV_clm2 ;
  double HSMHV_clm3 ;
  double HSMHV_clm5 ;
  double HSMHV_clm6 ;
  double HSMHV_muetmp ;
  double HSMHV_vover ;
  double HSMHV_voverp ;
  double HSMHV_vovers ;
  double HSMHV_voversp ;
  double HSMHV_wfc ;
  double HSMHV_nsubcw ;
  double HSMHV_nsubcwp ;
  double HSMHV_qme1 ;
  double HSMHV_qme2 ;
  double HSMHV_qme3 ;
  double HSMHV_gidl1 ;
  double HSMHV_gidl2 ;
  double HSMHV_gidl3 ;
  double HSMHV_gidl4 ;
  double HSMHV_gidl5 ;
  double HSMHV_gleak1 ;
  double HSMHV_gleak2 ;
  double HSMHV_gleak3 ;
  double HSMHV_gleak4 ;
  double HSMHV_gleak5 ;
  double HSMHV_gleak6 ;
  double HSMHV_gleak7 ;
  double HSMHV_glpart1 ;
  double HSMHV_glksd1 ;
  double HSMHV_glksd2 ;
  double HSMHV_glksd3 ;
  double HSMHV_glkb1 ;
  double HSMHV_glkb2 ;
  double HSMHV_glkb3 ;
  double HSMHV_egig;
  double HSMHV_igtemp2;
  double HSMHV_igtemp3;
  double HSMHV_vzadd0 ;
  double HSMHV_pzadd0 ;
  double HSMHV_nftrp ;
  double HSMHV_nfalp ;
  double HSMHV_cit ;
  double HSMHV_falph ;
  double HSMHV_kappa ;  
  double HSMHV_pthrou ;
  double HSMHV_vdiffj ; 
  double HSMHV_dly1 ;
  double HSMHV_dly2 ;
  double HSMHV_dly3 ;
  double HSMHV_dlyov;
  double HSMHV_tnom ;
  double HSMHV_ovslp ;
  double HSMHV_ovmag ;
  /* substrate resistances */
  double HSMHV_gbmin;
  double HSMHV_rbpb ;
  double HSMHV_rbpd ;
  double HSMHV_rbps ;
  double HSMHV_rbdb ;
  double HSMHV_rbsb ;
  /* IBPC */
  double HSMHV_ibpc1 ;
  double HSMHV_ibpc2 ;
  /* DFM */
  double HSMHV_mphdfm ;

  double HSMHV_vbsmin ;
  double HSMHV_rdvg11 ;
  double HSMHV_rdvg12 ;
  double HSMHV_rd20 ;
  double HSMHV_qovsm ; 
  double HSMHV_ldrift ; 
  double HSMHV_rd21 ;
  double HSMHV_rd22 ;
  double HSMHV_rd22d ;
  double HSMHV_rd23 ;
  double HSMHV_rd24 ;
  double HSMHV_rd25 ;
  double HSMHV_rd26 ;
  double HSMHV_rdvdl ;
  double HSMHV_rdvdlp ;
  double HSMHV_rdvds ;
  double HSMHV_rdvdsp ;
  double HSMHV_rd23l ;
  double HSMHV_rd23lp ;
  double HSMHV_rd23s ;
  double HSMHV_rd23sp ;
  double HSMHV_rds ;
  double HSMHV_rdsp ;

  double HSMHV_rdvd ;
  double HSMHV_rdvb ;

  double HSMHV_rdvsub ; /* substrate effect */
  double HSMHV_rdvdsub ; /* substrate effect */
  double HSMHV_ddrift ;  /* substrate effect */
  double HSMHV_vbisub ;  /* substrate effect */
  double HSMHV_nsubsub ; /* substrate effect */

  double HSMHV_rth0 ;
  double HSMHV_cth0 ;
  double HSMHV_powrat ;

  double HSMHV_tcjbd ;
  double HSMHV_tcjbs ;
  double HSMHV_tcjbdsw ;
  double HSMHV_tcjbssw ;
  double HSMHV_tcjbdswg ;
  double HSMHV_tcjbsswg ;


  double HSMHV_rdtemp1 ;
  double HSMHV_rdtemp2 ;
  double HSMHV_rth0r ; /* heat radiation for SHE */
  double HSMHV_rdvdtemp1 ;
  double HSMHV_rdvdtemp2 ;
  double HSMHV_rth0w ;
  double HSMHV_rth0wp ;
  double HSMHV_rth0nf ;

  double HSMHV_rthtemp1 ;
  double HSMHV_rthtemp2 ;
  double HSMHV_prattemp1 ;
  double HSMHV_prattemp2 ;


  double HSMHV_cvdsover ;
  double HSMHV_shemax;

  /* binning parameters */
  double HSMHV_lmin ;
  double HSMHV_lmax ;
  double HSMHV_wmin ;
  double HSMHV_wmax ;
  double HSMHV_lbinn ;
  double HSMHV_wbinn ;

  /* Length dependence */
  double HSMHV_lvmax ;
  double HSMHV_lbgtmp1 ;
  double HSMHV_lbgtmp2 ;
  double HSMHV_leg0 ;
  double HSMHV_lvfbover ;
  double HSMHV_lnover ;
  double HSMHV_lnovers ;
  double HSMHV_lwl2 ;
  double HSMHV_lvfbc ;
  double HSMHV_lnsubc ;
  double HSMHV_lnsubp ;
  double HSMHV_lscp1 ;
  double HSMHV_lscp2 ;
  double HSMHV_lscp3 ;
  double HSMHV_lsc1 ;
  double HSMHV_lsc2 ;
  double HSMHV_lsc3 ;
  double HSMHV_lpgd1 ;
  double HSMHV_lpgd3 ;
  double HSMHV_lndep ;
  double HSMHV_lninv ;
  double HSMHV_lmuecb0 ;
  double HSMHV_lmuecb1 ;
  double HSMHV_lmueph1 ;
  double HSMHV_lvtmp ;
  double HSMHV_lwvth0 ;
  double HSMHV_lmuesr1 ;
  double HSMHV_lmuetmp ;
  double HSMHV_lsub1 ;
  double HSMHV_lsub2 ;
  double HSMHV_lsvds ;
  double HSMHV_lsvbs ;
  double HSMHV_lsvgs ;
  double HSMHV_lfn1 ;
  double HSMHV_lfn2 ;
  double HSMHV_lfn3 ;
  double HSMHV_lfvbs ;
  double HSMHV_lnsti ;
  double HSMHV_lwsti ;
  double HSMHV_lscsti1 ;
  double HSMHV_lscsti2 ;
  double HSMHV_lvthsti ;
  double HSMHV_lmuesti1 ;
  double HSMHV_lmuesti2 ;
  double HSMHV_lmuesti3 ;
  double HSMHV_lnsubpsti1 ;
  double HSMHV_lnsubpsti2 ;
  double HSMHV_lnsubpsti3 ;
  double HSMHV_lcgso ;
  double HSMHV_lcgdo ;
  double HSMHV_ljs0 ;
  double HSMHV_ljs0sw ;
  double HSMHV_lnj ;
  double HSMHV_lcisbk ;
  double HSMHV_lclm1 ;
  double HSMHV_lclm2 ;
  double HSMHV_lclm3 ;
  double HSMHV_lwfc ;
  double HSMHV_lgidl1 ;
  double HSMHV_lgidl2 ;
  double HSMHV_lgleak1 ;
  double HSMHV_lgleak2 ;
  double HSMHV_lgleak3 ;
  double HSMHV_lgleak6 ;
  double HSMHV_lglksd1 ;
  double HSMHV_lglksd2 ;
  double HSMHV_lglkb1 ;
  double HSMHV_lglkb2 ;
  double HSMHV_lnftrp ;
  double HSMHV_lnfalp ;
  double HSMHV_lpthrou ;
  double HSMHV_lvdiffj ;
  double HSMHV_libpc1 ;
  double HSMHV_libpc2 ;
  double HSMHV_lcgbo ;
  double HSMHV_lcvdsover ;
  double HSMHV_lfalph ;
  double HSMHV_lnpext ;
  double HSMHV_lpowrat ;
  double HSMHV_lrd ;
  double HSMHV_lrd22 ;
  double HSMHV_lrd23 ;
  double HSMHV_lrd24 ;
  double HSMHV_lrdict1 ;
  double HSMHV_lrdov13 ;
  double HSMHV_lrdslp1 ;
  double HSMHV_lrdvb ;
  double HSMHV_lrdvd ;
  double HSMHV_lrdvg11 ;
  double HSMHV_lrs ;
  double HSMHV_lrth0 ;
  double HSMHV_lvover ;

  /* Width dependence */
  double HSMHV_wvmax ;
  double HSMHV_wbgtmp1 ;
  double HSMHV_wbgtmp2 ;
  double HSMHV_weg0 ;
  double HSMHV_wvfbover ;
  double HSMHV_wnover ;
  double HSMHV_wnovers ;
  double HSMHV_wwl2 ;
  double HSMHV_wvfbc ;
  double HSMHV_wnsubc ;
  double HSMHV_wnsubp ;
  double HSMHV_wscp1 ;
  double HSMHV_wscp2 ;
  double HSMHV_wscp3 ;
  double HSMHV_wsc1 ;
  double HSMHV_wsc2 ;
  double HSMHV_wsc3 ;
  double HSMHV_wpgd1 ;
  double HSMHV_wpgd3 ;
  double HSMHV_wndep ;
  double HSMHV_wninv ;
  double HSMHV_wmuecb0 ;
  double HSMHV_wmuecb1 ;
  double HSMHV_wmueph1 ;
  double HSMHV_wvtmp ;
  double HSMHV_wwvth0 ;
  double HSMHV_wmuesr1 ;
  double HSMHV_wmuetmp ;
  double HSMHV_wsub1 ;
  double HSMHV_wsub2 ;
  double HSMHV_wsvds ;
  double HSMHV_wsvbs ;
  double HSMHV_wsvgs ;
  double HSMHV_wfn1 ;
  double HSMHV_wfn2 ;
  double HSMHV_wfn3 ;
  double HSMHV_wfvbs ;
  double HSMHV_wnsti ;
  double HSMHV_wwsti ;
  double HSMHV_wscsti1 ;
  double HSMHV_wscsti2 ;
  double HSMHV_wvthsti ;
  double HSMHV_wmuesti1 ;
  double HSMHV_wmuesti2 ;
  double HSMHV_wmuesti3 ;
  double HSMHV_wnsubpsti1 ;
  double HSMHV_wnsubpsti2 ;
  double HSMHV_wnsubpsti3 ;
  double HSMHV_wcgso ;
  double HSMHV_wcgdo ;
  double HSMHV_wjs0 ;
  double HSMHV_wjs0sw ;
  double HSMHV_wnj ;
  double HSMHV_wcisbk ;
  double HSMHV_wclm1 ;
  double HSMHV_wclm2 ;
  double HSMHV_wclm3 ;
  double HSMHV_wwfc ;
  double HSMHV_wgidl1 ;
  double HSMHV_wgidl2 ;
  double HSMHV_wgleak1 ;
  double HSMHV_wgleak2 ;
  double HSMHV_wgleak3 ;
  double HSMHV_wgleak6 ;
  double HSMHV_wglksd1 ;
  double HSMHV_wglksd2 ;
  double HSMHV_wglkb1 ;
  double HSMHV_wglkb2 ;
  double HSMHV_wnftrp ;
  double HSMHV_wnfalp ;
  double HSMHV_wpthrou ;
  double HSMHV_wvdiffj ;
  double HSMHV_wibpc1 ;
  double HSMHV_wibpc2 ;
  double HSMHV_wcgbo ;
  double HSMHV_wcvdsover ;
  double HSMHV_wfalph ;
  double HSMHV_wnpext ;
  double HSMHV_wpowrat ;
  double HSMHV_wrd ;
  double HSMHV_wrd22 ;
  double HSMHV_wrd23 ;
  double HSMHV_wrd24 ;
  double HSMHV_wrdict1 ;
  double HSMHV_wrdov13 ;
  double HSMHV_wrdslp1 ;
  double HSMHV_wrdvb ;
  double HSMHV_wrdvd ;
  double HSMHV_wrdvg11 ;
  double HSMHV_wrs ;
  double HSMHV_wrth0 ;
  double HSMHV_wvover ;

  /* Cross-term dependence */
  double HSMHV_pvmax ;
  double HSMHV_pbgtmp1 ;
  double HSMHV_pbgtmp2 ;
  double HSMHV_peg0 ;
  double HSMHV_pvfbover ;
  double HSMHV_pnover ;
  double HSMHV_pnovers ;
  double HSMHV_pwl2 ;
  double HSMHV_pvfbc ;
  double HSMHV_pnsubc ;
  double HSMHV_pnsubp ;
  double HSMHV_pscp1 ;
  double HSMHV_pscp2 ;
  double HSMHV_pscp3 ;
  double HSMHV_psc1 ;
  double HSMHV_psc2 ;
  double HSMHV_psc3 ;
  double HSMHV_ppgd1 ;
  double HSMHV_ppgd3 ;
  double HSMHV_pndep ;
  double HSMHV_pninv ;
  double HSMHV_pmuecb0 ;
  double HSMHV_pmuecb1 ;
  double HSMHV_pmueph1 ;
  double HSMHV_pvtmp ;
  double HSMHV_pwvth0 ;
  double HSMHV_pmuesr1 ;
  double HSMHV_pmuetmp ;
  double HSMHV_psub1 ;
  double HSMHV_psub2 ;
  double HSMHV_psvds ;
  double HSMHV_psvbs ;
  double HSMHV_psvgs ;
  double HSMHV_pfn1 ;
  double HSMHV_pfn2 ;
  double HSMHV_pfn3 ;
  double HSMHV_pfvbs ;
  double HSMHV_pnsti ;
  double HSMHV_pwsti ;
  double HSMHV_pscsti1 ;
  double HSMHV_pscsti2 ;
  double HSMHV_pvthsti ;
  double HSMHV_pmuesti1 ;
  double HSMHV_pmuesti2 ;
  double HSMHV_pmuesti3 ;
  double HSMHV_pnsubpsti1 ;
  double HSMHV_pnsubpsti2 ;
  double HSMHV_pnsubpsti3 ;
  double HSMHV_pcgso ;
  double HSMHV_pcgdo ;
  double HSMHV_pjs0 ;
  double HSMHV_pjs0sw ;
  double HSMHV_pnj ;
  double HSMHV_pcisbk ;
  double HSMHV_pclm1 ;
  double HSMHV_pclm2 ;
  double HSMHV_pclm3 ;
  double HSMHV_pwfc ;
  double HSMHV_pgidl1 ;
  double HSMHV_pgidl2 ;
  double HSMHV_pgleak1 ;
  double HSMHV_pgleak2 ;
  double HSMHV_pgleak3 ;
  double HSMHV_pgleak6 ;
  double HSMHV_pglksd1 ;
  double HSMHV_pglksd2 ;
  double HSMHV_pglkb1 ;
  double HSMHV_pglkb2 ;
  double HSMHV_pnftrp ;
  double HSMHV_pnfalp ;
  double HSMHV_ppthrou ;
  double HSMHV_pvdiffj ;
  double HSMHV_pibpc1 ;
  double HSMHV_pibpc2 ;
  double HSMHV_pcgbo ;
  double HSMHV_pcvdsover ;
  double HSMHV_pfalph ;
  double HSMHV_pnpext ;
  double HSMHV_ppowrat ;
  double HSMHV_prd ;
  double HSMHV_prd22 ;
  double HSMHV_prd23 ;
  double HSMHV_prd24 ;
  double HSMHV_prdict1 ;
  double HSMHV_prdov13 ;
  double HSMHV_prdslp1 ;
  double HSMHV_prdvb ;
  double HSMHV_prdvd ;
  double HSMHV_prdvg11 ;
  double HSMHV_prs ;
  double HSMHV_prth0 ;
  double HSMHV_pvover ;

  /* internal variables */
  double HSMHV_vcrit ;
  int HSMHV_flg_qme ;
  double HSMHV_qme12 ;
  double HSMHV_ktnom ;

  double HSMHVvgsMax;
  double HSMHVvgdMax;
  double HSMHVvgbMax;
  double HSMHVvdsMax;
  double HSMHVvbsMax;
  double HSMHVvbdMax;
  double HSMHVvgsrMax;
  double HSMHVvgdrMax;
  double HSMHVvgbrMax;
  double HSMHVvbsrMax;
  double HSMHVvbdrMax;

  HSMHVmodelMKSParam modelMKS ; /* unit-converted parameters */


  /* flag for model */
  unsigned HSMHV_type_Given  :1;
  unsigned HSMHV_level_Given  :1;
  unsigned HSMHV_info_Given  :1;
  unsigned HSMHV_noise_Given :1;
  unsigned HSMHV_version_Given :1;
  unsigned HSMHV_show_Given :1;
  unsigned HSMHV_corsrd_Given  :1;
  unsigned HSMHV_corg_Given    :1;
  unsigned HSMHV_coiprv_Given  :1;
  unsigned HSMHV_copprv_Given  :1;
  unsigned HSMHV_coadov_Given  :1;
  unsigned HSMHV_coisub_Given  :1;
  unsigned HSMHV_coiigs_Given  :1;
  unsigned HSMHV_cogidl_Given  :1;
  unsigned HSMHV_coovlp_Given  :1;
  unsigned HSMHV_coovlps_Given  :1;
  unsigned HSMHV_coflick_Given  :1;
  unsigned HSMHV_coisti_Given  :1;
  unsigned HSMHV_conqs_Given  :1;
  unsigned HSMHV_corbnet_Given  :1;
  unsigned HSMHV_cothrml_Given  :1;
  unsigned HSMHV_coign_Given  :1;      /* Induced gate noise */
  unsigned HSMHV_codfm_Given  :1;      /* DFM */
  unsigned HSMHV_coqovsm_Given  :1;
  unsigned HSMHV_coselfheat_Given  :1; /* Self-heating model */
  unsigned HSMHV_cosubnode_Given  :1;  /* switch tempNode to subNode */
  unsigned HSMHV_cosym_Given  :1;      /* Symmetry model for HV */
  unsigned HSMHV_cotemp_Given  :1;
  unsigned HSMHV_coldrift_Given  :1;
  unsigned HSMHV_kappa_Given :1;  
  unsigned HSMHV_pthrou_Given :1;
  unsigned HSMHV_vdiffj_Given :1; 
  unsigned HSMHV_vmax_Given  :1;
  unsigned HSMHV_vmaxt1_Given  :1;
  unsigned HSMHV_vmaxt2_Given  :1;
  unsigned HSMHV_bgtmp1_Given  :1;
  unsigned HSMHV_bgtmp2_Given  :1;
  unsigned HSMHV_eg0_Given  :1;
  unsigned HSMHV_tox_Given  :1;
  unsigned HSMHV_xld_Given  :1;
  unsigned HSMHV_xldld_Given  :1;
  unsigned HSMHV_xwdld_Given  :1;
  unsigned HSMHV_lover_Given  :1;
  unsigned HSMHV_lovers_Given  :1;
  unsigned HSMHV_rdov11_Given  :1;
  unsigned HSMHV_rdov12_Given  :1;
  unsigned HSMHV_rdov13_Given  :1;
  unsigned HSMHV_rdslp1_Given  :1;
  unsigned HSMHV_rdict1_Given  :1;
  unsigned HSMHV_rdslp2_Given  :1;
  unsigned HSMHV_rdict2_Given  :1;
  unsigned HSMHV_loverld_Given  :1;
  unsigned HSMHV_ldrift1_Given  :1;
  unsigned HSMHV_ldrift2_Given  :1;
  unsigned HSMHV_ldrift1s_Given  :1;
  unsigned HSMHV_ldrift2s_Given  :1;
  unsigned HSMHV_subld1_Given  :1;
  unsigned HSMHV_subld2_Given  :1;
  unsigned HSMHV_ddltmax_Given  :1;
  unsigned HSMHV_ddltslp_Given  :1;
  unsigned HSMHV_ddltict_Given  :1;
  unsigned HSMHV_vfbover_Given  :1;
  unsigned HSMHV_nover_Given  :1;
  unsigned HSMHV_novers_Given  :1;
  unsigned HSMHV_xwd_Given  :1; 
  unsigned HSMHV_xwdc_Given  :1; 
  unsigned HSMHV_xl_Given  :1;
  unsigned HSMHV_xw_Given  :1;
  unsigned HSMHV_saref_Given  :1;
  unsigned HSMHV_sbref_Given  :1;
  unsigned HSMHV_ll_Given  :1;
  unsigned HSMHV_lld_Given  :1;
  unsigned HSMHV_lln_Given  :1;
  unsigned HSMHV_wl_Given  :1;
  unsigned HSMHV_wl1_Given  :1;
  unsigned HSMHV_wl1p_Given  :1;
  unsigned HSMHV_wl2_Given  :1;
  unsigned HSMHV_wl2p_Given  :1;
  unsigned HSMHV_wld_Given  :1;
  unsigned HSMHV_wln_Given  :1; 
  unsigned HSMHV_xqy_Given  :1;   
  unsigned HSMHV_xqy1_Given  :1;   
  unsigned HSMHV_xqy2_Given  :1;   
  unsigned HSMHV_rs_Given  :1;
  unsigned HSMHV_rd_Given  :1;
  unsigned HSMHV_rsh_Given  :1;
  unsigned HSMHV_rshg_Given  :1;
/*   unsigned HSMHV_ngcon_Given  :1; */
/*   unsigned HSMHV_xgw_Given  :1; */
/*   unsigned HSMHV_xgl_Given  :1; */
/*   unsigned HSMHV_nf_Given  :1; */
  unsigned HSMHV_vfbc_Given  :1;
  unsigned HSMHV_vbi_Given  :1;
  unsigned HSMHV_nsubc_Given  :1;
  unsigned HSMHV_parl2_Given  :1;
  unsigned HSMHV_lp_Given  :1;
  unsigned HSMHV_nsubp_Given  :1;
  unsigned HSMHV_nsubp0_Given  :1;
  unsigned HSMHV_nsubwp_Given  :1;
  unsigned HSMHV_scp1_Given  :1;
  unsigned HSMHV_scp2_Given  :1;
  unsigned HSMHV_scp3_Given  :1;
  unsigned HSMHV_sc1_Given  :1;
  unsigned HSMHV_sc2_Given  :1;
  unsigned HSMHV_sc3_Given  :1;
  unsigned HSMHV_sc4_Given  :1;
  unsigned HSMHV_pgd1_Given  :1;
  unsigned HSMHV_pgd2_Given  :1;
  unsigned HSMHV_pgd3_Given  :1;
  unsigned HSMHV_pgd4_Given  :1;
  unsigned HSMHV_ndep_Given  :1;
  unsigned HSMHV_ndepl_Given  :1;
  unsigned HSMHV_ndeplp_Given  :1;
  unsigned HSMHV_ninv_Given  :1;
  unsigned HSMHV_muecb0_Given  :1;
  unsigned HSMHV_muecb1_Given  :1;
  unsigned HSMHV_mueph1_Given  :1;
  unsigned HSMHV_mueph0_Given  :1;
  unsigned HSMHV_muephw_Given  :1;
  unsigned HSMHV_muepwp_Given  :1;
  unsigned HSMHV_muephl_Given  :1;
  unsigned HSMHV_mueplp_Given  :1;
  unsigned HSMHV_muephs_Given  :1;
  unsigned HSMHV_muepsp_Given  :1;
  unsigned HSMHV_vtmp_Given  :1;
  unsigned HSMHV_wvth0_Given  :1;
  unsigned HSMHV_muesr1_Given  :1;
  unsigned HSMHV_muesr0_Given  :1;
  unsigned HSMHV_muesrl_Given  :1;
  unsigned HSMHV_mueslp_Given  :1;
  unsigned HSMHV_muesrw_Given  :1;
  unsigned HSMHV_mueswp_Given  :1;
  unsigned HSMHV_bb_Given  :1;
  unsigned HSMHV_sub1_Given  :1;
  unsigned HSMHV_sub2_Given  :1;
  unsigned HSMHV_svgs_Given  :1; 
  unsigned HSMHV_svbs_Given  :1; 
  unsigned HSMHV_svbsl_Given  :1; 
  unsigned HSMHV_svds_Given  :1; 
  unsigned HSMHV_slg_Given  :1; 
  unsigned HSMHV_sub1l_Given  :1; 
  unsigned HSMHV_sub2l_Given  :1;  
  unsigned HSMHV_fn1_Given  :1;   
  unsigned HSMHV_fn2_Given  :1;  
  unsigned HSMHV_fn3_Given  :1;
  unsigned HSMHV_fvbs_Given  :1;
  unsigned HSMHV_svgsl_Given  :1;
  unsigned HSMHV_svgslp_Given  :1;
  unsigned HSMHV_svgswp_Given  :1;
  unsigned HSMHV_svgsw_Given  :1;
  unsigned HSMHV_svbslp_Given  :1;
  unsigned HSMHV_slgl_Given  :1;
  unsigned HSMHV_slglp_Given  :1;
  unsigned HSMHV_sub1lp_Given  :1;
  unsigned HSMHV_nsti_Given  :1;  
  unsigned HSMHV_wsti_Given  :1;
  unsigned HSMHV_wstil_Given  :1;
  unsigned HSMHV_wstilp_Given  :1;
  unsigned HSMHV_wstiw_Given  :1;
  unsigned HSMHV_wstiwp_Given  :1;
  unsigned HSMHV_scsti1_Given  :1;
  unsigned HSMHV_scsti2_Given  :1;
  unsigned HSMHV_vthsti_Given  :1;
  unsigned HSMHV_vdsti_Given  :1;
  unsigned HSMHV_muesti1_Given  :1;
  unsigned HSMHV_muesti2_Given  :1;
  unsigned HSMHV_muesti3_Given  :1;
  unsigned HSMHV_nsubpsti1_Given  :1;
  unsigned HSMHV_nsubpsti2_Given  :1;
  unsigned HSMHV_nsubpsti3_Given  :1;
  unsigned HSMHV_lpext_Given  :1;
  unsigned HSMHV_npext_Given  :1;
  unsigned HSMHV_scp22_Given  :1;
  unsigned HSMHV_scp21_Given  :1;
  unsigned HSMHV_bs1_Given  :1;
  unsigned HSMHV_bs2_Given  :1;
  unsigned HSMHV_cgso_Given  :1;
  unsigned HSMHV_cgdo_Given  :1;
  unsigned HSMHV_cgbo_Given  :1;
  unsigned HSMHV_tpoly_Given  :1;
  unsigned HSMHV_js0_Given  :1;
  unsigned HSMHV_js0sw_Given  :1;
  unsigned HSMHV_nj_Given  :1;
  unsigned HSMHV_njsw_Given  :1;  
  unsigned HSMHV_xti_Given  :1;
  unsigned HSMHV_cj_Given  :1;
  unsigned HSMHV_cjsw_Given  :1;
  unsigned HSMHV_cjswg_Given  :1;
  unsigned HSMHV_mj_Given  :1;
  unsigned HSMHV_mjsw_Given  :1;
  unsigned HSMHV_mjswg_Given  :1;
  unsigned HSMHV_xti2_Given  :1;
  unsigned HSMHV_cisb_Given  :1;
  unsigned HSMHV_cvb_Given  :1;
  unsigned HSMHV_ctemp_Given  :1;
  unsigned HSMHV_cisbk_Given  :1;
  unsigned HSMHV_cvbk_Given  :1;
  unsigned HSMHV_divx_Given  :1;
  unsigned HSMHV_pb_Given  :1;
  unsigned HSMHV_pbsw_Given  :1;
  unsigned HSMHV_pbswg_Given  :1;
  unsigned HSMHV_clm1_Given  :1;
  unsigned HSMHV_clm2_Given  :1;
  unsigned HSMHV_clm3_Given  :1;
  unsigned HSMHV_clm5_Given  :1;
  unsigned HSMHV_clm6_Given  :1;
  unsigned HSMHV_muetmp_Given  :1;
  unsigned HSMHV_vover_Given  :1;
  unsigned HSMHV_voverp_Given  :1;
  unsigned HSMHV_vovers_Given  :1;
  unsigned HSMHV_voversp_Given  :1;
  unsigned HSMHV_wfc_Given  :1;
  unsigned HSMHV_nsubcw_Given  :1;
  unsigned HSMHV_nsubcwp_Given  :1;
  unsigned HSMHV_qme1_Given  :1;
  unsigned HSMHV_qme2_Given  :1;
  unsigned HSMHV_qme3_Given  :1;
  unsigned HSMHV_gidl1_Given  :1;
  unsigned HSMHV_gidl2_Given  :1;
  unsigned HSMHV_gidl3_Given  :1;
  unsigned HSMHV_gidl4_Given  :1;
  unsigned HSMHV_gidl5_Given  :1;
  unsigned HSMHV_gleak1_Given  :1;
  unsigned HSMHV_gleak2_Given  :1;
  unsigned HSMHV_gleak3_Given  :1;
  unsigned HSMHV_gleak4_Given  :1;
  unsigned HSMHV_gleak5_Given  :1;
  unsigned HSMHV_gleak6_Given  :1;
  unsigned HSMHV_gleak7_Given  :1;
  unsigned HSMHV_glpart1_Given  :1;
  unsigned HSMHV_glksd1_Given  :1;
  unsigned HSMHV_glksd2_Given  :1;
  unsigned HSMHV_glksd3_Given  :1;
  unsigned HSMHV_glkb1_Given  :1;
  unsigned HSMHV_glkb2_Given  :1;
  unsigned HSMHV_glkb3_Given  :1;
  unsigned HSMHV_egig_Given  :1;
  unsigned HSMHV_igtemp2_Given  :1;
  unsigned HSMHV_igtemp3_Given  :1;
  unsigned HSMHV_vzadd0_Given  :1;
  unsigned HSMHV_pzadd0_Given  :1;
  unsigned HSMHV_nftrp_Given  :1;
  unsigned HSMHV_nfalp_Given  :1;
  unsigned HSMHV_cit_Given  :1;
  unsigned HSMHV_falph_Given  :1;
  unsigned HSMHV_dly1_Given :1;
  unsigned HSMHV_dly2_Given :1;
  unsigned HSMHV_dly3_Given :1;
  unsigned HSMHV_dlyov_Given :1;
  unsigned HSMHV_tnom_Given :1;
  unsigned HSMHV_ovslp_Given :1;
  unsigned HSMHV_ovmag_Given :1;
  unsigned HSMHV_gbmin_Given :1;
  unsigned HSMHV_rbpb_Given :1;
  unsigned HSMHV_rbpd_Given :1;
  unsigned HSMHV_rbps_Given :1;
  unsigned HSMHV_rbdb_Given :1;
  unsigned HSMHV_rbsb_Given :1;
  unsigned HSMHV_ibpc1_Given :1;
  unsigned HSMHV_ibpc2_Given :1;
  unsigned HSMHV_mphdfm_Given :1;
  unsigned HSMHV_rdvg11_Given  :1;
  unsigned HSMHV_rdvg12_Given  :1;
  unsigned HSMHV_qovsm_Given  :1;
  unsigned HSMHV_ldrift_Given  :1;
  unsigned HSMHV_rd20_Given  :1;
  unsigned HSMHV_rd21_Given  :1;
  unsigned HSMHV_rd22_Given  :1;
  unsigned HSMHV_rd22d_Given :1;
  unsigned HSMHV_rd23_Given  :1;
  unsigned HSMHV_rd24_Given  :1;
  unsigned HSMHV_rd25_Given  :1;
  unsigned HSMHV_rd26_Given  :1;
  unsigned HSMHV_rdvdl_Given  :1;
  unsigned HSMHV_rdvdlp_Given  :1;
  unsigned HSMHV_rdvds_Given  :1;
  unsigned HSMHV_rdvdsp_Given  :1;
  unsigned HSMHV_rd23l_Given  :1;
  unsigned HSMHV_rd23lp_Given  :1;
  unsigned HSMHV_rd23s_Given  :1;
  unsigned HSMHV_rd23sp_Given  :1;
  unsigned HSMHV_rds_Given  :1;
  unsigned HSMHV_rdsp_Given  :1;
  unsigned HSMHV_vbsmin_Given  :1;
  unsigned HSMHV_ninvd_Given  :1;
  unsigned HSMHV_ninvdw_Given  :1;
  unsigned HSMHV_ninvdwp_Given  :1;
  unsigned HSMHV_ninvdt1_Given  :1;
  unsigned HSMHV_ninvdt2_Given  :1;
  unsigned HSMHV_rdvb_Given  :1;
  unsigned HSMHV_rth0nf_Given  :1;

  unsigned HSMHV_rthtemp1_Given  :1;
  unsigned HSMHV_rthtemp2_Given  :1;
  unsigned HSMHV_prattemp1_Given  :1;
  unsigned HSMHV_prattemp2_Given  :1;


  unsigned HSMHV_rth0_Given :1;
  unsigned HSMHV_cth0_Given :1;
  unsigned HSMHV_powrat_Given :1;


  unsigned HSMHV_tcjbd_Given :1;
  unsigned HSMHV_tcjbs_Given :1;
  unsigned HSMHV_tcjbdsw_Given :1;
  unsigned HSMHV_tcjbssw_Given :1;
  unsigned HSMHV_tcjbdswg_Given :1;
  unsigned HSMHV_tcjbsswg_Given :1;


/*   unsigned HSMHV_wth0_Given :1; */
  unsigned HSMHV_qdftvd_Given  :1;
  unsigned HSMHV_rdvd_Given  :1;
  unsigned HSMHV_rdtemp1_Given :1;
  unsigned HSMHV_rdtemp2_Given :1;
  unsigned HSMHV_rth0r_Given :1;
  unsigned HSMHV_rdvdtemp1_Given :1;
  unsigned HSMHV_rdvdtemp2_Given :1;
  unsigned HSMHV_rth0w_Given :1;
  unsigned HSMHV_rth0wp_Given :1;

  unsigned HSMHV_cvdsover_Given :1;

  /* substrate effect */
  unsigned HSMHV_rdvsub_Given  :1;  /* substrate effect */
  unsigned HSMHV_rdvdsub_Given  :1; /* substrate effect */
  unsigned HSMHV_ddrift_Given  :1;  /* substrate effect */
  unsigned HSMHV_vbisub_Given  :1;  /* substrate effect */
  unsigned HSMHV_nsubsub_Given  :1; /* substrate effect */
  unsigned HSMHV_shemax_Given :1;

  /* binning parameters */
  unsigned HSMHV_lmin_Given :1;
  unsigned HSMHV_lmax_Given :1;
  unsigned HSMHV_wmin_Given :1;
  unsigned HSMHV_wmax_Given :1;
  unsigned HSMHV_lbinn_Given :1;
  unsigned HSMHV_wbinn_Given :1;

  /* Length dependence */
  unsigned HSMHV_lvmax_Given :1;
  unsigned HSMHV_lbgtmp1_Given :1;
  unsigned HSMHV_lbgtmp2_Given :1;
  unsigned HSMHV_leg0_Given :1;
  unsigned HSMHV_lvfbover_Given :1;
  unsigned HSMHV_lnover_Given :1;
  unsigned HSMHV_lnovers_Given :1;
  unsigned HSMHV_lwl2_Given :1;
  unsigned HSMHV_lvfbc_Given :1;
  unsigned HSMHV_lnsubc_Given :1;
  unsigned HSMHV_lnsubp_Given :1;
  unsigned HSMHV_lscp1_Given :1;
  unsigned HSMHV_lscp2_Given :1;
  unsigned HSMHV_lscp3_Given :1;
  unsigned HSMHV_lsc1_Given :1;
  unsigned HSMHV_lsc2_Given :1;
  unsigned HSMHV_lsc3_Given :1;
  unsigned HSMHV_lpgd1_Given :1;
  unsigned HSMHV_lpgd3_Given :1;
  unsigned HSMHV_lndep_Given :1;
  unsigned HSMHV_lninv_Given :1;
  unsigned HSMHV_lmuecb0_Given :1;
  unsigned HSMHV_lmuecb1_Given :1;
  unsigned HSMHV_lmueph1_Given :1;
  unsigned HSMHV_lvtmp_Given :1;
  unsigned HSMHV_lwvth0_Given :1;
  unsigned HSMHV_lmuesr1_Given :1;
  unsigned HSMHV_lmuetmp_Given :1;
  unsigned HSMHV_lsub1_Given :1;
  unsigned HSMHV_lsub2_Given :1;
  unsigned HSMHV_lsvds_Given :1;
  unsigned HSMHV_lsvbs_Given :1;
  unsigned HSMHV_lsvgs_Given :1;
  unsigned HSMHV_lfn1_Given :1;
  unsigned HSMHV_lfn2_Given :1;
  unsigned HSMHV_lfn3_Given :1;
  unsigned HSMHV_lfvbs_Given :1;
  unsigned HSMHV_lnsti_Given :1;
  unsigned HSMHV_lwsti_Given :1;
  unsigned HSMHV_lscsti1_Given :1;
  unsigned HSMHV_lscsti2_Given :1;
  unsigned HSMHV_lvthsti_Given :1;
  unsigned HSMHV_lmuesti1_Given :1;
  unsigned HSMHV_lmuesti2_Given :1;
  unsigned HSMHV_lmuesti3_Given :1;
  unsigned HSMHV_lnsubpsti1_Given :1;
  unsigned HSMHV_lnsubpsti2_Given :1;
  unsigned HSMHV_lnsubpsti3_Given :1;
  unsigned HSMHV_lcgso_Given :1;
  unsigned HSMHV_lcgdo_Given :1;
  unsigned HSMHV_ljs0_Given :1;
  unsigned HSMHV_ljs0sw_Given :1;
  unsigned HSMHV_lnj_Given :1;
  unsigned HSMHV_lcisbk_Given :1;
  unsigned HSMHV_lclm1_Given :1;
  unsigned HSMHV_lclm2_Given :1;
  unsigned HSMHV_lclm3_Given :1;
  unsigned HSMHV_lwfc_Given :1;
  unsigned HSMHV_lgidl1_Given :1;
  unsigned HSMHV_lgidl2_Given :1;
  unsigned HSMHV_lgleak1_Given :1;
  unsigned HSMHV_lgleak2_Given :1;
  unsigned HSMHV_lgleak3_Given :1;
  unsigned HSMHV_lgleak6_Given :1;
  unsigned HSMHV_lglksd1_Given :1;
  unsigned HSMHV_lglksd2_Given :1;
  unsigned HSMHV_lglkb1_Given :1;
  unsigned HSMHV_lglkb2_Given :1;
  unsigned HSMHV_lnftrp_Given :1;
  unsigned HSMHV_lnfalp_Given :1;
  unsigned HSMHV_lpthrou_Given :1;
  unsigned HSMHV_lvdiffj_Given :1;
  unsigned HSMHV_libpc1_Given :1;
  unsigned HSMHV_libpc2_Given :1;
  unsigned HSMHV_lcgbo_Given :1;
  unsigned HSMHV_lcvdsover_Given :1;
  unsigned HSMHV_lfalph_Given :1;
  unsigned HSMHV_lnpext_Given :1;
  unsigned HSMHV_lpowrat_Given :1;
  unsigned HSMHV_lrd_Given :1;
  unsigned HSMHV_lrd22_Given :1;
  unsigned HSMHV_lrd23_Given :1;
  unsigned HSMHV_lrd24_Given :1;
  unsigned HSMHV_lrdict1_Given :1;
  unsigned HSMHV_lrdov13_Given :1;
  unsigned HSMHV_lrdslp1_Given :1;
  unsigned HSMHV_lrdvb_Given :1;
  unsigned HSMHV_lrdvd_Given :1;
  unsigned HSMHV_lrdvg11_Given :1;
  unsigned HSMHV_lrs_Given :1;
  unsigned HSMHV_lrth0_Given :1;
  unsigned HSMHV_lvover_Given :1;

  /* Width dependence */
  unsigned HSMHV_wvmax_Given :1;
  unsigned HSMHV_wbgtmp1_Given :1;
  unsigned HSMHV_wbgtmp2_Given :1;
  unsigned HSMHV_weg0_Given :1;
  unsigned HSMHV_wvfbover_Given :1;
  unsigned HSMHV_wnover_Given :1;
  unsigned HSMHV_wnovers_Given :1;
  unsigned HSMHV_wwl2_Given :1;
  unsigned HSMHV_wvfbc_Given :1;
  unsigned HSMHV_wnsubc_Given :1;
  unsigned HSMHV_wnsubp_Given :1;
  unsigned HSMHV_wscp1_Given :1;
  unsigned HSMHV_wscp2_Given :1;
  unsigned HSMHV_wscp3_Given :1;
  unsigned HSMHV_wsc1_Given :1;
  unsigned HSMHV_wsc2_Given :1;
  unsigned HSMHV_wsc3_Given :1;
  unsigned HSMHV_wpgd1_Given :1;
  unsigned HSMHV_wpgd3_Given :1;
  unsigned HSMHV_wndep_Given :1;
  unsigned HSMHV_wninv_Given :1;
  unsigned HSMHV_wmuecb0_Given :1;
  unsigned HSMHV_wmuecb1_Given :1;
  unsigned HSMHV_wmueph1_Given :1;
  unsigned HSMHV_wvtmp_Given :1;
  unsigned HSMHV_wwvth0_Given :1;
  unsigned HSMHV_wmuesr1_Given :1;
  unsigned HSMHV_wmuetmp_Given :1;
  unsigned HSMHV_wsub1_Given :1;
  unsigned HSMHV_wsub2_Given :1;
  unsigned HSMHV_wsvds_Given :1;
  unsigned HSMHV_wsvbs_Given :1;
  unsigned HSMHV_wsvgs_Given :1;
  unsigned HSMHV_wfn1_Given :1;
  unsigned HSMHV_wfn2_Given :1;
  unsigned HSMHV_wfn3_Given :1;
  unsigned HSMHV_wfvbs_Given :1;
  unsigned HSMHV_wnsti_Given :1;
  unsigned HSMHV_wwsti_Given :1;
  unsigned HSMHV_wscsti1_Given :1;
  unsigned HSMHV_wscsti2_Given :1;
  unsigned HSMHV_wvthsti_Given :1;
  unsigned HSMHV_wmuesti1_Given :1;
  unsigned HSMHV_wmuesti2_Given :1;
  unsigned HSMHV_wmuesti3_Given :1;
  unsigned HSMHV_wnsubpsti1_Given :1;
  unsigned HSMHV_wnsubpsti2_Given :1;
  unsigned HSMHV_wnsubpsti3_Given :1;
  unsigned HSMHV_wcgso_Given :1;
  unsigned HSMHV_wcgdo_Given :1;
  unsigned HSMHV_wjs0_Given :1;
  unsigned HSMHV_wjs0sw_Given :1;
  unsigned HSMHV_wnj_Given :1;
  unsigned HSMHV_wcisbk_Given :1;
  unsigned HSMHV_wclm1_Given :1;
  unsigned HSMHV_wclm2_Given :1;
  unsigned HSMHV_wclm3_Given :1;
  unsigned HSMHV_wwfc_Given :1;
  unsigned HSMHV_wgidl1_Given :1;
  unsigned HSMHV_wgidl2_Given :1;
  unsigned HSMHV_wgleak1_Given :1;
  unsigned HSMHV_wgleak2_Given :1;
  unsigned HSMHV_wgleak3_Given :1;
  unsigned HSMHV_wgleak6_Given :1;
  unsigned HSMHV_wglksd1_Given :1;
  unsigned HSMHV_wglksd2_Given :1;
  unsigned HSMHV_wglkb1_Given :1;
  unsigned HSMHV_wglkb2_Given :1;
  unsigned HSMHV_wnftrp_Given :1;
  unsigned HSMHV_wnfalp_Given :1;
  unsigned HSMHV_wpthrou_Given :1;
  unsigned HSMHV_wvdiffj_Given :1;
  unsigned HSMHV_wibpc1_Given :1;
  unsigned HSMHV_wibpc2_Given :1;
  unsigned HSMHV_wcgbo_Given :1;
  unsigned HSMHV_wcvdsover_Given :1;
  unsigned HSMHV_wfalph_Given :1;
  unsigned HSMHV_wnpext_Given :1;
  unsigned HSMHV_wpowrat_Given :1;
  unsigned HSMHV_wrd_Given :1;
  unsigned HSMHV_wrd22_Given :1;
  unsigned HSMHV_wrd23_Given :1;
  unsigned HSMHV_wrd24_Given :1;
  unsigned HSMHV_wrdict1_Given :1;
  unsigned HSMHV_wrdov13_Given :1;
  unsigned HSMHV_wrdslp1_Given :1;
  unsigned HSMHV_wrdvb_Given :1;
  unsigned HSMHV_wrdvd_Given :1;
  unsigned HSMHV_wrdvg11_Given :1;
  unsigned HSMHV_wrs_Given :1;
  unsigned HSMHV_wrth0_Given :1;
  unsigned HSMHV_wvover_Given :1;

  /* Cross-term dependence */
  unsigned HSMHV_pvmax_Given :1;
  unsigned HSMHV_pbgtmp1_Given :1;
  unsigned HSMHV_pbgtmp2_Given :1;
  unsigned HSMHV_peg0_Given :1;
  unsigned HSMHV_pvfbover_Given :1;
  unsigned HSMHV_pnover_Given :1;
  unsigned HSMHV_pnovers_Given :1;
  unsigned HSMHV_pwl2_Given :1;
  unsigned HSMHV_pvfbc_Given :1;
  unsigned HSMHV_pnsubc_Given :1;
  unsigned HSMHV_pnsubp_Given :1;
  unsigned HSMHV_pscp1_Given :1;
  unsigned HSMHV_pscp2_Given :1;
  unsigned HSMHV_pscp3_Given :1;
  unsigned HSMHV_psc1_Given :1;
  unsigned HSMHV_psc2_Given :1;
  unsigned HSMHV_psc3_Given :1;
  unsigned HSMHV_ppgd1_Given :1;
  unsigned HSMHV_ppgd3_Given :1;
  unsigned HSMHV_pndep_Given :1;
  unsigned HSMHV_pninv_Given :1;
  unsigned HSMHV_pmuecb0_Given :1;
  unsigned HSMHV_pmuecb1_Given :1;
  unsigned HSMHV_pmueph1_Given :1;
  unsigned HSMHV_pvtmp_Given :1;
  unsigned HSMHV_pwvth0_Given :1;
  unsigned HSMHV_pmuesr1_Given :1;
  unsigned HSMHV_pmuetmp_Given :1;
  unsigned HSMHV_psub1_Given :1;
  unsigned HSMHV_psub2_Given :1;
  unsigned HSMHV_psvds_Given :1;
  unsigned HSMHV_psvbs_Given :1;
  unsigned HSMHV_psvgs_Given :1;
  unsigned HSMHV_pfn1_Given :1;
  unsigned HSMHV_pfn2_Given :1;
  unsigned HSMHV_pfn3_Given :1;
  unsigned HSMHV_pfvbs_Given :1;
  unsigned HSMHV_pnsti_Given :1;
  unsigned HSMHV_pwsti_Given :1;
  unsigned HSMHV_pscsti1_Given :1;
  unsigned HSMHV_pscsti2_Given :1;
  unsigned HSMHV_pvthsti_Given :1;
  unsigned HSMHV_pmuesti1_Given :1;
  unsigned HSMHV_pmuesti2_Given :1;
  unsigned HSMHV_pmuesti3_Given :1;
  unsigned HSMHV_pnsubpsti1_Given :1;
  unsigned HSMHV_pnsubpsti2_Given :1;
  unsigned HSMHV_pnsubpsti3_Given :1;
  unsigned HSMHV_pcgso_Given :1;
  unsigned HSMHV_pcgdo_Given :1;
  unsigned HSMHV_pjs0_Given :1;
  unsigned HSMHV_pjs0sw_Given :1;
  unsigned HSMHV_pnj_Given :1;
  unsigned HSMHV_pcisbk_Given :1;
  unsigned HSMHV_pclm1_Given :1;
  unsigned HSMHV_pclm2_Given :1;
  unsigned HSMHV_pclm3_Given :1;
  unsigned HSMHV_pwfc_Given :1;
  unsigned HSMHV_pgidl1_Given :1;
  unsigned HSMHV_pgidl2_Given :1;
  unsigned HSMHV_pgleak1_Given :1;
  unsigned HSMHV_pgleak2_Given :1;
  unsigned HSMHV_pgleak3_Given :1;
  unsigned HSMHV_pgleak6_Given :1;
  unsigned HSMHV_pglksd1_Given :1;
  unsigned HSMHV_pglksd2_Given :1;
  unsigned HSMHV_pglkb1_Given :1;
  unsigned HSMHV_pglkb2_Given :1;
  unsigned HSMHV_pnftrp_Given :1;
  unsigned HSMHV_pnfalp_Given :1;
  unsigned HSMHV_ppthrou_Given :1;
  unsigned HSMHV_pvdiffj_Given :1;
  unsigned HSMHV_pibpc1_Given :1;
  unsigned HSMHV_pibpc2_Given :1;
  unsigned HSMHV_pcgbo_Given :1;
  unsigned HSMHV_pcvdsover_Given :1;
  unsigned HSMHV_pfalph_Given :1;
  unsigned HSMHV_pnpext_Given :1;
  unsigned HSMHV_ppowrat_Given :1;
  unsigned HSMHV_prd_Given :1;
  unsigned HSMHV_prd22_Given :1;
  unsigned HSMHV_prd23_Given :1;
  unsigned HSMHV_prd24_Given :1;
  unsigned HSMHV_prdict1_Given :1;
  unsigned HSMHV_prdov13_Given :1;
  unsigned HSMHV_prdslp1_Given :1;
  unsigned HSMHV_prdvb_Given :1;
  unsigned HSMHV_prdvd_Given :1;
  unsigned HSMHV_prdvg11_Given :1;
  unsigned HSMHV_prs_Given :1;
  unsigned HSMHV_prth0_Given :1;
  unsigned HSMHV_pvover_Given :1;

  unsigned  HSMHVvgsMaxGiven  :1;
  unsigned  HSMHVvgdMaxGiven  :1;
  unsigned  HSMHVvgbMaxGiven  :1;
  unsigned  HSMHVvdsMaxGiven  :1;
  unsigned  HSMHVvbsMaxGiven  :1;
  unsigned  HSMHVvbdMaxGiven  :1;
  unsigned  HSMHVvgsrMaxGiven  :1;
  unsigned  HSMHVvgdrMaxGiven  :1;
  unsigned  HSMHVvgbrMaxGiven  :1;
  unsigned  HSMHVvbsrMaxGiven  :1;
  unsigned  HSMHVvbdrMaxGiven  :1;

} HSMHVmodel;

#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/

#define HSMHV_BAD_PARAM -1

/* flags */
#define HSMHV_MOD_NMOS     1
#define HSMHV_MOD_PMOS     2
#define HSMHV_MOD_LEVEL    3
#define HSMHV_MOD_INFO     4
#define HSMHV_MOD_NOISE    5
#define HSMHV_MOD_VERSION  6
#define HSMHV_MOD_SHOW     7
#define HSMHV_MOD_CORSRD  11
#define HSMHV_MOD_COIPRV  12
#define HSMHV_MOD_COPPRV  13
#define HSMHV_MOD_COADOV  17
#define HSMHV_MOD_COISUB  21
#define HSMHV_MOD_COIIGS    22
#define HSMHV_MOD_COGIDL 23
#define HSMHV_MOD_COOVLP  24
#define HSMHV_MOD_COOVLPS  8
#define HSMHV_MOD_COFLICK 25
#define HSMHV_MOD_COISTI  26
#define HSMHV_MOD_CONQS   29
#define HSMHV_MOD_COTHRML 30
#define HSMHV_MOD_COIGN   31    /* Induced gate noise */
#define HSMHV_MOD_CORG    32
#define HSMHV_MOD_CORBNET 33
#define HSMHV_MOD_CODFM   36    /* DFM */
#define HSMHV_MOD_COQOVSM 34
#define HSMHV_MOD_COSELFHEAT 35 /* Self-heating model--SHE-- */
#define HSMHV_MOD_COSUBNODE  48 
#define HSMHV_MOD_COSYM   37    /* Symmery model for HV */
#define HSMHV_MOD_COTEMP  38
#define HSMHV_MOD_COLDRIFT 39
/* device parameters */
#define HSMHV_COSELFHEAT  49
#define HSMHV_COSUBNODE   50
#define HSMHV_L           51
#define HSMHV_W           52
#define HSMHV_AD          53
#define HSMHV_AS          54
#define HSMHV_PD          55
#define HSMHV_PS          56
#define HSMHV_NRD         57
#define HSMHV_NRS         58
/* #define HSMHV_TEMP        59	not used */
#define HSMHV_DTEMP       60
#define HSMHV_OFF         61
#define HSMHV_IC_VBS      62
#define HSMHV_IC_VDS      63
#define HSMHV_IC_VGS      64
#define HSMHV_IC          65
#define HSMHV_CORBNET     66
#define HSMHV_RBPB        67
#define HSMHV_RBPD        68
#define HSMHV_RBPS        69
#define HSMHV_RBDB        70
#define HSMHV_RBSB        71
#define HSMHV_CORG        72
/* #define HSMHV_RSHG        73 */
#define HSMHV_NGCON       74
#define HSMHV_XGW         75 
#define HSMHV_XGL         76
#define HSMHV_NF          77
#define HSMHV_SA          78
#define HSMHV_SB          79
#define HSMHV_SD          80
#define HSMHV_NSUBCDFM    82
#define HSMHV_M           83
#define HSMHV_SUBLD1      86
#define HSMHV_SUBLD2      87
#define HSMHV_LOVER       41
#define HSMHV_LOVERS      42
#define HSMHV_LOVERLD     43
#define HSMHV_LDRIFT1     88
#define HSMHV_LDRIFT2     89
#define HSMHV_LDRIFT1S    90
#define HSMHV_LDRIFT2S    91
/* #define HSMHV_RTH0        84	not used */
/* #define HSMHV_CTH0        85	not used */




/* model parameters */
#define HSMHV_MOD_VBSMIN    198
#define HSMHV_MOD_VMAX      500
#define HSMHV_MOD_VMAXT1    503
#define HSMHV_MOD_VMAXT2    504
#define HSMHV_MOD_BGTMP1    101
#define HSMHV_MOD_BGTMP2    102
#define HSMHV_MOD_EG0       103
#define HSMHV_MOD_TOX       104
#define HSMHV_MOD_XLD       105
#define HSMHV_MOD_LOVER     106
#define HSMHV_MOD_LOVERS    385
#define HSMHV_MOD_RDOV11    313
#define HSMHV_MOD_RDOV12    314
#define HSMHV_MOD_RDOV13    476
#define HSMHV_MOD_RDSLP1    315
#define HSMHV_MOD_RDICT1    316
#define HSMHV_MOD_RDSLP2    317
#define HSMHV_MOD_RDICT2    318
#define HSMHV_MOD_LOVERLD   436
#define HSMHV_MOD_LDRIFT1   319
#define HSMHV_MOD_LDRIFT2   320
#define HSMHV_MOD_LDRIFT1S  324
#define HSMHV_MOD_LDRIFT2S  325
#define HSMHV_MOD_SUBLD1    321
#define HSMHV_MOD_SUBLD2    322
#define HSMHV_MOD_DDLTMAX   421 /* Vdseff */
#define HSMHV_MOD_DDLTSLP   422 /* Vdseff */
#define HSMHV_MOD_DDLTICT   423 /* Vdseff */
#define HSMHV_MOD_VFBOVER   428
#define HSMHV_MOD_NOVER     430
#define HSMHV_MOD_NOVERS    431
#define HSMHV_MOD_XWD       107
#define HSMHV_MOD_XWDC      513
#define HSMHV_MOD_XL        112
#define HSMHV_MOD_XW        117
#define HSMHV_MOD_SAREF     433
#define HSMHV_MOD_SBREF     434
#define HSMHV_MOD_LL        108
#define HSMHV_MOD_LLD       109
#define HSMHV_MOD_LLN       110
#define HSMHV_MOD_WL        111
#define HSMHV_MOD_WL1       113
#define HSMHV_MOD_WL1P      114
#define HSMHV_MOD_WL2       407
#define HSMHV_MOD_WL2P      408
#define HSMHV_MOD_WLD       115
#define HSMHV_MOD_WLN       116

#define HSMHV_MOD_XQY       178
#define HSMHV_MOD_XQY1      118
#define HSMHV_MOD_XQY2      120
#define HSMHV_MOD_RSH       119

#define HSMHV_MOD_RSHG      384
/* #define HSMHV_MOD_NGCON     385 */
/* #define HSMHV_MOD_XGW       386 */
/* #define HSMHV_MOD_XGL       387 */
/* #define HSMHV_MOD_NF        388 */
#define HSMHV_MOD_RS        398
#define HSMHV_MOD_RD        399

#define HSMHV_MOD_VFBC      121
#define HSMHV_MOD_VBI       122
#define HSMHV_MOD_NSUBC     123
#define HSMHV_MOD_TNOM      124
#define HSMHV_MOD_PARL2     125
#define HSMHV_MOD_SC1       126
#define HSMHV_MOD_SC2       127
#define HSMHV_MOD_SC3       128
#define HSMHV_MOD_SC4       248
#define HSMHV_MOD_NDEP      129
#define HSMHV_MOD_NDEPL     419
#define HSMHV_MOD_NDEPLP    420
#define HSMHV_MOD_NINV      130
#define HSMHV_MOD_NINVD     505
#define HSMHV_MOD_NINVDW    506
#define HSMHV_MOD_NINVDWP   507
#define HSMHV_MOD_NINVDT1   508
#define HSMHV_MOD_NINVDT2   509
#define HSMHV_MOD_MUECB0    131
#define HSMHV_MOD_MUECB1    132
#define HSMHV_MOD_MUEPH1    133
#define HSMHV_MOD_MUEPH0    134
#define HSMHV_MOD_MUEPHW    135
#define HSMHV_MOD_MUEPWP    136
#define HSMHV_MOD_MUEPHL    137
#define HSMHV_MOD_MUEPLP    138
#define HSMHV_MOD_MUEPHS    139
#define HSMHV_MOD_MUEPSP    140
#define HSMHV_MOD_VTMP      141
#define HSMHV_MOD_WVTH0 	   142
#define HSMHV_MOD_MUESR1    143
#define HSMHV_MOD_MUESR0    144
#define HSMHV_MOD_MUESRL    145
#define HSMHV_MOD_MUESLP    146
#define HSMHV_MOD_MUESRW    147
#define HSMHV_MOD_MUESWP    148
#define HSMHV_MOD_BB        149

#define HSMHV_MOD_SUB1      151
#define HSMHV_MOD_SUB2      152
#define HSMHV_MOD_CGSO      154
#define HSMHV_MOD_CGDO      155
#define HSMHV_MOD_CGBO      156
#define HSMHV_MOD_JS0       157
#define HSMHV_MOD_JS0SW     158
#define HSMHV_MOD_NJ        159
#define HSMHV_MOD_NJSW      160
#define HSMHV_MOD_XTI       161
#define HSMHV_MOD_CJ        162
#define HSMHV_MOD_CJSW      163
#define HSMHV_MOD_CJSWG     164
#define HSMHV_MOD_MJ        165
#define HSMHV_MOD_MJSW      166
#define HSMHV_MOD_MJSWG     167
#define HSMHV_MOD_XTI2      168
#define HSMHV_MOD_CISB      169
#define HSMHV_MOD_CVB       170
#define HSMHV_MOD_CTEMP     171
#define HSMHV_MOD_CISBK     172
#define HSMHV_MOD_CVBK      173
#define HSMHV_MOD_DIVX      174
#define HSMHV_MOD_PB        175
#define HSMHV_MOD_PBSW      176
#define HSMHV_MOD_PBSWG     177
#define HSMHV_MOD_TPOLY     179
/* #define HSMHV_MOD_TPOLYLD   477	not used */
#define HSMHV_MOD_LP        180
#define HSMHV_MOD_NSUBP     181
#define HSMHV_MOD_NSUBP0    182
#define HSMHV_MOD_NSUBWP    183
#define HSMHV_MOD_SCP1      184
#define HSMHV_MOD_SCP2      185
#define HSMHV_MOD_SCP3      186
#define HSMHV_MOD_PGD1      187
#define HSMHV_MOD_PGD2      188
#define HSMHV_MOD_PGD3      189
#define HSMHV_MOD_PGD4      190
#define HSMHV_MOD_CLM1      191
#define HSMHV_MOD_CLM2      192
#define HSMHV_MOD_CLM3      193
#define HSMHV_MOD_CLM5      402
#define HSMHV_MOD_CLM6      403
#define HSMHV_MOD_MUETMP    195

#define HSMHV_MOD_VOVER     199
#define HSMHV_MOD_VOVERP    200
#define HSMHV_MOD_WFC       201
#define HSMHV_MOD_NSUBCW    249
#define HSMHV_MOD_NSUBCWP   250
#define HSMHV_MOD_QME1      202
#define HSMHV_MOD_QME2      203
#define HSMHV_MOD_QME3      204
#define HSMHV_MOD_GIDL1     205
#define HSMHV_MOD_GIDL2     206
#define HSMHV_MOD_GIDL3     207
#define HSMHV_MOD_GLEAK1    208
#define HSMHV_MOD_GLEAK2    209
#define HSMHV_MOD_GLEAK3    210
#define HSMHV_MOD_GLEAK4    211
#define HSMHV_MOD_GLEAK5    212
#define HSMHV_MOD_GLEAK6    213
#define HSMHV_MOD_GLEAK7    214
#define HSMHV_MOD_GLPART1   406
#define HSMHV_MOD_GLKSD1    215
#define HSMHV_MOD_GLKSD2    216
#define HSMHV_MOD_GLKSD3    217
#define HSMHV_MOD_GLKB1     218
#define HSMHV_MOD_GLKB2     219
#define HSMHV_MOD_GLKB3     429
#define HSMHV_MOD_EGIG      220
#define HSMHV_MOD_IGTEMP2   221
#define HSMHV_MOD_IGTEMP3   222
#define HSMHV_MOD_VZADD0    223
#define HSMHV_MOD_PZADD0    224
#define HSMHV_MOD_NSTI      225
#define HSMHV_MOD_WSTI      226
#define HSMHV_MOD_WSTIL     227
#define HSMHV_MOD_WSTILP    231
#define HSMHV_MOD_WSTIW     234
#define HSMHV_MOD_WSTIWP    228
#define HSMHV_MOD_SCSTI1    229
#define HSMHV_MOD_SCSTI2    230
#define HSMHV_MOD_VTHSTI    232
#define HSMHV_MOD_VDSTI     233
#define HSMHV_MOD_MUESTI1   235
#define HSMHV_MOD_MUESTI2   236
#define HSMHV_MOD_MUESTI3   237
#define HSMHV_MOD_NSUBPSTI1 238
#define HSMHV_MOD_NSUBPSTI2 239
#define HSMHV_MOD_NSUBPSTI3 240
#define HSMHV_MOD_LPEXT     241
#define HSMHV_MOD_NPEXT     242
#define HSMHV_MOD_SCP22     243
#define HSMHV_MOD_SCP21     244
#define HSMHV_MOD_BS1       245
#define HSMHV_MOD_BS2       246
#define HSMHV_MOD_KAPPA     251
#define HSMHV_MOD_PTHROU    253
#define HSMHV_MOD_VDIFFJ    254
#define HSMHV_MOD_DLY1      255
#define HSMHV_MOD_DLY2      256
#define HSMHV_MOD_DLY3      257
#define HSMHV_MOD_NFTRP     258
#define HSMHV_MOD_NFALP     259
#define HSMHV_MOD_CIT       260
#define HSMHV_MOD_FALPH     263
#define HSMHV_MOD_OVSLP     261
#define HSMHV_MOD_OVMAG     262
#define HSMHV_MOD_GIDL4     281
#define HSMHV_MOD_GIDL5     282
#define HSMHV_MOD_SVGS      283
#define HSMHV_MOD_SVBS      284
#define HSMHV_MOD_SVBSL     285
#define HSMHV_MOD_SVDS      286
#define HSMHV_MOD_SLG       287
#define HSMHV_MOD_SUB1L     290
#define HSMHV_MOD_SUB2L     292
#define HSMHV_MOD_FN1       294
#define HSMHV_MOD_FN2       295
#define HSMHV_MOD_FN3       296
#define HSMHV_MOD_FVBS      297
#define HSMHV_MOD_VOVERS    303
#define HSMHV_MOD_VOVERSP   304
#define HSMHV_MOD_SVGSL     305
#define HSMHV_MOD_SVGSLP    306
#define HSMHV_MOD_SVGSWP    307
#define HSMHV_MOD_SVGSW     308
#define HSMHV_MOD_SVBSLP    309
#define HSMHV_MOD_SLGL      310
#define HSMHV_MOD_SLGLP     311
#define HSMHV_MOD_SUB1LP    312
#define HSMHV_MOD_IBPC1     404
#define HSMHV_MOD_IBPC2     405
#define HSMHV_MOD_MPHDFM    409
#define HSMHV_MOD_RDVG11    424
#define HSMHV_MOD_RDVG12    425
#define HSMHV_MOD_RTH0      432
#define HSMHV_MOD_CTH0      462
#define HSMHV_MOD_POWRAT    463
/* #define HSMHV_MOD_WTH0      463 /\*---------SHE----------*\/ */
#define HSMHV_MOD_DLYOV     437
#define HSMHV_MOD_QDFTVD    438
#define HSMHV_MOD_XLDLD     439
#define HSMHV_MOD_XWDLD     494
#define HSMHV_MOD_RDVD      510
#define HSMHV_MOD_RDVB      301

#define HSMHV_MOD_RDVSUB    481 /* substrate effect */
#define HSMHV_MOD_RDVDSUB   482 /* substrate effect */
#define HSMHV_MOD_DDRIFT    483 /* substrate effect */
#define HSMHV_MOD_VBISUB    484 /* substrate effect */
#define HSMHV_MOD_NSUBSUB   485 /* substrate effect */

#define HSMHV_MOD_QOVSM     323
#define HSMHV_MOD_LDRIFT    458
#define HSMHV_MOD_RD20      447 
#define HSMHV_MOD_RD21      441 
#define HSMHV_MOD_RD22      442 
#define HSMHV_MOD_RD22D     478
#define HSMHV_MOD_RD23      443 
#define HSMHV_MOD_RD24      444 
#define HSMHV_MOD_RD25      445 
#define HSMHV_MOD_RD26      446 
#define HSMHV_MOD_RDVDL     448 
#define HSMHV_MOD_RDVDLP    449 
#define HSMHV_MOD_RDVDS     450 
#define HSMHV_MOD_RDVDSP    451 
#define HSMHV_MOD_RD23L     452 
#define HSMHV_MOD_RD23LP    453 
#define HSMHV_MOD_RD23S     454 
#define HSMHV_MOD_RD23SP    455 
#define HSMHV_MOD_RDS       456 
#define HSMHV_MOD_RDSP      457 
#define HSMHV_MOD_RDTEMP1   461
#define HSMHV_MOD_RDTEMP2   464
#define HSMHV_MOD_RTH0R     470
#define HSMHV_MOD_RDVDTEMP1 471
#define HSMHV_MOD_RDVDTEMP2 472
#define HSMHV_MOD_RTH0W     473
#define HSMHV_MOD_RTH0WP    474
#define HSMHV_MOD_RTH0NF    475

#define HSMHV_MOD_RTHTEMP1  490
#define HSMHV_MOD_RTHTEMP2  491
#define HSMHV_MOD_PRATTEMP1 492
#define HSMHV_MOD_PRATTEMP2 493


#define HSMHV_MOD_CVDSOVER  480
#define HSMHV_MOD_SHEMAX      100

/* binning parameters */
#define HSMHV_MOD_LMIN       1000
#define HSMHV_MOD_LMAX       1001
#define HSMHV_MOD_WMIN       1002
#define HSMHV_MOD_WMAX       1003
#define HSMHV_MOD_LBINN      1004
#define HSMHV_MOD_WBINN      1005

/* Length dependence */
#define HSMHV_MOD_LVMAX      1100
#define HSMHV_MOD_LBGTMP1    1101
#define HSMHV_MOD_LBGTMP2    1102
#define HSMHV_MOD_LEG0       1103
#define HSMHV_MOD_LVFBOVER   1428
#define HSMHV_MOD_LNOVER     1430
#define HSMHV_MOD_LNOVERS    1431
#define HSMHV_MOD_LWL2       1407
#define HSMHV_MOD_LVFBC      1121
#define HSMHV_MOD_LNSUBC     1123
#define HSMHV_MOD_LNSUBP     1181
#define HSMHV_MOD_LSCP1      1184
#define HSMHV_MOD_LSCP2      1185
#define HSMHV_MOD_LSCP3      1186
#define HSMHV_MOD_LSC1       1126
#define HSMHV_MOD_LSC2       1127
#define HSMHV_MOD_LSC3       1128
#define HSMHV_MOD_LPGD1      1187
#define HSMHV_MOD_LPGD3      1189
#define HSMHV_MOD_LNDEP      1129
#define HSMHV_MOD_LNINV      1130
#define HSMHV_MOD_LMUECB0    1131
#define HSMHV_MOD_LMUECB1    1132
#define HSMHV_MOD_LMUEPH1    1133
#define HSMHV_MOD_LVTMP      1141
#define HSMHV_MOD_LWVTH0     1142
#define HSMHV_MOD_LMUESR1    1143
#define HSMHV_MOD_LMUETMP    1195
#define HSMHV_MOD_LSUB1      1151
#define HSMHV_MOD_LSUB2      1152
#define HSMHV_MOD_LSVDS      1286
#define HSMHV_MOD_LSVBS      1284
#define HSMHV_MOD_LSVGS      1283
#define HSMHV_MOD_LFN1       1294
#define HSMHV_MOD_LFN2       1295
#define HSMHV_MOD_LFN3       1296
#define HSMHV_MOD_LFVBS      1297
#define HSMHV_MOD_LNSTI      1225
#define HSMHV_MOD_LWSTI      1226
#define HSMHV_MOD_LSCSTI1    1229
#define HSMHV_MOD_LSCSTI2    1230
#define HSMHV_MOD_LVTHSTI    1232
#define HSMHV_MOD_LMUESTI1   1235
#define HSMHV_MOD_LMUESTI2   1236
#define HSMHV_MOD_LMUESTI3   1237
#define HSMHV_MOD_LNSUBPSTI1 1238
#define HSMHV_MOD_LNSUBPSTI2 1239
#define HSMHV_MOD_LNSUBPSTI3 1240
#define HSMHV_MOD_LCGSO      1154
#define HSMHV_MOD_LCGDO      1155
#define HSMHV_MOD_LJS0       1157
#define HSMHV_MOD_LJS0SW     1158
#define HSMHV_MOD_LNJ        1159
#define HSMHV_MOD_LCISBK     1172
#define HSMHV_MOD_LCLM1      1191
#define HSMHV_MOD_LCLM2      1192
#define HSMHV_MOD_LCLM3      1193
#define HSMHV_MOD_LWFC       1201
#define HSMHV_MOD_LGIDL1     1205
#define HSMHV_MOD_LGIDL2     1206
#define HSMHV_MOD_LGLEAK1    1208
#define HSMHV_MOD_LGLEAK2    1209
#define HSMHV_MOD_LGLEAK3    1210
#define HSMHV_MOD_LGLEAK6    1213
#define HSMHV_MOD_LGLKSD1    1215
#define HSMHV_MOD_LGLKSD2    1216
#define HSMHV_MOD_LGLKB1     1218
#define HSMHV_MOD_LGLKB2     1219
#define HSMHV_MOD_LNFTRP     1258
#define HSMHV_MOD_LNFALP     1259
#define HSMHV_MOD_LPTHROU    1253
#define HSMHV_MOD_LVDIFFJ    1254
#define HSMHV_MOD_LIBPC1     1404
#define HSMHV_MOD_LIBPC2     1405
#define HSMHV_MOD_LCGBO      1156
#define HSMHV_MOD_LCVDSOVER  1480
#define HSMHV_MOD_LFALPH     1263
#define HSMHV_MOD_LNPEXT     1242
#define HSMHV_MOD_LPOWRAT    1463
#define HSMHV_MOD_LRD        1399
#define HSMHV_MOD_LRD22      1442
#define HSMHV_MOD_LRD23      1443
#define HSMHV_MOD_LRD24      1444
#define HSMHV_MOD_LRDICT1    1316
#define HSMHV_MOD_LRDOV13    1476
#define HSMHV_MOD_LRDSLP1    1315
#define HSMHV_MOD_LRDVB      1301
#define HSMHV_MOD_LRDVD      1510
#define HSMHV_MOD_LRDVG11    1424
#define HSMHV_MOD_LRS        1398
#define HSMHV_MOD_LRTH0      1432
#define HSMHV_MOD_LVOVER     1199

/* Width dependence */
#define HSMHV_MOD_WVMAX      2100
#define HSMHV_MOD_WBGTMP1    2101
#define HSMHV_MOD_WBGTMP2    2102
#define HSMHV_MOD_WEG0       2103
#define HSMHV_MOD_WVFBOVER   2428
#define HSMHV_MOD_WNOVER     2430
#define HSMHV_MOD_WNOVERS    2431
#define HSMHV_MOD_WWL2       2407
#define HSMHV_MOD_WVFBC      2121
#define HSMHV_MOD_WNSUBC     2123
#define HSMHV_MOD_WNSUBP     2181
#define HSMHV_MOD_WSCP1      2184
#define HSMHV_MOD_WSCP2      2185
#define HSMHV_MOD_WSCP3      2186
#define HSMHV_MOD_WSC1       2126
#define HSMHV_MOD_WSC2       2127
#define HSMHV_MOD_WSC3       2128
#define HSMHV_MOD_WPGD1      2187
#define HSMHV_MOD_WPGD3      2189
#define HSMHV_MOD_WNDEP      2129
#define HSMHV_MOD_WNINV      2130
#define HSMHV_MOD_WMUECB0    2131
#define HSMHV_MOD_WMUECB1    2132
#define HSMHV_MOD_WMUEPH1    2133
#define HSMHV_MOD_WVTMP      2141
#define HSMHV_MOD_WWVTH0     2142
#define HSMHV_MOD_WMUESR1    2143
#define HSMHV_MOD_WMUETMP    2195
#define HSMHV_MOD_WSUB1      2151
#define HSMHV_MOD_WSUB2      2152
#define HSMHV_MOD_WSVDS      2286
#define HSMHV_MOD_WSVBS      2284
#define HSMHV_MOD_WSVGS      2283
#define HSMHV_MOD_WFN1       2294
#define HSMHV_MOD_WFN2       2295
#define HSMHV_MOD_WFN3       2296
#define HSMHV_MOD_WFVBS      2297
#define HSMHV_MOD_WNSTI      2225
#define HSMHV_MOD_WWSTI      2226
#define HSMHV_MOD_WSCSTI1    2229
#define HSMHV_MOD_WSCSTI2    2230
#define HSMHV_MOD_WVTHSTI    2232
#define HSMHV_MOD_WMUESTI1   2235
#define HSMHV_MOD_WMUESTI2   2236
#define HSMHV_MOD_WMUESTI3   2237
#define HSMHV_MOD_WNSUBPSTI1 2238
#define HSMHV_MOD_WNSUBPSTI2 2239
#define HSMHV_MOD_WNSUBPSTI3 2240
#define HSMHV_MOD_WCGSO      2154
#define HSMHV_MOD_WCGDO      2155
#define HSMHV_MOD_WJS0       2157
#define HSMHV_MOD_WJS0SW     2158
#define HSMHV_MOD_WNJ        2159
#define HSMHV_MOD_WCISBK     2172
#define HSMHV_MOD_WCLM1      2191
#define HSMHV_MOD_WCLM2      2192
#define HSMHV_MOD_WCLM3      2193
#define HSMHV_MOD_WWFC       2201
#define HSMHV_MOD_WGIDL1     2205
#define HSMHV_MOD_WGIDL2     2206
#define HSMHV_MOD_WGLEAK1    2208
#define HSMHV_MOD_WGLEAK2    2209
#define HSMHV_MOD_WGLEAK3    2210
#define HSMHV_MOD_WGLEAK6    2213
#define HSMHV_MOD_WGLKSD1    2215
#define HSMHV_MOD_WGLKSD2    2216
#define HSMHV_MOD_WGLKB1     2218
#define HSMHV_MOD_WGLKB2     2219
#define HSMHV_MOD_WNFTRP     2258
#define HSMHV_MOD_WNFALP     2259
#define HSMHV_MOD_WPTHROU    2253
#define HSMHV_MOD_WVDIFFJ    2254
#define HSMHV_MOD_WIBPC1     2404
#define HSMHV_MOD_WIBPC2     2405
#define HSMHV_MOD_WCGBO      2156
#define HSMHV_MOD_WCVDSOVER  2480
#define HSMHV_MOD_WFALPH     2263
#define HSMHV_MOD_WNPEXT     2242
#define HSMHV_MOD_WPOWRAT    2463
#define HSMHV_MOD_WRD        2399
#define HSMHV_MOD_WRD22      2442
#define HSMHV_MOD_WRD23      2443
#define HSMHV_MOD_WRD24      2444
#define HSMHV_MOD_WRDICT1    2316
#define HSMHV_MOD_WRDOV13    2476
#define HSMHV_MOD_WRDSLP1    2315
#define HSMHV_MOD_WRDVB      2301
#define HSMHV_MOD_WRDVD      2510
#define HSMHV_MOD_WRDVG11    2424
#define HSMHV_MOD_WRS        2398
#define HSMHV_MOD_WRTH0      2432
#define HSMHV_MOD_WVOVER     2199

/* Cross-term dependence */
#define HSMHV_MOD_PVMAX      3100
#define HSMHV_MOD_PBGTMP1    3101
#define HSMHV_MOD_PBGTMP2    3102
#define HSMHV_MOD_PEG0       3103
#define HSMHV_MOD_PVFBOVER   3428
#define HSMHV_MOD_PNOVER     3430
#define HSMHV_MOD_PNOVERS    3431
#define HSMHV_MOD_PWL2       3407
#define HSMHV_MOD_PVFBC      3121
#define HSMHV_MOD_PNSUBC     3123
#define HSMHV_MOD_PNSUBP     3181
#define HSMHV_MOD_PSCP1      3184
#define HSMHV_MOD_PSCP2      3185
#define HSMHV_MOD_PSCP3      3186
#define HSMHV_MOD_PSC1       3126
#define HSMHV_MOD_PSC2       3127
#define HSMHV_MOD_PSC3       3128
#define HSMHV_MOD_PPGD1      3187
#define HSMHV_MOD_PPGD3      3189
#define HSMHV_MOD_PNDEP      3129
#define HSMHV_MOD_PNINV      3130
#define HSMHV_MOD_PMUECB0    3131
#define HSMHV_MOD_PMUECB1    3132
#define HSMHV_MOD_PMUEPH1    3133
#define HSMHV_MOD_PVTMP      3141
#define HSMHV_MOD_PWVTH0     3142
#define HSMHV_MOD_PMUESR1    3143
#define HSMHV_MOD_PMUETMP    3195
#define HSMHV_MOD_PSUB1      3151
#define HSMHV_MOD_PSUB2      3152
#define HSMHV_MOD_PSVDS      3286
#define HSMHV_MOD_PSVBS      3284
#define HSMHV_MOD_PSVGS      3283
#define HSMHV_MOD_PFN1       3294
#define HSMHV_MOD_PFN2       3295
#define HSMHV_MOD_PFN3       3296
#define HSMHV_MOD_PFVBS      3297
#define HSMHV_MOD_PNSTI      3225
#define HSMHV_MOD_PWSTI      3226
#define HSMHV_MOD_PSCSTI1    3229
#define HSMHV_MOD_PSCSTI2    3230
#define HSMHV_MOD_PVTHSTI    3232
#define HSMHV_MOD_PMUESTI1   3235
#define HSMHV_MOD_PMUESTI2   3236
#define HSMHV_MOD_PMUESTI3   3237
#define HSMHV_MOD_PNSUBPSTI1 3238
#define HSMHV_MOD_PNSUBPSTI2 3239
#define HSMHV_MOD_PNSUBPSTI3 3240
#define HSMHV_MOD_PCGSO      3154
#define HSMHV_MOD_PCGDO      3155
#define HSMHV_MOD_PJS0       3157
#define HSMHV_MOD_PJS0SW     3158
#define HSMHV_MOD_PNJ        3159
#define HSMHV_MOD_PCISBK     3172
#define HSMHV_MOD_PCLM1      3191
#define HSMHV_MOD_PCLM2      3192
#define HSMHV_MOD_PCLM3      3193
#define HSMHV_MOD_PWFC       3201
#define HSMHV_MOD_PGIDL1     3205
#define HSMHV_MOD_PGIDL2     3206
#define HSMHV_MOD_PGLEAK1    3208
#define HSMHV_MOD_PGLEAK2    3209
#define HSMHV_MOD_PGLEAK3    3210
#define HSMHV_MOD_PGLEAK6    3213
#define HSMHV_MOD_PGLKSD1    3215
#define HSMHV_MOD_PGLKSD2    3216
#define HSMHV_MOD_PGLKB1     3218
#define HSMHV_MOD_PGLKB2     3219
#define HSMHV_MOD_PNFTRP     3258
#define HSMHV_MOD_PNFALP     3259
#define HSMHV_MOD_PPTHROU    3253
#define HSMHV_MOD_PVDIFFJ    3254
#define HSMHV_MOD_PIBPC1     3404
#define HSMHV_MOD_PIBPC2     3405
#define HSMHV_MOD_PCGBO      3156
#define HSMHV_MOD_PCVDSOVER  3480
#define HSMHV_MOD_PFALPH     3263
#define HSMHV_MOD_PNPEXT     3242
#define HSMHV_MOD_PPOWRAT    3463
#define HSMHV_MOD_PRD        3399
#define HSMHV_MOD_PRD22      3442
#define HSMHV_MOD_PRD23      3443
#define HSMHV_MOD_PRD24      3444
#define HSMHV_MOD_PRDICT1    3316
#define HSMHV_MOD_PRDOV13    3476
#define HSMHV_MOD_PRDSLP1    3315
#define HSMHV_MOD_PRDVB      3301
#define HSMHV_MOD_PRDVD      3510
#define HSMHV_MOD_PRDVG11    3424
#define HSMHV_MOD_PRS        3398
#define HSMHV_MOD_PRTH0      3432
#define HSMHV_MOD_PVOVER     3199

/* device requests */
#define HSMHV_DNODE          341
#define HSMHV_GNODE          342
#define HSMHV_SNODE          343
#define HSMHV_BNODE          344
/* #define HSMHV_TEMPNODE       345	not used */
#define HSMHV_DNODEPRIME     346
#define HSMHV_SNODEPRIME     347
/* #define HSMHV_BNODEPRIME     395	not used */
/* #define HSMHV_DBNODE         396	not used */
/* #define HSMHV_SBNODE         397	not used */
/* #define HSMHV_VBD            347 */
#define HSMHV_VBD            466
#define HSMHV_VBS            348
#define HSMHV_VGS            349
#define HSMHV_VDS            350
#define HSMHV_CD             351
#define HSMHV_CBS            352
#define HSMHV_CBD            353
#define HSMHV_GM             354
#define HSMHV_GDS            355
#define HSMHV_GMBS           356
#define HSMHV_GMT            465
/* #define HSMHV_ISUBT          466 */
#define HSMHV_GBD            357
#define HSMHV_GBS            358
#define HSMHV_QB             359
#define HSMHV_CQB            360
/* #define HSMHV_QTH            467	not used */
/* #define HSMHV_CQTH           468	not used */
/* #define HSMHV_CTH            469	not used */
#define HSMHV_QG             361
#define HSMHV_CQG            362
#define HSMHV_QD             363
#define HSMHV_CQD            364
#define HSMHV_CGG            365
#define HSMHV_CGD            366
#define HSMHV_CGS            367
#define HSMHV_CBG            368
#define HSMHV_CAPBD          369
/* #define HSMHV_CQBD           370	not used */
#define HSMHV_CAPBS          371
/* #define HSMHV_CQBS           372	not used */
#define HSMHV_CDG            373
#define HSMHV_CDD            374
#define HSMHV_CDS            375
#define HSMHV_VON            376
#define HSMHV_VDSAT          377
#define HSMHV_QBS            378
#define HSMHV_QBD            379
#define HSMHV_SOURCECONDUCT  380
#define HSMHV_DRAINCONDUCT   381
#define HSMHV_CBDB           382
#define HSMHV_CBSB           383
#define HSMHV_MOD_RBPB       389
#define HSMHV_MOD_RBPD       390
#define HSMHV_MOD_RBPS       391
#define HSMHV_MOD_RBDB       392
#define HSMHV_MOD_RBSB       393
#define HSMHV_MOD_GBMIN      394

#define HSMHV_ISUB           410
#define HSMHV_IGIDL          411
#define HSMHV_IGISL          412
#define HSMHV_IGD            413
#define HSMHV_IGS            414
#define HSMHV_IGB            415
#define HSMHV_CGSO           416
#define HSMHV_CGBO           417
#define HSMHV_CGDO           418

#define HSMHV_MOD_TCJBD         92
#define HSMHV_MOD_TCJBS         93
#define HSMHV_MOD_TCJBDSW       94 
#define HSMHV_MOD_TCJBSSW       95  
#define HSMHV_MOD_TCJBDSWG      96   
#define HSMHV_MOD_TCJBSSWG      97   

#define HSMHV_MOD_VGS_MAX          4001
#define HSMHV_MOD_VGD_MAX          4002
#define HSMHV_MOD_VGB_MAX          4003
#define HSMHV_MOD_VDS_MAX          4004
#define HSMHV_MOD_VBS_MAX          4005
#define HSMHV_MOD_VBD_MAX          4006
#define HSMHV_MOD_VGSR_MAX         4007
#define HSMHV_MOD_VGDR_MAX         4008
#define HSMHV_MOD_VGBR_MAX         4009
#define HSMHV_MOD_VBSR_MAX         4010
#define HSMHV_MOD_VBDR_MAX         4011

#include "hsmhvext.h"

/* Prototype has to be adapted! 
extern void HSMHVevaluate(double,double,double,HSMHVinstance*,HSMHVmodel*,
        double*,double*,double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, CKTcircuit*);
*/

#endif /*HSMHV*/

