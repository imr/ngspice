/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 2  SUBVERSION : 2  REVISION : 0 ) 
 Model Parameter 'VERSION' : 2.20
 FILE : hsmhvdef

 DATE : 2014.6.11

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

/**********************************************************************

The following source code, and all copyrights, trade secrets or other
intellectual property rights in and to the source code in its entirety,
is owned by the Hiroshima University and the STARC organization.

All users need to follow the "HISIM_HV Distribution Statement and
Copyright Notice" attached to HiSIM_HV model.

-----HISIM_HV Distribution Statement and Copyright Notice--------------

Software is distributed as is, completely without warranty or service
support. Hiroshima University or STARC and its employees are not liable
for the condition or performance of the software.

Hiroshima University and STARC own the copyright and grant users a perpetual,
irrevocable, worldwide, non-exclusive, royalty-free license with respect 
to the software as set forth below.   

Hiroshima University and STARC hereby disclaims all implied warranties.

Hiroshima University and STARC grant the users the right to modify, copy,
and redistribute the software and documentation, both within the user's
organization and externally, subject to the following restrictions

1. The users agree not to charge for Hiroshima University and STARC code
itself but may charge for additions, extensions, or support.

2. In any product based on the software, the users agree to acknowledge
Hiroshima University and STARC that developed the software. This
acknowledgment shall appear in the product documentation.

3. The users agree to reproduce any copyright notice which appears on
the software on any copy or modification of such made available
to others."

Toshimasa Asahara, President, Hiroshima University
Mitiko Miura-Mattausch, Professor, Hiroshima University
Katsuhiro Shimohigashi, President&CEO, STARC
June 2008 (revised October 2011) 
*************************************************************************/

#ifndef HSMHV2
#define HSMHV2

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

/* declarations for HiSIMHV MOSFETs */

/* unit-converted model parameters */
typedef struct sHSMHV2modelMKSParam {
  double HSMHV2_npext ;
  double HSMHV2_vmax ;
  double HSMHV2_ll ;
  double HSMHV2_wl ;
  double HSMHV2_svgsl ;
  double HSMHV2_svgsw ;
  double HSMHV2_svbsl ;
  double HSMHV2_slgl ;
  double HSMHV2_sub1l ;
  double HSMHV2_slg ;
  double HSMHV2_sub2l ;
  double HSMHV2_subld2 ;
  double HSMHV2_rd22 ;
  double HSMHV2_rd23 ;
  double HSMHV2_rd24 ;
  double HSMHV2_rdtemp1 ;
  double HSMHV2_rdtemp2 ;
  double HSMHV2_rdvd ;
  double HSMHV2_rdvdtemp1 ;
  double HSMHV2_rdvdtemp2 ;
  double HSMHV2_nsubsub ;
  double HSMHV2_nsubpsti1 ;
  double HSMHV2_muesti1 ;
  double HSMHV2_wfc ;
  double HSMHV2_glksd1 ;
  double HSMHV2_glksd2 ;
  double HSMHV2_glksd3 ;
  double HSMHV2_gleak2 ;
  double HSMHV2_gleak4 ;
  double HSMHV2_gleak5 ;
  double HSMHV2_gleak7 ;
  double HSMHV2_glkb2 ;
  double HSMHV2_fn2 ;
  double HSMHV2_gidl1 ;
  double HSMHV2_gidl2 ;
  double HSMHV2_nfalp ;
  double HSMHV2_nftrp ;
  double HSMHV2_cit ;
  double HSMHV2_ovslp ;
  double HSMHV2_dly3 ;
  double HSMHV2_rth0 ;
  double HSMHV2_cth0 ;
  double HSMHV2_rdrmue ; 
  double HSMHV2_rdrvmax ; 
  double HSMHV2_ndepm ;
  double HSMHV2_depvmax ;

} HSMHV2modelMKSParam ;


/* binning parameters */
typedef struct sHSMHV2binningParam {
  double HSMHV2_vmax ;
  double HSMHV2_bgtmp1 ;
  double HSMHV2_bgtmp2 ;
  double HSMHV2_eg0 ;
  double HSMHV2_vfbover ;
  double HSMHV2_nover ;
  double HSMHV2_novers ;
  double HSMHV2_wl2 ;
  double HSMHV2_vfbc ;
  double HSMHV2_nsubc ;
  double HSMHV2_nsubp ;
  double HSMHV2_scp1 ;
  double HSMHV2_scp2 ;
  double HSMHV2_scp3 ;
  double HSMHV2_sc1 ;
  double HSMHV2_sc2 ;
  double HSMHV2_sc3 ;
  double HSMHV2_pgd1 ;
//double HSMHV2_pgd3 ;
  double HSMHV2_ndep ;
  double HSMHV2_ninv ;
  double HSMHV2_muecb0 ;
  double HSMHV2_muecb1 ;
  double HSMHV2_mueph1 ;
  double HSMHV2_vtmp ;
  double HSMHV2_wvth0 ;
  double HSMHV2_muesr1 ;
  double HSMHV2_muetmp ;
  double HSMHV2_sub1 ;
  double HSMHV2_sub2 ;
  double HSMHV2_svds ;
  double HSMHV2_svbs ;
  double HSMHV2_svgs ;
  double HSMHV2_fn1 ;
  double HSMHV2_fn2 ;
  double HSMHV2_fn3 ;
  double HSMHV2_fvbs ;
  double HSMHV2_nsti ;
  double HSMHV2_wsti ;
  double HSMHV2_scsti1 ;
  double HSMHV2_scsti2 ;
  double HSMHV2_vthsti ;
  double HSMHV2_muesti1 ;
  double HSMHV2_muesti2 ;
  double HSMHV2_muesti3 ;
  double HSMHV2_nsubpsti1 ;
  double HSMHV2_nsubpsti2 ;
  double HSMHV2_nsubpsti3 ;
  double HSMHV2_cgso ;
  double HSMHV2_cgdo ;
  double HSMHV2_js0 ;
  double HSMHV2_js0sw ;
  double HSMHV2_nj ;
  double HSMHV2_cisbk ;
  double HSMHV2_clm1 ;
  double HSMHV2_clm2 ;
  double HSMHV2_clm3 ;
  double HSMHV2_wfc ;
  double HSMHV2_gidl1 ;
  double HSMHV2_gidl2 ;
  double HSMHV2_gleak1 ;
  double HSMHV2_gleak2 ;
  double HSMHV2_gleak3 ;
  double HSMHV2_gleak6 ;
  double HSMHV2_glksd1 ;
  double HSMHV2_glksd2 ;
  double HSMHV2_glkb1 ;
  double HSMHV2_glkb2 ;
  double HSMHV2_nftrp ;
  double HSMHV2_nfalp ;
  double HSMHV2_vdiffj ;
  double HSMHV2_ibpc1 ;
  double HSMHV2_ibpc2 ;
  double HSMHV2_cgbo ;
  double HSMHV2_cvdsover ;
  double HSMHV2_falph ;
  double HSMHV2_npext ;
  double HSMHV2_powrat ;
  double HSMHV2_rd ;
  double HSMHV2_rd22 ;
  double HSMHV2_rd23 ;
  double HSMHV2_rd24 ;
  double HSMHV2_rdict1 ;
  double HSMHV2_rdov13 ;
  double HSMHV2_rdslp1 ;
  double HSMHV2_rdvb ;
  double HSMHV2_rdvd ;
  double HSMHV2_rdvg11 ;
  double HSMHV2_rs ;
  double HSMHV2_rth0 ;
  double HSMHV2_vover ;
  /*-----------SHE--------------*/
  double HSMHV2_rth ;
  double HSMHV2_cth ;
  double HSMHV2_js0d;
  double HSMHV2_js0swd;
  double HSMHV2_njd;
  double HSMHV2_cisbkd;
  double HSMHV2_vdiffjd;
  double HSMHV2_js0s;
  double HSMHV2_js0sws;
  double HSMHV2_njs;
  double HSMHV2_cisbks;
  double HSMHV2_vdiffjs;
  /*-----------------------------*/

} HSMHV2binningParam ;

 
/* unit-converted parameters for each instance */
typedef struct sHSMHV2hereMKSParam {
  double HSMHV2_vmax ;
  double HSMHV2_subld2 ;
  double HSMHV2_ndep ;
  double HSMHV2_ninv ;
  double HSMHV2_nsubc ;
  double HSMHV2_nsubcdfm ;
  double HSMHV2_nsubp ;
  double HSMHV2_nsubpsti1 ;
  double HSMHV2_muesti1 ;
  double HSMHV2_nsti ;
  double HSMHV2_npext ;
  double HSMHV2_nover ;
  double HSMHV2_novers ;
  double HSMHV2_wfc ;
  double HSMHV2_glksd1 ;
  double HSMHV2_glksd2 ;
  double HSMHV2_gleak2 ;
  double HSMHV2_glkb2 ;
  double HSMHV2_fn2 ;
  double HSMHV2_gidl1 ;
  double HSMHV2_gidl2 ;
  double HSMHV2_nfalp ;
  double HSMHV2_nftrp ;
} HSMHV2hereMKSParam ;


/* information needed for each instance */
typedef struct sHSMHV2instance {

  struct GENinstance gen;

#define HSMHV2modPtr(inst) ((struct sHSMHV2model *)((inst)->gen.GENmodPtr))
#define HSMHV2nextInstance(inst) ((struct sHSMHV2instance *)((inst)->gen.GENnextInstance))
#define HSMHV2name gen.GENname
#define HSMHV2states gen.GENstate

  const int HSMHV2dNode;      /* number of the drain node of the mosfet */
  const int HSMHV2gNode;      /* number of the gate node of the mosfet */
  const int HSMHV2sNode;      /* number of the source node of the mosfet */
  const int HSMHV2bNode;      /* number of the bulk node of the mosfet */
  const int HSMHV2subNodeExt; /* number of the substrate node */
  const int HSMHV2tempNodeExt;/* number of the temp node----------SHE--------*/
  int HSMHV2subNode;    /* number of the substrate node */
  int HSMHV2tempNode;   /* number of the temp node */
  int HSMHV2dNodePrime; /* number od the inner drain node */
  int HSMHV2gNodePrime; /* number of the inner gate node */
  int HSMHV2sNodePrime; /* number od the inner source node */
  int HSMHV2bNodePrime;
  int HSMHV2dbNode;
  int HSMHV2sbNode;
  int HSMHV2qiNode;     /* number of the qi node in case of NQS */
  int HSMHV2qbNode;     /* number of the qb node in case of NQS */

  double HSMHV2_noiflick; /* for 1/f noise calc. */
  double HSMHV2_noithrml; /* for thrmal noise calc. */
  double HSMHV2_noiigate; /* for induced gate noise */
  double HSMHV2_noicross; /* for induced gate noise */

  /* instance */
  int HSMHV2_coselfheat; /* Self-heating model */
  int HSMHV2_cosubnode;  /* switch tempNode to subNode */
  double HSMHV2_l;    /* the length of the channel region */
  double HSMHV2_w;    /* the width of the channel region */
  double HSMHV2_ad;   /* the area of the drain diffusion */
  double HSMHV2_as;   /* the area of the source diffusion */
  double HSMHV2_pd;   /* perimeter of drain junction [m] */
  double HSMHV2_ps;   /* perimeter of source junction [m] */
  double HSMHV2_nrd;  /* equivalent num of squares of drain [-] (unused) */
  double HSMHV2_nrs;  /* equivalent num of squares of source [-] (unused) */
  double HSMHV2_dtemp;

  double HSMHV2_weff;    /* the effective width of the channel region */
  double HSMHV2_weff_ld; /* the effective width of the drift region */
  double HSMHV2_weff_cv;  /* the effective width of the drift region for capacitance */
  double HSMHV2_weff_nf; /* Weff * NF */
  double HSMHV2_weffcv_nf;  /* Weffcv * NF */
  double HSMHV2_leff;    /* the effective length of the channel region */

  int HSMHV2_corbnet  ;
  double HSMHV2_rbpb ;
  double HSMHV2_rbpd ;
  double HSMHV2_rbps ;
  double HSMHV2_rbdb ;
  double HSMHV2_rbsb ;

  int HSMHV2_corg ;
  double HSMHV2_ngcon;
  double HSMHV2_xgw;
  double HSMHV2_xgl;
  double HSMHV2_nf;

  double HSMHV2_sa;
  double HSMHV2_sb;
  double HSMHV2_sd;
  double HSMHV2_nsubcdfm;
  double HSMHV2_m;
  double HSMHV2_subld1;
  double HSMHV2_subld2;
  double HSMHV2_lover;  
  double HSMHV2_lovers;  
  double HSMHV2_loverld;
  double HSMHV2_ldrift1;
  double HSMHV2_ldrift2;
  double HSMHV2_ldrift1s;
  double HSMHV2_ldrift2s;

  int HSMHV2_called; /* flag to check the first call */
  /* previous values to evaluate initial guess */
  double HSMHV2_mode_prv;
  double HSMHV2_vbsc_prv;
  double HSMHV2_vdsc_prv;
  double HSMHV2_vgsc_prv;
  double HSMHV2_ps0_prv;
  double HSMHV2_ps0_dvbs_prv;
  double HSMHV2_ps0_dvds_prv;
  double HSMHV2_ps0_dvgs_prv;
  double HSMHV2_ps0_dtemp_prv;
  double HSMHV2_pds_prv;
  double HSMHV2_pds_dvbs_prv;
  double HSMHV2_pds_dvds_prv;
  double HSMHV2_pds_dvgs_prv;
  double HSMHV2_pds_dtemp_prv;
  /* double HSMHV2_ids_prv;		not used */
  /* double HSMHV2_ids_dvbs_prv;		not used */
  /* double HSMHV2_ids_dvds_prv;		not used */
  /* double HSMHV2_ids_dvgs_prv;		not used */
  /* double HSMHV2_ids_dtemp_prv;	not used */
  double HSMHV2_mode_prv2;
  double HSMHV2_vbsc_prv2;
  double HSMHV2_vdsc_prv2;
  double HSMHV2_vgsc_prv2;
  double HSMHV2_ps0_prv2;	/* assigned but not used */
  double HSMHV2_ps0_dvbs_prv2;
  double HSMHV2_ps0_dvds_prv2;
  double HSMHV2_ps0_dvgs_prv2;
  double HSMHV2_pds_prv2;	/* assigned but not used */
  double HSMHV2_pds_dvbs_prv2;
  double HSMHV2_pds_dvds_prv2;
  double HSMHV2_pds_dvgs_prv2;
  double HSMHV2_temp_prv;
/*   double HSMHV2_time; /\* for debug print *\/ */

  /* output */
  /* int    HSMHV2_capop;	not used */
  /* double HSMHV2_gd;		not used */
  /* double HSMHV2_gs;		not used */
  double HSMHV2_cgso;		/* can be made local */
  double HSMHV2_cgdo;
  double HSMHV2_cgbo;
  double HSMHV2_cggo;
  double HSMHV2_cdso;		/* can be made local */
  double HSMHV2_cddo;
  double HSMHV2_cdgo;
  double HSMHV2_cdbo;
  double HSMHV2_csso;		/* can be made local */
  double HSMHV2_csdo;		/* can be made local */
  double HSMHV2_csgo;		/* can be made local */
  double HSMHV2_csbo;		/* can be made local */
  double HSMHV2_cbdo;
  double HSMHV2_cbgo;
  double HSMHV2_cbbo;
  /* double HSMHV2_cqyd;		not used */
  /* double HSMHV2_cqyg;		not used */
  /* double HSMHV2_cqyb;		not used */
  double HSMHV2_von; /* vth */
  double HSMHV2_vdsat;
  /* double HSMHV2_capgs;	not used */
  /* double HSMHV2_capgd;	not used */
  /* double HSMHV2_capgb;	not used */

  /* double HSMHV2_rth0;		not used */
  /* double HSMHV2_cth0;		not used */
  /* double HSMHV2_cth;		not used */


#define XDIM 14
  double HSMHV2_ydc_d[XDIM],  HSMHV2_ydc_dP[XDIM], HSMHV2_ydc_g[XDIM],  HSMHV2_ydc_gP[XDIM], HSMHV2_ydc_s[XDIM], HSMHV2_ydc_sP[XDIM], 
         HSMHV2_ydc_bP[XDIM], HSMHV2_ydc_b[XDIM],  HSMHV2_ydc_db[XDIM], HSMHV2_ydc_sb[XDIM], HSMHV2_ydc_t[XDIM], HSMHV2_ydc_qi[XDIM],
         HSMHV2_ydc_qb[XDIM];
  double HSMHV2_ydyn_d[XDIM],  HSMHV2_ydyn_dP[XDIM], HSMHV2_ydyn_g[XDIM],  HSMHV2_ydyn_gP[XDIM], HSMHV2_ydyn_s[XDIM], HSMHV2_ydyn_sP[XDIM], 
         HSMHV2_ydyn_bP[XDIM], HSMHV2_ydyn_b[XDIM],  HSMHV2_ydyn_db[XDIM], HSMHV2_ydyn_sb[XDIM], HSMHV2_ydyn_t[XDIM], HSMHV2_ydyn_qi[XDIM],
         HSMHV2_ydyn_qb[XDIM];

  /* resistances */
  double HSMHV2_Rd ; /* different from HSMHV2_rd */
  double HSMHV2_dRd_dVdse  ;
  double HSMHV2_dRd_dVgse  ;
  double HSMHV2_dRd_dVbse  ;
  double HSMHV2_dRd_dVddp  ;
  double HSMHV2_dRd_dVsubs ;
  double HSMHV2_dRd_dTi    ;
  double HSMHV2_dRd_dVds ;
  double HSMHV2_dRd_dVgs ;
  double HSMHV2_dRd_dVbs ;
  double HSMHV2_Rs ; /* different from HSMHV2_rs */
  double HSMHV2_dRs_dVdse  ;
  double HSMHV2_dRs_dVgse  ;
  double HSMHV2_dRs_dVbse  ;
  double HSMHV2_dRs_dVsubs ;
  double HSMHV2_dRs_dTi    ;
  /* drain current */
  double HSMHV2_ids;
  double HSMHV2_gds;		/* used for printout, but not loaded */
  double HSMHV2_gm;		/* used for printout, but not loaded */
  double HSMHV2_gmbs;		/* used for printout, but not loaded */
  double HSMHV2_dIds_dVdse ;
  double HSMHV2_dIds_dVgse ;
  double HSMHV2_dIds_dVbse ;
  double HSMHV2_dIds_dVdsi ;
  double HSMHV2_dIds_dVgsi ;
  double HSMHV2_dIds_dVbsi ;
  double HSMHV2_dIds_dTi   ;
  /* substrate current */
  double HSMHV2_isub;
  /* double HSMHV2_gbgs;		not used */
  /* double HSMHV2_gbds;		not used */
  /* double HSMHV2_gbbs;		not used */
  double HSMHV2_dIsub_dVdsi ;
  double HSMHV2_dIsub_dVgsi ;
  double HSMHV2_dIsub_dVbsi ;
  double HSMHV2_dIsub_dTi   ;
  double HSMHV2_isubld;
  double HSMHV2_dIsubLD_dVdsi ;
  double HSMHV2_dIsubLD_dVgsi ;
  double HSMHV2_dIsubLD_dVbsi ;
  double HSMHV2_dIsubLD_dTi   ;
  double HSMHV2_dIsubLD_dVddp ;
  double HSMHV2_idsibpc;
  double HSMHV2_dIdsIBPC_dVdsi ;
  double HSMHV2_dIdsIBPC_dVgsi ;
  double HSMHV2_dIdsIBPC_dVbsi ;
  double HSMHV2_dIdsIBPC_dTi   ;
  double HSMHV2_dIdsIBPC_dVddp ;
  /* gidl and gisl current */
  double HSMHV2_igidl; /* gate induced drain leakage */
  /* double HSMHV2_gigidlgs;	not used */
  /* double HSMHV2_gigidlds;	not used */
  /* double HSMHV2_gigidlbs;	not used */
  double HSMHV2_dIgidl_dVdsi ;
  double HSMHV2_dIgidl_dVgsi ;
  double HSMHV2_dIgidl_dVbsi ;
  double HSMHV2_dIgidl_dTi   ;
  double HSMHV2_igisl; /* gate induced source leakage */
  /* double HSMHV2_gigislgd;	not used */
  /* double HSMHV2_gigislsd;	not used */
  /* double HSMHV2_gigislbd;	not used */
  double HSMHV2_dIgisl_dVdsi ;
  double HSMHV2_dIgisl_dVgsi ;
  double HSMHV2_dIgisl_dVbsi ;
  double HSMHV2_dIgisl_dTi   ;
  /* gate leakage currents */
  double HSMHV2_igb; /* gate tunneling current (gate to bulk) */
  /* double HSMHV2_gigbg;	not used */
  /* double HSMHV2_gigbd;	not used */
  /* double HSMHV2_gigbb;	not used */
  /* double HSMHV2_gigbs;	not used */
  double HSMHV2_dIgb_dVdsi ;
  double HSMHV2_dIgb_dVgsi ;
  double HSMHV2_dIgb_dVbsi ;
  double HSMHV2_dIgb_dTi   ;
  double HSMHV2_igd; /* gate tunneling current (gate to drain) */
  /* double HSMHV2_gigdg;	not used */
  /* double HSMHV2_gigdd;	not used */
  /* double HSMHV2_gigdb;	not used */
  /* double HSMHV2_gigds;	not used */
  double HSMHV2_dIgd_dVdsi ;
  double HSMHV2_dIgd_dVgsi ;
  double HSMHV2_dIgd_dVbsi ;
  double HSMHV2_dIgd_dTi   ;
  double HSMHV2_igs; /* gate tunneling current (gate to source) */
  /* double HSMHV2_gigsg;	not used */
  /* double HSMHV2_gigsd;	not used */
  /* double HSMHV2_gigsb;	not used */
  /* double HSMHV2_gigss;	not used */
  double HSMHV2_dIgs_dVdsi ;
  double HSMHV2_dIgs_dVgsi ;
  double HSMHV2_dIgs_dVbsi ;
  double HSMHV2_dIgs_dTi   ;
  /* charges */
  double HSMHV2_qd;
  double HSMHV2_cdgb;		/* used for printout, but not loaded */
  /* double HSMHV2_cddb;		not used */
  /* double HSMHV2_cdsb;         not used */
  /* double HSMHV2cdT;		not used */
  double HSMHV2_dQdi_dVdsi ;
  double HSMHV2_dQdi_dVgsi ;
  double HSMHV2_dQdi_dVbsi ;
  double HSMHV2_dQdi_dTi   ;
  double HSMHV2_qg;
  double HSMHV2_cggb;		/* used for printout, but not loaded */
  double HSMHV2_cgdb;		/* used for printout, but not loaded */
  double HSMHV2_cgsb;		/* used for printout, but not loaded */
  /* double HSMHV2cgT;		not used */
  double HSMHV2_dQg_dVdsi ;
  double HSMHV2_dQg_dVgsi ;
  double HSMHV2_dQg_dVbsi ;
  double HSMHV2_dQg_dTi   ;
  double HSMHV2_qs;
  double HSMHV2_dQsi_dVdsi ;
  double HSMHV2_dQsi_dVgsi ;
  double HSMHV2_dQsi_dVbsi ;
  double HSMHV2_dQsi_dTi   ;
  double HSMHV2_qb;  /* bulk charge qb = -(qg + qd + qs) */
  double HSMHV2_cbgb;		/* used for printout, but not loaded */
  /* double HSMHV2_cbdb;		not used */
  /* double HSMHV2_cbsb;		not used */
  /* double HSMHV2cbT;		not used */
  double HSMHV2_dQb_dVdsi ;     /* Qb: bulk charge inclusive overlaps, Qbulk: bulk charge without overlaps (see above) */
  double HSMHV2_dQb_dVgsi ;
  double HSMHV2_dQb_dVbsi ;
  double HSMHV2_dQb_dTi   ;
  /* outer charges (fringing etc.) */
  double HSMHV2_qdp ;
  double HSMHV2_dqdp_dVdse ;
  double HSMHV2_dqdp_dVgse ;
  double HSMHV2_dqdp_dVbse ;
  double HSMHV2_dqdp_dTi   ;
  double HSMHV2_qsp ;
  double HSMHV2_dqsp_dVdse ;
  double HSMHV2_dqsp_dVgse ;
  double HSMHV2_dqsp_dVbse ;
  double HSMHV2_dqsp_dTi   ;
  double HSMHV2_qgext ;
  double HSMHV2_dQgext_dVdse ;
  double HSMHV2_dQgext_dVgse ;
  double HSMHV2_dQgext_dVbse ;
  double HSMHV2_dQgext_dTi   ;
  double HSMHV2_qdext ;
  double HSMHV2_dQdext_dVdse ;
  double HSMHV2_dQdext_dVgse ;
  double HSMHV2_dQdext_dVbse ;
  double HSMHV2_dQdext_dTi   ;
  double HSMHV2_qbext ;
  double HSMHV2_dQbext_dVdse ;
  double HSMHV2_dQbext_dVgse ;
  double HSMHV2_dQbext_dVbse ;
  double HSMHV2_dQbext_dTi   ;
  double HSMHV2_qsext ;
  double HSMHV2_dQsext_dVdse ;
  double HSMHV2_dQsext_dVgse ;
  double HSMHV2_dQsext_dVbse ;
  double HSMHV2_dQsext_dTi   ;
  /* junctions */
  double HSMHV2_ibd;
  double HSMHV2_gbd;
  double HSMHV2_gbdT;
  double HSMHV2_ibs;
  double HSMHV2_gbs;
  double HSMHV2_gbsT;
  double HSMHV2_qbd;
  double HSMHV2_capbd;
  double HSMHV2_gcbdT;
  double HSMHV2_qbs;
  double HSMHV2_capbs;
  double HSMHV2_gcbsT;

  /* double HSMHV2_gtempg;	not used */
  /* double HSMHV2_gtempt;	not used */
  /* double HSMHV2_gtempd;	not used */
  /* double HSMHV2_gtempb;	not used */

  /* double HSMHV2_gmt;		not used */
  /* double HSMHV2_isubt;	not used */


  /* NQS */
  double HSMHV2_tau ;
  double HSMHV2_tau_dVgsi ;
  double HSMHV2_tau_dVdsi ;
  double HSMHV2_tau_dVbsi ;
  double HSMHV2_tau_dTi   ;
  double HSMHV2_Xd  ;
  double HSMHV2_Xd_dVgsi  ;
  double HSMHV2_Xd_dVdsi  ;
  double HSMHV2_Xd_dVbsi  ;
  double HSMHV2_Xd_dTi    ;
  double HSMHV2_Qi  ;
  double HSMHV2_Qi_dVgsi  ;
  double HSMHV2_Qi_dVdsi  ;
  double HSMHV2_Qi_dVbsi  ;
  double HSMHV2_Qi_dTi    ;
  double HSMHV2_taub  ;
  double HSMHV2_taub_dVgsi  ;
  double HSMHV2_taub_dVdsi  ;
  double HSMHV2_taub_dVbsi  ;
  double HSMHV2_taub_dTi    ;
  double HSMHV2_Qbulk  ;                          /* Qbulk: without overlaps, Qb: inclusive overlaps (see below) */
  double HSMHV2_Qbulk_dVgsi  ;
  double HSMHV2_Qbulk_dVdsi  ;
  double HSMHV2_Qbulk_dVbsi  ;
  double HSMHV2_Qbulk_dTi    ;




  /* internal variables */
  double HSMHV2_exptempd ;
  double HSMHV2_exptemps ;
  double HSMHV2_jd_nvtm_invd ;
  double HSMHV2_jd_nvtm_invs ;
  double HSMHV2_eg ;
  double HSMHV2_beta ;
  double HSMHV2_beta_inv ;
  double HSMHV2_beta2 ;
  double HSMHV2_betatnom ;
  double HSMHV2_nin ;
  double HSMHV2_egp12 ;
  double HSMHV2_egp32 ;
  double HSMHV2_lgate ;
  double HSMHV2_wg ;
  double HSMHV2_mueph ;
  double HSMHV2_mphn0 ;
  double HSMHV2_depmphn0 ;
  double HSMHV2_mphn1 ;
  double HSMHV2_depmphn1 ;
  double HSMHV2_muesr ;
  double HSMHV2_rdvd ;
  double HSMHV2_rsvd ; /* for the reverse mode */
  double HSMHV2_rd23 ;

  double HSMHV2_ninvd ;
  double HSMHV2_ninvd0 ;

  double HSMHV2_nsub ;
  double HSMHV2_qnsub ;
  double HSMHV2_qnsub_esi ;
  double HSMHV2_2qnsub_esi ;
  double HSMHV2_ptovr0 ;
  double HSMHV2_ptovr ;
  double HSMHV2_vmax0 ;
  double HSMHV2_vmax ;
  double HSMHV2_pb2 ;
  double HSMHV2_pb20 ;
  double HSMHV2_pb2c ;
  double HSMHV2_cnst0 ;
  double HSMHV2_cnst1 ;
  double HSMHV2_isbd ;
  double HSMHV2_isbd2 ;
  double HSMHV2_isbs ;
  double HSMHV2_isbs2 ;
  double HSMHV2_vbdt ;
  double HSMHV2_vbst ;
  double HSMHV2_wsti ;
  double HSMHV2_cnstpgd ;
  /* double HSMHV2_ninvp0 ;	not used */
  /* double HSMHV2_ninv0 ;	not used */
  double HSMHV2_grbpb ;
  double HSMHV2_grbpd ;
  double HSMHV2_grbps ;
  double HSMHV2_grg ;
  double HSMHV2_rs ;
  double HSMHV2_rs0 ;
  double HSMHV2_rd ;
  double HSMHV2_rd0 ;
  double HSMHV2_rdtemp0 ;
  double HSMHV2_clmmod ;
  double HSMHV2_lgatesm ;
  double HSMHV2_dVthsm ;
  double HSMHV2_ddlt ;
  double HSMHV2_xsub1 ;
  double HSMHV2_xsub2 ;
  double HSMHV2_ibpc1 ;
  double HSMHV2_xgate ;
  double HSMHV2_xvbs ;
  double HSMHV2_vg2const ;
  double HSMHV2_wdpl ;
  double HSMHV2_wdplp ;
  double HSMHV2_cfrng ;
  double HSMHV2_jd_expcd ;
  double HSMHV2_jd_expcs ;
  double HSMHV2_sqrt_eg ;

  double HSMHV2_egtnom ;
  double HSMHV2_cecox ;
  double HSMHV2_msc ;
  int HSMHV2_flg_pgd ;
  double HSMHV2_ndep_o_esi ;
  double HSMHV2_ninv_o_esi ;
  double HSMHV2_cqyb0 ;
  double HSMHV2_cnst0over ;
  double HSMHV2_cnst0overs ;
  double HSMHV2_costi00 ;
  double HSMHV2_nsti_p2 ;
  double HSMHV2_costi0 ;
  double HSMHV2_costi0_p2 ;
  double HSMHV2_costi1 ;
  double HSMHV2_ptl0;
  double HSMHV2_pt40;
  double HSMHV2_gdl0;
  double HSMHV2_rdvdtemp0 ;
  double HSMHV2_rthtemp0 ;
  double HSMHV2_powratio ;

  double HSMHV2_rdrmue ;
  double HSMHV2_rdrvmax ;

  double HSMHV2_depvmax ;

  double HSMHV2_rdrcx ;
  double HSMHV2_rdrcar ;

  double HSMHV2_xpdv ; 
  double HSMHV2_Ps0LD ;
  double HSMHV2_Ps0LD_dVds ;
  double HSMHV2_Ps0LD_dVgs ;
  double HSMHV2_Ps0LD_dVbs ;
  double HSMHV2_Ps0LD_dTi ;
  double HSMHV2_QbuLD ;
  double HSMHV2_QbuLD_dVds ;
  double HSMHV2_QbuLD_dVgs ;
  double HSMHV2_QbuLD_dVbs ;
  double HSMHV2_QbuLD_dTi ;

  double HSMHV2_kjunc ;
  double HSMHV2_kdep  ;
  double HSMHV2_Xmax  ;
  double HSMHV2_rdrcxw ;
  double HSMHV2_rdrvmaxw ;
  double HSMHV2_rdrvmaxl ;
  double HSMHV2_rdrmuel ;

  double HSMHV2_mueph1 ;
  double HSMHV2_nsubp;
  double HSMHV2_nsubc;

  double HSMHV2_Tratio;

  double HSMHV2_ndepm ;
  double HSMHV2_Pb2n ;
  double HSMHV2_Vbipn ;

  double HSMHV2_rdrbb ;

  int HSMHV2_cordrift ;

  double HSMHV2_Vdserevz ;
  double HSMHV2_Vdserevz_dVd ;
  double HSMHV2_Vsubsrev ;

  HSMHV2hereMKSParam hereMKS ; /* unit-converted parameters */

  HSMHV2binningParam pParam ; /* binning parameters */
  
  /* no use in SPICE3f5
      double HSMHV2drainSquares;       the length of the drain in squares
      double HSMHV2sourceSquares;      the length of the source in squares */
  double HSMHV2sourceConductance; /* cond. of source (or 0): set in setup */
  double HSMHV2drainConductance;  /* cond. of drain (or 0): set in setup */

  double HSMHV2_icVBS; /* initial condition B-S voltage */
  double HSMHV2_icVDS; /* initial condition D-S voltage */
  double HSMHV2_icVGS; /* initial condition G-S voltage */
  int HSMHV2_off;      /* non-zero to indicate device is off for dc analysis */
  int HSMHV2_mode;     /* device mode : 1 = normal, -1 = inverse */

  unsigned HSMHV2_coselfheat_Given :1;
  unsigned HSMHV2_cosubnode_Given :1;
  unsigned HSMHV2_l_Given :1;
  unsigned HSMHV2_w_Given :1;
  unsigned HSMHV2_ad_Given :1;
  unsigned HSMHV2_as_Given    :1;
  unsigned HSMHV2_pd_Given    :1;
  unsigned HSMHV2_ps_Given   :1;
  unsigned HSMHV2_nrd_Given  :1;
  unsigned HSMHV2_nrs_Given  :1;
  unsigned HSMHV2_dtemp_Given  :1;
  unsigned HSMHV2_icVBS_Given :1;
  unsigned HSMHV2_icVDS_Given :1;
  unsigned HSMHV2_icVGS_Given :1;
  unsigned HSMHV2_corbnet_Given  :1;
  unsigned HSMHV2_rbpb_Given :1;
  unsigned HSMHV2_rbpd_Given :1;
  unsigned HSMHV2_rbps_Given :1;
  unsigned HSMHV2_rbdb_Given :1;
  unsigned HSMHV2_rbsb_Given :1;
  unsigned HSMHV2_corg_Given  :1;
  unsigned HSMHV2_ngcon_Given  :1;
  unsigned HSMHV2_xgw_Given  :1;
  unsigned HSMHV2_xgl_Given  :1;
  unsigned HSMHV2_nf_Given  :1;
  unsigned HSMHV2_sa_Given  :1;
  unsigned HSMHV2_sb_Given  :1;
  unsigned HSMHV2_sd_Given  :1;
  unsigned HSMHV2_nsubcdfm_Given  :1;
  unsigned HSMHV2_m_Given  :1;
  unsigned HSMHV2_subld1_Given  :1;
  unsigned HSMHV2_subld2_Given  :1;
  unsigned HSMHV2_lover_Given  :1;
  unsigned HSMHV2_lovers_Given  :1;
  unsigned HSMHV2_loverld_Given  :1;
  unsigned HSMHV2_ldrift1_Given  :1;
  unsigned HSMHV2_ldrift2_Given  :1;
  unsigned HSMHV2_ldrift1s_Given :1;
  unsigned HSMHV2_ldrift2s_Given :1;

  /* unsigned HSMHV2_rth0_Given :1;	not used */
  /* unsigned HSMHV2_cth0_Given :1;	not used */

  

  /* pointers to sparse matrix */

  double *HSMHV2GgPtr;   /* pointer to sparse matrix element at (gate node,gate node) */
  double *HSMHV2GgpPtr;  /* pointer to sparse matrix element at (gate node,gate prime node) */
  /* double *HSMHV2GdpPtr;	not used */
  /* double *HSMHV2GspPtr;	not used */
  /* double *HSMHV2GbpPtr;	not used */

  double *HSMHV2GPgPtr;  /* pointer to sparse matrix element at (gate prime node,gate node) */
  double *HSMHV2GPgpPtr; /* pointer to sparse matrix element at (gate prime node,gate prime node) */
  double *HSMHV2GPdpPtr; /* pointer to sparse matrix element at (gate prime node,drain prime node) */
  double *HSMHV2GPspPtr; /* pointer to sparse matrix element at (gate prime node,source prime node) */
  double *HSMHV2GPbpPtr; /* pointer to sparse matrix element at (gate prime node,bulk prime node) */

  double *HSMHV2DPdPtr;  /* pointer to sparse matrix element at (drain prime node,drain node) */
  double *HSMHV2DPdpPtr; /* pointer to sparse matrix element at (drain prime node,drain prime node) */
  double *HSMHV2DPgpPtr; /* pointer to sparse matrix element at (drain prime node,gate prime node) */
  double *HSMHV2DPspPtr; /* pointer to sparse matrix element at (drain prime node,source prime node) */
  double *HSMHV2DPbpPtr; /* pointer to sparse matrix element at (drain prime node,bulk prime node) */

  double *HSMHV2DdPtr;   /* pointer to sparse matrix element at (Drain node,drain node) */
  double *HSMHV2DdpPtr;  /* pointer to sparse matrix element at (drain node,drain prime node) */
  double *HSMHV2DspPtr;  /* pointer to sparse matrix element at (drain node,source prime node) */
  double *HSMHV2DdbPtr;  /* pointer to sparse matrix element at (drain node,drain body node) */

  double *HSMHV2SPsPtr;  /* pointer to sparse matrix element at (source prime node,source node) */
  double *HSMHV2SPspPtr; /* pointer to sparse matrix element at (source prime node,source prime node) */
  double *HSMHV2SPgpPtr; /* pointer to sparse matrix element at (source prime node,gate prime node) */
  double *HSMHV2SPdpPtr; /* pointer to sparse matrix element at (source prime node,drain prime node) */
  double *HSMHV2SPbpPtr; /* pointer to sparse matrix element at (source prime node,bulk prime node) */

  double *HSMHV2SsPtr;   /* pointer to sparse matrix element at (source node,source node) */
  double *HSMHV2SspPtr;  /* pointer to sparse matrix element at (source node,source prime node) */
  double *HSMHV2SdpPtr;  /* pointer to sparse matrix element at (source node,drain prime node) */
  double *HSMHV2SsbPtr;  /* pointer to sparse matrix element at (source node,source body node) */

  double *HSMHV2BPgpPtr; /* pointer to sparse matrix element at (bulk prime node,gate prime node) */
  double *HSMHV2BPbpPtr; /* pointer to sparse matrix element at (bulk prime node,bulk prime node) */
  double *HSMHV2BPdPtr;  /* pointer to sparse matrix element at (bulk prime node,drain node) */
  double *HSMHV2BPdpPtr; /* pointer to sparse matrix element at (bulk prime node,drain prime node) */
  double *HSMHV2BPspPtr; /* pointer to sparse matrix element at (bulk prime node,source prime node) */
  double *HSMHV2BPsPtr;  /* pointer to sparse matrix element at (bulk prime node,source node) */
  double *HSMHV2BPbPtr;  /* pointer to sparse matrix element at (bulk prime node,bulk node) */
  double *HSMHV2BPdbPtr; /* pointer to sparse matrix element at (bulk prime node,source body node) */
  double *HSMHV2BPsbPtr; /* pointer to sparse matrix element at (bulk prime node,source body node) */

  double *HSMHV2DBdPtr;  /* pointer to sparse matrix element at (drain body node,drain node) */
  double *HSMHV2DBdbPtr; /* pointer to sparse matrix element at (drain body node,drain body node) */
  double *HSMHV2DBbpPtr; /* pointer to sparse matrix element at (drain body node,bulk prime node) */
  /* double *HSMHV2DBbPtr;	not used */
  
  double *HSMHV2SBsPtr;  /* pointer to sparse matrix element at (source body node,source node) */
  double *HSMHV2SBbpPtr; /* pointer to sparse matrix element at (source body node,bulk prime node) */
  /* double *HSMHV2SBbPtr;	not used */
  double *HSMHV2SBsbPtr; /* pointer to sparse matrix element at (source body node,source body node) */

  /* double *HSMHV2BsbPtr;	not used */
  double *HSMHV2BbpPtr;  /* pointer to sparse matrix element at (bulk node,bulk prime node) */
  /* double *HSMHV2BdbPtr;	not used */
  double *HSMHV2BbPtr;   /* pointer to sparse matrix element at (bulk node,bulk node) */
  
  double *HSMHV2TemptempPtr; /* pointer to sparse matrix element at (temp node, temp node) */
  double *HSMHV2TempdPtr;    /* pointer to sparse matrix element at (temp node, drain node) */
  double *HSMHV2TempdpPtr;   /* pointer to sparse matrix element at (temp node, drain prime node) */
  double *HSMHV2TempsPtr;    /* pointer to sparse matrix element at (temp node, source node) */
  double *HSMHV2TempspPtr;   /* pointer to sparse matrix element at (temp node, source prime node) */
  /* double *HSMHV2TempgPtr;	not used */
  double *HSMHV2TempgpPtr;   /* pointer to sparse matrix element at (temp node, gate prime node) */
  /* double *HSMHV2TempbPtr;	not used */
  double *HSMHV2TempbpPtr;   /* pointer to sparse matrix element at (temp node, bulk prime node) */
  /* double *HSMHV2GtempPtr;	not used */
  double *HSMHV2GPtempPtr;   /* pointer to sparse matrix element at (gate prime node, temp node) */
  double *HSMHV2DPtempPtr;   /* pointer to sparse matrix element at (drain prime node, temp node) */
  double *HSMHV2SPtempPtr;   /* pointer to sparse matrix element at (source prime node, temp node) */
  /* double *HSMHV2BtempPtr;	not used */
  double *HSMHV2BPtempPtr;   /* pointer to sparse matrix element at (bulk prime node, temp node) */
  double *HSMHV2DBtempPtr;   /* pointer to sparse matrix element at (drain bulk node, temp node) */
  double *HSMHV2SBtempPtr;   /* pointer to sparse matrix element at (source bulk node, temp node) */

  double *HSMHV2DgpPtr;      /* pointer to sparse matrix element at (drain node, gate prime node) */
  double *HSMHV2DsPtr;       /* pointer to sparse matrix element at (drain node, source node) */
  double *HSMHV2DbpPtr;      /* pointer to sparse matrix element at (drain node, bulk prime node) */
  double *HSMHV2DtempPtr;    /* pointer to sparse matrix element at (drain node, temp node) */
  double *HSMHV2DPsPtr;      /* pointer to sparse matrix element at (drain prime node, source node) */
  double *HSMHV2GPdPtr;      /* pointer to sparse matrix element at (gate prime node, drain node) */
  double *HSMHV2GPsPtr;      /* pointer to sparse matrix element at (gate prime node, source node) */
  double *HSMHV2SdPtr;       /* pointer to sparse matrix element at (source node, drain node) */
  double *HSMHV2SgpPtr;      /* pointer to sparse matrix element at (source node, gate prime node) */
  double *HSMHV2SbpPtr;      /* pointer to sparse matrix element at (source node, bulk prime node) */
  double *HSMHV2StempPtr;    /* pointer to sparse matrix element at (source node, temp node) */
  double *HSMHV2SPdPtr;      /* pointer to sparse matrix element at (source prime node, drain node) */

  /* nqs related pointers */
  double *HSMHV2DPqiPtr;     /* pointer to sparse matrix element at (drain prime node, qi_nqs node) */
  double *HSMHV2GPqiPtr;     /* pointer to sparse matrix element at (gate prime node, qi_nqs node) */
  double *HSMHV2GPqbPtr;     /* pointer to sparse matrix element at (gate prime node, qb_nqs node) */
  double *HSMHV2SPqiPtr;     /* pointer to sparse matrix element at (source prime node, qi_nqs node) */
  double *HSMHV2BPqbPtr;     /* pointer to sparse matrix element at (bulk prime node, qb_nqs node) */
  double *HSMHV2QIdpPtr;     /* pointer to sparse matrix element at (qi_nqs node, drain prime node) */
  double *HSMHV2QIgpPtr;     /* pointer to sparse matrix element at (qi_nqs node, gate prime node) */
  double *HSMHV2QIspPtr;     /* pointer to sparse matrix element at (qi_nqs node, source prime node) */
  double *HSMHV2QIbpPtr;     /* pointer to sparse matrix element at (qi_nqs node, bulk prime node) */
  double *HSMHV2QIqiPtr;     /* pointer to sparse matrix element at (qi_nqs node, qi_nqs node) */
  double *HSMHV2QBdpPtr;     /* pointer to sparse matrix element at (qb_nqs node, drain prime node) */
  double *HSMHV2QBgpPtr;     /* pointer to sparse matrix element at (qb_nqs node, gate prime node) */
  double *HSMHV2QBspPtr;     /* pointer to sparse matrix element at (qb_nqs node, source prime node) */
  double *HSMHV2QBbpPtr;     /* pointer to sparse matrix element at (qb_nqs node, bulk prime node) */
  double *HSMHV2QBqbPtr;     /* pointer to sparse matrix element at (qb_nqs node, qb_nqs node) */
  double *HSMHV2QItempPtr;   /* pointer to sparse matrix element at (qi_nqs node, temp node) */
  double *HSMHV2QBtempPtr;   /* pointer to sparse matrix element at (qb_nqs node, temp node) */

  /* Substrate effect related pointers */
  double *HSMHV2DsubPtr;     /* pointer to sparse matrix element at (drain node, substrate node) */
  double *HSMHV2DPsubPtr;    /* pointer to sparse matrix element at (drain prime node, substrate node) */
  double *HSMHV2SsubPtr;     /* pointer to sparse matrix element at (source node, substrate node) */
  double *HSMHV2SPsubPtr;    /* pointer to sparse matrix element at (source prime node, substrate node) */


  /* common state values in hisim module */
#define HSMHV2vbd HSMHV2states+ 0
#define HSMHV2vbs HSMHV2states+ 1
#define HSMHV2vgs HSMHV2states+ 2
#define HSMHV2vds HSMHV2states+ 3
#define HSMHV2vdbs HSMHV2states+ 4
#define HSMHV2vdbd HSMHV2states+ 5
#define HSMHV2vsbs HSMHV2states+ 6
#define HSMHV2vges HSMHV2states+ 7
#define HSMHV2vsubs HSMHV2states+ 8 /* substrate bias */
#define HSMHV2deltemp HSMHV2states+ 9
#define HSMHV2vdse HSMHV2states+ 10
#define HSMHV2vgse HSMHV2states+ 11
#define HSMHV2vbse HSMHV2states+ 12

#define HSMHV2qb  HSMHV2states+ 13
#define HSMHV2cqb HSMHV2states+ 14
#define HSMHV2qg  HSMHV2states+ 15
#define HSMHV2cqg HSMHV2states+ 16
#define HSMHV2qd  HSMHV2states+ 17
#define HSMHV2cqd HSMHV2states+ 18

#define HSMHV2qbs HSMHV2states+ 19
#define HSMHV2cqbs HSMHV2states+ 20
#define HSMHV2qbd HSMHV2states+ 21
#define HSMHV2cqbd HSMHV2states+ 22

#define HSMHV2qth HSMHV2states+ 23
#define HSMHV2cqth HSMHV2states+ 24

/*add fringing capacitance*/
#define HSMHV2qfd HSMHV2states+ 25
#define HSMHV2cqfd HSMHV2states+ 26
#define HSMHV2qfs HSMHV2states+ 27
#define HSMHV2cqfs HSMHV2states+ 28

/*add external drain capacitance*/
#define HSMHV2qdE HSMHV2states+ 29
#define HSMHV2cqdE HSMHV2states+ 30

#define HSMHV2numStates 31

/* nqs charges */
#define HSMHV2qi_nqs HSMHV2states+ 32
#define HSMHV2dotqi_nqs HSMHV2states + 33
#define HSMHV2qb_nqs HSMHV2states+ 34
#define HSMHV2dotqb_nqs HSMHV2states + 35

#define HSMHV2numStatesNqs 36

/* indices to the array of HiSIMHV NOISE SOURCES */
#define HSMHV2RDNOIZ       0
#define HSMHV2RSNOIZ       1
#define HSMHV2IDNOIZ       2
#define HSMHV2FLNOIZ       3
#define HSMHV2IGNOIZ       4
#define HSMHV2TOTNOIZ      5

#define HSMHV2NSRCS        6  /* the number of HiSIMHV MOSFET noise sources */

#ifndef NONOISE
  double HSMHV2nVar[NSTATVARS][HSMHV2NSRCS];
#else /* NONOISE */
  double **HSMHV2nVar;
#endif /* NONOISE */

} HSMHV2instance ;


/* per model data */

typedef struct sHSMHV2model {     /* model structure for a resistor */

  struct GENmodel gen;

#define HSMHV2modType gen.GENmodType
#define HSMHV2nextModel(inst) ((struct sHSMHV2model *)((inst)->gen.GENnextModel))
#define HSMHV2instances(inst) ((HSMHV2instance *)((inst)->gen.GENinstances))
#define HSMHV2modName gen.GENmodName

  int HSMHV2_type;      		/* device type: 1 = nmos,  -1 = pmos */
  int HSMHV2_level;               /* level */
  int HSMHV2_info;                /* information */
  int HSMHV2_noise;               /* noise model selecter see hsmhvnoi.c */
  char *HSMHV2_version;           /* model version */
  int HSMHV2_show;                /* show physical value 1, 2, ... , 11 */


  int HSMHV2_corsrd ;
  int HSMHV2_corg   ;
  int HSMHV2_coiprv ;
  int HSMHV2_copprv ;
  int HSMHV2_coadov ;
  int HSMHV2_coisub ;
  int HSMHV2_coiigs ;
  int HSMHV2_cogidl ;
  int HSMHV2_coovlp ;
  int HSMHV2_coovlps ;
  int HSMHV2_coflick ;
  int HSMHV2_coisti ;
  int HSMHV2_conqs  ;
  int HSMHV2_corbnet ;
  int HSMHV2_cothrml;
  int HSMHV2_coign;      /* Induced gate noise */
  int HSMHV2_codfm;      /* DFM */
  int HSMHV2_coqovsm ;
  int HSMHV2_coselfheat; /* Self-heating model */
  int HSMHV2_cosubnode;  /* switch tempNode to subNode */
  int HSMHV2_cosym;      /* Symmetry model for HV */
  int HSMHV2_cotemp;
  int HSMHV2_coldrift;
  int HSMHV2_cordrift;
  int HSMHV2_coerrrep;
  int HSMHV2_codep;
  int HSMHV2_coddlt;

  double HSMHV2_vmax ;
  double HSMHV2_vmaxt1 ;
  double HSMHV2_vmaxt2 ;
  double HSMHV2_bgtmp1 ;
  double HSMHV2_bgtmp2 ;
  double HSMHV2_eg0 ;
  double HSMHV2_tox ;
  double HSMHV2_xld ;
  double HSMHV2_xldld ;
  double HSMHV2_xwdld ;
  double HSMHV2_lover ;
  double HSMHV2_lovers ;
  double HSMHV2_rdov11 ;
  double HSMHV2_rdov12 ;
  double HSMHV2_rdov13 ;
  double HSMHV2_rdslp1 ;
  double HSMHV2_rdict1 ;
  double HSMHV2_rdslp2 ;
  double HSMHV2_rdict2 ;
  double HSMHV2_loverld ;
  double HSMHV2_ldrift1 ;
  double HSMHV2_ldrift2 ;
  double HSMHV2_ldrift1s ;
  double HSMHV2_ldrift2s ;
  double HSMHV2_subld1 ;
  double HSMHV2_subld1l ;
  double HSMHV2_subld1lp ;
  double HSMHV2_subld2 ;
  double HSMHV2_xpdv ;
  double HSMHV2_xpvdth ;
  double HSMHV2_xpvdthg ;
  double HSMHV2_ddltmax ;
  double HSMHV2_ddltslp ;
  double HSMHV2_ddltict ;
  double HSMHV2_vfbover ;
  double HSMHV2_nover ;
  double HSMHV2_novers ;
  double HSMHV2_xwd ;
  double HSMHV2_xwdc ;
  double HSMHV2_xl ;
  double HSMHV2_xw ;
  double HSMHV2_saref ;
  double HSMHV2_sbref ;
  double HSMHV2_ll ;
  double HSMHV2_lld ;
  double HSMHV2_lln ;
  double HSMHV2_wl ;
  double HSMHV2_wl1 ;
  double HSMHV2_wl1p ;
  double HSMHV2_wl2 ;
  double HSMHV2_wl2p ;
  double HSMHV2_wld ;
  double HSMHV2_wln ;
  double HSMHV2_xqy ;
  double HSMHV2_xqy1 ;
  double HSMHV2_xqy2 ;
  double HSMHV2_rs;     /* source contact resistance */
  double HSMHV2_rd;     /* drain contact resistance */
  double HSMHV2_rsh;    /* source/drain diffusion sheet resistance */
  double HSMHV2_rshg;
/*   double HSMHV2_ngcon; */
/*   double HSMHV2_xgw; */
/*   double HSMHV2_xgl; */
/*   double HSMHV2_nf; */
  double HSMHV2_vfbc ;
  double HSMHV2_vbi ;
  double HSMHV2_nsubc ;
  double HSMHV2_qdftvd ;
  double HSMHV2_parl2 ;
  double HSMHV2_lp ;
  double HSMHV2_nsubp ;
  double HSMHV2_nsubp0 ;
  double HSMHV2_nsubwp ;
  double HSMHV2_scp1 ;
  double HSMHV2_scp2 ;
  double HSMHV2_scp3 ;
  double HSMHV2_sc1 ;
  double HSMHV2_sc2 ;
  double HSMHV2_sc3 ;
  double HSMHV2_sc4 ;
  double HSMHV2_pgd1 ;
  double HSMHV2_pgd2 ;
//double HSMHV2_pgd3 ;
  double HSMHV2_pgd4 ;
  double HSMHV2_ndep ;
  double HSMHV2_ndepl ;
  double HSMHV2_ndeplp ;
  double HSMHV2_ninv ;
  double HSMHV2_ninvd ;
  double HSMHV2_ninvdw ;
  double HSMHV2_ninvdwp ;
  double HSMHV2_ninvdt1 ;
  double HSMHV2_ninvdt2 ;
  double HSMHV2_muecb0 ;
  double HSMHV2_muecb1 ;
  double HSMHV2_mueph1 ;
  double HSMHV2_mueph0 ;
  double HSMHV2_muephw ;
  double HSMHV2_muepwp ;
  double HSMHV2_muephl ;
  double HSMHV2_mueplp ;
  double HSMHV2_muephs ;
  double HSMHV2_muepsp ;
  double HSMHV2_vtmp ;
  double HSMHV2_wvth0 ;
  double HSMHV2_muesr1 ;
  double HSMHV2_muesr0 ;
  double HSMHV2_muesrw ;
  double HSMHV2_mueswp ;
  double HSMHV2_muesrl ;
  double HSMHV2_mueslp ;
  double HSMHV2_bb ;
  double HSMHV2_sub1 ;
  double HSMHV2_sub2 ;
  double HSMHV2_svgs ;
  double HSMHV2_svbs ;
  double HSMHV2_svbsl ;
  double HSMHV2_svds ;
  double HSMHV2_slg ;
  double HSMHV2_sub1l ;
  double HSMHV2_sub2l ;
  double HSMHV2_fn1 ;
  double HSMHV2_fn2 ;
  double HSMHV2_fn3 ;
  double HSMHV2_fvbs ;
  double HSMHV2_svgsl ;
  double HSMHV2_svgslp ;
  double HSMHV2_svgswp ;
  double HSMHV2_svgsw ;
  double HSMHV2_svbslp ;
  double HSMHV2_slgl ;
  double HSMHV2_slglp ;
  double HSMHV2_sub1lp ;
  double HSMHV2_nsti ;  
  double HSMHV2_wsti ;
  double HSMHV2_wstil ;
  double HSMHV2_wstilp ;
  double HSMHV2_wstiw ;
  double HSMHV2_wstiwp ;
  double HSMHV2_scsti1 ;
  double HSMHV2_scsti2 ;
  double HSMHV2_vthsti ;
  double HSMHV2_vdsti ;
  double HSMHV2_muesti1 ;
  double HSMHV2_muesti2 ;
  double HSMHV2_muesti3 ;
  double HSMHV2_nsubpsti1 ;
  double HSMHV2_nsubpsti2 ;
  double HSMHV2_nsubpsti3 ;
  double HSMHV2_lpext ;
  double HSMHV2_npext ;
  double HSMHV2_scp22 ;
  double HSMHV2_scp21 ;
  double HSMHV2_bs1 ;
  double HSMHV2_bs2 ;
  double HSMHV2_cgso ;
  double HSMHV2_cgdo ;
  double HSMHV2_cgbo ;
  double HSMHV2_tpoly ;
  double HSMHV2_js0 ;
  double HSMHV2_js0sw ;
  double HSMHV2_nj ;
  double HSMHV2_njsw ;
  double HSMHV2_xti ;
  double HSMHV2_cj ;
  double HSMHV2_cjsw ;
  double HSMHV2_cjswg ;
  double HSMHV2_mj ;
  double HSMHV2_mjsw ;
  double HSMHV2_mjswg ;
  double HSMHV2_xti2 ;
  double HSMHV2_cisb ;
  double HSMHV2_cvb ;
  double HSMHV2_ctemp ;
  double HSMHV2_cisbk ;
  double HSMHV2_cvbk ;
  double HSMHV2_divx ;
  double HSMHV2_pb ;
  double HSMHV2_pbsw ;
  double HSMHV2_pbswg ;
  double HSMHV2_clm1 ;
  double HSMHV2_clm2 ;
  double HSMHV2_clm3 ;
  double HSMHV2_clm5 ;
  double HSMHV2_clm6 ;
  double HSMHV2_muetmp ;
  double HSMHV2_vover ;
  double HSMHV2_voverp ;
  double HSMHV2_vovers ;
  double HSMHV2_voversp ;
  double HSMHV2_wfc ;
  double HSMHV2_nsubcw ;
  double HSMHV2_nsubcwp ;
  double HSMHV2_qme1 ;
  double HSMHV2_qme2 ;
  double HSMHV2_qme3 ;
  double HSMHV2_gidl1 ;
  double HSMHV2_gidl2 ;
  double HSMHV2_gidl3 ;
  double HSMHV2_gidl4 ;
  double HSMHV2_gidl5 ;
  double HSMHV2_gleak1 ;
  double HSMHV2_gleak2 ;
  double HSMHV2_gleak3 ;
  double HSMHV2_gleak4 ;
  double HSMHV2_gleak5 ;
  double HSMHV2_gleak6 ;
  double HSMHV2_gleak7 ;
  double HSMHV2_glpart1 ;
  double HSMHV2_glksd1 ;
  double HSMHV2_glksd2 ;
  double HSMHV2_glksd3 ;
  double HSMHV2_glkb1 ;
  double HSMHV2_glkb2 ;
  double HSMHV2_glkb3 ;
  double HSMHV2_egig;
  double HSMHV2_igtemp2;
  double HSMHV2_igtemp3;
  double HSMHV2_vzadd0 ;
  double HSMHV2_pzadd0 ;
  double HSMHV2_nftrp ;
  double HSMHV2_nfalp ;
  double HSMHV2_cit ;
  double HSMHV2_falph ;
  double HSMHV2_kappa ;  
  double HSMHV2_vdiffj ; 
  double HSMHV2_dly1 ;
  double HSMHV2_dly2 ;
  double HSMHV2_dly3 ;
  double HSMHV2_dlyov;
  double HSMHV2_tnom ;
  double HSMHV2_ovslp ;
  double HSMHV2_ovmag ;
  /* substrate resistances */
  double HSMHV2_gbmin;
  double HSMHV2_rbpb ;
  double HSMHV2_rbpd ;
  double HSMHV2_rbps ;
  double HSMHV2_rbdb ;
  double HSMHV2_rbsb ;
  /* IBPC */
  double HSMHV2_ibpc1 ;
  double HSMHV2_ibpc1l ;
  double HSMHV2_ibpc1lp ;
  double HSMHV2_ibpc2 ;
  /* DFM */
  double HSMHV2_mphdfm ;

  double HSMHV2_ptl, HSMHV2_ptp, HSMHV2_pt2, HSMHV2_ptlp, HSMHV2_gdl, HSMHV2_gdlp  ;

  double HSMHV2_gdld ;
  double HSMHV2_pt4 ;
  double HSMHV2_pt4p ;

  double HSMHV2_vbsmin ;
  double HSMHV2_rdvg11 ;
  double HSMHV2_rdvg12 ;
  double HSMHV2_rd20 ;
//double HSMHV2_qovsm ; 
  double HSMHV2_ldrift ; 
  double HSMHV2_rd21 ;
  double HSMHV2_rd22 ;
  double HSMHV2_rd22d ;
  double HSMHV2_rd23 ;
  double HSMHV2_rd24 ;
  double HSMHV2_rd25 ;
  double HSMHV2_rdvdl ;
  double HSMHV2_rdvdlp ;
  double HSMHV2_rdvds ;
  double HSMHV2_rdvdsp ;
  double HSMHV2_rd23l ;
  double HSMHV2_rd23lp ;
  double HSMHV2_rd23s ;
  double HSMHV2_rd23sp ;
  double HSMHV2_rds ;
  double HSMHV2_rdsp ;

  double HSMHV2_rdvd ;
  double HSMHV2_rdvb ;

  double HSMHV2_rdvsub ; /* substrate effect */
  double HSMHV2_rdvdsub ; /* substrate effect */
  double HSMHV2_ddrift ;  /* substrate effect */
  double HSMHV2_vbisub ;  /* substrate effect */
  double HSMHV2_nsubsub ; /* substrate effect */

  double HSMHV2_rth0 ;
  double HSMHV2_cth0 ;
  double HSMHV2_powrat ;

  double HSMHV2_tcjbd ;
  double HSMHV2_tcjbs ;
  double HSMHV2_tcjbdsw ;
  double HSMHV2_tcjbssw ;
  double HSMHV2_tcjbdswg ;
  double HSMHV2_tcjbsswg ;


  double HSMHV2_rdtemp1 ;
  double HSMHV2_rdtemp2 ;
  double HSMHV2_rth0r ; /* heat radiation for SHE */
  double HSMHV2_rdvdtemp1 ;
  double HSMHV2_rdvdtemp2 ;
  double HSMHV2_rth0w ;
  double HSMHV2_rth0wp ;
  double HSMHV2_rth0nf ;

  double HSMHV2_rthtemp1 ;
  double HSMHV2_rthtemp2 ;
  double HSMHV2_prattemp1 ;
  double HSMHV2_prattemp2 ;


  double HSMHV2_cvdsover ;

  double HSMHV2_rdrmue ; 
  double HSMHV2_rdrvmax ; 
  double HSMHV2_rdrmuetmp ; 
  double HSMHV2_ndepm ;
  double HSMHV2_tndep ;
  double HSMHV2_depmue0 ;
  double HSMHV2_depmue1 ;
  double HSMHV2_depmueback0 ;
  double HSMHV2_depmueback1 ;
  double HSMHV2_depvmax ;
  double HSMHV2_depvdsef1 ;
  double HSMHV2_depvdsef2 ;
  double HSMHV2_depmueph0 ;
  double HSMHV2_depmueph1 ;
  double HSMHV2_depbb ;
  double HSMHV2_depleak ;
  double HSMHV2_depeta ;
  double HSMHV2_depvtmp ;
  double HSMHV2_depmuetmp ;

  double HSMHV2_isbreak ;
  double HSMHV2_rwell ;

  double HSMHV2_rdrvtmp ; 
  //  double HSMHV2_rdrvmaxt1 ; 
  //  double HSMHV2_rdrvmaxt2 ; 
  double HSMHV2_rdrdjunc ; 
  double HSMHV2_rdrcx ; 
  double HSMHV2_rdrcar ; 
  double HSMHV2_rdrdl1 ; 
  double HSMHV2_rdrdl2 ; 
  double HSMHV2_rdrvmaxw ;
  double HSMHV2_rdrvmaxwp ;
  double HSMHV2_rdrvmaxl ;
  double HSMHV2_rdrvmaxlp ;
  double HSMHV2_rdrmuel ;
  double HSMHV2_rdrmuelp ;
  double HSMHV2_qovadd;
  double HSMHV2_rdrqover ;
  double HSMHV2_js0d;
  double HSMHV2_js0swd;
  double HSMHV2_njd;
  double HSMHV2_njswd;
  double HSMHV2_xtid;
  double HSMHV2_cjd;
  double HSMHV2_cjswd;
  double HSMHV2_cjswgd;
  double HSMHV2_mjd;
  double HSMHV2_mjswd;
  double HSMHV2_mjswgd;
  double HSMHV2_pbd;
  double HSMHV2_pbswd;
  double HSMHV2_pbswgd;
  double HSMHV2_xti2d;
  double HSMHV2_cisbd;
  double HSMHV2_cvbd;
  double HSMHV2_ctempd;
  double HSMHV2_cisbkd;
  double HSMHV2_divxd;
  double HSMHV2_vdiffjd;
  double HSMHV2_js0s;
  double HSMHV2_js0sws;
  double HSMHV2_njs;
  double HSMHV2_njsws;
  double HSMHV2_xtis;
  double HSMHV2_cjs;
  double HSMHV2_cjsws;
  double HSMHV2_cjswgs;
  double HSMHV2_mjs;
  double HSMHV2_mjsws;
  double HSMHV2_mjswgs;
  double HSMHV2_pbs;
  double HSMHV2_pbsws;
  double HSMHV2_pbswgs;
  double HSMHV2_xti2s;
  double HSMHV2_cisbs;
  double HSMHV2_cvbs;
  double HSMHV2_ctemps;
  double HSMHV2_cisbks;
  double HSMHV2_divxs;
  double HSMHV2_vdiffjs;
  double HSMHV2_shemax;
  double HSMHV2_vgsmin;
  double HSMHV2_gdsleak;
  double HSMHV2_rdrbb;
  double HSMHV2_rdrbbtmp;


  /* binning parameters */
  double HSMHV2_lmin ;
  double HSMHV2_lmax ;
  double HSMHV2_wmin ;
  double HSMHV2_wmax ;
  double HSMHV2_lbinn ;
  double HSMHV2_wbinn ;

  /* Length dependence */
  double HSMHV2_lvmax ;
  double HSMHV2_lbgtmp1 ;
  double HSMHV2_lbgtmp2 ;
  double HSMHV2_leg0 ;
  double HSMHV2_lvfbover ;
  double HSMHV2_lnover ;
  double HSMHV2_lnovers ;
  double HSMHV2_lwl2 ;
  double HSMHV2_lvfbc ;
  double HSMHV2_lnsubc ;
  double HSMHV2_lnsubp ;
  double HSMHV2_lscp1 ;
  double HSMHV2_lscp2 ;
  double HSMHV2_lscp3 ;
  double HSMHV2_lsc1 ;
  double HSMHV2_lsc2 ;
  double HSMHV2_lsc3 ;
  double HSMHV2_lpgd1 ;
//double HSMHV2_lpgd3 ;
  double HSMHV2_lndep ;
  double HSMHV2_lninv ;
  double HSMHV2_lmuecb0 ;
  double HSMHV2_lmuecb1 ;
  double HSMHV2_lmueph1 ;
  double HSMHV2_lvtmp ;
  double HSMHV2_lwvth0 ;
  double HSMHV2_lmuesr1 ;
  double HSMHV2_lmuetmp ;
  double HSMHV2_lsub1 ;
  double HSMHV2_lsub2 ;
  double HSMHV2_lsvds ;
  double HSMHV2_lsvbs ;
  double HSMHV2_lsvgs ;
  double HSMHV2_lfn1 ;
  double HSMHV2_lfn2 ;
  double HSMHV2_lfn3 ;
  double HSMHV2_lfvbs ;
  double HSMHV2_lnsti ;
  double HSMHV2_lwsti ;
  double HSMHV2_lscsti1 ;
  double HSMHV2_lscsti2 ;
  double HSMHV2_lvthsti ;
  double HSMHV2_lmuesti1 ;
  double HSMHV2_lmuesti2 ;
  double HSMHV2_lmuesti3 ;
  double HSMHV2_lnsubpsti1 ;
  double HSMHV2_lnsubpsti2 ;
  double HSMHV2_lnsubpsti3 ;
  double HSMHV2_lcgso ;
  double HSMHV2_lcgdo ;
  double HSMHV2_ljs0 ;
  double HSMHV2_ljs0sw ;
  double HSMHV2_lnj ;
  double HSMHV2_lcisbk ;
  double HSMHV2_lclm1 ;
  double HSMHV2_lclm2 ;
  double HSMHV2_lclm3 ;
  double HSMHV2_lwfc ;
  double HSMHV2_lgidl1 ;
  double HSMHV2_lgidl2 ;
  double HSMHV2_lgleak1 ;
  double HSMHV2_lgleak2 ;
  double HSMHV2_lgleak3 ;
  double HSMHV2_lgleak6 ;
  double HSMHV2_lglksd1 ;
  double HSMHV2_lglksd2 ;
  double HSMHV2_lglkb1 ;
  double HSMHV2_lglkb2 ;
  double HSMHV2_lnftrp ;
  double HSMHV2_lnfalp ;
  double HSMHV2_lvdiffj ;
  double HSMHV2_libpc1 ;
  double HSMHV2_libpc2 ;
  double HSMHV2_lcgbo ;
  double HSMHV2_lcvdsover ;
  double HSMHV2_lfalph ;
  double HSMHV2_lnpext ;
  double HSMHV2_lpowrat ;
  double HSMHV2_lrd ;
  double HSMHV2_lrd22 ;
  double HSMHV2_lrd23 ;
  double HSMHV2_lrd24 ;
  double HSMHV2_lrdict1 ;
  double HSMHV2_lrdov13 ;
  double HSMHV2_lrdslp1 ;
  double HSMHV2_lrdvb ;
  double HSMHV2_lrdvd ;
  double HSMHV2_lrdvg11 ;
  double HSMHV2_lrs ;
  double HSMHV2_lrth0 ;
  double HSMHV2_lvover ;
  double HSMHV2_ljs0d;
  double HSMHV2_ljs0swd;
  double HSMHV2_lnjd;
  double HSMHV2_lcisbkd;
  double HSMHV2_lvdiffjd;
  double HSMHV2_ljs0s;
  double HSMHV2_ljs0sws;
  double HSMHV2_lnjs;
  double HSMHV2_lcisbks;
  double HSMHV2_lvdiffjs;

  /* Width dependence */
  double HSMHV2_wvmax ;
  double HSMHV2_wbgtmp1 ;
  double HSMHV2_wbgtmp2 ;
  double HSMHV2_weg0 ;
  double HSMHV2_wvfbover ;
  double HSMHV2_wnover ;
  double HSMHV2_wnovers ;
  double HSMHV2_wwl2 ;
  double HSMHV2_wvfbc ;
  double HSMHV2_wnsubc ;
  double HSMHV2_wnsubp ;
  double HSMHV2_wscp1 ;
  double HSMHV2_wscp2 ;
  double HSMHV2_wscp3 ;
  double HSMHV2_wsc1 ;
  double HSMHV2_wsc2 ;
  double HSMHV2_wsc3 ;
  double HSMHV2_wpgd1 ;
//double HSMHV2_wpgd3 ;
  double HSMHV2_wndep ;
  double HSMHV2_wninv ;
  double HSMHV2_wmuecb0 ;
  double HSMHV2_wmuecb1 ;
  double HSMHV2_wmueph1 ;
  double HSMHV2_wvtmp ;
  double HSMHV2_wwvth0 ;
  double HSMHV2_wmuesr1 ;
  double HSMHV2_wmuetmp ;
  double HSMHV2_wsub1 ;
  double HSMHV2_wsub2 ;
  double HSMHV2_wsvds ;
  double HSMHV2_wsvbs ;
  double HSMHV2_wsvgs ;
  double HSMHV2_wfn1 ;
  double HSMHV2_wfn2 ;
  double HSMHV2_wfn3 ;
  double HSMHV2_wfvbs ;
  double HSMHV2_wnsti ;
  double HSMHV2_wwsti ;
  double HSMHV2_wscsti1 ;
  double HSMHV2_wscsti2 ;
  double HSMHV2_wvthsti ;
  double HSMHV2_wmuesti1 ;
  double HSMHV2_wmuesti2 ;
  double HSMHV2_wmuesti3 ;
  double HSMHV2_wnsubpsti1 ;
  double HSMHV2_wnsubpsti2 ;
  double HSMHV2_wnsubpsti3 ;
  double HSMHV2_wcgso ;
  double HSMHV2_wcgdo ;
  double HSMHV2_wjs0 ;
  double HSMHV2_wjs0sw ;
  double HSMHV2_wnj ;
  double HSMHV2_wcisbk ;
  double HSMHV2_wclm1 ;
  double HSMHV2_wclm2 ;
  double HSMHV2_wclm3 ;
  double HSMHV2_wwfc ;
  double HSMHV2_wgidl1 ;
  double HSMHV2_wgidl2 ;
  double HSMHV2_wgleak1 ;
  double HSMHV2_wgleak2 ;
  double HSMHV2_wgleak3 ;
  double HSMHV2_wgleak6 ;
  double HSMHV2_wglksd1 ;
  double HSMHV2_wglksd2 ;
  double HSMHV2_wglkb1 ;
  double HSMHV2_wglkb2 ;
  double HSMHV2_wnftrp ;
  double HSMHV2_wnfalp ;
  double HSMHV2_wvdiffj ;
  double HSMHV2_wibpc1 ;
  double HSMHV2_wibpc2 ;
  double HSMHV2_wcgbo ;
  double HSMHV2_wcvdsover ;
  double HSMHV2_wfalph ;
  double HSMHV2_wnpext ;
  double HSMHV2_wpowrat ;
  double HSMHV2_wrd ;
  double HSMHV2_wrd22 ;
  double HSMHV2_wrd23 ;
  double HSMHV2_wrd24 ;
  double HSMHV2_wrdict1 ;
  double HSMHV2_wrdov13 ;
  double HSMHV2_wrdslp1 ;
  double HSMHV2_wrdvb ;
  double HSMHV2_wrdvd ;
  double HSMHV2_wrdvg11 ;
  double HSMHV2_wrs ;
  double HSMHV2_wrth0 ;
  double HSMHV2_wvover ;
  double HSMHV2_wjs0d;
  double HSMHV2_wjs0swd;
  double HSMHV2_wnjd;
  double HSMHV2_wcisbkd;
  double HSMHV2_wvdiffjd;
  double HSMHV2_wjs0s;
  double HSMHV2_wjs0sws;
  double HSMHV2_wnjs;
  double HSMHV2_wcisbks;
  double HSMHV2_wvdiffjs;

  /* Cross-term dependence */
  double HSMHV2_pvmax ;
  double HSMHV2_pbgtmp1 ;
  double HSMHV2_pbgtmp2 ;
  double HSMHV2_peg0 ;
  double HSMHV2_pvfbover ;
  double HSMHV2_pnover ;
  double HSMHV2_pnovers ;
  double HSMHV2_pwl2 ;
  double HSMHV2_pvfbc ;
  double HSMHV2_pnsubc ;
  double HSMHV2_pnsubp ;
  double HSMHV2_pscp1 ;
  double HSMHV2_pscp2 ;
  double HSMHV2_pscp3 ;
  double HSMHV2_psc1 ;
  double HSMHV2_psc2 ;
  double HSMHV2_psc3 ;
  double HSMHV2_ppgd1 ;
//double HSMHV2_ppgd3 ;
  double HSMHV2_pndep ;
  double HSMHV2_pninv ;
  double HSMHV2_pmuecb0 ;
  double HSMHV2_pmuecb1 ;
  double HSMHV2_pmueph1 ;
  double HSMHV2_pvtmp ;
  double HSMHV2_pwvth0 ;
  double HSMHV2_pmuesr1 ;
  double HSMHV2_pmuetmp ;
  double HSMHV2_psub1 ;
  double HSMHV2_psub2 ;
  double HSMHV2_psvds ;
  double HSMHV2_psvbs ;
  double HSMHV2_psvgs ;
  double HSMHV2_pfn1 ;
  double HSMHV2_pfn2 ;
  double HSMHV2_pfn3 ;
  double HSMHV2_pfvbs ;
  double HSMHV2_pnsti ;
  double HSMHV2_pwsti ;
  double HSMHV2_pscsti1 ;
  double HSMHV2_pscsti2 ;
  double HSMHV2_pvthsti ;
  double HSMHV2_pmuesti1 ;
  double HSMHV2_pmuesti2 ;
  double HSMHV2_pmuesti3 ;
  double HSMHV2_pnsubpsti1 ;
  double HSMHV2_pnsubpsti2 ;
  double HSMHV2_pnsubpsti3 ;
  double HSMHV2_pcgso ;
  double HSMHV2_pcgdo ;
  double HSMHV2_pjs0 ;
  double HSMHV2_pjs0sw ;
  double HSMHV2_pnj ;
  double HSMHV2_pcisbk ;
  double HSMHV2_pclm1 ;
  double HSMHV2_pclm2 ;
  double HSMHV2_pclm3 ;
  double HSMHV2_pwfc ;
  double HSMHV2_pgidl1 ;
  double HSMHV2_pgidl2 ;
  double HSMHV2_pgleak1 ;
  double HSMHV2_pgleak2 ;
  double HSMHV2_pgleak3 ;
  double HSMHV2_pgleak6 ;
  double HSMHV2_pglksd1 ;
  double HSMHV2_pglksd2 ;
  double HSMHV2_pglkb1 ;
  double HSMHV2_pglkb2 ;
  double HSMHV2_pnftrp ;
  double HSMHV2_pnfalp ;
  double HSMHV2_pvdiffj ;
  double HSMHV2_pibpc1 ;
  double HSMHV2_pibpc2 ;
  double HSMHV2_pcgbo ;
  double HSMHV2_pcvdsover ;
  double HSMHV2_pfalph ;
  double HSMHV2_pnpext ;
  double HSMHV2_ppowrat ;
  double HSMHV2_prd ;
  double HSMHV2_prd22 ;
  double HSMHV2_prd23 ;
  double HSMHV2_prd24 ;
  double HSMHV2_prdict1 ;
  double HSMHV2_prdov13 ;
  double HSMHV2_prdslp1 ;
  double HSMHV2_prdvb ;
  double HSMHV2_prdvd ;
  double HSMHV2_prdvg11 ;
  double HSMHV2_prs ;
  double HSMHV2_prth0 ;
  double HSMHV2_pvover ;
  double HSMHV2_pjs0d;
  double HSMHV2_pjs0swd;
  double HSMHV2_pnjd;
  double HSMHV2_pcisbkd;
  double HSMHV2_pvdiffjd;
  double HSMHV2_pjs0s;
  double HSMHV2_pjs0sws;
  double HSMHV2_pnjs;
  double HSMHV2_pcisbks;
  double HSMHV2_pvdiffjs;

  /* internal variables */
  double HSMHV2_exptempd ;
  double HSMHV2_exptemps ;
  double HSMHV2_jd_nvtm_invd ;
  double HSMHV2_jd_nvtm_invs ;
  double HSMHV2_vcrit ;
  int HSMHV2_flg_qme ;
  double HSMHV2_qme12 ;
  double HSMHV2_ktnom ;

  int HSMHV2_subversion ;
  int HSMHV2_revision ;

  double HSMHV2vgsMax;
  double HSMHV2vgdMax;
  double HSMHV2vgbMax;
  double HSMHV2vdsMax;
  double HSMHV2vbsMax;
  double HSMHV2vbdMax;
  double HSMHV2vgsrMax;
  double HSMHV2vgdrMax;
  double HSMHV2vgbrMax;
  double HSMHV2vbsrMax;
  double HSMHV2vbdrMax;

  HSMHV2modelMKSParam modelMKS ; /* unit-converted parameters */


  /* flag for model */
  unsigned HSMHV2_type_Given  :1;
  unsigned HSMHV2_level_Given  :1;
  unsigned HSMHV2_info_Given  :1;
  unsigned HSMHV2_noise_Given :1;
  unsigned HSMHV2_version_Given :1;
  unsigned HSMHV2_show_Given :1;
  unsigned HSMHV2_corsrd_Given  :1;
  unsigned HSMHV2_corg_Given    :1;
  unsigned HSMHV2_coiprv_Given  :1;
  unsigned HSMHV2_copprv_Given  :1;
  unsigned HSMHV2_coadov_Given  :1;
  unsigned HSMHV2_coisub_Given  :1;
  unsigned HSMHV2_coiigs_Given  :1;
  unsigned HSMHV2_cogidl_Given  :1;
  unsigned HSMHV2_coovlp_Given  :1;
  unsigned HSMHV2_coovlps_Given  :1;
  unsigned HSMHV2_coflick_Given  :1;
  unsigned HSMHV2_coisti_Given  :1;
  unsigned HSMHV2_conqs_Given  :1;
  unsigned HSMHV2_corbnet_Given  :1;
  unsigned HSMHV2_cothrml_Given  :1;
  unsigned HSMHV2_coign_Given  :1;      /* Induced gate noise */
  unsigned HSMHV2_codfm_Given  :1;      /* DFM */
  unsigned HSMHV2_coqovsm_Given  :1;
  unsigned HSMHV2_coselfheat_Given  :1; /* Self-heating model */
  unsigned HSMHV2_cosubnode_Given  :1;  /* switch tempNode to subNode */
  unsigned HSMHV2_cosym_Given  :1;      /* Symmetry model for HV */
  unsigned HSMHV2_cotemp_Given  :1;
  unsigned HSMHV2_coldrift_Given  :1;
  unsigned HSMHV2_cordrift_Given  :1;
  unsigned HSMHV2_coerrrep_Given  :1;
  unsigned HSMHV2_codep_Given  :1;
  unsigned HSMHV2_coddlt_Given  :1;
  unsigned HSMHV2_kappa_Given :1;  
  unsigned HSMHV2_vdiffj_Given :1; 
  unsigned HSMHV2_vmax_Given  :1;
  unsigned HSMHV2_vmaxt1_Given  :1;
  unsigned HSMHV2_vmaxt2_Given  :1;
  unsigned HSMHV2_bgtmp1_Given  :1;
  unsigned HSMHV2_bgtmp2_Given  :1;
  unsigned HSMHV2_eg0_Given  :1;
  unsigned HSMHV2_tox_Given  :1;
  unsigned HSMHV2_xld_Given  :1;
  unsigned HSMHV2_xldld_Given  :1;
  unsigned HSMHV2_xwdld_Given  :1;
  unsigned HSMHV2_lover_Given  :1;
  unsigned HSMHV2_lovers_Given  :1;
  unsigned HSMHV2_rdov11_Given  :1;
  unsigned HSMHV2_rdov12_Given  :1;
  unsigned HSMHV2_rdov13_Given  :1;
  unsigned HSMHV2_rdslp1_Given  :1;
  unsigned HSMHV2_rdict1_Given  :1;
  unsigned HSMHV2_rdslp2_Given  :1;
  unsigned HSMHV2_rdict2_Given  :1;
  unsigned HSMHV2_loverld_Given  :1;
  unsigned HSMHV2_ldrift1_Given  :1;
  unsigned HSMHV2_ldrift2_Given  :1;
  unsigned HSMHV2_ldrift1s_Given  :1;
  unsigned HSMHV2_ldrift2s_Given  :1;
  unsigned HSMHV2_subld1_Given  :1;
  unsigned HSMHV2_subld1l_Given  :1;
  unsigned HSMHV2_subld1lp_Given  :1;
  unsigned HSMHV2_subld2_Given  :1;
  unsigned HSMHV2_xpdv_Given  :1;
  unsigned HSMHV2_xpvdth_Given  :1;
  unsigned HSMHV2_xpvdthg_Given  :1;
  unsigned HSMHV2_ddltmax_Given  :1;
  unsigned HSMHV2_ddltslp_Given  :1;
  unsigned HSMHV2_ddltict_Given  :1;
  unsigned HSMHV2_vfbover_Given  :1;
  unsigned HSMHV2_nover_Given  :1;
  unsigned HSMHV2_novers_Given  :1;
  unsigned HSMHV2_xwd_Given  :1; 
  unsigned HSMHV2_xwdc_Given  :1; 
  unsigned HSMHV2_xl_Given  :1;
  unsigned HSMHV2_xw_Given  :1;
  unsigned HSMHV2_saref_Given  :1;
  unsigned HSMHV2_sbref_Given  :1;
  unsigned HSMHV2_ll_Given  :1;
  unsigned HSMHV2_lld_Given  :1;
  unsigned HSMHV2_lln_Given  :1;
  unsigned HSMHV2_wl_Given  :1;
  unsigned HSMHV2_wl1_Given  :1;
  unsigned HSMHV2_wl1p_Given  :1;
  unsigned HSMHV2_wl2_Given  :1;
  unsigned HSMHV2_wl2p_Given  :1;
  unsigned HSMHV2_wld_Given  :1;
  unsigned HSMHV2_wln_Given  :1; 
  unsigned HSMHV2_xqy_Given  :1;   
  unsigned HSMHV2_xqy1_Given  :1;   
  unsigned HSMHV2_xqy2_Given  :1;   
  unsigned HSMHV2_rs_Given  :1;
  unsigned HSMHV2_rd_Given  :1;
  unsigned HSMHV2_rsh_Given  :1;
  unsigned HSMHV2_rshg_Given  :1;
/*   unsigned HSMHV2_ngcon_Given  :1; */
/*   unsigned HSMHV2_xgw_Given  :1; */
/*   unsigned HSMHV2_xgl_Given  :1; */
/*   unsigned HSMHV2_nf_Given  :1; */
  unsigned HSMHV2_vfbc_Given  :1;
  unsigned HSMHV2_vbi_Given  :1;
  unsigned HSMHV2_nsubc_Given  :1;
  unsigned HSMHV2_parl2_Given  :1;
  unsigned HSMHV2_lp_Given  :1;
  unsigned HSMHV2_nsubp_Given  :1;
  unsigned HSMHV2_ndepm_Given  :1;
  unsigned HSMHV2_tndep_Given  :1;
  unsigned HSMHV2_depmue0_Given  :1;
  unsigned HSMHV2_depmue1_Given  :1;
  unsigned HSMHV2_depmueback0_Given  :1;
  unsigned HSMHV2_depmueback1_Given  :1;
  unsigned HSMHV2_depleak_Given  :1;
  unsigned HSMHV2_depeta_Given  :1;
  unsigned HSMHV2_depvmax_Given  :1;
  unsigned HSMHV2_depvdsef1_Given  :1;
  unsigned HSMHV2_depvdsef2_Given  :1;
  unsigned HSMHV2_depmueph0_Given  :1;
  unsigned HSMHV2_depmueph1_Given  :1;
  unsigned HSMHV2_depbb_Given  :1;
  unsigned HSMHV2_depvtmp_Given  :1;
  unsigned HSMHV2_depmuetmp_Given  :1;

  unsigned HSMHV2_isbreak_Given  :1;
  unsigned HSMHV2_rwell_Given  :1;


  unsigned HSMHV2_nsubp0_Given  :1;
  unsigned HSMHV2_nsubwp_Given  :1;
  unsigned HSMHV2_scp1_Given  :1;
  unsigned HSMHV2_scp2_Given  :1;
  unsigned HSMHV2_scp3_Given  :1;
  unsigned HSMHV2_sc1_Given  :1;
  unsigned HSMHV2_sc2_Given  :1;
  unsigned HSMHV2_sc3_Given  :1;
  unsigned HSMHV2_sc4_Given  :1;
  unsigned HSMHV2_pgd1_Given  :1;
  unsigned HSMHV2_pgd2_Given  :1;
  unsigned HSMHV2_pgd4_Given  :1;
  unsigned HSMHV2_ndep_Given  :1;
  unsigned HSMHV2_ndepl_Given  :1;
  unsigned HSMHV2_ndeplp_Given  :1;
  unsigned HSMHV2_ninv_Given  :1;
  unsigned HSMHV2_muecb0_Given  :1;
  unsigned HSMHV2_muecb1_Given  :1;
  unsigned HSMHV2_mueph1_Given  :1;
  unsigned HSMHV2_mueph0_Given  :1;
  unsigned HSMHV2_muephw_Given  :1;
  unsigned HSMHV2_muepwp_Given  :1;
  unsigned HSMHV2_muephl_Given  :1;
  unsigned HSMHV2_mueplp_Given  :1;
  unsigned HSMHV2_muephs_Given  :1;
  unsigned HSMHV2_muepsp_Given  :1;
  unsigned HSMHV2_vtmp_Given  :1;
  unsigned HSMHV2_wvth0_Given  :1;
  unsigned HSMHV2_muesr1_Given  :1;
  unsigned HSMHV2_muesr0_Given  :1;
  unsigned HSMHV2_muesrl_Given  :1;
  unsigned HSMHV2_mueslp_Given  :1;
  unsigned HSMHV2_muesrw_Given  :1;
  unsigned HSMHV2_mueswp_Given  :1;
  unsigned HSMHV2_bb_Given  :1;
  unsigned HSMHV2_sub1_Given  :1;
  unsigned HSMHV2_sub2_Given  :1;
  unsigned HSMHV2_svgs_Given  :1; 
  unsigned HSMHV2_svbs_Given  :1; 
  unsigned HSMHV2_svbsl_Given  :1; 
  unsigned HSMHV2_svds_Given  :1; 
  unsigned HSMHV2_slg_Given  :1; 
  unsigned HSMHV2_sub1l_Given  :1; 
  unsigned HSMHV2_sub2l_Given  :1;  
  unsigned HSMHV2_fn1_Given  :1;   
  unsigned HSMHV2_fn2_Given  :1;  
  unsigned HSMHV2_fn3_Given  :1;
  unsigned HSMHV2_fvbs_Given  :1;
  unsigned HSMHV2_svgsl_Given  :1;
  unsigned HSMHV2_svgslp_Given  :1;
  unsigned HSMHV2_svgswp_Given  :1;
  unsigned HSMHV2_svgsw_Given  :1;
  unsigned HSMHV2_svbslp_Given  :1;
  unsigned HSMHV2_slgl_Given  :1;
  unsigned HSMHV2_slglp_Given  :1;
  unsigned HSMHV2_sub1lp_Given  :1;
  unsigned HSMHV2_nsti_Given  :1;  
  unsigned HSMHV2_wsti_Given  :1;
  unsigned HSMHV2_wstil_Given  :1;
  unsigned HSMHV2_wstilp_Given  :1;
  unsigned HSMHV2_wstiw_Given  :1;
  unsigned HSMHV2_wstiwp_Given  :1;
  unsigned HSMHV2_scsti1_Given  :1;
  unsigned HSMHV2_scsti2_Given  :1;
  unsigned HSMHV2_vthsti_Given  :1;
  unsigned HSMHV2_vdsti_Given  :1;
  unsigned HSMHV2_muesti1_Given  :1;
  unsigned HSMHV2_muesti2_Given  :1;
  unsigned HSMHV2_muesti3_Given  :1;
  unsigned HSMHV2_nsubpsti1_Given  :1;
  unsigned HSMHV2_nsubpsti2_Given  :1;
  unsigned HSMHV2_nsubpsti3_Given  :1;
  unsigned HSMHV2_lpext_Given  :1;
  unsigned HSMHV2_npext_Given  :1;
  unsigned HSMHV2_scp22_Given  :1;
  unsigned HSMHV2_scp21_Given  :1;
  unsigned HSMHV2_bs1_Given  :1;
  unsigned HSMHV2_bs2_Given  :1;
  unsigned HSMHV2_cgso_Given  :1;
  unsigned HSMHV2_cgdo_Given  :1;
  unsigned HSMHV2_cgbo_Given  :1;
  unsigned HSMHV2_tpoly_Given  :1;
  unsigned HSMHV2_js0_Given  :1;
  unsigned HSMHV2_js0sw_Given  :1;
  unsigned HSMHV2_nj_Given  :1;
  unsigned HSMHV2_njsw_Given  :1;  
  unsigned HSMHV2_xti_Given  :1;
  unsigned HSMHV2_cj_Given  :1;
  unsigned HSMHV2_cjsw_Given  :1;
  unsigned HSMHV2_cjswg_Given  :1;
  unsigned HSMHV2_mj_Given  :1;
  unsigned HSMHV2_mjsw_Given  :1;
  unsigned HSMHV2_mjswg_Given  :1;
  unsigned HSMHV2_xti2_Given  :1;
  unsigned HSMHV2_cisb_Given  :1;
  unsigned HSMHV2_cvb_Given  :1;
  unsigned HSMHV2_ctemp_Given  :1;
  unsigned HSMHV2_cisbk_Given  :1;
  unsigned HSMHV2_cvbk_Given  :1;
  unsigned HSMHV2_divx_Given  :1;
  unsigned HSMHV2_pb_Given  :1;
  unsigned HSMHV2_pbsw_Given  :1;
  unsigned HSMHV2_pbswg_Given  :1;
  unsigned HSMHV2_clm1_Given  :1;
  unsigned HSMHV2_clm2_Given  :1;
  unsigned HSMHV2_clm3_Given  :1;
  unsigned HSMHV2_clm5_Given  :1;
  unsigned HSMHV2_clm6_Given  :1;
  unsigned HSMHV2_muetmp_Given  :1;
  unsigned HSMHV2_vover_Given  :1;
  unsigned HSMHV2_voverp_Given  :1;
  unsigned HSMHV2_vovers_Given  :1;
  unsigned HSMHV2_voversp_Given  :1;
  unsigned HSMHV2_wfc_Given  :1;
  unsigned HSMHV2_nsubcw_Given  :1;
  unsigned HSMHV2_nsubcwp_Given  :1;
  unsigned HSMHV2_qme1_Given  :1;
  unsigned HSMHV2_qme2_Given  :1;
  unsigned HSMHV2_qme3_Given  :1;
  unsigned HSMHV2_gidl1_Given  :1;
  unsigned HSMHV2_gidl2_Given  :1;
  unsigned HSMHV2_gidl3_Given  :1;
  unsigned HSMHV2_gidl4_Given  :1;
  unsigned HSMHV2_gidl5_Given  :1;
  unsigned HSMHV2_gleak1_Given  :1;
  unsigned HSMHV2_gleak2_Given  :1;
  unsigned HSMHV2_gleak3_Given  :1;
  unsigned HSMHV2_gleak4_Given  :1;
  unsigned HSMHV2_gleak5_Given  :1;
  unsigned HSMHV2_gleak6_Given  :1;
  unsigned HSMHV2_gleak7_Given  :1;
  unsigned HSMHV2_glpart1_Given  :1;
  unsigned HSMHV2_glksd1_Given  :1;
  unsigned HSMHV2_glksd2_Given  :1;
  unsigned HSMHV2_glksd3_Given  :1;
  unsigned HSMHV2_glkb1_Given  :1;
  unsigned HSMHV2_glkb2_Given  :1;
  unsigned HSMHV2_glkb3_Given  :1;
  unsigned HSMHV2_egig_Given  :1;
  unsigned HSMHV2_igtemp2_Given  :1;
  unsigned HSMHV2_igtemp3_Given  :1;
  unsigned HSMHV2_vzadd0_Given  :1;
  unsigned HSMHV2_pzadd0_Given  :1;
  unsigned HSMHV2_nftrp_Given  :1;
  unsigned HSMHV2_nfalp_Given  :1;
  unsigned HSMHV2_cit_Given  :1;
  unsigned HSMHV2_falph_Given  :1;
  unsigned HSMHV2_dly1_Given :1;
  unsigned HSMHV2_dly2_Given :1;
  unsigned HSMHV2_dly3_Given :1;
  unsigned HSMHV2_dlyov_Given :1;
  unsigned HSMHV2_tnom_Given :1;
  unsigned HSMHV2_ovslp_Given :1;
  unsigned HSMHV2_ovmag_Given :1;
  unsigned HSMHV2_gbmin_Given :1;
  unsigned HSMHV2_rbpb_Given :1;
  unsigned HSMHV2_rbpd_Given :1;
  unsigned HSMHV2_rbps_Given :1;
  unsigned HSMHV2_rbdb_Given :1;
  unsigned HSMHV2_rbsb_Given :1;
  unsigned HSMHV2_ibpc1_Given :1;
  unsigned HSMHV2_ibpc1l_Given :1;
  unsigned HSMHV2_ibpc1lp_Given :1;
  unsigned HSMHV2_ibpc2_Given :1;
  unsigned HSMHV2_mphdfm_Given :1;

  unsigned HSMHV2_ptl_Given :1;
  unsigned HSMHV2_ptp_Given :1;
  unsigned HSMHV2_pt2_Given :1;
  unsigned HSMHV2_ptlp_Given :1;
  unsigned HSMHV2_gdl_Given :1;
  unsigned HSMHV2_gdlp_Given :1;

  unsigned HSMHV2_gdld_Given :1;
  unsigned HSMHV2_pt4_Given :1;
  unsigned HSMHV2_pt4p_Given :1;
  unsigned HSMHV2_rdvg11_Given  :1;
  unsigned HSMHV2_rdvg12_Given  :1;
  unsigned HSMHV2_ldrift_Given  :1;
  unsigned HSMHV2_rd20_Given  :1;
  unsigned HSMHV2_rd21_Given  :1;
  unsigned HSMHV2_rd22_Given  :1;
  unsigned HSMHV2_rd22d_Given :1;
  unsigned HSMHV2_rd23_Given  :1;
  unsigned HSMHV2_rd24_Given  :1;
  unsigned HSMHV2_rd25_Given  :1;
  unsigned HSMHV2_rdvdl_Given  :1;
  unsigned HSMHV2_rdvdlp_Given  :1;
  unsigned HSMHV2_rdvds_Given  :1;
  unsigned HSMHV2_rdvdsp_Given  :1;
  unsigned HSMHV2_rd23l_Given  :1;
  unsigned HSMHV2_rd23lp_Given  :1;
  unsigned HSMHV2_rd23s_Given  :1;
  unsigned HSMHV2_rd23sp_Given  :1;
  unsigned HSMHV2_rds_Given  :1;
  unsigned HSMHV2_rdsp_Given  :1;
  unsigned HSMHV2_vbsmin_Given  :1;
  unsigned HSMHV2_ninvd_Given  :1;
  unsigned HSMHV2_ninvdw_Given  :1;
  unsigned HSMHV2_ninvdwp_Given  :1;
  unsigned HSMHV2_ninvdt1_Given  :1;
  unsigned HSMHV2_ninvdt2_Given  :1;
  unsigned HSMHV2_rdvb_Given  :1;
  unsigned HSMHV2_rth0nf_Given  :1;

  unsigned HSMHV2_rthtemp1_Given  :1;
  unsigned HSMHV2_rthtemp2_Given  :1;
  unsigned HSMHV2_prattemp1_Given  :1;
  unsigned HSMHV2_prattemp2_Given  :1;


  unsigned HSMHV2_rth0_Given :1;
  unsigned HSMHV2_cth0_Given :1;
  unsigned HSMHV2_powrat_Given :1;


  unsigned HSMHV2_tcjbd_Given :1;
  unsigned HSMHV2_tcjbs_Given :1;
  unsigned HSMHV2_tcjbdsw_Given :1;
  unsigned HSMHV2_tcjbssw_Given :1;
  unsigned HSMHV2_tcjbdswg_Given :1;
  unsigned HSMHV2_tcjbsswg_Given :1;


/*   unsigned HSMHV2_wth0_Given :1; */
  unsigned HSMHV2_qdftvd_Given  :1;
  unsigned HSMHV2_rdvd_Given  :1;
  unsigned HSMHV2_rdtemp1_Given :1;
  unsigned HSMHV2_rdtemp2_Given :1;
  unsigned HSMHV2_rth0r_Given :1;
  unsigned HSMHV2_rdvdtemp1_Given :1;
  unsigned HSMHV2_rdvdtemp2_Given :1;
  unsigned HSMHV2_rth0w_Given :1;
  unsigned HSMHV2_rth0wp_Given :1;

  unsigned HSMHV2_cvdsover_Given :1;

  /* substrate effect */
  unsigned HSMHV2_rdvsub_Given  :1;  /* substrate effect */
  unsigned HSMHV2_rdvdsub_Given  :1; /* substrate effect */
  unsigned HSMHV2_ddrift_Given  :1;  /* substrate effect */
  unsigned HSMHV2_vbisub_Given  :1;  /* substrate effect */
  unsigned HSMHV2_nsubsub_Given  :1; /* substrate effect */

  unsigned HSMHV2_rdrmue_Given  :1;
  unsigned HSMHV2_rdrvmax_Given  :1;
  unsigned HSMHV2_rdrmuetmp_Given  :1;
  unsigned HSMHV2_myu0_leak_Given  :1;
  unsigned HSMHV2_myu0_leaktmp_Given  :1;
  unsigned HSMHV2_myu0_vmax_Given  :1;
  unsigned HSMHV2_myu0_cotemp1_Given  :1;
  unsigned HSMHV2_myu0_cotemp2_Given  :1;

  unsigned HSMHV2_rdrvtmp_Given  :1;
  unsigned HSMHV2_rdrdjunc_Given  :1;
  unsigned HSMHV2_rdrcx_Given  :1;
  unsigned HSMHV2_rdrcar_Given  :1;
  unsigned HSMHV2_rdrdl1_Given  :1;
  unsigned HSMHV2_rdrdl2_Given  :1;
  unsigned HSMHV2_rdrvmaxw_Given  :1;
  unsigned HSMHV2_rdrvmaxwp_Given  :1;
  unsigned HSMHV2_rdrvmaxl_Given  :1;
  unsigned HSMHV2_rdrvmaxlp_Given  :1;
  unsigned HSMHV2_rdrmuel_Given  :1;
  unsigned HSMHV2_rdrmuelp_Given  :1;
  unsigned HSMHV2_qovadd_Given :1;
  unsigned HSMHV2_rdrqover_Given  :1;
  unsigned HSMHV2_js0d_Given :1;
  unsigned HSMHV2_js0swd_Given :1;
  unsigned HSMHV2_njd_Given :1;
  unsigned HSMHV2_njswd_Given :1;
  unsigned HSMHV2_xtid_Given :1;
  unsigned HSMHV2_cjd_Given :1;
  unsigned HSMHV2_cjswd_Given :1;
  unsigned HSMHV2_cjswgd_Given :1;
  unsigned HSMHV2_mjd_Given :1;
  unsigned HSMHV2_mjswd_Given :1;
  unsigned HSMHV2_mjswgd_Given :1;
  unsigned HSMHV2_pbd_Given :1;
  unsigned HSMHV2_pbswd_Given :1;
  unsigned HSMHV2_pbswgd_Given :1;
  unsigned HSMHV2_xti2d_Given :1;
  unsigned HSMHV2_cisbd_Given :1;
  unsigned HSMHV2_cvbd_Given :1;
  unsigned HSMHV2_ctempd_Given :1;
  unsigned HSMHV2_cisbkd_Given :1;
  unsigned HSMHV2_divxd_Given :1;
  unsigned HSMHV2_vdiffjd_Given :1;
  unsigned HSMHV2_js0s_Given :1;
  unsigned HSMHV2_js0sws_Given :1;
  unsigned HSMHV2_njs_Given :1;
  unsigned HSMHV2_njsws_Given :1;
  unsigned HSMHV2_xtis_Given :1;
  unsigned HSMHV2_cjs_Given :1;
  unsigned HSMHV2_cjsws_Given :1;
  unsigned HSMHV2_cjswgs_Given :1;
  unsigned HSMHV2_mjs_Given :1;
  unsigned HSMHV2_mjsws_Given :1;
  unsigned HSMHV2_mjswgs_Given :1;
  unsigned HSMHV2_pbs_Given :1;
  unsigned HSMHV2_pbsws_Given :1;
  unsigned HSMHV2_pbswgs_Given :1;
  unsigned HSMHV2_xti2s_Given :1;
  unsigned HSMHV2_cisbs_Given :1;
  unsigned HSMHV2_cvbs_Given :1;
  unsigned HSMHV2_ctemps_Given :1;
  unsigned HSMHV2_cisbks_Given :1;
  unsigned HSMHV2_divxs_Given :1;
  unsigned HSMHV2_vdiffjs_Given :1;
  unsigned HSMHV2_shemax_Given :1;
  unsigned HSMHV2_vgsmin_Given :1;
  unsigned HSMHV2_gdsleak_Given :1;
  unsigned HSMHV2_rdrbb_Given :1;
  unsigned HSMHV2_rdrbbtmp_Given :1;


  /* binning parameters */
  unsigned HSMHV2_lmin_Given :1;
  unsigned HSMHV2_lmax_Given :1;
  unsigned HSMHV2_wmin_Given :1;
  unsigned HSMHV2_wmax_Given :1;
  unsigned HSMHV2_lbinn_Given :1;
  unsigned HSMHV2_wbinn_Given :1;

  /* Length dependence */
  unsigned HSMHV2_lvmax_Given :1;
  unsigned HSMHV2_lbgtmp1_Given :1;
  unsigned HSMHV2_lbgtmp2_Given :1;
  unsigned HSMHV2_leg0_Given :1;
  unsigned HSMHV2_lvfbover_Given :1;
  unsigned HSMHV2_lnover_Given :1;
  unsigned HSMHV2_lnovers_Given :1;
  unsigned HSMHV2_lwl2_Given :1;
  unsigned HSMHV2_lvfbc_Given :1;
  unsigned HSMHV2_lnsubc_Given :1;
  unsigned HSMHV2_lnsubp_Given :1;
  unsigned HSMHV2_lscp1_Given :1;
  unsigned HSMHV2_lscp2_Given :1;
  unsigned HSMHV2_lscp3_Given :1;
  unsigned HSMHV2_lsc1_Given :1;
  unsigned HSMHV2_lsc2_Given :1;
  unsigned HSMHV2_lsc3_Given :1;
  unsigned HSMHV2_lpgd1_Given :1;
  unsigned HSMHV2_lndep_Given :1;
  unsigned HSMHV2_lninv_Given :1;
  unsigned HSMHV2_lmuecb0_Given :1;
  unsigned HSMHV2_lmuecb1_Given :1;
  unsigned HSMHV2_lmueph1_Given :1;
  unsigned HSMHV2_lvtmp_Given :1;
  unsigned HSMHV2_lwvth0_Given :1;
  unsigned HSMHV2_lmuesr1_Given :1;
  unsigned HSMHV2_lmuetmp_Given :1;
  unsigned HSMHV2_lsub1_Given :1;
  unsigned HSMHV2_lsub2_Given :1;
  unsigned HSMHV2_lsvds_Given :1;
  unsigned HSMHV2_lsvbs_Given :1;
  unsigned HSMHV2_lsvgs_Given :1;
  unsigned HSMHV2_lfn1_Given :1;
  unsigned HSMHV2_lfn2_Given :1;
  unsigned HSMHV2_lfn3_Given :1;
  unsigned HSMHV2_lfvbs_Given :1;
  unsigned HSMHV2_lnsti_Given :1;
  unsigned HSMHV2_lwsti_Given :1;
  unsigned HSMHV2_lscsti1_Given :1;
  unsigned HSMHV2_lscsti2_Given :1;
  unsigned HSMHV2_lvthsti_Given :1;
  unsigned HSMHV2_lmuesti1_Given :1;
  unsigned HSMHV2_lmuesti2_Given :1;
  unsigned HSMHV2_lmuesti3_Given :1;
  unsigned HSMHV2_lnsubpsti1_Given :1;
  unsigned HSMHV2_lnsubpsti2_Given :1;
  unsigned HSMHV2_lnsubpsti3_Given :1;
  unsigned HSMHV2_lcgso_Given :1;
  unsigned HSMHV2_lcgdo_Given :1;
  unsigned HSMHV2_ljs0_Given :1;
  unsigned HSMHV2_ljs0sw_Given :1;
  unsigned HSMHV2_lnj_Given :1;
  unsigned HSMHV2_lcisbk_Given :1;
  unsigned HSMHV2_lclm1_Given :1;
  unsigned HSMHV2_lclm2_Given :1;
  unsigned HSMHV2_lclm3_Given :1;
  unsigned HSMHV2_lwfc_Given :1;
  unsigned HSMHV2_lgidl1_Given :1;
  unsigned HSMHV2_lgidl2_Given :1;
  unsigned HSMHV2_lgleak1_Given :1;
  unsigned HSMHV2_lgleak2_Given :1;
  unsigned HSMHV2_lgleak3_Given :1;
  unsigned HSMHV2_lgleak6_Given :1;
  unsigned HSMHV2_lglksd1_Given :1;
  unsigned HSMHV2_lglksd2_Given :1;
  unsigned HSMHV2_lglkb1_Given :1;
  unsigned HSMHV2_lglkb2_Given :1;
  unsigned HSMHV2_lnftrp_Given :1;
  unsigned HSMHV2_lnfalp_Given :1;
  unsigned HSMHV2_lvdiffj_Given :1;
  unsigned HSMHV2_libpc1_Given :1;
  unsigned HSMHV2_libpc2_Given :1;
  unsigned HSMHV2_lcgbo_Given :1;
  unsigned HSMHV2_lcvdsover_Given :1;
  unsigned HSMHV2_lfalph_Given :1;
  unsigned HSMHV2_lnpext_Given :1;
  unsigned HSMHV2_lpowrat_Given :1;
  unsigned HSMHV2_lrd_Given :1;
  unsigned HSMHV2_lrd22_Given :1;
  unsigned HSMHV2_lrd23_Given :1;
  unsigned HSMHV2_lrd24_Given :1;
  unsigned HSMHV2_lrdict1_Given :1;
  unsigned HSMHV2_lrdov13_Given :1;
  unsigned HSMHV2_lrdslp1_Given :1;
  unsigned HSMHV2_lrdvb_Given :1;
  unsigned HSMHV2_lrdvd_Given :1;
  unsigned HSMHV2_lrdvg11_Given :1;
  unsigned HSMHV2_lrs_Given :1;
  unsigned HSMHV2_lrth0_Given :1;
  unsigned HSMHV2_lvover_Given :1;
  unsigned HSMHV2_ljs0d_Given :1;
  unsigned HSMHV2_ljs0swd_Given :1;
  unsigned HSMHV2_lnjd_Given :1;
  unsigned HSMHV2_lcisbkd_Given :1;
  unsigned HSMHV2_lvdiffjd_Given :1;
  unsigned HSMHV2_ljs0s_Given :1;
  unsigned HSMHV2_ljs0sws_Given :1;
  unsigned HSMHV2_lnjs_Given :1;
  unsigned HSMHV2_lcisbks_Given :1;
  unsigned HSMHV2_lvdiffjs_Given :1;

  /* Width dependence */
  unsigned HSMHV2_wvmax_Given :1;
  unsigned HSMHV2_wbgtmp1_Given :1;
  unsigned HSMHV2_wbgtmp2_Given :1;
  unsigned HSMHV2_weg0_Given :1;
  unsigned HSMHV2_wvfbover_Given :1;
  unsigned HSMHV2_wnover_Given :1;
  unsigned HSMHV2_wnovers_Given :1;
  unsigned HSMHV2_wwl2_Given :1;
  unsigned HSMHV2_wvfbc_Given :1;
  unsigned HSMHV2_wnsubc_Given :1;
  unsigned HSMHV2_wnsubp_Given :1;
  unsigned HSMHV2_wscp1_Given :1;
  unsigned HSMHV2_wscp2_Given :1;
  unsigned HSMHV2_wscp3_Given :1;
  unsigned HSMHV2_wsc1_Given :1;
  unsigned HSMHV2_wsc2_Given :1;
  unsigned HSMHV2_wsc3_Given :1;
  unsigned HSMHV2_wpgd1_Given :1;
  unsigned HSMHV2_wndep_Given :1;
  unsigned HSMHV2_wninv_Given :1;
  unsigned HSMHV2_wmuecb0_Given :1;
  unsigned HSMHV2_wmuecb1_Given :1;
  unsigned HSMHV2_wmueph1_Given :1;
  unsigned HSMHV2_wvtmp_Given :1;
  unsigned HSMHV2_wwvth0_Given :1;
  unsigned HSMHV2_wmuesr1_Given :1;
  unsigned HSMHV2_wmuetmp_Given :1;
  unsigned HSMHV2_wsub1_Given :1;
  unsigned HSMHV2_wsub2_Given :1;
  unsigned HSMHV2_wsvds_Given :1;
  unsigned HSMHV2_wsvbs_Given :1;
  unsigned HSMHV2_wsvgs_Given :1;
  unsigned HSMHV2_wfn1_Given :1;
  unsigned HSMHV2_wfn2_Given :1;
  unsigned HSMHV2_wfn3_Given :1;
  unsigned HSMHV2_wfvbs_Given :1;
  unsigned HSMHV2_wnsti_Given :1;
  unsigned HSMHV2_wwsti_Given :1;
  unsigned HSMHV2_wscsti1_Given :1;
  unsigned HSMHV2_wscsti2_Given :1;
  unsigned HSMHV2_wvthsti_Given :1;
  unsigned HSMHV2_wmuesti1_Given :1;
  unsigned HSMHV2_wmuesti2_Given :1;
  unsigned HSMHV2_wmuesti3_Given :1;
  unsigned HSMHV2_wnsubpsti1_Given :1;
  unsigned HSMHV2_wnsubpsti2_Given :1;
  unsigned HSMHV2_wnsubpsti3_Given :1;
  unsigned HSMHV2_wcgso_Given :1;
  unsigned HSMHV2_wcgdo_Given :1;
  unsigned HSMHV2_wjs0_Given :1;
  unsigned HSMHV2_wjs0sw_Given :1;
  unsigned HSMHV2_wnj_Given :1;
  unsigned HSMHV2_wcisbk_Given :1;
  unsigned HSMHV2_wclm1_Given :1;
  unsigned HSMHV2_wclm2_Given :1;
  unsigned HSMHV2_wclm3_Given :1;
  unsigned HSMHV2_wwfc_Given :1;
  unsigned HSMHV2_wgidl1_Given :1;
  unsigned HSMHV2_wgidl2_Given :1;
  unsigned HSMHV2_wgleak1_Given :1;
  unsigned HSMHV2_wgleak2_Given :1;
  unsigned HSMHV2_wgleak3_Given :1;
  unsigned HSMHV2_wgleak6_Given :1;
  unsigned HSMHV2_wglksd1_Given :1;
  unsigned HSMHV2_wglksd2_Given :1;
  unsigned HSMHV2_wglkb1_Given :1;
  unsigned HSMHV2_wglkb2_Given :1;
  unsigned HSMHV2_wnftrp_Given :1;
  unsigned HSMHV2_wnfalp_Given :1;
  unsigned HSMHV2_wvdiffj_Given :1;
  unsigned HSMHV2_wibpc1_Given :1;
  unsigned HSMHV2_wibpc2_Given :1;
  unsigned HSMHV2_wcgbo_Given :1;
  unsigned HSMHV2_wcvdsover_Given :1;
  unsigned HSMHV2_wfalph_Given :1;
  unsigned HSMHV2_wnpext_Given :1;
  unsigned HSMHV2_wpowrat_Given :1;
  unsigned HSMHV2_wrd_Given :1;
  unsigned HSMHV2_wrd22_Given :1;
  unsigned HSMHV2_wrd23_Given :1;
  unsigned HSMHV2_wrd24_Given :1;
  unsigned HSMHV2_wrdict1_Given :1;
  unsigned HSMHV2_wrdov13_Given :1;
  unsigned HSMHV2_wrdslp1_Given :1;
  unsigned HSMHV2_wrdvb_Given :1;
  unsigned HSMHV2_wrdvd_Given :1;
  unsigned HSMHV2_wrdvg11_Given :1;
  unsigned HSMHV2_wrs_Given :1;
  unsigned HSMHV2_wrth0_Given :1;
  unsigned HSMHV2_wvover_Given :1;
  unsigned HSMHV2_wjs0d_Given :1;
  unsigned HSMHV2_wjs0swd_Given :1;
  unsigned HSMHV2_wnjd_Given :1;
  unsigned HSMHV2_wcisbkd_Given :1;
  unsigned HSMHV2_wvdiffjd_Given :1;
  unsigned HSMHV2_wjs0s_Given :1;
  unsigned HSMHV2_wjs0sws_Given :1;
  unsigned HSMHV2_wnjs_Given :1;
  unsigned HSMHV2_wcisbks_Given :1;
  unsigned HSMHV2_wvdiffjs_Given :1;

  /* Cross-term dependence */
  unsigned HSMHV2_pvmax_Given :1;
  unsigned HSMHV2_pbgtmp1_Given :1;
  unsigned HSMHV2_pbgtmp2_Given :1;
  unsigned HSMHV2_peg0_Given :1;
  unsigned HSMHV2_pvfbover_Given :1;
  unsigned HSMHV2_pnover_Given :1;
  unsigned HSMHV2_pnovers_Given :1;
  unsigned HSMHV2_pwl2_Given :1;
  unsigned HSMHV2_pvfbc_Given :1;
  unsigned HSMHV2_pnsubc_Given :1;
  unsigned HSMHV2_pnsubp_Given :1;
  unsigned HSMHV2_pscp1_Given :1;
  unsigned HSMHV2_pscp2_Given :1;
  unsigned HSMHV2_pscp3_Given :1;
  unsigned HSMHV2_psc1_Given :1;
  unsigned HSMHV2_psc2_Given :1;
  unsigned HSMHV2_psc3_Given :1;
  unsigned HSMHV2_ppgd1_Given :1;
  unsigned HSMHV2_pndep_Given :1;
  unsigned HSMHV2_pninv_Given :1;
  unsigned HSMHV2_pmuecb0_Given :1;
  unsigned HSMHV2_pmuecb1_Given :1;
  unsigned HSMHV2_pmueph1_Given :1;
  unsigned HSMHV2_pvtmp_Given :1;
  unsigned HSMHV2_pwvth0_Given :1;
  unsigned HSMHV2_pmuesr1_Given :1;
  unsigned HSMHV2_pmuetmp_Given :1;
  unsigned HSMHV2_psub1_Given :1;
  unsigned HSMHV2_psub2_Given :1;
  unsigned HSMHV2_psvds_Given :1;
  unsigned HSMHV2_psvbs_Given :1;
  unsigned HSMHV2_psvgs_Given :1;
  unsigned HSMHV2_pfn1_Given :1;
  unsigned HSMHV2_pfn2_Given :1;
  unsigned HSMHV2_pfn3_Given :1;
  unsigned HSMHV2_pfvbs_Given :1;
  unsigned HSMHV2_pnsti_Given :1;
  unsigned HSMHV2_pwsti_Given :1;
  unsigned HSMHV2_pscsti1_Given :1;
  unsigned HSMHV2_pscsti2_Given :1;
  unsigned HSMHV2_pvthsti_Given :1;
  unsigned HSMHV2_pmuesti1_Given :1;
  unsigned HSMHV2_pmuesti2_Given :1;
  unsigned HSMHV2_pmuesti3_Given :1;
  unsigned HSMHV2_pnsubpsti1_Given :1;
  unsigned HSMHV2_pnsubpsti2_Given :1;
  unsigned HSMHV2_pnsubpsti3_Given :1;
  unsigned HSMHV2_pcgso_Given :1;
  unsigned HSMHV2_pcgdo_Given :1;
  unsigned HSMHV2_pjs0_Given :1;
  unsigned HSMHV2_pjs0sw_Given :1;
  unsigned HSMHV2_pnj_Given :1;
  unsigned HSMHV2_pcisbk_Given :1;
  unsigned HSMHV2_pclm1_Given :1;
  unsigned HSMHV2_pclm2_Given :1;
  unsigned HSMHV2_pclm3_Given :1;
  unsigned HSMHV2_pwfc_Given :1;
  unsigned HSMHV2_pgidl1_Given :1;
  unsigned HSMHV2_pgidl2_Given :1;
  unsigned HSMHV2_pgleak1_Given :1;
  unsigned HSMHV2_pgleak2_Given :1;
  unsigned HSMHV2_pgleak3_Given :1;
  unsigned HSMHV2_pgleak6_Given :1;
  unsigned HSMHV2_pglksd1_Given :1;
  unsigned HSMHV2_pglksd2_Given :1;
  unsigned HSMHV2_pglkb1_Given :1;
  unsigned HSMHV2_pglkb2_Given :1;
  unsigned HSMHV2_pnftrp_Given :1;
  unsigned HSMHV2_pnfalp_Given :1;
  unsigned HSMHV2_pvdiffj_Given :1;
  unsigned HSMHV2_pibpc1_Given :1;
  unsigned HSMHV2_pibpc2_Given :1;
  unsigned HSMHV2_pcgbo_Given :1;
  unsigned HSMHV2_pcvdsover_Given :1;
  unsigned HSMHV2_pfalph_Given :1;
  unsigned HSMHV2_pnpext_Given :1;
  unsigned HSMHV2_ppowrat_Given :1;
  unsigned HSMHV2_prd_Given :1;
  unsigned HSMHV2_prd22_Given :1;
  unsigned HSMHV2_prd23_Given :1;
  unsigned HSMHV2_prd24_Given :1;
  unsigned HSMHV2_prdict1_Given :1;
  unsigned HSMHV2_prdov13_Given :1;
  unsigned HSMHV2_prdslp1_Given :1;
  unsigned HSMHV2_prdvb_Given :1;
  unsigned HSMHV2_prdvd_Given :1;
  unsigned HSMHV2_prdvg11_Given :1;
  unsigned HSMHV2_prs_Given :1;
  unsigned HSMHV2_prth0_Given :1;
  unsigned HSMHV2_pvover_Given :1;
  unsigned HSMHV2_pjs0d_Given :1;
  unsigned HSMHV2_pjs0swd_Given :1;
  unsigned HSMHV2_pnjd_Given :1;
  unsigned HSMHV2_pcisbkd_Given :1;
  unsigned HSMHV2_pvdiffjd_Given :1;
  unsigned HSMHV2_pjs0s_Given :1;
  unsigned HSMHV2_pjs0sws_Given :1;
  unsigned HSMHV2_pnjs_Given :1;
  unsigned HSMHV2_pcisbks_Given :1;
  unsigned HSMHV2_pvdiffjs_Given :1;

  unsigned HSMHV2vgsMaxGiven  :1;
  unsigned HSMHV2vgdMaxGiven  :1;
  unsigned HSMHV2vgbMaxGiven  :1;
  unsigned HSMHV2vdsMaxGiven  :1;
  unsigned HSMHV2vbsMaxGiven  :1;
  unsigned HSMHV2vbdMaxGiven  :1;
  unsigned HSMHV2vgsrMaxGiven  :1;
  unsigned HSMHV2vgdrMaxGiven  :1;
  unsigned HSMHV2vgbrMaxGiven  :1;
  unsigned HSMHV2vbsrMaxGiven  :1;
  unsigned HSMHV2vbdrMaxGiven  :1;

} HSMHV2model;

#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/

#define HSMHV2_BAD_PARAM -1

/* flags */
#define HSMHV2_MOD_NMOS     1
#define HSMHV2_MOD_PMOS     2
#define HSMHV2_MOD_LEVEL    3
#define HSMHV2_MOD_INFO     4
#define HSMHV2_MOD_NOISE    5
#define HSMHV2_MOD_VERSION  6
#define HSMHV2_MOD_SHOW     7
#define HSMHV2_MOD_CORSRD  11
#define HSMHV2_MOD_COIPRV  12
#define HSMHV2_MOD_COPPRV  13
#define HSMHV2_MOD_COADOV  17
#define HSMHV2_MOD_COISUB  21
#define HSMHV2_MOD_COIIGS    22
#define HSMHV2_MOD_COGIDL 23
#define HSMHV2_MOD_COOVLP  24
#define HSMHV2_MOD_COOVLPS  8
#define HSMHV2_MOD_COFLICK 25
#define HSMHV2_MOD_COISTI  26
#define HSMHV2_MOD_CONQS   29
#define HSMHV2_MOD_COTHRML 30
#define HSMHV2_MOD_COIGN   31    /* Induced gate noise */
#define HSMHV2_MOD_CORG    32
#define HSMHV2_MOD_CORBNET 33
#define HSMHV2_MOD_CODFM   36    /* DFM */
#define HSMHV2_MOD_COQOVSM 34
#define HSMHV2_MOD_COSELFHEAT 35 /* Self-heating model--SHE-- */
#define HSMHV2_MOD_COSUBNODE  48 
#define HSMHV2_MOD_COSYM   37    /* Symmery model for HV */
#define HSMHV2_MOD_COTEMP  38
#define HSMHV2_MOD_COLDRIFT 39
#define HSMHV2_MOD_CORDRIFT 40
#define HSMHV2_MOD_COERRREP 44
#define HSMHV2_MOD_CODEP    45
#define HSMHV2_MOD_CODDLT   46
/* device parameters */
#define HSMHV2_COSELFHEAT  49
#define HSMHV2_COSUBNODE   50
#define HSMHV2_L           51
#define HSMHV2_W           52
#define HSMHV2_AD          53
#define HSMHV2_AS          54
#define HSMHV2_PD          55
#define HSMHV2_PS          56
#define HSMHV2_NRD         57
#define HSMHV2_NRS         58
/* #define HSMHV2_TEMP        59	not used */
#define HSMHV2_DTEMP       60
#define HSMHV2_OFF         61
#define HSMHV2_IC_VBS      62
#define HSMHV2_IC_VDS      63
#define HSMHV2_IC_VGS      64
#define HSMHV2_IC          65
#define HSMHV2_CORBNET     66
#define HSMHV2_RBPB        67
#define HSMHV2_RBPD        68
#define HSMHV2_RBPS        69
#define HSMHV2_RBDB        70
#define HSMHV2_RBSB        71
#define HSMHV2_CORG        72
/* #define HSMHV2_RSHG        73 */
#define HSMHV2_NGCON       74
#define HSMHV2_XGW         75 
#define HSMHV2_XGL         76
#define HSMHV2_NF          77
#define HSMHV2_SA          78
#define HSMHV2_SB          79
#define HSMHV2_SD          80
#define HSMHV2_NSUBCDFM    82
#define HSMHV2_M           83
#define HSMHV2_SUBLD1      86
#define HSMHV2_SUBLD2      87
#define HSMHV2_LOVER       41
#define HSMHV2_LOVERS      42
#define HSMHV2_LOVERLD     43
#define HSMHV2_LDRIFT1     88
#define HSMHV2_LDRIFT2     89
#define HSMHV2_LDRIFT1S    90
#define HSMHV2_LDRIFT2S    91
/* #define HSMHV2_RTH0        84	not used */
/* #define HSMHV2_CTH0        85	not used */




/* model parameters */
#define HSMHV2_MOD_VBSMIN    198
#define HSMHV2_MOD_VMAX      500
#define HSMHV2_MOD_VMAXT1    503
#define HSMHV2_MOD_VMAXT2    504
#define HSMHV2_MOD_BGTMP1    101
#define HSMHV2_MOD_BGTMP2    102
#define HSMHV2_MOD_EG0       103
#define HSMHV2_MOD_TOX       104
#define HSMHV2_MOD_XLD       105
#define HSMHV2_MOD_LOVER     106
#define HSMHV2_MOD_LOVERS    385
#define HSMHV2_MOD_RDOV11    313
#define HSMHV2_MOD_RDOV12    314
#define HSMHV2_MOD_RDOV13    476
#define HSMHV2_MOD_RDSLP1    315
#define HSMHV2_MOD_RDICT1    316
#define HSMHV2_MOD_RDSLP2    317
#define HSMHV2_MOD_RDICT2    318
#define HSMHV2_MOD_LOVERLD   436
#define HSMHV2_MOD_LDRIFT1   319
#define HSMHV2_MOD_LDRIFT2   320
#define HSMHV2_MOD_LDRIFT1S  324
#define HSMHV2_MOD_LDRIFT2S  325
#define HSMHV2_MOD_SUBLD1    321
#define HSMHV2_MOD_SUBLD1L   329
#define HSMHV2_MOD_SUBLD1LP  330
#define HSMHV2_MOD_SUBLD2    322
#define HSMHV2_MOD_XPDV      326
#define HSMHV2_MOD_XPVDTH    327
#define HSMHV2_MOD_XPVDTHG   328
#define HSMHV2_MOD_DDLTMAX   421 /* Vdseff */
#define HSMHV2_MOD_DDLTSLP   422 /* Vdseff */
#define HSMHV2_MOD_DDLTICT   423 /* Vdseff */
#define HSMHV2_MOD_VFBOVER   428
#define HSMHV2_MOD_NOVER     430
#define HSMHV2_MOD_NOVERS    431
#define HSMHV2_MOD_XWD       107
#define HSMHV2_MOD_XWDC      513
#define HSMHV2_MOD_XL        112
#define HSMHV2_MOD_XW        117
#define HSMHV2_MOD_SAREF     433
#define HSMHV2_MOD_SBREF     434
#define HSMHV2_MOD_LL        108
#define HSMHV2_MOD_LLD       109
#define HSMHV2_MOD_LLN       110
#define HSMHV2_MOD_WL        111
#define HSMHV2_MOD_WL1       113
#define HSMHV2_MOD_WL1P      114
#define HSMHV2_MOD_WL2       407
#define HSMHV2_MOD_WL2P      408
#define HSMHV2_MOD_WLD       115
#define HSMHV2_MOD_WLN       116

#define HSMHV2_MOD_XQY       178
#define HSMHV2_MOD_XQY1      118
#define HSMHV2_MOD_XQY2      120
#define HSMHV2_MOD_RSH       119

#define HSMHV2_MOD_RSHG      384
/* #define HSMHV2_MOD_NGCON     385 */
/* #define HSMHV2_MOD_XGW       386 */
/* #define HSMHV2_MOD_XGL       387 */
/* #define HSMHV2_MOD_NF        388 */
#define HSMHV2_MOD_RS        398
#define HSMHV2_MOD_RD        399

#define HSMHV2_MOD_VFBC      121
#define HSMHV2_MOD_VBI       122
#define HSMHV2_MOD_NSUBC     123
#define HSMHV2_MOD_TNOM      124
#define HSMHV2_MOD_PARL2     125
#define HSMHV2_MOD_SC1       126
#define HSMHV2_MOD_SC2       127
#define HSMHV2_MOD_SC3       128
#define HSMHV2_MOD_SC4       248
#define HSMHV2_MOD_NDEP      129
#define HSMHV2_MOD_NDEPL     419
#define HSMHV2_MOD_NDEPLP    420
#define HSMHV2_MOD_NINV      130
#define HSMHV2_MOD_NINVD     505
#define HSMHV2_MOD_NINVDW    506
#define HSMHV2_MOD_NINVDWP   507
#define HSMHV2_MOD_NINVDT1   508
#define HSMHV2_MOD_NINVDT2   509
#define HSMHV2_MOD_MUECB0    131
#define HSMHV2_MOD_MUECB1    132
#define HSMHV2_MOD_MUEPH1    133
#define HSMHV2_MOD_MUEPH0    134
#define HSMHV2_MOD_MUEPHW    135
#define HSMHV2_MOD_MUEPWP    136
#define HSMHV2_MOD_MUEPHL    137
#define HSMHV2_MOD_MUEPLP    138
#define HSMHV2_MOD_MUEPHS    139
#define HSMHV2_MOD_MUEPSP    140
#define HSMHV2_MOD_VTMP      141
#define HSMHV2_MOD_WVTH0 	   142
#define HSMHV2_MOD_MUESR1    143
#define HSMHV2_MOD_MUESR0    144
#define HSMHV2_MOD_MUESRL    145
#define HSMHV2_MOD_MUESLP    146
#define HSMHV2_MOD_MUESRW    147
#define HSMHV2_MOD_MUESWP    148
#define HSMHV2_MOD_BB        149

#define HSMHV2_MOD_SUB1      151
#define HSMHV2_MOD_SUB2      152
#define HSMHV2_MOD_CGSO      154
#define HSMHV2_MOD_CGDO      155
#define HSMHV2_MOD_CGBO      156
#define HSMHV2_MOD_JS0       157
#define HSMHV2_MOD_JS0SW     158
#define HSMHV2_MOD_NJ        159
#define HSMHV2_MOD_NJSW      160
#define HSMHV2_MOD_XTI       161
#define HSMHV2_MOD_CJ        162
#define HSMHV2_MOD_CJSW      163
#define HSMHV2_MOD_CJSWG     164
#define HSMHV2_MOD_MJ        165
#define HSMHV2_MOD_MJSW      166
#define HSMHV2_MOD_MJSWG     167
#define HSMHV2_MOD_XTI2      168
#define HSMHV2_MOD_CISB      169
#define HSMHV2_MOD_CVB       170
#define HSMHV2_MOD_CTEMP     171
#define HSMHV2_MOD_CISBK     172
#define HSMHV2_MOD_CVBK      173
#define HSMHV2_MOD_DIVX      174
#define HSMHV2_MOD_PB        175
#define HSMHV2_MOD_PBSW      176
#define HSMHV2_MOD_PBSWG     177
#define HSMHV2_MOD_TPOLY     179
/* #define HSMHV2_MOD_TPOLYLD   	not used */
#define HSMHV2_MOD_LP        180
#define HSMHV2_MOD_NSUBP     181
#define HSMHV2_MOD_NSUBP0    182
#define HSMHV2_MOD_NSUBWP    183
#define HSMHV2_MOD_NDEPM     600 
#define HSMHV2_MOD_TNDEP     601
#define HSMHV2_MOD_DEPMUE0 605
#define HSMHV2_MOD_DEPMUE1 606
#define HSMHV2_MOD_DEPMUEBACK0 607
#define HSMHV2_MOD_DEPMUEBACK1 608
#define HSMHV2_MOD_DEPVMAX 609
#define HSMHV2_MOD_DEPBB 610
#define HSMHV2_MOD_DEPVDSEF1 611
#define HSMHV2_MOD_DEPVDSEF2 612
#define HSMHV2_MOD_DEPMUEPH0 613
#define HSMHV2_MOD_DEPMUEPH1 614
#define HSMHV2_MOD_DEPLEAK   615
#define HSMHV2_MOD_DEPETA    616
#define HSMHV2_MOD_DEPVTMP   617
#define HSMHV2_MOD_DEPMUETMP 618
#define HSMHV2_MOD_ISBREAK   619
#define HSMHV2_MOD_RWELL     620
#define HSMHV2_MOD_SCP1      184
#define HSMHV2_MOD_SCP2      185
#define HSMHV2_MOD_SCP3      186
#define HSMHV2_MOD_PGD1      187
#define HSMHV2_MOD_PGD2      188
//#define HSMHV2_MOD_PGD3      
#define HSMHV2_MOD_PGD4      190
#define HSMHV2_MOD_CLM1      191
#define HSMHV2_MOD_CLM2      192
#define HSMHV2_MOD_CLM3      193
#define HSMHV2_MOD_CLM5      402
#define HSMHV2_MOD_CLM6      403
#define HSMHV2_MOD_MUETMP    195

#define HSMHV2_MOD_VOVER     199
#define HSMHV2_MOD_VOVERP    200
#define HSMHV2_MOD_WFC       201
#define HSMHV2_MOD_NSUBCW    249
#define HSMHV2_MOD_NSUBCWP   250
#define HSMHV2_MOD_QME1      202
#define HSMHV2_MOD_QME2      203
#define HSMHV2_MOD_QME3      204
#define HSMHV2_MOD_GIDL1     205
#define HSMHV2_MOD_GIDL2     206
#define HSMHV2_MOD_GIDL3     207
#define HSMHV2_MOD_GLEAK1    208
#define HSMHV2_MOD_GLEAK2    209
#define HSMHV2_MOD_GLEAK3    210
#define HSMHV2_MOD_GLEAK4    211
#define HSMHV2_MOD_GLEAK5    212
#define HSMHV2_MOD_GLEAK6    213
#define HSMHV2_MOD_GLEAK7    214
#define HSMHV2_MOD_GLPART1   406
#define HSMHV2_MOD_GLKSD1    215
#define HSMHV2_MOD_GLKSD2    216
#define HSMHV2_MOD_GLKSD3    217
#define HSMHV2_MOD_GLKB1     218
#define HSMHV2_MOD_GLKB2     219
#define HSMHV2_MOD_GLKB3     429
#define HSMHV2_MOD_EGIG      220
#define HSMHV2_MOD_IGTEMP2   221
#define HSMHV2_MOD_IGTEMP3   222
#define HSMHV2_MOD_VZADD0    223
#define HSMHV2_MOD_PZADD0    224
#define HSMHV2_MOD_NSTI      225
#define HSMHV2_MOD_WSTI      226
#define HSMHV2_MOD_WSTIL     227
#define HSMHV2_MOD_WSTILP    231
#define HSMHV2_MOD_WSTIW     234
#define HSMHV2_MOD_WSTIWP    228
#define HSMHV2_MOD_SCSTI1    229
#define HSMHV2_MOD_SCSTI2    230
#define HSMHV2_MOD_VTHSTI    232
#define HSMHV2_MOD_VDSTI     233
#define HSMHV2_MOD_MUESTI1   235
#define HSMHV2_MOD_MUESTI2   236
#define HSMHV2_MOD_MUESTI3   237
#define HSMHV2_MOD_NSUBPSTI1 238
#define HSMHV2_MOD_NSUBPSTI2 239
#define HSMHV2_MOD_NSUBPSTI3 240
#define HSMHV2_MOD_LPEXT     241
#define HSMHV2_MOD_NPEXT     242
#define HSMHV2_MOD_SCP22     243
#define HSMHV2_MOD_SCP21     244
#define HSMHV2_MOD_BS1       245
#define HSMHV2_MOD_BS2       246
#define HSMHV2_MOD_KAPPA     251
//#define HSMHV2_MOD_PTHROU    253
#define HSMHV2_MOD_VDIFFJ    254
#define HSMHV2_MOD_DLY1      255
#define HSMHV2_MOD_DLY2      256
#define HSMHV2_MOD_DLY3      257
#define HSMHV2_MOD_NFTRP     258
#define HSMHV2_MOD_NFALP     259
#define HSMHV2_MOD_CIT       260
#define HSMHV2_MOD_FALPH     263
#define HSMHV2_MOD_OVSLP     261
#define HSMHV2_MOD_OVMAG     262
#define HSMHV2_MOD_GIDL4     281
#define HSMHV2_MOD_GIDL5     282
#define HSMHV2_MOD_SVGS      283
#define HSMHV2_MOD_SVBS      284
#define HSMHV2_MOD_SVBSL     285
#define HSMHV2_MOD_SVDS      286
#define HSMHV2_MOD_SLG       287
#define HSMHV2_MOD_SUB1L     290
#define HSMHV2_MOD_SUB2L     292
#define HSMHV2_MOD_FN1       294
#define HSMHV2_MOD_FN2       295
#define HSMHV2_MOD_FN3       296
#define HSMHV2_MOD_FVBS      297
#define HSMHV2_MOD_VOVERS    303
#define HSMHV2_MOD_VOVERSP   304
#define HSMHV2_MOD_SVGSL     305
#define HSMHV2_MOD_SVGSLP    306
#define HSMHV2_MOD_SVGSWP    307
#define HSMHV2_MOD_SVGSW     308
#define HSMHV2_MOD_SVBSLP    309
#define HSMHV2_MOD_SLGL      310
#define HSMHV2_MOD_SLGLP     311
#define HSMHV2_MOD_SUB1LP    312
#define HSMHV2_MOD_IBPC1     404
#define HSMHV2_MOD_IBPC1L    331
#define HSMHV2_MOD_IBPC1LP   332
#define HSMHV2_MOD_IBPC2     405
#define HSMHV2_MOD_MPHDFM    409

#define HSMHV2_MOD_PTL       530
#define HSMHV2_MOD_PTP       531
#define HSMHV2_MOD_PT2       532
#define HSMHV2_MOD_PTLP      533
#define HSMHV2_MOD_GDL       534
#define HSMHV2_MOD_GDLP      535

#define HSMHV2_MOD_GDLD      536
#define HSMHV2_MOD_PT4       537
#define HSMHV2_MOD_PT4P      538
#define HSMHV2_MOD_RDVG11    424
#define HSMHV2_MOD_RDVG12    425
#define HSMHV2_MOD_RTH0      432
#define HSMHV2_MOD_CTH0      462
#define HSMHV2_MOD_POWRAT    463
/* #define HSMHV2_MOD_WTH0      463 /\*---------SHE----------*\/ */
#define HSMHV2_MOD_DLYOV     437
#define HSMHV2_MOD_QDFTVD    438
#define HSMHV2_MOD_XLDLD     439
#define HSMHV2_MOD_XWDLD     494
#define HSMHV2_MOD_RDVD      510
#define HSMHV2_MOD_RDVB      301

#define HSMHV2_MOD_RDVSUB    481 /* substrate effect */
#define HSMHV2_MOD_RDVDSUB   482 /* substrate effect */
#define HSMHV2_MOD_DDRIFT    483 /* substrate effect */
#define HSMHV2_MOD_VBISUB    484 /* substrate effect */
#define HSMHV2_MOD_NSUBSUB   485 /* substrate effect */

//#define HSMHV2_MOD_QOVSM     323
//#define HSMHV2_MOD_LDRIFT    458
#define HSMHV2_MOD_RD20      447 
#define HSMHV2_MOD_RD21      441 
#define HSMHV2_MOD_RD22      442 
#define HSMHV2_MOD_RD22D     478
#define HSMHV2_MOD_RD23      443 
#define HSMHV2_MOD_RD24      444 
#define HSMHV2_MOD_RD25      445 
#define HSMHV2_MOD_RDVDL     448 
#define HSMHV2_MOD_RDVDLP    449 
#define HSMHV2_MOD_RDVDS     450 
#define HSMHV2_MOD_RDVDSP    451 
#define HSMHV2_MOD_RD23L     452 
#define HSMHV2_MOD_RD23LP    453 
#define HSMHV2_MOD_RD23S     454 
#define HSMHV2_MOD_RD23SP    455 
#define HSMHV2_MOD_RDS       456 
#define HSMHV2_MOD_RDSP      457 
#define HSMHV2_MOD_RDTEMP1   461
#define HSMHV2_MOD_RDTEMP2   464
#define HSMHV2_MOD_RTH0R     470
#define HSMHV2_MOD_RDVDTEMP1 471
#define HSMHV2_MOD_RDVDTEMP2 472
#define HSMHV2_MOD_RTH0W     473
#define HSMHV2_MOD_RTH0WP    474
#define HSMHV2_MOD_RTH0NF    475

#define HSMHV2_MOD_RTHTEMP1  490
#define HSMHV2_MOD_RTHTEMP2  491
#define HSMHV2_MOD_PRATTEMP1 492
#define HSMHV2_MOD_PRATTEMP2 493


#define HSMHV2_MOD_CVDSOVER  480

#define HSMHV2_MOD_RDRMUE    520 
#define HSMHV2_MOD_RDRVMAX   521 
#define HSMHV2_MOD_RDRMUETMP 522 
#define HSMHV2_MOD_M0LEAK    524 
#define HSMHV2_MOD_M0LEAKTMP 525 
#define HSMHV2_MOD_LEAKVMAX  543 
#define HSMHV2_MOD_M0TEMP1   551 
#define HSMHV2_MOD_M0TEMP2   553 

#define HSMHV2_MOD_RDRVTMP   523 
#define HSMHV2_MOD_RDRDJUNC  527 
#define HSMHV2_MOD_RDRCX     528 
#define HSMHV2_MOD_RDRCAR    529 
#define HSMHV2_MOD_RDRDL1    540 
#define HSMHV2_MOD_RDRDL2    541 
#define HSMHV2_MOD_RDRVMAXW  544
#define HSMHV2_MOD_RDRVMAXWP 545
#define HSMHV2_MOD_RDRVMAXL  546
#define HSMHV2_MOD_RDRVMAXLP 547
#define HSMHV2_MOD_RDRMUEL   548
#define HSMHV2_MOD_RDRMUELP  549
#define HSMHV2_MOD_RDRQOVER  552
#define HSMHV2_MOD_QOVADD      338
#define HSMHV2_MOD_JS0D        100
#define HSMHV2_MOD_JS0SWD      150
#define HSMHV2_MOD_NJD         153
#define HSMHV2_MOD_NJSWD       189
#define HSMHV2_MOD_XTID        194
#define HSMHV2_MOD_CJD         196
#define HSMHV2_MOD_CJSWD       197
#define HSMHV2_MOD_CJSWGD      247
#define HSMHV2_MOD_MJD         252
#define HSMHV2_MOD_MJSWD       253
#define HSMHV2_MOD_MJSWGD      264
#define HSMHV2_MOD_PBD         265
#define HSMHV2_MOD_PBSWD       266
#define HSMHV2_MOD_PBSWDG      267
#define HSMHV2_MOD_XTI2D       268
#define HSMHV2_MOD_CISBD       269
#define HSMHV2_MOD_CVBD        270
#define HSMHV2_MOD_CTEMPD      271
#define HSMHV2_MOD_CISBKD      272
#define HSMHV2_MOD_DIVXD       274
#define HSMHV2_MOD_VDIFFJD     275
#define HSMHV2_MOD_JS0S        276
#define HSMHV2_MOD_JS0SWS      277
#define HSMHV2_MOD_NJS         278
#define HSMHV2_MOD_NJSWS       279
#define HSMHV2_MOD_XTIS        280
#define HSMHV2_MOD_CJS         288
#define HSMHV2_MOD_CJSSW       289
#define HSMHV2_MOD_CJSWGS      291
#define HSMHV2_MOD_MJS         293
#define HSMHV2_MOD_MJSWS       298
#define HSMHV2_MOD_MJSWGS      299
#define HSMHV2_MOD_PBS         300
#define HSMHV2_MOD_PBSWS       302
#define HSMHV2_MOD_PBSWSG      323
#define HSMHV2_MOD_XTI2S       333
#define HSMHV2_MOD_CISBS       334
#define HSMHV2_MOD_CVBS        335
#define HSMHV2_MOD_CTEMPS      336
#define HSMHV2_MOD_CISBKS      337
#define HSMHV2_MOD_DIVXS       339
#define HSMHV2_MOD_VDIFFJS     340
#define HSMHV2_MOD_SHEMAX      501
#define HSMHV2_MOD_VGSMIN      502
#define HSMHV2_MOD_GDSLEAK     511
#define HSMHV2_MOD_RDRBB       273
#define HSMHV2_MOD_RDRBBTMP    602

/* binning parameters */
#define HSMHV2_MOD_LMIN       1000
#define HSMHV2_MOD_LMAX       1001
#define HSMHV2_MOD_WMIN       1002
#define HSMHV2_MOD_WMAX       1003
#define HSMHV2_MOD_LBINN      1004
#define HSMHV2_MOD_WBINN      1005

/* Length dependence */
#define HSMHV2_MOD_LVMAX      1100
#define HSMHV2_MOD_LBGTMP1    1101
#define HSMHV2_MOD_LBGTMP2    1102
#define HSMHV2_MOD_LEG0       1103
#define HSMHV2_MOD_LVFBOVER   1428
#define HSMHV2_MOD_LNOVER     1430
#define HSMHV2_MOD_LNOVERS    1431
#define HSMHV2_MOD_LWL2       1407
#define HSMHV2_MOD_LVFBC      1121
#define HSMHV2_MOD_LNSUBC     1123
#define HSMHV2_MOD_LNSUBP     1181
#define HSMHV2_MOD_LSCP1      1184
#define HSMHV2_MOD_LSCP2      1185
#define HSMHV2_MOD_LSCP3      1186
#define HSMHV2_MOD_LSC1       1126
#define HSMHV2_MOD_LSC2       1127
#define HSMHV2_MOD_LSC3       1128
#define HSMHV2_MOD_LPGD1      1187
//#define HSMHV2_MOD_LPGD3      1189
#define HSMHV2_MOD_LNDEP      1129
#define HSMHV2_MOD_LNINV      1130
#define HSMHV2_MOD_LMUECB0    1131
#define HSMHV2_MOD_LMUECB1    1132
#define HSMHV2_MOD_LMUEPH1    1133
#define HSMHV2_MOD_LVTMP      1141
#define HSMHV2_MOD_LWVTH0     1142
#define HSMHV2_MOD_LMUESR1    1143
#define HSMHV2_MOD_LMUETMP    1195
#define HSMHV2_MOD_LSUB1      1151
#define HSMHV2_MOD_LSUB2      1152
#define HSMHV2_MOD_LSVDS      1286
#define HSMHV2_MOD_LSVBS      1284
#define HSMHV2_MOD_LSVGS      1283
#define HSMHV2_MOD_LFN1       1294
#define HSMHV2_MOD_LFN2       1295
#define HSMHV2_MOD_LFN3       1296
#define HSMHV2_MOD_LFVBS      1297
#define HSMHV2_MOD_LNSTI      1225
#define HSMHV2_MOD_LWSTI      1226
#define HSMHV2_MOD_LSCSTI1    1229
#define HSMHV2_MOD_LSCSTI2    1230
#define HSMHV2_MOD_LVTHSTI    1232
#define HSMHV2_MOD_LMUESTI1   1235
#define HSMHV2_MOD_LMUESTI2   1236
#define HSMHV2_MOD_LMUESTI3   1237
#define HSMHV2_MOD_LNSUBPSTI1 1238
#define HSMHV2_MOD_LNSUBPSTI2 1239
#define HSMHV2_MOD_LNSUBPSTI3 1240
#define HSMHV2_MOD_LCGSO      1154
#define HSMHV2_MOD_LCGDO      1155
#define HSMHV2_MOD_LJS0       1157
#define HSMHV2_MOD_LJS0SW     1158
#define HSMHV2_MOD_LNJ        1159
#define HSMHV2_MOD_LCISBK     1172
#define HSMHV2_MOD_LCLM1      1191
#define HSMHV2_MOD_LCLM2      1192
#define HSMHV2_MOD_LCLM3      1193
#define HSMHV2_MOD_LWFC       1201
#define HSMHV2_MOD_LGIDL1     1205
#define HSMHV2_MOD_LGIDL2     1206
#define HSMHV2_MOD_LGLEAK1    1208
#define HSMHV2_MOD_LGLEAK2    1209
#define HSMHV2_MOD_LGLEAK3    1210
#define HSMHV2_MOD_LGLEAK6    1213
#define HSMHV2_MOD_LGLKSD1    1215
#define HSMHV2_MOD_LGLKSD2    1216
#define HSMHV2_MOD_LGLKB1     1218
#define HSMHV2_MOD_LGLKB2     1219
#define HSMHV2_MOD_LNFTRP     1258
#define HSMHV2_MOD_LNFALP     1259
//#define HSMHV2_MOD_LPTHROU    1253
#define HSMHV2_MOD_LVDIFFJ    1254
#define HSMHV2_MOD_LIBPC1     1404
#define HSMHV2_MOD_LIBPC2     1405
#define HSMHV2_MOD_LCGBO      1156
#define HSMHV2_MOD_LCVDSOVER  1480
#define HSMHV2_MOD_LFALPH     1263
#define HSMHV2_MOD_LNPEXT     1242
#define HSMHV2_MOD_LPOWRAT    1463
#define HSMHV2_MOD_LRD        1399
#define HSMHV2_MOD_LRD22      1442
#define HSMHV2_MOD_LRD23      1443
#define HSMHV2_MOD_LRD24      1444
#define HSMHV2_MOD_LRDICT1    1316
#define HSMHV2_MOD_LRDOV13    1476
#define HSMHV2_MOD_LRDSLP1    1315
#define HSMHV2_MOD_LRDVB      1301
#define HSMHV2_MOD_LRDVD      1510
#define HSMHV2_MOD_LRDVG11    1424
#define HSMHV2_MOD_LRS        1398
#define HSMHV2_MOD_LRTH0      1432
#define HSMHV2_MOD_LVOVER     1199
#define HSMHV2_MOD_LJS0D       345
#define HSMHV2_MOD_LJS0SWD     370
#define HSMHV2_MOD_LNJD        372
#define HSMHV2_MOD_LCISBKD     386
#define HSMHV2_MOD_LVDIFFJD    387
#define HSMHV2_MOD_LJS0S       388
#define HSMHV2_MOD_LJS0SWS     395
#define HSMHV2_MOD_LNJS        396
#define HSMHV2_MOD_LCISBKS     397
#define HSMHV2_MOD_LVDIFFJS    400

/* Width dependence */
#define HSMHV2_MOD_WVMAX      2100
#define HSMHV2_MOD_WBGTMP1    2101
#define HSMHV2_MOD_WBGTMP2    2102
#define HSMHV2_MOD_WEG0       2103
#define HSMHV2_MOD_WVFBOVER   2428
#define HSMHV2_MOD_WNOVER     2430
#define HSMHV2_MOD_WNOVERS    2431
#define HSMHV2_MOD_WWL2       2407
#define HSMHV2_MOD_WVFBC      2121
#define HSMHV2_MOD_WNSUBC     2123
#define HSMHV2_MOD_WNSUBP     2181
#define HSMHV2_MOD_WSCP1      2184
#define HSMHV2_MOD_WSCP2      2185
#define HSMHV2_MOD_WSCP3      2186
#define HSMHV2_MOD_WSC1       2126
#define HSMHV2_MOD_WSC2       2127
#define HSMHV2_MOD_WSC3       2128
#define HSMHV2_MOD_WPGD1      2187
//#define HSMHV2_MOD_WPGD3      2189
#define HSMHV2_MOD_WNDEP      2129
#define HSMHV2_MOD_WNINV      2130
#define HSMHV2_MOD_WMUECB0    2131
#define HSMHV2_MOD_WMUECB1    2132
#define HSMHV2_MOD_WMUEPH1    2133
#define HSMHV2_MOD_WVTMP      2141
#define HSMHV2_MOD_WWVTH0     2142
#define HSMHV2_MOD_WMUESR1    2143
#define HSMHV2_MOD_WMUETMP    2195
#define HSMHV2_MOD_WSUB1      2151
#define HSMHV2_MOD_WSUB2      2152
#define HSMHV2_MOD_WSVDS      2286
#define HSMHV2_MOD_WSVBS      2284
#define HSMHV2_MOD_WSVGS      2283
#define HSMHV2_MOD_WFN1       2294
#define HSMHV2_MOD_WFN2       2295
#define HSMHV2_MOD_WFN3       2296
#define HSMHV2_MOD_WFVBS      2297
#define HSMHV2_MOD_WNSTI      2225
#define HSMHV2_MOD_WWSTI      2226
#define HSMHV2_MOD_WSCSTI1    2229
#define HSMHV2_MOD_WSCSTI2    2230
#define HSMHV2_MOD_WVTHSTI    2232
#define HSMHV2_MOD_WMUESTI1   2235
#define HSMHV2_MOD_WMUESTI2   2236
#define HSMHV2_MOD_WMUESTI3   2237
#define HSMHV2_MOD_WNSUBPSTI1 2238
#define HSMHV2_MOD_WNSUBPSTI2 2239
#define HSMHV2_MOD_WNSUBPSTI3 2240
#define HSMHV2_MOD_WCGSO      2154
#define HSMHV2_MOD_WCGDO      2155
#define HSMHV2_MOD_WJS0       2157
#define HSMHV2_MOD_WJS0SW     2158
#define HSMHV2_MOD_WNJ        2159
#define HSMHV2_MOD_WCISBK     2172
#define HSMHV2_MOD_WCLM1      2191
#define HSMHV2_MOD_WCLM2      2192
#define HSMHV2_MOD_WCLM3      2193
#define HSMHV2_MOD_WWFC       2201
#define HSMHV2_MOD_WGIDL1     2205
#define HSMHV2_MOD_WGIDL2     2206
#define HSMHV2_MOD_WGLEAK1    2208
#define HSMHV2_MOD_WGLEAK2    2209
#define HSMHV2_MOD_WGLEAK3    2210
#define HSMHV2_MOD_WGLEAK6    2213
#define HSMHV2_MOD_WGLKSD1    2215
#define HSMHV2_MOD_WGLKSD2    2216
#define HSMHV2_MOD_WGLKB1     2218
#define HSMHV2_MOD_WGLKB2     2219
#define HSMHV2_MOD_WNFTRP     2258
#define HSMHV2_MOD_WNFALP     2259
//#define HSMHV2_MOD_WPTHROU    2253
#define HSMHV2_MOD_WVDIFFJ    2254
#define HSMHV2_MOD_WIBPC1     2404
#define HSMHV2_MOD_WIBPC2     2405
#define HSMHV2_MOD_WCGBO      2156
#define HSMHV2_MOD_WCVDSOVER  2480
#define HSMHV2_MOD_WFALPH     2263
#define HSMHV2_MOD_WNPEXT     2242
#define HSMHV2_MOD_WPOWRAT    2463
#define HSMHV2_MOD_WRD        2399
#define HSMHV2_MOD_WRD22      2442
#define HSMHV2_MOD_WRD23      2443
#define HSMHV2_MOD_WRD24      2444
#define HSMHV2_MOD_WRDICT1    2316
#define HSMHV2_MOD_WRDOV13    2476
#define HSMHV2_MOD_WRDSLP1    2315
#define HSMHV2_MOD_WRDVB      2301
#define HSMHV2_MOD_WRDVD      2510
#define HSMHV2_MOD_WRDVG11    2424
#define HSMHV2_MOD_WRS        2398
#define HSMHV2_MOD_WRTH0      2432
#define HSMHV2_MOD_WVOVER     2199
#define HSMHV2_MOD_WJS0D       401
#define HSMHV2_MOD_WJS0SWD     435
#define HSMHV2_MOD_WNJD        440
#define HSMHV2_MOD_WCISBKD     446
#define HSMHV2_MOD_WVDIFFJD    459
#define HSMHV2_MOD_WJS0S       460
#define HSMHV2_MOD_WJS0SWS     467
#define HSMHV2_MOD_WNJS        468
#define HSMHV2_MOD_WCISBKS     469
#define HSMHV2_MOD_WVDIFFJS    477

/* Cross-term dependence */
#define HSMHV2_MOD_PVMAX      3100
#define HSMHV2_MOD_PBGTMP1    3101
#define HSMHV2_MOD_PBGTMP2    3102
#define HSMHV2_MOD_PEG0       3103
#define HSMHV2_MOD_PVFBOVER   3428
#define HSMHV2_MOD_PNOVER     3430
#define HSMHV2_MOD_PNOVERS    3431
#define HSMHV2_MOD_PWL2       3407
#define HSMHV2_MOD_PVFBC      3121
#define HSMHV2_MOD_PNSUBC     3123
#define HSMHV2_MOD_PNSUBP     3181
#define HSMHV2_MOD_PSCP1      3184
#define HSMHV2_MOD_PSCP2      3185
#define HSMHV2_MOD_PSCP3      3186
#define HSMHV2_MOD_PSC1       3126
#define HSMHV2_MOD_PSC2       3127
#define HSMHV2_MOD_PSC3       3128
#define HSMHV2_MOD_PPGD1      3187
//#define HSMHV2_MOD_PPGD3      3189
#define HSMHV2_MOD_PNDEP      3129
#define HSMHV2_MOD_PNINV      3130
#define HSMHV2_MOD_PMUECB0    3131
#define HSMHV2_MOD_PMUECB1    3132
#define HSMHV2_MOD_PMUEPH1    3133
#define HSMHV2_MOD_PVTMP      3141
#define HSMHV2_MOD_PWVTH0     3142
#define HSMHV2_MOD_PMUESR1    3143
#define HSMHV2_MOD_PMUETMP    3195
#define HSMHV2_MOD_PSUB1      3151
#define HSMHV2_MOD_PSUB2      3152
#define HSMHV2_MOD_PSVDS      3286
#define HSMHV2_MOD_PSVBS      3284
#define HSMHV2_MOD_PSVGS      3283
#define HSMHV2_MOD_PFN1       3294
#define HSMHV2_MOD_PFN2       3295
#define HSMHV2_MOD_PFN3       3296
#define HSMHV2_MOD_PFVBS      3297
#define HSMHV2_MOD_PNSTI      3225
#define HSMHV2_MOD_PWSTI      3226
#define HSMHV2_MOD_PSCSTI1    3229
#define HSMHV2_MOD_PSCSTI2    3230
#define HSMHV2_MOD_PVTHSTI    3232
#define HSMHV2_MOD_PMUESTI1   3235
#define HSMHV2_MOD_PMUESTI2   3236
#define HSMHV2_MOD_PMUESTI3   3237
#define HSMHV2_MOD_PNSUBPSTI1 3238
#define HSMHV2_MOD_PNSUBPSTI2 3239
#define HSMHV2_MOD_PNSUBPSTI3 3240
#define HSMHV2_MOD_PCGSO      3154
#define HSMHV2_MOD_PCGDO      3155
#define HSMHV2_MOD_PJS0       3157
#define HSMHV2_MOD_PJS0SW     3158
#define HSMHV2_MOD_PNJ        3159
#define HSMHV2_MOD_PCISBK     3172
#define HSMHV2_MOD_PCLM1      3191
#define HSMHV2_MOD_PCLM2      3192
#define HSMHV2_MOD_PCLM3      3193
#define HSMHV2_MOD_PWFC       3201
#define HSMHV2_MOD_PGIDL1     3205
#define HSMHV2_MOD_PGIDL2     3206
#define HSMHV2_MOD_PGLEAK1    3208
#define HSMHV2_MOD_PGLEAK2    3209
#define HSMHV2_MOD_PGLEAK3    3210
#define HSMHV2_MOD_PGLEAK6    3213
#define HSMHV2_MOD_PGLKSD1    3215
#define HSMHV2_MOD_PGLKSD2    3216
#define HSMHV2_MOD_PGLKB1     3218
#define HSMHV2_MOD_PGLKB2     3219
#define HSMHV2_MOD_PNFTRP     3258
#define HSMHV2_MOD_PNFALP     3259
//#define HSMHV2_MOD_PPTHROU    3253
#define HSMHV2_MOD_PVDIFFJ    3254
#define HSMHV2_MOD_PIBPC1     3404
#define HSMHV2_MOD_PIBPC2     3405
#define HSMHV2_MOD_PCGBO      3156
#define HSMHV2_MOD_PCVDSOVER  3480
#define HSMHV2_MOD_PFALPH     3263
#define HSMHV2_MOD_PNPEXT     3242
#define HSMHV2_MOD_PPOWRAT    3463
#define HSMHV2_MOD_PRD        3399
#define HSMHV2_MOD_PRD22      3442
#define HSMHV2_MOD_PRD23      3443
#define HSMHV2_MOD_PRD24      3444
#define HSMHV2_MOD_PRDICT1    3316
#define HSMHV2_MOD_PRDOV13    3476
#define HSMHV2_MOD_PRDSLP1    3315
#define HSMHV2_MOD_PRDVB      3301
#define HSMHV2_MOD_PRDVD      3510
#define HSMHV2_MOD_PRDVG11    3424
#define HSMHV2_MOD_PRS        3398
#define HSMHV2_MOD_PRTH0      3432
#define HSMHV2_MOD_PVOVER     3199
#define HSMHV2_MOD_PJS0D       479
#define HSMHV2_MOD_PJS0SWD     486
#define HSMHV2_MOD_PNJD        487
#define HSMHV2_MOD_PCISBKD     488
#define HSMHV2_MOD_PVDIFFJD    489
#define HSMHV2_MOD_PJS0S       495
#define HSMHV2_MOD_PJS0SWS     496
#define HSMHV2_MOD_PNJS        497
#define HSMHV2_MOD_PCISBKS     498
#define HSMHV2_MOD_PVDIFFJS    499

/* device requests */
#define HSMHV2_DNODE          341
#define HSMHV2_GNODE          342
#define HSMHV2_SNODE          343
#define HSMHV2_BNODE          344
/* #define HSMHV2_TEMPNODE       345	not used */
#define HSMHV2_DNODEPRIME     346
#define HSMHV2_SNODEPRIME     347
/* #define HSMHV2_BNODEPRIME     395	not used */
/* #define HSMHV2_DBNODE         396	not used */
/* #define HSMHV2_SBNODE         397	not used */
/* #define HSMHV2_VBD            347 */
#define HSMHV2_VBD            466
#define HSMHV2_VBS            348
#define HSMHV2_VGS            349
#define HSMHV2_VDS            350
#define HSMHV2_CD             351
#define HSMHV2_CBS            352
#define HSMHV2_CBD            353
#define HSMHV2_GM             354
#define HSMHV2_GDS            355
#define HSMHV2_GMBS           356
#define HSMHV2_GMT            465
/* #define HSMHV2_ISUBT          466 */
#define HSMHV2_GBD            357
#define HSMHV2_GBS            358
#define HSMHV2_QB             359
#define HSMHV2_CQB            360
/* #define HSMHV2_QTH            467	not used */
/* #define HSMHV2_CQTH           468	not used */
/* #define HSMHV2_CTH            469	not used */
#define HSMHV2_QG             361
#define HSMHV2_CQG            362
#define HSMHV2_QD             363
#define HSMHV2_CQD            364
#define HSMHV2_CGG            365
#define HSMHV2_CGD            366
#define HSMHV2_CGS            367
#define HSMHV2_CBG            368
#define HSMHV2_CAPBD          369
/* #define HSMHV2_CQBD           370	not used */
#define HSMHV2_CAPBS          371
/* #define HSMHV2_CQBS           372	not used */
#define HSMHV2_CDG            373
#define HSMHV2_CDD            374
#define HSMHV2_CDS            375
#define HSMHV2_VON            376
#define HSMHV2_VDSAT          377
#define HSMHV2_QBS            378
#define HSMHV2_QBD            379
#define HSMHV2_SOURCECONDUCT  380
#define HSMHV2_DRAINCONDUCT   381
#define HSMHV2_CBDB           382
#define HSMHV2_CBSB           383
#define HSMHV2_MOD_RBPB       389
#define HSMHV2_MOD_RBPD       390
#define HSMHV2_MOD_RBPS       391
#define HSMHV2_MOD_RBDB       392
#define HSMHV2_MOD_RBSB       393
#define HSMHV2_MOD_GBMIN      394

#define HSMHV2_ISUB           410
#define HSMHV2_ISUBLD         426
#define HSMHV2_IDSIBPC        427
#define HSMHV2_IGIDL          411
#define HSMHV2_IGISL          412
#define HSMHV2_IGD            413
#define HSMHV2_IGS            414
#define HSMHV2_IGB            415
#define HSMHV2_CGSO           416
#define HSMHV2_CGBO           417
#define HSMHV2_CGDO           418

#define HSMHV2_MOD_TCJBD         92
#define HSMHV2_MOD_TCJBS         93
#define HSMHV2_MOD_TCJBDSW       94 
#define HSMHV2_MOD_TCJBSSW       95  
#define HSMHV2_MOD_TCJBDSWG      96   
#define HSMHV2_MOD_TCJBSSWG      97   

#define HSMHV2_MOD_VGS_MAX          4001
#define HSMHV2_MOD_VGD_MAX          4002
#define HSMHV2_MOD_VGB_MAX          4003
#define HSMHV2_MOD_VDS_MAX          4004
#define HSMHV2_MOD_VBS_MAX          4005
#define HSMHV2_MOD_VBD_MAX          4006
#define HSMHV2_MOD_VGSR_MAX         4007
#define HSMHV2_MOD_VGDR_MAX         4008
#define HSMHV2_MOD_VGBR_MAX         4009
#define HSMHV2_MOD_VBSR_MAX         4010
#define HSMHV2_MOD_VBDR_MAX         4011

#include "hsmhv2ext.h"

/* Prototype has to be adapted! 
extern void HSMHV2evaluate(double,double,double,HSMHV2instance*,HSMHV2model*,
        double*,double*,double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, CKTcircuit*);
*/

#endif /*HSMHV2*/

