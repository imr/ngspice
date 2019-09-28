/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 2  SUBVERSION : 2  REVISION : 0 ) 
 Model Parameter 'VERSION' : 2.20
 FILE : hsmhvset.c

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

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "hsmhv2def.h"
#include "hsmhv2evalenv.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#define C_m2cm    (1.0e2)
#define C_cm2m_p3 (1.0e-6)
#define C_m2cm_p1o2 (1.0e1)

#define BINNING(param) pParam->HSMHV2_##param = model->HSMHV2_##param \
  + model->HSMHV2_l##param / Lbin + model->HSMHV2_w##param / Wbin \
  + model->HSMHV2_p##param / LWbin ;

#define RANGECHECK(param, min, max, pname)          \
  if ( model->HSMHV2_coerrrep && ((param) < (min) || (param) > (max)) ) { \
    printf("warning(HiSIM_HV(%s)): (%s = %g) range [%g , %g].\n", model->HSMHV2modName,\
           (pname), (param), (min*1.0), (max*1.0) );            \
  }
#define RANGERESET(param, min, max, pname)              \
  if ( model->HSMHV2_coerrrep && ((param) > (max)) ) {   \
    printf("reset(HiSIM_HV(%s)): (%s = %g to %g) range [%g , %g].\n", model->HSMHV2modName,\
           (pname), (param), (max*1.0), (min*1.0), (max*1.0) );     \
  } \
  if ( model->HSMHV2_coerrrep && ((param) < (min)) ) {   \
    printf("reset(HiSIM_HV(%s)): (%s = %g to %g) range [%g , %g].\n",model->HSMHV2modName, \
           (pname), (param), (min*1.0), (min*1.0), (max*1.0) );     \
  } \
  if ( (param) < (min) ) {  param  = (min); }    \
  if ( (param) > (max) ) {  param  = (max); }            
#define MINCHECK(param, min, pname)                     \
  if ( model->HSMHV2_coerrrep && ((param) < (min)) ) {   \
    printf("warning(HiSIM_HV(%s)): (%s = %g) range [%g , %g].\n",model->HSMHV2modName, \
           (pname), (param), (min*1.0), (min*1.0) );                   \
  }
#define MINRESET(param, min, pname)                     \
  if ( model->HSMHV2_coerrrep && ((param) < (min)) ) {   \
    printf("reset(HiSIM_HV(%s)): (%s = %g to %g) range [%g , %g].\n",model->HSMHV2modName,    \
           (pname), (param), (min*1.0), (min*1.0), (min*1.0) );            \
  } \
  if ( (param) < (min) ) {  param  = (min); } 

int HSMHV2setup(
     SMPmatrix *matrix,
     GENmodel *inModel,
     CKTcircuit *ckt,
     int *states)
     /* load the HSMHV2 device structure with those pointers needed later 
      * for fast matrix loading 
      */
{
  HSMHV2model *model = (HSMHV2model*)inModel;
  HSMHV2instance *here;
  int error=0 ;
  CKTnode *tmp;
  double T2, Rd, Rs ;
  HSMHV2binningParam *pParam ;
  HSMHV2modelMKSParam *modelMKS ;
  HSMHV2hereMKSParam  *hereMKS ;
  double LG=0.0, WG =0.0, Lgate =0.0, Wgate =0.0 ;
  double Lbin=0.0, Wbin=0.0, LWbin =0.0; /* binning */
  
  
  /*  loop through all the HSMHV2 device models */
  for ( ;model != NULL ;model = HSMHV2nextModel(model)) {
    /* Default value Processing for HVMOS Models */
    if ( !model->HSMHV2_type_Given )
      model->HSMHV2_type = NMOS ;

    if ( !model->HSMHV2_info_Given ) model->HSMHV2_info = 0 ;

    model->HSMHV2_noise = 1;

    if ( !model->HSMHV2_version_Given) {
      model->HSMHV2_version = "2.20" ;
      printf("HiSIM_HV(%s): 2.20 is selected for VERSION. (default) \n",model->HSMHV2modName);
      model->HSMHV2_subversion = 2 ;
    } else {
      if (strncmp(model->HSMHV2_version,"2.20", 4) == 0 ) {
        printf("HiSIM_HV(%s): 2.20 is selected for VERSION. (default) \n",model->HSMHV2modName);
        model->HSMHV2_subversion = 2 ;
      } else if (strncmp(model->HSMHV2_version,"2.2", 3) == 0 ) {
        printf("HiSIM_HV(%s): 2.20 is selected for VERSION. (default) \n",model->HSMHV2modName);
        model->HSMHV2_subversion = 2 ;
      } else {
        printf("warning(HiSIM_HV(%s)): invalid version %s is specified, reset to 2.20 \n",
        model->HSMHV2modName,model->HSMHV2_version);
        model->HSMHV2_subversion = 2 ;
      }
    }
    

    if ( !model->HSMHV2_corsrd_Given     ) model->HSMHV2_corsrd     = 3 ;
    if ( !model->HSMHV2_corg_Given       ) model->HSMHV2_corg       = 0 ;
    if ( !model->HSMHV2_coiprv_Given     ) model->HSMHV2_coiprv     = 1 ;
    if ( !model->HSMHV2_copprv_Given     ) model->HSMHV2_copprv     = 1 ;
    if ( !model->HSMHV2_coadov_Given     ) model->HSMHV2_coadov     = 1 ;
    if ( !model->HSMHV2_coisub_Given     ) model->HSMHV2_coisub     = 0 ;
    if ( !model->HSMHV2_coiigs_Given     ) model->HSMHV2_coiigs     = 0 ;
    if ( !model->HSMHV2_cogidl_Given     ) model->HSMHV2_cogidl     = 0 ;
    if ( !model->HSMHV2_coovlp_Given     ) model->HSMHV2_coovlp     = 1 ;
    if ( !model->HSMHV2_coovlps_Given    ) model->HSMHV2_coovlps    = 0 ;
    if ( !model->HSMHV2_coflick_Given    ) model->HSMHV2_coflick    = 0 ;
    if ( !model->HSMHV2_coisti_Given     ) model->HSMHV2_coisti     = 0 ;
    if ( !model->HSMHV2_conqs_Given      ) model->HSMHV2_conqs      = 0 ; /* QS (default) */
    if ( !model->HSMHV2_cothrml_Given    ) model->HSMHV2_cothrml    = 0 ;
    if ( !model->HSMHV2_coign_Given      ) model->HSMHV2_coign      = 0 ; /* induced gate noise */
    if ( !model->HSMHV2_codfm_Given      ) model->HSMHV2_codfm      = 0 ; /* DFM */
    if ( !model->HSMHV2_coqovsm_Given    ) model->HSMHV2_coqovsm    = 1 ; 
    if ( !model->HSMHV2_corbnet_Given    ) model->HSMHV2_corbnet    = 0 ; 
    else if ( model->HSMHV2_corbnet != 0 && model->HSMHV2_corbnet != 1 ) {
      model->HSMHV2_corbnet = 0;
      printf("warning(HiSIM_HV(%s)): CORBNET has been set to its default value: %d.\n", 
      model->HSMHV2modName,model->HSMHV2_corbnet);
    }
    if ( !model->HSMHV2_coselfheat_Given ) model->HSMHV2_coselfheat = 0 ; /* Self-heating model */
    if ( !model->HSMHV2_cosubnode_Given  ) model->HSMHV2_cosubnode  = 0 ; 
    if ( !model->HSMHV2_cosym_Given ) model->HSMHV2_cosym = 0 ;           /* Symmetry model for HV */
    if ( !model->HSMHV2_cotemp_Given ) model->HSMHV2_cotemp = 0 ;
    if ( !model->HSMHV2_cordrift_Given ) model->HSMHV2_cordrift = 1 ;
    model->HSMHV2_coldrift = 1 ;
    if (  model->HSMHV2_coldrift_Given ) {
      fprintf(stderr,"warning(HiSIM_HV(%s)): COLDRIFT has been inactivated ( Ldrift = LDRIFT1 + LDRIFT2 ).\n",model->HSMHV2modName);
    }
    if ( !model->HSMHV2_coerrrep_Given ) model->HSMHV2_coerrrep = 1 ;
    if ( !model->HSMHV2_codep_Given ) model->HSMHV2_codep = 0 ;
    if ( model->HSMHV2_codep_Given ) {
      if( model->HSMHV2_codep != 0 && model->HSMHV2_codep != 1 ) {
        printf("warning(HiSIM_HV(%s)): Invalid model parameter CODEP  (= %d) was specified, reset to 0.\n",model->HSMHV2modName,model->HSMHV2_codep);
        model->HSMHV2_codep = 0 ;
      }
    }

    if ( !model->HSMHV2_coddlt_Given ) model->HSMHV2_coddlt = 1 ;
    if ( model->HSMHV2_coddlt_Given ) {
      if( model->HSMHV2_coddlt != 0 && model->HSMHV2_coddlt != 1 ) {
        printf("warning(HiSIM_HV(%s)): Invalid model parameter CODDLT  (= %d) was specified, reset to 1.\n",model->HSMHV2modName,model->HSMHV2_coddlt);
        model->HSMHV2_coddlt = 1 ;
      }
    }

    if ( !model->HSMHV2_vmax_Given    ) model->HSMHV2_vmax    = 1.0e7 ;
    if ( !model->HSMHV2_vmaxt1_Given  ) model->HSMHV2_vmaxt1  = 0.0 ;
    if ( !model->HSMHV2_vmaxt2_Given  ) model->HSMHV2_vmaxt2  = 0.0 ;
    if ( !model->HSMHV2_bgtmp1_Given  ) model->HSMHV2_bgtmp1  = 90.25e-6 ;
    if ( !model->HSMHV2_bgtmp2_Given  ) model->HSMHV2_bgtmp2  = 1.0e-7 ;
    if ( !model->HSMHV2_eg0_Given     ) model->HSMHV2_eg0     = 1.1785e0 ;
    if ( !model->HSMHV2_tox_Given     ) model->HSMHV2_tox     = 7e-9 ;
    if ( !model->HSMHV2_xld_Given     ) model->HSMHV2_xld     = 0 ;
    if ( !model->HSMHV2_lover_Given   ) model->HSMHV2_lover   = 30e-9 ;
//  if ( !model->HSMHV2_lovers_Given  ) model->HSMHV2_lovers  = 30e-9 ;
//  if (  model->HSMHV2_lover_Given   ) model->HSMHV2_lovers  = model->HSMHV2_lover ;
    if ( !model->HSMHV2_rdov11_Given  ) model->HSMHV2_rdov11   = 0.0 ;
    if ( !model->HSMHV2_rdov12_Given  ) model->HSMHV2_rdov12   = 1.0 ;
    if ( !model->HSMHV2_rdov13_Given  ) model->HSMHV2_rdov13   = 1.0 ;
    if ( !model->HSMHV2_rdslp1_Given  ) model->HSMHV2_rdslp1   = 0.0 ;
    if ( !model->HSMHV2_rdict1_Given  ) model->HSMHV2_rdict1   = 1.0 ;
    if ( !model->HSMHV2_rdslp2_Given  ) model->HSMHV2_rdslp2   = 1.0 ;
    if ( !model->HSMHV2_rdict2_Given  ) model->HSMHV2_rdict2   = 0.0 ;
    if ( !model->HSMHV2_loverld_Given ) model->HSMHV2_loverld  = 1.0e-6 ;
    if ( !model->HSMHV2_ldrift1_Given ) model->HSMHV2_ldrift1  = 1.0e-6 ;
    if ( !model->HSMHV2_ldrift2_Given ) model->HSMHV2_ldrift2  = 1.0e-6 ;
    if ( !model->HSMHV2_ldrift1s_Given ) model->HSMHV2_ldrift1s  = 0.0 ;
    if ( !model->HSMHV2_ldrift2s_Given ) model->HSMHV2_ldrift2s  = 1.0e-6 ;
    if ( !model->HSMHV2_subld1_Given  ) model->HSMHV2_subld1  = 0.0 ;
    if ( !model->HSMHV2_subld1l_Given  ) model->HSMHV2_subld1l  = 0.0 ;
    if ( !model->HSMHV2_subld1lp_Given  ) model->HSMHV2_subld1lp  = 1.0 ;
    if ( !model->HSMHV2_subld2_Given  ) model->HSMHV2_subld2  = 0.0 ;
    if ( !model->HSMHV2_xpdv_Given    ) model->HSMHV2_xpdv    = 0.0 ;
    if ( !model->HSMHV2_xpvdth_Given  ) model->HSMHV2_xpvdth  = 0.0 ;
    if ( !model->HSMHV2_xpvdthg_Given ) model->HSMHV2_xpvdthg = 0.0 ;
    if ( !model->HSMHV2_ddltmax_Given ) model->HSMHV2_ddltmax = 10  ;  /* Vdseff */
    if ( !model->HSMHV2_ddltslp_Given ) model->HSMHV2_ddltslp = 0.0 ;  /* Vdseff */
    if ( !model->HSMHV2_ddltict_Given ) model->HSMHV2_ddltict = 10.0 ; /* Vdseff */
    if ( !model->HSMHV2_vfbover_Given ) model->HSMHV2_vfbover = -0.5 ;
    if ( !model->HSMHV2_nover_Given   ) model->HSMHV2_nover   = 3e16 ;
    if ( !model->HSMHV2_novers_Given  ) model->HSMHV2_novers  = 1e17 ;
    if ( !model->HSMHV2_xwd_Given     ) model->HSMHV2_xwd     = 0.0 ;
    if ( !model->HSMHV2_xwdc_Given    ) model->HSMHV2_xwdc    = model->HSMHV2_xwd ;

    if ( !model->HSMHV2_xl_Given   ) model->HSMHV2_xl   = 0.0 ;
    if ( !model->HSMHV2_xw_Given   ) model->HSMHV2_xw   = 0.0 ;
    if ( !model->HSMHV2_saref_Given   ) model->HSMHV2_saref   = 1e-6 ;
    if ( !model->HSMHV2_sbref_Given   ) model->HSMHV2_sbref   = 1e-6 ;
    if ( !model->HSMHV2_ll_Given   ) model->HSMHV2_ll   = 0.0 ;
    if ( !model->HSMHV2_lld_Given  ) model->HSMHV2_lld  = 0.0 ;
    if ( !model->HSMHV2_lln_Given  ) model->HSMHV2_lln  = 0.0 ;
    if ( !model->HSMHV2_wl_Given   ) model->HSMHV2_wl   = 0.0 ;
    if ( !model->HSMHV2_wl1_Given  ) model->HSMHV2_wl1  = 0.0 ;
    if ( !model->HSMHV2_wl1p_Given ) model->HSMHV2_wl1p = 1.0 ;
    if ( !model->HSMHV2_wl2_Given  ) model->HSMHV2_wl2  = 0.0 ;
    if ( !model->HSMHV2_wl2p_Given ) model->HSMHV2_wl2p = 1.0 ;
    if ( !model->HSMHV2_wld_Given  ) model->HSMHV2_wld  = 0.0 ;
    if ( !model->HSMHV2_wln_Given  ) model->HSMHV2_wln  = 0.0 ;

    if ( !model->HSMHV2_rsh_Given  ) model->HSMHV2_rsh  = 0.0 ;
    if ( !model->HSMHV2_rshg_Given ) model->HSMHV2_rshg = 0.0 ;

    if ( !model->HSMHV2_xqy_Given    ) model->HSMHV2_xqy   = 0.0 ;
    if ( !model->HSMHV2_xqy1_Given   ) model->HSMHV2_xqy1  = 0.0 ;
    if ( !model->HSMHV2_xqy2_Given   ) model->HSMHV2_xqy2  = 2.0 ;
    if ( !model->HSMHV2_rs_Given     ) model->HSMHV2_rs    = 0.0 ;
    if ( !model->HSMHV2_rd_Given     ) model->HSMHV2_rd    = 0.0 ;
    if ( !model->HSMHV2_vfbc_Given   ) model->HSMHV2_vfbc  = -1.0 ;
    if ( !model->HSMHV2_vbi_Given    ) model->HSMHV2_vbi   = 1.1 ;
    if ( !model->HSMHV2_nsubc_Given  ) model->HSMHV2_nsubc = 3.0e17 ;
    if ( !model->HSMHV2_parl2_Given  ) model->HSMHV2_parl2 = 10.0e-9 ;
    if ( !model->HSMHV2_lp_Given     ) model->HSMHV2_lp    = 15e-9 ;
    if ( !model->HSMHV2_nsubp_Given  ) model->HSMHV2_nsubp = 1.0e18 ;
   
    if ( !model->HSMHV2_nsubp0_Given ) model->HSMHV2_nsubp0 = 0.0 ;
    if ( !model->HSMHV2_nsubwp_Given ) model->HSMHV2_nsubwp = 1.0 ;

    if ( !model->HSMHV2_scp1_Given  ) model->HSMHV2_scp1 = 0.0 ;
    if ( !model->HSMHV2_scp2_Given  ) model->HSMHV2_scp2 = 0.0 ;
    if ( !model->HSMHV2_scp3_Given  ) model->HSMHV2_scp3 = 0.0 ;
    if ( !model->HSMHV2_sc1_Given   ) model->HSMHV2_sc1  = 0.0 ;
    if ( !model->HSMHV2_sc2_Given   ) model->HSMHV2_sc2  = 0.0 ;
    if ( !model->HSMHV2_sc3_Given   ) model->HSMHV2_sc3  = 0.0 ;
    if ( !model->HSMHV2_sc4_Given   ) model->HSMHV2_sc4  = 0.0 ;
    if ( !model->HSMHV2_pgd1_Given  ) model->HSMHV2_pgd1 = 0.0 ;
    if ( !model->HSMHV2_pgd2_Given  ) model->HSMHV2_pgd2 = 1.0 ;
    if ( !model->HSMHV2_pgd4_Given  ) model->HSMHV2_pgd4 = 0.0 ;

    if ( !model->HSMHV2_ndep_Given   ) model->HSMHV2_ndep   = 1.0 ;
    if ( !model->HSMHV2_ndepl_Given  ) model->HSMHV2_ndepl  = 0.0 ;
    if ( !model->HSMHV2_ndeplp_Given ) model->HSMHV2_ndeplp = 1.0 ;
    if ( !model->HSMHV2_ninv_Given   ) model->HSMHV2_ninv   = 0.5 ;
    if ( !model->HSMHV2_muecb0_Given ) model->HSMHV2_muecb0 = 1.0e3 ;
    if ( !model->HSMHV2_muecb1_Given ) model->HSMHV2_muecb1 = 100.0 ;
    if ( !model->HSMHV2_mueph0_Given ) model->HSMHV2_mueph0 = 300.0e-3 ;
    if ( !model->HSMHV2_mueph1_Given ) {
      if (model->HSMHV2_type == NMOS) model->HSMHV2_mueph1 = 20.0e3 ;
      else model->HSMHV2_mueph1 = 9.0e3 ;
    }
    if ( !model->HSMHV2_muephw_Given ) model->HSMHV2_muephw = 0.0 ;
    if ( !model->HSMHV2_muepwp_Given ) model->HSMHV2_muepwp = 1.0 ;
    if ( !model->HSMHV2_muephl_Given ) model->HSMHV2_muephl = 0.0 ;
    if ( !model->HSMHV2_mueplp_Given ) model->HSMHV2_mueplp = 1.0 ;
    if ( !model->HSMHV2_muephs_Given ) model->HSMHV2_muephs = 0.0 ;
    if ( !model->HSMHV2_muepsp_Given ) model->HSMHV2_muepsp = 1.0 ;

    if ( !model->HSMHV2_vtmp_Given  ) model->HSMHV2_vtmp  = 0.0 ;

    if ( !model->HSMHV2_wvth0_Given ) model->HSMHV2_wvth0 = 0.0 ;

    if ( !model->HSMHV2_muesr0_Given ) model->HSMHV2_muesr0 = 2.0 ;
    if ( !model->HSMHV2_muesr1_Given ) model->HSMHV2_muesr1 = 6.0e14 ;
    if ( !model->HSMHV2_muesrl_Given ) model->HSMHV2_muesrl = 0.0 ;
    if ( !model->HSMHV2_muesrw_Given ) model->HSMHV2_muesrw = 0.0 ;
    if ( !model->HSMHV2_mueswp_Given ) model->HSMHV2_mueswp = 1.0 ;
    if ( !model->HSMHV2_mueslp_Given ) model->HSMHV2_mueslp = 1.0 ;

    if ( !model->HSMHV2_muetmp_Given  ) model->HSMHV2_muetmp = 1.5 ;

    if ( !model->HSMHV2_bb_Given ) {
      if (model->HSMHV2_type == NMOS) model->HSMHV2_bb = 2.0 ;
      else model->HSMHV2_bb = 1.0 ;
    }

    if ( !model->HSMHV2_sub1_Given  ) model->HSMHV2_sub1  = 10 ;
    if ( !model->HSMHV2_sub2_Given  ) model->HSMHV2_sub2  = 25 ;
    if ( !model->HSMHV2_svgs_Given  ) model->HSMHV2_svgs  = 0.8e0 ;
    if ( !model->HSMHV2_svbs_Given  ) model->HSMHV2_svbs  = 0.5e0 ;
    if ( !model->HSMHV2_svbsl_Given ) model->HSMHV2_svbsl = 0e0 ;
    if ( !model->HSMHV2_svds_Given  ) model->HSMHV2_svds  = 0.8e0 ;
    if ( !model->HSMHV2_slg_Given   ) model->HSMHV2_slg   = 30e-9 ;
    if ( !model->HSMHV2_sub1l_Given ) model->HSMHV2_sub1l = 2.5e-3 ;
    if ( !model->HSMHV2_sub2l_Given ) model->HSMHV2_sub2l = 2e-6 ;
    if ( !model->HSMHV2_fn1_Given   ) model->HSMHV2_fn1   = 50e0 ;
    if ( !model->HSMHV2_fn2_Given   ) model->HSMHV2_fn2   = 170e-6 ;
    if ( !model->HSMHV2_fn3_Given   ) model->HSMHV2_fn3   = 0e0 ;
    if ( !model->HSMHV2_fvbs_Given  ) model->HSMHV2_fvbs  = 12e-3 ;

    if ( !model->HSMHV2_svgsl_Given  ) model->HSMHV2_svgsl  = 0.0 ;
    if ( !model->HSMHV2_svgslp_Given ) model->HSMHV2_svgslp = 1.0 ;
    if ( !model->HSMHV2_svgswp_Given ) model->HSMHV2_svgswp = 1.0 ;
    if ( !model->HSMHV2_svgsw_Given  ) model->HSMHV2_svgsw  = 0.0 ;
    if ( !model->HSMHV2_svbslp_Given ) model->HSMHV2_svbslp = 1.0 ;
    if ( !model->HSMHV2_slgl_Given   ) model->HSMHV2_slgl   = 0.0 ;
    if ( !model->HSMHV2_slglp_Given  ) model->HSMHV2_slglp  = 1.0 ;
    if ( !model->HSMHV2_sub1lp_Given ) model->HSMHV2_sub1lp = 1.0 ; 

    if ( !model->HSMHV2_nsti_Given      ) model->HSMHV2_nsti      = 5.0e17 ;
    if ( !model->HSMHV2_wsti_Given      ) model->HSMHV2_wsti      = 0.0 ;
    if ( !model->HSMHV2_wstil_Given     ) model->HSMHV2_wstil     = 0.0 ;
    if ( !model->HSMHV2_wstilp_Given    ) model->HSMHV2_wstilp    = 1.0 ;
    if ( !model->HSMHV2_wstiw_Given     ) model->HSMHV2_wstiw     = 0.0 ;
    if ( !model->HSMHV2_wstiwp_Given    ) model->HSMHV2_wstiwp    = 1.0 ;
    if ( !model->HSMHV2_scsti1_Given    ) model->HSMHV2_scsti1    = 0.0 ;
    if ( !model->HSMHV2_scsti2_Given    ) model->HSMHV2_scsti2    = 0.0 ;
    if ( !model->HSMHV2_vthsti_Given    ) model->HSMHV2_vthsti    = 0.0 ;
    if ( !model->HSMHV2_vdsti_Given     ) model->HSMHV2_vdsti     = 0.0 ;
    if ( !model->HSMHV2_muesti1_Given   ) model->HSMHV2_muesti1   = 0.0 ;
    if ( !model->HSMHV2_muesti2_Given   ) model->HSMHV2_muesti2   = 0.0 ;
    if ( !model->HSMHV2_muesti3_Given   ) model->HSMHV2_muesti3   = 1.0 ;
    if ( !model->HSMHV2_nsubpsti1_Given ) model->HSMHV2_nsubpsti1 = 0.0 ;
    if ( !model->HSMHV2_nsubpsti2_Given ) model->HSMHV2_nsubpsti2 = 0.0 ;
    if ( !model->HSMHV2_nsubpsti3_Given ) model->HSMHV2_nsubpsti3 = 1.0 ;

    if ( !model->HSMHV2_lpext_Given ) model->HSMHV2_lpext = 1.0e-50 ;
    if ( !model->HSMHV2_npext_Given ) model->HSMHV2_npext = 5.0e17 ;
    if ( !model->HSMHV2_scp21_Given ) model->HSMHV2_scp21 = 0.0 ;
    if ( !model->HSMHV2_scp22_Given ) model->HSMHV2_scp22 = 0.0 ;
    if ( !model->HSMHV2_bs1_Given   ) model->HSMHV2_bs1   = 0.0 ;
    if ( !model->HSMHV2_bs2_Given   ) model->HSMHV2_bs2   = 0.9 ;

    if ( !model->HSMHV2_tpoly_Given ) model->HSMHV2_tpoly = 200e-9 ;
    if ( !model->HSMHV2_cgbo_Given  ) model->HSMHV2_cgbo  = 0.0 ;
    if ( !model->HSMHV2_js0_Given   ) model->HSMHV2_js0   = 0.5e-6 ;
    if ( !model->HSMHV2_js0sw_Given ) model->HSMHV2_js0sw = 0.0 ;
    if ( !model->HSMHV2_nj_Given    ) model->HSMHV2_nj    = 1.0 ;
    if ( !model->HSMHV2_njsw_Given  ) model->HSMHV2_njsw  = 1.0 ;
    if ( !model->HSMHV2_xti_Given   ) model->HSMHV2_xti   = 2.0 ;
    if ( !model->HSMHV2_cj_Given    ) model->HSMHV2_cj    = 5.0e-04 ;
    if ( !model->HSMHV2_cjsw_Given  ) model->HSMHV2_cjsw  = 5.0e-10 ;
    if ( !model->HSMHV2_cjswg_Given ) model->HSMHV2_cjswg = 5.0e-10 ;
    if ( !model->HSMHV2_mj_Given    ) model->HSMHV2_mj    = 0.5e0 ;
    if ( !model->HSMHV2_mjsw_Given  ) model->HSMHV2_mjsw  = 0.33e0 ;
    if ( !model->HSMHV2_mjswg_Given ) model->HSMHV2_mjswg = 0.33e0 ;
    if ( !model->HSMHV2_pb_Given    ) model->HSMHV2_pb    = 1.0e0 ;
    if ( !model->HSMHV2_pbsw_Given  ) model->HSMHV2_pbsw  = 1.0e0 ;
    if ( !model->HSMHV2_pbswg_Given ) model->HSMHV2_pbswg = 1.0e0 ;
    if ( !model->HSMHV2_xti2_Given  ) model->HSMHV2_xti2  = 0.0e0 ;
    if ( !model->HSMHV2_cisb_Given  ) model->HSMHV2_cisb  = 0.0e0 ;
    if ( !model->HSMHV2_cvb_Given   ) model->HSMHV2_cvb   = 0.0e0 ;
    if ( !model->HSMHV2_ctemp_Given ) model->HSMHV2_ctemp = 0.0e0 ;
    if ( !model->HSMHV2_cisbk_Given ) model->HSMHV2_cisbk = 0.0e0 ;
    if ( !model->HSMHV2_divx_Given  ) model->HSMHV2_divx  = 0.0e0 ;

    if ( !model->HSMHV2_clm1_Given  ) model->HSMHV2_clm1 = 50e-3 ;
    if ( !model->HSMHV2_clm2_Given  ) model->HSMHV2_clm2 = 2.0 ;
    if ( !model->HSMHV2_clm3_Given  ) model->HSMHV2_clm3 = 1.0 ;
    if ( !model->HSMHV2_clm5_Given   ) model->HSMHV2_clm5   = 1.0 ;
    if ( !model->HSMHV2_clm6_Given   ) model->HSMHV2_clm6   = 0.0 ;
    if ( !model->HSMHV2_vover_Given  ) model->HSMHV2_vover  = 0.3 ;
    if ( !model->HSMHV2_voverp_Given ) model->HSMHV2_voverp = 0.3 ;
    if ( !model->HSMHV2_wfc_Given    ) model->HSMHV2_wfc    = 0.0 ;
    if ( !model->HSMHV2_nsubcw_Given    ) model->HSMHV2_nsubcw  = 0.0 ;
    if ( !model->HSMHV2_nsubcwp_Given   ) model->HSMHV2_nsubcwp = 1.0 ;
    if ( !model->HSMHV2_qme1_Given   ) model->HSMHV2_qme1   = 0.0 ;
    if ( !model->HSMHV2_qme2_Given   ) model->HSMHV2_qme2   = 2.0 ;
    if ( !model->HSMHV2_qme3_Given   ) model->HSMHV2_qme3   = 0.0 ;

    if ( !model->HSMHV2_vovers_Given  ) model->HSMHV2_vovers  = 0.0 ;
    if ( !model->HSMHV2_voversp_Given ) model->HSMHV2_voversp = 0.0 ;

    if ( !model->HSMHV2_gidl1_Given ) model->HSMHV2_gidl1 = 2e0 ;
    if ( !model->HSMHV2_gidl2_Given ) model->HSMHV2_gidl2 = 3e7 ;
    if ( !model->HSMHV2_gidl3_Given ) model->HSMHV2_gidl3 = 0.9e0 ;
    if ( !model->HSMHV2_gidl4_Given ) model->HSMHV2_gidl4 = 0.0 ;
    if ( !model->HSMHV2_gidl5_Given ) model->HSMHV2_gidl5 = 0.2e0 ;

    if ( !model->HSMHV2_gleak1_Given ) model->HSMHV2_gleak1 = 50e0 ;
    if ( !model->HSMHV2_gleak2_Given ) model->HSMHV2_gleak2 = 10e6 ;
    if ( !model->HSMHV2_gleak3_Given ) model->HSMHV2_gleak3 = 60e-3 ;
    if ( !model->HSMHV2_gleak4_Given ) model->HSMHV2_gleak4 = 4e0 ;
    if ( !model->HSMHV2_gleak5_Given ) model->HSMHV2_gleak5 = 7.5e3 ;
    if ( !model->HSMHV2_gleak6_Given ) model->HSMHV2_gleak6 = 250e-3 ;
    if ( !model->HSMHV2_gleak7_Given ) model->HSMHV2_gleak7 = 1e-6 ;

    if ( !model->HSMHV2_glpart1_Given ) model->HSMHV2_glpart1 = 0.5 ;
    if ( !model->HSMHV2_glksd1_Given  ) model->HSMHV2_glksd1  = 1.0e-15 ;
    if ( !model->HSMHV2_glksd2_Given  ) model->HSMHV2_glksd2  = 1e3 ;
    if ( !model->HSMHV2_glksd3_Given  ) model->HSMHV2_glksd3  = -1e3 ;
    if ( !model->HSMHV2_glkb1_Given   ) model->HSMHV2_glkb1   = 5e-16 ;
    if ( !model->HSMHV2_glkb2_Given   ) model->HSMHV2_glkb2   = 1e0 ;
    if ( !model->HSMHV2_glkb3_Given   ) model->HSMHV2_glkb3   = 0e0 ;
    if ( !model->HSMHV2_egig_Given    ) model->HSMHV2_egig    = 0e0 ;
    if ( !model->HSMHV2_igtemp2_Given ) model->HSMHV2_igtemp2 = 0e0 ;
    if ( !model->HSMHV2_igtemp3_Given ) model->HSMHV2_igtemp3 = 0e0 ;
    if ( !model->HSMHV2_vzadd0_Given  ) model->HSMHV2_vzadd0  = 10.0e-3 ;
    if ( !model->HSMHV2_pzadd0_Given  ) model->HSMHV2_pzadd0  = 5.0e-3 ;
    if ( !model->HSMHV2_nftrp_Given   ) model->HSMHV2_nftrp   = 10e9 ;
    if ( !model->HSMHV2_nfalp_Given   ) model->HSMHV2_nfalp   = 1.0e-19 ;
    if ( !model->HSMHV2_cit_Given     ) model->HSMHV2_cit     = 0e0 ;
    if ( !model->HSMHV2_falph_Given   ) model->HSMHV2_falph   = 1.0 ;

    if ( !model->HSMHV2_kappa_Given ) model->HSMHV2_kappa = 3.90e0 ;
    if ( !model->HSMHV2_cgso_Given  ) model->HSMHV2_cgso  = 0.0 ;
    if ( !model->HSMHV2_cgdo_Given  ) model->HSMHV2_cgdo  = 0.0 ;


    if ( !model->HSMHV2_vdiffj_Given ) model->HSMHV2_vdiffj = 0.6e-3 ;
    if ( !model->HSMHV2_dly1_Given   ) model->HSMHV2_dly1   = 100.0e-12 ;
    if ( !model->HSMHV2_dly2_Given   ) model->HSMHV2_dly2   = 0.7e0 ;
    if ( !model->HSMHV2_dly3_Given   ) model->HSMHV2_dly3   = 0.8e-6 ;
    if ( !model->HSMHV2_tnom_Given   ) model->HSMHV2_tnom   = 27.0 ; /* [C] */

    if ( !model->HSMHV2_ovslp_Given  ) model->HSMHV2_ovslp = 2.1e-7 ;
    if ( !model->HSMHV2_ovmag_Given  ) model->HSMHV2_ovmag = 0.6 ;

    if ( !model->HSMHV2_gbmin_Given  ) model->HSMHV2_gbmin = 1.0e-12; /* in mho */
    if ( !model->HSMHV2_rbpb_Given   ) model->HSMHV2_rbpb  = 50.0e0 ;
    if ( !model->HSMHV2_rbpd_Given   ) model->HSMHV2_rbpd  = 50.0e0 ;
    if ( !model->HSMHV2_rbps_Given   ) model->HSMHV2_rbps  = 50.0e0 ;
    if ( !model->HSMHV2_rbdb_Given   ) model->HSMHV2_rbdb  = 50.0e0 ; /* not used in this version */
    if ( !model->HSMHV2_rbsb_Given   ) model->HSMHV2_rbsb  = 50.0e0 ; /* not used in this version */

    if ( !model->HSMHV2_ibpc1_Given  ) model->HSMHV2_ibpc1 = 0.0 ;
    if ( !model->HSMHV2_ibpc1l_Given  ) model->HSMHV2_ibpc1l = 0.0 ;
    if ( !model->HSMHV2_ibpc1lp_Given  ) model->HSMHV2_ibpc1lp = -1.0 ;
    if ( !model->HSMHV2_ibpc2_Given  ) model->HSMHV2_ibpc2 = 0.0 ;

    if ( !model->HSMHV2_mphdfm_Given ) model->HSMHV2_mphdfm = -0.3 ;

    if ( !model->HSMHV2_ptl_Given  ) model->HSMHV2_ptl  = 0.0 ;
    if ( !model->HSMHV2_ptp_Given  ) model->HSMHV2_ptp  = 3.5 ;
    if ( !model->HSMHV2_pt2_Given  ) model->HSMHV2_pt2  = 0.0 ;
    if ( !model->HSMHV2_ptlp_Given ) model->HSMHV2_ptlp = 1.0 ;
    if ( !model->HSMHV2_gdl_Given  ) model->HSMHV2_gdl  = 0.0 ;
    if ( !model->HSMHV2_gdlp_Given ) model->HSMHV2_gdlp = 0.0 ;

    if ( !model->HSMHV2_gdld_Given ) model->HSMHV2_gdld = 0.0 ;
    if ( !model->HSMHV2_pt4_Given  ) model->HSMHV2_pt4  = 0.0 ;
    if ( !model->HSMHV2_pt4p_Given ) model->HSMHV2_pt4p = 1.0 ;
    if ( !model->HSMHV2_rdvg11_Given ) model->HSMHV2_rdvg11  = 0.0 ;
    if ( !model->HSMHV2_rdvg12_Given ) model->HSMHV2_rdvg12  = 100.0 ;
    if ( !model->HSMHV2_rth0_Given   ) model->HSMHV2_rth0    = 0.1 ;    /* Self-heating model */
    if ( !model->HSMHV2_cth0_Given   ) model->HSMHV2_cth0    = 1.0e-7 ; /* Self-heating model */
    if ( !model->HSMHV2_powrat_Given ) model->HSMHV2_powrat  = 1.0 ;    /* Self-heating model */

    if ( !model->HSMHV2_tcjbd_Given    ) model->HSMHV2_tcjbd    = 0.0 ; /* Self-heating model */    
    if ( !model->HSMHV2_tcjbs_Given    ) model->HSMHV2_tcjbs    = 0.0 ; /* Self-heating model */    
    if ( !model->HSMHV2_tcjbdsw_Given  ) model->HSMHV2_tcjbdsw  = 0.0 ; /* Self-heating model */    
    if ( !model->HSMHV2_tcjbssw_Given  ) model->HSMHV2_tcjbssw  = 0.0 ; /* Self-heating model */    
    if ( !model->HSMHV2_tcjbdswg_Given ) model->HSMHV2_tcjbdswg = 0.0 ; /* Self-heating model */    
    if ( !model->HSMHV2_tcjbsswg_Given ) model->HSMHV2_tcjbsswg = 0.0 ; /* Self-heating model */

                                      /* value reset to switch off NQS for QbdLD:           */
                                        model->HSMHV2_dlyov    = 0.0 ; /* 1.0e3 ;            */
    if ( !model->HSMHV2_qdftvd_Given   ) model->HSMHV2_qdftvd   = 1.0 ;
    if ( !model->HSMHV2_xldld_Given    ) model->HSMHV2_xldld    = 1.0e-6 ;
    if ( !model->HSMHV2_xwdld_Given    ) model->HSMHV2_xwdld    = model->HSMHV2_xwd ;
    if ( !model->HSMHV2_rdvd_Given     ) model->HSMHV2_rdvd     = 7.0e-2 ;

    if ( !model->HSMHV2_rd20_Given    ) model->HSMHV2_rd20    = 0.0 ;
    if ( !model->HSMHV2_rd21_Given    ) model->HSMHV2_rd21    = 1.0 ;
    if ( !model->HSMHV2_rd22_Given    ) model->HSMHV2_rd22    = 0.0 ;
    if ( !model->HSMHV2_rd22d_Given   ) model->HSMHV2_rd22d   = 0.0 ;
    if ( !model->HSMHV2_rd23_Given    ) model->HSMHV2_rd23    = 5e-3 ;
    if ( !model->HSMHV2_rd24_Given    ) model->HSMHV2_rd24    = 0.0 ;
    if ( !model->HSMHV2_rd25_Given    ) model->HSMHV2_rd25    = 0.0 ;

    if ( !model->HSMHV2_rdvdl_Given    ) model->HSMHV2_rdvdl     = 0.0 ;
    if ( !model->HSMHV2_rdvdlp_Given   ) model->HSMHV2_rdvdlp    = 1.0 ;
    if ( !model->HSMHV2_rdvds_Given    ) model->HSMHV2_rdvds     = 0.0 ;
    if ( !model->HSMHV2_rdvdsp_Given   ) model->HSMHV2_rdvdsp    = 1.0 ;
    if ( !model->HSMHV2_rd23l_Given    ) model->HSMHV2_rd23l     = 0.0 ;
    if ( !model->HSMHV2_rd23lp_Given   ) model->HSMHV2_rd23lp    = 1.0 ;
    if ( !model->HSMHV2_rd23s_Given    ) model->HSMHV2_rd23s     = 0.0 ;
    if ( !model->HSMHV2_rd23sp_Given   ) model->HSMHV2_rd23sp    = 1.0 ;
    if ( !model->HSMHV2_rds_Given      ) model->HSMHV2_rds       = 0.0 ;
    if ( !model->HSMHV2_rdsp_Given     ) model->HSMHV2_rdsp      = 1.0 ;
    if ( !model->HSMHV2_rdtemp1_Given  ) model->HSMHV2_rdtemp1   = 0.0 ;
    if ( !model->HSMHV2_rdtemp2_Given  ) model->HSMHV2_rdtemp2   = 0.0 ;
                                        model->HSMHV2_rth0r     = 0.0 ; /* not used in this version */
    if ( !model->HSMHV2_rdvdtemp1_Given) model->HSMHV2_rdvdtemp1 = 0.0 ;
    if ( !model->HSMHV2_rdvdtemp2_Given) model->HSMHV2_rdvdtemp2 = 0.0 ;
    if ( !model->HSMHV2_rth0w_Given    ) model->HSMHV2_rth0w     = 0.0 ;
    if ( !model->HSMHV2_rth0wp_Given   ) model->HSMHV2_rth0wp    = 1.0 ;

    if ( !model->HSMHV2_cvdsover_Given ) model->HSMHV2_cvdsover  = 0.0 ;

    if ( !model->HSMHV2_ninvd_Given    ) model->HSMHV2_ninvd     = 0.0 ;
    if ( !model->HSMHV2_ninvdw_Given   ) model->HSMHV2_ninvdw    = 0.0 ;
    if ( !model->HSMHV2_ninvdwp_Given  ) model->HSMHV2_ninvdwp   = 1.0 ;
    if ( !model->HSMHV2_ninvdt1_Given  ) model->HSMHV2_ninvdt1   = 0.0 ;
    if ( !model->HSMHV2_ninvdt2_Given  ) model->HSMHV2_ninvdt2   = 0.0 ;
    if ( !model->HSMHV2_vbsmin_Given   ) model->HSMHV2_vbsmin    = -10.5 ;
    if ( !model->HSMHV2_rdvb_Given     ) model->HSMHV2_rdvb      = 0.0 ;
    if ( !model->HSMHV2_rth0nf_Given   ) model->HSMHV2_rth0nf    = 0.0 ;

    if ( !model->HSMHV2_rthtemp1_Given   ) model->HSMHV2_rthtemp1    = 0.0 ;
    if ( !model->HSMHV2_rthtemp2_Given   ) model->HSMHV2_rthtemp2    = 0.0 ;
    if ( !model->HSMHV2_prattemp1_Given  ) model->HSMHV2_prattemp1   = 0.0 ;
    if ( !model->HSMHV2_prattemp2_Given  ) model->HSMHV2_prattemp2   = 0.0 ;

    if ( !model->HSMHV2_rdvsub_Given   ) model->HSMHV2_rdvsub     = 1.0 ;    /* [-] substrate effect */
    if ( !model->HSMHV2_rdvdsub_Given  ) model->HSMHV2_rdvdsub    = 0.3 ;    /* [-] substrate effect */
    if ( !model->HSMHV2_ddrift_Given   ) model->HSMHV2_ddrift     = 1.0e-6 ; /* [m] substrate effect */
    if ( !model->HSMHV2_vbisub_Given   ) model->HSMHV2_vbisub     = 0.7 ;    /* [V] substrate effect */
    if ( !model->HSMHV2_nsubsub_Given  ) model->HSMHV2_nsubsub    = 1.0e15 ; /* [cm^-3] substrate effect */

    if ( !model->HSMHV2_rdrmue_Given    ) model->HSMHV2_rdrmue    = 1.0e3 ; 
    if ( !model->HSMHV2_rdrvmax_Given   ) model->HSMHV2_rdrvmax   = 3.0e7 ; 
    if ( !model->HSMHV2_rdrmuetmp_Given ) model->HSMHV2_rdrmuetmp = 0.0 ; 

    if ( !model->HSMHV2_ndepm_Given )     model->HSMHV2_ndepm = 1e17 ;
    if ( !model->HSMHV2_tndep_Given )     model->HSMHV2_tndep = 0.2e-6 ;
    if ( !model->HSMHV2_depmue0_Given ) model->HSMHV2_depmue0 = 1.0e3 ;
    if ( !model->HSMHV2_depmue1_Given ) model->HSMHV2_depmue1 = 0.0 ;
    if ( !model->HSMHV2_depmueback0_Given ) model->HSMHV2_depmueback0 = 1.0e2 ;
    if ( !model->HSMHV2_depmueback1_Given ) model->HSMHV2_depmueback1 = 0.0 ;
    if ( !model->HSMHV2_depleak_Given ) model->HSMHV2_depleak = 0.5 ;
    if ( !model->HSMHV2_depeta_Given ) model->HSMHV2_depeta = 0.0 ;
    if ( !model->HSMHV2_depvmax_Given ) model->HSMHV2_depvmax = 3.0e7 ;
    if ( !model->HSMHV2_depvdsef1_Given ) model->HSMHV2_depvdsef1 = 2.0 ;
    if ( !model->HSMHV2_depvdsef2_Given ) model->HSMHV2_depvdsef2 = 0.5 ;
    if ( !model->HSMHV2_depmueph0_Given ) model->HSMHV2_depmueph0 = 0.3 ;
    if ( !model->HSMHV2_depmueph1_Given ) model->HSMHV2_depmueph1 = 5.0e3 ;
    if ( !model->HSMHV2_depbb_Given ) model->HSMHV2_depbb = 1.0 ;
    if ( !model->HSMHV2_depvtmp_Given ) model->HSMHV2_depvtmp = 0.0 ;
    if ( !model->HSMHV2_depmuetmp_Given ) model->HSMHV2_depmuetmp = 1.5 ;

    if ( model->HSMHV2_codep ) {
      if ( !model->HSMHV2_copprv_Given ) model->HSMHV2_copprv = 0 ;
      if ( !model->HSMHV2_vfbc_Given   ) model->HSMHV2_vfbc   = -0.2 ;
      if ( !model->HSMHV2_nsubc_Given  ) model->HSMHV2_nsubc  = 5.0e16 ;
      if ( !model->HSMHV2_lp_Given     ) model->HSMHV2_lp     = 0.0 ;
      if ( !model->HSMHV2_nsubp_Given  ) model->HSMHV2_nsubp  = 1.0e17 ;
      if ( !model->HSMHV2_muesr1_Given ) model->HSMHV2_muesr1 = 5.0e15 ;
    }

    if ( !model->HSMHV2_isbreak_Given ) model->HSMHV2_isbreak = 1.0e-12 ;
    if ( !model->HSMHV2_rwell_Given ) model->HSMHV2_rwell = 1.0e3 ;

    if ( !model->HSMHV2_rdrvtmp_Given   ) model->HSMHV2_rdrvtmp   = 0.0 ; 
    if ( !model->HSMHV2_rdrdjunc_Given  ) model->HSMHV2_rdrdjunc  = 1.0e-6 ; 
    if ( !model->HSMHV2_rdrcx_Given     ) model->HSMHV2_rdrcx     = 0.0 ; 
    if ( !model->HSMHV2_rdrcar_Given    ) model->HSMHV2_rdrcar    = 1.0e-8 ;
    if ( !model->HSMHV2_rdrdl1_Given    ) model->HSMHV2_rdrdl1    = 0.0 ; 
    if ( !model->HSMHV2_rdrdl2_Given    ) model->HSMHV2_rdrdl2    = 0.0 ; 
    if ( !model->HSMHV2_rdrvmaxw_Given  ) model->HSMHV2_rdrvmaxw  = 0.0 ; 
    if ( !model->HSMHV2_rdrvmaxwp_Given ) model->HSMHV2_rdrvmaxwp = 1.0 ; 
    if ( !model->HSMHV2_rdrvmaxl_Given  ) model->HSMHV2_rdrvmaxl  = 0.0 ; 
    if ( !model->HSMHV2_rdrvmaxlp_Given ) model->HSMHV2_rdrvmaxlp = 1.0 ; 
    if ( !model->HSMHV2_rdrmuel_Given   ) model->HSMHV2_rdrmuel   = 0.0 ; 
    if ( !model->HSMHV2_rdrmuelp_Given  ) model->HSMHV2_rdrmuelp  = 1.0 ; 
    if ( !model->HSMHV2_rdrqover_Given  ) model->HSMHV2_rdrqover  = 1E5 ; 
    if ( !model->HSMHV2_qovadd_Given    ) model->HSMHV2_qovadd    = 0.0 ;
    if ( !model->HSMHV2_js0d_Given      ) model->HSMHV2_js0d      = model->HSMHV2_js0 ;
    if ( !model->HSMHV2_js0swd_Given    ) model->HSMHV2_js0swd    = model->HSMHV2_js0sw ;
    if ( !model->HSMHV2_njd_Given       ) model->HSMHV2_njd       = model->HSMHV2_nj ;
    if ( !model->HSMHV2_njswd_Given     ) model->HSMHV2_njswd     = model->HSMHV2_njsw ;
    if ( !model->HSMHV2_xtid_Given      ) model->HSMHV2_xtid      = model->HSMHV2_xti ;
    if ( !model->HSMHV2_cjd_Given       ) model->HSMHV2_cjd       = model->HSMHV2_cj ;
    if ( !model->HSMHV2_cjswd_Given     ) model->HSMHV2_cjswd     = model->HSMHV2_cjsw ;
    if ( !model->HSMHV2_cjswgd_Given    ) model->HSMHV2_cjswgd    = model->HSMHV2_cjswg ;
    if ( !model->HSMHV2_mjd_Given       ) model->HSMHV2_mjd       = model->HSMHV2_mj ;
    if ( !model->HSMHV2_mjswd_Given     ) model->HSMHV2_mjswd     = model->HSMHV2_mjsw ;
    if ( !model->HSMHV2_mjswgd_Given    ) model->HSMHV2_mjswgd    = model->HSMHV2_mjswg ;
    if ( !model->HSMHV2_pbd_Given       ) model->HSMHV2_pbd       = model->HSMHV2_pb ;
    if ( !model->HSMHV2_pbswd_Given     ) model->HSMHV2_pbswd     = model->HSMHV2_pbsw ;
    if ( !model->HSMHV2_pbswgd_Given    ) model->HSMHV2_pbswgd    = model->HSMHV2_pbswg ;
    if ( !model->HSMHV2_xti2d_Given     ) model->HSMHV2_xti2d     = model->HSMHV2_xti2 ;
    if ( !model->HSMHV2_cisbd_Given     ) model->HSMHV2_cisbd     = model->HSMHV2_cisb ;
    if ( !model->HSMHV2_cvbd_Given      ) model->HSMHV2_cvbd      = model->HSMHV2_cvb ;
    if ( !model->HSMHV2_ctempd_Given    ) model->HSMHV2_ctempd    = model->HSMHV2_ctemp ;
    if ( !model->HSMHV2_cisbkd_Given    ) model->HSMHV2_cisbkd    = model->HSMHV2_cisbk ;
    if ( !model->HSMHV2_divxd_Given     ) model->HSMHV2_divxd     = model->HSMHV2_divx ;
    if ( !model->HSMHV2_vdiffjd_Given   ) model->HSMHV2_vdiffjd   = model->HSMHV2_vdiffj ;
    if ( !model->HSMHV2_js0s_Given      ) model->HSMHV2_js0s      = model->HSMHV2_js0d ;
    if ( !model->HSMHV2_js0sws_Given    ) model->HSMHV2_js0sws    = model->HSMHV2_js0swd ;
    if ( !model->HSMHV2_njs_Given       ) model->HSMHV2_njs       = model->HSMHV2_njd ;
    if ( !model->HSMHV2_njsws_Given     ) model->HSMHV2_njsws     = model->HSMHV2_njswd ;
    if ( !model->HSMHV2_xtis_Given      ) model->HSMHV2_xtis      = model->HSMHV2_xtid ;
    if ( !model->HSMHV2_cjs_Given       ) model->HSMHV2_cjs       = model->HSMHV2_cjd ;
    if ( !model->HSMHV2_cjsws_Given     ) model->HSMHV2_cjsws     = model->HSMHV2_cjswd ;
    if ( !model->HSMHV2_cjswgs_Given    ) model->HSMHV2_cjswgs    = model->HSMHV2_cjswgd ;
    if ( !model->HSMHV2_mjs_Given       ) model->HSMHV2_mjs       = model->HSMHV2_mjd ;
    if ( !model->HSMHV2_mjsws_Given     ) model->HSMHV2_mjsws     = model->HSMHV2_mjswd ;
    if ( !model->HSMHV2_mjswgs_Given    ) model->HSMHV2_mjswgs    = model->HSMHV2_mjswgd ;
    if ( !model->HSMHV2_pbs_Given       ) model->HSMHV2_pbs       = model->HSMHV2_pbd ;
    if ( !model->HSMHV2_pbsws_Given     ) model->HSMHV2_pbsws     = model->HSMHV2_pbswd ;
    if ( !model->HSMHV2_pbswgs_Given    ) model->HSMHV2_pbswgs    = model->HSMHV2_pbswgd ;
    if ( !model->HSMHV2_xti2s_Given     ) model->HSMHV2_xti2s     = model->HSMHV2_xti2d ;
    if ( !model->HSMHV2_cisbs_Given     ) model->HSMHV2_cisbs     = model->HSMHV2_cisbd ;
    if ( !model->HSMHV2_cvbs_Given      ) model->HSMHV2_cvbs      = model->HSMHV2_cvbd ;
    if ( !model->HSMHV2_ctemps_Given    ) model->HSMHV2_ctemps    = model->HSMHV2_ctempd ;
    if ( !model->HSMHV2_cisbks_Given    ) model->HSMHV2_cisbks    = model->HSMHV2_cisbkd ;
    if ( !model->HSMHV2_divxs_Given     ) model->HSMHV2_divxs     = model->HSMHV2_divxd ;
    if ( !model->HSMHV2_vdiffjs_Given   ) model->HSMHV2_vdiffjs   = model->HSMHV2_vdiffjd ;
    if ( !model->HSMHV2_shemax_Given    ) model->HSMHV2_shemax    = 500 ;
    if ( !model->HSMHV2_vgsmin_Given    ) model->HSMHV2_vgsmin    = -100 * model->HSMHV2_type ;
    if ( !model->HSMHV2_gdsleak_Given   ) model->HSMHV2_gdsleak   = 0.0 ;
    if ( !model->HSMHV2_rdrbb_Given     ) model->HSMHV2_rdrbb     = 1 ;
    if ( !model->HSMHV2_rdrbbtmp_Given  ) model->HSMHV2_rdrbbtmp     = 0 ;

    /* binning parameters */
    if ( !model->HSMHV2_lmin_Given ) model->HSMHV2_lmin = 0.0 ;
    if ( !model->HSMHV2_lmax_Given ) model->HSMHV2_lmax = 1.0 ;
    if ( !model->HSMHV2_wmin_Given ) model->HSMHV2_wmin = 0.0 ;
    if ( !model->HSMHV2_wmax_Given ) model->HSMHV2_wmax = 1.0 ;
    if ( !model->HSMHV2_lbinn_Given ) model->HSMHV2_lbinn = 1.0 ;
    if ( !model->HSMHV2_wbinn_Given ) model->HSMHV2_wbinn = 1.0 ;

    /* Length dependence */
    if ( !model->HSMHV2_lvmax_Given ) model->HSMHV2_lvmax = 0.0 ;
    if ( !model->HSMHV2_lbgtmp1_Given ) model->HSMHV2_lbgtmp1 = 0.0 ;
    if ( !model->HSMHV2_lbgtmp2_Given ) model->HSMHV2_lbgtmp2 = 0.0 ;
    if ( !model->HSMHV2_leg0_Given ) model->HSMHV2_leg0 = 0.0 ;
    if ( !model->HSMHV2_lvfbover_Given ) model->HSMHV2_lvfbover = 0.0 ;
    if ( !model->HSMHV2_lnover_Given ) model->HSMHV2_lnover = 0.0 ;
    if ( !model->HSMHV2_lnovers_Given ) model->HSMHV2_lnovers = 0.0 ;
    if ( !model->HSMHV2_lwl2_Given ) model->HSMHV2_lwl2 = 0.0 ;
    if ( !model->HSMHV2_lvfbc_Given ) model->HSMHV2_lvfbc = 0.0 ;
    if ( !model->HSMHV2_lnsubc_Given ) model->HSMHV2_lnsubc = 0.0 ;
    if ( !model->HSMHV2_lnsubp_Given ) model->HSMHV2_lnsubp = 0.0 ;
    if ( !model->HSMHV2_lscp1_Given ) model->HSMHV2_lscp1 = 0.0 ;
    if ( !model->HSMHV2_lscp2_Given ) model->HSMHV2_lscp2 = 0.0 ;
    if ( !model->HSMHV2_lscp3_Given ) model->HSMHV2_lscp3 = 0.0 ;
    if ( !model->HSMHV2_lsc1_Given ) model->HSMHV2_lsc1 = 0.0 ;
    if ( !model->HSMHV2_lsc2_Given ) model->HSMHV2_lsc2 = 0.0 ;
    if ( !model->HSMHV2_lsc3_Given ) model->HSMHV2_lsc3 = 0.0 ;
    if ( !model->HSMHV2_lpgd1_Given ) model->HSMHV2_lpgd1 = 0.0 ;
    if ( !model->HSMHV2_lndep_Given ) model->HSMHV2_lndep = 0.0 ;
    if ( !model->HSMHV2_lninv_Given ) model->HSMHV2_lninv = 0.0 ;
    if ( !model->HSMHV2_lmuecb0_Given ) model->HSMHV2_lmuecb0 = 0.0 ;
    if ( !model->HSMHV2_lmuecb1_Given ) model->HSMHV2_lmuecb1 = 0.0 ;
    if ( !model->HSMHV2_lmueph1_Given ) model->HSMHV2_lmueph1 = 0.0 ;
    if ( !model->HSMHV2_lvtmp_Given ) model->HSMHV2_lvtmp = 0.0 ;
    if ( !model->HSMHV2_lwvth0_Given ) model->HSMHV2_lwvth0 = 0.0 ;
    if ( !model->HSMHV2_lmuesr1_Given ) model->HSMHV2_lmuesr1 = 0.0 ;
    if ( !model->HSMHV2_lmuetmp_Given ) model->HSMHV2_lmuetmp = 0.0 ;
    if ( !model->HSMHV2_lsub1_Given ) model->HSMHV2_lsub1 = 0.0 ;
    if ( !model->HSMHV2_lsub2_Given ) model->HSMHV2_lsub2 = 0.0 ;
    if ( !model->HSMHV2_lsvds_Given ) model->HSMHV2_lsvds = 0.0 ;
    if ( !model->HSMHV2_lsvbs_Given ) model->HSMHV2_lsvbs = 0.0 ;
    if ( !model->HSMHV2_lsvgs_Given ) model->HSMHV2_lsvgs = 0.0 ;
    if ( !model->HSMHV2_lfn1_Given ) model->HSMHV2_lfn1 = 0.0 ;
    if ( !model->HSMHV2_lfn2_Given ) model->HSMHV2_lfn2 = 0.0 ;
    if ( !model->HSMHV2_lfn3_Given ) model->HSMHV2_lfn3 = 0.0 ;
    if ( !model->HSMHV2_lfvbs_Given ) model->HSMHV2_lfvbs = 0.0 ;
    if ( !model->HSMHV2_lnsti_Given ) model->HSMHV2_lnsti = 0.0 ;
    if ( !model->HSMHV2_lwsti_Given ) model->HSMHV2_lwsti = 0.0 ;
    if ( !model->HSMHV2_lscsti1_Given ) model->HSMHV2_lscsti1 = 0.0 ;
    if ( !model->HSMHV2_lscsti2_Given ) model->HSMHV2_lscsti2 = 0.0 ;
    if ( !model->HSMHV2_lvthsti_Given ) model->HSMHV2_lvthsti = 0.0 ;
    if ( !model->HSMHV2_lmuesti1_Given ) model->HSMHV2_lmuesti1 = 0.0 ;
    if ( !model->HSMHV2_lmuesti2_Given ) model->HSMHV2_lmuesti2 = 0.0 ;
    if ( !model->HSMHV2_lmuesti3_Given ) model->HSMHV2_lmuesti3 = 0.0 ;
    if ( !model->HSMHV2_lnsubpsti1_Given ) model->HSMHV2_lnsubpsti1 = 0.0 ;
    if ( !model->HSMHV2_lnsubpsti2_Given ) model->HSMHV2_lnsubpsti2 = 0.0 ;
    if ( !model->HSMHV2_lnsubpsti3_Given ) model->HSMHV2_lnsubpsti3 = 0.0 ;
    if ( !model->HSMHV2_lcgso_Given ) model->HSMHV2_lcgso = 0.0 ;
    if ( !model->HSMHV2_lcgdo_Given ) model->HSMHV2_lcgdo = 0.0 ;
    if ( !model->HSMHV2_ljs0_Given ) model->HSMHV2_ljs0 = 0.0 ;
    if ( !model->HSMHV2_ljs0sw_Given ) model->HSMHV2_ljs0sw = 0.0 ;
    if ( !model->HSMHV2_lnj_Given ) model->HSMHV2_lnj = 0.0 ;
    if ( !model->HSMHV2_lcisbk_Given ) model->HSMHV2_lcisbk = 0.0 ;
    if ( !model->HSMHV2_lclm1_Given ) model->HSMHV2_lclm1 = 0.0 ;
    if ( !model->HSMHV2_lclm2_Given ) model->HSMHV2_lclm2 = 0.0 ;
    if ( !model->HSMHV2_lclm3_Given ) model->HSMHV2_lclm3 = 0.0 ;
    if ( !model->HSMHV2_lwfc_Given ) model->HSMHV2_lwfc = 0.0 ;
    if ( !model->HSMHV2_lgidl1_Given ) model->HSMHV2_lgidl1 = 0.0 ;
    if ( !model->HSMHV2_lgidl2_Given ) model->HSMHV2_lgidl2 = 0.0 ;
    if ( !model->HSMHV2_lgleak1_Given ) model->HSMHV2_lgleak1 = 0.0 ;
    if ( !model->HSMHV2_lgleak2_Given ) model->HSMHV2_lgleak2 = 0.0 ;
    if ( !model->HSMHV2_lgleak3_Given ) model->HSMHV2_lgleak3 = 0.0 ;
    if ( !model->HSMHV2_lgleak6_Given ) model->HSMHV2_lgleak6 = 0.0 ;
    if ( !model->HSMHV2_lglksd1_Given ) model->HSMHV2_lglksd1 = 0.0 ;
    if ( !model->HSMHV2_lglksd2_Given ) model->HSMHV2_lglksd2 = 0.0 ;
    if ( !model->HSMHV2_lglkb1_Given ) model->HSMHV2_lglkb1 = 0.0 ;
    if ( !model->HSMHV2_lglkb2_Given ) model->HSMHV2_lglkb2 = 0.0 ;
    if ( !model->HSMHV2_lnftrp_Given ) model->HSMHV2_lnftrp = 0.0 ;
    if ( !model->HSMHV2_lnfalp_Given ) model->HSMHV2_lnfalp = 0.0 ;
    if ( !model->HSMHV2_lvdiffj_Given ) model->HSMHV2_lvdiffj = 0.0 ;
    if ( !model->HSMHV2_libpc1_Given ) model->HSMHV2_libpc1 = 0.0 ;
    if ( !model->HSMHV2_libpc2_Given ) model->HSMHV2_libpc2 = 0.0 ;
    if ( !model->HSMHV2_lcgbo_Given ) model->HSMHV2_lcgbo = 0.0 ;
    if ( !model->HSMHV2_lcvdsover_Given ) model->HSMHV2_lcvdsover = 0.0 ;
    if ( !model->HSMHV2_lfalph_Given ) model->HSMHV2_lfalph = 0.0 ;
    if ( !model->HSMHV2_lnpext_Given ) model->HSMHV2_lnpext = 0.0 ;
    if ( !model->HSMHV2_lpowrat_Given ) model->HSMHV2_lpowrat = 0.0 ;
    if ( !model->HSMHV2_lrd_Given ) model->HSMHV2_lrd = 0.0 ;
    if ( !model->HSMHV2_lrd22_Given ) model->HSMHV2_lrd22 = 0.0 ;
    if ( !model->HSMHV2_lrd23_Given ) model->HSMHV2_lrd23 = 0.0 ;
    if ( !model->HSMHV2_lrd24_Given ) model->HSMHV2_lrd24 = 0.0 ;
    if ( !model->HSMHV2_lrdict1_Given ) model->HSMHV2_lrdict1 = 0.0 ;
    if ( !model->HSMHV2_lrdov13_Given ) model->HSMHV2_lrdov13 = 0.0 ;
    if ( !model->HSMHV2_lrdslp1_Given ) model->HSMHV2_lrdslp1 = 0.0 ;
    if ( !model->HSMHV2_lrdvb_Given ) model->HSMHV2_lrdvb = 0.0 ;
    if ( !model->HSMHV2_lrdvd_Given ) model->HSMHV2_lrdvd = 0.0 ;
    if ( !model->HSMHV2_lrdvg11_Given ) model->HSMHV2_lrdvg11 = 0.0 ;
    if ( !model->HSMHV2_lrs_Given ) model->HSMHV2_lrs = 0.0 ;
    if ( !model->HSMHV2_lrth0_Given ) model->HSMHV2_lrth0 = 0.0 ;
    if ( !model->HSMHV2_lvover_Given ) model->HSMHV2_lvover = 0.0 ;
    if ( !model->HSMHV2_ljs0d_Given     ) model->HSMHV2_ljs0d     = model->HSMHV2_ljs0 ;
    if ( !model->HSMHV2_ljs0swd_Given   ) model->HSMHV2_ljs0swd   = model->HSMHV2_ljs0sw ;
    if ( !model->HSMHV2_lnjd_Given      ) model->HSMHV2_lnjd      = model->HSMHV2_lnj ;
    if ( !model->HSMHV2_lcisbkd_Given   ) model->HSMHV2_lcisbkd   = model->HSMHV2_lcisbk ;
    if ( !model->HSMHV2_lvdiffjd_Given  ) model->HSMHV2_lvdiffjd  = model->HSMHV2_lvdiffj ;
    if ( !model->HSMHV2_ljs0s_Given     ) model->HSMHV2_ljs0s     = model->HSMHV2_ljs0d ;
    if ( !model->HSMHV2_ljs0sws_Given   ) model->HSMHV2_ljs0sws   = model->HSMHV2_ljs0swd ;
    if ( !model->HSMHV2_lnjs_Given      ) model->HSMHV2_lnjs      = model->HSMHV2_lnjd ;
    if ( !model->HSMHV2_lcisbks_Given   ) model->HSMHV2_lcisbks   = model->HSMHV2_lcisbkd ;
    if ( !model->HSMHV2_lvdiffjs_Given  ) model->HSMHV2_lvdiffjs  = model->HSMHV2_lvdiffjd ;

    /* Width dependence */
    if ( !model->HSMHV2_wvmax_Given ) model->HSMHV2_wvmax = 0.0 ;
    if ( !model->HSMHV2_wbgtmp1_Given ) model->HSMHV2_wbgtmp1 = 0.0 ;
    if ( !model->HSMHV2_wbgtmp2_Given ) model->HSMHV2_wbgtmp2 = 0.0 ;
    if ( !model->HSMHV2_weg0_Given ) model->HSMHV2_weg0 = 0.0 ;
    if ( !model->HSMHV2_wvfbover_Given ) model->HSMHV2_wvfbover = 0.0 ;
    if ( !model->HSMHV2_wnover_Given ) model->HSMHV2_wnover = 0.0 ;
    if ( !model->HSMHV2_wnovers_Given ) model->HSMHV2_wnovers = 0.0 ;
    if ( !model->HSMHV2_wwl2_Given ) model->HSMHV2_wwl2 = 0.0 ;
    if ( !model->HSMHV2_wvfbc_Given ) model->HSMHV2_wvfbc = 0.0 ;
    if ( !model->HSMHV2_wnsubc_Given ) model->HSMHV2_wnsubc = 0.0 ;
    if ( !model->HSMHV2_wnsubp_Given ) model->HSMHV2_wnsubp = 0.0 ;
    if ( !model->HSMHV2_wscp1_Given ) model->HSMHV2_wscp1 = 0.0 ;
    if ( !model->HSMHV2_wscp2_Given ) model->HSMHV2_wscp2 = 0.0 ;
    if ( !model->HSMHV2_wscp3_Given ) model->HSMHV2_wscp3 = 0.0 ;
    if ( !model->HSMHV2_wsc1_Given ) model->HSMHV2_wsc1 = 0.0 ;
    if ( !model->HSMHV2_wsc2_Given ) model->HSMHV2_wsc2 = 0.0 ;
    if ( !model->HSMHV2_wsc3_Given ) model->HSMHV2_wsc3 = 0.0 ;
    if ( !model->HSMHV2_wpgd1_Given ) model->HSMHV2_wpgd1 = 0.0 ;
    if ( !model->HSMHV2_wndep_Given ) model->HSMHV2_wndep = 0.0 ;
    if ( !model->HSMHV2_wninv_Given ) model->HSMHV2_wninv = 0.0 ;
    if ( !model->HSMHV2_wmuecb0_Given ) model->HSMHV2_wmuecb0 = 0.0 ;
    if ( !model->HSMHV2_wmuecb1_Given ) model->HSMHV2_wmuecb1 = 0.0 ;
    if ( !model->HSMHV2_wmueph1_Given ) model->HSMHV2_wmueph1 = 0.0 ;
    if ( !model->HSMHV2_wvtmp_Given ) model->HSMHV2_wvtmp = 0.0 ;
    if ( !model->HSMHV2_wwvth0_Given ) model->HSMHV2_wwvth0 = 0.0 ;
    if ( !model->HSMHV2_wmuesr1_Given ) model->HSMHV2_wmuesr1 = 0.0 ;
    if ( !model->HSMHV2_wmuetmp_Given ) model->HSMHV2_wmuetmp = 0.0 ;
    if ( !model->HSMHV2_wsub1_Given ) model->HSMHV2_wsub1 = 0.0 ;
    if ( !model->HSMHV2_wsub2_Given ) model->HSMHV2_wsub2 = 0.0 ;
    if ( !model->HSMHV2_wsvds_Given ) model->HSMHV2_wsvds = 0.0 ;
    if ( !model->HSMHV2_wsvbs_Given ) model->HSMHV2_wsvbs = 0.0 ;
    if ( !model->HSMHV2_wsvgs_Given ) model->HSMHV2_wsvgs = 0.0 ;
    if ( !model->HSMHV2_wfn1_Given ) model->HSMHV2_wfn1 = 0.0 ;
    if ( !model->HSMHV2_wfn2_Given ) model->HSMHV2_wfn2 = 0.0 ;
    if ( !model->HSMHV2_wfn3_Given ) model->HSMHV2_wfn3 = 0.0 ;
    if ( !model->HSMHV2_wfvbs_Given ) model->HSMHV2_wfvbs = 0.0 ;
    if ( !model->HSMHV2_wnsti_Given ) model->HSMHV2_wnsti = 0.0 ;
    if ( !model->HSMHV2_wwsti_Given ) model->HSMHV2_wwsti = 0.0 ;
    if ( !model->HSMHV2_wscsti1_Given ) model->HSMHV2_wscsti1 = 0.0 ;
    if ( !model->HSMHV2_wscsti2_Given ) model->HSMHV2_wscsti2 = 0.0 ;
    if ( !model->HSMHV2_wvthsti_Given ) model->HSMHV2_wvthsti = 0.0 ;
    if ( !model->HSMHV2_wmuesti1_Given ) model->HSMHV2_wmuesti1 = 0.0 ;
    if ( !model->HSMHV2_wmuesti2_Given ) model->HSMHV2_wmuesti2 = 0.0 ;
    if ( !model->HSMHV2_wmuesti3_Given ) model->HSMHV2_wmuesti3 = 0.0 ;
    if ( !model->HSMHV2_wnsubpsti1_Given ) model->HSMHV2_wnsubpsti1 = 0.0 ;
    if ( !model->HSMHV2_wnsubpsti2_Given ) model->HSMHV2_wnsubpsti2 = 0.0 ;
    if ( !model->HSMHV2_wnsubpsti3_Given ) model->HSMHV2_wnsubpsti3 = 0.0 ;
    if ( !model->HSMHV2_wcgso_Given ) model->HSMHV2_wcgso = 0.0 ;
    if ( !model->HSMHV2_wcgdo_Given ) model->HSMHV2_wcgdo = 0.0 ;
    if ( !model->HSMHV2_wjs0_Given ) model->HSMHV2_wjs0 = 0.0 ;
    if ( !model->HSMHV2_wjs0sw_Given ) model->HSMHV2_wjs0sw = 0.0 ;
    if ( !model->HSMHV2_wnj_Given ) model->HSMHV2_wnj = 0.0 ;
    if ( !model->HSMHV2_wcisbk_Given ) model->HSMHV2_wcisbk = 0.0 ;
    if ( !model->HSMHV2_wclm1_Given ) model->HSMHV2_wclm1 = 0.0 ;
    if ( !model->HSMHV2_wclm2_Given ) model->HSMHV2_wclm2 = 0.0 ;
    if ( !model->HSMHV2_wclm3_Given ) model->HSMHV2_wclm3 = 0.0 ;
    if ( !model->HSMHV2_wwfc_Given ) model->HSMHV2_wwfc = 0.0 ;
    if ( !model->HSMHV2_wgidl1_Given ) model->HSMHV2_wgidl1 = 0.0 ;
    if ( !model->HSMHV2_wgidl2_Given ) model->HSMHV2_wgidl2 = 0.0 ;
    if ( !model->HSMHV2_wgleak1_Given ) model->HSMHV2_wgleak1 = 0.0 ;
    if ( !model->HSMHV2_wgleak2_Given ) model->HSMHV2_wgleak2 = 0.0 ;
    if ( !model->HSMHV2_wgleak3_Given ) model->HSMHV2_wgleak3 = 0.0 ;
    if ( !model->HSMHV2_wgleak6_Given ) model->HSMHV2_wgleak6 = 0.0 ;
    if ( !model->HSMHV2_wglksd1_Given ) model->HSMHV2_wglksd1 = 0.0 ;
    if ( !model->HSMHV2_wglksd2_Given ) model->HSMHV2_wglksd2 = 0.0 ;
    if ( !model->HSMHV2_wglkb1_Given ) model->HSMHV2_wglkb1 = 0.0 ;
    if ( !model->HSMHV2_wglkb2_Given ) model->HSMHV2_wglkb2 = 0.0 ;
    if ( !model->HSMHV2_wnftrp_Given ) model->HSMHV2_wnftrp = 0.0 ;
    if ( !model->HSMHV2_wnfalp_Given ) model->HSMHV2_wnfalp = 0.0 ;
    if ( !model->HSMHV2_wvdiffj_Given ) model->HSMHV2_wvdiffj = 0.0 ;
    if ( !model->HSMHV2_wibpc1_Given ) model->HSMHV2_wibpc1 = 0.0 ;
    if ( !model->HSMHV2_wibpc2_Given ) model->HSMHV2_wibpc2 = 0.0 ;
    if ( !model->HSMHV2_wcgbo_Given ) model->HSMHV2_wcgbo = 0.0 ;
    if ( !model->HSMHV2_wcvdsover_Given ) model->HSMHV2_wcvdsover = 0.0 ;
    if ( !model->HSMHV2_wfalph_Given ) model->HSMHV2_wfalph = 0.0 ;
    if ( !model->HSMHV2_wnpext_Given ) model->HSMHV2_wnpext = 0.0 ;
    if ( !model->HSMHV2_wpowrat_Given ) model->HSMHV2_wpowrat = 0.0 ;
    if ( !model->HSMHV2_wrd_Given ) model->HSMHV2_wrd = 0.0 ;
    if ( !model->HSMHV2_wrd22_Given ) model->HSMHV2_wrd22 = 0.0 ;
    if ( !model->HSMHV2_wrd23_Given ) model->HSMHV2_wrd23 = 0.0 ;
    if ( !model->HSMHV2_wrd24_Given ) model->HSMHV2_wrd24 = 0.0 ;
    if ( !model->HSMHV2_wrdict1_Given ) model->HSMHV2_wrdict1 = 0.0 ;
    if ( !model->HSMHV2_wrdov13_Given ) model->HSMHV2_wrdov13 = 0.0 ;
    if ( !model->HSMHV2_wrdslp1_Given ) model->HSMHV2_wrdslp1 = 0.0 ;
    if ( !model->HSMHV2_wrdvb_Given ) model->HSMHV2_wrdvb = 0.0 ;
    if ( !model->HSMHV2_wrdvd_Given ) model->HSMHV2_wrdvd = 0.0 ;
    if ( !model->HSMHV2_wrdvg11_Given ) model->HSMHV2_wrdvg11 = 0.0 ;
    if ( !model->HSMHV2_wrs_Given ) model->HSMHV2_wrs = 0.0 ;
    if ( !model->HSMHV2_wrth0_Given ) model->HSMHV2_wrth0 = 0.0 ;
    if ( !model->HSMHV2_wvover_Given ) model->HSMHV2_wvover = 0.0 ;
    if ( !model->HSMHV2_wjs0d_Given     ) model->HSMHV2_wjs0d     = model->HSMHV2_wjs0 ;
    if ( !model->HSMHV2_wjs0swd_Given   ) model->HSMHV2_wjs0swd   = model->HSMHV2_wjs0sw ;
    if ( !model->HSMHV2_wnjd_Given      ) model->HSMHV2_wnjd      = model->HSMHV2_wnj ;
    if ( !model->HSMHV2_wcisbkd_Given   ) model->HSMHV2_wcisbkd   = model->HSMHV2_wcisbk ;
    if ( !model->HSMHV2_wvdiffjd_Given  ) model->HSMHV2_wvdiffjd  = model->HSMHV2_wvdiffj ;
    if ( !model->HSMHV2_wjs0s_Given     ) model->HSMHV2_wjs0s     = model->HSMHV2_wjs0d ;
    if ( !model->HSMHV2_wjs0sws_Given   ) model->HSMHV2_wjs0sws   = model->HSMHV2_wjs0swd ;
    if ( !model->HSMHV2_wnjs_Given      ) model->HSMHV2_wnjs      = model->HSMHV2_wnjd ;
    if ( !model->HSMHV2_wcisbks_Given   ) model->HSMHV2_wcisbks   = model->HSMHV2_wcisbkd ;
    if ( !model->HSMHV2_wvdiffjs_Given  ) model->HSMHV2_wvdiffjs  = model->HSMHV2_wvdiffjd ;

    /* Cross-term dependence */
    if ( !model->HSMHV2_pvmax_Given ) model->HSMHV2_pvmax = 0.0 ;
    if ( !model->HSMHV2_pbgtmp1_Given ) model->HSMHV2_pbgtmp1 = 0.0 ;
    if ( !model->HSMHV2_pbgtmp2_Given ) model->HSMHV2_pbgtmp2 = 0.0 ;
    if ( !model->HSMHV2_peg0_Given ) model->HSMHV2_peg0 = 0.0 ;
    if ( !model->HSMHV2_pvfbover_Given ) model->HSMHV2_pvfbover = 0.0 ;
    if ( !model->HSMHV2_pnover_Given ) model->HSMHV2_pnover = 0.0 ;
    if ( !model->HSMHV2_pnovers_Given ) model->HSMHV2_pnovers = 0.0 ;
    if ( !model->HSMHV2_pwl2_Given ) model->HSMHV2_pwl2 = 0.0 ;
    if ( !model->HSMHV2_pvfbc_Given ) model->HSMHV2_pvfbc = 0.0 ;
    if ( !model->HSMHV2_pnsubc_Given ) model->HSMHV2_pnsubc = 0.0 ;
    if ( !model->HSMHV2_pnsubp_Given ) model->HSMHV2_pnsubp = 0.0 ;
    if ( !model->HSMHV2_pscp1_Given ) model->HSMHV2_pscp1 = 0.0 ;
    if ( !model->HSMHV2_pscp2_Given ) model->HSMHV2_pscp2 = 0.0 ;
    if ( !model->HSMHV2_pscp3_Given ) model->HSMHV2_pscp3 = 0.0 ;
    if ( !model->HSMHV2_psc1_Given ) model->HSMHV2_psc1 = 0.0 ;
    if ( !model->HSMHV2_psc2_Given ) model->HSMHV2_psc2 = 0.0 ;
    if ( !model->HSMHV2_psc3_Given ) model->HSMHV2_psc3 = 0.0 ;
    if ( !model->HSMHV2_ppgd1_Given ) model->HSMHV2_ppgd1 = 0.0 ;
    if ( !model->HSMHV2_pndep_Given ) model->HSMHV2_pndep = 0.0 ;
    if ( !model->HSMHV2_pninv_Given ) model->HSMHV2_pninv = 0.0 ;
    if ( !model->HSMHV2_pmuecb0_Given ) model->HSMHV2_pmuecb0 = 0.0 ;
    if ( !model->HSMHV2_pmuecb1_Given ) model->HSMHV2_pmuecb1 = 0.0 ;
    if ( !model->HSMHV2_pmueph1_Given ) model->HSMHV2_pmueph1 = 0.0 ;
    if ( !model->HSMHV2_pvtmp_Given ) model->HSMHV2_pvtmp = 0.0 ;
    if ( !model->HSMHV2_pwvth0_Given ) model->HSMHV2_pwvth0 = 0.0 ;
    if ( !model->HSMHV2_pmuesr1_Given ) model->HSMHV2_pmuesr1 = 0.0 ;
    if ( !model->HSMHV2_pmuetmp_Given ) model->HSMHV2_pmuetmp = 0.0 ;
    if ( !model->HSMHV2_psub1_Given ) model->HSMHV2_psub1 = 0.0 ;
    if ( !model->HSMHV2_psub2_Given ) model->HSMHV2_psub2 = 0.0 ;
    if ( !model->HSMHV2_psvds_Given ) model->HSMHV2_psvds = 0.0 ;
    if ( !model->HSMHV2_psvbs_Given ) model->HSMHV2_psvbs = 0.0 ;
    if ( !model->HSMHV2_psvgs_Given ) model->HSMHV2_psvgs = 0.0 ;
    if ( !model->HSMHV2_pfn1_Given ) model->HSMHV2_pfn1 = 0.0 ;
    if ( !model->HSMHV2_pfn2_Given ) model->HSMHV2_pfn2 = 0.0 ;
    if ( !model->HSMHV2_pfn3_Given ) model->HSMHV2_pfn3 = 0.0 ;
    if ( !model->HSMHV2_pfvbs_Given ) model->HSMHV2_pfvbs = 0.0 ;
    if ( !model->HSMHV2_pnsti_Given ) model->HSMHV2_pnsti = 0.0 ;
    if ( !model->HSMHV2_pwsti_Given ) model->HSMHV2_pwsti = 0.0 ;
    if ( !model->HSMHV2_pscsti1_Given ) model->HSMHV2_pscsti1 = 0.0 ;
    if ( !model->HSMHV2_pscsti2_Given ) model->HSMHV2_pscsti2 = 0.0 ;
    if ( !model->HSMHV2_pvthsti_Given ) model->HSMHV2_pvthsti = 0.0 ;
    if ( !model->HSMHV2_pmuesti1_Given ) model->HSMHV2_pmuesti1 = 0.0 ;
    if ( !model->HSMHV2_pmuesti2_Given ) model->HSMHV2_pmuesti2 = 0.0 ;
    if ( !model->HSMHV2_pmuesti3_Given ) model->HSMHV2_pmuesti3 = 0.0 ;
    if ( !model->HSMHV2_pnsubpsti1_Given ) model->HSMHV2_pnsubpsti1 = 0.0 ;
    if ( !model->HSMHV2_pnsubpsti2_Given ) model->HSMHV2_pnsubpsti2 = 0.0 ;
    if ( !model->HSMHV2_pnsubpsti3_Given ) model->HSMHV2_pnsubpsti3 = 0.0 ;
    if ( !model->HSMHV2_pcgso_Given ) model->HSMHV2_pcgso = 0.0 ;
    if ( !model->HSMHV2_pcgdo_Given ) model->HSMHV2_pcgdo = 0.0 ;
    if ( !model->HSMHV2_pjs0_Given ) model->HSMHV2_pjs0 = 0.0 ;
    if ( !model->HSMHV2_pjs0sw_Given ) model->HSMHV2_pjs0sw = 0.0 ;
    if ( !model->HSMHV2_pnj_Given ) model->HSMHV2_pnj = 0.0 ;
    if ( !model->HSMHV2_pcisbk_Given ) model->HSMHV2_pcisbk = 0.0 ;
    if ( !model->HSMHV2_pclm1_Given ) model->HSMHV2_pclm1 = 0.0 ;
    if ( !model->HSMHV2_pclm2_Given ) model->HSMHV2_pclm2 = 0.0 ;
    if ( !model->HSMHV2_pclm3_Given ) model->HSMHV2_pclm3 = 0.0 ;
    if ( !model->HSMHV2_pwfc_Given ) model->HSMHV2_pwfc = 0.0 ;
    if ( !model->HSMHV2_pgidl1_Given ) model->HSMHV2_pgidl1 = 0.0 ;
    if ( !model->HSMHV2_pgidl2_Given ) model->HSMHV2_pgidl2 = 0.0 ;
    if ( !model->HSMHV2_pgleak1_Given ) model->HSMHV2_pgleak1 = 0.0 ;
    if ( !model->HSMHV2_pgleak2_Given ) model->HSMHV2_pgleak2 = 0.0 ;
    if ( !model->HSMHV2_pgleak3_Given ) model->HSMHV2_pgleak3 = 0.0 ;
    if ( !model->HSMHV2_pgleak6_Given ) model->HSMHV2_pgleak6 = 0.0 ;
    if ( !model->HSMHV2_pglksd1_Given ) model->HSMHV2_pglksd1 = 0.0 ;
    if ( !model->HSMHV2_pglksd2_Given ) model->HSMHV2_pglksd2 = 0.0 ;
    if ( !model->HSMHV2_pglkb1_Given ) model->HSMHV2_pglkb1 = 0.0 ;
    if ( !model->HSMHV2_pglkb2_Given ) model->HSMHV2_pglkb2 = 0.0 ;
    if ( !model->HSMHV2_pnftrp_Given ) model->HSMHV2_pnftrp = 0.0 ;
    if ( !model->HSMHV2_pnfalp_Given ) model->HSMHV2_pnfalp = 0.0 ;
    if ( !model->HSMHV2_pvdiffj_Given ) model->HSMHV2_pvdiffj = 0.0 ;
    if ( !model->HSMHV2_pibpc1_Given ) model->HSMHV2_pibpc1 = 0.0 ;
    if ( !model->HSMHV2_pibpc2_Given ) model->HSMHV2_pibpc2 = 0.0 ;
    if ( !model->HSMHV2_pcgbo_Given ) model->HSMHV2_pcgbo = 0.0 ;
    if ( !model->HSMHV2_pcvdsover_Given ) model->HSMHV2_pcvdsover = 0.0 ;
    if ( !model->HSMHV2_pfalph_Given ) model->HSMHV2_pfalph = 0.0 ;
    if ( !model->HSMHV2_pnpext_Given ) model->HSMHV2_pnpext = 0.0 ;
    if ( !model->HSMHV2_ppowrat_Given ) model->HSMHV2_ppowrat = 0.0 ;
    if ( !model->HSMHV2_prd_Given ) model->HSMHV2_prd = 0.0 ;
    if ( !model->HSMHV2_prd22_Given ) model->HSMHV2_prd22 = 0.0 ;
    if ( !model->HSMHV2_prd23_Given ) model->HSMHV2_prd23 = 0.0 ;
    if ( !model->HSMHV2_prd24_Given ) model->HSMHV2_prd24 = 0.0 ;
    if ( !model->HSMHV2_prdict1_Given ) model->HSMHV2_prdict1 = 0.0 ;
    if ( !model->HSMHV2_prdov13_Given ) model->HSMHV2_prdov13 = 0.0 ;
    if ( !model->HSMHV2_prdslp1_Given ) model->HSMHV2_prdslp1 = 0.0 ;
    if ( !model->HSMHV2_prdvb_Given ) model->HSMHV2_prdvb = 0.0 ;
    if ( !model->HSMHV2_prdvd_Given ) model->HSMHV2_prdvd = 0.0 ;
    if ( !model->HSMHV2_prdvg11_Given ) model->HSMHV2_prdvg11 = 0.0 ;
    if ( !model->HSMHV2_prs_Given ) model->HSMHV2_prs = 0.0 ;
    if ( !model->HSMHV2_prth0_Given ) model->HSMHV2_prth0 = 0.0 ;
    if ( !model->HSMHV2_pvover_Given ) model->HSMHV2_pvover = 0.0 ;
    if ( !model->HSMHV2_pjs0d_Given     ) model->HSMHV2_pjs0d     = model->HSMHV2_pjs0 ;
    if ( !model->HSMHV2_pjs0swd_Given   ) model->HSMHV2_pjs0swd   = model->HSMHV2_pjs0sw ;
    if ( !model->HSMHV2_pnjd_Given      ) model->HSMHV2_pnjd      = model->HSMHV2_pnj ;
    if ( !model->HSMHV2_pcisbkd_Given   ) model->HSMHV2_pcisbkd   = model->HSMHV2_pcisbk ;
    if ( !model->HSMHV2_pvdiffjd_Given  ) model->HSMHV2_pvdiffjd  = model->HSMHV2_pvdiffj ;
    if ( !model->HSMHV2_pjs0s_Given     ) model->HSMHV2_pjs0s     = model->HSMHV2_pjs0d ;
    if ( !model->HSMHV2_pjs0sws_Given   ) model->HSMHV2_pjs0sws   = model->HSMHV2_pjs0swd ;
    if ( !model->HSMHV2_pnjs_Given      ) model->HSMHV2_pnjs      = model->HSMHV2_pnjd ;
    if ( !model->HSMHV2_pcisbks_Given   ) model->HSMHV2_pcisbks   = model->HSMHV2_pcisbkd ;
    if ( !model->HSMHV2_pvdiffjs_Given  ) model->HSMHV2_pvdiffjs  = model->HSMHV2_pvdiffjd ;

    if (  model->HSMHV2_ldrift_Given ) model->HSMHV2_ldrift2 = model->HSMHV2_ldrift ;

    if (!model->HSMHV2vgsMaxGiven) model->HSMHV2vgsMax = 1e99;
    if (!model->HSMHV2vgdMaxGiven) model->HSMHV2vgdMax = 1e99;
    if (!model->HSMHV2vgbMaxGiven) model->HSMHV2vgbMax = 1e99;
    if (!model->HSMHV2vdsMaxGiven) model->HSMHV2vdsMax = 1e99;
    if (!model->HSMHV2vbsMaxGiven) model->HSMHV2vbsMax = 1e99;
    if (!model->HSMHV2vbdMaxGiven) model->HSMHV2vbdMax = 1e99;
    if (!model->HSMHV2vgsrMaxGiven) model->HSMHV2vgsrMax = 1e99;
    if (!model->HSMHV2vgdrMaxGiven) model->HSMHV2vgdrMax = 1e99;
    if (!model->HSMHV2vgbrMaxGiven) model->HSMHV2vgbrMax = 1e99;
    if (!model->HSMHV2vbsrMaxGiven) model->HSMHV2vbsrMax = 1e99;
    if (!model->HSMHV2vbdrMaxGiven) model->HSMHV2vbdrMax = 1e99;

    /* For Symmetrical Device */
    if (  model->HSMHV2_cosym ) {
       if(!model->HSMHV2_rs_Given  )
         { model->HSMHV2_rs = model->HSMHV2_rd  ; }
       if(!model->HSMHV2_coovlps_Given  )
         { model->HSMHV2_coovlps = model->HSMHV2_coovlp  ; }
       if(!model->HSMHV2_novers_Given   )
         { model->HSMHV2_novers  = model->HSMHV2_nover   ; }
/*        if(!model->HSMHV2_xld_Given   ) */
/*          { model->HSMHV2_xld  = model->HSMHV2_xldld   ; } */
       if(!model->HSMHV2_lover_Given    ) 
         { model->HSMHV2_lover  = model->HSMHV2_loverld ; }
//     if(!model->HSMHV2_lovers_Given   ) 
//       { model->HSMHV2_lovers  = model->HSMHV2_loverld ; }
       if(!model->HSMHV2_ldrift1s_Given ) 
         { model->HSMHV2_ldrift1s = model->HSMHV2_ldrift1 ; }
       if(!model->HSMHV2_ldrift2s_Given )  
         { model->HSMHV2_ldrift2s = model->HSMHV2_ldrift2 ; }
       if(!model->HSMHV2_cgso_Given     ) { model->HSMHV2_cgso       = model->HSMHV2_cgdo ;
                                           model->HSMHV2_cgso_Given = model->HSMHV2_cgdo_Given ; }
    }
    if ( !model->HSMHV2_lovers_Given   ) model->HSMHV2_lovers  = model->HSMHV2_lover ;

    if ( model->HSMHV2_cvbk_Given  ) {
      fprintf(stderr,"warning(HiSIM_HV(%s)): CVBK has been inactivated by CVB.\n",model->HSMHV2modName);
    }
    if (  model->HSMHV2_cordrift ) { model->HSMHV2_corsrd = 0 ; }


    modelMKS = &model->modelMKS ;

    /* loop through all the instances of the model */
    for ( here = HSMHV2instances(model);here != NULL ;
         here = HSMHV2nextInstance(here)) {
      /* allocate a chunk of the state vector */
      here->HSMHV2states = *states;
      if (model->HSMHV2_conqs)
	*states += HSMHV2numStatesNqs;
      else 
	*states += HSMHV2numStates;

      hereMKS  = &here->hereMKS ;
      /* perform the device parameter defaulting */
      if ( !here->HSMHV2_coselfheat_Given ) here->HSMHV2_coselfheat = model->HSMHV2_coselfheat ;
      if ( !here->HSMHV2_cosubnode_Given  ) here->HSMHV2_cosubnode  = model->HSMHV2_cosubnode ;
      if ( !here->HSMHV2_l_Given      ) here->HSMHV2_l      = 2.0e-6 ;
      if ( !here->HSMHV2_w_Given      ) here->HSMHV2_w      = 5.0e-6 ;
      if ( !here->HSMHV2_ad_Given     ) here->HSMHV2_ad     = 0.0 ;
      if ( !here->HSMHV2_as_Given     ) here->HSMHV2_as     = 0.0 ;
      if ( !here->HSMHV2_pd_Given     ) here->HSMHV2_pd     = 0.0 ;
      if ( !here->HSMHV2_ps_Given     ) here->HSMHV2_ps     = 0.0 ;
      if ( !here->HSMHV2_nrd_Given    ) here->HSMHV2_nrd    = 1.0 ;
      if ( !here->HSMHV2_nrs_Given    ) here->HSMHV2_nrs    = 1.0 ;
      if ( !here->HSMHV2_ngcon_Given  ) here->HSMHV2_ngcon  = 1.0 ;
      if ( !here->HSMHV2_xgw_Given    ) here->HSMHV2_xgw    = 0e0 ;
      if ( !here->HSMHV2_xgl_Given    ) here->HSMHV2_xgl    = 0e0 ;
      if ( !here->HSMHV2_nf_Given     ) here->HSMHV2_nf     = 1.0 ;
      if ( !here->HSMHV2_sa_Given     ) here->HSMHV2_sa     = 0 ;
      if ( !here->HSMHV2_sb_Given     ) here->HSMHV2_sb     = 0 ;
      if ( !here->HSMHV2_sd_Given     ) here->HSMHV2_sd     = 0 ;
      if ( !here->HSMHV2_dtemp_Given  ) here->HSMHV2_dtemp  = 0.0 ;

      if ( !here->HSMHV2_icVBS_Given ) here->HSMHV2_icVBS = 0.0;
      if ( !here->HSMHV2_icVDS_Given ) here->HSMHV2_icVDS = 0.0;
      if ( !here->HSMHV2_icVGS_Given ) here->HSMHV2_icVGS = 0.0;

      if ( !here->HSMHV2_corbnet_Given )
	here->HSMHV2_corbnet = model->HSMHV2_corbnet ;
      else if ( here->HSMHV2_corbnet != 0 && here->HSMHV2_corbnet != 1 ) {
	here->HSMHV2_corbnet = model->HSMHV2_corbnet ;
	printf("warning(HiSIM_HV(%s)): CORBNET has been set to its default value: %d.\n", model->HSMHV2modName,here->HSMHV2_corbnet);
      }
      if ( !here->HSMHV2_rbdb_Given) here->HSMHV2_rbdb = model->HSMHV2_rbdb; /* not used in this version */
      if ( !here->HSMHV2_rbsb_Given) here->HSMHV2_rbsb = model->HSMHV2_rbsb; /* not used in this version */
      if ( !here->HSMHV2_rbpb_Given) here->HSMHV2_rbpb = model->HSMHV2_rbpb;
      if ( !here->HSMHV2_rbps_Given) here->HSMHV2_rbps = model->HSMHV2_rbps;
      if ( !here->HSMHV2_rbpd_Given) here->HSMHV2_rbpd = model->HSMHV2_rbpd;

      if ( !here->HSMHV2_corg_Given )
	here->HSMHV2_corg = model->HSMHV2_corg ;
      else if ( here->HSMHV2_corg != 0 && here->HSMHV2_corg != 1 ) {
	here->HSMHV2_corg = model->HSMHV2_corg ;
	printf("warning(HiSIM_HV(%s)): CORG has been set to its default value: %d.\n", model->HSMHV2modName,here->HSMHV2_corg);
      }

      if ( !here->HSMHV2_m_Given      ) here->HSMHV2_m      = 1.0 ;
      if (  here->HSMHV2_subld1_Given  ) {
            printf("warning(HiSIM_HV(%s)): SUBLD1 has been inactived in instance param.\n",model->HSMHV2modName);
      }
      if (  here->HSMHV2_subld2_Given  ) {
            printf("warning(HiSIM_HV(%s)): SUBLD2 has been inactived in instance param.\n",model->HSMHV2modName);
      }
      if ( !here->HSMHV2_lovers_Given  ) here->HSMHV2_lovers  = model->HSMHV2_lovers ;
      if ( !here->HSMHV2_lover_Given   ) here->HSMHV2_lover   = model->HSMHV2_lover ;
      if ( !here->HSMHV2_loverld_Given ) here->HSMHV2_loverld = model->HSMHV2_loverld ;
      if ( !here->HSMHV2_ldrift1_Given  ) here->HSMHV2_ldrift1 = model->HSMHV2_ldrift1 ;
      if ( !here->HSMHV2_ldrift2_Given  ) here->HSMHV2_ldrift2 = model->HSMHV2_ldrift2 ;
      if ( !here->HSMHV2_ldrift1s_Given ) here->HSMHV2_ldrift1s = model->HSMHV2_ldrift1s ;
      if ( !here->HSMHV2_ldrift2s_Given ) here->HSMHV2_ldrift2s = model->HSMHV2_ldrift2s ;

      if (  model->HSMHV2_cosym ) {
//                                                                        here->HSMHV2_lover  = here->HSMHV2_lovers ;
         if ( !here->HSMHV2_lover_Given    && !model->HSMHV2_lover_Given  ) here->HSMHV2_lover  = here->HSMHV2_loverld ; 
         if ( !here->HSMHV2_lovers_Given   && !model->HSMHV2_lovers_Given ) here->HSMHV2_lovers = here->HSMHV2_lover ;
         if ( !here->HSMHV2_ldrift1s_Given && !model->HSMHV2_ldrift1s_Given ) here->HSMHV2_ldrift1s = here->HSMHV2_ldrift1 ;
         if ( !here->HSMHV2_ldrift2s_Given && !model->HSMHV2_ldrift2s_Given ) here->HSMHV2_ldrift2s = here->HSMHV2_ldrift2 ;
      }


      /* process drain series resistance */
      /* rough check if Rd != 0 *  ****  don't forget to change if Rd processing is changed *******/
      T2 = ( here->HSMHV2_ldrift1 * model->HSMHV2_rdslp1 * C_m2um  + model->HSMHV2_rdict1 )
	   * ( here->HSMHV2_ldrift2 * model->HSMHV2_rdslp2 * C_m2um  + model->HSMHV2_rdict2 ) ;
      Rd = model->HSMHV2_rsh * here->HSMHV2_nrd * here->HSMHV2_nf + (model->HSMHV2_rd + model->HSMHV2_rdvd) * T2 ;
      if ( ( ( ( model->HSMHV2_corsrd == 1 || model->HSMHV2_corsrd == 3 ) && Rd > 0.0 ) 
	     || model->HSMHV2_cordrift == 1 ) ) {
	if( here->HSMHV2dNodePrime <= 0) {
        error = CKTmkVolt(ckt, &tmp, here->HSMHV2name, "drain");
	if (error) return(error);
	here->HSMHV2dNodePrime = tmp->number;
       }
      } else {
	here->HSMHV2dNodePrime = here->HSMHV2dNode;
      }
      here->HSMHV2drainConductance = 0.0 ; /* initialized for hsmhvnoi.c */
      
      /* process source series resistance */
      /* rough check if Rs != 0 *  ***** don't forget to change if Rs processing is changed *******/
      if(model->HSMHV2_corsrd == 1 || model->HSMHV2_corsrd == 3) {
      T2 = ( here->HSMHV2_ldrift1s * model->HSMHV2_rdslp1 * C_m2um + model->HSMHV2_rdict1 )
	   * ( here->HSMHV2_ldrift2s * model->HSMHV2_rdslp2 * C_m2um + model->HSMHV2_rdict2 ) ;
      }else{ T2 = 0.0; }
      Rs = model->HSMHV2_rsh * here->HSMHV2_nrs * here->HSMHV2_nf + model->HSMHV2_rs * T2 ;
      if ( (model->HSMHV2_corsrd == 1 || model->HSMHV2_corsrd == 3 || model->HSMHV2_cordrift == 1)
           && Rs > 0.0 ) {
       if( here->HSMHV2sNodePrime == 0) {
        error = CKTmkVolt(ckt, &tmp, here->HSMHV2name, "source");
	if (error) return(error);
	here->HSMHV2sNodePrime = tmp->number;
       }
      } else {
	here->HSMHV2sNodePrime = here->HSMHV2sNode;
      }
      here->HSMHV2sourceConductance = 0.0 ; /* initialized for hsmhvnoi.c */

      /* process gate resistance */
      if ( (here->HSMHV2_corg == 1 && model->HSMHV2_rshg > 0.0) ) {
       if (here->HSMHV2gNodePrime == 0) {
	error = CKTmkVolt(ckt, &tmp, here->HSMHV2name, "gate");
	if (error) return(error);
	here->HSMHV2gNodePrime = tmp->number;
       }
      } else {
	here->HSMHV2gNodePrime = here->HSMHV2gNode;
      }

      /* internal body nodes for body resistance model */
      if ( here->HSMHV2_corbnet == 1 ) {
	if (here->HSMHV2dbNode == 0) {
	  error = CKTmkVolt(ckt, &tmp, here->HSMHV2name, "dbody");
	  if (error) return(error);
	  here->HSMHV2dbNode = tmp->number;
	}
	if (here->HSMHV2bNodePrime == 0) {
	  error = CKTmkVolt(ckt, &tmp,here->HSMHV2name, "body");
	  if (error) return(error);
	  here->HSMHV2bNodePrime = tmp->number;
	}
	if (here->HSMHV2sbNode == 0) {
	  error = CKTmkVolt(ckt, &tmp, here->HSMHV2name,"sbody");
	  if (error) return(error);
	  here->HSMHV2sbNode = tmp->number;
	}
      } else {
	here->HSMHV2dbNode = here->HSMHV2bNodePrime = here->HSMHV2sbNode = here->HSMHV2bNode;
      }

      here->HSMHV2tempNode = here->HSMHV2tempNodeExt;
      here->HSMHV2subNode = here->HSMHV2subNodeExt;

      if ( here->HSMHV2_cosubnode == 0 && here->HSMHV2subNode >= 0 ) {
        if ( here->HSMHV2tempNode >= 0 ) {
      /* FATAL Error when 6th node is defined and COSUBNODE=0 */
          IFuid namarr[2];
          namarr[0] = here->HSMHV2name;
          namarr[1] = model->HSMHV2modName;
          (*(SPfrontEnd->IFerror))
            ( 
             ERR_FATAL, 
             "HiSIM_HV: MOSFET(%s) MODEL(%s): 6th node is defined and COSUBNODE=0", 
             namarr 
             );
          return (E_BADPARM);
        } else {
      /* 5th node is switched to tempNode, if COSUBNODE=0 and 5 external nodes are assigned. */
	here->HSMHV2tempNode = here->HSMHV2subNode ;
	here->HSMHV2subNode  = -1 ; 
        }
      }

      /* self heating*/
      if ( here->HSMHV2_coselfheat >  0 && here->HSMHV2tempNode <= 0 ){
	error = CKTmkVolt(ckt, &tmp, here->HSMHV2name,"temp");
	if(error) return(error);
	here->HSMHV2tempNode = tmp->number;
      }
      if ( here->HSMHV2_coselfheat <= 0 ) here->HSMHV2tempNode = -1;

      /* flat handling of NQS */
      if ( model->HSMHV2_conqs ){
	error = CKTmkVolt(ckt, &tmp, here->HSMHV2name,"qi_nqs");
	if(error) return(error);
	here->HSMHV2qiNode = tmp->number;
	error = CKTmkVolt(ckt, &tmp, here->HSMHV2name,"qb_nqs");
	if(error) return(error);
	here->HSMHV2qbNode = tmp->number;
      }
      
                   
      /* set Sparse Matrix Pointers */
      
      /* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==NULL){\
    return(E_NOMEM);\
} } while(0)

      TSTALLOC(HSMHV2DPbpPtr, HSMHV2dNodePrime, HSMHV2bNodePrime);
      TSTALLOC(HSMHV2SPbpPtr, HSMHV2sNodePrime, HSMHV2bNodePrime);
      TSTALLOC(HSMHV2GPbpPtr, HSMHV2gNodePrime, HSMHV2bNodePrime);

      TSTALLOC(HSMHV2BPdPtr,  HSMHV2bNodePrime, HSMHV2dNode);
      TSTALLOC(HSMHV2BPsPtr,  HSMHV2bNodePrime, HSMHV2sNode);
      TSTALLOC(HSMHV2BPdpPtr, HSMHV2bNodePrime, HSMHV2dNodePrime);
      TSTALLOC(HSMHV2BPspPtr, HSMHV2bNodePrime, HSMHV2sNodePrime);
      TSTALLOC(HSMHV2BPgpPtr, HSMHV2bNodePrime, HSMHV2gNodePrime);
      TSTALLOC(HSMHV2BPbpPtr, HSMHV2bNodePrime, HSMHV2bNodePrime);

      TSTALLOC(HSMHV2DdPtr, HSMHV2dNode, HSMHV2dNode);
      TSTALLOC(HSMHV2GPgpPtr, HSMHV2gNodePrime, HSMHV2gNodePrime);
      TSTALLOC(HSMHV2SsPtr, HSMHV2sNode, HSMHV2sNode);
      TSTALLOC(HSMHV2DPdpPtr, HSMHV2dNodePrime, HSMHV2dNodePrime);
      TSTALLOC(HSMHV2SPspPtr, HSMHV2sNodePrime, HSMHV2sNodePrime);
      TSTALLOC(HSMHV2DdpPtr, HSMHV2dNode, HSMHV2dNodePrime);
      TSTALLOC(HSMHV2GPdpPtr, HSMHV2gNodePrime, HSMHV2dNodePrime);
      TSTALLOC(HSMHV2GPspPtr, HSMHV2gNodePrime, HSMHV2sNodePrime);
      TSTALLOC(HSMHV2SspPtr, HSMHV2sNode, HSMHV2sNodePrime);
      TSTALLOC(HSMHV2DPspPtr, HSMHV2dNodePrime, HSMHV2sNodePrime);
      TSTALLOC(HSMHV2DPdPtr, HSMHV2dNodePrime, HSMHV2dNode);
      TSTALLOC(HSMHV2DPgpPtr, HSMHV2dNodePrime, HSMHV2gNodePrime);
      TSTALLOC(HSMHV2SPgpPtr, HSMHV2sNodePrime, HSMHV2gNodePrime);
      TSTALLOC(HSMHV2SPsPtr, HSMHV2sNodePrime, HSMHV2sNode);
      TSTALLOC(HSMHV2SPdpPtr, HSMHV2sNodePrime, HSMHV2dNodePrime);

      TSTALLOC(HSMHV2GgPtr, HSMHV2gNode, HSMHV2gNode);
      TSTALLOC(HSMHV2GgpPtr, HSMHV2gNode, HSMHV2gNodePrime);
      TSTALLOC(HSMHV2GPgPtr, HSMHV2gNodePrime, HSMHV2gNode);
      /* TSTALLOC(HSMHV2GdpPtr, HSMHV2gNode, HSMHV2dNodePrime);	not used */
      /* TSTALLOC(HSMHV2GspPtr, HSMHV2gNode, HSMHV2sNodePrime);	not used */
      /* TSTALLOC(HSMHV2GbpPtr, HSMHV2gNode, HSMHV2bNodePrime);	not used */
      TSTALLOC(HSMHV2DdbPtr, HSMHV2dNode, HSMHV2dbNode);
      TSTALLOC(HSMHV2SsbPtr, HSMHV2sNode, HSMHV2sbNode);

      TSTALLOC(HSMHV2DBdPtr, HSMHV2dbNode, HSMHV2dNode);
      TSTALLOC(HSMHV2DBdbPtr, HSMHV2dbNode, HSMHV2dbNode);
      TSTALLOC(HSMHV2DBbpPtr, HSMHV2dbNode, HSMHV2bNodePrime);
      /* TSTALLOC(HSMHV2DBbPtr, HSMHV2dbNode, HSMHV2bNode);	not used */

      TSTALLOC(HSMHV2BPdbPtr, HSMHV2bNodePrime, HSMHV2dbNode);
      TSTALLOC(HSMHV2BPbPtr, HSMHV2bNodePrime, HSMHV2bNode);
      TSTALLOC(HSMHV2BPsbPtr, HSMHV2bNodePrime, HSMHV2sbNode);

      TSTALLOC(HSMHV2SBsPtr, HSMHV2sbNode, HSMHV2sNode);
      TSTALLOC(HSMHV2SBbpPtr, HSMHV2sbNode, HSMHV2bNodePrime);
      /* TSTALLOC(HSMHV2SBbPtr, HSMHV2sbNode, HSMHV2bNode);	not used */
      TSTALLOC(HSMHV2SBsbPtr, HSMHV2sbNode, HSMHV2sbNode);

      /* TSTALLOC(HSMHV2BdbPtr, HSMHV2bNode, HSMHV2dbNode);	not used */
      TSTALLOC(HSMHV2BbpPtr, HSMHV2bNode, HSMHV2bNodePrime);
      /* TSTALLOC(HSMHV2BsbPtr, HSMHV2bNode, HSMHV2sbNode);	not used */
      TSTALLOC(HSMHV2BbPtr, HSMHV2bNode, HSMHV2bNode);

      TSTALLOC(HSMHV2DgpPtr, HSMHV2dNode, HSMHV2gNodePrime);
      TSTALLOC(HSMHV2DsPtr, HSMHV2dNode, HSMHV2sNode);
      TSTALLOC(HSMHV2DbpPtr, HSMHV2dNode, HSMHV2bNodePrime);
      TSTALLOC(HSMHV2DspPtr, HSMHV2dNode, HSMHV2sNodePrime);
      TSTALLOC(HSMHV2DPsPtr, HSMHV2dNodePrime, HSMHV2sNode);

      TSTALLOC(HSMHV2SgpPtr, HSMHV2sNode, HSMHV2gNodePrime);
      TSTALLOC(HSMHV2SdPtr, HSMHV2sNode, HSMHV2dNode);
      TSTALLOC(HSMHV2SbpPtr, HSMHV2sNode, HSMHV2bNodePrime);
      TSTALLOC(HSMHV2SdpPtr, HSMHV2sNode, HSMHV2dNodePrime);
      TSTALLOC(HSMHV2SPdPtr, HSMHV2sNodePrime, HSMHV2dNode);

      TSTALLOC(HSMHV2GPdPtr, HSMHV2gNodePrime, HSMHV2dNode);
      TSTALLOC(HSMHV2GPsPtr, HSMHV2gNodePrime, HSMHV2sNode);
	
      if ( here->HSMHV2subNode > 0 ) { /* 5th substrate node */
	TSTALLOC(HSMHV2DsubPtr,  HSMHV2dNode,      HSMHV2subNode);
	TSTALLOC(HSMHV2DPsubPtr, HSMHV2dNodePrime, HSMHV2subNode);
	TSTALLOC(HSMHV2SsubPtr,  HSMHV2sNode,      HSMHV2subNode);
	TSTALLOC(HSMHV2SPsubPtr, HSMHV2sNodePrime, HSMHV2subNode);
      }
      if ( here->HSMHV2tempNode >  0 ) { /* self heating */
	TSTALLOC(HSMHV2TemptempPtr, HSMHV2tempNode, HSMHV2tempNode);
	TSTALLOC(HSMHV2TempdPtr, HSMHV2tempNode, HSMHV2dNode);
	TSTALLOC(HSMHV2TempdpPtr, HSMHV2tempNode, HSMHV2dNodePrime);
	TSTALLOC(HSMHV2TempsPtr, HSMHV2tempNode, HSMHV2sNode);
	TSTALLOC(HSMHV2TempspPtr, HSMHV2tempNode, HSMHV2sNodePrime);
	TSTALLOC(HSMHV2DPtempPtr, HSMHV2dNodePrime, HSMHV2tempNode);
	TSTALLOC(HSMHV2SPtempPtr, HSMHV2sNodePrime, HSMHV2tempNode);
  
        TSTALLOC(HSMHV2TempgpPtr, HSMHV2tempNode, HSMHV2gNodePrime);
	TSTALLOC(HSMHV2TempbpPtr, HSMHV2tempNode, HSMHV2bNodePrime);

	TSTALLOC(HSMHV2GPtempPtr, HSMHV2gNodePrime, HSMHV2tempNode);
	TSTALLOC(HSMHV2BPtempPtr, HSMHV2bNodePrime, HSMHV2tempNode);

	TSTALLOC(HSMHV2DBtempPtr, HSMHV2dbNode, HSMHV2tempNode);
	TSTALLOC(HSMHV2SBtempPtr, HSMHV2sbNode, HSMHV2tempNode);
	TSTALLOC(HSMHV2DtempPtr, HSMHV2dNode, HSMHV2tempNode);
	TSTALLOC(HSMHV2StempPtr, HSMHV2sNode, HSMHV2tempNode);
      }
      if ( model->HSMHV2_conqs ) { /* flat handling of NQS */
	TSTALLOC(HSMHV2DPqiPtr, HSMHV2dNodePrime, HSMHV2qiNode);
	TSTALLOC(HSMHV2GPqiPtr, HSMHV2gNodePrime, HSMHV2qiNode);
	TSTALLOC(HSMHV2GPqbPtr, HSMHV2gNodePrime, HSMHV2qbNode);
	TSTALLOC(HSMHV2SPqiPtr, HSMHV2sNodePrime, HSMHV2qiNode);
	TSTALLOC(HSMHV2BPqbPtr, HSMHV2bNodePrime, HSMHV2qbNode);
	TSTALLOC(HSMHV2QIdpPtr, HSMHV2qiNode, HSMHV2dNodePrime);
	TSTALLOC(HSMHV2QIgpPtr, HSMHV2qiNode, HSMHV2gNodePrime);
	TSTALLOC(HSMHV2QIspPtr, HSMHV2qiNode, HSMHV2sNodePrime);
	TSTALLOC(HSMHV2QIbpPtr, HSMHV2qiNode, HSMHV2bNodePrime);
	TSTALLOC(HSMHV2QIqiPtr, HSMHV2qiNode, HSMHV2qiNode);
	TSTALLOC(HSMHV2QBdpPtr, HSMHV2qbNode, HSMHV2dNodePrime);
	TSTALLOC(HSMHV2QBgpPtr, HSMHV2qbNode, HSMHV2gNodePrime);
	TSTALLOC(HSMHV2QBspPtr, HSMHV2qbNode, HSMHV2sNodePrime);
	TSTALLOC(HSMHV2QBbpPtr, HSMHV2qbNode, HSMHV2bNodePrime);
	TSTALLOC(HSMHV2QBqbPtr, HSMHV2qbNode, HSMHV2qbNode);
        if ( here->HSMHV2tempNode >  0 ) { /* self heating */
	  TSTALLOC(HSMHV2QItempPtr, HSMHV2qiNode, HSMHV2tempNode);
	  TSTALLOC(HSMHV2QBtempPtr, HSMHV2qbNode, HSMHV2tempNode);
        }
      }

      /*-----------------------------------------------------------*
       * Range check of instance parameters
       *-----------------*/
      RANGECHECK(here->HSMHV2_l, model->HSMHV2_lmin, model->HSMHV2_lmax, "L") ;
      RANGECHECK(here->HSMHV2_w/here->HSMHV2_nf, model->HSMHV2_wmin, model->HSMHV2_wmax, "W/NF") ;

      /* binning calculation */
      pParam = &here->pParam ;
      Lgate = here->HSMHV2_l + model->HSMHV2_xl ;
      Wgate = here->HSMHV2_w / here->HSMHV2_nf  + model->HSMHV2_xw ;
      LG = Lgate * C_m2um ;
      WG = Wgate * C_m2um ;
      Lbin = pow(LG, model->HSMHV2_lbinn) ;
      Wbin = pow(WG, model->HSMHV2_wbinn) ;
      LWbin = Lbin * Wbin ;

      BINNING(vmax);
      BINNING(js0d);
      BINNING(js0swd);
      BINNING(njd);
      BINNING(cisbkd);
      BINNING(vdiffjd);
      BINNING(js0s);
      BINNING(js0sws);
      BINNING(njs);
      BINNING(cisbks);
      BINNING(vdiffjs);
      BINNING(bgtmp1);
      BINNING(bgtmp2);
      BINNING(eg0);
      BINNING(vfbover);
      BINNING(nover);
      BINNING(novers);
      BINNING(wl2);
      BINNING(vfbc);
      BINNING(nsubc);
      BINNING(nsubp);
      BINNING(scp1);
      BINNING(scp2);
      BINNING(scp3);
      BINNING(sc1);
      BINNING(sc2);
      BINNING(sc3);
      BINNING(pgd1);
      BINNING(ndep);
      BINNING(ninv);
      BINNING(muecb0);
      BINNING(muecb1);
      BINNING(mueph1);
      BINNING(vtmp);
      BINNING(wvth0);
      BINNING(muesr1);
      BINNING(muetmp);
      BINNING(sub1);
      BINNING(sub2);
      BINNING(svds);
      BINNING(svbs);
      BINNING(svgs);
      BINNING(fn1);
      BINNING(fn2);
      BINNING(fn3);
      BINNING(fvbs);
      BINNING(nsti);
      BINNING(wsti);
      BINNING(scsti1);
      BINNING(scsti2);
      BINNING(vthsti);
      BINNING(muesti1);
      BINNING(muesti2);
      BINNING(muesti3);
      BINNING(nsubpsti1);
      BINNING(nsubpsti2);
      BINNING(nsubpsti3);
      BINNING(cgso);
      BINNING(cgdo);
      BINNING(clm1);
      BINNING(clm2);
      BINNING(clm3);
      BINNING(wfc);
      BINNING(gidl1);
      BINNING(gidl2);
      BINNING(gleak1);
      BINNING(gleak2);
      BINNING(gleak3);
      BINNING(gleak6);
      BINNING(glksd1);
      BINNING(glksd2);
      BINNING(glkb1);
      BINNING(glkb2);
      BINNING(nftrp);
      BINNING(nfalp);
      BINNING(ibpc1);
      BINNING(ibpc2);
      BINNING(cgbo);
      BINNING(cvdsover);
      BINNING(falph);
      BINNING(npext);
      BINNING(powrat);
      BINNING(rd);
      BINNING(rd22);
      BINNING(rd23);
      BINNING(rd24);
      BINNING(rdict1);
      BINNING(rdov13);
      BINNING(rdslp1);
      BINNING(rdvb);
      BINNING(rdvd);
      BINNING(rdvg11);
      BINNING(rs);
      BINNING(rth0);
      BINNING(vover);

      /*-----------------------------------------------------------*
       * Range check of model parameters
       *-----------------*/
      RANGECHECK(pParam->HSMHV2_vmax,     1.0e6,   20.0e6, "VMAX") ;
      RANGECHECK(pParam->HSMHV2_bgtmp1, 50.0e-6,   1.0e-3, "BGTMP1") ;
      RANGECHECK(pParam->HSMHV2_bgtmp2, -1.0e-6,   1.0e-6, "BGTMP2") ;
      RANGECHECK(pParam->HSMHV2_eg0,        1.0,      1.3, "EG0") ;
      RANGECHECK(pParam->HSMHV2_vfbover,   -1.2,      1.0, "VFBOVER") ;
      if( model->HSMHV2_codep == 0 ) {
        RANGECHECK(pParam->HSMHV2_vfbc,      -1.2,      0.0, "VFBC") ;
      } else {
        RANGECHECK(pParam->HSMHV2_vfbc,      -1.2,      0.8, "VFBC") ;
      }
      RANGECHECK(pParam->HSMHV2_nsubc,   1.0e16,   1.0e19, "NSUBC") ;
      RANGECHECK(pParam->HSMHV2_nsubp,   1.0e16,   1.0e19, "NSUBP") ;
      RANGECHECK(pParam->HSMHV2_scp1,       0.0,     10.0, "SCP1") ;
      RANGECHECK(pParam->HSMHV2_scp2,       0.0,      1.0, "SCP2") ;
      RANGECHECK(pParam->HSMHV2_scp3,       0.0,   200e-9, "SCP3") ;
      RANGECHECK(pParam->HSMHV2_sc1,        0.0,     10.0, "SC1") ;
      RANGECHECK(pParam->HSMHV2_sc2,        0.0,      1.0, "SC2") ;
      RANGECHECK(pParam->HSMHV2_sc3,        0.0,   20e-6, "SC3") ;
      RANGECHECK(pParam->HSMHV2_pgd1,       0.0,  30.0e-3, "PGD1") ;
      RANGECHECK(pParam->HSMHV2_ndep,       0.0,      1.0, "NDEP") ;
      RANGECHECK(pParam->HSMHV2_ninv,       0.0,      1.0, "NINV") ;
      RANGECHECK(pParam->HSMHV2_muecb0,   100.0,  100.0e3, "MUECB0") ;
      RANGECHECK(pParam->HSMHV2_muecb1,     5.0,   10.0e3, "MUECB1") ;
      RANGECHECK(pParam->HSMHV2_mueph1,   2.0e3,   30.0e3, "MUEPH1") ;
      RANGECHECK(pParam->HSMHV2_vtmp,      -2.0,      1.0, "VTMP") ;
      RANGECHECK(pParam->HSMHV2_muesr1,  1.0e14,   1.0e16, "MUESR1") ;
      RANGECHECK(pParam->HSMHV2_muetmp,     0.5,      2.5, "MUETMP") ;
      RANGECHECK(pParam->HSMHV2_clm1,      0.01,      1.0, "CLM1") ;
      RANGECHECK(pParam->HSMHV2_clm2,       1.0,      4.0, "CLM2") ;
      RANGECHECK(pParam->HSMHV2_clm3,       0.5,      5.0, "CLM3") ;
      RANGECHECK(pParam->HSMHV2_wfc,   -5.0e-15,   1.0e-6, "WFC") ;
      RANGECHECK(pParam->HSMHV2_cgso,       0.0, 100e-9 * C_VAC*model->HSMHV2_kappa/model->HSMHV2_tox, "CGSO") ;
      RANGECHECK(pParam->HSMHV2_cgdo,       0.0, 100e-9 * C_VAC*model->HSMHV2_kappa/model->HSMHV2_tox, "CGDO") ;
      RANGECHECK(pParam->HSMHV2_ibpc1,      0.0,   1.0e12, "IBPC1") ;
      RANGECHECK(pParam->HSMHV2_ibpc2,      0.0,   1.0e12, "IBPC2") ;
      RANGECHECK(pParam->HSMHV2_cvdsover,   0.0,      1.0, "CVDSOVER") ;
      RANGECHECK(pParam->HSMHV2_nsti,    1.0e16,   1.0e19, "NSTI") ;
      MINCHECK(  pParam->HSMHV2_cgbo,       0.0,           "CGBO") ;
      RANGECHECK(pParam->HSMHV2_npext,   1.0e16,   1.0e18, "NPEXT") ;
      RANGECHECK(pParam->HSMHV2_rd,         0.0,  100.0e-3, "RD") ;
      RANGECHECK(pParam->HSMHV2_rd22,      -5.0,      0.0, "RD22") ;
      RANGECHECK(pParam->HSMHV2_rd23,       0.0,      2.0, "RD23") ;
      RANGECHECK(pParam->HSMHV2_rd24,       0.0,      0.1, "RD24") ;
      RANGECHECK(pParam->HSMHV2_rdict1,   -10.0,     10.0, "RDICT1") ;
      RANGECHECK(pParam->HSMHV2_rdov13,     0.0,      1.0, "RDOV13") ;
      RANGECHECK(pParam->HSMHV2_rdslp1,   -10.0,     10.0, "RDSLP1") ;
      RANGECHECK(pParam->HSMHV2_rdvb,       0.0,      2.0, "RDVB") ;
      RANGECHECK(pParam->HSMHV2_rdvd,       0.0,      2.0, "RDVD") ;
      MINCHECK(  pParam->HSMHV2_rdvg11,     0.0,           "RDVG11") ;
      RANGECHECK(pParam->HSMHV2_rs,         0.0,  10.0e-3, "RS") ;
      RANGECHECK(pParam->HSMHV2_rth0,       0.0,     10.0, "RTH0") ;
      RANGECHECK(pParam->HSMHV2_vover,      0.0,      4.0, "VOVER") ;

      if ( model->HSMHV2_xpdv * model->HSMHV2_xldld > 1 ) {
	      here->HSMHV2_xpdv = 1/model->HSMHV2_xldld ; 
      }else { here->HSMHV2_xpdv = model->HSMHV2_xpdv; }

      here->HSMHV2_cordrift = model->HSMHV2_cordrift ;
      if ( model->HSMHV2_cordrift  && pParam->HSMHV2_nover == 0.0 ) {
        fprintf(stderr,"warning(HiSIM_HV(%s)): CORDRIFT has been inactivated when NOVER = 0.0.\n",model->HSMHV2modName);
        here->HSMHV2_cordrift = 0 ;
      }

      /*-----------------------------------------------------------*
       * Change unit into MKS for instance parameters.
       *-----------------*/

      hereMKS->HSMHV2_nsubcdfm  = here->HSMHV2_nsubcdfm / C_cm2m_p3 ;
      hereMKS->HSMHV2_subld2    = here->HSMHV2_subld2   * C_m2cm ;

      pParam->HSMHV2_nsubc      = pParam->HSMHV2_nsubc  / C_cm2m_p3 ;
      pParam->HSMHV2_nsubp      = pParam->HSMHV2_nsubp  / C_cm2m_p3 ;
      pParam->HSMHV2_nsti       = pParam->HSMHV2_nsti   / C_cm2m_p3 ;
      pParam->HSMHV2_nover      = pParam->HSMHV2_nover  / C_cm2m_p3 ;
      pParam->HSMHV2_novers     = pParam->HSMHV2_novers / C_cm2m_p3 ;
      pParam->HSMHV2_nsubpsti1  = pParam->HSMHV2_nsubpsti1 / C_m2cm ;
      pParam->HSMHV2_muesti1    = pParam->HSMHV2_muesti1 / C_m2cm ;
      pParam->HSMHV2_ndep       = pParam->HSMHV2_ndep / C_m2cm ;
      pParam->HSMHV2_ninv       = pParam->HSMHV2_ninv / C_m2cm ;

      pParam->HSMHV2_vmax       = pParam->HSMHV2_vmax   / C_m2cm ;
      pParam->HSMHV2_wfc        = pParam->HSMHV2_wfc    * C_m2cm_p2 ;
      pParam->HSMHV2_glksd1     = pParam->HSMHV2_glksd1 / C_m2cm ;
      pParam->HSMHV2_glksd2     = pParam->HSMHV2_glksd2 * C_m2cm ;
      pParam->HSMHV2_gleak2     = pParam->HSMHV2_gleak2 * C_m2cm ;
      pParam->HSMHV2_glkb2      = pParam->HSMHV2_glkb2  * C_m2cm ;
      pParam->HSMHV2_fn2        = pParam->HSMHV2_fn2    * C_m2cm ;
      pParam->HSMHV2_gidl1      = pParam->HSMHV2_gidl1  / C_m2cm_p1o2 ;
      pParam->HSMHV2_gidl2      = pParam->HSMHV2_gidl2  * C_m2cm ;
      pParam->HSMHV2_nfalp      = pParam->HSMHV2_nfalp  / C_m2cm ;
      pParam->HSMHV2_nftrp      = pParam->HSMHV2_nftrp  * C_m2cm_p2 ;

      pParam->HSMHV2_npext      = pParam->HSMHV2_npext     / C_cm2m_p3 ;
      pParam->HSMHV2_rd22       = pParam->HSMHV2_rd22      / C_m2cm ;
      pParam->HSMHV2_rd23       = pParam->HSMHV2_rd23      / C_m2cm ;
      pParam->HSMHV2_rd24       = pParam->HSMHV2_rd24      / C_m2cm ;
      pParam->HSMHV2_rdvd       = pParam->HSMHV2_rdvd      / C_m2cm ;
      pParam->HSMHV2_rth0       = pParam->HSMHV2_rth0      / C_m2cm ;

      pParam->HSMHV2_vfbover    = -pParam->HSMHV2_vfbover ; /* for Backword Compitibility */

    } /* instance */



    /*-----------------------------------------------------------*
     * Range check of model parameters
     *-----------------*/
     RANGECHECK(model->HSMHV2_shemax   ,    300,     600, "SHEMAX");
     RANGECHECK(model->HSMHV2_cvbd     ,   -0.1,     0.2, "CVBD");
     RANGECHECK(model->HSMHV2_cvbs     ,   -0.1,     0.2, "CVBS");
    if ( model->HSMHV2_tox <= 0 && model->HSMHV2_coerrrep ) {
      printf("warning(HiSIM_HV(%s)): TOX = %e\n ", model->HSMHV2modName,model->HSMHV2_tox);
      printf("warning(HiSIM_HV(%s)): The model parameter TOX must be positive.\n",model->HSMHV2modName);
    }
    RANGECHECK(model->HSMHV2_xld,        0.0,  50.0e-9, "XLD") ;
    RANGECHECK(model->HSMHV2_xwd,  -100.0e-9, 300.0e-9, "XWD") ;
    RANGECHECK(model->HSMHV2_xwdc,  -10.0e-9, 100.0e-9, "XWDC") ;
    RANGECHECK(model->HSMHV2_rsh,        0.0,    500.0, "RSH") ;
    RANGECHECK(model->HSMHV2_rshg,       0.0,    100.0, "RSHG") ;
    if(model->HSMHV2_xqy != 0.0) RANGECHECK(model->HSMHV2_xqy,        10.0e-9,  50.0e-9, "XQY") ;
    MINCHECK  (model->HSMHV2_xqy1,       0.0,           "XQY1") ;
    MINCHECK  (model->HSMHV2_xqy2,       0.0,           "XQY2") ;
    RANGECHECK(model->HSMHV2_vbi,        1.0,      1.2, "VBI") ;
    RANGECHECK(model->HSMHV2_parl2,      0.0,  50.0e-9, "PARL2") ;
    RANGECHECK(model->HSMHV2_lp,         0.0, 300.0e-9, "LP") ;
    RANGECHECK(model->HSMHV2_pgd2,       0.0,      1.5, "PGD2") ;
    RANGECHECK(model->HSMHV2_pgd4,       0.0,      3.0, "PGD4") ;
    RANGECHECK(model->HSMHV2_mueph0,    0.25,     0.35, "MUEPH0") ;
    RANGECHECK(model->HSMHV2_muesr0,     1.8,      2.2, "MUESR0") ;
    RANGECHECK(model->HSMHV2_lpext,  1.0e-50,  10.0e-6, "LPEXT") ;
    MINCHECK  (model->HSMHV2_sc4,        0.0,           "SC4") ;
    RANGECHECK(model->HSMHV2_scp21,      0.0,      5.0, "SCP21") ;
    RANGERESET(model->HSMHV2_scp22,      0.0,      0.0, "SCP22") ;
    RANGECHECK(model->HSMHV2_bs1,        0.0,  50.0e-3, "BS1") ;
    RANGECHECK(model->HSMHV2_bs2,        0.5,      1.0, "BS2") ;
    MINCHECK  (model->HSMHV2_ptl,        0.0,           "PTL") ;
    RANGECHECK(model->HSMHV2_ptp,        3.0,      4.0, "PTP") ;
    MINCHECK  (model->HSMHV2_pt2,        0.0,           "PT2") ;
    MINCHECK  (model->HSMHV2_pt4,        0.0,           "PT4") ;
    MINCHECK  (model->HSMHV2_pt4p,       0.0,           "PT4P") ;
    RANGECHECK(model->HSMHV2_gdl,        0.0,   220e-9, "GDL") ;
    MINCHECK  (model->HSMHV2_ninvd,      0.0,           "NINVD") ;
    MINCHECK  (model->HSMHV2_ninvdw,     0.0,           "NINVDW") ;
    MINCHECK  (model->HSMHV2_ninvdwp,    0.0,           "NINVDWP") ;
    MINCHECK  (model->HSMHV2_ninvdt1,    0.0,           "NINVDT1") ;
    MINCHECK  (model->HSMHV2_ninvdt2,    0.0,           "NINVDT2") ;
    RANGECHECK(model->HSMHV2_clm5,       0.0,      2.0, "CLM5") ;
    RANGECHECK(model->HSMHV2_clm6,       0.0,     20.0, "CLM6") ;
    RANGECHECK(model->HSMHV2_sub2l,      0.0,      1.0, "SUB2L") ;
    RANGECHECK(model->HSMHV2_voverp,     0.0,      2.0, "VOVERP") ;
    RANGECHECK(model->HSMHV2_qme1,       0.0,     1e-9, "QME1") ;
    RANGECHECK(model->HSMHV2_qme2,       1.0,      3.0, "QME2") ;
    RANGECHECK(model->HSMHV2_qme3,       0.0,  500e-12, "QME3") ;
    RANGECHECK(model->HSMHV2_glpart1,    0.0,      1.0, "GLPART1") ;
    RANGECHECK(model->HSMHV2_tnom,      22.0,     32.0, "TNOM") ;
    RANGECHECK(model->HSMHV2_ddltmax,    1.0,     10.0, "DDLTMAX") ;
    RANGECHECK(model->HSMHV2_ddltict,   -3.0,     20.0, "DDLTICT") ;
    RANGECHECK(model->HSMHV2_ddltslp,    0.0,     20.0, "DDLTSLP") ;
    RANGECHECK(model->HSMHV2_mphdfm,    -3.0,      3.0, "MPHDFM") ;
    RANGECHECK(model->HSMHV2_cvb,       -0.1,      0.2, "CVB") ;
    RANGECHECK(model->HSMHV2_rd20,       0.0,     30.0, "RD20") ;
    RANGECHECK(model->HSMHV2_rd21,       0.0,      1.0, "RD21") ;
    RANGECHECK(model->HSMHV2_rd22d,      0.0,      2.0, "RD22D") ;
    MINCHECK(  model->HSMHV2_rd25,       0.0,           "RD25") ;
    RANGECHECK(model->HSMHV2_rdtemp1,  -1e-3,     2e-2, "RDTEMP1") ;
    RANGECHECK(model->HSMHV2_rdtemp2,  -1e-5,     1e-5, "RDTEMP2") ;
    RANGECHECK(model->HSMHV2_rdvdtemp1,-1e-3,     1e-2, "RDVDTEMP1") ;
    RANGECHECK(model->HSMHV2_rdvdtemp2,-1e-5,     1e-5, "RDVDTEMP2") ;
    MINCHECK(  model->HSMHV2_rdvg12,     0.0,           "RDVG12") ;
    RANGECHECK(model->HSMHV2_rthtemp1,  -1.0,      1.0, "RTHTEMP1") ;
    RANGECHECK(model->HSMHV2_rthtemp2,  -1.0,      1.0, "RTHTEMP2") ;
    RANGECHECK(model->HSMHV2_rth0w,     -100,      100, "RTH0W") ;
    RANGECHECK(model->HSMHV2_rth0wp,     -10,       10, "RTH0WP") ;
    RANGECHECK(model->HSMHV2_rth0nf,    -5.0,      5.0, "RTH0NF") ;
    RANGECHECK(model->HSMHV2_powrat,     0.0,      1.0, "POWRAT") ;
    RANGECHECK(model->HSMHV2_prattemp1, -1.0,      1.0, "PRATTEMP1") ;
    RANGECHECK(model->HSMHV2_prattemp2, -1.0,      1.0, "PRATTEMP2") ;
    MINRESET ( model->HSMHV2_xldld,      0.0,           "XLDLD") ;
    MINCHECK(  model->HSMHV2_loverld,    0.0,           "LOVERLD") ;
    MINCHECK(  model->HSMHV2_lovers,     0.0,           "LOVERS") ;
    MINCHECK(  model->HSMHV2_lover,      0.0,           "LOVER") ;
    MINCHECK(  model->HSMHV2_ldrift1,    0.0,           "LDRIFT1") ;
    MINCHECK(  model->HSMHV2_ldrift1s,   0.0,           "LDRIFT1S") ;
    MINCHECK(  model->HSMHV2_ldrift2,    0.0,           "LDRIFT2") ;
    MINCHECK(  model->HSMHV2_ldrift2s,   0.0,           "LDRIFT2S") ;
//    MINCHECK(  model->HSMHV2_ldrift,     0.0,           "LDRIFT") ;
    RANGECHECK(model->HSMHV2_rds,       -100,      100, "RDS") ;
    RANGECHECK(model->HSMHV2_rdsp,       -10,       10, "RDSP") ;
    RANGECHECK(model->HSMHV2_rdvdl,     -100,      100, "RDVDL") ;
    RANGECHECK(model->HSMHV2_rdvdlp,     -10,       10, "RDVDLP") ;
    RANGECHECK(model->HSMHV2_rdvds,     -100,      100, "RDVDS") ;
    RANGECHECK(model->HSMHV2_rdvdsp,     -10,       10, "RDVDSP") ;
    RANGECHECK(model->HSMHV2_rd23l,     -100,      100, "RD23L") ;
    RANGECHECK(model->HSMHV2_rd23lp,     -10,       10, "RD23LP") ;
    RANGECHECK(model->HSMHV2_rd23s,     -100,      100, "RD23S") ;
    RANGECHECK(model->HSMHV2_rd23sp,     -10,       10, "RD23SP") ;
    RANGECHECK(model->HSMHV2_rdov11,     0.0,       10, "RDOV11") ;
    RANGECHECK(model->HSMHV2_rdov12,     0.0,      2.0, "RDOV12") ;
    RANGECHECK(model->HSMHV2_rdslp2,   -10.0,     10.0, "RDSLP2") ;
    RANGECHECK(model->HSMHV2_rdict2,   -10.0,     10.0, "RDICT2") ;
    RANGECHECK(model->HSMHV2_rdrvmax,    1e6,    100e6, "RDRVMAX"   ) ;
    RANGECHECK(model->HSMHV2_rdrmue,     1e2,      3e3, "RDRMUE"    ) ;
    RANGECHECK(model->HSMHV2_rdrqover,   0.0,      1e7, "RDRDQOVER"  ) ;
    RANGERESET(model->HSMHV2_rdrcx,      0.0,      1.0, "RDRCX"     ) ;
    RANGECHECK(model->HSMHV2_rdrcar,     0.0,    50e-9, "RDRCAR"    ) ;
    RANGECHECK(model->HSMHV2_rdrmuetmp,  0.0,      2.0, "RDRMUETMP" ) ;
    RANGECHECK(model->HSMHV2_rdrvtmp,   -2.0,      1.0, "RDRVTMP"   ) ;
    MINCHECK(  model->HSMHV2_xpdv,       0.0,           "XPDV"      ) ;
    MINCHECK(  model->HSMHV2_xpvdth,     0.0,           "XPVDTH"    ) ;
    RANGECHECK(model->HSMHV2_xpvdthg,   -1.0,      1.0, "XPVDTHG"   ) ;
    MINCHECK(  model->HSMHV2_ibpc1l,     0.0,           "IBPC1L"    ) ;
    RANGERESET(model->HSMHV2_ndepm,      5e15,   2e17,  "NDEPM" ) ;
    RANGERESET(model->HSMHV2_tndep,      1e-7,   1e-6,  "TNDEP" ) ;
    RANGERESET(model->HSMHV2_depmue0,       1,    1e5,   "DEPMUE0" ) ;
    RANGECHECK(model->HSMHV2_depmueback0,   1,    1e5,   "DEPMUEBACK0" ) ;
    RANGECHECK(model->HSMHV2_depvdsef2,   0.1,    4.0,   "DEPVDSEF2" ) ;
    RANGECHECK(model->HSMHV2_depmueph1,     1,    1e5,   "DEPMUEPH1" ) ;
    RANGECHECK(model->HSMHV2_depleak,     0.0,    5.0,   "DEPLEAK" ) ;
    
    if( model->HSMHV2_codep == 1 && model->HSMHV2_coerrrep ) {
      if( model->HSMHV2_copprv == 1 ) {
        printf("warning(HiSIM_HV(%s)): COPPRV is not supported yet in Depletion mode mode, reset to 0.\n",model->HSMHV2modName);
      }
      if( model->HSMHV2_coisti == 1 ) { 
        printf("warning(HiSIM_HV(%s)): STI leak model is not supported yet in Depletion mode model, skipped\n",model->HSMHV2modName);
      } 
      if( model->HSMHV2_cothrml == 1 ) {
        printf("warning(HiSIM_HV(%s)): Thermal noise model is not supported yet in Depletion mode model, skipped\n",model->HSMHV2modName);
      }
      if( model->HSMHV2_coign == 1 ) {
        printf("warning(HiSIM_HV(%s)): Induced gate noise model is not supported yet in Depletion mode model, skipped\n",model->HSMHV2modName);
      }
    }
    if( model->HSMHV2_codep && model->HSMHV2_copprv == 1 ) { model->HSMHV2_copprv = 0 ; }

    if ( model->HSMHV2_xpdv * model->HSMHV2_xldld > 1 && model->HSMHV2_coerrrep ) {
	printf("warning(HiSIM_HV(%s)): The model parameter XPDV (= %e) must be smaller than 1/XLDLD (= %e).\n", 
	       model->HSMHV2modName,model->HSMHV2_xpdv, 1/model->HSMHV2_xldld ); 
      	printf("warning(HiSIM_HV(%s)): The model parameter XPDV (= %e) has been changed to %e.\n", 
	       model->HSMHV2modName,model->HSMHV2_xpdv, 1/model->HSMHV2_xldld );			
    }


    /*-----------------------------------------------------------*
     * Change units into MKS.
     *-----------------*/

     modelMKS->HSMHV2_vmax      = model->HSMHV2_vmax      / C_m2cm ;
     modelMKS->HSMHV2_ll        = model->HSMHV2_ll        / pow( C_m2cm , model->HSMHV2_lln ) ;
     modelMKS->HSMHV2_wl        = model->HSMHV2_wl        / pow( C_m2cm , model->HSMHV2_wln ) ;
     modelMKS->HSMHV2_svgsl     = model->HSMHV2_svgsl     / pow( C_m2cm , model->HSMHV2_svgslp ) ;
     modelMKS->HSMHV2_svgsw     = model->HSMHV2_svgsw     / pow( C_m2cm , model->HSMHV2_svgswp ) ;
     modelMKS->HSMHV2_svbsl     = model->HSMHV2_svbsl     / pow( C_m2cm , model->HSMHV2_svbslp ) ;
     modelMKS->HSMHV2_slgl      = model->HSMHV2_slgl      / pow( C_m2cm , model->HSMHV2_slglp ) ;
     modelMKS->HSMHV2_sub1l     = model->HSMHV2_sub1l     / pow( C_m2cm , model->HSMHV2_sub1lp ) ;
     modelMKS->HSMHV2_slg       = model->HSMHV2_slg       / C_m2cm ;
     modelMKS->HSMHV2_sub2l     = model->HSMHV2_sub2l     / C_m2cm ;
     modelMKS->HSMHV2_subld2    = model->HSMHV2_subld2    * C_m2cm ;
     modelMKS->HSMHV2_rdtemp1   = model->HSMHV2_rdtemp1   / C_m2cm ;
     modelMKS->HSMHV2_rdtemp2   = model->HSMHV2_rdtemp2   / C_m2cm ;
     modelMKS->HSMHV2_rdvdtemp1 = model->HSMHV2_rdvdtemp1 / C_m2cm ;
     modelMKS->HSMHV2_rdvdtemp2 = model->HSMHV2_rdvdtemp2 / C_m2cm ;
     modelMKS->HSMHV2_nsubsub   = model->HSMHV2_nsubsub   / C_cm2m_p3 ;
     modelMKS->HSMHV2_nsubpsti1 = model->HSMHV2_nsubpsti1 / C_m2cm ;
     modelMKS->HSMHV2_muesti1   = model->HSMHV2_muesti1 / C_m2cm ;
     modelMKS->HSMHV2_wfc       = model->HSMHV2_wfc       * C_m2cm_p2 ;
     modelMKS->HSMHV2_glksd1    = model->HSMHV2_glksd1    / C_m2cm ;
     modelMKS->HSMHV2_glksd2    = model->HSMHV2_glksd2    * C_m2cm ;
     modelMKS->HSMHV2_glksd3    = model->HSMHV2_glksd3    * C_m2cm ;
     modelMKS->HSMHV2_gleak2    = model->HSMHV2_gleak2    * C_m2cm ;
     modelMKS->HSMHV2_gleak4    = model->HSMHV2_gleak4    * C_m2cm ;
     modelMKS->HSMHV2_gleak5    = model->HSMHV2_gleak5    * C_m2cm ;
     modelMKS->HSMHV2_gleak7    = model->HSMHV2_gleak7    / C_m2cm_p2 ;
     modelMKS->HSMHV2_glkb2     = model->HSMHV2_glkb2     * C_m2cm ;
     modelMKS->HSMHV2_fn2       = model->HSMHV2_fn2       * C_m2cm ;
     modelMKS->HSMHV2_gidl1     = model->HSMHV2_gidl1     / C_m2cm_p1o2 ;
     modelMKS->HSMHV2_gidl2     = model->HSMHV2_gidl2     * C_m2cm ;
     modelMKS->HSMHV2_nfalp     = model->HSMHV2_nfalp     / C_m2cm ;
     modelMKS->HSMHV2_nftrp     = model->HSMHV2_nftrp     * C_m2cm_p2 ;
     modelMKS->HSMHV2_cit       = model->HSMHV2_cit       * C_m2cm_p2 ;
     modelMKS->HSMHV2_ovslp     = model->HSMHV2_ovslp     / C_m2cm ;
     modelMKS->HSMHV2_dly3      = model->HSMHV2_dly3      / C_m2cm_p2 ;
     modelMKS->HSMHV2_cth0      = model->HSMHV2_cth0      * C_m2cm ;
     modelMKS->HSMHV2_rdrmue    = model->HSMHV2_rdrmue    / C_m2cm_p2 ;
     modelMKS->HSMHV2_rdrvmax   = model->HSMHV2_rdrvmax   / C_m2cm ;
     modelMKS->HSMHV2_ndepm     = model->HSMHV2_ndepm     / C_cm2m_p3 ;
     modelMKS->HSMHV2_depvmax   = model->HSMHV2_depvmax   / C_m2cm ;


    /*-----------------------------------------------------------*
     * Change unit into Kelvin.
     *-----------------*/
     model->HSMHV2_ktnom =  model->HSMHV2_tnom + 273.15 ; /* [C] -> [K] */


  } /* model */


  return(OK);
} 

int
HSMHV2unsetup(
    GENmodel *inModel,
    CKTcircuit *ckt)
{
#ifndef HAS_BATCHSIM
    HSMHV2model *model;
    HSMHV2instance *here;
 
    for (model = (HSMHV2model *)inModel; model != NULL;
            model = HSMHV2nextModel(model))
    {
        for (here = HSMHV2instances(model); here != NULL;
                here=HSMHV2nextInstance(here))
        {
            if (here->HSMHV2tempNode > 0 &&
                here->HSMHV2tempNode != here->HSMHV2tempNodeExt &&
                here->HSMHV2tempNode != here->HSMHV2subNodeExt)
                CKTdltNNum(ckt, here->HSMHV2tempNode);
            here->HSMHV2tempNode = 0;

            here->HSMHV2subNode = 0;

            if (here->HSMHV2qbNode > 0)
                CKTdltNNum(ckt, here->HSMHV2qbNode);
            here->HSMHV2qbNode = 0;

            if (here->HSMHV2qiNode > 0)
                CKTdltNNum(ckt, here->HSMHV2qiNode);
            here->HSMHV2qiNode = 0;

            if (here->HSMHV2sbNode > 0
                    && here->HSMHV2sbNode != here->HSMHV2bNode)
                CKTdltNNum(ckt, here->HSMHV2sbNode);
            here->HSMHV2sbNode = 0;

            if (here->HSMHV2bNodePrime > 0
                    && here->HSMHV2bNodePrime != here->HSMHV2bNode)
                CKTdltNNum(ckt, here->HSMHV2bNodePrime);
            here->HSMHV2bNodePrime = 0;

            if (here->HSMHV2dbNode > 0
                    && here->HSMHV2dbNode != here->HSMHV2bNode)
                CKTdltNNum(ckt, here->HSMHV2dbNode);
            here->HSMHV2dbNode = 0;

            if (here->HSMHV2gNodePrime > 0
                    && here->HSMHV2gNodePrime != here->HSMHV2gNode)
                CKTdltNNum(ckt, here->HSMHV2gNodePrime);
            here->HSMHV2gNodePrime = 0;

            if (here->HSMHV2sNodePrime > 0
                    && here->HSMHV2sNodePrime != here->HSMHV2sNode)
                CKTdltNNum(ckt, here->HSMHV2sNodePrime);
            here->HSMHV2sNodePrime = 0;

            if (here->HSMHV2dNodePrime > 0
                    && here->HSMHV2dNodePrime != here->HSMHV2dNode)
                CKTdltNNum(ckt, here->HSMHV2dNodePrime);
            here->HSMHV2dNodePrime = 0;
        }
    }
#endif
    return OK;
}
