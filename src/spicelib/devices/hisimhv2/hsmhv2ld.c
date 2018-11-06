/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV
 ( VERSION : 2  SUBVERSION : 2  REVISION : 0 )
 Model Parameter 'VERSION' : 2.20
 FILE : hsmhvld.c

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
#include "hisimhv2.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"

#define SHOW_EPS_QUANT 1.0e-15

static void ShowPhysVals
(
 HSMHV2instance *here,
 HSMHV2model *model,
 int isFirst,
 double vds,
 double vgs,
 double vbs,
 double vgd,
 double vbd,
 double vgb
 )
{

  NG_IGNORE(vgd);
  NG_IGNORE(vbd);

  /*                                                            */
  /*                                                            */
  /* Note: This function is not yet adapted to the flat version */
  /*                                                            */
  /*                                                            */

  /* regard the epsilon-quantity as 0.0 */
  vds = (fabs(vds) < SHOW_EPS_QUANT) ? 0.0 : vds;
  vgs = (fabs(vgs) < SHOW_EPS_QUANT) ? 0.0 : vgs;
  vbs = (fabs(vbs) < SHOW_EPS_QUANT) ? 0.0 : vbs;
  vgb = (fabs(vgb) < SHOW_EPS_QUANT) ? 0.0 : vgb;
  switch (model->HSMHV2_show) {
  case 1:
    if (isFirst) printf("Vds        Ids\n");
    printf("%e %e\n", model->HSMHV2_type*vds, here->HSMHV2_mode*here->HSMHV2_ids);
    break;
  case 2:
    if (isFirst) printf("Vgs        Ids\n");
    printf("%e %e\n", model->HSMHV2_type*vgs, here->HSMHV2_mode*here->HSMHV2_ids);
    break;
  case 3:
    if (isFirst) printf("Vgs        log10(|Ids|)\n");
    printf("%e %e\n", model->HSMHV2_type*vgs, log10(here->HSMHV2_ids));
    break;
  case 4:
    if (isFirst) printf("log10(|Ids|)    gm/|Ids|\n");
    if (here->HSMHV2_ids == 0.0)
      printf("I can't show gm/Ids - log10(Ids), because Ids = 0.\n");
    else
      printf("%e %e\n",  log10(here->HSMHV2_ids), here->HSMHV2_gm/here->HSMHV2_ids);
    break;
  case 5:
    if (isFirst) printf("Vds        gds\n");
    printf("%e %e\n", model->HSMHV2_type*vds, here->HSMHV2_gds);
    break;
  case 6:
    if (isFirst) printf("Vgs        gm\n");
    printf("%e %e\n", model->HSMHV2_type*vgs, here->HSMHV2_gm);
    break;
  case 7:
    if (isFirst) printf("Vbs        gbs\n");
    printf("%e %e\n", model->HSMHV2_type*vbs, here->HSMHV2_gmbs);
    break;
  case 8:
    if (isFirst) printf("Vgs        Cgg\n");
    printf("%e %e\n", model->HSMHV2_type*vgs, here->HSMHV2_cggb);
    break;
  case 9:
    if (isFirst) printf("Vgs        Cgs\n");
    printf("%e %e\n", model->HSMHV2_type*vgs, here->HSMHV2_cgsb);
    break;
  case 10:
    if (isFirst) printf("Vgs        Cgd\n");
    printf("%e %e\n", model->HSMHV2_type*vgs, here->HSMHV2_cgdb);
    break;
  case 11:
    if (isFirst) printf("Vgs        Cgb\n");
    printf("%e %e\n", model->HSMHV2_type*vgs, -(here->HSMHV2_cggb+here->HSMHV2_cgsb+here->HSMHV2_cgdb));
    break;
  case 12:
    if (isFirst) printf("Vds        Csg\n");
    printf("%e %e\n", model->HSMHV2_type*vds, -(here->HSMHV2_cggb+here->HSMHV2_cbgb+here->HSMHV2_cdgb));
    break;
  case 13:
    if (isFirst) printf("Vds        Cdg\n");
    printf("%e %e\n", model->HSMHV2_type*vds, here->HSMHV2_cdgb);
    break;
  case 14:
    if (isFirst) printf("Vds        Cbg\n");
    printf("%e %e\n", model->HSMHV2_type*vds, here->HSMHV2_cbgb);
    break;
  case 15:
    if (isFirst) printf("Vds        Cgg\n");
    printf("%e %e\n", model->HSMHV2_type*vds, here->HSMHV2_cggb);
    break;
  case 16:
    if (isFirst) printf("Vds        Cgs\n");
    printf("%e %e\n", model->HSMHV2_type*vds, here->HSMHV2_cgsb);
    break;
  case 17:
    if (isFirst) printf("Vds        Cgd\n");
    printf("%e %e\n", model->HSMHV2_type*vds, here->HSMHV2_cgdb);
    break;
  case 18:
    if (isFirst) printf("Vds        Cgb\n");
    printf("%e %e\n", model->HSMHV2_type*vds, -(here->HSMHV2_cggb+here->HSMHV2_cgsb+here->HSMHV2_cgdb));
    break;
  case 19:
    if (isFirst) printf("Vgs        Csg\n");
    printf("%e %e\n", model->HSMHV2_type*vgs, -(here->HSMHV2_cggb+here->HSMHV2_cbgb+here->HSMHV2_cdgb));
    break;
  case 20:
    if (isFirst) printf("Vgs        Cdg\n");
    printf("%e %e\n", model->HSMHV2_type*vgs, here->HSMHV2_cdgb);
    break;
  case 21:
    if (isFirst) printf("Vgs        Cbg\n");
    printf("%e %e\n", model->HSMHV2_type*vgs, here->HSMHV2_cbgb);
    break;
  case 22:
    if (isFirst) printf("Vgb        Cgb\n");
    printf("%e %e\n", model->HSMHV2_type*vgb, -(here->HSMHV2_cggb+here->HSMHV2_cgsb+here->HSMHV2_cgdb));
    break;
  case 50:
    if (isFirst) printf("Vgs  Vds  Vbs  Vgb  Ids  log10(|Ids|)  gm/|Ids|  gm  gds  gbs  Cgg  Cgs  Cgb  Cgd  Csg  Cbg  Cdg\n");
    printf("%e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n",
            model->HSMHV2_type*vgs, model->HSMHV2_type*vds, model->HSMHV2_type*vbs, model->HSMHV2_type*vgb, here->HSMHV2_mode*here->HSMHV2_ids,
            log10(here->HSMHV2_ids), here->HSMHV2_gm/here->HSMHV2_ids, here->HSMHV2_gm, here->HSMHV2_gds, here->HSMHV2_gmbs, here->HSMHV2_cggb,
            here->HSMHV2_cgsb, -(here->HSMHV2_cggb+here->HSMHV2_cgsb+here->HSMHV2_cgdb), here->HSMHV2_cgdb,
            -(here->HSMHV2_cggb+here->HSMHV2_cbgb+here->HSMHV2_cdgb), here->HSMHV2_cbgb, here->HSMHV2_cdgb);
    break;
  default:
    /*
      printf("There is no physical value corrsponding to %d\n", flag);
    */
    break;
  }
}

int HSMHV2load(
     GENmodel *inModel,
     CKTcircuit *ckt)
     /* actually load the current value into the
      * sparse matrix previously provided
      */
{
  HSMHV2model *model = (HSMHV2model*)inModel;
  HSMHV2instance *here;
  HSMHV2binningParam *pParam;
  HSMHV2modelMKSParam *modelMKS ;
  HSMHV2hereMKSParam  *hereMKS ;



  /* node voltages */
#define XDIM 14
  double x[XDIM] ;

  /* branch voltages */
  double vbd=0.0,  vbs=0.0,  vds=0.0,  vgb=0.0, vgd=0.0, vgdo=0.0, vgs=0.0 ;
  double vdbs=0.0, vsbs=0.0, vdbd=0.0 ;
  double vges=0.0, vged=0.0, vgedo=0.0 ;
  double vbs_jct=0.0, vbd_jct=0.0;
  double deltemp = 0.0 , deltemp_old = 0.0 ;
  double vggp=0, vddp=0, vssp=0, vbpb=0, vbpsb=0, vbpdb=0 ;
  double vdse=0.0, vgse=0.0, vbse=0.0 ;
  double vsubs=0.0 ; /* substrate bias */

  double ivds=0.0,  ivgs=0.0,  ivbs=0.0 ;
  double ivgse=0.0, ivdse=0.0, ivbse=0.0 ;

  /* delta of branch voltages */
  double delvbs=0.0, delvds=0.0, delvgs=0.0, delvsbs=0.0, delvdbd=0.0, deldeltemp = 0.0 ;
  double delvdse=0.0, delvgse=0.0, delvbse=0.0, delvddp=0.0, delvssp=0.0, delvggp=0.0, delvbpb=0.0, delvbpdb=0.0, delvbpsb =0.0 ;
  double delvsubs=0.0; /* substrate bias change */

  /* branch currents */
  double Ids=0.0, gds=0.0,     gm=0.0,     gmbs=0.0,    gmT=0.0,
                  gds_ext=0.0, gm_ext=0.0, gmbs_ext=0.0 ;
  double Igd=0.0, dIgd_dVd=0.0, dIgd_dVg=0.0, dIgd_dVb=0.0, dIgd_dVs=0.0, dIgd_dT=0.0 ;
  double Igs=0.0, dIgs_dVd=0.0, dIgs_dVg=0.0, dIgs_dVb=0.0, dIgs_dVs=0.0, dIgs_dT=0.0 ;
  double Igb=0.0, dIgb_dVd=0.0, dIgb_dVg=0.0, dIgb_dVb=0.0, dIgb_dVs=0.0, dIgb_dT=0.0 ;
  double Isub=0.0, dIsub_dVds=0.0, dIsub_dVgs=0.0, dIsub_dVbs=0.0, dIsub_dT=0.0 ;
  double Isubs=0.0, dIsubs_dVds=0.0, dIsubs_dVgs=0.0, dIsubs_dVbs=0.0, dIsubs_dT=0.0 ;
  double IsubLD=0.0, dIsubLD_dVds=0.0, dIsubLD_dVgs=0.0, dIsubLD_dVbs=0.0, dIsubLD_dT=0.0 ;
  double IsubLDs=0.0, dIsubLDs_dVds=0.0, dIsubLDs_dVgs=0.0, dIsubLDs_dVbs=0.0, dIsubLDs_dT=0.0 ;
  double dIsubLD_dVddp=0.0, dIsubLDs_dVddp=0.0 ;
  double IdsIBPC=0.0, dIdsIBPC_dVds=0.0, dIdsIBPC_dVgs=0.0, dIdsIBPC_dVbs=0.0, dIdsIBPC_dT=0.0 ;
  double IdsIBPCs=0.0, dIdsIBPCs_dVds=0.0, dIdsIBPCs_dVgs=0.0, dIdsIBPCs_dVbs=0.0, dIdsIBPCs_dT=0.0 ;
  double dIdsIBPC_dVddp=0.0, dIdsIBPCs_dVddp=0.0 ;
  double Igidl=0.0, dIgidl_dVds=0.0, dIgidl_dVgs=0.0, dIgidl_dVbs=0.0, dIgidl_dT=0.0 ;
  double Igisl=0.0, dIgisl_dVds=0.0, dIgisl_dVgs=0.0, dIgisl_dVbs=0.0, dIgisl_dT=0.0 ;
  double Ibd=0.0, Gbd=0.0, Gbdt=0.0 ;
  double Ibs=0.0, Gbs=0.0, Gbst=0.0 ;
  double Iddp=0.0, dIddp_dVddp=0.0, dIddp_dVdse=0.0, dIddp_dVgse=0.0, dIddp_dVbse=0.0, dIddp_dVsubs=0.0, dIddp_dT =0.0 , dIddp_dVds =0.0, dIddp_dVgs =0.0, dIddp_dVbs =0.0 ;
  double Issp=0.0, dIssp_dVssp=0.0, dIssp_dVdse=0.0, dIssp_dVgse=0.0, dIssp_dVbse=0.0, dIssp_dVsubs=0.0, dIssp_dT =0.0 ;
  double Iggp=0.0, dIggp_dVggp =0.0 ;
  double Ibpb=0.0, dIbpb_dVbpb =0.0 ;
  double Ibpdb=0.0, dIbpdb_dVbpdb =0.0 ;
  double Ibpsb=0.0, dIbpsb_dVbpsb =0.0 ;
  double Ith=0.0,   dIth_dT =0.0 ;

  /* displacement currents */
  double cq_d=0.0,  cq_dP=0.0,  cq_g=0.0,  cq_gP=0.0,  cq_s=0.0,  cq_sP=0.0,  cq_bP=0.0,  cq_b=0.0,  cq_db=0.0,    cq_sb=0.0,  cq_t=0.0  ;
  double cq_dE=0.0, cq_gE=0.0,  cq_sE=0.0,  cq_bE=0.0 ;

  /* node currents */
  double cur_d=0.0, cur_dP=0.0, cur_g=0.0, cur_gP=0.0, cur_s=0.0, cur_sP=0.0, cur_bP=0.0, cur_b=0.0, cur_db=0.0,   cur_sb=0.0, cur_t=0.0 ;
  double i_d=0.0,   i_dP=0.0,   i_g=0.0,   i_gP=0.0,   i_s=0.0,   i_sP=0.0,   i_bP=0.0,   i_b=0.0,   i_db=0.0,     i_sb=0.0,   i_t=0.0   ;

  /* resistances and conductances */
  double Rd=0.0, dRd_dVdse=0.0, dRd_dVgse=0.0, dRd_dVbse=0.0, dRd_dVsubs=0.0, dRd_dT=0.0, dRd_dVddp=0.0, dRd_dVds=0.0, dRd_dVgs=0.0, dRd_dVbs=0.0 ;
  double Rs=0.0, dRs_dVdse=0.0, dRs_dVgse=0.0, dRs_dVbse=0.0, dRs_dVsubs=0.0, dRs_dT=0.0 ;

  double GD=0.0, GD_dVds=0.0, GD_dVgs=0.0, GD_dVbs=0.0, GD_dVsubs=0.0, GD_dT=0.0, GD_dVddp=0.0, GD_dVdse=0.0, GD_dVgse=0.0, GD_dVbse=0.0 ;
  double GS=0.0, GS_dVds=0.0, GS_dVgs=0.0, GS_dVbs=0.0, GS_dVsubs=0.0, GS_dT=0.0 ;
  double Gth=0.0 ;
  double GG=0.0,  GRBPD=0.0, GRBPS=0.0, GRBPB=0.0;

  /* charges */
  double Qd=0.0, dQd_dVds=0.0, dQd_dVgs=0.0, dQd_dVbs=0.0, dQd_dT=0.0 ;
  double Qg=0.0, dQg_dVds=0.0, dQg_dVgs=0.0, dQg_dVbs=0.0, dQg_dT=0.0 ;
  double Qs=0.0, dQs_dVds=0.0, dQs_dVgs=0.0, dQs_dVbs=0.0, dQs_dT=0.0 ;
  double Qb=0.0, dQb_dVds=0.0, dQb_dVgs=0.0, dQb_dVbs=0.0, dQb_dT=0.0 ;
  double Qbd=0.0, Cbd=0.0, Cbdt=0.0,
         Qbs=0.0, Cbs=0.0, Cbst=0.0 ;
  double Qth=0.0, Cth =0.0 ;
  double Qfd=0.0, dQfd_dVdse=0.0, dQfd_dVgse=0.0, dQfd_dVbse=0.0, dQfd_dT=0.0 ;
  double Qfs=0.0, dQfs_dVdse=0.0, dQfs_dVgse=0.0, dQfs_dVbse=0.0, dQfs_dT=0.0 ;

  double Qdext=0.0, dQdext_dVdse=0.0, dQdext_dVgse=0.0, dQdext_dVbse=0.0, dQdext_dT=0.0 ;
  double Qgext=0.0, dQgext_dVdse=0.0, dQgext_dVgse=0.0, dQgext_dVbse=0.0, dQgext_dT=0.0 ;
  double Qsext=0.0, dQsext_dVdse=0.0, dQsext_dVgse=0.0, dQsext_dVbse=0.0, dQsext_dT=0.0 ;
  double Qbext=0.0, dQbext_dVdse=0.0, dQbext_dVgse=0.0, dQbext_dVbse=0.0, dQbext_dT=0.0 ;

  /* 5th substrate node */
  int flg_subNode = 0 ;

  /* self heating */
  double Veffpower=0.0, dVeffpower_dVds=0.0, dVeffpower_dVdse =0.0 ;
  double P=0.0, dP_dVds=0.0,  dP_dVgs=0.0,  dP_dVbs=0.0, dP_dT =0.0,
                dP_dVdse=0.0, dP_dVgse=0.0, dP_dVbse =0.0 ;
  double T0 , T1 , T2 ;

#define SHE_MAX_dlt 0.1

#define C_RTH_MIN 1.0e-4

  double TMF1 , TMF2  ;
/*---------------------------------------------------*
* smoothUpper: ceiling.
*      y = xmax - 0.5 ( arg + sqrt( arg^2 + 4 xmax delta ) )
*    arg = xmax - x - delta
*-----------------*/

#define Fn_SU( y , x , xmax , delta , dx ) { \
    TMF1 = ( xmax ) - ( x ) - ( delta ) ; \
    TMF2 = 4.0 * ( xmax ) * ( delta) ; \
    TMF2 = TMF2 > 0.0 ?  TMF2 :  -( TMF2 ) ; \
    TMF2 = sqrt ( TMF1 *  TMF1 + TMF2 ) ; \
    dx = 0.5 * ( 1.0 + TMF1 / TMF2 ) ; \
    y = ( xmax ) - 0.5 * ( TMF1 + TMF2 ) ; \
  }

  /* NQS related variables */
  int flg_nqs =0 ;
  double Qi_nqs=0.0, Qb_nqs=0.0, delQi_nqs=0.0, delQb_nqs=0.0, i_qi=0.0, i_qb=0.0, cq_qi=0.0, cq_qb=0.0, cur_qi=0.0, cur_qb =0.0 ;
  double Iqi_nqs=0.0, dIqi_nqs_dVds=0.0, dIqi_nqs_dVgs=0.0, dIqi_nqs_dVbs=0.0, dIqi_nqs_dT=0.0, dIqi_nqs_dQi_nqs =0.0 ;
  double Iqb_nqs=0.0, dIqb_nqs_dVds=0.0, dIqb_nqs_dVgs=0.0, dIqb_nqs_dVbs=0.0, dIqb_nqs_dT=0.0, dIqb_nqs_dQb_nqs =0.0 ;
  double Qd_nqs=0.0,  dQd_nqs_dVds=0.0,  dQd_nqs_dVgs=0.0,  dQd_nqs_dVbs=0.0,  dQd_nqs_dT=0.0,  dQd_nqs_dQi_nqs  =0.0 ;
  double Qs_nqs=0.0,  dQs_nqs_dVds=0.0,  dQs_nqs_dVgs=0.0,  dQs_nqs_dVbs=0.0,  dQs_nqs_dT=0.0,  dQs_nqs_dQi_nqs  =0.0 ;
  double Qg_nqs=0.0,  dQg_nqs_dQi_nqs=0.0, dQg_nqs_dQb_nqs =0.0 ;
  double tau=0.0,     dtau_dVds=0.0,     dtau_dVgs=0.0,     dtau_dVbs=0.0,     dtau_dT   =0.0 ;
  double taub=0.0,    dtaub_dVds=0.0,    dtaub_dVgs=0.0,    dtaub_dVbs=0.0,    dtaub_dT  =0.0 ;
  double Qdrat=0.0,   dQdrat_dVds=0.0,   dQdrat_dVgs=0.0,   dQdrat_dVbs=0.0,   dQdrat_dT =0.0 ;
  double Qi=0.0,      dQi_dVds=0.0,      dQi_dVgs=0.0,      dQi_dVbs=0.0,      dQi_dT    =0.0 ;
  double Qbulk=0.0,   dQbulk_dVds=0.0,   dQbulk_dVgs=0.0,   dQbulk_dVbs=0.0,   dQbulk_dT =0.0 ;

  /* output related variables */
  double dQi_nqs_dVds=0.0, dQi_nqs_dVgs=0.0, dQi_nqs_dVbs=0.0,
         dQb_nqs_dVds=0.0, dQb_nqs_dVgs=0.0, dQb_nqs_dVbs=0.0 ;
  double cgdb_nqs=0.0,     cggb_nqs=0.0,     cgsb_nqs=0.0,
         cbdb_nqs=0.0,     cbgb_nqs=0.0,     cbsb_nqs=0.0,
         cddb_nqs=0.0,     cdgb_nqs=0.0,     cdsb_nqs=0.0 ;
  double cgdb=0.0,         cggb=0.0,         cgsb=0.0,
         cbdb=0.0,         cbgb=0.0,         cbsb=0.0,
         cddb=0.0,         cdgb=0.0,         cdsb=0.0 ;

  /* rows of conductance and capacitance matrix stamp */
  double ydc_d[XDIM],   ydc_dP[XDIM],  ydc_g[XDIM],   ydc_gP[XDIM],  ydc_s[XDIM],  ydc_sP[XDIM] ;
  double ydc_bP[XDIM],  ydc_b[XDIM],   ydc_db[XDIM],  ydc_sb[XDIM],  ydc_t[XDIM],  ydc_qi[XDIM],  ydc_qb[XDIM]  ;
  double ydyn_d[XDIM],  ydyn_dP[XDIM], ydyn_g[XDIM],  ydyn_gP[XDIM], ydyn_s[XDIM], ydyn_sP[XDIM] ;
  double ydyn_bP[XDIM], ydyn_b[XDIM],  ydyn_db[XDIM], ydyn_sb[XDIM], ydyn_t[XDIM], ydyn_qi[XDIM], ydyn_qb[XDIM] ;

  /* limiter, bypass, and convergence */
  int ByPass=0, Check=0, Check1=0, Check2=0, Check3=0 ;
  double von=0.0, limval =0.0 ;
  double i_dP_hat=0.0, i_gP_hat=0.0, i_sP_hat=0.0, i_db_hat=0.0, i_sb_hat =0.0         ;

#define LIM_TOL 1.0e0
#define LIM_TOL2 1.0e0

  /* predictor and numerical integration stuff */
  double ag0=0.0, xfact=0.0 ;
  double ceq=0.0, geq=0.0 ;
  int ChargeComputationNeeded =
    ((ckt->CKTmode & (MODEAC | MODETRAN | MODEINITSMSIG)) ||
     ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)))
    ? 1 : 0;
  int showPhysVal=0 ;
  int isConv=0 ;
  double vds_pre=0.0 ;
  int i=0, noncon_old=0 ;



#define dNode        0
#define dNodePrime   1
#define gNode        2
#define gNodePrime   3
#define sNode        4
#define sNodePrime   5
#define bNodePrime   6
#define bNode        7
#define dbNode       8
#define sbNode       9
#define subNode     10
#define tempNode    11
#define qiNode      12
#define qbNode      13
#define lastNode    13        /* must be the last node! */


#define SPICE_rhs 1           /* set to 0 if rhs to be loaded for standard Newton */
                              /* set to 1 if rhs_eq to be loaded, e.g. for SPICE  */

  if (SPICE_rhs) {
    for (i=0; i<XDIM; i++) {
      x[i] = 0.0 ;
    }
  }

  /*  loop through all the HSMHV2 device models */
  for ( ; model != NULL; model = HSMHV2nextModel(model)) {
    /* loop through all the instances of the model */

    modelMKS = &model->modelMKS ;

    for (here = HSMHV2instances(model); here != NULL ;
         here = HSMHV2nextInstance(here)) {

      hereMKS = &here->hereMKS ;
      pParam = &here->pParam ;
      showPhysVal = 0;
      Check=1;
      ByPass = 0;
      vsubs = 0.0 ; /* substrate bias */
      deltemp = 0.0 ;
      noncon_old = ckt->CKTnoncon;
      flg_nqs = model->HSMHV2_conqs ;
      flg_subNode = here->HSMHV2subNode ; /* if flg_subNode > 0, external(/internal) substrate node exists */

#ifdef DEBUG_HISIMHVLD_VX
      printf("mode = %x\n", ckt->CKTmode);
      printf("Vd Vg Vs Vb %e %e %e %e\n", *(ckt->CKTrhsOld+here->HSMHV2dNodePrime),
             *(ckt->CKTrhsOld+here->HSMHV2gNodePrime),
             *(ckt->CKTrhsOld+here->HSMHV2sNodePrime),
             *(ckt->CKTrhsOld+here->HSMHV2bNodePrime));
#endif

      if ( ckt->CKTmode & MODEINITSMSIG ) {
        vbs = *(ckt->CKTstate0 + here->HSMHV2vbs);
        vgs = *(ckt->CKTstate0 + here->HSMHV2vgs);
        vds = *(ckt->CKTstate0 + here->HSMHV2vds);

        vges = *(ckt->CKTstate0 + here->HSMHV2vges);
        vdbd = *(ckt->CKTstate0 + here->HSMHV2vdbd);
        vsbs = *(ckt->CKTstate0 + here->HSMHV2vsbs);
        if (flg_subNode > 0) vsubs = *(ckt->CKTstate0 + here->HSMHV2vsubs);
        if( here->HSMHV2_coselfheat > 0 ){
          deltemp = *(ckt->CKTstate0 + here->HSMHV2deltemp);
        }
        vdse = *(ckt->CKTstate0 + here->HSMHV2vdse) ;
        vgse = *(ckt->CKTstate0 + here->HSMHV2vgse) ;
        vbse = *(ckt->CKTstate0 + here->HSMHV2vbse) ;
        if ( flg_nqs ) {
          Qi_nqs = *(ckt->CKTstate0 + here->HSMHV2qi_nqs) ;
          Qb_nqs = *(ckt->CKTstate0 + here->HSMHV2qb_nqs) ;
        } else {
          Qi_nqs = 0.0 ;
          Qb_nqs = 0.0 ;
        }
      /* printf("HSMHV2_load: (from state0) vds.. = %e %e %e %e %e %e\n",
                                              vds,vgs,vbs,vdse,vgse,vbse); */
      }
      else if ( ckt->CKTmode & MODEINITTRAN ) {
/* #include "printf_ld_converged.inc" */
        vbs = *(ckt->CKTstate1 + here->HSMHV2vbs);
        vgs = *(ckt->CKTstate1 + here->HSMHV2vgs);
        vds = *(ckt->CKTstate1 + here->HSMHV2vds);

        vges = *(ckt->CKTstate1 + here->HSMHV2vges);
        vdbd = *(ckt->CKTstate1 + here->HSMHV2vdbd);
        vsbs = *(ckt->CKTstate1 + here->HSMHV2vsbs);
        if (flg_subNode > 0) vsubs = *(ckt->CKTstate1 + here->HSMHV2vsubs);
        if( here->HSMHV2_coselfheat > 0 ){
          deltemp = *(ckt->CKTstate1 + here->HSMHV2deltemp);
        }
        vdse = *(ckt->CKTstate1 + here->HSMHV2vdse) ;
        vgse = *(ckt->CKTstate1 + here->HSMHV2vgse) ;
        vbse = *(ckt->CKTstate1 + here->HSMHV2vbse) ;
        if ( flg_nqs ) {
          Qi_nqs = *(ckt->CKTstate1 + here->HSMHV2qi_nqs) ;
          Qb_nqs = *(ckt->CKTstate1 + here->HSMHV2qb_nqs) ;
        } else {
          Qi_nqs = 0.0 ;
          Qb_nqs = 0.0 ;
        }
      }
      else if ( (ckt->CKTmode & MODEINITJCT) && !here->HSMHV2_off ) {
        vds = model->HSMHV2_type * here->HSMHV2_icVDS;
        vgs = vges = model->HSMHV2_type * here->HSMHV2_icVGS;
        vbs = vsbs = model->HSMHV2_type * here->HSMHV2_icVBS;
        vdbd = 0.0 ;
        if ( (vds == 0.0) && (vgs == 0.0) && (vbs == 0.0) &&
             ( (ckt->CKTmode & (MODETRAN|MODEAC|MODEDCOP|MODEDCTRANCURVE)) ||
               !(ckt->CKTmode & MODEUIC) ) ) {
          /* set biases for starting analysis */
          vbs = vdbd = vsbs = 0.0;
          vgs = vges = 0.1;
          vds = 0.1;
        }
        if (flg_subNode > 0) vsubs = 0.0;
        if( here->HSMHV2_coselfheat > 0 ) deltemp=0.0;
        vdse = vds ;
        vgse = vgs ;
        Qi_nqs = Qb_nqs = 0.0 ;
      }
      else if ( ( ckt->CKTmode & (MODEINITJCT | MODEINITFIX) ) &&
                here->HSMHV2_off ) {
        vbs = vgs = vds = 0.0; vges = 0.0; vdbd = vsbs = 0.0;
        if (flg_subNode > 0) vsubs = 0.0;
        if( here->HSMHV2_coselfheat > 0 ) deltemp=0.0;
        vdse = vds ;
        vgse = vgs ;
        Qi_nqs = Qb_nqs = 0.0 ;
      }
      else {
#ifndef PREDICTOR /* BSIM3 style */
        if (ckt->CKTmode & MODEINITPRED) {
/* #include "printf_ld_converged.inc" */
          /* if (here->HSMHV2_mode > 0) {
             gds_ext = here->HSMHV2_dIds_dVdse ;
          } else {
             gds_ext = + (here->HSMHV2_dIds_dVdse + here->HSMHV2_dIds_dVgse + here->HSMHV2_dIds_dVbse) ;
          }
          printf("zzz %e %e\n",ckt->CKTtime,gds_ext) ; */
          xfact = ckt->CKTdelta / ckt->CKTdeltaOld[1];
          *(ckt->CKTstate0 + here->HSMHV2vbs) =
            *(ckt->CKTstate1 + here->HSMHV2vbs);
          vbs = (1.0 + xfact)* (*(ckt->CKTstate1 + here->HSMHV2vbs))
            -(xfact * (*(ckt->CKTstate2 + here->HSMHV2vbs)));
          *(ckt->CKTstate0 + here->HSMHV2vgs) =
            *(ckt->CKTstate1 + here->HSMHV2vgs);
          vgs = (1.0 + xfact)* (*(ckt->CKTstate1 + here->HSMHV2vgs))
            -(xfact * (*(ckt->CKTstate2 + here->HSMHV2vgs)));
          *(ckt->CKTstate0 + here->HSMHV2vds) =
            *(ckt->CKTstate1 + here->HSMHV2vds);
          vds = (1.0 + xfact)* (*(ckt->CKTstate1 + here->HSMHV2vds))
            -(xfact * (*(ckt->CKTstate2 + here->HSMHV2vds)));
          *(ckt->CKTstate0 + here->HSMHV2vbd) =
            *(ckt->CKTstate0 + here->HSMHV2vbs)-
            *(ckt->CKTstate0 + here->HSMHV2vds);

          *(ckt->CKTstate0 + here->HSMHV2vges) =
            *(ckt->CKTstate1 + here->HSMHV2vges);
          vges = (1.0 + xfact)* (*(ckt->CKTstate1 + here->HSMHV2vges))
            -(xfact * (*(ckt->CKTstate2 + here->HSMHV2vges)));
          *(ckt->CKTstate0 + here->HSMHV2vdbd) =
            *(ckt->CKTstate1 + here->HSMHV2vdbd);
          vdbd = (1.0 + xfact)* (*(ckt->CKTstate1 + here->HSMHV2vdbd))
            - (xfact * (*(ckt->CKTstate2 + here->HSMHV2vdbd)));
          *(ckt->CKTstate0 + here->HSMHV2vdbs) =
            *(ckt->CKTstate0 + here->HSMHV2vdbd)
            + *(ckt->CKTstate0 + here->HSMHV2vds);
          *(ckt->CKTstate0 + here->HSMHV2vsbs) =
            *(ckt->CKTstate1 + here->HSMHV2vsbs);
          vsbs = (1.0 + xfact)* (*(ckt->CKTstate1 + here->HSMHV2vsbs))
            - (xfact * (*(ckt->CKTstate2 + here->HSMHV2vsbs)));
          if (flg_subNode > 0){
            vsubs = (1.0 + xfact)* (*(ckt->CKTstate1 + here->HSMHV2vsubs))
              - ( xfact * (*(ckt->CKTstate2 + here->HSMHV2vsubs)));
          }
          if( here->HSMHV2_coselfheat > 0 ){
            deltemp = (1.0 + xfact)* (*(ckt->CKTstate1 + here->HSMHV2deltemp))
              - ( xfact * (*(ckt->CKTstate2 + here->HSMHV2deltemp)));

            *(ckt->CKTstate0 + here->HSMHV2deltemp) =
              *(ckt->CKTstate1 + here->HSMHV2deltemp);
          }
          *(ckt->CKTstate0 + here->HSMHV2vdse) =
            *(ckt->CKTstate1 + here->HSMHV2vdse);
          vdse = (1.0 + xfact)* (*(ckt->CKTstate1 + here->HSMHV2vdse))
            -(xfact * (*(ckt->CKTstate2 + here->HSMHV2vdse)));
          *(ckt->CKTstate0 + here->HSMHV2vgse) =
            *(ckt->CKTstate1 + here->HSMHV2vgse);
          vgse = (1.0 + xfact)* (*(ckt->CKTstate1 + here->HSMHV2vgse))
            -(xfact * (*(ckt->CKTstate2 + here->HSMHV2vgse)));
          *(ckt->CKTstate0 + here->HSMHV2vbse) =
            *(ckt->CKTstate1 + here->HSMHV2vbse);
          vbse = (1.0 + xfact)* (*(ckt->CKTstate1 + here->HSMHV2vbse))
            -(xfact * (*(ckt->CKTstate2 + here->HSMHV2vbse)));
          if (flg_nqs) {
            *(ckt->CKTstate0 + here->HSMHV2qi_nqs) =
              *(ckt->CKTstate1 + here->HSMHV2qi_nqs);
            Qi_nqs = (1.0 + xfact)* (*(ckt->CKTstate1 + here->HSMHV2qi_nqs))
              -(xfact * (*(ckt->CKTstate2 + here->HSMHV2qi_nqs)));
            *(ckt->CKTstate0 + here->HSMHV2qb_nqs) =
              *(ckt->CKTstate1 + here->HSMHV2qb_nqs);
            Qb_nqs = (1.0 + xfact)* (*(ckt->CKTstate1 + here->HSMHV2qb_nqs))
              -(xfact * (*(ckt->CKTstate2 + here->HSMHV2qb_nqs)));
          } else {
            Qi_nqs = Qb_nqs = 0.0 ;
          }
        }
        else {
#endif /* PREDICTOR */
/*          here->HSMHV2_time = ckt->CKTtime; /\* for debug print *\/ */
          /* get biases from CKT */
          vbs = model->HSMHV2_type *
            (*(ckt->CKTrhsOld+here->HSMHV2bNodePrime) -
             *(ckt->CKTrhsOld+here->HSMHV2sNodePrime));
          vgs = model->HSMHV2_type *
            (*(ckt->CKTrhsOld+here->HSMHV2gNodePrime) -
             *(ckt->CKTrhsOld+here->HSMHV2sNodePrime));
          vds = model->HSMHV2_type *
            (*(ckt->CKTrhsOld+here->HSMHV2dNodePrime) -
             *(ckt->CKTrhsOld+here->HSMHV2sNodePrime));

          vges = model->HSMHV2_type *
            (*(ckt->CKTrhsOld+here->HSMHV2gNode) -
             *(ckt->CKTrhsOld+here->HSMHV2sNodePrime));
          vdbd = model->HSMHV2_type
            * (*(ckt->CKTrhsOld + here->HSMHV2dbNode)
            - *(ckt->CKTrhsOld + here->HSMHV2dNode));
          vsbs = model->HSMHV2_type
            * (*(ckt->CKTrhsOld + here->HSMHV2sbNode)
            - *(ckt->CKTrhsOld + here->HSMHV2sNode));
          if (flg_subNode > 0){
            vsubs = model->HSMHV2_type
              * (*(ckt->CKTrhsOld + here->HSMHV2subNode)
              - *(ckt->CKTrhsOld + here->HSMHV2sNode));
          }
          if( here->HSMHV2_coselfheat > 0 ){
            deltemp = *(ckt->CKTrhsOld + here->HSMHV2tempNode);
          }
          vbse = model->HSMHV2_type *
            (*(ckt->CKTrhsOld+here->HSMHV2bNodePrime) -
             *(ckt->CKTrhsOld+here->HSMHV2sNode));
          vgse = model->HSMHV2_type *
            (*(ckt->CKTrhsOld+here->HSMHV2gNodePrime) -
             *(ckt->CKTrhsOld+here->HSMHV2sNode));
          vdse = model->HSMHV2_type *
            (*(ckt->CKTrhsOld+here->HSMHV2dNode) -
             *(ckt->CKTrhsOld+here->HSMHV2sNode));
          if ( flg_nqs ) {
            Qi_nqs = *(ckt->CKTrhsOld + here->HSMHV2qiNode);
            Qb_nqs = *(ckt->CKTrhsOld + here->HSMHV2qbNode);
          } else {
            Qi_nqs = Qb_nqs = 0.0 ;
          }
#ifndef PREDICTOR
        }
#endif /* PREDICTOR */

        /* printf("HSMHV2_load: (from rhs   ) vds.. = %e %e %e %e %e %e\n",
                                                 vds,vgs,vbs,vdse,vgse,vbse); */

        vbd = vbs - vds;
        vgd = vgs - vds;
        vged = vges - vds;
        vdbs = vdbd + vdse;
        vgdo = *(ckt->CKTstate0 + here->HSMHV2vgs) - *(ckt->CKTstate0 + here->HSMHV2vds);
        vgedo = *(ckt->CKTstate0 + here->HSMHV2vges) - *(ckt->CKTstate0 + here->HSMHV2vds);

        vds_pre = vds;

#ifndef NOBYPASS
        /* start of bypass section
           ... no bypass in case of selfheating  */
        if ( !(ckt->CKTmode & MODEINITPRED) && ckt->CKTbypass && !here->HSMHV2_coselfheat) {
          delvds  = vds  - *(ckt->CKTstate0 + here->HSMHV2vds) ;
          delvgs  = vgs  - *(ckt->CKTstate0 + here->HSMHV2vgs) ;
          delvbs  = vbs  - *(ckt->CKTstate0 + here->HSMHV2vbs) ;
          delvdse = vdse - *(ckt->CKTstate0 + here->HSMHV2vdse) ;
          delvgse = vgse - *(ckt->CKTstate0 + here->HSMHV2vgse) ;
          delvbse = vbse - *(ckt->CKTstate0 + here->HSMHV2vbse) ;
          delvdbd = vdbd - *(ckt->CKTstate0 + here->HSMHV2vdbd) ;
          delvsbs = vsbs - *(ckt->CKTstate0 + here->HSMHV2vsbs) ;
          if (flg_subNode > 0) delvsubs = vsubs - *(ckt->CKTstate0 + here->HSMHV2vsubs) ; /* substrate bias change */
          deldeltemp = deltemp - *(ckt->CKTstate0 + here->HSMHV2deltemp) ;
          if (flg_nqs) {
            delQi_nqs = Qi_nqs - *(ckt->CKTstate0 + here->HSMHV2qi_nqs) ;
            delQb_nqs = Qb_nqs - *(ckt->CKTstate0 + here->HSMHV2qb_nqs) ;
          } else {
            delQi_nqs = delQb_nqs = 0.0 ;
          }

          /* now let's see if we can bypass                     */
          /* ... first perform the easy cheap bypass checks ... */
 /*          1 2     3       3                       3    4    4     4 5                               543                   2    1 */
          if ( ( fabs(delvds ) < ckt->CKTreltol * MAX(fabs(vds ),fabs(*(ckt->CKTstate0 + here->HSMHV2vds ))) + ckt->CKTvoltTol ) &&
               ( fabs(delvgs ) < ckt->CKTreltol * MAX(fabs(vgs ),fabs(*(ckt->CKTstate0 + here->HSMHV2vgs ))) + ckt->CKTvoltTol ) &&
               ( fabs(delvbs ) < ckt->CKTreltol * MAX(fabs(vbs ),fabs(*(ckt->CKTstate0 + here->HSMHV2vbs ))) + ckt->CKTvoltTol ) &&
               ( fabs(delvdse) < ckt->CKTreltol * MAX(fabs(vdse),fabs(*(ckt->CKTstate0 + here->HSMHV2vdse))) + ckt->CKTvoltTol ) &&
               ( fabs(delvgse) < ckt->CKTreltol * MAX(fabs(vgse),fabs(*(ckt->CKTstate0 + here->HSMHV2vgse))) + ckt->CKTvoltTol ) &&
               ( fabs(delvbse) < ckt->CKTreltol * MAX(fabs(vbse),fabs(*(ckt->CKTstate0 + here->HSMHV2vbse))) + ckt->CKTvoltTol ) &&
               ( fabs(delvdbd) < ckt->CKTreltol * MAX(fabs(vdbd),fabs(*(ckt->CKTstate0 + here->HSMHV2vdbd))) + ckt->CKTvoltTol ) &&
               ( fabs(delvsbs) < ckt->CKTreltol * MAX(fabs(vsbs),fabs(*(ckt->CKTstate0 + here->HSMHV2vsbs))) + ckt->CKTvoltTol ) &&
               ( fabs(delvsubs) < ckt->CKTreltol * MAX(fabs(vsubs),fabs(*(ckt->CKTstate0 + here->HSMHV2vsubs))) + ckt->CKTvoltTol ) &&
               ( fabs(delQi_nqs) < ckt->CKTreltol *   fabs(Qi_nqs) + ckt->CKTchgtol*ckt->CKTabstol + 1.0e-20  ) &&
               ( fabs(delQb_nqs) < ckt->CKTreltol *   fabs(Qb_nqs) + ckt->CKTchgtol*ckt->CKTabstol + 1.0e-20  )    )
                                                                                  /* 1.0e-20: heuristic value, must be small enough     */
                                                                                  /* to ensure that bypass does not destroy convergence */
          { /* ... the first bypass checks are passed -> now do the more expensive checks ...*/
            if ( here->HSMHV2_mode > 0 ) { /* forward mode */
              Ids        = here->HSMHV2_ids ;
              gds        = here->HSMHV2_dIds_dVdsi ;
              gm         = here->HSMHV2_dIds_dVgsi ;
              gmbs       = here->HSMHV2_dIds_dVbsi ;
              gmT        = 0.0  ;
              gmbs_ext   = here->HSMHV2_dIds_dVbse;
              gds_ext    = here->HSMHV2_dIds_dVdse ;
              gm_ext     = here->HSMHV2_dIds_dVgse;
              Isub         = here->HSMHV2_isub ;
              dIsub_dVds   = here->HSMHV2_dIsub_dVdsi ;
              dIsub_dVgs   = here->HSMHV2_dIsub_dVgsi ;
              dIsub_dVbs   = here->HSMHV2_dIsub_dVbsi ;
              dIsub_dT     = 0.0  ;
              Isubs        = 0.0 ;
              dIsubs_dVds  = 0.0 ;
              dIsubs_dVgs  = 0.0 ;
              dIsubs_dVbs  = 0.0 ;
              dIsubs_dT    = 0.0 ;
              IsubLD         = here->HSMHV2_isubld ;
              dIsubLD_dVds   = here->HSMHV2_dIsubLD_dVdsi ;
              dIsubLD_dVgs   = here->HSMHV2_dIsubLD_dVgsi ;
              dIsubLD_dVbs   = here->HSMHV2_dIsubLD_dVbsi ;
              dIsubLD_dT     = 0.0  ;
              dIsubLD_dVddp  = here->HSMHV2_dIsubLD_dVddp ;
              IsubLDs        = 0.0 ;
              dIsubLDs_dVds  = 0.0 ;
              dIsubLDs_dVgs  = 0.0 ;
              dIsubLDs_dVbs  = 0.0 ;
              dIsubLDs_dT    = 0.0 ;
              dIsubLDs_dVddp = 0.0 ;
              IdsIBPC         = here->HSMHV2_idsibpc ;
              dIdsIBPC_dVds   = here->HSMHV2_dIdsIBPC_dVdsi ;
              dIdsIBPC_dVgs   = here->HSMHV2_dIdsIBPC_dVgsi ;
              dIdsIBPC_dVbs   = here->HSMHV2_dIdsIBPC_dVbsi ;
              dIdsIBPC_dT     = 0.0  ;
              dIdsIBPC_dVddp  = here->HSMHV2_dIdsIBPC_dVddp ;
              IdsIBPCs        = 0.0 ;
              dIdsIBPCs_dVds  = 0.0 ;
              dIdsIBPCs_dVgs  = 0.0 ;
              dIdsIBPCs_dVbs  = 0.0 ;
              dIdsIBPCs_dT    = 0.0 ;
              dIdsIBPCs_dVddp = 0.0 ;
              Igidl        = here->HSMHV2_igidl ;
              dIgidl_dVds  = here->HSMHV2_dIgidl_dVdsi ;
              dIgidl_dVgs  = here->HSMHV2_dIgidl_dVgsi ;
              dIgidl_dVbs  = here->HSMHV2_dIgidl_dVbsi ;
              dIgidl_dT    = 0.0  ;
              Igisl        = here->HSMHV2_igisl ;
              dIgisl_dVds  = here->HSMHV2_dIgisl_dVdsi ;
              dIgisl_dVgs  = here->HSMHV2_dIgisl_dVgsi ;
              dIgisl_dVbs  = here->HSMHV2_dIgisl_dVbsi ;
              dIgisl_dT    = 0.0  ;
              Igd          = here->HSMHV2_igd ;
              dIgd_dVd   = here->HSMHV2_dIgd_dVdsi ;
              dIgd_dVg   = here->HSMHV2_dIgd_dVgsi ;
              dIgd_dVb   = here->HSMHV2_dIgd_dVbsi ;
              dIgd_dT      = 0.0  ;
              Igs          = here->HSMHV2_igs ;
              dIgs_dVd   = here->HSMHV2_dIgs_dVdsi ;
              dIgs_dVg   = here->HSMHV2_dIgs_dVgsi ;
              dIgs_dVb   = here->HSMHV2_dIgs_dVbsi ;
              dIgs_dT      = 0.0  ;
              Igb          = here->HSMHV2_igb ;
              dIgb_dVd   = here->HSMHV2_dIgb_dVdsi ;
              dIgb_dVg   = here->HSMHV2_dIgb_dVgsi ;
              dIgb_dVb   = here->HSMHV2_dIgb_dVbsi ;
              dIgb_dT      = 0.0  ;
              Ibd = here->HSMHV2_ibd ;
              Gbd = here->HSMHV2_gbd ;
              Gbdt = 0.0 ;
              Ibs = here->HSMHV2_ibs ;
              Gbs = here->HSMHV2_gbs ;
              Gbst = 0.0 ;
            } else { /* reverse mode */
              Ids       = - here->HSMHV2_ids ;
              gds       = + (here->HSMHV2_dIds_dVdsi + here->HSMHV2_dIds_dVgsi + here->HSMHV2_dIds_dVbsi) ;
              gm        = - here->HSMHV2_dIds_dVgsi ;
              gmbs      = - here->HSMHV2_dIds_dVbsi ;
              gmT       = 0.0  ;
              gds_ext   = + (here->HSMHV2_dIds_dVdse + here->HSMHV2_dIds_dVgse + here->HSMHV2_dIds_dVbse) ;
              gm_ext    = - here->HSMHV2_dIds_dVgse;
              gmbs_ext  = - here->HSMHV2_dIds_dVbse;
              Isub         = 0.0 ;
              dIsub_dVds   = 0.0 ;
              dIsub_dVgs   = 0.0 ;
              dIsub_dVbs   = 0.0 ;
              dIsub_dT     = 0.0 ;
              Isubs        =   here->HSMHV2_isub ;
              dIsubs_dVds  = - (here->HSMHV2_dIsub_dVdsi + here->HSMHV2_dIsub_dVgsi + here->HSMHV2_dIsub_dVbsi) ;
              dIsubs_dVgs  =   here->HSMHV2_dIsub_dVgsi ;
              dIsubs_dVbs  =   here->HSMHV2_dIsub_dVbsi ;
              dIsubs_dT    =   0.0 ;
              IsubLD         = 0.0 ;
              dIsubLD_dVds   = 0.0 ;
              dIsubLD_dVgs   = 0.0 ;
              dIsubLD_dVbs   = 0.0 ;
              dIsubLD_dT     = 0.0 ;
              dIsubLD_dVddp  = 0.0 ;
              IsubLDs        =   here->HSMHV2_isubld ;
              dIsubLDs_dVds  = - (here->HSMHV2_dIsubLD_dVdsi + here->HSMHV2_dIsubLD_dVgsi + here->HSMHV2_dIsubLD_dVbsi) ;
              dIsubLDs_dVgs  =   here->HSMHV2_dIsubLD_dVgsi ;
              dIsubLDs_dVbs  =   here->HSMHV2_dIsubLD_dVbsi ;
              dIsubLDs_dT    =   0.0 ;
              dIsubLDs_dVddp = - here->HSMHV2_dIsubLD_dVddp ;
              IdsIBPC         = 0.0 ;
              dIdsIBPC_dVds   = 0.0 ;
              dIdsIBPC_dVgs   = 0.0 ;
              dIdsIBPC_dVbs   = 0.0 ;
              dIdsIBPC_dT     = 0.0 ;
              dIdsIBPC_dVddp  = 0.0 ;
              IdsIBPCs        =   here->HSMHV2_idsibpc ;
              dIdsIBPCs_dVds  = - (here->HSMHV2_dIdsIBPC_dVdsi + here->HSMHV2_dIdsIBPC_dVgsi + here->HSMHV2_dIdsIBPC_dVbsi) ;
              dIdsIBPCs_dVgs  =   here->HSMHV2_dIdsIBPC_dVgsi ;
              dIdsIBPCs_dVbs  =   here->HSMHV2_dIdsIBPC_dVbsi ;
              dIdsIBPCs_dT    =   0.0 ;
              dIdsIBPCs_dVddp = - here->HSMHV2_dIdsIBPC_dVddp ;
              Igidl        =   here->HSMHV2_igisl ;
              dIgidl_dVds  = - (here->HSMHV2_dIgisl_dVdsi + here->HSMHV2_dIgisl_dVgsi + here->HSMHV2_dIgisl_dVbsi) ;
              dIgidl_dVgs  =   here->HSMHV2_dIgisl_dVgsi ;
              dIgidl_dVbs  =   here->HSMHV2_dIgisl_dVbsi ;
              dIgidl_dT    =   0.0  ;
              Igisl        =   here->HSMHV2_igidl ;
              dIgisl_dVds  = - (here->HSMHV2_dIgidl_dVdsi + here->HSMHV2_dIgidl_dVgsi + here->HSMHV2_dIgidl_dVbsi) ;
              dIgisl_dVgs  =   here->HSMHV2_dIgidl_dVgsi ;
              dIgisl_dVbs  =   here->HSMHV2_dIgidl_dVbsi ;
              dIgisl_dT    =   0.0  ;
              Igd          =   here->HSMHV2_igd ;
              dIgd_dVd   = - (here->HSMHV2_dIgs_dVdsi + here->HSMHV2_dIgs_dVgsi + here->HSMHV2_dIgs_dVbsi) ;
              dIgd_dVg   =   here->HSMHV2_dIgs_dVgsi ;
              dIgd_dVb   =   here->HSMHV2_dIgs_dVbsi ;
              dIgd_dT      =   0.0  ;
              Igs          =   here->HSMHV2_igs ;
              dIgs_dVd   = - (here->HSMHV2_dIgd_dVdsi + here->HSMHV2_dIgd_dVgsi + here->HSMHV2_dIgd_dVbsi) ;
              dIgs_dVg   =   here->HSMHV2_dIgd_dVgsi ;
              dIgs_dVb   =   here->HSMHV2_dIgd_dVbsi ;
              dIgs_dT      =   0.0  ;
              Igb          =   here->HSMHV2_igb ;
              dIgb_dVd   = - (here->HSMHV2_dIgb_dVdsi + here->HSMHV2_dIgb_dVgsi + here->HSMHV2_dIgb_dVbsi) ;
              dIgb_dVg   =   here->HSMHV2_dIgb_dVgsi ;
              dIgb_dVb   =   here->HSMHV2_dIgb_dVbsi ;
              dIgb_dT      =   0.0  ;
              Ibd = here->HSMHV2_ibd ;
              Gbd = here->HSMHV2_gbd ;
              Gbdt = 0.0 ;
              Ibs = here->HSMHV2_ibs ;
              Gbs = here->HSMHV2_gbs ;
              Gbst = 0.0 ;
            } /* end of reverse mode */

            /* for bypass control, only nonlinear static currents are considered: */
            i_dP     = Ids  + Isub  + Igidl - Igd ;
            i_dP_hat = i_dP + gm         *delvgs + gds        *delvds + gmbs       *delvbs + gmT      *deldeltemp
                            + dIsub_dVgs *delvgs + dIsub_dVds *delvds + dIsub_dVbs *delvbs + dIsub_dT *deldeltemp
                            + dIgidl_dVgs*delvgs + dIgidl_dVds*delvds + dIgidl_dVbs*delvbs + dIgidl_dT*deldeltemp
                            -(dIgd_dVg   *delvgs + dIgd_dVd   *delvds + dIgd_dVb   *delvbs + dIgd_dT  *deldeltemp)
                            + gm_ext     *delvgse + gds_ext    *delvdse + gmbs_ext   *delvbse ;

            i_gP     = Igd  + Igs   + Igb ;
            i_gP_hat = i_gP + dIgd_dVg   *delvgs + dIgd_dVd   *delvds + dIgd_dVb   *delvbs + dIgd_dT  *deldeltemp
                            + dIgs_dVg   *delvgs + dIgs_dVd   *delvds + dIgs_dVb   *delvbs + dIgs_dT  *deldeltemp
                            + dIgb_dVg   *delvgs + dIgb_dVd   *delvds + dIgb_dVb   *delvbs + dIgb_dT  *deldeltemp ;

            i_sP     =-Ids  + Isubs + Igisl - Igs ;
            i_sP_hat = i_sP -(gm         *delvgs + gds        *delvds + gmbs       *delvbs + gmT      *deldeltemp)
                            + dIsubs_dVgs*delvgs + dIsubs_dVds*delvds + dIsubs_dVbs*delvbs + dIsubs_dT*deldeltemp
                            + dIgisl_dVgs*delvgs + dIgisl_dVds*delvds + dIgisl_dVbs*delvbs + dIgisl_dT*deldeltemp
                            -(dIgs_dVg   *delvgs + dIgs_dVd   *delvds + dIgs_dVb   *delvbs + dIgs_dT  *deldeltemp)
                            -(gm_ext     *delvgse + gds_ext    *delvdse + gmbs_ext   *delvbse) ;

            i_db     = Ibd ;
            i_db_hat = i_db + Gbd*delvdbd + Gbdt*deldeltemp ;

            i_sb     = Ibs ;
            i_sb_hat = i_sb + Gbs*delvsbs + Gbst*deldeltemp ;

            /* ... second part of bypass checks: */
 /*            1 2     3               3                       3    4        4     4    43                  2    1 */
            if ( ( fabs(i_dP_hat - i_dP) < ckt->CKTreltol * MAX(fabs(i_dP_hat),fabs(i_dP)) + ckt->CKTabstol ) &&
                 ( fabs(i_gP_hat - i_gP) < ckt->CKTreltol * MAX(fabs(i_gP_hat),fabs(i_gP)) + ckt->CKTabstol ) &&
                 ( fabs(i_sP_hat - i_sP) < ckt->CKTreltol * MAX(fabs(i_sP_hat),fabs(i_sP)) + ckt->CKTabstol ) &&
                 ( fabs(i_db_hat - i_db) < ckt->CKTreltol * MAX(fabs(i_db_hat),fabs(i_db)) + ckt->CKTabstol ) &&
                 ( fabs(i_sb_hat - i_sb) < ckt->CKTreltol * MAX(fabs(i_sb_hat),fabs(i_sb)) + ckt->CKTabstol )    )
            {
              /* bypass code */
              vds  = *(ckt->CKTstate0 + here->HSMHV2vds );
              vgs  = *(ckt->CKTstate0 + here->HSMHV2vgs );
              vbs  = *(ckt->CKTstate0 + here->HSMHV2vbs );
              vdse = *(ckt->CKTstate0 + here->HSMHV2vdse);
              vgse = *(ckt->CKTstate0 + here->HSMHV2vgse);
              vbse = *(ckt->CKTstate0 + here->HSMHV2vbse);
              vdbd = *(ckt->CKTstate0 + here->HSMHV2vdbd);
              vsbs = *(ckt->CKTstate0 + here->HSMHV2vsbs);
              vsubs = *(ckt->CKTstate0 + here->HSMHV2vsubs);
              deltemp = *(ckt->CKTstate0 + here->HSMHV2deltemp);
              if ( flg_nqs ) {
                Qi_nqs = *(ckt->CKTstate0 + here->HSMHV2qi_nqs);
                Qb_nqs = *(ckt->CKTstate0 + here->HSMHV2qb_nqs);
              }

              vges = *(ckt->CKTstate0 + here->HSMHV2vges);

              vbd  = vbs - vds;
              vgd  = vgs - vds;
              vgb  = vgs - vbs;
              vged = vges - vds;

              vbs_jct = vsbs;
              vbd_jct = vdbd;

              /* linear branch currents */
              vddp = model->HSMHV2_type * (*(ckt->CKTrhsOld+here->HSMHV2dNode) - *(ckt->CKTrhsOld+here->HSMHV2dNodePrime));
              vggp = model->HSMHV2_type * (*(ckt->CKTrhsOld+here->HSMHV2gNode) - *(ckt->CKTrhsOld+here->HSMHV2gNodePrime));
              vssp = model->HSMHV2_type * (*(ckt->CKTrhsOld+here->HSMHV2sNode) - *(ckt->CKTrhsOld+here->HSMHV2sNodePrime));
              vbpb  = model->HSMHV2_type * (*(ckt->CKTrhsOld+here->HSMHV2bNodePrime) - *(ckt->CKTrhsOld+here->HSMHV2bNode));
              vbpdb = model->HSMHV2_type * (*(ckt->CKTrhsOld+here->HSMHV2bNodePrime) - *(ckt->CKTrhsOld+here->HSMHV2dbNode));
              vbpsb = model->HSMHV2_type * (*(ckt->CKTrhsOld+here->HSMHV2bNodePrime) - *(ckt->CKTrhsOld+here->HSMHV2sbNode));

              ByPass = 1;
              goto line755;
            }
          }
        }  /* end of Bypass section */
#endif /*NOBYPASS*/

#ifdef DEBUG_HISIMHVLD_VX
        printf( "vbd_p    = %12.5e\n" , vbd );
        printf( "vbs_p    = %12.5e\n" , vbs );
        printf( "vgs_p    = %12.5e\n" , vgs );
        printf( "vds_p    = %12.5e\n" , vds );
#endif

        /* start limiting of nonlinear branch voltages */

        von = here->HSMHV2_von;
        Check3 = 0 ;
        if(*(ckt->CKTstate0 + here->HSMHV2vds) >= 0.0) { /* case vds>=0 for limiting */
          limval = DEVfetlim(vgs, *(ckt->CKTstate0 + here->HSMHV2vgs), von);
          if (vgs != limval) {
            vgs = limval ;
            Check3 = 1 ;
          }
          if (Check3) vds = vgs - vgd;
          limval = DEVlimvds(vds, *(ckt->CKTstate0 + here->HSMHV2vds));
          if (vds != limval) {
            vds = limval ;
            Check3 = 2 ;
          }
          vgd = vgs - vds;

          if (here->HSMHV2_corg == 1) {
            limval = DEVfetlim(vges, *(ckt->CKTstate0 + here->HSMHV2vges), von);
            if (vges != limval) {
              vges = limval ;
              Check3 = 3 ;
            }
            vged = vges - vds;
          }

        } else { /* case vds < 0 for limiting */
          limval = DEVfetlim(vgd, vgdo, von);
          if (vgd != limval) {
            vgd = limval ;
            Check3 = 4 ;
          }
          if (Check3) vds = vgs - vgd;
          limval = -DEVlimvds(-vds, -(*(ckt->CKTstate0 + here->HSMHV2vds)));
          if (vds != limval) {
            vds = limval ;
            Check3 = 5 ;
          }
          vgs = vgd + vds;

          if (here->HSMHV2_corg == 1) {
            limval = DEVfetlim(vged, vgedo, von);
            if (vged != limval) {
              vged = limval ;
              Check3 = 6 ;
            }
            vges = vged + vds;
          }
        } /* end of case vds< 0 for limiting */

        if (vds >= 0.0) { /* case vds >=0 for limiting of junctions */
          vbs = DEVpnjlim(vbs, *(ckt->CKTstate0 + here->HSMHV2vbs),
                          CONSTvt0, model->HSMHV2_vcrit, &Check1);
          if (Check1) Check3 = 7 ;
          vbd = vbs - vds;
          if (here->HSMHV2_corbnet) {
            vsbs = DEVpnjlim(vsbs, *(ckt->CKTstate0 + here->HSMHV2vsbs),
                             CONSTvt0, model->HSMHV2_vcrit, &Check2);
            if (Check2) Check3 = 8 ;
          }
        }
        else { /* case vds < 0 for limiting of junctions */
          vbd = DEVpnjlim(vbd, *(ckt->CKTstate0 + here->HSMHV2vbd),
                          CONSTvt0, model->HSMHV2_vcrit, &Check1);
          if (Check1) Check3 = 9 ;
          vbs = vbd + vds;
          if (here->HSMHV2_corbnet) {
            vdbd = DEVpnjlim(vdbd, *(ckt->CKTstate0 + here->HSMHV2vdbd),
                             CONSTvt0, model->HSMHV2_vcrit, &Check2);
            if (Check2) {
               Check3 = 10 ;
               vdbs = vdbd + vdse;
            }
          }
        }

        if( here->HSMHV2_coselfheat > 0 ){
           /* Logarithmic damping of deltemp beyond LIM_TOL */
           deltemp_old = *(ckt->CKTstate0 + here->HSMHV2deltemp);
           if (deltemp > deltemp_old + LIM_TOL)
             {deltemp = deltemp_old + LIM_TOL + log10((deltemp-deltemp_old)/LIM_TOL);
               Check3 = 11;}
           else if (deltemp < deltemp_old - LIM_TOL)
             {deltemp = deltemp_old - LIM_TOL - log10((deltemp_old-deltemp)/LIM_TOL);
               Check3 = 12;}
        }

        /* if (Check3) printf("HSMHV2_load: Check3=%d\n",Check3) ; */

        /* limiting completed */
        if (Check3 == 0 ) Check = 0 ;
      } /* loading and limiting of nonlinear branch voltages is completed */


      vbd = vbs - vds;
      vgd = vgs - vds;
      vgb = vgs - vbs;
      vged = vges - vds;

      vbs_jct = vsbs;
      vbd_jct = vdbd;

      /* linear branch voltages */
      if ( (ckt->CKTmode & MODEINITJCT) && !here->HSMHV2_off ) {
        vddp = vggp = vssp = vbpdb = vbpb = vbpsb = 0.0;
      } else {
        vddp = model->HSMHV2_type *
             (*(ckt->CKTrhsOld+here->HSMHV2dNode) -
                *(ckt->CKTrhsOld+here->HSMHV2dNodePrime));

        vggp = model->HSMHV2_type *
             (*(ckt->CKTrhsOld+here->HSMHV2gNode) -
                *(ckt->CKTrhsOld+here->HSMHV2gNodePrime));

        vssp = model->HSMHV2_type *
             (*(ckt->CKTrhsOld+here->HSMHV2sNode) -
                *(ckt->CKTrhsOld+here->HSMHV2sNodePrime));

        vbpdb = model->HSMHV2_type *
             (*(ckt->CKTrhsOld+here->HSMHV2bNodePrime) -
                *(ckt->CKTrhsOld+here->HSMHV2dbNode));

        vbpb = model->HSMHV2_type *
             (*(ckt->CKTrhsOld+here->HSMHV2bNodePrime) -
                *(ckt->CKTrhsOld+here->HSMHV2bNode));

        vbpsb = model->HSMHV2_type *
             (*(ckt->CKTrhsOld+here->HSMHV2bNodePrime) -
                *(ckt->CKTrhsOld+here->HSMHV2sbNode));
      }

#ifdef DEBUG_HISIMHVLD_VX
      printf( "vbd    = %12.5e\n" , vbd );
      printf( "vbs    = %12.5e\n" , vbs );
      printf( "vgs    = %12.5e\n" , vgs );
      printf( "vds    = %12.5e\n" , vds );
#endif

      /* After loading (and limiting of branch voltages: Start model evaluation */

      /* printf("HSMHV2_load: vds=%e vgs=%e vbs=%e vsd=%e vgd=%e vbd=%e\n",
                          vds,vgs,vbs,-vds,vgs-vds,vbs-vds); */

      if (vds >= 0) { /* normal mode */
        here->HSMHV2_mode = 1;
        ivds = vds;
        ivgs = vgs;
        ivbs = vbs;
        ivdse = vdse;
        ivgse = vgse;
        ivbse = vbse;
      } else { /* reverse mode */
        here->HSMHV2_mode = -1;
        ivds = -vds;
        ivgs = vgd;
        ivbs = vbd;
        ivdse = -vdse;
        ivgse = vgse - vdse;
        ivbse = vbse - vdse;
      }

      if ( model->HSMHV2_info >= 5 ) { /* mode, bias conditions ... */
        printf( "--- variables given to HSMHV2evaluate() ----\n" );
        printf( "type   = %s\n" , (model->HSMHV2_type>0) ? "NMOS" : "PMOS" );
        printf( "mode   = %s\n" , (here->HSMHV2_mode>0) ? "NORMAL" : "REVERSE" );

        printf( "vbse vbs    = %12.5e %12.5e\n" , vbse, ivbs );
        printf( "vdse vds    = %12.5e %12.5e\n" , vdse, ivds );
        printf( "vgse vgs    = %12.5e %12.5e\n" , vgse, ivgs );
      }
      if ( model->HSMHV2_info >= 6 ) { /* input flags */
        printf( "corsrd = %s\n" , (model->HSMHV2_corsrd)  ? "true" : "false" ) ;
        printf( "coadov = %s\n" , (model->HSMHV2_coadov)  ? "true" : "false" ) ;
        printf( "coisub = %s\n" , (model->HSMHV2_coisub)  ? "true" : "false" ) ;
        printf( "coiigs = %s\n" , (model->HSMHV2_coiigs)  ? "true" : "false" ) ;
        printf( "cogidl = %s\n" , (model->HSMHV2_cogidl)  ? "true" : "false" ) ;
        printf( "coovlp = %s\n" , (model->HSMHV2_coovlp)  ? "true" : "false" ) ;
        printf( "coovlps = %s\n" , (model->HSMHV2_coovlps) ? "true" : "false" ) ;
        printf( "coflick = %s\n", (model->HSMHV2_coflick) ? "true" : "false" ) ;
        printf( "coisti = %s\n" , (model->HSMHV2_coisti)  ? "true" : "false" ) ;
        printf( "conqs  = %s\n" , (model->HSMHV2_conqs)   ? "true" : "false" ) ;
        printf( "cothrml = %s\n", (model->HSMHV2_cothrml) ? "true" : "false" ) ;
        printf( "coign = %s\n"  , (model->HSMHV2_coign)   ? "true" : "false" ) ;
        printf( "cosym   = %s\n" , (model->HSMHV2_cosym) ? "true" : "false" ) ;
        printf( "coselfheat = %s\n" , (here->HSMHV2_coselfheat) ? "true" : "false" ) ;
      }
      /* print inputs ------------AA */

 #ifdef DEBUG_HISIMHVCGG
      /* Print convergence flag */
      printf("isConv %d ", isConv );
      printf("CKTtime %e ", ckt->CKTtime );
      printf("Vb %1.3e ", (model->HSMHV2_type>0) ? vbs:-vbs );
      printf("Vd %1.3e ", (model->HSMHV2_type>0) ? vds:-vds );
      printf("Vg %1.3e ", (model->HSMHV2_type>0) ? vgs:-vgs );
 #endif

      /* call model evaluation */
      if ( HSMHV2evaluate(ivdse,ivgse,ivbse,ivds, ivgs, ivbs, vbs_jct, vbd_jct, vsubs, vddp, deltemp, here, model, ckt) == HiSIM_ERROR )
        return (HiSIM_ERROR);
      if ( here->HSMHV2_cordrift == 1 ) {
        if ( HSMHV2rdrift(vddp, vds, vbs, vsubs, deltemp, here, model, ckt) == HiSIM_ERROR )
        return (HiSIM_ERROR);
      }
      if ( HSMHV2dio(vbs_jct, vbd_jct, deltemp, here, model, ckt) == HiSIM_ERROR )
        return (HiSIM_ERROR);

#ifdef DEBUG_HISIMHVCGG
      printf("HSMHV2_ids %e ", here->HSMHV2_ids ) ;
      printf("HSMHV2_cggb %e ", here->HSMHV2_cggb ) ;
      printf("\n") ;
#endif

      here->HSMHV2_called += 1;

line755: /* standard entry if HSMHV2evaluate is bypassed */
         /* (could be shifted a bit forward ...)       */
      if ( here->HSMHV2_mode > 0 ) { /* forward mode */
        Rd         = here->HSMHV2_Rd ;
        dRd_dVddp  = here->HSMHV2_dRd_dVddp  ;
        dRd_dVdse  = here->HSMHV2_dRd_dVdse  ;
        dRd_dVgse  = here->HSMHV2_dRd_dVgse  ;
        dRd_dVbse  = here->HSMHV2_dRd_dVbse  ;
        dRd_dVsubs = (flg_subNode > 0) ? here->HSMHV2_dRd_dVsubs : 0.0 ; /* derivative w.r.t. Vsubs */
        dRd_dT     = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dRd_dTi : 0.0  ;
        dRd_dVds   = here->HSMHV2_dRd_dVds   ;
        dRd_dVgs   = here->HSMHV2_dRd_dVgs   ;
        dRd_dVbs   = here->HSMHV2_dRd_dVbs   ;
        Rs         = here->HSMHV2_Rs ;
        dRs_dVdse  = here->HSMHV2_dRs_dVdse  ;
        dRs_dVgse  = here->HSMHV2_dRs_dVgse  ;
        dRs_dVbse  = here->HSMHV2_dRs_dVbse  ;
        dRs_dVsubs = (flg_subNode > 0) ? here->HSMHV2_dRs_dVsubs : 0.0 ; /* derivative w.r.t. Vsubs */
        dRs_dT     = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dRs_dTi : 0.0  ;
        Ids        = here->HSMHV2_ids ;
        gds        = here->HSMHV2_dIds_dVdsi ;
        gm         = here->HSMHV2_dIds_dVgsi ;
        gmbs       = here->HSMHV2_dIds_dVbsi ;
        gmT        = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dIds_dTi : 0.0  ;
        gmbs_ext   = here->HSMHV2_dIds_dVbse ;
        gds_ext    = here->HSMHV2_dIds_dVdse ;
        gm_ext     = here->HSMHV2_dIds_dVgse ;

        Qd        = here->HSMHV2_qd ;
        dQd_dVds  = here->HSMHV2_dQdi_dVdsi ;
        dQd_dVgs  = here->HSMHV2_dQdi_dVgsi ;
        dQd_dVbs  = here->HSMHV2_dQdi_dVbsi ;
        dQd_dT    = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dQdi_dTi : 0.0  ;
        Qg         = here->HSMHV2_qg ;
        dQg_dVds   = here->HSMHV2_dQg_dVdsi ;
        dQg_dVgs   = here->HSMHV2_dQg_dVgsi ;
        dQg_dVbs   = here->HSMHV2_dQg_dVbsi ;
        dQg_dT     = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dQg_dTi : 0.0  ;
        Qs        = here->HSMHV2_qs ;
        dQs_dVds  = here->HSMHV2_dQsi_dVdsi ;
        dQs_dVgs  = here->HSMHV2_dQsi_dVgsi ;
        dQs_dVbs  = here->HSMHV2_dQsi_dVbsi ;
        dQs_dT    = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dQsi_dTi : 0.0  ;
        Qb         = - (here->HSMHV2_qg + here->HSMHV2_qd + here->HSMHV2_qs) ;
        dQb_dVds   = here->HSMHV2_dQb_dVdsi ;
        dQb_dVgs   = here->HSMHV2_dQb_dVgsi ;
        dQb_dVbs   = here->HSMHV2_dQb_dVbsi ;
        dQb_dT     = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dQb_dTi : 0.0  ;
        Qfd        = here->HSMHV2_qdp ;
        dQfd_dVdse = here->HSMHV2_dqdp_dVdse ;
        dQfd_dVgse = here->HSMHV2_dqdp_dVgse ;
        dQfd_dVbse = here->HSMHV2_dqdp_dVbse ;
        dQfd_dT    = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dqdp_dTi : 0.0  ;
        Qfs        = here->HSMHV2_qsp ;
        dQfs_dVdse = here->HSMHV2_dqsp_dVdse ;
        dQfs_dVgse = here->HSMHV2_dqsp_dVgse ;
        dQfs_dVbse = here->HSMHV2_dqsp_dVbse ;
        dQfs_dT    = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dqsp_dTi : 0.0  ;

        Qdext        = here->HSMHV2_qdext ;
        dQdext_dVdse = here->HSMHV2_dQdext_dVdse ;
        dQdext_dVgse = here->HSMHV2_dQdext_dVgse ;
        dQdext_dVbse = here->HSMHV2_dQdext_dVbse ;
        dQdext_dT    = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dQdext_dTi : 0.0  ;
        Qgext        = here->HSMHV2_qgext ;
        dQgext_dVdse = here->HSMHV2_dQgext_dVdse ;
        dQgext_dVgse = here->HSMHV2_dQgext_dVgse ;
        dQgext_dVbse = here->HSMHV2_dQgext_dVbse ;
        dQgext_dT    = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dQgext_dTi : 0.0  ;
        Qsext        = here->HSMHV2_qsext ;
        dQsext_dVdse = here->HSMHV2_dQsext_dVdse ;
        dQsext_dVgse = here->HSMHV2_dQsext_dVgse ;
        dQsext_dVbse = here->HSMHV2_dQsext_dVbse ;
        dQsext_dT    = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dQsext_dTi : 0.0  ;
        Qbext        = - (here->HSMHV2_qgext + here->HSMHV2_qdext + here->HSMHV2_qsext) ;
        dQbext_dVdse = here->HSMHV2_dQbext_dVdse ;
        dQbext_dVgse = here->HSMHV2_dQbext_dVgse ;
        dQbext_dVbse = here->HSMHV2_dQbext_dVbse ;
        dQbext_dT    = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dQbext_dTi : 0.0  ;
        Isub         = here->HSMHV2_isub ;
        dIsub_dVds   = here->HSMHV2_dIsub_dVdsi ;
        dIsub_dVgs   = here->HSMHV2_dIsub_dVgsi ;
        dIsub_dVbs   = here->HSMHV2_dIsub_dVbsi ;
        dIsub_dT     = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dIsub_dTi : 0.0  ;
        Isubs        = 0.0 ;
        dIsubs_dVds  = 0.0 ;
        dIsubs_dVgs  = 0.0 ;
        dIsubs_dVbs  = 0.0 ;
        dIsubs_dT    = 0.0 ;
        IsubLD         = here->HSMHV2_isubld ;
        dIsubLD_dVds   = here->HSMHV2_dIsubLD_dVdsi ;
        dIsubLD_dVgs   = here->HSMHV2_dIsubLD_dVgsi ;
        dIsubLD_dVbs   = here->HSMHV2_dIsubLD_dVbsi ;
        dIsubLD_dT     = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dIsubLD_dTi : 0.0  ;
        dIsubLD_dVddp  = here->HSMHV2_dIsubLD_dVddp ;
        IsubLDs        = 0.0 ;
        dIsubLDs_dVds  = 0.0 ;
        dIsubLDs_dVgs  = 0.0 ;
        dIsubLDs_dVbs  = 0.0 ;
        dIsubLDs_dT    = 0.0 ;
        dIsubLDs_dVddp = 0.0 ;
        IdsIBPC         = here->HSMHV2_idsibpc ;
        dIdsIBPC_dVds   = here->HSMHV2_dIdsIBPC_dVdsi ;
        dIdsIBPC_dVgs   = here->HSMHV2_dIdsIBPC_dVgsi ;
        dIdsIBPC_dVbs   = here->HSMHV2_dIdsIBPC_dVbsi ;
        dIdsIBPC_dT     = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dIdsIBPC_dTi : 0.0  ;
        dIdsIBPC_dVddp  = here->HSMHV2_dIdsIBPC_dVddp ;
        IdsIBPCs        = 0.0 ;
        dIdsIBPCs_dVds  = 0.0 ;
        dIdsIBPCs_dVgs  = 0.0 ;
        dIdsIBPCs_dVbs  = 0.0 ;
        dIdsIBPCs_dT    = 0.0 ;
        dIdsIBPCs_dVddp = 0.0 ;
        Igidl        = here->HSMHV2_igidl ;
        dIgidl_dVds  = here->HSMHV2_dIgidl_dVdsi ;
        dIgidl_dVgs  = here->HSMHV2_dIgidl_dVgsi ;
        dIgidl_dVbs  = here->HSMHV2_dIgidl_dVbsi ;
        dIgidl_dT    = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dIgidl_dTi : 0.0  ;
        Igisl        = here->HSMHV2_igisl ;
        dIgisl_dVds  = here->HSMHV2_dIgisl_dVdsi ;
        dIgisl_dVgs  = here->HSMHV2_dIgisl_dVgsi ;
        dIgisl_dVbs  = here->HSMHV2_dIgisl_dVbsi ;
        dIgisl_dT    = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dIgisl_dTi : 0.0  ;
        Igd          = here->HSMHV2_igd ;
        dIgd_dVd   = here->HSMHV2_dIgd_dVdsi ;
        dIgd_dVg   = here->HSMHV2_dIgd_dVgsi ;
        dIgd_dVb   = here->HSMHV2_dIgd_dVbsi ;
        dIgd_dT      = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dIgd_dTi : 0.0  ;
        Igs          = here->HSMHV2_igs ;
        dIgs_dVd   = here->HSMHV2_dIgs_dVdsi ;
        dIgs_dVg   = here->HSMHV2_dIgs_dVgsi ;
        dIgs_dVb   = here->HSMHV2_dIgs_dVbsi ;
        dIgs_dT      = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dIgs_dTi : 0.0  ;
        Igb          = here->HSMHV2_igb ;
        dIgb_dVd   = here->HSMHV2_dIgb_dVdsi ;
        dIgb_dVg   = here->HSMHV2_dIgb_dVgsi ;
        dIgb_dVb   = here->HSMHV2_dIgb_dVbsi ;
        dIgb_dT      = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dIgb_dTi : 0.0  ;

        /*---------------------------------------------------*
         * Junction diode.
         *-----------------*/
        Ibd = here->HSMHV2_ibd ;
        Gbd = here->HSMHV2_gbd ;
        Gbdt = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_gbdT : 0.0 ;

        /* Qbd = here->HSMHV2_qbd ; */
        Qbd = *(ckt->CKTstate0 + here->HSMHV2qbd) ;
        Cbd = here->HSMHV2_capbd ;
        Cbdt = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_gcbdT : 0.0 ;

        Ibs = here->HSMHV2_ibs ;
        Gbs = here->HSMHV2_gbs ;
        Gbst = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_gbsT : 0.0 ;

        /* Qbs = here->HSMHV2_qbs ; */
        Qbs = *(ckt->CKTstate0 + here->HSMHV2qbs) ;
        Cbs = here->HSMHV2_capbs ;
        Cbst = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_gcbsT : 0.0 ;

        if (flg_nqs) {
          tau         = here->HSMHV2_tau       ;
          dtau_dVds   = here->HSMHV2_tau_dVdsi ;
          dtau_dVgs   = here->HSMHV2_tau_dVgsi ;
          dtau_dVbs   = here->HSMHV2_tau_dVbsi ;
          dtau_dT     = here->HSMHV2_tau_dTi   ;
          taub        = here->HSMHV2_taub      ;
          dtaub_dVds  = here->HSMHV2_taub_dVdsi ;
          dtaub_dVgs  = here->HSMHV2_taub_dVgsi ;
          dtaub_dVbs  = here->HSMHV2_taub_dVbsi ;
          dtaub_dT    = here->HSMHV2_taub_dTi   ;
          Qdrat       = here->HSMHV2_Xd         ;
          dQdrat_dVds = here->HSMHV2_Xd_dVdsi   ;
          dQdrat_dVgs = here->HSMHV2_Xd_dVgsi   ;
          dQdrat_dVbs = here->HSMHV2_Xd_dVbsi   ;
          dQdrat_dT   = here->HSMHV2_Xd_dTi     ;
          Qi          = here->HSMHV2_Qi         ;
          dQi_dVds    = here->HSMHV2_Qi_dVdsi   ;
          dQi_dVgs    = here->HSMHV2_Qi_dVgsi   ;
          dQi_dVbs    = here->HSMHV2_Qi_dVbsi   ;
          dQi_dT      = here->HSMHV2_Qi_dTi     ;
          Qbulk       = here->HSMHV2_Qbulk       ;
          dQbulk_dVds = here->HSMHV2_Qbulk_dVdsi ;
          dQbulk_dVgs = here->HSMHV2_Qbulk_dVgsi ;
          dQbulk_dVbs = here->HSMHV2_Qbulk_dVbsi ;
          dQbulk_dT   = here->HSMHV2_Qbulk_dTi   ;
        }

      } else { /* reverse mode */
        /* note: here->HSMHV2_Rd and here->HSMHV2_Rs are already subjected to mode handling,
           while the following derivatives here->HSMHV2_Rd_dVdse, ... are not! */
        Rd        = here->HSMHV2_Rd ;
        dRd_dVddp = here->HSMHV2_dRd_dVddp  ;
        dRd_dVdse = here->HSMHV2_dRd_dVdse ;
        dRd_dVgse = here->HSMHV2_dRd_dVgse ;
        dRd_dVbse = here->HSMHV2_dRd_dVbse ;
        dRd_dVsubs= (flg_subNode > 0) ? here->HSMHV2_dRd_dVsubs : 0.0 ; /* derivative w.r.t. Vsubs */
        dRd_dT    = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dRd_dTi : 0.0  ;
        dRd_dVds  = here->HSMHV2_dRd_dVds  ;
        dRd_dVgs  = here->HSMHV2_dRd_dVgs  ;
        dRd_dVbs  = here->HSMHV2_dRd_dVbs  ;
        Rs        = here->HSMHV2_Rs ;
        dRs_dVdse = here->HSMHV2_dRs_dVdse ;
        dRs_dVgse = here->HSMHV2_dRs_dVgse ;
        dRs_dVbse = here->HSMHV2_dRs_dVbse ;
        dRs_dVsubs= (flg_subNode > 0) ? here->HSMHV2_dRs_dVsubs : 0.0 ; /* derivative w.r.t. Vsubs */
        dRs_dT    = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dRs_dTi : 0.0  ;
        Ids       = - here->HSMHV2_ids ;
        gds       = + (here->HSMHV2_dIds_dVdsi + here->HSMHV2_dIds_dVgsi + here->HSMHV2_dIds_dVbsi) ;
        gm        = - here->HSMHV2_dIds_dVgsi ;
        gmbs      = - here->HSMHV2_dIds_dVbsi ;
        gmT       = (here->HSMHV2_coselfheat > 0) ? - here->HSMHV2_dIds_dTi : 0.0  ;
        gds_ext   = + (here->HSMHV2_dIds_dVdse + here->HSMHV2_dIds_dVgse + here->HSMHV2_dIds_dVbse) ;
        gm_ext    = - here->HSMHV2_dIds_dVgse;
        gmbs_ext  = - here->HSMHV2_dIds_dVbse;

        Qd        =   here->HSMHV2_qs ;
        dQd_dVds  = - (here->HSMHV2_dQsi_dVdsi + here->HSMHV2_dQsi_dVgsi + here->HSMHV2_dQsi_dVbsi) ;
        dQd_dVgs  =   here->HSMHV2_dQsi_dVgsi ;
        dQd_dVbs  =   here->HSMHV2_dQsi_dVbsi ;
        dQd_dT    =   (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dQsi_dTi : 0.0  ;
        Qg         =   here->HSMHV2_qg ;
        dQg_dVds   = - (here->HSMHV2_dQg_dVdsi + here->HSMHV2_dQg_dVgsi + here->HSMHV2_dQg_dVbsi) ;
        dQg_dVgs   =   here->HSMHV2_dQg_dVgsi ;
        dQg_dVbs   =   here->HSMHV2_dQg_dVbsi ;
        dQg_dT     =   (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dQg_dTi : 0.0  ;
        Qs        =   here->HSMHV2_qd ;
        dQs_dVds  = - (here->HSMHV2_dQdi_dVdsi + here->HSMHV2_dQdi_dVgsi + here->HSMHV2_dQdi_dVbsi) ;
        dQs_dVgs  =   here->HSMHV2_dQdi_dVgsi ;
        dQs_dVbs  =   here->HSMHV2_dQdi_dVbsi ;
        dQs_dT    =   (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dQdi_dTi : 0.0  ;
        Qb         = - (here->HSMHV2_qg + here->HSMHV2_qd + here->HSMHV2_qs) ;
        dQb_dVds   = - (here->HSMHV2_dQb_dVdsi + here->HSMHV2_dQb_dVgsi + here->HSMHV2_dQb_dVbsi) ;
        dQb_dVgs   =   here->HSMHV2_dQb_dVgsi ;
        dQb_dVbs   =   here->HSMHV2_dQb_dVbsi ;
        dQb_dT     =   (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dQb_dTi : 0.0  ;
        Qfd        =   here->HSMHV2_qsp ;
        dQfd_dVdse = - (here->HSMHV2_dqsp_dVdse + here->HSMHV2_dqsp_dVgse + here->HSMHV2_dqsp_dVbse) ;
        dQfd_dVgse =   here->HSMHV2_dqsp_dVgse ;
        dQfd_dVbse =   here->HSMHV2_dqsp_dVbse ;
        dQfd_dT    =   (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dqsp_dTi : 0.0  ;
        Qfs        =   here->HSMHV2_qdp ;
        dQfs_dVdse = - (here->HSMHV2_dqdp_dVdse + here->HSMHV2_dqdp_dVgse + here->HSMHV2_dqdp_dVbse) ;
        dQfs_dVgse =   here->HSMHV2_dqdp_dVgse ;
        dQfs_dVbse =   here->HSMHV2_dqdp_dVbse ;
        dQfs_dT    =   (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dqdp_dTi : 0.0  ;

        Qdext        = here->HSMHV2_qsext ;
        dQdext_dVdse = - (here->HSMHV2_dQsext_dVdse + here->HSMHV2_dQsext_dVgse + here->HSMHV2_dQsext_dVbse);
        dQdext_dVgse = here->HSMHV2_dQsext_dVgse ;
        dQdext_dVbse = here->HSMHV2_dQsext_dVbse ;
        dQdext_dT    = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dQsext_dTi : 0.0  ;
        Qgext        = here->HSMHV2_qgext ;
        dQgext_dVdse = - (here->HSMHV2_dQgext_dVdse + here->HSMHV2_dQgext_dVgse + here->HSMHV2_dQgext_dVbse);
        dQgext_dVgse = here->HSMHV2_dQgext_dVgse ;
        dQgext_dVbse = here->HSMHV2_dQgext_dVbse ;
        dQgext_dT    = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dQgext_dTi : 0.0  ;
        Qsext        = here->HSMHV2_qdext ;
        dQsext_dVdse = - (here->HSMHV2_dQdext_dVdse + here->HSMHV2_dQdext_dVgse + here->HSMHV2_dQdext_dVbse);
        dQsext_dVgse = here->HSMHV2_dQdext_dVgse ;
        dQsext_dVbse = here->HSMHV2_dQdext_dVbse ;
        dQsext_dT    = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dQdext_dTi : 0.0  ;
        Qbext        = - (here->HSMHV2_qgext + here->HSMHV2_qdext + here->HSMHV2_qsext) ;
        dQbext_dVdse = - (here->HSMHV2_dQbext_dVdse + here->HSMHV2_dQbext_dVgse + here->HSMHV2_dQbext_dVbse);
        dQbext_dVgse = here->HSMHV2_dQbext_dVgse ;
        dQbext_dVbse = here->HSMHV2_dQbext_dVbse ;
        dQbext_dT    = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dQbext_dTi : 0.0  ;
        Isub         = 0.0 ;
        dIsub_dVds   = 0.0 ;
        dIsub_dVgs   = 0.0 ;
        dIsub_dVbs   = 0.0 ;
        dIsub_dT     = 0.0 ;
        Isubs        =   here->HSMHV2_isub ;
        dIsubs_dVds  = - (here->HSMHV2_dIsub_dVdsi + here->HSMHV2_dIsub_dVgsi + here->HSMHV2_dIsub_dVbsi) ;
        dIsubs_dVgs  =   here->HSMHV2_dIsub_dVgsi ;
        dIsubs_dVbs  =   here->HSMHV2_dIsub_dVbsi ;
        dIsubs_dT    =   (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dIsub_dTi : 0.0 ;
        IsubLD         = 0.0 ;
        dIsubLD_dVds   = 0.0 ;
        dIsubLD_dVgs   = 0.0 ;
        dIsubLD_dVbs   = 0.0 ;
        dIsubLD_dT     = 0.0 ;
        dIsubLD_dVddp  = 0.0 ;
        IsubLDs        =   here->HSMHV2_isubld ;
        dIsubLDs_dVds  = - (here->HSMHV2_dIsubLD_dVdsi + here->HSMHV2_dIsubLD_dVgsi + here->HSMHV2_dIsubLD_dVbsi) ;
        dIsubLDs_dVgs  =   here->HSMHV2_dIsubLD_dVgsi ;
        dIsubLDs_dVbs  =   here->HSMHV2_dIsubLD_dVbsi ;
        dIsubLDs_dT    =   (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dIsubLD_dTi : 0.0 ;
        dIsubLDs_dVddp = - here->HSMHV2_dIsubLD_dVddp ;
        IdsIBPC         = 0.0 ;
        dIdsIBPC_dVds   = 0.0 ;
        dIdsIBPC_dVgs   = 0.0 ;
        dIdsIBPC_dVbs   = 0.0 ;
        dIdsIBPC_dT     = 0.0 ;
        dIdsIBPC_dVddp  = 0.0 ;
        IdsIBPCs        =   here->HSMHV2_idsibpc ;
        dIdsIBPCs_dVds  = - (here->HSMHV2_dIdsIBPC_dVdsi + here->HSMHV2_dIdsIBPC_dVgsi + here->HSMHV2_dIdsIBPC_dVbsi) ;
        dIdsIBPCs_dVgs  =   here->HSMHV2_dIdsIBPC_dVgsi ;
        dIdsIBPCs_dVbs  =   here->HSMHV2_dIdsIBPC_dVbsi ;
        dIdsIBPCs_dT    =   (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dIdsIBPC_dTi : 0.0 ;
        dIdsIBPCs_dVddp = - here->HSMHV2_dIdsIBPC_dVddp ;
        Igidl        =   here->HSMHV2_igisl ;
        dIgidl_dVds  = - (here->HSMHV2_dIgisl_dVdsi + here->HSMHV2_dIgisl_dVgsi + here->HSMHV2_dIgisl_dVbsi) ;
        dIgidl_dVgs  =   here->HSMHV2_dIgisl_dVgsi ;
        dIgidl_dVbs  =   here->HSMHV2_dIgisl_dVbsi ;
        dIgidl_dT    =   (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dIgisl_dTi : 0.0  ;
        Igisl        =   here->HSMHV2_igidl ;
        dIgisl_dVds  = - (here->HSMHV2_dIgidl_dVdsi + here->HSMHV2_dIgidl_dVgsi + here->HSMHV2_dIgidl_dVbsi) ;
        dIgisl_dVgs  =   here->HSMHV2_dIgidl_dVgsi ;
        dIgisl_dVbs  =   here->HSMHV2_dIgidl_dVbsi ;
        dIgisl_dT    =   (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dIgidl_dTi : 0.0  ;
        /* note: here->HSMHV2_igd and here->HSMHV2_igs are already subjected to mode handling,
           while the following derivatives here->HSMHV2_dIgd_dVdsi, ... are not! */
        Igd          =   here->HSMHV2_igd ;
        dIgd_dVd   = - (here->HSMHV2_dIgs_dVdsi + here->HSMHV2_dIgs_dVgsi + here->HSMHV2_dIgs_dVbsi) ;
        dIgd_dVg   =   here->HSMHV2_dIgs_dVgsi ;
        dIgd_dVb   =   here->HSMHV2_dIgs_dVbsi ;
        dIgd_dT      =   (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dIgs_dTi : 0.0  ;
        Igs          =   here->HSMHV2_igs ;
        dIgs_dVd   = - (here->HSMHV2_dIgd_dVdsi + here->HSMHV2_dIgd_dVgsi + here->HSMHV2_dIgd_dVbsi) ;
        dIgs_dVg   =   here->HSMHV2_dIgd_dVgsi ;
        dIgs_dVb   =   here->HSMHV2_dIgd_dVbsi ;
        dIgs_dT      =   (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dIgd_dTi : 0.0  ;
        Igb          =   here->HSMHV2_igb ;
        dIgb_dVd   = - (here->HSMHV2_dIgb_dVdsi + here->HSMHV2_dIgb_dVgsi + here->HSMHV2_dIgb_dVbsi) ;
        dIgb_dVg   =   here->HSMHV2_dIgb_dVgsi ;
        dIgb_dVb   =   here->HSMHV2_dIgb_dVbsi ;
        dIgb_dT      =   (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_dIgb_dTi : 0.0  ;

        /*---------------------------------------------------*
         * Junction diode.
         *-----------------*/
        Ibd = here->HSMHV2_ibd ;
        Gbd = here->HSMHV2_gbd ;
        Gbdt = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_gbdT : 0.0 ;

        /* Qbd = here->HSMHV2_qbd ; */
        Qbd = *(ckt->CKTstate0 + here->HSMHV2qbd) ;
        Cbd = here->HSMHV2_capbd ;
        Cbdt = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_gcbdT : 0.0 ;

        Ibs = here->HSMHV2_ibs ;
        Gbs = here->HSMHV2_gbs ;
        Gbst = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_gbsT : 0.0 ;

        /* Qbs = here->HSMHV2_qbs ; */
        Qbs = *(ckt->CKTstate0 + here->HSMHV2qbs) ;
        Cbs = here->HSMHV2_capbs ;
        Cbst = (here->HSMHV2_coselfheat > 0) ? here->HSMHV2_gcbsT : 0.0 ;

        if (flg_nqs) {
          tau         =   here->HSMHV2_tau       ;
          dtau_dVds   = -(here->HSMHV2_tau_dVdsi + here->HSMHV2_tau_dVgsi + here->HSMHV2_tau_dVbsi) ;
          dtau_dVgs   =   here->HSMHV2_tau_dVgsi ;
          dtau_dVbs   =   here->HSMHV2_tau_dVbsi ;
          dtau_dT     =   here->HSMHV2_tau_dTi   ;
          taub        =   here->HSMHV2_taub      ;
          dtaub_dVds  = -(here->HSMHV2_taub_dVdsi + here->HSMHV2_taub_dVgsi + here->HSMHV2_taub_dVbsi);
          dtaub_dVgs  =   here->HSMHV2_taub_dVgsi ;
          dtaub_dVbs  =   here->HSMHV2_taub_dVbsi ;
          dtaub_dT    =   here->HSMHV2_taub_dTi   ;
          Qdrat       =   1.0 - here->HSMHV2_Xd         ;
          dQdrat_dVds = +(here->HSMHV2_Xd_dVdsi + here->HSMHV2_Xd_dVgsi + here->HSMHV2_Xd_dVbsi) ;
          dQdrat_dVgs = - here->HSMHV2_Xd_dVgsi   ;
          dQdrat_dVbs = - here->HSMHV2_Xd_dVbsi   ;
          dQdrat_dT   = - here->HSMHV2_Xd_dTi     ;
          Qi          =   here->HSMHV2_Qi         ;
          dQi_dVds    = -(here->HSMHV2_Qi_dVdsi + here->HSMHV2_Qi_dVgsi + here->HSMHV2_Qi_dVbsi) ;
          dQi_dVgs    =   here->HSMHV2_Qi_dVgsi   ;
          dQi_dVbs    =   here->HSMHV2_Qi_dVbsi   ;
          dQi_dT      =   here->HSMHV2_Qi_dTi     ;
          Qbulk       =   here->HSMHV2_Qbulk       ;
          dQbulk_dVds = -(here->HSMHV2_Qbulk_dVdsi + here->HSMHV2_Qbulk_dVgsi + here->HSMHV2_Qbulk_dVbsi) ;
          dQbulk_dVgs =   here->HSMHV2_Qbulk_dVgsi ;
          dQbulk_dVbs =   here->HSMHV2_Qbulk_dVbsi ;
          dQbulk_dT   =   here->HSMHV2_Qbulk_dTi   ;
        }
      } /* end of reverse mode */

      if (here->HSMHV2_coselfheat > 0) {
        if (pParam->HSMHV2_rth > C_RTH_MIN) {
          Gth = 1.0/pParam->HSMHV2_rth ;
        } else {
          Gth = 1.0/C_RTH_MIN ;
        }
        Ith         = Gth * deltemp ;
        dIth_dT     = Gth ;
        Cth         = pParam->HSMHV2_cth ;
        Qth         = Cth * deltemp ;
        /*     P = Ids * (Vdsi + param * ( Vdse - Vdsi)) */
        /*       = Ids * Veffpower                       */
        if ( vds * (vdse - vds) >= 0.0) {
          if ( pParam->HSMHV2_powrat == 1.0 ) {
            Veffpower        = vdse ;
            dVeffpower_dVds  = 0.0 ;
            dVeffpower_dVdse = 1.0 ;
          } else {
            Veffpower        = vds + here->HSMHV2_powratio * (vdse - vds) ;
            dVeffpower_dVds  = (1.0 - here->HSMHV2_powratio) ;
            dVeffpower_dVdse = here->HSMHV2_powratio ;
          }
        } else {
          Veffpower        = vds ;
          dVeffpower_dVds  = 1.0 ;
          dVeffpower_dVdse = 0.0 ;
        }
        P           = Ids  * Veffpower ;
        dP_dVds     = gds  * Veffpower + Ids * dVeffpower_dVds;
        dP_dVgs     = gm   * Veffpower ;
        dP_dVbs     = gmbs * Veffpower ;
        dP_dT       = gmT  * Veffpower ;
        dP_dVdse    = gds_ext  * Veffpower + Ids * dVeffpower_dVdse ;
        dP_dVgse    = gm_ext   * Veffpower ;
        dP_dVbse    = gmbs_ext * Veffpower ;

        /* Clamping the maximum rise tempaerarure (SHEMAX) */
        T1 = model->HSMHV2_shemax * Gth ;
        Fn_SU( T2 , P , T1 , SHE_MAX_dlt * Gth , T0 ) ;
        P = T2 ;
        dP_dVds     = T0 * dP_dVds ;
        dP_dVgs     = T0 * dP_dVgs ;
        dP_dVbs     = T0 * dP_dVbs ;
        dP_dT       = T0 * dP_dT ;
        dP_dVdse    = T0 * dP_dVdse ;
        dP_dVgse    = T0 * dP_dVgse ;
        dP_dVbse    = T0 * dP_dVbse ;

      } else {
        Gth         = 0.0 ;
        Ith         = 0.0 ;
        dIth_dT     = 0.0 ;
        Cth         = 0.0 ;
        Qth         = 0.0 ;
        P           = 0.0 ;
        dP_dVds     = 0.0 ;
        dP_dVgs     = 0.0 ;
        dP_dVbs     = 0.0 ;
        dP_dT       = 0.0 ;
        dP_dVdse    = 0.0 ;
        dP_dVgse    = 0.0 ;
        dP_dVbse    = 0.0 ;
      }

      /* in case of nqs: construct static contributions to the nqs equations (Iqi_nqs, Iqb_nqs)       */
      /*   and nqs charge contributions to inner drain, gate and source node (Qd_nqs, Qg_nqs, Qs_nqs) */

      if (flg_nqs) {
        /* .. tau, taub must be > 0 */
        if (tau < 1.0e-18) {
          tau = 1.e-18 ;
          dtau_dVds = dtau_dVgs = dtau_dVbs = dtau_dT = 0.0 ;
        }
        if (taub < 1.0e-18) {
          taub = 1.0e-18 ;
          dtaub_dVds = dtaub_dVgs = dtaub_dVbs = dtaub_dT = 0.0 ;
        }

        Iqi_nqs          = (Qi_nqs - Qi) / tau ;
        dIqi_nqs_dVds    = - (dQi_dVds + Iqi_nqs * dtau_dVds) / tau ;
        dIqi_nqs_dVgs    = - (dQi_dVgs + Iqi_nqs * dtau_dVgs) / tau ;
        dIqi_nqs_dVbs    = - (dQi_dVbs + Iqi_nqs * dtau_dVbs) / tau ;
        dIqi_nqs_dT      = - (dQi_dT   + Iqi_nqs * dtau_dT  ) / tau ;
        dIqi_nqs_dQi_nqs = 1.0 / tau ;

        Iqb_nqs          = (Qb_nqs - Qbulk) / taub ;
        dIqb_nqs_dVds    = - (dQbulk_dVds + Iqb_nqs * dtaub_dVds) / taub ;
        dIqb_nqs_dVgs    = - (dQbulk_dVgs + Iqb_nqs * dtaub_dVgs) / taub ;
        dIqb_nqs_dVbs    = - (dQbulk_dVbs + Iqb_nqs * dtaub_dVbs) / taub ;
        dIqb_nqs_dT      = - (dQbulk_dT   + Iqb_nqs * dtaub_dT  ) / taub ;
        dIqb_nqs_dQb_nqs = 1.0 / taub ;

        Qd_nqs           = Qi_nqs * Qdrat ;
        dQd_nqs_dVds     = Qi_nqs * dQdrat_dVds ;
        dQd_nqs_dVgs     = Qi_nqs * dQdrat_dVgs ;
        dQd_nqs_dVbs     = Qi_nqs * dQdrat_dVbs ;
        dQd_nqs_dT       = Qi_nqs * dQdrat_dT   ;
        dQd_nqs_dQi_nqs  = Qdrat ;

        Qg_nqs           = - Qi_nqs - Qb_nqs ;
        dQg_nqs_dQi_nqs  = - 1.0 ;
        dQg_nqs_dQb_nqs  = - 1.0 ;

        Qs_nqs           =   Qi_nqs * (1.0 - Qdrat) ;
        dQs_nqs_dVds     = - Qi_nqs * dQdrat_dVds ;
        dQs_nqs_dVgs     = - Qi_nqs * dQdrat_dVgs ;
        dQs_nqs_dVbs     = - Qi_nqs * dQdrat_dVbs ;
        dQs_nqs_dT       = - Qi_nqs * dQdrat_dT   ;
        dQs_nqs_dQi_nqs  =   1.0 - Qdrat ;
      } else {
        Iqi_nqs = Iqb_nqs = Qd_nqs = Qg_nqs = Qs_nqs = 0.0 ;
      }

      dIgd_dVs = - (dIgd_dVd + dIgd_dVg + dIgd_dVb) ;
      dIgs_dVs = - (dIgs_dVd + dIgs_dVg + dIgs_dVb) ;
      dIgb_dVs = - (dIgb_dVd + dIgb_dVg + dIgb_dVb) ;

     /*---------------------------------------------------*
      * External Resistances
      *-----------------*/
      if(model->HSMHV2_corsrd == 1 || model->HSMHV2_corsrd == 3 || model->HSMHV2_cordrift == 1 ) {
        if(Rd > 0){
          GD = 1.0/Rd;
          GD_dVgs   = - dRd_dVgs   /Rd/Rd;
          GD_dVds   = - dRd_dVds   /Rd/Rd;
          GD_dVddp  = - dRd_dVddp  /Rd/Rd;
          GD_dVbs   = - dRd_dVbs   /Rd/Rd;
          GD_dVsubs = - dRd_dVsubs /Rd/Rd;
          GD_dT     = - dRd_dT     /Rd/Rd;
          GD_dVgse  = - dRd_dVgse  /Rd/Rd;
          GD_dVdse  = - dRd_dVdse  /Rd/Rd;
          GD_dVbse  = - dRd_dVbse  /Rd/Rd;
        }else{
          GD=0.0;
          GD_dVgs=0.0;
          GD_dVds=0.0;
          GD_dVddp = 0.0;
          GD_dVbs=0.0;
          GD_dVsubs=0.0;
          GD_dT  =0.0;
          GD_dVgse  =0.0;
          GD_dVdse  =0.0;
          GD_dVbse  =0.0;
        }
        if(Rs > 0){
          GS = 1.0/Rs;
          GS_dVgs = - dRs_dVgse /Rs/Rs;
          GS_dVds = - dRs_dVdse /Rs/Rs;
          GS_dVbs = - dRs_dVbse /Rs/Rs;
          GS_dVsubs = - dRs_dVsubs /Rs/Rs;
          GS_dT = - dRs_dT /Rs/Rs;
        }else{
          GS=0.0;
          GS_dVgs=0.0;
          GS_dVds=0.0;
          GS_dVbs=0.0;
          GS_dVsubs=0.0;
          GS_dT  =0.0;
        }
      }
      Iddp        = GD        * vddp;
      dIddp_dVddp = GD_dVddp  * vddp + GD ;
      dIddp_dVdse = GD_dVdse  * vddp;
      dIddp_dVgse = GD_dVgse  * vddp;
      dIddp_dVbse = GD_dVbse  * vddp;
      dIddp_dVsubs= GD_dVsubs * vddp;
      dIddp_dT    = GD_dT     * vddp;
      dIddp_dVds  = GD_dVds   * vddp;
      dIddp_dVgs  = GD_dVgs   * vddp;
      dIddp_dVbs  = GD_dVbs   * vddp;

      Issp        = GS * vssp;
      dIssp_dVssp = GS;
      dIssp_dVdse = GS_dVds * vssp ;
      dIssp_dVgse = GS_dVgs * vssp;
      dIssp_dVbse = GS_dVbs * vssp;
      dIssp_dVsubs= GS_dVsubs * vssp;
      dIssp_dT    = GS_dT * vssp;

      if( model->HSMHV2_corg > 0.0 ){
        GG = here->HSMHV2_grg ;
      }else{
        GG = 0.0 ;
      }
      Iggp        = GG * vggp;
      dIggp_dVggp = GG;

      if(model->HSMHV2_corbnet == 1 && here->HSMHV2_rbpb > 0.0 ){
        GRBPB = here->HSMHV2_grbpb ;
      }else{
        GRBPB = 0.0 ;
      }
      Ibpb        = GRBPB * vbpb;
      dIbpb_dVbpb = GRBPB;

      if(model->HSMHV2_corbnet == 1 && here->HSMHV2_rbpd > 0.0 ){
        GRBPD = here->HSMHV2_grbpd ;
      }else{
        GRBPD = 0.0 ;
      }
      Ibpdb         = GRBPD * vbpdb;
      dIbpdb_dVbpdb = GRBPD;

      if(model->HSMHV2_corbnet == 1 && here->HSMHV2_rbps > 0.0 ){
        GRBPS = here->HSMHV2_grbps ;
      }else{
        GRBPS = 0.0 ;
      }
      Ibpsb         = GRBPS * vbpsb;
      dIbpsb_dVbpsb = GRBPS;

      /* printf("HSMHV2_load: ByPass=%d\n",ByPass) ; */

      if (!ByPass) { /* no convergence check in case of Bypass */
        /*
         *  check convergence
         */
        isConv = 1;
        if ( (here->HSMHV2_off == 0) || !(ckt->CKTmode & MODEINITFIX) ) {
          if (Check == 1) {
            ckt->CKTnoncon++;
            isConv = 0;
#ifndef NEWCONV
          } else {
          /* convergence check for branch currents is done in function HSMHV2convTest */
#endif /* NEWCONV */
          }
        }

        *(ckt->CKTstate0 + here->HSMHV2vbs) = vbs;
        *(ckt->CKTstate0 + here->HSMHV2vbd) = vbd;
        *(ckt->CKTstate0 + here->HSMHV2vgs) = vgs;
        *(ckt->CKTstate0 + here->HSMHV2vds) = vds;
        *(ckt->CKTstate0 + here->HSMHV2vsbs) = vsbs;
        *(ckt->CKTstate0 + here->HSMHV2vdbs) = vdbs;
        *(ckt->CKTstate0 + here->HSMHV2vdbd) = vdbd;
        *(ckt->CKTstate0 + here->HSMHV2vges) = vges;
        *(ckt->CKTstate0 + here->HSMHV2vsubs) = vsubs;
        *(ckt->CKTstate0 + here->HSMHV2deltemp) = deltemp;
        *(ckt->CKTstate0 + here->HSMHV2vdse) = vdse;
        *(ckt->CKTstate0 + here->HSMHV2vgse) = vgse;
        *(ckt->CKTstate0 + here->HSMHV2vbse) = vbse;
        if ( flg_nqs ) {
          *(ckt->CKTstate0 + here->HSMHV2qi_nqs) = Qi_nqs;
          *(ckt->CKTstate0 + here->HSMHV2qb_nqs) = Qb_nqs;
        }
        /* printf("HSMHV2_load: (into state0) vds.. = %e %e %e %e %e %e\n",
                                               vds,vgs,vbs,vdse,vgse,vbse); */

        if ((ckt->CKTmode & MODEDC) &&
          !(ckt->CKTmode & MODEINITFIX) && !(ckt->CKTmode & MODEINITJCT))
          showPhysVal = 1;
        if (model->HSMHV2_show_Given && showPhysVal && isConv) {
          static int isFirst = 1;
          if (vds != vds_pre)
            ShowPhysVals(here, model, isFirst, vds_pre, vgs, vbs, vgd, vbd, vgb);
          else
            ShowPhysVals(here, model, isFirst, vds, vgs, vbs, vgd, vbd, vgb);
          if (isFirst) isFirst = 0;
        }
      }

#include "hsmhv2ld_info_eval.h"

      /* For standard Newton method (SPICE_rhs == 0):                          */
      /*   if currents (and charges) are limited -> extrapolate onto x-values  */
      /* in SPICE mode (SPICE_rhs == 1):                                       */
      /*   extrapolate onto x = 0 (-> rhs_eq)                                  */
      /*                                                                       */
      /* note that                                                             */
      /* the charge extrapolation is replaced by extrapolation of displacement */
      /* currents, see below                                                   */

      /* ...... just for easier handling: collect node voltages in vector x:   */
      if (!SPICE_rhs) {
        x[dNode]      = model->HSMHV2_type *( *(ckt->CKTrhsOld+here->HSMHV2dNode));
        x[dNodePrime] = model->HSMHV2_type *( *(ckt->CKTrhsOld+here->HSMHV2dNodePrime));
        x[gNode]      = model->HSMHV2_type *( *(ckt->CKTrhsOld+here->HSMHV2gNode));
        x[gNodePrime] = model->HSMHV2_type *( *(ckt->CKTrhsOld+here->HSMHV2gNodePrime));
        x[sNode]      = model->HSMHV2_type *( *(ckt->CKTrhsOld+here->HSMHV2sNode));
        x[sNodePrime] = model->HSMHV2_type *( *(ckt->CKTrhsOld+here->HSMHV2sNodePrime));
        x[bNodePrime] = model->HSMHV2_type *( *(ckt->CKTrhsOld+here->HSMHV2bNodePrime));
        x[bNode]      = model->HSMHV2_type *( *(ckt->CKTrhsOld+here->HSMHV2bNode));
        x[dbNode]     = model->HSMHV2_type *( *(ckt->CKTrhsOld+here->HSMHV2dbNode));
        x[sbNode]     = model->HSMHV2_type *( *(ckt->CKTrhsOld+here->HSMHV2sbNode));
        if (flg_subNode > 0)
          x[subNode]  = model->HSMHV2_type *( *(ckt->CKTrhsOld+here->HSMHV2subNode)); /* previous vsub */
        else
          x[subNode]  = 0.0;
        if (here->HSMHV2_coselfheat > 0)
          x[tempNode] =  *(ckt->CKTrhsOld+here->HSMHV2tempNode);
        else
          x[tempNode] = 0.0;
        if ( flg_nqs ) {
          x[qiNode] =  *(ckt->CKTrhsOld+here->HSMHV2qiNode);
          x[qbNode] =  *(ckt->CKTrhsOld+here->HSMHV2qbNode);
        } else {
          x[qiNode] = 0.0;
          x[qbNode] = 0.0;
        }
      }

      delvgs = (x[gNodePrime] - x[sNodePrime]) - vgs;
      delvds = (x[dNodePrime] - x[sNodePrime]) - vds;
      delvbs = (x[bNodePrime] - x[sNodePrime]) - vbs;
      deldeltemp = x[tempNode] - deltemp;

      if (delvgs || delvds || delvbs ||deldeltemp) {
        Ids  += gm         *delvgs + gds        *delvds + gmbs       *delvbs + gmT      *deldeltemp ;
        Isub += dIsub_dVgs *delvgs + dIsub_dVds *delvds + dIsub_dVbs *delvbs + dIsub_dT *deldeltemp ;
        Isubs+= dIsubs_dVgs*delvgs + dIsubs_dVds*delvds + dIsubs_dVbs*delvbs + dIsubs_dT*deldeltemp ;
        IsubLD += dIsubLD_dVgs *delvgs + dIsubLD_dVds *delvds + dIsubLD_dVbs *delvbs + dIsubLD_dT *deldeltemp ;
        IsubLDs+= dIsubLDs_dVgs*delvgs + dIsubLDs_dVds*delvds + dIsubLDs_dVbs*delvbs + dIsubLDs_dT*deldeltemp ;
        IdsIBPC += dIdsIBPC_dVgs *delvgs + dIdsIBPC_dVds *delvds + dIdsIBPC_dVbs *delvbs + dIdsIBPC_dT *deldeltemp ;
        IdsIBPCs+= dIdsIBPCs_dVgs*delvgs + dIdsIBPCs_dVds*delvds + dIdsIBPCs_dVbs*delvbs + dIdsIBPCs_dT*deldeltemp ;
        Igd  += dIgd_dVg   *delvgs + dIgd_dVd   *delvds + dIgd_dVb   *delvbs + dIgd_dT  *deldeltemp ;
        Igs  += dIgs_dVg   *delvgs + dIgs_dVd   *delvds + dIgs_dVb   *delvbs + dIgs_dT  *deldeltemp ;
        Igb  += dIgb_dVg   *delvgs + dIgb_dVd   *delvds + dIgb_dVb   *delvbs + dIgb_dT  *deldeltemp ;
        Igidl+= dIgidl_dVgs*delvgs + dIgidl_dVds*delvds + dIgidl_dVbs*delvbs + dIgidl_dT*deldeltemp ;
        Igisl+= dIgisl_dVgs*delvgs + dIgisl_dVds*delvds + dIgisl_dVbs*delvbs + dIgisl_dT*deldeltemp ;
        P    += dP_dVgs    *delvgs + dP_dVds    *delvds + dP_dVbs    *delvbs + dP_dT    *deldeltemp ;
        if (flg_nqs) {
          Iqi_nqs += dIqi_nqs_dVgs*delvgs + dIqi_nqs_dVds*delvds + dIqi_nqs_dVbs*delvbs + dIqi_nqs_dT*deldeltemp ;
          Iqb_nqs += dIqb_nqs_dVgs*delvgs + dIqb_nqs_dVds*delvds + dIqb_nqs_dVbs*delvbs + dIqb_nqs_dT*deldeltemp ;
        }
      }

      delvgse = (x[gNodePrime] - x[sNode]) - vgse;
      delvdse = (x[dNode]      - x[sNode]) - vdse;
      delvbse = (x[bNodePrime] - x[sNode]) - vbse;
      if (flg_subNode > 0) delvsubs = (x[subNode] - x[sNode]) - vsubs; /* substrate bias change */

      if (delvgse || delvdse || delvbse ) {
        Ids  += gm_ext     *delvgse + gds_ext    *delvdse + gmbs_ext   *delvbse ;
        P    += dP_dVgse   *delvgse + dP_dVdse   *delvdse + dP_dVbse   *delvbse ;
        Iddp += dIddp_dVgse*delvgse + dIddp_dVdse*delvdse + dIddp_dVbse*delvbse ;
        Issp += dIssp_dVgse*delvgse + dIssp_dVdse*delvdse + dIssp_dVbse*delvbse ;
      }

      if (delvsubs) {
        Iddp += dIddp_dVsubs*delvsubs ;
        Issp += dIssp_dVsubs*delvsubs ;
      }

      if (deldeltemp) {
        Iddp += dIddp_dT*deldeltemp ;
        Issp += dIssp_dT*deldeltemp ;
        Ith  += dIth_dT *deldeltemp ;
      }

      delvdbd = (x[dbNode] - x[dNode]) - vbd_jct ;
      if (delvdbd || deldeltemp) {
        Ibd += Gbd*delvdbd + Gbdt*deldeltemp ;
      }

      delvsbs = (x[sbNode] - x[sNode]) - vbs_jct ;
      if (delvsbs || deldeltemp) {
        Ibs += Gbs*delvsbs + Gbst*deldeltemp ;
      }

      delvddp = (x[dNode] - x[dNodePrime]) - vddp ;
      if (delvddp) {
        Iddp += dIddp_dVddp * delvddp ;
        IsubLD  += dIsubLD_dVddp   * delvddp ;
        IsubLDs += dIsubLDs_dVddp  * delvddp ;
        IdsIBPC += dIdsIBPC_dVddp  * delvddp ;
        IdsIBPCs+= dIdsIBPCs_dVddp * delvddp ;
      }

      delvds = (x[dNodePrime] - x[sNodePrime]) - vds ;
      if (delvds) {
        Iddp += dIddp_dVds * delvds ;
      }

      delvgs = (x[gNodePrime] - x[sNodePrime]) - vgs ;
      if (delvgs) {
        Iddp += dIddp_dVgs * delvgs ;
      }

      delvbs = (x[bNodePrime] - x[sNodePrime]) - vbs ;
      if (delvbs) {
        Iddp += dIddp_dVbs * delvbs ;
      }

      delvssp = (x[sNode] - x[sNodePrime]) - vssp ;
      if (delvssp) {
        Issp += dIssp_dVssp * delvssp ;
      }

      delvggp = (x[gNode] - x[gNodePrime]) - vggp ;
      if (delvggp) {
        Iggp += dIggp_dVggp * delvggp ;
      }

      delvbpb = (x[bNodePrime] - x[bNode]) - vbpb ;
      if (delvbpb) {
        Ibpb += dIbpb_dVbpb * delvbpb ;
      }

      delvbpdb = (x[bNodePrime] - x[dbNode]) - vbpdb ;
      if (delvbpdb) {
        Ibpdb += dIbpdb_dVbpdb * delvbpdb ;
      }

      delvbpsb = (x[bNodePrime] - x[sbNode]) - vbpsb ;
      if (delvbpsb) {
        Ibpsb += dIbpsb_dVbpsb * delvbpsb ;
      }

      if (flg_nqs) {
        delQi_nqs = x[qiNode] - Qi_nqs ;
        if (delQi_nqs) {
          Iqi_nqs += dIqi_nqs_dQi_nqs * delQi_nqs ;
        }
        delQb_nqs = x[qbNode] - Qb_nqs ;
        if (delQb_nqs) {
          Iqb_nqs += dIqb_nqs_dQb_nqs * delQb_nqs ;
        }
      }


      /* Assemble currents into nodes */
      /* ... static part              */


      /*  drain node  */
      i_d = Iddp - Ibd + IsubLD + IdsIBPC - IdsIBPCs ;
      /*  intrinsic drain node */
      i_dP = -Iddp + Ids + Isub + Igidl - Igd ;
      /*  gate node */
      i_g = Iggp ;
      /*  intrinsic gate node */
      i_gP = - Iggp + Igd + Igs + Igb ;
      /*  source node  */
      i_s = Issp - Ibs + IsubLDs - IdsIBPC + IdsIBPCs ;
      /*  intrinsic source node  */
      i_sP = - Issp - Ids + Isubs + Igisl - Igs ;
      /*  intrinsic bulk node */
      i_bP = - Isub - Isubs - IsubLD - IsubLDs- Igidl -Igb - Igisl  + Ibpdb + Ibpb + Ibpsb ;
      /*  base node */
      i_b = - Ibpb ;
      /*  drain bulk node  */
      i_db = Ibd - Ibpdb ;
      /*  source bulk node  */
      i_sb = Ibs - Ibpsb ;
      /*  temp node  */
      if (here->HSMHV2_coselfheat > 0){
        i_t = Ith - P ;
      } else {
        i_t = 0.0;
      }
      /* nqs equations */
      i_qi = Iqi_nqs ;
      i_qb = Iqb_nqs ;

      for (i = 0; i < XDIM; i++) {
        ydc_d[i] = ydc_dP[i] = ydc_g[i] = ydc_gP[i] = ydc_s[i] = ydc_sP[i] = ydc_bP[i] = ydc_b[i]
        = ydc_db[i] = ydc_sb[i] = ydc_t[i] = 0.0;
      }
      if (flg_nqs) {
        for (i = 0; i < XDIM; i++) {
           ydc_qi[i] = ydc_qb[i] = 0.0;
        }
      }

      /*  drain node  */
      ydc_d[dNode] = dIddp_dVddp + dIddp_dVdse + Gbd + dIsubLD_dVddp + dIdsIBPC_dVddp - dIdsIBPCs_dVddp ;
      ydc_d[dNodePrime] = -dIddp_dVddp + dIddp_dVds + dIsubLD_dVds  + dIdsIBPC_dVds - dIdsIBPCs_dVds - dIsubLD_dVddp  - dIdsIBPC_dVddp + dIdsIBPCs_dVddp ;
      /* ydc_d[gNode] = 0.0 ; */
      ydc_d[gNodePrime] = dIddp_dVgse + dIddp_dVgs + dIsubLD_dVgs  + dIdsIBPC_dVgs - dIdsIBPCs_dVgs ;
      ydc_d[sNode] = - ( dIddp_dVdse + dIddp_dVgse + dIddp_dVbse ) - dIddp_dVsubs ;
      ydc_d[sNodePrime] =  - dIsubLD_dVds - dIsubLD_dVgs - dIsubLD_dVbs  - dIdsIBPC_dVds  - dIdsIBPC_dVgs  - dIdsIBPC_dVbs  - (- dIdsIBPCs_dVds  - dIdsIBPCs_dVgs  - dIdsIBPCs_dVbs ) - ( dIddp_dVds + dIddp_dVgs + dIddp_dVbs ) ;
      ydc_d[bNodePrime] =  dIddp_dVbse + dIddp_dVbs + dIsubLD_dVbs  + dIdsIBPC_dVbs - dIdsIBPCs_dVbs ;
      /* ydc_d[bNode] = 0.0 ; */
      ydc_d[dbNode] = - Gbd ;
      /* ydc_d[sbNode] = 0.0 ; */
      ydc_d[subNode] = dIddp_dVsubs ;
      ydc_d[tempNode] = dIddp_dT - Gbdt + dIsubLD_dT  + dIdsIBPC_dT - dIdsIBPCs_dT ;

      /*  intrinsic drain node  */
      ydc_dP[dNode] = - (dIddp_dVddp + dIddp_dVdse) + gds_ext ;
      ydc_dP[dNodePrime] = dIddp_dVddp - dIddp_dVds + gds + dIsub_dVds + dIgidl_dVds - dIgd_dVd ;
      /* ydc_dP[gNode] = 0.0; */
      ydc_dP[gNodePrime] = -dIddp_dVgse - dIddp_dVgs + gm_ext
        + gm + dIsub_dVgs + dIgidl_dVgs - dIgd_dVg ;
      ydc_dP[sNode] =  dIddp_dVdse + dIddp_dVgse + dIddp_dVbse + dIddp_dVsubs + (-gds_ext -gm_ext -gmbs_ext);
      ydc_dP[sNodePrime] = -( gds + dIsub_dVds + dIgidl_dVds )
        - ( gm + dIsub_dVgs + dIgidl_dVgs )
        - ( gmbs + dIsub_dVbs + dIgidl_dVbs ) - dIgd_dVs
        + ( dIddp_dVds + dIddp_dVgs + dIddp_dVbs ) ;
      ydc_dP[bNodePrime] = - dIddp_dVbse - dIddp_dVbs + gmbs_ext
        + gmbs + dIsub_dVbs + dIgidl_dVbs - dIgd_dVb;
      /* ydc_dP[bNode] = 0.0; */
      /* ydc_dP[dbNode] = 0.0 ; */
      /* ydc_dP[sbNode] = 0.0 ; */
      ydc_dP[subNode] = - dIddp_dVsubs ;
      ydc_dP[tempNode] = - dIddp_dT
        + gmT + dIsub_dT + dIgidl_dT - dIgd_dT ;

      /*  gate node  */
      /* ydc_g[dNode] = 0.0 ; */
      /* ydc_g[dNodePrime] = 0.0 ; */
      ydc_g[gNode] = dIggp_dVggp ;
      ydc_g[gNodePrime] = - dIggp_dVggp ;
      /* ydc_g[sNode] = 0.0 ; */
      /* ydc_g[sNodePrime] = 0.0 ; */
      /* ydc_g[bNodePrime] = 0.0 ; */
      /* ydc_g[bNode] = 0.0 ; */
      /* ydc_g[dbNode] = 0.0 ; */
      /* ydc_g[sbNode] = 0.0 ; */
      /* ydc_g[tempNode] = 0.0 ; */

      /*  intrinsic gate node  */
      /* ydc_gP[dNode] = 0.0 ; */
      ydc_gP[dNodePrime] = dIgd_dVd + dIgs_dVd + dIgb_dVd ;
      ydc_gP[gNode] = - dIggp_dVggp ;
      ydc_gP[gNodePrime] = dIggp_dVggp + dIgd_dVg + dIgs_dVg + dIgb_dVg ;
      /* ydc_gP[sNode] = 0.0 ; */
      ydc_gP[sNodePrime] = dIgd_dVs + dIgs_dVs + dIgb_dVs ;
      ydc_gP[bNodePrime] = dIgd_dVb + dIgs_dVb + dIgb_dVb ;
      /* ydc_gP[bNode] = 0.0 ; */
      /* ydc_gP[dbNode] = 0.0 ; */
      /* ydc_gP[sbNode] = 0.0 ; */
      ydc_gP[tempNode] = dIgd_dT + dIgs_dT + dIgb_dT ;

      /*  source node */
      ydc_s[dNode] = dIssp_dVdse + dIsubLDs_dVddp  - dIdsIBPC_dVddp + dIdsIBPCs_dVddp ;
      ydc_s[dNodePrime] = dIsubLDs_dVds  - dIdsIBPC_dVds + dIdsIBPCs_dVds - dIsubLDs_dVddp  + dIdsIBPC_dVddp - dIdsIBPCs_dVddp ;
      /* ydc_s[gNode] = 0.0 */
      ydc_s[gNodePrime] = dIssp_dVgse + dIsubLDs_dVgs  - dIdsIBPC_dVgs + dIdsIBPCs_dVgs ;
      ydc_s[sNode] = dIssp_dVssp - ( dIssp_dVgse + dIssp_dVdse + dIssp_dVbse ) - dIssp_dVsubs + Gbs ;
      ydc_s[sNodePrime] = - dIssp_dVssp - dIsubLDs_dVds - dIsubLDs_dVgs - dIsubLDs_dVbs   - (- dIdsIBPC_dVds  - dIdsIBPC_dVgs  - dIdsIBPC_dVbs )  - dIdsIBPCs_dVds  - dIdsIBPCs_dVgs  - dIdsIBPCs_dVbs ;
      ydc_s[bNodePrime] = dIssp_dVbse + dIsubLDs_dVbs  - dIdsIBPC_dVbs + dIdsIBPCs_dVbs ;
      /* ydc_s[bNode] = 0.0 */
      /* ydc_s[dbNode] = 0.0 */
      ydc_s[sbNode]     = - Gbs ;
      ydc_s[subNode] = dIssp_dVsubs;
      ydc_s[tempNode] = dIssp_dT - Gbst + dIsubLDs_dT  - dIdsIBPC_dT + dIdsIBPCs_dT ;

      /*  intrinsic source node */
      ydc_sP[dNode] = - dIssp_dVdse -gds_ext ;
      ydc_sP[dNodePrime] = - gds + dIsubs_dVds + dIgisl_dVds - dIgs_dVd ;
      /* ydc_sP[gNode] = 0.0 ; */
      ydc_sP[gNodePrime] = -dIssp_dVgse -gm_ext
        - gm + dIsubs_dVgs + dIgisl_dVgs - dIgs_dVg ;
      ydc_sP[sNode] = - dIssp_dVssp - ( - dIssp_dVdse - dIssp_dVgse - dIssp_dVbse ) + dIssp_dVsubs +(gds_ext + gm_ext + gmbs_ext);
      ydc_sP[sNodePrime] = dIssp_dVssp - ( - gds + dIsubs_dVds + dIgisl_dVds )
        - ( - gm + dIsubs_dVgs + dIgisl_dVgs )
        - ( - gmbs + dIsubs_dVbs + dIgisl_dVbs ) - dIgs_dVs ;
      ydc_sP[bNodePrime] = -dIssp_dVbse -gmbs_ext
        - gmbs + dIsubs_dVbs + dIgisl_dVbs - dIgs_dVb ;
      /* ydc_sP[bNode] = 0.0 ; */
      /* ydc_sP[dbNode] = 0.0 ; */
      /* ydc_sP[sbNode] = 0.0 ; */
      ydc_sP[subNode] = - dIssp_dVsubs;
      ydc_sP[tempNode] = -dIssp_dT
              - gmT + dIsubs_dT + dIgisl_dT - dIgs_dT;

      /*  intrinsic bulk node */
      ydc_bP[dNode] = - dIsubLD_dVddp - dIsubLDs_dVddp ;
      ydc_bP[dNodePrime] = - dIsub_dVds - dIsubs_dVds  - dIsubLD_dVds - dIsubLDs_dVds - dIgidl_dVds - dIgb_dVd - dIgisl_dVds - ( - dIsubLD_dVddp - dIsubLDs_dVddp ) ;
      /* ydc_bP[gNode] = 0.0 ; */
      ydc_bP[gNodePrime] = - dIsub_dVgs - dIsubs_dVgs  - dIsubLD_dVgs - dIsubLDs_dVgs - dIgidl_dVgs - dIgb_dVg - dIgisl_dVgs ;
      /* ydc_bP[sNode] = 0.0 ;*/
      ydc_bP[sNodePrime] = - ( - dIsub_dVds - dIsubs_dVds - dIsubLD_dVds - dIsubLDs_dVds - dIgidl_dVds - dIgisl_dVds )
       - ( - dIsub_dVgs - dIsubs_dVgs - dIsubLD_dVgs - dIsubLDs_dVgs - dIgidl_dVgs - dIgisl_dVgs )
       - ( - dIsub_dVbs - dIsubs_dVbs - dIsubLD_dVbs - dIsubLDs_dVbs - dIgidl_dVbs - dIgisl_dVbs ) - dIgb_dVs ;
      ydc_bP[bNodePrime] = - dIsub_dVbs - dIsubs_dVbs  - dIsubLD_dVbs - dIsubLDs_dVbs - dIgidl_dVbs - dIgb_dVb - dIgisl_dVbs + dIbpdb_dVbpdb + dIbpb_dVbpb + dIbpsb_dVbpsb ;
      ydc_bP[bNode] = - dIbpb_dVbpb ;
      ydc_bP[dbNode] = - dIbpdb_dVbpdb ;
      ydc_bP[sbNode] =  - dIbpsb_dVbpsb ;
      ydc_bP[tempNode] = - dIsub_dT - dIsubs_dT - dIsubLD_dT - dIsubLDs_dT - dIgidl_dT - dIgb_dT - dIgisl_dT ;

      /*  bulk node */
      /* ydc_b[dNode] = 0.0 ; */
      /* ydc_b[dNodePrime] = 0.0 ; */
      /* ydc_b[gNode] = 0.0 ; */
      /* ydc_b[gNodePrime] = 0.0 ; */
      /* ydc_b[sNode] = 0.0 ; */
      /* ydc_b[sNodePrime] = 0.0 ; */
      ydc_b[bNodePrime] = - dIbpb_dVbpb ;
      ydc_b[bNode] = dIbpb_dVbpb ;
      /* ydc_b[dbNode] = 0.0 ; */
      /* ydc_b[sbNode] = 0.0 ; */
      /* ydc_b[tempNode] = 0.0 ; */

      /*  drain bulk node */
      ydc_db[dNode] = - Gbd ;
      /* ydc_db[dNodePrime] = 0.0 ; */
      /* ydc_db[gNode] = 0.0 ; */
      /* ydc_db[gNodePrime] = 0.0 ; */
      /* ydc_db[sNode] = 0.0 ; */
      /* ydc_db[sNodePrime] = 0.0 ; */
      ydc_db[bNodePrime] = - dIbpdb_dVbpdb ;
      /* ydc_db[bNode] = 0.0 ; */
      ydc_db[dbNode] = Gbd + dIbpdb_dVbpdb ;
      /* ydc_db[sbNode] = 0.0 ; */
      ydc_db[tempNode] = Gbdt ;

      /* source bulk node */
      /* ydc_sb[dNode] = 0.0 ; */
      /* ydc_sb[dNodePrime] = 0.0 ; */
      /* ydc_sb[gNode] = 0.0 ; */
      /* ydc_sb[gNodePrime] = 0.0 ; */
      ydc_sb[sNode] = - Gbs ;
      /* ydc_sb[sNodePrime] = 0.0 ; */
      ydc_sb[bNodePrime] = - dIbpsb_dVbpsb ;
      /* ydc_sb[bNode] = 0.0 ; */
      /* ydc_sb[dbNode] = 0.0 ; */
      ydc_sb[sbNode] = Gbs + dIbpsb_dVbpsb ;
      ydc_sb[tempNode] = Gbst ;

      /*  temp node */
      ydc_t[dNode] = - dP_dVdse ;
      ydc_t[dNodePrime] = - dP_dVds ;
      /* ydc_t[gNode] = 0.0 ; */
      ydc_t[gNodePrime] = - dP_dVgs - dP_dVgse ;
      ydc_t[sNode] = - ( - dP_dVdse - dP_dVgse - dP_dVbse ) ;
      ydc_t[sNodePrime] = - ( - dP_dVds - dP_dVgs - dP_dVbs ) ;
      ydc_t[bNodePrime] = - dP_dVbs - dP_dVbse ;
      /* ydc_t[bNode] = 0.0 ; */
      /* ydc_t[dbNode] = 0.0 ; */
      /* ydc_t[sbNode] = 0.0 ; */
      ydc_t[tempNode] = dIth_dT - dP_dT ;

      /* additional entries for flat nqs handling */
      if ( flg_nqs ) {
        ydc_qi[dNodePrime]   =   dIqi_nqs_dVds ;
        ydc_qi[gNodePrime]   =   dIqi_nqs_dVgs ;
        ydc_qi[sNodePrime]   = -(dIqi_nqs_dVds + dIqi_nqs_dVgs + dIqi_nqs_dVbs) ;
        ydc_qi[bNodePrime]   =   dIqi_nqs_dVbs ;
        ydc_qi[tempNode] =   dIqi_nqs_dT   ;
        ydc_qi[qiNode]   =   dIqi_nqs_dQi_nqs ;
        /* ydc_qi[qbNode]=   0.0 ;  */
        ydc_qb[dNodePrime]   =   dIqb_nqs_dVds ;
        ydc_qb[gNodePrime]   =   dIqb_nqs_dVgs ;
        ydc_qb[sNodePrime]   = -(dIqb_nqs_dVds + dIqb_nqs_dVgs + dIqb_nqs_dVbs) ;
        ydc_qb[bNodePrime]   =   dIqb_nqs_dVbs ;
        ydc_qb[tempNode] =   dIqb_nqs_dT   ;
        /* ydc_qb[qiNode]=   0.0 ; */
        ydc_qb[qbNode]   =   dIqb_nqs_dQb_nqs ;
      }


      /* Preset vectors and matrix for dynamic part */

      cq_d = cq_dP = cq_g = cq_gP = cq_s = cq_sP = cq_bP = cq_b = cq_db = cq_sb = cq_t = cq_qi = cq_qb = 0.0 ;
      for (i = 0; i < XDIM ; i++) {
        ydyn_d[i] = ydyn_dP[i] = ydyn_g[i] = ydyn_gP[i] = ydyn_s[i] = ydyn_sP[i] = ydyn_bP[i] = ydyn_b[i]
        = ydyn_db[i] = ydyn_sb[i] = ydyn_t[i] = 0.0;
      }
      if (flg_nqs) {
         for (i = 0; i < XDIM ; i++) {
          ydyn_qi[i] = ydyn_qb[i] = 0.0;
        }
      }

      ag0 = ckt->CKTag[0];

      if (ChargeComputationNeeded) { /* start handling of dynamic part */

        if (!ByPass) { /* loading of state vector not necessary in case of Bypass */

          /*  intrinsic gate node (without fringing charges) */
          *(ckt->CKTstate0 + here->HSMHV2qg) = Qg + Qg_nqs + Qgext;

          /*  intrinsic drain node */
          *(ckt->CKTstate0 + here->HSMHV2qd) = Qd + Qd_nqs ;

          /* intrinsic bulk node */
          *(ckt->CKTstate0 + here->HSMHV2qb) = Qb + Qb_nqs + Qbext;

          /*  drain bulk node */
          *(ckt->CKTstate0 + here->HSMHV2qbd) = Qbd ;

          /* source bulk node */
          *(ckt->CKTstate0 + here->HSMHV2qbs) = Qbs ;

          /* temp node */
          *(ckt->CKTstate0 + here->HSMHV2qth) = Qth ;

          /* fringing charges */
          *(ckt->CKTstate0 + here->HSMHV2qfd) = Qfd ;
          *(ckt->CKTstate0 + here->HSMHV2qfs) = Qfs ;

          /* external drain node */
          *(ckt->CKTstate0 + here->HSMHV2qdE) = Qdext;

          /* nqs charges Qi_nqs, Qb_nqs: already loaded above */
          /* if ( flg_nqs ) {                                 */
          /*   *(ckt->CKTstate0 + here->HSMHV2qi_nqs) = Qi_nqs; */
          /*   *(ckt->CKTstate0 + here->HSMHV2qb_nqs) = Qb_nqs; */
          /* }                                                */
        }

        /* ... assemble capacitance matrix */

        /* ...... drain node */
        ydyn_d[dNode] = dQfd_dVdse + Cbd + dQdext_dVdse ;
        /* ydyn_d[dNodePrime] = 0.0 ; */
        /* ydyn_d[gNode] = 0.0 ; */
        ydyn_d[gNodePrime] = dQfd_dVgse + dQdext_dVgse ;
        ydyn_d[sNode] = - (dQfd_dVdse + dQfd_dVgse+ dQfd_dVbse) - ( dQdext_dVdse + dQdext_dVgse + dQdext_dVbse ) ;
        /* ydyn_d[sNodePrime ] = 0.0 ; */
        ydyn_d[bNodePrime] = dQfd_dVbse + dQdext_dVbse;
        /* ydyn_d[bNode ] = 0.0 ; */
        ydyn_d[dbNode] = - Cbd ;
        /* ydyn_d[sbNode ] = 0.0 ; */
        ydyn_d[tempNode] = dQfd_dT - Cbdt + dQdext_dT ;

        /* ...... intrinsic drain node */
        /* ydyn_dP[dNode] = 0.0 ; */
        ydyn_dP[dNodePrime] = dQd_dVds ;
        /* ydyn_dP[gNode] = 0.0 ; */
        ydyn_dP[gNodePrime] = dQd_dVgs ;
        /* ydyn_dP[sNode] = 0.0 ; */
        ydyn_dP[sNodePrime] = - ( dQd_dVds + dQd_dVgs + dQd_dVbs ) ;
        ydyn_dP[bNodePrime] = dQd_dVbs ;
        /* ydyn_dP[bNode] = 0.0 ; */
        /* ydyn_dP[dbNode] = 0.0 ; */
        /* ydyn_dP[sbNode] = 0.0 ; */
        ydyn_dP[tempNode] = dQd_dT ;

        /* ...... gate node  */
        /*        (no entry) */

        /* ...... intrinsic gate node */
        ydyn_gP[dNode] = -dQfd_dVdse - dQfs_dVdse + dQgext_dVdse ;
        ydyn_gP[dNodePrime] = dQg_dVds ;
        /* ydyn_gP[gNode] = 0.0 ; */
        ydyn_gP[gNodePrime] = dQg_dVgs -dQfd_dVgse - dQfs_dVgse + dQgext_dVgse ;
        ydyn_gP[sNode] = dQfd_dVdse + dQfs_dVdse + dQfd_dVgse + dQfs_dVgse + dQfd_dVbse + dQfs_dVbse
                       - ( dQgext_dVdse + dQgext_dVgse + dQgext_dVbse ) ;
        ydyn_gP[sNodePrime] = -( dQg_dVds + dQg_dVgs + dQg_dVbs ) ;
        ydyn_gP[bNodePrime] = dQg_dVbs -dQfd_dVbse - dQfs_dVbse + dQgext_dVbse ;
        /* ydyn_gP[bNode] = 0.0 ; */
        /* ydyn_gP[dbNode] = 0.0 ; */
        /* ydyn_gP[sbNode] = 0.0 ; */
        ydyn_gP[tempNode] = dQg_dT - dQfd_dT - dQfs_dT + dQgext_dT ;

        /* ...... source node */
        ydyn_s[dNode] = dQfs_dVdse + dQsext_dVdse ;
        /* ydyn_d[dNodePrime ] = 0.0 ; */
        /* ydyn_d[gNode ] = 0.0 ; */
        ydyn_s[gNodePrime] = dQfs_dVgse + dQsext_dVgse ;
        ydyn_s[sNode] = Cbs - (dQfs_dVdse + dQfs_dVgse+ dQfs_dVbse) - ( dQsext_dVdse + dQsext_dVgse + dQsext_dVbse ) ;
        /* ydyn_d[sNodePrime ] = 0.0 ; */
        ydyn_s[bNodePrime] = dQfs_dVbse + dQsext_dVbse ;
        /* ydyn_d[bNode ] = 0.0 ; */
        /* ydyn_d[dbNode ] = 0.0 ; */
        ydyn_s[sbNode] = - Cbs ;
        ydyn_s[tempNode] = dQfs_dT - Cbst + dQsext_dT ;

        /* ...... intrinsic source node */
        /* ydyn_sP[dNode] = 0.0 ; */
        ydyn_sP[dNodePrime] = dQs_dVds ;
        /* ydyn_sP[gNode] = 0.0 ; */
        ydyn_sP[gNodePrime] =  dQs_dVgs ;
        /* ydyn_sP[sNode] = 0.0 ; */
        ydyn_sP[sNodePrime] = - ( dQs_dVds + dQs_dVgs + dQs_dVbs );
        ydyn_sP[bNodePrime] = dQs_dVbs ;
        /* ydyn_sP[bNode] = 0.0 ; */
        /* ydyn_sP[dbNode] = 0.0 ; */
        /* ydyn_sP[sbNode] = 0.0 ; */
        ydyn_sP[tempNode] = dQs_dT ;

        /* ...... intrinsic bulk node */
        ydyn_bP[dNode] = dQbext_dVdse ;
        ydyn_bP[dNodePrime] =  dQb_dVds ;
        /* ydyn_bP[gNode] = 0.0 ; */
        ydyn_bP[gNodePrime] = dQb_dVgs + dQbext_dVgse ;
        ydyn_bP[sNode] = - ( dQbext_dVdse + dQbext_dVgse + dQbext_dVbse ) ;
        ydyn_bP[sNodePrime] = - ( dQb_dVds + dQb_dVgs + dQb_dVbs );
        ydyn_bP[bNodePrime] = dQb_dVbs + dQbext_dVbse ;
        /* ydyn_bP[bNode] = 0.0 ; */
        /* ydyn_bP[dbNode] = 0.0 ; */
        /* ydyn_bP[sbNode] = 0.0 ; */
        ydyn_bP[tempNode] = dQb_dT + dQbext_dT ;

        /* ...... bulk node  */
        /*        (no entry) */

        /* ...... drain bulk node */
        ydyn_db[dNode] = - Cbd ;
        /* ydyn_db[dNodePrime] = 0.0 ; */
        /* ydyn_db[gNode] = 0.0 ; */
        /* ydyn_db[gNodePrime] = 0.0 ; */
        /* ydyn_db[sNode] = 0.0 ; */
        /* ydyn_db[sNodePrime] = 0.0 ; */
        /* ydyn_db[bNodePrime] = 0.0 ; */
        /* ydyn_db[bNode] = 0.0 ; */
        ydyn_db[dbNode] = Cbd ;
        /* ydyn_db[sbNode] = 0.0 ; */
        ydyn_db[tempNode] = Cbdt ;

        /* ...... source bulk  node */
        /* ydyn_sb[dNode] = 0.0 ; */
        /* ydyn_sb[dNodePrime] = 0.0 ; */
        /* ydyn_sb[gNode] = 0.0 ; */
        /* ydyn_sb[gNodePrime] = 0.0 ; */
        ydyn_sb[sNode] = - Cbs ;
        /* ydyn_sb[sNodePrime] = 0.0 ; */
        /* ydyn_sb[bNodePrime] = 0.0 ; */
        /* ydyn_sb[bNode] = 0.0 ; */
        /* ydyn_sb[dbNode] = 0.0 ; */
        ydyn_sb[sbNode] = Cbs ;
        ydyn_sb[tempNode] = Cbst ;

        /* ...... temp  node */
        /* ydyn_t[dNode] = 0.0 ; */
        /* ydyn_t[dNodePrime] = 0.0 ; */
        /* ydyn_t[gNode] = 0.0 ; */
        /* ydyn_t[gNodePrime] = 0.0 ; */
        /* ydyn_t[sNode] = 0.0 ; */
        /* ydyn_t[sNodePrime] = 0.0 ; */
        /* ydyn_t[bNodePrime] = 0.0 ; */
        /* ydyn_t[bNode] = 0.0 ; */
        /* ydyn_t[dbNode] = 0.0 ; */
        /* ydyn_t[sbNode] = 0.0 ; */
        ydyn_t[tempNode] = Cth ;

        /* additional entries for flat nqs handling */
        if (flg_nqs) {
          /* ...... intrinsic drain node */
          /* ydyn_dP[dNode] += 0.0 ; */
          ydyn_dP[dNodePrime] += dQd_nqs_dVds ;
          /* ydyn_dP[gNode] += 0.0 ; */
          ydyn_dP[gNodePrime] += dQd_nqs_dVgs ;
          /* ydyn_dP[sNode] += 0.0 ; */
          ydyn_dP[sNodePrime] += - ( dQd_nqs_dVds + dQd_nqs_dVgs + dQd_nqs_dVbs ) ;
          ydyn_dP[bNodePrime] += dQd_nqs_dVbs ;
          /* ydyn_dP[bNode] += 0.0 ; */
          /* ydyn_dP[dbNode] += 0.0 ; */
          /* ydyn_dP[sbNode] += 0.0 ; */
          ydyn_dP[tempNode] += dQd_nqs_dT ;
          ydyn_dP[qiNode]    = dQd_nqs_dQi_nqs ;

          /* ...... intrinsic gate node */
          /* ydyn_gP[dNode] += 0.0 ; */
          /* ydyn_gP[dNodePrime] += 0.0 ; */
          /* ydyn_gP[gNode] += 0.0 ; */
          /* ydyn_gP[gNodePrime] += 0.0 ; */
          /* ydyn_gP[sNode] += 0.0 ; */
          /* ydyn_gP[sNodePrime] += 0.0 ; */
          /* ydyn_gP[bNodePrime] += 0.0 ; */
          /* ydyn_gP[bNode] += 0.0 ; */
          /* ydyn_gP[dbNode] += 0.0 ; */
          /* ydyn_gP[sbNode] += 0.0 ; */
          /* ydyn_gP[tempNode] += 0.0 ; */
          ydyn_gP[qiNode] = dQg_nqs_dQi_nqs ;
          ydyn_gP[qbNode] = dQg_nqs_dQb_nqs ;

          /* ...... intrinsic source node */
          /* ydyn_sP[dNode] += 0.0 ; */
          ydyn_sP[dNodePrime] += dQs_nqs_dVds ;
          /* ydyn_sP[gNode] += 0.0 ; */
          ydyn_sP[gNodePrime] +=  dQs_nqs_dVgs ;
          /* ydyn_sP[sNode] += 0.0 ; */
          ydyn_sP[sNodePrime] += - ( dQs_nqs_dVds + dQs_nqs_dVgs + dQs_nqs_dVbs );
          ydyn_sP[bNodePrime] += dQs_nqs_dVbs ;
          /* ydyn_sP[bNode] += 0.0 ; */
          /* ydyn_sP[dbNode] += 0.0 ; */
          /* ydyn_sP[sbNode] += 0.0 ; */
          ydyn_sP[tempNode] += dQs_nqs_dT ;
          ydyn_sP[qiNode]    = dQs_nqs_dQi_nqs ;

          /* ...... intrinsic bulk node */
          /* ydyn_bP[dNode] += 0.0 ; */
          /* ydyn_bP[dNodePrime] += 0.0 ; */
          /* ydyn_bP[gNode] += 0.0 ; */
          /* ydyn_bP[gNodePrime] += 0.0 ; */
          /* ydyn_bP[sNode] += 0.0 ; */
          /* ydyn_bP[sNodePrime] += 0.0 ; */
          /* ydyn_bP[bNodePrime] += 0.0 ; */
          /* ydyn_bP[bNode] += 0.0 ; */
          /* ydyn_bP[dbNode] += 0.0 ; */
          /* ydyn_bP[sbNode] += 0.0 ; */
          /* ydyn_bP[tempNode] += 0.0 ; */
          ydyn_bP[qbNode] = 1.0 ;

          /* ...... qi node */
          /* ydyn_qi[dNodePrime]   = 0.0 ; */
          /* ydyn_qi[gNodePrime]   = 0.0 ; */
          /* ydyn_qi[sNodePrime]   = 0.0 ; */
          /* ydyn_qi[bNodePrime]   = 0.0 ; */
          /* ydyn_qi[tempNode] = 0.0 ; */
          ydyn_qi[qiNode]      = 1.0 ;
          /* ydyn_qi[qbNode]   = 0.0 ; */

          /* ...... qb node */
          /* ydyn_qb[dNodePrime]   = 0.0 ; */
          /* ydyn_qb[gNodePrime]   = 0.0 ; */
          /* ydyn_qb[sNodePrime]   = 0.0 ; */
          /* ydyn_qb[bNodePrime]   = 0.0 ; */
          /* ydyn_qb[tempNode] = 0.0 ; */
          /* ydyn_qb[qiNode]   = 0.0 ; */
          ydyn_qb[qbNode]      = 1.0 ;
        }

        if (!ByPass) { /* integrate etc. only necessary if not in Bypass mode! */

          /* store small signal parameters */
          if (ckt->CKTmode & MODEINITSMSIG) {
            /* printf("HSMHV2_load: (small signal) ByPass=%d\n",ByPass); */
/*          printf("HSMHV2_load: ydc_dP=%e %e %e %e %e %e %e %e\n",
                    ydc_dP[0],ydc_dP[1],ydc_dP[2],ydc_dP[3],ydc_dP[4],ydc_dP[5],ydc_dP[6],ydc_dP[7]);
            printf("HSMHV2_load: ych_dP=%e %e %e %e %e %e %e %e\n",
                    ydyn_dP[0],ydyn_dP[1],ydyn_dP[2],ydyn_dP[3],ydyn_dP[4],ydyn_dP[5],ydyn_dP[6],ydyn_dP[7]);
*/
            /* dc matrix into structure 0724*/
            for (i = 0; i < XDIM; i++) {
              here->HSMHV2_ydc_d[i] = ydc_d[i];
              here->HSMHV2_ydc_dP[i] = ydc_dP[i];
              here->HSMHV2_ydc_g[i] = ydc_g[i];
              here->HSMHV2_ydc_gP[i] = ydc_gP[i];
              here->HSMHV2_ydc_s[i] = ydc_s[i];
              here->HSMHV2_ydc_sP[i] = ydc_sP[i];
              here->HSMHV2_ydc_bP[i] = ydc_bP[i];
              here->HSMHV2_ydc_b[i] = ydc_b[i];
              here->HSMHV2_ydc_db[i] = ydc_db[i];
              here->HSMHV2_ydc_sb[i] = ydc_sb[i];
              here->HSMHV2_ydc_t[i] = ydc_t[i];
            }
            /* capacitance matrix into structure 0724*/
            for (i = 0; i < XDIM; i++) {
              here->HSMHV2_ydyn_d[i] = ydyn_d[i];
              here->HSMHV2_ydyn_dP[i] = ydyn_dP[i];
              here->HSMHV2_ydyn_g[i] = ydyn_g[i];
              here->HSMHV2_ydyn_gP[i] = ydyn_gP[i];
              here->HSMHV2_ydyn_s[i] = ydyn_s[i];
              here->HSMHV2_ydyn_sP[i] = ydyn_sP[i];
              here->HSMHV2_ydyn_bP[i] = ydyn_bP[i];
              here->HSMHV2_ydyn_b[i] = ydyn_b[i];
              here->HSMHV2_ydyn_db[i] = ydyn_db[i];
              here->HSMHV2_ydyn_sb[i] = ydyn_sb[i];
              here->HSMHV2_ydyn_t[i] = ydyn_t[i];
            }
            if (flg_nqs) {
              for (i = 0; i < XDIM; i++) {
                here->HSMHV2_ydc_qi[i] = ydc_qi[i];
                here->HSMHV2_ydc_qb[i] = ydc_qb[i];
                here->HSMHV2_ydyn_qi[i] = ydyn_qi[i];
                here->HSMHV2_ydyn_qb[i] = ydyn_qb[i];
              }
            }
            goto line1000; /* that's all for small signal analyses */
          }

          /* Continue handling of dynamic part: */
          /* ... calculate time derivatives of node charges */

          if (ckt->CKTmode & MODEINITTRAN) {
            /* at the very first iteration of the first timepoint:
           copy charges into previous state -> the integrator may use them ... */
            *(ckt->CKTstate1 + here->HSMHV2qb) = *(ckt->CKTstate0 + here->HSMHV2qb);
            *(ckt->CKTstate1 + here->HSMHV2qg) = *(ckt->CKTstate0 + here->HSMHV2qg);
            *(ckt->CKTstate1 + here->HSMHV2qd) = *(ckt->CKTstate0 + here->HSMHV2qd);
            *(ckt->CKTstate1 + here->HSMHV2qth) = *(ckt->CKTstate0 + here->HSMHV2qth);
            *(ckt->CKTstate1 + here->HSMHV2qbs) = *(ckt->CKTstate0 + here->HSMHV2qbs);
            *(ckt->CKTstate1 + here->HSMHV2qbd) = *(ckt->CKTstate0 + here->HSMHV2qbd);

            *(ckt->CKTstate1 + here->HSMHV2qfd) = *(ckt->CKTstate0 + here->HSMHV2qfd);
            *(ckt->CKTstate1 + here->HSMHV2qfs) = *(ckt->CKTstate0 + here->HSMHV2qfs);

            *(ckt->CKTstate1 + here->HSMHV2qdE) = *(ckt->CKTstate0 + here->HSMHV2qdE);

            if (flg_nqs) {
              *(ckt->CKTstate1 + here->HSMHV2qi_nqs) = *(ckt->CKTstate0 + here->HSMHV2qi_nqs);
              *(ckt->CKTstate1 + here->HSMHV2qb_nqs) = *(ckt->CKTstate0 + here->HSMHV2qb_nqs);
            }
          }

          return_if_error (NIintegrate(ckt, &geq, &ceq, 0.0, here->HSMHV2qb));
          return_if_error (NIintegrate(ckt, &geq, &ceq, 0.0, here->HSMHV2qg));
          return_if_error (NIintegrate(ckt, &geq, &ceq, 0.0, here->HSMHV2qd));
          return_if_error (NIintegrate(ckt, &geq, &ceq, 0.0, here->HSMHV2qbs));
          return_if_error (NIintegrate(ckt, &geq, &ceq, 0.0, here->HSMHV2qbd));

          return_if_error (NIintegrate(ckt, &geq, &ceq, 0.0, here->HSMHV2qth));

          return_if_error (NIintegrate(ckt, &geq, &ceq, 0.0, here->HSMHV2qfd));
          return_if_error (NIintegrate(ckt, &geq, &ceq, 0.0, here->HSMHV2qfs));

          return_if_error (NIintegrate(ckt, &geq, &ceq, 0.0, here->HSMHV2qdE));

          if (flg_nqs) {
            return_if_error (NIintegrate(ckt, &geq, &ceq, 0.0, here->HSMHV2qi_nqs));
            return_if_error (NIintegrate(ckt, &geq, &ceq, 0.0, here->HSMHV2qb_nqs));
          }

          if (ckt->CKTmode & MODEINITTRAN) {
            /* at the very first iteration of the first timepoint:
               copy currents into previous state -> the integrator may use them ... */
            *(ckt->CKTstate1 + here->HSMHV2cqb) = *(ckt->CKTstate0 + here->HSMHV2cqb);
            *(ckt->CKTstate1 + here->HSMHV2cqg) = *(ckt->CKTstate0 + here->HSMHV2cqg);
            *(ckt->CKTstate1 + here->HSMHV2cqd) = *(ckt->CKTstate0 + here->HSMHV2cqd);

            *(ckt->CKTstate1 + here->HSMHV2cqth) = *(ckt->CKTstate0 + here->HSMHV2cqth);

            *(ckt->CKTstate1 + here->HSMHV2cqbs) = *(ckt->CKTstate0 + here->HSMHV2cqbs);
            *(ckt->CKTstate1 + here->HSMHV2cqbd) = *(ckt->CKTstate0 + here->HSMHV2cqbd);
            *(ckt->CKTstate1 + here->HSMHV2cqfd) = *(ckt->CKTstate0 + here->HSMHV2cqfd);
            *(ckt->CKTstate1 + here->HSMHV2cqfs) = *(ckt->CKTstate0 + here->HSMHV2cqfs);

            *(ckt->CKTstate1 + here->HSMHV2cqdE) = *(ckt->CKTstate0 + here->HSMHV2cqdE);
            if (flg_nqs) {
              *(ckt->CKTstate1 + here->HSMHV2dotqi_nqs) = *(ckt->CKTstate0 + here->HSMHV2dotqi_nqs);
              *(ckt->CKTstate1 + here->HSMHV2dotqb_nqs) = *(ckt->CKTstate0 + here->HSMHV2dotqb_nqs);
            }
          }
        }


        /* ... finally gather displacement currents from data structures */

        cq_dP = *(ckt->CKTstate0 + here->HSMHV2cqd);
        cq_gP = *(ckt->CKTstate0 + here->HSMHV2cqg);
        cq_bP = *(ckt->CKTstate0 + here->HSMHV2cqb);
        cq_sP = - *(ckt->CKTstate0 + here->HSMHV2cqg)
                - *(ckt->CKTstate0 + here->HSMHV2cqb)
                - *(ckt->CKTstate0 + here->HSMHV2cqd);
        cq_dE = *(ckt->CKTstate0 + here->HSMHV2cqdE);
        cq_db = *(ckt->CKTstate0 + here->HSMHV2cqbd);
        cq_sb = *(ckt->CKTstate0 + here->HSMHV2cqbs);
        cq_g = 0.0 ;
        cq_b = 0.0 ;

        /* displacement currents at outer drain/source node (fringing part only!) */
        cq_d = *(ckt->CKTstate0 + here->HSMHV2cqfd);
        cq_s = *(ckt->CKTstate0 + here->HSMHV2cqfs);

        cq_t = *(ckt->CKTstate0 + here->HSMHV2cqth);

        /* displacement currents due to nqs */
        if (flg_nqs) {
          cq_qi = *(ckt->CKTstate0 + here->HSMHV2dotqi_nqs);
          cq_qb = *(ckt->CKTstate0 + here->HSMHV2dotqb_nqs);
        } else {
          cq_qi = cq_qb = 0.0 ;
        }

        /* ... and, if necessary: Extrapolate displacement currents onto x-values */

        if (delvdbd || deldeltemp) {
          cq_db += ag0*(Cbd*delvdbd + Cbdt*deldeltemp) ;
        }
        if (delvsbs || deldeltemp) {
          cq_sb += ag0*(Cbs*delvsbs + Cbst*deldeltemp) ;
        }
        if (delvgs || delvds || delvbs || deldeltemp) {
          cq_gP += ag0*(dQg_dVgs*delvgs + dQg_dVds*delvds + dQg_dVbs*delvbs + dQg_dT*deldeltemp) ;
          cq_dP += ag0*(dQd_dVgs*delvgs + dQd_dVds*delvds + dQd_dVbs*delvbs + dQd_dT*deldeltemp) ;
          cq_sP += ag0*(dQs_dVgs*delvgs + dQs_dVds*delvds + dQs_dVbs*delvbs + dQs_dT*deldeltemp) ;
          cq_bP  = - ( cq_gP + cq_dP + cq_sP );
        }
        if (deldeltemp) {
          cq_t += ag0*Cth    *deldeltemp ;
          cq_d += ag0*dQfd_dT*deldeltemp ;
          cq_s += ag0*dQfs_dT*deldeltemp ;

          cq_gE += ag0*dQgext_dT*deldeltemp ;
          cq_dE += ag0*dQdext_dT*deldeltemp ;
          cq_bE += ag0*dQbext_dT*deldeltemp ;
          cq_sE  = - ( cq_gE + cq_dE + cq_bE );
        }
        if (delvgse || delvdse || delvbse) {
          cq_d += ag0*(dQfd_dVgse*delvgse + dQfd_dVdse*delvdse + dQfd_dVbse*delvbse) ;
          cq_s += ag0*(dQfs_dVgse*delvgse + dQfs_dVdse*delvdse + dQfs_dVbse*delvbse) ;

          cq_gE += ag0*(dQgext_dVgse*delvgse + dQgext_dVdse*delvdse + dQgext_dVbse*delvbse) ;
          cq_dE += ag0*(dQdext_dVgse*delvgse + dQdext_dVdse*delvdse + dQdext_dVbse*delvbse) ;
          cq_bE += ag0*(dQbext_dVgse*delvgse + dQbext_dVdse*delvdse + dQbext_dVbse*delvbse) ;
          cq_sE  = - ( cq_gE + cq_dE + cq_bE );
        }

        if (flg_nqs) {
          if (delvgs || delvds || delvbs || deldeltemp) {
            cq_dP += ag0*(dQd_nqs_dVgs*delvgs + dQd_nqs_dVds*delvds + dQd_nqs_dVbs*delvbs + dQd_nqs_dT*deldeltemp) ;
            cq_sP += ag0*(dQs_nqs_dVgs*delvgs + dQs_nqs_dVds*delvds + dQs_nqs_dVbs*delvbs + dQs_nqs_dT*deldeltemp) ;
            cq_bP  = - ( cq_gP + cq_dP + cq_sP ); /* should be superfluous ? */
          }
          if (delQi_nqs) {
            cq_dP += ag0*dQd_nqs_dQi_nqs*delQi_nqs ;
            cq_gP += ag0*dQg_nqs_dQi_nqs*delQi_nqs ;
            cq_sP += ag0*dQs_nqs_dQi_nqs*delQi_nqs ;
            cq_qi += ag0*     1.0       *delQi_nqs ;
          }
          if (delQb_nqs) {
            cq_gP += ag0*dQg_nqs_dQb_nqs*delQb_nqs ;
            cq_bP += ag0*     1.0       *delQb_nqs ;
            cq_qb += ag0*     1.0       *delQb_nqs ;
          }
        }
      } /* End of handling dynamic part */


      /* Assemble total node currents (type-handling shifted to stamping) */

      cur_d  = i_d  + cq_d - cq_db + cq_dE ;
      cur_dP = i_dP + cq_dP ;
      cur_g  = i_g  + cq_g  ;
      cur_gP = i_gP + cq_gP - cq_d - cq_s + cq_gE ;
      cur_s  = i_s  + cq_s - cq_sb + cq_sE ;
      cur_sP = i_sP + cq_sP;
      cur_bP = i_bP + cq_bP + cq_bE ;
      cur_b  = i_b  + cq_b;
      cur_db = i_db + cq_db;
      cur_sb = i_sb + cq_sb;
      cur_t  = i_t  + cq_t;
      cur_qi = i_qi + cq_qi;
      cur_qb = i_qb + cq_qb;


      /* Now we can start stamping ...                                    */
      /* ... right hand side: subtract total node currents                */

      *(ckt->CKTrhs + here->HSMHV2dNode)      -= model->HSMHV2_type * cur_d;
      *(ckt->CKTrhs + here->HSMHV2dNodePrime) -= model->HSMHV2_type * cur_dP;
      *(ckt->CKTrhs + here->HSMHV2gNode)      -= model->HSMHV2_type * cur_g;
      *(ckt->CKTrhs + here->HSMHV2gNodePrime) -= model->HSMHV2_type * cur_gP;
      *(ckt->CKTrhs + here->HSMHV2sNode)      -= model->HSMHV2_type * cur_s;
      *(ckt->CKTrhs + here->HSMHV2sNodePrime) -= model->HSMHV2_type * cur_sP;
      *(ckt->CKTrhs + here->HSMHV2bNodePrime) -= model->HSMHV2_type * cur_bP;
      *(ckt->CKTrhs + here->HSMHV2bNode)      -= model->HSMHV2_type * cur_b;
      *(ckt->CKTrhs + here->HSMHV2dbNode)     -= model->HSMHV2_type * cur_db;
      *(ckt->CKTrhs + here->HSMHV2sbNode)     -= model->HSMHV2_type * cur_sb;
      if( here->HSMHV2_coselfheat > 0) {
        *(ckt->CKTrhs + here->HSMHV2tempNode) -= cur_t;  /* temp node independent of model type! */
      }
      if (flg_nqs) {
        *(ckt->CKTrhs + here->HSMHV2qiNode) -= cur_qi;
        *(ckt->CKTrhs + here->HSMHV2qbNode) -= cur_qb;
      }


      /* ... finally stamp matrix */

      /*drain*/
      *(here->HSMHV2DdPtr)  += ydc_d[dNode]         + ag0*ydyn_d[dNode];
      *(here->HSMHV2DdpPtr) += ydc_d[dNodePrime]    + ag0*ydyn_d[dNodePrime];
      *(here->HSMHV2DgpPtr) += ydc_d[gNodePrime]    + ag0*ydyn_d[gNodePrime];
      *(here->HSMHV2DsPtr)  += ydc_d[sNode]         + ag0*ydyn_d[sNode];
      *(here->HSMHV2DspPtr) += ydc_d[sNodePrime]    + ag0*ydyn_d[sNodePrime];
      *(here->HSMHV2DbpPtr) += ydc_d[bNodePrime]    + ag0*ydyn_d[bNodePrime];
      *(here->HSMHV2DdbPtr) += ydc_d[dbNode]        + ag0*ydyn_d[dbNode];
      if (flg_subNode > 0) {
        *(here->HSMHV2DsubPtr) += ydc_d[subNode];
      }
      if( here->HSMHV2_coselfheat > 0) {
        /* temp entries in matrix dependent on model type */
        *(here->HSMHV2DtempPtr) += model->HSMHV2_type * (ydc_d[tempNode] + ag0*ydyn_d[tempNode]);
      }

      /*drain prime*/
      *(here->HSMHV2DPdPtr)  +=  ydc_dP[dNode]      + ag0*ydyn_dP[dNode];
      *(here->HSMHV2DPdpPtr) +=  ydc_dP[dNodePrime] + ag0*ydyn_dP[dNodePrime];
      *(here->HSMHV2DPgpPtr) +=  ydc_dP[gNodePrime] + ag0*ydyn_dP[gNodePrime];
      *(here->HSMHV2DPsPtr)  +=  ydc_dP[sNode]      + ag0*ydyn_dP[sNode];
      *(here->HSMHV2DPspPtr) +=  ydc_dP[sNodePrime] + ag0*ydyn_dP[sNodePrime];
      *(here->HSMHV2DPbpPtr) +=  ydc_dP[bNodePrime] + ag0*ydyn_dP[bNodePrime];
      if (flg_subNode > 0) {
        *(here->HSMHV2DPsubPtr) += ydc_dP[subNode];
      }
      if( here->HSMHV2_coselfheat > 0) {
        /* temp entries in matrix dependent on model type */
        *(here->HSMHV2DPtempPtr) +=  model->HSMHV2_type * (ydc_dP[tempNode] + ag0*ydyn_dP[tempNode]);
      }
      if (flg_nqs) {
        *(here->HSMHV2DPqiPtr) +=  model->HSMHV2_type * (ydc_dP[qiNode] + ag0*ydyn_dP[qiNode]);
      }

      /*gate*/
      *(here->HSMHV2GgPtr)   +=  ydc_g[gNode]       + ag0*ydyn_g[gNode];
      *(here->HSMHV2GgpPtr)  +=  ydc_g[gNodePrime]  + ag0*ydyn_g[gNodePrime];

      /*gate prime*/
      *(here->HSMHV2GPdPtr)  +=  ydc_gP[dNode]      + ag0*ydyn_gP[dNode];
      *(here->HSMHV2GPdpPtr) +=  ydc_gP[dNodePrime] + ag0*ydyn_gP[dNodePrime];
      *(here->HSMHV2GPgPtr)  +=  ydc_gP[gNode]      + ag0*ydyn_gP[gNode];
      *(here->HSMHV2GPgpPtr) +=  ydc_gP[gNodePrime] + ag0*ydyn_gP[gNodePrime];
      *(here->HSMHV2GPsPtr)  +=  ydc_gP[sNode]      + ag0*ydyn_gP[sNode];
      *(here->HSMHV2GPspPtr) +=  ydc_gP[sNodePrime] + ag0*ydyn_gP[sNodePrime];
      *(here->HSMHV2GPbpPtr) +=  ydc_gP[bNodePrime] + ag0*ydyn_gP[bNodePrime];
      if( here->HSMHV2_coselfheat > 0) {
        /* temp entries in matrix dependent on model type */
        *(here->HSMHV2GPtempPtr) +=  model->HSMHV2_type * (ydc_gP[tempNode] + ag0*ydyn_gP[tempNode]);
      }
      if (flg_nqs) {
        *(here->HSMHV2GPqiPtr) +=  model->HSMHV2_type * (ydc_gP[qiNode] + ag0*ydyn_gP[qiNode]);
        *(here->HSMHV2GPqbPtr) +=  model->HSMHV2_type * (ydc_gP[qbNode] + ag0*ydyn_gP[qbNode]);
      }

      /*source*/
      *(here->HSMHV2SdPtr)  += ydc_s[dNode]         + ag0*ydyn_s[dNode];
      *(here->HSMHV2SsPtr)  += ydc_s[sNode]         + ag0*ydyn_s[sNode];
      *(here->HSMHV2SdpPtr) += ydc_s[dNodePrime]    + ag0*ydyn_s[dNodePrime];
      *(here->HSMHV2SgpPtr) += ydc_s[gNodePrime]    + ag0*ydyn_s[gNodePrime];
      *(here->HSMHV2SspPtr) += ydc_s[sNodePrime]    + ag0*ydyn_s[sNodePrime];
      *(here->HSMHV2SbpPtr) += ydc_s[bNodePrime]    + ag0*ydyn_s[bNodePrime];
      *(here->HSMHV2SsbPtr) += ydc_s[sbNode]        + ag0*ydyn_s[sbNode];
      if (flg_subNode > 0) {
        *(here->HSMHV2SsubPtr) += ydc_s[subNode];
      }
      if( here->HSMHV2_coselfheat > 0) {
        /* temp entries in matrix dependent on model type */
        *(here->HSMHV2StempPtr) += model->HSMHV2_type * (ydc_s[tempNode]+ ag0*ydyn_s[tempNode]);
      }

      /*source prime*/
      *(here->HSMHV2SPdPtr)  +=  ydc_sP[dNode]      + ag0*ydyn_sP[dNode];
      *(here->HSMHV2SPdpPtr) +=  ydc_sP[dNodePrime] + ag0*ydyn_sP[dNodePrime];
      *(here->HSMHV2SPgpPtr) +=  ydc_sP[gNodePrime] + ag0*ydyn_sP[gNodePrime];
      *(here->HSMHV2SPsPtr)  +=  ydc_sP[sNode]      + ag0*ydyn_sP[sNode];
      *(here->HSMHV2SPspPtr) +=  ydc_sP[sNodePrime] + ag0*ydyn_sP[sNodePrime];
      *(here->HSMHV2SPbpPtr) +=  ydc_sP[bNodePrime] + ag0*ydyn_sP[bNodePrime];
      if (flg_subNode > 0) {
        *(here->HSMHV2SPsubPtr) += ydc_sP[subNode];
      }
      if( here->HSMHV2_coselfheat > 0) {
        /* temp entries in matrix dependent on model type */
        *(here->HSMHV2SPtempPtr) +=  model->HSMHV2_type * (ydc_sP[tempNode] + ag0*ydyn_sP[tempNode]);
      }
      if (flg_nqs) {
        *(here->HSMHV2SPqiPtr) +=  model->HSMHV2_type * (ydc_sP[qiNode] + ag0*ydyn_sP[qiNode]);
      }

      /*bulk prime*/
      *(here->HSMHV2BPdPtr)  +=  ydc_bP[dNode]      + ag0*ydyn_bP[dNode];
      *(here->HSMHV2BPdpPtr) +=  ydc_bP[dNodePrime] + ag0*ydyn_bP[dNodePrime];
      *(here->HSMHV2BPgpPtr) +=  ydc_bP[gNodePrime] + ag0*ydyn_bP[gNodePrime];
      *(here->HSMHV2BPspPtr) +=  ydc_bP[sNodePrime] + ag0*ydyn_bP[sNodePrime];
      *(here->HSMHV2BPsPtr)  +=  ydc_bP[sNode]      + ag0*ydyn_bP[sNode];
      *(here->HSMHV2BPbpPtr) +=  ydc_bP[bNodePrime] + ag0*ydyn_bP[bNodePrime];
      *(here->HSMHV2BPbPtr)  +=  ydc_bP[bNode]      + ag0*ydyn_bP[bNode];
      *(here->HSMHV2BPdbPtr) +=  ydc_bP[dbNode]     + ag0*ydyn_bP[dbNode];
      *(here->HSMHV2BPsbPtr) +=  ydc_bP[sbNode]     + ag0*ydyn_bP[sbNode];
      if( here->HSMHV2_coselfheat > 0) {
        /* temp entries in matrix dependent on model type */
        *(here->HSMHV2BPtempPtr) +=  model->HSMHV2_type * (ydc_bP[tempNode] + ag0*ydyn_bP[tempNode]);
      }
      if (flg_nqs) {
        *(here->HSMHV2BPqbPtr) +=  model->HSMHV2_type * (ydc_bP[qbNode] + ag0*ydyn_bP[qbNode]);
      }

      /*bulk*/
      *(here->HSMHV2BbpPtr) +=  ydc_b[bNodePrime]   + ag0*ydyn_b[bNodePrime];
      *(here->HSMHV2BbPtr)  +=  ydc_b[bNode]        + ag0*ydyn_b[bNode];

      /*drain bulk*/
      *(here->HSMHV2DBdPtr)  +=  ydc_db[dNode]      + ag0*ydyn_db[dNode];
      *(here->HSMHV2DBbpPtr) +=  ydc_db[bNodePrime] + ag0*ydyn_db[bNodePrime];
      *(here->HSMHV2DBdbPtr) +=  ydc_db[dbNode]     + ag0*ydyn_db[dbNode];
      if( here->HSMHV2_coselfheat > 0) {
        /* temp entries in matrix dependent on model type */
        *(here->HSMHV2DBtempPtr) +=  model->HSMHV2_type * (ydc_db[tempNode] + ag0*ydyn_db[tempNode]);
      }

      /*source bulk*/
      *(here->HSMHV2SBsPtr)  +=  ydc_sb[sNode]      + ag0*ydyn_sb[sNode];
      *(here->HSMHV2SBbpPtr) +=  ydc_sb[bNodePrime] + ag0*ydyn_sb[bNodePrime];
      *(here->HSMHV2SBsbPtr) +=  ydc_sb[sbNode]     + ag0*ydyn_sb[sbNode];
      if( here->HSMHV2_coselfheat > 0) {
        /* temp entries in matrix dependent on model type */
        *(here->HSMHV2SBtempPtr) +=  model->HSMHV2_type * (ydc_sb[tempNode] + ag0*ydyn_sb[tempNode]);
      }

      /*temp*/
      if( here->HSMHV2_coselfheat > 0) {
        /* temp entries in matrix dependent on model type */
        *(here->HSMHV2TempdPtr)  +=  model->HSMHV2_type * (ydc_t[dNode]      + ag0*ydyn_t[dNode]     );
        *(here->HSMHV2TempdpPtr) +=  model->HSMHV2_type * (ydc_t[dNodePrime] + ag0*ydyn_t[dNodePrime]);
        *(here->HSMHV2TempgpPtr) +=  model->HSMHV2_type * (ydc_t[gNodePrime] + ag0*ydyn_t[gNodePrime]);
        *(here->HSMHV2TempsPtr)  +=  model->HSMHV2_type * (ydc_t[sNode]      + ag0*ydyn_t[sNode]     );
        *(here->HSMHV2TempspPtr) +=  model->HSMHV2_type * (ydc_t[sNodePrime] + ag0*ydyn_t[sNodePrime]);
        *(here->HSMHV2TempbpPtr) +=  model->HSMHV2_type * (ydc_t[bNodePrime] + ag0*ydyn_t[bNodePrime]);
        /* no type factor at main diagonal temp entry! */
        *(here->HSMHV2TemptempPtr) +=  ydc_t[tempNode] + ag0*ydyn_t[tempNode];
      }

      /* additional entries for flat nqs handling */
      if ( flg_nqs ) {
        /*qi*/
        *(here->HSMHV2QIdpPtr) +=  model->HSMHV2_type * (ydc_qi[dNodePrime] + ag0*ydyn_qi[dNodePrime]);
        *(here->HSMHV2QIgpPtr) +=  model->HSMHV2_type * (ydc_qi[gNodePrime] + ag0*ydyn_qi[gNodePrime]);
        *(here->HSMHV2QIspPtr) +=  model->HSMHV2_type * (ydc_qi[sNodePrime] + ag0*ydyn_qi[sNodePrime]);
        *(here->HSMHV2QIbpPtr) +=  model->HSMHV2_type * (ydc_qi[bNodePrime] + ag0*ydyn_qi[bNodePrime]);
        *(here->HSMHV2QIqiPtr) +=                     (ydc_qi[qiNode] + ag0*ydyn_qi[qiNode]);
        if ( here->HSMHV2_coselfheat > 0 ) { /* self heating */
          *(here->HSMHV2QItempPtr) +=                     (ydc_qi[tempNode] + ag0*ydyn_qi[tempNode]);
        }

        /*qb*/
        *(here->HSMHV2QBdpPtr) +=  model->HSMHV2_type * (ydc_qb[dNodePrime] + ag0*ydyn_qb[dNodePrime]);
        *(here->HSMHV2QBgpPtr) +=  model->HSMHV2_type * (ydc_qb[gNodePrime] + ag0*ydyn_qb[gNodePrime]);
        *(here->HSMHV2QBspPtr) +=  model->HSMHV2_type * (ydc_qb[sNodePrime] + ag0*ydyn_qb[sNodePrime]);
        *(here->HSMHV2QBbpPtr) +=  model->HSMHV2_type * (ydc_qb[bNodePrime] + ag0*ydyn_qb[bNodePrime]);
        *(here->HSMHV2QBqbPtr) +=                     (ydc_qb[qbNode] + ag0*ydyn_qb[qbNode]);
        if ( here->HSMHV2_coselfheat > 0 ) { /* self heating */
          *(here->HSMHV2QBtempPtr) +=                     (ydc_qb[tempNode] + ag0*ydyn_qb[tempNode]);
        }
      }


line1000:

      if (ckt->CKTnoncon != noncon_old) {
        ckt->CKTtroubleElt = (GENinstance *) here;
      }


    } /* End of MOSFET Instance */
  } /* End of Model Instance */

  return(OK);
}
