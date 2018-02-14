/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 2  SUBVERSION : 2  REVISION : 0 ) 
 Model Parameter 'VERSION' : 2.20
 FILE : hsmhvcvtest.c

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
#include "ngspice/cktdefs.h"
#include "hsmhv2def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int HSMHV2convTest(
     GENmodel *inModel,
     CKTcircuit *ckt)
{
  HSMHV2model *model = (HSMHV2model*)inModel;
  HSMHV2instance *here;
  double      vds=0.0,    vgs=0.0,    vbs=0.0,    vdse=0.0,    vgse=0.0,    vbse=0.0,    vdbd=0.0,    vsbs=0.0,    deltemp =0.0 ;
  double   delvds=0.0, delvgs=0.0, delvbs=0.0, delvdse=0.0, delvgse=0.0, delvbse=0.0, delvdbd=0.0, delvsbs=0.0, deldeltemp =0.0 ;
  double   Ids=0.0, gds=0.0, gm=0.0, gmbs=0.0, gmT=0.0, gmbs_ext=0.0, gds_ext=0.0, gm_ext=0.0,
           Isub=0.0, dIsub_dVds=0.0, dIsub_dVgs=0.0, dIsub_dVbs=0.0, dIsub_dT=0.0,
           Isubs=0.0, dIsubs_dVds=0.0, dIsubs_dVgs=0.0, dIsubs_dVbs=0.0, dIsubs_dT=0.0,
           Igidl=0.0, dIgidl_dVds=0.0, dIgidl_dVgs=0.0, dIgidl_dVbs=0.0, dIgidl_dT=0.0,
           Igisl=0.0, dIgisl_dVds=0.0, dIgisl_dVgs=0.0, dIgisl_dVbs=0.0, dIgisl_dT=0.0,
           Igd=0.0, dIgd_dVd=0.0, dIgd_dVg=0.0, dIgd_dVb=0.0, dIgd_dT=0.0,
           Igs=0.0, dIgs_dVd=0.0, dIgs_dVg=0.0, dIgs_dVb=0.0, dIgs_dT=0.0,
           Igb=0.0, dIgb_dVd=0.0, dIgb_dVg=0.0, dIgb_dVb=0.0, dIgb_dT=0.0,
           Ibd=0.0, Gbd=0.0, Gbdt=0.0,
           Ibs=0.0, Gbs=0.0, Gbst =0.0 ;
  double   i_dP=0.0,     i_gP=0.0,     i_sP=0.0,     i_db=0.0,     i_sb=0.0,
           i_dP_hat=0.0, i_gP_hat=0.0, i_sP_hat=0.0, i_db_hat=0.0, i_sb_hat =0.0 ;
  double   tol0=0.0, tol1=0.0, tol2=0.0, tol3=0.0, tol4 =0.0 ;


  /*  loop through all the HSMHV2 device models */
  for ( ; model != NULL; model = HSMHV2nextModel(model)) {
    /* loop through all the instances of the model */
    for ( here = HSMHV2instances(model); here != NULL ;
	  here = HSMHV2nextInstance(here)) {

      vds = model->HSMHV2_type * 
            (*(ckt->CKTrhsOld+here->HSMHV2dNodePrime) -
             *(ckt->CKTrhsOld+here->HSMHV2sNodePrime));
      vgs = model->HSMHV2_type * 
            (*(ckt->CKTrhsOld+here->HSMHV2gNodePrime) -
             *(ckt->CKTrhsOld+here->HSMHV2sNodePrime));
      vbs = model->HSMHV2_type * 
            (*(ckt->CKTrhsOld+here->HSMHV2bNodePrime) -
             *(ckt->CKTrhsOld+here->HSMHV2sNodePrime));
      vdse = model->HSMHV2_type * 
	      (*(ckt->CKTrhsOld+here->HSMHV2dNode) -
	       *(ckt->CKTrhsOld+here->HSMHV2sNode));
      vgse = model->HSMHV2_type * 
	      (*(ckt->CKTrhsOld+here->HSMHV2gNodePrime) -
	       *(ckt->CKTrhsOld+here->HSMHV2sNode));
      vbse = model->HSMHV2_type * 
	      (*(ckt->CKTrhsOld+here->HSMHV2bNodePrime) -
	       *(ckt->CKTrhsOld+here->HSMHV2sNode));
      vdbd = model->HSMHV2_type
            * (*(ckt->CKTrhsOld + here->HSMHV2dbNode)
               - *(ckt->CKTrhsOld + here->HSMHV2dNode));
      vsbs = model->HSMHV2_type
            * (*(ckt->CKTrhsOld + here->HSMHV2sbNode)
               - *(ckt->CKTrhsOld + here->HSMHV2sNode));
      if( here->HSMHV2tempNode > 0 ){
	deltemp = *(ckt->CKTrhsOld + here->HSMHV2tempNode);
      } else {
        deltemp = 0.0 ;
      }

      delvds  = vds  - *(ckt->CKTstate0 + here->HSMHV2vds) ;
      delvgs  = vgs  - *(ckt->CKTstate0 + here->HSMHV2vgs) ;
      delvbs  = vbs  - *(ckt->CKTstate0 + here->HSMHV2vbs) ;
      delvdse = vdse - *(ckt->CKTstate0 + here->HSMHV2vdse) ;
      delvgse = vgse - *(ckt->CKTstate0 + here->HSMHV2vgse) ;
      delvbse = vbse - *(ckt->CKTstate0 + here->HSMHV2vbse) ;
      delvdbd = vdbd - *(ckt->CKTstate0 + here->HSMHV2vdbd) ;
      delvsbs = vsbs - *(ckt->CKTstate0 + here->HSMHV2vsbs) ;
      if( here->HSMHV2tempNode > 0 ){
        deldeltemp = deltemp - *(ckt->CKTstate0 + here->HSMHV2deltemp) ;
      } else {
        deldeltemp = 0.0 ;
      }

      if ( here->HSMHV2_mode > 0 ) { /* forward mode */
        Ids        = here->HSMHV2_ids ;
        gds        = here->HSMHV2_dIds_dVdsi ;
        gm         = here->HSMHV2_dIds_dVgsi ;
        gmbs       = here->HSMHV2_dIds_dVbsi ;
        gmT        = (here->HSMHV2tempNode > 0) ? here->HSMHV2_dIds_dTi : 0.0  ;
	gmbs_ext   = here->HSMHV2_dIds_dVbse;
	gds_ext    = here->HSMHV2_dIds_dVdse ;
	gm_ext     = here->HSMHV2_dIds_dVgse;
	Isub         = here->HSMHV2_isub ;
	dIsub_dVds   = here->HSMHV2_dIsub_dVdsi ; 
	dIsub_dVgs   = here->HSMHV2_dIsub_dVgsi ; 
	dIsub_dVbs   = here->HSMHV2_dIsub_dVbsi ;
        dIsub_dT     = (here->HSMHV2tempNode > 0) ? here->HSMHV2_dIsub_dTi : 0.0  ; 
	Isubs        = 0.0 ;
	dIsubs_dVds  = 0.0 ; 
	dIsubs_dVgs  = 0.0 ; 
	dIsubs_dVbs  = 0.0 ;
        dIsubs_dT    = 0.0 ;
        Igidl        = here->HSMHV2_igidl ;
        dIgidl_dVds  = here->HSMHV2_dIgidl_dVdsi ;
        dIgidl_dVgs  = here->HSMHV2_dIgidl_dVgsi ;
        dIgidl_dVbs  = here->HSMHV2_dIgidl_dVbsi ;
        dIgidl_dT    = (here->HSMHV2tempNode > 0) ? here->HSMHV2_dIgidl_dTi : 0.0  ;
        Igisl        = here->HSMHV2_igisl ;
        dIgisl_dVds  = here->HSMHV2_dIgisl_dVdsi ;
        dIgisl_dVgs  = here->HSMHV2_dIgisl_dVgsi ;
        dIgisl_dVbs  = here->HSMHV2_dIgisl_dVbsi ;
        dIgisl_dT    = (here->HSMHV2tempNode > 0) ? here->HSMHV2_dIgisl_dTi : 0.0  ;
        Igd          = here->HSMHV2_igd ;
        dIgd_dVd   = here->HSMHV2_dIgd_dVdsi ;
        dIgd_dVg   = here->HSMHV2_dIgd_dVgsi ;
        dIgd_dVb   = here->HSMHV2_dIgd_dVbsi ;
        dIgd_dT      = (here->HSMHV2tempNode > 0) ? here->HSMHV2_dIgd_dTi : 0.0  ;
        Igs          = here->HSMHV2_igs ;
        dIgs_dVd   = here->HSMHV2_dIgs_dVdsi ;
        dIgs_dVg   = here->HSMHV2_dIgs_dVgsi ;
        dIgs_dVb   = here->HSMHV2_dIgs_dVbsi ;
        dIgs_dT      = (here->HSMHV2tempNode > 0) ? here->HSMHV2_dIgs_dTi : 0.0  ;
        Igb          = here->HSMHV2_igb ;
        dIgb_dVd   = here->HSMHV2_dIgb_dVdsi ;
        dIgb_dVg   = here->HSMHV2_dIgb_dVgsi ;
        dIgb_dVb   = here->HSMHV2_dIgb_dVbsi ;
        dIgb_dT      = (here->HSMHV2tempNode > 0) ? here->HSMHV2_dIgb_dTi : 0.0  ;
	Ibd = here->HSMHV2_ibd ;
	Gbd = here->HSMHV2_gbd ;
	Gbdt = (here->HSMHV2tempNode > 0) ? here->HSMHV2_gbdT : 0.0 ;
	Ibs = here->HSMHV2_ibs ;
	Gbs = here->HSMHV2_gbs ;
	Gbst = (here->HSMHV2tempNode > 0) ? here->HSMHV2_gbsT : 0.0 ;
      } else { /* reverse mode */
        Ids       = - here->HSMHV2_ids ;
        gds       = + (here->HSMHV2_dIds_dVdsi + here->HSMHV2_dIds_dVgsi + here->HSMHV2_dIds_dVbsi) ;
        gm        = - here->HSMHV2_dIds_dVgsi ;
        gmbs      = - here->HSMHV2_dIds_dVbsi ;
        gmT       = (here->HSMHV2tempNode > 0) ? - here->HSMHV2_dIds_dTi : 0.0  ;
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
        dIsubs_dT    =   (here->HSMHV2tempNode > 0) ? here->HSMHV2_dIsub_dTi : 0.0 ;
        Igidl        =   here->HSMHV2_igisl ;
        dIgidl_dVds  = - (here->HSMHV2_dIgisl_dVdsi + here->HSMHV2_dIgisl_dVgsi + here->HSMHV2_dIgisl_dVbsi) ;
        dIgidl_dVgs  =   here->HSMHV2_dIgisl_dVgsi ;
        dIgidl_dVbs  =   here->HSMHV2_dIgisl_dVbsi ;
        dIgidl_dT    =   (here->HSMHV2tempNode > 0) ? here->HSMHV2_dIgisl_dTi : 0.0  ;
        Igisl        =   here->HSMHV2_igidl ;
        dIgisl_dVds  = - (here->HSMHV2_dIgidl_dVdsi + here->HSMHV2_dIgidl_dVgsi + here->HSMHV2_dIgidl_dVbsi) ;
        dIgisl_dVgs  =   here->HSMHV2_dIgidl_dVgsi ;
        dIgisl_dVbs  =   here->HSMHV2_dIgidl_dVbsi ;
        dIgisl_dT    =   (here->HSMHV2tempNode > 0) ? here->HSMHV2_dIgidl_dTi : 0.0  ;
        Igd          =   here->HSMHV2_igd ;
        dIgd_dVd   = - (here->HSMHV2_dIgs_dVdsi + here->HSMHV2_dIgs_dVgsi + here->HSMHV2_dIgs_dVbsi) ;
        dIgd_dVg   =   here->HSMHV2_dIgs_dVgsi ;
        dIgd_dVb   =   here->HSMHV2_dIgs_dVbsi ;
        dIgd_dT      =   (here->HSMHV2tempNode > 0) ? here->HSMHV2_dIgs_dTi : 0.0  ;
        Igs          =   here->HSMHV2_igs ;
        dIgs_dVd   = - (here->HSMHV2_dIgd_dVdsi + here->HSMHV2_dIgd_dVgsi + here->HSMHV2_dIgd_dVbsi) ;
        dIgs_dVg   =   here->HSMHV2_dIgd_dVgsi ;
        dIgs_dVb   =   here->HSMHV2_dIgd_dVbsi ;
        dIgs_dT      =   (here->HSMHV2tempNode > 0) ? here->HSMHV2_dIgd_dTi : 0.0  ;
        Igb          =   here->HSMHV2_igb ;
        dIgb_dVd   = - (here->HSMHV2_dIgb_dVdsi + here->HSMHV2_dIgb_dVgsi + here->HSMHV2_dIgb_dVbsi) ;
        dIgb_dVg   =   here->HSMHV2_dIgb_dVgsi ;
        dIgb_dVb   =   here->HSMHV2_dIgb_dVbsi ;
        dIgb_dT      =   (here->HSMHV2tempNode > 0) ? here->HSMHV2_dIgb_dTi : 0.0  ;
	Ibd = here->HSMHV2_ibd ;
	Gbd = here->HSMHV2_gbd ;
	Gbdt = (here->HSMHV2tempNode > 0) ? here->HSMHV2_gbdT : 0.0 ;
	Ibs = here->HSMHV2_ibs ;
	Gbs = here->HSMHV2_gbs ;
	Gbst = (here->HSMHV2tempNode > 0) ? here->HSMHV2_gbsT : 0.0 ;
      } /* end of reverse mode */

      /* for convergence control, only nonlinear static currents are considered: */
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

      /* to be added: power source for thermal network */


      /*
       *  check convergence
       */
      if ( here->HSMHV2_off == 0  || !(ckt->CKTmode & MODEINITFIX) ) {
	tol0 = ckt->CKTreltol * MAX(fabs(i_dP_hat), fabs(i_dP)) + ckt->CKTabstol;
	tol1 = ckt->CKTreltol * MAX(fabs(i_gP_hat), fabs(i_gP)) + ckt->CKTabstol;
	tol2 = ckt->CKTreltol * MAX(fabs(i_sP_hat), fabs(i_sP)) + ckt->CKTabstol;
	tol3 = ckt->CKTreltol * MAX(fabs(i_db_hat), fabs(i_db)) + ckt->CKTabstol;
	tol4 = ckt->CKTreltol * MAX(fabs(i_sb_hat), fabs(i_sb)) + ckt->CKTabstol;

	if (    (fabs(i_dP_hat - i_dP) >= tol0)
	     || (fabs(i_gP_hat - i_gP) >= tol1) 
	     || (fabs(i_sP_hat - i_sP) >= tol2)
	     || (fabs(i_db_hat - i_db) >= tol3) 
	     || (fabs(i_sb_hat - i_sb) >= tol4) ) {
	  ckt->CKTnoncon++;
	  return(OK);
	}
      }
    }
  }
  return(OK);
}
