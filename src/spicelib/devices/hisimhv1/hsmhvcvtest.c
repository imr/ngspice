/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 1  SUBVERSION : 2  REVISION : 4 )
 Model Parameter VERSION : 1.23
 FILE : hsmhvcvtest.c

 DATE : 2013.04.30

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hsmhvdef.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int HSMHVconvTest(
     GENmodel *inModel,
     register CKTcircuit *ckt)
{
  register HSMHVmodel *model = (HSMHVmodel*)inModel;
  register HSMHVinstance *here;
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


  /*  loop through all the HSMHV device models */
  for ( ; model != NULL; model = HSMHVnextModel(model)) {
    /* loop through all the instances of the model */
    for ( here = HSMHVinstances(model); here != NULL ;
	  here = HSMHVnextInstance(here)) {

      vds = model->HSMHV_type * 
            (*(ckt->CKTrhsOld+here->HSMHVdNodePrime) -
             *(ckt->CKTrhsOld+here->HSMHVsNodePrime));
      vgs = model->HSMHV_type * 
            (*(ckt->CKTrhsOld+here->HSMHVgNodePrime) -
             *(ckt->CKTrhsOld+here->HSMHVsNodePrime));
      vbs = model->HSMHV_type * 
            (*(ckt->CKTrhsOld+here->HSMHVbNodePrime) -
             *(ckt->CKTrhsOld+here->HSMHVsNodePrime));
      vdse = model->HSMHV_type * 
	      (*(ckt->CKTrhsOld+here->HSMHVdNode) -
	       *(ckt->CKTrhsOld+here->HSMHVsNode));
      vgse = model->HSMHV_type * 
	      (*(ckt->CKTrhsOld+here->HSMHVgNodePrime) -
	       *(ckt->CKTrhsOld+here->HSMHVsNode));
      vbse = model->HSMHV_type * 
	      (*(ckt->CKTrhsOld+here->HSMHVbNodePrime) -
	       *(ckt->CKTrhsOld+here->HSMHVsNode));
      vdbd = model->HSMHV_type
            * (*(ckt->CKTrhsOld + here->HSMHVdbNode)
               - *(ckt->CKTrhsOld + here->HSMHVdNode));
      vsbs = model->HSMHV_type
            * (*(ckt->CKTrhsOld + here->HSMHVsbNode)
               - *(ckt->CKTrhsOld + here->HSMHVsNode));
      if( here->HSMHVtempNode > 0 ){
	deltemp = *(ckt->CKTrhsOld + here->HSMHVtempNode);
      } else {
        deltemp = 0.0 ;
      }

      delvds  = vds  - *(ckt->CKTstate0 + here->HSMHVvds) ;
      delvgs  = vgs  - *(ckt->CKTstate0 + here->HSMHVvgs) ;
      delvbs  = vbs  - *(ckt->CKTstate0 + here->HSMHVvbs) ;
      delvdse = vdse - *(ckt->CKTstate0 + here->HSMHVvdse) ;
      delvgse = vgse - *(ckt->CKTstate0 + here->HSMHVvgse) ;
      delvbse = vbse - *(ckt->CKTstate0 + here->HSMHVvbse) ;
      delvdbd = vdbd - *(ckt->CKTstate0 + here->HSMHVvdbd) ;
      delvsbs = vsbs - *(ckt->CKTstate0 + here->HSMHVvsbs) ;
      if( here->HSMHVtempNode > 0 ){
        deldeltemp = deltemp - *(ckt->CKTstate0 + here->HSMHVdeltemp) ;
      } else {
        deldeltemp = 0.0 ;
      }

      if ( here->HSMHV_mode > 0 ) { /* forward mode */
        Ids        = here->HSMHV_ids ;
        gds        = here->HSMHV_dIds_dVdsi ;
        gm         = here->HSMHV_dIds_dVgsi ;
        gmbs       = here->HSMHV_dIds_dVbsi ;
        gmT        = (here->HSMHVtempNode > 0) ? here->HSMHV_dIds_dTi : 0.0  ;
	gmbs_ext   = here->HSMHV_dIds_dVbse;
	gds_ext    = here->HSMHV_dIds_dVdse ;
	gm_ext     = here->HSMHV_dIds_dVgse;
	Isub         = here->HSMHV_isub ;
	dIsub_dVds   = here->HSMHV_dIsub_dVdsi ; 
	dIsub_dVgs   = here->HSMHV_dIsub_dVgsi ; 
	dIsub_dVbs   = here->HSMHV_dIsub_dVbsi ;
        dIsub_dT     = (here->HSMHVtempNode > 0) ? here->HSMHV_dIsub_dTi : 0.0  ; 
	Isubs        = 0.0 ;
	dIsubs_dVds  = 0.0 ; 
	dIsubs_dVgs  = 0.0 ; 
	dIsubs_dVbs  = 0.0 ;
        dIsubs_dT    = 0.0 ;
        Igidl        = here->HSMHV_igidl ;
        dIgidl_dVds  = here->HSMHV_dIgidl_dVdsi ;
        dIgidl_dVgs  = here->HSMHV_dIgidl_dVgsi ;
        dIgidl_dVbs  = here->HSMHV_dIgidl_dVbsi ;
        dIgidl_dT    = (here->HSMHVtempNode > 0) ? here->HSMHV_dIgidl_dTi : 0.0  ;
        Igisl        = here->HSMHV_igisl ;
        dIgisl_dVds  = here->HSMHV_dIgisl_dVdsi ;
        dIgisl_dVgs  = here->HSMHV_dIgisl_dVgsi ;
        dIgisl_dVbs  = here->HSMHV_dIgisl_dVbsi ;
        dIgisl_dT    = (here->HSMHVtempNode > 0) ? here->HSMHV_dIgisl_dTi : 0.0  ;
        Igd          = here->HSMHV_igd ;
        dIgd_dVd   = here->HSMHV_dIgd_dVdsi ;
        dIgd_dVg   = here->HSMHV_dIgd_dVgsi ;
        dIgd_dVb   = here->HSMHV_dIgd_dVbsi ;
        dIgd_dT      = (here->HSMHVtempNode > 0) ? here->HSMHV_dIgd_dTi : 0.0  ;
        Igs          = here->HSMHV_igs ;
        dIgs_dVd   = here->HSMHV_dIgs_dVdsi ;
        dIgs_dVg   = here->HSMHV_dIgs_dVgsi ;
        dIgs_dVb   = here->HSMHV_dIgs_dVbsi ;
        dIgs_dT      = (here->HSMHVtempNode > 0) ? here->HSMHV_dIgs_dTi : 0.0  ;
        Igb          = here->HSMHV_igb ;
        dIgb_dVd   = here->HSMHV_dIgb_dVdsi ;
        dIgb_dVg   = here->HSMHV_dIgb_dVgsi ;
        dIgb_dVb   = here->HSMHV_dIgb_dVbsi ;
        dIgb_dT      = (here->HSMHVtempNode > 0) ? here->HSMHV_dIgb_dTi : 0.0  ;
	Ibd = here->HSMHV_ibd ;
	Gbd = here->HSMHV_gbd ;
	Gbdt = (here->HSMHVtempNode > 0) ? here->HSMHV_gbdT : 0.0 ;
	Ibs = here->HSMHV_ibs ;
	Gbs = here->HSMHV_gbs ;
	Gbst = (here->HSMHVtempNode > 0) ? here->HSMHV_gbsT : 0.0 ;
      } else { /* reverse mode */
        Ids       = - here->HSMHV_ids ;
        gds       = + (here->HSMHV_dIds_dVdsi + here->HSMHV_dIds_dVgsi + here->HSMHV_dIds_dVbsi) ;
        gm        = - here->HSMHV_dIds_dVgsi ;
        gmbs      = - here->HSMHV_dIds_dVbsi ;
        gmT       = (here->HSMHVtempNode > 0) ? - here->HSMHV_dIds_dTi : 0.0  ;
	gds_ext   = + (here->HSMHV_dIds_dVdse + here->HSMHV_dIds_dVgse + here->HSMHV_dIds_dVbse) ;
	gm_ext    = - here->HSMHV_dIds_dVgse;
	gmbs_ext  = - here->HSMHV_dIds_dVbse;
	Isub         = 0.0 ;
	dIsub_dVds   = 0.0 ; 
	dIsub_dVgs   = 0.0 ; 
	dIsub_dVbs   = 0.0 ;
        dIsub_dT     = 0.0 ; 
	Isubs        =   here->HSMHV_isub ;
	dIsubs_dVds  = - (here->HSMHV_dIsub_dVdsi + here->HSMHV_dIsub_dVgsi + here->HSMHV_dIsub_dVbsi) ; 
	dIsubs_dVgs  =   here->HSMHV_dIsub_dVgsi ; 
	dIsubs_dVbs  =   here->HSMHV_dIsub_dVbsi ;
        dIsubs_dT    =   (here->HSMHVtempNode > 0) ? here->HSMHV_dIsub_dTi : 0.0 ;
        Igidl        =   here->HSMHV_igisl ;
        dIgidl_dVds  = - (here->HSMHV_dIgisl_dVdsi + here->HSMHV_dIgisl_dVgsi + here->HSMHV_dIgisl_dVbsi) ;
        dIgidl_dVgs  =   here->HSMHV_dIgisl_dVgsi ;
        dIgidl_dVbs  =   here->HSMHV_dIgisl_dVbsi ;
        dIgidl_dT    =   (here->HSMHVtempNode > 0) ? here->HSMHV_dIgisl_dTi : 0.0  ;
        Igisl        =   here->HSMHV_igidl ;
        dIgisl_dVds  = - (here->HSMHV_dIgidl_dVdsi + here->HSMHV_dIgidl_dVgsi + here->HSMHV_dIgidl_dVbsi) ;
        dIgisl_dVgs  =   here->HSMHV_dIgidl_dVgsi ;
        dIgisl_dVbs  =   here->HSMHV_dIgidl_dVbsi ;
        dIgisl_dT    =   (here->HSMHVtempNode > 0) ? here->HSMHV_dIgidl_dTi : 0.0  ;
        Igd          =   here->HSMHV_igd ;
        dIgd_dVd   = - (here->HSMHV_dIgs_dVdsi + here->HSMHV_dIgs_dVgsi + here->HSMHV_dIgs_dVbsi) ;
        dIgd_dVg   =   here->HSMHV_dIgs_dVgsi ;
        dIgd_dVb   =   here->HSMHV_dIgs_dVbsi ;
        dIgd_dT      =   (here->HSMHVtempNode > 0) ? here->HSMHV_dIgs_dTi : 0.0  ;
        Igs          =   here->HSMHV_igs ;
        dIgs_dVd   = - (here->HSMHV_dIgd_dVdsi + here->HSMHV_dIgd_dVgsi + here->HSMHV_dIgd_dVbsi) ;
        dIgs_dVg   =   here->HSMHV_dIgd_dVgsi ;
        dIgs_dVb   =   here->HSMHV_dIgd_dVbsi ;
        dIgs_dT      =   (here->HSMHVtempNode > 0) ? here->HSMHV_dIgd_dTi : 0.0  ;
        Igb          =   here->HSMHV_igb ;
        dIgb_dVd   = - (here->HSMHV_dIgb_dVdsi + here->HSMHV_dIgb_dVgsi + here->HSMHV_dIgb_dVbsi) ;
        dIgb_dVg   =   here->HSMHV_dIgb_dVgsi ;
        dIgb_dVb   =   here->HSMHV_dIgb_dVbsi ;
        dIgb_dT      =   (here->HSMHVtempNode > 0) ? here->HSMHV_dIgb_dTi : 0.0  ;
	Ibd = here->HSMHV_ibd ;
	Gbd = here->HSMHV_gbd ;
	Gbdt = (here->HSMHVtempNode > 0) ? here->HSMHV_gbdT : 0.0 ;
	Ibs = here->HSMHV_ibs ;
	Gbs = here->HSMHV_gbs ;
	Gbst = (here->HSMHVtempNode > 0) ? here->HSMHV_gbsT : 0.0 ;
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
      if ( here->HSMHV_off == 0  || !(ckt->CKTmode & MODEINITFIX) ) {
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
