/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 2  SUBVERSION : 2  REVISION : 0 ) 
 Model Parameter 'VERSION' : 2.20
 FILE : hsmhvtemp.c

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


#define RANGECHECK(param, min, max, pname)                              \
  if ( (param) < (min) || (param) > (max) ) {             \
    printf("warning(HiSIM_HV(%s)): The model/instance parameter %s (= %e) must be in the range [%e , %e].\n", model->HSMHV2modName,\
           (pname), (param), (min), (max) );                     \
  }

#define Fn_SU( y , x , xmax , delta , dx ) { \
    TMF1 = ( xmax ) - ( x ) - ( delta ) ; \
    TMF2 = 4.0 * ( xmax ) * ( delta ) ; \
    TMF2 = TMF2 > 0.0 ?  TMF2 : -( TMF2) ; \
    TMF2 = sqrt ( TMF1 * TMF1 + TMF2 ) ; \
    dx = 0.5 * ( 1.0 + TMF1 / TMF2 ) ; \
    y = ( xmax ) - 0.5 * ( TMF1 + TMF2 ) ; \
  }

#define Fn_SL( y , x , xmin , delta , dx ) { \
    TMF1 = ( x ) - ( xmin ) - ( delta ) ; \
    TMF2 = 4.0 * ( xmin ) * ( delta ) ; \
    TMF2 = TMF2 > 0.0 ?  TMF2 : -( TMF2 ); \
    TMF2 = sqrt ( TMF1 * TMF1 + TMF2 ) ; \
    dx = 0.5 * ( 1.0 + TMF1 / TMF2 ) ; \
    y = ( xmin ) + 0.5 * ( TMF1 + TMF2 ) ; \
  }


#define C_m2cm    (1.0e2) 

int HSMHV2temp(
     GENmodel *inModel,
     CKTcircuit *ckt)
{
  HSMHV2model *model = (HSMHV2model *)inModel ;
  HSMHV2instance *here ;
  HSMHV2binningParam *pParam ;
  HSMHV2modelMKSParam *modelMKS ;
  HSMHV2hereMKSParam  *hereMKS ;
  double mueph =0.0 ;
  double Leff=0.0, dL =0.0, dLLD=0.0, LG=0.0, Weff=0.0, dW =0.0, dWLD=0.0, dWCV=0.0, WG =0.0, WL =0.0, Lgate =0.0, Wgate =0.0 ;
  double Nsubpp=0.0, Nsubps=0.0, Nsub=0.0, q_Nsub=0.0, Nsubb=0.0, Npext =0.0 ;
  double Lod_half=0.0, Lod_half_ref =0.0 ;
  double T0, T1, T2, T3, T4, T5, T6, T7 ;
  /* temperature-dependent variables */
  double Eg =0.0, TTEMP0=0.0, TTEMP=0.0, beta=0.0, Nin=0.0 ;
  double Tdiff0 = 0.0, Tdiff0_2 = 0.0, Tdiff = 0.0, Tdiff_2 = 0.0;
  double js=0.0, jssw=0.0, js2=0.0, jssw2 =0.0 ;
  int i=0 ;
  double TMF1 , TMF2 ;
  double GDLD =0.0 ;
  double log_Tratio =0.0 ;
  const double small = 1.0e-50 ;
  const double dlt_rd23 = 1.0e-6 / C_m2cm ;
  const double large_arg = 80 ;

  for ( ;model ;model = HSMHV2nextModel(model)) {

    modelMKS = &model->modelMKS ;

    model->HSMHV2_vcrit = CONSTvt0 * log( CONSTvt0 / (CONSTroot2 * 1.0e-14) ) ;

    /* Quantum Mechanical Effect */
    if ( ( model->HSMHV2_qme1 == 0.0 && model->HSMHV2_qme3 == 0.0 ) || model->HSMHV2_qme2 == 0.0 ) {
      model->HSMHV2_flg_qme = 0 ;
    } else {
      model->HSMHV2_flg_qme = 1 ;
      model->HSMHV2_qme12 = model->HSMHV2_qme1 / ( model->HSMHV2_qme2 * model->HSMHV2_qme2 ) ;
    }

    for ( here = HSMHV2instances(model); here; here = HSMHV2nextInstance(here)) {

      pParam = &here->pParam ;

      hereMKS = &here->hereMKS ;

      here->HSMHV2_lgate = Lgate = here->HSMHV2_l + model->HSMHV2_xl ;
      Wgate = here->HSMHV2_w / here->HSMHV2_nf  + model->HSMHV2_xw ;

      LG = Lgate * C_m2um ;
      here->HSMHV2_wg = WG = Wgate * C_m2um ; 
      WL = WG * LG ;
      GDLD = model->HSMHV2_gdld * C_m2um ;


      /* Band gap */
      here->HSMHV2_egtnom = pParam->HSMHV2_eg0 - model->HSMHV2_ktnom    
        * ( 90.25e-6 + model->HSMHV2_ktnom * 1.0e-7 ) ;
  
      /* C_EOX */
      here->HSMHV2_cecox = C_VAC * model->HSMHV2_kappa ;
  
      /* Vth reduction for small Vds */
      here->HSMHV2_msc = model->HSMHV2_scp22  ;

      /* Poly-Si Gate Depletion */
      if ( pParam->HSMHV2_pgd1 == 0.0 ) {
        here->HSMHV2_flg_pgd = 0 ;
      } else {
        here->HSMHV2_flg_pgd = 1 ;
      }


      /* CLM5 & CLM6 */
      here->HSMHV2_clmmod = 1e0 + pow( LG , model->HSMHV2_clm5 ) * model->HSMHV2_clm6 ;

      /* Half length of diffusion */
      T1 = 1.0 / (model->HSMHV2_saref + 0.5 * here->HSMHV2_l)
         + 1.0 / (model->HSMHV2_sbref + 0.5 * here->HSMHV2_l);
      Lod_half_ref = 2.0 / T1 ;

      if (here->HSMHV2_sa > 0.0 && here->HSMHV2_sb > 0.0 &&
	  (here->HSMHV2_nf == 1.0 ||
           (here->HSMHV2_nf > 1.0 && here->HSMHV2_sd > 0.0))) {
        T1 = 0.0;
        for (i = 0; i < here->HSMHV2_nf; i++) {
          T1 += 1.0 / (here->HSMHV2_sa + 0.5 * here->HSMHV2_l
                       + i * (here->HSMHV2_sd + here->HSMHV2_l))
              + 1.0 / (here->HSMHV2_sb + 0.5 * here->HSMHV2_l
                       + i * (here->HSMHV2_sd + here->HSMHV2_l));
        }
        Lod_half = 2.0 * here->HSMHV2_nf / T1;
      } else {
        Lod_half = 0.0;
      }

      Npext = pParam->HSMHV2_npext ;
      here->HSMHV2_mueph1 = pParam->HSMHV2_mueph1 ;
      here->HSMHV2_nsubp  = pParam->HSMHV2_nsubp ;
      here->HSMHV2_nsubc  = pParam->HSMHV2_nsubc ;

      /* DFM */
      if ( model->HSMHV2_codfm == 1 && here->HSMHV2_nsubcdfm_Given ) {
	RANGECHECK(here->HSMHV2_nsubcdfm,   1.0e16,   1.0e19, "NSUBCDFM") ;
 	here->HSMHV2_mueph1 *= model->HSMHV2_mphdfm
	  * ( log(hereMKS->HSMHV2_nsubcdfm) - log(here->HSMHV2_nsubc) ) + 1.0 ;
	here->HSMHV2_nsubp += hereMKS->HSMHV2_nsubcdfm - here->HSMHV2_nsubc ;
	Npext += hereMKS->HSMHV2_nsubcdfm - here->HSMHV2_nsubc ;
 	here->HSMHV2_nsubc = hereMKS->HSMHV2_nsubcdfm ;
      }

      /* Phonon Scattering (temperature-independent part) */
      mueph = here->HSMHV2_mueph1  
        * (1.0e0 + (model->HSMHV2_muephw / pow( WG, model->HSMHV2_muepwp))) 
        * (1.0e0 + (model->HSMHV2_muephl / pow( LG, model->HSMHV2_mueplp))) 
        * (1.0e0 + (model->HSMHV2_muephs / pow( WL, model->HSMHV2_muepsp)));  
      if (Lod_half > 0.0) {
        T1 = 1.0e0 / (1.0e0 + pParam->HSMHV2_muesti2) ;
        T2 = pow (pParam->HSMHV2_muesti1 / Lod_half, pParam->HSMHV2_muesti3) ;
        T3 = pow (pParam->HSMHV2_muesti1 / Lod_half_ref, pParam->HSMHV2_muesti3) ;
        here->HSMHV2_mueph = mueph * (1.0e0 + T1 * T2) / (1.0e0 + T1 * T3); 
      } else {
        here->HSMHV2_mueph = mueph;
      }
      
      /* Surface Roughness Scattering */
      here->HSMHV2_muesr = model->HSMHV2_muesr0 
        * (1.0e0 + (model->HSMHV2_muesrl / pow (LG, model->HSMHV2_mueslp))) 
        * (1.0e0 + (model->HSMHV2_muesrw / pow (WG, model->HSMHV2_mueswp))) ;

      /* Coefficients of Qbm for Eeff */
      T1 = pow( LG, model->HSMHV2_ndeplp ) ;
      here->HSMHV2_ndep_o_esi = ( pParam->HSMHV2_ndep * T1 ) / ( T1 + model->HSMHV2_ndepl )
	/ C_ESI ;
      here->HSMHV2_ninv_o_esi = pParam->HSMHV2_ninv / C_ESI ;
      here->HSMHV2_ninvd0 = model->HSMHV2_ninvd * ( 1.0 + (model->HSMHV2_ninvdw / pow( WG, model->HSMHV2_ninvdwp)));

      /* Metallurgical channel geometry */
      dL = model->HSMHV2_xld 
        + (modelMKS->HSMHV2_ll / pow (Lgate + model->HSMHV2_lld, model->HSMHV2_lln)) ;
      dLLD = model->HSMHV2_xldld 
        + (modelMKS->HSMHV2_ll / pow (Lgate + model->HSMHV2_lld, model->HSMHV2_lln)) ;
   
      dW = model->HSMHV2_xwd 
        + (modelMKS->HSMHV2_wl / pow (Wgate + model->HSMHV2_wld, model->HSMHV2_wln)) ;  
      dWLD = model->HSMHV2_xwdld 
        + (modelMKS->HSMHV2_wl / pow (Wgate + model->HSMHV2_wld, model->HSMHV2_wln)) ;  
      dWCV = model->HSMHV2_xwdc 
        + (modelMKS->HSMHV2_wl / pow (Wgate + model->HSMHV2_wld, model->HSMHV2_wln)) ;  
    
      Leff = Lgate - ( dL + dLLD ) ;
      if ( Leff <= 0.0 ) {   
        IFuid namarr[2];
        namarr[0] = here->HSMHV2name;
        namarr[1] = model->HSMHV2modName;
        (*(SPfrontEnd->IFerror))
          ( 
           ERR_FATAL, 
           "HiSIM_HV: MOSFET(%s) MODEL(%s): effective channel length is negative or 0", 
           namarr 
           );
        return (E_BADPARM);
      }
      here->HSMHV2_leff = Leff ;

      /* Wg dependence for short channel devices */
      here->HSMHV2_lgatesm = Lgate + model->HSMHV2_wl1 / pow( WL , model->HSMHV2_wl1p ) ;
      here->HSMHV2_dVthsm = pParam->HSMHV2_wl2 / pow( WL , model->HSMHV2_wl2p ) ;

      /* Lg dependence of wsti */
      T1 = 1.0e0 + model->HSMHV2_wstil / pow( here->HSMHV2_lgatesm * C_m2um  , model->HSMHV2_wstilp ) ;
      T2 = 1.0e0 + model->HSMHV2_wstiw / pow( WG , model->HSMHV2_wstiwp ) ;
      here->HSMHV2_wsti = pParam->HSMHV2_wsti * T1 * T2 ;

      here->HSMHV2_weff = Weff = Wgate - 2.0e0 * dW ;
      here->HSMHV2_weff_ld     = Wgate - 2.0e0 * dWLD ;
      here->HSMHV2_weff_cv     = Wgate - 2.0e0 * dWCV ;
      if ( Weff <= 0.0 ) {   
        IFuid namarr[2];
        namarr[0] = here->HSMHV2name;
        namarr[1] = model->HSMHV2modName;
        (*(SPfrontEnd->IFerror))
          ( 
           ERR_FATAL, 
           "HiSIM_HV: MOSFET(%s) MODEL(%s): effective channel width is negative or 0", 
           namarr 
           );
        return (E_BADPARM);
      }
      here->HSMHV2_weff_nf = Weff * here->HSMHV2_nf ;
      here->HSMHV2_weffcv_nf = here->HSMHV2_weff_cv * here->HSMHV2_nf ;

      /* Surface impurity profile */
      /* Note: Sign Changed --> */
      Nsubpp = here->HSMHV2_nsubp  
        * (1.0e0 + (model->HSMHV2_nsubp0 / pow (WG, model->HSMHV2_nsubwp))) ;
      /* <-- Note: Sign Changed */

      if (Lod_half > 0.0) {
        T1 = 1.0e0 / (1.0e0 + pParam->HSMHV2_nsubpsti2) ;
        T2 = pow (pParam->HSMHV2_nsubpsti1 / Lod_half, pParam->HSMHV2_nsubpsti3) ;
        T3 = pow (pParam->HSMHV2_nsubpsti1 / Lod_half_ref, pParam->HSMHV2_nsubpsti3) ;
        Nsubps = Nsubpp * (1.0e0 + T1 * T2) / (1.0e0 + T1 * T3) ;
      } else {
        Nsubps = Nsubpp ;
      }
   
      here->HSMHV2_nsubc *= 1.0e0 + ( model->HSMHV2_nsubcw / pow ( WG, model->HSMHV2_nsubcwp )) ;

      if( Lgate > model->HSMHV2_lp ){
        Nsub = (here->HSMHV2_nsubc * (Lgate - model->HSMHV2_lp) 
                +  Nsubps  * model->HSMHV2_lp) / Lgate ;
      } else {
        Nsub = Nsubps
          + (Nsubps - here->HSMHV2_nsubc) * (model->HSMHV2_lp - Lgate) 
          / model->HSMHV2_lp ;
      }
      T3 = 0.5e0 * Lgate - model->HSMHV2_lp ;
      T1 = 1.0e0 / ( 1.0e0 / T3 + 1.0e0 / model->HSMHV2_lpext ) ;
      T2 = Fn_Max (0.0e0, T1) ;
      here->HSMHV2_nsub = 
	Nsub = Nsub + T2 * (Npext - here->HSMHV2_nsubc) / Lgate ;
      here->HSMHV2_qnsub = q_Nsub  = C_QE * Nsub ;
      here->HSMHV2_qnsub_esi = q_Nsub * C_ESI ;
      here->HSMHV2_2qnsub_esi = 2.0 * here->HSMHV2_qnsub_esi ;

      /* Pocket Overlap (temperature-independent part) */
      if ( Lgate <= 2.0e0 * model->HSMHV2_lp ) {
        Nsubb = 2.0e0 * Nsubps 
          - (Nsubps - here->HSMHV2_nsubc) * Lgate 
          / model->HSMHV2_lp - here->HSMHV2_nsubc ;
        here->HSMHV2_ptovr0 = log (Nsubb / here->HSMHV2_nsubc) ;
        /* here->HSMHV2_ptovr0 will be divided by beta later. */
      } else {
        here->HSMHV2_ptovr0 = 0.0e0 ;
      }

      /* depletion MOS (temperature-independent part) */
      here->HSMHV2_ndepm = modelMKS->HSMHV2_ndepm ;

      /* costi0 and costi1 for STI transistor model (temperature-independent part) */
      here->HSMHV2_costi00 = sqrt (2.0 * C_QE * pParam->HSMHV2_nsti * C_ESI ) ;
      here->HSMHV2_nsti_p2 = 1.0 / ( pParam->HSMHV2_nsti * pParam->HSMHV2_nsti ) ;

      /* Velocity Temperature Dependence (Temperature-dependent part will be multiplied later.) */
      here->HSMHV2_vmax0 = (1.0e0 + (pParam->HSMHV2_vover / pow (LG, model->HSMHV2_voverp)))
        * (1.0e0 + (model->HSMHV2_vovers / pow (WL, model->HSMHV2_voversp))) ;

      /* 2 phi_B (temperature-independent) */
      /* @300K, with pocket */
      here->HSMHV2_pb20 = 2.0e0 / C_b300 * log (Nsub / C_Nin0) ;
      /* @300K, w/o pocket */
      here->HSMHV2_pb2c = 2.0e0 / C_b300 * log (here->HSMHV2_nsubc / C_Nin0) ;

      /* constant for Poly depletion */
      here->HSMHV2_cnstpgd = pow ( 1e0 + 1e0 / LG , model->HSMHV2_pgd4 ) 
        * pParam->HSMHV2_pgd1 ;

      /* Gate resistance */
      if ( here->HSMHV2_corg == 1 ) {
        T1 = here->HSMHV2_xgw + Weff / (3.0e0 * here->HSMHV2_ngcon);
        T2 = Lgate - here->HSMHV2_xgl;
        here->HSMHV2_grg = model->HSMHV2_rshg * T1 / (here->HSMHV2_ngcon * T2 * here->HSMHV2_nf);
        if (here->HSMHV2_grg > 1.0e-3) here->HSMHV2_grg = here->HSMHV2_m / here->HSMHV2_grg;
        else {
          here->HSMHV2_grg = here->HSMHV2_m * 1.0e3;
          printf("warning(HiSIM_HV(%s)): The gate conductance reset to 1.0e3 mho.\n",model->HSMHV2modName);
        }
      }

      /* Process source/drain series resistamce */

      if ( model->HSMHV2_rsh > 0.0 ) {
        here->HSMHV2_rd0 = model->HSMHV2_rsh * here->HSMHV2_nrd ;
      } else {
	here->HSMHV2_rd0 = 0.0 ;
      }
      if ( pParam->HSMHV2_rd > 0.0 || pParam->HSMHV2_rs > 0.0 ) {
        here->HSMHV2_rdtemp0 = 1.0 + model->HSMHV2_rds / pow( WL , model->HSMHV2_rdsp ) ;
	if( pParam->HSMHV2_rdvd != 0.0 ){
	  T7 = ( 1.0 + model->HSMHV2_rdvds / pow( WL , model->HSMHV2_rdvdsp ) );
          T6 = ( - model->HSMHV2_rdvdl * pow( LG , model->HSMHV2_rdvdlp ) ) ;
          if(T6 > large_arg) T6 = large_arg ;  
          T6 = exp( T6 ) ;
          here->HSMHV2_rdvdtemp0 = T6 * T7 ;
        }
      }
      if( pParam->HSMHV2_rd23 != 0.0 ){
	T2 = ( 1.0 + model->HSMHV2_rd23s / pow( WL , model->HSMHV2_rd23sp ) );
        T1 = ( - model->HSMHV2_rd23l * pow( LG , model->HSMHV2_rd23lp ) ) ;
        if(T1 > large_arg)  T1 = large_arg ; 
        T1 = exp( T1 ) ;
        T3 = pParam->HSMHV2_rd23 * T2 * T1 ;
        here->HSMHV2_rd23 = 0.5 * ( T3 + sqrt ( T3 * T3 + 4.0 * dlt_rd23 * dlt_rd23 ) ) ;
      } else {
	here->HSMHV2_rd23 = 0.0 ;
      }
      if ( model->HSMHV2_rsh > 0.0 ) {
        here->HSMHV2_rs0 = model->HSMHV2_rsh * here->HSMHV2_nrs ;
      } else {
	here->HSMHV2_rs0 = 0.0 ;
      }

      here->HSMHV2_Xmax  = sqrt ( model->HSMHV2_rdrdjunc * model->HSMHV2_rdrdjunc + model->HSMHV2_xldld * model->HSMHV2_xldld ) ;
      if(pParam->HSMHV2_nover != 0.0) {
      here->HSMHV2_kdep  = 2.0 * C_ESI / ( C_QE * pParam->HSMHV2_nover ) ;
      here->HSMHV2_kjunc = 2.0 * C_ESI / C_QE * here->HSMHV2_nsubc / ( pParam->HSMHV2_nover + here->HSMHV2_nsubc ) / pParam->HSMHV2_nover ;
      }
      here->HSMHV2_rdrcxw   = 1.0e0 ;
      here->HSMHV2_rdrvmaxw = 1.0e0 + (model->HSMHV2_rdrvmaxw / pow( WG, model->HSMHV2_rdrvmaxwp)) ;
      here->HSMHV2_rdrvmaxl = 1.0e0 + (model->HSMHV2_rdrvmaxl / pow( LG, model->HSMHV2_rdrvmaxlp)) ;
      here->HSMHV2_rdrmuel  = 1.0e0 + (model->HSMHV2_rdrmuel  / pow( LG, model->HSMHV2_rdrmuelp )) ;

      /* Body resistance */
      if ( here->HSMHV2_corbnet == 1 ) {
        if (here->HSMHV2_rbpb < 1.0e-3) here->HSMHV2_grbpb = here->HSMHV2_m * 1.0e3 ;
        else here->HSMHV2_grbpb = here->HSMHV2_m * ( model->HSMHV2_gbmin + 1.0 / here->HSMHV2_rbpb ) ;

        if (here->HSMHV2_rbps < 1.0e-3) here->HSMHV2_grbps = here->HSMHV2_m * 1.0e3 ;
        else here->HSMHV2_grbps = here->HSMHV2_m * ( model->HSMHV2_gbmin + 1.0 / here->HSMHV2_rbps ) ;

        if (here->HSMHV2_rbpd < 1.0e-3) here->HSMHV2_grbpd = here->HSMHV2_m * 1.0e3 ;
        else here->HSMHV2_grbpd = here->HSMHV2_m * ( model->HSMHV2_gbmin + 1.0 / here->HSMHV2_rbpd ) ;
      }

      /* Vdseff */
      if( model->HSMHV2_coddlt == 0) {
        T1 = model->HSMHV2_ddltslp * LG + model->HSMHV2_ddltict ;
        if ( T1 < 0.0 ) { T1 = 0.0 ; }
        here->HSMHV2_ddlt = T1 * model->HSMHV2_ddltmax / ( T1 + model->HSMHV2_ddltmax ) + 1.0 ;
      } else {
        T1 = model->HSMHV2_ddltslp * LG  ;
        if ( T1 < 0.0 ) { T1 = 0.0 ; }
        here->HSMHV2_ddlt = T1 * model->HSMHV2_ddltmax / ( T1 + model->HSMHV2_ddltmax ) +  model->HSMHV2_ddltict + small ;
      }


      /* Isub */
      T2 = pow( Weff , model->HSMHV2_svgswp ) ;
      here->HSMHV2_vg2const = pParam->HSMHV2_svgs
         * ( 1.0e0
           + modelMKS->HSMHV2_svgsl / pow( here->HSMHV2_lgate , model->HSMHV2_svgslp ) )
         * ( T2 / ( T2 + modelMKS->HSMHV2_svgsw ) ) ; 

      here->HSMHV2_xvbs = pParam->HSMHV2_svbs 
         * ( 1.0e0
           + modelMKS->HSMHV2_svbsl / pow( here->HSMHV2_lgate , model->HSMHV2_svbslp ) ) ;
      here->HSMHV2_xgate = modelMKS->HSMHV2_slg  
         * ( 1.0
         + modelMKS->HSMHV2_slgl / pow( here->HSMHV2_lgate , model->HSMHV2_slglp ) ) ;

      here->HSMHV2_xsub1 = pParam->HSMHV2_sub1 
         * ( 1.0 
         + modelMKS->HSMHV2_sub1l / pow( here->HSMHV2_lgate , model->HSMHV2_sub1lp ) ) ;

      here->HSMHV2_xsub2 = pParam->HSMHV2_sub2
         * ( 1.0 + modelMKS->HSMHV2_sub2l / here->HSMHV2_lgate ) ;

      here->HSMHV2_subld1 = model->HSMHV2_subld1
         * ( 1.0 + model->HSMHV2_subld1l / pow( LG , model->HSMHV2_subld1lp ) ) ;

      /* IBPC */
      here->HSMHV2_ibpc1  = pParam->HSMHV2_ibpc1
         * ( 1.0 + model->HSMHV2_ibpc1l  / pow( LG , model->HSMHV2_ibpc1lp ) ) ;

      /* Fringing capacitance */
      here->HSMHV2_cfrng = C_EOX / ( C_Pi / 2.0e0 ) * here->HSMHV2_weff_nf
         * log( 1.0e0 + model->HSMHV2_tpoly / model->HSMHV2_tox ) ;

      /* Additional term of lateral-field-induced capacitance */
      here->HSMHV2_cqyb0 = C_m2um * here->HSMHV2_weff_nf
	* model->HSMHV2_xqy1 / pow( LG , model->HSMHV2_xqy2 ) ;

      /* Parasitic component of the channel current */
      here->HSMHV2_ptl0 = model->HSMHV2_ptl * pow( LG        , - model->HSMHV2_ptlp ) ;
      here->HSMHV2_pt40 = model->HSMHV2_pt4 * pow( LG        , - model->HSMHV2_pt4p ) ;
      here->HSMHV2_gdl0 = model->HSMHV2_gdl * pow( LG + GDLD , - model->HSMHV2_gdlp ) ;

      /* Self heating */
      pParam->HSMHV2_rth = pParam->HSMHV2_rth0 / ( here->HSMHV2_m * here->HSMHV2_weff_nf )
	* ( 1.0 + model->HSMHV2_rth0w / pow( WG , model->HSMHV2_rth0wp ) );
      pParam->HSMHV2_cth = modelMKS->HSMHV2_cth0 * ( here->HSMHV2_m * here->HSMHV2_weff_nf ) ;

      pParam->HSMHV2_rth *= ( 1.0 / pow( here->HSMHV2_nf , model->HSMHV2_rth0nf ) ) ;
      
      here->HSMHV2_rthtemp0 = 1.0 / pow( here->HSMHV2_nf , model->HSMHV2_rth0nf ) / ( here->HSMHV2_m * here->HSMHV2_weff_nf )
       * ( 1.0 + model->HSMHV2_rth0w / pow( WG , model->HSMHV2_rth0wp ) );


      /*-----------------------------------------------------------*
       * Temperature dependent constants. 
       *-----------------*/
      if ( here->HSMHV2tempNode < 0 || pParam->HSMHV2_rth0 == 0.0 ) {

#include "hsmhv2temp_eval.h"
#include "hsmhv2temp_eval_rdri.h"
#include "hsmhv2temp_eval_dio.h"

      } /* end of if ( here->HSMHV2tempNode < 0 || pParam->HSMHV2_rth0 == 0.0 ) */

    }
  }
  return(OK);
}
