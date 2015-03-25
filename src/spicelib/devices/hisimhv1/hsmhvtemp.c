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
#include "hsmhvdef.h"
#include "hsmhvevalenv.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


#define RANGECHECK(param, min, max, pname)                              \
  if ( (param) < (min) || (param) > (max) ) {             \
    printf("warning(HiSIM_HV(%s)): The model/instance parameter %s (= %e) must be in the range [%e , %e].\n", model->HSMHVmodName,\
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

int HSMHVtemp(
     GENmodel *inModel,
     CKTcircuit *ckt)
{
  HSMHVmodel *model = (HSMHVmodel *)inModel ;
  HSMHVinstance *here ;
  HSMHVbinningParam *pParam ;
  HSMHVmodelMKSParam *modelMKS ;
  HSMHVhereMKSParam  *hereMKS ;
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

  for ( ;model ;model = model->HSMHVnextModel ) {

    modelMKS = &model->modelMKS ;

    model->HSMHV_vcrit = CONSTvt0 * log( CONSTvt0 / (CONSTroot2 * 1.0e-14) ) ;

    /* Quantum Mechanical Effect */
    if ( ( model->HSMHV_qme1 == 0.0 && model->HSMHV_qme3 == 0.0 ) || model->HSMHV_qme2 == 0.0 ) {
      model->HSMHV_flg_qme = 0 ;
    } else {
      model->HSMHV_flg_qme = 1 ;
      model->HSMHV_qme12 = model->HSMHV_qme1 / ( model->HSMHV_qme2 * model->HSMHV_qme2 ) ;
    }

    for ( here = model->HSMHVinstances; here; here = here->HSMHVnextInstance ) {

      pParam = &here->pParam ;

      hereMKS = &here->hereMKS ;

      here->HSMHV_lgate = Lgate = here->HSMHV_l + model->HSMHV_xl ;
      Wgate = here->HSMHV_w / here->HSMHV_nf  + model->HSMHV_xw ;

      LG = Lgate * C_m2um ;
      here->HSMHV_wg = WG = Wgate * C_m2um ; 
      WL = WG * LG ;
      GDLD = model->HSMHV_gdld * C_m2um ;


      /* Band gap */
      here->HSMHV_egtnom = pParam->HSMHV_eg0 - model->HSMHV_ktnom    
        * ( 90.25e-6 + model->HSMHV_ktnom * 1.0e-7 ) ;
  
      /* C_EOX */
      here->HSMHV_cecox = C_VAC * model->HSMHV_kappa ;
  
      /* Vth reduction for small Vds */
      here->HSMHV_msc = model->HSMHV_scp22  ;

      /* Poly-Si Gate Depletion */
      if ( pParam->HSMHV_pgd1 == 0.0 ) {
        here->HSMHV_flg_pgd = 0 ;
      } else {
        here->HSMHV_flg_pgd = 1 ;
      }


      /* CLM5 & CLM6 */
      here->HSMHV_clmmod = 1e0 + pow( LG , model->HSMHV_clm5 ) * model->HSMHV_clm6 ;

      /* Half length of diffusion */
      T1 = 1.0 / (model->HSMHV_saref + 0.5 * here->HSMHV_l)
         + 1.0 / (model->HSMHV_sbref + 0.5 * here->HSMHV_l);
      Lod_half_ref = 2.0 / T1 ;

      if (here->HSMHV_sa > 0.0 && here->HSMHV_sb > 0.0 &&
	  (here->HSMHV_nf == 1.0 ||
           (here->HSMHV_nf > 1.0 && here->HSMHV_sd > 0.0))) {
        T1 = 0.0;
        for (i = 0; i < here->HSMHV_nf; i++) {
          T1 += 1.0 / (here->HSMHV_sa + 0.5 * here->HSMHV_l
                       + i * (here->HSMHV_sd + here->HSMHV_l))
              + 1.0 / (here->HSMHV_sb + 0.5 * here->HSMHV_l
                       + i * (here->HSMHV_sd + here->HSMHV_l));
        }
        Lod_half = 2.0 * here->HSMHV_nf / T1;
      } else {
        Lod_half = 0.0;
      }

      Npext = pParam->HSMHV_npext ;
      here->HSMHV_mueph1 = pParam->HSMHV_mueph1 ;
      here->HSMHV_nsubp  = pParam->HSMHV_nsubp ;
      here->HSMHV_nsubc  = pParam->HSMHV_nsubc ;

      /* DFM */
      if ( model->HSMHV_codfm == 1 && here->HSMHV_nsubcdfm_Given ) {
	RANGECHECK(here->HSMHV_nsubcdfm,   1.0e16,   1.0e19, "NSUBCDFM") ;
 	here->HSMHV_mueph1 *= model->HSMHV_mphdfm
	  * ( log(hereMKS->HSMHV_nsubcdfm) - log(here->HSMHV_nsubc) ) + 1.0 ;
	here->HSMHV_nsubp += hereMKS->HSMHV_nsubcdfm - here->HSMHV_nsubc ;
	Npext += hereMKS->HSMHV_nsubcdfm - here->HSMHV_nsubc ;
 	here->HSMHV_nsubc = hereMKS->HSMHV_nsubcdfm ;
      }

      /* Phonon Scattering (temperature-independent part) */
      mueph = here->HSMHV_mueph1  
        * (1.0e0 + (model->HSMHV_muephw / pow( WG, model->HSMHV_muepwp))) 
        * (1.0e0 + (model->HSMHV_muephl / pow( LG, model->HSMHV_mueplp))) 
        * (1.0e0 + (model->HSMHV_muephs / pow( WL, model->HSMHV_muepsp)));  
      if (Lod_half > 0.0) {
        T1 = 1.0e0 / (1.0e0 + pParam->HSMHV_muesti2) ;
        T2 = pow (pParam->HSMHV_muesti1 / Lod_half, pParam->HSMHV_muesti3) ;
        T3 = pow (pParam->HSMHV_muesti1 / Lod_half_ref, pParam->HSMHV_muesti3) ;
        here->HSMHV_mueph = mueph * (1.0e0 + T1 * T2) / (1.0e0 + T1 * T3); 
      } else {
        here->HSMHV_mueph = mueph;
      }
      
      /* Surface Roughness Scattering */
      here->HSMHV_muesr = model->HSMHV_muesr0 
        * (1.0e0 + (model->HSMHV_muesrl / pow (LG, model->HSMHV_mueslp))) 
        * (1.0e0 + (model->HSMHV_muesrw / pow (WG, model->HSMHV_mueswp))) ;

      /* Coefficients of Qbm for Eeff */
      T1 = pow( LG, model->HSMHV_ndeplp ) ;
      here->HSMHV_ndep_o_esi = ( pParam->HSMHV_ndep * T1 ) / ( T1 + model->HSMHV_ndepl )
	/ C_ESI ;
      here->HSMHV_ninv_o_esi = pParam->HSMHV_ninv / C_ESI ;
      here->HSMHV_ninvd0 = model->HSMHV_ninvd * ( 1.0 + (model->HSMHV_ninvdw / pow( WG, model->HSMHV_ninvdwp)));

      /* Metallurgical channel geometry */
      dL = model->HSMHV_xld 
        + (modelMKS->HSMHV_ll / pow (Lgate + model->HSMHV_lld, model->HSMHV_lln)) ;
      dLLD = model->HSMHV_xldld 
        + (modelMKS->HSMHV_ll / pow (Lgate + model->HSMHV_lld, model->HSMHV_lln)) ;
   
      dW = model->HSMHV_xwd 
        + (modelMKS->HSMHV_wl / pow (Wgate + model->HSMHV_wld, model->HSMHV_wln)) ;  
      dWLD = model->HSMHV_xwdld 
        + (modelMKS->HSMHV_wl / pow (Wgate + model->HSMHV_wld, model->HSMHV_wln)) ;  
      dWCV = model->HSMHV_xwdc 
        + (modelMKS->HSMHV_wl / pow (Wgate + model->HSMHV_wld, model->HSMHV_wln)) ;  
    
      Leff = Lgate - ( dL + dLLD ) ;
      if ( Leff <= 0.0 ) {   
        IFuid namarr[2];
        namarr[0] = here->HSMHVname;
        namarr[1] = model->HSMHVmodName;
        (*(SPfrontEnd->IFerror))
          ( 
           ERR_FATAL, 
           "HiSIM_HV: MOSFET(%s) MODEL(%s): effective channel length is negative or 0", 
           namarr 
           );
        return (E_BADPARM);
      }
      here->HSMHV_leff = Leff ;

      /* Wg dependence for short channel devices */
      here->HSMHV_lgatesm = Lgate + model->HSMHV_wl1 / pow( WL , model->HSMHV_wl1p ) ;
      here->HSMHV_dVthsm = pParam->HSMHV_wl2 / pow( WL , model->HSMHV_wl2p ) ;

      /* Lg dependence of wsti */
      T1 = 1.0e0 + model->HSMHV_wstil / pow( here->HSMHV_lgatesm * C_m2um  , model->HSMHV_wstilp ) ;
      T2 = 1.0e0 + model->HSMHV_wstiw / pow( WG , model->HSMHV_wstiwp ) ;
      here->HSMHV_wsti = pParam->HSMHV_wsti * T1 * T2 ;

      here->HSMHV_weff = Weff = Wgate - 2.0e0 * dW ;
      here->HSMHV_weff_ld     = Wgate - 2.0e0 * dWLD ;
      here->HSMHV_weff_cv     = Wgate - 2.0e0 * dWCV ;
      if ( Weff <= 0.0 ) {   
        IFuid namarr[2];
        namarr[0] = here->HSMHVname;
        namarr[1] = model->HSMHVmodName;
        (*(SPfrontEnd->IFerror))
          ( 
           ERR_FATAL, 
           "HiSIM_HV: MOSFET(%s) MODEL(%s): effective channel width is negative or 0", 
           namarr 
           );
        return (E_BADPARM);
      }
      here->HSMHV_weff_nf = Weff * here->HSMHV_nf ;
      here->HSMHV_weffcv_nf = here->HSMHV_weff_cv * here->HSMHV_nf ;

      /* Surface impurity profile */
      /* Note: Sign Changed --> */
      Nsubpp = here->HSMHV_nsubp  
        * (1.0e0 + (model->HSMHV_nsubp0 / pow (WG, model->HSMHV_nsubwp))) ;
      /* <-- Note: Sign Changed */

      if (Lod_half > 0.0) {
        T1 = 1.0e0 / (1.0e0 + pParam->HSMHV_nsubpsti2) ;
        T2 = pow (pParam->HSMHV_nsubpsti1 / Lod_half, pParam->HSMHV_nsubpsti3) ;
        T3 = pow (pParam->HSMHV_nsubpsti1 / Lod_half_ref, pParam->HSMHV_nsubpsti3) ;
        Nsubps = Nsubpp * (1.0e0 + T1 * T2) / (1.0e0 + T1 * T3) ;
      } else {
        Nsubps = Nsubpp ;
      }
   
      here->HSMHV_nsubc *= 1.0e0 + ( model->HSMHV_nsubcw / pow ( WG, model->HSMHV_nsubcwp )) ;

      if( Lgate > model->HSMHV_lp ){
        Nsub = (here->HSMHV_nsubc * (Lgate - model->HSMHV_lp) 
                +  Nsubps  * model->HSMHV_lp) / Lgate ;
      } else {
        Nsub = Nsubps
          + (Nsubps - here->HSMHV_nsubc) * (model->HSMHV_lp - Lgate) 
          / model->HSMHV_lp ;
      }
      T3 = 0.5e0 * Lgate - model->HSMHV_lp ;
      T1 = 1.0e0 / ( 1.0e0 / T3 + 1.0e0 / model->HSMHV_lpext ) ;
      T2 = Fn_Max (0.0e0, T1) ;
      here->HSMHV_nsub = 
	Nsub = Nsub + T2 * (Npext - here->HSMHV_nsubc) / Lgate ;
      here->HSMHV_qnsub = q_Nsub  = C_QE * Nsub ;
      here->HSMHV_qnsub_esi = q_Nsub * C_ESI ;
      here->HSMHV_2qnsub_esi = 2.0 * here->HSMHV_qnsub_esi ;

      /* Pocket Overlap (temperature-independent part) */
      if ( Lgate <= 2.0e0 * model->HSMHV_lp ) {
        Nsubb = 2.0e0 * Nsubps 
          - (Nsubps - here->HSMHV_nsubc) * Lgate 
          / model->HSMHV_lp - here->HSMHV_nsubc ;
        here->HSMHV_ptovr0 = log (Nsubb / here->HSMHV_nsubc) ;
        /* here->HSMHV_ptovr0 will be divided by beta later. */
      } else {
        here->HSMHV_ptovr0 = 0.0e0 ;
      }

      /* depletion MOS (temperature-independent part) */
      here->HSMHV_ndepm = modelMKS->HSMHV_ndepm ;

      /* costi0 and costi1 for STI transistor model (temperature-independent part) */
      here->HSMHV_costi00 = sqrt (2.0 * C_QE * pParam->HSMHV_nsti * C_ESI ) ;
      here->HSMHV_nsti_p2 = 1.0 / ( pParam->HSMHV_nsti * pParam->HSMHV_nsti ) ;

      /* Velocity Temperature Dependence (Temperature-dependent part will be multiplied later.) */
      here->HSMHV_vmax0 = (1.0e0 + (pParam->HSMHV_vover / pow (LG, model->HSMHV_voverp)))
        * (1.0e0 + (model->HSMHV_vovers / pow (WL, model->HSMHV_voversp))) ;

      /* 2 phi_B (temperature-independent) */
      /* @300K, with pocket */
      here->HSMHV_pb20 = 2.0e0 / C_b300 * log (Nsub / C_Nin0) ;
      /* @300K, w/o pocket */
      here->HSMHV_pb2c = 2.0e0 / C_b300 * log (here->HSMHV_nsubc / C_Nin0) ;

      /* constant for Poly depletion */
      here->HSMHV_cnstpgd = pow ( 1e0 + 1e0 / LG , model->HSMHV_pgd4 ) 
        * pParam->HSMHV_pgd1 ;

      /* Gate resistance */
      if ( here->HSMHV_corg == 1 ) {
        T1 = here->HSMHV_xgw + Weff / (3.0e0 * here->HSMHV_ngcon);
        T2 = Lgate - here->HSMHV_xgl;
        here->HSMHV_grg = model->HSMHV_rshg * T1 / (here->HSMHV_ngcon * T2 * here->HSMHV_nf);
        if (here->HSMHV_grg > 1.0e-3) here->HSMHV_grg = here->HSMHV_m / here->HSMHV_grg;
        else {
          here->HSMHV_grg = here->HSMHV_m * 1.0e3;
          printf("warning(HiSIM_HV(%s)): The gate conductance reset to 1.0e3 mho.\n",model->HSMHVmodName);
        }
      }

      /* Process source/drain series resistamce */

      if ( model->HSMHV_rsh > 0.0 ) {
        here->HSMHV_rd0 = model->HSMHV_rsh * here->HSMHV_nrd ;
      } else {
	here->HSMHV_rd0 = 0.0 ;
      }
      if ( pParam->HSMHV_rd > 0.0 || pParam->HSMHV_rs > 0.0 ) {
        here->HSMHV_rdtemp0 = 1.0 + model->HSMHV_rds / pow( WL , model->HSMHV_rdsp ) ;
	if( pParam->HSMHV_rdvd != 0.0 ){
	  T7 = ( 1.0 + model->HSMHV_rdvds / pow( WL , model->HSMHV_rdvdsp ) );
          T6 = ( - model->HSMHV_rdvdl * pow( LG , model->HSMHV_rdvdlp ) ) ;
          if(T6 > large_arg) T6 = large_arg ;  
          T6 = exp( T6 ) ;
          here->HSMHV_rdvdtemp0 = T6 * T7 ;
        }
      }
      if( pParam->HSMHV_rd23 != 0.0 ){
	T2 = ( 1.0 + model->HSMHV_rd23s / pow( WL , model->HSMHV_rd23sp ) );
        T1 = ( - model->HSMHV_rd23l * pow( LG , model->HSMHV_rd23lp ) ) ;
        if(T1 > large_arg)  T1 = large_arg ; 
        T1 = exp( T1 ) ;
        T3 = pParam->HSMHV_rd23 * T2 * T1 ;
        here->HSMHV_rd23 = 0.5 * ( T3 + sqrt ( T3 * T3 + 4.0 * dlt_rd23 * dlt_rd23 ) ) ;
      } else {
	here->HSMHV_rd23 = 0.0 ;
      }
      if ( model->HSMHV_rsh > 0.0 ) {
        here->HSMHV_rs0 = model->HSMHV_rsh * here->HSMHV_nrs ;
      } else {
	here->HSMHV_rs0 = 0.0 ;
      }

      here->HSMHV_Xmax  = sqrt ( model->HSMHV_rdrdjunc * model->HSMHV_rdrdjunc + model->HSMHV_xldld * model->HSMHV_xldld ) ;
      if(pParam->HSMHV_nover != 0.0) {
      here->HSMHV_kdep  = 2.0 * C_ESI / ( C_QE * pParam->HSMHV_nover ) ;
      here->HSMHV_kjunc = 2.0 * C_ESI / C_QE * here->HSMHV_nsubc / ( pParam->HSMHV_nover + here->HSMHV_nsubc ) / pParam->HSMHV_nover ;
      }
      here->HSMHV_rdrcxw   = 1.0e0 ;
      here->HSMHV_rdrvmaxw = 1.0e0 + (model->HSMHV_rdrvmaxw / pow( WG, model->HSMHV_rdrvmaxwp)) ;
      here->HSMHV_rdrvmaxl = 1.0e0 + (model->HSMHV_rdrvmaxl / pow( LG, model->HSMHV_rdrvmaxlp)) ;
      here->HSMHV_rdrmuel  = 1.0e0 + (model->HSMHV_rdrmuel  / pow( LG, model->HSMHV_rdrmuelp )) ;

      /* Body resistance */
      if ( here->HSMHV_corbnet == 1 ) {
        if (here->HSMHV_rbpb < 1.0e-3) here->HSMHV_grbpb = here->HSMHV_m * 1.0e3 ;
        else here->HSMHV_grbpb = here->HSMHV_m * ( model->HSMHV_gbmin + 1.0 / here->HSMHV_rbpb ) ;

        if (here->HSMHV_rbps < 1.0e-3) here->HSMHV_grbps = here->HSMHV_m * 1.0e3 ;
        else here->HSMHV_grbps = here->HSMHV_m * ( model->HSMHV_gbmin + 1.0 / here->HSMHV_rbps ) ;

        if (here->HSMHV_rbpd < 1.0e-3) here->HSMHV_grbpd = here->HSMHV_m * 1.0e3 ;
        else here->HSMHV_grbpd = here->HSMHV_m * ( model->HSMHV_gbmin + 1.0 / here->HSMHV_rbpd ) ;
      }

      /* Vdseff */
      if( model->HSMHV_coddlt == 0) {
        T1 = model->HSMHV_ddltslp * LG + model->HSMHV_ddltict ;
        if ( T1 < 0.0 ) { T1 = 0.0 ; }
        here->HSMHV_ddlt = T1 * model->HSMHV_ddltmax / ( T1 + model->HSMHV_ddltmax ) + 1.0 ;
      } else {
        T1 = model->HSMHV_ddltslp * LG  ;
        if ( T1 < 0.0 ) { T1 = 0.0 ; }
        here->HSMHV_ddlt = T1 * model->HSMHV_ddltmax / ( T1 + model->HSMHV_ddltmax ) +  model->HSMHV_ddltict + small ;
      }


      /* Isub */
      T2 = pow( Weff , model->HSMHV_svgswp ) ;
      here->HSMHV_vg2const = pParam->HSMHV_svgs
         * ( 1.0e0
           + modelMKS->HSMHV_svgsl / pow( here->HSMHV_lgate , model->HSMHV_svgslp ) )
         * ( T2 / ( T2 + modelMKS->HSMHV_svgsw ) ) ; 

      here->HSMHV_xvbs = pParam->HSMHV_svbs 
         * ( 1.0e0
           + modelMKS->HSMHV_svbsl / pow( here->HSMHV_lgate , model->HSMHV_svbslp ) ) ;
      here->HSMHV_xgate = modelMKS->HSMHV_slg  
         * ( 1.0
         + modelMKS->HSMHV_slgl / pow( here->HSMHV_lgate , model->HSMHV_slglp ) ) ;

      here->HSMHV_xsub1 = pParam->HSMHV_sub1 
         * ( 1.0 
         + modelMKS->HSMHV_sub1l / pow( here->HSMHV_lgate , model->HSMHV_sub1lp ) ) ;

      here->HSMHV_xsub2 = pParam->HSMHV_sub2
         * ( 1.0 + modelMKS->HSMHV_sub2l / here->HSMHV_lgate ) ;

      here->HSMHV_subld1 = model->HSMHV_subld1
         * ( 1.0 + model->HSMHV_subld1l / pow( LG , model->HSMHV_subld1lp ) ) ;

      /* IBPC */
      here->HSMHV_ibpc1  = pParam->HSMHV_ibpc1
         * ( 1.0 + model->HSMHV_ibpc1l  / pow( LG , model->HSMHV_ibpc1lp ) ) ;

      /* Fringing capacitance */
      here->HSMHV_cfrng = C_EOX / ( C_Pi / 2.0e0 ) * here->HSMHV_weff_nf
         * log( 1.0e0 + model->HSMHV_tpoly / model->HSMHV_tox ) ;

      /* Additional term of lateral-field-induced capacitance */
      here->HSMHV_cqyb0 = C_m2um * here->HSMHV_weff_nf
	* model->HSMHV_xqy1 / pow( LG , model->HSMHV_xqy2 ) ;

      /* Parasitic component of the channel current */
      here->HSMHV_ptl0 = model->HSMHV_ptl * pow( LG        , - model->HSMHV_ptlp ) ;
      here->HSMHV_pt40 = model->HSMHV_pt4 * pow( LG        , - model->HSMHV_pt4p ) ;
      here->HSMHV_gdl0 = model->HSMHV_gdl * pow( LG + GDLD , - model->HSMHV_gdlp ) ;

      /* Self heating */
      pParam->HSMHV_rth = pParam->HSMHV_rth0 / ( here->HSMHV_m * here->HSMHV_weff_nf )
	* ( 1.0 + model->HSMHV_rth0w / pow( WG , model->HSMHV_rth0wp ) );
      pParam->HSMHV_cth = modelMKS->HSMHV_cth0 * ( here->HSMHV_m * here->HSMHV_weff_nf ) ;

      pParam->HSMHV_rth *= ( 1.0 / pow( here->HSMHV_nf , model->HSMHV_rth0nf ) ) ;
      
      here->HSMHV_rthtemp0 = 1.0 / pow( here->HSMHV_nf , model->HSMHV_rth0nf ) / ( here->HSMHV_m * here->HSMHV_weff_nf )
       * ( 1.0 + model->HSMHV_rth0w / pow( WG , model->HSMHV_rth0wp ) );


      /*-----------------------------------------------------------*
       * Temperature dependent constants. 
       *-----------------*/
      if ( here->HSMHVtempNode < 0 || pParam->HSMHV_rth0 == 0.0 ) {

#include "hsmhvtemp_eval.h"
#include "hsmhvtemp_eval_rdri.h"
#include "hsmhvtemp_eval_dio.h"

      } /* end of if ( here->HSMHVtempNode < 0 || pParam->HSMHV_rth0 == 0.0 ) */

    }
  }
  return(OK);
}
