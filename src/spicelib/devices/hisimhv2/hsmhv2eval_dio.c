/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 2  SUBVERSION : 2  REVISION : 0 ) 
 Model Parameter 'VERSION' : 2.20
 FILE : hsmhveval_dio.c

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

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
 * Substrate-source/drain junction diode
 *++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/ 

/*===========================================================*
* Preamble
*=================*/
/*---------------------------------------------------*
* Header files
*-----------------*/
#include "ngspice/ngspice.h"

/*-----------------------------------*
* HiSIM macros
*-----------------*/
#include "hisimhv2.h"
#include "hsmhv2evalenv.h"

/*===========================================================*
* Function HSMHV2dio
*=================*/
int HSMHV2dio
(
 double        vbs_jct,
 double        vbd_jct,
 double        deltemp,
 HSMHV2instance *here,
 HSMHV2model    *model,
 CKTcircuit   *ckt
 )
{
  HSMHV2binningParam *pParam = &here->pParam ;

  /* junction currents */
  double Ibs =0.0, Gbs =0.0, Ibs_dT =0.0 ;
  double Ibd =0.0, Gbd =0.0, Ibd_dT =0.0 ;

  /* junction capacitances */
  double Qbs =0.0, Capbs =0.0, Qbs_dT =0.0 ;
  double Qbd =0.0, Capbd =0.0, Qbd_dT =0.0 ;
  double czbd =0.0,    czbd_dT=0.0 ;
  double czbdsw =0.0,  czbdsw_dT=0.0 ;
  double czbdswg =0.0, czbdswg_dT=0.0 ;
  double czbs =0.0,    czbs_dT=0.0 ;
  double czbssw =0.0,  czbssw_dT=0.0 ;
  double czbsswg =0.0, czbsswg_dT=0.0 ;
  double arg =0.0,     sarg =0.0 ;

  /* temperature-dependent variables for SHE model */
  double log_Tratio =0.0 ;
  double TTEMP =0.0, TTEMP0 =0.0 ;
  double beta =0.0,     beta_dT =0.0 ;
  double beta_inv =0.0, beta_inv_dT =0.0 ;
  double Eg =0.0,    Eg_dT =0.0 ;
  double js =0.0,    js_dT =0.0 ;
  double jssw =0.0,  jssw_dT =0.0 ;
  double js2 =0.0,   js2_dT =0.0 ;
  double jssw2 =0.0, jssw2_dT =0.0 ;

  double isbd_dT =0.0,      isbs_dT =0.0 ;
  double isbd2_dT =0.0,     isbs2_dT =0.0 ;
  double vbdt_dT =0.0,      vbst_dT = 0.0 ;
  double jd_expcd_dT =0.0 , jd_expcs_dT =0.0 ;
  double jd_nvtm_invd_dT =0.0 , jd_nvtm_invs_dT =0.0 ;
  double exptempd_dT = 0.0 , exptemps_dT = 0.0 ;
  double tcjbd =0.0,    tcjbs =0.0,
         tcjbdsw =0.0,  tcjbssw =0.0,
         tcjbdswg =0.0, tcjbsswg =0.0 ;

  /* options */
  double Mfactor = here->HSMHV2_m;

  /* Internal flags  --------------------*/
  int flg_err = 0;   /* error level */
  int flg_info = model->HSMHV2_info;

  /* temporary vars. & derivatives */
  double TX =0.0 ;
  double T0 =0.0, T0_dT =0.0 ;
  double T1 =0.0, T1_dVb =0.0, T1_dT =0.0 ;
  double T2 =0.0, T2_dVb =0.0, T2_dT =0.0 ;
  double T3 =0.0, T3_dVb =0.0, T3_dT =0.0 ;
  double T4 =0.0, T4_dT =0.0 ;
  double T9 =0.0, T9_dT =0.0 ;
  double T10 =0.0, T10_dT =0.0 ;
  double T12 =0.0,                                           T12_dT =0.0 ;


  /*================ Start of executable code.=================*/

  /*-----------------------------------------------------------*
   * Temperature dependent constants. 
   *-----------------*/
  if ( here->HSMHV2tempNode > 0 && pParam->HSMHV2_rth0 != 0.0 ) {

#define HSMHV2EVAL
#include "hsmhv2temp_eval_dio.h"

  } else {
    TTEMP = ckt->CKTtemp;
    if ( here->HSMHV2_dtemp_Given ) { TTEMP = TTEMP + here->HSMHV2_dtemp ; }
  }


  /*-----------------------------------------------------------*
   * Cbsj, Cbdj: node-base S/D biases.
   *-----------------*/

  /* ibd */
  T10 = model->HSMHV2_cvbd * here->HSMHV2_jd_nvtm_invd ;
  T10_dT = model->HSMHV2_cvbd * beta_dT / pParam->HSMHV2_njd ;

  T9 = model->HSMHV2_cisbd * here->HSMHV2_exptempd ;
  T9_dT = model->HSMHV2_cisbd * exptempd_dT ;
  T0 = here->HSMHV2_isbd2 * T9 ;
  T0_dT = here->HSMHV2_isbd2 * T9_dT + isbd2_dT * T9 ;

  TX = - vbd_jct * T10 ;
  T2 = exp ( TX );
  T2_dVb = - T2 * T10 ;
  T2_dT = T2 * TX * beta_dT * beta_inv ;

  T3 = T2 ;
  T3_dVb = T2_dVb ;
  T3_dT = T2_dT ;

  if ( vbd_jct < here->HSMHV2_vbdt ) {
    TX = vbd_jct * here->HSMHV2_jd_nvtm_invd ;

    if ( TX < - 3*EXP_THR ) {
      T1 = 0.0 ;
      T1_dVb = 0.0 ;
      T1_dT =  0.0 ;
    } else {
      T1 = exp ( TX ) ;
      T1_dVb = T1 * here->HSMHV2_jd_nvtm_invd ;
      T1_dT = T1 * TX * beta_dT * beta_inv ;
    }

    Ibd = here->HSMHV2_isbd * (T1 - 1.0) 
      + T0 * (T2 - 1.0) 
      + pParam->HSMHV2_cisbkd * (T3 - 1.0);   
    Gbd = here->HSMHV2_isbd * T1_dVb 
      + T0 * T2_dVb 
      + pParam->HSMHV2_cisbkd * T3_dVb ;
    Ibd_dT = here->HSMHV2_isbd * T1_dT + isbd_dT * ( T1 - 1.0 )
      + T0 * T2_dT + T0_dT * ( T2 - 1.0 )
      + pParam->HSMHV2_cisbkd * T3_dT ;

  } else {
    T1 = here->HSMHV2_jd_expcd ;

    T4 = here->HSMHV2_isbd * here->HSMHV2_jd_nvtm_invd  * T1 ;

    Ibd = here->HSMHV2_isbd * (T1 - 1.0) 
      + T4 * (vbd_jct - here->HSMHV2_vbdt) 
      + T0 * (T2 - 1.0)
      + pParam->HSMHV2_cisbkd * (T3 - 1.0) ;
    Gbd = T4 
      + T0 * T2_dVb 
      + pParam->HSMHV2_cisbkd * T3_dVb ;

    T1_dT = jd_expcd_dT ;
    T4_dT = isbd_dT * here->HSMHV2_jd_nvtm_invd * T1
      + here->HSMHV2_isbd * jd_nvtm_invd_dT * T1
      + here->HSMHV2_isbd * here->HSMHV2_jd_nvtm_invd * T1_dT ;
    Ibd_dT = isbd_dT * ( T1 - 1.0 ) + here->HSMHV2_isbd * T1_dT
      + T4_dT * ( vbd_jct - here->HSMHV2_vbdt ) - T4 * vbdt_dT
      + T0_dT * ( T2 - 1.0 ) + T0 * T2_dT
      + pParam->HSMHV2_cisbkd * T3_dT ;
  }  
  T12 = model->HSMHV2_divxd * here->HSMHV2_isbd2 ;
  Ibd += T12 * vbd_jct ;
  Gbd += T12 ;

  T12_dT = model->HSMHV2_divxd * isbd2_dT ;
  Ibd_dT += T12_dT * vbd_jct ;

  /* ibs */
  T10 = model->HSMHV2_cvbs * here->HSMHV2_jd_nvtm_invs ;
  T10_dT = model->HSMHV2_cvbs * beta_dT / pParam->HSMHV2_njs ;

  T9 = model->HSMHV2_cisbs * here->HSMHV2_exptemps ;
  T9_dT = model->HSMHV2_cisbs * exptemps_dT ;
  T0 = here->HSMHV2_isbs2 * T9 ;
  T0_dT = here->HSMHV2_isbs2 * T9_dT + isbs2_dT * T9 ;

  TX = - vbs_jct * T10 ;
  T2 = exp ( TX );
  T2_dVb = - T2 * T10 ;
  T2_dT = T2 * TX * beta_dT * beta_inv ;

  T3 = T2 ;
  T3_dVb = T2_dVb ;
  T3_dT = T2_dT ;

  if ( vbs_jct < here->HSMHV2_vbst ) {
    TX = vbs_jct * here->HSMHV2_jd_nvtm_invs ;
    if ( TX < - 3*EXP_THR ) {
      T1 = 0.0 ;
      T1_dVb = 0.0 ;
      T1_dT =  0.0 ;
    } else {
      T1 = exp ( TX ) ;
      T1_dVb = T1 * here->HSMHV2_jd_nvtm_invs ;
      T1_dT = T1 * TX * beta_dT * beta_inv ;
    }
    Ibs = here->HSMHV2_isbs * (T1 - 1.0) 
      + T0 * (T2 - 1.0) 
      + pParam->HSMHV2_cisbks * (T3 - 1.0);
    Gbs = here->HSMHV2_isbs * T1_dVb 
      + T0 * T2_dVb
      + pParam->HSMHV2_cisbks * T3_dVb ;
    Ibs_dT = here->HSMHV2_isbs * T1_dT + isbs_dT * ( T1 - 1.0 )
      + T0 * T2_dT + T0_dT * ( T2 - 1.0 )
      + pParam->HSMHV2_cisbks * T3_dT ;
  } else {
    T1 = here->HSMHV2_jd_expcs ;

    T4 = here->HSMHV2_isbs * here->HSMHV2_jd_nvtm_invs  * T1 ;

    Ibs = here->HSMHV2_isbs * (T1 - 1.0)
      + T4 * (vbs_jct - here->HSMHV2_vbst)
      + T0 * (T2 - 1.0) 
      + pParam->HSMHV2_cisbks * (T3 - 1.0) ;
    Gbs = T4 
      + T0 * T2_dVb 
      + pParam->HSMHV2_cisbks * T3_dVb ;

    T1_dT = jd_expcs_dT ;
    T4_dT = isbs_dT * here->HSMHV2_jd_nvtm_invs * T1
      + here->HSMHV2_isbs * jd_nvtm_invs_dT * T1
      + here->HSMHV2_isbs * here->HSMHV2_jd_nvtm_invs * T1_dT ;
    Ibs_dT = isbs_dT * ( T1 - 1.0 ) + here->HSMHV2_isbs * T1_dT
      + T4_dT * ( vbs_jct - here->HSMHV2_vbst) - T4 * vbst_dT
      + T0_dT * ( T2 - 1.0 ) + T0 * T2_dT
      + pParam->HSMHV2_cisbks * T3_dT ;
  }
  T12 = model->HSMHV2_divxs * here->HSMHV2_isbs2 ;
  Ibs += T12 * vbs_jct ;
  Gbs += T12 ;

  T12_dT = model->HSMHV2_divxs * isbs2_dT ;
  Ibs_dT += T12_dT * vbs_jct ;


  /*-----------------------------------------------------------*
   * Charges and Capacitances.
   *-----------------*/
  /*  charge storage elements
   *  bulk-drain and bulk-source depletion capacitances
   *  czbd : zero bias drain junction capacitance
   *  czbs : zero bias source junction capacitance
   *  czbdsw:zero bias drain junction sidewall capacitance
   *  czbssw:zero bias source junction sidewall capacitance
   */

  /* Source Bulk Junction */
  tcjbs = model->HSMHV2_tcjbs ;
  tcjbssw = model->HSMHV2_tcjbssw ;
  tcjbsswg = model->HSMHV2_tcjbsswg ;

  czbs = model->HSMHV2_cjs * here->HSMHV2_as ;
  czbs = czbs * ( 1.0 + tcjbs * ( TTEMP - model->HSMHV2_ktnom )) ;
  czbs_dT = ( model->HSMHV2_cjs * here->HSMHV2_as ) * tcjbs ;

  if (here->HSMHV2_ps > here->HSMHV2_weff_nf) {
    czbssw = model->HSMHV2_cjsws * ( here->HSMHV2_ps - here->HSMHV2_weff_nf ) ;
    czbssw = czbssw * ( 1.0 + tcjbssw * ( TTEMP - model->HSMHV2_ktnom )) ;
    czbssw_dT = ( model->HSMHV2_cjsws * ( here->HSMHV2_ps - here->HSMHV2_weff_nf )) * tcjbssw ;

    czbsswg = model->HSMHV2_cjswgs * here->HSMHV2_weff_nf ;
    czbsswg = czbsswg * ( 1.0 + tcjbsswg * ( TTEMP - model->HSMHV2_ktnom )) ;
    czbsswg_dT = ( model->HSMHV2_cjswgs * here->HSMHV2_weff_nf ) * tcjbsswg ;

//  if (vbs_jct == 0.0) {  
    if (0) {  
      Qbs = 0.0 ;
      Qbs_dT = 0.0 ;
      Capbs = czbs + czbssw + czbsswg ;
    } else if (vbs_jct < 0.0) { 
      if (czbs > 0.0) {
        arg = 1.0 - vbs_jct / model->HSMHV2_pbs ;
        if (model->HSMHV2_mjs == 0.5) 
          sarg = 1.0 / sqrt(arg) ;
        else 
          sarg = Fn_Pow( arg , -model->HSMHV2_mjs ) ;
        Qbs = model->HSMHV2_pbs * czbs * (1.0 - arg * sarg) / (1.0 - model->HSMHV2_mjs) ;
        Qbs_dT = model->HSMHV2_pbs * czbs_dT * (1.0 - arg * sarg) / (1.0 - model->HSMHV2_mjs) ;
        Capbs = czbs * sarg ;
      } else {
        Qbs = 0.0 ;
        Qbs_dT = 0.0 ;
        Capbs = 0.0 ;
      }
      if (czbssw > 0.0) {
        arg = 1.0 - vbs_jct / model->HSMHV2_pbsws ;
        if (model->HSMHV2_mjsws == 0.5) 
          sarg = 1.0 / sqrt(arg) ;
        else 
          sarg = Fn_Pow( arg , -model->HSMHV2_mjsws ) ;
        Qbs += model->HSMHV2_pbsws * czbssw * (1.0 - arg * sarg) / (1.0 - model->HSMHV2_mjsws) ;
        Qbs_dT += model->HSMHV2_pbsws * czbssw_dT * (1.0 - arg * sarg) / (1.0 - model->HSMHV2_mjsws) ;
        Capbs += czbssw * sarg ;
      }
      if (czbsswg > 0.0) {
        arg = 1.0 - vbs_jct / model->HSMHV2_pbswgs ;
        if (model->HSMHV2_mjswgs == 0.5)
          sarg = 1.0 / sqrt(arg) ;
        else
          sarg = Fn_Pow( arg , -model->HSMHV2_mjswgs ) ;
        Qbs += model->HSMHV2_pbswgs * czbsswg * (1.0 - arg * sarg) / (1.0 - model->HSMHV2_mjswgs) ;
        Qbs_dT += model->HSMHV2_pbswgs * czbsswg_dT * (1.0 - arg * sarg) / (1.0 - model->HSMHV2_mjswgs) ;
        Capbs += czbsswg * sarg ;
      }
    } else {
      T1 = czbs + czbssw + czbsswg ;
      T1_dT = czbs_dT + czbssw_dT + czbsswg_dT ;
      T2 = czbs * model->HSMHV2_mjs / model->HSMHV2_pbs  
        + czbssw * model->HSMHV2_mjsws / model->HSMHV2_pbsws  
        + czbsswg * model->HSMHV2_mjswgs / model->HSMHV2_pbswgs ;
      T2_dT = czbs_dT * model->HSMHV2_mjs / model->HSMHV2_pbs  
        + czbssw_dT * model->HSMHV2_mjsws / model->HSMHV2_pbsws  
        + czbsswg_dT * model->HSMHV2_mjswgs / model->HSMHV2_pbswgs ;
      Qbs = vbs_jct * (T1 + vbs_jct * 0.5 * T2) ;
      Qbs_dT = vbs_jct * (T1_dT + vbs_jct * 0.5 * T2_dT) ;
      Capbs = T1 + vbs_jct * T2 ;
    }
  } else {
    czbsswg = model->HSMHV2_cjswgs * here->HSMHV2_ps ;
    czbsswg = czbsswg * ( 1.0 + tcjbsswg * ( TTEMP - model->HSMHV2_ktnom )) ;
    czbsswg_dT = ( model->HSMHV2_cjswgs * here->HSMHV2_ps ) * tcjbsswg ;

//  if (vbs_jct == 0.0) {
    if (0) {
      Qbs = 0.0 ;
      Qbs_dT = 0.0 ;
      Capbs = czbs + czbsswg ;
    } else if (vbs_jct < 0.0) {
      if (czbs > 0.0) {
        arg = 1.0 - vbs_jct / model->HSMHV2_pbs ;
        if (model->HSMHV2_mjs == 0.5)
          sarg = 1.0 / sqrt(arg) ;
        else
          sarg = Fn_Pow( arg , -model->HSMHV2_mjs ) ;
        Qbs = model->HSMHV2_pbs * czbs * (1.0 - arg * sarg) / (1.0 - model->HSMHV2_mjs) ;
        Qbs_dT = model->HSMHV2_pbs * czbs_dT * (1.0 - arg * sarg) / (1.0 - model->HSMHV2_mjs) ;
        Capbs = czbs * sarg ;
      } else {
        Qbs = 0.0 ;
        Qbs_dT = 0.0 ;
        Capbs = 0.0 ;
      }
      if (czbsswg > 0.0) {
        arg = 1.0 - vbs_jct / model->HSMHV2_pbswgs ;
        if (model->HSMHV2_mjswgs == 0.5)
          sarg = 1.0 / sqrt(arg) ;
        else
          sarg = Fn_Pow( arg , -model->HSMHV2_mjswgs ) ;
        Qbs += model->HSMHV2_pbswgs * czbsswg * (1.0 - arg * sarg) / (1.0 - model->HSMHV2_mjswgs) ;
        Qbs_dT += model->HSMHV2_pbswgs * czbsswg_dT * (1.0 - arg * sarg) / (1.0 - model->HSMHV2_mjswgs) ;
        Capbs += czbsswg * sarg ;
      }
    } else {
      T1 = czbs + czbsswg ;
      T1_dT = czbs_dT + czbsswg_dT ;
      T2 = czbs * model->HSMHV2_mjs / model->HSMHV2_pbs 
        + czbsswg * model->HSMHV2_mjswgs / model->HSMHV2_pbswgs ;
      T2_dT = czbs_dT * model->HSMHV2_mjs / model->HSMHV2_pbs 
        + czbsswg_dT * model->HSMHV2_mjswgs / model->HSMHV2_pbswgs ;
      Qbs = vbs_jct * (T1 + vbs_jct * 0.5 * T2) ;
      Qbs_dT = vbs_jct * (T1_dT + vbs_jct * 0.5 * T2_dT) ;
      Capbs = T1 + vbs_jct * T2 ;
    }
  }    
    
  /* Drain Bulk Junction */
  tcjbd = model->HSMHV2_tcjbd ;
  tcjbdsw = model->HSMHV2_tcjbdsw ;
  tcjbdswg = model->HSMHV2_tcjbdswg ;

  czbd = model->HSMHV2_cjd * here->HSMHV2_ad ;
  czbd = czbd * ( 1.0 + tcjbd * ( TTEMP - model->HSMHV2_ktnom )) ;
  czbd_dT = ( model->HSMHV2_cjd * here->HSMHV2_ad ) * tcjbd ;

  if (here->HSMHV2_pd > here->HSMHV2_weff_nf) {

    czbdsw = model->HSMHV2_cjswd * ( here->HSMHV2_pd - here->HSMHV2_weff_nf ) ;
    czbdsw = czbdsw * ( 1.0 + tcjbdsw * ( TTEMP - model->HSMHV2_ktnom )) ;
    czbdsw_dT = ( model->HSMHV2_cjswd * ( here->HSMHV2_pd - here->HSMHV2_weff_nf )) * tcjbdsw ;

    czbdswg = model->HSMHV2_cjswgd * here->HSMHV2_weff_nf ;
    czbdswg = czbdswg * ( 1.0 + tcjbdswg * ( TTEMP - model->HSMHV2_ktnom )) ;
    czbdswg_dT = ( model->HSMHV2_cjswgd * here->HSMHV2_weff_nf ) * tcjbdswg ;

//  if (vbd_jct == 0.0) {
    if (0) {
      Qbd = 0.0 ;
      Qbd_dT = 0.0 ;
      Capbd = czbd + czbdsw + czbdswg ;
    } else if (vbd_jct < 0.0) {
      if (czbd > 0.0) {
        arg = 1.0 - vbd_jct / model->HSMHV2_pbd ;
        if (model->HSMHV2_mjd == 0.5)
          sarg = 1.0 / sqrt(arg) ;
        else
          sarg = Fn_Pow( arg , -model->HSMHV2_mjd ) ;
        Qbd = model->HSMHV2_pbd * czbd * (1.0 - arg * sarg) / (1.0 - model->HSMHV2_mjd) ;
        Qbd_dT = model->HSMHV2_pbd * czbd_dT * (1.0 - arg * sarg) / (1.0 - model->HSMHV2_mjd) ;
        Capbd = czbd * sarg ;
      } else {
        Qbd = 0.0 ;
        Qbd_dT = 0.0 ;
        Capbd = 0.0 ;
      }
      if (czbdsw > 0.0) {
        arg = 1.0 - vbd_jct / model->HSMHV2_pbswd ;
        if (model->HSMHV2_mjswd == 0.5)
          sarg = 1.0 / sqrt(arg) ;
        else
          sarg = Fn_Pow( arg , -model->HSMHV2_mjswd ) ;
        Qbd += model->HSMHV2_pbswd * czbdsw * (1.0 - arg * sarg) / (1.0 - model->HSMHV2_mjswd) ;
        Qbd_dT += model->HSMHV2_pbswd * czbdsw_dT * (1.0 - arg * sarg) / (1.0 - model->HSMHV2_mjswd) ;
        Capbd += czbdsw * sarg ;
      }
      if (czbdswg > 0.0) {
        arg = 1.0 - vbd_jct / model->HSMHV2_pbswgd ;
        if (model->HSMHV2_mjswgd == 0.5)
          sarg = 1.0 / sqrt(arg) ;
        else
          sarg = Fn_Pow( arg , -model->HSMHV2_mjswgd ) ;
        Qbd += model->HSMHV2_pbswgd * czbdswg * (1.0 - arg * sarg) / (1.0 - model->HSMHV2_mjswgd) ;
        Qbd_dT += model->HSMHV2_pbswgd * czbdswg_dT * (1.0 - arg * sarg) / (1.0 - model->HSMHV2_mjswgd) ;
        Capbd += czbdswg * sarg ;
      
      }
    } else {
      T1 = czbd + czbdsw + czbdswg ;
      T1_dT = czbd_dT + czbdsw_dT + czbdswg_dT ;
      T2 = czbd * model->HSMHV2_mjd / model->HSMHV2_pbd 
        + czbdsw * model->HSMHV2_mjswd / model->HSMHV2_pbswd  
        + czbdswg * model->HSMHV2_mjswgd / model->HSMHV2_pbswgd ;
      T2_dT = czbd_dT * model->HSMHV2_mjd / model->HSMHV2_pbd 
        + czbdsw_dT * model->HSMHV2_mjswd / model->HSMHV2_pbswd  
        + czbdswg_dT * model->HSMHV2_mjswgd / model->HSMHV2_pbswgd ;
      Qbd = vbd_jct * (T1 + vbd_jct * 0.5 * T2) ;
      Qbd_dT = vbd_jct * (T1_dT + vbd_jct * 0.5 * T2_dT) ;
      Capbd = T1 + vbd_jct * T2 ;
    }
    
  } else {
    czbdswg = model->HSMHV2_cjswgd * here->HSMHV2_pd ;
    czbdswg = czbdswg * ( 1.0 + tcjbdswg * ( TTEMP - model->HSMHV2_ktnom )) ;
    czbdswg_dT = ( model->HSMHV2_cjswgd * here->HSMHV2_pd ) * tcjbdswg ;

//  if (vbd_jct == 0.0) {   
    if (0) {   
      Qbd = 0.0 ;
      Qbd_dT = 0.0 ;
      Capbd = czbd + czbdswg ;
    } else if (vbd_jct < 0.0) {
      if (czbd > 0.0) {
        arg = 1.0 - vbd_jct / model->HSMHV2_pbd ;
        if (model->HSMHV2_mjd == 0.5)
          sarg = 1.0 / sqrt(arg) ;
	else
          sarg = Fn_Pow( arg , -model->HSMHV2_mjd ) ;
        Qbd = model->HSMHV2_pbd * czbd * (1.0 - arg * sarg) / (1.0 - model->HSMHV2_mjd) ;
        Qbd_dT = model->HSMHV2_pbd * czbd_dT * (1.0 - arg * sarg) / (1.0 - model->HSMHV2_mjd) ;
        Capbd = czbd * sarg ;
      } else {
        Qbd = 0.0 ;
        Qbd_dT = 0.0 ;
        Capbd = 0.0 ;
      }
      if (czbdswg > 0.0) {
        arg = 1.0 - vbd_jct / model->HSMHV2_pbswgd ;
        if (model->HSMHV2_mjswgd == 0.5)
          sarg = 1.0 / sqrt(arg) ;
        else
          sarg = Fn_Pow( arg , -model->HSMHV2_mjswgd ) ;
        Qbd += model->HSMHV2_pbswgd * czbdswg * (1.0 - arg * sarg) / (1.0 - model->HSMHV2_mjswgd) ;
        Qbd_dT += model->HSMHV2_pbswgd * czbdswg_dT * (1.0 - arg * sarg) / (1.0 - model->HSMHV2_mjswgd) ;
        Capbd += czbdswg * sarg ;
      }
    } else {
      T1 = czbd + czbdswg ;
      T1_dT = czbd_dT + czbdswg_dT ;
      T2 = czbd * model->HSMHV2_mjd / model->HSMHV2_pbd 
        + czbdswg * model->HSMHV2_mjswgd / model->HSMHV2_pbswgd ;
      T2_dT = czbd_dT * model->HSMHV2_mjd / model->HSMHV2_pbd 
        + czbdswg_dT * model->HSMHV2_mjswgd / model->HSMHV2_pbswgd ;
      Qbd = vbd_jct * (T1 + vbd_jct * 0.5 * T2) ;
      Qbd_dT = vbd_jct * (T1_dT + vbd_jct * 0.5 * T2_dT) ;
      Capbd = T1 + vbd_jct * T2 ;
    }
  }


  /*---------------------------------------------------* 
   * Junction diode.
   *-----------------*/ 
  here->HSMHV2_ibs = Mfactor * Ibs ;
  here->HSMHV2_ibd = Mfactor * Ibd ;
  here->HSMHV2_gbs = Mfactor * Gbs ;
  here->HSMHV2_gbd = Mfactor * Gbd ;
  *(ckt->CKTstate0 + here->HSMHV2qbs) = Mfactor * Qbs ;
  *(ckt->CKTstate0 + here->HSMHV2qbd) = Mfactor * Qbd ;
  here->HSMHV2_capbs = Mfactor * Capbs ;
  here->HSMHV2_capbd = Mfactor * Capbd ;

  here->HSMHV2_gbdT = Mfactor * Ibd_dT ;
  here->HSMHV2_gbsT = Mfactor * Ibs_dT ;
  here->HSMHV2_gcbdT = Mfactor * Qbd_dT ;
  here->HSMHV2_gcbsT = Mfactor * Qbs_dT ;

  /*-----------------------------------------------------------* 
   * Warn floating-point exceptions.
   * - Function finite() in libm is called.
   *-----------------*/
  T1 = here->HSMHV2_ibs + here->HSMHV2_ibd + here->HSMHV2_gbs + here->HSMHV2_gbd;
  T1 = T1 + *(ckt->CKTstate0 + here->HSMHV2qbs)
          + *(ckt->CKTstate0 + here->HSMHV2qbd)
          + here->HSMHV2_capbs
          + here->HSMHV2_capbd;
  if ( ! finite (T1) ) {
    flg_err = 1 ;
    fprintf(stderr ,
            "*** warning(HiSIM_HV(%s)): FP-exception (junction diode)\n",model->HSMHV2modName) ;
    if ( flg_info >= 1 ) {
      printf ("*** warning(HiSIM_HV(%s)): FP-exception\n",model->HSMHV2modName) ;
    }
  }

  /*-----------------------------------------------------------* 
   * End of HSMHV2eval_dio
   *-----------------*/ 

  return ( HiSIM_OK ) ;
  
}
