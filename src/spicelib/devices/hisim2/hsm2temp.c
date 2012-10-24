/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 VERSION : HiSIM 2.6.1 
 FILE : hsm2temp.c

 date : 2012.4.6

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "hsm2def.h"
#include "hsm2evalenv.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#define BINNING(param) pParam->HSM2_##param = model->HSM2_##param \
  + model->HSM2_l##param / Lbin + model->HSM2_w##param / Wbin \
  + model->HSM2_p##param / LWbin ;

#define RANGECHECK(param, min, max, pname)                              \
  if ( (param) < (min) || (param) > (max) ) {             \
    printf("warning(HiSIM2): The model/instance parameter %s (= %e) must be in the range [%e , %e].\n", \
           (pname), (param), (min), (max) );                     \
  }


/*---------------------------------------------------*
* smoothZero: flooring to zero.
*      y = 0.5 ( x + sqrt( x^2 + 4 delta^2 ) )
*-----------------*/

#define Fn_SZtemp( y , x , delta ) { \
    T1 = sqrt ( ( x ) *  ( x ) + 4.0 * ( delta ) * ( delta) ) ; \
    y = 0.5 * ( ( x ) + T1 ) ; \
  }

#define Fn_SUtemp( y , x , xmax , delta ) { \
    T1 = ( xmax ) - ( x ) - ( delta ) ; \
    T2 = sqrt ( T1 *  T1 + 4.0 * ( xmax ) * ( delta) ) ; \
    y = ( xmax ) - 0.5 * ( T1 + T2 ) ; \
  }

#define Fn_SLtemp( y , x , xmin , delta ) { \
    T1 = ( x ) - ( xmin ) - ( delta ) ; \
    T2 = sqrt ( T1 *  T1 + 4.0 * ( xmin ) * ( delta ) ) ; \
    y = ( xmin ) + 0.5 * ( T1 + T2 ) ; \
  }

int HSM2temp(
     GENmodel *inModel,
     CKTcircuit *ckt)
{
  register HSM2model *model = (HSM2model *)inModel ;
  register HSM2instance *here ;
  HSM2binningParam *pParam ;
  HSM2modelCGSParam *modelCGS ;
  HSM2hereCGSParam *hereCGS ;
  double mueph ;
  double Leff, dL , LG, Weff, dW , WG , WL , Lgate , Wgate;
  double Lbin, Wbin, LWbin ; /* binning */
  double Nsubpp, Nsubps, Nsub, q_Nsub, Nsubb, Npext ;
  double Lod_half, Lod_half_ref ;
  double MUEPWD = 0.0 ;
  double MUEPLD = 0.0 ;
  double GDLD = 0.0 ;
  double T1, T2, T3 ;
  /* temperature-dependent variables */
  double Eg ,TTEMP, beta, Nin;
  double js, jssw, js2, jssw2 ;
  int i;

  /* declarations for the sc3 clamping part */
  double A, beta_inv, c_eox, cnst0, cnst1, Cox, Cox_inv;
  double Denom, dPpg, dVth, dVthLP, dVthLP_dVb, dVthSC, dVthW;
  double dVth0, dVth0_dVb, fac1, limVgp_dVbs, Pb20, Ps0, Ps0_dVbs;
  double Ps0_min, Qb0, sc3lim, sc3Vbs, sc3Vgs, term1, term2, term3, term4;
  double Tox, T0, T3_dVb, T4, T5, T6, T6_dVb, T8, T8_dVb;
  double T9, T9_dVb, Vgp, Vgs_min, Vfb, Vthp, Vth0;


  for ( ;model ;model = model->HSM2nextModel ) {
    modelCGS = &model->modelCGS ;

    /*-----------------------------------------------------------*
     * Range check of model parameters
     *-----------------*/
    if ( model->HSM2_tox <= 0 ) {
      printf("warning(HiSIM2): TOX = %e\n ", model->HSM2_tox);
      printf("warning(HiSIM2): The model parameter TOX must be positive.\n");
    }
    RANGECHECK(model->HSM2_xld,        0.0,  50.0e-9, "XLD") ;
    RANGECHECK(model->HSM2_xwd,   -10.0e-9, 100.0e-9, "XWD") ;
    RANGECHECK(model->HSM2_rsh,        0.0,   1.0e-3, "RSH") ;
    RANGECHECK(model->HSM2_rshg,       0.0,    100.0, "RSHG") ;
    RANGECHECK(model->HSM2_xqy,    10.0e-9,  50.0e-9, "XQY") ;
    RANGECHECK(model->HSM2_rs,         0.0,  10.0e-3, "RS") ;
    RANGECHECK(model->HSM2_rd,         0.0,  10.0e-3, "RD") ;
    RANGECHECK(model->HSM2_vbi,        1.0,      1.2, "VBI") ;
    RANGECHECK(model->HSM2_parl2,      0.0,  50.0e-9, "PARL2") ;
    RANGECHECK(model->HSM2_lp,         0.0, 300.0e-9, "LP") ;
    RANGECHECK(model->HSM2_pgd2,       0.0,      1.5, "PGD2") ;
    RANGECHECK(model->HSM2_pgd4,       0.0,      3.0, "PGD4") ;
    RANGECHECK(model->HSM2_muecb0lp,   0.0,      2.0, "MUECB0LP") ;
    RANGECHECK(model->HSM2_muecb1lp,   0.0,      2.0, "MUECB1LP") ;
    RANGECHECK(model->HSM2_mueph0,    0.25,     0.35, "MUEPH0") ;
    RANGECHECK(model->HSM2_muesr0,     1.8,      2.2, "MUESR0") ;
    RANGECHECK(model->HSM2_lpext,  1.0e-50,  10.0e-6, "LPEXT") ;
    RANGECHECK(model->HSM2_npext,   1.0e16,   1.0e18, "NPEXT") ;
    RANGECHECK(model->HSM2_scp21,      0.0,      5.0, "SCP21") ;
    RANGECHECK(model->HSM2_scp22,      0.0,      0.0, "SCP22") ;
    RANGECHECK(model->HSM2_bs1,        0.0,  50.0e-3, "BS1") ;
    RANGECHECK(model->HSM2_bs2,        0.5,      1.0, "BS2") ;
    if ( model->HSM2_cgbo < 0.0 ) { 
      printf("warning(HiSIM2): %s = %e\n", "CGBO", model->HSM2_cgbo ); 
      printf("warning(HiSIM2): The model parameter %s must not be less than %s.\n", "CGBO", "0.0" ); 
    }
    RANGECHECK(model->HSM2_clm5,       0.0,      2.0, "CLM5") ;
    RANGECHECK(model->HSM2_clm6,       0.0,     20.0, "CLM6") ;
    RANGECHECK(model->HSM2_vover,      0.0,     50.0, "VOVER") ;
    RANGECHECK(model->HSM2_voverp,     0.0,      2.0, "VOVERP") ;
    RANGECHECK(model->HSM2_qme1,       0.0, 300.0e-9, "QME1") ;
    RANGECHECK(model->HSM2_qme3,       0.0,800.0e-12, "QME3") ;
    RANGECHECK(model->HSM2_tnom,      22.0,     32.0, "TNOM") ;
    RANGECHECK(model->HSM2_ddltmax,    1.0,     20.0, "DDLTMAX") ;
    RANGECHECK(model->HSM2_ddltict,   -3.0,     20.0, "DDLTICT") ;
    RANGECHECK(model->HSM2_ddltslp,    0.0,     20.0, "DDLTSLP") ;
    RANGECHECK(model->HSM2_cvb,       -0.1,      0.2, "CVB") ;
    RANGECHECK(model->HSM2_cvbk,      -0.1,      0.2, "CVBK") ;
    RANGECHECK(model->HSM2_byptol,     0.0,      1.0, "BYPTOL") ;
    RANGECHECK(model->HSM2_sc3Vbs,    -3.0,      0.0, "SC3VBS") ;

    /*-----------------------------------------------------------*
     * Change units into CGS.
     *-----------------*/
    modelCGS->HSM2_tox   = model->HSM2_tox * C_m2cm ;
    modelCGS->HSM2_xld   = model->HSM2_xld * C_m2cm ;
    modelCGS->HSM2_xwd   = model->HSM2_xwd * C_m2cm ;
    modelCGS->HSM2_xqy   = model->HSM2_xqy * C_m2cm ;
    modelCGS->HSM2_xl    = model->HSM2_xl * C_m2cm ;
    modelCGS->HSM2_xw    = model->HSM2_xw * C_m2cm ;
    modelCGS->HSM2_saref = model->HSM2_saref * C_m2cm ;
    modelCGS->HSM2_sbref = model->HSM2_sbref * C_m2cm ;
    modelCGS->HSM2_ll    = model->HSM2_ll * C_m2cm ;
    modelCGS->HSM2_lld   = model->HSM2_lld * C_m2cm ;
    modelCGS->HSM2_wl    = model->HSM2_wl * C_m2cm ;
    modelCGS->HSM2_wld   = model->HSM2_wld * C_m2cm ;
    modelCGS->HSM2_lp    = model->HSM2_lp * C_m2cm ;
    modelCGS->HSM2_tpoly = model->HSM2_tpoly * C_m2cm ;
    modelCGS->HSM2_parl2 = model->HSM2_parl2 * C_m2cm ;
    modelCGS->HSM2_qme1  = model->HSM2_qme1 * C_m2cm ;
    modelCGS->HSM2_qme3  = model->HSM2_qme3 * C_m2cm ;
    modelCGS->HSM2_cgbo  = model->HSM2_cgbo / C_m2cm ;
    modelCGS->HSM2_cj    = model->HSM2_cj / C_m2cm_p2 ;
    modelCGS->HSM2_cjsw  = model->HSM2_cjsw / C_m2cm ;
    modelCGS->HSM2_cjswg = model->HSM2_cjswg / C_m2cm ;
    modelCGS->HSM2_lpext = model->HSM2_lpext * C_m2cm ;
    modelCGS->HSM2_wl1   = model->HSM2_wl1 * C_m2cm ;
    modelCGS->HSM2_rs    = model->HSM2_rs * C_m2cm ;
    modelCGS->HSM2_rd    = model->HSM2_rd * C_m2cm ;
    GDLD = model->HSM2_gdld * C_m2um ;

    /*-----------------------------------------------------------*
     * Change unit into Kelvin.
     *-----------------*/
    model->HSM2_ktnom = model->HSM2_tnom + 273.15 ; /* [C] -> [K] */


    /* SourceSatCurrent = 1.0e-14 */
    /* DrainSatCurrent = 1.0e-14 */
    model->HSM2_vcrit = CONSTvt0 * log( CONSTvt0 / (CONSTroot2 * 1.0e-14) ) ;

    /* Quantum Mechanical Effect */
    if ( ( model->HSM2_qme1 == 0.0 && model->HSM2_qme3 == 0.0 ) || model->HSM2_qme2 == 0.0 ) {
      model->HSM2_flg_qme = 0 ;
    } else {
      model->HSM2_flg_qme = 1 ;
      model->HSM2_qme12 = model->HSM2_qme1 / ( model->HSM2_qme2 * model->HSM2_qme2 ) ;
    }

    for ( here = model->HSM2instances; here; here = here->HSM2nextInstance ) {
      hereCGS = &here->hereCGS ;
      pParam = &here->pParam ;

      /*-----------------------------------------------------------*
       * Range check of instance parameters
       *-----------------*/
      RANGECHECK(here->HSM2_l, model->HSM2_lmin, model->HSM2_lmax, "L") ;
      RANGECHECK(here->HSM2_w/here->HSM2_nf, model->HSM2_wmin, model->HSM2_wmax, "W/NF") ;
      RANGECHECK(here->HSM2_mphdfm,        -3.0,              3.0, "MPHDFM") ;

      /*-----------------------------------------------------------*
       * Change units into CGS.
       *-----------------*/
      hereCGS->HSM2_l    = here->HSM2_l  * C_m2cm ;
      hereCGS->HSM2_w    = here->HSM2_w  * C_m2cm ;
      hereCGS->HSM2_as   = here->HSM2_as * C_m2cm_p2 ;
      hereCGS->HSM2_ad   = here->HSM2_ad * C_m2cm_p2 ;
      hereCGS->HSM2_ps   = here->HSM2_ps * C_m2cm ;
      hereCGS->HSM2_pd   = here->HSM2_pd * C_m2cm ;
      hereCGS->HSM2_xgw  = here->HSM2_xgw * C_m2cm ;
      hereCGS->HSM2_xgl  = here->HSM2_xgl * C_m2cm ;
      hereCGS->HSM2_sa   = here->HSM2_sa  * C_m2cm ;
      hereCGS->HSM2_sb   = here->HSM2_sb  * C_m2cm ;
      hereCGS->HSM2_sd   = here->HSM2_sd  * C_m2cm ;
    /*-----------------------------------------------------------*
     * Change unit into Kelvin.
     *-----------------*/
      here->HSM2_ktemp = here->HSM2_temp + 273.15 ; /* [C] -> [K] */


      here->HSM2_lgate = Lgate = hereCGS->HSM2_l + modelCGS->HSM2_xl ;
      Wgate = hereCGS->HSM2_w / here->HSM2_nf  + modelCGS->HSM2_xw ;

      LG = Lgate * 1.0e4 ;
      here->HSM2_wg = WG = Wgate * 1.0e4 ; 
      WL = WG * LG ;
      MUEPWD = model->HSM2_muepwd * C_m2um ;
      MUEPLD = model->HSM2_muepld * C_m2um ;

      /* binning calculation */
      Lbin = pow(LG, model->HSM2_lbinn) ;
      Wbin = pow(WG, model->HSM2_wbinn) ;
      LWbin = Lbin * Wbin ;

      BINNING(vmax)
      BINNING(bgtmp1)
      BINNING(bgtmp2)
      BINNING(eg0)
      BINNING(lover)
      BINNING(vfbover)
      BINNING(nover)
      BINNING(wl2)
      BINNING(vfbc)
      BINNING(nsubc)
      BINNING(nsubp)
      BINNING(scp1)
      BINNING(scp2)
      BINNING(scp3)
      BINNING(sc1)
      BINNING(sc2)
      BINNING(sc3)
      BINNING(sc4)
      BINNING(pgd1)
      BINNING(ndep)
      BINNING(ninv)
      BINNING(muecb0)
      BINNING(muecb1)
      BINNING(mueph1)
      BINNING(vtmp)
      BINNING(wvth0)
      BINNING(muesr1)
      BINNING(muetmp)
      BINNING(sub1)
      BINNING(sub2)
      BINNING(svds)
      BINNING(svbs)
      BINNING(svgs)
      BINNING(nsti)
      BINNING(wsti)
      BINNING(scsti1)
      BINNING(scsti2)
      BINNING(vthsti)
      BINNING(muesti1)
      BINNING(muesti2)
      BINNING(muesti3)
      BINNING(nsubpsti1)
      BINNING(nsubpsti2)
      BINNING(nsubpsti3)
      BINNING(cgso)
      BINNING(cgdo)
      BINNING(js0)
      BINNING(js0sw)
      BINNING(nj)
      BINNING(cisbk)
      BINNING(clm1)
      BINNING(clm2)
      BINNING(clm3)
      BINNING(wfc)
      BINNING(gidl1)
      BINNING(gidl2)
      BINNING(gleak1)
      BINNING(gleak2)
      BINNING(gleak3)
      BINNING(gleak6)
      BINNING(glksd1)
      BINNING(glksd2)
      BINNING(glkb1)
      BINNING(glkb2)
      BINNING(nftrp)
      BINNING(nfalp)
      BINNING(vdiffj)
      BINNING(ibpc1)
      BINNING(ibpc2)

      /*-----------------------------------------------------------*
       * Range check of model parameters
       *-----------------*/
      RANGECHECK(pParam->HSM2_vmax,     1.0e5,   20.0e6, "VMAX") ;
      RANGECHECK(pParam->HSM2_bgtmp1, 50.0e-6,   1.0e-3, "BGTMP1") ;
      RANGECHECK(pParam->HSM2_bgtmp2, -1.0e-6,   1.0e-6, "BGTMP2") ;
      RANGECHECK(pParam->HSM2_eg0,        1.0,      1.3, "EG0") ;
      RANGECHECK(pParam->HSM2_vfbc,      -1.2,     -0.8, "VFBC") ;
      RANGECHECK(pParam->HSM2_vfbover,   -0.2,      0.2, "VFBOVER") ;
      RANGECHECK(pParam->HSM2_nsubc,   1.0e16,   1.0e19, "NSUBC") ;
      RANGECHECK(pParam->HSM2_nsubp,   1.0e16,   1.0e19, "NSUBP") ;
      RANGECHECK(pParam->HSM2_scp1,       0.0,     20.0, "SCP1") ;
      RANGECHECK(pParam->HSM2_scp2,       0.0,      2.0, "SCP2") ;
      RANGECHECK(pParam->HSM2_scp3,       0.0, 100.0e-9, "SCP3") ;
      RANGECHECK(pParam->HSM2_sc1,        0.0,     20.0, "SC1") ;
      RANGECHECK(pParam->HSM2_sc2,        0.0,      2.0, "SC2") ;
      RANGECHECK(pParam->HSM2_sc3,        0.0, 200.0e-9, "SC3") ;
      RANGECHECK(pParam->HSM2_pgd1,       0.0,  50.0e-3, "PGD1") ;
      RANGECHECK(pParam->HSM2_ndep,       0.0,      1.0, "NDEP") ;
      RANGECHECK(pParam->HSM2_ninv,       0.0,      1.0, "NINV") ;
      RANGECHECK(pParam->HSM2_muecb0,   100.0,  100.0e3, "MUECB0") ;
      RANGECHECK(pParam->HSM2_muecb1,     5.0,    1.0e4, "MUECB1") ;
      RANGECHECK(pParam->HSM2_mueph1,   2.0e3,   35.0e3, "MUEPH1") ;
      RANGECHECK(pParam->HSM2_vtmp,      -5.0,      1.0, "VTMP") ;
      RANGECHECK(pParam->HSM2_muesr1,  1.0e13,   1.0e16, "MUESR1") ;
      RANGECHECK(pParam->HSM2_muetmp,     0.5,      2.0, "MUETMP") ;
      RANGECHECK(pParam->HSM2_clm1,       0.5,      1.0, "CLM1") ;
      RANGECHECK(pParam->HSM2_clm2,       1.0,      4.0, "CLM2") ;
      RANGECHECK(pParam->HSM2_clm3,       0.5,      5.0, "CLM3") ;
      RANGECHECK(pParam->HSM2_wfc,   -5.0e-15,   1.0e-6, "WFC") ;
      RANGECHECK(pParam->HSM2_cgso,       0.0, 100e-9 * 100*C_VAC*model->HSM2_kappa/model->HSM2_tox*C_m2cm, "CGSO") ;
      RANGECHECK(pParam->HSM2_cgdo,       0.0, 100e-9 * 100*C_VAC*model->HSM2_kappa/model->HSM2_tox*C_m2cm, "CGDO") ;
      RANGECHECK(pParam->HSM2_ibpc1,      0.0,   1.0e12, "IBPC1") ;
      RANGECHECK(pParam->HSM2_ibpc2,      0.0,   1.0e12, "IBPC2") ;
      RANGECHECK(pParam->HSM2_nsti,    1.0e16,   1.0e19, "NSTI") ;

      /*-----------------------------------------------------------*
       * Change units into CGS.
       *-----------------*/
      pParam->HSM2_lover  *= C_m2cm ;
      pParam->HSM2_sc3    *= C_m2cm ;
      pParam->HSM2_scp3   *= C_m2cm ;
      pParam->HSM2_wfc    *= C_m2cm ;
      pParam->HSM2_wsti   *= C_m2cm ;
      pParam->HSM2_gidl1  *= C_m2cm_p1o2 ;
      pParam->HSM2_cgso   /= C_m2cm ;
      pParam->HSM2_cgdo   /= C_m2cm ;
      pParam->HSM2_js0    /= C_m2cm_p2 ;
      pParam->HSM2_js0sw  /= C_m2cm ;

      /* Band gap */
      here->HSM2_egtnom = pParam->HSM2_eg0 - model->HSM2_ktnom    
        * ( 90.25e-6 + model->HSM2_ktnom * 1.0e-7 ) ;
  
      /* C_EOX */
      here->HSM2_cecox = C_VAC * model->HSM2_kappa ;
  
      /* Vth reduction for small Vds */
      here->HSM2_msc = model->HSM2_scp22  ;

      /* Poly-Si Gate Depletion */
      if ( pParam->HSM2_pgd1 == 0.0 ) {
        here->HSM2_flg_pgd = 0 ;
      } else {
        here->HSM2_flg_pgd = 1 ;
      }


      /* CLM5 & CLM6 */
      here->HSM2_clmmod = 1e0 + pow( LG , model->HSM2_clm5 ) * model->HSM2_clm6 ;

      /* Half length of diffusion */
      T1 = 1.0 / (modelCGS->HSM2_saref + 0.5 * hereCGS->HSM2_l) 
         + 1.0 / (modelCGS->HSM2_sbref + 0.5 * hereCGS->HSM2_l);
      Lod_half_ref = 2.0 / T1 ;

      if (hereCGS->HSM2_sa > 0.0 && hereCGS->HSM2_sb > 0.0 &&
	  (here->HSM2_nf == 1.0 ||
           (here->HSM2_nf > 1.0 && hereCGS->HSM2_sd > 0.0))) {
        T1 = 0.0;
        for (i = 0; i < here->HSM2_nf; i++) {
          T1 += 1.0 / (hereCGS->HSM2_sa + 0.5 * hereCGS->HSM2_l 
                       + i * (hereCGS->HSM2_sd + hereCGS->HSM2_l))
              + 1.0 / (hereCGS->HSM2_sb + 0.5 * hereCGS->HSM2_l 
                       + i * (hereCGS->HSM2_sd + hereCGS->HSM2_l));
        }
        Lod_half = 2.0 * here->HSM2_nf / T1;
      } else {
        Lod_half = 0.0;
      }

      Npext = model->HSM2_npext * ( 1.0 + model->HSM2_npextw / pow( WG, model->HSM2_npextwp ) ); /* new */
      here->HSM2_mueph1 = pParam->HSM2_mueph1 ;
      here->HSM2_nsubp  = pParam->HSM2_nsubp ;
      here->HSM2_nsubc  = pParam->HSM2_nsubc ;

      /* DFM */
      if ( model->HSM2_codfm == 1 && here->HSM2_nsubcdfm_Given ) {
	RANGECHECK(here->HSM2_nsubcdfm,   1.0e16,   1.0e19, "NSUBCDFM") ;
 	here->HSM2_mueph1 *= here->HSM2_mphdfm
	  * ( log(here->HSM2_nsubcdfm) - log(here->HSM2_nsubc) ) + 1.0 ;
	here->HSM2_nsubp += here->HSM2_nsubcdfm - here->HSM2_nsubc ;
 	Npext += here->HSM2_nsubcdfm - here->HSM2_nsubc ;
 	here->HSM2_nsubc = here->HSM2_nsubcdfm ;
      }

	/* WPE */
        T0 = model->HSM2_nsubcwpe *
              ( here->HSM2_sca
                + model->HSM2_web * here->HSM2_scb
                + model->HSM2_wec * here->HSM2_scc ) ;
        here->HSM2_nsubc +=  T0 ;
        Fn_SLtemp( here->HSM2_nsubc , here->HSM2_nsubc , 1e15 , 0.01 ) ;
        T0 = model->HSM2_nsubpwpe *
              ( here->HSM2_sca
                + model->HSM2_web * here->HSM2_scb
                + model->HSM2_wec * here->HSM2_scc ) ;
        here->HSM2_nsubp +=  T0 ;
        Fn_SLtemp( here->HSM2_nsubp , here->HSM2_nsubp , 1e15 , 0.01 ) ;
        T0 = model->HSM2_npextwpe *
              ( here->HSM2_sca
                + model->HSM2_web * here->HSM2_scb
                + model->HSM2_wec * here->HSM2_scc ) ;
        Npext +=  T0 ;
        Fn_SLtemp( Npext , Npext , 1e15 , 0.01 ) ;
	/* WPE end */

      /* Coulomb Scattering */
      here->HSM2_muecb0 = pParam->HSM2_muecb0 * pow( LG, model->HSM2_muecb0lp );
      here->HSM2_muecb1 = pParam->HSM2_muecb1 * pow( LG, model->HSM2_muecb1lp );

      /* Phonon Scattering (temperature-independent part) */
      mueph = pParam->HSM2_mueph1 
        * (1.0e0 + (model->HSM2_muephw / pow( WG + MUEPWD , model->HSM2_muepwp))) 
        * (1.0e0 + (model->HSM2_muephl / pow( LG + MUEPLD , model->HSM2_mueplp))) 
        * (1.0e0 + (model->HSM2_muephw2 / pow( WG, model->HSM2_muepwp2))) 
        * (1.0e0 + (model->HSM2_muephl2 / pow( LG, model->HSM2_mueplp2))) 
        * (1.0e0 + (model->HSM2_muephs / pow( WL, model->HSM2_muepsp)));  
      if (Lod_half > 0.0) {
        T1 = 1.0e0 / (1.0e0 + pParam->HSM2_muesti2) ;
        T2 = pow (pParam->HSM2_muesti1 / Lod_half, pParam->HSM2_muesti3) ;
        T3 = pow (pParam->HSM2_muesti1 / Lod_half_ref, pParam->HSM2_muesti3) ;
        here->HSM2_mueph = mueph * (1.0e0 + T1 * T2) / (1.0e0 + T1 * T3); 
      } else {
        here->HSM2_mueph = mueph;
      }
      
      /* Surface Roughness Scattering */
      here->HSM2_muesr = model->HSM2_muesr0 
        * (1.0e0 + (model->HSM2_muesrl / pow (LG, model->HSM2_mueslp))) 
        * (1.0e0 + (model->HSM2_muesrw / pow (WG, model->HSM2_mueswp))) ;

      /* Coefficients of Qbm for Eeff */
      T1 = pow( LG, model->HSM2_ndeplp ) ;
      T2 = pow( WG, model->HSM2_ndepwp ) ; /* new */
      T3 = T1 + model->HSM2_ndepl ;
      T4 = T2 + model->HSM2_ndepw ;
      if( T3 < 1e-8 ) { T3 = 1e-8; } 
      if( T4 < 1e-8 ) { T4 = 1e-8; } 
      here->HSM2_ndep_o_esi = ( pParam->HSM2_ndep * T1 ) / T3  * T2 / T4
	/ C_ESI ;
      here->HSM2_ninv_o_esi = pParam->HSM2_ninv / C_ESI ;

      /* Metallurgical channel geometry */
      dL = modelCGS->HSM2_xld  
        + (modelCGS->HSM2_ll / pow (Lgate + modelCGS->HSM2_lld, model->HSM2_lln)) ;
      dW = modelCGS->HSM2_xwd  
        + (modelCGS->HSM2_wl / pow (Wgate + modelCGS->HSM2_wld, model->HSM2_wln)) ;  
    
      Leff = Lgate - 2.0e0 * dL ;
      if ( Leff <= 1.0e-7 ) {   
        IFuid namarr[2];
        namarr[0] = model->HSM2modName;
        namarr[1] = here->HSM2name;
        (*(SPfrontEnd->IFerror))
          ( 
           ERR_FATAL, 
           "HiSIM2: MOSFET(%s) MODEL(%s): effective channel length is smaller than 1nm", 
           namarr 
           );
        return (E_BADPARM);
      }
      here->HSM2_leff = Leff ;

      /* Wg dependence for short channel devices */
      here->HSM2_lgatesm = Lgate + modelCGS->HSM2_wl1 / pow( WL , model->HSM2_wl1p ) ;
      here->HSM2_dVthsm = pParam->HSM2_wl2 / pow( WL , model->HSM2_wl2p ) ;

      /* Lg dependence of wsti */
      T1 = 1.0e0 + model->HSM2_wstil / pow( here->HSM2_lgatesm * 1e4 , model->HSM2_wstilp ) ;
      T2 = 1.0e0 + model->HSM2_wstiw / pow( WG , model->HSM2_wstiwp ) ;
      here->HSM2_wsti = pParam->HSM2_wsti * T1 * T2 ;

      here->HSM2_weff = Weff = Wgate - 2.0e0 * dW ;
      if ( Weff <= 0.0 ) {   
        IFuid namarr[2];
        namarr[0] = model->HSM2modName;
        namarr[1] = here->HSM2name;
        (*(SPfrontEnd->IFerror))
          ( 
           ERR_FATAL, 
           "HiSIM2: MOSFET(%s) MODEL(%s): effective channel width is negative or 0", 
           namarr 
           );
        return (E_BADPARM);
      }
      here->HSM2_weff_nf = Weff * here->HSM2_nf ;

      /* Surface impurity profile */

      T1 = 2.0 * ( 1.0 - model->HSM2_nsubpfac ) / model->HSM2_nsubpl * LG + 2.0 * model->HSM2_nsubpfac - 1.0 ;
      Fn_SUtemp( T1 , T1 , 1 , 0.01 ) ;
      Fn_SLtemp( T1 , T1 , model->HSM2_nsubpfac  , 0.01 ) ;
      here->HSM2_nsubp *= T1 ;

      /* Note: Sign Changed --> */
      Nsubpp = here->HSM2_nsubp  
        * (1.0e0 + (model->HSM2_nsubpw / pow (WG, model->HSM2_nsubpwp))) ;
      /* <-- Note: Sign Changed */

      if (Lod_half > 0.0) {
        T1 = 1.0e0 / (1.0e0 + pParam->HSM2_nsubpsti2) ;
        T2 = pow (pParam->HSM2_nsubpsti1 / Lod_half, pParam->HSM2_nsubpsti3) ;
        T3 = pow (pParam->HSM2_nsubpsti1 / Lod_half_ref, pParam->HSM2_nsubpsti3) ;
        Nsubps = Nsubpp * (1.0e0 + T1 * T2) / (1.0e0 + T1 * T3) ;
      } else {
        Nsubps = Nsubpp ;
      }

      T2 = 1.0e0 + ( model->HSM2_nsubcw / pow ( WG, model->HSM2_nsubcwp )) ;
      T2 *= 1.0e0 + ( model->HSM2_nsubcw2 / pow ( WG, model->HSM2_nsubcwp2 )) ;
      T3 = model->HSM2_nsubcmax / here->HSM2_nsubc ;

      Fn_SUtemp( T1 , T2 , T3 , 0.01 ) ;
      here->HSM2_nsubc *= T1 ;

      if ( here->HSM2_nsubc <= 0.0 ) {
        fprintf ( stderr , "*** warning(HiSIM): actual NSUBC value is negative -> reset to 1E+15.\n" ) ;
        fprintf ( stderr , "    The model parameter  NSUBCW/NSUBCWP and/or NSUBCW2/NSUBCW2P might be wrong.\n" ) ;
        here->HSM2_nsubc = 1e15 ;
      }
      if(Npext < here->HSM2_nsubc || Npext > here->HSM2_nsubp) {
        fprintf ( stderr , "*** warning(HiSIM): actual NPEXT value is smaller than NSUBC and/or greater than NSUBP.\n" ) ;
        fprintf ( stderr , "    ( Npext = %e , NSUBC = %e , NSUBP = %e ) \n",Npext,here->HSM2_nsubc,here->HSM2_nsubp);
        fprintf ( stderr , "    The model parameter  NPEXTW and/or NPEXTWP might be wrong.\n" ) ;
      }

      if( Lgate > modelCGS->HSM2_lp ){
        Nsub = (here->HSM2_nsubc * (Lgate - modelCGS->HSM2_lp) 
                +  Nsubps  * modelCGS->HSM2_lp) / Lgate ;
      } else {
        Nsub = Nsubps
          + (Nsubps - here->HSM2_nsubc) * (modelCGS->HSM2_lp - Lgate) 
          / modelCGS->HSM2_lp ;
      }
      T3 = 0.5e0 * Lgate - modelCGS->HSM2_lp ;
      Fn_SZtemp( T3 , T3 , 1e-8 ) ;
      T1 = Fn_Max(0.0e0, modelCGS->HSM2_lpext ) ;
      T2 = T3 * T1 / ( T3 + T1 ) ;

      here->HSM2_nsub = 
        Nsub = Nsub + T2 * (Npext - here->HSM2_nsubc) / Lgate ;
      here->HSM2_qnsub = q_Nsub  = C_QE * Nsub ;
      here->HSM2_qnsub_esi = q_Nsub * C_ESI ;
      here->HSM2_2qnsub_esi = 2.0 * here->HSM2_qnsub_esi ;

      /* Pocket Overlap (temperature-independent part) */
      if ( Lgate <= 2.0e0 * modelCGS->HSM2_lp ) {
        Nsubb = 2.0e0 * Nsubps 
          - (Nsubps - here->HSM2_nsubc) * Lgate 
          / modelCGS->HSM2_lp - here->HSM2_nsubc ;
        here->HSM2_ptovr0 = log (Nsubb / here->HSM2_nsubc) ;
      } else {
        here->HSM2_ptovr0 = 0.0e0 ;
      }

      /* costi0 and costi1 for STI transistor model (temperature-independent part) */
      here->HSM2_costi00 = sqrt (2.0 * C_QE * pParam->HSM2_nsti * C_ESI ) ;
      here->HSM2_nsti_p2 = 1.0 / ( pParam->HSM2_nsti * pParam->HSM2_nsti ) ;

      /* Velocity Temperature Dependence (Temperature-dependent part will be multiplied later.) */
      here->HSM2_vmax0 = (1.0e0 + (model->HSM2_vover / pow (LG, model->HSM2_voverp)))
        * (1.0e0 + (model->HSM2_vovers / pow (WL, model->HSM2_voversp))) ;

      /* 2 phi_B (temperature-independent) */
      /* @300K, with pocket */
      here->HSM2_pb20 = 2.0e0 / C_b300 * log (Nsub / C_Nin0) ;
      /* @300K, w/o pocket */
      here->HSM2_pb2c = 2.0e0 / C_b300 * log (here->HSM2_nsubc / C_Nin0) ;


      /* constant for Poly depletion */
      here->HSM2_cnstpgd = pow ( 1e0 + 1e0 / LG , model->HSM2_pgd4 ) 
        * pParam->HSM2_pgd1 ;




      /* Gate resistance */
      if ( here->HSM2_corg == 1 ) {
        T1 = hereCGS->HSM2_xgw + Weff / (3.0e0 * here->HSM2_ngcon);
        T2 = Lgate - hereCGS->HSM2_xgl; 
        here->HSM2_grg = model->HSM2_rshg * T1 / (here->HSM2_ngcon * T2 * here->HSM2_nf);
        if (here->HSM2_grg > 1.0e-3) here->HSM2_grg = here->HSM2_m / here->HSM2_grg;
        else {
          here->HSM2_grg = here->HSM2_m * 1.0e3;
          printf("warning(HiSIM2): The gate conductance reset to 1.0e3 mho.\n");
        }
      }

      /* Process source/drain series resistamce */
      here->HSM2_rd = 0.0;
      if ( model->HSM2_rsh > 0.0 ) {
        here->HSM2_rd += model->HSM2_rsh * here->HSM2_nrd ;
      } 
      if ( modelCGS->HSM2_rd > 0.0 ) {
       here->HSM2_rd += modelCGS->HSM2_rd / here->HSM2_weff_nf ;
     }

      here->HSM2_rs = 0.0;
      if ( model->HSM2_rsh > 0.0 ) {
        here->HSM2_rs += model->HSM2_rsh * here->HSM2_nrs ;
      }
      if ( modelCGS->HSM2_rs > 0.0 ) {
        here->HSM2_rs += modelCGS->HSM2_rs / here->HSM2_weff_nf ; 
      }

      if (model->HSM2_corsrd < 0) {
        if ( here->HSM2_rd > 0.0 ) {
          here->HSM2drainConductance = here->HSM2_m / here->HSM2_rd ;
        } else {
          here->HSM2drainConductance = 0.0;
        }
        if ( here->HSM2_rs > 0.0 ) {
          here->HSM2sourceConductance = here->HSM2_m / here->HSM2_rs ;
        } else {
          here->HSM2sourceConductance = 0.0;
        }
      } else if (model->HSM2_corsrd > 0) {
        here->HSM2drainConductance = 0.0 ;
        here->HSM2sourceConductance = 0.0 ;
        if ( here->HSM2_rd > 0.0 && model->HSM2_cothrml != 0 ) {
          here->HSM2internalGd = here->HSM2_m / here->HSM2_rd ;
        } else {
          here->HSM2internalGd = 0.0;
        }
        if ( here->HSM2_rs > 0.0 && model->HSM2_cothrml != 0 ) {
          here->HSM2internalGs = here->HSM2_m / here->HSM2_rs ;
        } else {
          here->HSM2internalGs = 0.0;
        }
      } else {
        here->HSM2drainConductance = 0.0 ;
        here->HSM2sourceConductance = 0.0 ;
      }


      /* Body resistance */
      if ( here->HSM2_corbnet == 1 ) {
        if (here->HSM2_rbdb < 1.0e-3) here->HSM2_grbdb = here->HSM2_m * 1.0e3 ; /* in mho */
        else here->HSM2_grbdb = here->HSM2_m * ( model->HSM2_gbmin + 1.0 / here->HSM2_rbdb ) ;

        if (here->HSM2_rbpb < 1.0e-3) here->HSM2_grbpb = here->HSM2_m * 1.0e3 ;
        else here->HSM2_grbpb = here->HSM2_m * ( model->HSM2_gbmin + 1.0 / here->HSM2_rbpb ) ;

        if (here->HSM2_rbps < 1.0e-3) here->HSM2_grbps = here->HSM2_m * 1.0e3 ;
        else here->HSM2_grbps = here->HSM2_m * ( model->HSM2_gbmin + 1.0 / here->HSM2_rbps ) ;

        if (here->HSM2_rbsb < 1.0e-3) here->HSM2_grbsb = here->HSM2_m * 1.0e3 ;
        else here->HSM2_grbsb = here->HSM2_m * ( model->HSM2_gbmin + 1.0 / here->HSM2_rbsb ) ;

        if (here->HSM2_rbpd < 1.0e-3) here->HSM2_grbpd = here->HSM2_m * 1.0e3 ;
        else here->HSM2_grbpd = here->HSM2_m * ( model->HSM2_gbmin + 1.0 / here->HSM2_rbpd ) ;
      }

      /* Vdseff */
      T1 = model->HSM2_ddltslp * LG + model->HSM2_ddltict ;
      here->HSM2_ddlt = T1 * model->HSM2_ddltmax / ( T1 + model->HSM2_ddltmax ) + 1.0 ;

      /* Isub */
      T2 = pow( Weff , model->HSM2_svgswp ) ;
      here->HSM2_vg2const = pParam->HSM2_svgs
         * ( 1.0e0
           + model->HSM2_svgsl / pow( here->HSM2_lgate , model->HSM2_svgslp ) )
         * ( T2 / ( T2 + model->HSM2_svgsw ) ) ; 

      here->HSM2_xvbs = pParam->HSM2_svbs 
         * ( 1.0e0
           + model->HSM2_svbsl / pow( here->HSM2_lgate , model->HSM2_svbslp ) ) ;
      here->HSM2_xgate = model->HSM2_slg 
         * ( 1.0
         + model->HSM2_slgl / pow( here->HSM2_lgate , model->HSM2_slglp ) ) ;

      here->HSM2_xsub1 = pParam->HSM2_sub1 
         * ( 1.0 
         + model->HSM2_sub1l / pow( here->HSM2_lgate , model->HSM2_sub1lp ) ) ;

      here->HSM2_xsub2 = pParam->HSM2_sub2
         * ( 1.0 + model->HSM2_sub2l / here->HSM2_lgate ) ;

      /* Fringing capacitance */
      here->HSM2_cfrng = C_EOX / ( C_Pi / 2.0e0 ) * here->HSM2_weff_nf 
         * log( 1.0e0 + modelCGS->HSM2_tpoly / modelCGS->HSM2_tox ) ; 

      /* Additional term of lateral-field-induced capacitance */
      here->HSM2_cqyb0 = 1.0e4 * here->HSM2_weff_nf
	* model->HSM2_xqy1 / pow( LG , model->HSM2_xqy2 ) ;

      /* Parasitic component of the channel current */
      here->HSM2_ptl0 = model->HSM2_ptl * pow( LG        , - model->HSM2_ptlp ) ;
      here->HSM2_pt40 = model->HSM2_pt4 * pow( LG        , - model->HSM2_pt4p ) ;
      here->HSM2_gdl0 = model->HSM2_gdl * pow( LG + modelCGS->HSM2_gdld , - model->HSM2_gdlp ) ;


      /*-----------------------------------------------------------*
       * Temperature dependent constants. 
       *-----------------*/
        TTEMP = ckt->CKTtemp ;
        if ( here->HSM2_temp_Given ) TTEMP = here->HSM2_ktemp ;
        if ( here->HSM2_dtemp_Given ) {
          TTEMP = TTEMP + here->HSM2_dtemp ;
          here->HSM2_ktemp = TTEMP ;
        }

        /* Band gap */
        T1 = TTEMP - model->HSM2_ktnom ;
        T2 = TTEMP * TTEMP - model->HSM2_ktnom * model->HSM2_ktnom ;
        here->HSM2_eg = Eg = here->HSM2_egtnom - pParam->HSM2_bgtmp1 * T1
          - pParam->HSM2_bgtmp2 * T2 ;
        here->HSM2_sqrt_eg = sqrt( Eg ) ;


	T1 = 1.0 / TTEMP ;
	T2 = 1.0 / model->HSM2_ktnom ;
	T3 = here->HSM2_egtnom + model->HSM2_egig
	  + model->HSM2_igtemp2 * ( T1 - T2 )
	  + model->HSM2_igtemp3 * ( T1 * T1 - T2 * T2 ) ;
 	here->HSM2_egp12 = sqrt ( T3 ) ;
 	here->HSM2_egp32 = T3 * here->HSM2_egp12 ;

        
        /* Inverse of the thermal voltage */
        here->HSM2_beta = beta = C_QE / (C_KB * TTEMP) ;
        here->HSM2_beta_inv = 1.0 / beta ;
        here->HSM2_beta2 = beta * beta ;
        here->HSM2_betatnom = C_QE / (C_KB * model->HSM2_ktnom) ;

        /* Intrinsic carrier concentration */
        here->HSM2_nin = Nin = C_Nin0 * pow (TTEMP / model->HSM2_ktnom, 1.5e0) 
          * exp (- Eg / 2.0e0 * beta + here->HSM2_egtnom / 2.0e0 * here->HSM2_betatnom) ;


        /* Phonon Scattering (temperature-dependent part) */
        T1 =  pow (TTEMP / model->HSM2_ktnom, pParam->HSM2_muetmp) ;
        here->HSM2_mphn0 = T1 / here->HSM2_mueph ;
        here->HSM2_mphn1 = here->HSM2_mphn0 * model->HSM2_mueph0 ;


        /* Pocket Overlap (temperature-dependent part) */
        here->HSM2_ptovr = here->HSM2_ptovr0 / beta ;


        /* Velocity Temperature Dependence */
        T1 =  TTEMP / model->HSM2_ktnom ;
        here->HSM2_vmax = here->HSM2_vmax0 * pParam->HSM2_vmax 
          / (1.8 + 0.4 * T1 + 0.1 * T1 * T1 - pParam->HSM2_vtmp * (1.0e0 - T1)) ;


        /* Coefficient of the F function for bulk charge */
        here->HSM2_cnst0 = sqrt ( 2.0 * C_ESI * C_QE * here->HSM2_nsub / beta ) ;
	here->HSM2_cnst0over = here->HSM2_cnst0 * sqrt( pParam->HSM2_nover / here->HSM2_nsub ) ;     

        /* 2 phi_B (temperature-dependent) */
        /* @temp, with pocket */
        here->HSM2_pb2 =  2.0e0 / beta * log (here->HSM2_nsub / Nin) ;
        if ( pParam->HSM2_nover != 0.0) {
	   here->HSM2_pb2over = 2.0 / beta * log( pParam->HSM2_nover / Nin ) ;

           /* (1 / cnst1 / cnstCoxi) for Ps0LD_iniB */
           T1 = here->HSM2_cnst0over * modelCGS->HSM2_tox / here->HSM2_cecox  ;
           T2 = pParam->HSM2_nover / Nin ;
           T1 = T2 * T2 / ( T1 * T1 ) ;
           here->HSM2_ps0ldinib  = T1 ; /* (1 / cnst1 / cnstCoxi) */

        }else {
           here->HSM2_pb2over = 0.0 ;
           here->HSM2_ps0ldinib  = 0.0 ;
        }


        /* Depletion Width */
        T1 = 2.0e0 * C_ESI / C_QE ;
        here->HSM2_wdpl = sqrt ( T1 / here->HSM2_nsub ) ;
        here->HSM2_wdplp = sqrt( T1 / ( here->HSM2_nsubp ) ) ; 

        /* cnst1: n_{p0} / p_{p0} */
        T1 = Nin / here->HSM2_nsub ;
        here->HSM2_cnst1 = T1 * T1 ;


        /* for substrate-source/drain junction diode. */
        js   = pParam->HSM2_js0
          * exp ((here->HSM2_egtnom * here->HSM2_betatnom - Eg * beta
                  + model->HSM2_xti * log (TTEMP / model->HSM2_ktnom)) / pParam->HSM2_nj) ;
        jssw = pParam->HSM2_js0sw
          * exp ((here->HSM2_egtnom * here->HSM2_betatnom - Eg * beta 
                  + model->HSM2_xti * log (TTEMP / model->HSM2_ktnom)) / model->HSM2_njsw) ;

        js2  = pParam->HSM2_js0
          * exp ((here->HSM2_egtnom * here->HSM2_betatnom - Eg * beta
                  + model->HSM2_xti2 * log (TTEMP / model->HSM2_ktnom)) / pParam->HSM2_nj) ;  
        jssw2 = pParam->HSM2_js0sw
          * exp ((here->HSM2_egtnom * here->HSM2_betatnom - Eg * beta
                  + model->HSM2_xti2 * log (TTEMP / model->HSM2_ktnom)) / model->HSM2_njsw) ; 
      
        here->HSM2_isbd = hereCGS->HSM2_ad * js + hereCGS->HSM2_pd * jssw ;
        here->HSM2_isbd2 = hereCGS->HSM2_ad * js2 + hereCGS->HSM2_pd * jssw2 ;
        here->HSM2_isbs = hereCGS->HSM2_as * js + hereCGS->HSM2_ps * jssw ;
        here->HSM2_isbs2 = hereCGS->HSM2_as * js2 + hereCGS->HSM2_ps * jssw2 ;

        here->HSM2_vbdt = pParam->HSM2_nj / beta 
          * log (pParam->HSM2_vdiffj * (TTEMP / model->HSM2_ktnom) * (TTEMP / model->HSM2_ktnom) 
                 / (here->HSM2_isbd + 1.0e-50) + 1) ;
        here->HSM2_vbst = pParam->HSM2_nj / beta 
          * log (pParam->HSM2_vdiffj * (TTEMP / model->HSM2_ktnom) * (TTEMP / model->HSM2_ktnom) 
                 / (here->HSM2_isbs + 1.0e-50) + 1) ;

        here->HSM2_exptemp = exp (((TTEMP / model->HSM2_ktnom) - 1) * model->HSM2_ctemp) ;
        here->HSM2_jd_nvtm_inv = 1.0 / ( pParam->HSM2_nj / beta ) ;
        here->HSM2_jd_expcd = exp (here->HSM2_vbdt * here->HSM2_jd_nvtm_inv ) ;
        here->HSM2_jd_expcs = exp (here->HSM2_vbst * here->HSM2_jd_nvtm_inv ) ;

	/* costi0 and costi1 for STI transistor model (temperature-dependent part) */
	here->HSM2_costi0 = here->HSM2_costi00 * sqrt(here->HSM2_beta_inv) ;
	here->HSM2_costi0_p2 = here->HSM2_costi0 * here->HSM2_costi0 ; 
	here->HSM2_costi1 = here->HSM2_nin * here->HSM2_nin * here->HSM2_nsti_p2 ;

        /* check if SC3 is too large */
        if (pParam->HSM2_sc3 && model->HSM2_sc3Vbs < 0.0) {

          beta     = here->HSM2_beta ;
          beta_inv = here->HSM2_beta_inv ;
          Weff     = here->HSM2_weff ;
          Vfb      = pParam->HSM2_vfbc ;
          Pb20     = here->HSM2_pb20 ; 
          cnst0    = here->HSM2_cnst0 ;
          cnst1    = here->HSM2_cnst1 ;
          c_eox    = here->HSM2_cecox ;
          Tox      = modelCGS->HSM2_tox ;
          Cox      = c_eox / Tox ;
          Cox_inv  = 1.0 / Cox ;
          fac1     = cnst0 * Cox_inv ;
          Vgs_min  = model->HSM2_type * model->HSM2_Vgsmin ;
          sc3Vbs   = model->HSM2_sc3Vbs ;
          sc3Vgs   = 2.0 ;
          Ps0_min  = 2.0 * beta_inv * log(-Vgs_min/fac1) ;

          /* approximate solution of Poisson equation for large
             Vgs and negative Vbs (3 iterations!)*/
          Vgp = sc3Vgs - Vfb;
          Denom = fac1*sqrt(cnst1);
          Ps0 = 2.0 * beta_inv * log(Vgp/Denom);
          Ps0 = 2.0 * beta_inv * log((Vgp-Ps0)/Denom);
          Ps0 = 2.0 * beta_inv * log((Vgp-Ps0)/Denom);
          Ps0 = 2.0 * beta_inv * log((Vgp-Ps0)/Denom);
          Ps0_dVbs = 0.0;

          T1   = here->HSM2_2qnsub_esi ;
          Qb0  = sqrt ( T1 ) ;
          Vthp = Ps0 + Vfb + Qb0 * Cox_inv + here->HSM2_ptovr ;

          T1     = 2.0 * C_QE * here->HSM2_nsubc * C_ESI ;
          T2     = sqrt( T1 ) ;
          Vth0   = Ps0 + Vfb + T2 * Cox_inv ;
          T1     = C_ESI * Cox_inv ;
          T2     = here->HSM2_wdplp ;
          T4     = 1.0e0 / ( modelCGS->HSM2_lp * modelCGS->HSM2_lp ) ;
          T3     = 2.0 * ( model->HSM2_vbi - Pb20 ) * T2 * T4 ;
          T5     = T1 * T3 ;
          T6     = Ps0 - sc3Vbs ;
          T6_dVb = Ps0_dVbs - 1.0 ;
          dVth0  = T5 * sqrt( T6 ) ;
          dVth0_dVb = T5 * 0.5 / sqrt( T6 ) * T6_dVb;
          T1     = Vthp - Vth0 ;
          T9     = Ps0 - sc3Vbs ;
          T9_dVb = Ps0_dVbs - 1.0 ;
          T3     = pParam->HSM2_scp1 + pParam->HSM2_scp3 * T9 / modelCGS->HSM2_lp; 
          T3_dVb = pParam->HSM2_scp3 * T9_dVb / modelCGS->HSM2_lp ;
          dVthLP = T1 * dVth0 * T3 ;
          dVthLP_dVb = T1 * dVth0_dVb * T3 + T1 * dVth0 * T3_dVb;

          T3 = here->HSM2_lgate - modelCGS->HSM2_parl2 ;
          T4 = 1.0e0 / ( T3 * T3 ) ;
          T0 = C_ESI * here->HSM2_wdpl * 2.0e0 * ( model->HSM2_vbi - Pb20 ) * T4 ;
          T2 = T0 * Cox_inv ;
          T5 = pParam->HSM2_sc3 / here->HSM2_lgate ;
          T6 = pParam->HSM2_sc1 + T5 * ( Ps0 - sc3Vbs ) ;
          T1 = T6 ;
          A  = T2 * T1 ;
          T9 = Ps0 - sc3Vbs + Ps0_min ;
          T9_dVb = Ps0_dVbs - 1.0 ;
          T8     = sqrt( T9 ) ;
          T8_dVb = 0.5 * T9_dVb / T8 ;
          dVthSC = A * T8 ;

          T1 = 1.0 / Cox ;
          T3 = 1.0 / ( Cox + pParam->HSM2_wfc / Weff ) ;
          T5 = T1 - T3 ;
          dVthW = Qb0 * T5 + pParam->HSM2_wvth0 / here->HSM2_wg ;

          dVth = dVthSC + dVthLP + dVthW + here->HSM2_dVthsm ;
          dPpg = 0.0 ;
          Vgp = sc3Vgs - Vfb + dVth - dPpg ;

          /* Recalculation of Ps0, using more accurate Vgp */
          Ps0 = 2.0 * beta_inv * log(Vgp/Denom);
          Ps0 = 2.0 * beta_inv * log((Vgp-Ps0)/Denom);
          Ps0 = 2.0 * beta_inv * log((Vgp-Ps0)/Denom);
          Ps0 = 2.0 * beta_inv * log((Vgp-Ps0)/Denom);

          term1 = Vgp - Ps0;
          term2 = sqrt(beta*(Ps0-sc3Vbs)-1.0);
          term3 = term1 + fac1 * term2;
          term4 = cnst1 * exp(beta*Ps0);
          limVgp_dVbs = - beta * (term3 + 0.5*fac1 * term4/term2)
                  / (2.0*term1/fac1/fac1*term3 - term4);
          T2     = T0 * Cox_inv ;
          sc3lim = here->HSM2_lgate / T2 
               * (limVgp_dVbs - dVthLP_dVb - T2*pParam->HSM2_sc1*T8_dVb)
               / ((Ps0-sc3Vbs)*T8_dVb +(Ps0_dVbs-1.0)*T8);
          if (sc3lim < 1.0e-20)
              sc3lim = 1e-20 ;
          if (sc3lim < pParam->HSM2_sc3 * 0.999) {
            pParam->HSM2_sc3 = sc3lim;
          }
        }
    } /* End of instance loop */
  }
  return(OK);
}
