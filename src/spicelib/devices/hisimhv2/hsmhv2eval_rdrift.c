/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 2  SUBVERSION : 2  REVISION : 0 ) 
 Model Parameter 'VERSION' : 2.20
 FILE : hsmhveval_rdrift.c

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

/*===========================================================*
* Preamble.
*=================*/
/*---------------------------------------------------*
* Header files.
*-----------------*/
#include "ngspice/ngspice.h"
#ifdef __STDC__
/* #include <ieeefp.h> */
#endif

/*-----------------------------------*
* HiSIM macros
*-----------------*/
#include "hisimhv2.h"
#include "hsmhv2evalenv.h"

/* local variables used in macro functions */
static double TMF0 , TMF1 , TMF2 , TMF3 ;

/*===========================================================*
* pow
*=================*/
#ifdef POW_TO_EXP_AND_LOG
#define Fn_Pow( x , y )  exp( y * log( x )  ) 
#else
#define Fn_Pow( x , y )  pow( x , y )
#endif

/*===========================================================*
* Macro Functions for ceiling/flooring/symmetrization.
*=================*/
/*---------------------------------------------------*
* smoothUpper: ceiling.
*      y = xmax - 0.5 ( arg + sqrt( arg^2 + 4 xmax delta ) )
*    arg = xmax - x - delta
*-----------------*/

#define Fn_SU( y , x , xmax , delta , dx ) { \
    TMF1 = ( xmax ) - ( x ) - ( delta ) ; \
    TMF2 = 4.0 * ( xmax ) * ( delta) ; \
    TMF2 = TMF2 > 0.0 ?  TMF2 : - ( TMF2 ) ; \
    TMF2 = sqrt ( TMF1 * TMF1 + TMF2 ) ; \
    dx = 0.5 * ( 1.0 + TMF1 / TMF2 ) ; \
    y = ( xmax ) - 0.5 * ( TMF1 + TMF2 ) ; \
  }

#define Fn_SU2( y , x , xmax , delta , dy_dx , dy_dxmax ) { \
    TMF1 = ( xmax ) - ( x ) - ( delta ) ; \
    TMF2 = 4.0 * ( xmax ) * ( delta) ; \
    TMF2 = TMF2 > 0.0 ?  TMF2 : - ( TMF2 ) ; \
    TMF2 = sqrt ( TMF1 * TMF1 + TMF2 ) ; \
    dy_dx = 0.5 * ( 1.0 + TMF1 / TMF2 ) ; \
    dy_dxmax = 0.5 * ( 1.0 - ( TMF1 + 2.0 * delta ) / TMF2 ) ; \
    y = ( xmax ) - 0.5 * ( TMF1 + TMF2 ) ; \
  }

/*---------------------------------------------------*
* smoothLower: flooring.
*      y = xmin + 0.5 ( arg + sqrt( arg^2 + 4 xmin delta ) )
*    arg = x - xmin - delta
*-----------------*/

#define Fn_SL( y , x , xmin , delta , dx ) { \
    TMF1 = ( x ) - ( xmin ) - ( delta ) ; \
    TMF2 = 4.0 * ( xmin ) * ( delta ) ; \
    TMF2 = TMF2 > 0.0 ?  TMF2 : - ( TMF2 ) ; \
    TMF2 = sqrt ( TMF1 * TMF1 + TMF2 ) ; \
    dx = 0.5 * ( 1.0 + TMF1 / TMF2 ) ; \
    y = ( xmin ) + 0.5 * ( TMF1 + TMF2 ) ; \
  }

#define Fn_SL2( y , x , xmin , delta , dy_dx, dy_dxmin ) { \
    TMF1 = ( x ) - ( xmin ) - ( delta ) ; \
    TMF2 = 4.0 * ( xmin ) * ( delta ) ; \
    TMF2 = TMF2 > 0.0 ?  TMF2 : - ( TMF2 ) ; \
    TMF2 = sqrt ( TMF1 * TMF1 + TMF2 ) ; \
    dy_dx = 0.5 * ( 1.0 + TMF1 / TMF2 ) ; \
    dy_dxmin = 0.5 * ( 1.0 - ( TMF1 - 2.0 * delta ) / TMF2 ) ; \
    y = ( xmin ) + 0.5 * ( TMF1 + TMF2 ) ; \
  }

/*---------------------------------------------------*
* smoothZero: flooring to zero.
*      y = 0.5 ( x + sqrt( x^2 + 4 delta^2 ) )
*-----------------*/

#define Fn_SZ( y , x , delta , dx ) { \
    TMF2 = sqrt ( ( x ) *  ( x ) + 4.0 * ( delta ) * ( delta ) ) ; \
    dx = 0.5 * ( 1.0 + ( x ) / TMF2 ) ; \
    y = 0.5 * ( ( x ) + TMF2 ) ; \
    if( y < 0.0 ) { y=0.0; dx=0.0; } \
  }

/*---------------------------------------------------*
* SymAdd: evaluate additional term for symmetry.
*-----------------*/

#define Fn_SymAdd( y , x , add0 , dx ) \
{ \
   if( ( x ) < 1e6 ) { \
    TMF1 = 2.0 * ( x ) / ( add0 ) ; \
    TMF2 = 1.0 + TMF1 * ( (1.0/2) + TMF1 * ( (1.0/6) \
               + TMF1 * ( (1.0/24) + TMF1 * ( (1.0/120) \
               + TMF1 * ( (1.0/720) + TMF1 * (1.0/5040) ) ) ) ) ) ; \
    TMF3 = (1.0/2) + TMF1 * ( (1.0/3) \
               + TMF1 * ( (1.0/8) + TMF1 * ( (1.0/30) \
               + TMF1 * ( (1.0/144) + TMF1 * (1.0/840) ) ) ) ) ; \
    y = add0 / TMF2 ; \
    dx = - 2.0 * TMF3 / ( TMF2 * TMF2 ) ; \
   } else { y=0.0; dx=0.9; } \
}

#define Fn_CP( y , x , xmax , pw , dx ) { \
  double x2 = (x) * (x) ; \
  double xmax2 = (xmax) * (xmax) ; \
  double xp = 1.0 , xmp = 1.0 ; \
  int   m =0, mm =0; \
  double arg =0.0, dnm =0.0; \
  for ( m = 0 ; m < pw ; m ++ ) { xp *= x2 ; xmp *= xmax2 ; } \
  arg = xp + xmp ; \
  dnm = arg ; \
  if ( pw == 1 || pw == 2 || pw == 4 || pw == 8 ) { \
    if ( pw == 1 ) { mm = 1 ; \
    } else if ( pw == 2 ) { mm = 2 ; \
    } else if ( pw == 4 ) { mm = 3 ; \
    } else if ( pw == 8 ) { mm = 4 ; } \
    for ( m = 0 ; m < mm ; m ++ ) { dnm = sqrt( dnm ) ; } \
  } else { dnm = Fn_Pow( dnm , 1.0 / ( 2.0 * pw ) ) ; } \
  dnm = 1.0 / dnm ; \
  y = (x) * (xmax) * dnm ; \
  dx = (xmax) * xmp * dnm / arg ; \
}

#define Fn_SU_CP( y , x , xmax , delta , pw , dx ) { \
 if(x > xmax - delta && delta >= 0.0) { \
   TMF1 = x - xmax + delta ; \
   Fn_CP( TMF0 , TMF1 , delta , pw , dx )  \
   y = xmax - delta + TMF0 ; \
   dx = dx ; \
 } else { \
   y = x ; \
   dx = 1.0 ; \
 } \
}

/*===========================================================*
* Function hsmhvrdrift.
*=================*/

int HSMHV2rdrift
(
 double        Vddp,
 double        Vds,
 double        Vbs,
 double        Vsubs, /* substrate-source voltage */
 double        deltemp,
 HSMHV2instance *here,
 HSMHV2model    *model,
 CKTcircuit    *ckt
)
{
  HSMHV2binningParam *pParam = &here->pParam ;
  HSMHV2modelMKSParam *modelMKS = &model->modelMKS ;

  const double small = 1.0e-50 ;

  double Mfactor =0.0, WeffLD_nf  =0.0 ;
  double Ldrift  =0.0, Xldld      =0.0 ;
  double Nover   =0.0                  ;

  /* temporary vars. & derivatives*/
  double T0 =0.0, T0_dVb =0.0, T0_dVd =0.0, T0_dVg =0.0, T0_dT =0.0 ;
  double T1 =0.0, T1_dVd =0.0, T1_dT =0.0, T1_dVddp =0.0 ;
  double T2 =0.0, T2_dVb =0.0, T2_dVd =0.0, T2_dT =0.0, T2_dVddp =0.0 ;
  double T3 =0.0, T3_dT =0.0, T3_dVddp =0.0 ;
  double T4 =0.0, T4_dT =0.0, T4_dVddp =0.0 ;
  double T5 =0.0, T5_dT =0.0, T5_dVddp =0.0 ;
  double T6 =0.0, T6_dT =0.0, T6_dVddp =0.0 ;
  double T9 =0.0 ;

  /* bias-dependent Rd, Rs */

  double Edri  =0.0, Edri_dVddp =0.0 ; 
  double Vdri  =0.0, Vdri_dVddp =0.0, Vdri_dT =0.0 ;
  double Vmax  =0.0, Vmax_dT    =0.0 ;
  double Mu0   =0.0, Mu0_dT  =0.0 ; 
  double Cx    =0.0, Cx_dT      =0.0 ;
  double Car   =0.0, Car_dT     =0.0 ;
  double Mu    =0.0, Mu_dVddp = 0.0, Mu_dT   =0.0 ; 
  double Xov =0.0, Xov_dVds =0.0, Xov_dVgs =0.0, Xov_dVbs =0.0, Xov_dT =0.0 ;
  double Carr =0.0, Carr_dVds=0.0, Carr_dVgs=0.0, Carr_dVbs=0.0, Carr_dVddp =0.0, Carr_dT =0.0 ; 

  double GD   =0.0, GD_dVddp   =0.0, GD_dVgse   =0.0, GD_dT   =0.0, GD_dVds   =0.0, GD_dVgs   =0.0, GD_dVbs   =0.0 ;
  double Rd   =0.0, Rd_dVddp   =0.0, Rd_dVdse   =0.0, Rd_dVgse   =0.0, Rd_dVbse   =0.0, Rd_dT   =0.0, Rd_dVds   =0.0, Rd_dVgs   =0.0, Rd_dVbs   =0.0 ;
  double Vddpz=0.0, Vddpz_dVddp=0.0, Vzadd      =0.0, Vzadd_dVddp=0.0 ;

  /* temperature-dependent variables for SHE model */
  double TTEMP  =0.0, TTEMP0   =0.0 ;

  /* Wdepl and Wjunc */
  double Wdepl, Wdepl_dVd, Wdepl_dVg, Wdepl_dVb, Wdepl_dT;
  double Wjunc0, Wjunc0_dVd, Wjunc0_dVb;
  double Wrdrdjunc, Wjunc, Wjunc_dVd, Wjunc_dVb;

  const double Res_min = 1.0e-4 ;
  const double epsm10 = 10.0e0 * C_EPS_M ;
  const double ps_conv = 1.0e-12 ;

  double Rdrbb_dT =0.0 ;

  double Wdep = 0.0, Wdep_dVdserev = 0.0, Wdep_dVsubsrev = 0.0 ;
  double T1_dVdserev = 0.0, T1_dVsubsrev = 0.0, T6_dVdserev = 0.0, T6_dVsubsrev = 0.0 ;
  double Rd_dVsubs=0.0 ;

#define C_sub_delta 0.1 /* CHECK! */
#define C_sub_delta2 1.0e-9 /* CHECK! */

  NG_IGNORE(Vsubs);

  /*================ Start of executable code.=================*/

  /*-----------------------------------------------------------*
   * Temperature dependent constants. 
   *-----------------*/
  if ( here->HSMHV2tempNode > 0 && pParam->HSMHV2_rth0 != 0.0 ) {

#define HSMHV2EVAL
#include "hsmhv2temp_eval_rdri.h"

  } else {
    if ( here->HSMHV2_dtemp_Given ) { TTEMP = TTEMP + here->HSMHV2_dtemp ; }
    Mu0_dT      = 0.0 ;
    Vmax_dT     = 0.0 ;
    Cx_dT       = 0.0 ;
    Car_dT      = 0.0 ;
    Rdrbb_dT    = 0.0 ;
  }

  Mfactor   = here->HSMHV2_m ; 
  WeffLD_nf = here->HSMHV2_weff_ld * here->HSMHV2_nf ;
  Ldrift    = here->HSMHV2_ldrift1 + here->HSMHV2_ldrift2 ;
  Xldld     = model->HSMHV2_xldld + small ;
  Nover     = pParam->HSMHV2_nover ;


  Mu0    = here->HSMHV2_rdrmue * here->HSMHV2_rdrmuel ;
  Mu0_dT = Mu0_dT * here->HSMHV2_rdrmuel ;
  Vmax    = here->HSMHV2_rdrvmax * here->HSMHV2_rdrvmaxw * here->HSMHV2_rdrvmaxl + small ;
  Vmax_dT = Vmax_dT * here->HSMHV2_rdrvmaxw * here->HSMHV2_rdrvmaxl ;
  Cx      = here->HSMHV2_rdrcx * here->HSMHV2_rdrcxw ;
  Cx_dT   = Cx_dT * here->HSMHV2_rdrcxw ;
  Car     = here->HSMHV2_rdrcar ;

  //-----------------------------------------------------------*
  // Modified bias introduced to realize symmetry at Vddp=0.
  //-----------------//
  if(Vddp < 0) {
    Fn_SymAdd( Vzadd , -Vddp / 2 , model->HSMHV2_vzadd0 , T2 ) ;
    Vzadd_dVddp = - T2 / 2.0 ;
    if( Vzadd < ps_conv ) {
        Vzadd = ps_conv ;
        Vzadd_dVddp = 0.0 ;
    }
    Vddpz = Vddp - 2 * Vzadd ;
    Vddpz_dVddp = 1.0 - 2 * Vzadd_dVddp ;
  } else {
    Fn_SymAdd( Vzadd ,  Vddp / 2 , model->HSMHV2_vzadd0 , T2 ) ;
    Vzadd_dVddp =   T2 / 2.0 ;
    if( Vzadd < ps_conv ) {
        Vzadd = ps_conv ;
        Vzadd_dVddp = 0.0 ;
    }
    Vddpz = Vddp + 2 * Vzadd ;
    Vddpz_dVddp = 1.0 + 2 * Vzadd_dVddp ;
  }

  Edri       = Vddpz / Ldrift ;
  Edri_dVddp = Vddpz_dVddp  / Ldrift ;

  Vdri       = Mu0    * Edri ;
  Vdri_dVddp = Mu0    * Edri_dVddp ;
  Vdri_dT    = Mu0_dT * Edri ;

  /*-----------------------------------------------------------*
   * Mu : mobility
   *-----------------*/
  if ( Vddp >= 0 ) { 
    T1       = Vdri / Vmax ;
    T1_dVddp = Vdri_dVddp / Vmax ;
    T1_dT    = ( Vdri_dT * Vmax -  Vdri * Vmax_dT ) / ( Vmax * Vmax );
  } else  { 
    T1       = - Vdri / Vmax ;
    T1_dVddp = - Vdri_dVddp / Vmax ;
    T1_dT    = - ( Vdri_dT * Vmax - Vdri * Vmax_dT ) / ( Vmax * Vmax );
  }

  if( model->HSMHV2_rdrbbtmp == 0.0 ) {
    if( T1 == 0.0 ) { 
      T2 = 0.0 ; T2_dT = 0.0 ; T2_dVddp = 0.0 ; 
      T4 = 1.0 ; T4_dT = 0.0 ; T4_dVddp = 0.0 ; 
    }else { 
      if ( 1.0e0 - epsm10 <= here->HSMHV2_rdrbb && here->HSMHV2_rdrbb <= 1.0e0 + epsm10 )  { 
        T3 = 1.0e0 ;
        T3_dT    = 0.0e0 ;
        T3_dVddp = 0.0e0 ;
      } else if ( 2.0e0 - epsm10 <= here->HSMHV2_rdrbb && here->HSMHV2_rdrbb <= 2.0e0 + epsm10 )  { 
        T3 = T1 ;
        T3_dT    = T1_dT ;
        T3_dVddp = T1_dVddp ;
      } else  { 
        T3 = Fn_Pow( T1 , here->HSMHV2_rdrbb - 1.0e0 ) ;
        T3_dT    = ( here->HSMHV2_rdrbb - 1.0e0 )* Fn_Pow( T1 , here->HSMHV2_rdrbb - 2.0e0 ) * T1_dT ;
        T3_dVddp = ( here->HSMHV2_rdrbb - 1.0e0 )* Fn_Pow( T1 , here->HSMHV2_rdrbb - 2.0e0 ) * T1_dVddp ;
      }
      T2 = T1 * T3 ;
      T2_dT    = T1 * T3_dT  + T3 * T1_dT ;
      T2_dVddp = T1 * T3_dVddp + T3 * T1_dVddp ;
      T4 = 1.0e0 + T2 ;
      T4_dT    = T2_dT ;
      T4_dVddp = T2_dVddp ;
    } 

    if ( 1.0e0 - epsm10 <= here->HSMHV2_rdrbb && here->HSMHV2_rdrbb <= 1.0e0 + epsm10 )  { 
      T5 = 1.0 / T4 ;
      T5_dT    = - T5 * T5 * T4_dT ; 
      T5_dVddp = - T5 * T5 * T4_dVddp ; 
    } else if ( 2.0e0 - epsm10 <= here->HSMHV2_rdrbb && here->HSMHV2_rdrbb <= 2.0e0 + epsm10 )  { 
      T5 = 1.0 / sqrt( T4 ) ;
      T5_dT    = - 0.5e0 / ( T4 * sqrt(T4) ) * T4_dT ;
      T5_dVddp = - 0.5e0 / ( T4 * sqrt(T4) ) * T4_dVddp;
    } else  { 
      T6 = Fn_Pow( T4 , ( - 1.0e0 / here->HSMHV2_rdrbb - 1.0e0 ) ) ;
      T5 = T4 * T6 ;
      T6_dT    = ( - 1.0e0 / here->HSMHV2_rdrbb - 1.0e0 ) * Fn_Pow( T4 , ( - 1.0e0 / here->HSMHV2_rdrbb - 2.0e0 ) ) * T4_dT ;
      T6_dVddp = ( - 1.0e0 / here->HSMHV2_rdrbb - 1.0e0 ) * Fn_Pow( T4 , ( - 1.0e0 / here->HSMHV2_rdrbb - 2.0e0 ) ) * T4_dVddp ;
      T5_dT    = T4_dT * T6  + T4 * T6_dT ;
      T5_dVddp = T4_dVddp * T6 + T4 * T6_dVddp ;
    }

  } else {
    if( T1 == 0.0 ) {
      T2 = 0.0 ; T2_dT = 0.0 ; T2_dVddp = 0.0 ;
      T4 = 1.0 ; T4_dT = 0.0 ; T4_dVddp = 0.0 ;
    }else {
      T3 = Fn_Pow( T1 , here->HSMHV2_rdrbb - 1.0e0 ) ;
      T3_dT    = ( here->HSMHV2_rdrbb - 1.0e0 )* Fn_Pow( T1 , here->HSMHV2_rdrbb - 2.0e0 ) * T1_dT + T3*log(T1)*Rdrbb_dT ;
      T3_dVddp = ( here->HSMHV2_rdrbb - 1.0e0 )* Fn_Pow( T1 , here->HSMHV2_rdrbb - 2.0e0 ) * T1_dVddp ;
      T2 = T1 * T3 ;
      T2_dT    = T1 * T3_dT  + T3 * T1_dT ;
      T2_dVddp = T1 * T3_dVddp + T3 * T1_dVddp ;
      T4 = 1.0e0 + T2 ;
      T4_dT    = T2_dT ;
      T4_dVddp = T2_dVddp ;
    } 
    T6 = Fn_Pow( T4 , ( - 1.0e0 / here->HSMHV2_rdrbb - 1.0e0 ) ) ;
    T5 = T4 * T6 ;
    T6_dT    = ( - 1.0e0 / here->HSMHV2_rdrbb - 1.0e0 ) * Fn_Pow( T4 , ( - 1.0e0 / here->HSMHV2_rdrbb - 2.0e0 ) ) * T4_dT +T6*log(T4)/here->HSMHV2_rdrbb/here->HSMHV2_rdrbb*Rdrbb_dT ;
    T6_dVddp = ( - 1.0e0 / here->HSMHV2_rdrbb - 1.0e0 ) * Fn_Pow( T4 , ( - 1.0e0 / here->HSMHV2_rdrbb - 2.0e0 ) ) * T4_dVddp ;
    T5_dT    = T4_dT * T6  + T4 * T6_dT ;
    T5_dVddp = T4_dVddp * T6 + T4 * T6_dVddp ;
  }

  Mu       = Mu0 * T5 ;
  Mu_dVddp = Mu0 * T5_dVddp ;
  Mu_dT    = Mu0_dT * T5 + Mu0 * T5_dT ;

  /*-----------------------------------------------------------*
   * Carr : carrier density 
   *-----------------*/

  T4     = 1.0e0 + T1 ;
  T4_dVddp = T1_dVddp ;
  T4_dT    = T1_dT ;

  T5     = 1.0 / T4 ;
  T5_dVddp = - T5 * T5 * T4_dVddp ;
  T5_dT    = - T5 * T5 * T4_dT ;

  Carr       = Nover * ( 1.0 + Car * ( 1.0 - T5 ) * Vddpz /  ( Ldrift - model->HSMHV2_rdrdl2 ) ) ;
  Carr_dVddp = Nover * Car * ( - T5_dVddp * Vddpz + ( 1.0 - T5 ) * Vddpz_dVddp ) / ( Ldrift - model->HSMHV2_rdrdl2 ) ;
  Carr_dT    = Nover * ( Car_dT * ( 1.0 - T5 ) + Car * ( - T5_dT ) ) * Vddpz / ( Ldrift - model->HSMHV2_rdrdl2 ) ;

  Carr      += - here->HSMHV2_QbuLD      / C_QE * model->HSMHV2_rdrqover;
  Carr_dVds  = - here->HSMHV2_QbuLD_dVds / C_QE * model->HSMHV2_rdrqover;
  Carr_dVgs  = - here->HSMHV2_QbuLD_dVgs / C_QE * model->HSMHV2_rdrqover;
  Carr_dVbs  = - here->HSMHV2_QbuLD_dVbs / C_QE * model->HSMHV2_rdrqover;
  Carr_dT   += - here->HSMHV2_QbuLD_dTi  / C_QE * model->HSMHV2_rdrqover;

  /*-----------------------------------------------------------*
   * Xov : depth of the current flow
   *-----------------*/
  T0     = -here->HSMHV2_Ps0LD ;
  T0_dVd = -here->HSMHV2_Ps0LD_dVds ;
  T0_dVg = -here->HSMHV2_Ps0LD_dVgs ;
  T0_dVb = -here->HSMHV2_Ps0LD_dVbs ;
  T0_dT  = -here->HSMHV2_Ps0LD_dTi  ;

  Fn_SZ( T0 , T0 , 1.0e-2 , T9 ) ;
  T0 +=  epsm10 ;
  T0_dVd *= T9 ;
  T0_dVg *= T9 ;
  T0_dVb *= T9 ;
  T0_dT  *= T9 ;

  Wdepl     = sqrt ( here->HSMHV2_kdep * T0 ) ;
  Wdepl_dVd = here->HSMHV2_kdep / ( 2.0 * Wdepl ) * T0_dVd ;
  Wdepl_dVg = here->HSMHV2_kdep / ( 2.0 * Wdepl ) * T0_dVg ;
  Wdepl_dVb = here->HSMHV2_kdep / ( 2.0 * Wdepl ) * T0_dVb ;
  Wdepl_dT  = here->HSMHV2_kdep / ( 2.0 * Wdepl ) * T0_dT  ;

  T2     = Vds - Vbs + model->HSMHV2_vbi ;
  T2_dVd = 1.0  ;
  T2_dVb = -1.0 ;

  Fn_SZ( T2 , T2 , 1.0e-2 , T9 ) ;
  T2 +=  epsm10 ;
  T2_dVd *= T9 ;
  T2_dVb *= T9 ;

  Wjunc0     = sqrt ( here->HSMHV2_kjunc * T2 ) ;
  Wjunc0_dVd = here->HSMHV2_kjunc / ( 2.0 * Wjunc0 ) * T2_dVd ;
  Wjunc0_dVb = here->HSMHV2_kjunc / ( 2.0 * Wjunc0 ) * T2_dVb ;
  Fn_SU( Wjunc, Wjunc0, Xldld, 10e-3*Xldld, T0 );
  Wjunc_dVd = Wjunc0_dVd * T0;
  Wjunc_dVb = Wjunc0_dVb * T0;
//  Wrdrdjunc = model->HSMHV2_rdrdjunc + small ;
  Wrdrdjunc = model->HSMHV2_rdrdjunc + epsm10 ;

 
  Xov      = here->HSMHV2_Xmax - Cx * (  here->HSMHV2_Xmax 
             / Wrdrdjunc * Wdepl +  here->HSMHV2_Xmax / Xldld * Wjunc ) ;
  Xov_dVds = - Cx *  here->HSMHV2_Xmax / Wrdrdjunc * Wdepl_dVd 
             - Cx *  here->HSMHV2_Xmax / Xldld * Wjunc_dVd ;
  Xov_dVgs = - Cx *  here->HSMHV2_Xmax / Wrdrdjunc * Wdepl_dVg ;
  Xov_dVbs = - Cx *  here->HSMHV2_Xmax / Wrdrdjunc * Wdepl_dVb 
             - Cx *  here->HSMHV2_Xmax / Xldld * Wjunc_dVb ;
  Xov_dT   = - Cx_dT * (  here->HSMHV2_Xmax 
             / Wrdrdjunc * Wdepl +  here->HSMHV2_Xmax / Xldld * Wjunc ) 
             - Cx *  here->HSMHV2_Xmax / Wrdrdjunc * Wdepl_dT  ;

  Fn_SZ( Xov , Xov , (1.0 - here->HSMHV2_rdrcx) * here->HSMHV2_Xmax / 100 , T9 ) ;

  Xov_dVds *= T9 ;
  Xov_dVgs *= T9 ;
  Xov_dVbs *= T9 ;
  Xov_dT   *= T9 ;

  /*-----------------------------------------------------------*
   * Rd : drift resistance 
   *-----------------*/
  T0 = C_QE / ( Ldrift + model->HSMHV2_rdrdl1 );
  T1 = T0;
  T1_dVd = 0.0 ;

  GD       = T1 * Xov * Mu * Carr ;
  GD_dVddp = T1 * Xov * Mu_dVddp * Carr
             + T1 * Xov * Mu * Carr_dVddp ;
  GD_dVgse = 0.0 ;
  GD_dT    = T1 * Xov * Mu_dT * Carr
             + T1 * Xov_dT * Mu * Carr
             + T1 * Xov * Mu * Carr_dT ;
  GD_dVds  = T1 * Mu * (Xov_dVds * Carr + Xov * Carr_dVds)
             + T1_dVd * Mu * Xov * Carr;
  GD_dVgs  = T1 * Mu * (Xov_dVgs * Carr + Xov * Carr_dVgs);
  GD_dVbs  = T1 * Mu * (Xov_dVbs * Carr + Xov * Carr_dVbs);

  if ( GD <= 0 ) {
//    GD       = small ;
    GD       = epsm10 ;
    GD_dVddp = 0.0 ;
    GD_dVgse = 0.0 ;
    GD_dT    = 0.0 ;
    GD_dVds  = 0.0 ;
    GD_dVgs  = 0.0 ;
    GD_dVbs  = 0.0 ;
  }

  Rd       = 1 / GD ;
  Rd_dVddp = - GD_dVddp * Rd * Rd ;
  Rd_dVgse = - GD_dVgse * Rd * Rd ;
  Rd_dT    = - GD_dT    * Rd * Rd ;
  Rd_dVds  = - GD_dVds  * Rd * Rd ;
  Rd_dVgs  = - GD_dVgs  * Rd * Rd ;
  Rd_dVbs  = - GD_dVbs  * Rd * Rd ;

  /* Weff dependence of the resistances */
  Rd = Rd  /  WeffLD_nf ;

  Fn_SU_CP( Rd, Rd, 1e6, 1e3, 2, T0 ) ;

  Rd_dVddp = Rd_dVddp*T0/WeffLD_nf ;
  Rd_dVgse = Rd_dVgse*T0/WeffLD_nf ;
  Rd_dT    = Rd_dT*T0/WeffLD_nf ;
  Rd_dVds  = Rd_dVds*T0/WeffLD_nf ;
  Rd_dVgs  = Rd_dVgs*T0/WeffLD_nf ;
  Rd_dVbs  = Rd_dVbs*T0/WeffLD_nf ;

  if ( here->HSMHV2subNode >= 0 &&  
       ( pParam->HSMHV2_nover * ( modelMKS->HSMHV2_nsubsub + pParam->HSMHV2_nover ) ) > 0 ) {
      /* external substrate node exists && LDMOS case: */
      /* Substrate Effect */
    T0 = model->HSMHV2_vbisub - model->HSMHV2_rdvdsub * here->HSMHV2_Vdserevz - model->HSMHV2_rdvsub * here->HSMHV2_Vsubsrev ;

    Fn_SZ( T1, T0, 10.0, T2 ) ;
    T1 += epsm10 ;

    T1_dVdserev  = - model->HSMHV2_rdvdsub * here->HSMHV2_Vdserevz_dVd * T2 ;
    T1_dVsubsrev = - model->HSMHV2_rdvsub  * T2 ;

    T0 = modelMKS->HSMHV2_nsubsub / ( pParam->HSMHV2_nover * ( modelMKS->HSMHV2_nsubsub + pParam->HSMHV2_nover ) ) ;

    T4 = 2 * C_ESI / C_QE * T0 ;
    Wdep = sqrt ( T4 * T1 ) + small ;

    Wdep_dVdserev =  0.5 * T4 * T1_dVdserev  / Wdep ;
    Wdep_dVsubsrev = 0.5 * T4 * T1_dVsubsrev / Wdep ;

    Fn_SU( Wdep, Wdep, model->HSMHV2_ddrift, C_sub_delta * model->HSMHV2_ddrift, T0 ) ;
    Wdep_dVdserev *=  T0 ;
    Wdep_dVsubsrev *= T0 ;

    T0 = model->HSMHV2_ddrift - Wdep ;
    Fn_SZ( T0, T0, C_sub_delta2, T2 ) ;
    T0 += epsm10;

    T6 = (here->HSMHV2_ldrift1 +  here->HSMHV2_ldrift2 ) / T0 ; 
    T6_dVdserev =  T2 * Wdep_dVdserev  * T6 / T0 ;
    T6_dVsubsrev = T2 * Wdep_dVsubsrev * T6 / T0 ;

    T0 = Rd ;
    Rd        = T0 * T6 ;
    Rd_dVddp  = Rd_dVddp * T6 ;
    Rd_dVgse  = Rd_dVgse * T6 ;
    Rd_dVdse  = T0 * T6_dVdserev ;
    Rd_dVbse  = Rd_dVgse * T6 ;

    Rd_dVds   = Rd_dVds * T6 ;
    Rd_dVgs   = Rd_dVgs * T6 ;
    Rd_dVbs   = Rd_dVbs * T6 ;
    Rd_dVsubs = T0 * T6_dVsubsrev ;
    Rd_dT     = Rd_dT    * T6 ;
 
  }


  /* Sheet resistances are added. */
  Rd += here->HSMHV2_rd0 ;

  /* Re-stamps for hsmhvnoi.c */
  /* Please see hsmhvnoi.c */
  if ( Rd > Res_min && model->HSMHV2_cothrml ) 
       here->HSMHV2drainConductance = Mfactor / Rd ;
  else here->HSMHV2drainConductance = 0.0 ;
  if ( here->HSMHV2_Rs > Res_min && model->HSMHV2_cothrml ) 
       here->HSMHV2sourceConductance = Mfactor / here->HSMHV2_rs0 ;
  else here->HSMHV2sourceConductance = 0.0 ;

  /* Clamping to Res_min */
  here->HSMHV2_Rs = here->HSMHV2_rs0 / Mfactor ; 
  if(here->HSMHV2_Rs < Res_min) { here->HSMHV2_Rs = Res_min ; }
  here->HSMHV2_dRs_dVdse  = 0.0 ;
  here->HSMHV2_dRs_dVgse  = 0.0 ;
  here->HSMHV2_dRs_dVbse  = 0.0 ;
  here->HSMHV2_dRs_dVsubs = 0.0 ;
  here->HSMHV2_dRs_dTi = 0.0 ;


    /* Clamping to Res_min */
    here->HSMHV2_Rd = Rd / Mfactor ;
    if(here->HSMHV2_Rd < Res_min) {
      here->HSMHV2_Rd = Res_min ;
      here->HSMHV2_dRd_dVddp  = 0.0 ;
      here->HSMHV2_dRd_dVdse  = 0.0 ;
      here->HSMHV2_dRd_dVgse  = 0.0 ;
      here->HSMHV2_dRd_dVbse  = 0.0 ;
      here->HSMHV2_dRd_dVsubs = 0.0 ;
      here->HSMHV2_dRd_dTi    = 0.0 ;
      here->HSMHV2_dRd_dVds   = 0.0 ;
      here->HSMHV2_dRd_dVgs   = 0.0 ;
      here->HSMHV2_dRd_dVbs   = 0.0 ;
    } else {
      here->HSMHV2_dRd_dVddp = Rd_dVddp / Mfactor ;
      here->HSMHV2_dRd_dVdse = Rd_dVdse / Mfactor ;
      here->HSMHV2_dRd_dVgse = Rd_dVgse / Mfactor ;
      here->HSMHV2_dRd_dVbse = Rd_dVbse / Mfactor ;
      here->HSMHV2_dRd_dVsubs= Rd_dVsubs / Mfactor ;
      here->HSMHV2_dRd_dTi   = Rd_dT / Mfactor ;
      here->HSMHV2_dRd_dVds  = Rd_dVds / Mfactor ;
      here->HSMHV2_dRd_dVgs  = Rd_dVgs / Mfactor ;
      here->HSMHV2_dRd_dVbs  = Rd_dVbs / Mfactor ;
    }


  return ( HiSIM_OK ) ;

}
