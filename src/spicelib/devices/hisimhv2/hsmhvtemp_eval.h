/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 2  SUBVERSION : 2  REVISION : 0 ) 
 Model Parameter 'VERSION' : 2.20
 FILE : hsmhvtemp_eval.h

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

#define C_rdtemp_min 5.0e-3
#define C_rdtemp_dlt 1.0e-2

    TTEMP = ckt->CKTtemp;
    if ( here->HSMHV2_dtemp_Given ) { TTEMP = TTEMP + here->HSMHV2_dtemp ; }
    TTEMP0 = TTEMP ; 
#ifdef HSMHV2EVAL
    /* Self heating */
    TTEMP = TTEMP + deltemp ; 
#endif
Tdiff0 =   TTEMP0 - model->HSMHV2_ktnom ;
Tdiff0_2 = TTEMP0 * TTEMP0 - model->HSMHV2_ktnom * model->HSMHV2_ktnom ;
Tdiff =    TTEMP  - model->HSMHV2_ktnom ;
Tdiff_2 =  TTEMP  * TTEMP  - model->HSMHV2_ktnom * model->HSMHV2_ktnom ;
        here->HSMHV2_Tratio = TTEMP / model->HSMHV2_ktnom ;

        /* Band gap */
        here->HSMHV2_eg = Eg = here->HSMHV2_egtnom - pParam->HSMHV2_bgtmp1 * Tdiff
          - pParam->HSMHV2_bgtmp2 * Tdiff_2 ;
        here->HSMHV2_sqrt_eg = sqrt( Eg ) ;
#ifdef HSMHV2EVAL
        Eg_dT = -pParam->HSMHV2_bgtmp1 - 2.0e0 * TTEMP * pParam->HSMHV2_bgtmp2 ;
#endif

	T1 = 1.0 / TTEMP ;
	T2 = 1.0 / model->HSMHV2_ktnom ;
	T3 = here->HSMHV2_egtnom + model->HSMHV2_egig
	  + model->HSMHV2_igtemp2 * ( T1 - T2 )
	  + model->HSMHV2_igtemp3 * ( T1 * T1 - T2 * T2 ) ;
 	here->HSMHV2_egp12 = sqrt ( T3 ) ;
 	here->HSMHV2_egp32 = T3 * here->HSMHV2_egp12 ;

        
        /* Inverse of the thermal voltage */
        here->HSMHV2_beta = beta = C_QE / (C_KB * TTEMP) ;
        here->HSMHV2_beta_inv = 1.0 / beta ;
        here->HSMHV2_beta2 = beta * beta ;
        here->HSMHV2_betatnom = C_QE / (C_KB * model->HSMHV2_ktnom) ;
#ifdef HSMHV2EVAL
        beta_dT=-C_QE/(C_KB*TTEMP*TTEMP);
        beta_inv_dT = C_KB / C_QE ;
#endif

        /* Intrinsic carrier concentration */
        here->HSMHV2_nin = Nin = C_Nin0 * Fn_Pow (here->HSMHV2_Tratio, 1.5e0) 
          * exp (- Eg / 2.0e0 * beta + here->HSMHV2_egtnom / 2.0e0 * here->HSMHV2_betatnom) ;
#ifdef HSMHV2EVAL
        Nin_dT = C_Nin0 * exp (- Eg / 2.0e0 * beta + here->HSMHV2_egtnom / 2.0e0 * here->HSMHV2_betatnom)
          * 1.5e0 * Fn_Pow ( here->HSMHV2_Tratio , 0.5e0 ) / model->HSMHV2_ktnom 
          + C_Nin0 * Fn_Pow (here->HSMHV2_Tratio, 1.5e0) 
          * exp (- Eg / 2.0e0 * beta + here->HSMHV2_egtnom / 2.0e0 * here->HSMHV2_betatnom)
          * ( - Eg / 2.0e0 * beta_dT - beta / 2.0e0 * Eg_dT );
#endif

        /* Phonon Scattering (temperature-dependent part) */
        T1 =  Fn_Pow (here->HSMHV2_Tratio, pParam->HSMHV2_muetmp) ;
        here->HSMHV2_mphn0 = T1 / here->HSMHV2_mueph ;
        here->HSMHV2_mphn1 = here->HSMHV2_mphn0 * model->HSMHV2_mueph0 ;
#ifdef HSMHV2EVAL
        T1_dT = pParam->HSMHV2_muetmp * Fn_Pow(here->HSMHV2_Tratio, pParam->HSMHV2_muetmp - 1.0 )
          / model->HSMHV2_ktnom ;
        mphn0_dT = T1_dT / here->HSMHV2_mueph ;
#endif

        if( model->HSMHV2_codep == 1 ) {
        /* depletion MOS parameter (temperature-dependent part) */
          here->HSMHV2_Pb2n = 2.0/beta*log(here->HSMHV2_ndepm/Nin) ;
          here->HSMHV2_Vbipn = 1.0/beta*log(here->HSMHV2_ndepm*here->HSMHV2_nsub/Nin/Nin) ;
          here->HSMHV2_cnst0 = sqrt ( 2.0 * C_ESI * C_QE * here->HSMHV2_ndepm / beta ) ;
          here->HSMHV2_cnst1 = Nin*Nin/here->HSMHV2_ndepm/here->HSMHV2_ndepm ;
          T1 =  Fn_Pow (here->HSMHV2_Tratio, model->HSMHV2_depmuetmp) ;
          here->HSMHV2_depmphn0 = T1 / model->HSMHV2_depmueph1 ;
          here->HSMHV2_depmphn1 = here->HSMHV2_depmphn0 * model->HSMHV2_depmueph0 ;

          T0 = 1.8 + 0.4 * here->HSMHV2_Tratio + 0.1 * here->HSMHV2_Tratio * here->HSMHV2_Tratio - model->HSMHV2_depvtmp * ( 1.0 - here->HSMHV2_Tratio ) ;
          here->HSMHV2_depvmax = modelMKS->HSMHV2_depvmax / T0 ;

#ifdef HSMHV2EVAL
          Pb2n_dT = -here->HSMHV2_Pb2n/beta*beta_dT-2.0/beta/Nin*Nin_dT ;
          Vbipn_dT = -here->HSMHV2_Vbipn/beta*beta_dT-2/beta/Nin*Nin_dT ;
          cnst0_dT = 0.5e0 / here->HSMHV2_cnst0 * 2.0 * C_ESI * C_QE * here->HSMHV2_ndepm * beta_inv_dT ;
          cnst1_dT = 2.0e0 * Nin * Nin_dT / here->HSMHV2_ndepm / here->HSMHV2_ndepm ;
          T1_dT = model->HSMHV2_depmuetmp * Fn_Pow(here->HSMHV2_Tratio, model->HSMHV2_depmuetmp - 1.0 ) 
          / model->HSMHV2_ktnom ;
          depmphn0_dT = T1_dT / model->HSMHV2_depmueph1 ;
          T0_dT = 1 / model->HSMHV2_ktnom * ( 0.4 + 0.2 * here->HSMHV2_Tratio + model->HSMHV2_depvtmp ) ;
          depVmax_dT = - modelMKS->HSMHV2_depvmax / ( T0 * T0 ) * T0_dT ;

#endif
        }
 
        /* Pocket Overlap (temperature-dependent part) */
        here->HSMHV2_ptovr = here->HSMHV2_ptovr0 / beta ;
#ifdef HSMHV2EVAL
        ptovr_dT =  here->HSMHV2_ptovr0 * beta_inv_dT ; 
#endif

        /* Velocity Temperature Dependence */
        T1 = TTEMP  / model->HSMHV2_ktnom ;
        T0 = 1.8 + 0.4 * T1 + 0.1 * T1 * T1 - pParam->HSMHV2_vtmp * (1.0 - T1) ;
	 if ( model->HSMHV2_cotemp != 2 ) { /* without deltemp (COTEMP=0,1,3) */
        here->HSMHV2_vmax = here->HSMHV2_vmax0 * pParam->HSMHV2_vmax
          / T0
          * ( 1.0 + model->HSMHV2_vmaxt1 * Tdiff0 + model->HSMHV2_vmaxt2 * Tdiff0_2 ) ;
#ifdef HSMHV2EVAL
        Vmax_dT=-here->HSMHV2_vmax0 * pParam->HSMHV2_vmax 
          / ( T0 * T0 ) * ( 1.0 + model->HSMHV2_vmaxt1 * Tdiff0 + model->HSMHV2_vmaxt2 * Tdiff0_2 )
          * 1/model->HSMHV2_ktnom * (0.4 + 0.2 * T1 + pParam->HSMHV2_vtmp) ;
#endif
	 } else { /* with deltemp (COTEMP=2) */
	   here->HSMHV2_vmax = here->HSMHV2_vmax0 * pParam->HSMHV2_vmax
	     / T0 
	     * ( 1.0 + model->HSMHV2_vmaxt1 * Tdiff + model->HSMHV2_vmaxt2 * Tdiff_2 ) ;
#ifdef HSMHV2EVAL
	   /* under development */
	   Vmax_dT = here->HSMHV2_vmax0 * pParam->HSMHV2_vmax 
	     / ( T0 * T0 )
	     * ( ( model->HSMHV2_vmaxt1 + 2.0 * TTEMP * model->HSMHV2_vmaxt2 ) * T0 
		- ( 1.0 + model->HSMHV2_vmaxt1 * Tdiff + model->HSMHV2_vmaxt2 * Tdiff_2 )
		* 1/model->HSMHV2_ktnom * (0.4 + 0.2 * T1 + pParam->HSMHV2_vtmp) ) ;
#endif
	 }
	 if ( model->HSMHV2_cotemp != 2 ) { /* without deltemp (COTEMP=0,1,3) */
        here->HSMHV2_ninvd = here->HSMHV2_ninvd0 * ( 1.0 + model->HSMHV2_ninvdt1 * Tdiff0 + model->HSMHV2_ninvdt2 * Tdiff0_2 ) ;
#ifdef HSMHV2EVAL
	ninvd_dT = 0.0 ;
#endif
	 } else { /* with deltemp (COTEMP=2) */
	   /* under development */
	   here->HSMHV2_ninvd = here->HSMHV2_ninvd0 * ( 1.0 + model->HSMHV2_ninvdt1 * Tdiff + model->HSMHV2_ninvdt2 * Tdiff_2 ) ;
#ifdef HSMHV2EVAL
	   ninvd_dT = here->HSMHV2_ninvd0 * ( model->HSMHV2_ninvdt1 + 2.0 * TTEMP * model->HSMHV2_ninvdt2 ) ;
#endif
	 }
      
	/* Temperature Dependence of RTH0 */
	pParam->HSMHV2_rth = ( pParam->HSMHV2_rth0 + model->HSMHV2_rthtemp1 * Tdiff0 + model->HSMHV2_rthtemp2 * Tdiff0_2  ) * here->HSMHV2_rthtemp0 ;


	/* Temperature Dependence of POWRAT */
        T2 = pParam->HSMHV2_powrat + model->HSMHV2_prattemp1 * Tdiff0 + model->HSMHV2_prattemp2 * Tdiff0_2  ;
	Fn_SL( T2 , T2 , 0 , 0.05 , T0 );
	Fn_SU( here->HSMHV2_powratio , T2 , 1 , 0.05 , T0 );


        /* 2 phi_B (temperature-dependent) */
        /* @temp, with pocket */
        here->HSMHV2_pb2 =  2.0e0 / beta * log (here->HSMHV2_nsub / Nin) ;
#ifdef HSMHV2EVAL
        Pb2_dT = - (here->HSMHV2_pb2 * beta_dT  + 2.0e0 / Nin * Nin_dT ) / beta ;
#endif

        /* Depletion Width */
        T1 = 2.0e0 * C_ESI / C_QE ;
        here->HSMHV2_wdpl = sqrt ( T1 / here->HSMHV2_nsub ) ;
        here->HSMHV2_wdplp = sqrt( T1 / ( here->HSMHV2_nsubp ) ) ; 

        
        if( model->HSMHV2_codep == 0  ) {
          /* Coefficient of the F function for bulk charge */
          here->HSMHV2_cnst0 = sqrt ( 2.0 * C_ESI * C_QE * here->HSMHV2_nsub / beta ) ;

          /* cnst1: n_{p0} / p_{p0} */
          T1 = Nin / here->HSMHV2_nsub ;
          here->HSMHV2_cnst1 = T1 * T1 ;
#ifdef HSMHV2EVAL
          cnst0_dT = 0.5e0 / here->HSMHV2_cnst0 * 2.0 * C_ESI * C_QE * here->HSMHV2_nsub * beta_inv_dT ;
          cnst1_dT = 2.0e0 * Nin * Nin_dT / here->HSMHV2_nsub / here->HSMHV2_nsub ;
#endif
        }
        

        if( model->HSMHV2_codep == 0  ) {

          if ( pParam->HSMHV2_nover != 0.0 ) {
	    here->HSMHV2_cnst0over = here->HSMHV2_cnst0 * sqrt( pParam->HSMHV2_nover / here->HSMHV2_nsub ) ;     
#ifdef HSMHV2EVAL
           cnst0over_dT = cnst0_dT * sqrt( pParam->HSMHV2_nover / here->HSMHV2_nsub ) ; 
#endif
          }
          if ( pParam->HSMHV2_novers != 0.0 ) {
	    here->HSMHV2_cnst0overs = here->HSMHV2_cnst0 * sqrt( pParam->HSMHV2_novers / here->HSMHV2_nsub ) ;     
#ifdef HSMHV2EVAL
            cnst0overs_dT = cnst0_dT * sqrt( pParam->HSMHV2_novers / here->HSMHV2_nsub ) ;
#endif
          }
        } else {
          if ( pParam->HSMHV2_nover != 0.0 ) {
            here->HSMHV2_cnst0over = here->HSMHV2_cnst0 * sqrt( pParam->HSMHV2_nover / here->HSMHV2_ndepm ) ;
#ifdef HSMHV2EVAL
           cnst0over_dT = cnst0_dT * sqrt( pParam->HSMHV2_nover / here->HSMHV2_ndepm ) ;
#endif
          }
          if ( pParam->HSMHV2_novers != 0.0 ) {
            here->HSMHV2_cnst0overs = here->HSMHV2_cnst0 * sqrt( pParam->HSMHV2_novers / here->HSMHV2_ndepm ) ;
#ifdef HSMHV2EVAL
            cnst0overs_dT = cnst0_dT * sqrt( pParam->HSMHV2_novers / here->HSMHV2_ndepm ) ;
#endif
          }

        }


	/* temperature-dependent resistance model */
	/* drain side */
	if ( pParam->HSMHV2_rd > 0.0 ) {
         T2 = here->HSMHV2_rdtemp0
	   * ( here->HSMHV2_ldrift1 * pParam->HSMHV2_rdslp1 * C_m2um   + pParam->HSMHV2_rdict1 )
	   * ( here->HSMHV2_ldrift2 * model->HSMHV2_rdslp2 * C_m2um   + model->HSMHV2_rdict2 ) ;

	 if ( model->HSMHV2_cotemp == 1 ) { /* without deltemp (COTEMP=1) */
	 here->HSMHV2_rd = ( pParam->HSMHV2_rd + modelMKS->HSMHV2_rdtemp1 * Tdiff0 + modelMKS->HSMHV2_rdtemp2 * Tdiff0_2 ) * T2 ;
	 Fn_SL( here->HSMHV2_rd, here->HSMHV2_rd, C_rdtemp_min * pParam->HSMHV2_rd, C_rdtemp_dlt * pParam->HSMHV2_rd, T0 );
#ifdef HSMHV2EVAL
	 Rd0_dT = 0.0 ;
#endif
	 } else { /* with deltemp (COTEMP=0,2,3) */
	   here->HSMHV2_rd = ( pParam->HSMHV2_rd + modelMKS->HSMHV2_rdtemp1 * Tdiff + modelMKS->HSMHV2_rdtemp2 * Tdiff_2 ) * T2 ;
	   Fn_SL( here->HSMHV2_rd, here->HSMHV2_rd, C_rdtemp_min * pParam->HSMHV2_rd, C_rdtemp_dlt * pParam->HSMHV2_rd, T0 );
#ifdef HSMHV2EVAL
	   Rd0_dT = ( modelMKS->HSMHV2_rdtemp1 + 2.0 * TTEMP * modelMKS->HSMHV2_rdtemp2 ) * T2 * T0 ;
#endif
	 }

	} else {
	  here->HSMHV2_rd = 0.0 ;
	}
	/* source side (asymmetric case) */
	if ( pParam->HSMHV2_rs > 0.0 ) {
	    T2 = here->HSMHV2_rdtemp0
	      * ( here->HSMHV2_ldrift1s * pParam->HSMHV2_rdslp1 * C_m2um   + pParam->HSMHV2_rdict1 ) 
	      * ( here->HSMHV2_ldrift2s * model->HSMHV2_rdslp2 * C_m2um   + model->HSMHV2_rdict2 ) ;
	    
	 if ( model->HSMHV2_cotemp == 1 ) { /* without deltemp (COTEMP=1) */
	    here->HSMHV2_rs = ( pParam->HSMHV2_rs + modelMKS->HSMHV2_rdtemp1 * Tdiff0 + modelMKS->HSMHV2_rdtemp2 * Tdiff0_2 ) * T2 ;
	    Fn_SL( here->HSMHV2_rs, here->HSMHV2_rs, C_rdtemp_min * pParam->HSMHV2_rs, C_rdtemp_dlt * pParam->HSMHV2_rs, T0 );
#ifdef HSMHV2EVAL
	    Rs0_dT = 0.0 ;
#endif
	 } else { /* with deltemp (COTEMP=0,2,3) */
	    here->HSMHV2_rs = ( pParam->HSMHV2_rs + modelMKS->HSMHV2_rdtemp1 * Tdiff + modelMKS->HSMHV2_rdtemp2 * Tdiff_2 ) * T2 ;
	    Fn_SL( here->HSMHV2_rs, here->HSMHV2_rs, C_rdtemp_min * pParam->HSMHV2_rs, C_rdtemp_dlt * pParam->HSMHV2_rs, T0 );
#ifdef HSMHV2EVAL
	    Rs0_dT = ( modelMKS->HSMHV2_rdtemp1 + 2.0 * TTEMP * modelMKS->HSMHV2_rdtemp2 ) * T2 * T0 ;
#endif
	 }

	} else {
	  here->HSMHV2_rs = 0.0 ;
	}
        if ( pParam->HSMHV2_rdvd > 0.0 ) {
          T4 = here->HSMHV2_rdvdtemp0 * ( here->HSMHV2_ldrift1 * pParam->HSMHV2_rdslp1 * C_m2um  + pParam->HSMHV2_rdict1 )
	                              * ( here->HSMHV2_ldrift2 * model->HSMHV2_rdslp2 * C_m2um  + model->HSMHV2_rdict2 ) ;

	  T1 = ( 1 -  pParam->HSMHV2_rdov13 ) * here->HSMHV2_loverld * C_m2um ; 
	  T0 = - model->HSMHV2_rdov11 / ( model->HSMHV2_rdov12 + small ) ;
	  T3 = ( T0 * here->HSMHV2_loverld * C_m2um  + 1.0 + model->HSMHV2_rdov11 ) ;
	  Fn_SL( T5 , T3 * T4 , T4 , 10.0e-3 , T6 ) ;
	  Fn_SU( T7  , T5    , T4 * ( model->HSMHV2_rdov11 + 1.0) , 50.0e-6 , T6 ) ;
	  Fn_SL( T2  , T7 + T1 * T4  , 0, 50.0e-6 , T6 ) ;

	 if ( model->HSMHV2_cotemp == 0 || model->HSMHV2_cotemp == 1 ) { /* without deltemp (COTEMP=0,1) */
	  here->HSMHV2_rdvd = ( pParam->HSMHV2_rdvd + modelMKS->HSMHV2_rdvdtemp1 * Tdiff0 + modelMKS->HSMHV2_rdvdtemp2 * Tdiff0_2 ) * T2 ;
	  Fn_SL( here->HSMHV2_rdvd, here->HSMHV2_rdvd, C_rdtemp_min * pParam->HSMHV2_rdvd, C_rdtemp_dlt * pParam->HSMHV2_rdvd, T0 );
#ifdef HSMHV2EVAL
          Rdvd_dT = 0.0 ;
#endif
	 } else { /* with deltemp (COTEMP=2,3) */
	   here->HSMHV2_rdvd = ( pParam->HSMHV2_rdvd + modelMKS->HSMHV2_rdvdtemp1 * Tdiff + modelMKS->HSMHV2_rdvdtemp2 * Tdiff_2 ) * T2 ;
	   Fn_SL( here->HSMHV2_rdvd, here->HSMHV2_rdvd, C_rdtemp_min * pParam->HSMHV2_rdvd, C_rdtemp_dlt * pParam->HSMHV2_rdvd, T0 );
#ifdef HSMHV2EVAL
	   Rdvd_dT = ( modelMKS->HSMHV2_rdvdtemp1 + 2.0 * TTEMP * modelMKS->HSMHV2_rdvdtemp2 ) * T2 * T0 ;
#endif
	 }

          T4 = here->HSMHV2_rdvdtemp0 * ( here->HSMHV2_ldrift1s * pParam->HSMHV2_rdslp1 * C_m2um  + pParam->HSMHV2_rdict1 )
	                              * ( here->HSMHV2_ldrift2s * model->HSMHV2_rdslp2 * C_m2um  + model->HSMHV2_rdict2 ) ;

	  T1 = ( 1 -  pParam->HSMHV2_rdov13 ) * here->HSMHV2_lovers * C_m2um ; 
	  T0 = - model->HSMHV2_rdov11 / ( model->HSMHV2_rdov12 + small ) ;
	  T3 = ( T0 * here->HSMHV2_lovers * C_m2um + 1.0 + model->HSMHV2_rdov11 ) ;
	  Fn_SL( T5 , T3 * T4 , T4 , 10.0e-3 , T6 ) ;
	  Fn_SU( T7  , T5    , T4 * ( model->HSMHV2_rdov11 + 1.0) , 50.0e-6 , T6 ) ;
	  Fn_SL( T2  , T7 + T1 * T4  , 0, 50.0e-6 , T6 ) ;

	 if ( model->HSMHV2_cotemp == 0 || model->HSMHV2_cotemp == 1 ) { /* without deltemp (COTEMP=0,1) */
	  here->HSMHV2_rsvd = ( pParam->HSMHV2_rdvd + modelMKS->HSMHV2_rdvdtemp1 * Tdiff0 + modelMKS->HSMHV2_rdvdtemp2 * Tdiff0_2 ) * T2 ;
	  Fn_SL( here->HSMHV2_rsvd, here->HSMHV2_rsvd, C_rdtemp_min * pParam->HSMHV2_rdvd, C_rdtemp_dlt * pParam->HSMHV2_rdvd, T0 );
#ifdef HSMHV2EVAL
          Rsvd_dT = 0.0 ;
#endif
	 } else { /* with deltemp (COTEMP=2,3) */
	   here->HSMHV2_rsvd = ( pParam->HSMHV2_rdvd + modelMKS->HSMHV2_rdvdtemp1 * Tdiff + modelMKS->HSMHV2_rdvdtemp2 * Tdiff_2 ) * T2 ;
	   Fn_SL( here->HSMHV2_rsvd, here->HSMHV2_rsvd, C_rdtemp_min * pParam->HSMHV2_rdvd, C_rdtemp_dlt * pParam->HSMHV2_rdvd, T0 );
#ifdef HSMHV2EVAL
	   Rsvd_dT = ( modelMKS->HSMHV2_rdvdtemp1 + 2.0 * TTEMP * modelMKS->HSMHV2_rdvdtemp2 ) * T2 * T0 ;
#endif
	 }
	} else {
	  here->HSMHV2_rdvd = 0.0 ;
	  here->HSMHV2_rsvd = 0.0 ;
	}

      
      

       /* costi0 and costi1 for STI transistor model (temperature-dependent part) */
       here->HSMHV2_costi0 = here->HSMHV2_costi00 * sqrt(here->HSMHV2_beta_inv) ;
       here->HSMHV2_costi0_p2 = here->HSMHV2_costi0 * here->HSMHV2_costi0 ;
       here->HSMHV2_costi1 = here->HSMHV2_nin * here->HSMHV2_nin * here->HSMHV2_nsti_p2 ;

/* end of HSMHV2temp_eval.h */
