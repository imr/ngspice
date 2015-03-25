/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 2  SUBVERSION : 2  REVISION : 0 ) 
 Model Parameter 'VERSION' : 2.20
 FILE : hsmhvtemp_eval_dio.h

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

#define small 1.0e-50

        TTEMP = ckt->CKTtemp;
        if ( here->HSMHV_dtemp_Given ) { TTEMP = TTEMP + here->HSMHV_dtemp ; }
        TTEMP0 = TTEMP ; 
#ifdef HSMHVEVAL
        /* Self heating */
        TTEMP = TTEMP + deltemp ; 
#endif


        /* Band gap */
        T1 = TTEMP - model->HSMHV_ktnom ;
        T2 = TTEMP * TTEMP - model->HSMHV_ktnom * model->HSMHV_ktnom ;
        Eg = here->HSMHV_egtnom - pParam->HSMHV_bgtmp1 * T1
          - pParam->HSMHV_bgtmp2 * T2 ;
#ifdef HSMHVEVAL
        Eg_dT = -pParam->HSMHV_bgtmp1 - 2.0e0 * TTEMP * pParam->HSMHV_bgtmp2 ;
#endif


        /* Inverse of the thermal voltage */
        here->HSMHV_beta = beta = C_QE / (C_KB * TTEMP) ;
        here->HSMHV_beta_inv = 1.0 / beta ;
        here->HSMHV_beta2 = beta * beta ;
        here->HSMHV_betatnom = C_QE / (C_KB * model->HSMHV_ktnom) ;
#ifdef HSMHVEVAL
        beta_dT=-C_QE/(C_KB*TTEMP*TTEMP);
        beta_inv_dT = C_KB / C_QE ;
#endif


        log_Tratio = log (here->HSMHV_Tratio) ;
        /* for substrate-drain junction diode. */
        js   = pParam->HSMHV_js0d 
          * exp ((here->HSMHV_egtnom * here->HSMHV_betatnom - Eg * beta
                  + model->HSMHV_xtid * log_Tratio) / pParam->HSMHV_njd) ;
        jssw = pParam->HSMHV_js0swd  
          * exp ((here->HSMHV_egtnom * here->HSMHV_betatnom - Eg * beta 
                  + model->HSMHV_xtid * log_Tratio) / model->HSMHV_njswd) ;

        js2  = pParam->HSMHV_js0d 
          * exp ((here->HSMHV_egtnom * here->HSMHV_betatnom - Eg * beta
                  + model->HSMHV_xti2d * log_Tratio) / pParam->HSMHV_njd) ;  
        jssw2 = pParam->HSMHV_js0swd  
          * exp ((here->HSMHV_egtnom * here->HSMHV_betatnom - Eg * beta
                  + model->HSMHV_xti2d * log_Tratio) / model->HSMHV_njswd) ; 
      
#ifdef HSMHVEVAL
	T0 = - Eg * beta_dT - Eg_dT * beta ; /* Self heating */
	T1 = T0 + model->HSMHV_xtid  / TTEMP ; /* Self heating */
	T2 = T0 + model->HSMHV_xti2d / TTEMP ; /* Self heating */

	js_dT =    js    * T1  / pParam->HSMHV_njd; /* Self heating */
	jssw_dT =  jssw  * T1/ model->HSMHV_njswd ; /* Self heating */
	js2_dT =   js2   * T2  / pParam->HSMHV_njd; /* Self heating */
	jssw2_dT = jssw2 * T2 / model->HSMHV_njswd; /* Self heating */
#endif
      
        here->HSMHV_isbd = here->HSMHV_ad * js + here->HSMHV_pd * jssw ;
        here->HSMHV_isbd2 = here->HSMHV_ad * js2 + here->HSMHV_pd * jssw2 ;

#ifdef HSMHVEVAL
	isbd_dT =  here->HSMHV_ad * js_dT  + here->HSMHV_pd * jssw_dT  ; /* Self heating */
	isbd2_dT = here->HSMHV_ad * js2_dT + here->HSMHV_pd * jssw2_dT ; /* Self heating */
#endif


        T0 = here->HSMHV_Tratio * here->HSMHV_Tratio ;
        T2 = here->HSMHV_isbd + small ;
#ifdef HSMHVEVAL
        T1_dT = 1.0 / model->HSMHV_ktnom ; /* Self heating */
        T0_dT = 2.0 * here->HSMHV_Tratio * T1_dT ;       /* Self heating */
        T2_dT = isbd_dT ;                /* Self heating */
#endif

        here->HSMHV_vbdt = pParam->HSMHV_njd / beta 
          * log ( pParam->HSMHV_vdiffjd * T0 / T2 + 1.0 ) ;

        here->HSMHV_exptempd = exp (( here->HSMHV_Tratio - 1.0 ) * model->HSMHV_ctempd ) ;

#ifdef HSMHVEVAL
	vbdt_dT = - beta_dT / beta * here->HSMHV_vbdt
	  + pParam->HSMHV_njd / beta * pParam->HSMHV_vdiffjd / ( pParam->HSMHV_vdiffjd * T0 / T2 + 1.0 ) 
	  * ( T0_dT / T2 - T0 / T2 / T2 * T2_dT ) ; /* Self heating */
#endif

        here->HSMHV_jd_nvtm_invd = 1.0 / ( pParam->HSMHV_njd / beta ) ;
        here->HSMHV_jd_expcd = exp (here->HSMHV_vbdt * here->HSMHV_jd_nvtm_invd ) ;

#ifdef HSMHVEVAL
        exptempd_dT = model->HSMHV_ctempd / model->HSMHV_ktnom * here->HSMHV_exptempd ;       /* Self heating */
        jd_nvtm_invd_dT = beta_dT / pParam->HSMHV_njd ;                                  /* Self heating */
        jd_expcd_dT = here->HSMHV_jd_expcd
         * ( vbdt_dT * here->HSMHV_jd_nvtm_invd + here->HSMHV_vbdt * jd_nvtm_invd_dT ) ; /* Self heating */
#endif


        /* for substrate-source junction diode. */
        js   = pParam->HSMHV_js0s 
          * exp ((here->HSMHV_egtnom * here->HSMHV_betatnom - Eg * beta
                  + model->HSMHV_xtis * log_Tratio) / pParam->HSMHV_njs) ;
        jssw = pParam->HSMHV_js0sws  
          * exp ((here->HSMHV_egtnom * here->HSMHV_betatnom - Eg * beta 
                  + model->HSMHV_xtis * log_Tratio) / model->HSMHV_njsws) ;

        js2  = pParam->HSMHV_js0s 
          * exp ((here->HSMHV_egtnom * here->HSMHV_betatnom - Eg * beta
                  + model->HSMHV_xti2s * log_Tratio) / pParam->HSMHV_njs) ;  
        jssw2 = pParam->HSMHV_js0sws  
          * exp ((here->HSMHV_egtnom * here->HSMHV_betatnom - Eg * beta
                  + model->HSMHV_xti2s * log_Tratio) / model->HSMHV_njsws) ; 
      
#ifdef HSMHVEVAL
	T0 = - Eg * beta_dT - Eg_dT * beta ; /* Self heating */
	T1 = T0 + model->HSMHV_xtis  / TTEMP ; /* Self heating */
	T2 = T0 + model->HSMHV_xti2s / TTEMP ; /* Self heating */

	js_dT =    js    * T1  / pParam->HSMHV_njs; /* Self heating */
	jssw_dT =  jssw  * T1/ model->HSMHV_njsws ; /* Self heating */
	js2_dT =   js2   * T2  / pParam->HSMHV_njs; /* Self heating */
	jssw2_dT = jssw2 * T2 / model->HSMHV_njsws; /* Self heating */
#endif
      
        here->HSMHV_isbs = here->HSMHV_as * js + here->HSMHV_ps * jssw ;
        here->HSMHV_isbs2 = here->HSMHV_as * js2 + here->HSMHV_ps * jssw2 ;

#ifdef HSMHVEVAL
	isbs_dT =  here->HSMHV_as * js_dT  + here->HSMHV_ps * jssw_dT  ; /* Self heating */
	isbs2_dT = here->HSMHV_as * js2_dT + here->HSMHV_ps * jssw2_dT ; /* Self heating */
#endif


        T0 = here->HSMHV_Tratio * here->HSMHV_Tratio ;
        T3 = here->HSMHV_isbs + small ;
#ifdef HSMHVEVAL
        T1_dT = 1.0 / model->HSMHV_ktnom ; /* Self heating */
        T0_dT = 2.0 * here->HSMHV_Tratio * T1_dT ;       /* Self heating */
	T3_dT = isbs_dT ;                /* Self heating */
#endif

        here->HSMHV_vbst = pParam->HSMHV_njs / beta 
          * log ( pParam->HSMHV_vdiffjs * T0 / T3 + 1.0 ) ;

        here->HSMHV_exptemps = exp (( here->HSMHV_Tratio - 1.0 ) * model->HSMHV_ctemps ) ;

#ifdef HSMHVEVAL
	vbst_dT = - beta_dT / beta * here->HSMHV_vbst
	  + pParam->HSMHV_njs / beta * pParam->HSMHV_vdiffjs / ( pParam->HSMHV_vdiffjs * T0 / T3 + 1.0 ) 
	  * ( T0_dT / T3 - T0 / T3 / T3 * T3_dT ) ; /* Self heating */
#endif

        here->HSMHV_jd_nvtm_invs = 1.0 / ( pParam->HSMHV_njs / beta ) ;
        here->HSMHV_jd_expcs = exp (here->HSMHV_vbst * here->HSMHV_jd_nvtm_invs ) ;

#ifdef HSMHVEVAL
       exptemps_dT = model->HSMHV_ctemps / model->HSMHV_ktnom * here->HSMHV_exptemps ;       /* Self heating */
	jd_nvtm_invs_dT = beta_dT / pParam->HSMHV_njs ;                                  /* Self heating */
	jd_expcs_dT = here->HSMHV_jd_expcs
	  * ( vbst_dT * here->HSMHV_jd_nvtm_invs + here->HSMHV_vbst * jd_nvtm_invs_dT ) ; /* Self heating */
#endif


/* end of HSMHVtemp_eval_dio.h */
