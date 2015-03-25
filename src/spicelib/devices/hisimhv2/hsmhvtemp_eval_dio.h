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
        if ( here->HSMHV2_dtemp_Given ) { TTEMP = TTEMP + here->HSMHV2_dtemp ; }
        TTEMP0 = TTEMP ; 
#ifdef HSMHV2EVAL
        /* Self heating */
        TTEMP = TTEMP + deltemp ; 
#endif


        /* Band gap */
        T1 = TTEMP - model->HSMHV2_ktnom ;
        T2 = TTEMP * TTEMP - model->HSMHV2_ktnom * model->HSMHV2_ktnom ;
        Eg = here->HSMHV2_egtnom - pParam->HSMHV2_bgtmp1 * T1
          - pParam->HSMHV2_bgtmp2 * T2 ;
#ifdef HSMHV2EVAL
        Eg_dT = -pParam->HSMHV2_bgtmp1 - 2.0e0 * TTEMP * pParam->HSMHV2_bgtmp2 ;
#endif


        /* Inverse of the thermal voltage */
        here->HSMHV2_beta = beta = C_QE / (C_KB * TTEMP) ;
        here->HSMHV2_beta_inv = 1.0 / beta ;
        here->HSMHV2_beta2 = beta * beta ;
        here->HSMHV2_betatnom = C_QE / (C_KB * model->HSMHV2_ktnom) ;
#ifdef HSMHV2EVAL
        beta_dT=-C_QE/(C_KB*TTEMP*TTEMP);
        beta_inv_dT = C_KB / C_QE ;
#endif


        log_Tratio = log (here->HSMHV2_Tratio) ;
        /* for substrate-drain junction diode. */
        js   = pParam->HSMHV2_js0d 
          * exp ((here->HSMHV2_egtnom * here->HSMHV2_betatnom - Eg * beta
                  + model->HSMHV2_xtid * log_Tratio) / pParam->HSMHV2_njd) ;
        jssw = pParam->HSMHV2_js0swd  
          * exp ((here->HSMHV2_egtnom * here->HSMHV2_betatnom - Eg * beta 
                  + model->HSMHV2_xtid * log_Tratio) / model->HSMHV2_njswd) ;

        js2  = pParam->HSMHV2_js0d 
          * exp ((here->HSMHV2_egtnom * here->HSMHV2_betatnom - Eg * beta
                  + model->HSMHV2_xti2d * log_Tratio) / pParam->HSMHV2_njd) ;  
        jssw2 = pParam->HSMHV2_js0swd  
          * exp ((here->HSMHV2_egtnom * here->HSMHV2_betatnom - Eg * beta
                  + model->HSMHV2_xti2d * log_Tratio) / model->HSMHV2_njswd) ; 
      
#ifdef HSMHV2EVAL
	T0 = - Eg * beta_dT - Eg_dT * beta ; /* Self heating */
	T1 = T0 + model->HSMHV2_xtid  / TTEMP ; /* Self heating */
	T2 = T0 + model->HSMHV2_xti2d / TTEMP ; /* Self heating */

	js_dT =    js    * T1  / pParam->HSMHV2_njd; /* Self heating */
	jssw_dT =  jssw  * T1/ model->HSMHV2_njswd ; /* Self heating */
	js2_dT =   js2   * T2  / pParam->HSMHV2_njd; /* Self heating */
	jssw2_dT = jssw2 * T2 / model->HSMHV2_njswd; /* Self heating */
#endif
      
        here->HSMHV2_isbd = here->HSMHV2_ad * js + here->HSMHV2_pd * jssw ;
        here->HSMHV2_isbd2 = here->HSMHV2_ad * js2 + here->HSMHV2_pd * jssw2 ;

#ifdef HSMHV2EVAL
	isbd_dT =  here->HSMHV2_ad * js_dT  + here->HSMHV2_pd * jssw_dT  ; /* Self heating */
	isbd2_dT = here->HSMHV2_ad * js2_dT + here->HSMHV2_pd * jssw2_dT ; /* Self heating */
#endif


        T0 = here->HSMHV2_Tratio * here->HSMHV2_Tratio ;
        T2 = here->HSMHV2_isbd + small ;
#ifdef HSMHV2EVAL
        T1_dT = 1.0 / model->HSMHV2_ktnom ; /* Self heating */
        T0_dT = 2.0 * here->HSMHV2_Tratio * T1_dT ;       /* Self heating */
        T2_dT = isbd_dT ;                /* Self heating */
#endif

        here->HSMHV2_vbdt = pParam->HSMHV2_njd / beta 
          * log ( pParam->HSMHV2_vdiffjd * T0 / T2 + 1.0 ) ;

        here->HSMHV2_exptempd = exp (( here->HSMHV2_Tratio - 1.0 ) * model->HSMHV2_ctempd ) ;

#ifdef HSMHV2EVAL
	vbdt_dT = - beta_dT / beta * here->HSMHV2_vbdt
	  + pParam->HSMHV2_njd / beta * pParam->HSMHV2_vdiffjd / ( pParam->HSMHV2_vdiffjd * T0 / T2 + 1.0 ) 
	  * ( T0_dT / T2 - T0 / T2 / T2 * T2_dT ) ; /* Self heating */
#endif

        here->HSMHV2_jd_nvtm_invd = 1.0 / ( pParam->HSMHV2_njd / beta ) ;
        here->HSMHV2_jd_expcd = exp (here->HSMHV2_vbdt * here->HSMHV2_jd_nvtm_invd ) ;

#ifdef HSMHV2EVAL
        exptempd_dT = model->HSMHV2_ctempd / model->HSMHV2_ktnom * here->HSMHV2_exptempd ;       /* Self heating */
        jd_nvtm_invd_dT = beta_dT / pParam->HSMHV2_njd ;                                  /* Self heating */
        jd_expcd_dT = here->HSMHV2_jd_expcd
         * ( vbdt_dT * here->HSMHV2_jd_nvtm_invd + here->HSMHV2_vbdt * jd_nvtm_invd_dT ) ; /* Self heating */
#endif


        /* for substrate-source junction diode. */
        js   = pParam->HSMHV2_js0s 
          * exp ((here->HSMHV2_egtnom * here->HSMHV2_betatnom - Eg * beta
                  + model->HSMHV2_xtis * log_Tratio) / pParam->HSMHV2_njs) ;
        jssw = pParam->HSMHV2_js0sws  
          * exp ((here->HSMHV2_egtnom * here->HSMHV2_betatnom - Eg * beta 
                  + model->HSMHV2_xtis * log_Tratio) / model->HSMHV2_njsws) ;

        js2  = pParam->HSMHV2_js0s 
          * exp ((here->HSMHV2_egtnom * here->HSMHV2_betatnom - Eg * beta
                  + model->HSMHV2_xti2s * log_Tratio) / pParam->HSMHV2_njs) ;  
        jssw2 = pParam->HSMHV2_js0sws  
          * exp ((here->HSMHV2_egtnom * here->HSMHV2_betatnom - Eg * beta
                  + model->HSMHV2_xti2s * log_Tratio) / model->HSMHV2_njsws) ; 
      
#ifdef HSMHV2EVAL
	T0 = - Eg * beta_dT - Eg_dT * beta ; /* Self heating */
	T1 = T0 + model->HSMHV2_xtis  / TTEMP ; /* Self heating */
	T2 = T0 + model->HSMHV2_xti2s / TTEMP ; /* Self heating */

	js_dT =    js    * T1  / pParam->HSMHV2_njs; /* Self heating */
	jssw_dT =  jssw  * T1/ model->HSMHV2_njsws ; /* Self heating */
	js2_dT =   js2   * T2  / pParam->HSMHV2_njs; /* Self heating */
	jssw2_dT = jssw2 * T2 / model->HSMHV2_njsws; /* Self heating */
#endif
      
        here->HSMHV2_isbs = here->HSMHV2_as * js + here->HSMHV2_ps * jssw ;
        here->HSMHV2_isbs2 = here->HSMHV2_as * js2 + here->HSMHV2_ps * jssw2 ;

#ifdef HSMHV2EVAL
	isbs_dT =  here->HSMHV2_as * js_dT  + here->HSMHV2_ps * jssw_dT  ; /* Self heating */
	isbs2_dT = here->HSMHV2_as * js2_dT + here->HSMHV2_ps * jssw2_dT ; /* Self heating */
#endif


        T0 = here->HSMHV2_Tratio * here->HSMHV2_Tratio ;
        T3 = here->HSMHV2_isbs + small ;
#ifdef HSMHV2EVAL
        T1_dT = 1.0 / model->HSMHV2_ktnom ; /* Self heating */
        T0_dT = 2.0 * here->HSMHV2_Tratio * T1_dT ;       /* Self heating */
	T3_dT = isbs_dT ;                /* Self heating */
#endif

        here->HSMHV2_vbst = pParam->HSMHV2_njs / beta 
          * log ( pParam->HSMHV2_vdiffjs * T0 / T3 + 1.0 ) ;

        here->HSMHV2_exptemps = exp (( here->HSMHV2_Tratio - 1.0 ) * model->HSMHV2_ctemps ) ;

#ifdef HSMHV2EVAL
	vbst_dT = - beta_dT / beta * here->HSMHV2_vbst
	  + pParam->HSMHV2_njs / beta * pParam->HSMHV2_vdiffjs / ( pParam->HSMHV2_vdiffjs * T0 / T3 + 1.0 ) 
	  * ( T0_dT / T3 - T0 / T3 / T3 * T3_dT ) ; /* Self heating */
#endif

        here->HSMHV2_jd_nvtm_invs = 1.0 / ( pParam->HSMHV2_njs / beta ) ;
        here->HSMHV2_jd_expcs = exp (here->HSMHV2_vbst * here->HSMHV2_jd_nvtm_invs ) ;

#ifdef HSMHV2EVAL
       exptemps_dT = model->HSMHV2_ctemps / model->HSMHV2_ktnom * here->HSMHV2_exptemps ;       /* Self heating */
	jd_nvtm_invs_dT = beta_dT / pParam->HSMHV2_njs ;                                  /* Self heating */
	jd_expcs_dT = here->HSMHV2_jd_expcs
	  * ( vbst_dT * here->HSMHV2_jd_nvtm_invs + here->HSMHV2_vbst * jd_nvtm_invs_dT ) ; /* Self heating */
#endif


/* end of HSMHV2temp_eval_dio.h */
