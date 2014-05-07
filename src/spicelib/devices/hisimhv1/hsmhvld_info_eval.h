/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 1  SUBVERSION : 2  REVISION : 4 )
 Model Parameter VERSION : 1.23
 FILE : hsmhvld_info_eval.h

 DATE : 2013.04.30

 recent changes: - 2009.01.09 some bugfixes

 released by
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

    /* print all outputs ------------VV */
    if ( model->HSMHV_info >= 4 ) {
        here->HSMHV_csdo = - (here->HSMHV_cddo + here->HSMHV_cgdo + here->HSMHV_cbdo) ;
        here->HSMHV_csgo = - (here->HSMHV_cdgo + here->HSMHV_cggo + here->HSMHV_cbgo) ;
        here->HSMHV_csbo = - (here->HSMHV_cdbo + here->HSMHV_cgbo + here->HSMHV_cbbo) ;

        here->HSMHV_cdso = - (here->HSMHV_cddo + here->HSMHV_cdgo + here->HSMHV_cdbo) ;
        here->HSMHV_cgso = - (here->HSMHV_cgdo + here->HSMHV_cggo + here->HSMHV_cgbo) ;
        here->HSMHV_csso = - (here->HSMHV_csdo + here->HSMHV_csgo + here->HSMHV_csbo) ;

        cgdb =    dQg_dVds                        - ((here->HSMHV_mode > 0) ? here->HSMHV_cgdo : here->HSMHV_cgso) ;
        cggb =    dQg_dVgs                        -                          here->HSMHV_cggo                    ;
        cgsb = - (dQg_dVds + dQg_dVgs + dQg_dVbs) - ((here->HSMHV_mode > 0) ? here->HSMHV_cgso : here->HSMHV_cgdo) ;
        cbdb =    dQb_dVds                        - ((here->HSMHV_mode > 0) ? here->HSMHV_cbdo                    : 
                                                                             -(here->HSMHV_cbdo+here->HSMHV_cbgo+here->HSMHV_cbbo)) ;
        cbgb =    dQb_dVgs                        -                          here->HSMHV_cbgo                    ;
        cbsb = - (dQb_dVds + dQb_dVgs + dQb_dVbs) - ((here->HSMHV_mode > 0) ? -(here->HSMHV_cbdo+here->HSMHV_cbgo+here->HSMHV_cbbo)
                                                                                             : here->HSMHV_cbdo) ;
        cddb =    dQd_dVds                        - ((here->HSMHV_mode > 0) ? here->HSMHV_cddo : here->HSMHV_csso) ;
        cdgb =    dQd_dVgs                        - ((here->HSMHV_mode > 0) ? here->HSMHV_cdgo : here->HSMHV_csgo) ;
        cdsb = - (dQd_dVds + dQd_dVgs + dQd_dVbs) - ((here->HSMHV_mode > 0) ? here->HSMHV_cdso : here->HSMHV_csdo) ;

        if (flg_nqs) {
          /* by implicit differentiation of the nqs equations: */
          dQi_nqs_dVds = (dQi_dVds + Iqi_nqs * dtau_dVds )/(1.0 + ag0 * tau ) ;
          dQi_nqs_dVgs = (dQi_dVgs + Iqi_nqs * dtau_dVgs )/(1.0 + ag0 * tau ) ; 
          dQi_nqs_dVbs = (dQi_dVbs + Iqi_nqs * dtau_dVbs )/(1.0 + ag0 * tau ) ;
          dQb_nqs_dVds = (dQbulk_dVds + Iqb_nqs * dtaub_dVds)/(1.0 + ag0 * taub) ; 
          dQb_nqs_dVgs = (dQbulk_dVgs + Iqb_nqs * dtaub_dVgs)/(1.0 + ag0 * taub) ; 
          dQb_nqs_dVbs = (dQbulk_dVbs + Iqb_nqs * dtaub_dVbs)/(1.0 + ag0 * taub) ;
          cgdb_nqs =   dQg_nqs_dQi_nqs * dQi_nqs_dVds + dQg_nqs_dQb_nqs * dQb_nqs_dVds ;     
          cggb_nqs =   dQg_nqs_dQi_nqs * dQi_nqs_dVgs + dQg_nqs_dQb_nqs * dQb_nqs_dVgs ;
          cgsb_nqs = - dQg_nqs_dQi_nqs * (dQi_nqs_dVds + dQi_nqs_dVgs + dQi_nqs_dVbs)
                     - dQg_nqs_dQb_nqs * (dQb_nqs_dVds + dQb_nqs_dVgs + dQb_nqs_dVbs) ;
          cbdb_nqs =   dQb_nqs_dVds ;
          cbgb_nqs =   dQb_nqs_dVgs ;
          cbsb_nqs = -(dQb_nqs_dVds + dQb_nqs_dVgs + dQb_nqs_dVbs) ;
          cddb_nqs =   dQd_nqs_dVds + dQd_nqs_dQi_nqs * dQi_nqs_dVds ;
          cdgb_nqs=    dQd_nqs_dVgs + dQd_nqs_dQi_nqs * dQi_nqs_dVgs ;
          cdsb_nqs=  -(dQd_nqs_dVds + dQd_nqs_dVgs + dQd_nqs_dVbs) - dQd_nqs_dQi_nqs * (dQi_nqs_dVds + dQi_nqs_dVgs + dQi_nqs_dVbs) ;
        } else {
          cgdb_nqs = cggb_nqs = cgsb_nqs = cbdb_nqs = cbgb_nqs = cbsb_nqs = cddb_nqs = cdgb_nqs = cdsb_nqs = 0.0 ;
        }


	printf( "--- variables returned from HSMHVevaluate() ----\n" ) ;
	
	printf( "von    = %12.5e\n" , here->HSMHV_von ) ;
	printf( "vdsat  = %12.5e\n" , here->HSMHV_vdsat ) ;
	printf( "ids    = %12.5e\n" , here->HSMHV_ids ) ;
	
	printf( "gds    = %12.5e\n" , here->HSMHV_dIds_dVdsi ) ;
	printf( "gm     = %12.5e\n" , here->HSMHV_dIds_dVgsi ) ;
	printf( "gmbs   = %12.5e\n" , here->HSMHV_dIds_dVbsi ) ;

	printf( "cggo   = %12.5e\n" , here->HSMHV_cggo ) ;
	printf( "cgdo   = %12.5e\n" , (here->HSMHV_mode > 0) ? here->HSMHV_cgdo : here->HSMHV_cgso ) ;
	printf( "cgso   = %12.5e\n" , (here->HSMHV_mode > 0) ? here->HSMHV_cgso : here->HSMHV_cgdo ) ;
	printf( "cdgo   = %12.5e\n" , (here->HSMHV_mode > 0) ? here->HSMHV_cdgo : here->HSMHV_csgo ) ;
	printf( "cddo   = %12.5e\n" , (here->HSMHV_mode > 0) ? here->HSMHV_cddo : here->HSMHV_csso ) ;
	printf( "cdso   = %12.5e\n" , (here->HSMHV_mode > 0) ? here->HSMHV_cdso : here->HSMHV_csdo ) ;
	printf( "csgo   = %12.5e\n" , (here->HSMHV_mode > 0) ? here->HSMHV_csgo : here->HSMHV_cdgo ) ;
	printf( "csdo   = %12.5e\n" , (here->HSMHV_mode > 0) ? here->HSMHV_csdo : here->HSMHV_cdso ) ;
	printf( "csso   = %12.5e\n" , (here->HSMHV_mode > 0) ? here->HSMHV_csso : here->HSMHV_cddo ) ;
	
	printf( "qg     = %12.5e\n" , Qg + Qg_nqs) ;
	printf( "qd     = %12.5e\n" , Qd + Qd_nqs) ;
	printf( "qs     = %12.5e\n" , Qs + Qs_nqs) ;
	
	printf( "cggb   = %12.5e\n" , cggb + cggb_nqs ) ;
	printf( "cgsb   = %12.5e\n" , cgsb + cgsb_nqs ) ;
	printf( "cgdb   = %12.5e\n" , cgdb + cgdb_nqs ) ;
	printf( "cbgb   = %12.5e\n" , cbgb + cbgb_nqs ) ;
	printf( "cbsb   = %12.5e\n" , cbsb + cbsb_nqs ) ;
	printf( "cbdb   = %12.5e\n" , cbdb + cbdb_nqs ) ;
	printf( "cdgb   = %12.5e\n" , cdgb + cdgb_nqs ) ;
	printf( "cdsb   = %12.5e\n" , cdsb + cdsb_nqs ) ;
	printf( "cddb   = %12.5e\n" , cddb + cddb_nqs ) ;
      
	printf( "ibd    = %12.5e\n" , Ibd ) ;
	printf( "ibs    = %12.5e\n" , Ibs ) ;
	printf( "gbd    = %12.5e\n" , Gbd ) ;
	printf( "gbs    = %12.5e\n" , Gbs ) ;
	printf( "capbd  = %12.5e\n" , Cbd ) ;
	printf( "capbs  = %12.5e\n" , Cbs ) ;
	printf( "qbd    = %12.5e\n" , Qbd ) ;
	printf( "qbs    = %12.5e\n" , Qbs ) ;

	printf( "isub   = %12.5e\n" , here->HSMHV_isub ) ;
	printf( "gbgs   = %12.5e\n" , dIsub_dVgs + dIsubs_dVgs ) ;
	printf( "gbds   = %12.5e\n" , dIsub_dVds + dIsubs_dVds ) ;
	printf( "gbbs   = %12.5e\n" , dIsub_dVbs + dIsubs_dVbs ) ;

 	printf( "S_flicker_noise * ( freq / gain ) = %.16e\n" , here->HSMHV_noiflick ) ;
 	printf( "S_thermal_noise / ( gain * 4kT )  = %.16e\n" , here->HSMHV_noithrml ) ;
 	printf( "S_induced_gate_noise / ( gain * freq^2 ) = %.16e\n" , here->HSMHV_noiigate ) ;
 	printf( "cross-correlation coefficient (= Sigid/sqrt(Sig*Sid) ) = %.16e\n" , here->HSMHV_noicross ) ;
	/* print Surface Potentials */
/*	printf( "ivds %e ivgs %e ivbs %e Ps0 %.16e Pds %.16e\n" ,                 */
/*		 ivds,   ivgs,   ivbs, here->HSMHV_ps0_prv, here->HSMHV_pds_prv ) ; */
    }
    /* print all outputs ------------AA */

/* End of HSMHVld_info_eval.h */
