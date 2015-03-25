/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 2  SUBVERSION : 2  REVISION : 0 ) 
 Model Parameter 'VERSION' : 2.10
 FILE : hsmhvld_info_eval.h

 DATE : 2014.6.11

 recent changes: - 2009.01.09 some bugfixes

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

    /* print all outputs ------------VV */
    if ( model->HSMHV2_info >= 4 ) {
        here->HSMHV2_csdo = - (here->HSMHV2_cddo + here->HSMHV2_cgdo + here->HSMHV2_cbdo) ;
        here->HSMHV2_csgo = - (here->HSMHV2_cdgo + here->HSMHV2_cggo + here->HSMHV2_cbgo) ;
        here->HSMHV2_csbo = - (here->HSMHV2_cdbo + here->HSMHV2_cgbo + here->HSMHV2_cbbo) ;

        here->HSMHV2_cdso = - (here->HSMHV2_cddo + here->HSMHV2_cdgo + here->HSMHV2_cdbo) ;
        here->HSMHV2_cgso = - (here->HSMHV2_cgdo + here->HSMHV2_cggo + here->HSMHV2_cgbo) ;
        here->HSMHV2_csso = - (here->HSMHV2_csdo + here->HSMHV2_csgo + here->HSMHV2_csbo) ;

        cgdb =    dQg_dVds                        - ((here->HSMHV2_mode > 0) ? here->HSMHV2_cgdo : here->HSMHV2_cgso) ;
        cggb =    dQg_dVgs                        -                          here->HSMHV2_cggo                    ;
        cgsb = - (dQg_dVds + dQg_dVgs + dQg_dVbs) - ((here->HSMHV2_mode > 0) ? here->HSMHV2_cgso : here->HSMHV2_cgdo) ;
        cbdb =    dQb_dVds                        - ((here->HSMHV2_mode > 0) ? here->HSMHV2_cbdo                    : 
                                                                             -(here->HSMHV2_cbdo+here->HSMHV2_cbgo+here->HSMHV2_cbbo)) ;
        cbgb =    dQb_dVgs                        -                          here->HSMHV2_cbgo                    ;
        cbsb = - (dQb_dVds + dQb_dVgs + dQb_dVbs) - ((here->HSMHV2_mode > 0) ? -(here->HSMHV2_cbdo+here->HSMHV2_cbgo+here->HSMHV2_cbbo)
                                                                                             : here->HSMHV2_cbdo) ;
        cddb =    dQd_dVds                        - ((here->HSMHV2_mode > 0) ? here->HSMHV2_cddo : here->HSMHV2_csso) ;
        cdgb =    dQd_dVgs                        - ((here->HSMHV2_mode > 0) ? here->HSMHV2_cdgo : here->HSMHV2_csgo) ;
        cdsb = - (dQd_dVds + dQd_dVgs + dQd_dVbs) - ((here->HSMHV2_mode > 0) ? here->HSMHV2_cdso : here->HSMHV2_csdo) ;

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


	printf( "--- variables returned from HSMHV2evaluate() ----\n" ) ;
	
	printf( "von    = %12.5e\n" , here->HSMHV2_von ) ;
	printf( "vdsat  = %12.5e\n" , here->HSMHV2_vdsat ) ;
	printf( "ids    = %12.5e\n" , here->HSMHV2_ids ) ;
	
	printf( "gds    = %12.5e\n" , here->HSMHV2_dIds_dVdsi ) ;
	printf( "gm     = %12.5e\n" , here->HSMHV2_dIds_dVgsi ) ;
	printf( "gmbs   = %12.5e\n" , here->HSMHV2_dIds_dVbsi ) ;

	printf( "cggo   = %12.5e\n" , here->HSMHV2_cggo ) ;
	printf( "cgdo   = %12.5e\n" , (here->HSMHV2_mode > 0) ? here->HSMHV2_cgdo : here->HSMHV2_cgso ) ;
	printf( "cgso   = %12.5e\n" , (here->HSMHV2_mode > 0) ? here->HSMHV2_cgso : here->HSMHV2_cgdo ) ;
	printf( "cdgo   = %12.5e\n" , (here->HSMHV2_mode > 0) ? here->HSMHV2_cdgo : here->HSMHV2_csgo ) ;
	printf( "cddo   = %12.5e\n" , (here->HSMHV2_mode > 0) ? here->HSMHV2_cddo : here->HSMHV2_csso ) ;
	printf( "cdso   = %12.5e\n" , (here->HSMHV2_mode > 0) ? here->HSMHV2_cdso : here->HSMHV2_csdo ) ;
	printf( "csgo   = %12.5e\n" , (here->HSMHV2_mode > 0) ? here->HSMHV2_csgo : here->HSMHV2_cdgo ) ;
	printf( "csdo   = %12.5e\n" , (here->HSMHV2_mode > 0) ? here->HSMHV2_csdo : here->HSMHV2_cdso ) ;
	printf( "csso   = %12.5e\n" , (here->HSMHV2_mode > 0) ? here->HSMHV2_csso : here->HSMHV2_cddo ) ;
	
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

	printf( "isub   = %12.5e\n" , here->HSMHV2_isub ) ;
	printf( "gbgs   = %12.5e\n" , dIsub_dVgs + dIsubs_dVgs ) ;
	printf( "gbds   = %12.5e\n" , dIsub_dVds + dIsubs_dVds ) ;
	printf( "gbbs   = %12.5e\n" , dIsub_dVbs + dIsubs_dVbs ) ;

 	printf( "S_flicker_noise * ( freq / gain ) = %.16e\n" , here->HSMHV2_noiflick ) ;
 	printf( "S_thermal_noise / ( gain * 4kT )  = %.16e\n" , here->HSMHV2_noithrml ) ;
 	printf( "S_induced_gate_noise / ( gain * freq^2 ) = %.16e\n" , here->HSMHV2_noiigate ) ;
 	printf( "cross-correlation coefficient (= Sigid/sqrt(Sig*Sid) ) = %.16e\n" , here->HSMHV2_noicross ) ;
	/* print Surface Potentials */
/*	printf( "ivds %e ivgs %e ivbs %e Ps0 %.16e Pds %.16e\n" ,                 */
/*		 ivds,   ivgs,   ivbs, here->HSMHV2_ps0_prv, here->HSMHV2_pds_prv ) ; */
    }
    /* print all outputs ------------AA */

/* End of HSMHV2ld_info_eval.h */
