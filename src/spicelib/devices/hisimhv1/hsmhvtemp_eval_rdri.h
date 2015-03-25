/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 2  SUBVERSION : 2  REVISION : 0 ) 
 Model Parameter 'VERSION' : 2.20
 FILE : hsmhvtemp_eval_rdri.h

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

        TTEMP = ckt->CKTtemp;
        if ( here->HSMHV_dtemp_Given ) { TTEMP = TTEMP + here->HSMHV_dtemp ; }
        TTEMP0 = TTEMP ; 
#ifdef HSMHVEVAL
        /* Self heating */
        TTEMP = TTEMP + deltemp ; 
#endif


        /* Phonon Scattering (temperature-dependent part) */
        T1 =  Fn_Pow ( here->HSMHV_Tratio, model->HSMHV_rdrmuetmp ) ;
        here->HSMHV_rdrmue = modelMKS->HSMHV_rdrmue / T1 ;
#ifdef HSMHVEVAL
        T1_dT = model->HSMHV_rdrmuetmp * Fn_Pow( here->HSMHV_Tratio, model->HSMHV_rdrmuetmp - 1.0 )
          / model->HSMHV_ktnom ;
        Mu0_dT = - modelMKS->HSMHV_rdrmue / ( T1 * T1 ) * T1_dT ;
#endif


        /* Velocity Temperature Dependence */
        T0 = 1.8 + 0.4 * here->HSMHV_Tratio + 0.1 * here->HSMHV_Tratio * here->HSMHV_Tratio - model->HSMHV_rdrvtmp * ( 1.0 - here->HSMHV_Tratio ) ;
#ifdef HSMHVEVAL
        T0_dT = 1 / model->HSMHV_ktnom * ( 0.4 + 0.2 * here->HSMHV_Tratio + model->HSMHV_rdrvtmp ) ;
#endif
        here->HSMHV_rdrvmax = modelMKS->HSMHV_rdrvmax / T0 ;
#ifdef HSMHVEVAL
        Vmax_dT = - modelMKS->HSMHV_rdrvmax / ( T0 * T0 ) * T0_dT ;
#endif


        here->HSMHV_rdrcx  = model->HSMHV_rdrcx ;
        here->HSMHV_rdrcar = model->HSMHV_rdrcar ;
#ifdef HSMHVEVAL
        Cx_dT              = 0.0 ; 
        Car_dT             = 0.0 ; 
#endif

      //Toshiba model //
        here->HSMHV_rdrbb = model->HSMHV_rdrbb+model->HSMHV_rdrbbtmp*(TTEMP-model->HSMHV_ktnom) ;
#ifdef HSMHVEVAL
        Rdrbb_dT          = model->HSMHV_rdrbbtmp ;
#endif

/* end of HSMHVtemp_eval_rdri.h */
