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
        if ( here->HSMHV2_dtemp_Given ) { TTEMP = TTEMP + here->HSMHV2_dtemp ; }
        TTEMP0 = TTEMP ; 
#ifdef HSMHV2EVAL
        /* Self heating */
        TTEMP = TTEMP + deltemp ; 
#endif


        /* Phonon Scattering (temperature-dependent part) */
        T1 =  Fn_Pow ( here->HSMHV2_Tratio, model->HSMHV2_rdrmuetmp ) ;
        here->HSMHV2_rdrmue = modelMKS->HSMHV2_rdrmue / T1 ;
#ifdef HSMHV2EVAL
        T1_dT = model->HSMHV2_rdrmuetmp * Fn_Pow( here->HSMHV2_Tratio, model->HSMHV2_rdrmuetmp - 1.0 )
          / model->HSMHV2_ktnom ;
        Mu0_dT = - modelMKS->HSMHV2_rdrmue / ( T1 * T1 ) * T1_dT ;
#endif


        /* Velocity Temperature Dependence */
        T0 = 1.8 + 0.4 * here->HSMHV2_Tratio + 0.1 * here->HSMHV2_Tratio * here->HSMHV2_Tratio - model->HSMHV2_rdrvtmp * ( 1.0 - here->HSMHV2_Tratio ) ;
#ifdef HSMHV2EVAL
        T0_dT = 1 / model->HSMHV2_ktnom * ( 0.4 + 0.2 * here->HSMHV2_Tratio + model->HSMHV2_rdrvtmp ) ;
#endif
        here->HSMHV2_rdrvmax = modelMKS->HSMHV2_rdrvmax / T0 ;
#ifdef HSMHV2EVAL
        Vmax_dT = - modelMKS->HSMHV2_rdrvmax / ( T0 * T0 ) * T0_dT ;
#endif


        here->HSMHV2_rdrcx  = model->HSMHV2_rdrcx ;
        here->HSMHV2_rdrcar = model->HSMHV2_rdrcar ;
#ifdef HSMHV2EVAL
        Cx_dT              = 0.0 ; 
        Car_dT             = 0.0 ; 
#endif

      //Toshiba model //
        here->HSMHV2_rdrbb = model->HSMHV2_rdrbb+model->HSMHV2_rdrbbtmp*(TTEMP-model->HSMHV2_ktnom) ;
#ifdef HSMHV2EVAL
        Rdrbb_dT          = model->HSMHV2_rdrbbtmp ;
#endif

/* end of HSMHV2temp_eval_rdri.h */
