/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 2  SUBVERSION : 2  REVISION : 0 ) 
 Model Parameter 'VERSION' : 2.20
 FILE : hsmhveval_dep.h

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

{ // Begin : hsmhveval_dep

/* define local variavles */
  int    depmode ;
  double afact, afact2, afact3, bfact, cfact ;
  double W_bsub0, W_bsubL, W_s0, W_sL, W_sub0, W_subL, W_b0, W_bL, vthn ;
  double phi_s0_DEP = 0.0, phi_sL_DEP = 0.0 , Vbi_DEP ;
  double phi_s0_DEP_dVgs, phi_s0_DEP_dVbs, phi_s0_DEP_dVds, phi_s0_DEP_dT ;
  double phi_sL_DEP_dVgs, phi_sL_DEP_dVbs, phi_sL_DEP_dVds, phi_sL_DEP_dT ;
  double phi_j0_DEP, phi_jL_DEP, Psbmax, phi_b0_DEP_lim, phi_bL_DEP_lim ;


  double phi_jL_DEP_dVgs, phi_jL_DEP_dVds, phi_jL_DEP_dVbs, phi_jL_DEP_dT ;

  double Vgp0, Vgp1, Vgp0old, phi_j0_DEP_old, phi_jL_DEP_old, phi_b0_DEP_old, phi_bL_DEP_old, phi_s0_DEP_old, phi_sL_DEP_old ;
  double phi_j0_DEP_acc, phi_jL_DEP_acc ;


  double Q_s0, Q_sL = 0.0 ;
  double Q_s0_dVgs, Q_sL_dVgs = 0.0, Q_s0_dVds, Q_sL_dVds = 0.0, Q_s0_dVbs, Q_sL_dVbs = 0.0, Q_s0_dT, Q_sL_dT = 0.0 ;
  double Q_sub0, Q_subL, Q_sub0_dVgs, Q_subL_dVgs, Q_sub0_dVds, Q_subL_dVds, Q_sub0_dVbs, Q_subL_dVbs, Q_sub0_dT, Q_subL_dT ;
  double Qn_res0, Qn_res0_dVgs, Qn_res0_dVds, Qn_res0_dVbs, Qn_res0_dT ;


  double y1, y2, dety ;
  double y11, y12 ;
  double y21, y22 ;
  
  double y1_dVgs, y1_dVds, y1_dVbs, y1_dT ;
  double y2_dVgs, y2_dVds, y2_dVbs, y2_dT ;

  double rev11 = 0.0, rev12 = 0.0 ;
  double rev21 = 0.0, rev22 = 0.0 ;

  double phi_b0_DEP_ini ;
  double y0, dydPsm ;
  
  double W_b0_dVgs, W_b0_dVds, W_b0_dVbs, W_b0_dT ;

  double W_res0 ;
  double W_s0_dVgs, W_s0_dVds, W_s0_dVbs, W_s0_dT ;

  double phi_b0_DEP,  Q_b0_dep, Q_sub0_dep ;
  double phi_b0_DEP_dVgs, phi_b0_DEP_dVds, phi_b0_DEP_dVbs, phi_b0_DEP_dT ;
  double phi_j0_DEP_dVgs, phi_j0_DEP_dVds, phi_j0_DEP_dVbs, phi_j0_DEP_dT ;
  double Q_b0_dep_dVgs, Q_b0_dep_dVds, Q_b0_dep_dVbs, Q_b0_dep_dT ;
  double Q_sub0_dep_dVgs, Q_sub0_dep_dVds, Q_sub0_dep_dVbs, Q_sub0_dep_dT ;

  double phi_bL_DEP, Q_bL_dep, Q_subL_dep ;
  double phi_bL_DEP_dVgs, phi_bL_DEP_dVds, phi_bL_DEP_dVbs, phi_bL_DEP_dT ;
  double Q_bL_dep_dVgs, Q_bL_dep_dVds, Q_bL_dep_dVbs, Q_bL_dep_dT ;
  double Q_subL_dep_dVgs, Q_subL_dep_dVds, Q_subL_dep_dVbs, Q_subL_dep_dT ;

  double q_Ndepm_esi, Idd_drift,Idd_diffu ;
  double Qn_bac0 ;
  double Qn_bac0_dVgs, Qn_bac0_dVds, Qn_bac0_dVbs, Qn_bac0_dT ;

  double Mu_res, Mu_bac ;
  double Mu_res_dVgs, Mu_res_dVds, Mu_res_dVbs, Mu_res_dT ;
  double Mu_bac_dVgs, Mu_bac_dVds, Mu_bac_dVbs, Mu_bac_dT ;

  double Q_n0_cur, Q_nL_cur ;
  double Q_n0_cur_dVgs, Q_n0_cur_dVds, Q_n0_cur_dVbs, Q_n0_cur_dT ;
  double Q_nL_cur_dVgs, Q_nL_cur_dVds, Q_nL_cur_dVbs, Q_nL_cur_dT ;

  double Q_s0_dep, Q_sL_dep ;
  double Q_s0_dep_dVgs, Q_s0_dep_dVds, Q_s0_dep_dVbs, Q_s0_dep_dT ;
  double Q_sL_dep_dVgs, Q_sL_dep_dVds, Q_sL_dep_dVbs, Q_sL_dep_dT ;

  double sm_delta ;
  double phib_ref, phib_ref_dPs, phib_ref_dPd ;
  double Q_s0_dPs, Q_sL_dPs, Q_s0_dPb, Q_sL_dPb ;
  double Q_b0_dep_dPb, Q_bL_dep_dPb, Q_b0_dep_dPd, Q_bL_dep_dPd, Q_sub0_dep_dPd, Q_subL_dep_dPd ;
  double phi_j0_DEP_dPb, phi_jL_DEP_dPb ;
  double NdepmpNsub_inv1, NdepmpNsub ;



  double Q_n0, Q_n0_dVgs, Q_n0_dVds, Q_n0_dVbs, Q_n0_dT ;
  double Q_nL, Q_nL_dVgs, Q_nL_dVds, Q_nL_dVbs, Q_nL_dT ;

  double phi_s0_DEP_ini, phi_sL_DEP_ini ;


  double C_QE2, C_ESI2, Tn2 ; 
  double Ndepm2, q_Ndepm ; 
  double C_2ESIpq_Ndepm, C_2ESIpq_Ndepm_inv , C_2ESI_q_Ndepm ; 
  double C_2ESIpq_Nsub , C_2ESIpq_Nsub_inv ; 
  double ps_conv3 , ps_conv23 ; 
  double Ids_res, Ids_bac, Edri ;
  double Ids_res_dVgs, Ids_res_dVds, Ids_res_dVbs ;
  double Ids_res_dT ;
  double Ids_bac_dVgs, Ids_bac_dVds, Ids_bac_dVbs, Ids_bac_dT ;
  double Edri_dVgs, Edri_dVds, Edri_dVbs, Edri_dT ;

  double T1_dVgs, T1_dVds, T1_dVbs ;
  double T2_dVgs, T2_dVds, T2_dVbs ;
  double T3_dVgs, T3_dVds, T3_dVbs ;
  double T4_dVgs, T4_dVds, T4_dVbs ;
  double T5_dVgs, T5_dVds, T5_dVbs ;


  double Vgpp ;
  double Vgpp_dVgs, Vgpp_dVds, Vgpp_dVbs,Vgpp_dT ;
  double Vdseff0, Vdseff0_dVgs, Vdseff0_dVds, Vdseff0_dVbs,Vdseff0_dT ;
  double phib_ref_dVgs, phib_ref_dVds, phib_ref_dVbs,phib_ref_dT ;

  double Qn_delta, Qn_delta_dT ;
  double Qn_drift, Qn_drift_dVgs, Qn_drift_dVds, Qn_drift_dVbs, Qn_drift_dT ;

  double Ey_suf, Ey_suf_dVgs, Ey_suf_dVds, Ey_suf_dVbs, Ey_suf_dT ;

  double DEPQFN3 = 0.3 ;
  double DEPQFN_dlt = 2.0 ;
  double Ps_delta = 0.06 ;
  double Ps_delta0 = 0.10 ;

    // Constants
    Vbi_DEP = here->HSMHV2_Vbipn ;
    q_Ndepm = C_QE * here->HSMHV2_ndepm ;
    Ndepm2  = here->HSMHV2_ndepm * here->HSMHV2_ndepm ;
    q_Ndepm_esi = C_QE * here->HSMHV2_ndepm * C_ESI ;
    q_Nsub = C_QE * here->HSMHV2_nsub ;
    C_QE2  = C_QE * C_QE ;
    C_ESI2 = C_ESI * C_ESI ;
    Tn2    = model->HSMHV2_tndep * model->HSMHV2_tndep ;
    C_2ESIpq_Ndepm = 2.0 * C_ESI/q_Ndepm ;
    C_2ESIpq_Ndepm_inv = q_Ndepm / (2.0 * C_ESI) ;
    C_2ESI_q_Ndepm = 2.0 * C_ESI * q_Ndepm ;
    C_2ESIpq_Nsub  = 2.0 * C_ESI / q_Nsub  ;
    C_2ESIpq_Nsub_inv  = q_Nsub / (2.0 * C_ESI) ;
    NdepmpNsub  = here->HSMHV2_ndepm / here->HSMHV2_nsub ;
    NdepmpNsub_inv1  = 1.0 / (1.0 + NdepmpNsub ) ;
    ps_conv3  = ps_conv * 1000.0 ;
    ps_conv23 = ps_conv2 * 1000.0 ;
 
     //---------------------------------------------------*
     // depletion MOS mode  
     //------------------//

     /*---------------------------------------------------*
      * initial potential phi_s0_DEP,phi_b0_DEP,phi_j0_DEP calculated.
      *------------------*/

       Vgp = Vgp + epsm10 * 1.0e7 ;


      afact = Cox * Cox / here->HSMHV2_cnst0 / here->HSMHV2_cnst0 ;
      afact2 = afact / here->HSMHV2_nin / here->HSMHV2_nin * Ndepm2 ;
      W_bsub0 = sqrt(2.0e0 * C_ESI / C_QE * here->HSMHV2_nsub / (here->HSMHV2_nsub
              + here->HSMHV2_ndepm) / here->HSMHV2_ndepm * ( - Vbscl + Vbi_DEP)) ;

      if( W_bsub0 > model->HSMHV2_tndep ) {

        Vgp0 = 0.0;

        W_b0 = model->HSMHV2_tndep ;
        phi_b0_DEP = 0.0 ;
        phi_j0_DEP = phi_b0_DEP - C_2ESIpq_Ndepm_inv * W_b0 * W_b0 ;
        phi_b0_DEP_lim = 0.0 ;

        Vgp0old = Vgp0 ;
        phi_j0_DEP_old = phi_j0_DEP ;

        for ( lp_s0 = 1 ; lp_s0 <= lp_se_max + 1 ; lp_s0 ++ ) {

          W_b0 = sqrt(C_2ESIpq_Ndepm * (phi_b0_DEP - phi_j0_DEP) ) ;
          Fn_SU_CP( W_b0 , W_b0 , model->HSMHV2_tndep , 1e-8, 2 , T0 )
          W_sub0 = sqrt(C_2ESIpq_Nsub * (phi_j0_DEP - Vbscl + Vbi_DEP) ) ;

          Q_b0_dep = W_b0 * q_Ndepm ;
          Q_b0_dep_dPd = - C_ESI / W_b0 * T0 ;
          Q_sub0_dep = - W_sub0 * q_Nsub ;
          Q_sub0_dep_dPd = - C_ESI / W_sub0 ;

          y1 = Cox * (Vgp0 - phi_b0_DEP) + Q_b0_dep + Q_sub0_dep ;
          y11 = Cox ;
          y12 = Q_b0_dep_dPd + Q_sub0_dep_dPd ;

          y2 = phi_j0_DEP - NdepmpNsub_inv1 * (NdepmpNsub * phi_b0_DEP + Vbscl - Vbi_DEP) ;
          y21 = 0.0 ;
          y22 = 1.0 ;

          dety = y11 * y22 - y21 * y12;
          rev11 = (y22) / dety ;
          rev12 = ( - y12) / dety ;
          rev21 = ( - y21) / dety ;
          rev22 = (y11) / dety ;

          if( fabs( rev11 * y1 + rev12 * y2 ) > 0.5 ) {
            Vgp0 = Vgp0 - 0.5 * Fn_Sgn( rev11 * y1 + rev12 * y2 ) ;
            phi_j0_DEP = phi_j0_DEP - 0.5 * Fn_Sgn( rev21 * y1 + rev22 * y2 ) ;
          } else {
            Vgp0 = Vgp0 - ( rev11 * y1 + rev12 * y2 ) ;
            phi_j0_DEP = phi_j0_DEP - ( rev21 * y1 + rev22 * y2 ) ;
          }

          if( fabs(Vgp0 - Vgp0old) <= ps_conv &&
                fabs(phi_j0_DEP - phi_j0_DEP_old) <= ps_conv ) lp_s0=lp_se_max + 1 ;

          Vgp0old = Vgp0 ;
          phi_j0_DEP_old = phi_j0_DEP ;
        } 
        phi_j0_DEP_acc = phi_j0_DEP ;

        W_sub0 = model->HSMHV2_tndep * NdepmpNsub ;
        phi_j0_DEP = C_2ESIpq_Nsub_inv * W_sub0 * W_sub0 + Vbscl - Vbi_DEP ;
        phi_b0_DEP = phi_j0_DEP + C_2ESIpq_Ndepm_inv * Tn2 ;
        phi_s0_DEP = phi_b0_DEP ;
        Psbmax = phi_b0_DEP ;
        Vgp1 = phi_b0_DEP ;
        if( Vgp > Vgp0 ) { 
          depmode = 1 ;
        } else if(Vgp > Vgp1 ) {
          depmode = 3 ;
        } else {
          depmode = 2 ;
        }

      } else {
        Vgp0 = 0.0 ;
        Vgp1 = Vgp0 ;
        Psbmax = 0.0 ;
        phi_b0_DEP_lim = Vgp0 ;
        W_b0 = W_bsub0 ;
        W_sub0 = W_b0 * NdepmpNsub ;
        phi_j0_DEP = C_2ESIpq_Nsub_inv * W_sub0 * W_sub0 + Vbscl - Vbi_DEP ;
        phi_b0_DEP = C_2ESIpq_Ndepm_inv * W_b0 * W_b0 + phi_j0_DEP ;
        phi_j0_DEP_acc = phi_j0_DEP ;
        if( Vgp > Vgp0 ) { 
          depmode = 1 ;
        } else {
          depmode = 2 ;
        }

      }


      T1 = C_2ESI_q_Ndepm * ( Psbmax - ( - here->HSMHV2_Pb2n + Vbscl)) ;
      if ( T1 > 0.0 ) {
        vthn = - here->HSMHV2_Pb2n + Vbscl - sqrt(T1) / Cox ;
      } else {
        vthn = - here->HSMHV2_Pb2n + Vbscl ;
      }

      /* primary value */

       if( Vgp > Vgp0 ) {
        /* accumulation region */
         phi_j0_DEP = phi_j0_DEP_acc ;
         phi_b0_DEP = 0.0 ;
         phi_s0_DEP_ini = log(afact * Vgp * Vgp) / (beta + 2.0 / Vgp) + phi_b0_DEP ;

         if( phi_s0_DEP_ini < phi_b0_DEP_lim + ps_conv23 ) phi_s0_DEP_ini = phi_b0_DEP_lim + ps_conv23 ;

       } else if( Vgp > Vgp1 ) {
        /* depletion region */

         phi_s0_DEP_ini = phi_s0_DEP ;

       } else {
        /* depletion and inversion region */

         if( Vgp > vthn ) {
          /* depletion */
           bfact = - 2.0 * afact * Vgp + beta ;
           cfact = afact * Vgp * Vgp - beta * phi_b0_DEP ;
           phi_b0_DEP_old = phi_b0_DEP ;

           phi_s0_DEP_ini = ( - bfact + sqrt(bfact * bfact - 4.0 * afact * cfact)) / 2.0 / afact ;
           if( phi_s0_DEP_ini > Psbmax - ps_conv3 ) phi_s0_DEP_ini = Psbmax - ps_conv3 ;

           W_s0 = sqrt(C_2ESIpq_Ndepm * (phi_b0_DEP - phi_s0_DEP_ini) ) ;
           W_b0 = sqrt(C_2ESIpq_Ndepm * (phi_b0_DEP - phi_j0_DEP) ) ;

           if( W_s0 + W_b0 > model->HSMHV2_tndep ) {
             for ( lp_s0 = 1 ; lp_s0 <= lp_se_max + 1 ; lp_s0 ++ ) {

               y0 = W_s0 + W_b0 - model->HSMHV2_tndep ;

               dydPsm = C_ESI / q_Ndepm / W_s0 
                  + C_ESI / q_Ndepm * ( 1.0 - (here->HSMHV2_ndepm
                  / here->HSMHV2_nsub) / ( 1.0 + (NdepmpNsub))) / W_b0 ;

               if( fabs(y0 / dydPsm) > 0.5 ) {
                 phi_b0_DEP = phi_b0_DEP - 0.5 * Fn_Sgn(y0 / dydPsm) ;
               } else {
                 phi_b0_DEP = phi_b0_DEP - y0 / dydPsm ;
               }

               if( (phi_b0_DEP - Vbscl + Vbi_DEP) < epsm10 )
                  phi_b0_DEP=Vbscl - Vbi_DEP + epsm10 ;

               cfact = afact * Vgp * Vgp - beta * phi_b0_DEP ;
               T1 = bfact * bfact - 4.0 * afact * cfact ;
               if( T1 > 0.0 ) {
                 phi_s0_DEP_ini = ( - bfact + sqrt(T1)) / 2.0 / afact ;
               } else {
                 phi_s0_DEP_ini = ( - bfact) / 2.0 / afact ;
               }

               if( phi_s0_DEP_ini > Psbmax ) phi_s0_DEP_ini = Psbmax ;
               if( phi_s0_DEP_ini > phi_b0_DEP ) {
                 phi_s0_DEP_ini = phi_b0_DEP - ps_conv23 ;
                 lp_s0=lp_se_max + 1 ;
               }

               W_s0 = sqrt(C_2ESIpq_Ndepm * (phi_b0_DEP - phi_s0_DEP_ini) ) ;
               phi_j0_DEP = ( NdepmpNsub * phi_b0_DEP 
                 + Vbscl - Vbi_DEP) / (1.0 + NdepmpNsub) ;
               W_b0 = sqrt(C_2ESIpq_Ndepm * (phi_b0_DEP - phi_j0_DEP) ) ;

              if( fabs(phi_b0_DEP - phi_b0_DEP_old) <= 1.0e-8 ) lp_s0=lp_se_max + 1 ;
               phi_b0_DEP_old = phi_b0_DEP ;
             }
           }

         } else {
           afact3 = afact2 / exp(beta * Vbscl) ;
           phi_b0_DEP_old = phi_b0_DEP ;
           phi_s0_DEP_ini = log(afact3 * Vgp * Vgp) / ( - beta + 2.0 / Vgp) ;
           W_s0 = sqrt(C_2ESIpq_Ndepm * (phi_b0_DEP - phi_s0_DEP_ini) ) ;
           W_b0 = sqrt(C_2ESIpq_Ndepm * (phi_b0_DEP - phi_j0_DEP) ) ;
           if( W_s0 + W_b0 >  model->HSMHV2_tndep ) {
             for ( lp_s0 = 1 ; lp_s0 <= lp_s0_max + 1 ; lp_s0 ++ ) {

               y0 = W_s0 + W_b0 - model->HSMHV2_tndep ;
               dydPsm = C_ESI / q_Ndepm / W_s0 
                 + C_ESI / q_Ndepm * ( 1.0 - (here->HSMHV2_ndepm / 
                 here->HSMHV2_nsub) / ( 1.0 + (NdepmpNsub))) / W_b0 ;

               if( fabs(y0 / dydPsm) > 0.5 ) {
                 phi_b0_DEP = phi_b0_DEP - 0.5 * Fn_Sgn(y0 / dydPsm) ;
               } else {
                 phi_b0_DEP = phi_b0_DEP - y0 / dydPsm ;
               } 
               if( (phi_b0_DEP - Vbscl + Vbi_DEP) < epsm10 ) 
                    phi_b0_DEP=Vbscl - Vbi_DEP + epsm10 ;

               W_s0 = sqrt(C_2ESIpq_Ndepm * (phi_b0_DEP - phi_s0_DEP_ini) ) ;
               phi_j0_DEP = ( NdepmpNsub * phi_b0_DEP
                 + Vbscl - Vbi_DEP) / (1.0 + NdepmpNsub) ;
               W_b0 = sqrt(C_2ESIpq_Ndepm * (phi_b0_DEP - phi_j0_DEP) ) ;

               if( fabs(phi_b0_DEP - phi_b0_DEP_old) <= 1.0e-5 ) lp_s0=lp_s0_max + 1 ;
               phi_b0_DEP_old = phi_b0_DEP ;
             }

           }
         } // end of phi_b0_DEP loop //

       }
       phi_b0_DEP_ini = phi_b0_DEP ;

       /*                               */
       /* solve poisson at source side  */
       /*                               */

       sm_delta = 0.12 ;

       flg_conv = 0 ;

       phi_s0_DEP = phi_s0_DEP_ini ;
       phi_b0_DEP = phi_b0_DEP_ini ;

       phi_s0_DEP_old = phi_s0_DEP ;
       phi_b0_DEP_old = phi_b0_DEP ;

       for ( lp_s0 = 1 ; lp_s0 <= lp_se_max + 1 ; lp_s0 ++ ) {

         phi_j0_DEP  = NdepmpNsub_inv1 * (NdepmpNsub * phi_b0_DEP + Vbscl - Vbi_DEP) ;
         phi_j0_DEP_dPb = NdepmpNsub_inv1 * NdepmpNsub ;

         T1 = phi_b0_DEP - phi_j0_DEP ;
         Fn_SL_CP( T2 , T1 , 0.0 , 0.1, 2 , T7 )
         W_b0 = sqrt(C_2ESIpq_Ndepm * (T2) ) ;
         Fn_SU_CP( W_b0 , W_b0 , model->HSMHV2_tndep, 1e-8, 2 , T8 )
         W_sub0 = sqrt(C_2ESIpq_Nsub * (phi_j0_DEP - Vbscl + Vbi_DEP) ) ;
         Q_b0_dep = W_b0 * q_Ndepm ;
         Q_b0_dep_dPb = C_ESI / W_b0 * T7 * T8 ;
         Q_b0_dep_dPd = - C_ESI / W_b0 * T7 * T8 ;
         Q_sub0_dep = - W_sub0 * q_Nsub ;
         Q_sub0_dep_dPd = - C_ESI / W_sub0 ;

         T10 = 8.0 * q_Ndepm_esi * Tn2 ;
         phib_ref = (4.0 * phi_j0_DEP * phi_j0_DEP * C_ESI2 - 8.0 * phi_j0_DEP * C_ESI2 * phi_s0_DEP
              + 4.0 * C_ESI2 * phi_s0_DEP * phi_s0_DEP
              + 4.0 * phi_j0_DEP * q_Ndepm_esi * Tn2
              + 4.0 * phi_s0_DEP * q_Ndepm_esi * Tn2
              + Ndepm2 * C_QE2 * model->HSMHV2_tndep
              * Tn2 * model->HSMHV2_tndep) / T10 ;
         phib_ref_dPs = ( - 8.0 * phi_j0_DEP * C_ESI2 + 4.0 * C_ESI2 * phi_s0_DEP * 2.0
              + 4.0 * q_Ndepm_esi * Tn2) / T10 ;
         phib_ref_dPd = (4.0 * phi_j0_DEP * C_ESI2 * 2.0 - 8.0 * C_ESI2 * phi_s0_DEP
              + 4.0 * q_Ndepm_esi * Tn2) / T10 ;

         T1 = beta * (phi_s0_DEP - phi_b0_DEP) ;
         T2 = exp(T1) ; 
         if( phi_s0_DEP >= phi_b0_DEP ) { 
           Q_s0 = - here->HSMHV2_cnst0 * sqrt(T2 - 1.0 - T1 + 1e-15) ;
           Q_s0_dPs = 0.5 * here->HSMHV2_cnst0 * here->HSMHV2_cnst0 / Q_s0 * (beta * T2 - beta ) ;
           Q_s0_dPb = - Q_s0_dPs ;
         } else {
           T3 = exp( - beta * (phi_s0_DEP - Vbscl)) ;
           T4 = exp( - beta * (phi_b0_DEP - Vbscl)) ;
           Q_s0 = here->HSMHV2_cnst0 * sqrt(T2 - 1.0 - T1 + 1e-15 + here->HSMHV2_cnst1 * (T3 - T4) ) ;
           T5 = 0.5 * here->HSMHV2_cnst0 * here->HSMHV2_cnst0 / Q_s0 ;
           Q_s0_dPs = T5 * (beta * T2 - beta + here->HSMHV2_cnst1 * ( - beta * T3) ) ;
           Q_s0_dPb = T5 * ( - beta * T2 + beta + here->HSMHV2_cnst1 * beta * T4 ) ;
         }

         Fn_SU_CP( T1 , phib_ref , phi_b0_DEP_lim , sm_delta, 4 , T9 )

         y1 = phi_b0_DEP - T1 ; 
         y11 = - phib_ref_dPs * T9 ;
         y12 = 1.0 - phib_ref_dPd * phi_j0_DEP_dPb * T9 ;

         y2 = Cox * (Vgp - phi_s0_DEP) + Q_s0 + Q_b0_dep + Q_sub0_dep ;
         y21 = - Cox + Q_s0_dPs ;
         y22 = Q_s0_dPb + Q_b0_dep_dPb + Q_b0_dep_dPd * phi_j0_DEP_dPb + Q_sub0_dep_dPd * phi_j0_DEP_dPb ;

         dety = y11 * y22 - y21 * y12;
         rev11 = (y22) / dety ;
         rev12 = ( - y12) / dety ;
         rev21 = ( - y21) / dety ;
         rev22 = (y11) / dety ;
         if( fabs( rev21 * y1 + rev22 * y2 ) > 0.5 ) {
           phi_s0_DEP = phi_s0_DEP - 0.5 * Fn_Sgn( rev11 * y1 + rev12 * y2 ) ;
           phi_b0_DEP = phi_b0_DEP - 0.5 * Fn_Sgn( rev21 * y1 + rev22 * y2 ) ;
         } else {
           phi_s0_DEP = phi_s0_DEP - ( rev11 * y1 + rev12 * y2 ) ;
           phi_b0_DEP = phi_b0_DEP - ( rev21 * y1 + rev22 * y2 ) ;
         }

         if( fabs(phi_s0_DEP - phi_s0_DEP_old) <= ps_conv &&  fabs(phi_b0_DEP - phi_b0_DEP_old) <= ps_conv ) {
           lp_s0=lp_se_max + 1 ;
           flg_conv = 1 ;
         }

         phi_s0_DEP_old = phi_s0_DEP ;
         phi_b0_DEP_old = phi_b0_DEP ;

       }

       if( flg_conv == 0 ) {
         printf( "*** warning(HiSIM_HV(%s)): Went Over Iteration Maximum(Ps0)\n",model->HSMHV2modName ) ;
         printf( " Vbse   = %7.3f Vdse = %7.3f Vgse = %7.3f\n" ,Vbse , Vdse , Vgse ) ;
       }

      /* caluculate derivative */

       y1_dVgs = 0.0 ;
       y1_dVds = 0.0 ;
       y1_dVbs = - (8.0 * phi_j0_DEP * C_ESI2 - 8.0 * C_ESI2 * phi_s0_DEP 
                  + 4.0 * q_Ndepm_esi * Tn2) / T10
                  * T9 * NdepmpNsub_inv1 * Vbscl_dVbs ;
       y1_dT   = - (8.0 * phi_j0_DEP * C_ESI2 - 8.0 * C_ESI2 * phi_s0_DEP 
                  + 4.0 * q_Ndepm_esi * Tn2) / T10
                  * T9 * NdepmpNsub_inv1 * (Vbscl_dT - Vbipn_dT) ;

       Q_b0_dep_dVbs = - C_ESI / W_b0 * T7 * T8 * NdepmpNsub_inv1 * Vbscl_dVbs ;
       Q_b0_dep_dT   = - C_ESI / W_b0 * T7 * T8 * NdepmpNsub_inv1 * (Vbscl_dT - Vbipn_dT) ;

       Q_sub0_dep_dVbs = - C_ESI / W_sub0 * (NdepmpNsub_inv1 * Vbscl_dVbs - Vbscl_dVbs) ;
       Q_sub0_dep_dT   = - C_ESI / W_sub0 * (NdepmpNsub_inv1 * (Vbscl_dT - Vbipn_dT) - Vbscl_dT + Vbipn_dT) ;

       T1 = beta * (phi_s0_DEP - phi_b0_DEP) ;
       T2 = exp(T1) ;
       if( phi_s0_DEP >= phi_b0_DEP ) {
         Q_s0_dVbs = 0.0 ;
         Q_s0_dT   = - cnst0_dT * sqrt(T2 - 1.0 - T1 + 1e-15)
                   - here->HSMHV2_cnst0 / 2.0 / sqrt(T2 - 1.0 - T1 + 1e-15) * ((phi_s0_DEP - phi_b0_DEP) * T2 * beta_dT
                   - (phi_s0_DEP - phi_b0_DEP) * beta_dT) ;
       } else {
         T3 = exp( - beta * (phi_s0_DEP - Vbscl)) ;
         T4 = exp( - beta * (phi_b0_DEP - Vbscl)) ;
         T5 = sqrt(T2 - 1.0 - T1 + 1e-15 + here->HSMHV2_cnst1 * (T3 - T4)) ;
         Q_s0_dVbs = here->HSMHV2_cnst0 / 2.0 / T5 * 
                  (here->HSMHV2_cnst1 * (beta * T3 * Vbscl_dVbs - beta * T4 * Vbscl_dVbs) ) ;
         Q_s0_dT   = cnst0_dT * T5
                  + here->HSMHV2_cnst0 / 2.0 / T5 * 
                   ((phi_s0_DEP - phi_b0_DEP) * T2 * beta_dT - (phi_s0_DEP - phi_b0_DEP) * beta_dT
                   + cnst1_dT * (T3 - T4)
                   + here->HSMHV2_cnst1 * ( - (phi_s0_DEP - Vbscl) * T3 * beta_dT + beta * T3 * Vbscl_dT
                                       + (phi_b0_DEP - Vbscl) * T4 * beta_dT - beta * T4 * Vbscl_dT) ) ;
       }

       y2_dVgs = Cox_dVg * (Vgp - phi_s0_DEP) + Cox * Vgp_dVgs ;
       y2_dVds = Cox_dVd * (Vgp - phi_s0_DEP) + Cox * Vgp_dVds ;
       y2_dVbs = Cox_dVb * (Vgp - phi_s0_DEP) + Cox * Vgp_dVbs + Q_s0_dVbs + Q_b0_dep_dVbs + Q_sub0_dep_dVbs ;
       y2_dT   = Cox_dT * (Vgp - phi_s0_DEP) + Cox * Vgp_dT + Q_s0_dT + Q_b0_dep_dT + Q_sub0_dep_dT ;

       phi_s0_DEP_dVgs = - ( rev11 * y1_dVgs + rev12 * y2_dVgs ) ;
       phi_s0_DEP_dVds = - ( rev11 * y1_dVds + rev12 * y2_dVds ) ;
       phi_s0_DEP_dVbs = - ( rev11 * y1_dVbs + rev12 * y2_dVbs ) ;
       phi_s0_DEP_dT   = - ( rev11 * y1_dT + rev12 * y2_dT   ) ;

       phi_b0_DEP_dVgs = - ( rev21 * y1_dVgs + rev22 * y2_dVgs ) ;
       phi_b0_DEP_dVds = - ( rev21 * y1_dVds + rev22 * y2_dVds ) ;
       phi_b0_DEP_dVbs = - ( rev21 * y1_dVbs + rev22 * y2_dVbs ) ;
       phi_b0_DEP_dT   = - ( rev21 * y1_dT + rev22 * y2_dT   ) ;

       if( W_bsub0 > model->HSMHV2_tndep && depmode !=2 ) {
         Fn_SU_CP2(phi_b0_DEP , phi_b0_DEP , phi_s0_DEP , 0.02, 2 , T1, T2 )
         phi_b0_DEP_dVgs = phi_b0_DEP_dVgs * T1 + phi_s0_DEP_dVgs * T2 ;
         phi_b0_DEP_dVds = phi_b0_DEP_dVds * T1 + phi_s0_DEP_dVds * T2 ;
         phi_b0_DEP_dVbs = phi_b0_DEP_dVbs * T1 + phi_s0_DEP_dVbs * T2 ;
         phi_b0_DEP_dT   = phi_b0_DEP_dT * T1 + phi_s0_DEP_dT * T2 ;
       }

       phi_j0_DEP  = NdepmpNsub_inv1 * (NdepmpNsub * phi_b0_DEP + Vbscl - Vbi_DEP) ;
       phi_j0_DEP_dVgs = NdepmpNsub_inv1 * NdepmpNsub * phi_b0_DEP_dVgs ;
       phi_j0_DEP_dVds = NdepmpNsub_inv1 * NdepmpNsub * phi_b0_DEP_dVds ;
       phi_j0_DEP_dVbs = NdepmpNsub_inv1 * NdepmpNsub * phi_b0_DEP_dVbs + NdepmpNsub_inv1 * Vbscl_dVbs  ;
       phi_j0_DEP_dT   = NdepmpNsub_inv1 * NdepmpNsub * phi_b0_DEP_dT
                  + NdepmpNsub_inv1 * (Vbscl_dT - Vbipn_dT)  ;

       phib_ref = (4.0 * phi_j0_DEP * phi_j0_DEP * C_ESI2 - 8.0 * phi_j0_DEP * C_ESI2 * phi_s0_DEP
              + 4.0 * C_ESI2 * phi_s0_DEP * phi_s0_DEP
              + 4.0 * phi_j0_DEP * q_Ndepm_esi * Tn2
              + 4.0 * phi_s0_DEP * q_Ndepm_esi * Tn2
              + Ndepm2 * C_QE2 * model->HSMHV2_tndep
              * Tn2 * model->HSMHV2_tndep) / T10 ;

       phib_ref_dVgs = ( 8.0 * phi_j0_DEP * phi_j0_DEP_dVgs * C_ESI2 - 8.0 * phi_j0_DEP_dVgs * C_ESI2 * phi_s0_DEP
                       - 8.0 * phi_j0_DEP * C_ESI2 * phi_s0_DEP_dVgs + 8.0 * C_ESI2 * phi_s0_DEP * phi_s0_DEP_dVgs
            + 4.0 * phi_j0_DEP_dVgs * q_Ndepm_esi * Tn2
            + 4.0 * phi_s0_DEP_dVgs * q_Ndepm_esi * Tn2 ) / T10 ;
       phib_ref_dVds = ( 8.0 * phi_j0_DEP * phi_j0_DEP_dVds * C_ESI2 - 8.0 * phi_j0_DEP_dVds * C_ESI2 * phi_s0_DEP
                       - 8.0 * phi_j0_DEP * C_ESI2 * phi_s0_DEP_dVds + 8.0 * C_ESI2 * phi_s0_DEP * phi_s0_DEP_dVds
            + 4.0 * phi_j0_DEP_dVds * q_Ndepm_esi * Tn2
            + 4.0 * phi_s0_DEP_dVds * q_Ndepm_esi * Tn2 ) / T10 ;
       phib_ref_dVbs = ( 8.0 * phi_j0_DEP * phi_j0_DEP_dVbs * C_ESI2 - 8.0 * phi_j0_DEP_dVbs * C_ESI2 * phi_s0_DEP
                       - 8.0 * phi_j0_DEP * C_ESI2 * phi_s0_DEP_dVbs + 8.0 * C_ESI2 * phi_s0_DEP * phi_s0_DEP_dVbs
            + 4.0 * phi_j0_DEP_dVbs * q_Ndepm_esi * Tn2
            + 4.0 * phi_s0_DEP_dVbs * q_Ndepm_esi * Tn2 ) / T10 ;
       phib_ref_dT = ( 8.0 * phi_j0_DEP * phi_j0_DEP_dT * C_ESI2 - 8.0 * phi_j0_DEP_dT * C_ESI2 * phi_s0_DEP
                       - 8.0 * phi_j0_DEP * C_ESI2 * phi_s0_DEP_dT + 8.0 * C_ESI2 * phi_s0_DEP * phi_s0_DEP_dT
            + 4.0 * phi_j0_DEP_dT * q_Ndepm_esi * Tn2
            + 4.0 * phi_s0_DEP_dT * q_Ndepm_esi * Tn2 ) / T10 ;

       T1 = beta * (phi_s0_DEP - phi_b0_DEP) ;
       T1_dVgs = beta * (phi_s0_DEP_dVgs - phi_b0_DEP_dVgs) ;
       T1_dVds = beta * (phi_s0_DEP_dVds - phi_b0_DEP_dVds) ;
       T1_dVbs = beta * (phi_s0_DEP_dVbs - phi_b0_DEP_dVbs) ;
       T1_dT   = beta * (phi_s0_DEP_dT - phi_b0_DEP_dT) + beta_dT * (phi_s0_DEP - phi_b0_DEP) ;

       T2 = exp(T1) ; 
       T2_dVgs = T1_dVgs * T2 ;
       T2_dVds = T1_dVds * T2 ;
       T2_dVbs = T1_dVbs * T2 ;
       T2_dT   = T1_dT * T2 ;

       if( phi_s0_DEP >= phi_b0_DEP ) {

         T3 = sqrt(T2 - 1.0e0 - T1 + 1e-15 ) ;
         T3_dVgs = (T2_dVgs - T1_dVgs) / 2.0 / T3 ;
         T3_dVds = (T2_dVds - T1_dVds) / 2.0 / T3 ;
         T3_dVbs = (T2_dVbs - T1_dVbs) / 2.0 / T3 ;
         T3_dT   = (T2_dT - T1_dT) / 2.0 / T3 ;

         Q_s0 = - here->HSMHV2_cnst0 * T3 ;

         Q_s0_dep = 0.0 ;
         Q_sub0 = 0.0 ;
///      Qg = Cox * (Vgp - phi_s0_DEP) ;

         W_b0 = sqrt(C_2ESIpq_Ndepm * (phi_b0_DEP - phi_j0_DEP) ) ;
         Fn_SU_CP( T9 , W_b0 , model->HSMHV2_tndep, 1e-8, 2 , T4 )

         W_sub0 = sqrt(C_2ESIpq_Nsub * (phi_j0_DEP - Vbscl + Vbi_DEP) ) ;
         Q_b0_dep = T9 * q_Ndepm ;
         Q_sub0_dep = - W_sub0 * q_Nsub ;
 
        /* derivative */
         Q_s0_dVgs = - here->HSMHV2_cnst0 * T3_dVgs ;
         Q_s0_dVds = - here->HSMHV2_cnst0 * T3_dVds ;
         Q_s0_dVbs = - here->HSMHV2_cnst0 * T3_dVbs ;
         Q_s0_dT = - cnst0_dT * T3
                 - here->HSMHV2_cnst0 * T3_dT ;

         Q_n0 = Q_s0 ;
         Q_n0_dVgs = Q_s0_dVgs ;
         Q_n0_dVds = Q_s0_dVds ;
         Q_n0_dVbs = Q_s0_dVbs ;
         Q_n0_dT   = Q_s0_dT ;

         Q_b0_dep_dVgs = C_ESI / W_b0 * (phi_b0_DEP_dVgs - phi_j0_DEP_dVgs) * T4 ;
         Q_b0_dep_dVds = C_ESI / W_b0 * (phi_b0_DEP_dVds - phi_j0_DEP_dVds) * T4 ;
         Q_b0_dep_dVbs = C_ESI / W_b0 * (phi_b0_DEP_dVbs - phi_j0_DEP_dVbs) * T4 ;
         Q_b0_dep_dT   = C_ESI / W_b0 * (phi_b0_DEP_dT - phi_j0_DEP_dT) * T4 ;

         Q_sub0_dep_dVgs = - C_ESI / W_sub0 * phi_j0_DEP_dVgs ;
         Q_sub0_dep_dVds = - C_ESI / W_sub0 * phi_j0_DEP_dVds ;
         Q_sub0_dep_dVbs = - C_ESI / W_sub0 * (phi_j0_DEP_dVbs - Vbscl_dVbs) ;
         Q_sub0_dep_dT   = - C_ESI / W_sub0 * (phi_j0_DEP_dT - Vbscl_dT + Vbipn_dT) ;
         
         Q_sub0_dVgs = 0.0 ;
         Q_sub0_dVds = 0.0 ;
         Q_sub0_dVbs = 0.0 ;
         Q_sub0_dT   = 0.0 ;

         Q_s0_dep_dVgs = 0.0 ;
         Q_s0_dep_dVds = 0.0 ;
         Q_s0_dep_dVbs = 0.0 ;
         Q_s0_dep_dT   = 0.0 ;

       } else {

         T3 = exp( - beta * (phi_s0_DEP - Vbscl)) ;
         T4 = exp( - beta * (phi_b0_DEP - Vbscl)) ;
         T5 = sqrt(T2 - 1.0 - T1 + here->HSMHV2_cnst1 * (T3 - T4) + 1e-15) ;
         Q_s0 = here->HSMHV2_cnst0 * T5 ;

         T3_dVgs = - beta * T3 * phi_s0_DEP_dVgs ;
         T3_dVds = - beta * T3 * phi_s0_DEP_dVds ;
         T3_dVbs = - beta * T3 * (phi_s0_DEP_dVbs - Vbscl_dVbs) ;
         T3_dT  = - beta * T3 * (phi_s0_DEP_dT - Vbscl_dT) - (phi_s0_DEP - Vbscl) * T3 * beta_dT ;

         T4_dVgs = - beta * T4 * phi_b0_DEP_dVgs ;
         T4_dVds = - beta * T4 * phi_b0_DEP_dVds ;
         T4_dVbs = - beta * T4 * (phi_b0_DEP_dVbs - Vbscl_dVbs) ;
         T4_dT  = - beta * T4 * (phi_b0_DEP_dT - Vbscl_dT) - (phi_b0_DEP - Vbscl) * T4 * beta_dT ;

         T5_dVgs = (T2_dVgs - T1_dVgs + here->HSMHV2_cnst1 * (T3_dVgs - T4_dVgs)) / 2.0 / T5 ;
         T5_dVds = (T2_dVds - T1_dVds + here->HSMHV2_cnst1 * (T3_dVds - T4_dVds)) / 2.0 / T5 ;
         T5_dVbs = (T2_dVbs - T1_dVbs + here->HSMHV2_cnst1 * (T3_dVbs - T4_dVbs)) / 2.0 / T5 ;
         T5_dT   = (T2_dT - T1_dT + here->HSMHV2_cnst1 * (T3_dT - T4_dT) + cnst1_dT * (T3 - T4)) / 2.0 / T5 ;
 
         Q_s0_dVgs = here->HSMHV2_cnst0 * T5_dVgs ;
         Q_s0_dVds = here->HSMHV2_cnst0 * T5_dVds ;
         Q_s0_dVbs = here->HSMHV2_cnst0 * T5_dVbs ;
         Q_s0_dT   = here->HSMHV2_cnst0 * T5_dT + cnst0_dT * T5 ;
 
         if( W_bsub0 > model->HSMHV2_tndep && depmode !=2 ) {
           Q_sub0 = 0.0 ;
           Q_s0_dep = 0.0 ;

           Q_sub0_dVgs = 0.0 ;
           Q_sub0_dVds = 0.0 ;
           Q_sub0_dVbs = 0.0 ;
           Q_sub0_dT   = 0.0 ;

           Q_s0_dep_dVgs = 0.0 ;
           Q_s0_dep_dVds = 0.0 ;
           Q_s0_dep_dVbs = 0.0 ;
           Q_s0_dep_dT   = 0.0 ;
         } else {
           T3 = exp( - beta * (phi_s0_DEP - Vbscl)) ;
           T4 = exp( - beta * (phi_b0_DEP - Vbscl)) ;
           T5 = sqrt( - T1 + here->HSMHV2_cnst1 * (T3 - T4)) ;
           Q_sub0 = here->HSMHV2_cnst0 * T5 - here->HSMHV2_cnst0 * sqrt( - T1)  ;
           T6 = sqrt(T2 - 1.0e0 - T1 + 1e-15) ;
           Q_s0_dep = here->HSMHV2_cnst0 * T6 ;

           Q_sub0_dVgs = here->HSMHV2_cnst0 / 2.0 / T5 * ( - beta * (phi_s0_DEP_dVgs - phi_b0_DEP_dVgs)
              + here->HSMHV2_cnst1 * ( - beta * T3 * phi_s0_DEP_dVgs + beta * T4 * phi_b0_DEP_dVgs))
              - here->HSMHV2_cnst0 / 2.0 / sqrt( - T1) * ( - beta * (phi_s0_DEP_dVgs - phi_b0_DEP_dVgs)) ;
           Q_sub0_dVds = here->HSMHV2_cnst0 / 2.0 / T5 * ( - beta * (phi_s0_DEP_dVds - phi_b0_DEP_dVds)
              + here->HSMHV2_cnst1 * ( - beta * T3 * phi_s0_DEP_dVds + beta * T4 * phi_b0_DEP_dVds))
              - here->HSMHV2_cnst0 / 2.0 / sqrt( - T1) * ( - beta * (phi_s0_DEP_dVds - phi_b0_DEP_dVds)) ;
           Q_sub0_dVbs = here->HSMHV2_cnst0 / 2.0 / T5 * ( - beta * (phi_s0_DEP_dVbs - phi_b0_DEP_dVbs)
              + here->HSMHV2_cnst1 * ( - beta * T3 * (phi_s0_DEP_dVbs - Vbscl_dVbs) + beta * T4 * (phi_b0_DEP_dVbs - Vbscl_dVbs)))
              - here->HSMHV2_cnst0 / 2.0 / sqrt( - T1) * ( - beta * (phi_s0_DEP_dVbs - phi_b0_DEP_dVbs)) ;
           Q_sub0_dT = cnst0_dT * T5 - cnst0_dT * sqrt( - T1)
              + here->HSMHV2_cnst0 / 2.0 / T5 * ( - beta * (phi_s0_DEP_dT - phi_b0_DEP_dT) - beta_dT * (phi_s0_DEP - phi_b0_DEP)
              + cnst1_dT * (T3 - T4)
              + here->HSMHV2_cnst1 * ( - beta * T3 * (phi_s0_DEP_dT - Vbscl_dT) - beta_dT * (phi_s0_DEP - Vbscl) * T3
                                  + beta * T4 * (phi_b0_DEP_dT - Vbscl_dT) + beta_dT * (phi_b0_DEP - Vbscl) * T4))
              - here->HSMHV2_cnst0 / 2.0 / sqrt( - T1) * ( - beta * (phi_s0_DEP_dT - phi_b0_DEP_dT) - beta_dT * (phi_s0_DEP - phi_b0_DEP)) ;

           Q_s0_dep_dVgs = here->HSMHV2_cnst0 / 2.0 / T6 * beta * (phi_s0_DEP_dVgs - phi_b0_DEP_dVgs) * (T2 - 1) ;
           Q_s0_dep_dVds = here->HSMHV2_cnst0 / 2.0 / T6 * beta * (phi_s0_DEP_dVds - phi_b0_DEP_dVds) * (T2 - 1) ;
           Q_s0_dep_dVbs = here->HSMHV2_cnst0 / 2.0 / T6 * beta * (phi_s0_DEP_dVbs - phi_b0_DEP_dVbs) * (T2 - 1) ;
           Q_s0_dep_dT = cnst0_dT * T6
                 + here->HSMHV2_cnst0 / 2.0 / T6 * 
                 (beta * (phi_s0_DEP_dT - phi_b0_DEP_dT) * (T2 - 1) + beta_dT * (phi_s0_DEP - phi_b0_DEP) * (T2 - 1)) ;

         }

         Q_n0 = 0.0 ;
         Q_n0_dVgs = 0.0 ;
         Q_n0_dVds = 0.0 ;
         Q_n0_dVbs = 0.0 ;
         Q_n0_dT   = 0.0 ;

///      Qg = Cox * (Vgp - phi_s0_DEP) ;

         T1 = phi_b0_DEP - phi_j0_DEP ;
         Fn_SL_CP( T2 , T1 , 0.0 , 0.1, 2 , T0 )
         W_b0 = sqrt(C_2ESIpq_Ndepm * (T2) ) ; 
         Fn_SU_CP( T9 , W_b0 , model->HSMHV2_tndep, 1e-8, 2 , T3 )
         W_sub0 = sqrt(C_2ESIpq_Nsub * (phi_j0_DEP - Vbscl + Vbi_DEP) ) ;
         Q_b0_dep = T9 * q_Ndepm ;
         Q_sub0_dep = - W_sub0 * q_Nsub ;

         Q_b0_dep_dVgs = C_ESI / W_b0 * (phi_b0_DEP_dVgs - phi_j0_DEP_dVgs) * T0 * T3 ;
         Q_b0_dep_dVds = C_ESI / W_b0 * (phi_b0_DEP_dVds - phi_j0_DEP_dVds) * T0 * T3 ;
         Q_b0_dep_dVbs = C_ESI / W_b0 * (phi_b0_DEP_dVbs - phi_j0_DEP_dVbs) * T0 * T3 ;
         Q_b0_dep_dT   = C_ESI / W_b0 * (phi_b0_DEP_dT - phi_j0_DEP_dT) * T0 * T3 ;

         Q_sub0_dep_dVgs = - C_ESI / W_sub0 * phi_j0_DEP_dVgs ;
         Q_sub0_dep_dVds = - C_ESI / W_sub0 * phi_j0_DEP_dVds ;
         Q_sub0_dep_dVbs = - C_ESI / W_sub0 * (phi_j0_DEP_dVbs - Vbscl_dVbs) ;
         Q_sub0_dep_dT = - C_ESI / W_sub0 * (phi_j0_DEP_dT - Vbscl_dT + Vbipn_dT) ;

       }

       T1 = phi_b0_DEP - phi_j0_DEP ;
       Fn_SL_CP( T2 , T1 , 0.0 , 0.1, 2 , T0 )
       W_b0 = sqrt(C_2ESIpq_Ndepm * (T2) ) ;
       Fn_SU_CP( T9, W_b0, model->HSMHV2_tndep, 1e-8, 2 , T3 )
       W_b0_dVgs = C_ESI / q_Ndepm / W_b0 * (phi_b0_DEP_dVgs - phi_j0_DEP_dVgs) * T0 * T3 ;
       W_b0_dVds = C_ESI / q_Ndepm / W_b0 * (phi_b0_DEP_dVds - phi_j0_DEP_dVds) * T0 * T3 ;
       W_b0_dVbs = C_ESI / q_Ndepm / W_b0 * (phi_b0_DEP_dVbs - phi_j0_DEP_dVbs) * T0 * T3 ;
       W_b0_dT   = C_ESI / q_Ndepm / W_b0 * (phi_b0_DEP_dT - phi_j0_DEP_dT) * T0 * T3 ;

       T1 = phi_b0_DEP - phi_s0_DEP ;
       Fn_SL_CP( T2 , T1 , 0.0 , 0.05, 2 , T0 )
       W_s0 = sqrt(C_2ESIpq_Ndepm * (T2) ) ;

       W_s0_dVgs = C_ESI / q_Ndepm / W_s0 * (phi_b0_DEP_dVgs - phi_s0_DEP_dVgs) * T0 ;
       W_s0_dVds = C_ESI / q_Ndepm / W_s0 * (phi_b0_DEP_dVds - phi_s0_DEP_dVds) * T0 ;
       W_s0_dVbs = C_ESI / q_Ndepm / W_s0 * (phi_b0_DEP_dVbs - phi_s0_DEP_dVbs) * T0 ;
       W_s0_dT   = C_ESI / q_Ndepm / W_s0 * (phi_b0_DEP_dT - phi_s0_DEP_dT) * T0 ;

       T1 = model->HSMHV2_tndep - T9 - W_s0 ;
       Fn_SL_CP( W_res0 , T1 , 1.0e-25 , 1.0e-18, 2 , T0 )

       Qn_res0 = - W_res0 * q_Ndepm ;
       Qn_res0_dVgs = (W_s0_dVgs + W_b0_dVgs) * q_Ndepm * T0 ;
       Qn_res0_dVds = (W_s0_dVds + W_b0_dVds) * q_Ndepm * T0 ;
       Qn_res0_dVbs = (W_s0_dVbs + W_b0_dVbs) * q_Ndepm * T0 ;
       Qn_res0_dT   = (W_s0_dT + W_b0_dT) * q_Ndepm * T0 ;

       if( W_bsub0 > model->HSMHV2_tndep && depmode !=2 ) {
         Fn_SU_CP(T3 , phi_s0_DEP , phi_b0_DEP_lim , 0.8, 2 , T1 )
         T3_dVgs = phi_s0_DEP_dVgs * T1 ;
         T3_dVds = phi_s0_DEP_dVds * T1 ;
         T3_dVbs = phi_s0_DEP_dVbs * T1 ;
         T3_dT   = phi_s0_DEP_dT * T1  ;
       } else {
         Fn_SU_CP(T3 , phib_ref , phi_b0_DEP_lim , 0.8, 2 , T0 )
         T3_dVgs = phib_ref_dVgs * T0 ;
         T3_dVds = phib_ref_dVds * T0 ;
         T3_dVbs = phib_ref_dVbs * T0 ;
         T3_dT   = phib_ref_dT * T0 ;
       }

       T4 = exp(beta * (T3 - phi_b0_DEP_lim)) ;
       T5 = - C_QE * here->HSMHV2_ndepm ;
       Qn_bac0 = T5 * T4 * T9 ;
       Qn_bac0_dVgs = T5 * (beta * T4 * T3_dVgs * T9 + T4 * W_b0_dVgs) ;
       Qn_bac0_dVds = T5 * (beta * T4 * T3_dVds * T9 + T4 * W_b0_dVds) ;
       Qn_bac0_dVbs = T5 * (beta * T4 * T3_dVbs * T9 + T4 * W_b0_dVbs) ;
       Qn_bac0_dT   = T5 * ((beta * T4 * T3_dT + beta_dT * (T3 - phi_b0_DEP_lim) * T4) * T9
                           + T4 * W_b0_dT) ;


       T1 = phi_s0_DEP - phi_b0_DEP_lim ;
       Fn_SL_CP( T2 , T1 , 0.0, Ps_delta, 2 , T0 )
       T2_dVgs = phi_s0_DEP_dVgs * T0 ;
       T2_dVds = phi_s0_DEP_dVds * T0 ;
       T2_dVbs = phi_s0_DEP_dVbs * T0 ;
       T2_dT   = phi_s0_DEP_dT * T0 ;

       T3 = exp(beta * (T2)) ;
       T3_dVgs = beta * T3 * T2_dVgs ;
       T3_dVds = beta * T3 * T2_dVds ;
       T3_dVbs = beta * T3 * T2_dVbs ;
       T3_dT   = beta * T3 * T2_dT + T2 * T3 * beta_dT ;

       T4 = T3 - 1.0 - beta * T2 ;
       
       T4_dVgs = T3_dVgs - beta * T2_dVgs ;
       T4_dVds = T3_dVds - beta * T2_dVds ;
       T4_dVbs = T3_dVbs - beta * T2_dVbs ;
       T4_dT   = T3_dT - beta * T2_dT - beta_dT * T2 ;

       T5 = sqrt(T4) ;
       Q_n0_cur = - here->HSMHV2_cnst0 * T5 ;
       Q_n0_cur_dVgs = - here->HSMHV2_cnst0 / 2.0 / T5 * T4_dVgs ;
       Q_n0_cur_dVds = - here->HSMHV2_cnst0 / 2.0 / T5 * T4_dVds ;
       Q_n0_cur_dVbs = - here->HSMHV2_cnst0 / 2.0 / T5 * T4_dVbs ;
       Q_n0_cur_dT   = - here->HSMHV2_cnst0 / 2.0 / T5 * T4_dT
                      - cnst0_dT * T5 ;

       T4 = exp(beta * Ps_delta0) - 1.0 - beta * Ps_delta0 ;
       T4_dT = Ps_delta0 * exp(beta * Ps_delta0) * beta_dT - beta_dT * Ps_delta0 ;
       T5 = sqrt(T4) ;
       T5_dT = 0.5 / T5 * T4_dT ;
       Qn_delta = here->HSMHV2_cnst0 * T5 ;

       Qn_delta_dT =  cnst0_dT * T5 + here->HSMHV2_cnst0 * T5_dT ;


     /*-----------------------------------------------------------*
      * Start point of phi_sL_DEP(= phi_s0_DEP + Pds) calculation.(label)
      *-----------------*/

     /* Vdseff (begin) */
       Vdsorg = Vds ;

       if( Vds > 1e-3 ) {

         here->HSMHV2_qnsub_esi = q_Ndepm_esi ;
         T2 = here->HSMHV2_qnsub_esi / ( Cox * Cox ) ;
         T4 = - 2.0e0 * T2 / Cox ;
         T2_dVb = T4 * Cox_dVb ; 
         T2_dVd = T4 * Cox_dVd ;
         T2_dVg = T4 * Cox_dVg ;
         T2_dT  = T4 * Cox_dT  ;

         T0 = Vgp + 2.0 - beta_inv - Vbsz ;
         T0_dVg = Vgp_dVgs ;
         T0_dVd = Vgp_dVds - Vbsz_dVds ;
         T0_dVb = Vgp_dVbs - Vbsz_dVbs ;
         T0_dT = Vgp_dT - beta_inv_dT - Vbsz_dT ;

         T4 = 1.0e0 + 2.0e0 / T2 * T0 ;
         T4_dVg = 2.0 / T2 * T0_dVg - 2.0 / T2 / T2 * T0 * T2_dVg ;
         T4_dVd = 2.0 / T2 * T0_dVd - 2.0 / T2 / T2 * T0 * T2_dVd ;
         T4_dVb = 2.0 / T2 * T0_dVb - 2.0 / T2 / T2 * T0 * T2_dVb ;
         T4_dT  = 2.0 / T2 * T0_dT - 2.0 / T2 / T2 * T0 * T2_dT ;

         Fn_SL_CP( T9 , T4 , 0 , DEPQFN_dlt, 2 , T0 )
         T9_dVg = T4_dVg * T0 ;
         T9_dVd = T4_dVd * T0 ;
         T9_dVb = T4_dVb * T0 ;
         T9_dT  = T4_dT * T0 ;

         T9 +=small ; 
         T3 = sqrt( T9 ) ;
         T3_dVg = 0.5 / T3 * T9_dVg  ;
         T3_dVd = 0.5 / T3 * T9_dVd  ;
         T3_dVb = 0.5 / T3 * T9_dVb  ;
         T3_dT  = 0.5 / T3 * T9_dT  ;

         T10 = Vgp + 2.0 + T2 * ( 1.0e0 - T3 ) ;
         T10_dVb = Vgp_dVbs + T2_dVb * ( 1.0e0 - T3 ) - T2 * T3_dVb ;
         T10_dVd = Vgp_dVds + T2_dVd * ( 1.0e0 - T3 ) - T2 * T3_dVd ;
         T10_dVg = Vgp_dVgs + T2_dVg * ( 1.0e0 - T3 ) - T2 * T3_dVg ;
         T10_dT  = Vgp_dT + T2_dT * ( 1.0e0 - T3 ) - T2 * T3_dT ;

         Fn_SL_CP( T10 , T10 , DEPQFN3, 0.2, 4 , T0 )
         T10 = T10 + epsm10 ; 
         T10_dVb *=T0 ;
         T10_dVd *=T0 ;
         T10_dVg *=T0 ;
         T10_dT *= T0 ;

         T1 = Vds / T10 ;
         T2 = Fn_Pow( T1 , here->HSMHV2_ddlt - 1.0e0 ) ;
         T7 = T2 * T1 ;
         T0 = here->HSMHV2_ddlt * T2 / ( T10 * T10 ) ;
         T7_dVb = T0 * ( - Vds * T10_dVb ) ;
         T7_dVd = T0 * ( T10 - Vds * T10_dVd ) ;
         T7_dVg = T0 * ( - Vds * T10_dVg ) ;
         T7_dT =  T0 * ( - Vds * T10_dT ) ;

         T3 = 1.0 + T7 ; 
         T4 = Fn_Pow( T3 , 1.0 / here->HSMHV2_ddlt - 1.0 ) ;
         T6 = T4 * T3 ;
         T0 = T4 / here->HSMHV2_ddlt ;
         T6_dVb = T0 * T7_dVb ;
         T6_dVd = T0 * T7_dVd ;
         T6_dVg = T0 * T7_dVg ;
         T6_dT =  T0 * T7_dT ;

         Vdseff = Vds / T6 ;
         T0 = 1.0 / ( T6 * T6 ) ; 
         Vdseff0_dVbs = - Vds * T6_dVb * T0 ;
         Vdseff0_dVds = ( T6 - Vds * T6_dVd ) * T0 ;
         Vdseff0_dVgs = - Vds * T6_dVg * T0 ;
         Vdseff0_dT = - Vds * T6_dT * T0 ;

         Fn_SL_CP( Vgpp , Vgp , 0.0 , 0.5, 2 , T0 )
         Vgpp_dVgs = T0 * Vgp_dVgs ;
         Vgpp_dVds = T0 * Vgp_dVds ;
         Vgpp_dVbs = T0 * Vgp_dVbs ;
         Vgpp_dT   = T0 * Vgp_dT ;

         T1 = Vgpp * 0.8 ;
         T1_dVg = Vgpp_dVgs * 0.8 ;
         T1_dVd = Vgpp_dVds * 0.8 ;
         T1_dVb = Vgpp_dVbs * 0.8 ;
         T1_dT  = Vgpp_dT * 0.8 ;

         Fn_SU_CP3( Vds , Vdseff , Vgpp , T1, 2 , T3, T4, T5 )
         Vdseff_dVgs = Vdseff0_dVgs * T3 + Vgpp_dVgs * T4 + T1_dVg * T5 ;
         Vdseff_dVds = Vdseff0_dVds * T3 + Vgpp_dVds * T4 + T1_dVd * T5 ;
         Vdseff_dVbs = Vdseff0_dVbs * T3 + Vgpp_dVbs * T4 + T1_dVb * T5 ;
         Vdseff_dT   = Vdseff0_dT * T3 + Vgpp_dT * T4 + T1_dT * T5 ;

       } else {

         Vdseff = Vds ;
         Vdseff0_dVgs = 0.0 ;
         Vdseff0_dVds = 1.0 ;
         Vdseff0_dVbs = 0.0 ;
         Vdseff0_dT = 0.0 ;

         Vdseff_dVgs = 0.0 ;
         Vdseff_dVds = 1.0 ;
         Vdseff_dVbs = 0.0 ;
         Vdseff_dT = 0.0 ;

       }
       /* Vdseff (end) */

     /*---------------------------------------------------*
      * start of phi_sL_DEP calculation. (label)
      *--------------------------------*/

       if( Vds <= 0.0e0 ) {

         phi_sL_DEP = phi_s0_DEP ;
         phi_sL_DEP_dVgs = phi_s0_DEP_dVgs ;
         phi_sL_DEP_dVds = phi_s0_DEP_dVds ;
         phi_sL_DEP_dVbs = phi_s0_DEP_dVbs ;
         phi_sL_DEP_dT   = phi_s0_DEP_dT ;

         phi_bL_DEP = phi_b0_DEP ;
         phi_bL_DEP_dVgs = phi_b0_DEP_dVgs ;
         phi_bL_DEP_dVds = phi_b0_DEP_dVds ;
         phi_bL_DEP_dVbs = phi_b0_DEP_dVbs ;
         phi_bL_DEP_dT   = phi_b0_DEP_dT ;

         phi_jL_DEP = phi_j0_DEP ;
         phi_jL_DEP_dVgs = phi_j0_DEP_dVgs ;
         phi_jL_DEP_dVds = phi_j0_DEP_dVds ;
         phi_jL_DEP_dVbs = phi_j0_DEP_dVbs ;
         phi_jL_DEP_dT   = phi_j0_DEP_dT ;

         Q_subL = Q_sub0 ;
         Q_subL_dVgs = Q_sub0_dVgs ;
         Q_subL_dVds = Q_sub0_dVds ;
         Q_subL_dVbs = Q_sub0_dVbs ;
         Q_subL_dT   = Q_sub0_dT ;

         Q_nL = Q_n0 ;
         Q_nL_dVgs = Q_n0_dVgs ;
         Q_nL_dVds = Q_n0_dVds ;
         Q_nL_dVbs = Q_n0_dVbs ;
         Q_nL_dT   = Q_n0_dT ;

	 Q_bL_dep = Q_b0_dep ;
	 Q_bL_dep_dVgs = Q_b0_dep_dVgs ;
	 Q_bL_dep_dVds = Q_b0_dep_dVds ;
	 Q_bL_dep_dVbs = Q_b0_dep_dVbs ;
	 Q_bL_dep_dT   = Q_b0_dep_dT ;

	 Q_subL_dep = Q_sub0_dep ;
	 Q_subL_dep_dVgs = Q_sub0_dep_dVgs ;
	 Q_subL_dep_dVds = Q_sub0_dep_dVds ;
	 Q_subL_dep_dVbs = Q_sub0_dep_dVbs ;
	 Q_subL_dep_dT   = Q_sub0_dep_dT ;

	 Q_sL_dep = Q_s0_dep ;
	 Q_sL_dep_dVgs = Q_s0_dep_dVgs ;
	 Q_sL_dep_dVds = Q_s0_dep_dVds ;
	 Q_sL_dep_dVbs = Q_s0_dep_dVbs ;
	 Q_sL_dep_dT   = Q_s0_dep_dT ;

	 Q_nL_cur = Q_n0_cur ;
	 Q_nL_cur_dVgs = Q_n0_cur_dVgs ;
	 Q_nL_cur_dVds = Q_n0_cur_dVds ;
	 Q_nL_cur_dVbs = Q_n0_cur_dVbs ;
	 Q_nL_cur_dT   = Q_n0_cur_dT ;

       } else {

         W_bsubL = sqrt(C_2ESIpq_Ndepm * here->HSMHV2_nsub / (here->HSMHV2_nsub + here->HSMHV2_ndepm) * (Vds - Vbscl + Vbi_DEP)) ;

       /*---------------------------------------------------*
        * region judgement  
        *------------------*/

        /* fully depleted case */
         if( W_bsubL > model->HSMHV2_tndep ) {

           Vgp0 = Vds ;
           W_bL = model->HSMHV2_tndep ;
           phi_bL_DEP = Vds ;
           phi_bL_DEP_lim = Vds ;
           phi_jL_DEP = phi_bL_DEP - C_2ESIpq_Ndepm_inv * W_bL * W_bL ;

           Vgp0old = Vgp0 ;
           phi_jL_DEP_old = phi_jL_DEP ;

           Q_bL_dep = W_bL * q_Ndepm ;

           for ( lp_s0 = 1 ; lp_s0 <= lp_se_max + 1 ; lp_s0 ++ ) {

             W_bL = sqrt(C_2ESIpq_Ndepm * (phi_bL_DEP - phi_jL_DEP) ) ;
             Fn_SU_CP( W_bL , W_bL , model->HSMHV2_tndep , 1e-8, 2 , T0 )
             W_subL = sqrt(C_2ESIpq_Nsub * (phi_jL_DEP - Vbscl + Vbi_DEP) ) ;

             Q_bL_dep = W_bL * q_Ndepm ;
             Q_bL_dep_dPd = - C_ESI / W_bL * T0 ;
             Q_subL_dep = - W_subL * q_Nsub ;
             Q_subL_dep_dPd = - C_ESI / W_subL ;

             y1 = Cox * (Vgp0 - phi_bL_DEP) + Q_bL_dep + Q_subL_dep ;
             y11 = Cox ;
             y12 = Q_bL_dep_dPd + Q_subL_dep_dPd ;

             y2 = phi_jL_DEP - NdepmpNsub_inv1 * (NdepmpNsub * phi_bL_DEP + Vbscl - Vbi_DEP) ;
             y21 = 0.0 ;
             y22 = 1.0 ;

             dety = y11 * y22 - y21 * y12;
             rev11 = (y22) / dety ;
             rev12 = ( - y12) / dety ;
             rev21 = ( - y21) / dety ;
             rev22 = (y11) / dety ;

             if( fabs( rev11 * y1 + rev12 * y2 ) > 0.5 ) {
               Vgp0 = Vgp0 - 0.5 * Fn_Sgn( rev11 * y1 + rev12 * y2 ) ;
               phi_jL_DEP = phi_jL_DEP - 0.5 * Fn_Sgn( rev21 * y1 + rev22 * y2 ) ;
             } else {
               Vgp0 = Vgp0 - ( rev11 * y1 + rev12 * y2 ) ;
               phi_jL_DEP = phi_jL_DEP - ( rev21 * y1 + rev22 * y2 ) ;
             }

             if( fabs(Vgp0 - Vgp0old) <= ps_conv &&
                 fabs(phi_jL_DEP - phi_jL_DEP_old) <= ps_conv ) lp_s0=lp_se_max + 1 ;

             Vgp0old = Vgp0 ;
             phi_jL_DEP_old = phi_jL_DEP ;
           }
           phi_jL_DEP_acc = phi_jL_DEP ;

           W_subL = model->HSMHV2_tndep * NdepmpNsub ;
           phi_jL_DEP = C_2ESIpq_Nsub_inv * W_subL * W_subL + Vbscl - Vbi_DEP ;
           phi_bL_DEP = phi_jL_DEP + C_2ESIpq_Ndepm_inv * Tn2 ;
           phi_sL_DEP = phi_bL_DEP ;
           Psbmax = phi_bL_DEP ;
           Vgp1 = phi_bL_DEP ;
           if( Vgp > Vgp0 ) {
             depmode = 1 ; 
           } else if(Vgp > Vgp1 ) {
             depmode = 3 ;
           } else {
             depmode = 2 ;
           }

        /* else */
         } else {
           Vgp0 = Vds ;
           Vgp1 = Vgp0 ;
           Psbmax = Vgp0 ;
           phi_bL_DEP_lim = Vgp0 ;
           W_bL = W_bsubL ;
           W_subL = W_bL * NdepmpNsub ;
           phi_jL_DEP = C_2ESIpq_Nsub_inv * W_subL * W_subL + Vbscl - Vbi_DEP ;
           phi_bL_DEP = C_2ESIpq_Ndepm_inv * W_bL * W_bL + phi_jL_DEP ;
           phi_jL_DEP_acc = phi_jL_DEP ;
           if( Vgp > Vgp0 ) {
             depmode = 1 ;
           } else {
             depmode = 2 ;
           }

         }

         T1 = C_2ESI_q_Ndepm * ( Psbmax - ( - here->HSMHV2_Pb2n + Vbscl)) ;
         if ( T1 > 0.0 ) {
           vthn = - here->HSMHV2_Pb2n + Vbscl - sqrt(T1) / Cox ;
         } else {
           vthn = - here->HSMHV2_Pb2n + Vbscl ;
         }

       /*---------------------------------------------------*
        * initial potential phi_s0_DEP,phi_bL_DEP,phi_jL_DEP calculated.
        *------------------*/


        /* accumulation region */
         if( Vgp > Vgp0 ) {
           phi_jL_DEP = phi_jL_DEP_acc ;
           phi_bL_DEP = Vds ;
           phi_sL_DEP_ini = log(afact * Vgp * Vgp) / (beta + 2.0 / Vgp) + Vds ;

           if( phi_sL_DEP_ini < phi_bL_DEP_lim + ps_conv23 ) phi_sL_DEP_ini = phi_bL_DEP_lim + ps_conv23 ;

        /* fully depleted region */
         } else if( Vgp > Vgp1 ) {

           phi_sL_DEP_ini = phi_sL_DEP ;

        /* depletion & inversion */

         } else {

          /* depletion */
           if( Vgp > vthn ) {
             bfact = - 2.0 * afact * Vgp + beta ;
             cfact = afact * Vgp * Vgp - beta * phi_bL_DEP ;
             phi_bL_DEP_old = phi_bL_DEP ;
             phi_sL_DEP_ini = ( - bfact + sqrt(bfact * bfact - 4.0 * afact * cfact)) / 2.0 / afact ;
             if( phi_sL_DEP_ini > Psbmax - ps_conv23 ) phi_sL_DEP_ini = Psbmax - ps_conv23 ;
             W_sL = sqrt(C_2ESIpq_Ndepm * (phi_bL_DEP - phi_sL_DEP_ini) ) ;
             W_bL = sqrt(C_2ESIpq_Ndepm * (phi_bL_DEP - phi_jL_DEP) ) ;

             if( W_sL + W_bL > model->HSMHV2_tndep ) {
               for ( lp_s0 = 1 ; lp_s0 <= lp_se_max + 1 ; lp_s0 ++ ) {

                 y0 = W_sL + W_bL - model->HSMHV2_tndep ;

                 dydPsm = C_ESI / q_Ndepm / W_sL 
                          + C_ESI / q_Ndepm * ( 1.0 - (here->HSMHV2_ndepm
                          / here->HSMHV2_nsub) / ( 1.0 + (NdepmpNsub))) / W_bL ;

                 if( fabs(y0 / dydPsm) > 0.5 ) {
                   phi_bL_DEP = phi_bL_DEP - 0.5 * Fn_Sgn(y0 / dydPsm) ;
                 } else {
                   phi_bL_DEP = phi_bL_DEP - y0 / dydPsm ;
                 }
                 if( (phi_bL_DEP - Vbscl + Vbi_DEP) < epsm10 ) 
                    phi_bL_DEP=Vbscl - Vbi_DEP + epsm10 ;

                 cfact = afact * Vgp * Vgp - beta * phi_bL_DEP ;
                 T1 = bfact * bfact - 4.0 * afact * cfact ;
                 if( T1 > 0.0 ) {
                   phi_sL_DEP_ini = ( - bfact + sqrt(T1)) / 2.0 / afact ;
                 } else {
                   phi_sL_DEP_ini = ( - bfact) / 2.0 / afact ;
                 }

                 if( phi_sL_DEP_ini > Psbmax ) phi_sL_DEP_ini = Psbmax ;
                 if( phi_sL_DEP_ini > phi_bL_DEP ) {
                   phi_sL_DEP_ini = phi_bL_DEP - ps_conv23 ;
                   lp_s0=lp_se_max + 1 ;
                 }
                 W_sL = sqrt(C_2ESIpq_Ndepm * (phi_bL_DEP - phi_sL_DEP_ini) ) ;
                 phi_jL_DEP = ( NdepmpNsub * phi_bL_DEP 
                   + Vbscl - Vbi_DEP) / (1.0 + NdepmpNsub) ;
                 W_bL = sqrt(C_2ESIpq_Ndepm * (phi_bL_DEP - phi_jL_DEP) ) ;

                 if( fabs(phi_bL_DEP - phi_bL_DEP_old) <= 1.0e-8 ) lp_s0=lp_se_max + 1 ;
                 phi_bL_DEP_old = phi_bL_DEP ;
               }
             }

          /* inversion */
           } else {

             phi_bL_DEP = phi_b0_DEP ;
             phi_jL_DEP = phi_j0_DEP ;
             phi_sL_DEP_ini = phi_s0_DEP ;

           }

         }

         phi_b0_DEP_ini = phi_bL_DEP ;
         /*                              */
         /* solve poisson  at drain side */
         /*                              */

         flg_conv = 0 ; 

        /* accumulation */

         phi_sL_DEP = phi_sL_DEP_ini ;
         phi_bL_DEP = phi_b0_DEP_ini ;

         phi_sL_DEP_old = phi_sL_DEP ; 
         phi_bL_DEP_old = phi_bL_DEP ;

         for ( lp_s0 = 1 ; lp_s0 <= lp_se_max + 1 ; lp_s0 ++ ) {

           phi_jL_DEP  = NdepmpNsub_inv1 * (NdepmpNsub * phi_bL_DEP + Vbscl - Vbi_DEP) ;
           phi_jL_DEP_dPb = NdepmpNsub_inv1 * NdepmpNsub ;

           T1 = phi_bL_DEP - phi_jL_DEP ;
           Fn_SL_CP( T2 , T1 , 0.0 , 0.1, 2 , T7 )
           W_bL = sqrt(C_2ESIpq_Ndepm * (T2) ) ;
           Fn_SU_CP( W_bL , W_bL , model->HSMHV2_tndep , 1e-8, 2 , T8 )
           W_subL = sqrt(C_2ESIpq_Nsub * (phi_jL_DEP - Vbscl + Vbi_DEP) ) ;
           Q_bL_dep = W_bL * q_Ndepm ;
           Q_bL_dep_dPb = C_ESI / W_bL * T7 * T8 ;
           Q_bL_dep_dPd = - C_ESI / W_bL * T7 * T8 ;
           Q_subL_dep = - W_subL * q_Nsub ;
           Q_subL_dep_dPd = - C_ESI / W_subL ;

           T10 = 8.0 * q_Ndepm_esi * Tn2 ;
           phib_ref = (4.0 * phi_jL_DEP * phi_jL_DEP * C_ESI2 - 8.0 * phi_jL_DEP * C_ESI2 * phi_sL_DEP
              + 4.0 * C_ESI2 * phi_sL_DEP * phi_sL_DEP
              + 4.0 * phi_jL_DEP * q_Ndepm_esi * Tn2
              + 4.0 * phi_sL_DEP * q_Ndepm_esi * Tn2
              + Ndepm2 * C_QE2 * model->HSMHV2_tndep
              * Tn2 * model->HSMHV2_tndep) / T10 ;
           phib_ref_dPs = ( - 8.0 * phi_jL_DEP * C_ESI2 + 4.0 * C_ESI2 * phi_sL_DEP * 2.0
              + 4.0 * q_Ndepm_esi * Tn2) / T10 ;
           phib_ref_dPd = (4.0 * phi_jL_DEP * C_ESI2 * 2.0 - 8.0 * C_ESI2 * phi_sL_DEP
              + 4.0 * q_Ndepm_esi * Tn2) / T10 ;

           T1 = beta * (phi_sL_DEP - phi_bL_DEP) ;
           T2 = exp(T1) ;
           if( phi_sL_DEP >= phi_bL_DEP ) {
             Q_sL = - here->HSMHV2_cnst0 * sqrt(T2 - 1.0 - T1 + 1e-15) ;
             Q_sL_dPs = 0.5 * here->HSMHV2_cnst0 * here->HSMHV2_cnst0 / Q_sL * (beta * T2 - beta) ;
             Q_sL_dPb = - Q_sL_dPs ;
           } else {
             T3 = exp( - beta * (phi_sL_DEP - Vbscl)) ;
             T4 = exp( - beta * (phi_bL_DEP - Vbscl)) ;
             Q_sL = here->HSMHV2_cnst0 * sqrt(T2 - 1.0 - T1 + here->HSMHV2_cnst1 * (T3 - T4) + 1e-15) ;
             T5 = 0.5 * here->HSMHV2_cnst0 * here->HSMHV2_cnst0 / Q_sL ;
             Q_sL_dPs = T5 * (beta * T2 - beta + here->HSMHV2_cnst1 * ( - beta * T3) ) ;
             Q_sL_dPb = T5 * ( - beta * T2 + beta + here->HSMHV2_cnst1 * beta * T4 ) ;
           }

           Fn_SU_CP2( T1 , phib_ref , phi_bL_DEP_lim , sm_delta, 4 , T9, T11 )

           y1 = phi_bL_DEP - T1 ;
           y11 = - phib_ref_dPs * T9 ;
           y12 = 1.0 - phib_ref_dPd * phi_jL_DEP_dPb * T9 ;

           y2 = Cox * (Vgp - phi_sL_DEP) + Q_sL + Q_bL_dep + Q_subL_dep ;
           y21 = - Cox + Q_sL_dPs ;
           y22 = Q_sL_dPb + Q_bL_dep_dPb + Q_bL_dep_dPd * phi_jL_DEP_dPb + Q_subL_dep_dPd * phi_jL_DEP_dPb ;

           dety = y11 * y22 - y21 * y12;
           rev11 = (y22) / dety ;
           rev12 = ( - y12) / dety ;
           rev21 = ( - y21) / dety ;
           rev22 = (y11) / dety ;
           if( fabs( rev21 * y1 + rev22 * y2 ) > 0.5 ) {
             phi_sL_DEP = phi_sL_DEP - 0.5 * Fn_Sgn( rev11 * y1 + rev12 * y2 ) ;
             phi_bL_DEP = phi_bL_DEP - 0.5 * Fn_Sgn( rev21 * y1 + rev22 * y2 ) ;
           } else {
             phi_sL_DEP = phi_sL_DEP - ( rev11 * y1 + rev12 * y2 ) ;
             phi_bL_DEP = phi_bL_DEP - ( rev21 * y1 + rev22 * y2 ) ;
           }

           if( fabs(phi_sL_DEP - phi_sL_DEP_old) <= ps_conv &&  fabs(phi_bL_DEP - phi_bL_DEP_old) <= ps_conv ) {
             lp_s0=lp_se_max + 1 ;
             flg_conv = 1 ;
           }

           phi_sL_DEP_old = phi_sL_DEP ;
           phi_bL_DEP_old = phi_bL_DEP ;

         }
         if( flg_conv == 0 ) {
           printf( "*** warning(HiSIM_HV(%s)): Went Over Iteration Maximum(Psl)\n",model->HSMHV2modName ) ;
           printf( " Vbse   = %7.3f Vdse = %7.3f Vgse = %7.3f\n" ,Vbse , Vdse , Vgse ) ;
         } 

        /* caluculate derivative */

         y1_dVgs = - Vdseff_dVgs * T11 ;
         y1_dVds = - Vdseff_dVds * T11 ;
         y1_dVbs = - (8.0 * phi_jL_DEP * C_ESI2 - 8.0 * C_ESI2 * phi_sL_DEP
                  + 4.0 * q_Ndepm_esi * Tn2) / T10
                  * T9 * NdepmpNsub_inv1 * Vbscl_dVbs - Vdseff_dVbs * T11 ;
         y1_dT   = - (8.0 * phi_jL_DEP * C_ESI2 - 8.0 * C_ESI2 * phi_sL_DEP
                  + 4.0 * q_Ndepm_esi * Tn2) / T10
                  * T9 * NdepmpNsub_inv1 * (Vbscl_dT - Vbipn_dT) - Vdseff_dT * T11 ;

         Q_bL_dep_dVbs = - C_ESI / W_bL * T7 * T8 * NdepmpNsub_inv1 * Vbscl_dVbs ;
         Q_bL_dep_dT   = - C_ESI / W_bL * T7 * T8 * NdepmpNsub_inv1 * (Vbscl_dT - Vbipn_dT) ;

         Q_subL_dep_dVbs = - C_ESI / W_subL * (NdepmpNsub_inv1 * Vbscl_dVbs - Vbscl_dVbs) ;
         Q_subL_dep_dT   = - C_ESI / W_subL * (NdepmpNsub_inv1 * (Vbscl_dT - Vbipn_dT) - Vbscl_dT + Vbipn_dT) ;

         T1 = beta * (phi_sL_DEP - phi_bL_DEP) ;
         T2 = exp(T1) ; 
         if( phi_sL_DEP >= phi_bL_DEP ) {
           Q_sL_dVbs = 0.0 ;
           Q_sL_dT   = - cnst0_dT * sqrt(T2 - 1.0 - T1 + 1e-15)
                   - here->HSMHV2_cnst0 / 2.0 / sqrt(T2 - 1.0 - T1 + 1e-15) * ((phi_sL_DEP - phi_bL_DEP) * T2 * beta_dT
                   - (phi_sL_DEP - phi_bL_DEP) * beta_dT) ;
         } else {
           T3 = exp( - beta * (phi_sL_DEP - Vbscl)) ;
           T4 = exp( - beta * (phi_bL_DEP - Vbscl)) ;
           T5 = sqrt(T2 - 1.0 - T1 + 1e-15 + here->HSMHV2_cnst1 * (T3 - T4)) ;
           Q_sL_dVbs = here->HSMHV2_cnst0 / 2.0 / T5 * 
                  (here->HSMHV2_cnst1 * (beta * T3 * Vbscl_dVbs - beta * T4 * Vbscl_dVbs) ) ;
           Q_sL_dT   = cnst0_dT * T5
                  + here->HSMHV2_cnst0 / 2.0 / T5 * 
                   ((phi_sL_DEP - phi_bL_DEP) * T2 * beta_dT - (phi_sL_DEP - phi_bL_DEP) * beta_dT
                   + cnst1_dT * (T3 - T4)
                   + here->HSMHV2_cnst1 * ( - (phi_sL_DEP - Vbscl) * T3 * beta_dT + beta * T3 * Vbscl_dT
                                       + (phi_bL_DEP - Vbscl) * T4 * beta_dT - beta * T4 * Vbscl_dT) ) ;
         }

         y2_dVgs = Cox_dVg * (Vgp - phi_sL_DEP) + Cox * Vgp_dVgs ;
         y2_dVds = Cox_dVd * (Vgp - phi_sL_DEP) + Cox * Vgp_dVds ;
         y2_dVbs = Cox_dVb * (Vgp - phi_sL_DEP) + Cox * Vgp_dVbs + Q_sL_dVbs + Q_bL_dep_dVbs + Q_subL_dep_dVbs ;
         y2_dT   = Cox_dT * (Vgp - phi_sL_DEP) + Cox * Vgp_dT + Q_sL_dT + Q_bL_dep_dT + Q_subL_dep_dT ;

         phi_sL_DEP_dVgs = - ( rev11 * y1_dVgs + rev12 * y2_dVgs ) ;
         phi_sL_DEP_dVds = - ( rev11 * y1_dVds + rev12 * y2_dVds ) ;
         phi_sL_DEP_dVbs = - ( rev11 * y1_dVbs + rev12 * y2_dVbs ) ;
         phi_sL_DEP_dT   = - ( rev11 * y1_dT + rev12 * y2_dT   ) ;

         phi_bL_DEP_dVgs = - ( rev21 * y1_dVgs + rev22 * y2_dVgs ) ;
         phi_bL_DEP_dVds = - ( rev21 * y1_dVds + rev22 * y2_dVds ) ;
         phi_bL_DEP_dVbs = - ( rev21 * y1_dVbs + rev22 * y2_dVbs ) ;
         phi_bL_DEP_dT   = - ( rev21 * y1_dT + rev22 * y2_dT   ) ;

         if( W_bsubL > model->HSMHV2_tndep && depmode !=2 ) {
           Fn_SU_CP2(phi_bL_DEP , phi_bL_DEP , phi_sL_DEP , 0.02, 2 , T1, T2 )
           phi_bL_DEP_dVgs = phi_bL_DEP_dVgs * T1 + phi_sL_DEP_dVgs * T2 ;
           phi_bL_DEP_dVds = phi_bL_DEP_dVds * T1 + phi_sL_DEP_dVds * T2 ;
           phi_bL_DEP_dVbs = phi_bL_DEP_dVbs * T1 + phi_sL_DEP_dVbs * T2 ;
           phi_bL_DEP_dT   = phi_bL_DEP_dT * T1 + phi_sL_DEP_dT * T2 ;
         }

         phi_jL_DEP  = NdepmpNsub_inv1 * (NdepmpNsub * phi_bL_DEP + Vbscl - Vbi_DEP) ;
         phi_jL_DEP_dVgs = NdepmpNsub_inv1 * NdepmpNsub * phi_bL_DEP_dVgs ;
         phi_jL_DEP_dVds = NdepmpNsub_inv1 * NdepmpNsub * phi_bL_DEP_dVds ;
         phi_jL_DEP_dVbs = NdepmpNsub_inv1 * NdepmpNsub * phi_bL_DEP_dVbs + NdepmpNsub_inv1 * Vbscl_dVbs  ;
         phi_jL_DEP_dT   = NdepmpNsub_inv1 * NdepmpNsub * phi_bL_DEP_dT
                  + NdepmpNsub_inv1 * (Vbscl_dT - Vbipn_dT)  ;

         phib_ref = (4.0 * phi_jL_DEP * phi_jL_DEP * C_ESI2 - 8.0 * phi_jL_DEP * C_ESI2 * phi_sL_DEP
              + 4.0 * C_ESI2 * phi_sL_DEP * phi_sL_DEP
              + 4.0 * phi_jL_DEP * q_Ndepm_esi * Tn2
              + 4.0 * phi_sL_DEP * q_Ndepm_esi * Tn2
              + Ndepm2 * C_QE2 * model->HSMHV2_tndep
              * Tn2 * model->HSMHV2_tndep) / T10 ;

         phib_ref_dVgs = ( 8.0 * phi_jL_DEP * phi_jL_DEP_dVgs * C_ESI2 - 8.0 * phi_jL_DEP_dVgs * C_ESI2 * phi_sL_DEP
                       - 8.0 * phi_jL_DEP * C_ESI2 * phi_sL_DEP_dVgs + 8.0 * C_ESI2 * phi_sL_DEP * phi_sL_DEP_dVgs
            + 4.0 * phi_jL_DEP_dVgs * q_Ndepm_esi * Tn2
            + 4.0 * phi_sL_DEP_dVgs * q_Ndepm_esi * Tn2 ) / T10 ;
         phib_ref_dVds = ( 8.0 * phi_jL_DEP * phi_jL_DEP_dVds * C_ESI2 - 8.0 * phi_jL_DEP_dVds * C_ESI2 * phi_sL_DEP
                       - 8.0 * phi_jL_DEP * C_ESI2 * phi_sL_DEP_dVds + 8.0 * C_ESI2 * phi_sL_DEP * phi_sL_DEP_dVds
            + 4.0 * phi_jL_DEP_dVds * q_Ndepm_esi * Tn2
            + 4.0 * phi_sL_DEP_dVds * q_Ndepm_esi * Tn2 ) / T10 ;
         phib_ref_dVbs = ( 8.0 * phi_jL_DEP * phi_jL_DEP_dVbs * C_ESI2 - 8.0 * phi_jL_DEP_dVbs * C_ESI2 * phi_sL_DEP
                       - 8.0 * phi_jL_DEP * C_ESI2 * phi_sL_DEP_dVbs + 8.0 * C_ESI2 * phi_sL_DEP * phi_sL_DEP_dVbs
            + 4.0 * phi_jL_DEP_dVbs * q_Ndepm_esi * Tn2
            + 4.0 * phi_sL_DEP_dVbs * q_Ndepm_esi * Tn2 ) / T10 ;
         phib_ref_dT = ( 8.0 * phi_jL_DEP * phi_jL_DEP_dT * C_ESI2 - 8.0 * phi_jL_DEP_dT * C_ESI2 * phi_sL_DEP
                       - 8.0 * phi_jL_DEP * C_ESI2 * phi_sL_DEP_dT + 8.0 * C_ESI2 * phi_sL_DEP * phi_sL_DEP_dT
            + 4.0 * phi_jL_DEP_dT * q_Ndepm_esi * Tn2
            + 4.0 * phi_sL_DEP_dT * q_Ndepm_esi * Tn2 ) / T10 ;

         T1 = beta * (phi_sL_DEP - phi_bL_DEP) ;
         T1_dVgs = beta * (phi_sL_DEP_dVgs - phi_bL_DEP_dVgs) ;
         T1_dVds = beta * (phi_sL_DEP_dVds - phi_bL_DEP_dVds) ;
         T1_dVbs = beta * (phi_sL_DEP_dVbs - phi_bL_DEP_dVbs) ;
         T1_dT   = beta * (phi_sL_DEP_dT - phi_bL_DEP_dT) + beta_dT * (phi_sL_DEP - phi_bL_DEP) ;

         T2 = exp(T1) ;
         T2_dVgs = T1_dVgs * T2 ;
         T2_dVds = T1_dVds * T2 ;
         T2_dVbs = T1_dVbs * T2 ;
         T2_dT   = T1_dT * T2 ;

         if( phi_sL_DEP >= phi_bL_DEP ) {
           T3 = sqrt(T2 - 1.0e0 - T1 + 1e-15 ) ;
           T3_dVgs = (T2_dVgs - T1_dVgs) / 2.0 / T3 ;
           T3_dVds = (T2_dVds - T1_dVds) / 2.0 / T3 ;
           T3_dVbs = (T2_dVbs - T1_dVbs) / 2.0 / T3 ;
           T3_dT   = (T2_dT - T1_dT) / 2.0 / T3 ;

           Q_sL = - here->HSMHV2_cnst0 * T3 ;

           Q_sL_dep = 0.0 ;
           Q_subL = 0.0 ;
///        Qg = Cox * (Vgp - phi_sL_DEP) ;

           W_bL = sqrt(C_2ESIpq_Ndepm * (phi_bL_DEP - phi_jL_DEP) ) ;
           Fn_SU_CP( T9 , W_bL , model->HSMHV2_tndep, 1e-8, 2 , T4 )

           W_subL = sqrt(C_2ESIpq_Nsub * (phi_jL_DEP - Vbscl + Vbi_DEP) ) ;
           Q_bL_dep = T9 * q_Ndepm ;
           Q_subL_dep = - W_subL * q_Nsub ;

          /* derivative */
           Q_sL_dVgs = - here->HSMHV2_cnst0 * T3_dVgs ;
           Q_sL_dVds = - here->HSMHV2_cnst0 * T3_dVds ;
           Q_sL_dVbs = - here->HSMHV2_cnst0 * T3_dVbs ;
           Q_sL_dT = - cnst0_dT * T3
                 - here->HSMHV2_cnst0 * T3_dT ;

           Q_nL = Q_sL ;
           Q_nL_dVgs = Q_sL_dVgs ;
           Q_nL_dVds = Q_sL_dVds ;
           Q_nL_dVbs = Q_sL_dVbs ;
           Q_nL_dT   = Q_sL_dT ;

           Q_bL_dep_dVgs = C_ESI / W_bL * (phi_bL_DEP_dVgs - phi_jL_DEP_dVgs) * T4 ;
           Q_bL_dep_dVds = C_ESI / W_bL * (phi_bL_DEP_dVds - phi_jL_DEP_dVds) * T4 ;
           Q_bL_dep_dVbs = C_ESI / W_bL * (phi_bL_DEP_dVbs - phi_jL_DEP_dVbs) * T4 ;
           Q_bL_dep_dT   = C_ESI / W_bL * (phi_bL_DEP_dT - phi_jL_DEP_dT) * T4 ;

           Q_subL_dep_dVgs = - C_ESI / W_subL * phi_jL_DEP_dVgs ;
           Q_subL_dep_dVds = - C_ESI / W_subL * phi_jL_DEP_dVds ;
           Q_subL_dep_dVbs = - C_ESI / W_subL * (phi_jL_DEP_dVbs - Vbscl_dVbs) ;
           Q_subL_dep_dT   = - C_ESI / W_subL * (phi_jL_DEP_dT - Vbscl_dT + Vbipn_dT) ;

           Q_subL_dVgs = 0.0 ;
           Q_subL_dVds = 0.0 ;
           Q_subL_dVbs = 0.0 ;
           Q_subL_dT   = 0.0 ;

           Q_sL_dep_dVgs = 0.0 ;
           Q_sL_dep_dVds = 0.0 ;
           Q_sL_dep_dVbs = 0.0 ;
           Q_sL_dep_dT   = 0.0 ;

         } else {

           T3 = exp( - beta * (phi_sL_DEP - Vbscl)) ;
           T4 = exp( - beta * (phi_bL_DEP - Vbscl)) ;
           T5 = sqrt(T2 - 1.0 - T1 + here->HSMHV2_cnst1 * (T3 - T4) + 1e-15) ;
           Q_sL = here->HSMHV2_cnst0 * T5 ;

           T3_dVgs = - beta * T3 * phi_sL_DEP_dVgs ;
           T3_dVds = - beta * T3 * phi_sL_DEP_dVds ;
           T3_dVbs = - beta * T3 * (phi_sL_DEP_dVbs - Vbscl_dVbs) ;
           T3_dT  = - beta * T3 * (phi_sL_DEP_dT - Vbscl_dT) - (phi_sL_DEP - Vbscl) * T3 * beta_dT ;

           T4_dVgs = - beta * T4 * phi_bL_DEP_dVgs ;
           T4_dVds = - beta * T4 * phi_bL_DEP_dVds ;
           T4_dVbs = - beta * T4 * (phi_bL_DEP_dVbs - Vbscl_dVbs) ;
           T4_dT  = - beta * T4 * (phi_bL_DEP_dT - Vbscl_dT) - (phi_bL_DEP - Vbscl) * T4 * beta_dT ;

           T5_dVgs = (T2_dVgs - T1_dVgs + here->HSMHV2_cnst1 * (T3_dVgs - T4_dVgs)) / 2.0 / T5 ;
           T5_dVds = (T2_dVds - T1_dVds + here->HSMHV2_cnst1 * (T3_dVds - T4_dVds)) / 2.0 / T5 ;
           T5_dVbs = (T2_dVbs - T1_dVbs + here->HSMHV2_cnst1 * (T3_dVbs - T4_dVbs)) / 2.0 / T5 ;
           T5_dT   = (T2_dT - T1_dT + here->HSMHV2_cnst1 * (T3_dT - T4_dT) + cnst1_dT * (T3 - T4)) / 2.0 / T5 ;

           Q_sL_dVgs = here->HSMHV2_cnst0 * T5_dVgs ;
           Q_sL_dVds = here->HSMHV2_cnst0 * T5_dVds ;
           Q_sL_dVbs = here->HSMHV2_cnst0 * T5_dVbs ;
           Q_sL_dT   = here->HSMHV2_cnst0 * T5_dT + cnst0_dT * T5 ;

           if( W_bsubL > model->HSMHV2_tndep && depmode !=2 ) {
             Q_subL = 0.0 ;
             Q_sL_dep = 0.0 ;

             Q_subL_dVgs = 0.0 ;
             Q_subL_dVds = 0.0 ;
             Q_subL_dVbs = 0.0 ;
             Q_subL_dT   = 0.0 ;

             Q_sL_dep_dVgs = 0.0 ;
             Q_sL_dep_dVds = 0.0 ;
             Q_sL_dep_dVbs = 0.0 ;
             Q_sL_dep_dT   = 0.0 ;

           } else {
             T3 = exp( - beta * (phi_sL_DEP - Vbscl)) ;
             T4 = exp( - beta * (phi_bL_DEP - Vbscl)) ;
             T5 = sqrt( - T1 + here->HSMHV2_cnst1 * (T3 - T4)) ;
             Q_subL = here->HSMHV2_cnst0 * T5 - here->HSMHV2_cnst0 * sqrt( - T1)  ;
             T6 = sqrt(T2 - 1.0e0 - T1 + 1e-15) ;
             Q_sL_dep = here->HSMHV2_cnst0 * T6 ;

             Q_subL_dVgs = here->HSMHV2_cnst0 / 2.0 / T5 * ( - beta * (phi_sL_DEP_dVgs - phi_bL_DEP_dVgs)
                + here->HSMHV2_cnst1 * ( - beta * T3 * phi_sL_DEP_dVgs + beta * T4 * phi_bL_DEP_dVgs))
                - here->HSMHV2_cnst0 / 2.0 / sqrt( - T1) * ( - beta * (phi_sL_DEP_dVgs - phi_bL_DEP_dVgs)) ;
             Q_subL_dVds = here->HSMHV2_cnst0 / 2.0 / T5 * ( - beta * (phi_sL_DEP_dVds - phi_bL_DEP_dVds)
                + here->HSMHV2_cnst1 * ( - beta * T3 * phi_sL_DEP_dVds + beta * T4 * phi_bL_DEP_dVds))
                - here->HSMHV2_cnst0 / 2.0 / sqrt( - T1) * ( - beta * (phi_sL_DEP_dVds - phi_bL_DEP_dVds)) ;
             Q_subL_dVbs = here->HSMHV2_cnst0 / 2.0 / T5 * ( - beta * (phi_sL_DEP_dVbs - phi_bL_DEP_dVbs)
                + here->HSMHV2_cnst1 * ( - beta * T3 * (phi_sL_DEP_dVbs - Vbscl_dVbs) + beta * T4 * (phi_bL_DEP_dVbs - Vbscl_dVbs)))
                - here->HSMHV2_cnst0 / 2.0 / sqrt( - T1) * ( - beta * (phi_sL_DEP_dVbs - phi_bL_DEP_dVbs)) ;
             Q_subL_dT = cnst0_dT * T5 - cnst0_dT * sqrt( - T1)
                + here->HSMHV2_cnst0 / 2.0 / T5 * ( - beta * (phi_sL_DEP_dT - phi_bL_DEP_dT) - beta_dT * (phi_sL_DEP - phi_bL_DEP)
                + cnst1_dT * (T3 - T4)
                + here->HSMHV2_cnst1 * ( - beta * T3 * (phi_sL_DEP_dT - Vbscl_dT) - beta_dT * (phi_sL_DEP - Vbscl) * T3
                                  + beta * T4 * (phi_bL_DEP_dT - Vbscl_dT) + beta_dT * (phi_bL_DEP - Vbscl) * T4))
                - here->HSMHV2_cnst0 / 2.0 / sqrt( - T1) * ( - beta * (phi_sL_DEP_dT - phi_bL_DEP_dT) - beta_dT * (phi_sL_DEP - phi_bL_DEP)) ;

             Q_sL_dep_dVgs = here->HSMHV2_cnst0 / 2.0 / T6 * beta * (phi_sL_DEP_dVgs - phi_bL_DEP_dVgs) * (T2 - 1) ;
             Q_sL_dep_dVds = here->HSMHV2_cnst0 / 2.0 / T6 * beta * (phi_sL_DEP_dVds - phi_bL_DEP_dVds) * (T2 - 1) ;
             Q_sL_dep_dVbs = here->HSMHV2_cnst0 / 2.0 / T6 * beta * (phi_sL_DEP_dVbs - phi_bL_DEP_dVbs) * (T2 - 1) ;
             Q_sL_dep_dT = cnst0_dT * T6
                 + here->HSMHV2_cnst0 / 2.0 / T6 * 
                 (beta * (phi_sL_DEP_dT - phi_bL_DEP_dT) * (T2 - 1) + beta_dT * (phi_sL_DEP - phi_bL_DEP) * (T2 - 1)) ;

           }

           Q_nL = 0.0 ;
           Q_nL_dVgs = 0.0 ;
           Q_nL_dVds = 0.0 ;
           Q_nL_dVbs = 0.0 ;
           Q_nL_dT   = 0.0 ;

///        Qg = Cox * (Vgp - phi_sL_DEP) ;

           T1 = phi_bL_DEP - phi_jL_DEP ;
           Fn_SL_CP( T2 , T1 , 0.0 , 0.1, 2 , T0 )
           W_bL = sqrt(C_2ESIpq_Ndepm * (T2) ) ;
           Fn_SU_CP( T9 , W_bL , model->HSMHV2_tndep, 1e-8, 2 , T3 )
           W_subL = sqrt(C_2ESIpq_Nsub * (phi_jL_DEP - Vbscl + Vbi_DEP) ) ;
           Q_bL_dep = T9 * q_Ndepm ;
           Q_subL_dep = - W_subL * q_Nsub ;

           Q_bL_dep_dVgs = C_ESI / W_bL * (phi_bL_DEP_dVgs - phi_jL_DEP_dVgs) * T0 * T3 ;
           Q_bL_dep_dVds = C_ESI / W_bL * (phi_bL_DEP_dVds - phi_jL_DEP_dVds) * T0 * T3 ;
           Q_bL_dep_dVbs = C_ESI / W_bL * (phi_bL_DEP_dVbs - phi_jL_DEP_dVbs) * T0 * T3 ;
           Q_bL_dep_dT   = C_ESI / W_bL * (phi_bL_DEP_dT - phi_jL_DEP_dT) * T0 * T3 ;

           Q_subL_dep_dVgs = - C_ESI / W_subL * phi_jL_DEP_dVgs ;
           Q_subL_dep_dVds = - C_ESI / W_subL * phi_jL_DEP_dVds ;
           Q_subL_dep_dVbs = - C_ESI / W_subL * (phi_jL_DEP_dVbs - Vbscl_dVbs) ;
           Q_subL_dep_dT = - C_ESI / W_subL * (phi_jL_DEP_dT - Vbscl_dT + Vbipn_dT) ;

         }


         T1 = phi_sL_DEP - phi_bL_DEP_lim ;
         Fn_SL_CP( T2 , T1 , 0.0, Ps_delta, 2 , T0 )
         T2_dVgs = (phi_sL_DEP_dVgs - Vdseff_dVgs) * T0 ;
         T2_dVds = (phi_sL_DEP_dVds - Vdseff_dVds) * T0 ;
         T2_dVbs = (phi_sL_DEP_dVbs - Vdseff_dVbs) * T0 ;
         T2_dT   = (phi_sL_DEP_dT - Vdseff_dT) * T0 ; 

         T3 = exp(beta * (T2)) ;
         T3_dVgs = beta * T3 * T2_dVgs ;
         T3_dVds = beta * T3 * T2_dVds ;
         T3_dVbs = beta * T3 * T2_dVbs ;
         T3_dT   = beta * T3 * T2_dT + T2 * T3 * beta_dT ;

         T4 = T3 - 1.0 - beta * T2 ;

         T4_dVgs = T3_dVgs - beta * T2_dVgs ;
         T4_dVds = T3_dVds - beta * T2_dVds ;
         T4_dVbs = T3_dVbs - beta * T2_dVbs ;
         T4_dT   = T3_dT - beta * T2_dT - beta_dT * T2 ;

         T5 = sqrt(T4) ;
         Q_nL_cur = - here->HSMHV2_cnst0 * T5 ;
         Q_nL_cur_dVgs = - here->HSMHV2_cnst0 / 2.0 / T5 * T4_dVgs ;
         Q_nL_cur_dVds = - here->HSMHV2_cnst0 / 2.0 / T5 * T4_dVds ;
         Q_nL_cur_dVbs = - here->HSMHV2_cnst0 / 2.0 / T5 * T4_dVbs ;
         Q_nL_cur_dT   = - here->HSMHV2_cnst0 / 2.0 / T5 * T4_dT
                        - cnst0_dT * T5 ;

       } 

     /*---------------------------------------------------*
      * Assign Pds.
      *-----------------*/
       Ps0 = phi_s0_DEP ;
       Psl = phi_sL_DEP ;
       Pds = phi_sL_DEP - phi_s0_DEP + epsm10 ;

       Pds_dVgs = phi_sL_DEP_dVgs - phi_s0_DEP_dVgs ;
       Pds_dVds = phi_sL_DEP_dVds - phi_s0_DEP_dVds ;
       Pds_dVbs = phi_sL_DEP_dVbs - phi_s0_DEP_dVbs ;
       Pds_dT   = phi_sL_DEP_dT - phi_s0_DEP_dT ;

       if( Pds < 0.0 ) { // take care of numerical noise //
         Pds = 0.0 ; 
         Pds_dVgs = 0.0 ;
         Pds_dVds = 0.0 ;
         Pds_dVbs = 0.0 ;
         Pds_dT   = 0.0 ;

         phi_sL_DEP = phi_s0_DEP ;
         phi_sL_DEP_dVgs = phi_s0_DEP_dVgs ;
         phi_sL_DEP_dVds = phi_s0_DEP_dVds ;
         phi_sL_DEP_dVbs = phi_s0_DEP_dVbs ;
         phi_sL_DEP_dT = phi_s0_DEP_dT ;

         Idd = 0.0 ;
         Idd_dVgs = 0.0 ;
         Idd_dVds = 0.0 ;
         Idd_dVbs = 0.0 ;
         Idd_dT   = 0.0 ;

       } else {

         T1 = - (Q_s0 + Q_sL) ;
         T1_dVgs = - (Q_s0_dVgs + Q_sL_dVgs) ;
         T1_dVds = - (Q_s0_dVds + Q_sL_dVds) ;
         T1_dVbs = - (Q_s0_dVbs + Q_sL_dVbs) ;
         T1_dT   = - (Q_s0_dT + Q_sL_dT) ;

         Fn_SL_CP3( Qn_drift , T1 , 0.0, Qn_delta , 2 , T3, T4, T5 )
         Qn_drift_dVgs = T1_dVgs * T3 ;
         Qn_drift_dVds = T1_dVds * T3 ;
         Qn_drift_dVbs = T1_dVbs * T3 ;
         Qn_drift_dT   = T1_dT * T3 + Qn_delta_dT * T5 ;

         Idd_drift =  beta * Qn_drift / 2.0 * Pds ;
         Idd_diffu = - ( - Q_nL_cur + Q_n0_cur);

         Idd = Idd_drift + Idd_diffu ;
         Idd_dVgs = beta * Qn_drift_dVgs / 2.0 * Pds + beta * Qn_drift / 2.0 * Pds_dVgs
                    - ( - Q_nL_cur_dVgs + Q_n0_cur_dVgs ) ;
         Idd_dVds = beta * Qn_drift_dVds / 2.0 * Pds + beta * Qn_drift / 2.0 * Pds_dVds
                    - ( - Q_nL_cur_dVds + Q_n0_cur_dVds ) ;
         Idd_dVbs = beta * Qn_drift_dVbs / 2.0 * Pds + beta * Qn_drift / 2.0 * Pds_dVbs
                    - ( - Q_nL_cur_dVbs + Q_n0_cur_dVbs ) ;
         Idd_dT   = (beta_dT * Qn_drift + beta * Qn_drift_dT) / 2.0 * Pds
                    + beta * Qn_drift / 2.0 * Pds_dT - ( - Q_nL_cur_dT + Q_n0_cur_dT ) ;

       } 


       Qiu = - Q_n0_cur ;
       Qiu_dVgs = - Q_n0_cur_dVgs ;
       Qiu_dVds = - Q_n0_cur_dVds ;
       Qiu_dVbs = - Q_n0_cur_dVbs ;
       Qiu_dT   = - Q_n0_cur_dT ;

       Lch = Leff ;

      /*-----------------------------------------------------------*
       * Muun : universal mobility.  (CGS unit)
       *-----------------*/

       T2 = here->HSMHV2_ninv_o_esi ;

       T0 = here->HSMHV2_ninvd ;
       T4 = 1.0 + ( phi_sL_DEP - phi_s0_DEP ) * T0 ;
       T4_dVb = ( phi_sL_DEP_dVbs - phi_s0_DEP_dVbs ) * T0 ;
       T4_dVd = ( phi_sL_DEP_dVds - phi_s0_DEP_dVds ) * T0 ;
       T4_dVg = ( phi_sL_DEP_dVgs - phi_s0_DEP_dVgs ) * T0 ;
       T4_dT =  ( phi_sL_DEP_dT - phi_s0_DEP_dT ) * T0 + ( phi_sL_DEP - phi_s0_DEP ) * ninvd_dT ;

       T5     = T2 * Qiu ;
       T5_dVb = T2 * Qiu_dVbs ;
       T5_dVd = T2 * Qiu_dVds ;
       T5_dVg = T2 * Qiu_dVgs ;
       T5_dT  = T2 * Qiu_dT   ;

       T3     = T5 / T4 ;
       T3_dVb = ( - T4_dVb * T5 + T4 * T5_dVb ) / T4 / T4 ;
       T3_dVd = ( - T4_dVd * T5 + T4 * T5_dVd ) / T4 / T4 ;
       T3_dVg = ( - T4_dVg * T5 + T4 * T5_dVg ) / T4 / T4 ;
       T3_dT  = ( - T4_dT * T5 + T4 * T5_dT  ) / T4 / T4 ;

       Eeff = T3 ;
       Eeff_dVbs = T3_dVb ;
       Eeff_dVds = T3_dVd ;
       Eeff_dVgs = T3_dVg ;
       Eeff_dT  = T3_dT ;

       T5 = Fn_Pow( Eeff , model->HSMHV2_mueph0 - 1.0e0 ) ;
       T8 = T5 * Eeff ;
       T7 = Fn_Pow( Eeff , here->HSMHV2_muesr - 1.0e0 ) ;
       T6 = T7 * Eeff ;
       T8_dT = model->HSMHV2_mueph0 * T5 * Eeff_dT ;
       T6_dT = here->HSMHV2_muesr * T7 * Eeff_dT ;

       T9 = C_QE * C_m2cm_p2 ;
       Rns = Qiu / T9 ;
       Rns_dT = Qiu_dT / T9 ;

       T1 = 1.0e0 / ( pParam->HSMHV2_muecb0 + pParam->HSMHV2_muecb1 * Rns / 1.0e11 )
         + here->HSMHV2_mphn0 * T8 + T6 / pParam->HSMHV2_muesr1 ;


       T1_dT = - 1.0e0 / ( pParam->HSMHV2_muecb0 + pParam->HSMHV2_muecb1 * Rns / 1.0e11 )
         / ( pParam->HSMHV2_muecb0 + pParam->HSMHV2_muecb1 * Rns / 1.0e11 )
         * pParam->HSMHV2_muecb1 * Rns_dT / 1.0e11
         + here->HSMHV2_mphn0 * T8_dT + mphn0_dT * T8 + T6_dT / pParam->HSMHV2_muesr1 ;

       Muun = 1.0e0 / T1 ;
       Muun_dT = - Muun / T1 * T1_dT ;

       T1 = 1.0e0 / ( T1 * T1 ) ;
       T2 = pParam->HSMHV2_muecb0 + pParam->HSMHV2_muecb1 * Rns / 1.0e11 ;
       T2 = 1.0e0 / ( T2 * T2 ) ;
       T3 = here->HSMHV2_mphn1 * T5 ;
       T4 = here->HSMHV2_muesr * T7 / pParam->HSMHV2_muesr1 ;
       T5 = - 1.0e-11 * pParam->HSMHV2_muecb1 / C_QE * T2 / C_m2cm_p2 ;
       Muun_dVbs = - ( T5 * Qiu_dVbs
                    + Eeff_dVbs * T3 + Eeff_dVbs * T4 ) * T1 ;
       Muun_dVds = - ( T5 * Qiu_dVds
                    + Eeff_dVds * T3 + Eeff_dVds * T4 ) * T1 ;
       Muun_dVgs = - ( T5 * Qiu_dVgs
                    + Eeff_dVgs * T3 + Eeff_dVgs * T4 ) * T1 ;

      /*  Change to MKS unit */
       Muun /=C_m2cm_p2 ;
       Muun_dT /=C_m2cm_p2 ;
       Muun_dVbs /=C_m2cm_p2 ;
       Muun_dVds /=C_m2cm_p2 ;
       Muun_dVgs /=C_m2cm_p2 ;

      /*-----------------------------------------------------------*
       * Mu : mobility 
       *-----------------*/
       T2 = beta * (Qiu + small) * Lch ;

       T1 = 1.0e0 / T2 ;
       T3 = T1 * T1 ;
       T4 = - beta * T3 ;
       T5 = T4 * Lch ;
       T6 = T4 * (Qiu + small) ;
       T1_dVb = ( T5 * Qiu_dVbs ) ;
       T1_dVd = ( T5 * Qiu_dVds ) ;
       T1_dVg = ( T5 * Qiu_dVgs ) ;
       T2_dT = beta_dT * (Qiu + small) * Lch + beta * Qiu_dT * Lch  ;
       T1_dT = - T1 / T2 * T2_dT ;

       TY = Idd * T1 ;
       TY_dVbs = Idd_dVbs * T1 + Idd * T1_dVb ;
       TY_dVds = Idd_dVds * T1 + Idd * T1_dVd ;
       TY_dVgs = Idd_dVgs * T1 + Idd * T1_dVg ;
       TY_dT = Idd_dT * T1 + Idd * T1_dT ;

       T2 = 0.2 * Vmax / Muun ;
       T3 = - T2 / Muun ;
       T2_dVb = T3 * Muun_dVbs ;
       T2_dVd = T3 * Muun_dVds ;
       T2_dVg = T3 * Muun_dVgs ;
       T2_dT = 0.2 * ( Vmax_dT * Muun - Muun_dT * Vmax ) / ( Muun * Muun ) ;

       Ey = sqrt( TY * TY + T2 * T2 ) ;
       T4 = 1.0 / Ey ;
       Ey_dVbs = T4 * ( TY * TY_dVbs + T2 * T2_dVb ) ;
       Ey_dVds = T4 * ( TY * TY_dVds + T2 * T2_dVd ) ;
       Ey_dVgs = T4 * ( TY * TY_dVgs + T2 * T2_dVg ) ;
       Ey_dT = T4 * ( TY * TY_dT + T2 * T2_dT ) ;

       Em = Muun * Ey ;
       Em_dVbs = Muun_dVbs * Ey + Muun * Ey_dVbs ;
       Em_dVds = Muun_dVds * Ey + Muun * Ey_dVds ;
       Em_dVgs = Muun_dVgs * Ey + Muun * Ey_dVgs ;
       Em_dT = Ey * Muun_dT + Ey_dT * Muun ;

       T1  = Em / Vmax ; 
       T1_dT = ( Em_dT * Vmax - Vmax_dT * Em ) / ( Vmax * Vmax );
    
       Ey_suf = Ey ;
       Ey_suf_dVgs = Ey_dVgs ;
       Ey_suf_dVds = Ey_dVds ;
       Ey_suf_dVbs = Ey_dVbs ;
       Ey_suf_dT   = Ey_dT ;

      /* note: model->HSMHV2_bb = 2 (electron) ;1 (hole) */
       if ( 1.0e0 - epsm10 <= model->HSMHV2_bb && model->HSMHV2_bb <= 1.0e0 + epsm10 ) {
         T3 = 1.0e0 ;
         T3_dT = 0.0e0 ; 
       } else if ( 2.0e0 - epsm10 <= model->HSMHV2_bb && model->HSMHV2_bb <= 2.0e0 + epsm10 ) {
         T3 = T1 ;
         T3_dT = T1_dT ;
       } else {
         T3 = Fn_Pow( T1 , model->HSMHV2_bb - 1.0e0 ) ;
         T3_dT = ( model->HSMHV2_bb - 1.0e0 ) * Fn_Pow( T1 , model->HSMHV2_bb - 2.0e0 ) * T1_dT ;
       }
       T2 = T1 * T3 ;
       T4 = 1.0e0 + T2 ;
       T2_dT = T1 * T3_dT + T3 * T1_dT ;
       T4_dT = T2_dT ;

       if ( 1.0e0 - epsm10 <= model->HSMHV2_bb && model->HSMHV2_bb <= 1.0e0 + epsm10 ) {
         T5 = 1.0 / T4 ;
         T6 = T5 / T4 ; 
         T5_dT = - T5 * T5 * T4_dT ; 
         T6_dT = T5 * T5 * ( T5_dT * T4 - T5 * T4_dT ) ;
       } else if ( 2.0e0 - epsm10 <= model->HSMHV2_bb && model->HSMHV2_bb <= 2.0e0 + epsm10 ) {
         T5 = 1.0 / sqrt( T4 ) ;
         T6 = T5 / T4 ;
         T5_dT = - 0.5e0 / ( T4 * sqrt(T4)) * T4_dT ;
         T6_dT = ( T5_dT * T4 - T5 * T4_dT ) / T4 / T4 ;
       } else {
         T6 = Fn_Pow( T4 , ( - 1.0e0 / model->HSMHV2_bb - 1.0e0 ) ) ;
         T5 = T4 * T6 ;
         T6_dT =( - 1.0e0 / model->HSMHV2_bb - 1.0e0 ) * Fn_Pow( T4 , ( - 1.0e0 / model->HSMHV2_bb - 2.0e0 ) ) * T4_dT ;
         T5_dT = T4_dT * T6 + T4 * T6_dT ;
       }

       T7 = Muun / Vmax * T6 * T3 ;

       Mu = Muun * T5 ;

       Mu_dVbs = Muun_dVbs * T5 - T7 * Em_dVbs ;
       Mu_dVds = Muun_dVds * T5 - T7 * Em_dVds ;
       Mu_dVgs = Muun_dVgs * T5 - T7 * Em_dVgs ;
       Mu_dT = Muun_dT * T5 + Muun * T5_dT ;

     //-----------------------------------------------------------*
     //*  resistor region current.  (CGS unit)
     //*-----------------//

       if( Vdsorg > 1e-3 ) {

         T2 = here->HSMHV2_qnsub_esi / ( Cox * Cox ) ;
         T4 = - 2.0e0 * T2 / Cox ;
         T2_dVb = T4 * Cox_dVb ;
         T2_dVd = T4 * Cox_dVd ;
         T2_dVg = T4 * Cox_dVg ;
         T2_dT  = T4 * Cox_dT  ;

         T0 = Vgp + model->HSMHV2_depvdsef1 - beta_inv - Vbsz ;
         T0_dVg = Vgp_dVgs ;
         T0_dVd = Vgp_dVds - Vbsz_dVds ;
         T0_dVb = Vgp_dVbs - Vbsz_dVbs ;
         T0_dT = Vgp_dT - beta_inv_dT - Vbsz_dT ;

         T4 = 1.0e0 + 2.0e0 / T2 * T0 ;
         T4_dVg = 2.0 / T2 * T0_dVg - 2.0 / T2 / T2 * T0 * T2_dVg ;
         T4_dVd = 2.0 / T2 * T0_dVd - 2.0 / T2 / T2 * T0 * T2_dVd ;
         T4_dVb = 2.0 / T2 * T0_dVb - 2.0 / T2 / T2 * T0 * T2_dVb ;
         T4_dT  = 2.0 / T2 * T0_dT - 2.0 / T2 / T2 * T0 * T2_dT ;

         Fn_SL_CP( T9 , T4 , 0 , DEPQFN_dlt, 2 , T0 )
         T9_dVg = T4_dVg * T0 ;
         T9_dVd = T4_dVd * T0 ;
         T9_dVb = T4_dVb * T0 ;
         T9_dT  = T4_dT * T0 ;

         T9 +=small ;
         T3 = sqrt( T9 ) ;
         T3_dVg = 0.5 / T3 * T9_dVg  ;
         T3_dVd = 0.5 / T3 * T9_dVd  ;
         T3_dVb = 0.5 / T3 * T9_dVb  ;
         T3_dT  = 0.5 / T3 * T9_dT  ;

         T10 = Vgp + model->HSMHV2_depvdsef1 + T2 * ( 1.0e0 - T3 ) ;
         T10 = T10 * model->HSMHV2_depvdsef2 ;
         T10_dVb = (Vgp_dVbs + T2_dVb * ( 1.0e0 - T3 ) - T2 * T3_dVb)
                   * model->HSMHV2_depvdsef2 ;
         T10_dVd = (Vgp_dVds + T2_dVd * ( 1.0e0 - T3 ) - T2 * T3_dVd)
                   * model->HSMHV2_depvdsef2 ; 
         T10_dVg = (Vgp_dVgs + T2_dVg * ( 1.0e0 - T3 ) - T2 * T3_dVg) 
                   * model->HSMHV2_depvdsef2 ;
         T10_dT  = (Vgp_dT + T2_dT * ( 1.0e0 - T3 ) - T2 * T3_dT) 
                   * model->HSMHV2_depvdsef2 ;

         Fn_SL_CP( T10 , T10 , model->HSMHV2_depleak, 4.0, 4 , T0 )
         T10 = T10 + epsm10 ;
         T10_dVb *=T0 ;
         T10_dVd *=T0 ;
         T10_dVg *=T0 ;
         T10_dT *= T0 ;

         T1 = Vdsorg / T10 ;
         T2 = Fn_Pow( T1 , here->HSMHV2_ddlt - 1.0e0 ) ;
         T7 = T2 * T1 ;
         T0 = here->HSMHV2_ddlt * T2 / ( T10 * T10 ) ;
         T7_dVb = T0 * ( - Vdsorg * T10_dVb ) ;
         T7_dVd = T0 * ( T10 - Vdsorg * T10_dVd ) ;
         T7_dVg = T0 * ( - Vdsorg * T10_dVg ) ;
         T7_dT =  T0 * ( - Vdsorg * T10_dT ) ;

         T3 = 1.0 + T7 ; 
         T4 = Fn_Pow( T3 , 1.0 / here->HSMHV2_ddlt - 1.0 ) ;
         T6 = T4 * T3 ;
         T0 = T4 / here->HSMHV2_ddlt ;
         T6_dVb = T0 * T7_dVb ;
         T6_dVd = T0 * T7_dVd ;
         T6_dVg = T0 * T7_dVg ;
         T6_dT =  T0 * T7_dT ;

         Vdseff0 = Vdsorg / T6 ;
         T0 = 1.0 / ( T6 * T6 ) ; 
         Vdseff0_dVbs = - Vdsorg * T6_dVb * T0 ;
         Vdseff0_dVds = ( T6 - Vdsorg * T6_dVd ) * T0 ;
         Vdseff0_dVgs = - Vdsorg * T6_dVg * T0 ;
         Vdseff0_dT = - Vdsorg * T6_dT * T0 ;

       } else {

         Vdseff0 = Vdsorg ;
         Vdseff0_dVgs = 0.0 ;
         Vdseff0_dVds = 1.0 ;
         Vdseff0_dVbs = 0.0 ;
         Vdseff0_dT = 0.0 ;

       }
 
       T0 = here->HSMHV2_ninvd ;
       T4 = 1.0 + ( phi_sL_DEP - phi_s0_DEP ) * T0 ;
       T4_dVb = ( phi_sL_DEP_dVbs - phi_s0_DEP_dVbs ) * T0 ;
       T4_dVd = ( phi_sL_DEP_dVds - phi_s0_DEP_dVds ) * T0 ;
       T4_dVg = ( phi_sL_DEP_dVgs - phi_s0_DEP_dVgs ) * T0 ;
       T4_dT =  ( phi_sL_DEP_dT - phi_s0_DEP_dT ) * T0 + ( phi_sL_DEP - phi_s0_DEP ) * ninvd_dT ;

       Qiu = - Qn_res0 ;
       Qiu_dVgs = - Qn_res0_dVgs ;
       Qiu_dVds = - Qn_res0_dVds ;
       Qiu_dVbs = - Qn_res0_dVbs ;
       Qiu_dT   = - Qn_res0_dT ;

       T5     = Qiu ;
       T5_dVb = Qiu_dVbs ;
       T5_dVd = Qiu_dVds ;
       T5_dVg = Qiu_dVgs ;
       T5_dT  = Qiu_dT   ;

       T3     = T5 / T4 ;
       T3_dVb = ( - T4_dVb * T5 + T4 * T5_dVb ) / T4 / T4 ;
       T3_dVd = ( - T4_dVd * T5 + T4 * T5_dVd ) / T4 / T4 ;
       T3_dVg = ( - T4_dVg * T5 + T4 * T5_dVg ) / T4 / T4 ;
       T3_dT  = ( - T4_dT * T5 + T4 * T5_dT  ) / T4 / T4 ;

       Eeff = T3 ;
       Eeff_dVbs = T3_dVb ;
       Eeff_dVds = T3_dVd ;
       Eeff_dVgs = T3_dVg ;
       Eeff_dT  = T3_dT ;

       T5 = Fn_Pow( Eeff , model->HSMHV2_depmueph0 - 1.0e0 ) ;
       T8 = T5 * Eeff ;
       T8_dT = model->HSMHV2_mueph0 * T5 * Eeff_dT ;

       T9 = C_QE * C_m2cm_p2 ;
       Rns = Qiu / T9 ;
       Rns_dT = Qiu_dT / T9 ;

       T1 = 1.0e0 / ( model->HSMHV2_depmue0 + model->HSMHV2_depmue1 * Rns / 1.0e11 )
         + here->HSMHV2_depmphn0 * T8  ;

       T1_dT = - 1.0e0 / ( model->HSMHV2_depmue0 + model->HSMHV2_depmue1 * Rns / 1.0e11 )
         / ( model->HSMHV2_depmue0 + model->HSMHV2_depmue1 * Rns / 1.0e11 )
         * model->HSMHV2_depmue1 * Rns_dT / 1.0e11
         + here->HSMHV2_depmphn0 * T8_dT + depmphn0_dT * T8  ;

       Muun = 1.0e0 / T1 ;
       Muun_dT = - Muun / T1 * T1_dT ;

       T1 = 1.0e0 / ( T1 * T1 ) ;
       T2 = model->HSMHV2_depmue0 + model->HSMHV2_depmue1 * Rns / 1.0e11 ;
       T2 = 1.0e0 / ( T2 * T2 ) ;
       T3 = here->HSMHV2_depmphn1 * T5 ;
       T5 = - 1.0e-11 * model->HSMHV2_depmue1 / C_QE * T2 / C_m2cm_p2 ;
       Muun_dVbs = - ( T5 * Qiu_dVbs
                    + Eeff_dVbs * T3 ) * T1 ;
       Muun_dVds = - ( T5 * Qiu_dVds
                    + Eeff_dVds * T3 ) * T1 ;
       Muun_dVgs = - ( T5 * Qiu_dVgs
                    + Eeff_dVgs * T3 ) * T1 ;

      /*  Change to MKS unit */
       Muun /=C_m2cm_p2 ;
       Muun_dT /=C_m2cm_p2 ;
       Muun_dVbs /=C_m2cm_p2 ;
       Muun_dVds /=C_m2cm_p2 ;
       Muun_dVgs /=C_m2cm_p2 ;

       Edri = Vdseff0 / Lch ;
       Edri_dVgs = Vdseff0_dVgs / Lch ;
       Edri_dVds = Vdseff0_dVds / Lch ;
       Edri_dVbs = Vdseff0_dVbs / Lch ;
       Edri_dT   = Vdseff0_dT / Lch ;

       T1 = Muun * Edri / here->HSMHV2_depvmax ;
       T1_dVgs = (Muun_dVgs * Edri + Muun * Edri_dVgs) / here->HSMHV2_depvmax ;
       T1_dVds = (Muun_dVds * Edri + Muun * Edri_dVds) / here->HSMHV2_depvmax ;
       T1_dVbs = (Muun_dVbs * Edri + Muun * Edri_dVbs) / here->HSMHV2_depvmax ;
       T1_dT   = (Muun_dT * Edri + Muun * Edri_dT) / here->HSMHV2_depvmax
                 - Muun * Edri / here->HSMHV2_depvmax / here->HSMHV2_depvmax * depVmax_dT ;
       
       T1 = T1 + small ;
       T2 = Fn_Pow(T1,model->HSMHV2_depbb) ;
       T2_dVgs = model->HSMHV2_depbb * T1_dVgs / T1 * T2 ;
       T2_dVds = model->HSMHV2_depbb * T1_dVds / T1 * T2 ;
       T2_dVbs = model->HSMHV2_depbb * T1_dVbs / T1 * T2 ;
       T2_dT   = model->HSMHV2_depbb * T1_dT / T1 * T2 ;

       T3 = 1.0 + T2 ;
       T4 = Fn_Pow(T3,1.0 / model->HSMHV2_depbb) ;
       T4_dVgs = 1.0 / model->HSMHV2_depbb * T2_dVgs / T3 * T4 ;
       T4_dVds = 1.0 / model->HSMHV2_depbb * T2_dVds / T3 * T4 ;
       T4_dVbs = 1.0 / model->HSMHV2_depbb * T2_dVbs / T3 * T4 ;
       T4_dT   = 1.0 / model->HSMHV2_depbb * T2_dT / T3 * T4 ;

       Mu_res = Muun / T4 ;
       Mu_res_dVgs = Muun_dVgs / T4 - Muun / T4 / T4 * T4_dVgs ;
       Mu_res_dVds = Muun_dVds / T4 - Muun / T4 / T4 * T4_dVds ;
       Mu_res_dVbs = Muun_dVbs / T4 - Muun / T4 / T4 * T4_dVbs ;
       Mu_res_dT = Muun_dT / T4 - Muun / T4 / T4 * T4_dT ;

       Ids_res = here->HSMHV2_weff_nf * ( - Qn_res0) * Mu_res * Edri ;
       Ids_res_dVgs = here->HSMHV2_weff_nf * ( - Qn_res0_dVgs * Mu_res * Edri
               - Qn_res0 * Mu_res_dVgs * Edri - Qn_res0 * Mu_res * Edri_dVgs) ;
       Ids_res_dVds = here->HSMHV2_weff_nf * ( - Qn_res0_dVds * Mu_res * Edri
               - Qn_res0 * Mu_res_dVds * Edri - Qn_res0 * Mu_res * Edri_dVds) ;
       Ids_res_dVbs = here->HSMHV2_weff_nf * ( - Qn_res0_dVbs * Mu_res * Edri
               - Qn_res0 * Mu_res_dVbs * Edri - Qn_res0 * Mu_res * Edri_dVbs) ;
       Ids_res_dT   = here->HSMHV2_weff_nf * ( - Qn_res0_dT * Mu_res * Edri
               - Qn_res0 * Mu_res_dT * Edri - Qn_res0 * Mu_res * Edri_dT ) ;

     //-----------------------------------------------------------*
     //*  back region universal mobility.  (CGS unit)
     //*-----------------//

       T0 = here->HSMHV2_ninvd ;
       T4 = 1.0 + ( phi_sL_DEP - phi_s0_DEP ) * T0 ;
       T4_dVb = ( phi_sL_DEP_dVbs - phi_s0_DEP_dVbs ) * T0 ;
       T4_dVd = ( phi_sL_DEP_dVds - phi_s0_DEP_dVds ) * T0 ;
       T4_dVg = ( phi_sL_DEP_dVgs - phi_s0_DEP_dVgs ) * T0 ;
       T4_dT =  ( phi_sL_DEP_dT - phi_s0_DEP_dT ) * T0 + ( phi_sL_DEP - phi_s0_DEP ) * ninvd_dT ;

       Qiu = - Qn_bac0 ;
       Qiu_dVgs = - Qn_bac0_dVgs ;
       Qiu_dVds = - Qn_bac0_dVds ;
       Qiu_dVbs = - Qn_bac0_dVbs ;
       Qiu_dT   = - Qn_bac0_dT ;

       T5     = Qiu ;
       T5_dVb = Qiu_dVbs ;
       T5_dVd = Qiu_dVds ;
       T5_dVg = Qiu_dVgs ;
       T5_dT  = Qiu_dT   ;

       T3     = T5 / T4 ;
       T3_dVb = ( - T4_dVb * T5 + T4 * T5_dVb ) / T4 / T4 ;
       T3_dVd = ( - T4_dVd * T5 + T4 * T5_dVd ) / T4 / T4 ;
       T3_dVg = ( - T4_dVg * T5 + T4 * T5_dVg ) / T4 / T4 ;
       T3_dT  = ( - T4_dT * T5 + T4 * T5_dT  ) / T4 / T4 ;
       Eeff = T3 ;
       Eeff_dVbs = T3_dVb ;
       Eeff_dVds = T3_dVd ;
       Eeff_dVgs = T3_dVg ;
       Eeff_dT  = T3_dT ;

       T5 = Fn_Pow( Eeff , model->HSMHV2_depmueph0 - 1.0e0 ) ;
       T8 = T5 * Eeff ;
       T8_dT = model->HSMHV2_mueph0 * T5 * Eeff_dT ;

       T9 = C_QE * C_m2cm_p2 ;
       Rns = Qiu / T9 ;
       Rns_dT = Qiu_dT / T9 ;

       T1 = 1.0e0 / ( model->HSMHV2_depmueback0 + model->HSMHV2_depmueback1 * Rns / 1.0e11 )
         + here->HSMHV2_depmphn0 * T8  ;

       T1_dT = - 1.0e0 / ( model->HSMHV2_depmueback0 + model->HSMHV2_depmueback1 * Rns / 1.0e11 )
         / ( model->HSMHV2_depmueback0 + model->HSMHV2_depmueback1 * Rns / 1.0e11 )
         * model->HSMHV2_depmueback1 * Rns_dT / 1.0e11
         + here->HSMHV2_depmphn0 * T8_dT + depmphn0_dT * T8  ;

       Muun = 1.0e0 / T1 ;
       Muun_dT = - Muun / T1 * T1_dT ;

       T1 = 1.0e0 / ( T1 * T1 ) ;
       T2 = model->HSMHV2_depmueback0 + model->HSMHV2_depmueback1 * Rns / 1.0e11 ;
       T2 = 1.0e0 / ( T2 * T2 ) ;
       T3 = here->HSMHV2_depmphn1 * T5 ;
       T5 = - 1.0e-11 * model->HSMHV2_depmueback1 / C_QE * T2 / C_m2cm_p2 ;
       Muun_dVbs = - ( T5 * Qiu_dVbs 
                    + Eeff_dVbs * T3 ) * T1 ;
       Muun_dVds = - ( T5 * Qiu_dVds 
                    + Eeff_dVds * T3 ) * T1 ;
       Muun_dVgs = - ( T5 * Qiu_dVgs 
                    + Eeff_dVgs * T3 ) * T1 ;

      /*  Change to MKS unit */
       Muun /=C_m2cm_p2 ;
       Muun_dT /=C_m2cm_p2 ;
       Muun_dVbs /=C_m2cm_p2 ;
       Muun_dVds /=C_m2cm_p2 ;
       Muun_dVgs /=C_m2cm_p2 ;

       Edri = Vdseff0 / Lch ;
       Edri_dVgs = Vdseff0_dVgs / Lch ;
       Edri_dVds = Vdseff0_dVds / Lch ;
       Edri_dVbs = Vdseff0_dVbs / Lch ;
       Edri_dT   = Vdseff0_dT / Lch ;

       T1 = Muun * Edri / here->HSMHV2_depvmax ;
       T1_dVgs = (Muun_dVgs * Edri + Muun * Edri_dVgs) / here->HSMHV2_depvmax ;
       T1_dVds = (Muun_dVds * Edri + Muun * Edri_dVds) / here->HSMHV2_depvmax ;
       T1_dVbs = (Muun_dVbs * Edri + Muun * Edri_dVbs) / here->HSMHV2_depvmax ;
       T1_dT   = (Muun_dT * Edri + Muun * Edri_dT) / here->HSMHV2_depvmax
                 - Muun * Edri / here->HSMHV2_depvmax / here->HSMHV2_depvmax * depVmax_dT ;

       T1 = T1 + small ;
       T2 = Fn_Pow(T1,model->HSMHV2_depbb) ;
       T2_dVgs = model->HSMHV2_depbb * T1_dVgs / T1 * T2 ;
       T2_dVds = model->HSMHV2_depbb * T1_dVds / T1 * T2 ;
       T2_dVbs = model->HSMHV2_depbb * T1_dVbs / T1 * T2 ;
       T2_dT   = model->HSMHV2_depbb * T1_dT / T1 * T2 ;

       T3 = 1.0 + T2 ;
       T4 = Fn_Pow(T3,1.0 / model->HSMHV2_depbb) ;
       T4_dVgs = 1.0 / model->HSMHV2_depbb * T2_dVgs / T3 * T4 ;
       T4_dVds = 1.0 / model->HSMHV2_depbb * T2_dVds / T3 * T4 ;
       T4_dVbs = 1.0 / model->HSMHV2_depbb * T2_dVbs / T3 * T4 ;
       T4_dT   = 1.0 / model->HSMHV2_depbb * T2_dT / T3 * T4 ;


       Mu_bac = Muun / T4 ;
       Mu_bac_dVgs = Muun_dVgs / T4 - Muun / T4 / T4 * T4_dVgs ;
       Mu_bac_dVds = Muun_dVds / T4 - Muun / T4 / T4 * T4_dVds ;
       Mu_bac_dVbs = Muun_dVbs / T4 - Muun / T4 / T4 * T4_dVbs ;
       Mu_bac_dT = Muun_dT / T4 - Muun / T4 / T4 * T4_dT ;

       Ids_bac = here->HSMHV2_weff_nf * ( - Qn_bac0) * Mu_bac * Edri ;
       Ids_bac_dVgs = here->HSMHV2_weff_nf * ( - Qn_bac0_dVgs * Mu_bac * Edri
               - Qn_bac0 * Mu_bac_dVgs * Edri - Qn_bac0 * Mu_bac * Edri_dVgs) ;
       Ids_bac_dVds = here->HSMHV2_weff_nf * ( - Qn_bac0_dVds * Mu_bac * Edri
               - Qn_bac0 * Mu_bac_dVds * Edri - Qn_bac0 * Mu_bac * Edri_dVds) ;
       Ids_bac_dVbs = here->HSMHV2_weff_nf * ( - Qn_bac0_dVbs * Mu_bac * Edri
               - Qn_bac0 * Mu_bac_dVbs * Edri - Qn_bac0 * Mu_bac * Edri_dVbs) ;
       Ids_bac_dT   = here->HSMHV2_weff_nf * ( - Qn_bac0_dT * Mu_bac * Edri
               - Qn_bac0 * Mu_bac_dT * Edri - Qn_bac0 * Mu_bac * Edri_dT ) ;

     /*-----------------------------------------------------------*
      * Ids: channel current.
      *-----------------*/
       betaWL = here->HSMHV2_weff_nf * beta_inv / Lch ;
       T1 = - betaWL / Lch ;
       betaWL_dT = here->HSMHV2_weff_nf * beta_inv_dT / Lch  ;

       Ids0 = betaWL * Idd * Mu + Ids_res + Ids_bac ;

       Ids0_dVgs = betaWL * ( Idd_dVgs * Mu + Idd * Mu_dVgs )
                   + Ids_res_dVgs + Ids_bac_dVgs ;
       Ids0_dVds = betaWL * ( Idd_dVds * Mu + Idd * Mu_dVds )
                   + Ids_res_dVds + Ids_bac_dVds ;
       Ids0_dVbs = betaWL * ( Idd_dVbs * Mu + Idd * Mu_dVbs )
                   + Ids_res_dVbs + Ids_bac_dVbs ;
       Ids0_dT   = betaWL_dT * Idd * Mu + betaWL * ( Idd_dT * Mu + Idd * Mu_dT )  
                   + Ids_res_dT + Ids_bac_dT ;

       // Vdseff //

       Vds = Vdsorg;

      /*-----------------------------------------------------------*
       * Adding parasitic components to the channel current.
       *-----------------*/
       if( model->HSMHV2_ptl != 0 ){
         T1 =  0.5 * ( Vds - Pds ) ;
         Fn_SymAdd( T6 , T1 , 0.01 , T2 ) ;
         T2 *=0.5 ;
         T6_dVb = T2 * ( - Pds_dVbs ) ;
         T6_dVd = T2 * ( 1.0 - Pds_dVds ) ;
         T6_dVg = T2 * ( - Pds_dVgs ) ;
         T6_dT = T2 * ( - Pds_dT ) ;

         T1     = 1.1 - ( phi_s0_DEP + T6 );
         T1_dVb = - ( phi_s0_DEP_dVbs + T6_dVb );
         T1_dVd = - ( phi_s0_DEP_dVds + T6_dVd );
         T1_dVg = - ( phi_s0_DEP_dVgs + T6_dVg );
         T1_dT = - ( phi_s0_DEP_dT + T6_dT );
 
         Fn_SZ( T2 , T1 , 0.05 , T0 ) ;
         T2 +=small ;
         T2_dVb = T1_dVb * T0 ;
         T2_dVd = T1_dVd * T0 ;
         T2_dVg = T1_dVg * T0 ;
         T2_dT = T1_dT * T0 ;

         T0 = beta * here->HSMHV2_ptl0 ;
         T0_dT = beta_dT * here->HSMHV2_ptl0 ;
         T3 = Cox * T0 ;
         T3_dVb = Cox_dVb * T0 ;
         T3_dVd = Cox_dVd * T0 ;
         T3_dVg = Cox_dVg * T0 ;
         T3_dT = Cox_dT * T0 + Cox * T0_dT ;
         T0 = pow( T2 , model->HSMHV2_ptp ) ;
         T9     = T3 * T0 ;
         T9_dVb = T3 * model->HSMHV2_ptp * T0 / T2 * T2_dVb + T3_dVb * T0 ;
         T9_dVd = T3 * model->HSMHV2_ptp * T0 / T2 * T2_dVd + T3_dVd * T0 ;
         T9_dVg = T3 * model->HSMHV2_ptp * T0 / T2 * T2_dVg + T3_dVg * T0 ;
         T9_dT = T3 * model->HSMHV2_ptp * T0 / T2 * T2_dT + T3_dT * T0 ;


         T4 = 1.0 + Vdsz * model->HSMHV2_pt2 ;
         T4_dVb = Vdsz_dVbs * model->HSMHV2_pt2 ;
         T4_dVd = Vdsz_dVds * model->HSMHV2_pt2 ;
         T4_dVg = 0.0 ;
         T4_dT = Vdsz_dT * model->HSMHV2_pt2 ;

         T0 = here->HSMHV2_pt40 ;
         T5 = phi_s0_DEP + T6 - Vbsz ;
         T5_dVb = phi_s0_DEP_dVbs + T6_dVb - Vbsz_dVbs ;
         T5_dVd = phi_s0_DEP_dVds + T6_dVd - Vbsz_dVds ;
         T5_dVg = phi_s0_DEP_dVgs + T6_dVg ;
         T5_dT = phi_s0_DEP_dT + T6_dT - Vbsz_dT ;
         T4 +=Vdsz * T0 * T5 ;
         T4_dVb +=Vdsz * T0 * T5_dVb + Vdsz_dVbs * T0 * T5 ;
         T4_dVd +=Vdsz * T0 * T5_dVd + Vdsz_dVds * T0 * T5 ;
         T4_dVg +=Vdsz * T0 * T5_dVg ;
         T4_dT +=Vdsz * T0 * T5_dT + Vdsz_dT * T0 * T5 ;
         T6     = T9 * T4 ;
         T9_dVb = T9_dVb * T4 + T9 * T4_dVb ;
         T9_dVd = T9_dVd * T4 + T9 * T4_dVd ;
         T9_dVg = T9_dVg * T4 + T9 * T4_dVg ;
         T9_dT = T9_dT * T4 + T9 * T4_dT ;
         T9     = T6 ;

       } else {
         T9 = 0.0 ;
         T9_dVb = 0.0 ;
         T9_dVd = 0.0 ;
         T9_dVg = 0.0 ;
         T9_dT = 0.0 ;
       }

       if( model->HSMHV2_gdl != 0 ){
         T1 = beta * here->HSMHV2_gdl0 ;
         T1_dT = beta_dT * here->HSMHV2_gdl0 ;
         T2 = Cox * T1 ;
         T2_dVb = Cox_dVb * T1 ;
         T2_dVd = Cox_dVd * T1 ;
         T2_dVg = Cox_dVg * T1 ;
         T2_dT = Cox_dT * T1 + Cox * T1_dT ;
         T8     = T2 * Vdsz ;
         T8_dVb = T2_dVb * Vdsz + T2 * Vdsz_dVbs ;
         T8_dVd = T2_dVd * Vdsz + T2 * Vdsz_dVds ;
         T8_dVg = T2_dVg * Vdsz ;
         T8_dT = T2_dT * Vdsz + T2 * Vdsz_dT ;
       } else {
         T8 = 0.0 ;
         T8_dVb = 0.0 ;
         T8_dVd = 0.0 ;
         T8_dVg = 0.0 ;
         T8_dT = 0.0 ;
       }

       if ( ( T9 + T8 ) > 0.0 ) {
         Idd1 = Pds * ( T9 + T8 ) ;
         Idd1_dVbs = Pds_dVbs * ( T9 + T8 ) + Pds * ( T9_dVb + T8_dVb ) ;
         Idd1_dVds = Pds_dVds * ( T9 + T8 ) + Pds * ( T9_dVd + T8_dVd ) ;
         Idd1_dVgs = Pds_dVgs * ( T9 + T8 ) + Pds * ( T9_dVg + T8_dVg ) ;
         Idd1_dT = Pds_dT * ( T9 + T8 ) + Pds * ( T9_dT + T8_dT ) ;

         Ids0 +=betaWL * Idd1 * Mu ;
         T1 = betaWL * Idd1 ;
         T2 = Idd1 * Mu ;
         T3 = Mu * betaWL ;
         Ids0_dVbs +=T3 * Idd1_dVbs + T1 * Mu_dVbs + T2 * betaWL_dVbs ;
         Ids0_dVds +=T3 * Idd1_dVds + T1 * Mu_dVds + T2 * betaWL_dVds ;
         Ids0_dVgs +=T3 * Idd1_dVgs + T1 * Mu_dVgs + T2 * betaWL_dVgs ;
         Ids0_dT +=T3 * Idd1_dT + T1 * Mu_dT + T2 * betaWL_dT ;
       }


      /* note: rpock procedure was removed. */
       if( flg_rsrd == 2 || flg_rsrd == 3 ){
         if( model->HSMHV2_rd20 > 0.0 ){
           T4 = here->HSMHV2_rd23 ;
           T1 = pParam->HSMHV2_rd24 * ( Vgse-model->HSMHV2_rd25 ) ;
           T1_dVg = pParam->HSMHV2_rd24 ;

           Fn_SL( T2 , T1 , T4 , delta_rd , T0 ) ;
           T2_dVg = T1_dVg * T0 ;
           T3 = T4 * ( model->HSMHV2_rd20 + 1.0 ) ;
           Fn_SU( T7 , T2 , T3 , delta_rd , T0 ) ;
           T7_dVg = T2_dVg * T0 ;

         } else {
           T7 = here->HSMHV2_rd23;
           T7_dVg = 0.0e0 ;
         }

       /* after testing we can remove Vdse_eff_dVbs, Vdse_eff_dVds, Vdse_eff_dVgs
         and Vdse_eff_dVbse, Vdse_eff_dVgse                                      */
         if (Vdse >= 0.0) {
           Vdse_eff = Vdse ;
          /* Vdse_eff_dVbs  = 0.0 ; */
          /* Vdse_eff_dVds  = 0.0 ; */
          /* Vdse_eff_dVgs  = 0.0 ; */
          /* Vdse_eff_dVbse = 0.0 ; */
           Vdse_eff_dVdse = 1.0 ;
          /* Vdse_eff_dVgse = 0.0 ; */
         } else {
           Vdse_eff = 0.0 ;
          /* Vdse_eff_dVbs  = 0.0 ; */
           /* Vdse_eff_dVds  = 0.0 ; */
          /* Vdse_eff_dVgs  = 0.0 ; */
          /* Vdse_eff_dVbse = 0.0 ; */
           Vdse_eff_dVdse = 0.0 ;
          /* Vdse_eff_dVgse = 0.0 ; */
         }
        /* smoothing of Ra for Vdse_eff close to zero */
        /* ... smoothing parameter is Ra_N            */
         if (Vdse_eff < Ra_N * small2) {
           Ra_alpha = pow( Ra_N + 1.0 , model->HSMHV2_rd21 - 1.0 )
                     * (Ra_N + 1.0 - 0.5 * model->HSMHV2_rd21 * Ra_N)
                     * pow( small2,model->HSMHV2_rd21 );
           Ra_beta = 0.5 * model->HSMHV2_rd21
                    * pow( Ra_N + 1.0 , model->HSMHV2_rd21 - 1.0 ) / Ra_N
                    * pow( small2, model->HSMHV2_rd21 - 2.0 );
           T1 = Ra_alpha + Ra_beta * Vdse_eff * Vdse_eff;
           T1_dVdse_eff = 2.0 * Ra_beta * Vdse_eff;
         } else {
           T1           = pow( Vdse_eff + small2 , model->HSMHV2_rd21 ) ;
           T1_dVdse_eff = model->HSMHV2_rd21 * pow( Vdse_eff + small2 , model->HSMHV2_rd21 - 1.0 ) ;
         }

         T9           = pow( Vdse_eff + small2 , model->HSMHV2_rd22d ) ;
         T9_dVdse_eff = model->HSMHV2_rd22d * pow( Vdse_eff + small2 , model->HSMHV2_rd22d - 1.0 ) ;

         Ra           = ( T7 * T1 + Vbse * pParam->HSMHV2_rd22 * T9 ) / here->HSMHV2_weff_nf ;
         Ra_dVdse_eff = ( T7 * T1_dVdse_eff + Vbse * pParam->HSMHV2_rd22 * T9_dVdse_eff ) / here->HSMHV2_weff_nf ;
         Ra_dVbs      =  Ra_dVdse_eff * Vdse_eff_dVbs ;
         Ra_dVds      =  Ra_dVdse_eff * Vdse_eff_dVds ;
         Ra_dVgs      =  Ra_dVdse_eff * Vdse_eff_dVgs + T7_dVg * T1 / here->HSMHV2_weff_nf ;
         Ra_dVbse     =  Ra_dVdse_eff * Vdse_eff_dVbse+pParam->HSMHV2_rd22 * T9 / here->HSMHV2_weff_nf ;
         Ra_dVdse     =  Ra_dVdse_eff * Vdse_eff_dVdse ;
         Ra_dVgse     =  Ra_dVdse_eff * Vdse_eff_dVgse ;
         T0 = Ra * Ids0 ;
         T0_dVb = Ra_dVbs * Ids0 + Ra * Ids0_dVbs ;
         T0_dVd = Ra_dVds * Ids0 + Ra * Ids0_dVds ;
         T0_dVg = Ra_dVgs * Ids0 + Ra * Ids0_dVgs ;
         T0_dT  =                  Ra * Ids0_dT ;

         T1 = Vds + small2 ;
         T2 = 1.0 / T1 ;
         T3 = 1.0 + T0 * T2 ;
         T3_dVb = T0_dVb * T2 ;
         T3_dVd = ( T0_dVd * T1 - T0 ) * T2 * T2 ;
         T3_dVg = T0_dVg * T2 ;
         T3_dT = T0_dT * T2 ;

         T4 = 1.0 / T3 ;
         Ids = Ids0 * T4 ;
         T5 = T4 * T4 ;
         Ids_dVbs = ( Ids0_dVbs * T3 - Ids0 * T3_dVb ) * T5 ;
         Ids_dVds = ( Ids0_dVds * T3 - Ids0 * T3_dVd ) * T5 ;
         Ids_dVgs = ( Ids0_dVgs * T3 - Ids0 * T3_dVg ) * T5 ;
         Ids_dT = ( Ids0_dT * T3 - Ids0 * T3_dT ) * T5 ;
         Ids_dRa = - Ids * Ids / ( Vds + small ) ;

       } else {
         Ids = Ids0 ;
         Ids_dVbs = Ids0_dVbs ;
         Ids_dVds = Ids0_dVds ;
         Ids_dVgs = Ids0_dVgs ;
         Ids_dT = Ids0_dT ;
         Ra = 0.0 ;
         Ra_dVbs = Ra_dVds = Ra_dVgs = 0.0 ;
         Ra_dVbse = Ra_dVdse = Ra_dVgse = 0.0 ;
         Ids_dRa = 0.0 ;
       }

      /*---------------------------------------------------*
       * Qbu : - Qb in unit area.
       *-----------------*/
       Qbu = - 0.5 * (Q_sub0 + Q_subL + Q_sub0_dep + Q_subL_dep ) ;
       Qbu_dVgs = - 0.5 * ( Q_sub0_dVgs + Q_subL_dVgs + Q_sub0_dep_dVgs + Q_subL_dep_dVgs ) ;
       Qbu_dVds = - 0.5 * ( Q_sub0_dVds + Q_subL_dVds + Q_sub0_dep_dVds + Q_subL_dep_dVds ) ;
       Qbu_dVbs = - 0.5 * ( Q_sub0_dVbs + Q_subL_dVbs + Q_sub0_dep_dVbs + Q_subL_dep_dVbs ) ;
       Qbu_dT   = - 0.5 * ( Q_sub0_dT + Q_subL_dT + Q_sub0_dep_dT + Q_subL_dep_dT ) ;

       Qiu = - 0.5 * (Q_n0 + Q_nL + Q_s0_dep + Q_sL_dep + Q_b0_dep + Q_bL_dep) ;
       Qiu_dVgs = - 0.5 * ( Q_n0_dVgs + Q_nL_dVgs + Q_s0_dep_dVgs + Q_sL_dep_dVgs + Q_b0_dep_dVgs + Q_bL_dep_dVgs ) ;
       Qiu_dVds = - 0.5 * ( Q_n0_dVds + Q_nL_dVds + Q_s0_dep_dVds + Q_sL_dep_dVds + Q_b0_dep_dVds + Q_bL_dep_dVds ) ;
       Qiu_dVbs = - 0.5 * ( Q_n0_dVbs + Q_nL_dVbs + Q_s0_dep_dVbs + Q_sL_dep_dVbs + Q_b0_dep_dVbs + Q_bL_dep_dVbs ) ;
       Qiu_dT   = - 0.5 * ( Q_n0_dT + Q_nL_dT + Q_s0_dep_dT + Q_sL_dep_dT + Q_b0_dep_dT + Q_bL_dep_dT ) ;

       Qdrat = 0.5;
       Qdrat_dVgs = 0.0 ;
       Qdrat_dVds = 0.0 ;
       Qdrat_dVbs = 0.0 ;
       Qdrat_dT = 0.0 ;

      /*-------------------------------------------------*
       * set flg_noqi 
       *-----------------*/
       Qiu_noi = - 0.5 * (Q_n0 + Q_nL ) ;
       Qn0 = - Q_n0 ; 
       Qn0_dVgs = - Q_n0_dVgs ;
       Qn0_dVds = - Q_n0_dVds ;
       Qn0_dVbs = - Q_n0_dVbs ;
       Qn0_dT = - Q_n0_dT ;

       Ey = Ey_suf ;
       Ey_dVgs = Ey_suf_dVgs ;
       Ey_dVds = Ey_suf_dVds ;
       Ey_dVbs = Ey_suf_dVbs ;
       Ey_dT = Ey_suf_dT ;

       if( Qn0 < small ){ 
         flg_noqi = 1 ;
       }


} /* End of hsmhveval_dep */


