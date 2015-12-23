/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 2  SUBVERSION : 8  REVISION : 0 ) 
 Model Parameter 'VERSION' : 2.80
 FILE : hsm2eval_dep.h

 DATE : 2014.5.12

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

/**********************************************************************

The following source code, and all copyrights, trade secrets or other
intellectual property rights in and to the source code in its entirety,
is owned by the Hiroshima University and the STARC organization.

All users need to follow the "HISIM_HV Distribution Statement and
Copyright Notice" attached to HiSIM model.

-----HISIM Distribution Statement and Copyright Notice--------------

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

{ // Begin : hsm2eval_dep

/* define local variavles */
  int    depmode ;
  double afact, afact2, afact3, bfact, cfact ;
  double W_bsub0, W_bsubL, W_s0, W_sL, W_sub0, W_subL, W_b0, W_bL, vthn ;
  double phi_s0_DEP = 0.0, phi_sL_DEP = 0.0 , Vbi_DEP ;
  double phi_s0_DEP_dVgs, phi_s0_DEP_dVbs, phi_s0_DEP_dVds ;
  double phi_sL_DEP_dVgs, phi_sL_DEP_dVbs, phi_sL_DEP_dVds ;
  double phi_j0_DEP, phi_jL_DEP, Psbmax, phi_b0_DEP_lim, phi_bL_DEP_lim ;


  double phi_jL_DEP_dVgs, phi_jL_DEP_dVds, phi_jL_DEP_dVbs ;

  double Vgp0, Vgp1, Vgp0old, phi_j0_DEP_old, phi_jL_DEP_old, phi_b0_DEP_old, phi_bL_DEP_old, phi_s0_DEP_old, phi_sL_DEP_old ;
  double phi_j0_DEP_acc, phi_jL_DEP_acc ;


  double Q_s0, Q_sL = 0.0 ;
  double Q_s0_dVgs, Q_sL_dVgs = 0.0, Q_s0_dVds, Q_sL_dVds = 0.0, Q_s0_dVbs, Q_sL_dVbs = 0.0 ;
  double Q_sub0, Q_subL, Q_sub0_dVgs, Q_subL_dVgs, Q_sub0_dVds, Q_subL_dVds, Q_sub0_dVbs, Q_subL_dVbs ;
  double Qn_res0, Qn_res0_dVgs, Qn_res0_dVds, Qn_res0_dVbs ;


  double y1, y2, dety ;
  double y11, y12 ;
  double y21, y22 ;
  
  double y1_dVgs, y1_dVds, y1_dVbs ;
  double y2_dVgs, y2_dVds, y2_dVbs ;

  double rev11 = 0.0, rev12 = 0.0 ;
  double rev21 = 0.0, rev22 = 0.0 ;

  double phi_b0_DEP_ini ;
  double y0, dydPsm ;
  
  double W_b0_dVgs, W_b0_dVds, W_b0_dVbs ;

  double W_res0 ;
  double W_s0_dVgs, W_s0_dVds, W_s0_dVbs ;

  double phi_b0_DEP,  Q_b0_dep, Q_sub0_dep ;
  double phi_b0_DEP_dVgs, phi_b0_DEP_dVds, phi_b0_DEP_dVbs ;
  double phi_j0_DEP_dVgs, phi_j0_DEP_dVds, phi_j0_DEP_dVbs ;
  double Q_b0_dep_dVgs, Q_b0_dep_dVds, Q_b0_dep_dVbs ;
  double Q_sub0_dep_dVgs, Q_sub0_dep_dVds, Q_sub0_dep_dVbs ;

  double phi_bL_DEP, Q_bL_dep, Q_subL_dep ;
  double phi_bL_DEP_dVgs, phi_bL_DEP_dVds, phi_bL_DEP_dVbs ;
  double Q_bL_dep_dVgs, Q_bL_dep_dVds, Q_bL_dep_dVbs ;
  double Q_subL_dep_dVgs, Q_subL_dep_dVds, Q_subL_dep_dVbs ;

  double q_Ndepm_esi, Idd_drift,Idd_diffu ;
  double Qn_bac0 ;
  double Qn_bac0_dVgs, Qn_bac0_dVds, Qn_bac0_dVbs ;

  double Mu_res, Mu_bac ;
  double Mu_res_dVgs, Mu_res_dVds, Mu_res_dVbs ;
  double Mu_bac_dVgs, Mu_bac_dVds, Mu_bac_dVbs ;

  double Q_n0_cur, Q_nL_cur ;
  double Q_n0_cur_dVgs, Q_n0_cur_dVds, Q_n0_cur_dVbs ;
  double Q_nL_cur_dVgs, Q_nL_cur_dVds, Q_nL_cur_dVbs ;

  double Q_s0_dep, Q_sL_dep ;
  double Q_s0_dep_dVgs, Q_s0_dep_dVds, Q_s0_dep_dVbs ;
  double Q_sL_dep_dVgs, Q_sL_dep_dVds, Q_sL_dep_dVbs ;

  double sm_delta ;
  double phib_ref, phib_ref_dPs, phib_ref_dPd ;
  double Q_s0_dPs, Q_sL_dPs, Q_s0_dPb, Q_sL_dPb ;
  double Q_b0_dep_dPb, Q_bL_dep_dPb, Q_b0_dep_dPd, Q_bL_dep_dPd, Q_sub0_dep_dPd, Q_subL_dep_dPd ;
  double phi_j0_DEP_dPb, phi_jL_DEP_dPb ;
  double NdepmpNsub_inv1, NdepmpNsub ;



  double Q_n0, Q_n0_dVgs, Q_n0_dVds, Q_n0_dVbs ;
  double Q_nL, Q_nL_dVgs, Q_nL_dVds, Q_nL_dVbs ;

  double phi_s0_DEP_ini, phi_sL_DEP_ini ;


  double C_QE2, C_ESI2, Tn2 ; 
  double Ndepm2, q_Ndepm ; 
  double C_2ESIpq_Ndepm, C_2ESIpq_Ndepm_inv , C_2ESI_q_Ndepm ; 
  double C_2ESIpq_Nsub , C_2ESIpq_Nsub_inv ; 
  double ps_conv3 , ps_conv23 ; 
  double Ids_res, Ids_bac, Edri ;
  double Ids_res_dVgs, Ids_res_dVds, Ids_res_dVbs ;
  double Ids_bac_dVgs, Ids_bac_dVds, Ids_bac_dVbs ;
  double Edri_dVgs, Edri_dVds, Edri_dVbs ;

  double T0_dVg, T0_dVb,T0_dVd ;
  double T1_dVgs, T1_dVds, T1_dVbs ;
  double T2_dVgs, T2_dVds, T2_dVbs ;
  double T3_dVgs, T3_dVds, T3_dVbs ;
  double T4_dVgs, T4_dVds, T4_dVbs ;
  double T5_dVgs, T5_dVds, T5_dVbs ;


  double Vgpp ;
  double Vgpp_dVgs, Vgpp_dVds, Vgpp_dVbs ;
  double Vdseff0, Vdseff0_dVgs, Vdseff0_dVds, Vdseff0_dVbs ;
  double phib_ref_dVgs, phib_ref_dVds, phib_ref_dVbs ;

  double Qn_delta ;
  double Qn_drift, Qn_drift_dVgs, Qn_drift_dVds, Qn_drift_dVbs ;

  double Ey_suf, Ey_suf_dVgs, Ey_suf_dVds, Ey_suf_dVbs ;

  double DEPQFN1 = 2.0 ;
  double DEPQFN3 = 0.3 ;
  double DEPQFN_dlt = 2.0 ;
  double Ps_delta = 0.06 ;
  double Ps_delta0 = 0.08 ;

/*--------------------------------------*
 * CeilingPow with derivatives for delta 
 *-----------------*/
#define Fn_SL_CP3( y , x , xmin , delta , pw , dx , dxmin, ddelta) { \
 if(x < xmin + delta && delta >= 0.0) { \
   double TMF0, TMF1; \
   TMF1 = xmin + delta - x ; \
   Fn_CP2( TMF0 , TMF1 , delta , pw , dx , ddelta )  \
   y = xmin + delta - TMF0 ; \
   dx = dx ; \
   dxmin = 1.0-dx ; \
   ddelta = 1.0-dx-ddelta; \
 } else { \
   y = x ; \
   dx = 1.0 ; \
   dxmin = 0.0 ; \
   ddelta = 0.0 ; \
 } \
} 


    // Constants
    Vbi_DEP = here->HSM2_Vbipn ;
    q_Ndepm = C_QE * here->HSM2_ndepm ;
    Ndepm2  = here->HSM2_ndepm * here->HSM2_ndepm ;
    q_Ndepm_esi = C_QE * here->HSM2_ndepm * C_ESI ;
    q_Nsub = C_QE * here->HSM2_nsub ;
    C_QE2  = C_QE * C_QE ;
    C_ESI2 = C_ESI * C_ESI ;
    Tn2    = model->HSM2_tndep * model->HSM2_tndep ;
    C_2ESIpq_Ndepm = 2.0 * C_ESI/q_Ndepm ;
    C_2ESIpq_Ndepm_inv = q_Ndepm / (2.0 * C_ESI) ;
    C_2ESI_q_Ndepm = 2.0 * C_ESI * q_Ndepm ;
    C_2ESIpq_Nsub  = 2.0 * C_ESI / q_Nsub  ;
    C_2ESIpq_Nsub_inv  = q_Nsub / (2.0 * C_ESI) ;
    NdepmpNsub  = here->HSM2_ndepm / here->HSM2_nsub ;
    NdepmpNsub_inv1  = 1.0 / (1.0 + NdepmpNsub ) ;
    ps_conv3  = ps_conv * 1000.0 ;
    ps_conv23 = ps_conv2 * 1000.0 ;
    here->HSM2_qnsub_esi = q_Ndepm_esi ;

 
     //---------------------------------------------------*
     // depletion MOS mode  
     //------------------//

     Vbsc = Vbs ;
     Vbsc_dVbse = 1.0 ;

     /*---------------------------------------------------*
      * initial potential phi_s0_DEP,phi_b0_DEP,phi_j0_DEP calculated.
      *------------------*/

       Vgp = Vgp + epsm10 * 1.0e7 ;


      afact = Cox * Cox / here->HSM2_cnst0 / here->HSM2_cnst0 ;
      afact2 = afact / here->HSM2_nin / here->HSM2_nin * Ndepm2 ;
      W_bsub0 = sqrt(2.0e0 * C_ESI / C_QE * here->HSM2_nsub / (here->HSM2_nsub
              + here->HSM2_ndepm) / here->HSM2_ndepm * ( - Vbsc + Vbi_DEP)) ;

      if( W_bsub0 > model->HSM2_tndep ) {

        Vgp0 = 0.0;

        W_b0 = model->HSM2_tndep ;
        phi_b0_DEP = 0.0 ;
        phi_j0_DEP = phi_b0_DEP - C_2ESIpq_Ndepm_inv * W_b0 * W_b0 ;
        phi_b0_DEP_lim = 0.0 ;

        Vgp0old = Vgp0 ;
        phi_j0_DEP_old = phi_j0_DEP ;

        for ( lp_s0 = 1 ; lp_s0 <= lp_se_max + 1 ; lp_s0 ++ ) {

          W_b0 = sqrt(C_2ESIpq_Ndepm * (phi_b0_DEP - phi_j0_DEP) ) ;
          Fn_SU_CP( W_b0 , W_b0 , model->HSM2_tndep , 1e-8, 2 , T0 )
          W_sub0 = sqrt(C_2ESIpq_Nsub * (phi_j0_DEP - Vbsc + Vbi_DEP) ) ;

          Q_b0_dep = W_b0 * q_Ndepm ;
          Q_b0_dep_dPd = - C_ESI / W_b0 * T0 ;
          Q_sub0_dep = - W_sub0 * q_Nsub ;
          Q_sub0_dep_dPd = - C_ESI / W_sub0 ;

          y1 = Cox * (Vgp0 - phi_b0_DEP) + Q_b0_dep + Q_sub0_dep ;
          y11 = Cox ;
          y12 = Q_b0_dep_dPd + Q_sub0_dep_dPd ;

          y2 = phi_j0_DEP - NdepmpNsub_inv1 * (NdepmpNsub * phi_b0_DEP + Vbsc - Vbi_DEP) ;
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

        W_sub0 = model->HSM2_tndep * NdepmpNsub ;
        phi_j0_DEP = C_2ESIpq_Nsub_inv * W_sub0 * W_sub0 + Vbsc - Vbi_DEP ;
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
        phi_j0_DEP = C_2ESIpq_Nsub_inv * W_sub0 * W_sub0 + Vbsc - Vbi_DEP ;
        phi_b0_DEP = C_2ESIpq_Ndepm_inv * W_b0 * W_b0 + phi_j0_DEP ;
        phi_j0_DEP_acc = phi_j0_DEP ;
        if( Vgp > Vgp0 ) { 
          depmode = 1 ;
        } else {
          depmode = 2 ;
        }

      }


      T1 = C_2ESI_q_Ndepm * ( Psbmax - ( - here->HSM2_Pb2n + Vbsc)) ;
      if ( T1 > 0.0 ) {
        vthn = - here->HSM2_Pb2n + Vbsc - sqrt(T1) / Cox ;
      } else {
        vthn = - here->HSM2_Pb2n + Vbsc ;
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

           if( W_s0 + W_b0 > model->HSM2_tndep ) {
             for ( lp_s0 = 1 ; lp_s0 <= lp_se_max + 1 ; lp_s0 ++ ) {

               y0 = W_s0 + W_b0 - model->HSM2_tndep ;

               dydPsm = C_ESI / q_Ndepm / W_s0 
                  + C_ESI / q_Ndepm * ( 1.0 - (here->HSM2_ndepm
                  / here->HSM2_nsub) / ( 1.0 + (NdepmpNsub))) / W_b0 ;

               if( fabs(y0 / dydPsm) > 0.5 ) {
                 phi_b0_DEP = phi_b0_DEP - 0.5 * Fn_Sgn(y0 / dydPsm) ;
               } else {
                 phi_b0_DEP = phi_b0_DEP - y0 / dydPsm ;
               }

               if( (phi_b0_DEP - Vbsc + Vbi_DEP) < epsm10 )
                  phi_b0_DEP=Vbsc - Vbi_DEP + epsm10 ;

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
                 + Vbsc - Vbi_DEP) / (1.0 + NdepmpNsub) ;
               W_b0 = sqrt(C_2ESIpq_Ndepm * (phi_b0_DEP - phi_j0_DEP) ) ;

              if( fabs(phi_b0_DEP - phi_b0_DEP_old) <= 1.0e-8 ) lp_s0=lp_se_max + 1 ;
               phi_b0_DEP_old = phi_b0_DEP ;
             }
           }

         } else {
           afact3 = afact2 / exp(beta * Vbsc) ;
           phi_b0_DEP_old = phi_b0_DEP ;
           phi_s0_DEP_ini = log(afact3 * Vgp * Vgp) / ( - beta + 2.0 / Vgp) ;
           W_s0 = sqrt(C_2ESIpq_Ndepm * (phi_b0_DEP - phi_s0_DEP_ini) ) ;
           W_b0 = sqrt(C_2ESIpq_Ndepm * (phi_b0_DEP - phi_j0_DEP) ) ;
           if( W_s0 + W_b0 >  model->HSM2_tndep ) {
             for ( lp_s0 = 1 ; lp_s0 <= lp_s0_max + 1 ; lp_s0 ++ ) {

               y0 = W_s0 + W_b0 - model->HSM2_tndep ;
               dydPsm = C_ESI / q_Ndepm / W_s0 
                 + C_ESI / q_Ndepm * ( 1.0 - (here->HSM2_ndepm / 
                 here->HSM2_nsub) / ( 1.0 + (NdepmpNsub))) / W_b0 ;

               if( fabs(y0 / dydPsm) > 0.5 ) {
                 phi_b0_DEP = phi_b0_DEP - 0.5 * Fn_Sgn(y0 / dydPsm) ;
               } else {
                 phi_b0_DEP = phi_b0_DEP - y0 / dydPsm ;
               } 
               if( (phi_b0_DEP - Vbsc + Vbi_DEP) < epsm10 ) 
                    phi_b0_DEP=Vbsc - Vbi_DEP + epsm10 ;

               W_s0 = sqrt(C_2ESIpq_Ndepm * (phi_b0_DEP - phi_s0_DEP_ini) ) ;
               phi_j0_DEP = ( NdepmpNsub * phi_b0_DEP
                 + Vbsc - Vbi_DEP) / (1.0 + NdepmpNsub) ;
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

         phi_j0_DEP  = NdepmpNsub_inv1 * (NdepmpNsub * phi_b0_DEP + Vbsc - Vbi_DEP) ;
         phi_j0_DEP_dPb = NdepmpNsub_inv1 * NdepmpNsub ;

         T1 = phi_b0_DEP - phi_j0_DEP ;
         Fn_SL_CP( T2 , T1 , 0.0 , 0.1, 2 , T7 )
         W_b0 = sqrt(C_2ESIpq_Ndepm * (T2) ) ;
         Fn_SU_CP( W_b0 , W_b0 , model->HSM2_tndep, 1e-8, 2 , T8 )
         W_sub0 = sqrt(C_2ESIpq_Nsub * (phi_j0_DEP - Vbsc + Vbi_DEP) ) ;
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
                  + Ndepm2 * C_QE2 * Tn2 * Tn2 ) / T10 ;
         phib_ref_dPs = ( - 8.0 * phi_j0_DEP * C_ESI2 + 8.0 * C_ESI2 * phi_s0_DEP 
                      + 4.0 * q_Ndepm_esi * Tn2 ) / T10 ;
         phib_ref_dPd = (   8.0 * phi_j0_DEP * C_ESI2 - 8.0 * C_ESI2 * phi_s0_DEP
                      + 4.0 * q_Ndepm_esi * Tn2 ) / T10 ;

         T1 = beta * (phi_s0_DEP - phi_b0_DEP) ;
         T2 = exp(T1) ; 
         if( phi_s0_DEP >= phi_b0_DEP ) { 
           Q_s0 = - here->HSM2_cnst0 * sqrt(T2 - 1.0 - T1 + 1e-15) ;
           Q_s0_dPs = 0.5 * here->HSM2_cnst0 * here->HSM2_cnst0 / Q_s0 * (beta * T2 - beta ) ;
           Q_s0_dPb = - Q_s0_dPs ;
         } else {
           T3 = exp( - beta * (phi_s0_DEP - Vbsc)) ;
           T4 = exp( - beta * (phi_b0_DEP - Vbsc)) ;
           Q_s0 = here->HSM2_cnst0 * sqrt(T2 - 1.0 - T1 + 1e-15 + here->HSM2_cnst1 * (T3 - T4) ) ;
           T5 = 0.5 * here->HSM2_cnst0 * here->HSM2_cnst0 / Q_s0 ;
           Q_s0_dPs = T5 * (beta * T2 - beta + here->HSM2_cnst1 * ( - beta * T3) ) ;
           Q_s0_dPb = T5 * ( - beta * T2 + beta + here->HSM2_cnst1 * beta * T4 ) ;
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
         printf( "*** warning(HiSIM(%s)): Went Over Iteration Maximum(Ps0)\n",model->HSM2modName ) ;
         printf( " Vbse   = %7.3f Vdse = %7.3f Vgse = %7.3f\n" ,Vbse , Vdse , Vgse ) ;
       }

      /* caluculate derivative */

       y1_dVgs = 0.0 ;
       y1_dVds = 0.0 ;
       y1_dVbs = - (8.0 * phi_j0_DEP * C_ESI2 - 8.0 * C_ESI2 * phi_s0_DEP 
                  + 4.0 * q_Ndepm_esi * Tn2) / T10
                  * T9 * NdepmpNsub_inv1 * Vbsc_dVbse ;

       Q_b0_dep_dVbs = - C_ESI / W_b0 * T7 * T8 * NdepmpNsub_inv1 * Vbsc_dVbse ;

       Q_sub0_dep_dVbs = - C_ESI / W_sub0 * (NdepmpNsub_inv1 * Vbsc_dVbse - Vbsc_dVbse) ;

       T1 = beta * (phi_s0_DEP - phi_b0_DEP) ;
       T2 = exp(T1) ;
       if( phi_s0_DEP >= phi_b0_DEP ) {
         Q_s0_dVbs = 0.0 ;
       } else {
         T3 = exp( - beta * (phi_s0_DEP - Vbsc)) ;
         T4 = exp( - beta * (phi_b0_DEP - Vbsc)) ;
         T5 = sqrt(T2 - 1.0 - T1 + 1e-15 + here->HSM2_cnst1 * (T3 - T4)) ;
         Q_s0_dVbs = here->HSM2_cnst0 / 2.0 / T5 * 
                  (here->HSM2_cnst1 * (beta * T3 * Vbsc_dVbse - beta * T4 * Vbsc_dVbse) ) ;
       }

       y2_dVgs = Cox_dVg * (Vgp - phi_s0_DEP) + Cox * Vgp_dVgs ;
       y2_dVds = Cox_dVd * (Vgp - phi_s0_DEP) + Cox * Vgp_dVds ;
       y2_dVbs = Cox_dVb * (Vgp - phi_s0_DEP) + Cox * Vgp_dVbs + Q_s0_dVbs + Q_b0_dep_dVbs + Q_sub0_dep_dVbs ;

       phi_s0_DEP_dVgs = - ( rev11 * y1_dVgs + rev12 * y2_dVgs ) ;
       phi_s0_DEP_dVds = - ( rev11 * y1_dVds + rev12 * y2_dVds ) ;
       phi_s0_DEP_dVbs = - ( rev11 * y1_dVbs + rev12 * y2_dVbs ) ;

       phi_b0_DEP_dVgs = - ( rev21 * y1_dVgs + rev22 * y2_dVgs ) ;
       phi_b0_DEP_dVds = - ( rev21 * y1_dVds + rev22 * y2_dVds ) ;
       phi_b0_DEP_dVbs = - ( rev21 * y1_dVbs + rev22 * y2_dVbs ) ;

       if( W_bsub0 > model->HSM2_tndep && depmode !=2 ) {
         Fn_SU_CP2(phi_b0_DEP , phi_b0_DEP , phi_s0_DEP , 0.04, 2 , T1, T2 )  // HV_dlt=0.02
         phi_b0_DEP_dVgs = phi_b0_DEP_dVgs * T1 + phi_s0_DEP_dVgs * T2 ;
         phi_b0_DEP_dVds = phi_b0_DEP_dVds * T1 + phi_s0_DEP_dVds * T2 ;
         phi_b0_DEP_dVbs = phi_b0_DEP_dVbs * T1 + phi_s0_DEP_dVbs * T2 ;
       }

       phi_j0_DEP  = NdepmpNsub_inv1 * (NdepmpNsub * phi_b0_DEP + Vbsc - Vbi_DEP) ;
       phi_j0_DEP_dVgs = NdepmpNsub_inv1 * NdepmpNsub * phi_b0_DEP_dVgs ;
       phi_j0_DEP_dVds = NdepmpNsub_inv1 * NdepmpNsub * phi_b0_DEP_dVds ;
       phi_j0_DEP_dVbs = NdepmpNsub_inv1 * NdepmpNsub * phi_b0_DEP_dVbs + NdepmpNsub_inv1 * Vbsc_dVbse  ;

       phib_ref = (4.0 * phi_j0_DEP * phi_j0_DEP * C_ESI2 - 8.0 * phi_j0_DEP * C_ESI2 * phi_s0_DEP
                + 4.0 * C_ESI2 * phi_s0_DEP * phi_s0_DEP
                + 4.0 * phi_j0_DEP * q_Ndepm_esi * Tn2
                + 4.0 * phi_s0_DEP * q_Ndepm_esi * Tn2
                + Ndepm2 * C_QE2 * Tn2 * Tn2 ) / T10 ;

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

       T1 = beta * (phi_s0_DEP - phi_b0_DEP) ;
       T1_dVgs = beta * (phi_s0_DEP_dVgs - phi_b0_DEP_dVgs) ;
       T1_dVds = beta * (phi_s0_DEP_dVds - phi_b0_DEP_dVds) ;
       T1_dVbs = beta * (phi_s0_DEP_dVbs - phi_b0_DEP_dVbs) ;

       T2 = exp(T1) ; 
       T2_dVgs = T1_dVgs * T2 ;
       T2_dVds = T1_dVds * T2 ;
       T2_dVbs = T1_dVbs * T2 ;

       if( phi_s0_DEP >= phi_b0_DEP ) {

         T3 = sqrt(T2 - 1.0e0 - T1 + 1e-15 ) ;
         T3_dVgs = (T2_dVgs - T1_dVgs) / 2.0 / T3 ;
         T3_dVds = (T2_dVds - T1_dVds) / 2.0 / T3 ;
         T3_dVbs = (T2_dVbs - T1_dVbs) / 2.0 / T3 ;

         Q_s0 = - here->HSM2_cnst0 * T3 ;

         Q_s0_dep = 0.0 ;
         Q_sub0 = 0.0 ;

         W_b0 = sqrt(C_2ESIpq_Ndepm * (phi_b0_DEP - phi_j0_DEP) ) ;
         Fn_SU_CP( T9 , W_b0 , model->HSM2_tndep, 5e-8, 2 , T4 )  // HV_dlt=1e-8

         W_sub0 = sqrt(C_2ESIpq_Nsub * (phi_j0_DEP - Vbsc + Vbi_DEP) ) ;
         Q_b0_dep = T9 * q_Ndepm ;
         Q_sub0_dep = - W_sub0 * q_Nsub ;
 
        /* derivative */
         Q_s0_dVgs = - here->HSM2_cnst0 * T3_dVgs ;
         Q_s0_dVds = - here->HSM2_cnst0 * T3_dVds ;
         Q_s0_dVbs = - here->HSM2_cnst0 * T3_dVbs ;

         Q_n0 = Q_s0 ;
         Q_n0_dVgs = Q_s0_dVgs ;
         Q_n0_dVds = Q_s0_dVds ;
         Q_n0_dVbs = Q_s0_dVbs ;

         Q_b0_dep_dVgs = C_ESI / W_b0 * (phi_b0_DEP_dVgs - phi_j0_DEP_dVgs) * T4 ;
         Q_b0_dep_dVds = C_ESI / W_b0 * (phi_b0_DEP_dVds - phi_j0_DEP_dVds) * T4 ;
         Q_b0_dep_dVbs = C_ESI / W_b0 * (phi_b0_DEP_dVbs - phi_j0_DEP_dVbs) * T4 ;

         Q_sub0_dep_dVgs = - C_ESI / W_sub0 * phi_j0_DEP_dVgs ;
         Q_sub0_dep_dVds = - C_ESI / W_sub0 * phi_j0_DEP_dVds ;
         Q_sub0_dep_dVbs = - C_ESI / W_sub0 * (phi_j0_DEP_dVbs - Vbsc_dVbse) ;
         
         Q_sub0_dVgs = 0.0 ;
         Q_sub0_dVds = 0.0 ;
         Q_sub0_dVbs = 0.0 ;

         Q_s0_dep_dVgs = 0.0 ;
         Q_s0_dep_dVds = 0.0 ;
         Q_s0_dep_dVbs = 0.0 ;

       } else {

         T3 = exp( - beta * (phi_s0_DEP - Vbsc)) ;
         T4 = exp( - beta * (phi_b0_DEP - Vbsc)) ;
         T5 = sqrt(T2 - 1.0 - T1 + here->HSM2_cnst1 * (T3 - T4) + 1e-15) ;
         Q_s0 = here->HSM2_cnst0 * T5 ;

         T3_dVgs = - beta * T3 * phi_s0_DEP_dVgs ;
         T3_dVds = - beta * T3 * phi_s0_DEP_dVds ;
         T3_dVbs = - beta * T3 * (phi_s0_DEP_dVbs - Vbsc_dVbse) ;

         T4_dVgs = - beta * T4 * phi_b0_DEP_dVgs ;
         T4_dVds = - beta * T4 * phi_b0_DEP_dVds ;
         T4_dVbs = - beta * T4 * (phi_b0_DEP_dVbs - Vbsc_dVbse) ;

         T5_dVgs = (T2_dVgs - T1_dVgs + here->HSM2_cnst1 * (T3_dVgs - T4_dVgs)) / 2.0 / T5 ;
         T5_dVds = (T2_dVds - T1_dVds + here->HSM2_cnst1 * (T3_dVds - T4_dVds)) / 2.0 / T5 ;
         T5_dVbs = (T2_dVbs - T1_dVbs + here->HSM2_cnst1 * (T3_dVbs - T4_dVbs)) / 2.0 / T5 ;
 
         Q_s0_dVgs = here->HSM2_cnst0 * T5_dVgs ;
         Q_s0_dVds = here->HSM2_cnst0 * T5_dVds ;
         Q_s0_dVbs = here->HSM2_cnst0 * T5_dVbs ;
 
         if( W_bsub0 > model->HSM2_tndep && depmode !=2 ) {
           Q_sub0 = 0.0 ;
           Q_s0_dep = 0.0 ;

           Q_sub0_dVgs = 0.0 ;
           Q_sub0_dVds = 0.0 ;
           Q_sub0_dVbs = 0.0 ;

           Q_s0_dep_dVgs = 0.0 ;
           Q_s0_dep_dVds = 0.0 ;
           Q_s0_dep_dVbs = 0.0 ;
         } else {
           T3 = exp( - beta * (phi_s0_DEP - Vbsc)) ;
           T4 = exp( - beta * (phi_b0_DEP - Vbsc)) ;
           T5 = sqrt( - T1 + here->HSM2_cnst1 * (T3 - T4)) ;
           Q_sub0 = here->HSM2_cnst0 * T5 - here->HSM2_cnst0 * sqrt( - T1)  ;
           T6 = sqrt(T2 - 1.0e0 - T1 + 1e-15) ;
           Q_s0_dep = here->HSM2_cnst0 * T6 ;

           Q_sub0_dVgs = here->HSM2_cnst0 / 2.0 / T5 * ( - beta * (phi_s0_DEP_dVgs - phi_b0_DEP_dVgs)
              + here->HSM2_cnst1 * ( - beta * T3 * phi_s0_DEP_dVgs + beta * T4 * phi_b0_DEP_dVgs))
              - here->HSM2_cnst0 / 2.0 / sqrt( - T1) * ( - beta * (phi_s0_DEP_dVgs - phi_b0_DEP_dVgs)) ;
           Q_sub0_dVds = here->HSM2_cnst0 / 2.0 / T5 * ( - beta * (phi_s0_DEP_dVds - phi_b0_DEP_dVds)
              + here->HSM2_cnst1 * ( - beta * T3 * phi_s0_DEP_dVds + beta * T4 * phi_b0_DEP_dVds))
              - here->HSM2_cnst0 / 2.0 / sqrt( - T1) * ( - beta * (phi_s0_DEP_dVds - phi_b0_DEP_dVds)) ;
           Q_sub0_dVbs = here->HSM2_cnst0 / 2.0 / T5 * ( - beta * (phi_s0_DEP_dVbs - phi_b0_DEP_dVbs)
              + here->HSM2_cnst1 * ( - beta * T3 * (phi_s0_DEP_dVbs - Vbsc_dVbse) + beta * T4 * (phi_b0_DEP_dVbs - Vbsc_dVbse)))
              - here->HSM2_cnst0 / 2.0 / sqrt( - T1) * ( - beta * (phi_s0_DEP_dVbs - phi_b0_DEP_dVbs)) ;

           Q_s0_dep_dVgs = here->HSM2_cnst0 / 2.0 / T6 * beta * (phi_s0_DEP_dVgs - phi_b0_DEP_dVgs) * (T2 - 1) ;
           Q_s0_dep_dVds = here->HSM2_cnst0 / 2.0 / T6 * beta * (phi_s0_DEP_dVds - phi_b0_DEP_dVds) * (T2 - 1) ;
           Q_s0_dep_dVbs = here->HSM2_cnst0 / 2.0 / T6 * beta * (phi_s0_DEP_dVbs - phi_b0_DEP_dVbs) * (T2 - 1) ;

         }

         Q_n0 = 0.0 ;
         Q_n0_dVgs = 0.0 ;
         Q_n0_dVds = 0.0 ;
         Q_n0_dVbs = 0.0 ;


         T1 = phi_b0_DEP - phi_j0_DEP ;
         Fn_SL_CP( T2 , T1 , 0.0 , 0.1, 2 , T0 )
         W_b0 = sqrt(C_2ESIpq_Ndepm * (T2) ) ; 
         Fn_SU_CP( T9 , W_b0 , model->HSM2_tndep, 5e-8, 2 , T3 )  // HV_dlt=1e-8
         W_sub0 = sqrt(C_2ESIpq_Nsub * (phi_j0_DEP - Vbsc + Vbi_DEP) ) ;
         Q_b0_dep = T9 * q_Ndepm ;
         Q_sub0_dep = - W_sub0 * q_Nsub ;

         Q_b0_dep_dVgs = C_ESI / W_b0 * (phi_b0_DEP_dVgs - phi_j0_DEP_dVgs) * T0 * T3 ;
         Q_b0_dep_dVds = C_ESI / W_b0 * (phi_b0_DEP_dVds - phi_j0_DEP_dVds) * T0 * T3 ;
         Q_b0_dep_dVbs = C_ESI / W_b0 * (phi_b0_DEP_dVbs - phi_j0_DEP_dVbs) * T0 * T3 ;

         Q_sub0_dep_dVgs = - C_ESI / W_sub0 * phi_j0_DEP_dVgs ;
         Q_sub0_dep_dVds = - C_ESI / W_sub0 * phi_j0_DEP_dVds ;
         Q_sub0_dep_dVbs = - C_ESI / W_sub0 * (phi_j0_DEP_dVbs - Vbsc_dVbse) ;

       }

       T1 = phi_b0_DEP - phi_j0_DEP ;
       Fn_SL_CP( T2 , T1 , 0.0 , 0.1, 2 , T0 )
       W_b0 = sqrt(C_2ESIpq_Ndepm * (T2) ) ;
       Fn_SU_CP( T9, W_b0, model->HSM2_tndep, 1e-8, 2 , T3 )
       W_b0_dVgs = C_ESI / q_Ndepm / W_b0 * (phi_b0_DEP_dVgs - phi_j0_DEP_dVgs) * T0 * T3 ;
       W_b0_dVds = C_ESI / q_Ndepm / W_b0 * (phi_b0_DEP_dVds - phi_j0_DEP_dVds) * T0 * T3 ;
       W_b0_dVbs = C_ESI / q_Ndepm / W_b0 * (phi_b0_DEP_dVbs - phi_j0_DEP_dVbs) * T0 * T3 ;

       T1 = phi_b0_DEP - phi_s0_DEP ;
       Fn_SL_CP( T2 , T1 , 0.0 , 0.1, 2 , T0 ) // HV_dlt=0.05
       W_s0 = sqrt(C_2ESIpq_Ndepm * (T2) ) ;

       W_s0_dVgs = C_ESI / q_Ndepm / W_s0 * (phi_b0_DEP_dVgs - phi_s0_DEP_dVgs) * T0 ;
       W_s0_dVds = C_ESI / q_Ndepm / W_s0 * (phi_b0_DEP_dVds - phi_s0_DEP_dVds) * T0 ;
       W_s0_dVbs = C_ESI / q_Ndepm / W_s0 * (phi_b0_DEP_dVbs - phi_s0_DEP_dVbs) * T0 ;

       T1 = model->HSM2_tndep - T9 - W_s0 ;
       Fn_SL_CP( W_res0 , T1 , 1.0e-20 , 1.0e-14, 2 , T0 ) // HV_dlt=1.0e-25,1.0e-18

       Qn_res0 = - W_res0 * q_Ndepm ;
       Qn_res0_dVgs = (W_s0_dVgs + W_b0_dVgs) * q_Ndepm * T0 ;
       Qn_res0_dVds = (W_s0_dVds + W_b0_dVds) * q_Ndepm * T0 ;
       Qn_res0_dVbs = (W_s0_dVbs + W_b0_dVbs) * q_Ndepm * T0 ;

       if( W_bsub0 > model->HSM2_tndep && depmode !=2 ) {
         Fn_SU_CP(T3 , phi_s0_DEP , phi_b0_DEP_lim , 0.8, 2 , T1 )
         T3_dVgs = phi_s0_DEP_dVgs * T1 ;
         T3_dVds = phi_s0_DEP_dVds * T1 ;
         T3_dVbs = phi_s0_DEP_dVbs * T1 ;
       } else {
         Fn_SU_CP(T3 , phib_ref , phi_b0_DEP_lim , 0.8, 2 , T0 )
         T3_dVgs = phib_ref_dVgs * T0 ;
         T3_dVds = phib_ref_dVds * T0 ;
         T3_dVbs = phib_ref_dVbs * T0 ;
       }

       T4 = exp(beta * (T3 - phi_b0_DEP_lim)) ;
       T5 = - C_QE * here->HSM2_ndepm ;
       Qn_bac0 = T5 * T4 * T9 ;
       Qn_bac0_dVgs = T5 * (beta * T4 * T3_dVgs * T9 + T4 * W_b0_dVgs) ;
       Qn_bac0_dVds = T5 * (beta * T4 * T3_dVds * T9 + T4 * W_b0_dVds) ;
       Qn_bac0_dVbs = T5 * (beta * T4 * T3_dVbs * T9 + T4 * W_b0_dVbs) ;


       T1 = phi_s0_DEP - phi_b0_DEP_lim ;
       Fn_SL_CP( T2 , T1 , 0.0, Ps_delta, 2 , T0 )
       T2_dVgs = phi_s0_DEP_dVgs * T0 ;
       T2_dVds = phi_s0_DEP_dVds * T0 ;
       T2_dVbs = phi_s0_DEP_dVbs * T0 ;

       T3 = exp(beta * (T2)) ;
       T3_dVgs = beta * T3 * T2_dVgs ;
       T3_dVds = beta * T3 * T2_dVds ;
       T3_dVbs = beta * T3 * T2_dVbs ;

       T4 = T3 - 1.0 - beta * T2 ;
       
       T4_dVgs = T3_dVgs - beta * T2_dVgs ;
       T4_dVds = T3_dVds - beta * T2_dVds ;
       T4_dVbs = T3_dVbs - beta * T2_dVbs ;

       T5 = sqrt(T4) ;
       Q_n0_cur = - here->HSM2_cnst0 * T5 ;
       Q_n0_cur_dVgs = - here->HSM2_cnst0 / 2.0 / T5 * T4_dVgs ;
       Q_n0_cur_dVds = - here->HSM2_cnst0 / 2.0 / T5 * T4_dVds ;
       Q_n0_cur_dVbs = - here->HSM2_cnst0 / 2.0 / T5 * T4_dVbs ;

       T4 = exp(beta * Ps_delta0) - 1.0 - beta * Ps_delta0 ;
       T5 = sqrt(T4) ;
       Qn_delta = here->HSM2_cnst0 * T5 ;



     /*-----------------------------------------------------------*
      * Start point of phi_sL_DEP(= phi_s0_DEP + Pds) calculation.(label)
      *-----------------*/

     /* Vdseff (begin) */
       Vdsorg = Vds ;

       if( Vds > 1e-3 ) {

         T2 = q_Ndepm_esi / ( Cox * Cox ) ;
         T4 = - 2.0e0 * T2 / Cox ;
         T2_dVb = T4 * Cox_dVb ; 
         T2_dVd = T4 * Cox_dVd ;
         T2_dVg = T4 * Cox_dVg ;

         T0 = Vgp + DEPQFN1 - beta_inv - Vbsz ;
         T0_dVg = Vgp_dVgs ;
         T0_dVd = Vgp_dVds - Vbsz_dVds ;
         T0_dVb = Vgp_dVbs - Vbsz_dVbs ;

         T4 = 1.0e0 + 2.0e0 / T2 * T0 ;
         T4_dVg = 2.0 / T2 * T0_dVg - 2.0 / T2 / T2 * T0 * T2_dVg ;
         T4_dVd = 2.0 / T2 * T0_dVd - 2.0 / T2 / T2 * T0 * T2_dVd ;
         T4_dVb = 2.0 / T2 * T0_dVb - 2.0 / T2 / T2 * T0 * T2_dVb ;

         Fn_SL_CP( T9 , T4 , 0 , DEPQFN_dlt, 2 , T0 )
         T9_dVg = T4_dVg * T0 ;
         T9_dVd = T4_dVd * T0 ;
         T9_dVb = T4_dVb * T0 ;

         T9 +=small ; 
         T3 = sqrt( T9 ) ;
         T3_dVg = 0.5 / T3 * T9_dVg  ;
         T3_dVd = 0.5 / T3 * T9_dVd  ;
         T3_dVb = 0.5 / T3 * T9_dVb  ;

         T10 = Vgp + DEPQFN1 + T2 * ( 1.0e0 - T3 ) ;
         T10_dVb = Vgp_dVbs + T2_dVb * ( 1.0e0 - T3 ) - T2 * T3_dVb ;
         T10_dVd = Vgp_dVds + T2_dVd * ( 1.0e0 - T3 ) - T2 * T3_dVd ;
         T10_dVg = Vgp_dVgs + T2_dVg * ( 1.0e0 - T3 ) - T2 * T3_dVg ;

         Fn_SL_CP( T10 , T10 , DEPQFN3, 0.2, 4 , T0 )
//       T10 = T10 + epsm10 ; 
         T10_dVb *=T0 ;
         T10_dVd *=T0 ;
         T10_dVg *=T0 ;

         T1 = Vds / T10 ;
         T2 = Fn_Pow( T1 , here->HSM2_ddlt - 1.0e0 ) ;
         T7 = T2 * T1 ;
         T0 = here->HSM2_ddlt * T2 / ( T10 * T10 ) ;
         T7_dVb = T0 * ( - Vds * T10_dVb ) ;
         T7_dVd = T0 * ( T10 - Vds * T10_dVd ) ;
         T7_dVg = T0 * ( - Vds * T10_dVg ) ;

         T3 = 1.0 + T7 ; 
         T4 = Fn_Pow( T3 , 1.0 / here->HSM2_ddlt - 1.0 ) ;
         T6 = T4 * T3 ;
         T0 = T4 / here->HSM2_ddlt ;
         T6_dVb = T0 * T7_dVb ;
         T6_dVd = T0 * T7_dVd ;
         T6_dVg = T0 * T7_dVg ;

         Vdseff = Vds / T6 ;
         T0 = 1.0 / ( T6 * T6 ) ; 
         Vdseff0_dVbs = - Vds * T6_dVb * T0 ;
         Vdseff0_dVds = ( T6 - Vds * T6_dVd ) * T0 ;
         Vdseff0_dVgs = - Vds * T6_dVg * T0 ;

         Fn_SL_CP( Vgpp , Vgp , 0.0 , 0.5, 2 , T0 )
         Vgpp_dVgs = T0 * Vgp_dVgs ;
         Vgpp_dVds = T0 * Vgp_dVds ;
         Vgpp_dVbs = T0 * Vgp_dVbs ;

         T1 = Vgpp * 0.8 ;
         T1_dVg = Vgpp_dVgs * 0.8 ;
         T1_dVd = Vgpp_dVds * 0.8 ;
         T1_dVb = Vgpp_dVbs * 0.8 ;

         Fn_SU_CP3( Vds , Vdseff , Vgpp , T1, 2 , T3, T4, T5 )
         Vdseff_dVgs = Vdseff0_dVgs * T3 + Vgpp_dVgs * T4 + T1_dVg * T5 ;
         Vdseff_dVds = Vdseff0_dVds * T3 + Vgpp_dVds * T4 + T1_dVd * T5 ;
         Vdseff_dVbs = Vdseff0_dVbs * T3 + Vgpp_dVbs * T4 + T1_dVb * T5 ;

       } else {

         Vdseff = Vds ;
         Vdseff0_dVgs = 0.0 ;
         Vdseff0_dVds = 1.0 ;
         Vdseff0_dVbs = 0.0 ;

         Vdseff_dVgs = 0.0 ;
         Vdseff_dVds = 1.0 ;
         Vdseff_dVbs = 0.0 ;

       }
       /* Vdseff (end) */

     /*---------------------------------------------------*
      * start of phi_sL_DEP calculation. (label)
      *--------------------------------*/

       if( Vds < 0.0e0 ) {

         phi_sL_DEP = phi_s0_DEP ;
         phi_sL_DEP_dVgs = phi_s0_DEP_dVgs ;
         phi_sL_DEP_dVds = phi_s0_DEP_dVds ;
         phi_sL_DEP_dVbs = phi_s0_DEP_dVbs ;

         phi_bL_DEP = phi_b0_DEP ;
         phi_bL_DEP_dVgs = phi_b0_DEP_dVgs ;
         phi_bL_DEP_dVds = phi_b0_DEP_dVds ;
         phi_bL_DEP_dVbs = phi_b0_DEP_dVbs ;

         phi_jL_DEP = phi_j0_DEP ;
         phi_jL_DEP_dVgs = phi_j0_DEP_dVgs ;
         phi_jL_DEP_dVds = phi_j0_DEP_dVds ;
         phi_jL_DEP_dVbs = phi_j0_DEP_dVbs ;

         Q_subL = Q_sub0 ;
         Q_subL_dVgs = Q_sub0_dVgs ;
         Q_subL_dVds = Q_sub0_dVds ;
         Q_subL_dVbs = Q_sub0_dVbs ;

         Q_nL = Q_n0 ;
         Q_nL_dVgs = Q_n0_dVgs ;
         Q_nL_dVds = Q_n0_dVds ;
         Q_nL_dVbs = Q_n0_dVbs ;

	 Q_bL_dep = Q_b0_dep ;
	 Q_bL_dep_dVgs = Q_b0_dep_dVgs ;
	 Q_bL_dep_dVds = Q_b0_dep_dVds ;
	 Q_bL_dep_dVbs = Q_b0_dep_dVbs ;

	 Q_subL_dep = Q_sub0_dep ;
	 Q_subL_dep_dVgs = Q_sub0_dep_dVgs ;
	 Q_subL_dep_dVds = Q_sub0_dep_dVds ;
	 Q_subL_dep_dVbs = Q_sub0_dep_dVbs ;

	 Q_sL_dep = Q_s0_dep ;
	 Q_sL_dep_dVgs = Q_s0_dep_dVgs ;
	 Q_sL_dep_dVds = Q_s0_dep_dVds ;
	 Q_sL_dep_dVbs = Q_s0_dep_dVbs ;

	 Q_nL_cur = Q_n0_cur ;
	 Q_nL_cur_dVgs = Q_n0_cur_dVgs ;
	 Q_nL_cur_dVds = Q_n0_cur_dVds ;
	 Q_nL_cur_dVbs = Q_n0_cur_dVbs ;

       } else {

         W_bsubL = sqrt(C_2ESIpq_Ndepm * here->HSM2_nsub / (here->HSM2_nsub + here->HSM2_ndepm) * (Vds - Vbsc + Vbi_DEP)) ;

       /*---------------------------------------------------*
        * region judgement  
        *------------------*/

        /* fully depleted case */
         if( W_bsubL > model->HSM2_tndep ) {

           Vgp0 = Vds ;
           W_bL = model->HSM2_tndep ;
           phi_bL_DEP = Vds ;
           phi_bL_DEP_lim = Vds ;
           phi_jL_DEP = phi_bL_DEP - C_2ESIpq_Ndepm_inv * W_bL * W_bL ;

           Vgp0old = Vgp0 ;
           phi_jL_DEP_old = phi_jL_DEP ;

           Q_bL_dep = W_bL * q_Ndepm ;

           for ( lp_s0 = 1 ; lp_s0 <= lp_se_max + 1 ; lp_s0 ++ ) {

             W_bL = sqrt(C_2ESIpq_Ndepm * (phi_bL_DEP - phi_jL_DEP) ) ;
             Fn_SU_CP( W_bL , W_bL , model->HSM2_tndep , 1e-8, 2 , T0 )
             W_subL = sqrt(C_2ESIpq_Nsub * (phi_jL_DEP - Vbsc + Vbi_DEP) ) ;

             Q_bL_dep = W_bL * q_Ndepm ;
             Q_bL_dep_dPd = - C_ESI / W_bL * T0 ;
             Q_subL_dep = - W_subL * q_Nsub ;
             Q_subL_dep_dPd = - C_ESI / W_subL ;

             y1 = Cox * (Vgp0 - phi_bL_DEP) + Q_bL_dep + Q_subL_dep ;
             y11 = Cox ;
             y12 = Q_bL_dep_dPd + Q_subL_dep_dPd ;

             y2 = phi_jL_DEP - NdepmpNsub_inv1 * (NdepmpNsub * phi_bL_DEP + Vbsc - Vbi_DEP) ;
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

           W_subL = model->HSM2_tndep * NdepmpNsub ;
           phi_jL_DEP = C_2ESIpq_Nsub_inv * W_subL * W_subL + Vbsc - Vbi_DEP ;
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
           phi_jL_DEP = C_2ESIpq_Nsub_inv * W_subL * W_subL + Vbsc - Vbi_DEP ;
           phi_bL_DEP = C_2ESIpq_Ndepm_inv * W_bL * W_bL + phi_jL_DEP ;
           phi_jL_DEP_acc = phi_jL_DEP ;
           if( Vgp > Vgp0 ) {
             depmode = 1 ;
           } else {
             depmode = 2 ;
           }

         }

         T1 = C_2ESI_q_Ndepm * ( Psbmax - ( - here->HSM2_Pb2n + Vbsc)) ;
         if ( T1 > 0.0 ) {
           vthn = - here->HSM2_Pb2n + Vbsc - sqrt(T1) / Cox ;
         } else {
           vthn = - here->HSM2_Pb2n + Vbsc ;
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

             if( W_sL + W_bL > model->HSM2_tndep ) {
               for ( lp_s0 = 1 ; lp_s0 <= lp_se_max + 1 ; lp_s0 ++ ) {

                 y0 = W_sL + W_bL - model->HSM2_tndep ;

                 dydPsm = C_ESI / q_Ndepm / W_sL 
                          + C_ESI / q_Ndepm * ( 1.0 - (here->HSM2_ndepm
                          / here->HSM2_nsub) / ( 1.0 + (NdepmpNsub))) / W_bL ;

                 if( fabs(y0 / dydPsm) > 0.5 ) {
                   phi_bL_DEP = phi_bL_DEP - 0.5 * Fn_Sgn(y0 / dydPsm) ;
                 } else {
                   phi_bL_DEP = phi_bL_DEP - y0 / dydPsm ;
                 }
                 if( (phi_bL_DEP - Vbsc + Vbi_DEP) < epsm10 ) 
                    phi_bL_DEP=Vbsc - Vbi_DEP + epsm10 ;

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
                   + Vbsc - Vbi_DEP) / (1.0 + NdepmpNsub) ;
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

           phi_jL_DEP  = NdepmpNsub_inv1 * (NdepmpNsub * phi_bL_DEP + Vbsc - Vbi_DEP) ;
           phi_jL_DEP_dPb = NdepmpNsub_inv1 * NdepmpNsub ;

           T1 = phi_bL_DEP - phi_jL_DEP ;
           Fn_SL_CP( T2 , T1 , 0.0 , 0.1, 2 , T7 )
           W_bL = sqrt(C_2ESIpq_Ndepm * (T2) ) ;
           Fn_SU_CP( W_bL , W_bL , model->HSM2_tndep , 1e-8, 2 , T8 )
           W_subL = sqrt(C_2ESIpq_Nsub * (phi_jL_DEP - Vbsc + Vbi_DEP) ) ;
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
                    + Ndepm2 * C_QE2 * Tn2 * Tn2  ) / T10 ;
           phib_ref_dPs = ( - 8.0 * phi_jL_DEP * C_ESI2 + 8.0 * C_ESI2 * phi_sL_DEP 
                        + 4.0 * q_Ndepm_esi * Tn2 ) / T10 ;
           phib_ref_dPd = (   8.0 * phi_jL_DEP * C_ESI2 - 8.0 * C_ESI2 * phi_sL_DEP
                        + 4.0 * q_Ndepm_esi * Tn2 ) / T10 ;

           T1 = beta * (phi_sL_DEP - phi_bL_DEP) ;
           T2 = exp(T1) ;
           if( phi_sL_DEP >= phi_bL_DEP ) {
             Q_sL = - here->HSM2_cnst0 * sqrt(T2 - 1.0 - T1 + 1e-15) ;
             Q_sL_dPs = 0.5 * here->HSM2_cnst0 * here->HSM2_cnst0 / Q_sL * (beta * T2 - beta) ;
             Q_sL_dPb = - Q_sL_dPs ;
           } else {
             T3 = exp( - beta * (phi_sL_DEP - Vbsc)) ;
             T4 = exp( - beta * (phi_bL_DEP - Vbsc)) ;
             Q_sL = here->HSM2_cnst0 * sqrt(T2 - 1.0 - T1 + here->HSM2_cnst1 * (T3 - T4) + 1e-15) ;
             T5 = 0.5 * here->HSM2_cnst0 * here->HSM2_cnst0 / Q_sL ;
             Q_sL_dPs = T5 * (beta * T2 - beta + here->HSM2_cnst1 * ( - beta * T3) ) ;
             Q_sL_dPb = T5 * ( - beta * T2 + beta + here->HSM2_cnst1 * beta * T4 ) ;
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
           if( fabs( rev21 * y1 + rev22 * y2 ) > 0.2 ) { // HV_dlt=0.5
             phi_sL_DEP = phi_sL_DEP - 0.2 * Fn_Sgn( rev11 * y1 + rev12 * y2 ) ;
             phi_bL_DEP = phi_bL_DEP - 0.2 * Fn_Sgn( rev21 * y1 + rev22 * y2 ) ;
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
           printf( "*** warning(HiSIM(%s)): Went Over Iteration Maximum(Psl)\n",model->HSM2modName ) ;
           printf( " Vbse   = %7.3f Vdse = %7.3f Vgse = %7.3f\n" ,Vbse , Vdse , Vgse ) ;
         } 

        /* caluculate derivative */

         y1_dVgs = - Vdseff_dVgs * T11 ;
         y1_dVds = - Vdseff_dVds * T11 ;
         y1_dVbs = - (8.0 * phi_jL_DEP * C_ESI2 - 8.0 * C_ESI2 * phi_sL_DEP
                  + 4.0 * q_Ndepm_esi * Tn2) / T10
                  * T9 * NdepmpNsub_inv1 * Vbsc_dVbse - Vdseff_dVbs * T11 ;

         Q_bL_dep_dVbs = - C_ESI / W_bL * T7 * T8 * NdepmpNsub_inv1 * Vbsc_dVbse ;

         Q_subL_dep_dVbs = - C_ESI / W_subL * (NdepmpNsub_inv1 * Vbsc_dVbse - Vbsc_dVbse) ;

         T1 = beta * (phi_sL_DEP - phi_bL_DEP) ;
         T2 = exp(T1) ; 
         if( phi_sL_DEP >= phi_bL_DEP ) {
           Q_sL_dVbs = 0.0 ;
         } else {
           T3 = exp( - beta * (phi_sL_DEP - Vbsc)) ;
           T4 = exp( - beta * (phi_bL_DEP - Vbsc)) ;
           T5 = sqrt(T2 - 1.0 - T1 + 1e-15 + here->HSM2_cnst1 * (T3 - T4)) ;
           Q_sL_dVbs = here->HSM2_cnst0 / 2.0 / T5 * 
                  (here->HSM2_cnst1 * (beta * T3 * Vbsc_dVbse - beta * T4 * Vbsc_dVbse) ) ;
         }

         y2_dVgs = Cox_dVg * (Vgp - phi_sL_DEP) + Cox * Vgp_dVgs ;
         y2_dVds = Cox_dVd * (Vgp - phi_sL_DEP) + Cox * Vgp_dVds ;
         y2_dVbs = Cox_dVb * (Vgp - phi_sL_DEP) + Cox * Vgp_dVbs + Q_sL_dVbs + Q_bL_dep_dVbs + Q_subL_dep_dVbs ;

         phi_sL_DEP_dVgs = - ( rev11 * y1_dVgs + rev12 * y2_dVgs ) ;
         phi_sL_DEP_dVds = - ( rev11 * y1_dVds + rev12 * y2_dVds ) ;
         phi_sL_DEP_dVbs = - ( rev11 * y1_dVbs + rev12 * y2_dVbs ) ;

         phi_bL_DEP_dVgs = - ( rev21 * y1_dVgs + rev22 * y2_dVgs ) ;
         phi_bL_DEP_dVds = - ( rev21 * y1_dVds + rev22 * y2_dVds ) ;
         phi_bL_DEP_dVbs = - ( rev21 * y1_dVbs + rev22 * y2_dVbs ) ;

         if( W_bsubL > model->HSM2_tndep && depmode !=2 ) {
           Fn_SU_CP2(phi_bL_DEP , phi_bL_DEP , phi_sL_DEP , 0.04, 2 , T1, T2 ) // HV_dlt=0.02
           phi_bL_DEP_dVgs = phi_bL_DEP_dVgs * T1 + phi_sL_DEP_dVgs * T2 ;
           phi_bL_DEP_dVds = phi_bL_DEP_dVds * T1 + phi_sL_DEP_dVds * T2 ;
           phi_bL_DEP_dVbs = phi_bL_DEP_dVbs * T1 + phi_sL_DEP_dVbs * T2 ;
         }

         phi_jL_DEP  = NdepmpNsub_inv1 * (NdepmpNsub * phi_bL_DEP + Vbsc - Vbi_DEP) ;
         phi_jL_DEP_dVgs = NdepmpNsub_inv1 * NdepmpNsub * phi_bL_DEP_dVgs ;
         phi_jL_DEP_dVds = NdepmpNsub_inv1 * NdepmpNsub * phi_bL_DEP_dVds ;
         phi_jL_DEP_dVbs = NdepmpNsub_inv1 * NdepmpNsub * phi_bL_DEP_dVbs + NdepmpNsub_inv1 * Vbsc_dVbse  ;

         phib_ref = (4.0 * phi_jL_DEP * phi_jL_DEP * C_ESI2 - 8.0 * phi_jL_DEP * C_ESI2 * phi_sL_DEP
                  + 4.0 * C_ESI2 * phi_sL_DEP * phi_sL_DEP
                  + 4.0 * phi_jL_DEP * q_Ndepm_esi * Tn2
                  + 4.0 * phi_sL_DEP * q_Ndepm_esi * Tn2
                  + Ndepm2 * C_QE2 * Tn2 * Tn2 ) / T10 ; 

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

         T1 = beta * (phi_sL_DEP - phi_bL_DEP) ;
         T1_dVgs = beta * (phi_sL_DEP_dVgs - phi_bL_DEP_dVgs) ;
         T1_dVds = beta * (phi_sL_DEP_dVds - phi_bL_DEP_dVds) ;
         T1_dVbs = beta * (phi_sL_DEP_dVbs - phi_bL_DEP_dVbs) ;

         T2 = exp(T1) ;
         T2_dVgs = T1_dVgs * T2 ;
         T2_dVds = T1_dVds * T2 ;
         T2_dVbs = T1_dVbs * T2 ;

         if( phi_sL_DEP >= phi_bL_DEP ) {
           T3 = sqrt(T2 - 1.0e0 - T1 + 1e-15 ) ;
           T3_dVgs = (T2_dVgs - T1_dVgs) / 2.0 / T3 ;
           T3_dVds = (T2_dVds - T1_dVds) / 2.0 / T3 ;
           T3_dVbs = (T2_dVbs - T1_dVbs) / 2.0 / T3 ;

           Q_sL = - here->HSM2_cnst0 * T3 ;

           Q_sL_dep = 0.0 ;
           Q_subL = 0.0 ;

           W_bL = sqrt(C_2ESIpq_Ndepm * (phi_bL_DEP - phi_jL_DEP) ) ;
           Fn_SU_CP( T9 , W_bL , model->HSM2_tndep, 1e-8, 2 , T4 )

           W_subL = sqrt(C_2ESIpq_Nsub * (phi_jL_DEP - Vbsc + Vbi_DEP) ) ;
           Q_bL_dep = T9 * q_Ndepm ;
           Q_subL_dep = - W_subL * q_Nsub ;

          /* derivative */
           Q_sL_dVgs = - here->HSM2_cnst0 * T3_dVgs ;
           Q_sL_dVds = - here->HSM2_cnst0 * T3_dVds ;
           Q_sL_dVbs = - here->HSM2_cnst0 * T3_dVbs ;

           Q_nL = Q_sL ;
           Q_nL_dVgs = Q_sL_dVgs ;
           Q_nL_dVds = Q_sL_dVds ;
           Q_nL_dVbs = Q_sL_dVbs ;

           Q_bL_dep_dVgs = C_ESI / W_bL * (phi_bL_DEP_dVgs - phi_jL_DEP_dVgs) * T4 ;
           Q_bL_dep_dVds = C_ESI / W_bL * (phi_bL_DEP_dVds - phi_jL_DEP_dVds) * T4 ;
           Q_bL_dep_dVbs = C_ESI / W_bL * (phi_bL_DEP_dVbs - phi_jL_DEP_dVbs) * T4 ;

           Q_subL_dep_dVgs = - C_ESI / W_subL * phi_jL_DEP_dVgs ;
           Q_subL_dep_dVds = - C_ESI / W_subL * phi_jL_DEP_dVds ;
           Q_subL_dep_dVbs = - C_ESI / W_subL * (phi_jL_DEP_dVbs - Vbsc_dVbse) ;

           Q_subL_dVgs = 0.0 ;
           Q_subL_dVds = 0.0 ;
           Q_subL_dVbs = 0.0 ;

           Q_sL_dep_dVgs = 0.0 ;
           Q_sL_dep_dVds = 0.0 ;
           Q_sL_dep_dVbs = 0.0 ;

         } else {

           T3 = exp( - beta * (phi_sL_DEP - Vbsc)) ;
           T4 = exp( - beta * (phi_bL_DEP - Vbsc)) ;
           T5 = sqrt(T2 - 1.0 - T1 + here->HSM2_cnst1 * (T3 - T4) + 1e-15) ;
           Q_sL = here->HSM2_cnst0 * T5 ;

           T3_dVgs = - beta * T3 * phi_sL_DEP_dVgs ;
           T3_dVds = - beta * T3 * phi_sL_DEP_dVds ;
           T3_dVbs = - beta * T3 * (phi_sL_DEP_dVbs - Vbsc_dVbse) ;

           T4_dVgs = - beta * T4 * phi_bL_DEP_dVgs ;
           T4_dVds = - beta * T4 * phi_bL_DEP_dVds ;
           T4_dVbs = - beta * T4 * (phi_bL_DEP_dVbs - Vbsc_dVbse) ;

           T5_dVgs = (T2_dVgs - T1_dVgs + here->HSM2_cnst1 * (T3_dVgs - T4_dVgs)) / 2.0 / T5 ;
           T5_dVds = (T2_dVds - T1_dVds + here->HSM2_cnst1 * (T3_dVds - T4_dVds)) / 2.0 / T5 ;
           T5_dVbs = (T2_dVbs - T1_dVbs + here->HSM2_cnst1 * (T3_dVbs - T4_dVbs)) / 2.0 / T5 ;

           Q_sL_dVgs = here->HSM2_cnst0 * T5_dVgs ;
           Q_sL_dVds = here->HSM2_cnst0 * T5_dVds ;
           Q_sL_dVbs = here->HSM2_cnst0 * T5_dVbs ;

           if( W_bsubL > model->HSM2_tndep && depmode !=2 ) {
             Q_subL = 0.0 ;
             Q_sL_dep = 0.0 ;

             Q_subL_dVgs = 0.0 ;
             Q_subL_dVds = 0.0 ;
             Q_subL_dVbs = 0.0 ;

             Q_sL_dep_dVgs = 0.0 ;
             Q_sL_dep_dVds = 0.0 ;
             Q_sL_dep_dVbs = 0.0 ;

           } else {
             T3 = exp( - beta * (phi_sL_DEP - Vbsc)) ;
             T4 = exp( - beta * (phi_bL_DEP - Vbsc)) ;
             T5 = sqrt( - T1 + here->HSM2_cnst1 * (T3 - T4)) ;
             Q_subL = here->HSM2_cnst0 * T5 - here->HSM2_cnst0 * sqrt( - T1)  ;
             T6 = sqrt(T2 - 1.0e0 - T1 + 1e-15) ;
             Q_sL_dep = here->HSM2_cnst0 * T6 ;

             Q_subL_dVgs = here->HSM2_cnst0 / 2.0 / T5 * ( - beta * (phi_sL_DEP_dVgs - phi_bL_DEP_dVgs)
                + here->HSM2_cnst1 * ( - beta * T3 * phi_sL_DEP_dVgs + beta * T4 * phi_bL_DEP_dVgs))
                - here->HSM2_cnst0 / 2.0 / sqrt( - T1) * ( - beta * (phi_sL_DEP_dVgs - phi_bL_DEP_dVgs)) ;
             Q_subL_dVds = here->HSM2_cnst0 / 2.0 / T5 * ( - beta * (phi_sL_DEP_dVds - phi_bL_DEP_dVds)
                + here->HSM2_cnst1 * ( - beta * T3 * phi_sL_DEP_dVds + beta * T4 * phi_bL_DEP_dVds))
                - here->HSM2_cnst0 / 2.0 / sqrt( - T1) * ( - beta * (phi_sL_DEP_dVds - phi_bL_DEP_dVds)) ;
             Q_subL_dVbs = here->HSM2_cnst0 / 2.0 / T5 * ( - beta * (phi_sL_DEP_dVbs - phi_bL_DEP_dVbs)
                + here->HSM2_cnst1 * ( - beta * T3 * (phi_sL_DEP_dVbs - Vbsc_dVbse) + beta * T4 * (phi_bL_DEP_dVbs - Vbsc_dVbse)))
                - here->HSM2_cnst0 / 2.0 / sqrt( - T1) * ( - beta * (phi_sL_DEP_dVbs - phi_bL_DEP_dVbs)) ;

             Q_sL_dep_dVgs = here->HSM2_cnst0 / 2.0 / T6 * beta * (phi_sL_DEP_dVgs - phi_bL_DEP_dVgs) * (T2 - 1) ;
             Q_sL_dep_dVds = here->HSM2_cnst0 / 2.0 / T6 * beta * (phi_sL_DEP_dVds - phi_bL_DEP_dVds) * (T2 - 1) ;
             Q_sL_dep_dVbs = here->HSM2_cnst0 / 2.0 / T6 * beta * (phi_sL_DEP_dVbs - phi_bL_DEP_dVbs) * (T2 - 1) ;

           }

           Q_nL = 0.0 ;
           Q_nL_dVgs = 0.0 ;
           Q_nL_dVds = 0.0 ;
           Q_nL_dVbs = 0.0 ;

///        Qg = Cox * (Vgp - phi_sL_DEP) ;

           T1 = phi_bL_DEP - phi_jL_DEP ;
           Fn_SL_CP( T2 , T1 , 0.0 , 0.1, 2 , T0 )
           W_bL = sqrt(C_2ESIpq_Ndepm * (T2) ) ;
           Fn_SU_CP( T9 , W_bL , model->HSM2_tndep, 1e-8, 2 , T3 )
           W_subL = sqrt(C_2ESIpq_Nsub * (phi_jL_DEP - Vbsc + Vbi_DEP) ) ;
           Q_bL_dep = T9 * q_Ndepm ;
           Q_subL_dep = - W_subL * q_Nsub ;

           Q_bL_dep_dVgs = C_ESI / W_bL * (phi_bL_DEP_dVgs - phi_jL_DEP_dVgs) * T0 * T3 ;
           Q_bL_dep_dVds = C_ESI / W_bL * (phi_bL_DEP_dVds - phi_jL_DEP_dVds) * T0 * T3 ;
           Q_bL_dep_dVbs = C_ESI / W_bL * (phi_bL_DEP_dVbs - phi_jL_DEP_dVbs) * T0 * T3 ;

           Q_subL_dep_dVgs = - C_ESI / W_subL * phi_jL_DEP_dVgs ;
           Q_subL_dep_dVds = - C_ESI / W_subL * phi_jL_DEP_dVds ;
           Q_subL_dep_dVbs = - C_ESI / W_subL * (phi_jL_DEP_dVbs - Vbsc_dVbse) ;

         }


         T1 = phi_sL_DEP - phi_bL_DEP_lim ;
         Fn_SL_CP( T2 , T1 , 0.0, Ps_delta, 2 , T0 )
         T2_dVgs = (phi_sL_DEP_dVgs - Vdseff_dVgs) * T0 ;
         T2_dVds = (phi_sL_DEP_dVds - Vdseff_dVds) * T0 ;
         T2_dVbs = (phi_sL_DEP_dVbs - Vdseff_dVbs) * T0 ;

         T3 = exp(beta * (T2)) ;
         T3_dVgs = beta * T3 * T2_dVgs ;
         T3_dVds = beta * T3 * T2_dVds ;
         T3_dVbs = beta * T3 * T2_dVbs ;

         T4 = T3 - 1.0 - beta * T2 ;

         T4_dVgs = T3_dVgs - beta * T2_dVgs ;
         T4_dVds = T3_dVds - beta * T2_dVds ;
         T4_dVbs = T3_dVbs - beta * T2_dVbs ;

         T5 = sqrt(T4) ;
         Q_nL_cur = - here->HSM2_cnst0 * T5 ;
         Q_nL_cur_dVgs = - here->HSM2_cnst0 / 2.0 / T5 * T4_dVgs ;
         Q_nL_cur_dVds = - here->HSM2_cnst0 / 2.0 / T5 * T4_dVds ;
         Q_nL_cur_dVbs = - here->HSM2_cnst0 / 2.0 / T5 * T4_dVbs ;

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

       if( Pds < 0.0 ) { // take care of numerical noise //
         Pds = 0.0 ; 
         Pds_dVgs = 0.0 ;
         Pds_dVds = 0.0 ;
         Pds_dVbs = 0.0 ;

         phi_sL_DEP = phi_s0_DEP ;
         phi_sL_DEP_dVgs = phi_s0_DEP_dVgs ;
         phi_sL_DEP_dVds = phi_s0_DEP_dVds ;
         phi_sL_DEP_dVbs = phi_s0_DEP_dVbs ;

         Idd = 0.0 ;
         Idd_dVgs = 0.0 ;
         Idd_dVds = 0.0 ;
         Idd_dVbs = 0.0 ;

       } else {

         T1 = - (Q_s0 + Q_sL) ;
         T1_dVgs = - (Q_s0_dVgs + Q_sL_dVgs) ;
         T1_dVds = - (Q_s0_dVds + Q_sL_dVds) ;
         T1_dVbs = - (Q_s0_dVbs + Q_sL_dVbs) ;

         Fn_SL_CP3( Qn_drift , T1 , 0.0, Qn_delta , 2 , T3, T4, T5 )
         Qn_drift_dVgs = T1_dVgs * T3 ;
         Qn_drift_dVds = T1_dVds * T3 ;
         Qn_drift_dVbs = T1_dVbs * T3 ;

         Idd_drift =  beta * Qn_drift / 2.0 * Pds ;
         Idd_diffu = - ( - Q_nL_cur + Q_n0_cur);

         Idd = Idd_drift + Idd_diffu ;
         Idd_dVgs = beta * Qn_drift_dVgs / 2.0 * Pds + beta * Qn_drift / 2.0 * Pds_dVgs
                    - ( - Q_nL_cur_dVgs + Q_n0_cur_dVgs ) ;
         Idd_dVds = beta * Qn_drift_dVds / 2.0 * Pds + beta * Qn_drift / 2.0 * Pds_dVds
                    - ( - Q_nL_cur_dVds + Q_n0_cur_dVds ) ;
         Idd_dVbs = beta * Qn_drift_dVbs / 2.0 * Pds + beta * Qn_drift / 2.0 * Pds_dVbs
                    - ( - Q_nL_cur_dVbs + Q_n0_cur_dVbs ) ;

       } 


       Qiu = - Q_n0_cur ;
       Qiu_dVgs = - Q_n0_cur_dVgs ;
       Qiu_dVds = - Q_n0_cur_dVds ;
       Qiu_dVbs = - Q_n0_cur_dVbs ;

       Lch = Leff ;

      /*-----------------------------------------------------------*
       * Muun : universal mobility.  (CGS unit)
       *-----------------*/

       T2 = here->HSM2_ninv_o_esi / C_m2cm  ;

       T0 = here->HSM2_ninvd ;
       T3 = sqrt(Pds*Pds + model->HSM2_vzadd0) ;
       Pdsz = T3 - sqrt(model->HSM2_vzadd0) ;
       Pdsz_dVbs = Pds_dVbs * Pds / T3  ;
       Pdsz_dVds = Pds_dVds * Pds / T3  ;
       Pdsz_dVgs = Pds_dVgs * Pds / T3  ;

       T4 = 1.0 + ( Pdsz ) * T0 ;
       T4_dVb = ( Pdsz_dVbs ) * T0 ;
       T4_dVd = ( Pdsz_dVds ) * T0 ;
       T4_dVg = ( Pdsz_dVgs ) * T0 ;

       T5     = T2 * Qiu ;
       T5_dVb = T2 * Qiu_dVbs ;
       T5_dVd = T2 * Qiu_dVds ;
       T5_dVg = T2 * Qiu_dVgs ;

       T3     = T5 / T4 ;
       T3_dVb = ( - T4_dVb * T5 + T4 * T5_dVb ) / T4 / T4 ;
       T3_dVd = ( - T4_dVd * T5 + T4 * T5_dVd ) / T4 / T4 ;
       T3_dVg = ( - T4_dVg * T5 + T4 * T5_dVg ) / T4 / T4 ;

       Eeff = T3 ;
       Eeff_dVbs = T3_dVb ;
       Eeff_dVds = T3_dVd ;
       Eeff_dVgs = T3_dVg ;

       T5 = Fn_Pow( Eeff , model->HSM2_mueph0 - 1.0e0 ) ;
       T8 = T5 * Eeff ;
       T7 = Fn_Pow( Eeff , here->HSM2_muesr - 1.0e0 ) ;
       T6 = T7 * Eeff ;

       T9 = C_QE * C_m2cm_p2 ;
       Rns = Qiu / T9 ;

       T1 = 1.0e0 / ( here->HSM2_muecb0 + here->HSM2_muecb1 * Rns / 1.0e11 )
         + here->HSM2_mphn0 * T8 + T6 / pParam->HSM2_muesr1 ;

       Muun = 1.0e0 / T1 ;

       T1 = 1.0e0 / ( T1 * T1 ) ;
       T2 = here->HSM2_muecb0 + here->HSM2_muecb1 * Rns / 1.0e11 ;
       T2 = 1.0e0 / ( T2 * T2 ) ;
       /*  here->HSM2_mphn1 = here->HSM2_mphn0 * model->HSM2_mueph0 */
       T3 = here->HSM2_mphn1 * T5 ;
       T4 = here->HSM2_muesr * T7 / pParam->HSM2_muesr1 ;
       T5 = - 1.0e-11 * here->HSM2_muecb1 / C_QE * T2 / C_m2cm_p2 ;
       Muun_dVbs = - ( T5 * Qiu_dVbs
                    + Eeff_dVbs * T3 + Eeff_dVbs * T4 ) * T1 ;
       Muun_dVds = - ( T5 * Qiu_dVds
                    + Eeff_dVds * T3 + Eeff_dVds * T4 ) * T1 ;
       Muun_dVgs = - ( T5 * Qiu_dVgs
                    + Eeff_dVgs * T3 + Eeff_dVgs * T4 ) * T1 ;

      /*  Change to MKS unit */
       Muun /=C_m2cm_p2 ;
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

       TY = Idd * T1 ;
       TY_dVbs = Idd_dVbs * T1 + Idd * T1_dVb ;
       TY_dVds = Idd_dVds * T1 + Idd * T1_dVd ;
       TY_dVgs = Idd_dVgs * T1 + Idd * T1_dVg ;

       T2 = 0.2 * Vmax / Muun ;
       T3 = - T2 / Muun ;
       T2_dVb = T3 * Muun_dVbs ;
       T2_dVd = T3 * Muun_dVds ;
       T2_dVg = T3 * Muun_dVgs ;

       Ey = sqrt( TY * TY + T2 * T2 ) ;
       T4 = 1.0 / Ey ;
       Ey_dVbs = T4 * ( TY * TY_dVbs + T2 * T2_dVb ) ;
       Ey_dVds = T4 * ( TY * TY_dVds + T2 * T2_dVd ) ;
       Ey_dVgs = T4 * ( TY * TY_dVgs + T2 * T2_dVg ) ;

       Em = Muun * Ey ;
       Em_dVbs = Muun_dVbs * Ey + Muun * Ey_dVbs ;
       Em_dVds = Muun_dVds * Ey + Muun * Ey_dVds ;
       Em_dVgs = Muun_dVgs * Ey + Muun * Ey_dVgs ;

       T1  = Em / Vmax ; 
    
       Ey_suf = Ey ;
       Ey_suf_dVgs = Ey_dVgs ;
       Ey_suf_dVds = Ey_dVds ;
       Ey_suf_dVbs = Ey_dVbs ;

      /* note: model->HSM2_bb = 2 (electron) ;1 (hole) */
       if ( 1.0e0 - epsm10 <= model->HSM2_bb && model->HSM2_bb <= 1.0e0 + epsm10 ) {
         T3 = 1.0e0 ;
       } else if ( 2.0e0 - epsm10 <= model->HSM2_bb && model->HSM2_bb <= 2.0e0 + epsm10 ) {
         T3 = T1 ;
       } else {
         T3 = Fn_Pow( T1 , model->HSM2_bb - 1.0e0 ) ;
       }
       T2 = T1 * T3 ;
       T4 = 1.0e0 + T2 ;

       if ( 1.0e0 - epsm10 <= model->HSM2_bb && model->HSM2_bb <= 1.0e0 + epsm10 ) {
         T5 = 1.0 / T4 ;
         T6 = T5 / T4 ; 
       } else if ( 2.0e0 - epsm10 <= model->HSM2_bb && model->HSM2_bb <= 2.0e0 + epsm10 ) {
         T5 = 1.0 / sqrt( T4 ) ;
         T6 = T5 / T4 ;
       } else {
         T6 = Fn_Pow( T4 , ( - 1.0e0 / model->HSM2_bb - 1.0e0 ) ) ;
         T5 = T4 * T6 ;
       }

       T7 = Muun / Vmax * T6 * T3 ;

       Mu = Muun * T5 ;

       Mu_dVbs = Muun_dVbs * T5 - T7 * Em_dVbs ;
       Mu_dVds = Muun_dVds * T5 - T7 * Em_dVds ;
       Mu_dVgs = Muun_dVgs * T5 - T7 * Em_dVgs ;

     //-----------------------------------------------------------*
     //*  resistor region current.  (CGS unit)
     //*-----------------//

       if( Vdsorg > 1e-3 ) {

         T2 = q_Ndepm_esi / ( Cox * Cox ) ;
         T4 = - 2.0e0 * T2 / Cox ;
         T2_dVb = T4 * Cox_dVb ;
         T2_dVd = T4 * Cox_dVd ;
         T2_dVg = T4 * Cox_dVg ;

         T0 = Vgp + here->HSM2_depvdsef1 - beta_inv - Vbsz ;
         T0_dVg = Vgp_dVgs ;
         T0_dVd = Vgp_dVds - Vbsz_dVds ;
         T0_dVb = Vgp_dVbs - Vbsz_dVbs ;

         T4 = 1.0e0 + 2.0e0 / T2 * T0 ;
         T4_dVg = 2.0 / T2 * T0_dVg - 2.0 / T2 / T2 * T0 * T2_dVg ;
         T4_dVd = 2.0 / T2 * T0_dVd - 2.0 / T2 / T2 * T0 * T2_dVd ;
         T4_dVb = 2.0 / T2 * T0_dVb - 2.0 / T2 / T2 * T0 * T2_dVb ;

         Fn_SL_CP( T9 , T4 , 0 , DEPQFN_dlt, 2 , T0 )
         T9_dVg = T4_dVg * T0 ;
         T9_dVd = T4_dVd * T0 ;
         T9_dVb = T4_dVb * T0 ;

         T9 = T9 + small ;
         T3 = sqrt( T9 ) ;
         T3_dVg = 0.5 / T3 * T9_dVg  ;
         T3_dVd = 0.5 / T3 * T9_dVd  ;
         T3_dVb = 0.5 / T3 * T9_dVb  ;

         T10 = Vgp + here->HSM2_depvdsef1 + T2 * ( 1.0e0 - T3 ) ;
         T10 = T10 * here->HSM2_depvdsef2 ;
         T10_dVb = (Vgp_dVbs + T2_dVb * ( 1.0e0 - T3 ) - T2 * T3_dVb)
                   * here->HSM2_depvdsef2 ;
         T10_dVd = (Vgp_dVds + T2_dVd * ( 1.0e0 - T3 ) - T2 * T3_dVd)
                   * here->HSM2_depvdsef2 ; 
         T10_dVg = (Vgp_dVgs + T2_dVg * ( 1.0e0 - T3 ) - T2 * T3_dVg) 
                   * here->HSM2_depvdsef2 ;

         Fn_SL_CP( T10 , T10 , here->HSM2_depleak, 1.0, 4 , T0 )
         T10 = T10 + epsm10 ;
         T10_dVb *=T0 ;
         T10_dVd *=T0 ;
         T10_dVg *=T0 ;

         T1 = Vdsorg / T10 ;
         T2 = Fn_Pow( T1 , here->HSM2_ddlt - 1.0e0 ) ;
         T7 = T2 * T1 ;
         T0 = here->HSM2_ddlt * T2 / ( T10 * T10 ) ;
         T7_dVb = T0 * ( - Vdsorg * T10_dVb ) ;
         T7_dVd = T0 * ( T10 - Vdsorg * T10_dVd ) ;
         T7_dVg = T0 * ( - Vdsorg * T10_dVg ) ;

         T3 = 1.0 + T7 ; 
         T4 = Fn_Pow( T3 , 1.0 / here->HSM2_ddlt - 1.0 ) ;
         T6 = T4 * T3 ;
         T0 = T4 / here->HSM2_ddlt ;
         T6_dVb = T0 * T7_dVb ;
         T6_dVd = T0 * T7_dVd ;
         T6_dVg = T0 * T7_dVg ;

         Vdseff0 = Vdsorg / T6 ;
         T0 = 1.0 / ( T6 * T6 ) ; 
         Vdseff0_dVbs = - Vdsorg * T6_dVb * T0 ;
         Vdseff0_dVds = ( T6 - Vdsorg * T6_dVd ) * T0 ;
         Vdseff0_dVgs = - Vdsorg * T6_dVg * T0 ;

       } else {

         Vdseff0 = Vdsorg ;
         Vdseff0_dVgs = 0.0 ;
         Vdseff0_dVds = 1.0 ;
         Vdseff0_dVbs = 0.0 ;

       }
 
       T0 = here->HSM2_ninvd ; 

       T4 = 1.0 + ( Pdsz ) * T0 ;
       T4_dVb = ( Pdsz_dVbs ) * T0 ;
       T4_dVd = ( Pdsz_dVds ) * T0 ;
       T4_dVg = ( Pdsz_dVgs ) * T0 ;

       Qiu = - Qn_res0 ;
       Qiu_dVgs = - Qn_res0_dVgs ;
       Qiu_dVds = - Qn_res0_dVds ;
       Qiu_dVbs = - Qn_res0_dVbs ;

       T5     = Qiu ;
       T5_dVb = Qiu_dVbs ;
       T5_dVd = Qiu_dVds ;
       T5_dVg = Qiu_dVgs ;

       T3     = T5 / T4 ;
       T3_dVb = ( - T4_dVb * T5 + T4 * T5_dVb ) / T4 / T4 ;
       T3_dVd = ( - T4_dVd * T5 + T4 * T5_dVd ) / T4 / T4 ;
       T3_dVg = ( - T4_dVg * T5 + T4 * T5_dVg ) / T4 / T4 ;

       Eeff = T3 ;
       Eeff_dVbs = T3_dVb ;
       Eeff_dVds = T3_dVd ;
       Eeff_dVgs = T3_dVg ;

       T5 = Fn_Pow( Eeff , model->HSM2_depmueph0 - 1.0e0 ) ;
       T8 = T5 * Eeff ;

       T9 = C_QE * C_m2cm_p2 ;
       Rns = Qiu / T9 ;

       T1 = 1.0e0 / ( here->HSM2_depmue0 + here->HSM2_depmue1 * Rns / 1.0e11 )
         + here->HSM2_depmphn0 * T8  ;

       Muun = 1.0e0 / T1 ;

       T1 = 1.0e0 / ( T1 * T1 ) ;
       T2 = here->HSM2_depmue0 + here->HSM2_depmue1 * Rns / 1.0e11 ;
       T2 = 1.0e0 / ( T2 * T2 ) ;
       T3 = here->HSM2_depmphn1 * T5 ;
       T5 = - 1.0e-11 * here->HSM2_depmue1 / C_QE * T2 / C_m2cm_p2 ;
       Muun_dVbs = - ( T5 * Qiu_dVbs
                    + Eeff_dVbs * T3 ) * T1 ;
       Muun_dVds = - ( T5 * Qiu_dVds
                    + Eeff_dVds * T3 ) * T1 ;
       Muun_dVgs = - ( T5 * Qiu_dVgs
                    + Eeff_dVgs * T3 ) * T1 ;

      /*  Change to MKS unit */
       Muun /=C_m2cm_p2 ;
       Muun_dVbs /=C_m2cm_p2 ;
       Muun_dVds /=C_m2cm_p2 ;
       Muun_dVgs /=C_m2cm_p2 ;

       Edri = Vdseff0 / Lch ;
       Edri_dVgs = Vdseff0_dVgs / Lch ;
       Edri_dVds = Vdseff0_dVds / Lch ;
       Edri_dVbs = Vdseff0_dVbs / Lch ;

       T1 = Muun * Edri / here->HSM2_depvmax ;
       T1_dVgs = (Muun_dVgs * Edri + Muun * Edri_dVgs) / here->HSM2_depvmax ;
       T1_dVds = (Muun_dVds * Edri + Muun * Edri_dVds) / here->HSM2_depvmax ;
       T1_dVbs = (Muun_dVbs * Edri + Muun * Edri_dVbs) / here->HSM2_depvmax ;
       
       T1 = T1 + small ;
       T2 = Fn_Pow(T1,model->HSM2_depbb) ;
       T2_dVgs = model->HSM2_depbb * T1_dVgs / T1 * T2 ;
       T2_dVds = model->HSM2_depbb * T1_dVds / T1 * T2 ;
       T2_dVbs = model->HSM2_depbb * T1_dVbs / T1 * T2 ;

       T3 = 1.0 + T2 ;
       T4 = Fn_Pow(T3,1.0 / model->HSM2_depbb) ;
       T4_dVgs = 1.0 / model->HSM2_depbb * T2_dVgs / T3 * T4 ;
       T4_dVds = 1.0 / model->HSM2_depbb * T2_dVds / T3 * T4 ;
       T4_dVbs = 1.0 / model->HSM2_depbb * T2_dVbs / T3 * T4 ;

       Mu_res = Muun / T4 ;
       Mu_res_dVgs = Muun_dVgs / T4 - Muun / T4 / T4 * T4_dVgs ;
       Mu_res_dVds = Muun_dVds / T4 - Muun / T4 / T4 * T4_dVds ;
       Mu_res_dVbs = Muun_dVbs / T4 - Muun / T4 / T4 * T4_dVbs ;

       Ids_res = here->HSM2_weff_nf * ( - Qn_res0) * Mu_res * Edri ;
       Ids_res_dVgs = here->HSM2_weff_nf * ( - Qn_res0_dVgs * Mu_res * Edri
               - Qn_res0 * Mu_res_dVgs * Edri - Qn_res0 * Mu_res * Edri_dVgs) ;
       Ids_res_dVds = here->HSM2_weff_nf * ( - Qn_res0_dVds * Mu_res * Edri
               - Qn_res0 * Mu_res_dVds * Edri - Qn_res0 * Mu_res * Edri_dVds) ;
       Ids_res_dVbs = here->HSM2_weff_nf * ( - Qn_res0_dVbs * Mu_res * Edri
               - Qn_res0 * Mu_res_dVbs * Edri - Qn_res0 * Mu_res * Edri_dVbs) ;


     //-----------------------------------------------------------*
     //*  back region universal mobility.  (CGS unit)
     //*-----------------//

       T0 = here->HSM2_ninvd ;

       T4 = 1.0 + ( Pdsz ) * T0 ;
       T4_dVb = ( Pdsz_dVbs ) * T0 ;
       T4_dVd = ( Pdsz_dVds ) * T0 ;
       T4_dVg = ( Pdsz_dVgs ) * T0 ;

       Qiu = - Qn_bac0 ;
       Qiu_dVgs = - Qn_bac0_dVgs ;
       Qiu_dVds = - Qn_bac0_dVds ;
       Qiu_dVbs = - Qn_bac0_dVbs ;

       T5     = Qiu ;
       T5_dVb = Qiu_dVbs ;
       T5_dVd = Qiu_dVds ;
       T5_dVg = Qiu_dVgs ;

       T3     = T5 / T4 ;
       T3_dVb = ( - T4_dVb * T5 + T4 * T5_dVb ) / T4 / T4 ;
       T3_dVd = ( - T4_dVd * T5 + T4 * T5_dVd ) / T4 / T4 ;
       T3_dVg = ( - T4_dVg * T5 + T4 * T5_dVg ) / T4 / T4 ;

       Eeff = T3 ;
       Eeff_dVbs = T3_dVb ;
       Eeff_dVds = T3_dVd ;
       Eeff_dVgs = T3_dVg ;

       T5 = Fn_Pow( Eeff , model->HSM2_depmueph0 - 1.0e0 ) ;
       T8 = T5 * Eeff ;

       T9 = C_QE * C_m2cm_p2 ;
       Rns = Qiu / T9 ;

       T1 = 1.0e0 / ( here->HSM2_depmueback0 + here->HSM2_depmueback1 * Rns / 1.0e11 )
         + here->HSM2_depmphn0 * T8  ;

       Muun = 1.0e0 / T1 ;

       T1 = 1.0e0 / ( T1 * T1 ) ;
       T2 = here->HSM2_depmueback0 + here->HSM2_depmueback1 * Rns / 1.0e11 ;
       T2 = 1.0e0 / ( T2 * T2 ) ;
       T3 = here->HSM2_depmphn1 * T5 ;
       T5 = - 1.0e-11 * here->HSM2_depmueback1 / C_QE * T2 / C_m2cm_p2 ;
       Muun_dVbs = - ( T5 * Qiu_dVbs 
                    + Eeff_dVbs * T3 ) * T1 ;
       Muun_dVds = - ( T5 * Qiu_dVds 
                    + Eeff_dVds * T3 ) * T1 ;
       Muun_dVgs = - ( T5 * Qiu_dVgs 
                    + Eeff_dVgs * T3 ) * T1 ;

      /*  Change to MKS unit */
       Muun /=C_m2cm_p2 ;
       Muun_dVbs /=C_m2cm_p2 ;
       Muun_dVds /=C_m2cm_p2 ;
       Muun_dVgs /=C_m2cm_p2 ;

       Edri = Vdseff0 / Lch ;
       Edri_dVgs = Vdseff0_dVgs / Lch ;
       Edri_dVds = Vdseff0_dVds / Lch ;
       Edri_dVbs = Vdseff0_dVbs / Lch ;

       T1 = Muun * Edri / here->HSM2_depvmax ;
       T1_dVgs = (Muun_dVgs * Edri + Muun * Edri_dVgs) / here->HSM2_depvmax ;
       T1_dVds = (Muun_dVds * Edri + Muun * Edri_dVds) / here->HSM2_depvmax ;
       T1_dVbs = (Muun_dVbs * Edri + Muun * Edri_dVbs) / here->HSM2_depvmax ;

       T1 = T1 + small ;
       T2 = Fn_Pow(T1,model->HSM2_depbb) ;
       T2_dVgs = model->HSM2_depbb * T1_dVgs / T1 * T2 ;
       T2_dVds = model->HSM2_depbb * T1_dVds / T1 * T2 ;
       T2_dVbs = model->HSM2_depbb * T1_dVbs / T1 * T2 ;

       T3 = 1.0 + T2 ;
       T4 = Fn_Pow(T3,1.0 / model->HSM2_depbb) ;
       T4_dVgs = 1.0 / model->HSM2_depbb * T2_dVgs / T3 * T4 ;
       T4_dVds = 1.0 / model->HSM2_depbb * T2_dVds / T3 * T4 ;
       T4_dVbs = 1.0 / model->HSM2_depbb * T2_dVbs / T3 * T4 ;


       Mu_bac = Muun / T4 ;
       Mu_bac_dVgs = Muun_dVgs / T4 - Muun / T4 / T4 * T4_dVgs ;
       Mu_bac_dVds = Muun_dVds / T4 - Muun / T4 / T4 * T4_dVds ;
       Mu_bac_dVbs = Muun_dVbs / T4 - Muun / T4 / T4 * T4_dVbs ;

       Ids_bac = here->HSM2_weff_nf * ( - Qn_bac0) * Mu_bac * Edri ;
       Ids_bac_dVgs = here->HSM2_weff_nf * ( - Qn_bac0_dVgs * Mu_bac * Edri
               - Qn_bac0 * Mu_bac_dVgs * Edri - Qn_bac0 * Mu_bac * Edri_dVgs) ;
       Ids_bac_dVds = here->HSM2_weff_nf * ( - Qn_bac0_dVds * Mu_bac * Edri
               - Qn_bac0 * Mu_bac_dVds * Edri - Qn_bac0 * Mu_bac * Edri_dVds) ;
       Ids_bac_dVbs = here->HSM2_weff_nf * ( - Qn_bac0_dVbs * Mu_bac * Edri
               - Qn_bac0 * Mu_bac_dVbs * Edri - Qn_bac0 * Mu_bac * Edri_dVbs) ;


     /*-----------------------------------------------------------*
      * Ids: channel current.
      *-----------------*/
       betaWL = here->HSM2_weff_nf * beta_inv / Lch ;
       T1 = - betaWL / Lch ;

       Ids0 = betaWL * Idd * Mu + Ids_res + Ids_bac ;

       Ids0_dVgs = betaWL * ( Idd_dVgs * Mu + Idd * Mu_dVgs )
                   + Ids_res_dVgs + Ids_bac_dVgs ;
       Ids0_dVds = betaWL * ( Idd_dVds * Mu + Idd * Mu_dVds )
                   + Ids_res_dVds + Ids_bac_dVds ;
       Ids0_dVbs = betaWL * ( Idd_dVbs * Mu + Idd * Mu_dVbs )
                   + Ids_res_dVbs + Ids_bac_dVbs ;


       // Vdseff //

       Vds = Vdsorg;

      /*-----------------------------------------------------------*
       * Adding parasitic components to the channel current.
       *-----------------*/
       if( model->HSM2_ptl != 0 ){
         T1 =  0.5 * ( Vds - Pds ) ;
         Fn_SymAdd( T6 , T1 , 0.01 , T2 ) ;
         T2 = T2 * 0.5 ;
         T6_dVb = T2 * ( - Pds_dVbs ) ;
         T6_dVd = T2 * ( 1.0 - Pds_dVds ) ;
         T6_dVg = T2 * ( - Pds_dVgs ) ;

         T1     = 1.1 - ( phi_s0_DEP + T6 );
         T1_dVb = - ( phi_s0_DEP_dVbs + T6_dVb );
         T1_dVd = - ( phi_s0_DEP_dVds + T6_dVd );
         T1_dVg = - ( phi_s0_DEP_dVgs + T6_dVg );
 
         Fn_SZ( T2 , T1 , 0.05 , T0 ) ;
         T2 = T2 + small ;
         T2_dVb = T1_dVb * T0 ;
         T2_dVd = T1_dVd * T0 ;
         T2_dVg = T1_dVg * T0 ;

         T0 = beta * here->HSM2_ptl0 ;
         T3 = Cox * T0 ;
         T3_dVb = Cox_dVb * T0 ;
         T3_dVd = Cox_dVd * T0 ;
         T3_dVg = Cox_dVg * T0 ;
         T0 = pow( T2 , model->HSM2_ptp ) ;
         T9     = T3 * T0 ;
         T9_dVb = T3 * model->HSM2_ptp * T0 / T2 * T2_dVb + T3_dVb * T0 ;
         T9_dVd = T3 * model->HSM2_ptp * T0 / T2 * T2_dVd + T3_dVd * T0 ;
         T9_dVg = T3 * model->HSM2_ptp * T0 / T2 * T2_dVg + T3_dVg * T0 ;


         T4 = 1.0 + Vdsz * model->HSM2_pt2 ;
         T4_dVb = Vdsz_dVbs * model->HSM2_pt2 ;
         T4_dVd = Vdsz_dVds * model->HSM2_pt2 ;
         T4_dVg = 0.0 ;

         T0 = here->HSM2_pt40 ;
         T5 = phi_s0_DEP + T6 - Vbsz ;
         T5_dVb = phi_s0_DEP_dVbs + T6_dVb - Vbsz_dVbs ;
         T5_dVd = phi_s0_DEP_dVds + T6_dVd - Vbsz_dVds ;
         T5_dVg = phi_s0_DEP_dVgs + T6_dVg ;
         T4 = T4 + Vdsz * T0 * T5 ;
         T4_dVb = T4_dVb + Vdsz * T0 * T5_dVb + Vdsz_dVbs * T0 * T5 ;
         T4_dVd = T4_dVd + Vdsz * T0 * T5_dVd + Vdsz_dVds * T0 * T5 ;
         T4_dVg = T4_dVg + Vdsz * T0 * T5_dVg ;
         T6     = T9 * T4 ;
         T9_dVb = T9_dVb * T4 + T9 * T4_dVb ;
         T9_dVd = T9_dVd * T4 + T9 * T4_dVd ;
         T9_dVg = T9_dVg * T4 + T9 * T4_dVg ;
         T9     = T6 ;

       } else {
         T9 = 0.0 ;
         T9_dVb = 0.0 ;
         T9_dVd = 0.0 ;
         T9_dVg = 0.0 ;
       }

       if( model->HSM2_gdl != 0 ){
         T1 = beta * here->HSM2_gdl0 ;
         T2 = Cox * T1 ;
         T2_dVb = Cox_dVb * T1 ;
         T2_dVd = Cox_dVd * T1 ;
         T2_dVg = Cox_dVg * T1 ;
         T8     = T2 * Vdsz ;
         T8_dVb = T2_dVb * Vdsz + T2 * Vdsz_dVbs ;
         T8_dVd = T2_dVd * Vdsz + T2 * Vdsz_dVds ;
         T8_dVg = T2_dVg * Vdsz ;
       } else {
         T8 = 0.0 ;
         T8_dVb = 0.0 ;
         T8_dVd = 0.0 ;
         T8_dVg = 0.0 ;
       }

       if ( ( T9 + T8 ) > 0.0 ) {
         Idd1 = Pds * ( T9 + T8 ) ;
         Idd1_dVbs = Pds_dVbs * ( T9 + T8 ) + Pds * ( T9_dVb + T8_dVb ) ;
         Idd1_dVds = Pds_dVds * ( T9 + T8 ) + Pds * ( T9_dVd + T8_dVd ) ;
         Idd1_dVgs = Pds_dVgs * ( T9 + T8 ) + Pds * ( T9_dVg + T8_dVg ) ;

         Ids0 = Ids0 +betaWL * Idd1 * Mu ;
         T1 = betaWL * Idd1 ;
         T2 = Idd1 * Mu ;
         T3 = Mu * betaWL ;
         Ids0_dVbs +=T3 * Idd1_dVbs + T1 * Mu_dVbs + T2 * betaWL_dVbs ;
         Ids0_dVds +=T3 * Idd1_dVds + T1 * Mu_dVds + T2 * betaWL_dVds ;
         Ids0_dVgs +=T3 * Idd1_dVgs + T1 * Mu_dVgs + T2 * betaWL_dVgs ;
       }

       /* Ids calculation */
      if ( flg_rsrd == 2 ) {
         Rd  = here->HSM2_rd ;
         T0 = Rd * Ids0 ;
         T1 = Vds + small ;
         T2 = 1.0 / T1 ;
         T3 = 1.0 + T0 * T2 ;
         T3_dVb =   Rd * Ids0_dVbs             * T2 ;
         T3_dVd = ( Rd * Ids0_dVds * T1 - T0 ) * T2 * T2 ;
         T3_dVg =   Rd * Ids0_dVgs             * T2 ;
         T4 = 1.0 / T3 ;
         Ids = Ids0 * T4 ;
         T5 = T4 * T4 ;
         Ids_dVbs = ( Ids0_dVbs * T3 - Ids0 * T3_dVb ) * T5 ;
         Ids_dVds = ( Ids0_dVds * T3 - Ids0 * T3_dVd ) * T5 ;
         Ids_dVgs = ( Ids0_dVgs * T3 - Ids0 * T3_dVg ) * T5 ;
       } else {
         Ids = Ids0 ;
         Ids_dVbs = Ids0_dVbs ;
         Ids_dVds = Ids0_dVds ;
         Ids_dVgs = Ids0_dVgs ;
       }

      /* charge calculation */
      /*---------------------------------------------------*
       * Qbu : - Qb in unit area.
       *-----------------*/
       Qbu = - 0.5 * (Q_sub0 + Q_subL + Q_sub0_dep + Q_subL_dep ) ;
       Qbu_dVgs = - 0.5 * ( Q_sub0_dVgs + Q_subL_dVgs + Q_sub0_dep_dVgs + Q_subL_dep_dVgs ) ;
       Qbu_dVds = - 0.5 * ( Q_sub0_dVds + Q_subL_dVds + Q_sub0_dep_dVds + Q_subL_dep_dVds ) ;
       Qbu_dVbs = - 0.5 * ( Q_sub0_dVbs + Q_subL_dVbs + Q_sub0_dep_dVbs + Q_subL_dep_dVbs ) ;

       Qiu = - 0.5 * (Q_n0 + Q_nL + Q_s0_dep + Q_sL_dep + Q_b0_dep + Q_bL_dep) ;
       Qiu_dVgs = - 0.5 * ( Q_n0_dVgs + Q_nL_dVgs + Q_s0_dep_dVgs + Q_sL_dep_dVgs + Q_b0_dep_dVgs + Q_bL_dep_dVgs ) ;
       Qiu_dVds = - 0.5 * ( Q_n0_dVds + Q_nL_dVds + Q_s0_dep_dVds + Q_sL_dep_dVds + Q_b0_dep_dVds + Q_bL_dep_dVds ) ;
       Qiu_dVbs = - 0.5 * ( Q_n0_dVbs + Q_nL_dVbs + Q_s0_dep_dVbs + Q_sL_dep_dVbs + Q_b0_dep_dVbs + Q_bL_dep_dVbs ) ;

       Qdrat = 0.5;
       Qdrat_dVgs = 0.0 ;
       Qdrat_dVds = 0.0 ;
       Qdrat_dVbs = 0.0 ;

      /*-------------------------------------------------*
       * set flg_noqi , Qn0 , Ey for noise calc.
       *-----------------*/
       Qiu_noi = - 0.5 * (Q_n0 + Q_nL ) ;
       Qn0 = - Q_n0 ;
       Qn0_dVgs = - Q_n0_dVgs ;
       Qn0_dVds = - Q_n0_dVds ;
       Qn0_dVbs = - Q_n0_dVbs ;

       Ey = Ey_suf ;
       Ey_dVbs = Ey_suf_dVbs ;
       Ey_dVds = Ey_suf_dVds ;
       Ey_dVgs = Ey_suf_dVgs ;

       if( Qn0 < small ) { flg_noqi = 1; }

} /* End of hsmhveval_dep */


