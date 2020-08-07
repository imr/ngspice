
/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * Original File: b3ld.c of BSIM3v3.2.4
 * Author: 1991 JianHui Huang and Min-Chie Jeng.
 * Modified by Mansun Chan (1995).
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001 Xuemei Xi
 * Modified by Xuemei Xi, 10/05, 12/21, 2001.
 * Modified by Paolo Nenzi 2002 and Dietmar Warning 2003
 * Modified by Florian Ballenegger 2020 for SIMD version generation
 **********/

 /**********
 * Modified 2020 by Florian Ballenegger, Anamosic Ballenegger Design
 * Distributed under the same license terms as the original code,
 * see file "B3TERMS_OF_USE"
 **********/
#ifndef USE_OMP
#pragma message "Warning: simd configured for USE_OMP but compiled without - use anyway"
#endif
#ifdef NOBYPASS
#pragma message "Warning: simd configured without NOBYPASS but compiled with - ignored"
#endif
#ifndef OMP_EFFMEM
#pragma message "Warning: simd configured for OMP_EFFMEM but compiled without - use anyway"
#endif
#ifndef NEWCONV
#pragma message "Warning: simd configured for NEWCONV but compiled without - use anyway"
#endif
#ifdef PREDICTOR
#pragma message "Warning: simd configured without PREDICTOR but compiled with - ignored"
#endif

{
  Vec8d SourceSatCurrent;
  Vec8d DrainSatCurrent;
  double ag0;
  Vec8d qgd;
  Vec8d qgs;
  Vec8d qgb;
  double von;
  Vec8d cbhat;
  Vec8d VgstNVt;
  Vec8d ExpVgst;
  Vec8d cdrain;
  Vec8d cdhat;
  Vec8d cdreq;
  Vec8d ceqbd;
  Vec8d ceqbs;
  Vec8d ceqqb;
  Vec8d ceqqd;
  Vec8d ceqqg;
  double ceq;
  double geq;
  Vec8d czbd;
  Vec8d czbdsw;
  Vec8d czbdswg;
  Vec8d czbs;
  Vec8d czbssw;
  Vec8d czbsswg;
  Vec8d evbd;
  Vec8d evbs;
  Vec8d arg;
  Vec8d sarg;
  double delvbd;
  double delvbs;
  double delvds;
  double delvgd;
  double delvgs;
  Vec8d Vfbeff;
  Vec8d dVfbeff_dVg;
  Vec8d dVfbeff_dVb;
  Vec8d V3;
  Vec8d V4;
  Vec8d gcbdb;
  Vec8d gcbgb;
  Vec8d gcbsb;
  Vec8d gcddb;
  Vec8d gcdgb;
  Vec8d gcdsb;
  Vec8d gcgdb;
  Vec8d gcggb;
  Vec8d gcgsb;
  Vec8d gcsdb;
  Vec8d gcsgb;
  Vec8d gcssb;
  double MJ;
  double MJSW;
  double MJSWG;
  Vec8d vbd;
  Vec8d vbs;
  Vec8d vds;
  Vec8d vgb;
  Vec8d vgd;
  Vec8d vgs;
  double vgdo;
  double xfact;
  Vec8d qgate = (Vec8d ){0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Vec8d qbulk = (Vec8d ){0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Vec8d qdrn = (Vec8d ){0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Vec8d qsrc;
  Vec8d qinoi;
  Vec8d cqgate;
  Vec8d cqbulk;
  Vec8d cqdrn;
  Vec8d Vds;
  Vec8d Vgs;
  Vec8d Vbs;
  Vec8d Gmbs;
  Vec8d FwdSum;
  Vec8d RevSum;
  Vec8d Vgs_eff;
  Vec8d Vfb;
  Vec8d Phis;
  Vec8d dPhis_dVb;
  Vec8d sqrtPhis;
  Vec8d dsqrtPhis_dVb;
  Vec8d Vth;
  Vec8d dVth_dVb;
  Vec8d dVth_dVd;
  Vec8d Vgst;
  Vec8d dVgst_dVg;
  Vec8d dVgst_dVb;
  Vec8d dVgs_eff_dVg;
  double Nvtm;
  double Vtm;
  Vec8d n;
  Vec8d dn_dVb;
  Vec8d dn_dVd;
  double voffcv;
  Vec8d noff;
  Vec8d dnoff_dVd;
  Vec8d dnoff_dVb;
  Vec8d ExpArg;
  double V0;
  Vec8d CoxWLcen;
  Vec8d QovCox;
  double LINK;
  Vec8d DeltaPhi;
  Vec8d dDeltaPhi_dVg;
  Vec8d VgDP;
  Vec8d dVgDP_dVg;
  double Cox;
  double Tox;
  Vec8d Tcen;
  Vec8d dTcen_dVg;
  Vec8d dTcen_dVd;
  Vec8d dTcen_dVb;
  Vec8d Ccen;
  Vec8d Coxeff;
  Vec8d dCoxeff_dVg;
  Vec8d dCoxeff_dVd;
  Vec8d dCoxeff_dVb;
  Vec8d Denomi;
  Vec8d dDenomi_dVg;
  Vec8d dDenomi_dVd;
  Vec8d dDenomi_dVb;
  Vec8d ueff;
  Vec8d dueff_dVg;
  Vec8d dueff_dVd;
  Vec8d dueff_dVb;
  Vec8d Esat;
  Vec8d Vdsat;
  Vec8d EsatL;
  Vec8d dEsatL_dVg;
  Vec8d dEsatL_dVd;
  Vec8d dEsatL_dVb;
  Vec8d dVdsat_dVg;
  Vec8d dVdsat_dVb;
  Vec8d dVdsat_dVd;
  Vec8d Vasat;
  Vec8d dAlphaz_dVg;
  Vec8d dAlphaz_dVb;
  Vec8d dVasat_dVg;
  Vec8d dVasat_dVb;
  Vec8d dVasat_dVd;
  Vec8d Va;
  Vec8d dVa_dVd;
  Vec8d dVa_dVg;
  Vec8d dVa_dVb;
  Vec8d Vbseff;
  Vec8d dVbseff_dVb;
  Vec8d VbseffCV;
  Vec8d dVbseffCV_dVb;
  Vec8d Arg1;
  Vec8d One_Third_CoxWL;
  Vec8d Two_Third_CoxWL;
  Vec8d Alphaz;
  double CoxWL;
  Vec8d T0;
  Vec8d dT0_dVg;
  Vec8d dT0_dVd;
  Vec8d dT0_dVb;
  Vec8d T1;
  Vec8d dT1_dVg;
  Vec8d dT1_dVd;
  Vec8d dT1_dVb;
  Vec8d T2;
  Vec8d dT2_dVg;
  Vec8d dT2_dVd;
  Vec8d dT2_dVb;
  Vec8d T3;
  Vec8d dT3_dVg;
  Vec8d dT3_dVd;
  Vec8d dT3_dVb;
  Vec8d T4;
  Vec8d T5;
  Vec8d T6;
  Vec8d T7;
  Vec8d T8;
  Vec8d T9;
  Vec8d T10;
  Vec8d T11;
  Vec8d T12;
  Vec8d tmp;
  Vec8d Abulk;
  Vec8d dAbulk_dVb;
  Vec8d Abulk0;
  Vec8d dAbulk0_dVb;
  double tmpuni;
  Vec8d VACLM;
  Vec8d dVACLM_dVg;
  Vec8d dVACLM_dVd;
  Vec8d dVACLM_dVb;
  Vec8d VADIBL;
  Vec8d dVADIBL_dVg;
  Vec8d dVADIBL_dVd;
  Vec8d dVADIBL_dVb;
  Vec8d Xdep;
  Vec8d dXdep_dVb;
  Vec8d lt1;
  Vec8d dlt1_dVb;
  Vec8d ltw;
  Vec8d dltw_dVb;
  Vec8d Delt_vth;
  Vec8d dDelt_vth_dVb;
  Vec8d Theta0;
  Vec8d dTheta0_dVb;
  double TempRatio;
  Vec8d tmp1;
  Vec8d tmp2;
  Vec8d tmp3;
  Vec8d tmp4;
  Vec8d DIBL_Sft;
  Vec8d dDIBL_Sft_dVd;
  Vec8d Lambda;
  Vec8d dLambda_dVg;
  double Idtot;
  double Ibtot;
  double tempv;
  double a1;
  double ScalingFactor;
  Vec8d Vgsteff;
  Vec8d dVgsteff_dVg;
  Vec8d dVgsteff_dVd;
  Vec8d dVgsteff_dVb;
  Vec8d Vdseff;
  Vec8d dVdseff_dVg;
  Vec8d dVdseff_dVd;
  Vec8d dVdseff_dVb;
  Vec8d VdseffCV;
  Vec8d dVdseffCV_dVg;
  Vec8d dVdseffCV_dVd;
  Vec8d dVdseffCV_dVb;
  Vec8d diffVds;
  Vec8d dAbulk_dVg;
  Vec8d beta;
  Vec8d dbeta_dVg;
  Vec8d dbeta_dVd;
  Vec8d dbeta_dVb;
  Vec8d gche;
  Vec8d dgche_dVg;
  Vec8d dgche_dVd;
  Vec8d dgche_dVb;
  Vec8d fgche1;
  Vec8d dfgche1_dVg;
  Vec8d dfgche1_dVd;
  Vec8d dfgche1_dVb;
  Vec8d fgche2;
  Vec8d dfgche2_dVg;
  Vec8d dfgche2_dVd;
  Vec8d dfgche2_dVb;
  Vec8d Idl;
  Vec8d dIdl_dVg;
  Vec8d dIdl_dVd;
  Vec8d dIdl_dVb;
  Vec8d Idsa;
  Vec8d dIdsa_dVg;
  Vec8d dIdsa_dVd;
  Vec8d dIdsa_dVb;
  Vec8d Ids;
  Vec8d Gm;
  Vec8d Gds;
  Vec8d Gmb;
  Vec8d Isub;
  Vec8d Gbd;
  Vec8d Gbg;
  Vec8d Gbb;
  Vec8d VASCBE;
  Vec8d dVASCBE_dVg;
  Vec8d dVASCBE_dVd;
  Vec8d dVASCBE_dVb;
  Vec8d CoxWovL;
  Vec8d Rds;
  Vec8d dRds_dVg;
  Vec8d dRds_dVb;
  Vec8d WVCox;
  Vec8d WVCoxRds;
  Vec8d Vgst2Vtm;
  Vec8d VdsatCV;
  Vec8d dVdsatCV_dVg;
  Vec8d dVdsatCV_dVb;
  double Leff;
  Vec8d Weff;
  Vec8d dWeff_dVg;
  Vec8d dWeff_dVb;
  Vec8d AbulkCV;
  Vec8d dAbulkCV_dVb;
  Vec8d qgdo;
  Vec8d qgso;
  Vec8d cgdo;
  Vec8d cgso;
  Vec8d qcheq = (Vec8d ){0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Vec8d qdef;
  Vec8d gqdef = (Vec8d ){0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Vec8d cqdef;
  Vec8d cqcheq;
  Vec8d gtau_diff;
  Vec8d gtau_drift;
  Vec8d gcqdb = (Vec8d ){0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Vec8d gcqsb = (Vec8d ){0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Vec8d gcqgb = (Vec8d ){0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Vec8d gcqbb = (Vec8d ){0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Vec8d dxpart;
  Vec8d sxpart;
  Vec8d ggtg;
  Vec8d ggtd;
  Vec8d ggts;
  Vec8d ggtb;
  Vec8d ddxpart_dVd;
  Vec8d ddxpart_dVg;
  Vec8d ddxpart_dVb;
  Vec8d ddxpart_dVs;
  Vec8d dsxpart_dVd;
  Vec8d dsxpart_dVg;
  Vec8d dsxpart_dVb;
  Vec8d dsxpart_dVs;
  Vec8d gbspsp;
  Vec8d gbbdp;
  Vec8d gbbsp;
  Vec8d gbspg;
  Vec8d gbspb;
  Vec8d gbspdp;
  Vec8d gbdpdp;
  Vec8d gbdpg;
  Vec8d gbdpb;
  Vec8d gbdpsp;
  Vec8d Cgg;
  Vec8d Cgd;
  Vec8d Cgb;
  Vec8d Cdg;
  Vec8d Cdd;
  Vec8d Cds;
  Vec8d Csg;
  Vec8d Csd;
  Vec8d Css;
  Vec8d Csb;
  Vec8d Cbg;
  Vec8d Cbd;
  Vec8d Cbb;
  Vec8d Cgg1;
  Vec8d Cgb1;
  Vec8d Cgd1;
  Vec8d Cbg1;
  Vec8d Cbb1;
  Vec8d Cbd1;
  Vec8d Qac0;
  Vec8d Qsub0;
  Vec8d dQac0_dVg;
  Vec8d dQac0_dVb;
  Vec8d dQsub0_dVg;
  Vec8d dQsub0_dVd;
  Vec8d dQsub0_dVb;
  Vec8d m;
  struct bsim3SizeDependParam *pParam;
  int ByPass;
  Vec8m Check;
  int ChargeComputationNeeded;
  int error;
  Vec8m nonconcount;
  Vec8m BSIM3mode;
  ScalingFactor = 1.0e-9;
  ChargeComputationNeeded = ((ckt->CKTmode & (((MODEDCTRANCURVE | MODEAC) | MODETRAN) | MODEINITSMSIG)) || ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC))) ? (1) : (0);
  pParam = heres[0]->pParam;
  vbs = (Vec8d ){heres[0]->BSIM3SIMDvbs, heres[1]->BSIM3SIMDvbs, heres[2]->BSIM3SIMDvbs, heres[3]->BSIM3SIMDvbs, heres[4]->BSIM3SIMDvbs, heres[5]->BSIM3SIMDvbs, heres[6]->BSIM3SIMDvbs, heres[7]->BSIM3SIMDvbs};
  vgs = (Vec8d ){heres[0]->BSIM3SIMDvgs, heres[1]->BSIM3SIMDvgs, heres[2]->BSIM3SIMDvgs, heres[3]->BSIM3SIMDvgs, heres[4]->BSIM3SIMDvgs, heres[5]->BSIM3SIMDvgs, heres[6]->BSIM3SIMDvgs, heres[7]->BSIM3SIMDvgs};
  vds = (Vec8d ){heres[0]->BSIM3SIMDvds, heres[1]->BSIM3SIMDvds, heres[2]->BSIM3SIMDvds, heres[3]->BSIM3SIMDvds, heres[4]->BSIM3SIMDvds, heres[5]->BSIM3SIMDvds, heres[6]->BSIM3SIMDvds, heres[7]->BSIM3SIMDvds};
  qdef = (Vec8d ){heres[0]->BSIM3SIMDqdef, heres[1]->BSIM3SIMDqdef, heres[2]->BSIM3SIMDqdef, heres[3]->BSIM3SIMDqdef, heres[4]->BSIM3SIMDqdef, heres[5]->BSIM3SIMDqdef, heres[6]->BSIM3SIMDqdef, heres[7]->BSIM3SIMDqdef};
  cdhat = (Vec8d ){heres[0]->BSIM3SIMDcdhat, heres[1]->BSIM3SIMDcdhat, heres[2]->BSIM3SIMDcdhat, heres[3]->BSIM3SIMDcdhat, heres[4]->BSIM3SIMDcdhat, heres[5]->BSIM3SIMDcdhat, heres[6]->BSIM3SIMDcdhat, heres[7]->BSIM3SIMDcdhat};
  cbhat = (Vec8d ){heres[0]->BSIM3SIMDcbhat, heres[1]->BSIM3SIMDcbhat, heres[2]->BSIM3SIMDcbhat, heres[3]->BSIM3SIMDcbhat, heres[4]->BSIM3SIMDcbhat, heres[5]->BSIM3SIMDcbhat, heres[6]->BSIM3SIMDcbhat, heres[7]->BSIM3SIMDcbhat};
  Check = (Vec8m ){heres[0]->BSIM3SIMDCheck, heres[1]->BSIM3SIMDCheck, heres[2]->BSIM3SIMDCheck, heres[3]->BSIM3SIMDCheck, heres[4]->BSIM3SIMDCheck, heres[5]->BSIM3SIMDCheck, heres[6]->BSIM3SIMDCheck, heres[7]->BSIM3SIMDCheck};
  vbd = vbs - vds;
  vgd = vgs - vds;
  vgb = vgs - vbs;
  Nvtm = model->BSIM3vtm * model->BSIM3jctEmissionCoeff;
  if (model->BSIM3acmMod == 0)
  {
    SourceSatCurrent = vec8_SIMDTOVECTOR(1.0e-14);
    if (1)
    {
      Vec8m condmask0 = (((Vec8d ){heres[0]->BSIM3sourceArea, heres[1]->BSIM3sourceArea, heres[2]->BSIM3sourceArea, heres[3]->BSIM3sourceArea, heres[4]->BSIM3sourceArea, heres[5]->BSIM3sourceArea, heres[6]->BSIM3sourceArea, heres[7]->BSIM3sourceArea}) <= 0.0) & (((Vec8d ){heres[0]->BSIM3sourcePerimeter, heres[1]->BSIM3sourcePerimeter, heres[2]->BSIM3sourcePerimeter, heres[3]->BSIM3sourcePerimeter, heres[4]->BSIM3sourcePerimeter, heres[5]->BSIM3sourcePerimeter, heres[6]->BSIM3sourcePerimeter, heres[7]->BSIM3sourcePerimeter}) <= 0.0);
      Vec8m condmask_true0 = condmask0;
      Vec8m condmask_false0 = ~condmask0;
      {
        ;
      }
      {
        SourceSatCurrent = vec8_blend(SourceSatCurrent, (((Vec8d ){heres[0]->BSIM3sourceArea, heres[1]->BSIM3sourceArea, heres[2]->BSIM3sourceArea, heres[3]->BSIM3sourceArea, heres[4]->BSIM3sourceArea, heres[5]->BSIM3sourceArea, heres[6]->BSIM3sourceArea, heres[7]->BSIM3sourceArea}) * model->BSIM3jctTempSatCurDensity) + (((Vec8d ){heres[0]->BSIM3sourcePerimeter, heres[1]->BSIM3sourcePerimeter, heres[2]->BSIM3sourcePerimeter, heres[3]->BSIM3sourcePerimeter, heres[4]->BSIM3sourcePerimeter, heres[5]->BSIM3sourcePerimeter, heres[6]->BSIM3sourcePerimeter, heres[7]->BSIM3sourcePerimeter}) * model->BSIM3jctSidewallTempSatCurDensity), condmask_false0);
      }
    }

    DrainSatCurrent = vec8_SIMDTOVECTOR(1.0e-14);
    if (1)
    {
      Vec8m condmask0 = (((Vec8d ){heres[0]->BSIM3drainArea, heres[1]->BSIM3drainArea, heres[2]->BSIM3drainArea, heres[3]->BSIM3drainArea, heres[4]->BSIM3drainArea, heres[5]->BSIM3drainArea, heres[6]->BSIM3drainArea, heres[7]->BSIM3drainArea}) <= 0.0) & (((Vec8d ){heres[0]->BSIM3drainPerimeter, heres[1]->BSIM3drainPerimeter, heres[2]->BSIM3drainPerimeter, heres[3]->BSIM3drainPerimeter, heres[4]->BSIM3drainPerimeter, heres[5]->BSIM3drainPerimeter, heres[6]->BSIM3drainPerimeter, heres[7]->BSIM3drainPerimeter}) <= 0.0);
      Vec8m condmask_true0 = condmask0;
      Vec8m condmask_false0 = ~condmask0;
      {
        ;
      }
      {
        DrainSatCurrent = vec8_blend(DrainSatCurrent, (((Vec8d ){heres[0]->BSIM3drainArea, heres[1]->BSIM3drainArea, heres[2]->BSIM3drainArea, heres[3]->BSIM3drainArea, heres[4]->BSIM3drainArea, heres[5]->BSIM3drainArea, heres[6]->BSIM3drainArea, heres[7]->BSIM3drainArea}) * model->BSIM3jctTempSatCurDensity) + (((Vec8d ){heres[0]->BSIM3drainPerimeter, heres[1]->BSIM3drainPerimeter, heres[2]->BSIM3drainPerimeter, heres[3]->BSIM3drainPerimeter, heres[4]->BSIM3drainPerimeter, heres[5]->BSIM3drainPerimeter, heres[6]->BSIM3drainPerimeter, heres[7]->BSIM3drainPerimeter}) * model->BSIM3jctSidewallTempSatCurDensity), condmask_false0);
      }
    }

  }
  else
  {
    error = vec8_BSIM3_ACM_saturationCurrents(model, heres, &DrainSatCurrent, &SourceSatCurrent);
    if (SIMDANY(error))
      return error;

  }

  if (1)
  {
    Vec8m condmask0 = SourceSatCurrent <= 0.0;
    Vec8m condmask_true0 = condmask0;
    Vec8m condmask_false0 = ~condmask0;
    {
      {
        if (condmask_true0[0])
          heres[0]->BSIM3gbs = ckt->CKTgmin;

        if (condmask_true0[1])
          heres[1]->BSIM3gbs = ckt->CKTgmin;

        if (condmask_true0[2])
          heres[2]->BSIM3gbs = ckt->CKTgmin;

        if (condmask_true0[3])
          heres[3]->BSIM3gbs = ckt->CKTgmin;

        if (condmask_true0[4])
          heres[4]->BSIM3gbs = ckt->CKTgmin;

        if (condmask_true0[5])
          heres[5]->BSIM3gbs = ckt->CKTgmin;

        if (condmask_true0[6])
          heres[6]->BSIM3gbs = ckt->CKTgmin;

        if (condmask_true0[7])
          heres[7]->BSIM3gbs = ckt->CKTgmin;

      }
      {
        Vec8d val = ((Vec8d ){heres[0]->BSIM3gbs, heres[1]->BSIM3gbs, heres[2]->BSIM3gbs, heres[3]->BSIM3gbs, heres[4]->BSIM3gbs, heres[5]->BSIM3gbs, heres[6]->BSIM3gbs, heres[7]->BSIM3gbs}) * vbs;
        if (condmask_true0[0])
          heres[0]->BSIM3cbs = val[0];

        if (condmask_true0[1])
          heres[1]->BSIM3cbs = val[1];

        if (condmask_true0[2])
          heres[2]->BSIM3cbs = val[2];

        if (condmask_true0[3])
          heres[3]->BSIM3cbs = val[3];

        if (condmask_true0[4])
          heres[4]->BSIM3cbs = val[4];

        if (condmask_true0[5])
          heres[5]->BSIM3cbs = val[5];

        if (condmask_true0[6])
          heres[6]->BSIM3cbs = val[6];

        if (condmask_true0[7])
          heres[7]->BSIM3cbs = val[7];

      }
    }
    {
      if (model->BSIM3ijth == 0.0)
      {
        evbs = vec8_blend(evbs, vec8_exp(vbs / Nvtm), condmask_false0);
        {
          Vec8d val = ((SourceSatCurrent * evbs) / Nvtm) + ckt->CKTgmin;
          if (condmask_false0[0])
            heres[0]->BSIM3gbs = val[0];

          if (condmask_false0[1])
            heres[1]->BSIM3gbs = val[1];

          if (condmask_false0[2])
            heres[2]->BSIM3gbs = val[2];

          if (condmask_false0[3])
            heres[3]->BSIM3gbs = val[3];

          if (condmask_false0[4])
            heres[4]->BSIM3gbs = val[4];

          if (condmask_false0[5])
            heres[5]->BSIM3gbs = val[5];

          if (condmask_false0[6])
            heres[6]->BSIM3gbs = val[6];

          if (condmask_false0[7])
            heres[7]->BSIM3gbs = val[7];

        }
        {
          Vec8d val = (SourceSatCurrent * (evbs - 1.0)) + (ckt->CKTgmin * vbs);
          if (condmask_false0[0])
            heres[0]->BSIM3cbs = val[0];

          if (condmask_false0[1])
            heres[1]->BSIM3cbs = val[1];

          if (condmask_false0[2])
            heres[2]->BSIM3cbs = val[2];

          if (condmask_false0[3])
            heres[3]->BSIM3cbs = val[3];

          if (condmask_false0[4])
            heres[4]->BSIM3cbs = val[4];

          if (condmask_false0[5])
            heres[5]->BSIM3cbs = val[5];

          if (condmask_false0[6])
            heres[6]->BSIM3cbs = val[6];

          if (condmask_false0[7])
            heres[7]->BSIM3cbs = val[7];

        }
      }
      else
      {
        if (1)
        {
          Vec8m condmask1 = vbs < ((Vec8d ){heres[0]->BSIM3vjsm, heres[1]->BSIM3vjsm, heres[2]->BSIM3vjsm, heres[3]->BSIM3vjsm, heres[4]->BSIM3vjsm, heres[5]->BSIM3vjsm, heres[6]->BSIM3vjsm, heres[7]->BSIM3vjsm});
          Vec8m condmask_true1 = condmask_false0 & condmask1;
          Vec8m condmask_false1 = condmask_false0 & (~condmask1);
          {
            evbs = vec8_blend(evbs, vec8_exp(vbs / Nvtm), condmask_true1);
            {
              Vec8d val = ((SourceSatCurrent * evbs) / Nvtm) + ckt->CKTgmin;
              if (condmask_true1[0])
                heres[0]->BSIM3gbs = val[0];

              if (condmask_true1[1])
                heres[1]->BSIM3gbs = val[1];

              if (condmask_true1[2])
                heres[2]->BSIM3gbs = val[2];

              if (condmask_true1[3])
                heres[3]->BSIM3gbs = val[3];

              if (condmask_true1[4])
                heres[4]->BSIM3gbs = val[4];

              if (condmask_true1[5])
                heres[5]->BSIM3gbs = val[5];

              if (condmask_true1[6])
                heres[6]->BSIM3gbs = val[6];

              if (condmask_true1[7])
                heres[7]->BSIM3gbs = val[7];

            }
            {
              Vec8d val = (SourceSatCurrent * (evbs - 1.0)) + (ckt->CKTgmin * vbs);
              if (condmask_true1[0])
                heres[0]->BSIM3cbs = val[0];

              if (condmask_true1[1])
                heres[1]->BSIM3cbs = val[1];

              if (condmask_true1[2])
                heres[2]->BSIM3cbs = val[2];

              if (condmask_true1[3])
                heres[3]->BSIM3cbs = val[3];

              if (condmask_true1[4])
                heres[4]->BSIM3cbs = val[4];

              if (condmask_true1[5])
                heres[5]->BSIM3cbs = val[5];

              if (condmask_true1[6])
                heres[6]->BSIM3cbs = val[6];

              if (condmask_true1[7])
                heres[7]->BSIM3cbs = val[7];

            }
          }
          {
            T0 = vec8_blend(T0, ((Vec8d ){heres[0]->BSIM3IsEvjsm, heres[1]->BSIM3IsEvjsm, heres[2]->BSIM3IsEvjsm, heres[3]->BSIM3IsEvjsm, heres[4]->BSIM3IsEvjsm, heres[5]->BSIM3IsEvjsm, heres[6]->BSIM3IsEvjsm, heres[7]->BSIM3IsEvjsm}) / Nvtm, condmask_false1);
            {
              Vec8d val = T0 + ckt->CKTgmin;
              if (condmask_false1[0])
                heres[0]->BSIM3gbs = val[0];

              if (condmask_false1[1])
                heres[1]->BSIM3gbs = val[1];

              if (condmask_false1[2])
                heres[2]->BSIM3gbs = val[2];

              if (condmask_false1[3])
                heres[3]->BSIM3gbs = val[3];

              if (condmask_false1[4])
                heres[4]->BSIM3gbs = val[4];

              if (condmask_false1[5])
                heres[5]->BSIM3gbs = val[5];

              if (condmask_false1[6])
                heres[6]->BSIM3gbs = val[6];

              if (condmask_false1[7])
                heres[7]->BSIM3gbs = val[7];

            }
            {
              Vec8d val = ((((Vec8d ){heres[0]->BSIM3IsEvjsm, heres[1]->BSIM3IsEvjsm, heres[2]->BSIM3IsEvjsm, heres[3]->BSIM3IsEvjsm, heres[4]->BSIM3IsEvjsm, heres[5]->BSIM3IsEvjsm, heres[6]->BSIM3IsEvjsm, heres[7]->BSIM3IsEvjsm}) - SourceSatCurrent) + (T0 * (vbs - ((Vec8d ){heres[0]->BSIM3vjsm, heres[1]->BSIM3vjsm, heres[2]->BSIM3vjsm, heres[3]->BSIM3vjsm, heres[4]->BSIM3vjsm, heres[5]->BSIM3vjsm, heres[6]->BSIM3vjsm, heres[7]->BSIM3vjsm})))) + (ckt->CKTgmin * vbs);
              if (condmask_false1[0])
                heres[0]->BSIM3cbs = val[0];

              if (condmask_false1[1])
                heres[1]->BSIM3cbs = val[1];

              if (condmask_false1[2])
                heres[2]->BSIM3cbs = val[2];

              if (condmask_false1[3])
                heres[3]->BSIM3cbs = val[3];

              if (condmask_false1[4])
                heres[4]->BSIM3cbs = val[4];

              if (condmask_false1[5])
                heres[5]->BSIM3cbs = val[5];

              if (condmask_false1[6])
                heres[6]->BSIM3cbs = val[6];

              if (condmask_false1[7])
                heres[7]->BSIM3cbs = val[7];

            }
          }
        }

      }

    }
  }

  if (1)
  {
    Vec8m condmask0 = DrainSatCurrent <= 0.0;
    Vec8m condmask_true0 = condmask0;
    Vec8m condmask_false0 = ~condmask0;
    {
      {
        if (condmask_true0[0])
          heres[0]->BSIM3gbd = ckt->CKTgmin;

        if (condmask_true0[1])
          heres[1]->BSIM3gbd = ckt->CKTgmin;

        if (condmask_true0[2])
          heres[2]->BSIM3gbd = ckt->CKTgmin;

        if (condmask_true0[3])
          heres[3]->BSIM3gbd = ckt->CKTgmin;

        if (condmask_true0[4])
          heres[4]->BSIM3gbd = ckt->CKTgmin;

        if (condmask_true0[5])
          heres[5]->BSIM3gbd = ckt->CKTgmin;

        if (condmask_true0[6])
          heres[6]->BSIM3gbd = ckt->CKTgmin;

        if (condmask_true0[7])
          heres[7]->BSIM3gbd = ckt->CKTgmin;

      }
      {
        Vec8d val = ((Vec8d ){heres[0]->BSIM3gbd, heres[1]->BSIM3gbd, heres[2]->BSIM3gbd, heres[3]->BSIM3gbd, heres[4]->BSIM3gbd, heres[5]->BSIM3gbd, heres[6]->BSIM3gbd, heres[7]->BSIM3gbd}) * vbd;
        if (condmask_true0[0])
          heres[0]->BSIM3cbd = val[0];

        if (condmask_true0[1])
          heres[1]->BSIM3cbd = val[1];

        if (condmask_true0[2])
          heres[2]->BSIM3cbd = val[2];

        if (condmask_true0[3])
          heres[3]->BSIM3cbd = val[3];

        if (condmask_true0[4])
          heres[4]->BSIM3cbd = val[4];

        if (condmask_true0[5])
          heres[5]->BSIM3cbd = val[5];

        if (condmask_true0[6])
          heres[6]->BSIM3cbd = val[6];

        if (condmask_true0[7])
          heres[7]->BSIM3cbd = val[7];

      }
    }
    {
      if (model->BSIM3ijth == 0.0)
      {
        evbd = vec8_blend(evbd, vec8_exp(vbd / Nvtm), condmask_false0);
        {
          Vec8d val = ((DrainSatCurrent * evbd) / Nvtm) + ckt->CKTgmin;
          if (condmask_false0[0])
            heres[0]->BSIM3gbd = val[0];

          if (condmask_false0[1])
            heres[1]->BSIM3gbd = val[1];

          if (condmask_false0[2])
            heres[2]->BSIM3gbd = val[2];

          if (condmask_false0[3])
            heres[3]->BSIM3gbd = val[3];

          if (condmask_false0[4])
            heres[4]->BSIM3gbd = val[4];

          if (condmask_false0[5])
            heres[5]->BSIM3gbd = val[5];

          if (condmask_false0[6])
            heres[6]->BSIM3gbd = val[6];

          if (condmask_false0[7])
            heres[7]->BSIM3gbd = val[7];

        }
        {
          Vec8d val = (DrainSatCurrent * (evbd - 1.0)) + (ckt->CKTgmin * vbd);
          if (condmask_false0[0])
            heres[0]->BSIM3cbd = val[0];

          if (condmask_false0[1])
            heres[1]->BSIM3cbd = val[1];

          if (condmask_false0[2])
            heres[2]->BSIM3cbd = val[2];

          if (condmask_false0[3])
            heres[3]->BSIM3cbd = val[3];

          if (condmask_false0[4])
            heres[4]->BSIM3cbd = val[4];

          if (condmask_false0[5])
            heres[5]->BSIM3cbd = val[5];

          if (condmask_false0[6])
            heres[6]->BSIM3cbd = val[6];

          if (condmask_false0[7])
            heres[7]->BSIM3cbd = val[7];

        }
      }
      else
      {
        if (1)
        {
          Vec8m condmask1 = vbd < ((Vec8d ){heres[0]->BSIM3vjdm, heres[1]->BSIM3vjdm, heres[2]->BSIM3vjdm, heres[3]->BSIM3vjdm, heres[4]->BSIM3vjdm, heres[5]->BSIM3vjdm, heres[6]->BSIM3vjdm, heres[7]->BSIM3vjdm});
          Vec8m condmask_true1 = condmask_false0 & condmask1;
          Vec8m condmask_false1 = condmask_false0 & (~condmask1);
          {
            evbd = vec8_blend(evbd, vec8_exp(vbd / Nvtm), condmask_true1);
            {
              Vec8d val = ((DrainSatCurrent * evbd) / Nvtm) + ckt->CKTgmin;
              if (condmask_true1[0])
                heres[0]->BSIM3gbd = val[0];

              if (condmask_true1[1])
                heres[1]->BSIM3gbd = val[1];

              if (condmask_true1[2])
                heres[2]->BSIM3gbd = val[2];

              if (condmask_true1[3])
                heres[3]->BSIM3gbd = val[3];

              if (condmask_true1[4])
                heres[4]->BSIM3gbd = val[4];

              if (condmask_true1[5])
                heres[5]->BSIM3gbd = val[5];

              if (condmask_true1[6])
                heres[6]->BSIM3gbd = val[6];

              if (condmask_true1[7])
                heres[7]->BSIM3gbd = val[7];

            }
            {
              Vec8d val = (DrainSatCurrent * (evbd - 1.0)) + (ckt->CKTgmin * vbd);
              if (condmask_true1[0])
                heres[0]->BSIM3cbd = val[0];

              if (condmask_true1[1])
                heres[1]->BSIM3cbd = val[1];

              if (condmask_true1[2])
                heres[2]->BSIM3cbd = val[2];

              if (condmask_true1[3])
                heres[3]->BSIM3cbd = val[3];

              if (condmask_true1[4])
                heres[4]->BSIM3cbd = val[4];

              if (condmask_true1[5])
                heres[5]->BSIM3cbd = val[5];

              if (condmask_true1[6])
                heres[6]->BSIM3cbd = val[6];

              if (condmask_true1[7])
                heres[7]->BSIM3cbd = val[7];

            }
          }
          {
            T0 = vec8_blend(T0, ((Vec8d ){heres[0]->BSIM3IsEvjdm, heres[1]->BSIM3IsEvjdm, heres[2]->BSIM3IsEvjdm, heres[3]->BSIM3IsEvjdm, heres[4]->BSIM3IsEvjdm, heres[5]->BSIM3IsEvjdm, heres[6]->BSIM3IsEvjdm, heres[7]->BSIM3IsEvjdm}) / Nvtm, condmask_false1);
            {
              Vec8d val = T0 + ckt->CKTgmin;
              if (condmask_false1[0])
                heres[0]->BSIM3gbd = val[0];

              if (condmask_false1[1])
                heres[1]->BSIM3gbd = val[1];

              if (condmask_false1[2])
                heres[2]->BSIM3gbd = val[2];

              if (condmask_false1[3])
                heres[3]->BSIM3gbd = val[3];

              if (condmask_false1[4])
                heres[4]->BSIM3gbd = val[4];

              if (condmask_false1[5])
                heres[5]->BSIM3gbd = val[5];

              if (condmask_false1[6])
                heres[6]->BSIM3gbd = val[6];

              if (condmask_false1[7])
                heres[7]->BSIM3gbd = val[7];

            }
            {
              Vec8d val = ((((Vec8d ){heres[0]->BSIM3IsEvjdm, heres[1]->BSIM3IsEvjdm, heres[2]->BSIM3IsEvjdm, heres[3]->BSIM3IsEvjdm, heres[4]->BSIM3IsEvjdm, heres[5]->BSIM3IsEvjdm, heres[6]->BSIM3IsEvjdm, heres[7]->BSIM3IsEvjdm}) - DrainSatCurrent) + (T0 * (vbd - ((Vec8d ){heres[0]->BSIM3vjdm, heres[1]->BSIM3vjdm, heres[2]->BSIM3vjdm, heres[3]->BSIM3vjdm, heres[4]->BSIM3vjdm, heres[5]->BSIM3vjdm, heres[6]->BSIM3vjdm, heres[7]->BSIM3vjdm})))) + (ckt->CKTgmin * vbd);
              if (condmask_false1[0])
                heres[0]->BSIM3cbd = val[0];

              if (condmask_false1[1])
                heres[1]->BSIM3cbd = val[1];

              if (condmask_false1[2])
                heres[2]->BSIM3cbd = val[2];

              if (condmask_false1[3])
                heres[3]->BSIM3cbd = val[3];

              if (condmask_false1[4])
                heres[4]->BSIM3cbd = val[4];

              if (condmask_false1[5])
                heres[5]->BSIM3cbd = val[5];

              if (condmask_false1[6])
                heres[6]->BSIM3cbd = val[6];

              if (condmask_false1[7])
                heres[7]->BSIM3cbd = val[7];

            }
          }
        }

      }

    }
  }

  BSIM3mode = vds >= 0.0;
  if (1)
  {
    Vec8m condmask0 = vds >= 0.0;
    Vec8m condmask_true0 = condmask0;
    Vec8m condmask_false0 = ~condmask0;
    {
      Vds = vec8_blend(Vds, vds, condmask_true0);
      Vgs = vec8_blend(Vgs, vgs, condmask_true0);
      Vbs = vec8_blend(Vbs, vbs, condmask_true0);
    }
    {
      Vds = vec8_blend(Vds, -vds, condmask_false0);
      Vgs = vec8_blend(Vgs, vgd, condmask_false0);
      Vbs = vec8_blend(Vbs, vbd, condmask_false0);
    }
  }

  Vec8m modesym;
  modesym = (2 * (BSIM3mode & 0x1)) - 1;
  {
    heres[0]->BSIM3mode = modesym[0];
    heres[1]->BSIM3mode = modesym[1];
    heres[2]->BSIM3mode = modesym[2];
    heres[3]->BSIM3mode = modesym[3];
    heres[4]->BSIM3mode = modesym[4];
    heres[5]->BSIM3mode = modesym[5];
    heres[6]->BSIM3mode = modesym[6];
    heres[7]->BSIM3mode = modesym[7];
  }
  T0 = (Vbs - pParam->BSIM3vbsc) - 0.001;
  T1 = vec8_sqrt((T0 * T0) - (0.004 * pParam->BSIM3vbsc));
  Vbseff = pParam->BSIM3vbsc + (0.5 * (T0 + T1));
  dVbseff_dVb = 0.5 * (1.0 + (T0 / T1));
  if (1)
  {
    Vec8m condmask0 = Vbseff < Vbs;
    Vec8m condmask_true0 = condmask0;
    {
      Vbseff = vec8_blend(Vbseff, Vbs, condmask_true0);
    }
  }

  if (1)
  {
    Vec8m condmask0 = Vbseff > 0.0;
    Vec8m condmask_true0 = condmask0;
    Vec8m condmask_false0 = ~condmask0;
    {
      T0 = vec8_blend(T0, pParam->BSIM3phi / (pParam->BSIM3phi + Vbseff), condmask_true0);
      Phis = vec8_blend(Phis, pParam->BSIM3phi * T0, condmask_true0);
      dPhis_dVb = vec8_blend(dPhis_dVb, (-T0) * T0, condmask_true0);
      sqrtPhis = vec8_blend(sqrtPhis, pParam->BSIM3phis3 / (pParam->BSIM3phi + (0.5 * Vbseff)), condmask_true0);
      dsqrtPhis_dVb = vec8_blend(dsqrtPhis_dVb, (((-0.5) * sqrtPhis) * sqrtPhis) / pParam->BSIM3phis3, condmask_true0);
    }
    {
      Phis = vec8_blend(Phis, pParam->BSIM3phi - Vbseff, condmask_false0);
      dPhis_dVb = vec8_blend(dPhis_dVb, vec8_SIMDTOVECTOR(-1.0), condmask_false0);
      sqrtPhis = vec8_blend(sqrtPhis, vec8_sqrt(Phis), condmask_false0);
      dsqrtPhis_dVb = vec8_blend(dsqrtPhis_dVb, (-0.5) / sqrtPhis, condmask_false0);
    }
  }

  Xdep = (pParam->BSIM3Xdep0 * sqrtPhis) / pParam->BSIM3sqrtPhi;
  dXdep_dVb = (pParam->BSIM3Xdep0 / pParam->BSIM3sqrtPhi) * dsqrtPhis_dVb;
  Leff = pParam->BSIM3leff;
  Vtm = model->BSIM3vtm;
  T3 = vec8_sqrt(Xdep);
  V0 = pParam->BSIM3vbi - pParam->BSIM3phi;
  T0 = pParam->BSIM3dvt2 * Vbseff;
  T2 = vec8_SIMDTOVECTOR(pParam->BSIM3dvt2);
  if (1)
  {
    Vec8m condmask0 = T0 >= (-0.5);
    Vec8m condmask_true0 = condmask0;
    Vec8m condmask_false0 = ~condmask0;
    {
      T1 = vec8_blend(T1, 1.0 + T0, condmask_true0);
    }
    {
      T4 = vec8_blend(T4, 1.0 / (3.0 + (8.0 * T0)), condmask_false0);
      T1 = vec8_blend(T1, (1.0 + (3.0 * T0)) * T4, condmask_false0);
      T2 = vec8_blend(T2, (T2 * T4) * T4, condmask_false0);
    }
  }

  lt1 = (model->BSIM3factor1 * T3) * T1;
  dlt1_dVb = model->BSIM3factor1 * ((((0.5 / T3) * T1) * dXdep_dVb) + (T3 * T2));
  T0 = pParam->BSIM3dvt2w * Vbseff;
  if (1)
  {
    Vec8m condmask0 = T0 >= (-0.5);
    Vec8m condmask_true0 = condmask0;
    Vec8m condmask_false0 = ~condmask0;
    {
      T1 = vec8_blend(T1, 1.0 + T0, condmask_true0);
      T2 = vec8_blend(T2, vec8_SIMDTOVECTOR(pParam->BSIM3dvt2w), condmask_true0);
    }
    {
      T4 = vec8_blend(T4, 1.0 / (3.0 + (8.0 * T0)), condmask_false0);
      T1 = vec8_blend(T1, (1.0 + (3.0 * T0)) * T4, condmask_false0);
      T2 = vec8_blend(T2, (pParam->BSIM3dvt2w * T4) * T4, condmask_false0);
    }
  }

  ltw = (model->BSIM3factor1 * T3) * T1;
  dltw_dVb = model->BSIM3factor1 * ((((0.5 / T3) * T1) * dXdep_dVb) + (T3 * T2));
  T0 = (((-0.5) * pParam->BSIM3dvt1) * Leff) / lt1;
  if (1)
  {
    Vec8m condmask0 = T0 > (-EXP_THRESHOLD);
    Vec8m condmask_true0 = condmask0;
    Vec8m condmask_false0 = ~condmask0;
    {
      T1 = vec8_blend(T1, vec8_exp(T0), condmask_true0);
      Theta0 = vec8_blend(Theta0, T1 * (1.0 + (2.0 * T1)), condmask_true0);
      dT1_dVb = vec8_blend(dT1_dVb, (((-T0) / lt1) * T1) * dlt1_dVb, condmask_true0);
      dTheta0_dVb = vec8_blend(dTheta0_dVb, (1.0 + (4.0 * T1)) * dT1_dVb, condmask_true0);
    }
    {
      T1 = vec8_blend(T1, vec8_SIMDTOVECTOR(MIN_EXP), condmask_false0);
      Theta0 = vec8_blend(Theta0, T1 * (1.0 + (2.0 * T1)), condmask_false0);
      dTheta0_dVb = vec8_blend(dTheta0_dVb, vec8_SIMDTOVECTOR(0.0), condmask_false0);
    }
  }

  {
    Vec8d val = pParam->BSIM3dvt0 * Theta0;
    heres[0]->BSIM3thetavth = val[0];
    heres[1]->BSIM3thetavth = val[1];
    heres[2]->BSIM3thetavth = val[2];
    heres[3]->BSIM3thetavth = val[3];
    heres[4]->BSIM3thetavth = val[4];
    heres[5]->BSIM3thetavth = val[5];
    heres[6]->BSIM3thetavth = val[6];
    heres[7]->BSIM3thetavth = val[7];
  }
  Delt_vth = (pParam->BSIM3dvt0 * Theta0) * V0;
  dDelt_vth_dVb = (pParam->BSIM3dvt0 * dTheta0_dVb) * V0;
  T0 = ((((-0.5) * pParam->BSIM3dvt1w) * pParam->BSIM3weff) * Leff) / ltw;
  if (1)
  {
    Vec8m condmask0 = T0 > (-EXP_THRESHOLD);
    Vec8m condmask_true0 = condmask0;
    Vec8m condmask_false0 = ~condmask0;
    {
      T1 = vec8_blend(T1, vec8_exp(T0), condmask_true0);
      T2 = vec8_blend(T2, T1 * (1.0 + (2.0 * T1)), condmask_true0);
      dT1_dVb = vec8_blend(dT1_dVb, (((-T0) / ltw) * T1) * dltw_dVb, condmask_true0);
      dT2_dVb = vec8_blend(dT2_dVb, (1.0 + (4.0 * T1)) * dT1_dVb, condmask_true0);
    }
    {
      T1 = vec8_blend(T1, vec8_SIMDTOVECTOR(MIN_EXP), condmask_false0);
      T2 = vec8_blend(T2, T1 * (1.0 + (2.0 * T1)), condmask_false0);
      dT2_dVb = vec8_blend(dT2_dVb, vec8_SIMDTOVECTOR(0.0), condmask_false0);
    }
  }

  T0 = pParam->BSIM3dvt0w * T2;
  T2 = T0 * V0;
  dT2_dVb = (pParam->BSIM3dvt0w * dT2_dVb) * V0;
  TempRatio = (ckt->CKTtemp / model->BSIM3tnom) - 1.0;
  T0 = vec8_SIMDTOVECTOR(sqrt(1.0 + (pParam->BSIM3nlx / Leff)));
  T1 = ((pParam->BSIM3k1ox * (T0 - 1.0)) * pParam->BSIM3sqrtPhi) + (((pParam->BSIM3kt1 + (pParam->BSIM3kt1l / Leff)) + (pParam->BSIM3kt2 * Vbseff)) * TempRatio);
  tmp2 = vec8_SIMDTOVECTOR((model->BSIM3tox * pParam->BSIM3phi) / (pParam->BSIM3weff + pParam->BSIM3w0));
  T3 = pParam->BSIM3eta0 + (pParam->BSIM3etab * Vbseff);
  if (1)
  {
    Vec8m condmask0 = T3 < 1.0e-4;
    Vec8m condmask_true0 = condmask0;
    Vec8m condmask_false0 = ~condmask0;
    {
      T9 = vec8_blend(T9, 1.0 / (3.0 - (2.0e4 * T3)), condmask_true0);
      T3 = vec8_blend(T3, (2.0e-4 - T3) * T9, condmask_true0);
      T4 = vec8_blend(T4, T9 * T9, condmask_true0);
    }
    {
      T4 = vec8_blend(T4, vec8_SIMDTOVECTOR(1.0), condmask_false0);
    }
  }

  dDIBL_Sft_dVd = T3 * pParam->BSIM3theta0vb0;
  DIBL_Sft = dDIBL_Sft_dVd * Vds;
  Vth = ((((((((model->BSIM3type * ((Vec8d ){heres[0]->BSIM3vth0, heres[1]->BSIM3vth0, heres[2]->BSIM3vth0, heres[3]->BSIM3vth0, heres[4]->BSIM3vth0, heres[5]->BSIM3vth0, heres[6]->BSIM3vth0, heres[7]->BSIM3vth0})) - (pParam->BSIM3k1 * pParam->BSIM3sqrtPhi)) + (pParam->BSIM3k1ox * sqrtPhis)) - (pParam->BSIM3k2ox * Vbseff)) - Delt_vth) - T2) + ((pParam->BSIM3k3 + (pParam->BSIM3k3b * Vbseff)) * tmp2)) + T1) - DIBL_Sft;
  {
    heres[0]->BSIM3von = Vth[0];
    heres[1]->BSIM3von = Vth[1];
    heres[2]->BSIM3von = Vth[2];
    heres[3]->BSIM3von = Vth[3];
    heres[4]->BSIM3von = Vth[4];
    heres[5]->BSIM3von = Vth[5];
    heres[6]->BSIM3von = Vth[6];
    heres[7]->BSIM3von = Vth[7];
  }
  dVth_dVb = ((((((pParam->BSIM3k1ox * dsqrtPhis_dVb) - pParam->BSIM3k2ox) - dDelt_vth_dVb) - dT2_dVb) + (pParam->BSIM3k3b * tmp2)) - (((pParam->BSIM3etab * Vds) * pParam->BSIM3theta0vb0) * T4)) + (pParam->BSIM3kt2 * TempRatio);
  dVth_dVd = -dDIBL_Sft_dVd;
  tmp2 = (pParam->BSIM3nfactor * EPSSI) / Xdep;
  tmp3 = (pParam->BSIM3cdsc + (pParam->BSIM3cdscb * Vbseff)) + (pParam->BSIM3cdscd * Vds);
  tmp4 = ((tmp2 + (tmp3 * Theta0)) + pParam->BSIM3cit) / model->BSIM3cox;
  if (1)
  {
    Vec8m condmask0 = tmp4 >= (-0.5);
    Vec8m condmask_true0 = condmask0;
    Vec8m condmask_false0 = ~condmask0;
    {
      n = vec8_blend(n, 1.0 + tmp4, condmask_true0);
      dn_dVb = vec8_blend(dn_dVb, (((((-tmp2) / Xdep) * dXdep_dVb) + (tmp3 * dTheta0_dVb)) + (pParam->BSIM3cdscb * Theta0)) / model->BSIM3cox, condmask_true0);
      dn_dVd = vec8_blend(dn_dVd, (pParam->BSIM3cdscd * Theta0) / model->BSIM3cox, condmask_true0);
    }
    {
      T0 = vec8_blend(T0, 1.0 / (3.0 + (8.0 * tmp4)), condmask_false0);
      n = vec8_blend(n, (1.0 + (3.0 * tmp4)) * T0, condmask_false0);
      T0 = vec8_blend(T0, T0 * T0, condmask_false0);
      dn_dVb = vec8_blend(dn_dVb, ((((((-tmp2) / Xdep) * dXdep_dVb) + (tmp3 * dTheta0_dVb)) + (pParam->BSIM3cdscb * Theta0)) / model->BSIM3cox) * T0, condmask_false0);
      dn_dVd = vec8_blend(dn_dVd, ((pParam->BSIM3cdscd * Theta0) / model->BSIM3cox) * T0, condmask_false0);
    }
  }

  T0 = ((Vec8d ){heres[0]->BSIM3vfb, heres[1]->BSIM3vfb, heres[2]->BSIM3vfb, heres[3]->BSIM3vfb, heres[4]->BSIM3vfb, heres[5]->BSIM3vfb, heres[6]->BSIM3vfb, heres[7]->BSIM3vfb}) + pParam->BSIM3phi;
  Vgs_eff = Vgs;
  dVgs_eff_dVg = vec8_SIMDTOVECTOR(1.0);
  if ((pParam->BSIM3ngate > 1.e18) && (pParam->BSIM3ngate < 1.e25))
    if (1)
  {
    Vec8m condmask0 = Vgs > T0;
    Vec8m condmask_true0 = condmask0;
    {
      T1 = vec8_blend(T1, vec8_SIMDTOVECTOR((((1.0e6 * Charge_q) * EPSSI) * pParam->BSIM3ngate) / (model->BSIM3cox * model->BSIM3cox)), condmask_true0);
      T4 = vec8_blend(T4, vec8_sqrt(1.0 + ((2.0 * (Vgs - T0)) / T1)), condmask_true0);
      T2 = vec8_blend(T2, T1 * (T4 - 1.0), condmask_true0);
      T3 = vec8_blend(T3, ((0.5 * T2) * T2) / T1, condmask_true0);
      T7 = vec8_blend(T7, (1.12 - T3) - 0.05, condmask_true0);
      T6 = vec8_blend(T6, vec8_sqrt((T7 * T7) + 0.224), condmask_true0);
      T5 = vec8_blend(T5, 1.12 - (0.5 * (T7 + T6)), condmask_true0);
      Vgs_eff = vec8_blend(Vgs_eff, Vgs - T5, condmask_true0);
      dVgs_eff_dVg = vec8_blend(dVgs_eff_dVg, 1.0 - ((0.5 - (0.5 / T4)) * (1.0 + (T7 / T6))), condmask_true0);
    }
  }


  Vgst = Vgs_eff - Vth;
  T10 = (2.0 * n) * Vtm;
  VgstNVt = Vgst / T10;
  ExpArg = ((2.0 * pParam->BSIM3voff) - Vgst) / T10;
  T0 = VgstNVt;
  if (1)
  {
    Vec8m condmask0 = ExpArg > EXP_THRESHOLD;
    Vec8m condmask_true0 = condmask0;
    T0 = vec8_blend(T0, (Vgst - pParam->BSIM3voff) / (n * Vtm), condmask_true0);
  }

  ExpVgst = vec8_exp(T0);
  if (1)
  {
    Vec8m condmask0 = VgstNVt > EXP_THRESHOLD;
    Vec8m condmask_true0 = condmask0;
    Vec8m condmask_false0 = ~condmask0;
    {
      Vgsteff = vec8_blend(Vgsteff, Vgst, condmask_true0);
      dVgsteff_dVg = vec8_blend(dVgsteff_dVg, dVgs_eff_dVg, condmask_true0);
      dVgsteff_dVd = vec8_blend(dVgsteff_dVd, -dVth_dVd, condmask_true0);
      dVgsteff_dVb = vec8_blend(dVgsteff_dVb, -dVth_dVb, condmask_true0);
    }
    if (1)
    {
      Vec8m condmask1 = ExpArg > EXP_THRESHOLD;
      Vec8m condmask_true1 = condmask_false0 & condmask1;
      Vec8m condmask_false1 = condmask_false0 & (~condmask1);
      {
        Vgsteff = vec8_blend(Vgsteff, ((Vtm * pParam->BSIM3cdep0) / model->BSIM3cox) * ExpVgst, condmask_true1);
        dVgsteff_dVg = vec8_blend(dVgsteff_dVg, Vgsteff / (n * Vtm), condmask_true1);
        dVgsteff_dVd = vec8_blend(dVgsteff_dVd, (-dVgsteff_dVg) * (dVth_dVd + ((T0 * Vtm) * dn_dVd)), condmask_true1);
        dVgsteff_dVb = vec8_blend(dVgsteff_dVb, (-dVgsteff_dVg) * (dVth_dVb + ((T0 * Vtm) * dn_dVb)), condmask_true1);
        dVgsteff_dVg = vec8_blend(dVgsteff_dVg, dVgsteff_dVg * dVgs_eff_dVg, condmask_true1);
      }
      {
        T1 = vec8_blend(T1, T10 * vec8_log(1.0 + ExpVgst), condmask_false1);
        dT1_dVg = vec8_blend(dT1_dVg, ExpVgst / (1.0 + ExpVgst), condmask_false1);
        dT1_dVb = vec8_blend(dT1_dVb, ((-dT1_dVg) * (dVth_dVb + ((Vgst / n) * dn_dVb))) + ((T1 / n) * dn_dVb), condmask_false1);
        dT1_dVd = vec8_blend(dT1_dVd, ((-dT1_dVg) * (dVth_dVd + ((Vgst / n) * dn_dVd))) + ((T1 / n) * dn_dVd), condmask_false1);
        dT2_dVg = vec8_blend(dT2_dVg, ((-model->BSIM3cox) / (Vtm * pParam->BSIM3cdep0)) * vec8_exp(ExpArg), condmask_false1);
        T2 = vec8_blend(T2, 1.0 - (T10 * dT2_dVg), condmask_false1);
        dT2_dVd = vec8_blend(dT2_dVd, ((-dT2_dVg) * (dVth_dVd - (((2.0 * Vtm) * ExpArg) * dn_dVd))) + (((T2 - 1.0) / n) * dn_dVd), condmask_false1);
        dT2_dVb = vec8_blend(dT2_dVb, ((-dT2_dVg) * (dVth_dVb - (((2.0 * Vtm) * ExpArg) * dn_dVb))) + (((T2 - 1.0) / n) * dn_dVb), condmask_false1);
        Vgsteff = vec8_blend(Vgsteff, T1 / T2, condmask_false1);
        T3 = vec8_blend(T3, T2 * T2, condmask_false1);
        dVgsteff_dVg = vec8_blend(dVgsteff_dVg, (((T2 * dT1_dVg) - (T1 * dT2_dVg)) / T3) * dVgs_eff_dVg, condmask_false1);
        dVgsteff_dVd = vec8_blend(dVgsteff_dVd, ((T2 * dT1_dVd) - (T1 * dT2_dVd)) / T3, condmask_false1);
        dVgsteff_dVb = vec8_blend(dVgsteff_dVb, ((T2 * dT1_dVb) - (T1 * dT2_dVb)) / T3, condmask_false1);
      }
    }

  }

  {
    heres[0]->BSIM3Vgsteff = Vgsteff[0];
    heres[1]->BSIM3Vgsteff = Vgsteff[1];
    heres[2]->BSIM3Vgsteff = Vgsteff[2];
    heres[3]->BSIM3Vgsteff = Vgsteff[3];
    heres[4]->BSIM3Vgsteff = Vgsteff[4];
    heres[5]->BSIM3Vgsteff = Vgsteff[5];
    heres[6]->BSIM3Vgsteff = Vgsteff[6];
    heres[7]->BSIM3Vgsteff = Vgsteff[7];
  }
  T9 = sqrtPhis - pParam->BSIM3sqrtPhi;
  Weff = pParam->BSIM3weff - (2.0 * ((pParam->BSIM3dwg * Vgsteff) + (pParam->BSIM3dwb * T9)));
  dWeff_dVg = vec8_SIMDTOVECTOR((-2.0) * pParam->BSIM3dwg);
  dWeff_dVb = ((-2.0) * pParam->BSIM3dwb) * dsqrtPhis_dVb;
  if (1)
  {
    Vec8m condmask0 = Weff < 2.0e-8;
    Vec8m condmask_true0 = condmask0;
    {
      T0 = vec8_blend(T0, 1.0 / (6.0e-8 - (2.0 * Weff)), condmask_true0);
      Weff = vec8_blend(Weff, (2.0e-8 * (4.0e-8 - Weff)) * T0, condmask_true0);
      T0 = vec8_blend(T0, T0 * (T0 * 4.0e-16), condmask_true0);
      dWeff_dVg = vec8_blend(dWeff_dVg, dWeff_dVg * T0, condmask_true0);
      dWeff_dVb = vec8_blend(dWeff_dVb, dWeff_dVb * T0, condmask_true0);
    }
  }

  T0 = (pParam->BSIM3prwg * Vgsteff) + (pParam->BSIM3prwb * T9);
  if (1)
  {
    Vec8m condmask0 = T0 >= (-0.9);
    Vec8m condmask_true0 = condmask0;
    Vec8m condmask_false0 = ~condmask0;
    {
      Rds = vec8_blend(Rds, pParam->BSIM3rds0 * (1.0 + T0), condmask_true0);
      dRds_dVg = vec8_blend(dRds_dVg, vec8_SIMDTOVECTOR(pParam->BSIM3rds0 * pParam->BSIM3prwg), condmask_true0);
      dRds_dVb = vec8_blend(dRds_dVb, (pParam->BSIM3rds0 * pParam->BSIM3prwb) * dsqrtPhis_dVb, condmask_true0);
    }
    {
      T1 = vec8_blend(T1, 1.0 / (17.0 + (20.0 * T0)), condmask_false0);
      Rds = vec8_blend(Rds, (pParam->BSIM3rds0 * (0.8 + T0)) * T1, condmask_false0);
      T1 = vec8_blend(T1, T1 * T1, condmask_false0);
      dRds_dVg = vec8_blend(dRds_dVg, (pParam->BSIM3rds0 * pParam->BSIM3prwg) * T1, condmask_false0);
      dRds_dVb = vec8_blend(dRds_dVb, ((pParam->BSIM3rds0 * pParam->BSIM3prwb) * dsqrtPhis_dVb) * T1, condmask_false0);
    }
  }

  {
    heres[0]->BSIM3rds = Rds[0];
    heres[1]->BSIM3rds = Rds[1];
    heres[2]->BSIM3rds = Rds[2];
    heres[3]->BSIM3rds = Rds[3];
    heres[4]->BSIM3rds = Rds[4];
    heres[5]->BSIM3rds = Rds[5];
    heres[6]->BSIM3rds = Rds[6];
    heres[7]->BSIM3rds = Rds[7];
  }
  T1 = (0.5 * pParam->BSIM3k1ox) / sqrtPhis;
  dT1_dVb = ((-T1) / sqrtPhis) * dsqrtPhis_dVb;
  T9 = vec8_sqrt(pParam->BSIM3xj * Xdep);
  tmp1 = Leff + (2.0 * T9);
  T5 = Leff / tmp1;
  tmp2 = pParam->BSIM3a0 * T5;
  tmp3 = vec8_SIMDTOVECTOR(pParam->BSIM3weff + pParam->BSIM3b1);
  tmp4 = pParam->BSIM3b0 / tmp3;
  T2 = tmp2 + tmp4;
  dT2_dVb = (((-T9) / tmp1) / Xdep) * dXdep_dVb;
  T6 = T5 * T5;
  T7 = T5 * T6;
  Abulk0 = 1.0 + (T1 * T2);
  dAbulk0_dVb = ((T1 * tmp2) * dT2_dVb) + (T2 * dT1_dVb);
  T8 = (pParam->BSIM3ags * pParam->BSIM3a0) * T7;
  dAbulk_dVg = (-T1) * T8;
  Abulk = Abulk0 + (dAbulk_dVg * Vgsteff);
  dAbulk_dVb = dAbulk0_dVb - ((T8 * Vgsteff) * (dT1_dVb + ((3.0 * T1) * dT2_dVb)));
  if (1)
  {
    Vec8m condmask0 = Abulk0 < 0.1;
    Vec8m condmask_true0 = condmask0;
    {
      T9 = vec8_blend(T9, 1.0 / (3.0 - (20.0 * Abulk0)), condmask_true0);
      Abulk0 = vec8_blend(Abulk0, (0.2 - Abulk0) * T9, condmask_true0);
      dAbulk0_dVb = vec8_blend(dAbulk0_dVb, dAbulk0_dVb * (T9 * T9), condmask_true0);
    }
  }

  if (1)
  {
    Vec8m condmask0 = Abulk < 0.1;
    Vec8m condmask_true0 = condmask0;
    {
      T9 = vec8_blend(T9, 1.0 / (3.0 - (20.0 * Abulk)), condmask_true0);
      Abulk = vec8_blend(Abulk, (0.2 - Abulk) * T9, condmask_true0);
      T10 = vec8_blend(T10, T9 * T9, condmask_true0);
      dAbulk_dVb = vec8_blend(dAbulk_dVb, dAbulk_dVb * T10, condmask_true0);
      dAbulk_dVg = vec8_blend(dAbulk_dVg, dAbulk_dVg * T10, condmask_true0);
    }
  }

  {
    heres[0]->BSIM3Abulk = Abulk[0];
    heres[1]->BSIM3Abulk = Abulk[1];
    heres[2]->BSIM3Abulk = Abulk[2];
    heres[3]->BSIM3Abulk = Abulk[3];
    heres[4]->BSIM3Abulk = Abulk[4];
    heres[5]->BSIM3Abulk = Abulk[5];
    heres[6]->BSIM3Abulk = Abulk[6];
    heres[7]->BSIM3Abulk = Abulk[7];
  }
  T2 = pParam->BSIM3keta * Vbseff;
  if (1)
  {
    Vec8m condmask0 = T2 >= (-0.9);
    Vec8m condmask_true0 = condmask0;
    Vec8m condmask_false0 = ~condmask0;
    {
      T0 = vec8_blend(T0, 1.0 / (1.0 + T2), condmask_true0);
      dT0_dVb = vec8_blend(dT0_dVb, ((-pParam->BSIM3keta) * T0) * T0, condmask_true0);
    }
    {
      T1 = vec8_blend(T1, 1.0 / (0.8 + T2), condmask_false0);
      T0 = vec8_blend(T0, (17.0 + (20.0 * T2)) * T1, condmask_false0);
      dT0_dVb = vec8_blend(dT0_dVb, ((-pParam->BSIM3keta) * T1) * T1, condmask_false0);
    }
  }

  dAbulk_dVg *= T0;
  dAbulk_dVb = (dAbulk_dVb * T0) + (Abulk * dT0_dVb);
  dAbulk0_dVb = (dAbulk0_dVb * T0) + (Abulk0 * dT0_dVb);
  Abulk *= T0;
  Abulk0 *= T0;
  if (model->BSIM3mobMod == 1)
  {
    T0 = (Vgsteff + Vth) + Vth;
    T2 = pParam->BSIM3ua + (pParam->BSIM3uc * Vbseff);
    T3 = T0 / model->BSIM3tox;
    T5 = T3 * (T2 + (pParam->BSIM3ub * T3));
    dDenomi_dVg = (T2 + ((2.0 * pParam->BSIM3ub) * T3)) / model->BSIM3tox;
    dDenomi_dVd = (dDenomi_dVg * 2.0) * dVth_dVd;
    dDenomi_dVb = ((dDenomi_dVg * 2.0) * dVth_dVb) + (pParam->BSIM3uc * T3);
  }
  else
    if (model->BSIM3mobMod == 2)
  {
    T5 = (Vgsteff / model->BSIM3tox) * ((pParam->BSIM3ua + (pParam->BSIM3uc * Vbseff)) + ((pParam->BSIM3ub * Vgsteff) / model->BSIM3tox));
    dDenomi_dVg = ((pParam->BSIM3ua + (pParam->BSIM3uc * Vbseff)) + (((2.0 * pParam->BSIM3ub) * Vgsteff) / model->BSIM3tox)) / model->BSIM3tox;
    dDenomi_dVd = vec8_SIMDTOVECTOR(0.0);
    dDenomi_dVb = (Vgsteff * pParam->BSIM3uc) / model->BSIM3tox;
  }
  else
  {
    T0 = (Vgsteff + Vth) + Vth;
    T2 = 1.0 + (pParam->BSIM3uc * Vbseff);
    T3 = T0 / model->BSIM3tox;
    T4 = T3 * (pParam->BSIM3ua + (pParam->BSIM3ub * T3));
    T5 = T4 * T2;
    dDenomi_dVg = ((pParam->BSIM3ua + ((2.0 * pParam->BSIM3ub) * T3)) * T2) / model->BSIM3tox;
    dDenomi_dVd = (dDenomi_dVg * 2.0) * dVth_dVd;
    dDenomi_dVb = ((dDenomi_dVg * 2.0) * dVth_dVb) + (pParam->BSIM3uc * T4);
  }


  if (1)
  {
    Vec8m condmask0 = T5 >= (-0.8);
    Vec8m condmask_true0 = condmask0;
    Vec8m condmask_false0 = ~condmask0;
    {
      Denomi = vec8_blend(Denomi, 1.0 + T5, condmask_true0);
    }
    {
      T9 = vec8_blend(T9, 1.0 / (7.0 + (10.0 * T5)), condmask_false0);
      Denomi = vec8_blend(Denomi, (0.6 + T5) * T9, condmask_false0);
      T9 = vec8_blend(T9, T9 * T9, condmask_false0);
      dDenomi_dVg = vec8_blend(dDenomi_dVg, dDenomi_dVg * T9, condmask_false0);
      dDenomi_dVd = vec8_blend(dDenomi_dVd, dDenomi_dVd * T9, condmask_false0);
      dDenomi_dVb = vec8_blend(dDenomi_dVb, dDenomi_dVb * T9, condmask_false0);
    }
  }

  {
    Vec8d val = ueff = ((Vec8d ){heres[0]->BSIM3u0temp, heres[1]->BSIM3u0temp, heres[2]->BSIM3u0temp, heres[3]->BSIM3u0temp, heres[4]->BSIM3u0temp, heres[5]->BSIM3u0temp, heres[6]->BSIM3u0temp, heres[7]->BSIM3u0temp}) / Denomi;
    heres[0]->BSIM3ueff = val[0];
    heres[1]->BSIM3ueff = val[1];
    heres[2]->BSIM3ueff = val[2];
    heres[3]->BSIM3ueff = val[3];
    heres[4]->BSIM3ueff = val[4];
    heres[5]->BSIM3ueff = val[5];
    heres[6]->BSIM3ueff = val[6];
    heres[7]->BSIM3ueff = val[7];
  }
  T9 = (-ueff) / Denomi;
  dueff_dVg = T9 * dDenomi_dVg;
  dueff_dVd = T9 * dDenomi_dVd;
  dueff_dVb = T9 * dDenomi_dVb;
  WVCox = (Weff * pParam->BSIM3vsattemp) * model->BSIM3cox;
  WVCoxRds = WVCox * Rds;
  Esat = (2.0 * pParam->BSIM3vsattemp) / ueff;
  EsatL = Esat * Leff;
  T0 = (-EsatL) / ueff;
  dEsatL_dVg = T0 * dueff_dVg;
  dEsatL_dVd = T0 * dueff_dVd;
  dEsatL_dVb = T0 * dueff_dVb;
  a1 = pParam->BSIM3a1;
  if (a1 == 0.0)
  {
    Lambda = vec8_SIMDTOVECTOR(pParam->BSIM3a2);
    dLambda_dVg = vec8_SIMDTOVECTOR(0.0);
  }
  else
    if (a1 > 0.0)
  {
    T0 = vec8_SIMDTOVECTOR(1.0 - pParam->BSIM3a2);
    T1 = (T0 - (pParam->BSIM3a1 * Vgsteff)) - 0.0001;
    T2 = vec8_sqrt((T1 * T1) + (0.0004 * T0));
    Lambda = (pParam->BSIM3a2 + T0) - (0.5 * (T1 + T2));
    dLambda_dVg = (0.5 * pParam->BSIM3a1) * (1.0 + (T1 / T2));
  }
  else
  {
    T1 = (pParam->BSIM3a2 + (pParam->BSIM3a1 * Vgsteff)) - 0.0001;
    T2 = vec8_sqrt((T1 * T1) + (0.0004 * pParam->BSIM3a2));
    Lambda = 0.5 * (T1 + T2);
    dLambda_dVg = (0.5 * pParam->BSIM3a1) * (1.0 + (T1 / T2));
  }


  Vgst2Vtm = Vgsteff + (2.0 * Vtm);
  {
    Vec8d val = Abulk / Vgst2Vtm;
    heres[0]->BSIM3AbovVgst2Vtm = val[0];
    heres[1]->BSIM3AbovVgst2Vtm = val[1];
    heres[2]->BSIM3AbovVgst2Vtm = val[2];
    heres[3]->BSIM3AbovVgst2Vtm = val[3];
    heres[4]->BSIM3AbovVgst2Vtm = val[4];
    heres[5]->BSIM3AbovVgst2Vtm = val[5];
    heres[6]->BSIM3AbovVgst2Vtm = val[6];
    heres[7]->BSIM3AbovVgst2Vtm = val[7];
  }
  if (1)
  {
    Vec8m condmask0 = Rds > 0;
    Vec8m condmask_true0 = condmask0;
    Vec8m condmask_false0 = ~condmask0;
    {
      tmp2 = vec8_blend(tmp2, (dRds_dVg / Rds) + (dWeff_dVg / Weff), condmask_true0);
      tmp3 = vec8_blend(tmp3, (dRds_dVb / Rds) + (dWeff_dVb / Weff), condmask_true0);
    }
    {
      tmp2 = vec8_blend(tmp2, dWeff_dVg / Weff, condmask_false0);
      tmp3 = vec8_blend(tmp3, dWeff_dVb / Weff, condmask_false0);
    }
  }

  if (1)
  {
    Vec8m condmask0 = (Rds == 0.0) & (Lambda == 1.0);
    Vec8m condmask_true0 = condmask0;
    Vec8m condmask_false0 = ~condmask0;
    {
      T0 = vec8_blend(T0, 1.0 / ((Abulk * EsatL) + Vgst2Vtm), condmask_true0);
      tmp1 = vec8_blend(tmp1, vec8_SIMDTOVECTOR(0.0), condmask_true0);
      T1 = vec8_blend(T1, T0 * T0, condmask_true0);
      T2 = vec8_blend(T2, Vgst2Vtm * T0, condmask_true0);
      T3 = vec8_blend(T3, EsatL * Vgst2Vtm, condmask_true0);
      Vdsat = vec8_blend(Vdsat, T3 * T0, condmask_true0);
      dT0_dVg = vec8_blend(dT0_dVg, (-(((Abulk * dEsatL_dVg) + (EsatL * dAbulk_dVg)) + 1.0)) * T1, condmask_true0);
      dT0_dVd = vec8_blend(dT0_dVd, (-(Abulk * dEsatL_dVd)) * T1, condmask_true0);
      dT0_dVb = vec8_blend(dT0_dVb, (-((Abulk * dEsatL_dVb) + (dAbulk_dVb * EsatL))) * T1, condmask_true0);
      dVdsat_dVg = vec8_blend(dVdsat_dVg, ((T3 * dT0_dVg) + (T2 * dEsatL_dVg)) + (EsatL * T0), condmask_true0);
      dVdsat_dVd = vec8_blend(dVdsat_dVd, (T3 * dT0_dVd) + (T2 * dEsatL_dVd), condmask_true0);
      dVdsat_dVb = vec8_blend(dVdsat_dVb, (T3 * dT0_dVb) + (T2 * dEsatL_dVb), condmask_true0);
    }
    {
      tmp1 = vec8_blend(tmp1, dLambda_dVg / (Lambda * Lambda), condmask_false0);
      T9 = vec8_blend(T9, Abulk * WVCoxRds, condmask_false0);
      T8 = vec8_blend(T8, Abulk * T9, condmask_false0);
      T7 = vec8_blend(T7, Vgst2Vtm * T9, condmask_false0);
      T6 = vec8_blend(T6, Vgst2Vtm * WVCoxRds, condmask_false0);
      T0 = vec8_blend(T0, (2.0 * Abulk) * ((T9 - 1.0) + (1.0 / Lambda)), condmask_false0);
      dT0_dVg = vec8_blend(dT0_dVg, 2.0 * (((T8 * tmp2) - (Abulk * tmp1)) + ((((2.0 * T9) + (1.0 / Lambda)) - 1.0) * dAbulk_dVg)), condmask_false0);
      dT0_dVb = vec8_blend(dT0_dVb, 2.0 * ((T8 * (((2.0 / Abulk) * dAbulk_dVb) + tmp3)) + (((1.0 / Lambda) - 1.0) * dAbulk_dVb)), condmask_false0);
      dT0_dVd = vec8_blend(dT0_dVd, vec8_SIMDTOVECTOR(0.0), condmask_false0);
      T1 = vec8_blend(T1, ((Vgst2Vtm * ((2.0 / Lambda) - 1.0)) + (Abulk * EsatL)) + (3.0 * T7), condmask_false0);
      dT1_dVg = vec8_blend(dT1_dVg, (((((2.0 / Lambda) - 1.0) - ((2.0 * Vgst2Vtm) * tmp1)) + (Abulk * dEsatL_dVg)) + (EsatL * dAbulk_dVg)) + (3.0 * ((T9 + (T7 * tmp2)) + (T6 * dAbulk_dVg))), condmask_false0);
      dT1_dVb = vec8_blend(dT1_dVb, ((Abulk * dEsatL_dVb) + (EsatL * dAbulk_dVb)) + (3.0 * ((T6 * dAbulk_dVb) + (T7 * tmp3))), condmask_false0);
      dT1_dVd = vec8_blend(dT1_dVd, Abulk * dEsatL_dVd, condmask_false0);
      T2 = vec8_blend(T2, Vgst2Vtm * (EsatL + (2.0 * T6)), condmask_false0);
      dT2_dVg = vec8_blend(dT2_dVg, (EsatL + (Vgst2Vtm * dEsatL_dVg)) + (T6 * (4.0 + ((2.0 * Vgst2Vtm) * tmp2))), condmask_false0);
      dT2_dVb = vec8_blend(dT2_dVb, Vgst2Vtm * (dEsatL_dVb + ((2.0 * T6) * tmp3)), condmask_false0);
      dT2_dVd = vec8_blend(dT2_dVd, Vgst2Vtm * dEsatL_dVd, condmask_false0);
      T3 = vec8_blend(T3, vec8_sqrt((T1 * T1) - ((2.0 * T0) * T2)), condmask_false0);
      Vdsat = vec8_blend(Vdsat, (T1 - T3) / T0, condmask_false0);
      dT3_dVg = vec8_blend(dT3_dVg, ((T1 * dT1_dVg) - (2.0 * ((T0 * dT2_dVg) + (T2 * dT0_dVg)))) / T3, condmask_false0);
      dT3_dVd = vec8_blend(dT3_dVd, ((T1 * dT1_dVd) - (2.0 * ((T0 * dT2_dVd) + (T2 * dT0_dVd)))) / T3, condmask_false0);
      dT3_dVb = vec8_blend(dT3_dVb, ((T1 * dT1_dVb) - (2.0 * ((T0 * dT2_dVb) + (T2 * dT0_dVb)))) / T3, condmask_false0);
      dVdsat_dVg = vec8_blend(dVdsat_dVg, ((dT1_dVg - ((((T1 * dT1_dVg) - (dT0_dVg * T2)) - (T0 * dT2_dVg)) / T3)) - (Vdsat * dT0_dVg)) / T0, condmask_false0);
      dVdsat_dVb = vec8_blend(dVdsat_dVb, ((dT1_dVb - ((((T1 * dT1_dVb) - (dT0_dVb * T2)) - (T0 * dT2_dVb)) / T3)) - (Vdsat * dT0_dVb)) / T0, condmask_false0);
      dVdsat_dVd = vec8_blend(dVdsat_dVd, (dT1_dVd - (((T1 * dT1_dVd) - (T0 * dT2_dVd)) / T3)) / T0, condmask_false0);
    }
  }

  {
    heres[0]->BSIM3vdsat = Vdsat[0];
    heres[1]->BSIM3vdsat = Vdsat[1];
    heres[2]->BSIM3vdsat = Vdsat[2];
    heres[3]->BSIM3vdsat = Vdsat[3];
    heres[4]->BSIM3vdsat = Vdsat[4];
    heres[5]->BSIM3vdsat = Vdsat[5];
    heres[6]->BSIM3vdsat = Vdsat[6];
    heres[7]->BSIM3vdsat = Vdsat[7];
  }
  T1 = (Vdsat - Vds) - pParam->BSIM3delta;
  dT1_dVg = dVdsat_dVg;
  dT1_dVd = dVdsat_dVd - 1.0;
  dT1_dVb = dVdsat_dVb;
  T2 = vec8_sqrt((T1 * T1) + ((4.0 * pParam->BSIM3delta) * Vdsat));
  T0 = T1 / T2;
  T3 = (2.0 * pParam->BSIM3delta) / T2;
  dT2_dVg = (T0 * dT1_dVg) + (T3 * dVdsat_dVg);
  dT2_dVd = (T0 * dT1_dVd) + (T3 * dVdsat_dVd);
  dT2_dVb = (T0 * dT1_dVb) + (T3 * dVdsat_dVb);
  Vdseff = Vdsat - (0.5 * (T1 + T2));
  dVdseff_dVg = dVdsat_dVg - (0.5 * (dT1_dVg + dT2_dVg));
  dVdseff_dVd = dVdsat_dVd - (0.5 * (dT1_dVd + dT2_dVd));
  dVdseff_dVb = dVdsat_dVb - (0.5 * (dT1_dVb + dT2_dVb));
  if (1)
  {
    Vec8m condmask0 = Vds == 0.0;
    Vec8m condmask_true0 = condmask0;
    {
      Vdseff = vec8_blend(Vdseff, vec8_SIMDTOVECTOR(0.0), condmask_true0);
      dVdseff_dVg = vec8_blend(dVdseff_dVg, vec8_SIMDTOVECTOR(0.0), condmask_true0);
      dVdseff_dVb = vec8_blend(dVdseff_dVb, vec8_SIMDTOVECTOR(0.0), condmask_true0);
    }
  }

  tmp4 = 1.0 - (((0.5 * Abulk) * Vdsat) / Vgst2Vtm);
  T9 = WVCoxRds * Vgsteff;
  T8 = T9 / Vgst2Vtm;
  T0 = (EsatL + Vdsat) + ((2.0 * T9) * tmp4);
  T7 = (2.0 * WVCoxRds) * tmp4;
  dT0_dVg = ((dEsatL_dVg + dVdsat_dVg) + (T7 * (1.0 + (tmp2 * Vgsteff)))) - (T8 * (((Abulk * dVdsat_dVg) - ((Abulk * Vdsat) / Vgst2Vtm)) + (Vdsat * dAbulk_dVg)));
  dT0_dVb = ((dEsatL_dVb + dVdsat_dVb) + ((T7 * tmp3) * Vgsteff)) - (T8 * ((dAbulk_dVb * Vdsat) + (Abulk * dVdsat_dVb)));
  dT0_dVd = (dEsatL_dVd + dVdsat_dVd) - ((T8 * Abulk) * dVdsat_dVd);
  T9 = WVCoxRds * Abulk;
  T1 = ((2.0 / Lambda) - 1.0) + T9;
  dT1_dVg = ((-2.0) * tmp1) + (WVCoxRds * ((Abulk * tmp2) + dAbulk_dVg));
  dT1_dVb = (dAbulk_dVb * WVCoxRds) + (T9 * tmp3);
  Vasat = T0 / T1;
  dVasat_dVg = (dT0_dVg - (Vasat * dT1_dVg)) / T1;
  dVasat_dVb = (dT0_dVb - (Vasat * dT1_dVb)) / T1;
  dVasat_dVd = dT0_dVd / T1;
  if (1)
  {
    Vec8m condmask0 = Vdseff > Vds;
    Vec8m condmask_true0 = condmask0;
    Vdseff = vec8_blend(Vdseff, Vds, condmask_true0);
  }

  diffVds = Vds - Vdseff;
  {
    heres[0]->BSIM3Vdseff = Vdseff[0];
    heres[1]->BSIM3Vdseff = Vdseff[1];
    heres[2]->BSIM3Vdseff = Vdseff[2];
    heres[3]->BSIM3Vdseff = Vdseff[3];
    heres[4]->BSIM3Vdseff = Vdseff[4];
    heres[5]->BSIM3Vdseff = Vdseff[5];
    heres[6]->BSIM3Vdseff = Vdseff[6];
    heres[7]->BSIM3Vdseff = Vdseff[7];
  }
  VACLM = vec8_SIMDTOVECTOR(MAX_EXP);
  dVACLM_dVd = (dVACLM_dVg = (dVACLM_dVb = vec8_SIMDTOVECTOR(0.0)));
  if (pParam->BSIM3pclm > 0.0)
    if (1)
  {
    Vec8m condmask0 = diffVds > 1.0e-10;
    Vec8m condmask_true0 = condmask0;
    {
      T0 = vec8_blend(T0, 1.0 / ((pParam->BSIM3pclm * Abulk) * pParam->BSIM3litl), condmask_true0);
      dT0_dVb = vec8_blend(dT0_dVb, ((-T0) / Abulk) * dAbulk_dVb, condmask_true0);
      dT0_dVg = vec8_blend(dT0_dVg, ((-T0) / Abulk) * dAbulk_dVg, condmask_true0);
      T2 = vec8_blend(T2, Vgsteff / EsatL, condmask_true0);
      T1 = vec8_blend(T1, Leff * (Abulk + T2), condmask_true0);
      dT1_dVg = vec8_blend(dT1_dVg, Leff * (((1.0 - (T2 * dEsatL_dVg)) / EsatL) + dAbulk_dVg), condmask_true0);
      dT1_dVb = vec8_blend(dT1_dVb, Leff * (dAbulk_dVb - ((T2 * dEsatL_dVb) / EsatL)), condmask_true0);
      dT1_dVd = vec8_blend(dT1_dVd, ((-T2) * dEsatL_dVd) / Esat, condmask_true0);
      T9 = vec8_blend(T9, T0 * T1, condmask_true0);
      VACLM = vec8_blend(VACLM, T9 * diffVds, condmask_true0);
      dVACLM_dVg = vec8_blend(dVACLM_dVg, (((T0 * dT1_dVg) * diffVds) - (T9 * dVdseff_dVg)) + ((T1 * diffVds) * dT0_dVg), condmask_true0);
      dVACLM_dVb = vec8_blend(dVACLM_dVb, (((dT0_dVb * T1) + (T0 * dT1_dVb)) * diffVds) - (T9 * dVdseff_dVb), condmask_true0);
      dVACLM_dVd = vec8_blend(dVACLM_dVd, ((T0 * dT1_dVd) * diffVds) + (T9 * (1.0 - dVdseff_dVd)), condmask_true0);
    }
  }


  if (pParam->BSIM3thetaRout > 0.0)
  {
    T8 = Abulk * Vdsat;
    T0 = Vgst2Vtm * T8;
    dT0_dVg = (((Vgst2Vtm * Abulk) * dVdsat_dVg) + T8) + ((Vgst2Vtm * Vdsat) * dAbulk_dVg);
    dT0_dVb = Vgst2Vtm * ((dAbulk_dVb * Vdsat) + (Abulk * dVdsat_dVb));
    dT0_dVd = (Vgst2Vtm * Abulk) * dVdsat_dVd;
    T1 = Vgst2Vtm + T8;
    dT1_dVg = (1.0 + (Abulk * dVdsat_dVg)) + (Vdsat * dAbulk_dVg);
    dT1_dVb = (Abulk * dVdsat_dVb) + (dAbulk_dVb * Vdsat);
    dT1_dVd = Abulk * dVdsat_dVd;
    T9 = T1 * T1;
    T2 = vec8_SIMDTOVECTOR(pParam->BSIM3thetaRout);
    VADIBL = (Vgst2Vtm - (T0 / T1)) / T2;
    dVADIBL_dVg = ((1.0 - (dT0_dVg / T1)) + ((T0 * dT1_dVg) / T9)) / T2;
    dVADIBL_dVb = (((-dT0_dVb) / T1) + ((T0 * dT1_dVb) / T9)) / T2;
    dVADIBL_dVd = (((-dT0_dVd) / T1) + ((T0 * dT1_dVd) / T9)) / T2;
    T7 = pParam->BSIM3pdiblb * Vbseff;
    if (1)
    {
      Vec8m condmask0 = T7 >= (-0.9);
      Vec8m condmask_true0 = condmask0;
      Vec8m condmask_false0 = ~condmask0;
      {
        T3 = vec8_blend(T3, 1.0 / (1.0 + T7), condmask_true0);
        VADIBL = vec8_blend(VADIBL, VADIBL * T3, condmask_true0);
        dVADIBL_dVg = vec8_blend(dVADIBL_dVg, dVADIBL_dVg * T3, condmask_true0);
        dVADIBL_dVb = vec8_blend(dVADIBL_dVb, (dVADIBL_dVb - (VADIBL * pParam->BSIM3pdiblb)) * T3, condmask_true0);
        dVADIBL_dVd = vec8_blend(dVADIBL_dVd, dVADIBL_dVd * T3, condmask_true0);
      }
      {
        T4 = vec8_blend(T4, 1.0 / (0.8 + T7), condmask_false0);
        T3 = vec8_blend(T3, (17.0 + (20.0 * T7)) * T4, condmask_false0);
        dVADIBL_dVg = vec8_blend(dVADIBL_dVg, dVADIBL_dVg * T3, condmask_false0);
        dVADIBL_dVb = vec8_blend(dVADIBL_dVb, (dVADIBL_dVb * T3) - (((VADIBL * pParam->BSIM3pdiblb) * T4) * T4), condmask_false0);
        dVADIBL_dVd = vec8_blend(dVADIBL_dVd, dVADIBL_dVd * T3, condmask_false0);
        VADIBL = vec8_blend(VADIBL, VADIBL * T3, condmask_false0);
      }
    }

  }
  else
  {
    VADIBL = vec8_SIMDTOVECTOR(MAX_EXP);
    dVADIBL_dVd = (dVADIBL_dVg = (dVADIBL_dVb = vec8_SIMDTOVECTOR(0.0)));
  }

  T8 = pParam->BSIM3pvag / EsatL;
  T9 = T8 * Vgsteff;
  if (1)
  {
    Vec8m condmask0 = T9 > (-0.9);
    Vec8m condmask_true0 = condmask0;
    Vec8m condmask_false0 = ~condmask0;
    {
      T0 = vec8_blend(T0, 1.0 + T9, condmask_true0);
      dT0_dVg = vec8_blend(dT0_dVg, T8 * (1.0 - ((Vgsteff * dEsatL_dVg) / EsatL)), condmask_true0);
      dT0_dVb = vec8_blend(dT0_dVb, ((-T9) * dEsatL_dVb) / EsatL, condmask_true0);
      dT0_dVd = vec8_blend(dT0_dVd, ((-T9) * dEsatL_dVd) / EsatL, condmask_true0);
    }
    {
      T1 = vec8_blend(T1, 1.0 / (17.0 + (20.0 * T9)), condmask_false0);
      T0 = vec8_blend(T0, (0.8 + T9) * T1, condmask_false0);
      T1 = vec8_blend(T1, T1 * T1, condmask_false0);
      dT0_dVg = vec8_blend(dT0_dVg, (T8 * (1.0 - ((Vgsteff * dEsatL_dVg) / EsatL))) * T1, condmask_false0);
      T9 = vec8_blend(T9, T9 * (T1 / EsatL), condmask_false0);
      dT0_dVb = vec8_blend(dT0_dVb, (-T9) * dEsatL_dVb, condmask_false0);
      dT0_dVd = vec8_blend(dT0_dVd, (-T9) * dEsatL_dVd, condmask_false0);
    }
  }

  tmp1 = VACLM * VACLM;
  tmp2 = VADIBL * VADIBL;
  tmp3 = VACLM + VADIBL;
  T1 = (VACLM * VADIBL) / tmp3;
  tmp3 *= tmp3;
  dT1_dVg = ((tmp1 * dVADIBL_dVg) + (tmp2 * dVACLM_dVg)) / tmp3;
  dT1_dVd = ((tmp1 * dVADIBL_dVd) + (tmp2 * dVACLM_dVd)) / tmp3;
  dT1_dVb = ((tmp1 * dVADIBL_dVb) + (tmp2 * dVACLM_dVb)) / tmp3;
  Va = Vasat + (T0 * T1);
  dVa_dVg = (dVasat_dVg + (T1 * dT0_dVg)) + (T0 * dT1_dVg);
  dVa_dVd = (dVasat_dVd + (T1 * dT0_dVd)) + (T0 * dT1_dVd);
  dVa_dVb = (dVasat_dVb + (T1 * dT0_dVb)) + (T0 * dT1_dVb);
  VASCBE = vec8_SIMDTOVECTOR(MAX_EXP);
  dVASCBE_dVg = (dVASCBE_dVd = (dVASCBE_dVb = vec8_SIMDTOVECTOR(0.0)));
  if (pParam->BSIM3pscbe2 > 0.0)
  {
    if (1)
    {
      Vec8m condmask0 = diffVds > ((pParam->BSIM3pscbe1 * pParam->BSIM3litl) / EXP_THRESHOLD);
      Vec8m condmask_true0 = condmask0;
      Vec8m condmask_false0 = ~condmask0;
      {
        T0 = vec8_blend(T0, (pParam->BSIM3pscbe1 * pParam->BSIM3litl) / diffVds, condmask_true0);
        VASCBE = vec8_blend(VASCBE, (Leff * vec8_exp(T0)) / pParam->BSIM3pscbe2, condmask_true0);
        T1 = vec8_blend(T1, (T0 * VASCBE) / diffVds, condmask_true0);
        dVASCBE_dVg = vec8_blend(dVASCBE_dVg, T1 * dVdseff_dVg, condmask_true0);
        dVASCBE_dVd = vec8_blend(dVASCBE_dVd, (-T1) * (1.0 - dVdseff_dVd), condmask_true0);
        dVASCBE_dVb = vec8_blend(dVASCBE_dVb, T1 * dVdseff_dVb, condmask_true0);
      }
      {
        VASCBE = vec8_blend(VASCBE, vec8_SIMDTOVECTOR((MAX_EXP * Leff) / pParam->BSIM3pscbe2), condmask_false0);
      }
    }

  }

  CoxWovL = (model->BSIM3cox * Weff) / Leff;
  beta = ueff * CoxWovL;
  dbeta_dVg = (CoxWovL * dueff_dVg) + ((beta * dWeff_dVg) / Weff);
  dbeta_dVd = CoxWovL * dueff_dVd;
  dbeta_dVb = (CoxWovL * dueff_dVb) + ((beta * dWeff_dVb) / Weff);
  T0 = 1.0 - (((0.5 * Abulk) * Vdseff) / Vgst2Vtm);
  dT0_dVg = ((-0.5) * (((Abulk * dVdseff_dVg) - ((Abulk * Vdseff) / Vgst2Vtm)) + (Vdseff * dAbulk_dVg))) / Vgst2Vtm;
  dT0_dVd = (((-0.5) * Abulk) * dVdseff_dVd) / Vgst2Vtm;
  dT0_dVb = ((-0.5) * ((Abulk * dVdseff_dVb) + (dAbulk_dVb * Vdseff))) / Vgst2Vtm;
  fgche1 = Vgsteff * T0;
  dfgche1_dVg = (Vgsteff * dT0_dVg) + T0;
  dfgche1_dVd = Vgsteff * dT0_dVd;
  dfgche1_dVb = Vgsteff * dT0_dVb;
  T9 = Vdseff / EsatL;
  fgche2 = 1.0 + T9;
  dfgche2_dVg = (dVdseff_dVg - (T9 * dEsatL_dVg)) / EsatL;
  dfgche2_dVd = (dVdseff_dVd - (T9 * dEsatL_dVd)) / EsatL;
  dfgche2_dVb = (dVdseff_dVb - (T9 * dEsatL_dVb)) / EsatL;
  gche = (beta * fgche1) / fgche2;
  dgche_dVg = (((beta * dfgche1_dVg) + (fgche1 * dbeta_dVg)) - (gche * dfgche2_dVg)) / fgche2;
  dgche_dVd = (((beta * dfgche1_dVd) + (fgche1 * dbeta_dVd)) - (gche * dfgche2_dVd)) / fgche2;
  dgche_dVb = (((beta * dfgche1_dVb) + (fgche1 * dbeta_dVb)) - (gche * dfgche2_dVb)) / fgche2;
  T0 = 1.0 + (gche * Rds);
  T9 = Vdseff / T0;
  Idl = gche * T9;
  dIdl_dVg = (((gche * dVdseff_dVg) + (T9 * dgche_dVg)) / T0) - (((Idl * gche) / T0) * dRds_dVg);
  dIdl_dVd = ((gche * dVdseff_dVd) + (T9 * dgche_dVd)) / T0;
  dIdl_dVb = (((gche * dVdseff_dVb) + (T9 * dgche_dVb)) - ((Idl * dRds_dVb) * gche)) / T0;
  T9 = diffVds / Va;
  T0 = 1.0 + T9;
  Idsa = Idl * T0;
  dIdsa_dVg = (T0 * dIdl_dVg) - ((Idl * (dVdseff_dVg + (T9 * dVa_dVg))) / Va);
  dIdsa_dVd = (T0 * dIdl_dVd) + ((Idl * ((1.0 - dVdseff_dVd) - (T9 * dVa_dVd))) / Va);
  dIdsa_dVb = (T0 * dIdl_dVb) - ((Idl * (dVdseff_dVb + (T9 * dVa_dVb))) / Va);
  T9 = diffVds / VASCBE;
  T0 = 1.0 + T9;
  Ids = Idsa * T0;
  Gm = (T0 * dIdsa_dVg) - ((Idsa * (dVdseff_dVg + (T9 * dVASCBE_dVg))) / VASCBE);
  Gds = (T0 * dIdsa_dVd) + ((Idsa * ((1.0 - dVdseff_dVd) - (T9 * dVASCBE_dVd))) / VASCBE);
  Gmb = (T0 * dIdsa_dVb) - ((Idsa * (dVdseff_dVb + (T9 * dVASCBE_dVb))) / VASCBE);
  Gds += Gm * dVgsteff_dVd;
  Gmb += Gm * dVgsteff_dVb;
  Gm *= dVgsteff_dVg;
  Gmb *= dVbseff_dVb;
  tmpuni = pParam->BSIM3alpha0 + (pParam->BSIM3alpha1 * Leff);
  if ((tmpuni <= 0.0) || (pParam->BSIM3beta0 <= 0.0))
  {
    Isub = (Gbd = (Gbb = (Gbg = vec8_SIMDTOVECTOR(0.0))));
  }
  else
  {
    T2 = vec8_SIMDTOVECTOR(tmpuni / Leff);
    if (1)
    {
      Vec8m condmask0 = diffVds > (pParam->BSIM3beta0 / EXP_THRESHOLD);
      Vec8m condmask_true0 = condmask0;
      Vec8m condmask_false0 = ~condmask0;
      {
        T0 = vec8_blend(T0, (-pParam->BSIM3beta0) / diffVds, condmask_true0);
        T1 = vec8_blend(T1, (T2 * diffVds) * vec8_exp(T0), condmask_true0);
        T3 = vec8_blend(T3, (T1 / diffVds) * (T0 - 1.0), condmask_true0);
        dT1_dVg = vec8_blend(dT1_dVg, T3 * dVdseff_dVg, condmask_true0);
        dT1_dVd = vec8_blend(dT1_dVd, T3 * (dVdseff_dVd - 1.0), condmask_true0);
        dT1_dVb = vec8_blend(dT1_dVb, T3 * dVdseff_dVb, condmask_true0);
      }
      {
        T3 = vec8_blend(T3, T2 * MIN_EXP, condmask_false0);
        T1 = vec8_blend(T1, T3 * diffVds, condmask_false0);
        dT1_dVg = vec8_blend(dT1_dVg, (-T3) * dVdseff_dVg, condmask_false0);
        dT1_dVd = vec8_blend(dT1_dVd, T3 * (1.0 - dVdseff_dVd), condmask_false0);
        dT1_dVb = vec8_blend(dT1_dVb, (-T3) * dVdseff_dVb, condmask_false0);
      }
    }

    Isub = T1 * Idsa;
    Gbg = (T1 * dIdsa_dVg) + (Idsa * dT1_dVg);
    Gbd = (T1 * dIdsa_dVd) + (Idsa * dT1_dVd);
    Gbb = (T1 * dIdsa_dVb) + (Idsa * dT1_dVb);
    Gbd += Gbg * dVgsteff_dVd;
    Gbb += Gbg * dVgsteff_dVb;
    Gbg *= dVgsteff_dVg;
    Gbb *= dVbseff_dVb;
  }

  cdrain = Ids;
  {
    heres[0]->BSIM3gds = Gds[0];
    heres[1]->BSIM3gds = Gds[1];
    heres[2]->BSIM3gds = Gds[2];
    heres[3]->BSIM3gds = Gds[3];
    heres[4]->BSIM3gds = Gds[4];
    heres[5]->BSIM3gds = Gds[5];
    heres[6]->BSIM3gds = Gds[6];
    heres[7]->BSIM3gds = Gds[7];
  }
  {
    heres[0]->BSIM3gm = Gm[0];
    heres[1]->BSIM3gm = Gm[1];
    heres[2]->BSIM3gm = Gm[2];
    heres[3]->BSIM3gm = Gm[3];
    heres[4]->BSIM3gm = Gm[4];
    heres[5]->BSIM3gm = Gm[5];
    heres[6]->BSIM3gm = Gm[6];
    heres[7]->BSIM3gm = Gm[7];
  }
  {
    heres[0]->BSIM3gmbs = Gmb[0];
    heres[1]->BSIM3gmbs = Gmb[1];
    heres[2]->BSIM3gmbs = Gmb[2];
    heres[3]->BSIM3gmbs = Gmb[3];
    heres[4]->BSIM3gmbs = Gmb[4];
    heres[5]->BSIM3gmbs = Gmb[5];
    heres[6]->BSIM3gmbs = Gmb[6];
    heres[7]->BSIM3gmbs = Gmb[7];
  }
  {
    heres[0]->BSIM3gbbs = Gbb[0];
    heres[1]->BSIM3gbbs = Gbb[1];
    heres[2]->BSIM3gbbs = Gbb[2];
    heres[3]->BSIM3gbbs = Gbb[3];
    heres[4]->BSIM3gbbs = Gbb[4];
    heres[5]->BSIM3gbbs = Gbb[5];
    heres[6]->BSIM3gbbs = Gbb[6];
    heres[7]->BSIM3gbbs = Gbb[7];
  }
  {
    heres[0]->BSIM3gbgs = Gbg[0];
    heres[1]->BSIM3gbgs = Gbg[1];
    heres[2]->BSIM3gbgs = Gbg[2];
    heres[3]->BSIM3gbgs = Gbg[3];
    heres[4]->BSIM3gbgs = Gbg[4];
    heres[5]->BSIM3gbgs = Gbg[5];
    heres[6]->BSIM3gbgs = Gbg[6];
    heres[7]->BSIM3gbgs = Gbg[7];
  }
  {
    heres[0]->BSIM3gbds = Gbd[0];
    heres[1]->BSIM3gbds = Gbd[1];
    heres[2]->BSIM3gbds = Gbd[2];
    heres[3]->BSIM3gbds = Gbd[3];
    heres[4]->BSIM3gbds = Gbd[4];
    heres[5]->BSIM3gbds = Gbd[5];
    heres[6]->BSIM3gbds = Gbd[6];
    heres[7]->BSIM3gbds = Gbd[7];
  }
  {
    heres[0]->BSIM3csub = Isub[0];
    heres[1]->BSIM3csub = Isub[1];
    heres[2]->BSIM3csub = Isub[2];
    heres[3]->BSIM3csub = Isub[3];
    heres[4]->BSIM3csub = Isub[4];
    heres[5]->BSIM3csub = Isub[5];
    heres[6]->BSIM3csub = Isub[6];
    heres[7]->BSIM3csub = Isub[7];
  }
  if ((model->BSIM3xpart < 0) || (!ChargeComputationNeeded))
  {
    qgate = (qdrn = (qsrc = (qbulk = vec8_SIMDTOVECTOR(0.0))));
    {
      heres[0]->BSIM3cggb = 0.0;
      heres[1]->BSIM3cggb = 0.0;
      heres[2]->BSIM3cggb = 0.0;
      heres[3]->BSIM3cggb = 0.0;
      heres[4]->BSIM3cggb = 0.0;
      heres[5]->BSIM3cggb = 0.0;
      heres[6]->BSIM3cggb = 0.0;
      heres[7]->BSIM3cggb = 0.0;
    }
    {
      heres[0]->BSIM3cgsb = 0.0;
      heres[1]->BSIM3cgsb = 0.0;
      heres[2]->BSIM3cgsb = 0.0;
      heres[3]->BSIM3cgsb = 0.0;
      heres[4]->BSIM3cgsb = 0.0;
      heres[5]->BSIM3cgsb = 0.0;
      heres[6]->BSIM3cgsb = 0.0;
      heres[7]->BSIM3cgsb = 0.0;
    }
    {
      heres[0]->BSIM3cgdb = 0.0;
      heres[1]->BSIM3cgdb = 0.0;
      heres[2]->BSIM3cgdb = 0.0;
      heres[3]->BSIM3cgdb = 0.0;
      heres[4]->BSIM3cgdb = 0.0;
      heres[5]->BSIM3cgdb = 0.0;
      heres[6]->BSIM3cgdb = 0.0;
      heres[7]->BSIM3cgdb = 0.0;
    }
    {
      heres[0]->BSIM3cdgb = 0.0;
      heres[1]->BSIM3cdgb = 0.0;
      heres[2]->BSIM3cdgb = 0.0;
      heres[3]->BSIM3cdgb = 0.0;
      heres[4]->BSIM3cdgb = 0.0;
      heres[5]->BSIM3cdgb = 0.0;
      heres[6]->BSIM3cdgb = 0.0;
      heres[7]->BSIM3cdgb = 0.0;
    }
    {
      heres[0]->BSIM3cdsb = 0.0;
      heres[1]->BSIM3cdsb = 0.0;
      heres[2]->BSIM3cdsb = 0.0;
      heres[3]->BSIM3cdsb = 0.0;
      heres[4]->BSIM3cdsb = 0.0;
      heres[5]->BSIM3cdsb = 0.0;
      heres[6]->BSIM3cdsb = 0.0;
      heres[7]->BSIM3cdsb = 0.0;
    }
    {
      heres[0]->BSIM3cddb = 0.0;
      heres[1]->BSIM3cddb = 0.0;
      heres[2]->BSIM3cddb = 0.0;
      heres[3]->BSIM3cddb = 0.0;
      heres[4]->BSIM3cddb = 0.0;
      heres[5]->BSIM3cddb = 0.0;
      heres[6]->BSIM3cddb = 0.0;
      heres[7]->BSIM3cddb = 0.0;
    }
    {
      heres[0]->BSIM3cbgb = 0.0;
      heres[1]->BSIM3cbgb = 0.0;
      heres[2]->BSIM3cbgb = 0.0;
      heres[3]->BSIM3cbgb = 0.0;
      heres[4]->BSIM3cbgb = 0.0;
      heres[5]->BSIM3cbgb = 0.0;
      heres[6]->BSIM3cbgb = 0.0;
      heres[7]->BSIM3cbgb = 0.0;
    }
    {
      heres[0]->BSIM3cbsb = 0.0;
      heres[1]->BSIM3cbsb = 0.0;
      heres[2]->BSIM3cbsb = 0.0;
      heres[3]->BSIM3cbsb = 0.0;
      heres[4]->BSIM3cbsb = 0.0;
      heres[5]->BSIM3cbsb = 0.0;
      heres[6]->BSIM3cbsb = 0.0;
      heres[7]->BSIM3cbsb = 0.0;
    }
    {
      heres[0]->BSIM3cbdb = 0.0;
      heres[1]->BSIM3cbdb = 0.0;
      heres[2]->BSIM3cbdb = 0.0;
      heres[3]->BSIM3cbdb = 0.0;
      heres[4]->BSIM3cbdb = 0.0;
      heres[5]->BSIM3cbdb = 0.0;
      heres[6]->BSIM3cbdb = 0.0;
      heres[7]->BSIM3cbdb = 0.0;
    }
    {
      heres[0]->BSIM3cqdb = 0.0;
      heres[1]->BSIM3cqdb = 0.0;
      heres[2]->BSIM3cqdb = 0.0;
      heres[3]->BSIM3cqdb = 0.0;
      heres[4]->BSIM3cqdb = 0.0;
      heres[5]->BSIM3cqdb = 0.0;
      heres[6]->BSIM3cqdb = 0.0;
      heres[7]->BSIM3cqdb = 0.0;
    }
    {
      heres[0]->BSIM3cqsb = 0.0;
      heres[1]->BSIM3cqsb = 0.0;
      heres[2]->BSIM3cqsb = 0.0;
      heres[3]->BSIM3cqsb = 0.0;
      heres[4]->BSIM3cqsb = 0.0;
      heres[5]->BSIM3cqsb = 0.0;
      heres[6]->BSIM3cqsb = 0.0;
      heres[7]->BSIM3cqsb = 0.0;
    }
    {
      heres[0]->BSIM3cqgb = 0.0;
      heres[1]->BSIM3cqgb = 0.0;
      heres[2]->BSIM3cqgb = 0.0;
      heres[3]->BSIM3cqgb = 0.0;
      heres[4]->BSIM3cqgb = 0.0;
      heres[5]->BSIM3cqgb = 0.0;
      heres[6]->BSIM3cqgb = 0.0;
      heres[7]->BSIM3cqgb = 0.0;
    }
    {
      heres[0]->BSIM3cqbb = 0.0;
      heres[1]->BSIM3cqbb = 0.0;
      heres[2]->BSIM3cqbb = 0.0;
      heres[3]->BSIM3cqbb = 0.0;
      heres[4]->BSIM3cqbb = 0.0;
      heres[5]->BSIM3cqbb = 0.0;
      heres[6]->BSIM3cqbb = 0.0;
      heres[7]->BSIM3cqbb = 0.0;
    }
    {
      heres[0]->BSIM3gtau = 0.0;
      heres[1]->BSIM3gtau = 0.0;
      heres[2]->BSIM3gtau = 0.0;
      heres[3]->BSIM3gtau = 0.0;
      heres[4]->BSIM3gtau = 0.0;
      heres[5]->BSIM3gtau = 0.0;
      heres[6]->BSIM3gtau = 0.0;
      heres[7]->BSIM3gtau = 0.0;
    }
    goto finished;
  }
  else
    if (model->BSIM3capMod == 0)
  {
    if (1)
    {
      Vec8m condmask0 = Vbseff < 0.0;
      Vec8m condmask_true0 = condmask0;
      Vec8m condmask_false0 = ~condmask0;
      {
        Vbseff = vec8_blend(Vbseff, Vbs, condmask_true0);
        dVbseff_dVb = vec8_blend(dVbseff_dVb, vec8_SIMDTOVECTOR(1.0), condmask_true0);
      }
      {
        Vbseff = vec8_blend(Vbseff, pParam->BSIM3phi - Phis, condmask_false0);
        dVbseff_dVb = vec8_blend(dVbseff_dVb, -dPhis_dVb, condmask_false0);
      }
    }

    Vfb = vec8_SIMDTOVECTOR(pParam->BSIM3vfbcv);
    Vth = (Vfb + pParam->BSIM3phi) + (pParam->BSIM3k1ox * sqrtPhis);
    Vgst = Vgs_eff - Vth;
    dVth_dVb = pParam->BSIM3k1ox * dsqrtPhis_dVb;
    dVgst_dVb = -dVth_dVb;
    dVgst_dVg = dVgs_eff_dVg;
    CoxWL = (model->BSIM3cox * pParam->BSIM3weffCV) * pParam->BSIM3leffCV;
    Arg1 = (Vgs_eff - Vbseff) - Vfb;
    T1 = vec8_SIMDTOVECTOR(0.5 * pParam->BSIM3k1ox);
    T2 = vec8_sqrt((T1 * T1) + Arg1);
    T0 = (CoxWL * T1) / T2;
    if (1)
    {
      Vec8m condmask0 = Arg1 <= 0.0;
      Vec8m condmask_true0 = condmask0;
      Vec8m condmask_false0 = ~condmask0;
      {
        qgate = vec8_blend(qgate, Arg1, condmask_true0);
        T0 = vec8_blend(T0, vec8_SIMDTOVECTOR(CoxWL), condmask_true0);
        {
          Vec8d val = (-CoxWL) * dVgs_eff_dVg;
          if (condmask_true0[0])
            heres[0]->BSIM3cbgb = val[0];

          if (condmask_true0[1])
            heres[1]->BSIM3cbgb = val[1];

          if (condmask_true0[2])
            heres[2]->BSIM3cbgb = val[2];

          if (condmask_true0[3])
            heres[3]->BSIM3cbgb = val[3];

          if (condmask_true0[4])
            heres[4]->BSIM3cbgb = val[4];

          if (condmask_true0[5])
            heres[5]->BSIM3cbgb = val[5];

          if (condmask_true0[6])
            heres[6]->BSIM3cbgb = val[6];

          if (condmask_true0[7])
            heres[7]->BSIM3cbgb = val[7];

        }
      }
      {
        qgate = vec8_blend(qgate, pParam->BSIM3k1ox * (T2 - T1), condmask_false0);
        if (1)
        {
          Vec8m condmask1 = Vgst <= 0.0;
          Vec8m condmask_true1 = condmask_false0 & condmask1;
          {
            Vec8d val = -((Vec8d ){heres[0]->BSIM3cggb, heres[1]->BSIM3cggb, heres[2]->BSIM3cggb, heres[3]->BSIM3cggb, heres[4]->BSIM3cggb, heres[5]->BSIM3cggb, heres[6]->BSIM3cggb, heres[7]->BSIM3cggb});
            if (condmask_false0[0])
              heres[0]->BSIM3cbgb = val[0];

            if (condmask_false0[1])
              heres[1]->BSIM3cbgb = val[1];

            if (condmask_false0[2])
              heres[2]->BSIM3cbgb = val[2];

            if (condmask_false0[3])
              heres[3]->BSIM3cbgb = val[3];

            if (condmask_false0[4])
              heres[4]->BSIM3cbgb = val[4];

            if (condmask_false0[5])
              heres[5]->BSIM3cbgb = val[5];

            if (condmask_false0[6])
              heres[6]->BSIM3cbgb = val[6];

            if (condmask_false0[7])
              heres[7]->BSIM3cbgb = val[7];

          }
        }

      }
    }

    qgate = CoxWL * qgate;
    qbulk = -qgate;
    qdrn = vec8_SIMDTOVECTOR(0.0);
    if (1)
    {
      Vec8m condmask0 = (Arg1 <= 0.0) | (Vgst <= 0.0);
      Vec8m condmask_true0 = condmask0;
      Vec8m condmask_false0 = ~condmask0;
      {
        {
          Vec8d val = T0 * dVgs_eff_dVg;
          if (condmask_true0[0])
            heres[0]->BSIM3cggb = val[0];

          if (condmask_true0[1])
            heres[1]->BSIM3cggb = val[1];

          if (condmask_true0[2])
            heres[2]->BSIM3cggb = val[2];

          if (condmask_true0[3])
            heres[3]->BSIM3cggb = val[3];

          if (condmask_true0[4])
            heres[4]->BSIM3cggb = val[4];

          if (condmask_true0[5])
            heres[5]->BSIM3cggb = val[5];

          if (condmask_true0[6])
            heres[6]->BSIM3cggb = val[6];

          if (condmask_true0[7])
            heres[7]->BSIM3cggb = val[7];

        }
        {
          if (condmask_true0[0])
            heres[0]->BSIM3cgdb = 0.0;

          if (condmask_true0[1])
            heres[1]->BSIM3cgdb = 0.0;

          if (condmask_true0[2])
            heres[2]->BSIM3cgdb = 0.0;

          if (condmask_true0[3])
            heres[3]->BSIM3cgdb = 0.0;

          if (condmask_true0[4])
            heres[4]->BSIM3cgdb = 0.0;

          if (condmask_true0[5])
            heres[5]->BSIM3cgdb = 0.0;

          if (condmask_true0[6])
            heres[6]->BSIM3cgdb = 0.0;

          if (condmask_true0[7])
            heres[7]->BSIM3cgdb = 0.0;

        }
        {
          Vec8d val = T0 * (dVbseff_dVb - dVgs_eff_dVg);
          if (condmask_true0[0])
            heres[0]->BSIM3cgsb = val[0];

          if (condmask_true0[1])
            heres[1]->BSIM3cgsb = val[1];

          if (condmask_true0[2])
            heres[2]->BSIM3cgsb = val[2];

          if (condmask_true0[3])
            heres[3]->BSIM3cgsb = val[3];

          if (condmask_true0[4])
            heres[4]->BSIM3cgsb = val[4];

          if (condmask_true0[5])
            heres[5]->BSIM3cgsb = val[5];

          if (condmask_true0[6])
            heres[6]->BSIM3cgsb = val[6];

          if (condmask_true0[7])
            heres[7]->BSIM3cgsb = val[7];

        }
        {
          if (condmask_true0[0])
            heres[0]->BSIM3cdgb = 0.0;

          if (condmask_true0[1])
            heres[1]->BSIM3cdgb = 0.0;

          if (condmask_true0[2])
            heres[2]->BSIM3cdgb = 0.0;

          if (condmask_true0[3])
            heres[3]->BSIM3cdgb = 0.0;

          if (condmask_true0[4])
            heres[4]->BSIM3cdgb = 0.0;

          if (condmask_true0[5])
            heres[5]->BSIM3cdgb = 0.0;

          if (condmask_true0[6])
            heres[6]->BSIM3cdgb = 0.0;

          if (condmask_true0[7])
            heres[7]->BSIM3cdgb = 0.0;

        }
        {
          if (condmask_true0[0])
            heres[0]->BSIM3cddb = 0.0;

          if (condmask_true0[1])
            heres[1]->BSIM3cddb = 0.0;

          if (condmask_true0[2])
            heres[2]->BSIM3cddb = 0.0;

          if (condmask_true0[3])
            heres[3]->BSIM3cddb = 0.0;

          if (condmask_true0[4])
            heres[4]->BSIM3cddb = 0.0;

          if (condmask_true0[5])
            heres[5]->BSIM3cddb = 0.0;

          if (condmask_true0[6])
            heres[6]->BSIM3cddb = 0.0;

          if (condmask_true0[7])
            heres[7]->BSIM3cddb = 0.0;

        }
        {
          if (condmask_true0[0])
            heres[0]->BSIM3cdsb = 0.0;

          if (condmask_true0[1])
            heres[1]->BSIM3cdsb = 0.0;

          if (condmask_true0[2])
            heres[2]->BSIM3cdsb = 0.0;

          if (condmask_true0[3])
            heres[3]->BSIM3cdsb = 0.0;

          if (condmask_true0[4])
            heres[4]->BSIM3cdsb = 0.0;

          if (condmask_true0[5])
            heres[5]->BSIM3cdsb = 0.0;

          if (condmask_true0[6])
            heres[6]->BSIM3cdsb = 0.0;

          if (condmask_true0[7])
            heres[7]->BSIM3cdsb = 0.0;

        }
        {
          if (condmask_true0[0])
            heres[0]->BSIM3cbdb = 0.0;

          if (condmask_true0[1])
            heres[1]->BSIM3cbdb = 0.0;

          if (condmask_true0[2])
            heres[2]->BSIM3cbdb = 0.0;

          if (condmask_true0[3])
            heres[3]->BSIM3cbdb = 0.0;

          if (condmask_true0[4])
            heres[4]->BSIM3cbdb = 0.0;

          if (condmask_true0[5])
            heres[5]->BSIM3cbdb = 0.0;

          if (condmask_true0[6])
            heres[6]->BSIM3cbdb = 0.0;

          if (condmask_true0[7])
            heres[7]->BSIM3cbdb = 0.0;

        }
        {
          Vec8d val = -((Vec8d ){heres[0]->BSIM3cgsb, heres[1]->BSIM3cgsb, heres[2]->BSIM3cgsb, heres[3]->BSIM3cgsb, heres[4]->BSIM3cgsb, heres[5]->BSIM3cgsb, heres[6]->BSIM3cgsb, heres[7]->BSIM3cgsb});
          if (condmask_true0[0])
            heres[0]->BSIM3cbsb = val[0];

          if (condmask_true0[1])
            heres[1]->BSIM3cbsb = val[1];

          if (condmask_true0[2])
            heres[2]->BSIM3cbsb = val[2];

          if (condmask_true0[3])
            heres[3]->BSIM3cbsb = val[3];

          if (condmask_true0[4])
            heres[4]->BSIM3cbsb = val[4];

          if (condmask_true0[5])
            heres[5]->BSIM3cbsb = val[5];

          if (condmask_true0[6])
            heres[6]->BSIM3cbsb = val[6];

          if (condmask_true0[7])
            heres[7]->BSIM3cbsb = val[7];

        }
        {
          if (condmask_true0[0])
            heres[0]->BSIM3qinv = 0.0;

          if (condmask_true0[1])
            heres[1]->BSIM3qinv = 0.0;

          if (condmask_true0[2])
            heres[2]->BSIM3qinv = 0.0;

          if (condmask_true0[3])
            heres[3]->BSIM3qinv = 0.0;

          if (condmask_true0[4])
            heres[4]->BSIM3qinv = 0.0;

          if (condmask_true0[5])
            heres[5]->BSIM3qinv = 0.0;

          if (condmask_true0[6])
            heres[6]->BSIM3qinv = 0.0;

          if (condmask_true0[7])
            heres[7]->BSIM3qinv = 0.0;

        }
      }
      {
        One_Third_CoxWL = vec8_blend(One_Third_CoxWL, vec8_SIMDTOVECTOR(CoxWL / 3.0), condmask_false0);
        Two_Third_CoxWL = vec8_blend(Two_Third_CoxWL, 2.0 * One_Third_CoxWL, condmask_false0);
        AbulkCV = vec8_blend(AbulkCV, Abulk0 * pParam->BSIM3abulkCVfactor, condmask_false0);
        dAbulkCV_dVb = vec8_blend(dAbulkCV_dVb, pParam->BSIM3abulkCVfactor * dAbulk0_dVb, condmask_false0);
        Vdsat = vec8_blend(Vdsat, Vgst / AbulkCV, condmask_false0);
        dVdsat_dVg = vec8_blend(dVdsat_dVg, dVgs_eff_dVg / AbulkCV, condmask_false0);
        dVdsat_dVb = vec8_blend(dVdsat_dVb, (-((Vdsat * dAbulkCV_dVb) + dVth_dVb)) / AbulkCV, condmask_false0);
        if (model->BSIM3xpart > 0.5)
        {
          if (1)
          {
            Vec8m condmask1 = Vdsat <= Vds;
            Vec8m condmask_true1 = condmask_false0 & condmask1;
            Vec8m condmask_false1 = condmask_false0 & (~condmask1);
            {
              T1 = vec8_blend(T1, Vdsat / 3.0, condmask_true1);
              qgate = vec8_blend(qgate, CoxWL * (((Vgs_eff - Vfb) - pParam->BSIM3phi) - T1), condmask_true1);
              T2 = vec8_blend(T2, (-Two_Third_CoxWL) * Vgst, condmask_true1);
              qbulk = vec8_blend(qbulk, -(qgate + T2), condmask_true1);
              qdrn = vec8_blend(qdrn, vec8_SIMDTOVECTOR(0.0), condmask_true1);
              {
                Vec8d val = (One_Third_CoxWL * (3.0 - dVdsat_dVg)) * dVgs_eff_dVg;
                if (condmask_true1[0])
                  heres[0]->BSIM3cggb = val[0];

                if (condmask_true1[1])
                  heres[1]->BSIM3cggb = val[1];

                if (condmask_true1[2])
                  heres[2]->BSIM3cggb = val[2];

                if (condmask_true1[3])
                  heres[3]->BSIM3cggb = val[3];

                if (condmask_true1[4])
                  heres[4]->BSIM3cggb = val[4];

                if (condmask_true1[5])
                  heres[5]->BSIM3cggb = val[5];

                if (condmask_true1[6])
                  heres[6]->BSIM3cggb = val[6];

                if (condmask_true1[7])
                  heres[7]->BSIM3cggb = val[7];

              }
              T2 = vec8_blend(T2, (-One_Third_CoxWL) * dVdsat_dVb, condmask_true1);
              {
                Vec8d val = -(((Vec8d ){heres[0]->BSIM3cggb, heres[1]->BSIM3cggb, heres[2]->BSIM3cggb, heres[3]->BSIM3cggb, heres[4]->BSIM3cggb, heres[5]->BSIM3cggb, heres[6]->BSIM3cggb, heres[7]->BSIM3cggb}) + T2);
                if (condmask_true1[0])
                  heres[0]->BSIM3cgsb = val[0];

                if (condmask_true1[1])
                  heres[1]->BSIM3cgsb = val[1];

                if (condmask_true1[2])
                  heres[2]->BSIM3cgsb = val[2];

                if (condmask_true1[3])
                  heres[3]->BSIM3cgsb = val[3];

                if (condmask_true1[4])
                  heres[4]->BSIM3cgsb = val[4];

                if (condmask_true1[5])
                  heres[5]->BSIM3cgsb = val[5];

                if (condmask_true1[6])
                  heres[6]->BSIM3cgsb = val[6];

                if (condmask_true1[7])
                  heres[7]->BSIM3cgsb = val[7];

              }
              {
                if (condmask_true1[0])
                  heres[0]->BSIM3cgdb = 0.0;

                if (condmask_true1[1])
                  heres[1]->BSIM3cgdb = 0.0;

                if (condmask_true1[2])
                  heres[2]->BSIM3cgdb = 0.0;

                if (condmask_true1[3])
                  heres[3]->BSIM3cgdb = 0.0;

                if (condmask_true1[4])
                  heres[4]->BSIM3cgdb = 0.0;

                if (condmask_true1[5])
                  heres[5]->BSIM3cgdb = 0.0;

                if (condmask_true1[6])
                  heres[6]->BSIM3cgdb = 0.0;

                if (condmask_true1[7])
                  heres[7]->BSIM3cgdb = 0.0;

              }
              {
                if (condmask_true1[0])
                  heres[0]->BSIM3cdgb = 0.0;

                if (condmask_true1[1])
                  heres[1]->BSIM3cdgb = 0.0;

                if (condmask_true1[2])
                  heres[2]->BSIM3cdgb = 0.0;

                if (condmask_true1[3])
                  heres[3]->BSIM3cdgb = 0.0;

                if (condmask_true1[4])
                  heres[4]->BSIM3cdgb = 0.0;

                if (condmask_true1[5])
                  heres[5]->BSIM3cdgb = 0.0;

                if (condmask_true1[6])
                  heres[6]->BSIM3cdgb = 0.0;

                if (condmask_true1[7])
                  heres[7]->BSIM3cdgb = 0.0;

              }
              {
                if (condmask_true1[0])
                  heres[0]->BSIM3cddb = 0.0;

                if (condmask_true1[1])
                  heres[1]->BSIM3cddb = 0.0;

                if (condmask_true1[2])
                  heres[2]->BSIM3cddb = 0.0;

                if (condmask_true1[3])
                  heres[3]->BSIM3cddb = 0.0;

                if (condmask_true1[4])
                  heres[4]->BSIM3cddb = 0.0;

                if (condmask_true1[5])
                  heres[5]->BSIM3cddb = 0.0;

                if (condmask_true1[6])
                  heres[6]->BSIM3cddb = 0.0;

                if (condmask_true1[7])
                  heres[7]->BSIM3cddb = 0.0;

              }
              {
                if (condmask_true1[0])
                  heres[0]->BSIM3cdsb = 0.0;

                if (condmask_true1[1])
                  heres[1]->BSIM3cdsb = 0.0;

                if (condmask_true1[2])
                  heres[2]->BSIM3cdsb = 0.0;

                if (condmask_true1[3])
                  heres[3]->BSIM3cdsb = 0.0;

                if (condmask_true1[4])
                  heres[4]->BSIM3cdsb = 0.0;

                if (condmask_true1[5])
                  heres[5]->BSIM3cdsb = 0.0;

                if (condmask_true1[6])
                  heres[6]->BSIM3cdsb = 0.0;

                if (condmask_true1[7])
                  heres[7]->BSIM3cdsb = 0.0;

              }
              {
                Vec8d val = -(((Vec8d ){heres[0]->BSIM3cggb, heres[1]->BSIM3cggb, heres[2]->BSIM3cggb, heres[3]->BSIM3cggb, heres[4]->BSIM3cggb, heres[5]->BSIM3cggb, heres[6]->BSIM3cggb, heres[7]->BSIM3cggb}) - (Two_Third_CoxWL * dVgs_eff_dVg));
                if (condmask_true1[0])
                  heres[0]->BSIM3cbgb = val[0];

                if (condmask_true1[1])
                  heres[1]->BSIM3cbgb = val[1];

                if (condmask_true1[2])
                  heres[2]->BSIM3cbgb = val[2];

                if (condmask_true1[3])
                  heres[3]->BSIM3cbgb = val[3];

                if (condmask_true1[4])
                  heres[4]->BSIM3cbgb = val[4];

                if (condmask_true1[5])
                  heres[5]->BSIM3cbgb = val[5];

                if (condmask_true1[6])
                  heres[6]->BSIM3cbgb = val[6];

                if (condmask_true1[7])
                  heres[7]->BSIM3cbgb = val[7];

              }
              T3 = vec8_blend(T3, -(T2 + (Two_Third_CoxWL * dVth_dVb)), condmask_true1);
              {
                Vec8d val = -(((Vec8d ){heres[0]->BSIM3cbgb, heres[1]->BSIM3cbgb, heres[2]->BSIM3cbgb, heres[3]->BSIM3cbgb, heres[4]->BSIM3cbgb, heres[5]->BSIM3cbgb, heres[6]->BSIM3cbgb, heres[7]->BSIM3cbgb}) + T3);
                if (condmask_true1[0])
                  heres[0]->BSIM3cbsb = val[0];

                if (condmask_true1[1])
                  heres[1]->BSIM3cbsb = val[1];

                if (condmask_true1[2])
                  heres[2]->BSIM3cbsb = val[2];

                if (condmask_true1[3])
                  heres[3]->BSIM3cbsb = val[3];

                if (condmask_true1[4])
                  heres[4]->BSIM3cbsb = val[4];

                if (condmask_true1[5])
                  heres[5]->BSIM3cbsb = val[5];

                if (condmask_true1[6])
                  heres[6]->BSIM3cbsb = val[6];

                if (condmask_true1[7])
                  heres[7]->BSIM3cbsb = val[7];

              }
              {
                if (condmask_true1[0])
                  heres[0]->BSIM3cbdb = 0.0;

                if (condmask_true1[1])
                  heres[1]->BSIM3cbdb = 0.0;

                if (condmask_true1[2])
                  heres[2]->BSIM3cbdb = 0.0;

                if (condmask_true1[3])
                  heres[3]->BSIM3cbdb = 0.0;

                if (condmask_true1[4])
                  heres[4]->BSIM3cbdb = 0.0;

                if (condmask_true1[5])
                  heres[5]->BSIM3cbdb = 0.0;

                if (condmask_true1[6])
                  heres[6]->BSIM3cbdb = 0.0;

                if (condmask_true1[7])
                  heres[7]->BSIM3cbdb = 0.0;

              }
              {
                Vec8d val = -(qgate + qbulk);
                if (condmask_true1[0])
                  heres[0]->BSIM3qinv = val[0];

                if (condmask_true1[1])
                  heres[1]->BSIM3qinv = val[1];

                if (condmask_true1[2])
                  heres[2]->BSIM3qinv = val[2];

                if (condmask_true1[3])
                  heres[3]->BSIM3qinv = val[3];

                if (condmask_true1[4])
                  heres[4]->BSIM3qinv = val[4];

                if (condmask_true1[5])
                  heres[5]->BSIM3qinv = val[5];

                if (condmask_true1[6])
                  heres[6]->BSIM3qinv = val[6];

                if (condmask_true1[7])
                  heres[7]->BSIM3qinv = val[7];

              }
            }
            {
              Alphaz = vec8_blend(Alphaz, Vgst / Vdsat, condmask_false1);
              T1 = vec8_blend(T1, (2.0 * Vdsat) - Vds, condmask_false1);
              T2 = vec8_blend(T2, Vds / (3.0 * T1), condmask_false1);
              T3 = vec8_blend(T3, T2 * Vds, condmask_false1);
              T9 = vec8_blend(T9, vec8_SIMDTOVECTOR(0.25 * CoxWL), condmask_false1);
              T4 = vec8_blend(T4, T9 * Alphaz, condmask_false1);
              T7 = vec8_blend(T7, ((2.0 * Vds) - T1) - (3.0 * T3), condmask_false1);
              T8 = vec8_blend(T8, (T3 - T1) - (2.0 * Vds), condmask_false1);
              qgate = vec8_blend(qgate, CoxWL * (((Vgs_eff - Vfb) - pParam->BSIM3phi) - (0.5 * (Vds - T3))), condmask_false1);
              T10 = vec8_blend(T10, T4 * T8, condmask_false1);
              qdrn = vec8_blend(qdrn, T4 * T7, condmask_false1);
              qbulk = vec8_blend(qbulk, -((qgate + qdrn) + T10), condmask_false1);
              T5 = vec8_blend(T5, T3 / T1, condmask_false1);
              {
                Vec8d val = (CoxWL * (1.0 - (T5 * dVdsat_dVg))) * dVgs_eff_dVg;
                if (condmask_false1[0])
                  heres[0]->BSIM3cggb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cggb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cggb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cggb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cggb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cggb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cggb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cggb = val[7];

              }
              T11 = vec8_blend(T11, ((-CoxWL) * T5) * dVdsat_dVb, condmask_false1);
              {
                Vec8d val = CoxWL * ((T2 - 0.5) + (0.5 * T5));
                if (condmask_false1[0])
                  heres[0]->BSIM3cgdb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cgdb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cgdb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cgdb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cgdb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cgdb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cgdb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cgdb = val[7];

              }
              {
                Vec8d val = -((((Vec8d ){heres[0]->BSIM3cggb, heres[1]->BSIM3cggb, heres[2]->BSIM3cggb, heres[3]->BSIM3cggb, heres[4]->BSIM3cggb, heres[5]->BSIM3cggb, heres[6]->BSIM3cggb, heres[7]->BSIM3cggb}) + T11) + ((Vec8d ){heres[0]->BSIM3cgdb, heres[1]->BSIM3cgdb, heres[2]->BSIM3cgdb, heres[3]->BSIM3cgdb, heres[4]->BSIM3cgdb, heres[5]->BSIM3cgdb, heres[6]->BSIM3cgdb, heres[7]->BSIM3cgdb}));
                if (condmask_false1[0])
                  heres[0]->BSIM3cgsb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cgsb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cgsb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cgsb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cgsb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cgsb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cgsb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cgsb = val[7];

              }
              T6 = vec8_blend(T6, 1.0 / Vdsat, condmask_false1);
              dAlphaz_dVg = vec8_blend(dAlphaz_dVg, T6 * (1.0 - (Alphaz * dVdsat_dVg)), condmask_false1);
              dAlphaz_dVb = vec8_blend(dAlphaz_dVb, (-T6) * (dVth_dVb + (Alphaz * dVdsat_dVb)), condmask_false1);
              T7 = vec8_blend(T7, T9 * T7, condmask_false1);
              T8 = vec8_blend(T8, T9 * T8, condmask_false1);
              T9 = vec8_blend(T9, (2.0 * T4) * (1.0 - (3.0 * T5)), condmask_false1);
              {
                Vec8d val = ((T7 * dAlphaz_dVg) - (T9 * dVdsat_dVg)) * dVgs_eff_dVg;
                if (condmask_false1[0])
                  heres[0]->BSIM3cdgb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cdgb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cdgb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cdgb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cdgb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cdgb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cdgb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cdgb = val[7];

              }
              T12 = vec8_blend(T12, (T7 * dAlphaz_dVb) - (T9 * dVdsat_dVb), condmask_false1);
              {
                Vec8d val = T4 * ((3.0 - (6.0 * T2)) - (3.0 * T5));
                if (condmask_false1[0])
                  heres[0]->BSIM3cddb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cddb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cddb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cddb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cddb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cddb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cddb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cddb = val[7];

              }
              {
                Vec8d val = -((((Vec8d ){heres[0]->BSIM3cdgb, heres[1]->BSIM3cdgb, heres[2]->BSIM3cdgb, heres[3]->BSIM3cdgb, heres[4]->BSIM3cdgb, heres[5]->BSIM3cdgb, heres[6]->BSIM3cdgb, heres[7]->BSIM3cdgb}) + T12) + ((Vec8d ){heres[0]->BSIM3cddb, heres[1]->BSIM3cddb, heres[2]->BSIM3cddb, heres[3]->BSIM3cddb, heres[4]->BSIM3cddb, heres[5]->BSIM3cddb, heres[6]->BSIM3cddb, heres[7]->BSIM3cddb}));
                if (condmask_false1[0])
                  heres[0]->BSIM3cdsb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cdsb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cdsb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cdsb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cdsb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cdsb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cdsb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cdsb = val[7];

              }
              T9 = vec8_blend(T9, (2.0 * T4) * (1.0 + T5), condmask_false1);
              T10 = vec8_blend(T10, ((T8 * dAlphaz_dVg) - (T9 * dVdsat_dVg)) * dVgs_eff_dVg, condmask_false1);
              T11 = vec8_blend(T11, (T8 * dAlphaz_dVb) - (T9 * dVdsat_dVb), condmask_false1);
              T12 = vec8_blend(T12, T4 * (((2.0 * T2) + T5) - 1.0), condmask_false1);
              T0 = vec8_blend(T0, -((T10 + T11) + T12), condmask_false1);
              {
                Vec8d val = -((((Vec8d ){heres[0]->BSIM3cggb, heres[1]->BSIM3cggb, heres[2]->BSIM3cggb, heres[3]->BSIM3cggb, heres[4]->BSIM3cggb, heres[5]->BSIM3cggb, heres[6]->BSIM3cggb, heres[7]->BSIM3cggb}) + ((Vec8d ){heres[0]->BSIM3cdgb, heres[1]->BSIM3cdgb, heres[2]->BSIM3cdgb, heres[3]->BSIM3cdgb, heres[4]->BSIM3cdgb, heres[5]->BSIM3cdgb, heres[6]->BSIM3cdgb, heres[7]->BSIM3cdgb})) + T10);
                if (condmask_false1[0])
                  heres[0]->BSIM3cbgb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cbgb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cbgb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cbgb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cbgb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cbgb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cbgb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cbgb = val[7];

              }
              {
                Vec8d val = -((((Vec8d ){heres[0]->BSIM3cgdb, heres[1]->BSIM3cgdb, heres[2]->BSIM3cgdb, heres[3]->BSIM3cgdb, heres[4]->BSIM3cgdb, heres[5]->BSIM3cgdb, heres[6]->BSIM3cgdb, heres[7]->BSIM3cgdb}) + ((Vec8d ){heres[0]->BSIM3cddb, heres[1]->BSIM3cddb, heres[2]->BSIM3cddb, heres[3]->BSIM3cddb, heres[4]->BSIM3cddb, heres[5]->BSIM3cddb, heres[6]->BSIM3cddb, heres[7]->BSIM3cddb})) + T12);
                if (condmask_false1[0])
                  heres[0]->BSIM3cbdb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cbdb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cbdb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cbdb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cbdb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cbdb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cbdb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cbdb = val[7];

              }
              {
                Vec8d val = -((((Vec8d ){heres[0]->BSIM3cgsb, heres[1]->BSIM3cgsb, heres[2]->BSIM3cgsb, heres[3]->BSIM3cgsb, heres[4]->BSIM3cgsb, heres[5]->BSIM3cgsb, heres[6]->BSIM3cgsb, heres[7]->BSIM3cgsb}) + ((Vec8d ){heres[0]->BSIM3cdsb, heres[1]->BSIM3cdsb, heres[2]->BSIM3cdsb, heres[3]->BSIM3cdsb, heres[4]->BSIM3cdsb, heres[5]->BSIM3cdsb, heres[6]->BSIM3cdsb, heres[7]->BSIM3cdsb})) + T0);
                if (condmask_false1[0])
                  heres[0]->BSIM3cbsb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cbsb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cbsb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cbsb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cbsb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cbsb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cbsb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cbsb = val[7];

              }
              {
                Vec8d val = -(qgate + qbulk);
                if (condmask_false1[0])
                  heres[0]->BSIM3qinv = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3qinv = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3qinv = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3qinv = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3qinv = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3qinv = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3qinv = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3qinv = val[7];

              }
            }
          }

        }
        else
          if (model->BSIM3xpart < 0.5)
        {
          if (1)
          {
            Vec8m condmask1 = Vds >= Vdsat;
            Vec8m condmask_true1 = condmask_false0 & condmask1;
            Vec8m condmask_false1 = condmask_false0 & (~condmask1);
            {
              T1 = vec8_blend(T1, Vdsat / 3.0, condmask_true1);
              qgate = vec8_blend(qgate, CoxWL * (((Vgs_eff - Vfb) - pParam->BSIM3phi) - T1), condmask_true1);
              T2 = vec8_blend(T2, (-Two_Third_CoxWL) * Vgst, condmask_true1);
              qbulk = vec8_blend(qbulk, -(qgate + T2), condmask_true1);
              qdrn = vec8_blend(qdrn, 0.4 * T2, condmask_true1);
              {
                Vec8d val = (One_Third_CoxWL * (3.0 - dVdsat_dVg)) * dVgs_eff_dVg;
                if (condmask_true1[0])
                  heres[0]->BSIM3cggb = val[0];

                if (condmask_true1[1])
                  heres[1]->BSIM3cggb = val[1];

                if (condmask_true1[2])
                  heres[2]->BSIM3cggb = val[2];

                if (condmask_true1[3])
                  heres[3]->BSIM3cggb = val[3];

                if (condmask_true1[4])
                  heres[4]->BSIM3cggb = val[4];

                if (condmask_true1[5])
                  heres[5]->BSIM3cggb = val[5];

                if (condmask_true1[6])
                  heres[6]->BSIM3cggb = val[6];

                if (condmask_true1[7])
                  heres[7]->BSIM3cggb = val[7];

              }
              T2 = vec8_blend(T2, (-One_Third_CoxWL) * dVdsat_dVb, condmask_true1);
              {
                Vec8d val = -(((Vec8d ){heres[0]->BSIM3cggb, heres[1]->BSIM3cggb, heres[2]->BSIM3cggb, heres[3]->BSIM3cggb, heres[4]->BSIM3cggb, heres[5]->BSIM3cggb, heres[6]->BSIM3cggb, heres[7]->BSIM3cggb}) + T2);
                if (condmask_true1[0])
                  heres[0]->BSIM3cgsb = val[0];

                if (condmask_true1[1])
                  heres[1]->BSIM3cgsb = val[1];

                if (condmask_true1[2])
                  heres[2]->BSIM3cgsb = val[2];

                if (condmask_true1[3])
                  heres[3]->BSIM3cgsb = val[3];

                if (condmask_true1[4])
                  heres[4]->BSIM3cgsb = val[4];

                if (condmask_true1[5])
                  heres[5]->BSIM3cgsb = val[5];

                if (condmask_true1[6])
                  heres[6]->BSIM3cgsb = val[6];

                if (condmask_true1[7])
                  heres[7]->BSIM3cgsb = val[7];

              }
              {
                if (condmask_true1[0])
                  heres[0]->BSIM3cgdb = 0.0;

                if (condmask_true1[1])
                  heres[1]->BSIM3cgdb = 0.0;

                if (condmask_true1[2])
                  heres[2]->BSIM3cgdb = 0.0;

                if (condmask_true1[3])
                  heres[3]->BSIM3cgdb = 0.0;

                if (condmask_true1[4])
                  heres[4]->BSIM3cgdb = 0.0;

                if (condmask_true1[5])
                  heres[5]->BSIM3cgdb = 0.0;

                if (condmask_true1[6])
                  heres[6]->BSIM3cgdb = 0.0;

                if (condmask_true1[7])
                  heres[7]->BSIM3cgdb = 0.0;

              }
              T3 = vec8_blend(T3, 0.4 * Two_Third_CoxWL, condmask_true1);
              {
                Vec8d val = (-T3) * dVgs_eff_dVg;
                if (condmask_true1[0])
                  heres[0]->BSIM3cdgb = val[0];

                if (condmask_true1[1])
                  heres[1]->BSIM3cdgb = val[1];

                if (condmask_true1[2])
                  heres[2]->BSIM3cdgb = val[2];

                if (condmask_true1[3])
                  heres[3]->BSIM3cdgb = val[3];

                if (condmask_true1[4])
                  heres[4]->BSIM3cdgb = val[4];

                if (condmask_true1[5])
                  heres[5]->BSIM3cdgb = val[5];

                if (condmask_true1[6])
                  heres[6]->BSIM3cdgb = val[6];

                if (condmask_true1[7])
                  heres[7]->BSIM3cdgb = val[7];

              }
              {
                if (condmask_true1[0])
                  heres[0]->BSIM3cddb = 0.0;

                if (condmask_true1[1])
                  heres[1]->BSIM3cddb = 0.0;

                if (condmask_true1[2])
                  heres[2]->BSIM3cddb = 0.0;

                if (condmask_true1[3])
                  heres[3]->BSIM3cddb = 0.0;

                if (condmask_true1[4])
                  heres[4]->BSIM3cddb = 0.0;

                if (condmask_true1[5])
                  heres[5]->BSIM3cddb = 0.0;

                if (condmask_true1[6])
                  heres[6]->BSIM3cddb = 0.0;

                if (condmask_true1[7])
                  heres[7]->BSIM3cddb = 0.0;

              }
              T4 = vec8_blend(T4, T3 * dVth_dVb, condmask_true1);
              {
                Vec8d val = -(T4 + ((Vec8d ){heres[0]->BSIM3cdgb, heres[1]->BSIM3cdgb, heres[2]->BSIM3cdgb, heres[3]->BSIM3cdgb, heres[4]->BSIM3cdgb, heres[5]->BSIM3cdgb, heres[6]->BSIM3cdgb, heres[7]->BSIM3cdgb}));
                if (condmask_true1[0])
                  heres[0]->BSIM3cdsb = val[0];

                if (condmask_true1[1])
                  heres[1]->BSIM3cdsb = val[1];

                if (condmask_true1[2])
                  heres[2]->BSIM3cdsb = val[2];

                if (condmask_true1[3])
                  heres[3]->BSIM3cdsb = val[3];

                if (condmask_true1[4])
                  heres[4]->BSIM3cdsb = val[4];

                if (condmask_true1[5])
                  heres[5]->BSIM3cdsb = val[5];

                if (condmask_true1[6])
                  heres[6]->BSIM3cdsb = val[6];

                if (condmask_true1[7])
                  heres[7]->BSIM3cdsb = val[7];

              }
              {
                Vec8d val = -(((Vec8d ){heres[0]->BSIM3cggb, heres[1]->BSIM3cggb, heres[2]->BSIM3cggb, heres[3]->BSIM3cggb, heres[4]->BSIM3cggb, heres[5]->BSIM3cggb, heres[6]->BSIM3cggb, heres[7]->BSIM3cggb}) - (Two_Third_CoxWL * dVgs_eff_dVg));
                if (condmask_true1[0])
                  heres[0]->BSIM3cbgb = val[0];

                if (condmask_true1[1])
                  heres[1]->BSIM3cbgb = val[1];

                if (condmask_true1[2])
                  heres[2]->BSIM3cbgb = val[2];

                if (condmask_true1[3])
                  heres[3]->BSIM3cbgb = val[3];

                if (condmask_true1[4])
                  heres[4]->BSIM3cbgb = val[4];

                if (condmask_true1[5])
                  heres[5]->BSIM3cbgb = val[5];

                if (condmask_true1[6])
                  heres[6]->BSIM3cbgb = val[6];

                if (condmask_true1[7])
                  heres[7]->BSIM3cbgb = val[7];

              }
              T3 = vec8_blend(T3, -(T2 + (Two_Third_CoxWL * dVth_dVb)), condmask_true1);
              {
                Vec8d val = -(((Vec8d ){heres[0]->BSIM3cbgb, heres[1]->BSIM3cbgb, heres[2]->BSIM3cbgb, heres[3]->BSIM3cbgb, heres[4]->BSIM3cbgb, heres[5]->BSIM3cbgb, heres[6]->BSIM3cbgb, heres[7]->BSIM3cbgb}) + T3);
                if (condmask_true1[0])
                  heres[0]->BSIM3cbsb = val[0];

                if (condmask_true1[1])
                  heres[1]->BSIM3cbsb = val[1];

                if (condmask_true1[2])
                  heres[2]->BSIM3cbsb = val[2];

                if (condmask_true1[3])
                  heres[3]->BSIM3cbsb = val[3];

                if (condmask_true1[4])
                  heres[4]->BSIM3cbsb = val[4];

                if (condmask_true1[5])
                  heres[5]->BSIM3cbsb = val[5];

                if (condmask_true1[6])
                  heres[6]->BSIM3cbsb = val[6];

                if (condmask_true1[7])
                  heres[7]->BSIM3cbsb = val[7];

              }
              {
                if (condmask_true1[0])
                  heres[0]->BSIM3cbdb = 0.0;

                if (condmask_true1[1])
                  heres[1]->BSIM3cbdb = 0.0;

                if (condmask_true1[2])
                  heres[2]->BSIM3cbdb = 0.0;

                if (condmask_true1[3])
                  heres[3]->BSIM3cbdb = 0.0;

                if (condmask_true1[4])
                  heres[4]->BSIM3cbdb = 0.0;

                if (condmask_true1[5])
                  heres[5]->BSIM3cbdb = 0.0;

                if (condmask_true1[6])
                  heres[6]->BSIM3cbdb = 0.0;

                if (condmask_true1[7])
                  heres[7]->BSIM3cbdb = 0.0;

              }
              {
                Vec8d val = -(qgate + qbulk);
                if (condmask_true1[0])
                  heres[0]->BSIM3qinv = val[0];

                if (condmask_true1[1])
                  heres[1]->BSIM3qinv = val[1];

                if (condmask_true1[2])
                  heres[2]->BSIM3qinv = val[2];

                if (condmask_true1[3])
                  heres[3]->BSIM3qinv = val[3];

                if (condmask_true1[4])
                  heres[4]->BSIM3qinv = val[4];

                if (condmask_true1[5])
                  heres[5]->BSIM3qinv = val[5];

                if (condmask_true1[6])
                  heres[6]->BSIM3qinv = val[6];

                if (condmask_true1[7])
                  heres[7]->BSIM3qinv = val[7];

              }
            }
            {
              Alphaz = vec8_blend(Alphaz, Vgst / Vdsat, condmask_false1);
              T1 = vec8_blend(T1, (2.0 * Vdsat) - Vds, condmask_false1);
              T2 = vec8_blend(T2, Vds / (3.0 * T1), condmask_false1);
              T3 = vec8_blend(T3, T2 * Vds, condmask_false1);
              T9 = vec8_blend(T9, vec8_SIMDTOVECTOR(0.25 * CoxWL), condmask_false1);
              T4 = vec8_blend(T4, T9 * Alphaz, condmask_false1);
              qgate = vec8_blend(qgate, CoxWL * (((Vgs_eff - Vfb) - pParam->BSIM3phi) - (0.5 * (Vds - T3))), condmask_false1);
              T5 = vec8_blend(T5, T3 / T1, condmask_false1);
              {
                Vec8d val = (CoxWL * (1.0 - (T5 * dVdsat_dVg))) * dVgs_eff_dVg;
                if (condmask_false1[0])
                  heres[0]->BSIM3cggb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cggb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cggb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cggb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cggb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cggb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cggb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cggb = val[7];

              }
              tmp = vec8_blend(tmp, ((-CoxWL) * T5) * dVdsat_dVb, condmask_false1);
              {
                Vec8d val = CoxWL * ((T2 - 0.5) + (0.5 * T5));
                if (condmask_false1[0])
                  heres[0]->BSIM3cgdb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cgdb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cgdb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cgdb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cgdb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cgdb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cgdb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cgdb = val[7];

              }
              {
                Vec8d val = -((((Vec8d ){heres[0]->BSIM3cggb, heres[1]->BSIM3cggb, heres[2]->BSIM3cggb, heres[3]->BSIM3cggb, heres[4]->BSIM3cggb, heres[5]->BSIM3cggb, heres[6]->BSIM3cggb, heres[7]->BSIM3cggb}) + ((Vec8d ){heres[0]->BSIM3cgdb, heres[1]->BSIM3cgdb, heres[2]->BSIM3cgdb, heres[3]->BSIM3cgdb, heres[4]->BSIM3cgdb, heres[5]->BSIM3cgdb, heres[6]->BSIM3cgdb, heres[7]->BSIM3cgdb})) + tmp);
                if (condmask_false1[0])
                  heres[0]->BSIM3cgsb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cgsb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cgsb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cgsb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cgsb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cgsb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cgsb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cgsb = val[7];

              }
              T6 = vec8_blend(T6, 1.0 / Vdsat, condmask_false1);
              dAlphaz_dVg = vec8_blend(dAlphaz_dVg, T6 * (1.0 - (Alphaz * dVdsat_dVg)), condmask_false1);
              dAlphaz_dVb = vec8_blend(dAlphaz_dVb, (-T6) * (dVth_dVb + (Alphaz * dVdsat_dVb)), condmask_false1);
              T6 = vec8_blend(T6, (((8.0 * Vdsat) * Vdsat) - ((6.0 * Vdsat) * Vds)) + ((1.2 * Vds) * Vds), condmask_false1);
              T8 = vec8_blend(T8, T2 / T1, condmask_false1);
              T7 = vec8_blend(T7, (Vds - T1) - (T8 * T6), condmask_false1);
              qdrn = vec8_blend(qdrn, T4 * T7, condmask_false1);
              T7 = vec8_blend(T7, T7 * T9, condmask_false1);
              tmp = vec8_blend(tmp, T8 / T1, condmask_false1);
              tmp1 = vec8_blend(tmp1, T4 * ((2.0 - ((4.0 * tmp) * T6)) + (T8 * ((16.0 * Vdsat) - (6.0 * Vds)))), condmask_false1);
              {
                Vec8d val = ((T7 * dAlphaz_dVg) - (tmp1 * dVdsat_dVg)) * dVgs_eff_dVg;
                if (condmask_false1[0])
                  heres[0]->BSIM3cdgb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cdgb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cdgb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cdgb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cdgb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cdgb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cdgb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cdgb = val[7];

              }
              T10 = vec8_blend(T10, (T7 * dAlphaz_dVb) - (tmp1 * dVdsat_dVb), condmask_false1);
              {
                Vec8d val = T4 * ((2.0 - (((1.0 / ((3.0 * T1) * T1)) + (2.0 * tmp)) * T6)) + (T8 * ((6.0 * Vdsat) - (2.4 * Vds))));
                if (condmask_false1[0])
                  heres[0]->BSIM3cddb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cddb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cddb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cddb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cddb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cddb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cddb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cddb = val[7];

              }
              {
                Vec8d val = -((((Vec8d ){heres[0]->BSIM3cdgb, heres[1]->BSIM3cdgb, heres[2]->BSIM3cdgb, heres[3]->BSIM3cdgb, heres[4]->BSIM3cdgb, heres[5]->BSIM3cdgb, heres[6]->BSIM3cdgb, heres[7]->BSIM3cdgb}) + T10) + ((Vec8d ){heres[0]->BSIM3cddb, heres[1]->BSIM3cddb, heres[2]->BSIM3cddb, heres[3]->BSIM3cddb, heres[4]->BSIM3cddb, heres[5]->BSIM3cddb, heres[6]->BSIM3cddb, heres[7]->BSIM3cddb}));
                if (condmask_false1[0])
                  heres[0]->BSIM3cdsb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cdsb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cdsb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cdsb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cdsb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cdsb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cdsb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cdsb = val[7];

              }
              T7 = vec8_blend(T7, 2.0 * (T1 + T3), condmask_false1);
              qbulk = vec8_blend(qbulk, -(qgate - (T4 * T7)), condmask_false1);
              T7 = vec8_blend(T7, T7 * T9, condmask_false1);
              T0 = vec8_blend(T0, (4.0 * T4) * (1.0 - T5), condmask_false1);
              T12 = vec8_blend(T12, ((((-T7) * dAlphaz_dVg) - ((Vec8d ){heres[0]->BSIM3cdgb, heres[1]->BSIM3cdgb, heres[2]->BSIM3cdgb, heres[3]->BSIM3cdgb, heres[4]->BSIM3cdgb, heres[5]->BSIM3cdgb, heres[6]->BSIM3cdgb, heres[7]->BSIM3cdgb})) - (T0 * dVdsat_dVg)) * dVgs_eff_dVg, condmask_false1);
              T11 = vec8_blend(T11, (((-T7) * dAlphaz_dVb) - T10) - (T0 * dVdsat_dVb), condmask_false1);
              T10 = vec8_blend(T10, (((-4.0) * T4) * ((T2 - 0.5) + (0.5 * T5))) - ((Vec8d ){heres[0]->BSIM3cddb, heres[1]->BSIM3cddb, heres[2]->BSIM3cddb, heres[3]->BSIM3cddb, heres[4]->BSIM3cddb, heres[5]->BSIM3cddb, heres[6]->BSIM3cddb, heres[7]->BSIM3cddb}), condmask_false1);
              tmp = vec8_blend(tmp, -((T10 + T11) + T12), condmask_false1);
              {
                Vec8d val = -((((Vec8d ){heres[0]->BSIM3cggb, heres[1]->BSIM3cggb, heres[2]->BSIM3cggb, heres[3]->BSIM3cggb, heres[4]->BSIM3cggb, heres[5]->BSIM3cggb, heres[6]->BSIM3cggb, heres[7]->BSIM3cggb}) + ((Vec8d ){heres[0]->BSIM3cdgb, heres[1]->BSIM3cdgb, heres[2]->BSIM3cdgb, heres[3]->BSIM3cdgb, heres[4]->BSIM3cdgb, heres[5]->BSIM3cdgb, heres[6]->BSIM3cdgb, heres[7]->BSIM3cdgb})) + T12);
                if (condmask_false1[0])
                  heres[0]->BSIM3cbgb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cbgb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cbgb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cbgb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cbgb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cbgb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cbgb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cbgb = val[7];

              }
              {
                Vec8d val = -((((Vec8d ){heres[0]->BSIM3cgdb, heres[1]->BSIM3cgdb, heres[2]->BSIM3cgdb, heres[3]->BSIM3cgdb, heres[4]->BSIM3cgdb, heres[5]->BSIM3cgdb, heres[6]->BSIM3cgdb, heres[7]->BSIM3cgdb}) + ((Vec8d ){heres[0]->BSIM3cddb, heres[1]->BSIM3cddb, heres[2]->BSIM3cddb, heres[3]->BSIM3cddb, heres[4]->BSIM3cddb, heres[5]->BSIM3cddb, heres[6]->BSIM3cddb, heres[7]->BSIM3cddb})) + T10);
                if (condmask_false1[0])
                  heres[0]->BSIM3cbdb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cbdb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cbdb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cbdb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cbdb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cbdb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cbdb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cbdb = val[7];

              }
              {
                Vec8d val = -((((Vec8d ){heres[0]->BSIM3cgsb, heres[1]->BSIM3cgsb, heres[2]->BSIM3cgsb, heres[3]->BSIM3cgsb, heres[4]->BSIM3cgsb, heres[5]->BSIM3cgsb, heres[6]->BSIM3cgsb, heres[7]->BSIM3cgsb}) + ((Vec8d ){heres[0]->BSIM3cdsb, heres[1]->BSIM3cdsb, heres[2]->BSIM3cdsb, heres[3]->BSIM3cdsb, heres[4]->BSIM3cdsb, heres[5]->BSIM3cdsb, heres[6]->BSIM3cdsb, heres[7]->BSIM3cdsb})) + tmp);
                if (condmask_false1[0])
                  heres[0]->BSIM3cbsb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cbsb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cbsb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cbsb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cbsb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cbsb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cbsb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cbsb = val[7];

              }
              {
                Vec8d val = -(qgate + qbulk);
                if (condmask_false1[0])
                  heres[0]->BSIM3qinv = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3qinv = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3qinv = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3qinv = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3qinv = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3qinv = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3qinv = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3qinv = val[7];

              }
            }
          }

        }
        else
        {
          if (1)
          {
            Vec8m condmask1 = Vds >= Vdsat;
            Vec8m condmask_true1 = condmask_false0 & condmask1;
            Vec8m condmask_false1 = condmask_false0 & (~condmask1);
            {
              T1 = vec8_blend(T1, Vdsat / 3.0, condmask_true1);
              qgate = vec8_blend(qgate, CoxWL * (((Vgs_eff - Vfb) - pParam->BSIM3phi) - T1), condmask_true1);
              T2 = vec8_blend(T2, (-Two_Third_CoxWL) * Vgst, condmask_true1);
              qbulk = vec8_blend(qbulk, -(qgate + T2), condmask_true1);
              qdrn = vec8_blend(qdrn, 0.5 * T2, condmask_true1);
              {
                Vec8d val = (One_Third_CoxWL * (3.0 - dVdsat_dVg)) * dVgs_eff_dVg;
                if (condmask_true1[0])
                  heres[0]->BSIM3cggb = val[0];

                if (condmask_true1[1])
                  heres[1]->BSIM3cggb = val[1];

                if (condmask_true1[2])
                  heres[2]->BSIM3cggb = val[2];

                if (condmask_true1[3])
                  heres[3]->BSIM3cggb = val[3];

                if (condmask_true1[4])
                  heres[4]->BSIM3cggb = val[4];

                if (condmask_true1[5])
                  heres[5]->BSIM3cggb = val[5];

                if (condmask_true1[6])
                  heres[6]->BSIM3cggb = val[6];

                if (condmask_true1[7])
                  heres[7]->BSIM3cggb = val[7];

              }
              T2 = vec8_blend(T2, (-One_Third_CoxWL) * dVdsat_dVb, condmask_true1);
              {
                Vec8d val = -(((Vec8d ){heres[0]->BSIM3cggb, heres[1]->BSIM3cggb, heres[2]->BSIM3cggb, heres[3]->BSIM3cggb, heres[4]->BSIM3cggb, heres[5]->BSIM3cggb, heres[6]->BSIM3cggb, heres[7]->BSIM3cggb}) + T2);
                if (condmask_true1[0])
                  heres[0]->BSIM3cgsb = val[0];

                if (condmask_true1[1])
                  heres[1]->BSIM3cgsb = val[1];

                if (condmask_true1[2])
                  heres[2]->BSIM3cgsb = val[2];

                if (condmask_true1[3])
                  heres[3]->BSIM3cgsb = val[3];

                if (condmask_true1[4])
                  heres[4]->BSIM3cgsb = val[4];

                if (condmask_true1[5])
                  heres[5]->BSIM3cgsb = val[5];

                if (condmask_true1[6])
                  heres[6]->BSIM3cgsb = val[6];

                if (condmask_true1[7])
                  heres[7]->BSIM3cgsb = val[7];

              }
              {
                if (condmask_true1[0])
                  heres[0]->BSIM3cgdb = 0.0;

                if (condmask_true1[1])
                  heres[1]->BSIM3cgdb = 0.0;

                if (condmask_true1[2])
                  heres[2]->BSIM3cgdb = 0.0;

                if (condmask_true1[3])
                  heres[3]->BSIM3cgdb = 0.0;

                if (condmask_true1[4])
                  heres[4]->BSIM3cgdb = 0.0;

                if (condmask_true1[5])
                  heres[5]->BSIM3cgdb = 0.0;

                if (condmask_true1[6])
                  heres[6]->BSIM3cgdb = 0.0;

                if (condmask_true1[7])
                  heres[7]->BSIM3cgdb = 0.0;

              }
              {
                Vec8d val = (-One_Third_CoxWL) * dVgs_eff_dVg;
                if (condmask_true1[0])
                  heres[0]->BSIM3cdgb = val[0];

                if (condmask_true1[1])
                  heres[1]->BSIM3cdgb = val[1];

                if (condmask_true1[2])
                  heres[2]->BSIM3cdgb = val[2];

                if (condmask_true1[3])
                  heres[3]->BSIM3cdgb = val[3];

                if (condmask_true1[4])
                  heres[4]->BSIM3cdgb = val[4];

                if (condmask_true1[5])
                  heres[5]->BSIM3cdgb = val[5];

                if (condmask_true1[6])
                  heres[6]->BSIM3cdgb = val[6];

                if (condmask_true1[7])
                  heres[7]->BSIM3cdgb = val[7];

              }
              {
                if (condmask_true1[0])
                  heres[0]->BSIM3cddb = 0.0;

                if (condmask_true1[1])
                  heres[1]->BSIM3cddb = 0.0;

                if (condmask_true1[2])
                  heres[2]->BSIM3cddb = 0.0;

                if (condmask_true1[3])
                  heres[3]->BSIM3cddb = 0.0;

                if (condmask_true1[4])
                  heres[4]->BSIM3cddb = 0.0;

                if (condmask_true1[5])
                  heres[5]->BSIM3cddb = 0.0;

                if (condmask_true1[6])
                  heres[6]->BSIM3cddb = 0.0;

                if (condmask_true1[7])
                  heres[7]->BSIM3cddb = 0.0;

              }
              T4 = vec8_blend(T4, One_Third_CoxWL * dVth_dVb, condmask_true1);
              {
                Vec8d val = -(T4 + ((Vec8d ){heres[0]->BSIM3cdgb, heres[1]->BSIM3cdgb, heres[2]->BSIM3cdgb, heres[3]->BSIM3cdgb, heres[4]->BSIM3cdgb, heres[5]->BSIM3cdgb, heres[6]->BSIM3cdgb, heres[7]->BSIM3cdgb}));
                if (condmask_true1[0])
                  heres[0]->BSIM3cdsb = val[0];

                if (condmask_true1[1])
                  heres[1]->BSIM3cdsb = val[1];

                if (condmask_true1[2])
                  heres[2]->BSIM3cdsb = val[2];

                if (condmask_true1[3])
                  heres[3]->BSIM3cdsb = val[3];

                if (condmask_true1[4])
                  heres[4]->BSIM3cdsb = val[4];

                if (condmask_true1[5])
                  heres[5]->BSIM3cdsb = val[5];

                if (condmask_true1[6])
                  heres[6]->BSIM3cdsb = val[6];

                if (condmask_true1[7])
                  heres[7]->BSIM3cdsb = val[7];

              }
              {
                Vec8d val = -(((Vec8d ){heres[0]->BSIM3cggb, heres[1]->BSIM3cggb, heres[2]->BSIM3cggb, heres[3]->BSIM3cggb, heres[4]->BSIM3cggb, heres[5]->BSIM3cggb, heres[6]->BSIM3cggb, heres[7]->BSIM3cggb}) - (Two_Third_CoxWL * dVgs_eff_dVg));
                if (condmask_true1[0])
                  heres[0]->BSIM3cbgb = val[0];

                if (condmask_true1[1])
                  heres[1]->BSIM3cbgb = val[1];

                if (condmask_true1[2])
                  heres[2]->BSIM3cbgb = val[2];

                if (condmask_true1[3])
                  heres[3]->BSIM3cbgb = val[3];

                if (condmask_true1[4])
                  heres[4]->BSIM3cbgb = val[4];

                if (condmask_true1[5])
                  heres[5]->BSIM3cbgb = val[5];

                if (condmask_true1[6])
                  heres[6]->BSIM3cbgb = val[6];

                if (condmask_true1[7])
                  heres[7]->BSIM3cbgb = val[7];

              }
              T3 = vec8_blend(T3, -(T2 + (Two_Third_CoxWL * dVth_dVb)), condmask_true1);
              {
                Vec8d val = -(((Vec8d ){heres[0]->BSIM3cbgb, heres[1]->BSIM3cbgb, heres[2]->BSIM3cbgb, heres[3]->BSIM3cbgb, heres[4]->BSIM3cbgb, heres[5]->BSIM3cbgb, heres[6]->BSIM3cbgb, heres[7]->BSIM3cbgb}) + T3);
                if (condmask_true1[0])
                  heres[0]->BSIM3cbsb = val[0];

                if (condmask_true1[1])
                  heres[1]->BSIM3cbsb = val[1];

                if (condmask_true1[2])
                  heres[2]->BSIM3cbsb = val[2];

                if (condmask_true1[3])
                  heres[3]->BSIM3cbsb = val[3];

                if (condmask_true1[4])
                  heres[4]->BSIM3cbsb = val[4];

                if (condmask_true1[5])
                  heres[5]->BSIM3cbsb = val[5];

                if (condmask_true1[6])
                  heres[6]->BSIM3cbsb = val[6];

                if (condmask_true1[7])
                  heres[7]->BSIM3cbsb = val[7];

              }
              {
                if (condmask_true1[0])
                  heres[0]->BSIM3cbdb = 0.0;

                if (condmask_true1[1])
                  heres[1]->BSIM3cbdb = 0.0;

                if (condmask_true1[2])
                  heres[2]->BSIM3cbdb = 0.0;

                if (condmask_true1[3])
                  heres[3]->BSIM3cbdb = 0.0;

                if (condmask_true1[4])
                  heres[4]->BSIM3cbdb = 0.0;

                if (condmask_true1[5])
                  heres[5]->BSIM3cbdb = 0.0;

                if (condmask_true1[6])
                  heres[6]->BSIM3cbdb = 0.0;

                if (condmask_true1[7])
                  heres[7]->BSIM3cbdb = 0.0;

              }
              {
                Vec8d val = -(qgate + qbulk);
                if (condmask_true1[0])
                  heres[0]->BSIM3qinv = val[0];

                if (condmask_true1[1])
                  heres[1]->BSIM3qinv = val[1];

                if (condmask_true1[2])
                  heres[2]->BSIM3qinv = val[2];

                if (condmask_true1[3])
                  heres[3]->BSIM3qinv = val[3];

                if (condmask_true1[4])
                  heres[4]->BSIM3qinv = val[4];

                if (condmask_true1[5])
                  heres[5]->BSIM3qinv = val[5];

                if (condmask_true1[6])
                  heres[6]->BSIM3qinv = val[6];

                if (condmask_true1[7])
                  heres[7]->BSIM3qinv = val[7];

              }
            }
            {
              Alphaz = vec8_blend(Alphaz, Vgst / Vdsat, condmask_false1);
              T1 = vec8_blend(T1, (2.0 * Vdsat) - Vds, condmask_false1);
              T2 = vec8_blend(T2, Vds / (3.0 * T1), condmask_false1);
              T3 = vec8_blend(T3, T2 * Vds, condmask_false1);
              T9 = vec8_blend(T9, vec8_SIMDTOVECTOR(0.25 * CoxWL), condmask_false1);
              T4 = vec8_blend(T4, T9 * Alphaz, condmask_false1);
              qgate = vec8_blend(qgate, CoxWL * (((Vgs_eff - Vfb) - pParam->BSIM3phi) - (0.5 * (Vds - T3))), condmask_false1);
              T5 = vec8_blend(T5, T3 / T1, condmask_false1);
              {
                Vec8d val = (CoxWL * (1.0 - (T5 * dVdsat_dVg))) * dVgs_eff_dVg;
                if (condmask_false1[0])
                  heres[0]->BSIM3cggb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cggb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cggb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cggb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cggb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cggb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cggb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cggb = val[7];

              }
              tmp = vec8_blend(tmp, ((-CoxWL) * T5) * dVdsat_dVb, condmask_false1);
              {
                Vec8d val = CoxWL * ((T2 - 0.5) + (0.5 * T5));
                if (condmask_false1[0])
                  heres[0]->BSIM3cgdb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cgdb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cgdb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cgdb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cgdb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cgdb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cgdb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cgdb = val[7];

              }
              {
                Vec8d val = -((((Vec8d ){heres[0]->BSIM3cggb, heres[1]->BSIM3cggb, heres[2]->BSIM3cggb, heres[3]->BSIM3cggb, heres[4]->BSIM3cggb, heres[5]->BSIM3cggb, heres[6]->BSIM3cggb, heres[7]->BSIM3cggb}) + ((Vec8d ){heres[0]->BSIM3cgdb, heres[1]->BSIM3cgdb, heres[2]->BSIM3cgdb, heres[3]->BSIM3cgdb, heres[4]->BSIM3cgdb, heres[5]->BSIM3cgdb, heres[6]->BSIM3cgdb, heres[7]->BSIM3cgdb})) + tmp);
                if (condmask_false1[0])
                  heres[0]->BSIM3cgsb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cgsb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cgsb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cgsb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cgsb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cgsb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cgsb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cgsb = val[7];

              }
              T6 = vec8_blend(T6, 1.0 / Vdsat, condmask_false1);
              dAlphaz_dVg = vec8_blend(dAlphaz_dVg, T6 * (1.0 - (Alphaz * dVdsat_dVg)), condmask_false1);
              dAlphaz_dVb = vec8_blend(dAlphaz_dVb, (-T6) * (dVth_dVb + (Alphaz * dVdsat_dVb)), condmask_false1);
              T7 = vec8_blend(T7, T1 + T3, condmask_false1);
              qdrn = vec8_blend(qdrn, (-T4) * T7, condmask_false1);
              qbulk = vec8_blend(qbulk, -((qgate + qdrn) + qdrn), condmask_false1);
              T7 = vec8_blend(T7, T7 * T9, condmask_false1);
              T0 = vec8_blend(T0, T4 * ((2.0 * T5) - 2.0), condmask_false1);
              {
                Vec8d val = ((T0 * dVdsat_dVg) - (T7 * dAlphaz_dVg)) * dVgs_eff_dVg;
                if (condmask_false1[0])
                  heres[0]->BSIM3cdgb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cdgb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cdgb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cdgb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cdgb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cdgb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cdgb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cdgb = val[7];

              }
              T12 = vec8_blend(T12, (T0 * dVdsat_dVb) - (T7 * dAlphaz_dVb), condmask_false1);
              {
                Vec8d val = T4 * ((1.0 - (2.0 * T2)) - T5);
                if (condmask_false1[0])
                  heres[0]->BSIM3cddb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cddb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cddb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cddb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cddb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cddb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cddb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cddb = val[7];

              }
              {
                Vec8d val = -((((Vec8d ){heres[0]->BSIM3cdgb, heres[1]->BSIM3cdgb, heres[2]->BSIM3cdgb, heres[3]->BSIM3cdgb, heres[4]->BSIM3cdgb, heres[5]->BSIM3cdgb, heres[6]->BSIM3cdgb, heres[7]->BSIM3cdgb}) + T12) + ((Vec8d ){heres[0]->BSIM3cddb, heres[1]->BSIM3cddb, heres[2]->BSIM3cddb, heres[3]->BSIM3cddb, heres[4]->BSIM3cddb, heres[5]->BSIM3cddb, heres[6]->BSIM3cddb, heres[7]->BSIM3cddb}));
                if (condmask_false1[0])
                  heres[0]->BSIM3cdsb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cdsb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cdsb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cdsb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cdsb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cdsb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cdsb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cdsb = val[7];

              }
              {
                Vec8d val = -(((Vec8d ){heres[0]->BSIM3cggb, heres[1]->BSIM3cggb, heres[2]->BSIM3cggb, heres[3]->BSIM3cggb, heres[4]->BSIM3cggb, heres[5]->BSIM3cggb, heres[6]->BSIM3cggb, heres[7]->BSIM3cggb}) + (2.0 * ((Vec8d ){heres[0]->BSIM3cdgb, heres[1]->BSIM3cdgb, heres[2]->BSIM3cdgb, heres[3]->BSIM3cdgb, heres[4]->BSIM3cdgb, heres[5]->BSIM3cdgb, heres[6]->BSIM3cdgb, heres[7]->BSIM3cdgb})));
                if (condmask_false1[0])
                  heres[0]->BSIM3cbgb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cbgb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cbgb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cbgb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cbgb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cbgb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cbgb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cbgb = val[7];

              }
              {
                Vec8d val = -(((Vec8d ){heres[0]->BSIM3cgdb, heres[1]->BSIM3cgdb, heres[2]->BSIM3cgdb, heres[3]->BSIM3cgdb, heres[4]->BSIM3cgdb, heres[5]->BSIM3cgdb, heres[6]->BSIM3cgdb, heres[7]->BSIM3cgdb}) + (2.0 * ((Vec8d ){heres[0]->BSIM3cddb, heres[1]->BSIM3cddb, heres[2]->BSIM3cddb, heres[3]->BSIM3cddb, heres[4]->BSIM3cddb, heres[5]->BSIM3cddb, heres[6]->BSIM3cddb, heres[7]->BSIM3cddb})));
                if (condmask_false1[0])
                  heres[0]->BSIM3cbdb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cbdb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cbdb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cbdb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cbdb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cbdb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cbdb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cbdb = val[7];

              }
              {
                Vec8d val = -(((Vec8d ){heres[0]->BSIM3cgsb, heres[1]->BSIM3cgsb, heres[2]->BSIM3cgsb, heres[3]->BSIM3cgsb, heres[4]->BSIM3cgsb, heres[5]->BSIM3cgsb, heres[6]->BSIM3cgsb, heres[7]->BSIM3cgsb}) + (2.0 * ((Vec8d ){heres[0]->BSIM3cdsb, heres[1]->BSIM3cdsb, heres[2]->BSIM3cdsb, heres[3]->BSIM3cdsb, heres[4]->BSIM3cdsb, heres[5]->BSIM3cdsb, heres[6]->BSIM3cdsb, heres[7]->BSIM3cdsb})));
                if (condmask_false1[0])
                  heres[0]->BSIM3cbsb = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3cbsb = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3cbsb = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3cbsb = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3cbsb = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3cbsb = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3cbsb = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3cbsb = val[7];

              }
              {
                Vec8d val = -(qgate + qbulk);
                if (condmask_false1[0])
                  heres[0]->BSIM3qinv = val[0];

                if (condmask_false1[1])
                  heres[1]->BSIM3qinv = val[1];

                if (condmask_false1[2])
                  heres[2]->BSIM3qinv = val[2];

                if (condmask_false1[3])
                  heres[3]->BSIM3qinv = val[3];

                if (condmask_false1[4])
                  heres[4]->BSIM3qinv = val[4];

                if (condmask_false1[5])
                  heres[5]->BSIM3qinv = val[5];

                if (condmask_false1[6])
                  heres[6]->BSIM3qinv = val[6];

                if (condmask_false1[7])
                  heres[7]->BSIM3qinv = val[7];

              }
            }
          }

        }


      }
    }

  }
  else
  {
    if (1)
    {
      Vec8m condmask0 = Vbseff < 0.0;
      Vec8m condmask_true0 = condmask0;
      Vec8m condmask_false0 = ~condmask0;
      {
        VbseffCV = vec8_blend(VbseffCV, Vbseff, condmask_true0);
        dVbseffCV_dVb = vec8_blend(dVbseffCV_dVb, vec8_SIMDTOVECTOR(1.0), condmask_true0);
      }
      {
        VbseffCV = vec8_blend(VbseffCV, pParam->BSIM3phi - Phis, condmask_false0);
        dVbseffCV_dVb = vec8_blend(dVbseffCV_dVb, -dPhis_dVb, condmask_false0);
      }
    }

    CoxWL = (model->BSIM3cox * pParam->BSIM3weffCV) * pParam->BSIM3leffCV;
    noff = n * pParam->BSIM3noff;
    dnoff_dVd = pParam->BSIM3noff * dn_dVd;
    dnoff_dVb = pParam->BSIM3noff * dn_dVb;
    T0 = Vtm * noff;
    voffcv = pParam->BSIM3voffcv;
    VgstNVt = (Vgst - voffcv) / T0;
    ExpVgst = vec8_exp(VgstNVt);
    if (1)
    {
      Vec8m condmask0 = VgstNVt < (-EXP_THRESHOLD);
      Vec8m condmask_true0 = condmask0;
      ExpVgst = vec8_blend(ExpVgst, vec8_SIMDTOVECTOR(MIN_EXP), condmask_true0);
    }

    Vgsteff = T0 * vec8_log(1.0 + ExpVgst);
    if (1)
    {
      Vec8m condmask0 = VgstNVt > EXP_THRESHOLD;
      Vec8m condmask_true0 = condmask0;
      Vec8m condmask_false0 = ~condmask0;
      {
        Vgsteff = vec8_blend(Vgsteff, Vgst - voffcv, condmask_true0);
        dVgsteff_dVg = vec8_blend(dVgsteff_dVg, dVgs_eff_dVg, condmask_true0);
        dVgsteff_dVd = vec8_blend(dVgsteff_dVd, -dVth_dVd, condmask_true0);
        dVgsteff_dVb = vec8_blend(dVgsteff_dVb, -dVth_dVb, condmask_true0);
      }
      if (1)
      {
        Vec8m condmask1 = VgstNVt < (-EXP_THRESHOLD);
        Vec8m condmask_true1 = condmask_false0 & condmask1;
        Vec8m condmask_false1 = condmask_false0 & (~condmask1);
        {
          dVgsteff_dVg = vec8_blend(dVgsteff_dVg, vec8_SIMDTOVECTOR(0.0), condmask_true1);
          dVgsteff_dVd = vec8_blend(dVgsteff_dVd, Vgsteff / noff, condmask_true1);
          dVgsteff_dVb = vec8_blend(dVgsteff_dVb, dVgsteff_dVd * dnoff_dVb, condmask_true1);
          dVgsteff_dVd = vec8_blend(dVgsteff_dVd, dVgsteff_dVd * dnoff_dVd, condmask_true1);
        }
        {
          dVgsteff_dVg = vec8_blend(dVgsteff_dVg, ExpVgst / (1.0 + ExpVgst), condmask_false1);
          dVgsteff_dVd = vec8_blend(dVgsteff_dVd, ((-dVgsteff_dVg) * (dVth_dVd + (((Vgst - voffcv) / noff) * dnoff_dVd))) + ((Vgsteff / noff) * dnoff_dVd), condmask_false1);
          dVgsteff_dVb = vec8_blend(dVgsteff_dVb, ((-dVgsteff_dVg) * (dVth_dVb + (((Vgst - voffcv) / noff) * dnoff_dVb))) + ((Vgsteff / noff) * dnoff_dVb), condmask_false1);
          dVgsteff_dVg = vec8_blend(dVgsteff_dVg, dVgsteff_dVg * dVgs_eff_dVg, condmask_false1);
        }
      }

    }

    if (model->BSIM3capMod == 1)
    {
      Vfb = (Vec8d ){heres[0]->BSIM3vfbzb, heres[1]->BSIM3vfbzb, heres[2]->BSIM3vfbzb, heres[3]->BSIM3vfbzb, heres[4]->BSIM3vfbzb, heres[5]->BSIM3vfbzb, heres[6]->BSIM3vfbzb, heres[7]->BSIM3vfbzb};
      Arg1 = ((Vgs_eff - VbseffCV) - Vfb) - Vgsteff;
      if (1)
      {
        Vec8m condmask0 = Arg1 <= 0.0;
        Vec8m condmask_true0 = condmask0;
        Vec8m condmask_false0 = ~condmask0;
        {
          qgate = vec8_blend(qgate, CoxWL * Arg1, condmask_true0);
          Cgg = vec8_blend(Cgg, CoxWL * (dVgs_eff_dVg - dVgsteff_dVg), condmask_true0);
          Cgd = vec8_blend(Cgd, (-CoxWL) * dVgsteff_dVd, condmask_true0);
          Cgb = vec8_blend(Cgb, (-CoxWL) * (dVbseffCV_dVb + dVgsteff_dVb), condmask_true0);
        }
        {
          T0 = vec8_blend(T0, vec8_SIMDTOVECTOR(0.5 * pParam->BSIM3k1ox), condmask_false0);
          T1 = vec8_blend(T1, vec8_sqrt((T0 * T0) + Arg1), condmask_false0);
          T2 = vec8_blend(T2, (CoxWL * T0) / T1, condmask_false0);
          qgate = vec8_blend(qgate, (CoxWL * pParam->BSIM3k1ox) * (T1 - T0), condmask_false0);
          Cgg = vec8_blend(Cgg, T2 * (dVgs_eff_dVg - dVgsteff_dVg), condmask_false0);
          Cgd = vec8_blend(Cgd, (-T2) * dVgsteff_dVd, condmask_false0);
          Cgb = vec8_blend(Cgb, (-T2) * (dVbseffCV_dVb + dVgsteff_dVb), condmask_false0);
        }
      }

      qbulk = -qgate;
      Cbg = -Cgg;
      Cbd = -Cgd;
      Cbb = -Cgb;
      One_Third_CoxWL = vec8_SIMDTOVECTOR(CoxWL / 3.0);
      Two_Third_CoxWL = 2.0 * One_Third_CoxWL;
      AbulkCV = Abulk0 * pParam->BSIM3abulkCVfactor;
      dAbulkCV_dVb = pParam->BSIM3abulkCVfactor * dAbulk0_dVb;
      VdsatCV = Vgsteff / AbulkCV;
      if (1)
      {
        Vec8m condmask0 = VdsatCV < Vds;
        Vec8m condmask_true0 = condmask0;
        Vec8m condmask_false0 = ~condmask0;
        {
          dVdsatCV_dVg = vec8_blend(dVdsatCV_dVg, 1.0 / AbulkCV, condmask_true0);
          dVdsatCV_dVb = vec8_blend(dVdsatCV_dVb, ((-VdsatCV) * dAbulkCV_dVb) / AbulkCV, condmask_true0);
          T0 = vec8_blend(T0, Vgsteff - (VdsatCV / 3.0), condmask_true0);
          dT0_dVg = vec8_blend(dT0_dVg, 1.0 - (dVdsatCV_dVg / 3.0), condmask_true0);
          dT0_dVb = vec8_blend(dT0_dVb, (-dVdsatCV_dVb) / 3.0, condmask_true0);
          qgate = vec8_blend(qgate, qgate + (CoxWL * T0), condmask_true0);
          Cgg1 = vec8_blend(Cgg1, CoxWL * dT0_dVg, condmask_true0);
          Cgb1 = vec8_blend(Cgb1, (CoxWL * dT0_dVb) + (Cgg1 * dVgsteff_dVb), condmask_true0);
          Cgd1 = vec8_blend(Cgd1, Cgg1 * dVgsteff_dVd, condmask_true0);
          Cgg1 = vec8_blend(Cgg1, Cgg1 * dVgsteff_dVg, condmask_true0);
          Cgg = vec8_blend(Cgg, Cgg + Cgg1, condmask_true0);
          Cgb = vec8_blend(Cgb, Cgb + Cgb1, condmask_true0);
          Cgd = vec8_blend(Cgd, Cgd + Cgd1, condmask_true0);
          T0 = vec8_blend(T0, VdsatCV - Vgsteff, condmask_true0);
          dT0_dVg = vec8_blend(dT0_dVg, dVdsatCV_dVg - 1.0, condmask_true0);
          dT0_dVb = vec8_blend(dT0_dVb, dVdsatCV_dVb, condmask_true0);
          qbulk = vec8_blend(qbulk, qbulk + (One_Third_CoxWL * T0), condmask_true0);
          Cbg1 = vec8_blend(Cbg1, One_Third_CoxWL * dT0_dVg, condmask_true0);
          Cbb1 = vec8_blend(Cbb1, (One_Third_CoxWL * dT0_dVb) + (Cbg1 * dVgsteff_dVb), condmask_true0);
          Cbd1 = vec8_blend(Cbd1, Cbg1 * dVgsteff_dVd, condmask_true0);
          Cbg1 = vec8_blend(Cbg1, Cbg1 * dVgsteff_dVg, condmask_true0);
          Cbg = vec8_blend(Cbg, Cbg + Cbg1, condmask_true0);
          Cbb = vec8_blend(Cbb, Cbb + Cbb1, condmask_true0);
          Cbd = vec8_blend(Cbd, Cbd + Cbd1, condmask_true0);
          if (model->BSIM3xpart > 0.5)
            T0 = vec8_blend(T0, -Two_Third_CoxWL, condmask_true0);
          else
            if (model->BSIM3xpart < 0.5)
            T0 = vec8_blend(T0, vec8_SIMDTOVECTOR((-0.4) * CoxWL), condmask_true0);
          else
            T0 = vec8_blend(T0, -One_Third_CoxWL, condmask_true0);


          qsrc = vec8_blend(qsrc, T0 * Vgsteff, condmask_true0);
          Csg = vec8_blend(Csg, T0 * dVgsteff_dVg, condmask_true0);
          Csb = vec8_blend(Csb, T0 * dVgsteff_dVb, condmask_true0);
          Csd = vec8_blend(Csd, T0 * dVgsteff_dVd, condmask_true0);
          Cgb = vec8_blend(Cgb, Cgb * dVbseff_dVb, condmask_true0);
          Cbb = vec8_blend(Cbb, Cbb * dVbseff_dVb, condmask_true0);
          Csb = vec8_blend(Csb, Csb * dVbseff_dVb, condmask_true0);
        }
        {
          T0 = vec8_blend(T0, AbulkCV * Vds, condmask_false0);
          T1 = vec8_blend(T1, 12.0 * ((Vgsteff - (0.5 * T0)) + 1.e-20), condmask_false0);
          T2 = vec8_blend(T2, Vds / T1, condmask_false0);
          T3 = vec8_blend(T3, T0 * T2, condmask_false0);
          dT3_dVg = vec8_blend(dT3_dVg, (((-12.0) * T2) * T2) * AbulkCV, condmask_false0);
          dT3_dVd = vec8_blend(dT3_dVd, ((((6.0 * T0) * ((4.0 * Vgsteff) - T0)) / T1) / T1) - 0.5, condmask_false0);
          dT3_dVb = vec8_blend(dT3_dVb, (((12.0 * T2) * T2) * dAbulkCV_dVb) * Vgsteff, condmask_false0);
          qgate = vec8_blend(qgate, qgate + (CoxWL * ((Vgsteff - (0.5 * Vds)) + T3)), condmask_false0);
          Cgg1 = vec8_blend(Cgg1, CoxWL * (1.0 + dT3_dVg), condmask_false0);
          Cgb1 = vec8_blend(Cgb1, (CoxWL * dT3_dVb) + (Cgg1 * dVgsteff_dVb), condmask_false0);
          Cgd1 = vec8_blend(Cgd1, (CoxWL * dT3_dVd) + (Cgg1 * dVgsteff_dVd), condmask_false0);
          Cgg1 = vec8_blend(Cgg1, Cgg1 * dVgsteff_dVg, condmask_false0);
          Cgg = vec8_blend(Cgg, Cgg + Cgg1, condmask_false0);
          Cgb = vec8_blend(Cgb, Cgb + Cgb1, condmask_false0);
          Cgd = vec8_blend(Cgd, Cgd + Cgd1, condmask_false0);
          qbulk = vec8_blend(qbulk, qbulk + ((CoxWL * (1.0 - AbulkCV)) * ((0.5 * Vds) - T3)), condmask_false0);
          Cbg1 = vec8_blend(Cbg1, (-CoxWL) * ((1.0 - AbulkCV) * dT3_dVg), condmask_false0);
          Cbb1 = vec8_blend(Cbb1, ((-CoxWL) * (((1.0 - AbulkCV) * dT3_dVb) + (((0.5 * Vds) - T3) * dAbulkCV_dVb))) + (Cbg1 * dVgsteff_dVb), condmask_false0);
          Cbd1 = vec8_blend(Cbd1, (((-CoxWL) * (1.0 - AbulkCV)) * dT3_dVd) + (Cbg1 * dVgsteff_dVd), condmask_false0);
          Cbg1 = vec8_blend(Cbg1, Cbg1 * dVgsteff_dVg, condmask_false0);
          Cbg = vec8_blend(Cbg, Cbg + Cbg1, condmask_false0);
          Cbb = vec8_blend(Cbb, Cbb + Cbb1, condmask_false0);
          Cbd = vec8_blend(Cbd, Cbd + Cbd1, condmask_false0);
          if (model->BSIM3xpart > 0.5)
          {
            T1 = vec8_blend(T1, T1 + T1, condmask_false0);
            qsrc = vec8_blend(qsrc, (-CoxWL) * (((0.5 * Vgsteff) + (0.25 * T0)) - ((T0 * T0) / T1)), condmask_false0);
            Csg = vec8_blend(Csg, (-CoxWL) * (0.5 + (((((24.0 * T0) * Vds) / T1) / T1) * AbulkCV)), condmask_false0);
            Csb = vec8_blend(Csb, ((-CoxWL) * (((0.25 * Vds) * dAbulkCV_dVb) - ((((((12.0 * T0) * Vds) / T1) / T1) * ((4.0 * Vgsteff) - T0)) * dAbulkCV_dVb))) + (Csg * dVgsteff_dVb), condmask_false0);
            Csd = vec8_blend(Csd, ((-CoxWL) * ((0.25 * AbulkCV) - (((((12.0 * AbulkCV) * T0) / T1) / T1) * ((4.0 * Vgsteff) - T0)))) + (Csg * dVgsteff_dVd), condmask_false0);
            Csg = vec8_blend(Csg, Csg * dVgsteff_dVg, condmask_false0);
          }
          else
            if (model->BSIM3xpart < 0.5)
          {
            T1 = vec8_blend(T1, T1 / 12.0, condmask_false0);
            T2 = vec8_blend(T2, (0.5 * CoxWL) / (T1 * T1), condmask_false0);
            T3 = vec8_blend(T3, (Vgsteff * ((((2.0 * T0) * T0) / 3.0) + (Vgsteff * (Vgsteff - ((4.0 * T0) / 3.0))))) - ((((2.0 * T0) * T0) * T0) / 15.0), condmask_false0);
            qsrc = vec8_blend(qsrc, (-T2) * T3, condmask_false0);
            T4 = vec8_blend(T4, (((4.0 / 3.0) * Vgsteff) * (Vgsteff - T0)) + ((0.4 * T0) * T0), condmask_false0);
            Csg = vec8_blend(Csg, (((-2.0) * qsrc) / T1) - (T2 * ((Vgsteff * ((3.0 * Vgsteff) - ((8.0 * T0) / 3.0))) + (((2.0 * T0) * T0) / 3.0))), condmask_false0);
            Csb = vec8_blend(Csb, ((((qsrc / T1) * Vds) + ((T2 * T4) * Vds)) * dAbulkCV_dVb) + (Csg * dVgsteff_dVb), condmask_false0);
            Csd = vec8_blend(Csd, (((qsrc / T1) + (T2 * T4)) * AbulkCV) + (Csg * dVgsteff_dVd), condmask_false0);
            Csg = vec8_blend(Csg, Csg * dVgsteff_dVg, condmask_false0);
          }
          else
          {
            qsrc = vec8_blend(qsrc, (-0.5) * (qgate + qbulk), condmask_false0);
            Csg = vec8_blend(Csg, (-0.5) * (Cgg1 + Cbg1), condmask_false0);
            Csb = vec8_blend(Csb, (-0.5) * (Cgb1 + Cbb1), condmask_false0);
            Csd = vec8_blend(Csd, (-0.5) * (Cgd1 + Cbd1), condmask_false0);
          }


          Cgb = vec8_blend(Cgb, Cgb * dVbseff_dVb, condmask_false0);
          Cbb = vec8_blend(Cbb, Cbb * dVbseff_dVb, condmask_false0);
          Csb = vec8_blend(Csb, Csb * dVbseff_dVb, condmask_false0);
        }
      }

      qdrn = -((qgate + qbulk) + qsrc);
      {
        heres[0]->BSIM3cggb = Cgg[0];
        heres[1]->BSIM3cggb = Cgg[1];
        heres[2]->BSIM3cggb = Cgg[2];
        heres[3]->BSIM3cggb = Cgg[3];
        heres[4]->BSIM3cggb = Cgg[4];
        heres[5]->BSIM3cggb = Cgg[5];
        heres[6]->BSIM3cggb = Cgg[6];
        heres[7]->BSIM3cggb = Cgg[7];
      }
      {
        Vec8d val = -((Cgg + Cgd) + Cgb);
        heres[0]->BSIM3cgsb = val[0];
        heres[1]->BSIM3cgsb = val[1];
        heres[2]->BSIM3cgsb = val[2];
        heres[3]->BSIM3cgsb = val[3];
        heres[4]->BSIM3cgsb = val[4];
        heres[5]->BSIM3cgsb = val[5];
        heres[6]->BSIM3cgsb = val[6];
        heres[7]->BSIM3cgsb = val[7];
      }
      {
        heres[0]->BSIM3cgdb = Cgd[0];
        heres[1]->BSIM3cgdb = Cgd[1];
        heres[2]->BSIM3cgdb = Cgd[2];
        heres[3]->BSIM3cgdb = Cgd[3];
        heres[4]->BSIM3cgdb = Cgd[4];
        heres[5]->BSIM3cgdb = Cgd[5];
        heres[6]->BSIM3cgdb = Cgd[6];
        heres[7]->BSIM3cgdb = Cgd[7];
      }
      {
        Vec8d val = -((Cgg + Cbg) + Csg);
        heres[0]->BSIM3cdgb = val[0];
        heres[1]->BSIM3cdgb = val[1];
        heres[2]->BSIM3cdgb = val[2];
        heres[3]->BSIM3cdgb = val[3];
        heres[4]->BSIM3cdgb = val[4];
        heres[5]->BSIM3cdgb = val[5];
        heres[6]->BSIM3cdgb = val[6];
        heres[7]->BSIM3cdgb = val[7];
      }
      {
        Vec8d val = (((((((Cgg + Cgd) + Cgb) + Cbg) + Cbd) + Cbb) + Csg) + Csd) + Csb;
        heres[0]->BSIM3cdsb = val[0];
        heres[1]->BSIM3cdsb = val[1];
        heres[2]->BSIM3cdsb = val[2];
        heres[3]->BSIM3cdsb = val[3];
        heres[4]->BSIM3cdsb = val[4];
        heres[5]->BSIM3cdsb = val[5];
        heres[6]->BSIM3cdsb = val[6];
        heres[7]->BSIM3cdsb = val[7];
      }
      {
        Vec8d val = -((Cgd + Cbd) + Csd);
        heres[0]->BSIM3cddb = val[0];
        heres[1]->BSIM3cddb = val[1];
        heres[2]->BSIM3cddb = val[2];
        heres[3]->BSIM3cddb = val[3];
        heres[4]->BSIM3cddb = val[4];
        heres[5]->BSIM3cddb = val[5];
        heres[6]->BSIM3cddb = val[6];
        heres[7]->BSIM3cddb = val[7];
      }
      {
        heres[0]->BSIM3cbgb = Cbg[0];
        heres[1]->BSIM3cbgb = Cbg[1];
        heres[2]->BSIM3cbgb = Cbg[2];
        heres[3]->BSIM3cbgb = Cbg[3];
        heres[4]->BSIM3cbgb = Cbg[4];
        heres[5]->BSIM3cbgb = Cbg[5];
        heres[6]->BSIM3cbgb = Cbg[6];
        heres[7]->BSIM3cbgb = Cbg[7];
      }
      {
        Vec8d val = -((Cbg + Cbd) + Cbb);
        heres[0]->BSIM3cbsb = val[0];
        heres[1]->BSIM3cbsb = val[1];
        heres[2]->BSIM3cbsb = val[2];
        heres[3]->BSIM3cbsb = val[3];
        heres[4]->BSIM3cbsb = val[4];
        heres[5]->BSIM3cbsb = val[5];
        heres[6]->BSIM3cbsb = val[6];
        heres[7]->BSIM3cbsb = val[7];
      }
      {
        heres[0]->BSIM3cbdb = Cbd[0];
        heres[1]->BSIM3cbdb = Cbd[1];
        heres[2]->BSIM3cbdb = Cbd[2];
        heres[3]->BSIM3cbdb = Cbd[3];
        heres[4]->BSIM3cbdb = Cbd[4];
        heres[5]->BSIM3cbdb = Cbd[5];
        heres[6]->BSIM3cbdb = Cbd[6];
        heres[7]->BSIM3cbdb = Cbd[7];
      }
      {
        Vec8d val = -(qgate + qbulk);
        heres[0]->BSIM3qinv = val[0];
        heres[1]->BSIM3qinv = val[1];
        heres[2]->BSIM3qinv = val[2];
        heres[3]->BSIM3qinv = val[3];
        heres[4]->BSIM3qinv = val[4];
        heres[5]->BSIM3qinv = val[5];
        heres[6]->BSIM3qinv = val[6];
        heres[7]->BSIM3qinv = val[7];
      }
    }
    else
      if (model->BSIM3capMod == 2)
    {
      Vfb = (Vec8d ){heres[0]->BSIM3vfbzb, heres[1]->BSIM3vfbzb, heres[2]->BSIM3vfbzb, heres[3]->BSIM3vfbzb, heres[4]->BSIM3vfbzb, heres[5]->BSIM3vfbzb, heres[6]->BSIM3vfbzb, heres[7]->BSIM3vfbzb};
      V3 = ((Vfb - Vgs_eff) + VbseffCV) - DELTA_3;
      T0 = V3 * V3;
      T2 = (4.0 * DELTA_3) * Vfb;
      if (1)
      {
        Vec8m condmask0 = Vfb <= 0.0;
        Vec8m condmask_true0 = condmask0;
        Vec8m condmask_false0 = ~condmask0;
        {
          T0 = vec8_blend(T0, T0 - T2, condmask_true0);
          T2 = vec8_blend(T2, vec8_SIMDTOVECTOR(-DELTA_3), condmask_true0);
        }
        {
          T0 = vec8_blend(T0, T0 + T2, condmask_false0);
          T2 = vec8_blend(T2, vec8_SIMDTOVECTOR(DELTA_3), condmask_false0);
        }
      }

      T0 = vec8_sqrt(T0);
      T2 = T2 / T0;
      T1 = 0.5 * (1.0 + (V3 / T0));
      Vfbeff = Vfb - (0.5 * (V3 + T0));
      dVfbeff_dVg = T1 * dVgs_eff_dVg;
      dVfbeff_dVb = (-T1) * dVbseffCV_dVb;
      Qac0 = CoxWL * (Vfbeff - Vfb);
      dQac0_dVg = CoxWL * dVfbeff_dVg;
      dQac0_dVb = CoxWL * dVfbeff_dVb;
      T0 = vec8_SIMDTOVECTOR(0.5 * pParam->BSIM3k1ox);
      T3 = ((Vgs_eff - Vfbeff) - VbseffCV) - Vgsteff;
      if (pParam->BSIM3k1ox == 0.0)
      {
        T1 = vec8_SIMDTOVECTOR(0.0);
        T2 = vec8_SIMDTOVECTOR(0.0);
      }
      else
        if (1)
      {
        Vec8m condmask0 = T3 < 0.0;
        Vec8m condmask_true0 = condmask0;
        Vec8m condmask_false0 = ~condmask0;
        {
          T1 = vec8_blend(T1, T0 + (T3 / pParam->BSIM3k1ox), condmask_true0);
          T2 = vec8_blend(T2, vec8_SIMDTOVECTOR(CoxWL), condmask_true0);
        }
        {
          T1 = vec8_blend(T1, vec8_sqrt((T0 * T0) + T3), condmask_false0);
          T2 = vec8_blend(T2, (CoxWL * T0) / T1, condmask_false0);
        }
      }


      Qsub0 = (CoxWL * pParam->BSIM3k1ox) * (T1 - T0);
      dQsub0_dVg = T2 * ((dVgs_eff_dVg - dVfbeff_dVg) - dVgsteff_dVg);
      dQsub0_dVd = (-T2) * dVgsteff_dVd;
      dQsub0_dVb = (-T2) * ((dVfbeff_dVb + dVbseffCV_dVb) + dVgsteff_dVb);
      AbulkCV = Abulk0 * pParam->BSIM3abulkCVfactor;
      dAbulkCV_dVb = pParam->BSIM3abulkCVfactor * dAbulk0_dVb;
      VdsatCV = Vgsteff / AbulkCV;
      V4 = (VdsatCV - Vds) - DELTA_4;
      T0 = vec8_sqrt((V4 * V4) + ((4.0 * DELTA_4) * VdsatCV));
      VdseffCV = VdsatCV - (0.5 * (V4 + T0));
      T1 = 0.5 * (1.0 + (V4 / T0));
      T2 = DELTA_4 / T0;
      T3 = ((1.0 - T1) - T2) / AbulkCV;
      dVdseffCV_dVg = T3;
      dVdseffCV_dVd = T1;
      dVdseffCV_dVb = ((-T3) * VdsatCV) * dAbulkCV_dVb;
      if (1)
      {
        Vec8m condmask0 = Vds == 0.0;
        Vec8m condmask_true0 = condmask0;
        {
          VdseffCV = vec8_blend(VdseffCV, vec8_SIMDTOVECTOR(0.0), condmask_true0);
          dVdseffCV_dVg = vec8_blend(dVdseffCV_dVg, vec8_SIMDTOVECTOR(0.0), condmask_true0);
          dVdseffCV_dVb = vec8_blend(dVdseffCV_dVb, vec8_SIMDTOVECTOR(0.0), condmask_true0);
        }
      }

      T0 = AbulkCV * VdseffCV;
      T1 = 12.0 * ((Vgsteff - (0.5 * T0)) + 1e-20);
      T2 = VdseffCV / T1;
      T3 = T0 * T2;
      T4 = 1.0 - (((12.0 * T2) * T2) * AbulkCV);
      T5 = (((6.0 * T0) * ((4.0 * Vgsteff) - T0)) / (T1 * T1)) - 0.5;
      T6 = ((12.0 * T2) * T2) * Vgsteff;
      qinoi = (-CoxWL) * ((Vgsteff - (0.5 * T0)) + (AbulkCV * T3));
      qgate = CoxWL * ((Vgsteff - (0.5 * VdseffCV)) + T3);
      Cgg1 = CoxWL * (T4 + (T5 * dVdseffCV_dVg));
      Cgd1 = ((CoxWL * T5) * dVdseffCV_dVd) + (Cgg1 * dVgsteff_dVd);
      Cgb1 = (CoxWL * ((T5 * dVdseffCV_dVb) + (T6 * dAbulkCV_dVb))) + (Cgg1 * dVgsteff_dVb);
      Cgg1 *= dVgsteff_dVg;
      T7 = 1.0 - AbulkCV;
      qbulk = (CoxWL * T7) * ((0.5 * VdseffCV) - T3);
      T4 = (-T7) * (T4 - 1.0);
      T5 = (-T7) * T5;
      T6 = -((T7 * T6) + ((0.5 * VdseffCV) - T3));
      Cbg1 = CoxWL * (T4 + (T5 * dVdseffCV_dVg));
      Cbd1 = ((CoxWL * T5) * dVdseffCV_dVd) + (Cbg1 * dVgsteff_dVd);
      Cbb1 = (CoxWL * ((T5 * dVdseffCV_dVb) + (T6 * dAbulkCV_dVb))) + (Cbg1 * dVgsteff_dVb);
      Cbg1 *= dVgsteff_dVg;
      if (model->BSIM3xpart > 0.5)
      {
        T1 = T1 + T1;
        qsrc = (-CoxWL) * (((0.5 * Vgsteff) + (0.25 * T0)) - ((T0 * T0) / T1));
        T7 = ((4.0 * Vgsteff) - T0) / (T1 * T1);
        T4 = -(0.5 + (((24.0 * T0) * T0) / (T1 * T1)));
        T5 = -((0.25 * AbulkCV) - (((12.0 * AbulkCV) * T0) * T7));
        T6 = -((0.25 * VdseffCV) - (((12.0 * T0) * VdseffCV) * T7));
        Csg = CoxWL * (T4 + (T5 * dVdseffCV_dVg));
        Csd = ((CoxWL * T5) * dVdseffCV_dVd) + (Csg * dVgsteff_dVd);
        Csb = (CoxWL * ((T5 * dVdseffCV_dVb) + (T6 * dAbulkCV_dVb))) + (Csg * dVgsteff_dVb);
        Csg *= dVgsteff_dVg;
      }
      else
        if (model->BSIM3xpart < 0.5)
      {
        T1 = T1 / 12.0;
        T2 = (0.5 * CoxWL) / (T1 * T1);
        T3 = (Vgsteff * ((((2.0 * T0) * T0) / 3.0) + (Vgsteff * (Vgsteff - ((4.0 * T0) / 3.0))))) - ((((2.0 * T0) * T0) * T0) / 15.0);
        qsrc = (-T2) * T3;
        T7 = (((4.0 / 3.0) * Vgsteff) * (Vgsteff - T0)) + ((0.4 * T0) * T0);
        T4 = (((-2.0) * qsrc) / T1) - (T2 * ((Vgsteff * ((3.0 * Vgsteff) - ((8.0 * T0) / 3.0))) + (((2.0 * T0) * T0) / 3.0)));
        T5 = ((qsrc / T1) + (T2 * T7)) * AbulkCV;
        T6 = ((qsrc / T1) * VdseffCV) + ((T2 * T7) * VdseffCV);
        Csg = T4 + (T5 * dVdseffCV_dVg);
        Csd = (T5 * dVdseffCV_dVd) + (Csg * dVgsteff_dVd);
        Csb = ((T5 * dVdseffCV_dVb) + (T6 * dAbulkCV_dVb)) + (Csg * dVgsteff_dVb);
        Csg *= dVgsteff_dVg;
      }
      else
      {
        qsrc = (-0.5) * (qgate + qbulk);
        Csg = (-0.5) * (Cgg1 + Cbg1);
        Csb = (-0.5) * (Cgb1 + Cbb1);
        Csd = (-0.5) * (Cgd1 + Cbd1);
      }


      qgate += Qac0 + Qsub0;
      qbulk -= Qac0 + Qsub0;
      qdrn = -((qgate + qbulk) + qsrc);
      Cgg = (dQac0_dVg + dQsub0_dVg) + Cgg1;
      Cgd = dQsub0_dVd + Cgd1;
      Cgb = (dQac0_dVb + dQsub0_dVb) + Cgb1;
      Cbg = (Cbg1 - dQac0_dVg) - dQsub0_dVg;
      Cbd = Cbd1 - dQsub0_dVd;
      Cbb = (Cbb1 - dQac0_dVb) - dQsub0_dVb;
      Cgb *= dVbseff_dVb;
      Cbb *= dVbseff_dVb;
      Csb *= dVbseff_dVb;
      {
        heres[0]->BSIM3cggb = Cgg[0];
        heres[1]->BSIM3cggb = Cgg[1];
        heres[2]->BSIM3cggb = Cgg[2];
        heres[3]->BSIM3cggb = Cgg[3];
        heres[4]->BSIM3cggb = Cgg[4];
        heres[5]->BSIM3cggb = Cgg[5];
        heres[6]->BSIM3cggb = Cgg[6];
        heres[7]->BSIM3cggb = Cgg[7];
      }
      {
        Vec8d val = -((Cgg + Cgd) + Cgb);
        heres[0]->BSIM3cgsb = val[0];
        heres[1]->BSIM3cgsb = val[1];
        heres[2]->BSIM3cgsb = val[2];
        heres[3]->BSIM3cgsb = val[3];
        heres[4]->BSIM3cgsb = val[4];
        heres[5]->BSIM3cgsb = val[5];
        heres[6]->BSIM3cgsb = val[6];
        heres[7]->BSIM3cgsb = val[7];
      }
      {
        heres[0]->BSIM3cgdb = Cgd[0];
        heres[1]->BSIM3cgdb = Cgd[1];
        heres[2]->BSIM3cgdb = Cgd[2];
        heres[3]->BSIM3cgdb = Cgd[3];
        heres[4]->BSIM3cgdb = Cgd[4];
        heres[5]->BSIM3cgdb = Cgd[5];
        heres[6]->BSIM3cgdb = Cgd[6];
        heres[7]->BSIM3cgdb = Cgd[7];
      }
      {
        Vec8d val = -((Cgg + Cbg) + Csg);
        heres[0]->BSIM3cdgb = val[0];
        heres[1]->BSIM3cdgb = val[1];
        heres[2]->BSIM3cdgb = val[2];
        heres[3]->BSIM3cdgb = val[3];
        heres[4]->BSIM3cdgb = val[4];
        heres[5]->BSIM3cdgb = val[5];
        heres[6]->BSIM3cdgb = val[6];
        heres[7]->BSIM3cdgb = val[7];
      }
      {
        Vec8d val = (((((((Cgg + Cgd) + Cgb) + Cbg) + Cbd) + Cbb) + Csg) + Csd) + Csb;
        heres[0]->BSIM3cdsb = val[0];
        heres[1]->BSIM3cdsb = val[1];
        heres[2]->BSIM3cdsb = val[2];
        heres[3]->BSIM3cdsb = val[3];
        heres[4]->BSIM3cdsb = val[4];
        heres[5]->BSIM3cdsb = val[5];
        heres[6]->BSIM3cdsb = val[6];
        heres[7]->BSIM3cdsb = val[7];
      }
      {
        Vec8d val = -((Cgd + Cbd) + Csd);
        heres[0]->BSIM3cddb = val[0];
        heres[1]->BSIM3cddb = val[1];
        heres[2]->BSIM3cddb = val[2];
        heres[3]->BSIM3cddb = val[3];
        heres[4]->BSIM3cddb = val[4];
        heres[5]->BSIM3cddb = val[5];
        heres[6]->BSIM3cddb = val[6];
        heres[7]->BSIM3cddb = val[7];
      }
      {
        heres[0]->BSIM3cbgb = Cbg[0];
        heres[1]->BSIM3cbgb = Cbg[1];
        heres[2]->BSIM3cbgb = Cbg[2];
        heres[3]->BSIM3cbgb = Cbg[3];
        heres[4]->BSIM3cbgb = Cbg[4];
        heres[5]->BSIM3cbgb = Cbg[5];
        heres[6]->BSIM3cbgb = Cbg[6];
        heres[7]->BSIM3cbgb = Cbg[7];
      }
      {
        Vec8d val = -((Cbg + Cbd) + Cbb);
        heres[0]->BSIM3cbsb = val[0];
        heres[1]->BSIM3cbsb = val[1];
        heres[2]->BSIM3cbsb = val[2];
        heres[3]->BSIM3cbsb = val[3];
        heres[4]->BSIM3cbsb = val[4];
        heres[5]->BSIM3cbsb = val[5];
        heres[6]->BSIM3cbsb = val[6];
        heres[7]->BSIM3cbsb = val[7];
      }
      {
        heres[0]->BSIM3cbdb = Cbd[0];
        heres[1]->BSIM3cbdb = Cbd[1];
        heres[2]->BSIM3cbdb = Cbd[2];
        heres[3]->BSIM3cbdb = Cbd[3];
        heres[4]->BSIM3cbdb = Cbd[4];
        heres[5]->BSIM3cbdb = Cbd[5];
        heres[6]->BSIM3cbdb = Cbd[6];
        heres[7]->BSIM3cbdb = Cbd[7];
      }
      {
        heres[0]->BSIM3qinv = qinoi[0];
        heres[1]->BSIM3qinv = qinoi[1];
        heres[2]->BSIM3qinv = qinoi[2];
        heres[3]->BSIM3qinv = qinoi[3];
        heres[4]->BSIM3qinv = qinoi[4];
        heres[5]->BSIM3qinv = qinoi[5];
        heres[6]->BSIM3qinv = qinoi[6];
        heres[7]->BSIM3qinv = qinoi[7];
      }
    }
    else
      if (model->BSIM3capMod == 3)
    {
      V3 = ((((Vec8d ){heres[0]->BSIM3vfbzb, heres[1]->BSIM3vfbzb, heres[2]->BSIM3vfbzb, heres[3]->BSIM3vfbzb, heres[4]->BSIM3vfbzb, heres[5]->BSIM3vfbzb, heres[6]->BSIM3vfbzb, heres[7]->BSIM3vfbzb}) - Vgs_eff) + VbseffCV) - DELTA_3;
      if (1)
      {
        Vec8m condmask0 = ((Vec8d ){heres[0]->BSIM3vfbzb, heres[1]->BSIM3vfbzb, heres[2]->BSIM3vfbzb, heres[3]->BSIM3vfbzb, heres[4]->BSIM3vfbzb, heres[5]->BSIM3vfbzb, heres[6]->BSIM3vfbzb, heres[7]->BSIM3vfbzb}) <= 0.0;
        Vec8m condmask_true0 = condmask0;
        Vec8m condmask_false0 = ~condmask0;
        {
          T0 = vec8_blend(T0, vec8_sqrt((V3 * V3) - ((4.0 * DELTA_3) * ((Vec8d ){heres[0]->BSIM3vfbzb, heres[1]->BSIM3vfbzb, heres[2]->BSIM3vfbzb, heres[3]->BSIM3vfbzb, heres[4]->BSIM3vfbzb, heres[5]->BSIM3vfbzb, heres[6]->BSIM3vfbzb, heres[7]->BSIM3vfbzb}))), condmask_true0);
          T2 = vec8_blend(T2, (-DELTA_3) / T0, condmask_true0);
        }
        {
          T0 = vec8_blend(T0, vec8_sqrt((V3 * V3) + ((4.0 * DELTA_3) * ((Vec8d ){heres[0]->BSIM3vfbzb, heres[1]->BSIM3vfbzb, heres[2]->BSIM3vfbzb, heres[3]->BSIM3vfbzb, heres[4]->BSIM3vfbzb, heres[5]->BSIM3vfbzb, heres[6]->BSIM3vfbzb, heres[7]->BSIM3vfbzb}))), condmask_false0);
          T2 = vec8_blend(T2, DELTA_3 / T0, condmask_false0);
        }
      }

      T1 = 0.5 * (1.0 + (V3 / T0));
      Vfbeff = ((Vec8d ){heres[0]->BSIM3vfbzb, heres[1]->BSIM3vfbzb, heres[2]->BSIM3vfbzb, heres[3]->BSIM3vfbzb, heres[4]->BSIM3vfbzb, heres[5]->BSIM3vfbzb, heres[6]->BSIM3vfbzb, heres[7]->BSIM3vfbzb}) - (0.5 * (V3 + T0));
      dVfbeff_dVg = T1 * dVgs_eff_dVg;
      dVfbeff_dVb = (-T1) * dVbseffCV_dVb;
      Cox = model->BSIM3cox;
      Tox = 1.0e8 * model->BSIM3tox;
      T0 = ((Vgs_eff - VbseffCV) - ((Vec8d ){heres[0]->BSIM3vfbzb, heres[1]->BSIM3vfbzb, heres[2]->BSIM3vfbzb, heres[3]->BSIM3vfbzb, heres[4]->BSIM3vfbzb, heres[5]->BSIM3vfbzb, heres[6]->BSIM3vfbzb, heres[7]->BSIM3vfbzb})) / Tox;
      dT0_dVg = dVgs_eff_dVg / Tox;
      dT0_dVb = (-dVbseffCV_dVb) / Tox;
      tmp = T0 * pParam->BSIM3acde;
      dTcen_dVg = (dTcen_dVb = vec8_SIMDTOVECTOR(0.0));
      if (1)
      {
        Vec8m condmask0 = ((-EXP_THRESHOLD) < tmp) & (tmp < EXP_THRESHOLD);
        Vec8m condmask_true0 = condmask0;
        {
          Tcen = vec8_blend(Tcen, pParam->BSIM3ldeb * vec8_exp(tmp), condmask_true0);
          dTcen_dVg = vec8_blend(dTcen_dVg, pParam->BSIM3acde * Tcen, condmask_true0);
          dTcen_dVb = vec8_blend(dTcen_dVb, dTcen_dVg * dT0_dVb, condmask_true0);
          dTcen_dVg = vec8_blend(dTcen_dVg, dTcen_dVg * dT0_dVg, condmask_true0);
        }
      }

      if (1)
      {
        Vec8m condmask0 = tmp <= (-EXP_THRESHOLD);
        Vec8m condmask_true0 = condmask0;
        {
          Tcen = vec8_blend(Tcen, vec8_SIMDTOVECTOR(pParam->BSIM3ldeb * MIN_EXP), condmask_true0);
        }
      }

      if (1)
      {
        Vec8m condmask0 = tmp >= EXP_THRESHOLD;
        Vec8m condmask_true0 = condmask0;
        {
          Tcen = vec8_blend(Tcen, vec8_SIMDTOVECTOR(pParam->BSIM3ldeb * MAX_EXP), condmask_true0);
        }
      }

      LINK = 1.0e-3 * model->BSIM3tox;
      V3 = (pParam->BSIM3ldeb - Tcen) - LINK;
      V4 = vec8_sqrt((V3 * V3) + ((4.0 * LINK) * pParam->BSIM3ldeb));
      Tcen = pParam->BSIM3ldeb - (0.5 * (V3 + V4));
      T1 = 0.5 * (1.0 + (V3 / V4));
      dTcen_dVg *= T1;
      dTcen_dVb *= T1;
      Ccen = EPSSI / Tcen;
      T2 = Cox / (Cox + Ccen);
      Coxeff = T2 * Ccen;
      T3 = (-Ccen) / Tcen;
      dCoxeff_dVg = (T2 * T2) * T3;
      dCoxeff_dVb = dCoxeff_dVg * dTcen_dVb;
      dCoxeff_dVg *= dTcen_dVg;
      CoxWLcen = (CoxWL * Coxeff) / Cox;
      Qac0 = CoxWLcen * (Vfbeff - ((Vec8d ){heres[0]->BSIM3vfbzb, heres[1]->BSIM3vfbzb, heres[2]->BSIM3vfbzb, heres[3]->BSIM3vfbzb, heres[4]->BSIM3vfbzb, heres[5]->BSIM3vfbzb, heres[6]->BSIM3vfbzb, heres[7]->BSIM3vfbzb}));
      QovCox = Qac0 / Coxeff;
      dQac0_dVg = (CoxWLcen * dVfbeff_dVg) + (QovCox * dCoxeff_dVg);
      dQac0_dVb = (CoxWLcen * dVfbeff_dVb) + (QovCox * dCoxeff_dVb);
      T0 = vec8_SIMDTOVECTOR(0.5 * pParam->BSIM3k1ox);
      T3 = ((Vgs_eff - Vfbeff) - VbseffCV) - Vgsteff;
      if (pParam->BSIM3k1ox == 0.0)
      {
        T1 = vec8_SIMDTOVECTOR(0.0);
        T2 = vec8_SIMDTOVECTOR(0.0);
      }
      else
        if (1)
      {
        Vec8m condmask0 = T3 < 0.0;
        Vec8m condmask_true0 = condmask0;
        Vec8m condmask_false0 = ~condmask0;
        {
          T1 = vec8_blend(T1, T0 + (T3 / pParam->BSIM3k1ox), condmask_true0);
          T2 = vec8_blend(T2, CoxWLcen, condmask_true0);
        }
        {
          T1 = vec8_blend(T1, vec8_sqrt((T0 * T0) + T3), condmask_false0);
          T2 = vec8_blend(T2, (CoxWLcen * T0) / T1, condmask_false0);
        }
      }


      Qsub0 = (CoxWLcen * pParam->BSIM3k1ox) * (T1 - T0);
      QovCox = Qsub0 / Coxeff;
      dQsub0_dVg = (T2 * ((dVgs_eff_dVg - dVfbeff_dVg) - dVgsteff_dVg)) + (QovCox * dCoxeff_dVg);
      dQsub0_dVd = (-T2) * dVgsteff_dVd;
      dQsub0_dVb = ((-T2) * ((dVfbeff_dVb + dVbseffCV_dVb) + dVgsteff_dVb)) + (QovCox * dCoxeff_dVb);
      if (pParam->BSIM3k1ox <= 0.0)
      {
        Denomi = vec8_SIMDTOVECTOR((0.25 * pParam->BSIM3moin) * Vtm);
        T0 = vec8_SIMDTOVECTOR(0.5 * pParam->BSIM3sqrtPhi);
      }
      else
      {
        Denomi = vec8_SIMDTOVECTOR(((pParam->BSIM3moin * Vtm) * pParam->BSIM3k1ox) * pParam->BSIM3k1ox);
        T0 = vec8_SIMDTOVECTOR(pParam->BSIM3k1ox * pParam->BSIM3sqrtPhi);
      }

      T1 = (2.0 * T0) + Vgsteff;
      DeltaPhi = Vtm * vec8_log(1.0 + ((T1 * Vgsteff) / Denomi));
      dDeltaPhi_dVg = ((2.0 * Vtm) * (T1 - T0)) / (Denomi + (T1 * Vgsteff));
      T0 = (Vgsteff - DeltaPhi) - 0.001;
      dT0_dVg = 1.0 - dDeltaPhi_dVg;
      T1 = vec8_sqrt((T0 * T0) + (Vgsteff * 0.004));
      VgDP = 0.5 * (T0 + T1);
      dVgDP_dVg = 0.5 * (dT0_dVg + (((T0 * dT0_dVg) + 0.002) / T1));
      T3 = 4.0 * ((Vth - ((Vec8d ){heres[0]->BSIM3vfbzb, heres[1]->BSIM3vfbzb, heres[2]->BSIM3vfbzb, heres[3]->BSIM3vfbzb, heres[4]->BSIM3vfbzb, heres[5]->BSIM3vfbzb, heres[6]->BSIM3vfbzb, heres[7]->BSIM3vfbzb})) - pParam->BSIM3phi);
      Tox += Tox;
      if (1)
      {
        Vec8m condmask0 = T3 >= 0.0;
        Vec8m condmask_true0 = condmask0;
        Vec8m condmask_false0 = ~condmask0;
        {
          T0 = vec8_blend(T0, (Vgsteff + T3) / Tox, condmask_true0);
          dT0_dVd = vec8_blend(dT0_dVd, (dVgsteff_dVd + (4.0 * dVth_dVd)) / Tox, condmask_true0);
          dT0_dVb = vec8_blend(dT0_dVb, (dVgsteff_dVb + (4.0 * dVth_dVb)) / Tox, condmask_true0);
        }
        {
          T0 = vec8_blend(T0, (Vgsteff + 1.0e-20) / Tox, condmask_false0);
          dT0_dVd = vec8_blend(dT0_dVd, dVgsteff_dVd / Tox, condmask_false0);
          dT0_dVb = vec8_blend(dT0_dVb, dVgsteff_dVb / Tox, condmask_false0);
        }
      }

      tmp = vec8_exp(0.7 * vec8_log(T0));
      T1 = 1.0 + tmp;
      T2 = (0.7 * tmp) / (T0 * Tox);
      Tcen = 1.9e-9 / T1;
      dTcen_dVg = (((-1.9e-9) * T2) / T1) / T1;
      dTcen_dVd = Tox * dTcen_dVg;
      dTcen_dVb = dTcen_dVd * dT0_dVb;
      dTcen_dVd *= dT0_dVd;
      dTcen_dVg *= dVgsteff_dVg;
      Ccen = EPSSI / Tcen;
      T0 = Cox / (Cox + Ccen);
      Coxeff = T0 * Ccen;
      T1 = (-Ccen) / Tcen;
      dCoxeff_dVg = (T0 * T0) * T1;
      dCoxeff_dVd = dCoxeff_dVg * dTcen_dVd;
      dCoxeff_dVb = dCoxeff_dVg * dTcen_dVb;
      dCoxeff_dVg *= dTcen_dVg;
      CoxWLcen = (CoxWL * Coxeff) / Cox;
      AbulkCV = Abulk0 * pParam->BSIM3abulkCVfactor;
      dAbulkCV_dVb = pParam->BSIM3abulkCVfactor * dAbulk0_dVb;
      VdsatCV = VgDP / AbulkCV;
      T0 = (VdsatCV - Vds) - DELTA_4;
      dT0_dVg = dVgDP_dVg / AbulkCV;
      dT0_dVb = ((-VdsatCV) * dAbulkCV_dVb) / AbulkCV;
      T1 = vec8_sqrt((T0 * T0) + ((4.0 * DELTA_4) * VdsatCV));
      dT1_dVg = ((T0 + DELTA_4) + DELTA_4) / T1;
      dT1_dVd = (-T0) / T1;
      dT1_dVb = dT1_dVg * dT0_dVb;
      dT1_dVg *= dT0_dVg;
      if (1)
      {
        Vec8m condmask0 = T0 >= 0.0;
        Vec8m condmask_true0 = condmask0;
        Vec8m condmask_false0 = ~condmask0;
        {
          VdseffCV = vec8_blend(VdseffCV, VdsatCV - (0.5 * (T0 + T1)), condmask_true0);
          dVdseffCV_dVg = vec8_blend(dVdseffCV_dVg, 0.5 * (dT0_dVg - dT1_dVg), condmask_true0);
          dVdseffCV_dVd = vec8_blend(dVdseffCV_dVd, 0.5 * (1.0 - dT1_dVd), condmask_true0);
          dVdseffCV_dVb = vec8_blend(dVdseffCV_dVb, 0.5 * (dT0_dVb - dT1_dVb), condmask_true0);
        }
        {
          T3 = vec8_blend(T3, (DELTA_4 + DELTA_4) / (T1 - T0), condmask_false0);
          T4 = vec8_blend(T4, 1.0 - T3, condmask_false0);
          T5 = vec8_blend(T5, (VdsatCV * T3) / (T1 - T0), condmask_false0);
          VdseffCV = vec8_blend(VdseffCV, VdsatCV * T4, condmask_false0);
          dVdseffCV_dVg = vec8_blend(dVdseffCV_dVg, (dT0_dVg * T4) + (T5 * (dT1_dVg - dT0_dVg)), condmask_false0);
          dVdseffCV_dVd = vec8_blend(dVdseffCV_dVd, T5 * (dT1_dVd + 1.0), condmask_false0);
          dVdseffCV_dVb = vec8_blend(dVdseffCV_dVb, (dT0_dVb * (1.0 - T5)) + (T5 * dT1_dVb), condmask_false0);
        }
      }

      if (1)
      {
        Vec8m condmask0 = Vds == 0.0;
        Vec8m condmask_true0 = condmask0;
        {
          VdseffCV = vec8_blend(VdseffCV, vec8_SIMDTOVECTOR(0.0), condmask_true0);
          dVdseffCV_dVg = vec8_blend(dVdseffCV_dVg, vec8_SIMDTOVECTOR(0.0), condmask_true0);
          dVdseffCV_dVb = vec8_blend(dVdseffCV_dVb, vec8_SIMDTOVECTOR(0.0), condmask_true0);
        }
      }

      T0 = AbulkCV * VdseffCV;
      T1 = VgDP;
      T2 = 12.0 * ((T1 - (0.5 * T0)) + 1.0e-20);
      T3 = T0 / T2;
      T4 = 1.0 - ((12.0 * T3) * T3);
      T5 = AbulkCV * ((((6.0 * T0) * ((4.0 * T1) - T0)) / (T2 * T2)) - 0.5);
      T6 = (T5 * VdseffCV) / AbulkCV;
      qgate = (qinoi = CoxWLcen * (T1 - (T0 * (0.5 - T3))));
      QovCox = qgate / Coxeff;
      Cgg1 = CoxWLcen * ((T4 * dVgDP_dVg) + (T5 * dVdseffCV_dVg));
      Cgd1 = (((CoxWLcen * T5) * dVdseffCV_dVd) + (Cgg1 * dVgsteff_dVd)) + (QovCox * dCoxeff_dVd);
      Cgb1 = ((CoxWLcen * ((T5 * dVdseffCV_dVb) + (T6 * dAbulkCV_dVb))) + (Cgg1 * dVgsteff_dVb)) + (QovCox * dCoxeff_dVb);
      Cgg1 = (Cgg1 * dVgsteff_dVg) + (QovCox * dCoxeff_dVg);
      T7 = 1.0 - AbulkCV;
      T8 = T2 * T2;
      T9 = (((12.0 * T7) * T0) * T0) / (T8 * AbulkCV);
      T10 = T9 * dVgDP_dVg;
      T11 = ((-T7) * T5) / AbulkCV;
      T12 = -(((T9 * T1) / AbulkCV) + (VdseffCV * (0.5 - (T0 / T2))));
      qbulk = (CoxWLcen * T7) * ((0.5 * VdseffCV) - ((T0 * VdseffCV) / T2));
      QovCox = qbulk / Coxeff;
      Cbg1 = CoxWLcen * (T10 + (T11 * dVdseffCV_dVg));
      Cbd1 = (((CoxWLcen * T11) * dVdseffCV_dVd) + (Cbg1 * dVgsteff_dVd)) + (QovCox * dCoxeff_dVd);
      Cbb1 = ((CoxWLcen * ((T11 * dVdseffCV_dVb) + (T12 * dAbulkCV_dVb))) + (Cbg1 * dVgsteff_dVb)) + (QovCox * dCoxeff_dVb);
      Cbg1 = (Cbg1 * dVgsteff_dVg) + (QovCox * dCoxeff_dVg);
      if (model->BSIM3xpart > 0.5)
      {
        qsrc = (-CoxWLcen) * (((T1 / 2.0) + (T0 / 4.0)) - (((0.5 * T0) * T0) / T2));
        QovCox = qsrc / Coxeff;
        T2 += T2;
        T3 = T2 * T2;
        T7 = -(0.25 - (((12.0 * T0) * ((4.0 * T1) - T0)) / T3));
        T4 = (-(0.5 + (((24.0 * T0) * T0) / T3))) * dVgDP_dVg;
        T5 = T7 * AbulkCV;
        T6 = T7 * VdseffCV;
        Csg = CoxWLcen * (T4 + (T5 * dVdseffCV_dVg));
        Csd = (((CoxWLcen * T5) * dVdseffCV_dVd) + (Csg * dVgsteff_dVd)) + (QovCox * dCoxeff_dVd);
        Csb = ((CoxWLcen * ((T5 * dVdseffCV_dVb) + (T6 * dAbulkCV_dVb))) + (Csg * dVgsteff_dVb)) + (QovCox * dCoxeff_dVb);
        Csg = (Csg * dVgsteff_dVg) + (QovCox * dCoxeff_dVg);
      }
      else
        if (model->BSIM3xpart < 0.5)
      {
        T2 = T2 / 12.0;
        T3 = (0.5 * CoxWLcen) / (T2 * T2);
        T4 = (T1 * ((((2.0 * T0) * T0) / 3.0) + (T1 * (T1 - ((4.0 * T0) / 3.0))))) - ((((2.0 * T0) * T0) * T0) / 15.0);
        qsrc = (-T3) * T4;
        QovCox = qsrc / Coxeff;
        T8 = (((4.0 / 3.0) * T1) * (T1 - T0)) + ((0.4 * T0) * T0);
        T5 = (((-2.0) * qsrc) / T2) - (T3 * ((T1 * ((3.0 * T1) - ((8.0 * T0) / 3.0))) + (((2.0 * T0) * T0) / 3.0)));
        T6 = AbulkCV * ((qsrc / T2) + (T3 * T8));
        T7 = (T6 * VdseffCV) / AbulkCV;
        Csg = (T5 * dVgDP_dVg) + (T6 * dVdseffCV_dVg);
        Csd = ((Csg * dVgsteff_dVd) + (T6 * dVdseffCV_dVd)) + (QovCox * dCoxeff_dVd);
        Csb = (((Csg * dVgsteff_dVb) + (T6 * dVdseffCV_dVb)) + (T7 * dAbulkCV_dVb)) + (QovCox * dCoxeff_dVb);
        Csg = (Csg * dVgsteff_dVg) + (QovCox * dCoxeff_dVg);
      }
      else
      {
        qsrc = (-0.5) * qgate;
        Csg = (-0.5) * Cgg1;
        Csd = (-0.5) * Cgd1;
        Csb = (-0.5) * Cgb1;
      }


      qgate += (Qac0 + Qsub0) - qbulk;
      qbulk -= Qac0 + Qsub0;
      qdrn = -((qgate + qbulk) + qsrc);
      Cbg = (Cbg1 - dQac0_dVg) - dQsub0_dVg;
      Cbd = Cbd1 - dQsub0_dVd;
      Cbb = (Cbb1 - dQac0_dVb) - dQsub0_dVb;
      Cgg = Cgg1 - Cbg;
      Cgd = Cgd1 - Cbd;
      Cgb = Cgb1 - Cbb;
      Cgb *= dVbseff_dVb;
      Cbb *= dVbseff_dVb;
      Csb *= dVbseff_dVb;
      {
        heres[0]->BSIM3cggb = Cgg[0];
        heres[1]->BSIM3cggb = Cgg[1];
        heres[2]->BSIM3cggb = Cgg[2];
        heres[3]->BSIM3cggb = Cgg[3];
        heres[4]->BSIM3cggb = Cgg[4];
        heres[5]->BSIM3cggb = Cgg[5];
        heres[6]->BSIM3cggb = Cgg[6];
        heres[7]->BSIM3cggb = Cgg[7];
      }
      {
        Vec8d val = -((Cgg + Cgd) + Cgb);
        heres[0]->BSIM3cgsb = val[0];
        heres[1]->BSIM3cgsb = val[1];
        heres[2]->BSIM3cgsb = val[2];
        heres[3]->BSIM3cgsb = val[3];
        heres[4]->BSIM3cgsb = val[4];
        heres[5]->BSIM3cgsb = val[5];
        heres[6]->BSIM3cgsb = val[6];
        heres[7]->BSIM3cgsb = val[7];
      }
      {
        heres[0]->BSIM3cgdb = Cgd[0];
        heres[1]->BSIM3cgdb = Cgd[1];
        heres[2]->BSIM3cgdb = Cgd[2];
        heres[3]->BSIM3cgdb = Cgd[3];
        heres[4]->BSIM3cgdb = Cgd[4];
        heres[5]->BSIM3cgdb = Cgd[5];
        heres[6]->BSIM3cgdb = Cgd[6];
        heres[7]->BSIM3cgdb = Cgd[7];
      }
      {
        Vec8d val = -((Cgg + Cbg) + Csg);
        heres[0]->BSIM3cdgb = val[0];
        heres[1]->BSIM3cdgb = val[1];
        heres[2]->BSIM3cdgb = val[2];
        heres[3]->BSIM3cdgb = val[3];
        heres[4]->BSIM3cdgb = val[4];
        heres[5]->BSIM3cdgb = val[5];
        heres[6]->BSIM3cdgb = val[6];
        heres[7]->BSIM3cdgb = val[7];
      }
      {
        Vec8d val = (((((((Cgg + Cgd) + Cgb) + Cbg) + Cbd) + Cbb) + Csg) + Csd) + Csb;
        heres[0]->BSIM3cdsb = val[0];
        heres[1]->BSIM3cdsb = val[1];
        heres[2]->BSIM3cdsb = val[2];
        heres[3]->BSIM3cdsb = val[3];
        heres[4]->BSIM3cdsb = val[4];
        heres[5]->BSIM3cdsb = val[5];
        heres[6]->BSIM3cdsb = val[6];
        heres[7]->BSIM3cdsb = val[7];
      }
      {
        Vec8d val = -((Cgd + Cbd) + Csd);
        heres[0]->BSIM3cddb = val[0];
        heres[1]->BSIM3cddb = val[1];
        heres[2]->BSIM3cddb = val[2];
        heres[3]->BSIM3cddb = val[3];
        heres[4]->BSIM3cddb = val[4];
        heres[5]->BSIM3cddb = val[5];
        heres[6]->BSIM3cddb = val[6];
        heres[7]->BSIM3cddb = val[7];
      }
      {
        heres[0]->BSIM3cbgb = Cbg[0];
        heres[1]->BSIM3cbgb = Cbg[1];
        heres[2]->BSIM3cbgb = Cbg[2];
        heres[3]->BSIM3cbgb = Cbg[3];
        heres[4]->BSIM3cbgb = Cbg[4];
        heres[5]->BSIM3cbgb = Cbg[5];
        heres[6]->BSIM3cbgb = Cbg[6];
        heres[7]->BSIM3cbgb = Cbg[7];
      }
      {
        Vec8d val = -((Cbg + Cbd) + Cbb);
        heres[0]->BSIM3cbsb = val[0];
        heres[1]->BSIM3cbsb = val[1];
        heres[2]->BSIM3cbsb = val[2];
        heres[3]->BSIM3cbsb = val[3];
        heres[4]->BSIM3cbsb = val[4];
        heres[5]->BSIM3cbsb = val[5];
        heres[6]->BSIM3cbsb = val[6];
        heres[7]->BSIM3cbsb = val[7];
      }
      {
        heres[0]->BSIM3cbdb = Cbd[0];
        heres[1]->BSIM3cbdb = Cbd[1];
        heres[2]->BSIM3cbdb = Cbd[2];
        heres[3]->BSIM3cbdb = Cbd[3];
        heres[4]->BSIM3cbdb = Cbd[4];
        heres[5]->BSIM3cbdb = Cbd[5];
        heres[6]->BSIM3cbdb = Cbd[6];
        heres[7]->BSIM3cbdb = Cbd[7];
      }
      {
        Vec8d val = -qinoi;
        heres[0]->BSIM3qinv = val[0];
        heres[1]->BSIM3qinv = val[1];
        heres[2]->BSIM3qinv = val[2];
        heres[3]->BSIM3qinv = val[3];
        heres[4]->BSIM3qinv = val[4];
        heres[5]->BSIM3qinv = val[5];
        heres[6]->BSIM3qinv = val[6];
        heres[7]->BSIM3qinv = val[7];
      }
    }



  }


  finished:
  {
    heres[0]->BSIM3qgate = qgate[0];
    heres[1]->BSIM3qgate = qgate[1];
    heres[2]->BSIM3qgate = qgate[2];
    heres[3]->BSIM3qgate = qgate[3];
    heres[4]->BSIM3qgate = qgate[4];
    heres[5]->BSIM3qgate = qgate[5];
    heres[6]->BSIM3qgate = qgate[6];
    heres[7]->BSIM3qgate = qgate[7];
  }

  {
    heres[0]->BSIM3qbulk = qbulk[0];
    heres[1]->BSIM3qbulk = qbulk[1];
    heres[2]->BSIM3qbulk = qbulk[2];
    heres[3]->BSIM3qbulk = qbulk[3];
    heres[4]->BSIM3qbulk = qbulk[4];
    heres[5]->BSIM3qbulk = qbulk[5];
    heres[6]->BSIM3qbulk = qbulk[6];
    heres[7]->BSIM3qbulk = qbulk[7];
  }
  {
    heres[0]->BSIM3qdrn = qdrn[0];
    heres[1]->BSIM3qdrn = qdrn[1];
    heres[2]->BSIM3qdrn = qdrn[2];
    heres[3]->BSIM3qdrn = qdrn[3];
    heres[4]->BSIM3qdrn = qdrn[4];
    heres[5]->BSIM3qdrn = qdrn[5];
    heres[6]->BSIM3qdrn = qdrn[6];
    heres[7]->BSIM3qdrn = qdrn[7];
  }
  {
    heres[0]->BSIM3cd = cdrain[0];
    heres[1]->BSIM3cd = cdrain[1];
    heres[2]->BSIM3cd = cdrain[2];
    heres[3]->BSIM3cd = cdrain[3];
    heres[4]->BSIM3cd = cdrain[4];
    heres[5]->BSIM3cd = cdrain[5];
    heres[6]->BSIM3cd = cdrain[6];
    heres[7]->BSIM3cd = cdrain[7];
  }
  if (ChargeComputationNeeded)
  {
    Vec8d nstate_qbs = vec8_StateAccess(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3qbs, heres[1]->BSIM3qbs, heres[2]->BSIM3qbs, heres[3]->BSIM3qbs, heres[4]->BSIM3qbs, heres[5]->BSIM3qbs, heres[6]->BSIM3qbs, heres[7]->BSIM3qbs});
    Vec8d nstate_qbd = vec8_StateAccess(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3qbd, heres[1]->BSIM3qbd, heres[2]->BSIM3qbd, heres[3]->BSIM3qbd, heres[4]->BSIM3qbd, heres[5]->BSIM3qbd, heres[6]->BSIM3qbd, heres[7]->BSIM3qbd});
    if (model->BSIM3acmMod == 0)
    {
      czbd = model->BSIM3unitAreaTempJctCap * ((Vec8d ){heres[0]->BSIM3drainArea, heres[1]->BSIM3drainArea, heres[2]->BSIM3drainArea, heres[3]->BSIM3drainArea, heres[4]->BSIM3drainArea, heres[5]->BSIM3drainArea, heres[6]->BSIM3drainArea, heres[7]->BSIM3drainArea});
      czbs = model->BSIM3unitAreaTempJctCap * ((Vec8d ){heres[0]->BSIM3sourceArea, heres[1]->BSIM3sourceArea, heres[2]->BSIM3sourceArea, heres[3]->BSIM3sourceArea, heres[4]->BSIM3sourceArea, heres[5]->BSIM3sourceArea, heres[6]->BSIM3sourceArea, heres[7]->BSIM3sourceArea});
      if (1)
      {
        Vec8m condmask0 = ((Vec8d ){heres[0]->BSIM3drainPerimeter, heres[1]->BSIM3drainPerimeter, heres[2]->BSIM3drainPerimeter, heres[3]->BSIM3drainPerimeter, heres[4]->BSIM3drainPerimeter, heres[5]->BSIM3drainPerimeter, heres[6]->BSIM3drainPerimeter, heres[7]->BSIM3drainPerimeter}) < pParam->BSIM3weff;
        Vec8m condmask_true0 = condmask0;
        Vec8m condmask_false0 = ~condmask0;
        {
          czbdswg = vec8_blend(czbdswg, model->BSIM3unitLengthGateSidewallTempJctCap * ((Vec8d ){heres[0]->BSIM3drainPerimeter, heres[1]->BSIM3drainPerimeter, heres[2]->BSIM3drainPerimeter, heres[3]->BSIM3drainPerimeter, heres[4]->BSIM3drainPerimeter, heres[5]->BSIM3drainPerimeter, heres[6]->BSIM3drainPerimeter, heres[7]->BSIM3drainPerimeter}), condmask_true0);
          czbdsw = vec8_blend(czbdsw, vec8_SIMDTOVECTOR(0.0), condmask_true0);
        }
        {
          czbdsw = vec8_blend(czbdsw, model->BSIM3unitLengthSidewallTempJctCap * (((Vec8d ){heres[0]->BSIM3drainPerimeter, heres[1]->BSIM3drainPerimeter, heres[2]->BSIM3drainPerimeter, heres[3]->BSIM3drainPerimeter, heres[4]->BSIM3drainPerimeter, heres[5]->BSIM3drainPerimeter, heres[6]->BSIM3drainPerimeter, heres[7]->BSIM3drainPerimeter}) - pParam->BSIM3weff), condmask_false0);
          czbdswg = vec8_blend(czbdswg, vec8_SIMDTOVECTOR(model->BSIM3unitLengthGateSidewallTempJctCap * pParam->BSIM3weff), condmask_false0);
        }
      }

      if (1)
      {
        Vec8m condmask0 = ((Vec8d ){heres[0]->BSIM3sourcePerimeter, heres[1]->BSIM3sourcePerimeter, heres[2]->BSIM3sourcePerimeter, heres[3]->BSIM3sourcePerimeter, heres[4]->BSIM3sourcePerimeter, heres[5]->BSIM3sourcePerimeter, heres[6]->BSIM3sourcePerimeter, heres[7]->BSIM3sourcePerimeter}) < pParam->BSIM3weff;
        Vec8m condmask_true0 = condmask0;
        Vec8m condmask_false0 = ~condmask0;
        {
          czbssw = vec8_blend(czbssw, vec8_SIMDTOVECTOR(0.0), condmask_true0);
          czbsswg = vec8_blend(czbsswg, model->BSIM3unitLengthGateSidewallTempJctCap * ((Vec8d ){heres[0]->BSIM3sourcePerimeter, heres[1]->BSIM3sourcePerimeter, heres[2]->BSIM3sourcePerimeter, heres[3]->BSIM3sourcePerimeter, heres[4]->BSIM3sourcePerimeter, heres[5]->BSIM3sourcePerimeter, heres[6]->BSIM3sourcePerimeter, heres[7]->BSIM3sourcePerimeter}), condmask_true0);
        }
        {
          czbssw = vec8_blend(czbssw, model->BSIM3unitLengthSidewallTempJctCap * (((Vec8d ){heres[0]->BSIM3sourcePerimeter, heres[1]->BSIM3sourcePerimeter, heres[2]->BSIM3sourcePerimeter, heres[3]->BSIM3sourcePerimeter, heres[4]->BSIM3sourcePerimeter, heres[5]->BSIM3sourcePerimeter, heres[6]->BSIM3sourcePerimeter, heres[7]->BSIM3sourcePerimeter}) - pParam->BSIM3weff), condmask_false0);
          czbsswg = vec8_blend(czbsswg, vec8_SIMDTOVECTOR(model->BSIM3unitLengthGateSidewallTempJctCap * pParam->BSIM3weff), condmask_false0);
        }
      }

    }
    else
    {
      error = vec8_BSIM3_ACM_junctionCapacitances(model, heres, &czbd, &czbdsw, &czbdswg, &czbs, &czbssw, &czbsswg);
      if (SIMDANY(error))
        return error;

    }

    MJ = model->BSIM3bulkJctBotGradingCoeff;
    MJSW = model->BSIM3bulkJctSideGradingCoeff;
    MJSWG = model->BSIM3bulkJctGateSideGradingCoeff;
    if (1)
    {
      Vec8m condmask0 = vbs == 0.0;
      Vec8m condmask_true0 = condmask0;
      Vec8m condmask_false0 = ~condmask0;
      {
        nstate_qbs = vec8_blend(nstate_qbs, vec8_SIMDTOVECTOR(0.0), condmask_true0);
        {
          Vec8d val = (czbs + czbssw) + czbsswg;
          if (condmask_true0[0])
            heres[0]->BSIM3capbs = val[0];

          if (condmask_true0[1])
            heres[1]->BSIM3capbs = val[1];

          if (condmask_true0[2])
            heres[2]->BSIM3capbs = val[2];

          if (condmask_true0[3])
            heres[3]->BSIM3capbs = val[3];

          if (condmask_true0[4])
            heres[4]->BSIM3capbs = val[4];

          if (condmask_true0[5])
            heres[5]->BSIM3capbs = val[5];

          if (condmask_true0[6])
            heres[6]->BSIM3capbs = val[6];

          if (condmask_true0[7])
            heres[7]->BSIM3capbs = val[7];

        }
      }
      if (1)
      {
        Vec8m condmask1 = vbs < 0.0;
        Vec8m condmask_true1 = condmask_false0 & condmask1;
        Vec8m condmask_false1 = condmask_false0 & (~condmask1);
        {
          if (1)
          {
            Vec8m condmask2 = czbs > 0.0;
            Vec8m condmask_true2 = condmask_true1 & condmask2;
            Vec8m condmask_false2 = condmask_true1 & (~condmask2);
            {
              arg = vec8_blend(arg, 1.0 - (vbs / model->BSIM3PhiB), condmask_true2);
              if (MJ == 0.5)
                sarg = vec8_blend(sarg, 1.0 / vec8_sqrt(arg), condmask_true2);
              else
                sarg = vec8_blend(sarg, vec8_exp((-MJ) * vec8_log(arg)), condmask_true2);

              nstate_qbs = vec8_blend(nstate_qbs, ((model->BSIM3PhiB * czbs) * (1.0 - (arg * sarg))) / (1.0 - MJ), condmask_true2);
              {
                Vec8d val = czbs * sarg;
                if (condmask_true2[0])
                  heres[0]->BSIM3capbs = val[0];

                if (condmask_true2[1])
                  heres[1]->BSIM3capbs = val[1];

                if (condmask_true2[2])
                  heres[2]->BSIM3capbs = val[2];

                if (condmask_true2[3])
                  heres[3]->BSIM3capbs = val[3];

                if (condmask_true2[4])
                  heres[4]->BSIM3capbs = val[4];

                if (condmask_true2[5])
                  heres[5]->BSIM3capbs = val[5];

                if (condmask_true2[6])
                  heres[6]->BSIM3capbs = val[6];

                if (condmask_true2[7])
                  heres[7]->BSIM3capbs = val[7];

              }
            }
            {
              nstate_qbs = vec8_blend(nstate_qbs, vec8_SIMDTOVECTOR(0.0), condmask_false2);
              {
                if (condmask_false2[0])
                  heres[0]->BSIM3capbs = 0.0;

                if (condmask_false2[1])
                  heres[1]->BSIM3capbs = 0.0;

                if (condmask_false2[2])
                  heres[2]->BSIM3capbs = 0.0;

                if (condmask_false2[3])
                  heres[3]->BSIM3capbs = 0.0;

                if (condmask_false2[4])
                  heres[4]->BSIM3capbs = 0.0;

                if (condmask_false2[5])
                  heres[5]->BSIM3capbs = 0.0;

                if (condmask_false2[6])
                  heres[6]->BSIM3capbs = 0.0;

                if (condmask_false2[7])
                  heres[7]->BSIM3capbs = 0.0;

              }
            }
          }

          if (1)
          {
            Vec8m condmask2 = czbssw > 0.0;
            Vec8m condmask_true2 = condmask_true1 & condmask2;
            {
              arg = vec8_blend(arg, 1.0 - (vbs / model->BSIM3PhiBSW), condmask_true2);
              if (MJSW == 0.5)
                sarg = vec8_blend(sarg, 1.0 / vec8_sqrt(arg), condmask_true2);
              else
                sarg = vec8_blend(sarg, vec8_exp((-MJSW) * vec8_log(arg)), condmask_true2);

              nstate_qbs = vec8_blend(nstate_qbs, nstate_qbs + (((model->BSIM3PhiBSW * czbssw) * (1.0 - (arg * sarg))) / (1.0 - MJSW)), condmask_true2);
              {
                Vec8d val = czbssw * sarg;
                if (condmask_true2[0])
                  heres[0]->BSIM3capbs += val[0];

                if (condmask_true2[1])
                  heres[1]->BSIM3capbs += val[1];

                if (condmask_true2[2])
                  heres[2]->BSIM3capbs += val[2];

                if (condmask_true2[3])
                  heres[3]->BSIM3capbs += val[3];

                if (condmask_true2[4])
                  heres[4]->BSIM3capbs += val[4];

                if (condmask_true2[5])
                  heres[5]->BSIM3capbs += val[5];

                if (condmask_true2[6])
                  heres[6]->BSIM3capbs += val[6];

                if (condmask_true2[7])
                  heres[7]->BSIM3capbs += val[7];

              }
            }
          }

          if (1)
          {
            Vec8m condmask2 = czbsswg > 0.0;
            Vec8m condmask_true2 = condmask_true1 & condmask2;
            {
              arg = vec8_blend(arg, 1.0 - (vbs / model->BSIM3PhiBSWG), condmask_true2);
              if (MJSWG == 0.5)
                sarg = vec8_blend(sarg, 1.0 / vec8_sqrt(arg), condmask_true2);
              else
                sarg = vec8_blend(sarg, vec8_exp((-MJSWG) * vec8_log(arg)), condmask_true2);

              nstate_qbs = vec8_blend(nstate_qbs, nstate_qbs + (((model->BSIM3PhiBSWG * czbsswg) * (1.0 - (arg * sarg))) / (1.0 - MJSWG)), condmask_true2);
              {
                Vec8d val = czbsswg * sarg;
                if (condmask_true2[0])
                  heres[0]->BSIM3capbs += val[0];

                if (condmask_true2[1])
                  heres[1]->BSIM3capbs += val[1];

                if (condmask_true2[2])
                  heres[2]->BSIM3capbs += val[2];

                if (condmask_true2[3])
                  heres[3]->BSIM3capbs += val[3];

                if (condmask_true2[4])
                  heres[4]->BSIM3capbs += val[4];

                if (condmask_true2[5])
                  heres[5]->BSIM3capbs += val[5];

                if (condmask_true2[6])
                  heres[6]->BSIM3capbs += val[6];

                if (condmask_true2[7])
                  heres[7]->BSIM3capbs += val[7];

              }
            }
          }

        }
        {
          T0 = vec8_blend(T0, (czbs + czbssw) + czbsswg, condmask_false1);
          T1 = vec8_blend(T1, vbs * ((((czbs * MJ) / model->BSIM3PhiB) + ((czbssw * MJSW) / model->BSIM3PhiBSW)) + ((czbsswg * MJSWG) / model->BSIM3PhiBSWG)), condmask_false1);
          nstate_qbs = vec8_blend(nstate_qbs, vbs * (T0 + (0.5 * T1)), condmask_false1);
          {
            Vec8d val = T0 + T1;
            if (condmask_false1[0])
              heres[0]->BSIM3capbs = val[0];

            if (condmask_false1[1])
              heres[1]->BSIM3capbs = val[1];

            if (condmask_false1[2])
              heres[2]->BSIM3capbs = val[2];

            if (condmask_false1[3])
              heres[3]->BSIM3capbs = val[3];

            if (condmask_false1[4])
              heres[4]->BSIM3capbs = val[4];

            if (condmask_false1[5])
              heres[5]->BSIM3capbs = val[5];

            if (condmask_false1[6])
              heres[6]->BSIM3capbs = val[6];

            if (condmask_false1[7])
              heres[7]->BSIM3capbs = val[7];

          }
        }
      }

    }

    vec8_StateStore(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3qbs, heres[1]->BSIM3qbs, heres[2]->BSIM3qbs, heres[3]->BSIM3qbs, heres[4]->BSIM3qbs, heres[5]->BSIM3qbs, heres[6]->BSIM3qbs, heres[7]->BSIM3qbs}, nstate_qbs);
    if (1)
    {
      Vec8m condmask0 = vbd == 0.0;
      Vec8m condmask_true0 = condmask0;
      Vec8m condmask_false0 = ~condmask0;
      {
        nstate_qbd = vec8_blend(nstate_qbd, vec8_SIMDTOVECTOR(0.0), condmask_true0);
        {
          Vec8d val = (czbd + czbdsw) + czbdswg;
          if (condmask_true0[0])
            heres[0]->BSIM3capbd = val[0];

          if (condmask_true0[1])
            heres[1]->BSIM3capbd = val[1];

          if (condmask_true0[2])
            heres[2]->BSIM3capbd = val[2];

          if (condmask_true0[3])
            heres[3]->BSIM3capbd = val[3];

          if (condmask_true0[4])
            heres[4]->BSIM3capbd = val[4];

          if (condmask_true0[5])
            heres[5]->BSIM3capbd = val[5];

          if (condmask_true0[6])
            heres[6]->BSIM3capbd = val[6];

          if (condmask_true0[7])
            heres[7]->BSIM3capbd = val[7];

        }
      }
      if (1)
      {
        Vec8m condmask1 = vbd < 0.0;
        Vec8m condmask_true1 = condmask_false0 & condmask1;
        Vec8m condmask_false1 = condmask_false0 & (~condmask1);
        {
          if (1)
          {
            Vec8m condmask2 = czbd > 0.0;
            Vec8m condmask_true2 = condmask_true1 & condmask2;
            Vec8m condmask_false2 = condmask_true1 & (~condmask2);
            {
              arg = vec8_blend(arg, 1.0 - (vbd / model->BSIM3PhiB), condmask_true2);
              if (MJ == 0.5)
                sarg = vec8_blend(sarg, 1.0 / vec8_sqrt(arg), condmask_true2);
              else
                sarg = vec8_blend(sarg, vec8_exp((-MJ) * vec8_log(arg)), condmask_true2);

              nstate_qbd = vec8_blend(nstate_qbd, ((model->BSIM3PhiB * czbd) * (1.0 - (arg * sarg))) / (1.0 - MJ), condmask_true2);
              {
                Vec8d val = czbd * sarg;
                if (condmask_true2[0])
                  heres[0]->BSIM3capbd = val[0];

                if (condmask_true2[1])
                  heres[1]->BSIM3capbd = val[1];

                if (condmask_true2[2])
                  heres[2]->BSIM3capbd = val[2];

                if (condmask_true2[3])
                  heres[3]->BSIM3capbd = val[3];

                if (condmask_true2[4])
                  heres[4]->BSIM3capbd = val[4];

                if (condmask_true2[5])
                  heres[5]->BSIM3capbd = val[5];

                if (condmask_true2[6])
                  heres[6]->BSIM3capbd = val[6];

                if (condmask_true2[7])
                  heres[7]->BSIM3capbd = val[7];

              }
            }
            {
              nstate_qbd = vec8_blend(nstate_qbd, vec8_SIMDTOVECTOR(0.0), condmask_false2);
              {
                if (condmask_false2[0])
                  heres[0]->BSIM3capbd = 0.0;

                if (condmask_false2[1])
                  heres[1]->BSIM3capbd = 0.0;

                if (condmask_false2[2])
                  heres[2]->BSIM3capbd = 0.0;

                if (condmask_false2[3])
                  heres[3]->BSIM3capbd = 0.0;

                if (condmask_false2[4])
                  heres[4]->BSIM3capbd = 0.0;

                if (condmask_false2[5])
                  heres[5]->BSIM3capbd = 0.0;

                if (condmask_false2[6])
                  heres[6]->BSIM3capbd = 0.0;

                if (condmask_false2[7])
                  heres[7]->BSIM3capbd = 0.0;

              }
            }
          }

          if (1)
          {
            Vec8m condmask2 = czbdsw > 0.0;
            Vec8m condmask_true2 = condmask_true1 & condmask2;
            {
              arg = vec8_blend(arg, 1.0 - (vbd / model->BSIM3PhiBSW), condmask_true2);
              if (MJSW == 0.5)
                sarg = vec8_blend(sarg, 1.0 / vec8_sqrt(arg), condmask_true2);
              else
                sarg = vec8_blend(sarg, vec8_exp((-MJSW) * vec8_log(arg)), condmask_true2);

              nstate_qbd = vec8_blend(nstate_qbd, nstate_qbd + (((model->BSIM3PhiBSW * czbdsw) * (1.0 - (arg * sarg))) / (1.0 - MJSW)), condmask_true2);
              {
                Vec8d val = czbdsw * sarg;
                if (condmask_true2[0])
                  heres[0]->BSIM3capbd += val[0];

                if (condmask_true2[1])
                  heres[1]->BSIM3capbd += val[1];

                if (condmask_true2[2])
                  heres[2]->BSIM3capbd += val[2];

                if (condmask_true2[3])
                  heres[3]->BSIM3capbd += val[3];

                if (condmask_true2[4])
                  heres[4]->BSIM3capbd += val[4];

                if (condmask_true2[5])
                  heres[5]->BSIM3capbd += val[5];

                if (condmask_true2[6])
                  heres[6]->BSIM3capbd += val[6];

                if (condmask_true2[7])
                  heres[7]->BSIM3capbd += val[7];

              }
            }
          }

          if (1)
          {
            Vec8m condmask2 = czbdswg > 0.0;
            Vec8m condmask_true2 = condmask_true1 & condmask2;
            {
              arg = vec8_blend(arg, 1.0 - (vbd / model->BSIM3PhiBSWG), condmask_true2);
              if (MJSWG == 0.5)
                sarg = vec8_blend(sarg, 1.0 / vec8_sqrt(arg), condmask_true2);
              else
                sarg = vec8_blend(sarg, vec8_exp((-MJSWG) * vec8_log(arg)), condmask_true2);

              nstate_qbd = vec8_blend(nstate_qbd, nstate_qbd + (((model->BSIM3PhiBSWG * czbdswg) * (1.0 - (arg * sarg))) / (1.0 - MJSWG)), condmask_true2);
              {
                Vec8d val = czbdswg * sarg;
                if (condmask_true2[0])
                  heres[0]->BSIM3capbd += val[0];

                if (condmask_true2[1])
                  heres[1]->BSIM3capbd += val[1];

                if (condmask_true2[2])
                  heres[2]->BSIM3capbd += val[2];

                if (condmask_true2[3])
                  heres[3]->BSIM3capbd += val[3];

                if (condmask_true2[4])
                  heres[4]->BSIM3capbd += val[4];

                if (condmask_true2[5])
                  heres[5]->BSIM3capbd += val[5];

                if (condmask_true2[6])
                  heres[6]->BSIM3capbd += val[6];

                if (condmask_true2[7])
                  heres[7]->BSIM3capbd += val[7];

              }
            }
          }

        }
        {
          T0 = vec8_blend(T0, (czbd + czbdsw) + czbdswg, condmask_false1);
          T1 = vec8_blend(T1, vbd * ((((czbd * MJ) / model->BSIM3PhiB) + ((czbdsw * MJSW) / model->BSIM3PhiBSW)) + ((czbdswg * MJSWG) / model->BSIM3PhiBSWG)), condmask_false1);
          nstate_qbd = vec8_blend(nstate_qbd, vbd * (T0 + (0.5 * T1)), condmask_false1);
          {
            Vec8d val = T0 + T1;
            if (condmask_false1[0])
              heres[0]->BSIM3capbd = val[0];

            if (condmask_false1[1])
              heres[1]->BSIM3capbd = val[1];

            if (condmask_false1[2])
              heres[2]->BSIM3capbd = val[2];

            if (condmask_false1[3])
              heres[3]->BSIM3capbd = val[3];

            if (condmask_false1[4])
              heres[4]->BSIM3capbd = val[4];

            if (condmask_false1[5])
              heres[5]->BSIM3capbd = val[5];

            if (condmask_false1[6])
              heres[6]->BSIM3capbd = val[6];

            if (condmask_false1[7])
              heres[7]->BSIM3capbd = val[7];

          }
        }
      }

    }

    vec8_StateStore(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3qbd, heres[1]->BSIM3qbd, heres[2]->BSIM3qbd, heres[3]->BSIM3qbd, heres[4]->BSIM3qbd, heres[5]->BSIM3qbd, heres[6]->BSIM3qbd, heres[7]->BSIM3qbd}, nstate_qbd);
  }

  if ((heres[0]->BSIM3off == 0) || (!(ckt->CKTmode & MODEINITFIX)))
  {
    Vec8m nonconcount;
    nonconcount = Check;
    nonconcount = nonconcount & 1;
    {
      heres[0]->BSIM3noncon = nonconcount[0];
      heres[1]->BSIM3noncon = nonconcount[1];
      heres[2]->BSIM3noncon = nonconcount[2];
      heres[3]->BSIM3noncon = nonconcount[3];
      heres[4]->BSIM3noncon = nonconcount[4];
      heres[5]->BSIM3noncon = nonconcount[5];
      heres[6]->BSIM3noncon = nonconcount[6];
      heres[7]->BSIM3noncon = nonconcount[7];
    }
  }
  else
  {
    heres[0]->BSIM3noncon = 0;
    heres[1]->BSIM3noncon = 0;
    heres[2]->BSIM3noncon = 0;
    heres[3]->BSIM3noncon = 0;
    heres[4]->BSIM3noncon = 0;
    heres[5]->BSIM3noncon = 0;
    heres[6]->BSIM3noncon = 0;
    heres[7]->BSIM3noncon = 0;
  }

  vec8_StateStore(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3vbs, heres[1]->BSIM3vbs, heres[2]->BSIM3vbs, heres[3]->BSIM3vbs, heres[4]->BSIM3vbs, heres[5]->BSIM3vbs, heres[6]->BSIM3vbs, heres[7]->BSIM3vbs}, vbs);
  vec8_StateStore(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3vbd, heres[1]->BSIM3vbd, heres[2]->BSIM3vbd, heres[3]->BSIM3vbd, heres[4]->BSIM3vbd, heres[5]->BSIM3vbd, heres[6]->BSIM3vbd, heres[7]->BSIM3vbd}, vbd);
  vec8_StateStore(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3vgs, heres[1]->BSIM3vgs, heres[2]->BSIM3vgs, heres[3]->BSIM3vgs, heres[4]->BSIM3vgs, heres[5]->BSIM3vgs, heres[6]->BSIM3vgs, heres[7]->BSIM3vgs}, vgs);
  vec8_StateStore(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3vds, heres[1]->BSIM3vds, heres[2]->BSIM3vds, heres[3]->BSIM3vds, heres[4]->BSIM3vds, heres[5]->BSIM3vds, heres[6]->BSIM3vds, heres[7]->BSIM3vds}, vds);
  vec8_StateStore(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3qdef, heres[1]->BSIM3qdef, heres[2]->BSIM3qdef, heres[3]->BSIM3qdef, heres[4]->BSIM3qdef, heres[5]->BSIM3qdef, heres[6]->BSIM3qdef, heres[7]->BSIM3qdef}, qdef);
  if (!ChargeComputationNeeded)
    goto line850;

  line755:
  if (heres[0]->BSIM3nqsMod || heres[0]->BSIM3acnqsMod)
  {
    qcheq = -(qbulk + qgate);
    {
      Vec8d val = -(((Vec8d ){heres[0]->BSIM3cggb, heres[1]->BSIM3cggb, heres[2]->BSIM3cggb, heres[3]->BSIM3cggb, heres[4]->BSIM3cggb, heres[5]->BSIM3cggb, heres[6]->BSIM3cggb, heres[7]->BSIM3cggb}) + ((Vec8d ){heres[0]->BSIM3cbgb, heres[1]->BSIM3cbgb, heres[2]->BSIM3cbgb, heres[3]->BSIM3cbgb, heres[4]->BSIM3cbgb, heres[5]->BSIM3cbgb, heres[6]->BSIM3cbgb, heres[7]->BSIM3cbgb}));
      heres[0]->BSIM3cqgb = val[0];
      heres[1]->BSIM3cqgb = val[1];
      heres[2]->BSIM3cqgb = val[2];
      heres[3]->BSIM3cqgb = val[3];
      heres[4]->BSIM3cqgb = val[4];
      heres[5]->BSIM3cqgb = val[5];
      heres[6]->BSIM3cqgb = val[6];
      heres[7]->BSIM3cqgb = val[7];
    }
    {
      Vec8d val = -(((Vec8d ){heres[0]->BSIM3cgdb, heres[1]->BSIM3cgdb, heres[2]->BSIM3cgdb, heres[3]->BSIM3cgdb, heres[4]->BSIM3cgdb, heres[5]->BSIM3cgdb, heres[6]->BSIM3cgdb, heres[7]->BSIM3cgdb}) + ((Vec8d ){heres[0]->BSIM3cbdb, heres[1]->BSIM3cbdb, heres[2]->BSIM3cbdb, heres[3]->BSIM3cbdb, heres[4]->BSIM3cbdb, heres[5]->BSIM3cbdb, heres[6]->BSIM3cbdb, heres[7]->BSIM3cbdb}));
      heres[0]->BSIM3cqdb = val[0];
      heres[1]->BSIM3cqdb = val[1];
      heres[2]->BSIM3cqdb = val[2];
      heres[3]->BSIM3cqdb = val[3];
      heres[4]->BSIM3cqdb = val[4];
      heres[5]->BSIM3cqdb = val[5];
      heres[6]->BSIM3cqdb = val[6];
      heres[7]->BSIM3cqdb = val[7];
    }
    {
      Vec8d val = -(((Vec8d ){heres[0]->BSIM3cgsb, heres[1]->BSIM3cgsb, heres[2]->BSIM3cgsb, heres[3]->BSIM3cgsb, heres[4]->BSIM3cgsb, heres[5]->BSIM3cgsb, heres[6]->BSIM3cgsb, heres[7]->BSIM3cgsb}) + ((Vec8d ){heres[0]->BSIM3cbsb, heres[1]->BSIM3cbsb, heres[2]->BSIM3cbsb, heres[3]->BSIM3cbsb, heres[4]->BSIM3cbsb, heres[5]->BSIM3cbsb, heres[6]->BSIM3cbsb, heres[7]->BSIM3cbsb}));
      heres[0]->BSIM3cqsb = val[0];
      heres[1]->BSIM3cqsb = val[1];
      heres[2]->BSIM3cqsb = val[2];
      heres[3]->BSIM3cqsb = val[3];
      heres[4]->BSIM3cqsb = val[4];
      heres[5]->BSIM3cqsb = val[5];
      heres[6]->BSIM3cqsb = val[6];
      heres[7]->BSIM3cqsb = val[7];
    }
    {
      Vec8d val = -((((Vec8d ){heres[0]->BSIM3cqgb, heres[1]->BSIM3cqgb, heres[2]->BSIM3cqgb, heres[3]->BSIM3cqgb, heres[4]->BSIM3cqgb, heres[5]->BSIM3cqgb, heres[6]->BSIM3cqgb, heres[7]->BSIM3cqgb}) + ((Vec8d ){heres[0]->BSIM3cqdb, heres[1]->BSIM3cqdb, heres[2]->BSIM3cqdb, heres[3]->BSIM3cqdb, heres[4]->BSIM3cqdb, heres[5]->BSIM3cqdb, heres[6]->BSIM3cqdb, heres[7]->BSIM3cqdb})) + ((Vec8d ){heres[0]->BSIM3cqsb, heres[1]->BSIM3cqsb, heres[2]->BSIM3cqsb, heres[3]->BSIM3cqsb, heres[4]->BSIM3cqsb, heres[5]->BSIM3cqsb, heres[6]->BSIM3cqsb, heres[7]->BSIM3cqsb}));
      heres[0]->BSIM3cqbb = val[0];
      heres[1]->BSIM3cqbb = val[1];
      heres[2]->BSIM3cqbb = val[2];
      heres[3]->BSIM3cqbb = val[3];
      heres[4]->BSIM3cqbb = val[4];
      heres[5]->BSIM3cqbb = val[5];
      heres[6]->BSIM3cqbb = val[6];
      heres[7]->BSIM3cqbb = val[7];
    }
    gtau_drift = vec8_fabs(((Vec8d ){heres[0]->BSIM3tconst, heres[1]->BSIM3tconst, heres[2]->BSIM3tconst, heres[3]->BSIM3tconst, heres[4]->BSIM3tconst, heres[5]->BSIM3tconst, heres[6]->BSIM3tconst, heres[7]->BSIM3tconst}) * qcheq) * ScalingFactor;
    T0 = vec8_SIMDTOVECTOR(pParam->BSIM3leffCV * pParam->BSIM3leffCV);
    gtau_diff = (((16.0 * ((Vec8d ){heres[0]->BSIM3u0temp, heres[1]->BSIM3u0temp, heres[2]->BSIM3u0temp, heres[3]->BSIM3u0temp, heres[4]->BSIM3u0temp, heres[5]->BSIM3u0temp, heres[6]->BSIM3u0temp, heres[7]->BSIM3u0temp})) * model->BSIM3vtm) / T0) * ScalingFactor;
    {
      Vec8d val = gtau_drift + gtau_diff;
      heres[0]->BSIM3gtau = val[0];
      heres[1]->BSIM3gtau = val[1];
      heres[2]->BSIM3gtau = val[2];
      heres[3]->BSIM3gtau = val[3];
      heres[4]->BSIM3gtau = val[4];
      heres[5]->BSIM3gtau = val[5];
      heres[6]->BSIM3gtau = val[6];
      heres[7]->BSIM3gtau = val[7];
    }
    if (heres[0]->BSIM3acnqsMod)
    {
      Vec8d val = ScalingFactor / ((Vec8d ){heres[0]->BSIM3gtau, heres[1]->BSIM3gtau, heres[2]->BSIM3gtau, heres[3]->BSIM3gtau, heres[4]->BSIM3gtau, heres[5]->BSIM3gtau, heres[6]->BSIM3gtau, heres[7]->BSIM3gtau});
      heres[0]->BSIM3taunet = val[0];
      heres[1]->BSIM3taunet = val[1];
      heres[2]->BSIM3taunet = val[2];
      heres[3]->BSIM3taunet = val[3];
      heres[4]->BSIM3taunet = val[4];
      heres[5]->BSIM3taunet = val[5];
      heres[6]->BSIM3taunet = val[6];
      heres[7]->BSIM3taunet = val[7];
    }

  }


  if (model->BSIM3capMod == 0)
  {
    cgdo = vec8_SIMDTOVECTOR(pParam->BSIM3cgdo);
    qgdo = pParam->BSIM3cgdo * vgd;
    cgso = vec8_SIMDTOVECTOR(pParam->BSIM3cgso);
    qgso = pParam->BSIM3cgso * vgs;
  }
  else
    if (model->BSIM3capMod == 1)
  {
    if (1)
    {
      Vec8m condmask0 = vgd < 0.0;
      Vec8m condmask_true0 = condmask0;
      Vec8m condmask_false0 = ~condmask0;
      {
        T1 = vec8_blend(T1, vec8_sqrt(1.0 - ((4.0 * vgd) / pParam->BSIM3ckappa)), condmask_true0);
        cgdo = vec8_blend(cgdo, pParam->BSIM3cgdo + ((pParam->BSIM3weffCV * pParam->BSIM3cgdl) / T1), condmask_true0);
        qgdo = vec8_blend(qgdo, (pParam->BSIM3cgdo * vgd) - ((((pParam->BSIM3weffCV * 0.5) * pParam->BSIM3cgdl) * pParam->BSIM3ckappa) * (T1 - 1.0)), condmask_true0);
      }
      {
        cgdo = vec8_blend(cgdo, vec8_SIMDTOVECTOR(pParam->BSIM3cgdo + (pParam->BSIM3weffCV * pParam->BSIM3cgdl)), condmask_false0);
        qgdo = vec8_blend(qgdo, ((pParam->BSIM3weffCV * pParam->BSIM3cgdl) + pParam->BSIM3cgdo) * vgd, condmask_false0);
      }
    }

    if (1)
    {
      Vec8m condmask0 = vgs < 0.0;
      Vec8m condmask_true0 = condmask0;
      Vec8m condmask_false0 = ~condmask0;
      {
        T1 = vec8_blend(T1, vec8_sqrt(1.0 - ((4.0 * vgs) / pParam->BSIM3ckappa)), condmask_true0);
        cgso = vec8_blend(cgso, pParam->BSIM3cgso + ((pParam->BSIM3weffCV * pParam->BSIM3cgsl) / T1), condmask_true0);
        qgso = vec8_blend(qgso, (pParam->BSIM3cgso * vgs) - ((((pParam->BSIM3weffCV * 0.5) * pParam->BSIM3cgsl) * pParam->BSIM3ckappa) * (T1 - 1.0)), condmask_true0);
      }
      {
        cgso = vec8_blend(cgso, vec8_SIMDTOVECTOR(pParam->BSIM3cgso + (pParam->BSIM3weffCV * pParam->BSIM3cgsl)), condmask_false0);
        qgso = vec8_blend(qgso, ((pParam->BSIM3weffCV * pParam->BSIM3cgsl) + pParam->BSIM3cgso) * vgs, condmask_false0);
      }
    }

  }
  else
  {
    T0 = vgd + DELTA_1;
    T1 = vec8_sqrt((T0 * T0) + (4.0 * DELTA_1));
    T2 = 0.5 * (T0 - T1);
    T3 = vec8_SIMDTOVECTOR(pParam->BSIM3weffCV * pParam->BSIM3cgdl);
    T4 = vec8_sqrt(1.0 - ((4.0 * T2) / pParam->BSIM3ckappa));
    cgdo = (pParam->BSIM3cgdo + T3) - ((T3 * (1.0 - (1.0 / T4))) * (0.5 - ((0.5 * T0) / T1)));
    qgdo = ((pParam->BSIM3cgdo + T3) * vgd) - (T3 * (T2 + ((0.5 * pParam->BSIM3ckappa) * (T4 - 1.0))));
    T0 = vgs + DELTA_1;
    T1 = vec8_sqrt((T0 * T0) + (4.0 * DELTA_1));
    T2 = 0.5 * (T0 - T1);
    T3 = vec8_SIMDTOVECTOR(pParam->BSIM3weffCV * pParam->BSIM3cgsl);
    T4 = vec8_sqrt(1.0 - ((4.0 * T2) / pParam->BSIM3ckappa));
    cgso = (pParam->BSIM3cgso + T3) - ((T3 * (1.0 - (1.0 / T4))) * (0.5 - ((0.5 * T0) / T1)));
    qgso = ((pParam->BSIM3cgso + T3) * vgs) - (T3 * (T2 + ((0.5 * pParam->BSIM3ckappa) * (T4 - 1.0))));
  }


  {
    heres[0]->BSIM3cgdo = cgdo[0];
    heres[1]->BSIM3cgdo = cgdo[1];
    heres[2]->BSIM3cgdo = cgdo[2];
    heres[3]->BSIM3cgdo = cgdo[3];
    heres[4]->BSIM3cgdo = cgdo[4];
    heres[5]->BSIM3cgdo = cgdo[5];
    heres[6]->BSIM3cgdo = cgdo[6];
    heres[7]->BSIM3cgdo = cgdo[7];
  }
  {
    heres[0]->BSIM3cgso = cgso[0];
    heres[1]->BSIM3cgso = cgso[1];
    heres[2]->BSIM3cgso = cgso[2];
    heres[3]->BSIM3cgso = cgso[3];
    heres[4]->BSIM3cgso = cgso[4];
    heres[5]->BSIM3cgso = cgso[5];
    heres[6]->BSIM3cgso = cgso[6];
    heres[7]->BSIM3cgso = cgso[7];
  }
  ag0 = ckt->CKTag[0];
  ddxpart_dVd = (ddxpart_dVg = (ddxpart_dVb = (ddxpart_dVs = vec8_SIMDTOVECTOR(0.0))));
  dsxpart_dVd = (dsxpart_dVg = (dsxpart_dVb = (dsxpart_dVs = vec8_SIMDTOVECTOR(0.0))));
  ggtg = (ggtd = (ggtb = (ggts = vec8_SIMDTOVECTOR(0.0))));
  CoxWL = (model->BSIM3cox * pParam->BSIM3weffCV) * pParam->BSIM3leffCV;
  if (1)
  {
    Vec8m condmask0 = BSIM3mode;
    Vec8m condmask_true0 = condmask0;
    Vec8m condmask_false0 = ~condmask0;
    {
      if (heres[0]->BSIM3nqsMod == 0)
      {
        gcggb = vec8_blend(gcggb, (((((Vec8d ){heres[0]->BSIM3cggb, heres[1]->BSIM3cggb, heres[2]->BSIM3cggb, heres[3]->BSIM3cggb, heres[4]->BSIM3cggb, heres[5]->BSIM3cggb, heres[6]->BSIM3cggb, heres[7]->BSIM3cggb}) + cgdo) + cgso) + pParam->BSIM3cgbo) * ag0, condmask_true0);
        gcgdb = vec8_blend(gcgdb, (((Vec8d ){heres[0]->BSIM3cgdb, heres[1]->BSIM3cgdb, heres[2]->BSIM3cgdb, heres[3]->BSIM3cgdb, heres[4]->BSIM3cgdb, heres[5]->BSIM3cgdb, heres[6]->BSIM3cgdb, heres[7]->BSIM3cgdb}) - cgdo) * ag0, condmask_true0);
        gcgsb = vec8_blend(gcgsb, (((Vec8d ){heres[0]->BSIM3cgsb, heres[1]->BSIM3cgsb, heres[2]->BSIM3cgsb, heres[3]->BSIM3cgsb, heres[4]->BSIM3cgsb, heres[5]->BSIM3cgsb, heres[6]->BSIM3cgsb, heres[7]->BSIM3cgsb}) - cgso) * ag0, condmask_true0);
        gcdgb = vec8_blend(gcdgb, (((Vec8d ){heres[0]->BSIM3cdgb, heres[1]->BSIM3cdgb, heres[2]->BSIM3cdgb, heres[3]->BSIM3cdgb, heres[4]->BSIM3cdgb, heres[5]->BSIM3cdgb, heres[6]->BSIM3cdgb, heres[7]->BSIM3cdgb}) - cgdo) * ag0, condmask_true0);
        gcddb = vec8_blend(gcddb, ((((Vec8d ){heres[0]->BSIM3cddb, heres[1]->BSIM3cddb, heres[2]->BSIM3cddb, heres[3]->BSIM3cddb, heres[4]->BSIM3cddb, heres[5]->BSIM3cddb, heres[6]->BSIM3cddb, heres[7]->BSIM3cddb}) + ((Vec8d ){heres[0]->BSIM3capbd, heres[1]->BSIM3capbd, heres[2]->BSIM3capbd, heres[3]->BSIM3capbd, heres[4]->BSIM3capbd, heres[5]->BSIM3capbd, heres[6]->BSIM3capbd, heres[7]->BSIM3capbd})) + cgdo) * ag0, condmask_true0);
        gcdsb = vec8_blend(gcdsb, ((Vec8d ){heres[0]->BSIM3cdsb, heres[1]->BSIM3cdsb, heres[2]->BSIM3cdsb, heres[3]->BSIM3cdsb, heres[4]->BSIM3cdsb, heres[5]->BSIM3cdsb, heres[6]->BSIM3cdsb, heres[7]->BSIM3cdsb}) * ag0, condmask_true0);
        gcsgb = vec8_blend(gcsgb, (-(((((Vec8d ){heres[0]->BSIM3cggb, heres[1]->BSIM3cggb, heres[2]->BSIM3cggb, heres[3]->BSIM3cggb, heres[4]->BSIM3cggb, heres[5]->BSIM3cggb, heres[6]->BSIM3cggb, heres[7]->BSIM3cggb}) + ((Vec8d ){heres[0]->BSIM3cbgb, heres[1]->BSIM3cbgb, heres[2]->BSIM3cbgb, heres[3]->BSIM3cbgb, heres[4]->BSIM3cbgb, heres[5]->BSIM3cbgb, heres[6]->BSIM3cbgb, heres[7]->BSIM3cbgb})) + ((Vec8d ){heres[0]->BSIM3cdgb, heres[1]->BSIM3cdgb, heres[2]->BSIM3cdgb, heres[3]->BSIM3cdgb, heres[4]->BSIM3cdgb, heres[5]->BSIM3cdgb, heres[6]->BSIM3cdgb, heres[7]->BSIM3cdgb})) + cgso)) * ag0, condmask_true0);
        gcsdb = vec8_blend(gcsdb, (-((((Vec8d ){heres[0]->BSIM3cgdb, heres[1]->BSIM3cgdb, heres[2]->BSIM3cgdb, heres[3]->BSIM3cgdb, heres[4]->BSIM3cgdb, heres[5]->BSIM3cgdb, heres[6]->BSIM3cgdb, heres[7]->BSIM3cgdb}) + ((Vec8d ){heres[0]->BSIM3cbdb, heres[1]->BSIM3cbdb, heres[2]->BSIM3cbdb, heres[3]->BSIM3cbdb, heres[4]->BSIM3cbdb, heres[5]->BSIM3cbdb, heres[6]->BSIM3cbdb, heres[7]->BSIM3cbdb})) + ((Vec8d ){heres[0]->BSIM3cddb, heres[1]->BSIM3cddb, heres[2]->BSIM3cddb, heres[3]->BSIM3cddb, heres[4]->BSIM3cddb, heres[5]->BSIM3cddb, heres[6]->BSIM3cddb, heres[7]->BSIM3cddb}))) * ag0, condmask_true0);
        gcssb = vec8_blend(gcssb, ((((Vec8d ){heres[0]->BSIM3capbs, heres[1]->BSIM3capbs, heres[2]->BSIM3capbs, heres[3]->BSIM3capbs, heres[4]->BSIM3capbs, heres[5]->BSIM3capbs, heres[6]->BSIM3capbs, heres[7]->BSIM3capbs}) + cgso) - ((((Vec8d ){heres[0]->BSIM3cgsb, heres[1]->BSIM3cgsb, heres[2]->BSIM3cgsb, heres[3]->BSIM3cgsb, heres[4]->BSIM3cgsb, heres[5]->BSIM3cgsb, heres[6]->BSIM3cgsb, heres[7]->BSIM3cgsb}) + ((Vec8d ){heres[0]->BSIM3cbsb, heres[1]->BSIM3cbsb, heres[2]->BSIM3cbsb, heres[3]->BSIM3cbsb, heres[4]->BSIM3cbsb, heres[5]->BSIM3cbsb, heres[6]->BSIM3cbsb, heres[7]->BSIM3cbsb})) + ((Vec8d ){heres[0]->BSIM3cdsb, heres[1]->BSIM3cdsb, heres[2]->BSIM3cdsb, heres[3]->BSIM3cdsb, heres[4]->BSIM3cdsb, heres[5]->BSIM3cdsb, heres[6]->BSIM3cdsb, heres[7]->BSIM3cdsb}))) * ag0, condmask_true0);
        gcbgb = vec8_blend(gcbgb, (((Vec8d ){heres[0]->BSIM3cbgb, heres[1]->BSIM3cbgb, heres[2]->BSIM3cbgb, heres[3]->BSIM3cbgb, heres[4]->BSIM3cbgb, heres[5]->BSIM3cbgb, heres[6]->BSIM3cbgb, heres[7]->BSIM3cbgb}) - pParam->BSIM3cgbo) * ag0, condmask_true0);
        gcbdb = vec8_blend(gcbdb, (((Vec8d ){heres[0]->BSIM3cbdb, heres[1]->BSIM3cbdb, heres[2]->BSIM3cbdb, heres[3]->BSIM3cbdb, heres[4]->BSIM3cbdb, heres[5]->BSIM3cbdb, heres[6]->BSIM3cbdb, heres[7]->BSIM3cbdb}) - ((Vec8d ){heres[0]->BSIM3capbd, heres[1]->BSIM3capbd, heres[2]->BSIM3capbd, heres[3]->BSIM3capbd, heres[4]->BSIM3capbd, heres[5]->BSIM3capbd, heres[6]->BSIM3capbd, heres[7]->BSIM3capbd})) * ag0, condmask_true0);
        gcbsb = vec8_blend(gcbsb, (((Vec8d ){heres[0]->BSIM3cbsb, heres[1]->BSIM3cbsb, heres[2]->BSIM3cbsb, heres[3]->BSIM3cbsb, heres[4]->BSIM3cbsb, heres[5]->BSIM3cbsb, heres[6]->BSIM3cbsb, heres[7]->BSIM3cbsb}) - ((Vec8d ){heres[0]->BSIM3capbs, heres[1]->BSIM3capbs, heres[2]->BSIM3capbs, heres[3]->BSIM3capbs, heres[4]->BSIM3capbs, heres[5]->BSIM3capbs, heres[6]->BSIM3capbs, heres[7]->BSIM3capbs})) * ag0, condmask_true0);
        qgd = vec8_blend(qgd, qgdo, condmask_true0);
        qgs = vec8_blend(qgs, qgso, condmask_true0);
        qgb = vec8_blend(qgb, pParam->BSIM3cgbo * vgb, condmask_true0);
        qgate = vec8_blend(qgate, qgate + ((qgd + qgs) + qgb), condmask_true0);
        qbulk = vec8_blend(qbulk, qbulk - qgb, condmask_true0);
        qdrn = vec8_blend(qdrn, qdrn - qgd, condmask_true0);
        qsrc = vec8_blend(qsrc, -((qgate + qbulk) + qdrn), condmask_true0);
        sxpart = vec8_blend(sxpart, vec8_SIMDTOVECTOR(0.6), condmask_true0);
        dxpart = vec8_blend(dxpart, vec8_SIMDTOVECTOR(0.4), condmask_true0);
      }
      else
      {
        if (1)
        {
          Vec8m condmask1 = qcheq > 0.0;
          Vec8m condmask_true1 = condmask_true0 & condmask1;
          Vec8m condmask_false1 = condmask_true0 & (~condmask1);
          T0 = vec8_blend(T0, (((Vec8d ){heres[0]->BSIM3tconst, heres[1]->BSIM3tconst, heres[2]->BSIM3tconst, heres[3]->BSIM3tconst, heres[4]->BSIM3tconst, heres[5]->BSIM3tconst, heres[6]->BSIM3tconst, heres[7]->BSIM3tconst}) * qdef) * ScalingFactor, condmask_true1);
          T0 = vec8_blend(T0, ((-((Vec8d ){heres[0]->BSIM3tconst, heres[1]->BSIM3tconst, heres[2]->BSIM3tconst, heres[3]->BSIM3tconst, heres[4]->BSIM3tconst, heres[5]->BSIM3tconst, heres[6]->BSIM3tconst, heres[7]->BSIM3tconst})) * qdef) * ScalingFactor, condmask_false1);
        }

        ggtg = vec8_blend(ggtg, T0 * ((Vec8d ){heres[0]->BSIM3cqgb, heres[1]->BSIM3cqgb, heres[2]->BSIM3cqgb, heres[3]->BSIM3cqgb, heres[4]->BSIM3cqgb, heres[5]->BSIM3cqgb, heres[6]->BSIM3cqgb, heres[7]->BSIM3cqgb}), condmask_true0);
        {
          if (condmask_true0[0])
            heres[0]->BSIM3gtg = ggtg[0];

          if (condmask_true0[1])
            heres[1]->BSIM3gtg = ggtg[1];

          if (condmask_true0[2])
            heres[2]->BSIM3gtg = ggtg[2];

          if (condmask_true0[3])
            heres[3]->BSIM3gtg = ggtg[3];

          if (condmask_true0[4])
            heres[4]->BSIM3gtg = ggtg[4];

          if (condmask_true0[5])
            heres[5]->BSIM3gtg = ggtg[5];

          if (condmask_true0[6])
            heres[6]->BSIM3gtg = ggtg[6];

          if (condmask_true0[7])
            heres[7]->BSIM3gtg = ggtg[7];

        }
        ggtd = vec8_blend(ggtd, T0 * ((Vec8d ){heres[0]->BSIM3cqdb, heres[1]->BSIM3cqdb, heres[2]->BSIM3cqdb, heres[3]->BSIM3cqdb, heres[4]->BSIM3cqdb, heres[5]->BSIM3cqdb, heres[6]->BSIM3cqdb, heres[7]->BSIM3cqdb}), condmask_true0);
        {
          if (condmask_true0[0])
            heres[0]->BSIM3gtd = ggtd[0];

          if (condmask_true0[1])
            heres[1]->BSIM3gtd = ggtd[1];

          if (condmask_true0[2])
            heres[2]->BSIM3gtd = ggtd[2];

          if (condmask_true0[3])
            heres[3]->BSIM3gtd = ggtd[3];

          if (condmask_true0[4])
            heres[4]->BSIM3gtd = ggtd[4];

          if (condmask_true0[5])
            heres[5]->BSIM3gtd = ggtd[5];

          if (condmask_true0[6])
            heres[6]->BSIM3gtd = ggtd[6];

          if (condmask_true0[7])
            heres[7]->BSIM3gtd = ggtd[7];

        }
        ggts = vec8_blend(ggts, T0 * ((Vec8d ){heres[0]->BSIM3cqsb, heres[1]->BSIM3cqsb, heres[2]->BSIM3cqsb, heres[3]->BSIM3cqsb, heres[4]->BSIM3cqsb, heres[5]->BSIM3cqsb, heres[6]->BSIM3cqsb, heres[7]->BSIM3cqsb}), condmask_true0);
        {
          if (condmask_true0[0])
            heres[0]->BSIM3gts = ggts[0];

          if (condmask_true0[1])
            heres[1]->BSIM3gts = ggts[1];

          if (condmask_true0[2])
            heres[2]->BSIM3gts = ggts[2];

          if (condmask_true0[3])
            heres[3]->BSIM3gts = ggts[3];

          if (condmask_true0[4])
            heres[4]->BSIM3gts = ggts[4];

          if (condmask_true0[5])
            heres[5]->BSIM3gts = ggts[5];

          if (condmask_true0[6])
            heres[6]->BSIM3gts = ggts[6];

          if (condmask_true0[7])
            heres[7]->BSIM3gts = ggts[7];

        }
        ggtb = vec8_blend(ggtb, T0 * ((Vec8d ){heres[0]->BSIM3cqbb, heres[1]->BSIM3cqbb, heres[2]->BSIM3cqbb, heres[3]->BSIM3cqbb, heres[4]->BSIM3cqbb, heres[5]->BSIM3cqbb, heres[6]->BSIM3cqbb, heres[7]->BSIM3cqbb}), condmask_true0);
        {
          if (condmask_true0[0])
            heres[0]->BSIM3gtb = ggtb[0];

          if (condmask_true0[1])
            heres[1]->BSIM3gtb = ggtb[1];

          if (condmask_true0[2])
            heres[2]->BSIM3gtb = ggtb[2];

          if (condmask_true0[3])
            heres[3]->BSIM3gtb = ggtb[3];

          if (condmask_true0[4])
            heres[4]->BSIM3gtb = ggtb[4];

          if (condmask_true0[5])
            heres[5]->BSIM3gtb = ggtb[5];

          if (condmask_true0[6])
            heres[6]->BSIM3gtb = ggtb[6];

          if (condmask_true0[7])
            heres[7]->BSIM3gtb = ggtb[7];

        }
        gqdef = vec8_blend(gqdef, vec8_SIMDTOVECTOR(ScalingFactor * ag0), condmask_true0);
        gcqgb = vec8_blend(gcqgb, ((Vec8d ){heres[0]->BSIM3cqgb, heres[1]->BSIM3cqgb, heres[2]->BSIM3cqgb, heres[3]->BSIM3cqgb, heres[4]->BSIM3cqgb, heres[5]->BSIM3cqgb, heres[6]->BSIM3cqgb, heres[7]->BSIM3cqgb}) * ag0, condmask_true0);
        gcqdb = vec8_blend(gcqdb, ((Vec8d ){heres[0]->BSIM3cqdb, heres[1]->BSIM3cqdb, heres[2]->BSIM3cqdb, heres[3]->BSIM3cqdb, heres[4]->BSIM3cqdb, heres[5]->BSIM3cqdb, heres[6]->BSIM3cqdb, heres[7]->BSIM3cqdb}) * ag0, condmask_true0);
        gcqsb = vec8_blend(gcqsb, ((Vec8d ){heres[0]->BSIM3cqsb, heres[1]->BSIM3cqsb, heres[2]->BSIM3cqsb, heres[3]->BSIM3cqsb, heres[4]->BSIM3cqsb, heres[5]->BSIM3cqsb, heres[6]->BSIM3cqsb, heres[7]->BSIM3cqsb}) * ag0, condmask_true0);
        gcqbb = vec8_blend(gcqbb, ((Vec8d ){heres[0]->BSIM3cqbb, heres[1]->BSIM3cqbb, heres[2]->BSIM3cqbb, heres[3]->BSIM3cqbb, heres[4]->BSIM3cqbb, heres[5]->BSIM3cqbb, heres[6]->BSIM3cqbb, heres[7]->BSIM3cqbb}) * ag0, condmask_true0);
        gcggb = vec8_blend(gcggb, ((cgdo + cgso) + pParam->BSIM3cgbo) * ag0, condmask_true0);
        gcgdb = vec8_blend(gcgdb, (-cgdo) * ag0, condmask_true0);
        gcgsb = vec8_blend(gcgsb, (-cgso) * ag0, condmask_true0);
        gcdgb = vec8_blend(gcdgb, (-cgdo) * ag0, condmask_true0);
        gcddb = vec8_blend(gcddb, (((Vec8d ){heres[0]->BSIM3capbd, heres[1]->BSIM3capbd, heres[2]->BSIM3capbd, heres[3]->BSIM3capbd, heres[4]->BSIM3capbd, heres[5]->BSIM3capbd, heres[6]->BSIM3capbd, heres[7]->BSIM3capbd}) + cgdo) * ag0, condmask_true0);
        gcdsb = vec8_blend(gcdsb, vec8_SIMDTOVECTOR(0.0), condmask_true0);
        gcsgb = vec8_blend(gcsgb, (-cgso) * ag0, condmask_true0);
        gcsdb = vec8_blend(gcsdb, vec8_SIMDTOVECTOR(0.0), condmask_true0);
        gcssb = vec8_blend(gcssb, (((Vec8d ){heres[0]->BSIM3capbs, heres[1]->BSIM3capbs, heres[2]->BSIM3capbs, heres[3]->BSIM3capbs, heres[4]->BSIM3capbs, heres[5]->BSIM3capbs, heres[6]->BSIM3capbs, heres[7]->BSIM3capbs}) + cgso) * ag0, condmask_true0);
        gcbgb = vec8_blend(gcbgb, vec8_SIMDTOVECTOR((-pParam->BSIM3cgbo) * ag0), condmask_true0);
        gcbdb = vec8_blend(gcbdb, (-((Vec8d ){heres[0]->BSIM3capbd, heres[1]->BSIM3capbd, heres[2]->BSIM3capbd, heres[3]->BSIM3capbd, heres[4]->BSIM3capbd, heres[5]->BSIM3capbd, heres[6]->BSIM3capbd, heres[7]->BSIM3capbd})) * ag0, condmask_true0);
        gcbsb = vec8_blend(gcbsb, (-((Vec8d ){heres[0]->BSIM3capbs, heres[1]->BSIM3capbs, heres[2]->BSIM3capbs, heres[3]->BSIM3capbs, heres[4]->BSIM3capbs, heres[5]->BSIM3capbs, heres[6]->BSIM3capbs, heres[7]->BSIM3capbs})) * ag0, condmask_true0);
        if (1)
        {
          Vec8m condmask1 = vec8_fabs(qcheq) <= (1.0e-5 * CoxWL);
          Vec8m condmask_true1 = condmask_true0 & condmask1;
          Vec8m condmask_false1 = condmask_true0 & (~condmask1);
          {
            if (model->BSIM3xpart < 0.5)
            {
              dxpart = vec8_blend(dxpart, vec8_SIMDTOVECTOR(0.4), condmask_true1);
            }
            else
              if (model->BSIM3xpart > 0.5)
            {
              dxpart = vec8_blend(dxpart, vec8_SIMDTOVECTOR(0.0), condmask_true1);
            }
            else
            {
              dxpart = vec8_blend(dxpart, vec8_SIMDTOVECTOR(0.5), condmask_true1);
            }


          }
          {
            dxpart = vec8_blend(dxpart, qdrn / qcheq, condmask_false1);
            Cdd = vec8_blend(Cdd, (Vec8d ){heres[0]->BSIM3cddb, heres[1]->BSIM3cddb, heres[2]->BSIM3cddb, heres[3]->BSIM3cddb, heres[4]->BSIM3cddb, heres[5]->BSIM3cddb, heres[6]->BSIM3cddb, heres[7]->BSIM3cddb}, condmask_false1);
            Csd = vec8_blend(Csd, -((((Vec8d ){heres[0]->BSIM3cgdb, heres[1]->BSIM3cgdb, heres[2]->BSIM3cgdb, heres[3]->BSIM3cgdb, heres[4]->BSIM3cgdb, heres[5]->BSIM3cgdb, heres[6]->BSIM3cgdb, heres[7]->BSIM3cgdb}) + ((Vec8d ){heres[0]->BSIM3cddb, heres[1]->BSIM3cddb, heres[2]->BSIM3cddb, heres[3]->BSIM3cddb, heres[4]->BSIM3cddb, heres[5]->BSIM3cddb, heres[6]->BSIM3cddb, heres[7]->BSIM3cddb})) + ((Vec8d ){heres[0]->BSIM3cbdb, heres[1]->BSIM3cbdb, heres[2]->BSIM3cbdb, heres[3]->BSIM3cbdb, heres[4]->BSIM3cbdb, heres[5]->BSIM3cbdb, heres[6]->BSIM3cbdb, heres[7]->BSIM3cbdb})), condmask_false1);
            ddxpart_dVd = vec8_blend(ddxpart_dVd, (Cdd - (dxpart * (Cdd + Csd))) / qcheq, condmask_false1);
            Cdg = vec8_blend(Cdg, (Vec8d ){heres[0]->BSIM3cdgb, heres[1]->BSIM3cdgb, heres[2]->BSIM3cdgb, heres[3]->BSIM3cdgb, heres[4]->BSIM3cdgb, heres[5]->BSIM3cdgb, heres[6]->BSIM3cdgb, heres[7]->BSIM3cdgb}, condmask_false1);
            Csg = vec8_blend(Csg, -((((Vec8d ){heres[0]->BSIM3cggb, heres[1]->BSIM3cggb, heres[2]->BSIM3cggb, heres[3]->BSIM3cggb, heres[4]->BSIM3cggb, heres[5]->BSIM3cggb, heres[6]->BSIM3cggb, heres[7]->BSIM3cggb}) + ((Vec8d ){heres[0]->BSIM3cdgb, heres[1]->BSIM3cdgb, heres[2]->BSIM3cdgb, heres[3]->BSIM3cdgb, heres[4]->BSIM3cdgb, heres[5]->BSIM3cdgb, heres[6]->BSIM3cdgb, heres[7]->BSIM3cdgb})) + ((Vec8d ){heres[0]->BSIM3cbgb, heres[1]->BSIM3cbgb, heres[2]->BSIM3cbgb, heres[3]->BSIM3cbgb, heres[4]->BSIM3cbgb, heres[5]->BSIM3cbgb, heres[6]->BSIM3cbgb, heres[7]->BSIM3cbgb})), condmask_false1);
            ddxpart_dVg = vec8_blend(ddxpart_dVg, (Cdg - (dxpart * (Cdg + Csg))) / qcheq, condmask_false1);
            Cds = vec8_blend(Cds, (Vec8d ){heres[0]->BSIM3cdsb, heres[1]->BSIM3cdsb, heres[2]->BSIM3cdsb, heres[3]->BSIM3cdsb, heres[4]->BSIM3cdsb, heres[5]->BSIM3cdsb, heres[6]->BSIM3cdsb, heres[7]->BSIM3cdsb}, condmask_false1);
            Css = vec8_blend(Css, -((((Vec8d ){heres[0]->BSIM3cgsb, heres[1]->BSIM3cgsb, heres[2]->BSIM3cgsb, heres[3]->BSIM3cgsb, heres[4]->BSIM3cgsb, heres[5]->BSIM3cgsb, heres[6]->BSIM3cgsb, heres[7]->BSIM3cgsb}) + ((Vec8d ){heres[0]->BSIM3cdsb, heres[1]->BSIM3cdsb, heres[2]->BSIM3cdsb, heres[3]->BSIM3cdsb, heres[4]->BSIM3cdsb, heres[5]->BSIM3cdsb, heres[6]->BSIM3cdsb, heres[7]->BSIM3cdsb})) + ((Vec8d ){heres[0]->BSIM3cbsb, heres[1]->BSIM3cbsb, heres[2]->BSIM3cbsb, heres[3]->BSIM3cbsb, heres[4]->BSIM3cbsb, heres[5]->BSIM3cbsb, heres[6]->BSIM3cbsb, heres[7]->BSIM3cbsb})), condmask_false1);
            ddxpart_dVs = vec8_blend(ddxpart_dVs, (Cds - (dxpart * (Cds + Css))) / qcheq, condmask_false1);
            ddxpart_dVb = vec8_blend(ddxpart_dVb, -((ddxpart_dVd + ddxpart_dVg) + ddxpart_dVs), condmask_false1);
          }
        }

        sxpart = vec8_blend(sxpart, 1.0 - dxpart, condmask_true0);
        dsxpart_dVd = vec8_blend(dsxpart_dVd, -ddxpart_dVd, condmask_true0);
        dsxpart_dVg = vec8_blend(dsxpart_dVg, -ddxpart_dVg, condmask_true0);
        dsxpart_dVs = vec8_blend(dsxpart_dVs, -ddxpart_dVs, condmask_true0);
        dsxpart_dVb = vec8_blend(dsxpart_dVb, -((dsxpart_dVd + dsxpart_dVg) + dsxpart_dVs), condmask_true0);
        qgd = vec8_blend(qgd, qgdo, condmask_true0);
        qgs = vec8_blend(qgs, qgso, condmask_true0);
        qgb = vec8_blend(qgb, pParam->BSIM3cgbo * vgb, condmask_true0);
        qgate = vec8_blend(qgate, (qgd + qgs) + qgb, condmask_true0);
        qbulk = vec8_blend(qbulk, -qgb, condmask_true0);
        qdrn = vec8_blend(qdrn, -qgd, condmask_true0);
        qsrc = vec8_blend(qsrc, -((qgate + qbulk) + qdrn), condmask_true0);
      }

    }
    {
      if (heres[0]->BSIM3nqsMod == 0)
      {
        gcggb = vec8_blend(gcggb, (((((Vec8d ){heres[0]->BSIM3cggb, heres[1]->BSIM3cggb, heres[2]->BSIM3cggb, heres[3]->BSIM3cggb, heres[4]->BSIM3cggb, heres[5]->BSIM3cggb, heres[6]->BSIM3cggb, heres[7]->BSIM3cggb}) + cgdo) + cgso) + pParam->BSIM3cgbo) * ag0, condmask_false0);
        gcgdb = vec8_blend(gcgdb, (((Vec8d ){heres[0]->BSIM3cgsb, heres[1]->BSIM3cgsb, heres[2]->BSIM3cgsb, heres[3]->BSIM3cgsb, heres[4]->BSIM3cgsb, heres[5]->BSIM3cgsb, heres[6]->BSIM3cgsb, heres[7]->BSIM3cgsb}) - cgdo) * ag0, condmask_false0);
        gcgsb = vec8_blend(gcgsb, (((Vec8d ){heres[0]->BSIM3cgdb, heres[1]->BSIM3cgdb, heres[2]->BSIM3cgdb, heres[3]->BSIM3cgdb, heres[4]->BSIM3cgdb, heres[5]->BSIM3cgdb, heres[6]->BSIM3cgdb, heres[7]->BSIM3cgdb}) - cgso) * ag0, condmask_false0);
        gcdgb = vec8_blend(gcdgb, (-(((((Vec8d ){heres[0]->BSIM3cggb, heres[1]->BSIM3cggb, heres[2]->BSIM3cggb, heres[3]->BSIM3cggb, heres[4]->BSIM3cggb, heres[5]->BSIM3cggb, heres[6]->BSIM3cggb, heres[7]->BSIM3cggb}) + ((Vec8d ){heres[0]->BSIM3cbgb, heres[1]->BSIM3cbgb, heres[2]->BSIM3cbgb, heres[3]->BSIM3cbgb, heres[4]->BSIM3cbgb, heres[5]->BSIM3cbgb, heres[6]->BSIM3cbgb, heres[7]->BSIM3cbgb})) + ((Vec8d ){heres[0]->BSIM3cdgb, heres[1]->BSIM3cdgb, heres[2]->BSIM3cdgb, heres[3]->BSIM3cdgb, heres[4]->BSIM3cdgb, heres[5]->BSIM3cdgb, heres[6]->BSIM3cdgb, heres[7]->BSIM3cdgb})) + cgdo)) * ag0, condmask_false0);
        gcddb = vec8_blend(gcddb, ((((Vec8d ){heres[0]->BSIM3capbd, heres[1]->BSIM3capbd, heres[2]->BSIM3capbd, heres[3]->BSIM3capbd, heres[4]->BSIM3capbd, heres[5]->BSIM3capbd, heres[6]->BSIM3capbd, heres[7]->BSIM3capbd}) + cgdo) - ((((Vec8d ){heres[0]->BSIM3cgsb, heres[1]->BSIM3cgsb, heres[2]->BSIM3cgsb, heres[3]->BSIM3cgsb, heres[4]->BSIM3cgsb, heres[5]->BSIM3cgsb, heres[6]->BSIM3cgsb, heres[7]->BSIM3cgsb}) + ((Vec8d ){heres[0]->BSIM3cbsb, heres[1]->BSIM3cbsb, heres[2]->BSIM3cbsb, heres[3]->BSIM3cbsb, heres[4]->BSIM3cbsb, heres[5]->BSIM3cbsb, heres[6]->BSIM3cbsb, heres[7]->BSIM3cbsb})) + ((Vec8d ){heres[0]->BSIM3cdsb, heres[1]->BSIM3cdsb, heres[2]->BSIM3cdsb, heres[3]->BSIM3cdsb, heres[4]->BSIM3cdsb, heres[5]->BSIM3cdsb, heres[6]->BSIM3cdsb, heres[7]->BSIM3cdsb}))) * ag0, condmask_false0);
        gcdsb = vec8_blend(gcdsb, (-((((Vec8d ){heres[0]->BSIM3cgdb, heres[1]->BSIM3cgdb, heres[2]->BSIM3cgdb, heres[3]->BSIM3cgdb, heres[4]->BSIM3cgdb, heres[5]->BSIM3cgdb, heres[6]->BSIM3cgdb, heres[7]->BSIM3cgdb}) + ((Vec8d ){heres[0]->BSIM3cbdb, heres[1]->BSIM3cbdb, heres[2]->BSIM3cbdb, heres[3]->BSIM3cbdb, heres[4]->BSIM3cbdb, heres[5]->BSIM3cbdb, heres[6]->BSIM3cbdb, heres[7]->BSIM3cbdb})) + ((Vec8d ){heres[0]->BSIM3cddb, heres[1]->BSIM3cddb, heres[2]->BSIM3cddb, heres[3]->BSIM3cddb, heres[4]->BSIM3cddb, heres[5]->BSIM3cddb, heres[6]->BSIM3cddb, heres[7]->BSIM3cddb}))) * ag0, condmask_false0);
        gcsgb = vec8_blend(gcsgb, (((Vec8d ){heres[0]->BSIM3cdgb, heres[1]->BSIM3cdgb, heres[2]->BSIM3cdgb, heres[3]->BSIM3cdgb, heres[4]->BSIM3cdgb, heres[5]->BSIM3cdgb, heres[6]->BSIM3cdgb, heres[7]->BSIM3cdgb}) - cgso) * ag0, condmask_false0);
        gcsdb = vec8_blend(gcsdb, ((Vec8d ){heres[0]->BSIM3cdsb, heres[1]->BSIM3cdsb, heres[2]->BSIM3cdsb, heres[3]->BSIM3cdsb, heres[4]->BSIM3cdsb, heres[5]->BSIM3cdsb, heres[6]->BSIM3cdsb, heres[7]->BSIM3cdsb}) * ag0, condmask_false0);
        gcssb = vec8_blend(gcssb, ((((Vec8d ){heres[0]->BSIM3cddb, heres[1]->BSIM3cddb, heres[2]->BSIM3cddb, heres[3]->BSIM3cddb, heres[4]->BSIM3cddb, heres[5]->BSIM3cddb, heres[6]->BSIM3cddb, heres[7]->BSIM3cddb}) + ((Vec8d ){heres[0]->BSIM3capbs, heres[1]->BSIM3capbs, heres[2]->BSIM3capbs, heres[3]->BSIM3capbs, heres[4]->BSIM3capbs, heres[5]->BSIM3capbs, heres[6]->BSIM3capbs, heres[7]->BSIM3capbs})) + cgso) * ag0, condmask_false0);
        gcbgb = vec8_blend(gcbgb, (((Vec8d ){heres[0]->BSIM3cbgb, heres[1]->BSIM3cbgb, heres[2]->BSIM3cbgb, heres[3]->BSIM3cbgb, heres[4]->BSIM3cbgb, heres[5]->BSIM3cbgb, heres[6]->BSIM3cbgb, heres[7]->BSIM3cbgb}) - pParam->BSIM3cgbo) * ag0, condmask_false0);
        gcbdb = vec8_blend(gcbdb, (((Vec8d ){heres[0]->BSIM3cbsb, heres[1]->BSIM3cbsb, heres[2]->BSIM3cbsb, heres[3]->BSIM3cbsb, heres[4]->BSIM3cbsb, heres[5]->BSIM3cbsb, heres[6]->BSIM3cbsb, heres[7]->BSIM3cbsb}) - ((Vec8d ){heres[0]->BSIM3capbd, heres[1]->BSIM3capbd, heres[2]->BSIM3capbd, heres[3]->BSIM3capbd, heres[4]->BSIM3capbd, heres[5]->BSIM3capbd, heres[6]->BSIM3capbd, heres[7]->BSIM3capbd})) * ag0, condmask_false0);
        gcbsb = vec8_blend(gcbsb, (((Vec8d ){heres[0]->BSIM3cbdb, heres[1]->BSIM3cbdb, heres[2]->BSIM3cbdb, heres[3]->BSIM3cbdb, heres[4]->BSIM3cbdb, heres[5]->BSIM3cbdb, heres[6]->BSIM3cbdb, heres[7]->BSIM3cbdb}) - ((Vec8d ){heres[0]->BSIM3capbs, heres[1]->BSIM3capbs, heres[2]->BSIM3capbs, heres[3]->BSIM3capbs, heres[4]->BSIM3capbs, heres[5]->BSIM3capbs, heres[6]->BSIM3capbs, heres[7]->BSIM3capbs})) * ag0, condmask_false0);
        qgd = vec8_blend(qgd, qgdo, condmask_false0);
        qgs = vec8_blend(qgs, qgso, condmask_false0);
        qgb = vec8_blend(qgb, pParam->BSIM3cgbo * vgb, condmask_false0);
        qgate = vec8_blend(qgate, qgate + ((qgd + qgs) + qgb), condmask_false0);
        qbulk = vec8_blend(qbulk, qbulk - qgb, condmask_false0);
        qsrc = vec8_blend(qsrc, qdrn - qgs, condmask_false0);
        qdrn = vec8_blend(qdrn, -((qgate + qbulk) + qsrc), condmask_false0);
        sxpart = vec8_blend(sxpart, vec8_SIMDTOVECTOR(0.4), condmask_false0);
        dxpart = vec8_blend(dxpart, vec8_SIMDTOVECTOR(0.6), condmask_false0);
      }
      else
      {
        if (1)
        {
          Vec8m condmask1 = qcheq > 0.0;
          Vec8m condmask_true1 = condmask_false0 & condmask1;
          Vec8m condmask_false1 = condmask_false0 & (~condmask1);
          T0 = vec8_blend(T0, (((Vec8d ){heres[0]->BSIM3tconst, heres[1]->BSIM3tconst, heres[2]->BSIM3tconst, heres[3]->BSIM3tconst, heres[4]->BSIM3tconst, heres[5]->BSIM3tconst, heres[6]->BSIM3tconst, heres[7]->BSIM3tconst}) * qdef) * ScalingFactor, condmask_true1);
          T0 = vec8_blend(T0, ((-((Vec8d ){heres[0]->BSIM3tconst, heres[1]->BSIM3tconst, heres[2]->BSIM3tconst, heres[3]->BSIM3tconst, heres[4]->BSIM3tconst, heres[5]->BSIM3tconst, heres[6]->BSIM3tconst, heres[7]->BSIM3tconst})) * qdef) * ScalingFactor, condmask_false1);
        }

        ggtg = vec8_blend(ggtg, T0 * ((Vec8d ){heres[0]->BSIM3cqgb, heres[1]->BSIM3cqgb, heres[2]->BSIM3cqgb, heres[3]->BSIM3cqgb, heres[4]->BSIM3cqgb, heres[5]->BSIM3cqgb, heres[6]->BSIM3cqgb, heres[7]->BSIM3cqgb}), condmask_false0);
        {
          if (condmask_false0[0])
            heres[0]->BSIM3gtg = ggtg[0];

          if (condmask_false0[1])
            heres[1]->BSIM3gtg = ggtg[1];

          if (condmask_false0[2])
            heres[2]->BSIM3gtg = ggtg[2];

          if (condmask_false0[3])
            heres[3]->BSIM3gtg = ggtg[3];

          if (condmask_false0[4])
            heres[4]->BSIM3gtg = ggtg[4];

          if (condmask_false0[5])
            heres[5]->BSIM3gtg = ggtg[5];

          if (condmask_false0[6])
            heres[6]->BSIM3gtg = ggtg[6];

          if (condmask_false0[7])
            heres[7]->BSIM3gtg = ggtg[7];

        }
        ggts = vec8_blend(ggts, T0 * ((Vec8d ){heres[0]->BSIM3cqdb, heres[1]->BSIM3cqdb, heres[2]->BSIM3cqdb, heres[3]->BSIM3cqdb, heres[4]->BSIM3cqdb, heres[5]->BSIM3cqdb, heres[6]->BSIM3cqdb, heres[7]->BSIM3cqdb}), condmask_false0);
        {
          if (condmask_false0[0])
            heres[0]->BSIM3gtd = ggts[0];

          if (condmask_false0[1])
            heres[1]->BSIM3gtd = ggts[1];

          if (condmask_false0[2])
            heres[2]->BSIM3gtd = ggts[2];

          if (condmask_false0[3])
            heres[3]->BSIM3gtd = ggts[3];

          if (condmask_false0[4])
            heres[4]->BSIM3gtd = ggts[4];

          if (condmask_false0[5])
            heres[5]->BSIM3gtd = ggts[5];

          if (condmask_false0[6])
            heres[6]->BSIM3gtd = ggts[6];

          if (condmask_false0[7])
            heres[7]->BSIM3gtd = ggts[7];

        }
        ggtd = vec8_blend(ggtd, T0 * ((Vec8d ){heres[0]->BSIM3cqsb, heres[1]->BSIM3cqsb, heres[2]->BSIM3cqsb, heres[3]->BSIM3cqsb, heres[4]->BSIM3cqsb, heres[5]->BSIM3cqsb, heres[6]->BSIM3cqsb, heres[7]->BSIM3cqsb}), condmask_false0);
        {
          if (condmask_false0[0])
            heres[0]->BSIM3gts = ggtd[0];

          if (condmask_false0[1])
            heres[1]->BSIM3gts = ggtd[1];

          if (condmask_false0[2])
            heres[2]->BSIM3gts = ggtd[2];

          if (condmask_false0[3])
            heres[3]->BSIM3gts = ggtd[3];

          if (condmask_false0[4])
            heres[4]->BSIM3gts = ggtd[4];

          if (condmask_false0[5])
            heres[5]->BSIM3gts = ggtd[5];

          if (condmask_false0[6])
            heres[6]->BSIM3gts = ggtd[6];

          if (condmask_false0[7])
            heres[7]->BSIM3gts = ggtd[7];

        }
        ggtb = vec8_blend(ggtb, T0 * ((Vec8d ){heres[0]->BSIM3cqbb, heres[1]->BSIM3cqbb, heres[2]->BSIM3cqbb, heres[3]->BSIM3cqbb, heres[4]->BSIM3cqbb, heres[5]->BSIM3cqbb, heres[6]->BSIM3cqbb, heres[7]->BSIM3cqbb}), condmask_false0);
        {
          if (condmask_false0[0])
            heres[0]->BSIM3gtb = ggtb[0];

          if (condmask_false0[1])
            heres[1]->BSIM3gtb = ggtb[1];

          if (condmask_false0[2])
            heres[2]->BSIM3gtb = ggtb[2];

          if (condmask_false0[3])
            heres[3]->BSIM3gtb = ggtb[3];

          if (condmask_false0[4])
            heres[4]->BSIM3gtb = ggtb[4];

          if (condmask_false0[5])
            heres[5]->BSIM3gtb = ggtb[5];

          if (condmask_false0[6])
            heres[6]->BSIM3gtb = ggtb[6];

          if (condmask_false0[7])
            heres[7]->BSIM3gtb = ggtb[7];

        }
        gqdef = vec8_blend(gqdef, vec8_SIMDTOVECTOR(ScalingFactor * ag0), condmask_false0);
        gcqgb = vec8_blend(gcqgb, ((Vec8d ){heres[0]->BSIM3cqgb, heres[1]->BSIM3cqgb, heres[2]->BSIM3cqgb, heres[3]->BSIM3cqgb, heres[4]->BSIM3cqgb, heres[5]->BSIM3cqgb, heres[6]->BSIM3cqgb, heres[7]->BSIM3cqgb}) * ag0, condmask_false0);
        gcqdb = vec8_blend(gcqdb, ((Vec8d ){heres[0]->BSIM3cqsb, heres[1]->BSIM3cqsb, heres[2]->BSIM3cqsb, heres[3]->BSIM3cqsb, heres[4]->BSIM3cqsb, heres[5]->BSIM3cqsb, heres[6]->BSIM3cqsb, heres[7]->BSIM3cqsb}) * ag0, condmask_false0);
        gcqsb = vec8_blend(gcqsb, ((Vec8d ){heres[0]->BSIM3cqdb, heres[1]->BSIM3cqdb, heres[2]->BSIM3cqdb, heres[3]->BSIM3cqdb, heres[4]->BSIM3cqdb, heres[5]->BSIM3cqdb, heres[6]->BSIM3cqdb, heres[7]->BSIM3cqdb}) * ag0, condmask_false0);
        gcqbb = vec8_blend(gcqbb, ((Vec8d ){heres[0]->BSIM3cqbb, heres[1]->BSIM3cqbb, heres[2]->BSIM3cqbb, heres[3]->BSIM3cqbb, heres[4]->BSIM3cqbb, heres[5]->BSIM3cqbb, heres[6]->BSIM3cqbb, heres[7]->BSIM3cqbb}) * ag0, condmask_false0);
        gcggb = vec8_blend(gcggb, ((cgdo + cgso) + pParam->BSIM3cgbo) * ag0, condmask_false0);
        gcgdb = vec8_blend(gcgdb, (-cgdo) * ag0, condmask_false0);
        gcgsb = vec8_blend(gcgsb, (-cgso) * ag0, condmask_false0);
        gcdgb = vec8_blend(gcdgb, (-cgdo) * ag0, condmask_false0);
        gcddb = vec8_blend(gcddb, (((Vec8d ){heres[0]->BSIM3capbd, heres[1]->BSIM3capbd, heres[2]->BSIM3capbd, heres[3]->BSIM3capbd, heres[4]->BSIM3capbd, heres[5]->BSIM3capbd, heres[6]->BSIM3capbd, heres[7]->BSIM3capbd}) + cgdo) * ag0, condmask_false0);
        gcdsb = vec8_blend(gcdsb, vec8_SIMDTOVECTOR(0.0), condmask_false0);
        gcsgb = vec8_blend(gcsgb, (-cgso) * ag0, condmask_false0);
        gcsdb = vec8_blend(gcsdb, vec8_SIMDTOVECTOR(0.0), condmask_false0);
        gcssb = vec8_blend(gcssb, (((Vec8d ){heres[0]->BSIM3capbs, heres[1]->BSIM3capbs, heres[2]->BSIM3capbs, heres[3]->BSIM3capbs, heres[4]->BSIM3capbs, heres[5]->BSIM3capbs, heres[6]->BSIM3capbs, heres[7]->BSIM3capbs}) + cgso) * ag0, condmask_false0);
        gcbgb = vec8_blend(gcbgb, vec8_SIMDTOVECTOR((-pParam->BSIM3cgbo) * ag0), condmask_false0);
        gcbdb = vec8_blend(gcbdb, (-((Vec8d ){heres[0]->BSIM3capbd, heres[1]->BSIM3capbd, heres[2]->BSIM3capbd, heres[3]->BSIM3capbd, heres[4]->BSIM3capbd, heres[5]->BSIM3capbd, heres[6]->BSIM3capbd, heres[7]->BSIM3capbd})) * ag0, condmask_false0);
        gcbsb = vec8_blend(gcbsb, (-((Vec8d ){heres[0]->BSIM3capbs, heres[1]->BSIM3capbs, heres[2]->BSIM3capbs, heres[3]->BSIM3capbs, heres[4]->BSIM3capbs, heres[5]->BSIM3capbs, heres[6]->BSIM3capbs, heres[7]->BSIM3capbs})) * ag0, condmask_false0);
        if (1)
        {
          Vec8m condmask1 = vec8_fabs(qcheq) <= (1.0e-5 * CoxWL);
          Vec8m condmask_true1 = condmask_false0 & condmask1;
          Vec8m condmask_false1 = condmask_false0 & (~condmask1);
          {
            if (model->BSIM3xpart < 0.5)
            {
              sxpart = vec8_blend(sxpart, vec8_SIMDTOVECTOR(0.4), condmask_true1);
            }
            else
              if (model->BSIM3xpart > 0.5)
            {
              sxpart = vec8_blend(sxpart, vec8_SIMDTOVECTOR(0.0), condmask_true1);
            }
            else
            {
              sxpart = vec8_blend(sxpart, vec8_SIMDTOVECTOR(0.5), condmask_true1);
            }


          }
          {
            sxpart = vec8_blend(sxpart, qdrn / qcheq, condmask_false1);
            Css = vec8_blend(Css, (Vec8d ){heres[0]->BSIM3cddb, heres[1]->BSIM3cddb, heres[2]->BSIM3cddb, heres[3]->BSIM3cddb, heres[4]->BSIM3cddb, heres[5]->BSIM3cddb, heres[6]->BSIM3cddb, heres[7]->BSIM3cddb}, condmask_false1);
            Cds = vec8_blend(Cds, -((((Vec8d ){heres[0]->BSIM3cgdb, heres[1]->BSIM3cgdb, heres[2]->BSIM3cgdb, heres[3]->BSIM3cgdb, heres[4]->BSIM3cgdb, heres[5]->BSIM3cgdb, heres[6]->BSIM3cgdb, heres[7]->BSIM3cgdb}) + ((Vec8d ){heres[0]->BSIM3cddb, heres[1]->BSIM3cddb, heres[2]->BSIM3cddb, heres[3]->BSIM3cddb, heres[4]->BSIM3cddb, heres[5]->BSIM3cddb, heres[6]->BSIM3cddb, heres[7]->BSIM3cddb})) + ((Vec8d ){heres[0]->BSIM3cbdb, heres[1]->BSIM3cbdb, heres[2]->BSIM3cbdb, heres[3]->BSIM3cbdb, heres[4]->BSIM3cbdb, heres[5]->BSIM3cbdb, heres[6]->BSIM3cbdb, heres[7]->BSIM3cbdb})), condmask_false1);
            dsxpart_dVs = vec8_blend(dsxpart_dVs, (Css - (sxpart * (Css + Cds))) / qcheq, condmask_false1);
            Csg = vec8_blend(Csg, (Vec8d ){heres[0]->BSIM3cdgb, heres[1]->BSIM3cdgb, heres[2]->BSIM3cdgb, heres[3]->BSIM3cdgb, heres[4]->BSIM3cdgb, heres[5]->BSIM3cdgb, heres[6]->BSIM3cdgb, heres[7]->BSIM3cdgb}, condmask_false1);
            Cdg = vec8_blend(Cdg, -((((Vec8d ){heres[0]->BSIM3cggb, heres[1]->BSIM3cggb, heres[2]->BSIM3cggb, heres[3]->BSIM3cggb, heres[4]->BSIM3cggb, heres[5]->BSIM3cggb, heres[6]->BSIM3cggb, heres[7]->BSIM3cggb}) + ((Vec8d ){heres[0]->BSIM3cdgb, heres[1]->BSIM3cdgb, heres[2]->BSIM3cdgb, heres[3]->BSIM3cdgb, heres[4]->BSIM3cdgb, heres[5]->BSIM3cdgb, heres[6]->BSIM3cdgb, heres[7]->BSIM3cdgb})) + ((Vec8d ){heres[0]->BSIM3cbgb, heres[1]->BSIM3cbgb, heres[2]->BSIM3cbgb, heres[3]->BSIM3cbgb, heres[4]->BSIM3cbgb, heres[5]->BSIM3cbgb, heres[6]->BSIM3cbgb, heres[7]->BSIM3cbgb})), condmask_false1);
            dsxpart_dVg = vec8_blend(dsxpart_dVg, (Csg - (sxpart * (Csg + Cdg))) / qcheq, condmask_false1);
            Csd = vec8_blend(Csd, (Vec8d ){heres[0]->BSIM3cdsb, heres[1]->BSIM3cdsb, heres[2]->BSIM3cdsb, heres[3]->BSIM3cdsb, heres[4]->BSIM3cdsb, heres[5]->BSIM3cdsb, heres[6]->BSIM3cdsb, heres[7]->BSIM3cdsb}, condmask_false1);
            Cdd = vec8_blend(Cdd, -((((Vec8d ){heres[0]->BSIM3cgsb, heres[1]->BSIM3cgsb, heres[2]->BSIM3cgsb, heres[3]->BSIM3cgsb, heres[4]->BSIM3cgsb, heres[5]->BSIM3cgsb, heres[6]->BSIM3cgsb, heres[7]->BSIM3cgsb}) + ((Vec8d ){heres[0]->BSIM3cdsb, heres[1]->BSIM3cdsb, heres[2]->BSIM3cdsb, heres[3]->BSIM3cdsb, heres[4]->BSIM3cdsb, heres[5]->BSIM3cdsb, heres[6]->BSIM3cdsb, heres[7]->BSIM3cdsb})) + ((Vec8d ){heres[0]->BSIM3cbsb, heres[1]->BSIM3cbsb, heres[2]->BSIM3cbsb, heres[3]->BSIM3cbsb, heres[4]->BSIM3cbsb, heres[5]->BSIM3cbsb, heres[6]->BSIM3cbsb, heres[7]->BSIM3cbsb})), condmask_false1);
            dsxpart_dVd = vec8_blend(dsxpart_dVd, (Csd - (sxpart * (Csd + Cdd))) / qcheq, condmask_false1);
            dsxpart_dVb = vec8_blend(dsxpart_dVb, -((dsxpart_dVd + dsxpart_dVg) + dsxpart_dVs), condmask_false1);
          }
        }

        dxpart = vec8_blend(dxpart, 1.0 - sxpart, condmask_false0);
        ddxpart_dVd = vec8_blend(ddxpart_dVd, -dsxpart_dVd, condmask_false0);
        ddxpart_dVg = vec8_blend(ddxpart_dVg, -dsxpart_dVg, condmask_false0);
        ddxpart_dVs = vec8_blend(ddxpart_dVs, -dsxpart_dVs, condmask_false0);
        ddxpart_dVb = vec8_blend(ddxpart_dVb, -((ddxpart_dVd + ddxpart_dVg) + ddxpart_dVs), condmask_false0);
        qgd = vec8_blend(qgd, qgdo, condmask_false0);
        qgs = vec8_blend(qgs, qgso, condmask_false0);
        qgb = vec8_blend(qgb, pParam->BSIM3cgbo * vgb, condmask_false0);
        qgate = vec8_blend(qgate, (qgd + qgs) + qgb, condmask_false0);
        qbulk = vec8_blend(qbulk, -qgb, condmask_false0);
        qsrc = vec8_blend(qsrc, -qgs, condmask_false0);
        qdrn = vec8_blend(qdrn, -((qgate + qbulk) + qsrc), condmask_false0);
      }

    }
  }

  cqdef = (cqcheq = vec8_SIMDTOVECTOR(0.0));
  vec8_StateStore(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3qg, heres[1]->BSIM3qg, heres[2]->BSIM3qg, heres[3]->BSIM3qg, heres[4]->BSIM3qg, heres[5]->BSIM3qg, heres[6]->BSIM3qg, heres[7]->BSIM3qg}, qgate);
  vec8_StateStore(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3qd, heres[1]->BSIM3qd, heres[2]->BSIM3qd, heres[3]->BSIM3qd, heres[4]->BSIM3qd, heres[5]->BSIM3qd, heres[6]->BSIM3qd, heres[7]->BSIM3qd}, qdrn - vec8_StateAccess(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3qbd, heres[1]->BSIM3qbd, heres[2]->BSIM3qbd, heres[3]->BSIM3qbd, heres[4]->BSIM3qbd, heres[5]->BSIM3qbd, heres[6]->BSIM3qbd, heres[7]->BSIM3qbd}));
  vec8_StateStore(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3qb, heres[1]->BSIM3qb, heres[2]->BSIM3qb, heres[3]->BSIM3qb, heres[4]->BSIM3qb, heres[5]->BSIM3qb, heres[6]->BSIM3qb, heres[7]->BSIM3qb}, (qbulk + vec8_StateAccess(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3qbd, heres[1]->BSIM3qbd, heres[2]->BSIM3qbd, heres[3]->BSIM3qbd, heres[4]->BSIM3qbd, heres[5]->BSIM3qbd, heres[6]->BSIM3qbd, heres[7]->BSIM3qbd})) + vec8_StateAccess(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3qbs, heres[1]->BSIM3qbs, heres[2]->BSIM3qbs, heres[3]->BSIM3qbs, heres[4]->BSIM3qbs, heres[5]->BSIM3qbs, heres[6]->BSIM3qbs, heres[7]->BSIM3qbs}));
  if (heres[0]->BSIM3nqsMod)
  {
    vec8_StateStore(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3qcdump, heres[1]->BSIM3qcdump, heres[2]->BSIM3qcdump, heres[3]->BSIM3qcdump, heres[4]->BSIM3qcdump, heres[5]->BSIM3qcdump, heres[6]->BSIM3qcdump, heres[7]->BSIM3qcdump}, qdef * ScalingFactor);
    vec8_StateStore(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3qcheq, heres[1]->BSIM3qcheq, heres[2]->BSIM3qcheq, heres[3]->BSIM3qcheq, heres[4]->BSIM3qcheq, heres[5]->BSIM3qcheq, heres[6]->BSIM3qcheq, heres[7]->BSIM3qcheq}, qcheq);
  }

  if (ckt->CKTmode & MODEINITSMSIG)
  {
    goto line1000;
  }

  if (!ChargeComputationNeeded)
    goto line850;

  if (ckt->CKTmode & MODEINITTRAN)
  {
    vec8_StateStore(ckt->CKTstate1, (Vec8m ){heres[0]->BSIM3qb, heres[1]->BSIM3qb, heres[2]->BSIM3qb, heres[3]->BSIM3qb, heres[4]->BSIM3qb, heres[5]->BSIM3qb, heres[6]->BSIM3qb, heres[7]->BSIM3qb}, vec8_StateAccess(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3qb, heres[1]->BSIM3qb, heres[2]->BSIM3qb, heres[3]->BSIM3qb, heres[4]->BSIM3qb, heres[5]->BSIM3qb, heres[6]->BSIM3qb, heres[7]->BSIM3qb}));
    vec8_StateStore(ckt->CKTstate1, (Vec8m ){heres[0]->BSIM3qg, heres[1]->BSIM3qg, heres[2]->BSIM3qg, heres[3]->BSIM3qg, heres[4]->BSIM3qg, heres[5]->BSIM3qg, heres[6]->BSIM3qg, heres[7]->BSIM3qg}, vec8_StateAccess(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3qg, heres[1]->BSIM3qg, heres[2]->BSIM3qg, heres[3]->BSIM3qg, heres[4]->BSIM3qg, heres[5]->BSIM3qg, heres[6]->BSIM3qg, heres[7]->BSIM3qg}));
    vec8_StateStore(ckt->CKTstate1, (Vec8m ){heres[0]->BSIM3qd, heres[1]->BSIM3qd, heres[2]->BSIM3qd, heres[3]->BSIM3qd, heres[4]->BSIM3qd, heres[5]->BSIM3qd, heres[6]->BSIM3qd, heres[7]->BSIM3qd}, vec8_StateAccess(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3qd, heres[1]->BSIM3qd, heres[2]->BSIM3qd, heres[3]->BSIM3qd, heres[4]->BSIM3qd, heres[5]->BSIM3qd, heres[6]->BSIM3qd, heres[7]->BSIM3qd}));
    if (heres[0]->BSIM3nqsMod)
    {
      vec8_StateStore(ckt->CKTstate1, (Vec8m ){heres[0]->BSIM3qcheq, heres[1]->BSIM3qcheq, heres[2]->BSIM3qcheq, heres[3]->BSIM3qcheq, heres[4]->BSIM3qcheq, heres[5]->BSIM3qcheq, heres[6]->BSIM3qcheq, heres[7]->BSIM3qcheq}, vec8_StateAccess(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3qcheq, heres[1]->BSIM3qcheq, heres[2]->BSIM3qcheq, heres[3]->BSIM3qcheq, heres[4]->BSIM3qcheq, heres[5]->BSIM3qcheq, heres[6]->BSIM3qcheq, heres[7]->BSIM3qcheq}));
      vec8_StateStore(ckt->CKTstate1, (Vec8m ){heres[0]->BSIM3qcdump, heres[1]->BSIM3qcdump, heres[2]->BSIM3qcdump, heres[3]->BSIM3qcdump, heres[4]->BSIM3qcdump, heres[5]->BSIM3qcdump, heres[6]->BSIM3qcdump, heres[7]->BSIM3qcdump}, vec8_StateAccess(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3qcdump, heres[1]->BSIM3qcdump, heres[2]->BSIM3qcdump, heres[3]->BSIM3qcdump, heres[4]->BSIM3qcdump, heres[5]->BSIM3qcdump, heres[6]->BSIM3qcdump, heres[7]->BSIM3qcdump}));
    }

  }

  error = vec8_NIintegrate(ckt, &geq, &ceq, 0.0, (Vec8m ){heres[0]->BSIM3qb, heres[1]->BSIM3qb, heres[2]->BSIM3qb, heres[3]->BSIM3qb, heres[4]->BSIM3qb, heres[5]->BSIM3qb, heres[6]->BSIM3qb, heres[7]->BSIM3qb});
  if (SIMDANY(error))
    return error;

  error = vec8_NIintegrate(ckt, &geq, &ceq, 0.0, (Vec8m ){heres[0]->BSIM3qg, heres[1]->BSIM3qg, heres[2]->BSIM3qg, heres[3]->BSIM3qg, heres[4]->BSIM3qg, heres[5]->BSIM3qg, heres[6]->BSIM3qg, heres[7]->BSIM3qg});
  if (SIMDANY(error))
    return error;

  error = vec8_NIintegrate(ckt, &geq, &ceq, 0.0, (Vec8m ){heres[0]->BSIM3qd, heres[1]->BSIM3qd, heres[2]->BSIM3qd, heres[3]->BSIM3qd, heres[4]->BSIM3qd, heres[5]->BSIM3qd, heres[6]->BSIM3qd, heres[7]->BSIM3qd});
  if (SIMDANY(error))
    return error;

  if (heres[0]->BSIM3nqsMod)
  {
    error = vec8_NIintegrate(ckt, &geq, &ceq, 0.0, (Vec8m ){heres[0]->BSIM3qcdump, heres[1]->BSIM3qcdump, heres[2]->BSIM3qcdump, heres[3]->BSIM3qcdump, heres[4]->BSIM3qcdump, heres[5]->BSIM3qcdump, heres[6]->BSIM3qcdump, heres[7]->BSIM3qcdump});
    if (SIMDANY(error))
      return error;

    error = vec8_NIintegrate(ckt, &geq, &ceq, 0.0, (Vec8m ){heres[0]->BSIM3qcheq, heres[1]->BSIM3qcheq, heres[2]->BSIM3qcheq, heres[3]->BSIM3qcheq, heres[4]->BSIM3qcheq, heres[5]->BSIM3qcheq, heres[6]->BSIM3qcheq, heres[7]->BSIM3qcheq});
    if (SIMDANY(error))
      return error;

  }

  goto line860;
  line850:
  ceqqg = (ceqqb = (ceqqd = vec8_SIMDTOVECTOR(0.0)));

  cqcheq = (cqdef = vec8_SIMDTOVECTOR(0.0));
  gcdgb = (gcddb = (gcdsb = vec8_SIMDTOVECTOR(0.0)));
  gcsgb = (gcsdb = (gcssb = vec8_SIMDTOVECTOR(0.0)));
  gcggb = (gcgdb = (gcgsb = vec8_SIMDTOVECTOR(0.0)));
  gcbgb = (gcbdb = (gcbsb = vec8_SIMDTOVECTOR(0.0)));
  gqdef = (gcqgb = (gcqdb = (gcqsb = (gcqbb = vec8_SIMDTOVECTOR(0.0)))));
  ggtg = (ggtd = (ggtb = (ggts = vec8_SIMDTOVECTOR(0.0))));
  dxpart = vec8_SIMDTOVECTOR(0.6);
  if (1)
  {
    Vec8m condmask0 = BSIM3mode;
    Vec8m condmask_true0 = condmask0;
    dxpart = vec8_blend(dxpart, vec8_SIMDTOVECTOR(0.4), condmask_true0);
  }

  sxpart = 1.0 - dxpart;
  ddxpart_dVd = (ddxpart_dVg = (ddxpart_dVb = (ddxpart_dVs = vec8_SIMDTOVECTOR(0.0))));
  dsxpart_dVd = (dsxpart_dVg = (dsxpart_dVb = (dsxpart_dVs = vec8_SIMDTOVECTOR(0.0))));
  if (heres[0]->BSIM3nqsMod)
  {
    Vec8d val = ((((16.0 * ((Vec8d ){heres[0]->BSIM3u0temp, heres[1]->BSIM3u0temp, heres[2]->BSIM3u0temp, heres[3]->BSIM3u0temp, heres[4]->BSIM3u0temp, heres[5]->BSIM3u0temp, heres[6]->BSIM3u0temp, heres[7]->BSIM3u0temp})) * model->BSIM3vtm) / pParam->BSIM3leffCV) / pParam->BSIM3leffCV) * ScalingFactor;
    heres[0]->BSIM3gtau = val[0];
    heres[1]->BSIM3gtau = val[1];
    heres[2]->BSIM3gtau = val[2];
    heres[3]->BSIM3gtau = val[3];
    heres[4]->BSIM3gtau = val[4];
    heres[5]->BSIM3gtau = val[5];
    heres[6]->BSIM3gtau = val[6];
    heres[7]->BSIM3gtau = val[7];
  }
  else
  {
    heres[0]->BSIM3gtau = 0.0;
    heres[1]->BSIM3gtau = 0.0;
    heres[2]->BSIM3gtau = 0.0;
    heres[3]->BSIM3gtau = 0.0;
    heres[4]->BSIM3gtau = 0.0;
    heres[5]->BSIM3gtau = 0.0;
    heres[6]->BSIM3gtau = 0.0;
    heres[7]->BSIM3gtau = 0.0;
  }

  goto line900;
  line860:
  cqgate = vec8_StateAccess(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3cqg, heres[1]->BSIM3cqg, heres[2]->BSIM3cqg, heres[3]->BSIM3cqg, heres[4]->BSIM3cqg, heres[5]->BSIM3cqg, heres[6]->BSIM3cqg, heres[7]->BSIM3cqg});

  cqbulk = vec8_StateAccess(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3cqb, heres[1]->BSIM3cqb, heres[2]->BSIM3cqb, heres[3]->BSIM3cqb, heres[4]->BSIM3cqb, heres[5]->BSIM3cqb, heres[6]->BSIM3cqb, heres[7]->BSIM3cqb});
  cqdrn = vec8_StateAccess(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3cqd, heres[1]->BSIM3cqd, heres[2]->BSIM3cqd, heres[3]->BSIM3cqd, heres[4]->BSIM3cqd, heres[5]->BSIM3cqd, heres[6]->BSIM3cqd, heres[7]->BSIM3cqd});
  ceqqg = ((cqgate - (gcggb * vgb)) + (gcgdb * vbd)) + (gcgsb * vbs);
  ceqqb = ((cqbulk - (gcbgb * vgb)) + (gcbdb * vbd)) + (gcbsb * vbs);
  ceqqd = ((cqdrn - (gcdgb * vgb)) + (gcddb * vbd)) + (gcdsb * vbs);
  if (heres[0]->BSIM3nqsMod)
  {
    T0 = ((ggtg * vgb) - (ggtd * vbd)) - (ggts * vbs);
    ceqqg += T0;
    T1 = qdef * ((Vec8d ){heres[0]->BSIM3gtau, heres[1]->BSIM3gtau, heres[2]->BSIM3gtau, heres[3]->BSIM3gtau, heres[4]->BSIM3gtau, heres[5]->BSIM3gtau, heres[6]->BSIM3gtau, heres[7]->BSIM3gtau});
    ceqqd -= (dxpart * T0) + (T1 * (((ddxpart_dVg * vgb) - (ddxpart_dVd * vbd)) - (ddxpart_dVs * vbs)));
    cqdef = vec8_StateAccess(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3cqcdump, heres[1]->BSIM3cqcdump, heres[2]->BSIM3cqcdump, heres[3]->BSIM3cqcdump, heres[4]->BSIM3cqcdump, heres[5]->BSIM3cqcdump, heres[6]->BSIM3cqcdump, heres[7]->BSIM3cqcdump}) - (gqdef * qdef);
    cqcheq = (vec8_StateAccess(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3cqcheq, heres[1]->BSIM3cqcheq, heres[2]->BSIM3cqcheq, heres[3]->BSIM3cqcheq, heres[4]->BSIM3cqcheq, heres[5]->BSIM3cqcheq, heres[6]->BSIM3cqcheq, heres[7]->BSIM3cqcheq}) - (((gcqgb * vgb) - (gcqdb * vbd)) - (gcqsb * vbs))) + T0;
  }

  if (ckt->CKTmode & MODEINITTRAN)
  {
    vec8_StateStore(ckt->CKTstate1, (Vec8m ){heres[0]->BSIM3cqb, heres[1]->BSIM3cqb, heres[2]->BSIM3cqb, heres[3]->BSIM3cqb, heres[4]->BSIM3cqb, heres[5]->BSIM3cqb, heres[6]->BSIM3cqb, heres[7]->BSIM3cqb}, vec8_StateAccess(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3cqb, heres[1]->BSIM3cqb, heres[2]->BSIM3cqb, heres[3]->BSIM3cqb, heres[4]->BSIM3cqb, heres[5]->BSIM3cqb, heres[6]->BSIM3cqb, heres[7]->BSIM3cqb}));
    vec8_StateStore(ckt->CKTstate1, (Vec8m ){heres[0]->BSIM3cqg, heres[1]->BSIM3cqg, heres[2]->BSIM3cqg, heres[3]->BSIM3cqg, heres[4]->BSIM3cqg, heres[5]->BSIM3cqg, heres[6]->BSIM3cqg, heres[7]->BSIM3cqg}, vec8_StateAccess(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3cqg, heres[1]->BSIM3cqg, heres[2]->BSIM3cqg, heres[3]->BSIM3cqg, heres[4]->BSIM3cqg, heres[5]->BSIM3cqg, heres[6]->BSIM3cqg, heres[7]->BSIM3cqg}));
    vec8_StateStore(ckt->CKTstate1, (Vec8m ){heres[0]->BSIM3cqd, heres[1]->BSIM3cqd, heres[2]->BSIM3cqd, heres[3]->BSIM3cqd, heres[4]->BSIM3cqd, heres[5]->BSIM3cqd, heres[6]->BSIM3cqd, heres[7]->BSIM3cqd}, vec8_StateAccess(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3cqd, heres[1]->BSIM3cqd, heres[2]->BSIM3cqd, heres[3]->BSIM3cqd, heres[4]->BSIM3cqd, heres[5]->BSIM3cqd, heres[6]->BSIM3cqd, heres[7]->BSIM3cqd}));
    if (heres[0]->BSIM3nqsMod)
    {
      vec8_StateStore(ckt->CKTstate1, (Vec8m ){heres[0]->BSIM3cqcheq, heres[1]->BSIM3cqcheq, heres[2]->BSIM3cqcheq, heres[3]->BSIM3cqcheq, heres[4]->BSIM3cqcheq, heres[5]->BSIM3cqcheq, heres[6]->BSIM3cqcheq, heres[7]->BSIM3cqcheq}, vec8_StateAccess(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3cqcheq, heres[1]->BSIM3cqcheq, heres[2]->BSIM3cqcheq, heres[3]->BSIM3cqcheq, heres[4]->BSIM3cqcheq, heres[5]->BSIM3cqcheq, heres[6]->BSIM3cqcheq, heres[7]->BSIM3cqcheq}));
      vec8_StateStore(ckt->CKTstate1, (Vec8m ){heres[0]->BSIM3cqcdump, heres[1]->BSIM3cqcdump, heres[2]->BSIM3cqcdump, heres[3]->BSIM3cqcdump, heres[4]->BSIM3cqcdump, heres[5]->BSIM3cqcdump, heres[6]->BSIM3cqcdump, heres[7]->BSIM3cqcdump}, vec8_StateAccess(ckt->CKTstate0, (Vec8m ){heres[0]->BSIM3cqcdump, heres[1]->BSIM3cqcdump, heres[2]->BSIM3cqcdump, heres[3]->BSIM3cqcdump, heres[4]->BSIM3cqcdump, heres[5]->BSIM3cqcdump, heres[6]->BSIM3cqcdump, heres[7]->BSIM3cqcdump}));
    }

  }

  line900:
  if (1)
  {
    Vec8m condmask0 = BSIM3mode;
    Vec8m condmask_true0 = condmask0;
    Vec8m condmask_false0 = ~condmask0;
    {
      Gm = vec8_blend(Gm, (Vec8d ){heres[0]->BSIM3gm, heres[1]->BSIM3gm, heres[2]->BSIM3gm, heres[3]->BSIM3gm, heres[4]->BSIM3gm, heres[5]->BSIM3gm, heres[6]->BSIM3gm, heres[7]->BSIM3gm}, condmask_true0);
      Gmbs = vec8_blend(Gmbs, (Vec8d ){heres[0]->BSIM3gmbs, heres[1]->BSIM3gmbs, heres[2]->BSIM3gmbs, heres[3]->BSIM3gmbs, heres[4]->BSIM3gmbs, heres[5]->BSIM3gmbs, heres[6]->BSIM3gmbs, heres[7]->BSIM3gmbs}, condmask_true0);
      FwdSum = vec8_blend(FwdSum, Gm + Gmbs, condmask_true0);
      RevSum = vec8_blend(RevSum, vec8_SIMDTOVECTOR(0.0), condmask_true0);
      cdreq = vec8_blend(cdreq, model->BSIM3type * (((cdrain - (((Vec8d ){heres[0]->BSIM3gds, heres[1]->BSIM3gds, heres[2]->BSIM3gds, heres[3]->BSIM3gds, heres[4]->BSIM3gds, heres[5]->BSIM3gds, heres[6]->BSIM3gds, heres[7]->BSIM3gds}) * vds)) - (Gm * vgs)) - (Gmbs * vbs)), condmask_true0);
      ceqbd = vec8_blend(ceqbd, (-model->BSIM3type) * (((((Vec8d ){heres[0]->BSIM3csub, heres[1]->BSIM3csub, heres[2]->BSIM3csub, heres[3]->BSIM3csub, heres[4]->BSIM3csub, heres[5]->BSIM3csub, heres[6]->BSIM3csub, heres[7]->BSIM3csub}) - (((Vec8d ){heres[0]->BSIM3gbds, heres[1]->BSIM3gbds, heres[2]->BSIM3gbds, heres[3]->BSIM3gbds, heres[4]->BSIM3gbds, heres[5]->BSIM3gbds, heres[6]->BSIM3gbds, heres[7]->BSIM3gbds}) * vds)) - (((Vec8d ){heres[0]->BSIM3gbgs, heres[1]->BSIM3gbgs, heres[2]->BSIM3gbgs, heres[3]->BSIM3gbgs, heres[4]->BSIM3gbgs, heres[5]->BSIM3gbgs, heres[6]->BSIM3gbgs, heres[7]->BSIM3gbgs}) * vgs)) - (((Vec8d ){heres[0]->BSIM3gbbs, heres[1]->BSIM3gbbs, heres[2]->BSIM3gbbs, heres[3]->BSIM3gbbs, heres[4]->BSIM3gbbs, heres[5]->BSIM3gbbs, heres[6]->BSIM3gbbs, heres[7]->BSIM3gbbs}) * vbs)), condmask_true0);
      ceqbs = vec8_blend(ceqbs, vec8_SIMDTOVECTOR(0.0), condmask_true0);
      gbbdp = vec8_blend(gbbdp, -((Vec8d ){heres[0]->BSIM3gbds, heres[1]->BSIM3gbds, heres[2]->BSIM3gbds, heres[3]->BSIM3gbds, heres[4]->BSIM3gbds, heres[5]->BSIM3gbds, heres[6]->BSIM3gbds, heres[7]->BSIM3gbds}), condmask_true0);
      gbbsp = vec8_blend(gbbsp, (((Vec8d ){heres[0]->BSIM3gbds, heres[1]->BSIM3gbds, heres[2]->BSIM3gbds, heres[3]->BSIM3gbds, heres[4]->BSIM3gbds, heres[5]->BSIM3gbds, heres[6]->BSIM3gbds, heres[7]->BSIM3gbds}) + ((Vec8d ){heres[0]->BSIM3gbgs, heres[1]->BSIM3gbgs, heres[2]->BSIM3gbgs, heres[3]->BSIM3gbgs, heres[4]->BSIM3gbgs, heres[5]->BSIM3gbgs, heres[6]->BSIM3gbgs, heres[7]->BSIM3gbgs})) + ((Vec8d ){heres[0]->BSIM3gbbs, heres[1]->BSIM3gbbs, heres[2]->BSIM3gbbs, heres[3]->BSIM3gbbs, heres[4]->BSIM3gbbs, heres[5]->BSIM3gbbs, heres[6]->BSIM3gbbs, heres[7]->BSIM3gbbs}), condmask_true0);
      gbdpg = vec8_blend(gbdpg, (Vec8d ){heres[0]->BSIM3gbgs, heres[1]->BSIM3gbgs, heres[2]->BSIM3gbgs, heres[3]->BSIM3gbgs, heres[4]->BSIM3gbgs, heres[5]->BSIM3gbgs, heres[6]->BSIM3gbgs, heres[7]->BSIM3gbgs}, condmask_true0);
      gbdpdp = vec8_blend(gbdpdp, (Vec8d ){heres[0]->BSIM3gbds, heres[1]->BSIM3gbds, heres[2]->BSIM3gbds, heres[3]->BSIM3gbds, heres[4]->BSIM3gbds, heres[5]->BSIM3gbds, heres[6]->BSIM3gbds, heres[7]->BSIM3gbds}, condmask_true0);
      gbdpb = vec8_blend(gbdpb, (Vec8d ){heres[0]->BSIM3gbbs, heres[1]->BSIM3gbbs, heres[2]->BSIM3gbbs, heres[3]->BSIM3gbbs, heres[4]->BSIM3gbbs, heres[5]->BSIM3gbbs, heres[6]->BSIM3gbbs, heres[7]->BSIM3gbbs}, condmask_true0);
      gbdpsp = vec8_blend(gbdpsp, -((gbdpg + gbdpdp) + gbdpb), condmask_true0);
      gbspg = vec8_blend(gbspg, vec8_SIMDTOVECTOR(0.0), condmask_true0);
      gbspdp = vec8_blend(gbspdp, vec8_SIMDTOVECTOR(0.0), condmask_true0);
      gbspb = vec8_blend(gbspb, vec8_SIMDTOVECTOR(0.0), condmask_true0);
      gbspsp = vec8_blend(gbspsp, vec8_SIMDTOVECTOR(0.0), condmask_true0);
    }
    {
      Gm = vec8_blend(Gm, -((Vec8d ){heres[0]->BSIM3gm, heres[1]->BSIM3gm, heres[2]->BSIM3gm, heres[3]->BSIM3gm, heres[4]->BSIM3gm, heres[5]->BSIM3gm, heres[6]->BSIM3gm, heres[7]->BSIM3gm}), condmask_false0);
      Gmbs = vec8_blend(Gmbs, -((Vec8d ){heres[0]->BSIM3gmbs, heres[1]->BSIM3gmbs, heres[2]->BSIM3gmbs, heres[3]->BSIM3gmbs, heres[4]->BSIM3gmbs, heres[5]->BSIM3gmbs, heres[6]->BSIM3gmbs, heres[7]->BSIM3gmbs}), condmask_false0);
      FwdSum = vec8_blend(FwdSum, vec8_SIMDTOVECTOR(0.0), condmask_false0);
      RevSum = vec8_blend(RevSum, -(Gm + Gmbs), condmask_false0);
      cdreq = vec8_blend(cdreq, (-model->BSIM3type) * (((cdrain + (((Vec8d ){heres[0]->BSIM3gds, heres[1]->BSIM3gds, heres[2]->BSIM3gds, heres[3]->BSIM3gds, heres[4]->BSIM3gds, heres[5]->BSIM3gds, heres[6]->BSIM3gds, heres[7]->BSIM3gds}) * vds)) + (Gm * vgd)) + (Gmbs * vbd)), condmask_false0);
      ceqbs = vec8_blend(ceqbs, (-model->BSIM3type) * (((((Vec8d ){heres[0]->BSIM3csub, heres[1]->BSIM3csub, heres[2]->BSIM3csub, heres[3]->BSIM3csub, heres[4]->BSIM3csub, heres[5]->BSIM3csub, heres[6]->BSIM3csub, heres[7]->BSIM3csub}) + (((Vec8d ){heres[0]->BSIM3gbds, heres[1]->BSIM3gbds, heres[2]->BSIM3gbds, heres[3]->BSIM3gbds, heres[4]->BSIM3gbds, heres[5]->BSIM3gbds, heres[6]->BSIM3gbds, heres[7]->BSIM3gbds}) * vds)) - (((Vec8d ){heres[0]->BSIM3gbgs, heres[1]->BSIM3gbgs, heres[2]->BSIM3gbgs, heres[3]->BSIM3gbgs, heres[4]->BSIM3gbgs, heres[5]->BSIM3gbgs, heres[6]->BSIM3gbgs, heres[7]->BSIM3gbgs}) * vgd)) - (((Vec8d ){heres[0]->BSIM3gbbs, heres[1]->BSIM3gbbs, heres[2]->BSIM3gbbs, heres[3]->BSIM3gbbs, heres[4]->BSIM3gbbs, heres[5]->BSIM3gbbs, heres[6]->BSIM3gbbs, heres[7]->BSIM3gbbs}) * vbd)), condmask_false0);
      ceqbd = vec8_blend(ceqbd, vec8_SIMDTOVECTOR(0.0), condmask_false0);
      gbbsp = vec8_blend(gbbsp, -((Vec8d ){heres[0]->BSIM3gbds, heres[1]->BSIM3gbds, heres[2]->BSIM3gbds, heres[3]->BSIM3gbds, heres[4]->BSIM3gbds, heres[5]->BSIM3gbds, heres[6]->BSIM3gbds, heres[7]->BSIM3gbds}), condmask_false0);
      gbbdp = vec8_blend(gbbdp, (((Vec8d ){heres[0]->BSIM3gbds, heres[1]->BSIM3gbds, heres[2]->BSIM3gbds, heres[3]->BSIM3gbds, heres[4]->BSIM3gbds, heres[5]->BSIM3gbds, heres[6]->BSIM3gbds, heres[7]->BSIM3gbds}) + ((Vec8d ){heres[0]->BSIM3gbgs, heres[1]->BSIM3gbgs, heres[2]->BSIM3gbgs, heres[3]->BSIM3gbgs, heres[4]->BSIM3gbgs, heres[5]->BSIM3gbgs, heres[6]->BSIM3gbgs, heres[7]->BSIM3gbgs})) + ((Vec8d ){heres[0]->BSIM3gbbs, heres[1]->BSIM3gbbs, heres[2]->BSIM3gbbs, heres[3]->BSIM3gbbs, heres[4]->BSIM3gbbs, heres[5]->BSIM3gbbs, heres[6]->BSIM3gbbs, heres[7]->BSIM3gbbs}), condmask_false0);
      gbdpg = vec8_blend(gbdpg, vec8_SIMDTOVECTOR(0.0), condmask_false0);
      gbdpsp = vec8_blend(gbdpsp, vec8_SIMDTOVECTOR(0.0), condmask_false0);
      gbdpb = vec8_blend(gbdpb, vec8_SIMDTOVECTOR(0.0), condmask_false0);
      gbdpdp = vec8_blend(gbdpdp, vec8_SIMDTOVECTOR(0.0), condmask_false0);
      gbspg = vec8_blend(gbspg, (Vec8d ){heres[0]->BSIM3gbgs, heres[1]->BSIM3gbgs, heres[2]->BSIM3gbgs, heres[3]->BSIM3gbgs, heres[4]->BSIM3gbgs, heres[5]->BSIM3gbgs, heres[6]->BSIM3gbgs, heres[7]->BSIM3gbgs}, condmask_false0);
      gbspsp = vec8_blend(gbspsp, (Vec8d ){heres[0]->BSIM3gbds, heres[1]->BSIM3gbds, heres[2]->BSIM3gbds, heres[3]->BSIM3gbds, heres[4]->BSIM3gbds, heres[5]->BSIM3gbds, heres[6]->BSIM3gbds, heres[7]->BSIM3gbds}, condmask_false0);
      gbspb = vec8_blend(gbspb, (Vec8d ){heres[0]->BSIM3gbbs, heres[1]->BSIM3gbbs, heres[2]->BSIM3gbbs, heres[3]->BSIM3gbbs, heres[4]->BSIM3gbbs, heres[5]->BSIM3gbbs, heres[6]->BSIM3gbbs, heres[7]->BSIM3gbbs}, condmask_false0);
      gbspdp = vec8_blend(gbspdp, -((gbspg + gbspsp) + gbspb), condmask_false0);
    }
  }


  if (model->BSIM3type > 0)
  {
    ceqbs += ((Vec8d ){heres[0]->BSIM3cbs, heres[1]->BSIM3cbs, heres[2]->BSIM3cbs, heres[3]->BSIM3cbs, heres[4]->BSIM3cbs, heres[5]->BSIM3cbs, heres[6]->BSIM3cbs, heres[7]->BSIM3cbs}) - (((Vec8d ){heres[0]->BSIM3gbs, heres[1]->BSIM3gbs, heres[2]->BSIM3gbs, heres[3]->BSIM3gbs, heres[4]->BSIM3gbs, heres[5]->BSIM3gbs, heres[6]->BSIM3gbs, heres[7]->BSIM3gbs}) * vbs);
    ceqbd += ((Vec8d ){heres[0]->BSIM3cbd, heres[1]->BSIM3cbd, heres[2]->BSIM3cbd, heres[3]->BSIM3cbd, heres[4]->BSIM3cbd, heres[5]->BSIM3cbd, heres[6]->BSIM3cbd, heres[7]->BSIM3cbd}) - (((Vec8d ){heres[0]->BSIM3gbd, heres[1]->BSIM3gbd, heres[2]->BSIM3gbd, heres[3]->BSIM3gbd, heres[4]->BSIM3gbd, heres[5]->BSIM3gbd, heres[6]->BSIM3gbd, heres[7]->BSIM3gbd}) * vbd);
  }
  else
  {
    ceqbs -= ((Vec8d ){heres[0]->BSIM3cbs, heres[1]->BSIM3cbs, heres[2]->BSIM3cbs, heres[3]->BSIM3cbs, heres[4]->BSIM3cbs, heres[5]->BSIM3cbs, heres[6]->BSIM3cbs, heres[7]->BSIM3cbs}) - (((Vec8d ){heres[0]->BSIM3gbs, heres[1]->BSIM3gbs, heres[2]->BSIM3gbs, heres[3]->BSIM3gbs, heres[4]->BSIM3gbs, heres[5]->BSIM3gbs, heres[6]->BSIM3gbs, heres[7]->BSIM3gbs}) * vbs);
    ceqbd -= ((Vec8d ){heres[0]->BSIM3cbd, heres[1]->BSIM3cbd, heres[2]->BSIM3cbd, heres[3]->BSIM3cbd, heres[4]->BSIM3cbd, heres[5]->BSIM3cbd, heres[6]->BSIM3cbd, heres[7]->BSIM3cbd}) - (((Vec8d ){heres[0]->BSIM3gbd, heres[1]->BSIM3gbd, heres[2]->BSIM3gbd, heres[3]->BSIM3gbd, heres[4]->BSIM3gbd, heres[5]->BSIM3gbd, heres[6]->BSIM3gbd, heres[7]->BSIM3gbd}) * vbd);
    ceqqg = -ceqqg;
    ceqqb = -ceqqb;
    ceqqd = -ceqqd;
    cqdef = -cqdef;
    cqcheq = -cqcheq;
  }

  m = (Vec8d ){heres[0]->BSIM3m, heres[1]->BSIM3m, heres[2]->BSIM3m, heres[3]->BSIM3m, heres[4]->BSIM3m, heres[5]->BSIM3m, heres[6]->BSIM3m, heres[7]->BSIM3m};
  {
    Vec8d val = m * ceqqg;
    heres[0]->BSIM3rhsG = val[0];
    heres[1]->BSIM3rhsG = val[1];
    heres[2]->BSIM3rhsG = val[2];
    heres[3]->BSIM3rhsG = val[3];
    heres[4]->BSIM3rhsG = val[4];
    heres[5]->BSIM3rhsG = val[5];
    heres[6]->BSIM3rhsG = val[6];
    heres[7]->BSIM3rhsG = val[7];
  }
  {
    Vec8d val = m * ((ceqbs + ceqbd) + ceqqb);
    heres[0]->BSIM3rhsB = val[0];
    heres[1]->BSIM3rhsB = val[1];
    heres[2]->BSIM3rhsB = val[2];
    heres[3]->BSIM3rhsB = val[3];
    heres[4]->BSIM3rhsB = val[4];
    heres[5]->BSIM3rhsB = val[5];
    heres[6]->BSIM3rhsB = val[6];
    heres[7]->BSIM3rhsB = val[7];
  }
  {
    Vec8d val = m * ((ceqbd - cdreq) - ceqqd);
    heres[0]->BSIM3rhsD = val[0];
    heres[1]->BSIM3rhsD = val[1];
    heres[2]->BSIM3rhsD = val[2];
    heres[3]->BSIM3rhsD = val[3];
    heres[4]->BSIM3rhsD = val[4];
    heres[5]->BSIM3rhsD = val[5];
    heres[6]->BSIM3rhsD = val[6];
    heres[7]->BSIM3rhsD = val[7];
  }
  {
    Vec8d val = m * ((((cdreq + ceqbs) + ceqqg) + ceqqb) + ceqqd);
    heres[0]->BSIM3rhsS = val[0];
    heres[1]->BSIM3rhsS = val[1];
    heres[2]->BSIM3rhsS = val[2];
    heres[3]->BSIM3rhsS = val[3];
    heres[4]->BSIM3rhsS = val[4];
    heres[5]->BSIM3rhsS = val[5];
    heres[6]->BSIM3rhsS = val[6];
    heres[7]->BSIM3rhsS = val[7];
  }
  if (heres[0]->BSIM3nqsMod)
    vec8_StateAdd(ckt->CKTrhs, (Vec8m ){heres[0]->BSIM3qNode, heres[1]->BSIM3qNode, heres[2]->BSIM3qNode, heres[3]->BSIM3qNode, heres[4]->BSIM3qNode, heres[5]->BSIM3qNode, heres[6]->BSIM3qNode, heres[7]->BSIM3qNode}, m * (cqcheq - cqdef));

  T1 = qdef * ((Vec8d ){heres[0]->BSIM3gtau, heres[1]->BSIM3gtau, heres[2]->BSIM3gtau, heres[3]->BSIM3gtau, heres[4]->BSIM3gtau, heres[5]->BSIM3gtau, heres[6]->BSIM3gtau, heres[7]->BSIM3gtau});
  {
    Vec8d val = m * ((Vec8d ){heres[0]->BSIM3drainConductance, heres[1]->BSIM3drainConductance, heres[2]->BSIM3drainConductance, heres[3]->BSIM3drainConductance, heres[4]->BSIM3drainConductance, heres[5]->BSIM3drainConductance, heres[6]->BSIM3drainConductance, heres[7]->BSIM3drainConductance});
    heres[0]->BSIM3DdPt = val[0];
    heres[1]->BSIM3DdPt = val[1];
    heres[2]->BSIM3DdPt = val[2];
    heres[3]->BSIM3DdPt = val[3];
    heres[4]->BSIM3DdPt = val[4];
    heres[5]->BSIM3DdPt = val[5];
    heres[6]->BSIM3DdPt = val[6];
    heres[7]->BSIM3DdPt = val[7];
  }
  {
    Vec8d val = m * (gcggb - ggtg);
    heres[0]->BSIM3GgPt = val[0];
    heres[1]->BSIM3GgPt = val[1];
    heres[2]->BSIM3GgPt = val[2];
    heres[3]->BSIM3GgPt = val[3];
    heres[4]->BSIM3GgPt = val[4];
    heres[5]->BSIM3GgPt = val[5];
    heres[6]->BSIM3GgPt = val[6];
    heres[7]->BSIM3GgPt = val[7];
  }
  {
    Vec8d val = m * ((Vec8d ){heres[0]->BSIM3sourceConductance, heres[1]->BSIM3sourceConductance, heres[2]->BSIM3sourceConductance, heres[3]->BSIM3sourceConductance, heres[4]->BSIM3sourceConductance, heres[5]->BSIM3sourceConductance, heres[6]->BSIM3sourceConductance, heres[7]->BSIM3sourceConductance});
    heres[0]->BSIM3SsPt = val[0];
    heres[1]->BSIM3SsPt = val[1];
    heres[2]->BSIM3SsPt = val[2];
    heres[3]->BSIM3SsPt = val[3];
    heres[4]->BSIM3SsPt = val[4];
    heres[5]->BSIM3SsPt = val[5];
    heres[6]->BSIM3SsPt = val[6];
    heres[7]->BSIM3SsPt = val[7];
  }
  {
    Vec8d val = m * (((((((Vec8d ){heres[0]->BSIM3gbd, heres[1]->BSIM3gbd, heres[2]->BSIM3gbd, heres[3]->BSIM3gbd, heres[4]->BSIM3gbd, heres[5]->BSIM3gbd, heres[6]->BSIM3gbd, heres[7]->BSIM3gbd}) + ((Vec8d ){heres[0]->BSIM3gbs, heres[1]->BSIM3gbs, heres[2]->BSIM3gbs, heres[3]->BSIM3gbs, heres[4]->BSIM3gbs, heres[5]->BSIM3gbs, heres[6]->BSIM3gbs, heres[7]->BSIM3gbs})) - gcbgb) - gcbdb) - gcbsb) - ((Vec8d ){heres[0]->BSIM3gbbs, heres[1]->BSIM3gbbs, heres[2]->BSIM3gbbs, heres[3]->BSIM3gbbs, heres[4]->BSIM3gbbs, heres[5]->BSIM3gbbs, heres[6]->BSIM3gbbs, heres[7]->BSIM3gbbs}));
    heres[0]->BSIM3BbPt = val[0];
    heres[1]->BSIM3BbPt = val[1];
    heres[2]->BSIM3BbPt = val[2];
    heres[3]->BSIM3BbPt = val[3];
    heres[4]->BSIM3BbPt = val[4];
    heres[5]->BSIM3BbPt = val[5];
    heres[6]->BSIM3BbPt = val[6];
    heres[7]->BSIM3BbPt = val[7];
  }
  {
    Vec8d val = m * (((((((((Vec8d ){heres[0]->BSIM3drainConductance, heres[1]->BSIM3drainConductance, heres[2]->BSIM3drainConductance, heres[3]->BSIM3drainConductance, heres[4]->BSIM3drainConductance, heres[5]->BSIM3drainConductance, heres[6]->BSIM3drainConductance, heres[7]->BSIM3drainConductance}) + ((Vec8d ){heres[0]->BSIM3gds, heres[1]->BSIM3gds, heres[2]->BSIM3gds, heres[3]->BSIM3gds, heres[4]->BSIM3gds, heres[5]->BSIM3gds, heres[6]->BSIM3gds, heres[7]->BSIM3gds})) + ((Vec8d ){heres[0]->BSIM3gbd, heres[1]->BSIM3gbd, heres[2]->BSIM3gbd, heres[3]->BSIM3gbd, heres[4]->BSIM3gbd, heres[5]->BSIM3gbd, heres[6]->BSIM3gbd, heres[7]->BSIM3gbd})) + RevSum) + gcddb) + (dxpart * ggtd)) + (T1 * ddxpart_dVd)) + gbdpdp);
    heres[0]->BSIM3DPdpPt = val[0];
    heres[1]->BSIM3DPdpPt = val[1];
    heres[2]->BSIM3DPdpPt = val[2];
    heres[3]->BSIM3DPdpPt = val[3];
    heres[4]->BSIM3DPdpPt = val[4];
    heres[5]->BSIM3DPdpPt = val[5];
    heres[6]->BSIM3DPdpPt = val[6];
    heres[7]->BSIM3DPdpPt = val[7];
  }
  {
    Vec8d val = m * (((((((((Vec8d ){heres[0]->BSIM3sourceConductance, heres[1]->BSIM3sourceConductance, heres[2]->BSIM3sourceConductance, heres[3]->BSIM3sourceConductance, heres[4]->BSIM3sourceConductance, heres[5]->BSIM3sourceConductance, heres[6]->BSIM3sourceConductance, heres[7]->BSIM3sourceConductance}) + ((Vec8d ){heres[0]->BSIM3gds, heres[1]->BSIM3gds, heres[2]->BSIM3gds, heres[3]->BSIM3gds, heres[4]->BSIM3gds, heres[5]->BSIM3gds, heres[6]->BSIM3gds, heres[7]->BSIM3gds})) + ((Vec8d ){heres[0]->BSIM3gbs, heres[1]->BSIM3gbs, heres[2]->BSIM3gbs, heres[3]->BSIM3gbs, heres[4]->BSIM3gbs, heres[5]->BSIM3gbs, heres[6]->BSIM3gbs, heres[7]->BSIM3gbs})) + FwdSum) + gcssb) + (sxpart * ggts)) + (T1 * dsxpart_dVs)) + gbspsp);
    heres[0]->BSIM3SPspPt = val[0];
    heres[1]->BSIM3SPspPt = val[1];
    heres[2]->BSIM3SPspPt = val[2];
    heres[3]->BSIM3SPspPt = val[3];
    heres[4]->BSIM3SPspPt = val[4];
    heres[5]->BSIM3SPspPt = val[5];
    heres[6]->BSIM3SPspPt = val[6];
    heres[7]->BSIM3SPspPt = val[7];
  }
  {
    Vec8d val = m * ((Vec8d ){heres[0]->BSIM3drainConductance, heres[1]->BSIM3drainConductance, heres[2]->BSIM3drainConductance, heres[3]->BSIM3drainConductance, heres[4]->BSIM3drainConductance, heres[5]->BSIM3drainConductance, heres[6]->BSIM3drainConductance, heres[7]->BSIM3drainConductance});
    heres[0]->BSIM3DdpPt = val[0];
    heres[1]->BSIM3DdpPt = val[1];
    heres[2]->BSIM3DdpPt = val[2];
    heres[3]->BSIM3DdpPt = val[3];
    heres[4]->BSIM3DdpPt = val[4];
    heres[5]->BSIM3DdpPt = val[5];
    heres[6]->BSIM3DdpPt = val[6];
    heres[7]->BSIM3DdpPt = val[7];
  }
  {
    Vec8d val = m * (((gcggb + gcgdb) + gcgsb) + ggtb);
    heres[0]->BSIM3GbPt = val[0];
    heres[1]->BSIM3GbPt = val[1];
    heres[2]->BSIM3GbPt = val[2];
    heres[3]->BSIM3GbPt = val[3];
    heres[4]->BSIM3GbPt = val[4];
    heres[5]->BSIM3GbPt = val[5];
    heres[6]->BSIM3GbPt = val[6];
    heres[7]->BSIM3GbPt = val[7];
  }
  {
    Vec8d val = m * (gcgdb - ggtd);
    heres[0]->BSIM3GdpPt = val[0];
    heres[1]->BSIM3GdpPt = val[1];
    heres[2]->BSIM3GdpPt = val[2];
    heres[3]->BSIM3GdpPt = val[3];
    heres[4]->BSIM3GdpPt = val[4];
    heres[5]->BSIM3GdpPt = val[5];
    heres[6]->BSIM3GdpPt = val[6];
    heres[7]->BSIM3GdpPt = val[7];
  }
  {
    Vec8d val = m * (gcgsb - ggts);
    heres[0]->BSIM3GspPt = val[0];
    heres[1]->BSIM3GspPt = val[1];
    heres[2]->BSIM3GspPt = val[2];
    heres[3]->BSIM3GspPt = val[3];
    heres[4]->BSIM3GspPt = val[4];
    heres[5]->BSIM3GspPt = val[5];
    heres[6]->BSIM3GspPt = val[6];
    heres[7]->BSIM3GspPt = val[7];
  }
  {
    Vec8d val = m * ((Vec8d ){heres[0]->BSIM3sourceConductance, heres[1]->BSIM3sourceConductance, heres[2]->BSIM3sourceConductance, heres[3]->BSIM3sourceConductance, heres[4]->BSIM3sourceConductance, heres[5]->BSIM3sourceConductance, heres[6]->BSIM3sourceConductance, heres[7]->BSIM3sourceConductance});
    heres[0]->BSIM3SspPt = val[0];
    heres[1]->BSIM3SspPt = val[1];
    heres[2]->BSIM3SspPt = val[2];
    heres[3]->BSIM3SspPt = val[3];
    heres[4]->BSIM3SspPt = val[4];
    heres[5]->BSIM3SspPt = val[5];
    heres[6]->BSIM3SspPt = val[6];
    heres[7]->BSIM3SspPt = val[7];
  }
  {
    Vec8d val = m * (gcbgb - ((Vec8d ){heres[0]->BSIM3gbgs, heres[1]->BSIM3gbgs, heres[2]->BSIM3gbgs, heres[3]->BSIM3gbgs, heres[4]->BSIM3gbgs, heres[5]->BSIM3gbgs, heres[6]->BSIM3gbgs, heres[7]->BSIM3gbgs}));
    heres[0]->BSIM3BgPt = val[0];
    heres[1]->BSIM3BgPt = val[1];
    heres[2]->BSIM3BgPt = val[2];
    heres[3]->BSIM3BgPt = val[3];
    heres[4]->BSIM3BgPt = val[4];
    heres[5]->BSIM3BgPt = val[5];
    heres[6]->BSIM3BgPt = val[6];
    heres[7]->BSIM3BgPt = val[7];
  }
  {
    Vec8d val = m * ((gcbdb - ((Vec8d ){heres[0]->BSIM3gbd, heres[1]->BSIM3gbd, heres[2]->BSIM3gbd, heres[3]->BSIM3gbd, heres[4]->BSIM3gbd, heres[5]->BSIM3gbd, heres[6]->BSIM3gbd, heres[7]->BSIM3gbd})) + gbbdp);
    heres[0]->BSIM3BdpPt = val[0];
    heres[1]->BSIM3BdpPt = val[1];
    heres[2]->BSIM3BdpPt = val[2];
    heres[3]->BSIM3BdpPt = val[3];
    heres[4]->BSIM3BdpPt = val[4];
    heres[5]->BSIM3BdpPt = val[5];
    heres[6]->BSIM3BdpPt = val[6];
    heres[7]->BSIM3BdpPt = val[7];
  }
  {
    Vec8d val = m * ((gcbsb - ((Vec8d ){heres[0]->BSIM3gbs, heres[1]->BSIM3gbs, heres[2]->BSIM3gbs, heres[3]->BSIM3gbs, heres[4]->BSIM3gbs, heres[5]->BSIM3gbs, heres[6]->BSIM3gbs, heres[7]->BSIM3gbs})) + gbbsp);
    heres[0]->BSIM3BspPt = val[0];
    heres[1]->BSIM3BspPt = val[1];
    heres[2]->BSIM3BspPt = val[2];
    heres[3]->BSIM3BspPt = val[3];
    heres[4]->BSIM3BspPt = val[4];
    heres[5]->BSIM3BspPt = val[5];
    heres[6]->BSIM3BspPt = val[6];
    heres[7]->BSIM3BspPt = val[7];
  }
  {
    Vec8d val = m * ((Vec8d ){heres[0]->BSIM3drainConductance, heres[1]->BSIM3drainConductance, heres[2]->BSIM3drainConductance, heres[3]->BSIM3drainConductance, heres[4]->BSIM3drainConductance, heres[5]->BSIM3drainConductance, heres[6]->BSIM3drainConductance, heres[7]->BSIM3drainConductance});
    heres[0]->BSIM3DPdPt = val[0];
    heres[1]->BSIM3DPdPt = val[1];
    heres[2]->BSIM3DPdPt = val[2];
    heres[3]->BSIM3DPdPt = val[3];
    heres[4]->BSIM3DPdPt = val[4];
    heres[5]->BSIM3DPdPt = val[5];
    heres[6]->BSIM3DPdPt = val[6];
    heres[7]->BSIM3DPdPt = val[7];
  }
  {
    Vec8d val = m * ((((Gm + gcdgb) + (dxpart * ggtg)) + (T1 * ddxpart_dVg)) + gbdpg);
    heres[0]->BSIM3DPgPt = val[0];
    heres[1]->BSIM3DPgPt = val[1];
    heres[2]->BSIM3DPgPt = val[2];
    heres[3]->BSIM3DPgPt = val[3];
    heres[4]->BSIM3DPgPt = val[4];
    heres[5]->BSIM3DPgPt = val[5];
    heres[6]->BSIM3DPgPt = val[6];
    heres[7]->BSIM3DPgPt = val[7];
  }
  {
    Vec8d val = m * (((((((((Vec8d ){heres[0]->BSIM3gbd, heres[1]->BSIM3gbd, heres[2]->BSIM3gbd, heres[3]->BSIM3gbd, heres[4]->BSIM3gbd, heres[5]->BSIM3gbd, heres[6]->BSIM3gbd, heres[7]->BSIM3gbd}) - Gmbs) + gcdgb) + gcddb) + gcdsb) - (dxpart * ggtb)) - (T1 * ddxpart_dVb)) - gbdpb);
    heres[0]->BSIM3DPbPt = val[0];
    heres[1]->BSIM3DPbPt = val[1];
    heres[2]->BSIM3DPbPt = val[2];
    heres[3]->BSIM3DPbPt = val[3];
    heres[4]->BSIM3DPbPt = val[4];
    heres[5]->BSIM3DPbPt = val[5];
    heres[6]->BSIM3DPbPt = val[6];
    heres[7]->BSIM3DPbPt = val[7];
  }
  {
    Vec8d val = m * (((((((Vec8d ){heres[0]->BSIM3gds, heres[1]->BSIM3gds, heres[2]->BSIM3gds, heres[3]->BSIM3gds, heres[4]->BSIM3gds, heres[5]->BSIM3gds, heres[6]->BSIM3gds, heres[7]->BSIM3gds}) + FwdSum) - gcdsb) - (dxpart * ggts)) - (T1 * ddxpart_dVs)) - gbdpsp);
    heres[0]->BSIM3DPspPt = val[0];
    heres[1]->BSIM3DPspPt = val[1];
    heres[2]->BSIM3DPspPt = val[2];
    heres[3]->BSIM3DPspPt = val[3];
    heres[4]->BSIM3DPspPt = val[4];
    heres[5]->BSIM3DPspPt = val[5];
    heres[6]->BSIM3DPspPt = val[6];
    heres[7]->BSIM3DPspPt = val[7];
  }
  {
    Vec8d val = m * ((((gcsgb - Gm) + (sxpart * ggtg)) + (T1 * dsxpart_dVg)) + gbspg);
    heres[0]->BSIM3SPgPt = val[0];
    heres[1]->BSIM3SPgPt = val[1];
    heres[2]->BSIM3SPgPt = val[2];
    heres[3]->BSIM3SPgPt = val[3];
    heres[4]->BSIM3SPgPt = val[4];
    heres[5]->BSIM3SPgPt = val[5];
    heres[6]->BSIM3SPgPt = val[6];
    heres[7]->BSIM3SPgPt = val[7];
  }
  {
    Vec8d val = m * ((Vec8d ){heres[0]->BSIM3sourceConductance, heres[1]->BSIM3sourceConductance, heres[2]->BSIM3sourceConductance, heres[3]->BSIM3sourceConductance, heres[4]->BSIM3sourceConductance, heres[5]->BSIM3sourceConductance, heres[6]->BSIM3sourceConductance, heres[7]->BSIM3sourceConductance});
    heres[0]->BSIM3SPsPt = val[0];
    heres[1]->BSIM3SPsPt = val[1];
    heres[2]->BSIM3SPsPt = val[2];
    heres[3]->BSIM3SPsPt = val[3];
    heres[4]->BSIM3SPsPt = val[4];
    heres[5]->BSIM3SPsPt = val[5];
    heres[6]->BSIM3SPsPt = val[6];
    heres[7]->BSIM3SPsPt = val[7];
  }
  {
    Vec8d val = m * (((((((((Vec8d ){heres[0]->BSIM3gbs, heres[1]->BSIM3gbs, heres[2]->BSIM3gbs, heres[3]->BSIM3gbs, heres[4]->BSIM3gbs, heres[5]->BSIM3gbs, heres[6]->BSIM3gbs, heres[7]->BSIM3gbs}) + Gmbs) + gcsgb) + gcsdb) + gcssb) - (sxpart * ggtb)) - (T1 * dsxpart_dVb)) - gbspb);
    heres[0]->BSIM3SPbPt = val[0];
    heres[1]->BSIM3SPbPt = val[1];
    heres[2]->BSIM3SPbPt = val[2];
    heres[3]->BSIM3SPbPt = val[3];
    heres[4]->BSIM3SPbPt = val[4];
    heres[5]->BSIM3SPbPt = val[5];
    heres[6]->BSIM3SPbPt = val[6];
    heres[7]->BSIM3SPbPt = val[7];
  }
  {
    Vec8d val = m * (((((((Vec8d ){heres[0]->BSIM3gds, heres[1]->BSIM3gds, heres[2]->BSIM3gds, heres[3]->BSIM3gds, heres[4]->BSIM3gds, heres[5]->BSIM3gds, heres[6]->BSIM3gds, heres[7]->BSIM3gds}) + RevSum) - gcsdb) - (sxpart * ggtd)) - (T1 * dsxpart_dVd)) - gbspdp);
    heres[0]->BSIM3SPdpPt = val[0];
    heres[1]->BSIM3SPdpPt = val[1];
    heres[2]->BSIM3SPdpPt = val[2];
    heres[3]->BSIM3SPdpPt = val[3];
    heres[4]->BSIM3SPdpPt = val[4];
    heres[5]->BSIM3SPdpPt = val[5];
    heres[6]->BSIM3SPdpPt = val[6];
    heres[7]->BSIM3SPdpPt = val[7];
  }
  if (heres[0]->BSIM3nqsMod)
  {
    {
      Vec8d val = m * (gqdef + ((Vec8d ){heres[0]->BSIM3gtau, heres[1]->BSIM3gtau, heres[2]->BSIM3gtau, heres[3]->BSIM3gtau, heres[4]->BSIM3gtau, heres[5]->BSIM3gtau, heres[6]->BSIM3gtau, heres[7]->BSIM3gtau}));
      *heres[0]->BSIM3QqPtr += val[0];
      *heres[1]->BSIM3QqPtr += val[1];
      *heres[2]->BSIM3QqPtr += val[2];
      *heres[3]->BSIM3QqPtr += val[3];
      *heres[4]->BSIM3QqPtr += val[4];
      *heres[5]->BSIM3QqPtr += val[5];
      *heres[6]->BSIM3QqPtr += val[6];
      *heres[7]->BSIM3QqPtr += val[7];
    }
    {
      Vec8d val = m * (dxpart * ((Vec8d ){heres[0]->BSIM3gtau, heres[1]->BSIM3gtau, heres[2]->BSIM3gtau, heres[3]->BSIM3gtau, heres[4]->BSIM3gtau, heres[5]->BSIM3gtau, heres[6]->BSIM3gtau, heres[7]->BSIM3gtau}));
      *heres[0]->BSIM3DPqPtr += val[0];
      *heres[1]->BSIM3DPqPtr += val[1];
      *heres[2]->BSIM3DPqPtr += val[2];
      *heres[3]->BSIM3DPqPtr += val[3];
      *heres[4]->BSIM3DPqPtr += val[4];
      *heres[5]->BSIM3DPqPtr += val[5];
      *heres[6]->BSIM3DPqPtr += val[6];
      *heres[7]->BSIM3DPqPtr += val[7];
    }
    {
      Vec8d val = m * (sxpart * ((Vec8d ){heres[0]->BSIM3gtau, heres[1]->BSIM3gtau, heres[2]->BSIM3gtau, heres[3]->BSIM3gtau, heres[4]->BSIM3gtau, heres[5]->BSIM3gtau, heres[6]->BSIM3gtau, heres[7]->BSIM3gtau}));
      *heres[0]->BSIM3SPqPtr += val[0];
      *heres[1]->BSIM3SPqPtr += val[1];
      *heres[2]->BSIM3SPqPtr += val[2];
      *heres[3]->BSIM3SPqPtr += val[3];
      *heres[4]->BSIM3SPqPtr += val[4];
      *heres[5]->BSIM3SPqPtr += val[5];
      *heres[6]->BSIM3SPqPtr += val[6];
      *heres[7]->BSIM3SPqPtr += val[7];
    }
    {
      Vec8d val = m * ((Vec8d ){heres[0]->BSIM3gtau, heres[1]->BSIM3gtau, heres[2]->BSIM3gtau, heres[3]->BSIM3gtau, heres[4]->BSIM3gtau, heres[5]->BSIM3gtau, heres[6]->BSIM3gtau, heres[7]->BSIM3gtau});
      *heres[0]->BSIM3GqPtr -= val[0];
      *heres[1]->BSIM3GqPtr -= val[1];
      *heres[2]->BSIM3GqPtr -= val[2];
      *heres[3]->BSIM3GqPtr -= val[3];
      *heres[4]->BSIM3GqPtr -= val[4];
      *heres[5]->BSIM3GqPtr -= val[5];
      *heres[6]->BSIM3GqPtr -= val[6];
      *heres[7]->BSIM3GqPtr -= val[7];
    }
    {
      Vec8d val = m * (ggtg - gcqgb);
      *heres[0]->BSIM3QgPtr += val[0];
      *heres[1]->BSIM3QgPtr += val[1];
      *heres[2]->BSIM3QgPtr += val[2];
      *heres[3]->BSIM3QgPtr += val[3];
      *heres[4]->BSIM3QgPtr += val[4];
      *heres[5]->BSIM3QgPtr += val[5];
      *heres[6]->BSIM3QgPtr += val[6];
      *heres[7]->BSIM3QgPtr += val[7];
    }
    {
      Vec8d val = m * (ggtd - gcqdb);
      *heres[0]->BSIM3QdpPtr += val[0];
      *heres[1]->BSIM3QdpPtr += val[1];
      *heres[2]->BSIM3QdpPtr += val[2];
      *heres[3]->BSIM3QdpPtr += val[3];
      *heres[4]->BSIM3QdpPtr += val[4];
      *heres[5]->BSIM3QdpPtr += val[5];
      *heres[6]->BSIM3QdpPtr += val[6];
      *heres[7]->BSIM3QdpPtr += val[7];
    }
    {
      Vec8d val = m * (ggts - gcqsb);
      *heres[0]->BSIM3QspPtr += val[0];
      *heres[1]->BSIM3QspPtr += val[1];
      *heres[2]->BSIM3QspPtr += val[2];
      *heres[3]->BSIM3QspPtr += val[3];
      *heres[4]->BSIM3QspPtr += val[4];
      *heres[5]->BSIM3QspPtr += val[5];
      *heres[6]->BSIM3QspPtr += val[6];
      *heres[7]->BSIM3QspPtr += val[7];
    }
    {
      Vec8d val = m * (ggtb - gcqbb);
      *heres[0]->BSIM3QbPtr += val[0];
      *heres[1]->BSIM3QbPtr += val[1];
      *heres[2]->BSIM3QbPtr += val[2];
      *heres[3]->BSIM3QbPtr += val[3];
      *heres[4]->BSIM3QbPtr += val[4];
      *heres[5]->BSIM3QbPtr += val[5];
      *heres[6]->BSIM3QbPtr += val[6];
      *heres[7]->BSIM3QbPtr += val[7];
    }
  }

  line1000:
  ;

  return OK;
}

