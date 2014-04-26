/*
 * Copyright (c) 2014, NVIDIA Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation and/or
 *    other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to
 *    endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "ngspice/config.h"
#include "ngspice/const.h"
#include "ngspice/macros.h"
#include "ngspice/CUSPICE/cuniinteg.cuh"
#include "bsim4v7def.h"

#define MAX_EXPL 2.688117142e+43
#define MIN_EXPL 3.720075976e-44
#define EXPL_THRESHOLD 100.0
#define MAX_EXP 5.834617425e14
#define MIN_EXP 1.713908431e-15
#define EXP_THRESHOLD 34.0
#define EPS0 8.85418e-12
#define EPSSI 1.03594e-10
#define Charge_q 1.60219e-19
#define DELTA_1 0.02
#define DELTA_2 0.02
#define DELTA_3 0.02
#define DELTA_4 0.02
#define MM  3  /* smooth coeff */
#define DEXP(A,B,C) {                                                         \
        if (A > EXP_THRESHOLD) {                                              \
            B = MAX_EXP*(1.0+(A)-EXP_THRESHOLD);                              \
            C = MAX_EXP;                                                      \
        } else if (A < -EXP_THRESHOLD)  {                                     \
            B = MIN_EXP;                                                      \
            C = 0;                                                            \
        } else   {                                                            \
            B = exp(A);                                                       \
            C = B;                                                            \
        }                                                                     \
    }

extern "C"
__device__
static
double
DEVlimvds (double vnew, double vold)
{
    if (vold >= 3.5)
    {
        if (vnew > vold)
            vnew = MIN (vnew, (3 * vold) + 2) ;
        else
            if (vnew < 3.5)
                vnew = MAX (vnew, 2) ;
    } else {
        if (vnew > vold)
            vnew = MIN (vnew, 4) ;
        else
            vnew = MAX (vnew, -.5) ;
    }

    return (vnew) ;
}

extern "C"
__device__
static
double
DEVpnjlim (double vnew, double vold, double vt, double vcrit, int *icheck)
{
    double arg ;

    if ((vnew > vcrit) && (fabs (vnew - vold) > (vt + vt)))
    {
        if (vold > 0)
        {
            arg = (vnew - vold) / vt ;
            if (arg > 0)
                vnew = vold + vt * (2 + log (arg - 2)) ;
            else
                vnew = vold - vt * (2 + log (2 - arg)) ;
        } else
            vnew = vt * log (vnew / vt) ;

        *icheck = 1 ;

    } else {
        if (vnew < 0)
        {
            if (vold > 0)
                arg = -1 * vold - 1 ;
            else
                arg = 2 * vold - 1 ;

            if (vnew < arg)
            {
                vnew = arg ;
                *icheck = 1 ;
            } else {
                *icheck = 0 ;
            } ;
        } else {
           *icheck = 0 ;
        }
    }

    return (vnew) ;
}

extern "C"
__device__
static
double
DEVfetlim (double vnew, double vold, double vto)
{
    double vtsthi, vtstlo, vtox, delv, vtemp ;

    vtsthi = fabs (2 * (vold - vto)) + 2 ;
    vtstlo = fabs (vold - vto) + 1 ;
    vtox = vto + 3.5 ;
    delv = vnew - vold ;

    if (vold >= vto)
    {
        if (vold >= vtox)
        {
            if (delv <= 0)
            {
                /* going off */
                if (vnew >= vtox)
                {
                    if (-delv > vtstlo)
                        vnew =  vold - vtstlo ;
                } else
                    vnew = MAX (vnew, vto + 2) ;
            } else {
                /* staying on */
                if (delv >= vtsthi)
                    vnew = vold + vtsthi ;
            }
        } else {
            /* middle region */
            if (delv <= 0)
                /* decreasing */
                vnew = MAX (vnew, vto - .5) ;
            else
                /* increasing */
                vnew = MIN (vnew, vto + 4) ;
        }
    } else {
        /* off */
        if (delv <= 0)
        {
            if (-delv >vtsthi)
                vnew = vold - vtsthi ;
        } else {
            vtemp = vto + .5 ;
            if (vnew <= vtemp)
            {
                if (delv >vtstlo)
                    vnew = vold + vtstlo ;
            } else
                vnew = vtemp ;
        }
    }

    return (vnew) ;
}

/* function to compute poly depletion effect */
extern "C"
__device__
static
int
BSIM4v7polyDepletion (double phi, double ngate, double epsgate, double coxe,
                    double Vgs, double *Vgs_eff, double *dVgs_eff_dVg)
{
    double T1, T2, T3, T4, T5, T6, T7, T8 ;

    /* Poly Gate Si Depletion Effect */
    if ((ngate > 1.0e18) && (ngate < 1.0e25) && (Vgs > phi) && (epsgate!=0))
    {
        T1 = 1.0e6 * CHARGE * epsgate * ngate / (coxe * coxe) ;
        T8 = Vgs - phi ;
        T4 = sqrt (1.0 + 2.0 * T8 / T1) ;
        T2 = 2.0 * T8 / (T4 + 1.0) ;
        T3 = 0.5 * T2 * T2 / T1 ; /* T3 = Vpoly */
        T7 = 1.12 - T3 - 0.05 ;
        T6 = sqrt (T7 * T7 + 0.224) ;
        T5 = 1.12 - 0.5 * (T7 + T6) ;
        *Vgs_eff = Vgs - T5 ;
        *dVgs_eff_dVg = 1.0 - (0.5 - 0.5 / T4) * (1.0 + T7 / T6) ;
    } else {
        *Vgs_eff = Vgs ;
        *dVgs_eff_dVg = 1.0 ;
    }

    return 0 ;
}

extern "C"
__global__ void cuBSIM4v7load_kernel
(
BSIM4v7paramGPUstruct, struct bsim4SizeDependParam  **, double *, double *, double *, double *,
double, double, double, double, double,
double, double, double, int, int, int, int,
double, int *, double,
/* Model */
int, int, int, int, double, double,
double, double, double,
double, double, double,
double, int, double, double,
double, double, double,
double, double, double, double,
double, double, double, double,
double, double, double, double, double,
double, double, double, double, double,
double, double, double, double, double,
double, double, double, double, int,
int, double, double, double, double,
double, double, int, double, int,
int, int, double, int, int,
double, double, double, double, double,
double, double, double, double,
double, double, double, double, double,
double, double, double, double,
/* Position Vectors and CKTloadOutputs */
int *, double *, int *, double *
) ;

extern "C"
int
cuBSIM4v7load
(
GENmodel *inModel, CKTcircuit *ckt
)
{
    BSIM4v7model *model = (BSIM4v7model *)inModel ;
    int i, thread_x, thread_y, block_x ;

    cudaStream_t stream [2] ;

    cudaError_t status ;

    for (i = 0 ; i < 2 ; i++)
        cudaStreamCreate (&(stream [i])) ;

    i = 0 ;

    /*  loop through all the BSIM4v7 models */
    for ( ; model != NULL ; model = BSIM4v7nextModel(model))
    {
        /* Determining how many blocks should exist in the kernel */
        thread_x = 1 ;
        thread_y = 256 ;
        if (model->n_instances % thread_y != 0)
            block_x = (int)((model->n_instances + thread_y - 1) / thread_y) ;
        else
            block_x = model->n_instances / thread_y ;

        dim3 thread (thread_x, thread_y) ;

        /* Kernel launch */
        status = cudaGetLastError () ; // clear error status

        cuBSIM4v7load_kernel <<< block_x, thread, 0, stream [i] >>>
                                                   (model->BSIM4v7paramGPU, model->d_pParam, ckt->d_CKTrhsOld,
                                                    ckt->d_CKTstate0, ckt->d_CKTstate1, ckt->d_CKTstate2,
                                                    ckt->CKTdelta, ckt->CKTdeltaOld [1], ckt->CKTreltol,
                                                    ckt->CKTvoltTol, ckt->CKTabstol, ckt->CKTag [0], ckt->CKTag [1],
                                                    ckt->CKTgmin, ckt->CKTbypass, ckt->CKTmode, ckt->CKTorder,
                                                    model->n_instances, ckt->CKTtemp, ckt->d_CKTnoncon, CONSTvt0,
                                                    /* Model */
                                                    model->BSIM4v7type, model->BSIM4v7rdsMod, model->BSIM4v7igcMod,
                                                    model->BSIM4v7igbMod, model->BSIM4v7vcrit, model->BSIM4v7vtm,
                                                    model->BSIM4v7vtm0, model->BSIM4v7SjctEmissionCoeff,
                                                    model->BSIM4v7SjctTempSatCurDensity,
                                                    model->BSIM4v7SjctSidewallTempSatCurDensity,
                                                    model->BSIM4v7SjctGateSidewallTempSatCurDensity,
                                                    model->BSIM4v7xjbvs, model->BSIM4v7bvs, model->BSIM4v7dioMod,
                                                    model->BSIM4v7DjctEmissionCoeff, model->BSIM4v7DjctTempSatCurDensity,
                                                    model->BSIM4v7DjctSidewallTempSatCurDensity,
                                                    model->BSIM4v7DjctGateSidewallTempSatCurDensity, model->BSIM4v7xjbvd,
                                                    model->BSIM4v7bvd, model->BSIM4v7njtsswstemp, model->BSIM4v7njtsswgstemp,
                                                    model->BSIM4v7njtsstemp, model->BSIM4v7njtsswdtemp,
                                                    model->BSIM4v7njtsswgdtemp, model->BSIM4v7njtsdtemp, model->BSIM4v7vtss,
                                                    model->BSIM4v7vtsd, model->BSIM4v7vtssws, model->BSIM4v7vtsswd,
                                                    model->BSIM4v7vtsswgs, model->BSIM4v7vtsswgd, model->BSIM4v7mtrlMod,
                                                    model->BSIM4v7eot, model->BSIM4v7epsrsub, model->BSIM4v7epsrox,
                                                    model->BSIM4v7toxe, model->BSIM4v7factor1, model->BSIM4v7tnom,
                                                    model->BSIM4v7coxe, model->BSIM4v7tempMod, model->BSIM4v7epsrgate,
                                                    model->BSIM4v7mtrlCompatMod, model->BSIM4v7phig, model->BSIM4v7easub,
                                                    model->BSIM4v7Eg0, model->BSIM4v7mobMod, model->BSIM4v7lambdaGiven,
                                                    model->BSIM4v7lambda, model->BSIM4v7toxp, model->BSIM4v7bdos,
                                                    model->BSIM4v7ados, model->BSIM4v7coxp, model->BSIM4v7pditsl,
                                                    model->BSIM4v7vtlGiven, model->BSIM4v7vtl, model->BSIM4v7gidlMod,
                                                    model->BSIM4v7pigcdGiven, model->BSIM4v7tnoiMod, model->BSIM4v7xpart,
                                                    model->BSIM4v7capMod, model->BSIM4v7cvchargeMod,
                                                    model->BSIM4v7DunitAreaTempJctCap, model->BSIM4v7SunitAreaTempJctCap,
                                                    model->BSIM4v7DunitLengthSidewallTempJctCap,
                                                    model->BSIM4v7DunitLengthGateSidewallTempJctCap,
                                                    model->BSIM4v7SunitLengthSidewallTempJctCap,
                                                    model->BSIM4v7SunitLengthGateSidewallTempJctCap,
                                                    model->BSIM4v7SbulkJctBotGradingCoeff,
                                                    model->BSIM4v7SbulkJctSideGradingCoeff,
                                                    model->BSIM4v7SbulkJctGateSideGradingCoeff,
                                                    model->BSIM4v7DbulkJctBotGradingCoeff,
                                                    model->BSIM4v7DbulkJctSideGradingCoeff,
                                                    model->BSIM4v7DbulkJctGateSideGradingCoeff,
                                                    model->BSIM4v7PhiBS, model->BSIM4v7PhiBSWS, model->BSIM4v7PhiBSWGS,
                                                    model->BSIM4v7PhiBD, model->BSIM4v7PhiBSWD, model->BSIM4v7PhiBSWGD,
                                                    model->d_PositionVector, ckt->d_CKTloadOutput,
                                                    model->d_PositionVectorRHS, ckt->d_CKTloadOutputRHS) ;

        status = cudaGetLastError () ; // check for launch error
        if (status != cudaSuccess)
        {
            fprintf (stderr, "Kernel launch failure in the BSIM4v7 Model\n\n") ;
            return (E_NOMEM) ;
        }

        i++ ;
    }

    cudaDeviceSynchronize () ;

    /* Deallocation */
    for (i = 0 ; i < 2 ; i++)
        cudaStreamDestroy (stream [i]) ;

    return (OK) ;
}

extern "C"
__global__
void
cuBSIM4v7load_kernel
(
BSIM4v7paramGPUstruct BSIM4v7entry, struct bsim4SizeDependParam  **d_pParam, double *CKTrhsOld,
double *CKTstate_0, double *CKTstate_1, double *CKTstate_2,
double CKTdelta, double CKTdeltaOld_1, double CKTrelTol, double CKTvoltTol, double CKTabsTol,
double CKTag_0, double CKTag_1, double CKTgmin, int CKTbypass, int CKTmode, int CKTorder, int n_instances,
double CKTtemp, int *d_CKTnoncon, double CONSTvt0,
/* Model */
int BSIM4v7type, int BSIM4v7rdsMod, int BSIM4v7igcMod, int BSIM4v7igbMod, double BSIM4v7vcrit, double BSIM4v7vtm,
double BSIM4v7vtm0, double BSIM4v7SjctEmissionCoeff, double BSIM4v7SjctTempSatCurDensity,
double BSIM4v7SjctSidewallTempSatCurDensity, double BSIM4v7SjctGateSidewallTempSatCurDensity, double BSIM4v7xjbvs,
double BSIM4v7bvs, int BSIM4v7dioMod, double BSIM4v7DjctEmissionCoeff, double BSIM4v7DjctTempSatCurDensity,
double BSIM4v7DjctSidewallTempSatCurDensity, double BSIM4v7DjctGateSidewallTempSatCurDensity, double BSIM4v7xjbvd,
double BSIM4v7bvd, double BSIM4v7njtsswstemp, double BSIM4v7njtsswgstemp, double BSIM4v7njtsstemp,
double BSIM4v7njtsswdtemp, double BSIM4v7njtsswgdtemp, double BSIM4v7njtsdtemp, double BSIM4v7vtss,
double BSIM4v7vtsd, double BSIM4v7vtssws, double BSIM4v7vtsswd, double BSIM4v7vtsswgs, double BSIM4v7vtsswgd,
double BSIM4v7mtrlMod, double BSIM4v7eot, double BSIM4v7epsrsub, double BSIM4v7epsrox, double BSIM4v7toxe,
double BSIM4v7factor1, double BSIM4v7tnom, double BSIM4v7coxe, double BSIM4v7tempMod, double BSIM4v7epsrgate,
double BSIM4v7mtrlCompatMod, double BSIM4v7phig, double BSIM4v7easub, double BSIM4v7Eg0, int BSIM4v7mobMod,
int BSIM4v7lambdaGiven, double BSIM4v7lambda, double BSIM4v7toxp, double BSIM4v7bdos, double BSIM4v7ados,
double BSIM4v7coxp, double BSIM4v7pditsl, int BSIM4v7vtlGiven, double BSIM4v7vtl, int BSIM4v7gidlMod,
int BSIM4v7pigcdGiven, int BSIM4v7tnoiMod, double BSIM4v7xpart, int BSIM4v7capMod, int BSIM4v7cvchargeMod,
double BSIM4v7DunitAreaTempJctCap, double BSIM4v7SunitAreaTempJctCap, double BSIM4v7DunitLengthSidewallTempJctCap,
double BSIM4v7DunitLengthGateSidewallTempJctCap, double BSIM4v7SunitLengthSidewallTempJctCap,
double BSIM4v7SunitLengthGateSidewallTempJctCap, double BSIM4v7SbulkJctBotGradingCoeff,
double BSIM4v7SbulkJctSideGradingCoeff, double BSIM4v7SbulkJctGateSideGradingCoeff,
double BSIM4v7DbulkJctBotGradingCoeff, double BSIM4v7DbulkJctSideGradingCoeff,
double BSIM4v7DbulkJctGateSideGradingCoeff, double BSIM4v7PhiBS, double BSIM4v7PhiBSWS, double BSIM4v7PhiBSWGS,
double BSIM4v7PhiBD, double BSIM4v7PhiBSWD, double BSIM4v7PhiBSWGD,
/* Position Vectors and CKTloadOutputs */
int *d_PositionVector, double *d_CKTloadOutput, int *d_PositionVectorRHS, double *d_CKTloadOutputRHS
)
{
    int instance_ID, pos, posRHS, total_offset, total_offsetRHS ;

    double ceqgstot, dgstot_dvd, dgstot_dvg, dgstot_dvs, dgstot_dvb ;
    double ceqgdtot, dgdtot_dvd, dgdtot_dvg, dgdtot_dvs, dgdtot_dvb ;
    double gstot, gstotd, gstotg, gstots, gstotb, gspr, Rs, Rd ;
    double gdtot, gdtotd, gdtotg, gdtots, gdtotb, gdpr ;
    double vgs_eff, vgd_eff, dvgs_eff_dvg, dvgd_eff_dvg ;
    double dRs_dvg, dRd_dvg, dRs_dvb, dRd_dvb ;
    double dT0_dvg, dT1_dvb, dT3_dvg, dT3_dvb ;
    double vses, vdes, vdedo, delvses, delvded, delvdes ;
    double Isestot, cseshat, Idedtot, cdedhat ;

#ifndef NEWCONV
    double tol0, tol1, tol2, tol3, tol4, tol5, tol6 ;
#endif

    double geltd, gcrg, gcrgg, gcrgd, gcrgs, gcrgb, ceqgcrg ;
    double vges, vgms, vgedo, vgmdo, vged, vgmd ;
    double delvges, delvgms, vgmb ;
    double gcgmgmb = 0.0, gcgmdb = 0.0, gcgmsb = 0.0, gcdgmb, gcsgmb ;
    double gcgmbb = 0.0, gcbgmb, qgmb, qgmid = 0.0, ceqqgmid ;
    double vbd, vbs, vds, vgb, vgd, vgs, vgdo ;

#ifndef PREDICTOR
    double xfact ;
#endif

    double vdbs, vdbd, vsbs, vsbdo, vsbd ;
    double delvdbs, delvdbd, delvsbs ;
    double delvbd_jct, delvbs_jct, vbs_jct, vbd_jct ;
    double SourceSatCurrent, DrainSatCurrent ;
    double ag0, qgb, von, cbhat, VgstNVt, ExpVgst ;
    double ceqqb, ceqqd, ceqqg, ceqqjd = 0.0, ceqqjs = 0.0, ceq, geq ;
    double cdrain, cdhat, ceqdrn, ceqbd, ceqbs, ceqjd, ceqjs, gjbd, gjbs ;
    double czbd, czbdsw, czbdswg, czbs, czbssw, czbsswg, evbd, evbs, arg, sarg ;
    double delvbd, delvbs, delvds, delvgd, delvgs ;
    double Vfbeff, dVfbeff_dVg, dVfbeff_dVb, V3, V4 ;
    double gcbdb, gcbgb, gcbsb, gcddb, gcdgb, gcdsb, gcgdb, gcggb, gcgsb, gcsdb ;
    double gcgbb, gcdbb, gcsbb, gcbbb ;
    double gcdbdb, gcsbsb ;
    double gcsgb, gcssb, MJD, MJSWD, MJSWGD, MJS, MJSWS, MJSWGS ;
    double qgate = 0.0, qbulk = 0.0, qdrn = 0.0, qsrc, cqgate, cqbody, cqdrn ;
    double Vds, Vbs, Gmbs, FwdSum, RevSum ;
    double Igidl, Ggidld, Ggidlg, Ggidlb ;
    double Voxacc = 0.0, dVoxacc_dVg = 0.0, dVoxacc_dVb = 0.0 ;
    double Voxdepinv = 0.0, dVoxdepinv_dVg = 0.0, dVoxdepinv_dVd = 0.0, dVoxdepinv_dVb = 0.0 ;
    double VxNVt = 0.0, ExpVxNVt, Vaux = 0.0, dVaux_dVg = 0.0, dVaux_dVd = 0.0, dVaux_dVb = 0.0 ;
    double Igc, dIgc_dVg, dIgc_dVd, dIgc_dVb ;
    double Igcs, dIgcs_dVg, dIgcs_dVd, dIgcs_dVb ;
    double Igcd, dIgcd_dVg, dIgcd_dVd, dIgcd_dVb ;
    double Igs, dIgs_dVg, dIgs_dVs, Igd, dIgd_dVg, dIgd_dVd ;
    double Igbacc, dIgbacc_dVg, dIgbacc_dVb ;
    double Igbinv, dIgbinv_dVg, dIgbinv_dVd, dIgbinv_dVb ;
    double Pigcd, dPigcd_dVg, dPigcd_dVd, dPigcd_dVb ;
    double Istoteq, gIstotg, gIstotd, gIstots, gIstotb ;
    double Idtoteq, gIdtotg, gIdtotd, gIdtots, gIdtotb ;
    double Ibtoteq, gIbtotg, gIbtotd, gIbtots, gIbtotb ;
    double Igtoteq, gIgtotg, gIgtotd, gIgtots, gIgtotb ;
    double Igstot, cgshat, Igdtot, cgdhat, Igbtot, cgbhat ;
    double Vgs_eff, Vfb = 0.0, Vth_NarrowW ;

    /* double Vgd_eff, dVgd_eff_dVg;          v4.7.0 */

    double Phis, dPhis_dVb, sqrtPhis, dsqrtPhis_dVb, Vth, dVth_dVb, dVth_dVd ;
    double Vgst, dVgs_eff_dVg, Nvtms, Nvtmd ;
    double Vtm, Vtm0 ;
    double n, dn_dVb, dn_dVd, voffcv, noff, dnoff_dVd, dnoff_dVb ;
    double V0, CoxWLcen, QovCox, LINK ;
    double DeltaPhi, dDeltaPhi_dVg, VgDP, dVgDP_dVg ;
    double Cox, Tox, Tcen, dTcen_dVg, dTcen_dVd, dTcen_dVb ;
    double Ccen, Coxeff, dCoxeff_dVd, dCoxeff_dVg, dCoxeff_dVb ;
    double Denomi, dDenomi_dVg, dDenomi_dVd, dDenomi_dVb ;
    double ueff, dueff_dVg, dueff_dVd, dueff_dVb ;
    double Esat, Vdsat ;
    double EsatL, dEsatL_dVg, dEsatL_dVd, dEsatL_dVb ;
    double dVdsat_dVg, dVdsat_dVb, dVdsat_dVd, Vasat, dAlphaz_dVg, dAlphaz_dVb ;
    double dVasat_dVg, dVasat_dVb, dVasat_dVd, Va, dVa_dVd, dVa_dVg, dVa_dVb ;
    double Vbseff, dVbseff_dVb, VbseffCV, dVbseffCV_dVb ;
    double VgsteffVth, dT11_dVg ;
    double Arg1, One_Third_CoxWL, Two_Third_CoxWL, Alphaz, CoxWL ;
    double T0 = 0.0, dT0_dVg, dT0_dVd, dT0_dVb ;
    double T1, dT1_dVg, dT1_dVd, dT1_dVb ;
    double T2, dT2_dVg, dT2_dVd, dT2_dVb ;
    double T3, dT3_dVg, dT3_dVd, dT3_dVb ;
    double T4, dT4_dVd, dT4_dVb ;
    double T5, dT5_dVg, dT5_dVd, dT5_dVb ;
    double T6, dT6_dVg, dT6_dVd, dT6_dVb ;
    double T7, dT7_dVg, dT7_dVd, dT7_dVb ;
    double T8, dT8_dVg, dT8_dVd, dT8_dVb ;
    double T9, dT9_dVg, dT9_dVd, dT9_dVb ;
    double T10, dT10_dVg, dT10_dVb, dT10_dVd ;
    double T11, T12, T13, T14 ;
    double tmp, Abulk, dAbulk_dVb, Abulk0, dAbulk0_dVb ;
    double Cclm, dCclm_dVg, dCclm_dVd, dCclm_dVb ;
    double FP, dFP_dVg, PvagTerm, dPvagTerm_dVg, dPvagTerm_dVd, dPvagTerm_dVb ;
    double VADITS, dVADITS_dVg, dVADITS_dVd ;
    double Lpe_Vb, dDITS_Sft_dVb, dDITS_Sft_dVd ;

    /* v4.7 New DITS */
    double DITS_Sft2, dDITS_Sft2_dVd ;

    double VACLM, dVACLM_dVg, dVACLM_dVd, dVACLM_dVb ;
    double VADIBL, dVADIBL_dVg, dVADIBL_dVd, dVADIBL_dVb ;
    double Xdep, dXdep_dVb, lt1, dlt1_dVb, ltw, dltw_dVb, Delt_vth, dDelt_vth_dVb ;
    double Theta0, dTheta0_dVb ;
    double TempRatio, tmp1, tmp2, tmp3, tmp4 ;
    double DIBL_Sft, dDIBL_Sft_dVd, Lambda, dLambda_dVg ;
    double Idtot, Ibtot, a1, ScalingFactor ;
    double Vgsteff, dVgsteff_dVg, dVgsteff_dVd, dVgsteff_dVb ;
    double Vdseff, dVdseff_dVg, dVdseff_dVd, dVdseff_dVb ;
    double VdseffCV, dVdseffCV_dVg, dVdseffCV_dVd, dVdseffCV_dVb ;
    double diffVds, dAbulk_dVg ;
    double beta, dbeta_dVg, dbeta_dVd, dbeta_dVb ;
    double gche, dgche_dVg, dgche_dVd, dgche_dVb ;
    double fgche1, dfgche1_dVg, dfgche1_dVd, dfgche1_dVb ;
    double fgche2, dfgche2_dVg, dfgche2_dVd, dfgche2_dVb ;
    double Idl, dIdl_dVg, dIdl_dVd, dIdl_dVb ;
    double Idsa, dIdsa_dVg, dIdsa_dVd, dIdsa_dVb ;
    double Ids, Gm, Gds, Gmb, devbs_dvb, devbd_dvb ;
    double Isub, Gbd, Gbg, Gbb ;
    double VASCBE, dVASCBE_dVg, dVASCBE_dVd, dVASCBE_dVb ;
    double CoxeffWovL ;
    double Rds, dRds_dVg, dRds_dVb, WVCox, WVCoxRds ;
    double Vgst2Vtm, VdsatCV ;
    double Leff, Weff, dWeff_dVg, dWeff_dVb ;
    double AbulkCV, dAbulkCV_dVb ;
    double qcheq, qdef, gqdef = 0.0, cqdef = 0.0, cqcheq = 0.0 ;
    double gcqdb = 0.0, gcqsb = 0.0, gcqgb = 0.0, gcqbb = 0.0 ;
    double dxpart, sxpart, ggtg, ggtd, ggts, ggtb ;
    double ddxpart_dVd, ddxpart_dVg, ddxpart_dVb, ddxpart_dVs ;
    double dsxpart_dVd, dsxpart_dVg, dsxpart_dVb, dsxpart_dVs ;
    double gbspsp, gbbdp, gbbsp, gbspg, gbspb, gbspdp ;
    double gbdpdp, gbdpg, gbdpb, gbdpsp ;
    double qgdo, qgso, cgdo, cgso ;
    double Cgg, Cgd, Cgb, Cdg, Cdd, Cds ;
    double Csg, Csd, Css, Csb, Cbg, Cbd, Cbb ;
    double Cgg1, Cgd1, Cgb1, Cbg1, Cbb1, Cbd1, Qac0, Qsub0 ;
    double dQac0_dVg, dQac0_dVb, dQsub0_dVg, dQsub0_dVd, dQsub0_dVb ;
    double ggidld, ggidlg, ggidlb, ggislg, ggislb, ggisls ;
    double Igisl, Ggislg, Ggislb, Ggisls ;
    double Nvtmrss, Nvtmrssws, Nvtmrsswgs ;
    double Nvtmrsd, Nvtmrsswd, Nvtmrsswgd ;
    double vs, Fsevl, dvs_dVg, dvs_dVd, dvs_dVb, dFsevl_dVg, dFsevl_dVd, dFsevl_dVb ;
    double vgdx, vgsx, epssub, toxe, epsrox ;
    struct bsim4SizeDependParam *pParam ;
    int ByPass, ChargeComputationNeeded, error, Check, Check1, Check2 ;
    double m ;

    instance_ID = threadIdx.y + blockDim.y * blockIdx.x ;

    if (instance_ID < n_instances)
    {
        if (threadIdx.x == 0)
        {
            ScalingFactor = 1.0e-9 ;
            ChargeComputationNeeded = ((CKTmode & (MODEAC | MODETRAN | MODEINITSMSIG)) ||
                                      ((CKTmode & MODETRANOP) && (CKTmode & MODEUIC))) ? 1 : 0 ;

            Check = Check1 = Check2 = 1 ;
            ByPass = 0 ;
            pParam = d_pParam [instance_ID] ;

            /* 1 - non-divergent */
            if (CKTmode & MODEINITSMSIG)
            {
                vds = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 3] ;
                vgs = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 2] ;
                vbs = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 1] ;
                vges = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 7] ;
                vgms = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 8] ;
                vdbs = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 4] ;
                vsbs = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 6] ;
                vses = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 9] ;
                vdes = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 10] ;
                qdef = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 27] ;
            }
            else if (CKTmode & MODEINITTRAN)
            {
                vds = CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 3] ;
                vgs = CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 2] ;
                vbs = CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 1] ;
                vges = CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 7] ;
                vgms = CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 8] ;
                vdbs = CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 4] ;
                vsbs = CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 6] ;
                vses = CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 9] ;
                vdes = CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 10] ;
                qdef = CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 27] ;
            }
            else if ((CKTmode & MODEINITJCT) && !BSIM4v7entry.d_BSIM4v7offArray [instance_ID])
            {
                vds = BSIM4v7type * BSIM4v7entry.d_BSIM4v7icVDSArray [instance_ID] ;
                vgs = vges = vgms = BSIM4v7type * BSIM4v7entry.d_BSIM4v7icVGSArray [instance_ID] ;
                vbs = vdbs = vsbs = BSIM4v7type * BSIM4v7entry.d_BSIM4v7icVBSArray [instance_ID] ;

                /* 1 - DIVERGENT */
                if (vds > 0.0)
                {
                    vdes = vds + 0.01 ;
                    vses = -0.01 ;
                }
                else if (vds < 0.0)
                {
                    vdes = vds - 0.01 ;
                    vses = 0.01 ;
                }
                else
                    vdes = vses = 0.0 ;

                qdef = 0.0 ;

                /* 2 - DIVERGENT */
                if ((vds == 0.0) && (vgs == 0.0) && (vbs == 0.0)
                    && ((CKTmode & (MODETRAN | MODEAC|MODEDCOP |
                    MODEDCTRANCURVE)) || (!(CKTmode & MODEUIC))))
                {
                    vds = 0.1 ;
                    vdes = 0.11 ;
                    vses = -0.01 ;
                    vgs = vges = vgms = BSIM4v7type * BSIM4v7entry.d_BSIM4v7vth0Array [instance_ID] + 0.1 ;
                    vbs = vdbs = vsbs = 0.0 ;
                }
            }
            else if ((CKTmode & (MODEINITJCT | MODEINITFIX)) && (BSIM4v7entry.d_BSIM4v7offArray [instance_ID]))
            {
                vds = vgs = vbs = vges = vgms = 0.0 ;
                vdbs = vsbs = vdes = vses = qdef = 0.0 ;
            } else {

#ifndef PREDICTOR
                /* 2 - non-divergent */
                if (CKTmode & MODEINITPRED)
                {
                    xfact = CKTdelta / CKTdeltaOld_1 ;

                    CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 3] = CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 3] ;
                    vds = (1.0 + xfact) * CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 3] - xfact * CKTstate_2 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 3] ;

                    CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 2] = CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 2] ;
                    vgs = (1.0 + xfact) * CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 2] - xfact * CKTstate_2 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 2] ;

                    CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 7] = CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 7] ;
                    vges = (1.0 + xfact) * CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 7] - xfact * CKTstate_2 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 7] ;

                    CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 8] = CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 8] ;
                    vgms = (1.0 + xfact) * CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 8] - xfact * CKTstate_2 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 8] ;

                    CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 1] = CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 1] ;
                    vbs = (1.0 + xfact) * CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 1] - xfact * CKTstate_2 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 1] ;

                    CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID]] = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 1] - CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 3] ;
                    CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 4] = CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 4] ;
                    vdbs = (1.0 + xfact) * CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 4] - xfact * CKTstate_2 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 4] ;

                    CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 5] = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 4] - CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 3] ;
                    CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 6] = CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 6] ;
                    vsbs = (1.0 + xfact) * CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 6] - xfact * CKTstate_2 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 6] ;

                    CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 9] = CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 9] ;
                    vses = (1.0 + xfact) * CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 9] - xfact * CKTstate_2 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 9] ;

                    CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 10] = CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 10] ;
                    vdes = (1.0 + xfact) * CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 10] - xfact * CKTstate_2 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 10] ;

                    CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 27] = CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 27] ;
                    qdef = (1.0 + xfact) * CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 27] - xfact * CKTstate_2 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 27] ;
                } else {
#endif /* PREDICTOR */

                    vds = BSIM4v7type * (CKTrhsOld [BSIM4v7entry.d_BSIM4v7dNodePrimeArray [instance_ID]] - CKTrhsOld [BSIM4v7entry.d_BSIM4v7sNodePrimeArray [instance_ID]]) ;
                    vgs = BSIM4v7type * (CKTrhsOld [BSIM4v7entry.d_BSIM4v7gNodePrimeArray [instance_ID]] - CKTrhsOld [BSIM4v7entry.d_BSIM4v7sNodePrimeArray [instance_ID]]) ;
                    vbs = BSIM4v7type * (CKTrhsOld [BSIM4v7entry.d_BSIM4v7bNodePrimeArray [instance_ID]] - CKTrhsOld [BSIM4v7entry.d_BSIM4v7sNodePrimeArray [instance_ID]]) ;
                    vges = BSIM4v7type * (CKTrhsOld [BSIM4v7entry.d_BSIM4v7gNodeExtArray [instance_ID]] - CKTrhsOld [BSIM4v7entry.d_BSIM4v7sNodePrimeArray [instance_ID]]) ;
                    vgms = BSIM4v7type * (CKTrhsOld [BSIM4v7entry.d_BSIM4v7gNodeMidArray [instance_ID]] - CKTrhsOld [BSIM4v7entry.d_BSIM4v7sNodePrimeArray [instance_ID]]) ;
                    vdbs = BSIM4v7type * (CKTrhsOld [BSIM4v7entry.d_BSIM4v7dbNodeArray [instance_ID]] - CKTrhsOld [BSIM4v7entry.d_BSIM4v7sNodePrimeArray [instance_ID]]) ;
                    vsbs = BSIM4v7type * (CKTrhsOld [BSIM4v7entry.d_BSIM4v7sbNodeArray [instance_ID]] - CKTrhsOld [BSIM4v7entry.d_BSIM4v7sNodePrimeArray [instance_ID]]) ;
                    vses = BSIM4v7type * (CKTrhsOld [BSIM4v7entry.d_BSIM4v7sNodeArray [instance_ID]] - CKTrhsOld [BSIM4v7entry.d_BSIM4v7sNodePrimeArray [instance_ID]]) ;
                    vdes = BSIM4v7type * (CKTrhsOld [BSIM4v7entry.d_BSIM4v7dNodeArray [instance_ID]] - CKTrhsOld [BSIM4v7entry.d_BSIM4v7sNodePrimeArray [instance_ID]]) ;
                    qdef = BSIM4v7type * CKTrhsOld [BSIM4v7entry.d_BSIM4v7qNodeArray [instance_ID]] ;

#ifndef PREDICTOR
                }
#endif /* PREDICTOR */

                vgdo = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 2] - CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 3] ;
                vgedo = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 7] - CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 3] ;
                vgmdo = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 8] - CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 3] ;
                vbd = vbs - vds ;
                vdbd = vdbs - vds ;
                vgd = vgs - vds ;
                vged = vges - vds ;
                vgmd = vgms - vds ;
                delvbd = vbd - CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID]] ;
                delvdbd = vdbd - CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 5] ;
                delvgd = vgd - vgdo ;
                delvds = vds - CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 3] ;
                delvgs = vgs - CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 2] ;
                delvges = vges - CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 7] ;
                delvgms = vgms - CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 8] ;
                delvbs = vbs - CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 1] ;
                delvdbs = vdbs - CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 4] ;
                delvsbs = vsbs - CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 6] ;
                delvses = vses - CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 9] ;
                vdedo = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 10] - CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 3] ;
                delvdes = vdes - CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 10] ;
                delvded = vdes - vds - vdedo ;
                delvbd_jct = (!BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID]) ? delvbd : delvdbd ;
                delvbs_jct = (!BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID]) ? delvbs : delvsbs ;

                /* I DON'T KNOW */
                if (BSIM4v7entry.d_BSIM4v7modeArray [instance_ID] >= 0)
                {
                    Idtot = BSIM4v7entry.d_BSIM4v7cdRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7csubRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7cbdRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7IgidlRWArray [instance_ID] ;
                    cdhat = Idtot - BSIM4v7entry.d_BSIM4v7gbdRWArray [instance_ID] * delvbd_jct +
                            (BSIM4v7entry.d_BSIM4v7gmbsRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gbbsArray [instance_ID] + BSIM4v7entry.d_BSIM4v7ggidlbArray [instance_ID]) * delvbs +
                            (BSIM4v7entry.d_BSIM4v7gmRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gbgsArray [instance_ID] + BSIM4v7entry.d_BSIM4v7ggidlgArray [instance_ID]) * delvgs  +
                            (BSIM4v7entry.d_BSIM4v7gdsRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gbdsArray [instance_ID] + BSIM4v7entry.d_BSIM4v7ggidldArray [instance_ID]) * delvds ;

                    Ibtot = BSIM4v7entry.d_BSIM4v7cbsRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cbdRWArray [instance_ID]  - BSIM4v7entry.d_BSIM4v7IgidlRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7IgislRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7csubRWArray [instance_ID] ;
                    cbhat = Ibtot + BSIM4v7entry.d_BSIM4v7gbdRWArray [instance_ID] * delvbd_jct + BSIM4v7entry.d_BSIM4v7gbsRWArray [instance_ID] * delvbs_jct -
                            (BSIM4v7entry.d_BSIM4v7gbbsArray [instance_ID] + BSIM4v7entry.d_BSIM4v7ggidlbArray [instance_ID]) * delvbs -
                            (BSIM4v7entry.d_BSIM4v7gbgsArray [instance_ID] + BSIM4v7entry.d_BSIM4v7ggidlgArray [instance_ID]) * delvgs -
                            (BSIM4v7entry.d_BSIM4v7gbdsArray [instance_ID] + BSIM4v7entry.d_BSIM4v7ggidldArray [instance_ID] - BSIM4v7entry.d_BSIM4v7ggislsArray [instance_ID]) * delvds -
                            BSIM4v7entry.d_BSIM4v7ggislgArray [instance_ID] * delvgd - BSIM4v7entry.d_BSIM4v7ggislbArray [instance_ID] * delvbd ;

                    Igstot = BSIM4v7entry.d_BSIM4v7IgsRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7IgcsRWArray [instance_ID] ;
                    cgshat = Igstot + (BSIM4v7entry.d_BSIM4v7gIgsgArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gIgcsgArray [instance_ID]) * delvgs +
                             BSIM4v7entry.d_BSIM4v7gIgcsdArray [instance_ID] * delvds + BSIM4v7entry.d_BSIM4v7gIgcsbArray [instance_ID] * delvbs ;

                    Igdtot = BSIM4v7entry.d_BSIM4v7IgdRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7IgcdRWArray [instance_ID] ;
                    cgdhat = Igdtot + BSIM4v7entry.d_BSIM4v7gIgdgArray [instance_ID] * delvgd + BSIM4v7entry.d_BSIM4v7gIgcdgArray [instance_ID] * delvgs +
                             BSIM4v7entry.d_BSIM4v7gIgcddArray [instance_ID] * delvds + BSIM4v7entry.d_BSIM4v7gIgcdbArray [instance_ID] * delvbs ;

                    Igbtot = BSIM4v7entry.d_BSIM4v7IgbRWArray [instance_ID] ;
                    cgbhat = BSIM4v7entry.d_BSIM4v7IgbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gIgbgArray [instance_ID] * delvgs + BSIM4v7entry.d_BSIM4v7gIgbdArray [instance_ID] *
                             delvds + BSIM4v7entry.d_BSIM4v7gIgbbArray [instance_ID] * delvbs ;
                } else {
                    /* bugfix */
                    Idtot = BSIM4v7entry.d_BSIM4v7cdRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cbdRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7IgidlRWArray [instance_ID] ;
                    /* ------ */

                    cdhat = Idtot + BSIM4v7entry.d_BSIM4v7gbdRWArray [instance_ID] * delvbd_jct + BSIM4v7entry.d_BSIM4v7gmbsRWArray [instance_ID] * delvbd +
                            BSIM4v7entry.d_BSIM4v7gmRWArray [instance_ID] * delvgd - (BSIM4v7entry.d_BSIM4v7gdsRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7ggidlsArray [instance_ID]) * delvds -
                            BSIM4v7entry.d_BSIM4v7ggidlgArray [instance_ID] * delvgs - BSIM4v7entry.d_BSIM4v7ggidlbArray [instance_ID] * delvbs ;

                    Ibtot = BSIM4v7entry.d_BSIM4v7cbsRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cbdRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7IgidlRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7IgislRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7csubRWArray [instance_ID] ;
                    cbhat = Ibtot + BSIM4v7entry.d_BSIM4v7gbsRWArray [instance_ID] * delvbs_jct + BSIM4v7entry.d_BSIM4v7gbdRWArray [instance_ID] * delvbd_jct -
                            (BSIM4v7entry.d_BSIM4v7gbbsArray [instance_ID] + BSIM4v7entry.d_BSIM4v7ggislbArray [instance_ID]) * delvbd -
                            (BSIM4v7entry.d_BSIM4v7gbgsArray [instance_ID] + BSIM4v7entry.d_BSIM4v7ggislgArray [instance_ID]) * delvgd +
                            (BSIM4v7entry.d_BSIM4v7gbdsArray [instance_ID] + BSIM4v7entry.d_BSIM4v7ggisldArray [instance_ID] - BSIM4v7entry.d_BSIM4v7ggidlsArray [instance_ID]) * delvds -
                            BSIM4v7entry.d_BSIM4v7ggidlgArray [instance_ID] * delvgs - BSIM4v7entry.d_BSIM4v7ggidlbArray [instance_ID] * delvbs ;

                    Igstot = BSIM4v7entry.d_BSIM4v7IgsRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7IgcdRWArray [instance_ID] ;
                    cgshat = Igstot + BSIM4v7entry.d_BSIM4v7gIgsgArray [instance_ID] * delvgs + BSIM4v7entry.d_BSIM4v7gIgcdgArray [instance_ID] * delvgd -
                             BSIM4v7entry.d_BSIM4v7gIgcddArray [instance_ID] * delvds + BSIM4v7entry.d_BSIM4v7gIgcdbArray [instance_ID] * delvbd ;

                    Igdtot = BSIM4v7entry.d_BSIM4v7IgdRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7IgcsRWArray [instance_ID] ;
                    cgdhat = Igdtot + (BSIM4v7entry.d_BSIM4v7gIgdgArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gIgcsgArray [instance_ID]) * delvgd -
                             BSIM4v7entry.d_BSIM4v7gIgcsdArray [instance_ID] * delvds + BSIM4v7entry.d_BSIM4v7gIgcsbArray [instance_ID] * delvbd ;

                    Igbtot = BSIM4v7entry.d_BSIM4v7IgbRWArray [instance_ID] ;
                    cgbhat = BSIM4v7entry.d_BSIM4v7IgbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gIgbgArray [instance_ID] * delvgd - BSIM4v7entry.d_BSIM4v7gIgbdArray [instance_ID] * delvds +
                             BSIM4v7entry.d_BSIM4v7gIgbbArray [instance_ID] * delvbd ;
                }

                Isestot = BSIM4v7entry.d_BSIM4v7gstotArray [instance_ID] * CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 9] ;
                cseshat = Isestot + BSIM4v7entry.d_BSIM4v7gstotArray [instance_ID] * delvses + BSIM4v7entry.d_BSIM4v7gstotdArray [instance_ID] * delvds +
                          BSIM4v7entry.d_BSIM4v7gstotgArray [instance_ID] * delvgs + BSIM4v7entry.d_BSIM4v7gstotbArray [instance_ID] * delvbs ;

                Idedtot = BSIM4v7entry.d_BSIM4v7gdtotArray [instance_ID] * vdedo ;
                cdedhat = Idedtot + BSIM4v7entry.d_BSIM4v7gdtotArray [instance_ID] * delvded + BSIM4v7entry.d_BSIM4v7gdtotdArray [instance_ID] * delvds +
                          BSIM4v7entry.d_BSIM4v7gdtotgArray [instance_ID] * delvgs + BSIM4v7entry.d_BSIM4v7gdtotbArray [instance_ID] * delvbs ;

#ifndef NOBYPASS
                /* Following should be one IF statement, but some C compilers
                 * can't handle that all at once, so we split it into several
                 * successive IF's */

                /* 3 - DIVERGENT - CRITICAL */
                /* NESTED version */
                if ((!(CKTmode & MODEINITPRED)) && (CKTbypass))
                    if ((fabs (delvds) < (CKTrelTol * MAX (fabs (vds), fabs (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 3])) + CKTvoltTol)))
                        if ((fabs (delvgs) < (CKTrelTol * MAX(fabs (vgs), fabs (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 2])) +
                            CKTvoltTol)))
                            if ((fabs (delvbs) < (CKTrelTol * MAX (fabs (vbs), fabs (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 1])) +
                                CKTvoltTol)))
                                if ((fabs (delvbd) < (CKTrelTol * MAX (fabs (vbd), fabs (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID]])) +
                                    CKTvoltTol)))
                                    if ((BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 0) || (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 1) ||
                                        (fabs (delvges) < (CKTrelTol * MAX (fabs (vges),
                                        fabs (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 7])) + CKTvoltTol)))
                                        if ((BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] != 3) || (fabs (delvgms) < (CKTrelTol *
                                            MAX (fabs (vgms), fabs (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 8])) + CKTvoltTol)))
                                            if ((!BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID]) || (fabs (delvdbs) < (CKTrelTol *
                                                MAX (fabs (vdbs), fabs (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 4])) + CKTvoltTol)))
                                                if ((!BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID]) || (fabs (delvdbd) < (CKTrelTol *
                                                    MAX (fabs (vdbd), fabs (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 5])) + CKTvoltTol)))
                                                    if ((!BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID]) || (fabs (delvsbs) < (CKTrelTol *
                                                        MAX (fabs (vsbs), fabs (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 6])) +
                                                        CKTvoltTol)))
                                                        if ((!BSIM4v7rdsMod) || (fabs (delvses) < (CKTrelTol *
                                                            MAX (fabs (vses), fabs (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 9])) +
                                                            CKTvoltTol)))
                                                            if ((!BSIM4v7rdsMod) || (fabs (delvdes) < (CKTrelTol *
                                                                MAX (fabs (vdes), fabs (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 10])) +
                                                                CKTvoltTol)))
                                                                if ((fabs (cdhat - Idtot) < CKTrelTol *
                                                                    MAX (fabs (cdhat), fabs (Idtot)) + CKTabsTol))
                                                                    if ((fabs (cbhat - Ibtot) < CKTrelTol *
                                                                        MAX (fabs (cbhat), fabs (Ibtot)) + CKTabsTol))
                                                                        if ((!BSIM4v7igcMod) || ((fabs (cgshat - Igstot)
                                                                            < CKTrelTol * MAX (fabs (cgshat),
                                                                            fabs (Igstot)) + CKTabsTol)))
                                                                            if ((!BSIM4v7igcMod) || ((fabs (cgdhat -
                                                                                Igdtot) < CKTrelTol * MAX (fabs
                                                                                (cgdhat), fabs (Igdtot)) + CKTabsTol)))
                                                                                if ((!BSIM4v7igbMod) || ((fabs (cgbhat -
                                                                                    Igbtot) < CKTrelTol * MAX
                                                                                    (fabs (cgbhat), fabs (Igbtot))
                                                                                    + CKTabsTol)))
                                                                                    if ((!BSIM4v7rdsMod) ||
                                                                                        ((fabs (cseshat - Isestot) <
                                                                                        CKTrelTol * MAX (fabs (cseshat),
                                                                                        fabs (Isestot)) + CKTabsTol)))
                                                                                        if ((!BSIM4v7rdsMod) ||
                                                                                            ((fabs (cdedhat - Idedtot) <
                                                                                            CKTrelTol *
                                                                                            MAX (fabs (cdedhat),
                                                                                            fabs (Idedtot)) + CKTabsTol)))

                /* 3 - DIVERGENT - CRITICAL */
                /* NON-NESTED version
                if (((!(CKTmode & MODEINITPRED)) && (CKTbypass))
                    && ((fabs (delvds) < (CKTrelTol * MAX (fabs (vds), fabs (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 3])) + CKTvoltTol)))
                    && ((fabs (delvgs) < (CKTrelTol * MAX(fabs (vgs), fabs (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 2])) + CKTvoltTol)))
                    && ((fabs (delvbs) < (CKTrelTol * MAX (fabs (vbs), fabs (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 1])) +CKTvoltTol)))
                    && ((fabs (delvbd) < (CKTrelTol * MAX (fabs (vbd), fabs (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID]])) + CKTvoltTol)))
                    && ((BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 0) || (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 1) || (fabs (delvges) < (CKTrelTol *
                        MAX (fabs (vges), fabs (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 7])) + CKTvoltTol)))
                    && ((BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] != 3) || (fabs (delvgms) < (CKTrelTol *
                        MAX (fabs (vgms), fabs (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 8])) + CKTvoltTol)))
                    && ((!BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID]) || (fabs (delvdbs) < (CKTrelTol *
                        MAX (fabs (vdbs), fabs (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 4])) + CKTvoltTol)))
                    && ((!BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID]) || (fabs (delvdbd) < (CKTrelTol *
                        MAX (fabs (vdbd), fabs (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 5])) + CKTvoltTol)))
                    && ((!BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID]) || (fabs (delvsbs) < (CKTrelTol *
                        MAX (fabs (vsbs), fabs (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 6])) + CKTvoltTol)))
                    && ((!BSIM4v7rdsMod) || (fabs (delvses) < (CKTrelTol *
                        MAX (fabs (vses), fabs (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 9])) + CKTvoltTol)))
                    && ((!BSIM4v7rdsMod) || (fabs (delvdes) < (CKTrelTol *
                        MAX (fabs (vdes), fabs (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 10])) + CKTvoltTol)))
                    && ((fabs (cdhat - Idtot) < CKTrelTol * MAX (fabs (cdhat), fabs (Idtot)) + CKTabsTol))
                    && ((fabs (cbhat - Ibtot) < CKTrelTol * MAX (fabs (cbhat), fabs (Ibtot)) + CKTabsTol))
                    && ((!BSIM4v7igcMod) || ((fabs (cgshat - Igstot) < CKTrelTol *
                        MAX (fabs (cgshat), fabs (Igstot)) + CKTabsTol)))
                    && ((!BSIM4v7igcMod) || ((fabs (cgdhat - Igdtot) < CKTrelTol *
                        MAX (fabs (cgdhat), fabs (Igdtot)) + CKTabsTol)))
                    && ((!BSIM4v7igbMod) || ((fabs (cgbhat - Igbtot) < CKTrelTol *
                        MAX (fabs (cgbhat), fabs (Igbtot)) + CKTabsTol)))
                    && ((!BSIM4v7rdsMod) || ((fabs (cseshat - Isestot) < CKTrelTol *
                        MAX (fabs (cseshat), fabs (Isestot)) + CKTabsTol)))
                    && ((!BSIM4v7rdsMod) || ((fabs (cdedhat - Idedtot) < CKTrelTol *
                        MAX (fabs (cdedhat), fabs (Idedtot)) + CKTabsTol))))
                */
                {
                    /* It isn't possible to maintain correct indentation with the NESTED version */
                    vds = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 3] ;
                    vgs = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 2] ;
                    vbs = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 1] ;
                    vges = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 7] ;
                    vgms = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 8] ;

                    vbd = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID]] ;
                    vdbs = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 4] ;
                    vdbd = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 5] ;
                    vsbs = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 6] ;
                    vses = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 9] ;
                    vdes = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 10] ;

                    vgd = vgs - vds ;
                    vgb = vgs - vbs ;
                    vged = vges - vds ;
                    vgmd = vgms - vds ;
                    vgmb = vgms - vbs ;

                    vbs_jct = (!BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID]) ? vbs : vsbs ;
                    vbd_jct = (!BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID]) ? vbd : vdbd ;

/*** qdef should not be kept fixed even if vgs, vds & vbs has converged
****               qdef = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 27] ;
***/
                    cdrain = BSIM4v7entry.d_BSIM4v7cdRWArray [instance_ID] ;

                    if ((CKTmode & (MODETRAN | MODEAC)) || ((CKTmode & MODETRANOP) && (CKTmode & MODEUIC)))
                    {
                        ByPass = 1 ;

                        qgate = BSIM4v7entry.d_BSIM4v7qgateRWArray [instance_ID] ;
                        qbulk = BSIM4v7entry.d_BSIM4v7qbulkRWArray [instance_ID] ;
                        qdrn = BSIM4v7entry.d_BSIM4v7qdrnRWArray [instance_ID] ;
                        cgdo = BSIM4v7entry.d_BSIM4v7cgdoArray [instance_ID] ;
                        qgdo = BSIM4v7entry.d_BSIM4v7qgdoArray [instance_ID] ;
                        cgso = BSIM4v7entry.d_BSIM4v7cgsoArray [instance_ID] ;
                        qgso = BSIM4v7entry.d_BSIM4v7qgsoArray [instance_ID] ;

                        /* Unconditional jump */
                        goto line755 ;
                    } else {
                        /* Unconditional jump */
                        goto line850 ;
                    }
                }
#endif /*NOBYPASS*/

                von = BSIM4v7entry.d_BSIM4v7vonRWArray [instance_ID] ;

                /* 4 - DIVERGENT - CRITICAL */
                if (CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 3] >= 0.0)
                {
                    vgs = DEVfetlim (vgs, CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 2], von) ;
                    vds = vgs - vgd ;
                    vds = DEVlimvds (vds, CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 3]) ;
                    vgd = vgs - vds ;
                    if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 3)
                    {
                        vges = DEVfetlim (vges, CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 7], von) ;
                        vgms = DEVfetlim (vgms, CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 8], von) ;
                        vged = vges - vds ;
                        vgmd = vgms - vds ;
                    }
                    else if ((BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 1) || (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 2))
                    {
                        vges = DEVfetlim (vges, CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 7], von) ;
                        vged = vges - vds ;
                    }

                    if (BSIM4v7rdsMod)
                    {
                        vdes = DEVlimvds (vdes, CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 10]) ;
                        vses = -DEVlimvds (-vses, -(CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 9])) ;
                    }

                } else {
                    vgd = DEVfetlim (vgd, vgdo, von) ;
                    vds = vgs - vgd ;
                    vds = -DEVlimvds (-vds, -(CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 3])) ;
                    vgs = vgd + vds ;

                    if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 3)
                    {
                        vged = DEVfetlim (vged, vgedo, von) ;
                        vges = vged + vds ;
                        vgmd = DEVfetlim (vgmd, vgmdo, von) ;
                        vgms = vgmd + vds ;
                    }
                    if ((BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 1) || (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 2))
                    {
                        vged = DEVfetlim (vged, vgedo, von) ;
                        vges = vged + vds ;
                    }

                    if (BSIM4v7rdsMod)
                    {
                        vdes = -DEVlimvds (-vdes, -(CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 10])) ;
                        vses = DEVlimvds (vses, CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 9]) ;
                    }
                }

                /* 5 - DIVERGENT - CRITICAL */
                if (vds >= 0.0)
                {
                    vbs = DEVpnjlim (vbs, CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 1], CONSTvt0, BSIM4v7vcrit, &Check) ;
                    vbd = vbs - vds ;
                    if (BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID])
                    {
                        vdbs = DEVpnjlim (vdbs, CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 4], CONSTvt0, BSIM4v7vcrit, &Check1) ;
                        vdbd = vdbs - vds ;
                        vsbs = DEVpnjlim (vsbs, CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 6], CONSTvt0, BSIM4v7vcrit, &Check2) ;

                        if ((Check1 == 0) && (Check2 == 0))
                            Check = 0 ;
                        else
                            Check = 1 ;
                    }
                } else {
                    vbd = DEVpnjlim (vbd, CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID]], CONSTvt0, BSIM4v7vcrit, &Check) ;
                    vbs = vbd + vds ;
                    if (BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID])
                    {
                        vdbd = DEVpnjlim (vdbd, CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 5], CONSTvt0, BSIM4v7vcrit, &Check1) ;
                        vdbs = vdbd + vds ;
                        vsbdo = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 6] - CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 3] ;
                        vsbd = vsbs - vds ;
                        vsbd = DEVpnjlim (vsbd, vsbdo, CONSTvt0, BSIM4v7vcrit, &Check2) ;
                        vsbs = vsbd + vds ;

                        if ((Check1 == 0) && (Check2 == 0))
                        {
                            Check = 0 ;
                        } else {
                            Check = 1 ;
                        }
                    }
                }
            }

            /* Calculate DC currents and their derivatives */
            vbd = vbs - vds ;
            vgd = vgs - vds ;
            vgb = vgs - vbs ;
            vged = vges - vds ;
            vgmd = vgms - vds ;
            vgmb = vgms - vbs ;
            vdbd = vdbs - vds ;

            vbs_jct = (!BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID]) ? vbs : vsbs ;
            vbd_jct = (!BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID]) ? vbd : vdbd ;


            /* Source/drain junction diode DC model begins */
            Nvtms = BSIM4v7vtm * BSIM4v7SjctEmissionCoeff ;

/*
            if ((BSIM4v7entry.d_BSIM4v7AseffArray [instance_ID] <= 0.0) && (BSIM4v7entry.d_BSIM4v7PseffArray [instance_ID] <= 0.0))
                SourceSatCurrent = 1.0e-14 ;    //v4.7
*/
            /* POTENTIALLY DIVERGENT */
            if ((BSIM4v7entry.d_BSIM4v7AseffArray [instance_ID] <= 0.0) && (BSIM4v7entry.d_BSIM4v7PseffArray [instance_ID] <= 0.0))
            {
                SourceSatCurrent = 0.0 ;
            } else {
                SourceSatCurrent = BSIM4v7entry.d_BSIM4v7AseffArray [instance_ID] * BSIM4v7SjctTempSatCurDensity +
                                   BSIM4v7entry.d_BSIM4v7PseffArray [instance_ID] * BSIM4v7SjctSidewallTempSatCurDensity +
                                   pParam->BSIM4v7weffCJ * BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] * BSIM4v7SjctGateSidewallTempSatCurDensity ;
            }

            /* POTENTIALLY DIVERGENT */
            if (SourceSatCurrent <= 0.0)
            {
                BSIM4v7entry.d_BSIM4v7gbsRWArray [instance_ID] = CKTgmin ;
                BSIM4v7entry.d_BSIM4v7cbsRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7gbsRWArray [instance_ID] * vbs_jct ;
            } else {
                switch (BSIM4v7dioMod)
                {
                    case 0 :
                        evbs = exp (vbs_jct / Nvtms) ;
                        T1 = BSIM4v7xjbvs * exp (-(BSIM4v7bvs + vbs_jct) / Nvtms) ;
                        /* WDLiu: Magic T1 in this form; different from BSIM4v7 beta. */
                        BSIM4v7entry.d_BSIM4v7gbsRWArray [instance_ID] = SourceSatCurrent * (evbs + T1) / Nvtms + CKTgmin ;
                        BSIM4v7entry.d_BSIM4v7cbsRWArray [instance_ID] = SourceSatCurrent * (evbs + BSIM4v7entry.d_BSIM4v7XExpBVSArray [instance_ID] - T1 - 1.0) + CKTgmin * vbs_jct ;
                        break ;

                    case 1 :
                        T2 = vbs_jct / Nvtms ;
                        if (T2 < -EXP_THRESHOLD)
                        {
                            BSIM4v7entry.d_BSIM4v7gbsRWArray [instance_ID] = CKTgmin ;
                            BSIM4v7entry.d_BSIM4v7cbsRWArray [instance_ID] = SourceSatCurrent * (MIN_EXP - 1.0) + CKTgmin * vbs_jct ;
                        }
                        else if (vbs_jct <= BSIM4v7entry.d_BSIM4v7vjsmFwdArray [instance_ID])
                        {
                            evbs = exp (T2) ;
                            BSIM4v7entry.d_BSIM4v7gbsRWArray [instance_ID] = SourceSatCurrent * evbs / Nvtms + CKTgmin ;
                            BSIM4v7entry.d_BSIM4v7cbsRWArray [instance_ID] = SourceSatCurrent * (evbs - 1.0) + CKTgmin * vbs_jct ;
                        } else {
                            T0 = BSIM4v7entry.d_BSIM4v7IVjsmFwdArray [instance_ID] / Nvtms ;
                            BSIM4v7entry.d_BSIM4v7gbsRWArray [instance_ID] = T0 + CKTgmin ;
                            BSIM4v7entry.d_BSIM4v7cbsRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7IVjsmFwdArray [instance_ID] - SourceSatCurrent + T0 *
                                             (vbs_jct - BSIM4v7entry.d_BSIM4v7vjsmFwdArray [instance_ID]) + CKTgmin * vbs_jct ;
                        }
                        break ;

                    case 2 :
                        if (vbs_jct < BSIM4v7entry.d_BSIM4v7vjsmRevArray [instance_ID])
                        {
                            T0 = vbs_jct / Nvtms ;
                            if (T0 < -EXP_THRESHOLD)
                            {
                                evbs = MIN_EXP ;
                                devbs_dvb = 0.0 ;
                            }
                            else
                            {
                                evbs = exp (T0) ;
                                devbs_dvb = evbs / Nvtms ;
                            }

                            T1 = evbs - 1.0 ;
                            T2 = BSIM4v7entry.d_BSIM4v7IVjsmRevArray [instance_ID] + BSIM4v7entry.d_BSIM4v7SslpRevArray [instance_ID] * (vbs_jct - BSIM4v7entry.d_BSIM4v7vjsmRevArray [instance_ID]) ;
                            BSIM4v7entry.d_BSIM4v7gbsRWArray [instance_ID] = devbs_dvb * T2 + T1 * BSIM4v7entry.d_BSIM4v7SslpRevArray [instance_ID] + CKTgmin ;
                            BSIM4v7entry.d_BSIM4v7cbsRWArray [instance_ID] = T1 * T2 + CKTgmin * vbs_jct ;
                        }
                        else if (vbs_jct <= BSIM4v7entry.d_BSIM4v7vjsmFwdArray [instance_ID])
                        {
                            T0 = vbs_jct / Nvtms ;
                            if (T0 < -EXP_THRESHOLD)
                            {
                                evbs = MIN_EXP ;
                                devbs_dvb = 0.0 ;
                            }
                            else
                            {
                                evbs = exp (T0) ;
                                devbs_dvb = evbs / Nvtms ;
                            }

                            T1 = (BSIM4v7bvs + vbs_jct) / Nvtms ;
                            if (T1 > EXP_THRESHOLD)
                            {
                                T2 = MIN_EXP ;
                                T3 = 0.0 ;
                            }
                            else
                            {
                                T2 = exp (-T1) ;
                                T3 = -T2 /Nvtms ;
                            }
                            BSIM4v7entry.d_BSIM4v7gbsRWArray [instance_ID] = SourceSatCurrent * (devbs_dvb - BSIM4v7xjbvs * T3) + CKTgmin ;
                            BSIM4v7entry.d_BSIM4v7cbsRWArray [instance_ID] = SourceSatCurrent * (evbs + BSIM4v7entry.d_BSIM4v7XExpBVSArray [instance_ID] - 1.0 -
                                             BSIM4v7xjbvs * T2) + CKTgmin * vbs_jct ;
                        }
                        else
                        {
                            BSIM4v7entry.d_BSIM4v7gbsRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7SslpFwdArray [instance_ID] + CKTgmin ;
                            BSIM4v7entry.d_BSIM4v7cbsRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7IVjsmFwdArray [instance_ID] + BSIM4v7entry.d_BSIM4v7SslpFwdArray [instance_ID] *
                                             (vbs_jct - BSIM4v7entry.d_BSIM4v7vjsmFwdArray [instance_ID]) + CKTgmin * vbs_jct ;
                        }
                        break ;

                    default: break ;
                }
            }

            Nvtmd = BSIM4v7vtm * BSIM4v7DjctEmissionCoeff ;
/*
            if ((BSIM4v7entry.d_BSIM4v7AdeffArray [instance_ID] <= 0.0) && (BSIM4v7entry.d_BSIM4v7PdeffArray [instance_ID] <= 0.0))
                DrainSatCurrent = 1.0e-14 ;    //v4.7
*/
            /* POTENTIALLY DIVERGENT */
            if ((BSIM4v7entry.d_BSIM4v7AdeffArray [instance_ID] <= 0.0) && (BSIM4v7entry.d_BSIM4v7PdeffArray [instance_ID] <= 0.0))
            {
                DrainSatCurrent = 0.0 ;
            } else {
                DrainSatCurrent = BSIM4v7entry.d_BSIM4v7AdeffArray [instance_ID] * BSIM4v7DjctTempSatCurDensity +
                                  BSIM4v7entry.d_BSIM4v7PdeffArray [instance_ID] * BSIM4v7DjctSidewallTempSatCurDensity +
                                  pParam->BSIM4v7weffCJ * BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] * BSIM4v7DjctGateSidewallTempSatCurDensity ;
            }

            /* POTENTIALLY DIVERGENT */
            if (DrainSatCurrent <= 0.0)
            {
                BSIM4v7entry.d_BSIM4v7gbdRWArray [instance_ID] = CKTgmin ;
                BSIM4v7entry.d_BSIM4v7cbdRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7gbdRWArray [instance_ID] * vbd_jct ;
            } else {
                switch (BSIM4v7dioMod)
                {
                    case 0 :
                        evbd = exp (vbd_jct / Nvtmd) ;
                        T1 = BSIM4v7xjbvd * exp (-(BSIM4v7bvd + vbd_jct) / Nvtmd) ;
                        /* WDLiu: Magic T1 in this form; different from BSIM4v7 beta. */
                        BSIM4v7entry.d_BSIM4v7gbdRWArray [instance_ID] = DrainSatCurrent * (evbd + T1) / Nvtmd + CKTgmin ;
                        BSIM4v7entry.d_BSIM4v7cbdRWArray [instance_ID] = DrainSatCurrent * (evbd + BSIM4v7entry.d_BSIM4v7XExpBVDArray [instance_ID] - T1 - 1.0) + CKTgmin * vbd_jct ;
                        break ;

                    case 1 :
                        T2 = vbd_jct / Nvtmd ;
                        if (T2 < -EXP_THRESHOLD)
                        {
                            BSIM4v7entry.d_BSIM4v7gbdRWArray [instance_ID] = CKTgmin ;
                            BSIM4v7entry.d_BSIM4v7cbdRWArray [instance_ID] = DrainSatCurrent * (MIN_EXP - 1.0) + CKTgmin * vbd_jct ;
                        }
                        else if (vbd_jct <= BSIM4v7entry.d_BSIM4v7vjdmFwdArray [instance_ID])
                        {
                            evbd = exp (T2) ;
                            BSIM4v7entry.d_BSIM4v7gbdRWArray [instance_ID] = DrainSatCurrent * evbd / Nvtmd + CKTgmin ;
                            BSIM4v7entry.d_BSIM4v7cbdRWArray [instance_ID] = DrainSatCurrent * (evbd - 1.0) + CKTgmin * vbd_jct ;
                        } else {
                            T0 = BSIM4v7entry.d_BSIM4v7IVjdmFwdArray [instance_ID] / Nvtmd ;
                            BSIM4v7entry.d_BSIM4v7gbdRWArray [instance_ID] = T0 + CKTgmin ;
                            BSIM4v7entry.d_BSIM4v7cbdRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7IVjdmFwdArray [instance_ID] - DrainSatCurrent +
                                             T0 * (vbd_jct - BSIM4v7entry.d_BSIM4v7vjdmFwdArray [instance_ID]) + CKTgmin * vbd_jct ;
                        }
                        break ;

                    case 2 :
                        if (vbd_jct < BSIM4v7entry.d_BSIM4v7vjdmRevArray [instance_ID])
                        {
                            T0 = vbd_jct / Nvtmd ;
                            if (T0 < -EXP_THRESHOLD)
                            {
                                evbd = MIN_EXP ;
                                devbd_dvb = 0.0 ;
                            }
                            else
                            {
                                evbd = exp (T0) ;
                                devbd_dvb = evbd / Nvtmd ;
                            }

                            T1 = evbd - 1.0 ;
                            T2 = BSIM4v7entry.d_BSIM4v7IVjdmRevArray [instance_ID] + BSIM4v7entry.d_BSIM4v7DslpRevArray [instance_ID] * (vbd_jct - BSIM4v7entry.d_BSIM4v7vjdmRevArray [instance_ID]) ;
                            BSIM4v7entry.d_BSIM4v7gbdRWArray [instance_ID] = devbd_dvb * T2 + T1 * BSIM4v7entry.d_BSIM4v7DslpRevArray [instance_ID] + CKTgmin ;
                            BSIM4v7entry.d_BSIM4v7cbdRWArray [instance_ID] = T1 * T2 + CKTgmin * vbd_jct ;
                        }
                        else if (vbd_jct <= BSIM4v7entry.d_BSIM4v7vjdmFwdArray [instance_ID])
                        {
                            T0 = vbd_jct / Nvtmd ;
                            if (T0 < -EXP_THRESHOLD)
                            {
                                evbd = MIN_EXP ;
                                devbd_dvb = 0.0 ;
                            }
                            else
                            {
                                evbd = exp (T0) ;
                                devbd_dvb = evbd / Nvtmd ;
                            }

                            T1 = (BSIM4v7bvd + vbd_jct) / Nvtmd ;
                            if (T1 > EXP_THRESHOLD)
                            {
                                T2 = MIN_EXP ;
                                T3 = 0.0 ;
                            }
                            else
                            {
                                T2 = exp (-T1) ;
                                T3 = -T2 /Nvtmd ;
                            }
                            BSIM4v7entry.d_BSIM4v7gbdRWArray [instance_ID] = DrainSatCurrent * (devbd_dvb - BSIM4v7xjbvd * T3) + CKTgmin ;
                            BSIM4v7entry.d_BSIM4v7cbdRWArray [instance_ID] = DrainSatCurrent * (evbd + BSIM4v7entry.d_BSIM4v7XExpBVDArray [instance_ID] - 1.0 -
                                             BSIM4v7xjbvd * T2) + CKTgmin * vbd_jct ;
                        }
                        else
                        {
                            BSIM4v7entry.d_BSIM4v7gbdRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7DslpFwdArray [instance_ID] + CKTgmin ;
                            BSIM4v7entry.d_BSIM4v7cbdRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7IVjdmFwdArray [instance_ID] + BSIM4v7entry.d_BSIM4v7DslpFwdArray [instance_ID] *
                                             (vbd_jct - BSIM4v7entry.d_BSIM4v7vjdmFwdArray [instance_ID]) + CKTgmin * vbd_jct ;
                        }
                        break ;

                    default: break ;
                }
            }


            /* trap-assisted tunneling and recombination current for reverse bias  */
            Nvtmrssws = BSIM4v7vtm0 * BSIM4v7njtsswstemp ;
            Nvtmrsswgs = BSIM4v7vtm0 * BSIM4v7njtsswgstemp ;
            Nvtmrss = BSIM4v7vtm0 * BSIM4v7njtsstemp ;
            Nvtmrsswd = BSIM4v7vtm0 * BSIM4v7njtsswdtemp ;
            Nvtmrsswgd = BSIM4v7vtm0 * BSIM4v7njtsswgdtemp ;
            Nvtmrsd = BSIM4v7vtm0 * BSIM4v7njtsdtemp ;

            /* POSSIBLE DIVERGENT */
            if ((BSIM4v7vtss - vbs_jct) < (BSIM4v7vtss * 1e-3))
            {
                T9 = 1.0e3 ;
                T0 = - vbs_jct / Nvtmrss * T9 ;
                DEXP (T0, T1, T10) ;
                dT1_dVb = T10 / Nvtmrss * T9 ;
            } else {
                T9 = 1.0 / (BSIM4v7vtss - vbs_jct) ;
                T0 = -vbs_jct / Nvtmrss * BSIM4v7vtss * T9 ;
                dT0_dVb = BSIM4v7vtss / Nvtmrss * (T9 + vbs_jct * T9 * T9) ;
                DEXP (T0, T1, T10) ;
                dT1_dVb = T10 * dT0_dVb ;
            }

            if ((BSIM4v7vtsd - vbd_jct) < (BSIM4v7vtsd * 1e-3))
            {
                T9 = 1.0e3 ;
                T0 = -vbd_jct / Nvtmrsd * T9 ;
                DEXP (T0, T2, T10) ;
                dT2_dVb = T10 / Nvtmrsd * T9 ;
            } else {
                T9 = 1.0 / (BSIM4v7vtsd - vbd_jct) ;
                T0 = -vbd_jct / Nvtmrsd * BSIM4v7vtsd * T9 ;
                dT0_dVb = BSIM4v7vtsd / Nvtmrsd * (T9 + vbd_jct * T9 * T9) ;
                DEXP (T0, T2, T10) ;
                dT2_dVb = T10 * dT0_dVb ;
            }

            /* POSSIBLE DIVERGENT */
            if ((BSIM4v7vtssws - vbs_jct) < (BSIM4v7vtssws * 1e-3))
            {
                T9 = 1.0e3 ;
                T0 = -vbs_jct / Nvtmrssws * T9 ;
                DEXP (T0, T3, T10) ;
                dT3_dVb = T10 / Nvtmrssws * T9 ;
            } else {
                T9 = 1.0 / (BSIM4v7vtssws - vbs_jct) ;
                T0 = -vbs_jct / Nvtmrssws * BSIM4v7vtssws * T9 ;
                dT0_dVb = BSIM4v7vtssws / Nvtmrssws * (T9 + vbs_jct * T9 * T9) ;
                DEXP (T0, T3, T10) ;
                dT3_dVb = T10 * dT0_dVb ;
            }

            /* POSSIBLE DIVERGENT */
            if ((BSIM4v7vtsswd - vbd_jct) < (BSIM4v7vtsswd * 1e-3))
            {
                T9 = 1.0e3 ;
                T0 = -vbd_jct / Nvtmrsswd * T9 ;
                DEXP (T0, T4, T10) ;
                dT4_dVb = T10 / Nvtmrsswd * T9 ;
            } else {
                T9 = 1.0 / (BSIM4v7vtsswd - vbd_jct) ;
                T0 = -vbd_jct / Nvtmrsswd * BSIM4v7vtsswd * T9 ;
                dT0_dVb = BSIM4v7vtsswd / Nvtmrsswd * (T9 + vbd_jct * T9 * T9) ;
                DEXP(T0, T4, T10) ;
                dT4_dVb = T10 * dT0_dVb ;
            }

            /* POSSIBLE DIVERGENT */
            if ((BSIM4v7vtsswgs - vbs_jct) < (BSIM4v7vtsswgs * 1e-3))
            {
                T9 = 1.0e3 ;
                T0 = -vbs_jct / Nvtmrsswgs * T9 ;
                DEXP (T0, T5, T10) ;
                dT5_dVb = T10 / Nvtmrsswgs * T9 ;
            } else {
                T9 = 1.0 / (BSIM4v7vtsswgs - vbs_jct) ;
                T0 = -vbs_jct / Nvtmrsswgs * BSIM4v7vtsswgs * T9 ;
                dT0_dVb = BSIM4v7vtsswgs / Nvtmrsswgs * (T9 + vbs_jct * T9 * T9) ;
                DEXP(T0, T5, T10) ;
                dT5_dVb = T10 * dT0_dVb ;
            }

            /* POSSIBLE DIVERGENT */
            if ((BSIM4v7vtsswgd - vbd_jct) < (BSIM4v7vtsswgd * 1e-3))
            {
                T9 = 1.0e3 ;
                T0 = -vbd_jct / Nvtmrsswgd * T9 ;
                DEXP (T0, T6, T10) ;
                dT6_dVb = T10 / Nvtmrsswgd * T9 ;
            } else {
                T9 = 1.0 / (BSIM4v7vtsswgd - vbd_jct) ;
                T0 = -vbd_jct / Nvtmrsswgd * BSIM4v7vtsswgd * T9 ;
                dT0_dVb = BSIM4v7vtsswgd / Nvtmrsswgd * (T9 + vbd_jct * T9 * T9) ;
                DEXP (T0, T6, T10) ;
                dT6_dVb = T10 * dT0_dVb ;
            }

            BSIM4v7entry.d_BSIM4v7gbsRWArray [instance_ID] += BSIM4v7entry.d_BSIM4v7SjctTempRevSatCurArray [instance_ID] * dT1_dVb + BSIM4v7entry.d_BSIM4v7SswTempRevSatCurArray [instance_ID] * dT3_dVb +
                              BSIM4v7entry.d_BSIM4v7SswgTempRevSatCurArray [instance_ID] * dT5_dVb ;
            BSIM4v7entry.d_BSIM4v7cbsRWArray [instance_ID] -= BSIM4v7entry.d_BSIM4v7SjctTempRevSatCurArray [instance_ID] * (T1 - 1.0) + BSIM4v7entry.d_BSIM4v7SswTempRevSatCurArray [instance_ID] * (T3 - 1.0) +
                              BSIM4v7entry.d_BSIM4v7SswgTempRevSatCurArray [instance_ID] * (T5 - 1.0) ;
            BSIM4v7entry.d_BSIM4v7gbdRWArray [instance_ID] += BSIM4v7entry.d_BSIM4v7DjctTempRevSatCurArray [instance_ID] * dT2_dVb + BSIM4v7entry.d_BSIM4v7DswTempRevSatCurArray [instance_ID] * dT4_dVb +
                              BSIM4v7entry.d_BSIM4v7DswgTempRevSatCurArray [instance_ID] * dT6_dVb ;
            BSIM4v7entry.d_BSIM4v7cbdRWArray [instance_ID] -= BSIM4v7entry.d_BSIM4v7DjctTempRevSatCurArray [instance_ID] * (T2 - 1.0) + BSIM4v7entry.d_BSIM4v7DswTempRevSatCurArray [instance_ID] * (T4 - 1.0) +
                              BSIM4v7entry.d_BSIM4v7DswgTempRevSatCurArray [instance_ID] * (T6 - 1.0) ;
            /* End of diode DC model */


            /* 6 - DIVERGENT */
            if (vds >= 0.0)
            {
                BSIM4v7entry.d_BSIM4v7modeArray [instance_ID] = 1 ;
                Vds = vds ;
                Vbs = vbs ;

                /* WDLiu: for GIDL */
            } else {
                BSIM4v7entry.d_BSIM4v7modeArray [instance_ID] = -1 ;
                Vds = -vds ;
                Vbs = vbd ;
            }


            /* 3 - non-divergent */
            /* dunga */
            if (BSIM4v7mtrlMod)
            {
                epsrox = 3.9 ;
                toxe = BSIM4v7eot ;
                epssub = EPS0 * BSIM4v7epsrsub ;
            } else {
                epsrox = BSIM4v7epsrox ;
                toxe = BSIM4v7toxe ;
                epssub = EPSSI ;
            }


            T0 = Vbs - BSIM4v7entry.d_BSIM4v7vbscArray [instance_ID] - 0.001 ;
            T1 = sqrt (T0 * T0 - 0.004 * BSIM4v7entry.d_BSIM4v7vbscArray [instance_ID]) ;

            /* 7 - DIVERGENT */
            if (T0 >= 0.0)
            {
                Vbseff = BSIM4v7entry.d_BSIM4v7vbscArray [instance_ID] + 0.5 * (T0 + T1) ;
                dVbseff_dVb = 0.5 * (1.0 + T0 / T1) ;
            } else {
                T2 = -0.002 / (T1 - T0) ;
                Vbseff = BSIM4v7entry.d_BSIM4v7vbscArray [instance_ID] * (1.0 + T2) ;
                dVbseff_dVb = T2 * BSIM4v7entry.d_BSIM4v7vbscArray [instance_ID] / T1 ;
            }


            /* JX: Correction to forward body bias  */
            T9 = 0.95 * pParam->BSIM4v7phi ;
            T0 = T9 - Vbseff - 0.001 ;
            T1 = sqrt (T0 * T0 + 0.004 * T9) ;
            Vbseff = T9 - 0.5 * (T0 + T1) ;
            dVbseff_dVb *= 0.5 * (1.0 + T0 / T1) ;
            Phis = pParam->BSIM4v7phi - Vbseff ;
            dPhis_dVb = -1.0 ;
            sqrtPhis = sqrt (Phis) ;
            dsqrtPhis_dVb = -0.5 / sqrtPhis ;

            Xdep = pParam->BSIM4v7Xdep0 * sqrtPhis / pParam->BSIM4v7sqrtPhi ;
            dXdep_dVb = (pParam->BSIM4v7Xdep0 / pParam->BSIM4v7sqrtPhi) * dsqrtPhis_dVb ;

            Leff = pParam->BSIM4v7leff ;
            Vtm = BSIM4v7vtm ;
            Vtm0 = BSIM4v7vtm0 ;


            /* Vth Calculation */
            T3 = sqrt (Xdep) ;
            V0 = pParam->BSIM4v7vbi - pParam->BSIM4v7phi ;
            T0 = pParam->BSIM4v7dvt2 * Vbseff ;

            /* 8 - DIVERGENT */
            if (T0 >= - 0.5)
            {
                T1 = 1.0 + T0 ;
                T2 = pParam->BSIM4v7dvt2 ;
            } else {
                T4 = 1.0 / (3.0 + 8.0 * T0) ;
                T1 = (1.0 + 3.0 * T0) * T4 ;
                T2 = pParam->BSIM4v7dvt2 * T4 * T4 ;
            }
            lt1 = BSIM4v7factor1 * T3 * T1 ;
            dlt1_dVb = BSIM4v7factor1 * (0.5 / T3 * T1 * dXdep_dVb + T3 * T2) ;
            T0 = pParam->BSIM4v7dvt2w * Vbseff ;

            /* 9 - DIVERGENT */
            if (T0 >= - 0.5)
            {
                T1 = 1.0 + T0 ;
                T2 = pParam->BSIM4v7dvt2w ;
            } else {
                T4 = 1.0 / (3.0 + 8.0 * T0) ;
                T1 = (1.0 + 3.0 * T0) * T4 ;
                T2 = pParam->BSIM4v7dvt2w * T4 * T4 ;
            }
            ltw = BSIM4v7factor1 * T3 * T1 ;
            dltw_dVb = BSIM4v7factor1 * (0.5 / T3 * T1 * dXdep_dVb + T3 * T2) ;
            T0 = pParam->BSIM4v7dvt1 * Leff / lt1 ;

            /* 10 - DIVERGENT */
            if (T0 < EXP_THRESHOLD)
            {
                T1 = exp (T0) ;
                T2 = T1 - 1.0 ;
                T3 = T2 * T2 ;
                T4 = T3 + 2.0 * T1 * MIN_EXP ;
                Theta0 = T1 / T4 ;
                dT1_dVb = -T0 * T1 * dlt1_dVb / lt1 ;
                dTheta0_dVb = dT1_dVb * (T4 - 2.0 * T1 * (T2 + MIN_EXP)) / T4 / T4 ;
            } else {
                Theta0 = 1.0 / (MAX_EXP - 2.0) ; /* 3.0 * MIN_EXP omitted */
                dTheta0_dVb = 0.0 ;
            }
            BSIM4v7entry.d_BSIM4v7thetavthArray [instance_ID] = pParam->BSIM4v7dvt0 * Theta0 ;
            Delt_vth = BSIM4v7entry.d_BSIM4v7thetavthArray [instance_ID] * V0 ;
            dDelt_vth_dVb = pParam->BSIM4v7dvt0 * dTheta0_dVb * V0 ;
            T0 = pParam->BSIM4v7dvt1w * pParam->BSIM4v7weff * Leff / ltw ;

            /* 11 - DIVERGENT */
            if (T0 < EXP_THRESHOLD)
            {
                T1 = exp (T0) ;
                T2 = T1 - 1.0 ;
                T3 = T2 * T2 ;
                T4 = T3 + 2.0 * T1 * MIN_EXP ;
                T5 = T1 / T4 ;
                dT1_dVb = -T0 * T1 * dltw_dVb / ltw ;
                dT5_dVb = dT1_dVb * (T4 - 2.0 * T1 * (T2 + MIN_EXP)) / T4 / T4 ;
            } else {
                T5 = 1.0 / (MAX_EXP - 2.0) ; /* 3.0 * MIN_EXP omitted */
                dT5_dVb = 0.0 ;
            }
            T0 = pParam->BSIM4v7dvt0w * T5 ;
            T2 = T0 * V0 ;
            dT2_dVb = pParam->BSIM4v7dvt0w * dT5_dVb * V0 ;
            TempRatio =  CKTtemp / BSIM4v7tnom - 1.0 ;
            T0 = sqrt (1.0 + pParam->BSIM4v7lpe0 / Leff) ;
            T1 = pParam->BSIM4v7k1ox * (T0 - 1.0) * pParam->BSIM4v7sqrtPhi + (pParam->BSIM4v7kt1 +
                 pParam->BSIM4v7kt1l / Leff + pParam->BSIM4v7kt2 * Vbseff) * TempRatio ;
            Vth_NarrowW = toxe * pParam->BSIM4v7phi / (pParam->BSIM4v7weff + pParam->BSIM4v7w0) ;
            T3 = BSIM4v7entry.d_BSIM4v7eta0Array [instance_ID] + pParam->BSIM4v7etab * Vbseff ;

            /* 12 - DIVERGENT */
            if (T3 < 1.0e-4)
            {
                T9 = 1.0 / (3.0 - 2.0e4 * T3) ;
                T3 = (2.0e-4 - T3) * T9 ;
                T4 = T9 * T9 ;
            } else {
                T4 = 1.0 ;
            }

            dDIBL_Sft_dVd = T3 * pParam->BSIM4v7theta0vb0 ;
            DIBL_Sft = dDIBL_Sft_dVd * Vds ;
            Lpe_Vb = sqrt (1.0 + pParam->BSIM4v7lpeb / Leff) ;
            Vth = BSIM4v7type * BSIM4v7entry.d_BSIM4v7vth0Array [instance_ID] + (pParam->BSIM4v7k1ox * sqrtPhis - pParam->BSIM4v7k1 *
                  pParam->BSIM4v7sqrtPhi) * Lpe_Vb - BSIM4v7entry.d_BSIM4v7k2oxArray [instance_ID] * Vbseff - Delt_vth - T2 +
                  (pParam->BSIM4v7k3 + pParam->BSIM4v7k3b * Vbseff) * Vth_NarrowW + T1 - DIBL_Sft ;
            dVth_dVb = Lpe_Vb * pParam->BSIM4v7k1ox * dsqrtPhis_dVb - BSIM4v7entry.d_BSIM4v7k2oxArray [instance_ID] - dDelt_vth_dVb -
                       dT2_dVb + pParam->BSIM4v7k3b * Vth_NarrowW - pParam->BSIM4v7etab * Vds *
                       pParam->BSIM4v7theta0vb0 * T4 + pParam->BSIM4v7kt2 * TempRatio ;
            dVth_dVd = -dDIBL_Sft_dVd ;


            /* Calculate n */
            tmp1 = epssub / Xdep ;
            BSIM4v7entry.d_BSIM4v7nstarArray [instance_ID] = BSIM4v7vtm / Charge_q * (BSIM4v7coxe + tmp1 + pParam->BSIM4v7cit) ;
            tmp2 = pParam->BSIM4v7nfactor * tmp1 ;
            tmp3 = pParam->BSIM4v7cdsc + pParam->BSIM4v7cdscb * Vbseff + pParam->BSIM4v7cdscd * Vds ;
            tmp4 = (tmp2 + tmp3 * Theta0 + pParam->BSIM4v7cit) / BSIM4v7coxe ;

            /* 13 - DIVERGENT */
            if (tmp4 >= -0.5)
            {
                n = 1.0 + tmp4 ;
                dn_dVb = (-tmp2 / Xdep * dXdep_dVb + tmp3 *
                         dTheta0_dVb + pParam->BSIM4v7cdscb * Theta0) / BSIM4v7coxe ;
                dn_dVd = pParam->BSIM4v7cdscd * Theta0 / BSIM4v7coxe ;
            } else {
                T0 = 1.0 / (3.0 + 8.0 * tmp4) ;
                n = (1.0 + 3.0 * tmp4) * T0 ;
                T0 *= T0 ;
                dn_dVb = (-tmp2 / Xdep * dXdep_dVb + tmp3 *
                         dTheta0_dVb + pParam->BSIM4v7cdscb * Theta0) / BSIM4v7coxe * T0 ;
                dn_dVd = pParam->BSIM4v7cdscd * Theta0 / BSIM4v7coxe * T0 ;
            }


            /* Vth correction for Pocket implant */
            /* 14 - DIVERGENT */
            if (pParam->BSIM4v7dvtp0 > 0.0)
            {
                T0 = -pParam->BSIM4v7dvtp1 * Vds ;
                if (T0 < -EXP_THRESHOLD)
                {
                    T2 = MIN_EXP ;
                    dT2_dVd = 0.0 ;
                }
                else
                {
                    T2 = exp (T0) ;
                    dT2_dVd = -pParam->BSIM4v7dvtp1 * T2 ;
                }
                T3 = Leff + pParam->BSIM4v7dvtp0 * (1.0 + T2) ;
                dT3_dVd = pParam->BSIM4v7dvtp0 * dT2_dVd ;

                if (BSIM4v7tempMod < 2)
                {
                    T4 = Vtm * log (Leff / T3) ;
                    dT4_dVd = -Vtm * dT3_dVd / T3 ;
                }
                else
                {
                    T4 = BSIM4v7vtm0 * log (Leff / T3) ;
                    dT4_dVd = -BSIM4v7vtm0 * dT3_dVd / T3 ;
                }
                dDITS_Sft_dVd = dn_dVd * T4 + n * dT4_dVd ;
                dDITS_Sft_dVb = T4 * dn_dVb ;
                Vth -= n * T4 ;
                dVth_dVd -= dDITS_Sft_dVd ;
                dVth_dVb -= dDITS_Sft_dVb ;
            }

            /* v4.7 DITS_SFT2  */
            /* 15 - DIVERGENT */
            if ((pParam->BSIM4v7dvtp4  == 0.0) || (pParam->BSIM4v7dvtp2factor == 0.0))
            {
                T0 = 0.0 ;
                DITS_Sft2 = 0.0 ;
            } else {
                T1 = 2.0 * pParam->BSIM4v7dvtp4 * Vds ;
                DEXP (T1, T0, T10) ;
                DITS_Sft2 = pParam->BSIM4v7dvtp2factor * (T0 - 1) / (T0 + 1) ;
                dDITS_Sft2_dVd = pParam->BSIM4v7dvtp2factor * pParam->BSIM4v7dvtp4 * 4.0 * T10 / ((T0+1) * (T0+1)) ;
                Vth -= DITS_Sft2 ;
                dVth_dVd -= dDITS_Sft2_dVd ;
            }
            BSIM4v7entry.d_BSIM4v7vonRWArray [instance_ID] = Vth ;


            /* Poly Gate Si Depletion Effect */
            T0 = BSIM4v7entry.d_BSIM4v7vfbArray [instance_ID] + pParam->BSIM4v7phi;

            /* 16 - DIVERGENT */
            if (BSIM4v7mtrlMod == 0)
                T1 = EPSSI ;
            else
                T1 = BSIM4v7epsrgate * EPS0 ;

            BSIM4v7polyDepletion (T0, pParam->BSIM4v7ngate, T1, BSIM4v7coxe, vgs, &vgs_eff, &dvgs_eff_dvg) ;

            BSIM4v7polyDepletion (T0, pParam->BSIM4v7ngate, T1, BSIM4v7coxe, vgd, &vgd_eff, &dvgd_eff_dvg) ;

            /* 17 - DIVERGENT */
            if (BSIM4v7entry.d_BSIM4v7modeArray [instance_ID] > 0)
            {
                Vgs_eff = vgs_eff ;
                dVgs_eff_dVg = dvgs_eff_dvg ;
            } else {
                Vgs_eff = vgd_eff ;
                dVgs_eff_dVg = dvgd_eff_dvg ;
            }
            BSIM4v7entry.d_BSIM4v7vgs_effArray [instance_ID] = vgs_eff ;
            BSIM4v7entry.d_BSIM4v7vgd_effArray [instance_ID] = vgd_eff ;
            BSIM4v7entry.d_BSIM4v7dvgs_eff_dvgArray [instance_ID] = dvgs_eff_dvg ;
            BSIM4v7entry.d_BSIM4v7dvgd_eff_dvgArray [instance_ID] = dvgd_eff_dvg ;
            Vgst = Vgs_eff - Vth ;


            /* Calculate Vgsteff */
            T0 = n * Vtm ;
            T1 = pParam->BSIM4v7mstar * Vgst ;
            T2 = T1 / T0 ;

            /* 18 - DIVERGENT */
            if (T2 > EXP_THRESHOLD)
            {
                T10 = T1 ;
                dT10_dVg = pParam->BSIM4v7mstar * dVgs_eff_dVg ;
                dT10_dVd = -dVth_dVd * pParam->BSIM4v7mstar ;
                dT10_dVb = -dVth_dVb * pParam->BSIM4v7mstar ;
            }
            else if (T2 < -EXP_THRESHOLD)
            {
                T10 = Vtm * log (1.0 + MIN_EXP) ;
                dT10_dVg = 0.0 ;
                dT10_dVd = T10 * dn_dVd ;
                dT10_dVb = T10 * dn_dVb ;
                T10 *= n ;
            } else {
                ExpVgst = exp (T2) ;
                T3 = Vtm * log (1.0 + ExpVgst) ;
                T10 = n * T3 ;
                dT10_dVg = pParam->BSIM4v7mstar * ExpVgst / (1.0 + ExpVgst) ;
                dT10_dVb = T3 * dn_dVb - dT10_dVg * (dVth_dVb + Vgst * dn_dVb / n) ;
                dT10_dVd = T3 * dn_dVd - dT10_dVg * (dVth_dVd + Vgst * dn_dVd / n) ;
                dT10_dVg *= dVgs_eff_dVg ;
            }

            T1 = pParam->BSIM4v7voffcbn - (1.0 - pParam->BSIM4v7mstar) * Vgst ;
            T2 = T1 / T0 ;

            /* 19 - DIVERGENT */
            if (T2 < -EXP_THRESHOLD)
            {
                T3 = BSIM4v7coxe * MIN_EXP / pParam->BSIM4v7cdep0 ;
                T9 = pParam->BSIM4v7mstar + T3 * n ;
                dT9_dVg = 0.0 ;
                dT9_dVd = dn_dVd * T3 ;
                dT9_dVb = dn_dVb * T3 ;
            }
            else if (T2 > EXP_THRESHOLD)
            {
                T3 = BSIM4v7coxe * MAX_EXP / pParam->BSIM4v7cdep0 ;
                T9 = pParam->BSIM4v7mstar + T3 * n ;
                dT9_dVg = 0.0 ;
                dT9_dVd = dn_dVd * T3 ;
                dT9_dVb = dn_dVb * T3 ;
            } else {
                ExpVgst = exp (T2) ;
                T3 = BSIM4v7coxe / pParam->BSIM4v7cdep0 ;
                T4 = T3 * ExpVgst ;
                T5 = T1 * T4 / T0 ;
                T9 = pParam->BSIM4v7mstar + n * T4 ;
                dT9_dVg = T3 * (pParam->BSIM4v7mstar - 1.0) * ExpVgst / Vtm ;
                dT9_dVb = T4 * dn_dVb - dT9_dVg * dVth_dVb - T5 * dn_dVb ;
                dT9_dVd = T4 * dn_dVd - dT9_dVg * dVth_dVd - T5 * dn_dVd ;
                dT9_dVg *= dVgs_eff_dVg ;
            }
            BSIM4v7entry.d_BSIM4v7VgsteffArray [instance_ID] = Vgsteff = T10 / T9 ;
            T11 = T9 * T9 ;
            dVgsteff_dVg = (T9 * dT10_dVg - T10 * dT9_dVg) / T11 ;
            dVgsteff_dVd = (T9 * dT10_dVd - T10 * dT9_dVd) / T11 ;
            dVgsteff_dVb = (T9 * dT10_dVb - T10 * dT9_dVb) / T11 ;


            /* Calculate Effective Channel Geometry */
            T9 = sqrtPhis - pParam->BSIM4v7sqrtPhi;
            Weff = pParam->BSIM4v7weff - 2.0 * (pParam->BSIM4v7dwg * Vgsteff + pParam->BSIM4v7dwb * T9) ;
            dWeff_dVg = -2.0 * pParam->BSIM4v7dwg ;
            dWeff_dVb = -2.0 * pParam->BSIM4v7dwb * dsqrtPhis_dVb ;

            /* 20 - DIVERGENT */
            if (Weff < 2.0e-8) /* to avoid the discontinuity problem due to Weff*/
            {
                T0 = 1.0 / (6.0e-8 - 2.0 * Weff) ;
                Weff = 2.0e-8 * (4.0e-8 - Weff) * T0 ;
                T0 *= T0 * 4.0e-16 ;
                dWeff_dVg *= T0 ;
                dWeff_dVb *= T0 ;
            }

            /* 21 - DIVERGENT */
            if (BSIM4v7rdsMod == 1)
                Rds = dRds_dVg = dRds_dVb = 0.0 ;
            else
            {
                T0 = 1.0 + pParam->BSIM4v7prwg * Vgsteff ;
                dT0_dVg = -pParam->BSIM4v7prwg / T0 / T0 ;
                T1 = pParam->BSIM4v7prwb * T9 ;
                dT1_dVb = pParam->BSIM4v7prwb * dsqrtPhis_dVb ;
                T2 = 1.0 / T0 + T1 ;
                T3 = T2 + sqrt (T2 * T2 + 0.01) ; /* 0.01 = 4.0 * 0.05 * 0.05 */
                dT3_dVg = 1.0 + T2 / (T3 - T2) ;
                dT3_dVb = dT3_dVg * dT1_dVb ;
                dT3_dVg *= dT0_dVg ;
                T4 = pParam->BSIM4v7rds0 * 0.5 ;
                Rds = pParam->BSIM4v7rdswmin + T3 * T4 ;
                dRds_dVg = T4 * dT3_dVg ;
                dRds_dVb = T4 * dT3_dVb ;

                if (Rds > 0.0)
                    BSIM4v7entry.d_BSIM4v7grdswArray [instance_ID] = 1.0 / Rds* BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ; /*4.6.2*/
                else
                    BSIM4v7entry.d_BSIM4v7grdswArray [instance_ID] = 0.0 ;
            }


            /* Calculate Abulk */
            T9 = 0.5 * pParam->BSIM4v7k1ox * Lpe_Vb / sqrtPhis ;
            T1 = T9 + BSIM4v7entry.d_BSIM4v7k2oxArray [instance_ID] - pParam->BSIM4v7k3b * Vth_NarrowW ;
            dT1_dVb = -T9 / sqrtPhis * dsqrtPhis_dVb ;
            T9 = sqrt (pParam->BSIM4v7xj * Xdep) ;
            tmp1 = Leff + 2.0 * T9 ;
            T5 = Leff / tmp1 ;
            tmp2 = pParam->BSIM4v7a0 * T5 ;
            tmp3 = pParam->BSIM4v7weff + pParam->BSIM4v7b1 ;
            tmp4 = pParam->BSIM4v7b0 / tmp3 ;
            T2 = tmp2 + tmp4 ;
            dT2_dVb = -T9 / tmp1 / Xdep * dXdep_dVb ;
            T6 = T5 * T5 ;
            T7 = T5 * T6 ;
            Abulk0 = 1.0 + T1 * T2 ;
            dAbulk0_dVb = T1 * tmp2 * dT2_dVb + T2 * dT1_dVb ;
            T8 = pParam->BSIM4v7ags * pParam->BSIM4v7a0 * T7 ;
            dAbulk_dVg = -T1 * T8 ;
            Abulk = Abulk0 + dAbulk_dVg * Vgsteff ;
            dAbulk_dVb = dAbulk0_dVb - T8 * Vgsteff * (dT1_dVb + 3.0 * T1 * dT2_dVb) ;

            /* 22 - DIVERGENT */
            if (Abulk0 < 0.1) /* added to avoid the problems caused by Abulk0 */
            {
                T9 = 1.0 / (3.0 - 20.0 * Abulk0) ;
                Abulk0 = (0.2 - Abulk0) * T9 ;
                dAbulk0_dVb *= T9 * T9 ;
            }

            /* 23 - DIVERGENT */
            if (Abulk < 0.1)
            {
                T9 = 1.0 / (3.0 - 20.0 * Abulk) ;
                Abulk = (0.2 - Abulk) * T9 ;
                T10 = T9 * T9 ;
                dAbulk_dVb *= T10 ;
                dAbulk_dVg *= T10 ;
            }
            BSIM4v7entry.d_BSIM4v7AbulkArray [instance_ID] = Abulk ;
            T2 = pParam->BSIM4v7keta * Vbseff ;

            /* 24 - DIVERGENT */
            if (T2 >= -0.9)
            {
                T0 = 1.0 / (1.0 + T2) ;
                dT0_dVb = -pParam->BSIM4v7keta * T0 * T0 ;
            }
            else
            {
                T1 = 1.0 / (0.8 + T2) ;
                T0 = (17.0 + 20.0 * T2) * T1 ;
                dT0_dVb = -pParam->BSIM4v7keta * T1 * T1 ;
            }
            dAbulk_dVg *= T0 ;
            dAbulk_dVb = dAbulk_dVb * T0 + Abulk * dT0_dVb ;
            dAbulk0_dVb = dAbulk0_dVb * T0 + Abulk0 * dT0_dVb ;
            Abulk *= T0 ;
            Abulk0 *= T0 ;


            /* Mobility calculation */
            /* 4 - non-divergent */
            if (BSIM4v7mtrlMod && BSIM4v7mtrlCompatMod == 0)
                T14 = 2.0 * BSIM4v7type * (BSIM4v7phig - BSIM4v7easub - 0.5 * BSIM4v7Eg0 + 0.45) ;
            else
                T14 = 0.0 ;

            /* 5 - non-divergent */
            if (BSIM4v7mobMod == 0)
            {
                T0 = Vgsteff + Vth + Vth - T14 ;
                T2 = pParam->BSIM4v7ua + pParam->BSIM4v7uc * Vbseff ;
                T3 = T0 / toxe ;
                T12 = sqrt (Vth * Vth + 0.0001) ;
                T9 = 1.0 / (Vgsteff + 2 * T12) ;
                T10 = T9 * toxe ;
                T8 = pParam->BSIM4v7ud * T10 * T10 * Vth ;
                T6 = T8 * Vth ;
                T5 = T3 * (T2 + pParam->BSIM4v7ub * T3) + T6 ;
                T7 = -2.0 * T6 * T9 ;
                T11 = T7 * Vth / T12 ;
                dDenomi_dVg = (T2 + 2.0 * pParam->BSIM4v7ub * T3) / toxe ;
                T13 = 2.0 * (dDenomi_dVg + T11 + T8) ;
                dDenomi_dVd = T13 * dVth_dVd ;
                dDenomi_dVb = T13 * dVth_dVb + pParam->BSIM4v7uc * T3 ;
                dDenomi_dVg += T7 ;
            }
            else if (BSIM4v7mobMod == 1)
            {
                T0 = Vgsteff + Vth + Vth - T14 ;
                T2 = 1.0 + pParam->BSIM4v7uc * Vbseff ;
                T3 = T0 / toxe ;
                T4 = T3 * (pParam->BSIM4v7ua + pParam->BSIM4v7ub * T3) ;
                T12 = sqrt (Vth * Vth + 0.0001) ;
                T9 = 1.0 / (Vgsteff + 2 * T12) ;
                T10 = T9 * toxe ;
                T8 = pParam->BSIM4v7ud * T10 * T10 * Vth ;
                T6 = T8 * Vth ;
                T5 = T4 * T2 + T6 ;
                T7 = -2.0 * T6 * T9 ;
                T11 = T7 * Vth / T12 ;
                dDenomi_dVg = (pParam->BSIM4v7ua + 2.0 * pParam->BSIM4v7ub * T3) * T2 / toxe ;
                T13 = 2.0 * (dDenomi_dVg + T11 + T8) ;
                dDenomi_dVd = T13 * dVth_dVd ;
                dDenomi_dVb = T13 * dVth_dVb + pParam->BSIM4v7uc * T4 ;
                dDenomi_dVg += T7 ;
            }
            else if (BSIM4v7mobMod == 2)
            {
                T0 = (Vgsteff + BSIM4v7entry.d_BSIM4v7vtfbphi1Array [instance_ID]) / toxe ;
                T1 = exp (pParam->BSIM4v7eu * log (T0)) ;
                dT1_dVg = T1 * pParam->BSIM4v7eu / T0 / toxe ;
                T2 = pParam->BSIM4v7ua + pParam->BSIM4v7uc * Vbseff ;
                T3 = T0 / toxe ;    /*Do we need it?*/
                T12 = sqrt (Vth * Vth + 0.0001) ;
                T9 = 1.0 / (Vgsteff + 2 * T12) ;
                T10 = T9 * toxe ;
                T8 = pParam->BSIM4v7ud * T10 * T10 * Vth ;
                T6 = T8 * Vth ;
                T5 = T1 * T2 + T6 ;
                T7 = -2.0 * T6 * T9 ;
                T11 = T7 * Vth/T12 ;
                dDenomi_dVg = T2 * dT1_dVg + T7 ;
                T13 = 2.0 * (T11 + T8) ;
                dDenomi_dVd = T13 * dVth_dVd ;
                dDenomi_dVb = T13 * dVth_dVb + T1 * pParam->BSIM4v7uc ;
            }

            /*high K mobility*/
            else
            {
                /* univsersal mobility */
                T0 = (Vgsteff + BSIM4v7entry.d_BSIM4v7vtfbphi1Array [instance_ID]) * 1.0e-8 / toxe / 6.0 ;
                T1 = exp (pParam->BSIM4v7eu * log (T0)) ;
                dT1_dVg = T1 * pParam->BSIM4v7eu * 1.0e-8 / T0 / toxe / 6.0 ;
                T2 = pParam->BSIM4v7ua + pParam->BSIM4v7uc * Vbseff ;

                /*Coulombic*/
                VgsteffVth = pParam->BSIM4v7VgsteffVth ;
                T10 = exp (pParam->BSIM4v7ucs * log (0.5 + 0.5 * Vgsteff / VgsteffVth)) ;
                T11 =  pParam->BSIM4v7ud / T10;
                dT11_dVg = -0.5 * pParam->BSIM4v7ucs * T11 / (0.5 + 0.5 * Vgsteff / VgsteffVth) / VgsteffVth ;
                dDenomi_dVg = T2 * dT1_dVg + dT11_dVg ;
                dDenomi_dVd = 0.0 ;
                dDenomi_dVb = T1 * pParam->BSIM4v7uc ;
                T5 = T1 * T2 + T11 ;
            }

            /* 25 - DIVERGENT */
            if (T5 >= -0.8)
                Denomi = 1.0 + T5 ;
            else
            {
                T9 = 1.0 / (7.0 + 10.0 * T5) ;
                Denomi = (0.6 + T5) * T9 ;
                T9 *= T9 ;
                dDenomi_dVg *= T9 ;
                dDenomi_dVd *= T9 ;
                dDenomi_dVb *= T9 ;
            }
            BSIM4v7entry.d_BSIM4v7ueffArray [instance_ID] = ueff = BSIM4v7entry.d_BSIM4v7u0tempArray [instance_ID] / Denomi ;
            T9 = -ueff / Denomi ;
            dueff_dVg = T9 * dDenomi_dVg ;
            dueff_dVd = T9 * dDenomi_dVd ;
            dueff_dVb = T9 * dDenomi_dVb ;


            /* Saturation Drain Voltage  Vdsat */
            WVCox = Weff * BSIM4v7entry.d_BSIM4v7vsattempArray [instance_ID] * BSIM4v7coxe ;
            WVCoxRds = WVCox * Rds ;
            Esat = 2.0 * BSIM4v7entry.d_BSIM4v7vsattempArray [instance_ID] / ueff ;
            BSIM4v7entry.d_BSIM4v7EsatLArray [instance_ID] = EsatL = Esat * Leff ;
            T0 = -EsatL / ueff ;
            dEsatL_dVg = T0 * dueff_dVg ;
            dEsatL_dVd = T0 * dueff_dVd ;
            dEsatL_dVb = T0 * dueff_dVb ;

            /* Sqrt() */
            /* 26 - DIVERGENT */
            a1 = pParam->BSIM4v7a1 ;
            if (a1 == 0.0)
            {
                Lambda = pParam->BSIM4v7a2 ;
                dLambda_dVg = 0.0 ;
            }
            else if (a1 > 0.0)
            {
                T0 = 1.0 - pParam->BSIM4v7a2 ;
                T1 = T0 - pParam->BSIM4v7a1 * Vgsteff - 0.0001 ;
                T2 = sqrt (T1 * T1 + 0.0004 * T0) ;
                Lambda = pParam->BSIM4v7a2 + T0 - 0.5 * (T1 + T2) ;
                dLambda_dVg = 0.5 * pParam->BSIM4v7a1 * (1.0 + T1 / T2) ;
            }
            else
            {
                T1 = pParam->BSIM4v7a2 + pParam->BSIM4v7a1 * Vgsteff - 0.0001 ;
                T2 = sqrt (T1 * T1 + 0.0004 * pParam->BSIM4v7a2) ;
                Lambda = 0.5 * (T1 + T2) ;
                dLambda_dVg = 0.5 * pParam->BSIM4v7a1 * (1.0 + T1 / T2) ;
            }
            Vgst2Vtm = Vgsteff + 2.0 * Vtm ;

            /* 27 - DIVERGENT */
            if (Rds > 0)
            {
                tmp2 = dRds_dVg / Rds + dWeff_dVg / Weff ;
                tmp3 = dRds_dVb / Rds + dWeff_dVb / Weff ;
            }
            else
            {
                tmp2 = dWeff_dVg / Weff ;
                tmp3 = dWeff_dVb / Weff ;
            }

            /* 28 - DIVERGENT */
            if ((Rds == 0.0) && (Lambda == 1.0))
            {
                T0 = 1.0 / (Abulk * EsatL + Vgst2Vtm) ;
                tmp1 = 0.0 ;
                T1 = T0 * T0 ;
                T2 = Vgst2Vtm * T0 ;
                T3 = EsatL * Vgst2Vtm ;
                Vdsat = T3 * T0 ;
                dT0_dVg = -(Abulk * dEsatL_dVg + EsatL * dAbulk_dVg + 1.0) * T1 ;
                dT0_dVd = -(Abulk * dEsatL_dVd) * T1 ;
                dT0_dVb = -(Abulk * dEsatL_dVb + dAbulk_dVb * EsatL) * T1 ;
                dVdsat_dVg = T3 * dT0_dVg + T2 * dEsatL_dVg + EsatL * T0 ;
                dVdsat_dVd = T3 * dT0_dVd + T2 * dEsatL_dVd ;
                dVdsat_dVb = T3 * dT0_dVb + T2 * dEsatL_dVb ;
            }
            else
            {
                tmp1 = dLambda_dVg / (Lambda * Lambda) ;
                T9 = Abulk * WVCoxRds ;
                T8 = Abulk * T9 ;
                T7 = Vgst2Vtm * T9 ;
                T6 = Vgst2Vtm * WVCoxRds ;
                T0 = 2.0 * Abulk * (T9 - 1.0 + 1.0 / Lambda) ;
                dT0_dVg = 2.0 * (T8 * tmp2 - Abulk * tmp1 + (2.0 * T9 + 1.0 / Lambda - 1.0) * dAbulk_dVg) ;
                dT0_dVb = 2.0 * (T8 * (2.0 / Abulk * dAbulk_dVb + tmp3) + (1.0 / Lambda - 1.0) * dAbulk_dVb) ;
                dT0_dVd = 0.0;
                T1 = Vgst2Vtm * (2.0 / Lambda - 1.0) + Abulk * EsatL + 3.0 * T7 ;
                dT1_dVg = (2.0 / Lambda - 1.0) - 2.0 * Vgst2Vtm * tmp1 + Abulk * dEsatL_dVg +
                          EsatL * dAbulk_dVg + 3.0 * (T9 + T7 * tmp2 + T6 * dAbulk_dVg) ;
                dT1_dVb = Abulk * dEsatL_dVb + EsatL * dAbulk_dVb + 3.0 * (T6 * dAbulk_dVb + T7 * tmp3) ;
                dT1_dVd = Abulk * dEsatL_dVd ;
                T2 = Vgst2Vtm * (EsatL + 2.0 * T6) ;
                dT2_dVg = EsatL + Vgst2Vtm * dEsatL_dVg + T6 * (4.0 + 2.0 * Vgst2Vtm * tmp2) ;
                dT2_dVb = Vgst2Vtm * (dEsatL_dVb + 2.0 * T6 * tmp3) ;
                dT2_dVd = Vgst2Vtm * dEsatL_dVd ;
                T3 = sqrt (T1 * T1 - 2.0 * T0 * T2) ;
                Vdsat = (T1 - T3) / T0 ;
                dT3_dVg = (T1 * dT1_dVg - 2.0 * (T0 * dT2_dVg + T2 * dT0_dVg)) / T3 ;
                dT3_dVd = (T1 * dT1_dVd - 2.0 * (T0 * dT2_dVd + T2 * dT0_dVd)) / T3 ;
                dT3_dVb = (T1 * dT1_dVb - 2.0 * (T0 * dT2_dVb + T2 * dT0_dVb)) / T3 ;
                dVdsat_dVg = (dT1_dVg - (T1 * dT1_dVg - dT0_dVg * T2 - T0 * dT2_dVg) / T3 - Vdsat * dT0_dVg) / T0 ;
                dVdsat_dVb = (dT1_dVb - (T1 * dT1_dVb - dT0_dVb * T2 - T0 * dT2_dVb) / T3 - Vdsat * dT0_dVb) / T0 ;
                dVdsat_dVd = (dT1_dVd - (T1 * dT1_dVd - T0 * dT2_dVd) / T3) / T0 ;
            }
            BSIM4v7entry.d_BSIM4v7vdsatRWArray [instance_ID] = Vdsat ;


            /* Calculate Vdseff */
            T1 = Vdsat - Vds - pParam->BSIM4v7delta ;
            dT1_dVg = dVdsat_dVg ;
            dT1_dVd = dVdsat_dVd - 1.0 ;
            dT1_dVb = dVdsat_dVb ;
            T2 = sqrt (T1 * T1 + 4.0 * pParam->BSIM4v7delta * Vdsat) ;
            T0 = T1 / T2 ;
            T9 = 2.0 * pParam->BSIM4v7delta ;
            T3 = T9 / T2 ;
            dT2_dVg = T0 * dT1_dVg + T3 * dVdsat_dVg ;
            dT2_dVd = T0 * dT1_dVd + T3 * dVdsat_dVd ;
            dT2_dVb = T0 * dT1_dVb + T3 * dVdsat_dVb ;

            /* 29 - DIVERGENT */
            if (T1 >= 0.0)
            {
                Vdseff = Vdsat - 0.5 * (T1 + T2) ;
                dVdseff_dVg = dVdsat_dVg - 0.5 * (dT1_dVg + dT2_dVg) ;
                dVdseff_dVd = dVdsat_dVd - 0.5 * (dT1_dVd + dT2_dVd) ;
                dVdseff_dVb = dVdsat_dVb - 0.5 * (dT1_dVb + dT2_dVb) ;
            }
            else
            {
                T4 = T9 / (T2 - T1) ;
                T5 = 1.0 - T4 ;
                T6 = Vdsat * T4 / (T2 - T1) ;
                Vdseff = Vdsat * T5 ;
                dVdseff_dVg = dVdsat_dVg * T5 + T6 * (dT2_dVg - dT1_dVg) ;
                dVdseff_dVd = dVdsat_dVd * T5 + T6 * (dT2_dVd - dT1_dVd) ;
                dVdseff_dVb = dVdsat_dVb * T5 + T6 * (dT2_dVb - dT1_dVb) ;
            }

            /* 30 - DIVERGENT */
            if (Vds == 0.0)
            {
                Vdseff = 0.0 ;
                dVdseff_dVg = 0.0 ;
                dVdseff_dVb = 0.0 ;
            }

            /* 31 - DIVERGENT */
            if (Vdseff > Vds)
                Vdseff = Vds ;

            diffVds = Vds - Vdseff ;
            BSIM4v7entry.d_BSIM4v7VdseffArray [instance_ID] = Vdseff ;


            /* Velocity Overshoot */
            /* 6 - non-divergent */
            if((BSIM4v7lambdaGiven) && (BSIM4v7lambda > 0.0))
            {
                T1 =  Leff * ueff ;
                T2 = pParam->BSIM4v7lambda / T1 ;
                T3 = -T2 / T1 * Leff ;
                dT2_dVd = T3 * dueff_dVd ;
                dT2_dVg = T3 * dueff_dVg ;
                dT2_dVb = T3 * dueff_dVb ;
                T5 = 1.0 / (Esat * pParam->BSIM4v7litl) ;
                T4 = -T5 / EsatL ;
                dT5_dVg = dEsatL_dVg * T4 ;
                dT5_dVd = dEsatL_dVd * T4 ;
                dT5_dVb = dEsatL_dVb * T4 ;
                T6 = 1.0 + diffVds  * T5 ;
                dT6_dVg = dT5_dVg * diffVds - dVdseff_dVg * T5 ;
                dT6_dVd = dT5_dVd * diffVds + (1.0 - dVdseff_dVd) * T5 ;
                dT6_dVb = dT5_dVb * diffVds - dVdseff_dVb * T5 ;
                T7 = 2.0 / (T6 * T6 + 1.0) ;
                T8 = 1.0 - T7 ;
                T9 = T6 * T7 * T7 ;
                dT8_dVg = T9 * dT6_dVg ;
                dT8_dVd = T9 * dT6_dVd ;
                dT8_dVb = T9 * dT6_dVb ;
                T10 = 1.0 + T2 * T8 ;
                dT10_dVg = dT2_dVg * T8 + T2 * dT8_dVg ;
                dT10_dVd = dT2_dVd * T8 + T2 * dT8_dVd ;
                dT10_dVb = dT2_dVb * T8 + T2 * dT8_dVb ;

                if(T10 == 1.0)
                    dT10_dVg = dT10_dVd = dT10_dVb = 0.0 ;

                dEsatL_dVg *= T10 ;
                dEsatL_dVg += EsatL * dT10_dVg ;
                dEsatL_dVd *= T10 ;
                dEsatL_dVd += EsatL * dT10_dVd ;
                dEsatL_dVb *= T10 ;
                dEsatL_dVb += EsatL * dT10_dVb ;
                EsatL *= T10 ;
                Esat = EsatL / Leff ;    /* bugfix by Wenwei Yang (4.6.4) */
                BSIM4v7entry.d_BSIM4v7EsatLArray [instance_ID] = EsatL ;
            }


            /* Calculate Vasat */
            tmp4 = 1.0 - 0.5 * Abulk * Vdsat / Vgst2Vtm ;
            T9 = WVCoxRds * Vgsteff ;
            T8 = T9 / Vgst2Vtm ;
            T0 = EsatL + Vdsat + 2.0 * T9 * tmp4 ;
            T7 = 2.0 * WVCoxRds * tmp4 ;
            dT0_dVg = dEsatL_dVg + dVdsat_dVg + T7 * (1.0 + tmp2 * Vgsteff) - T8 *
                      (Abulk * dVdsat_dVg - Abulk * Vdsat / Vgst2Vtm + Vdsat * dAbulk_dVg) ;
            dT0_dVb = dEsatL_dVb + dVdsat_dVb + T7 * tmp3 * Vgsteff - T8 * (dAbulk_dVb * Vdsat + Abulk * dVdsat_dVb) ;
            dT0_dVd = dEsatL_dVd + dVdsat_dVd - T8 * Abulk * dVdsat_dVd ;
            T9 = WVCoxRds * Abulk ;
            T1 = 2.0 / Lambda - 1.0 + T9 ;
            dT1_dVg = -2.0 * tmp1 +  WVCoxRds * (Abulk * tmp2 + dAbulk_dVg) ;
            dT1_dVb = dAbulk_dVb * WVCoxRds + T9 * tmp3 ;
            Vasat = T0 / T1 ;
            dVasat_dVg = (dT0_dVg - Vasat * dT1_dVg) / T1 ;
            dVasat_dVb = (dT0_dVb - Vasat * dT1_dVb) / T1 ;
            dVasat_dVd = dT0_dVd / T1 ;


            /* Calculate Idl first */
            tmp1 = BSIM4v7entry.d_BSIM4v7vtfbphi2Array [instance_ID] ;
            tmp2 = 2.0e8 * BSIM4v7toxp ;
            dT0_dVg = 1.0 / tmp2 ;
            T0 = (Vgsteff + tmp1) * dT0_dVg ;
            tmp3 = exp (BSIM4v7bdos * 0.7 * log (T0)) ;
            T1 = 1.0 + tmp3 ;
            T2 = BSIM4v7bdos * 0.7 * tmp3 / T0 ;
            Tcen = BSIM4v7ados * 1.9e-9 / T1 ;
            dTcen_dVg = -Tcen * T2 * dT0_dVg / T1 ;
            Coxeff = epssub * BSIM4v7coxp / (epssub + BSIM4v7coxp * Tcen) ;
            BSIM4v7entry.d_BSIM4v7CoxeffArray [instance_ID] = Coxeff ;
            dCoxeff_dVg = -Coxeff * Coxeff * dTcen_dVg / epssub ;
            CoxeffWovL = Coxeff * Weff / Leff ;
            beta = ueff * CoxeffWovL ;
            T3 = ueff / Leff ;
            dbeta_dVg = CoxeffWovL * dueff_dVg + T3 * (Weff * dCoxeff_dVg + Coxeff * dWeff_dVg) ;
            dbeta_dVd = CoxeffWovL * dueff_dVd ;
            dbeta_dVb = CoxeffWovL * dueff_dVb + T3 * Coxeff * dWeff_dVb ;
            BSIM4v7entry.d_BSIM4v7AbovVgst2VtmArray [instance_ID] = Abulk / Vgst2Vtm ;
            T0 = 1.0 - 0.5 * Vdseff * BSIM4v7entry.d_BSIM4v7AbovVgst2VtmArray [instance_ID] ;
            dT0_dVg = -0.5 * (Abulk * dVdseff_dVg - Abulk * Vdseff / Vgst2Vtm + Vdseff * dAbulk_dVg) / Vgst2Vtm ;
            dT0_dVd = -0.5 * Abulk * dVdseff_dVd / Vgst2Vtm ;
            dT0_dVb = -0.5 * (Abulk * dVdseff_dVb + dAbulk_dVb * Vdseff) / Vgst2Vtm ;
            fgche1 = Vgsteff * T0 ;
            dfgche1_dVg = Vgsteff * dT0_dVg + T0 ;
            dfgche1_dVd = Vgsteff * dT0_dVd ;
            dfgche1_dVb = Vgsteff * dT0_dVb ;
            T9 = Vdseff / EsatL ;
            fgche2 = 1.0 + T9 ;
            dfgche2_dVg = (dVdseff_dVg - T9 * dEsatL_dVg) / EsatL ;
            dfgche2_dVd = (dVdseff_dVd - T9 * dEsatL_dVd) / EsatL ;
            dfgche2_dVb = (dVdseff_dVb - T9 * dEsatL_dVb) / EsatL ;
            gche = beta * fgche1 / fgche2 ;
            dgche_dVg = (beta * dfgche1_dVg + fgche1 * dbeta_dVg - gche * dfgche2_dVg) / fgche2 ;
            dgche_dVd = (beta * dfgche1_dVd + fgche1 * dbeta_dVd - gche * dfgche2_dVd) / fgche2 ;
            dgche_dVb = (beta * dfgche1_dVb + fgche1 * dbeta_dVb - gche * dfgche2_dVb) / fgche2 ;
            T0 = 1.0 + gche * Rds ;
            Idl = gche / T0 ;
            T1 = (1.0 - Idl * Rds) / T0 ;
            T2 = Idl * Idl ;
            dIdl_dVg = T1 * dgche_dVg - T2 * dRds_dVg ;
            dIdl_dVd = T1 * dgche_dVd ;
            dIdl_dVb = T1 * dgche_dVb - T2 * dRds_dVb ;


            /* Calculate degradation factor due to pocket implant */
            /* 31 - DIVERGENT */
            if (pParam->BSIM4v7fprout <= 0.0)
            {
                FP = 1.0 ;
                dFP_dVg = 0.0 ;
            }
            else
            {
                T9 = pParam->BSIM4v7fprout * sqrt (Leff) / Vgst2Vtm ;
                FP = 1.0 / (1.0 + T9) ;
                dFP_dVg = FP * FP * T9 / Vgst2Vtm ;
            }


            /* Calculate VACLM */
            T8 = pParam->BSIM4v7pvag / EsatL ;
            T9 = T8 * Vgsteff ;

            /* 32 - DIVERGENT */
            if (T9 > -0.9)
            {
                PvagTerm = 1.0 + T9 ;
                dPvagTerm_dVg = T8 * (1.0 - Vgsteff * dEsatL_dVg / EsatL) ;
                dPvagTerm_dVb = -T9 * dEsatL_dVb / EsatL ;
                dPvagTerm_dVd = -T9 * dEsatL_dVd / EsatL ;
            }
            else
            {
                T4 = 1.0 / (17.0 + 20.0 * T9) ;
                PvagTerm = (0.8 + T9) * T4 ;
                T4 *= T4 ;
                dPvagTerm_dVg = T8 * (1.0 - Vgsteff * dEsatL_dVg / EsatL) * T4 ;
                T9 *= T4 / EsatL ;
                dPvagTerm_dVb = -T9 * dEsatL_dVb ;
                dPvagTerm_dVd = -T9 * dEsatL_dVd ;
            }

            /* 33 - DIVERGENT */
            if ((pParam->BSIM4v7pclm > MIN_EXP) && (diffVds > 1.0e-10))
            {
                T0 = 1.0 + Rds * Idl ;
                dT0_dVg = dRds_dVg * Idl + Rds * dIdl_dVg ;
                dT0_dVd = Rds * dIdl_dVd ;
                dT0_dVb = dRds_dVb * Idl + Rds * dIdl_dVb ;
                T2 = Vdsat / Esat ;
                T1 = Leff + T2 ;
                dT1_dVg = (dVdsat_dVg - T2 * dEsatL_dVg / Leff) / Esat ;
                dT1_dVd = (dVdsat_dVd - T2 * dEsatL_dVd / Leff) / Esat ;
                dT1_dVb = (dVdsat_dVb - T2 * dEsatL_dVb / Leff) / Esat ;
                Cclm = FP * PvagTerm * T0 * T1 / (pParam->BSIM4v7pclm * pParam->BSIM4v7litl) ;
                dCclm_dVg = Cclm * (dFP_dVg / FP + dPvagTerm_dVg / PvagTerm + dT0_dVg / T0 + dT1_dVg / T1) ;
                dCclm_dVb = Cclm * (dPvagTerm_dVb / PvagTerm + dT0_dVb / T0 + dT1_dVb / T1) ;
                dCclm_dVd = Cclm * (dPvagTerm_dVd / PvagTerm + dT0_dVd / T0 + dT1_dVd / T1) ;
                VACLM = Cclm * diffVds ;
                dVACLM_dVg = dCclm_dVg * diffVds - dVdseff_dVg * Cclm ;
                dVACLM_dVb = dCclm_dVb * diffVds - dVdseff_dVb * Cclm ;
                dVACLM_dVd = dCclm_dVd * diffVds + (1.0 - dVdseff_dVd) * Cclm ;
            }
            else
            {
                VACLM = Cclm = MAX_EXP ;
                dVACLM_dVd = dVACLM_dVg = dVACLM_dVb = 0.0 ;
                dCclm_dVd = dCclm_dVg = dCclm_dVb = 0.0 ;
            }


            /* Calculate VADIBL */
            /* 34 - DIVERGENT */
            if (pParam->BSIM4v7thetaRout > MIN_EXP)
            {
                T8 = Abulk * Vdsat ;
                T0 = Vgst2Vtm * T8 ;
                dT0_dVg = Vgst2Vtm * Abulk * dVdsat_dVg + T8 + Vgst2Vtm * Vdsat * dAbulk_dVg ;
                dT0_dVb = Vgst2Vtm * (dAbulk_dVb * Vdsat + Abulk * dVdsat_dVb) ;
                dT0_dVd = Vgst2Vtm * Abulk * dVdsat_dVd ;
                T1 = Vgst2Vtm + T8 ;
                dT1_dVg = 1.0 + Abulk * dVdsat_dVg + Vdsat * dAbulk_dVg ;
                dT1_dVb = Abulk * dVdsat_dVb + dAbulk_dVb * Vdsat ;
                dT1_dVd = Abulk * dVdsat_dVd ;
                T9 = T1 * T1 ;
                T2 = pParam->BSIM4v7thetaRout ;
                VADIBL = (Vgst2Vtm - T0 / T1) / T2 ;
                dVADIBL_dVg = (1.0 - dT0_dVg / T1 + T0 * dT1_dVg / T9) / T2 ;
                dVADIBL_dVb = (-dT0_dVb / T1 + T0 * dT1_dVb / T9) / T2 ;
                dVADIBL_dVd = (-dT0_dVd / T1 + T0 * dT1_dVd / T9) / T2 ;
                T7 = pParam->BSIM4v7pdiblb * Vbseff ;
                if (T7 >= -0.9)
                {
                    T3 = 1.0 / (1.0 + T7) ;
                    VADIBL *= T3 ;
                    dVADIBL_dVg *= T3 ;
                    dVADIBL_dVb = (dVADIBL_dVb - VADIBL * pParam->BSIM4v7pdiblb) * T3 ;
                    dVADIBL_dVd *= T3 ;
                }
                else
                {
                    T4 = 1.0 / (0.8 + T7) ;
                    T3 = (17.0 + 20.0 * T7) * T4 ;
                    dVADIBL_dVg *= T3 ;
                    dVADIBL_dVb = dVADIBL_dVb * T3 - VADIBL * pParam->BSIM4v7pdiblb * T4 * T4 ;
                    dVADIBL_dVd *= T3 ;
                    VADIBL *= T3 ;
                }
                dVADIBL_dVg = dVADIBL_dVg * PvagTerm + VADIBL * dPvagTerm_dVg ;
                dVADIBL_dVb = dVADIBL_dVb * PvagTerm + VADIBL * dPvagTerm_dVb ;
                dVADIBL_dVd = dVADIBL_dVd * PvagTerm + VADIBL * dPvagTerm_dVd ;
                VADIBL *= PvagTerm ;
            }
            else
            {
                VADIBL = MAX_EXP ;
                dVADIBL_dVd = dVADIBL_dVg = dVADIBL_dVb = 0.0 ;
            }


            /* Calculate Va */
            Va = Vasat + VACLM ;
            dVa_dVg = dVasat_dVg + dVACLM_dVg ;
            dVa_dVb = dVasat_dVb + dVACLM_dVb ;
            dVa_dVd = dVasat_dVd + dVACLM_dVd ;


            /* Calculate VADITS */
            T0 = pParam->BSIM4v7pditsd * Vds ;

            /* 35 - DIVERGENT */
            if (T0 > EXP_THRESHOLD)
            {
                T1 = MAX_EXP ;
                dT1_dVd = 0 ;
            }
            else
            {
                T1 = exp (T0) ;
                dT1_dVd = T1 * pParam->BSIM4v7pditsd ;
            }

            /* 36 - DIVERGENT */
            if (pParam->BSIM4v7pdits > MIN_EXP)
            {
                T2 = 1.0 + BSIM4v7pditsl * Leff ;
                VADITS = (1.0 + T2 * T1) / pParam->BSIM4v7pdits ;
                dVADITS_dVg = VADITS * dFP_dVg ;
                dVADITS_dVd = FP * T2 * dT1_dVd / pParam->BSIM4v7pdits ;
                VADITS *= FP ;
            }
            else
            {
                VADITS = MAX_EXP ;
                dVADITS_dVg = dVADITS_dVd = 0 ;
            }


            /* Calculate VASCBE */
            /* 37 - DIVERGENT */
            if ((pParam->BSIM4v7pscbe2 > 0.0) && (pParam->BSIM4v7pscbe1 >= 0.0)) /*4.6.2*/
            {
                if (diffVds > pParam->BSIM4v7pscbe1 * pParam->BSIM4v7litl / EXP_THRESHOLD)
                {
                    T0 =  pParam->BSIM4v7pscbe1 * pParam->BSIM4v7litl / diffVds ;
                    VASCBE = Leff * exp (T0) / pParam->BSIM4v7pscbe2 ;
                    T1 = T0 * VASCBE / diffVds ;
                    dVASCBE_dVg = T1 * dVdseff_dVg ;
                    dVASCBE_dVd = -T1 * (1.0 - dVdseff_dVd) ;
                    dVASCBE_dVb = T1 * dVdseff_dVb ;
                }
                else
                {
                    VASCBE = MAX_EXP * Leff/pParam->BSIM4v7pscbe2 ;
                    dVASCBE_dVg = dVASCBE_dVd = dVASCBE_dVb = 0.0 ;
                }
            }
            else
            {
                VASCBE = MAX_EXP ;
                dVASCBE_dVg = dVASCBE_dVd = dVASCBE_dVb = 0.0 ;
            }


            /* Add DIBL to Ids */
            T9 = diffVds / VADIBL ;
            T0 = 1.0 + T9 ;
            Idsa = Idl * T0 ;
            dIdsa_dVg = T0 * dIdl_dVg - Idl * (dVdseff_dVg + T9 * dVADIBL_dVg) / VADIBL ;
            dIdsa_dVd = T0 * dIdl_dVd + Idl * (1.0 - dVdseff_dVd - T9 * dVADIBL_dVd) / VADIBL ;
            dIdsa_dVb = T0 * dIdl_dVb - Idl * (dVdseff_dVb + T9 * dVADIBL_dVb) / VADIBL ;


            /* Add DITS to Ids */
            T9 = diffVds / VADITS ;
            T0 = 1.0 + T9 ;
            dIdsa_dVg = T0 * dIdsa_dVg - Idsa * (dVdseff_dVg + T9 * dVADITS_dVg) / VADITS ;
            dIdsa_dVd = T0 * dIdsa_dVd + Idsa  * (1.0 - dVdseff_dVd - T9 * dVADITS_dVd) / VADITS ;
            dIdsa_dVb = T0 * dIdsa_dVb - Idsa * dVdseff_dVb / VADITS ;
            Idsa *= T0 ;


            /* Add CLM to Ids */
            T0 = log (Va / Vasat) ;
            dT0_dVg = dVa_dVg / Va - dVasat_dVg / Vasat ;
            dT0_dVb = dVa_dVb / Va - dVasat_dVb / Vasat ;
            dT0_dVd = dVa_dVd / Va - dVasat_dVd / Vasat ;
            T1 = T0 / Cclm ;
            T9 = 1.0 + T1 ;
            dT9_dVg = (dT0_dVg - T1 * dCclm_dVg) / Cclm ;
            dT9_dVb = (dT0_dVb - T1 * dCclm_dVb) / Cclm ;
            dT9_dVd = (dT0_dVd - T1 * dCclm_dVd) / Cclm ;
            dIdsa_dVg = dIdsa_dVg * T9 + Idsa * dT9_dVg ;
            dIdsa_dVb = dIdsa_dVb * T9 + Idsa * dT9_dVb ;
            dIdsa_dVd = dIdsa_dVd * T9 + Idsa * dT9_dVd ;
            Idsa *= T9 ;


            /* Substrate current begins */
            tmp = pParam->BSIM4v7alpha0 + pParam->BSIM4v7alpha1 * Leff ;

            /* 38 - DIVERGENT */
            if ((tmp <= 0.0) || (pParam->BSIM4v7beta0 <= 0.0))
                Isub = Gbd = Gbb = Gbg = 0.0 ;
            else
            {
                T2 = tmp / Leff ;
                if (diffVds > pParam->BSIM4v7beta0 / EXP_THRESHOLD)
                {
                    T0 = -pParam->BSIM4v7beta0 / diffVds ;
                    T1 = T2 * diffVds * exp (T0) ;
                    T3 = T1 / diffVds * (T0 - 1.0) ;
                    dT1_dVg = T3 * dVdseff_dVg ;
                    dT1_dVd = T3 * (dVdseff_dVd - 1.0) ;
                    dT1_dVb = T3 * dVdseff_dVb ;
                }
                else
                {
                    T3 = T2 * MIN_EXP ;
                    T1 = T3 * diffVds ;
                    dT1_dVg = -T3 * dVdseff_dVg ;
                    dT1_dVd = T3 * (1.0 - dVdseff_dVd) ;
                    dT1_dVb = -T3 * dVdseff_dVb ;
                }
                T4 = Idsa * Vdseff ;
                Isub = T1 * T4 ;
                Gbg = T1 * (dIdsa_dVg * Vdseff + Idsa * dVdseff_dVg) + T4 * dT1_dVg ;
                Gbd = T1 * (dIdsa_dVd * Vdseff + Idsa * dVdseff_dVd) + T4 * dT1_dVd ;
                Gbb = T1 * (dIdsa_dVb * Vdseff + Idsa * dVdseff_dVb) + T4 * dT1_dVb ;
                Gbd += Gbg * dVgsteff_dVd ;
                Gbb += Gbg * dVgsteff_dVb ;
                Gbg *= dVgsteff_dVg ;
                Gbb *= dVbseff_dVb ;
            }
            BSIM4v7entry.d_BSIM4v7csubRWArray [instance_ID] = Isub ;
            BSIM4v7entry.d_BSIM4v7gbbsArray [instance_ID] = Gbb ;
            BSIM4v7entry.d_BSIM4v7gbgsArray [instance_ID] = Gbg ;
            BSIM4v7entry.d_BSIM4v7gbdsArray [instance_ID] = Gbd ;


            /* Add SCBE to Ids */
            T9 = diffVds / VASCBE ;
            T0 = 1.0 + T9 ;
            Ids = Idsa * T0 ;
            Gm = T0 * dIdsa_dVg - Idsa * (dVdseff_dVg + T9 * dVASCBE_dVg) / VASCBE ;
            Gds = T0 * dIdsa_dVd + Idsa * (1.0 - dVdseff_dVd - T9 * dVASCBE_dVd) / VASCBE ;
            Gmb = T0 * dIdsa_dVb - Idsa * (dVdseff_dVb + T9 * dVASCBE_dVb) / VASCBE ;
            tmp1 = Gds + Gm * dVgsteff_dVd ;
            tmp2 = Gmb + Gm * dVgsteff_dVb ;
            tmp3 = Gm ;
            Gm = (Ids * dVdseff_dVg + Vdseff * tmp3) * dVgsteff_dVg ;
            Gds = Ids * (dVdseff_dVd + dVdseff_dVg * dVgsteff_dVd) + Vdseff * tmp1 ;
            Gmb = (Ids * (dVdseff_dVb + dVdseff_dVg * dVgsteff_dVb) + Vdseff * tmp2) * dVbseff_dVb ;
            cdrain = Ids * Vdseff ;


            /* Source End Velocity Limit  */
            /* 7 - non-divergent */
            if ((BSIM4v7vtlGiven) && (BSIM4v7vtl > 0.0))
            {
                T12 = 1.0 / Leff / CoxeffWovL ;
                T11 = T12 / Vgsteff ;
                T10 = -T11 / Vgsteff ;
                vs = cdrain * T11 ; /* vs */
                dvs_dVg = Gm * T11 + cdrain * T10 * dVgsteff_dVg ;
                dvs_dVd = Gds * T11 + cdrain * T10 * dVgsteff_dVd ;
                dvs_dVb = Gmb * T11 + cdrain * T10 * dVgsteff_dVb ;
                T0 = 2 * MM ;
                T1 = vs / (pParam->BSIM4v7vtl * pParam->BSIM4v7tfactor) ;

                /* 39 - DIVERGENT */
                if (T1 > 0.0)
                {
                    T2 = 1.0 + exp (T0 * log (T1)) ;
                    T3 = (T2 - 1.0) * T0 / vs ;
                    Fsevl = 1.0 / exp (log (T2)/ T0) ;
                    dT2_dVg = T3 * dvs_dVg ;
                    dT2_dVd = T3 * dvs_dVd ;
                    dT2_dVb = T3 * dvs_dVb ;
                    T4 = -1.0 / T0 * Fsevl / T2 ;
                    dFsevl_dVg = T4 * dT2_dVg ;
                    dFsevl_dVd = T4 * dT2_dVd ;
                    dFsevl_dVb = T4 * dT2_dVb ;
                } else {
                    Fsevl = 1.0 ;
                    dFsevl_dVg = 0.0 ;
                    dFsevl_dVd = 0.0 ;
                    dFsevl_dVb = 0.0 ;
                }
                Gm *= Fsevl ;
                Gm += cdrain * dFsevl_dVg ;
                Gmb *= Fsevl ;
                Gmb += cdrain * dFsevl_dVb ;
                Gds *= Fsevl ;
                Gds += cdrain * dFsevl_dVd ;
                cdrain *= Fsevl ;
            }
            BSIM4v7entry.d_BSIM4v7gdsRWArray [instance_ID] = Gds ;
            BSIM4v7entry.d_BSIM4v7gmRWArray [instance_ID] = Gm ;
            BSIM4v7entry.d_BSIM4v7gmbsRWArray [instance_ID] = Gmb ;
            BSIM4v7entry.d_BSIM4v7IdovVdsArray [instance_ID] = Ids ;

            if (BSIM4v7entry.d_BSIM4v7IdovVdsArray [instance_ID] <= 1.0e-9)
                BSIM4v7entry.d_BSIM4v7IdovVdsArray [instance_ID] = 1.0e-9 ;


            /* Calculate Rg */
            /* 40 - DIVERGENT */
            if ((BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] > 1) || (BSIM4v7entry.d_BSIM4v7trnqsModArray [instance_ID] != 0) || (BSIM4v7entry.d_BSIM4v7acnqsModArray [instance_ID] != 0))
            {
                T9 = pParam->BSIM4v7xrcrg2 * BSIM4v7vtm ;
                T0 = T9 * beta ;
                dT0_dVd = (dbeta_dVd + dbeta_dVg * dVgsteff_dVd) * T9 ;
                dT0_dVb = (dbeta_dVb + dbeta_dVg * dVgsteff_dVb) * T9 ;
                dT0_dVg = dbeta_dVg * T9 ;
                BSIM4v7entry.d_BSIM4v7gcrgRWArray [instance_ID] = pParam->BSIM4v7xrcrg1 * ( T0 + Ids) ;
                BSIM4v7entry.d_BSIM4v7gcrgdArray [instance_ID] = pParam->BSIM4v7xrcrg1 * (dT0_dVd + tmp1) ;
                BSIM4v7entry.d_BSIM4v7gcrgbArray [instance_ID] = pParam->BSIM4v7xrcrg1 * (dT0_dVb + tmp2) * dVbseff_dVb ;
                BSIM4v7entry.d_BSIM4v7gcrggArray [instance_ID] = pParam->BSIM4v7xrcrg1 * (dT0_dVg + tmp3) * dVgsteff_dVg ;
                if (BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] != 1.0)
                {
                    BSIM4v7entry.d_BSIM4v7gcrgRWArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                    BSIM4v7entry.d_BSIM4v7gcrggArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                    BSIM4v7entry.d_BSIM4v7gcrgdArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                    BSIM4v7entry.d_BSIM4v7gcrgbArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                }
                if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 2)
                {
                    T10 = BSIM4v7entry.d_BSIM4v7grgeltdArray [instance_ID] * BSIM4v7entry.d_BSIM4v7grgeltdArray [instance_ID] ;
                    T11 = BSIM4v7entry.d_BSIM4v7grgeltdArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gcrgRWArray [instance_ID] ;
                    BSIM4v7entry.d_BSIM4v7gcrgRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7grgeltdArray [instance_ID] * BSIM4v7entry.d_BSIM4v7gcrgRWArray [instance_ID] / T11 ;
                    T12 = T10 / T11 / T11 ;
                    BSIM4v7entry.d_BSIM4v7gcrggArray [instance_ID] *= T12 ;
                    BSIM4v7entry.d_BSIM4v7gcrgdArray [instance_ID] *= T12 ;
                    BSIM4v7entry.d_BSIM4v7gcrgbArray [instance_ID] *= T12 ;
                }
                BSIM4v7entry.d_BSIM4v7gcrgsArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7gcrggArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gcrgdArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gcrgbArray [instance_ID]) ;
            }


            /* Calculate bias-dependent external S/D resistance */
            /* 8 - non-divergent */
            if (BSIM4v7rdsMod)
            {
                /* Rs(V) */
                T0 = vgs - pParam->BSIM4v7vfbsd ;
                T1 = sqrt (T0 * T0 + 1.0e-4) ;
                vgs_eff = 0.5 * (T0 + T1) ;
                dvgs_eff_dvg = vgs_eff / T1 ;
                T0 = 1.0 + pParam->BSIM4v7prwg * vgs_eff ;
                dT0_dvg = -pParam->BSIM4v7prwg / T0 / T0 * dvgs_eff_dvg ;
                T1 = -pParam->BSIM4v7prwb * vbs ;
                dT1_dvb = -pParam->BSIM4v7prwb ;
                T2 = 1.0 / T0 + T1 ;
                T3 = T2 + sqrt (T2 * T2 + 0.01) ;
                dT3_dvg = T3 / (T3 - T2) ;
                dT3_dvb = dT3_dvg * dT1_dvb ;
                dT3_dvg *= dT0_dvg ;
                T4 = pParam->BSIM4v7rs0 * 0.5 ;
                Rs = pParam->BSIM4v7rswmin + T3 * T4 ;
                dRs_dvg = T4 * dT3_dvg ;
                dRs_dvb = T4 * dT3_dvb ;
                T0 = 1.0 + BSIM4v7entry.d_BSIM4v7sourceConductanceArray [instance_ID] * Rs ;
                BSIM4v7entry.d_BSIM4v7gstotArray [instance_ID] = BSIM4v7entry.d_BSIM4v7sourceConductanceArray [instance_ID] / T0 ;
                T0 = -BSIM4v7entry.d_BSIM4v7gstotArray [instance_ID] * BSIM4v7entry.d_BSIM4v7gstotArray [instance_ID] ;
                dgstot_dvd = 0.0 ;    /* place holder */
                dgstot_dvg = T0 * dRs_dvg ;
                dgstot_dvb = T0 * dRs_dvb ;
                dgstot_dvs = -(dgstot_dvg + dgstot_dvb + dgstot_dvd) ;

                /* Rd(V) */
                T0 = vgd - pParam->BSIM4v7vfbsd ;
                T1 = sqrt (T0 * T0 + 1.0e-4) ;
                vgd_eff = 0.5 * (T0 + T1) ;
                dvgd_eff_dvg = vgd_eff / T1 ;
                T0 = 1.0 + pParam->BSIM4v7prwg * vgd_eff ;
                dT0_dvg = -pParam->BSIM4v7prwg / T0 / T0 * dvgd_eff_dvg ;
                T1 = -pParam->BSIM4v7prwb * vbd ;
                dT1_dvb = -pParam->BSIM4v7prwb ;
                T2 = 1.0 / T0 + T1 ;
                T3 = T2 + sqrt (T2 * T2 + 0.01) ;
                dT3_dvg = T3 / (T3 - T2) ;
                dT3_dvb = dT3_dvg * dT1_dvb ;
                dT3_dvg *= dT0_dvg ;
                T4 = pParam->BSIM4v7rd0 * 0.5 ;
                Rd = pParam->BSIM4v7rdwmin + T3 * T4 ;
                dRd_dvg = T4 * dT3_dvg ;
                dRd_dvb = T4 * dT3_dvb ;
                T0 = 1.0 + BSIM4v7entry.d_BSIM4v7drainConductanceArray [instance_ID] * Rd ;
                BSIM4v7entry.d_BSIM4v7gdtotArray [instance_ID] = BSIM4v7entry.d_BSIM4v7drainConductanceArray [instance_ID] / T0 ;
                T0 = -BSIM4v7entry.d_BSIM4v7gdtotArray [instance_ID] * BSIM4v7entry.d_BSIM4v7gdtotArray [instance_ID] ;
                dgdtot_dvs = 0.0 ;
                dgdtot_dvg = T0 * dRd_dvg ;
                dgdtot_dvb = T0 * dRd_dvb ;
                dgdtot_dvd = -(dgdtot_dvg + dgdtot_dvb + dgdtot_dvs) ;
                BSIM4v7entry.d_BSIM4v7gstotdArray [instance_ID] = vses * dgstot_dvd ;
                BSIM4v7entry.d_BSIM4v7gstotgArray [instance_ID] = vses * dgstot_dvg ;
                BSIM4v7entry.d_BSIM4v7gstotsArray [instance_ID] = vses * dgstot_dvs ;
                BSIM4v7entry.d_BSIM4v7gstotbArray [instance_ID] = vses * dgstot_dvb ;
                T2 = vdes - vds ;
                BSIM4v7entry.d_BSIM4v7gdtotdArray [instance_ID] = T2 * dgdtot_dvd ;
                BSIM4v7entry.d_BSIM4v7gdtotgArray [instance_ID] = T2 * dgdtot_dvg ;
                BSIM4v7entry.d_BSIM4v7gdtotsArray [instance_ID] = T2 * dgdtot_dvs ;
                BSIM4v7entry.d_BSIM4v7gdtotbArray [instance_ID] = T2 * dgdtot_dvb ;
            }
            else /* WDLiu: for bypass */
            {
                BSIM4v7entry.d_BSIM4v7gstotArray [instance_ID] = BSIM4v7entry.d_BSIM4v7gstotdArray [instance_ID] = BSIM4v7entry.d_BSIM4v7gstotgArray [instance_ID] = 0.0 ;
                BSIM4v7entry.d_BSIM4v7gstotsArray [instance_ID] = BSIM4v7entry.d_BSIM4v7gstotbArray [instance_ID] = 0.0 ;
                BSIM4v7entry.d_BSIM4v7gdtotArray [instance_ID] = BSIM4v7entry.d_BSIM4v7gdtotdArray [instance_ID] = BSIM4v7entry.d_BSIM4v7gdtotgArray [instance_ID] = 0.0 ;
                BSIM4v7entry.d_BSIM4v7gdtotsArray [instance_ID] = BSIM4v7entry.d_BSIM4v7gdtotbArray [instance_ID] = 0.0 ;
            }


            /* GIDL/GISL Models */
            /* 9 - non-divergent */
            if (BSIM4v7mtrlMod == 0)
                T0 = 3.0 * toxe ;
            else
                T0 = BSIM4v7epsrsub * toxe / epsrox ;


            /* Calculate GIDL current */
            vgs_eff = BSIM4v7entry.d_BSIM4v7vgs_effArray [instance_ID] ;
            dvgs_eff_dvg = BSIM4v7entry.d_BSIM4v7dvgs_eff_dvgArray [instance_ID] ;
            vgd_eff = BSIM4v7entry.d_BSIM4v7vgd_effArray [instance_ID] ;
            dvgd_eff_dvg = BSIM4v7entry.d_BSIM4v7dvgd_eff_dvgArray [instance_ID] ;

            /* 10 - non-divergent */
            if (BSIM4v7gidlMod == 0)
            {
                /* 11 - non-divergent */
                if (BSIM4v7mtrlMod == 0)
                    T1 = (vds - vgs_eff - pParam->BSIM4v7egidl ) / T0 ;
                else
                    T1 = (vds - vgs_eff - pParam->BSIM4v7egidl + pParam->BSIM4v7vfbsd) / T0 ;

                /* 41 - DIVERGENT */
                if ((pParam->BSIM4v7agidl <= 0.0) || (pParam->BSIM4v7bgidl <= 0.0) ||
                    (T1 <= 0.0) || (pParam->BSIM4v7cgidl <= 0.0) || (vbd > 0.0))
                {
                    Igidl = Ggidld = Ggidlg = Ggidlb = 0.0 ;
                } else {
                    dT1_dVd = 1.0 / T0 ;
                    dT1_dVg = -dvgs_eff_dvg * dT1_dVd ;
                    T2 = pParam->BSIM4v7bgidl / T1 ;
                    if (T2 < 100.0)
                    {
                        Igidl = pParam->BSIM4v7agidl * pParam->BSIM4v7weffCJ * T1 * exp (-T2) ;
                        T3 = Igidl * (1.0 + T2) / T1 ;
                        Ggidld = T3 * dT1_dVd ;
                        Ggidlg = T3 * dT1_dVg ;
                    } else {
                        Igidl = pParam->BSIM4v7agidl * pParam->BSIM4v7weffCJ * 3.720075976e-44 ;
                        Ggidld = Igidl * dT1_dVd ;
                        Ggidlg = Igidl * dT1_dVg ;
                        Igidl *= T1 ;
                    }
                    T4 = vbd * vbd ;
                    T5 = -vbd * T4 ;
                    T6 = pParam->BSIM4v7cgidl + T5 ;
                    T7 = T5 / T6 ;
                    T8 = 3.0 * pParam->BSIM4v7cgidl * T4 / T6 / T6 ;
                    Ggidld = Ggidld * T7 + Igidl * T8 ;
                    Ggidlg = Ggidlg * T7 ;
                    Ggidlb = -Igidl * T8 ;
                    Igidl *= T7 ;
                }
                BSIM4v7entry.d_BSIM4v7IgidlRWArray [instance_ID] = Igidl ;
                BSIM4v7entry.d_BSIM4v7ggidldArray [instance_ID] = Ggidld ;
                BSIM4v7entry.d_BSIM4v7ggidlgArray [instance_ID] = Ggidlg ;
                BSIM4v7entry.d_BSIM4v7ggidlbArray [instance_ID] = Ggidlb ;


                /* Calculate GISL current  */
                /* 12 - non-divergent */
                if (BSIM4v7mtrlMod ==0)
                    T1 = (-vds - vgd_eff - pParam->BSIM4v7egisl ) / T0 ;
                else
                    T1 = (-vds - vgd_eff - pParam->BSIM4v7egisl + pParam->BSIM4v7vfbsd ) / T0 ;

                /* 42 - DIVERGENT */
                if ((pParam->BSIM4v7agisl <= 0.0) || (pParam->BSIM4v7bgisl <= 0.0) ||
                    (T1 <= 0.0) || (pParam->BSIM4v7cgisl <= 0.0) || (vbs > 0.0))
                {
                    Igisl = Ggisls = Ggislg = Ggislb = 0.0 ;
                } else {
                    dT1_dVd = 1.0 / T0 ;
                    dT1_dVg = -dvgd_eff_dvg * dT1_dVd ;
                    T2 = pParam->BSIM4v7bgisl / T1 ;
                    if (T2 < 100.0)
                    {
                        Igisl = pParam->BSIM4v7agisl * pParam->BSIM4v7weffCJ * T1 * exp (-T2) ;
                        T3 = Igisl * (1.0 + T2) / T1 ;
                        Ggisls = T3 * dT1_dVd ;
                        Ggislg = T3 * dT1_dVg ;
                    } else {
                        Igisl = pParam->BSIM4v7agisl * pParam->BSIM4v7weffCJ * 3.720075976e-44 ;
                        Ggisls = Igisl * dT1_dVd ;
                        Ggislg = Igisl * dT1_dVg ;
                        Igisl *= T1 ;
                    }
                    T4 = vbs * vbs ;
                    T5 = -vbs * T4 ;
                    T6 = pParam->BSIM4v7cgisl + T5 ;
                    T7 = T5 / T6 ;
                    T8 = 3.0 * pParam->BSIM4v7cgisl * T4 / T6 / T6 ;
                    Ggisls = Ggisls * T7 + Igisl * T8 ;
                    Ggislg = Ggislg * T7 ;
                    Ggislb = -Igisl * T8 ;
                    Igisl *= T7 ;
                }
                BSIM4v7entry.d_BSIM4v7IgislRWArray [instance_ID] = Igisl ;
                BSIM4v7entry.d_BSIM4v7ggislsArray [instance_ID] = Ggisls ;
                BSIM4v7entry.d_BSIM4v7ggislgArray [instance_ID] = Ggislg ;
                BSIM4v7entry.d_BSIM4v7ggislbArray [instance_ID] = Ggislb ;
            } else{    /* v4.7 New Gidl/GISL model */
                /* GISL */
                /* 13 - non-divergent */
                if (BSIM4v7mtrlMod==0)
                    T1 = (-vds - pParam->BSIM4v7rgisl * vgd_eff - pParam->BSIM4v7egisl) / T0 ;
                else
                    T1 = (-vds - pParam->BSIM4v7rgisl * vgd_eff - pParam->BSIM4v7egisl + pParam->BSIM4v7vfbsd) / T0 ;

                /* 43 - DIVERGENT */
                if ((pParam->BSIM4v7agisl <= 0.0) || (pParam->BSIM4v7bgisl <= 0.0) || (T1 <= 0.0) ||
                    (pParam->BSIM4v7cgisl < 0.0))
                {
                    Igisl = Ggisls = Ggislg = Ggislb = 0.0 ;
                } else {
                    dT1_dVd = 1 / T0 ;
                    dT1_dVg = - pParam->BSIM4v7rgisl * dT1_dVd * dvgd_eff_dvg ;
                    T2 = pParam->BSIM4v7bgisl / T1 ;
                    if (T2 < EXPL_THRESHOLD)
                    {
                        Igisl = pParam->BSIM4v7weffCJ * pParam->BSIM4v7agisl * T1 * exp (-T2) ;
                        T3 = Igisl / T1 * (T2 + 1) ;
                        Ggisls = T3 * dT1_dVd ;
                        Ggislg = T3 * dT1_dVg ;
                    }  else {
                        T3 = pParam->BSIM4v7weffCJ * pParam->BSIM4v7agisl * MIN_EXPL ;
                        Igisl = T3 * T1 ;
                        Ggisls  = T3 * dT1_dVd ;
                        Ggislg  = T3 * dT1_dVg ;
                    }
                    T4 = vbs - pParam->BSIM4v7fgisl ;

                    if (T4 == 0)
                        T5 = EXPL_THRESHOLD ;
                    else
                        T5 = pParam->BSIM4v7kgisl / T4 ;

                    if (T5 < EXPL_THRESHOLD)
                    {
                        T6 = exp (T5) ;
                        Ggislb = -Igisl * T6 * T5 / T4 ;
                    } else {
                        T6 = MAX_EXPL ;
                        Ggislb = 0.0 ;
                    }
                    Ggisls *= T6 ;
                    Ggislg *= T6 ;
                    Igisl *= T6 ;
                }
                BSIM4v7entry.d_BSIM4v7IgislRWArray [instance_ID] = Igisl ;
                BSIM4v7entry.d_BSIM4v7ggislsArray [instance_ID] = Ggisls ;
                BSIM4v7entry.d_BSIM4v7ggislgArray [instance_ID] = Ggislg ;
                BSIM4v7entry.d_BSIM4v7ggislbArray [instance_ID] = Ggislb ;
                /* End of GISL */

                /* GIDL */
                /* 14 - non-divergent */
                if (BSIM4v7mtrlMod == 0)
                    T1 = (vds - pParam->BSIM4v7rgidl * vgs_eff - pParam->BSIM4v7egidl) / T0 ;
                else
                    T1 = (vds - pParam->BSIM4v7rgidl * vgs_eff - pParam->BSIM4v7egidl + pParam->BSIM4v7vfbsd) / T0 ;

                /* 44 - DIVERGENT */
                if ((pParam->BSIM4v7agidl <= 0.0) || (pParam->BSIM4v7bgidl <= 0.0) || (T1 <= 0.0) ||
                    (pParam->BSIM4v7cgidl < 0.0))
                {
                    Igidl = Ggidld = Ggidlg = Ggidlb = 0.0 ;
                } else {
                    dT1_dVd = 1 / T0 ;
                    dT1_dVg = - pParam->BSIM4v7rgidl * dT1_dVd * dvgs_eff_dvg ;
                    T2 = pParam->BSIM4v7bgidl / T1 ;
                    if (T2 < EXPL_THRESHOLD)
                    {
                        Igidl = pParam->BSIM4v7weffCJ * pParam->BSIM4v7agidl * T1 * exp (-T2) ;
                        T3 = Igidl / T1 * (T2 + 1) ;
                        Ggidld = T3 * dT1_dVd ;
                        Ggidlg = T3 * dT1_dVg ;
                    } else {
                        T3 = pParam->BSIM4v7weffCJ * pParam->BSIM4v7agidl * MIN_EXPL ;
                        Igidl = T3 * T1 ;
                        Ggidld  = T3 * dT1_dVd ;
                        Ggidlg  = T3 * dT1_dVg ;
                    }
                    T4 = vbd - pParam->BSIM4v7fgidl ;

                    if (T4 == 0)
                        T5 = EXPL_THRESHOLD ;
                    else
                        T5 = pParam->BSIM4v7kgidl / T4 ;

                    if (T5<EXPL_THRESHOLD)
                    {
                        T6 = exp (T5) ;
                        Ggidlb = -Igidl * T6 * T5 / T4 ;
                    } else {
                        T6 = MAX_EXPL ;
                        Ggidlb = 0.0 ;
                    }
                    Ggidld *= T6 ;
                    Ggidlg *= T6 ;
                    Igidl *= T6 ;
                }
                BSIM4v7entry.d_BSIM4v7IgidlRWArray [instance_ID] = Igidl ;
                BSIM4v7entry.d_BSIM4v7ggidldArray [instance_ID] = Ggidld ;
                BSIM4v7entry.d_BSIM4v7ggidlgArray [instance_ID] = Ggidlg ;
                BSIM4v7entry.d_BSIM4v7ggidlbArray [instance_ID] = Ggidlb ;
                /* End of New GIDL */
            }
            /* End of Gidl */


            /* Calculate gate tunneling current */
            /* 15 non-divergent */
            if ((BSIM4v7igcMod != 0) || (BSIM4v7igbMod != 0))
            {
                Vfb = BSIM4v7entry.d_BSIM4v7vfbzbArray [instance_ID] ;
                V3 = Vfb - Vgs_eff + Vbseff - DELTA_3 ;

                if (Vfb <= 0.0)
                    T0 = sqrt (V3 * V3 - 4.0 * DELTA_3 * Vfb) ;
                else
                    T0 = sqrt (V3 * V3 + 4.0 * DELTA_3 * Vfb) ;

                T1 = 0.5 * (1.0 + V3 / T0) ;
                Vfbeff = Vfb - 0.5 * (V3 + T0) ;
                dVfbeff_dVg = T1 * dVgs_eff_dVg ;
                dVfbeff_dVb = -T1 ;    /* WDLiu: -No surprise? No. -Good! */
                Voxacc = Vfb - Vfbeff ;
                dVoxacc_dVg = -dVfbeff_dVg ;
                dVoxacc_dVb = -dVfbeff_dVb ;
                if (Voxacc < 0.0)    /* WDLiu: Avoiding numerical instability. */
                    Voxacc = dVoxacc_dVg = dVoxacc_dVb = 0.0 ;

                T0 = 0.5 * pParam->BSIM4v7k1ox ;
                T3 = Vgs_eff - Vfbeff - Vbseff - Vgsteff ;

                /* 45 - DIVERGENT */
                if (pParam->BSIM4v7k1ox == 0.0)
                    Voxdepinv = dVoxdepinv_dVg = dVoxdepinv_dVd = dVoxdepinv_dVb = 0.0 ;
                else if (T3 < 0.0)
                {
                    Voxdepinv = -T3 ;
                    dVoxdepinv_dVg = -dVgs_eff_dVg + dVfbeff_dVg + dVgsteff_dVg ;
                    dVoxdepinv_dVd = dVgsteff_dVd ;
                    dVoxdepinv_dVb = dVfbeff_dVb + 1.0 + dVgsteff_dVb ;
                }
                else
                {
                    T1 = sqrt (T0 * T0 + T3) ;
                    T2 = T0 / T1 ;
                    Voxdepinv = pParam->BSIM4v7k1ox * (T1 - T0) ;
                    dVoxdepinv_dVg = T2 * (dVgs_eff_dVg - dVfbeff_dVg - dVgsteff_dVg) ;
                    dVoxdepinv_dVd = -T2 * dVgsteff_dVd ;
                    dVoxdepinv_dVb = -T2 * (dVfbeff_dVb + 1.0 + dVgsteff_dVb) ;
                }
                Voxdepinv += Vgsteff ;
                dVoxdepinv_dVg += dVgsteff_dVg ;
                dVoxdepinv_dVd += dVgsteff_dVd ;
                dVoxdepinv_dVb += dVgsteff_dVb ;
            }

            /* 16 - non-divergent */
            if (BSIM4v7tempMod < 2)
                tmp = Vtm ;
            else    /* BSIM4v7tempMod = 2 , 3*/
                tmp = Vtm0 ;

            /* 17 - non-divergent */
            if (BSIM4v7igcMod)
            {
                T0 = tmp * pParam->BSIM4v7nigc ;
                if (BSIM4v7igcMod == 1)
                {
                    VxNVt = (Vgs_eff - BSIM4v7type * BSIM4v7entry.d_BSIM4v7vth0Array [instance_ID]) / T0 ;

                    /* 46 - DIVERGENT */
                    if (VxNVt > EXP_THRESHOLD)
                    {
                        Vaux = Vgs_eff - BSIM4v7type * BSIM4v7entry.d_BSIM4v7vth0Array [instance_ID] ;
                        dVaux_dVg = dVgs_eff_dVg ;
                        dVaux_dVd = 0.0 ;
                        dVaux_dVb = 0.0 ;
                    }
                }
                else if (BSIM4v7igcMod == 2)
                {
                    VxNVt = (Vgs_eff - BSIM4v7entry.d_BSIM4v7vonRWArray [instance_ID]) / T0 ;

                    /* 47 - DIVERGENT */
                    if (VxNVt > EXP_THRESHOLD)
                    {
                        Vaux = Vgs_eff - BSIM4v7entry.d_BSIM4v7vonRWArray [instance_ID] ;
                        dVaux_dVg = dVgs_eff_dVg ;
                        dVaux_dVd = -dVth_dVd ;
                        dVaux_dVb = -dVth_dVb ;
                    }
                }

                /* 48 - DIVERGENT */
                if (VxNVt < -EXP_THRESHOLD)
                {
                    Vaux = T0 * log (1.0 + MIN_EXP) ;
                    dVaux_dVg = dVaux_dVd = dVaux_dVb = 0.0 ;
                }
                else if ((VxNVt >= -EXP_THRESHOLD) && (VxNVt <= EXP_THRESHOLD))
                {
                    ExpVxNVt = exp (VxNVt) ;
                    Vaux = T0 * log (1.0 + ExpVxNVt) ;
                    dVaux_dVg = ExpVxNVt / (1.0 + ExpVxNVt) ;
                    if (BSIM4v7igcMod == 1)
                    {
                        dVaux_dVd = 0.0 ;
                        dVaux_dVb = 0.0 ;
                    } else if (BSIM4v7igcMod == 2)
                    {
                        dVaux_dVd = -dVgs_eff_dVg * dVth_dVd ;
                        dVaux_dVb = -dVgs_eff_dVg * dVth_dVb ;
                    }
                    dVaux_dVg *= dVgs_eff_dVg ;
                }
                T2 = Vgs_eff * Vaux ;
                dT2_dVg = dVgs_eff_dVg * Vaux + Vgs_eff * dVaux_dVg ;
                dT2_dVd = Vgs_eff * dVaux_dVd ;
                dT2_dVb = Vgs_eff * dVaux_dVb ;
                T11 = pParam->BSIM4v7Aechvb ;
                T12 = pParam->BSIM4v7Bechvb ;
                T3 = pParam->BSIM4v7aigc * pParam->BSIM4v7cigc - pParam->BSIM4v7bigc ;
                T4 = pParam->BSIM4v7bigc * pParam->BSIM4v7cigc ;
                T5 = T12 * (pParam->BSIM4v7aigc + T3 * Voxdepinv - T4 * Voxdepinv * Voxdepinv) ;

                /* 49 - DIVERGENT */
                if (T5 > EXP_THRESHOLD)
                {
                    T6 = MAX_EXP ;
                    dT6_dVg = dT6_dVd = dT6_dVb = 0.0 ;
                }
                else if (T5 < -EXP_THRESHOLD)
                {   T6 = MIN_EXP ;
                    dT6_dVg = dT6_dVd = dT6_dVb = 0.0 ;
                }
                else
                {
                    T6 = exp (T5) ;
                    dT6_dVg = T6 * T12 * (T3 - 2.0 * T4 * Voxdepinv) ;
                    dT6_dVd = dT6_dVg * dVoxdepinv_dVd ;
                    dT6_dVb = dT6_dVg * dVoxdepinv_dVb ;
                    dT6_dVg *= dVoxdepinv_dVg ;
                }
                Igc = T11 * T2 * T6 ;
                dIgc_dVg = T11 * (T2 * dT6_dVg + T6 * dT2_dVg) ;
                dIgc_dVd = T11 * (T2 * dT6_dVd + T6 * dT2_dVd) ;
                dIgc_dVb = T11 * (T2 * dT6_dVb + T6 * dT2_dVb) ;

                if (BSIM4v7pigcdGiven)
                {
                    Pigcd = pParam->BSIM4v7pigcd ;
                    dPigcd_dVg = dPigcd_dVd = dPigcd_dVb = 0.0 ;
                } else {
                    /* T11 = pParam->BSIM4v7Bechvb * toxe; v4.7 */
                    T11 = -pParam->BSIM4v7Bechvb ;
                    T12 = Vgsteff + 1.0e-20 ;
                    T13 = T11 / T12 / T12 ;
                    T14 = -T13 / T12 ;
                    Pigcd = T13 * (1.0 - 0.5 * Vdseff / T12) ;
                    dPigcd_dVg = T14 * (2.0 + 0.5 * (dVdseff_dVg - 3.0 * Vdseff / T12)) ;
                    dPigcd_dVd = 0.5 * T14 * dVdseff_dVd ;
                    dPigcd_dVb = 0.5 * T14 * dVdseff_dVb ;
                }

                /* bugfix */
                T7 = -Pigcd * Vdseff ;
                /* ------ */

                dT7_dVg = -Vdseff * dPigcd_dVg - Pigcd * dVdseff_dVg ;
                dT7_dVd = -Vdseff * dPigcd_dVd - Pigcd * dVdseff_dVd + dT7_dVg * dVgsteff_dVd ;
                dT7_dVb = -Vdseff * dPigcd_dVb - Pigcd * dVdseff_dVb + dT7_dVg * dVgsteff_dVb ;
                dT7_dVg *= dVgsteff_dVg ;
                dT7_dVb *= dVbseff_dVb ;
                T8 = T7 * T7 + 2.0e-4 ;
                dT8_dVg = 2.0 * T7 ;
                dT8_dVd = dT8_dVg * dT7_dVd ;
                dT8_dVb = dT8_dVg * dT7_dVb ;
                dT8_dVg *= dT7_dVg ;

                /* 50 - DIVERGENT */
                if (T7 > EXP_THRESHOLD)
                {
                    T9 = MAX_EXP ;
                    dT9_dVg = dT9_dVd = dT9_dVb = 0.0 ;
                }
                else if (T7 < -EXP_THRESHOLD)
                {
                    T9 = MIN_EXP ;
                    dT9_dVg = dT9_dVd = dT9_dVb = 0.0 ;
                }
                else
                {
                    T9 = exp (T7) ;
                    dT9_dVg = T9 * dT7_dVg ;
                    dT9_dVd = T9 * dT7_dVd ;
                    dT9_dVb = T9 * dT7_dVb ;
                }
                T0 = T8 * T8 ;
                T1 = T9 - 1.0 + 1.0e-4 ;
                T10 = (T1 - T7) / T8 ;
                dT10_dVg = (dT9_dVg - dT7_dVg - T10 * dT8_dVg) / T8 ;
                dT10_dVd = (dT9_dVd - dT7_dVd - T10 * dT8_dVd) / T8 ;
                dT10_dVb = (dT9_dVb - dT7_dVb - T10 * dT8_dVb) / T8 ;
                Igcs = Igc * T10 ;
                dIgcs_dVg = dIgc_dVg * T10 + Igc * dT10_dVg ;
                dIgcs_dVd = dIgc_dVd * T10 + Igc * dT10_dVd ;
                dIgcs_dVb = dIgc_dVb * T10 + Igc * dT10_dVb ;
                T1 = T9 - 1.0 - 1.0e-4 ;
                T10 = (T7 * T9 - T1) / T8 ;
                dT10_dVg = (dT7_dVg * T9 + (T7 - 1.0) * dT9_dVg - T10 * dT8_dVg) / T8 ;
                dT10_dVd = (dT7_dVd * T9 + (T7 - 1.0) * dT9_dVd - T10 * dT8_dVd) / T8 ;
                dT10_dVb = (dT7_dVb * T9 + (T7 - 1.0) * dT9_dVb - T10 * dT8_dVb) / T8 ;
                Igcd = Igc * T10 ;
                dIgcd_dVg = dIgc_dVg * T10 + Igc * dT10_dVg ;
                dIgcd_dVd = dIgc_dVd * T10 + Igc * dT10_dVd ;
                dIgcd_dVb = dIgc_dVb * T10 + Igc * dT10_dVb ;
                BSIM4v7entry.d_BSIM4v7IgcsRWArray [instance_ID] = Igcs ;
                BSIM4v7entry.d_BSIM4v7gIgcsgArray [instance_ID] = dIgcs_dVg ;
                BSIM4v7entry.d_BSIM4v7gIgcsdArray [instance_ID] = dIgcs_dVd ;
                BSIM4v7entry.d_BSIM4v7gIgcsbArray [instance_ID] =  dIgcs_dVb * dVbseff_dVb ;
                BSIM4v7entry.d_BSIM4v7IgcdRWArray [instance_ID] = Igcd ;
                BSIM4v7entry.d_BSIM4v7gIgcdgArray [instance_ID] = dIgcd_dVg ;
                BSIM4v7entry.d_BSIM4v7gIgcddArray [instance_ID] = dIgcd_dVd ;
                BSIM4v7entry.d_BSIM4v7gIgcdbArray [instance_ID] = dIgcd_dVb * dVbseff_dVb ;
                T0 = vgs - (pParam->BSIM4v7vfbsd + pParam->BSIM4v7vfbsdoff) ;
                vgs_eff = sqrt (T0 * T0 + 1.0e-4) ;
                dvgs_eff_dvg = T0 / vgs_eff ;
                T2 = vgs * vgs_eff ;
                dT2_dVg = vgs * dvgs_eff_dvg + vgs_eff ;
                T11 = pParam->BSIM4v7AechvbEdgeS ;
                T12 = pParam->BSIM4v7BechvbEdge ;
                T3 = pParam->BSIM4v7aigs * pParam->BSIM4v7cigs - pParam->BSIM4v7bigs ;
                T4 = pParam->BSIM4v7bigs * pParam->BSIM4v7cigs ;
                T5 = T12 * (pParam->BSIM4v7aigs + T3 * vgs_eff - T4 * vgs_eff * vgs_eff) ;

                /* 51 - DIVERGENT */
                if (T5 > EXP_THRESHOLD)
                {
                    T6 = MAX_EXP ;
                    dT6_dVg = 0.0 ;
                }
                else if (T5 < -EXP_THRESHOLD)
                {
                    T6 = MIN_EXP ;
                    dT6_dVg = 0.0 ;
                } else {
                    T6 = exp (T5) ;
                    dT6_dVg = T6 * T12 * (T3 - 2.0 * T4 * vgs_eff) * dvgs_eff_dvg ;
                }
                Igs = T11 * T2 * T6 ;
                dIgs_dVg = T11 * (T2 * dT6_dVg + T6 * dT2_dVg) ;
                dIgs_dVs = -dIgs_dVg ;
                T0 = vgd - (pParam->BSIM4v7vfbsd + pParam->BSIM4v7vfbsdoff) ;
                vgd_eff = sqrt (T0 * T0 + 1.0e-4) ;
                dvgd_eff_dvg = T0 / vgd_eff ;
                T2 = vgd * vgd_eff ;
                dT2_dVg = vgd * dvgd_eff_dvg + vgd_eff ;
                T11 = pParam->BSIM4v7AechvbEdgeD ;
                T3 = pParam->BSIM4v7aigd * pParam->BSIM4v7cigd - pParam->BSIM4v7bigd ;
                T4 = pParam->BSIM4v7bigd * pParam->BSIM4v7cigd ;
                T5 = T12 * (pParam->BSIM4v7aigd + T3 * vgd_eff - T4 * vgd_eff * vgd_eff) ;

                /* 52 - DIVERGENT */
                if (T5 > EXP_THRESHOLD)
                {
                    T6 = MAX_EXP ;
                    dT6_dVg = 0.0 ;
                }
                else if (T5 < -EXP_THRESHOLD)
                {
                    T6 = MIN_EXP ;
                    dT6_dVg = 0.0 ;
                } else {
                    T6 = exp (T5) ;
                    dT6_dVg = T6 * T12 * (T3 - 2.0 * T4 * vgd_eff) * dvgd_eff_dvg ;
                }
                Igd = T11 * T2 * T6 ;
                dIgd_dVg = T11 * (T2 * dT6_dVg + T6 * dT2_dVg) ;
                dIgd_dVd = -dIgd_dVg ;
                BSIM4v7entry.d_BSIM4v7IgsRWArray [instance_ID] = Igs ;
                BSIM4v7entry.d_BSIM4v7gIgsgArray [instance_ID] = dIgs_dVg ;
                BSIM4v7entry.d_BSIM4v7gIgssArray [instance_ID] = dIgs_dVs ;
                BSIM4v7entry.d_BSIM4v7IgdRWArray [instance_ID] = Igd ;
                BSIM4v7entry.d_BSIM4v7gIgdgArray [instance_ID] = dIgd_dVg ;
                BSIM4v7entry.d_BSIM4v7gIgddArray [instance_ID] = dIgd_dVd ;
            } else {
                BSIM4v7entry.d_BSIM4v7IgcsRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7gIgcsgArray [instance_ID] = BSIM4v7entry.d_BSIM4v7gIgcsdArray [instance_ID] = BSIM4v7entry.d_BSIM4v7gIgcsbArray [instance_ID] = 0.0 ;
                BSIM4v7entry.d_BSIM4v7IgcdRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7gIgcdgArray [instance_ID] = BSIM4v7entry.d_BSIM4v7gIgcddArray [instance_ID] = BSIM4v7entry.d_BSIM4v7gIgcdbArray [instance_ID] = 0.0 ;
                BSIM4v7entry.d_BSIM4v7IgsRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7gIgsgArray [instance_ID] = BSIM4v7entry.d_BSIM4v7gIgssArray [instance_ID] = 0.0 ;
                BSIM4v7entry.d_BSIM4v7IgdRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7gIgdgArray [instance_ID] = BSIM4v7entry.d_BSIM4v7gIgddArray [instance_ID] = 0.0 ;
            }

            /* 18 - non-divergent */
            if (BSIM4v7igbMod)
            {
                T0 = tmp * pParam->BSIM4v7nigbacc ;
                T1 = -Vgs_eff + Vbseff + Vfb ;
                VxNVt = T1 / T0 ;

                /* 53 - DIVERGENT */
                if (VxNVt > EXP_THRESHOLD)
                {
                    Vaux = T1 ;
                    dVaux_dVg = -dVgs_eff_dVg ;
                    dVaux_dVb = 1.0 ;
                }
                else if (VxNVt < -EXP_THRESHOLD)
                {
                    Vaux = T0 * log (1.0 + MIN_EXP) ;
                    dVaux_dVg = dVaux_dVb = 0.0 ;
                } else {
                    ExpVxNVt = exp (VxNVt) ;
                    Vaux = T0 * log (1.0 + ExpVxNVt) ;
                    dVaux_dVb = ExpVxNVt / (1.0 + ExpVxNVt) ;
                    dVaux_dVg = -dVaux_dVb * dVgs_eff_dVg ;
                }
                T2 = (Vgs_eff - Vbseff) * Vaux ;
                dT2_dVg = dVgs_eff_dVg * Vaux + (Vgs_eff - Vbseff) * dVaux_dVg ;
                dT2_dVb = -Vaux + (Vgs_eff - Vbseff) * dVaux_dVb ;
                T11 = 4.97232e-7 * pParam->BSIM4v7weff * pParam->BSIM4v7leff * pParam->BSIM4v7ToxRatio ;
                T12 = -7.45669e11 * toxe ;
                T3 = pParam->BSIM4v7aigbacc * pParam->BSIM4v7cigbacc - pParam->BSIM4v7bigbacc ;
                T4 = pParam->BSIM4v7bigbacc * pParam->BSIM4v7cigbacc ;
                T5 = T12 * (pParam->BSIM4v7aigbacc + T3 * Voxacc - T4 * Voxacc * Voxacc) ;

                /* 54 - DIVERGENT */
                if (T5 > EXP_THRESHOLD)
                {
                    T6 = MAX_EXP ;
                    dT6_dVg = dT6_dVb = 0.0 ;
                }
                else if (T5 < -EXP_THRESHOLD)
                {
                    T6 = MIN_EXP ;
                    dT6_dVg = dT6_dVb = 0.0 ;
                } else {
                    T6 = exp (T5) ;
                    dT6_dVg = T6 * T12 * (T3 - 2.0 * T4 * Voxacc) ;
                    dT6_dVb = dT6_dVg * dVoxacc_dVb ;
                    dT6_dVg *= dVoxacc_dVg ;
                }
                Igbacc = T11 * T2 * T6 ;
                dIgbacc_dVg = T11 * (T2 * dT6_dVg + T6 * dT2_dVg) ;
                dIgbacc_dVb = T11 * (T2 * dT6_dVb + T6 * dT2_dVb) ;
                T0 = tmp * pParam->BSIM4v7nigbinv ;
                T1 = Voxdepinv - pParam->BSIM4v7eigbinv ;
                VxNVt = T1 / T0 ;

                /* 55 - DIVERGENT */
                if (VxNVt > EXP_THRESHOLD)
                {
                    Vaux = T1 ;
                    dVaux_dVg = dVoxdepinv_dVg ;
                    dVaux_dVd = dVoxdepinv_dVd ;
                    dVaux_dVb = dVoxdepinv_dVb ;
                }
                else if (VxNVt < -EXP_THRESHOLD)
                {
                    Vaux = T0 * log (1.0 + MIN_EXP) ;
                    dVaux_dVg = dVaux_dVd = dVaux_dVb = 0.0 ;
                } else {
                    ExpVxNVt = exp (VxNVt) ;
                    Vaux = T0 * log (1.0 + ExpVxNVt) ;
                    dVaux_dVg = ExpVxNVt / (1.0 + ExpVxNVt) ;
                    dVaux_dVd = dVaux_dVg * dVoxdepinv_dVd ;
                    dVaux_dVb = dVaux_dVg * dVoxdepinv_dVb ;
                    dVaux_dVg *= dVoxdepinv_dVg ;
                }
                T2 = (Vgs_eff - Vbseff) * Vaux ;
                dT2_dVg = dVgs_eff_dVg * Vaux + (Vgs_eff - Vbseff) * dVaux_dVg ;
                dT2_dVd = (Vgs_eff - Vbseff) * dVaux_dVd ;
                dT2_dVb = -Vaux + (Vgs_eff - Vbseff) * dVaux_dVb ;
                T11 *= 0.75610 ;
                T12 *= 1.31724 ;
                T3 = pParam->BSIM4v7aigbinv * pParam->BSIM4v7cigbinv - pParam->BSIM4v7bigbinv ;
                T4 = pParam->BSIM4v7bigbinv * pParam->BSIM4v7cigbinv ;
                T5 = T12 * (pParam->BSIM4v7aigbinv + T3 * Voxdepinv - T4 * Voxdepinv * Voxdepinv) ;

                /* 56 - DIVERGENT */
                if (T5 > EXP_THRESHOLD)
                {
                    T6 = MAX_EXP ;
                    dT6_dVg = dT6_dVd = dT6_dVb = 0.0 ;
                }
                else if (T5 < -EXP_THRESHOLD)
                {
                    T6 = MIN_EXP ;
                    dT6_dVg = dT6_dVd = dT6_dVb = 0.0 ;
                } else {
                    T6 = exp (T5) ;
                    dT6_dVg = T6 * T12 * (T3 - 2.0 * T4 * Voxdepinv) ;
                    dT6_dVd = dT6_dVg * dVoxdepinv_dVd ;
                    dT6_dVb = dT6_dVg * dVoxdepinv_dVb ;
                    dT6_dVg *= dVoxdepinv_dVg ;
                }
                Igbinv = T11 * T2 * T6 ;
                dIgbinv_dVg = T11 * (T2 * dT6_dVg + T6 * dT2_dVg) ;
                dIgbinv_dVd = T11 * (T2 * dT6_dVd + T6 * dT2_dVd) ;
                dIgbinv_dVb = T11 * (T2 * dT6_dVb + T6 * dT2_dVb) ;
                BSIM4v7entry.d_BSIM4v7IgbRWArray [instance_ID] = Igbinv + Igbacc ;
                BSIM4v7entry.d_BSIM4v7gIgbgArray [instance_ID] = dIgbinv_dVg + dIgbacc_dVg ;
                BSIM4v7entry.d_BSIM4v7gIgbdArray [instance_ID] = dIgbinv_dVd ;
                BSIM4v7entry.d_BSIM4v7gIgbbArray [instance_ID] = (dIgbinv_dVb + dIgbacc_dVb) * dVbseff_dVb ;
            } else {
                BSIM4v7entry.d_BSIM4v7IgbRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7gIgbgArray [instance_ID] = BSIM4v7entry.d_BSIM4v7gIgbdArray [instance_ID] = BSIM4v7entry.d_BSIM4v7gIgbsArray [instance_ID] = BSIM4v7entry.d_BSIM4v7gIgbbArray [instance_ID] = 0.0 ;
            }
            /* End of Gate current */

            /* 57 - DIVERGENT */
            if (BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] != 1.0)
            {
                cdrain *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7gdsRWArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7gmRWArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7gmbsRWArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7IdovVdsArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7gbbsArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7gbgsArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7gbdsArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7csubRWArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7IgidlRWArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7ggidldArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7ggidlgArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7ggidlbArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7IgislRWArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7ggislsArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7ggislgArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7ggislbArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7IgcsRWArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7gIgcsgArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7gIgcsdArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7gIgcsbArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7IgcdRWArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7gIgcdgArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7gIgcddArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7gIgcdbArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7IgsRWArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7gIgsgArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7gIgssArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7IgdRWArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7gIgdgArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7gIgddArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7IgbRWArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7gIgbgArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7gIgbdArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                BSIM4v7entry.d_BSIM4v7gIgbbArray [instance_ID] *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
            }
            BSIM4v7entry.d_BSIM4v7ggidlsArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7ggidldArray [instance_ID] + BSIM4v7entry.d_BSIM4v7ggidlgArray [instance_ID] + BSIM4v7entry.d_BSIM4v7ggidlbArray [instance_ID]) ;
            BSIM4v7entry.d_BSIM4v7ggisldArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7ggislsArray [instance_ID] + BSIM4v7entry.d_BSIM4v7ggislgArray [instance_ID] + BSIM4v7entry.d_BSIM4v7ggislbArray [instance_ID]) ;
            BSIM4v7entry.d_BSIM4v7gIgbsArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7gIgbgArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gIgbdArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gIgbbArray [instance_ID]) ;
            BSIM4v7entry.d_BSIM4v7gIgcssArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7gIgcsgArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gIgcsdArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gIgcsbArray [instance_ID]) ;
            BSIM4v7entry.d_BSIM4v7gIgcdsArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7gIgcdgArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gIgcddArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gIgcdbArray [instance_ID]) ;
            BSIM4v7entry.d_BSIM4v7cdRWArray [instance_ID] = cdrain ;


            /* Calculations for noise analysis */
            /* 19 - non-divergent */
            if (BSIM4v7tnoiMod == 0)
            {
                Abulk = Abulk0 * pParam->BSIM4v7abulkCVfactor ;
                Vdsat = Vgsteff / Abulk ;
                T0 = Vdsat - Vds - DELTA_4 ;
                T1 = sqrt (T0 * T0 + 4.0 * DELTA_4 * Vdsat) ;

                /* 58 - DIVERGENT */
                if (T0 >= 0.0)
                    Vdseff = Vdsat - 0.5 * (T0 + T1) ;
                else
                {
                    T3 = (DELTA_4 + DELTA_4) / (T1 - T0) ;
                    T4 = 1.0 - T3 ;
                    T5 = Vdsat * T3 / (T1 - T0) ;
                    Vdseff = Vdsat * T4 ;
                }
                if (Vds == 0.0)
                    Vdseff = 0.0 ;

                T0 = Abulk * Vdseff ;
                T1 = 12.0 * (Vgsteff - 0.5 * T0 + 1.0e-20) ;
                T2 = Vdseff / T1 ;
                T3 = T0 * T2 ;
                BSIM4v7entry.d_BSIM4v7qinvRWArray [instance_ID] = Coxeff * pParam->BSIM4v7weffCV * BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] * pParam->BSIM4v7leffCV *
                                  (Vgsteff - 0.5 * T0 + Abulk * T3) ;
            }
            else if(BSIM4v7tnoiMod == 2)
                BSIM4v7entry.d_BSIM4v7noiGd0Array [instance_ID] = BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] * beta * Vgsteff / (1.0 + gche * Rds) ;


            /* BSIM4v7 C-V begins */
            /* 20 - non-divergent */
            if ((BSIM4v7xpart < 0) || (!ChargeComputationNeeded))
            {
                qgate  = qdrn = qsrc = qbulk = 0.0 ;
                BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] = 0.0 ;
                BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] = 0.0 ;
                BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7cbsbRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7cbdbRWArray [instance_ID] = 0.0 ;
                BSIM4v7entry.d_BSIM4v7csgbRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7cssbRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7csdbRWArray [instance_ID] = 0.0 ;
                BSIM4v7entry.d_BSIM4v7cgbbRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7csbbRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7cdbbRWArray [instance_ID] = BSIM4v7entry.d_BSIM4v7cbbbRWArray [instance_ID] = 0.0 ;
                BSIM4v7entry.d_BSIM4v7cqdbArray [instance_ID] = BSIM4v7entry.d_BSIM4v7cqsbArray [instance_ID] = BSIM4v7entry.d_BSIM4v7cqgbArray [instance_ID] = BSIM4v7entry.d_BSIM4v7cqbbArray [instance_ID] = 0.0 ;
                BSIM4v7entry.d_BSIM4v7gtauRWArray [instance_ID] = 0.0 ;
                goto finished ;
            }
            else if (BSIM4v7capMod == 0)
            {
                if (Vbseff < 0.0)
                {
                    VbseffCV = Vbs ; /*4.6.2*/
                    dVbseffCV_dVb = 1.0 ;
                } else {
                    VbseffCV = pParam->BSIM4v7phi - Phis ;
                    dVbseffCV_dVb = -dPhis_dVb * dVbseff_dVb ; /*4.6.2*/
                }
                Vfb = pParam->BSIM4v7vfbcv ;
                Vth = Vfb + pParam->BSIM4v7phi + pParam->BSIM4v7k1ox * sqrtPhis ;
                Vgst = Vgs_eff - Vth ;
                dVth_dVb = pParam->BSIM4v7k1ox * dsqrtPhis_dVb *dVbseff_dVb ; /*4.6.2*/
                CoxWL = BSIM4v7coxe * pParam->BSIM4v7weffCV * pParam->BSIM4v7leffCV * BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                Arg1 = Vgs_eff - VbseffCV - Vfb ;

                /* 59 - DIVERGENT */
                if (Arg1 <= 0.0)
                {
                    qgate = CoxWL * Arg1 ;
                    qbulk = -qgate ;
                    qdrn = 0.0 ;
                    BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] = CoxWL * dVgs_eff_dVg ;
                    BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] = 0.0 ;
                    BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] = CoxWL * (dVbseffCV_dVb - dVgs_eff_dVg) ;
                    BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] = 0.0 ;
                    BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] = 0.0 ;
                    BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID] = 0.0 ;
                    BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] = -CoxWL * dVgs_eff_dVg ;
                    BSIM4v7entry.d_BSIM4v7cbdbRWArray [instance_ID] = 0.0 ;
                    BSIM4v7entry.d_BSIM4v7cbsbRWArray [instance_ID] = -BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] ;
                } /* Arg1 <= 0.0, end of accumulation */
                else if (Vgst <= 0.0)
                {
                    T1 = 0.5 * pParam->BSIM4v7k1ox ;
                    T2 = sqrt (T1 * T1 + Arg1) ;
                    qgate = CoxWL * pParam->BSIM4v7k1ox * (T2 - T1) ;
                    qbulk = -qgate ;
                    qdrn = 0.0 ;
                    T0 = CoxWL * T1 / T2 ;
                    BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] = T0 * dVgs_eff_dVg ;
                    BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] = 0.0 ;
                    BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] = T0 * (dVbseffCV_dVb - dVgs_eff_dVg) ;
                    BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] = 0.0 ;
                    BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] = 0.0 ;
                    BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID] = 0.0 ;
                    BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] = -BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] ;
                   BSIM4v7entry.d_BSIM4v7cbdbRWArray [instance_ID] = 0.0 ;
                    BSIM4v7entry.d_BSIM4v7cbsbRWArray [instance_ID] = -BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] ;
                } /* Vgst <= 0.0, end of depletion */
                else
                {
                    One_Third_CoxWL = CoxWL / 3.0 ;
                    Two_Third_CoxWL = 2.0 * One_Third_CoxWL ;
                    AbulkCV = Abulk0 * pParam->BSIM4v7abulkCVfactor ;
                    dAbulkCV_dVb = pParam->BSIM4v7abulkCVfactor * dAbulk0_dVb*dVbseff_dVb ;
                    dVdsat_dVg = 1.0 / AbulkCV ;  /*4.6.2*/
                    Vdsat = Vgst * dVdsat_dVg ;
                    dVdsat_dVb = -(Vdsat * dAbulkCV_dVb + dVth_dVb)* dVdsat_dVg ;

                    if (BSIM4v7xpart > 0.5)
                    {   /* 0/100 Charge partition model */

                        if (Vdsat <= Vds)
                        {
                            /* saturation region */
                            T1 = Vdsat / 3.0 ;
                            qgate = CoxWL * (Vgs_eff - Vfb - pParam->BSIM4v7phi - T1) ;
                            T2 = -Two_Third_CoxWL * Vgst ;
                            qbulk = -(qgate + T2) ;
                            qdrn = 0.0 ;
                            BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] = One_Third_CoxWL * (3.0 - dVdsat_dVg) * dVgs_eff_dVg ;
                            T2 = -One_Third_CoxWL * dVdsat_dVb ;
                            BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] + T2) ;
                            BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] = 0.0 ;
                            BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] = 0.0 ;
                            BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] = 0.0 ;
                            BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID] = 0.0 ;
                            BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] - Two_Third_CoxWL * dVgs_eff_dVg) ;
                            T3 = -(T2 + Two_Third_CoxWL * dVth_dVb) ;
                            BSIM4v7entry.d_BSIM4v7cbsbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] + T3) ;
                            BSIM4v7entry.d_BSIM4v7cbdbRWArray [instance_ID] = 0.0 ;
                        } else {
                            /* linear region */
                            Alphaz = Vgst / Vdsat ;
                            T1 = 2.0 * Vdsat - Vds ;
                            T2 = Vds / (3.0 * T1) ;
                            T3 = T2 * Vds ;
                            T9 = 0.25 * CoxWL ;
                            T4 = T9 * Alphaz ;
                            T7 = 2.0 * Vds - T1 - 3.0 * T3 ;
                            T8 = T3 - T1 - 2.0 * Vds ;
                            qgate = CoxWL * (Vgs_eff - Vfb - pParam->BSIM4v7phi - 0.5 * (Vds - T3)) ;
                            T10 = T4 * T8 ;
                            qdrn = T4 * T7 ;
                            qbulk = -(qgate + qdrn + T10) ;
                            T5 = T3 / T1 ;
                            BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] = CoxWL * (1.0 - T5 * dVdsat_dVg) * dVgs_eff_dVg ;
                            T11 = -CoxWL * T5 * dVdsat_dVb ;
                            BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] = CoxWL * (T2 - 0.5 + 0.5 * T5) ;
                            BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] + T11 + BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID]) ;
                            T6 = 1.0 / Vdsat ;
                            dAlphaz_dVg = T6 * (1.0 - Alphaz * dVdsat_dVg) ;
                            dAlphaz_dVb = -T6 * (dVth_dVb + Alphaz * dVdsat_dVb) ;
                            T7 = T9 * T7 ;
                            T8 = T9 * T8 ;
                            T9 = 2.0 * T4 * (1.0 - 3.0 * T5) ;
                            BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] = (T7 * dAlphaz_dVg - T9 * dVdsat_dVg) * dVgs_eff_dVg ;
                            T12 = T7 * dAlphaz_dVb - T9 * dVdsat_dVb ;
                            BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] = T4 * (3.0 - 6.0 * T2 - 3.0 * T5) ;
                            BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] + T12 + BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID]) ;
                            T9 = 2.0 * T4 * (1.0 + T5) ;
                            T10 = (T8 * dAlphaz_dVg - T9 * dVdsat_dVg) * dVgs_eff_dVg ;
                            T11 = T8 * dAlphaz_dVb - T9 * dVdsat_dVb ;
                            T12 = T4 * (2.0 * T2 + T5 - 1.0) ;
                            T0 = -(T10 + T11 + T12) ;
                            BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] + T10) ;
                            BSIM4v7entry.d_BSIM4v7cbdbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] + T12) ;
                            BSIM4v7entry.d_BSIM4v7cbsbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID] + T0) ;
                        }
                    }
                    else if (BSIM4v7xpart < 0.5)
                    {   /* 40/60 Charge partition model */

                        if (Vds >= Vdsat)
                        {
                             /* saturation region */
                            T1 = Vdsat / 3.0 ;
                            qgate = CoxWL * (Vgs_eff - Vfb - pParam->BSIM4v7phi - T1) ;
                            T2 = -Two_Third_CoxWL * Vgst ;
                            qbulk = -(qgate + T2) ;
                            qdrn = 0.4 * T2 ;
                            BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] = One_Third_CoxWL * (3.0 - dVdsat_dVg) * dVgs_eff_dVg ;
                            T2 = -One_Third_CoxWL * dVdsat_dVb ;
                            BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] + T2) ;
                            BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] = 0.0 ;
                            T3 = 0.4 * Two_Third_CoxWL ;
                            BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] = -T3 * dVgs_eff_dVg ;
                            BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] = 0.0 ;
                            T4 = T3 * dVth_dVb ;
                            BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID] = -(T4 + BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID]) ;
                            BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] - Two_Third_CoxWL * dVgs_eff_dVg) ;
                            T3 = -(T2 + Two_Third_CoxWL * dVth_dVb) ;
                            BSIM4v7entry.d_BSIM4v7cbsbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] + T3) ;
                            BSIM4v7entry.d_BSIM4v7cbdbRWArray [instance_ID] = 0.0 ;
                        } else {
                            /* linear region  */
                            Alphaz = Vgst / Vdsat ;
                            T1 = 2.0 * Vdsat - Vds ;
                            T2 = Vds / (3.0 * T1) ;
                            T3 = T2 * Vds ;
                            T9 = 0.25 * CoxWL ;
                            T4 = T9 * Alphaz ;
                            qgate = CoxWL * (Vgs_eff - Vfb - pParam->BSIM4v7phi - 0.5 * (Vds - T3)) ;
                            T5 = T3 / T1 ;
                            BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] = CoxWL * (1.0 - T5 * dVdsat_dVg) * dVgs_eff_dVg ;
                            tmp = -CoxWL * T5 * dVdsat_dVb ;
                            BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] = CoxWL * (T2 - 0.5 + 0.5 * T5) ;
                            BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] + tmp) ;
                            T6 = 1.0 / Vdsat ;
                            dAlphaz_dVg = T6 * (1.0 - Alphaz * dVdsat_dVg) ;
                            dAlphaz_dVb = -T6 * (dVth_dVb + Alphaz * dVdsat_dVb) ;
                            T6 = 8.0 * Vdsat * Vdsat - 6.0 * Vdsat * Vds + 1.2 * Vds * Vds ;
                            T8 = T2 / T1 ;
                            T7 = Vds - T1 - T8 * T6 ;
                            qdrn = T4 * T7 ;
                            T7 *= T9 ;
                            tmp = T8 / T1 ;
                            tmp1 = T4 * (2.0 - 4.0 * tmp * T6 + T8 * (16.0 * Vdsat - 6.0 * Vds)) ;
                            BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] = (T7 * dAlphaz_dVg - tmp1 * dVdsat_dVg) * dVgs_eff_dVg ;
                            T10 = T7 * dAlphaz_dVb - tmp1 * dVdsat_dVb ;
                            BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] = T4 * (2.0 - (1.0 / (3.0 * T1 * T1) + 2.0 * tmp) * T6 + T8 *
                                              (6.0 * Vdsat - 2.4 * Vds)) ;
                            BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] + T10 + BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID]) ;
                            T7 = 2.0 * (T1 + T3) ;
                            qbulk = -(qgate - T4 * T7) ;
                            T7 *= T9 ;
                            T0 = 4.0 * T4 * (1.0 - T5) ;
                            T12 = (-T7 * dAlphaz_dVg - T0 * dVdsat_dVg) * dVgs_eff_dVg - BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] ;  /*4.6.2*/
                            T11 = -T7 * dAlphaz_dVb - T10 - T0 * dVdsat_dVb ;
                            T10 = -4.0 * T4 * (T2 - 0.5 + 0.5 * T5) - BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] ;
                            tmp = -(T10 + T11 + T12) ;
                            BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] + T12) ;
                            BSIM4v7entry.d_BSIM4v7cbdbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] + T10) ;
                            BSIM4v7entry.d_BSIM4v7cbsbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID] + tmp) ;
                        }
                    } else {
                        /* 50/50 partitioning */

                        if (Vds >= Vdsat)
                        {
                            /* saturation region */
                            T1 = Vdsat / 3.0 ;
                            qgate = CoxWL * (Vgs_eff - Vfb - pParam->BSIM4v7phi - T1) ;
                            T2 = -Two_Third_CoxWL * Vgst ;
                            qbulk = -(qgate + T2) ;
                            qdrn = 0.5 * T2 ;
                            BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] = One_Third_CoxWL * (3.0 - dVdsat_dVg) * dVgs_eff_dVg ;
                            T2 = -One_Third_CoxWL * dVdsat_dVb ;
                            BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] + T2) ;
                            BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] = 0.0 ;
                            BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] = -One_Third_CoxWL * dVgs_eff_dVg ;
                            BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] = 0.0 ;
                            T4 = One_Third_CoxWL * dVth_dVb ;
                            BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID] = -(T4 + BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID]) ;
                            BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] - Two_Third_CoxWL * dVgs_eff_dVg) ;
                            T3 = -(T2 + Two_Third_CoxWL * dVth_dVb) ;
                            BSIM4v7entry.d_BSIM4v7cbsbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] + T3) ;
                            BSIM4v7entry.d_BSIM4v7cbdbRWArray [instance_ID] = 0.0 ;
                        } else {
                            /* linear region */
                            Alphaz = Vgst / Vdsat ;
                            T1 = 2.0 * Vdsat - Vds ;
                            T2 = Vds / (3.0 * T1) ;
                            T3 = T2 * Vds ;
                            T9 = 0.25 * CoxWL ;
                            T4 = T9 * Alphaz ;
                            qgate = CoxWL * (Vgs_eff - Vfb - pParam->BSIM4v7phi - 0.5 * (Vds - T3)) ;
                            T5 = T3 / T1 ;
                            BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] = CoxWL * (1.0 - T5 * dVdsat_dVg) * dVgs_eff_dVg ;
                            tmp = -CoxWL * T5 * dVdsat_dVb ;
                            BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] = CoxWL * (T2 - 0.5 + 0.5 * T5) ;
                            BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] + tmp) ;
                            T6 = 1.0 / Vdsat ;
                            dAlphaz_dVg = T6 * (1.0 - Alphaz * dVdsat_dVg) ;
                            dAlphaz_dVb = -T6 * (dVth_dVb + Alphaz * dVdsat_dVb) ;
                            T7 = T1 + T3 ;
                            qdrn = -T4 * T7 ;
                            qbulk = -(qgate + qdrn + qdrn) ;
                            T7 *= T9 ;
                            T0 = T4 * (2.0 * T5 - 2.0) ;
                            BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] = (T0 * dVdsat_dVg - T7 * dAlphaz_dVg) * dVgs_eff_dVg ;
                            T12 = T0 * dVdsat_dVb - T7 * dAlphaz_dVb ;
                            BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] = T4 * (1.0 - 2.0 * T2 - T5) ;
                            BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] + T12 + BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID]) ;
                            BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] + 2.0 * BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID]) ;
                            BSIM4v7entry.d_BSIM4v7cbdbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] + 2.0 * BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID]) ;
                            BSIM4v7entry.d_BSIM4v7cbsbRWArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] + 2.0 * BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID]) ;
                        }    /* end of linear region */
                    }    /* end of 50/50 partition */
                }    /* end of inversion */
            }    /* end of capMod=0 */
            else
            {
                if (Vbseff < 0.0)
                {
                    VbseffCV = Vbseff ;
                    dVbseffCV_dVb = 1.0 ;
                } else {
                    VbseffCV = pParam->BSIM4v7phi - Phis ;
                    dVbseffCV_dVb = -dPhis_dVb ;
                }
                CoxWL = BSIM4v7coxe * pParam->BSIM4v7weffCV * pParam->BSIM4v7leffCV * BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;

                if (BSIM4v7cvchargeMod == 0)
                {
                    /* Seperate VgsteffCV with noff and voffcv */
                    noff = n * pParam->BSIM4v7noff ;
                    dnoff_dVd = pParam->BSIM4v7noff * dn_dVd ;
                    dnoff_dVb = pParam->BSIM4v7noff * dn_dVb ;
                    T0 = Vtm * noff ;
                    voffcv = pParam->BSIM4v7voffcv ;
                    VgstNVt = (Vgst - voffcv) / T0 ;

                    /* 60 - DIVERGENT */
                    if (VgstNVt > EXP_THRESHOLD)
                    {
                        Vgsteff = Vgst - voffcv ;
                        dVgsteff_dVg = dVgs_eff_dVg ;
                        dVgsteff_dVd = -dVth_dVd ;
                        dVgsteff_dVb = -dVth_dVb ;
                    }
                    else if (VgstNVt < -EXP_THRESHOLD)
                    {
                        Vgsteff = T0 * log (1.0 + MIN_EXP) ;
                        dVgsteff_dVg = 0.0 ;
                        dVgsteff_dVd = Vgsteff / noff ;
                        dVgsteff_dVb = dVgsteff_dVd * dnoff_dVb ;
                        dVgsteff_dVd *= dnoff_dVd ;
                    } else {
                        ExpVgst = exp (VgstNVt) ;
                        Vgsteff = T0 * log (1.0 + ExpVgst) ;
                        dVgsteff_dVg = ExpVgst / (1.0 + ExpVgst) ;
                        dVgsteff_dVd = -dVgsteff_dVg * (dVth_dVd + (Vgst - voffcv) / noff * dnoff_dVd) +
                                       Vgsteff / noff * dnoff_dVd ;
                        dVgsteff_dVb = -dVgsteff_dVg * (dVth_dVb + (Vgst - voffcv) / noff * dnoff_dVb) +
                                       Vgsteff / noff * dnoff_dVb ;
                        dVgsteff_dVg *= dVgs_eff_dVg ;
                    }    /* End of VgsteffCV for cvchargeMod = 0 */
                } else {
                    T0 = n * Vtm ;
                    T1 = pParam->BSIM4v7mstarcv * Vgst ;
                    T2 = T1 / T0 ;

                    /* 61 - DIVERGENT */
                    if (T2 > EXP_THRESHOLD)
                    {
                        T10 = T1 ;
                        dT10_dVg = pParam->BSIM4v7mstarcv * dVgs_eff_dVg ;
                        dT10_dVd = -dVth_dVd * pParam->BSIM4v7mstarcv ;
                        dT10_dVb = -dVth_dVb * pParam->BSIM4v7mstarcv ;
                    }
                    else if (T2 < -EXP_THRESHOLD)
                    {
                        T10 = Vtm * log (1.0 + MIN_EXP) ;
                        dT10_dVg = 0.0 ;
                        dT10_dVd = T10 * dn_dVd ;
                        dT10_dVb = T10 * dn_dVb ;
                        T10 *= n ;
                    } else {
                        ExpVgst = exp (T2) ;
                        T3 = Vtm * log (1.0 + ExpVgst) ;
                        T10 = n * T3 ;
                        dT10_dVg = pParam->BSIM4v7mstarcv * ExpVgst / (1.0 + ExpVgst) ;
                        dT10_dVb = T3 * dn_dVb - dT10_dVg * (dVth_dVb + Vgst * dn_dVb / n) ;
                        dT10_dVd = T3 * dn_dVd - dT10_dVg * (dVth_dVd + Vgst * dn_dVd / n) ;
                        dT10_dVg *= dVgs_eff_dVg ;
                    }
                    T1 = pParam->BSIM4v7voffcbncv - (1.0 - pParam->BSIM4v7mstarcv) * Vgst ;
                    T2 = T1 / T0 ;

                    /* 62 - DIVERGENT */
                    if (T2 < -EXP_THRESHOLD)
                    {
                        T3 = BSIM4v7coxe * MIN_EXP / pParam->BSIM4v7cdep0 ;
                        T9 = pParam->BSIM4v7mstarcv + T3 * n ;
                        dT9_dVg = 0.0 ;
                        dT9_dVd = dn_dVd * T3 ;
                        dT9_dVb = dn_dVb * T3 ;
                    }
                    else if (T2 > EXP_THRESHOLD)
                    {
                        T3 = BSIM4v7coxe * MAX_EXP / pParam->BSIM4v7cdep0 ;
                        T9 = pParam->BSIM4v7mstarcv + T3 * n ;
                        dT9_dVg = 0.0 ;
                        dT9_dVd = dn_dVd * T3 ;
                        dT9_dVb = dn_dVb * T3 ;
                    } else {
                        ExpVgst = exp (T2) ;
                        T3 = BSIM4v7coxe / pParam->BSIM4v7cdep0 ;
                        T4 = T3 * ExpVgst ;
                        T5 = T1 * T4 / T0 ;
                        T9 = pParam->BSIM4v7mstarcv + n * T4 ;
                        dT9_dVg = T3 * (pParam->BSIM4v7mstarcv - 1.0) * ExpVgst / Vtm ;
                        dT9_dVb = T4 * dn_dVb - dT9_dVg * dVth_dVb - T5 * dn_dVb ;
                        dT9_dVd = T4 * dn_dVd - dT9_dVg * dVth_dVd - T5 * dn_dVd ;
                        dT9_dVg *= dVgs_eff_dVg ;
                    }
                    Vgsteff = T10 / T9 ;
                    T11 = T9 * T9 ;
                    dVgsteff_dVg = (T9 * dT10_dVg - T10 * dT9_dVg) / T11 ;
                    dVgsteff_dVd = (T9 * dT10_dVd - T10 * dT9_dVd) / T11 ;
                    dVgsteff_dVb = (T9 * dT10_dVb - T10 * dT9_dVb) / T11 ;
                    /* End of VgsteffCV for cvchargeMod = 1 */
                }

                if (BSIM4v7capMod == 1)
                {
                    Vfb = BSIM4v7entry.d_BSIM4v7vfbzbArray [instance_ID] ;
                    V3 = Vfb - Vgs_eff + VbseffCV - DELTA_3 ;

                    if (Vfb <= 0.0)
                        T0 = sqrt (V3 * V3 - 4.0 * DELTA_3 * Vfb) ;
                    else
                        T0 = sqrt (V3 * V3 + 4.0 * DELTA_3 * Vfb) ;

                    T1 = 0.5 * (1.0 + V3 / T0) ;
                    Vfbeff = Vfb - 0.5 * (V3 + T0) ;
                    dVfbeff_dVg = T1 * dVgs_eff_dVg ;
                    dVfbeff_dVb = -T1 * dVbseffCV_dVb ;
                    Qac0 = CoxWL * (Vfbeff - Vfb) ;
                    dQac0_dVg = CoxWL * dVfbeff_dVg ;
                    dQac0_dVb = CoxWL * dVfbeff_dVb ;
                    T0 = 0.5 * pParam->BSIM4v7k1ox ;
                    T3 = Vgs_eff - Vfbeff - VbseffCV - Vgsteff ;

                    /* 63 - DIVERGENT */
                    if (pParam->BSIM4v7k1ox == 0.0)
                    {
                        T1 = 0.0 ;
                        T2 = 0.0 ;
                    }
                    else if (T3 < 0.0)
                    {
                        T1 = T0 + T3 / pParam->BSIM4v7k1ox ;
                        T2 = CoxWL ;
                    } else {
                        T1 = sqrt (T0 * T0 + T3) ;
                        T2 = CoxWL * T0 / T1 ;
                    }
                    Qsub0 = CoxWL * pParam->BSIM4v7k1ox * (T1 - T0) ;
                    dQsub0_dVg = T2 * (dVgs_eff_dVg - dVfbeff_dVg - dVgsteff_dVg) ;
                    dQsub0_dVd = -T2 * dVgsteff_dVd ;
                    dQsub0_dVb = -T2 * (dVfbeff_dVb + dVbseffCV_dVb + dVgsteff_dVb) ;
                    AbulkCV = Abulk0 * pParam->BSIM4v7abulkCVfactor ;
                    dAbulkCV_dVb = pParam->BSIM4v7abulkCVfactor * dAbulk0_dVb ;
                    VdsatCV = Vgsteff / AbulkCV ;
                    T0 = VdsatCV - Vds - DELTA_4 ;
                    dT0_dVg = 1.0 / AbulkCV ;
                    dT0_dVb = -VdsatCV * dAbulkCV_dVb / AbulkCV ;
                    T1 = sqrt(T0 * T0 + 4.0 * DELTA_4 * VdsatCV) ;
                    dT1_dVg = (T0 + DELTA_4 + DELTA_4) / T1 ;
                    dT1_dVd = -T0 / T1 ;
                    dT1_dVb = dT1_dVg * dT0_dVb ;
                    dT1_dVg *= dT0_dVg ;

                    /* 64 - DIVERGENT */
                    if (T0 >= 0.0)
                    {
                        VdseffCV = VdsatCV - 0.5 * (T0 + T1) ;
                        dVdseffCV_dVg = 0.5 * (dT0_dVg - dT1_dVg) ;
                        dVdseffCV_dVd = 0.5 * (1.0 - dT1_dVd) ;
                        dVdseffCV_dVb = 0.5 * (dT0_dVb - dT1_dVb) ;
                    } else {
                        T3 = (DELTA_4 + DELTA_4) / (T1 - T0) ;
                        T4 = 1.0 - T3 ;
                        T5 = VdsatCV * T3 / (T1 - T0) ;
                        VdseffCV = VdsatCV * T4 ;
                        dVdseffCV_dVg = dT0_dVg * T4 + T5 * (dT1_dVg - dT0_dVg) ;
                        dVdseffCV_dVd = T5 * (dT1_dVd + 1.0) ;
                        dVdseffCV_dVb = dT0_dVb * (T4 - T5) + T5 * dT1_dVb ;
                    }
                    if (Vds == 0.0)
                    {
                       VdseffCV = 0.0 ;
                       dVdseffCV_dVg = 0.0 ;
                       dVdseffCV_dVb = 0.0 ;
                    }
                    T0 = AbulkCV * VdseffCV ;
                    T1 = 12.0 * (Vgsteff - 0.5 * T0 + 1.0e-20) ;
                    T2 = VdseffCV / T1 ;
                    T3 = T0 * T2 ;
                    T4 = (1.0 - 12.0 * T2 * T2 * AbulkCV) ;
                    T5 = (6.0 * T0 * (4.0 * Vgsteff - T0) / (T1 * T1) - 0.5) ;
                    T6 = 12.0 * T2 * T2 * Vgsteff ;
                    qgate = CoxWL * (Vgsteff - 0.5 * VdseffCV + T3) ;
                    Cgg1 = CoxWL * (T4 + T5 * dVdseffCV_dVg) ;
                    Cgd1 = CoxWL * T5 * dVdseffCV_dVd + Cgg1 * dVgsteff_dVd ;
                    Cgb1 = CoxWL * (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb) + Cgg1 * dVgsteff_dVb ;
                    Cgg1 *= dVgsteff_dVg ;
                    T7 = 1.0 - AbulkCV ;
                    qbulk = CoxWL * T7 * (0.5 * VdseffCV - T3) ;
                    T4 = -T7 * (T4 - 1.0) ;
                    T5 = -T7 * T5 ;
                    T6 = -(T7 * T6 + (0.5 * VdseffCV - T3)) ;
                    Cbg1 = CoxWL * (T4 + T5 * dVdseffCV_dVg) ;
                    Cbd1 = CoxWL * T5 * dVdseffCV_dVd + Cbg1 * dVgsteff_dVd ;
                    Cbb1 = CoxWL * (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb) + Cbg1 * dVgsteff_dVb ;
                    Cbg1 *= dVgsteff_dVg ;

                    if (BSIM4v7xpart > 0.5)
                    {
                        /* 0/100 Charge petition model */
                        T1 = T1 + T1 ;
                        qsrc = -CoxWL * (0.5 * Vgsteff + 0.25 * T0 - T0 * T0 / T1) ;
                        T7 = (4.0 * Vgsteff - T0) / (T1 * T1) ;
                        T4 = -(0.5 + 24.0 * T0 * T0 / (T1 * T1)) ;
                        T5 = -(0.25 * AbulkCV - 12.0 * AbulkCV * T0 * T7) ;
                        T6 = -(0.25 * VdseffCV - 12.0 * T0 * VdseffCV * T7) ;
                        Csg = CoxWL * (T4 + T5 * dVdseffCV_dVg) ;
                        Csd = CoxWL * T5 * dVdseffCV_dVd + Csg * dVgsteff_dVd ;
                        Csb = CoxWL * (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb) + Csg * dVgsteff_dVb ;
                        Csg *= dVgsteff_dVg ;
                    }
                    else if (BSIM4v7xpart < 0.5)
                    {
                        /* 40/60 Charge petition model */
                        T1 = T1 / 12.0 ;
                        T2 = 0.5 * CoxWL / (T1 * T1) ;
                        T3 = Vgsteff * (2.0 * T0 * T0 / 3.0 + Vgsteff * (Vgsteff - 4.0 * T0 / 3.0)) -
                             2.0 * T0 * T0 * T0 / 15.0 ;
                        qsrc = -T2 * T3 ;
                        T7 = 4.0 / 3.0 * Vgsteff * (Vgsteff - T0) + 0.4 * T0 * T0 ;
                        T4 = -2.0 * qsrc / T1 - T2 * (Vgsteff * (3.0 * Vgsteff - 8.0 * T0 / 3.0) +
                             2.0 * T0 * T0 / 3.0) ;
                        T5 = (qsrc / T1 + T2 * T7) * AbulkCV ;
                        T6 = (qsrc / T1 * VdseffCV + T2 * T7 * VdseffCV) ;
                        Csg = (T4 + T5 * dVdseffCV_dVg) ;
                        Csd = T5 * dVdseffCV_dVd + Csg * dVgsteff_dVd ;
                        Csb = (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb) + Csg * dVgsteff_dVb ;
                        Csg *= dVgsteff_dVg ;
                    } else {
                        /* 50/50 Charge petition model */
                        qsrc = -0.5 * (qgate + qbulk) ;
                        Csg = -0.5 * (Cgg1 + Cbg1) ;
                        Csb = -0.5 * (Cgb1 + Cbb1) ;
                        Csd = -0.5 * (Cgd1 + Cbd1) ;
                    }
                    qgate += Qac0 + Qsub0 ;
                    qbulk -= (Qac0 + Qsub0) ;
                    qdrn = -(qgate + qbulk + qsrc) ;
                    Cgg = dQac0_dVg + dQsub0_dVg + Cgg1 ;
                    Cgd = dQsub0_dVd + Cgd1 ;
                    Cgb = dQac0_dVb + dQsub0_dVb + Cgb1 ;
                    Cbg = Cbg1 - dQac0_dVg - dQsub0_dVg ;
                    Cbd = Cbd1 - dQsub0_dVd ;
                    Cbb = Cbb1 - dQac0_dVb - dQsub0_dVb ;
                    Cgb *= dVbseff_dVb ;
                    Cbb *= dVbseff_dVb ;
                    Csb *= dVbseff_dVb ;
                    BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] = Cgg ;
                    BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] = -(Cgg + Cgd + Cgb) ;
                    BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] = Cgd ;
                    BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] = -(Cgg + Cbg + Csg) ;
                    BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID] = (Cgg + Cgd + Cgb + Cbg + Cbd + Cbb + Csg + Csd + Csb) ;
                    BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] = -(Cgd + Cbd + Csd) ;
                    BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] = Cbg ;
                    BSIM4v7entry.d_BSIM4v7cbsbRWArray [instance_ID] = -(Cbg + Cbd + Cbb) ;
                    BSIM4v7entry.d_BSIM4v7cbdbRWArray [instance_ID] = Cbd ;
                }


                /* Charge-Thickness capMod (CTM) begins */
                else if (BSIM4v7capMod == 2)
                {
                    V3 = BSIM4v7entry.d_BSIM4v7vfbzbArray [instance_ID] - Vgs_eff + VbseffCV - DELTA_3 ;

                    if (BSIM4v7entry.d_BSIM4v7vfbzbArray [instance_ID] <= 0.0)
                        T0 = sqrt (V3 * V3 - 4.0 * DELTA_3 * BSIM4v7entry.d_BSIM4v7vfbzbArray [instance_ID]) ;
                    else
                        T0 = sqrt (V3 * V3 + 4.0 * DELTA_3 * BSIM4v7entry.d_BSIM4v7vfbzbArray [instance_ID]) ;

                    T1 = 0.5 * (1.0 + V3 / T0) ;
                    Vfbeff = BSIM4v7entry.d_BSIM4v7vfbzbArray [instance_ID] - 0.5 * (V3 + T0) ;
                    dVfbeff_dVg = T1 * dVgs_eff_dVg ;
                    dVfbeff_dVb = -T1 * dVbseffCV_dVb ;
                    Cox = BSIM4v7coxp ;
                    Tox = 1.0e8 * BSIM4v7toxp ;
                    T0 = (Vgs_eff - VbseffCV - BSIM4v7entry.d_BSIM4v7vfbzbArray [instance_ID]) / Tox ;
                    dT0_dVg = dVgs_eff_dVg / Tox ;
                    dT0_dVb = -dVbseffCV_dVb / Tox ;
                    tmp = T0 * pParam->BSIM4v7acde ;

                    /* 65 - DIVERGENT */
                    if ((-EXP_THRESHOLD < tmp) && (tmp < EXP_THRESHOLD))
                    {
                        Tcen = pParam->BSIM4v7ldeb * exp (tmp) ;
                        dTcen_dVg = pParam->BSIM4v7acde * Tcen ;
                        dTcen_dVb = dTcen_dVg * dT0_dVb ;
                        dTcen_dVg *= dT0_dVg ;
                    }
                    else if (tmp <= -EXP_THRESHOLD)
                    {
                        Tcen = pParam->BSIM4v7ldeb * MIN_EXP ;
                        dTcen_dVg = dTcen_dVb = 0.0 ;
                    } else {
                        Tcen = pParam->BSIM4v7ldeb * MAX_EXP ;
                        dTcen_dVg = dTcen_dVb = 0.0 ;
                    }
                    LINK = 1.0e-3 * BSIM4v7toxp ;
                    V3 = pParam->BSIM4v7ldeb - Tcen - LINK ;
                    V4 = sqrt (V3 * V3 + 4.0 * LINK * pParam->BSIM4v7ldeb) ;
                    Tcen = pParam->BSIM4v7ldeb - 0.5 * (V3 + V4) ;
                    T1 = 0.5 * (1.0 + V3 / V4) ;
                    dTcen_dVg *= T1 ;
                    dTcen_dVb *= T1 ;
                    Ccen = epssub / Tcen ;
                    T2 = Cox / (Cox + Ccen) ;
                    Coxeff = T2 * Ccen ;
                    T3 = -Ccen / Tcen ;
                    dCoxeff_dVg = T2 * T2 * T3 ;
                    dCoxeff_dVb = dCoxeff_dVg * dTcen_dVb ;
                    dCoxeff_dVg *= dTcen_dVg ;
                    CoxWLcen = CoxWL * Coxeff / BSIM4v7coxe ;
                    Qac0 = CoxWLcen * (Vfbeff - BSIM4v7entry.d_BSIM4v7vfbzbArray [instance_ID]) ;
                    QovCox = Qac0 / Coxeff ;
                    dQac0_dVg = CoxWLcen * dVfbeff_dVg + QovCox * dCoxeff_dVg ;
                    dQac0_dVb = CoxWLcen * dVfbeff_dVb + QovCox * dCoxeff_dVb ;
                    T0 = 0.5 * pParam->BSIM4v7k1ox ;
                    T3 = Vgs_eff - Vfbeff - VbseffCV - Vgsteff ;

                    /* 66 - DIVERGENT */
                    if (pParam->BSIM4v7k1ox == 0.0)
                    {
                        T1 = 0.0 ;
                        T2 = 0.0 ;
                    }
                    else if (T3 < 0.0)
                    {
                        T1 = T0 + T3 / pParam->BSIM4v7k1ox ;
                        T2 = CoxWLcen ;
                    } else {
                        T1 = sqrt (T0 * T0 + T3) ;
                        T2 = CoxWLcen * T0 / T1 ;
                    }
                    Qsub0 = CoxWLcen * pParam->BSIM4v7k1ox * (T1 - T0) ;
                    QovCox = Qsub0 / Coxeff ;
                    dQsub0_dVg = T2 * (dVgs_eff_dVg - dVfbeff_dVg - dVgsteff_dVg) + QovCox * dCoxeff_dVg ;
                    dQsub0_dVd = -T2 * dVgsteff_dVd ;
                    dQsub0_dVb = -T2 * (dVfbeff_dVb + dVbseffCV_dVb + dVgsteff_dVb) + QovCox * dCoxeff_dVb ;


                    /* Gate-bias dependent delta Phis begins */
                    /* 67 - DIVERGENT */
                    if (pParam->BSIM4v7k1ox <= 0.0)
                    {
                        Denomi = 0.25 * pParam->BSIM4v7moin * Vtm ;
                        T0 = 0.5 * pParam->BSIM4v7sqrtPhi ;
                    } else {
                        Denomi = pParam->BSIM4v7moin * Vtm * pParam->BSIM4v7k1ox * pParam->BSIM4v7k1ox ;
                        T0 = pParam->BSIM4v7k1ox * pParam->BSIM4v7sqrtPhi ;
                    }
                    T1 = 2.0 * T0 + Vgsteff ;
                    DeltaPhi = Vtm * log (1.0 + T1 * Vgsteff / Denomi) ;
                    dDeltaPhi_dVg = 2.0 * Vtm * (T1 -T0) / (Denomi + T1 * Vgsteff) ;
                    /* End of delta Phis */


                    /* VgDP = Vgsteff - DeltaPhi */
                    T0 = Vgsteff - DeltaPhi - 0.001 ;
                    dT0_dVg = 1.0 - dDeltaPhi_dVg ;
                    T1 = sqrt (T0 * T0 + Vgsteff * 0.004) ;
                    VgDP = 0.5 * (T0 + T1) ;
                    dVgDP_dVg = 0.5 * (dT0_dVg + (T0 * dT0_dVg + 0.002) / T1) ;
                    Tox += Tox ;    /* WDLiu: Tcen reevaluated below due to different Vgsteff */
                    T0 = (Vgsteff + BSIM4v7entry.d_BSIM4v7vtfbphi2Array [instance_ID]) / Tox ;
                    tmp = exp (BSIM4v7bdos * 0.7 * log (T0)) ;
                    T1 = 1.0 + tmp ;
                    T2 = BSIM4v7bdos * 0.7 * tmp / (T0 * Tox) ;
                    Tcen = BSIM4v7ados * 1.9e-9 / T1 ;
                    dTcen_dVg = -Tcen * T2 / T1 ;
                    dTcen_dVd = dTcen_dVg * dVgsteff_dVd ;
                    dTcen_dVb = dTcen_dVg * dVgsteff_dVb ;
                    dTcen_dVg *= dVgsteff_dVg ;
                    Ccen = epssub / Tcen ;
                    T0 = Cox / (Cox + Ccen) ;
                    Coxeff = T0 * Ccen ;
                    T1 = -Ccen / Tcen ;
                    dCoxeff_dVg = T0 * T0 * T1 ;
                    dCoxeff_dVd = dCoxeff_dVg * dTcen_dVd ;
                    dCoxeff_dVb = dCoxeff_dVg * dTcen_dVb ;
                    dCoxeff_dVg *= dTcen_dVg ;
                    CoxWLcen = CoxWL * Coxeff / BSIM4v7coxe ;
                    AbulkCV = Abulk0 * pParam->BSIM4v7abulkCVfactor ;
                    dAbulkCV_dVb = pParam->BSIM4v7abulkCVfactor * dAbulk0_dVb ;
                    VdsatCV = VgDP / AbulkCV ;
                    T0 = VdsatCV - Vds - DELTA_4 ;
                    dT0_dVg = dVgDP_dVg / AbulkCV ;
                    dT0_dVb = -VdsatCV * dAbulkCV_dVb / AbulkCV ;
                    T1 = sqrt (T0 * T0 + 4.0 * DELTA_4 * VdsatCV) ;
                    dT1_dVg = (T0 + DELTA_4 + DELTA_4) / T1 ;
                    dT1_dVd = -T0 / T1 ;
                    dT1_dVb = dT1_dVg * dT0_dVb ;
                    dT1_dVg *= dT0_dVg ;

                    /* 68 - DIVERGENT */
                    if (T0 >= 0.0)
                    {
                        VdseffCV = VdsatCV - 0.5 * (T0 + T1) ;
                        dVdseffCV_dVg = 0.5 * (dT0_dVg - dT1_dVg) ;
                        dVdseffCV_dVd = 0.5 * (1.0 - dT1_dVd) ;
                        dVdseffCV_dVb = 0.5 * (dT0_dVb - dT1_dVb) ;
                    } else {
                        T3 = (DELTA_4 + DELTA_4) / (T1 - T0) ;
                        T4 = 1.0 - T3 ;
                        T5 = VdsatCV * T3 / (T1 - T0) ;
                        VdseffCV = VdsatCV * T4 ;
                        dVdseffCV_dVg = dT0_dVg * T4 + T5 * (dT1_dVg - dT0_dVg) ;
                        dVdseffCV_dVd = T5 * (dT1_dVd + 1.0) ;
                        dVdseffCV_dVb = dT0_dVb * (T4 - T5) + T5 * dT1_dVb ;
                    }

                    if (Vds == 0.0)
                    {
                        VdseffCV = 0.0 ;
                        dVdseffCV_dVg = 0.0 ;
                        dVdseffCV_dVb = 0.0 ;
                    }
                    T0 = AbulkCV * VdseffCV ;
                    T1 = VgDP ;
                    T2 = 12.0 * (T1 - 0.5 * T0 + 1.0e-20) ;
                    T3 = T0 / T2 ;
                    T4 = 1.0 - 12.0 * T3 * T3 ;
                    T5 = AbulkCV * (6.0 * T0 * (4.0 * T1 - T0) / (T2 * T2) - 0.5) ;
                    T6 = T5 * VdseffCV / AbulkCV ;
                    qgate = CoxWLcen * (T1 - T0 * (0.5 - T3)) ;
                    QovCox = qgate / Coxeff ;
                    Cgg1 = CoxWLcen * (T4 * dVgDP_dVg + T5 * dVdseffCV_dVg) ;
                    Cgd1 = CoxWLcen * T5 * dVdseffCV_dVd + Cgg1 * dVgsteff_dVd + QovCox * dCoxeff_dVd ;
                    Cgb1 = CoxWLcen * (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb) + Cgg1 * dVgsteff_dVb +
                           QovCox * dCoxeff_dVb ;
                    Cgg1 = Cgg1 * dVgsteff_dVg + QovCox * dCoxeff_dVg ;
                    T7 = 1.0 - AbulkCV ;
                    T8 = T2 * T2 ;
                    T9 = 12.0 * T7 * T0 * T0 / (T8 * AbulkCV) ;
                    T10 = T9 * dVgDP_dVg ;
                    T11 = -T7 * T5 / AbulkCV ;
                    T12 = -(T9 * T1 / AbulkCV + VdseffCV * (0.5 - T0 / T2)) ;
                    qbulk = CoxWLcen * T7 * (0.5 * VdseffCV - T0 * VdseffCV / T2) ;
                    QovCox = qbulk / Coxeff ;
                    Cbg1 = CoxWLcen * (T10 + T11 * dVdseffCV_dVg) ;
                    Cbd1 = CoxWLcen * T11 * dVdseffCV_dVd + Cbg1 * dVgsteff_dVd + QovCox * dCoxeff_dVd ;
                    Cbb1 = CoxWLcen * (T11 * dVdseffCV_dVb + T12 * dAbulkCV_dVb) + Cbg1 * dVgsteff_dVb +
                           QovCox * dCoxeff_dVb ;
                    Cbg1 = Cbg1 * dVgsteff_dVg + QovCox * dCoxeff_dVg ;

                    if (BSIM4v7xpart > 0.5)
                    {
                        /* 0/100 partition */
                        qsrc = -CoxWLcen * (T1 / 2.0 + T0 / 4.0 - 0.5 * T0 * T0 / T2) ;
                        QovCox = qsrc / Coxeff ;
                        T2 += T2 ;
                        T3 = T2 * T2 ;
                        T7 = -(0.25 - 12.0 * T0 * (4.0 * T1 - T0) / T3) ;
                        T4 = -(0.5 + 24.0 * T0 * T0 / T3) * dVgDP_dVg ;
                        T5 = T7 * AbulkCV ;
                        T6 = T7 * VdseffCV ;
                        Csg = CoxWLcen * (T4 + T5 * dVdseffCV_dVg) ;
                        Csd = CoxWLcen * T5 * dVdseffCV_dVd + Csg * dVgsteff_dVd + QovCox * dCoxeff_dVd ;
                        Csb = CoxWLcen * (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb) + Csg * dVgsteff_dVb +
                              QovCox * dCoxeff_dVb ;
                        Csg = Csg * dVgsteff_dVg + QovCox * dCoxeff_dVg ;
                    }
                    else if (BSIM4v7xpart < 0.5)
                    {
                        /* 40/60 partition */
                        T2 = T2 / 12.0 ;
                        T3 = 0.5 * CoxWLcen / (T2 * T2) ;
                        T4 = T1 * (2.0 * T0 * T0 / 3.0 + T1 * (T1 - 4.0 * T0 / 3.0)) - 2.0 * T0 * T0 * T0 / 15.0 ;
                        qsrc = -T3 * T4 ;
                        QovCox = qsrc / Coxeff ;
                        T8 = 4.0 / 3.0 * T1 * (T1 - T0) + 0.4 * T0 * T0 ;
                        T5 = -2.0 * qsrc / T2 - T3 * (T1 * (3.0 * T1 - 8.0 * T0 / 3.0) + 2.0 * T0 * T0 / 3.0) ;
                        T6 = AbulkCV * (qsrc / T2 + T3 * T8) ;
                        T7 = T6 * VdseffCV / AbulkCV ;
                        Csg = T5 * dVgDP_dVg + T6 * dVdseffCV_dVg ;
                        Csd = Csg * dVgsteff_dVd + T6 * dVdseffCV_dVd + QovCox * dCoxeff_dVd ;
                        Csb = Csg * dVgsteff_dVb + T6 * dVdseffCV_dVb + T7 * dAbulkCV_dVb + QovCox * dCoxeff_dVb ;
                        Csg = Csg * dVgsteff_dVg + QovCox * dCoxeff_dVg ;
                    } else {
                        /* 50/50 partition */
                        qsrc = -0.5 * qgate ;
                        Csg = -0.5 * Cgg1 ;
                        Csd = -0.5 * Cgd1 ;
                        Csb = -0.5 * Cgb1 ;
                    }
                    qgate += Qac0 + Qsub0 - qbulk ;
                    qbulk -= (Qac0 + Qsub0) ;
                    qdrn = -(qgate + qbulk + qsrc) ;
                    Cbg = Cbg1 - dQac0_dVg - dQsub0_dVg ;
                    Cbd = Cbd1 - dQsub0_dVd ;
                    Cbb = Cbb1 - dQac0_dVb - dQsub0_dVb ;
                    Cgg = Cgg1 - Cbg ;
                    Cgd = Cgd1 - Cbd ;
                    Cgb = Cgb1 - Cbb ;
                    Cgb *= dVbseff_dVb ;
                    Cbb *= dVbseff_dVb ;
                    Csb *= dVbseff_dVb ;
                    BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] = Cgg ;
                    BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] = -(Cgg + Cgd + Cgb) ;
                    BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] = Cgd ;
                    BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] = -(Cgg + Cbg + Csg) ;
                    BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID] = (Cgg + Cgd + Cgb + Cbg + Cbd + Cbb + Csg + Csd + Csb) ;
                    BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] = -(Cgd + Cbd + Csd) ;
                    BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] = Cbg ;
                    BSIM4v7entry.d_BSIM4v7cbsbRWArray [instance_ID] = -(Cbg + Cbd + Cbb) ;
                    BSIM4v7entry.d_BSIM4v7cbdbRWArray [instance_ID] = Cbd ;
                }    /* End of CTM */
            }

            BSIM4v7entry.d_BSIM4v7csgbRWArray [instance_ID] = - BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] ;
            BSIM4v7entry.d_BSIM4v7csdbRWArray [instance_ID] = - BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7cbdbRWArray [instance_ID] ;
            BSIM4v7entry.d_BSIM4v7cssbRWArray [instance_ID] = - BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7cbsbRWArray [instance_ID] ;
            BSIM4v7entry.d_BSIM4v7cgbbRWArray [instance_ID] = - BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] ;
            BSIM4v7entry.d_BSIM4v7cdbbRWArray [instance_ID] = - BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID] ;
            BSIM4v7entry.d_BSIM4v7cbbbRWArray [instance_ID] = - BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7cbdbRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7cbsbRWArray [instance_ID] ;
            BSIM4v7entry.d_BSIM4v7csbbRWArray [instance_ID] = - BSIM4v7entry.d_BSIM4v7cgbbRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7cdbbRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7cbbbRWArray [instance_ID] ;
            BSIM4v7entry.d_BSIM4v7qgateRWArray [instance_ID] = qgate ;
            BSIM4v7entry.d_BSIM4v7qbulkRWArray [instance_ID] = qbulk ;
            BSIM4v7entry.d_BSIM4v7qdrnRWArray [instance_ID] = qdrn ;
            BSIM4v7entry.d_BSIM4v7qsrcRWArray [instance_ID] = -(qgate + qbulk + qdrn) ;


            /* NQS begins */
            /* 69 - DIVERGENT */
            if ((BSIM4v7entry.d_BSIM4v7trnqsModArray [instance_ID]) || (BSIM4v7entry.d_BSIM4v7acnqsModArray [instance_ID]))
            {
                BSIM4v7entry.d_BSIM4v7qchqsArray [instance_ID] = qcheq = -(qbulk + qgate) ;
                BSIM4v7entry.d_BSIM4v7cqgbArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID]) ;
                BSIM4v7entry.d_BSIM4v7cqdbArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cbdbRWArray [instance_ID]) ;
                BSIM4v7entry.d_BSIM4v7cqsbArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cbsbRWArray [instance_ID]) ;
                BSIM4v7entry.d_BSIM4v7cqbbArray [instance_ID] = -(BSIM4v7entry.d_BSIM4v7cqgbArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cqdbArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cqsbArray [instance_ID]) ;
                CoxWL = BSIM4v7coxe * pParam->BSIM4v7weffCV * BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] * pParam->BSIM4v7leffCV ;
                T1 = BSIM4v7entry.d_BSIM4v7gcrgRWArray [instance_ID] / CoxWL ;    /* 1 / tau */
                BSIM4v7entry.d_BSIM4v7gtauRWArray [instance_ID] = T1 * ScalingFactor ;

                if (BSIM4v7entry.d_BSIM4v7acnqsModArray [instance_ID])
                    BSIM4v7entry.d_BSIM4v7taunetArray [instance_ID] = 1.0 / T1 ;

                CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 23] = qcheq ;
                if (CKTmode & MODEINITTRAN)
                    CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 23] = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 23] ;

                if (BSIM4v7entry.d_BSIM4v7trnqsModArray [instance_ID])
                {
                    error = cuNIintegrate_device_kernel (CKTstate_0, CKTstate_1, &geq, &ceq, 0.0,
                                                          BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 23, CKTag_0, CKTag_1, CKTorder) ;
                    if (error)
                        printf ("Integration error\n\n") ;
                        //return(error) ;
                }
            }


finished:

            /* Calculate junction C-V */
            if (ChargeComputationNeeded)
            {
                /* bug fix */
                czbd = BSIM4v7DunitAreaTempJctCap * BSIM4v7entry.d_BSIM4v7AdeffArray [instance_ID] ;
                /* ------- */

                czbs = BSIM4v7SunitAreaTempJctCap * BSIM4v7entry.d_BSIM4v7AseffArray [instance_ID] ;
                czbdsw = BSIM4v7DunitLengthSidewallTempJctCap * BSIM4v7entry.d_BSIM4v7PdeffArray [instance_ID] ;
                czbdswg = BSIM4v7DunitLengthGateSidewallTempJctCap * pParam->BSIM4v7weffCJ * BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                czbssw = BSIM4v7SunitLengthSidewallTempJctCap * BSIM4v7entry.d_BSIM4v7PseffArray [instance_ID] ;
                czbsswg = BSIM4v7SunitLengthGateSidewallTempJctCap * pParam->BSIM4v7weffCJ * BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                MJS = BSIM4v7SbulkJctBotGradingCoeff ;
                MJSWS = BSIM4v7SbulkJctSideGradingCoeff ;
                MJSWGS = BSIM4v7SbulkJctGateSideGradingCoeff ;
                MJD = BSIM4v7DbulkJctBotGradingCoeff ;
                MJSWD = BSIM4v7DbulkJctSideGradingCoeff ;
                MJSWGD = BSIM4v7DbulkJctGateSideGradingCoeff ;


                /* Source Bulk Junction */
                /* 70 - DIVERGENT */
                if (vbs_jct == 0.0)
                {
                    CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 19] = 0.0 ;
                    BSIM4v7entry.d_BSIM4v7capbsRWArray [instance_ID] = czbs + czbssw + czbsswg ;
                }
                else if (vbs_jct < 0.0)
                {
                    if (czbs > 0.0)
                    {
                        arg = 1.0 - vbs_jct / BSIM4v7PhiBS ;

                        if (MJS == 0.5)
                            sarg = 1.0 / sqrt (arg) ;
                        else
                            sarg = exp (-MJS * log (arg)) ;

                        CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 19] = BSIM4v7PhiBS * czbs * (1.0 - arg * sarg) / (1.0 - MJS) ;
                        BSIM4v7entry.d_BSIM4v7capbsRWArray [instance_ID] = czbs * sarg ;
                    } else {
                        CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 19] = 0.0 ;
                        BSIM4v7entry.d_BSIM4v7capbsRWArray [instance_ID] = 0.0 ;
                    }
                    if (czbssw > 0.0)
                    {
                        arg = 1.0 - vbs_jct / BSIM4v7PhiBSWS ;

                        if (MJSWS == 0.5)
                            sarg = 1.0 / sqrt (arg) ;
                        else
                            sarg = exp (-MJSWS * log (arg)) ;

                        CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 19] += BSIM4v7PhiBSWS * czbssw * (1.0 - arg * sarg) / (1.0 - MJSWS) ;
                        BSIM4v7entry.d_BSIM4v7capbsRWArray [instance_ID] += czbssw * sarg ;
                    }
                    if (czbsswg > 0.0)
                    {
                        arg = 1.0 - vbs_jct / BSIM4v7PhiBSWGS ;

                        if (MJSWGS == 0.5)
                            sarg = 1.0 / sqrt (arg) ;
                        else
                            sarg = exp (-MJSWGS * log (arg)) ;

                        CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 19] += BSIM4v7PhiBSWGS * czbsswg * (1.0 - arg * sarg) /
                                                       (1.0 - MJSWGS) ;
                        BSIM4v7entry.d_BSIM4v7capbsRWArray [instance_ID] += czbsswg * sarg ;
                    }
                }
                else
                {
                    T0 = czbs + czbssw + czbsswg ;
                    T1 = vbs_jct * (czbs * MJS / BSIM4v7PhiBS + czbssw * MJSWS / BSIM4v7PhiBSWS +
                         czbsswg * MJSWGS / BSIM4v7PhiBSWGS) ;
                    CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 19] = vbs_jct * (T0 + 0.5 * T1) ;
                    BSIM4v7entry.d_BSIM4v7capbsRWArray [instance_ID] = T0 + T1 ;
                }


                /* Drain Bulk Junction */
                /* 71 - DIVERGENT */
                if (vbd_jct == 0.0)
                {
                    CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 21] = 0.0 ;
                    BSIM4v7entry.d_BSIM4v7capbdRWArray [instance_ID] = czbd + czbdsw + czbdswg ;
                }
                else if (vbd_jct < 0.0)
                {
                    if (czbd > 0.0)
                    {
                        arg = 1.0 - vbd_jct / BSIM4v7PhiBD ;

                        if (MJD == 0.5)
                            sarg = 1.0 / sqrt (arg) ;
                        else
                            sarg = exp (-MJD * log (arg)) ;

                        CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 21] = BSIM4v7PhiBD* czbd * (1.0 - arg * sarg) / (1.0 - MJD) ;
                        BSIM4v7entry.d_BSIM4v7capbdRWArray [instance_ID] = czbd * sarg ;
                    }
                    else
                    {
                        CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 21] = 0.0 ;
                        BSIM4v7entry.d_BSIM4v7capbdRWArray [instance_ID] = 0.0 ;
                    }
                    if (czbdsw > 0.0)
                    {
                        arg = 1.0 - vbd_jct / BSIM4v7PhiBSWD ;

                        if (MJSWD == 0.5)
                            sarg = 1.0 / sqrt (arg) ;
                        else
                            sarg = exp (-MJSWD * log (arg)) ;

                        CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 21] += BSIM4v7PhiBSWD * czbdsw * (1.0 - arg * sarg) / (1.0 - MJSWD) ;
                        BSIM4v7entry.d_BSIM4v7capbdRWArray [instance_ID] += czbdsw * sarg ;
                    }
                    if (czbdswg > 0.0)
                    {
                        arg = 1.0 - vbd_jct / BSIM4v7PhiBSWGD ;

                        if (MJSWGD == 0.5)
                            sarg = 1.0 / sqrt (arg) ;
                        else
                            sarg = exp (-MJSWGD * log (arg)) ;

                        CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 21] += BSIM4v7PhiBSWGD * czbdswg * (1.0 - arg * sarg) /
                                                       (1.0 - MJSWGD) ;
                        BSIM4v7entry.d_BSIM4v7capbdRWArray [instance_ID] += czbdswg * sarg ;
                    }
                } else {
                    T0 = czbd + czbdsw + czbdswg ;
                    T1 = vbd_jct * (czbd * MJD / BSIM4v7PhiBD + czbdsw * MJSWD / BSIM4v7PhiBSWD +
                         czbdswg * MJSWGD / BSIM4v7PhiBSWGD) ;
                    CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 21] = vbd_jct * (T0 + 0.5 * T1) ;
                    BSIM4v7entry.d_BSIM4v7capbdRWArray [instance_ID] = T0 + T1 ;
                }
            }


            /* check convergence */
            /* 72 - DIVERGENT */
            if ((BSIM4v7entry.d_BSIM4v7offArray [instance_ID] == 0) || (!(CKTmode & MODEINITFIX)))
            {
                if (Check == 1)
                {
                    atomicAdd (d_CKTnoncon, 1) ;

#ifndef NEWCONV
                } else {

                    if (BSIM4v7entry.d_BSIM4v7modeArray [instance_ID] >= 0)
                        Idtot = BSIM4v7entry.d_BSIM4v7cdRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7csubRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7IgidlRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7cbdRWArray [instance_ID] ;
                    else
                        Idtot = BSIM4v7entry.d_BSIM4v7cdRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cbdRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7IgidlRWArray [instance_ID] ; /* bugfix */

                    tol0 = CKTrelTol * MAX (fabs (cdhat), fabs (Idtot)) + CKTabsTol ;
                    tol1 = CKTrelTol * MAX (fabs (cseshat), fabs (Isestot)) + CKTabsTol ;
                    tol2 = CKTrelTol * MAX (fabs (cdedhat), fabs (Idedtot)) + CKTabsTol ;
                    tol3 = CKTrelTol * MAX (fabs (cgshat), fabs (Igstot)) + CKTabsTol ;
                    tol4 = CKTrelTol * MAX (fabs (cgdhat), fabs (Igdtot)) + CKTabsTol ;
                    tol5 = CKTrelTol * MAX (fabs (cgbhat), fabs (Igbtot)) + CKTabsTol ;
                    if ((fabs (cdhat - Idtot) >= tol0) || (fabs (cseshat - Isestot) >= tol1) ||
                        (fabs(cdedhat - Idedtot) >= tol2))
                    {
                        atomicAdd (d_CKTnoncon, 1) ;
                    }
                    else if ((fabs(cgshat - Igstot) >= tol3) || (fabs(cgdhat - Igdtot) >= tol4) ||
                        (fabs(cgbhat - Igbtot) >= tol5))
                    {
                        atomicAdd (d_CKTnoncon, 1) ;
                    } else {
                        Ibtot = BSIM4v7entry.d_BSIM4v7cbsRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cbdRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7IgidlRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7IgislRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7csubRWArray [instance_ID] ;
                        tol6 = CKTrelTol * MAX (fabs (cbhat), fabs (Ibtot)) + CKTabsTol ;
                        if (fabs (cbhat - Ibtot) > tol6)
                        {
                            atomicAdd (d_CKTnoncon, 1) ;
                        }
                    }
#endif /* NEWCONV */

                }
            }
            CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 3] = vds ;
            CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 2] = vgs ;
            CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 1] = vbs ;
            CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID]] = vbd ;
            CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 7] = vges ;
            CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 8] = vgms ;
            CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 4] = vdbs ;
            CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 5] = vdbd ;
            CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 6] = vsbs ;
            CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 9] = vses ;
            CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 10] = vdes ;
            CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 27] = qdef ;


            if (!ChargeComputationNeeded)
                goto line850 ;

            /* 73 - DIVERGENT */
            if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 3)
            {
                vgdx = vgmd ;
                vgsx = vgms ;
            } else {
                /* For rgateMod == 0, 1 and 2 */
                vgdx = vgd ;
                vgsx = vgs ;
            }

            if (BSIM4v7capMod == 0)
            {
                cgdo = pParam->BSIM4v7cgdo ;
                qgdo = pParam->BSIM4v7cgdo * vgdx ;
                cgso = pParam->BSIM4v7cgso ;
                qgso = pParam->BSIM4v7cgso * vgsx ;
            } else {
                /* For both capMod == 1 and 2 */
                T0 = vgdx + DELTA_1 ;
                T1 = sqrt (T0 * T0 + 4.0 * DELTA_1) ;
                T2 = 0.5 * (T0 - T1) ;
                T3 = pParam->BSIM4v7weffCV * pParam->BSIM4v7cgdl ;
                T4 = sqrt(1.0 - 4.0 * T2 / pParam->BSIM4v7ckappad) ;
                cgdo = pParam->BSIM4v7cgdo + T3 - T3 * (1.0 - 1.0 / T4) * (0.5 - 0.5 * T0 / T1) ;
                qgdo = (pParam->BSIM4v7cgdo + T3) * vgdx - T3 * (T2 + 0.5 * pParam->BSIM4v7ckappad * (T4 - 1.0)) ;
                T0 = vgsx + DELTA_1 ;
                T1 = sqrt(T0 * T0 + 4.0 * DELTA_1) ;
                T2 = 0.5 * (T0 - T1) ;
                T3 = pParam->BSIM4v7weffCV * pParam->BSIM4v7cgsl ;
                T4 = sqrt(1.0 - 4.0 * T2 / pParam->BSIM4v7ckappas) ;
                cgso = pParam->BSIM4v7cgso + T3 - T3 * (1.0 - 1.0 / T4) * (0.5 - 0.5 * T0 / T1) ;
                qgso = (pParam->BSIM4v7cgso + T3) * vgsx - T3 * (T2 + 0.5 * pParam->BSIM4v7ckappas * (T4 - 1.0)) ;
            }

            /* 74 - DIVERGENT */
            if (BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] != 1.0)
            {
                cgdo *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                cgso *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                qgdo *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
                qgso *= BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] ;
            }
            BSIM4v7entry.d_BSIM4v7cgdoArray [instance_ID] = cgdo ;
            BSIM4v7entry.d_BSIM4v7qgdoArray [instance_ID] = qgdo ;
            BSIM4v7entry.d_BSIM4v7cgsoArray [instance_ID] = cgso ;
            BSIM4v7entry.d_BSIM4v7qgsoArray [instance_ID] = qgso ;

#ifndef NOBYPASS
line755:
#endif
            ag0 = CKTag_0 ;

            /* 75 - DIVERGENT - CRITICAL */
            if (BSIM4v7entry.d_BSIM4v7modeArray [instance_ID] > 0)
            {
                if (BSIM4v7entry.d_BSIM4v7trnqsModArray [instance_ID] == 0)
                {
                    qdrn -= qgdo ;
                    if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 3)
                    {
                        gcgmgmb = (cgdo + cgso + pParam->BSIM4v7cgbo) * ag0 ;
                        gcgmdb = -cgdo * ag0 ;
                        gcgmsb = -cgso * ag0 ;
                        gcgmbb = -pParam->BSIM4v7cgbo * ag0 ;
                        gcdgmb = gcgmdb ;
                        gcsgmb = gcgmsb ;
                        gcbgmb = gcgmbb ;
                        gcggb = BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] * ag0 ;
                        gcgdb = BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] * ag0 ;
                        gcgsb = BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] * ag0 ;
                        gcgbb = -(gcggb + gcgdb + gcgsb) ;
                        gcdgb = BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] * ag0 ;
                        gcsgb = -(BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID]) * ag0 ;
                        gcbgb = BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] * ag0 ;
                        qgmb = pParam->BSIM4v7cgbo * vgmb ;
                        qgmid = qgdo + qgso + qgmb ;
                        qbulk -= qgmb ;
                        qsrc = -(qgate + qgmid + qbulk + qdrn) ;
                    } else {
                        gcggb = (BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] + cgdo + cgso + pParam->BSIM4v7cgbo ) * ag0 ;
                        gcgdb = (BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] - cgdo) * ag0 ;
                        gcgsb = (BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] - cgso) * ag0 ;
                        gcgbb = -(gcggb + gcgdb + gcgsb) ;
                        gcdgb = (BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] - cgdo) * ag0 ;
                        gcsgb = -(BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] + cgso) * ag0 ;
                        gcbgb = (BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] - pParam->BSIM4v7cgbo) * ag0 ;
                        gcdgmb = gcsgmb = gcbgmb = 0.0 ;
                        qgb = pParam->BSIM4v7cgbo * vgb ;
                        qgate += qgdo + qgso + qgb ;
                        qbulk -= qgb ;
                        qsrc = -(qgate + qbulk + qdrn) ;
                    }
                    gcddb = (BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7capbdRWArray [instance_ID] + cgdo) * ag0 ;
                    gcdsb = BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID] * ag0 ;
                    gcsdb = -(BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cbdbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID]) * ag0 ;
                    gcssb = (BSIM4v7entry.d_BSIM4v7capbsRWArray [instance_ID] + cgso - (BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cbsbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID])) * ag0 ;

                    if (!BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID])
                    {
                        gcdbb = -(gcdgb + gcddb + gcdsb + gcdgmb) ;
                        gcsbb = -(gcsgb + gcsdb + gcssb + gcsgmb) ;
                        gcbdb = (BSIM4v7entry.d_BSIM4v7cbdbRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7capbdRWArray [instance_ID]) * ag0 ;
                        gcbsb = (BSIM4v7entry.d_BSIM4v7cbsbRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7capbsRWArray [instance_ID]) * ag0 ;
                        gcdbdb = 0.0 ;
                        gcsbsb = 0.0 ;
                    } else {
                        gcdbb = -(BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID]) * ag0 ;
                        gcsbb = -(gcsgb + gcsdb + gcssb + gcsgmb) + BSIM4v7entry.d_BSIM4v7capbsRWArray [instance_ID] * ag0 ;
                        gcbdb = BSIM4v7entry.d_BSIM4v7cbdbRWArray [instance_ID] * ag0 ;
                        gcbsb = BSIM4v7entry.d_BSIM4v7cbsbRWArray [instance_ID] * ag0 ;
                        gcdbdb = -BSIM4v7entry.d_BSIM4v7capbdRWArray [instance_ID] * ag0 ;
                        gcsbsb = -BSIM4v7entry.d_BSIM4v7capbsRWArray [instance_ID] * ag0 ;
                    }
                    gcbbb = -(gcbdb + gcbgb + gcbsb + gcbgmb) ;
                    ggtg = ggtd = ggtb = ggts = 0.0 ;
                    sxpart = 0.6 ;
                    dxpart = 0.4 ;
                    ddxpart_dVd = ddxpart_dVg = ddxpart_dVb = ddxpart_dVs = 0.0 ;
                    dsxpart_dVd = dsxpart_dVg = dsxpart_dVb = dsxpart_dVs = 0.0 ;
                } else {
                    qcheq = BSIM4v7entry.d_BSIM4v7qchqsArray [instance_ID] ;
                    CoxWL = BSIM4v7coxe * pParam->BSIM4v7weffCV * BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] * pParam->BSIM4v7leffCV ;
                    T0 = qdef * ScalingFactor / CoxWL ;
                    ggtg = BSIM4v7entry.d_BSIM4v7gtgArray [instance_ID] = T0 * BSIM4v7entry.d_BSIM4v7gcrggArray [instance_ID] ;
                    ggtd = BSIM4v7entry.d_BSIM4v7gtdArray [instance_ID] = T0 * BSIM4v7entry.d_BSIM4v7gcrgdArray [instance_ID] ;
                    ggts = BSIM4v7entry.d_BSIM4v7gtsArray [instance_ID] = T0 * BSIM4v7entry.d_BSIM4v7gcrgsArray [instance_ID] ;
                    ggtb = BSIM4v7entry.d_BSIM4v7gtbArray [instance_ID] = T0 * BSIM4v7entry.d_BSIM4v7gcrgbArray [instance_ID] ;
                    gqdef = ScalingFactor * ag0 ;
                    gcqgb = BSIM4v7entry.d_BSIM4v7cqgbArray [instance_ID] * ag0 ;
                    gcqdb = BSIM4v7entry.d_BSIM4v7cqdbArray [instance_ID] * ag0 ;
                    gcqsb = BSIM4v7entry.d_BSIM4v7cqsbArray [instance_ID] * ag0 ;
                    gcqbb = BSIM4v7entry.d_BSIM4v7cqbbArray [instance_ID] * ag0 ;

                    if (fabs (qcheq) <= 1.0e-5 * CoxWL)
                    {

                        if (BSIM4v7xpart < 0.5)
                            dxpart = 0.4 ;
                        else if (BSIM4v7xpart > 0.5)
                            dxpart = 0.0 ;
                        else
                            dxpart = 0.5 ;

                        ddxpart_dVd = ddxpart_dVg = ddxpart_dVb = ddxpart_dVs = 0.0 ;
                    } else {
                        dxpart = qdrn / qcheq ;
                        Cdd = BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] ;
                        Csd = -(BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cbdbRWArray [instance_ID]) ;
                        ddxpart_dVd = (Cdd - dxpart * (Cdd + Csd)) / qcheq ;
                        Cdg = BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] ;
                        Csg = -(BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID]) ;
                        ddxpart_dVg = (Cdg - dxpart * (Cdg + Csg)) / qcheq ;
                        Cds = BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID] ;
                        Css = -(BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cbsbRWArray [instance_ID]) ;
                        ddxpart_dVs = (Cds - dxpart * (Cds + Css)) / qcheq ;
                        ddxpart_dVb = -(ddxpart_dVd + ddxpart_dVg + ddxpart_dVs) ;
                    }
                    sxpart = 1.0 - dxpart ;
                    dsxpart_dVd = -ddxpart_dVd ;
                    dsxpart_dVg = -ddxpart_dVg ;
                    dsxpart_dVs = -ddxpart_dVs ;
                    dsxpart_dVb = -(dsxpart_dVd + dsxpart_dVg + dsxpart_dVs) ;

                    if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 3)
                    {
                        gcgmgmb = (cgdo + cgso + pParam->BSIM4v7cgbo) * ag0 ;
                        gcgmdb = -cgdo * ag0 ;
                        gcgmsb = -cgso * ag0 ;
                        gcgmbb = -pParam->BSIM4v7cgbo * ag0 ;
                        gcdgmb = gcgmdb ;
                        gcsgmb = gcgmsb ;
                        gcbgmb = gcgmbb ;
                        gcdgb = gcsgb = gcbgb = 0.0 ;
                        gcggb = gcgdb = gcgsb = gcgbb = 0.0 ;
                        qgmb = pParam->BSIM4v7cgbo * vgmb ;
                        qgmid = qgdo + qgso + qgmb ;
                        qgate = 0.0 ;
                        qbulk = -qgmb ;
                        qdrn = -qgdo ;
                        qsrc = -(qgmid + qbulk + qdrn) ;
                    } else {
                        gcggb = (cgdo + cgso + pParam->BSIM4v7cgbo) * ag0 ;
                        gcgdb = -cgdo * ag0 ;
                        gcgsb = -cgso * ag0 ;
                        gcgbb = -pParam->BSIM4v7cgbo * ag0 ;
                        gcdgb = gcgdb ;
                        gcsgb = gcgsb ;
                        gcbgb = gcgbb ;
                        gcdgmb = gcsgmb = gcbgmb = 0.0 ;
                        qgb = pParam->BSIM4v7cgbo * vgb ;
                        qgate = qgdo + qgso + qgb ;
                        qbulk = -qgb ;
                        qdrn = -qgdo ;
                        qsrc = -(qgate + qbulk + qdrn) ;
                    }
                    gcddb = (BSIM4v7entry.d_BSIM4v7capbdRWArray [instance_ID] + cgdo) * ag0 ;
                    gcdsb = gcsdb = 0.0 ;
                    gcssb = (BSIM4v7entry.d_BSIM4v7capbsRWArray [instance_ID] + cgso) * ag0 ;

                    if (!BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID])
                    {
                        gcdbb = -(gcdgb + gcddb + gcdgmb) ;
                        gcsbb = -(gcsgb + gcssb + gcsgmb) ;
                        gcbdb = -BSIM4v7entry.d_BSIM4v7capbdRWArray [instance_ID] * ag0 ;
                        gcbsb = -BSIM4v7entry.d_BSIM4v7capbsRWArray [instance_ID] * ag0 ;
                        gcdbdb = 0.0 ;
                        gcsbsb = 0.0 ;
                    } else {
                        gcdbb = gcsbb = gcbdb = gcbsb = 0.0 ;
                        gcdbdb = -BSIM4v7entry.d_BSIM4v7capbdRWArray [instance_ID] * ag0 ;
                        gcsbsb = -BSIM4v7entry.d_BSIM4v7capbsRWArray [instance_ID] * ag0 ;
                    }
                    gcbbb = -(gcbdb + gcbgb + gcbsb + gcbgmb) ;
                }
            } else {
                if (BSIM4v7entry.d_BSIM4v7trnqsModArray [instance_ID] == 0)
                {
                    qsrc = qdrn - qgso ;
                    if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 3)
                    {
                        gcgmgmb = (cgdo + cgso + pParam->BSIM4v7cgbo) * ag0 ;
                        gcgmdb = -cgdo * ag0 ;
                        gcgmsb = -cgso * ag0 ;
                        gcgmbb = -pParam->BSIM4v7cgbo * ag0 ;
                        gcdgmb = gcgmdb ;
                        gcsgmb = gcgmsb ;
                        gcbgmb = gcgmbb ;
                        gcggb = BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] * ag0 ;
                        gcgdb = BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] * ag0 ;
                        gcgsb = BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] * ag0 ;
                        gcgbb = -(gcggb + gcgdb + gcgsb) ;
                        gcdgb = -(BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID]) * ag0 ;
                        gcsgb = BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] * ag0 ;
                        gcbgb = BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] * ag0 ;
                        qgmb = pParam->BSIM4v7cgbo * vgmb ;
                        qgmid = qgdo + qgso + qgmb ;
                        qbulk -= qgmb ;
                        qdrn = -(qgate + qgmid + qbulk + qsrc) ;
                    } else {
                        gcggb = (BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] + cgdo + cgso + pParam->BSIM4v7cgbo ) * ag0 ;
                        gcgdb = (BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] - cgdo) * ag0 ;
                        gcgsb = (BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] - cgso) * ag0 ;
                        gcgbb = -(gcggb + gcgdb + gcgsb) ;
                        gcdgb = -(BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] + cgdo) * ag0 ;
                        gcsgb = (BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] - cgso) * ag0 ;
                        gcbgb = (BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID] - pParam->BSIM4v7cgbo) * ag0 ;
                        gcdgmb = gcsgmb = gcbgmb = 0.0 ;
                        qgb = pParam->BSIM4v7cgbo * vgb ;
                        qgate += qgdo + qgso + qgb ;
                        qbulk -= qgb ;
                        qdrn = -(qgate + qbulk + qsrc) ;
                    }
                    gcddb = (BSIM4v7entry.d_BSIM4v7capbdRWArray [instance_ID] + cgdo - (BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cbsbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID])) * ag0 ;
                    gcdsb = -(BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cbdbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID]) * ag0 ;
                    gcsdb = BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID] * ag0 ;
                    gcssb = (BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7capbsRWArray [instance_ID] + cgso) * ag0 ;

                    if (!BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID])
                    {
                        gcdbb = -(gcdgb + gcddb + gcdsb + gcdgmb) ;
                        gcsbb = -(gcsgb + gcsdb + gcssb + gcsgmb) ;
                        gcbdb = (BSIM4v7entry.d_BSIM4v7cbsbRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7capbdRWArray [instance_ID]) * ag0 ;
                        gcbsb = (BSIM4v7entry.d_BSIM4v7cbdbRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7capbsRWArray [instance_ID]) * ag0 ;
                        gcdbdb = 0.0 ;
                        gcsbsb = 0.0 ;
                    } else {
                        gcdbb = -(gcdgb + gcddb + gcdsb + gcdgmb) + BSIM4v7entry.d_BSIM4v7capbdRWArray [instance_ID] * ag0 ;
                        gcsbb = -(BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID]) * ag0 ;
                        gcbdb = BSIM4v7entry.d_BSIM4v7cbsbRWArray [instance_ID] * ag0 ;
                        gcbsb = BSIM4v7entry.d_BSIM4v7cbdbRWArray [instance_ID] * ag0 ;
                        gcdbdb = -BSIM4v7entry.d_BSIM4v7capbdRWArray [instance_ID] * ag0 ;
                        gcsbsb = -BSIM4v7entry.d_BSIM4v7capbsRWArray [instance_ID] * ag0 ;
                    }
                    gcbbb = -(gcbgb + gcbdb + gcbsb + gcbgmb) ;
                    ggtg = ggtd = ggtb = ggts = 0.0 ;
                    sxpart = 0.4 ;
                    dxpart = 0.6 ;
                    ddxpart_dVd = ddxpart_dVg = ddxpart_dVb = ddxpart_dVs = 0.0 ;
                    dsxpart_dVd = dsxpart_dVg = dsxpart_dVb = dsxpart_dVs = 0.0 ;
                } else {
                    qcheq = BSIM4v7entry.d_BSIM4v7qchqsArray [instance_ID] ;
                    CoxWL = BSIM4v7coxe * pParam->BSIM4v7weffCV * BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] * pParam->BSIM4v7leffCV ;
                    T0 = qdef * ScalingFactor / CoxWL ;
                    ggtg = BSIM4v7entry.d_BSIM4v7gtgArray [instance_ID] = T0 * BSIM4v7entry.d_BSIM4v7gcrggArray [instance_ID] ;
                    ggts = BSIM4v7entry.d_BSIM4v7gtsArray [instance_ID] = T0 * BSIM4v7entry.d_BSIM4v7gcrgdArray [instance_ID] ;
                    ggtd = BSIM4v7entry.d_BSIM4v7gtdArray [instance_ID] = T0 * BSIM4v7entry.d_BSIM4v7gcrgsArray [instance_ID] ;
                    ggtb = BSIM4v7entry.d_BSIM4v7gtbArray [instance_ID] = T0 * BSIM4v7entry.d_BSIM4v7gcrgbArray [instance_ID] ;
                    gqdef = ScalingFactor * ag0 ;
                    gcqgb = BSIM4v7entry.d_BSIM4v7cqgbArray [instance_ID] * ag0 ;
                    gcqdb = BSIM4v7entry.d_BSIM4v7cqsbArray [instance_ID] * ag0 ;
                    gcqsb = BSIM4v7entry.d_BSIM4v7cqdbArray [instance_ID] * ag0 ;
                    gcqbb = BSIM4v7entry.d_BSIM4v7cqbbArray [instance_ID] * ag0 ;

                    if (fabs (qcheq) <= 1.0e-5 * CoxWL)
                    {

                        if (BSIM4v7xpart < 0.5)
                            sxpart = 0.4 ;
                        else if (BSIM4v7xpart > 0.5)
                            sxpart = 0.0 ;
                        else
                            sxpart = 0.5 ;

                        dsxpart_dVd = dsxpart_dVg = dsxpart_dVb = dsxpart_dVs = 0.0 ;
                    } else {
                        sxpart = qdrn / qcheq ;
                        Css = BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] ;
                        Cds = -(BSIM4v7entry.d_BSIM4v7cgdbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cddbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cbdbRWArray [instance_ID]) ;
                        dsxpart_dVs = (Css - sxpart * (Css + Cds)) / qcheq ;
                        Csg = BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] ;
                        Cdg = -(BSIM4v7entry.d_BSIM4v7cggbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cdgbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cbgbRWArray [instance_ID]) ;
                        dsxpart_dVg = (Csg - sxpart * (Csg + Cdg)) / qcheq ;
                        Csd = BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID] ;
                        Cdd = -(BSIM4v7entry.d_BSIM4v7cgsbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cdsbRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7cbsbRWArray [instance_ID]) ;
                        dsxpart_dVd = (Csd - sxpart * (Csd + Cdd)) / qcheq ;
                        dsxpart_dVb = -(dsxpart_dVd + dsxpart_dVg + dsxpart_dVs) ;
                    }
                    dxpart = 1.0 - sxpart ;
                    ddxpart_dVd = -dsxpart_dVd ;
                    ddxpart_dVg = -dsxpart_dVg ;
                    ddxpart_dVs = -dsxpart_dVs ;
                    ddxpart_dVb = -(ddxpart_dVd + ddxpart_dVg + ddxpart_dVs) ;

                    if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 3)
                    {
                        gcgmgmb = (cgdo + cgso + pParam->BSIM4v7cgbo) * ag0 ;
                        gcgmdb = -cgdo * ag0 ;
                        gcgmsb = -cgso * ag0 ;
                        gcgmbb = -pParam->BSIM4v7cgbo * ag0 ;
                        gcdgmb = gcgmdb ;
                        gcsgmb = gcgmsb ;
                        gcbgmb = gcgmbb ;
                        gcdgb = gcsgb = gcbgb = 0.0 ;
                        gcggb = gcgdb = gcgsb = gcgbb = 0.0 ;
                        qgmb = pParam->BSIM4v7cgbo * vgmb ;
                        qgmid = qgdo + qgso + qgmb ;
                        qgate = 0.0 ;
                        qbulk = -qgmb ;
                        qdrn = -qgdo ;
                        qsrc = -qgso ;
                    } else {
                        gcggb = (cgdo + cgso + pParam->BSIM4v7cgbo ) * ag0 ;
                        gcgdb = -cgdo * ag0 ;
                        gcgsb = -cgso * ag0 ;
                        gcgbb = -pParam->BSIM4v7cgbo * ag0 ;
                        gcdgb = gcgdb ;
                        gcsgb = gcgsb ;
                        gcbgb = gcgbb ;
                        gcdgmb = gcsgmb = gcbgmb = 0.0 ;
                        qgb = pParam->BSIM4v7cgbo * vgb ;
                        qgate = qgdo + qgso + qgb ;
                        qbulk = -qgb ;
                        qdrn = -qgdo ;
                        qsrc = -qgso ;
                    }
                    gcddb = (BSIM4v7entry.d_BSIM4v7capbdRWArray [instance_ID] + cgdo) * ag0 ;
                    gcdsb = gcsdb = 0.0 ;
                    gcssb = (BSIM4v7entry.d_BSIM4v7capbsRWArray [instance_ID] + cgso) * ag0 ;

                    if (!BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID])
                    {
                        gcdbb = -(gcdgb + gcddb + gcdgmb) ;
                        gcsbb = -(gcsgb + gcssb + gcsgmb) ;
                        gcbdb = -BSIM4v7entry.d_BSIM4v7capbdRWArray [instance_ID] * ag0 ;
                        gcbsb = -BSIM4v7entry.d_BSIM4v7capbsRWArray [instance_ID] * ag0 ;
                        gcdbdb = 0.0 ;
                        gcsbsb = 0.0 ;
                    } else {
                        gcdbb = gcsbb = gcbdb = gcbsb = 0.0 ;
                        gcdbdb = -BSIM4v7entry.d_BSIM4v7capbdRWArray [instance_ID] * ag0 ;
                        gcsbsb = -BSIM4v7entry.d_BSIM4v7capbsRWArray [instance_ID] * ag0 ;
                    }
                    gcbbb = -(gcbdb + gcbgb + gcbsb + gcbgmb) ;
                }
            }

            /* 76 - DIVERGENT */
            if (BSIM4v7entry.d_BSIM4v7trnqsModArray [instance_ID])
            {
                CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 25] = qdef * ScalingFactor ;

                if (CKTmode & MODEINITTRAN)
                    CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 25] = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 25] ;

                error = cuNIintegrate_device_kernel (CKTstate_0, CKTstate_1, &geq, &ceq, 0.0,
                                                      BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 25, CKTag_0, CKTag_1, CKTorder) ;
                if (error)
                    printf ("Integration error\n\n") ;
                    //return (error) ;
            }

            if (ByPass)
                goto line860 ;

            CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 13] = qgate ;
            CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 15] = qdrn - CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 21] ;
            CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 28] = qsrc - CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 19] ;

            /* 77 - DIVERGENT */
            if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 3)
                CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 17] = qgmid ;

            /* 78 - DIVERGENT */
            if (!BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID])
                CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 11] = qbulk + CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 21] + CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 19] ;
            else
                CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 11] = qbulk ;


            /* Store small signal parameters */
            if (CKTmode & MODEINITSMSIG)
                goto line1000 ;

            /* I DON'T KNOW */
            if (!ChargeComputationNeeded)
                goto line850 ;

            if (CKTmode & MODEINITTRAN)
            {
                CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 11] = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 11] ;
                CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 13] = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 13] ;
                CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 15] = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 15] ;

                /* 79 - DIVERGENT */
                if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 3)
                    CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 17] = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 17] ;

                /* 80 - DIVERGENT */
                if (BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID])
                {
                    CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 19] = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 19] ;
                    CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 21] = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 21] ;
                }
            }

            error = cuNIintegrate_device_kernel (CKTstate_0, CKTstate_1, &geq, &ceq, 0.0,
                                                  BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 11, CKTag_0, CKTag_1, CKTorder) ;
            if (error)
                printf ("Integration error\n\n") ;
                //return (error) ;

            error = cuNIintegrate_device_kernel (CKTstate_0, CKTstate_1, &geq, &ceq, 0.0,
                                                  BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 13, CKTag_0, CKTag_1, CKTorder) ;
            if (error)
                printf ("Integration error\n\n") ;
                //return (error) ;

            error = cuNIintegrate_device_kernel (CKTstate_0, CKTstate_1, &geq, &ceq, 0.0,
                                                  BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 15, CKTag_0, CKTag_1, CKTorder) ;
            if (error)
                printf ("Integration error\n\n") ;
                //return (error) ;

            /* 81 - DIVERGENT */
            if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 3)
            {
                error = cuNIintegrate_device_kernel (CKTstate_0, CKTstate_1, &geq, &ceq, 0.0,
                                                      BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 17, CKTag_0, CKTag_1, CKTorder) ;
                if (error)
                    printf ("Integration error\n\n") ;
                    //return (error) ;
            }

            /* 82 - DIVERGENT */
            if (BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID])
            {
                error = cuNIintegrate_device_kernel (CKTstate_0, CKTstate_1, &geq, &ceq, 0.0,
                                                      BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 19, CKTag_0, CKTag_1, CKTorder) ;
                if (error)
                    printf ("Integration error\n\n") ;
                    //return (error) ;

                error = cuNIintegrate_device_kernel (CKTstate_0, CKTstate_1, &geq, &ceq, 0.0,
                                                      BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 21, CKTag_0, CKTag_1, CKTorder) ;
                if (error)
                    printf ("Integration error\n\n") ;
                    //return (error) ;
            }
            goto line860 ;


line850:
            /* Zero gcap and ceqcap if (!ChargeComputationNeeded) */
            ceqqg = ceqqb = ceqqd = 0.0 ;
            ceqqjd = ceqqjs = 0.0 ;
            cqcheq = cqdef = 0.0 ;
            gcdgb = gcddb = gcdsb = gcdbb = 0.0 ;
            gcsgb = gcsdb = gcssb = gcsbb = 0.0 ;
            gcggb = gcgdb = gcgsb = gcgbb = 0.0 ;
            gcbdb = gcbgb = gcbsb = gcbbb = 0.0 ;
            gcgmgmb = gcgmdb = gcgmsb = gcgmbb = 0.0 ;
            gcdgmb = gcsgmb = gcbgmb = ceqqgmid = 0.0 ;
            gcdbdb = gcsbsb = 0.0 ;
            gqdef = gcqgb = gcqdb = gcqsb = gcqbb = 0.0 ;
            ggtg = ggtd = ggtb = ggts = 0.0 ;
            sxpart = (1.0 - (dxpart = (BSIM4v7entry.d_BSIM4v7modeArray [instance_ID] > 0) ? 0.4 : 0.6)) ;
            ddxpart_dVd = ddxpart_dVg = ddxpart_dVb = ddxpart_dVs = 0.0 ;
            dsxpart_dVd = dsxpart_dVg = dsxpart_dVb = dsxpart_dVs = 0.0 ;

            /* 83 - DIVERGENT */
            if (BSIM4v7entry.d_BSIM4v7trnqsModArray [instance_ID])
            {
                CoxWL = BSIM4v7coxe * pParam->BSIM4v7weffCV * BSIM4v7entry.d_BSIM4v7nfArray [instance_ID] * pParam->BSIM4v7leffCV ;
                T1 = BSIM4v7entry.d_BSIM4v7gcrgRWArray [instance_ID] / CoxWL ;
                BSIM4v7entry.d_BSIM4v7gtauRWArray [instance_ID] = T1 * ScalingFactor ;
            }
            else
                BSIM4v7entry.d_BSIM4v7gtauRWArray [instance_ID] = 0.0 ;

            goto line900 ;


line860:
            /* Calculate equivalent charge current */
            cqgate = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 14] ;
            cqbody = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 12] ;
            cqdrn = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 16] ;
            ceqqg = cqgate - gcggb * vgb + gcgdb * vbd + gcgsb * vbs ;
            ceqqd = cqdrn - gcdgb * vgb - gcdgmb * vgmb + (gcddb + gcdbdb) * vbd - gcdbdb * vbd_jct + gcdsb * vbs ;
            ceqqb = cqbody - gcbgb * vgb - gcbgmb * vgmb + gcbdb * vbd + gcbsb * vbs ;

            /* 84 - DIVERGENT */
            if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 3)
                ceqqgmid = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 18] + gcgmdb * vbd + gcgmsb * vbs - gcgmgmb * vgmb ;
            else
                ceqqgmid = 0.0 ;

            /* 85 - DIVERGENT */
            if (BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID])
            {
                ceqqjs = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 20] + gcsbsb * vbs_jct ;
                ceqqjd = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 22] + gcdbdb * vbd_jct ;
            }

            /* 86 - DIVERGENT */
            if (BSIM4v7entry.d_BSIM4v7trnqsModArray [instance_ID])
            {
                T0 = ggtg * vgb - ggtd * vbd - ggts * vbs ;
                ceqqg += T0 ;
                T1 = qdef * BSIM4v7entry.d_BSIM4v7gtauRWArray [instance_ID] ;
                ceqqd -= dxpart * T0 + T1 * (ddxpart_dVg * vgb - ddxpart_dVd * vbd - ddxpart_dVs * vbs) ;
                cqdef = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 26] - gqdef * qdef ;
                cqcheq = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 24] - (gcqgb * vgb - gcqdb * vbd - gcqsb * vbs) + T0 ;
            }

            /* 21 - non-divergent */
            if (CKTmode & MODEINITTRAN)
            {
                CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 12] = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 12] ;
                CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 14] = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 14] ;
                CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 16] = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 16] ;

                /* 87 - DIVERGENT */
                if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 3)
                    CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 18] = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 18] ;

                /* 88 - DIVERGENT */
                if (BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID])
                {
                    CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 20] = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 20] ;
                    CKTstate_1 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 22] = CKTstate_0 [BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 22] ;
                }
            }


            /* Load current vector */
            /* 89 - DIVERGENT - CRITICAL */
line900:
            if (BSIM4v7entry.d_BSIM4v7modeArray [instance_ID] >= 0)
            {
                Gm = BSIM4v7entry.d_BSIM4v7gmRWArray [instance_ID] ;
                Gmbs = BSIM4v7entry.d_BSIM4v7gmbsRWArray [instance_ID] ;
                FwdSum = Gm + Gmbs ;
                RevSum = 0.0 ;
                ceqdrn = BSIM4v7type * (cdrain - BSIM4v7entry.d_BSIM4v7gdsRWArray [instance_ID] * vds - Gm * vgs - Gmbs * vbs) ;
                ceqbd = BSIM4v7type * (BSIM4v7entry.d_BSIM4v7csubRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7IgidlRWArray [instance_ID] -
                        (BSIM4v7entry.d_BSIM4v7gbdsArray [instance_ID] + BSIM4v7entry.d_BSIM4v7ggidldArray [instance_ID]) * vds -
                        (BSIM4v7entry.d_BSIM4v7gbgsArray [instance_ID] + BSIM4v7entry.d_BSIM4v7ggidlgArray [instance_ID]) * vgs - (BSIM4v7entry.d_BSIM4v7gbbsArray [instance_ID] + BSIM4v7entry.d_BSIM4v7ggidlbArray [instance_ID]) * vbs) ;
                ceqbs = BSIM4v7type * (BSIM4v7entry.d_BSIM4v7IgislRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7ggislsArray [instance_ID] * vds -
                        BSIM4v7entry.d_BSIM4v7ggislgArray [instance_ID] * vgd - BSIM4v7entry.d_BSIM4v7ggislbArray [instance_ID] * vbd) ;
                gbbdp = -(BSIM4v7entry.d_BSIM4v7gbdsArray [instance_ID]) ;
                gbbsp = BSIM4v7entry.d_BSIM4v7gbdsArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gbgsArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gbbsArray [instance_ID] ;
                gbdpg = BSIM4v7entry.d_BSIM4v7gbgsArray [instance_ID] ;
                gbdpdp = BSIM4v7entry.d_BSIM4v7gbdsArray [instance_ID] ;
                gbdpb = BSIM4v7entry.d_BSIM4v7gbbsArray [instance_ID] ;
                gbdpsp = -(gbdpg + gbdpdp + gbdpb) ;
                gbspg = 0.0 ;
                gbspdp = 0.0 ;
                gbspb = 0.0 ;
                gbspsp = 0.0 ;

                if (BSIM4v7igcMod)
                {
                    gIstotg = BSIM4v7entry.d_BSIM4v7gIgsgArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gIgcsgArray [instance_ID] ;
                    gIstotd = BSIM4v7entry.d_BSIM4v7gIgcsdArray [instance_ID] ;
                    gIstots = BSIM4v7entry.d_BSIM4v7gIgssArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gIgcssArray [instance_ID] ;
                    gIstotb = BSIM4v7entry.d_BSIM4v7gIgcsbArray [instance_ID] ;
                    Istoteq = BSIM4v7type * (BSIM4v7entry.d_BSIM4v7IgsRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7IgcsRWArray [instance_ID] - gIstotg * vgs -
                              BSIM4v7entry.d_BSIM4v7gIgcsdArray [instance_ID] * vds - BSIM4v7entry.d_BSIM4v7gIgcsbArray [instance_ID] * vbs) ;
                    gIdtotg = BSIM4v7entry.d_BSIM4v7gIgdgArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gIgcdgArray [instance_ID] ;
                    gIdtotd = BSIM4v7entry.d_BSIM4v7gIgddArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gIgcddArray [instance_ID] ;
                    gIdtots = BSIM4v7entry.d_BSIM4v7gIgcdsArray [instance_ID] ;
                    gIdtotb = BSIM4v7entry.d_BSIM4v7gIgcdbArray [instance_ID] ;
                    Idtoteq = BSIM4v7type * (BSIM4v7entry.d_BSIM4v7IgdRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7IgcdRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7gIgdgArray [instance_ID] * vgd -
                              BSIM4v7entry.d_BSIM4v7gIgcdgArray [instance_ID] * vgs - BSIM4v7entry.d_BSIM4v7gIgcddArray [instance_ID] * vds - BSIM4v7entry.d_BSIM4v7gIgcdbArray [instance_ID] * vbs) ;
                } else {
                    gIstotg = gIstotd = gIstots = gIstotb = Istoteq = 0.0 ;
                    gIdtotg = gIdtotd = gIdtots = gIdtotb = Idtoteq = 0.0 ;
                }

                if (BSIM4v7igbMod)
                {
                    gIbtotg = BSIM4v7entry.d_BSIM4v7gIgbgArray [instance_ID] ;
                    gIbtotd = BSIM4v7entry.d_BSIM4v7gIgbdArray [instance_ID] ;
                    gIbtots = BSIM4v7entry.d_BSIM4v7gIgbsArray [instance_ID] ;
                    gIbtotb = BSIM4v7entry.d_BSIM4v7gIgbbArray [instance_ID] ;
                    Ibtoteq = BSIM4v7type * (BSIM4v7entry.d_BSIM4v7IgbRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7gIgbgArray [instance_ID] * vgs - BSIM4v7entry.d_BSIM4v7gIgbdArray [instance_ID] * vds -
                              BSIM4v7entry.d_BSIM4v7gIgbbArray [instance_ID] * vbs) ;
                }
                else
                    gIbtotg = gIbtotd = gIbtots = gIbtotb = Ibtoteq = 0.0 ;

                if ((BSIM4v7igcMod != 0) || (BSIM4v7igbMod != 0))
                {
                    gIgtotg = gIstotg + gIdtotg + gIbtotg ;
                    gIgtotd = gIstotd + gIdtotd + gIbtotd ;
                    gIgtots = gIstots + gIdtots + gIbtots ;
                    gIgtotb = gIstotb + gIdtotb + gIbtotb ;
                    Igtoteq = Istoteq + Idtoteq + Ibtoteq ;
                }
                else
                    gIgtotg = gIgtotd = gIgtots = gIgtotb = Igtoteq = 0.0 ;

                if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 2)
                    T0 = vges - vgs ;
                else if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 3)
                    T0 = vgms - vgs ;

                if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] > 1)
                {
                    gcrgd = BSIM4v7entry.d_BSIM4v7gcrgdArray [instance_ID] * T0 ;
                    gcrgg = BSIM4v7entry.d_BSIM4v7gcrggArray [instance_ID] * T0 ;
                    gcrgs = BSIM4v7entry.d_BSIM4v7gcrgsArray [instance_ID] * T0 ;
                    gcrgb = BSIM4v7entry.d_BSIM4v7gcrgbArray [instance_ID] * T0 ;
                    ceqgcrg = -(gcrgd * vds + gcrgg * vgs + gcrgb * vbs) ;
                    gcrgg -= BSIM4v7entry.d_BSIM4v7gcrgRWArray [instance_ID] ;
                    gcrg = BSIM4v7entry.d_BSIM4v7gcrgRWArray [instance_ID] ;
                }
                else
                    ceqgcrg = gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0 ;
            } else {
                Gm = -BSIM4v7entry.d_BSIM4v7gmRWArray [instance_ID] ;
                Gmbs = -BSIM4v7entry.d_BSIM4v7gmbsRWArray [instance_ID] ;
                FwdSum = 0.0 ;
                RevSum = -(Gm + Gmbs) ;
                ceqdrn = -BSIM4v7type * (cdrain + BSIM4v7entry.d_BSIM4v7gdsRWArray [instance_ID] * vds + Gm * vgd + Gmbs * vbd) ;
                ceqbs = BSIM4v7type * (BSIM4v7entry.d_BSIM4v7csubRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7IgislRWArray [instance_ID] +
                        (BSIM4v7entry.d_BSIM4v7gbdsArray [instance_ID] + BSIM4v7entry.d_BSIM4v7ggislsArray [instance_ID]) * vds -
                        (BSIM4v7entry.d_BSIM4v7gbgsArray [instance_ID] + BSIM4v7entry.d_BSIM4v7ggislgArray [instance_ID]) * vgd - (BSIM4v7entry.d_BSIM4v7gbbsArray [instance_ID] + BSIM4v7entry.d_BSIM4v7ggislbArray [instance_ID]) * vbd) ;
                ceqbd = BSIM4v7type * (BSIM4v7entry.d_BSIM4v7IgidlRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7ggidldArray [instance_ID] * vds - BSIM4v7entry.d_BSIM4v7ggidlgArray [instance_ID] * vgs -
                        BSIM4v7entry.d_BSIM4v7ggidlbArray [instance_ID] * vbs) ;
                gbbsp = -(BSIM4v7entry.d_BSIM4v7gbdsArray [instance_ID]) ;
                gbbdp = BSIM4v7entry.d_BSIM4v7gbdsArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gbgsArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gbbsArray [instance_ID] ;
                gbdpg = 0.0 ;
                gbdpsp = 0.0 ;
                gbdpb = 0.0 ;
                gbdpdp = 0.0 ;
                gbspg = BSIM4v7entry.d_BSIM4v7gbgsArray [instance_ID] ;
                gbspsp = BSIM4v7entry.d_BSIM4v7gbdsArray [instance_ID] ;
                gbspb = BSIM4v7entry.d_BSIM4v7gbbsArray [instance_ID] ;
                gbspdp = -(gbspg + gbspsp + gbspb) ;

                if (BSIM4v7igcMod)
                {
                    gIstotg = BSIM4v7entry.d_BSIM4v7gIgsgArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gIgcdgArray [instance_ID] ;
                    gIstotd = BSIM4v7entry.d_BSIM4v7gIgcdsArray [instance_ID] ;
                    gIstots = BSIM4v7entry.d_BSIM4v7gIgssArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gIgcddArray [instance_ID] ;
                    gIstotb = BSIM4v7entry.d_BSIM4v7gIgcdbArray [instance_ID] ;
                    Istoteq = BSIM4v7type * (BSIM4v7entry.d_BSIM4v7IgsRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7IgcdRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7gIgsgArray [instance_ID] * vgs -
                              BSIM4v7entry.d_BSIM4v7gIgcdgArray [instance_ID] * vgd + BSIM4v7entry.d_BSIM4v7gIgcddArray [instance_ID] * vds - BSIM4v7entry.d_BSIM4v7gIgcdbArray [instance_ID] * vbd) ;
                    gIdtotg = BSIM4v7entry.d_BSIM4v7gIgdgArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gIgcsgArray [instance_ID] ;
                    gIdtotd = BSIM4v7entry.d_BSIM4v7gIgddArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gIgcssArray [instance_ID] ;
                    gIdtots = BSIM4v7entry.d_BSIM4v7gIgcsdArray [instance_ID] ;
                    gIdtotb = BSIM4v7entry.d_BSIM4v7gIgcsbArray [instance_ID] ;
                    Idtoteq = BSIM4v7type * (BSIM4v7entry.d_BSIM4v7IgdRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7IgcsRWArray [instance_ID] -
                              (BSIM4v7entry.d_BSIM4v7gIgdgArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gIgcsgArray [instance_ID]) * vgd + BSIM4v7entry.d_BSIM4v7gIgcsdArray [instance_ID] * vds -
                              BSIM4v7entry.d_BSIM4v7gIgcsbArray [instance_ID] * vbd) ;
                } else {
                    gIstotg = gIstotd = gIstots = gIstotb = Istoteq = 0.0 ;
                    gIdtotg = gIdtotd = gIdtots = gIdtotb = Idtoteq = 0.0 ;
                }

                if (BSIM4v7igbMod)
                {
                    gIbtotg = BSIM4v7entry.d_BSIM4v7gIgbgArray [instance_ID] ;
                    gIbtotd = BSIM4v7entry.d_BSIM4v7gIgbsArray [instance_ID] ;
                    gIbtots = BSIM4v7entry.d_BSIM4v7gIgbdArray [instance_ID] ;
                    gIbtotb = BSIM4v7entry.d_BSIM4v7gIgbbArray [instance_ID] ;
                    Ibtoteq = BSIM4v7type * (BSIM4v7entry.d_BSIM4v7IgbRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7gIgbgArray [instance_ID] * vgd +
                              BSIM4v7entry.d_BSIM4v7gIgbdArray [instance_ID] * vds - BSIM4v7entry.d_BSIM4v7gIgbbArray [instance_ID] * vbd) ;
                }
                else
                    gIbtotg = gIbtotd = gIbtots = gIbtotb = Ibtoteq = 0.0 ;

                if ((BSIM4v7igcMod != 0) || (BSIM4v7igbMod != 0))
                {
                    gIgtotg = gIstotg + gIdtotg + gIbtotg ;
                    gIgtotd = gIstotd + gIdtotd + gIbtotd ;
                    gIgtots = gIstots + gIdtots + gIbtots ;
                    gIgtotb = gIstotb + gIdtotb + gIbtotb ;
                    Igtoteq = Istoteq + Idtoteq + Ibtoteq ;
                }
                else
                    gIgtotg = gIgtotd = gIgtots = gIgtotb = Igtoteq = 0.0 ;

                if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 2)
                    T0 = vges - vgs ;
                else if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 3)
                    T0 = vgms - vgs ;

                if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] > 1)
                {
                    gcrgd = BSIM4v7entry.d_BSIM4v7gcrgsArray [instance_ID] * T0 ;
                    gcrgg = BSIM4v7entry.d_BSIM4v7gcrggArray [instance_ID] * T0 ;
                    gcrgs = BSIM4v7entry.d_BSIM4v7gcrgdArray [instance_ID] * T0 ;
                    gcrgb = BSIM4v7entry.d_BSIM4v7gcrgbArray [instance_ID] * T0 ;
                    ceqgcrg = -(gcrgg * vgd - gcrgs * vds + gcrgb * vbd) ;
                    gcrgg -= BSIM4v7entry.d_BSIM4v7gcrgRWArray [instance_ID] ;
                    gcrg = BSIM4v7entry.d_BSIM4v7gcrgRWArray [instance_ID] ;
                }
                else
                    ceqgcrg = gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0 ;
            }

            /* 22 - non-divergent */
            if (BSIM4v7rdsMod == 1)
            {
                ceqgstot = BSIM4v7type * (BSIM4v7entry.d_BSIM4v7gstotdArray [instance_ID] * vds + BSIM4v7entry.d_BSIM4v7gstotgArray [instance_ID] * vgs +
                           BSIM4v7entry.d_BSIM4v7gstotbArray [instance_ID] * vbs) ;

                /* WDLiu: ceqgstot flowing away from sNodePrime */
                gstot = BSIM4v7entry.d_BSIM4v7gstotArray [instance_ID] ;
                gstotd = BSIM4v7entry.d_BSIM4v7gstotdArray [instance_ID] ;
                gstotg = BSIM4v7entry.d_BSIM4v7gstotgArray [instance_ID] ;
                gstots = BSIM4v7entry.d_BSIM4v7gstotsArray [instance_ID] - gstot ;
                gstotb = BSIM4v7entry.d_BSIM4v7gstotbArray [instance_ID] ;
                ceqgdtot = -BSIM4v7type * (BSIM4v7entry.d_BSIM4v7gdtotdArray [instance_ID] * vds + BSIM4v7entry.d_BSIM4v7gdtotgArray [instance_ID] * vgs
                           + BSIM4v7entry.d_BSIM4v7gdtotbArray [instance_ID] * vbs) ;

                /* WDLiu: ceqgdtot defined as flowing into dNodePrime */
                gdtot = BSIM4v7entry.d_BSIM4v7gdtotArray [instance_ID] ;
                gdtotd = BSIM4v7entry.d_BSIM4v7gdtotdArray [instance_ID] - gdtot ;
                gdtotg = BSIM4v7entry.d_BSIM4v7gdtotgArray [instance_ID] ;
                gdtots = BSIM4v7entry.d_BSIM4v7gdtotsArray [instance_ID] ;
                gdtotb = BSIM4v7entry.d_BSIM4v7gdtotbArray [instance_ID] ;
            } else {
                gstot = gstotd = gstotg = gstots = gstotb = ceqgstot = 0.0 ;
                gdtot = gdtotd = gdtotg = gdtots = gdtotb = ceqgdtot = 0.0 ;
            }

            /* 23 - non-divergent */
            if (BSIM4v7type > 0)
            {
                ceqjs = (BSIM4v7entry.d_BSIM4v7cbsRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7gbsRWArray [instance_ID] * vbs_jct) ;
                ceqjd = (BSIM4v7entry.d_BSIM4v7cbdRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7gbdRWArray [instance_ID] * vbd_jct) ;
            } else {
                ceqjs = -(BSIM4v7entry.d_BSIM4v7cbsRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7gbsRWArray [instance_ID] * vbs_jct) ;
                ceqjd = -(BSIM4v7entry.d_BSIM4v7cbdRWArray [instance_ID] - BSIM4v7entry.d_BSIM4v7gbdRWArray [instance_ID] * vbd_jct) ;
                ceqqg = -ceqqg ;
                ceqqd = -ceqqd ;
                ceqqb = -ceqqb ;
                ceqgcrg = -ceqgcrg ;

                if (BSIM4v7entry.d_BSIM4v7trnqsModArray [instance_ID])
                {
                    cqdef = -cqdef ;
                    cqcheq = -cqcheq ;
                }

                if (BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID])
                {
                    ceqqjs = -ceqqjs ;
                    ceqqjd = -ceqqjd ;
                }

                if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 3)
                    ceqqgmid = -ceqqgmid ;
            }


            m = BSIM4v7entry.d_BSIM4v7mArray [instance_ID] ;


            /* Loading RHS */
            posRHS = d_PositionVectorRHS [instance_ID] ;
            total_offsetRHS = 0 ;

            d_CKTloadOutputRHS [posRHS + total_offsetRHS + 0] = m * (ceqjd - ceqbd + ceqgdtot - ceqdrn - ceqqd + Idtoteq) ;

            d_CKTloadOutputRHS [posRHS + total_offsetRHS + 1] = m * (ceqqg - ceqgcrg + Igtoteq) ;

            total_offsetRHS += 2 ;


            /* 90 - DIVERGENT */
            if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 2)
            {
                d_CKTloadOutputRHS [posRHS + total_offsetRHS + 0] = m * ceqgcrg ;

                total_offsetRHS += 1 ;
            }
            else if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 3)
            {
                d_CKTloadOutputRHS [posRHS + total_offsetRHS + 0] = m * (ceqqgmid + ceqgcrg) ;

                total_offsetRHS += 1 ;
            }

            /* 90 - DIVERGENT */
            if (!BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID])
            {
                d_CKTloadOutputRHS [posRHS + total_offsetRHS + 0] = m * (ceqbd + ceqbs - ceqjd - ceqjs - ceqqb + Ibtoteq) ;

                d_CKTloadOutputRHS [posRHS + total_offsetRHS + 1] = m * (ceqdrn - ceqbs + ceqjs + ceqqg + ceqqb + ceqqd + ceqqgmid - ceqgstot + Istoteq) ;

                total_offsetRHS += 2 ;

            } else {
                d_CKTloadOutputRHS [posRHS + total_offsetRHS + 0] = m * (ceqjd + ceqqjd) ;

                d_CKTloadOutputRHS [posRHS + total_offsetRHS + 1] = m * (ceqbd + ceqbs - ceqqb + Ibtoteq) ;

                d_CKTloadOutputRHS [posRHS + total_offsetRHS + 2] = m * (ceqjs + ceqqjs) ;

                d_CKTloadOutputRHS [posRHS + total_offsetRHS + 3] = m * (ceqdrn - ceqbs + ceqjs + ceqqd + ceqqg + ceqqb + ceqqjd + ceqqjs + ceqqgmid - ceqgstot + Istoteq) ;

                total_offsetRHS += 4 ;
            }

            /* 24 - non-divergent */
            if (BSIM4v7rdsMod)
            {
                d_CKTloadOutputRHS [posRHS + total_offsetRHS + 0] = m * ceqgdtot ;

                d_CKTloadOutputRHS [posRHS + total_offsetRHS + 1] = m * ceqgstot ;

                total_offsetRHS += 2 ;
            }

            /* 91 - DIVERGENT */
            if (BSIM4v7entry.d_BSIM4v7trnqsModArray [instance_ID])
                d_CKTloadOutputRHS [posRHS + total_offsetRHS + 2] = m * (cqcheq - cqdef) ;



            /* Loading Matrix */
            pos = d_PositionVector [instance_ID] ;
            total_offset = 0 ;

            /* 92 - DIVERGENT */
            if (!BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID])
            {
                gjbd = BSIM4v7entry.d_BSIM4v7gbdRWArray [instance_ID] ;
                gjbs = BSIM4v7entry.d_BSIM4v7gbsRWArray [instance_ID] ;
            } else
                gjbd = gjbs = 0.0 ;

            /* 25 - non-divergent */
            if (!BSIM4v7rdsMod)
            {
                gdpr = BSIM4v7entry.d_BSIM4v7drainConductanceArray [instance_ID] ;
                gspr = BSIM4v7entry.d_BSIM4v7sourceConductanceArray [instance_ID] ;
            } else
                gdpr = gspr = 0.0 ;

            geltd = BSIM4v7entry.d_BSIM4v7grgeltdArray [instance_ID] ;
            T1 = qdef * BSIM4v7entry.d_BSIM4v7gtauRWArray [instance_ID] ;

            /* 26 - non-divergent */
            if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 1)
            {
                d_CKTloadOutput [pos + total_offset + 0] = m * geltd ;

                d_CKTloadOutput [pos + total_offset + 1] = m * (gcggb + geltd - ggtg + gIgtotg) ;

                d_CKTloadOutput [pos + total_offset + 2] = m * (gcgdb - ggtd + gIgtotd) ;

                d_CKTloadOutput [pos + total_offset + 3] = m * (gcgsb - ggts + gIgtots) ;

                d_CKTloadOutput [pos + total_offset + 4] = m * (gcgbb - ggtb + gIgtotb) ;

                total_offset += 5 ;
            }
            else if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 2)
            {
                d_CKTloadOutput [pos + total_offset + 0] = m * gcrg ;

                d_CKTloadOutput [pos + total_offset + 1] = m * gcrgg ;

                d_CKTloadOutput [pos + total_offset + 2] = m * gcrgd ;

                d_CKTloadOutput [pos + total_offset + 3] = m * gcrgs ;

                d_CKTloadOutput [pos + total_offset + 4] = m * gcrgb ;

                d_CKTloadOutput [pos + total_offset + 5] = m * (gcggb  - gcrgg - ggtg + gIgtotg) ;

                d_CKTloadOutput [pos + total_offset + 6] = m * (gcgdb - gcrgd - ggtd + gIgtotd) ;

                d_CKTloadOutput [pos + total_offset + 7] = m * (gcgsb - gcrgs - ggts + gIgtots) ;

                d_CKTloadOutput [pos + total_offset + 8] = m * (gcgbb - gcrgb - ggtb + gIgtotb) ;

                total_offset += 9 ;
            }
            else if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 3)
            {
                d_CKTloadOutput [pos + total_offset + 0] = m * geltd ;

                d_CKTloadOutput [pos + total_offset + 1] = m * (geltd + gcrg + gcgmgmb) ;

                d_CKTloadOutput [pos + total_offset + 2] = m * (gcrgd + gcgmdb) ;

                d_CKTloadOutput [pos + total_offset + 3] = m * gcrgg ;

                d_CKTloadOutput [pos + total_offset + 4] = m * (gcrgs + gcgmsb) ;

                d_CKTloadOutput [pos + total_offset + 5] = m * (gcrgb + gcgmbb) ;

                d_CKTloadOutput [pos + total_offset + 6] = m * gcdgmb ;

                d_CKTloadOutput [pos + total_offset + 7] = m * gcrg ;

                d_CKTloadOutput [pos + total_offset + 8] = m * gcsgmb ;

                d_CKTloadOutput [pos + total_offset + 9] = m * gcbgmb ;

                d_CKTloadOutput [pos + total_offset + 10] = m * (gcggb - gcrgg - ggtg + gIgtotg) ;

                d_CKTloadOutput [pos + total_offset + 11] = m * (gcgdb - gcrgd - ggtd + gIgtotd) ;

                d_CKTloadOutput [pos + total_offset + 12] = m * (gcgsb - gcrgs - ggts + gIgtots) ;

                d_CKTloadOutput [pos + total_offset + 13] = m * (gcgbb - gcrgb - ggtb + gIgtotb) ;

                total_offset += 14 ;
            } else {
                d_CKTloadOutput [pos + total_offset + 0] = m * (gcggb - ggtg + gIgtotg) ;

                d_CKTloadOutput [pos + total_offset + 1] = m * (gcgdb - ggtd + gIgtotd) ;

                d_CKTloadOutput [pos + total_offset + 2] = m * (gcgsb - ggts + gIgtots) ;

                d_CKTloadOutput [pos + total_offset + 3] = m * (gcgbb - ggtb + gIgtotb) ;

                total_offset += 4 ;
            }

            /* 27 - non-divergent */
            if (BSIM4v7rdsMod)
            {
                d_CKTloadOutput [pos + total_offset + 0] = m * gdtotg ;

                d_CKTloadOutput [pos + total_offset + 1] = m * gdtots ;

                d_CKTloadOutput [pos + total_offset + 2] = m * gdtotb ;

                d_CKTloadOutput [pos + total_offset + 3] = m * gstotd ;

                total_offset += 4 ;
            }

            ggidld = BSIM4v7entry.d_BSIM4v7ggidldArray [instance_ID] ;
            ggidlg = BSIM4v7entry.d_BSIM4v7ggidlgArray [instance_ID] ;
            ggidlb = BSIM4v7entry.d_BSIM4v7ggidlbArray [instance_ID] ;
            ggislg = BSIM4v7entry.d_BSIM4v7ggislgArray [instance_ID] ;
            ggisls = BSIM4v7entry.d_BSIM4v7ggislsArray [instance_ID] ;
            ggislb = BSIM4v7entry.d_BSIM4v7ggislbArray [instance_ID] ;

            d_CKTloadOutput [pos + total_offset + 0] = m * (gdpr + BSIM4v7entry.d_BSIM4v7gdsRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gbdRWArray [instance_ID] + T1 * ddxpart_dVd - gdtotd + RevSum + gcddb + gbdpdp + dxpart * ggtd - gIdtotd) + m * ggidld ;

            d_CKTloadOutput [pos + total_offset + 1] = m * (gdpr + gdtot) ;

            d_CKTloadOutput [pos + total_offset + 2] = m * (Gm + gcdgb - gdtotg + gbdpg - gIdtotg + dxpart * ggtg + T1 * ddxpart_dVg) + m * ggidlg ;

            d_CKTloadOutput [pos + total_offset + 3] = m * (BSIM4v7entry.d_BSIM4v7gdsRWArray [instance_ID] + gdtots - dxpart * ggts + gIdtots - T1 * ddxpart_dVs + FwdSum - gcdsb - gbdpsp) + m * (ggidlg + ggidld + ggidlb) ;

            d_CKTloadOutput [pos + total_offset + 4] = m * (gjbd + gdtotb - Gmbs - gcdbb - gbdpb + gIdtotb - T1 * ddxpart_dVb - dxpart * ggtb) - m * ggidlb ;

            d_CKTloadOutput [pos + total_offset + 5] = m * (gdpr - gdtotd) ;

            d_CKTloadOutput [pos + total_offset + 6] = m * (BSIM4v7entry.d_BSIM4v7gdsRWArray [instance_ID] + gstotd + RevSum - gcsdb - gbspdp - T1 * dsxpart_dVd - sxpart * ggtd + gIstotd) + m * (ggisls + ggislg + ggislb) ;

            d_CKTloadOutput [pos + total_offset + 7] = m * (gcsgb - Gm - gstotg + gbspg + sxpart * ggtg + T1 * dsxpart_dVg - gIstotg) + m * ggislg ;

            d_CKTloadOutput [pos + total_offset + 8] = m * (gspr + BSIM4v7entry.d_BSIM4v7gdsRWArray [instance_ID] + BSIM4v7entry.d_BSIM4v7gbsRWArray [instance_ID] + T1 * dsxpart_dVs - gstots + FwdSum + gcssb + gbspsp + sxpart * ggts - gIstots) + m * ggisls ;

            d_CKTloadOutput [pos + total_offset + 9] = m * (gspr + gstot) ;

            d_CKTloadOutput [pos + total_offset + 10] = m * (gjbs + gstotb + Gmbs - gcsbb - gbspb - sxpart * ggtb - T1 * dsxpart_dVb + gIstotb) - m * ggislb ;

            d_CKTloadOutput [pos + total_offset + 11] = m * (gspr - gstots) ;

            d_CKTloadOutput [pos + total_offset + 12] = m * (gcbdb - gjbd + gbbdp - gIbtotd) - m * ggidld + m * (ggislg + ggisls + ggislb) ;

            d_CKTloadOutput [pos + total_offset + 13] = m * (gcbgb - BSIM4v7entry.d_BSIM4v7gbgsArray [instance_ID] - gIbtotg) - m * ggidlg - m * ggislg ;

            d_CKTloadOutput [pos + total_offset + 14] = m * (gcbsb - gjbs + gbbsp - gIbtots) + m * (ggidlg + ggidld + ggidlb) - m * ggisls ;

            d_CKTloadOutput [pos + total_offset + 15] = m * (gjbd + gjbs + gcbbb - BSIM4v7entry.d_BSIM4v7gbbsArray [instance_ID] - gIbtotb) - m * ggidlb - m * ggislb ;

            total_offset += 16 ;

            /* stamp gidl included above */
            /* stamp gisl included above */

            /* 94 - DIVERGENT */
            if (BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID])
            {
                d_CKTloadOutput [pos + total_offset + 0] = m * (gcdbdb - BSIM4v7entry.d_BSIM4v7gbdRWArray [instance_ID]) ;

                d_CKTloadOutput [pos + total_offset + 1] = m * (BSIM4v7entry.d_BSIM4v7gbsRWArray [instance_ID] - gcsbsb) ;


                d_CKTloadOutput [pos + total_offset + 2] = m * (BSIM4v7entry.d_BSIM4v7gbdRWArray [instance_ID] - gcdbdb + BSIM4v7entry.d_BSIM4v7grbpdArray [instance_ID] + BSIM4v7entry.d_BSIM4v7grbdbArray [instance_ID]) ;

                d_CKTloadOutput [pos + total_offset + 3] = m * BSIM4v7entry.d_BSIM4v7grbpdArray [instance_ID] ;

                d_CKTloadOutput [pos + total_offset + 4] = m * BSIM4v7entry.d_BSIM4v7grbdbArray [instance_ID] ;

                d_CKTloadOutput [pos + total_offset + 5] = m * BSIM4v7entry.d_BSIM4v7grbpbArray [instance_ID] ;

                d_CKTloadOutput [pos + total_offset + 6] = m * BSIM4v7entry.d_BSIM4v7grbpsArray [instance_ID] ;

                d_CKTloadOutput [pos + total_offset + 7] = m * (BSIM4v7entry.d_BSIM4v7grbpdArray [instance_ID] + BSIM4v7entry.d_BSIM4v7grbpsArray [instance_ID]  + BSIM4v7entry.d_BSIM4v7grbpbArray [instance_ID]) ;

                d_CKTloadOutput [pos + total_offset + 8] = m * (gcsbsb - BSIM4v7entry.d_BSIM4v7gbsRWArray [instance_ID]) ;

                d_CKTloadOutput [pos + total_offset + 9] = m * BSIM4v7entry.d_BSIM4v7grbsbArray [instance_ID] ;

                d_CKTloadOutput [pos + total_offset + 10] = m * (BSIM4v7entry.d_BSIM4v7gbsRWArray [instance_ID] - gcsbsb + BSIM4v7entry.d_BSIM4v7grbpsArray [instance_ID] + BSIM4v7entry.d_BSIM4v7grbsbArray [instance_ID]) ;

                d_CKTloadOutput [pos + total_offset + 11] = m * (BSIM4v7entry.d_BSIM4v7grbsbArray [instance_ID] + BSIM4v7entry.d_BSIM4v7grbdbArray [instance_ID] + BSIM4v7entry.d_BSIM4v7grbpbArray [instance_ID]) ;

                total_offset += 12 ;
            }

            /* 94 - DIVERGENT */
            if (BSIM4v7entry.d_BSIM4v7trnqsModArray [instance_ID])
            {
                d_CKTloadOutput [pos + total_offset + 0] = m * (gqdef + BSIM4v7entry.d_BSIM4v7gtauRWArray [instance_ID]) ;

                d_CKTloadOutput [pos + total_offset + 1] = m * (ggtg - gcqgb) ;

                d_CKTloadOutput [pos + total_offset + 2] = m * (ggtd - gcqdb) ;

                d_CKTloadOutput [pos + total_offset + 3] = m * (ggts - gcqsb) ;

                d_CKTloadOutput [pos + total_offset + 4] = m * (ggtb - gcqbb) ;

                d_CKTloadOutput [pos + total_offset + 5] = m * dxpart * BSIM4v7entry.d_BSIM4v7gtauRWArray [instance_ID] ;

                d_CKTloadOutput [pos + total_offset + 6] = m * sxpart * BSIM4v7entry.d_BSIM4v7gtauRWArray [instance_ID] ;

                d_CKTloadOutput [pos + total_offset + 7] = m * BSIM4v7entry.d_BSIM4v7gtauRWArray [instance_ID] ;
            }

line1000:  ;

        }
    }

    return ;
}
