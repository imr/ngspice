/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "mos1defs.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS1load(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current value into the
         * sparse matrix previously provided
         */
{
    MOS1model *model = (MOS1model *) inModel;
    MOS1instance *here;
    double Beta;
    double DrainSatCur;
    double EffectiveLength;
    double GateBulkOverlapCap;
    double GateDrainOverlapCap;
    double GateSourceOverlapCap;
    double OxideCap;
    double SourceSatCur;
    double cbhat;
    double cdhat;
    double cdrain;
    double cdreq;
    double ceq;
    double ceqbd;
    double ceqbs;
    double ceqgb;
    double ceqgd;
    double ceqgs;
    double delvbd;
    double delvbs;
    double delvds;
    double delvgd;
    double delvgs;
    double evbd;
    double evbs;
    double gcgb;
    double gcgd;
    double gcgs;
    double geq;
    double sargsw;
    double vbd;
    double vbs;
    double vds;
    double vdsat;
    double vgb1;
    double vgb;
    double vgd1;
    double vgd;
    double vgdo;
    double vgs1;
    double vgs;
    double von;
    double vt;
#ifndef PREDICTOR
    double xfact = 0.0;
#endif
    int xnrm;
    int xrev;
    double capgs = 0.0;   /* total gate-source capacitance */
    double capgd = 0.0;   /* total gate-drain capacitance */
    double capgb = 0.0;   /* total gate-bulk capacitance */
    int Check;
#ifndef NOBYPASS
    double tempv;
#endif /*NOBYPASS*/
    int error;
 #ifdef CAPBYPASS
    int senflag;
#endif /*CAPBYPASS*/
    int SenCond;



#ifdef CAPBYPASS
    senflag = 0;
    if(ckt->CKTsenInfo && ckt->CKTsenInfo->SENstatus == PERTURBATION &&
        (ckt->CKTsenInfo->SENmode & (ACSEN | TRANSEN))) {
        senflag = 1;
    }
#endif /* CAPBYPASS */

    /*  loop through all the MOS1 device models */
    for( ; model != NULL; model = MOS1nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = MOS1instances(model); here != NULL ;
         here=MOS1nextInstance(here)) {

            vt = CONSTKoverQ * here->MOS1temp;
            Check=1;
            if(ckt->CKTsenInfo){
#ifdef SENSDEBUG
                printf("MOS1load \n");
#endif /* SENSDEBUG */

                if((ckt->CKTsenInfo->SENstatus == PERTURBATION)&&
                        (here->MOS1senPertFlag == OFF))continue;

            }
            SenCond = ckt->CKTsenInfo && here->MOS1senPertFlag;

/*

*/

            /* first, we compute a few useful values - these could be
             * pre-computed, but for historical reasons are still done
             * here.  They may be moved at the expense of instance size
             */

            EffectiveLength=here->MOS1l - 2*model->MOS1latDiff;

            if( (here->MOS1tSatCurDens == 0) ||
                    (here->MOS1drainArea == 0) ||
                    (here->MOS1sourceArea == 0)) {
                DrainSatCur = here->MOS1m * here->MOS1tSatCur;
                SourceSatCur = here->MOS1m * here->MOS1tSatCur;
            } else {
                DrainSatCur = here->MOS1tSatCurDens *
                        here->MOS1m * here->MOS1drainArea;
                SourceSatCur = here->MOS1tSatCurDens *
                        here->MOS1m * here->MOS1sourceArea;
            }
            GateSourceOverlapCap = model->MOS1gateSourceOverlapCapFactor *
                    here->MOS1m * here->MOS1w;
            GateDrainOverlapCap = model->MOS1gateDrainOverlapCapFactor *
                    here->MOS1m * here->MOS1w;
            GateBulkOverlapCap = model->MOS1gateBulkOverlapCapFactor *
                    here->MOS1m * EffectiveLength;
            Beta = here->MOS1tTransconductance * here->MOS1m *
                    here->MOS1w/EffectiveLength;
            OxideCap = model->MOS1oxideCapFactor * EffectiveLength *
                    here->MOS1m * here->MOS1w;

            /*
             * ok - now to do the start-up operations
             *
             * we must get values for vbs, vds, and vgs from somewhere
             * so we either predict them or recover them from last iteration
             * These are the two most common cases - either a prediction
             * step or the general iteration step and they
             * share some code, so we put them first - others later on
             */

            if(SenCond){
#ifdef SENSDEBUG
                printf("MOS1senPertFlag = ON \n");
#endif /* SENSDEBUG */
                if((ckt->CKTsenInfo->SENmode == TRANSEN) &&
           (ckt->CKTmode & MODEINITTRAN)) {
                    vgs = *(ckt->CKTstate1 + here->MOS1vgs);
                    vds = *(ckt->CKTstate1 + here->MOS1vds);
                    vbs = *(ckt->CKTstate1 + here->MOS1vbs);
                    vbd = *(ckt->CKTstate1 + here->MOS1vbd);
                    vgb = vgs - vbs;
                    vgd = vgs - vds;
                }
                else if (ckt->CKTsenInfo->SENmode == ACSEN){
                    vgb = model->MOS1type * (
                        *(ckt->CKTrhsOp+here->MOS1gNode) -
                        *(ckt->CKTrhsOp+here->MOS1bNode));
                    vbs = *(ckt->CKTstate0 + here->MOS1vbs);
                    vbd = *(ckt->CKTstate0 + here->MOS1vbd);
                    vgd = vgb + vbd ;
                    vgs = vgb + vbs ;
                    vds = vbs - vbd ;
                }
                else{
                    vgs = *(ckt->CKTstate0 + here->MOS1vgs);
                    vds = *(ckt->CKTstate0 + here->MOS1vds);
                    vbs = *(ckt->CKTstate0 + here->MOS1vbs);
                    vbd = *(ckt->CKTstate0 + here->MOS1vbd);
                    vgb = vgs - vbs;
                    vgd = vgs - vds;
                }
#ifdef SENSDEBUG
                printf(" vbs = %.7e ,vbd = %.7e,vgb = %.7e\n",vbs,vbd,vgb);
                printf(" vgs = %.7e ,vds = %.7e,vgd = %.7e\n",vgs,vds,vgd);
#endif /* SENSDEBUG */
                goto next1;
            }


            if((ckt->CKTmode & (MODEINITFLOAT | MODEINITPRED | MODEINITSMSIG
                    | MODEINITTRAN)) ||
                    ( (ckt->CKTmode & MODEINITFIX) && (!here->MOS1off) )  ) {
#ifndef PREDICTOR
                if(ckt->CKTmode & (MODEINITPRED | MODEINITTRAN) ) {

                    /* predictor step */

                    xfact=ckt->CKTdelta/ckt->CKTdeltaOld[1];
                    *(ckt->CKTstate0 + here->MOS1vbs) =
                    *(ckt->CKTstate1 + here->MOS1vbs);
                    vbs = (1+xfact)* (*(ckt->CKTstate1 + here->MOS1vbs))
                            -(xfact * (*(ckt->CKTstate2 + here->MOS1vbs)));
                    *(ckt->CKTstate0 + here->MOS1vgs) =
                            *(ckt->CKTstate1 + here->MOS1vgs);
                    vgs = (1+xfact)* (*(ckt->CKTstate1 + here->MOS1vgs))
                            -(xfact * (*(ckt->CKTstate2 + here->MOS1vgs)));
                    *(ckt->CKTstate0 + here->MOS1vds) =
                            *(ckt->CKTstate1 + here->MOS1vds);
                    vds = (1+xfact)* (*(ckt->CKTstate1 + here->MOS1vds))
                            -(xfact * (*(ckt->CKTstate2 + here->MOS1vds)));
                    *(ckt->CKTstate0 + here->MOS1vbd) =
                            *(ckt->CKTstate0 + here->MOS1vbs)-
                    *(ckt->CKTstate0 + here->MOS1vds);
                } else {
#endif /* PREDICTOR */

                    /* general iteration */

                    vbs = model->MOS1type * (
                            *(ckt->CKTrhsOld+here->MOS1bNode) -
                            *(ckt->CKTrhsOld+here->MOS1sNodePrime));
                    vgs = model->MOS1type * (
                        *(ckt->CKTrhsOld+here->MOS1gNode) -
                        *(ckt->CKTrhsOld+here->MOS1sNodePrime));
                    vds = model->MOS1type * (
                        *(ckt->CKTrhsOld+here->MOS1dNodePrime) -
                        *(ckt->CKTrhsOld+here->MOS1sNodePrime));
#ifndef PREDICTOR
                }
#endif /* PREDICTOR */

                /* now some common crunching for some more useful quantities */

                vbd=vbs-vds;
                vgd=vgs-vds;
                vgdo = *(ckt->CKTstate0 + here->MOS1vgs) -
                        *(ckt->CKTstate0 + here->MOS1vds);
                delvbs = vbs - *(ckt->CKTstate0 + here->MOS1vbs);
                delvbd = vbd - *(ckt->CKTstate0 + here->MOS1vbd);
                delvgs = vgs - *(ckt->CKTstate0 + here->MOS1vgs);
                delvds = vds - *(ckt->CKTstate0 + here->MOS1vds);
                delvgd = vgd-vgdo;

                /* these are needed for convergence testing */

                if (here->MOS1mode >= 0) {
                    cdhat=
                            here->MOS1cd-
                            here->MOS1gbd * delvbd +
                            here->MOS1gmbs * delvbs +
                            here->MOS1gm * delvgs +
                            here->MOS1gds * delvds ;
                } else {
                    cdhat=
                            here->MOS1cd -
                            ( here->MOS1gbd -
                            here->MOS1gmbs) * delvbd -
                            here->MOS1gm * delvgd +
                        here->MOS1gds * delvds ;
                }
                cbhat=
                        here->MOS1cbs +
                        here->MOS1cbd +
                        here->MOS1gbd * delvbd +
                        here->MOS1gbs * delvbs ;
/*

*/

#ifndef NOBYPASS
                /* now lets see if we can bypass (ugh) */
                tempv = (MAX(fabs(cbhat),
                fabs(here->MOS1cbs + here->MOS1cbd)) +
                        ckt->CKTabstol);
                if ((!(ckt->CKTmode &
                        (MODEINITPRED|MODEINITTRAN|MODEINITSMSIG))) &&
                        (ckt->CKTbypass) &&
                        (fabs(cbhat-(here->MOS1cbs +
                            here->MOS1cbd)) < ckt->CKTreltol * tempv) &&
                        (fabs(delvbs) < (ckt->CKTreltol *
                            MAX(fabs(vbs),
                            fabs( *(ckt->CKTstate0 +
                            here->MOS1vbs))) +
                            ckt->CKTvoltTol)) &&
                        (fabs(delvbd) < (ckt->CKTreltol *
                                 MAX(fabs(vbd),
                                 fabs(*(ckt->CKTstate0 +
                                    here->MOS1vbd))) +
                                 ckt->CKTvoltTol)) &&
                        (fabs(delvgs) < (ckt->CKTreltol *
                                 MAX(fabs(vgs),
                                 fabs(*(ckt->CKTstate0 +
                                    here->MOS1vgs)))+
                                 ckt->CKTvoltTol)) &&
                        (fabs(delvds) < (ckt->CKTreltol *
                                 MAX(fabs(vds),
                                 fabs(*(ckt->CKTstate0 +
                                    here->MOS1vds))) +
                                 ckt->CKTvoltTol)) &&
                        (fabs(cdhat- here->MOS1cd) < (ckt->CKTreltol *
                                      MAX(fabs(cdhat),
                                          fabs(here->MOS1cd)) +
                                      ckt->CKTabstol))) {
                    /* bypass code */
                    /* nothing interesting has changed since last
                    * iteration on this device, so we just
                    * copy all the values computed last iteration out
                    * and keep going
                    */
                    vbs = *(ckt->CKTstate0 + here->MOS1vbs);
                    vbd = *(ckt->CKTstate0 + here->MOS1vbd);
                    vgs = *(ckt->CKTstate0 + here->MOS1vgs);
                    vds = *(ckt->CKTstate0 + here->MOS1vds);
                    vgd = vgs - vds;
                    vgb = vgs - vbs;
                    cdrain = here->MOS1mode * (here->MOS1cd + here->MOS1cbd);
                    if(ckt->CKTmode & (MODETRAN | MODETRANOP)) {
                        capgs = ( *(ckt->CKTstate0+here->MOS1capgs)+
                              *(ckt->CKTstate1+here->MOS1capgs) +
                              GateSourceOverlapCap );
                        capgd = ( *(ckt->CKTstate0+here->MOS1capgd)+
                              *(ckt->CKTstate1+here->MOS1capgd) +
                              GateDrainOverlapCap );
                        capgb = ( *(ckt->CKTstate0+here->MOS1capgb)+
                              *(ckt->CKTstate1+here->MOS1capgb) +
                              GateBulkOverlapCap );

                        if(ckt->CKTsenInfo){
                          here->MOS1cgs = capgs;
                          here->MOS1cgd = capgd;
                          here->MOS1cgb = capgb;
                        }
                    }
                    goto bypass;
                }
#endif /*NOBYPASS*/

/*

*/

                /* ok - bypass is out, do it the hard way */

                von = model->MOS1type * here->MOS1von;

#ifndef NODELIMITING
                /*
                 * limiting
                 *  we want to keep device voltages from changing
                 * so fast that the exponentials churn out overflows
                 * and similar rudeness
                 */

                if(*(ckt->CKTstate0 + here->MOS1vds) >=0) {
                    vgs = DEVfetlim(vgs,*(ckt->CKTstate0 + here->MOS1vgs)
                    ,von);
                    vds = vgs - vgd;
                    vds = DEVlimvds(vds,*(ckt->CKTstate0 + here->MOS1vds));
                    vgd = vgs - vds;
                } else {
                    vgd = DEVfetlim(vgd,vgdo,von);
                    vds = vgs - vgd;
                    if(!(ckt->CKTfixLimit)) {
                        vds = -DEVlimvds(-vds,-(*(ckt->CKTstate0 +
                          here->MOS1vds)));
                    }
                    vgs = vgd + vds;
                }
                if(vds >= 0) {
                    vbs = DEVpnjlim(vbs,*(ckt->CKTstate0 + here->MOS1vbs),
                    vt,here->MOS1sourceVcrit,&Check);
                    vbd = vbs-vds;
                } else {
                    vbd = DEVpnjlim(vbd,*(ckt->CKTstate0 + here->MOS1vbd),
                    vt,here->MOS1drainVcrit,&Check);
                    vbs = vbd + vds;
                }
#endif /*NODELIMITING*/
/*

*/

            } else {

                /* ok - not one of the simple cases, so we have to
                 * look at all of the possibilities for why we were
                 * called.  We still just initialize the three voltages
                 */

                if((ckt->CKTmode & MODEINITJCT) && !here->MOS1off) {
                    vds= model->MOS1type * here->MOS1icVDS;
                    vgs= model->MOS1type * here->MOS1icVGS;
                    vbs= model->MOS1type * here->MOS1icVBS;
                    if((vds==0) && (vgs==0) && (vbs==0) &&
                            ((ckt->CKTmode &
                            (MODETRAN|MODEDCOP|MODEDCTRANCURVE)) ||
                            (!(ckt->CKTmode & MODEUIC)))) {
                        vbs = -1;
                        vgs = model->MOS1type * here->MOS1tVto;
                        vds = 0;
                    }
                } else {
                    vbs=vgs=vds=0;
                }
            }
/*

*/

            /*
             * now all the preliminaries are over - we can start doing the
             * real work
             */
            vbd = vbs - vds;
            vgd = vgs - vds;
            vgb = vgs - vbs;


            /*
             * bulk-source and bulk-drain diodes
             *   here we just evaluate the ideal diode current and the
             *   corresponding derivative (conductance).
             */
next1:      if(vbs <= -3*vt) {
                here->MOS1gbs = ckt->CKTgmin;
                here->MOS1cbs = here->MOS1gbs*vbs-SourceSatCur;
            } else {
                evbs = exp(MIN(MAX_EXP_ARG,vbs/vt));
                here->MOS1gbs = SourceSatCur*evbs/vt + ckt->CKTgmin;
                here->MOS1cbs = SourceSatCur*(evbs-1) + ckt->CKTgmin*vbs;
            }
            if(vbd <= -3*vt) {
                here->MOS1gbd = ckt->CKTgmin;
                here->MOS1cbd = here->MOS1gbd*vbd-DrainSatCur;
            } else {
                evbd = exp(MIN(MAX_EXP_ARG,vbd/vt));
                here->MOS1gbd = DrainSatCur*evbd/vt + ckt->CKTgmin;
                here->MOS1cbd = DrainSatCur*(evbd-1) + ckt->CKTgmin*vbd;
            }
            /* now to determine whether the user was able to correctly
             * identify the source and drain of his device
             */
            if(vds >= 0) {
                /* normal mode */
                here->MOS1mode = 1;
            } else {
                /* inverse mode */
                here->MOS1mode = -1;
            }
/*

*/

            {
                /*
                 *     this block of code evaluates the drain current and its
                 *     derivatives using the shichman-hodges model and the
                 *     charges associated with the gate, channel and bulk for
                 *     mosfets
                 *
                 */

                /* the following 4 variables are local to this code block until
                 * it is obvious that they can be made global
                 */
                double arg;
                double betap;
                double sarg;
                double vgst;

                if ((here->MOS1mode==1?vbs:vbd) <= 0 ) {
                    sarg=sqrt(here->MOS1tPhi-(here->MOS1mode==1?vbs:vbd));
                } else {
                    sarg=sqrt(here->MOS1tPhi);
                    sarg=sarg-(here->MOS1mode==1?vbs:vbd)/(sarg+sarg);
                    sarg=MAX(0,sarg);
                }
                von=(here->MOS1tVbi*model->MOS1type)+model->MOS1gamma*sarg;
                vgst=(here->MOS1mode==1?vgs:vgd)-von;
                vdsat=MAX(vgst,0);
                if (sarg <= 0) {
                    arg=0;
                } else {
                    arg=model->MOS1gamma/(sarg+sarg);
                }
                if (vgst <= 0) {
                    /*
                     *     cutoff region
                     */
                    cdrain=0;
                    here->MOS1gm=0;
                    here->MOS1gds=0;
                    here->MOS1gmbs=0;
                } else{
                    /*
                     *     saturation region
                     */
                    betap=Beta*(1+model->MOS1lambda*(vds*here->MOS1mode));
                    if (vgst <= (vds*here->MOS1mode)){
                        cdrain=betap*vgst*vgst*.5;
                        here->MOS1gm=betap*vgst;
                        here->MOS1gds=model->MOS1lambda*Beta*vgst*vgst*.5;
                        here->MOS1gmbs=here->MOS1gm*arg;
                    } else {
                        /*
                         *     linear region
                         */
                        cdrain=betap*(vds*here->MOS1mode)*
                            (vgst-.5*(vds*here->MOS1mode));
                        here->MOS1gm=betap*(vds*here->MOS1mode);
                        here->MOS1gds=betap*(vgst-(vds*here->MOS1mode))+
                                model->MOS1lambda*Beta*
                                (vds*here->MOS1mode)*
                                (vgst-.5*(vds*here->MOS1mode));
                                        here->MOS1gmbs=here->MOS1gm*arg;
                    }
                }
                /*
                 *     finished
                 */
            }
/*

*/

            /* now deal with n vs p polarity */

            here->MOS1von = model->MOS1type * von;
            here->MOS1vdsat = model->MOS1type * vdsat;
            /* line 490 */
            /*
             *  COMPUTE EQUIVALENT DRAIN CURRENT SOURCE
             */
            here->MOS1cd=here->MOS1mode * cdrain - here->MOS1cbd;

            if (ckt->CKTmode & (MODETRAN | MODETRANOP | MODEINITSMSIG)) {
                /*
                 * now we do the hard part of the bulk-drain and bulk-source
                 * diode - we evaluate the non-linear capacitance and
                 * charge
                 *
                 * the basic equations are not hard, but the implementation
                 * is somewhat long in an attempt to avoid log/exponential
                 * evaluations
                 */
                /*
                 *  charge storage elements
                 *
                 *.. bulk-drain and bulk-source depletion capacitances
                 */
#ifdef CAPBYPASS
                if(((ckt->CKTmode & (MODEINITPRED | MODEINITTRAN) ) ||
                        fabs(delvbs) >= ckt->CKTreltol * MAX(fabs(vbs),
                        fabs(*(ckt->CKTstate0+here->MOS1vbs)))+
                        ckt->CKTvoltTol)|| senflag)
#endif /*CAPBYPASS*/

                {
                    /* can't bypass the diode capacitance calculations */
                    if(here->MOS1Cbs != 0 || here->MOS1Cbssw != 0 ) {
                        if (vbs < here->MOS1tDepCap){
                            const double arg=1-vbs/here->MOS1tBulkPot;
                            double sarg;
                            /*
                             * the following block looks somewhat long and messy,
                             * but since most users use the default grading
                             * coefficients of .5, and sqrt is MUCH faster than an
                             * exp(log()) we use this special case code to buy time.
                             * (as much as 10% of total job time!)
                             */
                            if(model->MOS1bulkJctBotGradingCoeff ==
                                    model->MOS1bulkJctSideGradingCoeff) {
                                if(model->MOS1bulkJctBotGradingCoeff == .5) {
                                    sarg = sargsw = 1/sqrt(arg);
                                } else {
                                    sarg = sargsw =
                                            exp(-model->MOS1bulkJctBotGradingCoeff*
                                            log(arg));
                                }
                            } else {
                                if(model->MOS1bulkJctBotGradingCoeff == .5) {
                                    sarg = 1/sqrt(arg);
                                } else {
                                    sarg = exp(-model->MOS1bulkJctBotGradingCoeff*
                                           log(arg));
                                }
                                if(model->MOS1bulkJctSideGradingCoeff == .5) {
                                    sargsw = 1/sqrt(arg);
                                } else {
                                    sargsw =exp(-model->MOS1bulkJctSideGradingCoeff*
                                            log(arg));
                                }
                            }
                            *(ckt->CKTstate0 + here->MOS1qbs) =
                            here->MOS1tBulkPot*(here->MOS1Cbs*
                                    (1-arg*sarg)/(1-model->MOS1bulkJctBotGradingCoeff)
                                    +here->MOS1Cbssw*
                                    (1-arg*sargsw)/
                                    (1-model->MOS1bulkJctSideGradingCoeff));
                            here->MOS1capbs=here->MOS1Cbs*sarg+
                                    here->MOS1Cbssw*sargsw;
                        } else {
                            *(ckt->CKTstate0 + here->MOS1qbs) = here->MOS1f4s +
                                            vbs*(here->MOS1f2s+vbs*(here->MOS1f3s/2));
                            here->MOS1capbs=here->MOS1f2s+here->MOS1f3s*vbs;
                        }
                    } else {
                        *(ckt->CKTstate0 + here->MOS1qbs) = 0;
                        here->MOS1capbs=0;
                    }
                }
#ifdef CAPBYPASS
                    if(((ckt->CKTmode & (MODEINITPRED | MODEINITTRAN) ) ||
                            fabs(delvbd) >= ckt->CKTreltol * MAX(fabs(vbd),
                            fabs(*(ckt->CKTstate0+here->MOS1vbd)))+
                            ckt->CKTvoltTol)|| senflag)
#endif /*CAPBYPASS*/

                    /* can't bypass the diode capacitance calculations */
                {
                    if(here->MOS1Cbd != 0 || here->MOS1Cbdsw != 0 ) {
                        if (vbd < here->MOS1tDepCap) {
                            const double arg=1-vbd/here->MOS1tBulkPot;
                            double sarg;
                            /*
                             * the following block looks somewhat long and messy,
                             * but since most users use the default grading
                             * coefficients of .5, and sqrt is MUCH faster than an
                             * exp(log()) we use this special case code to buy time.
                             * (as much as 10% of total job time!)
                             */
                            if(model->MOS1bulkJctBotGradingCoeff == .5 &&
                                    model->MOS1bulkJctSideGradingCoeff == .5) {
                                sarg = sargsw = 1/sqrt(arg);
                            } else {
                                if(model->MOS1bulkJctBotGradingCoeff == .5) {
                                    sarg = 1/sqrt(arg);
                                } else {
                                    sarg = exp(-model->MOS1bulkJctBotGradingCoeff*
                                           log(arg));
                                }
                                if(model->MOS1bulkJctSideGradingCoeff == .5) {
                                    sargsw = 1/sqrt(arg);
                                } else {
                                    sargsw =exp(-model->MOS1bulkJctSideGradingCoeff*
                                            log(arg));
                                }
                            }
                            *(ckt->CKTstate0 + here->MOS1qbd) =
                                    here->MOS1tBulkPot*(here->MOS1Cbd*
                                    (1-arg*sarg)
                                    /(1-model->MOS1bulkJctBotGradingCoeff)
                                    +here->MOS1Cbdsw*
                                    (1-arg*sargsw)
                                    /(1-model->MOS1bulkJctSideGradingCoeff));
                            here->MOS1capbd=here->MOS1Cbd*sarg+
                                    here->MOS1Cbdsw*sargsw;
                        } else {
                            *(ckt->CKTstate0 + here->MOS1qbd) = here->MOS1f4d +
                                    vbd * (here->MOS1f2d + vbd * here->MOS1f3d/2);
                            here->MOS1capbd=here->MOS1f2d + vbd * here->MOS1f3d;
                        }
                    } else {
                        *(ckt->CKTstate0 + here->MOS1qbd) = 0;
                        here->MOS1capbd = 0;
                    }
                }
/*

*/

                if(SenCond && (ckt->CKTsenInfo->SENmode==TRANSEN)) goto next2;

                if ( (ckt->CKTmode & MODETRAN) || ( (ckt->CKTmode&MODEINITTRAN)
                            && !(ckt->CKTmode&MODEUIC)) ) {
                    /* (above only excludes tranop, since we're only at this
                     * point if tran or tranop )
                     */

                    /*
                     *    calculate equivalent conductances and currents for
                     *    depletion capacitors
                     */

                    /* integrate the capacitors and save results */

                    error = NIintegrate(ckt,&geq,&ceq,here->MOS1capbd,
                    here->MOS1qbd);
                    if(error) return(error);
                    here->MOS1gbd += geq;
                    here->MOS1cbd += *(ckt->CKTstate0 + here->MOS1cqbd);
                    here->MOS1cd -= *(ckt->CKTstate0 + here->MOS1cqbd);
                    error = NIintegrate(ckt,&geq,&ceq,here->MOS1capbs,
                    here->MOS1qbs);
                    if(error) return(error);
                    here->MOS1gbs += geq;
                    here->MOS1cbs += *(ckt->CKTstate0 + here->MOS1cqbs);
                }
            }
/*

*/

            if(SenCond) goto next2;


            /*
             *  check convergence
             */
            if ( (here->MOS1off == 0)  ||
                    (!(ckt->CKTmode & (MODEINITFIX|MODEINITSMSIG))) ){
                if (Check == 1) {
                    ckt->CKTnoncon++;
                    ckt->CKTtroubleElt = (GENinstance *) here;
                }
            }
/*

*/

            /* save things away for next time */

next2:
            *(ckt->CKTstate0 + here->MOS1vbs) = vbs;
            *(ckt->CKTstate0 + here->MOS1vbd) = vbd;
            *(ckt->CKTstate0 + here->MOS1vgs) = vgs;
            *(ckt->CKTstate0 + here->MOS1vds) = vds;

/*

*/

            /*
             *     meyer's capacitor model
             */
            if ( ckt->CKTmode & (MODETRAN | MODETRANOP | MODEINITSMSIG) ) {
                /*
                 *     calculate meyer's capacitors
                 */
                /*
                 * new cmeyer - this just evaluates at the current time,
                 * expects you to remember values from previous time
                 * returns 1/2 of non-constant portion of capacitance
                 * you must add in the other half from previous time
                 * and the constant part
                 */
                if (here->MOS1mode > 0){
                    DEVqmeyer (vgs,vgd,vgb,von,vdsat,
                   (ckt->CKTstate0 + here->MOS1capgs),
                   (ckt->CKTstate0 + here->MOS1capgd),
                   (ckt->CKTstate0 + here->MOS1capgb),
                   here->MOS1tPhi,OxideCap);
                } else {
                    DEVqmeyer (vgd,vgs,vgb,von,vdsat,
                   (ckt->CKTstate0 + here->MOS1capgd),
                   (ckt->CKTstate0 + here->MOS1capgs),
                   (ckt->CKTstate0 + here->MOS1capgb),
                   here->MOS1tPhi,OxideCap);
                }
                vgs1 = *(ckt->CKTstate1 + here->MOS1vgs);
                vgd1 = vgs1 - *(ckt->CKTstate1 + here->MOS1vds);
                vgb1 = vgs1 - *(ckt->CKTstate1 + here->MOS1vbs);
                if(ckt->CKTmode & (MODETRANOP|MODEINITSMSIG)) {
                    capgs =  2 * *(ckt->CKTstate0+here->MOS1capgs)+
                            GateSourceOverlapCap ;
                    capgd =  2 * *(ckt->CKTstate0+here->MOS1capgd)+
                            GateDrainOverlapCap ;
                    capgb =  2 * *(ckt->CKTstate0+here->MOS1capgb)+
                            GateBulkOverlapCap ;
                } else {
                    capgs = ( *(ckt->CKTstate0+here->MOS1capgs)+
                            *(ckt->CKTstate1+here->MOS1capgs) +
                            GateSourceOverlapCap );
                    capgd = ( *(ckt->CKTstate0+here->MOS1capgd)+
                            *(ckt->CKTstate1+here->MOS1capgd) +
                            GateDrainOverlapCap );
                    capgb = ( *(ckt->CKTstate0+here->MOS1capgb)+
                            *(ckt->CKTstate1+here->MOS1capgb) +
                            GateBulkOverlapCap );
                }
                if(ckt->CKTsenInfo){
                    here->MOS1cgs = capgs;
                    here->MOS1cgd = capgd;
                    here->MOS1cgb = capgb;
                }
/*

*/

                /*
                 *     store small-signal parameters (for meyer's model)
                 *  all parameters already stored, so done...
                 */
                if(SenCond){
                    if((ckt->CKTsenInfo->SENmode == DCSEN)||
                            (ckt->CKTsenInfo->SENmode == ACSEN)){
                        continue;
                    }
                }

#ifndef PREDICTOR
                if (ckt->CKTmode & (MODEINITPRED | MODEINITTRAN) ) {
                    *(ckt->CKTstate0 + here->MOS1qgs) =
                            (1+xfact) * *(ckt->CKTstate1 + here->MOS1qgs)
                            - xfact * *(ckt->CKTstate2 + here->MOS1qgs);
                    *(ckt->CKTstate0 + here->MOS1qgd) =
                            (1+xfact) * *(ckt->CKTstate1 + here->MOS1qgd)
                            - xfact * *(ckt->CKTstate2 + here->MOS1qgd);
                    *(ckt->CKTstate0 + here->MOS1qgb) =
                            (1+xfact) * *(ckt->CKTstate1 + here->MOS1qgb)
                            - xfact * *(ckt->CKTstate2 + here->MOS1qgb);
                } else {
#endif /*PREDICTOR*/
                    if(ckt->CKTmode & MODETRAN) {
                        *(ckt->CKTstate0 + here->MOS1qgs) = (vgs-vgs1)*capgs +
                            *(ckt->CKTstate1 + here->MOS1qgs) ;
                        *(ckt->CKTstate0 + here->MOS1qgd) = (vgd-vgd1)*capgd +
                            *(ckt->CKTstate1 + here->MOS1qgd) ;
                        *(ckt->CKTstate0 + here->MOS1qgb) = (vgb-vgb1)*capgb +
                            *(ckt->CKTstate1 + here->MOS1qgb) ;
                    } else {
                        /* TRANOP only */
                        *(ckt->CKTstate0 + here->MOS1qgs) = vgs*capgs;
                        *(ckt->CKTstate0 + here->MOS1qgd) = vgd*capgd;
                        *(ckt->CKTstate0 + here->MOS1qgb) = vgb*capgb;
                    }
#ifndef PREDICTOR
                }
#endif /*PREDICTOR*/
            }
#ifndef NOBYPASS
    bypass:
#endif
            if(SenCond) continue;

            if ( (ckt->CKTmode & (MODEINITTRAN)) ||
                    (! (ckt->CKTmode & (MODETRAN)) )  ) {
                /*
                 *  initialize to zero charge conductances
                 *  and current
                 */
                gcgs=0;
                ceqgs=0;
                gcgd=0;
                ceqgd=0;
                gcgb=0;
                ceqgb=0;
            } else {
                if(capgs == 0) *(ckt->CKTstate0 + here->MOS1cqgs) =0;
                if(capgd == 0) *(ckt->CKTstate0 + here->MOS1cqgd) =0;
                if(capgb == 0) *(ckt->CKTstate0 + here->MOS1cqgb) =0;
                /*
                 *    calculate equivalent conductances and currents for
                 *    meyer"s capacitors
                 */
                error = NIintegrate(ckt,&gcgs,&ceqgs,capgs,here->MOS1qgs);
                if(error) return(error);
                error = NIintegrate(ckt,&gcgd,&ceqgd,capgd,here->MOS1qgd);
                if(error) return(error);
                error = NIintegrate(ckt,&gcgb,&ceqgb,capgb,here->MOS1qgb);
                if(error) return(error);
                ceqgs=ceqgs-gcgs*vgs+ckt->CKTag[0]*
                        *(ckt->CKTstate0 + here->MOS1qgs);
                ceqgd=ceqgd-gcgd*vgd+ckt->CKTag[0]*
                        *(ckt->CKTstate0 + here->MOS1qgd);
                ceqgb=ceqgb-gcgb*vgb+ckt->CKTag[0]*
                        *(ckt->CKTstate0 + here->MOS1qgb);
            }
            /*
             *     store charge storage info for meyer's cap in lx table
             */

            /*
             *  load current vector
             */
            ceqbs = model->MOS1type *
                    (here->MOS1cbs-(here->MOS1gbs)*vbs);
            ceqbd = model->MOS1type *
                (here->MOS1cbd-(here->MOS1gbd)*vbd);
            if (here->MOS1mode >= 0) {
                xnrm=1;
                xrev=0;
                cdreq=model->MOS1type*(cdrain-here->MOS1gds*vds-
                       here->MOS1gm*vgs-here->MOS1gmbs*vbs);
            } else {
                xnrm=0;
                xrev=1;
                cdreq = -(model->MOS1type)*(cdrain-here->MOS1gds*(-vds)-
                        here->MOS1gm*vgd-here->MOS1gmbs*vbd);
            }
            *(ckt->CKTrhs + here->MOS1gNode) -=
                    (model->MOS1type * (ceqgs + ceqgb + ceqgd));
            *(ckt->CKTrhs + here->MOS1bNode) -=
                    (ceqbs + ceqbd - model->MOS1type * ceqgb);
            *(ckt->CKTrhs + here->MOS1dNodePrime) +=
                    (ceqbd - cdreq + model->MOS1type * ceqgd);
            *(ckt->CKTrhs + here->MOS1sNodePrime) +=
                    cdreq + ceqbs + model->MOS1type * ceqgs;
            /*
             *  load y matrix
             */

            *(here->MOS1DdPtr) += (here->MOS1drainConductance);
            *(here->MOS1GgPtr) += ((gcgd+gcgs+gcgb));
            *(here->MOS1SsPtr) += (here->MOS1sourceConductance);
            *(here->MOS1BbPtr) += (here->MOS1gbd+here->MOS1gbs+gcgb);
            *(here->MOS1DPdpPtr) +=
                    (here->MOS1drainConductance+here->MOS1gds+
                    here->MOS1gbd+xrev*(here->MOS1gm+here->MOS1gmbs)+gcgd);
            *(here->MOS1SPspPtr) +=
                    (here->MOS1sourceConductance+here->MOS1gds+
                    here->MOS1gbs+xnrm*(here->MOS1gm+here->MOS1gmbs)+gcgs);
            *(here->MOS1DdpPtr) += (-here->MOS1drainConductance);
            *(here->MOS1GbPtr) -= gcgb;
            *(here->MOS1GdpPtr) -= gcgd;
            *(here->MOS1GspPtr) -= gcgs;
            *(here->MOS1SspPtr) += (-here->MOS1sourceConductance);
            *(here->MOS1BgPtr) -= gcgb;
            *(here->MOS1BdpPtr) -= here->MOS1gbd;
            *(here->MOS1BspPtr) -= here->MOS1gbs;
            *(here->MOS1DPdPtr) += (-here->MOS1drainConductance);
            *(here->MOS1DPgPtr) += ((xnrm-xrev)*here->MOS1gm-gcgd);
            *(here->MOS1DPbPtr) += (-here->MOS1gbd+(xnrm-xrev)*here->MOS1gmbs);
            *(here->MOS1DPspPtr) += (-here->MOS1gds-xnrm*
                     (here->MOS1gm+here->MOS1gmbs));
            *(here->MOS1SPgPtr) += (-(xnrm-xrev)*here->MOS1gm-gcgs);
            *(here->MOS1SPsPtr) += (-here->MOS1sourceConductance);
            *(here->MOS1SPbPtr) += (-here->MOS1gbs-(xnrm-xrev)*here->MOS1gmbs);
            *(here->MOS1SPdpPtr) += (-here->MOS1gds-xrev*
                     (here->MOS1gm+here->MOS1gmbs));
        }
    }
    return(OK);
} /* end of function MOS1load */
