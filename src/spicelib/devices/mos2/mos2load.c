/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 alansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/cktdefs.h"
#include "mos2defs.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* assuming silicon - make definition for epsilon of silicon */
#define EPSSIL (11.7 * 8.854214871e-12)

static double sig1[4] = {1.0, -1.0, 1.0, -1.0};
static double sig2[4] = {1.0,  1.0,-1.0, -1.0};

int
MOS2load(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current value into the 
         * sparse matrix previously provided 
         */
{
    MOS2model *model = (MOS2model *)inModel;
    MOS2instance *here;
    int error;
    double Beta;
    double DrainSatCur;
    double EffectiveLength;
    double GateBulkOverlapCap;
    double GateDrainOverlapCap;
    double GateSourceOverlapCap;
    double OxideCap;
    double SourceSatCur;
    double arg;
    double cbhat;
    double cdhat;
    double cdrain = 0.0;
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
    double sarg;
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
    double vt;      /* K * T / Q */
#ifndef PREDICTOR
    double xfact = 0.0;
#endif
    double capgs = 0.0;   /* total gate-source capacitance */
    double capgd = 0.0;   /* total gate-drain capacitance */
    double capgb = 0.0;   /* total gate-bulk capacitance */
    int xnrm;
    int xrev;
    int Check;
#ifndef NOBYPASS        
    double tempv;
#endif /*NOBYPASS*/
#ifdef CAPBYPASS
    int senflag;
#endif /* CAPBYPASS */    
    int SenCond=0;

#ifdef CAPBYPASS
    senflag = 0;
    if(ckt->CKTsenInfo){
        if(ckt->CKTsenInfo->SENstatus == PERTURBATION){
            if((ckt->CKTsenInfo->SENmode == ACSEN)||
                (ckt->CKTsenInfo->SENmode == TRANSEN)){
                senflag = 1;
            }
        }
    }
#endif /* CAPBYPASS */

    /*  loop through all the MOS2 device models */
    for( ; model != NULL; model = MOS2nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = MOS2instances(model); here != NULL ;
                here=MOS2nextInstance(here)) {

            vt = CONSTKoverQ * here->MOS2temp;
            Check=1;
            if(ckt->CKTsenInfo){
#ifdef SENSDEBUG
                printf("MOS2load %s\n",here->MOS2name);
#endif /* SENSDEBUG */

                if(ckt->CKTsenInfo->SENstatus == PERTURBATION) {
                    if(here->MOS2senPertFlag == OFF)continue;
                }
                SenCond = here->MOS2senPertFlag;

            }

            EffectiveLength=here->MOS2l - 2*model->MOS2latDiff;
            
            if( (here->MOS2tSatCurDens == 0) || 
                    (here->MOS2drainArea == 0) ||
                    (here->MOS2sourceArea == 0)) {
                DrainSatCur = here->MOS2m * here->MOS2tSatCur;
                SourceSatCur = here->MOS2m * here->MOS2tSatCur;
            } else {
                DrainSatCur = here->MOS2m * here->MOS2tSatCurDens * 
                        here->MOS2drainArea;
                SourceSatCur = here->MOS2m * here->MOS2tSatCurDens * 
                        here->MOS2sourceArea;
            }
            GateSourceOverlapCap = model->MOS2gateSourceOverlapCapFactor * 
                    here->MOS2m * here->MOS2w;
            GateDrainOverlapCap = model->MOS2gateDrainOverlapCapFactor * 
                    here->MOS2m * here->MOS2w;
            GateBulkOverlapCap = model->MOS2gateBulkOverlapCapFactor * 
                    here->MOS2m * EffectiveLength;
            Beta = here->MOS2tTransconductance * here->MOS2w *
                    here->MOS2m/EffectiveLength;
            OxideCap = model->MOS2oxideCapFactor * EffectiveLength * 
                    here->MOS2m * here->MOS2w;


            if(SenCond){
#ifdef SENSDEBUG
                printf("MOS2senPertFlag = ON \n");
#endif /* SENSDEBUG */
                if((ckt->CKTsenInfo->SENmode == TRANSEN) &&
                (ckt->CKTmode & MODEINITTRAN)) {
                    vgs = *(ckt->CKTstate1 + here->MOS2vgs);
                    vds = *(ckt->CKTstate1 + here->MOS2vds);
                    vbs = *(ckt->CKTstate1 + here->MOS2vbs);
                    vbd = *(ckt->CKTstate1 + here->MOS2vbd);
                    vgb = vgs - vbs;
                    vgd = vgs - vds;
                }
                else if (ckt->CKTsenInfo->SENmode == ACSEN){
                    vgb = model->MOS2type * ( 
                        *(ckt->CKTrhsOp+here->MOS2gNode) -
                        *(ckt->CKTrhsOp+here->MOS2bNode));
                    vbs = *(ckt->CKTstate0 + here->MOS2vbs);
                    vbd = *(ckt->CKTstate0 + here->MOS2vbd);
                    vgd = vgb + vbd ;
                    vgs = vgb + vbs ;
                    vds = vbs - vbd ;
                }
                else{
                    vgs = *(ckt->CKTstate0 + here->MOS2vgs);
                    vds = *(ckt->CKTstate0 + here->MOS2vds);
                    vbs = *(ckt->CKTstate0 + here->MOS2vbs);
                    vbd = *(ckt->CKTstate0 + here->MOS2vbd);
                    vgb = vgs - vbs;
                    vgd = vgs - vds;
                }
#ifdef SENSDEBUG
                printf(" vbs = %.7e ,vbd = %.7e,vgb = %.7e\n",vbs,vbd,vgb);
                printf(" vgs = %.7e ,vds = %.7e,vgd = %.7e\n",vgs,vds,vgd);
#endif /* SENSDEBUG */
                goto next1;
            }


            if((ckt->CKTmode & (MODEINITFLOAT | MODEINITPRED |MODEINITSMSIG
                    | MODEINITTRAN)) ||
                    ( (ckt->CKTmode & MODEINITFIX) && (!here->MOS2off) )  ) {
#ifndef PREDICTOR
                if(ckt->CKTmode & (MODEINITPRED | MODEINITTRAN) ) {

                    /* predictor step */

                    xfact=ckt->CKTdelta/ckt->CKTdeltaOld[1];
                    *(ckt->CKTstate0 + here->MOS2vbs) = 
                            *(ckt->CKTstate1 + here->MOS2vbs);
                    vbs = (1+xfact)* (*(ckt->CKTstate1 + here->MOS2vbs))
                            -(xfact * (*(ckt->CKTstate2 + here->MOS2vbs)));
                    *(ckt->CKTstate0 + here->MOS2vgs) = 
                            *(ckt->CKTstate1 + here->MOS2vgs);
                    vgs = (1+xfact)* (*(ckt->CKTstate1 + here->MOS2vgs))
                            -(xfact * (*(ckt->CKTstate2 + here->MOS2vgs)));
                    *(ckt->CKTstate0 + here->MOS2vds) = 
                            *(ckt->CKTstate1 + here->MOS2vds);
                    vds = (1+xfact)* (*(ckt->CKTstate1 + here->MOS2vds))
                            -(xfact * (*(ckt->CKTstate2 + here->MOS2vds)));
                    *(ckt->CKTstate0 + here->MOS2vbd) = 
                            *(ckt->CKTstate0 + here->MOS2vbs)-
                            *(ckt->CKTstate0 + here->MOS2vds);
                } else {
#endif /* PREDICTOR */

                    /* general iteration */


                    vbs = model->MOS2type * ( 
                        *(ckt->CKTrhsOld+here->MOS2bNode) -
                        *(ckt->CKTrhsOld+here->MOS2sNodePrime));
                    vgs = model->MOS2type * ( 
                        *(ckt->CKTrhsOld+here->MOS2gNode) -
                        *(ckt->CKTrhsOld+here->MOS2sNodePrime));
                    vds = model->MOS2type * ( 
                        *(ckt->CKTrhsOld+here->MOS2dNodePrime) -
                        *(ckt->CKTrhsOld+here->MOS2sNodePrime));
#ifndef PREDICTOR
                }
#endif /* PREDICTOR */

                /* now some common crunching for some more useful quantities */


                    vbd=vbs-vds;
                    vgd=vgs-vds;
                    vgdo = *(ckt->CKTstate0 + here->MOS2vgs) - 
                            *(ckt->CKTstate0 + here->MOS2vds);
                    delvbs = vbs - *(ckt->CKTstate0 + here->MOS2vbs);
                    delvbd = vbd - *(ckt->CKTstate0 + here->MOS2vbd);
                    delvgs = vgs - *(ckt->CKTstate0 + here->MOS2vgs);
                    delvds = vds - *(ckt->CKTstate0 + here->MOS2vds);
                    delvgd = vgd-vgdo;

                    /* these are needed for convergence testing */
                    if (here->MOS2mode >= 0) {
                        cdhat=
                            here->MOS2cd - 
                            here->MOS2gbd * delvbd +
                            here->MOS2gmbs * delvbs +
                            here->MOS2gm * delvgs + 
                            here->MOS2gds * delvds ;
                    } else {
                        cdhat=
                            here->MOS2cd +
                            ( here->MOS2gmbs -
                              here->MOS2gbd) * delvbd -
                            here->MOS2gm * delvgd + 
                            here->MOS2gds * delvds ;
                    }
                    cbhat=
                        here->MOS2cbs +
                        here->MOS2cbd +
                        here->MOS2gbd * delvbd +
                        here->MOS2gbs * delvbs ;


                /* now lets see if we can bypass (ugh) */
                /* the following massive if should all be one
                 * single compound if statement, but most compilers
                 * can't handle it in one piece, so it is broken up
                 * into several stages here
                 */
#ifndef NOBYPASS		 
                tempv = MAX(fabs(cbhat),fabs(here->MOS2cbs+here->MOS2cbd))+
                        ckt->CKTabstol;
                if((!(ckt->CKTmode & (MODEINITPRED|MODEINITTRAN|MODEINITSMSIG)
                        )) && (ckt->CKTbypass) )
                if ( (fabs(cbhat-(here->MOS2cbs + here->MOS2cbd))
                        < ckt->CKTreltol * tempv))
                if( (fabs(delvbs) < (ckt->CKTreltol * MAX(fabs(vbs),
                        fabs(*(ckt->CKTstate0+here->MOS2vbs)))+
                        ckt->CKTvoltTol)))
                if ( (fabs(delvbd) < (ckt->CKTreltol * MAX(fabs(vbd),
                        fabs(*(ckt->CKTstate0+here->MOS2vbd)))+
                        ckt->CKTvoltTol)) )
                if( (fabs(delvgs) < (ckt->CKTreltol * MAX(fabs(vgs),
                        fabs(*(ckt->CKTstate0+here->MOS2vgs)))+
                        ckt->CKTvoltTol)) )
                if ( (fabs(delvds) < (ckt->CKTreltol * MAX(fabs(vds),
                        fabs(*(ckt->CKTstate0+here->MOS2vds)))+
                        ckt->CKTvoltTol)) )
                if( (fabs(cdhat- here->MOS2cd) <
                        ckt->CKTreltol * MAX(fabs(cdhat),fabs(
                        here->MOS2cd)) + ckt->CKTabstol) ) {
                    /* bypass code */
                    /* nothing interesting has changed since last 
                     * iteration on this device, so we just
                     * copy all the values computed last iteration 
                     * out and keep going
                     */
                    vbs = *(ckt->CKTstate0 + here->MOS2vbs);
                    vbd = *(ckt->CKTstate0 + here->MOS2vbd);
                    vgs = *(ckt->CKTstate0 + here->MOS2vgs);
                    vds = *(ckt->CKTstate0 + here->MOS2vds);
                    vgd = vgs - vds;
                    vgb = vgs - vbs;
                    cdrain = here->MOS2mode * (here->MOS2cd + here->MOS2cbd);
                    if(ckt->CKTmode & (MODETRAN | MODETRANOP)) {
                        capgs = ( *(ckt->CKTstate0 + here->MOS2capgs)+
                                  *(ckt->CKTstate1 + here->MOS2capgs)+
                                  GateSourceOverlapCap );
                        capgd = ( *(ckt->CKTstate0 + here->MOS2capgd)+
                                  *(ckt->CKTstate1 + here->MOS2capgd)+
                                  GateDrainOverlapCap );
                        capgb = ( *(ckt->CKTstate0 + here->MOS2capgb)+
                                  *(ckt->CKTstate1 + here->MOS2capgb)+
                                  GateBulkOverlapCap );
                        if(ckt->CKTsenInfo){
                            here->MOS2cgs = capgs;
                            here->MOS2cgd = capgd;
                            here->MOS2cgb = capgb;
                        }
                    }
                    goto bypass;
                }
#endif /*NOBYPASS*/				
                /* ok - bypass is out, do it the hard way */

                von = model->MOS2type * here->MOS2von;
                /*
                 * limiting
                 * We want to keep device voltages from changing
                 * so fast that the exponentials churn out overflows 
                 * and similar rudeness
                 */
                if(*(ckt->CKTstate0 + here->MOS2vds) >=0) {
                    vgs = DEVfetlim(vgs,*(ckt->CKTstate0 + here->MOS2vgs)
                            ,von);
                    vds = vgs - vgd;
                    vds = DEVlimvds(vds,*(ckt->CKTstate0 + here->MOS2vds));
                    vgd = vgs - vds;
                } else {
                    vgd = DEVfetlim(vgd,vgdo,von);
                    vds = vgs - vgd;
                    if(!(ckt->CKTfixLimit)) {
                        vds = -DEVlimvds(-vds,-(*(ckt->CKTstate0 + 
                                here->MOS2vds)));
                    }
                    vgs = vgd + vds;
                }
                if(vds >= 0) {
                    vbs = DEVpnjlim(vbs,*(ckt->CKTstate0 + here->MOS2vbs),
                            vt,here->MOS2sourceVcrit,&Check);
                    vbd = vbs-vds;
                } else {
                    vbd = DEVpnjlim(vbd,*(ckt->CKTstate0 + here->MOS2vbd),
                            vt,here->MOS2drainVcrit,&Check);
                    vbs = vbd + vds;
                }
            } else {
                /* ok - not one of the simple cases, so we have to 
                 * look at other possibilities 
                 */

                if((ckt->CKTmode & MODEINITJCT) && !here->MOS2off) {
                    vds= model->MOS2type * here->MOS2icVDS;
                    vgs= model->MOS2type * here->MOS2icVGS;
                    vbs= model->MOS2type * here->MOS2icVBS;
                    if((vds==0) && (vgs==0) && (vbs==0) && 
                        ((ckt->CKTmode & 
                            (MODETRAN|MODEDCOP|MODEDCTRANCURVE)) ||
                         (!(ckt->CKTmode & MODEUIC)))) {
                        vbs = -1;
                        vgs = model->MOS2type * here->MOS2tVto;
                        vds = 0;
                    }
                } else {
                    vbs=vgs=vds=0;
                }
            } 

            /* now all the preliminaries are over - we can start doing the
             * real work 
             */

            vbd = vbs - vds;
            vgd = vgs - vds;
            vgb = vgs - vbs;

            /* bulk-source and bulk-drain doides
             * here we just evaluate the ideal diode current and the
             * correspoinding derivative (conductance).
             */

next1:      if(vbs <= -3*vt) {
                here->MOS2gbs = ckt->CKTgmin;
                here->MOS2cbs = here->MOS2gbs*vbs-SourceSatCur;
            } else {
                evbs = exp(MIN(MAX_EXP_ARG,vbs/vt));
                here->MOS2gbs = SourceSatCur*evbs/vt + ckt->CKTgmin;
                here->MOS2cbs = SourceSatCur*(evbs-1) + ckt->CKTgmin*vbs;
            }
            if(vbd <= -3*vt) {
                here->MOS2gbd = ckt->CKTgmin;
                here->MOS2cbd = here->MOS2gbd*vbd-DrainSatCur;
            } else {
                evbd = exp(MIN(MAX_EXP_ARG,vbd/vt));
                here->MOS2gbd = DrainSatCur*evbd/vt + ckt->CKTgmin;
                here->MOS2cbd = DrainSatCur*(evbd-1) + ckt->CKTgmin*vbd;
            }
            
            if(vds >= 0) {
                /* normal mode */
                here->MOS2mode = 1;
            } else {
                /* inverse mode */
                here->MOS2mode = -1;
            }
            {
            /* moseq2(vds,vbs,vgs,gm,gds,gmbs,qg,qc,qb,
             *        cggb,cgdb,cgsb,cbgb,cbdb,cbsb)
             */
            /* note:  cgdb, cgsb, cbdb, cbsb never used */

            /*
             *     this routine evaluates the drain current, its derivatives and
             *     the charges associated with the gate, channel and bulk
             *     for mosfets
             *
             */

            double arg1;
            double sarg1;
            double a4[4],b4[4],x4[8],poly4[8];
            double beta1;
            double dsrgdb;
            double d2sdb2;
            double sphi = 0.0;    /* square root of phi */
            double sphi3 = 0.0;   /* square root of phi cubed */
            double barg;
            double d2bdb2;
            double factor;
            double dbrgdb;
            double eta;
            double vbin;
            double argd = 0.0;
            double args = 0.0;
            double argss;
            double argsd;
            double argxs = 0.0;
            double argxd = 0.0;
            double daddb2;
            double dasdb2;
            double dbargd;
            double dbargs;
            double dbxwd;
            double dbxws;
            double dgddb2;
            double dgddvb;
            double dgdvds;
            double gamasd;
            double xwd;
            double xws;
            double ddxwd;
            double gammad;
            double vth;
            double cfs;
            double cdonco;
            double xn = 0.0;
            double argg = 0.0;
            double vgst;
            double sarg3;
            double sbiarg;
            double dgdvbs;
            double body;
            double gdbdv;
            double dodvbs;
            double dodvds = 0.0;
            double dxndvd = 0.0;
            double dxndvb = 0.0;
            double udenom;
            double dudvgs;
            double dudvds;
            double dudvbs;
            double gammd2;
            double argv;
            double vgsx;
            double ufact;
            double ueff;
            double dsdvgs;
            double dsdvbs;
            double a1;
            double a3;
            double a;
            double b1;
            double b3;
            double b;
            double c1;
            double c;
            double d1;
            double fi;
            double p0;
            double p2;
            double p3;
            double p4;
            double p;
            double r3;
            double r;
            double ro;
            double s2;
            double s;
            double v1;
            double v2;
            double xv;
            double y3;
            double delta4;
            double xvalid = 0.0;
            double bsarg = 0.0;
            double dbsrdb = 0.0;
            double bodys = 0.0;
            double gdbdvs = 0.0;
            double sargv;
            double xlfact;
            double dldsat;
            double xdv;
            double xlv;
            double vqchan;
            double dqdsat;
            double vl;
            double dfundg;
            double dfunds;
            double dfundb;
            double xls;
            double dldvgs;
            double dldvds;
            double dldvbs;
            double dfact;
            double clfact;
            double xleff;
            double deltal;
            double xwb;
            double vdson;
            double cdson;
            double didvds;
            double gdson;
            double gmw;
            double gbson;
            double expg;
            double xld;
            double xlamda = model->MOS2lambda;
            /* 'local' variables - these switch d & s around appropriately
             * so that we don't have to worry about vds < 0
             */
            double lvbs = here->MOS2mode==1?vbs:vbd;
            double lvds = here->MOS2mode*vds;
            double lvgs = here->MOS2mode==1?vgs:vgd;
            double phiMinVbs = here->MOS2tPhi - lvbs;
            double tmp; /* a temporary variable, not used for more than */
                        /* about 10 lines at a time */
            int iknt;
            int jknt;
            int i;
            int j;

            /*
             *  compute some useful quantities
             */

            if (lvbs <= 0.0) {
                sarg1 = sqrt(phiMinVbs);
                dsrgdb = -0.5/sarg1;
                d2sdb2 = 0.5*dsrgdb/phiMinVbs;
            } else {
                sphi = sqrt(here->MOS2tPhi);
                sphi3 = here->MOS2tPhi*sphi;
                sarg1 = sphi/(1.0+0.5*lvbs/here->MOS2tPhi);
                tmp = sarg1/sphi3;
                dsrgdb = -0.5*sarg1*tmp;
                d2sdb2 = -dsrgdb*tmp;
            }
            if ((lvbs-lvds) <= 0) {
                barg = sqrt(phiMinVbs+lvds);
                dbrgdb = -0.5/barg;
                d2bdb2 = 0.5*dbrgdb/(phiMinVbs+lvds);
            } else {
                sphi = sqrt(here->MOS2tPhi);/* added by HT 050523 */
                sphi3 = here->MOS2tPhi*sphi;/* added by HT 050523 */
                barg = sphi/(1.0+0.5*(lvbs-lvds)/here->MOS2tPhi);
                tmp = barg/sphi3;
                dbrgdb = -0.5*barg*tmp;
                d2bdb2 = -dbrgdb*tmp;
            }
            /*
             *  calculate threshold voltage (von)
             *     narrow-channel effect
             */

            /*XXX constant per device */
            factor = 0.125*model->MOS2narrowFactor*2.0*M_PI*EPSSIL/
                OxideCap*EffectiveLength;
            /*XXX constant per device */
            eta = 1.0+factor;
            vbin = here->MOS2tVbi*model->MOS2type+factor*phiMinVbs;
            if ((model->MOS2gamma > 0.0) || 
                    (model->MOS2substrateDoping > 0.0)) {
                xwd = model->MOS2xd*barg;
                xws = model->MOS2xd*sarg1;

                /*
                 *     short-channel effect with vds .ne. 0.0
                 */

                argss = 0.0;
                argsd = 0.0;
                dbargs = 0.0;
                dbargd = 0.0;
                dgdvds = 0.0;
                dgddb2 = 0.0;
                if (model->MOS2junctionDepth > 0) {
                    tmp = 2.0/model->MOS2junctionDepth;
                    argxs = 1.0+xws*tmp;
                    argxd = 1.0+xwd*tmp;
                    args = sqrt(argxs);
                    argd = sqrt(argxd);
                    tmp = .5*model->MOS2junctionDepth/EffectiveLength;
                    argss = tmp * (args-1.0);
                    argsd = tmp * (argd-1.0);
                }
                gamasd = model->MOS2gamma*(1.0-argss-argsd);
                dbxwd = model->MOS2xd*dbrgdb;
                dbxws = model->MOS2xd*dsrgdb;
                if (model->MOS2junctionDepth > 0) {
                    tmp = 0.5/EffectiveLength;
                    dbargs = tmp*dbxws/args;
                    dbargd = tmp*dbxwd/argd;
                    dasdb2 = -model->MOS2xd*( d2sdb2+dsrgdb*dsrgdb*
                        model->MOS2xd/(model->MOS2junctionDepth*argxs))/
                        (EffectiveLength*args);
                    daddb2 = -model->MOS2xd*( d2bdb2+dbrgdb*dbrgdb*
                        model->MOS2xd/(model->MOS2junctionDepth*argxd))/
                        (EffectiveLength*argd);
                    dgddb2 = -0.5*model->MOS2gamma*(dasdb2+daddb2);
                }
                dgddvb = -model->MOS2gamma*(dbargs+dbargd);
                if (model->MOS2junctionDepth > 0) {
                    ddxwd = -dbxwd;
                    dgdvds = -model->MOS2gamma*0.5*ddxwd/(EffectiveLength*argd);
                }
            } else {
                gamasd = model->MOS2gamma;
                gammad = model->MOS2gamma;
                dgddvb = 0.0;
                dgdvds = 0.0;
                dgddb2 = 0.0;
            }
            von = vbin+gamasd*sarg1;
            vth = von;
            vdsat = 0.0;
            if (model->MOS2fastSurfaceStateDensity != 0.0 && OxideCap != 0.0) {
                /* XXX constant per model */
                cfs = CHARGE*model->MOS2fastSurfaceStateDensity*
                    1e4 /*(cm**2/m**2)*/;
                cdonco = -(gamasd*dsrgdb+dgddvb*sarg1)+factor;
                
                xn = 1.0+cfs/OxideCap*here->MOS2m*
                      here->MOS2w*EffectiveLength+cdonco;
                
                tmp = vt*xn;
                von = von+tmp;
                argg = 1.0/tmp;
                vgst = lvgs-von;
            } else {
                vgst = lvgs-von;
                if (lvgs <= vbin) {
                    /*
                     *  cutoff region
                     */
                    here->MOS2gds = 0.0;
                    goto line1050;
                }
            }

            /*
             *  compute some more useful quantities
             */

            sarg3 = sarg1*sarg1*sarg1;
            /* XXX constant per model */
            sbiarg = sqrt(here->MOS2tBulkPot);
            gammad = gamasd;
            dgdvbs = dgddvb;
            body = barg*barg*barg-sarg3;
            gdbdv = 2.0*gammad*(barg*barg*dbrgdb-sarg1*sarg1*dsrgdb);
            dodvbs = -factor+dgdvbs*sarg1+gammad*dsrgdb;
            if (model->MOS2fastSurfaceStateDensity == 0.0) goto line400;
            if (OxideCap == 0.0) goto line410;
            dxndvb = 2.0*dgdvbs*dsrgdb+gammad*d2sdb2+dgddb2*sarg1;
            dodvbs = dodvbs+vt*dxndvb;
            dxndvd = dgdvds*dsrgdb;
            dodvds = dgdvds*sarg1+vt*dxndvd;
            /*
             *  evaluate effective mobility and its derivatives
             */
line400:
            if (OxideCap <= 0.0) goto line410;
            udenom = vgst;
            tmp = model->MOS2critField * 100 /* cm/m */ * EPSSIL/
                model->MOS2oxideCapFactor;
            if (udenom <= tmp) goto line410;
            ufact = exp(model->MOS2critFieldExp*log(tmp/udenom));
            ueff = model->MOS2surfaceMobility * 1e-4 /*(m**2/cm**2) */ *ufact;
            dudvgs = -ufact*model->MOS2critFieldExp/udenom;
            dudvds = 0.0;
            dudvbs = model->MOS2critFieldExp*ufact*dodvbs/vgst;
            goto line500;
line410:
            ufact = 1.0;
            ueff = model->MOS2surfaceMobility * 1e-4 /*(m**2/cm**2) */ ;
            dudvgs = 0.0;
            dudvds = 0.0;
            dudvbs = 0.0;
            /*
             *     evaluate saturation voltage and its derivatives according to
             *     grove-frohman equation
             */
line500:
            vgsx = lvgs;
            gammad = gamasd/eta;
            dgdvbs = dgddvb;
            if (model->MOS2fastSurfaceStateDensity != 0 && OxideCap != 0) {
                vgsx = MAX(lvgs,von);
            }
            if (gammad > 0) {
                gammd2 = gammad*gammad;
                argv = (vgsx-vbin)/eta+phiMinVbs;
                if (argv <= 0.0) {
                    vdsat = 0.0;
                    dsdvgs = 0.0;
                    dsdvbs = 0.0;
                } else {
                    arg1 = sqrt(1.0+4.0*argv/gammd2);
                    vdsat = (vgsx-vbin)/eta+gammd2*(1.0-arg1)/2.0;
                    vdsat = MAX(vdsat,0.0);
                    dsdvgs = (1.0-1.0/arg1)/eta;
                    dsdvbs = (gammad*(1.0-arg1)+2.0*argv/(gammad*arg1))/
                        eta*dgdvbs+1.0/arg1+factor*dsdvgs;
                }
            } else {
                vdsat = (vgsx-vbin)/eta;
                vdsat = MAX(vdsat,0.0);
                dsdvgs = 1.0;
                dsdvbs = 0.0;
            }
            if (model->MOS2maxDriftVel > 0) {
                /* 
                 *     evaluate saturation voltage and its derivatives 
                 *     according to baum's theory of scattering velocity 
                 *     saturation
                 */
                gammd2 = gammad*gammad;
                v1 = (vgsx-vbin)/eta+phiMinVbs;
                v2 = phiMinVbs;
                xv = model->MOS2maxDriftVel*EffectiveLength/ueff;
                a1 = gammad/0.75;
                b1 = -2.0*(v1+xv);
                c1 = -2.0*gammad*xv;
                d1 = 2.0*v1*(v2+xv)-v2*v2-4.0/3.0*gammad*sarg3;
                a = -b1;
                b = a1*c1-4.0*d1;
                c = -d1*(a1*a1-4.0*b1)-c1*c1;
                r = -a*a/3.0+b;
                s = 2.0*a*a*a/27.0-a*b/3.0+c;
                r3 = r*r*r;
                s2 = s*s;
                p = s2/4.0+r3/27.0;
                p0 = fabs(p);
                p2 = sqrt(p0);
                if (p < 0) {
                    ro = sqrt(s2/4.0+p0);
                    ro = log(ro)/3.0;
                    ro = exp(ro);
                    fi = atan(-2.0*p2/s);
                    y3 = 2.0*ro*cos(fi/3.0)-a/3.0;
                } else {
                    p3 = (-s/2.0+p2);
                    p3 = exp(log(fabs(p3))/3.0);
                    p4 = (-s/2.0-p2);
                    p4 = exp(log(fabs(p4))/3.0);
                    y3 = p3+p4-a/3.0;
                }
                iknt = 0;
                a3 = sqrt(a1*a1/4.0-b1+y3);
                b3 = sqrt(y3*y3/4.0-d1);
                for(i = 1;i<=4;i++) {
                    a4[i-1] = a1/2.0+sig1[i-1]*a3;
                    b4[i-1] = y3/2.0+sig2[i-1]*b3;
                    delta4 = a4[i-1]*a4[i-1]/4.0-b4[i-1];
                    if (delta4 < 0) continue;
                    iknt = iknt+1;
                    tmp = sqrt(delta4);
                    x4[iknt-1] = -a4[i-1]/2.0+tmp;
                    iknt = iknt+1;
                    x4[iknt-1] = -a4[i-1]/2.0-tmp;
                }
                jknt = 0;
                for(j = 1;j<=iknt;j++) {
                    if (x4[j-1] <= 0) continue;
                    /* XXX implement this sanely */
                    poly4[j-1] = x4[j-1]*x4[j-1]*x4[j-1]*x4[j-1]+a1*x4[j-1]*
                        x4[j-1]*x4[j-1];
                    poly4[j-1] = poly4[j-1]+b1*x4[j-1]*x4[j-1]+c1*x4[j-1]+d1;
                    if (fabs(poly4[j-1]) > 1.0e-6) continue;
                    jknt = jknt+1;
                    if (jknt <= 1) {
                        xvalid = x4[j-1];
                    }
                    if (x4[j-1] > xvalid) continue;
                    xvalid = x4[j-1];
                }
                if (jknt > 0) {
                    vdsat = xvalid*xvalid-phiMinVbs;
                }
            }
            /*
             *  evaluate effective channel length and its derivatives
             */
            if (lvds != 0.0) {
                gammad = gamasd;
                if ((lvbs-vdsat) <= 0) {
                    bsarg = sqrt(vdsat+phiMinVbs);
                    dbsrdb = -0.5/bsarg;
                } else {
					sphi = sqrt(here->MOS2tPhi);/* added by HT 050523 */
					sphi3 = here->MOS2tPhi*sphi;/* added by HT 050523 */
                    bsarg = sphi/(1.0+0.5*(lvbs-vdsat)/here->MOS2tPhi);
                    dbsrdb = -0.5*bsarg*bsarg/sphi3;
                }
                bodys = bsarg*bsarg*bsarg-sarg3;
                gdbdvs = 2.0*gammad*(bsarg*bsarg*dbsrdb-sarg1*sarg1*dsrgdb);
                if (model->MOS2maxDriftVel <= 0) {
                    if (model->MOS2substrateDoping == 0.0) goto line610;
                    if (xlamda > 0.0) goto line610;
                    argv = (lvds-vdsat)/4.0;
                    sargv = sqrt(1.0+argv*argv);
                    arg1 = sqrt(argv+sargv);
                    xlfact = model->MOS2xd/(EffectiveLength*lvds);
                    xlamda = xlfact*arg1;
                    dldsat = lvds*xlamda/(8.0*sargv);
                } else {
                    argv = (vgsx-vbin)/eta-vdsat;
                    xdv = model->MOS2xd/sqrt(model->MOS2channelCharge);
                    xlv = model->MOS2maxDriftVel*xdv/(2.0*ueff);
                    vqchan = argv-gammad*bsarg;
                    dqdsat = -1.0+gammad*dbsrdb;
                    vl = model->MOS2maxDriftVel*EffectiveLength;
                    dfunds = vl*dqdsat-ueff*vqchan;
                    dfundg = (vl-ueff*vdsat)/eta;
                    dfundb = -vl*(1.0+dqdsat-factor/eta)+ueff*
                        (gdbdvs-dgdvbs*bodys/1.5)/eta;
                    dsdvgs = -dfundg/dfunds;
                    dsdvbs = -dfundb/dfunds;
                    if (model->MOS2substrateDoping == 0.0) goto line610;
                    if (xlamda > 0.0) goto line610;
                    argv = lvds-vdsat;
                    argv = MAX(argv,0.0);
                    xls = sqrt(xlv*xlv+argv);
                    dldsat = xdv/(2.0*xls);
                    xlfact = xdv/(EffectiveLength*lvds);
                    xlamda = xlfact*(xls-xlv);
                    dldsat = dldsat/EffectiveLength;
                }
                dldvgs = dldsat*dsdvgs;
                dldvds = -xlamda+dldsat;
                dldvbs = dldsat*dsdvbs;
            } else {
line610:
                dldvgs = 0.0;
                dldvds = 0.0;
                dldvbs = 0.0;
            }
            /*
             *     limit channel shortening at punch-through
             */
            xwb = model->MOS2xd*sbiarg;
            xld = EffectiveLength-xwb;
            clfact = 1.0-xlamda*lvds;
            dldvds = -xlamda-dldvds;
            xleff = EffectiveLength*clfact;
            deltal = xlamda*lvds*EffectiveLength;
            if (model->MOS2substrateDoping == 0.0) xwb = 0.25e-6;
            if (xleff < xwb) {
                xleff = xwb/(1.0+(deltal-xld)/xwb);
                clfact = xleff/EffectiveLength;
                dfact = xleff*xleff/(xwb*xwb);
                dldvgs = dfact*dldvgs;
                dldvds = dfact*dldvds;
                dldvbs = dfact*dldvbs;
            }
            /*
             *  evaluate effective beta (effective kp)
             */
            beta1 = Beta*ufact/clfact;
            /*
             *  test for mode of operation and branch appropriately
             */
            gammad = gamasd;
            dgdvbs = dgddvb;
            if (lvds <= 1.0e-10) {
                if (lvgs <= von) {
                    if ((model->MOS2fastSurfaceStateDensity == 0.0) ||
                            (OxideCap == 0.0)) {
                        here->MOS2gds = 0.0;
                        goto line1050;
                    }

                    here->MOS2gds = beta1*(von-vbin-gammad*sarg1)*exp(argg*
                        (lvgs-von));
                    goto line1050;
                }


                here->MOS2gds = beta1*(lvgs-vbin-gammad*sarg1);
                goto line1050;
            }

            if (model->MOS2fastSurfaceStateDensity != 0 && OxideCap != 0) {
                if (lvgs > von) goto line900;
            } else {
                if (lvgs > vbin) goto line900;
                goto doneval;
            }
            
            if (lvgs > von) goto line900;
            /*
             *  subthreshold region
             */
            if (vdsat <= 0) {
                here->MOS2gds = 0.0;
                if (lvgs > vth) goto doneval;
                goto line1050;
            } 
            vdson = MIN(vdsat,lvds);
            if (lvds > vdsat) {
                barg = bsarg;
                dbrgdb = dbsrdb;
                body = bodys;
                gdbdv = gdbdvs;
            }
            cdson = beta1*((von-vbin-eta*vdson*0.5)*vdson-gammad*body/1.5);
            didvds = beta1*(von-vbin-eta*vdson-gammad*barg);
            gdson = -cdson*dldvds/clfact-beta1*dgdvds*body/1.5;
            if (lvds < vdsat) gdson = gdson+didvds;
            gbson = -cdson*dldvbs/clfact+beta1*
                (dodvbs*vdson+factor*vdson-dgdvbs*body/1.5-gdbdv);
            if (lvds > vdsat) gbson = gbson+didvds*dsdvbs;
            expg = exp(argg*(lvgs-von));
            cdrain = cdson*expg;
            gmw = cdrain*argg;
            here->MOS2gm = gmw;
            if (lvds > vdsat) here->MOS2gm = gmw+didvds*dsdvgs*expg;
            tmp = gmw*(lvgs-von)/xn;
            here->MOS2gds = gdson*expg-here->MOS2gm*dodvds-tmp*dxndvd;
            here->MOS2gmbs = gbson*expg-here->MOS2gm*dodvbs-tmp*dxndvb;
            goto doneval;

line900:
            if (lvds <= vdsat) {
                /*
                 *  linear region
                 */
                cdrain = beta1*((lvgs-vbin-eta*lvds/2.0)*lvds-gammad*body/1.5);
                arg1 = cdrain*(dudvgs/ufact-dldvgs/clfact);
                here->MOS2gm = arg1+beta1*lvds;
                arg1 = cdrain*(dudvds/ufact-dldvds/clfact);
                here->MOS2gds = arg1+beta1*(lvgs-vbin-eta*
                    lvds-gammad*barg-dgdvds*body/1.5);
                arg1 = cdrain*(dudvbs/ufact-dldvbs/clfact);
                here->MOS2gmbs = arg1-beta1*(gdbdv+dgdvbs*body/1.5-factor*lvds);
            } else {
                /* 
                 *  saturation region
                 */
                cdrain = beta1*((lvgs-vbin-eta*
                    vdsat/2.0)*vdsat-gammad*bodys/1.5);
                arg1 = cdrain*(dudvgs/ufact-dldvgs/clfact);
                here->MOS2gm = arg1+beta1*vdsat+beta1*(lvgs-
                    vbin-eta*vdsat-gammad*bsarg)*dsdvgs;
                here->MOS2gds = -cdrain*dldvds/clfact-beta1*dgdvds*bodys/1.5;
                arg1 = cdrain*(dudvbs/ufact-dldvbs/clfact);
                here->MOS2gmbs = arg1-beta1*(gdbdvs+dgdvbs*bodys/1.5-factor*
                        vdsat)+beta1* (lvgs-vbin-eta*vdsat-gammad*bsarg)*dsdvbs;
            }
            /*
             *     compute charges for "on" region
             */
            goto doneval;
            /*
             *  finish special cases
             */
line1050:
            cdrain = 0.0;
            here->MOS2gm = 0.0;
            here->MOS2gmbs = 0.0;
            /*
             *  finished
             */

            }
doneval:    
            here->MOS2von = model->MOS2type * von;
            here->MOS2vdsat = model->MOS2type * vdsat;
            /*
             *  COMPUTE EQUIVALENT DRAIN CURRENT SOURCE
             */
            here->MOS2cd=here->MOS2mode * cdrain - here->MOS2cbd;

            if (ckt->CKTmode & (MODETRAN | MODETRANOP|MODEINITSMSIG)) {
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
                        fabs(*(ckt->CKTstate0+here->MOS2vbs)))+
                        ckt->CKTvoltTol)|| senflag)
#endif /*CAPBYPASS*/
		 
                {
                    /* can't bypass the diode capacitance calculations */
                    if(here->MOS2Cbs != 0 || here->MOS2Cbssw != 0) {
                    if (vbs < here->MOS2tDepCap){		    
                        arg=1-vbs/here->MOS2tBulkPot;
                        /*
                         * the following block looks somewhat long and messy,
                         * but since most users use the default grading
                         * coefficients of .5, and sqrt is MUCH faster than an
                         * exp(log()) we use this special case code to buy time.
                         * (as much as 10% of total job time!)
                         */
                        if(model->MOS2bulkJctBotGradingCoeff ==
                                model->MOS2bulkJctSideGradingCoeff) {
                            if(model->MOS2bulkJctBotGradingCoeff == .5) {
                                sarg = sargsw = 1/sqrt(arg);
                            } else {
                                sarg = sargsw =
                                        exp(-model->MOS2bulkJctBotGradingCoeff*
                                        log(arg));
                            }
                        } else {
                            if(model->MOS2bulkJctBotGradingCoeff == .5) {
                                sarg = 1/sqrt(arg);
                            } else {
                                sarg = exp(-model->MOS2bulkJctBotGradingCoeff*
                                        log(arg));
                            }
                            if(model->MOS2bulkJctSideGradingCoeff == .5) {
                                sargsw = 1/sqrt(arg);
                            } else {
                                sargsw =exp(-model->MOS2bulkJctSideGradingCoeff*
                                        log(arg));
                            }
                        }
                        *(ckt->CKTstate0 + here->MOS2qbs) =
                            here->MOS2tBulkPot*(here->MOS2Cbs*
                            (1-arg*sarg)/(1-model->MOS2bulkJctBotGradingCoeff)
                            +here->MOS2Cbssw*
                            (1-arg*sargsw)/
                            (1-model->MOS2bulkJctSideGradingCoeff));
                        here->MOS2capbs=here->MOS2Cbs*sarg+
                                here->MOS2Cbssw*sargsw;
                    } else {
                        *(ckt->CKTstate0 + here->MOS2qbs) = here->MOS2f4s +
                                vbs*(here->MOS2f2s+vbs*(here->MOS2f3s/2));
                        here->MOS2capbs=here->MOS2f2s+here->MOS2f3s*vbs;
                    }
                    } else {
                        *(ckt->CKTstate0 + here->MOS2qbs) = 0;
                        here->MOS2capbs=0;
                    }
                }
#ifdef CAPBYPASS
                if(((ckt->CKTmode & (MODEINITPRED | MODEINITTRAN) ) ||
                        fabs(delvbd) >= ckt->CKTreltol * MAX(fabs(vbd),
                        fabs(*(ckt->CKTstate0+here->MOS2vbd)))+
                        ckt->CKTvoltTol)|| senflag)
#endif /*CAPBYPASS*/
				
                    /* can't bypass the diode capacitance calculations */
                {
                    if(here->MOS2Cbd != 0 || here->MOS2Cbdsw != 0 ) {
                    if (vbd < here->MOS2tDepCap) {
                        arg=1-vbd/here->MOS2tBulkPot;
                        /*
                         * the following block looks somewhat long and messy,
                         * but since most users use the default grading
                         * coefficients of .5, and sqrt is MUCH faster than an
                         * exp(log()) we use this special case code to buy time.
                         * (as much as 10% of total job time!)
                         */
                        if(model->MOS2bulkJctBotGradingCoeff == .5 &&
                                model->MOS2bulkJctSideGradingCoeff == .5) {
                            sarg = sargsw = 1/sqrt(arg);
                        } else {
                            if(model->MOS2bulkJctBotGradingCoeff == .5) {
                                sarg = 1/sqrt(arg);
                            } else {
                                sarg = exp(-model->MOS2bulkJctBotGradingCoeff*
                                        log(arg));
                            }
                            if(model->MOS2bulkJctSideGradingCoeff == .5) {
                                sargsw = 1/sqrt(arg);
                            } else {
                                sargsw =exp(-model->MOS2bulkJctSideGradingCoeff*
                                        log(arg));
                            }
                        }
                        *(ckt->CKTstate0 + here->MOS2qbd) =
                            here->MOS2tBulkPot*(here->MOS2Cbd*
                            (1-arg*sarg)
                            /(1-model->MOS2bulkJctBotGradingCoeff)
                            +here->MOS2Cbdsw*
                            (1-arg*sargsw)
                            /(1-model->MOS2bulkJctSideGradingCoeff));
                        here->MOS2capbd=here->MOS2Cbd*sarg+
                                here->MOS2Cbdsw*sargsw;
                    } else {
                        *(ckt->CKTstate0 + here->MOS2qbd) = here->MOS2f4d +
                                vbd * (here->MOS2f2d + vbd * here->MOS2f3d/2);
                        here->MOS2capbd=here->MOS2f2d + vbd * here->MOS2f3d;
                    }
                } else {
                    *(ckt->CKTstate0 + here->MOS2qbd) = 0;
                    here->MOS2capbd = 0;
                }
                }
                if(SenCond && (ckt->CKTsenInfo->SENmode==TRANSEN)) goto next2;

                if ( ckt->CKTmode & MODETRAN ) {
                    /* (above only excludes tranop, since we're only at this
                     * point if tran or tranop )
                     */

                    /*
                     *    calculate equivalent conductances and currents for
                     *    depletion capacitors
                     */

                    /* integrate the capacitors and save results */

                    error = NIintegrate(ckt,&geq,&ceq,here->MOS2capbd,
                            here->MOS2qbd);
                    if(error) return(error);
                    here->MOS2gbd += geq;
                    here->MOS2cbd += *(ckt->CKTstate0 + here->MOS2cqbd);
                    here->MOS2cd -= *(ckt->CKTstate0 + here->MOS2cqbd);
                    error = NIintegrate(ckt,&geq,&ceq,here->MOS2capbs,
                            here->MOS2qbs);
                    if(error) return(error);
                    here->MOS2gbs += geq;
                    here->MOS2cbs += *(ckt->CKTstate0 + here->MOS2cqbs);
                }
            }

            if(SenCond) goto next2;
            /*
             *  check convergence
             */
            if ( (here->MOS2off == 0)  || 
                    (!(ckt->CKTmode & (MODEINITFIX|MODEINITSMSIG))) ){
                if (Check == 1) {
                    ckt->CKTnoncon++;
		    ckt->CKTtroubleElt = (GENinstance *) here;
                }
            }
next2:      *(ckt->CKTstate0 + here->MOS2vbs) = vbs;
            *(ckt->CKTstate0 + here->MOS2vbd) = vbd;
            *(ckt->CKTstate0 + here->MOS2vgs) = vgs;
            *(ckt->CKTstate0 + here->MOS2vds) = vds;

            /*
             *     meyer's capacitor model
             */
            if ( ckt->CKTmode & (MODETRAN|MODETRANOP|MODEINITSMSIG)) {
                /*
                 *     calculate meyer's capacitors
                 */
                if (here->MOS2mode > 0){
                    DEVqmeyer (vgs,vgd,vgb,von,vdsat,
                        (ckt->CKTstate0 + here->MOS2capgs),
                        (ckt->CKTstate0 + here->MOS2capgd),
                        (ckt->CKTstate0 + here->MOS2capgb),
                        here->MOS2tPhi,OxideCap);
                } else {
                    DEVqmeyer (vgd,vgs,vgb,von,vdsat,
                        (ckt->CKTstate0 + here->MOS2capgd),
                        (ckt->CKTstate0 + here->MOS2capgs),
                        (ckt->CKTstate0 + here->MOS2capgb),
                        here->MOS2tPhi,OxideCap);
                }
                vgs1 = *(ckt->CKTstate1 + here->MOS2vgs);
                vgd1 = vgs1 - *(ckt->CKTstate1 + here->MOS2vds);
                vgb1 = vgs1 - *(ckt->CKTstate1 + here->MOS2vbs);
                if(ckt->CKTmode & MODETRANOP) {
                    capgs = 2 * *(ckt->CKTstate0 + here->MOS2capgs)+
                        GateSourceOverlapCap;
                    capgd = 2 * *(ckt->CKTstate0 + here->MOS2capgd)+
                        GateDrainOverlapCap;
                    capgb = 2 * *(ckt->CKTstate0 + here->MOS2capgb)+
                        GateBulkOverlapCap;
                } else {
                    capgs = *(ckt->CKTstate0 + here->MOS2capgs)+
                            *(ckt->CKTstate1 + here->MOS2capgs)+
                            GateSourceOverlapCap;
                    capgd = *(ckt->CKTstate0 + here->MOS2capgd)+
                            *(ckt->CKTstate1 + here->MOS2capgd)+
                            GateDrainOverlapCap;
                    capgb = *(ckt->CKTstate0 + here->MOS2capgb)+
                            *(ckt->CKTstate1 + here->MOS2capgb)+
                            GateBulkOverlapCap;
                }
                if(ckt->CKTsenInfo){
                    here->MOS2cgs = capgs;
                    here->MOS2cgd = capgd;
                    here->MOS2cgb = capgb;
                }
                /*
                 *     store small-signal parameters (for meyer's model)
                 *  all parameters already stored, so done...
                 */

                if(SenCond){
                    if(ckt->CKTsenInfo->SENmode & (DCSEN|ACSEN)){
                        continue;
                    }
                }

#ifndef PREDICTOR
                if(ckt->CKTmode & (MODEINITPRED | MODEINITTRAN) ) {
                    *(ckt->CKTstate0 +  here->MOS2qgs) =
                        (1+xfact) * *(ckt->CKTstate1 + here->MOS2qgs)
                        -xfact * *(ckt->CKTstate2 + here->MOS2qgs);
                    *(ckt->CKTstate0 +  here->MOS2qgd) =
                        (1+xfact) * *(ckt->CKTstate1 + here->MOS2qgd)
                        -xfact * *(ckt->CKTstate2 + here->MOS2qgd);
                    *(ckt->CKTstate0 +  here->MOS2qgb) =
                        (1+xfact) * *(ckt->CKTstate1 + here->MOS2qgb)
                        -xfact * *(ckt->CKTstate2 + here->MOS2qgb);
                } else {
#endif /* PREDICTOR */
                    if(ckt->CKTmode & MODETRAN) {
                        *(ckt->CKTstate0 + here->MOS2qgs) = (vgs-vgs1)*capgs +
                            *(ckt->CKTstate1 + here->MOS2qgs) ;
                        *(ckt->CKTstate0 + here->MOS2qgd) = (vgd-vgd1)*capgd +
                            *(ckt->CKTstate1 + here->MOS2qgd) ;
                        *(ckt->CKTstate0 + here->MOS2qgb) = (vgb-vgb1)*capgb +
                            *(ckt->CKTstate1 + here->MOS2qgb) ;
                    } else {
                        /* TRANOP */
                        *(ckt->CKTstate0 + here->MOS2qgs) = capgs*vgs;
                        *(ckt->CKTstate0 + here->MOS2qgd) = capgd*vgd;
                        *(ckt->CKTstate0 + here->MOS2qgb) = capgb*vgb;
                    }
#ifndef PREDICTOR
                }
#endif /* PREDICTOR */
            }
#ifndef NOBYPASS
bypass:
#endif /* NOBYPASS */

            if(SenCond) continue;


            if((ckt->CKTmode & MODEINITTRAN) || (!(ckt->CKTmode & MODETRAN)) ) {
                /* initialize to zero charge conductances and current */

                gcgs=0;
                ceqgs=0;
                gcgd=0;
                ceqgd=0;
                gcgb=0;
                ceqgb=0;
            } else {
                if(capgs == 0) *(ckt->CKTstate0 + here->MOS2cqgs) =0;
                if(capgd == 0) *(ckt->CKTstate0 + here->MOS2cqgd) =0;
                if(capgb == 0) *(ckt->CKTstate0 + here->MOS2cqgb) =0;
                /*
                 *    calculate equivalent conductances and currents for
                 *    meyer"s capacitors
                 */
                error = NIintegrate(ckt,&gcgs,&ceqgs,capgs,here->MOS2qgs);
                if(error) return(error);
                error = NIintegrate(ckt,&gcgd,&ceqgd,capgd,here->MOS2qgd);
                if(error) return(error);
                error = NIintegrate(ckt,&gcgb,&ceqgb,capgb,here->MOS2qgb);
                if(error) return(error);
                ceqgs=ceqgs-gcgs*vgs+ckt->CKTag[0]* 
                        *(ckt->CKTstate0 + here->MOS2qgs);
                ceqgd=ceqgd-gcgd*vgd+ckt->CKTag[0]*
                        *(ckt->CKTstate0 + here->MOS2qgd);
                ceqgb=ceqgb-gcgb*vgb+ckt->CKTag[0]*
                        *(ckt->CKTstate0 + here->MOS2qgb);
            }
            /*
             *     store charge storage info for meyer's cap in lx table
             */

            /*
             *  load current vector
             */
            ceqbs = model->MOS2type * 
                    (here->MOS2cbs-(here->MOS2gbs)*vbs);
            ceqbd = model->MOS2type * 
                    (here->MOS2cbd-(here->MOS2gbd)*vbd);
            if (here->MOS2mode >= 0) {
                xnrm=1;
                xrev=0;
                cdreq=model->MOS2type*(cdrain-here->MOS2gds*vds-
                        here->MOS2gm*vgs-here->MOS2gmbs*vbs);
            } else {
                xnrm=0;
                xrev=1;
                cdreq = -(model->MOS2type)*(cdrain-here->MOS2gds*(-vds)-
                        here->MOS2gm*vgd-here->MOS2gmbs*vbd);
            }
            *(ckt->CKTrhs + here->MOS2gNode) -= 
                    model->MOS2type * (ceqgs + ceqgb + ceqgd);
            *(ckt->CKTrhs + here->MOS2bNode) -=
                    (ceqbs+ceqbd-model->MOS2type * ceqgb);
            *(ckt->CKTrhs + here->MOS2dNodePrime) +=
                    ceqbd - cdreq + model->MOS2type * ceqgd;
            *(ckt->CKTrhs + here->MOS2sNodePrime) += 
                    cdreq + ceqbs + model->MOS2type * ceqgs;

#if 0
	    printf(" loading %s at time %g\n", here->MOS2name, ckt->CKTtime);
	    printf("%g %g %g %g %g\n", here->MOS2drainConductance,
		   gcgd+gcgs+gcgb, here->MOS2sourceConductance,
		   here->MOS2gbd, here->MOS2gbs);
	    printf("%g %g %g %g %g\n", -gcgb, 0.0, 0.0,
		   here->MOS2gds, here->MOS2gm);
	    printf("%g %g %g %g %g\n", here->MOS2gds, here->MOS2gmbs,
		   gcgd, -gcgs, -gcgd);
	    printf("%g %g %g %g %g\n", -gcgs, -gcgd, 0.0, -gcgs, 0.0);
#endif

            /*
             *  load y matrix
             */
            *(here->MOS2DdPtr) += (here->MOS2drainConductance);
            *(here->MOS2GgPtr) += gcgd+gcgs+gcgb;
            *(here->MOS2SsPtr) += (here->MOS2sourceConductance);
            *(here->MOS2BbPtr) += (here->MOS2gbd+here->MOS2gbs+gcgb);
            *(here->MOS2DPdpPtr) += here->MOS2drainConductance+here->MOS2gds+
                    here->MOS2gbd+xrev*(here->MOS2gm+here->MOS2gmbs)+gcgd;
            *(here->MOS2SPspPtr) += here->MOS2sourceConductance+here->MOS2gds+
                    here->MOS2gbs+xnrm*(here->MOS2gm+here->MOS2gmbs)+gcgs;
            *(here->MOS2DdpPtr) -= here->MOS2drainConductance;
            *(here->MOS2GbPtr) -= gcgb;
            *(here->MOS2GdpPtr) -= gcgd;
            *(here->MOS2GspPtr) -= gcgs;
            *(here->MOS2SspPtr) -= here->MOS2sourceConductance;
            *(here->MOS2BgPtr) -= gcgb;
            *(here->MOS2BdpPtr) -= here->MOS2gbd;
            *(here->MOS2BspPtr) -= here->MOS2gbs;
            *(here->MOS2DPdPtr) -= here->MOS2drainConductance;
            *(here->MOS2DPgPtr) += ((xnrm-xrev)*here->MOS2gm-gcgd);
            *(here->MOS2DPbPtr) += (-here->MOS2gbd+(xnrm-xrev)*here->MOS2gmbs);
            *(here->MOS2DPspPtr) -= here->MOS2gds+xnrm*(here->MOS2gm+
                    here->MOS2gmbs);
            *(here->MOS2SPgPtr) -= (xnrm-xrev)*here->MOS2gm+gcgs;
            *(here->MOS2SPsPtr) -= here->MOS2sourceConductance;
            *(here->MOS2SPbPtr) -= here->MOS2gbs+(xnrm-xrev)*here->MOS2gmbs;
            *(here->MOS2SPdpPtr) -= here->MOS2gds+xrev*(here->MOS2gm+
                    here->MOS2gmbs);
        }
    }
    return(OK);
}
