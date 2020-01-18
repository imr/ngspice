/**********
STAG version 2.7
Copyright 2000 owned by the United Kingdom Secretary of State for Defence
acting through the Defence Evaluation and Research Agency.
Developed by :     Jim Benson,
                   Department of Electronics and Computer Science,
                   University of Southampton,
                   United Kingdom.
With help from :   Nele D'Halleweyn, Ketan Mistry, Bill Redman-White,
						 and Craig Easson.

Based on STAG version 2.1
Developed by :     Mike Lee,
With help from :   Bernard Tenbroek, Bill Redman-White, Mike Uren, Chris Edwards
                   and John Bunyan.
Acknowledgements : Rupert Howes and Pete Mole.
**********/

/********** 
Modified by Paolo Nenzi 2002
ngspice integration
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "soi3defs.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

extern double DEVsoipnjlim(double, double, double, double, int *);

int
SOI3load(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current value into the
         * sparse matrix previously provided
         */
{
    SOI3model *model = (SOI3model *) inModel;
    SOI3instance *here;
    double Beta;
    double DrainSatCur;    /* for drain pn junction */
    double SourceSatCur;  /* for source pn junction */
    double DrainSatCur1;    /* for 2nd drain pn junction */
    double SourceSatCur1;  /* for 2nd source pn junction */
    double EffectiveLength,logL;
    double FrontGateBulkOverlapCap;
    double FrontGateDrainOverlapCap;
    double FrontGateSourceOverlapCap;
    double BackGateBulkOverlapCap;
    double BackGateDrainOverlapCap;
    double BackGateSourceOverlapCap;
    double Frontcapargs[6];
    double Backcapargs[6];
    double FrontOxideCap;
    double BackOxideCap;
    double arg;
    double ibhat;
    double idhat;
    double iPthat;  /* needed for convergence */
    double ieqPt;   /* value of equivalent current source */
    double idrain;
    double idreq;
    double ieq;
    double ieqct,ieqct1,ieqct2,ieqct3,ieqct4;
    double ieqbd;
    double ieqbs;
    double iMdbeq;
    double iMsbeq;
    double iBJTdbeq;
    double iBJTsbeq;
    double delvbd;
    double delvbs;
    double delvds;
    double delvgfd;
    double delvgfs;
    double delvgbd;
    double delvgbs;
    double deldeltaT;
    double evbd,evbd1;
    double evbs,evbs1;
    double rtargs[5];
    double grt[5];
    double gct[5];
    int tnodeindex;
    double geq;
    double sarg;
    double vbd;
    double vbs;
    double vds;
/* the next lot are so we can cast the problem with the body node as ref */
    double vsb;
    double vdb;
    double vgbb;
/* now back to our regular programming   */
/* vgfb exists already for gate cap calc */
    double deltaT,deltaT1 = 0.0,deltaT2 = 0.0,deltaT3 = 0.0;
    double deltaT4 = 0.0,deltaT5 = 0.0;

    double vdsat_ext;
    double vgfb;
    double vgfd;
    double vgbd;
    double vgfdo;
    double vgbdo;
    double vgfs;
    double vgbs;
    double von;
    double vt;
#ifndef PREDICTOR
    double xfact;
#endif
    int xnrm;
    int xrev;

/* now stuff needed for new charge model */
   double paramargs[10];
   double Bfargs[2],alpha_args[5];
   double psi_st0args[5];
   double vGTargs[5];
   double psi_sLargs[5],psi_s0args[5];
   double ldargs[5];
   double qgatef,qdrn,qsrc,qbody,qgateb;
   double ieqqgf,ieqqd,ieqqs,ieqqgb;

   double cgfgf,cgfd,cgfs,cgfdeltaT,cgfgb;
   double cdgf,cdd,cds,cddeltaT,cdgb;
   double csgf,csd,css,csdeltaT,csgb;
   double cbgf,cbd,cbs,cbdeltaT,cbgb;
   double cgbgf,cgbd,cgbs,cgbdeltaT,cgbgb;
   double gcgfgf,gcgfd,gcgfs,gcgfdeltaT,gcgfgb;
   double gcdgf,gcdd,gcds,gcddeltaT,gcdgb;
   double gcsgf,gcsd,gcss,gcsdeltaT,gcsgb;
   double gcbgf,gcbd,gcbs,gcbdeltaT,gcbgb;
   double gcgbgf,gcgbd,gcgbs,gcgbdeltaT,gcgbgb;

   /* remove compiler warnings */
   cgfgf=cgfd=cgfs=cgfdeltaT=cgfgb = 0;
   cdgf=cdd=cds=cddeltaT=cdgb = 0;
   csgf=csd=css=csdeltaT=csgb = 0;
   cbgf=cbd=cbs=cbdeltaT=cbgb = 0;
   cgbgf=cgbd=cgbs=cgbdeltaT=cgbgb = 0;

    double alphaBJT;
    double tauFBJTeff,tauRBJTeff;
    double ISts,IS1ts,IStd,IS1td;
    double ieqqBJTbs,ieqqBJTbd;
    double gcBJTbsbs,gcBJTbsdeltaT;
    double gcBJTbdbd,gcBJTbddeltaT;
    double ag0;

    int Check;
    int ByPass;

#ifndef NOBYPASS
    double tempv;
#endif /*NOBYPASS*/
    int error;
#ifdef CAPBYPASS
    int senflag;
#endif /* CAPBYPASS */

    double m;

    for( ; model != NULL; model = SOI3nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = SOI3instances(model); here != NULL ;
                here=SOI3nextInstance(here)) {

            vt = CONSTKoverQ * here->SOI3temp;
            Check=1;
            ByPass=0;

            /* first, we compute a few useful values - these could be
             * pre-computed, but for historical reasons are still done
             * here.  They may be moved at the expense of instance size
             */

            EffectiveLength=here->SOI3l - 2*model->SOI3latDiff;
            logL = log(EffectiveLength);
            if (here->SOI3tSatCurDens == 0) {
                DrainSatCur = here->SOI3tSatCur;
                SourceSatCur = here->SOI3tSatCur;
            } else {
                DrainSatCur = here->SOI3tSatCurDens *
                              here->SOI3w;
                SourceSatCur = here->SOI3tSatCurDens *
                               here->SOI3w;
            }
            if (here->SOI3tSatCurDens1 == 0) {
                DrainSatCur1 = here->SOI3tSatCur1;
                SourceSatCur1 = here->SOI3tSatCur1;
            } else {
                DrainSatCur1 = here->SOI3tSatCurDens1 *
                              here->SOI3w;
                SourceSatCur1 = here->SOI3tSatCurDens1 *
                               here->SOI3w;
            }
/* NB Junction sat. cur. density is NOW PER UNIT WIDTH */

            /* JimB - can use basic device geometry to estimate front and back */
            /* overlap capacitances to a first approximation. Use this default */
            /* model if capacitance factors aren't given in model netlist. */

            /* Calculate front gate overlap capacitances. */

        		if(model->SOI3frontGateSourceOverlapCapFactorGiven)
            {
               FrontGateSourceOverlapCap = model->SOI3frontGateSourceOverlapCapFactor * here->SOI3w;
        		}
            else
            {
               FrontGateSourceOverlapCap = model->SOI3latDiff * here->SOI3w * model->SOI3frontOxideCapFactor;
            }
        		if(model->SOI3frontGateDrainOverlapCapFactorGiven)
            {
				   FrontGateDrainOverlapCap = model->SOI3frontGateDrainOverlapCapFactor * here->SOI3w;
        		}
            else
            {
               FrontGateDrainOverlapCap = model->SOI3latDiff * here->SOI3w * model->SOI3frontOxideCapFactor;
            }
        		if(model->SOI3frontGateBulkOverlapCapFactorGiven)
            {
               FrontGateBulkOverlapCap = model->SOI3frontGateBulkOverlapCapFactor * EffectiveLength;
            }
            else
            {
               FrontGateBulkOverlapCap = EffectiveLength * (0.1*1e-6*model->SOI3minimumFeatureSize)
               			* model->SOI3frontOxideCapFactor;
            }

            /* Calculate back gate overlap capacitances. */

        		if( (model->SOI3backGateSourceOverlapCapAreaFactorGiven) &&
            (!model->SOI3backGateSourceOverlapCapAreaFactor || here->SOI3asGiven) )
            {
               BackGateSourceOverlapCap = model->SOI3backGateSourceOverlapCapAreaFactor * here->SOI3as;
        		}
            else
            {
               BackGateSourceOverlapCap = (2*1e-6*model->SOI3minimumFeatureSize + model->SOI3latDiff) * here->SOI3w
               			* model->SOI3backOxideCapFactor;
            }
        		if( (model->SOI3backGateDrainOverlapCapAreaFactorGiven) &&
            (!model->SOI3backGateDrainOverlapCapAreaFactor || here->SOI3adGiven) )
            {
               BackGateDrainOverlapCap = model->SOI3backGateDrainOverlapCapAreaFactor * here->SOI3ad;
        		}
            else
            {
               BackGateDrainOverlapCap = (2*1e-6*model->SOI3minimumFeatureSize + model->SOI3latDiff) * here->SOI3w
               			* model->SOI3backOxideCapFactor;
            }
        		if( (model->SOI3backGateBulkOverlapCapAreaFactorGiven) &&
            (!model->SOI3backGateBulkOverlapCapAreaFactor || here->SOI3abGiven) )
            {
               BackGateBulkOverlapCap = model->SOI3backGateBulkOverlapCapAreaFactor * here->SOI3ab;
            }
            else
            {
               BackGateBulkOverlapCap = EffectiveLength * (0.1*1e-6*model->SOI3minimumFeatureSize + here->SOI3w)
               			* model->SOI3backOxideCapFactor;
            }

            Beta = here->SOI3tTransconductance * here->SOI3w/EffectiveLength;
            /* reset mu_eff to ambient temp value in SI units */
            here->SOI3ueff = here->SOI3tTransconductance/
                             model->SOI3frontOxideCapFactor;
            FrontOxideCap = model->SOI3frontOxideCapFactor * EffectiveLength *
                    here->SOI3w;
            BackOxideCap = model->SOI3backOxideCapFactor * EffectiveLength *
                    here->SOI3w;
            /*
             * ok - now to do the start-up operations
             *
             * we must get values for vbs, vds, vgfs and vgbs from somewhere
             * so we either predict them or recover them from last iteration
             * These are the two most common cases - either a prediction
             * step or the general iteration step and they
             * share some code, so we put them first - others later on
             */

             /* yeah, but we'll then use vsb and vdb for the real work */

/* we will use conventional voltages to save *
 * work rewriting the convergence criteria   */

            if((ckt->CKTmode & (MODEINITFLOAT | MODEINITPRED | MODEINITSMSIG
                    | MODEINITTRAN)) ||
                    ( (ckt->CKTmode & MODEINITFIX) && (!here->SOI3off) )  )
            {
#ifndef PREDICTOR
                if(ckt->CKTmode & (MODEINITPRED | MODEINITTRAN) )
                {

                    /* predictor step */

                    xfact=ckt->CKTdelta/ckt->CKTdeltaOld[1];
                    *(ckt->CKTstate0 + here->SOI3vbs) =
                            *(ckt->CKTstate1 + here->SOI3vbs);
                    vbs = (1+xfact)* (*(ckt->CKTstate1 + here->SOI3vbs))
                            -(xfact * (*(ckt->CKTstate2 + here->SOI3vbs)));
                    *(ckt->CKTstate0 + here->SOI3vgfs) =
                            *(ckt->CKTstate1 + here->SOI3vgfs);
                    vgfs = (1+xfact)* (*(ckt->CKTstate1 + here->SOI3vgfs))
                            -(xfact * (*(ckt->CKTstate2 + here->SOI3vgfs)));
                    *(ckt->CKTstate0 + here->SOI3vgbs) =
                            *(ckt->CKTstate1 + here->SOI3vgbs);
                    vgbs = (1+xfact)* (*(ckt->CKTstate1 + here->SOI3vgbs))
                            -(xfact * (*(ckt->CKTstate2 + here->SOI3vgbs)));
                    *(ckt->CKTstate0 + here->SOI3vds) =
                            *(ckt->CKTstate1 + here->SOI3vds);
                    vds = (1+xfact)* (*(ckt->CKTstate1 + here->SOI3vds))
                            -(xfact * (*(ckt->CKTstate2 + here->SOI3vds)));
                    *(ckt->CKTstate0 + here->SOI3vbd) =
                            *(ckt->CKTstate0 + here->SOI3vbs)-
                            *(ckt->CKTstate0 + here->SOI3vds);
                    *(ckt->CKTstate0 + here->SOI3deltaT) =
                            *(ckt->CKTstate1 + here->SOI3deltaT);
                    deltaT = MAX(0,(1+xfact)* (*(ckt->CKTstate1 + here->SOI3deltaT))
                            -(xfact * (*(ckt->CKTstate2 + here->SOI3deltaT))));
                       /* need to stop deltaT being -ve */

                    /* JimB - 19/5/99 */
                    if (here->SOI3numThermalNodes == 0)
                    {
                      deltaT1=deltaT2=deltaT3=deltaT4=deltaT5=0.0;
                    }
                    if (here->SOI3numThermalNodes == 1)
                    {
                      deltaT1=deltaT;
                      deltaT2=deltaT3=deltaT4=deltaT5=0.0;
                    }
                    if (here->SOI3numThermalNodes == 2)
                    {
                      deltaT1 = MAX(0,(1+xfact)* (*(ckt->CKTstate1 + here->SOI3deltaT1))
                            -(xfact * (*(ckt->CKTstate2 + here->SOI3deltaT1))));
                      deltaT2 = MAX(0,(1+xfact)* (*(ckt->CKTstate1 + here->SOI3deltaT2))
                            -(xfact * (*(ckt->CKTstate2 + here->SOI3deltaT2))));
                      deltaT3=deltaT4=deltaT5=0.0;
                    }
                    if (here->SOI3numThermalNodes == 3)
                    {
                      deltaT1 = MAX(0,(1+xfact)* (*(ckt->CKTstate1 + here->SOI3deltaT1))
                            -(xfact * (*(ckt->CKTstate2 + here->SOI3deltaT1))));
                      deltaT2 = MAX(0,(1+xfact)* (*(ckt->CKTstate1 + here->SOI3deltaT2))
                            -(xfact * (*(ckt->CKTstate2 + here->SOI3deltaT2))));
                      deltaT3 = MAX(0,(1+xfact)* (*(ckt->CKTstate1 + here->SOI3deltaT3))
                            -(xfact * (*(ckt->CKTstate2 + here->SOI3deltaT3))));
                      deltaT4=deltaT5=0.0;
                    }
                    if (here->SOI3numThermalNodes == 4)
                    {
                      deltaT1 = MAX(0,(1+xfact)* (*(ckt->CKTstate1 + here->SOI3deltaT1))
                            -(xfact * (*(ckt->CKTstate2 + here->SOI3deltaT1))));
                      deltaT2 = MAX(0,(1+xfact)* (*(ckt->CKTstate1 + here->SOI3deltaT2))
                            -(xfact * (*(ckt->CKTstate2 + here->SOI3deltaT2))));
                      deltaT3 = MAX(0,(1+xfact)* (*(ckt->CKTstate1 + here->SOI3deltaT3))
                            -(xfact * (*(ckt->CKTstate2 + here->SOI3deltaT3))));
                      deltaT4 = MAX(0,(1+xfact)* (*(ckt->CKTstate1 + here->SOI3deltaT4))
                            -(xfact * (*(ckt->CKTstate2 + here->SOI3deltaT4))));
                      deltaT5=0.0;
                    }
                    if (here->SOI3numThermalNodes == 5)
                    {
                      deltaT1 = MAX(0,(1+xfact)* (*(ckt->CKTstate1 + here->SOI3deltaT1))
                            -(xfact * (*(ckt->CKTstate2 + here->SOI3deltaT1))));
                      deltaT2 = MAX(0,(1+xfact)* (*(ckt->CKTstate1 + here->SOI3deltaT2))
                            -(xfact * (*(ckt->CKTstate2 + here->SOI3deltaT2))));
                      deltaT3 = MAX(0,(1+xfact)* (*(ckt->CKTstate1 + here->SOI3deltaT3))
                            -(xfact * (*(ckt->CKTstate2 + here->SOI3deltaT3))));
                      deltaT4 = MAX(0,(1+xfact)* (*(ckt->CKTstate1 + here->SOI3deltaT4))
                            -(xfact * (*(ckt->CKTstate2 + here->SOI3deltaT4))));
                      deltaT5 =MAX(0,deltaT-deltaT1-deltaT2-deltaT3-deltaT4);
                    }

                    *(ckt->CKTstate0 + here->SOI3idrain) =
                            *(ckt->CKTstate1 + here->SOI3idrain);
                    *(ckt->CKTstate0 + here->SOI3cgfgf) =
                            *(ckt->CKTstate1 + here->SOI3cgfgf);
                    *(ckt->CKTstate0 + here->SOI3cgfd) =
                            *(ckt->CKTstate1 + here->SOI3cgfd);
                    *(ckt->CKTstate0 + here->SOI3cgfs) =
                            *(ckt->CKTstate1 + here->SOI3cgfs);
                    *(ckt->CKTstate0 + here->SOI3cgfdeltaT) =
                            *(ckt->CKTstate1 + here->SOI3cgfdeltaT);
                    *(ckt->CKTstate0 + here->SOI3cgfgb) =
                            *(ckt->CKTstate1 + here->SOI3cgfgb);
                    *(ckt->CKTstate0 + here->SOI3csgf) =
                            *(ckt->CKTstate1 + here->SOI3csgf);
                    *(ckt->CKTstate0 + here->SOI3csd) =
                            *(ckt->CKTstate1 + here->SOI3csd);
                    *(ckt->CKTstate0 + here->SOI3css) =
                            *(ckt->CKTstate1 + here->SOI3css);
                    *(ckt->CKTstate0 + here->SOI3csdeltaT) =
                            *(ckt->CKTstate1 + here->SOI3csdeltaT);
                    *(ckt->CKTstate0 + here->SOI3csgb) =
                            *(ckt->CKTstate1 + here->SOI3csgb);
                    *(ckt->CKTstate0 + here->SOI3cdgf) =
                            *(ckt->CKTstate1 + here->SOI3cdgf);
                    *(ckt->CKTstate0 + here->SOI3cdd) =
                            *(ckt->CKTstate1 + here->SOI3cdd);
                    *(ckt->CKTstate0 + here->SOI3cds) =
                            *(ckt->CKTstate1 + here->SOI3cds);
                    *(ckt->CKTstate0 + here->SOI3cddeltaT) =
                            *(ckt->CKTstate1 + here->SOI3cddeltaT);
                    *(ckt->CKTstate0 + here->SOI3cdgb) =
                            *(ckt->CKTstate1 + here->SOI3cdgb);
                    *(ckt->CKTstate0 + here->SOI3cgbgf) =
                            *(ckt->CKTstate1 + here->SOI3cgbgf);
                    *(ckt->CKTstate0 + here->SOI3cgbd) =
                            *(ckt->CKTstate1 + here->SOI3cgbd);
                    *(ckt->CKTstate0 + here->SOI3cgbs) =
                            *(ckt->CKTstate1 + here->SOI3cgbs);
                    *(ckt->CKTstate0 + here->SOI3cgbdeltaT) =
                            *(ckt->CKTstate1 + here->SOI3cgbdeltaT);
                    *(ckt->CKTstate0 + here->SOI3cgbgb) =
                            *(ckt->CKTstate1 + here->SOI3cgbgb);
                    *(ckt->CKTstate0 + here->SOI3cBJTbsbs) =
                            *(ckt->CKTstate1 + here->SOI3cBJTbsbs);
                    *(ckt->CKTstate0 + here->SOI3cBJTbsdeltaT) =
                            *(ckt->CKTstate1 + here->SOI3cBJTbsdeltaT);
                    *(ckt->CKTstate0 + here->SOI3cBJTbdbd) =
                            *(ckt->CKTstate1 + here->SOI3cBJTbdbd);
                    *(ckt->CKTstate0 + here->SOI3cBJTbddeltaT) =
                            *(ckt->CKTstate1 + here->SOI3cBJTbddeltaT);
                }
                else
                {
#endif /* PREDICTOR */

                    /* general iteration */

                    vbs = model->SOI3type * (
                        *(ckt->CKTrhsOld+here->SOI3bNode) -
                        *(ckt->CKTrhsOld+here->SOI3sNodePrime));
                    vgfs = model->SOI3type * (
                        *(ckt->CKTrhsOld+here->SOI3gfNode) -
                        *(ckt->CKTrhsOld+here->SOI3sNodePrime));
                    vgbs = model->SOI3type * (
                        *(ckt->CKTrhsOld+here->SOI3gbNode) -
                        *(ckt->CKTrhsOld+here->SOI3sNodePrime));
                    vds = model->SOI3type * (
                        *(ckt->CKTrhsOld+here->SOI3dNodePrime) -
                        *(ckt->CKTrhsOld+here->SOI3sNodePrime));
                    deltaT = MAX(0,*(ckt->CKTrhsOld+here->SOI3toutNode));
                     /* voltage deltaT is V(tout) wrt ground
                        and shoule be positive */
/* the next lot are needed for the (extra) thermal capacitances */
						  /* JimB - 19/5/99 */
                    if (here->SOI3numThermalNodes == 0)
                    {
                       deltaT1=deltaT2=deltaT3=deltaT4=deltaT5=0;
                    }
                    if (here->SOI3numThermalNodes == 1)
                    {
                       deltaT1=deltaT;
                       deltaT2=deltaT3=deltaT4=deltaT5=0;
                    }
                    if (here->SOI3numThermalNodes == 2)
                    {
                       deltaT2 = MAX(0,*(ckt->CKTrhsOld+here->SOI3tout1Node));
                       deltaT1 = deltaT - deltaT2;
                       deltaT3=deltaT4=deltaT5=0;
                    }
                    if (here->SOI3numThermalNodes == 3)
                    {
                       deltaT3 = MAX(0,*(ckt->CKTrhsOld+here->SOI3tout2Node));
                    	  deltaT2 = MAX(0,*(ckt->CKTrhsOld+here->SOI3tout1Node) - deltaT3);
                    	  deltaT1 = deltaT - deltaT2 - deltaT3;
                       deltaT4=deltaT5=0;
                    }
                    if (here->SOI3numThermalNodes == 4)
                    {
                       deltaT4 = MAX(0,*(ckt->CKTrhsOld+here->SOI3tout3Node));
                    	  deltaT3 = MAX(0,*(ckt->CKTrhsOld+here->SOI3tout2Node) - deltaT4);
                    	  deltaT2 = MAX(0,*(ckt->CKTrhsOld+here->SOI3tout1Node) - deltaT3 - deltaT4);
                    	  deltaT1 = deltaT - deltaT2 - deltaT3 - deltaT4;
                       deltaT5=0;
                    }
                    if (here->SOI3numThermalNodes == 5)
                    {
                       deltaT5 = MAX(0,*(ckt->CKTrhsOld+here->SOI3tout4Node));
                       deltaT4 = MAX(0,*(ckt->CKTrhsOld+here->SOI3tout3Node) - deltaT5);
                    	  deltaT3 = MAX(0,*(ckt->CKTrhsOld+here->SOI3tout2Node) - deltaT4 - deltaT5);
                    	  deltaT2 = MAX(0,*(ckt->CKTrhsOld+here->SOI3tout1Node) - deltaT3 - deltaT4 - deltaT5);
                    	  deltaT1 = deltaT - deltaT2 - deltaT3 - deltaT4 - deltaT5;
                    }
#ifndef PREDICTOR
                }
#endif /* PREDICTOR */


                /* now some common crunching for some more useful quantities */
                /* bloody useful in our case */

                vbd=vbs-vds;
                vdb=-vbd;
                vsb=-vbs;
                
                vgfd=vgfs-vds;
                vgbd=vgbs-vds;
                vgbb=vgbs-vbs;
                vgfdo = *(ckt->CKTstate0 + here->SOI3vgfs) -
                        *(ckt->CKTstate0 + here->SOI3vds);
                vgbdo = *(ckt->CKTstate0 + here->SOI3vgbs) -
                        *(ckt->CKTstate0 + here->SOI3vds);
                delvbs = vbs - *(ckt->CKTstate0 + here->SOI3vbs);
                delvbd = vbd - *(ckt->CKTstate0 + here->SOI3vbd);
                delvgfs = vgfs - *(ckt->CKTstate0 + here->SOI3vgfs);
                delvgbs = vgbs - *(ckt->CKTstate0 + here->SOI3vgbs);
                delvds = vds - *(ckt->CKTstate0 + here->SOI3vds);
                delvgfd = vgfd-vgfdo;
                delvgbd = vgbd-vgbdo;
                deldeltaT = deltaT - *(ckt->CKTstate0 + here->SOI3deltaT);

               /* these are needed for convergence testing */
               /* we're keeping these the same as convergence on
                  vgfs, vbs, vbd, vgbs, deltaT is equiv to conv.
                  on vgfb, vsb, vdb, vgb_b, deltaT         */

                if (here->SOI3mode >= 0) {  /* normal */
                    idhat=
                        here->SOI3id-
                        here->SOI3gbd * delvbd -
                        here->SOI3gbdT * deldeltaT +  /* for -ibd bit of id */
                        (here->SOI3gmbs +
                        here->SOI3gMmbs) * delvbs +
                        (here->SOI3gmf +
                        here->SOI3gMmf) * delvgfs +
                        (here->SOI3gmb +
                        here->SOI3gMmb) * delvgbs +
                        (here->SOI3gds +
                        here->SOI3gMd) * delvds +
                        (here->SOI3gt +
                        here->SOI3gMdeltaT) * deldeltaT +
                        here->SOI3gBJTdb_bs * delvbs +
                        here->SOI3gBJTdb_deltaT * deldeltaT;
                    ibhat=
                        here->SOI3ibs +
                        here->SOI3ibd +
                        here->SOI3gbd * delvbd +
                        here->SOI3gbdT * deldeltaT +
                        here->SOI3gbs * delvbs +
                        here->SOI3gbsT * deldeltaT -
                        here->SOI3iMdb -
                        here->SOI3gMmbs * delvbs -
                        here->SOI3gMmf * delvgfs -
                        here->SOI3gMmb * delvgbs -
                        here->SOI3gMd * delvds -
                        here->SOI3gMdeltaT * deldeltaT -
                        here->SOI3iBJTsb -
                        here->SOI3gBJTsb_bd * delvbd -
                        here->SOI3gBJTsb_deltaT * deldeltaT -
                        here->SOI3iBJTdb -
                        here->SOI3gBJTdb_bs * delvbs -
                        here->SOI3gBJTdb_deltaT * deldeltaT;
                } else {                   /* A over T */
                    idhat=
                        here->SOI3id -
                        ( here->SOI3gbd +
                        here->SOI3gmbs) * delvbd -
                        (here->SOI3gmf) * delvgfd -
                        (here->SOI3gmb) * delvgbd +
                        (here->SOI3gds) * delvds -
                        (here->SOI3gt +
                         here->SOI3gbdT) * deldeltaT +
                        here->SOI3gBJTdb_bs * delvbs +
                        here->SOI3gBJTdb_deltaT * deldeltaT;
                    ibhat=
                        here->SOI3ibs +
                        here->SOI3ibd +
                        here->SOI3gbd * delvbd +
                        here->SOI3gbdT * deldeltaT +
                        here->SOI3gbs * delvbs +
                        here->SOI3gbsT * deldeltaT -
                        here->SOI3iMsb -
                        here->SOI3gMmbs * delvbd -
                        here->SOI3gMmf * delvgfd -
                        here->SOI3gMmb * delvgbd +
                        here->SOI3gMd * delvds -   /* gMd should go with vsd */
                        here->SOI3gMdeltaT * deldeltaT -
                        here->SOI3iBJTsb -
                        here->SOI3gBJTsb_bd * delvbd -
                        here->SOI3gBJTsb_deltaT * deldeltaT -
                        here->SOI3iBJTdb -
                        here->SOI3gBJTdb_bs * delvbs -
                        here->SOI3gBJTdb_deltaT * deldeltaT;
                }
/* thermal current source comparator */
                iPthat =here->SOI3iPt +
                        here->SOI3gPmbs * delvbs +
                        here->SOI3gPmf  * delvgfs +
                        here->SOI3gPmb  * delvgbs +
                        here->SOI3gPds  * delvds * here->SOI3mode +
                        here->SOI3gPdT  * deldeltaT;

#ifndef NOBYPASS
                /* now lets see if we can bypass (ugh) */
                /* the following mess should be one if statement, but
                 * many compilers can't handle it all at once, so it
                 * is split into several successive if statements
                 */
                /* bypass just needs to check any four voltages have not changed
                   so leave as before to avoid hassle */
                tempv = MAX(fabs(ibhat),fabs(here->SOI3ibs
                        + here->SOI3ibd-here->SOI3iMsb
                        - here->SOI3iMdb - here->SOI3iBJTdb
                        - here->SOI3iBJTsb))+ckt->CKTabstol;
                if((!(ckt->CKTmode & (MODEINITPRED|MODEINITTRAN|MODEINITSMSIG)
                        )) && (ckt->CKTbypass) )
                if ( (fabs(ibhat-(here->SOI3ibs +
                        here->SOI3ibd-here->SOI3iMdb
                        - here->SOI3iMsb - here->SOI3iBJTdb
                        - here->SOI3iBJTsb)) < ckt->CKTreltol *
                        tempv))
                if( (fabs(delvbs) < (ckt->CKTreltol * MAX(fabs(vbs),
                        fabs(*(ckt->CKTstate0+here->SOI3vbs)))+
                        ckt->CKTvoltTol)))
                if ( (fabs(delvbd) < (ckt->CKTreltol * MAX(fabs(vbd),
                        fabs(*(ckt->CKTstate0+here->SOI3vbd)))+
                        ckt->CKTvoltTol)) )
                if( (fabs(delvgfs) < (ckt->CKTreltol * MAX(fabs(vgfs),
                        fabs(*(ckt->CKTstate0+here->SOI3vgfs)))+
                        ckt->CKTvoltTol)))
                if( (fabs(delvgbs) < (ckt->CKTreltol * MAX(fabs(vgbs),
                        fabs(*(ckt->CKTstate0+here->SOI3vgbs)))+
                        ckt->CKTvoltTol)))
                if ( (fabs(delvds) < (ckt->CKTreltol * MAX(fabs(vds),
                        fabs(*(ckt->CKTstate0+here->SOI3vds)))+
                        ckt->CKTvoltTol)) )
                if ( (fabs(deldeltaT) < (ckt->CKTreltol * MAX(fabs(deltaT),
                        fabs(*(ckt->CKTstate0+here->SOI3deltaT)))+
                        ckt->CKTvoltTol)) )
                if( (fabs(iPthat- here->SOI3iPt) <
                        ckt->CKTreltol * MAX(fabs(iPthat),fabs(
                        here->SOI3iPt)) + ckt->CKTabstol) )
                if( (fabs(idhat- here->SOI3id) <
                        ckt->CKTreltol * MAX(fabs(idhat),fabs(
                        here->SOI3id)) + ckt->CKTabstol) ) {
                    /* bypass code */
                    /* nothing interesting has changed since last
                     * iteration on this device, so we just
                     * copy all the values computed last iteration out
                     * and keep going
                     */
                    vbs = *(ckt->CKTstate0 + here->SOI3vbs);
                    vbd = *(ckt->CKTstate0 + here->SOI3vbd);
                    vgfs = *(ckt->CKTstate0 + here->SOI3vgfs);
                    vgbs = *(ckt->CKTstate0 + here->SOI3vgbs);
                    vds = *(ckt->CKTstate0 + here->SOI3vds);
                    deltaT = *(ckt->CKTstate0 + here->SOI3deltaT);
                    deltaT1 = *(ckt->CKTstate0 + here->SOI3deltaT1);
                    deltaT2 = *(ckt->CKTstate0 + here->SOI3deltaT2);
                    deltaT3 = *(ckt->CKTstate0 + here->SOI3deltaT3);
                    deltaT4 = *(ckt->CKTstate0 + here->SOI3deltaT4);
                    deltaT5 = *(ckt->CKTstate0 + here->SOI3deltaT5);
                    /* and now the extra ones */
                    vsb=-vbs;
                    vdb=-vbd;
                    vgfb = vgfs - vbs;
                    vgbb = vgbs - vbs;

                    /* JimB - 15/9/99 */
                    /* Code for multiple thermal time constants.  Start by moving all */
                    /* rt constants into arrays. */
                    rtargs[0]=here->SOI3rt;
                    rtargs[1]=here->SOI3rt1;
                    rtargs[2]=here->SOI3rt2;
                    rtargs[3]=here->SOI3rt3;
                    rtargs[4]=here->SOI3rt4;

                    /* Set all conductance components to zero. */
                    grt[0]=grt[1]=grt[2]=grt[3]=grt[4]=0.0;
                    /* Now calculate conductances from rt. */
                    /* Don't need to worry about divide by zero when calculating */
                    /* grt components, as soi3setup() only creates a thermal node */
                    /* if corresponding rt is greater than zero. */
                    for(tnodeindex=0;tnodeindex<here->SOI3numThermalNodes;tnodeindex++)
                    {
                       grt[tnodeindex]=1/rtargs[tnodeindex];
                    }
                    /* End JimB */

                    vgfd = vgfs - vds;
                    vgbd = vgbs - vds;
                    if (here->SOI3mode==1)
                    {
                        idrain =  here->SOI3id + here->SOI3ibd - here->SOI3iMdb
                                  - here->SOI3iBJTdb;
                    }
                    else
                    {
                    		idrain = -here->SOI3id - here->SOI3ibd
                                  + here->SOI3iBJTdb;
                    }
                    
                    /* Pt doesn't need changing as it's in here->SOI3iPt */

                    if((ckt->CKTmode & (MODETRAN | MODEAC)) || 
                            (ckt->CKTmode & MODETRANOP))
                    {
                      cgfgf = *(ckt->CKTstate0 + here->SOI3cgfgf);
                      cgfd  = *(ckt->CKTstate0 + here->SOI3cgfd);
                      cgfs  = *(ckt->CKTstate0 + here->SOI3cgfs);
                      cgfdeltaT  = *(ckt->CKTstate0 + here->SOI3cgfdeltaT);
                      cgfgb = *(ckt->CKTstate0 + here->SOI3cgfgb);
                      csgf = *(ckt->CKTstate0 + here->SOI3csgf);
                      csd  = *(ckt->CKTstate0 + here->SOI3csd);
                      css  = *(ckt->CKTstate0 + here->SOI3css);
                      csdeltaT  = *(ckt->CKTstate0 + here->SOI3csdeltaT);
                      csgb = *(ckt->CKTstate0 + here->SOI3csgb);
                      cdgf = *(ckt->CKTstate0 + here->SOI3cdgf);
                      cdd  = *(ckt->CKTstate0 + here->SOI3cdd);
                      cds  = *(ckt->CKTstate0 + here->SOI3cds);
                      cddeltaT  = *(ckt->CKTstate0 + here->SOI3cddeltaT);
                      cdgb  = *(ckt->CKTstate0 + here->SOI3cdgb);
                      cgbgf = *(ckt->CKTstate0 + here->SOI3cgbgf);
                      cgbd  = *(ckt->CKTstate0 + here->SOI3cgbd);
                      cgbs  = *(ckt->CKTstate0 + here->SOI3cgbs);
                      cgbdeltaT  = *(ckt->CKTstate0 + here->SOI3cgbdeltaT);
                      cgbgb = *(ckt->CKTstate0 + here->SOI3cgbgb);
                      cbgf = -(cgfgf + cdgf + csgf + cgbgf);
                      cbd = -(cgfd + cdd + csd + cgbd);
                      cbs = -(cgfs + cds + css + cgbs);
                      cbdeltaT = -(cgfdeltaT + cddeltaT + csdeltaT + cgbdeltaT);
                      cbgb = -(cgfgb + cdgb + csgb + cgbgb);
                      qgatef = *(ckt->CKTstate0 + here->SOI3qgf);
                      qdrn = *(ckt->CKTstate0 + here->SOI3qd);
                      qsrc = *(ckt->CKTstate0 + here->SOI3qs);
                      qgateb = *(ckt->CKTstate0 + here->SOI3qgb);
                      qbody = -(qgatef + qdrn + qsrc + qgateb);
                      ByPass = 1;
                      goto bypass1;
                    }   
                    goto bypass2;
                }
#endif /*NOBYPASS*/

                /* ok - bypass is out, do it the hard way */

                von = model->SOI3type * here->SOI3von;

#ifndef NODELIMITING
                /*
                 * limiting
                 *  we want to keep device voltages from changing
                 * so fast that the exponentials churn out overflows
                 * and similar rudeness
                 */

                if(*(ckt->CKTstate0 + here->SOI3vds) >=0)
                {
                    vgfs = DEVfetlim(vgfs,*(ckt->CKTstate0 + here->SOI3vgfs)
                            ,von);
                    vds = vgfs - vgfd;
                    vds = DEVlimvds(vds,*(ckt->CKTstate0 + here->SOI3vds));
                    vgfd = vgfs - vds;
                }
                else
                {
                    vgfd = DEVfetlim(vgfd,vgfdo,von);
                    vds = vgfs - vgfd;
                    if(!(ckt->CKTfixLimit))
                    {
                        vds = -DEVlimvds(-vds,-(*(ckt->CKTstate0 +
                                here->SOI3vds)));
                    }
                    vgfs = vgfd + vds;
                }
                if(vds >= 0)
                {
                    vbs = DEVsoipnjlim(vbs,*(ckt->CKTstate0 + here->SOI3vbs),
                            vt,here->SOI3sourceVcrit,&Check);
                    vbd = vbs-vds;
                }
                else
                {
                    vbd = DEVsoipnjlim(vbd,*(ckt->CKTstate0 + here->SOI3vbd),
                            vt,here->SOI3drainVcrit,&Check);
                    vbs = vbd + vds;
                }
                

                /* and now some limiting of the temperature rise */
                if (deltaT>(10 + *(ckt->CKTstate0 + here->SOI3deltaT)))
                {
                  deltaT = 10 + *(ckt->CKTstate0 + here->SOI3deltaT);

                  /* need limiting, therefore must also impose limits on other
                     thermal voltages.
                  */

                  /* JimB - 19/5/99 */
                  if (here->SOI3numThermalNodes == 0)
                  {
                  	deltaT1=deltaT2=deltaT3=deltaT4=deltaT5=0;
                  }
                  if (here->SOI3numThermalNodes == 1)
                  {
                  	deltaT1=deltaT;
                  	deltaT2=deltaT3=deltaT4=deltaT5=0;
                  }
                  if (here->SOI3numThermalNodes == 2)
                  {
                  	deltaT2 = MAX(0,*(ckt->CKTrhsOld+here->SOI3tout1Node));
                  	deltaT1 = deltaT - deltaT2;
                  	deltaT3=deltaT4=deltaT5=0;
                  }
                  if (here->SOI3numThermalNodes == 3)
                  {
                  	deltaT3 = MAX(0,*(ckt->CKTrhsOld+here->SOI3tout2Node));
                  	deltaT2 = MAX(0,*(ckt->CKTrhsOld+here->SOI3tout1Node) - deltaT3);
                  	deltaT1 = deltaT - deltaT2 - deltaT3;
                  	deltaT4=deltaT5=0;
                  }
                  if (here->SOI3numThermalNodes == 4)
                  {
                  	deltaT4 = MAX(0,*(ckt->CKTrhsOld+here->SOI3tout3Node));
                  	deltaT3 = MAX(0,*(ckt->CKTrhsOld+here->SOI3tout2Node) - deltaT4);
                  	deltaT2 = MAX(0,*(ckt->CKTrhsOld+here->SOI3tout1Node) - deltaT3 - deltaT4);
                  	deltaT1 = deltaT - deltaT2 - deltaT3 - deltaT4;
                  	deltaT5=0;
                  }
                  if (here->SOI3numThermalNodes == 5)
                  {
                  	deltaT5 = MAX(0,*(ckt->CKTrhsOld+here->SOI3tout4Node));
                  	deltaT4 = MAX(0,*(ckt->CKTrhsOld+here->SOI3tout3Node) - deltaT5);
                  	deltaT3 = MAX(0,*(ckt->CKTrhsOld+here->SOI3tout2Node) - deltaT4 - deltaT5);
                  	deltaT2 = MAX(0,*(ckt->CKTrhsOld+here->SOI3tout1Node) - deltaT3 - deltaT4 - deltaT5);
                  	deltaT1 = deltaT - deltaT2 - deltaT3 - deltaT4 - deltaT5;
                  }

                  Check = 1;
                }
#endif /*NODELIMITING*/

            } else {

                /* ok - not one of the simple cases, so we have to
                 * look at all of the possibilities for why we were
                 * called.  We still just initialize the three voltages
                 */

                if((ckt->CKTmode & MODEINITJCT) && !here->SOI3off)
                {
                    vds= model->SOI3type * here->SOI3icVDS;
                    vgfs= model->SOI3type * here->SOI3icVGFS;
                    vgbs= model->SOI3type * here->SOI3icVGBS;
                    vbs= model->SOI3type * here->SOI3icVBS;
                    deltaT=deltaT1=deltaT2=deltaT3=deltaT4=deltaT5=0.0;
                    if((vds==0) && (vgfs==0) && (vbs==0) && (vgbs==0) &&
                            ((ckt->CKTmode &
                                (MODETRAN|MODEDCOP|MODEDCTRANCURVE)) ||
                             (!(ckt->CKTmode & MODEUIC))))
                    {
                        vbs = -1;
                        vgfs = model->SOI3type * here->SOI3tVto;
                        vds = 0;
                        vgbs = 0;
                        deltaT=deltaT1=deltaT2=deltaT3=deltaT4=deltaT5=0.0;
                    }
                }
                else
                {
                   vbs=vgfs=vds=vgbs=deltaT=deltaT1=deltaT2=deltaT3=deltaT4=deltaT5=0.0;
                }
            }


            /*
             * now all the preliminaries are over - we can start doing the
             * real work
             */
            vbd = vbs - vds;
            vgfd = vgfs - vds;
            vgbd = vgbs - vds;
            vgfb = vgfs - vbs;
            vgbb = vgbs - vbs;
            vsb = -vbs;
            vdb = -vbd;
            

            /* now to determine whether the user was able to correctly
             * identify the source and drain of his device
             */
            if(vds >= 0) {
                /* normal mode */
                here->SOI3mode = 1;
            } else {
                /* inverse mode */
                here->SOI3mode = -1;
            }

            {             /* begin block  */

            /*
             *     This block works out drain current and derivatives.
             *     It does this via the calculation of the surface potential
             *     which is a bit complex mostly for reasons of continuity
             *     and robustness - c'est la vie.
             *     
             */

            /* the following variables are local to this code block until
             * it is obvious that they can be made global
             */
            double vg;
            double eta_s; /* 1+Citf/Cof */
            double gamma,sigma;
            double Egy = 0.0,vgy,Sgy;
            double psi_ss;
            double AL,A0,LL,L0;
            double ALx,A0x,EAL = 0.0,EA0 = 0.0;
            double logterm0,logtermL;
            double Vgconst0,VgconstL;
            double Egx0 = 0.0,EgxL = 0.0;
            double vgx0,vgxL;
            double psi_si0,psi_siL,psi_st0;
            double Esi0 = 0.0,EsiL = 0.0,Ess0 = 0.0;
            double theta2;
            double TVF; /* thermal velocity-saturation factor */
            double Mmob;
            double sqt0,sqt0one;
            double alpha,delta,Bf;
            double vGBF,vGT,vGBT;
            double PSI;
            double S;
            double psi_sLsat;
            double Esd0 = 0.0,EsdL = 0.0;

            double psi_s0,psi_sL;
            double fL,f0;
            double DfL_Dvgfb,DfL_Dvdb,DfL_Dvsb,DfL_DdeltaT;
            double Df0_Dvgfb,Df0_Dvdb,Df0_Dvsb,Df0_DdeltaT;
            double ld,Dld_Dvgfb,Dld_Dvdb,Dld_Dvsb,Dld_DdeltaT;

            double ich0;

            double pDpsi_si0_Dvgx0,pDpsi_siL_DvgxL;
            double Dvgx0_Dvgfb,pDvgx0_Dvg,Dvgx0_Dvsb,Dvgx0_Dvdb,Dvgx0_DdeltaT;
            double DvgxL_Dvgfb,pDvgxL_Dvg,DvgxL_Dvsb,DvgxL_Dvdb,DvgxL_DdeltaT;
            double pDpsi_si0_Dvsb,pDpsi_siL_Dvdb;
            double Dpsi_si0_Dvgfb,Dpsi_si0_Dvsb,Dpsi_si0_Dvdb;
            double Dpsi_siL_Dvgfb,Dpsi_siL_Dvsb,Dpsi_siL_Dvdb;
            double Dpsi_ss_Dvgfb,Dpsi_ss_Dvsb,Dpsi_ss_Dvdb;
            double Dpsi_st0_Dvgfb,Dpsi_st0_Dvsb,Dpsi_st0_Dvdb;
            double pDdelta_Dpsi_st0;
            double Dalpha_Dvgfb,Dalpha_Dvdb,Dalpha_Dvsb,Dalpha_DdeltaT;
            double DPSI_Dvgfb,DPSI_Dvdb,DPSI_Dvsb,DPSI_DdeltaT;
            double pDBf_Dpsi_st0,DvGT_Dvgfb,DvGT_Dvsb,DvGT_Dvdb,DvGT_DdeltaT;
            double D,DS_Dvgfb,DS_Dvsb,DS_Dvdb,DS_DdeltaT;
            double Dpsi_sLsat_Dvgfb,Dpsi_sLsat_Dvsb,Dpsi_sLsat_Dvdb;
            double gmg,gmd,gms,gmt;
            double Dpsi_si0_DdeltaT,Dpsi_siL_DdeltaT;
            double Dpsi_ss_DdeltaT,Dpsi_st0_DdeltaT;
            double Dpsi_sLsat_DdeltaT;
            double Dpsi_sL_Dvgfb,Dpsi_s0_Dvgfb,Dpsi_sL_Dvdb,Dpsi_s0_Dvdb;
            double Dpsi_sL_Dvsb,Dpsi_s0_Dvsb,Dpsi_sL_DdeltaT,Dpsi_s0_DdeltaT;

            double vdsat;
            double Dvdsat_Dvgfb,Dvdsat_Dvsb,Dvdsat_Dvdb,Dvdsat_DdeltaT;
            int    i;
            double vds2m,vdsat2m,Em,vdslim;
            double Dvdslim_Dvgfb,Dvdslim_Dvdb,Dvdslim_Dvsb,Dvdslim_DdeltaT;
            double Vmx;
            double DVmx_Dvgfb,DVmx_Dvdb,DVmx_Dvsb,DVmx_DdeltaT;
            double Vm1,Em1,Vm1x;
            double DVm1x_Dvgfb,DVm1x_Dvdb,DVm1x_Dvsb,DVm1x_DdeltaT;
            double vgeff,lm,lmeff,Elm,Dlm_Dvgfb,Dlm_Dvdb,Dlm_Dvsb,Dlm_DdeltaT;

            double Mminus1=0;
            double EM,betaM;
            double gMg,gMd,gMs;
            double Fm; /* mobility degradation factor */
            double Y; /* channel length modulation factor */
            double tmp,tmp1; /* temporary var to aid pre-calculation */
            double TMF; /* thermal mobility factor */
            int vgx0trans,vgxLtrans; /* flags to indicate if vg transform performed */
            int psi_s0_trans,psi_sL_trans=0;
            int Ess0_trans,Esd0_trans,EsdL_trans;
            int A0trans,ALtrans;

            double vT,EchiD,EchiD1;
            double BetaBJTeff;

/* Now we use a nasty trick - if device is A over T, must "swap" drain and source
   potentials.  we do a literal switch and change it back for the outside world. */
            if (here->SOI3mode == -1) {
              SWAP(double, vsb, vdb);
            }

/* Intrinsic Electrical Bit - has a bit of thermal due to TTC */            
            sigma = model->SOI3sigma/(EffectiveLength);
            eta_s = 1 + (model->SOI3C_ssf/model->SOI3frontOxideCapFactor);
            vg = vgfb - here->SOI3tVfbF * model->SOI3type
                      + sigma*(here->SOI3mode*vds)
                      - model->SOI3chiFB*deltaT;
            gamma = model->SOI3gamma * (1 + (model->SOI3deltaW)/(here->SOI3w))
                                     * (1 - (model->SOI3deltaL)/EffectiveLength);
/* must introduce some limiting to prevent exp overflow */
            if (vg > (vt*MAX_EXP_ARG)) {
              vgy = vg;
            } else {
              Egy = exp(vg/vt);
              vgy = vt*log(1+Egy);
            }
            Sgy = sqrt((vgy/eta_s) + (gamma*gamma)/(4*eta_s*eta_s));
            psi_ss = (Sgy-0.5*gamma/eta_s)*(Sgy-0.5*gamma/eta_s);
            A0 = here->SOI3tPhi + vsb;
            AL = here->SOI3tPhi + vdb;
            if (A0 > (vt*MAX_EXP_ARG)) {
              A0x = A0;
              A0trans = 0;
            } else {
              EA0 = exp(A0/vt);
              A0x = vt * log(1+EA0);
              A0trans = 1;
            }
            if (AL > (vt*MAX_EXP_ARG)) {
              ALx = AL;
              ALtrans = 0;
            } else {
              EAL = exp(AL/vt);
              ALx = vt * log(1+EAL);
              ALtrans = 1;
            }
/* if vg is very large then don't need to transform
   vg into vgx.
*/
            Vgconst0 = eta_s*A0x - vt*(eta_s - 1) + gamma*sqrt(A0x);
            if ((vg-Vgconst0) < (5 * vt * MAX_EXP_ARG)) {
              vgx0trans = 1;
              Egx0 = exp((vg - Vgconst0)/(5*vt));
              vgx0 = (5*vt) * log(1 + Egx0) + Vgconst0;
            } else {
              vgx0trans = 0; /* no transform performed */
              vgx0 = vg;
            }


            VgconstL = eta_s*ALx - vt*(eta_s - 1) + gamma*sqrt(ALx);
            if ((vg-VgconstL) < (5 * vt * MAX_EXP_ARG)) {
              vgxLtrans = 1;
              EgxL = exp((vg - VgconstL)/(5*vt));
              vgxL = (5*vt) * log(1 + EgxL) + VgconstL;
            } else {
              vgxLtrans = 0; /* no transform performed */
              vgxL = vg;
            }

            L0 = vgx0 - eta_s*(A0 - vt);
            LL = vgxL - eta_s*(AL - vt);
            logterm0 = ((L0/gamma)*(L0/gamma) - A0)/vt;
            logtermL = ((LL/gamma)*(LL/gamma) - AL)/vt;
            if (logterm0<=0) { /* can only happen due to numerical problems with sqrt*/
              psi_si0 = A0 + model->SOI3chiPHI*deltaT + vt*log(vt/(gamma*gamma));
            } else {
              psi_si0 = A0 + model->SOI3chiPHI*deltaT + vt*log(logterm0);
            }
            if (logtermL<=0) {
              psi_siL = AL + model->SOI3chiPHI*deltaT + vt*log(vt/(gamma*gamma));
            } else {
              psi_siL = AL + model->SOI3chiPHI*deltaT + vt*log(logtermL);
            }

            if ((psi_si0-psi_ss) < vt*MAX_EXP_ARG) {
              Ess0 = exp((psi_si0-psi_ss)/vt);
              Ess0_trans = 1;
              if (psi_si0 > (vt * MAX_EXP_ARG)) { /* psi_si0 is BIG */
                psi_st0 = psi_si0 - vt*log(1+Ess0);
                psi_s0_trans = 0;
              } else {
                Esi0 = exp(psi_si0/vt);
                psi_st0 = vt*log(1+(Esi0/(1+Ess0)));
                psi_s0_trans = 1;
              }
            } else {
              psi_st0 = psi_ss;
              Ess0_trans = 0;
              psi_s0_trans = 0;
            }

/* now for psi_sLsat - has thermal influence ! (if vel sat is included) */
            TMF = exp(-model->SOI3k*log(1+(deltaT/here->SOI3temp)));
            sqt0 = sqrt(psi_st0+1E-25);
            sqt0one = sqrt(1+psi_st0);
/* delta is formulated this way so we can change it here and nowhere else */
            delta = 0.5/sqt0one;
            pDdelta_Dpsi_st0 = -0.25/((1+psi_st0)*sqt0one);
            alpha = eta_s+gamma*delta;
            vGBF = vg - psi_st0;
            Bf = sqt0 - delta*psi_st0;
            vGT = vg - gamma*Bf;
            vGBT =  vGT + alpha*vt;

            if(model->SOI3vsat == 0) {
              theta2 = 0;
              TVF = 0.0; /* could be any nominal value */
            } else {
              TVF  = 0.8*exp((here->SOI3temp+deltaT)/600);
              theta2 = (here->SOI3tSurfMob/(model->SOI3vsat))*1e-2 /*cm to m*/
                       *TMF*(1+TVF)/(EffectiveLength*(1+model->SOI3TVF0));
                        /* theta2=1/(L.Ec) */
            }
            Mmob = theta2 - 0.5*model->SOI3theta;

            PSI = MAX(0,(vGT/alpha) - psi_st0);
            S = 0.5*(1+sqrt(1 + (2*Mmob*PSI)/(1+model->SOI3theta*vGBF)));
            psi_sLsat = psi_st0 + PSI/S;

            if ((psi_si0 - psi_sLsat)<(vt*MAX_EXP_ARG)) {
              Esd0 = exp((psi_si0 - psi_sLsat)/vt);
              Esd0_trans = 1;
              if (psi_s0_trans) {
                psi_s0 = vt*log(1+Esi0/(1+Esd0));
              } else {
                psi_s0 = psi_si0 - vt*log(1+Esd0);
              }
            } else {
              psi_s0 = psi_sLsat;
              Esd0_trans = 0;
            }

            if ((psi_siL - psi_sLsat)<(vt*MAX_EXP_ARG)) {
              EsdL = exp((psi_siL - psi_sLsat)/vt);
              EsdL_trans = 1;
              if (psi_siL < (vt * MAX_EXP_ARG)) {
                EsiL = exp(psi_siL/vt);
                psi_sL = vt*log(1+EsiL/(1+EsdL));
                psi_sL_trans = 1;
              } else {
                psi_sL = psi_siL - vt*log(1+EsdL);
                psi_sL_trans = 0;
              }
            } else {
              psi_sL = psi_sLsat;
              EsdL_trans = 0;
            }
/* if, after all that, we have impossible situation due to numerical limiting schemes ... */

            if (psi_s0>psi_sL) {
              psi_sL = psi_s0;
            }

/* now we can finally work out surface potential for source and drain end */

/* now surface potential is ready */

            f0 = (vGBT - 0.5*alpha*psi_s0)*psi_s0;
            fL = (vGBT - 0.5*alpha*psi_sL)*psi_sL;

            ich0 = Beta * (fL - f0); /* This is "intrinsic" bit */

/* now for derivatives - they're a bit of a nightmare   *
 * notation  pD... means PARTIAL derivative whilst D... *
 * means FULL derivative                                */

/* first sub-threshold */

            if (vg > (vt*MAX_EXP_ARG)) {
              Dpsi_ss_Dvgfb = (1-0.5*gamma/(eta_s*Sgy))/eta_s;
            } else {
              Dpsi_ss_Dvgfb = (1-0.5*gamma/(eta_s*Sgy))*(Egy/(1+Egy))/eta_s;
            }
            Dpsi_ss_Dvsb  = Dpsi_ss_Dvgfb*(-sigma);
            Dpsi_ss_Dvdb  = Dpsi_ss_Dvgfb*(sigma);
            Dpsi_ss_DdeltaT = Dpsi_ss_Dvgfb*(-model->SOI3chiFB);

/* now for strong inversion */

            pDpsi_si0_Dvgx0 = 2*vt*L0/(L0*L0-gamma*gamma*A0);
            pDpsi_siL_DvgxL = 2*vt*LL/(LL*LL-gamma*gamma*AL);

  /* if vg is transformed, must have deriv accordingly, else it is 1 */

            if (vgx0trans) {
              Dvgx0_Dvgfb = pDvgx0_Dvg  = Egx0/(1+Egx0);
              if (A0trans) {
                /* JimB - 27/8/98 */
                /* Get divide by zero errors when A0x is equal to zero */
                /* Temporary fix - add small constant to denominator */
/*                Dvgx0_Dvsb  = pDvgx0_Dvg * (-sigma) +
                               (1-pDvgx0_Dvg)*(eta_s+0.5*gamma/sqrt(A0x))*
                               (EA0/(1+EA0)); */
                Dvgx0_Dvsb  = pDvgx0_Dvg * (-sigma) +
                               (1-pDvgx0_Dvg)*(eta_s+0.5*gamma/(sqrt(A0x)+1e-25))*
                               (EA0/(1+EA0));
                /* End JimB */
              } else {
                /* JimB - 27/8/97 */
                /* Temporary fix - add small constant to denominator */
                /*Dvgx0_Dvsb  = pDvgx0_Dvg * (-sigma) +
                               (1-pDvgx0_Dvg)*(eta_s+0.5*gamma/sqrt(A0)); */
                Dvgx0_Dvsb  = pDvgx0_Dvg * (-sigma) +
                               (1-pDvgx0_Dvg)*(eta_s+0.5*gamma/(sqrt(A0)+1e-25));
                /* End JimB */
              }
              Dvgx0_Dvdb  = pDvgx0_Dvg*(sigma);
              Dvgx0_DdeltaT  = pDvgx0_Dvg*(-model->SOI3chiFB);
            } else {
              Dvgx0_Dvgfb = pDvgx0_Dvg  = 1;
              Dvgx0_Dvsb  = (-sigma);
              Dvgx0_Dvdb  = (sigma);
              Dvgx0_DdeltaT  = (-model->SOI3chiFB);
            }

            if (vgxLtrans) {
              DvgxL_Dvgfb = pDvgxL_Dvg = EgxL/(1+EgxL);
              DvgxL_Dvsb  = pDvgxL_Dvg * (-sigma);
              if (ALtrans) {
                /* JimB - 27/8/98 */
                /* Get divide by zero errors when ALx is equal to zero */
                /* Temporary fix - add small constant to denominator */
                /*DvgxL_Dvdb  = pDvgxL_Dvg * (sigma) +
                               (1-pDvgxL_Dvg)*(eta_s+0.5*gamma/sqrt(ALx))*
                               (EAL/(1+EAL)); */
                DvgxL_Dvdb  = pDvgxL_Dvg * (sigma) +
                               (1-pDvgxL_Dvg)*(eta_s+0.5*gamma/(sqrt(ALx)+1e-25))*
                               (EAL/(1+EAL));
                /* End JimB */
              } else {
                /* JimB - 27/8/97 */
                /* Temporary fix - add small constant to denominator */
/*                DvgxL_Dvdb  = pDvgxL_Dvg * (sigma) +
                               (1-pDvgxL_Dvg)*(eta_s+0.5*gamma/sqrt(AL)); */
                DvgxL_Dvdb  = pDvgxL_Dvg * (sigma) +
                               (1-pDvgxL_Dvg)*(eta_s+0.5*gamma/(sqrt(AL)+1e-25));
                /* End JimB */
              }
              DvgxL_DdeltaT  = pDvgxL_Dvg * (-model->SOI3chiFB);
            } else {
              DvgxL_Dvgfb = 1;
              DvgxL_Dvsb  = (-sigma);
              DvgxL_Dvdb  = (sigma);
              DvgxL_DdeltaT  = (-model->SOI3chiFB);
            }

            pDpsi_si0_Dvsb = 1 - (2*vt*eta_s*L0+vt*gamma*gamma)/(L0*L0-gamma*gamma*A0);
            pDpsi_siL_Dvdb = 1 - (2*vt*eta_s*LL+vt*gamma*gamma)/(LL*LL-gamma*gamma*AL);

            Dpsi_si0_Dvgfb = pDpsi_si0_Dvgx0*Dvgx0_Dvgfb;
            Dpsi_si0_Dvsb =  pDpsi_si0_Dvgx0*Dvgx0_Dvsb + pDpsi_si0_Dvsb;
            Dpsi_si0_Dvdb =  pDpsi_si0_Dvgx0*Dvgx0_Dvdb;
            Dpsi_si0_DdeltaT =  model->SOI3chiPHI +
                                pDpsi_si0_Dvgx0*Dvgx0_DdeltaT;

            Dpsi_siL_Dvgfb = pDpsi_siL_DvgxL*DvgxL_Dvgfb;
            Dpsi_siL_Dvsb =  pDpsi_siL_DvgxL*DvgxL_Dvsb;
            Dpsi_siL_Dvdb =  pDpsi_siL_DvgxL*DvgxL_Dvdb + pDpsi_siL_Dvdb;
            Dpsi_siL_DdeltaT =  model->SOI3chiPHI +
                                pDpsi_siL_DvgxL*DvgxL_DdeltaT;

/* now we can get full deriv of first guess 
   but also, partials of psi_s0 and psi_sL, the
   final values have similar structure, so do
   them now as well */

  /* deriv of psi_s etc wrt psi_si must be according to transform used */

            if (Ess0_trans) {
              if (psi_s0_trans) {
                tmp  = (Esi0/(1+Ess0+Esi0))/(1+Ess0);
                tmp1 = (Esi0/(1+Ess0+Esi0))*(Ess0/(1+Ess0));
              } else {
                tmp  = 1/(1+Ess0);
                tmp1 = Ess0*tmp;
              }
              Dpsi_st0_Dvgfb = tmp*Dpsi_si0_Dvgfb + tmp1*Dpsi_ss_Dvgfb;
              Dpsi_st0_Dvsb  = tmp*Dpsi_si0_Dvsb + tmp1*Dpsi_ss_Dvsb;
              Dpsi_st0_Dvdb  = tmp*Dpsi_si0_Dvdb + tmp1*Dpsi_ss_Dvdb;
              Dpsi_st0_DdeltaT = tmp*Dpsi_si0_DdeltaT + tmp1*Dpsi_ss_DdeltaT;
            } else {
              Dpsi_st0_Dvgfb = Dpsi_ss_Dvgfb;
              Dpsi_st0_Dvdb = Dpsi_ss_Dvdb;
              Dpsi_st0_Dvsb = Dpsi_ss_Dvsb;
              Dpsi_st0_DdeltaT = Dpsi_ss_DdeltaT;
            }

/* now some itsy bitsy quantities useful all over the shop */
  /* Ddelta_Dpsi_st0 is defined earlier with delta to allow
     change of delta expression in one place */

            pDBf_Dpsi_st0 = 0.5/sqt0 - psi_st0*pDdelta_Dpsi_st0 - delta;

            DvGT_Dvgfb = 1 - gamma*pDBf_Dpsi_st0*Dpsi_st0_Dvgfb;
            DvGT_Dvsb  = (-sigma) - gamma*pDBf_Dpsi_st0*Dpsi_st0_Dvsb;
            DvGT_Dvdb  = (sigma) - gamma*pDBf_Dpsi_st0*Dpsi_st0_Dvdb;
            DvGT_DdeltaT  = (-model->SOI3chiFB) - gamma*pDBf_Dpsi_st0*Dpsi_st0_DdeltaT;

            Dalpha_Dvgfb = gamma*pDdelta_Dpsi_st0*Dpsi_st0_Dvgfb;
            Dalpha_Dvsb  = gamma*pDdelta_Dpsi_st0*Dpsi_st0_Dvsb;
            Dalpha_Dvdb  = gamma*pDdelta_Dpsi_st0*Dpsi_st0_Dvdb;
            Dalpha_DdeltaT  = gamma*pDdelta_Dpsi_st0*Dpsi_st0_DdeltaT;

/* Now for saturation stuff psi_sLsat */
/* NB no need for special case, theta2=0 ==> Mmob=-theta/2 */

            if (PSI != 0) { /* stops unnecessary math if PSI = 0 */
              DPSI_Dvgfb = (DvGT_Dvgfb - (vGT/alpha)*Dalpha_Dvgfb
                           )/alpha - Dpsi_st0_Dvgfb;
              DPSI_Dvsb = (DvGT_Dvsb - (vGT/alpha)*Dalpha_Dvsb
                           )/alpha - Dpsi_st0_Dvsb;
              DPSI_Dvdb = (DvGT_Dvdb - (vGT/alpha)*Dalpha_Dvdb
                           )/alpha - Dpsi_st0_Dvdb;
              DPSI_DdeltaT = (DvGT_DdeltaT - (vGT/alpha)*Dalpha_DdeltaT
                           )/alpha - Dpsi_st0_DdeltaT;

              D = 2*(1+model->SOI3theta*vGBF)*
                  sqrt(1 + (2*Mmob*PSI)/(1+model->SOI3theta*vGBF));

              DS_Dvgfb = (Mmob/D)*DPSI_Dvgfb -
                         (Mmob*PSI*(model->SOI3theta)/(D*(1+model->SOI3theta*vGBF)))
                                   *(1-Dpsi_st0_Dvgfb);
              DS_Dvsb = (Mmob/D)*DPSI_Dvsb -
                        (Mmob*PSI*(model->SOI3theta)/(D*(1+model->SOI3theta*vGBF)))
                                   *((-sigma)-Dpsi_st0_Dvsb);
              DS_Dvdb = (Mmob/D)*DPSI_Dvdb -
                        (Mmob*PSI*(model->SOI3theta)/(D*(1+model->SOI3theta*vGBF)))
                                   *((sigma)-Dpsi_st0_Dvdb);
              DS_DdeltaT = (Mmob/D)*DPSI_DdeltaT -
                           (Mmob*PSI*(model->SOI3theta)/(D*(1+model->SOI3theta*vGBF)))
                                      *((-model->SOI3chiFB)-Dpsi_st0_DdeltaT) -
                                (PSI/D)*theta2*(model->SOI3k/(deltaT+here->SOI3temp) -
                                                  TVF/(600*(1+TVF)));

              Dpsi_sLsat_Dvgfb = Dpsi_st0_Dvgfb +
                                 DPSI_Dvgfb/S -
                                 PSI*DS_Dvgfb/(S*S);
              Dpsi_sLsat_Dvsb  = Dpsi_st0_Dvsb +
                                 DPSI_Dvsb/S -
                                 PSI*DS_Dvsb/(S*S);
              Dpsi_sLsat_Dvdb  = Dpsi_st0_Dvdb +
                                 DPSI_Dvdb/S -
                                 PSI*DS_Dvdb/(S*S);
              Dpsi_sLsat_DdeltaT  = Dpsi_st0_DdeltaT +
                                    DPSI_DdeltaT/S -
                                    PSI*DS_DdeltaT/(S*S);
            } else {
              Dpsi_sLsat_Dvgfb = Dpsi_st0_Dvgfb;
              Dpsi_sLsat_Dvsb  = Dpsi_st0_Dvsb;
              Dpsi_sLsat_Dvdb  = Dpsi_st0_Dvdb;
              Dpsi_sLsat_DdeltaT  = Dpsi_st0_DdeltaT;
            }

            if (Esd0_trans) {
              if (psi_s0_trans) {
                tmp  = (Esi0/(1+Esd0+Esi0))/(1+Esd0);
                tmp1 = (Esi0/(1+Esd0+Esi0))*(Esd0/(1+Esd0));
              } else {
                tmp  = 1/(1+Esd0);
                tmp1 = Esd0*tmp;
              }
              Dpsi_s0_Dvgfb = tmp*Dpsi_si0_Dvgfb + tmp1*Dpsi_sLsat_Dvgfb;
              Dpsi_s0_Dvdb = tmp*Dpsi_si0_Dvdb + tmp1*Dpsi_sLsat_Dvdb;
              Dpsi_s0_Dvsb = tmp*Dpsi_si0_Dvsb + tmp1*Dpsi_sLsat_Dvsb;
              Dpsi_s0_DdeltaT = tmp*Dpsi_si0_DdeltaT + tmp1*Dpsi_sLsat_DdeltaT;
            } else {
              Dpsi_s0_Dvgfb = Dpsi_sLsat_Dvgfb;
              Dpsi_s0_Dvdb = Dpsi_sLsat_Dvdb;
              Dpsi_s0_Dvsb = Dpsi_sLsat_Dvsb;
              Dpsi_s0_DdeltaT = Dpsi_sLsat_DdeltaT;
            }

            if (EsdL_trans) {
              if (psi_sL_trans) {
                tmp  = (EsiL/(1+EsdL+EsiL))/(1+EsdL);
                tmp1 = (EsiL/(1+EsdL+EsiL))*(EsdL/(1+EsdL));
              } else {
                tmp  = 1/(1+EsdL);
                tmp1 = EsdL*tmp;
              }
              Dpsi_sL_Dvgfb = tmp*Dpsi_siL_Dvgfb + tmp1*Dpsi_sLsat_Dvgfb;
              Dpsi_sL_Dvdb = tmp*Dpsi_siL_Dvdb + tmp1*Dpsi_sLsat_Dvdb;
              Dpsi_sL_Dvsb = tmp*Dpsi_siL_Dvsb + tmp1*Dpsi_sLsat_Dvsb;
              Dpsi_sL_DdeltaT = tmp*Dpsi_siL_DdeltaT + tmp1*Dpsi_sLsat_DdeltaT;
            } else {
              Dpsi_sL_Dvgfb = Dpsi_sLsat_Dvgfb;
              Dpsi_sL_Dvdb = Dpsi_sLsat_Dvdb;
              Dpsi_sL_Dvsb = Dpsi_sLsat_Dvsb;
              Dpsi_sL_DdeltaT = Dpsi_sLsat_DdeltaT;
            }

/* now for the whole kaboodle */

            DfL_Dvgfb = psi_sL*(DvGT_Dvgfb + vt*Dalpha_Dvgfb) -
                        psi_sL*psi_sL*0.5*Dalpha_Dvgfb +
                        (vGBT - alpha*psi_sL)*Dpsi_sL_Dvgfb;
            Df0_Dvgfb = psi_s0*(DvGT_Dvgfb + vt*Dalpha_Dvgfb) -
                        psi_s0*psi_s0*0.5*Dalpha_Dvgfb +
                        (vGBT - alpha*psi_s0)*Dpsi_s0_Dvgfb;

            DfL_Dvdb  = psi_sL*(DvGT_Dvdb + vt*Dalpha_Dvdb) -
                        psi_sL*psi_sL*0.5*Dalpha_Dvdb +
                        (vGBT - alpha*psi_sL)*Dpsi_sL_Dvdb;
            Df0_Dvdb  = psi_s0*(DvGT_Dvdb + vt*Dalpha_Dvdb) -
                        psi_s0*psi_s0*0.5*Dalpha_Dvdb +
                        (vGBT - alpha*psi_s0)*Dpsi_s0_Dvdb;

            DfL_Dvsb  = psi_sL*(DvGT_Dvsb + vt*Dalpha_Dvsb) -
                        psi_sL*psi_sL*0.5*Dalpha_Dvsb +
                        (vGBT - alpha*psi_sL)*Dpsi_sL_Dvsb;
            Df0_Dvsb  = psi_s0*(DvGT_Dvsb + vt*Dalpha_Dvsb) -
                        psi_s0*psi_s0*0.5*Dalpha_Dvsb +
                        (vGBT - alpha*psi_s0)*Dpsi_s0_Dvsb;
            
            DfL_DdeltaT = psi_sL*(DvGT_DdeltaT + vt*Dalpha_DdeltaT) -
                          psi_sL*psi_sL*0.5*Dalpha_DdeltaT +
                          (vGBT - alpha*psi_sL)*Dpsi_sL_DdeltaT;
            Df0_DdeltaT = psi_s0*(DvGT_DdeltaT + vt*Dalpha_DdeltaT) -
                          psi_s0*psi_s0*0.5*Dalpha_DdeltaT +
                          (vGBT - alpha*psi_s0)*Dpsi_s0_DdeltaT;

/* put them all together and what have you got ...? */

            gmg = Beta*(DfL_Dvgfb - Df0_Dvgfb);
                 
            gmd = Beta*(DfL_Dvdb - Df0_Dvdb);
                 
            gms = Beta*(DfL_Dvsb - Df0_Dvsb);

            gmt = Beta*(DfL_DdeltaT - Df0_DdeltaT);

/* End Intrinsic Electrical Bit */


/* 
 * High Field Mobility Effects
 */
                Fm = 1 + model->SOI3theta*(vg - 0.5*(psi_sL + psi_s0)) +
                     theta2 * (psi_sL - psi_s0);
                ich0 = ich0/Fm;
                here->SOI3ueff = here->SOI3ueff/Fm;
                gmg = (gmg-ich0*(model->SOI3theta*
                                  (1-0.5*(Dpsi_sL_Dvgfb + Dpsi_s0_Dvgfb)) + 
                                  theta2*(Dpsi_sL_Dvgfb - Dpsi_s0_Dvgfb)
                                )
                      )/Fm;
                gmd = (gmd-ich0*(model->SOI3theta*
                                  (sigma-0.5*(Dpsi_sL_Dvdb + Dpsi_s0_Dvdb)) + 
                                  theta2*(Dpsi_sL_Dvdb - Dpsi_s0_Dvdb)
                                )
                      )/Fm;
                gms = (gms-ich0*(model->SOI3theta*
                                  (-sigma-0.5*(Dpsi_sL_Dvsb + Dpsi_s0_Dvsb)) +
                                  theta2*(Dpsi_sL_Dvsb - Dpsi_s0_Dvsb)
                                )
                      )/Fm;
                gmt = (gmt-ich0*(model->SOI3theta*
                                 ((-model->SOI3chiFB)-0.5*(Dpsi_sL_DdeltaT + Dpsi_s0_DdeltaT)
                                 ) + 
                                  theta2*(Dpsi_sL_DdeltaT - Dpsi_s0_DdeltaT -
                                               (psi_sL - psi_s0)*(model->SOI3k/(deltaT+here->SOI3temp) -
                                                                  TVF/(600*(1+TVF))
                                                                 )
                                         )
                                )
                      )/Fm;

/* 
 * End High Field Mobility Effects
 */

/* Now to define bits which affect the drain region */
/*
 * Channel Length Modulation
 */

						/* JimB - add thermal voltage to vdsat to ensure it remains above zero in subthreshold */
 						vdsat = psi_sLsat - psi_st0 + vt;
            		Dvdsat_Dvgfb = Dpsi_sLsat_Dvgfb - Dpsi_st0_Dvgfb;
            		Dvdsat_Dvsb = Dpsi_sLsat_Dvsb - Dpsi_st0_Dvsb;
            		Dvdsat_Dvdb = Dpsi_sLsat_Dvdb - Dpsi_st0_Dvdb;
            		Dvdsat_DdeltaT = Dpsi_sLsat_DdeltaT - Dpsi_st0_DdeltaT;

   					m = model->SOI3mexp;

   					if (m>0)
                  {
                     if (vdsat>0)
                     {
   					      vds2m = 1;
   					      vdsat2m = 1;
   					      for (i=0; i<2*m; i=i+1)
                        {
   					         vds2m = vds2m*(vds*here->SOI3mode);
   						      vdsat2m = vdsat2m*vdsat;
                        }
   					      Em = exp(-log(vds2m+vdsat2m)/(2*m));
   					      vdslim = (here->SOI3mode*vds)*vdsat*Em;
      				      Dvdslim_Dvgfb = (here->SOI3mode*vds*Em*Dvdsat_Dvgfb*vds2m)/(vds2m+vdsat2m);
      				      Dvdslim_Dvdb = (here->SOI3mode*vds*Em*Dvdsat_Dvdb*vds2m + vdsat*Em*vdsat2m)/(vds2m+vdsat2m);
      				      Dvdslim_Dvsb = (here->SOI3mode*vds*Em*Dvdsat_Dvsb*vds2m - vdsat*Em*vdsat2m)/(vds2m+vdsat2m);
 						      Dvdslim_DdeltaT = (here->SOI3mode*vds*Em*Dvdsat_DdeltaT*vds2m)/(vds2m+vdsat2m);

                  	   Vmx = (here->SOI3mode*vds) - vdslim;
                		   DVmx_Dvgfb = -Dvdslim_Dvgfb;
                		   DVmx_Dvdb = 1 - Dvdslim_Dvdb;
                		   DVmx_Dvsb = -1 - Dvdslim_Dvsb;
                		   DVmx_DdeltaT = -Dvdslim_DdeltaT;
                     }
                     else
                     {
                  	   Vmx = 0;
                		   DVmx_Dvgfb = 0;
                		   DVmx_Dvdb = 0;
                		   DVmx_Dvsb = 0;
                		   DVmx_DdeltaT = 0;
                     }
                  }
                  else
                  {
                     Vmx = (here->SOI3mode*vds) - vdsat;
                     DVmx_Dvgfb = -Dvdsat_Dvgfb;
                     DVmx_Dvdb = 1 - Dvdsat_Dvdb;
                     DVmx_Dvsb = -1 - Dvdsat_Dvsb;
                     DVmx_DdeltaT = -Dvdsat_DdeltaT;
						}

                  if (model->SOI3useLAMBDA)
                  {
                     ld = model->SOI3lambda*Vmx;
                     tmp = model->SOI3lambda;
                  }
                  else
                  {
                     ld = model->SOI3lx * log(1 + Vmx/model->SOI3vp);
                     tmp = model->SOI3lx/(model->SOI3vp + Vmx);
                  }

                  Y = 1 + (ld/EffectiveLength);

                  Dld_Dvgfb = tmp * DVmx_Dvgfb;
                  Dld_Dvdb = tmp * DVmx_Dvdb;
                  Dld_Dvsb = tmp * DVmx_Dvsb;
                  Dld_DdeltaT = tmp * DVmx_DdeltaT;

                  gmg = gmg * Y + (ich0/EffectiveLength) * Dld_Dvgfb;
                  gmd = gmd * Y + (ich0/EffectiveLength) * Dld_Dvdb;
                  gms = gms * Y + (ich0/EffectiveLength) * Dld_Dvsb;
                  gmt = gmt * Y + (ich0/EffectiveLength) * Dld_DdeltaT;

                  ich0 = ich0 * Y;
                  /* Need to do ich0 last as its old value is needed for gds */

/*
 * End Channel Length Modulation
 */

                here->SOI3gdsnotherm = here->SOI3gds;
                
                /************** Thermal Mobility Stuff **************/

                /* thermal effect on intrinsic electrical circuit */
                idrain = ich0 * TMF; /* idrain has new value now */
                here->SOI3ueff *= TMF;
                gmg = gmg * TMF;
/*                here->SOI3gmb = here->SOI3gmb * TMF; */
                gms = gms * TMF;
                gmd = gmd * TMF;
                           /* deltaT is indpt voltage now */
                gmt = gmt*TMF - (model->SOI3k/(deltaT+here->SOI3temp)) * idrain;
                ich0 = idrain;

                /*
                 *     finished intrinsic electrical
                 */
/*
 * Impact Ionisation current sources
 */

                Vm1 = (here->SOI3mode*vds) + model->SOI3eta*(psi_s0 - psi_sLsat);
                if (Vm1 > (vt*MAX_EXP_ARG)) {
                  Vm1x = Vm1;
                  tmp = 1;
                } else {
                  Em1 = exp(MIN(MAX_EXP_ARG,Vm1/vt));
                  Vm1x = vt * log(1 + Em1) + 1e-25;
                  tmp = (Em1/(1+Em1));
                }
                DVm1x_Dvgfb = tmp*model->SOI3eta*(Dpsi_s0_Dvgfb - Dpsi_sLsat_Dvgfb);
                DVm1x_Dvdb = tmp*(model->SOI3eta*(Dpsi_s0_Dvdb - Dpsi_sLsat_Dvdb) + 1);
                DVm1x_Dvsb = tmp*(model->SOI3eta*(Dpsi_s0_Dvsb - Dpsi_sLsat_Dvsb) - 1);
                DVm1x_DdeltaT = tmp*model->SOI3eta*(Dpsi_s0_DdeltaT - Dpsi_sLsat_DdeltaT);

                vgeff = vg - vsb - eta_s*here->SOI3tPhi - gamma*sqrt(here->SOI3tPhi);
                lm = model->SOI3lm + model->SOI3lm1*(here->SOI3mode*vds-vgeff) 
                     + model->SOI3lm2*(here->SOI3mode*vds-vgeff)*(here->SOI3mode*vds-vgeff);
                Elm = exp(MIN(MAX_EXP_ARG,lm/MAX(1e-10,(model->SOI3lm/40))));
                lmeff = (model->SOI3lm/40)*log(1+Elm);
                betaM = 100*(model->SOI3beta0 + model->SOI3chibeta*deltaT);
                EM = exp(MIN(MAX_EXP_ARG,-(lmeff*betaM)/Vm1x));
                Mminus1 = (100*model->SOI3alpha0/betaM) * Vm1x * EM;
                if (here->SOI3mode==1) {
                  here->SOI3iMdb=Mminus1*ich0;
                  here->SOI3iMsb=0;
                } else {
                  here->SOI3iMsb=Mminus1*ich0;
                  here->SOI3iMdb=0;
                }
                tmp = (Elm/(1+Elm))*
                      (model->SOI3lm1 + 2*model->SOI3lm2*(here->SOI3mode*vds-vgeff));
                Dlm_Dvgfb = -tmp;
                Dlm_Dvdb = tmp*(1-sigma);
                Dlm_Dvsb = tmp*(sigma);
                Dlm_DdeltaT = tmp*(model->SOI3chiFB);
                tmp = (ich0/Vm1x);
                tmp1 = (1+(lmeff*betaM/Vm1x));
                gMg = Mminus1 * (gmg + tmp * (tmp1*DVm1x_Dvgfb - betaM*Dlm_Dvgfb));
                gMd = Mminus1 * (gmd + tmp * (tmp1*DVm1x_Dvdb - betaM*Dlm_Dvdb));
                gMs = Mminus1 * (gms + tmp * (tmp1*DVm1x_Dvsb - betaM*Dlm_Dvsb));
                
                here->SOI3gMmf = gMg;
                here->SOI3gMmb = 0;
                here->SOI3gMmbs= -(gMs + gMg + gMd);
                here->SOI3gMd  = gMd;
                here->SOI3gMdeltaT = Mminus1*(gmt+
                                     tmp*(tmp1*DVm1x_DdeltaT - betaM*Dlm_DdeltaT - 
                                           lmeff*model->SOI3chibeta*100)
                                           - (ich0*model->SOI3chibeta*100/betaM)
                                             );          

/*
 * End Impact Ionisation current sources
 */

/***** time to convert to conventional names for (trans)conductances  *****/

                here->SOI3gmf = gmg;
                here->SOI3gmb = 0; /* FOR NOW */
                here->SOI3gmbs = -(gms + gmg + gmd);
                here->SOI3gds = gmd;
                here->SOI3gt = gmt;
                vdsat_ext = psi_sLsat - psi_s0;

                /* now for thermal subcircuit values */
                tmp = (here->SOI3drainConductance==0?0:(1/(here->SOI3drainConductance)));
                tmp += (here->SOI3sourceConductance==0?0:(1/(here->SOI3sourceConductance)));
                /* tmp = RS+RD */
                here->SOI3iPt = (here->SOI3mode*vds + idrain*tmp)*idrain;
                here->SOI3gPmf = (here->SOI3mode*vds + 2*idrain*tmp)*here->SOI3gmf;
                here->SOI3gPmb = (here->SOI3mode*vds + 2*idrain*tmp)*here->SOI3gmb;
                here->SOI3gPmbs = (here->SOI3mode*vds + 2*idrain*tmp)*here->SOI3gmbs;
                here->SOI3gPds = idrain + (here->SOI3mode*vds + 2*idrain*tmp)*here->SOI3gds;
                here->SOI3gPdT = (here->SOI3mode*vds + 2*idrain*tmp)*here->SOI3gt;

                /* JimB - 15/9/99 */
                /* Code for multiple thermal time constants.  Start by moving all */
                /* rt constants into arrays. */
                rtargs[0]=here->SOI3rt;
                rtargs[1]=here->SOI3rt1;
                rtargs[2]=here->SOI3rt2;
                rtargs[3]=here->SOI3rt3;
                rtargs[4]=here->SOI3rt4;

                /* Set all conductance components to zero. */
                grt[0]=grt[1]=grt[2]=grt[3]=grt[4]=0.0;
                /* Now calculate conductances from rt. */
                /* Don't need to worry about divide by zero when calculating */
                /* grt components, as soi3setup() only creates a thermal node */
                /* if corresponding rt is greater than zero. */
                for(tnodeindex=0;tnodeindex<here->SOI3numThermalNodes;tnodeindex++)
                {
                   grt[tnodeindex]=1/rtargs[tnodeindex];
                }
                /* End JimB */

/* now end nasty trick - if vsb and vdb have been switched, reverse them back */
             	if (here->SOI3mode == -1)
               {
                       SWAP(double, vsb, vdb);
            	}
                
            /*
             * bulk-source and bulk-drain diodes
             *  includes parasitic BJT and 2nd diode
             *  
             */

                tmp = here->SOI3temp+deltaT;
                tmp1 = here->SOI3temp + model->SOI3dvt * deltaT;
                vT = CONSTKoverQ * tmp1;

                if ((model->SOI3betaEXP) != 1.0)
                {
                   if ((model->SOI3betaEXP) != 2.0)
                   {
                      BetaBJTeff = model->SOI3betaBJT *
                                   exp(-(model->SOI3betaEXP)*logL);
                   }
                   else
                   {
                      BetaBJTeff = model->SOI3betaBJT/
                      				  (EffectiveLength*EffectiveLength);
                   }
                }
                else
                {
            		 BetaBJTeff = model->SOI3betaBJT/EffectiveLength;
                }

                alphaBJT = BetaBJTeff/(BetaBJTeff + 1);

                EchiD = exp(MIN(MAX_EXP_ARG,
                                (model->SOI3chid * deltaT)/(here->SOI3temp*tmp)
                               )
                           );
                EchiD1 = exp(MIN(MAX_EXP_ARG,
                                (model->SOI3chid1 * deltaT)/(here->SOI3temp*tmp)
                               )
                           );
                ISts = SourceSatCur*EchiD;
                IS1ts = SourceSatCur1*EchiD1;

                evbs = exp(MIN(MAX_EXP_ARG,vbs/((model->SOI3etad)*vT)));
                evbs1 = exp(MIN(MAX_EXP_ARG,vbs/((model->SOI3etad1)*vT)));
/* First Junction */
                here->SOI3ibs = ISts * (evbs-1);
                here->SOI3gbs = ISts*evbs/((model->SOI3etad)*vT);
                here->SOI3gbsT = ISts*((evbs-1)*model->SOI3chid/(tmp*tmp) -
                                      evbs*vbs*model->SOI3dvt/((model->SOI3etad)*vT*tmp1));
/* Now Bipolar */
                here->SOI3iBJTdb = alphaBJT * ISts * (evbs-1);
                here->SOI3gBJTdb_bs = alphaBJT * here->SOI3gbs;
                here->SOI3gBJTdb_deltaT = alphaBJT * here->SOI3gbsT;
/* Now second junction and gmin */
                /* JimB - make gmin code consistent */
                here->SOI3ibs += IS1ts * (evbs1-1) + ckt->CKTgmin*vbs;
                here->SOI3gbs += IS1ts*evbs1/((model->SOI3etad1)*vT) + ckt->CKTgmin;
                /* End JimB */
                here->SOI3gbsT += IS1ts*((evbs1-1)*model->SOI3chid1/(tmp*tmp) -
                                        evbs1*vbs*model->SOI3dvt/((model->SOI3etad1)*vT*tmp1)
                                       );


                IStd = DrainSatCur*EchiD;
                IS1td = DrainSatCur1*EchiD1;

                evbd = exp(MIN(MAX_EXP_ARG,vbd/((model->SOI3etad)*vT)));
                evbd1 = exp(MIN(MAX_EXP_ARG,vbd/((model->SOI3etad1)*vT)));
/* First Junction */
                here->SOI3ibd = IStd *(evbd-1);
                here->SOI3gbd = IStd*evbd/((model->SOI3etad)*vT);
                here->SOI3gbdT = IStd*((evbd-1)*model->SOI3chid/(tmp*tmp) -
                                      evbd*vbd*model->SOI3dvt/((model->SOI3etad)*vT*tmp1));
/* Now Bipolar */
                here->SOI3iBJTsb = alphaBJT * IStd *(evbd-1);
                here->SOI3gBJTsb_bd = alphaBJT * here->SOI3gbd;
                here->SOI3gBJTsb_deltaT = alphaBJT * here->SOI3gbdT;
/* Now second junction and gmin */
                /* JimB - make gmin code consistent */
                here->SOI3ibd += IS1td *(evbd1-1) + ckt->CKTgmin*vbd;
                here->SOI3gbd += IS1td*evbd1/((model->SOI3etad1)*vT) + ckt->CKTgmin;
                /* End JimB */
                here->SOI3gbdT += IS1td*((evbd1-1)*model->SOI3chid1/(tmp*tmp) -
                                         evbd1*vbd*model->SOI3dvt/((model->SOI3etad1)*vT*tmp1)
                                        );


/* initialise von for voltage limiting purposes */
            von = (here->SOI3tVfbF * model->SOI3type) + psi_s0 + gamma*sqt0
                      + sigma*(here->SOI3mode*vds)
                      - model->SOI3chiFB*deltaT;

/* finally if we're going to do charge/capacitance calcs, store
 * some stuff to pass to function.  Need to use arrays 'cos there's
 * a limit to how many parameters you can pass in standard C
 */
            paramargs[0] = here->SOI3w*model->SOI3frontOxideCapFactor;
            paramargs[1] = EffectiveLength;
            paramargs[2] = gamma;
            paramargs[3] = eta_s;
            paramargs[4] = vt;
            paramargs[5] = delta;
            paramargs[6] = here->SOI3w*model->SOI3backOxideCapFactor;
            paramargs[7] = sigma;
            paramargs[8] = model->SOI3chiFB;
            paramargs[9] = model->SOI3satChargeShareFactor;
            Bfargs[0] = Bf;
            Bfargs[1] = pDBf_Dpsi_st0;
            alpha_args[0] = alpha;
            alpha_args[1] = Dalpha_Dvgfb;
            alpha_args[2] = Dalpha_Dvdb;
            alpha_args[3] = Dalpha_Dvsb;
            alpha_args[4] = Dalpha_DdeltaT;
            psi_st0args[0] = psi_st0;
            psi_st0args[1] = Dpsi_st0_Dvgfb;
            psi_st0args[2] = Dpsi_st0_Dvdb;
            psi_st0args[3] = Dpsi_st0_Dvsb;
            psi_st0args[4] = Dpsi_st0_DdeltaT;
            vGTargs[0] = vGT;
            vGTargs[1] = DvGT_Dvgfb;
            vGTargs[2] = DvGT_Dvdb;
            vGTargs[3] = DvGT_Dvsb;
            vGTargs[4] = DvGT_DdeltaT;
            psi_sLargs[0] = psi_sL;
            psi_sLargs[1] = Dpsi_sL_Dvgfb;
            psi_sLargs[2] = Dpsi_sL_Dvdb;
            psi_sLargs[3] = Dpsi_sL_Dvsb;
            psi_sLargs[4] = Dpsi_sL_DdeltaT;
            psi_s0args[0] = psi_s0;
            psi_s0args[1] = Dpsi_s0_Dvgfb;
            psi_s0args[2] = Dpsi_s0_Dvdb;
            psi_s0args[3] = Dpsi_s0_Dvsb;
            psi_s0args[4] = Dpsi_s0_DdeltaT;
            ldargs[0] = ld;
            ldargs[1] = Dld_Dvgfb;
            ldargs[2] = Dld_Dvdb;
            ldargs[3] = Dld_Dvsb;
            ldargs[4] = Dld_DdeltaT;
/* debug stuff */
                here->SOI3debug1 = psi_sL;
                here->SOI3debug2 = psi_s0;
                here->SOI3debug3 = fL;
                here->SOI3debug4 = f0;
                here->SOI3debug5 = gmd;
                here->SOI3debug6 = gms;

            }  /*  end block  */



            /* now deal with n vs p polarity */

            here->SOI3von = model->SOI3type * von;
            here->SOI3vdsat = model->SOI3type * vdsat_ext;
            /* line 490 */
            /*
             *  COMPUTE EQUIVALENT DRAIN CURRENT SOURCE
             *
    OLD        here->SOI3id=here->SOI3mode * idrain - here->SOI3ibd;
             */
              if (here->SOI3mode==1) {
                here->SOI3id= idrain - here->SOI3ibd + here->SOI3iMdb
                                 + here->SOI3iBJTdb;
              } else {
                here->SOI3id= -idrain - here->SOI3ibd
                                 + here->SOI3iBJTdb;
              }


            /* JimB - 4/1/99 */
            /* Tidy up depletion capacitance code, and remove unwanted */
            /* compile options. */

            if (ckt->CKTmode & (MODETRAN | MODETRANOP | MODEINITSMSIG))
            {
              /*
               * Now we do the hard part of the bulk-drain and bulk-source
               * diode - we evaluate the non-linear capacitance and
               * charge
               *
               * The basic equations are not hard, but the implementation
               * is somewhat long in an attempt to avoid log/exponential
               * evaluations.  This is because most users use the default
               * grading coefficients of 0.5, and sqrt is MUCH faster than
               * an exp(log()), so we use this special case to buy time -
               * as much as 10% of total job time!
               */

               /******** Bulk-source depletion capacitance ********/

#ifdef CAPBYPASS
				   if(((ckt->CKTmode & (MODEINITPRED | MODEINITTRAN) ) ||
                        fabs(delvbs) >= ckt->CKTreltol * MAX(fabs(vbs),
                        fabs(*(ckt->CKTstate0+here->SOI3vbs)))+
                        ckt->CKTvoltTol))
#endif /*CAPBYPASS*/
               {
                  if(here->SOI3Cbs != 0)
                  {
                     if (vbs < here->SOI3tDepCap)
                     {
                        arg=1-vbs/here->SOI3tBulkPot;
                        if(model->SOI3bulkJctSideGradingCoeff == 0.5)
                        {
                           sarg = 1/sqrt(arg);
                        }
                        else
                        {
                           sarg = exp(-model->SOI3bulkJctSideGradingCoeff*
                           				log(arg));
                        }
                        *(ckt->CKTstate0 + here->SOI3qbs) =
                            here->SOI3tBulkPot*(here->SOI3Cbs*
                            (1-arg*sarg)/(1-model->SOI3bulkJctSideGradingCoeff));
                        here->SOI3capbs=here->SOI3Cbs*sarg;
                     }
                     else
                     {
                        *(ckt->CKTstate0 + here->SOI3qbs) = here->SOI3f4s +
                                vbs*(here->SOI3f2s+vbs*(here->SOI3f3s/2));
                        here->SOI3capbs=here->SOI3f2s+here->SOI3f3s*vbs;
                     }
                  }
               	else
                  {
                     *(ckt->CKTstate0 + here->SOI3qbs) = 0;
                     here->SOI3capbs=0;
                  }
               }

               /******** End bulk-source depletion capcitance ********/

               /******** Bulk-drain depletion capacitance ********/

#ifdef CAPBYPASS
               if(((ckt->CKTmode & (MODEINITPRED | MODEINITTRAN) ) ||
                        fabs(delvbd) >= ckt->CKTreltol * MAX(fabs(vbd),
                        fabs(*(ckt->CKTstate0+here->SOI3vbd)))+
                        ckt->CKTvoltTol))
#endif /*CAPBYPASS*/
               {
                  if(here->SOI3Cbd != 0)
                  {
                     if (vbd < here->SOI3tDepCap)
                     {
                        arg=1-vbd/here->SOI3tBulkPot;
                        if(model->SOI3bulkJctSideGradingCoeff == 0.5)
                        {
                           sarg = 1/sqrt(arg);
                        }
                        else
                        {
                           sarg = exp(-model->SOI3bulkJctSideGradingCoeff*
                                        log(arg));
                        }
                        *(ckt->CKTstate0 + here->SOI3qbd) =
                            here->SOI3tBulkPot*(here->SOI3Cbd*
                            (1-arg*sarg)/(1-model->SOI3bulkJctSideGradingCoeff));
                        here->SOI3capbd=here->SOI3Cbd*sarg;
                     }
                     else
                     {
                        *(ckt->CKTstate0 + here->SOI3qbd) = here->SOI3f4d +
                                vbd * (here->SOI3f2d + vbd *(here->SOI3f3d/2));
                        here->SOI3capbd=here->SOI3f2d + vbd * here->SOI3f3d;
                     }
                  }
                  else
                  {
                     *(ckt->CKTstate0 + here->SOI3qbd) = 0;
                    		   here->SOI3capbd = 0;
                  }
               }

               /******** End bulk-drain depletion capacitance ********/


/* Need to work out charge on thermal cap */
                *(ckt->CKTstate0 + here->SOI3qt) = here->SOI3ct * deltaT1;
                *(ckt->CKTstate0 + here->SOI3qt1) = here->SOI3ct1 * deltaT2;
                *(ckt->CKTstate0 + here->SOI3qt2) = here->SOI3ct2 * deltaT3;
                *(ckt->CKTstate0 + here->SOI3qt3) = here->SOI3ct3 * deltaT4;
                *(ckt->CKTstate0 + here->SOI3qt4) = here->SOI3ct4 * deltaT5;
/* ct is constant, so integral is trivial */

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

                    error = NIintegrate(ckt,&geq,&ieq,here->SOI3capbd,
                            here->SOI3qbd);
                    if(error) return(error);
                    here->SOI3gbd += geq;
                    here->SOI3ibd += *(ckt->CKTstate0 + here->SOI3iqbd);
                    here->SOI3id -= *(ckt->CKTstate0 + here->SOI3iqbd);
                    error = NIintegrate(ckt,&geq,&ieq,here->SOI3capbs,
                            here->SOI3qbs);
                    if(error) return(error);
                    here->SOI3gbs += geq;
                    here->SOI3ibs += *(ckt->CKTstate0 + here->SOI3iqbs);
                }
            }


            /*
             *  check convergence
             */
            if ( (here->SOI3off == 0)  ||
                    (!(ckt->CKTmode & (MODEINITFIX|MODEINITSMSIG))) ){
                if (Check == 1) {
                    ckt->CKTnoncon++;
                }
            }

            /*
             *     new capacitor model
             */
            if ((ckt->CKTmode & (MODETRAN | MODEAC | MODEINITSMSIG)) ||
                ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC))
               ){
                /*
                 *     calculate charges and capacitances
                 */
                /*
                 * Partially Depleted model => back surface neglected
                 * so back capacitances are ignored.
                 */
                /*
                 * All the args assume vsb and vdb are 'correct'
                 * so must make sure we're getting right values
                 * back.
                 */
                if (here->SOI3mode > 0)
                {
                  SOI3cap((vgbb-here->SOI3tVfbB),(here->SOI3tPhi + vsb),model->SOI3gammaB,
                          paramargs,
                          Bfargs,alpha_args,psi_st0args,
                          vGTargs,
                          psi_sLargs,psi_s0args,
                          ldargs,
                          &qgatef,&qbody,&qdrn,&qgateb,
                          &cgfgf,&cgfd,&cgfs,&cgfdeltaT,&cgfgb,
                          &cbgf,&cbd,&cbs,&cbdeltaT,&cbgb,
                          &cdgf,&cdd,&cds,&cddeltaT,&cdgb,
                          &cgbgf,&cgbd,&cgbs,&cgbdeltaT,&cgbgb);
                  csgf = -(cgfgf + cdgf + cbgf + cgbgf);
                  csd  = -(cgfd + cdd + cbd + cgbd);
                  css  = -(cgfs + cds + cbs + cgbs);
                  csdeltaT = -(cgfdeltaT + cddeltaT + cbdeltaT + cgbdeltaT);
                  csgb = -(cgfgb + cdgb + cbgb + cgbgb);
                }
                else
                {
                  SOI3cap((vgbb-here->SOI3tVfbB),(here->SOI3tPhi + vdb),model->SOI3gammaB,
                          paramargs,
                          Bfargs,alpha_args,psi_st0args,
                          vGTargs,
                          psi_sLargs,psi_s0args,
                          ldargs,
                          &qgatef,&qbody,&qsrc,&qgateb,
                          &cgfgf,&cgfs,&cgfd,&cgfdeltaT,&cgfgb,
                          &cbgf,&cbs,&cbd,&cbdeltaT,&cbgb,
                          &csgf,&css,&csd,&csdeltaT,&csgb,
                          &cgbgf,&cgbs,&cgbd,&cgbdeltaT,&cgbgb);
                  cdgf = -(cgfgf + csgf + cbgf + cgbgf);
                  cdd  = -(cgfd + csd + cbd + cgbd);
                  cds  = -(cgfs + css + cbs + cgbs);
                  cddeltaT = -(cgfdeltaT + csdeltaT + cbdeltaT + cgbdeltaT);
                  cdgb = -(cgfgb + csgb + cbgb + cgbgb);
                }

                *(ckt->CKTstate0 + here->SOI3cgfgf) = cgfgf;
                *(ckt->CKTstate0 + here->SOI3cgfd) = cgfd;
                *(ckt->CKTstate0 + here->SOI3cgfs) = cgfs;
                *(ckt->CKTstate0 + here->SOI3cgfdeltaT) = cgfdeltaT;
                *(ckt->CKTstate0 + here->SOI3cgfgb) = cgfgb;

                *(ckt->CKTstate0 + here->SOI3csgf) = csgf;
                *(ckt->CKTstate0 + here->SOI3csd) = csd;
                *(ckt->CKTstate0 + here->SOI3css) = css;
                *(ckt->CKTstate0 + here->SOI3csdeltaT) = csdeltaT;
                *(ckt->CKTstate0 + here->SOI3csgb) = csgb;

                *(ckt->CKTstate0 + here->SOI3cdgf) = cdgf;
                *(ckt->CKTstate0 + here->SOI3cdd) = cdd;
                *(ckt->CKTstate0 + here->SOI3cds) = cds;
                *(ckt->CKTstate0 + here->SOI3cddeltaT) = cddeltaT;
                *(ckt->CKTstate0 + here->SOI3cdgb) = cdgb;

                *(ckt->CKTstate0 + here->SOI3cgbgf) = cgbgf;
                *(ckt->CKTstate0 + here->SOI3cgbd) = cgbd;
                *(ckt->CKTstate0 + here->SOI3cgbs) = cgbs;
                *(ckt->CKTstate0 + here->SOI3cgbdeltaT) = cgbdeltaT;
                *(ckt->CKTstate0 + here->SOI3cgbgb) = cgbgb;


/* got charges and caps now, must get equiv conductances/current sources
 * but first work out charge and caps for BJT charge storage
 */

                if ((model->SOI3tauEXP) != 2.0)
                {
                   if ((model->SOI3tauEXP) != 1.0)
                   {
                      tauFBJTeff = model->SOI3tauFBJT *
                                  exp((model->SOI3tauEXP)*logL);
                		 tauRBJTeff = model->SOI3tauRBJT *
                                  exp((model->SOI3tauEXP)*logL);
                   }
                   else
                   {
                      tauFBJTeff = model->SOI3tauFBJT*EffectiveLength;
                      tauRBJTeff = model->SOI3tauRBJT*EffectiveLength;
                   }
                }
                else
                {
                   tauFBJTeff = model->SOI3tauFBJT*(EffectiveLength*EffectiveLength);
                   tauRBJTeff = model->SOI3tauRBJT*(EffectiveLength*EffectiveLength);
                }

                *(ckt->CKTstate0 + here->SOI3qBJTbs) = tauFBJTeff *
                                   here->SOI3iBJTdb;
                *(ckt->CKTstate0 + here->SOI3cBJTbsbs) = tauFBJTeff *
                                   here->SOI3gBJTdb_bs;
                *(ckt->CKTstate0 + here->SOI3cBJTbsdeltaT) = tauFBJTeff *
                                   here->SOI3gBJTdb_deltaT;

                *(ckt->CKTstate0 + here->SOI3qBJTbd) = tauRBJTeff *
                                   here->SOI3iBJTsb;
                *(ckt->CKTstate0 + here->SOI3cBJTbdbd) = tauRBJTeff *
                                   here->SOI3gBJTsb_bd;
                *(ckt->CKTstate0 + here->SOI3cBJTbddeltaT) = tauRBJTeff *
                                   here->SOI3gBJTsb_deltaT;
            }   /* end of charge/cap calculations */


            /* save things away for next time */

            *(ckt->CKTstate0 + here->SOI3vbs) = vbs;
            *(ckt->CKTstate0 + here->SOI3vbd) = vbd;
            *(ckt->CKTstate0 + here->SOI3vgfs) = vgfs;
            *(ckt->CKTstate0 + here->SOI3vgbs) = vgbs;
            *(ckt->CKTstate0 + here->SOI3vds) = vds;
            *(ckt->CKTstate0 + here->SOI3deltaT) = deltaT;
            *(ckt->CKTstate0 + here->SOI3deltaT1) = deltaT1;
            *(ckt->CKTstate0 + here->SOI3deltaT2) = deltaT2;
            *(ckt->CKTstate0 + here->SOI3deltaT3) = deltaT3;
            *(ckt->CKTstate0 + here->SOI3deltaT4) = deltaT4;
            *(ckt->CKTstate0 + here->SOI3deltaT5) = deltaT5;
            *(ckt->CKTstate0 + here->SOI3idrain) = idrain;

            if((!(ckt->CKTmode & (MODETRAN | MODEAC))) &&
                    ((!(ckt->CKTmode & MODETRANOP)) ||
                    (!(ckt->CKTmode & MODEUIC)))  && (!(ckt->CKTmode 
                    &  MODEINITSMSIG))) goto bypass2;
#ifndef NOBYPASS
bypass1:
#endif            
            if (here->SOI3mode>0) {
              Frontcapargs[0] = FrontGateDrainOverlapCap;
              Frontcapargs[1] = FrontGateSourceOverlapCap;
              Frontcapargs[2] = FrontGateBulkOverlapCap;
              Frontcapargs[3] = vgfd;
              Frontcapargs[4] = vgfs;
              Frontcapargs[5] = vgfb;
              Backcapargs[0] = BackGateDrainOverlapCap;
              Backcapargs[1] = BackGateSourceOverlapCap;
              Backcapargs[2] = BackGateBulkOverlapCap;
              Backcapargs[3] = vgbd;
              Backcapargs[4] = vgbs;
              Backcapargs[5] = vgbb;
              SOI3capEval(ckt,
                          Frontcapargs,
                          Backcapargs,
                          cgfgf,cgfd,cgfs,cgfdeltaT,cgfgb,
                          cdgf,cdd,cds,cddeltaT,cdgb,
                          csgf,csd,css,csdeltaT,csgb,
                          cbgf,cbd,cbs,cbdeltaT,cbgb,
                          cgbgf,cgbd,cgbs,cgbdeltaT,cgbgb,
                          &gcgfgf,&gcgfd,&gcgfs,&gcgfdeltaT,&gcgfgb,
                          &gcdgf,&gcdd,&gcds,&gcddeltaT,&gcdgb,
                          &gcsgf,&gcsd,&gcss,&gcsdeltaT,&gcsgb,
                          &gcbgf,&gcbd,&gcbs,&gcbdeltaT,&gcbgb,
                          &gcgbgf,&gcgbd,&gcgbs,&gcgbdeltaT,&gcgbgb,
                          &qgatef,&qbody,&qdrn,&qsrc,&qgateb);
            } else {
              Frontcapargs[0] = FrontGateSourceOverlapCap;
              Frontcapargs[1] = FrontGateDrainOverlapCap;
              Frontcapargs[2] = FrontGateBulkOverlapCap;
              Frontcapargs[3] = vgfs;
              Frontcapargs[4] = vgfd;
              Frontcapargs[5] = vgfb;
              Backcapargs[0] = BackGateSourceOverlapCap;
              Backcapargs[1] = BackGateDrainOverlapCap;
              Backcapargs[2] = BackGateBulkOverlapCap;
              Backcapargs[3] = vgbs;
              Backcapargs[4] = vgbd;
              Backcapargs[5] = vgbb;
              SOI3capEval(ckt,
                          Frontcapargs,
                          Backcapargs,
                          cgfgf,cgfs,cgfd,cgfdeltaT,cgfgb,
                          csgf,css,csd,csdeltaT,csgb,
                          cdgf,cds,cdd,cddeltaT,cdgb,
                          cbgf,cbs,cbd,cbdeltaT,cbgb,
                          cgbgf,cgbs,cgbd,cgbdeltaT,cgbgb,
                          &gcgfgf,&gcgfs,&gcgfd,&gcgfdeltaT,&gcgfgb,
                          &gcsgf,&gcss,&gcsd,&gcsdeltaT,&gcsgb,
                          &gcdgf,&gcds,&gcdd,&gcddeltaT,&gcdgb,
                          &gcbgf,&gcbs,&gcbd,&gcbdeltaT,&gcbgb,
                          &gcgbgf,&gcgbs,&gcgbd,&gcgbdeltaT,&gcgbgb,
                          &qgatef,&qbody,&qsrc,&qdrn,&qgateb);
            }
            ag0 = ckt->CKTag[0];
            gcBJTbsbs = ag0 * *(ckt->CKTstate0+here->SOI3cBJTbsbs);
            gcBJTbsdeltaT = ag0 * *(ckt->CKTstate0+here->SOI3cBJTbsdeltaT);
            gcBJTbdbd = ag0 * *(ckt->CKTstate0+here->SOI3cBJTbdbd);
            gcBJTbddeltaT = ag0 * *(ckt->CKTstate0+here->SOI3cBJTbddeltaT);

            if (ByPass) goto line860; /* already stored charges */

            *(ckt->CKTstate0 + here->SOI3qgf) = qgatef;
            *(ckt->CKTstate0 + here->SOI3qd) = qdrn;
            *(ckt->CKTstate0 + here->SOI3qs) = qsrc;
            *(ckt->CKTstate0 + here->SOI3qgb) = qgateb;
            /* NB, we've kept charge/cap associated with diodes separately */

            if((!(ckt->CKTmode & (MODEAC | MODETRAN))) &&
                    (ckt->CKTmode & MODETRANOP ) && (ckt->CKTmode &
                    MODEUIC ))   goto bypass2;

            /* store small signal parameters */
            if(ckt->CKTmode & MODEINITSMSIG ) {  
                *(ckt->CKTstate0+here->SOI3cgfgf) = cgfgf;
                *(ckt->CKTstate0+here->SOI3cgfd)  = cgfd;
                *(ckt->CKTstate0+here->SOI3cgfs)  = cgfs;
                *(ckt->CKTstate0+here->SOI3cgfdeltaT)  = cgfdeltaT;
                *(ckt->CKTstate0+here->SOI3cgfgb) = cgfgb;
                *(ckt->CKTstate0+here->SOI3cdgf)  = cdgf;
                *(ckt->CKTstate0+here->SOI3cdd)   = cdd;
                *(ckt->CKTstate0+here->SOI3cds)   = cds;
                *(ckt->CKTstate0+here->SOI3cddeltaT)  = cddeltaT;
                *(ckt->CKTstate0+here->SOI3cdgb)  = cdgb;
                *(ckt->CKTstate0+here->SOI3csgf)  = csgf;
                *(ckt->CKTstate0+here->SOI3csd)   = csd;
                *(ckt->CKTstate0+here->SOI3css)   = css;
                *(ckt->CKTstate0+here->SOI3csdeltaT)  = csdeltaT;
                *(ckt->CKTstate0+here->SOI3csgb)  = csgb;
                *(ckt->CKTstate0+here->SOI3cgbgf) = cgbgf;
                *(ckt->CKTstate0+here->SOI3cgbd)  = cgbd;
                *(ckt->CKTstate0+here->SOI3cgbs)  = cgbs;
                *(ckt->CKTstate0+here->SOI3cgbdeltaT)  = cgbdeltaT;
                *(ckt->CKTstate0+here->SOI3cgbgb) = cgbgb;

                goto End;
            }

            if (ckt->CKTmode & MODEINITTRAN) {
              *(ckt->CKTstate1 + here->SOI3qgf) = 
                *(ckt->CKTstate0 + here->SOI3qgf);
              *(ckt->CKTstate1 + here->SOI3qd) = 
                *(ckt->CKTstate0 + here->SOI3qd);
              *(ckt->CKTstate1 + here->SOI3qs) = 
                *(ckt->CKTstate0 + here->SOI3qs);
              *(ckt->CKTstate1 + here->SOI3qgb) = 
                *(ckt->CKTstate0 + here->SOI3qgb);
              *(ckt->CKTstate1 + here->SOI3qBJTbs) = 
                *(ckt->CKTstate0 + here->SOI3qBJTbs);
              *(ckt->CKTstate1 + here->SOI3qBJTbd) = 
                *(ckt->CKTstate0 + here->SOI3qBJTbd);
            }

            /*
             *    numerical integration of intrinsic caps
             *    and BJT caps
             */
            error = NIintegrate(ckt,&geq,&ieq,0.0,here->SOI3qgf);
            if(error) return(error);
            error = NIintegrate(ckt,&geq,&ieq,0.0,here->SOI3qd);
            if(error) return(error);
            error = NIintegrate(ckt,&geq,&ieq,0.0,here->SOI3qs);
            if(error) return(error);
            error = NIintegrate(ckt,&geq,&ieq,0.0,here->SOI3qgb);
            if(error) return(error);

            error = NIintegrate(ckt,&geq,&ieq,0.0,here->SOI3qBJTbs);
            if(error) return(error);
            error = NIintegrate(ckt,&geq,&ieq,0.0,here->SOI3qBJTbd);
            if(error) return(error);
            goto line860;

bypass2:

            /*
             *  initialize to zero charge conductances
             *  and current (DC and TRANOP)
             */
            ieqqgf = ieqqd = ieqqs = ieqqgb = 0.0;
            gcgfgf = gcgfd = gcgfs = gcgfdeltaT = gcgfgb = 0.0;
            gcdgf  = gcdd  = gcds  = gcddeltaT  = gcdgb  = 0.0;
            gcsgf  = gcsd  = gcss  = gcsdeltaT  = gcsgb  = 0.0;
            gcbgf  = gcbd  = gcbs  = gcbdeltaT  = gcbgb  = 0.0;
            gcgbgf = gcgbd = gcgbs = gcgbdeltaT = gcgbgb = 0.0;
     			gct[0]=gct[1]=gct[2]=gct[3]=gct[4]=0.0;
            ieqct=ieqct1=ieqct2=ieqct3=ieqct4=0.0;
            ieqqBJTbs = ieqqBJTbd = 0.0;
            gcBJTbsbs = gcBJTbsdeltaT = gcBJTbdbd = gcBJTbddeltaT = 0.0;
            goto LoadUp;

line860:
            
            /* evaluate equivalent charge currents */

                ieqqgf = *(ckt->CKTstate0 + here->SOI3iqgf) -
                        gcgfgf * vgfb - gcgfd * vdb - gcgfs * vsb - gcgfdeltaT * deltaT -
                        gcgfgb * vgbb;
                ieqqd = *(ckt->CKTstate0 + here->SOI3iqd) -
                        gcdgf * vgfb - gcdd * vdb - gcds * vsb - gcddeltaT * deltaT -
                        gcdgb * vgbb;
                ieqqs = *(ckt->CKTstate0 + here->SOI3iqs) -
                        gcsgf * vgfb - gcsd * vdb - gcss * vsb - gcsdeltaT * deltaT -
                        gcsgb * vgbb;
                ieqqgb = *(ckt->CKTstate0 + here->SOI3iqgb) -
                        gcgbgf * vgfb - gcgbd * vdb - gcgbs * vsb - gcgbdeltaT * deltaT -
                        gcgbgb * vgbb;

                ieqqBJTbs = *(ckt->CKTstate0 + here->SOI3iqBJTbs) -
                        gcBJTbsbs * vbs - gcBJTbsdeltaT * deltaT;
                ieqqBJTbd = *(ckt->CKTstate0 + here->SOI3iqBJTbd) -
                        gcBJTbdbd * vbd - gcBJTbddeltaT * deltaT;

                    /* now for the thermal capacitance
                       is constant and linear so no need
                       to fart about with equiv current  */


                    error = NIintegrate(ckt,&gct[0],&ieqct,here->SOI3ct,
                            here->SOI3qt);
                    if(error) return(error);
                    error = NIintegrate(ckt,&gct[1],&ieqct1,here->SOI3ct1,
                            here->SOI3qt1);
                    if(error) return(error);
                    error = NIintegrate(ckt,&gct[2],&ieqct2,here->SOI3ct2,
                            here->SOI3qt2);
                    if(error) return(error);
                    error = NIintegrate(ckt,&gct[3],&ieqct3,here->SOI3ct3,
                            here->SOI3qt3);
                    if(error) return(error);
                    error = NIintegrate(ckt,&gct[4],&ieqct4,here->SOI3ct4,
                            here->SOI3qt4);
                    if(error) return(error);

LoadUp:

            /*
             *  load current vector
             */
	     
	     m = here->SOI3m;
	     
            ieqbs = model->SOI3type *
                    (here->SOI3ibs-(here->SOI3gbs-ckt->CKTgmin)*vbs
                                  -(here->SOI3gbsT)*deltaT);
            ieqbd = model->SOI3type *
                    (here->SOI3ibd-(here->SOI3gbd-ckt->CKTgmin)*vbd
                                  -(here->SOI3gbdT)*deltaT);
            iBJTdbeq = model->SOI3type *
                       (here->SOI3iBJTdb -(here->SOI3gBJTdb_bs)*vbs
                                         -(here->SOI3gBJTdb_deltaT)*deltaT);
            iBJTsbeq = model->SOI3type *
                       (here->SOI3iBJTsb -(here->SOI3gBJTsb_bd)*vbd
                                         -(here->SOI3gBJTsb_deltaT)*deltaT);
            ieqPt = here->SOI3iPt - (here->SOI3gPds*(here->SOI3mode*vds)+
                                     here->SOI3gPmf*(here->SOI3mode==1?vgfs:vgfd)+
                                     here->SOI3gPmb*(here->SOI3mode==1?vgbs:vgbd)+
                                     here->SOI3gPmbs*(here->SOI3mode==1?vbs:vbd)+
                                     here->SOI3gPdT*(deltaT));
            if (here->SOI3mode >= 0)
            {
                xnrm=1;
                xrev=0;
                idreq=model->SOI3type*(idrain-here->SOI3gds*vds-
                        here->SOI3gmf*vgfs-here->SOI3gmbs*vbs-
                        here->SOI3gmb*vgbs-here->SOI3gt*deltaT);
                iMdbeq=model->SOI3type*(here->SOI3iMdb-here->SOI3gMd*vds-
                        here->SOI3gMmf*vgfs-here->SOI3gMmbs*vbs-
                        here->SOI3gMmb*vgbs-here->SOI3gMdeltaT*deltaT);
                iMsbeq=0;
            }
            else
            {
                xnrm=0;
                xrev=1;
                idreq = -(model->SOI3type)*(idrain-here->SOI3gds*(-vds)-
                        here->SOI3gmf*vgfd-here->SOI3gmbs*vbd-
                        here->SOI3gmb*vgbd-here->SOI3gt*deltaT);
                iMsbeq= (model->SOI3type)*(here->SOI3iMsb-here->SOI3gMd*(-vds)-
                        here->SOI3gMmf*vgfd-here->SOI3gMmbs*vbd-
                        here->SOI3gMmb*vgbd-here->SOI3gMdeltaT*deltaT);
                iMdbeq = 0;
            }
            *(ckt->CKTrhs + here->SOI3gfNode) -= m * (model->SOI3type * ieqqgf);
            *(ckt->CKTrhs + here->SOI3gbNode) -= m * (model->SOI3type * ieqqgb);
            *(ckt->CKTrhs + here->SOI3bNode) += m * (-(ieqbs + ieqbd) +
                                                iMdbeq + iMsbeq /* one is 0 */
                                               +iBJTdbeq + iBJTsbeq
                                               + model->SOI3type * (ieqqgf + ieqqd + ieqqs + ieqqgb)
                                               - model->SOI3type * (ieqqBJTbs + ieqqBJTbd));
            *(ckt->CKTrhs + here->SOI3dNodePrime) += m * ((ieqbd-idreq) -
                                                     iMdbeq - iBJTdbeq
                                                    - model->SOI3type * (ieqqd)
                                                    + model->SOI3type * (ieqqBJTbd));
            *(ckt->CKTrhs + here->SOI3sNodePrime) += m * ((idreq + ieqbs) -
                                                      iMsbeq - iBJTsbeq
                                                    - model->SOI3type * (ieqqs)
                                                    + model->SOI3type * (ieqqBJTbs));
            *(ckt->CKTrhs + here->SOI3toutNode) += m * (ieqPt-ieqct);
            if (here->SOI3numThermalNodes > 1)
            {
              *(ckt->CKTrhs + here->SOI3tout1Node) += m * (ieqct-ieqct1);
            }
            if (here->SOI3numThermalNodes > 2)
            {
              *(ckt->CKTrhs + here->SOI3tout2Node) += m * (ieqct1-ieqct2);
            }
            if (here->SOI3numThermalNodes > 3)
            {
              *(ckt->CKTrhs + here->SOI3tout3Node) += m * (ieqct2-ieqct3);
            }
            if (here->SOI3numThermalNodes > 4)
            {
              *(ckt->CKTrhs + here->SOI3tout4Node) += m * (ieqct3-ieqct4);
            }


            /* load y matrix */
            
            *(here->SOI3D_dPtr)  += m * (here->SOI3drainConductance);
            *(here->SOI3D_dpPtr) += m * (-here->SOI3drainConductance);
            *(here->SOI3DP_dPtr) += m * (-here->SOI3drainConductance);

            *(here->SOI3S_sPtr)  += m * (here->SOI3sourceConductance);
            *(here->SOI3S_spPtr) += m * (-here->SOI3sourceConductance);
            *(here->SOI3SP_sPtr) += m * (-here->SOI3sourceConductance);

            *(here->SOI3GF_gfPtr) += m * gcgfgf;
            *(here->SOI3GF_dpPtr) += m * gcgfd;
            *(here->SOI3GF_spPtr) += m * gcgfs;
            *(here->SOI3GF_gbPtr) += m * gcgfgb;
            *(here->SOI3GF_bPtr)  -= m * (gcgfgf + gcgfd + gcgfs + gcgfgb);

            *(here->SOI3DP_gfPtr) += m * ((xnrm-xrev)*here->SOI3gmf + gcdgf+
                                      xnrm*here->SOI3gMmf);
            *(here->SOI3DP_dpPtr) += m * ((here->SOI3drainConductance+here->SOI3gds+
                    						  here->SOI3gbd+xrev*(here->SOI3gmf+here->SOI3gmbs+
                    						  here->SOI3gmb)+xnrm*here->SOI3gMd + gcdd)+
                    						  gcBJTbdbd);
            *(here->SOI3DP_spPtr) += m * ((-here->SOI3gds - here->SOI3gBJTdb_bs
                                      -xnrm*(here->SOI3gmf+here->SOI3gmb+here->SOI3gmbs +
                   						  here->SOI3gMmf+here->SOI3gMmb+here->SOI3gMmbs+here->SOI3gMd)) +
                                      gcds);
            *(here->SOI3DP_gbPtr) += m * (((xnrm-xrev)*here->SOI3gmb +
                                      xnrm*here->SOI3gMmb) + gcdgb);
            *(here->SOI3DP_bPtr)  += m * ((-here->SOI3gbd + here->SOI3gBJTdb_bs +
                                      (xnrm-xrev)*here->SOI3gmbs +
                                      xnrm*here->SOI3gMmbs) -
                                      (gcdgf + gcdd + gcds + gcdgb + gcBJTbdbd));

            *(here->SOI3SP_gfPtr) += m * ((-(xnrm-xrev)*here->SOI3gmf+
                                      xrev*here->SOI3gMmf) +
                                      gcsgf);
            *(here->SOI3SP_dpPtr) += m * ((-here->SOI3gds - here->SOI3gBJTsb_bd
                                      -xrev*(here->SOI3gmf+here->SOI3gmb+here->SOI3gmbs+
                   						  here->SOI3gMmf+here->SOI3gMmb+here->SOI3gMmbs+here->SOI3gMd)) +
                                      gcsd);
            *(here->SOI3SP_spPtr) += m * ((here->SOI3sourceConductance+here->SOI3gds+
            								  here->SOI3gbs+xnrm*(here->SOI3gmf+here->SOI3gmbs+
                    						  here->SOI3gmb)+xrev*here->SOI3gMd + gcss)+
                    						  gcBJTbsbs);
            *(here->SOI3SP_gbPtr) += m * ((-(xnrm-xrev)*here->SOI3gmb+
                                      xrev*here->SOI3gMmb) + gcsgb);
            *(here->SOI3SP_bPtr)  += m * ((-here->SOI3gbs + here->SOI3gBJTsb_bd -
                                      (xnrm-xrev)*here->SOI3gmbs+
                                      xrev*here->SOI3gMmbs) -
                                      (gcsgf + gcsd + gcss + gcsgb + gcBJTbsbs));

            *(here->SOI3GB_gfPtr) += m * gcgbgf;
            *(here->SOI3GB_dpPtr) += m * gcgbd;
            *(here->SOI3GB_spPtr) += m * gcgbs;
            *(here->SOI3GB_gbPtr) += m * gcgbgb;
            *(here->SOI3GB_bPtr)  -= m * (gcgbgf + gcgbd + gcgbs + gcgbgb);

            *(here->SOI3B_gfPtr) += m * (-here->SOI3gMmf + gcbgf);
            *(here->SOI3B_dpPtr) += m * (-(here->SOI3gbd) + here->SOI3gBJTsb_bd +
                                   xrev*(here->SOI3gMmf+here->SOI3gMmb+
                                         here->SOI3gMmbs+here->SOI3gMd) -
                                   xnrm*here->SOI3gMd +
                                   gcbd - gcBJTbdbd);
            *(here->SOI3B_spPtr) += m * (-(here->SOI3gbs) + here->SOI3gBJTdb_bs +
                                   xnrm*(here->SOI3gMmf+here->SOI3gMmb+
                                   		  here->SOI3gMmbs+here->SOI3gMd) -
                                   xrev*here->SOI3gMd +
                                   gcbs - gcBJTbsbs);
            *(here->SOI3B_gbPtr) += m * (-(here->SOI3gMmb) + gcbgb);
            *(here->SOI3B_bPtr)  += m * ((here->SOI3gbd+here->SOI3gbs -
            							  here->SOI3gMmbs -
                                   here->SOI3gBJTdb_bs - here->SOI3gBJTsb_bd) -
                                   (gcbgf+gcbd+gcbs+gcbgb) +
                                   gcBJTbsbs+gcBJTbdbd);

        
/* if no thermal behaviour specified, then put in zero valued indpt. voltage source
   between TOUT and ground */
            if (here->SOI3rt==0)
            {
              *(here->SOI3TOUT_ibrPtr) += m * 1.0 ;
              *(here->SOI3IBR_toutPtr) += m * 1.0 ;
              *(ckt->CKTrhs + (here->SOI3branch)) = 0 ;
            }
            else
            {
              *(here->SOI3TOUT_toutPtr) += m * (-(here->SOI3gPdT)+grt[0]+gct[0]);

              if (here->SOI3numThermalNodes > 1)
              {
                	*(here->SOI3TOUT_tout1Ptr) += m * (-grt[0]-gct[0]);
                	*(here->SOI3TOUT1_toutPtr) += m * (-grt[0]-gct[0]);
                	*(here->SOI3TOUT1_tout1Ptr) += m * (grt[0]+grt[1]+gct[0]+gct[1]);
              }
              if (here->SOI3numThermalNodes > 2)
              {
                 	*(here->SOI3TOUT1_tout2Ptr) += m * (-grt[1]-gct[1]);
                	*(here->SOI3TOUT2_tout1Ptr) += m * (-grt[1]-gct[1]);
                	*(here->SOI3TOUT2_tout2Ptr) += m * (grt[1]+grt[2]+gct[1]+gct[2]);
              }
              if (here->SOI3numThermalNodes > 3)
              {
                	*(here->SOI3TOUT2_tout3Ptr) += m * (-grt[2]-gct[2]);
                	*(here->SOI3TOUT3_tout2Ptr) += m * (-grt[2]-gct[2]);
                	*(here->SOI3TOUT3_tout3Ptr) += m * (grt[2]+grt[3]+gct[2]+gct[3]);
              }
              if (here->SOI3numThermalNodes > 4)
              {
                 	*(here->SOI3TOUT3_tout4Ptr) += m * (-grt[3]-gct[3]);
                	*(here->SOI3TOUT4_tout3Ptr) += m * (-grt[3]-gct[3]);
                	*(here->SOI3TOUT4_tout4Ptr) += m * (grt[3]+grt[4]+gct[3]+gct[4]);
              }

              *(here->SOI3TOUT_dpPtr) += m * (xnrm*(-(here->SOI3gPds*model->SOI3type))
                                        +xrev*(here->SOI3gPds+here->SOI3gPmf+
                                               here->SOI3gPmb+here->SOI3gPmbs)*
                                               model->SOI3type);
              *(here->SOI3TOUT_gfPtr) += m * (-(here->SOI3gPmf*model->SOI3type));
              *(here->SOI3TOUT_gbPtr) += m * (-(here->SOI3gPmb*model->SOI3type));
              *(here->SOI3TOUT_bPtr) += m * (-(here->SOI3gPmbs*model->SOI3type));
              *(here->SOI3TOUT_spPtr) += m * (xnrm*(here->SOI3gPds+here->SOI3gPmf+
                                          here->SOI3gPmb+here->SOI3gPmbs)*model->SOI3type
                                        +xrev*(-(here->SOI3gPds*model->SOI3type)));

              *(here->SOI3DP_toutPtr) += m * (xnrm-xrev)*here->SOI3gt*model->SOI3type;
              *(here->SOI3SP_toutPtr) += m * (xrev-xnrm)*here->SOI3gt*model->SOI3type;
/* need to mult by type in above as conductances will be used with exterior voltages
  which will be -ve for PMOS except for gPdT */
/* now for thermal influence on impact ionisation current and tranisent stuff */
              *(here->SOI3GF_toutPtr) += m * gcgfdeltaT*model->SOI3type;
              *(here->SOI3DP_toutPtr) += m * (xnrm*here->SOI3gMdeltaT + gcddeltaT -
                                          here->SOI3gbdT + here->SOI3gBJTdb_deltaT
                                          - gcBJTbddeltaT)*model->SOI3type;
              *(here->SOI3SP_toutPtr) += m * (xrev*here->SOI3gMdeltaT + gcsdeltaT -
                                          here->SOI3gbsT + here->SOI3gBJTsb_deltaT
                                          - gcBJTbsdeltaT)*model->SOI3type;
              *(here->SOI3GB_toutPtr) += m * gcgbdeltaT*model->SOI3type;
              *(here->SOI3B_toutPtr) -= m * (here->SOI3gMdeltaT - gcbdeltaT -
                                         here->SOI3gbsT - here->SOI3gbdT +
                                         here->SOI3gBJTdb_deltaT +
                                         here->SOI3gBJTsb_deltaT
                                         -gcBJTbsdeltaT-gcBJTbddeltaT)*model->SOI3type;
            }
End:        ;
        }
    }
    return(OK);
}


    /* DEVsoipnjlim(vnew,vold,vt,vcrit,icheck)
     * limit the per-iteration change of PN junction  voltages 
     */

double
DEVsoipnjlim(double vnew, double vold, double vt, double vcrit, int *icheck)
{
    double arg;

    if((vnew > vcrit) && (fabs(vnew - vold) > (vt + vt))) {
        if(vold > 0) {
            arg = 1 + (vnew - vold) / vt;
            if(arg > 0) {
                vnew = vold + vt * log(arg);
            } else {
                vnew = vcrit;
            }
        } else {
            vnew = vt *log(vnew/vt);
        }
        *icheck = 1;
    } else {
      if (fabs(vnew - vold) < (vt + vt)) {
        *icheck = 0;
      } else {
        if (vnew>vold) {
          *icheck = 0;
        } else {
          arg = 1 + (vold - vnew) / vt;
          vnew = vold - vt*log(arg);
          *icheck = 1;
        }
      }
    }
    return(vnew);
}
