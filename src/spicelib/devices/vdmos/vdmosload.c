/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
VDMOS: 2018 Holger Vogt
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "vdmosdefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
VDMOSload(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current value into the 
         * sparse matrix previously provided 
         */
{
    VDMOSmodel *model = (VDMOSmodel *) inModel;
    VDMOSinstance *here;
    double Beta;
    double DrainSatCur;
    double EffectiveLength;
    double GateBulkOverlapCap;
    double GateDrainOverlapCap;
    double GateSourceOverlapCap;
    double SourceSatCur;
    double arg;
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

    /*  loop through all the VDMOS device models */
    for( ; model != NULL; model = VDMOSnextModel(model)) {
        /* VDMOS capacitance parameters */
        const double cgdmin = model->VDMOScgdmin;
        const double cgdmax = model->VDMOScgdmax;
        const double a = model->VDMOSa;
        const double cgs = model->VDMOScgs;

        /* loop through all the instances of the model */
        for (here = VDMOSinstances(model); here != NULL ;
	     here=VDMOSnextInstance(here)) {

            vt = CONSTKoverQ * here->VDMOStemp;
            Check=1;
/*
  
*/

            /* first, we compute a few useful values - these could be
             * pre-computed, but for historical reasons are still done
             * here.  They may be moved at the expense of instance size
             */

            EffectiveLength=here->VDMOSl - 2*model->VDMOSlatDiff;
            
            if( (here->VDMOStSatCurDens == 0) || 
                    (here->VDMOSdrainArea == 0) ||
                    (here->VDMOSsourceArea == 0)) {
                DrainSatCur = here->VDMOSm * here->VDMOStSatCur;
                SourceSatCur = here->VDMOSm * here->VDMOStSatCur;
            } else {
                DrainSatCur = here->VDMOStSatCurDens * 
                        here->VDMOSm * here->VDMOSdrainArea;
                SourceSatCur = here->VDMOStSatCurDens * 
                        here->VDMOSm * here->VDMOSsourceArea;
            }
            GateSourceOverlapCap = model->VDMOSgateSourceOverlapCapFactor * 
                    here->VDMOSm * here->VDMOSw;
            GateDrainOverlapCap = model->VDMOSgateDrainOverlapCapFactor * 
                    here->VDMOSm * here->VDMOSw;
            GateBulkOverlapCap = model->VDMOSgateBulkOverlapCapFactor * 
                    here->VDMOSm * EffectiveLength;
            Beta = here->VDMOStTransconductance * here->VDMOSm *
                    here->VDMOSw/EffectiveLength;
           
            /* 
             * ok - now to do the start-up operations
             *
             * we must get values for vbs, vds, and vgs from somewhere
             * so we either predict them or recover them from last iteration
             * These are the two most common cases - either a prediction
             * step or the general iteration step and they
             * share some code, so we put them first - others later on
             */


            if((ckt->CKTmode & (MODEINITFLOAT | MODEINITPRED | MODEINITSMSIG
				| MODEINITTRAN)) ||
	       ( (ckt->CKTmode & MODEINITFIX) && (!here->VDMOSoff) )  ) {
#ifndef PREDICTOR
                if(ckt->CKTmode & (MODEINITPRED | MODEINITTRAN) ) {

                    /* predictor step */

                    xfact=ckt->CKTdelta/ckt->CKTdeltaOld[1];
                    *(ckt->CKTstate0 + here->VDMOSvbs) = 
			*(ckt->CKTstate1 + here->VDMOSvbs);
                    vbs = (1+xfact)* (*(ckt->CKTstate1 + here->VDMOSvbs))
			-(xfact * (*(ckt->CKTstate2 + here->VDMOSvbs)));
                    *(ckt->CKTstate0 + here->VDMOSvgs) = 
			*(ckt->CKTstate1 + here->VDMOSvgs);
                    vgs = (1+xfact)* (*(ckt->CKTstate1 + here->VDMOSvgs))
			-(xfact * (*(ckt->CKTstate2 + here->VDMOSvgs)));
                    *(ckt->CKTstate0 + here->VDMOSvds) = 
			*(ckt->CKTstate1 + here->VDMOSvds);
                    vds = (1+xfact)* (*(ckt->CKTstate1 + here->VDMOSvds))
			-(xfact * (*(ckt->CKTstate2 + here->VDMOSvds)));
                    *(ckt->CKTstate0 + here->VDMOSvbd) = 
			*(ckt->CKTstate0 + here->VDMOSvbs)-
			*(ckt->CKTstate0 + here->VDMOSvds);
                } else {
#endif /* PREDICTOR */

                    /* general iteration */

                    vbs = model->VDMOStype * ( 
                        *(ckt->CKTrhsOld+here->VDMOSbNode) -
                        *(ckt->CKTrhsOld+here->VDMOSsNodePrime));
                    vgs = model->VDMOStype * ( 
                        *(ckt->CKTrhsOld+here->VDMOSgNode) -
                        *(ckt->CKTrhsOld+here->VDMOSsNodePrime));
                    vds = model->VDMOStype * ( 
                        *(ckt->CKTrhsOld+here->VDMOSdNodePrime) -
                        *(ckt->CKTrhsOld+here->VDMOSsNodePrime));
#ifndef PREDICTOR
                }
#endif /* PREDICTOR */

                /* now some common crunching for some more useful quantities */

                vbd=vbs-vds;
                vgd=vgs-vds;
                vgdo = *(ckt->CKTstate0 + here->VDMOSvgs) - 
		    *(ckt->CKTstate0 + here->VDMOSvds);
                delvbs = vbs - *(ckt->CKTstate0 + here->VDMOSvbs);
                delvbd = vbd - *(ckt->CKTstate0 + here->VDMOSvbd);
                delvgs = vgs - *(ckt->CKTstate0 + here->VDMOSvgs);
                delvds = vds - *(ckt->CKTstate0 + here->VDMOSvds);
                delvgd = vgd-vgdo;

                /* these are needed for convergence testing */

                if (here->VDMOSmode >= 0) {
                    cdhat=
                        here->VDMOScd-
                        here->VDMOSgbd * delvbd +
                        here->VDMOSgmbs * delvbs +
                        here->VDMOSgm * delvgs + 
                        here->VDMOSgds * delvds ;
                } else {
                    cdhat=
                        here->VDMOScd -
                        ( here->VDMOSgbd -
			  here->VDMOSgmbs) * delvbd -
                        here->VDMOSgm * delvgd + 
                        here->VDMOSgds * delvds ;
                }
                cbhat=
                    here->VDMOScbs +
                    here->VDMOScbd +
                    here->VDMOSgbd * delvbd +
                    here->VDMOSgbs * delvbs ;
/*
  
*/

#ifndef NOBYPASS
                /* now lets see if we can bypass (ugh) */
                tempv = (MAX(fabs(cbhat),
			    fabs(here->VDMOScbs + here->VDMOScbd)) +
			 ckt->CKTabstol);
                if ((!(ckt->CKTmode &
		       (MODEINITPRED|MODEINITTRAN|MODEINITSMSIG))) &&
		    (ckt->CKTbypass) &&
		    (fabs(cbhat-(here->VDMOScbs + 
				 here->VDMOScbd)) < ckt->CKTreltol * tempv) &&
		    (fabs(delvbs) < (ckt->CKTreltol *
				     MAX(fabs(vbs),
					 fabs( *(ckt->CKTstate0 +
						 here->VDMOSvbs))) +
				     ckt->CKTvoltTol)) &&
		    (fabs(delvbd) < (ckt->CKTreltol *
				     MAX(fabs(vbd),
					 fabs(*(ckt->CKTstate0 +
						here->VDMOSvbd))) +
				     ckt->CKTvoltTol)) &&
		    (fabs(delvgs) < (ckt->CKTreltol *
				     MAX(fabs(vgs),
					 fabs(*(ckt->CKTstate0 +
						here->VDMOSvgs)))+
				     ckt->CKTvoltTol)) &&
		    (fabs(delvds) < (ckt->CKTreltol *
				     MAX(fabs(vds),
					 fabs(*(ckt->CKTstate0 +
						here->VDMOSvds))) +
				     ckt->CKTvoltTol)) &&
		    (fabs(cdhat- here->VDMOScd) < (ckt->CKTreltol *
						  MAX(fabs(cdhat),
						      fabs(here->VDMOScd)) +
						  ckt->CKTabstol))) {
		  /* bypass code */
		  /* nothing interesting has changed since last
		   * iteration on this device, so we just
		   * copy all the values computed last iteration out
		   * and keep going
		   */
		  vbs = *(ckt->CKTstate0 + here->VDMOSvbs);
		  vbd = *(ckt->CKTstate0 + here->VDMOSvbd);
		  vgs = *(ckt->CKTstate0 + here->VDMOSvgs);
		  vds = *(ckt->CKTstate0 + here->VDMOSvds);
		  vgd = vgs - vds;
		  vgb = vgs - vbs;
		  cdrain = here->VDMOSmode * (here->VDMOScd + here->VDMOScbd);
		  if(ckt->CKTmode & (MODETRAN | MODETRANOP)) {
		    capgs = ( *(ckt->CKTstate0+here->VDMOScapgs)+ 
			      *(ckt->CKTstate1+here->VDMOScapgs) +
			      GateSourceOverlapCap );
		    capgd = ( *(ckt->CKTstate0+here->VDMOScapgd)+ 
			      *(ckt->CKTstate1+here->VDMOScapgd) +
			      GateDrainOverlapCap );
		    capgb = ( *(ckt->CKTstate0+here->VDMOScapgb)+ 
			      *(ckt->CKTstate1+here->VDMOScapgb) +
			      GateBulkOverlapCap );
		    
		  }
		  goto bypass;
		}
#endif /*NOBYPASS*/

/*
  
*/

                /* ok - bypass is out, do it the hard way */

                von = model->VDMOStype * here->VDMOSvon;

#ifndef NODELIMITING
                /* 
                 * limiting
                 *  we want to keep device voltages from changing
                 * so fast that the exponentials churn out overflows
                 * and similar rudeness
                 */

                if(*(ckt->CKTstate0 + here->VDMOSvds) >=0) {
                    vgs = DEVfetlim(vgs,*(ckt->CKTstate0 + here->VDMOSvgs)
				    ,von);
                    vds = vgs - vgd;
                    vds = DEVlimvds(vds,*(ckt->CKTstate0 + here->VDMOSvds));
                    vgd = vgs - vds;
                } else {
                    vgd = DEVfetlim(vgd,vgdo,von);
                    vds = vgs - vgd;
                    if(!(ckt->CKTfixLimit)) {
                        vds = -DEVlimvds(-vds,-(*(ckt->CKTstate0 + 
						  here->VDMOSvds)));
                    }
                    vgs = vgd + vds;
                }
                if(vds >= 0) {
                    vbs = DEVpnjlim(vbs,*(ckt->CKTstate0 + here->VDMOSvbs),
				    vt,here->VDMOSsourceVcrit,&Check);
                    vbd = vbs-vds;
                } else {
                    vbd = DEVpnjlim(vbd,*(ckt->CKTstate0 + here->VDMOSvbd),
				    vt,here->VDMOSdrainVcrit,&Check);
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

                if((ckt->CKTmode & MODEINITJCT) && !here->VDMOSoff) {
                    vds= model->VDMOStype * here->VDMOSicVDS;
                    vgs= model->VDMOStype * here->VDMOSicVGS;
                    vbs= model->VDMOStype * here->VDMOSicVBS;
                    if((vds==0) && (vgs==0) && (vbs==0) && 
		       ((ckt->CKTmode & 
			 (MODETRAN|MODEDCOP|MODEDCTRANCURVE)) ||
			(!(ckt->CKTmode & MODEUIC)))) {
                        vbs = -1;
                        vgs = model->VDMOStype * here->VDMOStVto;
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
            if(vbs <= -3*vt) {
                here->VDMOSgbs = ckt->CKTgmin;
                here->VDMOScbs = here->VDMOSgbs*vbs-SourceSatCur;
            } else {
                evbs = exp(MIN(MAX_EXP_ARG,vbs/vt));
                here->VDMOSgbs = SourceSatCur*evbs/vt + ckt->CKTgmin;
                here->VDMOScbs = SourceSatCur*(evbs-1) + ckt->CKTgmin*vbs;
            }
            if(vbd <= -3*vt) {
                here->VDMOSgbd = ckt->CKTgmin;
                here->VDMOScbd = here->VDMOSgbd*vbd-DrainSatCur;
            } else {
                evbd = exp(MIN(MAX_EXP_ARG,vbd/vt));
                here->VDMOSgbd = DrainSatCur*evbd/vt + ckt->CKTgmin;
                here->VDMOScbd = DrainSatCur*(evbd-1) + ckt->CKTgmin*vbd;
            }
            /* now to determine whether the user was able to correctly
             * identify the source and drain of his device
             */
            if(vds >= 0) {
                /* normal mode */
                here->VDMOSmode = 1;
            } else {
                /* inverse mode */
                here->VDMOSmode = -1;
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

                if ((here->VDMOSmode==1?vbs:vbd) <= 0 ) {
                    sarg=sqrt(here->VDMOStPhi-(here->VDMOSmode==1?vbs:vbd));
                } else {
                    sarg=sqrt(here->VDMOStPhi);
                    sarg=sarg-(here->VDMOSmode==1?vbs:vbd)/(sarg+sarg);
                    sarg=MAX(0,sarg);
                }
                von=(here->VDMOStVbi*model->VDMOStype)+model->VDMOSgamma*sarg;
                vgst=(here->VDMOSmode==1?vgs:vgd)-von;
                vdsat=MAX(vgst,0);
                if (sarg <= 0) {
                    arg=0;
                } else {
                    arg=model->VDMOSgamma/(sarg+sarg);
                }
                if (vgst <= 0) {
                    /*
                     *     cutoff region
                     */
                    cdrain=0;
                    here->VDMOSgm=0;
                    here->VDMOSgds=0;
                    here->VDMOSgmbs=0;
                } else{
                    /*
                     *     saturation region
                     */
                    betap=Beta*(1+model->VDMOSlambda*(vds*here->VDMOSmode));
                    if (vgst <= (vds*here->VDMOSmode)){
                        cdrain=betap*vgst*vgst*.5;
                        here->VDMOSgm=betap*vgst;
                        here->VDMOSgds=model->VDMOSlambda*Beta*vgst*vgst*.5;
                        here->VDMOSgmbs=here->VDMOSgm*arg;
                    } else {
			/*
			 *     linear region
			 */
                        cdrain=betap*(vds*here->VDMOSmode)*
                            (vgst-.5*(vds*here->VDMOSmode));
                        here->VDMOSgm=betap*(vds*here->VDMOSmode);
                        here->VDMOSgds=betap*(vgst-(vds*here->VDMOSmode))+
			    model->VDMOSlambda*Beta*
			    (vds*here->VDMOSmode)*
			    (vgst-.5*(vds*here->VDMOSmode));
                        here->VDMOSgmbs=here->VDMOSgm*arg;
                    }
                }
                /*
                 *     finished
                 */
            }
/*
  
*/

            /* now deal with n vs p polarity */

            here->VDMOSvon = model->VDMOStype * von;
            here->VDMOSvdsat = model->VDMOStype * vdsat;
            /* line 490 */
            /*
             *  COMPUTE EQUIVALENT DRAIN CURRENT SOURCE
             */
            here->VDMOScd=here->VDMOSmode * cdrain - here->VDMOScbd;

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
                {
                    /* can't bypass the diode capacitance calculations */
                    if(here->VDMOSCbs != 0 || here->VDMOSCbssw != 0 ) {
			if (vbs < here->VDMOStDepCap){
			    arg=1-vbs/here->VDMOStBulkPot;
			    /*
			     * the following block looks somewhat long and messy,
			     * but since most users use the default grading
			     * coefficients of .5, and sqrt is MUCH faster than an
			     * exp(log()) we use this special case code to buy time.
			     * (as much as 10% of total job time!)
			     */
			    if(model->VDMOSbulkJctBotGradingCoeff ==
			       model->VDMOSbulkJctSideGradingCoeff) {
				if(model->VDMOSbulkJctBotGradingCoeff == .5) {
				    sarg = sargsw = 1/sqrt(arg);
				} else {
				    sarg = sargsw =
                                        exp(-model->VDMOSbulkJctBotGradingCoeff*
					    log(arg));
				}
			    } else {
				if(model->VDMOSbulkJctBotGradingCoeff == .5) {
				    sarg = 1/sqrt(arg);
				} else {
				    sarg = exp(-model->VDMOSbulkJctBotGradingCoeff*
					       log(arg));
				}
				if(model->VDMOSbulkJctSideGradingCoeff == .5) {
				    sargsw = 1/sqrt(arg);
				} else {
				    sargsw =exp(-model->VDMOSbulkJctSideGradingCoeff*
						log(arg));
				}
			    }
			    *(ckt->CKTstate0 + here->VDMOSqbs) =
				here->VDMOStBulkPot*(here->VDMOSCbs*
						    (1-arg*sarg)/(1-model->VDMOSbulkJctBotGradingCoeff)
						    +here->VDMOSCbssw*
						    (1-arg*sargsw)/
						    (1-model->VDMOSbulkJctSideGradingCoeff));
			    here->VDMOScapbs=here->VDMOSCbs*sarg+
                                here->VDMOSCbssw*sargsw;
			} else {
			    *(ckt->CKTstate0 + here->VDMOSqbs) = here->VDMOSf4s +
                                vbs*(here->VDMOSf2s+vbs*(here->VDMOSf3s/2));
			    here->VDMOScapbs=here->VDMOSf2s+here->VDMOSf3s*vbs;
			}
                    } else {
			*(ckt->CKTstate0 + here->VDMOSqbs) = 0;
                        here->VDMOScapbs=0;
                    }
                }
                {
                    if(here->VDMOSCbd != 0 || here->VDMOSCbdsw != 0 ) {
			if (vbd < here->VDMOStDepCap) {
			    arg=1-vbd/here->VDMOStBulkPot;
			    /*
			     * the following block looks somewhat long and messy,
			     * but since most users use the default grading
			     * coefficients of .5, and sqrt is MUCH faster than an
			     * exp(log()) we use this special case code to buy time.
			     * (as much as 10% of total job time!)
			     */
			    if(model->VDMOSbulkJctBotGradingCoeff == .5 &&
			       model->VDMOSbulkJctSideGradingCoeff == .5) {
				sarg = sargsw = 1/sqrt(arg);
			    } else {
				if(model->VDMOSbulkJctBotGradingCoeff == .5) {
				    sarg = 1/sqrt(arg);
				} else {
				    sarg = exp(-model->VDMOSbulkJctBotGradingCoeff*
					       log(arg));
				}
				if(model->VDMOSbulkJctSideGradingCoeff == .5) {
				    sargsw = 1/sqrt(arg);
				} else {
				    sargsw =exp(-model->VDMOSbulkJctSideGradingCoeff*
						log(arg));
				}
			    }
			    *(ckt->CKTstate0 + here->VDMOSqbd) =
				here->VDMOStBulkPot*(here->VDMOSCbd*
						    (1-arg*sarg)
						    /(1-model->VDMOSbulkJctBotGradingCoeff)
						    +here->VDMOSCbdsw*
						    (1-arg*sargsw)
						    /(1-model->VDMOSbulkJctSideGradingCoeff));
			    here->VDMOScapbd=here->VDMOSCbd*sarg+
                                here->VDMOSCbdsw*sargsw;
			} else {
			    *(ckt->CKTstate0 + here->VDMOSqbd) = here->VDMOSf4d +
                                vbd * (here->VDMOSf2d + vbd * here->VDMOSf3d/2);
			    here->VDMOScapbd=here->VDMOSf2d + vbd * here->VDMOSf3d;
			}
		    } else {
			*(ckt->CKTstate0 + here->VDMOSqbd) = 0;
			here->VDMOScapbd = 0;
		    }
                }
/*
  
*/


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

                    error = NIintegrate(ckt,&geq,&ceq,here->VDMOScapbd,
					here->VDMOSqbd);
                    if(error) return(error);
                    here->VDMOSgbd += geq;
                    here->VDMOScbd += *(ckt->CKTstate0 + here->VDMOScqbd);
                    here->VDMOScd -= *(ckt->CKTstate0 + here->VDMOScqbd);
                    error = NIintegrate(ckt,&geq,&ceq,here->VDMOScapbs,
					here->VDMOSqbs);
                    if(error) return(error);
                    here->VDMOSgbs += geq;
                    here->VDMOScbs += *(ckt->CKTstate0 + here->VDMOScqbs);
                }
            }
/*
  
*/


            /*
             *  check convergence
             */
            if ( (here->VDMOSoff == 0)  || 
		 (!(ckt->CKTmode & (MODEINITFIX|MODEINITSMSIG))) ){
                if (Check == 1) {
                    ckt->CKTnoncon++;
		    ckt->CKTtroubleElt = (GENinstance *) here;
                }
            }
/*
  
*/

            /* save things away for next time */

            *(ckt->CKTstate0 + here->VDMOSvbs) = vbs;
            *(ckt->CKTstate0 + here->VDMOSvbd) = vbd;
            *(ckt->CKTstate0 + here->VDMOSvgs) = vgs;
            *(ckt->CKTstate0 + here->VDMOSvds) = vds;

/*
  
*/

            /*
             * vdmos capacitor model
             */
            if ( ckt->CKTmode & (MODETRAN | MODETRANOP | MODEINITSMSIG) ) {
                /*
                 * calculate gate - drain, gate - source capacitors
                 * drain-source capacitor is evaluated with the bulk diode below
                 */
                /*
                 * this just evaluates at the current time,
                 * expects you to remember values from previous time
                 * returns 1/2 of non-constant portion of capacitance
                 * you must add in the other half from previous time
                 * and the constant part
                 */
                DevCapVDMOS(vgd, cgdmin, cgdmax, a, cgs,
                            (ckt->CKTstate0 + here->VDMOScapgs),
                            (ckt->CKTstate0 + here->VDMOScapgd),
                            (ckt->CKTstate0 + here->VDMOScapgb));

                vgs1 = *(ckt->CKTstate1 + here->VDMOSvgs);
                vgd1 = vgs1 - *(ckt->CKTstate1 + here->VDMOSvds);
                vgb1 = vgs1 - *(ckt->CKTstate1 + here->VDMOSvbs);
                if(ckt->CKTmode & (MODETRANOP|MODEINITSMSIG)) {
                    capgs =  2 * *(ckt->CKTstate0+here->VDMOScapgs)+ 
			GateSourceOverlapCap ;
                    capgd =  2 * *(ckt->CKTstate0+here->VDMOScapgd)+ 
			GateDrainOverlapCap ;
                    capgb =  2 * *(ckt->CKTstate0+here->VDMOScapgb)+ 
			GateBulkOverlapCap ;
                } else {
                    capgs = ( *(ckt->CKTstate0+here->VDMOScapgs)+ 
                              *(ckt->CKTstate1+here->VDMOScapgs) +
                              GateSourceOverlapCap );
                    capgd = ( *(ckt->CKTstate0+here->VDMOScapgd)+ 
                              *(ckt->CKTstate1+here->VDMOScapgd) +
                              GateDrainOverlapCap );
                    capgb = ( *(ckt->CKTstate0+here->VDMOScapgb)+ 
                              *(ckt->CKTstate1+here->VDMOScapgb) +
                              GateBulkOverlapCap );
                }
/*
  
*/

#ifndef PREDICTOR
                if (ckt->CKTmode & (MODEINITPRED | MODEINITTRAN) ) {
                    *(ckt->CKTstate0 + here->VDMOSqgs) =
                        (1+xfact) * *(ckt->CKTstate1 + here->VDMOSqgs)
                        - xfact * *(ckt->CKTstate2 + here->VDMOSqgs);
                    *(ckt->CKTstate0 + here->VDMOSqgd) =
                        (1+xfact) * *(ckt->CKTstate1 + here->VDMOSqgd)
                        - xfact * *(ckt->CKTstate2 + here->VDMOSqgd);
                    *(ckt->CKTstate0 + here->VDMOSqgb) =
                        (1+xfact) * *(ckt->CKTstate1 + here->VDMOSqgb)
                        - xfact * *(ckt->CKTstate2 + here->VDMOSqgb);
                } else {
#endif /*PREDICTOR*/
                    if(ckt->CKTmode & MODETRAN) {
                        *(ckt->CKTstate0 + here->VDMOSqgs) = (vgs-vgs1)*capgs +
                            *(ckt->CKTstate1 + here->VDMOSqgs) ;
                        *(ckt->CKTstate0 + here->VDMOSqgd) = (vgd-vgd1)*capgd +
                            *(ckt->CKTstate1 + here->VDMOSqgd) ;
                        *(ckt->CKTstate0 + here->VDMOSqgb) = (vgb-vgb1)*capgb +
                            *(ckt->CKTstate1 + here->VDMOSqgb) ;
                    } else {
                        /* TRANOP only */
                        *(ckt->CKTstate0 + here->VDMOSqgs) = vgs*capgs;
                        *(ckt->CKTstate0 + here->VDMOSqgd) = vgd*capgd;
                        *(ckt->CKTstate0 + here->VDMOSqgb) = vgb*capgb;
                    }
#ifndef PREDICTOR
                }
#endif /*PREDICTOR*/
            }
#ifndef NOBYPASS
	bypass:
#endif

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
                if(capgs == 0) *(ckt->CKTstate0 + here->VDMOScqgs) =0;
                if(capgd == 0) *(ckt->CKTstate0 + here->VDMOScqgd) =0;
                if(capgb == 0) *(ckt->CKTstate0 + here->VDMOScqgb) =0;
                /*
                 *    calculate equivalent conductances and currents for
                 *    meyer"s capacitors
                 */
                error = NIintegrate(ckt,&gcgs,&ceqgs,capgs,here->VDMOSqgs);
                if(error) return(error);
                error = NIintegrate(ckt,&gcgd,&ceqgd,capgd,here->VDMOSqgd);
                if(error) return(error);
                error = NIintegrate(ckt,&gcgb,&ceqgb,capgb,here->VDMOSqgb);
                if(error) return(error);
                ceqgs=ceqgs-gcgs*vgs+ckt->CKTag[0]* 
		    *(ckt->CKTstate0 + here->VDMOSqgs);
                ceqgd=ceqgd-gcgd*vgd+ckt->CKTag[0]*
		    *(ckt->CKTstate0 + here->VDMOSqgd);
                ceqgb=ceqgb-gcgb*vgb+ckt->CKTag[0]*
		    *(ckt->CKTstate0 + here->VDMOSqgb);
            }
            /*
             *     store charge storage info for meyer's cap in lx table
             */

            /*
             *  load current vector
             */
            ceqbs = model->VDMOStype * 
		(here->VDMOScbs-(here->VDMOSgbs)*vbs);
            ceqbd = model->VDMOStype * 
		(here->VDMOScbd-(here->VDMOSgbd)*vbd);
            if (here->VDMOSmode >= 0) {
                xnrm=1;
                xrev=0;
                cdreq=model->VDMOStype*(cdrain-here->VDMOSgds*vds-
				       here->VDMOSgm*vgs-here->VDMOSgmbs*vbs);
            } else {
                xnrm=0;
                xrev=1;
                cdreq = -(model->VDMOStype)*(cdrain-here->VDMOSgds*(-vds)-
					    here->VDMOSgm*vgd-here->VDMOSgmbs*vbd);
            }
            *(ckt->CKTrhs + here->VDMOSgNode) -= 
                (model->VDMOStype * (ceqgs + ceqgb + ceqgd));
            *(ckt->CKTrhs + here->VDMOSbNode) -=
                (ceqbs + ceqbd - model->VDMOStype * ceqgb);
            *(ckt->CKTrhs + here->VDMOSdNodePrime) +=
		(ceqbd - cdreq + model->VDMOStype * ceqgd);
            *(ckt->CKTrhs + here->VDMOSsNodePrime) += 
		cdreq + ceqbs + model->VDMOStype * ceqgs;
            /*
             *  load y matrix
             */

            *(here->VDMOSDdPtr) += (here->VDMOSdrainConductance);
            *(here->VDMOSGgPtr) += ((gcgd+gcgs+gcgb));
            *(here->VDMOSSsPtr) += (here->VDMOSsourceConductance);
            *(here->VDMOSBbPtr) += (here->VDMOSgbd+here->VDMOSgbs+gcgb);
            *(here->VDMOSDPdpPtr) += 
		(here->VDMOSdrainConductance+here->VDMOSgds+
		 here->VDMOSgbd+xrev*(here->VDMOSgm+here->VDMOSgmbs)+gcgd);
            *(here->VDMOSSPspPtr) += 
		(here->VDMOSsourceConductance+here->VDMOSgds+
		 here->VDMOSgbs+xnrm*(here->VDMOSgm+here->VDMOSgmbs)+gcgs);
            *(here->VDMOSDdpPtr) += (-here->VDMOSdrainConductance);
            *(here->VDMOSGbPtr) -= gcgb;
            *(here->VDMOSGdpPtr) -= gcgd;
            *(here->VDMOSGspPtr) -= gcgs;
            *(here->VDMOSSspPtr) += (-here->VDMOSsourceConductance);
            *(here->VDMOSBgPtr) -= gcgb;
            *(here->VDMOSBdpPtr) -= here->VDMOSgbd;
            *(here->VDMOSBspPtr) -= here->VDMOSgbs;
            *(here->VDMOSDPdPtr) += (-here->VDMOSdrainConductance);
            *(here->VDMOSDPgPtr) += ((xnrm-xrev)*here->VDMOSgm-gcgd);
            *(here->VDMOSDPbPtr) += (-here->VDMOSgbd+(xnrm-xrev)*here->VDMOSgmbs);
            *(here->VDMOSDPspPtr) += (-here->VDMOSgds-xnrm*
				     (here->VDMOSgm+here->VDMOSgmbs));
            *(here->VDMOSSPgPtr) += (-(xnrm-xrev)*here->VDMOSgm-gcgs);
            *(here->VDMOSSPsPtr) += (-here->VDMOSsourceConductance);
            *(here->VDMOSSPbPtr) += (-here->VDMOSgbs-(xnrm-xrev)*here->VDMOSgmbs);
            *(here->VDMOSSPdpPtr) += (-here->VDMOSgds-xrev*
				     (here->VDMOSgm+here->VDMOSgmbs));
        }
    }
    return(OK);
}
