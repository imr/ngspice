/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "mos3defs.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* actually load the current value into the sparse matrix previously
 * provided */
int
MOS3load(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS3model *model = (MOS3model *)inModel;
    MOS3instance *here;
    double Beta;
    double DrainSatCur;
    double EffectiveLength;
    double EffectiveWidth;
    double GateBulkOverlapCap;
    double GateDrainOverlapCap;
    double GateSourceOverlapCap;
    double OxideCap;
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
    double vt;  /* vt at instance temperature */

#ifdef CAPBYPASS
    senflag = 0;
#endif /* CAPBYPASS */
    if(ckt->CKTsenInfo){
        if(ckt->CKTsenInfo->SENstatus == PERTURBATION) {
            if((ckt->CKTsenInfo->SENmode == ACSEN)||
	       (ckt->CKTsenInfo->SENmode == TRANSEN)){
#ifdef CAPBYPASS
                senflag = 1;
#endif /* CAPBYPASS */
	       	       
            }
            goto next;
        }
    }

    /*  loop through all the MOS3 device models */
 next: 
    for( ; model != NULL; model = MOS3nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = MOS3instances(model); here != NULL ;
	     here=MOS3nextInstance(here)) {

            vt = CONSTKoverQ * here->MOS3temp;
            Check=1;

            if(ckt->CKTsenInfo){
#ifdef SENSDEBUG
                printf("MOS3load \n");
#endif /* SENSDEBUG */

                if(ckt->CKTsenInfo->SENstatus == PERTURBATION)
                    if(here->MOS3senPertFlag == OFF)continue;

            }
            SenCond = ckt->CKTsenInfo && here->MOS3senPertFlag;

            /* first, we compute a few useful values - these could be
             * pre-computed, but for historical reasons are still done
             * here.  They may be moved at the expense of instance
             * size */

            EffectiveWidth=here->MOS3w-2*model->MOS3widthNarrow+
		model->MOS3widthAdjust;
            EffectiveLength=here->MOS3l - 2*model->MOS3latDiff+
		model->MOS3lengthAdjust;
           
	    if( (here->MOS3tSatCurDens == 0) || 
		(here->MOS3drainArea == 0) ||
		(here->MOS3sourceArea == 0)) {
                DrainSatCur = here->MOS3m * here->MOS3tSatCur;
                SourceSatCur = here->MOS3m * here->MOS3tSatCur;
            } else {
                DrainSatCur = here->MOS3m * here->MOS3tSatCurDens * 
		    here->MOS3drainArea;
                SourceSatCur = here->MOS3m * here->MOS3tSatCurDens * 
		    here->MOS3sourceArea;
            }
            GateSourceOverlapCap = model->MOS3gateSourceOverlapCapFactor * 
		here->MOS3m * EffectiveWidth;
            GateDrainOverlapCap = model->MOS3gateDrainOverlapCapFactor * 
		here->MOS3m * EffectiveWidth;
            GateBulkOverlapCap = model->MOS3gateBulkOverlapCapFactor * 
		here->MOS3m * EffectiveLength;
            Beta = here->MOS3tTransconductance *
		here->MOS3m * EffectiveWidth/EffectiveLength;
            OxideCap = model->MOS3oxideCapFactor * EffectiveLength * 
		here->MOS3m * EffectiveWidth; 
           
            if(SenCond){
#ifdef SENSDEBUG
                printf("MOS3senPertFlag = ON \n");
#endif /* SENSDEBUG */
                if((ckt->CKTsenInfo->SENmode == TRANSEN) &&
		   (ckt->CKTmode & MODEINITTRAN)) {
                    vgs = *(ckt->CKTstate1 + here->MOS3vgs);
                    vds = *(ckt->CKTstate1 + here->MOS3vds);
                    vbs = *(ckt->CKTstate1 + here->MOS3vbs);
                    vbd = *(ckt->CKTstate1 + here->MOS3vbd);
                    vgb = vgs - vbs;
                    vgd = vgs - vds;
                }
                else if (ckt->CKTsenInfo->SENmode == ACSEN){
                    vgb = model->MOS3type * ( 
                        *(ckt->CKTrhsOp+here->MOS3gNode) -
                        *(ckt->CKTrhsOp+here->MOS3bNode));
                    vbs = *(ckt->CKTstate0 + here->MOS3vbs);
                    vbd = *(ckt->CKTstate0 + here->MOS3vbd);
                    vgd = vgb + vbd ;
                    vgs = vgb + vbs ;
                    vds = vbs - vbd ;
                }
                else{
                    vgs = *(ckt->CKTstate0 + here->MOS3vgs);
                    vds = *(ckt->CKTstate0 + here->MOS3vds);
                    vbs = *(ckt->CKTstate0 + here->MOS3vbs);
                    vbd = *(ckt->CKTstate0 + here->MOS3vbd);
                    vgb = vgs - vbs;
                    vgd = vgs - vds;
                }
#ifdef SENSDEBUG
                printf(" vbs = %.7e ,vbd = %.7e,vgb = %.7e\n",vbs,vbd,vgb);
                printf(" vgs = %.7e ,vds = %.7e,vgd = %.7e\n",vgs,vds,vgd);
#endif /* SENSDEBUG */
                goto next1;
            }


            /* 
             * ok - now to do the start-up operations
             *
             * we must get values for vbs, vds, and vgs from somewhere
             * so we either predict them or recover them from last
             * iteration These are the two most common cases - either
             * a prediction step or the general iteration step and
             * they share some code, so we put them first - others
             * later on */

            if((ckt->CKTmode & (MODEINITFLOAT | MODEINITPRED | MODEINITSMSIG | 
				MODEINITTRAN)) ||
	       ( (ckt->CKTmode & MODEINITFIX) && (!here->MOS3off) )  ) {
#ifndef PREDICTOR
                if(ckt->CKTmode & (MODEINITPRED | MODEINITTRAN) ) {

                    /* predictor step */

                    xfact=ckt->CKTdelta/ckt->CKTdeltaOld[1];
                    *(ckt->CKTstate0 + here->MOS3vbs) = 
			*(ckt->CKTstate1 + here->MOS3vbs);
                    vbs = (1+xfact)* (*(ckt->CKTstate1 + here->MOS3vbs))
			-(xfact * (*(ckt->CKTstate2 + here->MOS3vbs)));
                    *(ckt->CKTstate0 + here->MOS3vgs) = 
			*(ckt->CKTstate1 + here->MOS3vgs);
                    vgs = (1+xfact)* (*(ckt->CKTstate1 + here->MOS3vgs))
			-(xfact * (*(ckt->CKTstate2 + here->MOS3vgs)));
                    *(ckt->CKTstate0 + here->MOS3vds) = 
			*(ckt->CKTstate1 + here->MOS3vds);
                    vds = (1+xfact)* (*(ckt->CKTstate1 + here->MOS3vds))
			-(xfact * (*(ckt->CKTstate2 + here->MOS3vds)));
                    *(ckt->CKTstate0 + here->MOS3vbd) = 
			*(ckt->CKTstate0 + here->MOS3vbs)-
			*(ckt->CKTstate0 + here->MOS3vds);
                } else {
#endif /*PREDICTOR*/

                    /* general iteration */

                    vbs = model->MOS3type * ( 
                        *(ckt->CKTrhsOld+here->MOS3bNode) -
                        *(ckt->CKTrhsOld+here->MOS3sNodePrime));
                    vgs = model->MOS3type * ( 
                        *(ckt->CKTrhsOld+here->MOS3gNode) -
                        *(ckt->CKTrhsOld+here->MOS3sNodePrime));
                    vds = model->MOS3type * ( 
                        *(ckt->CKTrhsOld+here->MOS3dNodePrime) -
                        *(ckt->CKTrhsOld+here->MOS3sNodePrime));
#ifndef PREDICTOR
                }
#endif /*PREDICTOR*/

                /* now some common crunching for some more useful quantities */

                vbd=vbs-vds;
                vgd=vgs-vds;
                vgdo = *(ckt->CKTstate0 + here->MOS3vgs) - 
		    *(ckt->CKTstate0 + here->MOS3vds);
                delvbs = vbs - *(ckt->CKTstate0 + here->MOS3vbs);
                delvbd = vbd - *(ckt->CKTstate0 + here->MOS3vbd);
                delvgs = vgs - *(ckt->CKTstate0 + here->MOS3vgs);
                delvds = vds - *(ckt->CKTstate0 + here->MOS3vds);
                delvgd = vgd-vgdo;

                /* these are needed for convergence testing */

                if (here->MOS3mode >= 0) {
                    cdhat=
                        here->MOS3cd-
                        here->MOS3gbd * delvbd +
                        here->MOS3gmbs * delvbs +
                        here->MOS3gm * delvgs + 
                        here->MOS3gds * delvds ;
                } else {
                    cdhat=
                        here->MOS3cd -
                        ( here->MOS3gbd -
			  here->MOS3gmbs) * delvbd -
                        here->MOS3gm * delvgd + 
                        here->MOS3gds * delvds ;
                }
                cbhat=
                    here->MOS3cbs +
                    here->MOS3cbd +
                    here->MOS3gbd * delvbd +
                    here->MOS3gbs * delvbs ;
#ifndef NOBYPASS
                /* now lets see if we can bypass (ugh) */
                /* the following mess should be one if statement, but
                 * many compilers can't handle it all at once, so it
                 * is split into several successive if statements
                 */
                tempv = MAX(fabs(cbhat),fabs(here->MOS3cbs
					     + here->MOS3cbd))+ckt->CKTabstol;
                if((!(ckt->CKTmode & (MODEINITPRED|MODEINITTRAN|MODEINITSMSIG)
		    )) && (ckt->CKTbypass) )
		    if ( (fabs(cbhat-(here->MOS3cbs + 
				      here->MOS3cbd)) < ckt->CKTreltol * 
			  tempv)) 
			if( (fabs(delvbs) < (ckt->CKTreltol * MAX(fabs(vbs),
								  fabs(*(ckt->CKTstate0+here->MOS3vbs)))+
					     ckt->CKTvoltTol)))
			    if ( (fabs(delvbd) < (ckt->CKTreltol * MAX(fabs(vbd),
								       fabs(*(ckt->CKTstate0+here->MOS3vbd)))+
						  ckt->CKTvoltTol)) )
				if( (fabs(delvgs) < (ckt->CKTreltol * MAX(fabs(vgs),
									  fabs(*(ckt->CKTstate0+here->MOS3vgs)))+
						     ckt->CKTvoltTol)))
				    if ( (fabs(delvds) < (ckt->CKTreltol * MAX(fabs(vds),
									       fabs(*(ckt->CKTstate0+here->MOS3vds)))+
							  ckt->CKTvoltTol)) )
					if( (fabs(cdhat- here->MOS3cd) <
					     ckt->CKTreltol * MAX(fabs(cdhat),fabs(
						 here->MOS3cd)) + ckt->CKTabstol) ) {
					    /* bypass code */
					    /* nothing interesting has changed since last
					     * iteration on this device, so we just
					     * copy all the values computed last iteration out
					     * and keep going
					     */
					    vbs = *(ckt->CKTstate0 + here->MOS3vbs);
					    vbd = *(ckt->CKTstate0 + here->MOS3vbd);
					    vgs = *(ckt->CKTstate0 + here->MOS3vgs);
					    vds = *(ckt->CKTstate0 + here->MOS3vds);
					    vgd = vgs - vds;
					    vgb = vgs - vbs;
					    cdrain = here->MOS3mode * (here->MOS3cd + here->MOS3cbd);
					    if(ckt->CKTmode & (MODETRAN | MODETRANOP)) {
						capgs = ( *(ckt->CKTstate0+here->MOS3capgs)+ 
							  *(ckt->CKTstate1+here->MOS3capgs) +
							  GateSourceOverlapCap );
						capgd = ( *(ckt->CKTstate0+here->MOS3capgd)+ 
							  *(ckt->CKTstate1+here->MOS3capgd) +
							  GateDrainOverlapCap );
						capgb = ( *(ckt->CKTstate0+here->MOS3capgb)+ 
							  *(ckt->CKTstate1+here->MOS3capgb) +
							  GateBulkOverlapCap );
					    }
					    goto bypass;
					}
#endif /*NOBYPASS*/
                /* ok - bypass is out, do it the hard way */

                von = model->MOS3type * here->MOS3von;

#ifndef NODELIMITING
                /* 
                 * limiting
                 *  we want to keep device voltages from changing
                 * so fast that the exponentials churn out overflows
                 * and similar rudeness
                 */

                if(*(ckt->CKTstate0 + here->MOS3vds) >=0) {
                    vgs = DEVfetlim(vgs,*(ckt->CKTstate0 + here->MOS3vgs)
				    ,von);
                    vds = vgs - vgd;
                    vds = DEVlimvds(vds,*(ckt->CKTstate0 + here->MOS3vds));
                    vgd = vgs - vds;
                } else {
                    vgd = DEVfetlim(vgd,vgdo,von);
                    vds = vgs - vgd;
                    if(!(ckt->CKTfixLimit)) {
                        vds = -DEVlimvds(-vds,-(*(ckt->CKTstate0 + 
						  here->MOS3vds)));
                    }
                    vgs = vgd + vds;
                }
                if(vds >= 0) {
                    vbs = DEVpnjlim(vbs,*(ckt->CKTstate0 + here->MOS3vbs),
				    vt,here->MOS3sourceVcrit,&Check);
                    vbd = vbs-vds;
                } else {
                    vbd = DEVpnjlim(vbd,*(ckt->CKTstate0 + here->MOS3vbd),
				    vt,here->MOS3drainVcrit,&Check);
                    vbs = vbd + vds;
                }
#endif /*NODELIMITING*/

            } else {

                /* ok - not one of the simple cases, so we have to
                 * look at all of the possibilities for why we were
                 * called.  We still just initialize the three voltages
                 */

                if((ckt->CKTmode & MODEINITJCT) && !here->MOS3off) {
                    vds= model->MOS3type * here->MOS3icVDS;
                    vgs= model->MOS3type * here->MOS3icVGS;
                    vbs= model->MOS3type * here->MOS3icVBS;
                    if((vds==0) && (vgs==0) && (vbs==0) && 
		       ((ckt->CKTmode & 
			 (MODETRAN|MODEDCOP|MODEDCTRANCURVE)) ||
			(!(ckt->CKTmode & MODEUIC)))) {
                        vbs = -1;
                        vgs = model->MOS3type * here->MOS3tVto;
                        vds = 0;
                    }
                } else {
                    vbs=vgs=vds=0;
                } 
            }

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
                arg=3*vt/(vbs*CONSTe);
                arg = arg * arg * arg;
                here->MOS3cbs = -SourceSatCur*(1+arg)+ckt->CKTgmin*vbs;
                here->MOS3gbs = SourceSatCur*3*arg/vbs+ckt->CKTgmin;
            } else {
                evbs = exp(MIN(MAX_EXP_ARG,vbs/vt));
                here->MOS3gbs = SourceSatCur*evbs/vt + ckt->CKTgmin;
                here->MOS3cbs = SourceSatCur*(evbs-1) + ckt->CKTgmin*vbs;
            }
            if(vbd <= -3*vt) {
                arg=3*vt/(vbd*CONSTe);
                arg = arg * arg * arg;
                here->MOS3cbd = -DrainSatCur*(1+arg)+ckt->CKTgmin*vbd;
                here->MOS3gbd = DrainSatCur*3*arg/vbd+ckt->CKTgmin;
            } else {
                evbd = exp(MIN(MAX_EXP_ARG,vbd/vt));
                here->MOS3gbd = DrainSatCur*evbd/vt + ckt->CKTgmin;
                here->MOS3cbd = DrainSatCur*(evbd-1) + ckt->CKTgmin*vbd;
            }

            /* now to determine whether the user was able to correctly
             * identify the source and drain of his device
             */
            if(vds >= 0) {
                /* normal mode */
                here->MOS3mode = 1;
            } else {
                /* inverse mode */
                here->MOS3mode = -1;
            }

            {
		/*
		 * subroutine moseq3(vds,vbs,vgs,gm,gds,gmbs,
		 * qg,qc,qb,cggb,cgdb,cgsb,cbgb,cbdb,cbsb)
		 */

		/* this routine evaluates the drain current, its
		 * derivatives and the charges associated with the
		 * gate, channel and bulk for mosfets based on
		 * semi-empirical equations */

		double coeff0 = 0.0631353e0;
		double coeff1 = 0.8013292e0;
		double coeff2 = -0.01110777e0;
		double oneoverxl;   /* 1/effective length */
		double eta; /* eta from model after length factor */
		double phibs;   /* phi - vbs */
		double sqphbs;  /* square root of phibs */
		double dsqdvb;  /*  */
		double sqphis;  /* square root of phi */
		double sqphs3;  /* square root of phi cubed */
		double wps;
		double oneoverxj;   /* 1/junction depth */
		double xjonxl;  /* junction depth/effective length */
		double djonxj;
		double wponxj;
		double arga;
		double argb;
		double argc;
		double dwpdvb;
		double dadvb;
		double dbdvb;
		double gammas;
		double fbodys;
		double fbody;
		double onfbdy;
		double qbonco;
		double vbix;
		double wconxj;
		double dfsdvb;
		double dfbdvb;
		double dqbdvb;
		double vth;
		double dvtdvb;
		double csonco;
		double cdonco;
		double dxndvb = 0.0;
		double dvodvb = 0.0;
		double dvodvd = 0.0;
		double vgsx;
		double dvtdvd;
		double onfg;
		double fgate;
		double us;
		double dfgdvg;
		double dfgdvd;
		double dfgdvb;
		double dvsdvg;
		double dvsdvb;
		double dvsdvd;
		double xn = 0.0;
		double vdsc;
		double onvdsc = 0.0;
		double dvsdga;
		double vdsx;
		double dcodvb;
		double cdnorm;
		double cdo;
		double cd1;
		double fdrain = 0.0;
		double fd2;
		double dfddvg = 0.0;
		double dfddvb = 0.0;
		double dfddvd = 0.0;
		double gdsat;
		double cdsat;
		double gdoncd;
		double gdonfd;
		double gdonfg;
		double dgdvg;
		double dgdvd;
		double dgdvb;
		double emax;
		double emongd;
		double demdvg;
		double demdvd;
		double demdvb;
		double delxl;
		double dldvd;
		double dldem;
		double ddldvg;
		double ddldvd;
		double ddldvb;
		double dlonxl;
		double xlfact;
		double diddl;
		double gds0 = 0.0;
		double emoncd;
		double ondvt;
		double onxn;
		double wfact;
		double gms;
		double gmw;
		double fshort;
		    
		/*
		 *     bypasses the computation of charges
		 */
		    
		/*
		 *     reference cdrain equations to source and
		 *     charge equations to bulk
		 */
		vdsat = 0.0;
		oneoverxl = 1.0/EffectiveLength;
		eta = model->MOS3eta * 8.15e-22/(model->MOS3oxideCapFactor*
						 EffectiveLength*EffectiveLength*EffectiveLength);
		/*
		 *.....square root term
		 */
		if ( (here->MOS3mode==1?vbs:vbd) <=  0.0 ) {
		    phibs  =  here->MOS3tPhi-(here->MOS3mode==1?vbs:vbd);
		    sqphbs  =  sqrt(phibs);
		    dsqdvb  =  -0.5/sqphbs;
		} else {
		    sqphis = sqrt(here->MOS3tPhi);
		    sqphs3 = here->MOS3tPhi*sqphis;
		    sqphbs = sqphis/(1.0+(here->MOS3mode==1?vbs:vbd)/
				     (here->MOS3tPhi+here->MOS3tPhi));
		    phibs = sqphbs*sqphbs;
		    dsqdvb = -phibs/(sqphs3+sqphs3);
		}
		/*
		 *.....short channel effect factor
		 */
		if ( (model->MOS3junctionDepth != 0.0) && 
		     (model->MOS3coeffDepLayWidth != 0.0) ) {
		    wps = model->MOS3coeffDepLayWidth*sqphbs;
		    oneoverxj = 1.0/model->MOS3junctionDepth;
		    xjonxl = model->MOS3junctionDepth*oneoverxl;
		    djonxj = model->MOS3latDiff*oneoverxj;
		    wponxj = wps*oneoverxj;
		    wconxj = coeff0+coeff1*wponxj+coeff2*wponxj*wponxj;
		    arga = wconxj+djonxj;
		    argc = wponxj/(1.0+wponxj);
		    argb = sqrt(1.0-argc*argc);
		    fshort = 1.0-xjonxl*(arga*argb-djonxj);
		    dwpdvb = model->MOS3coeffDepLayWidth*dsqdvb;
		    dadvb = (coeff1+coeff2*(wponxj+wponxj))*dwpdvb*oneoverxj;
		    dbdvb = -argc*argc*(1.0-argc)*dwpdvb/(argb*wps);
		    dfsdvb = -xjonxl*(dadvb*argb+arga*dbdvb);
		} else {
		    fshort = 1.0;
		    dfsdvb = 0.0;
		}
		/*
		 *.....body effect
		 */
		gammas = model->MOS3gamma*fshort;
		fbodys = 0.5*gammas/(sqphbs+sqphbs);
		fbody = fbodys+model->MOS3narrowFactor/EffectiveWidth;
		onfbdy = 1.0/(1.0+fbody);
		dfbdvb = -fbodys*dsqdvb/sqphbs+fbodys*dfsdvb/fshort;
		qbonco =gammas*sqphbs+model->MOS3narrowFactor*phibs/EffectiveWidth;
		dqbdvb = gammas*dsqdvb+model->MOS3gamma*dfsdvb*sqphbs-
		    model->MOS3narrowFactor/EffectiveWidth;
		/*
		 *.....static feedback effect
		 */
		vbix = here->MOS3tVbi*model->MOS3type-eta*(here->MOS3mode*vds);
		/*
		 *.....threshold voltage
		 */
		vth = vbix+qbonco;
		dvtdvd = -eta;
		dvtdvb = dqbdvb;
		/*
		 *.....joint weak inversion and strong inversion
		 */
		von = vth;
		if ( model->MOS3fastSurfaceStateDensity != 0.0 ) {
		    csonco = CHARGE*model->MOS3fastSurfaceStateDensity * 
			1e4 /*(cm**2/m**2)*/ *EffectiveLength*EffectiveWidth *
			here->MOS3m/OxideCap;
		    cdonco = qbonco/(phibs+phibs);
		    xn = 1.0+csonco+cdonco;
		    von = vth+vt*xn;
		    dxndvb = dqbdvb/(phibs+phibs)-qbonco*dsqdvb/(phibs*sqphbs);
		    dvodvd = dvtdvd;
		    dvodvb = dvtdvb+vt*dxndvb;
		} else {
		    /*
		     *.....cutoff region
		     */
		    if ( (here->MOS3mode==1?vgs:vgd) <= von ) {
			cdrain = 0.0;
			here->MOS3gm = 0.0;
			here->MOS3gds = 0.0;
			here->MOS3gmbs = 0.0;
			goto innerline1000;
		    }
		}
		/*
		 *.....device is on
		 */
		vgsx = MAX((here->MOS3mode==1?vgs:vgd),von);
		/*
		 *.....mobility modulation by gate voltage
		 */
		onfg = 1.0+model->MOS3theta*(vgsx-vth);
		fgate = 1.0/onfg;
		us = here->MOS3tSurfMob * 1e-4 /*(m**2/cm**2)*/ *fgate;
		dfgdvg = -model->MOS3theta*fgate*fgate;
		dfgdvd = -dfgdvg*dvtdvd;
		dfgdvb = -dfgdvg*dvtdvb;
		/*
		 *.....saturation voltage
		 */
		vdsat = (vgsx-vth)*onfbdy;
		if ( model->MOS3maxDriftVel <= 0.0 ) {
		    dvsdvg = onfbdy;
		    dvsdvd = -dvsdvg*dvtdvd;
		    dvsdvb = -dvsdvg*dvtdvb-vdsat*dfbdvb*onfbdy;
		} else {
		    vdsc = EffectiveLength*model->MOS3maxDriftVel/us;
		    onvdsc = 1.0/vdsc;
		    arga = (vgsx-vth)*onfbdy;
		    argb = sqrt(arga*arga+vdsc*vdsc);
		    vdsat = arga+vdsc-argb;
		    dvsdga = (1.0-arga/argb)*onfbdy;
		    dvsdvg = dvsdga-(1.0-vdsc/argb)*vdsc*dfgdvg*onfg;
		    dvsdvd = -dvsdvg*dvtdvd;
		    dvsdvb = -dvsdvg*dvtdvb-arga*dvsdga*dfbdvb;
		}
		/*
		 *.....current factors in linear region
		 */
		vdsx = MIN((here->MOS3mode*vds),vdsat);
		if ( vdsx == 0.0 ) goto line900;
		cdo = vgsx-vth-0.5*(1.0+fbody)*vdsx;
		dcodvb = -dvtdvb-0.5*dfbdvb*vdsx;
		/* 
		 *.....normalized drain current
		 */
		cdnorm = cdo*vdsx;
		here->MOS3gm = vdsx;
		if ((here->MOS3mode*vds) > vdsat) here->MOS3gds = -dvtdvd*vdsx;
		else here->MOS3gds = vgsx-vth-(1.0+fbody+dvtdvd)*vdsx;
		here->MOS3gmbs = dcodvb*vdsx;
		/* 
		 *.....drain current without velocity saturation effect
		 */
		cd1 = Beta*cdnorm;
		Beta = Beta*fgate;
		cdrain = Beta*cdnorm;
		here->MOS3gm = Beta*here->MOS3gm+dfgdvg*cd1;
		here->MOS3gds = Beta*here->MOS3gds+dfgdvd*cd1;
		here->MOS3gmbs = Beta*here->MOS3gmbs+dfgdvb*cd1;
		/*
		 *.....velocity saturation factor
		 */
		if ( model->MOS3maxDriftVel > 0.0 ) {
		    fdrain = 1.0/(1.0+vdsx*onvdsc);
		    fd2 = fdrain*fdrain;
		    arga = fd2*vdsx*onvdsc*onfg;
		    dfddvg = -dfgdvg*arga;
		    if ((here->MOS3mode*vds) > vdsat) dfddvd = -dfgdvd*arga;
		    else dfddvd = -dfgdvd*arga-fd2*onvdsc;
		    dfddvb = -dfgdvb*arga;
		    /*
		     *.....drain current
		     */
		    here->MOS3gm = fdrain*here->MOS3gm+dfddvg*cdrain;
		    here->MOS3gds = fdrain*here->MOS3gds+dfddvd*cdrain;
		    here->MOS3gmbs = fdrain*here->MOS3gmbs+dfddvb*cdrain;
		    cdrain = fdrain*cdrain;
		    Beta = Beta*fdrain;
		}
		/*
		 *.....channel length modulation
		 */
		    
		if ( (here->MOS3mode*vds) <= vdsat ) {
		    if ( (model->MOS3maxDriftVel > 0.0) ||
			 (model->MOS3alpha == 0.0) ||
			 (ckt->CKTbadMos3)                    ) goto line700;
		    else {
			arga = (here->MOS3mode*vds)/vdsat;
			delxl = sqrt(model->MOS3kappa*model->MOS3alpha*vdsat/8);
			dldvd = 4*delxl*arga*arga*arga/vdsat;
			arga *= arga;
			arga *= arga;
			delxl *= arga;
			ddldvg = 0.0;
			ddldvd = -dldvd;
			ddldvb = 0.0;
			goto line520;
		    }
		}
			    
		if ( model->MOS3maxDriftVel <= 0.0 ) goto line510;
		if (model->MOS3alpha == 0.0) goto line700;
		cdsat = cdrain;
		gdsat = cdsat*(1.0-fdrain)*onvdsc;
		gdsat = MAX(1.0e-12,gdsat);
		gdoncd = gdsat/cdsat;
		gdonfd = gdsat/(1.0-fdrain);
		gdonfg = gdsat*onfg;
		dgdvg = gdoncd*here->MOS3gm-gdonfd*dfddvg+gdonfg*dfgdvg;
		dgdvd = gdoncd*here->MOS3gds-gdonfd*dfddvd+gdonfg*dfgdvd;
		dgdvb = gdoncd*here->MOS3gmbs-gdonfd*dfddvb+gdonfg*dfgdvb;
			    
		if (ckt->CKTbadMos3)
			emax = cdsat*oneoverxl/gdsat;
		else
			emax = model->MOS3kappa * cdsat*oneoverxl/gdsat;
		emoncd = emax/cdsat;
		emongd = emax/gdsat;
		demdvg = emoncd*here->MOS3gm-emongd*dgdvg;
		demdvd = emoncd*here->MOS3gds-emongd*dgdvd;
		demdvb = emoncd*here->MOS3gmbs-emongd*dgdvb;
			    
		arga = 0.5*emax*model->MOS3alpha;
		argc = model->MOS3kappa*model->MOS3alpha;
		argb = sqrt(arga*arga+argc*((here->MOS3mode*vds)-vdsat));
		delxl = argb-arga;
		if (argb != 0.0) {
			dldvd = argc/(argb+argb);
			dldem = 0.5*(arga/argb-1.0)*model->MOS3alpha;
		} else {
			dldvd = 0.0;
			dldem = 0.0;
		}
		ddldvg = dldem*demdvg;
		ddldvd = dldem*demdvd-dldvd;
		ddldvb = dldem*demdvb;
		goto line520;
line510:
		if (ckt->CKTbadMos3) {
		  delxl = sqrt(model->MOS3kappa*((here->MOS3mode*vds)-vdsat)*
				     model->MOS3alpha);
		  dldvd = 0.5*delxl/((here->MOS3mode*vds)-vdsat);
	       	} else {
		  delxl = sqrt(model->MOS3kappa*model->MOS3alpha*
				     ((here->MOS3mode*vds)-vdsat+(vdsat/8)));
		  dldvd =  0.5*delxl/((here->MOS3mode*vds)-vdsat+(vdsat/8));
	        }
		ddldvg = 0.0;
		ddldvd = -dldvd;
		ddldvb = 0.0;
		/*
		 *.....punch through approximation
		 */
line520:
		if ( delxl > (0.5*EffectiveLength) ) {
		delxl = EffectiveLength-(EffectiveLength*EffectiveLength/
						 (4.0*delxl));
		arga = 4.0*(EffectiveLength-delxl)*(EffectiveLength-delxl)/
			    (EffectiveLength*EffectiveLength);
		ddldvg = ddldvg*arga;
		ddldvd = ddldvd*arga;
		ddldvb = ddldvb*arga;
		dldvd =  dldvd*arga;
		}
		/*
		 *.....saturation region
		 */
		dlonxl = delxl*oneoverxl;
		xlfact = 1.0/(1.0-dlonxl);

		cd1 = cdrain;
		cdrain = cdrain*xlfact;
		diddl = cdrain/(EffectiveLength-delxl);
		here->MOS3gm = here->MOS3gm*xlfact+diddl*ddldvg;
		here->MOS3gmbs = here->MOS3gmbs*xlfact+diddl*ddldvb;
		gds0 = diddl*ddldvd;
		here->MOS3gm = here->MOS3gm+gds0*dvsdvg;
		here->MOS3gmbs = here->MOS3gmbs+gds0*dvsdvb;
		here->MOS3gds = here->MOS3gds*xlfact+diddl*dldvd+gds0*dvsdvd;
/*              here->MOS3gds = (here->MOS3gds*xlfact)+gds0*dvsdvd-
	           (cd1*ddldvd/(EffectiveLength*(1-2*dlonxl+dlonxl*dlonxl)));*/
			    
		/*
		 *.....finish strong inversion case
		 */
line700:
		if ( (here->MOS3mode==1?vgs:vgd) < von ) {
		/*
		 *.....weak inversion
		 */
		onxn = 1.0/xn;
		ondvt = onxn/vt;
		wfact = exp( ((here->MOS3mode==1?vgs:vgd)-von)*ondvt );
		cdrain = cdrain*wfact;
		gms = here->MOS3gm*wfact;
		gmw = cdrain*ondvt;
		here->MOS3gm = gmw;
		if ((here->MOS3mode*vds) > vdsat) {
		    here->MOS3gm = here->MOS3gm+gds0*dvsdvg*wfact;
		}
		here->MOS3gds = here->MOS3gds*wfact+(gms-gmw)*dvodvd;
		here->MOS3gmbs = here->MOS3gmbs*wfact+(gms-gmw)*dvodvb-gmw*
			    ((here->MOS3mode==1?vgs:vgd)-von)*onxn*dxndvb;
		}
		/*
		 *.....charge computation
		 */
		goto innerline1000;
		/*
		 *.....special case of vds = 0.0d0
		 */
line900:
		Beta = Beta*fgate;
		cdrain = 0.0;
		here->MOS3gm = 0.0;
		here->MOS3gds = Beta*(vgsx-vth);
		here->MOS3gmbs = 0.0;
		if ( (model->MOS3fastSurfaceStateDensity != 0.0) && 
		           ((here->MOS3mode==1?vgs:vgd) < von) ) {
		   here->MOS3gds *=exp(((here->MOS3mode==1?vgs:vgd)-von)/(vt*xn));
		}
innerline1000:;
		/* 
		 *.....done
		 */
	    }
		    
		    
		/* now deal with n vs p polarity */
		    
		here->MOS3von = model->MOS3type * von;
		here->MOS3vdsat = model->MOS3type * vdsat;
		/* line 490 */
		/*
		 *  COMPUTE EQUIVALENT DRAIN CURRENT SOURCE
		 */
		here->MOS3cd=here->MOS3mode * cdrain - here->MOS3cbd;
		    
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
                        fabs(*(ckt->CKTstate0+here->MOS3vbs)))+
                        ckt->CKTvoltTol)|| senflag )
#endif /*CAPBYPASS*/
		     		     
		    {
			/* can't bypass the diode capacitance calculations */
			if(here->MOS3Cbs != 0 || here->MOS3Cbssw != 0 ) {
			    if (vbs < here->MOS3tDepCap){
				arg=1-vbs/here->MOS3tBulkPot;
				/*
				 * the following block looks somewhat long and messy,
				 * but since most users use the default grading
				 * coefficients of .5, and sqrt is MUCH faster than an
				 * exp(log()) we use this special case code to buy time.
				 * (as much as 10% of total job time!)
				 */
				if(model->MOS3bulkJctBotGradingCoeff ==
				   model->MOS3bulkJctSideGradingCoeff) {
				    if(model->MOS3bulkJctBotGradingCoeff == .5) {
					sarg = sargsw = 1/sqrt(arg);
				    } else {
					sarg = sargsw =
					    exp(-model->MOS3bulkJctBotGradingCoeff*
						log(arg));
				    }
				} else {
				    if(model->MOS3bulkJctBotGradingCoeff == .5) {
					sarg = 1/sqrt(arg);
				    } else {
					sarg = exp(-model->MOS3bulkJctBotGradingCoeff*
						   log(arg));
				    }
				    if(model->MOS3bulkJctSideGradingCoeff == .5) {
					sargsw = 1/sqrt(arg);
				    } else {
					sargsw =exp(-model->MOS3bulkJctSideGradingCoeff*
						    log(arg));
				    }
				}
				*(ckt->CKTstate0 + here->MOS3qbs) =
				    here->MOS3tBulkPot*(here->MOS3Cbs*
							(1-arg*sarg)/(1-model->MOS3bulkJctBotGradingCoeff)
							+here->MOS3Cbssw*
							(1-arg*sargsw)/
							(1-model->MOS3bulkJctSideGradingCoeff));
				here->MOS3capbs=here->MOS3Cbs*sarg+
				    here->MOS3Cbssw*sargsw;
			    } else {
				*(ckt->CKTstate0 + here->MOS3qbs) = here->MOS3f4s +
				    vbs*(here->MOS3f2s+vbs*(here->MOS3f3s/2));
				here->MOS3capbs=here->MOS3f2s+here->MOS3f3s*vbs;
			    }
			} else {
			    *(ckt->CKTstate0 + here->MOS3qbs) = 0;
			    here->MOS3capbs=0;
			}
		    }
#ifdef CAPBYPASS
                if(((ckt->CKTmode & (MODEINITPRED | MODEINITTRAN) ) ||
                        fabs(delvbd) >= ckt->CKTreltol * MAX(fabs(vbd),
                        fabs(*(ckt->CKTstate0+here->MOS3vbd)))+
                        ckt->CKTvoltTol)|| senflag )
#endif /*CAPBYPASS*/		    		    
		    /* can't bypass the diode capacitance calculations */
		    {
			if(here->MOS3Cbd != 0 || here->MOS3Cbdsw != 0 ) {
			    if (vbd < here->MOS3tDepCap) {
				arg=1-vbd/here->MOS3tBulkPot;
				/*
				 * the following block looks somewhat long and messy,
				 * but since most users use the default grading
				 * coefficients of .5, and sqrt is MUCH faster than an
				 * exp(log()) we use this special case code to buy time.
				 * (as much as 10% of total job time!)
				 */
				if(model->MOS3bulkJctBotGradingCoeff == .5 &&
				   model->MOS3bulkJctSideGradingCoeff == .5) {
				    sarg = sargsw = 1/sqrt(arg);
				} else {
				    if(model->MOS3bulkJctBotGradingCoeff == .5) {
					sarg = 1/sqrt(arg);
				    } else {
					sarg = exp(-model->MOS3bulkJctBotGradingCoeff*
						   log(arg));
				    }
				    if(model->MOS3bulkJctSideGradingCoeff == .5) {
					sargsw = 1/sqrt(arg);
				    } else {
					sargsw =exp(-model->MOS3bulkJctSideGradingCoeff*
						    log(arg));
				    }
				}
				*(ckt->CKTstate0 + here->MOS3qbd) =
				    here->MOS3tBulkPot*(here->MOS3Cbd*
							(1-arg*sarg)
							/(1-model->MOS3bulkJctBotGradingCoeff)
							+here->MOS3Cbdsw*
							(1-arg*sargsw)
							/(1-model->MOS3bulkJctSideGradingCoeff));
				here->MOS3capbd=here->MOS3Cbd*sarg+
				    here->MOS3Cbdsw*sargsw;
			    } else {
				*(ckt->CKTstate0 + here->MOS3qbd) = here->MOS3f4d +
				    vbd * (here->MOS3f2d + vbd * here->MOS3f3d/2);
				here->MOS3capbd=here->MOS3f2d + vbd * here->MOS3f3d;
			    }
			} else {
			    *(ckt->CKTstate0 + here->MOS3qbd) = 0;
			    here->MOS3capbd = 0;
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
				    
			error = NIintegrate(ckt,&geq,&ceq,here->MOS3capbd,
					    here->MOS3qbd);
			if(error) return(error);
			here->MOS3gbd += geq;
			here->MOS3cbd += *(ckt->CKTstate0 + here->MOS3cqbd);
			here->MOS3cd -= *(ckt->CKTstate0 + here->MOS3cqbd);
			error = NIintegrate(ckt,&geq,&ceq,here->MOS3capbs,
					    here->MOS3qbs);
			if(error) return(error);
			here->MOS3gbs += geq;
			here->MOS3cbs += *(ckt->CKTstate0 + here->MOS3cqbs);
		    }
		}
		if(SenCond) goto next2;
		    
		/*
		 *  check convergence
		 */
		if ( (here->MOS3off == 0)  || 
		     (!(ckt->CKTmode & (MODEINITFIX|MODEINITSMSIG))) ){
		    if (Check == 1) {
			ckt->CKTnoncon++;
			ckt->CKTtroubleElt = (GENinstance *) here;
		    }
		}
		    
		    
		/* save things away for next time */
		    
	    next2:      *(ckt->CKTstate0 + here->MOS3vbs) = vbs;
		*(ckt->CKTstate0 + here->MOS3vbd) = vbd;
		*(ckt->CKTstate0 + here->MOS3vgs) = vgs;
		*(ckt->CKTstate0 + here->MOS3vds) = vds;
		    
		    
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
		    if (here->MOS3mode > 0){
			DEVqmeyer (vgs,vgd,vgb,von,vdsat,
				   (ckt->CKTstate0 + here->MOS3capgs),
				   (ckt->CKTstate0 + here->MOS3capgd),
				   (ckt->CKTstate0 + here->MOS3capgb),
				   here->MOS3tPhi,OxideCap);
		    } else {
			DEVqmeyer (vgd,vgs,vgb,von,vdsat,
				   (ckt->CKTstate0 + here->MOS3capgd),
				   (ckt->CKTstate0 + here->MOS3capgs),
				   (ckt->CKTstate0 + here->MOS3capgb),
				   here->MOS3tPhi,OxideCap);
		    }
		    vgs1 = *(ckt->CKTstate1 + here->MOS3vgs);
		    vgd1 = vgs1 - *(ckt->CKTstate1 + here->MOS3vds);
		    vgb1 = vgs1 - *(ckt->CKTstate1 + here->MOS3vbs);
		    if(ckt->CKTmode & MODETRANOP) {
			capgs =  2 * *(ckt->CKTstate0+here->MOS3capgs)+ 
			    GateSourceOverlapCap ;
			capgd =  2 * *(ckt->CKTstate0+here->MOS3capgd)+ 
			    GateDrainOverlapCap ;
			capgb =  2 * *(ckt->CKTstate0+here->MOS3capgb)+ 
			    GateBulkOverlapCap ;
		    } else {
			capgs = ( *(ckt->CKTstate0+here->MOS3capgs)+ 
				  *(ckt->CKTstate1+here->MOS3capgs) +
				  GateSourceOverlapCap );
			capgd = ( *(ckt->CKTstate0+here->MOS3capgd)+ 
				  *(ckt->CKTstate1+here->MOS3capgd) +
				  GateDrainOverlapCap );
			capgb = ( *(ckt->CKTstate0+here->MOS3capgb)+ 
				  *(ckt->CKTstate1+here->MOS3capgb) +
				  GateBulkOverlapCap );
		    }
		    if(ckt->CKTsenInfo){
			here->MOS3cgs = capgs;
			here->MOS3cgd = capgd;
			here->MOS3cgb = capgb;
		    }
			    
		    /*
		     *     store small-signal parameters (for meyer's model)
		     *  all parameters already stored, so done...
		     */
			    
			    
		    if(SenCond){
			if(ckt->CKTsenInfo->SENmode & (DCSEN|ACSEN)) {
			    continue;
			}
		    }
#ifndef PREDICTOR
		    if (ckt->CKTmode & (MODEINITPRED | MODEINITTRAN) ) {
			*(ckt->CKTstate0 + here->MOS3qgs) =
			    (1+xfact) * *(ckt->CKTstate1 + here->MOS3qgs)
			    - xfact * *(ckt->CKTstate2 + here->MOS3qgs);
			*(ckt->CKTstate0 + here->MOS3qgd) =
			    (1+xfact) * *(ckt->CKTstate1 + here->MOS3qgd)
			    - xfact * *(ckt->CKTstate2 + here->MOS3qgd);
			*(ckt->CKTstate0 + here->MOS3qgb) =
			    (1+xfact) * *(ckt->CKTstate1 + here->MOS3qgb)
			    - xfact * *(ckt->CKTstate2 + here->MOS3qgb);
		    } else {
#endif /*PREDICTOR*/
			if(ckt->CKTmode & MODETRAN) {
			    *(ckt->CKTstate0 + here->MOS3qgs) = (vgs-vgs1)*capgs +
				*(ckt->CKTstate1 + here->MOS3qgs) ;
			    *(ckt->CKTstate0 + here->MOS3qgd) = (vgd-vgd1)*capgd +
				*(ckt->CKTstate1 + here->MOS3qgd) ;
			    *(ckt->CKTstate0 + here->MOS3qgb) = (vgb-vgb1)*capgb +
				*(ckt->CKTstate1 + here->MOS3qgb) ;
			} else {
			    /* TRANOP only */
			    *(ckt->CKTstate0 + here->MOS3qgs) = vgs*capgs;
			    *(ckt->CKTstate0 + here->MOS3qgd) = vgd*capgd;
			    *(ckt->CKTstate0 + here->MOS3qgb) = vgb*capgb;
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
		    if(capgs == 0) *(ckt->CKTstate0 + here->MOS3cqgs) =0;
		    if(capgd == 0) *(ckt->CKTstate0 + here->MOS3cqgd) =0;
		    if(capgb == 0) *(ckt->CKTstate0 + here->MOS3cqgb) =0;
		    /*
		     *    calculate equivalent conductances and currents for
		     *    meyer"s capacitors
		     */
		    error = NIintegrate(ckt,&gcgs,&ceqgs,capgs,here->MOS3qgs);
		    if(error) return(error);
		    error = NIintegrate(ckt,&gcgd,&ceqgd,capgd,here->MOS3qgd);
		    if(error) return(error);
		    error = NIintegrate(ckt,&gcgb,&ceqgb,capgb,here->MOS3qgb);
		    if(error) return(error);
		    ceqgs=ceqgs-gcgs*vgs+ckt->CKTag[0]* 
			*(ckt->CKTstate0 + here->MOS3qgs);
		    ceqgd=ceqgd-gcgd*vgd+ckt->CKTag[0]*
			*(ckt->CKTstate0 + here->MOS3qgd);
		    ceqgb=ceqgb-gcgb*vgb+ckt->CKTag[0]*
			*(ckt->CKTstate0 + here->MOS3qgb);
		}
		/*
		 *     store charge storage info for meyer's cap in lx table
		 */
		    
		/*
		 *  load current vector
		 */
		ceqbs = model->MOS3type * 
		    (here->MOS3cbs-(here->MOS3gbs)*vbs);
		ceqbd = model->MOS3type * 
		    (here->MOS3cbd-(here->MOS3gbd)*vbd);
		if (here->MOS3mode >= 0) {
		    xnrm=1;
		    xrev=0;
		    cdreq=model->MOS3type*(cdrain-here->MOS3gds*vds-
					   here->MOS3gm*vgs-here->MOS3gmbs*vbs);
		} else {
		    xnrm=0;
		    xrev=1;
		    cdreq = -(model->MOS3type)*(cdrain-here->MOS3gds*(-vds)-
						here->MOS3gm*vgd-here->MOS3gmbs*vbd);
		}
		*(ckt->CKTrhs + here->MOS3gNode) -= 
		    (model->MOS3type * (ceqgs + ceqgb + ceqgd));
		*(ckt->CKTrhs + here->MOS3bNode) -=
		    (ceqbs + ceqbd - model->MOS3type * ceqgb);
		*(ckt->CKTrhs + here->MOS3dNodePrime) +=
		    (ceqbd - cdreq + model->MOS3type * ceqgd);
		*(ckt->CKTrhs + here->MOS3sNodePrime) += 
		    cdreq + ceqbs + model->MOS3type * ceqgs;
		/*
            *  load y matrix
            */
#if 0
printf(" loading %s at time %g\n",here->MOS3name,ckt->CKTtime);
printf("%g %g %g %g %g\n", here->MOS3drainConductance,gcgd+gcgs+gcgb,
        here->MOS3sourceConductance,here->MOS3gbd,here->MOS3gbs);
printf("%g %g %g %g %g\n",-gcgb,0.0,0.0,here->MOS3gds,here->MOS3gm);
printf("%g %g %g %g %g\n", here->MOS3gds,here->MOS3gmbs,gcgd,-gcgs,-gcgd);
printf("%g %g %g %g %g\n", -gcgs,-gcgd,0.0,-gcgs,0.0);
#endif		 	    	    
		*(here->MOS3DdPtr) += (here->MOS3drainConductance);
		*(here->MOS3GgPtr) += ((gcgd+gcgs+gcgb));
		*(here->MOS3SsPtr) += (here->MOS3sourceConductance);
		*(here->MOS3BbPtr) += (here->MOS3gbd+here->MOS3gbs+gcgb);
		*(here->MOS3DPdpPtr) += 
		    (here->MOS3drainConductance+here->MOS3gds+
		     here->MOS3gbd+xrev*(here->MOS3gm+here->MOS3gmbs)+gcgd);
		*(here->MOS3SPspPtr) += 
		    (here->MOS3sourceConductance+here->MOS3gds+
		     here->MOS3gbs+xnrm*(here->MOS3gm+here->MOS3gmbs)+gcgs);
		*(here->MOS3DdpPtr) += (-here->MOS3drainConductance);
		*(here->MOS3GbPtr) -= gcgb;
		*(here->MOS3GdpPtr) -= gcgd;
		*(here->MOS3GspPtr) -= gcgs;
		*(here->MOS3SspPtr) += (-here->MOS3sourceConductance);
		*(here->MOS3BgPtr) -= gcgb;
		*(here->MOS3BdpPtr) -= here->MOS3gbd;
		*(here->MOS3BspPtr) -= here->MOS3gbs;
		*(here->MOS3DPdPtr) += (-here->MOS3drainConductance);
		*(here->MOS3DPgPtr) += ((xnrm-xrev)*here->MOS3gm-gcgd);
		*(here->MOS3DPbPtr) += (-here->MOS3gbd+(xnrm-xrev)*here->MOS3gmbs);
		*(here->MOS3DPspPtr) += (-here->MOS3gds-
					 xnrm*(here->MOS3gm+here->MOS3gmbs));
		*(here->MOS3SPgPtr) += (-(xnrm-xrev)*here->MOS3gm-gcgs);
		*(here->MOS3SPsPtr) += (-here->MOS3sourceConductance);
		*(here->MOS3SPbPtr) += (-here->MOS3gbs-(xnrm-xrev)*here->MOS3gmbs);
		*(here->MOS3SPdpPtr) += (-here->MOS3gds-
					 xrev*(here->MOS3gm+here->MOS3gmbs));
	    }
    }
    return(OK);
}
    
