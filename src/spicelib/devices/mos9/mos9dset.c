/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jaijeet S Roychowdhury
Modified: Alan Gillespie
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "mos9defs.h"
#include "ngspice/trandefs.h"
#include "ngspice/distodef.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS9dSetup(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current value into the 
         * sparse matrix previously provided 
         */
{
    MOS9model *model = (MOS9model *)inModel;
    MOS9instance *here;
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
    double cdrain;
    double evbs;
    double sarg;
    double sargsw;
    double lvgs;
    double vbd;
    double vbs;
    double vds;
    double vdsat;
    double vgb;
    double vgd;
    double vgs;
    double von;
    double lcapgs2,lcapgs3;   /* total gate-source capacitance */
    double lcapgd2,lcapgd3;   /* total gate-drain capacitance */
    double lcapgb2,lcapgb3;   /* total gate-bulk capacitance */
    double lgbs, lgbs2, lgbs3;
    double lgbd, lgbd2, lgbd3;
    double gm2, gb2, gds2, gmb, gmds, gbds;
    double gm3, gb3, gds3, gm2ds, gm2b, gb2ds, gbds2, gmb2, gmds2, gmbds;
    double lcapbd, lcapbd2, lcapbd3;
    double lcapbs, lcapbs2, lcapbs3;
    double ebd;
    double vt;  /* vt at instance temperature */
    Dderivs d_cdrain;



    /*  loop through all the MOS9 device models */
    for( ; model != NULL; model = MOS9nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = MOS9instances(model); here != NULL ;
                here=MOS9nextInstance(here)) {

            vt = CONSTKoverQ * here->MOS9temp;


            /* first, we compute a few useful values - these could be
             * pre-computed, but for historical reasons are still done
             * here.  They may be moved at the expense of instance size
             */

            EffectiveWidth=here->MOS9w-2*model->MOS9widthNarrow+
                                                    model->MOS9widthAdjust;
            EffectiveLength=here->MOS9l - 2*model->MOS9latDiff+
                                                    model->MOS9lengthAdjust;

            if( (here->MOS9tSatCurDens == 0) || 
                    (here->MOS9drainArea == 0) ||
                    (here->MOS9sourceArea == 0)) {
                DrainSatCur = here->MOS9m * here->MOS9tSatCur;
                SourceSatCur = here->MOS9m * here->MOS9tSatCur;
            } else {
                DrainSatCur = here->MOS9tSatCurDens * 
                        here->MOS9m * here->MOS9drainArea;
                SourceSatCur = here->MOS9tSatCurDens * 
                        here->MOS9m * here->MOS9sourceArea;
            }
            GateSourceOverlapCap = model->MOS9gateSourceOverlapCapFactor * 
                    here->MOS9m * EffectiveWidth;
            GateDrainOverlapCap = model->MOS9gateDrainOverlapCapFactor * 
                    here->MOS9m * EffectiveWidth;
            GateBulkOverlapCap = model->MOS9gateBulkOverlapCapFactor * 
                    here->MOS9m * EffectiveLength;
            Beta = here->MOS9tTransconductance * here->MOS9m *
                    EffectiveWidth/EffectiveLength;
            OxideCap = model->MOS9oxideCapFactor * EffectiveLength * 
                    here->MOS9m * EffectiveWidth;


            /* 
             * ok - now to do the start-up operations
             *
             * we must get values for vbs, vds, and vgs from somewhere
             * so we either predict them or recover them from last iteration
             * These are the two most common cases - either a prediction
             * step or the general iteration step and they
             * share some code, so we put them first - others later on
             */


                    /* general iteration */

                    vbs = model->MOS9type * ( 
                        *(ckt->CKTrhsOld+here->MOS9bNode) -
                        *(ckt->CKTrhsOld+here->MOS9sNodePrime));
                    vgs = model->MOS9type * ( 
                        *(ckt->CKTrhsOld+here->MOS9gNode) -
                        *(ckt->CKTrhsOld+here->MOS9sNodePrime));
                    vds = model->MOS9type * ( 
                        *(ckt->CKTrhsOld+here->MOS9dNodePrime) -
                        *(ckt->CKTrhsOld+here->MOS9sNodePrime));

                /* now some common crunching for some more useful quantities */



            /*
             * now all the preliminaries are over - we can start doing the
             * real work
             */
            vbd = vbs - vds;
            vgd = vgs - vds;
            vgb = vgs - vbs;

            /* bulk-source and bulk-drain doides
             * here we just evaluate the ideal diode current and the
             * correspoinding derivative (conductance).
             */
/*next1:*/  if(vbs <= 0) {
                lgbs = SourceSatCur/vt;
                lgbs += ckt->CKTgmin;
		lgbs2 = lgbs3 = 0;
            } else {
                evbs = exp(MIN(MAX_EXP_ARG,vbs/vt));
                lgbs = SourceSatCur*evbs/vt + ckt->CKTgmin;
		lgbs2 = model->MOS9type *0.5 * (lgbs - ckt->CKTgmin)/vt;
		lgbs3 = model->MOS9type *lgbs2/(vt*3);

            }
            if(vbd <= 0) {
                lgbd = DrainSatCur/vt;
                lgbd += ckt->CKTgmin;
		lgbd2 = lgbd3 = 0;
            } else {
                ebd = exp(MIN(MAX_EXP_ARG,vbd/vt));
                lgbd = DrainSatCur*ebd/vt +ckt->CKTgmin;
		lgbd2 = model->MOS9type *0.5 * (lgbd - ckt->CKTgmin)/vt;
		lgbd3 = model->MOS9type *lgbd2/(vt*3);
            }


            /* now to determine whether the user was able to correctly
             * identify the source and drain of his device
             */
            if(vds >= 0) {
                /* normal mode */
                here->MOS9mode = 1;
            } else {
                /* inverse mode */
                here->MOS9mode = -1;
            }

            {
            /*
             * subroutine moseq3(vds,vbs,vgs,gm,gds,gmbs,
             * qg,qc,qb,cggb,cgdb,cgsb,cbgb,cbdb,cbsb)
             */

            /*
             *     this routine evaluates the drain current, its derivatives and
             *     the charges associated with the gate, channel and bulk
             *     for mosfets based on semi-empirical equations
             */

            /*
            common /mosarg/ vto,beta,gamma,phi,phib,cox,xnsub,xnfs,xd,xj,xld,
            1   xlamda,uo,uexp,vbp,utra,vmax,xneff,xl,xw,vbi,von,vdsat,qspof,
            2   beta0,beta1,cdrain,xqco,xqc,fnarrw,fshort,lev
            common /status/ omega,time,delta,delold(7),ag(7),vt,xni,egfet,
            1   xmu,sfactr,mode,modedc,icalc,initf,method,iord,maxord,noncon,
            2   iterno,itemno,nosolv,modac,ipiv,ivmflg,ipostp,iscrch,iofile
            common /knstnt/ twopi,xlog2,xlog10,root2,rad,boltz,charge,ctok,
            1   gmin,reltol,abstol,vntol,trtol,chgtol,eps0,epssil,epsox,
            2   pivtol,pivrel
            */

            /* equivalence (xlamda,alpha),(vbp,theta),(uexp,eta),(utra,xkappa)*/

            double coeff0 = 0.0631353e0;
            double coeff1 = 0.8013292e0;
            double coeff2 = -0.01110777e0;
            double oneoverxl;   /* 1/effective length */
            double eta; /* eta from model after length factor */
            double phibs;   /* phi - vbs */
            double sqphbs;  /* square root of phibs */
            double sqphis;  /* square root of phi */
            double wps;
            double oneoverxj;   /* 1/junction depth */
            double xjonxl;  /* junction depth/effective length */
            double djonxj;
            double wponxj;
            double arga;
            double argb;
            double argc;
            double gammas;
            double fbodys;
            double fbody;
            double onfbdy;
            double qbonco;
            double vbix;
            double wconxj;
            double vth;
            double csonco;
            double cdonco;
            double vgsx;
            double onfg;
            double fgate;
            double us;
            double xn = 0.0;
            double vdsc;
            double onvdsc = 0.0;
            double vdsx;
            double cdnorm;
            double cdo;
            double fdrain = 0.0;
            double gdsat;
            double cdsat;
            double emax;
            double delxl;
            double dlonxl;
            double xlfact;
            double ondvt;
            double onxn;
            double wfact;
            double fshort;
	    double lvds, lvbs, lvbd;
	    Dderivs d_onxn, d_ondvt, d_wfact, d_MOS9gds;
	    Dderivs d_emax, d_delxl, d_dlonxl, d_xlfact;
	    Dderivs d_cdonco, d_fdrain, d_cdsat, d_gdsat;
	    Dderivs d_vdsx, d_cdo, d_cdnorm, d_Beta, d_dummy;
	    Dderivs d_vdsc, d_onvdsc, d_arga, d_argb;
	    Dderivs d_onfg, d_fgate, d_us, d_vgsx;
	    Dderivs d_von, d_xn, d_vth, d_onfbdy, d_qbonco, d_vbix;
	    Dderivs d_argc, d_fshort, d_gammas, d_fbodys, d_fbody;
	    Dderivs d_wps, d_wconxj, d_wponxj;
	    Dderivs d_phibs, d_sqphbs;
	    Dderivs d_p, d_q, d_r, d_zero, d_vdsat;

            /*
             *     bypasses the computation of charges
             */
	     if (here->MOS9mode == 1) {
		lvgs = vgs;
		lvds = vds;
		lvbs = vbs;
		lvbd = vbd;
			} else {
			lvgs = vgd;
			lvds = -vds;
			lvbs = vbd;
			lvbd = vbs;
			}

            /*
             *     reference cdrain equations to source and
             *     charge equations to bulk
             */
d_p.value = 0.0;
d_p.d1_p = 1.0;
d_p.d1_q = 0.0;
d_p.d1_r = 0.0;
d_p.d2_p2 = 0.0;
d_p.d2_q2 = 0.0;
d_p.d2_r2 = 0.0;
d_p.d2_pq = 0.0;
d_p.d2_qr = 0.0;
d_p.d2_pr = 0.0;
d_p.d3_p3 = 0.0;
d_p.d3_q3 = 0.0;
d_p.d3_r3 = 0.0;
d_p.d3_p2r = 0.0;
d_p.d3_p2q = 0.0;
d_p.d3_q2r = 0.0;
d_p.d3_pq2 = 0.0;
d_p.d3_pr2 = 0.0;
d_p.d3_qr2 = 0.0;
d_p.d3_pqr = 0.0;
	EqualDeriv(&d_q,&d_p);
	EqualDeriv(&d_r,&d_p);
	EqualDeriv(&d_zero,&d_p);
    d_q.d1_p = d_r.d1_p = d_zero.d1_p = 0.0;
    d_q.d1_q = d_r.d1_r = 1.0;
            vdsat = 0.0;
	    EqualDeriv(&d_vdsat,&d_zero);
            oneoverxl = 1.0/EffectiveLength;/*const*/
            eta = model->MOS9eta * 8.15e-22/(model->MOS9oxideCapFactor*
                    EffectiveLength*EffectiveLength*EffectiveLength);/*const*/
            /*
             *.....square root term
             */
            if ( lvbs <=  0.0 ) {
                phibs  =  here->MOS9tPhi-lvbs;
		EqualDeriv(&d_phibs,&d_q);
		d_phibs.value = lvbs;
		TimesDeriv(&d_phibs,&d_phibs,-1.0);
		d_phibs.value += here->MOS9tPhi;
                sqphbs  =  sqrt(phibs);
		SqrtDeriv(&d_sqphbs,&d_phibs);
            } else {
                sqphis = sqrt(here->MOS9tPhi);/*const*/
                /*sqphs3 = here->MOS9tPhi*sqphis;const*/
                sqphbs = sqphis/(1.0+lvbs/
                    (here->MOS9tPhi+here->MOS9tPhi));
		EqualDeriv(&d_sqphbs,&d_q); d_sqphbs.value = lvbs;
		TimesDeriv(&d_sqphbs,&d_sqphbs,1/(here->MOS9tPhi+here->MOS9tPhi));
		d_sqphbs.value += 1.0;
		InvDeriv(&d_sqphbs,&d_sqphbs);
		TimesDeriv(&d_sqphbs,&d_sqphbs,sqphis);
                phibs = sqphbs*sqphbs;
		MultDeriv(&d_phibs,&d_sqphbs,&d_sqphbs);
            }
            /*
             *.....short channel effect factor
             */
            if ( (model->MOS9junctionDepth != 0.0) && 
                    (model->MOS9coeffDepLayWidth != 0.0) ) {
                wps = model->MOS9coeffDepLayWidth*sqphbs;
		TimesDeriv(&d_wps,&d_sqphbs,model->MOS9coeffDepLayWidth);
                oneoverxj = 1.0/model->MOS9junctionDepth;/*const*/
                xjonxl = model->MOS9junctionDepth*oneoverxl;/*const*/
                djonxj = model->MOS9latDiff*oneoverxj;/*const*/
                wponxj = wps*oneoverxj;
		TimesDeriv(&d_wponxj,&d_wps,oneoverxj);
                wconxj = coeff0+coeff1*wponxj+coeff2*wponxj*wponxj;
		TimesDeriv(&d_wconxj,&d_wponxj,coeff2);
		d_wconxj.value += coeff1;
		MultDeriv(&d_wconxj,&d_wconxj,&d_wponxj);
		d_wconxj.value += coeff0;
		arga = wconxj + djonxj;
		EqualDeriv(&d_arga,&d_wconxj); d_arga.value += djonxj;
                argc = wponxj/(1.0+wponxj);
		EqualDeriv(&d_argc,&d_wponxj);
		d_argc.value += 1.0;
		InvDeriv(&d_argc,&d_argc);
		MultDeriv(&d_argc,&d_argc,&d_wponxj);
                argb = sqrt(1.0-argc*argc);
		MultDeriv(&d_argb,&d_argc,&d_argc);
		TimesDeriv(&d_argb,&d_argb,-1.0);
		d_argb.value += 1.0;
		SqrtDeriv(&d_argb,&d_argb);

                fshort = 1.0-xjonxl*(arga*argb-djonxj);
		MultDeriv(&d_fshort,&d_arga,&d_argb);
		d_fshort.value -= djonxj;
		TimesDeriv(&d_fshort,&d_fshort,-xjonxl);
		d_fshort.value += 1.0;
            } else {
                fshort = 1.0;
		EqualDeriv(&d_fshort,&d_zero);
		d_fshort.value = 1.0;

            }
            /*
             *.....body effect
             */
            gammas = model->MOS9gamma*fshort;
	    TimesDeriv(&d_gammas,&d_fshort,model->MOS9gamma);
            fbodys = 0.5*gammas/(sqphbs+sqphbs);
	    DivDeriv(&d_fbodys,&d_gammas,&d_sqphbs);
	    TimesDeriv(&d_fbodys,&d_fbodys,0.25);
            fbody = fbodys+model->MOS9narrowFactor/EffectiveWidth;
	    EqualDeriv(&d_fbody,&d_fbodys);
	    d_fbody.value += fbody - fbodys;

            onfbdy = 1.0/(1.0+fbody);
	    EqualDeriv(&d_onfbdy,&d_fbody);
	    d_onfbdy.value += 1.0;
	    InvDeriv(&d_onfbdy,&d_onfbdy);
            qbonco =gammas*sqphbs+model->MOS9narrowFactor*phibs/EffectiveWidth;
	    EqualDeriv(&d_dummy,&d_phibs);
	    TimesDeriv(&d_dummy,&d_dummy,model->MOS9narrowFactor*EffectiveWidth);
	    MultDeriv(&d_qbonco,&d_gammas,&d_sqphbs);
	    PlusDeriv(&d_qbonco,&d_qbonco,&d_dummy);
            /*
             *.....static feedback effect
             */
            vbix = here->MOS9tVbi*model->MOS9type-eta*(lvds);
	    EqualDeriv(&d_vbix,&d_r); d_vbix.value = vbix;
	    d_vbix.d1_r = -eta;
            /*
             *.....threshold voltage
             */
            vth = vbix+qbonco;
	    PlusDeriv(&d_vth,&d_vbix,&d_qbonco);
            /*
             *.....joint weak inversion and strong inversion
             */
            von = vth;
	    EqualDeriv(&d_von,&d_vth);
            if ( model->MOS9fastSurfaceStateDensity != 0.0 ) {
                csonco = CHARGE*model->MOS9fastSurfaceStateDensity * 
                    1e4 /*(cm**2/m**2)*/ *EffectiveLength*EffectiveWidth *
                    here->MOS9m/OxideCap; /*const*/
                cdonco = 0.5*qbonco/phibs;
		DivDeriv(&d_cdonco,&d_qbonco,&d_phibs);
		TimesDeriv(&d_cdonco,&d_cdonco,0.5);
                xn = 1.0+csonco+cdonco;
		EqualDeriv(&d_xn,&d_cdonco);
		d_xn.value += 1.0 + csonco;
                von = vth+vt*xn;
		TimesDeriv(&d_von,&d_xn,vt);
		PlusDeriv(&d_von,&d_von,&d_vth);


            } else {
                /*
                 *.....cutoff region
                 */
                if ( lvgs <= von ) {
                    cdrain = 0.0;
		    EqualDeriv(&d_cdrain,&d_zero);
                    goto innerline1000;
                }
            }
            /*
             *.....device is on
             */
            vgsx = MAX(lvgs,von);
if (lvgs >= von) {
EqualDeriv(&d_vgsx,&d_p);
d_vgsx.value = lvgs;
} else {
EqualDeriv(&d_vgsx,&d_von);
}
            /*
             *.....mobility modulation by gate voltage
             */
            onfg = 1.0+model->MOS9theta*(vgsx-vth);
	    TimesDeriv(&d_onfg,&d_vth,-1.0);
	    PlusDeriv(&d_onfg,&d_onfg,&d_vgsx);
	    TimesDeriv(&d_onfg,&d_onfg,model->MOS9theta);
	    d_onfg.value += 1.0;
            fgate = 1.0/onfg;
	    InvDeriv(&d_fgate,&d_onfg);
            us = here->MOS9tSurfMob * 1e-4 /*(m**2/cm**2)*/ *fgate;
	    TimesDeriv(&d_us,&d_fgate,here->MOS9tSurfMob * 1e-4);
            /*
             *.....saturation voltage
             */
            vdsat = (vgsx-vth)*onfbdy;
	    TimesDeriv(&d_vdsat,&d_vth, -1.0);
	    PlusDeriv(&d_vdsat,&d_vdsat,&d_vgsx);
	    MultDeriv(&d_vdsat,&d_vdsat,&d_onfbdy);
            if ( model->MOS9maxDriftVel <= 0.0 ) {
            } else {
                vdsc = EffectiveLength*model->MOS9maxDriftVel/us;
		InvDeriv(&d_vdsc,&d_us);
		TimesDeriv(&d_vdsc,&d_vdsc,EffectiveLength*model->MOS9maxDriftVel);
                onvdsc = 1.0/vdsc;
		InvDeriv(&d_onvdsc,&d_vdsc);
                arga = (vgsx-vth)*onfbdy;
		/* note arga = vdsat at this point */
		EqualDeriv(&d_arga,&d_vdsat);
                argb = sqrt(arga*arga+vdsc*vdsc);
		MultDeriv(&d_dummy,&d_arga,&d_arga);
		MultDeriv(&d_argb,&d_vdsc,&d_vdsc);
		PlusDeriv(&d_argb,&d_argb,&d_dummy);
		SqrtDeriv(&d_argb,&d_argb);
                vdsat = arga+vdsc-argb;
		TimesDeriv(&d_vdsat,&d_argb,-1.0);
		PlusDeriv(&d_vdsat,&d_vdsat,&d_vdsc);
		PlusDeriv(&d_vdsat,&d_vdsat,&d_arga);
            }
            /*
             *.....current factors in linear region
             */
            vdsx = MIN((lvds),vdsat);
if (lvds < vdsat) {
EqualDeriv(&d_vdsx,&d_r);
d_vdsx.value = lvds;
} else {
EqualDeriv(&d_vdsx,&d_vdsat);
}

            if ( vdsx == 0.0 ) goto line900;
            cdo = vgsx-vth-0.5*(1.0+fbody)*vdsx;
	    EqualDeriv(&d_cdo,&d_fbody);
	    d_cdo.value += 1.0;
	    MultDeriv(&d_cdo,&d_cdo,&d_vdsx);
	    TimesDeriv(&d_cdo,&d_cdo,0.5);
	    PlusDeriv(&d_cdo,&d_cdo,&d_vth);
	    TimesDeriv(&d_cdo,&d_cdo,-1.0);
	    PlusDeriv(&d_cdo,&d_cdo,&d_vgsx);


            /* 
             *.....normalized drain current
             */
            cdnorm = cdo*vdsx;
	    MultDeriv(&d_cdnorm,&d_cdo,&d_vdsx);
            /* 
             *.....drain current without velocity saturation effect
             */
/* Beta is a constant till now */
            Beta = Beta*fgate;
	    TimesDeriv(&d_Beta,&d_fgate,Beta);
            cdrain = Beta*cdnorm;
	    MultDeriv(&d_cdrain,&d_Beta,&d_cdnorm);
            /*
             *.....velocity saturation factor
             */
            if ( model->MOS9maxDriftVel != 0.0 ) {
                fdrain = 1.0/(1.0+vdsx*onvdsc);
		MultDeriv(&d_fdrain,&d_vdsx,&d_onvdsc);
		d_fdrain.value += 1.0;
		InvDeriv(&d_fdrain,&d_fdrain);
                /*
                 *.....drain current
                 */
	    cdrain = fdrain*cdrain;
	    MultDeriv(&d_cdrain,&d_cdrain,&d_fdrain);
	    Beta = Beta*fdrain;
	    MultDeriv(&d_Beta,&d_Beta,&d_fdrain);
		
            }
            /*
             *.....channel length modulation
             */
            if ( (lvds) <= vdsat ) goto line700;
            if ( model->MOS9maxDriftVel == 0.0 ) goto line510;
            if (model->MOS9alpha == 0.0) goto line700;
            cdsat = cdrain;
	    EqualDeriv(&d_cdsat,&d_cdrain);
            gdsat = cdsat*(1.0-fdrain)*onvdsc;
	    TimesDeriv(&d_dummy,&d_fdrain,-1.0);
	    d_dummy.value += 1.0;
	    MultDeriv(&d_gdsat,&d_cdsat,&d_dummy);
	    MultDeriv(&d_gdsat,&d_gdsat,&d_onvdsc);
            gdsat = MAX(1.0e-12,gdsat);
	    if (gdsat == 1.0e-12) {
		EqualDeriv(&d_gdsat,&d_zero);
		d_gdsat.value = gdsat;
		}

            emax = cdsat*oneoverxl/gdsat;
	    DivDeriv(&d_emax,&d_cdsat,&d_gdsat);
	    TimesDeriv(&d_emax,&d_emax,oneoverxl);


            arga = 0.5*emax*model->MOS9alpha;
	    TimesDeriv(&d_arga,&d_emax,0.5*model->MOS9alpha);
            argc = model->MOS9kappa*model->MOS9alpha;/*const*/
            argb = sqrt(arga*arga+argc*((lvds)-vdsat));
	    TimesDeriv(&d_dummy,&d_vdsat,-1.0);
	    d_dummy.value += lvds;
	    d_dummy.d1_r += 1.0;
	    TimesDeriv(&d_argb,&d_dummy,argc);
	    MultDeriv(&d_dummy,&d_arga,&d_arga);
	    PlusDeriv(&d_argb,&d_argb,&d_dummy);
	    SqrtDeriv(&d_argb,&d_argb);
            delxl = argb-arga;
	    TimesDeriv(&d_delxl,&d_arga,-1.0);
	    PlusDeriv(&d_delxl,&d_argb,&d_delxl);
            goto line520;
line510:
            delxl = sqrt(model->MOS9kappa*((lvds)-vdsat)*model->MOS9alpha);
	    TimesDeriv(&d_delxl,&d_vdsat,-1.0);
	    d_delxl.value += lvds;
	    d_delxl.d1_r += 1.0;
	    TimesDeriv(&d_delxl,&d_delxl,model->MOS9kappa*model->MOS9alpha);
	    SqrtDeriv(&d_delxl,&d_delxl);
	    
            /*
             *.....punch through approximation
             */
line520:
            if ( delxl > (0.5*EffectiveLength) ) {
	    delxl = EffectiveLength - (EffectiveLength*EffectiveLength/
		delxl*0.25);
	    InvDeriv(&d_delxl,&d_delxl);
	    TimesDeriv(&d_delxl,&d_delxl,-EffectiveLength*EffectiveLength*0.25);
	    d_delxl.value += EffectiveLength;
            }
            /*
             *.....saturation region
             */
            dlonxl = delxl*oneoverxl;
	    TimesDeriv(&d_dlonxl,&d_delxl,oneoverxl);
            xlfact = 1.0/(1.0-dlonxl);
	    TimesDeriv(&d_xlfact,&d_dlonxl,-1.0);
	    d_xlfact.value += 1.0;
	    InvDeriv(&d_xlfact,&d_xlfact);

            cdrain = cdrain*xlfact;
	    MultDeriv(&d_cdrain,&d_cdrain,&d_xlfact);
            /*
             *.....finish strong inversion case
             */
line700:
            if ( lvgs < von ) {
                /*
                 *.....weak inversion
                 */
                onxn = 1.0/xn;
		InvDeriv(&d_onxn,&d_xn);
		ondvt = onxn/vt;
		TimesDeriv(&d_ondvt,&d_onxn,1/vt);
                wfact = exp( (lvgs-von)*ondvt );
		TimesDeriv(&d_wfact,&d_von,-1.0);
		d_wfact.value += lvgs;
		d_wfact.d1_p += 1.0;
		MultDeriv(&d_wfact,&d_wfact,&d_ondvt);
		ExpDeriv(&d_wfact,&d_wfact);
                cdrain = cdrain*wfact;
		MultDeriv(&d_cdrain,&d_cdrain,&d_wfact);
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
	    /* Beta is still a constant */
	    TimesDeriv(&d_Beta,&d_fgate,Beta);
            cdrain = 0.0;
	    EqualDeriv(&d_cdrain,&d_zero);
            here->MOS9gds = Beta*(vgsx-vth);
	    TimesDeriv(&d_MOS9gds,&d_vth,-1.0);
	    PlusDeriv(&d_MOS9gds,&d_MOS9gds,&d_vgsx);
	    MultDeriv(&d_MOS9gds,&d_MOS9gds,&d_Beta);
            if ( (model->MOS9fastSurfaceStateDensity != 0.0) && 
                    (lvgs < von) ) {
                here->MOS9gds *=exp((lvgs-von)/(vt*xn));
		TimesDeriv(&d_dummy,&d_von,-1.0);
		d_dummy.value += lvgs;
		d_dummy.d1_p += 1.0;
		DivDeriv(&d_dummy,&d_dummy,&d_xn);
		TimesDeriv(&d_dummy,&d_dummy,1/vt);
		ExpDeriv(&d_dummy,&d_dummy);
		MultDeriv(&d_MOS9gds,&d_MOS9gds,&d_dummy);
            }
	    d_cdrain.d1_r = d_MOS9gds.value;
	    d_cdrain.d2_r2 = d_MOS9gds.d1_r;
	    d_cdrain.d3_r3 = d_MOS9gds.d2_r2;



innerline1000:;
            /* 
             *.....done
             */
            }


            /*
             *  COMPUTE EQUIVALENT DRAIN CURRENT SOURCE
             */
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
                    if (vbs < here->MOS9tDepCap){
                        arg=1-vbs/here->MOS9tBulkPot;
                        /*
                         * the following block looks somewhat long and messy,
                         * but since most users use the default grading
                         * coefficients of .5, and sqrt is MUCH faster than an
                         * exp(log()) we use this special case code to buy time.
                         * (as much as 10% of total job time!)
                         */
                        if(model->MOS9bulkJctBotGradingCoeff ==
                                model->MOS9bulkJctSideGradingCoeff) {
                            if(model->MOS9bulkJctBotGradingCoeff == .5) {
                                sarg = sargsw = 1/sqrt(arg);
                            } else {
                                sarg = sargsw =
                                        exp(-model->MOS9bulkJctBotGradingCoeff*
                                        log(arg));
                            }
                        } else {
                            if(model->MOS9bulkJctBotGradingCoeff == .5) {
                                sarg = 1/sqrt(arg);
                            } else {
                                sarg = exp(-model->MOS9bulkJctBotGradingCoeff*
                                        log(arg));
                            }
                            if(model->MOS9bulkJctSideGradingCoeff == .5) {
                                sargsw = 1/sqrt(arg);
                            } else {
                                sargsw =exp(-model->MOS9bulkJctSideGradingCoeff*
                                        log(arg));
                            }
                        }
		    lcapbs=here->MOS9Cbs*sarg+
                                here->MOS9Cbssw*sargsw;
		    lcapbs2 = model->MOS9type*0.5/here->MOS9tBulkPot*(
			here->MOS9Cbs*model->MOS9bulkJctBotGradingCoeff*
			sarg/arg + here->MOS9Cbssw*
			model->MOS9bulkJctSideGradingCoeff*sargsw/arg);
		    lcapbs3 = here->MOS9Cbs*sarg*
			model->MOS9bulkJctBotGradingCoeff*
			(model->MOS9bulkJctBotGradingCoeff+1);
		    lcapbs3 += here->MOS9Cbssw*sargsw*
			model->MOS9bulkJctSideGradingCoeff*
			(model->MOS9bulkJctSideGradingCoeff+1);
		    lcapbs3 = lcapbs3/(6*here->MOS9tBulkPot*
			here->MOS9tBulkPot*arg*arg);
                    } else {
                    /*    *(ckt->CKTstate0 + here->MOS9qbs)= here->MOS9f4s +
                                vbs*(here->MOS9f2s+vbs*(here->MOS9f3s/2));*/
                        lcapbs=here->MOS9f2s+here->MOS9f3s*vbs;
			lcapbs2 = 0.5*here->MOS9f3s;
			lcapbs3 = 0;
                    }
                    if (vbd < here->MOS9tDepCap) {
                        arg=1-vbd/here->MOS9tBulkPot;
                        /*
                         * the following block looks somewhat long and messy,
                         * but since most users use the default grading
                         * coefficients of .5, and sqrt is MUCH faster than an
                         * exp(log()) we use this special case code to buy time.
                         * (as much as 10% of total job time!)
                         */
                        if(model->MOS9bulkJctBotGradingCoeff == .5 &&
                                model->MOS9bulkJctSideGradingCoeff == .5) {
                            sarg = sargsw = 1/sqrt(arg);
                        } else {
                            if(model->MOS9bulkJctBotGradingCoeff == .5) {
                                sarg = 1/sqrt(arg);
                            } else {
                                sarg = exp(-model->MOS9bulkJctBotGradingCoeff*
                                        log(arg));
                            }
                            if(model->MOS9bulkJctSideGradingCoeff == .5) {
                                sargsw = 1/sqrt(arg);
                            } else {
                                sargsw =exp(-model->MOS9bulkJctSideGradingCoeff*
                                        log(arg));
                            }
                        }
		    lcapbd=here->MOS9Cbd*sarg+
                                here->MOS9Cbdsw*sargsw;
		    lcapbd2 = model->MOS9type*0.5/here->MOS9tBulkPot*(
			here->MOS9Cbd*model->MOS9bulkJctBotGradingCoeff*
			sarg/arg + here->MOS9Cbdsw*
			model->MOS9bulkJctSideGradingCoeff*sargsw/arg);
		    lcapbd3 = here->MOS9Cbd*sarg*
			model->MOS9bulkJctBotGradingCoeff*
			(model->MOS9bulkJctBotGradingCoeff+1);
		    lcapbd3 += here->MOS9Cbdsw*sargsw*
			model->MOS9bulkJctSideGradingCoeff*
			(model->MOS9bulkJctSideGradingCoeff+1);
		    lcapbd3 = lcapbd3/(6*here->MOS9tBulkPot*
			here->MOS9tBulkPot*arg*arg);
                    } else {
                        lcapbd=here->MOS9f2d + vbd * here->MOS9f3d;
			lcapbd2=0.5*here->MOS9f3d;
			lcapbd3=0;
                    }
            /*
             *     meyer's capacitor model
             */
	/*
	 * the meyer capacitance equations are in DEVqmeyer
	 * these expressions are derived from those equations.
	 * these expressions are incorrect; they assume just one
	 * controlling variable for each charge storage element
	 * while actually there are several;  the MOS9 small
	 * signal ac linear model is also wrong because it 
	 * ignores controlled capacitive elements. these can be 
	 * corrected (as can the linear ss ac model) if the 
	 * expressions for the charge are available
	 */

	 
{


    double phi;
    double cox;
    double vddif;
    double vddif1;
    double vddif2;
    double vgst;
    /* von, lvgs and vdsat have already been adjusted for 
        possible source-drain interchange */



    vgst = lvgs -von;
    phi = here->MOS9tPhi;
    cox = OxideCap;
    if (vgst <= -phi) {
    lcapgb2=lcapgb3=lcapgs2=lcapgs3=lcapgd2=lcapgd3=0;
    } else if (vgst <= -phi/2) {
    lcapgb2= -cox/(4*phi);
    lcapgb3=lcapgs2=lcapgs3=lcapgd2=lcapgd3=0;
    } else if (vgst <= 0) {
    lcapgb2= -cox/(4*phi);
    lcapgb3=lcapgs3=lcapgd2=lcapgd3=0;
    lcapgs2 = cox/(3*phi);
    } else  {			/* the MOS9modes are around because 
					vds has not been adjusted */
        if (vdsat <= here->MOS9mode*vds) {
	lcapgb2=lcapgb3=lcapgs2=lcapgs3=lcapgd2=lcapgd3=0;
        } else {
            vddif = 2.0*vdsat-here->MOS9mode*vds;
            vddif1 = vdsat-here->MOS9mode*vds/*-1.0e-12*/;
            vddif2 = vddif*vddif;
	    lcapgd2 = -vdsat*here->MOS9mode*vds*cox/(3*vddif*vddif2);
	    lcapgd3 = - here->MOS9mode*vds*cox*(vddif - 6*vdsat)/(9*vddif2*vddif2);
	    lcapgs2 = -vddif1*here->MOS9mode*vds*cox/(3*vddif*vddif2);
	    lcapgs3 = - here->MOS9mode*vds*cox*(vddif - 6*vddif1)/(9*vddif2*vddif2);
	    lcapgb2=lcapgb3=0;
        }
    }
    }

		/* the b-s and b-d diodes need no processing ...  */
	here->capbs2 = lcapbs2;
	here->capbs3 = lcapbs3;
	here->capbd2 = lcapbd2;
	here->capbd3 = lcapbd3;
	here->gbs2 = lgbs2;
	here->gbs3 = lgbs3;
	here->gbd2 = lgbd2;
	here->gbd3 = lgbd3;
	here->capgb2 = model->MOS9type*lcapgb2;
	here->capgb3 = lcapgb3;
                /*
                 *   process to get Taylor coefficients, taking into
		 * account type and mode.
                 */
gm2 =  d_cdrain.d2_p2;
gb2 =  d_cdrain.d2_q2;
gds2 =  d_cdrain.d2_r2;
gmb =  d_cdrain.d2_pq;
gbds =  d_cdrain.d2_qr;
gmds =  d_cdrain.d2_pr;
gm3 =  d_cdrain.d3_p3;
gb3 =  d_cdrain.d3_q3;
gds3 =  d_cdrain.d3_r3;
gm2ds =  d_cdrain.d3_p2r;
gm2b =  d_cdrain.d3_p2q;
gb2ds =  d_cdrain.d3_q2r;
gmb2 =  d_cdrain.d3_pq2;
gmds2 =  d_cdrain.d3_pr2;
gbds2 =  d_cdrain.d3_qr2;
gmbds =  d_cdrain.d3_pqr;

	if (here->MOS9mode == 1)
		{
		/* normal mode - no source-drain interchange */

 here->cdr_x2 = gm2;
 here->cdr_y2 = gb2;;
 here->cdr_z2 = gds2;;
 here->cdr_xy = gmb;
 here->cdr_yz = gbds;
 here->cdr_xz = gmds;
 here->cdr_x3 = gm3;
 here->cdr_y3 = gb3;
 here->cdr_z3 = gds3;
 here->cdr_x2z = gm2ds;
 here->cdr_x2y = gm2b;
 here->cdr_y2z = gb2ds;
 here->cdr_xy2 = gmb2;
 here->cdr_xz2 = gmds2;
 here->cdr_yz2 = gbds2;
 here->cdr_xyz = gmbds;

		/* the gate caps have been divided and made into
			Taylor coeffs., but not adjusted for type */

	here->capgs2 = model->MOS9type*lcapgs2;
	here->capgs3 = lcapgs3;
	here->capgd2 = model->MOS9type*lcapgd2;
	here->capgd3 = lcapgd3;
} else {
		/*
		 * inverse mode - source and drain interchanged
		 */

here->cdr_x2 = -gm2;
here->cdr_y2 = -gb2;
here->cdr_z2 = -(gm2 + gb2 + gds2 + 2*(gmb + gmds + gbds));
here->cdr_xy = -gmb;
here->cdr_yz = gmb + gb2 + gbds;
here->cdr_xz = gm2 + gmb + gmds;
here->cdr_x3 = -gm3;
here->cdr_y3 = -gb3;
here->cdr_z3 = gm3 + gb3 + gds3 + 
	3*(gm2b + gm2ds + gmb2 + gb2ds + gmds2 + gbds2) + 6*gmbds ;
here->cdr_x2z = gm3 + gm2b + gm2ds;
here->cdr_x2y = -gm2b;
here->cdr_y2z = gmb2 + gb3 + gb2ds;
here->cdr_xy2 = -gmb2;
here->cdr_xz2 = -(gm3 + 2*(gm2b + gm2ds + gmbds) +
					gmb2 + gmds2);
here->cdr_yz2 = -(gb3 + 2*(gmb2 + gb2ds + gmbds) +
					gm2b + gbds2);
here->cdr_xyz = gm2b + gmb2 + gmbds;

          here->capgs2 = model->MOS9type*lcapgd2;
	  here->capgs3 = lcapgd3;

	  here->capgd2 = model->MOS9type*lcapgs2;
	  here->capgd3 = lcapgs3;

}

/* now to adjust for type and multiply by factors to convert to Taylor coeffs. */

here->cdr_x2 = 0.5*model->MOS9type*here->cdr_x2;
here->cdr_y2 = 0.5*model->MOS9type*here->cdr_y2;
here->cdr_z2 = 0.5*model->MOS9type*here->cdr_z2;
here->cdr_xy = model->MOS9type*here->cdr_xy;
here->cdr_yz = model->MOS9type*here->cdr_yz;
here->cdr_xz = model->MOS9type*here->cdr_xz;
here->cdr_x3 = here->cdr_x3/6.;
here->cdr_y3 = here->cdr_y3/6.;
here->cdr_z3 = here->cdr_z3/6.;
here->cdr_x2z = 0.5*here->cdr_x2z;
here->cdr_x2y = 0.5*here->cdr_x2y;
here->cdr_y2z = 0.5*here->cdr_y2z;
here->cdr_xy2 = 0.5*here->cdr_xy2;
here->cdr_xz2 = 0.5*here->cdr_xz2;
here->cdr_yz2 = 0.5*here->cdr_yz2;


        }
    }
    return(OK);
}
