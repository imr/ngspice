/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jaijeet S Roychowdhury
Modified: 2000 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/distodef.h"
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
MOS2dSetup(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current value into the 
         * sparse matrix previously provided 
         */
{
    MOS2model *model = (MOS2model *)inModel;
    MOS2instance *here;
    double Beta;
    double DrainSatCur;
    double EffectiveLength;
    double GateBulkOverlapCap;
    double GateDrainOverlapCap;
    double GateSourceOverlapCap;
    double OxideCap;
    double SourceSatCur;
    double arg;
    double cdrain;
    double ebd;
    double evbs;
    double sarg;
    double sargsw;
    double vbd;
    double vbs;
    double vds;
    double vdsat;
    double vgb;
    double vgd;
    double vgs;
    double von;
    double vt;      /* K * T / Q */
    double lcapgs2;
    double lcapgd2; 
    double lcapgb2;
    double lcapgs3;  
    double lcapgd3; 
    double lcapgb3;
    double lgbs, lgbs2, lgbs3;
    double lgbd, lgbd2, lgbd3;
            double vgst;
	double lcapbs, lcapbs2, lcapbs3;
	double lcapbd, lcapbd2, lcapbd3;
	double gm2, gb2, gds2;
	double gmb, gmds, gbds;
	double gm3, gb3, gds3;
	double gm2b, gm2ds, gmb2, gmds2, gbds2, gb2ds;
	double gmbds;
		Dderivs d_cdrain;
	/*remove compiler warnings */
	d_cdrain.value = 0.0;
	d_cdrain.d1_p = 0.0;
	d_cdrain.d1_q = 0.0;
	d_cdrain.d1_r = 0.0;
	d_cdrain.d2_p2 = 0.0;
	d_cdrain.d2_q2 = 0.0;
	d_cdrain.d2_r2 = 0.0;
	d_cdrain.d2_pq = 0.0;
	d_cdrain.d2_qr = 0.0;
	d_cdrain.d2_pr = 0.0;
	d_cdrain.d3_p3 = 0.0;
	d_cdrain.d3_q3 = 0.0;
	d_cdrain.d3_r3 = 0.0;
	d_cdrain.d3_p2q = 0.0;
	d_cdrain.d3_p2r = 0.0;
	d_cdrain.d3_pq2 = 0.0;
	d_cdrain.d3_q2r = 0.0;
	d_cdrain.d3_pr2 = 0.0;
	d_cdrain.d3_qr2 = 0.0;
	d_cdrain.d3_pqr = 0.0;

    /*  loop through all the MOS2 device models */
    for( ; model != NULL; model = MOS2nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = MOS2instances(model); here != NULL ;
                here=MOS2nextInstance(here)) {

            vt = CONSTKoverQ * here->MOS2temp;

            EffectiveLength=here->MOS2l - 2*model->MOS2latDiff;
           
             if( (here->MOS2tSatCurDens == 0) || 
                    (here->MOS2drainArea == 0) ||
                    (here->MOS2sourceArea == 0)) {
                DrainSatCur = here->MOS2m * here->MOS2tSatCur;
                SourceSatCur = here->MOS2m * here->MOS2tSatCur;
            } else {
                DrainSatCur = here->MOS2tSatCurDens * 
                        here->MOS2m * here->MOS2drainArea;
                SourceSatCur = here->MOS2tSatCurDens * 
                        here->MOS2m * here->MOS2sourceArea;
            }
            GateSourceOverlapCap = model->MOS2gateSourceOverlapCapFactor * 
                    here->MOS2m * here->MOS2w;
            GateDrainOverlapCap = model->MOS2gateDrainOverlapCapFactor * 
                    here->MOS2m * here->MOS2w;
            GateBulkOverlapCap = model->MOS2gateBulkOverlapCapFactor * 
                    here->MOS2m * EffectiveLength;
            Beta = here->MOS2tTransconductance * here->MOS2m *
                    here->MOS2w/EffectiveLength;
            OxideCap = model->MOS2oxideCapFactor * EffectiveLength * 
                    here->MOS2m * here->MOS2w;





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

                /* now some common crunching for some more useful quantities */


                    vbd=vbs-vds;
                    vgd=vgs-vds;

            /* now all the preliminaries are over - we can start doing the             * real work 
             */

            vgb = vgs - vbs;

            /* bulk-source and bulk-drain doides here we just evaluate
             * the ideal diode current and the correspoinding
             * derivative (conductance).  */
	    if(vbs <= 0) {
                lgbs = SourceSatCur/vt;
                lgbs += ckt->CKTgmin;
		lgbs2 = lgbs3 = 0;
            } else {
                evbs = exp(MIN(MAX_EXP_ARG,vbs/vt));
                lgbs = SourceSatCur*evbs/vt + ckt->CKTgmin;
		lgbs2 = model->MOS2type *0.5 * (lgbs - ckt->CKTgmin)/vt;
		lgbs3 = model->MOS2type *lgbs2/(vt*3);

            }
            if(vbd <= 0) {
                lgbd = DrainSatCur/vt;
                lgbd += ckt->CKTgmin;
		lgbd2 = lgbd3 = 0;
            } else {
                ebd = exp(MIN(MAX_EXP_ARG,vbd/vt));
                lgbd = DrainSatCur*ebd/vt +ckt->CKTgmin;
		lgbd2 = model->MOS2type *0.5 * (lgbd - ckt->CKTgmin)/vt;
		lgbd3 = model->MOS2type *lgbd2/(vt*3);
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
             *     this routine evaluates the drain current, its derivatives and             *     the charges associated with the gate, channel and bulk             *     for mosfets             *
             */

            double arg1;
            double sarg1;
            double a4[4],b4[4],x4[8],poly4[8];
            double beta1;
            double sphi = 0.0;    /* square root of phi */
            double sphi3 = 0.0;   /* square root of phi cubed */
            double barg;
            double factor;
            double eta;
            double vbin;
            double argd = 0.0;
            double args = 0.0;
            double argss;
            double argsd;
            double argxs;
            double argxd;
            double gamasd;
            double xwd;
            double xws;
            double gammad;
            double cfs;
            double cdonco;
            double xn;
            double argg = 0.0;
            double sarg3;
            double sbiarg;
            double body;
            double udenom;
            double gammd2;
            double argv;
            double vgsx;
            double ufact;
            double ueff;
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
            double bodys = 0.0;
            double sargv;
            double xlfact;
            double xdv;
            double xlv;
            double xls;
            double clfact;
            double xleff;
            double deltal;
            double xwb;
            double vdson;
            double cdson;
            double expg;
            double xld;
	double xlamda = model->MOS2lambda;
	Dderivs d_xleff, d_delta1;
	Dderivs d_xlfact;
	Dderivs d_xlv, d_xls;
	Dderivs d_bsarg, d_bodys, d_vdsat, d_sargv;
	Dderivs d_delta4, d_a4[3], d_b4[3], d_x4[3], d_poly4[3];
	Dderivs d_xvalid;
	Dderivs d_ro, d_fi, d_y3, d_p3, d_p4, d_a3, d_b3;
	Dderivs d_r3, d_s2, d_pee, d_p0, d_p2;
	Dderivs d_b1, d_c1, d_d1, d_a, d_b, d_c, d_arr, d_s;
	Dderivs d_xv, d_a1;
	Dderivs d_v1, d_v2;
	Dderivs d_argv,d_gammd2;
	Dderivs d_ufact;
	Dderivs d_udenom;
	Dderivs d_sarg3, d_body;
	Dderivs d_vgst;
	Dderivs d_argg;
	Dderivs d_cdonco,d_tmp,d_xn;
	Dderivs d_dbargs,d_dbargd,d_dgddvb;
	Dderivs d_dbxwd,d_dbxws;
	Dderivs d_dsrgdb, d_dbrgdb;
	Dderivs d_gamasd, d_gammad, d_args, d_argd;
	Dderivs d_argxs, d_argxd;
	Dderivs d_argss, d_argsd;
	Dderivs d_xwd, d_xws;
	Dderivs d_zero;
	Dderivs d_vbin;
	Dderivs d_barg;
	Dderivs d_sarg;
	Dderivs d_phiMinVbs;
    Dderivs d_p, d_q, d_r;
    Dderivs d_von, d_dummy, d_vgsx, d_arg, d_dumarg;
    Dderivs d_ueff, d_beta1, d_clfact, d_xlamda,d_mos2gds;
    Dderivs d_vdson, d_cdson, d_expg;
    double dsrgdb, dbrgdb, dbxwd, dbxws, dbargs = 0.0, dbargd = 0.0;
    double dgddvb;


/* from now on, p=vgs, q=vbs, r=vds  */

/*
 * 'local' variables - these switch d & s around appropriately
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

	EqualDeriv(&d_phiMinVbs,&d_q);
	d_phiMinVbs.value = phiMinVbs;
	d_phiMinVbs.d1_q = -  d_phiMinVbs.d1_q;

	if (lvbs <= 0.0) {
	    sarg1 = sqrt(phiMinVbs);
	    SqrtDeriv(&d_sarg, &d_phiMinVbs);
	    dsrgdb = -0.5/sarg1;
	    InvDeriv(&d_dsrgdb,&d_sarg);
	    TimesDeriv(&d_dsrgdb,&d_dsrgdb,-0.5);

	} else {
	    sphi = sqrt(here->MOS2tPhi); /*const*/
	    sphi3 = here->MOS2tPhi*sphi; /*const*/
	    sarg1 = sphi/(1.0+0.5*lvbs/here->MOS2tPhi);
	    EqualDeriv(&d_sarg,&d_q); d_sarg.value = lvbs;
	    TimesDeriv(&d_sarg,&d_sarg,0.5/here->MOS2tPhi);
	    d_sarg.value += 1.0;
	    InvDeriv(&d_sarg,&d_sarg);
	    TimesDeriv(&d_sarg,&d_sarg,sphi);
	    dsrgdb = -0.5*sarg1*sarg1/sphi3;
	    MultDeriv(&d_dsrgdb,&d_sarg,&d_sarg);
	    TimesDeriv(&d_dsrgdb,&d_dsrgdb,-0.5/sphi3);
	    /* tmp = sarg/sphi3; */
	}
	if ((lvds-lvbs) >= 0) {
	    barg = sqrt(phiMinVbs+lvds);
	    EqualDeriv(&d_barg,&d_phiMinVbs);
	    d_barg.value += lvds; d_barg.d1_r += 1.0;
	    SqrtDeriv(&d_barg,&d_barg);
	    dbrgdb = -0.5/barg;
	    InvDeriv(&d_dbrgdb,&d_barg);
	    TimesDeriv(&d_dbrgdb,&d_dbrgdb,-0.5);
	} else {
	    sphi = sqrt(here->MOS2tPhi); /* added by HT 050523 */
	    sphi3 = here->MOS2tPhi*sphi; /* added by HT 050523 */
	    barg = sphi/(1.0+0.5*(lvbs-lvds)/here->MOS2tPhi);
	    EqualDeriv(&d_barg,&d_q); d_barg.value = lvbs - lvds;
	    d_barg.d1_r -= 1.0;
	    TimesDeriv(&d_barg,&d_barg,0.5/here->MOS2tPhi);
	    d_barg.value += 1.0;
	    InvDeriv(&d_barg,&d_barg);
	    TimesDeriv(&d_barg,&d_barg,sphi);
	    dbrgdb = -0.5*barg*barg/sphi3;
	    MultDeriv(&d_dbrgdb,&d_barg,&d_barg);
	    TimesDeriv(&d_dbrgdb,&d_dbrgdb,-0.5/sphi3);
	    /* tmp = barg/sphi3; */
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
	/* mistake! fixed Dec 7 '89
	 TimesDeriv(&d_vbin,&d_phiMinVbs,here->MOS2tVbi*
		    model->MOS2type+factor);
		    */
	 TimesDeriv(&d_vbin,&d_phiMinVbs,factor);
	 d_vbin.value += here->MOS2tVbi*model->MOS2type;

	if ((model->MOS2gamma > 0.0) || 
		(model->MOS2substrateDoping > 0.0)) {
	    xwd = model->MOS2xd*barg;
	    xws = model->MOS2xd*sarg1;
	    TimesDeriv(&d_xwd,&d_barg,model->MOS2xd);
	    TimesDeriv(&d_xws,&d_sarg,model->MOS2xd);

	    /*
	     *     short-channel effect with vds .ne. 0.0           */

	    argss = 0.0;
	    argsd = 0.0;
	    EqualDeriv(&d_argss,&d_zero);
	    EqualDeriv(&d_argsd,&d_zero);
	    if (model->MOS2junctionDepth > 0) {
		tmp = 2.0/model->MOS2junctionDepth; /*const*/
		argxs = 1.0+xws*tmp;
		TimesDeriv(&d_argxs,&d_xws,tmp);
		d_argxs.value += 1.0;
		argxd = 1.0+xwd*tmp;
		TimesDeriv(&d_argxd,&d_xwd,tmp);
		d_argxd.value += 1.0;
		args = sqrt(argxs);
		SqrtDeriv(&d_args,&d_argxs);
		argd = sqrt(argxd);
		SqrtDeriv(&d_argd,&d_argxd);
		tmp = .5*model->MOS2junctionDepth/EffectiveLength;
		argss = tmp * (args-1.0);
		TimesDeriv(&d_argss,&d_args,tmp);
		d_argss.value -= tmp;
		argsd = tmp * (argd-1.0);
		TimesDeriv(&d_argsd,&d_argd,tmp);
		d_argsd.value -= tmp;
	    }
	    gamasd = model->MOS2gamma*(1.0-argss-argsd);
	    PlusDeriv(&d_gamasd,&d_argss,&d_argsd);
	    d_gamasd.value -= 1.0;
	    TimesDeriv(&d_gamasd,&d_gamasd,-model->MOS2gamma);
	    dbxwd = model->MOS2xd*dbrgdb;
	    dbxws = model->MOS2xd*dsrgdb;
	    TimesDeriv(&d_dbxwd,&d_dbrgdb,model->MOS2xd);
	    TimesDeriv(&d_dbxws,&d_dsrgdb,model->MOS2xd);
	    if (model->MOS2junctionDepth > 0) {
		tmp = 0.5/EffectiveLength;
		dbargs = tmp*dbxws/args;
		dbargd = tmp*dbxwd/argd;
		DivDeriv(&d_dbargs,&d_dbxws,&d_args);
		DivDeriv(&d_dbargd,&d_dbxwd,&d_argd);
		TimesDeriv(&d_dbargs,&d_dbargs,tmp);
		TimesDeriv(&d_dbargd,&d_dbargd,tmp);
	    }
	    dgddvb = -model->MOS2gamma*(dbargs+dbargd);
	    PlusDeriv(&d_dgddvb,&d_dbargs,&d_dbargd);
	    TimesDeriv(&d_dgddvb,&d_dgddvb,-model->MOS2gamma);
	    if (model->MOS2junctionDepth > 0) {
	    }
	} else {
	    gamasd = model->MOS2gamma;
	    gammad = model->MOS2gamma;
	    EqualDeriv(&d_gamasd,&d_zero);
	    EqualDeriv(&d_gammad,&d_zero);
	    d_gamasd.value = d_gammad.value = model->MOS2gamma;
	    dgddvb = 0.0;
	    EqualDeriv(&d_dgddvb,&d_zero);
	}
	von = vbin+gamasd*sarg1;
	MultDeriv(&d_von,&d_gamasd,&d_sarg);
	PlusDeriv(&d_von,&d_von,&d_vbin);
	/*
	vth = von;
	EqualDeriv(&d_vth,&d_von);
	*/
	vdsat = 0.0;
	EqualDeriv(&d_vdsat,&d_zero);
	if (model->MOS2fastSurfaceStateDensity != 0.0 && OxideCap != 0.0) {
	    /* XXX constant per model */
	    cfs = CHARGE*model->MOS2fastSurfaceStateDensity*
		1e4 /*(cm**2/m**2)*/;
	    cdonco = -(gamasd*dsrgdb + dgddvb*sarg1) + factor;
	    MultDeriv(&d_dummy,&d_dgddvb,&d_sarg);
	    MultDeriv(&d_cdonco,&d_gamasd,&d_dsrgdb);
	    PlusDeriv(&d_cdonco,&d_cdonco,&d_dummy);
	    TimesDeriv(&d_cdonco,&d_cdonco,-1.0);
	    d_cdonco.value += factor;
	   xn = 1.0+cfs/OxideCap*here->MOS2m*here->MOS2w*EffectiveLength+cdonco;
	   EqualDeriv(&d_xn,&d_cdonco);
	   d_xn.value = xn;
	    tmp = vt*xn;
	    TimesDeriv(&d_tmp,&d_xn,vt);
	    von = von+tmp;
	    PlusDeriv(&d_von,&d_von,&d_tmp);
	    argg = 1.0/tmp;
	    InvDeriv(&d_argg,&d_tmp);
	    vgst = lvgs-von;
	    TimesDeriv(&d_vgst,&d_von,-1.0);
	    PlusDeriv(&d_vgst,&d_vgst,&d_p);
	    d_vgst.value += lvgs;
	} else {
	    vgst = lvgs-von;
	    TimesDeriv(&d_vgst,&d_von,-1.0);
	    PlusDeriv(&d_vgst,&d_vgst,&d_p);
	    d_vgst.value += lvgs;

	    if (lvgs <= von) {
		/*
		 *  cutoff region                     */
		here->MOS2gds = 0.0; /* look at this later */
		goto line1050;
	    }
	}

	/*
	 *  compute some more useful quantities             */

	sarg3 = sarg1*sarg1*sarg1;
	CubeDeriv(&d_sarg3,&d_sarg);
	/* XXX constant per model */
	sbiarg = sqrt(here->MOS2tBulkPot); /*const*/
	gammad = gamasd;
	EqualDeriv(&d_gammad,&d_gamasd);
	body = barg*barg*barg-sarg3;
	TimesDeriv(&d_body,&d_sarg3,-1.0);
	CubeDeriv(&d_dummy,&d_barg);
	PlusDeriv(&d_body,&d_body,&d_dummy);
	if (model->MOS2fastSurfaceStateDensity == 0.0) goto line400;
	if (OxideCap == 0.0) goto line410;
	/*
	 *  evaluate effective mobility and its derivatives             */
line400:
	if (OxideCap <= 0.0) goto line410;
	udenom = vgst;
	EqualDeriv(&d_udenom,&d_vgst);
	tmp = model->MOS2critField * 100 /* cm/m */ * EPSSIL/
	    model->MOS2oxideCapFactor;
	if (udenom <= tmp) goto line410;
	ufact = exp(model->MOS2critFieldExp*log(tmp/udenom));
	/* dummy = tmp/udenom */
	InvDeriv(&d_dummy,&d_udenom);
	TimesDeriv(&d_dummy,&d_dummy,tmp);
	PowDeriv(&d_ufact,&d_dummy,model->MOS2critFieldExp);
	ueff = model->MOS2surfaceMobility * 1e-4 /*(m**2/cm**2) */ *ufact;
	TimesDeriv(&d_ueff,&d_ufact,model->MOS2surfaceMobility * 1e-4);
	goto line500;
line410:
	ufact = 0.0;
	EqualDeriv(&d_ufact,&d_zero);
	ueff = model->MOS2surfaceMobility * 1e-4 /*(m**2/cm**2) */ ;
	EqualDeriv(&d_ueff,&d_zero);
	d_ueff.value = ueff;

	/*
	 *     evaluate saturation voltage and its derivatives according to
	 *     grove-frohman equation             */
line500:
	vgsx = lvgs;
	EqualDeriv(&d_vgsx,&d_p); d_vgsx.value = lvgs;
	gammad = gamasd/eta; /* eta is a constant */
	TimesDeriv(&d_gammad,&d_gamasd,1/eta);
	if (model->MOS2fastSurfaceStateDensity != 0 && OxideCap != 0) {
	    vgsx = MAX(lvgs,von);
	    /* mistake! fixed Dec 8 '89 if (vgsx < von) { } */
	 if (lvgs > von) {
	     EqualDeriv(&d_vgsx,&d_p);d_vgsx.value = lvgs;
	 } else {
	    EqualDeriv(&d_vgsx,&d_von);
	 }
	}

	if (gammad > 0) {
	    gammd2 = gammad*gammad;
	    MultDeriv(&d_gammd2,&d_gammad,&d_gammad);
	    argv = (vgsx-vbin)/eta+phiMinVbs;
	    TimesDeriv(&d_argv,&d_vbin,-1.0);
	    PlusDeriv(&d_argv,&d_vgsx,&d_vgsx);
	    TimesDeriv(&d_argv,&d_argv,1/eta);
	    PlusDeriv(&d_argv,&d_argv,&d_phiMinVbs);
	    if (argv <= 0.0) {
		vdsat = 0.0;
		EqualDeriv(&d_vdsat,&d_zero);
	    } else {
		arg1 = sqrt(1.0+4.0*argv/gammd2);
		DivDeriv(&d_arg,&d_argv,&d_gammd2);
		TimesDeriv(&d_arg,&d_arg,4.0);d_arg.value += 1.0;
		SqrtDeriv(&d_arg,&d_arg);
#if 0
		 dumarg= sqrt(gammd2*gammd2 +4*argv*gammd2);
		 TimesDeriv(&d_dumarg,&d_argv,4.0);
		 PlusDeriv(&d_dumarg,&d_dumarg,&d_gammd2);
		 MultDeriv(&d_dumarg,&d_dumarg,&d_gammd2);
		 SqrtDeriv(&d_dumarg,&d_dumarg);
		 
		vdsat = (vgsx-vbin)/eta+gammd2*(1.0-arg)/2.0;
		/* distortion vdsat=(vgsx-vbin)/eta + (gammd2 - dumarg)/2.0 
		   = argv - phiMinVbs + (gammd2 - dumarg)/2 */
#endif
		TimesDeriv(&d_dummy,&d_dumarg,-1.0);
		PlusDeriv(&d_dummy,&d_dummy,&d_gammd2);
		TimesDeriv(&d_dummy,&d_dummy,0.5);
		TimesDeriv(&d_vdsat,&d_phiMinVbs,-1.0);
		PlusDeriv(&d_vdsat,&d_vdsat,&d_argv);
		PlusDeriv(&d_vdsat,&d_dummy,&d_vdsat);
		vdsat = MAX(vdsat,0.0);
		if (vdsat < 0.0) {
		EqualDeriv(&d_vdsat,&d_zero);
	}
	    }
	} else {
	    vdsat = (vgsx-vbin)/eta;
	    TimesDeriv(&d_vdsat,&d_vbin,-1.0);
	    PlusDeriv(&d_vdsat,&d_vgsx,&d_vdsat);
	    TimesDeriv(&d_vdsat,&d_vdsat,1/eta);
	    vdsat = MAX(vdsat,0.0);
	    if (vdsat < 0.0) {
		EqualDeriv(&d_vdsat,&d_zero);
	    }
	}
	if (model->MOS2maxDriftVel > 0) {
	    /* 
	     *     evaluate saturation voltage and its derivatives 
	     *     according to baum's theory of scattering velocity 
	     *     saturation
	     */
	    gammd2 = gammad*gammad;
	    MultDeriv(&d_gammd2,&d_gammad,&d_gammad);
	    v1 = (vgsx-vbin)/eta+phiMinVbs;
	    TimesDeriv(&d_v1,&d_vbin,-1.0);
#if 0
	    /* mistake ! (fixed Dec 7 '89) thanks to Jean Hsu */
	    PlusDeriv(&d_v1,&d_vgsx,&d_vgsx);
#endif
	    PlusDeriv(&d_v1,&d_v1,&d_vgsx);
	    TimesDeriv(&d_v1,&d_v1,1/eta);
	    PlusDeriv(&d_v1,&d_v1,&d_phiMinVbs);
	    v2 = phiMinVbs;
	    EqualDeriv(&d_v2,&d_phiMinVbs);
	    xv = model->MOS2maxDriftVel*EffectiveLength/ueff;
	    InvDeriv(&d_xv,&d_ueff);
	    TimesDeriv(&d_xv,&d_xv,model->MOS2maxDriftVel*EffectiveLength);
	    a1 = gammad/0.75;
	    TimesDeriv(&d_a1,&d_gammad,4.0/3.0);
	    /* dummy1 = a1 */
	    b1 = -2.0*(v1+xv);
	    PlusDeriv(&d_b1,&d_v1,&d_xv);
	    TimesDeriv(&d_b1,&d_b1,-2.0);
                /* dummy2 = b1 */
                c1 = -2.0*gammad*xv;
		MultDeriv(&d_c1,&d_gammad,&d_xv);
		TimesDeriv(&d_c1,&d_c1,-2.0);
                /* dummy3 = c1 */
                d1 = 2.0*v1*(v2+xv)-v2*v2-4.0/3.0*gammad*sarg3;
		MultDeriv(&d_d1,&d_gammad,&d_sarg3);
		TimesDeriv(&d_d1,&d_d1,4.0/3.0);
		MultDeriv(&d_dummy,&d_v2,&d_v2);
		PlusDeriv(&d_d1,&d_d1,&d_dummy);
		TimesDeriv(&d_d1,&d_d1,-1.0);
		PlusDeriv(&d_dummy,&d_v2,&d_xv);
		MultDeriv(&d_dummy,&d_dummy,&d_v1);
		TimesDeriv(&d_dummy,&d_dummy,2.0);
		PlusDeriv(&d_d1,&d_d1,&d_dummy);
                a = -b1;
		TimesDeriv(&d_a,&d_b1,-1.0);
                b = a1*c1-4.0*d1;
		TimesDeriv(&d_b,&d_d1,-4.0);
		MultDeriv(&d_dummy,&d_a1,&d_c1);
		/* mistake! - fixed Dec 8 '89
		PlusDeriv(&d_d1,&d_d1,&d_dummy);
		*/
		PlusDeriv(&d_b,&d_b,&d_dummy);
                c = -d1*(a1*a1-4.0*b1)-c1*c1;
		TimesDeriv(&d_dummy,&d_b1,-4.0);
		MultDeriv(&d_c,&d_a1,&d_a1);
		PlusDeriv(&d_dummy,&d_dummy,&d_c);
		MultDeriv(&d_c,&d_dummy,&d_d1);
		MultDeriv(&d_dummy,&d_c1,&d_c1);
		PlusDeriv(&d_c,&d_c,&d_dummy);
		TimesDeriv(&d_c,&d_c,-1.0);

                r = -a*a/3.0+b;
		MultDeriv(&d_arr,&d_a,&d_a);
		TimesDeriv(&d_arr,&d_arr,-1.0/3.0);
		PlusDeriv(&d_arr,&d_arr,&d_b);

                s = 2.0*a*a*a/27.0-a*b/3.0+c;
		CubeDeriv(&d_s,&d_a);
		TimesDeriv(&d_s,&d_s,2.0/27.0);
		PlusDeriv(&d_s,&d_s,&d_c);
		MultDeriv(&d_dummy,&d_a,&d_b);
		TimesDeriv(&d_dummy,&d_dummy,-1.0/3.0);
		PlusDeriv(&d_s,&d_s,&d_dummy);

                r3 = r*r*r;
		CubeDeriv(&d_r3,&d_arr);

                s2 = s*s;
		MultDeriv(&d_s2,&d_s,&d_s);

                p = s2/4.0+r3/27.0;
		TimesDeriv(&d_dummy,&d_r3,1.0/27.0);
		TimesDeriv(&d_pee,&d_s2,0.25);
		PlusDeriv(&d_pee,&d_pee,&d_dummy);
                p0 = fabs(p);
		if (p < 0.0)
			/* mistake! fixed Dec 8 '89
			TimesDeriv(&d_pee,&d_pee, -1.0);
			*/
			TimesDeriv(&d_p0,&d_pee, -1.0);
                p2 = sqrt(p0);
		SqrtDeriv(&d_p2,&d_p0);
                if (p < 0) {
                    ro = sqrt(s2/4.0+p0);
                    ro = log(ro)/3.0;
                    ro = exp(ro);
                    /* the above is eqvt. to                    
			ro = (s2/4.0 + p0)^1/6; */
		    TimesDeriv(&d_ro,&d_s2,0.25);
		    PlusDeriv(&d_ro,&d_ro,&d_p0);
		    PowDeriv(&d_ro,&d_ro,1.0/6.0);
                    fi = atan(-2.0*p2/s);
		    DivDeriv(&d_fi,&d_p2,&d_s);
		    TimesDeriv(&d_fi,&d_fi,-2.0);
		    AtanDeriv(&d_fi,&d_fi);
                    y3 = 2.0*ro*cos(fi/3.0)-a/3.0;
		    TimesDeriv(&d_dummy,&d_fi,1.0/3.0);
		    CosDeriv(&d_dummy,&d_dummy);
		    MultDeriv(&d_y3,&d_ro,&d_dummy);
		    TimesDeriv(&d_y3,&d_y3,2.0);
		    /* mistake! fixed Dec 8 '89
		    TimesDeriv(&d_dummy,&d_a,-3.0);
		    */
		    TimesDeriv(&d_dummy,&d_a,-1/3.0);
		    PlusDeriv(&d_y3,&d_y3,&d_dummy);
                } else {
                    p3 = (-s/2.0+p2);
		    TimesDeriv(&d_p3,&d_s,-0.5);
		    PlusDeriv(&d_p3,&d_p3,&d_p2);
                    p3 = exp(log(fabs(p3))/3.0);
		    /* eqvt. to (fabs(p3)) ^ 1/3 */
		    if (p3 < 0.0)
			TimesDeriv(&d_p3,&d_p3,-1.0);
		    PowDeriv(&d_p3,&d_p3,1.0/3.0);
                    p4 = (-s/2.0-p2);
		    TimesDeriv(&d_p4,&d_s,0.5);
		    PlusDeriv(&d_p4,&d_p4,&d_p2);
		    if (p4 < 0.0)
		    TimesDeriv(&d_p4,&d_p4,-1.0); /* this is fabs(p4) */
                    p4 = exp(log(fabs(p4))/3.0);
		    PowDeriv(&d_p4,&d_p4,1.0/3.0);

                    y3 = p3+p4-a/3.0;
		    TimesDeriv(&d_y3,&d_a,-1.0/3.0);
		    PlusDeriv(&d_y3,&d_y3,&d_p4);
		    PlusDeriv(&d_y3,&d_y3,&d_p3);
                }
                iknt = 0;
                a3 = sqrt(a1*a1/4.0-b1+y3);
		MultDeriv(&d_a3,&d_a1,&d_a1);
		TimesDeriv(&d_a3,&d_a3,0.25);
		PlusDeriv(&d_a3,&d_a3,&d_y3);
		TimesDeriv(&d_dummy,&d_b1,-1.0);
		PlusDeriv(&d_a3,&d_a3,&d_dummy);
		SqrtDeriv(&d_a3,&d_a3);

                b3 = sqrt(y3*y3/4.0-d1);
		MultDeriv(&d_b3,&d_y3,&d_y3);
		TimesDeriv(&d_b3,&d_b3,0.25);
		TimesDeriv(&d_dummy,&d_d1,-1.0);
		PlusDeriv(&d_b3,&d_b3,&d_dummy);
		SqrtDeriv(&d_b3,&d_b3);

                for(i = 1;i<=4;i++) {
                    a4[i-1] = a1/2.0+sig1[i-1]*a3;
		    TimesDeriv(&d_a4[i-1],&d_a1,0.5);
		    TimesDeriv(&d_dummy,&d_a3,sig1[i-1]);
		    PlusDeriv(&d_a4[i-1],&d_a4[i-1],&d_dummy);
                    b4[i-1] = y3/2.0+sig2[i-1]*b3;
		    TimesDeriv(&d_b4[i-1],&d_y3,0.5);
		    TimesDeriv(&d_dummy,&d_b3,sig2[i-1]);
		    PlusDeriv(&d_b4[i-1],&d_b4[i-1],&d_dummy);
                    delta4 = a4[i-1]*a4[i-1]/4.0-b4[i-1];
		    MultDeriv(&d_delta4,&d_a4[i-1],&d_a4[i-1]);
		    TimesDeriv(&d_delta4,&d_delta4,0.25);
		    TimesDeriv(&d_dummy,&d_b4[i-1],-1.0);
		    PlusDeriv(&d_delta4,&d_delta4,&d_dummy);

                    if (delta4 < 0) continue;
                    iknt = iknt+1;
                    tmp = sqrt(delta4);
		    SqrtDeriv(&d_tmp,&d_delta4);
                    x4[iknt-1] = -a4[i-1]/2.0+tmp;
		    TimesDeriv(&d_x4[iknt-1],&d_a4[i-1],-0.5);
		    PlusDeriv(&d_x4[iknt-1],&d_x4[iknt-1],&d_tmp);
                    iknt = iknt+1;
                    x4[iknt-1] = -a4[i-1]/2.0-tmp;
		    TimesDeriv(&d_x4[iknt-1],&d_a4[i-1],-0.5);
		    PlusDeriv(&d_x4[iknt-1],&d_x4[iknt-1],&d_tmp);
                }
                jknt = 0;
                for(j = 1;j<=iknt;j++) {
                    if (x4[j-1] <= 0) continue;
                    /* XXX implement this sanely */
                    poly4[j-1] = x4[j-1]*x4[j-1]*x4[j-1]*x4[j-1]+a1*x4[j-1]*
                        x4[j-1]*x4[j-1];
		    CubeDeriv(&d_dummy,&d_x4[j-1]);
		    PlusDeriv(&d_poly4[j-1],&d_x4[j-1],&d_a1);
		    MultDeriv(&d_poly4[j-1],&d_poly4[j-1],&d_dummy);
                    poly4[j-1] = poly4[j-1]+b1*x4[j-1]*x4[j-1]+c1*x4[j-1]+d1;
		    PlusDeriv(&d_poly4[j-1],&d_poly4[j-1],&d_d1);
		    MultDeriv(&d_dummy,&d_b1,&d_x4[j-1]);
		    PlusDeriv(&d_dummy,&d_dummy,&d_c1);
		    MultDeriv(&d_dummy,&d_dummy,&d_x4[j-1]);
		    PlusDeriv(&d_poly4[j-1],&d_poly4[j-1],&d_dummy);
                    if (fabs(poly4[j-1]) > 1.0e-6) continue;
                    jknt = jknt+1;
                    if (jknt <= 1) {
                        xvalid = x4[j-1];
			EqualDeriv(&d_xvalid,&d_x4[j-1]);
                    }
                    if (x4[j-1] > xvalid) continue;
                    xvalid = x4[j-1];
		    EqualDeriv(&d_xvalid,&d_x4[j-1]);
                }
                if (jknt > 0) {
                    vdsat = xvalid*xvalid-phiMinVbs;
		    MultDeriv(&d_vdsat,&d_xvalid,&d_xvalid);
		    TimesDeriv(&d_dummy,&d_phiMinVbs,-1.0);
		    PlusDeriv(&d_vdsat,&d_vdsat,&d_dummy);
                }
            }
            /*
             *  evaluate effective channel length and its derivatives             */
            if (lvds != 0.0) {
                gammad = gamasd;
		EqualDeriv(&d_gammad,&d_gamasd);
                if ((lvbs-vdsat) <= 0) {
                    bsarg = sqrt(vdsat+phiMinVbs);
		    PlusDeriv(&d_bsarg,&d_vdsat,&d_phiMinVbs);
		    SqrtDeriv(&d_bsarg,&d_bsarg);

                } else {
                    bsarg = sphi/(1.0+0.5*(lvbs-vdsat)/here->MOS2tPhi);
		    TimesDeriv(&d_bsarg,&d_vdsat,-1.0);
		    d_bsarg.value += lvbs; d_bsarg.d1_r += 1.0;
		    TimesDeriv(&d_bsarg,&d_bsarg,0.5/here->MOS2tPhi);
		    d_bsarg.value += 1.0;
		    InvDeriv(&d_bsarg,&d_bsarg);
		    TimesDeriv(&d_bsarg,&d_bsarg,sphi);

                }
                bodys = bsarg*bsarg*bsarg-sarg3;
		CubeDeriv(&d_bodys,&d_bsarg);
		TimesDeriv(&d_dummy,&d_sarg3,-1.0);
		PlusDeriv(&d_bodys,&d_bodys,&d_dummy);
                if (model->MOS2maxDriftVel <= 0) {
                    if (model->MOS2substrateDoping == 0.0) goto line610;
                    if (xlamda > 0.0) goto line610;
                    argv = (lvds-vdsat)/4.0;
		    TimesDeriv(&d_argv,&d_vdsat,-1.0);
		    d_argv.value += lvds; d_argv.d1_r += 1.0;
		    TimesDeriv(&d_argv,&d_argv,0.25);

                    sargv = sqrt(1.0+argv*argv);
		    MultDeriv(&d_sargv,&d_argv,&d_argv);
		    d_sargv.value += 1.0;
		    SqrtDeriv(&d_sargv,&d_sargv);
                    arg1 = sqrt(argv+sargv);
		    PlusDeriv(&d_arg,&d_sargv,&d_argv);
		    SqrtDeriv(&d_arg,&d_arg);
                    xlfact = model->MOS2xd/(EffectiveLength*lvds);
		    EqualDeriv(&d_xlfact,&d_r); d_xlfact.value = lvds;
		    InvDeriv(&d_xlfact,&d_xlfact);
		    TimesDeriv(&d_xlfact,&d_xlfact,model->MOS2xd/EffectiveLength);
                    xlamda = xlfact*arg1;
		    MultDeriv(&d_xlamda,&d_xlfact,&d_arg);
                } else {
                    argv = (vgsx-vbin)/eta-vdsat;
		    TimesDeriv(&d_argv,&d_vbin,-1.0);
		    PlusDeriv(&d_argv,&d_argv,&d_vgsx);
		    TimesDeriv(&d_argv,&d_argv,1/eta);
		    TimesDeriv(&d_dummy,&d_vdsat,-1.0);
		    PlusDeriv(&d_argv,&d_argv,&d_dummy);
                    xdv = model->MOS2xd/sqrt(model->MOS2channelCharge); /*const*/
                    xlv = model->MOS2maxDriftVel*xdv/(2.0*ueff);
		    InvDeriv(&d_xlv,&d_ueff);
		    TimesDeriv(&d_xlv,&d_xlv,model->MOS2maxDriftVel*xdv*0.5);
		    /* retained for historical interest
                    vqchan = argv-gammad*bsarg;
		    MultDeriv(&d_vqchan,&d_gammad,&d_bsarg);
		    TimesDeriv(&d_vqchan,&d_vqchan,-1);
		    PlusDeriv(&d_vqchan,&d_vqchan,&d_argv);
		    */
                    /* gammad = gamasd 
                    vl = model->MOS2maxDriftVel*EffectiveLength;const*/
                    if (model->MOS2substrateDoping == 0.0) goto line610;
                    if (xlamda > 0.0) goto line610;
                    argv = lvds-vdsat;
		    TimesDeriv(&d_argv,&d_vdsat,-1.0);
		    d_argv.value += lvds;
		    d_argv.d1_r += 1.0;
		    if (argv < 0.0)
			EqualDeriv(&d_argv,&d_zero);
                    argv = MAX(argv,0.0);
                    xls = sqrt(xlv*xlv+argv);
		    MultDeriv(&d_xls,&d_xlv,&d_xlv);
		    PlusDeriv(&d_xls,&d_xls,&d_argv);
		    SqrtDeriv(&d_xls,&d_xls);
                    /* dummy9 = xlv*xlv + argv */
                    xlfact = xdv/(EffectiveLength*lvds);
		    EqualDeriv(&d_xlfact,&d_r);
		    d_xlfact.value += lvds;
		    InvDeriv(&d_xlfact,&d_xlfact);
		    TimesDeriv(&d_xlfact,&d_xlfact,xdv/EffectiveLength);
                    xlamda = xlfact*(xls-xlv);
		    TimesDeriv(&d_xlamda,&d_xlv,-1.0);
		    PlusDeriv(&d_xlamda,&d_xlamda,&d_xls);
		    MultDeriv(&d_xlamda,&d_xlamda,&d_xlfact);

                }
            }
line610:
            
            /*
             *     limit channel shortening at punch-through             */
            xwb = model->MOS2xd*sbiarg; /*const*/
            xld = EffectiveLength-xwb; /*const*/
            clfact = 1.0-xlamda*lvds;
	    EqualDeriv(&d_clfact,&d_r); d_clfact.value = lvds;
	    d_clfact.d1_r = -1;
	    MultDeriv(&d_clfact,&d_clfact,&d_xlamda);
	    d_clfact.value += 1.0;
            xleff = EffectiveLength*clfact;
	    TimesDeriv(&d_xleff,&d_clfact,EffectiveLength);
            deltal = xlamda*lvds*EffectiveLength;
	    EqualDeriv(&d_delta1,&d_r); 
	    d_delta1.value = EffectiveLength*lvds;
	    d_delta1.d1_r = EffectiveLength;
	    MultDeriv(&d_delta1,&d_delta1,&d_xlamda);


            if (model->MOS2substrateDoping == 0.0) xwb = 0.25e-6;
            if (xleff < xwb) {
                xleff = xwb/(1.0+(deltal-xld)/xwb);
		EqualDeriv(&d_xleff,&d_delta1);d_xleff.value -= xld;
		TimesDeriv(&d_xleff,&d_xleff,1/xwb);d_xleff.value += 1.0;
		InvDeriv(&d_xleff,&d_xleff);
		TimesDeriv(&d_xleff,&d_xleff,xwb);
                clfact = xleff/EffectiveLength;
		TimesDeriv(&d_clfact,&d_xleff,1/EffectiveLength);

 /*               dfact = xleff*xleff/(xwb*xwb); */
            }
            /*
             *  evaluate effective beta (effective kp)
             */
            beta1 = Beta*ufact/clfact;
	    DivDeriv(&d_beta1,&d_ufact,&d_clfact);
	    TimesDeriv(&d_beta1,&d_beta1,Beta);
            /*
             *  test for mode of operation and branch appropriately             */
            gammad = gamasd;
	    EqualDeriv(&d_gammad,&d_gamasd);
            if (lvds <= 1.0e-10) {
                if (lvgs <= von) {
                    if ((model->MOS2fastSurfaceStateDensity == 0.0) ||
                            (OxideCap == 0.0)) {
                        here->MOS2gds = 0.0;
                    d_cdrain.d1_q = 0.0;
                    d_cdrain.d2_q2 = 0.0;
                    d_cdrain.d3_q3 = 0.0;
                        goto line1050;
                    }

                    here->MOS2gds = beta1*(von-vbin-gammad*sarg1)*exp(argg*
                        (lvgs-von));
		    MultDeriv(&d_dummy,&d_gammad,&d_sarg);
		    PlusDeriv(&d_dummy,&d_dummy,&d_vbin);
		    TimesDeriv(&d_dummy,&d_dummy,-1.0);
		    PlusDeriv(&d_dummy,&d_dummy,&d_von);
		    MultDeriv(&d_mos2gds,&d_beta1,&d_dummy);
		    TimesDeriv(&d_dummy,&d_von,-1.0);
		    PlusDeriv(&d_dummy,&d_dummy,&d_p);
		    d_dummy.value += lvgs;
		    MultDeriv(&d_dummy,&d_dummy,&d_argg);
		    ExpDeriv(&d_dummy,&d_dummy);
		    MultDeriv(&d_mos2gds,&d_mos2gds,&d_dummy);
		    d_cdrain.d1_r = d_mos2gds.value;
		    d_cdrain.d2_r2 = d_mos2gds.d1_r;
		    d_cdrain.d3_r3 = d_mos2gds.d2_r2;
                        /* dummy1 = von - vbin - gamasd*sarg */
                    goto line1050;
                }


                here->MOS2gds = beta1*(lvgs-vbin-gammad*sarg1);
		MultDeriv(&d_mos2gds,&d_gammad,&d_sarg);
		PlusDeriv(&d_mos2gds,&d_mos2gds,&d_vbin);
		TimesDeriv(&d_mos2gds,&d_mos2gds,-1.0);
		MultDeriv(&d_mos2gds,&d_mos2gds,&d_beta1);
		    d_cdrain.d1_r = d_mos2gds.value;
		    d_cdrain.d2_r2 = d_mos2gds.d1_r;
		    d_cdrain.d3_r3 = d_mos2gds.d2_r2;

                goto line1050;
            }

            if (lvgs > von) goto line900;
            /*
             *  subthreshold region             */
            if (vdsat <= 0) {
                here->MOS2gds = 0.0;
		    d_cdrain.d1_r = 0.0;
		    d_cdrain.d2_r2 = 0.0;
		    d_cdrain.d3_r3 = 0.0;
                /* if (lvgs > vth) goto doneval; */
                goto line1050;
            } 
            vdson = MIN(vdsat,lvds);
	    if (vdsat <= lvds) {
		EqualDeriv(&d_vdson,&d_vdsat);
		} else {
		EqualDeriv(&d_vdson,&d_r);
		d_vdson.value = lvds;
		}
            if (lvds > vdsat) {
                barg = bsarg;
		EqualDeriv(&d_barg,&d_bsarg);
                body = bodys;
		EqualDeriv(&d_body,&d_bodys);
            }
            cdson = beta1*((von-vbin-eta*vdson*0.5)*vdson-gammad*body/1.5);
	    MultDeriv(&d_dummy,&d_gammad,&d_body);
	    TimesDeriv(&d_cdson,&d_dummy,-1/1.5);
	    TimesDeriv(&d_dummy,&d_vdson,0.5*eta);
	    PlusDeriv(&d_dummy,&d_dummy,&d_vbin);
	    TimesDeriv(&d_dummy,&d_dummy,-1.0);
	    PlusDeriv(&d_dummy,&d_dummy,&d_von);
	    MultDeriv(&d_dummy,&d_dummy,&d_vdson);
	    PlusDeriv(&d_dummy,&d_dummy,&d_cdson);
	    MultDeriv(&d_cdson,&d_dummy,&d_beta1);
            expg = exp(argg*(lvgs-von));
	    TimesDeriv(&d_expg,&d_von,-1.0);
	    d_expg.value += lvgs;
	    d_expg.d1_p += 1.0;
	    MultDeriv(&d_expg,&d_expg,&d_argg);
	    ExpDeriv(&d_expg,&d_expg);

            cdrain = cdson*expg;
	    MultDeriv(&d_cdrain,&d_cdson,&d_expg);
	    /*
            gmw = cdrain*argg;
            here->MOS2gm = gmw;
            tmp = gmw*(lvgs-von)/xn;
	    */
            goto doneval;

line900:
            if (lvds <= vdsat) {
                /*
                 *  linear region                 */
                cdrain = beta1*((lvgs-vbin-eta*lvds/2.0)*lvds-gammad*body/1.5);
		MultDeriv(&d_dummy,&d_gammad,&d_body);
		TimesDeriv(&d_dummy,&d_dummy,-1/1.5);
		EqualDeriv(&d_cdrain,&d_r);
		d_cdrain.value = eta*lvds*0.5;
		d_cdrain.d1_r = 0.5*eta;
		PlusDeriv(&d_cdrain,&d_cdrain,&d_vbin);
		TimesDeriv(&d_cdrain,&d_cdrain,-1.0);
		d_cdrain.value += lvgs;
		d_cdrain.d1_p += 1.0;
		EqualDeriv(&d_dummy,&d_r);
		d_dummy.value = lvds;
		MultDeriv(&d_cdrain,&d_cdrain,&d_dummy);
		MultDeriv(&d_dummy,&d_gammad,&d_body);
		TimesDeriv(&d_dummy,&d_dummy,-1/1.5);
		PlusDeriv(&d_cdrain,&d_cdrain,&d_dummy);
		MultDeriv(&d_cdrain,&d_cdrain,&d_beta1);
            } else {
                /* 
                 *  saturation region                 */
                cdrain = beta1*((lvgs-vbin-eta*
                    vdsat/2.0)*vdsat-gammad*bodys/1.5);
	    TimesDeriv(&d_cdrain,&d_vdsat,0.5*eta);
	    PlusDeriv(&d_cdrain,&d_cdrain,&d_vbin);
	    TimesDeriv(&d_cdrain,&d_cdrain,-1.0);
	    d_cdrain.value += lvgs;
	    d_cdrain.d1_p += 1.0;
	    MultDeriv(&d_cdrain,&d_cdrain,&d_vdsat);
	    MultDeriv(&d_dummy,&d_gammad,&d_bodys);
	    TimesDeriv(&d_dummy,&d_dummy,-1/1.5);
	    PlusDeriv(&d_cdrain,&d_cdrain,&d_dummy);
	    MultDeriv(&d_cdrain,&d_cdrain,&d_beta1);
            }
            /*
             *     compute charges for "on" region             */
            goto doneval;
            /*
             *  finish special cases             */
line1050:
            cdrain = 0.0;
            here->MOS2gm = 0.0;
            here->MOS2gmbs = 0.0;
d_cdrain.value = 0.0;
d_cdrain.d1_p = 0.0;
d_cdrain.d1_q = 0.0;
d_cdrain.d2_p2 = 0.0;
d_cdrain.d2_q2 = 0.0;
d_cdrain.d2_pq = 0.0;
d_cdrain.d2_qr = 0.0;
d_cdrain.d2_pr = 0.0;
d_cdrain.d3_p3 = 0.0;
d_cdrain.d3_q3 = 0.0;
d_cdrain.d3_p2r = 0.0;
d_cdrain.d3_p2q = 0.0;
d_cdrain.d3_q2r = 0.0;
d_cdrain.d3_pq2 = 0.0;
d_cdrain.d3_pr2 = 0.0;
d_cdrain.d3_qr2 = 0.0;
d_cdrain.d3_pqr = 0.0;
}

            /*
             *  finished             
	     */

/*================HERE=================*/
doneval:    
            /*
             *  COMPUTE EQUIVALENT DRAIN CURRENT SOURCE
             */
            /* 
             * now we do the hard part of the bulk-drain and bulk-source
	     * diode - we evaluate the non-linear capacitance and 
	     * charge                 
             * the basic equations are not hard, but the implementation
	     * is somewhat long in an attempt to avoid log/exponential
	     * evaluations                 
	     */
                /*
                 *  charge storage elements                 *
                 *.. bulk-drain and bulk-source depletion capacitances
		 */
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
		    lcapbs=here->MOS2Cbs*sarg+
                                here->MOS2Cbssw*sargsw;
		    lcapbs2 = model->MOS2type*0.5/here->MOS2tBulkPot*(
			here->MOS2Cbs*model->MOS2bulkJctBotGradingCoeff*
			sarg/arg + here->MOS2Cbssw*
			model->MOS2bulkJctSideGradingCoeff*sargsw/arg);
		    lcapbs3 = here->MOS2Cbs*sarg*
			model->MOS2bulkJctBotGradingCoeff*
			(model->MOS2bulkJctBotGradingCoeff+1);
		    lcapbs3 += here->MOS2Cbssw*sargsw*
			model->MOS2bulkJctSideGradingCoeff*
			(model->MOS2bulkJctSideGradingCoeff+1);
		    lcapbs3 = lcapbs3/(6*here->MOS2tBulkPot*
			here->MOS2tBulkPot*arg*arg);
                    } else {
                    /*    *(ckt->CKTstate0 + here->MOS2qbs)= here->MOS2f4s +
                                vbs*(here->MOS2f2s+vbs*(here->MOS2f3s/2));*/
                        lcapbs=here->MOS2f2s+here->MOS2f3s*vbs;
			lcapbs2 = 0.5*here->MOS2f3s;
			lcapbs3 = 0;
                    }
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
		    lcapbd=here->MOS2Cbd*sarg+
                                here->MOS2Cbdsw*sargsw;
		    lcapbd2 = model->MOS2type*0.5/here->MOS2tBulkPot*(
			here->MOS2Cbd*model->MOS2bulkJctBotGradingCoeff*
			sarg/arg + here->MOS2Cbdsw*
			model->MOS2bulkJctSideGradingCoeff*sargsw/arg);
		    lcapbd3 = here->MOS2Cbd*sarg*
			model->MOS2bulkJctBotGradingCoeff*
			(model->MOS2bulkJctBotGradingCoeff+1);
		    lcapbd3 += here->MOS2Cbdsw*sargsw*
			model->MOS2bulkJctSideGradingCoeff*
			(model->MOS2bulkJctSideGradingCoeff+1);
		    lcapbd3 = lcapbd3/(6*here->MOS2tBulkPot*
			here->MOS2tBulkPot*arg*arg);
                    } else {
                        lcapbd=here->MOS2f2d + vbd * here->MOS2f3d;
			lcapbd2=0.5*here->MOS2f3d;
			lcapbd3=0;
                    }
            /*
             *     meyer's capacitor model             */
	/*
	 * the meyer capacitance equations are in DEVqmeyer	 
	 * these expressions are derived from those equations.
	 * these expressions are incorrect; they assume just one
	 * controlling variable for each charge storage element	 
	 * while actually there are several;  the MOS2 small	 
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
    /* von, vgst and vdsat have already been adjusted for 
        possible source-drain interchange */



    phi = here->MOS2tPhi;
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
    } else  {			/* the MOS2modes are around because 
					vds has not been adjusted */
        if (vdsat <= here->MOS2mode*vds) {
	lcapgb2=lcapgb3=lcapgs2=lcapgs3=lcapgd2=lcapgd3=0;
        } else {
            vddif = 2.0*vdsat-here->MOS2mode*vds;
            vddif1 = vdsat-here->MOS2mode*vds/*-1.0e-12*/;
            vddif2 = vddif*vddif;
	    lcapgd2 = -vdsat*here->MOS2mode*vds*cox/(3*vddif*vddif2);
	    lcapgd3 = - here->MOS2mode*vds*cox*(vddif - 6*vdsat)/(9*vddif2*vddif2);
	    lcapgs2 = -vddif1*here->MOS2mode*vds*cox/(3*vddif*vddif2);
	    lcapgs3 = - here->MOS2mode*vds*cox*(vddif - 6*vddif1)/(9*vddif2*vddif2);
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
	here->capgb2 = model->MOS2type*lcapgb2;
	here->capgb3 = lcapgb3;
                /*
                 *   process to get Taylor coefficients, taking into		 * account type and mode.
                 */
gm2 = d_cdrain.d2_p2;
gb2 = d_cdrain.d2_q2;
gds2 = d_cdrain.d2_r2;
gmb = d_cdrain.d2_pq;
gbds = d_cdrain.d2_qr;
gmds = d_cdrain.d2_pr;
gm3 = d_cdrain.d3_p3;
gb3 = d_cdrain.d3_q3;
gds3 = d_cdrain.d3_r3;
gm2ds = d_cdrain.d3_p2r;
gm2b = d_cdrain.d3_p2q;
gb2ds = d_cdrain.d3_q2r;
gmb2 = d_cdrain.d3_pq2;
gmds2 = d_cdrain.d3_pr2;
gbds2 = d_cdrain.d3_qr2;
gmbds = d_cdrain.d3_pqr;

	if (here->MOS2mode == 1)
		{
		/* normal mode - no source-drain interchange */

 here->cdr_x2 = gm2;
 here->cdr_y2 = gb2;
 here->cdr_z2 = gds2;
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

		/* the gate caps have been divided and made into			Taylor coeffs., but not adjusted for type */

	here->capgs2 = model->MOS2type*lcapgs2;
	here->capgs3 = lcapgs3;
	here->capgd2 = model->MOS2type*lcapgd2;
	here->capgd3 = lcapgd3;
} else {
		/*
		 * inverse mode - source and drain interchanged		 */
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

          here->capgs2 = model->MOS2type*lcapgd2;
	  here->capgs3 = lcapgd3;

	  here->capgd2 = model->MOS2type*lcapgs2;
	  here->capgd3 = lcapgs3;

}

/* now to adjust for type and multiply by factors to convert to Taylor coeffs. */

here->cdr_x2 = 0.5*model->MOS2type*here->cdr_x2;
here->cdr_y2 = 0.5*model->MOS2type*here->cdr_y2;
here->cdr_z2 = 0.5*model->MOS2type*here->cdr_z2;
here->cdr_xy = model->MOS2type*here->cdr_xy;
here->cdr_yz = model->MOS2type*here->cdr_yz;
here->cdr_xz = model->MOS2type*here->cdr_xz;
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
