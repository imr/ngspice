/***********************************************************************
 HiSIM v1.1.0
 File: hsm1ld.c of HiSIM v1.1.0

 Copyright (C) 2002 STARC

 June 30, 2002: developed by Hiroshima University and STARC
 June 30, 2002: posted by Keiichi MORIKAWA, STARC Physical Design Group
***********************************************************************/

/*
 * Modified by Paolo Nenzi 2002
 * ngspice integration
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "hsm1def.h"
#include "hisim.h"
#include "trandefs.h"
#include "const.h"
#include "sperror.h"
#include "devdefs.h"
#include "suffix.h"

static void ShowPhysVals(HSM1instance *here, int flag, int isFirst, double vds,
             double vgs, double vbs)
{
  switch (flag) {
  case 1:
    if (isFirst) printf("Vds        Ids\n");
    printf("%e %e\n", vds, here->HSM1_ids);
    break;
  case 2:
    if (isFirst) printf("Vgs        Ids\n");
    printf("%e %e\n", vgs, here->HSM1_ids);
    break;
  case 3:
    if (isFirst) printf("vgs        ln(Ids)\n");
    printf("%e %e\n", vgs, log(here->HSM1_ids));
    break;
  case 4:
    if (isFirst) printf("ln(Ids)    gm/Ids\n");
    if (here->HSM1_ids == 0.0)
      printf("I can't show gm/Ids - ln(Ids), because Ids = 0.\n");
    else
      printf("%e %e\n",  log(here->HSM1_ids), here->HSM1_gm/here->HSM1_ids);
    break;
  case 5:
    if (isFirst) printf("Vds        gds\n");
    printf("%e %e\n", vds, here->HSM1_gds);
    break;
  case 6:
    if (isFirst) printf("Vgs        gm\n");
    printf("%e %e\n", vgs, here->HSM1_gm);
    break;
  case 7:
    if (isFirst) printf("Vbs        gbs\n");
    printf("%e %e\n", vbs, here->HSM1_gbs);
    break;
  case 8:
    if (isFirst) printf("Vgs        Cgg\n");
    printf("%e %e\n", vgs, here->HSM1_cggb);
    break;
  case 9:
    if (isFirst) printf("Vgs        Cgs\n");
    printf("%e %e\n", vgs, here->HSM1_cgsb);
    break;
  case 10:
    if (isFirst) printf("Vgs        Cgd\n");
    printf("%e %e\n", vgs, here->HSM1_cgdb);
    break;
  case 11:
    if (isFirst) printf("Vgs        Cgb\n");
    printf("%e %e\n", vgs, -(here->HSM1_cggb+here->HSM1_cgsb+here->HSM1_cgdb));
    break;
  case 12:
    if (isFirst) printf("Vds        Csg\n");
    printf("%e %e\n", vds, -(here->HSM1_cggb+here->HSM1_cbgb+here->HSM1_cdgb));
    break;
  case 13:
    if (isFirst) printf("Vds        Cdg\n");
    printf("%e %e\n", vds, here->HSM1_cdgb);
    break;
  case 14:
    if (isFirst) printf("Vds        Cbg\n");
    printf("%e %e\n", vds, here->HSM1_cbgb);
    break;
  deafult:
    /*
    printf("There is no physical vaule corrsponding to %d\n", flag);
    */
    break;
  }
}

int HSM1load(GENmodel *inModel, CKTcircuit *ckt)
     /* actually load the current value into the 
      * sparse matrix previously provided 
      */
{
  HSM1model *model = (HSM1model*)inModel;
  HSM1instance *here;
  HiSIM_input sIN;
  HiSIM_output sOT;
  HiSIM_messenger sMS;
  double cbhat, cdrain, cdhat, cdreq;
  double Ibtot, Idtot;
  double ceq, ceqbd, ceqbs, ceqqb, ceqqd, ceqqg;
  double delvbd, delvbs, delvds, delvgd, delvgs;
  double gcbdb, gcbgb, gcbsb, gcddb, gcdgb, gcdsb;
  double gcgdb, gcggb, gcgsb, gcsdb, gcsgb, gcssb;
  double geq, tol, xfact;
  double vbd, vbs, vcrit, vds, vgb, vgd, vgdo, vgs, von;
  double qgd, qgs, qgb;
  double gbbdp, gbbsp, gbspg, gbspdp, gbspb, gbspsp;
  double qgate, qbulk, qdrn, qsrc;
  double cqgate, cqbulk, cqdrn;
  double gbdpdp, gbdpg, gbdpb, gbdpsp; 
  double cgdo, cgso, cgbo;
  double gm, gmbs, FwdSum, RevSum;
  double vt0, ag0;
  int ByPass, Check, error;
#ifndef NOBYPASS
  double tempv;
#endif /*NOBYPASS*/
  double tmp;
  int ChargeComputationNeeded =  
    ((ckt->CKTmode & (MODEAC | MODETRAN | MODEINITSMSIG)) ||
     ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)))
    ? 1 : 0;
  int showPhysVal;
  
  double m;

  /*  loop through all the HSM1 device models */
  for ( ; model != NULL; model = model->HSM1nextModel ) {
    /* loop through all the instances of the model */
    for (here = model->HSM1instances; here != NULL ;
	 here = here->HSM1nextInstance) {
	 
      if (here->HSM1owner != ARCHme)
              continue;

	 
      showPhysVal = 0;
      Check=1;
      ByPass = 0;
      if ( ckt->CKTmode & MODEINITSMSIG ) {
	vbs= *(ckt->CKTstate0 + here->HSM1vbs);
	vgs= *(ckt->CKTstate0 + here->HSM1vgs);
	vds= *(ckt->CKTstate0 + here->HSM1vds);
      } 
      else if ( ckt->CKTmode & MODEINITTRAN ) {
	vbs= *(ckt->CKTstate1 + here->HSM1vbs);
	vgs= *(ckt->CKTstate1 + here->HSM1vgs);
	vds= *(ckt->CKTstate1 + here->HSM1vds);
      } 
      else if ( (ckt->CKTmode & MODEINITJCT) && !here->HSM1_off ) {
	vds= model->HSM1_type * here->HSM1_icVDS;
	vgs= model->HSM1_type * here->HSM1_icVGS;
	vbs= model->HSM1_type * here->HSM1_icVBS;
	if ( (vds == 0.0) && (vgs == 0.0) && (vbs == 0.0) && 
	     ( (ckt->CKTmode & (MODETRAN|MODEAC|MODEDCOP|MODEDCTRANCURVE)) ||
	       !(ckt->CKTmode & MODEUIC) ) ) { 
	  /* set biases for starting  analysis (same as BSIM3)*/
	  vbs = 0.0;
	  /*	  vt0 = model->HSM1_type * (-1.0 * model->HSM1_vfbc);*/
	  vt0 = model->HSM1_type * model->HSM1_vfbc;
	  vgs = vt0 + 0.1;
	  vds = 0.1;
	}
      } 
      else if ( ( ckt->CKTmode & (MODEINITJCT | MODEINITFIX) ) && 
		here->HSM1_off ) {
	vbs = vgs = vds = 0.0;
      } 
      else {
#ifndef PREDICTOR /* BSIM3 style */
	if(ckt->CKTmode & MODEINITPRED) {
	  xfact = ckt->CKTdelta / ckt->CKTdeltaOld[1];
	  *(ckt->CKTstate0 + here->HSM1vbs) = 
	    *(ckt->CKTstate1 + here->HSM1vbs);
	  vbs = (1.0 + xfact)* (*(ckt->CKTstate1 + here->HSM1vbs))
	    -(xfact * (*(ckt->CKTstate2 + here->HSM1vbs)));
	  *(ckt->CKTstate0 + here->HSM1vgs) = 
	    *(ckt->CKTstate1 + here->HSM1vgs);
	  vgs = (1.0 + xfact)* (*(ckt->CKTstate1 + here->HSM1vgs))
	    -(xfact * (*(ckt->CKTstate2 + here->HSM1vgs)));
	  *(ckt->CKTstate0 + here->HSM1vds) = 
	    *(ckt->CKTstate1 + here->HSM1vds);
	  vds = (1.0 + xfact)* (*(ckt->CKTstate1 + here->HSM1vds))
	    -(xfact * (*(ckt->CKTstate2 + here->HSM1vds)));
	  *(ckt->CKTstate0 + here->HSM1vbd) = 
	    *(ckt->CKTstate0 + here->HSM1vbs)-
	    *(ckt->CKTstate0 + here->HSM1vds);
	  showPhysVal = 1;
	} 
	else {
#endif /* PREDICTOR */
	  /* get biases from CKT */
	  vbs = model->HSM1_type * 
	    (*(ckt->CKTrhsOld+here->HSM1bNode) -
	     *(ckt->CKTrhsOld+here->HSM1sNodePrime));
	  vgs = model->HSM1_type * 
	    (*(ckt->CKTrhsOld+here->HSM1gNode) -
	     *(ckt->CKTrhsOld+here->HSM1sNodePrime));
	  vds = model->HSM1_type * 
	    (*(ckt->CKTrhsOld+here->HSM1dNodePrime) -
	     *(ckt->CKTrhsOld+here->HSM1sNodePrime));
	  showPhysVal = 1;
#ifndef PREDICTOR
	}
#endif /* PREDICTOR */
	vbd = vbs - vds;
	vgd = vgs - vds;
	vgdo = *(ckt->CKTstate0 + here->HSM1vgs) - 
	  *(ckt->CKTstate0 + here->HSM1vds);
	delvbs = vbs - *(ckt->CKTstate0 + here->HSM1vbs);
	delvbd = vbd - *(ckt->CKTstate0 + here->HSM1vbd);
	delvgs = vgs - *(ckt->CKTstate0 + here->HSM1vgs);
	delvds = vds - *(ckt->CKTstate0 + here->HSM1vds);
	delvgd = vgd - vgdo;
	  
	if (here->HSM1_mode >= 0) {
	  Idtot = here->HSM1_ids + here->HSM1_isub - here->HSM1_ibd;
	  cdhat = Idtot - here->HSM1_gbd * delvbd
	    + (here->HSM1_gmbs + here->HSM1_gbbs) * delvbs 
	    + (here->HSM1_gm + here->HSM1_gbgs) * delvgs
	    + (here->HSM1_gds + here->HSM1_gbds) * delvds;
	  Ibtot = here->HSM1_ibs + here->HSM1_ibd - here->HSM1_isub;
	  cbhat = Ibtot + here->HSM1_gbd * delvbd
	    + (here->HSM1_gbs -  here->HSM1_gbbs) * delvbs
	    - here->HSM1_gbgs * delvgs
	    - here->HSM1_gbds * delvds;
	} 
	else {
	  Idtot = here->HSM1_ids  - here->HSM1_ibd;
	  cdhat = Idtot - (here->HSM1_gbd - here->HSM1_gmbs) * delvbd
	    + here->HSM1_gm * delvgd
	    - here->HSM1_gds * delvds;
	  Ibtot = here->HSM1_ibs + here->HSM1_ibd - here->HSM1_isub;
	  cbhat = Ibtot + here->HSM1_gbs * delvbs
	    + (here->HSM1_gbd - here->HSM1_gbbs) * delvbd
	    - here->HSM1_gbgs * delvgd
	    + here->HSM1_gbds * delvds;
	}

#ifndef NOBYPASS /* BSIM3 style */
	/* now lets see if we can bypass (ugh) */
	
	/* following should be one big if connected by && all over
	 * the place, but some C compilers can't handle that, so
	 * we split it up here to let them digest it in stages
	 */
	if ( !(ckt->CKTmode & MODEINITPRED) && ckt->CKTbypass )
	  if ( fabs(delvbs) < 
	       ( ckt->CKTreltol * 
		 MAX(fabs(vbs), fabs(*(ckt->CKTstate0+here->HSM1vbs))) + 
		 ckt->CKTvoltTol ) )
	    if ( fabs(delvbd) < 
		 ( ckt->CKTreltol * 
		   MAX(fabs(vbd), fabs(*(ckt->CKTstate0+here->HSM1vbd))) + 
		   ckt->CKTvoltTol ) )
	      if ( fabs(delvgs) < 
		   ( ckt->CKTreltol * 
		     MAX(fabs(vgs), fabs(*(ckt->CKTstate0+here->HSM1vgs))) +
		     ckt->CKTvoltTol ) )
		if ( fabs(delvds) < 
		     ( ckt->CKTreltol * 
		       MAX(fabs(vds), fabs(*(ckt->CKTstate0+here->HSM1vds))) + 
		       ckt->CKTvoltTol ) )
		  if ( fabs(cdhat - Idtot) < 
		       ( ckt->CKTreltol * 
			 MAX(fabs(cdhat),fabs(Idtot)) + ckt->CKTabstol ) ) {
		    tempv = MAX(fabs(cbhat),fabs(Ibtot)) + ckt->CKTabstol;
		    if ((fabs(cbhat - Ibtot)) < ckt->CKTreltol * tempv) {
		      /* bypass code */
		      vbs = *(ckt->CKTstate0 + here->HSM1vbs);
		      vbd = *(ckt->CKTstate0 + here->HSM1vbd);
		      vgs = *(ckt->CKTstate0 + here->HSM1vgs);
		      vds = *(ckt->CKTstate0 + here->HSM1vds);
		      
		      vgd = vgs - vds;
		      vgb = vgs - vbs;
		      
		      cdrain = here->HSM1_ids;
		      if ((ckt->CKTmode & (MODETRAN | MODEAC)) || 
			  ((ckt->CKTmode & MODETRANOP) && 
			   (ckt->CKTmode & MODEUIC))) {
			ByPass = 1;
			qgate = here->HSM1_qg;
			qbulk = here->HSM1_qb;
			qdrn = here->HSM1_qd;
			goto line755;
			}
		      else
			goto line850;
		    }
		  }
#endif /*NOBYPASS*/

	/*	von = model->HSM1_type * here->HSM1_von;*/
	von = here->HSM1_von; 
	if(*(ckt->CKTstate0 + here->HSM1vds) >= 0.0) {
	  vgs = DEVfetlim(vgs, *(ckt->CKTstate0 + here->HSM1vgs), von);
	  vds = vgs - vgd;
	  vds = DEVlimvds(vds, *(ckt->CKTstate0 + here->HSM1vds));
	  vgd = vgs - vds;
	} 
	else {
	  vgd = DEVfetlim(vgd, vgdo, von);
	  vds = vgs - vgd;
	  vds = -DEVlimvds(-vds, -(*(ckt->CKTstate0 + here->HSM1vds)));
	  vgs = vgd + vds;
	}
	if (vds >= 0.0) {
	  vcrit = CONSTvt0 * log( CONSTvt0 / (CONSTroot2 * 1.0e-14) );
	  /* SourceSatCurrent = 1.0e-14 */
	  vbs = DEVpnjlim(vbs,*(ckt->CKTstate0 + here->HSM1vbs),
			  CONSTvt0,vcrit,&Check);
	  vbd = vbs - vds;
	} 
	else {
	  vcrit = CONSTvt0 * log( CONSTvt0 / (CONSTroot2 * 1.0e-14) );
	  /* DrainSatCurrent = 1.0e-14 */
	  vbd = DEVpnjlim(vbd,*(ckt->CKTstate0 + here->HSM1vbd),
			  CONSTvt0,vcrit,&Check);
	  vbs = vbd + vds;
	}
      }
      
      vbd = vbs - vds;
      vgd = vgs - vds;
      vgb = vgs - vbs;

      /* Input data for HiSIM evaluation part */
      sIN.gmin = ckt->CKTgmin; /* minimal conductance */

      if (vds >= 0) { /* normal mode */
	here->HSM1_mode = 1;
	sIN.vds = vds;
	sIN.vgs = vgs;
	sIN.vbs = vbs;
      }
      else { /* inverse mode */
	here->HSM1_mode = -1;
	sIN.vds = -vds;
	sIN.vgs = vgd;
	sIN.vbs = vbd;
      }

      sIN.type = model->HSM1_type;
      sIN.mode = here->HSM1_mode;
      /*      sIN.qflag = ChargeComputationNeeded; no use */
      sIN.info = model->HSM1_info;
      
      sIN.xl = here->HSM1_l;
      sIN.xw = here->HSM1_w;
      
      sIN.ad = here->HSM1_ad;
      sIN.as = here->HSM1_as;
      
      sIN.pd = here->HSM1_pd;
      sIN.ps = here->HSM1_ps;
      
      sIN.nrd = here->HSM1_nrd;
      sIN.nrs = here->HSM1_nrs;
      
      sIN.temp = ckt->CKTtemp;
      if ( here->HSM1_temp_Given ) sIN.temp = here->HSM1_temp;
      if ( here->HSM1_dtemp_Given ) sIN.temp = ckt->CKTtemp + here->HSM1_dtemp;
      
      sIN.tox = model->HSM1_tox;
      sIN.xld = model->HSM1_dl;
      sIN.xwd = model->HSM1_dw;
      
      if (model->HSM1_version == 100) { /* HiSIM 1.0.0 */
	sIN.xj = model->HSM1_xj;
      }
      else if (model->HSM1_version == 110) {	/* HiSIM 1.1.0 */
	sIN.xqy = model->HSM1_xqy;
      }
      sIN.vmax = model->HSM1_vmax;
      sIN.bgtmp1 = model->HSM1_bgtmp1;
      sIN.bgtmp2 = model->HSM1_bgtmp2;
      sIN.rs = model->HSM1_rs;
      sIN.rd = model->HSM1_rd;
      sIN.vfbc = model->HSM1_vfbc;
      sIN.nsubc = model->HSM1_nsubc;
      sIN.lp = model->HSM1_lp;
      sIN.nsubp = model->HSM1_nsubp;
      sIN.scp1 = model->HSM1_scp1;
      sIN.scp2 = model->HSM1_scp2;
      sIN.scp3 = model->HSM1_scp3;    
      sIN.parl1 = model->HSM1_parl1;
      sIN.parl2 = model->HSM1_parl2;
      sIN.sc1 = model->HSM1_sc1;
      sIN.sc2 = model->HSM1_sc2;
      sIN.sc3 = model->HSM1_sc3;
      sIN.pgd1 = model->HSM1_pgd1;
      sIN.pgd2 = model->HSM1_pgd2;
      sIN.pgd3 = model->HSM1_pgd3;
      sIN.ndep = model->HSM1_ndep;
      sIN.ninv = model->HSM1_ninv;
      sIN.ninvd = model->HSM1_ninvd;
      sIN.muecb0 = model->HSM1_muecb0;
      sIN.muecb1 = model->HSM1_muecb1;
      sIN.mueph1 = model->HSM1_mueph1;
      sIN.mueph0 = model->HSM1_mueph0;
      sIN.mueph2 = model->HSM1_mueph2;
      sIN.w0 = model->HSM1_w0;
      sIN.muesr1 = model->HSM1_muesr1;
      sIN.muesr0 = model->HSM1_muesr0;
      sIN.muetmp = model->HSM1_muetmp;
      if (model->HSM1_version == 110) { /* HiSIM 1.1.0 */
	sIN.wvthsc = model->HSM1_wvthsc;
	sIN.nsti = model->HSM1_nsti;
	sIN.wsti = model->HSM1_wsti;
      }
      
      if ( model->HSM1_bb_Given ) sIN.bb = model->HSM1_bb;
      else
	if ( model->HSM1_type == NMOS ) sIN.bb = 2.0e0;
	else sIN.bb = 1.0e0;
      
      sIN.vds0 = model->HSM1_vds0;
      sIN.bc0 = model->HSM1_bc0; 
      sIN.bc1 = model->HSM1_bc1;
      sIN.sub1 = model->HSM1_sub1;
      sIN.sub2 = model->HSM1_sub2;
      sIN.sub3 = model->HSM1_sub3;
     
      if ( model->HSM1_cgso_Given ) sIN.cocgso = 1;
      else sIN.cocgso = 0;

      if ( model->HSM1_cgdo_Given ) sIN.cocgdo = 1;
      else sIN.cocgdo = 0;

      if ( model->HSM1_cgbo_Given ) sIN.cocgbo = 1;
      else sIN.cocgbo = 0;
      
      if ( here->HSM1_mode == 1 ) {
	sIN.cgso   = model->HSM1_cgso;
	sIN.cgdo   = model->HSM1_cgdo;
      } 
      else {
	tmp = sIN.cocgso;
	sIN.cocgso = sIN.cocgdo;
	sIN.cocgdo = tmp;
	sIN.cgso   = model->HSM1_cgdo;
	sIN.cgdo   = model->HSM1_cgso;
      }
      
      sIN.cgbo = model->HSM1_cgbo;
      sIN.tpoly = model->HSM1_tpoly;
      sIN.js0 = model->HSM1_js0;
      sIN.js0sw = model->HSM1_js0sw;
      sIN.nj = model->HSM1_nj;
      sIN.njsw = model->HSM1_njsw;    
      sIN.xti = model->HSM1_xti;
      sIN.cj = model->HSM1_cj; 
      sIN.cjsw = model->HSM1_cjsw;
      sIN.cjswg = model->HSM1_cjswg;
      sIN.mj = model->HSM1_mj;
      sIN.mjsw = model->HSM1_mjsw;
      sIN.mjswg = model->HSM1_mjswg;
      sIN.pb = model->HSM1_pb;
      sIN.pbsw = model->HSM1_pbsw;
      sIN.pbswg = model->HSM1_pbswg;
      sIN.xpolyd = model->HSM1_xpolyd;
      sIN.clm1 = model->HSM1_clm1;
      sIN.clm2 = model->HSM1_clm2;
      sIN.clm3 = model->HSM1_clm3;
      sIN.rpock1 = model->HSM1_rpock1;
      sIN.rpock2 = model->HSM1_rpock2;
      if (model->HSM1_version == 110) { /* HiSIM 1.1.0 */
	sIN.rpocp1 = model->HSM1_rpocp1;
	sIN.rpocp2 = model->HSM1_rpocp2;
      }
      sIN.vover = model->HSM1_vover;
      sIN.voverp = model->HSM1_voverp;
      sIN.wfc = model->HSM1_wfc;
      sIN.qme1 = model->HSM1_qme1;
      sIN.qme2 = model->HSM1_qme2;
      sIN.qme3 = model->HSM1_qme3;
      sIN.gidl1 = model->HSM1_gidl1;
      sIN.gidl2 = model->HSM1_gidl2;
      sIN.gidl3 = model->HSM1_gidl3;
      sIN.gleak1 = model->HSM1_gleak1;
      sIN.gleak2 = model->HSM1_gleak2;
      sIN.gleak3 = model->HSM1_gleak3;
      sIN.vzadd0 = model->HSM1_vzadd0;
      sIN.pzadd0 = model->HSM1_pzadd0;
      
      /*  no use in SPICE3f5 */
      /*
      sMS.ims[0] = 0;
      sMS.dms[0] = 0.0e0;
      sMS.ims[1] = 0; */
      /*      sMS.dms[1] = pslot->timepoint; I don't know 
      * no use in SPICE3f5 */
      
      if ( ! strcmp(here->HSM1_called, "yes" ) ) {
	sIN.has_prv = 1;
      } 
      else {
	sIN.has_prv = 0;
	strcpy( here->HSM1_called , "yes" ) ;
      }

      sIN.corsrd = model->HSM1_corsrd;
      sIN.coiprv = model->HSM1_coiprv;
      sIN.copprv = model->HSM1_copprv;
      sIN.cocgbo = model->HSM1_cocgbo;
      sIN.coadov = model->HSM1_coadov;
      sIN.coxx08 = model->HSM1_coxx08;
      sIN.coxx09 = model->HSM1_coxx09;
      sIN.coisub = model->HSM1_coisub;
      sIN.coiigs = model->HSM1_coiigs;
      sIN.cogidl = model->HSM1_cogidl;
      sIN.coovlp = model->HSM1_coovlp;
      sIN.conois = model->HSM1_conois;
      if (model->HSM1_version == 110) { /* HiSIM 1.1.0 */
	sIN.coisti = model->HSM1_coisti;
      }

      sIN.vbsc_prv = here->HSM1_vbsc_prv;
      sIN.vdsc_prv = here->HSM1_vdsc_prv;
      sIN.vgsc_prv = here->HSM1_vgsc_prv;
      sIN.ps0_prv = here->HSM1_ps0_prv;
      sIN.ps0_dvbs_prv = here->HSM1_ps0_dvbs_prv;
      sIN.ps0_dvds_prv = here->HSM1_ps0_dvds_prv;
      sIN.ps0_dvgs_prv = here->HSM1_ps0_dvgs_prv;
      sIN.pds_prv = here->HSM1_pds_prv;
      sIN.pds_dvbs_prv = here->HSM1_pds_dvbs_prv;
      sIN.pds_dvds_prv = here->HSM1_pds_dvds_prv;
      sIN.pds_dvgs_prv = here->HSM1_pds_dvgs_prv;
      sIN.ids_prv = here->HSM1_ids_prv;
      sIN.ids_dvbs_prv = here->HSM1_ids_dvbs_prv;
      sIN.ids_dvds_prv = here->HSM1_ids_dvds_prv;
      sIN.ids_dvgs_prv = here->HSM1_ids_dvgs_prv;
      
      sIN.nftrp = model->HSM1_nftrp;
      sIN.nfalp = model->HSM1_nfalp;
      sIN.cit = model->HSM1_cit;

      if ( sIN.info >= 5 ) { /* mode, bias conditions ... */
	printf( "--- variables given to HSM1evaluate() ----\n" );
	printf( "type   = %s\n" , (sIN.type>0) ? "NMOS" : "PMOS" );
	printf( "mode   = %s\n" , (sIN.mode>0) ? "NORMAL" : "REVERSE" );
	
	printf( "vbs    = %12.5e\n" , sIN.vbs );
	printf( "vds    = %12.5e\n" , sIN.vds );
	printf( "vgs    = %12.5e\n" , sIN.vgs );
      }
      if ( sIN.info >= 6 ) { /* input flags */
	printf( "corsrd = %s\n" , (sIN.corsrd) ? "true" : "false" ) ;
	printf( "coiprv = %s\n" , (sIN.coiprv) ? "true" : "false" ) ;
	printf( "copprv = %s\n" , (sIN.copprv) ? "true" : "false" ) ;
	printf( "cocgso = %s\n" , (sIN.cocgso) ? "true" : "false" ) ;
	printf( "cocgdo = %s\n" , (sIN.cocgdo) ? "true" : "false" ) ;
	printf( "cocgbo = %s\n" , (sIN.cocgbo) ? "true" : "false" ) ;
	printf( "coadov = %s\n" , (sIN.coadov) ? "true" : "false" ) ;
	printf( "coisub = %s\n" , (sIN.coisub) ? "true" : "false" ) ;
	printf( "coiigs = %s\n" , (sIN.coiigs) ? "true" : "false" ) ;
	printf( "cogidl = %s\n" , (sIN.cogidl) ? "true" : "false" ) ;
	printf( "coovlp = %s\n" , (sIN.coovlp) ? "true" : "false" ) ;
	printf( "conois = %s\n" , (sIN.conois) ? "true" : "false" ) ;
	if (model->HSM1_version == 110) { /* HiSIM 1.1.0 */
	  printf( "coisti = %s\n" , (sIN.coisti) ? "true" : "false" ) ;
	}
      }
      /* print inputs ------------AA */

      /* call model evaluation */
      if (model->HSM1_version == 100) { /* HiSIM 1.0.0 */
	if ( HSM1evaluate1_0(sIN, &sOT, &sMS) == HiSIM_ERROR ) 
	  return (HiSIM_ERROR);
      }
      else if (model->HSM1_version == 110) { /* HiSIM 1.1.0 */
	if ( HSM1evaluate1_1(sIN, &sOT, &sMS) == HiSIM_ERROR ) 
	  return (HiSIM_ERROR);
      }
      
      /* store values for next calculation */
      /* note: derivatives are ones w.r.t. internal biases */
      here->HSM1_vbsc_prv = sOT.vbsc ;
      here->HSM1_vdsc_prv = sOT.vdsc ;
      here->HSM1_vgsc_prv = sOT.vgsc ;
      here->HSM1_ps0_prv = sOT.ps0 ;
      here->HSM1_ps0_dvbs_prv = sOT.ps0_dvbs ;
      here->HSM1_ps0_dvds_prv = sOT.ps0_dvds ;
      here->HSM1_ps0_dvgs_prv = sOT.ps0_dvgs ;
      here->HSM1_pds_prv = sOT.pds ;
      here->HSM1_pds_dvbs_prv = sOT.pds_dvbs ;
      here->HSM1_pds_dvds_prv = sOT.pds_dvds ;
      here->HSM1_pds_dvgs_prv = sOT.pds_dvgs ;
      here->HSM1_ids_prv = sOT.ids ;
      here->HSM1_ids_dvbs_prv = sOT.ids_dvbs ;
      here->HSM1_ids_dvds_prv = sOT.ids_dvds ;
      here->HSM1_ids_dvgs_prv = sOT.ids_dvgs ;

      /* for noise calc */
      here->HSM1_nfc = sOT.nf ;
      
      /* outputs */
      here->HSM1_gd = sOT.gd ; /* drain conductance */
      here->HSM1_gs = sOT.gs ; /* source conductance */

      here->HSM1_von = model->HSM1_type * sOT.von;
      here->HSM1_vdsat = model->HSM1_type * sOT.vdsat;
      /*
      here->HSM1_von = sOT.von;
      here->HSM1_vdsat = sOT.vdsat;
      */
      cdrain = here->HSM1_ids = sOT.ids ; /* cdrain */
      
      here->HSM1_gds = sOT.gds ;
      here->HSM1_gm = sOT.gm ;
      here->HSM1_gmbs = sOT.gmbs ;

      /* overlap capacitances */
      here->HSM1_cgso = sOT.cgso ; /* G-S */
      here->HSM1_cgdo = sOT.cgdo ; /* G-D */
      here->HSM1_cgbo = sOT.cgbo ; /* G-B */
   
      /* capop */
      /*      pslot->capop  = 13 ; capacitor selector ??? 
       * no use in SPICE3f5 */
      
      /* Meyer's capacitances */
      /*
      here->HSM1_capgs = sOT.capgs ;
      here->HSM1_capgd = sOT.capgd ;
      here->HSM1_capgb  = sOT.capgb ; may be not necessary */

      /* intrinsic charge ONLY */
      qgate = here->HSM1_qg = sOT.qg ; /* gate */
      qdrn = here->HSM1_qd = sOT.qd ;  /* drain */
      /*      here->HSM1_qs = sOT.qs ; source */
      qbulk = here->HSM1_qb = -1.0 * ( sOT.qg + sOT.qd + sOT.qs ); /* bulk */

      /* charge caps (intrisic ONLY) */
      here->HSM1_cggb = sOT.cggb ;
      here->HSM1_cgsb = sOT.cgsb ;
      here->HSM1_cgdb = sOT.cgdb ;
      here->HSM1_cbgb = sOT.cbgb ;
      here->HSM1_cbsb = sOT.cbsb ;
      here->HSM1_cbdb = sOT.cbdb ;
      here->HSM1_cdgb = sOT.cdgb ;
      here->HSM1_cdsb = sOT.cdsb ;
      here->HSM1_cddb = sOT.cddb ;

      /* substrate diode */
      here->HSM1_ibd = sOT.ibd ;
      here->HSM1_ibs = sOT.ibs ;
      /*
      here->HSM1_ibd = model->HSM1_type * sOT.ibd ;
      here->HSM1_ibs = model->HSM1_type * sOT.ibs ;
      */
      here->HSM1_gbd = sOT.gbd ;
      here->HSM1_gbs = sOT.gbs ;
      here->HSM1_capbd = sOT.capbd ;
      here->HSM1_capbs = sOT.capbs ;
      *(ckt->CKTstate0 + here->HSM1qbd) = sOT.qbd ;
      *(ckt->CKTstate0 + here->HSM1qbs) = sOT.qbs ;

      /* substrate impact ionization current */
      here->HSM1_isub = sOT.isub ;
      here->HSM1_gbgs = sOT.gbgs ;
      here->HSM1_gbds = sOT.gbds ;
      here->HSM1_gbbs = sOT.gbbs ;

      /* 1/f noise */
      /*      here->HSM1_nois_idsfl = sOT.nois_idsfl ;*/

      /* mobility added by K.M. */
      here->HSM1_mu = sOT.mu;

      /* print all outputs ------------VV */
      if ( sIN.info >= 4 ) {
	printf( "--- variables returned from HSM1evaluate() ----\n" ) ;
	printf( "gd     = %12.5e\n" , sOT.gd ) ;
	printf( "gs     = %12.5e\n" , sOT.gs ) ;
	
	printf( "von    = %12.5e\n" , sOT.von ) ;
	printf( "vdsat  = %12.5e\n" , sOT.vdsat ) ;
	printf( "ids    = %12.5e\n" , sOT.ids ) ;
	
	printf( "gds    = %12.5e\n" , sOT.gds ) ;
	printf( "gm     = %12.5e\n" , sOT.gm ) ;
	printf( "gmbs   = %12.5e\n" , sOT.gmbs ) ;
	
	printf( "cgbo   = %12.5e\n" , sOT.cgbo ) ;
	
	printf( "capgs  = %12.5e\n" , sOT.capgs ) ;
	printf( "capgd  = %12.5e\n" , sOT.capgd ) ;
	printf( "capgb  = %12.5e\n" , sOT.capgb ) ;
      
	printf( "qg     = %12.5e\n" , sOT.qg ) ;
	printf( "qd     = %12.5e\n" , sOT.qd ) ;
	printf( "qs     = %12.5e\n" , sOT.qs ) ;
	
	printf( "cggb   = %12.5e\n" , sOT.cggb ) ;
	printf( "cgsb   = %12.5e\n" , sOT.cgsb ) ;
	printf( "cgdb   = %12.5e\n" , sOT.cgdb ) ;
	printf( "cbgb   = %12.5e\n" , sOT.cbgb ) ;
	printf( "cbsb   = %12.5e\n" , sOT.cbsb ) ;
	printf( "cbdb   = %12.5e\n" , sOT.cbdb ) ;
	printf( "cdgb   = %12.5e\n" , sOT.cdgb ) ;
	printf( "cdsb   = %12.5e\n" , sOT.cdsb ) ;
	printf( "cddb   = %12.5e\n" , sOT.cddb ) ;
      
	printf( "ibd    = %12.5e\n" , sOT.ibd ) ;
	printf( "ibs    = %12.5e\n" , sOT.ibs ) ;
	printf( "gbd    = %12.5e\n" , sOT.gbd ) ;
	printf( "gbs    = %12.5e\n" , sOT.gbs ) ;
	printf( "capbd  = %12.5e\n" , sOT.capbd ) ;
	printf( "capbs  = %12.5e\n" , sOT.capbs ) ;
	printf( "qbd    = %12.5e\n" , sOT.qbd ) ;
	printf( "qbs    = %12.5e\n" , sOT.qbs ) ;

	printf( "isub   = %12.5e\n" , sOT.isub ) ;
	printf( "gbgs   = %12.5e\n" , sOT.gbgs ) ;
	printf( "gbds   = %12.5e\n" , sOT.gbds ) ;
	printf( "gbbs   = %12.5e\n" , sOT.gbbs ) ;
      
	/* 	printf( "flicker noise   = %12.5e\n" , sOT.nois_idsfl ) ;*/
 	printf( "flicker noise   = %12.5e\n" , sOT.nf ) ;
      }
      /* print all outputs ------------AA */

      if ( sIN.info >= 3 ) { /* physical valiables vs bias */
	static int isFirst = 1;
	if (isFirst) {
	  printf("# vbs vds vgs cggb cgdb cgsb cbgb cbdb cbsb cdgb cddb cdsb, Ps0 Psl\n");
	  isFirst = 0;
	}
	printf("%12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e\n", 
	       vbs, vds, vgs , 
	       sOT.cggb, sOT.cgdb, sOT.cgsb, 
	       sOT.cbgb, sOT.cbdb, sOT.cbsb,
	       sOT.cdgb, sOT.cddb, sOT.cdsb, 
	       sOT.ps0, sOT.ps0 + sOT.pds);
	  
      }

      /*
       *  check convergence
       */
      if ( (here->HSM1_off == 0) || !(ckt->CKTmode & MODEINITFIX) ) {
	if (Check == 1) {
	  ckt->CKTnoncon++;
      }
    }
    *(ckt->CKTstate0 + here->HSM1vbs) = vbs;
    *(ckt->CKTstate0 + here->HSM1vbd) = vbd;
    *(ckt->CKTstate0 + here->HSM1vgs) = vgs;
    *(ckt->CKTstate0 + here->HSM1vds) = vds;

    if (model->HSM1_show_Given && showPhysVal) {
      static int isFirst = 1;
      ShowPhysVals(here, model->HSM1_show, isFirst, vds, vgs, vbs);
      if (isFirst) isFirst = 0;
    }

    /* bulk and channel charge plus overlaps */
    
    if (!ChargeComputationNeeded) goto line850; 
    
 line755:
    cgdo = here->HSM1_cgdo;
    cgso = here->HSM1_cgso;
    cgbo = here->HSM1_cgbo;

    ag0 = ckt->CKTag[0];
    if (here->HSM1_mode > 0) { /* NORAMAL mode */
      gcggb = (here->HSM1_cggb + cgdo + cgso + cgbo ) * ag0;
      gcgdb = (here->HSM1_cgdb - cgdo) * ag0;
      gcgsb = (here->HSM1_cgsb - cgso) * ag0;

      gcdgb = (here->HSM1_cdgb - cgdo) * ag0;
      gcddb = (here->HSM1_cddb + here->HSM1_capbd + cgdo) * ag0;
      gcdsb = here->HSM1_cdsb * ag0;

      gcsgb = -(here->HSM1_cggb + here->HSM1_cbgb
		+ here->HSM1_cdgb + cgso) * ag0;
      gcsdb = -(here->HSM1_cgdb + here->HSM1_cbdb
		+ here->HSM1_cddb) * ag0;
      gcssb = (here->HSM1_capbs + cgso - 
	       (here->HSM1_cgsb + here->HSM1_cbsb + here->HSM1_cdsb)) * ag0;

      gcbgb = (here->HSM1_cbgb - cgbo) * ag0;
      gcbdb = (here->HSM1_cbdb - here->HSM1_capbd) * ag0;
      gcbsb = (here->HSM1_cbsb - here->HSM1_capbs) * ag0;

      if (sIN.coadov != 1) {
	qgd = cgdo * vgd;
	qgs = cgso * vgs;
	qgb = cgbo * vgb;
	qgate += qgd + qgs + qgb;
	qbulk -= qgb;
	qdrn -= qgd;
	/*      qsrc = -(qgate + qbulk + qdrn); */
      }
    }
    else { /* REVERSE mode */
      gcggb = (here->HSM1_cggb + cgdo + cgso + cgbo ) * ag0;
      gcgdb = (here->HSM1_cgsb - cgdo) * ag0;
      gcgsb = (here->HSM1_cgdb - cgso) * ag0;
      
      gcdgb = -(here->HSM1_cggb + 
		here->HSM1_cbgb + here->HSM1_cdgb + cgdo) * ag0;
      gcddb = (here->HSM1_capbd + cgdo - 
	       (here->HSM1_cgsb + here->HSM1_cbsb + here->HSM1_cdsb)) * ag0;
      gcdsb = -(here->HSM1_cgdb + here->HSM1_cbdb + here->HSM1_cddb) * ag0;
      
      gcsgb = (here->HSM1_cdgb - cgso) * ag0;
      gcsdb = here->HSM1_cdsb * ag0;
      gcssb = (here->HSM1_cddb + here->HSM1_capbs + cgso) * ag0;
      
      gcbgb = (here->HSM1_cbgb - cgbo) * ag0;
      gcbdb = (here->HSM1_cbsb - here->HSM1_capbd) * ag0;
      gcbsb = (here->HSM1_cbdb - here->HSM1_capbs) * ag0;

      if (sIN.coadov != 1) {
	qgd = cgdo * vgd;
	qgs = cgso * vgs;
	qgb = cgbo * vgb;
	qgate += qgd + qgs + qgb;
	qbulk -= qgb;
	qsrc = qdrn - qgs;
	qdrn = -(qgate + qbulk + qsrc);
      }
      else {
	qdrn = -(qgate + qbulk + qdrn);
      }
    }

    if (ByPass) goto line860;
    
    *(ckt->CKTstate0 + here->HSM1qg) = qgate;
    *(ckt->CKTstate0 + here->HSM1qd) = qdrn - *(ckt->CKTstate0 + here->HSM1qbd);
    *(ckt->CKTstate0 + here->HSM1qb) = 
      qbulk + *(ckt->CKTstate0 + here->HSM1qbd) + *(ckt->CKTstate0 + here->HSM1qbs);

    /*
    printf( "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq\n" ) ;
    printf( "HSM1qg   = %12.5e\n" , *(ckt->CKTstate0 + here->HSM1qg) ) ;    
    printf( "HSM1qd   = %12.5e\n" , *(ckt->CKTstate0 + here->HSM1qd) ) ;    
    printf( "HSM1qb   = %12.5e\n" , *(ckt->CKTstate0 + here->HSM1qb) ) ;    
    printf( "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq\n" ) ;
    */

    /* store small signal parameters */
    if (ckt->CKTmode & MODEINITSMSIG) goto line1000;
    if (!ChargeComputationNeeded) goto line850;

    if (ckt->CKTmode & MODEINITTRAN) {
      *(ckt->CKTstate1 + here->HSM1qb) = *(ckt->CKTstate0 + here->HSM1qb);
      *(ckt->CKTstate1 + here->HSM1qg) = *(ckt->CKTstate0 + here->HSM1qg);
      *(ckt->CKTstate1 + here->HSM1qd) = *(ckt->CKTstate0 + here->HSM1qd);
    }
    
    if ((error = NIintegrate(ckt, &geq, &ceq, 0.0, here->HSM1qb))) return(error);
    if ((error = NIintegrate(ckt, &geq, &ceq, 0.0, here->HSM1qg))) return(error);
    if ((error = NIintegrate(ckt, &geq, &ceq, 0.0, here->HSM1qd))) return(error);

    goto line860;
    
 line850:
    /* initialize to zero charge conductance and current */
    ceqqg = ceqqb = ceqqd = 0.0;
	  
    gcdgb = gcddb = gcdsb = 0.0;
    gcsgb = gcsdb = gcssb = 0.0;
    gcggb = gcgdb = gcgsb = 0.0;
    gcbgb = gcbdb = gcbsb = 0.0;

    goto line900;
    
 line860:
    /* evaluate equivalent charge current */
    
    cqgate = *(ckt->CKTstate0 + here->HSM1cqg);
    cqbulk = *(ckt->CKTstate0 + here->HSM1cqb);
    cqdrn = *(ckt->CKTstate0 + here->HSM1cqd);

    /*
    printf( "iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii\n" ) ;
    printf( "cqgate   = %12.5e\n" , cqgate ) ;
    printf( "cqbulk   = %12.5e\n" , cqbulk ) ;
    printf( "cqdrn   = %12.5e\n" , cqdrn ) ;
    printf( "iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii\n" ) ;
    */
    
    ceqqg = cqgate - gcggb * vgb + gcgdb * vbd + gcgsb * vbs;
    ceqqb = cqbulk - gcbgb * vgb + gcbdb * vbd + gcbsb * vbs;
    ceqqd = cqdrn - gcdgb * vgb + gcddb * vbd + gcdsb * vbs;

    if (ckt->CKTmode & MODEINITTRAN) {
      *(ckt->CKTstate1 + here->HSM1cqb) = *(ckt->CKTstate0 + here->HSM1cqb);
      *(ckt->CKTstate1 + here->HSM1cqg) = *(ckt->CKTstate0 + here->HSM1cqg);
      *(ckt->CKTstate1 + here->HSM1cqd) = *(ckt->CKTstate0 + here->HSM1cqd);
    }

    /*
     *  load current vector
     */
 line900:

    m = here->HSM1_m;

    if (here->HSM1_mode >= 0) { /* NORMAL mode */
      gm = here->HSM1_gm;
      gmbs = here->HSM1_gmbs;
      FwdSum = gm + gmbs;
      RevSum = 0.0;

      cdreq = model->HSM1_type * 
	(cdrain - here->HSM1_gds * vds - gm * vgs - gmbs * vbs); 
      ceqbd = -model->HSM1_type * 
	(here->HSM1_isub - here->HSM1_gbds * vds - here->HSM1_gbgs * vgs
	 - here->HSM1_gbbs * vbs);
      ceqbs = 0.0;
      
      gbbdp = -here->HSM1_gbds;
      gbbsp = here->HSM1_gbds + here->HSM1_gbgs + here->HSM1_gbbs;

      gbdpg = here->HSM1_gbgs;
      gbdpdp = here->HSM1_gbds;
      gbdpb = here->HSM1_gbbs;
      gbdpsp = -(gbdpg + gbdpdp + gbdpb);

      gbspg = 0.0;
      gbspdp = 0.0;
      gbspb = 0.0;
      gbspsp = 0.0;
    }
    else { /* REVERSE mode */
      gm = -here->HSM1_gm;
      gmbs = -here->HSM1_gmbs;
      FwdSum = 0.0;
      RevSum = -(gm + gmbs);

      cdreq = -model->HSM1_type * 
	(cdrain + here->HSM1_gds * vds + gm * vgd + gmbs * vbd);
      ceqbs = -model->HSM1_type * 
	(here->HSM1_isub + here->HSM1_gbds * vds - here->HSM1_gbgs * vgd
	 - here->HSM1_gbbs * vbd);
      ceqbd = 0.0;

      gbbsp = -here->HSM1_gbds;
      gbbdp = here->HSM1_gbds + here->HSM1_gbgs + here->HSM1_gbbs;

      gbdpg = 0.0;
      gbdpsp = 0.0;
      gbdpb = 0.0;
      gbdpdp = 0.0;

      gbspg = here->HSM1_gbgs;
      gbspsp = here->HSM1_gbds;
      gbspb = here->HSM1_gbbs;
      gbspdp = -(gbspg + gbspsp + gbspb);
    }
    
    if (model->HSM1_type > 0) { 
      ceqbs += here->HSM1_ibs - here->HSM1_gbs * vbs;
      ceqbd += here->HSM1_ibd - here->HSM1_gbd * vbd;
    }
    else {
      ceqbs -= here->HSM1_ibs - here->HSM1_gbs * vbs; 
      ceqbd -= here->HSM1_ibd - here->HSM1_gbd * vbd;
      ceqqg = -ceqqg;
      ceqqb = -ceqqb;
      ceqqd = -ceqqd;
    }

    /*
    printf( "----------------------------------------------------\n" ) ;
    printf( "ceqqg   = %12.5e\n" , ceqqg ) ;
    printf( "....................................................\n" ) ;
    printf( "ceqbs   = %12.5e\n" , ceqbs ) ;
    printf( "ceqbd   = %12.5e\n" , ceqbd ) ;
    printf( "ceqqb   = %12.5e\n" , ceqqb ) ;
    printf( "....................................................\n" ) ;
    printf( "ceqbd   = %12.5e\n" , ceqbd ) ;
    printf( "cdreq   = %12.5e\n" , cdreq ) ;
    printf( "ceqqd   = %12.5e\n" , ceqqd ) ;
    printf( "----------------------------------------------------\n" ) ;
    */
    
    *(ckt->CKTrhs + here->HSM1gNode) -= m * ceqqg;
    *(ckt->CKTrhs + here->HSM1bNode) -= m * (ceqbs + ceqbd + ceqqb);
    *(ckt->CKTrhs + here->HSM1dNodePrime) += m * (ceqbd - cdreq - ceqqd);
    *(ckt->CKTrhs + here->HSM1sNodePrime) += m * (cdreq 
      + ceqbs + ceqqg + ceqqb + ceqqd);
    
    /*
     *  load y matrix
     */

    *(here->HSM1DdPtr) += m * here->HSM1drainConductance;
    *(here->HSM1GgPtr) += m * gcggb;
    *(here->HSM1SsPtr) += m * here->HSM1sourceConductance;
    *(here->HSM1BbPtr) += m * (here->HSM1_gbd + here->HSM1_gbs
      - gcbgb - gcbdb - gcbsb - here->HSM1_gbbs);
    *(here->HSM1DPdpPtr) += m * (here->HSM1drainConductance
      + here->HSM1_gds + here->HSM1_gbd + RevSum + gcddb + gbdpdp);
    *(here->HSM1SPspPtr) += m * (here->HSM1sourceConductance
      + here->HSM1_gds + here->HSM1_gbs + FwdSum + gcssb + gbspsp);
    *(here->HSM1DdpPtr) -= m * here->HSM1drainConductance;
    *(here->HSM1GbPtr) -= m * (gcggb + gcgdb + gcgsb);
    *(here->HSM1GdpPtr) += m * gcgdb;
    *(here->HSM1GspPtr) += m * gcgsb;
    *(here->HSM1SspPtr) -= m * here->HSM1sourceConductance;
    *(here->HSM1BgPtr) += m * (gcbgb - here->HSM1_gbgs);
    *(here->HSM1BdpPtr) += m * (gcbdb - here->HSM1_gbd + gbbdp);
    *(here->HSM1BspPtr) += m * (gcbsb - here->HSM1_gbs + gbbsp);
    *(here->HSM1DPdPtr) -= m * here->HSM1drainConductance;
    *(here->HSM1DPgPtr) += m * (gm + gcdgb + gbdpg);
    *(here->HSM1DPbPtr) -= m * (here->HSM1_gbd - gmbs + gcdgb + gcddb + gcdsb - gbdpb);
    *(here->HSM1DPspPtr) -= m * (here->HSM1_gds + FwdSum - gcdsb - gbdpsp);
    *(here->HSM1SPgPtr) += m * (gcsgb - gm + gbspg);
    *(here->HSM1SPsPtr) -= m * here->HSM1sourceConductance;
    *(here->HSM1SPbPtr) -= m * (here->HSM1_gbs + gmbs + gcsgb + gcsdb + gcssb - gbspb);
    *(here->HSM1SPdpPtr) -= m * (here->HSM1_gds + RevSum - gcsdb - gbspdp);

  line1000:
    ;
    
   } /* End of MOSFET Instance */
  } /* End of Model Instance */
  return(OK);
}
