/***********************************************************************
 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2003 STARC

 VERSION : HiSIM 1.2.0
 FILE : hsm1ld.c of HiSIM 1.2.0

 April 9, 2003 : released by STARC Physical Design Group
***********************************************************************/

#include "ngspice.h"
#include "cktdefs.h"
#include "hsm1def.h"
#include "hisim.h"
#include "trandefs.h"
#include "const.h"
#include "sperror.h"
#include "devdefs.h"
#include "suffix.h"

#define SHOW_EPS_QUANT 1.0e-15

static void 
ShowPhysVals(HSM1instance *here, HSM1model *model,int isFirst, double vds, 
             double vgs, double vbs, double vgd, double vbd, double vgb)
{
  /* regard the epsilon-quantity as 0.0 */
  vds = (fabs(vds) < SHOW_EPS_QUANT) ? 0.0 : vds;
  vgs = (fabs(vgs) < SHOW_EPS_QUANT) ? 0.0 : vgs;
  vbs = (fabs(vbs) < SHOW_EPS_QUANT) ? 0.0 : vbs;
  vgb = (fabs(vgb) < SHOW_EPS_QUANT) ? 0.0 : vgb;
  switch (model->HSM1_show) {
  case 1:
    if (isFirst) printf("Vds        Ids\n");
    printf("%e %e\n", model->HSM1_type*vds, here->HSM1_mode*here->HSM1_ids);
    break;
  case 2:
    if (isFirst) printf("Vgs        Ids\n");
    printf("%e %e\n", model->HSM1_type*vgs, here->HSM1_mode*here->HSM1_ids);
    break;
  case 3:
    if (isFirst) printf("Vgs        log10(|Ids|)\n");
    printf("%e %e\n", model->HSM1_type*vgs, log10(here->HSM1_ids));
    break;
  case 4:
    if (isFirst) printf("log10(|Ids|)    gm/|Ids|\n");
    if (here->HSM1_ids == 0.0)
      printf("I can't show gm/Ids - log10(Ids), because Ids = 0.\n");
    else
      printf("%e %e\n",  log10(here->HSM1_ids), here->HSM1_gm/here->HSM1_ids);
    break;
  case 5:
    if (isFirst) printf("Vds        gds\n");
    printf("%e %e\n", model->HSM1_type*vds, here->HSM1_gds);
    break;
  case 6:
    if (isFirst) printf("Vgs        gm\n");
    printf("%e %e\n", model->HSM1_type*vgs, here->HSM1_gm);
    break;
  case 7:
    if (isFirst) printf("Vbs        gbs\n");
    printf("%e %e\n", model->HSM1_type*vbs, here->HSM1_gmbs);
    break;
  case 8:
    if (isFirst) printf("Vgs        Cgg\n");
    printf("%e %e\n", model->HSM1_type*vgs, here->HSM1_cggb);
    break;
  case 9:
    if (isFirst) printf("Vgs        Cgs\n");
    printf("%e %e\n", model->HSM1_type*vgs, here->HSM1_cgsb);
    break;
  case 10:
    if (isFirst) printf("Vgs        Cgd\n");
    printf("%e %e\n", model->HSM1_type*vgs, here->HSM1_cgdb);
    break;
  case 11:
    if (isFirst) printf("Vgs        Cgb\n");
    printf("%e %e\n", model->HSM1_type*vgs, -(here->HSM1_cggb+here->HSM1_cgsb+here->HSM1_cgdb));
    break;
  case 12:
    if (isFirst) printf("Vds        Csg\n");
    printf("%e %e\n", model->HSM1_type*vds, -(here->HSM1_cggb+here->HSM1_cbgb+here->HSM1_cdgb));
    break;
  case 13:
    if (isFirst) printf("Vds        Cdg\n");
    printf("%e %e\n", model->HSM1_type*vds, here->HSM1_cdgb);
    break;
  case 14:
    if (isFirst) printf("Vds        Cbg\n");
    printf("%e %e\n", model->HSM1_type*vds, here->HSM1_cbgb);
    break;
  case 15:
    if (isFirst) printf("Vds        Cgg\n");
    printf("%e %e\n", model->HSM1_type*vds, here->HSM1_cggb);
    break;
  case 16:
    if (isFirst) printf("Vds        Cgs\n");
    printf("%e %e\n", model->HSM1_type*vds, here->HSM1_cgsb);
    break;
  case 17:
    if (isFirst) printf("Vds        Cgd\n");
    printf("%e %e\n", model->HSM1_type*vds, here->HSM1_cgdb);
    break;
  case 18:
    if (isFirst) printf("Vds        Cgb\n");
    printf("%e %e\n", model->HSM1_type*vds, -(here->HSM1_cggb+here->HSM1_cgsb+here->HSM1_cgdb));
    break;
  case 19:
    if (isFirst) printf("Vgs        Csg\n");
    printf("%e %e\n", model->HSM1_type*vgs, -(here->HSM1_cggb+here->HSM1_cbgb+here->HSM1_cdgb));
    break;
  case 20:
    if (isFirst) printf("Vgs        Cdg\n");
    printf("%e %e\n", model->HSM1_type*vgs, here->HSM1_cdgb);
    break;
  case 21:
    if (isFirst) printf("Vgs        Cbg\n");
    printf("%e %e\n", model->HSM1_type*vgs, here->HSM1_cbgb);
    break;
  case 22:
    if (isFirst) printf("Vgb        Cgb\n");
    printf("%e %e\n", model->HSM1_type*vgb, -(here->HSM1_cggb+here->HSM1_cgsb+here->HSM1_cgdb));
    break;
  case 50:
    if (isFirst) printf("Vgs  Vds  Vbs  Vgb  Ids  log10(|Ids|)  gm/|Ids|  gm  gds  gbs  Cgg  Cgs  Cgb  Cgd  Csg  Cbg  Cdg\n");
    printf("%e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n", model->HSM1_type*vgs, model->HSM1_type*vds, model->HSM1_type*vbs, model->HSM1_type*vgb, here->HSM1_mode*here->HSM1_ids, log10(here->HSM1_ids), here->HSM1_gm/here->HSM1_ids, here->HSM1_gm, here->HSM1_gds, here->HSM1_gmbs, here->HSM1_cggb, here->HSM1_cgsb, -(here->HSM1_cggb+here->HSM1_cgsb+here->HSM1_cgdb), here->HSM1_cgdb, -(here->HSM1_cggb+here->HSM1_cbgb+here->HSM1_cdgb), here->HSM1_cbgb, here->HSM1_cdgb);
    break;
  default:
    /*
    printf("There is no physical value corrsponding to %d\n", flag);
    */
    break;
  }
}

int HSM1load(GENmodel *inModel, register CKTcircuit *ckt)
     /* actually load the current value into the 
      * sparse matrix previously provided 
      */
{
  register HSM1model *model = (HSM1model*)inModel;
  register HSM1instance *here;
  HiSIM_input sIN;
  HiSIM_output sOT;
  HiSIM_messenger sMS;
  double cbhat = 0.0, cdrain, cdhat = 0.0, cdreq;
  double cgbhat = 0.0, cgshat = 0.0, cgdhat = 0.0;
  double Ibtot = 0.0, Idtot, Igbtot = 0.0, Igstot = 0.0, Igdtot = 0.0;
  double ceq, ceqbd, ceqbs, ceqqb, ceqqd, ceqqg;
  double delvbd, delvbs, delvds, delvgd, delvgs;
  double gcbdb, gcbgb, gcbsb, gcddb, gcdgb, gcdsb;
  double gcgdb, gcggb, gcgsb, gcsdb, gcsgb, gcssb;
  double geq, xfact;
  double vbd, vbs, vcrit, vds, vgb, vgd, vgdo, vgs, von;
  double qgd, qgs, qgb;
  double gbbdp, gbbsp, gbspg, gbspdp, gbspb, gbspsp;
  double qgate, qbulk, qdrn, qsrc;
  double cqgate, cqbulk, cqdrn;
  double gbdpdp, gbdpg, gbdpb, gbdpsp; 
  double cgdo, cgso, cgbo;
  double gm, gmbs, FwdSum, RevSum;
  double vt0, ag0;
  double Ibtoteq, gIbtotg, gIbtotd, gIbtots, gIbtotb;
  double Igtoteq, gIgtotg, gIgtotd, gIgtots, gIgtotb;
  double Idtoteq, gIdtotg, gIdtotd, gIdtots, gIdtotb;
  double Istoteq, gIstotg, gIstotd, gIstots, gIstotb;
  int ByPass, Check, error;
#ifndef NOBYPASS
  double tempv;
#endif /*NOBYPASS*/
  double tmp;
#ifndef NEWCONV
  double tol, tol2, tol3, tol4;
#endif
  int ChargeComputationNeeded =  
    ((ckt->CKTmode & (MODEAC | MODETRAN | MODEINITSMSIG)) ||
     ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)))
    ? 1 : 0;
  int showPhysVal;
  int isConv;
  double vds_pre = 0.0;

  double m; /* Parallel multiplier */
  
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
	if (ckt->CKTmode & MODEINITPRED) {
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
	  Idtot = here->HSM1_ids + here->HSM1_isub - here->HSM1_ibd
	    + here->HSM1_igidl;
	  cdhat = Idtot - here->HSM1_gbd * delvbd
	    + (here->HSM1_gmbs + here->HSM1_gbbs + here->HSM1_gigidlbs) * delvbs 
	    + (here->HSM1_gm + here->HSM1_gbgs + here->HSM1_gigidlgs) * delvgs
	    + (here->HSM1_gds + here->HSM1_gbds + here->HSM1_gigidlds) * delvds;
	  Ibtot = here->HSM1_ibs + here->HSM1_ibd - here->HSM1_isub
	    - here->HSM1_igidl - here->HSM1_igisl;
	  cbhat = Ibtot + here->HSM1_gbd * delvbd
	    + (here->HSM1_gbs -  here->HSM1_gbbs - here->HSM1_gigidlbs) * delvbs
	    - (here->HSM1_gbgs + here->HSM1_gigidlgs) * delvgs
	    - (here->HSM1_gbds + here->HSM1_gigidlds) * delvds
	    - here->HSM1_gigislgd * delvgd - here->HSM1_gigislbd * delvbd
	    + here->HSM1_gigislsd * delvds;
	  Igstot = here->HSM1_igs;
	  cgshat = Igstot + here->HSM1_gigsg * delvgs + 
	    here->HSM1_gigsd * delvds + here->HSM1_gigsb * delvbs;
	  Igdtot = here->HSM1_igd;
	  cgdhat = Igdtot + here->HSM1_gigdg * delvgs + 
	    here->HSM1_gigdd * delvds + here->HSM1_gigdb * delvbs;
	  Igbtot = here->HSM1_igb;
	  cgbhat = Igbtot + here->HSM1_gigbg * delvgs + 
	    here->HSM1_gigbd * delvds + here->HSM1_gigbb * delvbs;
	} 
	else {
	  Idtot = here->HSM1_ids + here->HSM1_ibd - here->HSM1_igidl;
	  cdhat = Idtot + (here->HSM1_gbd + here->HSM1_gmbs) * delvbd
	    + here->HSM1_gm * delvgd - here->HSM1_gds * delvds
	    - here->HSM1_gigidlgs * delvgd - here->HSM1_gigidlbs * delvbd 
	    + here->HSM1_gigidlds * delvds ;
	  Ibtot = here->HSM1_ibs + here->HSM1_ibd - here->HSM1_isub
	    - here->HSM1_igidl - here->HSM1_igisl;
	  cbhat = Ibtot + here->HSM1_gbs * delvbs
	    + (here->HSM1_gbd - here->HSM1_gbbs - here->HSM1_gigidlbs) * delvbd
	    - (here->HSM1_gbgs + here->HSM1_gigidlgs) * delvgd
	    + (here->HSM1_gbds + here->HSM1_gigidlds) * delvds
	    - here->HSM1_gigislgd * delvgd - here->HSM1_gigislbd * delvbd
	    + here->HSM1_gigislsd * delvds;
	  Igbtot = here->HSM1_igb;
	  cgbhat = Igbtot + here->HSM1_gigbg * delvgd
	    - here->HSM1_gigbs * delvds + here->HSM1_gigbb * delvbd;
	  Igstot = here->HSM1_igs;
	  cgshat = Igstot + here->HSM1_gigsg * delvgd
	    - here->HSM1_gigss * delvds + here->HSM1_gigsb * delvbd;
	  Igdtot = here->HSM1_igd;
	  cgdhat = Igdtot + here->HSM1_gigdg * delvgd
	    - here->HSM1_gigds * delvds + here->HSM1_gigdb * delvbd;
	}

	vds_pre = vds;

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
			 MAX(fabs(cdhat),fabs(Idtot)) + ckt->CKTabstol ) ) 
		    if (!model->HSM1_coiigs || 
			(fabs(cgbhat - Igbtot) < ckt->CKTreltol
			 * MAX(fabs(cgbhat), fabs(Igbtot)) + ckt->CKTabstol))
		      if (!model->HSM1_coiigs || 
			  (fabs(cgshat - Igstot) < ckt->CKTreltol
			   * MAX(fabs(cgshat), fabs(Igstot)) + ckt->CKTabstol))
			if (!model->HSM1_coiigs || 
			    (fabs(cgdhat - Igdtot) < ckt->CKTreltol
			     * MAX(fabs(cgdhat), fabs(Igdtot)) + ckt->CKTabstol)){
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
      sIN.gmin = ckt->CKTgmin; /* minimum conductance */

      if (vds >= 0) { /* normal mode */
	here->HSM1_mode = 1;
	sIN.vds = vds;
	sIN.vgs = vgs;
	sIN.vbs = vbs;
      }
      else { /* reverse mode */
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
      sIN.xld = model->HSM1_xld;
      sIN.xwd = model->HSM1_xwd;
      
      if (model->HSM1_version == 102) { /* HiSIM 1.0.2 */
	sIN.xj = model->HSM1_xj;
      }
      else if (model->HSM1_version == 112 || 
	       model->HSM1_version == 120) { /* HiSIM 1.1.2 / 1.2.0 */
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
      if (model->HSM1_version == 112 ||
	  model->HSM1_version == 120) { /* HiSIM 1.1.2 / 1.2.0 */
	sIN.wvthsc = model->HSM1_wvthsc;
	sIN.nsti = model->HSM1_nsti;
	sIN.wsti = model->HSM1_wsti;
      }
      
      if ( model->HSM1_bb_Given ) sIN.bb = model->HSM1_bb;
      else
	if ( model->HSM1_type == NMOS ) sIN.bb = 2.0e0;
	else sIN.bb = 1.0e0;
      
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
      if (model->HSM1_version == 112 ||
	  model->HSM1_version == 120) { /* HiSIM 1.1.2 / 1.2.0 */
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
      sIN.cogisl = model->HSM1_cogisl;
      sIN.coovlp = model->HSM1_coovlp;
      sIN.conois = model->HSM1_conois;
      if (model->HSM1_version == 112 ||
	  model->HSM1_version == 120) { /* HiSIM 1.1.2 / 1.2.0 */
	sIN.coisti = model->HSM1_coisti;
      }
      if (model->HSM1_version == 120) { /* HiSIM 1.2.0 */
	sIN.cosmbi = model->HSM1_cosmbi;
	sIN.glpart1 = model->HSM1_glpart1;
	sIN.glpart2 = model->HSM1_glpart2;
	sIN.kappa = model->HSM1_kappa;
	sIN.xdiffd = model->HSM1_xdiffd;
	sIN.pthrou = model->HSM1_pthrou;
	sIN.vdiffj = model->HSM1_vdiffj;
      }
      sIN.version = model->HSM1_version ;

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
	printf( "cogisl = %s\n" , (sIN.cogisl) ? "true" : "false" ) ;
	printf( "coovlp = %s\n" , (sIN.coovlp) ? "true" : "false" ) ;
	printf( "conois = %s\n" , (sIN.conois) ? "true" : "false" ) ;
	if (model->HSM1_version == 112 ||
	    model->HSM1_version == 120) { /* HiSIM 1.1.2 / 1.2.0 */
	  printf( "coisti = %s\n" , (sIN.coisti) ? "true" : "false" ) ;
	}
      }
      /* print inputs ------------AA */

      /* call model evaluation */
      if (model->HSM1_version == 102) { /* HiSIM 1.0.2 */
	if ( HSM1evaluate102(sIN, &sOT, &sMS) == HiSIM_ERROR ) 
	  return (HiSIM_ERROR);
      }
      else if (model->HSM1_version == 112) { /* HiSIM 1.1.2 */
	if ( HSM1evaluate112(sIN, &sOT, &sMS) == HiSIM_ERROR ) 
	  return (HiSIM_ERROR);
      }
      else if (model->HSM1_version == 120) { /* HiSIM 1.2.0 */
	if ( HSM1evaluate120(sIN, &sOT, &sMS) == HiSIM_ERROR ) 
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
      /* gate induced drain leakage */
      here->HSM1_igidl = sOT.igidl;
      here->HSM1_gigidlgs = sOT.ggidlgs;
      here->HSM1_gigidlds = sOT.ggidlds;
      here->HSM1_gigidlbs = sOT.ggidlbs;
      if (model->HSM1_version == 120) { /* HiSIM 1.2.0 */
	/* gate induced source leakage */
	here->HSM1_igisl = sOT.igisl;
	here->HSM1_gigislgd = sOT.ggislgd;
	here->HSM1_gigislsd = sOT.ggislsd;
	here->HSM1_gigislbd = sOT.ggislbd;

	/* gate tunneling current (gate to bulk) */
	here->HSM1_igb = sOT.igateb;
	here->HSM1_gigbg = sOT.ggbgs;
	here->HSM1_gigbd = sOT.ggbds;
	here->HSM1_gigbb = sOT.ggbbs;
	here->HSM1_gigbs = -( here->HSM1_gigbg + here->HSM1_gigbd 
			      + here->HSM1_gigbb );
	/* gate tunneling current (gate to source) */
	here->HSM1_igs = sOT.igates;
	here->HSM1_gigsg = sOT.ggsgs;
	here->HSM1_gigsd = sOT.ggsds;
	here->HSM1_gigsb = sOT.ggsbs;
	here->HSM1_gigss = -( here->HSM1_gigsg + here->HSM1_gigsd
			      + here->HSM1_gigsb );
	/* gate tunneling current (gate to drain) */
	here->HSM1_igd = sOT.igated;
	here->HSM1_gigdg = sOT.ggdgs;
	here->HSM1_gigdd = sOT.ggdds;
	here->HSM1_gigdb = sOT.ggdbs;
	here->HSM1_gigds = -( here->HSM1_gigdg + here->HSM1_gigdd
			      + here->HSM1_gigdb );
      }
      else {
	/* gate induced source leakage */
	here->HSM1_igisl = 0.0;
	here->HSM1_gigislgd = 0.0;
	here->HSM1_gigislsd = 0.0;
	here->HSM1_gigislbd = 0.0;

	/* gate tunneling current (gate to bulk) */
	here->HSM1_igb = sOT.igate;
	here->HSM1_gigbg = sOT.gggs;
	here->HSM1_gigbd = sOT.ggds;
	here->HSM1_gigbb = sOT.ggbs;
	here->HSM1_gigbs = -( here->HSM1_gigbg + here->HSM1_gigbd 
			      + here->HSM1_gigbb );
	/* gate tunneling current (gate to source) */
	here->HSM1_igs = 0.0;
	here->HSM1_gigsg = 0.0;
	here->HSM1_gigsd = 0.0;
	here->HSM1_gigsb = 0.0;
	here->HSM1_gigss = 0.0;
	/* gate tunneling current (gate to drain) */
	here->HSM1_igd = 0.0;
	here->HSM1_gigdg = 0.0;
	here->HSM1_gigdd = 0.0;
	here->HSM1_gigdb = 0.0;
	here->HSM1_gigds = 0.0;
      }
      /* intrinsic charges without overlap charge etc. */
      here->HSM1_qg_int = sOT.qg_int;
      here->HSM1_qd_int = sOT.qd_int;
      here->HSM1_qs_int = sOT.qs_int;
      here->HSM1_qb_int = sOT.qb_int;

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
      isConv = 1;
      if ( (here->HSM1_off == 0) || !(ckt->CKTmode & MODEINITFIX) ) {
	if (Check == 1) {
	  ckt->CKTnoncon++;
	  isConv = 0;
	} 
#ifndef NEWCONV
//DW	} 
	else {
	  if (here->HSM1_mode >= 0) 
	    Idtot = here->HSM1_ids + here->HSM1_isub 
	      - here->HSM1_ibd + here->HSM1_igidl;
	  else
	    Idtot = here->HSM1_ids + here->HSM1_ibd 
	      - here->HSM1_igidl;
	  tol = ckt->CKTreltol * MAX(fabs(cdhat), fabs(Idtot)) 
	    + ckt->CKTabstol;
	  tol2 = ckt->CKTreltol * MAX(fabs(cgbhat), fabs(Igbtot)) 
	    + ckt->CKTabstol;
	  tol3 = ckt->CKTreltol * MAX(fabs(cgshat), fabs(Igstot)) 
	    + ckt->CKTabstol;
	  tol4 = ckt->CKTreltol * MAX(fabs(cgdhat), fabs(Igdtot)) 
	    + ckt->CKTabstol;
	  if (fabs(cdhat - Idtot) >= tol) {
	    ckt->CKTnoncon++;
	    isConv = 0;
	  }
	  else if (fabs(cgbhat - Igbtot) >= tol2 || 
		   fabs(cgshat - Igstot) >= tol3 ||
		   fabs(cgdhat - Igdtot) >= tol4) {
	    ckt->CKTnoncon++;
	    isConv = 0;
	  }
	  else {
	    Ibtot = here->HSM1_ibs + here->HSM1_ibd 
	      - here->HSM1_isub - here->HSM1_igidl - here->HSM1_igisl;
	    tol = ckt->CKTreltol * 
	      MAX(fabs(cbhat), fabs(Ibtot)) + ckt->CKTabstol;
	    if (fabs(cbhat - Ibtot) > tol) {
	      ckt->CKTnoncon++;
	      isConv = 0;
	    }
	  }
	}
#endif /* NEWCONV */
      }
//DW    }
    *(ckt->CKTstate0 + here->HSM1vbs) = vbs;
    *(ckt->CKTstate0 + here->HSM1vbd) = vbd;
    *(ckt->CKTstate0 + here->HSM1vgs) = vgs;
    *(ckt->CKTstate0 + here->HSM1vds) = vds;

    if ((ckt->CKTmode & MODEDC) && 
	!(ckt->CKTmode & MODEINITFIX) && !(ckt->CKTmode & MODEINITJCT)) 
      showPhysVal = 1;

    /*
    if ((ckt->CKTmode & MODEDC) && !(ckt->CKTmode & MODEINITFIX) &&
	!(ckt->CKTmode & MODEINITJCT)) {
      if ((!(ckt->CKTmode & MODEINITPRED) && vds != vds_pre) || 
	  ((ckt->CKTmode & MODEINITPRED) && vds == vds_pre))
	showPhysVal = 1;
    }
    */

    if (model->HSM1_show_Given && showPhysVal && isConv) {
      static int isFirst = 1;
      if (vds != vds_pre) 
	ShowPhysVals(here, model, isFirst, vds_pre, vgs, vbs, vgd, vbd, vgb);
      else 
	ShowPhysVals(here, model, isFirst, vds, vgs, vbs, vgd, vbd, vgb);
      if (isFirst) isFirst = 0;
    }

    /* bulk and channel charge plus overlaps */
    
    if (!ChargeComputationNeeded) goto line850; 
    
 line755:
    cgdo = here->HSM1_cgdo;
    cgso = here->HSM1_cgso;
    cgbo = here->HSM1_cgbo;

    ag0 = ckt->CKTag[0];
    if (here->HSM1_mode > 0) { /* NORMAL mode */
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
	(cdrain - here->HSM1_gds * vds - here->HSM1_gm * vgs - here->HSM1_gmbs * vbs); 
      ceqbd = -model->HSM1_type * 
	(here->HSM1_isub + here->HSM1_igidl 
	 - (here->HSM1_gbds + here->HSM1_gigidlds) * vds 
	 - (here->HSM1_gbgs + here->HSM1_gigidlgs) * vgs 
	 - (here->HSM1_gbbs + here->HSM1_gigidlbs) * vbs);
      ceqbs = -model->HSM1_type * 
	(here->HSM1_igisl 
	 + here->HSM1_gigislsd * vds 
	 - here->HSM1_gigislgd * vgd 
	 - here->HSM1_gigislbd * vbd);
      
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

      if (model->HSM1_coiigs) {
	gIbtotg = here->HSM1_gigbg;
	gIbtotd = here->HSM1_gigbd;
	gIbtots = here->HSM1_gigbs;
	gIbtotb = here->HSM1_gigbb;
	Ibtoteq = model->HSM1_type * 
	  (here->HSM1_igb - here->HSM1_gigbg * vgs 
	   - here->HSM1_gigbd * vds - here->HSM1_gigbb * vbs);

	gIstotg = here->HSM1_gigsg;
	gIstotd = here->HSM1_gigsd;
	gIstots = here->HSM1_gigss;
	gIstotb = here->HSM1_gigsb;
	Istoteq = model->HSM1_type * 
	  (here->HSM1_igs - here->HSM1_gigsg * vgs
	   - here->HSM1_gigsd * vds - here->HSM1_gigsb * vbs);

	gIdtotg = here->HSM1_gigdg;
	gIdtotd = here->HSM1_gigdd;
	gIdtots = here->HSM1_gigds;
	gIdtotb = here->HSM1_gigdb;
	Idtoteq = model->HSM1_type * 
	  (here->HSM1_igd - here->HSM1_gigdg * vgs
	   - here->HSM1_gigdd * vds - here->HSM1_gigdb * vbs);
      }
      else {
	gIbtotg = gIbtotd = gIbtots = gIbtotb = Ibtoteq = 0.0;
	gIstotg = gIstotd = gIstots = gIstotb = Istoteq = 0.0;
	gIdtotg = gIdtotd = gIdtots = gIdtotb = Idtoteq = 0.0;
      }

      if (model->HSM1_coiigs) {
	gIgtotg = gIbtotg + gIstotg + gIdtotg;
	gIgtotd = gIbtotd + gIstotd + gIdtotd;
	gIgtots = gIbtots + gIstots + gIdtots;
	gIgtotb = gIbtotb + gIstotb + gIdtotb;
	Igtoteq = Ibtoteq + Istoteq + Idtoteq;
      }
      else
	gIgtotg = gIgtotd = gIgtots = gIgtotb = Igtoteq = 0.0;

    }
    else { /* REVERSE mode */
      gm = -here->HSM1_gm;
      gmbs = -here->HSM1_gmbs;
      FwdSum = 0.0;
      RevSum = -(gm + gmbs);

      cdreq = -model->HSM1_type * 
	(cdrain + here->HSM1_gds * vds 
	 - here->HSM1_gm * vgd - here->HSM1_gmbs * vbd);
      ceqbs = -model->HSM1_type * 
	(here->HSM1_isub + here->HSM1_igisl 
	 + (here->HSM1_gbds + here->HSM1_gigislsd) * vds
	 - (here->HSM1_gbgs + here->HSM1_gigislgd) * vgd
	 - (here->HSM1_gbbs + here->HSM1_gigislbd) * vbd);
      ceqbd = -model->HSM1_type * 
	( here->HSM1_igidl 
	 - here->HSM1_gigidlds * vds 
	 - here->HSM1_gigidlgs * vgs 
	 - here->HSM1_gigidlbs * vbs);

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

      if (model->HSM1_coiigs) {
	gIbtotg = here->HSM1_gigbg;
	gIbtotd = here->HSM1_gigbd;
	gIbtots = here->HSM1_gigbs;
	gIbtotb = here->HSM1_gigbb;
	Ibtoteq = model->HSM1_type * 
	  (here->HSM1_igb - here->HSM1_gigbg * vgd 
	   + here->HSM1_gigbs * vds - here->HSM1_gigbb * vbd);

	gIstotg = here->HSM1_gigsg;
	gIstotd = here->HSM1_gigsd;
	gIstots = here->HSM1_gigss;
	gIstotb = here->HSM1_gigsb;
	Istoteq = model->HSM1_type * 
	  (here->HSM1_igs - here->HSM1_gigsg * vgd
	   + here->HSM1_gigss * vds - here->HSM1_gigsb * vbd);
	gIdtotg = here->HSM1_gigdg;
	gIdtotd = here->HSM1_gigdd;
	gIdtots = here->HSM1_gigds;
	gIdtotb = here->HSM1_gigdb;
	Idtoteq = model->HSM1_type *
	  (here->HSM1_igd - here->HSM1_gigdg * vgd
	   + here->HSM1_gigds * vds - here->HSM1_gigdb * vbd);
      }
      else {
	gIbtotg = gIbtotd = gIbtots = gIbtotb = Ibtoteq = 0.0;
	gIstotg = gIstotd = gIstots = gIstotb = Istoteq = 0.0;
	gIdtotg = gIdtotd = gIdtots = gIdtotb = Idtoteq = 0.0;
      }

      if (model->HSM1_coiigs) {
	gIgtotg = gIbtotg + gIstotg + gIdtotg;
	gIgtotd = gIbtotd + gIstotd + gIdtotd;
	gIgtots = gIbtots + gIstots + gIdtots;
	gIgtotb = gIbtotb + gIstotb + gIdtotb;
	Igtoteq = Ibtoteq + Istoteq + Idtoteq;
      }
      else
	gIgtotg = gIgtotd = gIgtots = gIgtotb = Igtoteq = 0.0;

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
    
    *(ckt->CKTrhs + here->HSM1gNode)      -= m * (ceqqg + Igtoteq);
    *(ckt->CKTrhs + here->HSM1bNode)      -= m * (ceqbs + ceqbd + ceqqb - Ibtoteq);
    *(ckt->CKTrhs + here->HSM1dNodePrime) += m * (ceqbd - cdreq - ceqqd + Idtoteq);
    *(ckt->CKTrhs + here->HSM1sNodePrime) += m * (cdreq + ceqbs + ceqqg + ceqqb + ceqqd + Istoteq);
    
    /*
     *  load y matrix
     */

    *(here->HSM1DdPtr)   += m * here->HSM1drainConductance;
    *(here->HSM1GgPtr)   += m * (gcggb + gIgtotg);
    *(here->HSM1SsPtr)   += m * here->HSM1sourceConductance;
    *(here->HSM1BbPtr)   += m * (here->HSM1_gbd + here->HSM1_gbs
                                 - gcbgb - gcbdb - gcbsb - 
				 here->HSM1_gbbs - gIbtotb);
    *(here->HSM1DPdpPtr) += m * (here->HSM1drainConductance
                                 + here->HSM1_gds + here->HSM1_gbd 
				 + RevSum + gcddb + gbdpdp - gIdtotd);
    *(here->HSM1SPspPtr) += m * (here->HSM1sourceConductance
                                 + here->HSM1_gds + here->HSM1_gbs 
				 + FwdSum + gcssb + gbspsp - gIstots);
    *(here->HSM1DdpPtr)  -= m * here->HSM1drainConductance;
    *(here->HSM1GbPtr)   -= m * (gcggb + gcgdb + gcgsb - gIgtotb);
    *(here->HSM1GdpPtr)  += m * (gcgdb + gIgtotd);
    *(here->HSM1GspPtr)  += m * (gcgsb + gIgtots);
    *(here->HSM1SspPtr)  -= m * (here->HSM1sourceConductance);
    *(here->HSM1BgPtr)   += m * (gcbgb - here->HSM1_gbgs - gIbtotg);
    *(here->HSM1BdpPtr)  += m * (gcbdb - here->HSM1_gbd + gbbdp 
                                 - gIbtotd);
    *(here->HSM1BspPtr)  += m * (gcbsb - here->HSM1_gbs + gbbsp 
                                 - gIbtots);
    *(here->HSM1DPdPtr)  -= m * (here->HSM1drainConductance);
    *(here->HSM1DPgPtr)  += m * (gm + gcdgb + gbdpg - gIdtotg);
    *(here->HSM1DPbPtr)  -= m * (here->HSM1_gbd - gmbs + gcdgb + gcddb 
                                 + gcdsb - gIdtotb);
    *(here->HSM1DPspPtr) -= m * (here->HSM1_gds + FwdSum - gcdsb - gbdpsp 
                                 + gIdtots);
    *(here->HSM1SPgPtr)  += m * (gcsgb - gm + gbspg - gIstotg);
    *(here->HSM1SPsPtr)  -= m * (here->HSM1sourceConductance);
    *(here->HSM1SPbPtr)  -= m * (here->HSM1_gbs + gmbs + gcsgb + gcsdb 
                                 + gcssb - gbspb + gIstotb);
    *(here->HSM1SPdpPtr) -= m * (here->HSM1_gds + RevSum - gcsdb - gbspdp 
                                 + gIstotd);

    /* stamp GIDL */
    *(here->HSM1DPdpPtr) += m * here->HSM1_gigidlds;
    *(here->HSM1DPgPtr)  += m * here->HSM1_gigidlgs;
    *(here->HSM1DPspPtr) -= m *(here->HSM1_gigidlgs 
                                + here->HSM1_gigidlds 
				+ here->HSM1_gigidlbs);
    *(here->HSM1DPbPtr)  += m * here->HSM1_gigidlbs;
    *(here->HSM1BdpPtr)  -= m * here->HSM1_gigidlds;
    *(here->HSM1BgPtr)   -= m * here->HSM1_gigidlgs;
    *(here->HSM1BspPtr)  += m * (here->HSM1_gigidlgs 
                                 + here->HSM1_gigidlds 
				 + here->HSM1_gigidlbs);
    *(here->HSM1BbPtr)   -= m * here->HSM1_gigidlbs;
    /* stamp GISL */
    *(here->HSM1SPdpPtr) -= m * (here->HSM1_gigislsd 
                                 + here->HSM1_gigislgd 
			         + here->HSM1_gigislbd);
    *(here->HSM1SPgPtr)  += m * here->HSM1_gigislgd;
    *(here->HSM1SPspPtr) += m * here->HSM1_gigislsd;
    *(here->HSM1SPbPtr)  += m * here->HSM1_gigislbd;
    *(here->HSM1BdpPtr)  += m * (here->HSM1_gigislgd 
                                 + here->HSM1_gigislsd 
				 + here->HSM1_gigislbd);
    *(here->HSM1BgPtr)   -= m * here->HSM1_gigislgd;
    *(here->HSM1BspPtr)  -= m * here->HSM1_gigislsd;
    *(here->HSM1BbPtr)   -= m * here->HSM1_gigislbd;

  line1000:
    ;
    
   } /* End of MOSFET Instance */
  } /* End of Model Instance */
  return(OK);
}
