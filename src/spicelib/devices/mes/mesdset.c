/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jaijeet Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/cktdefs.h"
#include "mesdefs.h"
#include "ngspice/distodef.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MESdSetup(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current resistance value into the 
         * sparse matrix previously provided 
         */
{
    MESmodel *model = (MESmodel*)inModel;
    MESinstance *here;
    double afact;
    double beta;
    double betap;
    double cdrain;
    double cg;
    double cgd;
    double csat;
    double czgd;
    double czgs;
    double denom;
    double evgd;
    double evgs;
    double gdpr;
    double gspr;
    double invdenom;
    double lfact;
    double phib;
    double prod;
    double vcap;
    double vcrit;
    double vds;
    double vgd;
    double vgs;
    double vgst;
    double vto;
                double lggd1;
		double lggd2;
		double lggd3;
                double lggs1;
		double lggs2;
		double lggs3;
		Dderivs d_cdrain, d_qgs, d_qgd;
	    Dderivs d_p, d_q, d_r, d_zero;

    /*  loop through all the models */
    for( ; model != NULL; model = MESnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = MESinstances(model); here != NULL ;
                here=MESnextInstance(here)) {

            /*
             *  dc model parameters 
             */
            beta = model->MESbeta * here->MESm * here->MESarea;
            gdpr = model->MESdrainConduct * here->MESm * here->MESarea;
            gspr = model->MESsourceConduct * here->MESm * here->MESarea;
            csat = model->MESgateSatCurrent * here->MESm * here->MESarea;
            vcrit = model->MESvcrit;
            vto = model->MESthreshold;
            /*
             *    initialization
             */
                    /*
                     *  compute new nonlinear branch voltages 
                     */
                    vgs = model->MEStype*
                        (*(ckt->CKTrhsOld+ here->MESgateNode)-
                        *(ckt->CKTrhsOld+ 
                        here->MESsourcePrimeNode));
                    vgd = model->MEStype*
                        (*(ckt->CKTrhsOld+here->MESgateNode)-
                        *(ckt->CKTrhsOld+
                        here->MESdrainPrimeNode));
                
            /*
             *   determine dc current and derivatives 
             */
            vds = vgs-vgd;
            if (vgs <= -5*CONSTvt0) {
                lggs1 = -csat/vgs+ckt->CKTgmin;
		lggs2=lggs3=0;
                cg = lggs1*vgs;
            } else {
                evgs = exp(vgs/CONSTvt0);
                lggs1 = csat*evgs/CONSTvt0+ckt->CKTgmin;
		lggs2 = (lggs1-ckt->CKTgmin)/(CONSTvt0*2);
		lggs3 = lggs2/(3*CONSTvt0);
                cg = csat*(evgs-1)+ckt->CKTgmin*vgs;
            }
            if (vgd <= -5*CONSTvt0) {
                lggd1 = -csat/vgd+ckt->CKTgmin;
		lggd2=lggd3=0;
                cgd = lggd1*vgd;
            } else {
                evgd = exp(vgd/CONSTvt0);
                lggd1 = csat*evgd/CONSTvt0+ckt->CKTgmin;
		lggd2 = (lggd1-ckt->CKTgmin)/(CONSTvt0*2);
		lggd3 = lggd2/(3*CONSTvt0);
                cgd = csat*(evgd-1)+ckt->CKTgmin*vgd;
            }
            cg = cg+cgd;
            /*
             *   compute drain current and derivitives for normal mode 
             */
	    /* until now, we were using the real vgs, vgd, and vds */
            {
	    /* converting (temporarily) to local vgs, vgd, and vds */
	    double vgsreal=vgs;
	    double vgdreal=vgd;
	    double vdsreal=vds;
	    Dderivs d_afact, d_lfact;
	    Dderivs d_betap, d_denom, d_invdenom;
	    Dderivs d_prod;
	    Dderivs d_vgst;

	    if (vdsreal < 0.0) {
		vgs = vgdreal;
		vgd = vgsreal;
		vds = -vdsreal;
		here->MESmode = -1;

		/* source-drain  interchange */

		}
		else 
			here->MESmode = 1;
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
    d_p.value = vgs;  d_r.value = vds;

    /* p =vgs; q= nothing in particular ; r = vds */

                vgst = vgs-model->MESthreshold;
		EqualDeriv(&d_vgst,&d_p);
		d_vgst.value = vgst;
                /*
                 *   normal mode, cutoff region 
                 */
                if (vgst <= 0) {
                    cdrain = 0;
		    EqualDeriv(&d_cdrain,&d_zero);
                } else {
                    prod = 1 + model->MESlModulation * vds;
		    TimesDeriv(&d_prod,&d_r,model->MESlModulation);
		    d_prod.value = prod;
                    betap = beta * prod;
		    TimesDeriv(&d_betap,&d_prod,beta);
                    denom = 1 + model->MESb * vgst;
		    TimesDeriv(&d_denom,&d_vgst,model->MESb);
		    d_denom.value = denom;
                    invdenom = 1 / denom;
		    InvDeriv(&d_invdenom,&d_denom);
                            /*
                             *   normal mode, saturation region 
                             */
                        cdrain = betap * vgst * vgst * invdenom;
			MultDeriv(&d_cdrain,&d_betap,&d_vgst);
			MultDeriv(&d_cdrain,&d_cdrain,&d_vgst);
			MultDeriv(&d_cdrain,&d_cdrain,&d_invdenom);

                    if (vds < ( 3 / model->MESalpha ) ) {
                        /*
                         *   normal mode, linear region 
                         */
                        afact = 1 - model->MESalpha * vds / 3;
			TimesDeriv(&d_afact,&d_r,-model->MESalpha/3.0);
			d_afact.value = afact;
                        lfact = 1 - afact * afact * afact;
			CubeDeriv(&d_lfact,&d_afact);
			TimesDeriv(&d_lfact,&d_lfact,-1.0);
			d_lfact.value += 1.0;
			cdrain = betap*vgst*vgst*invdenom*lfact;
			MultDeriv(&d_cdrain,&d_betap,&d_vgst);
			MultDeriv(&d_cdrain,&d_cdrain,&d_vgst);
			MultDeriv(&d_cdrain,&d_cdrain,&d_invdenom);
			MultDeriv(&d_cdrain,&d_cdrain,&d_lfact);
                    }
                }

		/* converting back to real vgs, vgd, vds */

		if (here->MESmode == -1) {
			vgs = vgsreal;
			vgd = vgdreal;
			vds = vdsreal;
			}
		}
                /* 
                 *    charge storage elements 
                 */
{ /* code block */
czgs = model->MEScapGS * here->MESm * here->MESarea;
czgd = model->MEScapGD * here->MESm * here->MESarea;
phib = model->MESgatePotential;
vcap = 1 / model->MESalpha;

/*
 * qgga = qggnew(vgs,vgd,phib,vcap,vto,czgs,czgd,&cgsna,&cgdna);
 */
/* function qggnew  - private, used by MESload*/
{
    double veroot,veff1,veff2,del,vnroot,vnew1,vnew3,vmax,ext;
    double qroot,par1,cfact,cplus,cminus;
    Dderivs d_vnroot;
    Dderivs d_cgsnew, d_cgdnew, d_dummy, d_dummy2;
    Dderivs d_ext, d_qroot, d_par1, d_cfact, d_cplus, d_cminus;
    Dderivs d_veroot, d_veff1, d_veff2, d_vnew1, d_vnew3;

/* now p=vgs, q=vgd, r= nothing */

    d_q.value = vgd; d_p.value = vgs; 
    veroot = sqrt( (vgs - vgd) * (vgs - vgd) + vcap*vcap );
    TimesDeriv(&d_veroot,&d_q,-1.0);
    PlusDeriv(&d_veroot,&d_veroot,&d_p);
    MultDeriv(&d_veroot,&d_veroot,&d_veroot);
    d_veroot.value += vcap*vcap;
    SqrtDeriv(&d_veroot,&d_veroot);
    veff1 = 0.5 * (vgs + vgd + veroot);
    PlusDeriv(&d_veff1,&d_veroot,&d_p);
    PlusDeriv(&d_veff1,&d_veff1,&d_q);
    TimesDeriv(&d_veff1,&d_veff1,0.5);
    veff2 = veff1 - veroot;
    TimesDeriv(&d_veff2,&d_veroot,-1.0);
    PlusDeriv(&d_veff2,&d_veff2,&d_veff1);

    del = 0.2;/*const*/
    vnroot = sqrt( (veff1 - vto)*(veff1 - vto) + del * del );
    EqualDeriv(&d_vnroot,&d_veff1);
    d_vnroot.value -= vto;
    MultDeriv(&d_vnroot,&d_vnroot,&d_vnroot);
    d_vnroot.value += del*del;
    SqrtDeriv(&d_vnroot,&d_vnroot);
    vnew1 = 0.5 * (veff1 + vto + vnroot);
    PlusDeriv(&d_vnew1,&d_veff1,&d_vnroot);
    d_vnew1.value += vto;
    TimesDeriv(&d_vnew1,&d_vnew1,0.5);
    vnew3 = vnew1;
    EqualDeriv(&d_vnew3,&d_vnew1);
    vmax = 0.5;/*const*/
    if ( vnew1 < vmax ) {
        ext=0;
	EqualDeriv(&d_ext,&d_zero);
    } else {
        vnew1 = vmax;
	EqualDeriv(&d_vnew1,&d_zero);
	d_vnew1.value = vmax;
        ext = (vnew3 - vmax)/sqrt(1 - vmax/phib);
	EqualDeriv(&d_ext,&d_vnew3);
	d_ext.value -= vmax;
	TimesDeriv(&d_ext,&d_ext,1/sqrt(1 - vmax/phib));
    }

    qroot = sqrt(1 - vnew1/phib);
    TimesDeriv(&d_qroot,&d_vnew1,-1/phib);
    d_qroot.value += 1.0;
    SqrtDeriv(&d_qroot,&d_qroot);
    /*
     * qggval = czgs * (2*phib*(1-qroot) + ext) + czgd*veff2;
     */
    par1 = 0.5 * ( 1 + (veff1-vto)/vnroot);
    EqualDeriv(&d_par1,&d_veff1);
    d_par1.value -= vto;
    DivDeriv(&d_par1,&d_par1,&d_vnroot);
    d_par1.value += 1.0;
    TimesDeriv(&d_par1,&d_par1,0.5);
    cfact = (vgs- vgd)/veroot;
    TimesDeriv(&d_cfact,&d_q,-1.0);
    PlusDeriv(&d_cfact,&d_cfact,&d_p);
    DivDeriv(&d_cfact,&d_cfact,&d_veroot);
    cplus = 0.5 * (1 + cfact);
    TimesDeriv(&d_cplus,&d_cfact,0.5);
    d_cplus.value += 0.5;
    cminus = cplus - cfact;
    TimesDeriv(&d_cminus,&d_cfact,-0.5);
    d_cminus.value += 0.5;
    /*
     *cgsnew = czgs/qroot*par1*cplus + czgd*cminus;
     *cgdnew = czgs/qroot*par1*cminus + czgd*cplus;
     *
     * assuming qgs = vgs*cgsnew
     * and      qgd = vgd*cgsnew
     *
     * This is probably wrong but then so is the a.c. analysis 
     * routine and everything else
     *
     */

     MultDeriv(&d_dummy,&d_qroot,&d_par1);
     InvDeriv(&d_dummy,&d_dummy);
     TimesDeriv(&d_dummy,&d_dummy,czgs);

     TimesDeriv(&d_cgsnew,&d_cminus,czgd);
     MultDeriv(&d_dummy2,&d_dummy,&d_cplus);
     PlusDeriv(&d_cgsnew,&d_cgsnew,&d_dummy2);

     TimesDeriv(&d_cgdnew,&d_cplus,czgd);
     MultDeriv(&d_dummy2,&d_dummy,&d_cminus);
     PlusDeriv(&d_cgdnew,&d_cgdnew,&d_dummy2);

     MultDeriv(&d_qgs,&d_cgsnew,&d_p);
     MultDeriv(&d_qgd,&d_cgdnew,&d_q);
}
}


	if (here->MESmode == 1)
		{
		/* normal mode - no source-drain interchange */
here->cdr_x = d_cdrain.d1_p;
here->cdr_z = d_cdrain.d1_r;
here->cdr_x2 = d_cdrain.d2_p2;
here->cdr_z2 = d_cdrain.d2_r2;
here->cdr_xz = d_cdrain.d2_pr;
here->cdr_x3 = d_cdrain.d3_p3;
here->cdr_z3 = d_cdrain.d3_r3;;
here->cdr_x2z = d_cdrain.d3_p2r;
here->cdr_xz2 = d_cdrain.d3_pr2;

} else {
		/*
		 * inverse mode - source and drain interchanged
		 */


here->cdr_x = -d_cdrain.d1_p;
here->cdr_z = d_cdrain.d1_p + d_cdrain.d1_r;
here->cdr_x2 = -d_cdrain.d2_p2;
here->cdr_z2 = -(d_cdrain.d2_p2 + d_cdrain.d2_r2 + 2*d_cdrain.d2_pr);
here->cdr_xz = d_cdrain.d2_p2 + d_cdrain.d2_pr;
here->cdr_x3 = -d_cdrain.d3_p3;
here->cdr_z3 = d_cdrain.d3_p3 + d_cdrain.d3_r3 + 3*(d_cdrain.d3_p2r + d_cdrain.d3_pr2 ) ;
here->cdr_x2z = d_cdrain.d3_p3 + d_cdrain.d3_p2r;
here->cdr_xz2 = -(d_cdrain.d3_p3 + 2*d_cdrain.d3_p2r + d_cdrain.d3_pr2);

}

/* now to adjust for type and multiply by factors to convert to Taylor coeffs. */

here->cdr_x2 = 0.5*model->MEStype*here->cdr_x2;
here->cdr_z2 = 0.5*model->MEStype*here->cdr_z2;
here->cdr_xz = model->MEStype*here->cdr_xz;
here->cdr_x3 = here->cdr_x3/6.;
here->cdr_z3 = here->cdr_z3/6.;
here->cdr_x2z = 0.5*here->cdr_x2z;
here->cdr_xz2 = 0.5*here->cdr_xz2;


here->ggs3 = lggs3;
here->ggd3 = lggd3;
here->ggs2 = model->MEStype*lggs2;
here->ggd2 = model->MEStype*lggd2;

here->qgs_x2 = 0.5*model->MEStype*d_qgs.d2_p2;
here->qgs_y2 = 0.5*model->MEStype*d_qgs.d2_q2;
here->qgs_xy = model->MEStype*d_qgs.d2_pq;
here->qgs_x3 = d_qgs.d3_p3/6.;
here->qgs_y3 = d_qgs.d3_q3/6.;
here->qgs_x2y = 0.5*d_qgs.d3_p2q;
here->qgs_xy2 = 0.5*d_qgs.d3_pq2;

here->qgd_x2 = 0.5*model->MEStype*d_qgd.d2_p2;
here->qgd_y2 = 0.5*model->MEStype*d_qgd.d2_q2;
here->qgd_xy = model->MEStype*d_qgd.d2_pq;
here->qgd_x3 = d_qgd.d3_p3/6.;
here->qgd_y3 = d_qgd.d3_q3/6.;
here->qgd_x2y = 0.5*d_qgd.d3_p2q;
here->qgd_xy2 = 0.5*d_qgd.d3_pq2;
        }
    }
    return(OK);
}
