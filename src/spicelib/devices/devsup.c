/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

/* support routines for device models */

#include "ngspice.h"
#include "devdefs.h"
#include "cktdefs.h"
#include "suffix.h"

/* 
 * Limit the per-iteration change of VDS 
 */
double
DEVlimvds(double vnew, double vold)
{

    if(vold >= 3.5) {
        if(vnew > vold) {
            vnew = MIN(vnew,(3 * vold) +2);
        } else {
            if (vnew < 3.5) {
                vnew = MAX(vnew,2);
            }
        }
    } else {
        if(vnew > vold) {
            vnew = MIN(vnew,4);
        } else {
            vnew = MAX(vnew,-.5);
        }
    }
    return(vnew);
}


/*  
 * Limit the per-iteration change of PN junction voltages 
 *
 * This code has been fixed by Alan Gillespie adding limiting
 * for negative voltages.
 */
double
DEVpnjlim(double vnew, double vold, double vt, double vcrit, int *icheck)
{
    double arg;

    if((vnew > vcrit) && (fabs(vnew - vold) > (vt + vt))) {
        if(vold > 0) {
            arg = (vnew - vold) / vt;
            if(arg > 0) {
                vnew = vold + vt * (2+log(arg-2));
            } else {
                vnew = vold - vt * (2+log(2-arg));
            }
        } else {
            vnew = vt *log(vnew/vt);
        }
        *icheck = 1;
    } else {
       if (vnew < 0) {
           if (vold > 0) {
               arg = -1*vold-1;
           } else {
               arg = 2*vold-1;
           }
           if (vnew < arg) {
              vnew = arg;
              *icheck = 1;
           } else {
              *icheck = 0;
           };
        } else {
           *icheck = 0;
        }
    }
    return(vnew);
}

/* 
 * Limit the per-iteration change of FET voltages 
 *
 * This code has been fixed by Alan Gillespie: a new
 * definition for vtstlo. 
 */
double
DEVfetlim(double vnew, double vold, double vto)
{
    double vtsthi;
    double vtstlo;
    double vtox;
    double delv;
    double vtemp;

    vtsthi = fabs(2*(vold-vto))+2;
    vtstlo = fabs(vold-vto)+1;
    vtox = vto + 3.5;
    delv = vnew-vold;

    if (vold >= vto) {
        if(vold >= vtox) {
            if(delv <= 0) {
                /* going off */
                if(vnew >= vtox) {
                    if(-delv >vtstlo) {
                        vnew =  vold - vtstlo;
                    }
                } else {
                    vnew = MAX(vnew,vto+2);
                }
            } else {
                /* staying on */
                if(delv >= vtsthi) {
                    vnew = vold + vtsthi;
                }
            }
        } else {
            /* middle region */
            if(delv <= 0) {
                /* decreasing */
                vnew = MAX(vnew,vto-.5);
            } else {
                /* increasing */
                vnew = MIN(vnew,vto+4);
            }
        }
    } else {
        /* off */
        if(delv <= 0) {
            if(-delv >vtsthi) {
                vnew = vold - vtsthi;
            } 
        } else {
            vtemp = vto + .5;
            if(vnew <= vtemp) {
                if(delv >vtstlo) {
                    vnew = vold + vtstlo;
                }
            } else {
                vnew = vtemp;
            }
        }
    }
    return(vnew);
}


/* Compute the MOS overlap capacitances as functions of the device
 * terminal voltages
 *
 * PN 2002: As of ngspice this code is not used by any device. 
 */
void
DEVcmeyer(double vgs0,		/* initial voltage gate-source */
	  double vgd0,		/* initial voltage gate-drain */
	  double vgb0,		/* initial voltage gate-bulk */
	  double von0,
	  double vdsat0,
	  double vgs1,		/* final voltage gate-source */
	  double vgd1,		/* final voltage gate-drain */
	  double vgb1,		/* final voltage gate-bulk */
	  double covlgs,	/* overlap capacitance gate-source */
	  double covlgd,	/* overlap capacitance gate-drain */
	  double covlgb,	/* overlap capacitance gate-bulk */
	  double *cgs,
	  double *cgd,
	  double *cgb,
	  double phi,
	  double cox,
	  double von,
	  double vdsat)
{


    double vdb;
    double vdbsat;
    double vddif;
    double vddif1;
    double vddif2;
    double vgbt;

    *cgs = 0;
    *cgd = 0;
    *cgb = 0;

    vgbt = vgs1-von;
    if (vgbt <= -phi) {
        *cgb = cox;
    } else if (vgbt <= -phi/2) {
        *cgb = -vgbt*cox/phi;
    } else if (vgbt <= 0) {
        *cgb = -vgbt*cox/phi;
        *cgs = cox/(7.5e-1*phi)*vgbt+cox/1.5;
    } else {
        vdbsat = vdsat-(vgs1-vgb1);
        vdb = vgb1-vgd1;
        if (vdbsat <= vdb) {
            *cgs = cox/1.5;
        } else {
            vddif = 2.0*vdbsat-vdb;
            vddif1 = vdbsat-vdb-1.0e-12;
            vddif2 = vddif*vddif;
            *cgd = cox*(1.0-vdbsat*vdbsat/vddif2)/1.5;
            *cgs = cox*(1.0-vddif1*vddif1/vddif2)/1.5;
        }
    }

    vgbt = vgs0-von0;
    if (vgbt <= -phi) {
        *cgb += cox;
    } else if (vgbt <= -phi/2) {
        *cgb += -vgbt*cox/phi;
    } else if (vgbt <= 0) {
        *cgb += -vgbt*cox/phi;
        *cgs += cox/(7.5e-1*phi)*vgbt+cox/1.5;
    } else  {
        vdbsat = vdsat0-(vgs0-vgb0);
        vdb = vgb0-vgd0;
        if (vdbsat <= vdb) {
            *cgs += cox/1.5;
        } else {
            vddif = 2.0*vdbsat-vdb;
            vddif1 = vdbsat-vdb-1.0e-12;
            vddif2 = vddif*vddif;
            *cgd += cox*(1.0-vdbsat*vdbsat/vddif2)/1.5;
            *cgs += cox*(1.0-vddif1*vddif1/vddif2)/1.5;
        }
    }

    *cgs = *cgs *.5 + covlgs;
    *cgd = *cgd *.5 + covlgd;
    *cgb = *cgb *.5 + covlgb;
}


/* Compute the MOS overlap capacitances as functions of the device
 * terminal voltages 
 *
 * PN 2002: This is the Meyer model used by  MOS1 MOS2 MOS3 MOS6 and MOS9
 *          device models.
 */
void
DEVqmeyer(double vgs,		/* initial voltage gate-source */
	  double vgd,		/* initial voltage gate-drain */
	  double vgb,		/* initial voltage gate-bulk */
	  double von,
	  double vdsat,
	  double *capgs,	/* non-constant portion of g-s overlap
                                   capacitance */
	  double *capgd,	/* non-constant portion of g-d overlap
                                   capacitance */
	  double *capgb,	/* non-constant portion of g-b overlap
                                   capacitance */
	  double phi,
	  double cox)		/* oxide capactiance */
{
    double vds;
    double vddif;
    double vddif1;
    double vddif2;
    double vgst;

#define MAGIC_VDS 0.025

    vgst = vgs-von;
    vdsat = MAX(vdsat, MAGIC_VDS);
    if (vgst <= -phi) {
        *capgb = cox/2;
        *capgs = 0;
        *capgd = 0;
    } else if (vgst <= -phi/2) {
        *capgb = -vgst*cox/(2*phi);
        *capgs = 0;
        *capgd = 0;
    } else if (vgst <= 0) {
        *capgb = -vgst*cox/(2*phi);
        *capgs = vgst*cox/(1.5*phi)+cox/3;
        vds = vgs-vgd;
        if (vds>=vdsat) {
           *capgd = 0;
        } else {
            vddif  = 2.0*vdsat-vds;
            vddif1 = vdsat-vds/*-1.0e-12*/;
            vddif2 = vddif*vddif;
            *capgd = *capgs*(1.0-vdsat*vdsat/vddif2);
            *capgs = *capgs*(1.0-vddif1*vddif1/vddif2);
        }
    } else  {
        vds = vgs-vgd;
        vdsat = MAX(vdsat, MAGIC_VDS);
        if (vdsat <= vds) {
            *capgs = cox/3;
            *capgd = 0;
            *capgb = 0;
        } else {
            vddif = 2.0*vdsat-vds;
            vddif1 = vdsat-vds/*-1.0e-12*/;
            vddif2 = vddif*vddif;
            *capgd = cox*(1.0-vdsat*vdsat/vddif2)/3;
            *capgs = cox*(1.0-vddif1*vddif1/vddif2)/3;
            *capgb = 0;
        }
    }

}


#ifdef notdef
/* XXX This is no longer used, apparently 
 * PN 2002: This is industrial archaelology
 */
void
DEVcap(CKTcircuit *ckt, double vgd, double vgs, double vgb, double covlgd,
       double covlgs, double covlgb, double capbd, double capbs, double cggb,
       double cgdb, double cgsb, double cbgb, double cbdb, double cbsb,
       double *gcggb, double *gcgdb, double *gcgsb, double *gcbgb, 
       double *gcbdb, double *gcbsb, double *gcdgb, double *gcddb, 
       double *gcdsb, double *gcsgb, double *gcsdb, double *gcssb,
       double qgate, double qchan, double qbulk, double *qdrn, double *qsrc,
       double xqc)

    /*
     *     compute equivalent conductances
     *     divide up the channel charge (1-xqc)/xqc to source and drain
     */
{

    double gcd;
    double gcdxd;
    double gcdxs;
    double gcg;
    double gcgxd;
    double gcgxs;
    double gcs;
    double gcsxd;
    double gcsxs;
    double qgb;
    double qgd;
    double qgs;

    gcg = (cggb+cbgb)*ckt->CKTag[1];
    gcd = (cgdb+cbdb)*ckt->CKTag[1];
    gcs = (cgsb+cbsb)*ckt->CKTag[1];
    gcgxd = -xqc*gcg;
    gcgxs = -(1-xqc)*gcg;
    gcdxd = -xqc*gcd;
    gcdxs = -(1-xqc)*gcd;
    gcsxd = -xqc*gcs;
    gcsxs = -(1-xqc)*gcs;
    *gcdgb = gcgxd-covlgd*ckt->CKTag[1];
    *gcddb = gcdxd+(capbd+covlgd)*ckt->CKTag[1];
    *gcdsb = gcsxd;
    *gcsgb = gcgxs-covlgs*ckt->CKTag[1];
    *gcsdb = gcdxs;
    *gcssb = gcsxs+(capbs+covlgs)*ckt->CKTag[1];
    *gcggb = (cggb+covlgd+covlgs+covlgb)*ckt->CKTag[1];
    *gcgdb = (cgdb-covlgd)*ckt->CKTag[1];
    *gcgsb = (cgsb-covlgs)*ckt->CKTag[1];
    *gcbgb = (cbgb-covlgb)*ckt->CKTag[1];
    *gcbdb = (cbdb-capbd)*ckt->CKTag[1];
    *gcbsb = (cbsb-capbs)*ckt->CKTag[1];
    /*
     *     compute total terminal charges
     */
    qgd = covlgd*vgd;
    qgs = covlgs*vgs;
    qgb = covlgb*vgb;
    qgate = qgate+qgd+qgs+qgb;
    qbulk = qbulk-qgb;
    *qdrn = xqc*qchan-qgd;
    *qsrc = (1-xqc)*qchan-qgs;
    /*
     *     finished
     */
}
#endif


/* Predict a value for the capacitor at loct by extrapolating from
 * previous values */
double
DEVpred(CKTcircuit *ckt, int loct)
{
#ifndef NEWTRUNC
    double xfact;

    xfact = ckt->CKTdelta/ckt->CKTdeltaOld[1];
    return( ( (1+xfact) * *(ckt->CKTstate1+loct) ) -
            (    xfact  * *(ckt->CKTstate2+loct) )  );
#endif /* NEWTRUNC */
}
