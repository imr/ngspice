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
#include "ngspice/suffix.h"
#include "soi3defs.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"


void
SOI3cap(double vgB, double Phiplusvsb, double gammaB,
           double paramargs[10],
           double Bfargs[2], double alpha_args[5], double psi_st0args[5],
           double vGTargs[5],
           double psi_sLargs[5], double psi_s0args[5],
           double ldargs[5],
           double *Qg, double *Qb, double *Qd, double *QgB,
           double *cgfgf, double *cgfd, double *cgfs, double *cgfdeltaT,
	   double *cgfgb, double *cbgf, double *cbd, double *cbs, 
	   double *cbdeltaT, double *cbgb, double *cdgf, double *cdd,
	   double *cds, double *cddeltaT, double *cdgb, double *cgbgf,
	   double *cgbd, double *cgbs, double *cgbdeltaT, double *cgbgb
           )
/****** Part 1 - declare local variables.  ******/

{
double WCox,WCob,L;
double gamma,eta_s,vt,delta,sigma,chiFB;
double Bf,pDBf_Dpsi_st0;
double alpha,Dalpha_Dvgfb,Dalpha_Dvdb,Dalpha_Dvsb,Dalpha_DdeltaT;
double Dpsi_st0_Dvgfb,Dpsi_st0_Dvdb,Dpsi_st0_Dvsb,Dpsi_st0_DdeltaT;
double vGT,DvGT_Dvgfb,DvGT_Dvdb,DvGT_Dvsb,DvGT_DdeltaT;
double psi_sL,Dpsi_sL_Dvgfb,Dpsi_sL_Dvdb,Dpsi_sL_Dvsb,Dpsi_sL_DdeltaT;
double psi_s0,Dpsi_s0_Dvgfb,Dpsi_s0_Dvdb,Dpsi_s0_Dvsb,Dpsi_s0_DdeltaT;
double ld,Dld_Dvgfb,Dld_Dvdb,Dld_Dvsb,Dld_DdeltaT;

double Lprime,Fc;
double Qbprime,Qcprime,Qdprime,Qgprime,Qb2prime,Qc2prime,Qd2prime,Qg2prime;
double Dlimc,Dlimd;
double Vqd,Vqs;
double F,F2;
double cq,dq;
double dercq,derdq;
double sigmaC,Eqc,Eqd;
double DVqs_Dvgfb,DVqs_Dvdb,DVqs_Dvsb,DVqs_DdeltaT;
double DF_Dvgfb,DF_Dvdb,DF_Dvsb,DF_DdeltaT;
double ccgf,ccd,ccs,ccdeltaT,ccgb;


double vg,vgacc,Egacc,tmpacc,Qacc;
double csf;

NG_IGNORE(gammaB);
NG_IGNORE(Phiplusvsb);
NG_IGNORE(vgB);

/****** Part 2 - extract variables passed from soi3load(), which  ******/
/****** have been passed to soi3cap() in *arg arrays.             ******/

WCox  = paramargs[0];
L     = paramargs[1];
gamma = paramargs[2];
eta_s = paramargs[3];
vt    = paramargs[4];
delta = paramargs[5];
WCob  = paramargs[6];
sigma = paramargs[7];
chiFB = paramargs[8];
csf   = paramargs[9];
Bf            = Bfargs[0];
pDBf_Dpsi_st0 = Bfargs[1];
alpha          = alpha_args[0];
Dalpha_Dvgfb   = alpha_args[1];
Dalpha_Dvdb    = alpha_args[2];
Dalpha_Dvsb    = alpha_args[3];
Dalpha_DdeltaT = alpha_args[4];

Dpsi_st0_Dvgfb   = psi_st0args[1];
Dpsi_st0_Dvdb    = psi_st0args[2];
Dpsi_st0_Dvsb    = psi_st0args[3];
Dpsi_st0_DdeltaT = psi_st0args[4];

vGT          = vGTargs[0];
DvGT_Dvgfb   = vGTargs[1];
DvGT_Dvdb    = vGTargs[2];
DvGT_Dvsb    = vGTargs[3];
DvGT_DdeltaT = vGTargs[4];

psi_sL          = psi_sLargs[0];
Dpsi_sL_Dvgfb   = psi_sLargs[1];
Dpsi_sL_Dvdb    = psi_sLargs[2];
Dpsi_sL_Dvsb    = psi_sLargs[3];
Dpsi_sL_DdeltaT = psi_sLargs[4];
psi_s0          = psi_s0args[0];
Dpsi_s0_Dvgfb   = psi_s0args[1];
Dpsi_s0_Dvdb    = psi_s0args[2];
Dpsi_s0_Dvsb    = psi_s0args[3];
Dpsi_s0_DdeltaT = psi_s0args[4];
ld          = ldargs[0];
Dld_Dvgfb   = ldargs[1];
Dld_Dvdb    = ldargs[2];
Dld_Dvsb    = ldargs[3];
Dld_DdeltaT = ldargs[4];


/****** Part 3 - define some important quantities.  ******/

sigmaC = 1E-8;

Vqd = (vGT - alpha*psi_sL); /* This is -qd/Cof */
Vqs = (vGT - alpha*psi_s0); /* This is -qs/Cof */
if (Vqs<=0)
{ /* deep subthreshold contingency */
  F = 1;
} else
{
  F = Vqd/Vqs;
  if (F<0)
  { /* physically impossible situation */
    F=0;
  }
}
F2 = F*F;

Fc = 1 + ld/L;
Lprime = L/Fc;


/****** Part 4 - calculate normalised (see note below) terminal  ******/
/****** charge expressions for the GCA region.                   ******/

/* JimB - important note */
/* The charge expressions Qcprime, Qd2prime etc in this file are not charges */
/* but voltages! Each expression is equal to the derived expression for the  */
/* total charge in each region, but divided by a factor WL'Cof.  This is     */
/* compensated for later on. */

/* Channel charge Qc1 */
cq = (F*F + F + 1)/(F+1);
Qcprime = -2*Vqs*cq/3;

if ((-Qcprime/sigmaC)<MAX_EXP_ARG) {
  Eqc = exp(-Qcprime/sigmaC);
  Qcprime = -sigmaC*log(1 + Eqc);
  Dlimc = Eqc/(1+Eqc);
} else {
  Dlimc = 1;
}

/* Drain charge Qd1 */
dq = (3*F2*F + 6*F2 + 4*F + 2)/((1+F)*(1+F));
Qdprime = -2*Vqs*dq/15;

if((-Qdprime/sigmaC)<MAX_EXP_ARG) {
  Eqd = exp(-Qdprime/sigmaC);
  Qdprime = -sigmaC*log(1 + Eqd);
  Dlimd = Eqd/(1+Eqd);
} else {
  Dlimd = 1;
}

/* Body charge Qb1 */
Qbprime = -gamma*(Bf + (delta/alpha)*(vGT + Qcprime));

/* Gate charge Qg1 */
Qgprime = -Qcprime-Qbprime;


/****** Part 5 - calculate capacitances and transcapacitances  ******/
/****** for the GCA region.  For the moment, we are not taking ******/
/****** account of the bias dependence of ld and Lprime.  This ******/
/****** will be done in Part 8, when both GCA and drain charge ******/
/****** terms will be included in the final capacitance        ******/
/****** expressions.                                           ******/

DVqs_Dvgfb = DvGT_Dvgfb - alpha*Dpsi_s0_Dvgfb - psi_s0*Dalpha_Dvgfb;
DVqs_Dvdb = DvGT_Dvdb - alpha*Dpsi_s0_Dvdb - psi_s0*Dalpha_Dvdb;
DVqs_Dvsb = DvGT_Dvsb - alpha*Dpsi_s0_Dvsb - psi_s0*Dalpha_Dvsb;
DVqs_DdeltaT = DvGT_DdeltaT - alpha*Dpsi_s0_DdeltaT - psi_s0*Dalpha_DdeltaT;

if (Vqs==0)
{
  DF_Dvgfb = 0;
  DF_Dvdb = 0;
  DF_Dvsb = 0;
  DF_DdeltaT = 0;
}
else
{
  DF_Dvgfb = (DvGT_Dvgfb - alpha*Dpsi_sL_Dvgfb - psi_sL*Dalpha_Dvgfb -
              F*DVqs_Dvgfb)/Vqs;
  DF_Dvdb = (DvGT_Dvdb - alpha*Dpsi_sL_Dvdb - psi_sL*Dalpha_Dvdb -
              F*DVqs_Dvdb)/Vqs;
  DF_Dvsb = (DvGT_Dvsb - alpha*Dpsi_sL_Dvsb - psi_sL*Dalpha_Dvsb -
              F*DVqs_Dvsb)/Vqs;
  DF_DdeltaT = (DvGT_DdeltaT - alpha*Dpsi_sL_DdeltaT - psi_sL*Dalpha_DdeltaT -
              F*DVqs_DdeltaT)/Vqs;
}

dercq = F*(2+F)/((1+F)*(1+F));

ccgf = Dlimc*(-2*(DVqs_Dvgfb*cq + Vqs*dercq*DF_Dvgfb)/3);
ccd  = Dlimc*(-2*(DVqs_Dvdb*cq + Vqs*dercq*DF_Dvdb)/3);
ccs  = Dlimc*(-2*(DVqs_Dvsb*cq + Vqs*dercq*DF_Dvsb)/3);
ccdeltaT  = Dlimc*(-2*(DVqs_DdeltaT*cq + Vqs*dercq*DF_DdeltaT)/3);
ccgb = 0;

derdq = F*(3*F2 + 9*F + 8)/((1+F)*(1+F)*(1+F));

*cdgf = Dlimd*(-2*(DVqs_Dvgfb * dq + Vqs*derdq*DF_Dvgfb)/15);
*cdd  = Dlimd*(-2*(DVqs_Dvdb * dq + Vqs*derdq*DF_Dvdb)/15);
*cds  = Dlimd*(-2*(DVqs_Dvsb * dq + Vqs*derdq*DF_Dvsb)/15);
*cddeltaT  = Dlimd*(-2*(DVqs_DdeltaT * dq + Vqs*derdq*DF_DdeltaT)/15);
*cdgb = 0;

/* JimB - note that for the following expressions, the Vx dependence of */
/* delta is accounted for by the term (vGT+Qcprime)*(Dalpha_Dvx/gamma). */

*cbgf = -gamma * (pDBf_Dpsi_st0*Dpsi_st0_Dvgfb +
                 (alpha*(delta*(DvGT_Dvgfb + ccgf) +
                    (vGT+Qcprime)*(Dalpha_Dvgfb/gamma)) -
                  delta*(vGT+Qcprime)*Dalpha_Dvgfb
                 )/(alpha*alpha)
                );
*cbd = -gamma * (pDBf_Dpsi_st0*Dpsi_st0_Dvdb +
                 (alpha*(delta*(DvGT_Dvdb + ccd) +
                    (vGT+Qcprime)*(Dalpha_Dvdb/gamma)) -
                  delta*(vGT+Qcprime)*Dalpha_Dvdb
                 )/(alpha*alpha)
                );
*cbs = -gamma * (pDBf_Dpsi_st0*Dpsi_st0_Dvsb +
                 (alpha*(delta*(DvGT_Dvsb + ccs) +
                    (vGT+Qcprime)*(Dalpha_Dvsb/gamma)) -
                  delta*(vGT+Qcprime)*Dalpha_Dvsb
                 )/(alpha*alpha)
                );
*cbdeltaT = -gamma * (pDBf_Dpsi_st0*Dpsi_st0_DdeltaT +
                 (alpha*(delta*(DvGT_DdeltaT + ccdeltaT) +
                    (vGT+Qcprime)*(Dalpha_DdeltaT/gamma)) -
                  delta*(vGT+Qcprime)*Dalpha_DdeltaT
                 )/(alpha*alpha)
                );
*cbgb = 0;


/****** Part 6 - Normalised expressions from part 4 are adjusted  ******/
/****** by WCox*Lprime, then accumulation charge is added to give ******/
/****** final expression for GCA region charges.                  ******/

/* Accumulation charge to be added to Qb */

vg = vGT + gamma*Bf;
if ((-vg/vt) > MAX_EXP_ARG) {
  vgacc = vg;
  tmpacc = 1;
} else {
  Egacc = exp(-vg/vt);
  vgacc = -vt*log(1+Egacc);
  tmpacc = Egacc/(1+Egacc);
}
Qacc = -WCox*L*vgacc;

/* Now work out GCA region charges */

*Qb = WCox*Lprime*Qbprime + Qacc;

*Qd = WCox*Lprime*Qdprime;

*Qg = WCox*Lprime*Qgprime - Qacc;


/****** Part 7 - calculate normalised (see note below) terminal  ******/
/****** charge expressions for the saturated drain region.       ******/

Qc2prime = -Vqd;

/* Basic expression for the intrinsic body charge in the saturation region is */
/* modified by csf, to reflect the fact that the body charge will be shared   */
/* between the gate and the drain/body depletion region.  This factor must be */
/* between 0 and 1, since it represents the fraction of ld over which qb is   */
/* integrated to give Qb2. */
Qb2prime = -gamma*csf*(Bf + delta*psi_sL);

Qd2prime = 0.5*Qc2prime;
/* JimB - 9/1/99. Re-partition drain region charge */
/* Qd2prime = Qc2prime; */

Qg2prime = -Qc2prime-Qb2prime;

*Qb += WCox*ld*Qb2prime;
*Qd += WCox*ld*Qd2prime;
*Qg += WCox*ld*Qg2prime;


/****** Part 8 - calculate full capacitance expressions, accounting ******/
/****** for both GCA and drain region contributions.  As explained  ******/
/****** in part 5, *cbgf, *cbd etc only derivatives of GCA charge   ******/
/****** expression w.r.t. Vx.  Now need to include Lprime/ld        ******/
/****** dependence on Vx as well.                                   ******/

*cbgf = WCox*(Lprime*(*cbgf) - ld*csf*(pDBf_Dpsi_st0*Dpsi_st0_Dvgfb + delta*Dpsi_sL_Dvgfb +
					(psi_sL*Dalpha_Dvgfb/gamma)) +
					(Qb2prime - Qbprime/(Fc*Fc))*Dld_Dvgfb
               );
*cbd  = WCox*(Lprime*(*cbd) - ld*csf*(pDBf_Dpsi_st0*Dpsi_st0_Dvdb + delta*Dpsi_sL_Dvdb +
					(psi_sL*Dalpha_Dvdb/gamma)) +
					(Qb2prime - Qbprime/(Fc*Fc))*Dld_Dvdb
               );
*cbs  = WCox*(Lprime*(*cbs) - ld*csf*(pDBf_Dpsi_st0*Dpsi_st0_Dvsb + delta*Dpsi_sL_Dvsb +
					(psi_sL*Dalpha_Dvsb/gamma)) +
					(Qb2prime - Qbprime/(Fc*Fc))*Dld_Dvsb
               );
*cbdeltaT  = WCox*(Lprime*(*cbdeltaT) - ld*csf*(pDBf_Dpsi_st0*Dpsi_st0_DdeltaT + delta*Dpsi_sL_DdeltaT +
					(psi_sL*Dalpha_DdeltaT/gamma)) +
					(Qb2prime - Qbprime/(Fc*Fc))*Dld_DdeltaT
               );
*cbgb = 0;

               
ccgf = WCox*(Lprime*(ccgf) - ld*(DvGT_Dvgfb - alpha*Dpsi_sL_Dvgfb - psi_sL*Dalpha_Dvgfb) +
              (Qc2prime - Qcprime/(Fc*Fc))*Dld_Dvgfb
             );
ccd  = WCox*(Lprime*(ccd) - ld*(DvGT_Dvdb - alpha*Dpsi_sL_Dvdb - psi_sL*Dalpha_Dvdb) +
              (Qc2prime - Qcprime/(Fc*Fc))*Dld_Dvdb
             );
ccs  = WCox*(Lprime*(ccs) - ld*(DvGT_Dvsb - alpha*Dpsi_sL_Dvsb - psi_sL*Dalpha_Dvsb) +
              (Qc2prime - Qcprime/(Fc*Fc))*Dld_Dvsb
             );
ccdeltaT  = WCox*(Lprime*(ccdeltaT) - 
              ld*(DvGT_DdeltaT - alpha*Dpsi_sL_DdeltaT - psi_sL*Dalpha_DdeltaT) +
              (Qc2prime - Qcprime/(Fc*Fc))*Dld_DdeltaT
             );
ccgb = 0;


*cdgf = WCox*(Lprime*(*cdgf) - 0.5*ld*(DvGT_Dvgfb - alpha*Dpsi_sL_Dvgfb - psi_sL*Dalpha_Dvgfb) +
              (Qd2prime - Qdprime/(Fc*Fc))*Dld_Dvgfb
             );
*cdd  = WCox*(Lprime*(*cdd) - 0.5*ld*(DvGT_Dvdb - alpha*Dpsi_sL_Dvdb - psi_sL*Dalpha_Dvdb) +
              (Qd2prime - Qdprime/(Fc*Fc))*Dld_Dvdb
             );
*cds  = WCox*(Lprime*(*cds) - 0.5*ld*(DvGT_Dvsb - alpha*Dpsi_sL_Dvsb - psi_sL*Dalpha_Dvsb) +
              (Qd2prime - Qdprime/(Fc*Fc))*Dld_Dvsb
             );
*cddeltaT  = WCox*(Lprime*(*cddeltaT) - 0.5*ld*(DvGT_DdeltaT -
                    alpha*Dpsi_sL_DdeltaT - psi_sL*Dalpha_DdeltaT) +
              (Qd2prime - Qdprime/(Fc*Fc))*Dld_DdeltaT
             );
*cdgb = 0;

/****** Part 9 - Finally, include accumulation charge derivatives. ******/

/* Now include accumulation charge derivs */

*cbgf += -WCox*L*tmpacc;
*cbd  += -WCox*L*tmpacc*sigma;
*cbs  += -WCox*L*tmpacc*(-sigma);
*cbdeltaT += -WCox*L*tmpacc*chiFB;
*cbgb += 0;

*cgfgf = -(ccgf + *cbgf);
*cgfd = -(ccd + *cbd);
*cgfs = -(ccs + *cbs);
*cgfdeltaT = -(ccdeltaT + *cbdeltaT);
*cgfgb = 0;

/****** Part 10 - Back gate stuff - doesn't work, so set to zero.  ******/

/* Should move this before the accumulation section for consistency, but */
/* doesn't matter, as all intrinsic back-gate capacitances set to zero
anyway. */

*QgB = 0;

*cgbgf = 0;
*cgbd = 0;
*cgbs = 0;
*cgbgb = 0;
*cgbdeltaT = 0;

}

void
SOI3capEval(CKTcircuit *ckt,
            double Frontcapargs[6],
            double Backcapargs[6],
            double cgfgf, double cgfd, double cgfs, double cgfdeltaT, double cgfgb,
            double cdgf, double cdd, double cds, double cddeltaT, double cdgb,
            double csgf, double csd, double css, double csdeltaT, double csgb,
            double cbgf, double cbd, double cbs, double cbdeltaT, double cbgb,
            double cgbgf, double cgbd, double cgbs, double cgbdeltaT, double cgbgb,
            
	    double *gcgfgf, double *gcgfd, double *gcgfs, double *gcgfdeltaT, double *gcgfgb,
            double *gcdgf, double *gcdd, double *gcds, double *gcddeltaT, double *gcdgb,
            double *gcsgf, double *gcsd, double *gcss, double *gcsdeltaT, double *gcsgb,
            double *gcbgf, double *gcbd, double *gcbs, double *gcbdeltaT, double *gcbgb,
            double *gcgbgf, double *gcgbd, double *gcgbs, double *gcgbdeltaT, double *gcgbgb,
            double *qgatef, double *qbody, double *qdrn, double *qsrc, double *qgateb)

{
double cgfd0,cgfs0,cgfb0;
double cgbd0,cgbs0,cgbb0;
double vgfd,vgfs,vgfb;
double vgbd,vgbs,vgbb;
double ag0;
double qgfd,qgfs,qgfb;
double qgbd,qgbs,qgbb;

cgfd0 = Frontcapargs[0];
cgfs0 = Frontcapargs[1];
cgfb0 = Frontcapargs[2];
vgfd = Frontcapargs[3];
vgfs = Frontcapargs[4];
vgfb = Frontcapargs[5];

cgbd0 = Backcapargs[0];
cgbs0 = Backcapargs[1];
cgbb0 = Backcapargs[2];
vgbd = Backcapargs[3];
vgbs = Backcapargs[4];
vgbb = Backcapargs[5];

/* stuff below includes overlap caps' conductances */
ag0 = ckt->CKTag[0];

*gcgfgf = (cgfgf + cgfd0 + cgfs0 + cgfb0) * ag0;
*gcgfd  = (cgfd - cgfd0) * ag0;
*gcgfs  = (cgfs - cgfs0) * ag0;
*gcgfdeltaT = cgfdeltaT * ag0;
*gcgfgb = cgfgb * ag0;

*gcdgf = (cdgf - cgfd0) * ag0;
*gcdd  = (cdd + cgfd0 + cgbd0) * ag0;
*gcds  = cds * ag0;
*gcddeltaT = cddeltaT * ag0;
*gcdgb = (cdgb - cgbd0) * ag0;

*gcsgf = (csgf - cgfs0) * ag0;
*gcsd  = csd * ag0;
*gcss  = (css + cgfs0 + cgbs0) * ag0;
*gcsdeltaT = csdeltaT * ag0;
*gcsgb = (csgb - cgbs0) * ag0;

*gcbgf = (cbgf - cgfb0) * ag0;
*gcbd  = cbd * ag0;
*gcbs  = cbs * ag0;
*gcbdeltaT = cbdeltaT * ag0;
*gcbgb = (cbgb - cgbb0) * ag0;

*gcgbgf = cgbgf * ag0;
*gcgbd  = (cgbd - cgbd0) * ag0;
*gcgbs  = (cgbs - cgbs0) * ag0;
*gcgbdeltaT = cgbdeltaT * ag0;
*gcgbgb = (cgbgb + cgbd0 + cgbs0 + cgbb0) * ag0;

qgfd = cgfd0 * vgfd;
qgfs = cgfs0 * vgfs;
qgfb = cgfb0 * vgfb;

qgbd = cgbd0 * vgbd;
qgbs = cgbs0 * vgbs;
qgbb = cgbb0 * vgbb;

*qgatef = *qgatef + qgfd + qgfs + qgfb;
*qbody  = *qbody - qgfb - qgbb;
*qdrn   = *qdrn - qgfd - qgbd;
*qgateb = *qgateb + qgbd + qgbs + qgbb;
*qsrc = -(*qgatef + *qbody + *qdrn + *qgateb);
}
