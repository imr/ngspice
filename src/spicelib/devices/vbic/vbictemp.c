/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1995 Colin McAndrew Motorola
Spice3 Implementation: 2003 Dietmar Warning DAnalyse GmbH
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"
#include "vbicdefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/ifsim.h"
#include "ngspice/suffix.h"


/* ARGSUSED */

int iret, vbic_4T_et_cf_t(double *, double *, double *);

int
VBICtemp(GENmodel *inModel, CKTcircuit *ckt)
        /* Pre-compute many useful parameters
         */
{
    VBICmodel *model = (VBICmodel *)inModel;
    VBICinstance *here;
    double p[108], pnom[108], TAMB;
    double vt;

    /*  loop through all the bipolar models */
    for( ; model != NULL; model = VBICnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VBICinstances(model); here != NULL ;
                here=VBICnextInstance(here)) {

            if(!here->VBICtempGiven) here->VBICtemp = ckt->CKTtemp;

            if(here->VBICdtempGiven) here->VBICtemp = here->VBICtemp + here->VBICdtemp;

            TAMB = here->VBICtemp - CONSTCtoK;

            memcpy (&pnom, &model->VBICtnom, sizeof(pnom));

            iret = vbic_4T_et_cf_t(p,pnom,&TAMB);

            here->VBICtextCollResist = p[1];
            here->VBICtintCollResist = p[2];
            here->VBICtepiSatVoltage = p[3];
            here->VBICtepiDoping = p[4];
            here->VBICtextBaseResist = p[6];
            here->VBICtintBaseResist = p[7];
            here->VBICtemitterResist = p[8];
            here->VBICtsubstrateResist = p[9];
            here->VBICtparBaseResist = p[10];
            here->VBICtsatCur = p[11];
            here->VBICtemissionCoeffF = p[12];
            here->VBICtemissionCoeffR = p[13];
            here->VBICtdepletionCapBE = p[16];
            here->VBICtpotentialBE = p[17];
            here->VBICtdepletionCapBC = p[21];
            here->VBICtextCapBC = p[23];
            here->VBICtpotentialBC = p[24];
            here->VBICtextCapSC = p[27];
            here->VBICtpotentialSC = p[28];
            here->VBICtidealSatCurBE = p[31];
            here->VBICtnidealSatCurBE = p[34];
            here->VBICtidealSatCurBC = p[36];
            here->VBICtnidealSatCurBC = p[38];
            here->VBICtavalanchePar2BC = p[41];
            here->VBICtparasitSatCur = p[42];
            here->VBICtidealParasitSatCurBE = p[45];
            here->VBICtnidealParasitSatCurBE = p[46];
            here->VBICtidealParasitSatCurBC = p[47];
            here->VBICtnidealParasitSatCurBC = p[49];
            here->VBICtrollOffF = p[53];
            here->VBICtsepISRR = p[94];
            here->VBICtvbbe = p[98];
            here->VBICtnbbe = p[99];

            vt = here->VBICtemp * CONSTKoverQ;
            here->VBICtVcrit = vt *
                     log(vt / (CONSTroot2*model->VBICsatCur*here->VBICarea*here->VBICm));
        }
    }
    return(OK);
}

int vbic_4T_et_cf_t(double *p, double *pnom, double *TAMB)
{
        double Tini, Tdev, Vtv, rT, dT, xvar1;
        double xvar2, xvar3, xvar4, xvar5, xvar6, psiio;
        double psiin;

/*      Direct copy p<-pnom for temperature independent parameters */

        p[5]=pnom[5];
        p[14]=pnom[14];
        p[15]=pnom[15];
        p[18]=pnom[18];
        p[19]=pnom[19];
        p[20]=pnom[20];
        p[22]=pnom[22];
        p[25]=pnom[25];
        p[26]=pnom[26];
        p[29]=pnom[29];
        p[30]=pnom[30];
        p[32]=pnom[32];
        p[33]=pnom[33];
        p[35]=pnom[35];
        p[37]=pnom[37];
        p[39]=pnom[39];
        p[40]=pnom[40];
        p[43]=pnom[43];
        p[44]=pnom[44];
        p[48]=pnom[48];
        p[50]=pnom[50];
        p[51]=pnom[51];
        p[52]=pnom[52];
        p[54]=pnom[54];
        p[55]=pnom[55];
        p[56]=pnom[56];
        p[57]=pnom[57];
        p[58]=pnom[58];
        p[59]=pnom[59];
        p[60]=pnom[60];
        p[61]=pnom[61];
        p[62]=pnom[62];
        p[63]=pnom[63];
        p[64]=pnom[64];
        p[65]=pnom[65];
        p[66]=pnom[66];
        p[67]=pnom[67];
        p[68]=pnom[68];
        p[69]=pnom[69];
        p[70]=pnom[70];
        p[71]=pnom[71];
        p[72]=pnom[72];
        p[73]=pnom[73];
        p[74]=pnom[74];
        p[75]=pnom[75];
        p[76]=pnom[76];
        p[77]=pnom[77];
        p[78]=pnom[78];
        p[79]=pnom[79];
        p[80]=pnom[80];
        p[81]=pnom[81];
        p[82]=pnom[82];
        p[83]=pnom[83];
        p[84]=pnom[84];
        p[85]=pnom[85];
        p[86]=pnom[86];
        p[87]=pnom[87];
        p[88]=pnom[88];
        p[89]=pnom[89];
        p[90]=pnom[90];
        p[91]=pnom[91];
        p[92]=pnom[92];
        p[93]=pnom[93];
        p[95]=pnom[95];
        p[96]=pnom[96];
        p[97]=pnom[97];
        p[100]=pnom[100];
        p[101]=pnom[101];
        p[102]=pnom[102];
        p[103]=pnom[103];
        p[105]=pnom[105];
        p[106]=pnom[106];
        p[107]=pnom[107];

/*      Temperature mappings for model parameters */

        Tini=2.731500e+02+pnom[0];
        Tdev=(2.731500e+02+(*TAMB))+pnom[105];
        Vtv=1.380662e-23*Tdev/1.602189e-19;
        rT=Tdev/Tini;
        dT=Tdev-Tini;
        xvar1=pow(rT,pnom[90]);
        p[53]=pnom[53]*xvar1;
        xvar1=pow(rT,pnom[91]);
        p[1]=pnom[1]*xvar1;
        xvar1=pow(rT,pnom[68]);
        p[2]=pnom[2]*xvar1;
        xvar1=pow(rT,pnom[92]);
        p[6]=pnom[6]*xvar1;
        xvar1=pow(rT,pnom[67]);
        p[7]=pnom[7]*xvar1;
        xvar1=pow(rT,pnom[66]);
        p[8]=pnom[8]*xvar1;
        xvar1=pow(rT,pnom[69]);
        p[9]=pnom[9]*xvar1;
        xvar1=pow(rT,pnom[93]);
        p[10]=pnom[10]*xvar1;
        xvar2=pow(rT,pnom[78]);
        xvar3=-pnom[71]*(1.0-rT)/Vtv;
        xvar4=exp(xvar3);
        xvar1=(xvar2*xvar4);
        xvar5=(1.0/pnom[12]);
        xvar6=pow(xvar1,xvar5);
        p[11]=pnom[11]*xvar6;
        xvar2=pow(rT,pnom[95]);
        xvar3=-pnom[96]*(1.0-rT)/Vtv;
        xvar4=exp(xvar3);
        xvar1=(xvar2*xvar4);
        xvar5=(1.0/pnom[13]);
        xvar6=pow(xvar1,xvar5);
        p[94]=pnom[94]*xvar6;
        xvar2=pow(rT,pnom[78]);
        xvar3=-pnom[97]*(1.0-rT)/Vtv;
        xvar4=exp(xvar3);
        xvar1=(xvar2*xvar4);
        xvar5=(1.0/pnom[44]);
        xvar6=pow(xvar1,xvar5);
        p[42]=pnom[42]*xvar6;
        xvar2=pow(rT,pnom[79]);
        xvar3=-pnom[72]*(1.0-rT)/Vtv;
        xvar4=exp(xvar3);
        xvar1=(xvar2*xvar4);
        xvar5=(1.0/pnom[33]);
        xvar6=pow(xvar1,xvar5);
        p[31]=pnom[31]*xvar6;
        xvar2=pow(rT,pnom[80]);
        xvar3=-pnom[75]*(1.0-rT)/Vtv;
        xvar4=exp(xvar3);
        xvar1=(xvar2*xvar4);
        xvar5=(1.0/pnom[35]);
        xvar6=pow(xvar1,xvar5);
        p[34]=pnom[34]*xvar6;
        xvar2=pow(rT,pnom[79]);
        xvar3=-pnom[73]*(1.0-rT)/Vtv;
        xvar4=exp(xvar3);
        xvar1=(xvar2*xvar4);
        xvar5=(1.0/pnom[37]);
        xvar6=pow(xvar1,xvar5);
        p[36]=pnom[36]*xvar6;
        xvar2=pow(rT,pnom[80]);
        xvar3=-pnom[76]*(1.0-rT)/Vtv;
        xvar4=exp(xvar3);
        xvar1=(xvar2*xvar4);
        xvar5=(1.0/pnom[39]);
        xvar6=pow(xvar1,xvar5);
        p[38]=pnom[38]*xvar6;
        xvar2=pow(rT,pnom[79]);
        xvar3=-pnom[73]*(1.0-rT)/Vtv;
        xvar4=exp(xvar3);
        xvar1=(xvar2*xvar4);
        xvar5=(1.0/pnom[37]);
        xvar6=pow(xvar1,xvar5);
        p[45]=pnom[45]*xvar6;
        xvar2=pow(rT,pnom[80]);
        xvar3=-pnom[76]*(1.0-rT)/Vtv;
        xvar4=exp(xvar3);
        xvar1=(xvar2*xvar4);
        xvar5=(1.0/pnom[39]);
        xvar6=pow(xvar1,xvar5);
        p[46]=pnom[46]*xvar6;
        xvar2=pow(rT,pnom[79]);
        xvar3=-pnom[74]*(1.0-rT)/Vtv;
        xvar4=exp(xvar3);
        xvar1=(xvar2*xvar4);
        xvar5=(1.0/pnom[48]);
        xvar6=pow(xvar1,xvar5);
        p[47]=pnom[47]*xvar6;
        xvar2=pow(rT,pnom[80]);
        xvar3=-pnom[77]*(1.0-rT)/Vtv;
        xvar4=exp(xvar3);
        xvar1=(xvar2*xvar4);
        xvar5=(1.0/pnom[50]);
        xvar6=pow(xvar1,xvar5);
        p[49]=pnom[49]*xvar6;
        p[12]=pnom[12]*(1.0+dT*pnom[81]);
        p[13]=pnom[13]*(1.0+dT*pnom[81]);
        p[41]=pnom[41]*(1.0+dT*pnom[82]);
        p[98]=pnom[98]*(1.0+dT*(pnom[101]+dT*pnom[102]));
        p[99]=pnom[99]*(1.0+dT*pnom[103]);
        xvar2=0.5*pnom[17]*rT/Vtv;
        xvar3=exp(xvar2);
        xvar4=-0.5*pnom[17]*rT/Vtv;
        xvar5=exp(xvar4);
        xvar1=xvar3-xvar5;
        xvar6=log(xvar1);
        psiio=2.0*(Vtv/rT)*xvar6;
        xvar1=log(rT);
        psiin=psiio*rT-3.0*Vtv*xvar1-pnom[72]*(rT-1.0);
        xvar2=-psiin/Vtv;
        xvar3=exp(xvar2);
        xvar1=0.5*(1.0+sqrt(1.0+4.0*xvar3));
        xvar4=log(xvar1);
        p[17]=psiin+2.0*Vtv*xvar4;
        xvar2=0.5*pnom[24]*rT/Vtv;
        xvar3=exp(xvar2);
        xvar4=-0.5*pnom[24]*rT/Vtv;
        xvar5=exp(xvar4);
        xvar1=xvar3-xvar5;
        xvar6=log(xvar1);
        psiio=2.0*(Vtv/rT)*xvar6;
        xvar1=log(rT);
        psiin=psiio*rT-3.0*Vtv*xvar1-pnom[73]*(rT-1.0);
        xvar2=-psiin/Vtv;
        xvar3=exp(xvar2);
        xvar1=0.5*(1.0+sqrt(1.0+4.0*xvar3));
        xvar4=log(xvar1);
        p[24]=psiin+2.0*Vtv*xvar4;
        xvar2=0.5*pnom[28]*rT/Vtv;
        xvar3=exp(xvar2);
        xvar4=-0.5*pnom[28]*rT/Vtv;
        xvar5=exp(xvar4);
        xvar1=xvar3-xvar5;
        xvar6=log(xvar1);
        psiio=2.0*(Vtv/rT)*xvar6;
        xvar1=log(rT);
        psiin=psiio*rT-3.0*Vtv*xvar1-pnom[74]*(rT-1.0);
        xvar2=-psiin/Vtv;
        xvar3=exp(xvar2);
        xvar1=0.5*(1.0+sqrt(1.0+4.0*xvar3));
        xvar4=log(xvar1);
        p[28]=psiin+2.0*Vtv*xvar4;
        xvar1=pnom[17]/p[17];
        xvar2=pow(xvar1,pnom[18]);
        p[16]=pnom[16]*xvar2;
        xvar1=pnom[24]/p[24];
        xvar2=pow(xvar1,pnom[25]);
        p[21]=pnom[21]*xvar2;
        xvar1=pnom[24]/p[24];
        xvar2=pow(xvar1,pnom[25]);
        p[23]=pnom[23]*xvar2;
        xvar1=pnom[28]/p[28];
        xvar2=pow(xvar1,pnom[29]);
        p[27]=pnom[27]*xvar2;
        xvar1=pow(rT,pnom[78]);
        xvar2=-pnom[71]*(1.0-rT)/Vtv;
        xvar3=exp(xvar2);
        p[4]=pnom[4]*xvar1*xvar3;
        xvar1=pow(rT,pnom[70]);
        p[3]=pnom[3]*xvar1;
        xvar1=-p[98]/(p[99]*Vtv);
        p[104]=exp(xvar1);
        p[0]=(*TAMB)+p[105];
        return(0);
}
