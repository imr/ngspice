/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soiset.c          98/5/01
Modified by Pin Su and Jan Feng	99/2/15
Modified by Pin Su 99/4/30
Modified by Pin Su, Wei Jin 99/9/27
Modified by Pin Su 00/3/1
Modified by Pin Su 01/2/15
Modified by Pin Su 02/3/5
Modified by Pin Su 02/5/20 
Modified by Paolo Nenzi 2002
**********/


#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "b3soidef.h"
#include "const.h"
#include "sperror.h"
#include "suffix.h"

#define SMOOTHFACTOR 0.1
#define EPSOX 3.453133e-11
#define EPSSI 1.03594e-10
#define PI 3.141592654
#define Charge_q 1.60219e-19
#define Meter2Micron 1.0e6

int
B3SOIsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
{
B3SOImodel *model = (B3SOImodel*)inModel;
B3SOIinstance *here;
int error;
CKTnode *tmp;

double Cboxt;


/* Alan's Nodeset Fix */
CKTnode *tmpNode;
IFuid tmpName;


    /*  loop through all the B3SOI device models */
    for( ; model != NULL; model = model->B3SOInextModel )
    {
/* Default value Processing for B3SOI MOSFET Models */

        if (!model->B3SOItypeGiven)
            model->B3SOItype = NMOS;     
        if (!model->B3SOImobModGiven) 
            model->B3SOImobMod = 1;
        if (!model->B3SOIbinUnitGiven) 
            model->B3SOIbinUnit = 1;
        if (!model->B3SOIparamChkGiven) 
            model->B3SOIparamChk = 0;
        if (!model->B3SOIcapModGiven) 
            model->B3SOIcapMod = 2;
        if (!model->B3SOInoiModGiven) 
            model->B3SOInoiMod = 1;
        if (!model->B3SOIshModGiven) 
            model->B3SOIshMod = 0;
        if (!model->B3SOIversionGiven) 
            model->B3SOIversion = 2.0;
        if (!model->B3SOItoxGiven)
            model->B3SOItox = 100.0e-10;
        model->B3SOIcox = 3.453133e-11 / model->B3SOItox;

/* v2.2.3 */
        if (!model->B3SOIdtoxcvGiven)
            model->B3SOIdtoxcv = 0.0;

        if (!model->B3SOIcdscGiven)
	    model->B3SOIcdsc = 2.4e-4;   /* unit Q/V/m^2  */
        if (!model->B3SOIcdscbGiven)
	    model->B3SOIcdscb = 0.0;   /* unit Q/V/m^2  */    
	    if (!model->B3SOIcdscdGiven)
	    model->B3SOIcdscd = 0.0;   /* unit Q/V/m^2  */
        if (!model->B3SOIcitGiven)
	    model->B3SOIcit = 0.0;   /* unit Q/V/m^2  */
        if (!model->B3SOInfactorGiven)
	    model->B3SOInfactor = 1;
        if (!model->B3SOIvsatGiven)
            model->B3SOIvsat = 8.0e4;    /* unit m/s */ 
        if (!model->B3SOIatGiven)
            model->B3SOIat = 3.3e4;    /* unit m/s */ 
        if (!model->B3SOIa0Given)
            model->B3SOIa0 = 1.0;  
        if (!model->B3SOIagsGiven)
            model->B3SOIags = 0.0;
        if (!model->B3SOIa1Given)
            model->B3SOIa1 = 0.0;
        if (!model->B3SOIa2Given)
            model->B3SOIa2 = 1.0;
        if (!model->B3SOIketaGiven)
            model->B3SOIketa = -0.6;    /* unit  / V */
        if (!model->B3SOInsubGiven)
            model->B3SOInsub = 6.0e16;   /* unit 1/cm3 */
        if (!model->B3SOInpeakGiven)
            model->B3SOInpeak = 1.7e17;   /* unit 1/cm3 */
        if (!model->B3SOIngateGiven)
            model->B3SOIngate = 0;   /* unit 1/cm3 */
        if (!model->B3SOIvbmGiven)
	    model->B3SOIvbm = -3.0;
        if (!model->B3SOIxtGiven)
	    model->B3SOIxt = 1.55e-7;
        if (!model->B3SOIkt1Given)
            model->B3SOIkt1 = -0.11;      /* unit V */
        if (!model->B3SOIkt1lGiven)
            model->B3SOIkt1l = 0.0;      /* unit V*m */
        if (!model->B3SOIkt2Given)
            model->B3SOIkt2 = 0.022;      /* No unit */
        if (!model->B3SOIk3Given)
            model->B3SOIk3 = 0.0;      
        if (!model->B3SOIk3bGiven)
            model->B3SOIk3b = 0.0;      
        if (!model->B3SOIw0Given)
            model->B3SOIw0 = 2.5e-6;    
        if (!model->B3SOInlxGiven)
            model->B3SOInlx = 1.74e-7;     
        if (!model->B3SOIdvt0Given)
            model->B3SOIdvt0 = 2.2;    
        if (!model->B3SOIdvt1Given)
            model->B3SOIdvt1 = 0.53;      
        if (!model->B3SOIdvt2Given)
            model->B3SOIdvt2 = -0.032;   /* unit 1 / V */     

        if (!model->B3SOIdvt0wGiven)
            model->B3SOIdvt0w = 0.0;    
        if (!model->B3SOIdvt1wGiven)
            model->B3SOIdvt1w = 5.3e6;    
        if (!model->B3SOIdvt2wGiven)
            model->B3SOIdvt2w = -0.032;   

        if (!model->B3SOIdroutGiven)
            model->B3SOIdrout = 0.56;     
        if (!model->B3SOIdsubGiven)
            model->B3SOIdsub = model->B3SOIdrout;     
        if (!model->B3SOIvth0Given)
            model->B3SOIvth0 = (model->B3SOItype == NMOS) ? 0.7 : -0.7;
        if (!model->B3SOIuaGiven)
            model->B3SOIua = 2.25e-9;      /* unit m/V */
        if (!model->B3SOIua1Given)
            model->B3SOIua1 = 4.31e-9;      /* unit m/V */
        if (!model->B3SOIubGiven)
            model->B3SOIub = 5.87e-19;     /* unit (m/V)**2 */
        if (!model->B3SOIub1Given)
            model->B3SOIub1 = -7.61e-18;     /* unit (m/V)**2 */
        if (!model->B3SOIucGiven)
            model->B3SOIuc = (model->B3SOImobMod == 3) ? -0.0465 : -0.0465e-9;   
        if (!model->B3SOIuc1Given)
            model->B3SOIuc1 = (model->B3SOImobMod == 3) ? -0.056 : -0.056e-9;   
        if (!model->B3SOIu0Given)
            model->B3SOIu0 = (model->B3SOItype == NMOS) ? 0.067 : 0.025;
        if (!model->B3SOIuteGiven)
	    model->B3SOIute = -1.5;    
        if (!model->B3SOIvoffGiven)
	    model->B3SOIvoff = -0.08;
        if (!model->B3SOIdeltaGiven)  
           model->B3SOIdelta = 0.01;
        if (!model->B3SOIrdswGiven)
            model->B3SOIrdsw = 100;
        if (!model->B3SOIprwgGiven)
            model->B3SOIprwg = 0.0;      /* unit 1/V */
        if (!model->B3SOIprwbGiven)
            model->B3SOIprwb = 0.0;      
        if (!model->B3SOIprtGiven)
        if (!model->B3SOIprtGiven)
            model->B3SOIprt = 0.0;      
        if (!model->B3SOIeta0Given)
            model->B3SOIeta0 = 0.08;      /* no unit  */ 
        if (!model->B3SOIetabGiven)
            model->B3SOIetab = -0.07;      /* unit  1/V */ 
        if (!model->B3SOIpclmGiven)
            model->B3SOIpclm = 1.3;      /* no unit  */ 
        if (!model->B3SOIpdibl1Given)
            model->B3SOIpdibl1 = .39;    /* no unit  */
        if (!model->B3SOIpdibl2Given)
            model->B3SOIpdibl2 = 0.0086;    /* no unit  */ 
        if (!model->B3SOIpdiblbGiven)
            model->B3SOIpdiblb = 0.0;    /* 1/V  */ 
        if (!model->B3SOIpvagGiven)
            model->B3SOIpvag = 0.0;     
        if (!model->B3SOIwrGiven)  
            model->B3SOIwr = 1.0;
        if (!model->B3SOIdwgGiven)  
            model->B3SOIdwg = 0.0;
        if (!model->B3SOIdwbGiven)  
            model->B3SOIdwb = 0.0;
        if (!model->B3SOIb0Given)
            model->B3SOIb0 = 0.0;
        if (!model->B3SOIb1Given)  
            model->B3SOIb1 = 0.0;
        if (!model->B3SOIalpha0Given)  
            model->B3SOIalpha0 = 0.0;

        if (!model->B3SOIcgslGiven)  
            model->B3SOIcgsl = 0.0;
        if (!model->B3SOIcgdlGiven)  
            model->B3SOIcgdl = 0.0;
        if (!model->B3SOIckappaGiven)  
            model->B3SOIckappa = 0.6;
        if (!model->B3SOIclcGiven)  
            model->B3SOIclc = 0.1e-7;
        if (!model->B3SOIcleGiven)  
            model->B3SOIcle = 0.0;
        if (!model->B3SOItboxGiven)  
            model->B3SOItbox = 3e-7;
        if (!model->B3SOItsiGiven)  
            model->B3SOItsi = 1e-7;
        if (!model->B3SOIxjGiven)  
            model->B3SOIxj = model->B3SOItsi;
        if (!model->B3SOIrbodyGiven)  
            model->B3SOIrbody = 0.0;
        if (!model->B3SOIrbshGiven)  
            model->B3SOIrbsh = 0.0;
        if (!model->B3SOIrth0Given)  
            model->B3SOIrth0 = 0;

/* v3.0 bug fix */
        if (!model->B3SOIcth0Given)
            model->B3SOIcth0 = 1e-5;

        if (!model->B3SOIagidlGiven)  
            model->B3SOIagidl = 0.0;
        if (!model->B3SOIbgidlGiven)  
            model->B3SOIbgidl = 0.0;
        if (!model->B3SOIngidlGiven)  
            model->B3SOIngidl = 1.2;
        if (!model->B3SOIndiodeGiven)  
            model->B3SOIndiode = 1.0;
        if (!model->B3SOIntunGiven)  
            model->B3SOIntun = 10.0;

        if (!model->B3SOInrecf0Given)
            model->B3SOInrecf0 = 2.0;
        if (!model->B3SOInrecr0Given)
            model->B3SOInrecr0 = 10.0;
 
        if (!model->B3SOIisbjtGiven)  
            model->B3SOIisbjt = 1e-6;
        if (!model->B3SOIisdifGiven)  
            model->B3SOIisdif = 0.0;
        if (!model->B3SOIisrecGiven)  
            model->B3SOIisrec = 1e-5;
        if (!model->B3SOIistunGiven)  
            model->B3SOIistun = 0.0;
        if (!model->B3SOIxbjtGiven)  
            model->B3SOIxbjt = 1;
/*
        if (!model->B3SOIxdifGiven)  
            model->B3SOIxdif = 1;
*/
        if (!model->B3SOIxdifGiven)
            model->B3SOIxdif = model->B3SOIxbjt;

        if (!model->B3SOIxrecGiven)  
            model->B3SOIxrec = 1;
        if (!model->B3SOIxtunGiven)  
            model->B3SOIxtun = 0;
        if (!model->B3SOIttGiven)  
            model->B3SOItt = 1e-12;
        if (!model->B3SOIasdGiven)  
            model->B3SOIasd = 0.3;

        /* unit degree celcius */
        if (!model->B3SOItnomGiven)  
	    model->B3SOItnom = ckt->CKTnomTemp; 
        if (!model->B3SOILintGiven)  
           model->B3SOILint = 0.0;
        if (!model->B3SOILlGiven)  
           model->B3SOILl = 0.0;
        if (!model->B3SOILlcGiven) 
           model->B3SOILlc = 0.0; /* v2.2.3 */
        if (!model->B3SOILlnGiven)  
           model->B3SOILln = 1.0;
        if (!model->B3SOILwGiven)  
           model->B3SOILw = 0.0;
        if (!model->B3SOILwcGiven) 
           model->B3SOILwc = 0.0; /* v2.2.3 */
        if (!model->B3SOILwnGiven)  
           model->B3SOILwn = 1.0;
        if (!model->B3SOILwlGiven)  
           model->B3SOILwl = 0.0;
        if (!model->B3SOILwlcGiven) 
           model->B3SOILwlc = 0.0; /* v2.2.3 */
        if (!model->B3SOILminGiven)  
           model->B3SOILmin = 0.0;
        if (!model->B3SOILmaxGiven)  
           model->B3SOILmax = 1.0;
        if (!model->B3SOIWintGiven)  
           model->B3SOIWint = 0.0;
        if (!model->B3SOIWlGiven)  
           model->B3SOIWl = 0.0;
        if (!model->B3SOIWlcGiven) 
           model->B3SOIWlc = 0.0; /* v2.2.3 */
        if (!model->B3SOIWlnGiven)  
           model->B3SOIWln = 1.0;
        if (!model->B3SOIWwGiven)  
           model->B3SOIWw = 0.0;
        if (!model->B3SOIWwcGiven) 
           model->B3SOIWwc = 0.0; /* v2.2.3 */
        if (!model->B3SOIWwnGiven)  
           model->B3SOIWwn = 1.0;
        if (!model->B3SOIWwlGiven)  
           model->B3SOIWwl = 0.0;
        if (!model->B3SOIWwlcGiven)
           model->B3SOIWwlc = 0.0; /* v2.2.3 */
        if (!model->B3SOIWminGiven)  
           model->B3SOIWmin = 0.0;
        if (!model->B3SOIWmaxGiven)  
           model->B3SOIWmax = 1.0;
        if (!model->B3SOIdwcGiven)  
           model->B3SOIdwc = model->B3SOIWint;
        if (!model->B3SOIdlcGiven)  
           model->B3SOIdlc = model->B3SOILint;
        if (!model->B3SOIdlcigGiven)
           model->B3SOIdlcig = model->B3SOILint; /* v3.0 */

/* v3.0 */
        if (!model->B3SOIsoimodGiven)
            model->B3SOIsoiMod = 0;
        if (!model->B3SOIvbsaGiven)
            model->B3SOIvbsa = 0.0;
        if (!model->B3SOInofffdGiven)
            model->B3SOInofffd = 1.0;
        if (!model->B3SOIvofffdGiven)
            model->B3SOIvofffd = 0.0;
        if (!model->B3SOIk1bGiven)
            model->B3SOIk1b = 1.0;
        if (!model->B3SOIk2bGiven)
            model->B3SOIk2b = 0.0;
        if (!model->B3SOIdk2bGiven)
            model->B3SOIdk2b = 0.0;
        if (!model->B3SOIdvbd0Given)
            model->B3SOIdvbd0 = 0.0;
        if (!model->B3SOIdvbd1Given)
            model->B3SOIdvbd1 = 0.0;
        if (!model->B3SOImoinFDGiven)
            model->B3SOImoinFD = 1e3;


/* v2.2 release */
        if (!model->B3SOIwth0Given)
           model->B3SOIwth0 = 0.0;
        if (!model->B3SOIrhaloGiven)
           model->B3SOIrhalo = 1e15;
        if (!model->B3SOIntoxGiven)
           model->B3SOIntox = 1;
        if (!model->B3SOItoxrefGiven)
           model->B3SOItoxref = 2.5e-9;
        if (!model->B3SOIebgGiven)
           model->B3SOIebg = 1.2;
        if (!model->B3SOIvevbGiven)
           model->B3SOIvevb = 0.075;
        if (!model->B3SOIalphaGB1Given)
           model->B3SOIalphaGB1 = 0.35;
        if (!model->B3SOIbetaGB1Given)
           model->B3SOIbetaGB1 = 0.03;
        if (!model->B3SOIvgb1Given)
           model->B3SOIvgb1 = 300;
        if (!model->B3SOIalphaGB2Given)
           model->B3SOIalphaGB2 = 0.43;
        if (!model->B3SOIbetaGB2Given)
           model->B3SOIbetaGB2 = 0.05;
        if (!model->B3SOIvecbGiven)
           model->B3SOIvecb = 0.026;
        if (!model->B3SOIvgb2Given)
           model->B3SOIvgb2 = 17;
        if (!model->B3SOItoxqmGiven)
           model->B3SOItoxqm = model->B3SOItox;
        if (!model->B3SOIvoxhGiven)
           model->B3SOIvoxh = 5.0;
        if (!model->B3SOIdeltavoxGiven)
           model->B3SOIdeltavox = 0.005;

/* v3.0 */
        if (!model->B3SOIigbModGiven)
           model->B3SOIigbMod = 0;
        if (!model->B3SOIigcModGiven)
           model->B3SOIigcMod = 0;
        if (!model->B3SOInigcGiven)
            model->B3SOInigc = 1.0;
        if (!model->B3SOIaigcGiven)
            model->B3SOIaigc = (model->B3SOItype == NMOS) ? 0.43 : 0.31;
        if (!model->B3SOIbigcGiven)
            model->B3SOIbigc = (model->B3SOItype == NMOS) ? 0.054 : 0.024;
        if (!model->B3SOIcigcGiven)
            model->B3SOIcigc = (model->B3SOItype == NMOS) ? 0.075 : 0.03;
        if (!model->B3SOIaigsdGiven)
            model->B3SOIaigsd = (model->B3SOItype == NMOS) ? 0.43 : 0.31;
        if (!model->B3SOIbigsdGiven)
            model->B3SOIbigsd = (model->B3SOItype == NMOS) ? 0.054 : 0.024;
        if (!model->B3SOIcigsdGiven)
            model->B3SOIcigsd = (model->B3SOItype == NMOS) ? 0.075 : 0.03;
        if (!model->B3SOIpigcdGiven)
            model->B3SOIpigcd = 1.0;
        if (!model->B3SOIpoxedgeGiven)
            model->B3SOIpoxedge = 1.0;



/* v2.0 release */
        if (!model->B3SOIk1w1Given) 
           model->B3SOIk1w1 = 0.0;
        if (!model->B3SOIk1w2Given)
           model->B3SOIk1w2 = 0.0;
        if (!model->B3SOIketasGiven)
           model->B3SOIketas = 0.0;
        if (!model->B3SOIdwbcGiven)
           model->B3SOIdwbc = 0.0;
        if (!model->B3SOIbeta0Given)
           model->B3SOIbeta0 = 0.0;
        if (!model->B3SOIbeta1Given)
           model->B3SOIbeta1 = 0.0;
        if (!model->B3SOIbeta2Given)
           model->B3SOIbeta2 = 0.1;
        if (!model->B3SOIvdsatii0Given)
           model->B3SOIvdsatii0 = 0.9;
        if (!model->B3SOItiiGiven)
           model->B3SOItii = 0.0;
        if (!model->B3SOIliiGiven)
           model->B3SOIlii = 0.0;
        if (!model->B3SOIsii0Given)
           model->B3SOIsii0 = 0.5;
        if (!model->B3SOIsii1Given)
           model->B3SOIsii1 = 0.1;
        if (!model->B3SOIsii2Given)
           model->B3SOIsii2 = 0.0;
        if (!model->B3SOIsiidGiven)
           model->B3SOIsiid = 0.0;
        if (!model->B3SOIfbjtiiGiven)
           model->B3SOIfbjtii = 0.0;
        if (!model->B3SOIesatiiGiven)
            model->B3SOIesatii = 1e7;
        if (!model->B3SOIlnGiven)
           model->B3SOIln = 2e-6;
        if (!model->B3SOIvrec0Given)
           model->B3SOIvrec0 = 0;
        if (!model->B3SOIvtun0Given)
           model->B3SOIvtun0 = 0;
        if (!model->B3SOInbjtGiven)
           model->B3SOInbjt = 1.0;
        if (!model->B3SOIlbjt0Given)
           model->B3SOIlbjt0 = 0.20e-6;
        if (!model->B3SOIldif0Given)
           model->B3SOIldif0 = 1.0;
        if (!model->B3SOIvabjtGiven)
           model->B3SOIvabjt = 10.0;
        if (!model->B3SOIaelyGiven)
           model->B3SOIaely = 0;
        if (!model->B3SOIahliGiven)
           model->B3SOIahli = 0;
        if (!model->B3SOIrbodyGiven)
           model->B3SOIrbody = 0.0;
        if (!model->B3SOIrbshGiven)
           model->B3SOIrbsh = 0.0;
        if (!model->B3SOIntrecfGiven)
           model->B3SOIntrecf = 0.0;
        if (!model->B3SOIntrecrGiven)
           model->B3SOIntrecr = 0.0;
        if (!model->B3SOIndifGiven)
           model->B3SOIndif = -1.0;
        if (!model->B3SOIdlcbGiven)
           model->B3SOIdlcb = 0.0;
        if (!model->B3SOIfbodyGiven)
           model->B3SOIfbody = 1.0;
        if (!model->B3SOItcjswgGiven)
           model->B3SOItcjswg = 0.0;
        if (!model->B3SOItpbswgGiven)
           model->B3SOItpbswg = 0.0;
        if (!model->B3SOIacdeGiven)
           model->B3SOIacde = 1.0;
        if (!model->B3SOImoinGiven)
           model->B3SOImoin = 15.0;
        if (!model->B3SOIdelvtGiven)
           model->B3SOIdelvt = 0.0;
        if (!model->B3SOIkb1Given)
           model->B3SOIkb1 = 1.0;
        if (!model->B3SOIdlbgGiven)
           model->B3SOIdlbg = 0.0;

/* Added for binning - START */
        /* Length dependence */
/* v3.0 */
        if (!model->B3SOIlnigcGiven)
            model->B3SOIlnigc = 0.0;
        if (!model->B3SOIlpigcdGiven)
            model->B3SOIlpigcd = 0.0;
        if (!model->B3SOIlpoxedgeGiven)
            model->B3SOIlpoxedge = 0.0;
        if (!model->B3SOIlaigcGiven)
            model->B3SOIlaigc = 0.0;
        if (!model->B3SOIlbigcGiven)
            model->B3SOIlbigc = 0.0;
        if (!model->B3SOIlcigcGiven)
            model->B3SOIlcigc = 0.0;
        if (!model->B3SOIlaigsdGiven)
            model->B3SOIlaigsd = 0.0;
        if (!model->B3SOIlbigsdGiven)
            model->B3SOIlbigsd = 0.0;
        if (!model->B3SOIlcigsdGiven)
            model->B3SOIlcigsd = 0.0;

        if (!model->B3SOIlnpeakGiven)
            model->B3SOIlnpeak = 0.0;
        if (!model->B3SOIlnsubGiven)
            model->B3SOIlnsub = 0.0;
        if (!model->B3SOIlngateGiven)
            model->B3SOIlngate = 0.0;
        if (!model->B3SOIlvth0Given)
           model->B3SOIlvth0 = 0.0;
        if (!model->B3SOIlk1Given)
            model->B3SOIlk1 = 0.0;
        if (!model->B3SOIlk1w1Given)
            model->B3SOIlk1w1 = 0.0;
        if (!model->B3SOIlk1w2Given)
            model->B3SOIlk1w2 = 0.0;
        if (!model->B3SOIlk2Given)
            model->B3SOIlk2 = 0.0;
        if (!model->B3SOIlk3Given)
            model->B3SOIlk3 = 0.0;
        if (!model->B3SOIlk3bGiven)
            model->B3SOIlk3b = 0.0;
        if (!model->B3SOIlkb1Given)
           model->B3SOIlkb1 = 0.0;
        if (!model->B3SOIlw0Given)
            model->B3SOIlw0 = 0.0;
        if (!model->B3SOIlnlxGiven)
            model->B3SOIlnlx = 0.0;
        if (!model->B3SOIldvt0Given)
            model->B3SOIldvt0 = 0.0;
        if (!model->B3SOIldvt1Given)
            model->B3SOIldvt1 = 0.0;
        if (!model->B3SOIldvt2Given)
            model->B3SOIldvt2 = 0.0;
        if (!model->B3SOIldvt0wGiven)
            model->B3SOIldvt0w = 0.0;
        if (!model->B3SOIldvt1wGiven)
            model->B3SOIldvt1w = 0.0;
        if (!model->B3SOIldvt2wGiven)
            model->B3SOIldvt2w = 0.0;
        if (!model->B3SOIlu0Given)
            model->B3SOIlu0 = 0.0;
        if (!model->B3SOIluaGiven)
            model->B3SOIlua = 0.0;
        if (!model->B3SOIlubGiven)
            model->B3SOIlub = 0.0;
        if (!model->B3SOIlucGiven)
            model->B3SOIluc = 0.0;
        if (!model->B3SOIlvsatGiven)
            model->B3SOIlvsat = 0.0;
        if (!model->B3SOIla0Given)
            model->B3SOIla0 = 0.0;
        if (!model->B3SOIlagsGiven)
            model->B3SOIlags = 0.0;
        if (!model->B3SOIlb0Given)
            model->B3SOIlb0 = 0.0;
        if (!model->B3SOIlb1Given)
            model->B3SOIlb1 = 0.0;
        if (!model->B3SOIlketaGiven)
            model->B3SOIlketa = 0.0;
        if (!model->B3SOIlketasGiven)
            model->B3SOIlketas = 0.0;
        if (!model->B3SOIla1Given)
            model->B3SOIla1 = 0.0;
        if (!model->B3SOIla2Given)
            model->B3SOIla2 = 0.0;
        if (!model->B3SOIlrdswGiven)
            model->B3SOIlrdsw = 0.0;
        if (!model->B3SOIlprwbGiven)
            model->B3SOIlprwb = 0.0;
        if (!model->B3SOIlprwgGiven)
            model->B3SOIlprwg = 0.0;
        if (!model->B3SOIlwrGiven)
            model->B3SOIlwr = 0.0;
        if (!model->B3SOIlnfactorGiven)
            model->B3SOIlnfactor = 0.0;
        if (!model->B3SOIldwgGiven)
            model->B3SOIldwg = 0.0;
        if (!model->B3SOIldwbGiven)
            model->B3SOIldwb = 0.0;
        if (!model->B3SOIlvoffGiven)
            model->B3SOIlvoff = 0.0;
        if (!model->B3SOIleta0Given)
            model->B3SOIleta0 = 0.0;
        if (!model->B3SOIletabGiven)
            model->B3SOIletab = 0.0;
        if (!model->B3SOIldsubGiven)
            model->B3SOIldsub = 0.0;
        if (!model->B3SOIlcitGiven)
            model->B3SOIlcit = 0.0;
        if (!model->B3SOIlcdscGiven)
            model->B3SOIlcdsc = 0.0;
        if (!model->B3SOIlcdscbGiven)
            model->B3SOIlcdscb = 0.0;
        if (!model->B3SOIlcdscdGiven)
            model->B3SOIlcdscd = 0.0;
        if (!model->B3SOIlpclmGiven)
            model->B3SOIlpclm = 0.0;
        if (!model->B3SOIlpdibl1Given)
            model->B3SOIlpdibl1 = 0.0;
        if (!model->B3SOIlpdibl2Given)
            model->B3SOIlpdibl2 = 0.0;
        if (!model->B3SOIlpdiblbGiven)
            model->B3SOIlpdiblb = 0.0;
        if (!model->B3SOIldroutGiven)
            model->B3SOIldrout = 0.0;
        if (!model->B3SOIlpvagGiven)
            model->B3SOIlpvag = 0.0;
        if (!model->B3SOIldeltaGiven)
            model->B3SOIldelta = 0.0;
        if (!model->B3SOIlalpha0Given)
            model->B3SOIlalpha0 = 0.0;
        if (!model->B3SOIlfbjtiiGiven)
            model->B3SOIlfbjtii = 0.0;
        if (!model->B3SOIlbeta0Given)
            model->B3SOIlbeta0 = 0.0;
        if (!model->B3SOIlbeta1Given)
            model->B3SOIlbeta1 = 0.0;
        if (!model->B3SOIlbeta2Given)
            model->B3SOIlbeta2 = 0.0;
        if (!model->B3SOIlvdsatii0Given)
            model->B3SOIlvdsatii0 = 0.0;
        if (!model->B3SOIlliiGiven)
            model->B3SOIllii = 0.0;
        if (!model->B3SOIlesatiiGiven)
            model->B3SOIlesatii = 0.0;
        if (!model->B3SOIlsii0Given)
            model->B3SOIlsii0 = 0.0;
        if (!model->B3SOIlsii1Given)
            model->B3SOIlsii1 = 0.0;
        if (!model->B3SOIlsii2Given)
            model->B3SOIlsii2 = 0.0;
        if (!model->B3SOIlsiidGiven)
            model->B3SOIlsiid = 0.0;
        if (!model->B3SOIlagidlGiven)
            model->B3SOIlagidl = 0.0;
        if (!model->B3SOIlbgidlGiven)
            model->B3SOIlbgidl = 0.0;
        if (!model->B3SOIlngidlGiven)
            model->B3SOIlngidl = 0.0;
        if (!model->B3SOIlntunGiven)
            model->B3SOIlntun = 0.0;
        if (!model->B3SOIlndiodeGiven)
            model->B3SOIlndiode = 0.0;
        if (!model->B3SOIlnrecf0Given)
            model->B3SOIlnrecf0 = 0.0;
        if (!model->B3SOIlnrecr0Given)
            model->B3SOIlnrecr0 = 0.0;
        if (!model->B3SOIlisbjtGiven)
            model->B3SOIlisbjt = 0.0;
        if (!model->B3SOIlisdifGiven)
            model->B3SOIlisdif = 0.0;
        if (!model->B3SOIlisrecGiven)
            model->B3SOIlisrec = 0.0;
        if (!model->B3SOIlistunGiven)
            model->B3SOIlistun = 0.0;
        if (!model->B3SOIlvrec0Given)
            model->B3SOIlvrec0 = 0.0;
        if (!model->B3SOIlvtun0Given)
            model->B3SOIlvtun0 = 0.0;
        if (!model->B3SOIlnbjtGiven)
            model->B3SOIlnbjt = 0.0;
        if (!model->B3SOIllbjt0Given)
            model->B3SOIllbjt0 = 0.0;
        if (!model->B3SOIlvabjtGiven)
            model->B3SOIlvabjt = 0.0;
        if (!model->B3SOIlaelyGiven)
            model->B3SOIlaely = 0.0;
        if (!model->B3SOIlahliGiven)
            model->B3SOIlahli = 0.0;
	/* CV Model */
        if (!model->B3SOIlvsdfbGiven)
            model->B3SOIlvsdfb = 0.0;
        if (!model->B3SOIlvsdthGiven)
            model->B3SOIlvsdth = 0.0;
        if (!model->B3SOIldelvtGiven)
            model->B3SOIldelvt = 0.0;
        if (!model->B3SOIlacdeGiven)
            model->B3SOIlacde = 0.0;
        if (!model->B3SOIlmoinGiven)
            model->B3SOIlmoin = 0.0;

        /* Width dependence */
/* v3.0 */
        if (!model->B3SOIwnigcGiven)
            model->B3SOIwnigc = 0.0;
        if (!model->B3SOIwpigcdGiven)
            model->B3SOIwpigcd = 0.0;
        if (!model->B3SOIwpoxedgeGiven)
            model->B3SOIwpoxedge = 0.0;
        if (!model->B3SOIwaigcGiven)
            model->B3SOIwaigc = 0.0;
        if (!model->B3SOIwbigcGiven)
            model->B3SOIwbigc = 0.0;
        if (!model->B3SOIwcigcGiven)
            model->B3SOIwcigc = 0.0;
        if (!model->B3SOIwaigsdGiven)
            model->B3SOIwaigsd = 0.0;
        if (!model->B3SOIwbigsdGiven)
            model->B3SOIwbigsd = 0.0;
        if (!model->B3SOIwcigsdGiven)
            model->B3SOIwcigsd = 0.0;

        if (!model->B3SOIwnpeakGiven)
            model->B3SOIwnpeak = 0.0;
        if (!model->B3SOIwnsubGiven)
            model->B3SOIwnsub = 0.0;
        if (!model->B3SOIwngateGiven)
            model->B3SOIwngate = 0.0;
        if (!model->B3SOIwvth0Given)
           model->B3SOIwvth0 = 0.0;
        if (!model->B3SOIwk1Given)
            model->B3SOIwk1 = 0.0;
        if (!model->B3SOIwk1w1Given)
            model->B3SOIwk1w1 = 0.0;
        if (!model->B3SOIwk1w2Given)
            model->B3SOIwk1w2 = 0.0;
        if (!model->B3SOIwk2Given)
            model->B3SOIwk2 = 0.0;
        if (!model->B3SOIwk3Given)
            model->B3SOIwk3 = 0.0;
        if (!model->B3SOIwk3bGiven)
            model->B3SOIwk3b = 0.0;
        if (!model->B3SOIwkb1Given)
           model->B3SOIwkb1 = 0.0;
        if (!model->B3SOIww0Given)
            model->B3SOIww0 = 0.0;
        if (!model->B3SOIwnlxGiven)
            model->B3SOIwnlx = 0.0;
        if (!model->B3SOIwdvt0Given)
            model->B3SOIwdvt0 = 0.0;
        if (!model->B3SOIwdvt1Given)
            model->B3SOIwdvt1 = 0.0;
        if (!model->B3SOIwdvt2Given)
            model->B3SOIwdvt2 = 0.0;
        if (!model->B3SOIwdvt0wGiven)
            model->B3SOIwdvt0w = 0.0;
        if (!model->B3SOIwdvt1wGiven)
            model->B3SOIwdvt1w = 0.0;
        if (!model->B3SOIwdvt2wGiven)
            model->B3SOIwdvt2w = 0.0;
        if (!model->B3SOIwu0Given)
            model->B3SOIwu0 = 0.0;
        if (!model->B3SOIwuaGiven)
            model->B3SOIwua = 0.0;
        if (!model->B3SOIwubGiven)
            model->B3SOIwub = 0.0;
        if (!model->B3SOIwucGiven)
            model->B3SOIwuc = 0.0;
        if (!model->B3SOIwvsatGiven)
            model->B3SOIwvsat = 0.0;
        if (!model->B3SOIwa0Given)
            model->B3SOIwa0 = 0.0;
        if (!model->B3SOIwagsGiven)
            model->B3SOIwags = 0.0;
        if (!model->B3SOIwb0Given)
            model->B3SOIwb0 = 0.0;
        if (!model->B3SOIwb1Given)
            model->B3SOIwb1 = 0.0;
        if (!model->B3SOIwketaGiven)
            model->B3SOIwketa = 0.0;
        if (!model->B3SOIwketasGiven)
            model->B3SOIwketas = 0.0;
        if (!model->B3SOIwa1Given)
            model->B3SOIwa1 = 0.0;
        if (!model->B3SOIwa2Given)
            model->B3SOIwa2 = 0.0;
        if (!model->B3SOIwrdswGiven)
            model->B3SOIwrdsw = 0.0;
        if (!model->B3SOIwprwbGiven)
            model->B3SOIwprwb = 0.0;
        if (!model->B3SOIwprwgGiven)
            model->B3SOIwprwg = 0.0;
        if (!model->B3SOIwwrGiven)
            model->B3SOIwwr = 0.0;
        if (!model->B3SOIwnfactorGiven)
            model->B3SOIwnfactor = 0.0;
        if (!model->B3SOIwdwgGiven)
            model->B3SOIwdwg = 0.0;
        if (!model->B3SOIwdwbGiven)
            model->B3SOIwdwb = 0.0;
        if (!model->B3SOIwvoffGiven)
            model->B3SOIwvoff = 0.0;
        if (!model->B3SOIweta0Given)
            model->B3SOIweta0 = 0.0;
        if (!model->B3SOIwetabGiven)
            model->B3SOIwetab = 0.0;
        if (!model->B3SOIwdsubGiven)
            model->B3SOIwdsub = 0.0;
        if (!model->B3SOIwcitGiven)
            model->B3SOIwcit = 0.0;
        if (!model->B3SOIwcdscGiven)
            model->B3SOIwcdsc = 0.0;
        if (!model->B3SOIwcdscbGiven)
            model->B3SOIwcdscb = 0.0;
        if (!model->B3SOIwcdscdGiven)
            model->B3SOIwcdscd = 0.0;
        if (!model->B3SOIwpclmGiven)
            model->B3SOIwpclm = 0.0;
        if (!model->B3SOIwpdibl1Given)
            model->B3SOIwpdibl1 = 0.0;
        if (!model->B3SOIwpdibl2Given)
            model->B3SOIwpdibl2 = 0.0;
        if (!model->B3SOIwpdiblbGiven)
            model->B3SOIwpdiblb = 0.0;
        if (!model->B3SOIwdroutGiven)
            model->B3SOIwdrout = 0.0;
        if (!model->B3SOIwpvagGiven)
            model->B3SOIwpvag = 0.0;
        if (!model->B3SOIwdeltaGiven)
            model->B3SOIwdelta = 0.0;
        if (!model->B3SOIwalpha0Given)
            model->B3SOIwalpha0 = 0.0;
        if (!model->B3SOIwfbjtiiGiven)
            model->B3SOIwfbjtii = 0.0;
        if (!model->B3SOIwbeta0Given)
            model->B3SOIwbeta0 = 0.0;
        if (!model->B3SOIwbeta1Given)
            model->B3SOIwbeta1 = 0.0;
        if (!model->B3SOIwbeta2Given)
            model->B3SOIwbeta2 = 0.0;
        if (!model->B3SOIwvdsatii0Given)
            model->B3SOIwvdsatii0 = 0.0;
        if (!model->B3SOIwliiGiven)
            model->B3SOIwlii = 0.0;
        if (!model->B3SOIwesatiiGiven)
            model->B3SOIwesatii = 0.0;
        if (!model->B3SOIwsii0Given)
            model->B3SOIwsii0 = 0.0;
        if (!model->B3SOIwsii1Given)
            model->B3SOIwsii1 = 0.0;
        if (!model->B3SOIwsii2Given)
            model->B3SOIwsii2 = 0.0;
        if (!model->B3SOIwsiidGiven)
            model->B3SOIwsiid = 0.0;
        if (!model->B3SOIwagidlGiven)
            model->B3SOIwagidl = 0.0;
        if (!model->B3SOIwbgidlGiven)
            model->B3SOIwbgidl = 0.0;
        if (!model->B3SOIwngidlGiven)
            model->B3SOIwngidl = 0.0;
        if (!model->B3SOIwntunGiven)
            model->B3SOIwntun = 0.0;
        if (!model->B3SOIwndiodeGiven)
            model->B3SOIwndiode = 0.0;
        if (!model->B3SOIwnrecf0Given)
            model->B3SOIwnrecf0 = 0.0;
        if (!model->B3SOIwnrecr0Given)
            model->B3SOIwnrecr0 = 0.0;
        if (!model->B3SOIwisbjtGiven)
            model->B3SOIwisbjt = 0.0;
        if (!model->B3SOIwisdifGiven)
            model->B3SOIwisdif = 0.0;
        if (!model->B3SOIwisrecGiven)
            model->B3SOIwisrec = 0.0;
        if (!model->B3SOIwistunGiven)
            model->B3SOIwistun = 0.0;
        if (!model->B3SOIwvrec0Given)
            model->B3SOIwvrec0 = 0.0;
        if (!model->B3SOIwvtun0Given)
            model->B3SOIwvtun0 = 0.0;
        if (!model->B3SOIwnbjtGiven)
            model->B3SOIwnbjt = 0.0;
        if (!model->B3SOIwlbjt0Given)
            model->B3SOIwlbjt0 = 0.0;
        if (!model->B3SOIwvabjtGiven)
            model->B3SOIwvabjt = 0.0;
        if (!model->B3SOIwaelyGiven)
            model->B3SOIwaely = 0.0;
        if (!model->B3SOIwahliGiven)
            model->B3SOIwahli = 0.0;
	/* CV Model */
        if (!model->B3SOIwvsdfbGiven)
            model->B3SOIwvsdfb = 0.0;
        if (!model->B3SOIwvsdthGiven)
            model->B3SOIwvsdth = 0.0;
        if (!model->B3SOIwdelvtGiven)
            model->B3SOIwdelvt = 0.0;
        if (!model->B3SOIwacdeGiven)
            model->B3SOIwacde = 0.0;
        if (!model->B3SOIwmoinGiven)
            model->B3SOIwmoin = 0.0;

        /* Cross-term dependence */
        if (!model->B3SOIpnigcGiven)
            model->B3SOIpnigc = 0.0;
        if (!model->B3SOIppigcdGiven)
            model->B3SOIppigcd = 0.0;
        if (!model->B3SOIppoxedgeGiven)
            model->B3SOIppoxedge = 0.0;
        if (!model->B3SOIpaigcGiven)
            model->B3SOIpaigc = 0.0;
        if (!model->B3SOIpbigcGiven)
            model->B3SOIpbigc = 0.0;
        if (!model->B3SOIpcigcGiven)
            model->B3SOIpcigc = 0.0;
        if (!model->B3SOIpaigsdGiven)
            model->B3SOIpaigsd = 0.0;
        if (!model->B3SOIpbigsdGiven)
            model->B3SOIpbigsd = 0.0;
        if (!model->B3SOIpcigsdGiven)
            model->B3SOIpcigsd = 0.0;

        if (!model->B3SOIpnpeakGiven)
            model->B3SOIpnpeak = 0.0;
        if (!model->B3SOIpnsubGiven)
            model->B3SOIpnsub = 0.0;
        if (!model->B3SOIpngateGiven)
            model->B3SOIpngate = 0.0;
        if (!model->B3SOIpvth0Given)
           model->B3SOIpvth0 = 0.0;
        if (!model->B3SOIpk1Given)
            model->B3SOIpk1 = 0.0;
        if (!model->B3SOIpk1w1Given)
            model->B3SOIpk1w1 = 0.0;
        if (!model->B3SOIpk1w2Given)
            model->B3SOIpk1w2 = 0.0;
        if (!model->B3SOIpk2Given)
            model->B3SOIpk2 = 0.0;
        if (!model->B3SOIpk3Given)
            model->B3SOIpk3 = 0.0;
        if (!model->B3SOIpk3bGiven)
            model->B3SOIpk3b = 0.0;
        if (!model->B3SOIpkb1Given)
           model->B3SOIpkb1 = 0.0;
        if (!model->B3SOIpw0Given)
            model->B3SOIpw0 = 0.0;
        if (!model->B3SOIpnlxGiven)
            model->B3SOIpnlx = 0.0;
        if (!model->B3SOIpdvt0Given)
            model->B3SOIpdvt0 = 0.0;
        if (!model->B3SOIpdvt1Given)
            model->B3SOIpdvt1 = 0.0;
        if (!model->B3SOIpdvt2Given)
            model->B3SOIpdvt2 = 0.0;
        if (!model->B3SOIpdvt0wGiven)
            model->B3SOIpdvt0w = 0.0;
        if (!model->B3SOIpdvt1wGiven)
            model->B3SOIpdvt1w = 0.0;
        if (!model->B3SOIpdvt2wGiven)
            model->B3SOIpdvt2w = 0.0;
        if (!model->B3SOIpu0Given)
            model->B3SOIpu0 = 0.0;
        if (!model->B3SOIpuaGiven)
            model->B3SOIpua = 0.0;
        if (!model->B3SOIpubGiven)
            model->B3SOIpub = 0.0;
        if (!model->B3SOIpucGiven)
            model->B3SOIpuc = 0.0;
        if (!model->B3SOIpvsatGiven)
            model->B3SOIpvsat = 0.0;
        if (!model->B3SOIpa0Given)
            model->B3SOIpa0 = 0.0;
        if (!model->B3SOIpagsGiven)
            model->B3SOIpags = 0.0;
        if (!model->B3SOIpb0Given)
            model->B3SOIpb0 = 0.0;
        if (!model->B3SOIpb1Given)
            model->B3SOIpb1 = 0.0;
        if (!model->B3SOIpketaGiven)
            model->B3SOIpketa = 0.0;
        if (!model->B3SOIpketasGiven)
            model->B3SOIpketas = 0.0;
        if (!model->B3SOIpa1Given)
            model->B3SOIpa1 = 0.0;
        if (!model->B3SOIpa2Given)
            model->B3SOIpa2 = 0.0;
        if (!model->B3SOIprdswGiven)
            model->B3SOIprdsw = 0.0;
        if (!model->B3SOIpprwbGiven)
            model->B3SOIpprwb = 0.0;
        if (!model->B3SOIpprwgGiven)
            model->B3SOIpprwg = 0.0;
        if (!model->B3SOIpwrGiven)
            model->B3SOIpwr = 0.0;
        if (!model->B3SOIpnfactorGiven)
            model->B3SOIpnfactor = 0.0;
        if (!model->B3SOIpdwgGiven)
            model->B3SOIpdwg = 0.0;
        if (!model->B3SOIpdwbGiven)
            model->B3SOIpdwb = 0.0;
        if (!model->B3SOIpvoffGiven)
            model->B3SOIpvoff = 0.0;
        if (!model->B3SOIpeta0Given)
            model->B3SOIpeta0 = 0.0;
        if (!model->B3SOIpetabGiven)
            model->B3SOIpetab = 0.0;
        if (!model->B3SOIpdsubGiven)
            model->B3SOIpdsub = 0.0;
        if (!model->B3SOIpcitGiven)
            model->B3SOIpcit = 0.0;
        if (!model->B3SOIpcdscGiven)
            model->B3SOIpcdsc = 0.0;
        if (!model->B3SOIpcdscbGiven)
            model->B3SOIpcdscb = 0.0;
        if (!model->B3SOIpcdscdGiven)
            model->B3SOIpcdscd = 0.0;
        if (!model->B3SOIppclmGiven)
            model->B3SOIppclm = 0.0;
        if (!model->B3SOIppdibl1Given)
            model->B3SOIppdibl1 = 0.0;
        if (!model->B3SOIppdibl2Given)
            model->B3SOIppdibl2 = 0.0;
        if (!model->B3SOIppdiblbGiven)
            model->B3SOIppdiblb = 0.0;
        if (!model->B3SOIpdroutGiven)
            model->B3SOIpdrout = 0.0;
        if (!model->B3SOIppvagGiven)
            model->B3SOIppvag = 0.0;
        if (!model->B3SOIpdeltaGiven)
            model->B3SOIpdelta = 0.0;
        if (!model->B3SOIpalpha0Given)
            model->B3SOIpalpha0 = 0.0;
        if (!model->B3SOIpfbjtiiGiven)
            model->B3SOIpfbjtii = 0.0;
        if (!model->B3SOIpbeta0Given)
            model->B3SOIpbeta0 = 0.0;
        if (!model->B3SOIpbeta1Given)
            model->B3SOIpbeta1 = 0.0;
        if (!model->B3SOIpbeta2Given)
            model->B3SOIpbeta2 = 0.0;
        if (!model->B3SOIpvdsatii0Given)
            model->B3SOIpvdsatii0 = 0.0;
        if (!model->B3SOIpliiGiven)
            model->B3SOIplii = 0.0;
        if (!model->B3SOIpesatiiGiven)
            model->B3SOIpesatii = 0.0;
        if (!model->B3SOIpsii0Given)
            model->B3SOIpsii0 = 0.0;
        if (!model->B3SOIpsii1Given)
            model->B3SOIpsii1 = 0.0;
        if (!model->B3SOIpsii2Given)
            model->B3SOIpsii2 = 0.0;
        if (!model->B3SOIpsiidGiven)
            model->B3SOIpsiid = 0.0;
        if (!model->B3SOIpagidlGiven)
            model->B3SOIpagidl = 0.0;
        if (!model->B3SOIpbgidlGiven)
            model->B3SOIpbgidl = 0.0;
        if (!model->B3SOIpngidlGiven)
            model->B3SOIpngidl = 0.0;
        if (!model->B3SOIpntunGiven)
            model->B3SOIpntun = 0.0;
        if (!model->B3SOIpndiodeGiven)
            model->B3SOIpndiode = 0.0;
        if (!model->B3SOIpnrecf0Given)
            model->B3SOIpnrecf0 = 0.0;
        if (!model->B3SOIpnrecr0Given)
            model->B3SOIpnrecr0 = 0.0;
        if (!model->B3SOIpisbjtGiven)
            model->B3SOIpisbjt = 0.0;
        if (!model->B3SOIpisdifGiven)
            model->B3SOIpisdif = 0.0;
        if (!model->B3SOIpisrecGiven)
            model->B3SOIpisrec = 0.0;
        if (!model->B3SOIpistunGiven)
            model->B3SOIpistun = 0.0;
        if (!model->B3SOIpvrec0Given)
            model->B3SOIpvrec0 = 0.0;
        if (!model->B3SOIpvtun0Given)
            model->B3SOIpvtun0 = 0.0;
        if (!model->B3SOIpnbjtGiven)
            model->B3SOIpnbjt = 0.0;
        if (!model->B3SOIplbjt0Given)
            model->B3SOIplbjt0 = 0.0;
        if (!model->B3SOIpvabjtGiven)
            model->B3SOIpvabjt = 0.0;
        if (!model->B3SOIpaelyGiven)
            model->B3SOIpaely = 0.0;
        if (!model->B3SOIpahliGiven)
            model->B3SOIpahli = 0.0;
	/* CV Model */
        if (!model->B3SOIpvsdfbGiven)
            model->B3SOIpvsdfb = 0.0;
        if (!model->B3SOIpvsdthGiven)
            model->B3SOIpvsdth = 0.0;
        if (!model->B3SOIpdelvtGiven)
            model->B3SOIpdelvt = 0.0;
        if (!model->B3SOIpacdeGiven)
            model->B3SOIpacde = 0.0;
        if (!model->B3SOIpmoinGiven)
            model->B3SOIpmoin = 0.0;
/* Added for binning - END */

	if (!model->B3SOIcfGiven)
            model->B3SOIcf = 2.0 * EPSOX / PI
			   * log(1.0 + 0.4e-6 / model->B3SOItox);
        if (!model->B3SOIcgdoGiven)
	{   if (model->B3SOIdlcGiven && (model->B3SOIdlc > 0.0))
	    {   model->B3SOIcgdo = model->B3SOIdlc * model->B3SOIcox
				 - model->B3SOIcgdl ;
	    }
	    else
	        model->B3SOIcgdo = 0.6 * model->B3SOIxj * model->B3SOIcox; 
	}
        if (!model->B3SOIcgsoGiven)
	{   if (model->B3SOIdlcGiven && (model->B3SOIdlc > 0.0))
	    {   model->B3SOIcgso = model->B3SOIdlc * model->B3SOIcox
				 - model->B3SOIcgsl ;
	    }
	    else
	        model->B3SOIcgso = 0.6 * model->B3SOIxj * model->B3SOIcox; 
	}

        if (!model->B3SOIcgeoGiven)
	{   model->B3SOIcgeo = 0.0;
	}
        if (!model->B3SOIxpartGiven)
            model->B3SOIxpart = 0.0;
        if (!model->B3SOIsheetResistanceGiven)
            model->B3SOIsheetResistance = 0.0;
        if (!model->B3SOIcsdeswGiven)
            model->B3SOIcsdesw = 0.0;
        if (!model->B3SOIunitLengthGateSidewallJctCapGiven)
            model->B3SOIunitLengthGateSidewallJctCap = 1e-10;
        if (!model->B3SOIGatesidewallJctPotentialGiven)
            model->B3SOIGatesidewallJctPotential = 0.7;
        if (!model->B3SOIbodyJctGateSideGradingCoeffGiven)
            model->B3SOIbodyJctGateSideGradingCoeff = 0.5;
        if (!model->B3SOIoxideTrapDensityAGiven)
	{   if (model->B3SOItype == NMOS)
                model->B3SOIoxideTrapDensityA = 1e20;
            else
                model->B3SOIoxideTrapDensityA=9.9e18;
	}
        if (!model->B3SOIoxideTrapDensityBGiven)
	{   if (model->B3SOItype == NMOS)
                model->B3SOIoxideTrapDensityB = 5e4;
            else
                model->B3SOIoxideTrapDensityB = 2.4e3;
	}
        if (!model->B3SOIoxideTrapDensityCGiven)
	{   if (model->B3SOItype == NMOS)
                model->B3SOIoxideTrapDensityC = -1.4e-12;
            else
                model->B3SOIoxideTrapDensityC = 1.4e-12;

	}
        if (!model->B3SOIemGiven)
            model->B3SOIem = 4.1e7; /* V/m */
        if (!model->B3SOIefGiven)
            model->B3SOIef = 1.0;
        if (!model->B3SOIafGiven)
            model->B3SOIaf = 1.0;
        if (!model->B3SOIkfGiven)
            model->B3SOIkf = 0.0;
        if (!model->B3SOInoifGiven)
            model->B3SOInoif = 1.0;

        /* loop through all the instances of the model */
        for (here = model->B3SOIinstances; here != NULL ;
             here=here->B3SOInextInstance) 
	{   
	
	   if (here->B3SOIowner == ARCHme)
           {
                    /* allocate a chunk of the state vector */
                    here->B3SOIstates = *states;
                    *states += B3SOInumStates;
           }
	   
            /* perform the parameter defaulting */
            if (!here->B3SOIdrainAreaGiven)
                here->B3SOIdrainArea = 0.0;
            if (!here->B3SOIdrainPerimeterGiven)
                here->B3SOIdrainPerimeter = 0.0;
            if (!here->B3SOIdrainSquaresGiven)
                here->B3SOIdrainSquares = 1.0;
            if (!here->B3SOIicVBSGiven)
                here->B3SOIicVBS = 0;
            if (!here->B3SOIicVDSGiven)
                here->B3SOIicVDS = 0;
            if (!here->B3SOIicVGSGiven)
                here->B3SOIicVGS = 0;
            if (!here->B3SOIicVESGiven)
                here->B3SOIicVES = 0;
            if (!here->B3SOIicVPSGiven)
                here->B3SOIicVPS = 0;
	    if (!here->B3SOIbjtoffGiven)
		here->B3SOIbjtoff = 0;
	    if (!here->B3SOIdebugModGiven)
		here->B3SOIdebugMod = 0;
	    if (!here->B3SOIrth0Given)
		here->B3SOIrth0 = model->B3SOIrth0;
	    if (!here->B3SOIcth0Given)
		here->B3SOIcth0 = model->B3SOIcth0;
            if (!here->B3SOIbodySquaresGiven)
                here->B3SOIbodySquares = 1.0;
            if (!here->B3SOIfrbodyGiven)
                here->B3SOIfrbody = 1.0;
            if (!here->B3SOIlGiven)
                here->B3SOIl = 5e-6;
            if (!here->B3SOIsourceAreaGiven)
                here->B3SOIsourceArea = 0;
            if (!here->B3SOIsourcePerimeterGiven)
                here->B3SOIsourcePerimeter = 0;
            if (!here->B3SOIsourceSquaresGiven)
                here->B3SOIsourceSquares = 1;
            if (!here->B3SOIwGiven)
                here->B3SOIw = 5e-6;

            if (!here->B3SOImGiven)
                here->B3SOIm = 1;


/* v2.0 release */
            if (!here->B3SOInbcGiven)
                here->B3SOInbc = 0;
            if (!here->B3SOInsegGiven)
                here->B3SOInseg = 1;
            if (!here->B3SOIpdbcpGiven)
                here->B3SOIpdbcp = 0;
            if (!here->B3SOIpsbcpGiven)
                here->B3SOIpsbcp = 0;
            if (!here->B3SOIagbcpGiven)
                here->B3SOIagbcp = 0;
            if (!here->B3SOIaebcpGiven)
                here->B3SOIaebcp = 0;

            if (!here->B3SOIoffGiven)
                here->B3SOIoff = 0;

            /* process drain series resistance */
            if ((model->B3SOIsheetResistance > 0.0) && 
                (here->B3SOIdrainSquares > 0.0 ) &&
                (here->B3SOIdNodePrime == 0))
	    {   error = CKTmkVolt(ckt,&tmp,here->B3SOIname,"drain");
                if(error) return(error);
                here->B3SOIdNodePrime = tmp->number;
		
		 if (ckt->CKTcopyNodesets) {
                  if (CKTinst2Node(ckt,here,1,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }

            }
	    else
	    {   here->B3SOIdNodePrime = here->B3SOIdNode;
            }
                   
            /* process source series resistance */
            if ((model->B3SOIsheetResistance > 0.0) && 
                (here->B3SOIsourceSquares > 0.0 ) &&
                (here->B3SOIsNodePrime == 0)) 
	    {   error = CKTmkVolt(ckt,&tmp,here->B3SOIname,"source");
                if(error) return(error);
                here->B3SOIsNodePrime = tmp->number;
		
		 if (ckt->CKTcopyNodesets) {
                  if (CKTinst2Node(ckt,here,3,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }
		
            }
	    else 
	    {   here->B3SOIsNodePrime = here->B3SOIsNode;
            }

            /* process effective silicon film thickness */
            model->B3SOIcbox = 3.453133e-11 / model->B3SOItbox;
            model->B3SOIcsi = 1.03594e-10 / model->B3SOItsi;
            Cboxt = model->B3SOIcbox * model->B3SOIcsi / (model->B3SOIcbox + model->B3SOIcsi);
            model->B3SOIqsi = Charge_q*model->B3SOInpeak*1e6*model->B3SOItsi;


            here->B3SOIfloat = 0;
            if (here->B3SOIpNode == -1) {  /*  floating body case -- 4-node  */
                error = CKTmkVolt(ckt,&tmp,here->B3SOIname,"Body");
                if (error) return(error);
                here->B3SOIbNode = tmp->number;
                here->B3SOIpNode = 0; 
                here->B3SOIfloat = 1;
                here->B3SOIbodyMod = 0;
            }
            else /* the 5th Node has been assigned */
            {
              if (!here->B3SOItnodeoutGiven) { /* if t-node not assigned  */
                 if (here->B3SOIbNode == -1)
                 { /* 5-node body tie, bNode has not been assigned */
                    if ((model->B3SOIrbody == 0.0) && (model->B3SOIrbsh == 0.0))
                    { /* ideal body tie, pNode is not used */
                       here->B3SOIbNode = here->B3SOIpNode;
                       here->B3SOIbodyMod = 2;
                    }
                    else { /* nonideal body tie */
                      error = CKTmkVolt(ckt, &tmp, here->B3SOIname, "Body");
                      if (error) return(error);
                      here->B3SOIbNode = tmp->number;
                      here->B3SOIbodyMod = 1;
                    }
                 }
                 else { /* 6-node body tie, bNode has been assigned */
                    if ((model->B3SOIrbody == 0.0) && (model->B3SOIrbsh == 0.0))
                    { 
                       printf("\n      Warning: model parameter rbody=0!\n");
                       model->B3SOIrbody = 1e0;
                       here->B3SOIbodyMod = 1;
                    }
                    else { /* nonideal body tie */
                        here->B3SOIbodyMod = 1;
                    }
                 }
              }
              else {    /*  t-node assigned   */
                 if (here->B3SOIbNode == -1)
                 { /* 4 nodes & t-node, floating body */
                    error = CKTmkVolt(ckt, &tmp, here->B3SOIname, "Body");
                    if (error) return(error);
                    here->B3SOIbNode = tmp->number;
                    here->B3SOItempNode = here->B3SOIpNode;
                    here->B3SOIpNode = 0; 
                    here->B3SOIfloat = 1;
                    here->B3SOIbodyMod = 0;
                 }
                 else { /* 5 or 6 nodes & t-node, body-contact device */
                   if (here->B3SOItempNode == -1) { /* 5 nodes & tnode */
                     if ((model->B3SOIrbody == 0.0) && (model->B3SOIrbsh == 0.0))
                     { /* ideal body tie, pNode is not used */
                        here->B3SOItempNode = here->B3SOIbNode;
                        here->B3SOIbNode = here->B3SOIpNode;
                        here->B3SOIbodyMod = 2;
                     }
                     else { /* nonideal body tie */
                       here->B3SOItempNode = here->B3SOIbNode;
                       error = CKTmkVolt(ckt, &tmp, here->B3SOIname, "Body");
                       if (error) return(error);
                       here->B3SOIbNode = tmp->number;
                       here->B3SOIbodyMod = 1;
                     }
                   }
                   else {  /* 6 nodes & t-node */
                     if ((model->B3SOIrbody == 0.0) && (model->B3SOIrbsh == 0.0))
                     { 
                        printf("\n      Warning: model parameter rbody=0!\n");
                        model->B3SOIrbody = 1e0;
                        here->B3SOIbodyMod = 1;
                     }
                     else { /* nonideal body tie */
                        here->B3SOIbodyMod = 1;
                     }
                   }
                 }
              }
            }


            if ((model->B3SOIshMod == 1) && (here->B3SOIrth0!=0))
            {
               if (here->B3SOItempNode == -1) {
	          error = CKTmkVolt(ckt,&tmp,here->B3SOIname,"Temp");
                  if (error) return(error);
                     here->B3SOItempNode = tmp->number;
               }

            } else {
                here->B3SOItempNode = 0;
            }

/* here for debugging purpose only */
            if (here->B3SOIdebugMod != 0)
            {
               /* The real Vbs value */
               error = CKTmkVolt(ckt, &tmp, here->B3SOIname, "Vbs");
               if(error) return(error);
               here->B3SOIvbsNode = tmp->number;   

               error = CKTmkVolt(ckt, &tmp, here->B3SOIname, "Ids");
               if(error) return(error);
               here->B3SOIidsNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIname, "Ic");
               if(error) return(error);
               here->B3SOIicNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIname, "Ibs");
               if(error) return(error);
               here->B3SOIibsNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIname, "Ibd");
               if(error) return(error);
               here->B3SOIibdNode = tmp->number;
       
               error = CKTmkVolt(ckt, &tmp, here->B3SOIname, "Iii");
               if(error) return(error);
               here->B3SOIiiiNode = tmp->number;
     
               error = CKTmkVolt(ckt, &tmp, here->B3SOIname, "Ig");
               if(error) return(error);
               here->B3SOIigNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIname, "Gigg");
               if(error) return(error);
               here->B3SOIgiggNode = tmp->number;
               
               error = CKTmkVolt(ckt, &tmp, here->B3SOIname, "Gigd");
               if(error) return(error);
               here->B3SOIgigdNode = tmp->number;
              
               error = CKTmkVolt(ckt, &tmp, here->B3SOIname, "Gigb");
               if(error) return(error);
               here->B3SOIgigbNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIname, "Igidl");
               if(error) return(error);
               here->B3SOIigidlNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIname, "Itun");
               if(error) return(error);
               here->B3SOIitunNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIname, "Ibp");
               if(error) return(error);
               here->B3SOIibpNode = tmp->number;
       
               error = CKTmkVolt(ckt, &tmp, here->B3SOIname, "Cbb");
               if(error) return(error);
               here->B3SOIcbbNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIname, "Cbd");
               if(error) return(error);
               here->B3SOIcbdNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIname, "Cbg");
               if(error) return(error);
               here->B3SOIcbgNode = tmp->number;


               error = CKTmkVolt(ckt, &tmp, here->B3SOIname, "Qbf");
               if(error) return(error);
               here->B3SOIqbfNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIname, "Qjs");
               if(error) return(error);
               here->B3SOIqjsNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIname, "Qjd");
               if(error) return(error);
               here->B3SOIqjdNode = tmp->number;


           }

        /* set Sparse Matrix Pointers */

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==(double *)NULL){\
    return(E_NOMEM);\
}


        if ((model->B3SOIshMod == 1) && (here->B3SOIrth0!=0.0)) {
            TSTALLOC(B3SOITemptempPtr, B3SOItempNode, B3SOItempNode)
            TSTALLOC(B3SOITempdpPtr, B3SOItempNode, B3SOIdNodePrime)
            TSTALLOC(B3SOITempspPtr, B3SOItempNode, B3SOIsNodePrime)
            TSTALLOC(B3SOITempgPtr, B3SOItempNode, B3SOIgNode)
            TSTALLOC(B3SOITempbPtr, B3SOItempNode, B3SOIbNode)

            TSTALLOC(B3SOIGtempPtr, B3SOIgNode, B3SOItempNode)
            TSTALLOC(B3SOIDPtempPtr, B3SOIdNodePrime, B3SOItempNode)
            TSTALLOC(B3SOISPtempPtr, B3SOIsNodePrime, B3SOItempNode)
            TSTALLOC(B3SOIEtempPtr, B3SOIeNode, B3SOItempNode)
            TSTALLOC(B3SOIBtempPtr, B3SOIbNode, B3SOItempNode)

            if (here->B3SOIbodyMod == 1) {
                TSTALLOC(B3SOIPtempPtr, B3SOIpNode, B3SOItempNode)
            }

/* v3.0 */
            if (model->B3SOIsoiMod != 0) {
               TSTALLOC(B3SOITempePtr, B3SOItempNode, B3SOIeNode) 
            }

        }
        if (here->B3SOIbodyMod == 2) {
            /* Don't create any Jacobian entry for pNode */
        }
        else if (here->B3SOIbodyMod == 1) {
            TSTALLOC(B3SOIBpPtr, B3SOIbNode, B3SOIpNode)
            TSTALLOC(B3SOIPbPtr, B3SOIpNode, B3SOIbNode)
            TSTALLOC(B3SOIPpPtr, B3SOIpNode, B3SOIpNode)
        }

        TSTALLOC(B3SOIEbPtr, B3SOIeNode, B3SOIbNode)
        TSTALLOC(B3SOIGbPtr, B3SOIgNode, B3SOIbNode)
        TSTALLOC(B3SOIDPbPtr, B3SOIdNodePrime, B3SOIbNode)
        TSTALLOC(B3SOISPbPtr, B3SOIsNodePrime, B3SOIbNode)
        TSTALLOC(B3SOIBePtr, B3SOIbNode, B3SOIeNode)
        TSTALLOC(B3SOIBgPtr, B3SOIbNode, B3SOIgNode)
        TSTALLOC(B3SOIBdpPtr, B3SOIbNode, B3SOIdNodePrime)
        TSTALLOC(B3SOIBspPtr, B3SOIbNode, B3SOIsNodePrime)
        TSTALLOC(B3SOIBbPtr, B3SOIbNode, B3SOIbNode)

        TSTALLOC(B3SOIEgPtr, B3SOIeNode, B3SOIgNode)
        TSTALLOC(B3SOIEdpPtr, B3SOIeNode, B3SOIdNodePrime)
        TSTALLOC(B3SOIEspPtr, B3SOIeNode, B3SOIsNodePrime)
        TSTALLOC(B3SOIGePtr, B3SOIgNode, B3SOIeNode)
        TSTALLOC(B3SOIDPePtr, B3SOIdNodePrime, B3SOIeNode)
        TSTALLOC(B3SOISPePtr, B3SOIsNodePrime, B3SOIeNode)

        TSTALLOC(B3SOIEbPtr, B3SOIeNode, B3SOIbNode)
        TSTALLOC(B3SOIEePtr, B3SOIeNode, B3SOIeNode)

        TSTALLOC(B3SOIGgPtr, B3SOIgNode, B3SOIgNode)
        TSTALLOC(B3SOIGdpPtr, B3SOIgNode, B3SOIdNodePrime)
        TSTALLOC(B3SOIGspPtr, B3SOIgNode, B3SOIsNodePrime)

        TSTALLOC(B3SOIDPgPtr, B3SOIdNodePrime, B3SOIgNode)
        TSTALLOC(B3SOIDPdpPtr, B3SOIdNodePrime, B3SOIdNodePrime)
        TSTALLOC(B3SOIDPspPtr, B3SOIdNodePrime, B3SOIsNodePrime)
        TSTALLOC(B3SOIDPdPtr, B3SOIdNodePrime, B3SOIdNode)

        TSTALLOC(B3SOISPgPtr, B3SOIsNodePrime, B3SOIgNode)
        TSTALLOC(B3SOISPdpPtr, B3SOIsNodePrime, B3SOIdNodePrime)
        TSTALLOC(B3SOISPspPtr, B3SOIsNodePrime, B3SOIsNodePrime)
        TSTALLOC(B3SOISPsPtr, B3SOIsNodePrime, B3SOIsNode)

        TSTALLOC(B3SOIDdPtr, B3SOIdNode, B3SOIdNode)
        TSTALLOC(B3SOIDdpPtr, B3SOIdNode, B3SOIdNodePrime)

        TSTALLOC(B3SOISsPtr, B3SOIsNode, B3SOIsNode)
        TSTALLOC(B3SOISspPtr, B3SOIsNode, B3SOIsNodePrime)

/* here for debugging purpose only */
         if (here->B3SOIdebugMod != 0)
         {
            TSTALLOC(B3SOIVbsPtr, B3SOIvbsNode, B3SOIvbsNode) 
            TSTALLOC(B3SOIIdsPtr, B3SOIidsNode, B3SOIidsNode)
            TSTALLOC(B3SOIIcPtr, B3SOIicNode, B3SOIicNode)
            TSTALLOC(B3SOIIbsPtr, B3SOIibsNode, B3SOIibsNode)
            TSTALLOC(B3SOIIbdPtr, B3SOIibdNode, B3SOIibdNode)
            TSTALLOC(B3SOIIiiPtr, B3SOIiiiNode, B3SOIiiiNode)
            TSTALLOC(B3SOIIgPtr, B3SOIigNode, B3SOIigNode)
            TSTALLOC(B3SOIGiggPtr, B3SOIgiggNode, B3SOIgiggNode)
            TSTALLOC(B3SOIGigdPtr, B3SOIgigdNode, B3SOIgigdNode)
            TSTALLOC(B3SOIGigbPtr, B3SOIgigbNode, B3SOIgigbNode)
            TSTALLOC(B3SOIIgidlPtr, B3SOIigidlNode, B3SOIigidlNode)
            TSTALLOC(B3SOIItunPtr, B3SOIitunNode, B3SOIitunNode)
            TSTALLOC(B3SOIIbpPtr, B3SOIibpNode, B3SOIibpNode)
            TSTALLOC(B3SOICbbPtr, B3SOIcbbNode, B3SOIcbbNode)
            TSTALLOC(B3SOICbdPtr, B3SOIcbdNode, B3SOIcbdNode)
            TSTALLOC(B3SOICbgPtr, B3SOIcbgNode, B3SOIcbgNode)
            TSTALLOC(B3SOIQbfPtr, B3SOIqbfNode, B3SOIqbfNode)
            TSTALLOC(B3SOIQjsPtr, B3SOIqjsNode, B3SOIqjsNode)
            TSTALLOC(B3SOIQjdPtr, B3SOIqjdNode, B3SOIqjdNode)

         }

        }
    }
    return(OK);
}  

int
B3SOIunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    B3SOImodel *model;
    B3SOIinstance *here;
 
    for (model = (B3SOImodel *)inModel; model != NULL;
            model = model->B3SOInextModel)
    {
        for (here = model->B3SOIinstances; here != NULL;
                here=here->B3SOInextInstance)
        {
            if (here->B3SOIdNodePrime
                    && here->B3SOIdNodePrime != here->B3SOIdNode)
            {
                CKTdltNNum(ckt, here->B3SOIdNodePrime);
                here->B3SOIdNodePrime = 0;
            }
            if (here->B3SOIsNodePrime
                    && here->B3SOIsNodePrime != here->B3SOIsNode)
            {
                CKTdltNNum(ckt, here->B3SOIsNodePrime);
                here->B3SOIsNodePrime = 0;
            }
        }
    }
    return OK;
}

