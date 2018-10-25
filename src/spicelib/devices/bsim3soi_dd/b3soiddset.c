/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: Weidong Liu and Pin Su         Feb 1999
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
Modified by Pin Su, Wei Jin 99/9/27
File: b3soiddset.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMDD2.1 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "b3soidddef.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#define MAX_EXP 5.834617425e14
#define MIN_EXP 1.713908431e-15
#define EXP_THRESHOLD 34.0
#define SMOOTHFACTOR 0.1
#define EPSOX 3.453133e-11
#define EPSSI 1.03594e-10
#define PI 3.141592654
#define Charge_q 1.60219e-19
#define Meter2Micron 1.0e6

int
B3SOIDDsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, 
             int *states)
{
B3SOIDDmodel *model = (B3SOIDDmodel*)inModel;
B3SOIDDinstance *here;
int error;
CKTnode *tmp;

double tmp1, tmp2;
double nfb0, Cboxt;

CKTnode *tmpNode;
IFuid tmpName;


    /*  loop through all the B3SOIDD device models */
    for( ; model != NULL; model = B3SOIDDnextModel(model))
    {
/* Default value Processing for B3SOIDD MOSFET Models */

        if (!model->B3SOIDDtypeGiven)
            model->B3SOIDDtype = NMOS;     
        if (!model->B3SOIDDmobModGiven) 
            model->B3SOIDDmobMod = 1;
        if (!model->B3SOIDDbinUnitGiven) 
            model->B3SOIDDbinUnit = 1;
        if (!model->B3SOIDDparamChkGiven) 
            model->B3SOIDDparamChk = 0;
        if (!model->B3SOIDDcapModGiven) 
            model->B3SOIDDcapMod = 2;
        if (!model->B3SOIDDnoiModGiven) 
            model->B3SOIDDnoiMod = 1;
        if (!model->B3SOIDDshModGiven) 
            model->B3SOIDDshMod = 0;
        if (!model->B3SOIDDversionGiven) 
            model->B3SOIDDversion = 2.0;
        if (!model->B3SOIDDtoxGiven)
            model->B3SOIDDtox = 100.0e-10;
        model->B3SOIDDcox = 3.453133e-11 / model->B3SOIDDtox;

        if (!model->B3SOIDDcdscGiven)
	    model->B3SOIDDcdsc = 2.4e-4;   /* unit Q/V/m^2  */
        if (!model->B3SOIDDcdscbGiven)
	    model->B3SOIDDcdscb = 0.0;   /* unit Q/V/m^2  */    
        if (!model->B3SOIDDcdscdGiven)
	    model->B3SOIDDcdscd = 0.0;   /* unit Q/V/m^2  */
        if (!model->B3SOIDDcitGiven)
	    model->B3SOIDDcit = 0.0;   /* unit Q/V/m^2  */
        if (!model->B3SOIDDnfactorGiven)
	    model->B3SOIDDnfactor = 1;
        if (!model->B3SOIDDvsatGiven)
            model->B3SOIDDvsat = 8.0e4;    /* unit m/s */ 
        if (!model->B3SOIDDatGiven)
            model->B3SOIDDat = 3.3e4;    /* unit m/s */ 
        if (!model->B3SOIDDa0Given)
            model->B3SOIDDa0 = 1.0;  
        if (!model->B3SOIDDagsGiven)
            model->B3SOIDDags = 0.0;
        if (!model->B3SOIDDa1Given)
            model->B3SOIDDa1 = 0.0;
        if (!model->B3SOIDDa2Given)
            model->B3SOIDDa2 = 1.0;
        if (!model->B3SOIDDketaGiven)
            model->B3SOIDDketa = -0.6;    /* unit  / V */
        if (!model->B3SOIDDnsubGiven)
            model->B3SOIDDnsub = 6.0e16;   /* unit 1/cm3 */
        if (!model->B3SOIDDnpeakGiven)
            model->B3SOIDDnpeak = 1.7e17;   /* unit 1/cm3 */
        if (!model->B3SOIDDngateGiven)
            model->B3SOIDDngate = 0;   /* unit 1/cm3 */
        if (!model->B3SOIDDvbmGiven)
	    model->B3SOIDDvbm = -3.0;
        if (!model->B3SOIDDxtGiven)
	    model->B3SOIDDxt = 1.55e-7;
        if (!model->B3SOIDDkt1Given)
            model->B3SOIDDkt1 = -0.11;      /* unit V */
        if (!model->B3SOIDDkt1lGiven)
            model->B3SOIDDkt1l = 0.0;      /* unit V*m */
        if (!model->B3SOIDDkt2Given)
            model->B3SOIDDkt2 = 0.022;      /* No unit */
        if (!model->B3SOIDDk3Given)
            model->B3SOIDDk3 = 0.0;      
        if (!model->B3SOIDDk3bGiven)
            model->B3SOIDDk3b = 0.0;      
        if (!model->B3SOIDDw0Given)
            model->B3SOIDDw0 = 2.5e-6;    
        if (!model->B3SOIDDnlxGiven)
            model->B3SOIDDnlx = 1.74e-7;     
        if (!model->B3SOIDDdvt0Given)
            model->B3SOIDDdvt0 = 2.2;    
        if (!model->B3SOIDDdvt1Given)
            model->B3SOIDDdvt1 = 0.53;      
        if (!model->B3SOIDDdvt2Given)
            model->B3SOIDDdvt2 = -0.032;   /* unit 1 / V */     

        if (!model->B3SOIDDdvt0wGiven)
            model->B3SOIDDdvt0w = 0.0;    
        if (!model->B3SOIDDdvt1wGiven)
            model->B3SOIDDdvt1w = 5.3e6;    
        if (!model->B3SOIDDdvt2wGiven)
            model->B3SOIDDdvt2w = -0.032;   

        if (!model->B3SOIDDdroutGiven)
            model->B3SOIDDdrout = 0.56;     
        if (!model->B3SOIDDdsubGiven)
            model->B3SOIDDdsub = model->B3SOIDDdrout;     
        if (!model->B3SOIDDvth0Given)
            model->B3SOIDDvth0 = (model->B3SOIDDtype == NMOS) ? 0.7 : -0.7;
        if (!model->B3SOIDDuaGiven)
            model->B3SOIDDua = 2.25e-9;      /* unit m/V */
        if (!model->B3SOIDDua1Given)
            model->B3SOIDDua1 = 4.31e-9;      /* unit m/V */
        if (!model->B3SOIDDubGiven)
            model->B3SOIDDub = 5.87e-19;     /* unit (m/V)**2 */
        if (!model->B3SOIDDub1Given)
            model->B3SOIDDub1 = -7.61e-18;     /* unit (m/V)**2 */
        if (!model->B3SOIDDucGiven)
            model->B3SOIDDuc = (model->B3SOIDDmobMod == 3) ? -0.0465 : -0.0465e-9;   
        if (!model->B3SOIDDuc1Given)
            model->B3SOIDDuc1 = (model->B3SOIDDmobMod == 3) ? -0.056 : -0.056e-9;   
        if (!model->B3SOIDDu0Given)
            model->B3SOIDDu0 = (model->B3SOIDDtype == NMOS) ? 0.067 : 0.025;
        if (!model->B3SOIDDuteGiven)
	    model->B3SOIDDute = -1.5;    
        if (!model->B3SOIDDvoffGiven)
	    model->B3SOIDDvoff = -0.08;
        if (!model->B3SOIDDdeltaGiven)  
           model->B3SOIDDdelta = 0.01;
        if (!model->B3SOIDDrdswGiven)
            model->B3SOIDDrdsw = 100;
        if (!model->B3SOIDDprwgGiven)
            model->B3SOIDDprwg = 0.0;      /* unit 1/V */
        if (!model->B3SOIDDprwbGiven)
            model->B3SOIDDprwb = 0.0;      
        if (!model->B3SOIDDprtGiven)
            model->B3SOIDDprt = 0.0;      
        if (!model->B3SOIDDeta0Given)
            model->B3SOIDDeta0 = 0.08;      /* no unit  */ 
        if (!model->B3SOIDDetabGiven)
            model->B3SOIDDetab = -0.07;      /* unit  1/V */ 
        if (!model->B3SOIDDpclmGiven)
            model->B3SOIDDpclm = 1.3;      /* no unit  */ 
        if (!model->B3SOIDDpdibl1Given)
            model->B3SOIDDpdibl1 = .39;    /* no unit  */
        if (!model->B3SOIDDpdibl2Given)
            model->B3SOIDDpdibl2 = 0.0086;    /* no unit  */ 
        if (!model->B3SOIDDpdiblbGiven)
            model->B3SOIDDpdiblb = 0.0;    /* 1/V  */ 
        if (!model->B3SOIDDpvagGiven)
            model->B3SOIDDpvag = 0.0;     
        if (!model->B3SOIDDwrGiven)  
            model->B3SOIDDwr = 1.0;
        if (!model->B3SOIDDdwgGiven)  
            model->B3SOIDDdwg = 0.0;
        if (!model->B3SOIDDdwbGiven)  
            model->B3SOIDDdwb = 0.0;
        if (!model->B3SOIDDb0Given)
            model->B3SOIDDb0 = 0.0;
        if (!model->B3SOIDDb1Given)  
            model->B3SOIDDb1 = 0.0;
        if (!model->B3SOIDDalpha0Given)  
            model->B3SOIDDalpha0 = 0.0;
        if (!model->B3SOIDDalpha1Given)  
            model->B3SOIDDalpha1 = 1.0;
        if (!model->B3SOIDDbeta0Given)  
            model->B3SOIDDbeta0 = 30.0;

        if (!model->B3SOIDDcgslGiven)  
            model->B3SOIDDcgsl = 0.0;
        if (!model->B3SOIDDcgdlGiven)  
            model->B3SOIDDcgdl = 0.0;
        if (!model->B3SOIDDckappaGiven)  
            model->B3SOIDDckappa = 0.6;
        if (!model->B3SOIDDclcGiven)  
            model->B3SOIDDclc = 0.1e-7;
        if (!model->B3SOIDDcleGiven)  
            model->B3SOIDDcle = 0.0;
        if (!model->B3SOIDDtboxGiven)  
            model->B3SOIDDtbox = 3e-7;
        if (!model->B3SOIDDtsiGiven)  
            model->B3SOIDDtsi = 1e-7;
        if (!model->B3SOIDDxjGiven)  
            model->B3SOIDDxj = model->B3SOIDDtsi;
        if (!model->B3SOIDDkb1Given)  
            model->B3SOIDDkb1 = 1;
        if (!model->B3SOIDDkb3Given)  
            model->B3SOIDDkb3 = 1;
        if (!model->B3SOIDDdvbd0Given)  
            model->B3SOIDDdvbd0 = 0.0;
        if (!model->B3SOIDDdvbd1Given)  
            model->B3SOIDDdvbd1 = 0.0;
        if (!model->B3SOIDDvbsaGiven)  
            model->B3SOIDDvbsa = 0.0;
        if (!model->B3SOIDDdelpGiven)  
            model->B3SOIDDdelp = 0.02;
        if (!model->B3SOIDDrbodyGiven)  
            model->B3SOIDDrbody = 0.0;
        if (!model->B3SOIDDrbshGiven)  
            model->B3SOIDDrbsh = 0.0;
        if (!model->B3SOIDDadice0Given)  
            model->B3SOIDDadice0 = 1;
        if (!model->B3SOIDDabpGiven)  
            model->B3SOIDDabp = 1;
        if (!model->B3SOIDDmxcGiven)  
            model->B3SOIDDmxc = -0.9;
        if (!model->B3SOIDDrth0Given)  
            model->B3SOIDDrth0 = 0;
        if (!model->B3SOIDDcth0Given)
            model->B3SOIDDcth0 = 0;

        if (!model->B3SOIDDaiiGiven)  
            model->B3SOIDDaii = 0.0;
        if (!model->B3SOIDDbiiGiven)  
            model->B3SOIDDbii = 0.0;
        if (!model->B3SOIDDciiGiven)  
            model->B3SOIDDcii = 0.0;
        if (!model->B3SOIDDdiiGiven)  
            model->B3SOIDDdii = -1.0;
        if (!model->B3SOIDDagidlGiven)  
            model->B3SOIDDagidl = 0.0;
        if (!model->B3SOIDDbgidlGiven)  
            model->B3SOIDDbgidl = 0.0;
        if (!model->B3SOIDDngidlGiven)  
            model->B3SOIDDngidl = 1.2;
        if (!model->B3SOIDDndiodeGiven)  
            model->B3SOIDDndiode = 1.0;
        if (!model->B3SOIDDntunGiven)  
            model->B3SOIDDntun = 10.0;
        if (!model->B3SOIDDisbjtGiven)  
            model->B3SOIDDisbjt = 1e-6;
        if (!model->B3SOIDDisdifGiven)  
            model->B3SOIDDisdif = 0.0;
        if (!model->B3SOIDDisrecGiven)  
            model->B3SOIDDisrec = 1e-5;
        if (!model->B3SOIDDistunGiven)  
            model->B3SOIDDistun = 0.0;
        if (!model->B3SOIDDxbjtGiven)  
            model->B3SOIDDxbjt = 2;
        if (!model->B3SOIDDxdifGiven)  
            model->B3SOIDDxdif = 2;
        if (!model->B3SOIDDxrecGiven)  
            model->B3SOIDDxrec = 20;
        if (!model->B3SOIDDxtunGiven)  
            model->B3SOIDDxtun = 0;
        if (!model->B3SOIDDedlGiven)  
            model->B3SOIDDedl = 2e-6;
        if (!model->B3SOIDDkbjt1Given)  
            model->B3SOIDDkbjt1 = 0;
        if (!model->B3SOIDDttGiven)  
            model->B3SOIDDtt = 1e-12;
        if (!model->B3SOIDDasdGiven)  
            model->B3SOIDDasd = 0.3;

        /* unit degree celcius */
        if (!model->B3SOIDDtnomGiven)  
	    model->B3SOIDDtnom = ckt->CKTnomTemp; 
        if (!model->B3SOIDDLintGiven)  
           model->B3SOIDDLint = 0.0;
        if (!model->B3SOIDDLlGiven)  
           model->B3SOIDDLl = 0.0;
        if (!model->B3SOIDDLlnGiven)  
           model->B3SOIDDLln = 1.0;
        if (!model->B3SOIDDLwGiven)  
           model->B3SOIDDLw = 0.0;
        if (!model->B3SOIDDLwnGiven)  
           model->B3SOIDDLwn = 1.0;
        if (!model->B3SOIDDLwlGiven)  
           model->B3SOIDDLwl = 0.0;
        if (!model->B3SOIDDLminGiven)  
           model->B3SOIDDLmin = 0.0;
        if (!model->B3SOIDDLmaxGiven)  
           model->B3SOIDDLmax = 1.0;
        if (!model->B3SOIDDWintGiven)  
           model->B3SOIDDWint = 0.0;
        if (!model->B3SOIDDWlGiven)  
           model->B3SOIDDWl = 0.0;
        if (!model->B3SOIDDWlnGiven)  
           model->B3SOIDDWln = 1.0;
        if (!model->B3SOIDDWwGiven)  
           model->B3SOIDDWw = 0.0;
        if (!model->B3SOIDDWwnGiven)  
           model->B3SOIDDWwn = 1.0;
        if (!model->B3SOIDDWwlGiven)  
           model->B3SOIDDWwl = 0.0;
        if (!model->B3SOIDDWminGiven)  
           model->B3SOIDDWmin = 0.0;
        if (!model->B3SOIDDWmaxGiven)  
           model->B3SOIDDWmax = 1.0;
        if (!model->B3SOIDDdwcGiven)  
           model->B3SOIDDdwc = model->B3SOIDDWint;
        if (!model->B3SOIDDdlcGiven)  
           model->B3SOIDDdlc = model->B3SOIDDLint;

/* Added for binning - START */
        /* Length dependence */
        if (!model->B3SOIDDlnpeakGiven)
            model->B3SOIDDlnpeak = 0.0;
        if (!model->B3SOIDDlnsubGiven)
            model->B3SOIDDlnsub = 0.0;
        if (!model->B3SOIDDlngateGiven)
            model->B3SOIDDlngate = 0.0;
        if (!model->B3SOIDDlvth0Given)
           model->B3SOIDDlvth0 = 0.0;
        if (!model->B3SOIDDlk1Given)
            model->B3SOIDDlk1 = 0.0;
        if (!model->B3SOIDDlk2Given)
            model->B3SOIDDlk2 = 0.0;
        if (!model->B3SOIDDlk3Given)
            model->B3SOIDDlk3 = 0.0;
        if (!model->B3SOIDDlk3bGiven)
            model->B3SOIDDlk3b = 0.0;
        if (!model->B3SOIDDlvbsaGiven)
            model->B3SOIDDlvbsa = 0.0;
        if (!model->B3SOIDDldelpGiven)
            model->B3SOIDDldelp = 0.0;
        if (!model->B3SOIDDlkb1Given)
           model->B3SOIDDlkb1 = 0.0;
        if (!model->B3SOIDDlkb3Given)
           model->B3SOIDDlkb3 = 1.0;
        if (!model->B3SOIDDldvbd0Given)
           model->B3SOIDDldvbd0 = 1.0;
        if (!model->B3SOIDDldvbd1Given)
           model->B3SOIDDldvbd1 = 1.0;
        if (!model->B3SOIDDlw0Given)
            model->B3SOIDDlw0 = 0.0;
        if (!model->B3SOIDDlnlxGiven)
            model->B3SOIDDlnlx = 0.0;
        if (!model->B3SOIDDldvt0Given)
            model->B3SOIDDldvt0 = 0.0;
        if (!model->B3SOIDDldvt1Given)
            model->B3SOIDDldvt1 = 0.0;
        if (!model->B3SOIDDldvt2Given)
            model->B3SOIDDldvt2 = 0.0;
        if (!model->B3SOIDDldvt0wGiven)
            model->B3SOIDDldvt0w = 0.0;
        if (!model->B3SOIDDldvt1wGiven)
            model->B3SOIDDldvt1w = 0.0;
        if (!model->B3SOIDDldvt2wGiven)
            model->B3SOIDDldvt2w = 0.0;
        if (!model->B3SOIDDlu0Given)
            model->B3SOIDDlu0 = 0.0;
        if (!model->B3SOIDDluaGiven)
            model->B3SOIDDlua = 0.0;
        if (!model->B3SOIDDlubGiven)
            model->B3SOIDDlub = 0.0;
        if (!model->B3SOIDDlucGiven)
            model->B3SOIDDluc = 0.0;
        if (!model->B3SOIDDlvsatGiven)
            model->B3SOIDDlvsat = 0.0;
        if (!model->B3SOIDDla0Given)
            model->B3SOIDDla0 = 0.0;
        if (!model->B3SOIDDlagsGiven)
            model->B3SOIDDlags = 0.0;
        if (!model->B3SOIDDlb0Given)
            model->B3SOIDDlb0 = 0.0;
        if (!model->B3SOIDDlb1Given)
            model->B3SOIDDlb1 = 0.0;
        if (!model->B3SOIDDlketaGiven)
            model->B3SOIDDlketa = 0.0;
        if (!model->B3SOIDDlabpGiven)
            model->B3SOIDDlabp = 0.0;
        if (!model->B3SOIDDlmxcGiven)
            model->B3SOIDDlmxc = 0.0;
        if (!model->B3SOIDDladice0Given)
            model->B3SOIDDladice0 = 0.0;
        if (!model->B3SOIDDla1Given)
            model->B3SOIDDla1 = 0.0;
        if (!model->B3SOIDDla2Given)
            model->B3SOIDDla2 = 0.0;
        if (!model->B3SOIDDlrdswGiven)
            model->B3SOIDDlrdsw = 0.0;
        if (!model->B3SOIDDlprwbGiven)
            model->B3SOIDDlprwb = 0.0;
        if (!model->B3SOIDDlprwgGiven)
            model->B3SOIDDlprwg = 0.0;
        if (!model->B3SOIDDlwrGiven)
            model->B3SOIDDlwr = 0.0;
        if (!model->B3SOIDDlnfactorGiven)
            model->B3SOIDDlnfactor = 0.0;
        if (!model->B3SOIDDldwgGiven)
            model->B3SOIDDldwg = 0.0;
        if (!model->B3SOIDDldwbGiven)
            model->B3SOIDDldwb = 0.0;
        if (!model->B3SOIDDlvoffGiven)
            model->B3SOIDDlvoff = 0.0;
        if (!model->B3SOIDDleta0Given)
            model->B3SOIDDleta0 = 0.0;
        if (!model->B3SOIDDletabGiven)
            model->B3SOIDDletab = 0.0;
        if (!model->B3SOIDDldsubGiven)
            model->B3SOIDDldsub = 0.0;
        if (!model->B3SOIDDlcitGiven)
            model->B3SOIDDlcit = 0.0;
        if (!model->B3SOIDDlcdscGiven)
            model->B3SOIDDlcdsc = 0.0;
        if (!model->B3SOIDDlcdscbGiven)
            model->B3SOIDDlcdscb = 0.0;
        if (!model->B3SOIDDlcdscdGiven)
            model->B3SOIDDlcdscd = 0.0;
        if (!model->B3SOIDDlpclmGiven)
            model->B3SOIDDlpclm = 0.0;
        if (!model->B3SOIDDlpdibl1Given)
            model->B3SOIDDlpdibl1 = 0.0;
        if (!model->B3SOIDDlpdibl2Given)
            model->B3SOIDDlpdibl2 = 0.0;
        if (!model->B3SOIDDlpdiblbGiven)
            model->B3SOIDDlpdiblb = 0.0;
        if (!model->B3SOIDDldroutGiven)
            model->B3SOIDDldrout = 0.0;
        if (!model->B3SOIDDlpvagGiven)
            model->B3SOIDDlpvag = 0.0;
        if (!model->B3SOIDDldeltaGiven)
            model->B3SOIDDldelta = 0.0;
        if (!model->B3SOIDDlaiiGiven)
            model->B3SOIDDlaii = 0.0;
        if (!model->B3SOIDDlbiiGiven)
            model->B3SOIDDlbii = 0.0;
        if (!model->B3SOIDDlciiGiven)
            model->B3SOIDDlcii = 0.0;
        if (!model->B3SOIDDldiiGiven)
            model->B3SOIDDldii = 0.0;
        if (!model->B3SOIDDlalpha0Given)
            model->B3SOIDDlalpha0 = 0.0;
        if (!model->B3SOIDDlalpha1Given)
            model->B3SOIDDlalpha1 = 0.0;
        if (!model->B3SOIDDlbeta0Given)
            model->B3SOIDDlbeta0 = 0.0;
        if (!model->B3SOIDDlagidlGiven)
            model->B3SOIDDlagidl = 0.0;
        if (!model->B3SOIDDlbgidlGiven)
            model->B3SOIDDlbgidl = 0.0;
        if (!model->B3SOIDDlngidlGiven)
            model->B3SOIDDlngidl = 0.0;
        if (!model->B3SOIDDlntunGiven)
            model->B3SOIDDlntun = 0.0;
        if (!model->B3SOIDDlndiodeGiven)
            model->B3SOIDDlndiode = 0.0;
        if (!model->B3SOIDDlisbjtGiven)
            model->B3SOIDDlisbjt = 0.0;
        if (!model->B3SOIDDlisdifGiven)
            model->B3SOIDDlisdif = 0.0;
        if (!model->B3SOIDDlisrecGiven)
            model->B3SOIDDlisrec = 0.0;
        if (!model->B3SOIDDlistunGiven)
            model->B3SOIDDlistun = 0.0;
        if (!model->B3SOIDDledlGiven)
            model->B3SOIDDledl = 0.0;
        if (!model->B3SOIDDlkbjt1Given)
            model->B3SOIDDlkbjt1 = 0.0;
	/* CV Model */
        if (!model->B3SOIDDlvsdfbGiven)
            model->B3SOIDDlvsdfb = 0.0;
        if (!model->B3SOIDDlvsdthGiven)
            model->B3SOIDDlvsdth = 0.0;
        /* Width dependence */
        if (!model->B3SOIDDwnpeakGiven)
            model->B3SOIDDwnpeak = 0.0;
        if (!model->B3SOIDDwnsubGiven)
            model->B3SOIDDwnsub = 0.0;
        if (!model->B3SOIDDwngateGiven)
            model->B3SOIDDwngate = 0.0;
        if (!model->B3SOIDDwvth0Given)
           model->B3SOIDDwvth0 = 0.0;
        if (!model->B3SOIDDwk1Given)
            model->B3SOIDDwk1 = 0.0;
        if (!model->B3SOIDDwk2Given)
            model->B3SOIDDwk2 = 0.0;
        if (!model->B3SOIDDwk3Given)
            model->B3SOIDDwk3 = 0.0;
        if (!model->B3SOIDDwk3bGiven)
            model->B3SOIDDwk3b = 0.0;
        if (!model->B3SOIDDwvbsaGiven)
            model->B3SOIDDwvbsa = 0.0;
        if (!model->B3SOIDDwdelpGiven)
            model->B3SOIDDwdelp = 0.0;
        if (!model->B3SOIDDwkb1Given)
           model->B3SOIDDwkb1 = 0.0;
        if (!model->B3SOIDDwkb3Given)
           model->B3SOIDDwkb3 = 1.0;
        if (!model->B3SOIDDwdvbd0Given)
           model->B3SOIDDwdvbd0 = 1.0;
        if (!model->B3SOIDDwdvbd1Given)
           model->B3SOIDDwdvbd1 = 1.0;
        if (!model->B3SOIDDww0Given)
            model->B3SOIDDww0 = 0.0;
        if (!model->B3SOIDDwnlxGiven)
            model->B3SOIDDwnlx = 0.0;
        if (!model->B3SOIDDwdvt0Given)
            model->B3SOIDDwdvt0 = 0.0;
        if (!model->B3SOIDDwdvt1Given)
            model->B3SOIDDwdvt1 = 0.0;
        if (!model->B3SOIDDwdvt2Given)
            model->B3SOIDDwdvt2 = 0.0;
        if (!model->B3SOIDDwdvt0wGiven)
            model->B3SOIDDwdvt0w = 0.0;
        if (!model->B3SOIDDwdvt1wGiven)
            model->B3SOIDDwdvt1w = 0.0;
        if (!model->B3SOIDDwdvt2wGiven)
            model->B3SOIDDwdvt2w = 0.0;
        if (!model->B3SOIDDwu0Given)
            model->B3SOIDDwu0 = 0.0;
        if (!model->B3SOIDDwuaGiven)
            model->B3SOIDDwua = 0.0;
        if (!model->B3SOIDDwubGiven)
            model->B3SOIDDwub = 0.0;
        if (!model->B3SOIDDwucGiven)
            model->B3SOIDDwuc = 0.0;
        if (!model->B3SOIDDwvsatGiven)
            model->B3SOIDDwvsat = 0.0;
        if (!model->B3SOIDDwa0Given)
            model->B3SOIDDwa0 = 0.0;
        if (!model->B3SOIDDwagsGiven)
            model->B3SOIDDwags = 0.0;
        if (!model->B3SOIDDwb0Given)
            model->B3SOIDDwb0 = 0.0;
        if (!model->B3SOIDDwb1Given)
            model->B3SOIDDwb1 = 0.0;
        if (!model->B3SOIDDwketaGiven)
            model->B3SOIDDwketa = 0.0;
        if (!model->B3SOIDDwabpGiven)
            model->B3SOIDDwabp = 0.0;
        if (!model->B3SOIDDwmxcGiven)
            model->B3SOIDDwmxc = 0.0;
        if (!model->B3SOIDDwadice0Given)
            model->B3SOIDDwadice0 = 0.0;
        if (!model->B3SOIDDwa1Given)
            model->B3SOIDDwa1 = 0.0;
        if (!model->B3SOIDDwa2Given)
            model->B3SOIDDwa2 = 0.0;
        if (!model->B3SOIDDwrdswGiven)
            model->B3SOIDDwrdsw = 0.0;
        if (!model->B3SOIDDwprwbGiven)
            model->B3SOIDDwprwb = 0.0;
        if (!model->B3SOIDDwprwgGiven)
            model->B3SOIDDwprwg = 0.0;
        if (!model->B3SOIDDwwrGiven)
            model->B3SOIDDwwr = 0.0;
        if (!model->B3SOIDDwnfactorGiven)
            model->B3SOIDDwnfactor = 0.0;
        if (!model->B3SOIDDwdwgGiven)
            model->B3SOIDDwdwg = 0.0;
        if (!model->B3SOIDDwdwbGiven)
            model->B3SOIDDwdwb = 0.0;
        if (!model->B3SOIDDwvoffGiven)
            model->B3SOIDDwvoff = 0.0;
        if (!model->B3SOIDDweta0Given)
            model->B3SOIDDweta0 = 0.0;
        if (!model->B3SOIDDwetabGiven)
            model->B3SOIDDwetab = 0.0;
        if (!model->B3SOIDDwdsubGiven)
            model->B3SOIDDwdsub = 0.0;
        if (!model->B3SOIDDwcitGiven)
            model->B3SOIDDwcit = 0.0;
        if (!model->B3SOIDDwcdscGiven)
            model->B3SOIDDwcdsc = 0.0;
        if (!model->B3SOIDDwcdscbGiven)
            model->B3SOIDDwcdscb = 0.0;
        if (!model->B3SOIDDwcdscdGiven)
            model->B3SOIDDwcdscd = 0.0;
        if (!model->B3SOIDDwpclmGiven)
            model->B3SOIDDwpclm = 0.0;
        if (!model->B3SOIDDwpdibl1Given)
            model->B3SOIDDwpdibl1 = 0.0;
        if (!model->B3SOIDDwpdibl2Given)
            model->B3SOIDDwpdibl2 = 0.0;
        if (!model->B3SOIDDwpdiblbGiven)
            model->B3SOIDDwpdiblb = 0.0;
        if (!model->B3SOIDDwdroutGiven)
            model->B3SOIDDwdrout = 0.0;
        if (!model->B3SOIDDwpvagGiven)
            model->B3SOIDDwpvag = 0.0;
        if (!model->B3SOIDDwdeltaGiven)
            model->B3SOIDDwdelta = 0.0;
        if (!model->B3SOIDDwaiiGiven)
            model->B3SOIDDwaii = 0.0;
        if (!model->B3SOIDDwbiiGiven)
            model->B3SOIDDwbii = 0.0;
        if (!model->B3SOIDDwciiGiven)
            model->B3SOIDDwcii = 0.0;
        if (!model->B3SOIDDwdiiGiven)
            model->B3SOIDDwdii = 0.0;
        if (!model->B3SOIDDwalpha0Given)
            model->B3SOIDDwalpha0 = 0.0;
        if (!model->B3SOIDDwalpha1Given)
            model->B3SOIDDwalpha1 = 0.0;
        if (!model->B3SOIDDwbeta0Given)
            model->B3SOIDDwbeta0 = 0.0;
        if (!model->B3SOIDDwagidlGiven)
            model->B3SOIDDwagidl = 0.0;
        if (!model->B3SOIDDwbgidlGiven)
            model->B3SOIDDwbgidl = 0.0;
        if (!model->B3SOIDDwngidlGiven)
            model->B3SOIDDwngidl = 0.0;
        if (!model->B3SOIDDwntunGiven)
            model->B3SOIDDwntun = 0.0;
        if (!model->B3SOIDDwndiodeGiven)
            model->B3SOIDDwndiode = 0.0;
        if (!model->B3SOIDDwisbjtGiven)
            model->B3SOIDDwisbjt = 0.0;
        if (!model->B3SOIDDwisdifGiven)
            model->B3SOIDDwisdif = 0.0;
        if (!model->B3SOIDDwisrecGiven)
            model->B3SOIDDwisrec = 0.0;
        if (!model->B3SOIDDwistunGiven)
            model->B3SOIDDwistun = 0.0;
        if (!model->B3SOIDDwedlGiven)
            model->B3SOIDDwedl = 0.0;
        if (!model->B3SOIDDwkbjt1Given)
            model->B3SOIDDwkbjt1 = 0.0;
	/* CV Model */
        if (!model->B3SOIDDwvsdfbGiven)
            model->B3SOIDDwvsdfb = 0.0;
        if (!model->B3SOIDDwvsdthGiven)
            model->B3SOIDDwvsdth = 0.0;
        /* Cross-term dependence */
        if (!model->B3SOIDDpnpeakGiven)
            model->B3SOIDDpnpeak = 0.0;
        if (!model->B3SOIDDpnsubGiven)
            model->B3SOIDDpnsub = 0.0;
        if (!model->B3SOIDDpngateGiven)
            model->B3SOIDDpngate = 0.0;
        if (!model->B3SOIDDpvth0Given)
           model->B3SOIDDpvth0 = 0.0;
        if (!model->B3SOIDDpk1Given)
            model->B3SOIDDpk1 = 0.0;
        if (!model->B3SOIDDpk2Given)
            model->B3SOIDDpk2 = 0.0;
        if (!model->B3SOIDDpk3Given)
            model->B3SOIDDpk3 = 0.0;
        if (!model->B3SOIDDpk3bGiven)
            model->B3SOIDDpk3b = 0.0;
        if (!model->B3SOIDDpvbsaGiven)
            model->B3SOIDDpvbsa = 0.0;
        if (!model->B3SOIDDpdelpGiven)
            model->B3SOIDDpdelp = 0.0;
        if (!model->B3SOIDDpkb1Given)
           model->B3SOIDDpkb1 = 0.0;
        if (!model->B3SOIDDpkb3Given)
           model->B3SOIDDpkb3 = 1.0;
        if (!model->B3SOIDDpdvbd0Given)
           model->B3SOIDDpdvbd0 = 1.0;
        if (!model->B3SOIDDpdvbd1Given)
           model->B3SOIDDpdvbd1 = 1.0;
        if (!model->B3SOIDDpw0Given)
            model->B3SOIDDpw0 = 0.0;
        if (!model->B3SOIDDpnlxGiven)
            model->B3SOIDDpnlx = 0.0;
        if (!model->B3SOIDDpdvt0Given)
            model->B3SOIDDpdvt0 = 0.0;
        if (!model->B3SOIDDpdvt1Given)
            model->B3SOIDDpdvt1 = 0.0;
        if (!model->B3SOIDDpdvt2Given)
            model->B3SOIDDpdvt2 = 0.0;
        if (!model->B3SOIDDpdvt0wGiven)
            model->B3SOIDDpdvt0w = 0.0;
        if (!model->B3SOIDDpdvt1wGiven)
            model->B3SOIDDpdvt1w = 0.0;
        if (!model->B3SOIDDpdvt2wGiven)
            model->B3SOIDDpdvt2w = 0.0;
        if (!model->B3SOIDDpu0Given)
            model->B3SOIDDpu0 = 0.0;
        if (!model->B3SOIDDpuaGiven)
            model->B3SOIDDpua = 0.0;
        if (!model->B3SOIDDpubGiven)
            model->B3SOIDDpub = 0.0;
        if (!model->B3SOIDDpucGiven)
            model->B3SOIDDpuc = 0.0;
        if (!model->B3SOIDDpvsatGiven)
            model->B3SOIDDpvsat = 0.0;
        if (!model->B3SOIDDpa0Given)
            model->B3SOIDDpa0 = 0.0;
        if (!model->B3SOIDDpagsGiven)
            model->B3SOIDDpags = 0.0;
        if (!model->B3SOIDDpb0Given)
            model->B3SOIDDpb0 = 0.0;
        if (!model->B3SOIDDpb1Given)
            model->B3SOIDDpb1 = 0.0;
        if (!model->B3SOIDDpketaGiven)
            model->B3SOIDDpketa = 0.0;
        if (!model->B3SOIDDpabpGiven)
            model->B3SOIDDpabp = 0.0;
        if (!model->B3SOIDDpmxcGiven)
            model->B3SOIDDpmxc = 0.0;
        if (!model->B3SOIDDpadice0Given)
            model->B3SOIDDpadice0 = 0.0;
        if (!model->B3SOIDDpa1Given)
            model->B3SOIDDpa1 = 0.0;
        if (!model->B3SOIDDpa2Given)
            model->B3SOIDDpa2 = 0.0;
        if (!model->B3SOIDDprdswGiven)
            model->B3SOIDDprdsw = 0.0;
        if (!model->B3SOIDDpprwbGiven)
            model->B3SOIDDpprwb = 0.0;
        if (!model->B3SOIDDpprwgGiven)
            model->B3SOIDDpprwg = 0.0;
        if (!model->B3SOIDDpwrGiven)
            model->B3SOIDDpwr = 0.0;
        if (!model->B3SOIDDpnfactorGiven)
            model->B3SOIDDpnfactor = 0.0;
        if (!model->B3SOIDDpdwgGiven)
            model->B3SOIDDpdwg = 0.0;
        if (!model->B3SOIDDpdwbGiven)
            model->B3SOIDDpdwb = 0.0;
        if (!model->B3SOIDDpvoffGiven)
            model->B3SOIDDpvoff = 0.0;
        if (!model->B3SOIDDpeta0Given)
            model->B3SOIDDpeta0 = 0.0;
        if (!model->B3SOIDDpetabGiven)
            model->B3SOIDDpetab = 0.0;
        if (!model->B3SOIDDpdsubGiven)
            model->B3SOIDDpdsub = 0.0;
        if (!model->B3SOIDDpcitGiven)
            model->B3SOIDDpcit = 0.0;
        if (!model->B3SOIDDpcdscGiven)
            model->B3SOIDDpcdsc = 0.0;
        if (!model->B3SOIDDpcdscbGiven)
            model->B3SOIDDpcdscb = 0.0;
        if (!model->B3SOIDDpcdscdGiven)
            model->B3SOIDDpcdscd = 0.0;
        if (!model->B3SOIDDppclmGiven)
            model->B3SOIDDppclm = 0.0;
        if (!model->B3SOIDDppdibl1Given)
            model->B3SOIDDppdibl1 = 0.0;
        if (!model->B3SOIDDppdibl2Given)
            model->B3SOIDDppdibl2 = 0.0;
        if (!model->B3SOIDDppdiblbGiven)
            model->B3SOIDDppdiblb = 0.0;
        if (!model->B3SOIDDpdroutGiven)
            model->B3SOIDDpdrout = 0.0;
        if (!model->B3SOIDDppvagGiven)
            model->B3SOIDDppvag = 0.0;
        if (!model->B3SOIDDpdeltaGiven)
            model->B3SOIDDpdelta = 0.0;
        if (!model->B3SOIDDpaiiGiven)
            model->B3SOIDDpaii = 0.0;
        if (!model->B3SOIDDpbiiGiven)
            model->B3SOIDDpbii = 0.0;
        if (!model->B3SOIDDpciiGiven)
            model->B3SOIDDpcii = 0.0;
        if (!model->B3SOIDDpdiiGiven)
            model->B3SOIDDpdii = 0.0;
        if (!model->B3SOIDDpalpha0Given)
            model->B3SOIDDpalpha0 = 0.0;
        if (!model->B3SOIDDpalpha1Given)
            model->B3SOIDDpalpha1 = 0.0;
        if (!model->B3SOIDDpbeta0Given)
            model->B3SOIDDpbeta0 = 0.0;
        if (!model->B3SOIDDpagidlGiven)
            model->B3SOIDDpagidl = 0.0;
        if (!model->B3SOIDDpbgidlGiven)
            model->B3SOIDDpbgidl = 0.0;
        if (!model->B3SOIDDpngidlGiven)
            model->B3SOIDDpngidl = 0.0;
        if (!model->B3SOIDDpntunGiven)
            model->B3SOIDDpntun = 0.0;
        if (!model->B3SOIDDpndiodeGiven)
            model->B3SOIDDpndiode = 0.0;
        if (!model->B3SOIDDpisbjtGiven)
            model->B3SOIDDpisbjt = 0.0;
        if (!model->B3SOIDDpisdifGiven)
            model->B3SOIDDpisdif = 0.0;
        if (!model->B3SOIDDpisrecGiven)
            model->B3SOIDDpisrec = 0.0;
        if (!model->B3SOIDDpistunGiven)
            model->B3SOIDDpistun = 0.0;
        if (!model->B3SOIDDpedlGiven)
            model->B3SOIDDpedl = 0.0;
        if (!model->B3SOIDDpkbjt1Given)
            model->B3SOIDDpkbjt1 = 0.0;
	/* CV Model */
        if (!model->B3SOIDDpvsdfbGiven)
            model->B3SOIDDpvsdfb = 0.0;
        if (!model->B3SOIDDpvsdthGiven)
            model->B3SOIDDpvsdth = 0.0;
/* Added for binning - END */

	if (!model->B3SOIDDcfGiven)
            model->B3SOIDDcf = 2.0 * EPSOX / PI
			   * log(1.0 + 0.4e-6 / model->B3SOIDDtox);
        if (!model->B3SOIDDcgdoGiven)
	{   if (model->B3SOIDDdlcGiven && (model->B3SOIDDdlc > 0.0))
	    {   model->B3SOIDDcgdo = model->B3SOIDDdlc * model->B3SOIDDcox
				 - model->B3SOIDDcgdl ;
	    }
	    else
	        model->B3SOIDDcgdo = 0.6 * model->B3SOIDDxj * model->B3SOIDDcox; 
	}
        if (!model->B3SOIDDcgsoGiven)
	{   if (model->B3SOIDDdlcGiven && (model->B3SOIDDdlc > 0.0))
	    {   model->B3SOIDDcgso = model->B3SOIDDdlc * model->B3SOIDDcox
				 - model->B3SOIDDcgsl ;
	    }
	    else
	        model->B3SOIDDcgso = 0.6 * model->B3SOIDDxj * model->B3SOIDDcox; 
	}

        if (!model->B3SOIDDcgeoGiven)
	{   model->B3SOIDDcgeo = 0.0;
	}
        if (!model->B3SOIDDxpartGiven)
            model->B3SOIDDxpart = 0.0;
        if (!model->B3SOIDDsheetResistanceGiven)
            model->B3SOIDDsheetResistance = 0.0;
        if (!model->B3SOIDDcsdeswGiven)
            model->B3SOIDDcsdesw = 0.0;
        if (!model->B3SOIDDunitLengthGateSidewallJctCapGiven)
            model->B3SOIDDunitLengthGateSidewallJctCap = 1e-10;
        if (!model->B3SOIDDGatesidewallJctPotentialGiven)
            model->B3SOIDDGatesidewallJctPotential = 0.7;
        if (!model->B3SOIDDbodyJctGateSideGradingCoeffGiven)
            model->B3SOIDDbodyJctGateSideGradingCoeff = 0.5;
        if (!model->B3SOIDDoxideTrapDensityAGiven)
	{   if (model->B3SOIDDtype == NMOS)
                model->B3SOIDDoxideTrapDensityA = 1e20;
            else
                model->B3SOIDDoxideTrapDensityA=9.9e18;
	}
        if (!model->B3SOIDDoxideTrapDensityBGiven)
	{   if (model->B3SOIDDtype == NMOS)
                model->B3SOIDDoxideTrapDensityB = 5e4;
            else
                model->B3SOIDDoxideTrapDensityB = 2.4e3;
	}
        if (!model->B3SOIDDoxideTrapDensityCGiven)
	{   if (model->B3SOIDDtype == NMOS)
                model->B3SOIDDoxideTrapDensityC = -1.4e-12;
            else
                model->B3SOIDDoxideTrapDensityC = 1.4e-12;

	}
        if (!model->B3SOIDDemGiven)
            model->B3SOIDDem = 4.1e7; /* V/m */
        if (!model->B3SOIDDefGiven)
            model->B3SOIDDef = 1.0;
        if (!model->B3SOIDDafGiven)
            model->B3SOIDDaf = 1.0;
        if (!model->B3SOIDDkfGiven)
            model->B3SOIDDkf = 0.0;
        if (!model->B3SOIDDnoifGiven)
            model->B3SOIDDnoif = 1.0;

        /* loop through all the instances of the model */
        for (here = B3SOIDDinstances(model); here != NULL ;
             here=B3SOIDDnextInstance(here)) 
	{
            /* allocate a chunk of the state vector */
            here->B3SOIDDstates = *states;
            *states += B3SOIDDnumStates;
	    
	    /* perform the parameter defaulting */
            if (!here->B3SOIDDdrainAreaGiven)
                here->B3SOIDDdrainArea = 0.0;
            if (!here->B3SOIDDdrainPerimeterGiven)
                here->B3SOIDDdrainPerimeter = 0.0;
            if (!here->B3SOIDDdrainSquaresGiven)
                here->B3SOIDDdrainSquares = 1.0;
            if (!here->B3SOIDDicVBSGiven)
                here->B3SOIDDicVBS = 0;
            if (!here->B3SOIDDicVDSGiven)
                here->B3SOIDDicVDS = 0;
            if (!here->B3SOIDDicVGSGiven)
                here->B3SOIDDicVGS = 0;
            if (!here->B3SOIDDicVESGiven)
                here->B3SOIDDicVES = 0;
            if (!here->B3SOIDDicVPSGiven)
                here->B3SOIDDicVPS = 0;
	    if (!here->B3SOIDDbjtoffGiven)
		here->B3SOIDDbjtoff = 0;
	    if (!here->B3SOIDDdebugModGiven)
		here->B3SOIDDdebugMod = 0;
	    if (!here->B3SOIDDrth0Given)
		here->B3SOIDDrth0 = model->B3SOIDDrth0;
	    if (!here->B3SOIDDcth0Given)
		here->B3SOIDDcth0 = model->B3SOIDDcth0;
            if (!here->B3SOIDDbodySquaresGiven)
                here->B3SOIDDbodySquares = 1.0;
            if (!here->B3SOIDDlGiven)
                here->B3SOIDDl = 5e-6;
            if (!here->B3SOIDDsourceAreaGiven)
                here->B3SOIDDsourceArea = 0;
            if (!here->B3SOIDDsourcePerimeterGiven)
                here->B3SOIDDsourcePerimeter = 0;
            if (!here->B3SOIDDsourceSquaresGiven)
                here->B3SOIDDsourceSquares = 1;
            if (!here->B3SOIDDwGiven)
                here->B3SOIDDw = 5e-6;
            
	    if (!here->B3SOIDDmGiven)
                here->B3SOIDDm = 1;
	    
	    if (!here->B3SOIDDoffGiven)
                here->B3SOIDDoff = 0;

            /* process drain series resistance */
            if ((model->B3SOIDDsheetResistance > 0.0) && 
                (here->B3SOIDDdrainSquares > 0.0 ) &&
                (here->B3SOIDDdNodePrime == 0))
	    {   error = CKTmkVolt(ckt,&tmp,here->B3SOIDDname,"drain");
                if(error) return(error);
                here->B3SOIDDdNodePrime = tmp->number;
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
	    {   here->B3SOIDDdNodePrime = here->B3SOIDDdNode;
            }
                   
            /* process source series resistance */
            if ((model->B3SOIDDsheetResistance > 0.0) && 
                (here->B3SOIDDsourceSquares > 0.0 ) &&
                (here->B3SOIDDsNodePrime == 0)) 
	    {   error = CKTmkVolt(ckt,&tmp,here->B3SOIDDname,"source");
                if(error) return(error);
                here->B3SOIDDsNodePrime = tmp->number;
		
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
	    {   here->B3SOIDDsNodePrime = here->B3SOIDDsNode;
            }

            /* process effective silicon film thickness */
            model->B3SOIDDcbox = 3.453133e-11 / model->B3SOIDDtbox;
            model->B3SOIDDcsi = 1.03594e-10 / model->B3SOIDDtsi;
            Cboxt = model->B3SOIDDcbox * model->B3SOIDDcsi / (model->B3SOIDDcbox + model->B3SOIDDcsi);
            model->B3SOIDDqsi = Charge_q*model->B3SOIDDnpeak*1e6*model->B3SOIDDtsi;
            /* Tsieff */
            tmp1 = 2.0 * EPSSI * model->B3SOIDDvbsa / Charge_q 
                 / (1e6*model->B3SOIDDnpeak);
            tmp2 = model->B3SOIDDtsi * model->B3SOIDDtsi;
            if (tmp2 < tmp1)
            {
               fprintf(stderr, "vbsa = %.3f is too large for this tsi = %.3e and is automatically set to zero\n", model->B3SOIDDvbsa, model->B3SOIDDtsi);
               model->B3SOIDDcsieff = model->B3SOIDDcsi;
               model->B3SOIDDqsieff = model->B3SOIDDqsi;
            }
            else
            {
               tmp1 = sqrt(model->B3SOIDDtsi * model->B3SOIDDtsi - 
                      2.0 * EPSSI * model->B3SOIDDvbsa / Charge_q / 
                      (1e6*model->B3SOIDDnpeak));
               model->B3SOIDDcsieff = 1.03594e-10 / tmp1;
               model->B3SOIDDqsieff = Charge_q*model->B3SOIDDnpeak*1e6*tmp1;
            }
            model->B3SOIDDcsit = 1/(1/model->B3SOIDDcox + 1/model->B3SOIDDcsieff);
            model->B3SOIDDcboxt = 1/(1/model->B3SOIDDcbox + 1/model->B3SOIDDcsieff);
            nfb0 = 1/(1 + model->B3SOIDDcbox / model->B3SOIDDcsit);
            model->B3SOIDDnfb = model->B3SOIDDkb3 * nfb0;
            model->B3SOIDDadice = model->B3SOIDDadice0  / ( 1 + Cboxt / model->B3SOIDDcox);

            here->B3SOIDDfloat = 0;

            here->B3SOIDDpNode = here->B3SOIDDpNodeExt;
            here->B3SOIDDbNode = here->B3SOIDDbNodeExt;
            here->B3SOIDDtempNode = here->B3SOIDDtempNodeExt;

	    if (here->B3SOIDDbNode == -1) 
            /* no body contact but bNode to be created for SPICE iteration */
            {  error = CKTmkVolt(ckt,&tmp,here->B3SOIDDname,"Body");
               if(error) return(error);
               here->B3SOIDDbNode = tmp->number;
               here->B3SOIDDpNode = 0;
               here->B3SOIDDfloat = 1;
               here->B3SOIDDbodyMod = 0;
	    }
            else /* if body tied */ 
            { /* ideal body tie */
               if ((model->B3SOIDDrbody == 0.0) && (model->B3SOIDDrbsh == 0.0))
               {
                  here->B3SOIDDbodyMod = 2;
                  /* pNode is not used in this case */
               }
               else { 
                  error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Body");
                  if(error) return(error);
                  here->B3SOIDDbodyMod = 1;
                  here->B3SOIDDpNode = here->B3SOIDDbNode;
                  here->B3SOIDDbNode = tmp->number;
               }
            }

            if ((model->B3SOIDDshMod == 1) && (here->B3SOIDDrth0!=0))
            {
	        error = CKTmkVolt(ckt,&tmp,here->B3SOIDDname,"Temp");
                if(error) return(error);
                   here->B3SOIDDtempNode = tmp->number;

            } else {
                here->B3SOIDDtempNode = 0;
            }

/* here for debugging purpose only */
            if ((here->B3SOIDDdebugMod > 1) || (here->B3SOIDDdebugMod == -1))
            {
               /* The real Vbs value */
               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Vbs");
               if(error) return(error);
               here->B3SOIDDvbsNode = tmp->number;   

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Ids");
               if(error) return(error);
               here->B3SOIDDidsNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Ic");
               if(error) return(error);
               here->B3SOIDDicNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Ibs");
               if(error) return(error);
               here->B3SOIDDibsNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Ibd");
               if(error) return(error);
               here->B3SOIDDibdNode = tmp->number;
       
               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Iii");
               if(error) return(error);
               here->B3SOIDDiiiNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Igidl");
               if(error) return(error);
               here->B3SOIDDigidlNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Itun");
               if(error) return(error);
               here->B3SOIDDitunNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Ibp");
               if(error) return(error);
               here->B3SOIDDibpNode = tmp->number;
       
               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Abeff");
               if(error) return(error);
               here->B3SOIDDabeffNode = tmp->number;
       
               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Vbs0eff");
               if(error) return(error);
               here->B3SOIDDvbs0effNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Vbseff");
               if(error) return(error);
               here->B3SOIDDvbseffNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Xc");
               if(error) return(error);
               here->B3SOIDDxcNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Cbb");
               if(error) return(error);
               here->B3SOIDDcbbNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Cbd");
               if(error) return(error);
               here->B3SOIDDcbdNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Cbg");
               if(error) return(error);
               here->B3SOIDDcbgNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Qbody");
               if(error) return(error);
               here->B3SOIDDqbNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Qbf");
               if(error) return(error);
               here->B3SOIDDqbfNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Qjs");
               if(error) return(error);
               here->B3SOIDDqjsNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Qjd");
               if(error) return(error);
               here->B3SOIDDqjdNode = tmp->number;

               /* clean up last */
               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Gm");
               if(error) return(error);
               here->B3SOIDDgmNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Gmbs");
               if(error) return(error);
               here->B3SOIDDgmbsNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Gds");
               if(error) return(error);
               here->B3SOIDDgdsNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Gme");
               if(error) return(error);
               here->B3SOIDDgmeNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Vbs0teff");
               if(error) return(error);
               here->B3SOIDDvbs0teffNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Vth");
               if(error) return(error);
               here->B3SOIDDvthNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Vgsteff");
               if(error) return(error);
               here->B3SOIDDvgsteffNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Xcsat");
               if(error) return(error);
               here->B3SOIDDxcsatNode = tmp->number;
       
               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Qac0");
               if(error) return(error);
               here->B3SOIDDqaccNode = tmp->number;
       
               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Qsub0");
               if(error) return(error);
               here->B3SOIDDqsub0Node = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Qsubs1");
               if(error) return(error);
               here->B3SOIDDqsubs1Node = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Qsubs2");
               if(error) return(error);
               here->B3SOIDDqsubs2Node = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Qsub");
               if(error) return(error);
               here->B3SOIDDqeNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Qdrn");
               if(error) return(error);
               here->B3SOIDDqdNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Qgate");
               if(error) return(error);
               here->B3SOIDDqgNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Vdscv");
               if(error) return(error);
               here->B3SOIDDvdscvNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Vcscv");
               if(error) return(error);
               here->B3SOIDDvcscvNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Cbe");
               if(error) return(error);
               here->B3SOIDDcbeNode = tmp->number;
       
               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Dum1");
               if(error) return(error);
               here->B3SOIDDdum1Node = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Dum2");
               if(error) return(error);
               here->B3SOIDDdum2Node = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Dum3");
               if(error) return(error);
               here->B3SOIDDdum3Node = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Dum4");
               if(error) return(error);
               here->B3SOIDDdum4Node = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIDDname, "Dum5");
               if(error) return(error);
               here->B3SOIDDdum5Node = tmp->number;
           }

        /* set Sparse Matrix Pointers */

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)


        if ((model->B3SOIDDshMod == 1) && (here->B3SOIDDrth0!=0.0)) {
            TSTALLOC(B3SOIDDTemptempPtr, B3SOIDDtempNode, B3SOIDDtempNode);
            TSTALLOC(B3SOIDDTempdpPtr, B3SOIDDtempNode, B3SOIDDdNodePrime);
            TSTALLOC(B3SOIDDTempspPtr, B3SOIDDtempNode, B3SOIDDsNodePrime);
            TSTALLOC(B3SOIDDTempgPtr, B3SOIDDtempNode, B3SOIDDgNode);
            TSTALLOC(B3SOIDDTempbPtr, B3SOIDDtempNode, B3SOIDDbNode);
            TSTALLOC(B3SOIDDTempePtr, B3SOIDDtempNode, B3SOIDDeNode);

            TSTALLOC(B3SOIDDGtempPtr, B3SOIDDgNode, B3SOIDDtempNode);
            TSTALLOC(B3SOIDDDPtempPtr, B3SOIDDdNodePrime, B3SOIDDtempNode);
            TSTALLOC(B3SOIDDSPtempPtr, B3SOIDDsNodePrime, B3SOIDDtempNode);
            TSTALLOC(B3SOIDDEtempPtr, B3SOIDDeNode, B3SOIDDtempNode);
            TSTALLOC(B3SOIDDBtempPtr, B3SOIDDbNode, B3SOIDDtempNode);

            if (here->B3SOIDDbodyMod == 1) {
                TSTALLOC(B3SOIDDPtempPtr, B3SOIDDpNode, B3SOIDDtempNode);
            }
        }
        if (here->B3SOIDDbodyMod == 2) {
            /* Don't create any Jacobian entry for pNode */
        }
        else if (here->B3SOIDDbodyMod == 1) {
            TSTALLOC(B3SOIDDBpPtr, B3SOIDDbNode, B3SOIDDpNode);
            TSTALLOC(B3SOIDDPbPtr, B3SOIDDpNode, B3SOIDDbNode);
            TSTALLOC(B3SOIDDPpPtr, B3SOIDDpNode, B3SOIDDpNode);
            TSTALLOC(B3SOIDDPgPtr, B3SOIDDpNode, B3SOIDDgNode);
            TSTALLOC(B3SOIDDPdpPtr, B3SOIDDpNode, B3SOIDDdNodePrime);
            TSTALLOC(B3SOIDDPspPtr, B3SOIDDpNode, B3SOIDDsNodePrime);
            TSTALLOC(B3SOIDDPePtr, B3SOIDDpNode, B3SOIDDeNode);
        }
        
            TSTALLOC(B3SOIDDEgPtr, B3SOIDDeNode, B3SOIDDgNode);
            TSTALLOC(B3SOIDDEdpPtr, B3SOIDDeNode, B3SOIDDdNodePrime);
            TSTALLOC(B3SOIDDEspPtr, B3SOIDDeNode, B3SOIDDsNodePrime);
            TSTALLOC(B3SOIDDGePtr, B3SOIDDgNode, B3SOIDDeNode);
            TSTALLOC(B3SOIDDDPePtr, B3SOIDDdNodePrime, B3SOIDDeNode);
            TSTALLOC(B3SOIDDSPePtr, B3SOIDDsNodePrime, B3SOIDDeNode);
        
            TSTALLOC(B3SOIDDEbPtr, B3SOIDDeNode, B3SOIDDbNode);
            TSTALLOC(B3SOIDDGbPtr, B3SOIDDgNode, B3SOIDDbNode);
            TSTALLOC(B3SOIDDDPbPtr, B3SOIDDdNodePrime, B3SOIDDbNode);
            TSTALLOC(B3SOIDDSPbPtr, B3SOIDDsNodePrime, B3SOIDDbNode);
            TSTALLOC(B3SOIDDBePtr, B3SOIDDbNode, B3SOIDDeNode);
            TSTALLOC(B3SOIDDBgPtr, B3SOIDDbNode, B3SOIDDgNode);
            TSTALLOC(B3SOIDDBdpPtr, B3SOIDDbNode, B3SOIDDdNodePrime);
            TSTALLOC(B3SOIDDBspPtr, B3SOIDDbNode, B3SOIDDsNodePrime);
            TSTALLOC(B3SOIDDBbPtr, B3SOIDDbNode, B3SOIDDbNode);

            TSTALLOC(B3SOIDDEePtr, B3SOIDDeNode, B3SOIDDeNode);

            TSTALLOC(B3SOIDDGgPtr, B3SOIDDgNode, B3SOIDDgNode);
            TSTALLOC(B3SOIDDGdpPtr, B3SOIDDgNode, B3SOIDDdNodePrime);
            TSTALLOC(B3SOIDDGspPtr, B3SOIDDgNode, B3SOIDDsNodePrime);

            TSTALLOC(B3SOIDDDPgPtr, B3SOIDDdNodePrime, B3SOIDDgNode);
            TSTALLOC(B3SOIDDDPdpPtr, B3SOIDDdNodePrime, B3SOIDDdNodePrime);
            TSTALLOC(B3SOIDDDPspPtr, B3SOIDDdNodePrime, B3SOIDDsNodePrime);
            TSTALLOC(B3SOIDDDPdPtr, B3SOIDDdNodePrime, B3SOIDDdNode);

            TSTALLOC(B3SOIDDSPgPtr, B3SOIDDsNodePrime, B3SOIDDgNode);
            TSTALLOC(B3SOIDDSPdpPtr, B3SOIDDsNodePrime, B3SOIDDdNodePrime);
            TSTALLOC(B3SOIDDSPspPtr, B3SOIDDsNodePrime, B3SOIDDsNodePrime);
            TSTALLOC(B3SOIDDSPsPtr, B3SOIDDsNodePrime, B3SOIDDsNode);

            TSTALLOC(B3SOIDDDdPtr, B3SOIDDdNode, B3SOIDDdNode);
            TSTALLOC(B3SOIDDDdpPtr, B3SOIDDdNode, B3SOIDDdNodePrime);

            TSTALLOC(B3SOIDDSsPtr, B3SOIDDsNode, B3SOIDDsNode);
            TSTALLOC(B3SOIDDSspPtr, B3SOIDDsNode, B3SOIDDsNodePrime);

/* here for debugging purpose only */
         if ((here->B3SOIDDdebugMod > 1) || (here->B3SOIDDdebugMod == -1))
         {
            TSTALLOC(B3SOIDDVbsPtr, B3SOIDDvbsNode, B3SOIDDvbsNode) ;
            TSTALLOC(B3SOIDDIdsPtr, B3SOIDDidsNode, B3SOIDDidsNode);
            TSTALLOC(B3SOIDDIcPtr, B3SOIDDicNode, B3SOIDDicNode);
            TSTALLOC(B3SOIDDIbsPtr, B3SOIDDibsNode, B3SOIDDibsNode);
            TSTALLOC(B3SOIDDIbdPtr, B3SOIDDibdNode, B3SOIDDibdNode);
            TSTALLOC(B3SOIDDIiiPtr, B3SOIDDiiiNode, B3SOIDDiiiNode);
            TSTALLOC(B3SOIDDIgidlPtr, B3SOIDDigidlNode, B3SOIDDigidlNode);
            TSTALLOC(B3SOIDDItunPtr, B3SOIDDitunNode, B3SOIDDitunNode);
            TSTALLOC(B3SOIDDIbpPtr, B3SOIDDibpNode, B3SOIDDibpNode);
            TSTALLOC(B3SOIDDAbeffPtr, B3SOIDDabeffNode, B3SOIDDabeffNode);
            TSTALLOC(B3SOIDDVbs0effPtr, B3SOIDDvbs0effNode, B3SOIDDvbs0effNode);
            TSTALLOC(B3SOIDDVbseffPtr, B3SOIDDvbseffNode, B3SOIDDvbseffNode);
            TSTALLOC(B3SOIDDXcPtr, B3SOIDDxcNode, B3SOIDDxcNode);
            TSTALLOC(B3SOIDDCbbPtr, B3SOIDDcbbNode, B3SOIDDcbbNode);
            TSTALLOC(B3SOIDDCbdPtr, B3SOIDDcbdNode, B3SOIDDcbdNode);
            TSTALLOC(B3SOIDDCbgPtr, B3SOIDDcbgNode, B3SOIDDcbgNode);
            TSTALLOC(B3SOIDDqbPtr, B3SOIDDqbNode, B3SOIDDqbNode);
            TSTALLOC(B3SOIDDQbfPtr, B3SOIDDqbfNode, B3SOIDDqbfNode);
            TSTALLOC(B3SOIDDQjsPtr, B3SOIDDqjsNode, B3SOIDDqjsNode);
            TSTALLOC(B3SOIDDQjdPtr, B3SOIDDqjdNode, B3SOIDDqjdNode);

            /* clean up last */
            TSTALLOC(B3SOIDDGmPtr, B3SOIDDgmNode, B3SOIDDgmNode);
            TSTALLOC(B3SOIDDGmbsPtr, B3SOIDDgmbsNode, B3SOIDDgmbsNode);
            TSTALLOC(B3SOIDDGdsPtr, B3SOIDDgdsNode, B3SOIDDgdsNode);
            TSTALLOC(B3SOIDDGmePtr, B3SOIDDgmeNode, B3SOIDDgmeNode);
            TSTALLOC(B3SOIDDVbs0teffPtr, B3SOIDDvbs0teffNode, B3SOIDDvbs0teffNode);
            TSTALLOC(B3SOIDDVthPtr, B3SOIDDvthNode, B3SOIDDvthNode);
            TSTALLOC(B3SOIDDVgsteffPtr, B3SOIDDvgsteffNode, B3SOIDDvgsteffNode);
            TSTALLOC(B3SOIDDXcsatPtr, B3SOIDDxcsatNode, B3SOIDDxcsatNode);
            TSTALLOC(B3SOIDDVcscvPtr, B3SOIDDvcscvNode, B3SOIDDvcscvNode);
            TSTALLOC(B3SOIDDVdscvPtr, B3SOIDDvdscvNode, B3SOIDDvdscvNode);
            TSTALLOC(B3SOIDDCbePtr, B3SOIDDcbeNode, B3SOIDDcbeNode);
            TSTALLOC(B3SOIDDDum1Ptr, B3SOIDDdum1Node, B3SOIDDdum1Node);
            TSTALLOC(B3SOIDDDum2Ptr, B3SOIDDdum2Node, B3SOIDDdum2Node);
            TSTALLOC(B3SOIDDDum3Ptr, B3SOIDDdum3Node, B3SOIDDdum3Node);
            TSTALLOC(B3SOIDDDum4Ptr, B3SOIDDdum4Node, B3SOIDDdum4Node);
            TSTALLOC(B3SOIDDDum5Ptr, B3SOIDDdum5Node, B3SOIDDdum5Node);
            TSTALLOC(B3SOIDDQaccPtr, B3SOIDDqaccNode, B3SOIDDqaccNode);
            TSTALLOC(B3SOIDDQsub0Ptr, B3SOIDDqsub0Node, B3SOIDDqsub0Node);
            TSTALLOC(B3SOIDDQsubs1Ptr, B3SOIDDqsubs1Node, B3SOIDDqsubs1Node);
            TSTALLOC(B3SOIDDQsubs2Ptr, B3SOIDDqsubs2Node, B3SOIDDqsubs2Node);
            TSTALLOC(B3SOIDDqePtr, B3SOIDDqeNode, B3SOIDDqeNode);
            TSTALLOC(B3SOIDDqdPtr, B3SOIDDqdNode, B3SOIDDqdNode);
            TSTALLOC(B3SOIDDqgPtr, B3SOIDDqgNode, B3SOIDDqgNode);
         }

        }
    }
    return(OK);
}  

int
B3SOIDDunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    B3SOIDDmodel *model;
    B3SOIDDinstance *here;
 
    for (model = (B3SOIDDmodel *)inModel; model != NULL;
            model = B3SOIDDnextModel(model))
    {
        for (here = B3SOIDDinstances(model); here != NULL;
                here=B3SOIDDnextInstance(here))
        {
            /* here for debugging purpose only */
            if (here->B3SOIDDdum5Node > 0)
                CKTdltNNum(ckt, here->B3SOIDDdum5Node);
            here->B3SOIDDdum5Node = 0;

            if (here->B3SOIDDdum4Node > 0)
                CKTdltNNum(ckt, here->B3SOIDDdum4Node);
            here->B3SOIDDdum4Node = 0;

            if (here->B3SOIDDdum3Node > 0)
                CKTdltNNum(ckt, here->B3SOIDDdum3Node);
            here->B3SOIDDdum3Node = 0;

            if (here->B3SOIDDdum2Node > 0)
                CKTdltNNum(ckt, here->B3SOIDDdum2Node);
            here->B3SOIDDdum2Node = 0;

            if (here->B3SOIDDdum1Node > 0)
                CKTdltNNum(ckt, here->B3SOIDDdum1Node);
            here->B3SOIDDdum1Node = 0;

            if (here->B3SOIDDcbeNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDcbeNode);
            here->B3SOIDDcbeNode = 0;

            if (here->B3SOIDDvcscvNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDvcscvNode);
            here->B3SOIDDvcscvNode = 0;

            if (here->B3SOIDDvdscvNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDvdscvNode);
            here->B3SOIDDvdscvNode = 0;

            if (here->B3SOIDDqgNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDqgNode);
            here->B3SOIDDqgNode = 0;

            if (here->B3SOIDDqdNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDqdNode);
            here->B3SOIDDqdNode = 0;

            if (here->B3SOIDDqeNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDqeNode);
            here->B3SOIDDqeNode = 0;

            if (here->B3SOIDDqsubs2Node > 0)
                CKTdltNNum(ckt, here->B3SOIDDqsubs2Node);
            here->B3SOIDDqsubs2Node = 0;

            if (here->B3SOIDDqsubs1Node > 0)
                CKTdltNNum(ckt, here->B3SOIDDqsubs1Node);
            here->B3SOIDDqsubs1Node = 0;

            if (here->B3SOIDDqsub0Node > 0)
                CKTdltNNum(ckt, here->B3SOIDDqsub0Node);
            here->B3SOIDDqsub0Node = 0;

            if (here->B3SOIDDqaccNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDqaccNode);
            here->B3SOIDDqaccNode = 0;

            if (here->B3SOIDDxcsatNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDxcsatNode);
            here->B3SOIDDxcsatNode = 0;

            if (here->B3SOIDDvgsteffNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDvgsteffNode);
            here->B3SOIDDvgsteffNode = 0;

            if (here->B3SOIDDvthNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDvthNode);
            here->B3SOIDDvthNode = 0;

            if (here->B3SOIDDvbs0teffNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDvbs0teffNode);
            here->B3SOIDDvbs0teffNode = 0;

            if (here->B3SOIDDgmeNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDgmeNode);
            here->B3SOIDDgmeNode = 0;

            if (here->B3SOIDDgdsNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDgdsNode);
            here->B3SOIDDgdsNode = 0;

            if (here->B3SOIDDgmbsNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDgmbsNode);
            here->B3SOIDDgmbsNode = 0;

            if (here->B3SOIDDgmNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDgmNode);
            here->B3SOIDDgmNode = 0;

            if (here->B3SOIDDqjdNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDqjdNode);
            here->B3SOIDDqjdNode = 0;

            if (here->B3SOIDDqjsNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDqjsNode);
            here->B3SOIDDqjsNode = 0;

            if (here->B3SOIDDqbfNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDqbfNode);
            here->B3SOIDDqbfNode = 0;

            if (here->B3SOIDDqbNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDqbNode);
            here->B3SOIDDqbNode = 0;

            if (here->B3SOIDDcbgNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDcbgNode);
            here->B3SOIDDcbgNode = 0;

            if (here->B3SOIDDcbdNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDcbdNode);
            here->B3SOIDDcbdNode = 0;

            if (here->B3SOIDDcbbNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDcbbNode);
            here->B3SOIDDcbbNode = 0;

            if (here->B3SOIDDxcNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDxcNode);
            here->B3SOIDDxcNode = 0;

            if (here->B3SOIDDvbseffNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDvbseffNode);
            here->B3SOIDDvbseffNode = 0;

            if (here->B3SOIDDvbs0effNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDvbs0effNode);
            here->B3SOIDDvbs0effNode = 0;

            if (here->B3SOIDDabeffNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDabeffNode);
            here->B3SOIDDabeffNode = 0;

            if (here->B3SOIDDibpNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDibpNode);
            here->B3SOIDDibpNode = 0;

            if (here->B3SOIDDitunNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDitunNode);
            here->B3SOIDDitunNode = 0;

            if (here->B3SOIDDigidlNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDigidlNode);
            here->B3SOIDDigidlNode = 0;

            if (here->B3SOIDDiiiNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDiiiNode);
            here->B3SOIDDiiiNode = 0;

            if (here->B3SOIDDibdNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDibdNode);
            here->B3SOIDDibdNode = 0;

            if (here->B3SOIDDibsNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDibsNode);
            here->B3SOIDDibsNode = 0;

            if (here->B3SOIDDicNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDicNode);
            here->B3SOIDDicNode = 0;

            if (here->B3SOIDDidsNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDidsNode);
            here->B3SOIDDidsNode = 0;

            if (here->B3SOIDDvbsNode > 0)
                CKTdltNNum(ckt, here->B3SOIDDvbsNode);
            here->B3SOIDDvbsNode = 0;


            if (here->B3SOIDDtempNode > 0 &&
                here->B3SOIDDtempNode != here->B3SOIDDtempNodeExt &&
                here->B3SOIDDtempNode != here->B3SOIDDbNodeExt &&
                here->B3SOIDDtempNode != here->B3SOIDDpNodeExt)
                CKTdltNNum(ckt, here->B3SOIDDtempNode);
            here->B3SOIDDtempNode = 0;

            if (here->B3SOIDDbNode > 0 &&
                here->B3SOIDDbNode != here->B3SOIDDbNodeExt &&
                here->B3SOIDDbNode != here->B3SOIDDpNodeExt)
                CKTdltNNum(ckt, here->B3SOIDDbNode);
            here->B3SOIDDbNode = 0;

            here->B3SOIDDpNode = 0;

            if (here->B3SOIDDsNodePrime > 0
                    && here->B3SOIDDsNodePrime != here->B3SOIDDsNode)
                CKTdltNNum(ckt, here->B3SOIDDsNodePrime);
            here->B3SOIDDsNodePrime = 0;

            if (here->B3SOIDDdNodePrime > 0
                    && here->B3SOIDDdNodePrime != here->B3SOIDDdNode)
                CKTdltNNum(ckt, here->B3SOIDDdNodePrime);
            here->B3SOIDDdNodePrime = 0;
        }
    }
    return OK;
}

