/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: Weidong Liu and Pin Su         Feb 1999
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soifdset.c          98/5/01
Modified by Pin Su, Wei Jin 99/9/27
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMFD2.1 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "b3soifddef.h"
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
B3SOIFDsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt,
             int *states)
{
B3SOIFDmodel *model = (B3SOIFDmodel*)inModel;
B3SOIFDinstance *here;
int error;
CKTnode *tmp;

double tmp1, tmp2;
double nfb0, Cboxt;

CKTnode *tmpNode;
IFuid tmpName;


    /*  loop through all the B3SOIFD device models */
    for( ; model != NULL; model = B3SOIFDnextModel(model))
    {
/* Default value Processing for B3SOIFD MOSFET Models */

        if (!model->B3SOIFDtypeGiven)
            model->B3SOIFDtype = NMOS;     
        if (!model->B3SOIFDmobModGiven) 
            model->B3SOIFDmobMod = 1;
        if (!model->B3SOIFDbinUnitGiven) 
            model->B3SOIFDbinUnit = 1;
        if (!model->B3SOIFDparamChkGiven) 
            model->B3SOIFDparamChk = 0;
        if (!model->B3SOIFDcapModGiven) 
            model->B3SOIFDcapMod = 2;
        if (!model->B3SOIFDnoiModGiven) 
            model->B3SOIFDnoiMod = 1;
        if (!model->B3SOIFDshModGiven) 
            model->B3SOIFDshMod = 0;
        if (!model->B3SOIFDversionGiven) 
            model->B3SOIFDversion = 2.0;
        if (!model->B3SOIFDtoxGiven)
            model->B3SOIFDtox = 100.0e-10;
        model->B3SOIFDcox = 3.453133e-11 / model->B3SOIFDtox;

        if (!model->B3SOIFDcdscGiven)
	    model->B3SOIFDcdsc = 2.4e-4;   /* unit Q/V/m^2  */
        if (!model->B3SOIFDcdscbGiven)
	    model->B3SOIFDcdscb = 0.0;   /* unit Q/V/m^2  */    
        if (!model->B3SOIFDcdscdGiven)
	    model->B3SOIFDcdscd = 0.0;   /* unit Q/V/m^2  */
        if (!model->B3SOIFDcitGiven)
	    model->B3SOIFDcit = 0.0;   /* unit Q/V/m^2  */
        if (!model->B3SOIFDnfactorGiven)
	    model->B3SOIFDnfactor = 1;
        if (!model->B3SOIFDvsatGiven)
            model->B3SOIFDvsat = 8.0e4;    /* unit m/s */ 
        if (!model->B3SOIFDatGiven)
            model->B3SOIFDat = 3.3e4;    /* unit m/s */ 
        if (!model->B3SOIFDa0Given)
            model->B3SOIFDa0 = 1.0;  
        if (!model->B3SOIFDagsGiven)
            model->B3SOIFDags = 0.0;
        if (!model->B3SOIFDa1Given)
            model->B3SOIFDa1 = 0.0;
        if (!model->B3SOIFDa2Given)
            model->B3SOIFDa2 = 1.0;
        if (!model->B3SOIFDketaGiven)
            model->B3SOIFDketa = -0.6;    /* unit  / V */
        if (!model->B3SOIFDnsubGiven)
            model->B3SOIFDnsub = 6.0e16;   /* unit 1/cm3 */
        if (!model->B3SOIFDnpeakGiven)
            model->B3SOIFDnpeak = 1.7e17;   /* unit 1/cm3 */
        if (!model->B3SOIFDngateGiven)
            model->B3SOIFDngate = 0;   /* unit 1/cm3 */
        if (!model->B3SOIFDvbmGiven)
	    model->B3SOIFDvbm = -3.0;
        if (!model->B3SOIFDxtGiven)
	    model->B3SOIFDxt = 1.55e-7;
        if (!model->B3SOIFDkt1Given)
            model->B3SOIFDkt1 = -0.11;      /* unit V */
        if (!model->B3SOIFDkt1lGiven)
            model->B3SOIFDkt1l = 0.0;      /* unit V*m */
        if (!model->B3SOIFDkt2Given)
            model->B3SOIFDkt2 = 0.022;      /* No unit */
        if (!model->B3SOIFDk3Given)
            model->B3SOIFDk3 = 0.0;      
        if (!model->B3SOIFDk3bGiven)
            model->B3SOIFDk3b = 0.0;      
        if (!model->B3SOIFDw0Given)
            model->B3SOIFDw0 = 2.5e-6;    
        if (!model->B3SOIFDnlxGiven)
            model->B3SOIFDnlx = 1.74e-7;     
        if (!model->B3SOIFDdvt0Given)
            model->B3SOIFDdvt0 = 2.2;    
        if (!model->B3SOIFDdvt1Given)
            model->B3SOIFDdvt1 = 0.53;      
        if (!model->B3SOIFDdvt2Given)
            model->B3SOIFDdvt2 = -0.032;   /* unit 1 / V */     

        if (!model->B3SOIFDdvt0wGiven)
            model->B3SOIFDdvt0w = 0.0;    
        if (!model->B3SOIFDdvt1wGiven)
            model->B3SOIFDdvt1w = 5.3e6;    
        if (!model->B3SOIFDdvt2wGiven)
            model->B3SOIFDdvt2w = -0.032;   

        if (!model->B3SOIFDdroutGiven)
            model->B3SOIFDdrout = 0.56;     
        if (!model->B3SOIFDdsubGiven)
            model->B3SOIFDdsub = model->B3SOIFDdrout;     
        if (!model->B3SOIFDvth0Given)
            model->B3SOIFDvth0 = (model->B3SOIFDtype == NMOS) ? 0.7 : -0.7;
        if (!model->B3SOIFDuaGiven)
            model->B3SOIFDua = 2.25e-9;      /* unit m/V */
        if (!model->B3SOIFDua1Given)
            model->B3SOIFDua1 = 4.31e-9;      /* unit m/V */
        if (!model->B3SOIFDubGiven)
            model->B3SOIFDub = 5.87e-19;     /* unit (m/V)**2 */
        if (!model->B3SOIFDub1Given)
            model->B3SOIFDub1 = -7.61e-18;     /* unit (m/V)**2 */
        if (!model->B3SOIFDucGiven)
            model->B3SOIFDuc = (model->B3SOIFDmobMod == 3) ? -0.0465 : -0.0465e-9;   
        if (!model->B3SOIFDuc1Given)
            model->B3SOIFDuc1 = (model->B3SOIFDmobMod == 3) ? -0.056 : -0.056e-9;   
        if (!model->B3SOIFDu0Given)
            model->B3SOIFDu0 = (model->B3SOIFDtype == NMOS) ? 0.067 : 0.025;
        if (!model->B3SOIFDuteGiven)
	    model->B3SOIFDute = -1.5;    
        if (!model->B3SOIFDvoffGiven)
	    model->B3SOIFDvoff = -0.08;
        if (!model->B3SOIFDdeltaGiven)  
           model->B3SOIFDdelta = 0.01;
        if (!model->B3SOIFDrdswGiven)
            model->B3SOIFDrdsw = 100;
        if (!model->B3SOIFDprwgGiven)
            model->B3SOIFDprwg = 0.0;      /* unit 1/V */
        if (!model->B3SOIFDprwbGiven)
            model->B3SOIFDprwb = 0.0;      
        if (!model->B3SOIFDprtGiven)
            model->B3SOIFDprt = 0.0;      
        if (!model->B3SOIFDeta0Given)
            model->B3SOIFDeta0 = 0.08;      /* no unit  */ 
        if (!model->B3SOIFDetabGiven)
            model->B3SOIFDetab = -0.07;      /* unit  1/V */ 
        if (!model->B3SOIFDpclmGiven)
            model->B3SOIFDpclm = 1.3;      /* no unit  */ 
        if (!model->B3SOIFDpdibl1Given)
            model->B3SOIFDpdibl1 = .39;    /* no unit  */
        if (!model->B3SOIFDpdibl2Given)
            model->B3SOIFDpdibl2 = 0.0086;    /* no unit  */ 
        if (!model->B3SOIFDpdiblbGiven)
            model->B3SOIFDpdiblb = 0.0;    /* 1/V  */ 
        if (!model->B3SOIFDpvagGiven)
            model->B3SOIFDpvag = 0.0;     
        if (!model->B3SOIFDwrGiven)  
            model->B3SOIFDwr = 1.0;
        if (!model->B3SOIFDdwgGiven)  
            model->B3SOIFDdwg = 0.0;
        if (!model->B3SOIFDdwbGiven)  
            model->B3SOIFDdwb = 0.0;
        if (!model->B3SOIFDb0Given)
            model->B3SOIFDb0 = 0.0;
        if (!model->B3SOIFDb1Given)  
            model->B3SOIFDb1 = 0.0;
        if (!model->B3SOIFDalpha0Given)  
            model->B3SOIFDalpha0 = 0.0;
        if (!model->B3SOIFDalpha1Given)  
            model->B3SOIFDalpha1 = 1.0;
        if (!model->B3SOIFDbeta0Given)  
            model->B3SOIFDbeta0 = 30.0;

        if (!model->B3SOIFDcgslGiven)  
            model->B3SOIFDcgsl = 0.0;
        if (!model->B3SOIFDcgdlGiven)  
            model->B3SOIFDcgdl = 0.0;
        if (!model->B3SOIFDckappaGiven)  
            model->B3SOIFDckappa = 0.6;
        if (!model->B3SOIFDclcGiven)  
            model->B3SOIFDclc = 0.1e-7;
        if (!model->B3SOIFDcleGiven)  
            model->B3SOIFDcle = 0.0;
        if (!model->B3SOIFDtboxGiven)  
            model->B3SOIFDtbox = 3e-7;
        if (!model->B3SOIFDtsiGiven)  
            model->B3SOIFDtsi = 1e-7;
        if (!model->B3SOIFDxjGiven)  
            model->B3SOIFDxj = model->B3SOIFDtsi;
        if (!model->B3SOIFDkb1Given)  
            model->B3SOIFDkb1 = 1;
        if (!model->B3SOIFDkb3Given)  
            model->B3SOIFDkb3 = 1;
        if (!model->B3SOIFDdvbd0Given)  
            model->B3SOIFDdvbd0 = 0.0;
        if (!model->B3SOIFDdvbd1Given)  
            model->B3SOIFDdvbd1 = 0.0;
        if (!model->B3SOIFDvbsaGiven)  
            model->B3SOIFDvbsa = 0.0;
        if (!model->B3SOIFDdelpGiven)  
            model->B3SOIFDdelp = 0.02;
        if (!model->B3SOIFDrbodyGiven)  
            model->B3SOIFDrbody = 0.0;
        if (!model->B3SOIFDrbshGiven)  
            model->B3SOIFDrbsh = 0.0;
        if (!model->B3SOIFDadice0Given)  
            model->B3SOIFDadice0 = 1;
        if (!model->B3SOIFDabpGiven)  
            model->B3SOIFDabp = 1;
        if (!model->B3SOIFDmxcGiven)  
            model->B3SOIFDmxc = -0.9;
        if (!model->B3SOIFDrth0Given)  
            model->B3SOIFDrth0 = 0;

        if (!model->B3SOIFDcth0Given)
            model->B3SOIFDcth0 =0;

        if (!model->B3SOIFDaiiGiven)  
            model->B3SOIFDaii = 0.0;
        if (!model->B3SOIFDbiiGiven)  
            model->B3SOIFDbii = 0.0;
        if (!model->B3SOIFDciiGiven)  
            model->B3SOIFDcii = 0.0;
        if (!model->B3SOIFDdiiGiven)  
            model->B3SOIFDdii = -1.0;
        if (!model->B3SOIFDagidlGiven)  
            model->B3SOIFDagidl = 0.0;
        if (!model->B3SOIFDbgidlGiven)  
            model->B3SOIFDbgidl = 0.0;
        if (!model->B3SOIFDngidlGiven)  
            model->B3SOIFDngidl = 1.2;
        if (!model->B3SOIFDndiodeGiven)  
            model->B3SOIFDndiode = 1.0;
        if (!model->B3SOIFDntunGiven)  
            model->B3SOIFDntun = 10.0;
        if (!model->B3SOIFDisbjtGiven)  
            model->B3SOIFDisbjt = 1e-6;
        if (!model->B3SOIFDisdifGiven)  
            model->B3SOIFDisdif = 0.0;
        if (!model->B3SOIFDisrecGiven)  
            model->B3SOIFDisrec = 1e-5;
        if (!model->B3SOIFDistunGiven)  
            model->B3SOIFDistun = 0.0;
        if (!model->B3SOIFDxbjtGiven)  
            model->B3SOIFDxbjt = 2;
        if (!model->B3SOIFDxdifGiven)  
            model->B3SOIFDxdif = 2;
        if (!model->B3SOIFDxrecGiven)  
            model->B3SOIFDxrec = 20;
        if (!model->B3SOIFDxtunGiven)  
            model->B3SOIFDxtun = 0;
        if (!model->B3SOIFDedlGiven)  
            model->B3SOIFDedl = 2e-6;
        if (!model->B3SOIFDkbjt1Given)  
            model->B3SOIFDkbjt1 = 0;
        if (!model->B3SOIFDttGiven)  
            model->B3SOIFDtt = 1e-12;
        if (!model->B3SOIFDasdGiven)  
            model->B3SOIFDasd = 0.3;

        /* unit degree celcius */
        if (!model->B3SOIFDtnomGiven)  
	    model->B3SOIFDtnom = ckt->CKTnomTemp; 
        if (!model->B3SOIFDLintGiven)  
           model->B3SOIFDLint = 0.0;
        if (!model->B3SOIFDLlGiven)  
           model->B3SOIFDLl = 0.0;
        if (!model->B3SOIFDLlnGiven)  
           model->B3SOIFDLln = 1.0;
        if (!model->B3SOIFDLwGiven)  
           model->B3SOIFDLw = 0.0;
        if (!model->B3SOIFDLwnGiven)  
           model->B3SOIFDLwn = 1.0;
        if (!model->B3SOIFDLwlGiven)  
           model->B3SOIFDLwl = 0.0;
        if (!model->B3SOIFDLminGiven)  
           model->B3SOIFDLmin = 0.0;
        if (!model->B3SOIFDLmaxGiven)  
           model->B3SOIFDLmax = 1.0;
        if (!model->B3SOIFDWintGiven)  
           model->B3SOIFDWint = 0.0;
        if (!model->B3SOIFDWlGiven)  
           model->B3SOIFDWl = 0.0;
        if (!model->B3SOIFDWlnGiven)  
           model->B3SOIFDWln = 1.0;
        if (!model->B3SOIFDWwGiven)  
           model->B3SOIFDWw = 0.0;
        if (!model->B3SOIFDWwnGiven)  
           model->B3SOIFDWwn = 1.0;
        if (!model->B3SOIFDWwlGiven)  
           model->B3SOIFDWwl = 0.0;
        if (!model->B3SOIFDWminGiven)  
           model->B3SOIFDWmin = 0.0;
        if (!model->B3SOIFDWmaxGiven)  
           model->B3SOIFDWmax = 1.0;
        if (!model->B3SOIFDdwcGiven)  
           model->B3SOIFDdwc = model->B3SOIFDWint;
        if (!model->B3SOIFDdlcGiven)  
           model->B3SOIFDdlc = model->B3SOIFDLint;

/* Added for binning - START */
        /* Length dependence */
        if (!model->B3SOIFDlnpeakGiven)
            model->B3SOIFDlnpeak = 0.0;
        if (!model->B3SOIFDlnsubGiven)
            model->B3SOIFDlnsub = 0.0;
        if (!model->B3SOIFDlngateGiven)
            model->B3SOIFDlngate = 0.0;
        if (!model->B3SOIFDlvth0Given)
           model->B3SOIFDlvth0 = 0.0;
        if (!model->B3SOIFDlk1Given)
            model->B3SOIFDlk1 = 0.0;
        if (!model->B3SOIFDlk2Given)
            model->B3SOIFDlk2 = 0.0;
        if (!model->B3SOIFDlk3Given)
            model->B3SOIFDlk3 = 0.0;
        if (!model->B3SOIFDlk3bGiven)
            model->B3SOIFDlk3b = 0.0;
        if (!model->B3SOIFDlvbsaGiven)
            model->B3SOIFDlvbsa = 0.0;
        if (!model->B3SOIFDldelpGiven)
            model->B3SOIFDldelp = 0.0;
        if (!model->B3SOIFDlkb1Given)
           model->B3SOIFDlkb1 = 0.0;
        if (!model->B3SOIFDlkb3Given)
           model->B3SOIFDlkb3 = 1.0;
        if (!model->B3SOIFDldvbd0Given)
           model->B3SOIFDldvbd0 = 1.0;
        if (!model->B3SOIFDldvbd1Given)
           model->B3SOIFDldvbd1 = 1.0;
        if (!model->B3SOIFDlw0Given)
            model->B3SOIFDlw0 = 0.0;
        if (!model->B3SOIFDlnlxGiven)
            model->B3SOIFDlnlx = 0.0;
        if (!model->B3SOIFDldvt0Given)
            model->B3SOIFDldvt0 = 0.0;
        if (!model->B3SOIFDldvt1Given)
            model->B3SOIFDldvt1 = 0.0;
        if (!model->B3SOIFDldvt2Given)
            model->B3SOIFDldvt2 = 0.0;
        if (!model->B3SOIFDldvt0wGiven)
            model->B3SOIFDldvt0w = 0.0;
        if (!model->B3SOIFDldvt1wGiven)
            model->B3SOIFDldvt1w = 0.0;
        if (!model->B3SOIFDldvt2wGiven)
            model->B3SOIFDldvt2w = 0.0;
        if (!model->B3SOIFDlu0Given)
            model->B3SOIFDlu0 = 0.0;
        if (!model->B3SOIFDluaGiven)
            model->B3SOIFDlua = 0.0;
        if (!model->B3SOIFDlubGiven)
            model->B3SOIFDlub = 0.0;
        if (!model->B3SOIFDlucGiven)
            model->B3SOIFDluc = 0.0;
        if (!model->B3SOIFDlvsatGiven)
            model->B3SOIFDlvsat = 0.0;
        if (!model->B3SOIFDla0Given)
            model->B3SOIFDla0 = 0.0;
        if (!model->B3SOIFDlagsGiven)
            model->B3SOIFDlags = 0.0;
        if (!model->B3SOIFDlb0Given)
            model->B3SOIFDlb0 = 0.0;
        if (!model->B3SOIFDlb1Given)
            model->B3SOIFDlb1 = 0.0;
        if (!model->B3SOIFDlketaGiven)
            model->B3SOIFDlketa = 0.0;
        if (!model->B3SOIFDlabpGiven)
            model->B3SOIFDlabp = 0.0;
        if (!model->B3SOIFDlmxcGiven)
            model->B3SOIFDlmxc = 0.0;
        if (!model->B3SOIFDladice0Given)
            model->B3SOIFDladice0 = 0.0;
        if (!model->B3SOIFDla1Given)
            model->B3SOIFDla1 = 0.0;
        if (!model->B3SOIFDla2Given)
            model->B3SOIFDla2 = 0.0;
        if (!model->B3SOIFDlrdswGiven)
            model->B3SOIFDlrdsw = 0.0;
        if (!model->B3SOIFDlprwbGiven)
            model->B3SOIFDlprwb = 0.0;
        if (!model->B3SOIFDlprwgGiven)
            model->B3SOIFDlprwg = 0.0;
        if (!model->B3SOIFDlwrGiven)
            model->B3SOIFDlwr = 0.0;
        if (!model->B3SOIFDlnfactorGiven)
            model->B3SOIFDlnfactor = 0.0;
        if (!model->B3SOIFDldwgGiven)
            model->B3SOIFDldwg = 0.0;
        if (!model->B3SOIFDldwbGiven)
            model->B3SOIFDldwb = 0.0;
        if (!model->B3SOIFDlvoffGiven)
            model->B3SOIFDlvoff = 0.0;
        if (!model->B3SOIFDleta0Given)
            model->B3SOIFDleta0 = 0.0;
        if (!model->B3SOIFDletabGiven)
            model->B3SOIFDletab = 0.0;
        if (!model->B3SOIFDldsubGiven)
            model->B3SOIFDldsub = 0.0;
        if (!model->B3SOIFDlcitGiven)
            model->B3SOIFDlcit = 0.0;
        if (!model->B3SOIFDlcdscGiven)
            model->B3SOIFDlcdsc = 0.0;
        if (!model->B3SOIFDlcdscbGiven)
            model->B3SOIFDlcdscb = 0.0;
        if (!model->B3SOIFDlcdscdGiven)
            model->B3SOIFDlcdscd = 0.0;
        if (!model->B3SOIFDlpclmGiven)
            model->B3SOIFDlpclm = 0.0;
        if (!model->B3SOIFDlpdibl1Given)
            model->B3SOIFDlpdibl1 = 0.0;
        if (!model->B3SOIFDlpdibl2Given)
            model->B3SOIFDlpdibl2 = 0.0;
        if (!model->B3SOIFDlpdiblbGiven)
            model->B3SOIFDlpdiblb = 0.0;
        if (!model->B3SOIFDldroutGiven)
            model->B3SOIFDldrout = 0.0;
        if (!model->B3SOIFDlpvagGiven)
            model->B3SOIFDlpvag = 0.0;
        if (!model->B3SOIFDldeltaGiven)
            model->B3SOIFDldelta = 0.0;
        if (!model->B3SOIFDlaiiGiven)
            model->B3SOIFDlaii = 0.0;
        if (!model->B3SOIFDlbiiGiven)
            model->B3SOIFDlbii = 0.0;
        if (!model->B3SOIFDlciiGiven)
            model->B3SOIFDlcii = 0.0;
        if (!model->B3SOIFDldiiGiven)
            model->B3SOIFDldii = 0.0;
        if (!model->B3SOIFDlalpha0Given)
            model->B3SOIFDlalpha0 = 0.0;
        if (!model->B3SOIFDlalpha1Given)
            model->B3SOIFDlalpha1 = 0.0;
        if (!model->B3SOIFDlbeta0Given)
            model->B3SOIFDlbeta0 = 0.0;
        if (!model->B3SOIFDlagidlGiven)
            model->B3SOIFDlagidl = 0.0;
        if (!model->B3SOIFDlbgidlGiven)
            model->B3SOIFDlbgidl = 0.0;
        if (!model->B3SOIFDlngidlGiven)
            model->B3SOIFDlngidl = 0.0;
        if (!model->B3SOIFDlntunGiven)
            model->B3SOIFDlntun = 0.0;
        if (!model->B3SOIFDlndiodeGiven)
            model->B3SOIFDlndiode = 0.0;
        if (!model->B3SOIFDlisbjtGiven)
            model->B3SOIFDlisbjt = 0.0;
        if (!model->B3SOIFDlisdifGiven)
            model->B3SOIFDlisdif = 0.0;
        if (!model->B3SOIFDlisrecGiven)
            model->B3SOIFDlisrec = 0.0;
        if (!model->B3SOIFDlistunGiven)
            model->B3SOIFDlistun = 0.0;
        if (!model->B3SOIFDledlGiven)
            model->B3SOIFDledl = 0.0;
        if (!model->B3SOIFDlkbjt1Given)
            model->B3SOIFDlkbjt1 = 0.0;
	/* CV Model */
        if (!model->B3SOIFDlvsdfbGiven)
            model->B3SOIFDlvsdfb = 0.0;
        if (!model->B3SOIFDlvsdthGiven)
            model->B3SOIFDlvsdth = 0.0;
        /* Width dependence */
        if (!model->B3SOIFDwnpeakGiven)
            model->B3SOIFDwnpeak = 0.0;
        if (!model->B3SOIFDwnsubGiven)
            model->B3SOIFDwnsub = 0.0;
        if (!model->B3SOIFDwngateGiven)
            model->B3SOIFDwngate = 0.0;
        if (!model->B3SOIFDwvth0Given)
           model->B3SOIFDwvth0 = 0.0;
        if (!model->B3SOIFDwk1Given)
            model->B3SOIFDwk1 = 0.0;
        if (!model->B3SOIFDwk2Given)
            model->B3SOIFDwk2 = 0.0;
        if (!model->B3SOIFDwk3Given)
            model->B3SOIFDwk3 = 0.0;
        if (!model->B3SOIFDwk3bGiven)
            model->B3SOIFDwk3b = 0.0;
        if (!model->B3SOIFDwvbsaGiven)
            model->B3SOIFDwvbsa = 0.0;
        if (!model->B3SOIFDwdelpGiven)
            model->B3SOIFDwdelp = 0.0;
        if (!model->B3SOIFDwkb1Given)
           model->B3SOIFDwkb1 = 0.0;
        if (!model->B3SOIFDwkb3Given)
           model->B3SOIFDwkb3 = 1.0;
        if (!model->B3SOIFDwdvbd0Given)
           model->B3SOIFDwdvbd0 = 1.0;
        if (!model->B3SOIFDwdvbd1Given)
           model->B3SOIFDwdvbd1 = 1.0;
        if (!model->B3SOIFDww0Given)
            model->B3SOIFDww0 = 0.0;
        if (!model->B3SOIFDwnlxGiven)
            model->B3SOIFDwnlx = 0.0;
        if (!model->B3SOIFDwdvt0Given)
            model->B3SOIFDwdvt0 = 0.0;
        if (!model->B3SOIFDwdvt1Given)
            model->B3SOIFDwdvt1 = 0.0;
        if (!model->B3SOIFDwdvt2Given)
            model->B3SOIFDwdvt2 = 0.0;
        if (!model->B3SOIFDwdvt0wGiven)
            model->B3SOIFDwdvt0w = 0.0;
        if (!model->B3SOIFDwdvt1wGiven)
            model->B3SOIFDwdvt1w = 0.0;
        if (!model->B3SOIFDwdvt2wGiven)
            model->B3SOIFDwdvt2w = 0.0;
        if (!model->B3SOIFDwu0Given)
            model->B3SOIFDwu0 = 0.0;
        if (!model->B3SOIFDwuaGiven)
            model->B3SOIFDwua = 0.0;
        if (!model->B3SOIFDwubGiven)
            model->B3SOIFDwub = 0.0;
        if (!model->B3SOIFDwucGiven)
            model->B3SOIFDwuc = 0.0;
        if (!model->B3SOIFDwvsatGiven)
            model->B3SOIFDwvsat = 0.0;
        if (!model->B3SOIFDwa0Given)
            model->B3SOIFDwa0 = 0.0;
        if (!model->B3SOIFDwagsGiven)
            model->B3SOIFDwags = 0.0;
        if (!model->B3SOIFDwb0Given)
            model->B3SOIFDwb0 = 0.0;
        if (!model->B3SOIFDwb1Given)
            model->B3SOIFDwb1 = 0.0;
        if (!model->B3SOIFDwketaGiven)
            model->B3SOIFDwketa = 0.0;
        if (!model->B3SOIFDwabpGiven)
            model->B3SOIFDwabp = 0.0;
        if (!model->B3SOIFDwmxcGiven)
            model->B3SOIFDwmxc = 0.0;
        if (!model->B3SOIFDwadice0Given)
            model->B3SOIFDwadice0 = 0.0;
        if (!model->B3SOIFDwa1Given)
            model->B3SOIFDwa1 = 0.0;
        if (!model->B3SOIFDwa2Given)
            model->B3SOIFDwa2 = 0.0;
        if (!model->B3SOIFDwrdswGiven)
            model->B3SOIFDwrdsw = 0.0;
        if (!model->B3SOIFDwprwbGiven)
            model->B3SOIFDwprwb = 0.0;
        if (!model->B3SOIFDwprwgGiven)
            model->B3SOIFDwprwg = 0.0;
        if (!model->B3SOIFDwwrGiven)
            model->B3SOIFDwwr = 0.0;
        if (!model->B3SOIFDwnfactorGiven)
            model->B3SOIFDwnfactor = 0.0;
        if (!model->B3SOIFDwdwgGiven)
            model->B3SOIFDwdwg = 0.0;
        if (!model->B3SOIFDwdwbGiven)
            model->B3SOIFDwdwb = 0.0;
        if (!model->B3SOIFDwvoffGiven)
            model->B3SOIFDwvoff = 0.0;
        if (!model->B3SOIFDweta0Given)
            model->B3SOIFDweta0 = 0.0;
        if (!model->B3SOIFDwetabGiven)
            model->B3SOIFDwetab = 0.0;
        if (!model->B3SOIFDwdsubGiven)
            model->B3SOIFDwdsub = 0.0;
        if (!model->B3SOIFDwcitGiven)
            model->B3SOIFDwcit = 0.0;
        if (!model->B3SOIFDwcdscGiven)
            model->B3SOIFDwcdsc = 0.0;
        if (!model->B3SOIFDwcdscbGiven)
            model->B3SOIFDwcdscb = 0.0;
        if (!model->B3SOIFDwcdscdGiven)
            model->B3SOIFDwcdscd = 0.0;
        if (!model->B3SOIFDwpclmGiven)
            model->B3SOIFDwpclm = 0.0;
        if (!model->B3SOIFDwpdibl1Given)
            model->B3SOIFDwpdibl1 = 0.0;
        if (!model->B3SOIFDwpdibl2Given)
            model->B3SOIFDwpdibl2 = 0.0;
        if (!model->B3SOIFDwpdiblbGiven)
            model->B3SOIFDwpdiblb = 0.0;
        if (!model->B3SOIFDwdroutGiven)
            model->B3SOIFDwdrout = 0.0;
        if (!model->B3SOIFDwpvagGiven)
            model->B3SOIFDwpvag = 0.0;
        if (!model->B3SOIFDwdeltaGiven)
            model->B3SOIFDwdelta = 0.0;
        if (!model->B3SOIFDwaiiGiven)
            model->B3SOIFDwaii = 0.0;
        if (!model->B3SOIFDwbiiGiven)
            model->B3SOIFDwbii = 0.0;
        if (!model->B3SOIFDwciiGiven)
            model->B3SOIFDwcii = 0.0;
        if (!model->B3SOIFDwdiiGiven)
            model->B3SOIFDwdii = 0.0;
        if (!model->B3SOIFDwalpha0Given)
            model->B3SOIFDwalpha0 = 0.0;
        if (!model->B3SOIFDwalpha1Given)
            model->B3SOIFDwalpha1 = 0.0;
        if (!model->B3SOIFDwbeta0Given)
            model->B3SOIFDwbeta0 = 0.0;
        if (!model->B3SOIFDwagidlGiven)
            model->B3SOIFDwagidl = 0.0;
        if (!model->B3SOIFDwbgidlGiven)
            model->B3SOIFDwbgidl = 0.0;
        if (!model->B3SOIFDwngidlGiven)
            model->B3SOIFDwngidl = 0.0;
        if (!model->B3SOIFDwntunGiven)
            model->B3SOIFDwntun = 0.0;
        if (!model->B3SOIFDwndiodeGiven)
            model->B3SOIFDwndiode = 0.0;
        if (!model->B3SOIFDwisbjtGiven)
            model->B3SOIFDwisbjt = 0.0;
        if (!model->B3SOIFDwisdifGiven)
            model->B3SOIFDwisdif = 0.0;
        if (!model->B3SOIFDwisrecGiven)
            model->B3SOIFDwisrec = 0.0;
        if (!model->B3SOIFDwistunGiven)
            model->B3SOIFDwistun = 0.0;
        if (!model->B3SOIFDwedlGiven)
            model->B3SOIFDwedl = 0.0;
        if (!model->B3SOIFDwkbjt1Given)
            model->B3SOIFDwkbjt1 = 0.0;
	/* CV Model */
        if (!model->B3SOIFDwvsdfbGiven)
            model->B3SOIFDwvsdfb = 0.0;
        if (!model->B3SOIFDwvsdthGiven)
            model->B3SOIFDwvsdth = 0.0;
        /* Cross-term dependence */
        if (!model->B3SOIFDpnpeakGiven)
            model->B3SOIFDpnpeak = 0.0;
        if (!model->B3SOIFDpnsubGiven)
            model->B3SOIFDpnsub = 0.0;
        if (!model->B3SOIFDpngateGiven)
            model->B3SOIFDpngate = 0.0;
        if (!model->B3SOIFDpvth0Given)
           model->B3SOIFDpvth0 = 0.0;
        if (!model->B3SOIFDpk1Given)
            model->B3SOIFDpk1 = 0.0;
        if (!model->B3SOIFDpk2Given)
            model->B3SOIFDpk2 = 0.0;
        if (!model->B3SOIFDpk3Given)
            model->B3SOIFDpk3 = 0.0;
        if (!model->B3SOIFDpk3bGiven)
            model->B3SOIFDpk3b = 0.0;
        if (!model->B3SOIFDpvbsaGiven)
            model->B3SOIFDpvbsa = 0.0;
        if (!model->B3SOIFDpdelpGiven)
            model->B3SOIFDpdelp = 0.0;
        if (!model->B3SOIFDpkb1Given)
           model->B3SOIFDpkb1 = 0.0;
        if (!model->B3SOIFDpkb3Given)
           model->B3SOIFDpkb3 = 1.0;
        if (!model->B3SOIFDpdvbd0Given)
           model->B3SOIFDpdvbd0 = 1.0;
        if (!model->B3SOIFDpdvbd1Given)
           model->B3SOIFDpdvbd1 = 1.0;
        if (!model->B3SOIFDpw0Given)
            model->B3SOIFDpw0 = 0.0;
        if (!model->B3SOIFDpnlxGiven)
            model->B3SOIFDpnlx = 0.0;
        if (!model->B3SOIFDpdvt0Given)
            model->B3SOIFDpdvt0 = 0.0;
        if (!model->B3SOIFDpdvt1Given)
            model->B3SOIFDpdvt1 = 0.0;
        if (!model->B3SOIFDpdvt2Given)
            model->B3SOIFDpdvt2 = 0.0;
        if (!model->B3SOIFDpdvt0wGiven)
            model->B3SOIFDpdvt0w = 0.0;
        if (!model->B3SOIFDpdvt1wGiven)
            model->B3SOIFDpdvt1w = 0.0;
        if (!model->B3SOIFDpdvt2wGiven)
            model->B3SOIFDpdvt2w = 0.0;
        if (!model->B3SOIFDpu0Given)
            model->B3SOIFDpu0 = 0.0;
        if (!model->B3SOIFDpuaGiven)
            model->B3SOIFDpua = 0.0;
        if (!model->B3SOIFDpubGiven)
            model->B3SOIFDpub = 0.0;
        if (!model->B3SOIFDpucGiven)
            model->B3SOIFDpuc = 0.0;
        if (!model->B3SOIFDpvsatGiven)
            model->B3SOIFDpvsat = 0.0;
        if (!model->B3SOIFDpa0Given)
            model->B3SOIFDpa0 = 0.0;
        if (!model->B3SOIFDpagsGiven)
            model->B3SOIFDpags = 0.0;
        if (!model->B3SOIFDpb0Given)
            model->B3SOIFDpb0 = 0.0;
        if (!model->B3SOIFDpb1Given)
            model->B3SOIFDpb1 = 0.0;
        if (!model->B3SOIFDpketaGiven)
            model->B3SOIFDpketa = 0.0;
        if (!model->B3SOIFDpabpGiven)
            model->B3SOIFDpabp = 0.0;
        if (!model->B3SOIFDpmxcGiven)
            model->B3SOIFDpmxc = 0.0;
        if (!model->B3SOIFDpadice0Given)
            model->B3SOIFDpadice0 = 0.0;
        if (!model->B3SOIFDpa1Given)
            model->B3SOIFDpa1 = 0.0;
        if (!model->B3SOIFDpa2Given)
            model->B3SOIFDpa2 = 0.0;
        if (!model->B3SOIFDprdswGiven)
            model->B3SOIFDprdsw = 0.0;
        if (!model->B3SOIFDpprwbGiven)
            model->B3SOIFDpprwb = 0.0;
        if (!model->B3SOIFDpprwgGiven)
            model->B3SOIFDpprwg = 0.0;
        if (!model->B3SOIFDpwrGiven)
            model->B3SOIFDpwr = 0.0;
        if (!model->B3SOIFDpnfactorGiven)
            model->B3SOIFDpnfactor = 0.0;
        if (!model->B3SOIFDpdwgGiven)
            model->B3SOIFDpdwg = 0.0;
        if (!model->B3SOIFDpdwbGiven)
            model->B3SOIFDpdwb = 0.0;
        if (!model->B3SOIFDpvoffGiven)
            model->B3SOIFDpvoff = 0.0;
        if (!model->B3SOIFDpeta0Given)
            model->B3SOIFDpeta0 = 0.0;
        if (!model->B3SOIFDpetabGiven)
            model->B3SOIFDpetab = 0.0;
        if (!model->B3SOIFDpdsubGiven)
            model->B3SOIFDpdsub = 0.0;
        if (!model->B3SOIFDpcitGiven)
            model->B3SOIFDpcit = 0.0;
        if (!model->B3SOIFDpcdscGiven)
            model->B3SOIFDpcdsc = 0.0;
        if (!model->B3SOIFDpcdscbGiven)
            model->B3SOIFDpcdscb = 0.0;
        if (!model->B3SOIFDpcdscdGiven)
            model->B3SOIFDpcdscd = 0.0;
        if (!model->B3SOIFDppclmGiven)
            model->B3SOIFDppclm = 0.0;
        if (!model->B3SOIFDppdibl1Given)
            model->B3SOIFDppdibl1 = 0.0;
        if (!model->B3SOIFDppdibl2Given)
            model->B3SOIFDppdibl2 = 0.0;
        if (!model->B3SOIFDppdiblbGiven)
            model->B3SOIFDppdiblb = 0.0;
        if (!model->B3SOIFDpdroutGiven)
            model->B3SOIFDpdrout = 0.0;
        if (!model->B3SOIFDppvagGiven)
            model->B3SOIFDppvag = 0.0;
        if (!model->B3SOIFDpdeltaGiven)
            model->B3SOIFDpdelta = 0.0;
        if (!model->B3SOIFDpaiiGiven)
            model->B3SOIFDpaii = 0.0;
        if (!model->B3SOIFDpbiiGiven)
            model->B3SOIFDpbii = 0.0;
        if (!model->B3SOIFDpciiGiven)
            model->B3SOIFDpcii = 0.0;
        if (!model->B3SOIFDpdiiGiven)
            model->B3SOIFDpdii = 0.0;
        if (!model->B3SOIFDpalpha0Given)
            model->B3SOIFDpalpha0 = 0.0;
        if (!model->B3SOIFDpalpha1Given)
            model->B3SOIFDpalpha1 = 0.0;
        if (!model->B3SOIFDpbeta0Given)
            model->B3SOIFDpbeta0 = 0.0;
        if (!model->B3SOIFDpagidlGiven)
            model->B3SOIFDpagidl = 0.0;
        if (!model->B3SOIFDpbgidlGiven)
            model->B3SOIFDpbgidl = 0.0;
        if (!model->B3SOIFDpngidlGiven)
            model->B3SOIFDpngidl = 0.0;
        if (!model->B3SOIFDpntunGiven)
            model->B3SOIFDpntun = 0.0;
        if (!model->B3SOIFDpndiodeGiven)
            model->B3SOIFDpndiode = 0.0;
        if (!model->B3SOIFDpisbjtGiven)
            model->B3SOIFDpisbjt = 0.0;
        if (!model->B3SOIFDpisdifGiven)
            model->B3SOIFDpisdif = 0.0;
        if (!model->B3SOIFDpisrecGiven)
            model->B3SOIFDpisrec = 0.0;
        if (!model->B3SOIFDpistunGiven)
            model->B3SOIFDpistun = 0.0;
        if (!model->B3SOIFDpedlGiven)
            model->B3SOIFDpedl = 0.0;
        if (!model->B3SOIFDpkbjt1Given)
            model->B3SOIFDpkbjt1 = 0.0;
	/* CV Model */
        if (!model->B3SOIFDpvsdfbGiven)
            model->B3SOIFDpvsdfb = 0.0;
        if (!model->B3SOIFDpvsdthGiven)
            model->B3SOIFDpvsdth = 0.0;
/* Added for binning - END */

	if (!model->B3SOIFDcfGiven)
            model->B3SOIFDcf = 2.0 * EPSOX / PI
			   * log(1.0 + 0.4e-6 / model->B3SOIFDtox);
        if (!model->B3SOIFDcgdoGiven)
	{   if (model->B3SOIFDdlcGiven && (model->B3SOIFDdlc > 0.0))
	    {   model->B3SOIFDcgdo = model->B3SOIFDdlc * model->B3SOIFDcox
				 - model->B3SOIFDcgdl ;
	    }
	    else
	        model->B3SOIFDcgdo = 0.6 * model->B3SOIFDxj * model->B3SOIFDcox; 
	}
        if (!model->B3SOIFDcgsoGiven)
	{   if (model->B3SOIFDdlcGiven && (model->B3SOIFDdlc > 0.0))
	    {   model->B3SOIFDcgso = model->B3SOIFDdlc * model->B3SOIFDcox
				 - model->B3SOIFDcgsl ;
	    }
	    else
	        model->B3SOIFDcgso = 0.6 * model->B3SOIFDxj * model->B3SOIFDcox; 
	}

        if (!model->B3SOIFDcgeoGiven)
	{   model->B3SOIFDcgeo = 0.0;
	}
        if (!model->B3SOIFDxpartGiven)
            model->B3SOIFDxpart = 0.0;
        if (!model->B3SOIFDsheetResistanceGiven)
            model->B3SOIFDsheetResistance = 0.0;
        if (!model->B3SOIFDcsdeswGiven)
            model->B3SOIFDcsdesw = 0.0;
        if (!model->B3SOIFDunitLengthGateSidewallJctCapGiven)
            model->B3SOIFDunitLengthGateSidewallJctCap = 1e-10;
        if (!model->B3SOIFDGatesidewallJctPotentialGiven)
            model->B3SOIFDGatesidewallJctPotential = 0.7;
        if (!model->B3SOIFDbodyJctGateSideGradingCoeffGiven)
            model->B3SOIFDbodyJctGateSideGradingCoeff = 0.5;
        if (!model->B3SOIFDoxideTrapDensityAGiven)
	{   if (model->B3SOIFDtype == NMOS)
                model->B3SOIFDoxideTrapDensityA = 1e20;
            else
                model->B3SOIFDoxideTrapDensityA=9.9e18;
	}
        if (!model->B3SOIFDoxideTrapDensityBGiven)
	{   if (model->B3SOIFDtype == NMOS)
                model->B3SOIFDoxideTrapDensityB = 5e4;
            else
                model->B3SOIFDoxideTrapDensityB = 2.4e3;
	}
        if (!model->B3SOIFDoxideTrapDensityCGiven)
	{   if (model->B3SOIFDtype == NMOS)
                model->B3SOIFDoxideTrapDensityC = -1.4e-12;
            else
                model->B3SOIFDoxideTrapDensityC = 1.4e-12;

	}
        if (!model->B3SOIFDemGiven)
            model->B3SOIFDem = 4.1e7; /* V/m */
        if (!model->B3SOIFDefGiven)
            model->B3SOIFDef = 1.0;
        if (!model->B3SOIFDafGiven)
            model->B3SOIFDaf = 1.0;
        if (!model->B3SOIFDkfGiven)
            model->B3SOIFDkf = 0.0;
        if (!model->B3SOIFDnoifGiven)
            model->B3SOIFDnoif = 1.0;

        /* loop through all the instances of the model */
        for (here = B3SOIFDinstances(model); here != NULL ;
             here=B3SOIFDnextInstance(here)) 
	{	
            /* allocate a chunk of the state vector */
            here->B3SOIFDstates = *states;
            *states += B3SOIFDnumStates;
	    
	    /* perform the parameter defaulting */
            if (!here->B3SOIFDdrainAreaGiven)
                here->B3SOIFDdrainArea = 0.0;
            if (!here->B3SOIFDdrainPerimeterGiven)
                here->B3SOIFDdrainPerimeter = 0.0;
            if (!here->B3SOIFDdrainSquaresGiven)
                here->B3SOIFDdrainSquares = 1.0;
            if (!here->B3SOIFDicVBSGiven)
                here->B3SOIFDicVBS = 0;
            if (!here->B3SOIFDicVDSGiven)
                here->B3SOIFDicVDS = 0;
            if (!here->B3SOIFDicVGSGiven)
                here->B3SOIFDicVGS = 0;
            if (!here->B3SOIFDicVESGiven)
                here->B3SOIFDicVES = 0;
            if (!here->B3SOIFDicVPSGiven)
                here->B3SOIFDicVPS = 0;
	    if (!here->B3SOIFDbjtoffGiven)
		here->B3SOIFDbjtoff = 0;
	    if (!here->B3SOIFDdebugModGiven)
		here->B3SOIFDdebugMod = 0;
	    if (!here->B3SOIFDrth0Given)
		here->B3SOIFDrth0 = model->B3SOIFDrth0;
	    if (!here->B3SOIFDcth0Given)
		here->B3SOIFDcth0 = model->B3SOIFDcth0;
            if (!here->B3SOIFDbodySquaresGiven)
                here->B3SOIFDbodySquares = 1.0;
            if (!here->B3SOIFDlGiven)
                here->B3SOIFDl = 5e-6;
            if (!here->B3SOIFDsourceAreaGiven)
                here->B3SOIFDsourceArea = 0;
            if (!here->B3SOIFDsourcePerimeterGiven)
                here->B3SOIFDsourcePerimeter = 0;
            if (!here->B3SOIFDsourceSquaresGiven)
                here->B3SOIFDsourceSquares = 1;
            if (!here->B3SOIFDwGiven)
                here->B3SOIFDw = 5e-6;

            if (!here->B3SOIFDmGiven)
                here->B3SOIFDm = 1;

            if (!here->B3SOIFDoffGiven)
                here->B3SOIFDoff = 0;
 

            /* process drain series resistance */
            if ((model->B3SOIFDsheetResistance > 0.0) && 
                (here->B3SOIFDdrainSquares > 0.0 ) &&
                (here->B3SOIFDdNodePrime == 0))
	    {   error = CKTmkVolt(ckt,&tmp,here->B3SOIFDname,"drain");
                if(error) return(error);
                here->B3SOIFDdNodePrime = tmp->number;
		
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
	    {   here->B3SOIFDdNodePrime = here->B3SOIFDdNode;
            }
                   
            /* process source series resistance */
            if ((model->B3SOIFDsheetResistance > 0.0) && 
                (here->B3SOIFDsourceSquares > 0.0 ) &&
                (here->B3SOIFDsNodePrime == 0)) 
	    {   error = CKTmkVolt(ckt,&tmp,here->B3SOIFDname,"source");
                if(error) return(error);
                here->B3SOIFDsNodePrime = tmp->number;
		
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
	    {   here->B3SOIFDsNodePrime = here->B3SOIFDsNode;
            }

            /* process effective silicon film thickness */
            model->B3SOIFDcbox = 3.453133e-11 / model->B3SOIFDtbox;
            model->B3SOIFDcsi = 1.03594e-10 / model->B3SOIFDtsi;
            Cboxt = model->B3SOIFDcbox * model->B3SOIFDcsi / (model->B3SOIFDcbox + model->B3SOIFDcsi);
            model->B3SOIFDqsi = Charge_q*model->B3SOIFDnpeak*1e6*model->B3SOIFDtsi;
            /* Tsieff */
            tmp1 = 2.0 * EPSSI * model->B3SOIFDvbsa / Charge_q 
                 / (1e6*model->B3SOIFDnpeak);
            tmp2 = model->B3SOIFDtsi * model->B3SOIFDtsi;
            if (tmp2 < tmp1)
            {
               fprintf(stderr, "vbsa = %.3f is too large for this tsi = %.3e and is automatically set to zero\n", model->B3SOIFDvbsa, model->B3SOIFDtsi);
               model->B3SOIFDcsieff = model->B3SOIFDcsi;
               model->B3SOIFDqsieff = model->B3SOIFDqsi;
            }
            else
            {
               tmp1 = sqrt(model->B3SOIFDtsi * model->B3SOIFDtsi - 
                      2.0 * EPSSI * model->B3SOIFDvbsa / Charge_q / 
                      (1e6*model->B3SOIFDnpeak));
               model->B3SOIFDcsieff = 1.03594e-10 / tmp1;
               model->B3SOIFDqsieff = Charge_q*model->B3SOIFDnpeak*1e6*tmp1;
            }
            model->B3SOIFDcsit = 1/(1/model->B3SOIFDcox + 1/model->B3SOIFDcsieff);
            model->B3SOIFDcboxt = 1/(1/model->B3SOIFDcbox + 1/model->B3SOIFDcsieff);
            nfb0 = 1/(1 + model->B3SOIFDcbox / model->B3SOIFDcsit);
            model->B3SOIFDnfb = model->B3SOIFDkb3 * nfb0;
            model->B3SOIFDadice = model->B3SOIFDadice0  / ( 1 + Cboxt / model->B3SOIFDcox);

            here->B3SOIFDfloat = 0;

            here->B3SOIFDpNode = here->B3SOIFDpNodeExt;
            here->B3SOIFDbNode = here->B3SOIFDbNodeExt;
            here->B3SOIFDtempNode = here->B3SOIFDtempNodeExt;

	    if (here->B3SOIFDbNode == -1) 
            /* no internal body node is needed for SPICE iteration  */
            {   here->B3SOIFDbNode = here->B3SOIFDpNode = 0;
                here->B3SOIFDbodyMod = 0;
	    }
            else /* body tied */ 
            {   if ((model->B3SOIFDrbody == 0.0) && (model->B3SOIFDrbsh == 0.0))
                {  /* ideal body tie */
                   here->B3SOIFDbodyMod = 2;
                   /* pNode is not used in this case */
                }
                else { 
                   error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Body");
                   if(error) return(error);
                   here->B3SOIFDbodyMod = 1;
                   here->B3SOIFDpNode = here->B3SOIFDbNode;
                   here->B3SOIFDbNode = tmp->number;
                }
            }

            if ((model->B3SOIFDshMod == 1) && (here->B3SOIFDrth0!=0))
            {
	        error = CKTmkVolt(ckt,&tmp,here->B3SOIFDname,"Temp");
                if(error) return(error);
                   here->B3SOIFDtempNode = tmp->number;

            } else {
                here->B3SOIFDtempNode = 0;
            }

/* here for debugging purpose only */
            if ((here->B3SOIFDdebugMod > 1) || (here->B3SOIFDdebugMod == -1))
            {
               /* The real Vbs value */
               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Vbs");
               if(error) return(error);
               here->B3SOIFDvbsNode = tmp->number;   

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Ids");
               if(error) return(error);
               here->B3SOIFDidsNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Ic");
               if(error) return(error);
               here->B3SOIFDicNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Ibs");
               if(error) return(error);
               here->B3SOIFDibsNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Ibd");
               if(error) return(error);
               here->B3SOIFDibdNode = tmp->number;
       
               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Iii");
               if(error) return(error);
               here->B3SOIFDiiiNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Igidl");
               if(error) return(error);
               here->B3SOIFDigidlNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Itun");
               if(error) return(error);
               here->B3SOIFDitunNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Ibp");
               if(error) return(error);
               here->B3SOIFDibpNode = tmp->number;
       
               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Abeff");
               if(error) return(error);
               here->B3SOIFDabeffNode = tmp->number;
       
               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Vbs0eff");
               if(error) return(error);
               here->B3SOIFDvbs0effNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Vbseff");
               if(error) return(error);
               here->B3SOIFDvbseffNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Xc");
               if(error) return(error);
               here->B3SOIFDxcNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Cbb");
               if(error) return(error);
               here->B3SOIFDcbbNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Cbd");
               if(error) return(error);
               here->B3SOIFDcbdNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Cbg");
               if(error) return(error);
               here->B3SOIFDcbgNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Qbody");
               if(error) return(error);
               here->B3SOIFDqbNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Qbf");
               if(error) return(error);
               here->B3SOIFDqbfNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Qjs");
               if(error) return(error);
               here->B3SOIFDqjsNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Qjd");
               if(error) return(error);
               here->B3SOIFDqjdNode = tmp->number;

               /* clean up last */
               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Gm");
               if(error) return(error);
               here->B3SOIFDgmNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Gmbs");
               if(error) return(error);
               here->B3SOIFDgmbsNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Gds");
               if(error) return(error);
               here->B3SOIFDgdsNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Gme");
               if(error) return(error);
               here->B3SOIFDgmeNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Vbs0teff");
               if(error) return(error);
               here->B3SOIFDvbs0teffNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Vth");
               if(error) return(error);
               here->B3SOIFDvthNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Vgsteff");
               if(error) return(error);
               here->B3SOIFDvgsteffNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Xcsat");
               if(error) return(error);
               here->B3SOIFDxcsatNode = tmp->number;
       
               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Qac0");
               if(error) return(error);
               here->B3SOIFDqaccNode = tmp->number;
       
               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Qsub0");
               if(error) return(error);
               here->B3SOIFDqsub0Node = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Qsubs1");
               if(error) return(error);
               here->B3SOIFDqsubs1Node = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Qsubs2");
               if(error) return(error);
               here->B3SOIFDqsubs2Node = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Qsub");
               if(error) return(error);
               here->B3SOIFDqeNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Qdrn");
               if(error) return(error);
               here->B3SOIFDqdNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Qgate");
               if(error) return(error);
               here->B3SOIFDqgNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Vdscv");
               if(error) return(error);
               here->B3SOIFDvdscvNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Vcscv");
               if(error) return(error);
               here->B3SOIFDvcscvNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Cbe");
               if(error) return(error);
               here->B3SOIFDcbeNode = tmp->number;
       
               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Dum1");
               if(error) return(error);
               here->B3SOIFDdum1Node = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Dum2");
               if(error) return(error);
               here->B3SOIFDdum2Node = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Dum3");
               if(error) return(error);
               here->B3SOIFDdum3Node = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Dum4");
               if(error) return(error);
               here->B3SOIFDdum4Node = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIFDname, "Dum5");
               if(error) return(error);
               here->B3SOIFDdum5Node = tmp->number;
           }

        /* set Sparse Matrix Pointers */

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)


        if ((model->B3SOIFDshMod == 1) && (here->B3SOIFDrth0!=0.0)) {
            TSTALLOC(B3SOIFDTemptempPtr, B3SOIFDtempNode, B3SOIFDtempNode);
            TSTALLOC(B3SOIFDTempdpPtr, B3SOIFDtempNode, B3SOIFDdNodePrime);
            TSTALLOC(B3SOIFDTempspPtr, B3SOIFDtempNode, B3SOIFDsNodePrime);
            TSTALLOC(B3SOIFDTempgPtr, B3SOIFDtempNode, B3SOIFDgNode);
            TSTALLOC(B3SOIFDTempbPtr, B3SOIFDtempNode, B3SOIFDbNode);
            TSTALLOC(B3SOIFDTempePtr, B3SOIFDtempNode, B3SOIFDeNode);

            TSTALLOC(B3SOIFDGtempPtr, B3SOIFDgNode, B3SOIFDtempNode);
            TSTALLOC(B3SOIFDDPtempPtr, B3SOIFDdNodePrime, B3SOIFDtempNode);
            TSTALLOC(B3SOIFDSPtempPtr, B3SOIFDsNodePrime, B3SOIFDtempNode);
            TSTALLOC(B3SOIFDEtempPtr, B3SOIFDeNode, B3SOIFDtempNode);
            TSTALLOC(B3SOIFDBtempPtr, B3SOIFDbNode, B3SOIFDtempNode);

            if (here->B3SOIFDbodyMod == 1) {
                TSTALLOC(B3SOIFDPtempPtr, B3SOIFDpNode, B3SOIFDtempNode);
            }
        }
        if (here->B3SOIFDbodyMod == 2) {
            /* Don't create any Jacobian entry for pNode */
        }
        else if (here->B3SOIFDbodyMod == 1) {
            TSTALLOC(B3SOIFDBpPtr, B3SOIFDbNode, B3SOIFDpNode);
            TSTALLOC(B3SOIFDPbPtr, B3SOIFDpNode, B3SOIFDbNode);
            TSTALLOC(B3SOIFDPpPtr, B3SOIFDpNode, B3SOIFDpNode);
            TSTALLOC(B3SOIFDPgPtr, B3SOIFDpNode, B3SOIFDgNode);
            TSTALLOC(B3SOIFDPdpPtr, B3SOIFDpNode, B3SOIFDdNodePrime);
            TSTALLOC(B3SOIFDPspPtr, B3SOIFDpNode, B3SOIFDsNodePrime);
            TSTALLOC(B3SOIFDPePtr, B3SOIFDpNode, B3SOIFDeNode);
        }

        TSTALLOC(B3SOIFDEgPtr, B3SOIFDeNode, B3SOIFDgNode);
        TSTALLOC(B3SOIFDEdpPtr, B3SOIFDeNode, B3SOIFDdNodePrime);
        TSTALLOC(B3SOIFDEspPtr, B3SOIFDeNode, B3SOIFDsNodePrime);
        TSTALLOC(B3SOIFDGePtr, B3SOIFDgNode, B3SOIFDeNode);
        TSTALLOC(B3SOIFDDPePtr, B3SOIFDdNodePrime, B3SOIFDeNode);
        TSTALLOC(B3SOIFDSPePtr, B3SOIFDsNodePrime, B3SOIFDeNode);

        TSTALLOC(B3SOIFDEbPtr, B3SOIFDeNode, B3SOIFDbNode);
        TSTALLOC(B3SOIFDEePtr, B3SOIFDeNode, B3SOIFDeNode);

        TSTALLOC(B3SOIFDGgPtr, B3SOIFDgNode, B3SOIFDgNode);
        TSTALLOC(B3SOIFDGdpPtr, B3SOIFDgNode, B3SOIFDdNodePrime);
        TSTALLOC(B3SOIFDGspPtr, B3SOIFDgNode, B3SOIFDsNodePrime);

        TSTALLOC(B3SOIFDDPgPtr, B3SOIFDdNodePrime, B3SOIFDgNode);
        TSTALLOC(B3SOIFDDPdpPtr, B3SOIFDdNodePrime, B3SOIFDdNodePrime);
        TSTALLOC(B3SOIFDDPspPtr, B3SOIFDdNodePrime, B3SOIFDsNodePrime);
        TSTALLOC(B3SOIFDDPdPtr, B3SOIFDdNodePrime, B3SOIFDdNode);

        TSTALLOC(B3SOIFDSPgPtr, B3SOIFDsNodePrime, B3SOIFDgNode);
        TSTALLOC(B3SOIFDSPdpPtr, B3SOIFDsNodePrime, B3SOIFDdNodePrime);
        TSTALLOC(B3SOIFDSPspPtr, B3SOIFDsNodePrime, B3SOIFDsNodePrime);
        TSTALLOC(B3SOIFDSPsPtr, B3SOIFDsNodePrime, B3SOIFDsNode);

        TSTALLOC(B3SOIFDDdPtr, B3SOIFDdNode, B3SOIFDdNode);
        TSTALLOC(B3SOIFDDdpPtr, B3SOIFDdNode, B3SOIFDdNodePrime);

        TSTALLOC(B3SOIFDSsPtr, B3SOIFDsNode, B3SOIFDsNode);
        TSTALLOC(B3SOIFDSspPtr, B3SOIFDsNode, B3SOIFDsNodePrime);

/* here for debugging purpose only */
         if ((here->B3SOIFDdebugMod > 1) || (here->B3SOIFDdebugMod == -1))
         {
            TSTALLOC(B3SOIFDVbsPtr, B3SOIFDvbsNode, B3SOIFDvbsNode) ;
            TSTALLOC(B3SOIFDIdsPtr, B3SOIFDidsNode, B3SOIFDidsNode);
            TSTALLOC(B3SOIFDIcPtr, B3SOIFDicNode, B3SOIFDicNode);
            TSTALLOC(B3SOIFDIbsPtr, B3SOIFDibsNode, B3SOIFDibsNode);
            TSTALLOC(B3SOIFDIbdPtr, B3SOIFDibdNode, B3SOIFDibdNode);
            TSTALLOC(B3SOIFDIiiPtr, B3SOIFDiiiNode, B3SOIFDiiiNode);
            TSTALLOC(B3SOIFDIgidlPtr, B3SOIFDigidlNode, B3SOIFDigidlNode);
            TSTALLOC(B3SOIFDItunPtr, B3SOIFDitunNode, B3SOIFDitunNode);
            TSTALLOC(B3SOIFDIbpPtr, B3SOIFDibpNode, B3SOIFDibpNode);
            TSTALLOC(B3SOIFDAbeffPtr, B3SOIFDabeffNode, B3SOIFDabeffNode);
            TSTALLOC(B3SOIFDVbs0effPtr, B3SOIFDvbs0effNode, B3SOIFDvbs0effNode);
            TSTALLOC(B3SOIFDVbseffPtr, B3SOIFDvbseffNode, B3SOIFDvbseffNode);
            TSTALLOC(B3SOIFDXcPtr, B3SOIFDxcNode, B3SOIFDxcNode);
            TSTALLOC(B3SOIFDCbbPtr, B3SOIFDcbbNode, B3SOIFDcbbNode);
            TSTALLOC(B3SOIFDCbdPtr, B3SOIFDcbdNode, B3SOIFDcbdNode);
            TSTALLOC(B3SOIFDCbgPtr, B3SOIFDcbgNode, B3SOIFDcbgNode);
            TSTALLOC(B3SOIFDqbPtr, B3SOIFDqbNode, B3SOIFDqbNode);
            TSTALLOC(B3SOIFDQbfPtr, B3SOIFDqbfNode, B3SOIFDqbfNode);
            TSTALLOC(B3SOIFDQjsPtr, B3SOIFDqjsNode, B3SOIFDqjsNode);
            TSTALLOC(B3SOIFDQjdPtr, B3SOIFDqjdNode, B3SOIFDqjdNode);

            /* clean up last */
            TSTALLOC(B3SOIFDGmPtr, B3SOIFDgmNode, B3SOIFDgmNode);
            TSTALLOC(B3SOIFDGmbsPtr, B3SOIFDgmbsNode, B3SOIFDgmbsNode);
            TSTALLOC(B3SOIFDGdsPtr, B3SOIFDgdsNode, B3SOIFDgdsNode);
            TSTALLOC(B3SOIFDGmePtr, B3SOIFDgmeNode, B3SOIFDgmeNode);
            TSTALLOC(B3SOIFDVbs0teffPtr, B3SOIFDvbs0teffNode, B3SOIFDvbs0teffNode);
            TSTALLOC(B3SOIFDVthPtr, B3SOIFDvthNode, B3SOIFDvthNode);
            TSTALLOC(B3SOIFDVgsteffPtr, B3SOIFDvgsteffNode, B3SOIFDvgsteffNode);
            TSTALLOC(B3SOIFDXcsatPtr, B3SOIFDxcsatNode, B3SOIFDxcsatNode);
            TSTALLOC(B3SOIFDVcscvPtr, B3SOIFDvcscvNode, B3SOIFDvcscvNode);
            TSTALLOC(B3SOIFDVdscvPtr, B3SOIFDvdscvNode, B3SOIFDvdscvNode);
            TSTALLOC(B3SOIFDCbePtr, B3SOIFDcbeNode, B3SOIFDcbeNode);
            TSTALLOC(B3SOIFDDum1Ptr, B3SOIFDdum1Node, B3SOIFDdum1Node);
            TSTALLOC(B3SOIFDDum2Ptr, B3SOIFDdum2Node, B3SOIFDdum2Node);
            TSTALLOC(B3SOIFDDum3Ptr, B3SOIFDdum3Node, B3SOIFDdum3Node);
            TSTALLOC(B3SOIFDDum4Ptr, B3SOIFDdum4Node, B3SOIFDdum4Node);
            TSTALLOC(B3SOIFDDum5Ptr, B3SOIFDdum5Node, B3SOIFDdum5Node);
            TSTALLOC(B3SOIFDQaccPtr, B3SOIFDqaccNode, B3SOIFDqaccNode);
            TSTALLOC(B3SOIFDQsub0Ptr, B3SOIFDqsub0Node, B3SOIFDqsub0Node);
            TSTALLOC(B3SOIFDQsubs1Ptr, B3SOIFDqsubs1Node, B3SOIFDqsubs1Node);
            TSTALLOC(B3SOIFDQsubs2Ptr, B3SOIFDqsubs2Node, B3SOIFDqsubs2Node);
            TSTALLOC(B3SOIFDqePtr, B3SOIFDqeNode, B3SOIFDqeNode);
            TSTALLOC(B3SOIFDqdPtr, B3SOIFDqdNode, B3SOIFDqdNode);
            TSTALLOC(B3SOIFDqgPtr, B3SOIFDqgNode, B3SOIFDqgNode);
         }

        }
    }
    return(OK);
}  

int
B3SOIFDunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    B3SOIFDmodel *model;
    B3SOIFDinstance *here;
 
    for (model = (B3SOIFDmodel *)inModel; model != NULL;
            model = B3SOIFDnextModel(model))
    {
        for (here = B3SOIFDinstances(model); here != NULL;
                here=B3SOIFDnextInstance(here))
        {
            /* here for debugging purpose only */
            if (here->B3SOIFDdum5Node > 0)
                CKTdltNNum(ckt, here->B3SOIFDdum5Node);
            here->B3SOIFDdum5Node = 0;

            if (here->B3SOIFDdum4Node > 0)
                CKTdltNNum(ckt, here->B3SOIFDdum4Node);
            here->B3SOIFDdum4Node = 0;

            if (here->B3SOIFDdum3Node > 0)
                CKTdltNNum(ckt, here->B3SOIFDdum3Node);
            here->B3SOIFDdum3Node = 0;

            if (here->B3SOIFDdum2Node > 0)
                CKTdltNNum(ckt, here->B3SOIFDdum2Node);
            here->B3SOIFDdum2Node = 0;

            if (here->B3SOIFDdum1Node > 0)
                CKTdltNNum(ckt, here->B3SOIFDdum1Node);
            here->B3SOIFDdum1Node = 0;

            if (here->B3SOIFDcbeNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDcbeNode);
            here->B3SOIFDcbeNode = 0;

            if (here->B3SOIFDvcscvNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDvcscvNode);
            here->B3SOIFDvcscvNode = 0;

            if (here->B3SOIFDvdscvNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDvdscvNode);
            here->B3SOIFDvdscvNode = 0;

            if (here->B3SOIFDqgNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDqgNode);
            here->B3SOIFDqgNode = 0;

            if (here->B3SOIFDqdNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDqdNode);
            here->B3SOIFDqdNode = 0;

            if (here->B3SOIFDqeNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDqeNode);
            here->B3SOIFDqeNode = 0;

            if (here->B3SOIFDqsubs2Node > 0)
                CKTdltNNum(ckt, here->B3SOIFDqsubs2Node);
            here->B3SOIFDqsubs2Node = 0;

            if (here->B3SOIFDqsubs1Node > 0)
                CKTdltNNum(ckt, here->B3SOIFDqsubs1Node);
            here->B3SOIFDqsubs1Node = 0;

            if (here->B3SOIFDqsub0Node > 0)
                CKTdltNNum(ckt, here->B3SOIFDqsub0Node);
            here->B3SOIFDqsub0Node = 0;

            if (here->B3SOIFDqaccNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDqaccNode);
            here->B3SOIFDqaccNode = 0;

            if (here->B3SOIFDxcsatNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDxcsatNode);
            here->B3SOIFDxcsatNode = 0;

            if (here->B3SOIFDvgsteffNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDvgsteffNode);
            here->B3SOIFDvgsteffNode = 0;

            if (here->B3SOIFDvthNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDvthNode);
            here->B3SOIFDvthNode = 0;

            if (here->B3SOIFDvbs0teffNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDvbs0teffNode);
            here->B3SOIFDvbs0teffNode = 0;

            if (here->B3SOIFDgmeNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDgmeNode);
            here->B3SOIFDgmeNode = 0;

            if (here->B3SOIFDgdsNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDgdsNode);
            here->B3SOIFDgdsNode = 0;

            if (here->B3SOIFDgmbsNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDgmbsNode);
            here->B3SOIFDgmbsNode = 0;

            if (here->B3SOIFDgmNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDgmNode);
            here->B3SOIFDgmNode = 0;

            if (here->B3SOIFDqjdNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDqjdNode);
            here->B3SOIFDqjdNode = 0;

            if (here->B3SOIFDqjsNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDqjsNode);
            here->B3SOIFDqjsNode = 0;

            if (here->B3SOIFDqbfNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDqbfNode);
            here->B3SOIFDqbfNode = 0;

            if (here->B3SOIFDqbNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDqbNode);
            here->B3SOIFDqbNode = 0;

            if (here->B3SOIFDcbgNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDcbgNode);
            here->B3SOIFDcbgNode = 0;

            if (here->B3SOIFDcbdNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDcbdNode);
            here->B3SOIFDcbdNode = 0;

            if (here->B3SOIFDcbbNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDcbbNode);
            here->B3SOIFDcbbNode = 0;

            if (here->B3SOIFDxcNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDxcNode);
            here->B3SOIFDxcNode = 0;

            if (here->B3SOIFDvbseffNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDvbseffNode);
            here->B3SOIFDvbseffNode = 0;

            if (here->B3SOIFDvbs0effNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDvbs0effNode);
            here->B3SOIFDvbs0effNode = 0;

            if (here->B3SOIFDabeffNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDabeffNode);
            here->B3SOIFDabeffNode = 0;

            if (here->B3SOIFDibpNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDibpNode);
            here->B3SOIFDibpNode = 0;

            if (here->B3SOIFDitunNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDitunNode);
            here->B3SOIFDitunNode = 0;

            if (here->B3SOIFDigidlNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDigidlNode);
            here->B3SOIFDigidlNode = 0;

            if (here->B3SOIFDiiiNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDiiiNode);
            here->B3SOIFDiiiNode = 0;

            if (here->B3SOIFDibdNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDibdNode);
            here->B3SOIFDibdNode = 0;

            if (here->B3SOIFDibsNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDibsNode);
            here->B3SOIFDibsNode = 0;

            if (here->B3SOIFDicNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDicNode);
            here->B3SOIFDicNode = 0;

            if (here->B3SOIFDidsNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDidsNode);
            here->B3SOIFDidsNode = 0;

            if (here->B3SOIFDvbsNode > 0)
                CKTdltNNum(ckt, here->B3SOIFDvbsNode);
            here->B3SOIFDvbsNode = 0;


            if (here->B3SOIFDtempNode > 0 &&
                here->B3SOIFDtempNode != here->B3SOIFDtempNodeExt &&
                here->B3SOIFDtempNode != here->B3SOIFDbNodeExt &&
                here->B3SOIFDtempNode != here->B3SOIFDpNodeExt)
                CKTdltNNum(ckt, here->B3SOIFDtempNode);
            here->B3SOIFDtempNode = 0;

            if (here->B3SOIFDbNode > 0 &&
                here->B3SOIFDbNode != here->B3SOIFDbNodeExt &&
                here->B3SOIFDbNode != here->B3SOIFDpNodeExt)
                CKTdltNNum(ckt, here->B3SOIFDbNode);
            here->B3SOIFDbNode = 0;

            here->B3SOIFDpNode = 0;

            if (here->B3SOIFDsNodePrime > 0
                    && here->B3SOIFDsNodePrime != here->B3SOIFDsNode)
                CKTdltNNum(ckt, here->B3SOIFDsNodePrime);
            here->B3SOIFDsNodePrime = 0;

            if (here->B3SOIFDdNodePrime > 0
                    && here->B3SOIFDdNodePrime != here->B3SOIFDdNode)
                CKTdltNNum(ckt, here->B3SOIFDdNodePrime);
            here->B3SOIFDdNodePrime = 0;
        }
    }
    return OK;
}

