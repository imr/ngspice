/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soipdset.c          98/5/01
Modified by Pin Su and Jan Feng	99/2/15
Modified by Pin Su 99/4/30
Modified by Pin Su, Wei Jin 99/9/27
Modified by Pin Su 00/3/1
Modified by Pin Su 01/2/15
Modified by Pin Su 02/3/5
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.2.3  02/3/5  Pin Su 
 * BSIMPD2.2.3 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "b3soipddef.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#define SMOOTHFACTOR 0.1
#define EPSOX 3.453133e-11
#define EPSSI 1.03594e-10
#define PI 3.141592654
#define Charge_q 1.60219e-19
#define Meter2Micron 1.0e6

int
B3SOIPDsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt,
             int *states)
{
B3SOIPDmodel *model = (B3SOIPDmodel*)inModel;
B3SOIPDinstance *here;
int error;
CKTnode *tmp;

double Cboxt;

CKTnode *tmpNode;
IFuid tmpName;


    /*  loop through all the B3SOIPD device models */
    for( ; model != NULL; model = B3SOIPDnextModel(model))
    {
/* Default value Processing for B3SOIPD MOSFET Models */

        if (!model->B3SOIPDtypeGiven)
            model->B3SOIPDtype = NMOS;     
        if (!model->B3SOIPDmobModGiven) 
            model->B3SOIPDmobMod = 1;
        if (!model->B3SOIPDbinUnitGiven) 
            model->B3SOIPDbinUnit = 1;
        if (!model->B3SOIPDparamChkGiven) 
            model->B3SOIPDparamChk = 0;
        if (!model->B3SOIPDcapModGiven) 
            model->B3SOIPDcapMod = 2;
        if (!model->B3SOIPDnoiModGiven) 
            model->B3SOIPDnoiMod = 1;
        if (!model->B3SOIPDshModGiven) 
            model->B3SOIPDshMod = 0;
        if (!model->B3SOIPDversionGiven) 
            model->B3SOIPDversion = 2.0;
        if (!model->B3SOIPDtoxGiven)
            model->B3SOIPDtox = 100.0e-10;
        model->B3SOIPDcox = 3.453133e-11 / model->B3SOIPDtox;

/* v2.2.3 */
        if (!model->B3SOIPDdtoxcvGiven)
            model->B3SOIPDdtoxcv = 0.0;

        if (!model->B3SOIPDcdscGiven)
	    model->B3SOIPDcdsc = 2.4e-4;   /* unit Q/V/m^2  */
        if (!model->B3SOIPDcdscbGiven)
	    model->B3SOIPDcdscb = 0.0;   /* unit Q/V/m^2  */    
        if (!model->B3SOIPDcdscdGiven)
	    model->B3SOIPDcdscd = 0.0;   /* unit Q/V/m^2  */
        if (!model->B3SOIPDcitGiven)
	    model->B3SOIPDcit = 0.0;   /* unit Q/V/m^2  */
        if (!model->B3SOIPDnfactorGiven)
	    model->B3SOIPDnfactor = 1;
        if (!model->B3SOIPDvsatGiven)
            model->B3SOIPDvsat = 8.0e4;    /* unit m/s */ 
        if (!model->B3SOIPDatGiven)
            model->B3SOIPDat = 3.3e4;    /* unit m/s */ 
        if (!model->B3SOIPDa0Given)
            model->B3SOIPDa0 = 1.0;  
        if (!model->B3SOIPDagsGiven)
            model->B3SOIPDags = 0.0;
        if (!model->B3SOIPDa1Given)
            model->B3SOIPDa1 = 0.0;
        if (!model->B3SOIPDa2Given)
            model->B3SOIPDa2 = 1.0;
        if (!model->B3SOIPDketaGiven)
            model->B3SOIPDketa = -0.6;    /* unit  / V */
        if (!model->B3SOIPDnsubGiven)
            model->B3SOIPDnsub = 6.0e16;   /* unit 1/cm3 */
        if (!model->B3SOIPDnpeakGiven)
            model->B3SOIPDnpeak = 1.7e17;   /* unit 1/cm3 */
        if (!model->B3SOIPDngateGiven)
            model->B3SOIPDngate = 0;   /* unit 1/cm3 */
        if (!model->B3SOIPDvbmGiven)
	    model->B3SOIPDvbm = -3.0;
        if (!model->B3SOIPDxtGiven)
	    model->B3SOIPDxt = 1.55e-7;
        if (!model->B3SOIPDkt1Given)
            model->B3SOIPDkt1 = -0.11;      /* unit V */
        if (!model->B3SOIPDkt1lGiven)
            model->B3SOIPDkt1l = 0.0;      /* unit V*m */
        if (!model->B3SOIPDkt2Given)
            model->B3SOIPDkt2 = 0.022;      /* No unit */
        if (!model->B3SOIPDk3Given)
            model->B3SOIPDk3 = 0.0;      
        if (!model->B3SOIPDk3bGiven)
            model->B3SOIPDk3b = 0.0;      
        if (!model->B3SOIPDw0Given)
            model->B3SOIPDw0 = 2.5e-6;    
        if (!model->B3SOIPDnlxGiven)
            model->B3SOIPDnlx = 1.74e-7;     
        if (!model->B3SOIPDdvt0Given)
            model->B3SOIPDdvt0 = 2.2;    
        if (!model->B3SOIPDdvt1Given)
            model->B3SOIPDdvt1 = 0.53;      
        if (!model->B3SOIPDdvt2Given)
            model->B3SOIPDdvt2 = -0.032;   /* unit 1 / V */     

        if (!model->B3SOIPDdvt0wGiven)
            model->B3SOIPDdvt0w = 0.0;    
        if (!model->B3SOIPDdvt1wGiven)
            model->B3SOIPDdvt1w = 5.3e6;    
        if (!model->B3SOIPDdvt2wGiven)
            model->B3SOIPDdvt2w = -0.032;   

        if (!model->B3SOIPDdroutGiven)
            model->B3SOIPDdrout = 0.56;     
        if (!model->B3SOIPDdsubGiven)
            model->B3SOIPDdsub = model->B3SOIPDdrout;     
        if (!model->B3SOIPDvth0Given)
            model->B3SOIPDvth0 = (model->B3SOIPDtype == NMOS) ? 0.7 : -0.7;
        if (!model->B3SOIPDuaGiven)
            model->B3SOIPDua = 2.25e-9;      /* unit m/V */
        if (!model->B3SOIPDua1Given)
            model->B3SOIPDua1 = 4.31e-9;      /* unit m/V */
        if (!model->B3SOIPDubGiven)
            model->B3SOIPDub = 5.87e-19;     /* unit (m/V)**2 */
        if (!model->B3SOIPDub1Given)
            model->B3SOIPDub1 = -7.61e-18;     /* unit (m/V)**2 */
        if (!model->B3SOIPDucGiven)
            model->B3SOIPDuc = (model->B3SOIPDmobMod == 3) ? -0.0465 : -0.0465e-9;   
        if (!model->B3SOIPDuc1Given)
            model->B3SOIPDuc1 = (model->B3SOIPDmobMod == 3) ? -0.056 : -0.056e-9;   
        if (!model->B3SOIPDu0Given)
            model->B3SOIPDu0 = (model->B3SOIPDtype == NMOS) ? 0.067 : 0.025;
        if (!model->B3SOIPDuteGiven)
	    model->B3SOIPDute = -1.5;    
        if (!model->B3SOIPDvoffGiven)
	    model->B3SOIPDvoff = -0.08;
        if (!model->B3SOIPDdeltaGiven)  
           model->B3SOIPDdelta = 0.01;
        if (!model->B3SOIPDrdswGiven)
            model->B3SOIPDrdsw = 100;
        if (!model->B3SOIPDprwgGiven)
            model->B3SOIPDprwg = 0.0;      /* unit 1/V */
        if (!model->B3SOIPDprwbGiven)
            model->B3SOIPDprwb = 0.0;      
        if (!model->B3SOIPDprtGiven)
            model->B3SOIPDprt = 0.0;      
        if (!model->B3SOIPDeta0Given)
            model->B3SOIPDeta0 = 0.08;      /* no unit  */ 
        if (!model->B3SOIPDetabGiven)
            model->B3SOIPDetab = -0.07;      /* unit  1/V */ 
        if (!model->B3SOIPDpclmGiven)
            model->B3SOIPDpclm = 1.3;      /* no unit  */ 
        if (!model->B3SOIPDpdibl1Given)
            model->B3SOIPDpdibl1 = .39;    /* no unit  */
        if (!model->B3SOIPDpdibl2Given)
            model->B3SOIPDpdibl2 = 0.0086;    /* no unit  */ 
        if (!model->B3SOIPDpdiblbGiven)
            model->B3SOIPDpdiblb = 0.0;    /* 1/V  */ 
        if (!model->B3SOIPDpvagGiven)
            model->B3SOIPDpvag = 0.0;     
        if (!model->B3SOIPDwrGiven)  
            model->B3SOIPDwr = 1.0;
        if (!model->B3SOIPDdwgGiven)  
            model->B3SOIPDdwg = 0.0;
        if (!model->B3SOIPDdwbGiven)  
            model->B3SOIPDdwb = 0.0;
        if (!model->B3SOIPDb0Given)
            model->B3SOIPDb0 = 0.0;
        if (!model->B3SOIPDb1Given)  
            model->B3SOIPDb1 = 0.0;
        if (!model->B3SOIPDalpha0Given)  
            model->B3SOIPDalpha0 = 0.0;

        if (!model->B3SOIPDcgslGiven)  
            model->B3SOIPDcgsl = 0.0;
        if (!model->B3SOIPDcgdlGiven)  
            model->B3SOIPDcgdl = 0.0;
        if (!model->B3SOIPDckappaGiven)  
            model->B3SOIPDckappa = 0.6;
        if (!model->B3SOIPDclcGiven)  
            model->B3SOIPDclc = 0.1e-7;
        if (!model->B3SOIPDcleGiven)  
            model->B3SOIPDcle = 0.0;
        if (!model->B3SOIPDtboxGiven)  
            model->B3SOIPDtbox = 3e-7;
        if (!model->B3SOIPDtsiGiven)  
            model->B3SOIPDtsi = 1e-7;
        if (!model->B3SOIPDxjGiven)  
            model->B3SOIPDxj = model->B3SOIPDtsi;
        if (!model->B3SOIPDrbodyGiven)  
            model->B3SOIPDrbody = 0.0;
        if (!model->B3SOIPDrbshGiven)  
            model->B3SOIPDrbsh = 0.0;
        if (!model->B3SOIPDrth0Given)  
            model->B3SOIPDrth0 = 0;

        if (!model->B3SOIPDcth0Given)
            model->B3SOIPDcth0 = 0;

        if (!model->B3SOIPDagidlGiven)  
            model->B3SOIPDagidl = 0.0;
        if (!model->B3SOIPDbgidlGiven)  
            model->B3SOIPDbgidl = 0.0;
        if (!model->B3SOIPDngidlGiven)  
            model->B3SOIPDngidl = 1.2;
        if (!model->B3SOIPDndiodeGiven)  
            model->B3SOIPDndiode = 1.0;
        if (!model->B3SOIPDntunGiven)  
            model->B3SOIPDntun = 10.0;

        if (!model->B3SOIPDnrecf0Given)
            model->B3SOIPDnrecf0 = 2.0;
        if (!model->B3SOIPDnrecr0Given)
            model->B3SOIPDnrecr0 = 10.0;
 
        if (!model->B3SOIPDisbjtGiven)  
            model->B3SOIPDisbjt = 1e-6;
        if (!model->B3SOIPDisdifGiven)  
            model->B3SOIPDisdif = 0.0;
        if (!model->B3SOIPDisrecGiven)  
            model->B3SOIPDisrec = 1e-5;
        if (!model->B3SOIPDistunGiven)  
            model->B3SOIPDistun = 0.0;
        if (!model->B3SOIPDxbjtGiven)  
            model->B3SOIPDxbjt = 1;
/*
        if (!model->B3SOIPDxdifGiven)  
            model->B3SOIPDxdif = 1;
*/
        if (!model->B3SOIPDxdifGiven)
            model->B3SOIPDxdif = model->B3SOIPDxbjt;

        if (!model->B3SOIPDxrecGiven)  
            model->B3SOIPDxrec = 1;
        if (!model->B3SOIPDxtunGiven)  
            model->B3SOIPDxtun = 0;
        if (!model->B3SOIPDttGiven)  
            model->B3SOIPDtt = 1e-12;
        if (!model->B3SOIPDasdGiven)  
            model->B3SOIPDasd = 0.3;

        /* unit degree celcius */
        if (!model->B3SOIPDtnomGiven)  
	    model->B3SOIPDtnom = ckt->CKTnomTemp; 
        if (!model->B3SOIPDLintGiven)  
           model->B3SOIPDLint = 0.0;
        if (!model->B3SOIPDLlGiven)  
           model->B3SOIPDLl = 0.0;
        if (!model->B3SOIPDLlcGiven) 
           model->B3SOIPDLlc = 0.0; /* v2.2.3 */
        if (!model->B3SOIPDLlnGiven)  
           model->B3SOIPDLln = 1.0;
        if (!model->B3SOIPDLwGiven)  
           model->B3SOIPDLw = 0.0;
        if (!model->B3SOIPDLwcGiven) 
           model->B3SOIPDLwc = 0.0; /* v2.2.3 */
        if (!model->B3SOIPDLwnGiven)  
           model->B3SOIPDLwn = 1.0;
        if (!model->B3SOIPDLwlGiven)  
           model->B3SOIPDLwl = 0.0;
        if (!model->B3SOIPDLwlcGiven) 
           model->B3SOIPDLwlc = 0.0; /* v2.2.3 */
        if (!model->B3SOIPDLminGiven)  
           model->B3SOIPDLmin = 0.0;
        if (!model->B3SOIPDLmaxGiven)  
           model->B3SOIPDLmax = 1.0;
        if (!model->B3SOIPDWintGiven)  
           model->B3SOIPDWint = 0.0;
        if (!model->B3SOIPDWlGiven)  
           model->B3SOIPDWl = 0.0;
        if (!model->B3SOIPDWlcGiven) 
           model->B3SOIPDWlc = 0.0; /* v2.2.3 */
        if (!model->B3SOIPDWlnGiven)  
           model->B3SOIPDWln = 1.0;
        if (!model->B3SOIPDWwGiven)  
           model->B3SOIPDWw = 0.0;
        if (!model->B3SOIPDWwcGiven) 
           model->B3SOIPDWwc = 0.0; /* v2.2.3 */
        if (!model->B3SOIPDWwnGiven)  
           model->B3SOIPDWwn = 1.0;
        if (!model->B3SOIPDWwlGiven)  
           model->B3SOIPDWwl = 0.0;
        if (!model->B3SOIPDWwlcGiven)
           model->B3SOIPDWwlc = 0.0; /* v2.2.3 */
        if (!model->B3SOIPDWminGiven)  
           model->B3SOIPDWmin = 0.0;
        if (!model->B3SOIPDWmaxGiven)  
           model->B3SOIPDWmax = 1.0;
        if (!model->B3SOIPDdwcGiven)  
           model->B3SOIPDdwc = model->B3SOIPDWint;
        if (!model->B3SOIPDdlcGiven)  
           model->B3SOIPDdlc = model->B3SOIPDLint;


/* v2.2 release */
        if (!model->B3SOIPDwth0Given)
           model->B3SOIPDwth0 = 0.0;
        if (!model->B3SOIPDrhaloGiven)
           model->B3SOIPDrhalo = 1e15;
        if (!model->B3SOIPDntoxGiven)
           model->B3SOIPDntox = 1;
        if (!model->B3SOIPDtoxrefGiven)
           model->B3SOIPDtoxref = 2.5e-9;
        if (!model->B3SOIPDebgGiven)
           model->B3SOIPDebg = 1.2;
        if (!model->B3SOIPDvevbGiven)
           model->B3SOIPDvevb = 0.075;
        if (!model->B3SOIPDalphaGB1Given)
           model->B3SOIPDalphaGB1 = 0.35;
        if (!model->B3SOIPDbetaGB1Given)
           model->B3SOIPDbetaGB1 = 0.03;
        if (!model->B3SOIPDvgb1Given)
           model->B3SOIPDvgb1 = 300;
        if (!model->B3SOIPDalphaGB2Given)
           model->B3SOIPDalphaGB2 = 0.43;
        if (!model->B3SOIPDbetaGB2Given)
           model->B3SOIPDbetaGB2 = 0.05;
        if (!model->B3SOIPDvecbGiven)
           model->B3SOIPDvecb = 0.026;
        if (!model->B3SOIPDvgb2Given)
           model->B3SOIPDvgb2 = 17;
        if (!model->B3SOIPDtoxqmGiven)
           model->B3SOIPDtoxqm = model->B3SOIPDtox;
        if (!model->B3SOIPDvoxhGiven)
           model->B3SOIPDvoxh = 5.0;
        if (!model->B3SOIPDdeltavoxGiven)
           model->B3SOIPDdeltavox = 0.005;
        if (!model->B3SOIPDigModGiven)
           model->B3SOIPDigMod = 0;


/* v2.0 release */
        if (!model->B3SOIPDk1w1Given) 
           model->B3SOIPDk1w1 = 0.0;
        if (!model->B3SOIPDk1w2Given)
           model->B3SOIPDk1w2 = 0.0;
        if (!model->B3SOIPDketasGiven)
           model->B3SOIPDketas = 0.0;
        if (!model->B3SOIPDdwbcGiven)
           model->B3SOIPDdwbc = 0.0;
        if (!model->B3SOIPDbeta0Given)
           model->B3SOIPDbeta0 = 0.0;
        if (!model->B3SOIPDbeta1Given)
           model->B3SOIPDbeta1 = 0.0;
        if (!model->B3SOIPDbeta2Given)
           model->B3SOIPDbeta2 = 0.1;
        if (!model->B3SOIPDvdsatii0Given)
           model->B3SOIPDvdsatii0 = 0.9;
        if (!model->B3SOIPDtiiGiven)
           model->B3SOIPDtii = 0.0;
        if (!model->B3SOIPDliiGiven)
           model->B3SOIPDlii = 0.0;
        if (!model->B3SOIPDsii0Given)
           model->B3SOIPDsii0 = 0.5;
        if (!model->B3SOIPDsii1Given)
           model->B3SOIPDsii1 = 0.1;
        if (!model->B3SOIPDsii2Given)
           model->B3SOIPDsii2 = 0.0;
        if (!model->B3SOIPDsiidGiven)
           model->B3SOIPDsiid = 0.0;
        if (!model->B3SOIPDfbjtiiGiven)
           model->B3SOIPDfbjtii = 0.0;
        if (!model->B3SOIPDesatiiGiven)
            model->B3SOIPDesatii = 1e7;
        if (!model->B3SOIPDlnGiven)
           model->B3SOIPDln = 2e-6;
        if (!model->B3SOIPDvrec0Given)
           model->B3SOIPDvrec0 = 0;
        if (!model->B3SOIPDvtun0Given)
           model->B3SOIPDvtun0 = 0;
        if (!model->B3SOIPDnbjtGiven)
           model->B3SOIPDnbjt = 1.0;
        if (!model->B3SOIPDlbjt0Given)
           model->B3SOIPDlbjt0 = 0.20e-6;
        if (!model->B3SOIPDldif0Given)
           model->B3SOIPDldif0 = 1.0;
        if (!model->B3SOIPDvabjtGiven)
           model->B3SOIPDvabjt = 10.0;
        if (!model->B3SOIPDaelyGiven)
           model->B3SOIPDaely = 0;
        if (!model->B3SOIPDahliGiven)
           model->B3SOIPDahli = 0;
        if (!model->B3SOIPDrbodyGiven)
           model->B3SOIPDrbody = 0.0;
        if (!model->B3SOIPDrbshGiven)
           model->B3SOIPDrbsh = 0.0;
        if (!model->B3SOIPDntrecfGiven)
           model->B3SOIPDntrecf = 0.0;
        if (!model->B3SOIPDntrecrGiven)
           model->B3SOIPDntrecr = 0.0;
        if (!model->B3SOIPDndifGiven)
           model->B3SOIPDndif = -1.0;
        if (!model->B3SOIPDdlcbGiven)
           model->B3SOIPDdlcb = 0.0;
        if (!model->B3SOIPDfbodyGiven)
           model->B3SOIPDfbody = 1.0;
        if (!model->B3SOIPDtcjswgGiven)
           model->B3SOIPDtcjswg = 0.0;
        if (!model->B3SOIPDtpbswgGiven)
           model->B3SOIPDtpbswg = 0.0;
        if (!model->B3SOIPDacdeGiven)
           model->B3SOIPDacde = 1.0;
        if (!model->B3SOIPDmoinGiven)
           model->B3SOIPDmoin = 15.0;
        if (!model->B3SOIPDdelvtGiven)
           model->B3SOIPDdelvt = 0.0;
        if (!model->B3SOIPDkb1Given)
           model->B3SOIPDkb1 = 1.0;
        if (!model->B3SOIPDdlbgGiven)
           model->B3SOIPDdlbg = 0.0;

/* Added for binning - START */
        /* Length dependence */
        if (!model->B3SOIPDlnpeakGiven)
            model->B3SOIPDlnpeak = 0.0;
        if (!model->B3SOIPDlnsubGiven)
            model->B3SOIPDlnsub = 0.0;
        if (!model->B3SOIPDlngateGiven)
            model->B3SOIPDlngate = 0.0;
        if (!model->B3SOIPDlvth0Given)
           model->B3SOIPDlvth0 = 0.0;
        if (!model->B3SOIPDlk1Given)
            model->B3SOIPDlk1 = 0.0;
        if (!model->B3SOIPDlk1w1Given)
            model->B3SOIPDlk1w1 = 0.0;
        if (!model->B3SOIPDlk1w2Given)
            model->B3SOIPDlk1w2 = 0.0;
        if (!model->B3SOIPDlk2Given)
            model->B3SOIPDlk2 = 0.0;
        if (!model->B3SOIPDlk3Given)
            model->B3SOIPDlk3 = 0.0;
        if (!model->B3SOIPDlk3bGiven)
            model->B3SOIPDlk3b = 0.0;
        if (!model->B3SOIPDlkb1Given)
           model->B3SOIPDlkb1 = 0.0;
        if (!model->B3SOIPDlw0Given)
            model->B3SOIPDlw0 = 0.0;
        if (!model->B3SOIPDlnlxGiven)
            model->B3SOIPDlnlx = 0.0;
        if (!model->B3SOIPDldvt0Given)
            model->B3SOIPDldvt0 = 0.0;
        if (!model->B3SOIPDldvt1Given)
            model->B3SOIPDldvt1 = 0.0;
        if (!model->B3SOIPDldvt2Given)
            model->B3SOIPDldvt2 = 0.0;
        if (!model->B3SOIPDldvt0wGiven)
            model->B3SOIPDldvt0w = 0.0;
        if (!model->B3SOIPDldvt1wGiven)
            model->B3SOIPDldvt1w = 0.0;
        if (!model->B3SOIPDldvt2wGiven)
            model->B3SOIPDldvt2w = 0.0;
        if (!model->B3SOIPDlu0Given)
            model->B3SOIPDlu0 = 0.0;
        if (!model->B3SOIPDluaGiven)
            model->B3SOIPDlua = 0.0;
        if (!model->B3SOIPDlubGiven)
            model->B3SOIPDlub = 0.0;
        if (!model->B3SOIPDlucGiven)
            model->B3SOIPDluc = 0.0;
        if (!model->B3SOIPDlvsatGiven)
            model->B3SOIPDlvsat = 0.0;
        if (!model->B3SOIPDla0Given)
            model->B3SOIPDla0 = 0.0;
        if (!model->B3SOIPDlagsGiven)
            model->B3SOIPDlags = 0.0;
        if (!model->B3SOIPDlb0Given)
            model->B3SOIPDlb0 = 0.0;
        if (!model->B3SOIPDlb1Given)
            model->B3SOIPDlb1 = 0.0;
        if (!model->B3SOIPDlketaGiven)
            model->B3SOIPDlketa = 0.0;
        if (!model->B3SOIPDlketasGiven)
            model->B3SOIPDlketas = 0.0;
        if (!model->B3SOIPDla1Given)
            model->B3SOIPDla1 = 0.0;
        if (!model->B3SOIPDla2Given)
            model->B3SOIPDla2 = 0.0;
        if (!model->B3SOIPDlrdswGiven)
            model->B3SOIPDlrdsw = 0.0;
        if (!model->B3SOIPDlprwbGiven)
            model->B3SOIPDlprwb = 0.0;
        if (!model->B3SOIPDlprwgGiven)
            model->B3SOIPDlprwg = 0.0;
        if (!model->B3SOIPDlwrGiven)
            model->B3SOIPDlwr = 0.0;
        if (!model->B3SOIPDlnfactorGiven)
            model->B3SOIPDlnfactor = 0.0;
        if (!model->B3SOIPDldwgGiven)
            model->B3SOIPDldwg = 0.0;
        if (!model->B3SOIPDldwbGiven)
            model->B3SOIPDldwb = 0.0;
        if (!model->B3SOIPDlvoffGiven)
            model->B3SOIPDlvoff = 0.0;
        if (!model->B3SOIPDleta0Given)
            model->B3SOIPDleta0 = 0.0;
        if (!model->B3SOIPDletabGiven)
            model->B3SOIPDletab = 0.0;
        if (!model->B3SOIPDldsubGiven)
            model->B3SOIPDldsub = 0.0;
        if (!model->B3SOIPDlcitGiven)
            model->B3SOIPDlcit = 0.0;
        if (!model->B3SOIPDlcdscGiven)
            model->B3SOIPDlcdsc = 0.0;
        if (!model->B3SOIPDlcdscbGiven)
            model->B3SOIPDlcdscb = 0.0;
        if (!model->B3SOIPDlcdscdGiven)
            model->B3SOIPDlcdscd = 0.0;
        if (!model->B3SOIPDlpclmGiven)
            model->B3SOIPDlpclm = 0.0;
        if (!model->B3SOIPDlpdibl1Given)
            model->B3SOIPDlpdibl1 = 0.0;
        if (!model->B3SOIPDlpdibl2Given)
            model->B3SOIPDlpdibl2 = 0.0;
        if (!model->B3SOIPDlpdiblbGiven)
            model->B3SOIPDlpdiblb = 0.0;
        if (!model->B3SOIPDldroutGiven)
            model->B3SOIPDldrout = 0.0;
        if (!model->B3SOIPDlpvagGiven)
            model->B3SOIPDlpvag = 0.0;
        if (!model->B3SOIPDldeltaGiven)
            model->B3SOIPDldelta = 0.0;
        if (!model->B3SOIPDlalpha0Given)
            model->B3SOIPDlalpha0 = 0.0;
        if (!model->B3SOIPDlfbjtiiGiven)
            model->B3SOIPDlfbjtii = 0.0;
        if (!model->B3SOIPDlbeta0Given)
            model->B3SOIPDlbeta0 = 0.0;
        if (!model->B3SOIPDlbeta1Given)
            model->B3SOIPDlbeta1 = 0.0;
        if (!model->B3SOIPDlbeta2Given)
            model->B3SOIPDlbeta2 = 0.0;
        if (!model->B3SOIPDlvdsatii0Given)
            model->B3SOIPDlvdsatii0 = 0.0;
        if (!model->B3SOIPDlliiGiven)
            model->B3SOIPDllii = 0.0;
        if (!model->B3SOIPDlesatiiGiven)
            model->B3SOIPDlesatii = 0.0;
        if (!model->B3SOIPDlsii0Given)
            model->B3SOIPDlsii0 = 0.0;
        if (!model->B3SOIPDlsii1Given)
            model->B3SOIPDlsii1 = 0.0;
        if (!model->B3SOIPDlsii2Given)
            model->B3SOIPDlsii2 = 0.0;
        if (!model->B3SOIPDlsiidGiven)
            model->B3SOIPDlsiid = 0.0;
        if (!model->B3SOIPDlagidlGiven)
            model->B3SOIPDlagidl = 0.0;
        if (!model->B3SOIPDlbgidlGiven)
            model->B3SOIPDlbgidl = 0.0;
        if (!model->B3SOIPDlngidlGiven)
            model->B3SOIPDlngidl = 0.0;
        if (!model->B3SOIPDlntunGiven)
            model->B3SOIPDlntun = 0.0;
        if (!model->B3SOIPDlndiodeGiven)
            model->B3SOIPDlndiode = 0.0;
        if (!model->B3SOIPDlnrecf0Given)
            model->B3SOIPDlnrecf0 = 0.0;
        if (!model->B3SOIPDlnrecr0Given)
            model->B3SOIPDlnrecr0 = 0.0;
        if (!model->B3SOIPDlisbjtGiven)
            model->B3SOIPDlisbjt = 0.0;
        if (!model->B3SOIPDlisdifGiven)
            model->B3SOIPDlisdif = 0.0;
        if (!model->B3SOIPDlisrecGiven)
            model->B3SOIPDlisrec = 0.0;
        if (!model->B3SOIPDlistunGiven)
            model->B3SOIPDlistun = 0.0;
        if (!model->B3SOIPDlvrec0Given)
            model->B3SOIPDlvrec0 = 0.0;
        if (!model->B3SOIPDlvtun0Given)
            model->B3SOIPDlvtun0 = 0.0;
        if (!model->B3SOIPDlnbjtGiven)
            model->B3SOIPDlnbjt = 0.0;
        if (!model->B3SOIPDllbjt0Given)
            model->B3SOIPDllbjt0 = 0.0;
        if (!model->B3SOIPDlvabjtGiven)
            model->B3SOIPDlvabjt = 0.0;
        if (!model->B3SOIPDlaelyGiven)
            model->B3SOIPDlaely = 0.0;
        if (!model->B3SOIPDlahliGiven)
            model->B3SOIPDlahli = 0.0;
	/* CV Model */
        if (!model->B3SOIPDlvsdfbGiven)
            model->B3SOIPDlvsdfb = 0.0;
        if (!model->B3SOIPDlvsdthGiven)
            model->B3SOIPDlvsdth = 0.0;
        if (!model->B3SOIPDldelvtGiven)
            model->B3SOIPDldelvt = 0.0;
        if (!model->B3SOIPDlacdeGiven)
            model->B3SOIPDlacde = 0.0;
        if (!model->B3SOIPDlmoinGiven)
            model->B3SOIPDlmoin = 0.0;

        /* Width dependence */
        if (!model->B3SOIPDwnpeakGiven)
            model->B3SOIPDwnpeak = 0.0;
        if (!model->B3SOIPDwnsubGiven)
            model->B3SOIPDwnsub = 0.0;
        if (!model->B3SOIPDwngateGiven)
            model->B3SOIPDwngate = 0.0;
        if (!model->B3SOIPDwvth0Given)
           model->B3SOIPDwvth0 = 0.0;
        if (!model->B3SOIPDwk1Given)
            model->B3SOIPDwk1 = 0.0;
        if (!model->B3SOIPDwk1w1Given)
            model->B3SOIPDwk1w1 = 0.0;
        if (!model->B3SOIPDwk1w2Given)
            model->B3SOIPDwk1w2 = 0.0;
        if (!model->B3SOIPDwk2Given)
            model->B3SOIPDwk2 = 0.0;
        if (!model->B3SOIPDwk3Given)
            model->B3SOIPDwk3 = 0.0;
        if (!model->B3SOIPDwk3bGiven)
            model->B3SOIPDwk3b = 0.0;
        if (!model->B3SOIPDwkb1Given)
           model->B3SOIPDwkb1 = 0.0;
        if (!model->B3SOIPDww0Given)
            model->B3SOIPDww0 = 0.0;
        if (!model->B3SOIPDwnlxGiven)
            model->B3SOIPDwnlx = 0.0;
        if (!model->B3SOIPDwdvt0Given)
            model->B3SOIPDwdvt0 = 0.0;
        if (!model->B3SOIPDwdvt1Given)
            model->B3SOIPDwdvt1 = 0.0;
        if (!model->B3SOIPDwdvt2Given)
            model->B3SOIPDwdvt2 = 0.0;
        if (!model->B3SOIPDwdvt0wGiven)
            model->B3SOIPDwdvt0w = 0.0;
        if (!model->B3SOIPDwdvt1wGiven)
            model->B3SOIPDwdvt1w = 0.0;
        if (!model->B3SOIPDwdvt2wGiven)
            model->B3SOIPDwdvt2w = 0.0;
        if (!model->B3SOIPDwu0Given)
            model->B3SOIPDwu0 = 0.0;
        if (!model->B3SOIPDwuaGiven)
            model->B3SOIPDwua = 0.0;
        if (!model->B3SOIPDwubGiven)
            model->B3SOIPDwub = 0.0;
        if (!model->B3SOIPDwucGiven)
            model->B3SOIPDwuc = 0.0;
        if (!model->B3SOIPDwvsatGiven)
            model->B3SOIPDwvsat = 0.0;
        if (!model->B3SOIPDwa0Given)
            model->B3SOIPDwa0 = 0.0;
        if (!model->B3SOIPDwagsGiven)
            model->B3SOIPDwags = 0.0;
        if (!model->B3SOIPDwb0Given)
            model->B3SOIPDwb0 = 0.0;
        if (!model->B3SOIPDwb1Given)
            model->B3SOIPDwb1 = 0.0;
        if (!model->B3SOIPDwketaGiven)
            model->B3SOIPDwketa = 0.0;
        if (!model->B3SOIPDwketasGiven)
            model->B3SOIPDwketas = 0.0;
        if (!model->B3SOIPDwa1Given)
            model->B3SOIPDwa1 = 0.0;
        if (!model->B3SOIPDwa2Given)
            model->B3SOIPDwa2 = 0.0;
        if (!model->B3SOIPDwrdswGiven)
            model->B3SOIPDwrdsw = 0.0;
        if (!model->B3SOIPDwprwbGiven)
            model->B3SOIPDwprwb = 0.0;
        if (!model->B3SOIPDwprwgGiven)
            model->B3SOIPDwprwg = 0.0;
        if (!model->B3SOIPDwwrGiven)
            model->B3SOIPDwwr = 0.0;
        if (!model->B3SOIPDwnfactorGiven)
            model->B3SOIPDwnfactor = 0.0;
        if (!model->B3SOIPDwdwgGiven)
            model->B3SOIPDwdwg = 0.0;
        if (!model->B3SOIPDwdwbGiven)
            model->B3SOIPDwdwb = 0.0;
        if (!model->B3SOIPDwvoffGiven)
            model->B3SOIPDwvoff = 0.0;
        if (!model->B3SOIPDweta0Given)
            model->B3SOIPDweta0 = 0.0;
        if (!model->B3SOIPDwetabGiven)
            model->B3SOIPDwetab = 0.0;
        if (!model->B3SOIPDwdsubGiven)
            model->B3SOIPDwdsub = 0.0;
        if (!model->B3SOIPDwcitGiven)
            model->B3SOIPDwcit = 0.0;
        if (!model->B3SOIPDwcdscGiven)
            model->B3SOIPDwcdsc = 0.0;
        if (!model->B3SOIPDwcdscbGiven)
            model->B3SOIPDwcdscb = 0.0;
        if (!model->B3SOIPDwcdscdGiven)
            model->B3SOIPDwcdscd = 0.0;
        if (!model->B3SOIPDwpclmGiven)
            model->B3SOIPDwpclm = 0.0;
        if (!model->B3SOIPDwpdibl1Given)
            model->B3SOIPDwpdibl1 = 0.0;
        if (!model->B3SOIPDwpdibl2Given)
            model->B3SOIPDwpdibl2 = 0.0;
        if (!model->B3SOIPDwpdiblbGiven)
            model->B3SOIPDwpdiblb = 0.0;
        if (!model->B3SOIPDwdroutGiven)
            model->B3SOIPDwdrout = 0.0;
        if (!model->B3SOIPDwpvagGiven)
            model->B3SOIPDwpvag = 0.0;
        if (!model->B3SOIPDwdeltaGiven)
            model->B3SOIPDwdelta = 0.0;
        if (!model->B3SOIPDwalpha0Given)
            model->B3SOIPDwalpha0 = 0.0;
        if (!model->B3SOIPDwfbjtiiGiven)
            model->B3SOIPDwfbjtii = 0.0;
        if (!model->B3SOIPDwbeta0Given)
            model->B3SOIPDwbeta0 = 0.0;
        if (!model->B3SOIPDwbeta1Given)
            model->B3SOIPDwbeta1 = 0.0;
        if (!model->B3SOIPDwbeta2Given)
            model->B3SOIPDwbeta2 = 0.0;
        if (!model->B3SOIPDwvdsatii0Given)
            model->B3SOIPDwvdsatii0 = 0.0;
        if (!model->B3SOIPDwliiGiven)
            model->B3SOIPDwlii = 0.0;
        if (!model->B3SOIPDwesatiiGiven)
            model->B3SOIPDwesatii = 0.0;
        if (!model->B3SOIPDwsii0Given)
            model->B3SOIPDwsii0 = 0.0;
        if (!model->B3SOIPDwsii1Given)
            model->B3SOIPDwsii1 = 0.0;
        if (!model->B3SOIPDwsii2Given)
            model->B3SOIPDwsii2 = 0.0;
        if (!model->B3SOIPDwsiidGiven)
            model->B3SOIPDwsiid = 0.0;
        if (!model->B3SOIPDwagidlGiven)
            model->B3SOIPDwagidl = 0.0;
        if (!model->B3SOIPDwbgidlGiven)
            model->B3SOIPDwbgidl = 0.0;
        if (!model->B3SOIPDwngidlGiven)
            model->B3SOIPDwngidl = 0.0;
        if (!model->B3SOIPDwntunGiven)
            model->B3SOIPDwntun = 0.0;
        if (!model->B3SOIPDwndiodeGiven)
            model->B3SOIPDwndiode = 0.0;
        if (!model->B3SOIPDwnrecf0Given)
            model->B3SOIPDwnrecf0 = 0.0;
        if (!model->B3SOIPDwnrecr0Given)
            model->B3SOIPDwnrecr0 = 0.0;
        if (!model->B3SOIPDwisbjtGiven)
            model->B3SOIPDwisbjt = 0.0;
        if (!model->B3SOIPDwisdifGiven)
            model->B3SOIPDwisdif = 0.0;
        if (!model->B3SOIPDwisrecGiven)
            model->B3SOIPDwisrec = 0.0;
        if (!model->B3SOIPDwistunGiven)
            model->B3SOIPDwistun = 0.0;
        if (!model->B3SOIPDwvrec0Given)
            model->B3SOIPDwvrec0 = 0.0;
        if (!model->B3SOIPDwvtun0Given)
            model->B3SOIPDwvtun0 = 0.0;
        if (!model->B3SOIPDwnbjtGiven)
            model->B3SOIPDwnbjt = 0.0;
        if (!model->B3SOIPDwlbjt0Given)
            model->B3SOIPDwlbjt0 = 0.0;
        if (!model->B3SOIPDwvabjtGiven)
            model->B3SOIPDwvabjt = 0.0;
        if (!model->B3SOIPDwaelyGiven)
            model->B3SOIPDwaely = 0.0;
        if (!model->B3SOIPDwahliGiven)
            model->B3SOIPDwahli = 0.0;
	/* CV Model */
        if (!model->B3SOIPDwvsdfbGiven)
            model->B3SOIPDwvsdfb = 0.0;
        if (!model->B3SOIPDwvsdthGiven)
            model->B3SOIPDwvsdth = 0.0;
        if (!model->B3SOIPDwdelvtGiven)
            model->B3SOIPDwdelvt = 0.0;
        if (!model->B3SOIPDwacdeGiven)
            model->B3SOIPDwacde = 0.0;
        if (!model->B3SOIPDwmoinGiven)
            model->B3SOIPDwmoin = 0.0;

        /* Cross-term dependence */
        if (!model->B3SOIPDpnpeakGiven)
            model->B3SOIPDpnpeak = 0.0;
        if (!model->B3SOIPDpnsubGiven)
            model->B3SOIPDpnsub = 0.0;
        if (!model->B3SOIPDpngateGiven)
            model->B3SOIPDpngate = 0.0;
        if (!model->B3SOIPDpvth0Given)
           model->B3SOIPDpvth0 = 0.0;
        if (!model->B3SOIPDpk1Given)
            model->B3SOIPDpk1 = 0.0;
        if (!model->B3SOIPDpk1w1Given)
            model->B3SOIPDpk1w1 = 0.0;
        if (!model->B3SOIPDpk1w2Given)
            model->B3SOIPDpk1w2 = 0.0;
        if (!model->B3SOIPDpk2Given)
            model->B3SOIPDpk2 = 0.0;
        if (!model->B3SOIPDpk3Given)
            model->B3SOIPDpk3 = 0.0;
        if (!model->B3SOIPDpk3bGiven)
            model->B3SOIPDpk3b = 0.0;
        if (!model->B3SOIPDpkb1Given)
           model->B3SOIPDpkb1 = 0.0;
        if (!model->B3SOIPDpw0Given)
            model->B3SOIPDpw0 = 0.0;
        if (!model->B3SOIPDpnlxGiven)
            model->B3SOIPDpnlx = 0.0;
        if (!model->B3SOIPDpdvt0Given)
            model->B3SOIPDpdvt0 = 0.0;
        if (!model->B3SOIPDpdvt1Given)
            model->B3SOIPDpdvt1 = 0.0;
        if (!model->B3SOIPDpdvt2Given)
            model->B3SOIPDpdvt2 = 0.0;
        if (!model->B3SOIPDpdvt0wGiven)
            model->B3SOIPDpdvt0w = 0.0;
        if (!model->B3SOIPDpdvt1wGiven)
            model->B3SOIPDpdvt1w = 0.0;
        if (!model->B3SOIPDpdvt2wGiven)
            model->B3SOIPDpdvt2w = 0.0;
        if (!model->B3SOIPDpu0Given)
            model->B3SOIPDpu0 = 0.0;
        if (!model->B3SOIPDpuaGiven)
            model->B3SOIPDpua = 0.0;
        if (!model->B3SOIPDpubGiven)
            model->B3SOIPDpub = 0.0;
        if (!model->B3SOIPDpucGiven)
            model->B3SOIPDpuc = 0.0;
        if (!model->B3SOIPDpvsatGiven)
            model->B3SOIPDpvsat = 0.0;
        if (!model->B3SOIPDpa0Given)
            model->B3SOIPDpa0 = 0.0;
        if (!model->B3SOIPDpagsGiven)
            model->B3SOIPDpags = 0.0;
        if (!model->B3SOIPDpb0Given)
            model->B3SOIPDpb0 = 0.0;
        if (!model->B3SOIPDpb1Given)
            model->B3SOIPDpb1 = 0.0;
        if (!model->B3SOIPDpketaGiven)
            model->B3SOIPDpketa = 0.0;
        if (!model->B3SOIPDpketasGiven)
            model->B3SOIPDpketas = 0.0;
        if (!model->B3SOIPDpa1Given)
            model->B3SOIPDpa1 = 0.0;
        if (!model->B3SOIPDpa2Given)
            model->B3SOIPDpa2 = 0.0;
        if (!model->B3SOIPDprdswGiven)
            model->B3SOIPDprdsw = 0.0;
        if (!model->B3SOIPDpprwbGiven)
            model->B3SOIPDpprwb = 0.0;
        if (!model->B3SOIPDpprwgGiven)
            model->B3SOIPDpprwg = 0.0;
        if (!model->B3SOIPDpwrGiven)
            model->B3SOIPDpwr = 0.0;
        if (!model->B3SOIPDpnfactorGiven)
            model->B3SOIPDpnfactor = 0.0;
        if (!model->B3SOIPDpdwgGiven)
            model->B3SOIPDpdwg = 0.0;
        if (!model->B3SOIPDpdwbGiven)
            model->B3SOIPDpdwb = 0.0;
        if (!model->B3SOIPDpvoffGiven)
            model->B3SOIPDpvoff = 0.0;
        if (!model->B3SOIPDpeta0Given)
            model->B3SOIPDpeta0 = 0.0;
        if (!model->B3SOIPDpetabGiven)
            model->B3SOIPDpetab = 0.0;
        if (!model->B3SOIPDpdsubGiven)
            model->B3SOIPDpdsub = 0.0;
        if (!model->B3SOIPDpcitGiven)
            model->B3SOIPDpcit = 0.0;
        if (!model->B3SOIPDpcdscGiven)
            model->B3SOIPDpcdsc = 0.0;
        if (!model->B3SOIPDpcdscbGiven)
            model->B3SOIPDpcdscb = 0.0;
        if (!model->B3SOIPDpcdscdGiven)
            model->B3SOIPDpcdscd = 0.0;
        if (!model->B3SOIPDppclmGiven)
            model->B3SOIPDppclm = 0.0;
        if (!model->B3SOIPDppdibl1Given)
            model->B3SOIPDppdibl1 = 0.0;
        if (!model->B3SOIPDppdibl2Given)
            model->B3SOIPDppdibl2 = 0.0;
        if (!model->B3SOIPDppdiblbGiven)
            model->B3SOIPDppdiblb = 0.0;
        if (!model->B3SOIPDpdroutGiven)
            model->B3SOIPDpdrout = 0.0;
        if (!model->B3SOIPDppvagGiven)
            model->B3SOIPDppvag = 0.0;
        if (!model->B3SOIPDpdeltaGiven)
            model->B3SOIPDpdelta = 0.0;
        if (!model->B3SOIPDpalpha0Given)
            model->B3SOIPDpalpha0 = 0.0;
        if (!model->B3SOIPDpfbjtiiGiven)
            model->B3SOIPDpfbjtii = 0.0;
        if (!model->B3SOIPDpbeta0Given)
            model->B3SOIPDpbeta0 = 0.0;
        if (!model->B3SOIPDpbeta1Given)
            model->B3SOIPDpbeta1 = 0.0;
        if (!model->B3SOIPDpbeta2Given)
            model->B3SOIPDpbeta2 = 0.0;
        if (!model->B3SOIPDpvdsatii0Given)
            model->B3SOIPDpvdsatii0 = 0.0;
        if (!model->B3SOIPDpliiGiven)
            model->B3SOIPDplii = 0.0;
        if (!model->B3SOIPDpesatiiGiven)
            model->B3SOIPDpesatii = 0.0;
        if (!model->B3SOIPDpsii0Given)
            model->B3SOIPDpsii0 = 0.0;
        if (!model->B3SOIPDpsii1Given)
            model->B3SOIPDpsii1 = 0.0;
        if (!model->B3SOIPDpsii2Given)
            model->B3SOIPDpsii2 = 0.0;
        if (!model->B3SOIPDpsiidGiven)
            model->B3SOIPDpsiid = 0.0;
        if (!model->B3SOIPDpagidlGiven)
            model->B3SOIPDpagidl = 0.0;
        if (!model->B3SOIPDpbgidlGiven)
            model->B3SOIPDpbgidl = 0.0;
        if (!model->B3SOIPDpngidlGiven)
            model->B3SOIPDpngidl = 0.0;
        if (!model->B3SOIPDpntunGiven)
            model->B3SOIPDpntun = 0.0;
        if (!model->B3SOIPDpndiodeGiven)
            model->B3SOIPDpndiode = 0.0;
        if (!model->B3SOIPDpnrecf0Given)
            model->B3SOIPDpnrecf0 = 0.0;
        if (!model->B3SOIPDpnrecr0Given)
            model->B3SOIPDpnrecr0 = 0.0;
        if (!model->B3SOIPDpisbjtGiven)
            model->B3SOIPDpisbjt = 0.0;
        if (!model->B3SOIPDpisdifGiven)
            model->B3SOIPDpisdif = 0.0;
        if (!model->B3SOIPDpisrecGiven)
            model->B3SOIPDpisrec = 0.0;
        if (!model->B3SOIPDpistunGiven)
            model->B3SOIPDpistun = 0.0;
        if (!model->B3SOIPDpvrec0Given)
            model->B3SOIPDpvrec0 = 0.0;
        if (!model->B3SOIPDpvtun0Given)
            model->B3SOIPDpvtun0 = 0.0;
        if (!model->B3SOIPDpnbjtGiven)
            model->B3SOIPDpnbjt = 0.0;
        if (!model->B3SOIPDplbjt0Given)
            model->B3SOIPDplbjt0 = 0.0;
        if (!model->B3SOIPDpvabjtGiven)
            model->B3SOIPDpvabjt = 0.0;
        if (!model->B3SOIPDpaelyGiven)
            model->B3SOIPDpaely = 0.0;
        if (!model->B3SOIPDpahliGiven)
            model->B3SOIPDpahli = 0.0;
	/* CV Model */
        if (!model->B3SOIPDpvsdfbGiven)
            model->B3SOIPDpvsdfb = 0.0;
        if (!model->B3SOIPDpvsdthGiven)
            model->B3SOIPDpvsdth = 0.0;
        if (!model->B3SOIPDpdelvtGiven)
            model->B3SOIPDpdelvt = 0.0;
        if (!model->B3SOIPDpacdeGiven)
            model->B3SOIPDpacde = 0.0;
        if (!model->B3SOIPDpmoinGiven)
            model->B3SOIPDpmoin = 0.0;
/* Added for binning - END */

	if (!model->B3SOIPDcfGiven)
            model->B3SOIPDcf = 2.0 * EPSOX / PI
			   * log(1.0 + 0.4e-6 / model->B3SOIPDtox);
        if (!model->B3SOIPDcgdoGiven)
	{   if (model->B3SOIPDdlcGiven && (model->B3SOIPDdlc > 0.0))
	    {   model->B3SOIPDcgdo = model->B3SOIPDdlc * model->B3SOIPDcox
				 - model->B3SOIPDcgdl ;
	    }
	    else
	        model->B3SOIPDcgdo = 0.6 * model->B3SOIPDxj * model->B3SOIPDcox; 
	}
        if (!model->B3SOIPDcgsoGiven)
	{   if (model->B3SOIPDdlcGiven && (model->B3SOIPDdlc > 0.0))
	    {   model->B3SOIPDcgso = model->B3SOIPDdlc * model->B3SOIPDcox
				 - model->B3SOIPDcgsl ;
	    }
	    else
	        model->B3SOIPDcgso = 0.6 * model->B3SOIPDxj * model->B3SOIPDcox; 
	}

        if (!model->B3SOIPDcgeoGiven)
	{   model->B3SOIPDcgeo = 0.0;
	}
        if (!model->B3SOIPDxpartGiven)
            model->B3SOIPDxpart = 0.0;
        if (!model->B3SOIPDsheetResistanceGiven)
            model->B3SOIPDsheetResistance = 0.0;
        if (!model->B3SOIPDcsdeswGiven)
            model->B3SOIPDcsdesw = 0.0;
        if (!model->B3SOIPDunitLengthGateSidewallJctCapGiven)
            model->B3SOIPDunitLengthGateSidewallJctCap = 1e-10;
        if (!model->B3SOIPDGatesidewallJctPotentialGiven)
            model->B3SOIPDGatesidewallJctPotential = 0.7;
        if (!model->B3SOIPDbodyJctGateSideGradingCoeffGiven)
            model->B3SOIPDbodyJctGateSideGradingCoeff = 0.5;
        if (!model->B3SOIPDoxideTrapDensityAGiven)
	{   if (model->B3SOIPDtype == NMOS)
                model->B3SOIPDoxideTrapDensityA = 1e20;
            else
                model->B3SOIPDoxideTrapDensityA=9.9e18;
	}
        if (!model->B3SOIPDoxideTrapDensityBGiven)
	{   if (model->B3SOIPDtype == NMOS)
                model->B3SOIPDoxideTrapDensityB = 5e4;
            else
                model->B3SOIPDoxideTrapDensityB = 2.4e3;
	}
        if (!model->B3SOIPDoxideTrapDensityCGiven)
	{   if (model->B3SOIPDtype == NMOS)
                model->B3SOIPDoxideTrapDensityC = -1.4e-12;
            else
                model->B3SOIPDoxideTrapDensityC = 1.4e-12;

	}
        if (!model->B3SOIPDemGiven)
            model->B3SOIPDem = 4.1e7; /* V/m */
        if (!model->B3SOIPDefGiven)
            model->B3SOIPDef = 1.0;
        if (!model->B3SOIPDafGiven)
            model->B3SOIPDaf = 1.0;
        if (!model->B3SOIPDkfGiven)
            model->B3SOIPDkf = 0.0;
        if (!model->B3SOIPDnoifGiven)
            model->B3SOIPDnoif = 1.0;

        /* loop through all the instances of the model */
        for (here = B3SOIPDinstances(model); here != NULL ;
             here=B3SOIPDnextInstance(here)) 
	{
            /* allocate a chunk of the state vector */
            here->B3SOIPDstates = *states;
            *states += B3SOIPDnumStates;
	    
            /* perform the parameter defaulting */
            if (!here->B3SOIPDdrainAreaGiven)
                here->B3SOIPDdrainArea = 0.0;
            if (!here->B3SOIPDdrainPerimeterGiven)
                here->B3SOIPDdrainPerimeter = 0.0;
            if (!here->B3SOIPDdrainSquaresGiven)
                here->B3SOIPDdrainSquares = 1.0;
            if (!here->B3SOIPDicVBSGiven)
                here->B3SOIPDicVBS = 0;
            if (!here->B3SOIPDicVDSGiven)
                here->B3SOIPDicVDS = 0;
            if (!here->B3SOIPDicVGSGiven)
                here->B3SOIPDicVGS = 0;
            if (!here->B3SOIPDicVESGiven)
                here->B3SOIPDicVES = 0;
            if (!here->B3SOIPDicVPSGiven)
                here->B3SOIPDicVPS = 0;
	    if (!here->B3SOIPDbjtoffGiven)
		here->B3SOIPDbjtoff = 0;
	    if (!here->B3SOIPDdebugModGiven)
		here->B3SOIPDdebugMod = 0;
	    if (!here->B3SOIPDrth0Given)
		here->B3SOIPDrth0 = model->B3SOIPDrth0;
	    if (!here->B3SOIPDcth0Given)
		here->B3SOIPDcth0 = model->B3SOIPDcth0;
            if (!here->B3SOIPDbodySquaresGiven)
                here->B3SOIPDbodySquares = 1.0;
            if (!here->B3SOIPDfrbodyGiven)
                here->B3SOIPDfrbody = 1.0;
            if (!here->B3SOIPDlGiven)
                here->B3SOIPDl = 5e-6;
            if (!here->B3SOIPDsourceAreaGiven)
                here->B3SOIPDsourceArea = 0;
            if (!here->B3SOIPDsourcePerimeterGiven)
                here->B3SOIPDsourcePerimeter = 0;
            if (!here->B3SOIPDsourceSquaresGiven)
                here->B3SOIPDsourceSquares = 1;
            if (!here->B3SOIPDwGiven)
                here->B3SOIPDw = 5e-6;
		
	    if (!here->B3SOIPDmGiven)
                here->B3SOIPDm = 1;	


/* v2.0 release */
            if (!here->B3SOIPDnbcGiven)
                here->B3SOIPDnbc = 0;
            if (!here->B3SOIPDnsegGiven)
                here->B3SOIPDnseg = 1;
            if (!here->B3SOIPDpdbcpGiven)
                here->B3SOIPDpdbcp = 0;
            if (!here->B3SOIPDpsbcpGiven)
                here->B3SOIPDpsbcp = 0;
            if (!here->B3SOIPDagbcpGiven)
                here->B3SOIPDagbcp = 0;
            if (!here->B3SOIPDaebcpGiven)
                here->B3SOIPDaebcp = 0;

            if (!here->B3SOIPDoffGiven)
                here->B3SOIPDoff = 0;

            /* process drain series resistance */
            if ((model->B3SOIPDsheetResistance > 0.0) && 
                (here->B3SOIPDdrainSquares > 0.0 ) &&
                (here->B3SOIPDdNodePrime == 0))
	    {   error = CKTmkVolt(ckt,&tmp,here->B3SOIPDname,"drain");
                if(error) return(error);
                here->B3SOIPDdNodePrime = tmp->number;
		
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
	    {   here->B3SOIPDdNodePrime = here->B3SOIPDdNode;
            }
                   
            /* process source series resistance */
            if ((model->B3SOIPDsheetResistance > 0.0) && 
                (here->B3SOIPDsourceSquares > 0.0 ) &&
                (here->B3SOIPDsNodePrime == 0)) 
	    {   error = CKTmkVolt(ckt,&tmp,here->B3SOIPDname,"source");
                if(error) return(error);
                here->B3SOIPDsNodePrime = tmp->number;
		
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
	    {   here->B3SOIPDsNodePrime = here->B3SOIPDsNode;
            }

            /* process effective silicon film thickness */
            model->B3SOIPDcbox = 3.453133e-11 / model->B3SOIPDtbox;
            model->B3SOIPDcsi = 1.03594e-10 / model->B3SOIPDtsi;
            Cboxt = model->B3SOIPDcbox * model->B3SOIPDcsi / (model->B3SOIPDcbox + model->B3SOIPDcsi);
            model->B3SOIPDqsi = Charge_q*model->B3SOIPDnpeak*1e6*model->B3SOIPDtsi;


            here->B3SOIPDfloat = 0;

            here->B3SOIPDpNode = here->B3SOIPDpNodeExt;
            here->B3SOIPDbNode = here->B3SOIPDbNodeExt;
            here->B3SOIPDtempNode = here->B3SOIPDtempNodeExt;

            if (here->B3SOIPDpNode == -1) {  /*  floating body case -- 4-node  */
                error = CKTmkVolt(ckt,&tmp,here->B3SOIPDname,"Body");
                if (error) return(error);
                here->B3SOIPDbNode = tmp->number;
                here->B3SOIPDpNode = 0; 
                here->B3SOIPDfloat = 1;
                here->B3SOIPDbodyMod = 0;
            }
            else /* the 5th Node has been assigned */
            {
              if (!here->B3SOIPDtnodeoutGiven) { /* if t-node not assigned  */
                 if (here->B3SOIPDbNode == -1)
                 { /* 5-node body tie, bNode has not been assigned */
                    if ((model->B3SOIPDrbody == 0.0) && (model->B3SOIPDrbsh == 0.0))
                    { /* ideal body tie, pNode is not used */
                       here->B3SOIPDbNode = here->B3SOIPDpNode;
                       here->B3SOIPDbodyMod = 2;
                    }
                    else { /* nonideal body tie */
                      error = CKTmkVolt(ckt, &tmp, here->B3SOIPDname, "Body");
                      if (error) return(error);
                      here->B3SOIPDbNode = tmp->number;
                      here->B3SOIPDbodyMod = 1;
                    }
                 }
                 else { /* 6-node body tie, bNode has been assigned */
                    if ((model->B3SOIPDrbody == 0.0) && (model->B3SOIPDrbsh == 0.0))
                    { 
                       printf("\n      Warning: model parameter rbody=0!\n");
                       model->B3SOIPDrbody = 1e0;
                       here->B3SOIPDbodyMod = 1;
                    }
                    else { /* nonideal body tie */
                        here->B3SOIPDbodyMod = 1;
                    }
                 }
              }
              else {    /*  t-node assigned   */
                 if (here->B3SOIPDbNode == -1)
                 { /* 4 nodes & t-node, floating body */
                    error = CKTmkVolt(ckt, &tmp, here->B3SOIPDname, "Body");
                    if (error) return(error);
                    here->B3SOIPDbNode = tmp->number;
                    here->B3SOIPDtempNode = here->B3SOIPDpNode;
                    here->B3SOIPDpNode = 0; 
                    here->B3SOIPDfloat = 1;
                    here->B3SOIPDbodyMod = 0;
                 }
                 else { /* 5 or 6 nodes & t-node, body-contact device */
                   if (here->B3SOIPDtempNode == -1) { /* 5 nodes & tnode */
                     if ((model->B3SOIPDrbody == 0.0) && (model->B3SOIPDrbsh == 0.0))
                     { /* ideal body tie, pNode is not used */
                        here->B3SOIPDtempNode = here->B3SOIPDbNode;
                        here->B3SOIPDbNode = here->B3SOIPDpNode;
                        here->B3SOIPDbodyMod = 2;
                     }
                     else { /* nonideal body tie */
                       here->B3SOIPDtempNode = here->B3SOIPDbNode;
                       error = CKTmkVolt(ckt, &tmp, here->B3SOIPDname, "Body");
                       if (error) return(error);
                       here->B3SOIPDbNode = tmp->number;
                       here->B3SOIPDbodyMod = 1;
                     }
                   }
                   else {  /* 6 nodes & t-node */
                     if ((model->B3SOIPDrbody == 0.0) && (model->B3SOIPDrbsh == 0.0))
                     { 
                        printf("\n      Warning: model parameter rbody=0!\n");
                        model->B3SOIPDrbody = 1e0;
                        here->B3SOIPDbodyMod = 1;
                     }
                     else { /* nonideal body tie */
                        here->B3SOIPDbodyMod = 1;
                     }
                   }
                 }
              }
            }


            if ((model->B3SOIPDshMod == 1) && (here->B3SOIPDrth0!=0))
            {
               if (here->B3SOIPDtempNode == -1) {
	          error = CKTmkVolt(ckt,&tmp,here->B3SOIPDname,"Temp");
                  if (error) return(error);
                     here->B3SOIPDtempNode = tmp->number;
               }

            } else {
                here->B3SOIPDtempNode = 0;
            }

/* here for debugging purpose only */
            if (here->B3SOIPDdebugMod != 0)
            {
               /* The real Vbs value */
               error = CKTmkVolt(ckt, &tmp, here->B3SOIPDname, "Vbs");
               if(error) return(error);
               here->B3SOIPDvbsNode = tmp->number;   

               error = CKTmkVolt(ckt, &tmp, here->B3SOIPDname, "Ids");
               if(error) return(error);
               here->B3SOIPDidsNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIPDname, "Ic");
               if(error) return(error);
               here->B3SOIPDicNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIPDname, "Ibs");
               if(error) return(error);
               here->B3SOIPDibsNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIPDname, "Ibd");
               if(error) return(error);
               here->B3SOIPDibdNode = tmp->number;
       
               error = CKTmkVolt(ckt, &tmp, here->B3SOIPDname, "Iii");
               if(error) return(error);
               here->B3SOIPDiiiNode = tmp->number;
     
               error = CKTmkVolt(ckt, &tmp, here->B3SOIPDname, "Ig");
               if(error) return(error);
               here->B3SOIPDigNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIPDname, "Gigg");
               if(error) return(error);
               here->B3SOIPDgiggNode = tmp->number;
               
               error = CKTmkVolt(ckt, &tmp, here->B3SOIPDname, "Gigd");
               if(error) return(error);
               here->B3SOIPDgigdNode = tmp->number;
              
               error = CKTmkVolt(ckt, &tmp, here->B3SOIPDname, "Gigb");
               if(error) return(error);
               here->B3SOIPDgigbNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIPDname, "Igidl");
               if(error) return(error);
               here->B3SOIPDigidlNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIPDname, "Itun");
               if(error) return(error);
               here->B3SOIPDitunNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIPDname, "Ibp");
               if(error) return(error);
               here->B3SOIPDibpNode = tmp->number;
       
               error = CKTmkVolt(ckt, &tmp, here->B3SOIPDname, "Cbb");
               if(error) return(error);
               here->B3SOIPDcbbNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIPDname, "Cbd");
               if(error) return(error);
               here->B3SOIPDcbdNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIPDname, "Cbg");
               if(error) return(error);
               here->B3SOIPDcbgNode = tmp->number;


               error = CKTmkVolt(ckt, &tmp, here->B3SOIPDname, "Qbf");
               if(error) return(error);
               here->B3SOIPDqbfNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIPDname, "Qjs");
               if(error) return(error);
               here->B3SOIPDqjsNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B3SOIPDname, "Qjd");
               if(error) return(error);
               here->B3SOIPDqjdNode = tmp->number;

           }

        /* set Sparse Matrix Pointers */

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)


        if ((model->B3SOIPDshMod == 1) && (here->B3SOIPDrth0!=0.0)) {
            TSTALLOC(B3SOIPDTemptempPtr, B3SOIPDtempNode, B3SOIPDtempNode);
            TSTALLOC(B3SOIPDTempdpPtr, B3SOIPDtempNode, B3SOIPDdNodePrime);
            TSTALLOC(B3SOIPDTempspPtr, B3SOIPDtempNode, B3SOIPDsNodePrime);
            TSTALLOC(B3SOIPDTempgPtr, B3SOIPDtempNode, B3SOIPDgNode);
            TSTALLOC(B3SOIPDTempbPtr, B3SOIPDtempNode, B3SOIPDbNode);

            TSTALLOC(B3SOIPDGtempPtr, B3SOIPDgNode, B3SOIPDtempNode);
            TSTALLOC(B3SOIPDDPtempPtr, B3SOIPDdNodePrime, B3SOIPDtempNode);
            TSTALLOC(B3SOIPDSPtempPtr, B3SOIPDsNodePrime, B3SOIPDtempNode);
            TSTALLOC(B3SOIPDEtempPtr, B3SOIPDeNode, B3SOIPDtempNode);
            TSTALLOC(B3SOIPDBtempPtr, B3SOIPDbNode, B3SOIPDtempNode);

            if (here->B3SOIPDbodyMod == 1) {
                TSTALLOC(B3SOIPDPtempPtr, B3SOIPDpNode, B3SOIPDtempNode);
            }
        }
        if (here->B3SOIPDbodyMod == 2) {
            /* Don't create any Jacobian entry for pNode */
        }
        else if (here->B3SOIPDbodyMod == 1) {
            TSTALLOC(B3SOIPDBpPtr, B3SOIPDbNode, B3SOIPDpNode);
            TSTALLOC(B3SOIPDPbPtr, B3SOIPDpNode, B3SOIPDbNode);
            TSTALLOC(B3SOIPDPpPtr, B3SOIPDpNode, B3SOIPDpNode);
        }

        TSTALLOC(B3SOIPDEbPtr, B3SOIPDeNode, B3SOIPDbNode);
        TSTALLOC(B3SOIPDGbPtr, B3SOIPDgNode, B3SOIPDbNode);
        TSTALLOC(B3SOIPDDPbPtr, B3SOIPDdNodePrime, B3SOIPDbNode);
        TSTALLOC(B3SOIPDSPbPtr, B3SOIPDsNodePrime, B3SOIPDbNode);
        TSTALLOC(B3SOIPDBePtr, B3SOIPDbNode, B3SOIPDeNode);
        TSTALLOC(B3SOIPDBgPtr, B3SOIPDbNode, B3SOIPDgNode);
        TSTALLOC(B3SOIPDBdpPtr, B3SOIPDbNode, B3SOIPDdNodePrime);
        TSTALLOC(B3SOIPDBspPtr, B3SOIPDbNode, B3SOIPDsNodePrime);
        TSTALLOC(B3SOIPDBbPtr, B3SOIPDbNode, B3SOIPDbNode);

        TSTALLOC(B3SOIPDEgPtr, B3SOIPDeNode, B3SOIPDgNode);
        TSTALLOC(B3SOIPDEdpPtr, B3SOIPDeNode, B3SOIPDdNodePrime);
        TSTALLOC(B3SOIPDEspPtr, B3SOIPDeNode, B3SOIPDsNodePrime);
        TSTALLOC(B3SOIPDGePtr, B3SOIPDgNode, B3SOIPDeNode);
        TSTALLOC(B3SOIPDDPePtr, B3SOIPDdNodePrime, B3SOIPDeNode);
        TSTALLOC(B3SOIPDSPePtr, B3SOIPDsNodePrime, B3SOIPDeNode);

        TSTALLOC(B3SOIPDEePtr, B3SOIPDeNode, B3SOIPDeNode);

        TSTALLOC(B3SOIPDGgPtr, B3SOIPDgNode, B3SOIPDgNode);
        TSTALLOC(B3SOIPDGdpPtr, B3SOIPDgNode, B3SOIPDdNodePrime);
        TSTALLOC(B3SOIPDGspPtr, B3SOIPDgNode, B3SOIPDsNodePrime);

        TSTALLOC(B3SOIPDDPgPtr, B3SOIPDdNodePrime, B3SOIPDgNode);
        TSTALLOC(B3SOIPDDPdpPtr, B3SOIPDdNodePrime, B3SOIPDdNodePrime);
        TSTALLOC(B3SOIPDDPspPtr, B3SOIPDdNodePrime, B3SOIPDsNodePrime);
        TSTALLOC(B3SOIPDDPdPtr, B3SOIPDdNodePrime, B3SOIPDdNode);

        TSTALLOC(B3SOIPDSPgPtr, B3SOIPDsNodePrime, B3SOIPDgNode);
        TSTALLOC(B3SOIPDSPdpPtr, B3SOIPDsNodePrime, B3SOIPDdNodePrime);
        TSTALLOC(B3SOIPDSPspPtr, B3SOIPDsNodePrime, B3SOIPDsNodePrime);
        TSTALLOC(B3SOIPDSPsPtr, B3SOIPDsNodePrime, B3SOIPDsNode);

        TSTALLOC(B3SOIPDDdPtr, B3SOIPDdNode, B3SOIPDdNode);
        TSTALLOC(B3SOIPDDdpPtr, B3SOIPDdNode, B3SOIPDdNodePrime);

        TSTALLOC(B3SOIPDSsPtr, B3SOIPDsNode, B3SOIPDsNode);
        TSTALLOC(B3SOIPDSspPtr, B3SOIPDsNode, B3SOIPDsNodePrime);

/* here for debugging purpose only */
         if (here->B3SOIPDdebugMod != 0)
         {
            TSTALLOC(B3SOIPDVbsPtr, B3SOIPDvbsNode, B3SOIPDvbsNode);
            TSTALLOC(B3SOIPDIdsPtr, B3SOIPDidsNode, B3SOIPDidsNode);
            TSTALLOC(B3SOIPDIcPtr, B3SOIPDicNode, B3SOIPDicNode);
            TSTALLOC(B3SOIPDIbsPtr, B3SOIPDibsNode, B3SOIPDibsNode);
            TSTALLOC(B3SOIPDIbdPtr, B3SOIPDibdNode, B3SOIPDibdNode);
            TSTALLOC(B3SOIPDIiiPtr, B3SOIPDiiiNode, B3SOIPDiiiNode);
            TSTALLOC(B3SOIPDIgPtr, B3SOIPDigNode, B3SOIPDigNode);
            TSTALLOC(B3SOIPDGiggPtr, B3SOIPDgiggNode, B3SOIPDgiggNode);
            TSTALLOC(B3SOIPDGigdPtr, B3SOIPDgigdNode, B3SOIPDgigdNode);
            TSTALLOC(B3SOIPDGigbPtr, B3SOIPDgigbNode, B3SOIPDgigbNode);
            TSTALLOC(B3SOIPDIgidlPtr, B3SOIPDigidlNode, B3SOIPDigidlNode);
            TSTALLOC(B3SOIPDItunPtr, B3SOIPDitunNode, B3SOIPDitunNode);
            TSTALLOC(B3SOIPDIbpPtr, B3SOIPDibpNode, B3SOIPDibpNode);
            TSTALLOC(B3SOIPDCbbPtr, B3SOIPDcbbNode, B3SOIPDcbbNode);
            TSTALLOC(B3SOIPDCbdPtr, B3SOIPDcbdNode, B3SOIPDcbdNode);
            TSTALLOC(B3SOIPDCbgPtr, B3SOIPDcbgNode, B3SOIPDcbgNode);
            TSTALLOC(B3SOIPDQbfPtr, B3SOIPDqbfNode, B3SOIPDqbfNode);
            TSTALLOC(B3SOIPDQjsPtr, B3SOIPDqjsNode, B3SOIPDqjsNode);
            TSTALLOC(B3SOIPDQjdPtr, B3SOIPDqjdNode, B3SOIPDqjdNode);

         }

        }
    }
    return(OK);
}  

int
B3SOIPDunsetup(
    GENmodel *inModel,
    CKTcircuit *ckt)
{
    B3SOIPDmodel *model;
    B3SOIPDinstance *here;
 
    for (model = (B3SOIPDmodel *)inModel; model != NULL;
            model = B3SOIPDnextModel(model))
    {
        for (here = B3SOIPDinstances(model); here != NULL;
                here=B3SOIPDnextInstance(here))
        {
            /* here for debugging purpose only */
            if (here->B3SOIPDqjdNode > 0)
                CKTdltNNum(ckt, here->B3SOIPDqjdNode);
            here->B3SOIPDqjdNode = 0;

            if (here->B3SOIPDqjsNode > 0)
                CKTdltNNum(ckt, here->B3SOIPDqjsNode);
            here->B3SOIPDqjsNode = 0;

            if (here->B3SOIPDqbfNode > 0)
                CKTdltNNum(ckt, here->B3SOIPDqbfNode);
            here->B3SOIPDqbfNode = 0;

            if (here->B3SOIPDcbgNode > 0)
                CKTdltNNum(ckt, here->B3SOIPDcbgNode);
            here->B3SOIPDcbgNode = 0;

            if (here->B3SOIPDcbdNode > 0)
                CKTdltNNum(ckt, here->B3SOIPDcbdNode);
            here->B3SOIPDcbdNode = 0;

            if (here->B3SOIPDcbbNode > 0)
                CKTdltNNum(ckt, here->B3SOIPDcbbNode);
            here->B3SOIPDcbbNode = 0;

            if (here->B3SOIPDibpNode > 0)
                CKTdltNNum(ckt, here->B3SOIPDibpNode);
            here->B3SOIPDibpNode = 0;

            if (here->B3SOIPDitunNode > 0)
                CKTdltNNum(ckt, here->B3SOIPDitunNode);
            here->B3SOIPDitunNode = 0;

            if (here->B3SOIPDigidlNode > 0)
                CKTdltNNum(ckt, here->B3SOIPDigidlNode);
            here->B3SOIPDigidlNode = 0;

            if (here->B3SOIPDgigbNode > 0)
                CKTdltNNum(ckt, here->B3SOIPDgigbNode);
            here->B3SOIPDgigbNode = 0;

            if (here->B3SOIPDgigdNode > 0)
                CKTdltNNum(ckt, here->B3SOIPDgigdNode);
            here->B3SOIPDgigdNode = 0;

            if (here->B3SOIPDgiggNode > 0)
                CKTdltNNum(ckt, here->B3SOIPDgiggNode);
            here->B3SOIPDgiggNode = 0;

            if (here->B3SOIPDigNode > 0)
                CKTdltNNum(ckt, here->B3SOIPDigNode);
            here->B3SOIPDigNode = 0;

            if (here->B3SOIPDiiiNode > 0)
                CKTdltNNum(ckt, here->B3SOIPDiiiNode);
            here->B3SOIPDiiiNode = 0;

            if (here->B3SOIPDibdNode > 0)
                CKTdltNNum(ckt, here->B3SOIPDibdNode);
            here->B3SOIPDibdNode = 0;

            if (here->B3SOIPDibsNode > 0)
                CKTdltNNum(ckt, here->B3SOIPDibsNode);
            here->B3SOIPDibsNode = 0;

            if (here->B3SOIPDicNode > 0)
                CKTdltNNum(ckt, here->B3SOIPDicNode);
            here->B3SOIPDicNode = 0;

            if (here->B3SOIPDidsNode > 0)
                CKTdltNNum(ckt, here->B3SOIPDidsNode);
            here->B3SOIPDidsNode = 0;

            if (here->B3SOIPDvbsNode > 0)
                CKTdltNNum(ckt, here->B3SOIPDvbsNode);
            here->B3SOIPDvbsNode = 0;


            if (here->B3SOIPDtempNode > 0 &&
                here->B3SOIPDtempNode != here->B3SOIPDtempNodeExt &&
                here->B3SOIPDtempNode != here->B3SOIPDbNodeExt &&
                here->B3SOIPDtempNode != here->B3SOIPDpNodeExt)
                CKTdltNNum(ckt, here->B3SOIPDtempNode);
            here->B3SOIPDtempNode = 0;

            if (here->B3SOIPDbNode > 0 &&
                here->B3SOIPDbNode != here->B3SOIPDbNodeExt &&
                here->B3SOIPDbNode != here->B3SOIPDpNodeExt)
                CKTdltNNum(ckt, here->B3SOIPDbNode);
            here->B3SOIPDbNode = 0;

            here->B3SOIPDpNode = 0;

            if (here->B3SOIPDsNodePrime > 0
                    && here->B3SOIPDsNodePrime != here->B3SOIPDsNode)
                CKTdltNNum(ckt, here->B3SOIPDsNodePrime);
            here->B3SOIPDsNodePrime = 0;

            if (here->B3SOIPDdNodePrime > 0
                    && here->B3SOIPDdNodePrime != here->B3SOIPDdNode)
                CKTdltNNum(ckt, here->B3SOIPDdNodePrime);
            here->B3SOIPDdNodePrime = 0;
        }
    }
    return OK;
}

