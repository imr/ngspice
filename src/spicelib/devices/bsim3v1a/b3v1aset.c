/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1aset.c
**********/

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "bsim3v1adef.h"
#include "const.h"
#include "sperror.h"
#include "suffix.h"

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
BSIM3v1Asetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt,
              int *states)
{
BSIM3v1Amodel *model = (BSIM3v1Amodel*)inModel;
BSIM3v1Ainstance *here;
int error;
CKTnode *tmp;

CKTnode *tmpNode;
IFuid tmpName;

    /*  loop through all the BSIM3v1A device models */
    for( ; model != NULL; model = model->BSIM3v1AnextModel )
    {
/* Default value Processing for BSIM3v1A MOSFET Models */
        if (!model->BSIM3v1AtypeGiven)
            model->BSIM3v1Atype = NMOS;     
        if (!model->BSIM3v1AmobModGiven) 
            model->BSIM3v1AmobMod = 1;
        if (!model->BSIM3v1AbinUnitGiven) 
            model->BSIM3v1AbinUnit = 1;
        if (!model->BSIM3v1AcapModGiven) 
            model->BSIM3v1AcapMod = 1;
        if (!model->BSIM3v1AnqsModGiven) 
            model->BSIM3v1AnqsMod = 0;
        if (!model->BSIM3v1AnoiModGiven) 
            model->BSIM3v1AnoiMod = 1;
        if (!model->BSIM3v1AtoxGiven)
            model->BSIM3v1Atox = 150.0e-10;
        model->BSIM3v1Acox = 3.453133e-11 / model->BSIM3v1Atox;

        if (!model->BSIM3v1AcdscGiven)
	    model->BSIM3v1Acdsc = 2.4e-4;   /* unit Q/V/m^2  */
        if (!model->BSIM3v1AcdscbGiven)
	    model->BSIM3v1Acdscb = 0.0;   /* unit Q/V/m^2  */    
	    if (!model->BSIM3v1AcdscdGiven)
	    model->BSIM3v1Acdscd = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM3v1AcitGiven)
	    model->BSIM3v1Acit = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM3v1AnfactorGiven)
	    model->BSIM3v1Anfactor = 1;
        if (!model->BSIM3v1AxjGiven)
            model->BSIM3v1Axj = .15e-6;
        if (!model->BSIM3v1AvsatGiven)
            model->BSIM3v1Avsat = 8.0e4;    /* unit m/s */ 
        if (!model->BSIM3v1AatGiven)
            model->BSIM3v1Aat = 3.3e4;    /* unit m/s */ 
        if (!model->BSIM3v1Aa0Given)
            model->BSIM3v1Aa0 = 1.0;  
        if (!model->BSIM3v1AagsGiven)
            model->BSIM3v1Aags = 0.0;
        if (!model->BSIM3v1Aa1Given)
            model->BSIM3v1Aa1 = 0.0;
        if (!model->BSIM3v1Aa2Given)
            model->BSIM3v1Aa2 = 1.0;
        if (!model->BSIM3v1AketaGiven)
            model->BSIM3v1Aketa = -0.047;    /* unit  / V */
        if (!model->BSIM3v1AnsubGiven)
            model->BSIM3v1Ansub = 6.0e16;   /* unit 1/cm3 */
        if (!model->BSIM3v1AnpeakGiven)
            model->BSIM3v1Anpeak = 1.7e17;   /* unit 1/cm3 */
        if (!model->BSIM3v1AvbmGiven)
	    model->BSIM3v1Avbm = -5.0;
        if (!model->BSIM3v1AxtGiven)
	    model->BSIM3v1Axt = 1.55e-7;
        if (!model->BSIM3v1Akt1Given)
            model->BSIM3v1Akt1 = -0.11;      /* unit V */
        if (!model->BSIM3v1Akt1lGiven)
            model->BSIM3v1Akt1l = 0.0;      /* unit V*m */
        if (!model->BSIM3v1Akt2Given)
            model->BSIM3v1Akt2 = 0.022;      /* No unit */
        if (!model->BSIM3v1Ak3Given)
            model->BSIM3v1Ak3 = 80.0;      
        if (!model->BSIM3v1Ak3bGiven)
            model->BSIM3v1Ak3b = 0.0;      
        if (!model->BSIM3v1Aw0Given)
            model->BSIM3v1Aw0 = 2.5e-6;    
        if (!model->BSIM3v1AnlxGiven)
            model->BSIM3v1Anlx = 1.74e-7;     
        if (!model->BSIM3v1Advt0Given)
            model->BSIM3v1Advt0 = 2.2;    
        if (!model->BSIM3v1Advt1Given)
            model->BSIM3v1Advt1 = 0.53;      
        if (!model->BSIM3v1Advt2Given)
            model->BSIM3v1Advt2 = -0.032;   /* unit 1 / V */     

        if (!model->BSIM3v1Advt0wGiven)
            model->BSIM3v1Advt0w = 0.0;    
        if (!model->BSIM3v1Advt1wGiven)
            model->BSIM3v1Advt1w = 5.3e6;    
        if (!model->BSIM3v1Advt2wGiven)
            model->BSIM3v1Advt2w = -0.032;   

        if (!model->BSIM3v1AdroutGiven)
            model->BSIM3v1Adrout = 0.56;     
        if (!model->BSIM3v1AdsubGiven)
            model->BSIM3v1Adsub = model->BSIM3v1Adrout;     
        if (!model->BSIM3v1Avth0Given)
            model->BSIM3v1Avth0 = (model->BSIM3v1Atype == NMOS) ? 0.7 : -0.7;
        if (!model->BSIM3v1AuaGiven)
            model->BSIM3v1Aua = 2.25e-9;      /* unit m/V */
        if (!model->BSIM3v1Aua1Given)
            model->BSIM3v1Aua1 = 4.31e-9;      /* unit m/V */
        if (!model->BSIM3v1AubGiven)
            model->BSIM3v1Aub = 5.87e-19;     /* unit (m/V)**2 */
        if (!model->BSIM3v1Aub1Given)
            model->BSIM3v1Aub1 = -7.61e-18;     /* unit (m/V)**2 */
        if (!model->BSIM3v1AucGiven)
            model->BSIM3v1Auc = (model->BSIM3v1AmobMod == 3) ? -0.0465 : -0.0465e-9;   
        if (!model->BSIM3v1Auc1Given)
            model->BSIM3v1Auc1 = (model->BSIM3v1AmobMod == 3) ? -0.056 : -0.056e-9;   
        if (!model->BSIM3v1Au0Given)
            model->BSIM3v1Au0 = (model->BSIM3v1Atype == NMOS) ? 0.067 : 0.025;
        else if (model->BSIM3v1Au0 > 1.0)
            model->BSIM3v1Au0 = model->BSIM3v1Au0 / 1.0e4;  
            /* if u0 > 1.0, cm.g.sec unit system is used.  If should be
               converted into SI unit system.  */
        if (!model->BSIM3v1AuteGiven)
	    model->BSIM3v1Aute = -1.5;    
        if (!model->BSIM3v1AvoffGiven)
	    model->BSIM3v1Avoff = -0.08;
        if (!model->BSIM3v1AdeltaGiven)  
           model->BSIM3v1Adelta = 0.01;
        if (!model->BSIM3v1ArdswGiven)
            model->BSIM3v1Ardsw = 0;      
        if (!model->BSIM3v1AprwgGiven)
            model->BSIM3v1Aprwg = 0.0;      /* unit 1/V */
        if (!model->BSIM3v1AprwbGiven)
            model->BSIM3v1Aprwb = 0.0;      
        if (!model->BSIM3v1AprtGiven)
        if (!model->BSIM3v1AprtGiven)
            model->BSIM3v1Aprt = 0.0;      
        if (!model->BSIM3v1Aeta0Given)
            model->BSIM3v1Aeta0 = 0.08;      /* no unit  */ 
        if (!model->BSIM3v1AetabGiven)
            model->BSIM3v1Aetab = -0.07;      /* unit  1/V */ 
        if (!model->BSIM3v1ApclmGiven)
            model->BSIM3v1Apclm = 1.3;      /* no unit  */ 
        if (!model->BSIM3v1Apdibl1Given)
            model->BSIM3v1Apdibl1 = .39;    /* no unit  */
        if (!model->BSIM3v1Apdibl2Given)
            model->BSIM3v1Apdibl2 = 0.0086;    /* no unit  */ 
        if (!model->BSIM3v1ApdiblbGiven)
            model->BSIM3v1Apdiblb = 0.0;    /* 1/V  */ 
        if (!model->BSIM3v1Apscbe1Given)
            model->BSIM3v1Apscbe1 = 4.24e8;     
        if (!model->BSIM3v1Apscbe2Given)
            model->BSIM3v1Apscbe2 = 1.0e-5;    
        if (!model->BSIM3v1ApvagGiven)
            model->BSIM3v1Apvag = 0.0;     
        if (!model->BSIM3v1AwrGiven)  
            model->BSIM3v1Awr = 1.0;
        if (!model->BSIM3v1AdwgGiven)  
            model->BSIM3v1Adwg = 0.0;
        if (!model->BSIM3v1AdwbGiven)  
            model->BSIM3v1Adwb = 0.0;
        if (!model->BSIM3v1Ab0Given)
            model->BSIM3v1Ab0 = 0.0;
        if (!model->BSIM3v1Ab1Given)  
            model->BSIM3v1Ab1 = 0.0;
        if (!model->BSIM3v1Aalpha0Given)  
            model->BSIM3v1Aalpha0 = 0.0;
        if (!model->BSIM3v1Abeta0Given)  
            model->BSIM3v1Abeta0 = 30.0;

        if (!model->BSIM3v1AelmGiven)  
            model->BSIM3v1Aelm = 5.0;
        if (!model->BSIM3v1AcgslGiven)  
            model->BSIM3v1Acgsl = 0.0;
        if (!model->BSIM3v1AcgdlGiven)  
            model->BSIM3v1Acgdl = 0.0;
        if (!model->BSIM3v1AckappaGiven)  
            model->BSIM3v1Ackappa = 0.6;
        if (!model->BSIM3v1AclcGiven)  
            model->BSIM3v1Aclc = 0.1e-6;
        if (!model->BSIM3v1AcleGiven)  
            model->BSIM3v1Acle = 0.6;

	/* Length dependence */
        if (!model->BSIM3v1AlcdscGiven)
	    model->BSIM3v1Alcdsc = 0.0;
        if (!model->BSIM3v1AlcdscbGiven)
	    model->BSIM3v1Alcdscb = 0.0;
	    if (!model->BSIM3v1AlcdscdGiven) 
	    model->BSIM3v1Alcdscd = 0.0;
        if (!model->BSIM3v1AlcitGiven)
	    model->BSIM3v1Alcit = 0.0;
        if (!model->BSIM3v1AlnfactorGiven)
	    model->BSIM3v1Alnfactor = 0.0;
        if (!model->BSIM3v1AlxjGiven)
            model->BSIM3v1Alxj = 0.0;
        if (!model->BSIM3v1AlvsatGiven)
            model->BSIM3v1Alvsat = 0.0;
        if (!model->BSIM3v1AlatGiven)
            model->BSIM3v1Alat = 0.0;
        if (!model->BSIM3v1Ala0Given)
            model->BSIM3v1Ala0 = 0.0; 
        if (!model->BSIM3v1AlagsGiven)
            model->BSIM3v1Alags = 0.0;
        if (!model->BSIM3v1Ala1Given)
            model->BSIM3v1Ala1 = 0.0;
        if (!model->BSIM3v1Ala2Given)
            model->BSIM3v1Ala2 = 0.0;
        if (!model->BSIM3v1AlketaGiven)
            model->BSIM3v1Alketa = 0.0;
        if (!model->BSIM3v1AlnsubGiven)
            model->BSIM3v1Alnsub = 0.0;
        if (!model->BSIM3v1AlnpeakGiven)
            model->BSIM3v1Alnpeak = 0.0;
        if (!model->BSIM3v1AlvbmGiven)
	    model->BSIM3v1Alvbm = 0.0;
        if (!model->BSIM3v1AlxtGiven)
	    model->BSIM3v1Alxt = 0.0;
        if (!model->BSIM3v1Alkt1Given)
            model->BSIM3v1Alkt1 = 0.0; 
        if (!model->BSIM3v1Alkt1lGiven)
            model->BSIM3v1Alkt1l = 0.0;
        if (!model->BSIM3v1Alkt2Given)
            model->BSIM3v1Alkt2 = 0.0;
        if (!model->BSIM3v1Alk3Given)
            model->BSIM3v1Alk3 = 0.0;      
        if (!model->BSIM3v1Alk3bGiven)
            model->BSIM3v1Alk3b = 0.0;      
        if (!model->BSIM3v1Alw0Given)
            model->BSIM3v1Alw0 = 0.0;    
        if (!model->BSIM3v1AlnlxGiven)
            model->BSIM3v1Alnlx = 0.0;     
        if (!model->BSIM3v1Aldvt0Given)
            model->BSIM3v1Aldvt0 = 0.0;    
        if (!model->BSIM3v1Aldvt1Given)
            model->BSIM3v1Aldvt1 = 0.0;      
        if (!model->BSIM3v1Aldvt2Given)
            model->BSIM3v1Aldvt2 = 0.0;
        if (!model->BSIM3v1Aldvt0wGiven)
            model->BSIM3v1Aldvt0w = 0.0;    
        if (!model->BSIM3v1Aldvt1wGiven)
            model->BSIM3v1Aldvt1w = 0.0;      
        if (!model->BSIM3v1Aldvt2wGiven)
            model->BSIM3v1Aldvt2w = 0.0;
        if (!model->BSIM3v1AldroutGiven)
            model->BSIM3v1Aldrout = 0.0;     
        if (!model->BSIM3v1AldsubGiven)
            model->BSIM3v1Aldsub = 0.0;
        if (!model->BSIM3v1Alvth0Given)
           model->BSIM3v1Alvth0 = 0.0;
        if (!model->BSIM3v1AluaGiven)
            model->BSIM3v1Alua = 0.0;
        if (!model->BSIM3v1Alua1Given)
            model->BSIM3v1Alua1 = 0.0;
        if (!model->BSIM3v1AlubGiven)
            model->BSIM3v1Alub = 0.0;
        if (!model->BSIM3v1Alub1Given)
            model->BSIM3v1Alub1 = 0.0;
        if (!model->BSIM3v1AlucGiven)
            model->BSIM3v1Aluc = 0.0;
        if (!model->BSIM3v1Aluc1Given)
            model->BSIM3v1Aluc1 = 0.0;
        if (!model->BSIM3v1Alu0Given)
            model->BSIM3v1Alu0 = 0.0;
        else if (model->BSIM3v1Au0 > 1.0)
            model->BSIM3v1Alu0 = model->BSIM3v1Alu0 / 1.0e4;  
        if (!model->BSIM3v1AluteGiven)
	    model->BSIM3v1Alute = 0.0;    
        if (!model->BSIM3v1AlvoffGiven)
	    model->BSIM3v1Alvoff = 0.0;
        if (!model->BSIM3v1AldeltaGiven)  
            model->BSIM3v1Aldelta = 0.0;
        if (!model->BSIM3v1AlrdswGiven)
            model->BSIM3v1Alrdsw = 0.0;
        if (!model->BSIM3v1AlprwbGiven)
            model->BSIM3v1Alprwb = 0.0;
        if (!model->BSIM3v1AlprwgGiven)
            model->BSIM3v1Alprwg = 0.0;
        if (!model->BSIM3v1AlprtGiven)
        if (!model->BSIM3v1AlprtGiven)
            model->BSIM3v1Alprt = 0.0;
        if (!model->BSIM3v1Aleta0Given)
            model->BSIM3v1Aleta0 = 0.0;
        if (!model->BSIM3v1AletabGiven)
            model->BSIM3v1Aletab = -0.0;
        if (!model->BSIM3v1AlpclmGiven)
            model->BSIM3v1Alpclm = 0.0; 
        if (!model->BSIM3v1Alpdibl1Given)
            model->BSIM3v1Alpdibl1 = 0.0;
        if (!model->BSIM3v1Alpdibl2Given)
            model->BSIM3v1Alpdibl2 = 0.0;
        if (!model->BSIM3v1AlpdiblbGiven)
            model->BSIM3v1Alpdiblb = 0.0;
        if (!model->BSIM3v1Alpscbe1Given)
            model->BSIM3v1Alpscbe1 = 0.0;
        if (!model->BSIM3v1Alpscbe2Given)
            model->BSIM3v1Alpscbe2 = 0.0;
        if (!model->BSIM3v1AlpvagGiven)
            model->BSIM3v1Alpvag = 0.0;     
        if (!model->BSIM3v1AlwrGiven)  
            model->BSIM3v1Alwr = 0.0;
        if (!model->BSIM3v1AldwgGiven)  
            model->BSIM3v1Aldwg = 0.0;
        if (!model->BSIM3v1AldwbGiven)  
            model->BSIM3v1Aldwb = 0.0;
        if (!model->BSIM3v1Alb0Given)
            model->BSIM3v1Alb0 = 0.0;
        if (!model->BSIM3v1Alb1Given)  
            model->BSIM3v1Alb1 = 0.0;
        if (!model->BSIM3v1Alalpha0Given)  
            model->BSIM3v1Alalpha0 = 0.0;
        if (!model->BSIM3v1Albeta0Given)  
            model->BSIM3v1Albeta0 = 0.0;

        if (!model->BSIM3v1AlelmGiven)  
            model->BSIM3v1Alelm = 0.0;
        if (!model->BSIM3v1AlcgslGiven)  
            model->BSIM3v1Alcgsl = 0.0;
        if (!model->BSIM3v1AlcgdlGiven)  
            model->BSIM3v1Alcgdl = 0.0;
        if (!model->BSIM3v1AlckappaGiven)  
            model->BSIM3v1Alckappa = 0.0;
        if (!model->BSIM3v1AlcfGiven)  
            model->BSIM3v1Alcf = 0.0;
        if (!model->BSIM3v1AlclcGiven)  
            model->BSIM3v1Alclc = 0.0;
        if (!model->BSIM3v1AlcleGiven)  
            model->BSIM3v1Alcle = 0.0;

	/* Width dependence */
        if (!model->BSIM3v1AwcdscGiven)
	    model->BSIM3v1Awcdsc = 0.0;
        if (!model->BSIM3v1AwcdscbGiven)
	    model->BSIM3v1Awcdscb = 0.0;  
	    if (!model->BSIM3v1AwcdscdGiven)
	    model->BSIM3v1Awcdscd = 0.0;
        if (!model->BSIM3v1AwcitGiven)
	    model->BSIM3v1Awcit = 0.0;
        if (!model->BSIM3v1AwnfactorGiven)
	    model->BSIM3v1Awnfactor = 0.0;
        if (!model->BSIM3v1AwxjGiven)
            model->BSIM3v1Awxj = 0.0;
        if (!model->BSIM3v1AwvsatGiven)
            model->BSIM3v1Awvsat = 0.0;
        if (!model->BSIM3v1AwatGiven)
            model->BSIM3v1Awat = 0.0;
        if (!model->BSIM3v1Awa0Given)
            model->BSIM3v1Awa0 = 0.0; 
        if (!model->BSIM3v1AwagsGiven)
            model->BSIM3v1Awags = 0.0;
        if (!model->BSIM3v1Awa1Given)
            model->BSIM3v1Awa1 = 0.0;
        if (!model->BSIM3v1Awa2Given)
            model->BSIM3v1Awa2 = 0.0;
        if (!model->BSIM3v1AwketaGiven)
            model->BSIM3v1Awketa = 0.0;
        if (!model->BSIM3v1AwnsubGiven)
            model->BSIM3v1Awnsub = 0.0;
        if (!model->BSIM3v1AwnpeakGiven)
            model->BSIM3v1Awnpeak = 0.0;
        if (!model->BSIM3v1AwvbmGiven)
	    model->BSIM3v1Awvbm = 0.0;
        if (!model->BSIM3v1AwxtGiven)
	    model->BSIM3v1Awxt = 0.0;
        if (!model->BSIM3v1Awkt1Given)
            model->BSIM3v1Awkt1 = 0.0; 
        if (!model->BSIM3v1Awkt1lGiven)
            model->BSIM3v1Awkt1l = 0.0;
        if (!model->BSIM3v1Awkt2Given)
            model->BSIM3v1Awkt2 = 0.0;
        if (!model->BSIM3v1Awk3Given)
            model->BSIM3v1Awk3 = 0.0;      
        if (!model->BSIM3v1Awk3bGiven)
            model->BSIM3v1Awk3b = 0.0;      
        if (!model->BSIM3v1Aww0Given)
            model->BSIM3v1Aww0 = 0.0;    
        if (!model->BSIM3v1AwnlxGiven)
            model->BSIM3v1Awnlx = 0.0;     
        if (!model->BSIM3v1Awdvt0Given)
            model->BSIM3v1Awdvt0 = 0.0;    
        if (!model->BSIM3v1Awdvt1Given)
            model->BSIM3v1Awdvt1 = 0.0;      
        if (!model->BSIM3v1Awdvt2Given)
            model->BSIM3v1Awdvt2 = 0.0;
        if (!model->BSIM3v1Awdvt0wGiven)
            model->BSIM3v1Awdvt0w = 0.0;    
        if (!model->BSIM3v1Awdvt1wGiven)
            model->BSIM3v1Awdvt1w = 0.0;      
        if (!model->BSIM3v1Awdvt2wGiven)
            model->BSIM3v1Awdvt2w = 0.0;
        if (!model->BSIM3v1AwdroutGiven)
            model->BSIM3v1Awdrout = 0.0;     
        if (!model->BSIM3v1AwdsubGiven)
            model->BSIM3v1Awdsub = 0.0;
        if (!model->BSIM3v1Awvth0Given)
           model->BSIM3v1Awvth0 = 0.0;
        if (!model->BSIM3v1AwuaGiven)
            model->BSIM3v1Awua = 0.0;
        if (!model->BSIM3v1Awua1Given)
            model->BSIM3v1Awua1 = 0.0;
        if (!model->BSIM3v1AwubGiven)
            model->BSIM3v1Awub = 0.0;
        if (!model->BSIM3v1Awub1Given)
            model->BSIM3v1Awub1 = 0.0;
        if (!model->BSIM3v1AwucGiven)
            model->BSIM3v1Awuc = 0.0;
        if (!model->BSIM3v1Awuc1Given)
            model->BSIM3v1Awuc1 = 0.0;
        if (!model->BSIM3v1Awu0Given)
            model->BSIM3v1Awu0 = 0.0;
        else if (model->BSIM3v1Au0 > 1.0)
            model->BSIM3v1Awu0 = model->BSIM3v1Awu0 / 1.0e4;
        if (!model->BSIM3v1AwuteGiven)
	    model->BSIM3v1Awute = 0.0;    
        if (!model->BSIM3v1AwvoffGiven)
	    model->BSIM3v1Awvoff = 0.0;
        if (!model->BSIM3v1AwdeltaGiven)  
            model->BSIM3v1Awdelta = 0.0;
        if (!model->BSIM3v1AwrdswGiven)
            model->BSIM3v1Awrdsw = 0.0;
        if (!model->BSIM3v1AwprwbGiven)
            model->BSIM3v1Awprwb = 0.0;
        if (!model->BSIM3v1AwprwgGiven)
            model->BSIM3v1Awprwg = 0.0;
        if (!model->BSIM3v1AwprtGiven)
            model->BSIM3v1Awprt = 0.0;
        if (!model->BSIM3v1Aweta0Given)
            model->BSIM3v1Aweta0 = 0.0;
        if (!model->BSIM3v1AwetabGiven)
            model->BSIM3v1Awetab = 0.0;
        if (!model->BSIM3v1AwpclmGiven)
            model->BSIM3v1Awpclm = 0.0; 
        if (!model->BSIM3v1Awpdibl1Given)
            model->BSIM3v1Awpdibl1 = 0.0;
        if (!model->BSIM3v1Awpdibl2Given)
            model->BSIM3v1Awpdibl2 = 0.0;
        if (!model->BSIM3v1AwpdiblbGiven)
            model->BSIM3v1Awpdiblb = 0.0;
        if (!model->BSIM3v1Awpscbe1Given)
            model->BSIM3v1Awpscbe1 = 0.0;
        if (!model->BSIM3v1Awpscbe2Given)
            model->BSIM3v1Awpscbe2 = 0.0;
        if (!model->BSIM3v1AwpvagGiven)
            model->BSIM3v1Awpvag = 0.0;     
        if (!model->BSIM3v1AwwrGiven)  
            model->BSIM3v1Awwr = 0.0;
        if (!model->BSIM3v1AwdwgGiven)  
            model->BSIM3v1Awdwg = 0.0;
        if (!model->BSIM3v1AwdwbGiven)  
            model->BSIM3v1Awdwb = 0.0;
        if (!model->BSIM3v1Awb0Given)
            model->BSIM3v1Awb0 = 0.0;
        if (!model->BSIM3v1Awb1Given)  
            model->BSIM3v1Awb1 = 0.0;
        if (!model->BSIM3v1Awalpha0Given)  
            model->BSIM3v1Awalpha0 = 0.0;
        if (!model->BSIM3v1Awbeta0Given)  
            model->BSIM3v1Awbeta0 = 0.0;

        if (!model->BSIM3v1AwelmGiven)  
            model->BSIM3v1Awelm = 0.0;
        if (!model->BSIM3v1AwcgslGiven)  
            model->BSIM3v1Awcgsl = 0.0;
        if (!model->BSIM3v1AwcgdlGiven)  
            model->BSIM3v1Awcgdl = 0.0;
        if (!model->BSIM3v1AwckappaGiven)  
            model->BSIM3v1Awckappa = 0.0;
        if (!model->BSIM3v1AwcfGiven)  
            model->BSIM3v1Awcf = 0.0;
        if (!model->BSIM3v1AwclcGiven)  
            model->BSIM3v1Awclc = 0.0;
        if (!model->BSIM3v1AwcleGiven)  
            model->BSIM3v1Awcle = 0.0;

	/* Cross-term dependence */
        if (!model->BSIM3v1ApcdscGiven)
	    model->BSIM3v1Apcdsc = 0.0;
        if (!model->BSIM3v1ApcdscbGiven)
	    model->BSIM3v1Apcdscb = 0.0;   
	    if (!model->BSIM3v1ApcdscdGiven)
	    model->BSIM3v1Apcdscd = 0.0;
        if (!model->BSIM3v1ApcitGiven)
	    model->BSIM3v1Apcit = 0.0;
        if (!model->BSIM3v1ApnfactorGiven)
	    model->BSIM3v1Apnfactor = 0.0;
        if (!model->BSIM3v1ApxjGiven)
            model->BSIM3v1Apxj = 0.0;
        if (!model->BSIM3v1ApvsatGiven)
            model->BSIM3v1Apvsat = 0.0;
        if (!model->BSIM3v1ApatGiven)
            model->BSIM3v1Apat = 0.0;
        if (!model->BSIM3v1Apa0Given)
            model->BSIM3v1Apa0 = 0.0; 
            
        if (!model->BSIM3v1ApagsGiven)
            model->BSIM3v1Apags = 0.0;
        if (!model->BSIM3v1Apa1Given)
            model->BSIM3v1Apa1 = 0.0;
        if (!model->BSIM3v1Apa2Given)
            model->BSIM3v1Apa2 = 0.0;
        if (!model->BSIM3v1ApketaGiven)
            model->BSIM3v1Apketa = 0.0;
        if (!model->BSIM3v1ApnsubGiven)
            model->BSIM3v1Apnsub = 0.0;
        if (!model->BSIM3v1ApnpeakGiven)
            model->BSIM3v1Apnpeak = 0.0;
        if (!model->BSIM3v1ApvbmGiven)
	    model->BSIM3v1Apvbm = 0.0;
        if (!model->BSIM3v1ApxtGiven)
	    model->BSIM3v1Apxt = 0.0;
        if (!model->BSIM3v1Apkt1Given)
            model->BSIM3v1Apkt1 = 0.0; 
        if (!model->BSIM3v1Apkt1lGiven)
            model->BSIM3v1Apkt1l = 0.0;
        if (!model->BSIM3v1Apkt2Given)
            model->BSIM3v1Apkt2 = 0.0;
        if (!model->BSIM3v1Apk3Given)
            model->BSIM3v1Apk3 = 0.0;      
        if (!model->BSIM3v1Apk3bGiven)
            model->BSIM3v1Apk3b = 0.0;      
        if (!model->BSIM3v1Apw0Given)
            model->BSIM3v1Apw0 = 0.0;    
        if (!model->BSIM3v1ApnlxGiven)
            model->BSIM3v1Apnlx = 0.0;     
        if (!model->BSIM3v1Apdvt0Given)
            model->BSIM3v1Apdvt0 = 0.0;    
        if (!model->BSIM3v1Apdvt1Given)
            model->BSIM3v1Apdvt1 = 0.0;      
        if (!model->BSIM3v1Apdvt2Given)
            model->BSIM3v1Apdvt2 = 0.0;
        if (!model->BSIM3v1Apdvt0wGiven)
            model->BSIM3v1Apdvt0w = 0.0;    
        if (!model->BSIM3v1Apdvt1wGiven)
            model->BSIM3v1Apdvt1w = 0.0;      
        if (!model->BSIM3v1Apdvt2wGiven)
            model->BSIM3v1Apdvt2w = 0.0;
        if (!model->BSIM3v1ApdroutGiven)
            model->BSIM3v1Apdrout = 0.0;     
        if (!model->BSIM3v1ApdsubGiven)
            model->BSIM3v1Apdsub = 0.0;
        if (!model->BSIM3v1Apvth0Given)
           model->BSIM3v1Apvth0 = 0.0;
        if (!model->BSIM3v1ApuaGiven)
            model->BSIM3v1Apua = 0.0;
        if (!model->BSIM3v1Apua1Given)
            model->BSIM3v1Apua1 = 0.0;
        if (!model->BSIM3v1ApubGiven)
            model->BSIM3v1Apub = 0.0;
        if (!model->BSIM3v1Apub1Given)
            model->BSIM3v1Apub1 = 0.0;
        if (!model->BSIM3v1ApucGiven)
            model->BSIM3v1Apuc = 0.0;
        if (!model->BSIM3v1Apuc1Given)
            model->BSIM3v1Apuc1 = 0.0;
        if (!model->BSIM3v1Apu0Given)
            model->BSIM3v1Apu0 = 0.0;
        else if (model->BSIM3v1Au0 > 1.0)
            model->BSIM3v1Apu0 = model->BSIM3v1Apu0 / 1.0e4;  
        if (!model->BSIM3v1AputeGiven)
	    model->BSIM3v1Apute = 0.0;    
        if (!model->BSIM3v1ApvoffGiven)
	    model->BSIM3v1Apvoff = 0.0;
        if (!model->BSIM3v1ApdeltaGiven)  
            model->BSIM3v1Apdelta = 0.0;
        if (!model->BSIM3v1AprdswGiven)
            model->BSIM3v1Aprdsw = 0.0;
        if (!model->BSIM3v1ApprwbGiven)
            model->BSIM3v1Apprwb = 0.0;
        if (!model->BSIM3v1ApprwgGiven)
            model->BSIM3v1Apprwg = 0.0;
        if (!model->BSIM3v1ApprtGiven)
            model->BSIM3v1Apprt = 0.0;
        if (!model->BSIM3v1Apeta0Given)
            model->BSIM3v1Apeta0 = 0.0;
        if (!model->BSIM3v1ApetabGiven)
            model->BSIM3v1Apetab = 0.0;
        if (!model->BSIM3v1AppclmGiven)
            model->BSIM3v1Appclm = 0.0; 
        if (!model->BSIM3v1Appdibl1Given)
            model->BSIM3v1Appdibl1 = 0.0;
        if (!model->BSIM3v1Appdibl2Given)
            model->BSIM3v1Appdibl2 = 0.0;
        if (!model->BSIM3v1AppdiblbGiven)
            model->BSIM3v1Appdiblb = 0.0;
        if (!model->BSIM3v1Appscbe1Given)
            model->BSIM3v1Appscbe1 = 0.0;
        if (!model->BSIM3v1Appscbe2Given)
            model->BSIM3v1Appscbe2 = 0.0;
        if (!model->BSIM3v1AppvagGiven)
            model->BSIM3v1Appvag = 0.0;     
        if (!model->BSIM3v1ApwrGiven)  
            model->BSIM3v1Apwr = 0.0;
        if (!model->BSIM3v1ApdwgGiven)  
            model->BSIM3v1Apdwg = 0.0;
        if (!model->BSIM3v1ApdwbGiven)  
            model->BSIM3v1Apdwb = 0.0;
        if (!model->BSIM3v1Apb0Given)
            model->BSIM3v1Apb0 = 0.0;
        if (!model->BSIM3v1Apb1Given)  
            model->BSIM3v1Apb1 = 0.0;
        if (!model->BSIM3v1Apalpha0Given)  
            model->BSIM3v1Apalpha0 = 0.0;
        if (!model->BSIM3v1Apbeta0Given)  
            model->BSIM3v1Apbeta0 = 0.0;

        if (!model->BSIM3v1ApelmGiven)  
            model->BSIM3v1Apelm = 0.0;
        if (!model->BSIM3v1ApcgslGiven)  
            model->BSIM3v1Apcgsl = 0.0;
        if (!model->BSIM3v1ApcgdlGiven)  
            model->BSIM3v1Apcgdl = 0.0;
        if (!model->BSIM3v1ApckappaGiven)  
            model->BSIM3v1Apckappa = 0.0;
        if (!model->BSIM3v1ApcfGiven)  
            model->BSIM3v1Apcf = 0.0;
        if (!model->BSIM3v1ApclcGiven)  
            model->BSIM3v1Apclc = 0.0;
        if (!model->BSIM3v1ApcleGiven)  
            model->BSIM3v1Apcle = 0.0;

        /* unit degree celcius */
        if (!model->BSIM3v1AtnomGiven)  
	    model->BSIM3v1Atnom = ckt->CKTnomTemp;   
        if (!model->BSIM3v1ALintGiven)  
           model->BSIM3v1ALint = 0.0;
        if (!model->BSIM3v1ALlGiven)  
           model->BSIM3v1ALl = 0.0;
        if (!model->BSIM3v1ALlnGiven)  
           model->BSIM3v1ALln = 1.0;
        if (!model->BSIM3v1ALwGiven)  
           model->BSIM3v1ALw = 0.0;
        if (!model->BSIM3v1ALwnGiven)  
           model->BSIM3v1ALwn = 1.0;
        if (!model->BSIM3v1ALwlGiven)  
           model->BSIM3v1ALwl = 0.0;
        if (!model->BSIM3v1ALminGiven)  
           model->BSIM3v1ALmin = 0.0;
        if (!model->BSIM3v1ALmaxGiven)  
           model->BSIM3v1ALmax = 1.0;
        if (!model->BSIM3v1AWintGiven)  
           model->BSIM3v1AWint = 0.0;
        if (!model->BSIM3v1AWlGiven)  
           model->BSIM3v1AWl = 0.0;
        if (!model->BSIM3v1AWlnGiven)  
           model->BSIM3v1AWln = 1.0;
        if (!model->BSIM3v1AWwGiven)  
           model->BSIM3v1AWw = 0.0;
        if (!model->BSIM3v1AWwnGiven)  
           model->BSIM3v1AWwn = 1.0;
        if (!model->BSIM3v1AWwlGiven)  
           model->BSIM3v1AWwl = 0.0;
        if (!model->BSIM3v1AWminGiven)  
           model->BSIM3v1AWmin = 0.0;
        if (!model->BSIM3v1AWmaxGiven)  
           model->BSIM3v1AWmax = 1.0;
        if (!model->BSIM3v1AdwcGiven)  
           model->BSIM3v1Adwc = model->BSIM3v1AWint;
        if (!model->BSIM3v1AdlcGiven)  
           model->BSIM3v1Adlc = model->BSIM3v1ALint;
	   if (!model->BSIM3v1AcfGiven)
            model->BSIM3v1Acf = 2.0 * EPSOX / PI
			   * log(1.0 + 0.4e-6 / model->BSIM3v1Atox);
        if (!model->BSIM3v1AcgdoGiven)
	{   if (model->BSIM3v1AdlcGiven && (model->BSIM3v1Adlc > 0.0))
	    {   model->BSIM3v1Acgdo = model->BSIM3v1Adlc * model->BSIM3v1Acox
				 - model->BSIM3v1Acgdl ;
		if (model->BSIM3v1Acgdo < 0.0)
		    model->BSIM3v1Acgdo = 0.0;
	    }
	    else
	        model->BSIM3v1Acgdo = 0.6 * model->BSIM3v1Axj * model->BSIM3v1Acox; 
	}
        if (!model->BSIM3v1AcgsoGiven)
	{   if (model->BSIM3v1AdlcGiven && (model->BSIM3v1Adlc > 0.0))
	    {   model->BSIM3v1Acgso = model->BSIM3v1Adlc * model->BSIM3v1Acox
				 - model->BSIM3v1Acgsl ;
		if (model->BSIM3v1Acgso < 0.0)
		    model->BSIM3v1Acgso = 0.0;
	    }
	    else
	        model->BSIM3v1Acgso = 0.6 * model->BSIM3v1Axj * model->BSIM3v1Acox; 
	}

        if (!model->BSIM3v1AcgboGiven)
            model->BSIM3v1Acgbo = 0.0;
        if (!model->BSIM3v1AxpartGiven)
            model->BSIM3v1Axpart = 0.0;
        if (!model->BSIM3v1AsheetResistanceGiven)
            model->BSIM3v1AsheetResistance = 0.0;
        if (!model->BSIM3v1AunitAreaJctCapGiven)
            model->BSIM3v1AunitAreaJctCap = 5.0E-4;
        if (!model->BSIM3v1AunitLengthSidewallJctCapGiven)
            model->BSIM3v1AunitLengthSidewallJctCap = 5.0E-10;
        if (!model->BSIM3v1AjctSatCurDensityGiven)
            model->BSIM3v1AjctSatCurDensity = 1.0E-4;
        if (!model->BSIM3v1AbulkJctPotentialGiven)
            model->BSIM3v1AbulkJctPotential = 1.0;
        if (!model->BSIM3v1AsidewallJctPotentialGiven)
            model->BSIM3v1AsidewallJctPotential = 1.0;
        if (!model->BSIM3v1AbulkJctBotGradingCoeffGiven)
            model->BSIM3v1AbulkJctBotGradingCoeff = 0.5;
        if (!model->BSIM3v1AbulkJctSideGradingCoeffGiven)
            model->BSIM3v1AbulkJctSideGradingCoeff = 0.33;
        if (!model->BSIM3v1AoxideTrapDensityAGiven)
	{   if (model->BSIM3v1Atype == NMOS)
                model->BSIM3v1AoxideTrapDensityA = 1e20;
            else
                model->BSIM3v1AoxideTrapDensityA=9.9e18;
	}
        if (!model->BSIM3v1AoxideTrapDensityBGiven)
	{   if (model->BSIM3v1Atype == NMOS)
                model->BSIM3v1AoxideTrapDensityB = 5e4;
            else
                model->BSIM3v1AoxideTrapDensityB = 2.4e3;
	}
        if (!model->BSIM3v1AoxideTrapDensityCGiven)
	{   if (model->BSIM3v1Atype == NMOS)
                model->BSIM3v1AoxideTrapDensityC = -1.4e-12;
            else
                model->BSIM3v1AoxideTrapDensityC = 1.4e-12;

	}
        if (!model->BSIM3v1AemGiven)
            model->BSIM3v1Aem = 4.1e7; /* V/m */
        if (!model->BSIM3v1AefGiven)
            model->BSIM3v1Aef = 1.0;
        if (!model->BSIM3v1AafGiven)
            model->BSIM3v1Aaf = 1.0;
        if (!model->BSIM3v1AkfGiven)
            model->BSIM3v1Akf = 0.0;
        /* loop through all the instances of the model */
        for (here = model->BSIM3v1Ainstances; here != NULL ;
             here=here->BSIM3v1AnextInstance) 
	{   
	    if (here->BSIM3v1Aowner == ARCHme)
            {  
	       /* allocate a chunk of the state vector */
                here->BSIM3v1Astates = *states;
                *states += BSIM3v1AnumStates;
            }
	    
	    
            /* perform the parameter defaulting */
            if (!here->BSIM3v1AdrainAreaGiven)
                here->BSIM3v1AdrainArea = 0.0;
            if (!here->BSIM3v1AdrainPerimeterGiven)
                here->BSIM3v1AdrainPerimeter = 0.0;
            if (!here->BSIM3v1AdrainSquaresGiven)
                here->BSIM3v1AdrainSquares = 1.0;
            if (!here->BSIM3v1AicVBSGiven)
                here->BSIM3v1AicVBS = 0;
            if (!here->BSIM3v1AicVDSGiven)
                here->BSIM3v1AicVDS = 0;
            if (!here->BSIM3v1AicVGSGiven)
                here->BSIM3v1AicVGS = 0;
            if (!here->BSIM3v1AlGiven)
                here->BSIM3v1Al = 5e-6;
            if (!here->BSIM3v1AsourceAreaGiven)
                here->BSIM3v1AsourceArea = 0;
            if (!here->BSIM3v1AsourcePerimeterGiven)
                here->BSIM3v1AsourcePerimeter = 0;
            if (!here->BSIM3v1AsourceSquaresGiven)
                here->BSIM3v1AsourceSquares = 1;
            if (!here->BSIM3v1AwGiven)
                here->BSIM3v1Aw = 5e-6;

            if (!here->BSIM3v1AmGiven)
                here->BSIM3v1Am = 1;

            if (!here->BSIM3v1AnqsModGiven)
                here->BSIM3v1AnqsMod = model->BSIM3v1AnqsMod;
                    
            /* process drain series resistance */
            if ((model->BSIM3v1AsheetResistance > 0.0) && 
                (here->BSIM3v1AdrainSquares > 0.0 ) &&
                (here->BSIM3v1AdNodePrime == 0))
	    {   error = CKTmkVolt(ckt,&tmp,here->BSIM3v1Aname,"drain");
                if(error) return(error);
                here->BSIM3v1AdNodePrime = tmp->number;

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
	    {   here->BSIM3v1AdNodePrime = here->BSIM3v1AdNode;
            }
                   
            /* process source series resistance */
            if ((model->BSIM3v1AsheetResistance > 0.0) && 
                (here->BSIM3v1AsourceSquares > 0.0 ) &&
                (here->BSIM3v1AsNodePrime == 0)) 
	    {   error = CKTmkVolt(ckt,&tmp,here->BSIM3v1Aname,"source");
                if(error) return(error);
                here->BSIM3v1AsNodePrime = tmp->number;

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
	    {   here->BSIM3v1AsNodePrime = here->BSIM3v1AsNode;
            }

 /* internal charge node */
                   
            if ((here->BSIM3v1AnqsMod) && (here->BSIM3v1AqNode == 0)) 
	    {   error = CKTmkVolt(ckt,&tmp,here->BSIM3v1Aname,"charge");
                if(error) return(error);
                here->BSIM3v1AqNode = tmp->number;
            }
	    else 
	    {   here->BSIM3v1AqNode = 0;
            }

        /* set Sparse Matrix Pointers */

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==(double *)NULL){\
    return(E_NOMEM);\
}

            TSTALLOC(BSIM3v1ADdPtr, BSIM3v1AdNode, BSIM3v1AdNode)
            TSTALLOC(BSIM3v1AGgPtr, BSIM3v1AgNode, BSIM3v1AgNode)
            TSTALLOC(BSIM3v1ASsPtr, BSIM3v1AsNode, BSIM3v1AsNode)
            TSTALLOC(BSIM3v1ABbPtr, BSIM3v1AbNode, BSIM3v1AbNode)
            TSTALLOC(BSIM3v1ADPdpPtr, BSIM3v1AdNodePrime, BSIM3v1AdNodePrime)
            TSTALLOC(BSIM3v1ASPspPtr, BSIM3v1AsNodePrime, BSIM3v1AsNodePrime)
            TSTALLOC(BSIM3v1ADdpPtr, BSIM3v1AdNode, BSIM3v1AdNodePrime)
            TSTALLOC(BSIM3v1AGbPtr, BSIM3v1AgNode, BSIM3v1AbNode)
            TSTALLOC(BSIM3v1AGdpPtr, BSIM3v1AgNode, BSIM3v1AdNodePrime)
            TSTALLOC(BSIM3v1AGspPtr, BSIM3v1AgNode, BSIM3v1AsNodePrime)
            TSTALLOC(BSIM3v1ASspPtr, BSIM3v1AsNode, BSIM3v1AsNodePrime)
            TSTALLOC(BSIM3v1ABdpPtr, BSIM3v1AbNode, BSIM3v1AdNodePrime)
            TSTALLOC(BSIM3v1ABspPtr, BSIM3v1AbNode, BSIM3v1AsNodePrime)
            TSTALLOC(BSIM3v1ADPspPtr, BSIM3v1AdNodePrime, BSIM3v1AsNodePrime)
            TSTALLOC(BSIM3v1ADPdPtr, BSIM3v1AdNodePrime, BSIM3v1AdNode)
            TSTALLOC(BSIM3v1ABgPtr, BSIM3v1AbNode, BSIM3v1AgNode)
            TSTALLOC(BSIM3v1ADPgPtr, BSIM3v1AdNodePrime, BSIM3v1AgNode)
            TSTALLOC(BSIM3v1ASPgPtr, BSIM3v1AsNodePrime, BSIM3v1AgNode)
            TSTALLOC(BSIM3v1ASPsPtr, BSIM3v1AsNodePrime, BSIM3v1AsNode)
            TSTALLOC(BSIM3v1ADPbPtr, BSIM3v1AdNodePrime, BSIM3v1AbNode)
            TSTALLOC(BSIM3v1ASPbPtr, BSIM3v1AsNodePrime, BSIM3v1AbNode)
            TSTALLOC(BSIM3v1ASPdpPtr, BSIM3v1AsNodePrime, BSIM3v1AdNodePrime)

            TSTALLOC(BSIM3v1AQqPtr, BSIM3v1AqNode, BSIM3v1AqNode)
 
            TSTALLOC(BSIM3v1AQdpPtr, BSIM3v1AqNode, BSIM3v1AdNodePrime)
            TSTALLOC(BSIM3v1AQspPtr, BSIM3v1AqNode, BSIM3v1AsNodePrime)
            TSTALLOC(BSIM3v1AQgPtr, BSIM3v1AqNode, BSIM3v1AgNode)
            TSTALLOC(BSIM3v1AQbPtr, BSIM3v1AqNode, BSIM3v1AbNode)
            TSTALLOC(BSIM3v1ADPqPtr, BSIM3v1AdNodePrime, BSIM3v1AqNode)
            TSTALLOC(BSIM3v1ASPqPtr, BSIM3v1AsNodePrime, BSIM3v1AqNode)
            TSTALLOC(BSIM3v1AGqPtr, BSIM3v1AgNode, BSIM3v1AqNode)
            TSTALLOC(BSIM3v1ABqPtr, BSIM3v1AbNode, BSIM3v1AqNode)

        }
    }
    return(OK);
}  


int
BSIM3v1Aunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v1Amodel *model;
    BSIM3v1Ainstance *here;

    for (model = (BSIM3v1Amodel *)inModel; model != NULL;
            model = model->BSIM3v1AnextModel)
    {
        for (here = model->BSIM3v1Ainstances; here != NULL;
                here=here->BSIM3v1AnextInstance)
        {
            if (here->BSIM3v1AdNodePrime
                    && here->BSIM3v1AdNodePrime != here->BSIM3v1AdNode)
            {
                CKTdltNNum(ckt, here->BSIM3v1AdNodePrime);
                here->BSIM3v1AdNodePrime = 0;
            }
            if (here->BSIM3v1AsNodePrime
                    && here->BSIM3v1AsNodePrime != here->BSIM3v1AsNode)
            {
                CKTdltNNum(ckt, here->BSIM3v1AsNodePrime);
                here->BSIM3v1AsNodePrime = 0;
            }
        }
    }
    return OK;
}


