/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1sset.c
**********/

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "bsim3v1sdef.h"
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
BSIM3v1Ssetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, 
              int *states)
{
BSIM3v1Smodel *model = (BSIM3v1Smodel*)inModel;
BSIM3v1Sinstance *here;
int error;
CKTnode *tmp;

CKTnode *tmpNode;
IFuid tmpName;


    /*  loop through all the BSIM3v1S device models */
    for( ; model != NULL; model = model->BSIM3v1SnextModel )
    {
/* Default value Processing for BSIM3v1S MOSFET Models */
        if (!model->BSIM3v1StypeGiven)
            model->BSIM3v1Stype = NMOS;     
        if (!model->BSIM3v1SmobModGiven) 
            model->BSIM3v1SmobMod = 1;
        if (!model->BSIM3v1SbinUnitGiven) 
            model->BSIM3v1SbinUnit = 1;
        if (!model->BSIM3v1SparamChkGiven) 
            model->BSIM3v1SparamChk = 0;
        if (!model->BSIM3v1ScapModGiven) 
            model->BSIM3v1ScapMod = 2;
        if (!model->BSIM3v1SnqsModGiven) 
            model->BSIM3v1SnqsMod = 0;
        if (!model->BSIM3v1SnoiModGiven) 
            model->BSIM3v1SnoiMod = 1;
        if (!model->BSIM3v1SversionGiven) 
            model->BSIM3v1Sversion = 3.1;
        if (!model->BSIM3v1StoxGiven)
            model->BSIM3v1Stox = 150.0e-10;
        model->BSIM3v1Scox = 3.453133e-11 / model->BSIM3v1Stox;

        if (!model->BSIM3v1ScdscGiven)
	    model->BSIM3v1Scdsc = 2.4e-4;   /* unit Q/V/m^2  */
        if (!model->BSIM3v1ScdscbGiven)
	    model->BSIM3v1Scdscb = 0.0;   /* unit Q/V/m^2  */    
	    if (!model->BSIM3v1ScdscdGiven)
	    model->BSIM3v1Scdscd = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM3v1ScitGiven)
	    model->BSIM3v1Scit = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM3v1SnfactorGiven)
	    model->BSIM3v1Snfactor = 1;
        if (!model->BSIM3v1SxjGiven)
            model->BSIM3v1Sxj = .15e-6;
        if (!model->BSIM3v1SvsatGiven)
            model->BSIM3v1Svsat = 8.0e4;    /* unit m/s */ 
        if (!model->BSIM3v1SatGiven)
            model->BSIM3v1Sat = 3.3e4;    /* unit m/s */ 
        if (!model->BSIM3v1Sa0Given)
            model->BSIM3v1Sa0 = 1.0;  
        if (!model->BSIM3v1SagsGiven)
            model->BSIM3v1Sags = 0.0;
        if (!model->BSIM3v1Sa1Given)
            model->BSIM3v1Sa1 = 0.0;
        if (!model->BSIM3v1Sa2Given)
            model->BSIM3v1Sa2 = 1.0;
        if (!model->BSIM3v1SketaGiven)
            model->BSIM3v1Sketa = -0.047;    /* unit  / V */
        if (!model->BSIM3v1SnsubGiven)
            model->BSIM3v1Snsub = 6.0e16;   /* unit 1/cm3 */
        if (!model->BSIM3v1SnpeakGiven)
            model->BSIM3v1Snpeak = 1.7e17;   /* unit 1/cm3 */
        if (!model->BSIM3v1SngateGiven)
            model->BSIM3v1Sngate = 0;   /* unit 1/cm3 */
        if (!model->BSIM3v1SvbmGiven)
	    model->BSIM3v1Svbm = -3.0;
        if (!model->BSIM3v1SxtGiven)
	    model->BSIM3v1Sxt = 1.55e-7;
        if (!model->BSIM3v1Skt1Given)
            model->BSIM3v1Skt1 = -0.11;      /* unit V */
        if (!model->BSIM3v1Skt1lGiven)
            model->BSIM3v1Skt1l = 0.0;      /* unit V*m */
        if (!model->BSIM3v1Skt2Given)
            model->BSIM3v1Skt2 = 0.022;      /* No unit */
        if (!model->BSIM3v1Sk3Given)
            model->BSIM3v1Sk3 = 80.0;      
        if (!model->BSIM3v1Sk3bGiven)
            model->BSIM3v1Sk3b = 0.0;      
        if (!model->BSIM3v1Sw0Given)
            model->BSIM3v1Sw0 = 2.5e-6;    
        if (!model->BSIM3v1SnlxGiven)
            model->BSIM3v1Snlx = 1.74e-7;     
        if (!model->BSIM3v1Sdvt0Given)
            model->BSIM3v1Sdvt0 = 2.2;    
        if (!model->BSIM3v1Sdvt1Given)
            model->BSIM3v1Sdvt1 = 0.53;      
        if (!model->BSIM3v1Sdvt2Given)
            model->BSIM3v1Sdvt2 = -0.032;   /* unit 1 / V */     

        if (!model->BSIM3v1Sdvt0wGiven)
            model->BSIM3v1Sdvt0w = 0.0;    
        if (!model->BSIM3v1Sdvt1wGiven)
            model->BSIM3v1Sdvt1w = 5.3e6;    
        if (!model->BSIM3v1Sdvt2wGiven)
            model->BSIM3v1Sdvt2w = -0.032;   

        if (!model->BSIM3v1SdroutGiven)
            model->BSIM3v1Sdrout = 0.56;     
        if (!model->BSIM3v1SdsubGiven)
            model->BSIM3v1Sdsub = model->BSIM3v1Sdrout;     
        if (!model->BSIM3v1Svth0Given)
            model->BSIM3v1Svth0 = (model->BSIM3v1Stype == NMOS) ? 0.7 : -0.7;
        if (!model->BSIM3v1SuaGiven)
            model->BSIM3v1Sua = 2.25e-9;      /* unit m/V */
        if (!model->BSIM3v1Sua1Given)
            model->BSIM3v1Sua1 = 4.31e-9;      /* unit m/V */
        if (!model->BSIM3v1SubGiven)
            model->BSIM3v1Sub = 5.87e-19;     /* unit (m/V)**2 */
        if (!model->BSIM3v1Sub1Given)
            model->BSIM3v1Sub1 = -7.61e-18;     /* unit (m/V)**2 */
        if (!model->BSIM3v1SucGiven)
            model->BSIM3v1Suc = (model->BSIM3v1SmobMod == 3) ? -0.0465 : -0.0465e-9;   
        if (!model->BSIM3v1Suc1Given)
            model->BSIM3v1Suc1 = (model->BSIM3v1SmobMod == 3) ? -0.056 : -0.056e-9;   
        if (!model->BSIM3v1Su0Given)
            model->BSIM3v1Su0 = (model->BSIM3v1Stype == NMOS) ? 0.067 : 0.025;
        if (!model->BSIM3v1SuteGiven)
	    model->BSIM3v1Sute = -1.5;    
        if (!model->BSIM3v1SvoffGiven)
	    model->BSIM3v1Svoff = -0.08;
        if (!model->BSIM3v1SdeltaGiven)  
           model->BSIM3v1Sdelta = 0.01;
        if (!model->BSIM3v1SrdswGiven)
            model->BSIM3v1Srdsw = 0;      
        if (!model->BSIM3v1SprwgGiven)
            model->BSIM3v1Sprwg = 0.0;      /* unit 1/V */
        if (!model->BSIM3v1SprwbGiven)
            model->BSIM3v1Sprwb = 0.0;      
        if (!model->BSIM3v1SprtGiven)
        if (!model->BSIM3v1SprtGiven)
            model->BSIM3v1Sprt = 0.0;      
        if (!model->BSIM3v1Seta0Given)
            model->BSIM3v1Seta0 = 0.08;      /* no unit  */ 
        if (!model->BSIM3v1SetabGiven)
            model->BSIM3v1Setab = -0.07;      /* unit  1/V */ 
        if (!model->BSIM3v1SpclmGiven)
            model->BSIM3v1Spclm = 1.3;      /* no unit  */ 
        if (!model->BSIM3v1Spdibl1Given)
            model->BSIM3v1Spdibl1 = .39;    /* no unit  */
        if (!model->BSIM3v1Spdibl2Given)
            model->BSIM3v1Spdibl2 = 0.0086;    /* no unit  */ 
        if (!model->BSIM3v1SpdiblbGiven)
            model->BSIM3v1Spdiblb = 0.0;    /* 1/V  */ 
        if (!model->BSIM3v1Spscbe1Given)
            model->BSIM3v1Spscbe1 = 4.24e8;     
        if (!model->BSIM3v1Spscbe2Given)
            model->BSIM3v1Spscbe2 = 1.0e-5;    
        if (!model->BSIM3v1SpvagGiven)
            model->BSIM3v1Spvag = 0.0;     
        if (!model->BSIM3v1SwrGiven)  
            model->BSIM3v1Swr = 1.0;
        if (!model->BSIM3v1SdwgGiven)  
            model->BSIM3v1Sdwg = 0.0;
        if (!model->BSIM3v1SdwbGiven)  
            model->BSIM3v1Sdwb = 0.0;
        if (!model->BSIM3v1Sb0Given)
            model->BSIM3v1Sb0 = 0.0;
        if (!model->BSIM3v1Sb1Given)  
            model->BSIM3v1Sb1 = 0.0;
        if (!model->BSIM3v1Salpha0Given)  
            model->BSIM3v1Salpha0 = 0.0;
        if (!model->BSIM3v1Sbeta0Given)  
            model->BSIM3v1Sbeta0 = 30.0;

        if (!model->BSIM3v1SelmGiven)  
            model->BSIM3v1Selm = 5.0;
        if (!model->BSIM3v1ScgslGiven)  
            model->BSIM3v1Scgsl = 0.0;
        if (!model->BSIM3v1ScgdlGiven)  
            model->BSIM3v1Scgdl = 0.0;
        if (!model->BSIM3v1SckappaGiven)  
            model->BSIM3v1Sckappa = 0.6;
        if (!model->BSIM3v1SclcGiven)  
            model->BSIM3v1Sclc = 0.1e-6;
        if (!model->BSIM3v1ScleGiven)  
            model->BSIM3v1Scle = 0.6;
        if (!model->BSIM3v1SvfbcvGiven)  
            model->BSIM3v1Svfbcv = -1.0;

	/* Length dependence */
        if (!model->BSIM3v1SlcdscGiven)
	    model->BSIM3v1Slcdsc = 0.0;
        if (!model->BSIM3v1SlcdscbGiven)
	    model->BSIM3v1Slcdscb = 0.0;
	    if (!model->BSIM3v1SlcdscdGiven) 
	    model->BSIM3v1Slcdscd = 0.0;
        if (!model->BSIM3v1SlcitGiven)
	    model->BSIM3v1Slcit = 0.0;
        if (!model->BSIM3v1SlnfactorGiven)
	    model->BSIM3v1Slnfactor = 0.0;
        if (!model->BSIM3v1SlxjGiven)
            model->BSIM3v1Slxj = 0.0;
        if (!model->BSIM3v1SlvsatGiven)
            model->BSIM3v1Slvsat = 0.0;
        if (!model->BSIM3v1SlatGiven)
            model->BSIM3v1Slat = 0.0;
        if (!model->BSIM3v1Sla0Given)
            model->BSIM3v1Sla0 = 0.0; 
        if (!model->BSIM3v1SlagsGiven)
            model->BSIM3v1Slags = 0.0;
        if (!model->BSIM3v1Sla1Given)
            model->BSIM3v1Sla1 = 0.0;
        if (!model->BSIM3v1Sla2Given)
            model->BSIM3v1Sla2 = 0.0;
        if (!model->BSIM3v1SlketaGiven)
            model->BSIM3v1Slketa = 0.0;
        if (!model->BSIM3v1SlnsubGiven)
            model->BSIM3v1Slnsub = 0.0;
        if (!model->BSIM3v1SlnpeakGiven)
            model->BSIM3v1Slnpeak = 0.0;
        if (!model->BSIM3v1SlngateGiven)
            model->BSIM3v1Slngate = 0.0;
        if (!model->BSIM3v1SlvbmGiven)
	    model->BSIM3v1Slvbm = 0.0;
        if (!model->BSIM3v1SlxtGiven)
	    model->BSIM3v1Slxt = 0.0;
        if (!model->BSIM3v1Slkt1Given)
            model->BSIM3v1Slkt1 = 0.0; 
        if (!model->BSIM3v1Slkt1lGiven)
            model->BSIM3v1Slkt1l = 0.0;
        if (!model->BSIM3v1Slkt2Given)
            model->BSIM3v1Slkt2 = 0.0;
        if (!model->BSIM3v1Slk3Given)
            model->BSIM3v1Slk3 = 0.0;      
        if (!model->BSIM3v1Slk3bGiven)
            model->BSIM3v1Slk3b = 0.0;      
        if (!model->BSIM3v1Slw0Given)
            model->BSIM3v1Slw0 = 0.0;    
        if (!model->BSIM3v1SlnlxGiven)
            model->BSIM3v1Slnlx = 0.0;     
        if (!model->BSIM3v1Sldvt0Given)
            model->BSIM3v1Sldvt0 = 0.0;    
        if (!model->BSIM3v1Sldvt1Given)
            model->BSIM3v1Sldvt1 = 0.0;      
        if (!model->BSIM3v1Sldvt2Given)
            model->BSIM3v1Sldvt2 = 0.0;
        if (!model->BSIM3v1Sldvt0wGiven)
            model->BSIM3v1Sldvt0w = 0.0;    
        if (!model->BSIM3v1Sldvt1wGiven)
            model->BSIM3v1Sldvt1w = 0.0;      
        if (!model->BSIM3v1Sldvt2wGiven)
            model->BSIM3v1Sldvt2w = 0.0;
        if (!model->BSIM3v1SldroutGiven)
            model->BSIM3v1Sldrout = 0.0;     
        if (!model->BSIM3v1SldsubGiven)
            model->BSIM3v1Sldsub = 0.0;
        if (!model->BSIM3v1Slvth0Given)
           model->BSIM3v1Slvth0 = 0.0;
        if (!model->BSIM3v1SluaGiven)
            model->BSIM3v1Slua = 0.0;
        if (!model->BSIM3v1Slua1Given)
            model->BSIM3v1Slua1 = 0.0;
        if (!model->BSIM3v1SlubGiven)
            model->BSIM3v1Slub = 0.0;
        if (!model->BSIM3v1Slub1Given)
            model->BSIM3v1Slub1 = 0.0;
        if (!model->BSIM3v1SlucGiven)
            model->BSIM3v1Sluc = 0.0;
        if (!model->BSIM3v1Sluc1Given)
            model->BSIM3v1Sluc1 = 0.0;
        if (!model->BSIM3v1Slu0Given)
            model->BSIM3v1Slu0 = 0.0;
        if (!model->BSIM3v1SluteGiven)
	    model->BSIM3v1Slute = 0.0;    
        if (!model->BSIM3v1SlvoffGiven)
	    model->BSIM3v1Slvoff = 0.0;
        if (!model->BSIM3v1SldeltaGiven)  
            model->BSIM3v1Sldelta = 0.0;
        if (!model->BSIM3v1SlrdswGiven)
            model->BSIM3v1Slrdsw = 0.0;
        if (!model->BSIM3v1SlprwbGiven)
            model->BSIM3v1Slprwb = 0.0;
        if (!model->BSIM3v1SlprwgGiven)
            model->BSIM3v1Slprwg = 0.0;
        if (!model->BSIM3v1SlprtGiven)
        if (!model->BSIM3v1SlprtGiven)
            model->BSIM3v1Slprt = 0.0;
        if (!model->BSIM3v1Sleta0Given)
            model->BSIM3v1Sleta0 = 0.0;
        if (!model->BSIM3v1SletabGiven)
            model->BSIM3v1Sletab = -0.0;
        if (!model->BSIM3v1SlpclmGiven)
            model->BSIM3v1Slpclm = 0.0; 
        if (!model->BSIM3v1Slpdibl1Given)
            model->BSIM3v1Slpdibl1 = 0.0;
        if (!model->BSIM3v1Slpdibl2Given)
            model->BSIM3v1Slpdibl2 = 0.0;
        if (!model->BSIM3v1SlpdiblbGiven)
            model->BSIM3v1Slpdiblb = 0.0;
        if (!model->BSIM3v1Slpscbe1Given)
            model->BSIM3v1Slpscbe1 = 0.0;
        if (!model->BSIM3v1Slpscbe2Given)
            model->BSIM3v1Slpscbe2 = 0.0;
        if (!model->BSIM3v1SlpvagGiven)
            model->BSIM3v1Slpvag = 0.0;     
        if (!model->BSIM3v1SlwrGiven)  
            model->BSIM3v1Slwr = 0.0;
        if (!model->BSIM3v1SldwgGiven)  
            model->BSIM3v1Sldwg = 0.0;
        if (!model->BSIM3v1SldwbGiven)  
            model->BSIM3v1Sldwb = 0.0;
        if (!model->BSIM3v1Slb0Given)
            model->BSIM3v1Slb0 = 0.0;
        if (!model->BSIM3v1Slb1Given)  
            model->BSIM3v1Slb1 = 0.0;
        if (!model->BSIM3v1Slalpha0Given)  
            model->BSIM3v1Slalpha0 = 0.0;
        if (!model->BSIM3v1Slbeta0Given)  
            model->BSIM3v1Slbeta0 = 0.0;

        if (!model->BSIM3v1SlelmGiven)  
            model->BSIM3v1Slelm = 0.0;
        if (!model->BSIM3v1SlcgslGiven)  
            model->BSIM3v1Slcgsl = 0.0;
        if (!model->BSIM3v1SlcgdlGiven)  
            model->BSIM3v1Slcgdl = 0.0;
        if (!model->BSIM3v1SlckappaGiven)  
            model->BSIM3v1Slckappa = 0.0;
        if (!model->BSIM3v1SlclcGiven)  
            model->BSIM3v1Slclc = 0.0;
        if (!model->BSIM3v1SlcleGiven)  
            model->BSIM3v1Slcle = 0.0;
        if (!model->BSIM3v1SlcfGiven)  
            model->BSIM3v1Slcf = 0.0;
        if (!model->BSIM3v1SlvfbcvGiven)  
            model->BSIM3v1Slvfbcv = 0.0;

	/* Width dependence */
        if (!model->BSIM3v1SwcdscGiven)
	    model->BSIM3v1Swcdsc = 0.0;
        if (!model->BSIM3v1SwcdscbGiven)
	    model->BSIM3v1Swcdscb = 0.0;  
	    if (!model->BSIM3v1SwcdscdGiven)
	    model->BSIM3v1Swcdscd = 0.0;
        if (!model->BSIM3v1SwcitGiven)
	    model->BSIM3v1Swcit = 0.0;
        if (!model->BSIM3v1SwnfactorGiven)
	    model->BSIM3v1Swnfactor = 0.0;
        if (!model->BSIM3v1SwxjGiven)
            model->BSIM3v1Swxj = 0.0;
        if (!model->BSIM3v1SwvsatGiven)
            model->BSIM3v1Swvsat = 0.0;
        if (!model->BSIM3v1SwatGiven)
            model->BSIM3v1Swat = 0.0;
        if (!model->BSIM3v1Swa0Given)
            model->BSIM3v1Swa0 = 0.0; 
        if (!model->BSIM3v1SwagsGiven)
            model->BSIM3v1Swags = 0.0;
        if (!model->BSIM3v1Swa1Given)
            model->BSIM3v1Swa1 = 0.0;
        if (!model->BSIM3v1Swa2Given)
            model->BSIM3v1Swa2 = 0.0;
        if (!model->BSIM3v1SwketaGiven)
            model->BSIM3v1Swketa = 0.0;
        if (!model->BSIM3v1SwnsubGiven)
            model->BSIM3v1Swnsub = 0.0;
        if (!model->BSIM3v1SwnpeakGiven)
            model->BSIM3v1Swnpeak = 0.0;
        if (!model->BSIM3v1SwngateGiven)
            model->BSIM3v1Swngate = 0.0;
        if (!model->BSIM3v1SwvbmGiven)
	    model->BSIM3v1Swvbm = 0.0;
        if (!model->BSIM3v1SwxtGiven)
	    model->BSIM3v1Swxt = 0.0;
        if (!model->BSIM3v1Swkt1Given)
            model->BSIM3v1Swkt1 = 0.0; 
        if (!model->BSIM3v1Swkt1lGiven)
            model->BSIM3v1Swkt1l = 0.0;
        if (!model->BSIM3v1Swkt2Given)
            model->BSIM3v1Swkt2 = 0.0;
        if (!model->BSIM3v1Swk3Given)
            model->BSIM3v1Swk3 = 0.0;      
        if (!model->BSIM3v1Swk3bGiven)
            model->BSIM3v1Swk3b = 0.0;      
        if (!model->BSIM3v1Sww0Given)
            model->BSIM3v1Sww0 = 0.0;    
        if (!model->BSIM3v1SwnlxGiven)
            model->BSIM3v1Swnlx = 0.0;     
        if (!model->BSIM3v1Swdvt0Given)
            model->BSIM3v1Swdvt0 = 0.0;    
        if (!model->BSIM3v1Swdvt1Given)
            model->BSIM3v1Swdvt1 = 0.0;      
        if (!model->BSIM3v1Swdvt2Given)
            model->BSIM3v1Swdvt2 = 0.0;
        if (!model->BSIM3v1Swdvt0wGiven)
            model->BSIM3v1Swdvt0w = 0.0;    
        if (!model->BSIM3v1Swdvt1wGiven)
            model->BSIM3v1Swdvt1w = 0.0;      
        if (!model->BSIM3v1Swdvt2wGiven)
            model->BSIM3v1Swdvt2w = 0.0;
        if (!model->BSIM3v1SwdroutGiven)
            model->BSIM3v1Swdrout = 0.0;     
        if (!model->BSIM3v1SwdsubGiven)
            model->BSIM3v1Swdsub = 0.0;
        if (!model->BSIM3v1Swvth0Given)
           model->BSIM3v1Swvth0 = 0.0;
        if (!model->BSIM3v1SwuaGiven)
            model->BSIM3v1Swua = 0.0;
        if (!model->BSIM3v1Swua1Given)
            model->BSIM3v1Swua1 = 0.0;
        if (!model->BSIM3v1SwubGiven)
            model->BSIM3v1Swub = 0.0;
        if (!model->BSIM3v1Swub1Given)
            model->BSIM3v1Swub1 = 0.0;
        if (!model->BSIM3v1SwucGiven)
            model->BSIM3v1Swuc = 0.0;
        if (!model->BSIM3v1Swuc1Given)
            model->BSIM3v1Swuc1 = 0.0;
        if (!model->BSIM3v1Swu0Given)
            model->BSIM3v1Swu0 = 0.0;
        if (!model->BSIM3v1SwuteGiven)
	    model->BSIM3v1Swute = 0.0;    
        if (!model->BSIM3v1SwvoffGiven)
	    model->BSIM3v1Swvoff = 0.0;
        if (!model->BSIM3v1SwdeltaGiven)  
            model->BSIM3v1Swdelta = 0.0;
        if (!model->BSIM3v1SwrdswGiven)
            model->BSIM3v1Swrdsw = 0.0;
        if (!model->BSIM3v1SwprwbGiven)
            model->BSIM3v1Swprwb = 0.0;
        if (!model->BSIM3v1SwprwgGiven)
            model->BSIM3v1Swprwg = 0.0;
        if (!model->BSIM3v1SwprtGiven)
            model->BSIM3v1Swprt = 0.0;
        if (!model->BSIM3v1Sweta0Given)
            model->BSIM3v1Sweta0 = 0.0;
        if (!model->BSIM3v1SwetabGiven)
            model->BSIM3v1Swetab = 0.0;
        if (!model->BSIM3v1SwpclmGiven)
            model->BSIM3v1Swpclm = 0.0; 
        if (!model->BSIM3v1Swpdibl1Given)
            model->BSIM3v1Swpdibl1 = 0.0;
        if (!model->BSIM3v1Swpdibl2Given)
            model->BSIM3v1Swpdibl2 = 0.0;
        if (!model->BSIM3v1SwpdiblbGiven)
            model->BSIM3v1Swpdiblb = 0.0;
        if (!model->BSIM3v1Swpscbe1Given)
            model->BSIM3v1Swpscbe1 = 0.0;
        if (!model->BSIM3v1Swpscbe2Given)
            model->BSIM3v1Swpscbe2 = 0.0;
        if (!model->BSIM3v1SwpvagGiven)
            model->BSIM3v1Swpvag = 0.0;     
        if (!model->BSIM3v1SwwrGiven)  
            model->BSIM3v1Swwr = 0.0;
        if (!model->BSIM3v1SwdwgGiven)  
            model->BSIM3v1Swdwg = 0.0;
        if (!model->BSIM3v1SwdwbGiven)  
            model->BSIM3v1Swdwb = 0.0;
        if (!model->BSIM3v1Swb0Given)
            model->BSIM3v1Swb0 = 0.0;
        if (!model->BSIM3v1Swb1Given)  
            model->BSIM3v1Swb1 = 0.0;
        if (!model->BSIM3v1Swalpha0Given)  
            model->BSIM3v1Swalpha0 = 0.0;
        if (!model->BSIM3v1Swbeta0Given)  
            model->BSIM3v1Swbeta0 = 0.0;

        if (!model->BSIM3v1SwelmGiven)  
            model->BSIM3v1Swelm = 0.0;
        if (!model->BSIM3v1SwcgslGiven)  
            model->BSIM3v1Swcgsl = 0.0;
        if (!model->BSIM3v1SwcgdlGiven)  
            model->BSIM3v1Swcgdl = 0.0;
        if (!model->BSIM3v1SwckappaGiven)  
            model->BSIM3v1Swckappa = 0.0;
        if (!model->BSIM3v1SwcfGiven)  
            model->BSIM3v1Swcf = 0.0;
        if (!model->BSIM3v1SwclcGiven)  
            model->BSIM3v1Swclc = 0.0;
        if (!model->BSIM3v1SwcleGiven)  
            model->BSIM3v1Swcle = 0.0;
        if (!model->BSIM3v1SwvfbcvGiven)  
            model->BSIM3v1Swvfbcv = 0.0;

	/* Cross-term dependence */
        if (!model->BSIM3v1SpcdscGiven)
	    model->BSIM3v1Spcdsc = 0.0;
        if (!model->BSIM3v1SpcdscbGiven)
	    model->BSIM3v1Spcdscb = 0.0;   
	    if (!model->BSIM3v1SpcdscdGiven)
	    model->BSIM3v1Spcdscd = 0.0;
        if (!model->BSIM3v1SpcitGiven)
	    model->BSIM3v1Spcit = 0.0;
        if (!model->BSIM3v1SpnfactorGiven)
	    model->BSIM3v1Spnfactor = 0.0;
        if (!model->BSIM3v1SpxjGiven)
            model->BSIM3v1Spxj = 0.0;
        if (!model->BSIM3v1SpvsatGiven)
            model->BSIM3v1Spvsat = 0.0;
        if (!model->BSIM3v1SpatGiven)
            model->BSIM3v1Spat = 0.0;
        if (!model->BSIM3v1Spa0Given)
            model->BSIM3v1Spa0 = 0.0; 
            
        if (!model->BSIM3v1SpagsGiven)
            model->BSIM3v1Spags = 0.0;
        if (!model->BSIM3v1Spa1Given)
            model->BSIM3v1Spa1 = 0.0;
        if (!model->BSIM3v1Spa2Given)
            model->BSIM3v1Spa2 = 0.0;
        if (!model->BSIM3v1SpketaGiven)
            model->BSIM3v1Spketa = 0.0;
        if (!model->BSIM3v1SpnsubGiven)
            model->BSIM3v1Spnsub = 0.0;
        if (!model->BSIM3v1SpnpeakGiven)
            model->BSIM3v1Spnpeak = 0.0;
        if (!model->BSIM3v1SpngateGiven)
            model->BSIM3v1Spngate = 0.0;
        if (!model->BSIM3v1SpvbmGiven)
	    model->BSIM3v1Spvbm = 0.0;
        if (!model->BSIM3v1SpxtGiven)
	    model->BSIM3v1Spxt = 0.0;
        if (!model->BSIM3v1Spkt1Given)
            model->BSIM3v1Spkt1 = 0.0; 
        if (!model->BSIM3v1Spkt1lGiven)
            model->BSIM3v1Spkt1l = 0.0;
        if (!model->BSIM3v1Spkt2Given)
            model->BSIM3v1Spkt2 = 0.0;
        if (!model->BSIM3v1Spk3Given)
            model->BSIM3v1Spk3 = 0.0;      
        if (!model->BSIM3v1Spk3bGiven)
            model->BSIM3v1Spk3b = 0.0;      
        if (!model->BSIM3v1Spw0Given)
            model->BSIM3v1Spw0 = 0.0;    
        if (!model->BSIM3v1SpnlxGiven)
            model->BSIM3v1Spnlx = 0.0;     
        if (!model->BSIM3v1Spdvt0Given)
            model->BSIM3v1Spdvt0 = 0.0;    
        if (!model->BSIM3v1Spdvt1Given)
            model->BSIM3v1Spdvt1 = 0.0;      
        if (!model->BSIM3v1Spdvt2Given)
            model->BSIM3v1Spdvt2 = 0.0;
        if (!model->BSIM3v1Spdvt0wGiven)
            model->BSIM3v1Spdvt0w = 0.0;    
        if (!model->BSIM3v1Spdvt1wGiven)
            model->BSIM3v1Spdvt1w = 0.0;      
        if (!model->BSIM3v1Spdvt2wGiven)
            model->BSIM3v1Spdvt2w = 0.0;
        if (!model->BSIM3v1SpdroutGiven)
            model->BSIM3v1Spdrout = 0.0;     
        if (!model->BSIM3v1SpdsubGiven)
            model->BSIM3v1Spdsub = 0.0;
        if (!model->BSIM3v1Spvth0Given)
           model->BSIM3v1Spvth0 = 0.0;
        if (!model->BSIM3v1SpuaGiven)
            model->BSIM3v1Spua = 0.0;
        if (!model->BSIM3v1Spua1Given)
            model->BSIM3v1Spua1 = 0.0;
        if (!model->BSIM3v1SpubGiven)
            model->BSIM3v1Spub = 0.0;
        if (!model->BSIM3v1Spub1Given)
            model->BSIM3v1Spub1 = 0.0;
        if (!model->BSIM3v1SpucGiven)
            model->BSIM3v1Spuc = 0.0;
        if (!model->BSIM3v1Spuc1Given)
            model->BSIM3v1Spuc1 = 0.0;
        if (!model->BSIM3v1Spu0Given)
            model->BSIM3v1Spu0 = 0.0;
        if (!model->BSIM3v1SputeGiven)
	    model->BSIM3v1Spute = 0.0;    
        if (!model->BSIM3v1SpvoffGiven)
	    model->BSIM3v1Spvoff = 0.0;
        if (!model->BSIM3v1SpdeltaGiven)  
            model->BSIM3v1Spdelta = 0.0;
        if (!model->BSIM3v1SprdswGiven)
            model->BSIM3v1Sprdsw = 0.0;
        if (!model->BSIM3v1SpprwbGiven)
            model->BSIM3v1Spprwb = 0.0;
        if (!model->BSIM3v1SpprwgGiven)
            model->BSIM3v1Spprwg = 0.0;
        if (!model->BSIM3v1SpprtGiven)
            model->BSIM3v1Spprt = 0.0;
        if (!model->BSIM3v1Speta0Given)
            model->BSIM3v1Speta0 = 0.0;
        if (!model->BSIM3v1SpetabGiven)
            model->BSIM3v1Spetab = 0.0;
        if (!model->BSIM3v1SppclmGiven)
            model->BSIM3v1Sppclm = 0.0; 
        if (!model->BSIM3v1Sppdibl1Given)
            model->BSIM3v1Sppdibl1 = 0.0;
        if (!model->BSIM3v1Sppdibl2Given)
            model->BSIM3v1Sppdibl2 = 0.0;
        if (!model->BSIM3v1SppdiblbGiven)
            model->BSIM3v1Sppdiblb = 0.0;
        if (!model->BSIM3v1Sppscbe1Given)
            model->BSIM3v1Sppscbe1 = 0.0;
        if (!model->BSIM3v1Sppscbe2Given)
            model->BSIM3v1Sppscbe2 = 0.0;
        if (!model->BSIM3v1SppvagGiven)
            model->BSIM3v1Sppvag = 0.0;     
        if (!model->BSIM3v1SpwrGiven)  
            model->BSIM3v1Spwr = 0.0;
        if (!model->BSIM3v1SpdwgGiven)  
            model->BSIM3v1Spdwg = 0.0;
        if (!model->BSIM3v1SpdwbGiven)  
            model->BSIM3v1Spdwb = 0.0;
        if (!model->BSIM3v1Spb0Given)
            model->BSIM3v1Spb0 = 0.0;
        if (!model->BSIM3v1Spb1Given)  
            model->BSIM3v1Spb1 = 0.0;
        if (!model->BSIM3v1Spalpha0Given)  
            model->BSIM3v1Spalpha0 = 0.0;
        if (!model->BSIM3v1Spbeta0Given)  
            model->BSIM3v1Spbeta0 = 0.0;

        if (!model->BSIM3v1SpelmGiven)  
            model->BSIM3v1Spelm = 0.0;
        if (!model->BSIM3v1SpcgslGiven)  
            model->BSIM3v1Spcgsl = 0.0;
        if (!model->BSIM3v1SpcgdlGiven)  
            model->BSIM3v1Spcgdl = 0.0;
        if (!model->BSIM3v1SpckappaGiven)  
            model->BSIM3v1Spckappa = 0.0;
        if (!model->BSIM3v1SpcfGiven)  
            model->BSIM3v1Spcf = 0.0;
        if (!model->BSIM3v1SpclcGiven)  
            model->BSIM3v1Spclc = 0.0;
        if (!model->BSIM3v1SpcleGiven)  
            model->BSIM3v1Spcle = 0.0;
        if (!model->BSIM3v1SpvfbcvGiven)  
            model->BSIM3v1Spvfbcv = 0.0;

        /* unit degree celcius */
        if (!model->BSIM3v1StnomGiven)  
	    model->BSIM3v1Stnom = ckt->CKTnomTemp; 
        if (!model->BSIM3v1SLintGiven)  
           model->BSIM3v1SLint = 0.0;
        if (!model->BSIM3v1SLlGiven)  
           model->BSIM3v1SLl = 0.0;
        if (!model->BSIM3v1SLlnGiven)  
           model->BSIM3v1SLln = 1.0;
        if (!model->BSIM3v1SLwGiven)  
           model->BSIM3v1SLw = 0.0;
        if (!model->BSIM3v1SLwnGiven)  
           model->BSIM3v1SLwn = 1.0;
        if (!model->BSIM3v1SLwlGiven)  
           model->BSIM3v1SLwl = 0.0;
        if (!model->BSIM3v1SLminGiven)  
           model->BSIM3v1SLmin = 0.0;
        if (!model->BSIM3v1SLmaxGiven)  
           model->BSIM3v1SLmax = 1.0;
        if (!model->BSIM3v1SWintGiven)  
           model->BSIM3v1SWint = 0.0;
        if (!model->BSIM3v1SWlGiven)  
           model->BSIM3v1SWl = 0.0;
        if (!model->BSIM3v1SWlnGiven)  
           model->BSIM3v1SWln = 1.0;
        if (!model->BSIM3v1SWwGiven)  
           model->BSIM3v1SWw = 0.0;
        if (!model->BSIM3v1SWwnGiven)  
           model->BSIM3v1SWwn = 1.0;
        if (!model->BSIM3v1SWwlGiven)  
           model->BSIM3v1SWwl = 0.0;
        if (!model->BSIM3v1SWminGiven)  
           model->BSIM3v1SWmin = 0.0;
        if (!model->BSIM3v1SWmaxGiven)  
           model->BSIM3v1SWmax = 1.0;
        if (!model->BSIM3v1SdwcGiven)  
           model->BSIM3v1Sdwc = model->BSIM3v1SWint;
        if (!model->BSIM3v1SdlcGiven)  
           model->BSIM3v1Sdlc = model->BSIM3v1SLint;
	if (!model->BSIM3v1ScfGiven)
            model->BSIM3v1Scf = 2.0 * EPSOX / PI
			   * log(1.0 + 0.4e-6 / model->BSIM3v1Stox);
        if (!model->BSIM3v1ScgdoGiven)
	{   if (model->BSIM3v1SdlcGiven && (model->BSIM3v1Sdlc > 0.0))
	    {   model->BSIM3v1Scgdo = model->BSIM3v1Sdlc * model->BSIM3v1Scox
				 - model->BSIM3v1Scgdl ;
	    }
	    else
	        model->BSIM3v1Scgdo = 0.6 * model->BSIM3v1Sxj * model->BSIM3v1Scox; 
	}
        if (!model->BSIM3v1ScgsoGiven)
	{   if (model->BSIM3v1SdlcGiven && (model->BSIM3v1Sdlc > 0.0))
	    {   model->BSIM3v1Scgso = model->BSIM3v1Sdlc * model->BSIM3v1Scox
				 - model->BSIM3v1Scgsl ;
	    }
	    else
	        model->BSIM3v1Scgso = 0.6 * model->BSIM3v1Sxj * model->BSIM3v1Scox; 
	}

        if (!model->BSIM3v1ScgboGiven)
	{   model->BSIM3v1Scgbo = 2.0 * model->BSIM3v1Sdwc * model->BSIM3v1Scox;
	}
        if (!model->BSIM3v1SxpartGiven)
            model->BSIM3v1Sxpart = 0.0;
        if (!model->BSIM3v1SsheetResistanceGiven)
            model->BSIM3v1SsheetResistance = 0.0;
        if (!model->BSIM3v1SunitAreaJctCapGiven)
            model->BSIM3v1SunitAreaJctCap = 5.0E-4;
        if (!model->BSIM3v1SunitLengthSidewallJctCapGiven)
            model->BSIM3v1SunitLengthSidewallJctCap = 5.0E-10;
        if (!model->BSIM3v1SunitLengthGateSidewallJctCapGiven)
            model->BSIM3v1SunitLengthGateSidewallJctCap = model->BSIM3v1SunitLengthSidewallJctCap ;
        if (!model->BSIM3v1SjctSatCurDensityGiven)
            model->BSIM3v1SjctSatCurDensity = 1.0E-4;
        if (!model->BSIM3v1SjctSidewallSatCurDensityGiven)
            model->BSIM3v1SjctSidewallSatCurDensity = 0.0;
        if (!model->BSIM3v1SbulkJctPotentialGiven)
            model->BSIM3v1SbulkJctPotential = 1.0;
        if (!model->BSIM3v1SsidewallJctPotentialGiven)
            model->BSIM3v1SsidewallJctPotential = 1.0;
        if (!model->BSIM3v1SGatesidewallJctPotentialGiven)
            model->BSIM3v1SGatesidewallJctPotential = model->BSIM3v1SsidewallJctPotential;
        if (!model->BSIM3v1SbulkJctBotGradingCoeffGiven)
            model->BSIM3v1SbulkJctBotGradingCoeff = 0.5;
        if (!model->BSIM3v1SbulkJctSideGradingCoeffGiven)
            model->BSIM3v1SbulkJctSideGradingCoeff = 0.33;
        if (!model->BSIM3v1SbulkJctGateSideGradingCoeffGiven)
            model->BSIM3v1SbulkJctGateSideGradingCoeff = model->BSIM3v1SbulkJctSideGradingCoeff;
        if (!model->BSIM3v1SjctEmissionCoeffGiven)
            model->BSIM3v1SjctEmissionCoeff = 1.0;
        if (!model->BSIM3v1SjctTempExponentGiven)
            model->BSIM3v1SjctTempExponent = 3.0;
        if (!model->BSIM3v1SoxideTrapDensityAGiven)
        if (!model->BSIM3v1SoxideTrapDensityAGiven)
	{   if (model->BSIM3v1Stype == NMOS)
                model->BSIM3v1SoxideTrapDensityA = 1e20;
            else
                model->BSIM3v1SoxideTrapDensityA=9.9e18;
	}
        if (!model->BSIM3v1SoxideTrapDensityBGiven)
	{   if (model->BSIM3v1Stype == NMOS)
                model->BSIM3v1SoxideTrapDensityB = 5e4;
            else
                model->BSIM3v1SoxideTrapDensityB = 2.4e3;
	}
        if (!model->BSIM3v1SoxideTrapDensityCGiven)
	{   if (model->BSIM3v1Stype == NMOS)
                model->BSIM3v1SoxideTrapDensityC = -1.4e-12;
            else
                model->BSIM3v1SoxideTrapDensityC = 1.4e-12;

	}
        if (!model->BSIM3v1SemGiven)
            model->BSIM3v1Sem = 4.1e7; /* V/m */
        if (!model->BSIM3v1SefGiven)
            model->BSIM3v1Sef = 1.0;
        if (!model->BSIM3v1SafGiven)
            model->BSIM3v1Saf = 1.0;
        if (!model->BSIM3v1SkfGiven)
            model->BSIM3v1Skf = 0.0;
        /* loop through all the instances of the model */
        for (here = model->BSIM3v1Sinstances; here != NULL ;
             here=here->BSIM3v1SnextInstance)
        
	{   
           if (here->BSIM3v1Sowner == ARCHme) { 
           /* allocate a chunk of the state vector */
            here->BSIM3v1Sstates = *states;
            *states += BSIM3v1SnumStates;
           }
	   
            /* perform the parameter defaulting */
            if(here->BSIM3v1Sm == 0.0)
              here->BSIM3v1Sm = 1.0;
            fprintf(stderr, "M = %.2f\n", here->BSIM3v1Sm);
            if (!here->BSIM3v1SwGiven)
	      here->BSIM3v1Sw = 5e-6;
            here->BSIM3v1Sw *= here->BSIM3v1Sm;

            if (!here->BSIM3v1SdrainAreaGiven)
            {
                if(model->BSIM3v1ShdifGiven)
                    here->BSIM3v1SdrainArea = here->BSIM3v1Sw * 2 * model->BSIM3v1Shdif;
                else
                    here->BSIM3v1SdrainArea = 0.0;
            }
            here->BSIM3v1SdrainArea *= here->BSIM3v1Sm;
            if (!here->BSIM3v1SdrainPerimeterGiven)
            {
                if(model->BSIM3v1ShdifGiven)
                    here->BSIM3v1SdrainPerimeter =
                        2 * here->BSIM3v1Sw + 4 * model->BSIM3v1Shdif;
                else
                    here->BSIM3v1SdrainPerimeter = 0.0;
            }
            here->BSIM3v1SdrainPerimeter *= here->BSIM3v1Sm;

            if (!here->BSIM3v1SdrainSquaresGiven)
                here->BSIM3v1SdrainSquares = 1.0;
            here->BSIM3v1SdrainSquares /= here->BSIM3v1Sm;

            if (!here->BSIM3v1SicVBSGiven)
                here->BSIM3v1SicVBS = 0;
            if (!here->BSIM3v1SicVDSGiven)
                here->BSIM3v1SicVDS = 0;
            if (!here->BSIM3v1SicVGSGiven)
                here->BSIM3v1SicVGS = 0;
            if (!here->BSIM3v1SlGiven)
                here->BSIM3v1Sl = 5e-6;
            if (!here->BSIM3v1SsourceAreaGiven)
            {
                if(model->BSIM3v1ShdifGiven)
                    here->BSIM3v1SsourceArea = here->BSIM3v1Sw * 2 * model->BSIM3v1Shdif;
                else
                    here->BSIM3v1SsourceArea = 0.0;
            }
            here->BSIM3v1SsourceArea *= here->BSIM3v1Sm;

            if (!here->BSIM3v1SsourcePerimeterGiven)
            {
                if(model->BSIM3v1ShdifGiven)
                    here->BSIM3v1SsourcePerimeter =
                        2 * here->BSIM3v1Sw + 4 * model->BSIM3v1Shdif;
                else
                    here->BSIM3v1SsourcePerimeter = 0.0;
            }
            here->BSIM3v1SsourcePerimeter *= here->BSIM3v1Sm;

            if (!here->BSIM3v1SsourceSquaresGiven)
                here->BSIM3v1SsourceSquares = 1.0;
            here->BSIM3v1SsourceSquares /= here->BSIM3v1Sm;

            if (!here->BSIM3v1SnqsModGiven)
                here->BSIM3v1SnqsMod = model->BSIM3v1SnqsMod;
                    
            /* process drain series resistance */
            if ((model->BSIM3v1SsheetResistance > 0.0) && 
                (here->BSIM3v1SdrainSquares > 0.0 ) &&
                (here->BSIM3v1SdNodePrime == 0))
	    {   error = CKTmkVolt(ckt,&tmp,here->BSIM3v1Sname,"drain");
                if(error) return(error);
                here->BSIM3v1SdNodePrime = tmp->number;
		
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
	    {   here->BSIM3v1SdNodePrime = here->BSIM3v1SdNode;
            }
                   
            /* process source series resistance */
            if ((model->BSIM3v1SsheetResistance > 0.0) && 
                (here->BSIM3v1SsourceSquares > 0.0 ) &&
                (here->BSIM3v1SsNodePrime == 0)) 
	    {   error = CKTmkVolt(ckt,&tmp,here->BSIM3v1Sname,"source");
                if(error) return(error);
                here->BSIM3v1SsNodePrime = tmp->number;
		
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
	    {   here->BSIM3v1SsNodePrime = here->BSIM3v1SsNode;
            }

 /* internal charge node */
                   
            if ((here->BSIM3v1SnqsMod) && (here->BSIM3v1SqNode == 0)) 
	    {   error = CKTmkVolt(ckt,&tmp,here->BSIM3v1Sname,"charge");
                if(error) return(error);
                here->BSIM3v1SqNode = tmp->number;
            }
	    else 
	    {   here->BSIM3v1SqNode = 0;
            }

        /* set Sparse Matrix Pointers */

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==(double *)NULL){\
    return(E_NOMEM);\
}

            TSTALLOC(BSIM3v1SDdPtr, BSIM3v1SdNode, BSIM3v1SdNode)
            TSTALLOC(BSIM3v1SGgPtr, BSIM3v1SgNode, BSIM3v1SgNode)
            TSTALLOC(BSIM3v1SSsPtr, BSIM3v1SsNode, BSIM3v1SsNode)
            TSTALLOC(BSIM3v1SBbPtr, BSIM3v1SbNode, BSIM3v1SbNode)
            TSTALLOC(BSIM3v1SDPdpPtr, BSIM3v1SdNodePrime, BSIM3v1SdNodePrime)
            TSTALLOC(BSIM3v1SSPspPtr, BSIM3v1SsNodePrime, BSIM3v1SsNodePrime)
            TSTALLOC(BSIM3v1SDdpPtr, BSIM3v1SdNode, BSIM3v1SdNodePrime)
            TSTALLOC(BSIM3v1SGbPtr, BSIM3v1SgNode, BSIM3v1SbNode)
            TSTALLOC(BSIM3v1SGdpPtr, BSIM3v1SgNode, BSIM3v1SdNodePrime)
            TSTALLOC(BSIM3v1SGspPtr, BSIM3v1SgNode, BSIM3v1SsNodePrime)
            TSTALLOC(BSIM3v1SSspPtr, BSIM3v1SsNode, BSIM3v1SsNodePrime)
            TSTALLOC(BSIM3v1SBdpPtr, BSIM3v1SbNode, BSIM3v1SdNodePrime)
            TSTALLOC(BSIM3v1SBspPtr, BSIM3v1SbNode, BSIM3v1SsNodePrime)
            TSTALLOC(BSIM3v1SDPspPtr, BSIM3v1SdNodePrime, BSIM3v1SsNodePrime)
            TSTALLOC(BSIM3v1SDPdPtr, BSIM3v1SdNodePrime, BSIM3v1SdNode)
            TSTALLOC(BSIM3v1SBgPtr, BSIM3v1SbNode, BSIM3v1SgNode)
            TSTALLOC(BSIM3v1SDPgPtr, BSIM3v1SdNodePrime, BSIM3v1SgNode)
            TSTALLOC(BSIM3v1SSPgPtr, BSIM3v1SsNodePrime, BSIM3v1SgNode)
            TSTALLOC(BSIM3v1SSPsPtr, BSIM3v1SsNodePrime, BSIM3v1SsNode)
            TSTALLOC(BSIM3v1SDPbPtr, BSIM3v1SdNodePrime, BSIM3v1SbNode)
            TSTALLOC(BSIM3v1SSPbPtr, BSIM3v1SsNodePrime, BSIM3v1SbNode)
            TSTALLOC(BSIM3v1SSPdpPtr, BSIM3v1SsNodePrime, BSIM3v1SdNodePrime)

            TSTALLOC(BSIM3v1SQqPtr, BSIM3v1SqNode, BSIM3v1SqNode)
 
            TSTALLOC(BSIM3v1SQdpPtr, BSIM3v1SqNode, BSIM3v1SdNodePrime)
            TSTALLOC(BSIM3v1SQspPtr, BSIM3v1SqNode, BSIM3v1SsNodePrime)
            TSTALLOC(BSIM3v1SQgPtr, BSIM3v1SqNode, BSIM3v1SgNode)
            TSTALLOC(BSIM3v1SQbPtr, BSIM3v1SqNode, BSIM3v1SbNode)
            TSTALLOC(BSIM3v1SDPqPtr, BSIM3v1SdNodePrime, BSIM3v1SqNode)
            TSTALLOC(BSIM3v1SSPqPtr, BSIM3v1SsNodePrime, BSIM3v1SqNode)
            TSTALLOC(BSIM3v1SGqPtr, BSIM3v1SgNode, BSIM3v1SqNode)
            TSTALLOC(BSIM3v1SBqPtr, BSIM3v1SbNode, BSIM3v1SqNode)

        }
    }
    return(OK);
}  


int
BSIM3v1Sunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v1Smodel *model;
    BSIM3v1Sinstance *here;

    for (model = (BSIM3v1Smodel *)inModel; model != NULL;
            model = model->BSIM3v1SnextModel)
    {
        for (here = model->BSIM3v1Sinstances; here != NULL;
                here=here->BSIM3v1SnextInstance)
        {
            if (here->BSIM3v1SdNodePrime
                    && here->BSIM3v1SdNodePrime != here->BSIM3v1SdNode)
            {
                CKTdltNNum(ckt, here->BSIM3v1SdNodePrime);
                here->BSIM3v1SdNodePrime = 0;
            }
            if (here->BSIM3v1SsNodePrime
                    && here->BSIM3v1SsNodePrime != here->BSIM3v1SsNode)
            {
                CKTdltNNum(ckt, here->BSIM3v1SsNodePrime);
                here->BSIM3v1SsNodePrime = 0;
            }
        }
    }
    return OK;
}



