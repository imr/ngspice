/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v0set.c
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "bsim3v0def.h"
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
BSIM3v0setup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, 
             int *states)
{
BSIM3v0model *model = (BSIM3v0model*)inModel;
BSIM3v0instance *here;
int error;
CKTnode *tmp;

CKTnode *tmpNode;
IFuid tmpName;

    /*  loop through all the BSIM3v0 device models */
    for( ; model != NULL; model = BSIM3v0nextModel(model))
    {
/* Default value Processing for BSIM3v0 MOSFET Models */
        if (!model->BSIM3v0typeGiven)
            model->BSIM3v0type = NMOS;     
        if (!model->BSIM3v0mobModGiven) 
            model->BSIM3v0mobMod = 1;
        if (!model->BSIM3v0binUnitGiven) 
            model->BSIM3v0binUnit = 1;
        if (!model->BSIM3v0capModGiven) 
            model->BSIM3v0capMod = 1;
        if (!model->BSIM3v0nqsModGiven) 
            model->BSIM3v0nqsMod = 0;
        if (!model->BSIM3v0noiModGiven) 
            model->BSIM3v0noiMod = 1;
        if (!model->BSIM3v0toxGiven)
            model->BSIM3v0tox = 150.0e-10;
        model->BSIM3v0cox = 3.453133e-11 / model->BSIM3v0tox;

        if (!model->BSIM3v0cdscGiven)
	    model->BSIM3v0cdsc = 2.4e-4;   /* unit Q/V/m^2  */
        if (!model->BSIM3v0cdscbGiven)
	    model->BSIM3v0cdscb = 0.0;   /* unit Q/V/m^2  */    
        if (!model->BSIM3v0cdscdGiven)
	    model->BSIM3v0cdscd = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM3v0citGiven)
	    model->BSIM3v0cit = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM3v0nfactorGiven)
	    model->BSIM3v0nfactor = 1;
        if (!model->BSIM3v0xjGiven)
            model->BSIM3v0xj = .15e-6;
        if (!model->BSIM3v0vsatGiven)
            model->BSIM3v0vsat = 8.0e4;    /* unit m/s */ 
        if (!model->BSIM3v0atGiven)
            model->BSIM3v0at = 3.3e4;    /* unit m/s */ 
        if (!model->BSIM3v0a0Given)
            model->BSIM3v0a0 = 1.0;  
        if (!model->BSIM3v0agsGiven)
            model->BSIM3v0ags = 0.0;
        if (!model->BSIM3v0a1Given)
            model->BSIM3v0a1 = 0.0;
        if (!model->BSIM3v0a2Given)
            model->BSIM3v0a2 = 1.0;
        if (!model->BSIM3v0ketaGiven)
            model->BSIM3v0keta = -0.047;    /* unit  / V */
        if (!model->BSIM3v0nsubGiven)
            model->BSIM3v0nsub = 6.0e16;   /* unit 1/cm3 */
        if (!model->BSIM3v0npeakGiven)
            model->BSIM3v0npeak = 1.7e17;   /* unit 1/cm3 */
        if (!model->BSIM3v0vbmGiven)
	    model->BSIM3v0vbm = -5.0;
        if (!model->BSIM3v0xtGiven)
	    model->BSIM3v0xt = 1.55e-7;
        if (!model->BSIM3v0kt1Given)
            model->BSIM3v0kt1 = -0.11;      /* unit V */
        if (!model->BSIM3v0kt1lGiven)
            model->BSIM3v0kt1l = 0.0;      /* unit V*m */
        if (!model->BSIM3v0kt2Given)
            model->BSIM3v0kt2 = 0.022;      /* No unit */
        if (!model->BSIM3v0k3Given)
            model->BSIM3v0k3 = 80.0;      
        if (!model->BSIM3v0k3bGiven)
            model->BSIM3v0k3b = 0.0;      
        if (!model->BSIM3v0w0Given)
            model->BSIM3v0w0 = 2.5e-6;    
        if (!model->BSIM3v0nlxGiven)
            model->BSIM3v0nlx = 1.74e-7;     
        if (!model->BSIM3v0dvt0Given)
            model->BSIM3v0dvt0 = 2.2;    
        if (!model->BSIM3v0dvt1Given)
            model->BSIM3v0dvt1 = 0.53;      
        if (!model->BSIM3v0dvt2Given)
            model->BSIM3v0dvt2 = -0.032;   /* unit 1 / V */     

        if (!model->BSIM3v0dvt0wGiven)
            model->BSIM3v0dvt0w = 0.0;    
        if (!model->BSIM3v0dvt1wGiven)
            model->BSIM3v0dvt1w = 5.3e6;    
        if (!model->BSIM3v0dvt2wGiven)
            model->BSIM3v0dvt2w = -0.032;   

        if (!model->BSIM3v0droutGiven)
            model->BSIM3v0drout = 0.56;     
        if (!model->BSIM3v0dsubGiven)
            model->BSIM3v0dsub = model->BSIM3v0drout;     
        if (!model->BSIM3v0vth0Given)
            model->BSIM3v0vth0 = (model->BSIM3v0type == NMOS) ? 0.7 : -0.7;
        if (!model->BSIM3v0uaGiven)
            model->BSIM3v0ua = 2.25e-9;      /* unit m/V */
        if (!model->BSIM3v0ua1Given)
            model->BSIM3v0ua1 = 4.31e-9;      /* unit m/V */
        if (!model->BSIM3v0ubGiven)
            model->BSIM3v0ub = 5.87e-19;     /* unit (m/V)**2 */
        if (!model->BSIM3v0ub1Given)
            model->BSIM3v0ub1 = -7.61e-18;     /* unit (m/V)**2 */
        if (!model->BSIM3v0ucGiven)
            model->BSIM3v0uc = (model->BSIM3v0mobMod == 3) ? -0.0465 : -0.0465e-9;   
        if (!model->BSIM3v0uc1Given)
            model->BSIM3v0uc1 = (model->BSIM3v0mobMod == 3) ? -0.056 : -0.056e-9;   
        if (!model->BSIM3v0u0Given)
            model->BSIM3v0u0 = (model->BSIM3v0type == NMOS) ? 0.067 : 0.025;
        else if (model->BSIM3v0u0 > 1.0)
            model->BSIM3v0u0 = model->BSIM3v0u0 / 1.0e4;  
            /* if u0 > 1.0, cm.g.sec unit system is used.  If should be
               converted into SI unit system.  */
        if (!model->BSIM3v0uteGiven)
	    model->BSIM3v0ute = -1.5;    
        if (!model->BSIM3v0voffGiven)
	    model->BSIM3v0voff = -0.08;
        if (!model->BSIM3v0deltaGiven)  
           model->BSIM3v0delta = 0.01;
        if (!model->BSIM3v0rdswGiven)
            model->BSIM3v0rdsw = 0;      
        if (!model->BSIM3v0prwgGiven)
            model->BSIM3v0prwg = 0.0;      /* unit 1/V */
        if (!model->BSIM3v0prwbGiven)
            model->BSIM3v0prwb = 0.0;      
        if (!model->BSIM3v0prtGiven)
            model->BSIM3v0prt = 0.0;      
        if (!model->BSIM3v0eta0Given)
            model->BSIM3v0eta0 = 0.08;      /* no unit  */ 
        if (!model->BSIM3v0etabGiven)
            model->BSIM3v0etab = -0.07;      /* unit  1/V */ 
        if (!model->BSIM3v0pclmGiven)
            model->BSIM3v0pclm = 1.3;      /* no unit  */ 
        if (!model->BSIM3v0pdibl1Given)
            model->BSIM3v0pdibl1 = .39;    /* no unit  */
        if (!model->BSIM3v0pdibl2Given)
            model->BSIM3v0pdibl2 = 0.0086;    /* no unit  */ 
        if (!model->BSIM3v0pdiblbGiven)
            model->BSIM3v0pdiblb = 0.0;    /* 1/V  */ 
        if (!model->BSIM3v0pscbe1Given)
            model->BSIM3v0pscbe1 = 4.24e8;     
        if (!model->BSIM3v0pscbe2Given)
            model->BSIM3v0pscbe2 = 1.0e-5;    
        if (!model->BSIM3v0pvagGiven)
            model->BSIM3v0pvag = 0.0;     
        if (!model->BSIM3v0wrGiven)  
            model->BSIM3v0wr = 1.0;
        if (!model->BSIM3v0dwgGiven)  
            model->BSIM3v0dwg = 0.0;
        if (!model->BSIM3v0dwbGiven)  
            model->BSIM3v0dwb = 0.0;
        if (!model->BSIM3v0b0Given)
            model->BSIM3v0b0 = 0.0;
        if (!model->BSIM3v0b1Given)  
            model->BSIM3v0b1 = 0.0;
        if (!model->BSIM3v0alpha0Given)  
            model->BSIM3v0alpha0 = 0.0;
        if (!model->BSIM3v0beta0Given)  
            model->BSIM3v0beta0 = 30.0;

        if (!model->BSIM3v0elmGiven)  
            model->BSIM3v0elm = 5.0;
        if (!model->BSIM3v0cgslGiven)  
            model->BSIM3v0cgsl = 0.0;
        if (!model->BSIM3v0cgdlGiven)  
            model->BSIM3v0cgdl = 0.0;
        if (!model->BSIM3v0ckappaGiven)  
            model->BSIM3v0ckappa = 0.6;
        if (!model->BSIM3v0clcGiven)  
            model->BSIM3v0clc = 0.1e-6;
        if (!model->BSIM3v0cleGiven)  
            model->BSIM3v0cle = 0.6;

	/* Length dependence */
        if (!model->BSIM3v0lcdscGiven)
	    model->BSIM3v0lcdsc = 0.0;
        if (!model->BSIM3v0lcdscbGiven)
	    model->BSIM3v0lcdscb = 0.0;
        if (!model->BSIM3v0lcdscdGiven)
	    model->BSIM3v0lcdscd = 0.0;
        if (!model->BSIM3v0lcitGiven)
	    model->BSIM3v0lcit = 0.0;
        if (!model->BSIM3v0lnfactorGiven)
	    model->BSIM3v0lnfactor = 0.0;
        if (!model->BSIM3v0lxjGiven)
            model->BSIM3v0lxj = 0.0;
        if (!model->BSIM3v0lvsatGiven)
            model->BSIM3v0lvsat = 0.0;
        if (!model->BSIM3v0latGiven)
            model->BSIM3v0lat = 0.0;
        if (!model->BSIM3v0la0Given)
            model->BSIM3v0la0 = 0.0; 
        if (!model->BSIM3v0lagsGiven)
            model->BSIM3v0lags = 0.0;
        if (!model->BSIM3v0la1Given)
            model->BSIM3v0la1 = 0.0;
        if (!model->BSIM3v0la2Given)
            model->BSIM3v0la2 = 0.0;
        if (!model->BSIM3v0lketaGiven)
            model->BSIM3v0lketa = 0.0;
        if (!model->BSIM3v0lnsubGiven)
            model->BSIM3v0lnsub = 0.0;
        if (!model->BSIM3v0lnpeakGiven)
            model->BSIM3v0lnpeak = 0.0;
        if (!model->BSIM3v0lvbmGiven)
	    model->BSIM3v0lvbm = 0.0;
        if (!model->BSIM3v0lxtGiven)
	    model->BSIM3v0lxt = 0.0;
        if (!model->BSIM3v0lkt1Given)
            model->BSIM3v0lkt1 = 0.0; 
        if (!model->BSIM3v0lkt1lGiven)
            model->BSIM3v0lkt1l = 0.0;
        if (!model->BSIM3v0lkt2Given)
            model->BSIM3v0lkt2 = 0.0;
        if (!model->BSIM3v0lk3Given)
            model->BSIM3v0lk3 = 0.0;      
        if (!model->BSIM3v0lk3bGiven)
            model->BSIM3v0lk3b = 0.0;      
        if (!model->BSIM3v0lw0Given)
            model->BSIM3v0lw0 = 0.0;    
        if (!model->BSIM3v0lnlxGiven)
            model->BSIM3v0lnlx = 0.0;     
        if (!model->BSIM3v0ldvt0Given)
            model->BSIM3v0ldvt0 = 0.0;    
        if (!model->BSIM3v0ldvt1Given)
            model->BSIM3v0ldvt1 = 0.0;      
        if (!model->BSIM3v0ldvt2Given)
            model->BSIM3v0ldvt2 = 0.0;
        if (!model->BSIM3v0ldvt0wGiven)
            model->BSIM3v0ldvt0w = 0.0;    
        if (!model->BSIM3v0ldvt1wGiven)
            model->BSIM3v0ldvt1w = 0.0;      
        if (!model->BSIM3v0ldvt2wGiven)
            model->BSIM3v0ldvt2w = 0.0;
        if (!model->BSIM3v0ldroutGiven)
            model->BSIM3v0ldrout = 0.0;     
        if (!model->BSIM3v0ldsubGiven)
            model->BSIM3v0ldsub = 0.0;
        if (!model->BSIM3v0lvth0Given)
           model->BSIM3v0lvth0 = 0.0;
        if (!model->BSIM3v0luaGiven)
            model->BSIM3v0lua = 0.0;
        if (!model->BSIM3v0lua1Given)
            model->BSIM3v0lua1 = 0.0;
        if (!model->BSIM3v0lubGiven)
            model->BSIM3v0lub = 0.0;
        if (!model->BSIM3v0lub1Given)
            model->BSIM3v0lub1 = 0.0;
        if (!model->BSIM3v0lucGiven)
            model->BSIM3v0luc = 0.0;
        if (!model->BSIM3v0luc1Given)
            model->BSIM3v0luc1 = 0.0;
        if (!model->BSIM3v0lu0Given)
            model->BSIM3v0lu0 = 0.0;
        else if (model->BSIM3v0u0 > 1.0)
            model->BSIM3v0lu0 = model->BSIM3v0lu0 / 1.0e4;  
        if (!model->BSIM3v0luteGiven)
	    model->BSIM3v0lute = 0.0;    
        if (!model->BSIM3v0lvoffGiven)
	    model->BSIM3v0lvoff = 0.0;
        if (!model->BSIM3v0ldeltaGiven)  
            model->BSIM3v0ldelta = 0.0;
        if (!model->BSIM3v0lrdswGiven)
            model->BSIM3v0lrdsw = 0.0;
        if (!model->BSIM3v0lprwbGiven)
            model->BSIM3v0lprwb = 0.0;
        if (!model->BSIM3v0lprwgGiven)
            model->BSIM3v0lprwg = 0.0;
        if (!model->BSIM3v0lprtGiven)
            model->BSIM3v0lprt = 0.0;
        if (!model->BSIM3v0leta0Given)
            model->BSIM3v0leta0 = 0.0;
        if (!model->BSIM3v0letabGiven)
            model->BSIM3v0letab = -0.0;
        if (!model->BSIM3v0lpclmGiven)
            model->BSIM3v0lpclm = 0.0; 
        if (!model->BSIM3v0lpdibl1Given)
            model->BSIM3v0lpdibl1 = 0.0;
        if (!model->BSIM3v0lpdibl2Given)
            model->BSIM3v0lpdibl2 = 0.0;
        if (!model->BSIM3v0lpdiblbGiven)
            model->BSIM3v0lpdiblb = 0.0;
        if (!model->BSIM3v0lpscbe1Given)
            model->BSIM3v0lpscbe1 = 0.0;
        if (!model->BSIM3v0lpscbe2Given)
            model->BSIM3v0lpscbe2 = 0.0;
        if (!model->BSIM3v0lpvagGiven)
            model->BSIM3v0lpvag = 0.0;     
        if (!model->BSIM3v0lwrGiven)  
            model->BSIM3v0lwr = 0.0;
        if (!model->BSIM3v0ldwgGiven)  
            model->BSIM3v0ldwg = 0.0;
        if (!model->BSIM3v0ldwbGiven)  
            model->BSIM3v0ldwb = 0.0;
        if (!model->BSIM3v0lb0Given)
            model->BSIM3v0lb0 = 0.0;
        if (!model->BSIM3v0lb1Given)  
            model->BSIM3v0lb1 = 0.0;
        if (!model->BSIM3v0lalpha0Given)  
            model->BSIM3v0lalpha0 = 0.0;
        if (!model->BSIM3v0lbeta0Given)  
            model->BSIM3v0lbeta0 = 0.0;

        if (!model->BSIM3v0lelmGiven)  
            model->BSIM3v0lelm = 0.0;
        if (!model->BSIM3v0lcgslGiven)  
            model->BSIM3v0lcgsl = 0.0;
        if (!model->BSIM3v0lcgdlGiven)  
            model->BSIM3v0lcgdl = 0.0;
        if (!model->BSIM3v0lckappaGiven)  
            model->BSIM3v0lckappa = 0.0;
        if (!model->BSIM3v0lcfGiven)  
            model->BSIM3v0lcf = 0.0;
        if (!model->BSIM3v0lclcGiven)  
            model->BSIM3v0lclc = 0.0;
        if (!model->BSIM3v0lcleGiven)  
            model->BSIM3v0lcle = 0.0;

	/* Width dependence */
        if (!model->BSIM3v0wcdscGiven)
	    model->BSIM3v0wcdsc = 0.0;
        if (!model->BSIM3v0wcdscbGiven)
	    model->BSIM3v0wcdscb = 0.0;  
        if (!model->BSIM3v0wcdscdGiven)
	    model->BSIM3v0wcdscd = 0.0;
        if (!model->BSIM3v0wcitGiven)
	    model->BSIM3v0wcit = 0.0;
        if (!model->BSIM3v0wnfactorGiven)
	    model->BSIM3v0wnfactor = 0.0;
        if (!model->BSIM3v0wxjGiven)
            model->BSIM3v0wxj = 0.0;
        if (!model->BSIM3v0wvsatGiven)
            model->BSIM3v0wvsat = 0.0;
        if (!model->BSIM3v0watGiven)
            model->BSIM3v0wat = 0.0;
        if (!model->BSIM3v0wa0Given)
            model->BSIM3v0wa0 = 0.0; 
        if (!model->BSIM3v0wagsGiven)
            model->BSIM3v0wags = 0.0;
        if (!model->BSIM3v0wa1Given)
            model->BSIM3v0wa1 = 0.0;
        if (!model->BSIM3v0wa2Given)
            model->BSIM3v0wa2 = 0.0;
        if (!model->BSIM3v0wketaGiven)
            model->BSIM3v0wketa = 0.0;
        if (!model->BSIM3v0wnsubGiven)
            model->BSIM3v0wnsub = 0.0;
        if (!model->BSIM3v0wnpeakGiven)
            model->BSIM3v0wnpeak = 0.0;
        if (!model->BSIM3v0wvbmGiven)
	    model->BSIM3v0wvbm = 0.0;
        if (!model->BSIM3v0wxtGiven)
	    model->BSIM3v0wxt = 0.0;
        if (!model->BSIM3v0wkt1Given)
            model->BSIM3v0wkt1 = 0.0; 
        if (!model->BSIM3v0wkt1lGiven)
            model->BSIM3v0wkt1l = 0.0;
        if (!model->BSIM3v0wkt2Given)
            model->BSIM3v0wkt2 = 0.0;
        if (!model->BSIM3v0wk3Given)
            model->BSIM3v0wk3 = 0.0;      
        if (!model->BSIM3v0wk3bGiven)
            model->BSIM3v0wk3b = 0.0;      
        if (!model->BSIM3v0ww0Given)
            model->BSIM3v0ww0 = 0.0;    
        if (!model->BSIM3v0wnlxGiven)
            model->BSIM3v0wnlx = 0.0;     
        if (!model->BSIM3v0wdvt0Given)
            model->BSIM3v0wdvt0 = 0.0;    
        if (!model->BSIM3v0wdvt1Given)
            model->BSIM3v0wdvt1 = 0.0;      
        if (!model->BSIM3v0wdvt2Given)
            model->BSIM3v0wdvt2 = 0.0;
        if (!model->BSIM3v0wdvt0wGiven)
            model->BSIM3v0wdvt0w = 0.0;    
        if (!model->BSIM3v0wdvt1wGiven)
            model->BSIM3v0wdvt1w = 0.0;      
        if (!model->BSIM3v0wdvt2wGiven)
            model->BSIM3v0wdvt2w = 0.0;
        if (!model->BSIM3v0wdroutGiven)
            model->BSIM3v0wdrout = 0.0;     
        if (!model->BSIM3v0wdsubGiven)
            model->BSIM3v0wdsub = 0.0;
        if (!model->BSIM3v0wvth0Given)
           model->BSIM3v0wvth0 = 0.0;
        if (!model->BSIM3v0wuaGiven)
            model->BSIM3v0wua = 0.0;
        if (!model->BSIM3v0wua1Given)
            model->BSIM3v0wua1 = 0.0;
        if (!model->BSIM3v0wubGiven)
            model->BSIM3v0wub = 0.0;
        if (!model->BSIM3v0wub1Given)
            model->BSIM3v0wub1 = 0.0;
        if (!model->BSIM3v0wucGiven)
            model->BSIM3v0wuc = 0.0;
        if (!model->BSIM3v0wuc1Given)
            model->BSIM3v0wuc1 = 0.0;
        if (!model->BSIM3v0wu0Given)
            model->BSIM3v0wu0 = 0.0;
        else if (model->BSIM3v0u0 > 1.0)
            model->BSIM3v0wu0 = model->BSIM3v0wu0 / 1.0e4;
        if (!model->BSIM3v0wuteGiven)
	    model->BSIM3v0wute = 0.0;    
        if (!model->BSIM3v0wvoffGiven)
	    model->BSIM3v0wvoff = 0.0;
        if (!model->BSIM3v0wdeltaGiven)  
            model->BSIM3v0wdelta = 0.0;
        if (!model->BSIM3v0wrdswGiven)
            model->BSIM3v0wrdsw = 0.0;
        if (!model->BSIM3v0wprwbGiven)
            model->BSIM3v0wprwb = 0.0;
        if (!model->BSIM3v0wprwgGiven)
            model->BSIM3v0wprwg = 0.0;
        if (!model->BSIM3v0wprtGiven)
            model->BSIM3v0wprt = 0.0;
        if (!model->BSIM3v0weta0Given)
            model->BSIM3v0weta0 = 0.0;
        if (!model->BSIM3v0wetabGiven)
            model->BSIM3v0wetab = 0.0;
        if (!model->BSIM3v0wpclmGiven)
            model->BSIM3v0wpclm = 0.0; 
        if (!model->BSIM3v0wpdibl1Given)
            model->BSIM3v0wpdibl1 = 0.0;
        if (!model->BSIM3v0wpdibl2Given)
            model->BSIM3v0wpdibl2 = 0.0;
        if (!model->BSIM3v0wpdiblbGiven)
            model->BSIM3v0wpdiblb = 0.0;
        if (!model->BSIM3v0wpscbe1Given)
            model->BSIM3v0wpscbe1 = 0.0;
        if (!model->BSIM3v0wpscbe2Given)
            model->BSIM3v0wpscbe2 = 0.0;
        if (!model->BSIM3v0wpvagGiven)
            model->BSIM3v0wpvag = 0.0;     
        if (!model->BSIM3v0wwrGiven)  
            model->BSIM3v0wwr = 0.0;
        if (!model->BSIM3v0wdwgGiven)  
            model->BSIM3v0wdwg = 0.0;
        if (!model->BSIM3v0wdwbGiven)  
            model->BSIM3v0wdwb = 0.0;
        if (!model->BSIM3v0wb0Given)
            model->BSIM3v0wb0 = 0.0;
        if (!model->BSIM3v0wb1Given)  
            model->BSIM3v0wb1 = 0.0;
        if (!model->BSIM3v0walpha0Given)  
            model->BSIM3v0walpha0 = 0.0;
        if (!model->BSIM3v0wbeta0Given)  
            model->BSIM3v0wbeta0 = 0.0;

        if (!model->BSIM3v0welmGiven)  
            model->BSIM3v0welm = 0.0;
        if (!model->BSIM3v0wcgslGiven)  
            model->BSIM3v0wcgsl = 0.0;
        if (!model->BSIM3v0wcgdlGiven)  
            model->BSIM3v0wcgdl = 0.0;
        if (!model->BSIM3v0wckappaGiven)  
            model->BSIM3v0wckappa = 0.0;
        if (!model->BSIM3v0wcfGiven)  
            model->BSIM3v0wcf = 0.0;
        if (!model->BSIM3v0wclcGiven)  
            model->BSIM3v0wclc = 0.0;
        if (!model->BSIM3v0wcleGiven)  
            model->BSIM3v0wcle = 0.0;

	/* Cross-term dependence */
        if (!model->BSIM3v0pcdscGiven)
	    model->BSIM3v0pcdsc = 0.0;
        if (!model->BSIM3v0pcdscbGiven)
	    model->BSIM3v0pcdscb = 0.0;   
        if (!model->BSIM3v0pcdscdGiven)
	    model->BSIM3v0pcdscd = 0.0;
        if (!model->BSIM3v0pcitGiven)
	    model->BSIM3v0pcit = 0.0;
        if (!model->BSIM3v0pnfactorGiven)
	    model->BSIM3v0pnfactor = 0.0;
        if (!model->BSIM3v0pxjGiven)
            model->BSIM3v0pxj = 0.0;
        if (!model->BSIM3v0pvsatGiven)
            model->BSIM3v0pvsat = 0.0;
        if (!model->BSIM3v0patGiven)
            model->BSIM3v0pat = 0.0;
        if (!model->BSIM3v0pa0Given)
            model->BSIM3v0pa0 = 0.0; 
            
        if (!model->BSIM3v0pagsGiven)
            model->BSIM3v0pags = 0.0;
        if (!model->BSIM3v0pa1Given)
            model->BSIM3v0pa1 = 0.0;
        if (!model->BSIM3v0pa2Given)
            model->BSIM3v0pa2 = 0.0;
        if (!model->BSIM3v0pketaGiven)
            model->BSIM3v0pketa = 0.0;
        if (!model->BSIM3v0pnsubGiven)
            model->BSIM3v0pnsub = 0.0;
        if (!model->BSIM3v0pnpeakGiven)
            model->BSIM3v0pnpeak = 0.0;
        if (!model->BSIM3v0pvbmGiven)
	    model->BSIM3v0pvbm = 0.0;
        if (!model->BSIM3v0pxtGiven)
	    model->BSIM3v0pxt = 0.0;
        if (!model->BSIM3v0pkt1Given)
            model->BSIM3v0pkt1 = 0.0; 
        if (!model->BSIM3v0pkt1lGiven)
            model->BSIM3v0pkt1l = 0.0;
        if (!model->BSIM3v0pkt2Given)
            model->BSIM3v0pkt2 = 0.0;
        if (!model->BSIM3v0pk3Given)
            model->BSIM3v0pk3 = 0.0;      
        if (!model->BSIM3v0pk3bGiven)
            model->BSIM3v0pk3b = 0.0;      
        if (!model->BSIM3v0pw0Given)
            model->BSIM3v0pw0 = 0.0;    
        if (!model->BSIM3v0pnlxGiven)
            model->BSIM3v0pnlx = 0.0;     
        if (!model->BSIM3v0pdvt0Given)
            model->BSIM3v0pdvt0 = 0.0;    
        if (!model->BSIM3v0pdvt1Given)
            model->BSIM3v0pdvt1 = 0.0;      
        if (!model->BSIM3v0pdvt2Given)
            model->BSIM3v0pdvt2 = 0.0;
        if (!model->BSIM3v0pdvt0wGiven)
            model->BSIM3v0pdvt0w = 0.0;    
        if (!model->BSIM3v0pdvt1wGiven)
            model->BSIM3v0pdvt1w = 0.0;      
        if (!model->BSIM3v0pdvt2wGiven)
            model->BSIM3v0pdvt2w = 0.0;
        if (!model->BSIM3v0pdroutGiven)
            model->BSIM3v0pdrout = 0.0;     
        if (!model->BSIM3v0pdsubGiven)
            model->BSIM3v0pdsub = 0.0;
        if (!model->BSIM3v0pvth0Given)
           model->BSIM3v0pvth0 = 0.0;
        if (!model->BSIM3v0puaGiven)
            model->BSIM3v0pua = 0.0;
        if (!model->BSIM3v0pua1Given)
            model->BSIM3v0pua1 = 0.0;
        if (!model->BSIM3v0pubGiven)
            model->BSIM3v0pub = 0.0;
        if (!model->BSIM3v0pub1Given)
            model->BSIM3v0pub1 = 0.0;
        if (!model->BSIM3v0pucGiven)
            model->BSIM3v0puc = 0.0;
        if (!model->BSIM3v0puc1Given)
            model->BSIM3v0puc1 = 0.0;
        if (!model->BSIM3v0pu0Given)
            model->BSIM3v0pu0 = 0.0;
        else if (model->BSIM3v0u0 > 1.0)
            model->BSIM3v0pu0 = model->BSIM3v0pu0 / 1.0e4;  
        if (!model->BSIM3v0puteGiven)
	    model->BSIM3v0pute = 0.0;    
        if (!model->BSIM3v0pvoffGiven)
	    model->BSIM3v0pvoff = 0.0;
        if (!model->BSIM3v0pdeltaGiven)  
            model->BSIM3v0pdelta = 0.0;
        if (!model->BSIM3v0prdswGiven)
            model->BSIM3v0prdsw = 0.0;
        if (!model->BSIM3v0pprwbGiven)
            model->BSIM3v0pprwb = 0.0;
        if (!model->BSIM3v0pprwgGiven)
            model->BSIM3v0pprwg = 0.0;
        if (!model->BSIM3v0pprtGiven)
            model->BSIM3v0pprt = 0.0;
        if (!model->BSIM3v0peta0Given)
            model->BSIM3v0peta0 = 0.0;
        if (!model->BSIM3v0petabGiven)
            model->BSIM3v0petab = 0.0;
        if (!model->BSIM3v0ppclmGiven)
            model->BSIM3v0ppclm = 0.0; 
        if (!model->BSIM3v0ppdibl1Given)
            model->BSIM3v0ppdibl1 = 0.0;
        if (!model->BSIM3v0ppdibl2Given)
            model->BSIM3v0ppdibl2 = 0.0;
        if (!model->BSIM3v0ppdiblbGiven)
            model->BSIM3v0ppdiblb = 0.0;
        if (!model->BSIM3v0ppscbe1Given)
            model->BSIM3v0ppscbe1 = 0.0;
        if (!model->BSIM3v0ppscbe2Given)
            model->BSIM3v0ppscbe2 = 0.0;
        if (!model->BSIM3v0ppvagGiven)
            model->BSIM3v0ppvag = 0.0;     
        if (!model->BSIM3v0pwrGiven)  
            model->BSIM3v0pwr = 0.0;
        if (!model->BSIM3v0pdwgGiven)  
            model->BSIM3v0pdwg = 0.0;
        if (!model->BSIM3v0pdwbGiven)  
            model->BSIM3v0pdwb = 0.0;
        if (!model->BSIM3v0pb0Given)
            model->BSIM3v0pb0 = 0.0;
        if (!model->BSIM3v0pb1Given)  
            model->BSIM3v0pb1 = 0.0;
        if (!model->BSIM3v0palpha0Given)  
            model->BSIM3v0palpha0 = 0.0;
        if (!model->BSIM3v0pbeta0Given)  
            model->BSIM3v0pbeta0 = 0.0;

        if (!model->BSIM3v0pelmGiven)  
            model->BSIM3v0pelm = 0.0;
        if (!model->BSIM3v0pcgslGiven)  
            model->BSIM3v0pcgsl = 0.0;
        if (!model->BSIM3v0pcgdlGiven)  
            model->BSIM3v0pcgdl = 0.0;
        if (!model->BSIM3v0pckappaGiven)  
            model->BSIM3v0pckappa = 0.0;
        if (!model->BSIM3v0pcfGiven)  
            model->BSIM3v0pcf = 0.0;
        if (!model->BSIM3v0pclcGiven)  
            model->BSIM3v0pclc = 0.0;
        if (!model->BSIM3v0pcleGiven)  
            model->BSIM3v0pcle = 0.0;

        /* unit degree celcius */
        if (!model->BSIM3v0tnomGiven)  
	    model->BSIM3v0tnom = ckt->CKTnomTemp; 
        if (!model->BSIM3v0LintGiven)  
           model->BSIM3v0Lint = 0.0;
        if (!model->BSIM3v0LlGiven)  
           model->BSIM3v0Ll = 0.0;
        if (!model->BSIM3v0LlnGiven)  
           model->BSIM3v0Lln = 1.0;
        if (!model->BSIM3v0LwGiven)  
           model->BSIM3v0Lw = 0.0;
        if (!model->BSIM3v0LwnGiven)  
           model->BSIM3v0Lwn = 1.0;
        if (!model->BSIM3v0LwlGiven)  
           model->BSIM3v0Lwl = 0.0;
        if (!model->BSIM3v0LminGiven)  
           model->BSIM3v0Lmin = 0.0;
        if (!model->BSIM3v0LmaxGiven)  
           model->BSIM3v0Lmax = 1.0;
        if (!model->BSIM3v0WintGiven)  
           model->BSIM3v0Wint = 0.0;
        if (!model->BSIM3v0WlGiven)  
           model->BSIM3v0Wl = 0.0;
        if (!model->BSIM3v0WlnGiven)  
           model->BSIM3v0Wln = 1.0;
        if (!model->BSIM3v0WwGiven)  
           model->BSIM3v0Ww = 0.0;
        if (!model->BSIM3v0WwnGiven)  
           model->BSIM3v0Wwn = 1.0;
        if (!model->BSIM3v0WwlGiven)  
           model->BSIM3v0Wwl = 0.0;
        if (!model->BSIM3v0WminGiven)  
           model->BSIM3v0Wmin = 0.0;
        if (!model->BSIM3v0WmaxGiven)  
           model->BSIM3v0Wmax = 1.0;
        if (!model->BSIM3v0dwcGiven)  
           model->BSIM3v0dwc = model->BSIM3v0Wint;
        if (!model->BSIM3v0dlcGiven)  
           model->BSIM3v0dlc = model->BSIM3v0Lint;
        if (!model->BSIM3v0cfGiven)
            model->BSIM3v0cf = 2.0 * EPSOX / PI
			   * log(1.0 + 0.4e-6 / model->BSIM3v0tox);
        if (!model->BSIM3v0cgdoGiven)
	{   if (model->BSIM3v0dlcGiven && (model->BSIM3v0dlc > 0.0))
	    {   model->BSIM3v0cgdo = model->BSIM3v0dlc * model->BSIM3v0cox
				 - model->BSIM3v0cgdl ;
		if (model->BSIM3v0cgdo < 0.0)
		    model->BSIM3v0cgdo = 0.0;
	    }
	    else
	        model->BSIM3v0cgdo = 0.6 * model->BSIM3v0xj * model->BSIM3v0cox; 
	}
        if (!model->BSIM3v0cgsoGiven)
	{   if (model->BSIM3v0dlcGiven && (model->BSIM3v0dlc > 0.0))
	    {   model->BSIM3v0cgso = model->BSIM3v0dlc * model->BSIM3v0cox
				 - model->BSIM3v0cgsl ;
		if (model->BSIM3v0cgso < 0.0)
		    model->BSIM3v0cgso = 0.0;
	    }
	    else
	        model->BSIM3v0cgso = 0.6 * model->BSIM3v0xj * model->BSIM3v0cox; 
	}

        if (!model->BSIM3v0cgboGiven)
            model->BSIM3v0cgbo = 0.0;
        if (!model->BSIM3v0xpartGiven)
            model->BSIM3v0xpart = 0.0;
        if (!model->BSIM3v0sheetResistanceGiven)
            model->BSIM3v0sheetResistance = 0.0;
        if (!model->BSIM3v0unitAreaJctCapGiven)
            model->BSIM3v0unitAreaJctCap = 5.0E-4;
        if (!model->BSIM3v0unitLengthSidewallJctCapGiven)
            model->BSIM3v0unitLengthSidewallJctCap = 5.0E-10;
        if (!model->BSIM3v0jctSatCurDensityGiven)
            model->BSIM3v0jctSatCurDensity = 1.0E-4;
        if (!model->BSIM3v0bulkJctPotentialGiven)
            model->BSIM3v0bulkJctPotential = 1.0;
        if (!model->BSIM3v0sidewallJctPotentialGiven)
            model->BSIM3v0sidewallJctPotential = 1.0;
        if (!model->BSIM3v0bulkJctBotGradingCoeffGiven)
            model->BSIM3v0bulkJctBotGradingCoeff = 0.5;
        if (!model->BSIM3v0bulkJctSideGradingCoeffGiven)
            model->BSIM3v0bulkJctSideGradingCoeff = 0.33;
        if (!model->BSIM3v0oxideTrapDensityAGiven)
	{   if (model->BSIM3v0type == NMOS)
                model->BSIM3v0oxideTrapDensityA = 1e20;
            else
                model->BSIM3v0oxideTrapDensityA=9.9e18;
	}
        if (!model->BSIM3v0oxideTrapDensityBGiven)
	{   if (model->BSIM3v0type == NMOS)
                model->BSIM3v0oxideTrapDensityB = 5e4;
            else
                model->BSIM3v0oxideTrapDensityB = 2.4e3;
	}
        if (!model->BSIM3v0oxideTrapDensityCGiven)
	{   if (model->BSIM3v0type == NMOS)
                model->BSIM3v0oxideTrapDensityC = -1.4e-12;
            else
                model->BSIM3v0oxideTrapDensityC = 1.4e-12;

	}
        if (!model->BSIM3v0emGiven)
            model->BSIM3v0em = 4.1e7; /* V/m */
        if (!model->BSIM3v0efGiven)
            model->BSIM3v0ef = 1.0;
        if (!model->BSIM3v0afGiven)
            model->BSIM3v0af = 1.0;
        if (!model->BSIM3v0kfGiven)
            model->BSIM3v0kf = 0.0;
        /* loop through all the instances of the model */
        for (here = BSIM3v0instances(model); here != NULL ;
             here=BSIM3v0nextInstance(here)) 
	{
            /* allocate a chunk of the state vector */
            here->BSIM3v0states = *states;
            *states += BSIM3v0numStates;    

	    /* perform the parameter defaulting */
	    if (!here->BSIM3v0drainAreaGiven)
                here->BSIM3v0drainArea = 0.0;
            if (!here->BSIM3v0drainPerimeterGiven)
                here->BSIM3v0drainPerimeter = 0.0;
            if (!here->BSIM3v0drainSquaresGiven)
                here->BSIM3v0drainSquares = 1.0;
            if (!here->BSIM3v0icVBSGiven)
                here->BSIM3v0icVBS = 0;
            if (!here->BSIM3v0icVDSGiven)
                here->BSIM3v0icVDS = 0;
            if (!here->BSIM3v0icVGSGiven)
                here->BSIM3v0icVGS = 0;
            if (!here->BSIM3v0lGiven)
                here->BSIM3v0l = 5e-6;
            if (!here->BSIM3v0sourceAreaGiven)
                here->BSIM3v0sourceArea = 0;
            if (!here->BSIM3v0sourcePerimeterGiven)
                here->BSIM3v0sourcePerimeter = 0;
            if (!here->BSIM3v0sourceSquaresGiven)
                here->BSIM3v0sourceSquares = 1;
            if (!here->BSIM3v0wGiven)
                here->BSIM3v0w = 5e-6;

            if (!here->BSIM3v0mGiven)
                here->BSIM3v0m = 1;

            if (!here->BSIM3v0nqsModGiven)
                here->BSIM3v0nqsMod = model->BSIM3v0nqsMod;
                    
            /* process drain series resistance */
            if ((model->BSIM3v0sheetResistance > 0.0) && 
                (here->BSIM3v0drainSquares > 0.0 ))
            {   if(here->BSIM3v0dNodePrime == 0)
	    {   error = CKTmkVolt(ckt,&tmp,here->BSIM3v0name,"drain");
                if(error) return(error);
                here->BSIM3v0dNodePrime = tmp->number;
		
		if (ckt->CKTcopyNodesets) {
                  if (CKTinst2Node(ckt,here,1,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }
            }
            }
	    else
	    {   here->BSIM3v0dNodePrime = here->BSIM3v0dNode;
            }
                   
            /* process source series resistance */
            if ((model->BSIM3v0sheetResistance > 0.0) && 
                (here->BSIM3v0sourceSquares > 0.0 ))
            {   if(here->BSIM3v0sNodePrime == 0)
	    {   error = CKTmkVolt(ckt,&tmp,here->BSIM3v0name,"source");
                if(error) return(error);
                here->BSIM3v0sNodePrime = tmp->number;
		
		if (ckt->CKTcopyNodesets) {
                  if (CKTinst2Node(ckt,here,3,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }
		
            }
            }
	    else 
	    {   here->BSIM3v0sNodePrime = here->BSIM3v0sNode;
            }

 /* internal charge node */
                   
            if ((here->BSIM3v0nqsMod))
            {   if(here->BSIM3v0qNode == 0)
	    {   error = CKTmkVolt(ckt,&tmp,here->BSIM3v0name,"charge");
                if(error) return(error);
                here->BSIM3v0qNode = tmp->number;
            }
            }
	    else 
	    {   here->BSIM3v0qNode = 0;
            }

        /* set Sparse Matrix Pointers */

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(BSIM3v0DdPtr, BSIM3v0dNode, BSIM3v0dNode);
            TSTALLOC(BSIM3v0GgPtr, BSIM3v0gNode, BSIM3v0gNode);
            TSTALLOC(BSIM3v0SsPtr, BSIM3v0sNode, BSIM3v0sNode);
            TSTALLOC(BSIM3v0BbPtr, BSIM3v0bNode, BSIM3v0bNode);
            TSTALLOC(BSIM3v0DPdpPtr, BSIM3v0dNodePrime, BSIM3v0dNodePrime);
            TSTALLOC(BSIM3v0SPspPtr, BSIM3v0sNodePrime, BSIM3v0sNodePrime);
            TSTALLOC(BSIM3v0DdpPtr, BSIM3v0dNode, BSIM3v0dNodePrime);
            TSTALLOC(BSIM3v0GbPtr, BSIM3v0gNode, BSIM3v0bNode);
            TSTALLOC(BSIM3v0GdpPtr, BSIM3v0gNode, BSIM3v0dNodePrime);
            TSTALLOC(BSIM3v0GspPtr, BSIM3v0gNode, BSIM3v0sNodePrime);
            TSTALLOC(BSIM3v0SspPtr, BSIM3v0sNode, BSIM3v0sNodePrime);
            TSTALLOC(BSIM3v0BdpPtr, BSIM3v0bNode, BSIM3v0dNodePrime);
            TSTALLOC(BSIM3v0BspPtr, BSIM3v0bNode, BSIM3v0sNodePrime);
            TSTALLOC(BSIM3v0DPspPtr, BSIM3v0dNodePrime, BSIM3v0sNodePrime);
            TSTALLOC(BSIM3v0DPdPtr, BSIM3v0dNodePrime, BSIM3v0dNode);
            TSTALLOC(BSIM3v0BgPtr, BSIM3v0bNode, BSIM3v0gNode);
            TSTALLOC(BSIM3v0DPgPtr, BSIM3v0dNodePrime, BSIM3v0gNode);
            TSTALLOC(BSIM3v0SPgPtr, BSIM3v0sNodePrime, BSIM3v0gNode);
            TSTALLOC(BSIM3v0SPsPtr, BSIM3v0sNodePrime, BSIM3v0sNode);
            TSTALLOC(BSIM3v0DPbPtr, BSIM3v0dNodePrime, BSIM3v0bNode);
            TSTALLOC(BSIM3v0SPbPtr, BSIM3v0sNodePrime, BSIM3v0bNode);
            TSTALLOC(BSIM3v0SPdpPtr, BSIM3v0sNodePrime, BSIM3v0dNodePrime);

            TSTALLOC(BSIM3v0QqPtr, BSIM3v0qNode, BSIM3v0qNode);
 
            TSTALLOC(BSIM3v0QdpPtr, BSIM3v0qNode, BSIM3v0dNodePrime);
            TSTALLOC(BSIM3v0QspPtr, BSIM3v0qNode, BSIM3v0sNodePrime);
            TSTALLOC(BSIM3v0QgPtr, BSIM3v0qNode, BSIM3v0gNode);
            TSTALLOC(BSIM3v0QbPtr, BSIM3v0qNode, BSIM3v0bNode);
            TSTALLOC(BSIM3v0DPqPtr, BSIM3v0dNodePrime, BSIM3v0qNode);
            TSTALLOC(BSIM3v0SPqPtr, BSIM3v0sNodePrime, BSIM3v0qNode);
            TSTALLOC(BSIM3v0GqPtr, BSIM3v0gNode, BSIM3v0qNode);
            TSTALLOC(BSIM3v0BqPtr, BSIM3v0bNode, BSIM3v0qNode);

        }
    }
    return(OK);
}  


int
BSIM3v0unsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v0model *model;
    BSIM3v0instance *here;

    for (model = (BSIM3v0model *)inModel; model != NULL;
            model = BSIM3v0nextModel(model))
    {
        for (here = BSIM3v0instances(model); here != NULL;
                here=BSIM3v0nextInstance(here))
        {
            if (here->BSIM3v0qNode > 0)
                CKTdltNNum(ckt, here->BSIM3v0qNode);
            here->BSIM3v0qNode = 0;

            if (here->BSIM3v0sNodePrime > 0
                    && here->BSIM3v0sNodePrime != here->BSIM3v0sNode)
                CKTdltNNum(ckt, here->BSIM3v0sNodePrime);
            here->BSIM3v0sNodePrime = 0;

            if (here->BSIM3v0dNodePrime > 0
                    && here->BSIM3v0dNodePrime != here->BSIM3v0dNode)
                CKTdltNNum(ckt, here->BSIM3v0dNodePrime);
            here->BSIM3v0dNodePrime = 0;
        }
    }
    return OK;
}


