/* $Id$  */
/* 
$Log$
Revision 1.1  2000-04-27 20:03:59  pnenzi
Initial revision

 * Revision 3.1  96/12/08  19:59:17  yuhua
 * BSIM3v3.1 release
 * 
*/
static char rcsid[] = "$Id$";

/*************************************/

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v1set.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "smpdefs.h"
#include "cktdefs.h"
#include "bsim3v1def.h"
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
BSIM3V1setup(matrix,inModel,ckt,states)
register SMPmatrix *matrix;
register GENmodel *inModel;
register CKTcircuit *ckt;
int *states;
{
register BSIM3V1model *model = (BSIM3V1model*)inModel;
register BSIM3V1instance *here;
int error;
CKTnode *tmp;

double tmp1, tmp2;

    /*  loop through all the BSIM3V1 device models */
    for( ; model != NULL; model = model->BSIM3V1nextModel )
    {
/* Default value Processing for BSIM3V1 MOSFET Models */
        if (!model->BSIM3V1typeGiven)
            model->BSIM3V1type = NMOS;     
        if (!model->BSIM3V1mobModGiven) 
            model->BSIM3V1mobMod = 1;
        if (!model->BSIM3V1binUnitGiven) 
            model->BSIM3V1binUnit = 1;
        if (!model->BSIM3V1paramChkGiven) 
            model->BSIM3V1paramChk = 0;
        if (!model->BSIM3V1capModGiven) 
            model->BSIM3V1capMod = 2;
        if (!model->BSIM3V1nqsModGiven) 
            model->BSIM3V1nqsMod = 0;
        if (!model->BSIM3V1noiModGiven) 
            model->BSIM3V1noiMod = 1;
        if (!model->BSIM3V1versionGiven) 
            model->BSIM3V1version = 3.1;
        if (!model->BSIM3V1toxGiven)
            model->BSIM3V1tox = 150.0e-10;
        model->BSIM3V1cox = 3.453133e-11 / model->BSIM3V1tox;

        if (!model->BSIM3V1cdscGiven)
	    model->BSIM3V1cdsc = 2.4e-4;   /* unit Q/V/m^2  */
        if (!model->BSIM3V1cdscbGiven)
	    model->BSIM3V1cdscb = 0.0;   /* unit Q/V/m^2  */    
	    if (!model->BSIM3V1cdscdGiven)
	    model->BSIM3V1cdscd = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM3V1citGiven)
	    model->BSIM3V1cit = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM3V1nfactorGiven)
	    model->BSIM3V1nfactor = 1;
        if (!model->BSIM3V1xjGiven)
            model->BSIM3V1xj = .15e-6;
        if (!model->BSIM3V1vsatGiven)
            model->BSIM3V1vsat = 8.0e4;    /* unit m/s */ 
        if (!model->BSIM3V1atGiven)
            model->BSIM3V1at = 3.3e4;    /* unit m/s */ 
        if (!model->BSIM3V1a0Given)
            model->BSIM3V1a0 = 1.0;  
        if (!model->BSIM3V1agsGiven)
            model->BSIM3V1ags = 0.0;
        if (!model->BSIM3V1a1Given)
            model->BSIM3V1a1 = 0.0;
        if (!model->BSIM3V1a2Given)
            model->BSIM3V1a2 = 1.0;
        if (!model->BSIM3V1ketaGiven)
            model->BSIM3V1keta = -0.047;    /* unit  / V */
        if (!model->BSIM3V1nsubGiven)
            model->BSIM3V1nsub = 6.0e16;   /* unit 1/cm3 */
        if (!model->BSIM3V1npeakGiven)
            model->BSIM3V1npeak = 1.7e17;   /* unit 1/cm3 */
        if (!model->BSIM3V1ngateGiven)
            model->BSIM3V1ngate = 0;   /* unit 1/cm3 */
        if (!model->BSIM3V1vbmGiven)
	    model->BSIM3V1vbm = -3.0;
        if (!model->BSIM3V1xtGiven)
	    model->BSIM3V1xt = 1.55e-7;
        if (!model->BSIM3V1kt1Given)
            model->BSIM3V1kt1 = -0.11;      /* unit V */
        if (!model->BSIM3V1kt1lGiven)
            model->BSIM3V1kt1l = 0.0;      /* unit V*m */
        if (!model->BSIM3V1kt2Given)
            model->BSIM3V1kt2 = 0.022;      /* No unit */
        if (!model->BSIM3V1k3Given)
            model->BSIM3V1k3 = 80.0;      
        if (!model->BSIM3V1k3bGiven)
            model->BSIM3V1k3b = 0.0;      
        if (!model->BSIM3V1w0Given)
            model->BSIM3V1w0 = 2.5e-6;    
        if (!model->BSIM3V1nlxGiven)
            model->BSIM3V1nlx = 1.74e-7;     
        if (!model->BSIM3V1dvt0Given)
            model->BSIM3V1dvt0 = 2.2;    
        if (!model->BSIM3V1dvt1Given)
            model->BSIM3V1dvt1 = 0.53;      
        if (!model->BSIM3V1dvt2Given)
            model->BSIM3V1dvt2 = -0.032;   /* unit 1 / V */     

        if (!model->BSIM3V1dvt0wGiven)
            model->BSIM3V1dvt0w = 0.0;    
        if (!model->BSIM3V1dvt1wGiven)
            model->BSIM3V1dvt1w = 5.3e6;    
        if (!model->BSIM3V1dvt2wGiven)
            model->BSIM3V1dvt2w = -0.032;   

        if (!model->BSIM3V1droutGiven)
            model->BSIM3V1drout = 0.56;     
        if (!model->BSIM3V1dsubGiven)
            model->BSIM3V1dsub = model->BSIM3V1drout;     
        if (!model->BSIM3V1vth0Given)
            model->BSIM3V1vth0 = (model->BSIM3V1type == NMOS) ? 0.7 : -0.7;
        if (!model->BSIM3V1uaGiven)
            model->BSIM3V1ua = 2.25e-9;      /* unit m/V */
        if (!model->BSIM3V1ua1Given)
            model->BSIM3V1ua1 = 4.31e-9;      /* unit m/V */
        if (!model->BSIM3V1ubGiven)
            model->BSIM3V1ub = 5.87e-19;     /* unit (m/V)**2 */
        if (!model->BSIM3V1ub1Given)
            model->BSIM3V1ub1 = -7.61e-18;     /* unit (m/V)**2 */
        if (!model->BSIM3V1ucGiven)
            model->BSIM3V1uc = (model->BSIM3V1mobMod == 3) ? -0.0465 : -0.0465e-9;   
        if (!model->BSIM3V1uc1Given)
            model->BSIM3V1uc1 = (model->BSIM3V1mobMod == 3) ? -0.056 : -0.056e-9;   
        if (!model->BSIM3V1u0Given)
            model->BSIM3V1u0 = (model->BSIM3V1type == NMOS) ? 0.067 : 0.025;
        if (!model->BSIM3V1uteGiven)
	    model->BSIM3V1ute = -1.5;    
        if (!model->BSIM3V1voffGiven)
	    model->BSIM3V1voff = -0.08;
        if (!model->BSIM3V1deltaGiven)  
           model->BSIM3V1delta = 0.01;
        if (!model->BSIM3V1rdswGiven)
            model->BSIM3V1rdsw = 0;      
        if (!model->BSIM3V1prwgGiven)
            model->BSIM3V1prwg = 0.0;      /* unit 1/V */
        if (!model->BSIM3V1prwbGiven)
            model->BSIM3V1prwb = 0.0;      
        if (!model->BSIM3V1prtGiven)
        if (!model->BSIM3V1prtGiven)
            model->BSIM3V1prt = 0.0;      
        if (!model->BSIM3V1eta0Given)
            model->BSIM3V1eta0 = 0.08;      /* no unit  */ 
        if (!model->BSIM3V1etabGiven)
            model->BSIM3V1etab = -0.07;      /* unit  1/V */ 
        if (!model->BSIM3V1pclmGiven)
            model->BSIM3V1pclm = 1.3;      /* no unit  */ 
        if (!model->BSIM3V1pdibl1Given)
            model->BSIM3V1pdibl1 = .39;    /* no unit  */
        if (!model->BSIM3V1pdibl2Given)
            model->BSIM3V1pdibl2 = 0.0086;    /* no unit  */ 
        if (!model->BSIM3V1pdiblbGiven)
            model->BSIM3V1pdiblb = 0.0;    /* 1/V  */ 
        if (!model->BSIM3V1pscbe1Given)
            model->BSIM3V1pscbe1 = 4.24e8;     
        if (!model->BSIM3V1pscbe2Given)
            model->BSIM3V1pscbe2 = 1.0e-5;    
        if (!model->BSIM3V1pvagGiven)
            model->BSIM3V1pvag = 0.0;     
        if (!model->BSIM3V1wrGiven)  
            model->BSIM3V1wr = 1.0;
        if (!model->BSIM3V1dwgGiven)  
            model->BSIM3V1dwg = 0.0;
        if (!model->BSIM3V1dwbGiven)  
            model->BSIM3V1dwb = 0.0;
        if (!model->BSIM3V1b0Given)
            model->BSIM3V1b0 = 0.0;
        if (!model->BSIM3V1b1Given)  
            model->BSIM3V1b1 = 0.0;
        if (!model->BSIM3V1alpha0Given)  
            model->BSIM3V1alpha0 = 0.0;
        if (!model->BSIM3V1beta0Given)  
            model->BSIM3V1beta0 = 30.0;

        if (!model->BSIM3V1elmGiven)  
            model->BSIM3V1elm = 5.0;
        if (!model->BSIM3V1cgslGiven)  
            model->BSIM3V1cgsl = 0.0;
        if (!model->BSIM3V1cgdlGiven)  
            model->BSIM3V1cgdl = 0.0;
        if (!model->BSIM3V1ckappaGiven)  
            model->BSIM3V1ckappa = 0.6;
        if (!model->BSIM3V1clcGiven)  
            model->BSIM3V1clc = 0.1e-6;
        if (!model->BSIM3V1cleGiven)  
            model->BSIM3V1cle = 0.6;
        if (!model->BSIM3V1vfbcvGiven)  
            model->BSIM3V1vfbcv = -1.0;

	/* Length dependence */
        if (!model->BSIM3V1lcdscGiven)
	    model->BSIM3V1lcdsc = 0.0;
        if (!model->BSIM3V1lcdscbGiven)
	    model->BSIM3V1lcdscb = 0.0;
	    if (!model->BSIM3V1lcdscdGiven) 
	    model->BSIM3V1lcdscd = 0.0;
        if (!model->BSIM3V1lcitGiven)
	    model->BSIM3V1lcit = 0.0;
        if (!model->BSIM3V1lnfactorGiven)
	    model->BSIM3V1lnfactor = 0.0;
        if (!model->BSIM3V1lxjGiven)
            model->BSIM3V1lxj = 0.0;
        if (!model->BSIM3V1lvsatGiven)
            model->BSIM3V1lvsat = 0.0;
        if (!model->BSIM3V1latGiven)
            model->BSIM3V1lat = 0.0;
        if (!model->BSIM3V1la0Given)
            model->BSIM3V1la0 = 0.0; 
        if (!model->BSIM3V1lagsGiven)
            model->BSIM3V1lags = 0.0;
        if (!model->BSIM3V1la1Given)
            model->BSIM3V1la1 = 0.0;
        if (!model->BSIM3V1la2Given)
            model->BSIM3V1la2 = 0.0;
        if (!model->BSIM3V1lketaGiven)
            model->BSIM3V1lketa = 0.0;
        if (!model->BSIM3V1lnsubGiven)
            model->BSIM3V1lnsub = 0.0;
        if (!model->BSIM3V1lnpeakGiven)
            model->BSIM3V1lnpeak = 0.0;
        if (!model->BSIM3V1lngateGiven)
            model->BSIM3V1lngate = 0.0;
        if (!model->BSIM3V1lvbmGiven)
	    model->BSIM3V1lvbm = 0.0;
        if (!model->BSIM3V1lxtGiven)
	    model->BSIM3V1lxt = 0.0;
        if (!model->BSIM3V1lkt1Given)
            model->BSIM3V1lkt1 = 0.0; 
        if (!model->BSIM3V1lkt1lGiven)
            model->BSIM3V1lkt1l = 0.0;
        if (!model->BSIM3V1lkt2Given)
            model->BSIM3V1lkt2 = 0.0;
        if (!model->BSIM3V1lk3Given)
            model->BSIM3V1lk3 = 0.0;      
        if (!model->BSIM3V1lk3bGiven)
            model->BSIM3V1lk3b = 0.0;      
        if (!model->BSIM3V1lw0Given)
            model->BSIM3V1lw0 = 0.0;    
        if (!model->BSIM3V1lnlxGiven)
            model->BSIM3V1lnlx = 0.0;     
        if (!model->BSIM3V1ldvt0Given)
            model->BSIM3V1ldvt0 = 0.0;    
        if (!model->BSIM3V1ldvt1Given)
            model->BSIM3V1ldvt1 = 0.0;      
        if (!model->BSIM3V1ldvt2Given)
            model->BSIM3V1ldvt2 = 0.0;
        if (!model->BSIM3V1ldvt0wGiven)
            model->BSIM3V1ldvt0w = 0.0;    
        if (!model->BSIM3V1ldvt1wGiven)
            model->BSIM3V1ldvt1w = 0.0;      
        if (!model->BSIM3V1ldvt2wGiven)
            model->BSIM3V1ldvt2w = 0.0;
        if (!model->BSIM3V1ldroutGiven)
            model->BSIM3V1ldrout = 0.0;     
        if (!model->BSIM3V1ldsubGiven)
            model->BSIM3V1ldsub = 0.0;
        if (!model->BSIM3V1lvth0Given)
           model->BSIM3V1lvth0 = 0.0;
        if (!model->BSIM3V1luaGiven)
            model->BSIM3V1lua = 0.0;
        if (!model->BSIM3V1lua1Given)
            model->BSIM3V1lua1 = 0.0;
        if (!model->BSIM3V1lubGiven)
            model->BSIM3V1lub = 0.0;
        if (!model->BSIM3V1lub1Given)
            model->BSIM3V1lub1 = 0.0;
        if (!model->BSIM3V1lucGiven)
            model->BSIM3V1luc = 0.0;
        if (!model->BSIM3V1luc1Given)
            model->BSIM3V1luc1 = 0.0;
        if (!model->BSIM3V1lu0Given)
            model->BSIM3V1lu0 = 0.0;
        if (!model->BSIM3V1luteGiven)
	    model->BSIM3V1lute = 0.0;    
        if (!model->BSIM3V1lvoffGiven)
	    model->BSIM3V1lvoff = 0.0;
        if (!model->BSIM3V1ldeltaGiven)  
            model->BSIM3V1ldelta = 0.0;
        if (!model->BSIM3V1lrdswGiven)
            model->BSIM3V1lrdsw = 0.0;
        if (!model->BSIM3V1lprwbGiven)
            model->BSIM3V1lprwb = 0.0;
        if (!model->BSIM3V1lprwgGiven)
            model->BSIM3V1lprwg = 0.0;
        if (!model->BSIM3V1lprtGiven)
        if (!model->BSIM3V1lprtGiven)
            model->BSIM3V1lprt = 0.0;
        if (!model->BSIM3V1leta0Given)
            model->BSIM3V1leta0 = 0.0;
        if (!model->BSIM3V1letabGiven)
            model->BSIM3V1letab = -0.0;
        if (!model->BSIM3V1lpclmGiven)
            model->BSIM3V1lpclm = 0.0; 
        if (!model->BSIM3V1lpdibl1Given)
            model->BSIM3V1lpdibl1 = 0.0;
        if (!model->BSIM3V1lpdibl2Given)
            model->BSIM3V1lpdibl2 = 0.0;
        if (!model->BSIM3V1lpdiblbGiven)
            model->BSIM3V1lpdiblb = 0.0;
        if (!model->BSIM3V1lpscbe1Given)
            model->BSIM3V1lpscbe1 = 0.0;
        if (!model->BSIM3V1lpscbe2Given)
            model->BSIM3V1lpscbe2 = 0.0;
        if (!model->BSIM3V1lpvagGiven)
            model->BSIM3V1lpvag = 0.0;     
        if (!model->BSIM3V1lwrGiven)  
            model->BSIM3V1lwr = 0.0;
        if (!model->BSIM3V1ldwgGiven)  
            model->BSIM3V1ldwg = 0.0;
        if (!model->BSIM3V1ldwbGiven)  
            model->BSIM3V1ldwb = 0.0;
        if (!model->BSIM3V1lb0Given)
            model->BSIM3V1lb0 = 0.0;
        if (!model->BSIM3V1lb1Given)  
            model->BSIM3V1lb1 = 0.0;
        if (!model->BSIM3V1lalpha0Given)  
            model->BSIM3V1lalpha0 = 0.0;
        if (!model->BSIM3V1lbeta0Given)  
            model->BSIM3V1lbeta0 = 0.0;

        if (!model->BSIM3V1lelmGiven)  
            model->BSIM3V1lelm = 0.0;
        if (!model->BSIM3V1lcgslGiven)  
            model->BSIM3V1lcgsl = 0.0;
        if (!model->BSIM3V1lcgdlGiven)  
            model->BSIM3V1lcgdl = 0.0;
        if (!model->BSIM3V1lckappaGiven)  
            model->BSIM3V1lckappa = 0.0;
        if (!model->BSIM3V1lclcGiven)  
            model->BSIM3V1lclc = 0.0;
        if (!model->BSIM3V1lcleGiven)  
            model->BSIM3V1lcle = 0.0;
        if (!model->BSIM3V1lcfGiven)  
            model->BSIM3V1lcf = 0.0;
        if (!model->BSIM3V1lvfbcvGiven)  
            model->BSIM3V1lvfbcv = 0.0;

	/* Width dependence */
        if (!model->BSIM3V1wcdscGiven)
	    model->BSIM3V1wcdsc = 0.0;
        if (!model->BSIM3V1wcdscbGiven)
	    model->BSIM3V1wcdscb = 0.0;  
	    if (!model->BSIM3V1wcdscdGiven)
	    model->BSIM3V1wcdscd = 0.0;
        if (!model->BSIM3V1wcitGiven)
	    model->BSIM3V1wcit = 0.0;
        if (!model->BSIM3V1wnfactorGiven)
	    model->BSIM3V1wnfactor = 0.0;
        if (!model->BSIM3V1wxjGiven)
            model->BSIM3V1wxj = 0.0;
        if (!model->BSIM3V1wvsatGiven)
            model->BSIM3V1wvsat = 0.0;
        if (!model->BSIM3V1watGiven)
            model->BSIM3V1wat = 0.0;
        if (!model->BSIM3V1wa0Given)
            model->BSIM3V1wa0 = 0.0; 
        if (!model->BSIM3V1wagsGiven)
            model->BSIM3V1wags = 0.0;
        if (!model->BSIM3V1wa1Given)
            model->BSIM3V1wa1 = 0.0;
        if (!model->BSIM3V1wa2Given)
            model->BSIM3V1wa2 = 0.0;
        if (!model->BSIM3V1wketaGiven)
            model->BSIM3V1wketa = 0.0;
        if (!model->BSIM3V1wnsubGiven)
            model->BSIM3V1wnsub = 0.0;
        if (!model->BSIM3V1wnpeakGiven)
            model->BSIM3V1wnpeak = 0.0;
        if (!model->BSIM3V1wngateGiven)
            model->BSIM3V1wngate = 0.0;
        if (!model->BSIM3V1wvbmGiven)
	    model->BSIM3V1wvbm = 0.0;
        if (!model->BSIM3V1wxtGiven)
	    model->BSIM3V1wxt = 0.0;
        if (!model->BSIM3V1wkt1Given)
            model->BSIM3V1wkt1 = 0.0; 
        if (!model->BSIM3V1wkt1lGiven)
            model->BSIM3V1wkt1l = 0.0;
        if (!model->BSIM3V1wkt2Given)
            model->BSIM3V1wkt2 = 0.0;
        if (!model->BSIM3V1wk3Given)
            model->BSIM3V1wk3 = 0.0;      
        if (!model->BSIM3V1wk3bGiven)
            model->BSIM3V1wk3b = 0.0;      
        if (!model->BSIM3V1ww0Given)
            model->BSIM3V1ww0 = 0.0;    
        if (!model->BSIM3V1wnlxGiven)
            model->BSIM3V1wnlx = 0.0;     
        if (!model->BSIM3V1wdvt0Given)
            model->BSIM3V1wdvt0 = 0.0;    
        if (!model->BSIM3V1wdvt1Given)
            model->BSIM3V1wdvt1 = 0.0;      
        if (!model->BSIM3V1wdvt2Given)
            model->BSIM3V1wdvt2 = 0.0;
        if (!model->BSIM3V1wdvt0wGiven)
            model->BSIM3V1wdvt0w = 0.0;    
        if (!model->BSIM3V1wdvt1wGiven)
            model->BSIM3V1wdvt1w = 0.0;      
        if (!model->BSIM3V1wdvt2wGiven)
            model->BSIM3V1wdvt2w = 0.0;
        if (!model->BSIM3V1wdroutGiven)
            model->BSIM3V1wdrout = 0.0;     
        if (!model->BSIM3V1wdsubGiven)
            model->BSIM3V1wdsub = 0.0;
        if (!model->BSIM3V1wvth0Given)
           model->BSIM3V1wvth0 = 0.0;
        if (!model->BSIM3V1wuaGiven)
            model->BSIM3V1wua = 0.0;
        if (!model->BSIM3V1wua1Given)
            model->BSIM3V1wua1 = 0.0;
        if (!model->BSIM3V1wubGiven)
            model->BSIM3V1wub = 0.0;
        if (!model->BSIM3V1wub1Given)
            model->BSIM3V1wub1 = 0.0;
        if (!model->BSIM3V1wucGiven)
            model->BSIM3V1wuc = 0.0;
        if (!model->BSIM3V1wuc1Given)
            model->BSIM3V1wuc1 = 0.0;
        if (!model->BSIM3V1wu0Given)
            model->BSIM3V1wu0 = 0.0;
        if (!model->BSIM3V1wuteGiven)
	    model->BSIM3V1wute = 0.0;    
        if (!model->BSIM3V1wvoffGiven)
	    model->BSIM3V1wvoff = 0.0;
        if (!model->BSIM3V1wdeltaGiven)  
            model->BSIM3V1wdelta = 0.0;
        if (!model->BSIM3V1wrdswGiven)
            model->BSIM3V1wrdsw = 0.0;
        if (!model->BSIM3V1wprwbGiven)
            model->BSIM3V1wprwb = 0.0;
        if (!model->BSIM3V1wprwgGiven)
            model->BSIM3V1wprwg = 0.0;
        if (!model->BSIM3V1wprtGiven)
            model->BSIM3V1wprt = 0.0;
        if (!model->BSIM3V1weta0Given)
            model->BSIM3V1weta0 = 0.0;
        if (!model->BSIM3V1wetabGiven)
            model->BSIM3V1wetab = 0.0;
        if (!model->BSIM3V1wpclmGiven)
            model->BSIM3V1wpclm = 0.0; 
        if (!model->BSIM3V1wpdibl1Given)
            model->BSIM3V1wpdibl1 = 0.0;
        if (!model->BSIM3V1wpdibl2Given)
            model->BSIM3V1wpdibl2 = 0.0;
        if (!model->BSIM3V1wpdiblbGiven)
            model->BSIM3V1wpdiblb = 0.0;
        if (!model->BSIM3V1wpscbe1Given)
            model->BSIM3V1wpscbe1 = 0.0;
        if (!model->BSIM3V1wpscbe2Given)
            model->BSIM3V1wpscbe2 = 0.0;
        if (!model->BSIM3V1wpvagGiven)
            model->BSIM3V1wpvag = 0.0;     
        if (!model->BSIM3V1wwrGiven)  
            model->BSIM3V1wwr = 0.0;
        if (!model->BSIM3V1wdwgGiven)  
            model->BSIM3V1wdwg = 0.0;
        if (!model->BSIM3V1wdwbGiven)  
            model->BSIM3V1wdwb = 0.0;
        if (!model->BSIM3V1wb0Given)
            model->BSIM3V1wb0 = 0.0;
        if (!model->BSIM3V1wb1Given)  
            model->BSIM3V1wb1 = 0.0;
        if (!model->BSIM3V1walpha0Given)  
            model->BSIM3V1walpha0 = 0.0;
        if (!model->BSIM3V1wbeta0Given)  
            model->BSIM3V1wbeta0 = 0.0;

        if (!model->BSIM3V1welmGiven)  
            model->BSIM3V1welm = 0.0;
        if (!model->BSIM3V1wcgslGiven)  
            model->BSIM3V1wcgsl = 0.0;
        if (!model->BSIM3V1wcgdlGiven)  
            model->BSIM3V1wcgdl = 0.0;
        if (!model->BSIM3V1wckappaGiven)  
            model->BSIM3V1wckappa = 0.0;
        if (!model->BSIM3V1wcfGiven)  
            model->BSIM3V1wcf = 0.0;
        if (!model->BSIM3V1wclcGiven)  
            model->BSIM3V1wclc = 0.0;
        if (!model->BSIM3V1wcleGiven)  
            model->BSIM3V1wcle = 0.0;
        if (!model->BSIM3V1wvfbcvGiven)  
            model->BSIM3V1wvfbcv = 0.0;

	/* Cross-term dependence */
        if (!model->BSIM3V1pcdscGiven)
	    model->BSIM3V1pcdsc = 0.0;
        if (!model->BSIM3V1pcdscbGiven)
	    model->BSIM3V1pcdscb = 0.0;   
	    if (!model->BSIM3V1pcdscdGiven)
	    model->BSIM3V1pcdscd = 0.0;
        if (!model->BSIM3V1pcitGiven)
	    model->BSIM3V1pcit = 0.0;
        if (!model->BSIM3V1pnfactorGiven)
	    model->BSIM3V1pnfactor = 0.0;
        if (!model->BSIM3V1pxjGiven)
            model->BSIM3V1pxj = 0.0;
        if (!model->BSIM3V1pvsatGiven)
            model->BSIM3V1pvsat = 0.0;
        if (!model->BSIM3V1patGiven)
            model->BSIM3V1pat = 0.0;
        if (!model->BSIM3V1pa0Given)
            model->BSIM3V1pa0 = 0.0; 
            
        if (!model->BSIM3V1pagsGiven)
            model->BSIM3V1pags = 0.0;
        if (!model->BSIM3V1pa1Given)
            model->BSIM3V1pa1 = 0.0;
        if (!model->BSIM3V1pa2Given)
            model->BSIM3V1pa2 = 0.0;
        if (!model->BSIM3V1pketaGiven)
            model->BSIM3V1pketa = 0.0;
        if (!model->BSIM3V1pnsubGiven)
            model->BSIM3V1pnsub = 0.0;
        if (!model->BSIM3V1pnpeakGiven)
            model->BSIM3V1pnpeak = 0.0;
        if (!model->BSIM3V1pngateGiven)
            model->BSIM3V1pngate = 0.0;
        if (!model->BSIM3V1pvbmGiven)
	    model->BSIM3V1pvbm = 0.0;
        if (!model->BSIM3V1pxtGiven)
	    model->BSIM3V1pxt = 0.0;
        if (!model->BSIM3V1pkt1Given)
            model->BSIM3V1pkt1 = 0.0; 
        if (!model->BSIM3V1pkt1lGiven)
            model->BSIM3V1pkt1l = 0.0;
        if (!model->BSIM3V1pkt2Given)
            model->BSIM3V1pkt2 = 0.0;
        if (!model->BSIM3V1pk3Given)
            model->BSIM3V1pk3 = 0.0;      
        if (!model->BSIM3V1pk3bGiven)
            model->BSIM3V1pk3b = 0.0;      
        if (!model->BSIM3V1pw0Given)
            model->BSIM3V1pw0 = 0.0;    
        if (!model->BSIM3V1pnlxGiven)
            model->BSIM3V1pnlx = 0.0;     
        if (!model->BSIM3V1pdvt0Given)
            model->BSIM3V1pdvt0 = 0.0;    
        if (!model->BSIM3V1pdvt1Given)
            model->BSIM3V1pdvt1 = 0.0;      
        if (!model->BSIM3V1pdvt2Given)
            model->BSIM3V1pdvt2 = 0.0;
        if (!model->BSIM3V1pdvt0wGiven)
            model->BSIM3V1pdvt0w = 0.0;    
        if (!model->BSIM3V1pdvt1wGiven)
            model->BSIM3V1pdvt1w = 0.0;      
        if (!model->BSIM3V1pdvt2wGiven)
            model->BSIM3V1pdvt2w = 0.0;
        if (!model->BSIM3V1pdroutGiven)
            model->BSIM3V1pdrout = 0.0;     
        if (!model->BSIM3V1pdsubGiven)
            model->BSIM3V1pdsub = 0.0;
        if (!model->BSIM3V1pvth0Given)
           model->BSIM3V1pvth0 = 0.0;
        if (!model->BSIM3V1puaGiven)
            model->BSIM3V1pua = 0.0;
        if (!model->BSIM3V1pua1Given)
            model->BSIM3V1pua1 = 0.0;
        if (!model->BSIM3V1pubGiven)
            model->BSIM3V1pub = 0.0;
        if (!model->BSIM3V1pub1Given)
            model->BSIM3V1pub1 = 0.0;
        if (!model->BSIM3V1pucGiven)
            model->BSIM3V1puc = 0.0;
        if (!model->BSIM3V1puc1Given)
            model->BSIM3V1puc1 = 0.0;
        if (!model->BSIM3V1pu0Given)
            model->BSIM3V1pu0 = 0.0;
        if (!model->BSIM3V1puteGiven)
	    model->BSIM3V1pute = 0.0;    
        if (!model->BSIM3V1pvoffGiven)
	    model->BSIM3V1pvoff = 0.0;
        if (!model->BSIM3V1pdeltaGiven)  
            model->BSIM3V1pdelta = 0.0;
        if (!model->BSIM3V1prdswGiven)
            model->BSIM3V1prdsw = 0.0;
        if (!model->BSIM3V1pprwbGiven)
            model->BSIM3V1pprwb = 0.0;
        if (!model->BSIM3V1pprwgGiven)
            model->BSIM3V1pprwg = 0.0;
        if (!model->BSIM3V1pprtGiven)
            model->BSIM3V1pprt = 0.0;
        if (!model->BSIM3V1peta0Given)
            model->BSIM3V1peta0 = 0.0;
        if (!model->BSIM3V1petabGiven)
            model->BSIM3V1petab = 0.0;
        if (!model->BSIM3V1ppclmGiven)
            model->BSIM3V1ppclm = 0.0; 
        if (!model->BSIM3V1ppdibl1Given)
            model->BSIM3V1ppdibl1 = 0.0;
        if (!model->BSIM3V1ppdibl2Given)
            model->BSIM3V1ppdibl2 = 0.0;
        if (!model->BSIM3V1ppdiblbGiven)
            model->BSIM3V1ppdiblb = 0.0;
        if (!model->BSIM3V1ppscbe1Given)
            model->BSIM3V1ppscbe1 = 0.0;
        if (!model->BSIM3V1ppscbe2Given)
            model->BSIM3V1ppscbe2 = 0.0;
        if (!model->BSIM3V1ppvagGiven)
            model->BSIM3V1ppvag = 0.0;     
        if (!model->BSIM3V1pwrGiven)  
            model->BSIM3V1pwr = 0.0;
        if (!model->BSIM3V1pdwgGiven)  
            model->BSIM3V1pdwg = 0.0;
        if (!model->BSIM3V1pdwbGiven)  
            model->BSIM3V1pdwb = 0.0;
        if (!model->BSIM3V1pb0Given)
            model->BSIM3V1pb0 = 0.0;
        if (!model->BSIM3V1pb1Given)  
            model->BSIM3V1pb1 = 0.0;
        if (!model->BSIM3V1palpha0Given)  
            model->BSIM3V1palpha0 = 0.0;
        if (!model->BSIM3V1pbeta0Given)  
            model->BSIM3V1pbeta0 = 0.0;

        if (!model->BSIM3V1pelmGiven)  
            model->BSIM3V1pelm = 0.0;
        if (!model->BSIM3V1pcgslGiven)  
            model->BSIM3V1pcgsl = 0.0;
        if (!model->BSIM3V1pcgdlGiven)  
            model->BSIM3V1pcgdl = 0.0;
        if (!model->BSIM3V1pckappaGiven)  
            model->BSIM3V1pckappa = 0.0;
        if (!model->BSIM3V1pcfGiven)  
            model->BSIM3V1pcf = 0.0;
        if (!model->BSIM3V1pclcGiven)  
            model->BSIM3V1pclc = 0.0;
        if (!model->BSIM3V1pcleGiven)  
            model->BSIM3V1pcle = 0.0;
        if (!model->BSIM3V1pvfbcvGiven)  
            model->BSIM3V1pvfbcv = 0.0;

        /* unit degree celcius */
        if (!model->BSIM3V1tnomGiven)  
	    model->BSIM3V1tnom = ckt->CKTnomTemp; 
        if (!model->BSIM3V1LintGiven)  
           model->BSIM3V1Lint = 0.0;
        if (!model->BSIM3V1LlGiven)  
           model->BSIM3V1Ll = 0.0;
        if (!model->BSIM3V1LlnGiven)  
           model->BSIM3V1Lln = 1.0;
        if (!model->BSIM3V1LwGiven)  
           model->BSIM3V1Lw = 0.0;
        if (!model->BSIM3V1LwnGiven)  
           model->BSIM3V1Lwn = 1.0;
        if (!model->BSIM3V1LwlGiven)  
           model->BSIM3V1Lwl = 0.0;
        if (!model->BSIM3V1LminGiven)  
           model->BSIM3V1Lmin = 0.0;
        if (!model->BSIM3V1LmaxGiven)  
           model->BSIM3V1Lmax = 1.0;
        if (!model->BSIM3V1WintGiven)  
           model->BSIM3V1Wint = 0.0;
        if (!model->BSIM3V1WlGiven)  
           model->BSIM3V1Wl = 0.0;
        if (!model->BSIM3V1WlnGiven)  
           model->BSIM3V1Wln = 1.0;
        if (!model->BSIM3V1WwGiven)  
           model->BSIM3V1Ww = 0.0;
        if (!model->BSIM3V1WwnGiven)  
           model->BSIM3V1Wwn = 1.0;
        if (!model->BSIM3V1WwlGiven)  
           model->BSIM3V1Wwl = 0.0;
        if (!model->BSIM3V1WminGiven)  
           model->BSIM3V1Wmin = 0.0;
        if (!model->BSIM3V1WmaxGiven)  
           model->BSIM3V1Wmax = 1.0;
        if (!model->BSIM3V1dwcGiven)  
           model->BSIM3V1dwc = model->BSIM3V1Wint;
        if (!model->BSIM3V1dlcGiven)  
           model->BSIM3V1dlc = model->BSIM3V1Lint;
	if (!model->BSIM3V1cfGiven)
            model->BSIM3V1cf = 2.0 * EPSOX / PI
			   * log(1.0 + 0.4e-6 / model->BSIM3V1tox);
        if (!model->BSIM3V1cgdoGiven)
	{   if (model->BSIM3V1dlcGiven && (model->BSIM3V1dlc > 0.0))
	    {   model->BSIM3V1cgdo = model->BSIM3V1dlc * model->BSIM3V1cox
				 - model->BSIM3V1cgdl ;
	    }
	    else
	        model->BSIM3V1cgdo = 0.6 * model->BSIM3V1xj * model->BSIM3V1cox; 
	}
        if (!model->BSIM3V1cgsoGiven)
	{   if (model->BSIM3V1dlcGiven && (model->BSIM3V1dlc > 0.0))
	    {   model->BSIM3V1cgso = model->BSIM3V1dlc * model->BSIM3V1cox
				 - model->BSIM3V1cgsl ;
	    }
	    else
	        model->BSIM3V1cgso = 0.6 * model->BSIM3V1xj * model->BSIM3V1cox; 
	}

        if (!model->BSIM3V1cgboGiven)
	{   model->BSIM3V1cgbo = 2.0 * model->BSIM3V1dwc * model->BSIM3V1cox;
	}
        if (!model->BSIM3V1xpartGiven)
            model->BSIM3V1xpart = 0.0;
        if (!model->BSIM3V1sheetResistanceGiven)
            model->BSIM3V1sheetResistance = 0.0;
        if (!model->BSIM3V1unitAreaJctCapGiven)
            model->BSIM3V1unitAreaJctCap = 5.0E-4;
        if (!model->BSIM3V1unitLengthSidewallJctCapGiven)
            model->BSIM3V1unitLengthSidewallJctCap = 5.0E-10;
        if (!model->BSIM3V1unitLengthGateSidewallJctCapGiven)
            model->BSIM3V1unitLengthGateSidewallJctCap = model->BSIM3V1unitLengthSidewallJctCap ;
        if (!model->BSIM3V1jctSatCurDensityGiven)
            model->BSIM3V1jctSatCurDensity = 1.0E-4;
        if (!model->BSIM3V1jctSidewallSatCurDensityGiven)
            model->BSIM3V1jctSidewallSatCurDensity = 0.0;
        if (!model->BSIM3V1bulkJctPotentialGiven)
            model->BSIM3V1bulkJctPotential = 1.0;
        if (!model->BSIM3V1sidewallJctPotentialGiven)
            model->BSIM3V1sidewallJctPotential = 1.0;
        if (!model->BSIM3V1GatesidewallJctPotentialGiven)
            model->BSIM3V1GatesidewallJctPotential = model->BSIM3V1sidewallJctPotential;
        if (!model->BSIM3V1bulkJctBotGradingCoeffGiven)
            model->BSIM3V1bulkJctBotGradingCoeff = 0.5;
        if (!model->BSIM3V1bulkJctSideGradingCoeffGiven)
            model->BSIM3V1bulkJctSideGradingCoeff = 0.33;
        if (!model->BSIM3V1bulkJctGateSideGradingCoeffGiven)
            model->BSIM3V1bulkJctGateSideGradingCoeff = model->BSIM3V1bulkJctSideGradingCoeff;
        if (!model->BSIM3V1jctEmissionCoeffGiven)
            model->BSIM3V1jctEmissionCoeff = 1.0;
        if (!model->BSIM3V1jctTempExponentGiven)
            model->BSIM3V1jctTempExponent = 3.0;
        if (!model->BSIM3V1oxideTrapDensityAGiven)
        if (!model->BSIM3V1oxideTrapDensityAGiven)
	{   if (model->BSIM3V1type == NMOS)
                model->BSIM3V1oxideTrapDensityA = 1e20;
            else
                model->BSIM3V1oxideTrapDensityA=9.9e18;
	}
        if (!model->BSIM3V1oxideTrapDensityBGiven)
	{   if (model->BSIM3V1type == NMOS)
                model->BSIM3V1oxideTrapDensityB = 5e4;
            else
                model->BSIM3V1oxideTrapDensityB = 2.4e3;
	}
        if (!model->BSIM3V1oxideTrapDensityCGiven)
	{   if (model->BSIM3V1type == NMOS)
                model->BSIM3V1oxideTrapDensityC = -1.4e-12;
            else
                model->BSIM3V1oxideTrapDensityC = 1.4e-12;

	}
        if (!model->BSIM3V1emGiven)
            model->BSIM3V1em = 4.1e7; /* V/m */
        if (!model->BSIM3V1efGiven)
            model->BSIM3V1ef = 1.0;
        if (!model->BSIM3V1afGiven)
            model->BSIM3V1af = 1.0;
        if (!model->BSIM3V1kfGiven)
            model->BSIM3V1kf = 0.0;
        /* loop through all the instances of the model */
        for (here = model->BSIM3V1instances; here != NULL ;
             here=here->BSIM3V1nextInstance)
        
	{   
           if (here->BSIM3V1owner == ARCHme) { 
           /* allocate a chunk of the state vector */
            here->BSIM3V1states = *states;
            *states += BSIM3V1numStates;
           }
            /* perform the parameter defaulting */
            if(here->BSIM3V1m == 0.0)
              here->BSIM3V1m = 1.0;
            fprintf(stderr, "M = %.2f\n", here->BSIM3V1m);
            if (!here->BSIM3V1wGiven)
	      here->BSIM3V1w = 5e-6;
            here->BSIM3V1w *= here->BSIM3V1m;

            if (!here->BSIM3V1drainAreaGiven)
            {
                if(model->BSIM3V1hdifGiven)
                    here->BSIM3V1drainArea = here->BSIM3V1w * 2 * model->BSIM3V1hdif;
                else
                    here->BSIM3V1drainArea = 0.0;
            }
            here->BSIM3V1drainArea *= here->BSIM3V1m;
            if (!here->BSIM3V1drainPerimeterGiven)
            {
                if(model->BSIM3V1hdifGiven)
                    here->BSIM3V1drainPerimeter =
                        2 * here->BSIM3V1w + 4 * model->BSIM3V1hdif;
                else
                    here->BSIM3V1drainPerimeter = 0.0;
            }
            here->BSIM3V1drainPerimeter *= here->BSIM3V1m;

            if (!here->BSIM3V1drainSquaresGiven)
                here->BSIM3V1drainSquares = 1.0;
            here->BSIM3V1drainSquares /= here->BSIM3V1m;

            if (!here->BSIM3V1icVBSGiven)
                here->BSIM3V1icVBS = 0;
            if (!here->BSIM3V1icVDSGiven)
                here->BSIM3V1icVDS = 0;
            if (!here->BSIM3V1icVGSGiven)
                here->BSIM3V1icVGS = 0;
            if (!here->BSIM3V1lGiven)
                here->BSIM3V1l = 5e-6;
            if (!here->BSIM3V1sourceAreaGiven)
            {
                if(model->BSIM3V1hdifGiven)
                    here->BSIM3V1sourceArea = here->BSIM3V1w * 2 * model->BSIM3V1hdif;
                else
                    here->BSIM3V1sourceArea = 0.0;
            }
            here->BSIM3V1sourceArea *= here->BSIM3V1m;

            if (!here->BSIM3V1sourcePerimeterGiven)
            {
                if(model->BSIM3V1hdifGiven)
                    here->BSIM3V1sourcePerimeter =
                        2 * here->BSIM3V1w + 4 * model->BSIM3V1hdif;
                else
                    here->BSIM3V1sourcePerimeter = 0.0;
            }
            here->BSIM3V1sourcePerimeter *= here->BSIM3V1m;

            if (!here->BSIM3V1sourceSquaresGiven)
                here->BSIM3V1sourceSquares = 1.0;
            here->BSIM3V1sourceSquares /= here->BSIM3V1m;

            if (!here->BSIM3V1nqsModGiven)
                here->BSIM3V1nqsMod = model->BSIM3V1nqsMod;
                    
            /* process drain series resistance */
            if ((model->BSIM3V1sheetResistance > 0.0) && 
                (here->BSIM3V1drainSquares > 0.0 ) &&
                (here->BSIM3V1dNodePrime == 0))
	    {   error = CKTmkVolt(ckt,&tmp,here->BSIM3V1name,"drain");
                if(error) return(error);
                here->BSIM3V1dNodePrime = tmp->number;
            }
	    else
	    {   here->BSIM3V1dNodePrime = here->BSIM3V1dNode;
            }
                   
            /* process source series resistance */
            if ((model->BSIM3V1sheetResistance > 0.0) && 
                (here->BSIM3V1sourceSquares > 0.0 ) &&
                (here->BSIM3V1sNodePrime == 0)) 
	    {   error = CKTmkVolt(ckt,&tmp,here->BSIM3V1name,"source");
                if(error) return(error);
                here->BSIM3V1sNodePrime = tmp->number;
            }
	    else 
	    {   here->BSIM3V1sNodePrime = here->BSIM3V1sNode;
            }

 /* internal charge node */
                   
            if ((here->BSIM3V1nqsMod) && (here->BSIM3V1qNode == 0)) 
	    {   error = CKTmkVolt(ckt,&tmp,here->BSIM3V1name,"charge");
                if(error) return(error);
                here->BSIM3V1qNode = tmp->number;
            }
	    else 
	    {   here->BSIM3V1qNode = 0;
            }

        /* set Sparse Matrix Pointers */

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==(double *)NULL){\
    return(E_NOMEM);\
}

            TSTALLOC(BSIM3V1DdPtr, BSIM3V1dNode, BSIM3V1dNode)
            TSTALLOC(BSIM3V1GgPtr, BSIM3V1gNode, BSIM3V1gNode)
            TSTALLOC(BSIM3V1SsPtr, BSIM3V1sNode, BSIM3V1sNode)
            TSTALLOC(BSIM3V1BbPtr, BSIM3V1bNode, BSIM3V1bNode)
            TSTALLOC(BSIM3V1DPdpPtr, BSIM3V1dNodePrime, BSIM3V1dNodePrime)
            TSTALLOC(BSIM3V1SPspPtr, BSIM3V1sNodePrime, BSIM3V1sNodePrime)
            TSTALLOC(BSIM3V1DdpPtr, BSIM3V1dNode, BSIM3V1dNodePrime)
            TSTALLOC(BSIM3V1GbPtr, BSIM3V1gNode, BSIM3V1bNode)
            TSTALLOC(BSIM3V1GdpPtr, BSIM3V1gNode, BSIM3V1dNodePrime)
            TSTALLOC(BSIM3V1GspPtr, BSIM3V1gNode, BSIM3V1sNodePrime)
            TSTALLOC(BSIM3V1SspPtr, BSIM3V1sNode, BSIM3V1sNodePrime)
            TSTALLOC(BSIM3V1BdpPtr, BSIM3V1bNode, BSIM3V1dNodePrime)
            TSTALLOC(BSIM3V1BspPtr, BSIM3V1bNode, BSIM3V1sNodePrime)
            TSTALLOC(BSIM3V1DPspPtr, BSIM3V1dNodePrime, BSIM3V1sNodePrime)
            TSTALLOC(BSIM3V1DPdPtr, BSIM3V1dNodePrime, BSIM3V1dNode)
            TSTALLOC(BSIM3V1BgPtr, BSIM3V1bNode, BSIM3V1gNode)
            TSTALLOC(BSIM3V1DPgPtr, BSIM3V1dNodePrime, BSIM3V1gNode)
            TSTALLOC(BSIM3V1SPgPtr, BSIM3V1sNodePrime, BSIM3V1gNode)
            TSTALLOC(BSIM3V1SPsPtr, BSIM3V1sNodePrime, BSIM3V1sNode)
            TSTALLOC(BSIM3V1DPbPtr, BSIM3V1dNodePrime, BSIM3V1bNode)
            TSTALLOC(BSIM3V1SPbPtr, BSIM3V1sNodePrime, BSIM3V1bNode)
            TSTALLOC(BSIM3V1SPdpPtr, BSIM3V1sNodePrime, BSIM3V1dNodePrime)

            TSTALLOC(BSIM3V1QqPtr, BSIM3V1qNode, BSIM3V1qNode)
 
            TSTALLOC(BSIM3V1QdpPtr, BSIM3V1qNode, BSIM3V1dNodePrime)
            TSTALLOC(BSIM3V1QspPtr, BSIM3V1qNode, BSIM3V1sNodePrime)
            TSTALLOC(BSIM3V1QgPtr, BSIM3V1qNode, BSIM3V1gNode)
            TSTALLOC(BSIM3V1QbPtr, BSIM3V1qNode, BSIM3V1bNode)
            TSTALLOC(BSIM3V1DPqPtr, BSIM3V1dNodePrime, BSIM3V1qNode)
            TSTALLOC(BSIM3V1SPqPtr, BSIM3V1sNodePrime, BSIM3V1qNode)
            TSTALLOC(BSIM3V1GqPtr, BSIM3V1gNode, BSIM3V1qNode)
            TSTALLOC(BSIM3V1BqPtr, BSIM3V1bNode, BSIM3V1qNode)

        }
    }
    return(OK);
}  





