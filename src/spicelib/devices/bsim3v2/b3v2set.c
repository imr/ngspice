/* $Id$  */
/*
 $Log$
 Revision 1.1.1.1  2000-04-27 20:03:59  pnenzi
 Imported sources

 * Revision 3.2 1998/6/16  18:00:00  Weidong 
 * BSIM3v3.2 release
 *
*/
static char rcsid[] = "$Id$";

/*************************************/

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Weidong Liu (1997-1998).
File: b3v2set.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "smpdefs.h"
#include "cktdefs.h"
#include "bsim3v2def.h"
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
BSIM3V2setup(matrix,inModel,ckt,states)
register SMPmatrix *matrix;
register GENmodel *inModel;
register CKTcircuit *ckt;
int *states;
{
register BSIM3V2model *model = (BSIM3V2model*)inModel;
register BSIM3V2instance *here;
int error;
CKTnode *tmp;

double tmp1, tmp2;

    /*  loop through all the BSIM3V2 device models */
    for( ; model != NULL; model = model->BSIM3V2nextModel )
    {
/* Default value Processing for BSIM3V2 MOSFET Models */
        if (!model->BSIM3V2typeGiven)
            model->BSIM3V2type = NMOS;     
        if (!model->BSIM3V2mobModGiven) 
            model->BSIM3V2mobMod = 1;
        if (!model->BSIM3V2binUnitGiven) 
            model->BSIM3V2binUnit = 1;
        if (!model->BSIM3V2paramChkGiven) 
            model->BSIM3V2paramChk = 0;
        if (!model->BSIM3V2capModGiven) 
            model->BSIM3V2capMod = 3;
        if (!model->BSIM3V2noiModGiven) 
            model->BSIM3V2noiMod = 1;
        if (!model->BSIM3V2versionGiven) 
            model->BSIM3V2version = 3.2;
        if (!model->BSIM3V2toxGiven)
            model->BSIM3V2tox = 150.0e-10;
        model->BSIM3V2cox = 3.453133e-11 / model->BSIM3V2tox;
        if (!model->BSIM3V2toxmGiven)
            model->BSIM3V2toxm = model->BSIM3V2tox;

        if (!model->BSIM3V2cdscGiven)
	    model->BSIM3V2cdsc = 2.4e-4;   /* unit Q/V/m^2  */
        if (!model->BSIM3V2cdscbGiven)
	    model->BSIM3V2cdscb = 0.0;   /* unit Q/V/m^2  */    
	    if (!model->BSIM3V2cdscdGiven)
	    model->BSIM3V2cdscd = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM3V2citGiven)
	    model->BSIM3V2cit = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM3V2nfactorGiven)
	    model->BSIM3V2nfactor = 1;
        if (!model->BSIM3V2xjGiven)
            model->BSIM3V2xj = .15e-6;
        if (!model->BSIM3V2vsatGiven)
            model->BSIM3V2vsat = 8.0e4;    /* unit m/s */ 
        if (!model->BSIM3V2atGiven)
            model->BSIM3V2at = 3.3e4;    /* unit m/s */ 
        if (!model->BSIM3V2a0Given)
            model->BSIM3V2a0 = 1.0;  
        if (!model->BSIM3V2agsGiven)
            model->BSIM3V2ags = 0.0;
        if (!model->BSIM3V2a1Given)
            model->BSIM3V2a1 = 0.0;
        if (!model->BSIM3V2a2Given)
            model->BSIM3V2a2 = 1.0;
        if (!model->BSIM3V2ketaGiven)
            model->BSIM3V2keta = -0.047;    /* unit  / V */
        if (!model->BSIM3V2nsubGiven)
            model->BSIM3V2nsub = 6.0e16;   /* unit 1/cm3 */
        if (!model->BSIM3V2npeakGiven)
            model->BSIM3V2npeak = 1.7e17;   /* unit 1/cm3 */
        if (!model->BSIM3V2ngateGiven)
            model->BSIM3V2ngate = 0;   /* unit 1/cm3 */
        if (!model->BSIM3V2vbmGiven)
	    model->BSIM3V2vbm = -3.0;
        if (!model->BSIM3V2xtGiven)
	    model->BSIM3V2xt = 1.55e-7;
        if (!model->BSIM3V2kt1Given)
            model->BSIM3V2kt1 = -0.11;      /* unit V */
        if (!model->BSIM3V2kt1lGiven)
            model->BSIM3V2kt1l = 0.0;      /* unit V*m */
        if (!model->BSIM3V2kt2Given)
            model->BSIM3V2kt2 = 0.022;      /* No unit */
        if (!model->BSIM3V2k3Given)
            model->BSIM3V2k3 = 80.0;      
        if (!model->BSIM3V2k3bGiven)
            model->BSIM3V2k3b = 0.0;      
        if (!model->BSIM3V2w0Given)
            model->BSIM3V2w0 = 2.5e-6;    
        if (!model->BSIM3V2nlxGiven)
            model->BSIM3V2nlx = 1.74e-7;     
        if (!model->BSIM3V2dvt0Given)
            model->BSIM3V2dvt0 = 2.2;    
        if (!model->BSIM3V2dvt1Given)
            model->BSIM3V2dvt1 = 0.53;      
        if (!model->BSIM3V2dvt2Given)
            model->BSIM3V2dvt2 = -0.032;   /* unit 1 / V */     

        if (!model->BSIM3V2dvt0wGiven)
            model->BSIM3V2dvt0w = 0.0;    
        if (!model->BSIM3V2dvt1wGiven)
            model->BSIM3V2dvt1w = 5.3e6;    
        if (!model->BSIM3V2dvt2wGiven)
            model->BSIM3V2dvt2w = -0.032;   

        if (!model->BSIM3V2droutGiven)
            model->BSIM3V2drout = 0.56;     
        if (!model->BSIM3V2dsubGiven)
            model->BSIM3V2dsub = model->BSIM3V2drout;     
        if (!model->BSIM3V2vth0Given)
            model->BSIM3V2vth0 = (model->BSIM3V2type == NMOS) ? 0.7 : -0.7;
        if (!model->BSIM3V2uaGiven)
            model->BSIM3V2ua = 2.25e-9;      /* unit m/V */
        if (!model->BSIM3V2ua1Given)
            model->BSIM3V2ua1 = 4.31e-9;      /* unit m/V */
        if (!model->BSIM3V2ubGiven)
            model->BSIM3V2ub = 5.87e-19;     /* unit (m/V)**2 */
        if (!model->BSIM3V2ub1Given)
            model->BSIM3V2ub1 = -7.61e-18;     /* unit (m/V)**2 */
        if (!model->BSIM3V2ucGiven)
            model->BSIM3V2uc = (model->BSIM3V2mobMod == 3) ? -0.0465 : -0.0465e-9;   
        if (!model->BSIM3V2uc1Given)
            model->BSIM3V2uc1 = (model->BSIM3V2mobMod == 3) ? -0.056 : -0.056e-9;   
        if (!model->BSIM3V2u0Given)
            model->BSIM3V2u0 = (model->BSIM3V2type == NMOS) ? 0.067 : 0.025;
        if (!model->BSIM3V2uteGiven)
	    model->BSIM3V2ute = -1.5;    
        if (!model->BSIM3V2voffGiven)
	    model->BSIM3V2voff = -0.08;
        if (!model->BSIM3V2deltaGiven)  
           model->BSIM3V2delta = 0.01;
        if (!model->BSIM3V2rdswGiven)
            model->BSIM3V2rdsw = 0;      
        if (!model->BSIM3V2prwgGiven)
            model->BSIM3V2prwg = 0.0;      /* unit 1/V */
        if (!model->BSIM3V2prwbGiven)
            model->BSIM3V2prwb = 0.0;      
        if (!model->BSIM3V2prtGiven)
        if (!model->BSIM3V2prtGiven)
            model->BSIM3V2prt = 0.0;      
        if (!model->BSIM3V2eta0Given)
            model->BSIM3V2eta0 = 0.08;      /* no unit  */ 
        if (!model->BSIM3V2etabGiven)
            model->BSIM3V2etab = -0.07;      /* unit  1/V */ 
        if (!model->BSIM3V2pclmGiven)
            model->BSIM3V2pclm = 1.3;      /* no unit  */ 
        if (!model->BSIM3V2pdibl1Given)
            model->BSIM3V2pdibl1 = .39;    /* no unit  */
        if (!model->BSIM3V2pdibl2Given)
            model->BSIM3V2pdibl2 = 0.0086;    /* no unit  */ 
        if (!model->BSIM3V2pdiblbGiven)
            model->BSIM3V2pdiblb = 0.0;    /* 1/V  */ 
        if (!model->BSIM3V2pscbe1Given)
            model->BSIM3V2pscbe1 = 4.24e8;     
        if (!model->BSIM3V2pscbe2Given)
            model->BSIM3V2pscbe2 = 1.0e-5;    
        if (!model->BSIM3V2pvagGiven)
            model->BSIM3V2pvag = 0.0;     
        if (!model->BSIM3V2wrGiven)  
            model->BSIM3V2wr = 1.0;
        if (!model->BSIM3V2dwgGiven)  
            model->BSIM3V2dwg = 0.0;
        if (!model->BSIM3V2dwbGiven)  
            model->BSIM3V2dwb = 0.0;
        if (!model->BSIM3V2b0Given)
            model->BSIM3V2b0 = 0.0;
        if (!model->BSIM3V2b1Given)  
            model->BSIM3V2b1 = 0.0;
        if (!model->BSIM3V2alpha0Given)  
            model->BSIM3V2alpha0 = 0.0;
        if (!model->BSIM3V2alpha1Given)
            model->BSIM3V2alpha1 = 0.0;
        if (!model->BSIM3V2beta0Given)  
            model->BSIM3V2beta0 = 30.0;
        if (!model->BSIM3V2ijthGiven)
            model->BSIM3V2ijth = 0.1; /* unit A */

        if (!model->BSIM3V2elmGiven)  
            model->BSIM3V2elm = 5.0;
        if (!model->BSIM3V2cgslGiven)  
            model->BSIM3V2cgsl = 0.0;
        if (!model->BSIM3V2cgdlGiven)  
            model->BSIM3V2cgdl = 0.0;
        if (!model->BSIM3V2ckappaGiven)  
            model->BSIM3V2ckappa = 0.6;
        if (!model->BSIM3V2clcGiven)  
            model->BSIM3V2clc = 0.1e-6;
        if (!model->BSIM3V2cleGiven)  
            model->BSIM3V2cle = 0.6;
        if (!model->BSIM3V2vfbcvGiven)  
            model->BSIM3V2vfbcv = -1.0;
        if (!model->BSIM3V2acdeGiven)
            model->BSIM3V2acde = 1.0;
        if (!model->BSIM3V2moinGiven)
            model->BSIM3V2moin = 15.0;
        if (!model->BSIM3V2noffGiven)
            model->BSIM3V2noff = 1.0;
        if (!model->BSIM3V2voffcvGiven)
            model->BSIM3V2voffcv = 0.0;
        if (!model->BSIM3V2tcjGiven)
            model->BSIM3V2tcj = 0.0;
        if (!model->BSIM3V2tpbGiven)
            model->BSIM3V2tpb = 0.0;
        if (!model->BSIM3V2tcjswGiven)
            model->BSIM3V2tcjsw = 0.0;
        if (!model->BSIM3V2tpbswGiven)
            model->BSIM3V2tpbsw = 0.0;
        if (!model->BSIM3V2tcjswgGiven)
            model->BSIM3V2tcjswg = 0.0;
        if (!model->BSIM3V2tpbswgGiven)
            model->BSIM3V2tpbswg = 0.0;

	/* Length dependence */
        if (!model->BSIM3V2lcdscGiven)
	    model->BSIM3V2lcdsc = 0.0;
        if (!model->BSIM3V2lcdscbGiven)
	    model->BSIM3V2lcdscb = 0.0;
	    if (!model->BSIM3V2lcdscdGiven) 
	    model->BSIM3V2lcdscd = 0.0;
        if (!model->BSIM3V2lcitGiven)
	    model->BSIM3V2lcit = 0.0;
        if (!model->BSIM3V2lnfactorGiven)
	    model->BSIM3V2lnfactor = 0.0;
        if (!model->BSIM3V2lxjGiven)
            model->BSIM3V2lxj = 0.0;
        if (!model->BSIM3V2lvsatGiven)
            model->BSIM3V2lvsat = 0.0;
        if (!model->BSIM3V2latGiven)
            model->BSIM3V2lat = 0.0;
        if (!model->BSIM3V2la0Given)
            model->BSIM3V2la0 = 0.0; 
        if (!model->BSIM3V2lagsGiven)
            model->BSIM3V2lags = 0.0;
        if (!model->BSIM3V2la1Given)
            model->BSIM3V2la1 = 0.0;
        if (!model->BSIM3V2la2Given)
            model->BSIM3V2la2 = 0.0;
        if (!model->BSIM3V2lketaGiven)
            model->BSIM3V2lketa = 0.0;
        if (!model->BSIM3V2lnsubGiven)
            model->BSIM3V2lnsub = 0.0;
        if (!model->BSIM3V2lnpeakGiven)
            model->BSIM3V2lnpeak = 0.0;
        if (!model->BSIM3V2lngateGiven)
            model->BSIM3V2lngate = 0.0;
        if (!model->BSIM3V2lvbmGiven)
	    model->BSIM3V2lvbm = 0.0;
        if (!model->BSIM3V2lxtGiven)
	    model->BSIM3V2lxt = 0.0;
        if (!model->BSIM3V2lkt1Given)
            model->BSIM3V2lkt1 = 0.0; 
        if (!model->BSIM3V2lkt1lGiven)
            model->BSIM3V2lkt1l = 0.0;
        if (!model->BSIM3V2lkt2Given)
            model->BSIM3V2lkt2 = 0.0;
        if (!model->BSIM3V2lk3Given)
            model->BSIM3V2lk3 = 0.0;      
        if (!model->BSIM3V2lk3bGiven)
            model->BSIM3V2lk3b = 0.0;      
        if (!model->BSIM3V2lw0Given)
            model->BSIM3V2lw0 = 0.0;    
        if (!model->BSIM3V2lnlxGiven)
            model->BSIM3V2lnlx = 0.0;     
        if (!model->BSIM3V2ldvt0Given)
            model->BSIM3V2ldvt0 = 0.0;    
        if (!model->BSIM3V2ldvt1Given)
            model->BSIM3V2ldvt1 = 0.0;      
        if (!model->BSIM3V2ldvt2Given)
            model->BSIM3V2ldvt2 = 0.0;
        if (!model->BSIM3V2ldvt0wGiven)
            model->BSIM3V2ldvt0w = 0.0;    
        if (!model->BSIM3V2ldvt1wGiven)
            model->BSIM3V2ldvt1w = 0.0;      
        if (!model->BSIM3V2ldvt2wGiven)
            model->BSIM3V2ldvt2w = 0.0;
        if (!model->BSIM3V2ldroutGiven)
            model->BSIM3V2ldrout = 0.0;     
        if (!model->BSIM3V2ldsubGiven)
            model->BSIM3V2ldsub = 0.0;
        if (!model->BSIM3V2lvth0Given)
           model->BSIM3V2lvth0 = 0.0;
        if (!model->BSIM3V2luaGiven)
            model->BSIM3V2lua = 0.0;
        if (!model->BSIM3V2lua1Given)
            model->BSIM3V2lua1 = 0.0;
        if (!model->BSIM3V2lubGiven)
            model->BSIM3V2lub = 0.0;
        if (!model->BSIM3V2lub1Given)
            model->BSIM3V2lub1 = 0.0;
        if (!model->BSIM3V2lucGiven)
            model->BSIM3V2luc = 0.0;
        if (!model->BSIM3V2luc1Given)
            model->BSIM3V2luc1 = 0.0;
        if (!model->BSIM3V2lu0Given)
            model->BSIM3V2lu0 = 0.0;
        if (!model->BSIM3V2luteGiven)
	    model->BSIM3V2lute = 0.0;    
        if (!model->BSIM3V2lvoffGiven)
	    model->BSIM3V2lvoff = 0.0;
        if (!model->BSIM3V2ldeltaGiven)  
            model->BSIM3V2ldelta = 0.0;
        if (!model->BSIM3V2lrdswGiven)
            model->BSIM3V2lrdsw = 0.0;
        if (!model->BSIM3V2lprwbGiven)
            model->BSIM3V2lprwb = 0.0;
        if (!model->BSIM3V2lprwgGiven)
            model->BSIM3V2lprwg = 0.0;
        if (!model->BSIM3V2lprtGiven)
            model->BSIM3V2lprt = 0.0;
        if (!model->BSIM3V2leta0Given)
            model->BSIM3V2leta0 = 0.0;
        if (!model->BSIM3V2letabGiven)
            model->BSIM3V2letab = -0.0;
        if (!model->BSIM3V2lpclmGiven)
            model->BSIM3V2lpclm = 0.0; 
        if (!model->BSIM3V2lpdibl1Given)
            model->BSIM3V2lpdibl1 = 0.0;
        if (!model->BSIM3V2lpdibl2Given)
            model->BSIM3V2lpdibl2 = 0.0;
        if (!model->BSIM3V2lpdiblbGiven)
            model->BSIM3V2lpdiblb = 0.0;
        if (!model->BSIM3V2lpscbe1Given)
            model->BSIM3V2lpscbe1 = 0.0;
        if (!model->BSIM3V2lpscbe2Given)
            model->BSIM3V2lpscbe2 = 0.0;
        if (!model->BSIM3V2lpvagGiven)
            model->BSIM3V2lpvag = 0.0;     
        if (!model->BSIM3V2lwrGiven)  
            model->BSIM3V2lwr = 0.0;
        if (!model->BSIM3V2ldwgGiven)  
            model->BSIM3V2ldwg = 0.0;
        if (!model->BSIM3V2ldwbGiven)  
            model->BSIM3V2ldwb = 0.0;
        if (!model->BSIM3V2lb0Given)
            model->BSIM3V2lb0 = 0.0;
        if (!model->BSIM3V2lb1Given)  
            model->BSIM3V2lb1 = 0.0;
        if (!model->BSIM3V2lalpha0Given)  
            model->BSIM3V2lalpha0 = 0.0;
        if (!model->BSIM3V2lalpha1Given)
            model->BSIM3V2lalpha1 = 0.0;
        if (!model->BSIM3V2lbeta0Given)  
            model->BSIM3V2lbeta0 = 0.0;
        if (!model->BSIM3V2lvfbGiven)
            model->BSIM3V2lvfb = 0.0;

        if (!model->BSIM3V2lelmGiven)  
            model->BSIM3V2lelm = 0.0;
        if (!model->BSIM3V2lcgslGiven)  
            model->BSIM3V2lcgsl = 0.0;
        if (!model->BSIM3V2lcgdlGiven)  
            model->BSIM3V2lcgdl = 0.0;
        if (!model->BSIM3V2lckappaGiven)  
            model->BSIM3V2lckappa = 0.0;
        if (!model->BSIM3V2lclcGiven)  
            model->BSIM3V2lclc = 0.0;
        if (!model->BSIM3V2lcleGiven)  
            model->BSIM3V2lcle = 0.0;
        if (!model->BSIM3V2lcfGiven)  
            model->BSIM3V2lcf = 0.0;
        if (!model->BSIM3V2lvfbcvGiven)  
            model->BSIM3V2lvfbcv = 0.0;
        if (!model->BSIM3V2lacdeGiven)
            model->BSIM3V2lacde = 0.0;
        if (!model->BSIM3V2lmoinGiven)
            model->BSIM3V2lmoin = 0.0;
        if (!model->BSIM3V2lnoffGiven)
            model->BSIM3V2lnoff = 0.0;
        if (!model->BSIM3V2lvoffcvGiven)
            model->BSIM3V2lvoffcv = 0.0;

	/* Width dependence */
        if (!model->BSIM3V2wcdscGiven)
	    model->BSIM3V2wcdsc = 0.0;
        if (!model->BSIM3V2wcdscbGiven)
	    model->BSIM3V2wcdscb = 0.0;  
	    if (!model->BSIM3V2wcdscdGiven)
	    model->BSIM3V2wcdscd = 0.0;
        if (!model->BSIM3V2wcitGiven)
	    model->BSIM3V2wcit = 0.0;
        if (!model->BSIM3V2wnfactorGiven)
	    model->BSIM3V2wnfactor = 0.0;
        if (!model->BSIM3V2wxjGiven)
            model->BSIM3V2wxj = 0.0;
        if (!model->BSIM3V2wvsatGiven)
            model->BSIM3V2wvsat = 0.0;
        if (!model->BSIM3V2watGiven)
            model->BSIM3V2wat = 0.0;
        if (!model->BSIM3V2wa0Given)
            model->BSIM3V2wa0 = 0.0; 
        if (!model->BSIM3V2wagsGiven)
            model->BSIM3V2wags = 0.0;
        if (!model->BSIM3V2wa1Given)
            model->BSIM3V2wa1 = 0.0;
        if (!model->BSIM3V2wa2Given)
            model->BSIM3V2wa2 = 0.0;
        if (!model->BSIM3V2wketaGiven)
            model->BSIM3V2wketa = 0.0;
        if (!model->BSIM3V2wnsubGiven)
            model->BSIM3V2wnsub = 0.0;
        if (!model->BSIM3V2wnpeakGiven)
            model->BSIM3V2wnpeak = 0.0;
        if (!model->BSIM3V2wngateGiven)
            model->BSIM3V2wngate = 0.0;
        if (!model->BSIM3V2wvbmGiven)
	    model->BSIM3V2wvbm = 0.0;
        if (!model->BSIM3V2wxtGiven)
	    model->BSIM3V2wxt = 0.0;
        if (!model->BSIM3V2wkt1Given)
            model->BSIM3V2wkt1 = 0.0; 
        if (!model->BSIM3V2wkt1lGiven)
            model->BSIM3V2wkt1l = 0.0;
        if (!model->BSIM3V2wkt2Given)
            model->BSIM3V2wkt2 = 0.0;
        if (!model->BSIM3V2wk3Given)
            model->BSIM3V2wk3 = 0.0;      
        if (!model->BSIM3V2wk3bGiven)
            model->BSIM3V2wk3b = 0.0;      
        if (!model->BSIM3V2ww0Given)
            model->BSIM3V2ww0 = 0.0;    
        if (!model->BSIM3V2wnlxGiven)
            model->BSIM3V2wnlx = 0.0;     
        if (!model->BSIM3V2wdvt0Given)
            model->BSIM3V2wdvt0 = 0.0;    
        if (!model->BSIM3V2wdvt1Given)
            model->BSIM3V2wdvt1 = 0.0;      
        if (!model->BSIM3V2wdvt2Given)
            model->BSIM3V2wdvt2 = 0.0;
        if (!model->BSIM3V2wdvt0wGiven)
            model->BSIM3V2wdvt0w = 0.0;    
        if (!model->BSIM3V2wdvt1wGiven)
            model->BSIM3V2wdvt1w = 0.0;      
        if (!model->BSIM3V2wdvt2wGiven)
            model->BSIM3V2wdvt2w = 0.0;
        if (!model->BSIM3V2wdroutGiven)
            model->BSIM3V2wdrout = 0.0;     
        if (!model->BSIM3V2wdsubGiven)
            model->BSIM3V2wdsub = 0.0;
        if (!model->BSIM3V2wvth0Given)
           model->BSIM3V2wvth0 = 0.0;
        if (!model->BSIM3V2wuaGiven)
            model->BSIM3V2wua = 0.0;
        if (!model->BSIM3V2wua1Given)
            model->BSIM3V2wua1 = 0.0;
        if (!model->BSIM3V2wubGiven)
            model->BSIM3V2wub = 0.0;
        if (!model->BSIM3V2wub1Given)
            model->BSIM3V2wub1 = 0.0;
        if (!model->BSIM3V2wucGiven)
            model->BSIM3V2wuc = 0.0;
        if (!model->BSIM3V2wuc1Given)
            model->BSIM3V2wuc1 = 0.0;
        if (!model->BSIM3V2wu0Given)
            model->BSIM3V2wu0 = 0.0;
        if (!model->BSIM3V2wuteGiven)
	    model->BSIM3V2wute = 0.0;    
        if (!model->BSIM3V2wvoffGiven)
	    model->BSIM3V2wvoff = 0.0;
        if (!model->BSIM3V2wdeltaGiven)  
            model->BSIM3V2wdelta = 0.0;
        if (!model->BSIM3V2wrdswGiven)
            model->BSIM3V2wrdsw = 0.0;
        if (!model->BSIM3V2wprwbGiven)
            model->BSIM3V2wprwb = 0.0;
        if (!model->BSIM3V2wprwgGiven)
            model->BSIM3V2wprwg = 0.0;
        if (!model->BSIM3V2wprtGiven)
            model->BSIM3V2wprt = 0.0;
        if (!model->BSIM3V2weta0Given)
            model->BSIM3V2weta0 = 0.0;
        if (!model->BSIM3V2wetabGiven)
            model->BSIM3V2wetab = 0.0;
        if (!model->BSIM3V2wpclmGiven)
            model->BSIM3V2wpclm = 0.0; 
        if (!model->BSIM3V2wpdibl1Given)
            model->BSIM3V2wpdibl1 = 0.0;
        if (!model->BSIM3V2wpdibl2Given)
            model->BSIM3V2wpdibl2 = 0.0;
        if (!model->BSIM3V2wpdiblbGiven)
            model->BSIM3V2wpdiblb = 0.0;
        if (!model->BSIM3V2wpscbe1Given)
            model->BSIM3V2wpscbe1 = 0.0;
        if (!model->BSIM3V2wpscbe2Given)
            model->BSIM3V2wpscbe2 = 0.0;
        if (!model->BSIM3V2wpvagGiven)
            model->BSIM3V2wpvag = 0.0;     
        if (!model->BSIM3V2wwrGiven)  
            model->BSIM3V2wwr = 0.0;
        if (!model->BSIM3V2wdwgGiven)  
            model->BSIM3V2wdwg = 0.0;
        if (!model->BSIM3V2wdwbGiven)  
            model->BSIM3V2wdwb = 0.0;
        if (!model->BSIM3V2wb0Given)
            model->BSIM3V2wb0 = 0.0;
        if (!model->BSIM3V2wb1Given)  
            model->BSIM3V2wb1 = 0.0;
        if (!model->BSIM3V2walpha0Given)  
            model->BSIM3V2walpha0 = 0.0;
        if (!model->BSIM3V2walpha1Given)
            model->BSIM3V2walpha1 = 0.0;
        if (!model->BSIM3V2wbeta0Given)  
            model->BSIM3V2wbeta0 = 0.0;
        if (!model->BSIM3V2wvfbGiven)
            model->BSIM3V2wvfb = 0.0;

        if (!model->BSIM3V2welmGiven)  
            model->BSIM3V2welm = 0.0;
        if (!model->BSIM3V2wcgslGiven)  
            model->BSIM3V2wcgsl = 0.0;
        if (!model->BSIM3V2wcgdlGiven)  
            model->BSIM3V2wcgdl = 0.0;
        if (!model->BSIM3V2wckappaGiven)  
            model->BSIM3V2wckappa = 0.0;
        if (!model->BSIM3V2wcfGiven)  
            model->BSIM3V2wcf = 0.0;
        if (!model->BSIM3V2wclcGiven)  
            model->BSIM3V2wclc = 0.0;
        if (!model->BSIM3V2wcleGiven)  
            model->BSIM3V2wcle = 0.0;
        if (!model->BSIM3V2wvfbcvGiven)  
            model->BSIM3V2wvfbcv = 0.0;
        if (!model->BSIM3V2wacdeGiven)
            model->BSIM3V2wacde = 0.0;
        if (!model->BSIM3V2wmoinGiven)
            model->BSIM3V2wmoin = 0.0;
        if (!model->BSIM3V2wnoffGiven)
            model->BSIM3V2wnoff = 0.0;
        if (!model->BSIM3V2wvoffcvGiven)
            model->BSIM3V2wvoffcv = 0.0;

	/* Cross-term dependence */
        if (!model->BSIM3V2pcdscGiven)
	    model->BSIM3V2pcdsc = 0.0;
        if (!model->BSIM3V2pcdscbGiven)
	    model->BSIM3V2pcdscb = 0.0;   
	    if (!model->BSIM3V2pcdscdGiven)
	    model->BSIM3V2pcdscd = 0.0;
        if (!model->BSIM3V2pcitGiven)
	    model->BSIM3V2pcit = 0.0;
        if (!model->BSIM3V2pnfactorGiven)
	    model->BSIM3V2pnfactor = 0.0;
        if (!model->BSIM3V2pxjGiven)
            model->BSIM3V2pxj = 0.0;
        if (!model->BSIM3V2pvsatGiven)
            model->BSIM3V2pvsat = 0.0;
        if (!model->BSIM3V2patGiven)
            model->BSIM3V2pat = 0.0;
        if (!model->BSIM3V2pa0Given)
            model->BSIM3V2pa0 = 0.0; 
            
        if (!model->BSIM3V2pagsGiven)
            model->BSIM3V2pags = 0.0;
        if (!model->BSIM3V2pa1Given)
            model->BSIM3V2pa1 = 0.0;
        if (!model->BSIM3V2pa2Given)
            model->BSIM3V2pa2 = 0.0;
        if (!model->BSIM3V2pketaGiven)
            model->BSIM3V2pketa = 0.0;
        if (!model->BSIM3V2pnsubGiven)
            model->BSIM3V2pnsub = 0.0;
        if (!model->BSIM3V2pnpeakGiven)
            model->BSIM3V2pnpeak = 0.0;
        if (!model->BSIM3V2pngateGiven)
            model->BSIM3V2pngate = 0.0;
        if (!model->BSIM3V2pvbmGiven)
	    model->BSIM3V2pvbm = 0.0;
        if (!model->BSIM3V2pxtGiven)
	    model->BSIM3V2pxt = 0.0;
        if (!model->BSIM3V2pkt1Given)
            model->BSIM3V2pkt1 = 0.0; 
        if (!model->BSIM3V2pkt1lGiven)
            model->BSIM3V2pkt1l = 0.0;
        if (!model->BSIM3V2pkt2Given)
            model->BSIM3V2pkt2 = 0.0;
        if (!model->BSIM3V2pk3Given)
            model->BSIM3V2pk3 = 0.0;      
        if (!model->BSIM3V2pk3bGiven)
            model->BSIM3V2pk3b = 0.0;      
        if (!model->BSIM3V2pw0Given)
            model->BSIM3V2pw0 = 0.0;    
        if (!model->BSIM3V2pnlxGiven)
            model->BSIM3V2pnlx = 0.0;     
        if (!model->BSIM3V2pdvt0Given)
            model->BSIM3V2pdvt0 = 0.0;    
        if (!model->BSIM3V2pdvt1Given)
            model->BSIM3V2pdvt1 = 0.0;      
        if (!model->BSIM3V2pdvt2Given)
            model->BSIM3V2pdvt2 = 0.0;
        if (!model->BSIM3V2pdvt0wGiven)
            model->BSIM3V2pdvt0w = 0.0;    
        if (!model->BSIM3V2pdvt1wGiven)
            model->BSIM3V2pdvt1w = 0.0;      
        if (!model->BSIM3V2pdvt2wGiven)
            model->BSIM3V2pdvt2w = 0.0;
        if (!model->BSIM3V2pdroutGiven)
            model->BSIM3V2pdrout = 0.0;     
        if (!model->BSIM3V2pdsubGiven)
            model->BSIM3V2pdsub = 0.0;
        if (!model->BSIM3V2pvth0Given)
           model->BSIM3V2pvth0 = 0.0;
        if (!model->BSIM3V2puaGiven)
            model->BSIM3V2pua = 0.0;
        if (!model->BSIM3V2pua1Given)
            model->BSIM3V2pua1 = 0.0;
        if (!model->BSIM3V2pubGiven)
            model->BSIM3V2pub = 0.0;
        if (!model->BSIM3V2pub1Given)
            model->BSIM3V2pub1 = 0.0;
        if (!model->BSIM3V2pucGiven)
            model->BSIM3V2puc = 0.0;
        if (!model->BSIM3V2puc1Given)
            model->BSIM3V2puc1 = 0.0;
        if (!model->BSIM3V2pu0Given)
            model->BSIM3V2pu0 = 0.0;
        if (!model->BSIM3V2puteGiven)
	    model->BSIM3V2pute = 0.0;    
        if (!model->BSIM3V2pvoffGiven)
	    model->BSIM3V2pvoff = 0.0;
        if (!model->BSIM3V2pdeltaGiven)  
            model->BSIM3V2pdelta = 0.0;
        if (!model->BSIM3V2prdswGiven)
            model->BSIM3V2prdsw = 0.0;
        if (!model->BSIM3V2pprwbGiven)
            model->BSIM3V2pprwb = 0.0;
        if (!model->BSIM3V2pprwgGiven)
            model->BSIM3V2pprwg = 0.0;
        if (!model->BSIM3V2pprtGiven)
            model->BSIM3V2pprt = 0.0;
        if (!model->BSIM3V2peta0Given)
            model->BSIM3V2peta0 = 0.0;
        if (!model->BSIM3V2petabGiven)
            model->BSIM3V2petab = 0.0;
        if (!model->BSIM3V2ppclmGiven)
            model->BSIM3V2ppclm = 0.0; 
        if (!model->BSIM3V2ppdibl1Given)
            model->BSIM3V2ppdibl1 = 0.0;
        if (!model->BSIM3V2ppdibl2Given)
            model->BSIM3V2ppdibl2 = 0.0;
        if (!model->BSIM3V2ppdiblbGiven)
            model->BSIM3V2ppdiblb = 0.0;
        if (!model->BSIM3V2ppscbe1Given)
            model->BSIM3V2ppscbe1 = 0.0;
        if (!model->BSIM3V2ppscbe2Given)
            model->BSIM3V2ppscbe2 = 0.0;
        if (!model->BSIM3V2ppvagGiven)
            model->BSIM3V2ppvag = 0.0;     
        if (!model->BSIM3V2pwrGiven)  
            model->BSIM3V2pwr = 0.0;
        if (!model->BSIM3V2pdwgGiven)  
            model->BSIM3V2pdwg = 0.0;
        if (!model->BSIM3V2pdwbGiven)  
            model->BSIM3V2pdwb = 0.0;
        if (!model->BSIM3V2pb0Given)
            model->BSIM3V2pb0 = 0.0;
        if (!model->BSIM3V2pb1Given)  
            model->BSIM3V2pb1 = 0.0;
        if (!model->BSIM3V2palpha0Given)  
            model->BSIM3V2palpha0 = 0.0;
        if (!model->BSIM3V2palpha1Given)
            model->BSIM3V2palpha1 = 0.0;
        if (!model->BSIM3V2pbeta0Given)  
            model->BSIM3V2pbeta0 = 0.0;
        if (!model->BSIM3V2pvfbGiven)
            model->BSIM3V2pvfb = 0.0;

        if (!model->BSIM3V2pelmGiven)  
            model->BSIM3V2pelm = 0.0;
        if (!model->BSIM3V2pcgslGiven)  
            model->BSIM3V2pcgsl = 0.0;
        if (!model->BSIM3V2pcgdlGiven)  
            model->BSIM3V2pcgdl = 0.0;
        if (!model->BSIM3V2pckappaGiven)  
            model->BSIM3V2pckappa = 0.0;
        if (!model->BSIM3V2pcfGiven)  
            model->BSIM3V2pcf = 0.0;
        if (!model->BSIM3V2pclcGiven)  
            model->BSIM3V2pclc = 0.0;
        if (!model->BSIM3V2pcleGiven)  
            model->BSIM3V2pcle = 0.0;
        if (!model->BSIM3V2pvfbcvGiven)  
            model->BSIM3V2pvfbcv = 0.0;
        if (!model->BSIM3V2pacdeGiven)
            model->BSIM3V2pacde = 0.0;
        if (!model->BSIM3V2pmoinGiven)
            model->BSIM3V2pmoin = 0.0;
        if (!model->BSIM3V2pnoffGiven)
            model->BSIM3V2pnoff = 0.0;
        if (!model->BSIM3V2pvoffcvGiven)
            model->BSIM3V2pvoffcv = 0.0;

        /* unit degree celcius */
        if (!model->BSIM3V2tnomGiven)  
    	    model->BSIM3V2tnom = ckt->CKTnomTemp; 
        /* else
            model->BSIM3V2tnom = model->BSIM3V2tnom + 273.15;    */
        if (!model->BSIM3V2LintGiven)  
           model->BSIM3V2Lint = 0.0;
        if (!model->BSIM3V2LlGiven)  
           model->BSIM3V2Ll = 0.0;
        if (!model->BSIM3V2LlcGiven)
           model->BSIM3V2Llc = model->BSIM3V2Ll;
        if (!model->BSIM3V2LlnGiven)  
           model->BSIM3V2Lln = 1.0;
        if (!model->BSIM3V2LwGiven)  
           model->BSIM3V2Lw = 0.0;
        if (!model->BSIM3V2LwcGiven)
           model->BSIM3V2Lwc = model->BSIM3V2Lw;
        if (!model->BSIM3V2LwnGiven)  
           model->BSIM3V2Lwn = 1.0;
        if (!model->BSIM3V2LwlGiven)  
           model->BSIM3V2Lwl = 0.0;
        if (!model->BSIM3V2LwlcGiven)
           model->BSIM3V2Lwlc = model->BSIM3V2Lwl;
        if (!model->BSIM3V2LminGiven)  
           model->BSIM3V2Lmin = 0.0;
        if (!model->BSIM3V2LmaxGiven)  
           model->BSIM3V2Lmax = 1.0;
        if (!model->BSIM3V2WintGiven)  
           model->BSIM3V2Wint = 0.0;
        if (!model->BSIM3V2WlGiven)  
           model->BSIM3V2Wl = 0.0;
        if (!model->BSIM3V2WlcGiven)
           model->BSIM3V2Wlc = model->BSIM3V2Wl;
        if (!model->BSIM3V2WlnGiven)  
           model->BSIM3V2Wln = 1.0;
        if (!model->BSIM3V2WwGiven)  
           model->BSIM3V2Ww = 0.0;
        if (!model->BSIM3V2WwcGiven)
           model->BSIM3V2Wwc = model->BSIM3V2Ww;
        if (!model->BSIM3V2WwnGiven)  
           model->BSIM3V2Wwn = 1.0;
        if (!model->BSIM3V2WwlGiven)  
           model->BSIM3V2Wwl = 0.0;
        if (!model->BSIM3V2WwlcGiven)
           model->BSIM3V2Wwlc = model->BSIM3V2Wwl;
        if (!model->BSIM3V2WminGiven)  
           model->BSIM3V2Wmin = 0.0;
        if (!model->BSIM3V2WmaxGiven)  
           model->BSIM3V2Wmax = 1.0;
        if (!model->BSIM3V2dwcGiven)  
           model->BSIM3V2dwc = model->BSIM3V2Wint;
        if (!model->BSIM3V2dlcGiven)  
           model->BSIM3V2dlc = model->BSIM3V2Lint;
	if (!model->BSIM3V2cfGiven)
            model->BSIM3V2cf = 2.0 * EPSOX / PI
			   * log(1.0 + 0.4e-6 / model->BSIM3V2tox);
        if (!model->BSIM3V2cgdoGiven)
	{   if (model->BSIM3V2dlcGiven && (model->BSIM3V2dlc > 0.0))
	    {   model->BSIM3V2cgdo = model->BSIM3V2dlc * model->BSIM3V2cox
				 - model->BSIM3V2cgdl ;
	    }
	    else
	        model->BSIM3V2cgdo = 0.6 * model->BSIM3V2xj * model->BSIM3V2cox; 
	}
        if (!model->BSIM3V2cgsoGiven)
	{   if (model->BSIM3V2dlcGiven && (model->BSIM3V2dlc > 0.0))
	    {   model->BSIM3V2cgso = model->BSIM3V2dlc * model->BSIM3V2cox
				 - model->BSIM3V2cgsl ;
	    }
	    else
	        model->BSIM3V2cgso = 0.6 * model->BSIM3V2xj * model->BSIM3V2cox; 
	}

        if (!model->BSIM3V2cgboGiven)
	{   model->BSIM3V2cgbo = 2.0 * model->BSIM3V2dwc * model->BSIM3V2cox;
	}
        if (!model->BSIM3V2xpartGiven)
            model->BSIM3V2xpart = 0.0;
        if (!model->BSIM3V2sheetResistanceGiven)
            model->BSIM3V2sheetResistance = 0.0;
        if (!model->BSIM3V2unitAreaJctCapGiven)
            model->BSIM3V2unitAreaJctCap = 5.0E-4;
        if (!model->BSIM3V2unitLengthSidewallJctCapGiven)
            model->BSIM3V2unitLengthSidewallJctCap = 5.0E-10;
        if (!model->BSIM3V2unitLengthGateSidewallJctCapGiven)
            model->BSIM3V2unitLengthGateSidewallJctCap = model->BSIM3V2unitLengthSidewallJctCap ;
        if (!model->BSIM3V2jctSatCurDensityGiven)
            model->BSIM3V2jctSatCurDensity = 1.0E-4;
        if (!model->BSIM3V2jctSidewallSatCurDensityGiven)
            model->BSIM3V2jctSidewallSatCurDensity = 0.0;
        if (!model->BSIM3V2bulkJctPotentialGiven)
            model->BSIM3V2bulkJctPotential = 1.0;
        if (!model->BSIM3V2sidewallJctPotentialGiven)
            model->BSIM3V2sidewallJctPotential = 1.0;
        if (!model->BSIM3V2GatesidewallJctPotentialGiven)
            model->BSIM3V2GatesidewallJctPotential = model->BSIM3V2sidewallJctPotential;
        if (!model->BSIM3V2bulkJctBotGradingCoeffGiven)
            model->BSIM3V2bulkJctBotGradingCoeff = 0.5;
        if (!model->BSIM3V2bulkJctSideGradingCoeffGiven)
            model->BSIM3V2bulkJctSideGradingCoeff = 0.33;
        if (!model->BSIM3V2bulkJctGateSideGradingCoeffGiven)
            model->BSIM3V2bulkJctGateSideGradingCoeff = model->BSIM3V2bulkJctSideGradingCoeff;
        if (!model->BSIM3V2jctEmissionCoeffGiven)
            model->BSIM3V2jctEmissionCoeff = 1.0;
        if (!model->BSIM3V2jctTempExponentGiven)
            model->BSIM3V2jctTempExponent = 3.0;
        if (!model->BSIM3V2oxideTrapDensityAGiven)
	{   if (model->BSIM3V2type == NMOS)
                model->BSIM3V2oxideTrapDensityA = 1e20;
            else
                model->BSIM3V2oxideTrapDensityA=9.9e18;
	}
        if (!model->BSIM3V2oxideTrapDensityBGiven)
	{   if (model->BSIM3V2type == NMOS)
                model->BSIM3V2oxideTrapDensityB = 5e4;
            else
                model->BSIM3V2oxideTrapDensityB = 2.4e3;
	}
        if (!model->BSIM3V2oxideTrapDensityCGiven)
	{   if (model->BSIM3V2type == NMOS)
                model->BSIM3V2oxideTrapDensityC = -1.4e-12;
            else
                model->BSIM3V2oxideTrapDensityC = 1.4e-12;

	}
        if (!model->BSIM3V2emGiven)
            model->BSIM3V2em = 4.1e7; /* V/m */
        if (!model->BSIM3V2efGiven)
            model->BSIM3V2ef = 1.0;
        if (!model->BSIM3V2afGiven)
            model->BSIM3V2af = 1.0;
        if (!model->BSIM3V2kfGiven)
            model->BSIM3V2kf = 0.0;
        /* loop through all the instances of the model */
        for (here = model->BSIM3V2instances; here != NULL ;
             here=here->BSIM3V2nextInstance) 
	{   
           if (here->BSIM3V2owner == ARCHme) {
            /* allocate a chunk of the state vector */
            here->BSIM3V2states = *states;
            *states += BSIM3V2numStates;
	   }
            /* perform the parameter defaulting */
            if (!here->BSIM3V2drainAreaGiven)
                here->BSIM3V2drainArea = 0.0;
            if (!here->BSIM3V2drainPerimeterGiven)
                here->BSIM3V2drainPerimeter = 0.0;
            if (!here->BSIM3V2drainSquaresGiven)
                here->BSIM3V2drainSquares = 1.0;
            if (!here->BSIM3V2icVBSGiven)
                here->BSIM3V2icVBS = 0.0;
            if (!here->BSIM3V2icVDSGiven)
                here->BSIM3V2icVDS = 0.0;
            if (!here->BSIM3V2icVGSGiven)
                here->BSIM3V2icVGS = 0.0;
            if (!here->BSIM3V2lGiven)
                here->BSIM3V2l = 5.0e-6;
            if (!here->BSIM3V2sourceAreaGiven)
                here->BSIM3V2sourceArea = 0.0;
            if (!here->BSIM3V2sourcePerimeterGiven)
                here->BSIM3V2sourcePerimeter = 0.0;
            if (!here->BSIM3V2sourceSquaresGiven)
                here->BSIM3V2sourceSquares = 1.0;
            if (!here->BSIM3V2wGiven)
                here->BSIM3V2w = 5.0e-6;
            if (!here->BSIM3V2nqsModGiven)
                here->BSIM3V2nqsMod = 0;
                    
            /* process drain series resistance */
            if ((model->BSIM3V2sheetResistance > 0.0) && 
                (here->BSIM3V2drainSquares > 0.0 ) &&
                (here->BSIM3V2dNodePrime == 0))
	    {   error = CKTmkVolt(ckt,&tmp,here->BSIM3V2name,"drain");
                if(error) return(error);
                here->BSIM3V2dNodePrime = tmp->number;
            }
	    else
	    {   here->BSIM3V2dNodePrime = here->BSIM3V2dNode;
            }
                   
            /* process source series resistance */
            if ((model->BSIM3V2sheetResistance > 0.0) && 
                (here->BSIM3V2sourceSquares > 0.0 ) &&
                (here->BSIM3V2sNodePrime == 0)) 
	    {   error = CKTmkVolt(ckt,&tmp,here->BSIM3V2name,"source");
                if(error) return(error);
                here->BSIM3V2sNodePrime = tmp->number;
            }
	    else 
	    {   here->BSIM3V2sNodePrime = here->BSIM3V2sNode;
            }

 /* internal charge node */
                   
            if ((here->BSIM3V2nqsMod) && (here->BSIM3V2qNode == 0)) 
	    {   error = CKTmkVolt(ckt,&tmp,here->BSIM3V2name,"charge");
                if(error) return(error);
                here->BSIM3V2qNode = tmp->number;
            }
	    else 
	    {   here->BSIM3V2qNode = 0;
            }

        /* set Sparse Matrix Pointers */

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==(double *)NULL){\
    return(E_NOMEM);\
}

            TSTALLOC(BSIM3V2DdPtr, BSIM3V2dNode, BSIM3V2dNode)
            TSTALLOC(BSIM3V2GgPtr, BSIM3V2gNode, BSIM3V2gNode)
            TSTALLOC(BSIM3V2SsPtr, BSIM3V2sNode, BSIM3V2sNode)
            TSTALLOC(BSIM3V2BbPtr, BSIM3V2bNode, BSIM3V2bNode)
            TSTALLOC(BSIM3V2DPdpPtr, BSIM3V2dNodePrime, BSIM3V2dNodePrime)
            TSTALLOC(BSIM3V2SPspPtr, BSIM3V2sNodePrime, BSIM3V2sNodePrime)
            TSTALLOC(BSIM3V2DdpPtr, BSIM3V2dNode, BSIM3V2dNodePrime)
            TSTALLOC(BSIM3V2GbPtr, BSIM3V2gNode, BSIM3V2bNode)
            TSTALLOC(BSIM3V2GdpPtr, BSIM3V2gNode, BSIM3V2dNodePrime)
            TSTALLOC(BSIM3V2GspPtr, BSIM3V2gNode, BSIM3V2sNodePrime)
            TSTALLOC(BSIM3V2SspPtr, BSIM3V2sNode, BSIM3V2sNodePrime)
            TSTALLOC(BSIM3V2BdpPtr, BSIM3V2bNode, BSIM3V2dNodePrime)
            TSTALLOC(BSIM3V2BspPtr, BSIM3V2bNode, BSIM3V2sNodePrime)
            TSTALLOC(BSIM3V2DPspPtr, BSIM3V2dNodePrime, BSIM3V2sNodePrime)
            TSTALLOC(BSIM3V2DPdPtr, BSIM3V2dNodePrime, BSIM3V2dNode)
            TSTALLOC(BSIM3V2BgPtr, BSIM3V2bNode, BSIM3V2gNode)
            TSTALLOC(BSIM3V2DPgPtr, BSIM3V2dNodePrime, BSIM3V2gNode)
            TSTALLOC(BSIM3V2SPgPtr, BSIM3V2sNodePrime, BSIM3V2gNode)
            TSTALLOC(BSIM3V2SPsPtr, BSIM3V2sNodePrime, BSIM3V2sNode)
            TSTALLOC(BSIM3V2DPbPtr, BSIM3V2dNodePrime, BSIM3V2bNode)
            TSTALLOC(BSIM3V2SPbPtr, BSIM3V2sNodePrime, BSIM3V2bNode)
            TSTALLOC(BSIM3V2SPdpPtr, BSIM3V2sNodePrime, BSIM3V2dNodePrime)

            TSTALLOC(BSIM3V2QqPtr, BSIM3V2qNode, BSIM3V2qNode)
 
            TSTALLOC(BSIM3V2QdpPtr, BSIM3V2qNode, BSIM3V2dNodePrime)
            TSTALLOC(BSIM3V2QspPtr, BSIM3V2qNode, BSIM3V2sNodePrime)
            TSTALLOC(BSIM3V2QgPtr, BSIM3V2qNode, BSIM3V2gNode)
            TSTALLOC(BSIM3V2QbPtr, BSIM3V2qNode, BSIM3V2bNode)
            TSTALLOC(BSIM3V2DPqPtr, BSIM3V2dNodePrime, BSIM3V2qNode)
            TSTALLOC(BSIM3V2SPqPtr, BSIM3V2sNodePrime, BSIM3V2qNode)
            TSTALLOC(BSIM3V2GqPtr, BSIM3V2gNode, BSIM3V2qNode)
            TSTALLOC(BSIM3V2BqPtr, BSIM3V2bNode, BSIM3V2qNode)

        }
    }
    return(OK);
}  


int
BSIM3V2unsetup(inModel,ckt)
    GENmodel *inModel;
    CKTcircuit *ckt;
{
#ifndef HAS_BATCHSIM
    BSIM3V2model *model;
    BSIM3V2instance *here;

    for (model = (BSIM3V2model *)inModel; model != NULL;
	    model = model->BSIM3V2nextModel)
    {
        for (here = model->BSIM3V2instances; here != NULL;
                here=here->BSIM3V2nextInstance)
	{
	    if (here->BSIM3V2dNodePrime
		    && here->BSIM3V2dNodePrime != here->BSIM3V2dNode)
	    {
		CKTdltNNum(ckt, here->BSIM3V2dNodePrime);
		here->BSIM3V2dNodePrime = 0;
	    }
	    if (here->BSIM3V2sNodePrime
		    && here->BSIM3V2sNodePrime != here->BSIM3V2sNode)
	    {
		CKTdltNNum(ckt, here->BSIM3V2sNodePrime);
		here->BSIM3V2sNodePrime = 0;
	    }
	}
    }
#endif
    return OK;
}
