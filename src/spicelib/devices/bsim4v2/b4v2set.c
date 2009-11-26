/**** BSIM4.2.1, Released by Xuemei Xi 10/05/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b4set.c of BSIM4.2.1.
 * Author: 2000 Weidong Liu
 * Authors: Xuemei Xi, Kanyu M. Cao, Hui Wan, Mansun Chan, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "jobdefs.h"  /* Needed because the model searches for noise Analysis */   
#include "ftedefs.h"  /*                   "       "                          */
#include "smpdefs.h"
#include "cktdefs.h"
#include "bsim4v2def.h"
#include "const.h"
#include "sperror.h"


#define MAX_EXP 5.834617425e14
#define MIN_EXP 1.713908431e-15
#define EXP_THRESHOLD 34.0
#define EPS0 8.85418e-12
#define EPSSI 1.03594e-10
#define PI 3.141592654
#define Charge_q 1.60219e-19

int
BSIM4v2RdseffGeo(double nf, int geo, int rgeo, int minSD, double Weffcj, double Rsh, double DMCG, double DMCI, double DMDG, int Type,double *Rtot);

int
BSIM4v2setup(
SMPmatrix *matrix,
GENmodel *inModel,
CKTcircuit *ckt,
int *states)
{
BSIM4v2model *model = (BSIM4v2model*)inModel;
BSIM4v2instance *here;
int error;
CKTnode *tmp;

int    noiseAnalGiven = 0, createNode;  /* Criteria for new node creation */
double Rtot, DMCGeff, DMCIeff, DMDGeff;
JOB   *job;

    /* Search for a noise analysis request */
    for (job = ((TSKtask *)(ft_curckt->ci_curTask))->jobs;job;job = job->JOBnextJob) {
        if(strcmp(job->JOBname,"Noise Analysis")==0) {
            noiseAnalGiven = 1;
            break;
        }
    }

    /*  loop through all the BSIM4v2 device models */
    for( ; model != NULL; model = model->BSIM4v2nextModel )
    {   /* process defaults of model parameters */
        if (!model->BSIM4v2typeGiven)
            model->BSIM4v2type = NMOS;     

        if (!model->BSIM4v2mobModGiven) 
            model->BSIM4v2mobMod = 0;
	else if ((model->BSIM4v2mobMod != 0) && (model->BSIM4v2mobMod != 1)
	         && (model->BSIM4v2mobMod != 2))
	{   model->BSIM4v2mobMod = 0;
            printf("Warning: mobMod has been set to its default value: 0.\n");
	}

        if (!model->BSIM4v2binUnitGiven) 
            model->BSIM4v2binUnit = 1;
        if (!model->BSIM4v2paramChkGiven) 
            model->BSIM4v2paramChk = 1;

        if (!model->BSIM4v2dioModGiven)
            model->BSIM4v2dioMod = 1;
        else if ((model->BSIM4v2dioMod != 0) && (model->BSIM4v2dioMod != 1)
            && (model->BSIM4v2dioMod != 2))
        {   model->BSIM4v2dioMod = 1;
            printf("Warning: dioMod has been set to its default value: 1.\n");
        }

        if (!model->BSIM4v2capModGiven) 
            model->BSIM4v2capMod = 2;
        else if ((model->BSIM4v2capMod != 0) && (model->BSIM4v2capMod != 1)
            && (model->BSIM4v2capMod != 2))
        {   model->BSIM4v2capMod = 2;
            printf("Warning: capMod has been set to its default value: 2.\n");
        }

        if (!model->BSIM4v2rdsModGiven)
            model->BSIM4v2rdsMod = 0;
	else if ((model->BSIM4v2rdsMod != 0) && (model->BSIM4v2rdsMod != 1))
        {   model->BSIM4v2rdsMod = 0;
	    printf("Warning: rdsMod has been set to its default value: 0.\n");
	}
        if (!model->BSIM4v2rbodyModGiven)
            model->BSIM4v2rbodyMod = 0;
        else if ((model->BSIM4v2rbodyMod != 0) && (model->BSIM4v2rbodyMod != 1))
        {   model->BSIM4v2rbodyMod = 0;
            printf("Warning: rbodyMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v2rgateModGiven)
            model->BSIM4v2rgateMod = 0;
        else if ((model->BSIM4v2rgateMod != 0) && (model->BSIM4v2rgateMod != 1)
            && (model->BSIM4v2rgateMod != 2) && (model->BSIM4v2rgateMod != 3))
        {   model->BSIM4v2rgateMod = 0;
            printf("Warning: rgateMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v2perModGiven)
            model->BSIM4v2perMod = 1;
        else if ((model->BSIM4v2perMod != 0) && (model->BSIM4v2perMod != 1))
        {   model->BSIM4v2perMod = 1;
            printf("Warning: perMod has been set to its default value: 1.\n");
        }

        if (!model->BSIM4v2geoModGiven)
            model->BSIM4v2geoMod = 0;

        if (!model->BSIM4v2fnoiModGiven) 
            model->BSIM4v2fnoiMod = 1;
        else if ((model->BSIM4v2fnoiMod != 0) && (model->BSIM4v2fnoiMod != 1))
        {   model->BSIM4v2fnoiMod = 1;
            printf("Warning: fnoiMod has been set to its default value: 1.\n");
        }
        if (!model->BSIM4v2tnoiModGiven)
            model->BSIM4v2tnoiMod = 0; /* WDLiu: tnoiMod=1 needs to set internal S/D nodes */
        else if ((model->BSIM4v2tnoiMod != 0) && (model->BSIM4v2tnoiMod != 1))
        {   model->BSIM4v2tnoiMod = 0;
            printf("Warning: tnoiMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v2trnqsModGiven)
            model->BSIM4v2trnqsMod = 0; 
        else if ((model->BSIM4v2trnqsMod != 0) && (model->BSIM4v2trnqsMod != 1))
        {   model->BSIM4v2trnqsMod = 0;
            printf("Warning: trnqsMod has been set to its default value: 0.\n");
        }
        if (!model->BSIM4v2acnqsModGiven)
            model->BSIM4v2acnqsMod = 0;
        else if ((model->BSIM4v2acnqsMod != 0) && (model->BSIM4v2acnqsMod != 1))
        {   model->BSIM4v2acnqsMod = 0;
            printf("Warning: acnqsMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v2igcModGiven)
            model->BSIM4v2igcMod = 0;
        else if ((model->BSIM4v2igcMod != 0) && (model->BSIM4v2igcMod != 1))
        {   model->BSIM4v2igcMod = 0;
            printf("Warning: igcMod has been set to its default value: 0.\n");
        }
        if (!model->BSIM4v2igbModGiven)
            model->BSIM4v2igbMod = 0;
        else if ((model->BSIM4v2igbMod != 0) && (model->BSIM4v2igbMod != 1))
        {   model->BSIM4v2igbMod = 0;
            printf("Warning: igbMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v2versionGiven) 
            model->BSIM4v2version = "4.2.1";
        if (!model->BSIM4v2toxrefGiven)
            model->BSIM4v2toxref = 30.0e-10;
        if (!model->BSIM4v2toxeGiven)
            model->BSIM4v2toxe = 30.0e-10;
        if (!model->BSIM4v2toxpGiven)
            model->BSIM4v2toxp = model->BSIM4v2toxe;
        if (!model->BSIM4v2toxmGiven)
            model->BSIM4v2toxm = model->BSIM4v2toxe;
        if (!model->BSIM4v2dtoxGiven)
            model->BSIM4v2dtox = 0.0;
        if (!model->BSIM4v2epsroxGiven)
            model->BSIM4v2epsrox = 3.9;

        if (!model->BSIM4v2cdscGiven)
	    model->BSIM4v2cdsc = 2.4e-4;   /* unit Q/V/m^2  */
        if (!model->BSIM4v2cdscbGiven)
	    model->BSIM4v2cdscb = 0.0;   /* unit Q/V/m^2  */    
	    if (!model->BSIM4v2cdscdGiven)
	    model->BSIM4v2cdscd = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM4v2citGiven)
	    model->BSIM4v2cit = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM4v2nfactorGiven)
	    model->BSIM4v2nfactor = 1.0;
        if (!model->BSIM4v2xjGiven)
            model->BSIM4v2xj = .15e-6;
        if (!model->BSIM4v2vsatGiven)
            model->BSIM4v2vsat = 8.0e4;    /* unit m/s */ 
        if (!model->BSIM4v2atGiven)
            model->BSIM4v2at = 3.3e4;    /* unit m/s */ 
        if (!model->BSIM4v2a0Given)
            model->BSIM4v2a0 = 1.0;  
        if (!model->BSIM4v2agsGiven)
            model->BSIM4v2ags = 0.0;
        if (!model->BSIM4v2a1Given)
            model->BSIM4v2a1 = 0.0;
        if (!model->BSIM4v2a2Given)
            model->BSIM4v2a2 = 1.0;
        if (!model->BSIM4v2ketaGiven)
            model->BSIM4v2keta = -0.047;    /* unit  / V */
        if (!model->BSIM4v2nsubGiven)
            model->BSIM4v2nsub = 6.0e16;   /* unit 1/cm3 */
        if (!model->BSIM4v2ndepGiven)
            model->BSIM4v2ndep = 1.7e17;   /* unit 1/cm3 */
        if (!model->BSIM4v2nsdGiven)
            model->BSIM4v2nsd = 1.0e20;   /* unit 1/cm3 */
        if (!model->BSIM4v2phinGiven)
            model->BSIM4v2phin = 0.0; /* unit V */
        if (!model->BSIM4v2ngateGiven)
            model->BSIM4v2ngate = 0;   /* unit 1/cm3 */
        if (!model->BSIM4v2vbmGiven)
	    model->BSIM4v2vbm = -3.0;
        if (!model->BSIM4v2xtGiven)
	    model->BSIM4v2xt = 1.55e-7;
        if (!model->BSIM4v2kt1Given)
            model->BSIM4v2kt1 = -0.11;      /* unit V */
        if (!model->BSIM4v2kt1lGiven)
            model->BSIM4v2kt1l = 0.0;      /* unit V*m */
        if (!model->BSIM4v2kt2Given)
            model->BSIM4v2kt2 = 0.022;      /* No unit */
        if (!model->BSIM4v2k3Given)
            model->BSIM4v2k3 = 80.0;      
        if (!model->BSIM4v2k3bGiven)
            model->BSIM4v2k3b = 0.0;      
        if (!model->BSIM4v2w0Given)
            model->BSIM4v2w0 = 2.5e-6;    
        if (!model->BSIM4v2lpe0Given)
            model->BSIM4v2lpe0 = 1.74e-7;     
        if (!model->BSIM4v2lpebGiven)
            model->BSIM4v2lpeb = 0.0;
        if (!model->BSIM4v2dvtp0Given)
            model->BSIM4v2dvtp0 = 0.0;
        if (!model->BSIM4v2dvtp1Given)
            model->BSIM4v2dvtp1 = 0.0;
        if (!model->BSIM4v2dvt0Given)
            model->BSIM4v2dvt0 = 2.2;    
        if (!model->BSIM4v2dvt1Given)
            model->BSIM4v2dvt1 = 0.53;      
        if (!model->BSIM4v2dvt2Given)
            model->BSIM4v2dvt2 = -0.032;   /* unit 1 / V */     

        if (!model->BSIM4v2dvt0wGiven)
            model->BSIM4v2dvt0w = 0.0;    
        if (!model->BSIM4v2dvt1wGiven)
            model->BSIM4v2dvt1w = 5.3e6;    
        if (!model->BSIM4v2dvt2wGiven)
            model->BSIM4v2dvt2w = -0.032;   

        if (!model->BSIM4v2droutGiven)
            model->BSIM4v2drout = 0.56;     
        if (!model->BSIM4v2dsubGiven)
            model->BSIM4v2dsub = model->BSIM4v2drout;     
        if (!model->BSIM4v2vth0Given)
            model->BSIM4v2vth0 = (model->BSIM4v2type == NMOS) ? 0.7 : -0.7;
        if (!model->BSIM4v2euGiven)
            model->BSIM4v2eu = (model->BSIM4v2type == NMOS) ? 1.67 : 1.0;;
        if (!model->BSIM4v2uaGiven)
            model->BSIM4v2ua = (model->BSIM4v2mobMod == 2) ? 1.0e-15 : 1.0e-9; /* unit m/V */
        if (!model->BSIM4v2ua1Given)
            model->BSIM4v2ua1 = 1.0e-9;      /* unit m/V */
        if (!model->BSIM4v2ubGiven)
            model->BSIM4v2ub = 1.0e-19;     /* unit (m/V)**2 */
        if (!model->BSIM4v2ub1Given)
            model->BSIM4v2ub1 = -1.0e-18;     /* unit (m/V)**2 */
        if (!model->BSIM4v2ucGiven)
            model->BSIM4v2uc = (model->BSIM4v2mobMod == 1) ? -0.0465 : -0.0465e-9;   
        if (!model->BSIM4v2uc1Given)
            model->BSIM4v2uc1 = (model->BSIM4v2mobMod == 1) ? -0.056 : -0.056e-9;   
        if (!model->BSIM4v2u0Given)
            model->BSIM4v2u0 = (model->BSIM4v2type == NMOS) ? 0.067 : 0.025;
        if (!model->BSIM4v2uteGiven)
	    model->BSIM4v2ute = -1.5;    
        if (!model->BSIM4v2voffGiven)
	    model->BSIM4v2voff = -0.08;
        if (!model->BSIM4v2vofflGiven)
            model->BSIM4v2voffl = 0.0;
        if (!model->BSIM4v2minvGiven)
            model->BSIM4v2minv = 0.0;
        if (!model->BSIM4v2fproutGiven)
            model->BSIM4v2fprout = 0.0;
        if (!model->BSIM4v2pditsGiven)
            model->BSIM4v2pdits = 0.0;
        if (!model->BSIM4v2pditsdGiven)
            model->BSIM4v2pditsd = 0.0;
        if (!model->BSIM4v2pditslGiven)
            model->BSIM4v2pditsl = 0.0;
        if (!model->BSIM4v2deltaGiven)  
           model->BSIM4v2delta = 0.01;
        if (!model->BSIM4v2rdswminGiven)
            model->BSIM4v2rdswmin = 0.0;
        if (!model->BSIM4v2rdwminGiven)
            model->BSIM4v2rdwmin = 0.0;
        if (!model->BSIM4v2rswminGiven)
            model->BSIM4v2rswmin = 0.0;
        if (!model->BSIM4v2rdswGiven)
	    model->BSIM4v2rdsw = 200.0; /* in ohm*um */     
        if (!model->BSIM4v2rdwGiven)
            model->BSIM4v2rdw = 100.0;
        if (!model->BSIM4v2rswGiven)
            model->BSIM4v2rsw = 100.0;
        if (!model->BSIM4v2prwgGiven)
            model->BSIM4v2prwg = 1.0; /* in 1/V */
        if (!model->BSIM4v2prwbGiven)
            model->BSIM4v2prwb = 0.0;      
        if (!model->BSIM4v2prtGiven)
        if (!model->BSIM4v2prtGiven)
            model->BSIM4v2prt = 0.0;      
        if (!model->BSIM4v2eta0Given)
            model->BSIM4v2eta0 = 0.08;      /* no unit  */ 
        if (!model->BSIM4v2etabGiven)
            model->BSIM4v2etab = -0.07;      /* unit  1/V */ 
        if (!model->BSIM4v2pclmGiven)
            model->BSIM4v2pclm = 1.3;      /* no unit  */ 
        if (!model->BSIM4v2pdibl1Given)
            model->BSIM4v2pdibl1 = 0.39;    /* no unit  */
        if (!model->BSIM4v2pdibl2Given)
            model->BSIM4v2pdibl2 = 0.0086;    /* no unit  */ 
        if (!model->BSIM4v2pdiblbGiven)
            model->BSIM4v2pdiblb = 0.0;    /* 1/V  */ 
        if (!model->BSIM4v2pscbe1Given)
            model->BSIM4v2pscbe1 = 4.24e8;     
        if (!model->BSIM4v2pscbe2Given)
            model->BSIM4v2pscbe2 = 1.0e-5;    
        if (!model->BSIM4v2pvagGiven)
            model->BSIM4v2pvag = 0.0;     
        if (!model->BSIM4v2wrGiven)  
            model->BSIM4v2wr = 1.0;
        if (!model->BSIM4v2dwgGiven)  
            model->BSIM4v2dwg = 0.0;
        if (!model->BSIM4v2dwbGiven)  
            model->BSIM4v2dwb = 0.0;
        if (!model->BSIM4v2b0Given)
            model->BSIM4v2b0 = 0.0;
        if (!model->BSIM4v2b1Given)  
            model->BSIM4v2b1 = 0.0;
        if (!model->BSIM4v2alpha0Given)  
            model->BSIM4v2alpha0 = 0.0;
        if (!model->BSIM4v2alpha1Given)
            model->BSIM4v2alpha1 = 0.0;
        if (!model->BSIM4v2beta0Given)  
            model->BSIM4v2beta0 = 30.0;
        if (!model->BSIM4v2agidlGiven)
            model->BSIM4v2agidl = 0.0;
        if (!model->BSIM4v2bgidlGiven)
            model->BSIM4v2bgidl = 2.3e9; /* V/m */
        if (!model->BSIM4v2cgidlGiven)
            model->BSIM4v2cgidl = 0.5; /* V^3 */
        if (!model->BSIM4v2egidlGiven)
            model->BSIM4v2egidl = 0.8; /* V */
        if (!model->BSIM4v2aigcGiven)
            model->BSIM4v2aigc = (model->BSIM4v2type == NMOS) ? 0.43 : 0.31;
        if (!model->BSIM4v2bigcGiven)
            model->BSIM4v2bigc = (model->BSIM4v2type == NMOS) ? 0.054 : 0.024;
        if (!model->BSIM4v2cigcGiven)
            model->BSIM4v2cigc = (model->BSIM4v2type == NMOS) ? 0.075 : 0.03;
        if (!model->BSIM4v2aigsdGiven)
            model->BSIM4v2aigsd = (model->BSIM4v2type == NMOS) ? 0.43 : 0.31;
        if (!model->BSIM4v2bigsdGiven)
            model->BSIM4v2bigsd = (model->BSIM4v2type == NMOS) ? 0.054 : 0.024;
        if (!model->BSIM4v2cigsdGiven)
            model->BSIM4v2cigsd = (model->BSIM4v2type == NMOS) ? 0.075 : 0.03;
        if (!model->BSIM4v2aigbaccGiven)
            model->BSIM4v2aigbacc = 0.43;
        if (!model->BSIM4v2bigbaccGiven)
            model->BSIM4v2bigbacc = 0.054;
        if (!model->BSIM4v2cigbaccGiven)
            model->BSIM4v2cigbacc = 0.075;
        if (!model->BSIM4v2aigbinvGiven)
            model->BSIM4v2aigbinv = 0.35;
        if (!model->BSIM4v2bigbinvGiven)
            model->BSIM4v2bigbinv = 0.03;
        if (!model->BSIM4v2cigbinvGiven)
            model->BSIM4v2cigbinv = 0.006;
        if (!model->BSIM4v2nigcGiven)
            model->BSIM4v2nigc = 1.0;
        if (!model->BSIM4v2nigbinvGiven)
            model->BSIM4v2nigbinv = 3.0;
        if (!model->BSIM4v2nigbaccGiven)
            model->BSIM4v2nigbacc = 1.0;
        if (!model->BSIM4v2ntoxGiven)
            model->BSIM4v2ntox = 1.0;
        if (!model->BSIM4v2eigbinvGiven)
            model->BSIM4v2eigbinv = 1.1;
        if (!model->BSIM4v2pigcdGiven)
            model->BSIM4v2pigcd = 1.0;
        if (!model->BSIM4v2poxedgeGiven)
            model->BSIM4v2poxedge = 1.0;
        if (!model->BSIM4v2xrcrg1Given)
            model->BSIM4v2xrcrg1 = 12.0;
        if (!model->BSIM4v2xrcrg2Given)
            model->BSIM4v2xrcrg2 = 1.0;
        if (!model->BSIM4v2ijthsfwdGiven)
            model->BSIM4v2ijthsfwd = 0.1; /* unit A */
        if (!model->BSIM4v2ijthdfwdGiven)
            model->BSIM4v2ijthdfwd = model->BSIM4v2ijthsfwd;
        if (!model->BSIM4v2ijthsrevGiven)
            model->BSIM4v2ijthsrev = 0.1; /* unit A */
        if (!model->BSIM4v2ijthdrevGiven)
            model->BSIM4v2ijthdrev = model->BSIM4v2ijthsrev;
        if (!model->BSIM4v2tnoiaGiven)
            model->BSIM4v2tnoia = 1.5;
        if (!model->BSIM4v2tnoibGiven)
            model->BSIM4v2tnoib = 3.5;
        if (!model->BSIM4v2ntnoiGiven)
            model->BSIM4v2ntnoi = 1.0;

        if (!model->BSIM4v2xjbvsGiven)
            model->BSIM4v2xjbvs = 1.0; /* no unit */
        if (!model->BSIM4v2xjbvdGiven)
            model->BSIM4v2xjbvd = model->BSIM4v2xjbvs;
        if (!model->BSIM4v2bvsGiven)
            model->BSIM4v2bvs = 10.0; /* V */
        if (!model->BSIM4v2bvdGiven)
            model->BSIM4v2bvd = model->BSIM4v2bvs;
        if (!model->BSIM4v2gbminGiven)
            model->BSIM4v2gbmin = 1.0e-12; /* in mho */
        if (!model->BSIM4v2rbdbGiven)
            model->BSIM4v2rbdb = 50.0; /* in ohm */
        if (!model->BSIM4v2rbpbGiven)
            model->BSIM4v2rbpb = 50.0;
        if (!model->BSIM4v2rbsbGiven)
            model->BSIM4v2rbsb = 50.0;
        if (!model->BSIM4v2rbpsGiven)
            model->BSIM4v2rbps = 50.0;
        if (!model->BSIM4v2rbpdGiven)
            model->BSIM4v2rbpd = 50.0;

        if (!model->BSIM4v2cgslGiven)  
            model->BSIM4v2cgsl = 0.0;
        if (!model->BSIM4v2cgdlGiven)  
            model->BSIM4v2cgdl = 0.0;
        if (!model->BSIM4v2ckappasGiven)  
            model->BSIM4v2ckappas = 0.6;
        if (!model->BSIM4v2ckappadGiven)
            model->BSIM4v2ckappad = model->BSIM4v2ckappas;
        if (!model->BSIM4v2clcGiven)  
            model->BSIM4v2clc = 0.1e-6;
        if (!model->BSIM4v2cleGiven)  
            model->BSIM4v2cle = 0.6;
        if (!model->BSIM4v2vfbcvGiven)  
            model->BSIM4v2vfbcv = -1.0;
        if (!model->BSIM4v2acdeGiven)
            model->BSIM4v2acde = 1.0;
        if (!model->BSIM4v2moinGiven)
            model->BSIM4v2moin = 15.0;
        if (!model->BSIM4v2noffGiven)
            model->BSIM4v2noff = 1.0;
        if (!model->BSIM4v2voffcvGiven)
            model->BSIM4v2voffcv = 0.0;
        if (!model->BSIM4v2dmcgGiven)
            model->BSIM4v2dmcg = 0.0;
        if (!model->BSIM4v2dmciGiven)
            model->BSIM4v2dmci = model->BSIM4v2dmcg;
        if (!model->BSIM4v2dmdgGiven)
            model->BSIM4v2dmdg = 0.0;
        if (!model->BSIM4v2dmcgtGiven)
            model->BSIM4v2dmcgt = 0.0;
        if (!model->BSIM4v2xgwGiven)
            model->BSIM4v2xgw = 0.0;
        if (!model->BSIM4v2xglGiven)
            model->BSIM4v2xgl = 0.0;
        if (!model->BSIM4v2rshgGiven)
            model->BSIM4v2rshg = 0.1;
        if (!model->BSIM4v2ngconGiven)
            model->BSIM4v2ngcon = 1.0;
        if (!model->BSIM4v2tcjGiven)
            model->BSIM4v2tcj = 0.0;
        if (!model->BSIM4v2tpbGiven)
            model->BSIM4v2tpb = 0.0;
        if (!model->BSIM4v2tcjswGiven)
            model->BSIM4v2tcjsw = 0.0;
        if (!model->BSIM4v2tpbswGiven)
            model->BSIM4v2tpbsw = 0.0;
        if (!model->BSIM4v2tcjswgGiven)
            model->BSIM4v2tcjswg = 0.0;
        if (!model->BSIM4v2tpbswgGiven)
            model->BSIM4v2tpbswg = 0.0;

	/* Length dependence */
        if (!model->BSIM4v2lcdscGiven)
	    model->BSIM4v2lcdsc = 0.0;
        if (!model->BSIM4v2lcdscbGiven)
	    model->BSIM4v2lcdscb = 0.0;
	    if (!model->BSIM4v2lcdscdGiven) 
	    model->BSIM4v2lcdscd = 0.0;
        if (!model->BSIM4v2lcitGiven)
	    model->BSIM4v2lcit = 0.0;
        if (!model->BSIM4v2lnfactorGiven)
	    model->BSIM4v2lnfactor = 0.0;
        if (!model->BSIM4v2lxjGiven)
            model->BSIM4v2lxj = 0.0;
        if (!model->BSIM4v2lvsatGiven)
            model->BSIM4v2lvsat = 0.0;
        if (!model->BSIM4v2latGiven)
            model->BSIM4v2lat = 0.0;
        if (!model->BSIM4v2la0Given)
            model->BSIM4v2la0 = 0.0; 
        if (!model->BSIM4v2lagsGiven)
            model->BSIM4v2lags = 0.0;
        if (!model->BSIM4v2la1Given)
            model->BSIM4v2la1 = 0.0;
        if (!model->BSIM4v2la2Given)
            model->BSIM4v2la2 = 0.0;
        if (!model->BSIM4v2lketaGiven)
            model->BSIM4v2lketa = 0.0;
        if (!model->BSIM4v2lnsubGiven)
            model->BSIM4v2lnsub = 0.0;
        if (!model->BSIM4v2lndepGiven)
            model->BSIM4v2lndep = 0.0;
        if (!model->BSIM4v2lnsdGiven)
            model->BSIM4v2lnsd = 0.0;
        if (!model->BSIM4v2lphinGiven)
            model->BSIM4v2lphin = 0.0;
        if (!model->BSIM4v2lngateGiven)
            model->BSIM4v2lngate = 0.0;
        if (!model->BSIM4v2lvbmGiven)
	    model->BSIM4v2lvbm = 0.0;
        if (!model->BSIM4v2lxtGiven)
	    model->BSIM4v2lxt = 0.0;
        if (!model->BSIM4v2lkt1Given)
            model->BSIM4v2lkt1 = 0.0; 
        if (!model->BSIM4v2lkt1lGiven)
            model->BSIM4v2lkt1l = 0.0;
        if (!model->BSIM4v2lkt2Given)
            model->BSIM4v2lkt2 = 0.0;
        if (!model->BSIM4v2lk3Given)
            model->BSIM4v2lk3 = 0.0;      
        if (!model->BSIM4v2lk3bGiven)
            model->BSIM4v2lk3b = 0.0;      
        if (!model->BSIM4v2lw0Given)
            model->BSIM4v2lw0 = 0.0;    
        if (!model->BSIM4v2llpe0Given)
            model->BSIM4v2llpe0 = 0.0;
        if (!model->BSIM4v2llpebGiven)
            model->BSIM4v2llpeb = model->BSIM4v2llpe0;
        if (!model->BSIM4v2ldvtp0Given)
            model->BSIM4v2ldvtp0 = 0.0;
        if (!model->BSIM4v2ldvtp1Given)
            model->BSIM4v2ldvtp1 = 0.0;
        if (!model->BSIM4v2ldvt0Given)
            model->BSIM4v2ldvt0 = 0.0;    
        if (!model->BSIM4v2ldvt1Given)
            model->BSIM4v2ldvt1 = 0.0;      
        if (!model->BSIM4v2ldvt2Given)
            model->BSIM4v2ldvt2 = 0.0;
        if (!model->BSIM4v2ldvt0wGiven)
            model->BSIM4v2ldvt0w = 0.0;    
        if (!model->BSIM4v2ldvt1wGiven)
            model->BSIM4v2ldvt1w = 0.0;      
        if (!model->BSIM4v2ldvt2wGiven)
            model->BSIM4v2ldvt2w = 0.0;
        if (!model->BSIM4v2ldroutGiven)
            model->BSIM4v2ldrout = 0.0;     
        if (!model->BSIM4v2ldsubGiven)
            model->BSIM4v2ldsub = 0.0;
        if (!model->BSIM4v2lvth0Given)
           model->BSIM4v2lvth0 = 0.0;
        if (!model->BSIM4v2luaGiven)
            model->BSIM4v2lua = 0.0;
        if (!model->BSIM4v2lua1Given)
            model->BSIM4v2lua1 = 0.0;
        if (!model->BSIM4v2lubGiven)
            model->BSIM4v2lub = 0.0;
        if (!model->BSIM4v2lub1Given)
            model->BSIM4v2lub1 = 0.0;
        if (!model->BSIM4v2lucGiven)
            model->BSIM4v2luc = 0.0;
        if (!model->BSIM4v2luc1Given)
            model->BSIM4v2luc1 = 0.0;
        if (!model->BSIM4v2lu0Given)
            model->BSIM4v2lu0 = 0.0;
        if (!model->BSIM4v2luteGiven)
	    model->BSIM4v2lute = 0.0;    
        if (!model->BSIM4v2lvoffGiven)
	    model->BSIM4v2lvoff = 0.0;
        if (!model->BSIM4v2lminvGiven)
            model->BSIM4v2lminv = 0.0;
        if (!model->BSIM4v2lfproutGiven)
            model->BSIM4v2lfprout = 0.0;
        if (!model->BSIM4v2lpditsGiven)
            model->BSIM4v2lpdits = 0.0;
        if (!model->BSIM4v2lpditsdGiven)
            model->BSIM4v2lpditsd = 0.0;
        if (!model->BSIM4v2ldeltaGiven)  
            model->BSIM4v2ldelta = 0.0;
        if (!model->BSIM4v2lrdswGiven)
            model->BSIM4v2lrdsw = 0.0;
        if (!model->BSIM4v2lrdwGiven)
            model->BSIM4v2lrdw = 0.0;
        if (!model->BSIM4v2lrswGiven)
            model->BSIM4v2lrsw = 0.0;
        if (!model->BSIM4v2lprwbGiven)
            model->BSIM4v2lprwb = 0.0;
        if (!model->BSIM4v2lprwgGiven)
            model->BSIM4v2lprwg = 0.0;
        if (!model->BSIM4v2lprtGiven)
            model->BSIM4v2lprt = 0.0;
        if (!model->BSIM4v2leta0Given)
            model->BSIM4v2leta0 = 0.0;
        if (!model->BSIM4v2letabGiven)
            model->BSIM4v2letab = -0.0;
        if (!model->BSIM4v2lpclmGiven)
            model->BSIM4v2lpclm = 0.0; 
        if (!model->BSIM4v2lpdibl1Given)
            model->BSIM4v2lpdibl1 = 0.0;
        if (!model->BSIM4v2lpdibl2Given)
            model->BSIM4v2lpdibl2 = 0.0;
        if (!model->BSIM4v2lpdiblbGiven)
            model->BSIM4v2lpdiblb = 0.0;
        if (!model->BSIM4v2lpscbe1Given)
            model->BSIM4v2lpscbe1 = 0.0;
        if (!model->BSIM4v2lpscbe2Given)
            model->BSIM4v2lpscbe2 = 0.0;
        if (!model->BSIM4v2lpvagGiven)
            model->BSIM4v2lpvag = 0.0;     
        if (!model->BSIM4v2lwrGiven)  
            model->BSIM4v2lwr = 0.0;
        if (!model->BSIM4v2ldwgGiven)  
            model->BSIM4v2ldwg = 0.0;
        if (!model->BSIM4v2ldwbGiven)  
            model->BSIM4v2ldwb = 0.0;
        if (!model->BSIM4v2lb0Given)
            model->BSIM4v2lb0 = 0.0;
        if (!model->BSIM4v2lb1Given)  
            model->BSIM4v2lb1 = 0.0;
        if (!model->BSIM4v2lalpha0Given)  
            model->BSIM4v2lalpha0 = 0.0;
        if (!model->BSIM4v2lalpha1Given)
            model->BSIM4v2lalpha1 = 0.0;
        if (!model->BSIM4v2lbeta0Given)  
            model->BSIM4v2lbeta0 = 0.0;
        if (!model->BSIM4v2lagidlGiven)
            model->BSIM4v2lagidl = 0.0;
        if (!model->BSIM4v2lbgidlGiven)
            model->BSIM4v2lbgidl = 0.0;
        if (!model->BSIM4v2lcgidlGiven)
            model->BSIM4v2lcgidl = 0.0;
        if (!model->BSIM4v2legidlGiven)
            model->BSIM4v2legidl = 0.0;
        if (!model->BSIM4v2laigcGiven)
            model->BSIM4v2laigc = 0.0;
        if (!model->BSIM4v2lbigcGiven)
            model->BSIM4v2lbigc = 0.0;
        if (!model->BSIM4v2lcigcGiven)
            model->BSIM4v2lcigc = 0.0;
        if (!model->BSIM4v2laigsdGiven)
            model->BSIM4v2laigsd = 0.0;
        if (!model->BSIM4v2lbigsdGiven)
            model->BSIM4v2lbigsd = 0.0;
        if (!model->BSIM4v2lcigsdGiven)
            model->BSIM4v2lcigsd = 0.0;
        if (!model->BSIM4v2laigbaccGiven)
            model->BSIM4v2laigbacc = 0.0;
        if (!model->BSIM4v2lbigbaccGiven)
            model->BSIM4v2lbigbacc = 0.0;
        if (!model->BSIM4v2lcigbaccGiven)
            model->BSIM4v2lcigbacc = 0.0;
        if (!model->BSIM4v2laigbinvGiven)
            model->BSIM4v2laigbinv = 0.0;
        if (!model->BSIM4v2lbigbinvGiven)
            model->BSIM4v2lbigbinv = 0.0;
        if (!model->BSIM4v2lcigbinvGiven)
            model->BSIM4v2lcigbinv = 0.0;
        if (!model->BSIM4v2lnigcGiven)
            model->BSIM4v2lnigc = 0.0;
        if (!model->BSIM4v2lnigbinvGiven)
            model->BSIM4v2lnigbinv = 0.0;
        if (!model->BSIM4v2lnigbaccGiven)
            model->BSIM4v2lnigbacc = 0.0;
        if (!model->BSIM4v2lntoxGiven)
            model->BSIM4v2lntox = 0.0;
        if (!model->BSIM4v2leigbinvGiven)
            model->BSIM4v2leigbinv = 0.0;
        if (!model->BSIM4v2lpigcdGiven)
            model->BSIM4v2lpigcd = 0.0;
        if (!model->BSIM4v2lpoxedgeGiven)
            model->BSIM4v2lpoxedge = 0.0;
        if (!model->BSIM4v2lxrcrg1Given)
            model->BSIM4v2lxrcrg1 = 0.0;
        if (!model->BSIM4v2lxrcrg2Given)
            model->BSIM4v2lxrcrg2 = 0.0;
        if (!model->BSIM4v2leuGiven)
            model->BSIM4v2leu = 0.0;
        if (!model->BSIM4v2lvfbGiven)
            model->BSIM4v2lvfb = 0.0;

        if (!model->BSIM4v2lcgslGiven)  
            model->BSIM4v2lcgsl = 0.0;
        if (!model->BSIM4v2lcgdlGiven)  
            model->BSIM4v2lcgdl = 0.0;
        if (!model->BSIM4v2lckappasGiven)  
            model->BSIM4v2lckappas = 0.0;
        if (!model->BSIM4v2lckappadGiven)
            model->BSIM4v2lckappad = 0.0;
        if (!model->BSIM4v2lclcGiven)  
            model->BSIM4v2lclc = 0.0;
        if (!model->BSIM4v2lcleGiven)  
            model->BSIM4v2lcle = 0.0;
        if (!model->BSIM4v2lcfGiven)  
            model->BSIM4v2lcf = 0.0;
        if (!model->BSIM4v2lvfbcvGiven)  
            model->BSIM4v2lvfbcv = 0.0;
        if (!model->BSIM4v2lacdeGiven)
            model->BSIM4v2lacde = 0.0;
        if (!model->BSIM4v2lmoinGiven)
            model->BSIM4v2lmoin = 0.0;
        if (!model->BSIM4v2lnoffGiven)
            model->BSIM4v2lnoff = 0.0;
        if (!model->BSIM4v2lvoffcvGiven)
            model->BSIM4v2lvoffcv = 0.0;

	/* Width dependence */
        if (!model->BSIM4v2wcdscGiven)
	    model->BSIM4v2wcdsc = 0.0;
        if (!model->BSIM4v2wcdscbGiven)
	    model->BSIM4v2wcdscb = 0.0;  
	    if (!model->BSIM4v2wcdscdGiven)
	    model->BSIM4v2wcdscd = 0.0;
        if (!model->BSIM4v2wcitGiven)
	    model->BSIM4v2wcit = 0.0;
        if (!model->BSIM4v2wnfactorGiven)
	    model->BSIM4v2wnfactor = 0.0;
        if (!model->BSIM4v2wxjGiven)
            model->BSIM4v2wxj = 0.0;
        if (!model->BSIM4v2wvsatGiven)
            model->BSIM4v2wvsat = 0.0;
        if (!model->BSIM4v2watGiven)
            model->BSIM4v2wat = 0.0;
        if (!model->BSIM4v2wa0Given)
            model->BSIM4v2wa0 = 0.0; 
        if (!model->BSIM4v2wagsGiven)
            model->BSIM4v2wags = 0.0;
        if (!model->BSIM4v2wa1Given)
            model->BSIM4v2wa1 = 0.0;
        if (!model->BSIM4v2wa2Given)
            model->BSIM4v2wa2 = 0.0;
        if (!model->BSIM4v2wketaGiven)
            model->BSIM4v2wketa = 0.0;
        if (!model->BSIM4v2wnsubGiven)
            model->BSIM4v2wnsub = 0.0;
        if (!model->BSIM4v2wndepGiven)
            model->BSIM4v2wndep = 0.0;
        if (!model->BSIM4v2wnsdGiven)
            model->BSIM4v2wnsd = 0.0;
        if (!model->BSIM4v2wphinGiven)
            model->BSIM4v2wphin = 0.0;
        if (!model->BSIM4v2wngateGiven)
            model->BSIM4v2wngate = 0.0;
        if (!model->BSIM4v2wvbmGiven)
	    model->BSIM4v2wvbm = 0.0;
        if (!model->BSIM4v2wxtGiven)
	    model->BSIM4v2wxt = 0.0;
        if (!model->BSIM4v2wkt1Given)
            model->BSIM4v2wkt1 = 0.0; 
        if (!model->BSIM4v2wkt1lGiven)
            model->BSIM4v2wkt1l = 0.0;
        if (!model->BSIM4v2wkt2Given)
            model->BSIM4v2wkt2 = 0.0;
        if (!model->BSIM4v2wk3Given)
            model->BSIM4v2wk3 = 0.0;      
        if (!model->BSIM4v2wk3bGiven)
            model->BSIM4v2wk3b = 0.0;      
        if (!model->BSIM4v2ww0Given)
            model->BSIM4v2ww0 = 0.0;    
        if (!model->BSIM4v2wlpe0Given)
            model->BSIM4v2wlpe0 = 0.0;
        if (!model->BSIM4v2wlpebGiven)
            model->BSIM4v2wlpeb = model->BSIM4v2wlpe0;
        if (!model->BSIM4v2wdvtp0Given)
            model->BSIM4v2wdvtp0 = 0.0;
        if (!model->BSIM4v2wdvtp1Given)
            model->BSIM4v2wdvtp1 = 0.0;
        if (!model->BSIM4v2wdvt0Given)
            model->BSIM4v2wdvt0 = 0.0;    
        if (!model->BSIM4v2wdvt1Given)
            model->BSIM4v2wdvt1 = 0.0;      
        if (!model->BSIM4v2wdvt2Given)
            model->BSIM4v2wdvt2 = 0.0;
        if (!model->BSIM4v2wdvt0wGiven)
            model->BSIM4v2wdvt0w = 0.0;    
        if (!model->BSIM4v2wdvt1wGiven)
            model->BSIM4v2wdvt1w = 0.0;      
        if (!model->BSIM4v2wdvt2wGiven)
            model->BSIM4v2wdvt2w = 0.0;
        if (!model->BSIM4v2wdroutGiven)
            model->BSIM4v2wdrout = 0.0;     
        if (!model->BSIM4v2wdsubGiven)
            model->BSIM4v2wdsub = 0.0;
        if (!model->BSIM4v2wvth0Given)
           model->BSIM4v2wvth0 = 0.0;
        if (!model->BSIM4v2wuaGiven)
            model->BSIM4v2wua = 0.0;
        if (!model->BSIM4v2wua1Given)
            model->BSIM4v2wua1 = 0.0;
        if (!model->BSIM4v2wubGiven)
            model->BSIM4v2wub = 0.0;
        if (!model->BSIM4v2wub1Given)
            model->BSIM4v2wub1 = 0.0;
        if (!model->BSIM4v2wucGiven)
            model->BSIM4v2wuc = 0.0;
        if (!model->BSIM4v2wuc1Given)
            model->BSIM4v2wuc1 = 0.0;
        if (!model->BSIM4v2wu0Given)
            model->BSIM4v2wu0 = 0.0;
        if (!model->BSIM4v2wuteGiven)
	    model->BSIM4v2wute = 0.0;    
        if (!model->BSIM4v2wvoffGiven)
	    model->BSIM4v2wvoff = 0.0;
        if (!model->BSIM4v2wminvGiven)
            model->BSIM4v2wminv = 0.0;
        if (!model->BSIM4v2wfproutGiven)
            model->BSIM4v2wfprout = 0.0;
        if (!model->BSIM4v2wpditsGiven)
            model->BSIM4v2wpdits = 0.0;
        if (!model->BSIM4v2wpditsdGiven)
            model->BSIM4v2wpditsd = 0.0;
        if (!model->BSIM4v2wdeltaGiven)  
            model->BSIM4v2wdelta = 0.0;
        if (!model->BSIM4v2wrdswGiven)
            model->BSIM4v2wrdsw = 0.0;
        if (!model->BSIM4v2wrdwGiven)
            model->BSIM4v2wrdw = 0.0;
        if (!model->BSIM4v2wrswGiven)
            model->BSIM4v2wrsw = 0.0;
        if (!model->BSIM4v2wprwbGiven)
            model->BSIM4v2wprwb = 0.0;
        if (!model->BSIM4v2wprwgGiven)
            model->BSIM4v2wprwg = 0.0;
        if (!model->BSIM4v2wprtGiven)
            model->BSIM4v2wprt = 0.0;
        if (!model->BSIM4v2weta0Given)
            model->BSIM4v2weta0 = 0.0;
        if (!model->BSIM4v2wetabGiven)
            model->BSIM4v2wetab = 0.0;
        if (!model->BSIM4v2wpclmGiven)
            model->BSIM4v2wpclm = 0.0; 
        if (!model->BSIM4v2wpdibl1Given)
            model->BSIM4v2wpdibl1 = 0.0;
        if (!model->BSIM4v2wpdibl2Given)
            model->BSIM4v2wpdibl2 = 0.0;
        if (!model->BSIM4v2wpdiblbGiven)
            model->BSIM4v2wpdiblb = 0.0;
        if (!model->BSIM4v2wpscbe1Given)
            model->BSIM4v2wpscbe1 = 0.0;
        if (!model->BSIM4v2wpscbe2Given)
            model->BSIM4v2wpscbe2 = 0.0;
        if (!model->BSIM4v2wpvagGiven)
            model->BSIM4v2wpvag = 0.0;     
        if (!model->BSIM4v2wwrGiven)  
            model->BSIM4v2wwr = 0.0;
        if (!model->BSIM4v2wdwgGiven)  
            model->BSIM4v2wdwg = 0.0;
        if (!model->BSIM4v2wdwbGiven)  
            model->BSIM4v2wdwb = 0.0;
        if (!model->BSIM4v2wb0Given)
            model->BSIM4v2wb0 = 0.0;
        if (!model->BSIM4v2wb1Given)  
            model->BSIM4v2wb1 = 0.0;
        if (!model->BSIM4v2walpha0Given)  
            model->BSIM4v2walpha0 = 0.0;
        if (!model->BSIM4v2walpha1Given)
            model->BSIM4v2walpha1 = 0.0;
        if (!model->BSIM4v2wbeta0Given)  
            model->BSIM4v2wbeta0 = 0.0;
        if (!model->BSIM4v2wagidlGiven)
            model->BSIM4v2wagidl = 0.0;
        if (!model->BSIM4v2wbgidlGiven)
            model->BSIM4v2wbgidl = 0.0;
        if (!model->BSIM4v2wcgidlGiven)
            model->BSIM4v2wcgidl = 0.0;
        if (!model->BSIM4v2wegidlGiven)
            model->BSIM4v2wegidl = 0.0;
        if (!model->BSIM4v2waigcGiven)
            model->BSIM4v2waigc = 0.0;
        if (!model->BSIM4v2wbigcGiven)
            model->BSIM4v2wbigc = 0.0;
        if (!model->BSIM4v2wcigcGiven)
            model->BSIM4v2wcigc = 0.0;
        if (!model->BSIM4v2waigsdGiven)
            model->BSIM4v2waigsd = 0.0;
        if (!model->BSIM4v2wbigsdGiven)
            model->BSIM4v2wbigsd = 0.0;
        if (!model->BSIM4v2wcigsdGiven)
            model->BSIM4v2wcigsd = 0.0;
        if (!model->BSIM4v2waigbaccGiven)
            model->BSIM4v2waigbacc = 0.0;
        if (!model->BSIM4v2wbigbaccGiven)
            model->BSIM4v2wbigbacc = 0.0;
        if (!model->BSIM4v2wcigbaccGiven)
            model->BSIM4v2wcigbacc = 0.0;
        if (!model->BSIM4v2waigbinvGiven)
            model->BSIM4v2waigbinv = 0.0;
        if (!model->BSIM4v2wbigbinvGiven)
            model->BSIM4v2wbigbinv = 0.0;
        if (!model->BSIM4v2wcigbinvGiven)
            model->BSIM4v2wcigbinv = 0.0;
        if (!model->BSIM4v2wnigcGiven)
            model->BSIM4v2wnigc = 0.0;
        if (!model->BSIM4v2wnigbinvGiven)
            model->BSIM4v2wnigbinv = 0.0;
        if (!model->BSIM4v2wnigbaccGiven)
            model->BSIM4v2wnigbacc = 0.0;
        if (!model->BSIM4v2wntoxGiven)
            model->BSIM4v2wntox = 0.0;
        if (!model->BSIM4v2weigbinvGiven)
            model->BSIM4v2weigbinv = 0.0;
        if (!model->BSIM4v2wpigcdGiven)
            model->BSIM4v2wpigcd = 0.0;
        if (!model->BSIM4v2wpoxedgeGiven)
            model->BSIM4v2wpoxedge = 0.0;
        if (!model->BSIM4v2wxrcrg1Given)
            model->BSIM4v2wxrcrg1 = 0.0;
        if (!model->BSIM4v2wxrcrg2Given)
            model->BSIM4v2wxrcrg2 = 0.0;
        if (!model->BSIM4v2weuGiven)
            model->BSIM4v2weu = 0.0;
        if (!model->BSIM4v2wvfbGiven)
            model->BSIM4v2wvfb = 0.0;

        if (!model->BSIM4v2wcgslGiven)  
            model->BSIM4v2wcgsl = 0.0;
        if (!model->BSIM4v2wcgdlGiven)  
            model->BSIM4v2wcgdl = 0.0;
        if (!model->BSIM4v2wckappasGiven)  
            model->BSIM4v2wckappas = 0.0;
        if (!model->BSIM4v2wckappadGiven)
            model->BSIM4v2wckappad = 0.0;
        if (!model->BSIM4v2wcfGiven)  
            model->BSIM4v2wcf = 0.0;
        if (!model->BSIM4v2wclcGiven)  
            model->BSIM4v2wclc = 0.0;
        if (!model->BSIM4v2wcleGiven)  
            model->BSIM4v2wcle = 0.0;
        if (!model->BSIM4v2wvfbcvGiven)  
            model->BSIM4v2wvfbcv = 0.0;
        if (!model->BSIM4v2wacdeGiven)
            model->BSIM4v2wacde = 0.0;
        if (!model->BSIM4v2wmoinGiven)
            model->BSIM4v2wmoin = 0.0;
        if (!model->BSIM4v2wnoffGiven)
            model->BSIM4v2wnoff = 0.0;
        if (!model->BSIM4v2wvoffcvGiven)
            model->BSIM4v2wvoffcv = 0.0;

	/* Cross-term dependence */
        if (!model->BSIM4v2pcdscGiven)
	    model->BSIM4v2pcdsc = 0.0;
        if (!model->BSIM4v2pcdscbGiven)
	    model->BSIM4v2pcdscb = 0.0;   
	    if (!model->BSIM4v2pcdscdGiven)
	    model->BSIM4v2pcdscd = 0.0;
        if (!model->BSIM4v2pcitGiven)
	    model->BSIM4v2pcit = 0.0;
        if (!model->BSIM4v2pnfactorGiven)
	    model->BSIM4v2pnfactor = 0.0;
        if (!model->BSIM4v2pxjGiven)
            model->BSIM4v2pxj = 0.0;
        if (!model->BSIM4v2pvsatGiven)
            model->BSIM4v2pvsat = 0.0;
        if (!model->BSIM4v2patGiven)
            model->BSIM4v2pat = 0.0;
        if (!model->BSIM4v2pa0Given)
            model->BSIM4v2pa0 = 0.0; 
            
        if (!model->BSIM4v2pagsGiven)
            model->BSIM4v2pags = 0.0;
        if (!model->BSIM4v2pa1Given)
            model->BSIM4v2pa1 = 0.0;
        if (!model->BSIM4v2pa2Given)
            model->BSIM4v2pa2 = 0.0;
        if (!model->BSIM4v2pketaGiven)
            model->BSIM4v2pketa = 0.0;
        if (!model->BSIM4v2pnsubGiven)
            model->BSIM4v2pnsub = 0.0;
        if (!model->BSIM4v2pndepGiven)
            model->BSIM4v2pndep = 0.0;
        if (!model->BSIM4v2pnsdGiven)
            model->BSIM4v2pnsd = 0.0;
        if (!model->BSIM4v2pphinGiven)
            model->BSIM4v2pphin = 0.0;
        if (!model->BSIM4v2pngateGiven)
            model->BSIM4v2pngate = 0.0;
        if (!model->BSIM4v2pvbmGiven)
	    model->BSIM4v2pvbm = 0.0;
        if (!model->BSIM4v2pxtGiven)
	    model->BSIM4v2pxt = 0.0;
        if (!model->BSIM4v2pkt1Given)
            model->BSIM4v2pkt1 = 0.0; 
        if (!model->BSIM4v2pkt1lGiven)
            model->BSIM4v2pkt1l = 0.0;
        if (!model->BSIM4v2pkt2Given)
            model->BSIM4v2pkt2 = 0.0;
        if (!model->BSIM4v2pk3Given)
            model->BSIM4v2pk3 = 0.0;      
        if (!model->BSIM4v2pk3bGiven)
            model->BSIM4v2pk3b = 0.0;      
        if (!model->BSIM4v2pw0Given)
            model->BSIM4v2pw0 = 0.0;    
        if (!model->BSIM4v2plpe0Given)
            model->BSIM4v2plpe0 = 0.0;
        if (!model->BSIM4v2plpebGiven)
            model->BSIM4v2plpeb = model->BSIM4v2plpe0;
        if (!model->BSIM4v2pdvtp0Given)
            model->BSIM4v2pdvtp0 = 0.0;
        if (!model->BSIM4v2pdvtp1Given)
            model->BSIM4v2pdvtp1 = 0.0;
        if (!model->BSIM4v2pdvt0Given)
            model->BSIM4v2pdvt0 = 0.0;    
        if (!model->BSIM4v2pdvt1Given)
            model->BSIM4v2pdvt1 = 0.0;      
        if (!model->BSIM4v2pdvt2Given)
            model->BSIM4v2pdvt2 = 0.0;
        if (!model->BSIM4v2pdvt0wGiven)
            model->BSIM4v2pdvt0w = 0.0;    
        if (!model->BSIM4v2pdvt1wGiven)
            model->BSIM4v2pdvt1w = 0.0;      
        if (!model->BSIM4v2pdvt2wGiven)
            model->BSIM4v2pdvt2w = 0.0;
        if (!model->BSIM4v2pdroutGiven)
            model->BSIM4v2pdrout = 0.0;     
        if (!model->BSIM4v2pdsubGiven)
            model->BSIM4v2pdsub = 0.0;
        if (!model->BSIM4v2pvth0Given)
           model->BSIM4v2pvth0 = 0.0;
        if (!model->BSIM4v2puaGiven)
            model->BSIM4v2pua = 0.0;
        if (!model->BSIM4v2pua1Given)
            model->BSIM4v2pua1 = 0.0;
        if (!model->BSIM4v2pubGiven)
            model->BSIM4v2pub = 0.0;
        if (!model->BSIM4v2pub1Given)
            model->BSIM4v2pub1 = 0.0;
        if (!model->BSIM4v2pucGiven)
            model->BSIM4v2puc = 0.0;
        if (!model->BSIM4v2puc1Given)
            model->BSIM4v2puc1 = 0.0;
        if (!model->BSIM4v2pu0Given)
            model->BSIM4v2pu0 = 0.0;
        if (!model->BSIM4v2puteGiven)
	    model->BSIM4v2pute = 0.0;    
        if (!model->BSIM4v2pvoffGiven)
	    model->BSIM4v2pvoff = 0.0;
        if (!model->BSIM4v2pminvGiven)
            model->BSIM4v2pminv = 0.0;
        if (!model->BSIM4v2pfproutGiven)
            model->BSIM4v2pfprout = 0.0;
        if (!model->BSIM4v2ppditsGiven)
            model->BSIM4v2ppdits = 0.0;
        if (!model->BSIM4v2ppditsdGiven)
            model->BSIM4v2ppditsd = 0.0;
        if (!model->BSIM4v2pdeltaGiven)  
            model->BSIM4v2pdelta = 0.0;
        if (!model->BSIM4v2prdswGiven)
            model->BSIM4v2prdsw = 0.0;
        if (!model->BSIM4v2prdwGiven)
            model->BSIM4v2prdw = 0.0;
        if (!model->BSIM4v2prswGiven)
            model->BSIM4v2prsw = 0.0;
        if (!model->BSIM4v2pprwbGiven)
            model->BSIM4v2pprwb = 0.0;
        if (!model->BSIM4v2pprwgGiven)
            model->BSIM4v2pprwg = 0.0;
        if (!model->BSIM4v2pprtGiven)
            model->BSIM4v2pprt = 0.0;
        if (!model->BSIM4v2peta0Given)
            model->BSIM4v2peta0 = 0.0;
        if (!model->BSIM4v2petabGiven)
            model->BSIM4v2petab = 0.0;
        if (!model->BSIM4v2ppclmGiven)
            model->BSIM4v2ppclm = 0.0; 
        if (!model->BSIM4v2ppdibl1Given)
            model->BSIM4v2ppdibl1 = 0.0;
        if (!model->BSIM4v2ppdibl2Given)
            model->BSIM4v2ppdibl2 = 0.0;
        if (!model->BSIM4v2ppdiblbGiven)
            model->BSIM4v2ppdiblb = 0.0;
        if (!model->BSIM4v2ppscbe1Given)
            model->BSIM4v2ppscbe1 = 0.0;
        if (!model->BSIM4v2ppscbe2Given)
            model->BSIM4v2ppscbe2 = 0.0;
        if (!model->BSIM4v2ppvagGiven)
            model->BSIM4v2ppvag = 0.0;     
        if (!model->BSIM4v2pwrGiven)  
            model->BSIM4v2pwr = 0.0;
        if (!model->BSIM4v2pdwgGiven)  
            model->BSIM4v2pdwg = 0.0;
        if (!model->BSIM4v2pdwbGiven)  
            model->BSIM4v2pdwb = 0.0;
        if (!model->BSIM4v2pb0Given)
            model->BSIM4v2pb0 = 0.0;
        if (!model->BSIM4v2pb1Given)  
            model->BSIM4v2pb1 = 0.0;
        if (!model->BSIM4v2palpha0Given)  
            model->BSIM4v2palpha0 = 0.0;
        if (!model->BSIM4v2palpha1Given)
            model->BSIM4v2palpha1 = 0.0;
        if (!model->BSIM4v2pbeta0Given)  
            model->BSIM4v2pbeta0 = 0.0;
        if (!model->BSIM4v2pagidlGiven)
            model->BSIM4v2pagidl = 0.0;
        if (!model->BSIM4v2pbgidlGiven)
            model->BSIM4v2pbgidl = 0.0;
        if (!model->BSIM4v2pcgidlGiven)
            model->BSIM4v2pcgidl = 0.0;
        if (!model->BSIM4v2pegidlGiven)
            model->BSIM4v2pegidl = 0.0;
        if (!model->BSIM4v2paigcGiven)
            model->BSIM4v2paigc = 0.0;
        if (!model->BSIM4v2pbigcGiven)
            model->BSIM4v2pbigc = 0.0;
        if (!model->BSIM4v2pcigcGiven)
            model->BSIM4v2pcigc = 0.0;
        if (!model->BSIM4v2paigsdGiven)
            model->BSIM4v2paigsd = 0.0;
        if (!model->BSIM4v2pbigsdGiven)
            model->BSIM4v2pbigsd = 0.0;
        if (!model->BSIM4v2pcigsdGiven)
            model->BSIM4v2pcigsd = 0.0;
        if (!model->BSIM4v2paigbaccGiven)
            model->BSIM4v2paigbacc = 0.0;
        if (!model->BSIM4v2pbigbaccGiven)
            model->BSIM4v2pbigbacc = 0.0;
        if (!model->BSIM4v2pcigbaccGiven)
            model->BSIM4v2pcigbacc = 0.0;
        if (!model->BSIM4v2paigbinvGiven)
            model->BSIM4v2paigbinv = 0.0;
        if (!model->BSIM4v2pbigbinvGiven)
            model->BSIM4v2pbigbinv = 0.0;
        if (!model->BSIM4v2pcigbinvGiven)
            model->BSIM4v2pcigbinv = 0.0;
        if (!model->BSIM4v2pnigcGiven)
            model->BSIM4v2pnigc = 0.0;
        if (!model->BSIM4v2pnigbinvGiven)
            model->BSIM4v2pnigbinv = 0.0;
        if (!model->BSIM4v2pnigbaccGiven)
            model->BSIM4v2pnigbacc = 0.0;
        if (!model->BSIM4v2pntoxGiven)
            model->BSIM4v2pntox = 0.0;
        if (!model->BSIM4v2peigbinvGiven)
            model->BSIM4v2peigbinv = 0.0;
        if (!model->BSIM4v2ppigcdGiven)
            model->BSIM4v2ppigcd = 0.0;
        if (!model->BSIM4v2ppoxedgeGiven)
            model->BSIM4v2ppoxedge = 0.0;
        if (!model->BSIM4v2pxrcrg1Given)
            model->BSIM4v2pxrcrg1 = 0.0;
        if (!model->BSIM4v2pxrcrg2Given)
            model->BSIM4v2pxrcrg2 = 0.0;
        if (!model->BSIM4v2peuGiven)
            model->BSIM4v2peu = 0.0;
        if (!model->BSIM4v2pvfbGiven)
            model->BSIM4v2pvfb = 0.0;

        if (!model->BSIM4v2pcgslGiven)  
            model->BSIM4v2pcgsl = 0.0;
        if (!model->BSIM4v2pcgdlGiven)  
            model->BSIM4v2pcgdl = 0.0;
        if (!model->BSIM4v2pckappasGiven)  
            model->BSIM4v2pckappas = 0.0;
        if (!model->BSIM4v2pckappadGiven)
            model->BSIM4v2pckappad = 0.0;
        if (!model->BSIM4v2pcfGiven)  
            model->BSIM4v2pcf = 0.0;
        if (!model->BSIM4v2pclcGiven)  
            model->BSIM4v2pclc = 0.0;
        if (!model->BSIM4v2pcleGiven)  
            model->BSIM4v2pcle = 0.0;
        if (!model->BSIM4v2pvfbcvGiven)  
            model->BSIM4v2pvfbcv = 0.0;
        if (!model->BSIM4v2pacdeGiven)
            model->BSIM4v2pacde = 0.0;
        if (!model->BSIM4v2pmoinGiven)
            model->BSIM4v2pmoin = 0.0;
        if (!model->BSIM4v2pnoffGiven)
            model->BSIM4v2pnoff = 0.0;
        if (!model->BSIM4v2pvoffcvGiven)
            model->BSIM4v2pvoffcv = 0.0;

        /* unit degree celcius */
        if (!model->BSIM4v2tnomGiven)  
	    model->BSIM4v2tnom = ckt->CKTnomTemp; 
        if (!model->BSIM4v2LintGiven)  
           model->BSIM4v2Lint = 0.0;
        if (!model->BSIM4v2LlGiven)  
           model->BSIM4v2Ll = 0.0;
        if (!model->BSIM4v2LlcGiven)
           model->BSIM4v2Llc = model->BSIM4v2Ll;
        if (!model->BSIM4v2LlnGiven)  
           model->BSIM4v2Lln = 1.0;
        if (!model->BSIM4v2LwGiven)  
           model->BSIM4v2Lw = 0.0;
        if (!model->BSIM4v2LwcGiven)
           model->BSIM4v2Lwc = model->BSIM4v2Lw;
        if (!model->BSIM4v2LwnGiven)  
           model->BSIM4v2Lwn = 1.0;
        if (!model->BSIM4v2LwlGiven)  
           model->BSIM4v2Lwl = 0.0;
        if (!model->BSIM4v2LwlcGiven)
           model->BSIM4v2Lwlc = model->BSIM4v2Lwl;
        if (!model->BSIM4v2LminGiven)  
           model->BSIM4v2Lmin = 0.0;
        if (!model->BSIM4v2LmaxGiven)  
           model->BSIM4v2Lmax = 1.0;
        if (!model->BSIM4v2WintGiven)  
           model->BSIM4v2Wint = 0.0;
        if (!model->BSIM4v2WlGiven)  
           model->BSIM4v2Wl = 0.0;
        if (!model->BSIM4v2WlcGiven)
           model->BSIM4v2Wlc = model->BSIM4v2Wl;
        if (!model->BSIM4v2WlnGiven)  
           model->BSIM4v2Wln = 1.0;
        if (!model->BSIM4v2WwGiven)  
           model->BSIM4v2Ww = 0.0;
        if (!model->BSIM4v2WwcGiven)
           model->BSIM4v2Wwc = model->BSIM4v2Ww;
        if (!model->BSIM4v2WwnGiven)  
           model->BSIM4v2Wwn = 1.0;
        if (!model->BSIM4v2WwlGiven)  
           model->BSIM4v2Wwl = 0.0;
        if (!model->BSIM4v2WwlcGiven)
           model->BSIM4v2Wwlc = model->BSIM4v2Wwl;
        if (!model->BSIM4v2WminGiven)  
           model->BSIM4v2Wmin = 0.0;
        if (!model->BSIM4v2WmaxGiven)  
           model->BSIM4v2Wmax = 1.0;
        if (!model->BSIM4v2dwcGiven)  
           model->BSIM4v2dwc = model->BSIM4v2Wint;
        if (!model->BSIM4v2dlcGiven)  
           model->BSIM4v2dlc = model->BSIM4v2Lint;
        if (!model->BSIM4v2xlGiven)  
           model->BSIM4v2xl = 0.0;
        if (!model->BSIM4v2xwGiven)  
           model->BSIM4v2xw = 0.0;
        if (!model->BSIM4v2dlcigGiven)
           model->BSIM4v2dlcig = model->BSIM4v2Lint;
        if (!model->BSIM4v2dwjGiven)
           model->BSIM4v2dwj = model->BSIM4v2dwc;
	if (!model->BSIM4v2cfGiven)
           model->BSIM4v2cf = 2.0 * model->BSIM4v2epsrox * EPS0 / PI
		          * log(1.0 + 0.4e-6 / model->BSIM4v2toxe);

        if (!model->BSIM4v2xpartGiven)
            model->BSIM4v2xpart = 0.0;
        if (!model->BSIM4v2sheetResistanceGiven)
            model->BSIM4v2sheetResistance = 0.0;

        if (!model->BSIM4v2SunitAreaJctCapGiven)
            model->BSIM4v2SunitAreaJctCap = 5.0E-4;
        if (!model->BSIM4v2DunitAreaJctCapGiven)
            model->BSIM4v2DunitAreaJctCap = model->BSIM4v2SunitAreaJctCap;
        if (!model->BSIM4v2SunitLengthSidewallJctCapGiven)
            model->BSIM4v2SunitLengthSidewallJctCap = 5.0E-10;
        if (!model->BSIM4v2DunitLengthSidewallJctCapGiven)
            model->BSIM4v2DunitLengthSidewallJctCap = model->BSIM4v2SunitLengthSidewallJctCap;
        if (!model->BSIM4v2SunitLengthGateSidewallJctCapGiven)
            model->BSIM4v2SunitLengthGateSidewallJctCap = model->BSIM4v2SunitLengthSidewallJctCap ;
        if (!model->BSIM4v2DunitLengthGateSidewallJctCapGiven)
            model->BSIM4v2DunitLengthGateSidewallJctCap = model->BSIM4v2SunitLengthGateSidewallJctCap;
        if (!model->BSIM4v2SjctSatCurDensityGiven)
            model->BSIM4v2SjctSatCurDensity = 1.0E-4;
        if (!model->BSIM4v2DjctSatCurDensityGiven)
            model->BSIM4v2DjctSatCurDensity = model->BSIM4v2SjctSatCurDensity;
        if (!model->BSIM4v2SjctSidewallSatCurDensityGiven)
            model->BSIM4v2SjctSidewallSatCurDensity = 0.0;
        if (!model->BSIM4v2DjctSidewallSatCurDensityGiven)
            model->BSIM4v2DjctSidewallSatCurDensity = model->BSIM4v2SjctSidewallSatCurDensity;
        if (!model->BSIM4v2SjctGateSidewallSatCurDensityGiven)
            model->BSIM4v2SjctGateSidewallSatCurDensity = 0.0;
        if (!model->BSIM4v2DjctGateSidewallSatCurDensityGiven)
            model->BSIM4v2DjctGateSidewallSatCurDensity = model->BSIM4v2SjctGateSidewallSatCurDensity;
        if (!model->BSIM4v2SbulkJctPotentialGiven)
            model->BSIM4v2SbulkJctPotential = 1.0;
        if (!model->BSIM4v2DbulkJctPotentialGiven)
            model->BSIM4v2DbulkJctPotential = model->BSIM4v2SbulkJctPotential;
        if (!model->BSIM4v2SsidewallJctPotentialGiven)
            model->BSIM4v2SsidewallJctPotential = 1.0;
        if (!model->BSIM4v2DsidewallJctPotentialGiven)
            model->BSIM4v2DsidewallJctPotential = model->BSIM4v2SsidewallJctPotential;
        if (!model->BSIM4v2SGatesidewallJctPotentialGiven)
            model->BSIM4v2SGatesidewallJctPotential = model->BSIM4v2SsidewallJctPotential;
        if (!model->BSIM4v2DGatesidewallJctPotentialGiven)
            model->BSIM4v2DGatesidewallJctPotential = model->BSIM4v2SGatesidewallJctPotential;
        if (!model->BSIM4v2SbulkJctBotGradingCoeffGiven)
            model->BSIM4v2SbulkJctBotGradingCoeff = 0.5;
        if (!model->BSIM4v2DbulkJctBotGradingCoeffGiven)
            model->BSIM4v2DbulkJctBotGradingCoeff = model->BSIM4v2SbulkJctBotGradingCoeff;
        if (!model->BSIM4v2SbulkJctSideGradingCoeffGiven)
            model->BSIM4v2SbulkJctSideGradingCoeff = 0.33;
        if (!model->BSIM4v2DbulkJctSideGradingCoeffGiven)
            model->BSIM4v2DbulkJctSideGradingCoeff = model->BSIM4v2SbulkJctSideGradingCoeff;
        if (!model->BSIM4v2SbulkJctGateSideGradingCoeffGiven)
            model->BSIM4v2SbulkJctGateSideGradingCoeff = model->BSIM4v2SbulkJctSideGradingCoeff;
        if (!model->BSIM4v2DbulkJctGateSideGradingCoeffGiven)
            model->BSIM4v2DbulkJctGateSideGradingCoeff = model->BSIM4v2SbulkJctGateSideGradingCoeff;
        if (!model->BSIM4v2SjctEmissionCoeffGiven)
            model->BSIM4v2SjctEmissionCoeff = 1.0;
        if (!model->BSIM4v2DjctEmissionCoeffGiven)
            model->BSIM4v2DjctEmissionCoeff = model->BSIM4v2SjctEmissionCoeff;
        if (!model->BSIM4v2SjctTempExponentGiven)
            model->BSIM4v2SjctTempExponent = 3.0;
        if (!model->BSIM4v2DjctTempExponentGiven)
            model->BSIM4v2DjctTempExponent = model->BSIM4v2SjctTempExponent;

        if (!model->BSIM4v2oxideTrapDensityAGiven)
	{   if (model->BSIM4v2type == NMOS)
                model->BSIM4v2oxideTrapDensityA = 6.25e41;
            else
                model->BSIM4v2oxideTrapDensityA= 6.188e40;
	}
        if (!model->BSIM4v2oxideTrapDensityBGiven)
	{   if (model->BSIM4v2type == NMOS)
                model->BSIM4v2oxideTrapDensityB = 3.125e26;
            else
                model->BSIM4v2oxideTrapDensityB = 1.5e25;
	}
        if (!model->BSIM4v2oxideTrapDensityCGiven)
            model->BSIM4v2oxideTrapDensityC = 8.75e9;
        if (!model->BSIM4v2emGiven)
            model->BSIM4v2em = 4.1e7; /* V/m */
        if (!model->BSIM4v2efGiven)
            model->BSIM4v2ef = 1.0;
        if (!model->BSIM4v2afGiven)
            model->BSIM4v2af = 1.0;
        if (!model->BSIM4v2kfGiven)
            model->BSIM4v2kf = 0.0;
        DMCGeff = model->BSIM4v2dmcg - model->BSIM4v2dmcgt;
        DMCIeff = model->BSIM4v2dmci;
        DMDGeff = model->BSIM4v2dmdg - model->BSIM4v2dmcgt;

	/*
         * End processing models and begin to loop
         * through all the instances of the model
         */

        for (here = model->BSIM4v2instances; here != NULL ;
             here=here->BSIM4v2nextInstance)      
	{   
	     if (here->BSIM4v2owner == ARCHme) {
	    /* allocate a chunk of the state vector */
            here->BSIM4v2states = *states;
            *states += BSIM4v2numStates;
            }
            /* perform the parameter defaulting */
            if (!here->BSIM4v2lGiven)
                here->BSIM4v2l = 5.0e-6;
            if (!here->BSIM4v2wGiven)
                here->BSIM4v2w = 5.0e-6;
            if (!here->BSIM4v2mGiven)
                here->BSIM4v2m = 1.0;
            if (!here->BSIM4v2nfGiven)
                here->BSIM4v2nf = 1.0;
            if (!here->BSIM4v2minGiven)
                here->BSIM4v2min = 0; /* integer */
            if (!here->BSIM4v2icVDSGiven)
                here->BSIM4v2icVDS = 0.0;
            if (!here->BSIM4v2icVGSGiven)
                here->BSIM4v2icVGS = 0.0;
            if (!here->BSIM4v2icVBSGiven)
                here->BSIM4v2icVBS = 0.0;
            if (!here->BSIM4v2drainAreaGiven)
                here->BSIM4v2drainArea = 0.0;
            if (!here->BSIM4v2drainPerimeterGiven)
                here->BSIM4v2drainPerimeter = 0.0;
            if (!here->BSIM4v2drainSquaresGiven)
                here->BSIM4v2drainSquares = 1.0;
            if (!here->BSIM4v2sourceAreaGiven)
                here->BSIM4v2sourceArea = 0.0;
            if (!here->BSIM4v2sourcePerimeterGiven)
                here->BSIM4v2sourcePerimeter = 0.0;
            if (!here->BSIM4v2sourceSquaresGiven)
                here->BSIM4v2sourceSquares = 1.0;

            if (!here->BSIM4v2rbdbGiven)
                here->BSIM4v2rbdb = model->BSIM4v2rbdb; /* in ohm */
            if (!here->BSIM4v2rbsbGiven)
                here->BSIM4v2rbsb = model->BSIM4v2rbsb;
            if (!here->BSIM4v2rbpbGiven)
                here->BSIM4v2rbpb = model->BSIM4v2rbpb;
            if (!here->BSIM4v2rbpsGiven)
                here->BSIM4v2rbps = model->BSIM4v2rbps;
            if (!here->BSIM4v2rbpdGiven)
                here->BSIM4v2rbpd = model->BSIM4v2rbpd;

                    
            /* Process instance model selectors, some
             * may override their global counterparts
	     */
            if (!here->BSIM4v2rbodyModGiven)
                here->BSIM4v2rbodyMod = model->BSIM4v2rbodyMod;
            else if ((here->BSIM4v2rbodyMod != 0) && (here->BSIM4v2rbodyMod != 1))
            {   here->BSIM4v2rbodyMod = model->BSIM4v2rbodyMod;
                printf("Warning: rbodyMod has been set to its global value %d.\n",
	        model->BSIM4v2rbodyMod);
            }

            if (!here->BSIM4v2rgateModGiven)
                here->BSIM4v2rgateMod = model->BSIM4v2rgateMod;
            else if ((here->BSIM4v2rgateMod != 0) && (here->BSIM4v2rgateMod != 1)
	        && (here->BSIM4v2rgateMod != 2) && (here->BSIM4v2rgateMod != 3))
            {   here->BSIM4v2rgateMod = model->BSIM4v2rgateMod;
                printf("Warning: rgateMod has been set to its global value %d.\n",
                model->BSIM4v2rgateMod);
            }

            if (!here->BSIM4v2geoModGiven)
                here->BSIM4v2geoMod = model->BSIM4v2geoMod;
            if (!here->BSIM4v2rgeoModGiven)
                here->BSIM4v2rgeoMod = 0.0;
            if (!here->BSIM4v2trnqsModGiven)
                here->BSIM4v2trnqsMod = model->BSIM4v2trnqsMod;
            else if ((here->BSIM4v2trnqsMod != 0) && (here->BSIM4v2trnqsMod != 1))
            {   here->BSIM4v2trnqsMod = model->BSIM4v2trnqsMod;
                printf("Warning: trnqsMod has been set to its global value %d.\n",
                model->BSIM4v2trnqsMod);
            }

            if (!here->BSIM4v2acnqsModGiven)
                here->BSIM4v2acnqsMod = model->BSIM4v2acnqsMod;
            else if ((here->BSIM4v2acnqsMod != 0) && (here->BSIM4v2acnqsMod != 1))
            {   here->BSIM4v2acnqsMod = model->BSIM4v2acnqsMod;
                printf("Warning: acnqsMod has been set to its global value %d.\n",
                model->BSIM4v2acnqsMod);
            }

            /* process drain series resistance */
            createNode = 0;
            if ( (model->BSIM4v2rdsMod != 0)
                            || (model->BSIM4v2tnoiMod != 0 && noiseAnalGiven))
            {
               createNode = 1;
            } else if (model->BSIM4v2sheetResistance > 0)
            {
                     if (here->BSIM4v2drainSquaresGiven
                                       && here->BSIM4v2drainSquares > 0)
                     {
                          createNode = 1;
                     } else if (!here->BSIM4v2drainSquaresGiven
                                       && (here->BSIM4v2rgeoMod != 0))
                     {
                          BSIM4v2RdseffGeo(here->BSIM4v2nf, here->BSIM4v2geoMod,
                                  here->BSIM4v2rgeoMod, here->BSIM4v2min,
                                  here->BSIM4v2w, model->BSIM4v2sheetResistance,
                                  DMCGeff, DMCIeff, DMDGeff, 0, &Rtot);
                          if(Rtot > 0)
                             createNode = 1;
                     }
            }
            if ( createNode != 0  && (here->BSIM4v2dNodePrime == 0))
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v2name,"drain");
                if(error) return(error);
                here->BSIM4v2dNodePrime = tmp->number;
            }
            else
            {   here->BSIM4v2dNodePrime = here->BSIM4v2dNode;
            }
            
            /* process source series resistance */
            createNode = 0;
            if ( (model->BSIM4v2rdsMod != 0)
                            || (model->BSIM4v2tnoiMod != 0 && noiseAnalGiven))
            {
               createNode = 1;
            } else if (model->BSIM4v2sheetResistance > 0)
            {
                     if (here->BSIM4v2sourceSquaresGiven
                                        && here->BSIM4v2sourceSquares > 0)
                     {
                          createNode = 1;
                     } else if (!here->BSIM4v2sourceSquaresGiven
                                        && (here->BSIM4v2rgeoMod != 0))
                     {
                          BSIM4v2RdseffGeo(here->BSIM4v2nf, here->BSIM4v2geoMod,
                                  here->BSIM4v2rgeoMod, here->BSIM4v2min,
                                  here->BSIM4v2w, model->BSIM4v2sheetResistance,
                                  DMCGeff, DMCIeff, DMDGeff, 1, &Rtot);
                          if(Rtot > 0)
                             createNode = 1;
                     }
            }
            if ( createNode != 0  && here->BSIM4v2sNodePrime == 0)
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v2name,"source");
                if(error) return(error);
                here->BSIM4v2sNodePrime = tmp->number;
            }
            else
                here->BSIM4v2sNodePrime = here->BSIM4v2sNode;

            if ((here->BSIM4v2rgateMod > 0) && (here->BSIM4v2gNodePrime == 0))
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v2name,"gate");
                if(error) return(error);
                   here->BSIM4v2gNodePrime = tmp->number;
                if (here->BSIM4v2rgateMod == 1)
                {   if (model->BSIM4v2rshg <= 0.0)
	                printf("Warning: rshg should be positive for rgateMod = 1.\n");
		}
                else if (here->BSIM4v2rgateMod == 2)
                {   if (model->BSIM4v2rshg <= 0.0)
                        printf("Warning: rshg <= 0.0 for rgateMod = 2!!!\n");
                    else if (model->BSIM4v2xrcrg1 <= 0.0)
                        printf("Warning: xrcrg1 <= 0.0 for rgateMod = 2!!!\n");
                }
            }
            else
                here->BSIM4v2gNodePrime = here->BSIM4v2gNodeExt;

            if ((here->BSIM4v2rgateMod == 3) && (here->BSIM4v2gNodeMid == 0))
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v2name,"midgate");
                if(error) return(error);
                   here->BSIM4v2gNodeMid = tmp->number;
                if (model->BSIM4v2rshg <= 0.0)
                    printf("Warning: rshg should be positive for rgateMod = 3.\n");
                else if (model->BSIM4v2xrcrg1 <= 0.0)
                    printf("Warning: xrcrg1 should be positive for rgateMod = 3.\n");
            }
            else
                here->BSIM4v2gNodeMid = here->BSIM4v2gNodeExt;
            

            /* internal body nodes for body resistance model */
            if (here->BSIM4v2rbodyMod)
            {   if (here->BSIM4v2dbNode == 0)
		{   error = CKTmkVolt(ckt,&tmp,here->BSIM4v2name,"dbody");
                    if(error) return(error);
                    here->BSIM4v2dbNode = tmp->number;
		}
		if (here->BSIM4v2bNodePrime == 0)
                {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v2name,"body");
                    if(error) return(error);
                    here->BSIM4v2bNodePrime = tmp->number;
                }
		if (here->BSIM4v2sbNode == 0)
                {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v2name,"sbody");
                    if(error) return(error);
                    here->BSIM4v2sbNode = tmp->number;
                }
            }
	    else
	        here->BSIM4v2dbNode = here->BSIM4v2bNodePrime = here->BSIM4v2sbNode
				  = here->BSIM4v2bNode;

            /* NQS node */
            if ((here->BSIM4v2trnqsMod) && (here->BSIM4v2qNode == 0)) 
	    {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v2name,"charge");
                if(error) return(error);
                here->BSIM4v2qNode = tmp->number;
            }
	    else 
	        here->BSIM4v2qNode = 0;

/* set Sparse Matrix Pointers 
 * macro to make elements with built-in out-of-memory test */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==(double *)NULL){\
    return(E_NOMEM);\
}

            TSTALLOC(BSIM4v2DPbpPtr, BSIM4v2dNodePrime, BSIM4v2bNodePrime)
            TSTALLOC(BSIM4v2GPbpPtr, BSIM4v2gNodePrime, BSIM4v2bNodePrime)
            TSTALLOC(BSIM4v2SPbpPtr, BSIM4v2sNodePrime, BSIM4v2bNodePrime)

            TSTALLOC(BSIM4v2BPdpPtr, BSIM4v2bNodePrime, BSIM4v2dNodePrime)
            TSTALLOC(BSIM4v2BPgpPtr, BSIM4v2bNodePrime, BSIM4v2gNodePrime)
            TSTALLOC(BSIM4v2BPspPtr, BSIM4v2bNodePrime, BSIM4v2sNodePrime)
            TSTALLOC(BSIM4v2BPbpPtr, BSIM4v2bNodePrime, BSIM4v2bNodePrime)

            TSTALLOC(BSIM4v2DdPtr, BSIM4v2dNode, BSIM4v2dNode)
            TSTALLOC(BSIM4v2GPgpPtr, BSIM4v2gNodePrime, BSIM4v2gNodePrime)
            TSTALLOC(BSIM4v2SsPtr, BSIM4v2sNode, BSIM4v2sNode)
            TSTALLOC(BSIM4v2DPdpPtr, BSIM4v2dNodePrime, BSIM4v2dNodePrime)
            TSTALLOC(BSIM4v2SPspPtr, BSIM4v2sNodePrime, BSIM4v2sNodePrime)
            TSTALLOC(BSIM4v2DdpPtr, BSIM4v2dNode, BSIM4v2dNodePrime)
            TSTALLOC(BSIM4v2GPdpPtr, BSIM4v2gNodePrime, BSIM4v2dNodePrime)
            TSTALLOC(BSIM4v2GPspPtr, BSIM4v2gNodePrime, BSIM4v2sNodePrime)
            TSTALLOC(BSIM4v2SspPtr, BSIM4v2sNode, BSIM4v2sNodePrime)
            TSTALLOC(BSIM4v2DPspPtr, BSIM4v2dNodePrime, BSIM4v2sNodePrime)
            TSTALLOC(BSIM4v2DPdPtr, BSIM4v2dNodePrime, BSIM4v2dNode)
            TSTALLOC(BSIM4v2DPgpPtr, BSIM4v2dNodePrime, BSIM4v2gNodePrime)
            TSTALLOC(BSIM4v2SPgpPtr, BSIM4v2sNodePrime, BSIM4v2gNodePrime)
            TSTALLOC(BSIM4v2SPsPtr, BSIM4v2sNodePrime, BSIM4v2sNode)
            TSTALLOC(BSIM4v2SPdpPtr, BSIM4v2sNodePrime, BSIM4v2dNodePrime)

            TSTALLOC(BSIM4v2QqPtr, BSIM4v2qNode, BSIM4v2qNode)
            TSTALLOC(BSIM4v2QbpPtr, BSIM4v2qNode, BSIM4v2bNodePrime) 
            TSTALLOC(BSIM4v2QdpPtr, BSIM4v2qNode, BSIM4v2dNodePrime)
            TSTALLOC(BSIM4v2QspPtr, BSIM4v2qNode, BSIM4v2sNodePrime)
            TSTALLOC(BSIM4v2QgpPtr, BSIM4v2qNode, BSIM4v2gNodePrime)
            TSTALLOC(BSIM4v2DPqPtr, BSIM4v2dNodePrime, BSIM4v2qNode)
            TSTALLOC(BSIM4v2SPqPtr, BSIM4v2sNodePrime, BSIM4v2qNode)
            TSTALLOC(BSIM4v2GPqPtr, BSIM4v2gNodePrime, BSIM4v2qNode)

            if (here->BSIM4v2rgateMod != 0)
            {   TSTALLOC(BSIM4v2GEgePtr, BSIM4v2gNodeExt, BSIM4v2gNodeExt)
                TSTALLOC(BSIM4v2GEgpPtr, BSIM4v2gNodeExt, BSIM4v2gNodePrime)
                TSTALLOC(BSIM4v2GPgePtr, BSIM4v2gNodePrime, BSIM4v2gNodeExt)
		TSTALLOC(BSIM4v2GEdpPtr, BSIM4v2gNodeExt, BSIM4v2dNodePrime)
		TSTALLOC(BSIM4v2GEspPtr, BSIM4v2gNodeExt, BSIM4v2sNodePrime)
		TSTALLOC(BSIM4v2GEbpPtr, BSIM4v2gNodeExt, BSIM4v2bNodePrime)

                TSTALLOC(BSIM4v2GMdpPtr, BSIM4v2gNodeMid, BSIM4v2dNodePrime)
                TSTALLOC(BSIM4v2GMgpPtr, BSIM4v2gNodeMid, BSIM4v2gNodePrime)
                TSTALLOC(BSIM4v2GMgmPtr, BSIM4v2gNodeMid, BSIM4v2gNodeMid)
                TSTALLOC(BSIM4v2GMgePtr, BSIM4v2gNodeMid, BSIM4v2gNodeExt)
                TSTALLOC(BSIM4v2GMspPtr, BSIM4v2gNodeMid, BSIM4v2sNodePrime)
                TSTALLOC(BSIM4v2GMbpPtr, BSIM4v2gNodeMid, BSIM4v2bNodePrime)
                TSTALLOC(BSIM4v2DPgmPtr, BSIM4v2dNodePrime, BSIM4v2gNodeMid)
                TSTALLOC(BSIM4v2GPgmPtr, BSIM4v2gNodePrime, BSIM4v2gNodeMid)
                TSTALLOC(BSIM4v2GEgmPtr, BSIM4v2gNodeExt, BSIM4v2gNodeMid)
                TSTALLOC(BSIM4v2SPgmPtr, BSIM4v2sNodePrime, BSIM4v2gNodeMid)
                TSTALLOC(BSIM4v2BPgmPtr, BSIM4v2bNodePrime, BSIM4v2gNodeMid)
            }	

            if (here->BSIM4v2rbodyMod)
            {   TSTALLOC(BSIM4v2DPdbPtr, BSIM4v2dNodePrime, BSIM4v2dbNode)
                TSTALLOC(BSIM4v2SPsbPtr, BSIM4v2sNodePrime, BSIM4v2sbNode)

                TSTALLOC(BSIM4v2DBdpPtr, BSIM4v2dbNode, BSIM4v2dNodePrime)
                TSTALLOC(BSIM4v2DBdbPtr, BSIM4v2dbNode, BSIM4v2dbNode)
                TSTALLOC(BSIM4v2DBbpPtr, BSIM4v2dbNode, BSIM4v2bNodePrime)
                TSTALLOC(BSIM4v2DBbPtr, BSIM4v2dbNode, BSIM4v2bNode)

                TSTALLOC(BSIM4v2BPdbPtr, BSIM4v2bNodePrime, BSIM4v2dbNode)
                TSTALLOC(BSIM4v2BPbPtr, BSIM4v2bNodePrime, BSIM4v2bNode)
                TSTALLOC(BSIM4v2BPsbPtr, BSIM4v2bNodePrime, BSIM4v2sbNode)

                TSTALLOC(BSIM4v2SBspPtr, BSIM4v2sbNode, BSIM4v2sNodePrime)
                TSTALLOC(BSIM4v2SBbpPtr, BSIM4v2sbNode, BSIM4v2bNodePrime)
                TSTALLOC(BSIM4v2SBbPtr, BSIM4v2sbNode, BSIM4v2bNode)
                TSTALLOC(BSIM4v2SBsbPtr, BSIM4v2sbNode, BSIM4v2sbNode)

                TSTALLOC(BSIM4v2BdbPtr, BSIM4v2bNode, BSIM4v2dbNode)
                TSTALLOC(BSIM4v2BbpPtr, BSIM4v2bNode, BSIM4v2bNodePrime)
                TSTALLOC(BSIM4v2BsbPtr, BSIM4v2bNode, BSIM4v2sbNode)
	        TSTALLOC(BSIM4v2BbPtr, BSIM4v2bNode, BSIM4v2bNode)
	    }

            if (model->BSIM4v2rdsMod)
            {   TSTALLOC(BSIM4v2DgpPtr, BSIM4v2dNode, BSIM4v2gNodePrime)
		TSTALLOC(BSIM4v2DspPtr, BSIM4v2dNode, BSIM4v2sNodePrime)
                TSTALLOC(BSIM4v2DbpPtr, BSIM4v2dNode, BSIM4v2bNodePrime)
                TSTALLOC(BSIM4v2SdpPtr, BSIM4v2sNode, BSIM4v2dNodePrime)
                TSTALLOC(BSIM4v2SgpPtr, BSIM4v2sNode, BSIM4v2gNodePrime)
                TSTALLOC(BSIM4v2SbpPtr, BSIM4v2sNode, BSIM4v2bNodePrime)
            }
        }
    }
    return(OK);
}  

int
BSIM4v2unsetup(
    GENmodel *inModel,
    CKTcircuit *ckt)
{

    BSIM4v2model *model;
    BSIM4v2instance *here;

    for (model = (BSIM4v2model *)inModel; model != NULL;
            model = model->BSIM4v2nextModel)
    {
        for (here = model->BSIM4v2instances; here != NULL;
                here=here->BSIM4v2nextInstance)
        {
            if (here->BSIM4v2dNodePrime
                    && here->BSIM4v2dNodePrime != here->BSIM4v2dNode)
            {
                CKTdltNNum(ckt, here->BSIM4v2dNodePrime);
                here->BSIM4v2dNodePrime = 0;
            }
            if (here->BSIM4v2sNodePrime
                    && here->BSIM4v2sNodePrime != here->BSIM4v2sNode)
            {
                CKTdltNNum(ckt, here->BSIM4v2sNodePrime);
                here->BSIM4v2sNodePrime = 0;
            }
        }
    }
    return OK;
}
