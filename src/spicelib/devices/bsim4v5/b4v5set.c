/**** BSIM4.5.0 Released by Xuemei (Jane) Xi 07/29/2005 ****/

/**********
 * Copyright 2005 Regents of the University of California. All rights reserved.
 * File: b4set.c of BSIM4.5.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 * Modified by Xuemei Xi, 03/04/2004.
 * Modified by Xuemei Xi, Mohan Dunga, 07/29/2005.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/jobdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "bsim4v5def.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#ifdef USE_OMP
#include "ngspice/cpextern.h"
#endif

#define MAX_EXP 5.834617425e14
#define MIN_EXP 1.713908431e-15
#define EXP_THRESHOLD 34.0
#define EPS0 8.85418e-12
#define EPSSI 1.03594e-10
#define PI 3.141592654
#define Charge_q 1.60219e-19


int
BSIM4v5setup(
SMPmatrix *matrix,
GENmodel *inModel,
CKTcircuit *ckt,
int *states)
{
BSIM4v5model *model = (BSIM4v5model*)inModel;
BSIM4v5instance *here;
int error;
CKTnode *tmp;
int    noiseAnalGiven = 0, createNode;  /* Criteria for new node creation */
double Rtot, DMCGeff, DMCIeff, DMDGeff;
JOB   *job;

#ifdef USE_OMP
int idx, InstCount;
BSIM4v5instance **InstArray;
#endif

    /* Search for a noise analysis request */
    for (job = ft_curckt->ci_curTask->jobs; job; job = job->JOBnextJob) {
        if(strcmp(job->JOBname,"Noise Analysis")==0) {
            noiseAnalGiven = 1;
            break;
        }
    }

    /*  loop through all the BSIM4v5 device models */
    for( ; model != NULL; model = BSIM4v5nextModel(model))
    {   /* process defaults of model parameters */
        if (!model->BSIM4v5typeGiven)
            model->BSIM4v5type = NMOS;     

        if (!model->BSIM4v5mobModGiven) 
            model->BSIM4v5mobMod = 0;
	else if ((model->BSIM4v5mobMod != 0) && (model->BSIM4v5mobMod != 1)
	         && (model->BSIM4v5mobMod != 2))
	{   model->BSIM4v5mobMod = 0;
            printf("Warning: mobMod has been set to its default value: 0.\n");
	}

        if (!model->BSIM4v5binUnitGiven) 
            model->BSIM4v5binUnit = 1;
        if (!model->BSIM4v5paramChkGiven) 
            model->BSIM4v5paramChk = 1;

        if (!model->BSIM4v5dioModGiven)
            model->BSIM4v5dioMod = 1;
        else if ((model->BSIM4v5dioMod != 0) && (model->BSIM4v5dioMod != 1)
            && (model->BSIM4v5dioMod != 2))
        {   model->BSIM4v5dioMod = 1;
            printf("Warning: dioMod has been set to its default value: 1.\n");
        }

        if (!model->BSIM4v5capModGiven) 
            model->BSIM4v5capMod = 2;
        else if ((model->BSIM4v5capMod != 0) && (model->BSIM4v5capMod != 1)
            && (model->BSIM4v5capMod != 2))
        {   model->BSIM4v5capMod = 2;
            printf("Warning: capMod has been set to its default value: 2.\n");
        }

        if (!model->BSIM4v5rdsModGiven)
            model->BSIM4v5rdsMod = 0;
	else if ((model->BSIM4v5rdsMod != 0) && (model->BSIM4v5rdsMod != 1))
        {   model->BSIM4v5rdsMod = 0;
	    printf("Warning: rdsMod has been set to its default value: 0.\n");
	}
        if (!model->BSIM4v5rbodyModGiven)
            model->BSIM4v5rbodyMod = 0;
        else if ((model->BSIM4v5rbodyMod != 0) && (model->BSIM4v5rbodyMod != 1) && (model->BSIM4v5rbodyMod != 2))
        {   model->BSIM4v5rbodyMod = 0;
            printf("Warning: rbodyMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v5rgateModGiven)
            model->BSIM4v5rgateMod = 0;
        else if ((model->BSIM4v5rgateMod != 0) && (model->BSIM4v5rgateMod != 1)
            && (model->BSIM4v5rgateMod != 2) && (model->BSIM4v5rgateMod != 3))
        {   model->BSIM4v5rgateMod = 0;
            printf("Warning: rgateMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v5perModGiven)
            model->BSIM4v5perMod = 1;
        else if ((model->BSIM4v5perMod != 0) && (model->BSIM4v5perMod != 1))
        {   model->BSIM4v5perMod = 1;
            printf("Warning: perMod has been set to its default value: 1.\n");
        }

        if (!model->BSIM4v5geoModGiven)
            model->BSIM4v5geoMod = 0;

        if (!model->BSIM4v5rgeoModGiven)
            model->BSIM4v5rgeoMod = 0;
        else if ((model->BSIM4v5rgeoMod != 0) && (model->BSIM4v5rgeoMod != 1))
        {   model->BSIM4v5rgeoMod = 1;
            printf("Warning: rgeoMod has been set to its default value: 1.\n");
        }

        if (!model->BSIM4v5fnoiModGiven) 
            model->BSIM4v5fnoiMod = 1;
        else if ((model->BSIM4v5fnoiMod != 0) && (model->BSIM4v5fnoiMod != 1))
        {   model->BSIM4v5fnoiMod = 1;
            printf("Warning: fnoiMod has been set to its default value: 1.\n");
        }
        if (!model->BSIM4v5tnoiModGiven)
            model->BSIM4v5tnoiMod = 0; /* WDLiu: tnoiMod=1 needs to set internal S/D nodes */
        else if ((model->BSIM4v5tnoiMod != 0) && (model->BSIM4v5tnoiMod != 1))
        {   model->BSIM4v5tnoiMod = 0;
            printf("Warning: tnoiMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v5trnqsModGiven)
            model->BSIM4v5trnqsMod = 0; 
        else if ((model->BSIM4v5trnqsMod != 0) && (model->BSIM4v5trnqsMod != 1))
        {   model->BSIM4v5trnqsMod = 0;
            printf("Warning: trnqsMod has been set to its default value: 0.\n");
        }
        if (!model->BSIM4v5acnqsModGiven)
            model->BSIM4v5acnqsMod = 0;
        else if ((model->BSIM4v5acnqsMod != 0) && (model->BSIM4v5acnqsMod != 1))
        {   model->BSIM4v5acnqsMod = 0;
            printf("Warning: acnqsMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v5igcModGiven)
            model->BSIM4v5igcMod = 0;
        else if ((model->BSIM4v5igcMod != 0) && (model->BSIM4v5igcMod != 1)
		  && (model->BSIM4v5igcMod != 2))
        {   model->BSIM4v5igcMod = 0;
            printf("Warning: igcMod has been set to its default value: 0.\n");
        }
        if (!model->BSIM4v5igbModGiven)
            model->BSIM4v5igbMod = 0;
        else if ((model->BSIM4v5igbMod != 0) && (model->BSIM4v5igbMod != 1))
        {   model->BSIM4v5igbMod = 0;
            printf("Warning: igbMod has been set to its default value: 0.\n");
        }
        if (!model->BSIM4v5tempModGiven)
            model->BSIM4v5tempMod = 0;
        else if ((model->BSIM4v5tempMod != 0) && (model->BSIM4v5tempMod != 1) 
		  && (model->BSIM4v5tempMod != 2))
        {   model->BSIM4v5tempMod = 0;
            printf("Warning: tempMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v5versionGiven) 
            model->BSIM4v5version = "4.5.0";
        if (!model->BSIM4v5toxrefGiven)
            model->BSIM4v5toxref = 30.0e-10;
        if (!model->BSIM4v5toxeGiven)
            model->BSIM4v5toxe = 30.0e-10;
        if (!model->BSIM4v5toxpGiven)
            model->BSIM4v5toxp = model->BSIM4v5toxe;
        if (!model->BSIM4v5toxmGiven)
            model->BSIM4v5toxm = model->BSIM4v5toxe;
        if (!model->BSIM4v5dtoxGiven)
            model->BSIM4v5dtox = 0.0;
        if (!model->BSIM4v5epsroxGiven)
            model->BSIM4v5epsrox = 3.9;

        if (!model->BSIM4v5cdscGiven)
	    model->BSIM4v5cdsc = 2.4e-4;   /* unit Q/V/m^2  */
        if (!model->BSIM4v5cdscbGiven)
	    model->BSIM4v5cdscb = 0.0;   /* unit Q/V/m^2  */    
        if (!model->BSIM4v5cdscdGiven)
	    model->BSIM4v5cdscd = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM4v5citGiven)
	    model->BSIM4v5cit = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM4v5nfactorGiven)
	    model->BSIM4v5nfactor = 1.0;
        if (!model->BSIM4v5xjGiven)
            model->BSIM4v5xj = .15e-6;
        if (!model->BSIM4v5vsatGiven)
            model->BSIM4v5vsat = 8.0e4;    /* unit m/s */ 
        if (!model->BSIM4v5atGiven)
            model->BSIM4v5at = 3.3e4;    /* unit m/s */ 
        if (!model->BSIM4v5a0Given)
            model->BSIM4v5a0 = 1.0;  
        if (!model->BSIM4v5agsGiven)
            model->BSIM4v5ags = 0.0;
        if (!model->BSIM4v5a1Given)
            model->BSIM4v5a1 = 0.0;
        if (!model->BSIM4v5a2Given)
            model->BSIM4v5a2 = 1.0;
        if (!model->BSIM4v5ketaGiven)
            model->BSIM4v5keta = -0.047;    /* unit  / V */
        if (!model->BSIM4v5nsubGiven)
            model->BSIM4v5nsub = 6.0e16;   /* unit 1/cm3 */
        if (!model->BSIM4v5ndepGiven)
            model->BSIM4v5ndep = 1.7e17;   /* unit 1/cm3 */
        if (!model->BSIM4v5nsdGiven)
            model->BSIM4v5nsd = 1.0e20;   /* unit 1/cm3 */
        if (!model->BSIM4v5phinGiven)
            model->BSIM4v5phin = 0.0; /* unit V */
        if (!model->BSIM4v5ngateGiven)
            model->BSIM4v5ngate = 0;   /* unit 1/cm3 */
        if (!model->BSIM4v5vbmGiven)
	    model->BSIM4v5vbm = -3.0;
        if (!model->BSIM4v5xtGiven)
	    model->BSIM4v5xt = 1.55e-7;
        if (!model->BSIM4v5kt1Given)
            model->BSIM4v5kt1 = -0.11;      /* unit V */
        if (!model->BSIM4v5kt1lGiven)
            model->BSIM4v5kt1l = 0.0;      /* unit V*m */
        if (!model->BSIM4v5kt2Given)
            model->BSIM4v5kt2 = 0.022;      /* No unit */
        if (!model->BSIM4v5k3Given)
            model->BSIM4v5k3 = 80.0;      
        if (!model->BSIM4v5k3bGiven)
            model->BSIM4v5k3b = 0.0;      
        if (!model->BSIM4v5w0Given)
            model->BSIM4v5w0 = 2.5e-6;    
        if (!model->BSIM4v5lpe0Given)
            model->BSIM4v5lpe0 = 1.74e-7;     
        if (!model->BSIM4v5lpebGiven)
            model->BSIM4v5lpeb = 0.0;
        if (!model->BSIM4v5dvtp0Given)
            model->BSIM4v5dvtp0 = 0.0;
        if (!model->BSIM4v5dvtp1Given)
            model->BSIM4v5dvtp1 = 0.0;
        if (!model->BSIM4v5dvt0Given)
            model->BSIM4v5dvt0 = 2.2;    
        if (!model->BSIM4v5dvt1Given)
            model->BSIM4v5dvt1 = 0.53;      
        if (!model->BSIM4v5dvt2Given)
            model->BSIM4v5dvt2 = -0.032;   /* unit 1 / V */     

        if (!model->BSIM4v5dvt0wGiven)
            model->BSIM4v5dvt0w = 0.0;    
        if (!model->BSIM4v5dvt1wGiven)
            model->BSIM4v5dvt1w = 5.3e6;    
        if (!model->BSIM4v5dvt2wGiven)
            model->BSIM4v5dvt2w = -0.032;   

        if (!model->BSIM4v5droutGiven)
            model->BSIM4v5drout = 0.56;     
        if (!model->BSIM4v5dsubGiven)
            model->BSIM4v5dsub = model->BSIM4v5drout;     
        if (!model->BSIM4v5vth0Given)
            model->BSIM4v5vth0 = (model->BSIM4v5type == NMOS) ? 0.7 : -0.7;
        if (!model->BSIM4v5euGiven)
            model->BSIM4v5eu = (model->BSIM4v5type == NMOS) ? 1.67 : 1.0;;
        if (!model->BSIM4v5uaGiven)
            model->BSIM4v5ua = (model->BSIM4v5mobMod == 2) ? 1.0e-15 : 1.0e-9; /* unit m/V */
        if (!model->BSIM4v5ua1Given)
            model->BSIM4v5ua1 = 1.0e-9;      /* unit m/V */
        if (!model->BSIM4v5ubGiven)
            model->BSIM4v5ub = 1.0e-19;     /* unit (m/V)**2 */
        if (!model->BSIM4v5ub1Given)
            model->BSIM4v5ub1 = -1.0e-18;     /* unit (m/V)**2 */
        if (!model->BSIM4v5ucGiven)
            model->BSIM4v5uc = (model->BSIM4v5mobMod == 1) ? -0.0465 : -0.0465e-9;   
        if (!model->BSIM4v5uc1Given)
            model->BSIM4v5uc1 = (model->BSIM4v5mobMod == 1) ? -0.056 : -0.056e-9;   
        if (!model->BSIM4v5udGiven)
            model->BSIM4v5ud = 1.0e14;     /* unit m**(-2) */
        if (!model->BSIM4v5ud1Given)
            model->BSIM4v5ud1 = 0.0;     
        if (!model->BSIM4v5upGiven)
            model->BSIM4v5up = 0.0;     
        if (!model->BSIM4v5lpGiven)
            model->BSIM4v5lp = 1.0e-8;     
        if (!model->BSIM4v5u0Given)
            model->BSIM4v5u0 = (model->BSIM4v5type == NMOS) ? 0.067 : 0.025;
        if (!model->BSIM4v5uteGiven)
	    model->BSIM4v5ute = -1.5;    
        if (!model->BSIM4v5voffGiven)
	    model->BSIM4v5voff = -0.08;
        if (!model->BSIM4v5vofflGiven)
            model->BSIM4v5voffl = 0.0;
        if (!model->BSIM4v5minvGiven)
            model->BSIM4v5minv = 0.0;
        if (!model->BSIM4v5fproutGiven)
            model->BSIM4v5fprout = 0.0;
        if (!model->BSIM4v5pditsGiven)
            model->BSIM4v5pdits = 0.0;
        if (!model->BSIM4v5pditsdGiven)
            model->BSIM4v5pditsd = 0.0;
        if (!model->BSIM4v5pditslGiven)
            model->BSIM4v5pditsl = 0.0;
        if (!model->BSIM4v5deltaGiven)  
           model->BSIM4v5delta = 0.01;
        if (!model->BSIM4v5rdswminGiven)
            model->BSIM4v5rdswmin = 0.0;
        if (!model->BSIM4v5rdwminGiven)
            model->BSIM4v5rdwmin = 0.0;
        if (!model->BSIM4v5rswminGiven)
            model->BSIM4v5rswmin = 0.0;
        if (!model->BSIM4v5rdswGiven)
	    model->BSIM4v5rdsw = 200.0; /* in ohm*um */     
        if (!model->BSIM4v5rdwGiven)
            model->BSIM4v5rdw = 100.0;
        if (!model->BSIM4v5rswGiven)
            model->BSIM4v5rsw = 100.0;
        if (!model->BSIM4v5prwgGiven)
            model->BSIM4v5prwg = 1.0; /* in 1/V */
        if (!model->BSIM4v5prwbGiven)
            model->BSIM4v5prwb = 0.0;      
        if (!model->BSIM4v5prtGiven)
            model->BSIM4v5prt = 0.0;      
        if (!model->BSIM4v5eta0Given)
            model->BSIM4v5eta0 = 0.08;      /* no unit  */ 
        if (!model->BSIM4v5etabGiven)
            model->BSIM4v5etab = -0.07;      /* unit  1/V */ 
        if (!model->BSIM4v5pclmGiven)
            model->BSIM4v5pclm = 1.3;      /* no unit  */ 
        if (!model->BSIM4v5pdibl1Given)
            model->BSIM4v5pdibl1 = 0.39;    /* no unit  */
        if (!model->BSIM4v5pdibl2Given)
            model->BSIM4v5pdibl2 = 0.0086;    /* no unit  */ 
        if (!model->BSIM4v5pdiblbGiven)
            model->BSIM4v5pdiblb = 0.0;    /* 1/V  */ 
        if (!model->BSIM4v5pscbe1Given)
            model->BSIM4v5pscbe1 = 4.24e8;     
        if (!model->BSIM4v5pscbe2Given)
            model->BSIM4v5pscbe2 = 1.0e-5;    
        if (!model->BSIM4v5pvagGiven)
            model->BSIM4v5pvag = 0.0;     
        if (!model->BSIM4v5wrGiven)  
            model->BSIM4v5wr = 1.0;
        if (!model->BSIM4v5dwgGiven)  
            model->BSIM4v5dwg = 0.0;
        if (!model->BSIM4v5dwbGiven)  
            model->BSIM4v5dwb = 0.0;
        if (!model->BSIM4v5b0Given)
            model->BSIM4v5b0 = 0.0;
        if (!model->BSIM4v5b1Given)  
            model->BSIM4v5b1 = 0.0;
        if (!model->BSIM4v5alpha0Given)  
            model->BSIM4v5alpha0 = 0.0;
        if (!model->BSIM4v5alpha1Given)
            model->BSIM4v5alpha1 = 0.0;
        if (!model->BSIM4v5beta0Given)  
            model->BSIM4v5beta0 = 0.0;
        if (!model->BSIM4v5agidlGiven)
            model->BSIM4v5agidl = 0.0;
        if (!model->BSIM4v5bgidlGiven)
            model->BSIM4v5bgidl = 2.3e9; /* V/m */
        if (!model->BSIM4v5cgidlGiven)
            model->BSIM4v5cgidl = 0.5; /* V^3 */
        if (!model->BSIM4v5egidlGiven)
            model->BSIM4v5egidl = 0.8; /* V */
        if (!model->BSIM4v5aigcGiven)
            model->BSIM4v5aigc = (model->BSIM4v5type == NMOS) ? 1.36e-2 : 9.80e-3;
        if (!model->BSIM4v5bigcGiven)
            model->BSIM4v5bigc = (model->BSIM4v5type == NMOS) ? 1.71e-3 : 7.59e-4;
        if (!model->BSIM4v5cigcGiven)
            model->BSIM4v5cigc = (model->BSIM4v5type == NMOS) ? 0.075 : 0.03;
        if (!model->BSIM4v5aigsdGiven)
            model->BSIM4v5aigsd = (model->BSIM4v5type == NMOS) ? 1.36e-2 : 9.80e-3;
        if (!model->BSIM4v5bigsdGiven)
            model->BSIM4v5bigsd = (model->BSIM4v5type == NMOS) ? 1.71e-3 : 7.59e-4;
        if (!model->BSIM4v5cigsdGiven)
            model->BSIM4v5cigsd = (model->BSIM4v5type == NMOS) ? 0.075 : 0.03;
        if (!model->BSIM4v5aigbaccGiven)
            model->BSIM4v5aigbacc = 1.36e-2;
        if (!model->BSIM4v5bigbaccGiven)
            model->BSIM4v5bigbacc = 1.71e-3;
        if (!model->BSIM4v5cigbaccGiven)
            model->BSIM4v5cigbacc = 0.075;
        if (!model->BSIM4v5aigbinvGiven)
            model->BSIM4v5aigbinv = 1.11e-2;
        if (!model->BSIM4v5bigbinvGiven)
            model->BSIM4v5bigbinv = 9.49e-4;
        if (!model->BSIM4v5cigbinvGiven)
            model->BSIM4v5cigbinv = 0.006;
        if (!model->BSIM4v5nigcGiven)
            model->BSIM4v5nigc = 1.0;
        if (!model->BSIM4v5nigbinvGiven)
            model->BSIM4v5nigbinv = 3.0;
        if (!model->BSIM4v5nigbaccGiven)
            model->BSIM4v5nigbacc = 1.0;
        if (!model->BSIM4v5ntoxGiven)
            model->BSIM4v5ntox = 1.0;
        if (!model->BSIM4v5eigbinvGiven)
            model->BSIM4v5eigbinv = 1.1;
        if (!model->BSIM4v5pigcdGiven)
            model->BSIM4v5pigcd = 1.0;
        if (!model->BSIM4v5poxedgeGiven)
            model->BSIM4v5poxedge = 1.0;
        if (!model->BSIM4v5xrcrg1Given)
            model->BSIM4v5xrcrg1 = 12.0;
        if (!model->BSIM4v5xrcrg2Given)
            model->BSIM4v5xrcrg2 = 1.0;
        if (!model->BSIM4v5ijthsfwdGiven)
            model->BSIM4v5ijthsfwd = 0.1; /* unit A */
        if (!model->BSIM4v5ijthdfwdGiven)
            model->BSIM4v5ijthdfwd = model->BSIM4v5ijthsfwd;
        if (!model->BSIM4v5ijthsrevGiven)
            model->BSIM4v5ijthsrev = 0.1; /* unit A */
        if (!model->BSIM4v5ijthdrevGiven)
            model->BSIM4v5ijthdrev = model->BSIM4v5ijthsrev;
        if (!model->BSIM4v5tnoiaGiven)
            model->BSIM4v5tnoia = 1.5;
        if (!model->BSIM4v5tnoibGiven)
            model->BSIM4v5tnoib = 3.5;
        if (!model->BSIM4v5rnoiaGiven)
            model->BSIM4v5rnoia = 0.577;
        if (!model->BSIM4v5rnoibGiven)
            model->BSIM4v5rnoib = 0.5164;
        if (!model->BSIM4v5ntnoiGiven)
            model->BSIM4v5ntnoi = 1.0;
        if (!model->BSIM4v5lambdaGiven)
            model->BSIM4v5lambda = 0.0;
        if (!model->BSIM4v5vtlGiven)
            model->BSIM4v5vtl = 2.0e5;    /* unit m/s */ 
        if (!model->BSIM4v5xnGiven)
            model->BSIM4v5xn = 3.0;   
        if (!model->BSIM4v5lcGiven)
            model->BSIM4v5lc = 5.0e-9;   
        if (!model->BSIM4v5vfbsdoffGiven)  
            model->BSIM4v5vfbsdoff = 0.0;  /* unit v */  
        if (!model->BSIM4v5tvfbsdoffGiven)
            model->BSIM4v5tvfbsdoff = 0.0;  
        if (!model->BSIM4v5tvoffGiven)
            model->BSIM4v5tvoff = 0.0;  
 
        if (!model->BSIM4v5lintnoiGiven)
            model->BSIM4v5lintnoi = 0.0;  /* unit m */  

        if (!model->BSIM4v5xjbvsGiven)
            model->BSIM4v5xjbvs = 1.0; /* no unit */
        if (!model->BSIM4v5xjbvdGiven)
            model->BSIM4v5xjbvd = model->BSIM4v5xjbvs;
        if (!model->BSIM4v5bvsGiven)
            model->BSIM4v5bvs = 10.0; /* V */
        if (!model->BSIM4v5bvdGiven)
            model->BSIM4v5bvd = model->BSIM4v5bvs;

        if (!model->BSIM4v5gbminGiven)
            model->BSIM4v5gbmin = 1.0e-12; /* in mho */
        if (!model->BSIM4v5rbdbGiven)
            model->BSIM4v5rbdb = 50.0; /* in ohm */
        if (!model->BSIM4v5rbpbGiven)
            model->BSIM4v5rbpb = 50.0;
        if (!model->BSIM4v5rbsbGiven)
            model->BSIM4v5rbsb = 50.0;
        if (!model->BSIM4v5rbpsGiven)
            model->BSIM4v5rbps = 50.0;
        if (!model->BSIM4v5rbpdGiven)
            model->BSIM4v5rbpd = 50.0;

        if (!model->BSIM4v5rbps0Given)
            model->BSIM4v5rbps0 = 50.0;
        if (!model->BSIM4v5rbpslGiven)
            model->BSIM4v5rbpsl = 0.0;
        if (!model->BSIM4v5rbpswGiven)
            model->BSIM4v5rbpsw = 0.0;
        if (!model->BSIM4v5rbpsnfGiven)
            model->BSIM4v5rbpsnf = 0.0;

        if (!model->BSIM4v5rbpd0Given)
            model->BSIM4v5rbpd0 = 50.0;
        if (!model->BSIM4v5rbpdlGiven)
            model->BSIM4v5rbpdl = 0.0;
        if (!model->BSIM4v5rbpdwGiven)
            model->BSIM4v5rbpdw = 0.0;
        if (!model->BSIM4v5rbpdnfGiven)
            model->BSIM4v5rbpdnf = 0.0;

        if (!model->BSIM4v5rbpbx0Given)
            model->BSIM4v5rbpbx0 = 100.0;
        if (!model->BSIM4v5rbpbxlGiven)
            model->BSIM4v5rbpbxl = 0.0;
        if (!model->BSIM4v5rbpbxwGiven)
            model->BSIM4v5rbpbxw = 0.0;
        if (!model->BSIM4v5rbpbxnfGiven)
            model->BSIM4v5rbpbxnf = 0.0;
        if (!model->BSIM4v5rbpby0Given)
            model->BSIM4v5rbpby0 = 100.0;
        if (!model->BSIM4v5rbpbylGiven)
            model->BSIM4v5rbpbyl = 0.0;
        if (!model->BSIM4v5rbpbywGiven)
            model->BSIM4v5rbpbyw = 0.0;
        if (!model->BSIM4v5rbpbynfGiven)
            model->BSIM4v5rbpbynf = 0.0;


        if (!model->BSIM4v5rbsbx0Given)
            model->BSIM4v5rbsbx0 = 100.0;
        if (!model->BSIM4v5rbsby0Given)
            model->BSIM4v5rbsby0 = 100.0;
        if (!model->BSIM4v5rbdbx0Given)
            model->BSIM4v5rbdbx0 = 100.0;
        if (!model->BSIM4v5rbdby0Given)
            model->BSIM4v5rbdby0 = 100.0;


        if (!model->BSIM4v5rbsdbxlGiven)
            model->BSIM4v5rbsdbxl = 0.0;
        if (!model->BSIM4v5rbsdbxwGiven)
            model->BSIM4v5rbsdbxw = 0.0;
        if (!model->BSIM4v5rbsdbxnfGiven)
            model->BSIM4v5rbsdbxnf = 0.0;
        if (!model->BSIM4v5rbsdbylGiven)
            model->BSIM4v5rbsdbyl = 0.0;
        if (!model->BSIM4v5rbsdbywGiven)
            model->BSIM4v5rbsdbyw = 0.0;
        if (!model->BSIM4v5rbsdbynfGiven)
            model->BSIM4v5rbsdbynf = 0.0;

        if (!model->BSIM4v5cgslGiven)  
            model->BSIM4v5cgsl = 0.0;
        if (!model->BSIM4v5cgdlGiven)  
            model->BSIM4v5cgdl = 0.0;
        if (!model->BSIM4v5ckappasGiven)  
            model->BSIM4v5ckappas = 0.6;
        if (!model->BSIM4v5ckappadGiven)
            model->BSIM4v5ckappad = model->BSIM4v5ckappas;
        if (!model->BSIM4v5clcGiven)  
            model->BSIM4v5clc = 0.1e-6;
        if (!model->BSIM4v5cleGiven)  
            model->BSIM4v5cle = 0.6;
        if (!model->BSIM4v5vfbcvGiven)  
            model->BSIM4v5vfbcv = -1.0;
        if (!model->BSIM4v5acdeGiven)
            model->BSIM4v5acde = 1.0;
        if (!model->BSIM4v5moinGiven)
            model->BSIM4v5moin = 15.0;
        if (!model->BSIM4v5noffGiven)
            model->BSIM4v5noff = 1.0;
        if (!model->BSIM4v5voffcvGiven)
            model->BSIM4v5voffcv = 0.0;
        if (!model->BSIM4v5dmcgGiven)
            model->BSIM4v5dmcg = 0.0;
        if (!model->BSIM4v5dmciGiven)
            model->BSIM4v5dmci = model->BSIM4v5dmcg;
        if (!model->BSIM4v5dmdgGiven)
            model->BSIM4v5dmdg = 0.0;
        if (!model->BSIM4v5dmcgtGiven)
            model->BSIM4v5dmcgt = 0.0;
        if (!model->BSIM4v5xgwGiven)
            model->BSIM4v5xgw = 0.0;
        if (!model->BSIM4v5xglGiven)
            model->BSIM4v5xgl = 0.0;
        if (!model->BSIM4v5rshgGiven)
            model->BSIM4v5rshg = 0.1;
        if (!model->BSIM4v5ngconGiven)
            model->BSIM4v5ngcon = 1.0;
        if (!model->BSIM4v5tcjGiven)
            model->BSIM4v5tcj = 0.0;
        if (!model->BSIM4v5tpbGiven)
            model->BSIM4v5tpb = 0.0;
        if (!model->BSIM4v5tcjswGiven)
            model->BSIM4v5tcjsw = 0.0;
        if (!model->BSIM4v5tpbswGiven)
            model->BSIM4v5tpbsw = 0.0;
        if (!model->BSIM4v5tcjswgGiven)
            model->BSIM4v5tcjswg = 0.0;
        if (!model->BSIM4v5tpbswgGiven)
            model->BSIM4v5tpbswg = 0.0;

	/* Length dependence */
        if (!model->BSIM4v5lcdscGiven)
	    model->BSIM4v5lcdsc = 0.0;
        if (!model->BSIM4v5lcdscbGiven)
	    model->BSIM4v5lcdscb = 0.0;
        if (!model->BSIM4v5lcdscdGiven)
	    model->BSIM4v5lcdscd = 0.0;
        if (!model->BSIM4v5lcitGiven)
	    model->BSIM4v5lcit = 0.0;
        if (!model->BSIM4v5lnfactorGiven)
	    model->BSIM4v5lnfactor = 0.0;
        if (!model->BSIM4v5lxjGiven)
            model->BSIM4v5lxj = 0.0;
        if (!model->BSIM4v5lvsatGiven)
            model->BSIM4v5lvsat = 0.0;
        if (!model->BSIM4v5latGiven)
            model->BSIM4v5lat = 0.0;
        if (!model->BSIM4v5la0Given)
            model->BSIM4v5la0 = 0.0; 
        if (!model->BSIM4v5lagsGiven)
            model->BSIM4v5lags = 0.0;
        if (!model->BSIM4v5la1Given)
            model->BSIM4v5la1 = 0.0;
        if (!model->BSIM4v5la2Given)
            model->BSIM4v5la2 = 0.0;
        if (!model->BSIM4v5lketaGiven)
            model->BSIM4v5lketa = 0.0;
        if (!model->BSIM4v5lnsubGiven)
            model->BSIM4v5lnsub = 0.0;
        if (!model->BSIM4v5lndepGiven)
            model->BSIM4v5lndep = 0.0;
        if (!model->BSIM4v5lnsdGiven)
            model->BSIM4v5lnsd = 0.0;
        if (!model->BSIM4v5lphinGiven)
            model->BSIM4v5lphin = 0.0;
        if (!model->BSIM4v5lngateGiven)
            model->BSIM4v5lngate = 0.0;
        if (!model->BSIM4v5lvbmGiven)
	    model->BSIM4v5lvbm = 0.0;
        if (!model->BSIM4v5lxtGiven)
	    model->BSIM4v5lxt = 0.0;
        if (!model->BSIM4v5lkt1Given)
            model->BSIM4v5lkt1 = 0.0; 
        if (!model->BSIM4v5lkt1lGiven)
            model->BSIM4v5lkt1l = 0.0;
        if (!model->BSIM4v5lkt2Given)
            model->BSIM4v5lkt2 = 0.0;
        if (!model->BSIM4v5lk3Given)
            model->BSIM4v5lk3 = 0.0;      
        if (!model->BSIM4v5lk3bGiven)
            model->BSIM4v5lk3b = 0.0;      
        if (!model->BSIM4v5lw0Given)
            model->BSIM4v5lw0 = 0.0;    
        if (!model->BSIM4v5llpe0Given)
            model->BSIM4v5llpe0 = 0.0;
        if (!model->BSIM4v5llpebGiven)
            model->BSIM4v5llpeb = 0.0; 
        if (!model->BSIM4v5ldvtp0Given)
            model->BSIM4v5ldvtp0 = 0.0;
        if (!model->BSIM4v5ldvtp1Given)
            model->BSIM4v5ldvtp1 = 0.0;
        if (!model->BSIM4v5ldvt0Given)
            model->BSIM4v5ldvt0 = 0.0;    
        if (!model->BSIM4v5ldvt1Given)
            model->BSIM4v5ldvt1 = 0.0;      
        if (!model->BSIM4v5ldvt2Given)
            model->BSIM4v5ldvt2 = 0.0;
        if (!model->BSIM4v5ldvt0wGiven)
            model->BSIM4v5ldvt0w = 0.0;    
        if (!model->BSIM4v5ldvt1wGiven)
            model->BSIM4v5ldvt1w = 0.0;      
        if (!model->BSIM4v5ldvt2wGiven)
            model->BSIM4v5ldvt2w = 0.0;
        if (!model->BSIM4v5ldroutGiven)
            model->BSIM4v5ldrout = 0.0;     
        if (!model->BSIM4v5ldsubGiven)
            model->BSIM4v5ldsub = 0.0;
        if (!model->BSIM4v5lvth0Given)
           model->BSIM4v5lvth0 = 0.0;
        if (!model->BSIM4v5luaGiven)
            model->BSIM4v5lua = 0.0;
        if (!model->BSIM4v5lua1Given)
            model->BSIM4v5lua1 = 0.0;
        if (!model->BSIM4v5lubGiven)
            model->BSIM4v5lub = 0.0;
        if (!model->BSIM4v5lub1Given)
            model->BSIM4v5lub1 = 0.0;
        if (!model->BSIM4v5lucGiven)
            model->BSIM4v5luc = 0.0;
        if (!model->BSIM4v5luc1Given)
            model->BSIM4v5luc1 = 0.0;
        if (!model->BSIM4v5ludGiven)
            model->BSIM4v5lud = 0.0;
        if (!model->BSIM4v5lud1Given)
            model->BSIM4v5lud1 = 0.0;
        if (!model->BSIM4v5lupGiven)
            model->BSIM4v5lup = 0.0;
        if (!model->BSIM4v5llpGiven)
            model->BSIM4v5llp = 0.0;
        if (!model->BSIM4v5lu0Given)
            model->BSIM4v5lu0 = 0.0;
        if (!model->BSIM4v5luteGiven)
	    model->BSIM4v5lute = 0.0;    
        if (!model->BSIM4v5lvoffGiven)
	    model->BSIM4v5lvoff = 0.0;
        if (!model->BSIM4v5lminvGiven)
            model->BSIM4v5lminv = 0.0;
        if (!model->BSIM4v5lfproutGiven)
            model->BSIM4v5lfprout = 0.0;
        if (!model->BSIM4v5lpditsGiven)
            model->BSIM4v5lpdits = 0.0;
        if (!model->BSIM4v5lpditsdGiven)
            model->BSIM4v5lpditsd = 0.0;
        if (!model->BSIM4v5ldeltaGiven)  
            model->BSIM4v5ldelta = 0.0;
        if (!model->BSIM4v5lrdswGiven)
            model->BSIM4v5lrdsw = 0.0;
        if (!model->BSIM4v5lrdwGiven)
            model->BSIM4v5lrdw = 0.0;
        if (!model->BSIM4v5lrswGiven)
            model->BSIM4v5lrsw = 0.0;
        if (!model->BSIM4v5lprwbGiven)
            model->BSIM4v5lprwb = 0.0;
        if (!model->BSIM4v5lprwgGiven)
            model->BSIM4v5lprwg = 0.0;
        if (!model->BSIM4v5lprtGiven)
            model->BSIM4v5lprt = 0.0;
        if (!model->BSIM4v5leta0Given)
            model->BSIM4v5leta0 = 0.0;
        if (!model->BSIM4v5letabGiven)
            model->BSIM4v5letab = -0.0;
        if (!model->BSIM4v5lpclmGiven)
            model->BSIM4v5lpclm = 0.0; 
        if (!model->BSIM4v5lpdibl1Given)
            model->BSIM4v5lpdibl1 = 0.0;
        if (!model->BSIM4v5lpdibl2Given)
            model->BSIM4v5lpdibl2 = 0.0;
        if (!model->BSIM4v5lpdiblbGiven)
            model->BSIM4v5lpdiblb = 0.0;
        if (!model->BSIM4v5lpscbe1Given)
            model->BSIM4v5lpscbe1 = 0.0;
        if (!model->BSIM4v5lpscbe2Given)
            model->BSIM4v5lpscbe2 = 0.0;
        if (!model->BSIM4v5lpvagGiven)
            model->BSIM4v5lpvag = 0.0;     
        if (!model->BSIM4v5lwrGiven)  
            model->BSIM4v5lwr = 0.0;
        if (!model->BSIM4v5ldwgGiven)  
            model->BSIM4v5ldwg = 0.0;
        if (!model->BSIM4v5ldwbGiven)  
            model->BSIM4v5ldwb = 0.0;
        if (!model->BSIM4v5lb0Given)
            model->BSIM4v5lb0 = 0.0;
        if (!model->BSIM4v5lb1Given)  
            model->BSIM4v5lb1 = 0.0;
        if (!model->BSIM4v5lalpha0Given)  
            model->BSIM4v5lalpha0 = 0.0;
        if (!model->BSIM4v5lalpha1Given)
            model->BSIM4v5lalpha1 = 0.0;
        if (!model->BSIM4v5lbeta0Given)  
            model->BSIM4v5lbeta0 = 0.0;
        if (!model->BSIM4v5lagidlGiven)
            model->BSIM4v5lagidl = 0.0;
        if (!model->BSIM4v5lbgidlGiven)
            model->BSIM4v5lbgidl = 0.0;
        if (!model->BSIM4v5lcgidlGiven)
            model->BSIM4v5lcgidl = 0.0;
        if (!model->BSIM4v5legidlGiven)
            model->BSIM4v5legidl = 0.0;
        if (!model->BSIM4v5laigcGiven)
            model->BSIM4v5laigc = 0.0;
        if (!model->BSIM4v5lbigcGiven)
            model->BSIM4v5lbigc = 0.0;
        if (!model->BSIM4v5lcigcGiven)
            model->BSIM4v5lcigc = 0.0;
        if (!model->BSIM4v5laigsdGiven)
            model->BSIM4v5laigsd = 0.0;
        if (!model->BSIM4v5lbigsdGiven)
            model->BSIM4v5lbigsd = 0.0;
        if (!model->BSIM4v5lcigsdGiven)
            model->BSIM4v5lcigsd = 0.0;
        if (!model->BSIM4v5laigbaccGiven)
            model->BSIM4v5laigbacc = 0.0;
        if (!model->BSIM4v5lbigbaccGiven)
            model->BSIM4v5lbigbacc = 0.0;
        if (!model->BSIM4v5lcigbaccGiven)
            model->BSIM4v5lcigbacc = 0.0;
        if (!model->BSIM4v5laigbinvGiven)
            model->BSIM4v5laigbinv = 0.0;
        if (!model->BSIM4v5lbigbinvGiven)
            model->BSIM4v5lbigbinv = 0.0;
        if (!model->BSIM4v5lcigbinvGiven)
            model->BSIM4v5lcigbinv = 0.0;
        if (!model->BSIM4v5lnigcGiven)
            model->BSIM4v5lnigc = 0.0;
        if (!model->BSIM4v5lnigbinvGiven)
            model->BSIM4v5lnigbinv = 0.0;
        if (!model->BSIM4v5lnigbaccGiven)
            model->BSIM4v5lnigbacc = 0.0;
        if (!model->BSIM4v5lntoxGiven)
            model->BSIM4v5lntox = 0.0;
        if (!model->BSIM4v5leigbinvGiven)
            model->BSIM4v5leigbinv = 0.0;
        if (!model->BSIM4v5lpigcdGiven)
            model->BSIM4v5lpigcd = 0.0;
        if (!model->BSIM4v5lpoxedgeGiven)
            model->BSIM4v5lpoxedge = 0.0;
        if (!model->BSIM4v5lxrcrg1Given)
            model->BSIM4v5lxrcrg1 = 0.0;
        if (!model->BSIM4v5lxrcrg2Given)
            model->BSIM4v5lxrcrg2 = 0.0;
        if (!model->BSIM4v5leuGiven)
            model->BSIM4v5leu = 0.0;
        if (!model->BSIM4v5lvfbGiven)
            model->BSIM4v5lvfb = 0.0;
        if (!model->BSIM4v5llambdaGiven)
            model->BSIM4v5llambda = 0.0;
        if (!model->BSIM4v5lvtlGiven)
            model->BSIM4v5lvtl = 0.0;  
        if (!model->BSIM4v5lxnGiven)
            model->BSIM4v5lxn = 0.0;  
        if (!model->BSIM4v5lvfbsdoffGiven)
            model->BSIM4v5lvfbsdoff = 0.0;   
        if (!model->BSIM4v5ltvfbsdoffGiven)
            model->BSIM4v5ltvfbsdoff = 0.0;  
        if (!model->BSIM4v5ltvoffGiven)
            model->BSIM4v5ltvoff = 0.0;  


        if (!model->BSIM4v5lcgslGiven)  
            model->BSIM4v5lcgsl = 0.0;
        if (!model->BSIM4v5lcgdlGiven)  
            model->BSIM4v5lcgdl = 0.0;
        if (!model->BSIM4v5lckappasGiven)  
            model->BSIM4v5lckappas = 0.0;
        if (!model->BSIM4v5lckappadGiven)
            model->BSIM4v5lckappad = 0.0;
        if (!model->BSIM4v5lclcGiven)  
            model->BSIM4v5lclc = 0.0;
        if (!model->BSIM4v5lcleGiven)  
            model->BSIM4v5lcle = 0.0;
        if (!model->BSIM4v5lcfGiven)  
            model->BSIM4v5lcf = 0.0;
        if (!model->BSIM4v5lvfbcvGiven)  
            model->BSIM4v5lvfbcv = 0.0;
        if (!model->BSIM4v5lacdeGiven)
            model->BSIM4v5lacde = 0.0;
        if (!model->BSIM4v5lmoinGiven)
            model->BSIM4v5lmoin = 0.0;
        if (!model->BSIM4v5lnoffGiven)
            model->BSIM4v5lnoff = 0.0;
        if (!model->BSIM4v5lvoffcvGiven)
            model->BSIM4v5lvoffcv = 0.0;

	/* Width dependence */
        if (!model->BSIM4v5wcdscGiven)
	    model->BSIM4v5wcdsc = 0.0;
        if (!model->BSIM4v5wcdscbGiven)
	    model->BSIM4v5wcdscb = 0.0;  
        if (!model->BSIM4v5wcdscdGiven)
	    model->BSIM4v5wcdscd = 0.0;
        if (!model->BSIM4v5wcitGiven)
	    model->BSIM4v5wcit = 0.0;
        if (!model->BSIM4v5wnfactorGiven)
	    model->BSIM4v5wnfactor = 0.0;
        if (!model->BSIM4v5wxjGiven)
            model->BSIM4v5wxj = 0.0;
        if (!model->BSIM4v5wvsatGiven)
            model->BSIM4v5wvsat = 0.0;
        if (!model->BSIM4v5watGiven)
            model->BSIM4v5wat = 0.0;
        if (!model->BSIM4v5wa0Given)
            model->BSIM4v5wa0 = 0.0; 
        if (!model->BSIM4v5wagsGiven)
            model->BSIM4v5wags = 0.0;
        if (!model->BSIM4v5wa1Given)
            model->BSIM4v5wa1 = 0.0;
        if (!model->BSIM4v5wa2Given)
            model->BSIM4v5wa2 = 0.0;
        if (!model->BSIM4v5wketaGiven)
            model->BSIM4v5wketa = 0.0;
        if (!model->BSIM4v5wnsubGiven)
            model->BSIM4v5wnsub = 0.0;
        if (!model->BSIM4v5wndepGiven)
            model->BSIM4v5wndep = 0.0;
        if (!model->BSIM4v5wnsdGiven)
            model->BSIM4v5wnsd = 0.0;
        if (!model->BSIM4v5wphinGiven)
            model->BSIM4v5wphin = 0.0;
        if (!model->BSIM4v5wngateGiven)
            model->BSIM4v5wngate = 0.0;
        if (!model->BSIM4v5wvbmGiven)
	    model->BSIM4v5wvbm = 0.0;
        if (!model->BSIM4v5wxtGiven)
	    model->BSIM4v5wxt = 0.0;
        if (!model->BSIM4v5wkt1Given)
            model->BSIM4v5wkt1 = 0.0; 
        if (!model->BSIM4v5wkt1lGiven)
            model->BSIM4v5wkt1l = 0.0;
        if (!model->BSIM4v5wkt2Given)
            model->BSIM4v5wkt2 = 0.0;
        if (!model->BSIM4v5wk3Given)
            model->BSIM4v5wk3 = 0.0;      
        if (!model->BSIM4v5wk3bGiven)
            model->BSIM4v5wk3b = 0.0;      
        if (!model->BSIM4v5ww0Given)
            model->BSIM4v5ww0 = 0.0;    
        if (!model->BSIM4v5wlpe0Given)
            model->BSIM4v5wlpe0 = 0.0;
        if (!model->BSIM4v5wlpebGiven)
            model->BSIM4v5wlpeb = 0.0; 
        if (!model->BSIM4v5wdvtp0Given)
            model->BSIM4v5wdvtp0 = 0.0;
        if (!model->BSIM4v5wdvtp1Given)
            model->BSIM4v5wdvtp1 = 0.0;
        if (!model->BSIM4v5wdvt0Given)
            model->BSIM4v5wdvt0 = 0.0;    
        if (!model->BSIM4v5wdvt1Given)
            model->BSIM4v5wdvt1 = 0.0;      
        if (!model->BSIM4v5wdvt2Given)
            model->BSIM4v5wdvt2 = 0.0;
        if (!model->BSIM4v5wdvt0wGiven)
            model->BSIM4v5wdvt0w = 0.0;    
        if (!model->BSIM4v5wdvt1wGiven)
            model->BSIM4v5wdvt1w = 0.0;      
        if (!model->BSIM4v5wdvt2wGiven)
            model->BSIM4v5wdvt2w = 0.0;
        if (!model->BSIM4v5wdroutGiven)
            model->BSIM4v5wdrout = 0.0;     
        if (!model->BSIM4v5wdsubGiven)
            model->BSIM4v5wdsub = 0.0;
        if (!model->BSIM4v5wvth0Given)
           model->BSIM4v5wvth0 = 0.0;
        if (!model->BSIM4v5wuaGiven)
            model->BSIM4v5wua = 0.0;
        if (!model->BSIM4v5wua1Given)
            model->BSIM4v5wua1 = 0.0;
        if (!model->BSIM4v5wubGiven)
            model->BSIM4v5wub = 0.0;
        if (!model->BSIM4v5wub1Given)
            model->BSIM4v5wub1 = 0.0;
        if (!model->BSIM4v5wucGiven)
            model->BSIM4v5wuc = 0.0;
        if (!model->BSIM4v5wuc1Given)
            model->BSIM4v5wuc1 = 0.0;
        if (!model->BSIM4v5wudGiven)
            model->BSIM4v5wud = 0.0;
        if (!model->BSIM4v5wud1Given)
            model->BSIM4v5wud1 = 0.0;
        if (!model->BSIM4v5wupGiven)
            model->BSIM4v5wup = 0.0;
        if (!model->BSIM4v5wlpGiven)
            model->BSIM4v5wlp = 0.0;
        if (!model->BSIM4v5wu0Given)
            model->BSIM4v5wu0 = 0.0;
        if (!model->BSIM4v5wuteGiven)
	    model->BSIM4v5wute = 0.0;    
        if (!model->BSIM4v5wvoffGiven)
	    model->BSIM4v5wvoff = 0.0;
        if (!model->BSIM4v5wminvGiven)
            model->BSIM4v5wminv = 0.0;
        if (!model->BSIM4v5wfproutGiven)
            model->BSIM4v5wfprout = 0.0;
        if (!model->BSIM4v5wpditsGiven)
            model->BSIM4v5wpdits = 0.0;
        if (!model->BSIM4v5wpditsdGiven)
            model->BSIM4v5wpditsd = 0.0;
        if (!model->BSIM4v5wdeltaGiven)  
            model->BSIM4v5wdelta = 0.0;
        if (!model->BSIM4v5wrdswGiven)
            model->BSIM4v5wrdsw = 0.0;
        if (!model->BSIM4v5wrdwGiven)
            model->BSIM4v5wrdw = 0.0;
        if (!model->BSIM4v5wrswGiven)
            model->BSIM4v5wrsw = 0.0;
        if (!model->BSIM4v5wprwbGiven)
            model->BSIM4v5wprwb = 0.0;
        if (!model->BSIM4v5wprwgGiven)
            model->BSIM4v5wprwg = 0.0;
        if (!model->BSIM4v5wprtGiven)
            model->BSIM4v5wprt = 0.0;
        if (!model->BSIM4v5weta0Given)
            model->BSIM4v5weta0 = 0.0;
        if (!model->BSIM4v5wetabGiven)
            model->BSIM4v5wetab = 0.0;
        if (!model->BSIM4v5wpclmGiven)
            model->BSIM4v5wpclm = 0.0; 
        if (!model->BSIM4v5wpdibl1Given)
            model->BSIM4v5wpdibl1 = 0.0;
        if (!model->BSIM4v5wpdibl2Given)
            model->BSIM4v5wpdibl2 = 0.0;
        if (!model->BSIM4v5wpdiblbGiven)
            model->BSIM4v5wpdiblb = 0.0;
        if (!model->BSIM4v5wpscbe1Given)
            model->BSIM4v5wpscbe1 = 0.0;
        if (!model->BSIM4v5wpscbe2Given)
            model->BSIM4v5wpscbe2 = 0.0;
        if (!model->BSIM4v5wpvagGiven)
            model->BSIM4v5wpvag = 0.0;     
        if (!model->BSIM4v5wwrGiven)  
            model->BSIM4v5wwr = 0.0;
        if (!model->BSIM4v5wdwgGiven)  
            model->BSIM4v5wdwg = 0.0;
        if (!model->BSIM4v5wdwbGiven)  
            model->BSIM4v5wdwb = 0.0;
        if (!model->BSIM4v5wb0Given)
            model->BSIM4v5wb0 = 0.0;
        if (!model->BSIM4v5wb1Given)  
            model->BSIM4v5wb1 = 0.0;
        if (!model->BSIM4v5walpha0Given)  
            model->BSIM4v5walpha0 = 0.0;
        if (!model->BSIM4v5walpha1Given)
            model->BSIM4v5walpha1 = 0.0;
        if (!model->BSIM4v5wbeta0Given)  
            model->BSIM4v5wbeta0 = 0.0;
        if (!model->BSIM4v5wagidlGiven)
            model->BSIM4v5wagidl = 0.0;
        if (!model->BSIM4v5wbgidlGiven)
            model->BSIM4v5wbgidl = 0.0;
        if (!model->BSIM4v5wcgidlGiven)
            model->BSIM4v5wcgidl = 0.0;
        if (!model->BSIM4v5wegidlGiven)
            model->BSIM4v5wegidl = 0.0;
        if (!model->BSIM4v5waigcGiven)
            model->BSIM4v5waigc = 0.0;
        if (!model->BSIM4v5wbigcGiven)
            model->BSIM4v5wbigc = 0.0;
        if (!model->BSIM4v5wcigcGiven)
            model->BSIM4v5wcigc = 0.0;
        if (!model->BSIM4v5waigsdGiven)
            model->BSIM4v5waigsd = 0.0;
        if (!model->BSIM4v5wbigsdGiven)
            model->BSIM4v5wbigsd = 0.0;
        if (!model->BSIM4v5wcigsdGiven)
            model->BSIM4v5wcigsd = 0.0;
        if (!model->BSIM4v5waigbaccGiven)
            model->BSIM4v5waigbacc = 0.0;
        if (!model->BSIM4v5wbigbaccGiven)
            model->BSIM4v5wbigbacc = 0.0;
        if (!model->BSIM4v5wcigbaccGiven)
            model->BSIM4v5wcigbacc = 0.0;
        if (!model->BSIM4v5waigbinvGiven)
            model->BSIM4v5waigbinv = 0.0;
        if (!model->BSIM4v5wbigbinvGiven)
            model->BSIM4v5wbigbinv = 0.0;
        if (!model->BSIM4v5wcigbinvGiven)
            model->BSIM4v5wcigbinv = 0.0;
        if (!model->BSIM4v5wnigcGiven)
            model->BSIM4v5wnigc = 0.0;
        if (!model->BSIM4v5wnigbinvGiven)
            model->BSIM4v5wnigbinv = 0.0;
        if (!model->BSIM4v5wnigbaccGiven)
            model->BSIM4v5wnigbacc = 0.0;
        if (!model->BSIM4v5wntoxGiven)
            model->BSIM4v5wntox = 0.0;
        if (!model->BSIM4v5weigbinvGiven)
            model->BSIM4v5weigbinv = 0.0;
        if (!model->BSIM4v5wpigcdGiven)
            model->BSIM4v5wpigcd = 0.0;
        if (!model->BSIM4v5wpoxedgeGiven)
            model->BSIM4v5wpoxedge = 0.0;
        if (!model->BSIM4v5wxrcrg1Given)
            model->BSIM4v5wxrcrg1 = 0.0;
        if (!model->BSIM4v5wxrcrg2Given)
            model->BSIM4v5wxrcrg2 = 0.0;
        if (!model->BSIM4v5weuGiven)
            model->BSIM4v5weu = 0.0;
        if (!model->BSIM4v5wvfbGiven)
            model->BSIM4v5wvfb = 0.0;
        if (!model->BSIM4v5wlambdaGiven)
            model->BSIM4v5wlambda = 0.0;
        if (!model->BSIM4v5wvtlGiven)
            model->BSIM4v5wvtl = 0.0;  
        if (!model->BSIM4v5wxnGiven)
            model->BSIM4v5wxn = 0.0;  
        if (!model->BSIM4v5wvfbsdoffGiven)
            model->BSIM4v5wvfbsdoff = 0.0;   
        if (!model->BSIM4v5wtvfbsdoffGiven)
            model->BSIM4v5wtvfbsdoff = 0.0;  
        if (!model->BSIM4v5wtvoffGiven)
            model->BSIM4v5wtvoff = 0.0;  

        if (!model->BSIM4v5wcgslGiven)  
            model->BSIM4v5wcgsl = 0.0;
        if (!model->BSIM4v5wcgdlGiven)  
            model->BSIM4v5wcgdl = 0.0;
        if (!model->BSIM4v5wckappasGiven)  
            model->BSIM4v5wckappas = 0.0;
        if (!model->BSIM4v5wckappadGiven)
            model->BSIM4v5wckappad = 0.0;
        if (!model->BSIM4v5wcfGiven)  
            model->BSIM4v5wcf = 0.0;
        if (!model->BSIM4v5wclcGiven)  
            model->BSIM4v5wclc = 0.0;
        if (!model->BSIM4v5wcleGiven)  
            model->BSIM4v5wcle = 0.0;
        if (!model->BSIM4v5wvfbcvGiven)  
            model->BSIM4v5wvfbcv = 0.0;
        if (!model->BSIM4v5wacdeGiven)
            model->BSIM4v5wacde = 0.0;
        if (!model->BSIM4v5wmoinGiven)
            model->BSIM4v5wmoin = 0.0;
        if (!model->BSIM4v5wnoffGiven)
            model->BSIM4v5wnoff = 0.0;
        if (!model->BSIM4v5wvoffcvGiven)
            model->BSIM4v5wvoffcv = 0.0;

	/* Cross-term dependence */
        if (!model->BSIM4v5pcdscGiven)
	    model->BSIM4v5pcdsc = 0.0;
        if (!model->BSIM4v5pcdscbGiven)
	    model->BSIM4v5pcdscb = 0.0;   
        if (!model->BSIM4v5pcdscdGiven)
	    model->BSIM4v5pcdscd = 0.0;
        if (!model->BSIM4v5pcitGiven)
	    model->BSIM4v5pcit = 0.0;
        if (!model->BSIM4v5pnfactorGiven)
	    model->BSIM4v5pnfactor = 0.0;
        if (!model->BSIM4v5pxjGiven)
            model->BSIM4v5pxj = 0.0;
        if (!model->BSIM4v5pvsatGiven)
            model->BSIM4v5pvsat = 0.0;
        if (!model->BSIM4v5patGiven)
            model->BSIM4v5pat = 0.0;
        if (!model->BSIM4v5pa0Given)
            model->BSIM4v5pa0 = 0.0; 
            
        if (!model->BSIM4v5pagsGiven)
            model->BSIM4v5pags = 0.0;
        if (!model->BSIM4v5pa1Given)
            model->BSIM4v5pa1 = 0.0;
        if (!model->BSIM4v5pa2Given)
            model->BSIM4v5pa2 = 0.0;
        if (!model->BSIM4v5pketaGiven)
            model->BSIM4v5pketa = 0.0;
        if (!model->BSIM4v5pnsubGiven)
            model->BSIM4v5pnsub = 0.0;
        if (!model->BSIM4v5pndepGiven)
            model->BSIM4v5pndep = 0.0;
        if (!model->BSIM4v5pnsdGiven)
            model->BSIM4v5pnsd = 0.0;
        if (!model->BSIM4v5pphinGiven)
            model->BSIM4v5pphin = 0.0;
        if (!model->BSIM4v5pngateGiven)
            model->BSIM4v5pngate = 0.0;
        if (!model->BSIM4v5pvbmGiven)
	    model->BSIM4v5pvbm = 0.0;
        if (!model->BSIM4v5pxtGiven)
	    model->BSIM4v5pxt = 0.0;
        if (!model->BSIM4v5pkt1Given)
            model->BSIM4v5pkt1 = 0.0; 
        if (!model->BSIM4v5pkt1lGiven)
            model->BSIM4v5pkt1l = 0.0;
        if (!model->BSIM4v5pkt2Given)
            model->BSIM4v5pkt2 = 0.0;
        if (!model->BSIM4v5pk3Given)
            model->BSIM4v5pk3 = 0.0;      
        if (!model->BSIM4v5pk3bGiven)
            model->BSIM4v5pk3b = 0.0;      
        if (!model->BSIM4v5pw0Given)
            model->BSIM4v5pw0 = 0.0;    
        if (!model->BSIM4v5plpe0Given)
            model->BSIM4v5plpe0 = 0.0;
        if (!model->BSIM4v5plpebGiven)
            model->BSIM4v5plpeb = 0.0;
        if (!model->BSIM4v5pdvtp0Given)
            model->BSIM4v5pdvtp0 = 0.0;
        if (!model->BSIM4v5pdvtp1Given)
            model->BSIM4v5pdvtp1 = 0.0;
        if (!model->BSIM4v5pdvt0Given)
            model->BSIM4v5pdvt0 = 0.0;    
        if (!model->BSIM4v5pdvt1Given)
            model->BSIM4v5pdvt1 = 0.0;      
        if (!model->BSIM4v5pdvt2Given)
            model->BSIM4v5pdvt2 = 0.0;
        if (!model->BSIM4v5pdvt0wGiven)
            model->BSIM4v5pdvt0w = 0.0;    
        if (!model->BSIM4v5pdvt1wGiven)
            model->BSIM4v5pdvt1w = 0.0;      
        if (!model->BSIM4v5pdvt2wGiven)
            model->BSIM4v5pdvt2w = 0.0;
        if (!model->BSIM4v5pdroutGiven)
            model->BSIM4v5pdrout = 0.0;     
        if (!model->BSIM4v5pdsubGiven)
            model->BSIM4v5pdsub = 0.0;
        if (!model->BSIM4v5pvth0Given)
           model->BSIM4v5pvth0 = 0.0;
        if (!model->BSIM4v5puaGiven)
            model->BSIM4v5pua = 0.0;
        if (!model->BSIM4v5pua1Given)
            model->BSIM4v5pua1 = 0.0;
        if (!model->BSIM4v5pubGiven)
            model->BSIM4v5pub = 0.0;
        if (!model->BSIM4v5pub1Given)
            model->BSIM4v5pub1 = 0.0;
        if (!model->BSIM4v5pucGiven)
            model->BSIM4v5puc = 0.0;
        if (!model->BSIM4v5puc1Given)
            model->BSIM4v5puc1 = 0.0;
        if (!model->BSIM4v5pudGiven)
            model->BSIM4v5pud = 0.0;
        if (!model->BSIM4v5pud1Given)
            model->BSIM4v5pud1 = 0.0;
        if (!model->BSIM4v5pupGiven)
            model->BSIM4v5pup = 0.0;
        if (!model->BSIM4v5plpGiven)
            model->BSIM4v5plp = 0.0;
        if (!model->BSIM4v5pu0Given)
            model->BSIM4v5pu0 = 0.0;
        if (!model->BSIM4v5puteGiven)
	    model->BSIM4v5pute = 0.0;    
        if (!model->BSIM4v5pvoffGiven)
	    model->BSIM4v5pvoff = 0.0;
        if (!model->BSIM4v5pminvGiven)
            model->BSIM4v5pminv = 0.0;
        if (!model->BSIM4v5pfproutGiven)
            model->BSIM4v5pfprout = 0.0;
        if (!model->BSIM4v5ppditsGiven)
            model->BSIM4v5ppdits = 0.0;
        if (!model->BSIM4v5ppditsdGiven)
            model->BSIM4v5ppditsd = 0.0;
        if (!model->BSIM4v5pdeltaGiven)  
            model->BSIM4v5pdelta = 0.0;
        if (!model->BSIM4v5prdswGiven)
            model->BSIM4v5prdsw = 0.0;
        if (!model->BSIM4v5prdwGiven)
            model->BSIM4v5prdw = 0.0;
        if (!model->BSIM4v5prswGiven)
            model->BSIM4v5prsw = 0.0;
        if (!model->BSIM4v5pprwbGiven)
            model->BSIM4v5pprwb = 0.0;
        if (!model->BSIM4v5pprwgGiven)
            model->BSIM4v5pprwg = 0.0;
        if (!model->BSIM4v5pprtGiven)
            model->BSIM4v5pprt = 0.0;
        if (!model->BSIM4v5peta0Given)
            model->BSIM4v5peta0 = 0.0;
        if (!model->BSIM4v5petabGiven)
            model->BSIM4v5petab = 0.0;
        if (!model->BSIM4v5ppclmGiven)
            model->BSIM4v5ppclm = 0.0; 
        if (!model->BSIM4v5ppdibl1Given)
            model->BSIM4v5ppdibl1 = 0.0;
        if (!model->BSIM4v5ppdibl2Given)
            model->BSIM4v5ppdibl2 = 0.0;
        if (!model->BSIM4v5ppdiblbGiven)
            model->BSIM4v5ppdiblb = 0.0;
        if (!model->BSIM4v5ppscbe1Given)
            model->BSIM4v5ppscbe1 = 0.0;
        if (!model->BSIM4v5ppscbe2Given)
            model->BSIM4v5ppscbe2 = 0.0;
        if (!model->BSIM4v5ppvagGiven)
            model->BSIM4v5ppvag = 0.0;     
        if (!model->BSIM4v5pwrGiven)  
            model->BSIM4v5pwr = 0.0;
        if (!model->BSIM4v5pdwgGiven)  
            model->BSIM4v5pdwg = 0.0;
        if (!model->BSIM4v5pdwbGiven)  
            model->BSIM4v5pdwb = 0.0;
        if (!model->BSIM4v5pb0Given)
            model->BSIM4v5pb0 = 0.0;
        if (!model->BSIM4v5pb1Given)  
            model->BSIM4v5pb1 = 0.0;
        if (!model->BSIM4v5palpha0Given)  
            model->BSIM4v5palpha0 = 0.0;
        if (!model->BSIM4v5palpha1Given)
            model->BSIM4v5palpha1 = 0.0;
        if (!model->BSIM4v5pbeta0Given)  
            model->BSIM4v5pbeta0 = 0.0;
        if (!model->BSIM4v5pagidlGiven)
            model->BSIM4v5pagidl = 0.0;
        if (!model->BSIM4v5pbgidlGiven)
            model->BSIM4v5pbgidl = 0.0;
        if (!model->BSIM4v5pcgidlGiven)
            model->BSIM4v5pcgidl = 0.0;
        if (!model->BSIM4v5pegidlGiven)
            model->BSIM4v5pegidl = 0.0;
        if (!model->BSIM4v5paigcGiven)
            model->BSIM4v5paigc = 0.0;
        if (!model->BSIM4v5pbigcGiven)
            model->BSIM4v5pbigc = 0.0;
        if (!model->BSIM4v5pcigcGiven)
            model->BSIM4v5pcigc = 0.0;
        if (!model->BSIM4v5paigsdGiven)
            model->BSIM4v5paigsd = 0.0;
        if (!model->BSIM4v5pbigsdGiven)
            model->BSIM4v5pbigsd = 0.0;
        if (!model->BSIM4v5pcigsdGiven)
            model->BSIM4v5pcigsd = 0.0;
        if (!model->BSIM4v5paigbaccGiven)
            model->BSIM4v5paigbacc = 0.0;
        if (!model->BSIM4v5pbigbaccGiven)
            model->BSIM4v5pbigbacc = 0.0;
        if (!model->BSIM4v5pcigbaccGiven)
            model->BSIM4v5pcigbacc = 0.0;
        if (!model->BSIM4v5paigbinvGiven)
            model->BSIM4v5paigbinv = 0.0;
        if (!model->BSIM4v5pbigbinvGiven)
            model->BSIM4v5pbigbinv = 0.0;
        if (!model->BSIM4v5pcigbinvGiven)
            model->BSIM4v5pcigbinv = 0.0;
        if (!model->BSIM4v5pnigcGiven)
            model->BSIM4v5pnigc = 0.0;
        if (!model->BSIM4v5pnigbinvGiven)
            model->BSIM4v5pnigbinv = 0.0;
        if (!model->BSIM4v5pnigbaccGiven)
            model->BSIM4v5pnigbacc = 0.0;
        if (!model->BSIM4v5pntoxGiven)
            model->BSIM4v5pntox = 0.0;
        if (!model->BSIM4v5peigbinvGiven)
            model->BSIM4v5peigbinv = 0.0;
        if (!model->BSIM4v5ppigcdGiven)
            model->BSIM4v5ppigcd = 0.0;
        if (!model->BSIM4v5ppoxedgeGiven)
            model->BSIM4v5ppoxedge = 0.0;
        if (!model->BSIM4v5pxrcrg1Given)
            model->BSIM4v5pxrcrg1 = 0.0;
        if (!model->BSIM4v5pxrcrg2Given)
            model->BSIM4v5pxrcrg2 = 0.0;
        if (!model->BSIM4v5peuGiven)
            model->BSIM4v5peu = 0.0;
        if (!model->BSIM4v5pvfbGiven)
            model->BSIM4v5pvfb = 0.0;
        if (!model->BSIM4v5plambdaGiven)
            model->BSIM4v5plambda = 0.0;
        if (!model->BSIM4v5pvtlGiven)
            model->BSIM4v5pvtl = 0.0;  
        if (!model->BSIM4v5pxnGiven)
            model->BSIM4v5pxn = 0.0;  
        if (!model->BSIM4v5pvfbsdoffGiven)
            model->BSIM4v5pvfbsdoff = 0.0;   
        if (!model->BSIM4v5ptvfbsdoffGiven)
            model->BSIM4v5ptvfbsdoff = 0.0;  
        if (!model->BSIM4v5ptvoffGiven)
            model->BSIM4v5ptvoff = 0.0;  

        if (!model->BSIM4v5pcgslGiven)  
            model->BSIM4v5pcgsl = 0.0;
        if (!model->BSIM4v5pcgdlGiven)  
            model->BSIM4v5pcgdl = 0.0;
        if (!model->BSIM4v5pckappasGiven)  
            model->BSIM4v5pckappas = 0.0;
        if (!model->BSIM4v5pckappadGiven)
            model->BSIM4v5pckappad = 0.0;
        if (!model->BSIM4v5pcfGiven)  
            model->BSIM4v5pcf = 0.0;
        if (!model->BSIM4v5pclcGiven)  
            model->BSIM4v5pclc = 0.0;
        if (!model->BSIM4v5pcleGiven)  
            model->BSIM4v5pcle = 0.0;
        if (!model->BSIM4v5pvfbcvGiven)  
            model->BSIM4v5pvfbcv = 0.0;
        if (!model->BSIM4v5pacdeGiven)
            model->BSIM4v5pacde = 0.0;
        if (!model->BSIM4v5pmoinGiven)
            model->BSIM4v5pmoin = 0.0;
        if (!model->BSIM4v5pnoffGiven)
            model->BSIM4v5pnoff = 0.0;
        if (!model->BSIM4v5pvoffcvGiven)
            model->BSIM4v5pvoffcv = 0.0;

        if (!model->BSIM4v5gamma1Given)
            model->BSIM4v5gamma1 = 0.0;
        if (!model->BSIM4v5lgamma1Given)
            model->BSIM4v5lgamma1 = 0.0;
        if (!model->BSIM4v5wgamma1Given)
            model->BSIM4v5wgamma1 = 0.0;
        if (!model->BSIM4v5pgamma1Given)
            model->BSIM4v5pgamma1 = 0.0;
        if (!model->BSIM4v5gamma2Given)
            model->BSIM4v5gamma2 = 0.0;
        if (!model->BSIM4v5lgamma2Given)
            model->BSIM4v5lgamma2 = 0.0;
        if (!model->BSIM4v5wgamma2Given)
            model->BSIM4v5wgamma2 = 0.0;
        if (!model->BSIM4v5pgamma2Given)
            model->BSIM4v5pgamma2 = 0.0;
        if (!model->BSIM4v5vbxGiven)
            model->BSIM4v5vbx = 0.0;
        if (!model->BSIM4v5lvbxGiven)
            model->BSIM4v5lvbx = 0.0;
        if (!model->BSIM4v5wvbxGiven)
            model->BSIM4v5wvbx = 0.0;
        if (!model->BSIM4v5pvbxGiven)
            model->BSIM4v5pvbx = 0.0;

        /* unit degree celcius */
        if (!model->BSIM4v5tnomGiven)  
	    model->BSIM4v5tnom = ckt->CKTnomTemp; 
        if (!model->BSIM4v5LintGiven)  
           model->BSIM4v5Lint = 0.0;
        if (!model->BSIM4v5LlGiven)  
           model->BSIM4v5Ll = 0.0;
        if (!model->BSIM4v5LlcGiven)
           model->BSIM4v5Llc = model->BSIM4v5Ll;
        if (!model->BSIM4v5LlnGiven)  
           model->BSIM4v5Lln = 1.0;
        if (!model->BSIM4v5LwGiven)  
           model->BSIM4v5Lw = 0.0;
        if (!model->BSIM4v5LwcGiven)
           model->BSIM4v5Lwc = model->BSIM4v5Lw;
        if (!model->BSIM4v5LwnGiven)  
           model->BSIM4v5Lwn = 1.0;
        if (!model->BSIM4v5LwlGiven)  
           model->BSIM4v5Lwl = 0.0;
        if (!model->BSIM4v5LwlcGiven)
           model->BSIM4v5Lwlc = model->BSIM4v5Lwl;
        if (!model->BSIM4v5LminGiven)  
           model->BSIM4v5Lmin = 0.0;
        if (!model->BSIM4v5LmaxGiven)  
           model->BSIM4v5Lmax = 1.0;
        if (!model->BSIM4v5WintGiven)  
           model->BSIM4v5Wint = 0.0;
        if (!model->BSIM4v5WlGiven)  
           model->BSIM4v5Wl = 0.0;
        if (!model->BSIM4v5WlcGiven)
           model->BSIM4v5Wlc = model->BSIM4v5Wl;
        if (!model->BSIM4v5WlnGiven)  
           model->BSIM4v5Wln = 1.0;
        if (!model->BSIM4v5WwGiven)  
           model->BSIM4v5Ww = 0.0;
        if (!model->BSIM4v5WwcGiven)
           model->BSIM4v5Wwc = model->BSIM4v5Ww;
        if (!model->BSIM4v5WwnGiven)  
           model->BSIM4v5Wwn = 1.0;
        if (!model->BSIM4v5WwlGiven)  
           model->BSIM4v5Wwl = 0.0;
        if (!model->BSIM4v5WwlcGiven)
           model->BSIM4v5Wwlc = model->BSIM4v5Wwl;
        if (!model->BSIM4v5WminGiven)  
           model->BSIM4v5Wmin = 0.0;
        if (!model->BSIM4v5WmaxGiven)  
           model->BSIM4v5Wmax = 1.0;
        if (!model->BSIM4v5dwcGiven)  
           model->BSIM4v5dwc = model->BSIM4v5Wint;
        if (!model->BSIM4v5dlcGiven)  
           model->BSIM4v5dlc = model->BSIM4v5Lint;
        if (!model->BSIM4v5xlGiven)  
           model->BSIM4v5xl = 0.0;
        if (!model->BSIM4v5xwGiven)  
           model->BSIM4v5xw = 0.0;
        if (!model->BSIM4v5dlcigGiven)
           model->BSIM4v5dlcig = model->BSIM4v5Lint;
        if (!model->BSIM4v5dwjGiven)
           model->BSIM4v5dwj = model->BSIM4v5dwc;
	if (!model->BSIM4v5cfGiven)
           model->BSIM4v5cf = 2.0 * model->BSIM4v5epsrox * EPS0 / PI
		          * log(1.0 + 0.4e-6 / model->BSIM4v5toxe);

        if (!model->BSIM4v5xpartGiven)
            model->BSIM4v5xpart = 0.0;
        if (!model->BSIM4v5sheetResistanceGiven)
            model->BSIM4v5sheetResistance = 0.0;

        if (!model->BSIM4v5SunitAreaJctCapGiven)
            model->BSIM4v5SunitAreaJctCap = 5.0E-4;
        if (!model->BSIM4v5DunitAreaJctCapGiven)
            model->BSIM4v5DunitAreaJctCap = model->BSIM4v5SunitAreaJctCap;
        if (!model->BSIM4v5SunitLengthSidewallJctCapGiven)
            model->BSIM4v5SunitLengthSidewallJctCap = 5.0E-10;
        if (!model->BSIM4v5DunitLengthSidewallJctCapGiven)
            model->BSIM4v5DunitLengthSidewallJctCap = model->BSIM4v5SunitLengthSidewallJctCap;
        if (!model->BSIM4v5SunitLengthGateSidewallJctCapGiven)
            model->BSIM4v5SunitLengthGateSidewallJctCap = model->BSIM4v5SunitLengthSidewallJctCap ;
        if (!model->BSIM4v5DunitLengthGateSidewallJctCapGiven)
            model->BSIM4v5DunitLengthGateSidewallJctCap = model->BSIM4v5SunitLengthGateSidewallJctCap;
        if (!model->BSIM4v5SjctSatCurDensityGiven)
            model->BSIM4v5SjctSatCurDensity = 1.0E-4;
        if (!model->BSIM4v5DjctSatCurDensityGiven)
            model->BSIM4v5DjctSatCurDensity = model->BSIM4v5SjctSatCurDensity;
        if (!model->BSIM4v5SjctSidewallSatCurDensityGiven)
            model->BSIM4v5SjctSidewallSatCurDensity = 0.0;
        if (!model->BSIM4v5DjctSidewallSatCurDensityGiven)
            model->BSIM4v5DjctSidewallSatCurDensity = model->BSIM4v5SjctSidewallSatCurDensity;
        if (!model->BSIM4v5SjctGateSidewallSatCurDensityGiven)
            model->BSIM4v5SjctGateSidewallSatCurDensity = 0.0;
        if (!model->BSIM4v5DjctGateSidewallSatCurDensityGiven)
            model->BSIM4v5DjctGateSidewallSatCurDensity = model->BSIM4v5SjctGateSidewallSatCurDensity;
        if (!model->BSIM4v5SbulkJctPotentialGiven)
            model->BSIM4v5SbulkJctPotential = 1.0;
        if (!model->BSIM4v5DbulkJctPotentialGiven)
            model->BSIM4v5DbulkJctPotential = model->BSIM4v5SbulkJctPotential;
        if (!model->BSIM4v5SsidewallJctPotentialGiven)
            model->BSIM4v5SsidewallJctPotential = 1.0;
        if (!model->BSIM4v5DsidewallJctPotentialGiven)
            model->BSIM4v5DsidewallJctPotential = model->BSIM4v5SsidewallJctPotential;
        if (!model->BSIM4v5SGatesidewallJctPotentialGiven)
            model->BSIM4v5SGatesidewallJctPotential = model->BSIM4v5SsidewallJctPotential;
        if (!model->BSIM4v5DGatesidewallJctPotentialGiven)
            model->BSIM4v5DGatesidewallJctPotential = model->BSIM4v5SGatesidewallJctPotential;
        if (!model->BSIM4v5SbulkJctBotGradingCoeffGiven)
            model->BSIM4v5SbulkJctBotGradingCoeff = 0.5;
        if (!model->BSIM4v5DbulkJctBotGradingCoeffGiven)
            model->BSIM4v5DbulkJctBotGradingCoeff = model->BSIM4v5SbulkJctBotGradingCoeff;
        if (!model->BSIM4v5SbulkJctSideGradingCoeffGiven)
            model->BSIM4v5SbulkJctSideGradingCoeff = 0.33;
        if (!model->BSIM4v5DbulkJctSideGradingCoeffGiven)
            model->BSIM4v5DbulkJctSideGradingCoeff = model->BSIM4v5SbulkJctSideGradingCoeff;
        if (!model->BSIM4v5SbulkJctGateSideGradingCoeffGiven)
            model->BSIM4v5SbulkJctGateSideGradingCoeff = model->BSIM4v5SbulkJctSideGradingCoeff;
        if (!model->BSIM4v5DbulkJctGateSideGradingCoeffGiven)
            model->BSIM4v5DbulkJctGateSideGradingCoeff = model->BSIM4v5SbulkJctGateSideGradingCoeff;
        if (!model->BSIM4v5SjctEmissionCoeffGiven)
            model->BSIM4v5SjctEmissionCoeff = 1.0;
        if (!model->BSIM4v5DjctEmissionCoeffGiven)
            model->BSIM4v5DjctEmissionCoeff = model->BSIM4v5SjctEmissionCoeff;
        if (!model->BSIM4v5SjctTempExponentGiven)
            model->BSIM4v5SjctTempExponent = 3.0;
        if (!model->BSIM4v5DjctTempExponentGiven)
            model->BSIM4v5DjctTempExponent = model->BSIM4v5SjctTempExponent;

        if (!model->BSIM4v5jtssGiven)
            model->BSIM4v5jtss = 0.0;
        if (!model->BSIM4v5jtsdGiven)
            model->BSIM4v5jtsd = model->BSIM4v5jtss;
        if (!model->BSIM4v5jtsswsGiven)
            model->BSIM4v5jtssws = 0.0;
        if (!model->BSIM4v5jtsswdGiven)
            model->BSIM4v5jtsswd = model->BSIM4v5jtssws;
        if (!model->BSIM4v5jtsswgsGiven)
            model->BSIM4v5jtsswgs = 0.0;
        if (!model->BSIM4v5jtsswgdGiven)
            model->BSIM4v5jtsswgd = model->BSIM4v5jtsswgs;
        if (!model->BSIM4v5njtsGiven)
            model->BSIM4v5njts = 20.0;
        if (!model->BSIM4v5njtsswGiven)
            model->BSIM4v5njtssw = 20.0;
        if (!model->BSIM4v5njtsswgGiven)
            model->BSIM4v5njtsswg = 20.0;
        if (!model->BSIM4v5xtssGiven)
            model->BSIM4v5xtss = 0.02;
        if (!model->BSIM4v5xtsdGiven)
            model->BSIM4v5xtsd = model->BSIM4v5xtss;
        if (!model->BSIM4v5xtsswsGiven)
            model->BSIM4v5xtssws = 0.02;
        if (!model->BSIM4v5xtsswdGiven)
            model->BSIM4v5xtsswd = model->BSIM4v5xtssws;
        if (!model->BSIM4v5xtsswgsGiven)
            model->BSIM4v5xtsswgs = 0.02;
        if (!model->BSIM4v5xtsswgdGiven)
            model->BSIM4v5xtsswgd = model->BSIM4v5xtsswgs;
        if (!model->BSIM4v5tnjtsGiven)
            model->BSIM4v5tnjts = 0.0;
        if (!model->BSIM4v5tnjtsswGiven)
            model->BSIM4v5tnjtssw = 0.0;
        if (!model->BSIM4v5tnjtsswgGiven)
            model->BSIM4v5tnjtsswg = 0.0;
        if (!model->BSIM4v5vtssGiven)
            model->BSIM4v5vtss = 10.0;
        if (!model->BSIM4v5vtsdGiven)
            model->BSIM4v5vtsd = model->BSIM4v5vtss;
        if (!model->BSIM4v5vtsswsGiven)
            model->BSIM4v5vtssws = 10.0;
        if (!model->BSIM4v5vtsswdGiven)
            model->BSIM4v5vtsswd = model->BSIM4v5vtssws;
        if (!model->BSIM4v5vtsswgsGiven)
            model->BSIM4v5vtsswgs = 10.0;
        if (!model->BSIM4v5vtsswgdGiven)
            model->BSIM4v5vtsswgd = model->BSIM4v5vtsswgs;

        if (!model->BSIM4v5oxideTrapDensityAGiven)
	{   if (model->BSIM4v5type == NMOS)
                model->BSIM4v5oxideTrapDensityA = 6.25e41;
            else
                model->BSIM4v5oxideTrapDensityA= 6.188e40;
	}
        if (!model->BSIM4v5oxideTrapDensityBGiven)
	{   if (model->BSIM4v5type == NMOS)
                model->BSIM4v5oxideTrapDensityB = 3.125e26;
            else
                model->BSIM4v5oxideTrapDensityB = 1.5e25;
	}
        if (!model->BSIM4v5oxideTrapDensityCGiven)
            model->BSIM4v5oxideTrapDensityC = 8.75e9;
        if (!model->BSIM4v5emGiven)
            model->BSIM4v5em = 4.1e7; /* V/m */
        if (!model->BSIM4v5efGiven)
            model->BSIM4v5ef = 1.0;
        if (!model->BSIM4v5afGiven)
            model->BSIM4v5af = 1.0;
        if (!model->BSIM4v5kfGiven)
            model->BSIM4v5kf = 0.0;

        if (!model->BSIM4v5vgsMaxGiven)
            model->BSIM4v5vgsMax = 1e99;
        if (!model->BSIM4v5vgdMaxGiven)
            model->BSIM4v5vgdMax = 1e99;
        if (!model->BSIM4v5vgbMaxGiven)
            model->BSIM4v5vgbMax = 1e99;
        if (!model->BSIM4v5vdsMaxGiven)
            model->BSIM4v5vdsMax = 1e99;
        if (!model->BSIM4v5vbsMaxGiven)
            model->BSIM4v5vbsMax = 1e99;
        if (!model->BSIM4v5vbdMaxGiven)
            model->BSIM4v5vbdMax = 1e99;
        if (!model->BSIM4v5vgsrMaxGiven)
            model->BSIM4v5vgsrMax = 1e99;
        if (!model->BSIM4v5vgdrMaxGiven)
            model->BSIM4v5vgdrMax = 1e99;
        if (!model->BSIM4v5vgbrMaxGiven)
            model->BSIM4v5vgbrMax = 1e99;
        if (!model->BSIM4v5vbsrMaxGiven)
            model->BSIM4v5vbsrMax = 1e99;
        if (!model->BSIM4v5vbdrMaxGiven)
            model->BSIM4v5vbdrMax = 1e99;

        /* stress effect */
        if (!model->BSIM4v5sarefGiven)
            model->BSIM4v5saref = 1e-6; /* m */
        if (!model->BSIM4v5sbrefGiven)
            model->BSIM4v5sbref = 1e-6;  /* m */
        if (!model->BSIM4v5wlodGiven)
            model->BSIM4v5wlod = 0;  /* m */
        if (!model->BSIM4v5ku0Given)
            model->BSIM4v5ku0 = 0; /* 1/m */
        if (!model->BSIM4v5kvsatGiven)
            model->BSIM4v5kvsat = 0;
        if (!model->BSIM4v5kvth0Given) /* m */
            model->BSIM4v5kvth0 = 0;
        if (!model->BSIM4v5tku0Given)
            model->BSIM4v5tku0 = 0;
        if (!model->BSIM4v5llodku0Given)
            model->BSIM4v5llodku0 = 0;
        if (!model->BSIM4v5wlodku0Given)
            model->BSIM4v5wlodku0 = 0;
        if (!model->BSIM4v5llodvthGiven)
            model->BSIM4v5llodvth = 0;
        if (!model->BSIM4v5wlodvthGiven)
            model->BSIM4v5wlodvth = 0;
        if (!model->BSIM4v5lku0Given)
            model->BSIM4v5lku0 = 0;
        if (!model->BSIM4v5wku0Given)
            model->BSIM4v5wku0 = 0;
        if (!model->BSIM4v5pku0Given)
            model->BSIM4v5pku0 = 0;
        if (!model->BSIM4v5lkvth0Given)
            model->BSIM4v5lkvth0 = 0;
        if (!model->BSIM4v5wkvth0Given)
            model->BSIM4v5wkvth0 = 0;
        if (!model->BSIM4v5pkvth0Given)
            model->BSIM4v5pkvth0 = 0;
        if (!model->BSIM4v5stk2Given)
            model->BSIM4v5stk2 = 0;
        if (!model->BSIM4v5lodk2Given)
            model->BSIM4v5lodk2 = 1.0;
        if (!model->BSIM4v5steta0Given)
            model->BSIM4v5steta0 = 0;
        if (!model->BSIM4v5lodeta0Given)
            model->BSIM4v5lodeta0 = 1.0;

	/* Well Proximity Effect  */
        if (!model->BSIM4v5webGiven)
            model->BSIM4v5web = 0.0; 
        if (!model->BSIM4v5wecGiven)
            model->BSIM4v5wec = 0.0;
        if (!model->BSIM4v5kvth0weGiven)
            model->BSIM4v5kvth0we = 0.0; 
        if (!model->BSIM4v5k2weGiven)
            model->BSIM4v5k2we = 0.0; 
        if (!model->BSIM4v5ku0weGiven)
            model->BSIM4v5ku0we = 0.0; 
        if (!model->BSIM4v5screfGiven)
            model->BSIM4v5scref = 1.0E-6; /* m */
        if (!model->BSIM4v5wpemodGiven)
            model->BSIM4v5wpemod = 0; 
        else if ((model->BSIM4v5wpemod != 0) && (model->BSIM4v5wpemod != 1))
        {   model->BSIM4v5wpemod = 0;
            printf("Warning: wpemod has been set to its default value: 0.\n");
        }
        if (!model->BSIM4v5lkvth0weGiven)
            model->BSIM4v5lkvth0we = 0; 
        if (!model->BSIM4v5lk2weGiven)
            model->BSIM4v5lk2we = 0;
        if (!model->BSIM4v5lku0weGiven)
            model->BSIM4v5lku0we = 0;
        if (!model->BSIM4v5wkvth0weGiven)
            model->BSIM4v5wkvth0we = 0; 
        if (!model->BSIM4v5wk2weGiven)
            model->BSIM4v5wk2we = 0;
        if (!model->BSIM4v5wku0weGiven)
            model->BSIM4v5wku0we = 0;
        if (!model->BSIM4v5pkvth0weGiven)
            model->BSIM4v5pkvth0we = 0; 
        if (!model->BSIM4v5pk2weGiven)
            model->BSIM4v5pk2we = 0;
        if (!model->BSIM4v5pku0weGiven)
            model->BSIM4v5pku0we = 0;

        DMCGeff = model->BSIM4v5dmcg - model->BSIM4v5dmcgt;
        DMCIeff = model->BSIM4v5dmci;
        DMDGeff = model->BSIM4v5dmdg - model->BSIM4v5dmcgt;

	/*
         * End processing models and begin to loop
         * through all the instances of the model
         */

        for (here = BSIM4v5instances(model); here != NULL ;
             here=BSIM4v5nextInstance(here)) 
        {   
            /* allocate a chunk of the state vector */
            here->BSIM4v5states = *states;
            *states += BSIM4v5numStates;

            /* perform the parameter defaulting */
            if (!here->BSIM4v5lGiven)
                here->BSIM4v5l = 5.0e-6;
            if (!here->BSIM4v5wGiven)
                here->BSIM4v5w = 5.0e-6;
            if (!here->BSIM4v5mGiven)
                here->BSIM4v5m = 1.0;
            if (!here->BSIM4v5nfGiven)
                here->BSIM4v5nf = 1.0;
            if (!here->BSIM4v5minGiven)
                here->BSIM4v5min = 0; /* integer */
            if (!here->BSIM4v5icVDSGiven)
                here->BSIM4v5icVDS = 0.0;
            if (!here->BSIM4v5icVGSGiven)
                here->BSIM4v5icVGS = 0.0;
            if (!here->BSIM4v5icVBSGiven)
                here->BSIM4v5icVBS = 0.0;
            if (!here->BSIM4v5drainAreaGiven)
                here->BSIM4v5drainArea = 0.0;
            if (!here->BSIM4v5drainPerimeterGiven)
                here->BSIM4v5drainPerimeter = 0.0;
            if (!here->BSIM4v5drainSquaresGiven)
                here->BSIM4v5drainSquares = 1.0;
            if (!here->BSIM4v5sourceAreaGiven)
                here->BSIM4v5sourceArea = 0.0;
            if (!here->BSIM4v5sourcePerimeterGiven)
                here->BSIM4v5sourcePerimeter = 0.0;
            if (!here->BSIM4v5sourceSquaresGiven)
                here->BSIM4v5sourceSquares = 1.0;

            if (!here->BSIM4v5saGiven)
                here->BSIM4v5sa = 0.0;
            if (!here->BSIM4v5sbGiven)
                here->BSIM4v5sb = 0.0;
            if (!here->BSIM4v5sdGiven)
                here->BSIM4v5sd = 0.0;

            if (!here->BSIM4v5rbdbGiven)
                here->BSIM4v5rbdb = model->BSIM4v5rbdb; /* in ohm */
            if (!here->BSIM4v5rbsbGiven)
                here->BSIM4v5rbsb = model->BSIM4v5rbsb;
            if (!here->BSIM4v5rbpbGiven)
                here->BSIM4v5rbpb = model->BSIM4v5rbpb;
            if (!here->BSIM4v5rbpsGiven)
                here->BSIM4v5rbps = model->BSIM4v5rbps;
            if (!here->BSIM4v5rbpdGiven)
                here->BSIM4v5rbpd = model->BSIM4v5rbpd;
            if (!here->BSIM4v5delvtoGiven)
                here->BSIM4v5delvto = 0.0;
            if (!here->BSIM4v5mulu0Given)
                here->BSIM4v5mulu0 = 1.0;
            if (!here->BSIM4v5xgwGiven)
                here->BSIM4v5xgw = model->BSIM4v5xgw;
            if (!here->BSIM4v5ngconGiven)
                here->BSIM4v5ngcon = model->BSIM4v5ngcon;

                    
            /* Process instance model selectors, some
             * may override their global counterparts
	     */
            if (!here->BSIM4v5rbodyModGiven)
                here->BSIM4v5rbodyMod = model->BSIM4v5rbodyMod;
            else if ((here->BSIM4v5rbodyMod != 0) && (here->BSIM4v5rbodyMod != 1) && (here->BSIM4v5rbodyMod != 2))
            {   here->BSIM4v5rbodyMod = model->BSIM4v5rbodyMod;
                printf("Warning: rbodyMod has been set to its global value %d.\n",
	        model->BSIM4v5rbodyMod);
            }

            if (!here->BSIM4v5rgateModGiven)
                here->BSIM4v5rgateMod = model->BSIM4v5rgateMod;
            else if ((here->BSIM4v5rgateMod != 0) && (here->BSIM4v5rgateMod != 1)
	        && (here->BSIM4v5rgateMod != 2) && (here->BSIM4v5rgateMod != 3))
            {   here->BSIM4v5rgateMod = model->BSIM4v5rgateMod;
                printf("Warning: rgateMod has been set to its global value %d.\n",
                model->BSIM4v5rgateMod);
            }

            if (!here->BSIM4v5geoModGiven)
                here->BSIM4v5geoMod = model->BSIM4v5geoMod;

            if (!here->BSIM4v5rgeoModGiven)
                here->BSIM4v5rgeoMod = model->BSIM4v5rgeoMod;
            else if ((here->BSIM4v5rgeoMod != 0) && (here->BSIM4v5rgeoMod != 1))
            {   here->BSIM4v5rgeoMod = model->BSIM4v5rgeoMod;
                printf("Warning: rgeoMod has been set to its global value %d.\n",
                model->BSIM4v5rgeoMod);
            }
            if (!here->BSIM4v5trnqsModGiven)
                here->BSIM4v5trnqsMod = model->BSIM4v5trnqsMod;
            else if ((here->BSIM4v5trnqsMod != 0) && (here->BSIM4v5trnqsMod != 1))
            {   here->BSIM4v5trnqsMod = model->BSIM4v5trnqsMod;
                printf("Warning: trnqsMod has been set to its global value %d.\n",
                model->BSIM4v5trnqsMod);
            }

            if (!here->BSIM4v5acnqsModGiven)
                here->BSIM4v5acnqsMod = model->BSIM4v5acnqsMod;
            else if ((here->BSIM4v5acnqsMod != 0) && (here->BSIM4v5acnqsMod != 1))
            {   here->BSIM4v5acnqsMod = model->BSIM4v5acnqsMod;
                printf("Warning: acnqsMod has been set to its global value %d.\n",
                model->BSIM4v5acnqsMod);
            }

            /* stress effect */
            if (!here->BSIM4v5saGiven)
                here->BSIM4v5sa = 0.0;
            if (!here->BSIM4v5sbGiven)
                here->BSIM4v5sb = 0.0;
            if (!here->BSIM4v5sdGiven)
                here->BSIM4v5sd = 2 * model->BSIM4v5dmcg;
	    /* Well Proximity Effect  */
	    if (!here->BSIM4v5scaGiven)
                here->BSIM4v5sca = 0.0;
	    if (!here->BSIM4v5scbGiven)
                here->BSIM4v5scb = 0.0;
	    if (!here->BSIM4v5sccGiven)
                here->BSIM4v5scc = 0.0;
	    if (!here->BSIM4v5scGiven)
                here->BSIM4v5sc = 0.0; /* m */

            /* process drain series resistance */
            createNode = 0;
            if ( (model->BSIM4v5rdsMod != 0)
                            || (model->BSIM4v5tnoiMod != 0 && noiseAnalGiven))
            {
               createNode = 1;
            } else if (model->BSIM4v5sheetResistance > 0)
            {
                     if (here->BSIM4v5drainSquaresGiven
                                       && here->BSIM4v5drainSquares > 0)
                     {
                          createNode = 1;
                     } else if (!here->BSIM4v5drainSquaresGiven
                                       && (here->BSIM4v5rgeoMod != 0))
                     {
                          BSIM4v5RdseffGeo(here->BSIM4v5nf*here->BSIM4v5m, here->BSIM4v5geoMod,
                                  here->BSIM4v5rgeoMod, here->BSIM4v5min,
                                  here->BSIM4v5w, model->BSIM4v5sheetResistance,
                                  DMCGeff, DMCIeff, DMDGeff, 0, &Rtot);
                          if(Rtot > 0)
                             createNode = 1;
                     }
            }
            if ( createNode != 0 )
            {   if (here->BSIM4v5dNodePrime == 0)
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v5name,"drain");
                if(error) return(error);
                here->BSIM4v5dNodePrime = tmp->number;

                if (ckt->CKTcopyNodesets) {
	          CKTnode *tmpNode;
		  IFuid tmpName;    
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
            {   here->BSIM4v5dNodePrime = here->BSIM4v5dNode;
            }
            
            /* process source series resistance */
            createNode = 0;
            if ( (model->BSIM4v5rdsMod != 0)
                            || (model->BSIM4v5tnoiMod != 0 && noiseAnalGiven))
            {
               createNode = 1;
            } else if (model->BSIM4v5sheetResistance > 0)
            {
                     if (here->BSIM4v5sourceSquaresGiven
                                        && here->BSIM4v5sourceSquares > 0)
                     {
                          createNode = 1;
                     } else if (!here->BSIM4v5sourceSquaresGiven
                                        && (here->BSIM4v5rgeoMod != 0))
                     {
                          BSIM4v5RdseffGeo(here->BSIM4v5nf*here->BSIM4v5m, here->BSIM4v5geoMod,
                                  here->BSIM4v5rgeoMod, here->BSIM4v5min,
                                  here->BSIM4v5w, model->BSIM4v5sheetResistance,
                                  DMCGeff, DMCIeff, DMDGeff, 1, &Rtot);
                          if(Rtot > 0)
                             createNode = 1;
                     }
            }
            if ( createNode != 0 )
            {   if (here->BSIM4v5sNodePrime == 0)
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v5name,"source");
                if(error) return(error);
                here->BSIM4v5sNodePrime = tmp->number;
		
		if (ckt->CKTcopyNodesets) {
		  CKTnode *tmpNode;
		  IFuid tmpName;
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
                here->BSIM4v5sNodePrime = here->BSIM4v5sNode;

            if (here->BSIM4v5rgateMod > 0)
            {   if (here->BSIM4v5gNodePrime == 0)
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v5name,"gate");
                if(error) return(error);
                   here->BSIM4v5gNodePrime = tmp->number;
		   
		   if (ckt->CKTcopyNodesets) {
		  CKTnode *tmpNode;
		  IFuid tmpName;
                  if (CKTinst2Node(ckt,here,2,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }
						
            }
            }
            else
                here->BSIM4v5gNodePrime = here->BSIM4v5gNodeExt;

            if (here->BSIM4v5rgateMod == 3)
            {   if (here->BSIM4v5gNodeMid == 0)
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v5name,"midgate");
                if(error) return(error);
                   here->BSIM4v5gNodeMid = tmp->number;
            }
            }
            else
                here->BSIM4v5gNodeMid = here->BSIM4v5gNodeExt;
            

            /* internal body nodes for body resistance model */
            if ((here->BSIM4v5rbodyMod ==1) || (here->BSIM4v5rbodyMod ==2))
            {   if (here->BSIM4v5dbNode == 0)
		{   error = CKTmkVolt(ckt,&tmp,here->BSIM4v5name,"dbody");
                    if(error) return(error);
                    here->BSIM4v5dbNode = tmp->number;
		    
		    if (ckt->CKTcopyNodesets) {
		  CKTnode *tmpNode;
		  IFuid tmpName;
                  if (CKTinst2Node(ckt,here,4,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }				
		}
		if (here->BSIM4v5bNodePrime == 0)
                {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v5name,"body");
                    if(error) return(error);
                    here->BSIM4v5bNodePrime = tmp->number;
                }
		if (here->BSIM4v5sbNode == 0)
                {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v5name,"sbody");
                    if(error) return(error);
                    here->BSIM4v5sbNode = tmp->number;
                }
            }
	    else
	        here->BSIM4v5dbNode = here->BSIM4v5bNodePrime = here->BSIM4v5sbNode
				  = here->BSIM4v5bNode;

            /* NQS node */
            if (here->BSIM4v5trnqsMod)
            {   if (here->BSIM4v5qNode == 0)
	    {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v5name,"charge");
                if(error) return(error);
                here->BSIM4v5qNode = tmp->number;
		
		if (ckt->CKTcopyNodesets) {
		  CKTnode *tmpNode;
		  IFuid tmpName;
                  if (CKTinst2Node(ckt,here,5,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }				
            }
            }
	    else 
	        here->BSIM4v5qNode = 0;

/* set Sparse Matrix Pointers 
 * macro to make elements with built-in out-of-memory test */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(BSIM4v5DPbpPtr, BSIM4v5dNodePrime, BSIM4v5bNodePrime);
            TSTALLOC(BSIM4v5GPbpPtr, BSIM4v5gNodePrime, BSIM4v5bNodePrime);
            TSTALLOC(BSIM4v5SPbpPtr, BSIM4v5sNodePrime, BSIM4v5bNodePrime);

            TSTALLOC(BSIM4v5BPdpPtr, BSIM4v5bNodePrime, BSIM4v5dNodePrime);
            TSTALLOC(BSIM4v5BPgpPtr, BSIM4v5bNodePrime, BSIM4v5gNodePrime);
            TSTALLOC(BSIM4v5BPspPtr, BSIM4v5bNodePrime, BSIM4v5sNodePrime);
            TSTALLOC(BSIM4v5BPbpPtr, BSIM4v5bNodePrime, BSIM4v5bNodePrime);

            TSTALLOC(BSIM4v5DdPtr, BSIM4v5dNode, BSIM4v5dNode);
            TSTALLOC(BSIM4v5GPgpPtr, BSIM4v5gNodePrime, BSIM4v5gNodePrime);
            TSTALLOC(BSIM4v5SsPtr, BSIM4v5sNode, BSIM4v5sNode);
            TSTALLOC(BSIM4v5DPdpPtr, BSIM4v5dNodePrime, BSIM4v5dNodePrime);
            TSTALLOC(BSIM4v5SPspPtr, BSIM4v5sNodePrime, BSIM4v5sNodePrime);
            TSTALLOC(BSIM4v5DdpPtr, BSIM4v5dNode, BSIM4v5dNodePrime);
            TSTALLOC(BSIM4v5GPdpPtr, BSIM4v5gNodePrime, BSIM4v5dNodePrime);
            TSTALLOC(BSIM4v5GPspPtr, BSIM4v5gNodePrime, BSIM4v5sNodePrime);
            TSTALLOC(BSIM4v5SspPtr, BSIM4v5sNode, BSIM4v5sNodePrime);
            TSTALLOC(BSIM4v5DPspPtr, BSIM4v5dNodePrime, BSIM4v5sNodePrime);
            TSTALLOC(BSIM4v5DPdPtr, BSIM4v5dNodePrime, BSIM4v5dNode);
            TSTALLOC(BSIM4v5DPgpPtr, BSIM4v5dNodePrime, BSIM4v5gNodePrime);
            TSTALLOC(BSIM4v5SPgpPtr, BSIM4v5sNodePrime, BSIM4v5gNodePrime);
            TSTALLOC(BSIM4v5SPsPtr, BSIM4v5sNodePrime, BSIM4v5sNode);
            TSTALLOC(BSIM4v5SPdpPtr, BSIM4v5sNodePrime, BSIM4v5dNodePrime);

            TSTALLOC(BSIM4v5QqPtr, BSIM4v5qNode, BSIM4v5qNode);
            TSTALLOC(BSIM4v5QbpPtr, BSIM4v5qNode, BSIM4v5bNodePrime) ;
            TSTALLOC(BSIM4v5QdpPtr, BSIM4v5qNode, BSIM4v5dNodePrime);
            TSTALLOC(BSIM4v5QspPtr, BSIM4v5qNode, BSIM4v5sNodePrime);
            TSTALLOC(BSIM4v5QgpPtr, BSIM4v5qNode, BSIM4v5gNodePrime);
            TSTALLOC(BSIM4v5DPqPtr, BSIM4v5dNodePrime, BSIM4v5qNode);
            TSTALLOC(BSIM4v5SPqPtr, BSIM4v5sNodePrime, BSIM4v5qNode);
            TSTALLOC(BSIM4v5GPqPtr, BSIM4v5gNodePrime, BSIM4v5qNode);

            if (here->BSIM4v5rgateMod != 0)
            {   TSTALLOC(BSIM4v5GEgePtr, BSIM4v5gNodeExt, BSIM4v5gNodeExt);
                TSTALLOC(BSIM4v5GEgpPtr, BSIM4v5gNodeExt, BSIM4v5gNodePrime);
                TSTALLOC(BSIM4v5GPgePtr, BSIM4v5gNodePrime, BSIM4v5gNodeExt);
		TSTALLOC(BSIM4v5GEdpPtr, BSIM4v5gNodeExt, BSIM4v5dNodePrime);
		TSTALLOC(BSIM4v5GEspPtr, BSIM4v5gNodeExt, BSIM4v5sNodePrime);
		TSTALLOC(BSIM4v5GEbpPtr, BSIM4v5gNodeExt, BSIM4v5bNodePrime);

                TSTALLOC(BSIM4v5GMdpPtr, BSIM4v5gNodeMid, BSIM4v5dNodePrime);
                TSTALLOC(BSIM4v5GMgpPtr, BSIM4v5gNodeMid, BSIM4v5gNodePrime);
                TSTALLOC(BSIM4v5GMgmPtr, BSIM4v5gNodeMid, BSIM4v5gNodeMid);
                TSTALLOC(BSIM4v5GMgePtr, BSIM4v5gNodeMid, BSIM4v5gNodeExt);
                TSTALLOC(BSIM4v5GMspPtr, BSIM4v5gNodeMid, BSIM4v5sNodePrime);
                TSTALLOC(BSIM4v5GMbpPtr, BSIM4v5gNodeMid, BSIM4v5bNodePrime);
                TSTALLOC(BSIM4v5DPgmPtr, BSIM4v5dNodePrime, BSIM4v5gNodeMid);
                TSTALLOC(BSIM4v5GPgmPtr, BSIM4v5gNodePrime, BSIM4v5gNodeMid);
                TSTALLOC(BSIM4v5GEgmPtr, BSIM4v5gNodeExt, BSIM4v5gNodeMid);
                TSTALLOC(BSIM4v5SPgmPtr, BSIM4v5sNodePrime, BSIM4v5gNodeMid);
                TSTALLOC(BSIM4v5BPgmPtr, BSIM4v5bNodePrime, BSIM4v5gNodeMid);
            }	

            if ((here->BSIM4v5rbodyMod ==1) || (here->BSIM4v5rbodyMod ==2))
            {   TSTALLOC(BSIM4v5DPdbPtr, BSIM4v5dNodePrime, BSIM4v5dbNode);
                TSTALLOC(BSIM4v5SPsbPtr, BSIM4v5sNodePrime, BSIM4v5sbNode);

                TSTALLOC(BSIM4v5DBdpPtr, BSIM4v5dbNode, BSIM4v5dNodePrime);
                TSTALLOC(BSIM4v5DBdbPtr, BSIM4v5dbNode, BSIM4v5dbNode);
                TSTALLOC(BSIM4v5DBbpPtr, BSIM4v5dbNode, BSIM4v5bNodePrime);
                TSTALLOC(BSIM4v5DBbPtr, BSIM4v5dbNode, BSIM4v5bNode);

                TSTALLOC(BSIM4v5BPdbPtr, BSIM4v5bNodePrime, BSIM4v5dbNode);
                TSTALLOC(BSIM4v5BPbPtr, BSIM4v5bNodePrime, BSIM4v5bNode);
                TSTALLOC(BSIM4v5BPsbPtr, BSIM4v5bNodePrime, BSIM4v5sbNode);

                TSTALLOC(BSIM4v5SBspPtr, BSIM4v5sbNode, BSIM4v5sNodePrime);
                TSTALLOC(BSIM4v5SBbpPtr, BSIM4v5sbNode, BSIM4v5bNodePrime);
                TSTALLOC(BSIM4v5SBbPtr, BSIM4v5sbNode, BSIM4v5bNode);
                TSTALLOC(BSIM4v5SBsbPtr, BSIM4v5sbNode, BSIM4v5sbNode);

                TSTALLOC(BSIM4v5BdbPtr, BSIM4v5bNode, BSIM4v5dbNode);
                TSTALLOC(BSIM4v5BbpPtr, BSIM4v5bNode, BSIM4v5bNodePrime);
                TSTALLOC(BSIM4v5BsbPtr, BSIM4v5bNode, BSIM4v5sbNode);
	        TSTALLOC(BSIM4v5BbPtr, BSIM4v5bNode, BSIM4v5bNode);
            }

            if (model->BSIM4v5rdsMod)
            {   TSTALLOC(BSIM4v5DgpPtr, BSIM4v5dNode, BSIM4v5gNodePrime);
		TSTALLOC(BSIM4v5DspPtr, BSIM4v5dNode, BSIM4v5sNodePrime);
                TSTALLOC(BSIM4v5DbpPtr, BSIM4v5dNode, BSIM4v5bNodePrime);
                TSTALLOC(BSIM4v5SdpPtr, BSIM4v5sNode, BSIM4v5dNodePrime);
                TSTALLOC(BSIM4v5SgpPtr, BSIM4v5sNode, BSIM4v5gNodePrime);
                TSTALLOC(BSIM4v5SbpPtr, BSIM4v5sNode, BSIM4v5bNodePrime);
            }
        }
    }

#ifdef USE_OMP
    InstCount = 0;
    model = (BSIM4v5model*)inModel;
    /* loop through all the BSIM4v6 device models
    to count the number of instances */

    for (; model != NULL; model = BSIM4v5nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = BSIM4v5instances(model); here != NULL;
             here = BSIM4v5nextInstance(here))
        {
            InstCount++;
        }
        model->BSIM4v5InstCount = 0;
        model->BSIM4v5InstanceArray = NULL;
    }
    InstArray = TMALLOC(BSIM4v5instance*, InstCount);
    model = (BSIM4v5model*)inModel;
    /* store this in the first model only */
    model->BSIM4v5InstCount = InstCount;
    model->BSIM4v5InstanceArray = InstArray;
    idx = 0;
    for (; model != NULL; model = BSIM4v5nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = BSIM4v5instances(model); here != NULL;
             here = BSIM4v5nextInstance(here))
        {
            InstArray[idx] = here;
            idx++;
        }
    }
#endif

    return(OK);
}  

int
BSIM4v5unsetup(
    GENmodel *inModel,
    CKTcircuit *ckt)
{
#ifndef HAS_BATCHSIM
    BSIM4v5model *model;
    BSIM4v5instance *here;

#ifdef USE_OMP
    model = (BSIM4v5model*)inModel;
    tfree(model->BSIM4v5InstanceArray);
#endif

    for (model = (BSIM4v5model *)inModel; model != NULL;
            model = BSIM4v5nextModel(model))
    {
        for (here = BSIM4v5instances(model); here != NULL;
                here=BSIM4v5nextInstance(here))
        {
            if (here->BSIM4v5qNode > 0)
                CKTdltNNum(ckt, here->BSIM4v5qNode);
            here->BSIM4v5qNode = 0;

            if (here->BSIM4v5sbNode > 0 &&
                here->BSIM4v5sbNode != here->BSIM4v5bNode)
                CKTdltNNum(ckt, here->BSIM4v5sbNode);
            here->BSIM4v5sbNode = 0;

            if (here->BSIM4v5bNodePrime > 0 &&
                here->BSIM4v5bNodePrime != here->BSIM4v5bNode)
                CKTdltNNum(ckt, here->BSIM4v5bNodePrime);
            here->BSIM4v5bNodePrime = 0;

            if (here->BSIM4v5dbNode > 0 &&
                here->BSIM4v5dbNode != here->BSIM4v5bNode)
                CKTdltNNum(ckt, here->BSIM4v5dbNode);
            here->BSIM4v5dbNode = 0;

            if (here->BSIM4v5gNodeMid > 0 &&
                here->BSIM4v5gNodeMid != here->BSIM4v5gNodeExt)
                CKTdltNNum(ckt, here->BSIM4v5gNodeMid);
            here->BSIM4v5gNodeMid = 0;

            if (here->BSIM4v5gNodePrime > 0 &&
                here->BSIM4v5gNodePrime != here->BSIM4v5gNodeExt)
                CKTdltNNum(ckt, here->BSIM4v5gNodePrime);
            here->BSIM4v5gNodePrime = 0;

            if (here->BSIM4v5sNodePrime > 0
                    && here->BSIM4v5sNodePrime != here->BSIM4v5sNode)
                CKTdltNNum(ckt, here->BSIM4v5sNodePrime);
            here->BSIM4v5sNodePrime = 0;

            if (here->BSIM4v5dNodePrime > 0
                    && here->BSIM4v5dNodePrime != here->BSIM4v5dNode)
                CKTdltNNum(ckt, here->BSIM4v5dNodePrime);
            here->BSIM4v5dNodePrime = 0;
        }
    }
#endif
    return OK;
}
