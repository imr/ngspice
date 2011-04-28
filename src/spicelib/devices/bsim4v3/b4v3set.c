/**** BSIM4.3.0 Released by Xuemei (Jane) Xi 05/09/2003 ****/

/**********
 * Copyright 2003 Regents of the University of California. All rights reserved.
 * File: b4v3set.c of BSIM4.3.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 **********/

#include "ngspice.h"
#include "jobdefs.h"
#include "ftedefs.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "bsim4v3def.h"
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
BSIM4v3setup(
SMPmatrix *matrix,
GENmodel *inModel,
CKTcircuit *ckt,
int *states)
{
BSIM4v3model *model = (BSIM4v3model*)inModel;
BSIM4v3instance *here;
int error;
CKTnode *tmp;
int    noiseAnalGiven = 0, createNode;  /* Criteria for new node creation */
double Rtot, DMCGeff, DMCIeff, DMDGeff;
JOB   *job;

    /* Search for a noise analysis request */
    for (job = ft_curckt->ci_curTask->jobs; job; job = job->JOBnextJob) {
        if(strcmp(job->JOBname,"Noise Analysis")==0) {
            noiseAnalGiven = 1;
            break;
        }
    }

    /*  loop through all the BSIM4v3 device models */
    for( ; model != NULL; model = model->BSIM4v3nextModel )
    {   /* process defaults of model parameters */
        if (!model->BSIM4v3typeGiven)
            model->BSIM4v3type = NMOS;     

        if (!model->BSIM4v3mobModGiven) 
            model->BSIM4v3mobMod = 0;
	else if ((model->BSIM4v3mobMod != 0) && (model->BSIM4v3mobMod != 1)
	         && (model->BSIM4v3mobMod != 2))
	{   model->BSIM4v3mobMod = 0;
            printf("Warning: mobMod has been set to its default value: 0.\n");
	}

        if (!model->BSIM4v3binUnitGiven) 
            model->BSIM4v3binUnit = 1;
        if (!model->BSIM4v3paramChkGiven) 
            model->BSIM4v3paramChk = 1;

        if (!model->BSIM4v3dioModGiven)
            model->BSIM4v3dioMod = 1;
        else if ((model->BSIM4v3dioMod != 0) && (model->BSIM4v3dioMod != 1)
            && (model->BSIM4v3dioMod != 2))
        {   model->BSIM4v3dioMod = 1;
            printf("Warning: dioMod has been set to its default value: 1.\n");
        }

        if (!model->BSIM4v3capModGiven) 
            model->BSIM4v3capMod = 2;
        else if ((model->BSIM4v3capMod != 0) && (model->BSIM4v3capMod != 1)
            && (model->BSIM4v3capMod != 2))
        {   model->BSIM4v3capMod = 2;
            printf("Warning: capMod has been set to its default value: 2.\n");
        }

        if (!model->BSIM4v3rdsModGiven)
            model->BSIM4v3rdsMod = 0;
	else if ((model->BSIM4v3rdsMod != 0) && (model->BSIM4v3rdsMod != 1))
        {   model->BSIM4v3rdsMod = 0;
	    printf("Warning: rdsMod has been set to its default value: 0.\n");
	}
        if (!model->BSIM4v3rbodyModGiven)
            model->BSIM4v3rbodyMod = 0;
        else if ((model->BSIM4v3rbodyMod != 0) && (model->BSIM4v3rbodyMod != 1))
        {   model->BSIM4v3rbodyMod = 0;
            printf("Warning: rbodyMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v3rgateModGiven)
            model->BSIM4v3rgateMod = 0;
        else if ((model->BSIM4v3rgateMod != 0) && (model->BSIM4v3rgateMod != 1)
            && (model->BSIM4v3rgateMod != 2) && (model->BSIM4v3rgateMod != 3))
        {   model->BSIM4v3rgateMod = 0;
            printf("Warning: rgateMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v3perModGiven)
            model->BSIM4v3perMod = 1;
        else if ((model->BSIM4v3perMod != 0) && (model->BSIM4v3perMod != 1))
        {   model->BSIM4v3perMod = 1;
            printf("Warning: perMod has been set to its default value: 1.\n");
        }

        if (!model->BSIM4v3geoModGiven)
            model->BSIM4v3geoMod = 0;

        if (!model->BSIM4v3fnoiModGiven) 
            model->BSIM4v3fnoiMod = 1;
        else if ((model->BSIM4v3fnoiMod != 0) && (model->BSIM4v3fnoiMod != 1))
        {   model->BSIM4v3fnoiMod = 1;
            printf("Warning: fnoiMod has been set to its default value: 1.\n");
        }
        if (!model->BSIM4v3tnoiModGiven)
            model->BSIM4v3tnoiMod = 0; /* WDLiu: tnoiMod=1 needs to set internal S/D nodes */
        else if ((model->BSIM4v3tnoiMod != 0) && (model->BSIM4v3tnoiMod != 1))
        {   model->BSIM4v3tnoiMod = 0;
            printf("Warning: tnoiMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v3trnqsModGiven)
            model->BSIM4v3trnqsMod = 0; 
        else if ((model->BSIM4v3trnqsMod != 0) && (model->BSIM4v3trnqsMod != 1))
        {   model->BSIM4v3trnqsMod = 0;
            printf("Warning: trnqsMod has been set to its default value: 0.\n");
        }
        if (!model->BSIM4v3acnqsModGiven)
            model->BSIM4v3acnqsMod = 0;
        else if ((model->BSIM4v3acnqsMod != 0) && (model->BSIM4v3acnqsMod != 1))
        {   model->BSIM4v3acnqsMod = 0;
            printf("Warning: acnqsMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v3igcModGiven)
            model->BSIM4v3igcMod = 0;
        else if ((model->BSIM4v3igcMod != 0) && (model->BSIM4v3igcMod != 1))
        {   model->BSIM4v3igcMod = 0;
            printf("Warning: igcMod has been set to its default value: 0.\n");
        }
        if (!model->BSIM4v3igbModGiven)
            model->BSIM4v3igbMod = 0;
        else if ((model->BSIM4v3igbMod != 0) && (model->BSIM4v3igbMod != 1))
        {   model->BSIM4v3igbMod = 0;
            printf("Warning: igbMod has been set to its default value: 0.\n");
        }
        if (!model->BSIM4v3tempModGiven)
            model->BSIM4v3tempMod = 0;
        else if ((model->BSIM4v3tempMod != 0) && (model->BSIM4v3tempMod != 1))
        {   model->BSIM4v3tempMod = 0;
            printf("Warning: tempMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v3versionGiven) 
            model->BSIM4v3version = "4.3.0";
        if (!model->BSIM4v3toxrefGiven)
            model->BSIM4v3toxref = 30.0e-10;
        if (!model->BSIM4v3toxeGiven)
            model->BSIM4v3toxe = 30.0e-10;
        if (!model->BSIM4v3toxpGiven)
            model->BSIM4v3toxp = model->BSIM4v3toxe;
        if (!model->BSIM4v3toxmGiven)
            model->BSIM4v3toxm = model->BSIM4v3toxe;
        if (!model->BSIM4v3dtoxGiven)
            model->BSIM4v3dtox = 0.0;
        if (!model->BSIM4v3epsroxGiven)
            model->BSIM4v3epsrox = 3.9;

        if (!model->BSIM4v3cdscGiven)
	    model->BSIM4v3cdsc = 2.4e-4;   /* unit Q/V/m^2  */
        if (!model->BSIM4v3cdscbGiven)
	    model->BSIM4v3cdscb = 0.0;   /* unit Q/V/m^2  */    
	    if (!model->BSIM4v3cdscdGiven)
	    model->BSIM4v3cdscd = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM4v3citGiven)
	    model->BSIM4v3cit = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM4v3nfactorGiven)
	    model->BSIM4v3nfactor = 1.0;
        if (!model->BSIM4v3xjGiven)
            model->BSIM4v3xj = .15e-6;
        if (!model->BSIM4v3vsatGiven)
            model->BSIM4v3vsat = 8.0e4;    /* unit m/s */ 
        if (!model->BSIM4v3atGiven)
            model->BSIM4v3at = 3.3e4;    /* unit m/s */ 
        if (!model->BSIM4v3a0Given)
            model->BSIM4v3a0 = 1.0;  
        if (!model->BSIM4v3agsGiven)
            model->BSIM4v3ags = 0.0;
        if (!model->BSIM4v3a1Given)
            model->BSIM4v3a1 = 0.0;
        if (!model->BSIM4v3a2Given)
            model->BSIM4v3a2 = 1.0;
        if (!model->BSIM4v3ketaGiven)
            model->BSIM4v3keta = -0.047;    /* unit  / V */
        if (!model->BSIM4v3nsubGiven)
            model->BSIM4v3nsub = 6.0e16;   /* unit 1/cm3 */
        if (!model->BSIM4v3ndepGiven)
            model->BSIM4v3ndep = 1.7e17;   /* unit 1/cm3 */
        if (!model->BSIM4v3nsdGiven)
            model->BSIM4v3nsd = 1.0e20;   /* unit 1/cm3 */
        if (!model->BSIM4v3phinGiven)
            model->BSIM4v3phin = 0.0; /* unit V */
        if (!model->BSIM4v3ngateGiven)
            model->BSIM4v3ngate = 0;   /* unit 1/cm3 */
        if (!model->BSIM4v3vbmGiven)
	    model->BSIM4v3vbm = -3.0;
        if (!model->BSIM4v3xtGiven)
	    model->BSIM4v3xt = 1.55e-7;
        if (!model->BSIM4v3kt1Given)
            model->BSIM4v3kt1 = -0.11;      /* unit V */
        if (!model->BSIM4v3kt1lGiven)
            model->BSIM4v3kt1l = 0.0;      /* unit V*m */
        if (!model->BSIM4v3kt2Given)
            model->BSIM4v3kt2 = 0.022;      /* No unit */
        if (!model->BSIM4v3k3Given)
            model->BSIM4v3k3 = 80.0;      
        if (!model->BSIM4v3k3bGiven)
            model->BSIM4v3k3b = 0.0;      
        if (!model->BSIM4v3w0Given)
            model->BSIM4v3w0 = 2.5e-6;    
        if (!model->BSIM4v3lpe0Given)
            model->BSIM4v3lpe0 = 1.74e-7;     
        if (!model->BSIM4v3lpebGiven)
            model->BSIM4v3lpeb = 0.0;
        if (!model->BSIM4v3dvtp0Given)
            model->BSIM4v3dvtp0 = 0.0;
        if (!model->BSIM4v3dvtp1Given)
            model->BSIM4v3dvtp1 = 0.0;
        if (!model->BSIM4v3dvt0Given)
            model->BSIM4v3dvt0 = 2.2;    
        if (!model->BSIM4v3dvt1Given)
            model->BSIM4v3dvt1 = 0.53;      
        if (!model->BSIM4v3dvt2Given)
            model->BSIM4v3dvt2 = -0.032;   /* unit 1 / V */     

        if (!model->BSIM4v3dvt0wGiven)
            model->BSIM4v3dvt0w = 0.0;    
        if (!model->BSIM4v3dvt1wGiven)
            model->BSIM4v3dvt1w = 5.3e6;    
        if (!model->BSIM4v3dvt2wGiven)
            model->BSIM4v3dvt2w = -0.032;   

        if (!model->BSIM4v3droutGiven)
            model->BSIM4v3drout = 0.56;     
        if (!model->BSIM4v3dsubGiven)
            model->BSIM4v3dsub = model->BSIM4v3drout;     
        if (!model->BSIM4v3vth0Given)
            model->BSIM4v3vth0 = (model->BSIM4v3type == NMOS) ? 0.7 : -0.7;
        if (!model->BSIM4v3euGiven)
            model->BSIM4v3eu = (model->BSIM4v3type == NMOS) ? 1.67 : 1.0;;
        if (!model->BSIM4v3uaGiven)
            model->BSIM4v3ua = (model->BSIM4v3mobMod == 2) ? 1.0e-15 : 1.0e-9; /* unit m/V */
        if (!model->BSIM4v3ua1Given)
            model->BSIM4v3ua1 = 1.0e-9;      /* unit m/V */
        if (!model->BSIM4v3ubGiven)
            model->BSIM4v3ub = 1.0e-19;     /* unit (m/V)**2 */
        if (!model->BSIM4v3ub1Given)
            model->BSIM4v3ub1 = -1.0e-18;     /* unit (m/V)**2 */
        if (!model->BSIM4v3ucGiven)
            model->BSIM4v3uc = (model->BSIM4v3mobMod == 1) ? -0.0465 : -0.0465e-9;   
        if (!model->BSIM4v3uc1Given)
            model->BSIM4v3uc1 = (model->BSIM4v3mobMod == 1) ? -0.056 : -0.056e-9;   
        if (!model->BSIM4v3u0Given)
            model->BSIM4v3u0 = (model->BSIM4v3type == NMOS) ? 0.067 : 0.025;
        if (!model->BSIM4v3uteGiven)
	    model->BSIM4v3ute = -1.5;    
        if (!model->BSIM4v3voffGiven)
	    model->BSIM4v3voff = -0.08;
        if (!model->BSIM4v3vofflGiven)
            model->BSIM4v3voffl = 0.0;
        if (!model->BSIM4v3minvGiven)
            model->BSIM4v3minv = 0.0;
        if (!model->BSIM4v3fproutGiven)
            model->BSIM4v3fprout = 0.0;
        if (!model->BSIM4v3pditsGiven)
            model->BSIM4v3pdits = 0.0;
        if (!model->BSIM4v3pditsdGiven)
            model->BSIM4v3pditsd = 0.0;
        if (!model->BSIM4v3pditslGiven)
            model->BSIM4v3pditsl = 0.0;
        if (!model->BSIM4v3deltaGiven)  
           model->BSIM4v3delta = 0.01;
        if (!model->BSIM4v3rdswminGiven)
            model->BSIM4v3rdswmin = 0.0;
        if (!model->BSIM4v3rdwminGiven)
            model->BSIM4v3rdwmin = 0.0;
        if (!model->BSIM4v3rswminGiven)
            model->BSIM4v3rswmin = 0.0;
        if (!model->BSIM4v3rdswGiven)
	    model->BSIM4v3rdsw = 200.0; /* in ohm*um */     
        if (!model->BSIM4v3rdwGiven)
            model->BSIM4v3rdw = 100.0;
        if (!model->BSIM4v3rswGiven)
            model->BSIM4v3rsw = 100.0;
        if (!model->BSIM4v3prwgGiven)
            model->BSIM4v3prwg = 1.0; /* in 1/V */
        if (!model->BSIM4v3prwbGiven)
            model->BSIM4v3prwb = 0.0;      
        if (!model->BSIM4v3prtGiven)
        if (!model->BSIM4v3prtGiven)
            model->BSIM4v3prt = 0.0;      
        if (!model->BSIM4v3eta0Given)
            model->BSIM4v3eta0 = 0.08;      /* no unit  */ 
        if (!model->BSIM4v3etabGiven)
            model->BSIM4v3etab = -0.07;      /* unit  1/V */ 
        if (!model->BSIM4v3pclmGiven)
            model->BSIM4v3pclm = 1.3;      /* no unit  */ 
        if (!model->BSIM4v3pdibl1Given)
            model->BSIM4v3pdibl1 = 0.39;    /* no unit  */
        if (!model->BSIM4v3pdibl2Given)
            model->BSIM4v3pdibl2 = 0.0086;    /* no unit  */ 
        if (!model->BSIM4v3pdiblbGiven)
            model->BSIM4v3pdiblb = 0.0;    /* 1/V  */ 
        if (!model->BSIM4v3pscbe1Given)
            model->BSIM4v3pscbe1 = 4.24e8;     
        if (!model->BSIM4v3pscbe2Given)
            model->BSIM4v3pscbe2 = 1.0e-5;    
        if (!model->BSIM4v3pvagGiven)
            model->BSIM4v3pvag = 0.0;     
        if (!model->BSIM4v3wrGiven)  
            model->BSIM4v3wr = 1.0;
        if (!model->BSIM4v3dwgGiven)  
            model->BSIM4v3dwg = 0.0;
        if (!model->BSIM4v3dwbGiven)  
            model->BSIM4v3dwb = 0.0;
        if (!model->BSIM4v3b0Given)
            model->BSIM4v3b0 = 0.0;
        if (!model->BSIM4v3b1Given)  
            model->BSIM4v3b1 = 0.0;
        if (!model->BSIM4v3alpha0Given)  
            model->BSIM4v3alpha0 = 0.0;
        if (!model->BSIM4v3alpha1Given)
            model->BSIM4v3alpha1 = 0.0;
        if (!model->BSIM4v3beta0Given)  
            model->BSIM4v3beta0 = 30.0;
        if (!model->BSIM4v3agidlGiven)
            model->BSIM4v3agidl = 0.0;
        if (!model->BSIM4v3bgidlGiven)
            model->BSIM4v3bgidl = 2.3e9; /* V/m */
        if (!model->BSIM4v3cgidlGiven)
            model->BSIM4v3cgidl = 0.5; /* V^3 */
        if (!model->BSIM4v3egidlGiven)
            model->BSIM4v3egidl = 0.8; /* V */
        if (!model->BSIM4v3aigcGiven)
            model->BSIM4v3aigc = (model->BSIM4v3type == NMOS) ? 0.43 : 0.31;
        if (!model->BSIM4v3bigcGiven)
            model->BSIM4v3bigc = (model->BSIM4v3type == NMOS) ? 0.054 : 0.024;
        if (!model->BSIM4v3cigcGiven)
            model->BSIM4v3cigc = (model->BSIM4v3type == NMOS) ? 0.075 : 0.03;
        if (!model->BSIM4v3aigsdGiven)
            model->BSIM4v3aigsd = (model->BSIM4v3type == NMOS) ? 0.43 : 0.31;
        if (!model->BSIM4v3bigsdGiven)
            model->BSIM4v3bigsd = (model->BSIM4v3type == NMOS) ? 0.054 : 0.024;
        if (!model->BSIM4v3cigsdGiven)
            model->BSIM4v3cigsd = (model->BSIM4v3type == NMOS) ? 0.075 : 0.03;
        if (!model->BSIM4v3aigbaccGiven)
            model->BSIM4v3aigbacc = 0.43;
        if (!model->BSIM4v3bigbaccGiven)
            model->BSIM4v3bigbacc = 0.054;
        if (!model->BSIM4v3cigbaccGiven)
            model->BSIM4v3cigbacc = 0.075;
        if (!model->BSIM4v3aigbinvGiven)
            model->BSIM4v3aigbinv = 0.35;
        if (!model->BSIM4v3bigbinvGiven)
            model->BSIM4v3bigbinv = 0.03;
        if (!model->BSIM4v3cigbinvGiven)
            model->BSIM4v3cigbinv = 0.006;
        if (!model->BSIM4v3nigcGiven)
            model->BSIM4v3nigc = 1.0;
        if (!model->BSIM4v3nigbinvGiven)
            model->BSIM4v3nigbinv = 3.0;
        if (!model->BSIM4v3nigbaccGiven)
            model->BSIM4v3nigbacc = 1.0;
        if (!model->BSIM4v3ntoxGiven)
            model->BSIM4v3ntox = 1.0;
        if (!model->BSIM4v3eigbinvGiven)
            model->BSIM4v3eigbinv = 1.1;
        if (!model->BSIM4v3pigcdGiven)
            model->BSIM4v3pigcd = 1.0;
        if (!model->BSIM4v3poxedgeGiven)
            model->BSIM4v3poxedge = 1.0;
        if (!model->BSIM4v3xrcrg1Given)
            model->BSIM4v3xrcrg1 = 12.0;
        if (!model->BSIM4v3xrcrg2Given)
            model->BSIM4v3xrcrg2 = 1.0;
        if (!model->BSIM4v3ijthsfwdGiven)
            model->BSIM4v3ijthsfwd = 0.1; /* unit A */
        if (!model->BSIM4v3ijthdfwdGiven)
            model->BSIM4v3ijthdfwd = model->BSIM4v3ijthsfwd;
        if (!model->BSIM4v3ijthsrevGiven)
            model->BSIM4v3ijthsrev = 0.1; /* unit A */
        if (!model->BSIM4v3ijthdrevGiven)
            model->BSIM4v3ijthdrev = model->BSIM4v3ijthsrev;
        if (!model->BSIM4v3tnoiaGiven)
            model->BSIM4v3tnoia = 1.5;
        if (!model->BSIM4v3tnoibGiven)
            model->BSIM4v3tnoib = 3.5;
        if (!model->BSIM4v3rnoiaGiven)
            model->BSIM4v3rnoia = 0.577;
        if (!model->BSIM4v3rnoibGiven)
            model->BSIM4v3rnoib = 0.37;
        if (!model->BSIM4v3ntnoiGiven)
            model->BSIM4v3ntnoi = 1.0;
        if (!model->BSIM4v3lambdaGiven)
            model->BSIM4v3lambda = 0.0;
        if (!model->BSIM4v3vtlGiven)
            model->BSIM4v3vtl = 2.0e5;    /* unit m/s */ 
        if (!model->BSIM4v3xnGiven)
            model->BSIM4v3xn = 3.0;   
        if (!model->BSIM4v3lcGiven)
            model->BSIM4v3lc = 5.0e-9;   

        if (!model->BSIM4v3xjbvsGiven)
            model->BSIM4v3xjbvs = 1.0; /* no unit */
        if (!model->BSIM4v3xjbvdGiven)
            model->BSIM4v3xjbvd = model->BSIM4v3xjbvs;
        if (!model->BSIM4v3bvsGiven)
            model->BSIM4v3bvs = 10.0; /* V */
        if (!model->BSIM4v3bvdGiven)
            model->BSIM4v3bvd = model->BSIM4v3bvs;
        if (!model->BSIM4v3gbminGiven)
            model->BSIM4v3gbmin = 1.0e-12; /* in mho */
        if (!model->BSIM4v3rbdbGiven)
            model->BSIM4v3rbdb = 50.0; /* in ohm */
        if (!model->BSIM4v3rbpbGiven)
            model->BSIM4v3rbpb = 50.0;
        if (!model->BSIM4v3rbsbGiven)
            model->BSIM4v3rbsb = 50.0;
        if (!model->BSIM4v3rbpsGiven)
            model->BSIM4v3rbps = 50.0;
        if (!model->BSIM4v3rbpdGiven)
            model->BSIM4v3rbpd = 50.0;

        if (!model->BSIM4v3cgslGiven)  
            model->BSIM4v3cgsl = 0.0;
        if (!model->BSIM4v3cgdlGiven)  
            model->BSIM4v3cgdl = 0.0;
        if (!model->BSIM4v3ckappasGiven)  
            model->BSIM4v3ckappas = 0.6;
        if (!model->BSIM4v3ckappadGiven)
            model->BSIM4v3ckappad = model->BSIM4v3ckappas;
        if (!model->BSIM4v3clcGiven)  
            model->BSIM4v3clc = 0.1e-6;
        if (!model->BSIM4v3cleGiven)  
            model->BSIM4v3cle = 0.6;
        if (!model->BSIM4v3vfbcvGiven)  
            model->BSIM4v3vfbcv = -1.0;
        if (!model->BSIM4v3acdeGiven)
            model->BSIM4v3acde = 1.0;
        if (!model->BSIM4v3moinGiven)
            model->BSIM4v3moin = 15.0;
        if (!model->BSIM4v3noffGiven)
            model->BSIM4v3noff = 1.0;
        if (!model->BSIM4v3voffcvGiven)
            model->BSIM4v3voffcv = 0.0;
        if (!model->BSIM4v3dmcgGiven)
            model->BSIM4v3dmcg = 0.0;
        if (!model->BSIM4v3dmciGiven)
            model->BSIM4v3dmci = model->BSIM4v3dmcg;
        if (!model->BSIM4v3dmdgGiven)
            model->BSIM4v3dmdg = 0.0;
        if (!model->BSIM4v3dmcgtGiven)
            model->BSIM4v3dmcgt = 0.0;
        if (!model->BSIM4v3xgwGiven)
            model->BSIM4v3xgw = 0.0;
        if (!model->BSIM4v3xglGiven)
            model->BSIM4v3xgl = 0.0;
        if (!model->BSIM4v3rshgGiven)
            model->BSIM4v3rshg = 0.1;
        if (!model->BSIM4v3ngconGiven)
            model->BSIM4v3ngcon = 1.0;
        if (!model->BSIM4v3tcjGiven)
            model->BSIM4v3tcj = 0.0;
        if (!model->BSIM4v3tpbGiven)
            model->BSIM4v3tpb = 0.0;
        if (!model->BSIM4v3tcjswGiven)
            model->BSIM4v3tcjsw = 0.0;
        if (!model->BSIM4v3tpbswGiven)
            model->BSIM4v3tpbsw = 0.0;
        if (!model->BSIM4v3tcjswgGiven)
            model->BSIM4v3tcjswg = 0.0;
        if (!model->BSIM4v3tpbswgGiven)
            model->BSIM4v3tpbswg = 0.0;

	/* Length dependence */
        if (!model->BSIM4v3lcdscGiven)
	    model->BSIM4v3lcdsc = 0.0;
        if (!model->BSIM4v3lcdscbGiven)
	    model->BSIM4v3lcdscb = 0.0;
	    if (!model->BSIM4v3lcdscdGiven) 
	    model->BSIM4v3lcdscd = 0.0;
        if (!model->BSIM4v3lcitGiven)
	    model->BSIM4v3lcit = 0.0;
        if (!model->BSIM4v3lnfactorGiven)
	    model->BSIM4v3lnfactor = 0.0;
        if (!model->BSIM4v3lxjGiven)
            model->BSIM4v3lxj = 0.0;
        if (!model->BSIM4v3lvsatGiven)
            model->BSIM4v3lvsat = 0.0;
        if (!model->BSIM4v3latGiven)
            model->BSIM4v3lat = 0.0;
        if (!model->BSIM4v3la0Given)
            model->BSIM4v3la0 = 0.0; 
        if (!model->BSIM4v3lagsGiven)
            model->BSIM4v3lags = 0.0;
        if (!model->BSIM4v3la1Given)
            model->BSIM4v3la1 = 0.0;
        if (!model->BSIM4v3la2Given)
            model->BSIM4v3la2 = 0.0;
        if (!model->BSIM4v3lketaGiven)
            model->BSIM4v3lketa = 0.0;
        if (!model->BSIM4v3lnsubGiven)
            model->BSIM4v3lnsub = 0.0;
        if (!model->BSIM4v3lndepGiven)
            model->BSIM4v3lndep = 0.0;
        if (!model->BSIM4v3lnsdGiven)
            model->BSIM4v3lnsd = 0.0;
        if (!model->BSIM4v3lphinGiven)
            model->BSIM4v3lphin = 0.0;
        if (!model->BSIM4v3lngateGiven)
            model->BSIM4v3lngate = 0.0;
        if (!model->BSIM4v3lvbmGiven)
	    model->BSIM4v3lvbm = 0.0;
        if (!model->BSIM4v3lxtGiven)
	    model->BSIM4v3lxt = 0.0;
        if (!model->BSIM4v3lkt1Given)
            model->BSIM4v3lkt1 = 0.0; 
        if (!model->BSIM4v3lkt1lGiven)
            model->BSIM4v3lkt1l = 0.0;
        if (!model->BSIM4v3lkt2Given)
            model->BSIM4v3lkt2 = 0.0;
        if (!model->BSIM4v3lk3Given)
            model->BSIM4v3lk3 = 0.0;      
        if (!model->BSIM4v3lk3bGiven)
            model->BSIM4v3lk3b = 0.0;      
        if (!model->BSIM4v3lw0Given)
            model->BSIM4v3lw0 = 0.0;    
        if (!model->BSIM4v3llpe0Given)
            model->BSIM4v3llpe0 = 0.0;
        if (!model->BSIM4v3llpebGiven)
            model->BSIM4v3llpeb = model->BSIM4v3llpe0;
        if (!model->BSIM4v3ldvtp0Given)
            model->BSIM4v3ldvtp0 = 0.0;
        if (!model->BSIM4v3ldvtp1Given)
            model->BSIM4v3ldvtp1 = 0.0;
        if (!model->BSIM4v3ldvt0Given)
            model->BSIM4v3ldvt0 = 0.0;    
        if (!model->BSIM4v3ldvt1Given)
            model->BSIM4v3ldvt1 = 0.0;      
        if (!model->BSIM4v3ldvt2Given)
            model->BSIM4v3ldvt2 = 0.0;
        if (!model->BSIM4v3ldvt0wGiven)
            model->BSIM4v3ldvt0w = 0.0;    
        if (!model->BSIM4v3ldvt1wGiven)
            model->BSIM4v3ldvt1w = 0.0;      
        if (!model->BSIM4v3ldvt2wGiven)
            model->BSIM4v3ldvt2w = 0.0;
        if (!model->BSIM4v3ldroutGiven)
            model->BSIM4v3ldrout = 0.0;     
        if (!model->BSIM4v3ldsubGiven)
            model->BSIM4v3ldsub = 0.0;
        if (!model->BSIM4v3lvth0Given)
           model->BSIM4v3lvth0 = 0.0;
        if (!model->BSIM4v3luaGiven)
            model->BSIM4v3lua = 0.0;
        if (!model->BSIM4v3lua1Given)
            model->BSIM4v3lua1 = 0.0;
        if (!model->BSIM4v3lubGiven)
            model->BSIM4v3lub = 0.0;
        if (!model->BSIM4v3lub1Given)
            model->BSIM4v3lub1 = 0.0;
        if (!model->BSIM4v3lucGiven)
            model->BSIM4v3luc = 0.0;
        if (!model->BSIM4v3luc1Given)
            model->BSIM4v3luc1 = 0.0;
        if (!model->BSIM4v3lu0Given)
            model->BSIM4v3lu0 = 0.0;
        if (!model->BSIM4v3luteGiven)
	    model->BSIM4v3lute = 0.0;    
        if (!model->BSIM4v3lvoffGiven)
	    model->BSIM4v3lvoff = 0.0;
        if (!model->BSIM4v3lminvGiven)
            model->BSIM4v3lminv = 0.0;
        if (!model->BSIM4v3lfproutGiven)
            model->BSIM4v3lfprout = 0.0;
        if (!model->BSIM4v3lpditsGiven)
            model->BSIM4v3lpdits = 0.0;
        if (!model->BSIM4v3lpditsdGiven)
            model->BSIM4v3lpditsd = 0.0;
        if (!model->BSIM4v3ldeltaGiven)  
            model->BSIM4v3ldelta = 0.0;
        if (!model->BSIM4v3lrdswGiven)
            model->BSIM4v3lrdsw = 0.0;
        if (!model->BSIM4v3lrdwGiven)
            model->BSIM4v3lrdw = 0.0;
        if (!model->BSIM4v3lrswGiven)
            model->BSIM4v3lrsw = 0.0;
        if (!model->BSIM4v3lprwbGiven)
            model->BSIM4v3lprwb = 0.0;
        if (!model->BSIM4v3lprwgGiven)
            model->BSIM4v3lprwg = 0.0;
        if (!model->BSIM4v3lprtGiven)
            model->BSIM4v3lprt = 0.0;
        if (!model->BSIM4v3leta0Given)
            model->BSIM4v3leta0 = 0.0;
        if (!model->BSIM4v3letabGiven)
            model->BSIM4v3letab = -0.0;
        if (!model->BSIM4v3lpclmGiven)
            model->BSIM4v3lpclm = 0.0; 
        if (!model->BSIM4v3lpdibl1Given)
            model->BSIM4v3lpdibl1 = 0.0;
        if (!model->BSIM4v3lpdibl2Given)
            model->BSIM4v3lpdibl2 = 0.0;
        if (!model->BSIM4v3lpdiblbGiven)
            model->BSIM4v3lpdiblb = 0.0;
        if (!model->BSIM4v3lpscbe1Given)
            model->BSIM4v3lpscbe1 = 0.0;
        if (!model->BSIM4v3lpscbe2Given)
            model->BSIM4v3lpscbe2 = 0.0;
        if (!model->BSIM4v3lpvagGiven)
            model->BSIM4v3lpvag = 0.0;     
        if (!model->BSIM4v3lwrGiven)  
            model->BSIM4v3lwr = 0.0;
        if (!model->BSIM4v3ldwgGiven)  
            model->BSIM4v3ldwg = 0.0;
        if (!model->BSIM4v3ldwbGiven)  
            model->BSIM4v3ldwb = 0.0;
        if (!model->BSIM4v3lb0Given)
            model->BSIM4v3lb0 = 0.0;
        if (!model->BSIM4v3lb1Given)  
            model->BSIM4v3lb1 = 0.0;
        if (!model->BSIM4v3lalpha0Given)  
            model->BSIM4v3lalpha0 = 0.0;
        if (!model->BSIM4v3lalpha1Given)
            model->BSIM4v3lalpha1 = 0.0;
        if (!model->BSIM4v3lbeta0Given)  
            model->BSIM4v3lbeta0 = 0.0;
        if (!model->BSIM4v3lagidlGiven)
            model->BSIM4v3lagidl = 0.0;
        if (!model->BSIM4v3lbgidlGiven)
            model->BSIM4v3lbgidl = 0.0;
        if (!model->BSIM4v3lcgidlGiven)
            model->BSIM4v3lcgidl = 0.0;
        if (!model->BSIM4v3legidlGiven)
            model->BSIM4v3legidl = 0.0;
        if (!model->BSIM4v3laigcGiven)
            model->BSIM4v3laigc = 0.0;
        if (!model->BSIM4v3lbigcGiven)
            model->BSIM4v3lbigc = 0.0;
        if (!model->BSIM4v3lcigcGiven)
            model->BSIM4v3lcigc = 0.0;
        if (!model->BSIM4v3laigsdGiven)
            model->BSIM4v3laigsd = 0.0;
        if (!model->BSIM4v3lbigsdGiven)
            model->BSIM4v3lbigsd = 0.0;
        if (!model->BSIM4v3lcigsdGiven)
            model->BSIM4v3lcigsd = 0.0;
        if (!model->BSIM4v3laigbaccGiven)
            model->BSIM4v3laigbacc = 0.0;
        if (!model->BSIM4v3lbigbaccGiven)
            model->BSIM4v3lbigbacc = 0.0;
        if (!model->BSIM4v3lcigbaccGiven)
            model->BSIM4v3lcigbacc = 0.0;
        if (!model->BSIM4v3laigbinvGiven)
            model->BSIM4v3laigbinv = 0.0;
        if (!model->BSIM4v3lbigbinvGiven)
            model->BSIM4v3lbigbinv = 0.0;
        if (!model->BSIM4v3lcigbinvGiven)
            model->BSIM4v3lcigbinv = 0.0;
        if (!model->BSIM4v3lnigcGiven)
            model->BSIM4v3lnigc = 0.0;
        if (!model->BSIM4v3lnigbinvGiven)
            model->BSIM4v3lnigbinv = 0.0;
        if (!model->BSIM4v3lnigbaccGiven)
            model->BSIM4v3lnigbacc = 0.0;
        if (!model->BSIM4v3lntoxGiven)
            model->BSIM4v3lntox = 0.0;
        if (!model->BSIM4v3leigbinvGiven)
            model->BSIM4v3leigbinv = 0.0;
        if (!model->BSIM4v3lpigcdGiven)
            model->BSIM4v3lpigcd = 0.0;
        if (!model->BSIM4v3lpoxedgeGiven)
            model->BSIM4v3lpoxedge = 0.0;
        if (!model->BSIM4v3lxrcrg1Given)
            model->BSIM4v3lxrcrg1 = 0.0;
        if (!model->BSIM4v3lxrcrg2Given)
            model->BSIM4v3lxrcrg2 = 0.0;
        if (!model->BSIM4v3leuGiven)
            model->BSIM4v3leu = 0.0;
        if (!model->BSIM4v3lvfbGiven)
            model->BSIM4v3lvfb = 0.0;
        if (!model->BSIM4v3llambdaGiven)
            model->BSIM4v3llambda = 0.0;
        if (!model->BSIM4v3lvtlGiven)
            model->BSIM4v3lvtl = 0.0;  
        if (!model->BSIM4v3lxnGiven)
            model->BSIM4v3lxn = 0.0;  

        if (!model->BSIM4v3lcgslGiven)  
            model->BSIM4v3lcgsl = 0.0;
        if (!model->BSIM4v3lcgdlGiven)  
            model->BSIM4v3lcgdl = 0.0;
        if (!model->BSIM4v3lckappasGiven)  
            model->BSIM4v3lckappas = 0.0;
        if (!model->BSIM4v3lckappadGiven)
            model->BSIM4v3lckappad = 0.0;
        if (!model->BSIM4v3lclcGiven)  
            model->BSIM4v3lclc = 0.0;
        if (!model->BSIM4v3lcleGiven)  
            model->BSIM4v3lcle = 0.0;
        if (!model->BSIM4v3lcfGiven)  
            model->BSIM4v3lcf = 0.0;
        if (!model->BSIM4v3lvfbcvGiven)  
            model->BSIM4v3lvfbcv = 0.0;
        if (!model->BSIM4v3lacdeGiven)
            model->BSIM4v3lacde = 0.0;
        if (!model->BSIM4v3lmoinGiven)
            model->BSIM4v3lmoin = 0.0;
        if (!model->BSIM4v3lnoffGiven)
            model->BSIM4v3lnoff = 0.0;
        if (!model->BSIM4v3lvoffcvGiven)
            model->BSIM4v3lvoffcv = 0.0;

	/* Width dependence */
        if (!model->BSIM4v3wcdscGiven)
	    model->BSIM4v3wcdsc = 0.0;
        if (!model->BSIM4v3wcdscbGiven)
	    model->BSIM4v3wcdscb = 0.0;  
	    if (!model->BSIM4v3wcdscdGiven)
	    model->BSIM4v3wcdscd = 0.0;
        if (!model->BSIM4v3wcitGiven)
	    model->BSIM4v3wcit = 0.0;
        if (!model->BSIM4v3wnfactorGiven)
	    model->BSIM4v3wnfactor = 0.0;
        if (!model->BSIM4v3wxjGiven)
            model->BSIM4v3wxj = 0.0;
        if (!model->BSIM4v3wvsatGiven)
            model->BSIM4v3wvsat = 0.0;
        if (!model->BSIM4v3watGiven)
            model->BSIM4v3wat = 0.0;
        if (!model->BSIM4v3wa0Given)
            model->BSIM4v3wa0 = 0.0; 
        if (!model->BSIM4v3wagsGiven)
            model->BSIM4v3wags = 0.0;
        if (!model->BSIM4v3wa1Given)
            model->BSIM4v3wa1 = 0.0;
        if (!model->BSIM4v3wa2Given)
            model->BSIM4v3wa2 = 0.0;
        if (!model->BSIM4v3wketaGiven)
            model->BSIM4v3wketa = 0.0;
        if (!model->BSIM4v3wnsubGiven)
            model->BSIM4v3wnsub = 0.0;
        if (!model->BSIM4v3wndepGiven)
            model->BSIM4v3wndep = 0.0;
        if (!model->BSIM4v3wnsdGiven)
            model->BSIM4v3wnsd = 0.0;
        if (!model->BSIM4v3wphinGiven)
            model->BSIM4v3wphin = 0.0;
        if (!model->BSIM4v3wngateGiven)
            model->BSIM4v3wngate = 0.0;
        if (!model->BSIM4v3wvbmGiven)
	    model->BSIM4v3wvbm = 0.0;
        if (!model->BSIM4v3wxtGiven)
	    model->BSIM4v3wxt = 0.0;
        if (!model->BSIM4v3wkt1Given)
            model->BSIM4v3wkt1 = 0.0; 
        if (!model->BSIM4v3wkt1lGiven)
            model->BSIM4v3wkt1l = 0.0;
        if (!model->BSIM4v3wkt2Given)
            model->BSIM4v3wkt2 = 0.0;
        if (!model->BSIM4v3wk3Given)
            model->BSIM4v3wk3 = 0.0;      
        if (!model->BSIM4v3wk3bGiven)
            model->BSIM4v3wk3b = 0.0;      
        if (!model->BSIM4v3ww0Given)
            model->BSIM4v3ww0 = 0.0;    
        if (!model->BSIM4v3wlpe0Given)
            model->BSIM4v3wlpe0 = 0.0;
        if (!model->BSIM4v3wlpebGiven)
            model->BSIM4v3wlpeb = model->BSIM4v3wlpe0;
        if (!model->BSIM4v3wdvtp0Given)
            model->BSIM4v3wdvtp0 = 0.0;
        if (!model->BSIM4v3wdvtp1Given)
            model->BSIM4v3wdvtp1 = 0.0;
        if (!model->BSIM4v3wdvt0Given)
            model->BSIM4v3wdvt0 = 0.0;    
        if (!model->BSIM4v3wdvt1Given)
            model->BSIM4v3wdvt1 = 0.0;      
        if (!model->BSIM4v3wdvt2Given)
            model->BSIM4v3wdvt2 = 0.0;
        if (!model->BSIM4v3wdvt0wGiven)
            model->BSIM4v3wdvt0w = 0.0;    
        if (!model->BSIM4v3wdvt1wGiven)
            model->BSIM4v3wdvt1w = 0.0;      
        if (!model->BSIM4v3wdvt2wGiven)
            model->BSIM4v3wdvt2w = 0.0;
        if (!model->BSIM4v3wdroutGiven)
            model->BSIM4v3wdrout = 0.0;     
        if (!model->BSIM4v3wdsubGiven)
            model->BSIM4v3wdsub = 0.0;
        if (!model->BSIM4v3wvth0Given)
           model->BSIM4v3wvth0 = 0.0;
        if (!model->BSIM4v3wuaGiven)
            model->BSIM4v3wua = 0.0;
        if (!model->BSIM4v3wua1Given)
            model->BSIM4v3wua1 = 0.0;
        if (!model->BSIM4v3wubGiven)
            model->BSIM4v3wub = 0.0;
        if (!model->BSIM4v3wub1Given)
            model->BSIM4v3wub1 = 0.0;
        if (!model->BSIM4v3wucGiven)
            model->BSIM4v3wuc = 0.0;
        if (!model->BSIM4v3wuc1Given)
            model->BSIM4v3wuc1 = 0.0;
        if (!model->BSIM4v3wu0Given)
            model->BSIM4v3wu0 = 0.0;
        if (!model->BSIM4v3wuteGiven)
	    model->BSIM4v3wute = 0.0;    
        if (!model->BSIM4v3wvoffGiven)
	    model->BSIM4v3wvoff = 0.0;
        if (!model->BSIM4v3wminvGiven)
            model->BSIM4v3wminv = 0.0;
        if (!model->BSIM4v3wfproutGiven)
            model->BSIM4v3wfprout = 0.0;
        if (!model->BSIM4v3wpditsGiven)
            model->BSIM4v3wpdits = 0.0;
        if (!model->BSIM4v3wpditsdGiven)
            model->BSIM4v3wpditsd = 0.0;
        if (!model->BSIM4v3wdeltaGiven)  
            model->BSIM4v3wdelta = 0.0;
        if (!model->BSIM4v3wrdswGiven)
            model->BSIM4v3wrdsw = 0.0;
        if (!model->BSIM4v3wrdwGiven)
            model->BSIM4v3wrdw = 0.0;
        if (!model->BSIM4v3wrswGiven)
            model->BSIM4v3wrsw = 0.0;
        if (!model->BSIM4v3wprwbGiven)
            model->BSIM4v3wprwb = 0.0;
        if (!model->BSIM4v3wprwgGiven)
            model->BSIM4v3wprwg = 0.0;
        if (!model->BSIM4v3wprtGiven)
            model->BSIM4v3wprt = 0.0;
        if (!model->BSIM4v3weta0Given)
            model->BSIM4v3weta0 = 0.0;
        if (!model->BSIM4v3wetabGiven)
            model->BSIM4v3wetab = 0.0;
        if (!model->BSIM4v3wpclmGiven)
            model->BSIM4v3wpclm = 0.0; 
        if (!model->BSIM4v3wpdibl1Given)
            model->BSIM4v3wpdibl1 = 0.0;
        if (!model->BSIM4v3wpdibl2Given)
            model->BSIM4v3wpdibl2 = 0.0;
        if (!model->BSIM4v3wpdiblbGiven)
            model->BSIM4v3wpdiblb = 0.0;
        if (!model->BSIM4v3wpscbe1Given)
            model->BSIM4v3wpscbe1 = 0.0;
        if (!model->BSIM4v3wpscbe2Given)
            model->BSIM4v3wpscbe2 = 0.0;
        if (!model->BSIM4v3wpvagGiven)
            model->BSIM4v3wpvag = 0.0;     
        if (!model->BSIM4v3wwrGiven)  
            model->BSIM4v3wwr = 0.0;
        if (!model->BSIM4v3wdwgGiven)  
            model->BSIM4v3wdwg = 0.0;
        if (!model->BSIM4v3wdwbGiven)  
            model->BSIM4v3wdwb = 0.0;
        if (!model->BSIM4v3wb0Given)
            model->BSIM4v3wb0 = 0.0;
        if (!model->BSIM4v3wb1Given)  
            model->BSIM4v3wb1 = 0.0;
        if (!model->BSIM4v3walpha0Given)  
            model->BSIM4v3walpha0 = 0.0;
        if (!model->BSIM4v3walpha1Given)
            model->BSIM4v3walpha1 = 0.0;
        if (!model->BSIM4v3wbeta0Given)  
            model->BSIM4v3wbeta0 = 0.0;
        if (!model->BSIM4v3wagidlGiven)
            model->BSIM4v3wagidl = 0.0;
        if (!model->BSIM4v3wbgidlGiven)
            model->BSIM4v3wbgidl = 0.0;
        if (!model->BSIM4v3wcgidlGiven)
            model->BSIM4v3wcgidl = 0.0;
        if (!model->BSIM4v3wegidlGiven)
            model->BSIM4v3wegidl = 0.0;
        if (!model->BSIM4v3waigcGiven)
            model->BSIM4v3waigc = 0.0;
        if (!model->BSIM4v3wbigcGiven)
            model->BSIM4v3wbigc = 0.0;
        if (!model->BSIM4v3wcigcGiven)
            model->BSIM4v3wcigc = 0.0;
        if (!model->BSIM4v3waigsdGiven)
            model->BSIM4v3waigsd = 0.0;
        if (!model->BSIM4v3wbigsdGiven)
            model->BSIM4v3wbigsd = 0.0;
        if (!model->BSIM4v3wcigsdGiven)
            model->BSIM4v3wcigsd = 0.0;
        if (!model->BSIM4v3waigbaccGiven)
            model->BSIM4v3waigbacc = 0.0;
        if (!model->BSIM4v3wbigbaccGiven)
            model->BSIM4v3wbigbacc = 0.0;
        if (!model->BSIM4v3wcigbaccGiven)
            model->BSIM4v3wcigbacc = 0.0;
        if (!model->BSIM4v3waigbinvGiven)
            model->BSIM4v3waigbinv = 0.0;
        if (!model->BSIM4v3wbigbinvGiven)
            model->BSIM4v3wbigbinv = 0.0;
        if (!model->BSIM4v3wcigbinvGiven)
            model->BSIM4v3wcigbinv = 0.0;
        if (!model->BSIM4v3wnigcGiven)
            model->BSIM4v3wnigc = 0.0;
        if (!model->BSIM4v3wnigbinvGiven)
            model->BSIM4v3wnigbinv = 0.0;
        if (!model->BSIM4v3wnigbaccGiven)
            model->BSIM4v3wnigbacc = 0.0;
        if (!model->BSIM4v3wntoxGiven)
            model->BSIM4v3wntox = 0.0;
        if (!model->BSIM4v3weigbinvGiven)
            model->BSIM4v3weigbinv = 0.0;
        if (!model->BSIM4v3wpigcdGiven)
            model->BSIM4v3wpigcd = 0.0;
        if (!model->BSIM4v3wpoxedgeGiven)
            model->BSIM4v3wpoxedge = 0.0;
        if (!model->BSIM4v3wxrcrg1Given)
            model->BSIM4v3wxrcrg1 = 0.0;
        if (!model->BSIM4v3wxrcrg2Given)
            model->BSIM4v3wxrcrg2 = 0.0;
        if (!model->BSIM4v3weuGiven)
            model->BSIM4v3weu = 0.0;
        if (!model->BSIM4v3wvfbGiven)
            model->BSIM4v3wvfb = 0.0;
        if (!model->BSIM4v3wlambdaGiven)
            model->BSIM4v3wlambda = 0.0;
        if (!model->BSIM4v3wvtlGiven)
            model->BSIM4v3wvtl = 0.0;  
        if (!model->BSIM4v3wxnGiven)
            model->BSIM4v3wxn = 0.0;  

        if (!model->BSIM4v3wcgslGiven)  
            model->BSIM4v3wcgsl = 0.0;
        if (!model->BSIM4v3wcgdlGiven)  
            model->BSIM4v3wcgdl = 0.0;
        if (!model->BSIM4v3wckappasGiven)  
            model->BSIM4v3wckappas = 0.0;
        if (!model->BSIM4v3wckappadGiven)
            model->BSIM4v3wckappad = 0.0;
        if (!model->BSIM4v3wcfGiven)  
            model->BSIM4v3wcf = 0.0;
        if (!model->BSIM4v3wclcGiven)  
            model->BSIM4v3wclc = 0.0;
        if (!model->BSIM4v3wcleGiven)  
            model->BSIM4v3wcle = 0.0;
        if (!model->BSIM4v3wvfbcvGiven)  
            model->BSIM4v3wvfbcv = 0.0;
        if (!model->BSIM4v3wacdeGiven)
            model->BSIM4v3wacde = 0.0;
        if (!model->BSIM4v3wmoinGiven)
            model->BSIM4v3wmoin = 0.0;
        if (!model->BSIM4v3wnoffGiven)
            model->BSIM4v3wnoff = 0.0;
        if (!model->BSIM4v3wvoffcvGiven)
            model->BSIM4v3wvoffcv = 0.0;

	/* Cross-term dependence */
        if (!model->BSIM4v3pcdscGiven)
	    model->BSIM4v3pcdsc = 0.0;
        if (!model->BSIM4v3pcdscbGiven)
	    model->BSIM4v3pcdscb = 0.0;   
	    if (!model->BSIM4v3pcdscdGiven)
	    model->BSIM4v3pcdscd = 0.0;
        if (!model->BSIM4v3pcitGiven)
	    model->BSIM4v3pcit = 0.0;
        if (!model->BSIM4v3pnfactorGiven)
	    model->BSIM4v3pnfactor = 0.0;
        if (!model->BSIM4v3pxjGiven)
            model->BSIM4v3pxj = 0.0;
        if (!model->BSIM4v3pvsatGiven)
            model->BSIM4v3pvsat = 0.0;
        if (!model->BSIM4v3patGiven)
            model->BSIM4v3pat = 0.0;
        if (!model->BSIM4v3pa0Given)
            model->BSIM4v3pa0 = 0.0; 
            
        if (!model->BSIM4v3pagsGiven)
            model->BSIM4v3pags = 0.0;
        if (!model->BSIM4v3pa1Given)
            model->BSIM4v3pa1 = 0.0;
        if (!model->BSIM4v3pa2Given)
            model->BSIM4v3pa2 = 0.0;
        if (!model->BSIM4v3pketaGiven)
            model->BSIM4v3pketa = 0.0;
        if (!model->BSIM4v3pnsubGiven)
            model->BSIM4v3pnsub = 0.0;
        if (!model->BSIM4v3pndepGiven)
            model->BSIM4v3pndep = 0.0;
        if (!model->BSIM4v3pnsdGiven)
            model->BSIM4v3pnsd = 0.0;
        if (!model->BSIM4v3pphinGiven)
            model->BSIM4v3pphin = 0.0;
        if (!model->BSIM4v3pngateGiven)
            model->BSIM4v3pngate = 0.0;
        if (!model->BSIM4v3pvbmGiven)
	    model->BSIM4v3pvbm = 0.0;
        if (!model->BSIM4v3pxtGiven)
	    model->BSIM4v3pxt = 0.0;
        if (!model->BSIM4v3pkt1Given)
            model->BSIM4v3pkt1 = 0.0; 
        if (!model->BSIM4v3pkt1lGiven)
            model->BSIM4v3pkt1l = 0.0;
        if (!model->BSIM4v3pkt2Given)
            model->BSIM4v3pkt2 = 0.0;
        if (!model->BSIM4v3pk3Given)
            model->BSIM4v3pk3 = 0.0;      
        if (!model->BSIM4v3pk3bGiven)
            model->BSIM4v3pk3b = 0.0;      
        if (!model->BSIM4v3pw0Given)
            model->BSIM4v3pw0 = 0.0;    
        if (!model->BSIM4v3plpe0Given)
            model->BSIM4v3plpe0 = 0.0;
        if (!model->BSIM4v3plpebGiven)
            model->BSIM4v3plpeb = model->BSIM4v3plpe0;
        if (!model->BSIM4v3pdvtp0Given)
            model->BSIM4v3pdvtp0 = 0.0;
        if (!model->BSIM4v3pdvtp1Given)
            model->BSIM4v3pdvtp1 = 0.0;
        if (!model->BSIM4v3pdvt0Given)
            model->BSIM4v3pdvt0 = 0.0;    
        if (!model->BSIM4v3pdvt1Given)
            model->BSIM4v3pdvt1 = 0.0;      
        if (!model->BSIM4v3pdvt2Given)
            model->BSIM4v3pdvt2 = 0.0;
        if (!model->BSIM4v3pdvt0wGiven)
            model->BSIM4v3pdvt0w = 0.0;    
        if (!model->BSIM4v3pdvt1wGiven)
            model->BSIM4v3pdvt1w = 0.0;      
        if (!model->BSIM4v3pdvt2wGiven)
            model->BSIM4v3pdvt2w = 0.0;
        if (!model->BSIM4v3pdroutGiven)
            model->BSIM4v3pdrout = 0.0;     
        if (!model->BSIM4v3pdsubGiven)
            model->BSIM4v3pdsub = 0.0;
        if (!model->BSIM4v3pvth0Given)
           model->BSIM4v3pvth0 = 0.0;
        if (!model->BSIM4v3puaGiven)
            model->BSIM4v3pua = 0.0;
        if (!model->BSIM4v3pua1Given)
            model->BSIM4v3pua1 = 0.0;
        if (!model->BSIM4v3pubGiven)
            model->BSIM4v3pub = 0.0;
        if (!model->BSIM4v3pub1Given)
            model->BSIM4v3pub1 = 0.0;
        if (!model->BSIM4v3pucGiven)
            model->BSIM4v3puc = 0.0;
        if (!model->BSIM4v3puc1Given)
            model->BSIM4v3puc1 = 0.0;
        if (!model->BSIM4v3pu0Given)
            model->BSIM4v3pu0 = 0.0;
        if (!model->BSIM4v3puteGiven)
	    model->BSIM4v3pute = 0.0;    
        if (!model->BSIM4v3pvoffGiven)
	    model->BSIM4v3pvoff = 0.0;
        if (!model->BSIM4v3pminvGiven)
            model->BSIM4v3pminv = 0.0;
        if (!model->BSIM4v3pfproutGiven)
            model->BSIM4v3pfprout = 0.0;
        if (!model->BSIM4v3ppditsGiven)
            model->BSIM4v3ppdits = 0.0;
        if (!model->BSIM4v3ppditsdGiven)
            model->BSIM4v3ppditsd = 0.0;
        if (!model->BSIM4v3pdeltaGiven)  
            model->BSIM4v3pdelta = 0.0;
        if (!model->BSIM4v3prdswGiven)
            model->BSIM4v3prdsw = 0.0;
        if (!model->BSIM4v3prdwGiven)
            model->BSIM4v3prdw = 0.0;
        if (!model->BSIM4v3prswGiven)
            model->BSIM4v3prsw = 0.0;
        if (!model->BSIM4v3pprwbGiven)
            model->BSIM4v3pprwb = 0.0;
        if (!model->BSIM4v3pprwgGiven)
            model->BSIM4v3pprwg = 0.0;
        if (!model->BSIM4v3pprtGiven)
            model->BSIM4v3pprt = 0.0;
        if (!model->BSIM4v3peta0Given)
            model->BSIM4v3peta0 = 0.0;
        if (!model->BSIM4v3petabGiven)
            model->BSIM4v3petab = 0.0;
        if (!model->BSIM4v3ppclmGiven)
            model->BSIM4v3ppclm = 0.0; 
        if (!model->BSIM4v3ppdibl1Given)
            model->BSIM4v3ppdibl1 = 0.0;
        if (!model->BSIM4v3ppdibl2Given)
            model->BSIM4v3ppdibl2 = 0.0;
        if (!model->BSIM4v3ppdiblbGiven)
            model->BSIM4v3ppdiblb = 0.0;
        if (!model->BSIM4v3ppscbe1Given)
            model->BSIM4v3ppscbe1 = 0.0;
        if (!model->BSIM4v3ppscbe2Given)
            model->BSIM4v3ppscbe2 = 0.0;
        if (!model->BSIM4v3ppvagGiven)
            model->BSIM4v3ppvag = 0.0;     
        if (!model->BSIM4v3pwrGiven)  
            model->BSIM4v3pwr = 0.0;
        if (!model->BSIM4v3pdwgGiven)  
            model->BSIM4v3pdwg = 0.0;
        if (!model->BSIM4v3pdwbGiven)  
            model->BSIM4v3pdwb = 0.0;
        if (!model->BSIM4v3pb0Given)
            model->BSIM4v3pb0 = 0.0;
        if (!model->BSIM4v3pb1Given)  
            model->BSIM4v3pb1 = 0.0;
        if (!model->BSIM4v3palpha0Given)  
            model->BSIM4v3palpha0 = 0.0;
        if (!model->BSIM4v3palpha1Given)
            model->BSIM4v3palpha1 = 0.0;
        if (!model->BSIM4v3pbeta0Given)  
            model->BSIM4v3pbeta0 = 0.0;
        if (!model->BSIM4v3pagidlGiven)
            model->BSIM4v3pagidl = 0.0;
        if (!model->BSIM4v3pbgidlGiven)
            model->BSIM4v3pbgidl = 0.0;
        if (!model->BSIM4v3pcgidlGiven)
            model->BSIM4v3pcgidl = 0.0;
        if (!model->BSIM4v3pegidlGiven)
            model->BSIM4v3pegidl = 0.0;
        if (!model->BSIM4v3paigcGiven)
            model->BSIM4v3paigc = 0.0;
        if (!model->BSIM4v3pbigcGiven)
            model->BSIM4v3pbigc = 0.0;
        if (!model->BSIM4v3pcigcGiven)
            model->BSIM4v3pcigc = 0.0;
        if (!model->BSIM4v3paigsdGiven)
            model->BSIM4v3paigsd = 0.0;
        if (!model->BSIM4v3pbigsdGiven)
            model->BSIM4v3pbigsd = 0.0;
        if (!model->BSIM4v3pcigsdGiven)
            model->BSIM4v3pcigsd = 0.0;
        if (!model->BSIM4v3paigbaccGiven)
            model->BSIM4v3paigbacc = 0.0;
        if (!model->BSIM4v3pbigbaccGiven)
            model->BSIM4v3pbigbacc = 0.0;
        if (!model->BSIM4v3pcigbaccGiven)
            model->BSIM4v3pcigbacc = 0.0;
        if (!model->BSIM4v3paigbinvGiven)
            model->BSIM4v3paigbinv = 0.0;
        if (!model->BSIM4v3pbigbinvGiven)
            model->BSIM4v3pbigbinv = 0.0;
        if (!model->BSIM4v3pcigbinvGiven)
            model->BSIM4v3pcigbinv = 0.0;
        if (!model->BSIM4v3pnigcGiven)
            model->BSIM4v3pnigc = 0.0;
        if (!model->BSIM4v3pnigbinvGiven)
            model->BSIM4v3pnigbinv = 0.0;
        if (!model->BSIM4v3pnigbaccGiven)
            model->BSIM4v3pnigbacc = 0.0;
        if (!model->BSIM4v3pntoxGiven)
            model->BSIM4v3pntox = 0.0;
        if (!model->BSIM4v3peigbinvGiven)
            model->BSIM4v3peigbinv = 0.0;
        if (!model->BSIM4v3ppigcdGiven)
            model->BSIM4v3ppigcd = 0.0;
        if (!model->BSIM4v3ppoxedgeGiven)
            model->BSIM4v3ppoxedge = 0.0;
        if (!model->BSIM4v3pxrcrg1Given)
            model->BSIM4v3pxrcrg1 = 0.0;
        if (!model->BSIM4v3pxrcrg2Given)
            model->BSIM4v3pxrcrg2 = 0.0;
        if (!model->BSIM4v3peuGiven)
            model->BSIM4v3peu = 0.0;
        if (!model->BSIM4v3pvfbGiven)
            model->BSIM4v3pvfb = 0.0;
        if (!model->BSIM4v3plambdaGiven)
            model->BSIM4v3plambda = 0.0;
        if (!model->BSIM4v3pvtlGiven)
            model->BSIM4v3pvtl = 0.0;  
        if (!model->BSIM4v3pxnGiven)
            model->BSIM4v3pxn = 0.0;  

        if (!model->BSIM4v3pcgslGiven)  
            model->BSIM4v3pcgsl = 0.0;
        if (!model->BSIM4v3pcgdlGiven)  
            model->BSIM4v3pcgdl = 0.0;
        if (!model->BSIM4v3pckappasGiven)  
            model->BSIM4v3pckappas = 0.0;
        if (!model->BSIM4v3pckappadGiven)
            model->BSIM4v3pckappad = 0.0;
        if (!model->BSIM4v3pcfGiven)  
            model->BSIM4v3pcf = 0.0;
        if (!model->BSIM4v3pclcGiven)  
            model->BSIM4v3pclc = 0.0;
        if (!model->BSIM4v3pcleGiven)  
            model->BSIM4v3pcle = 0.0;
        if (!model->BSIM4v3pvfbcvGiven)  
            model->BSIM4v3pvfbcv = 0.0;
        if (!model->BSIM4v3pacdeGiven)
            model->BSIM4v3pacde = 0.0;
        if (!model->BSIM4v3pmoinGiven)
            model->BSIM4v3pmoin = 0.0;
        if (!model->BSIM4v3pnoffGiven)
            model->BSIM4v3pnoff = 0.0;
        if (!model->BSIM4v3pvoffcvGiven)
            model->BSIM4v3pvoffcv = 0.0;

        /* unit degree celcius */
        if (!model->BSIM4v3tnomGiven)  
	    model->BSIM4v3tnom = ckt->CKTnomTemp; 
        if (!model->BSIM4v3LintGiven)  
           model->BSIM4v3Lint = 0.0;
        if (!model->BSIM4v3LlGiven)  
           model->BSIM4v3Ll = 0.0;
        if (!model->BSIM4v3LlcGiven)
           model->BSIM4v3Llc = model->BSIM4v3Ll;
        if (!model->BSIM4v3LlnGiven)  
           model->BSIM4v3Lln = 1.0;
        if (!model->BSIM4v3LwGiven)  
           model->BSIM4v3Lw = 0.0;
        if (!model->BSIM4v3LwcGiven)
           model->BSIM4v3Lwc = model->BSIM4v3Lw;
        if (!model->BSIM4v3LwnGiven)  
           model->BSIM4v3Lwn = 1.0;
        if (!model->BSIM4v3LwlGiven)  
           model->BSIM4v3Lwl = 0.0;
        if (!model->BSIM4v3LwlcGiven)
           model->BSIM4v3Lwlc = model->BSIM4v3Lwl;
        if (!model->BSIM4v3LminGiven)  
           model->BSIM4v3Lmin = 0.0;
        if (!model->BSIM4v3LmaxGiven)  
           model->BSIM4v3Lmax = 1.0;
        if (!model->BSIM4v3WintGiven)  
           model->BSIM4v3Wint = 0.0;
        if (!model->BSIM4v3WlGiven)  
           model->BSIM4v3Wl = 0.0;
        if (!model->BSIM4v3WlcGiven)
           model->BSIM4v3Wlc = model->BSIM4v3Wl;
        if (!model->BSIM4v3WlnGiven)  
           model->BSIM4v3Wln = 1.0;
        if (!model->BSIM4v3WwGiven)  
           model->BSIM4v3Ww = 0.0;
        if (!model->BSIM4v3WwcGiven)
           model->BSIM4v3Wwc = model->BSIM4v3Ww;
        if (!model->BSIM4v3WwnGiven)  
           model->BSIM4v3Wwn = 1.0;
        if (!model->BSIM4v3WwlGiven)  
           model->BSIM4v3Wwl = 0.0;
        if (!model->BSIM4v3WwlcGiven)
           model->BSIM4v3Wwlc = model->BSIM4v3Wwl;
        if (!model->BSIM4v3WminGiven)  
           model->BSIM4v3Wmin = 0.0;
        if (!model->BSIM4v3WmaxGiven)  
           model->BSIM4v3Wmax = 1.0;
        if (!model->BSIM4v3dwcGiven)  
           model->BSIM4v3dwc = model->BSIM4v3Wint;
        if (!model->BSIM4v3dlcGiven)  
           model->BSIM4v3dlc = model->BSIM4v3Lint;
        if (!model->BSIM4v3xlGiven)  
           model->BSIM4v3xl = 0.0;
        if (!model->BSIM4v3xwGiven)  
           model->BSIM4v3xw = 0.0;
        if (!model->BSIM4v3dlcigGiven)
           model->BSIM4v3dlcig = model->BSIM4v3Lint;
        if (!model->BSIM4v3dwjGiven)
           model->BSIM4v3dwj = model->BSIM4v3dwc;
	if (!model->BSIM4v3cfGiven)
           model->BSIM4v3cf = 2.0 * model->BSIM4v3epsrox * EPS0 / PI
		          * log(1.0 + 0.4e-6 / model->BSIM4v3toxe);

        if (!model->BSIM4v3xpartGiven)
            model->BSIM4v3xpart = 0.0;
        if (!model->BSIM4v3sheetResistanceGiven)
            model->BSIM4v3sheetResistance = 0.0;

        if (!model->BSIM4v3SunitAreaJctCapGiven)
            model->BSIM4v3SunitAreaJctCap = 5.0E-4;
        if (!model->BSIM4v3DunitAreaJctCapGiven)
            model->BSIM4v3DunitAreaJctCap = model->BSIM4v3SunitAreaJctCap;
        if (!model->BSIM4v3SunitLengthSidewallJctCapGiven)
            model->BSIM4v3SunitLengthSidewallJctCap = 5.0E-10;
        if (!model->BSIM4v3DunitLengthSidewallJctCapGiven)
            model->BSIM4v3DunitLengthSidewallJctCap = model->BSIM4v3SunitLengthSidewallJctCap;
        if (!model->BSIM4v3SunitLengthGateSidewallJctCapGiven)
            model->BSIM4v3SunitLengthGateSidewallJctCap = model->BSIM4v3SunitLengthSidewallJctCap ;
        if (!model->BSIM4v3DunitLengthGateSidewallJctCapGiven)
            model->BSIM4v3DunitLengthGateSidewallJctCap = model->BSIM4v3SunitLengthGateSidewallJctCap;
        if (!model->BSIM4v3SjctSatCurDensityGiven)
            model->BSIM4v3SjctSatCurDensity = 1.0E-4;
        if (!model->BSIM4v3DjctSatCurDensityGiven)
            model->BSIM4v3DjctSatCurDensity = model->BSIM4v3SjctSatCurDensity;
        if (!model->BSIM4v3SjctSidewallSatCurDensityGiven)
            model->BSIM4v3SjctSidewallSatCurDensity = 0.0;
        if (!model->BSIM4v3DjctSidewallSatCurDensityGiven)
            model->BSIM4v3DjctSidewallSatCurDensity = model->BSIM4v3SjctSidewallSatCurDensity;
        if (!model->BSIM4v3SjctGateSidewallSatCurDensityGiven)
            model->BSIM4v3SjctGateSidewallSatCurDensity = 0.0;
        if (!model->BSIM4v3DjctGateSidewallSatCurDensityGiven)
            model->BSIM4v3DjctGateSidewallSatCurDensity = model->BSIM4v3SjctGateSidewallSatCurDensity;
        if (!model->BSIM4v3SbulkJctPotentialGiven)
            model->BSIM4v3SbulkJctPotential = 1.0;
        if (!model->BSIM4v3DbulkJctPotentialGiven)
            model->BSIM4v3DbulkJctPotential = model->BSIM4v3SbulkJctPotential;
        if (!model->BSIM4v3SsidewallJctPotentialGiven)
            model->BSIM4v3SsidewallJctPotential = 1.0;
        if (!model->BSIM4v3DsidewallJctPotentialGiven)
            model->BSIM4v3DsidewallJctPotential = model->BSIM4v3SsidewallJctPotential;
        if (!model->BSIM4v3SGatesidewallJctPotentialGiven)
            model->BSIM4v3SGatesidewallJctPotential = model->BSIM4v3SsidewallJctPotential;
        if (!model->BSIM4v3DGatesidewallJctPotentialGiven)
            model->BSIM4v3DGatesidewallJctPotential = model->BSIM4v3SGatesidewallJctPotential;
        if (!model->BSIM4v3SbulkJctBotGradingCoeffGiven)
            model->BSIM4v3SbulkJctBotGradingCoeff = 0.5;
        if (!model->BSIM4v3DbulkJctBotGradingCoeffGiven)
            model->BSIM4v3DbulkJctBotGradingCoeff = model->BSIM4v3SbulkJctBotGradingCoeff;
        if (!model->BSIM4v3SbulkJctSideGradingCoeffGiven)
            model->BSIM4v3SbulkJctSideGradingCoeff = 0.33;
        if (!model->BSIM4v3DbulkJctSideGradingCoeffGiven)
            model->BSIM4v3DbulkJctSideGradingCoeff = model->BSIM4v3SbulkJctSideGradingCoeff;
        if (!model->BSIM4v3SbulkJctGateSideGradingCoeffGiven)
            model->BSIM4v3SbulkJctGateSideGradingCoeff = model->BSIM4v3SbulkJctSideGradingCoeff;
        if (!model->BSIM4v3DbulkJctGateSideGradingCoeffGiven)
            model->BSIM4v3DbulkJctGateSideGradingCoeff = model->BSIM4v3SbulkJctGateSideGradingCoeff;
        if (!model->BSIM4v3SjctEmissionCoeffGiven)
            model->BSIM4v3SjctEmissionCoeff = 1.0;
        if (!model->BSIM4v3DjctEmissionCoeffGiven)
            model->BSIM4v3DjctEmissionCoeff = model->BSIM4v3SjctEmissionCoeff;
        if (!model->BSIM4v3SjctTempExponentGiven)
            model->BSIM4v3SjctTempExponent = 3.0;
        if (!model->BSIM4v3DjctTempExponentGiven)
            model->BSIM4v3DjctTempExponent = model->BSIM4v3SjctTempExponent;

        if (!model->BSIM4v3oxideTrapDensityAGiven)
	{   if (model->BSIM4v3type == NMOS)
                model->BSIM4v3oxideTrapDensityA = 6.25e41;
            else
                model->BSIM4v3oxideTrapDensityA= 6.188e40;
	}
        if (!model->BSIM4v3oxideTrapDensityBGiven)
	{   if (model->BSIM4v3type == NMOS)
                model->BSIM4v3oxideTrapDensityB = 3.125e26;
            else
                model->BSIM4v3oxideTrapDensityB = 1.5e25;
	}
        if (!model->BSIM4v3oxideTrapDensityCGiven)
            model->BSIM4v3oxideTrapDensityC = 8.75e9;
        if (!model->BSIM4v3emGiven)
            model->BSIM4v3em = 4.1e7; /* V/m */
        if (!model->BSIM4v3efGiven)
            model->BSIM4v3ef = 1.0;
        if (!model->BSIM4v3afGiven)
            model->BSIM4v3af = 1.0;
        if (!model->BSIM4v3kfGiven)
            model->BSIM4v3kf = 0.0;

        /* stress effect */
        if (!model->BSIM4v3sarefGiven)
            model->BSIM4v3saref = 1e-6; /* m */
        if (!model->BSIM4v3sbrefGiven)
            model->BSIM4v3sbref = 1e-6;  /* m */
        if (!model->BSIM4v3wlodGiven)
            model->BSIM4v3wlod = 0;  /* m */
        if (!model->BSIM4v3ku0Given)
            model->BSIM4v3ku0 = 0; /* 1/m */
        if (!model->BSIM4v3kvsatGiven)
            model->BSIM4v3kvsat = 0;
        if (!model->BSIM4v3kvth0Given) /* m */
            model->BSIM4v3kvth0 = 0;
        if (!model->BSIM4v3tku0Given)
            model->BSIM4v3tku0 = 0;
        if (!model->BSIM4v3llodku0Given)
            model->BSIM4v3llodku0 = 0;
        if (!model->BSIM4v3wlodku0Given)
            model->BSIM4v3wlodku0 = 0;
        if (!model->BSIM4v3llodvthGiven)
            model->BSIM4v3llodvth = 0;
        if (!model->BSIM4v3wlodvthGiven)
            model->BSIM4v3wlodvth = 0;
        if (!model->BSIM4v3lku0Given)
            model->BSIM4v3lku0 = 0;
        if (!model->BSIM4v3wku0Given)
            model->BSIM4v3wku0 = 0;
        if (!model->BSIM4v3pku0Given)
            model->BSIM4v3pku0 = 0;
        if (!model->BSIM4v3lkvth0Given)
            model->BSIM4v3lkvth0 = 0;
        if (!model->BSIM4v3wkvth0Given)
            model->BSIM4v3wkvth0 = 0;
        if (!model->BSIM4v3pkvth0Given)
            model->BSIM4v3pkvth0 = 0;
        if (!model->BSIM4v3stk2Given)
            model->BSIM4v3stk2 = 0;
        if (!model->BSIM4v3lodk2Given)
            model->BSIM4v3lodk2 = 1.0;
        if (!model->BSIM4v3steta0Given)
            model->BSIM4v3steta0 = 0;
        if (!model->BSIM4v3lodeta0Given)
            model->BSIM4v3lodeta0 = 1.0;

        DMCGeff = model->BSIM4v3dmcg - model->BSIM4v3dmcgt;
        DMCIeff = model->BSIM4v3dmci;
        DMDGeff = model->BSIM4v3dmdg - model->BSIM4v3dmcgt;

	/*
         * End processing models and begin to loop
         * through all the instances of the model
         */

        for (here = model->BSIM4v3instances; here != NULL ;
             here=here->BSIM4v3nextInstance) 
	{   /* allocate a chunk of the state vector */
	  if ( here->BSIM4v3owner == ARCHme) {
            here->BSIM4v3states = *states;
            *states += BSIM4v3numStates;
	  }
            /* perform the parameter defaulting */
            if (!here->BSIM4v3lGiven)
                here->BSIM4v3l = 5.0e-6;
            if (!here->BSIM4v3wGiven)
                here->BSIM4v3w = 5.0e-6;
            if (!here->BSIM4v3mGiven)
                here->BSIM4v3m = 1.0;
            if (!here->BSIM4v3nfGiven)
                here->BSIM4v3nf = 1.0;
            if (!here->BSIM4v3minGiven)
                here->BSIM4v3min = 0; /* integer */
            if (!here->BSIM4v3icVDSGiven)
                here->BSIM4v3icVDS = 0.0;
            if (!here->BSIM4v3icVGSGiven)
                here->BSIM4v3icVGS = 0.0;
            if (!here->BSIM4v3icVBSGiven)
                here->BSIM4v3icVBS = 0.0;
            if (!here->BSIM4v3drainAreaGiven)
                here->BSIM4v3drainArea = 0.0;
            if (!here->BSIM4v3drainPerimeterGiven)
                here->BSIM4v3drainPerimeter = 0.0;
            if (!here->BSIM4v3drainSquaresGiven)
                here->BSIM4v3drainSquares = 1.0;
            if (!here->BSIM4v3sourceAreaGiven)
                here->BSIM4v3sourceArea = 0.0;
            if (!here->BSIM4v3sourcePerimeterGiven)
                here->BSIM4v3sourcePerimeter = 0.0;
            if (!here->BSIM4v3sourceSquaresGiven)
                here->BSIM4v3sourceSquares = 1.0;

            if (!here->BSIM4v3saGiven)
                here->BSIM4v3sa = 0.0;
            if (!here->BSIM4v3sbGiven)
                here->BSIM4v3sb = 0.0;
            if (!here->BSIM4v3sdGiven)
                here->BSIM4v3sd = 0.0;

            if (!here->BSIM4v3rbdbGiven)
                here->BSIM4v3rbdb = model->BSIM4v3rbdb; /* in ohm */
            if (!here->BSIM4v3rbsbGiven)
                here->BSIM4v3rbsb = model->BSIM4v3rbsb;
            if (!here->BSIM4v3rbpbGiven)
                here->BSIM4v3rbpb = model->BSIM4v3rbpb;
            if (!here->BSIM4v3rbpsGiven)
                here->BSIM4v3rbps = model->BSIM4v3rbps;
            if (!here->BSIM4v3rbpdGiven)
                here->BSIM4v3rbpd = model->BSIM4v3rbpd;

                    
            /* Process instance model selectors, some
             * may override their global counterparts
	     */
            if (!here->BSIM4v3rbodyModGiven)
                here->BSIM4v3rbodyMod = model->BSIM4v3rbodyMod;
            else if ((here->BSIM4v3rbodyMod != 0) && (here->BSIM4v3rbodyMod != 1))
            {   here->BSIM4v3rbodyMod = model->BSIM4v3rbodyMod;
                printf("Warning: rbodyMod has been set to its global value %d.\n",
	        model->BSIM4v3rbodyMod);
            }

            if (!here->BSIM4v3rgateModGiven)
                here->BSIM4v3rgateMod = model->BSIM4v3rgateMod;
            else if ((here->BSIM4v3rgateMod != 0) && (here->BSIM4v3rgateMod != 1)
	        && (here->BSIM4v3rgateMod != 2) && (here->BSIM4v3rgateMod != 3))
            {   here->BSIM4v3rgateMod = model->BSIM4v3rgateMod;
                printf("Warning: rgateMod has been set to its global value %d.\n",
                model->BSIM4v3rgateMod);
            }

            if (!here->BSIM4v3geoModGiven)
                here->BSIM4v3geoMod = model->BSIM4v3geoMod;
            if (!here->BSIM4v3rgeoModGiven)
                here->BSIM4v3rgeoMod = 0;
            if (!here->BSIM4v3trnqsModGiven)
                here->BSIM4v3trnqsMod = model->BSIM4v3trnqsMod;
            else if ((here->BSIM4v3trnqsMod != 0) && (here->BSIM4v3trnqsMod != 1))
            {   here->BSIM4v3trnqsMod = model->BSIM4v3trnqsMod;
                printf("Warning: trnqsMod has been set to its global value %d.\n",
                model->BSIM4v3trnqsMod);
            }

            if (!here->BSIM4v3acnqsModGiven)
                here->BSIM4v3acnqsMod = model->BSIM4v3acnqsMod;
            else if ((here->BSIM4v3acnqsMod != 0) && (here->BSIM4v3acnqsMod != 1))
            {   here->BSIM4v3acnqsMod = model->BSIM4v3acnqsMod;
                printf("Warning: acnqsMod has been set to its global value %d.\n",
                model->BSIM4v3acnqsMod);
            }

            /* stress effect */
            if (!here->BSIM4v3saGiven)
                here->BSIM4v3sa = 0.0;
            if (!here->BSIM4v3sbGiven)
                here->BSIM4v3sb = 0.0;
            if (!here->BSIM4v3sdGiven)
                here->BSIM4v3sd = 0.0;


            /* process drain series resistance */
            createNode = 0;
            if ( (model->BSIM4v3rdsMod != 0)
                            || (model->BSIM4v3tnoiMod != 0 && noiseAnalGiven))
            {
               createNode = 1;
            } else if (model->BSIM4v3sheetResistance > 0)
            {
                     if (here->BSIM4v3drainSquaresGiven
                                       && here->BSIM4v3drainSquares > 0)
                     {
                          createNode = 1;
                     } else if (!here->BSIM4v3drainSquaresGiven
                                       && (here->BSIM4v3rgeoMod != 0))
                     {
                          BSIM4v3RdseffGeo(here->BSIM4v3nf, here->BSIM4v3geoMod,
                                  here->BSIM4v3rgeoMod, here->BSIM4v3min,
                                  here->BSIM4v3w, model->BSIM4v3sheetResistance,
                                  DMCGeff, DMCIeff, DMDGeff, 0, &Rtot);
                          if(Rtot > 0)
                             createNode = 1;
                     }
            }
            if ( createNode != 0  && (here->BSIM4v3dNodePrime == 0))
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v3name,"drain");
                if(error) return(error);
                here->BSIM4v3dNodePrime = tmp->number;
            }
            else
            {   here->BSIM4v3dNodePrime = here->BSIM4v3dNode;
            }
            
            /* process source series resistance */
            createNode = 0;
            if ( (model->BSIM4v3rdsMod != 0)
                            || (model->BSIM4v3tnoiMod != 0 && noiseAnalGiven))
            {
               createNode = 1;
            } else if (model->BSIM4v3sheetResistance > 0)
            {
                     if (here->BSIM4v3sourceSquaresGiven
                                        && here->BSIM4v3sourceSquares > 0)
                     {
                          createNode = 1;
                     } else if (!here->BSIM4v3sourceSquaresGiven
                                        && (here->BSIM4v3rgeoMod != 0))
                     {
                          BSIM4v3RdseffGeo(here->BSIM4v3nf, here->BSIM4v3geoMod,
                                  here->BSIM4v3rgeoMod, here->BSIM4v3min,
                                  here->BSIM4v3w, model->BSIM4v3sheetResistance,
                                  DMCGeff, DMCIeff, DMDGeff, 1, &Rtot);
                          if(Rtot > 0)
                             createNode = 1;
                     }
            }
            if ( createNode != 0  && here->BSIM4v3sNodePrime == 0)
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v3name,"source");
                if(error) return(error);
                here->BSIM4v3sNodePrime = tmp->number;
            }
            else
                here->BSIM4v3sNodePrime = here->BSIM4v3sNode;

            if ((here->BSIM4v3rgateMod > 0) && (here->BSIM4v3gNodePrime == 0))
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v3name,"gate");
                if(error) return(error);
                   here->BSIM4v3gNodePrime = tmp->number;
            }
            else
                here->BSIM4v3gNodePrime = here->BSIM4v3gNodeExt;

            if ((here->BSIM4v3rgateMod == 3) && (here->BSIM4v3gNodeMid == 0))
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v3name,"midgate");
                if(error) return(error);
                   here->BSIM4v3gNodeMid = tmp->number;
            }
            else
                here->BSIM4v3gNodeMid = here->BSIM4v3gNodeExt;
            

            /* internal body nodes for body resistance model */
            if (here->BSIM4v3rbodyMod)
            {   if (here->BSIM4v3dbNode == 0)
		{   error = CKTmkVolt(ckt,&tmp,here->BSIM4v3name,"dbody");
                    if(error) return(error);
                    here->BSIM4v3dbNode = tmp->number;
		}
		if (here->BSIM4v3bNodePrime == 0)
                {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v3name,"body");
                    if(error) return(error);
                    here->BSIM4v3bNodePrime = tmp->number;
                }
		if (here->BSIM4v3sbNode == 0)
                {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v3name,"sbody");
                    if(error) return(error);
                    here->BSIM4v3sbNode = tmp->number;
                }
            }
	    else
	        here->BSIM4v3dbNode = here->BSIM4v3bNodePrime = here->BSIM4v3sbNode
				  = here->BSIM4v3bNode;

            /* NQS node */
            if ((here->BSIM4v3trnqsMod) && (here->BSIM4v3qNode == 0)) 
	    {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v3name,"charge");
                if(error) return(error);
                here->BSIM4v3qNode = tmp->number;
            }
	    else 
	        here->BSIM4v3qNode = 0;

/* set Sparse Matrix Pointers 
 * macro to make elements with built-in out-of-memory test */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
}

            TSTALLOC(BSIM4v3DPbpPtr, BSIM4v3dNodePrime, BSIM4v3bNodePrime)
            TSTALLOC(BSIM4v3GPbpPtr, BSIM4v3gNodePrime, BSIM4v3bNodePrime)
            TSTALLOC(BSIM4v3SPbpPtr, BSIM4v3sNodePrime, BSIM4v3bNodePrime)

            TSTALLOC(BSIM4v3BPdpPtr, BSIM4v3bNodePrime, BSIM4v3dNodePrime)
            TSTALLOC(BSIM4v3BPgpPtr, BSIM4v3bNodePrime, BSIM4v3gNodePrime)
            TSTALLOC(BSIM4v3BPspPtr, BSIM4v3bNodePrime, BSIM4v3sNodePrime)
            TSTALLOC(BSIM4v3BPbpPtr, BSIM4v3bNodePrime, BSIM4v3bNodePrime)

            TSTALLOC(BSIM4v3DdPtr, BSIM4v3dNode, BSIM4v3dNode)
            TSTALLOC(BSIM4v3GPgpPtr, BSIM4v3gNodePrime, BSIM4v3gNodePrime)
            TSTALLOC(BSIM4v3SsPtr, BSIM4v3sNode, BSIM4v3sNode)
            TSTALLOC(BSIM4v3DPdpPtr, BSIM4v3dNodePrime, BSIM4v3dNodePrime)
            TSTALLOC(BSIM4v3SPspPtr, BSIM4v3sNodePrime, BSIM4v3sNodePrime)
            TSTALLOC(BSIM4v3DdpPtr, BSIM4v3dNode, BSIM4v3dNodePrime)
            TSTALLOC(BSIM4v3GPdpPtr, BSIM4v3gNodePrime, BSIM4v3dNodePrime)
            TSTALLOC(BSIM4v3GPspPtr, BSIM4v3gNodePrime, BSIM4v3sNodePrime)
            TSTALLOC(BSIM4v3SspPtr, BSIM4v3sNode, BSIM4v3sNodePrime)
            TSTALLOC(BSIM4v3DPspPtr, BSIM4v3dNodePrime, BSIM4v3sNodePrime)
            TSTALLOC(BSIM4v3DPdPtr, BSIM4v3dNodePrime, BSIM4v3dNode)
            TSTALLOC(BSIM4v3DPgpPtr, BSIM4v3dNodePrime, BSIM4v3gNodePrime)
            TSTALLOC(BSIM4v3SPgpPtr, BSIM4v3sNodePrime, BSIM4v3gNodePrime)
            TSTALLOC(BSIM4v3SPsPtr, BSIM4v3sNodePrime, BSIM4v3sNode)
            TSTALLOC(BSIM4v3SPdpPtr, BSIM4v3sNodePrime, BSIM4v3dNodePrime)

            TSTALLOC(BSIM4v3QqPtr, BSIM4v3qNode, BSIM4v3qNode)
            TSTALLOC(BSIM4v3QbpPtr, BSIM4v3qNode, BSIM4v3bNodePrime) 
            TSTALLOC(BSIM4v3QdpPtr, BSIM4v3qNode, BSIM4v3dNodePrime)
            TSTALLOC(BSIM4v3QspPtr, BSIM4v3qNode, BSIM4v3sNodePrime)
            TSTALLOC(BSIM4v3QgpPtr, BSIM4v3qNode, BSIM4v3gNodePrime)
            TSTALLOC(BSIM4v3DPqPtr, BSIM4v3dNodePrime, BSIM4v3qNode)
            TSTALLOC(BSIM4v3SPqPtr, BSIM4v3sNodePrime, BSIM4v3qNode)
            TSTALLOC(BSIM4v3GPqPtr, BSIM4v3gNodePrime, BSIM4v3qNode)

            if (here->BSIM4v3rgateMod != 0)
            {   TSTALLOC(BSIM4v3GEgePtr, BSIM4v3gNodeExt, BSIM4v3gNodeExt)
                TSTALLOC(BSIM4v3GEgpPtr, BSIM4v3gNodeExt, BSIM4v3gNodePrime)
                TSTALLOC(BSIM4v3GPgePtr, BSIM4v3gNodePrime, BSIM4v3gNodeExt)
		TSTALLOC(BSIM4v3GEdpPtr, BSIM4v3gNodeExt, BSIM4v3dNodePrime)
		TSTALLOC(BSIM4v3GEspPtr, BSIM4v3gNodeExt, BSIM4v3sNodePrime)
		TSTALLOC(BSIM4v3GEbpPtr, BSIM4v3gNodeExt, BSIM4v3bNodePrime)

                TSTALLOC(BSIM4v3GMdpPtr, BSIM4v3gNodeMid, BSIM4v3dNodePrime)
                TSTALLOC(BSIM4v3GMgpPtr, BSIM4v3gNodeMid, BSIM4v3gNodePrime)
                TSTALLOC(BSIM4v3GMgmPtr, BSIM4v3gNodeMid, BSIM4v3gNodeMid)
                TSTALLOC(BSIM4v3GMgePtr, BSIM4v3gNodeMid, BSIM4v3gNodeExt)
                TSTALLOC(BSIM4v3GMspPtr, BSIM4v3gNodeMid, BSIM4v3sNodePrime)
                TSTALLOC(BSIM4v3GMbpPtr, BSIM4v3gNodeMid, BSIM4v3bNodePrime)
                TSTALLOC(BSIM4v3DPgmPtr, BSIM4v3dNodePrime, BSIM4v3gNodeMid)
                TSTALLOC(BSIM4v3GPgmPtr, BSIM4v3gNodePrime, BSIM4v3gNodeMid)
                TSTALLOC(BSIM4v3GEgmPtr, BSIM4v3gNodeExt, BSIM4v3gNodeMid)
                TSTALLOC(BSIM4v3SPgmPtr, BSIM4v3sNodePrime, BSIM4v3gNodeMid)
                TSTALLOC(BSIM4v3BPgmPtr, BSIM4v3bNodePrime, BSIM4v3gNodeMid)
            }	

            if (here->BSIM4v3rbodyMod)
            {   TSTALLOC(BSIM4v3DPdbPtr, BSIM4v3dNodePrime, BSIM4v3dbNode)
                TSTALLOC(BSIM4v3SPsbPtr, BSIM4v3sNodePrime, BSIM4v3sbNode)

                TSTALLOC(BSIM4v3DBdpPtr, BSIM4v3dbNode, BSIM4v3dNodePrime)
                TSTALLOC(BSIM4v3DBdbPtr, BSIM4v3dbNode, BSIM4v3dbNode)
                TSTALLOC(BSIM4v3DBbpPtr, BSIM4v3dbNode, BSIM4v3bNodePrime)
                TSTALLOC(BSIM4v3DBbPtr, BSIM4v3dbNode, BSIM4v3bNode)

                TSTALLOC(BSIM4v3BPdbPtr, BSIM4v3bNodePrime, BSIM4v3dbNode)
                TSTALLOC(BSIM4v3BPbPtr, BSIM4v3bNodePrime, BSIM4v3bNode)
                TSTALLOC(BSIM4v3BPsbPtr, BSIM4v3bNodePrime, BSIM4v3sbNode)

                TSTALLOC(BSIM4v3SBspPtr, BSIM4v3sbNode, BSIM4v3sNodePrime)
                TSTALLOC(BSIM4v3SBbpPtr, BSIM4v3sbNode, BSIM4v3bNodePrime)
                TSTALLOC(BSIM4v3SBbPtr, BSIM4v3sbNode, BSIM4v3bNode)
                TSTALLOC(BSIM4v3SBsbPtr, BSIM4v3sbNode, BSIM4v3sbNode)

                TSTALLOC(BSIM4v3BdbPtr, BSIM4v3bNode, BSIM4v3dbNode)
                TSTALLOC(BSIM4v3BbpPtr, BSIM4v3bNode, BSIM4v3bNodePrime)
                TSTALLOC(BSIM4v3BsbPtr, BSIM4v3bNode, BSIM4v3sbNode)
	        TSTALLOC(BSIM4v3BbPtr, BSIM4v3bNode, BSIM4v3bNode)
	    }

            if (model->BSIM4v3rdsMod)
            {   TSTALLOC(BSIM4v3DgpPtr, BSIM4v3dNode, BSIM4v3gNodePrime)
		TSTALLOC(BSIM4v3DspPtr, BSIM4v3dNode, BSIM4v3sNodePrime)
                TSTALLOC(BSIM4v3DbpPtr, BSIM4v3dNode, BSIM4v3bNodePrime)
                TSTALLOC(BSIM4v3SdpPtr, BSIM4v3sNode, BSIM4v3dNodePrime)
                TSTALLOC(BSIM4v3SgpPtr, BSIM4v3sNode, BSIM4v3gNodePrime)
                TSTALLOC(BSIM4v3SbpPtr, BSIM4v3sNode, BSIM4v3bNodePrime)
            }
        }
    }
    return(OK);
}  

int
BSIM4v3unsetup(
    GENmodel *inModel,
    CKTcircuit *ckt)
{
#ifndef HAS_BATCHSIM
    BSIM4v3model *model;
    BSIM4v3instance *here;

    for (model = (BSIM4v3model *)inModel; model != NULL;
            model = model->BSIM4v3nextModel)
    {
        for (here = model->BSIM4v3instances; here != NULL;
                here=here->BSIM4v3nextInstance)
        {
            if (here->BSIM4v3dNodePrime
                    && here->BSIM4v3dNodePrime != here->BSIM4v3dNode)
            {
                CKTdltNNum(ckt, here->BSIM4v3dNodePrime);
                here->BSIM4v3dNodePrime = 0;
            }
            if (here->BSIM4v3sNodePrime
                    && here->BSIM4v3sNodePrime != here->BSIM4v3sNode)
            {
                CKTdltNNum(ckt, here->BSIM4v3sNodePrime);
                here->BSIM4v3sNodePrime = 0;
            }
        }
    }
#endif
    return OK;
}
