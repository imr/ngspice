/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4set.c of BSIM4.4.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 * Modified by Xuemei Xi, 03/04/2004.
 **********/

#include "ngspice.h"
#include "jobdefs.h"  /* Needed because the model searches for noise Analysis */   
#include "ftedefs.h"  /*                   "       "                          */
#include "smpdefs.h"
#include "cktdefs.h"
#include "bsim4v4def.h"
#include "const.h"
#include "sperror.h"
#include "suffix.h"


#define MAX_EXP 5.834617425e14
#define MIN_EXP 1.713908431e-15
#define EXP_THRESHOLD 34.0
#define EPS0 8.85418e-12
#define EPSSI 1.03594e-10
#define PI 3.141592654
#define Charge_q 1.60219e-19

int
BSIM4V4PAeffGeo(double, int, int, double, double, double, double, double *, double *, double *, double *);
int
BSIM4V4RdseffGeo(double, int, int, int, double, double, double, double, double, int, double *);

int
BSIM4V4setup(matrix,inModel,ckt,states)
SMPmatrix *matrix;
GENmodel *inModel;
CKTcircuit *ckt;
int *states;
{
BSIM4V4model *model = (BSIM4V4model*)inModel;
BSIM4V4instance *here;
int error;
CKTnode *tmp;
int    noiseAnalGiven = 0, createNode;  /* Criteria for new node creation */
double Rtot, DMCGeff, DMCIeff, DMDGeff;
JOB   *job;

    /* Search for a noise analysis request */
    for (job = ((TSKtask *)ft_curckt->ci_curTask)->jobs;job;job = job->JOBnextJob) {
        if(strcmp(job->JOBname,"Noise Analysis")==0) {
            noiseAnalGiven = 1;
            break;
        }
    }

    /*  loop through all the BSIM4V4 device models */
    for( ; model != NULL; model = model->BSIM4V4nextModel )
    {   /* process defaults of model parameters */
        if (!model->BSIM4V4typeGiven)
            model->BSIM4V4type = NMOS;     

        if (!model->BSIM4V4mobModGiven) 
            model->BSIM4V4mobMod = 0;
	else if ((model->BSIM4V4mobMod != 0) && (model->BSIM4V4mobMod != 1)
	         && (model->BSIM4V4mobMod != 2))
	{   model->BSIM4V4mobMod = 0;
            printf("Warning: mobMod has been set to its default value: 0.\n");
	}

        if (!model->BSIM4V4binUnitGiven) 
            model->BSIM4V4binUnit = 1;
        if (!model->BSIM4V4paramChkGiven) 
            model->BSIM4V4paramChk = 1;

        if (!model->BSIM4V4dioModGiven)
            model->BSIM4V4dioMod = 1;
        else if ((model->BSIM4V4dioMod != 0) && (model->BSIM4V4dioMod != 1)
            && (model->BSIM4V4dioMod != 2))
        {   model->BSIM4V4dioMod = 1;
            printf("Warning: dioMod has been set to its default value: 1.\n");
        }

        if (!model->BSIM4V4capModGiven) 
            model->BSIM4V4capMod = 2;
        else if ((model->BSIM4V4capMod != 0) && (model->BSIM4V4capMod != 1)
            && (model->BSIM4V4capMod != 2))
        {   model->BSIM4V4capMod = 2;
            printf("Warning: capMod has been set to its default value: 2.\n");
        }

        if (!model->BSIM4V4rdsModGiven)
            model->BSIM4V4rdsMod = 0;
	else if ((model->BSIM4V4rdsMod != 0) && (model->BSIM4V4rdsMod != 1))
        {   model->BSIM4V4rdsMod = 0;
	    printf("Warning: rdsMod has been set to its default value: 0.\n");
	}
        if (!model->BSIM4V4rbodyModGiven)
            model->BSIM4V4rbodyMod = 0;
        else if ((model->BSIM4V4rbodyMod != 0) && (model->BSIM4V4rbodyMod != 1))
        {   model->BSIM4V4rbodyMod = 0;
            printf("Warning: rbodyMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4V4rgateModGiven)
            model->BSIM4V4rgateMod = 0;
        else if ((model->BSIM4V4rgateMod != 0) && (model->BSIM4V4rgateMod != 1)
            && (model->BSIM4V4rgateMod != 2) && (model->BSIM4V4rgateMod != 3))
        {   model->BSIM4V4rgateMod = 0;
            printf("Warning: rgateMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4V4perModGiven)
            model->BSIM4V4perMod = 1;
        else if ((model->BSIM4V4perMod != 0) && (model->BSIM4V4perMod != 1))
        {   model->BSIM4V4perMod = 1;
            printf("Warning: perMod has been set to its default value: 1.\n");
        }

        if (!model->BSIM4V4geoModGiven)
            model->BSIM4V4geoMod = 0;

        if (!model->BSIM4V4fnoiModGiven) 
            model->BSIM4V4fnoiMod = 1;
        else if ((model->BSIM4V4fnoiMod != 0) && (model->BSIM4V4fnoiMod != 1))
        {   model->BSIM4V4fnoiMod = 1;
            printf("Warning: fnoiMod has been set to its default value: 1.\n");
        }
        if (!model->BSIM4V4tnoiModGiven)
            model->BSIM4V4tnoiMod = 0; /* WDLiu: tnoiMod=1 needs to set internal S/D nodes */
        else if ((model->BSIM4V4tnoiMod != 0) && (model->BSIM4V4tnoiMod != 1))
        {   model->BSIM4V4tnoiMod = 0;
            printf("Warning: tnoiMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4V4trnqsModGiven)
            model->BSIM4V4trnqsMod = 0; 
        else if ((model->BSIM4V4trnqsMod != 0) && (model->BSIM4V4trnqsMod != 1))
        {   model->BSIM4V4trnqsMod = 0;
            printf("Warning: trnqsMod has been set to its default value: 0.\n");
        }
        if (!model->BSIM4V4acnqsModGiven)
            model->BSIM4V4acnqsMod = 0;
        else if ((model->BSIM4V4acnqsMod != 0) && (model->BSIM4V4acnqsMod != 1))
        {   model->BSIM4V4acnqsMod = 0;
            printf("Warning: acnqsMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4V4igcModGiven)
            model->BSIM4V4igcMod = 0;
        else if ((model->BSIM4V4igcMod != 0) && (model->BSIM4V4igcMod != 1))
        {   model->BSIM4V4igcMod = 0;
            printf("Warning: igcMod has been set to its default value: 0.\n");
        }
        if (!model->BSIM4V4igbModGiven)
            model->BSIM4V4igbMod = 0;
        else if ((model->BSIM4V4igbMod != 0) && (model->BSIM4V4igbMod != 1))
        {   model->BSIM4V4igbMod = 0;
            printf("Warning: igbMod has been set to its default value: 0.\n");
        }
        if (!model->BSIM4V4tempModGiven)
            model->BSIM4V4tempMod = 0;
        else if ((model->BSIM4V4tempMod != 0) && (model->BSIM4V4tempMod != 1))
        {   model->BSIM4V4tempMod = 0;
            printf("Warning: tempMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4V4versionGiven) 
            model->BSIM4V4version = "4.4.0";
        if (!model->BSIM4V4toxrefGiven)
            model->BSIM4V4toxref = 30.0e-10;
        if (!model->BSIM4V4toxeGiven)
            model->BSIM4V4toxe = 30.0e-10;
        if (!model->BSIM4V4toxpGiven)
            model->BSIM4V4toxp = model->BSIM4V4toxe;
        if (!model->BSIM4V4toxmGiven)
            model->BSIM4V4toxm = model->BSIM4V4toxe;
        if (!model->BSIM4V4dtoxGiven)
            model->BSIM4V4dtox = 0.0;
        if (!model->BSIM4V4epsroxGiven)
            model->BSIM4V4epsrox = 3.9;

        if (!model->BSIM4V4cdscGiven)
	    model->BSIM4V4cdsc = 2.4e-4;   /* unit Q/V/m^2  */
        if (!model->BSIM4V4cdscbGiven)
	    model->BSIM4V4cdscb = 0.0;   /* unit Q/V/m^2  */    
	    if (!model->BSIM4V4cdscdGiven)
	    model->BSIM4V4cdscd = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM4V4citGiven)
	    model->BSIM4V4cit = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM4V4nfactorGiven)
	    model->BSIM4V4nfactor = 1.0;
        if (!model->BSIM4V4xjGiven)
            model->BSIM4V4xj = .15e-6;
        if (!model->BSIM4V4vsatGiven)
            model->BSIM4V4vsat = 8.0e4;    /* unit m/s */ 
        if (!model->BSIM4V4atGiven)
            model->BSIM4V4at = 3.3e4;    /* unit m/s */ 
        if (!model->BSIM4V4a0Given)
            model->BSIM4V4a0 = 1.0;  
        if (!model->BSIM4V4agsGiven)
            model->BSIM4V4ags = 0.0;
        if (!model->BSIM4V4a1Given)
            model->BSIM4V4a1 = 0.0;
        if (!model->BSIM4V4a2Given)
            model->BSIM4V4a2 = 1.0;
        if (!model->BSIM4V4ketaGiven)
            model->BSIM4V4keta = -0.047;    /* unit  / V */
        if (!model->BSIM4V4nsubGiven)
            model->BSIM4V4nsub = 6.0e16;   /* unit 1/cm3 */
        if (!model->BSIM4V4ndepGiven)
            model->BSIM4V4ndep = 1.7e17;   /* unit 1/cm3 */
        if (!model->BSIM4V4nsdGiven)
            model->BSIM4V4nsd = 1.0e20;   /* unit 1/cm3 */
        if (!model->BSIM4V4phinGiven)
            model->BSIM4V4phin = 0.0; /* unit V */
        if (!model->BSIM4V4ngateGiven)
            model->BSIM4V4ngate = 0;   /* unit 1/cm3 */
        if (!model->BSIM4V4vbmGiven)
	    model->BSIM4V4vbm = -3.0;
        if (!model->BSIM4V4xtGiven)
	    model->BSIM4V4xt = 1.55e-7;
        if (!model->BSIM4V4kt1Given)
            model->BSIM4V4kt1 = -0.11;      /* unit V */
        if (!model->BSIM4V4kt1lGiven)
            model->BSIM4V4kt1l = 0.0;      /* unit V*m */
        if (!model->BSIM4V4kt2Given)
            model->BSIM4V4kt2 = 0.022;      /* No unit */
        if (!model->BSIM4V4k3Given)
            model->BSIM4V4k3 = 80.0;      
        if (!model->BSIM4V4k3bGiven)
            model->BSIM4V4k3b = 0.0;      
        if (!model->BSIM4V4w0Given)
            model->BSIM4V4w0 = 2.5e-6;    
        if (!model->BSIM4V4lpe0Given)
            model->BSIM4V4lpe0 = 1.74e-7;     
        if (!model->BSIM4V4lpebGiven)
            model->BSIM4V4lpeb = 0.0;
        if (!model->BSIM4V4dvtp0Given)
            model->BSIM4V4dvtp0 = 0.0;
        if (!model->BSIM4V4dvtp1Given)
            model->BSIM4V4dvtp1 = 0.0;
        if (!model->BSIM4V4dvt0Given)
            model->BSIM4V4dvt0 = 2.2;    
        if (!model->BSIM4V4dvt1Given)
            model->BSIM4V4dvt1 = 0.53;      
        if (!model->BSIM4V4dvt2Given)
            model->BSIM4V4dvt2 = -0.032;   /* unit 1 / V */     

        if (!model->BSIM4V4dvt0wGiven)
            model->BSIM4V4dvt0w = 0.0;    
        if (!model->BSIM4V4dvt1wGiven)
            model->BSIM4V4dvt1w = 5.3e6;    
        if (!model->BSIM4V4dvt2wGiven)
            model->BSIM4V4dvt2w = -0.032;   

        if (!model->BSIM4V4droutGiven)
            model->BSIM4V4drout = 0.56;     
        if (!model->BSIM4V4dsubGiven)
            model->BSIM4V4dsub = model->BSIM4V4drout;     
        if (!model->BSIM4V4vth0Given)
            model->BSIM4V4vth0 = (model->BSIM4V4type == NMOS) ? 0.7 : -0.7;
        if (!model->BSIM4V4euGiven)
            model->BSIM4V4eu = (model->BSIM4V4type == NMOS) ? 1.67 : 1.0;;
        if (!model->BSIM4V4uaGiven)
            model->BSIM4V4ua = (model->BSIM4V4mobMod == 2) ? 1.0e-15 : 1.0e-9; /* unit m/V */
        if (!model->BSIM4V4ua1Given)
            model->BSIM4V4ua1 = 1.0e-9;      /* unit m/V */
        if (!model->BSIM4V4ubGiven)
            model->BSIM4V4ub = 1.0e-19;     /* unit (m/V)**2 */
        if (!model->BSIM4V4ub1Given)
            model->BSIM4V4ub1 = -1.0e-18;     /* unit (m/V)**2 */
        if (!model->BSIM4V4ucGiven)
            model->BSIM4V4uc = (model->BSIM4V4mobMod == 1) ? -0.0465 : -0.0465e-9;   
        if (!model->BSIM4V4uc1Given)
            model->BSIM4V4uc1 = (model->BSIM4V4mobMod == 1) ? -0.056 : -0.056e-9;   
        if (!model->BSIM4V4u0Given)
            model->BSIM4V4u0 = (model->BSIM4V4type == NMOS) ? 0.067 : 0.025;
        if (!model->BSIM4V4uteGiven)
	    model->BSIM4V4ute = -1.5;    
        if (!model->BSIM4V4voffGiven)
	    model->BSIM4V4voff = -0.08;
        if (!model->BSIM4V4vofflGiven)
            model->BSIM4V4voffl = 0.0;
        if (!model->BSIM4V4minvGiven)
            model->BSIM4V4minv = 0.0;
        if (!model->BSIM4V4fproutGiven)
            model->BSIM4V4fprout = 0.0;
        if (!model->BSIM4V4pditsGiven)
            model->BSIM4V4pdits = 0.0;
        if (!model->BSIM4V4pditsdGiven)
            model->BSIM4V4pditsd = 0.0;
        if (!model->BSIM4V4pditslGiven)
            model->BSIM4V4pditsl = 0.0;
        if (!model->BSIM4V4deltaGiven)  
           model->BSIM4V4delta = 0.01;
        if (!model->BSIM4V4rdswminGiven)
            model->BSIM4V4rdswmin = 0.0;
        if (!model->BSIM4V4rdwminGiven)
            model->BSIM4V4rdwmin = 0.0;
        if (!model->BSIM4V4rswminGiven)
            model->BSIM4V4rswmin = 0.0;
        if (!model->BSIM4V4rdswGiven)
	    model->BSIM4V4rdsw = 200.0; /* in ohm*um */     
        if (!model->BSIM4V4rdwGiven)
            model->BSIM4V4rdw = 100.0;
        if (!model->BSIM4V4rswGiven)
            model->BSIM4V4rsw = 100.0;
        if (!model->BSIM4V4prwgGiven)
            model->BSIM4V4prwg = 1.0; /* in 1/V */
        if (!model->BSIM4V4prwbGiven)
            model->BSIM4V4prwb = 0.0;      
        if (!model->BSIM4V4prtGiven)
        if (!model->BSIM4V4prtGiven)
            model->BSIM4V4prt = 0.0;      
        if (!model->BSIM4V4eta0Given)
            model->BSIM4V4eta0 = 0.08;      /* no unit  */ 
        if (!model->BSIM4V4etabGiven)
            model->BSIM4V4etab = -0.07;      /* unit  1/V */ 
        if (!model->BSIM4V4pclmGiven)
            model->BSIM4V4pclm = 1.3;      /* no unit  */ 
        if (!model->BSIM4V4pdibl1Given)
            model->BSIM4V4pdibl1 = 0.39;    /* no unit  */
        if (!model->BSIM4V4pdibl2Given)
            model->BSIM4V4pdibl2 = 0.0086;    /* no unit  */ 
        if (!model->BSIM4V4pdiblbGiven)
            model->BSIM4V4pdiblb = 0.0;    /* 1/V  */ 
        if (!model->BSIM4V4pscbe1Given)
            model->BSIM4V4pscbe1 = 4.24e8;     
        if (!model->BSIM4V4pscbe2Given)
            model->BSIM4V4pscbe2 = 1.0e-5;    
        if (!model->BSIM4V4pvagGiven)
            model->BSIM4V4pvag = 0.0;     
        if (!model->BSIM4V4wrGiven)  
            model->BSIM4V4wr = 1.0;
        if (!model->BSIM4V4dwgGiven)  
            model->BSIM4V4dwg = 0.0;
        if (!model->BSIM4V4dwbGiven)  
            model->BSIM4V4dwb = 0.0;
        if (!model->BSIM4V4b0Given)
            model->BSIM4V4b0 = 0.0;
        if (!model->BSIM4V4b1Given)  
            model->BSIM4V4b1 = 0.0;
        if (!model->BSIM4V4alpha0Given)  
            model->BSIM4V4alpha0 = 0.0;
        if (!model->BSIM4V4alpha1Given)
            model->BSIM4V4alpha1 = 0.0;
        if (!model->BSIM4V4beta0Given)  
            model->BSIM4V4beta0 = 30.0;
        if (!model->BSIM4V4agidlGiven)
            model->BSIM4V4agidl = 0.0;
        if (!model->BSIM4V4bgidlGiven)
            model->BSIM4V4bgidl = 2.3e9; /* V/m */
        if (!model->BSIM4V4cgidlGiven)
            model->BSIM4V4cgidl = 0.5; /* V^3 */
        if (!model->BSIM4V4egidlGiven)
            model->BSIM4V4egidl = 0.8; /* V */
        if (!model->BSIM4V4aigcGiven)
            model->BSIM4V4aigc = (model->BSIM4V4type == NMOS) ? 0.43 : 0.31;
        if (!model->BSIM4V4bigcGiven)
            model->BSIM4V4bigc = (model->BSIM4V4type == NMOS) ? 0.054 : 0.024;
        if (!model->BSIM4V4cigcGiven)
            model->BSIM4V4cigc = (model->BSIM4V4type == NMOS) ? 0.075 : 0.03;
        if (!model->BSIM4V4aigsdGiven)
            model->BSIM4V4aigsd = (model->BSIM4V4type == NMOS) ? 0.43 : 0.31;
        if (!model->BSIM4V4bigsdGiven)
            model->BSIM4V4bigsd = (model->BSIM4V4type == NMOS) ? 0.054 : 0.024;
        if (!model->BSIM4V4cigsdGiven)
            model->BSIM4V4cigsd = (model->BSIM4V4type == NMOS) ? 0.075 : 0.03;
        if (!model->BSIM4V4aigbaccGiven)
            model->BSIM4V4aigbacc = 0.43;
        if (!model->BSIM4V4bigbaccGiven)
            model->BSIM4V4bigbacc = 0.054;
        if (!model->BSIM4V4cigbaccGiven)
            model->BSIM4V4cigbacc = 0.075;
        if (!model->BSIM4V4aigbinvGiven)
            model->BSIM4V4aigbinv = 0.35;
        if (!model->BSIM4V4bigbinvGiven)
            model->BSIM4V4bigbinv = 0.03;
        if (!model->BSIM4V4cigbinvGiven)
            model->BSIM4V4cigbinv = 0.006;
        if (!model->BSIM4V4nigcGiven)
            model->BSIM4V4nigc = 1.0;
        if (!model->BSIM4V4nigbinvGiven)
            model->BSIM4V4nigbinv = 3.0;
        if (!model->BSIM4V4nigbaccGiven)
            model->BSIM4V4nigbacc = 1.0;
        if (!model->BSIM4V4ntoxGiven)
            model->BSIM4V4ntox = 1.0;
        if (!model->BSIM4V4eigbinvGiven)
            model->BSIM4V4eigbinv = 1.1;
        if (!model->BSIM4V4pigcdGiven)
            model->BSIM4V4pigcd = 1.0;
        if (!model->BSIM4V4poxedgeGiven)
            model->BSIM4V4poxedge = 1.0;
        if (!model->BSIM4V4xrcrg1Given)
            model->BSIM4V4xrcrg1 = 12.0;
        if (!model->BSIM4V4xrcrg2Given)
            model->BSIM4V4xrcrg2 = 1.0;
        if (!model->BSIM4V4ijthsfwdGiven)
            model->BSIM4V4ijthsfwd = 0.1; /* unit A */
        if (!model->BSIM4V4ijthdfwdGiven)
            model->BSIM4V4ijthdfwd = model->BSIM4V4ijthsfwd;
        if (!model->BSIM4V4ijthsrevGiven)
            model->BSIM4V4ijthsrev = 0.1; /* unit A */
        if (!model->BSIM4V4ijthdrevGiven)
            model->BSIM4V4ijthdrev = model->BSIM4V4ijthsrev;
        if (!model->BSIM4V4tnoiaGiven)
            model->BSIM4V4tnoia = 1.5;
        if (!model->BSIM4V4tnoibGiven)
            model->BSIM4V4tnoib = 3.5;
        if (!model->BSIM4V4rnoiaGiven)
            model->BSIM4V4rnoia = 0.577;
        if (!model->BSIM4V4rnoibGiven)
            model->BSIM4V4rnoib = 0.5164;
        if (!model->BSIM4V4ntnoiGiven)
            model->BSIM4V4ntnoi = 1.0;
        if (!model->BSIM4V4lambdaGiven)
            model->BSIM4V4lambda = 0.0;
        if (!model->BSIM4V4vtlGiven)
            model->BSIM4V4vtl = 2.0e5;    /* unit m/s */ 
        if (!model->BSIM4V4xnGiven)
            model->BSIM4V4xn = 3.0;   
        if (!model->BSIM4V4lcGiven)
            model->BSIM4V4lc = 5.0e-9;   
        if (!model->BSIM4V4vfbsdoffGiven)  
            model->BSIM4V4vfbsdoff = 0.0;  /* unit v */  
        if (!model->BSIM4V4lintnoiGiven)
            model->BSIM4V4lintnoi = 0.0;  /* unit m */  

        if (!model->BSIM4V4xjbvsGiven)
            model->BSIM4V4xjbvs = 1.0; /* no unit */
        if (!model->BSIM4V4xjbvdGiven)
            model->BSIM4V4xjbvd = model->BSIM4V4xjbvs;
        if (!model->BSIM4V4bvsGiven)
            model->BSIM4V4bvs = 10.0; /* V */
        if (!model->BSIM4V4bvdGiven)
            model->BSIM4V4bvd = model->BSIM4V4bvs;

        if (!model->BSIM4V4gbminGiven)
            model->BSIM4V4gbmin = 1.0e-12; /* in mho */
        if (!model->BSIM4V4rbdbGiven)
            model->BSIM4V4rbdb = 50.0; /* in ohm */
        if (!model->BSIM4V4rbpbGiven)
            model->BSIM4V4rbpb = 50.0;
        if (!model->BSIM4V4rbsbGiven)
            model->BSIM4V4rbsb = 50.0;
        if (!model->BSIM4V4rbpsGiven)
            model->BSIM4V4rbps = 50.0;
        if (!model->BSIM4V4rbpdGiven)
            model->BSIM4V4rbpd = 50.0;

        if (!model->BSIM4V4cgslGiven)  
            model->BSIM4V4cgsl = 0.0;
        if (!model->BSIM4V4cgdlGiven)  
            model->BSIM4V4cgdl = 0.0;
        if (!model->BSIM4V4ckappasGiven)  
            model->BSIM4V4ckappas = 0.6;
        if (!model->BSIM4V4ckappadGiven)
            model->BSIM4V4ckappad = model->BSIM4V4ckappas;
        if (!model->BSIM4V4clcGiven)  
            model->BSIM4V4clc = 0.1e-6;
        if (!model->BSIM4V4cleGiven)  
            model->BSIM4V4cle = 0.6;
        if (!model->BSIM4V4vfbcvGiven)  
            model->BSIM4V4vfbcv = -1.0;
        if (!model->BSIM4V4acdeGiven)
            model->BSIM4V4acde = 1.0;
        if (!model->BSIM4V4moinGiven)
            model->BSIM4V4moin = 15.0;
        if (!model->BSIM4V4noffGiven)
            model->BSIM4V4noff = 1.0;
        if (!model->BSIM4V4voffcvGiven)
            model->BSIM4V4voffcv = 0.0;
        if (!model->BSIM4V4dmcgGiven)
            model->BSIM4V4dmcg = 0.0;
        if (!model->BSIM4V4dmciGiven)
            model->BSIM4V4dmci = model->BSIM4V4dmcg;
        if (!model->BSIM4V4dmdgGiven)
            model->BSIM4V4dmdg = 0.0;
        if (!model->BSIM4V4dmcgtGiven)
            model->BSIM4V4dmcgt = 0.0;
        if (!model->BSIM4V4xgwGiven)
            model->BSIM4V4xgw = 0.0;
        if (!model->BSIM4V4xglGiven)
            model->BSIM4V4xgl = 0.0;
        if (!model->BSIM4V4rshgGiven)
            model->BSIM4V4rshg = 0.1;
        if (!model->BSIM4V4ngconGiven)
            model->BSIM4V4ngcon = 1.0;
        if (!model->BSIM4V4tcjGiven)
            model->BSIM4V4tcj = 0.0;
        if (!model->BSIM4V4tpbGiven)
            model->BSIM4V4tpb = 0.0;
        if (!model->BSIM4V4tcjswGiven)
            model->BSIM4V4tcjsw = 0.0;
        if (!model->BSIM4V4tpbswGiven)
            model->BSIM4V4tpbsw = 0.0;
        if (!model->BSIM4V4tcjswgGiven)
            model->BSIM4V4tcjswg = 0.0;
        if (!model->BSIM4V4tpbswgGiven)
            model->BSIM4V4tpbswg = 0.0;

	/* Length dependence */
        if (!model->BSIM4V4lcdscGiven)
	    model->BSIM4V4lcdsc = 0.0;
        if (!model->BSIM4V4lcdscbGiven)
	    model->BSIM4V4lcdscb = 0.0;
	    if (!model->BSIM4V4lcdscdGiven) 
	    model->BSIM4V4lcdscd = 0.0;
        if (!model->BSIM4V4lcitGiven)
	    model->BSIM4V4lcit = 0.0;
        if (!model->BSIM4V4lnfactorGiven)
	    model->BSIM4V4lnfactor = 0.0;
        if (!model->BSIM4V4lxjGiven)
            model->BSIM4V4lxj = 0.0;
        if (!model->BSIM4V4lvsatGiven)
            model->BSIM4V4lvsat = 0.0;
        if (!model->BSIM4V4latGiven)
            model->BSIM4V4lat = 0.0;
        if (!model->BSIM4V4la0Given)
            model->BSIM4V4la0 = 0.0; 
        if (!model->BSIM4V4lagsGiven)
            model->BSIM4V4lags = 0.0;
        if (!model->BSIM4V4la1Given)
            model->BSIM4V4la1 = 0.0;
        if (!model->BSIM4V4la2Given)
            model->BSIM4V4la2 = 0.0;
        if (!model->BSIM4V4lketaGiven)
            model->BSIM4V4lketa = 0.0;
        if (!model->BSIM4V4lnsubGiven)
            model->BSIM4V4lnsub = 0.0;
        if (!model->BSIM4V4lndepGiven)
            model->BSIM4V4lndep = 0.0;
        if (!model->BSIM4V4lnsdGiven)
            model->BSIM4V4lnsd = 0.0;
        if (!model->BSIM4V4lphinGiven)
            model->BSIM4V4lphin = 0.0;
        if (!model->BSIM4V4lngateGiven)
            model->BSIM4V4lngate = 0.0;
        if (!model->BSIM4V4lvbmGiven)
	    model->BSIM4V4lvbm = 0.0;
        if (!model->BSIM4V4lxtGiven)
	    model->BSIM4V4lxt = 0.0;
        if (!model->BSIM4V4lkt1Given)
            model->BSIM4V4lkt1 = 0.0; 
        if (!model->BSIM4V4lkt1lGiven)
            model->BSIM4V4lkt1l = 0.0;
        if (!model->BSIM4V4lkt2Given)
            model->BSIM4V4lkt2 = 0.0;
        if (!model->BSIM4V4lk3Given)
            model->BSIM4V4lk3 = 0.0;      
        if (!model->BSIM4V4lk3bGiven)
            model->BSIM4V4lk3b = 0.0;      
        if (!model->BSIM4V4lw0Given)
            model->BSIM4V4lw0 = 0.0;    
        if (!model->BSIM4V4llpe0Given)
            model->BSIM4V4llpe0 = 0.0;
        if (!model->BSIM4V4llpebGiven)
            model->BSIM4V4llpeb = 0.0; 
        if (!model->BSIM4V4ldvtp0Given)
            model->BSIM4V4ldvtp0 = 0.0;
        if (!model->BSIM4V4ldvtp1Given)
            model->BSIM4V4ldvtp1 = 0.0;
        if (!model->BSIM4V4ldvt0Given)
            model->BSIM4V4ldvt0 = 0.0;    
        if (!model->BSIM4V4ldvt1Given)
            model->BSIM4V4ldvt1 = 0.0;      
        if (!model->BSIM4V4ldvt2Given)
            model->BSIM4V4ldvt2 = 0.0;
        if (!model->BSIM4V4ldvt0wGiven)
            model->BSIM4V4ldvt0w = 0.0;    
        if (!model->BSIM4V4ldvt1wGiven)
            model->BSIM4V4ldvt1w = 0.0;      
        if (!model->BSIM4V4ldvt2wGiven)
            model->BSIM4V4ldvt2w = 0.0;
        if (!model->BSIM4V4ldroutGiven)
            model->BSIM4V4ldrout = 0.0;     
        if (!model->BSIM4V4ldsubGiven)
            model->BSIM4V4ldsub = 0.0;
        if (!model->BSIM4V4lvth0Given)
           model->BSIM4V4lvth0 = 0.0;
        if (!model->BSIM4V4luaGiven)
            model->BSIM4V4lua = 0.0;
        if (!model->BSIM4V4lua1Given)
            model->BSIM4V4lua1 = 0.0;
        if (!model->BSIM4V4lubGiven)
            model->BSIM4V4lub = 0.0;
        if (!model->BSIM4V4lub1Given)
            model->BSIM4V4lub1 = 0.0;
        if (!model->BSIM4V4lucGiven)
            model->BSIM4V4luc = 0.0;
        if (!model->BSIM4V4luc1Given)
            model->BSIM4V4luc1 = 0.0;
        if (!model->BSIM4V4lu0Given)
            model->BSIM4V4lu0 = 0.0;
        if (!model->BSIM4V4luteGiven)
	    model->BSIM4V4lute = 0.0;    
        if (!model->BSIM4V4lvoffGiven)
	    model->BSIM4V4lvoff = 0.0;
        if (!model->BSIM4V4lminvGiven)
            model->BSIM4V4lminv = 0.0;
        if (!model->BSIM4V4lfproutGiven)
            model->BSIM4V4lfprout = 0.0;
        if (!model->BSIM4V4lpditsGiven)
            model->BSIM4V4lpdits = 0.0;
        if (!model->BSIM4V4lpditsdGiven)
            model->BSIM4V4lpditsd = 0.0;
        if (!model->BSIM4V4ldeltaGiven)  
            model->BSIM4V4ldelta = 0.0;
        if (!model->BSIM4V4lrdswGiven)
            model->BSIM4V4lrdsw = 0.0;
        if (!model->BSIM4V4lrdwGiven)
            model->BSIM4V4lrdw = 0.0;
        if (!model->BSIM4V4lrswGiven)
            model->BSIM4V4lrsw = 0.0;
        if (!model->BSIM4V4lprwbGiven)
            model->BSIM4V4lprwb = 0.0;
        if (!model->BSIM4V4lprwgGiven)
            model->BSIM4V4lprwg = 0.0;
        if (!model->BSIM4V4lprtGiven)
            model->BSIM4V4lprt = 0.0;
        if (!model->BSIM4V4leta0Given)
            model->BSIM4V4leta0 = 0.0;
        if (!model->BSIM4V4letabGiven)
            model->BSIM4V4letab = -0.0;
        if (!model->BSIM4V4lpclmGiven)
            model->BSIM4V4lpclm = 0.0; 
        if (!model->BSIM4V4lpdibl1Given)
            model->BSIM4V4lpdibl1 = 0.0;
        if (!model->BSIM4V4lpdibl2Given)
            model->BSIM4V4lpdibl2 = 0.0;
        if (!model->BSIM4V4lpdiblbGiven)
            model->BSIM4V4lpdiblb = 0.0;
        if (!model->BSIM4V4lpscbe1Given)
            model->BSIM4V4lpscbe1 = 0.0;
        if (!model->BSIM4V4lpscbe2Given)
            model->BSIM4V4lpscbe2 = 0.0;
        if (!model->BSIM4V4lpvagGiven)
            model->BSIM4V4lpvag = 0.0;     
        if (!model->BSIM4V4lwrGiven)  
            model->BSIM4V4lwr = 0.0;
        if (!model->BSIM4V4ldwgGiven)  
            model->BSIM4V4ldwg = 0.0;
        if (!model->BSIM4V4ldwbGiven)  
            model->BSIM4V4ldwb = 0.0;
        if (!model->BSIM4V4lb0Given)
            model->BSIM4V4lb0 = 0.0;
        if (!model->BSIM4V4lb1Given)  
            model->BSIM4V4lb1 = 0.0;
        if (!model->BSIM4V4lalpha0Given)  
            model->BSIM4V4lalpha0 = 0.0;
        if (!model->BSIM4V4lalpha1Given)
            model->BSIM4V4lalpha1 = 0.0;
        if (!model->BSIM4V4lbeta0Given)  
            model->BSIM4V4lbeta0 = 0.0;
        if (!model->BSIM4V4lagidlGiven)
            model->BSIM4V4lagidl = 0.0;
        if (!model->BSIM4V4lbgidlGiven)
            model->BSIM4V4lbgidl = 0.0;
        if (!model->BSIM4V4lcgidlGiven)
            model->BSIM4V4lcgidl = 0.0;
        if (!model->BSIM4V4legidlGiven)
            model->BSIM4V4legidl = 0.0;
        if (!model->BSIM4V4laigcGiven)
            model->BSIM4V4laigc = 0.0;
        if (!model->BSIM4V4lbigcGiven)
            model->BSIM4V4lbigc = 0.0;
        if (!model->BSIM4V4lcigcGiven)
            model->BSIM4V4lcigc = 0.0;
        if (!model->BSIM4V4laigsdGiven)
            model->BSIM4V4laigsd = 0.0;
        if (!model->BSIM4V4lbigsdGiven)
            model->BSIM4V4lbigsd = 0.0;
        if (!model->BSIM4V4lcigsdGiven)
            model->BSIM4V4lcigsd = 0.0;
        if (!model->BSIM4V4laigbaccGiven)
            model->BSIM4V4laigbacc = 0.0;
        if (!model->BSIM4V4lbigbaccGiven)
            model->BSIM4V4lbigbacc = 0.0;
        if (!model->BSIM4V4lcigbaccGiven)
            model->BSIM4V4lcigbacc = 0.0;
        if (!model->BSIM4V4laigbinvGiven)
            model->BSIM4V4laigbinv = 0.0;
        if (!model->BSIM4V4lbigbinvGiven)
            model->BSIM4V4lbigbinv = 0.0;
        if (!model->BSIM4V4lcigbinvGiven)
            model->BSIM4V4lcigbinv = 0.0;
        if (!model->BSIM4V4lnigcGiven)
            model->BSIM4V4lnigc = 0.0;
        if (!model->BSIM4V4lnigbinvGiven)
            model->BSIM4V4lnigbinv = 0.0;
        if (!model->BSIM4V4lnigbaccGiven)
            model->BSIM4V4lnigbacc = 0.0;
        if (!model->BSIM4V4lntoxGiven)
            model->BSIM4V4lntox = 0.0;
        if (!model->BSIM4V4leigbinvGiven)
            model->BSIM4V4leigbinv = 0.0;
        if (!model->BSIM4V4lpigcdGiven)
            model->BSIM4V4lpigcd = 0.0;
        if (!model->BSIM4V4lpoxedgeGiven)
            model->BSIM4V4lpoxedge = 0.0;
        if (!model->BSIM4V4lxrcrg1Given)
            model->BSIM4V4lxrcrg1 = 0.0;
        if (!model->BSIM4V4lxrcrg2Given)
            model->BSIM4V4lxrcrg2 = 0.0;
        if (!model->BSIM4V4leuGiven)
            model->BSIM4V4leu = 0.0;
        if (!model->BSIM4V4lvfbGiven)
            model->BSIM4V4lvfb = 0.0;
        if (!model->BSIM4V4llambdaGiven)
            model->BSIM4V4llambda = 0.0;
        if (!model->BSIM4V4lvtlGiven)
            model->BSIM4V4lvtl = 0.0;  
        if (!model->BSIM4V4lxnGiven)
            model->BSIM4V4lxn = 0.0;  
        if (!model->BSIM4V4lvfbsdoffGiven)
            model->BSIM4V4lvfbsdoff = 0.0;   

        if (!model->BSIM4V4lcgslGiven)  
            model->BSIM4V4lcgsl = 0.0;
        if (!model->BSIM4V4lcgdlGiven)  
            model->BSIM4V4lcgdl = 0.0;
        if (!model->BSIM4V4lckappasGiven)  
            model->BSIM4V4lckappas = 0.0;
        if (!model->BSIM4V4lckappadGiven)
            model->BSIM4V4lckappad = 0.0;
        if (!model->BSIM4V4lclcGiven)  
            model->BSIM4V4lclc = 0.0;
        if (!model->BSIM4V4lcleGiven)  
            model->BSIM4V4lcle = 0.0;
        if (!model->BSIM4V4lcfGiven)  
            model->BSIM4V4lcf = 0.0;
        if (!model->BSIM4V4lvfbcvGiven)  
            model->BSIM4V4lvfbcv = 0.0;
        if (!model->BSIM4V4lacdeGiven)
            model->BSIM4V4lacde = 0.0;
        if (!model->BSIM4V4lmoinGiven)
            model->BSIM4V4lmoin = 0.0;
        if (!model->BSIM4V4lnoffGiven)
            model->BSIM4V4lnoff = 0.0;
        if (!model->BSIM4V4lvoffcvGiven)
            model->BSIM4V4lvoffcv = 0.0;

	/* Width dependence */
        if (!model->BSIM4V4wcdscGiven)
	    model->BSIM4V4wcdsc = 0.0;
        if (!model->BSIM4V4wcdscbGiven)
	    model->BSIM4V4wcdscb = 0.0;  
	    if (!model->BSIM4V4wcdscdGiven)
	    model->BSIM4V4wcdscd = 0.0;
        if (!model->BSIM4V4wcitGiven)
	    model->BSIM4V4wcit = 0.0;
        if (!model->BSIM4V4wnfactorGiven)
	    model->BSIM4V4wnfactor = 0.0;
        if (!model->BSIM4V4wxjGiven)
            model->BSIM4V4wxj = 0.0;
        if (!model->BSIM4V4wvsatGiven)
            model->BSIM4V4wvsat = 0.0;
        if (!model->BSIM4V4watGiven)
            model->BSIM4V4wat = 0.0;
        if (!model->BSIM4V4wa0Given)
            model->BSIM4V4wa0 = 0.0; 
        if (!model->BSIM4V4wagsGiven)
            model->BSIM4V4wags = 0.0;
        if (!model->BSIM4V4wa1Given)
            model->BSIM4V4wa1 = 0.0;
        if (!model->BSIM4V4wa2Given)
            model->BSIM4V4wa2 = 0.0;
        if (!model->BSIM4V4wketaGiven)
            model->BSIM4V4wketa = 0.0;
        if (!model->BSIM4V4wnsubGiven)
            model->BSIM4V4wnsub = 0.0;
        if (!model->BSIM4V4wndepGiven)
            model->BSIM4V4wndep = 0.0;
        if (!model->BSIM4V4wnsdGiven)
            model->BSIM4V4wnsd = 0.0;
        if (!model->BSIM4V4wphinGiven)
            model->BSIM4V4wphin = 0.0;
        if (!model->BSIM4V4wngateGiven)
            model->BSIM4V4wngate = 0.0;
        if (!model->BSIM4V4wvbmGiven)
	    model->BSIM4V4wvbm = 0.0;
        if (!model->BSIM4V4wxtGiven)
	    model->BSIM4V4wxt = 0.0;
        if (!model->BSIM4V4wkt1Given)
            model->BSIM4V4wkt1 = 0.0; 
        if (!model->BSIM4V4wkt1lGiven)
            model->BSIM4V4wkt1l = 0.0;
        if (!model->BSIM4V4wkt2Given)
            model->BSIM4V4wkt2 = 0.0;
        if (!model->BSIM4V4wk3Given)
            model->BSIM4V4wk3 = 0.0;      
        if (!model->BSIM4V4wk3bGiven)
            model->BSIM4V4wk3b = 0.0;      
        if (!model->BSIM4V4ww0Given)
            model->BSIM4V4ww0 = 0.0;    
        if (!model->BSIM4V4wlpe0Given)
            model->BSIM4V4wlpe0 = 0.0;
        if (!model->BSIM4V4wlpebGiven)
            model->BSIM4V4wlpeb = 0.0; 
        if (!model->BSIM4V4wdvtp0Given)
            model->BSIM4V4wdvtp0 = 0.0;
        if (!model->BSIM4V4wdvtp1Given)
            model->BSIM4V4wdvtp1 = 0.0;
        if (!model->BSIM4V4wdvt0Given)
            model->BSIM4V4wdvt0 = 0.0;    
        if (!model->BSIM4V4wdvt1Given)
            model->BSIM4V4wdvt1 = 0.0;      
        if (!model->BSIM4V4wdvt2Given)
            model->BSIM4V4wdvt2 = 0.0;
        if (!model->BSIM4V4wdvt0wGiven)
            model->BSIM4V4wdvt0w = 0.0;    
        if (!model->BSIM4V4wdvt1wGiven)
            model->BSIM4V4wdvt1w = 0.0;      
        if (!model->BSIM4V4wdvt2wGiven)
            model->BSIM4V4wdvt2w = 0.0;
        if (!model->BSIM4V4wdroutGiven)
            model->BSIM4V4wdrout = 0.0;     
        if (!model->BSIM4V4wdsubGiven)
            model->BSIM4V4wdsub = 0.0;
        if (!model->BSIM4V4wvth0Given)
           model->BSIM4V4wvth0 = 0.0;
        if (!model->BSIM4V4wuaGiven)
            model->BSIM4V4wua = 0.0;
        if (!model->BSIM4V4wua1Given)
            model->BSIM4V4wua1 = 0.0;
        if (!model->BSIM4V4wubGiven)
            model->BSIM4V4wub = 0.0;
        if (!model->BSIM4V4wub1Given)
            model->BSIM4V4wub1 = 0.0;
        if (!model->BSIM4V4wucGiven)
            model->BSIM4V4wuc = 0.0;
        if (!model->BSIM4V4wuc1Given)
            model->BSIM4V4wuc1 = 0.0;
        if (!model->BSIM4V4wu0Given)
            model->BSIM4V4wu0 = 0.0;
        if (!model->BSIM4V4wuteGiven)
	    model->BSIM4V4wute = 0.0;    
        if (!model->BSIM4V4wvoffGiven)
	    model->BSIM4V4wvoff = 0.0;
        if (!model->BSIM4V4wminvGiven)
            model->BSIM4V4wminv = 0.0;
        if (!model->BSIM4V4wfproutGiven)
            model->BSIM4V4wfprout = 0.0;
        if (!model->BSIM4V4wpditsGiven)
            model->BSIM4V4wpdits = 0.0;
        if (!model->BSIM4V4wpditsdGiven)
            model->BSIM4V4wpditsd = 0.0;
        if (!model->BSIM4V4wdeltaGiven)  
            model->BSIM4V4wdelta = 0.0;
        if (!model->BSIM4V4wrdswGiven)
            model->BSIM4V4wrdsw = 0.0;
        if (!model->BSIM4V4wrdwGiven)
            model->BSIM4V4wrdw = 0.0;
        if (!model->BSIM4V4wrswGiven)
            model->BSIM4V4wrsw = 0.0;
        if (!model->BSIM4V4wprwbGiven)
            model->BSIM4V4wprwb = 0.0;
        if (!model->BSIM4V4wprwgGiven)
            model->BSIM4V4wprwg = 0.0;
        if (!model->BSIM4V4wprtGiven)
            model->BSIM4V4wprt = 0.0;
        if (!model->BSIM4V4weta0Given)
            model->BSIM4V4weta0 = 0.0;
        if (!model->BSIM4V4wetabGiven)
            model->BSIM4V4wetab = 0.0;
        if (!model->BSIM4V4wpclmGiven)
            model->BSIM4V4wpclm = 0.0; 
        if (!model->BSIM4V4wpdibl1Given)
            model->BSIM4V4wpdibl1 = 0.0;
        if (!model->BSIM4V4wpdibl2Given)
            model->BSIM4V4wpdibl2 = 0.0;
        if (!model->BSIM4V4wpdiblbGiven)
            model->BSIM4V4wpdiblb = 0.0;
        if (!model->BSIM4V4wpscbe1Given)
            model->BSIM4V4wpscbe1 = 0.0;
        if (!model->BSIM4V4wpscbe2Given)
            model->BSIM4V4wpscbe2 = 0.0;
        if (!model->BSIM4V4wpvagGiven)
            model->BSIM4V4wpvag = 0.0;     
        if (!model->BSIM4V4wwrGiven)  
            model->BSIM4V4wwr = 0.0;
        if (!model->BSIM4V4wdwgGiven)  
            model->BSIM4V4wdwg = 0.0;
        if (!model->BSIM4V4wdwbGiven)  
            model->BSIM4V4wdwb = 0.0;
        if (!model->BSIM4V4wb0Given)
            model->BSIM4V4wb0 = 0.0;
        if (!model->BSIM4V4wb1Given)  
            model->BSIM4V4wb1 = 0.0;
        if (!model->BSIM4V4walpha0Given)  
            model->BSIM4V4walpha0 = 0.0;
        if (!model->BSIM4V4walpha1Given)
            model->BSIM4V4walpha1 = 0.0;
        if (!model->BSIM4V4wbeta0Given)  
            model->BSIM4V4wbeta0 = 0.0;
        if (!model->BSIM4V4wagidlGiven)
            model->BSIM4V4wagidl = 0.0;
        if (!model->BSIM4V4wbgidlGiven)
            model->BSIM4V4wbgidl = 0.0;
        if (!model->BSIM4V4wcgidlGiven)
            model->BSIM4V4wcgidl = 0.0;
        if (!model->BSIM4V4wegidlGiven)
            model->BSIM4V4wegidl = 0.0;
        if (!model->BSIM4V4waigcGiven)
            model->BSIM4V4waigc = 0.0;
        if (!model->BSIM4V4wbigcGiven)
            model->BSIM4V4wbigc = 0.0;
        if (!model->BSIM4V4wcigcGiven)
            model->BSIM4V4wcigc = 0.0;
        if (!model->BSIM4V4waigsdGiven)
            model->BSIM4V4waigsd = 0.0;
        if (!model->BSIM4V4wbigsdGiven)
            model->BSIM4V4wbigsd = 0.0;
        if (!model->BSIM4V4wcigsdGiven)
            model->BSIM4V4wcigsd = 0.0;
        if (!model->BSIM4V4waigbaccGiven)
            model->BSIM4V4waigbacc = 0.0;
        if (!model->BSIM4V4wbigbaccGiven)
            model->BSIM4V4wbigbacc = 0.0;
        if (!model->BSIM4V4wcigbaccGiven)
            model->BSIM4V4wcigbacc = 0.0;
        if (!model->BSIM4V4waigbinvGiven)
            model->BSIM4V4waigbinv = 0.0;
        if (!model->BSIM4V4wbigbinvGiven)
            model->BSIM4V4wbigbinv = 0.0;
        if (!model->BSIM4V4wcigbinvGiven)
            model->BSIM4V4wcigbinv = 0.0;
        if (!model->BSIM4V4wnigcGiven)
            model->BSIM4V4wnigc = 0.0;
        if (!model->BSIM4V4wnigbinvGiven)
            model->BSIM4V4wnigbinv = 0.0;
        if (!model->BSIM4V4wnigbaccGiven)
            model->BSIM4V4wnigbacc = 0.0;
        if (!model->BSIM4V4wntoxGiven)
            model->BSIM4V4wntox = 0.0;
        if (!model->BSIM4V4weigbinvGiven)
            model->BSIM4V4weigbinv = 0.0;
        if (!model->BSIM4V4wpigcdGiven)
            model->BSIM4V4wpigcd = 0.0;
        if (!model->BSIM4V4wpoxedgeGiven)
            model->BSIM4V4wpoxedge = 0.0;
        if (!model->BSIM4V4wxrcrg1Given)
            model->BSIM4V4wxrcrg1 = 0.0;
        if (!model->BSIM4V4wxrcrg2Given)
            model->BSIM4V4wxrcrg2 = 0.0;
        if (!model->BSIM4V4weuGiven)
            model->BSIM4V4weu = 0.0;
        if (!model->BSIM4V4wvfbGiven)
            model->BSIM4V4wvfb = 0.0;
        if (!model->BSIM4V4wlambdaGiven)
            model->BSIM4V4wlambda = 0.0;
        if (!model->BSIM4V4wvtlGiven)
            model->BSIM4V4wvtl = 0.0;  
        if (!model->BSIM4V4wxnGiven)
            model->BSIM4V4wxn = 0.0;  
        if (!model->BSIM4V4wvfbsdoffGiven)
            model->BSIM4V4wvfbsdoff = 0.0;   

        if (!model->BSIM4V4wcgslGiven)  
            model->BSIM4V4wcgsl = 0.0;
        if (!model->BSIM4V4wcgdlGiven)  
            model->BSIM4V4wcgdl = 0.0;
        if (!model->BSIM4V4wckappasGiven)  
            model->BSIM4V4wckappas = 0.0;
        if (!model->BSIM4V4wckappadGiven)
            model->BSIM4V4wckappad = 0.0;
        if (!model->BSIM4V4wcfGiven)  
            model->BSIM4V4wcf = 0.0;
        if (!model->BSIM4V4wclcGiven)  
            model->BSIM4V4wclc = 0.0;
        if (!model->BSIM4V4wcleGiven)  
            model->BSIM4V4wcle = 0.0;
        if (!model->BSIM4V4wvfbcvGiven)  
            model->BSIM4V4wvfbcv = 0.0;
        if (!model->BSIM4V4wacdeGiven)
            model->BSIM4V4wacde = 0.0;
        if (!model->BSIM4V4wmoinGiven)
            model->BSIM4V4wmoin = 0.0;
        if (!model->BSIM4V4wnoffGiven)
            model->BSIM4V4wnoff = 0.0;
        if (!model->BSIM4V4wvoffcvGiven)
            model->BSIM4V4wvoffcv = 0.0;

	/* Cross-term dependence */
        if (!model->BSIM4V4pcdscGiven)
	    model->BSIM4V4pcdsc = 0.0;
        if (!model->BSIM4V4pcdscbGiven)
	    model->BSIM4V4pcdscb = 0.0;   
	    if (!model->BSIM4V4pcdscdGiven)
	    model->BSIM4V4pcdscd = 0.0;
        if (!model->BSIM4V4pcitGiven)
	    model->BSIM4V4pcit = 0.0;
        if (!model->BSIM4V4pnfactorGiven)
	    model->BSIM4V4pnfactor = 0.0;
        if (!model->BSIM4V4pxjGiven)
            model->BSIM4V4pxj = 0.0;
        if (!model->BSIM4V4pvsatGiven)
            model->BSIM4V4pvsat = 0.0;
        if (!model->BSIM4V4patGiven)
            model->BSIM4V4pat = 0.0;
        if (!model->BSIM4V4pa0Given)
            model->BSIM4V4pa0 = 0.0; 
            
        if (!model->BSIM4V4pagsGiven)
            model->BSIM4V4pags = 0.0;
        if (!model->BSIM4V4pa1Given)
            model->BSIM4V4pa1 = 0.0;
        if (!model->BSIM4V4pa2Given)
            model->BSIM4V4pa2 = 0.0;
        if (!model->BSIM4V4pketaGiven)
            model->BSIM4V4pketa = 0.0;
        if (!model->BSIM4V4pnsubGiven)
            model->BSIM4V4pnsub = 0.0;
        if (!model->BSIM4V4pndepGiven)
            model->BSIM4V4pndep = 0.0;
        if (!model->BSIM4V4pnsdGiven)
            model->BSIM4V4pnsd = 0.0;
        if (!model->BSIM4V4pphinGiven)
            model->BSIM4V4pphin = 0.0;
        if (!model->BSIM4V4pngateGiven)
            model->BSIM4V4pngate = 0.0;
        if (!model->BSIM4V4pvbmGiven)
	    model->BSIM4V4pvbm = 0.0;
        if (!model->BSIM4V4pxtGiven)
	    model->BSIM4V4pxt = 0.0;
        if (!model->BSIM4V4pkt1Given)
            model->BSIM4V4pkt1 = 0.0; 
        if (!model->BSIM4V4pkt1lGiven)
            model->BSIM4V4pkt1l = 0.0;
        if (!model->BSIM4V4pkt2Given)
            model->BSIM4V4pkt2 = 0.0;
        if (!model->BSIM4V4pk3Given)
            model->BSIM4V4pk3 = 0.0;      
        if (!model->BSIM4V4pk3bGiven)
            model->BSIM4V4pk3b = 0.0;      
        if (!model->BSIM4V4pw0Given)
            model->BSIM4V4pw0 = 0.0;    
        if (!model->BSIM4V4plpe0Given)
            model->BSIM4V4plpe0 = 0.0;
        if (!model->BSIM4V4plpebGiven)
            model->BSIM4V4plpeb = 0.0;
        if (!model->BSIM4V4pdvtp0Given)
            model->BSIM4V4pdvtp0 = 0.0;
        if (!model->BSIM4V4pdvtp1Given)
            model->BSIM4V4pdvtp1 = 0.0;
        if (!model->BSIM4V4pdvt0Given)
            model->BSIM4V4pdvt0 = 0.0;    
        if (!model->BSIM4V4pdvt1Given)
            model->BSIM4V4pdvt1 = 0.0;      
        if (!model->BSIM4V4pdvt2Given)
            model->BSIM4V4pdvt2 = 0.0;
        if (!model->BSIM4V4pdvt0wGiven)
            model->BSIM4V4pdvt0w = 0.0;    
        if (!model->BSIM4V4pdvt1wGiven)
            model->BSIM4V4pdvt1w = 0.0;      
        if (!model->BSIM4V4pdvt2wGiven)
            model->BSIM4V4pdvt2w = 0.0;
        if (!model->BSIM4V4pdroutGiven)
            model->BSIM4V4pdrout = 0.0;     
        if (!model->BSIM4V4pdsubGiven)
            model->BSIM4V4pdsub = 0.0;
        if (!model->BSIM4V4pvth0Given)
           model->BSIM4V4pvth0 = 0.0;
        if (!model->BSIM4V4puaGiven)
            model->BSIM4V4pua = 0.0;
        if (!model->BSIM4V4pua1Given)
            model->BSIM4V4pua1 = 0.0;
        if (!model->BSIM4V4pubGiven)
            model->BSIM4V4pub = 0.0;
        if (!model->BSIM4V4pub1Given)
            model->BSIM4V4pub1 = 0.0;
        if (!model->BSIM4V4pucGiven)
            model->BSIM4V4puc = 0.0;
        if (!model->BSIM4V4puc1Given)
            model->BSIM4V4puc1 = 0.0;
        if (!model->BSIM4V4pu0Given)
            model->BSIM4V4pu0 = 0.0;
        if (!model->BSIM4V4puteGiven)
	    model->BSIM4V4pute = 0.0;    
        if (!model->BSIM4V4pvoffGiven)
	    model->BSIM4V4pvoff = 0.0;
        if (!model->BSIM4V4pminvGiven)
            model->BSIM4V4pminv = 0.0;
        if (!model->BSIM4V4pfproutGiven)
            model->BSIM4V4pfprout = 0.0;
        if (!model->BSIM4V4ppditsGiven)
            model->BSIM4V4ppdits = 0.0;
        if (!model->BSIM4V4ppditsdGiven)
            model->BSIM4V4ppditsd = 0.0;
        if (!model->BSIM4V4pdeltaGiven)  
            model->BSIM4V4pdelta = 0.0;
        if (!model->BSIM4V4prdswGiven)
            model->BSIM4V4prdsw = 0.0;
        if (!model->BSIM4V4prdwGiven)
            model->BSIM4V4prdw = 0.0;
        if (!model->BSIM4V4prswGiven)
            model->BSIM4V4prsw = 0.0;
        if (!model->BSIM4V4pprwbGiven)
            model->BSIM4V4pprwb = 0.0;
        if (!model->BSIM4V4pprwgGiven)
            model->BSIM4V4pprwg = 0.0;
        if (!model->BSIM4V4pprtGiven)
            model->BSIM4V4pprt = 0.0;
        if (!model->BSIM4V4peta0Given)
            model->BSIM4V4peta0 = 0.0;
        if (!model->BSIM4V4petabGiven)
            model->BSIM4V4petab = 0.0;
        if (!model->BSIM4V4ppclmGiven)
            model->BSIM4V4ppclm = 0.0; 
        if (!model->BSIM4V4ppdibl1Given)
            model->BSIM4V4ppdibl1 = 0.0;
        if (!model->BSIM4V4ppdibl2Given)
            model->BSIM4V4ppdibl2 = 0.0;
        if (!model->BSIM4V4ppdiblbGiven)
            model->BSIM4V4ppdiblb = 0.0;
        if (!model->BSIM4V4ppscbe1Given)
            model->BSIM4V4ppscbe1 = 0.0;
        if (!model->BSIM4V4ppscbe2Given)
            model->BSIM4V4ppscbe2 = 0.0;
        if (!model->BSIM4V4ppvagGiven)
            model->BSIM4V4ppvag = 0.0;     
        if (!model->BSIM4V4pwrGiven)  
            model->BSIM4V4pwr = 0.0;
        if (!model->BSIM4V4pdwgGiven)  
            model->BSIM4V4pdwg = 0.0;
        if (!model->BSIM4V4pdwbGiven)  
            model->BSIM4V4pdwb = 0.0;
        if (!model->BSIM4V4pb0Given)
            model->BSIM4V4pb0 = 0.0;
        if (!model->BSIM4V4pb1Given)  
            model->BSIM4V4pb1 = 0.0;
        if (!model->BSIM4V4palpha0Given)  
            model->BSIM4V4palpha0 = 0.0;
        if (!model->BSIM4V4palpha1Given)
            model->BSIM4V4palpha1 = 0.0;
        if (!model->BSIM4V4pbeta0Given)  
            model->BSIM4V4pbeta0 = 0.0;
        if (!model->BSIM4V4pagidlGiven)
            model->BSIM4V4pagidl = 0.0;
        if (!model->BSIM4V4pbgidlGiven)
            model->BSIM4V4pbgidl = 0.0;
        if (!model->BSIM4V4pcgidlGiven)
            model->BSIM4V4pcgidl = 0.0;
        if (!model->BSIM4V4pegidlGiven)
            model->BSIM4V4pegidl = 0.0;
        if (!model->BSIM4V4paigcGiven)
            model->BSIM4V4paigc = 0.0;
        if (!model->BSIM4V4pbigcGiven)
            model->BSIM4V4pbigc = 0.0;
        if (!model->BSIM4V4pcigcGiven)
            model->BSIM4V4pcigc = 0.0;
        if (!model->BSIM4V4paigsdGiven)
            model->BSIM4V4paigsd = 0.0;
        if (!model->BSIM4V4pbigsdGiven)
            model->BSIM4V4pbigsd = 0.0;
        if (!model->BSIM4V4pcigsdGiven)
            model->BSIM4V4pcigsd = 0.0;
        if (!model->BSIM4V4paigbaccGiven)
            model->BSIM4V4paigbacc = 0.0;
        if (!model->BSIM4V4pbigbaccGiven)
            model->BSIM4V4pbigbacc = 0.0;
        if (!model->BSIM4V4pcigbaccGiven)
            model->BSIM4V4pcigbacc = 0.0;
        if (!model->BSIM4V4paigbinvGiven)
            model->BSIM4V4paigbinv = 0.0;
        if (!model->BSIM4V4pbigbinvGiven)
            model->BSIM4V4pbigbinv = 0.0;
        if (!model->BSIM4V4pcigbinvGiven)
            model->BSIM4V4pcigbinv = 0.0;
        if (!model->BSIM4V4pnigcGiven)
            model->BSIM4V4pnigc = 0.0;
        if (!model->BSIM4V4pnigbinvGiven)
            model->BSIM4V4pnigbinv = 0.0;
        if (!model->BSIM4V4pnigbaccGiven)
            model->BSIM4V4pnigbacc = 0.0;
        if (!model->BSIM4V4pntoxGiven)
            model->BSIM4V4pntox = 0.0;
        if (!model->BSIM4V4peigbinvGiven)
            model->BSIM4V4peigbinv = 0.0;
        if (!model->BSIM4V4ppigcdGiven)
            model->BSIM4V4ppigcd = 0.0;
        if (!model->BSIM4V4ppoxedgeGiven)
            model->BSIM4V4ppoxedge = 0.0;
        if (!model->BSIM4V4pxrcrg1Given)
            model->BSIM4V4pxrcrg1 = 0.0;
        if (!model->BSIM4V4pxrcrg2Given)
            model->BSIM4V4pxrcrg2 = 0.0;
        if (!model->BSIM4V4peuGiven)
            model->BSIM4V4peu = 0.0;
        if (!model->BSIM4V4pvfbGiven)
            model->BSIM4V4pvfb = 0.0;
        if (!model->BSIM4V4plambdaGiven)
            model->BSIM4V4plambda = 0.0;
        if (!model->BSIM4V4pvtlGiven)
            model->BSIM4V4pvtl = 0.0;  
        if (!model->BSIM4V4pxnGiven)
            model->BSIM4V4pxn = 0.0;  
        if (!model->BSIM4V4pvfbsdoffGiven)
            model->BSIM4V4pvfbsdoff = 0.0;   

        if (!model->BSIM4V4pcgslGiven)  
            model->BSIM4V4pcgsl = 0.0;
        if (!model->BSIM4V4pcgdlGiven)  
            model->BSIM4V4pcgdl = 0.0;
        if (!model->BSIM4V4pckappasGiven)  
            model->BSIM4V4pckappas = 0.0;
        if (!model->BSIM4V4pckappadGiven)
            model->BSIM4V4pckappad = 0.0;
        if (!model->BSIM4V4pcfGiven)  
            model->BSIM4V4pcf = 0.0;
        if (!model->BSIM4V4pclcGiven)  
            model->BSIM4V4pclc = 0.0;
        if (!model->BSIM4V4pcleGiven)  
            model->BSIM4V4pcle = 0.0;
        if (!model->BSIM4V4pvfbcvGiven)  
            model->BSIM4V4pvfbcv = 0.0;
        if (!model->BSIM4V4pacdeGiven)
            model->BSIM4V4pacde = 0.0;
        if (!model->BSIM4V4pmoinGiven)
            model->BSIM4V4pmoin = 0.0;
        if (!model->BSIM4V4pnoffGiven)
            model->BSIM4V4pnoff = 0.0;
        if (!model->BSIM4V4pvoffcvGiven)
            model->BSIM4V4pvoffcv = 0.0;

        if (!model->BSIM4V4gamma1Given)
            model->BSIM4V4gamma1 = 0.0;
        if (!model->BSIM4V4lgamma1Given)
            model->BSIM4V4lgamma1 = 0.0;
        if (!model->BSIM4V4wgamma1Given)
            model->BSIM4V4wgamma1 = 0.0;
        if (!model->BSIM4V4pgamma1Given)
            model->BSIM4V4pgamma1 = 0.0;
        if (!model->BSIM4V4gamma2Given)
            model->BSIM4V4gamma2 = 0.0;
        if (!model->BSIM4V4lgamma2Given)
            model->BSIM4V4lgamma2 = 0.0;
        if (!model->BSIM4V4wgamma2Given)
            model->BSIM4V4wgamma2 = 0.0;
        if (!model->BSIM4V4pgamma2Given)
            model->BSIM4V4pgamma2 = 0.0;
        if (!model->BSIM4V4vbxGiven)
            model->BSIM4V4vbx = 0.0;
        if (!model->BSIM4V4lvbxGiven)
            model->BSIM4V4lvbx = 0.0;
        if (!model->BSIM4V4wvbxGiven)
            model->BSIM4V4wvbx = 0.0;
        if (!model->BSIM4V4pvbxGiven)
            model->BSIM4V4pvbx = 0.0;

        /* unit degree celcius */
        if (!model->BSIM4V4tnomGiven)  
	    model->BSIM4V4tnom = ckt->CKTnomTemp; 
        if (!model->BSIM4V4LintGiven)  
           model->BSIM4V4Lint = 0.0;
        if (!model->BSIM4V4LlGiven)  
           model->BSIM4V4Ll = 0.0;
        if (!model->BSIM4V4LlcGiven)
           model->BSIM4V4Llc = model->BSIM4V4Ll;
        if (!model->BSIM4V4LlnGiven)  
           model->BSIM4V4Lln = 1.0;
        if (!model->BSIM4V4LwGiven)  
           model->BSIM4V4Lw = 0.0;
        if (!model->BSIM4V4LwcGiven)
           model->BSIM4V4Lwc = model->BSIM4V4Lw;
        if (!model->BSIM4V4LwnGiven)  
           model->BSIM4V4Lwn = 1.0;
        if (!model->BSIM4V4LwlGiven)  
           model->BSIM4V4Lwl = 0.0;
        if (!model->BSIM4V4LwlcGiven)
           model->BSIM4V4Lwlc = model->BSIM4V4Lwl;
        if (!model->BSIM4V4LminGiven)  
           model->BSIM4V4Lmin = 0.0;
        if (!model->BSIM4V4LmaxGiven)  
           model->BSIM4V4Lmax = 1.0;
        if (!model->BSIM4V4WintGiven)  
           model->BSIM4V4Wint = 0.0;
        if (!model->BSIM4V4WlGiven)  
           model->BSIM4V4Wl = 0.0;
        if (!model->BSIM4V4WlcGiven)
           model->BSIM4V4Wlc = model->BSIM4V4Wl;
        if (!model->BSIM4V4WlnGiven)  
           model->BSIM4V4Wln = 1.0;
        if (!model->BSIM4V4WwGiven)  
           model->BSIM4V4Ww = 0.0;
        if (!model->BSIM4V4WwcGiven)
           model->BSIM4V4Wwc = model->BSIM4V4Ww;
        if (!model->BSIM4V4WwnGiven)  
           model->BSIM4V4Wwn = 1.0;
        if (!model->BSIM4V4WwlGiven)  
           model->BSIM4V4Wwl = 0.0;
        if (!model->BSIM4V4WwlcGiven)
           model->BSIM4V4Wwlc = model->BSIM4V4Wwl;
        if (!model->BSIM4V4WminGiven)  
           model->BSIM4V4Wmin = 0.0;
        if (!model->BSIM4V4WmaxGiven)  
           model->BSIM4V4Wmax = 1.0;
        if (!model->BSIM4V4dwcGiven)  
           model->BSIM4V4dwc = model->BSIM4V4Wint;
        if (!model->BSIM4V4dlcGiven)  
           model->BSIM4V4dlc = model->BSIM4V4Lint;
        if (!model->BSIM4V4xlGiven)  
           model->BSIM4V4xl = 0.0;
        if (!model->BSIM4V4xwGiven)  
           model->BSIM4V4xw = 0.0;
        if (!model->BSIM4V4dlcigGiven)
           model->BSIM4V4dlcig = model->BSIM4V4Lint;
        if (!model->BSIM4V4dwjGiven)
           model->BSIM4V4dwj = model->BSIM4V4dwc;
	if (!model->BSIM4V4cfGiven)
           model->BSIM4V4cf = 2.0 * model->BSIM4V4epsrox * EPS0 / PI
		          * log(1.0 + 0.4e-6 / model->BSIM4V4toxe);

        if (!model->BSIM4V4xpartGiven)
            model->BSIM4V4xpart = 0.0;
        if (!model->BSIM4V4sheetResistanceGiven)
            model->BSIM4V4sheetResistance = 0.0;

        if (!model->BSIM4V4SunitAreaJctCapGiven)
            model->BSIM4V4SunitAreaJctCap = 5.0E-4;
        if (!model->BSIM4V4DunitAreaJctCapGiven)
            model->BSIM4V4DunitAreaJctCap = model->BSIM4V4SunitAreaJctCap;
        if (!model->BSIM4V4SunitLengthSidewallJctCapGiven)
            model->BSIM4V4SunitLengthSidewallJctCap = 5.0E-10;
        if (!model->BSIM4V4DunitLengthSidewallJctCapGiven)
            model->BSIM4V4DunitLengthSidewallJctCap = model->BSIM4V4SunitLengthSidewallJctCap;
        if (!model->BSIM4V4SunitLengthGateSidewallJctCapGiven)
            model->BSIM4V4SunitLengthGateSidewallJctCap = model->BSIM4V4SunitLengthSidewallJctCap ;
        if (!model->BSIM4V4DunitLengthGateSidewallJctCapGiven)
            model->BSIM4V4DunitLengthGateSidewallJctCap = model->BSIM4V4SunitLengthGateSidewallJctCap;
        if (!model->BSIM4V4SjctSatCurDensityGiven)
            model->BSIM4V4SjctSatCurDensity = 1.0E-4;
        if (!model->BSIM4V4DjctSatCurDensityGiven)
            model->BSIM4V4DjctSatCurDensity = model->BSIM4V4SjctSatCurDensity;
        if (!model->BSIM4V4SjctSidewallSatCurDensityGiven)
            model->BSIM4V4SjctSidewallSatCurDensity = 0.0;
        if (!model->BSIM4V4DjctSidewallSatCurDensityGiven)
            model->BSIM4V4DjctSidewallSatCurDensity = model->BSIM4V4SjctSidewallSatCurDensity;
        if (!model->BSIM4V4SjctGateSidewallSatCurDensityGiven)
            model->BSIM4V4SjctGateSidewallSatCurDensity = 0.0;
        if (!model->BSIM4V4DjctGateSidewallSatCurDensityGiven)
            model->BSIM4V4DjctGateSidewallSatCurDensity = model->BSIM4V4SjctGateSidewallSatCurDensity;
        if (!model->BSIM4V4SbulkJctPotentialGiven)
            model->BSIM4V4SbulkJctPotential = 1.0;
        if (!model->BSIM4V4DbulkJctPotentialGiven)
            model->BSIM4V4DbulkJctPotential = model->BSIM4V4SbulkJctPotential;
        if (!model->BSIM4V4SsidewallJctPotentialGiven)
            model->BSIM4V4SsidewallJctPotential = 1.0;
        if (!model->BSIM4V4DsidewallJctPotentialGiven)
            model->BSIM4V4DsidewallJctPotential = model->BSIM4V4SsidewallJctPotential;
        if (!model->BSIM4V4SGatesidewallJctPotentialGiven)
            model->BSIM4V4SGatesidewallJctPotential = model->BSIM4V4SsidewallJctPotential;
        if (!model->BSIM4V4DGatesidewallJctPotentialGiven)
            model->BSIM4V4DGatesidewallJctPotential = model->BSIM4V4SGatesidewallJctPotential;
        if (!model->BSIM4V4SbulkJctBotGradingCoeffGiven)
            model->BSIM4V4SbulkJctBotGradingCoeff = 0.5;
        if (!model->BSIM4V4DbulkJctBotGradingCoeffGiven)
            model->BSIM4V4DbulkJctBotGradingCoeff = model->BSIM4V4SbulkJctBotGradingCoeff;
        if (!model->BSIM4V4SbulkJctSideGradingCoeffGiven)
            model->BSIM4V4SbulkJctSideGradingCoeff = 0.33;
        if (!model->BSIM4V4DbulkJctSideGradingCoeffGiven)
            model->BSIM4V4DbulkJctSideGradingCoeff = model->BSIM4V4SbulkJctSideGradingCoeff;
        if (!model->BSIM4V4SbulkJctGateSideGradingCoeffGiven)
            model->BSIM4V4SbulkJctGateSideGradingCoeff = model->BSIM4V4SbulkJctSideGradingCoeff;
        if (!model->BSIM4V4DbulkJctGateSideGradingCoeffGiven)
            model->BSIM4V4DbulkJctGateSideGradingCoeff = model->BSIM4V4SbulkJctGateSideGradingCoeff;
        if (!model->BSIM4V4SjctEmissionCoeffGiven)
            model->BSIM4V4SjctEmissionCoeff = 1.0;
        if (!model->BSIM4V4DjctEmissionCoeffGiven)
            model->BSIM4V4DjctEmissionCoeff = model->BSIM4V4SjctEmissionCoeff;
        if (!model->BSIM4V4SjctTempExponentGiven)
            model->BSIM4V4SjctTempExponent = 3.0;
        if (!model->BSIM4V4DjctTempExponentGiven)
            model->BSIM4V4DjctTempExponent = model->BSIM4V4SjctTempExponent;

        if (!model->BSIM4V4jtssGiven)
            model->BSIM4V4jtss = 0.0;
        if (!model->BSIM4V4jtsdGiven)
            model->BSIM4V4jtsd = model->BSIM4V4jtss;
        if (!model->BSIM4V4jtsswsGiven)
            model->BSIM4V4jtssws = 0.0;
        if (!model->BSIM4V4jtsswdGiven)
            model->BSIM4V4jtsswd = model->BSIM4V4jtssws;
        if (!model->BSIM4V4jtsswgsGiven)
            model->BSIM4V4jtsswgs = 0.0;
        if (!model->BSIM4V4jtsswgdGiven)
            model->BSIM4V4jtsswgd = model->BSIM4V4jtsswgs;
        if (!model->BSIM4V4njtsGiven)
            model->BSIM4V4njts = 20.0;
        if (!model->BSIM4V4njtsswGiven)
            model->BSIM4V4njtssw = 20.0;
        if (!model->BSIM4V4njtsswgGiven)
            model->BSIM4V4njtsswg = 20.0;
        if (!model->BSIM4V4xtssGiven)
            model->BSIM4V4xtss = 0.02;
        if (!model->BSIM4V4xtsdGiven)
            model->BSIM4V4xtsd = model->BSIM4V4xtss;
        if (!model->BSIM4V4xtsswsGiven)
            model->BSIM4V4xtssws = 0.02;
        if (!model->BSIM4V4jtsswdGiven)
            model->BSIM4V4xtsswd = model->BSIM4V4xtssws;
        if (!model->BSIM4V4xtsswgsGiven)
            model->BSIM4V4xtsswgs = 0.02;
        if (!model->BSIM4V4xtsswgdGiven)
            model->BSIM4V4xtsswgd = model->BSIM4V4xtsswgs;
        if (!model->BSIM4V4tnjtsGiven)
            model->BSIM4V4tnjts = 0.0;
        if (!model->BSIM4V4tnjtsswGiven)
            model->BSIM4V4tnjtssw = 0.0;
        if (!model->BSIM4V4tnjtsswgGiven)
            model->BSIM4V4tnjtsswg = 0.0;
        if (!model->BSIM4V4vtssGiven)
            model->BSIM4V4vtss = 10.0;
        if (!model->BSIM4V4vtsdGiven)
            model->BSIM4V4vtsd = model->BSIM4V4vtss;
        if (!model->BSIM4V4vtsswsGiven)
            model->BSIM4V4vtssws = 10.0;
        if (!model->BSIM4V4vtsswdGiven)
            model->BSIM4V4vtsswd = model->BSIM4V4vtssws;
        if (!model->BSIM4V4vtsswgsGiven)
            model->BSIM4V4vtsswgs = 10.0;
        if (!model->BSIM4V4vtsswgdGiven)
            model->BSIM4V4vtsswgd = model->BSIM4V4vtsswgs;

        if (!model->BSIM4V4oxideTrapDensityAGiven)
	{   if (model->BSIM4V4type == NMOS)
                model->BSIM4V4oxideTrapDensityA = 6.25e41;
            else
                model->BSIM4V4oxideTrapDensityA= 6.188e40;
	}
        if (!model->BSIM4V4oxideTrapDensityBGiven)
	{   if (model->BSIM4V4type == NMOS)
                model->BSIM4V4oxideTrapDensityB = 3.125e26;
            else
                model->BSIM4V4oxideTrapDensityB = 1.5e25;
	}
        if (!model->BSIM4V4oxideTrapDensityCGiven)
            model->BSIM4V4oxideTrapDensityC = 8.75e9;
        if (!model->BSIM4V4emGiven)
            model->BSIM4V4em = 4.1e7; /* V/m */
        if (!model->BSIM4V4efGiven)
            model->BSIM4V4ef = 1.0;
        if (!model->BSIM4V4afGiven)
            model->BSIM4V4af = 1.0;
        if (!model->BSIM4V4kfGiven)
            model->BSIM4V4kf = 0.0;
        if (!model->BSIM4V4stimodGiven)
            model->BSIM4V4stimod = 0.0;
        if (!model->BSIM4V4rgeomodGiven)
            model->BSIM4V4rgeomod = 0;
        if (!model->BSIM4V4sa0Given)
            model->BSIM4V4sa0 = 0.0;
        if (!model->BSIM4V4sb0Given)
            model->BSIM4V4sb0 = 0.0;

        /* stress effect */
        if (!model->BSIM4V4sarefGiven)
            model->BSIM4V4saref = 1e-6; /* m */
        if (!model->BSIM4V4sbrefGiven)
            model->BSIM4V4sbref = 1e-6;  /* m */
        if (!model->BSIM4V4wlodGiven)
            model->BSIM4V4wlod = 0;  /* m */
        if (!model->BSIM4V4ku0Given)
            model->BSIM4V4ku0 = 0; /* 1/m */
        if (!model->BSIM4V4kvsatGiven)
            model->BSIM4V4kvsat = 0;
        if (!model->BSIM4V4kvth0Given) /* m */
            model->BSIM4V4kvth0 = 0;
        if (!model->BSIM4V4tku0Given)
            model->BSIM4V4tku0 = 0;
        if (!model->BSIM4V4llodku0Given)
            model->BSIM4V4llodku0 = 0;
        if (!model->BSIM4V4wlodku0Given)
            model->BSIM4V4wlodku0 = 0;
        if (!model->BSIM4V4llodvthGiven)
            model->BSIM4V4llodvth = 0;
        if (!model->BSIM4V4wlodvthGiven)
            model->BSIM4V4wlodvth = 0;
        if (!model->BSIM4V4lku0Given)
            model->BSIM4V4lku0 = 0;
        if (!model->BSIM4V4wku0Given)
            model->BSIM4V4wku0 = 0;
        if (!model->BSIM4V4pku0Given)
            model->BSIM4V4pku0 = 0;
        if (!model->BSIM4V4lkvth0Given)
            model->BSIM4V4lkvth0 = 0;
        if (!model->BSIM4V4wkvth0Given)
            model->BSIM4V4wkvth0 = 0;
        if (!model->BSIM4V4pkvth0Given)
            model->BSIM4V4pkvth0 = 0;
        if (!model->BSIM4V4stk2Given)
            model->BSIM4V4stk2 = 0;
        if (!model->BSIM4V4lodk2Given)
            model->BSIM4V4lodk2 = 1.0;
        if (!model->BSIM4V4steta0Given)
            model->BSIM4V4steta0 = 0;
        if (!model->BSIM4V4lodeta0Given)
            model->BSIM4V4lodeta0 = 1.0;

        DMCGeff = model->BSIM4V4dmcg - model->BSIM4V4dmcgt;
        DMCIeff = model->BSIM4V4dmci;
        DMDGeff = model->BSIM4V4dmdg - model->BSIM4V4dmcgt;

	/*
         * End processing models and begin to loop
         * through all the instances of the model
         */

        for (here = model->BSIM4V4instances; here != NULL ;
             here=here->BSIM4V4nextInstance) 
	{   
	         if (here->BSIM4V4owner == ARCHme) {
		          /* allocate a chunk of the state vector */
              here->BSIM4V4states = *states;
              *states += BSIM4V4numStates;
            }
            /* perform the parameter defaulting */
            if (!here->BSIM4V4lGiven)
                here->BSIM4V4l = 5.0e-6;
            if (!here->BSIM4V4wGiven)
                here->BSIM4V4w = 5.0e-6;
            if (!here->BSIM4V4mGiven)
                here->BSIM4V4m = 1.0;
            if (!here->BSIM4V4nfGiven)
                here->BSIM4V4nf = 1.0;
            if (!here->BSIM4V4minGiven)
                here->BSIM4V4min = 0; /* integer */
            if (!here->BSIM4V4icVDSGiven)
                here->BSIM4V4icVDS = 0.0;
            if (!here->BSIM4V4icVGSGiven)
                here->BSIM4V4icVGS = 0.0;
            if (!here->BSIM4V4icVBSGiven)
                here->BSIM4V4icVBS = 0.0;
            if (!here->BSIM4V4drainAreaGiven)
                here->BSIM4V4drainArea = 0.0;
            if (!here->BSIM4V4drainPerimeterGiven)
                here->BSIM4V4drainPerimeter = 0.0;
            if (!here->BSIM4V4drainSquaresGiven)
                here->BSIM4V4drainSquares = 1.0;
            if (!here->BSIM4V4sourceAreaGiven)
                here->BSIM4V4sourceArea = 0.0;
            if (!here->BSIM4V4sourcePerimeterGiven)
                here->BSIM4V4sourcePerimeter = 0.0;
            if (!here->BSIM4V4sourceSquaresGiven)
                here->BSIM4V4sourceSquares = 1.0;

            if (!here->BSIM4V4saGiven)
                here->BSIM4V4sa = 0.0;
            if (!here->BSIM4V4sbGiven)
                here->BSIM4V4sb = 0.0;
            if (!here->BSIM4V4sdGiven)
                here->BSIM4V4sd = 0.0;

            if (!here->BSIM4V4rbdbGiven)
                here->BSIM4V4rbdb = model->BSIM4V4rbdb; /* in ohm */
            if (!here->BSIM4V4rbsbGiven)
                here->BSIM4V4rbsb = model->BSIM4V4rbsb;
            if (!here->BSIM4V4rbpbGiven)
                here->BSIM4V4rbpb = model->BSIM4V4rbpb;
            if (!here->BSIM4V4rbpsGiven)
                here->BSIM4V4rbps = model->BSIM4V4rbps;
            if (!here->BSIM4V4rbpdGiven)
                here->BSIM4V4rbpd = model->BSIM4V4rbpd;

                    
            /* Process instance model selectors, some
             * may override their global counterparts
	     */
            if (!here->BSIM4V4rbodyModGiven)
                here->BSIM4V4rbodyMod = model->BSIM4V4rbodyMod;
            else if ((here->BSIM4V4rbodyMod != 0) && (here->BSIM4V4rbodyMod != 1))
            {   here->BSIM4V4rbodyMod = model->BSIM4V4rbodyMod;
                printf("Warning: rbodyMod has been set to its global value %d.\n",
	        model->BSIM4V4rbodyMod);
            }

            if (!here->BSIM4V4rgateModGiven)
                here->BSIM4V4rgateMod = model->BSIM4V4rgateMod;
            else if ((here->BSIM4V4rgateMod != 0) && (here->BSIM4V4rgateMod != 1)
	        && (here->BSIM4V4rgateMod != 2) && (here->BSIM4V4rgateMod != 3))
            {   here->BSIM4V4rgateMod = model->BSIM4V4rgateMod;
                printf("Warning: rgateMod has been set to its global value %d.\n",
                model->BSIM4V4rgateMod);
            }

            if (!here->BSIM4V4geoModGiven)
                here->BSIM4V4geoMod = model->BSIM4V4geoMod;
            if (!here->BSIM4V4rgeoModGiven)
                here->BSIM4V4rgeoMod = model->BSIM4V4rgeomod;
            if (!here->BSIM4V4trnqsModGiven)
                here->BSIM4V4trnqsMod = model->BSIM4V4trnqsMod;
            else if ((here->BSIM4V4trnqsMod != 0) && (here->BSIM4V4trnqsMod != 1))
            {   here->BSIM4V4trnqsMod = model->BSIM4V4trnqsMod;
                printf("Warning: trnqsMod has been set to its global value %d.\n",
                model->BSIM4V4trnqsMod);
            }

            if (!here->BSIM4V4acnqsModGiven)
                here->BSIM4V4acnqsMod = model->BSIM4V4acnqsMod;
            else if ((here->BSIM4V4acnqsMod != 0) && (here->BSIM4V4acnqsMod != 1))
            {   here->BSIM4V4acnqsMod = model->BSIM4V4acnqsMod;
                printf("Warning: acnqsMod has been set to its global value %d.\n",
                model->BSIM4V4acnqsMod);
            }

            /* stress effect */
            if (!here->BSIM4V4saGiven)
                here->BSIM4V4sa = 0.0;
            if (!here->BSIM4V4sbGiven)
                here->BSIM4V4sb = 0.0;
            if (!here->BSIM4V4sdGiven)
                here->BSIM4V4sd = 0.0;


            /* process drain series resistance */
            createNode = 0;
            if ( (model->BSIM4V4rdsMod != 0)
                            || (model->BSIM4V4tnoiMod != 0 && noiseAnalGiven))
            {
               createNode = 1;
            } else if (model->BSIM4V4sheetResistance > 0)
            {
                     if (here->BSIM4V4drainSquaresGiven
                                       && here->BSIM4V4drainSquares > 0)
                     {
                          createNode = 1;
                     } else if (!here->BSIM4V4drainSquaresGiven
                                       && (here->BSIM4V4rgeoMod != 0))
                     {
                          BSIM4V4RdseffGeo(here->BSIM4V4nf*here->BSIM4V4m, here->BSIM4V4geoMod,
                                  here->BSIM4V4rgeoMod, here->BSIM4V4min,
                                  here->BSIM4V4w, model->BSIM4V4sheetResistance,
                                  DMCGeff, DMCIeff, DMDGeff, 0, &Rtot);
                          if(Rtot > 0)
                             createNode = 1;
                     }
            }
            if ( createNode != 0  && (here->BSIM4V4dNodePrime == 0))
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4V4name,"drain");
                if(error) return(error);
                here->BSIM4V4dNodePrime = tmp->number;
		
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
            else
            {   here->BSIM4V4dNodePrime = here->BSIM4V4dNode;
            }
            
            /* process source series resistance */
            createNode = 0;
            if ( (model->BSIM4V4rdsMod != 0)
                            || (model->BSIM4V4tnoiMod != 0 && noiseAnalGiven))
            {
               createNode = 1;
            } else if (model->BSIM4V4sheetResistance > 0)
            {
                     if (here->BSIM4V4sourceSquaresGiven
                                        && here->BSIM4V4sourceSquares > 0)
                     {
                          createNode = 1;
                     } else if (!here->BSIM4V4sourceSquaresGiven
                                        && (here->BSIM4V4rgeoMod != 0))
                     {
                          BSIM4V4RdseffGeo(here->BSIM4V4nf*here->BSIM4V4m, here->BSIM4V4geoMod,
                                  here->BSIM4V4rgeoMod, here->BSIM4V4min,
                                  here->BSIM4V4w, model->BSIM4V4sheetResistance,
                                  DMCGeff, DMCIeff, DMDGeff, 1, &Rtot);
                          if(Rtot > 0)
                             createNode = 1;
                     }
            }
            if ( createNode != 0  && here->BSIM4V4sNodePrime == 0)
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4V4name,"source");
                if(error) return(error);
                here->BSIM4V4sNodePrime = tmp->number;
		
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
            else
                here->BSIM4V4sNodePrime = here->BSIM4V4sNode;

            if ((here->BSIM4V4rgateMod > 0) && (here->BSIM4V4gNodePrime == 0))
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4V4name,"gate");
                if(error) return(error);
                   here->BSIM4V4gNodePrime = tmp->number;
		   
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
            else
                here->BSIM4V4gNodePrime = here->BSIM4V4gNodeExt;

            if ((here->BSIM4V4rgateMod == 3) && (here->BSIM4V4gNodeMid == 0))
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4V4name,"midgate");
                if(error) return(error);
                   here->BSIM4V4gNodeMid = tmp->number;
            }
            else
                here->BSIM4V4gNodeMid = here->BSIM4V4gNodeExt;
            

            /* internal body nodes for body resistance model */
            if (here->BSIM4V4rbodyMod)
            {   if (here->BSIM4V4dbNode == 0)
		{   error = CKTmkVolt(ckt,&tmp,here->BSIM4V4name,"dbody");
                    if(error) return(error);
                    here->BSIM4V4dbNode = tmp->number;
		    
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
		if (here->BSIM4V4bNodePrime == 0)
                {   error = CKTmkVolt(ckt,&tmp,here->BSIM4V4name,"body");
                    if(error) return(error);
                    here->BSIM4V4bNodePrime = tmp->number;
                }
		if (here->BSIM4V4sbNode == 0)
                {   error = CKTmkVolt(ckt,&tmp,here->BSIM4V4name,"sbody");
                    if(error) return(error);
                    here->BSIM4V4sbNode = tmp->number;
                }
            }
	    else
	        here->BSIM4V4dbNode = here->BSIM4V4bNodePrime = here->BSIM4V4sbNode
				  = here->BSIM4V4bNode;

            /* NQS node */
            if ((here->BSIM4V4trnqsMod) && (here->BSIM4V4qNode == 0)) 
	    {   error = CKTmkVolt(ckt,&tmp,here->BSIM4V4name,"charge");
                if(error) return(error);
                here->BSIM4V4qNode = tmp->number;
		
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
	    else 
	        here->BSIM4V4qNode = 0;

/* set Sparse Matrix Pointers 
 * macro to make elements with built-in out-of-memory test */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==(double *)NULL){\
    return(E_NOMEM);\
}

            TSTALLOC(BSIM4V4DPbpPtr, BSIM4V4dNodePrime, BSIM4V4bNodePrime)
            TSTALLOC(BSIM4V4GPbpPtr, BSIM4V4gNodePrime, BSIM4V4bNodePrime)
            TSTALLOC(BSIM4V4SPbpPtr, BSIM4V4sNodePrime, BSIM4V4bNodePrime)

            TSTALLOC(BSIM4V4BPdpPtr, BSIM4V4bNodePrime, BSIM4V4dNodePrime)
            TSTALLOC(BSIM4V4BPgpPtr, BSIM4V4bNodePrime, BSIM4V4gNodePrime)
            TSTALLOC(BSIM4V4BPspPtr, BSIM4V4bNodePrime, BSIM4V4sNodePrime)
            TSTALLOC(BSIM4V4BPbpPtr, BSIM4V4bNodePrime, BSIM4V4bNodePrime)

            TSTALLOC(BSIM4V4DdPtr, BSIM4V4dNode, BSIM4V4dNode)
            TSTALLOC(BSIM4V4GPgpPtr, BSIM4V4gNodePrime, BSIM4V4gNodePrime)
            TSTALLOC(BSIM4V4SsPtr, BSIM4V4sNode, BSIM4V4sNode)
            TSTALLOC(BSIM4V4DPdpPtr, BSIM4V4dNodePrime, BSIM4V4dNodePrime)
            TSTALLOC(BSIM4V4SPspPtr, BSIM4V4sNodePrime, BSIM4V4sNodePrime)
            TSTALLOC(BSIM4V4DdpPtr, BSIM4V4dNode, BSIM4V4dNodePrime)
            TSTALLOC(BSIM4V4GPdpPtr, BSIM4V4gNodePrime, BSIM4V4dNodePrime)
            TSTALLOC(BSIM4V4GPspPtr, BSIM4V4gNodePrime, BSIM4V4sNodePrime)
            TSTALLOC(BSIM4V4SspPtr, BSIM4V4sNode, BSIM4V4sNodePrime)
            TSTALLOC(BSIM4V4DPspPtr, BSIM4V4dNodePrime, BSIM4V4sNodePrime)
            TSTALLOC(BSIM4V4DPdPtr, BSIM4V4dNodePrime, BSIM4V4dNode)
            TSTALLOC(BSIM4V4DPgpPtr, BSIM4V4dNodePrime, BSIM4V4gNodePrime)
            TSTALLOC(BSIM4V4SPgpPtr, BSIM4V4sNodePrime, BSIM4V4gNodePrime)
            TSTALLOC(BSIM4V4SPsPtr, BSIM4V4sNodePrime, BSIM4V4sNode)
            TSTALLOC(BSIM4V4SPdpPtr, BSIM4V4sNodePrime, BSIM4V4dNodePrime)

            TSTALLOC(BSIM4V4QqPtr, BSIM4V4qNode, BSIM4V4qNode)
            TSTALLOC(BSIM4V4QbpPtr, BSIM4V4qNode, BSIM4V4bNodePrime) 
            TSTALLOC(BSIM4V4QdpPtr, BSIM4V4qNode, BSIM4V4dNodePrime)
            TSTALLOC(BSIM4V4QspPtr, BSIM4V4qNode, BSIM4V4sNodePrime)
            TSTALLOC(BSIM4V4QgpPtr, BSIM4V4qNode, BSIM4V4gNodePrime)
            TSTALLOC(BSIM4V4DPqPtr, BSIM4V4dNodePrime, BSIM4V4qNode)
            TSTALLOC(BSIM4V4SPqPtr, BSIM4V4sNodePrime, BSIM4V4qNode)
            TSTALLOC(BSIM4V4GPqPtr, BSIM4V4gNodePrime, BSIM4V4qNode)

            if (here->BSIM4V4rgateMod != 0)
            {   TSTALLOC(BSIM4V4GEgePtr, BSIM4V4gNodeExt, BSIM4V4gNodeExt)
                TSTALLOC(BSIM4V4GEgpPtr, BSIM4V4gNodeExt, BSIM4V4gNodePrime)
                TSTALLOC(BSIM4V4GPgePtr, BSIM4V4gNodePrime, BSIM4V4gNodeExt)
		TSTALLOC(BSIM4V4GEdpPtr, BSIM4V4gNodeExt, BSIM4V4dNodePrime)
		TSTALLOC(BSIM4V4GEspPtr, BSIM4V4gNodeExt, BSIM4V4sNodePrime)
		TSTALLOC(BSIM4V4GEbpPtr, BSIM4V4gNodeExt, BSIM4V4bNodePrime)

                TSTALLOC(BSIM4V4GMdpPtr, BSIM4V4gNodeMid, BSIM4V4dNodePrime)
                TSTALLOC(BSIM4V4GMgpPtr, BSIM4V4gNodeMid, BSIM4V4gNodePrime)
                TSTALLOC(BSIM4V4GMgmPtr, BSIM4V4gNodeMid, BSIM4V4gNodeMid)
                TSTALLOC(BSIM4V4GMgePtr, BSIM4V4gNodeMid, BSIM4V4gNodeExt)
                TSTALLOC(BSIM4V4GMspPtr, BSIM4V4gNodeMid, BSIM4V4sNodePrime)
                TSTALLOC(BSIM4V4GMbpPtr, BSIM4V4gNodeMid, BSIM4V4bNodePrime)
                TSTALLOC(BSIM4V4DPgmPtr, BSIM4V4dNodePrime, BSIM4V4gNodeMid)
                TSTALLOC(BSIM4V4GPgmPtr, BSIM4V4gNodePrime, BSIM4V4gNodeMid)
                TSTALLOC(BSIM4V4GEgmPtr, BSIM4V4gNodeExt, BSIM4V4gNodeMid)
                TSTALLOC(BSIM4V4SPgmPtr, BSIM4V4sNodePrime, BSIM4V4gNodeMid)
                TSTALLOC(BSIM4V4BPgmPtr, BSIM4V4bNodePrime, BSIM4V4gNodeMid)
            }	

            if (here->BSIM4V4rbodyMod)
            {   TSTALLOC(BSIM4V4DPdbPtr, BSIM4V4dNodePrime, BSIM4V4dbNode)
                TSTALLOC(BSIM4V4SPsbPtr, BSIM4V4sNodePrime, BSIM4V4sbNode)

                TSTALLOC(BSIM4V4DBdpPtr, BSIM4V4dbNode, BSIM4V4dNodePrime)
                TSTALLOC(BSIM4V4DBdbPtr, BSIM4V4dbNode, BSIM4V4dbNode)
                TSTALLOC(BSIM4V4DBbpPtr, BSIM4V4dbNode, BSIM4V4bNodePrime)
                TSTALLOC(BSIM4V4DBbPtr, BSIM4V4dbNode, BSIM4V4bNode)

                TSTALLOC(BSIM4V4BPdbPtr, BSIM4V4bNodePrime, BSIM4V4dbNode)
                TSTALLOC(BSIM4V4BPbPtr, BSIM4V4bNodePrime, BSIM4V4bNode)
                TSTALLOC(BSIM4V4BPsbPtr, BSIM4V4bNodePrime, BSIM4V4sbNode)

                TSTALLOC(BSIM4V4SBspPtr, BSIM4V4sbNode, BSIM4V4sNodePrime)
                TSTALLOC(BSIM4V4SBbpPtr, BSIM4V4sbNode, BSIM4V4bNodePrime)
                TSTALLOC(BSIM4V4SBbPtr, BSIM4V4sbNode, BSIM4V4bNode)
                TSTALLOC(BSIM4V4SBsbPtr, BSIM4V4sbNode, BSIM4V4sbNode)

                TSTALLOC(BSIM4V4BdbPtr, BSIM4V4bNode, BSIM4V4dbNode)
                TSTALLOC(BSIM4V4BbpPtr, BSIM4V4bNode, BSIM4V4bNodePrime)
                TSTALLOC(BSIM4V4BsbPtr, BSIM4V4bNode, BSIM4V4sbNode)
	        TSTALLOC(BSIM4V4BbPtr, BSIM4V4bNode, BSIM4V4bNode)
	    }

            if (model->BSIM4V4rdsMod)
            {   TSTALLOC(BSIM4V4DgpPtr, BSIM4V4dNode, BSIM4V4gNodePrime)
		TSTALLOC(BSIM4V4DspPtr, BSIM4V4dNode, BSIM4V4sNodePrime)
                TSTALLOC(BSIM4V4DbpPtr, BSIM4V4dNode, BSIM4V4bNodePrime)
                TSTALLOC(BSIM4V4SdpPtr, BSIM4V4sNode, BSIM4V4dNodePrime)
                TSTALLOC(BSIM4V4SgpPtr, BSIM4V4sNode, BSIM4V4gNodePrime)
                TSTALLOC(BSIM4V4SbpPtr, BSIM4V4sNode, BSIM4V4bNodePrime)
            }
        }
    }
    return(OK);
}  

int
BSIM4V4unsetup(inModel,ckt)
    GENmodel *inModel;
    CKTcircuit *ckt;
{

    BSIM4V4model *model;
    BSIM4V4instance *here;

    for (model = (BSIM4V4model *)inModel; model != NULL;
            model = model->BSIM4V4nextModel)
    {
        for (here = model->BSIM4V4instances; here != NULL;
                here=here->BSIM4V4nextInstance)
        {
            if (here->BSIM4V4dNodePrime
                    && here->BSIM4V4dNodePrime != here->BSIM4V4dNode)
            {
                CKTdltNNum(ckt, here->BSIM4V4dNodePrime);
                here->BSIM4V4dNodePrime = 0;
            }
            if (here->BSIM4V4sNodePrime
                    && here->BSIM4V4sNodePrime != here->BSIM4V4sNode)
            {
                CKTdltNNum(ckt, here->BSIM4V4sNodePrime);
                here->BSIM4V4sNodePrime = 0;
            }
        }
    }
    return OK;
}
