/**** BSIM4.6.2 Released by Wenwei Yang 07/31/2008 ****/
/**** BSIM4.6.4 Update ngspice 08/22/2009 ****/
/**** OpenMP support ngspice 06/28/2010 ****/
/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4set.c of BSIM4.6.2.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
 * Authors: 2008- Wenwei Yang, Ali Niknejad, Chenming Hu 
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 * Modified by Xuemei Xi, 03/04/2004.
 * Modified by Xuemei Xi, Mohan Dunga, 07/29/2005.
 * Modified by Mohan Dunga, 12/13/2006
 * Modified by Mohan Dunga, Wenwei Yang, 05/18/2007.
 * Modified by Wenwei Yang, 07/31/2008.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/jobdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "bsim4v6def.h"
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
BSIM4v6setup(
SMPmatrix *matrix,
GENmodel *inModel,
CKTcircuit *ckt,
int *states)
{
BSIM4v6model *model = (BSIM4v6model*)inModel;
BSIM4v6instance *here;
int error;
CKTnode *tmp;
int    noiseAnalGiven = 0, createNode;  /* Criteria for new node creation */
double Rtot, DMCGeff, DMCIeff, DMDGeff;
JOB   *job;


#ifdef USE_OMP
int idx, InstCount;
BSIM4v6instance **InstArray;
#endif

    /* Search for a noise analysis request */
    for (job = ft_curckt->ci_curTask->jobs; job; job = job->JOBnextJob) {
        if(strcmp(job->JOBname,"Noise Analysis")==0) {
            noiseAnalGiven = 1;
            break;
        }
    }

    /*  loop through all the BSIM4v6 device models */
    for( ; model != NULL; model = BSIM4v6nextModel(model))
    {   /* process defaults of model parameters */
        if (!model->BSIM4v6typeGiven)
            model->BSIM4v6type = NMOS;     

        if (!model->BSIM4v6mobModGiven) 
            model->BSIM4v6mobMod = 0;
	else if ((model->BSIM4v6mobMod != 0) && (model->BSIM4v6mobMod != 1)
	         && (model->BSIM4v6mobMod != 2) && (model->BSIM4v6mobMod != 3))
	{   model->BSIM4v6mobMod = 0;
            printf("Warning: mobMod has been set to its default value: 0.\n");
	}

        if (!model->BSIM4v6binUnitGiven) 
            model->BSIM4v6binUnit = 1;
        if (!model->BSIM4v6paramChkGiven) 
            model->BSIM4v6paramChk = 1;

        if (!model->BSIM4v6dioModGiven)
            model->BSIM4v6dioMod = 1;
        else if ((model->BSIM4v6dioMod != 0) && (model->BSIM4v6dioMod != 1)
            && (model->BSIM4v6dioMod != 2))
        {   model->BSIM4v6dioMod = 1;
            printf("Warning: dioMod has been set to its default value: 1.\n");
        }

        if (!model->BSIM4v6cvchargeModGiven) 
            model->BSIM4v6cvchargeMod = 0;
        if (!model->BSIM4v6capModGiven) 
            model->BSIM4v6capMod = 2;
        else if ((model->BSIM4v6capMod != 0) && (model->BSIM4v6capMod != 1)
            && (model->BSIM4v6capMod != 2))
        {   model->BSIM4v6capMod = 2;
            printf("Warning: capMod has been set to its default value: 2.\n");
        }

        if (!model->BSIM4v6rdsModGiven)
            model->BSIM4v6rdsMod = 0;
	else if ((model->BSIM4v6rdsMod != 0) && (model->BSIM4v6rdsMod != 1))
        {   model->BSIM4v6rdsMod = 0;
	    printf("Warning: rdsMod has been set to its default value: 0.\n");
	}
        if (!model->BSIM4v6rbodyModGiven)
            model->BSIM4v6rbodyMod = 0;
        else if ((model->BSIM4v6rbodyMod != 0) && (model->BSIM4v6rbodyMod != 1) && (model->BSIM4v6rbodyMod != 2))
        {   model->BSIM4v6rbodyMod = 0;
            printf("Warning: rbodyMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v6rgateModGiven)
            model->BSIM4v6rgateMod = 0;
        else if ((model->BSIM4v6rgateMod != 0) && (model->BSIM4v6rgateMod != 1)
            && (model->BSIM4v6rgateMod != 2) && (model->BSIM4v6rgateMod != 3))
        {   model->BSIM4v6rgateMod = 0;
            printf("Warning: rgateMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v6perModGiven)
            model->BSIM4v6perMod = 1;
        else if ((model->BSIM4v6perMod != 0) && (model->BSIM4v6perMod != 1))
        {   model->BSIM4v6perMod = 1;
            printf("Warning: perMod has been set to its default value: 1.\n");
        }

        if (!model->BSIM4v6geoModGiven)
            model->BSIM4v6geoMod = 0;

        if (!model->BSIM4v6rgeoModGiven)
            model->BSIM4v6rgeoMod = 0;
        else if ((model->BSIM4v6rgeoMod != 0) && (model->BSIM4v6rgeoMod != 1))
        {   model->BSIM4v6rgeoMod = 1;
            printf("Warning: rgeoMod has been set to its default value: 1.\n");
        }
        if (!model->BSIM4v6fnoiModGiven) 
            model->BSIM4v6fnoiMod = 1;
        else if ((model->BSIM4v6fnoiMod != 0) && (model->BSIM4v6fnoiMod != 1))
        {   model->BSIM4v6fnoiMod = 1;
            printf("Warning: fnoiMod has been set to its default value: 1.\n");
        }
        if (!model->BSIM4v6tnoiModGiven)
            model->BSIM4v6tnoiMod = 0; /* WDLiu: tnoiMod=1 needs to set internal S/D nodes */
        else if ((model->BSIM4v6tnoiMod != 0) && (model->BSIM4v6tnoiMod != 1))
        {   model->BSIM4v6tnoiMod = 0;
            printf("Warning: tnoiMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v6trnqsModGiven)
            model->BSIM4v6trnqsMod = 0; 
        else if ((model->BSIM4v6trnqsMod != 0) && (model->BSIM4v6trnqsMod != 1))
        {   model->BSIM4v6trnqsMod = 0;
            printf("Warning: trnqsMod has been set to its default value: 0.\n");
        }
        if (!model->BSIM4v6acnqsModGiven)
            model->BSIM4v6acnqsMod = 0;
        else if ((model->BSIM4v6acnqsMod != 0) && (model->BSIM4v6acnqsMod != 1))
        {   model->BSIM4v6acnqsMod = 0;
            printf("Warning: acnqsMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v6mtrlModGiven)
            model->BSIM4v6mtrlMod = 0;

        if (!model->BSIM4v6igcModGiven)
            model->BSIM4v6igcMod = 0;
        else if ((model->BSIM4v6igcMod != 0) && (model->BSIM4v6igcMod != 1)
		  && (model->BSIM4v6igcMod != 2))
        {   model->BSIM4v6igcMod = 0;
            printf("Warning: igcMod has been set to its default value: 0.\n");
        }
        if (!model->BSIM4v6igbModGiven)
            model->BSIM4v6igbMod = 0;
        else if ((model->BSIM4v6igbMod != 0) && (model->BSIM4v6igbMod != 1))
        {   model->BSIM4v6igbMod = 0;
            printf("Warning: igbMod has been set to its default value: 0.\n");
        }
        if (!model->BSIM4v6tempModGiven)
            model->BSIM4v6tempMod = 0;
        else if ((model->BSIM4v6tempMod != 0) && (model->BSIM4v6tempMod != 1) 
		  && (model->BSIM4v6tempMod != 2) && (model->BSIM4v6tempMod != 3))
        {   model->BSIM4v6tempMod = 0;
            printf("Warning: tempMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v6versionGiven) 
            model->BSIM4v6version = "4.6.4";
        if (!model->BSIM4v6toxrefGiven)
            model->BSIM4v6toxref = 30.0e-10;
        if (!model->BSIM4v6eotGiven)
            model->BSIM4v6eot = 15.0e-10;
        if (!model->BSIM4v6vddeotGiven)
            model->BSIM4v6vddeot = (model->BSIM4v6type == NMOS) ? 1.5 : -1.5;
        if (!model->BSIM4v6tempeotGiven)
            model->BSIM4v6tempeot = 300.15;
        if (!model->BSIM4v6leffeotGiven)
            model->BSIM4v6leffeot = 1;
        if (!model->BSIM4v6weffeotGiven)
            model->BSIM4v6weffeot = 10;	
        if (!model->BSIM4v6adosGiven)
            model->BSIM4v6ados = 1.0;
        if (!model->BSIM4v6bdosGiven)
            model->BSIM4v6bdos = 1.0;
        if (!model->BSIM4v6toxeGiven)
            model->BSIM4v6toxe = 30.0e-10;
        if (!model->BSIM4v6toxpGiven)
            model->BSIM4v6toxp = model->BSIM4v6toxe;
        if (!model->BSIM4v6toxmGiven)
            model->BSIM4v6toxm = model->BSIM4v6toxe;
        if (!model->BSIM4v6dtoxGiven)
            model->BSIM4v6dtox = 0.0;
        if (!model->BSIM4v6epsroxGiven)
            model->BSIM4v6epsrox = 3.9;

        if (!model->BSIM4v6cdscGiven)
	    model->BSIM4v6cdsc = 2.4e-4;   /* unit Q/V/m^2  */
        if (!model->BSIM4v6cdscbGiven)
	    model->BSIM4v6cdscb = 0.0;   /* unit Q/V/m^2  */    
        if (!model->BSIM4v6cdscdGiven)
	    model->BSIM4v6cdscd = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM4v6citGiven)
	    model->BSIM4v6cit = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM4v6nfactorGiven)
	    model->BSIM4v6nfactor = 1.0;
        if (!model->BSIM4v6xjGiven)
            model->BSIM4v6xj = .15e-6;
        if (!model->BSIM4v6vsatGiven)
            model->BSIM4v6vsat = 8.0e4;    /* unit m/s */ 
        if (!model->BSIM4v6atGiven)
            model->BSIM4v6at = 3.3e4;    /* unit m/s */ 
        if (!model->BSIM4v6a0Given)
            model->BSIM4v6a0 = 1.0;  
        if (!model->BSIM4v6agsGiven)
            model->BSIM4v6ags = 0.0;
        if (!model->BSIM4v6a1Given)
            model->BSIM4v6a1 = 0.0;
        if (!model->BSIM4v6a2Given)
            model->BSIM4v6a2 = 1.0;
        if (!model->BSIM4v6ketaGiven)
            model->BSIM4v6keta = -0.047;    /* unit  / V */
        if (!model->BSIM4v6nsubGiven)
            model->BSIM4v6nsub = 6.0e16;   /* unit 1/cm3 */
        if (!model->BSIM4v6phigGiven)
	    model->BSIM4v6phig = 4.05;  
        if (!model->BSIM4v6epsrgateGiven)
	    model->BSIM4v6epsrgate = 11.7;  
        if (!model->BSIM4v6easubGiven)
            model->BSIM4v6easub = 4.05;  
        if (!model->BSIM4v6epsrsubGiven)
  	    model->BSIM4v6epsrsub = 11.7; 
        if (!model->BSIM4v6ni0subGiven)
            model->BSIM4v6ni0sub = 1.45e10;   /* unit 1/cm3 */
        if (!model->BSIM4v6bg0subGiven)
            model->BSIM4v6bg0sub =  1.16;     /* unit eV */
        if (!model->BSIM4v6tbgasubGiven)
  	    model->BSIM4v6tbgasub = 7.02e-4;  
        if (!model->BSIM4v6tbgbsubGiven)
	    model->BSIM4v6tbgbsub = 1108.0;  
        if (!model->BSIM4v6ndepGiven)
            model->BSIM4v6ndep = 1.7e17;   /* unit 1/cm3 */
        if (!model->BSIM4v6nsdGiven)
            model->BSIM4v6nsd = 1.0e20;   /* unit 1/cm3 */
        if (!model->BSIM4v6phinGiven)
            model->BSIM4v6phin = 0.0; /* unit V */
        if (!model->BSIM4v6ngateGiven)
            model->BSIM4v6ngate = 0;   /* unit 1/cm3 */
        if (!model->BSIM4v6vbmGiven)
	    model->BSIM4v6vbm = -3.0;
        if (!model->BSIM4v6xtGiven)
	    model->BSIM4v6xt = 1.55e-7;
        if (!model->BSIM4v6kt1Given)
            model->BSIM4v6kt1 = -0.11;      /* unit V */
        if (!model->BSIM4v6kt1lGiven)
            model->BSIM4v6kt1l = 0.0;      /* unit V*m */
        if (!model->BSIM4v6kt2Given)
            model->BSIM4v6kt2 = 0.022;      /* No unit */
        if (!model->BSIM4v6k3Given)
            model->BSIM4v6k3 = 80.0;      
        if (!model->BSIM4v6k3bGiven)
            model->BSIM4v6k3b = 0.0;      
        if (!model->BSIM4v6w0Given)
            model->BSIM4v6w0 = 2.5e-6;    
        if (!model->BSIM4v6lpe0Given)
            model->BSIM4v6lpe0 = 1.74e-7;     
        if (!model->BSIM4v6lpebGiven)
            model->BSIM4v6lpeb = 0.0;
        if (!model->BSIM4v6dvtp0Given)
            model->BSIM4v6dvtp0 = 0.0;
        if (!model->BSIM4v6dvtp1Given)
            model->BSIM4v6dvtp1 = 0.0;
        if (!model->BSIM4v6dvt0Given)
            model->BSIM4v6dvt0 = 2.2;    
        if (!model->BSIM4v6dvt1Given)
            model->BSIM4v6dvt1 = 0.53;      
        if (!model->BSIM4v6dvt2Given)
            model->BSIM4v6dvt2 = -0.032;   /* unit 1 / V */     

        if (!model->BSIM4v6dvt0wGiven)
            model->BSIM4v6dvt0w = 0.0;    
        if (!model->BSIM4v6dvt1wGiven)
            model->BSIM4v6dvt1w = 5.3e6;    
        if (!model->BSIM4v6dvt2wGiven)
            model->BSIM4v6dvt2w = -0.032;   

        if (!model->BSIM4v6droutGiven)
            model->BSIM4v6drout = 0.56;     
        if (!model->BSIM4v6dsubGiven)
            model->BSIM4v6dsub = model->BSIM4v6drout;     
        if (!model->BSIM4v6vth0Given)
            model->BSIM4v6vth0 = (model->BSIM4v6type == NMOS) ? 0.7 : -0.7;
	if (!model->BSIM4v6vfbGiven)
	    model->BSIM4v6vfb = -1.0;
        if (!model->BSIM4v6euGiven)
            model->BSIM4v6eu = (model->BSIM4v6type == NMOS) ? 1.67 : 1.0;
		if (!model->BSIM4v6ucsGiven)
            model->BSIM4v6ucs = (model->BSIM4v6type == NMOS) ? 1.67 : 1.0;
        if (!model->BSIM4v6uaGiven)
            model->BSIM4v6ua = ((model->BSIM4v6mobMod == 2)) ? 1.0e-15 : 1.0e-9; /* unit m/V */
        if (!model->BSIM4v6ua1Given)
            model->BSIM4v6ua1 = 1.0e-9;      /* unit m/V */
        if (!model->BSIM4v6ubGiven)
            model->BSIM4v6ub = 1.0e-19;     /* unit (m/V)**2 */
        if (!model->BSIM4v6ub1Given)
            model->BSIM4v6ub1 = -1.0e-18;     /* unit (m/V)**2 */
        if (!model->BSIM4v6ucGiven)
            model->BSIM4v6uc = (model->BSIM4v6mobMod == 1) ? -0.0465 : -0.0465e-9;   
        if (!model->BSIM4v6uc1Given)
            model->BSIM4v6uc1 = (model->BSIM4v6mobMod == 1) ? -0.056 : -0.056e-9;   
        if (!model->BSIM4v6udGiven)
            model->BSIM4v6ud = 0.0;     /* unit m**(-2) */
        if (!model->BSIM4v6ud1Given)
            model->BSIM4v6ud1 = 0.0;     
        if (!model->BSIM4v6upGiven)
            model->BSIM4v6up = 0.0;     
        if (!model->BSIM4v6lpGiven)
            model->BSIM4v6lp = 1.0e-8;     
        if (!model->BSIM4v6u0Given)
            model->BSIM4v6u0 = (model->BSIM4v6type == NMOS) ? 0.067 : 0.025;
        if (!model->BSIM4v6uteGiven)
	        model->BSIM4v6ute = -1.5; 
        if (!model->BSIM4v6ucsteGiven)
		    model->BSIM4v6ucste = -4.775e-3;
        if (!model->BSIM4v6voffGiven)
	    model->BSIM4v6voff = -0.08;
        if (!model->BSIM4v6vofflGiven)
            model->BSIM4v6voffl = 0.0;
        if (!model->BSIM4v6voffcvlGiven)
            model->BSIM4v6voffcvl = 0.0;
        if (!model->BSIM4v6minvGiven)
            model->BSIM4v6minv = 0.0;
        if (!model->BSIM4v6minvcvGiven)
            model->BSIM4v6minvcv = 0.0;
        if (!model->BSIM4v6fproutGiven)
            model->BSIM4v6fprout = 0.0;
        if (!model->BSIM4v6pditsGiven)
            model->BSIM4v6pdits = 0.0;
        if (!model->BSIM4v6pditsdGiven)
            model->BSIM4v6pditsd = 0.0;
        if (!model->BSIM4v6pditslGiven)
            model->BSIM4v6pditsl = 0.0;
        if (!model->BSIM4v6deltaGiven)  
           model->BSIM4v6delta = 0.01;
        if (!model->BSIM4v6rdswminGiven)
            model->BSIM4v6rdswmin = 0.0;
        if (!model->BSIM4v6rdwminGiven)
            model->BSIM4v6rdwmin = 0.0;
        if (!model->BSIM4v6rswminGiven)
            model->BSIM4v6rswmin = 0.0;
        if (!model->BSIM4v6rdswGiven)
	    model->BSIM4v6rdsw = 200.0; /* in ohm*um */     
        if (!model->BSIM4v6rdwGiven)
            model->BSIM4v6rdw = 100.0;
        if (!model->BSIM4v6rswGiven)
            model->BSIM4v6rsw = 100.0;
        if (!model->BSIM4v6prwgGiven)
            model->BSIM4v6prwg = 1.0; /* in 1/V */
        if (!model->BSIM4v6prwbGiven)
            model->BSIM4v6prwb = 0.0;      
        if (!model->BSIM4v6prtGiven)
            model->BSIM4v6prt = 0.0;      
        if (!model->BSIM4v6eta0Given)
            model->BSIM4v6eta0 = 0.08;      /* no unit  */ 
        if (!model->BSIM4v6etabGiven)
            model->BSIM4v6etab = -0.07;      /* unit  1/V */ 
        if (!model->BSIM4v6pclmGiven)
            model->BSIM4v6pclm = 1.3;      /* no unit  */ 
        if (!model->BSIM4v6pdibl1Given)
            model->BSIM4v6pdibl1 = 0.39;    /* no unit  */
        if (!model->BSIM4v6pdibl2Given)
            model->BSIM4v6pdibl2 = 0.0086;    /* no unit  */ 
        if (!model->BSIM4v6pdiblbGiven)
            model->BSIM4v6pdiblb = 0.0;    /* 1/V  */ 
        if (!model->BSIM4v6pscbe1Given)
            model->BSIM4v6pscbe1 = 4.24e8;     
        if (!model->BSIM4v6pscbe2Given)
            model->BSIM4v6pscbe2 = 1.0e-5;    
        if (!model->BSIM4v6pvagGiven)
            model->BSIM4v6pvag = 0.0;     
        if (!model->BSIM4v6wrGiven)  
            model->BSIM4v6wr = 1.0;
        if (!model->BSIM4v6dwgGiven)  
            model->BSIM4v6dwg = 0.0;
        if (!model->BSIM4v6dwbGiven)  
            model->BSIM4v6dwb = 0.0;
        if (!model->BSIM4v6b0Given)
            model->BSIM4v6b0 = 0.0;
        if (!model->BSIM4v6b1Given)  
            model->BSIM4v6b1 = 0.0;
        if (!model->BSIM4v6alpha0Given)  
            model->BSIM4v6alpha0 = 0.0;
        if (!model->BSIM4v6alpha1Given)
            model->BSIM4v6alpha1 = 0.0;
        if (!model->BSIM4v6beta0Given)  
            model->BSIM4v6beta0 = 0.0;
        if (!model->BSIM4v6agidlGiven)
            model->BSIM4v6agidl = 0.0;
        if (!model->BSIM4v6bgidlGiven)
            model->BSIM4v6bgidl = 2.3e9; /* V/m */
        if (!model->BSIM4v6cgidlGiven)
            model->BSIM4v6cgidl = 0.5; /* V^3 */
        if (!model->BSIM4v6egidlGiven)
            model->BSIM4v6egidl = 0.8; /* V */
        if (!model->BSIM4v6agislGiven)
        {
            if (model->BSIM4v6agidlGiven)
                model->BSIM4v6agisl = model->BSIM4v6agidl;
            else
                model->BSIM4v6agisl = 0.0;
        }
        if (!model->BSIM4v6bgislGiven)
        {
            if (model->BSIM4v6bgidlGiven)
                model->BSIM4v6bgisl = model->BSIM4v6bgidl;
            else
                model->BSIM4v6bgisl = 2.3e9; /* V/m */
        }
        if (!model->BSIM4v6cgislGiven)
        {
            if (model->BSIM4v6cgidlGiven)
                model->BSIM4v6cgisl = model->BSIM4v6cgidl;
            else
                model->BSIM4v6cgisl = 0.5; /* V^3 */
        }
        if (!model->BSIM4v6egislGiven)
        {
            if (model->BSIM4v6egidlGiven)
                model->BSIM4v6egisl = model->BSIM4v6egidl;
            else
                model->BSIM4v6egisl = 0.8; /* V */
        }
        if (!model->BSIM4v6aigcGiven)
            model->BSIM4v6aigc = (model->BSIM4v6type == NMOS) ? 1.36e-2 : 9.80e-3;
        if (!model->BSIM4v6bigcGiven)
            model->BSIM4v6bigc = (model->BSIM4v6type == NMOS) ? 1.71e-3 : 7.59e-4;
        if (!model->BSIM4v6cigcGiven)
            model->BSIM4v6cigc = (model->BSIM4v6type == NMOS) ? 0.075 : 0.03;
        if (model->BSIM4v6aigsdGiven)
        {
            model->BSIM4v6aigs = model->BSIM4v6aigd = model->BSIM4v6aigsd;
        }
        else
        {
            model->BSIM4v6aigsd = (model->BSIM4v6type == NMOS) ? 1.36e-2 : 9.80e-3;
            if (!model->BSIM4v6aigsGiven)
                model->BSIM4v6aigs = (model->BSIM4v6type == NMOS) ? 1.36e-2 : 9.80e-3;
            if (!model->BSIM4v6aigdGiven)
                model->BSIM4v6aigd = (model->BSIM4v6type == NMOS) ? 1.36e-2 : 9.80e-3;
        }
        if (model->BSIM4v6bigsdGiven)
        {
            model->BSIM4v6bigs = model->BSIM4v6bigd = model->BSIM4v6bigsd;
        }
        else
        {
            model->BSIM4v6bigsd = (model->BSIM4v6type == NMOS) ? 1.71e-3 : 7.59e-4;
            if (!model->BSIM4v6bigsGiven)
                model->BSIM4v6bigs = (model->BSIM4v6type == NMOS) ? 1.71e-3 : 7.59e-4;
            if (!model->BSIM4v6bigdGiven)
                model->BSIM4v6bigd = (model->BSIM4v6type == NMOS) ? 1.71e-3 : 7.59e-4;
        }
        if (model->BSIM4v6cigsdGiven)
        {
            model->BSIM4v6cigs = model->BSIM4v6cigd = model->BSIM4v6cigsd;
        }
        else
        {
             model->BSIM4v6cigsd = (model->BSIM4v6type == NMOS) ? 0.075 : 0.03;
           if (!model->BSIM4v6cigsGiven)
                model->BSIM4v6cigs = (model->BSIM4v6type == NMOS) ? 0.075 : 0.03;
            if (!model->BSIM4v6cigdGiven)
                model->BSIM4v6cigd = (model->BSIM4v6type == NMOS) ? 0.075 : 0.03;
        }
        if (!model->BSIM4v6aigbaccGiven)
            model->BSIM4v6aigbacc = 1.36e-2;
        if (!model->BSIM4v6bigbaccGiven)
            model->BSIM4v6bigbacc = 1.71e-3;
        if (!model->BSIM4v6cigbaccGiven)
            model->BSIM4v6cigbacc = 0.075;
        if (!model->BSIM4v6aigbinvGiven)
            model->BSIM4v6aigbinv = 1.11e-2;
        if (!model->BSIM4v6bigbinvGiven)
            model->BSIM4v6bigbinv = 9.49e-4;
        if (!model->BSIM4v6cigbinvGiven)
            model->BSIM4v6cigbinv = 0.006;
        if (!model->BSIM4v6nigcGiven)
            model->BSIM4v6nigc = 1.0;
        if (!model->BSIM4v6nigbinvGiven)
            model->BSIM4v6nigbinv = 3.0;
        if (!model->BSIM4v6nigbaccGiven)
            model->BSIM4v6nigbacc = 1.0;
        if (!model->BSIM4v6ntoxGiven)
            model->BSIM4v6ntox = 1.0;
        if (!model->BSIM4v6eigbinvGiven)
            model->BSIM4v6eigbinv = 1.1;
        if (!model->BSIM4v6pigcdGiven)
            model->BSIM4v6pigcd = 1.0;
        if (!model->BSIM4v6poxedgeGiven)
            model->BSIM4v6poxedge = 1.0;
        if (!model->BSIM4v6xrcrg1Given)
            model->BSIM4v6xrcrg1 = 12.0;
        if (!model->BSIM4v6xrcrg2Given)
            model->BSIM4v6xrcrg2 = 1.0;
        if (!model->BSIM4v6ijthsfwdGiven)
            model->BSIM4v6ijthsfwd = 0.1; /* unit A */
        if (!model->BSIM4v6ijthdfwdGiven)
            model->BSIM4v6ijthdfwd = model->BSIM4v6ijthsfwd;
        if (!model->BSIM4v6ijthsrevGiven)
            model->BSIM4v6ijthsrev = 0.1; /* unit A */
        if (!model->BSIM4v6ijthdrevGiven)
            model->BSIM4v6ijthdrev = model->BSIM4v6ijthsrev;
        if (!model->BSIM4v6tnoiaGiven)
            model->BSIM4v6tnoia = 1.5;
        if (!model->BSIM4v6tnoibGiven)
            model->BSIM4v6tnoib = 3.5;
        if (!model->BSIM4v6rnoiaGiven)
            model->BSIM4v6rnoia = 0.577;
        if (!model->BSIM4v6rnoibGiven)
            model->BSIM4v6rnoib = 0.5164;
        if (!model->BSIM4v6ntnoiGiven)
            model->BSIM4v6ntnoi = 1.0;
        if (!model->BSIM4v6lambdaGiven)
            model->BSIM4v6lambda = 0.0;
        if (!model->BSIM4v6vtlGiven)
            model->BSIM4v6vtl = 2.0e5;    /* unit m/s */ 
        if (!model->BSIM4v6xnGiven)
            model->BSIM4v6xn = 3.0;   
        if (!model->BSIM4v6lcGiven)
            model->BSIM4v6lc = 5.0e-9;   
        if (!model->BSIM4v6vfbsdoffGiven)  
            model->BSIM4v6vfbsdoff = 0.0;  /* unit v */  
        if (!model->BSIM4v6tvfbsdoffGiven)
            model->BSIM4v6tvfbsdoff = 0.0;  
        if (!model->BSIM4v6tvoffGiven)
            model->BSIM4v6tvoff = 0.0;  
 
        if (!model->BSIM4v6lintnoiGiven)
            model->BSIM4v6lintnoi = 0.0;  /* unit m */  

        if (!model->BSIM4v6xjbvsGiven)
            model->BSIM4v6xjbvs = 1.0; /* no unit */
        if (!model->BSIM4v6xjbvdGiven)
            model->BSIM4v6xjbvd = model->BSIM4v6xjbvs;
        if (!model->BSIM4v6bvsGiven)
            model->BSIM4v6bvs = 10.0; /* V */
        if (!model->BSIM4v6bvdGiven)
            model->BSIM4v6bvd = model->BSIM4v6bvs;

        if (!model->BSIM4v6gbminGiven)
            model->BSIM4v6gbmin = 1.0e-12; /* in mho */
        if (!model->BSIM4v6rbdbGiven)
            model->BSIM4v6rbdb = 50.0; /* in ohm */
        if (!model->BSIM4v6rbpbGiven)
            model->BSIM4v6rbpb = 50.0;
        if (!model->BSIM4v6rbsbGiven)
            model->BSIM4v6rbsb = 50.0;
        if (!model->BSIM4v6rbpsGiven)
            model->BSIM4v6rbps = 50.0;
        if (!model->BSIM4v6rbpdGiven)
            model->BSIM4v6rbpd = 50.0;

        if (!model->BSIM4v6rbps0Given)
            model->BSIM4v6rbps0 = 50.0;
        if (!model->BSIM4v6rbpslGiven)
            model->BSIM4v6rbpsl = 0.0;
        if (!model->BSIM4v6rbpswGiven)
            model->BSIM4v6rbpsw = 0.0;
        if (!model->BSIM4v6rbpsnfGiven)
            model->BSIM4v6rbpsnf = 0.0;

        if (!model->BSIM4v6rbpd0Given)
            model->BSIM4v6rbpd0 = 50.0;
        if (!model->BSIM4v6rbpdlGiven)
            model->BSIM4v6rbpdl = 0.0;
        if (!model->BSIM4v6rbpdwGiven)
            model->BSIM4v6rbpdw = 0.0;
        if (!model->BSIM4v6rbpdnfGiven)
            model->BSIM4v6rbpdnf = 0.0;

        if (!model->BSIM4v6rbpbx0Given)
            model->BSIM4v6rbpbx0 = 100.0;
        if (!model->BSIM4v6rbpbxlGiven)
            model->BSIM4v6rbpbxl = 0.0;
        if (!model->BSIM4v6rbpbxwGiven)
            model->BSIM4v6rbpbxw = 0.0;
        if (!model->BSIM4v6rbpbxnfGiven)
            model->BSIM4v6rbpbxnf = 0.0;
        if (!model->BSIM4v6rbpby0Given)
            model->BSIM4v6rbpby0 = 100.0;
        if (!model->BSIM4v6rbpbylGiven)
            model->BSIM4v6rbpbyl = 0.0;
        if (!model->BSIM4v6rbpbywGiven)
            model->BSIM4v6rbpbyw = 0.0;
        if (!model->BSIM4v6rbpbynfGiven)
            model->BSIM4v6rbpbynf = 0.0;


        if (!model->BSIM4v6rbsbx0Given)
            model->BSIM4v6rbsbx0 = 100.0;
        if (!model->BSIM4v6rbsby0Given)
            model->BSIM4v6rbsby0 = 100.0;
        if (!model->BSIM4v6rbdbx0Given)
            model->BSIM4v6rbdbx0 = 100.0;
        if (!model->BSIM4v6rbdby0Given)
            model->BSIM4v6rbdby0 = 100.0;


        if (!model->BSIM4v6rbsdbxlGiven)
            model->BSIM4v6rbsdbxl = 0.0;
        if (!model->BSIM4v6rbsdbxwGiven)
            model->BSIM4v6rbsdbxw = 0.0;
        if (!model->BSIM4v6rbsdbxnfGiven)
            model->BSIM4v6rbsdbxnf = 0.0;
        if (!model->BSIM4v6rbsdbylGiven)
            model->BSIM4v6rbsdbyl = 0.0;
        if (!model->BSIM4v6rbsdbywGiven)
            model->BSIM4v6rbsdbyw = 0.0;
        if (!model->BSIM4v6rbsdbynfGiven)
            model->BSIM4v6rbsdbynf = 0.0;

        if (!model->BSIM4v6cgslGiven)  
            model->BSIM4v6cgsl = 0.0;
        if (!model->BSIM4v6cgdlGiven)  
            model->BSIM4v6cgdl = 0.0;
        if (!model->BSIM4v6ckappasGiven)  
            model->BSIM4v6ckappas = 0.6;
        if (!model->BSIM4v6ckappadGiven)
            model->BSIM4v6ckappad = model->BSIM4v6ckappas;
        if (!model->BSIM4v6clcGiven)  
            model->BSIM4v6clc = 0.1e-6;
        if (!model->BSIM4v6cleGiven)  
            model->BSIM4v6cle = 0.6;
        if (!model->BSIM4v6vfbcvGiven)  
            model->BSIM4v6vfbcv = -1.0;
        if (!model->BSIM4v6acdeGiven)
            model->BSIM4v6acde = 1.0;
        if (!model->BSIM4v6moinGiven)
            model->BSIM4v6moin = 15.0;
        if (!model->BSIM4v6noffGiven)
            model->BSIM4v6noff = 1.0;
        if (!model->BSIM4v6voffcvGiven)
            model->BSIM4v6voffcv = 0.0;
        if (!model->BSIM4v6dmcgGiven)
            model->BSIM4v6dmcg = 0.0;
        if (!model->BSIM4v6dmciGiven)
            model->BSIM4v6dmci = model->BSIM4v6dmcg;
        if (!model->BSIM4v6dmdgGiven)
            model->BSIM4v6dmdg = 0.0;
        if (!model->BSIM4v6dmcgtGiven)
            model->BSIM4v6dmcgt = 0.0;
        if (!model->BSIM4v6xgwGiven)
            model->BSIM4v6xgw = 0.0;
        if (!model->BSIM4v6xglGiven)
            model->BSIM4v6xgl = 0.0;
        if (!model->BSIM4v6rshgGiven)
            model->BSIM4v6rshg = 0.1;
        if (!model->BSIM4v6ngconGiven)
            model->BSIM4v6ngcon = 1.0;
        if (!model->BSIM4v6tcjGiven)
            model->BSIM4v6tcj = 0.0;
        if (!model->BSIM4v6tpbGiven)
            model->BSIM4v6tpb = 0.0;
        if (!model->BSIM4v6tcjswGiven)
            model->BSIM4v6tcjsw = 0.0;
        if (!model->BSIM4v6tpbswGiven)
            model->BSIM4v6tpbsw = 0.0;
        if (!model->BSIM4v6tcjswgGiven)
            model->BSIM4v6tcjswg = 0.0;
        if (!model->BSIM4v6tpbswgGiven)
            model->BSIM4v6tpbswg = 0.0;

	/* Length dependence */
        if (!model->BSIM4v6lcdscGiven)
	    model->BSIM4v6lcdsc = 0.0;
        if (!model->BSIM4v6lcdscbGiven)
	    model->BSIM4v6lcdscb = 0.0;
        if (!model->BSIM4v6lcdscdGiven)
	    model->BSIM4v6lcdscd = 0.0;
        if (!model->BSIM4v6lcitGiven)
	    model->BSIM4v6lcit = 0.0;
        if (!model->BSIM4v6lnfactorGiven)
	    model->BSIM4v6lnfactor = 0.0;
        if (!model->BSIM4v6lxjGiven)
            model->BSIM4v6lxj = 0.0;
        if (!model->BSIM4v6lvsatGiven)
            model->BSIM4v6lvsat = 0.0;
        if (!model->BSIM4v6latGiven)
            model->BSIM4v6lat = 0.0;
        if (!model->BSIM4v6la0Given)
            model->BSIM4v6la0 = 0.0; 
        if (!model->BSIM4v6lagsGiven)
            model->BSIM4v6lags = 0.0;
        if (!model->BSIM4v6la1Given)
            model->BSIM4v6la1 = 0.0;
        if (!model->BSIM4v6la2Given)
            model->BSIM4v6la2 = 0.0;
        if (!model->BSIM4v6lketaGiven)
            model->BSIM4v6lketa = 0.0;
        if (!model->BSIM4v6lnsubGiven)
            model->BSIM4v6lnsub = 0.0;
        if (!model->BSIM4v6lndepGiven)
            model->BSIM4v6lndep = 0.0;
        if (!model->BSIM4v6lnsdGiven)
            model->BSIM4v6lnsd = 0.0;
        if (!model->BSIM4v6lphinGiven)
            model->BSIM4v6lphin = 0.0;
        if (!model->BSIM4v6lngateGiven)
            model->BSIM4v6lngate = 0.0;
        if (!model->BSIM4v6lvbmGiven)
	    model->BSIM4v6lvbm = 0.0;
        if (!model->BSIM4v6lxtGiven)
	    model->BSIM4v6lxt = 0.0;
        if (!model->BSIM4v6lkt1Given)
            model->BSIM4v6lkt1 = 0.0; 
        if (!model->BSIM4v6lkt1lGiven)
            model->BSIM4v6lkt1l = 0.0;
        if (!model->BSIM4v6lkt2Given)
            model->BSIM4v6lkt2 = 0.0;
        if (!model->BSIM4v6lk3Given)
            model->BSIM4v6lk3 = 0.0;      
        if (!model->BSIM4v6lk3bGiven)
            model->BSIM4v6lk3b = 0.0;      
        if (!model->BSIM4v6lw0Given)
            model->BSIM4v6lw0 = 0.0;    
        if (!model->BSIM4v6llpe0Given)
            model->BSIM4v6llpe0 = 0.0;
        if (!model->BSIM4v6llpebGiven)
            model->BSIM4v6llpeb = 0.0; 
        if (!model->BSIM4v6ldvtp0Given)
            model->BSIM4v6ldvtp0 = 0.0;
        if (!model->BSIM4v6ldvtp1Given)
            model->BSIM4v6ldvtp1 = 0.0;
        if (!model->BSIM4v6ldvt0Given)
            model->BSIM4v6ldvt0 = 0.0;    
        if (!model->BSIM4v6ldvt1Given)
            model->BSIM4v6ldvt1 = 0.0;      
        if (!model->BSIM4v6ldvt2Given)
            model->BSIM4v6ldvt2 = 0.0;
        if (!model->BSIM4v6ldvt0wGiven)
            model->BSIM4v6ldvt0w = 0.0;    
        if (!model->BSIM4v6ldvt1wGiven)
            model->BSIM4v6ldvt1w = 0.0;      
        if (!model->BSIM4v6ldvt2wGiven)
            model->BSIM4v6ldvt2w = 0.0;
        if (!model->BSIM4v6ldroutGiven)
            model->BSIM4v6ldrout = 0.0;     
        if (!model->BSIM4v6ldsubGiven)
            model->BSIM4v6ldsub = 0.0;
        if (!model->BSIM4v6lvth0Given)
           model->BSIM4v6lvth0 = 0.0;
        if (!model->BSIM4v6luaGiven)
            model->BSIM4v6lua = 0.0;
        if (!model->BSIM4v6lua1Given)
            model->BSIM4v6lua1 = 0.0;
        if (!model->BSIM4v6lubGiven)
            model->BSIM4v6lub = 0.0;
        if (!model->BSIM4v6lub1Given)
            model->BSIM4v6lub1 = 0.0;
        if (!model->BSIM4v6lucGiven)
            model->BSIM4v6luc = 0.0;
        if (!model->BSIM4v6luc1Given)
            model->BSIM4v6luc1 = 0.0;
        if (!model->BSIM4v6ludGiven)
            model->BSIM4v6lud = 0.0;
        if (!model->BSIM4v6lud1Given)
            model->BSIM4v6lud1 = 0.0;
        if (!model->BSIM4v6lupGiven)
            model->BSIM4v6lup = 0.0;
        if (!model->BSIM4v6llpGiven)
            model->BSIM4v6llp = 0.0;
        if (!model->BSIM4v6lu0Given)
            model->BSIM4v6lu0 = 0.0;
        if (!model->BSIM4v6luteGiven)
	    model->BSIM4v6lute = 0.0;  
          if (!model->BSIM4v6lucsteGiven)
	    model->BSIM4v6lucste = 0.0; 		
        if (!model->BSIM4v6lvoffGiven)
	    model->BSIM4v6lvoff = 0.0;
        if (!model->BSIM4v6lminvGiven)
            model->BSIM4v6lminv = 0.0;
        if (!model->BSIM4v6lminvcvGiven)
            model->BSIM4v6lminvcv = 0.0;
        if (!model->BSIM4v6lfproutGiven)
            model->BSIM4v6lfprout = 0.0;
        if (!model->BSIM4v6lpditsGiven)
            model->BSIM4v6lpdits = 0.0;
        if (!model->BSIM4v6lpditsdGiven)
            model->BSIM4v6lpditsd = 0.0;
        if (!model->BSIM4v6ldeltaGiven)  
            model->BSIM4v6ldelta = 0.0;
        if (!model->BSIM4v6lrdswGiven)
            model->BSIM4v6lrdsw = 0.0;
        if (!model->BSIM4v6lrdwGiven)
            model->BSIM4v6lrdw = 0.0;
        if (!model->BSIM4v6lrswGiven)
            model->BSIM4v6lrsw = 0.0;
        if (!model->BSIM4v6lprwbGiven)
            model->BSIM4v6lprwb = 0.0;
        if (!model->BSIM4v6lprwgGiven)
            model->BSIM4v6lprwg = 0.0;
        if (!model->BSIM4v6lprtGiven)
            model->BSIM4v6lprt = 0.0;
        if (!model->BSIM4v6leta0Given)
            model->BSIM4v6leta0 = 0.0;
        if (!model->BSIM4v6letabGiven)
            model->BSIM4v6letab = -0.0;
        if (!model->BSIM4v6lpclmGiven)
            model->BSIM4v6lpclm = 0.0; 
        if (!model->BSIM4v6lpdibl1Given)
            model->BSIM4v6lpdibl1 = 0.0;
        if (!model->BSIM4v6lpdibl2Given)
            model->BSIM4v6lpdibl2 = 0.0;
        if (!model->BSIM4v6lpdiblbGiven)
            model->BSIM4v6lpdiblb = 0.0;
        if (!model->BSIM4v6lpscbe1Given)
            model->BSIM4v6lpscbe1 = 0.0;
        if (!model->BSIM4v6lpscbe2Given)
            model->BSIM4v6lpscbe2 = 0.0;
        if (!model->BSIM4v6lpvagGiven)
            model->BSIM4v6lpvag = 0.0;     
        if (!model->BSIM4v6lwrGiven)  
            model->BSIM4v6lwr = 0.0;
        if (!model->BSIM4v6ldwgGiven)  
            model->BSIM4v6ldwg = 0.0;
        if (!model->BSIM4v6ldwbGiven)  
            model->BSIM4v6ldwb = 0.0;
        if (!model->BSIM4v6lb0Given)
            model->BSIM4v6lb0 = 0.0;
        if (!model->BSIM4v6lb1Given)  
            model->BSIM4v6lb1 = 0.0;
        if (!model->BSIM4v6lalpha0Given)  
            model->BSIM4v6lalpha0 = 0.0;
        if (!model->BSIM4v6lalpha1Given)
            model->BSIM4v6lalpha1 = 0.0;
        if (!model->BSIM4v6lbeta0Given)  
            model->BSIM4v6lbeta0 = 0.0;
        if (!model->BSIM4v6lagidlGiven)
            model->BSIM4v6lagidl = 0.0;
        if (!model->BSIM4v6lbgidlGiven)
            model->BSIM4v6lbgidl = 0.0;
        if (!model->BSIM4v6lcgidlGiven)
            model->BSIM4v6lcgidl = 0.0;
        if (!model->BSIM4v6legidlGiven)
            model->BSIM4v6legidl = 0.0;
        if (!model->BSIM4v6lagislGiven)
        {
            if (model->BSIM4v6lagidlGiven)
                model->BSIM4v6lagisl = model->BSIM4v6lagidl;
            else
                model->BSIM4v6lagisl = 0.0;
        }
        if (!model->BSIM4v6lbgislGiven)
        {
            if (model->BSIM4v6lbgidlGiven)
                model->BSIM4v6lbgisl = model->BSIM4v6lbgidl;
            else
                model->BSIM4v6lbgisl = 0.0;
        }
        if (!model->BSIM4v6lcgislGiven)
        {
            if (model->BSIM4v6lcgidlGiven)
                model->BSIM4v6lcgisl = model->BSIM4v6lcgidl;
            else
                model->BSIM4v6lcgisl = 0.0;
        }
        if (!model->BSIM4v6legislGiven)
        {
            if (model->BSIM4v6legidlGiven)
                model->BSIM4v6legisl = model->BSIM4v6legidl;
            else
                model->BSIM4v6legisl = 0.0; 
        }
        if (!model->BSIM4v6laigcGiven)
            model->BSIM4v6laigc = 0.0;
        if (!model->BSIM4v6lbigcGiven)
            model->BSIM4v6lbigc = 0.0;
        if (!model->BSIM4v6lcigcGiven)
            model->BSIM4v6lcigc = 0.0;
	if (!model->BSIM4v6aigsdGiven && (model->BSIM4v6aigsGiven || model->BSIM4v6aigdGiven))
        {
            if (!model->BSIM4v6laigsGiven)
                model->BSIM4v6laigs = 0.0;
            if (!model->BSIM4v6laigdGiven)
                model->BSIM4v6laigd = 0.0;
        }
        else
	{
	   if (!model->BSIM4v6laigsdGiven)
	       model->BSIM4v6laigsd = 0.0;
	   model->BSIM4v6laigs = model->BSIM4v6laigd = model->BSIM4v6laigsd;
        }
	if (!model->BSIM4v6bigsdGiven && (model->BSIM4v6bigsGiven || model->BSIM4v6bigdGiven))
        {
            if (!model->BSIM4v6lbigsGiven)
                model->BSIM4v6lbigs = 0.0;
            if (!model->BSIM4v6lbigdGiven)
                model->BSIM4v6lbigd = 0.0;
        }
        else
	{
	   if (!model->BSIM4v6lbigsdGiven)
	       model->BSIM4v6lbigsd = 0.0;
	   model->BSIM4v6lbigs = model->BSIM4v6lbigd = model->BSIM4v6lbigsd;
        }
	if (!model->BSIM4v6cigsdGiven && (model->BSIM4v6cigsGiven || model->BSIM4v6cigdGiven))
        {
            if (!model->BSIM4v6lcigsGiven)
                model->BSIM4v6lcigs = 0.0;
            if (!model->BSIM4v6lcigdGiven)
                model->BSIM4v6lcigd = 0.0;
        }
        else
	{
	   if (!model->BSIM4v6lcigsdGiven)
	       model->BSIM4v6lcigsd = 0.0;
	   model->BSIM4v6lcigs = model->BSIM4v6lcigd = model->BSIM4v6lcigsd;
        }
        if (!model->BSIM4v6laigbaccGiven)
            model->BSIM4v6laigbacc = 0.0;
        if (!model->BSIM4v6lbigbaccGiven)
            model->BSIM4v6lbigbacc = 0.0;
        if (!model->BSIM4v6lcigbaccGiven)
            model->BSIM4v6lcigbacc = 0.0;
        if (!model->BSIM4v6laigbinvGiven)
            model->BSIM4v6laigbinv = 0.0;
        if (!model->BSIM4v6lbigbinvGiven)
            model->BSIM4v6lbigbinv = 0.0;
        if (!model->BSIM4v6lcigbinvGiven)
            model->BSIM4v6lcigbinv = 0.0;
        if (!model->BSIM4v6lnigcGiven)
            model->BSIM4v6lnigc = 0.0;
        if (!model->BSIM4v6lnigbinvGiven)
            model->BSIM4v6lnigbinv = 0.0;
        if (!model->BSIM4v6lnigbaccGiven)
            model->BSIM4v6lnigbacc = 0.0;
        if (!model->BSIM4v6lntoxGiven)
            model->BSIM4v6lntox = 0.0;
        if (!model->BSIM4v6leigbinvGiven)
            model->BSIM4v6leigbinv = 0.0;
        if (!model->BSIM4v6lpigcdGiven)
            model->BSIM4v6lpigcd = 0.0;
        if (!model->BSIM4v6lpoxedgeGiven)
            model->BSIM4v6lpoxedge = 0.0;
        if (!model->BSIM4v6lxrcrg1Given)
            model->BSIM4v6lxrcrg1 = 0.0;
        if (!model->BSIM4v6lxrcrg2Given)
            model->BSIM4v6lxrcrg2 = 0.0;
        if (!model->BSIM4v6leuGiven)
            model->BSIM4v6leu = 0.0;
		if (!model->BSIM4v6lucsGiven)
            model->BSIM4v6lucs = 0.0;
        if (!model->BSIM4v6lvfbGiven)
            model->BSIM4v6lvfb = 0.0;
        if (!model->BSIM4v6llambdaGiven)
            model->BSIM4v6llambda = 0.0;
        if (!model->BSIM4v6lvtlGiven)
            model->BSIM4v6lvtl = 0.0;  
        if (!model->BSIM4v6lxnGiven)
            model->BSIM4v6lxn = 0.0;  
        if (!model->BSIM4v6lvfbsdoffGiven)
            model->BSIM4v6lvfbsdoff = 0.0;   
        if (!model->BSIM4v6ltvfbsdoffGiven)
            model->BSIM4v6ltvfbsdoff = 0.0;  
        if (!model->BSIM4v6ltvoffGiven)
            model->BSIM4v6ltvoff = 0.0;  


        if (!model->BSIM4v6lcgslGiven)  
            model->BSIM4v6lcgsl = 0.0;
        if (!model->BSIM4v6lcgdlGiven)  
            model->BSIM4v6lcgdl = 0.0;
        if (!model->BSIM4v6lckappasGiven)  
            model->BSIM4v6lckappas = 0.0;
        if (!model->BSIM4v6lckappadGiven)
            model->BSIM4v6lckappad = 0.0;
        if (!model->BSIM4v6lclcGiven)  
            model->BSIM4v6lclc = 0.0;
        if (!model->BSIM4v6lcleGiven)  
            model->BSIM4v6lcle = 0.0;
        if (!model->BSIM4v6lcfGiven)  
            model->BSIM4v6lcf = 0.0;
        if (!model->BSIM4v6lvfbcvGiven)  
            model->BSIM4v6lvfbcv = 0.0;
        if (!model->BSIM4v6lacdeGiven)
            model->BSIM4v6lacde = 0.0;
        if (!model->BSIM4v6lmoinGiven)
            model->BSIM4v6lmoin = 0.0;
        if (!model->BSIM4v6lnoffGiven)
            model->BSIM4v6lnoff = 0.0;
        if (!model->BSIM4v6lvoffcvGiven)
            model->BSIM4v6lvoffcv = 0.0;

	/* Width dependence */
        if (!model->BSIM4v6wcdscGiven)
	    model->BSIM4v6wcdsc = 0.0;
        if (!model->BSIM4v6wcdscbGiven)
	    model->BSIM4v6wcdscb = 0.0;  
        if (!model->BSIM4v6wcdscdGiven)
	    model->BSIM4v6wcdscd = 0.0;
        if (!model->BSIM4v6wcitGiven)
	    model->BSIM4v6wcit = 0.0;
        if (!model->BSIM4v6wnfactorGiven)
	    model->BSIM4v6wnfactor = 0.0;
        if (!model->BSIM4v6wxjGiven)
            model->BSIM4v6wxj = 0.0;
        if (!model->BSIM4v6wvsatGiven)
            model->BSIM4v6wvsat = 0.0;
        if (!model->BSIM4v6watGiven)
            model->BSIM4v6wat = 0.0;
        if (!model->BSIM4v6wa0Given)
            model->BSIM4v6wa0 = 0.0; 
        if (!model->BSIM4v6wagsGiven)
            model->BSIM4v6wags = 0.0;
        if (!model->BSIM4v6wa1Given)
            model->BSIM4v6wa1 = 0.0;
        if (!model->BSIM4v6wa2Given)
            model->BSIM4v6wa2 = 0.0;
        if (!model->BSIM4v6wketaGiven)
            model->BSIM4v6wketa = 0.0;
        if (!model->BSIM4v6wnsubGiven)
            model->BSIM4v6wnsub = 0.0;
        if (!model->BSIM4v6wndepGiven)
            model->BSIM4v6wndep = 0.0;
        if (!model->BSIM4v6wnsdGiven)
            model->BSIM4v6wnsd = 0.0;
        if (!model->BSIM4v6wphinGiven)
            model->BSIM4v6wphin = 0.0;
        if (!model->BSIM4v6wngateGiven)
            model->BSIM4v6wngate = 0.0;
        if (!model->BSIM4v6wvbmGiven)
	    model->BSIM4v6wvbm = 0.0;
        if (!model->BSIM4v6wxtGiven)
	    model->BSIM4v6wxt = 0.0;
        if (!model->BSIM4v6wkt1Given)
            model->BSIM4v6wkt1 = 0.0; 
        if (!model->BSIM4v6wkt1lGiven)
            model->BSIM4v6wkt1l = 0.0;
        if (!model->BSIM4v6wkt2Given)
            model->BSIM4v6wkt2 = 0.0;
        if (!model->BSIM4v6wk3Given)
            model->BSIM4v6wk3 = 0.0;      
        if (!model->BSIM4v6wk3bGiven)
            model->BSIM4v6wk3b = 0.0;      
        if (!model->BSIM4v6ww0Given)
            model->BSIM4v6ww0 = 0.0;    
        if (!model->BSIM4v6wlpe0Given)
            model->BSIM4v6wlpe0 = 0.0;
        if (!model->BSIM4v6wlpebGiven)
            model->BSIM4v6wlpeb = 0.0; 
        if (!model->BSIM4v6wdvtp0Given)
            model->BSIM4v6wdvtp0 = 0.0;
        if (!model->BSIM4v6wdvtp1Given)
            model->BSIM4v6wdvtp1 = 0.0;
        if (!model->BSIM4v6wdvt0Given)
            model->BSIM4v6wdvt0 = 0.0;    
        if (!model->BSIM4v6wdvt1Given)
            model->BSIM4v6wdvt1 = 0.0;      
        if (!model->BSIM4v6wdvt2Given)
            model->BSIM4v6wdvt2 = 0.0;
        if (!model->BSIM4v6wdvt0wGiven)
            model->BSIM4v6wdvt0w = 0.0;    
        if (!model->BSIM4v6wdvt1wGiven)
            model->BSIM4v6wdvt1w = 0.0;      
        if (!model->BSIM4v6wdvt2wGiven)
            model->BSIM4v6wdvt2w = 0.0;
        if (!model->BSIM4v6wdroutGiven)
            model->BSIM4v6wdrout = 0.0;     
        if (!model->BSIM4v6wdsubGiven)
            model->BSIM4v6wdsub = 0.0;
        if (!model->BSIM4v6wvth0Given)
           model->BSIM4v6wvth0 = 0.0;
        if (!model->BSIM4v6wuaGiven)
            model->BSIM4v6wua = 0.0;
        if (!model->BSIM4v6wua1Given)
            model->BSIM4v6wua1 = 0.0;
        if (!model->BSIM4v6wubGiven)
            model->BSIM4v6wub = 0.0;
        if (!model->BSIM4v6wub1Given)
            model->BSIM4v6wub1 = 0.0;
        if (!model->BSIM4v6wucGiven)
            model->BSIM4v6wuc = 0.0;
        if (!model->BSIM4v6wuc1Given)
            model->BSIM4v6wuc1 = 0.0;
        if (!model->BSIM4v6wudGiven)
            model->BSIM4v6wud = 0.0;
        if (!model->BSIM4v6wud1Given)
            model->BSIM4v6wud1 = 0.0;
        if (!model->BSIM4v6wupGiven)
            model->BSIM4v6wup = 0.0;
        if (!model->BSIM4v6wlpGiven)
            model->BSIM4v6wlp = 0.0;
        if (!model->BSIM4v6wu0Given)
            model->BSIM4v6wu0 = 0.0;
        if (!model->BSIM4v6wuteGiven)
	        model->BSIM4v6wute = 0.0; 
        if (!model->BSIM4v6wucsteGiven)
	        model->BSIM4v6wucste = 0.0; 		
        if (!model->BSIM4v6wvoffGiven)
	        model->BSIM4v6wvoff = 0.0;
        if (!model->BSIM4v6wminvGiven)
            model->BSIM4v6wminv = 0.0;
        if (!model->BSIM4v6wminvcvGiven)
            model->BSIM4v6wminvcv = 0.0;
        if (!model->BSIM4v6wfproutGiven)
            model->BSIM4v6wfprout = 0.0;
        if (!model->BSIM4v6wpditsGiven)
            model->BSIM4v6wpdits = 0.0;
        if (!model->BSIM4v6wpditsdGiven)
            model->BSIM4v6wpditsd = 0.0;
        if (!model->BSIM4v6wdeltaGiven)  
            model->BSIM4v6wdelta = 0.0;
        if (!model->BSIM4v6wrdswGiven)
            model->BSIM4v6wrdsw = 0.0;
        if (!model->BSIM4v6wrdwGiven)
            model->BSIM4v6wrdw = 0.0;
        if (!model->BSIM4v6wrswGiven)
            model->BSIM4v6wrsw = 0.0;
        if (!model->BSIM4v6wprwbGiven)
            model->BSIM4v6wprwb = 0.0;
        if (!model->BSIM4v6wprwgGiven)
            model->BSIM4v6wprwg = 0.0;
        if (!model->BSIM4v6wprtGiven)
            model->BSIM4v6wprt = 0.0;
        if (!model->BSIM4v6weta0Given)
            model->BSIM4v6weta0 = 0.0;
        if (!model->BSIM4v6wetabGiven)
            model->BSIM4v6wetab = 0.0;
        if (!model->BSIM4v6wpclmGiven)
            model->BSIM4v6wpclm = 0.0; 
        if (!model->BSIM4v6wpdibl1Given)
            model->BSIM4v6wpdibl1 = 0.0;
        if (!model->BSIM4v6wpdibl2Given)
            model->BSIM4v6wpdibl2 = 0.0;
        if (!model->BSIM4v6wpdiblbGiven)
            model->BSIM4v6wpdiblb = 0.0;
        if (!model->BSIM4v6wpscbe1Given)
            model->BSIM4v6wpscbe1 = 0.0;
        if (!model->BSIM4v6wpscbe2Given)
            model->BSIM4v6wpscbe2 = 0.0;
        if (!model->BSIM4v6wpvagGiven)
            model->BSIM4v6wpvag = 0.0;     
        if (!model->BSIM4v6wwrGiven)  
            model->BSIM4v6wwr = 0.0;
        if (!model->BSIM4v6wdwgGiven)  
            model->BSIM4v6wdwg = 0.0;
        if (!model->BSIM4v6wdwbGiven)  
            model->BSIM4v6wdwb = 0.0;
        if (!model->BSIM4v6wb0Given)
            model->BSIM4v6wb0 = 0.0;
        if (!model->BSIM4v6wb1Given)  
            model->BSIM4v6wb1 = 0.0;
        if (!model->BSIM4v6walpha0Given)  
            model->BSIM4v6walpha0 = 0.0;
        if (!model->BSIM4v6walpha1Given)
            model->BSIM4v6walpha1 = 0.0;
        if (!model->BSIM4v6wbeta0Given)  
            model->BSIM4v6wbeta0 = 0.0;
        if (!model->BSIM4v6wagidlGiven)
            model->BSIM4v6wagidl = 0.0;
        if (!model->BSIM4v6wbgidlGiven)
            model->BSIM4v6wbgidl = 0.0;
        if (!model->BSIM4v6wcgidlGiven)
            model->BSIM4v6wcgidl = 0.0;
        if (!model->BSIM4v6wegidlGiven)
            model->BSIM4v6wegidl = 0.0;
        if (!model->BSIM4v6wagislGiven)
        {
            if (model->BSIM4v6wagidlGiven)
                model->BSIM4v6wagisl = model->BSIM4v6wagidl;
            else
                model->BSIM4v6wagisl = 0.0;
        }
        if (!model->BSIM4v6wbgislGiven)
        {
            if (model->BSIM4v6wbgidlGiven)
                model->BSIM4v6wbgisl = model->BSIM4v6wbgidl;
            else
                model->BSIM4v6wbgisl = 0.0;
        }
        if (!model->BSIM4v6wcgislGiven)
        {
            if (model->BSIM4v6wcgidlGiven)
                model->BSIM4v6wcgisl = model->BSIM4v6wcgidl;
            else
                model->BSIM4v6wcgisl = 0.0;
        }
        if (!model->BSIM4v6wegislGiven)
        {
            if (model->BSIM4v6wegidlGiven)
                model->BSIM4v6wegisl = model->BSIM4v6wegidl;
            else
                model->BSIM4v6wegisl = 0.0; 
        }
        if (!model->BSIM4v6waigcGiven)
            model->BSIM4v6waigc = 0.0;
        if (!model->BSIM4v6wbigcGiven)
            model->BSIM4v6wbigc = 0.0;
        if (!model->BSIM4v6wcigcGiven)
            model->BSIM4v6wcigc = 0.0;
	if (!model->BSIM4v6aigsdGiven && (model->BSIM4v6aigsGiven || model->BSIM4v6aigdGiven))
        {
            if (!model->BSIM4v6waigsGiven)
                model->BSIM4v6waigs = 0.0;
            if (!model->BSIM4v6waigdGiven)
                model->BSIM4v6waigd = 0.0;
        }
        else
	{
	   if (!model->BSIM4v6waigsdGiven)
	       model->BSIM4v6waigsd = 0.0;
	   model->BSIM4v6waigs = model->BSIM4v6waigd = model->BSIM4v6waigsd;
        }
	if (!model->BSIM4v6bigsdGiven && (model->BSIM4v6bigsGiven || model->BSIM4v6bigdGiven))
        {
            if (!model->BSIM4v6wbigsGiven)
                model->BSIM4v6wbigs = 0.0;
            if (!model->BSIM4v6wbigdGiven)
                model->BSIM4v6wbigd = 0.0;
        }
        else
	{
	   if (!model->BSIM4v6wbigsdGiven)
	       model->BSIM4v6wbigsd = 0.0;
	   model->BSIM4v6wbigs = model->BSIM4v6wbigd = model->BSIM4v6wbigsd;
        }
	if (!model->BSIM4v6cigsdGiven && (model->BSIM4v6cigsGiven || model->BSIM4v6cigdGiven))
        {
            if (!model->BSIM4v6wcigsGiven)
                model->BSIM4v6wcigs = 0.0;
            if (!model->BSIM4v6wcigdGiven)
                model->BSIM4v6wcigd = 0.0;
        }
        else
	{
	   if (!model->BSIM4v6wcigsdGiven)
	       model->BSIM4v6wcigsd = 0.0;
	   model->BSIM4v6wcigs = model->BSIM4v6wcigd = model->BSIM4v6wcigsd;
        }
        if (!model->BSIM4v6waigbaccGiven)
            model->BSIM4v6waigbacc = 0.0;
        if (!model->BSIM4v6wbigbaccGiven)
            model->BSIM4v6wbigbacc = 0.0;
        if (!model->BSIM4v6wcigbaccGiven)
            model->BSIM4v6wcigbacc = 0.0;
        if (!model->BSIM4v6waigbinvGiven)
            model->BSIM4v6waigbinv = 0.0;
        if (!model->BSIM4v6wbigbinvGiven)
            model->BSIM4v6wbigbinv = 0.0;
        if (!model->BSIM4v6wcigbinvGiven)
            model->BSIM4v6wcigbinv = 0.0;
        if (!model->BSIM4v6wnigcGiven)
            model->BSIM4v6wnigc = 0.0;
        if (!model->BSIM4v6wnigbinvGiven)
            model->BSIM4v6wnigbinv = 0.0;
        if (!model->BSIM4v6wnigbaccGiven)
            model->BSIM4v6wnigbacc = 0.0;
        if (!model->BSIM4v6wntoxGiven)
            model->BSIM4v6wntox = 0.0;
        if (!model->BSIM4v6weigbinvGiven)
            model->BSIM4v6weigbinv = 0.0;
        if (!model->BSIM4v6wpigcdGiven)
            model->BSIM4v6wpigcd = 0.0;
        if (!model->BSIM4v6wpoxedgeGiven)
            model->BSIM4v6wpoxedge = 0.0;
        if (!model->BSIM4v6wxrcrg1Given)
            model->BSIM4v6wxrcrg1 = 0.0;
        if (!model->BSIM4v6wxrcrg2Given)
            model->BSIM4v6wxrcrg2 = 0.0;
        if (!model->BSIM4v6weuGiven)
            model->BSIM4v6weu = 0.0;
        if (!model->BSIM4v6wucsGiven)
            model->BSIM4v6wucs = 0.0;
        if (!model->BSIM4v6wvfbGiven)
            model->BSIM4v6wvfb = 0.0;
        if (!model->BSIM4v6wlambdaGiven)
            model->BSIM4v6wlambda = 0.0;
        if (!model->BSIM4v6wvtlGiven)
            model->BSIM4v6wvtl = 0.0;  
        if (!model->BSIM4v6wxnGiven)
            model->BSIM4v6wxn = 0.0;  
        if (!model->BSIM4v6wvfbsdoffGiven)
            model->BSIM4v6wvfbsdoff = 0.0;   
        if (!model->BSIM4v6wtvfbsdoffGiven)
            model->BSIM4v6wtvfbsdoff = 0.0;  
        if (!model->BSIM4v6wtvoffGiven)
            model->BSIM4v6wtvoff = 0.0;  

        if (!model->BSIM4v6wcgslGiven)  
            model->BSIM4v6wcgsl = 0.0;
        if (!model->BSIM4v6wcgdlGiven)  
            model->BSIM4v6wcgdl = 0.0;
        if (!model->BSIM4v6wckappasGiven)  
            model->BSIM4v6wckappas = 0.0;
        if (!model->BSIM4v6wckappadGiven)
            model->BSIM4v6wckappad = 0.0;
        if (!model->BSIM4v6wcfGiven)  
            model->BSIM4v6wcf = 0.0;
        if (!model->BSIM4v6wclcGiven)  
            model->BSIM4v6wclc = 0.0;
        if (!model->BSIM4v6wcleGiven)  
            model->BSIM4v6wcle = 0.0;
        if (!model->BSIM4v6wvfbcvGiven)  
            model->BSIM4v6wvfbcv = 0.0;
        if (!model->BSIM4v6wacdeGiven)
            model->BSIM4v6wacde = 0.0;
        if (!model->BSIM4v6wmoinGiven)
            model->BSIM4v6wmoin = 0.0;
        if (!model->BSIM4v6wnoffGiven)
            model->BSIM4v6wnoff = 0.0;
        if (!model->BSIM4v6wvoffcvGiven)
            model->BSIM4v6wvoffcv = 0.0;

	/* Cross-term dependence */
        if (!model->BSIM4v6pcdscGiven)
	    model->BSIM4v6pcdsc = 0.0;
        if (!model->BSIM4v6pcdscbGiven)
	    model->BSIM4v6pcdscb = 0.0;   
        if (!model->BSIM4v6pcdscdGiven)
	    model->BSIM4v6pcdscd = 0.0;
        if (!model->BSIM4v6pcitGiven)
	    model->BSIM4v6pcit = 0.0;
        if (!model->BSIM4v6pnfactorGiven)
	    model->BSIM4v6pnfactor = 0.0;
        if (!model->BSIM4v6pxjGiven)
            model->BSIM4v6pxj = 0.0;
        if (!model->BSIM4v6pvsatGiven)
            model->BSIM4v6pvsat = 0.0;
        if (!model->BSIM4v6patGiven)
            model->BSIM4v6pat = 0.0;
        if (!model->BSIM4v6pa0Given)
            model->BSIM4v6pa0 = 0.0; 
            
        if (!model->BSIM4v6pagsGiven)
            model->BSIM4v6pags = 0.0;
        if (!model->BSIM4v6pa1Given)
            model->BSIM4v6pa1 = 0.0;
        if (!model->BSIM4v6pa2Given)
            model->BSIM4v6pa2 = 0.0;
        if (!model->BSIM4v6pketaGiven)
            model->BSIM4v6pketa = 0.0;
        if (!model->BSIM4v6pnsubGiven)
            model->BSIM4v6pnsub = 0.0;
        if (!model->BSIM4v6pndepGiven)
            model->BSIM4v6pndep = 0.0;
        if (!model->BSIM4v6pnsdGiven)
            model->BSIM4v6pnsd = 0.0;
        if (!model->BSIM4v6pphinGiven)
            model->BSIM4v6pphin = 0.0;
        if (!model->BSIM4v6pngateGiven)
            model->BSIM4v6pngate = 0.0;
        if (!model->BSIM4v6pvbmGiven)
	    model->BSIM4v6pvbm = 0.0;
        if (!model->BSIM4v6pxtGiven)
	    model->BSIM4v6pxt = 0.0;
        if (!model->BSIM4v6pkt1Given)
            model->BSIM4v6pkt1 = 0.0; 
        if (!model->BSIM4v6pkt1lGiven)
            model->BSIM4v6pkt1l = 0.0;
        if (!model->BSIM4v6pkt2Given)
            model->BSIM4v6pkt2 = 0.0;
        if (!model->BSIM4v6pk3Given)
            model->BSIM4v6pk3 = 0.0;      
        if (!model->BSIM4v6pk3bGiven)
            model->BSIM4v6pk3b = 0.0;      
        if (!model->BSIM4v6pw0Given)
            model->BSIM4v6pw0 = 0.0;    
        if (!model->BSIM4v6plpe0Given)
            model->BSIM4v6plpe0 = 0.0;
        if (!model->BSIM4v6plpebGiven)
            model->BSIM4v6plpeb = 0.0;
        if (!model->BSIM4v6pdvtp0Given)
            model->BSIM4v6pdvtp0 = 0.0;
        if (!model->BSIM4v6pdvtp1Given)
            model->BSIM4v6pdvtp1 = 0.0;
        if (!model->BSIM4v6pdvt0Given)
            model->BSIM4v6pdvt0 = 0.0;    
        if (!model->BSIM4v6pdvt1Given)
            model->BSIM4v6pdvt1 = 0.0;      
        if (!model->BSIM4v6pdvt2Given)
            model->BSIM4v6pdvt2 = 0.0;
        if (!model->BSIM4v6pdvt0wGiven)
            model->BSIM4v6pdvt0w = 0.0;    
        if (!model->BSIM4v6pdvt1wGiven)
            model->BSIM4v6pdvt1w = 0.0;      
        if (!model->BSIM4v6pdvt2wGiven)
            model->BSIM4v6pdvt2w = 0.0;
        if (!model->BSIM4v6pdroutGiven)
            model->BSIM4v6pdrout = 0.0;     
        if (!model->BSIM4v6pdsubGiven)
            model->BSIM4v6pdsub = 0.0;
        if (!model->BSIM4v6pvth0Given)
           model->BSIM4v6pvth0 = 0.0;
        if (!model->BSIM4v6puaGiven)
            model->BSIM4v6pua = 0.0;
        if (!model->BSIM4v6pua1Given)
            model->BSIM4v6pua1 = 0.0;
        if (!model->BSIM4v6pubGiven)
            model->BSIM4v6pub = 0.0;
        if (!model->BSIM4v6pub1Given)
            model->BSIM4v6pub1 = 0.0;
        if (!model->BSIM4v6pucGiven)
            model->BSIM4v6puc = 0.0;
        if (!model->BSIM4v6puc1Given)
            model->BSIM4v6puc1 = 0.0;
        if (!model->BSIM4v6pudGiven)
            model->BSIM4v6pud = 0.0;
        if (!model->BSIM4v6pud1Given)
            model->BSIM4v6pud1 = 0.0;
        if (!model->BSIM4v6pupGiven)
            model->BSIM4v6pup = 0.0;
        if (!model->BSIM4v6plpGiven)
            model->BSIM4v6plp = 0.0;
        if (!model->BSIM4v6pu0Given)
            model->BSIM4v6pu0 = 0.0;
        if (!model->BSIM4v6puteGiven)
	    model->BSIM4v6pute = 0.0;  
     if (!model->BSIM4v6pucsteGiven)
	    model->BSIM4v6pucste = 0.0; 		
        if (!model->BSIM4v6pvoffGiven)
	    model->BSIM4v6pvoff = 0.0;
        if (!model->BSIM4v6pminvGiven)
            model->BSIM4v6pminv = 0.0;
        if (!model->BSIM4v6pminvcvGiven)
            model->BSIM4v6pminvcv = 0.0;
        if (!model->BSIM4v6pfproutGiven)
            model->BSIM4v6pfprout = 0.0;
        if (!model->BSIM4v6ppditsGiven)
            model->BSIM4v6ppdits = 0.0;
        if (!model->BSIM4v6ppditsdGiven)
            model->BSIM4v6ppditsd = 0.0;
        if (!model->BSIM4v6pdeltaGiven)  
            model->BSIM4v6pdelta = 0.0;
        if (!model->BSIM4v6prdswGiven)
            model->BSIM4v6prdsw = 0.0;
        if (!model->BSIM4v6prdwGiven)
            model->BSIM4v6prdw = 0.0;
        if (!model->BSIM4v6prswGiven)
            model->BSIM4v6prsw = 0.0;
        if (!model->BSIM4v6pprwbGiven)
            model->BSIM4v6pprwb = 0.0;
        if (!model->BSIM4v6pprwgGiven)
            model->BSIM4v6pprwg = 0.0;
        if (!model->BSIM4v6pprtGiven)
            model->BSIM4v6pprt = 0.0;
        if (!model->BSIM4v6peta0Given)
            model->BSIM4v6peta0 = 0.0;
        if (!model->BSIM4v6petabGiven)
            model->BSIM4v6petab = 0.0;
        if (!model->BSIM4v6ppclmGiven)
            model->BSIM4v6ppclm = 0.0; 
        if (!model->BSIM4v6ppdibl1Given)
            model->BSIM4v6ppdibl1 = 0.0;
        if (!model->BSIM4v6ppdibl2Given)
            model->BSIM4v6ppdibl2 = 0.0;
        if (!model->BSIM4v6ppdiblbGiven)
            model->BSIM4v6ppdiblb = 0.0;
        if (!model->BSIM4v6ppscbe1Given)
            model->BSIM4v6ppscbe1 = 0.0;
        if (!model->BSIM4v6ppscbe2Given)
            model->BSIM4v6ppscbe2 = 0.0;
        if (!model->BSIM4v6ppvagGiven)
            model->BSIM4v6ppvag = 0.0;     
        if (!model->BSIM4v6pwrGiven)  
            model->BSIM4v6pwr = 0.0;
        if (!model->BSIM4v6pdwgGiven)  
            model->BSIM4v6pdwg = 0.0;
        if (!model->BSIM4v6pdwbGiven)  
            model->BSIM4v6pdwb = 0.0;
        if (!model->BSIM4v6pb0Given)
            model->BSIM4v6pb0 = 0.0;
        if (!model->BSIM4v6pb1Given)  
            model->BSIM4v6pb1 = 0.0;
        if (!model->BSIM4v6palpha0Given)  
            model->BSIM4v6palpha0 = 0.0;
        if (!model->BSIM4v6palpha1Given)
            model->BSIM4v6palpha1 = 0.0;
        if (!model->BSIM4v6pbeta0Given)  
            model->BSIM4v6pbeta0 = 0.0;
        if (!model->BSIM4v6pagidlGiven)
            model->BSIM4v6pagidl = 0.0;
        if (!model->BSIM4v6pbgidlGiven)
            model->BSIM4v6pbgidl = 0.0;
        if (!model->BSIM4v6pcgidlGiven)
            model->BSIM4v6pcgidl = 0.0;
        if (!model->BSIM4v6pegidlGiven)
            model->BSIM4v6pegidl = 0.0;
        if (!model->BSIM4v6pagislGiven)
        {
            if (model->BSIM4v6pagidlGiven)
                model->BSIM4v6pagisl = model->BSIM4v6pagidl;
            else
                model->BSIM4v6pagisl = 0.0;
        }
        if (!model->BSIM4v6pbgislGiven)
        {
            if (model->BSIM4v6pbgidlGiven)
                model->BSIM4v6pbgisl = model->BSIM4v6pbgidl;
            else
                model->BSIM4v6pbgisl = 0.0;
        }
        if (!model->BSIM4v6pcgislGiven)
        {
            if (model->BSIM4v6pcgidlGiven)
                model->BSIM4v6pcgisl = model->BSIM4v6pcgidl;
            else
                model->BSIM4v6pcgisl = 0.0;
        }
        if (!model->BSIM4v6pegislGiven)
        {
            if (model->BSIM4v6pegidlGiven)
                model->BSIM4v6pegisl = model->BSIM4v6pegidl;
            else
                model->BSIM4v6pegisl = 0.0; 
        }
        if (!model->BSIM4v6paigcGiven)
            model->BSIM4v6paigc = 0.0;
        if (!model->BSIM4v6pbigcGiven)
            model->BSIM4v6pbigc = 0.0;
        if (!model->BSIM4v6pcigcGiven)
            model->BSIM4v6pcigc = 0.0;
	if (!model->BSIM4v6aigsdGiven && (model->BSIM4v6aigsGiven || model->BSIM4v6aigdGiven))
        {
            if (!model->BSIM4v6paigsGiven)
                model->BSIM4v6paigs = 0.0;
            if (!model->BSIM4v6paigdGiven)
                model->BSIM4v6paigd = 0.0;
        }
        else
	{
	   if (!model->BSIM4v6paigsdGiven)
	       model->BSIM4v6paigsd = 0.0;
	   model->BSIM4v6paigs = model->BSIM4v6paigd = model->BSIM4v6paigsd;
        }
	if (!model->BSIM4v6bigsdGiven && (model->BSIM4v6bigsGiven || model->BSIM4v6bigdGiven))
        {
            if (!model->BSIM4v6pbigsGiven)
                model->BSIM4v6pbigs = 0.0;
            if (!model->BSIM4v6pbigdGiven)
                model->BSIM4v6pbigd = 0.0;
        }
        else
	{
	   if (!model->BSIM4v6pbigsdGiven)
	       model->BSIM4v6pbigsd = 0.0;
	   model->BSIM4v6pbigs = model->BSIM4v6pbigd = model->BSIM4v6pbigsd;
        }
	if (!model->BSIM4v6cigsdGiven && (model->BSIM4v6cigsGiven || model->BSIM4v6cigdGiven))
        {
            if (!model->BSIM4v6pcigsGiven)
                model->BSIM4v6pcigs = 0.0;
            if (!model->BSIM4v6pcigdGiven)
                model->BSIM4v6pcigd = 0.0;
        }
        else
	{
	   if (!model->BSIM4v6pcigsdGiven)
	       model->BSIM4v6pcigsd = 0.0;
	   model->BSIM4v6pcigs = model->BSIM4v6pcigd = model->BSIM4v6pcigsd;
        }
        if (!model->BSIM4v6paigbaccGiven)
            model->BSIM4v6paigbacc = 0.0;
        if (!model->BSIM4v6pbigbaccGiven)
            model->BSIM4v6pbigbacc = 0.0;
        if (!model->BSIM4v6pcigbaccGiven)
            model->BSIM4v6pcigbacc = 0.0;
        if (!model->BSIM4v6paigbinvGiven)
            model->BSIM4v6paigbinv = 0.0;
        if (!model->BSIM4v6pbigbinvGiven)
            model->BSIM4v6pbigbinv = 0.0;
        if (!model->BSIM4v6pcigbinvGiven)
            model->BSIM4v6pcigbinv = 0.0;
        if (!model->BSIM4v6pnigcGiven)
            model->BSIM4v6pnigc = 0.0;
        if (!model->BSIM4v6pnigbinvGiven)
            model->BSIM4v6pnigbinv = 0.0;
        if (!model->BSIM4v6pnigbaccGiven)
            model->BSIM4v6pnigbacc = 0.0;
        if (!model->BSIM4v6pntoxGiven)
            model->BSIM4v6pntox = 0.0;
        if (!model->BSIM4v6peigbinvGiven)
            model->BSIM4v6peigbinv = 0.0;
        if (!model->BSIM4v6ppigcdGiven)
            model->BSIM4v6ppigcd = 0.0;
        if (!model->BSIM4v6ppoxedgeGiven)
            model->BSIM4v6ppoxedge = 0.0;
        if (!model->BSIM4v6pxrcrg1Given)
            model->BSIM4v6pxrcrg1 = 0.0;
        if (!model->BSIM4v6pxrcrg2Given)
            model->BSIM4v6pxrcrg2 = 0.0;
        if (!model->BSIM4v6peuGiven)
            model->BSIM4v6peu = 0.0;
		if (!model->BSIM4v6pucsGiven)
            model->BSIM4v6pucs = 0.0;
        if (!model->BSIM4v6pvfbGiven)
            model->BSIM4v6pvfb = 0.0;
        if (!model->BSIM4v6plambdaGiven)
            model->BSIM4v6plambda = 0.0;
        if (!model->BSIM4v6pvtlGiven)
            model->BSIM4v6pvtl = 0.0;  
        if (!model->BSIM4v6pxnGiven)
            model->BSIM4v6pxn = 0.0;  
        if (!model->BSIM4v6pvfbsdoffGiven)
            model->BSIM4v6pvfbsdoff = 0.0;   
        if (!model->BSIM4v6ptvfbsdoffGiven)
            model->BSIM4v6ptvfbsdoff = 0.0;  
        if (!model->BSIM4v6ptvoffGiven)
            model->BSIM4v6ptvoff = 0.0;  

        if (!model->BSIM4v6pcgslGiven)  
            model->BSIM4v6pcgsl = 0.0;
        if (!model->BSIM4v6pcgdlGiven)  
            model->BSIM4v6pcgdl = 0.0;
        if (!model->BSIM4v6pckappasGiven)  
            model->BSIM4v6pckappas = 0.0;
        if (!model->BSIM4v6pckappadGiven)
            model->BSIM4v6pckappad = 0.0;
        if (!model->BSIM4v6pcfGiven)  
            model->BSIM4v6pcf = 0.0;
        if (!model->BSIM4v6pclcGiven)  
            model->BSIM4v6pclc = 0.0;
        if (!model->BSIM4v6pcleGiven)  
            model->BSIM4v6pcle = 0.0;
        if (!model->BSIM4v6pvfbcvGiven)  
            model->BSIM4v6pvfbcv = 0.0;
        if (!model->BSIM4v6pacdeGiven)
            model->BSIM4v6pacde = 0.0;
        if (!model->BSIM4v6pmoinGiven)
            model->BSIM4v6pmoin = 0.0;
        if (!model->BSIM4v6pnoffGiven)
            model->BSIM4v6pnoff = 0.0;
        if (!model->BSIM4v6pvoffcvGiven)
            model->BSIM4v6pvoffcv = 0.0;

        if (!model->BSIM4v6gamma1Given)
            model->BSIM4v6gamma1 = 0.0;
        if (!model->BSIM4v6lgamma1Given)
            model->BSIM4v6lgamma1 = 0.0;
        if (!model->BSIM4v6wgamma1Given)
            model->BSIM4v6wgamma1 = 0.0;
        if (!model->BSIM4v6pgamma1Given)
            model->BSIM4v6pgamma1 = 0.0;
        if (!model->BSIM4v6gamma2Given)
            model->BSIM4v6gamma2 = 0.0;
        if (!model->BSIM4v6lgamma2Given)
            model->BSIM4v6lgamma2 = 0.0;
        if (!model->BSIM4v6wgamma2Given)
            model->BSIM4v6wgamma2 = 0.0;
        if (!model->BSIM4v6pgamma2Given)
            model->BSIM4v6pgamma2 = 0.0;
        if (!model->BSIM4v6vbxGiven)
            model->BSIM4v6vbx = 0.0;
        if (!model->BSIM4v6lvbxGiven)
            model->BSIM4v6lvbx = 0.0;
        if (!model->BSIM4v6wvbxGiven)
            model->BSIM4v6wvbx = 0.0;
        if (!model->BSIM4v6pvbxGiven)
            model->BSIM4v6pvbx = 0.0;

        /* unit degree celcius */
        if (!model->BSIM4v6tnomGiven)  
	    model->BSIM4v6tnom = ckt->CKTnomTemp; 
        if (!model->BSIM4v6LintGiven)  
           model->BSIM4v6Lint = 0.0;
        if (!model->BSIM4v6LlGiven)  
           model->BSIM4v6Ll = 0.0;
        if (!model->BSIM4v6LlcGiven)
           model->BSIM4v6Llc = model->BSIM4v6Ll;
        if (!model->BSIM4v6LlnGiven)  
           model->BSIM4v6Lln = 1.0;
        if (!model->BSIM4v6LwGiven)  
           model->BSIM4v6Lw = 0.0;
        if (!model->BSIM4v6LwcGiven)
           model->BSIM4v6Lwc = model->BSIM4v6Lw;
        if (!model->BSIM4v6LwnGiven)  
           model->BSIM4v6Lwn = 1.0;
        if (!model->BSIM4v6LwlGiven)  
           model->BSIM4v6Lwl = 0.0;
        if (!model->BSIM4v6LwlcGiven)
           model->BSIM4v6Lwlc = model->BSIM4v6Lwl;
        if (!model->BSIM4v6LminGiven)  
           model->BSIM4v6Lmin = 0.0;
        if (!model->BSIM4v6LmaxGiven)  
           model->BSIM4v6Lmax = 1.0;
        if (!model->BSIM4v6WintGiven)  
           model->BSIM4v6Wint = 0.0;
        if (!model->BSIM4v6WlGiven)  
           model->BSIM4v6Wl = 0.0;
        if (!model->BSIM4v6WlcGiven)
           model->BSIM4v6Wlc = model->BSIM4v6Wl;
        if (!model->BSIM4v6WlnGiven)  
           model->BSIM4v6Wln = 1.0;
        if (!model->BSIM4v6WwGiven)  
           model->BSIM4v6Ww = 0.0;
        if (!model->BSIM4v6WwcGiven)
           model->BSIM4v6Wwc = model->BSIM4v6Ww;
        if (!model->BSIM4v6WwnGiven)  
           model->BSIM4v6Wwn = 1.0;
        if (!model->BSIM4v6WwlGiven)  
           model->BSIM4v6Wwl = 0.0;
        if (!model->BSIM4v6WwlcGiven)
           model->BSIM4v6Wwlc = model->BSIM4v6Wwl;
        if (!model->BSIM4v6WminGiven)  
           model->BSIM4v6Wmin = 0.0;
        if (!model->BSIM4v6WmaxGiven)  
           model->BSIM4v6Wmax = 1.0;
        if (!model->BSIM4v6dwcGiven)  
           model->BSIM4v6dwc = model->BSIM4v6Wint;
        if (!model->BSIM4v6dlcGiven)  
           model->BSIM4v6dlc = model->BSIM4v6Lint;
        if (!model->BSIM4v6xlGiven)  
           model->BSIM4v6xl = 0.0;
        if (!model->BSIM4v6xwGiven)  
           model->BSIM4v6xw = 0.0;
        if (!model->BSIM4v6dlcigGiven)
           model->BSIM4v6dlcig = model->BSIM4v6Lint;
        if (!model->BSIM4v6dlcigdGiven)
        {
           if (model->BSIM4v6dlcigGiven) 
               model->BSIM4v6dlcigd = model->BSIM4v6dlcig;
	   else	     
               model->BSIM4v6dlcigd = model->BSIM4v6Lint;
        }
        if (!model->BSIM4v6dwjGiven)
           model->BSIM4v6dwj = model->BSIM4v6dwc;
	if (!model->BSIM4v6cfGiven)
           model->BSIM4v6cf = 2.0 * model->BSIM4v6epsrox * EPS0 / PI
		          * log(1.0 + 0.4e-6 / model->BSIM4v6toxe);

        if (!model->BSIM4v6xpartGiven)
            model->BSIM4v6xpart = 0.0;
        if (!model->BSIM4v6sheetResistanceGiven)
            model->BSIM4v6sheetResistance = 0.0;

        if (!model->BSIM4v6SunitAreaJctCapGiven)
            model->BSIM4v6SunitAreaJctCap = 5.0E-4;
        if (!model->BSIM4v6DunitAreaJctCapGiven)
            model->BSIM4v6DunitAreaJctCap = model->BSIM4v6SunitAreaJctCap;
        if (!model->BSIM4v6SunitLengthSidewallJctCapGiven)
            model->BSIM4v6SunitLengthSidewallJctCap = 5.0E-10;
        if (!model->BSIM4v6DunitLengthSidewallJctCapGiven)
            model->BSIM4v6DunitLengthSidewallJctCap = model->BSIM4v6SunitLengthSidewallJctCap;
        if (!model->BSIM4v6SunitLengthGateSidewallJctCapGiven)
            model->BSIM4v6SunitLengthGateSidewallJctCap = model->BSIM4v6SunitLengthSidewallJctCap ;
        if (!model->BSIM4v6DunitLengthGateSidewallJctCapGiven)
            model->BSIM4v6DunitLengthGateSidewallJctCap = model->BSIM4v6SunitLengthGateSidewallJctCap;
        if (!model->BSIM4v6SjctSatCurDensityGiven)
            model->BSIM4v6SjctSatCurDensity = 1.0E-4;
        if (!model->BSIM4v6DjctSatCurDensityGiven)
            model->BSIM4v6DjctSatCurDensity = model->BSIM4v6SjctSatCurDensity;
        if (!model->BSIM4v6SjctSidewallSatCurDensityGiven)
            model->BSIM4v6SjctSidewallSatCurDensity = 0.0;
        if (!model->BSIM4v6DjctSidewallSatCurDensityGiven)
            model->BSIM4v6DjctSidewallSatCurDensity = model->BSIM4v6SjctSidewallSatCurDensity;
        if (!model->BSIM4v6SjctGateSidewallSatCurDensityGiven)
            model->BSIM4v6SjctGateSidewallSatCurDensity = 0.0;
        if (!model->BSIM4v6DjctGateSidewallSatCurDensityGiven)
            model->BSIM4v6DjctGateSidewallSatCurDensity = model->BSIM4v6SjctGateSidewallSatCurDensity;
        if (!model->BSIM4v6SbulkJctPotentialGiven)
            model->BSIM4v6SbulkJctPotential = 1.0;
        if (!model->BSIM4v6DbulkJctPotentialGiven)
            model->BSIM4v6DbulkJctPotential = model->BSIM4v6SbulkJctPotential;
        if (!model->BSIM4v6SsidewallJctPotentialGiven)
            model->BSIM4v6SsidewallJctPotential = 1.0;
        if (!model->BSIM4v6DsidewallJctPotentialGiven)
            model->BSIM4v6DsidewallJctPotential = model->BSIM4v6SsidewallJctPotential;
        if (!model->BSIM4v6SGatesidewallJctPotentialGiven)
            model->BSIM4v6SGatesidewallJctPotential = model->BSIM4v6SsidewallJctPotential;
        if (!model->BSIM4v6DGatesidewallJctPotentialGiven)
            model->BSIM4v6DGatesidewallJctPotential = model->BSIM4v6SGatesidewallJctPotential;
        if (!model->BSIM4v6SbulkJctBotGradingCoeffGiven)
            model->BSIM4v6SbulkJctBotGradingCoeff = 0.5;
        if (!model->BSIM4v6DbulkJctBotGradingCoeffGiven)
            model->BSIM4v6DbulkJctBotGradingCoeff = model->BSIM4v6SbulkJctBotGradingCoeff;
        if (!model->BSIM4v6SbulkJctSideGradingCoeffGiven)
            model->BSIM4v6SbulkJctSideGradingCoeff = 0.33;
        if (!model->BSIM4v6DbulkJctSideGradingCoeffGiven)
            model->BSIM4v6DbulkJctSideGradingCoeff = model->BSIM4v6SbulkJctSideGradingCoeff;
        if (!model->BSIM4v6SbulkJctGateSideGradingCoeffGiven)
            model->BSIM4v6SbulkJctGateSideGradingCoeff = model->BSIM4v6SbulkJctSideGradingCoeff;
        if (!model->BSIM4v6DbulkJctGateSideGradingCoeffGiven)
            model->BSIM4v6DbulkJctGateSideGradingCoeff = model->BSIM4v6SbulkJctGateSideGradingCoeff;
        if (!model->BSIM4v6SjctEmissionCoeffGiven)
            model->BSIM4v6SjctEmissionCoeff = 1.0;
        if (!model->BSIM4v6DjctEmissionCoeffGiven)
            model->BSIM4v6DjctEmissionCoeff = model->BSIM4v6SjctEmissionCoeff;
        if (!model->BSIM4v6SjctTempExponentGiven)
            model->BSIM4v6SjctTempExponent = 3.0;
        if (!model->BSIM4v6DjctTempExponentGiven)
            model->BSIM4v6DjctTempExponent = model->BSIM4v6SjctTempExponent;

        if (!model->BSIM4v6jtssGiven)
            model->BSIM4v6jtss = 0.0;
        if (!model->BSIM4v6jtsdGiven)
            model->BSIM4v6jtsd = model->BSIM4v6jtss;
        if (!model->BSIM4v6jtsswsGiven)
            model->BSIM4v6jtssws = 0.0;
        if (!model->BSIM4v6jtsswdGiven)
            model->BSIM4v6jtsswd = model->BSIM4v6jtssws;
        if (!model->BSIM4v6jtsswgsGiven)
            model->BSIM4v6jtsswgs = 0.0;
        if (!model->BSIM4v6jtsswgdGiven)
            model->BSIM4v6jtsswgd = model->BSIM4v6jtsswgs;
		if (!model->BSIM4v6jtweffGiven)
		    model->BSIM4v6jtweff = 0.0;
        if (!model->BSIM4v6njtsGiven)
            model->BSIM4v6njts = 20.0;
        if (!model->BSIM4v6njtsswGiven)
            model->BSIM4v6njtssw = 20.0;
        if (!model->BSIM4v6njtsswgGiven)
            model->BSIM4v6njtsswg = 20.0;
	if (!model->BSIM4v6njtsdGiven)
        {
            if (model->BSIM4v6njtsGiven)
                model->BSIM4v6njtsd =  model->BSIM4v6njts;
	    else
	      model->BSIM4v6njtsd = 20.0;
        }
	if (!model->BSIM4v6njtsswdGiven)
        {
            if (model->BSIM4v6njtsswGiven)
                model->BSIM4v6njtsswd =  model->BSIM4v6njtssw;
	    else
	      model->BSIM4v6njtsswd = 20.0;
        }
	if (!model->BSIM4v6njtsswgdGiven)
        {
            if (model->BSIM4v6njtsswgGiven)
                model->BSIM4v6njtsswgd =  model->BSIM4v6njtsswg;
	    else
	      model->BSIM4v6njtsswgd = 20.0;
        }
        if (!model->BSIM4v6xtssGiven)
            model->BSIM4v6xtss = 0.02;
        if (!model->BSIM4v6xtsdGiven)
            model->BSIM4v6xtsd = model->BSIM4v6xtss;
        if (!model->BSIM4v6xtsswsGiven)
            model->BSIM4v6xtssws = 0.02;
        if (!model->BSIM4v6xtsswdGiven)
            model->BSIM4v6xtsswd = model->BSIM4v6xtssws;
        if (!model->BSIM4v6xtsswgsGiven)
            model->BSIM4v6xtsswgs = 0.02;
        if (!model->BSIM4v6xtsswgdGiven)
            model->BSIM4v6xtsswgd = model->BSIM4v6xtsswgs;
        if (!model->BSIM4v6tnjtsGiven)
            model->BSIM4v6tnjts = 0.0;
        if (!model->BSIM4v6tnjtsswGiven)
            model->BSIM4v6tnjtssw = 0.0;
        if (!model->BSIM4v6tnjtsswgGiven)
            model->BSIM4v6tnjtsswg = 0.0;
	if (!model->BSIM4v6tnjtsdGiven)
        {
            if (model->BSIM4v6tnjtsGiven)
                model->BSIM4v6tnjtsd =  model->BSIM4v6tnjts;
	    else
	      model->BSIM4v6tnjtsd = 0.0;
        }
	if (!model->BSIM4v6tnjtsswdGiven)
        {
            if (model->BSIM4v6tnjtsswGiven)
                model->BSIM4v6tnjtsswd =  model->BSIM4v6tnjtssw;
	    else
	      model->BSIM4v6tnjtsswd = 0.0;
        }
	if (!model->BSIM4v6tnjtsswgdGiven)
        {
            if (model->BSIM4v6tnjtsswgGiven)
                model->BSIM4v6tnjtsswgd =  model->BSIM4v6tnjtsswg;
	    else
	      model->BSIM4v6tnjtsswgd = 0.0;
        }
        if (!model->BSIM4v6vtssGiven)
            model->BSIM4v6vtss = 10.0;
        if (!model->BSIM4v6vtsdGiven)
            model->BSIM4v6vtsd = model->BSIM4v6vtss;
        if (!model->BSIM4v6vtsswsGiven)
            model->BSIM4v6vtssws = 10.0;
        if (!model->BSIM4v6vtsswdGiven)
            model->BSIM4v6vtsswd = model->BSIM4v6vtssws;
        if (!model->BSIM4v6vtsswgsGiven)
            model->BSIM4v6vtsswgs = 10.0;
        if (!model->BSIM4v6vtsswgdGiven)
            model->BSIM4v6vtsswgd = model->BSIM4v6vtsswgs;

        if (!model->BSIM4v6oxideTrapDensityAGiven)
	{   if (model->BSIM4v6type == NMOS)
                model->BSIM4v6oxideTrapDensityA = 6.25e41;
            else
                model->BSIM4v6oxideTrapDensityA= 6.188e40;
	}
        if (!model->BSIM4v6oxideTrapDensityBGiven)
	{   if (model->BSIM4v6type == NMOS)
                model->BSIM4v6oxideTrapDensityB = 3.125e26;
            else
                model->BSIM4v6oxideTrapDensityB = 1.5e25;
	}
        if (!model->BSIM4v6oxideTrapDensityCGiven)
            model->BSIM4v6oxideTrapDensityC = 8.75e9;
        if (!model->BSIM4v6emGiven)
            model->BSIM4v6em = 4.1e7; /* V/m */
        if (!model->BSIM4v6efGiven)
            model->BSIM4v6ef = 1.0;
        if (!model->BSIM4v6afGiven)
            model->BSIM4v6af = 1.0;
        if (!model->BSIM4v6kfGiven)
            model->BSIM4v6kf = 0.0;

        if (!model->BSIM4v6vgsMaxGiven)
            model->BSIM4v6vgsMax = 1e99;
        if (!model->BSIM4v6vgdMaxGiven)
            model->BSIM4v6vgdMax = 1e99;
        if (!model->BSIM4v6vgbMaxGiven)
            model->BSIM4v6vgbMax = 1e99;
        if (!model->BSIM4v6vdsMaxGiven)
            model->BSIM4v6vdsMax = 1e99;
        if (!model->BSIM4v6vbsMaxGiven)
            model->BSIM4v6vbsMax = 1e99;
        if (!model->BSIM4v6vbdMaxGiven)
            model->BSIM4v6vbdMax = 1e99;
        if (!model->BSIM4v6vgsrMaxGiven)
            model->BSIM4v6vgsrMax = 1e99;
        if (!model->BSIM4v6vgdrMaxGiven)
            model->BSIM4v6vgdrMax = 1e99;
        if (!model->BSIM4v6vgbrMaxGiven)
            model->BSIM4v6vgbrMax = 1e99;
        if (!model->BSIM4v6vbsrMaxGiven)
            model->BSIM4v6vbsrMax = 1e99;
        if (!model->BSIM4v6vbdrMaxGiven)
            model->BSIM4v6vbdrMax = 1e99;

        /* stress effect */
        if (!model->BSIM4v6sarefGiven)
            model->BSIM4v6saref = 1e-6; /* m */
        if (!model->BSIM4v6sbrefGiven)
            model->BSIM4v6sbref = 1e-6;  /* m */
        if (!model->BSIM4v6wlodGiven)
            model->BSIM4v6wlod = 0;  /* m */
        if (!model->BSIM4v6ku0Given)
            model->BSIM4v6ku0 = 0; /* 1/m */
        if (!model->BSIM4v6kvsatGiven)
            model->BSIM4v6kvsat = 0;
        if (!model->BSIM4v6kvth0Given) /* m */
            model->BSIM4v6kvth0 = 0;
        if (!model->BSIM4v6tku0Given)
            model->BSIM4v6tku0 = 0;
        if (!model->BSIM4v6llodku0Given)
            model->BSIM4v6llodku0 = 0;
        if (!model->BSIM4v6wlodku0Given)
            model->BSIM4v6wlodku0 = 0;
        if (!model->BSIM4v6llodvthGiven)
            model->BSIM4v6llodvth = 0;
        if (!model->BSIM4v6wlodvthGiven)
            model->BSIM4v6wlodvth = 0;
        if (!model->BSIM4v6lku0Given)
            model->BSIM4v6lku0 = 0;
        if (!model->BSIM4v6wku0Given)
            model->BSIM4v6wku0 = 0;
        if (!model->BSIM4v6pku0Given)
            model->BSIM4v6pku0 = 0;
        if (!model->BSIM4v6lkvth0Given)
            model->BSIM4v6lkvth0 = 0;
        if (!model->BSIM4v6wkvth0Given)
            model->BSIM4v6wkvth0 = 0;
        if (!model->BSIM4v6pkvth0Given)
            model->BSIM4v6pkvth0 = 0;
        if (!model->BSIM4v6stk2Given)
            model->BSIM4v6stk2 = 0;
        if (!model->BSIM4v6lodk2Given)
            model->BSIM4v6lodk2 = 1.0;
        if (!model->BSIM4v6steta0Given)
            model->BSIM4v6steta0 = 0;
        if (!model->BSIM4v6lodeta0Given)
            model->BSIM4v6lodeta0 = 1.0;

	/* Well Proximity Effect  */
        if (!model->BSIM4v6webGiven)
            model->BSIM4v6web = 0.0; 
        if (!model->BSIM4v6wecGiven)
            model->BSIM4v6wec = 0.0;
        if (!model->BSIM4v6kvth0weGiven)
            model->BSIM4v6kvth0we = 0.0; 
        if (!model->BSIM4v6k2weGiven)
            model->BSIM4v6k2we = 0.0; 
        if (!model->BSIM4v6ku0weGiven)
            model->BSIM4v6ku0we = 0.0; 
        if (!model->BSIM4v6screfGiven)
            model->BSIM4v6scref = 1.0E-6; /* m */
        if (!model->BSIM4v6wpemodGiven)
            model->BSIM4v6wpemod = 0; 
        else if ((model->BSIM4v6wpemod != 0) && (model->BSIM4v6wpemod != 1))
        {   model->BSIM4v6wpemod = 0;
            printf("Warning: wpemod has been set to its default value: 0.\n");
        }
        if (!model->BSIM4v6lkvth0weGiven)
            model->BSIM4v6lkvth0we = 0; 
        if (!model->BSIM4v6lk2weGiven)
            model->BSIM4v6lk2we = 0;
        if (!model->BSIM4v6lku0weGiven)
            model->BSIM4v6lku0we = 0;
        if (!model->BSIM4v6wkvth0weGiven)
            model->BSIM4v6wkvth0we = 0; 
        if (!model->BSIM4v6wk2weGiven)
            model->BSIM4v6wk2we = 0;
        if (!model->BSIM4v6wku0weGiven)
            model->BSIM4v6wku0we = 0;
        if (!model->BSIM4v6pkvth0weGiven)
            model->BSIM4v6pkvth0we = 0; 
        if (!model->BSIM4v6pk2weGiven)
            model->BSIM4v6pk2we = 0;
        if (!model->BSIM4v6pku0weGiven)
            model->BSIM4v6pku0we = 0;

        DMCGeff = model->BSIM4v6dmcg - model->BSIM4v6dmcgt;
        DMCIeff = model->BSIM4v6dmci;
        DMDGeff = model->BSIM4v6dmdg - model->BSIM4v6dmcgt;

	/*
         * End processing models and begin to loop
         * through all the instances of the model
         */

        for (here = BSIM4v6instances(model); here != NULL ;
             here=BSIM4v6nextInstance(here)) 
        {   
            /* allocate a chunk of the state vector */
            here->BSIM4v6states = *states;
            *states += BSIM4v6numStates;

            /* perform the parameter defaulting */
            if (!here->BSIM4v6lGiven)
                here->BSIM4v6l = 5.0e-6;
            if (!here->BSIM4v6wGiven)
                here->BSIM4v6w = 5.0e-6;
            if (!here->BSIM4v6mGiven)
                here->BSIM4v6m = 1.0;
            if (!here->BSIM4v6nfGiven)
                here->BSIM4v6nf = 1.0;
            if (!here->BSIM4v6minGiven)
                here->BSIM4v6min = 0; /* integer */
            if (!here->BSIM4v6icVDSGiven)
                here->BSIM4v6icVDS = 0.0;
            if (!here->BSIM4v6icVGSGiven)
                here->BSIM4v6icVGS = 0.0;
            if (!here->BSIM4v6icVBSGiven)
                here->BSIM4v6icVBS = 0.0;
            if (!here->BSIM4v6drainAreaGiven)
                here->BSIM4v6drainArea = 0.0;
            if (!here->BSIM4v6drainPerimeterGiven)
                here->BSIM4v6drainPerimeter = 0.0;
            if (!here->BSIM4v6drainSquaresGiven)
                here->BSIM4v6drainSquares = 1.0;
            if (!here->BSIM4v6sourceAreaGiven)
                here->BSIM4v6sourceArea = 0.0;
            if (!here->BSIM4v6sourcePerimeterGiven)
                here->BSIM4v6sourcePerimeter = 0.0;
            if (!here->BSIM4v6sourceSquaresGiven)
                here->BSIM4v6sourceSquares = 1.0;

            if (!here->BSIM4v6rbdbGiven)
                here->BSIM4v6rbdb = model->BSIM4v6rbdb; /* in ohm */
            if (!here->BSIM4v6rbsbGiven)
                here->BSIM4v6rbsb = model->BSIM4v6rbsb;
            if (!here->BSIM4v6rbpbGiven)
                here->BSIM4v6rbpb = model->BSIM4v6rbpb;
            if (!here->BSIM4v6rbpsGiven)
                here->BSIM4v6rbps = model->BSIM4v6rbps;
            if (!here->BSIM4v6rbpdGiven)
                here->BSIM4v6rbpd = model->BSIM4v6rbpd;
            if (!here->BSIM4v6delvtoGiven)
                here->BSIM4v6delvto = 0.0;
            if (!here->BSIM4v6mulu0Given)
                here->BSIM4v6mulu0 = 1.0;
            if (!here->BSIM4v6xgwGiven)
                here->BSIM4v6xgw = model->BSIM4v6xgw;
            if (!here->BSIM4v6ngconGiven)
                here->BSIM4v6ngcon = model->BSIM4v6ngcon;

                    
            /* Process instance model selectors, some
             * may override their global counterparts
	     */
            if (!here->BSIM4v6rbodyModGiven)
                here->BSIM4v6rbodyMod = model->BSIM4v6rbodyMod;
            else if ((here->BSIM4v6rbodyMod != 0) && (here->BSIM4v6rbodyMod != 1) && (here->BSIM4v6rbodyMod != 2))
            {   here->BSIM4v6rbodyMod = model->BSIM4v6rbodyMod;
                printf("Warning: rbodyMod has been set to its global value %d.\n",
	        model->BSIM4v6rbodyMod);
            }

            if (!here->BSIM4v6rgateModGiven)
                here->BSIM4v6rgateMod = model->BSIM4v6rgateMod;
            else if ((here->BSIM4v6rgateMod != 0) && (here->BSIM4v6rgateMod != 1)
	        && (here->BSIM4v6rgateMod != 2) && (here->BSIM4v6rgateMod != 3))
            {   here->BSIM4v6rgateMod = model->BSIM4v6rgateMod;
                printf("Warning: rgateMod has been set to its global value %d.\n",
                model->BSIM4v6rgateMod);
            }

            if (!here->BSIM4v6geoModGiven)
                here->BSIM4v6geoMod = model->BSIM4v6geoMod;

            if (!here->BSIM4v6rgeoModGiven)
                here->BSIM4v6rgeoMod = model->BSIM4v6rgeoMod;
            else if ((here->BSIM4v6rgeoMod != 0) && (here->BSIM4v6rgeoMod != 1))
            {   here->BSIM4v6rgeoMod = model->BSIM4v6rgeoMod;
                printf("Warning: rgeoMod has been set to its global value %d.\n",
                model->BSIM4v6rgeoMod);
            }
            if (!here->BSIM4v6trnqsModGiven)
                here->BSIM4v6trnqsMod = model->BSIM4v6trnqsMod;
            else if ((here->BSIM4v6trnqsMod != 0) && (here->BSIM4v6trnqsMod != 1))
            {   here->BSIM4v6trnqsMod = model->BSIM4v6trnqsMod;
                printf("Warning: trnqsMod has been set to its global value %d.\n",
                model->BSIM4v6trnqsMod);
            }

            if (!here->BSIM4v6acnqsModGiven)
                here->BSIM4v6acnqsMod = model->BSIM4v6acnqsMod;
            else if ((here->BSIM4v6acnqsMod != 0) && (here->BSIM4v6acnqsMod != 1))
            {   here->BSIM4v6acnqsMod = model->BSIM4v6acnqsMod;
                printf("Warning: acnqsMod has been set to its global value %d.\n",
                model->BSIM4v6acnqsMod);
            }

            /* stress effect */
            if (!here->BSIM4v6saGiven)
                here->BSIM4v6sa = 0.0;
            if (!here->BSIM4v6sbGiven)
                here->BSIM4v6sb = 0.0;
            if (!here->BSIM4v6sdGiven)
                here->BSIM4v6sd = 2 * model->BSIM4v6dmcg;
	    /* Well Proximity Effect  */
	    if (!here->BSIM4v6scaGiven)
                here->BSIM4v6sca = 0.0;
	    if (!here->BSIM4v6scbGiven)
                here->BSIM4v6scb = 0.0;
	    if (!here->BSIM4v6sccGiven)
                here->BSIM4v6scc = 0.0;
	    if (!here->BSIM4v6scGiven)
                here->BSIM4v6sc = 0.0; /* m */

            /* process drain series resistance */
            createNode = 0;
            if ( (model->BSIM4v6rdsMod != 0)
                            || (model->BSIM4v6tnoiMod != 0 && noiseAnalGiven))
            {
               createNode = 1;
            } else if (model->BSIM4v6sheetResistance > 0)
            {
                     if (here->BSIM4v6drainSquaresGiven
                                       && here->BSIM4v6drainSquares > 0)
                     {
                          createNode = 1;
                     } else if (!here->BSIM4v6drainSquaresGiven
                                       && (here->BSIM4v6rgeoMod != 0))
                     {
                          BSIM4v6RdseffGeo(here->BSIM4v6nf*here->BSIM4v6m, here->BSIM4v6geoMod,
                                  here->BSIM4v6rgeoMod, here->BSIM4v6min,
                                  here->BSIM4v6w, model->BSIM4v6sheetResistance,
                                  DMCGeff, DMCIeff, DMDGeff, 0, &Rtot);
                          if(Rtot > 0)
                             createNode = 1;
                     }
            }
            if ( createNode != 0 )
            {   if (here->BSIM4v6dNodePrime == 0)
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v6name,"drain");
                if(error) return(error);
                here->BSIM4v6dNodePrime = tmp->number;
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
            {   here->BSIM4v6dNodePrime = here->BSIM4v6dNode;
            }
            
            /* process source series resistance */
            createNode = 0;
            if ( (model->BSIM4v6rdsMod != 0)
                            || (model->BSIM4v6tnoiMod != 0 && noiseAnalGiven))
            {
               createNode = 1;
            } else if (model->BSIM4v6sheetResistance > 0)
            {
                     if (here->BSIM4v6sourceSquaresGiven
                                        && here->BSIM4v6sourceSquares > 0)
                     {
                          createNode = 1;
                     } else if (!here->BSIM4v6sourceSquaresGiven
                                        && (here->BSIM4v6rgeoMod != 0))
                     {
                          BSIM4v6RdseffGeo(here->BSIM4v6nf*here->BSIM4v6m, here->BSIM4v6geoMod,
                                  here->BSIM4v6rgeoMod, here->BSIM4v6min,
                                  here->BSIM4v6w, model->BSIM4v6sheetResistance,
                                  DMCGeff, DMCIeff, DMDGeff, 1, &Rtot);
                          if(Rtot > 0)
                             createNode = 1;
                     }
            }
            if ( createNode != 0 )
            {   if (here->BSIM4v6sNodePrime == 0)
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v6name,"source");
                if(error) return(error);
                here->BSIM4v6sNodePrime = tmp->number;
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
                here->BSIM4v6sNodePrime = here->BSIM4v6sNode;

            if (here->BSIM4v6rgateMod > 0)
            {   if (here->BSIM4v6gNodePrime == 0)
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v6name,"gate");
                if(error) return(error);
                   here->BSIM4v6gNodePrime = tmp->number;
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
                here->BSIM4v6gNodePrime = here->BSIM4v6gNodeExt;

            if (here->BSIM4v6rgateMod == 3)
            {   if (here->BSIM4v6gNodeMid == 0)
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v6name,"midgate");
                if(error) return(error);
                   here->BSIM4v6gNodeMid = tmp->number;
            }
            }
            else
                here->BSIM4v6gNodeMid = here->BSIM4v6gNodeExt;
            

            /* internal body nodes for body resistance model */
            if ((here->BSIM4v6rbodyMod ==1) || (here->BSIM4v6rbodyMod ==2))
            {   if (here->BSIM4v6dbNode == 0)
		{   error = CKTmkVolt(ckt,&tmp,here->BSIM4v6name,"dbody");
                    if(error) return(error);
                    here->BSIM4v6dbNode = tmp->number;
		}
		if (here->BSIM4v6bNodePrime == 0)
                {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v6name,"body");
                    if(error) return(error);
                    here->BSIM4v6bNodePrime = tmp->number;
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
		if (here->BSIM4v6sbNode == 0)
                {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v6name,"sbody");
                    if(error) return(error);
                    here->BSIM4v6sbNode = tmp->number;
                }
            }
	    else
	        here->BSIM4v6dbNode = here->BSIM4v6bNodePrime = here->BSIM4v6sbNode
				  = here->BSIM4v6bNode;

            /* NQS node */
            if (here->BSIM4v6trnqsMod)
            {   if (here->BSIM4v6qNode == 0)
	    {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v6name,"charge");
                if(error) return(error);
                here->BSIM4v6qNode = tmp->number;
            }
            }
	    else 
	        here->BSIM4v6qNode = 0;

/* set Sparse Matrix Pointers 
 * macro to make elements with built-in out-of-memory test */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(BSIM4v6DPbpPtr, BSIM4v6dNodePrime, BSIM4v6bNodePrime);
            TSTALLOC(BSIM4v6GPbpPtr, BSIM4v6gNodePrime, BSIM4v6bNodePrime);
            TSTALLOC(BSIM4v6SPbpPtr, BSIM4v6sNodePrime, BSIM4v6bNodePrime);

            TSTALLOC(BSIM4v6BPdpPtr, BSIM4v6bNodePrime, BSIM4v6dNodePrime);
            TSTALLOC(BSIM4v6BPgpPtr, BSIM4v6bNodePrime, BSIM4v6gNodePrime);
            TSTALLOC(BSIM4v6BPspPtr, BSIM4v6bNodePrime, BSIM4v6sNodePrime);
            TSTALLOC(BSIM4v6BPbpPtr, BSIM4v6bNodePrime, BSIM4v6bNodePrime);

            TSTALLOC(BSIM4v6DdPtr, BSIM4v6dNode, BSIM4v6dNode);
            TSTALLOC(BSIM4v6GPgpPtr, BSIM4v6gNodePrime, BSIM4v6gNodePrime);
            TSTALLOC(BSIM4v6SsPtr, BSIM4v6sNode, BSIM4v6sNode);
            TSTALLOC(BSIM4v6DPdpPtr, BSIM4v6dNodePrime, BSIM4v6dNodePrime);
            TSTALLOC(BSIM4v6SPspPtr, BSIM4v6sNodePrime, BSIM4v6sNodePrime);
            TSTALLOC(BSIM4v6DdpPtr, BSIM4v6dNode, BSIM4v6dNodePrime);
            TSTALLOC(BSIM4v6GPdpPtr, BSIM4v6gNodePrime, BSIM4v6dNodePrime);
            TSTALLOC(BSIM4v6GPspPtr, BSIM4v6gNodePrime, BSIM4v6sNodePrime);
            TSTALLOC(BSIM4v6SspPtr, BSIM4v6sNode, BSIM4v6sNodePrime);
            TSTALLOC(BSIM4v6DPspPtr, BSIM4v6dNodePrime, BSIM4v6sNodePrime);
            TSTALLOC(BSIM4v6DPdPtr, BSIM4v6dNodePrime, BSIM4v6dNode);
            TSTALLOC(BSIM4v6DPgpPtr, BSIM4v6dNodePrime, BSIM4v6gNodePrime);
            TSTALLOC(BSIM4v6SPgpPtr, BSIM4v6sNodePrime, BSIM4v6gNodePrime);
            TSTALLOC(BSIM4v6SPsPtr, BSIM4v6sNodePrime, BSIM4v6sNode);
            TSTALLOC(BSIM4v6SPdpPtr, BSIM4v6sNodePrime, BSIM4v6dNodePrime);

            TSTALLOC(BSIM4v6QqPtr, BSIM4v6qNode, BSIM4v6qNode);
            TSTALLOC(BSIM4v6QbpPtr, BSIM4v6qNode, BSIM4v6bNodePrime) ;
            TSTALLOC(BSIM4v6QdpPtr, BSIM4v6qNode, BSIM4v6dNodePrime);
            TSTALLOC(BSIM4v6QspPtr, BSIM4v6qNode, BSIM4v6sNodePrime);
            TSTALLOC(BSIM4v6QgpPtr, BSIM4v6qNode, BSIM4v6gNodePrime);
            TSTALLOC(BSIM4v6DPqPtr, BSIM4v6dNodePrime, BSIM4v6qNode);
            TSTALLOC(BSIM4v6SPqPtr, BSIM4v6sNodePrime, BSIM4v6qNode);
            TSTALLOC(BSIM4v6GPqPtr, BSIM4v6gNodePrime, BSIM4v6qNode);

            if (here->BSIM4v6rgateMod != 0)
            {   TSTALLOC(BSIM4v6GEgePtr, BSIM4v6gNodeExt, BSIM4v6gNodeExt);
                TSTALLOC(BSIM4v6GEgpPtr, BSIM4v6gNodeExt, BSIM4v6gNodePrime);
                TSTALLOC(BSIM4v6GPgePtr, BSIM4v6gNodePrime, BSIM4v6gNodeExt);
                TSTALLOC(BSIM4v6GEdpPtr, BSIM4v6gNodeExt, BSIM4v6dNodePrime);
                TSTALLOC(BSIM4v6GEspPtr, BSIM4v6gNodeExt, BSIM4v6sNodePrime);
                TSTALLOC(BSIM4v6GEbpPtr, BSIM4v6gNodeExt, BSIM4v6bNodePrime);

                TSTALLOC(BSIM4v6GMdpPtr, BSIM4v6gNodeMid, BSIM4v6dNodePrime);
                TSTALLOC(BSIM4v6GMgpPtr, BSIM4v6gNodeMid, BSIM4v6gNodePrime);
                TSTALLOC(BSIM4v6GMgmPtr, BSIM4v6gNodeMid, BSIM4v6gNodeMid);
                TSTALLOC(BSIM4v6GMgePtr, BSIM4v6gNodeMid, BSIM4v6gNodeExt);
                TSTALLOC(BSIM4v6GMspPtr, BSIM4v6gNodeMid, BSIM4v6sNodePrime);
                TSTALLOC(BSIM4v6GMbpPtr, BSIM4v6gNodeMid, BSIM4v6bNodePrime);
                TSTALLOC(BSIM4v6DPgmPtr, BSIM4v6dNodePrime, BSIM4v6gNodeMid);
                TSTALLOC(BSIM4v6GPgmPtr, BSIM4v6gNodePrime, BSIM4v6gNodeMid);
                TSTALLOC(BSIM4v6GEgmPtr, BSIM4v6gNodeExt, BSIM4v6gNodeMid);
                TSTALLOC(BSIM4v6SPgmPtr, BSIM4v6sNodePrime, BSIM4v6gNodeMid);
                TSTALLOC(BSIM4v6BPgmPtr, BSIM4v6bNodePrime, BSIM4v6gNodeMid);
            }	

            if ((here->BSIM4v6rbodyMod ==1) || (here->BSIM4v6rbodyMod ==2))
            {   TSTALLOC(BSIM4v6DPdbPtr, BSIM4v6dNodePrime, BSIM4v6dbNode);
                TSTALLOC(BSIM4v6SPsbPtr, BSIM4v6sNodePrime, BSIM4v6sbNode);

                TSTALLOC(BSIM4v6DBdpPtr, BSIM4v6dbNode, BSIM4v6dNodePrime);
                TSTALLOC(BSIM4v6DBdbPtr, BSIM4v6dbNode, BSIM4v6dbNode);
                TSTALLOC(BSIM4v6DBbpPtr, BSIM4v6dbNode, BSIM4v6bNodePrime);
                TSTALLOC(BSIM4v6DBbPtr, BSIM4v6dbNode, BSIM4v6bNode);

                TSTALLOC(BSIM4v6BPdbPtr, BSIM4v6bNodePrime, BSIM4v6dbNode);
                TSTALLOC(BSIM4v6BPbPtr, BSIM4v6bNodePrime, BSIM4v6bNode);
                TSTALLOC(BSIM4v6BPsbPtr, BSIM4v6bNodePrime, BSIM4v6sbNode);

                TSTALLOC(BSIM4v6SBspPtr, BSIM4v6sbNode, BSIM4v6sNodePrime);
                TSTALLOC(BSIM4v6SBbpPtr, BSIM4v6sbNode, BSIM4v6bNodePrime);
                TSTALLOC(BSIM4v6SBbPtr, BSIM4v6sbNode, BSIM4v6bNode);
                TSTALLOC(BSIM4v6SBsbPtr, BSIM4v6sbNode, BSIM4v6sbNode);

                TSTALLOC(BSIM4v6BdbPtr, BSIM4v6bNode, BSIM4v6dbNode);
                TSTALLOC(BSIM4v6BbpPtr, BSIM4v6bNode, BSIM4v6bNodePrime);
                TSTALLOC(BSIM4v6BsbPtr, BSIM4v6bNode, BSIM4v6sbNode);
                TSTALLOC(BSIM4v6BbPtr, BSIM4v6bNode, BSIM4v6bNode);
            }

            if (model->BSIM4v6rdsMod)
            {   TSTALLOC(BSIM4v6DgpPtr, BSIM4v6dNode, BSIM4v6gNodePrime);
                TSTALLOC(BSIM4v6DspPtr, BSIM4v6dNode, BSIM4v6sNodePrime);
                TSTALLOC(BSIM4v6DbpPtr, BSIM4v6dNode, BSIM4v6bNodePrime);
                TSTALLOC(BSIM4v6SdpPtr, BSIM4v6sNode, BSIM4v6dNodePrime);
                TSTALLOC(BSIM4v6SgpPtr, BSIM4v6sNode, BSIM4v6gNodePrime);
                TSTALLOC(BSIM4v6SbpPtr, BSIM4v6sNode, BSIM4v6bNodePrime);
            }
        }
    } /*  end of loop through all the BSIM4v6 device models */

#ifdef USE_OMP
    InstCount = 0;
    model = (BSIM4v6model*)inModel;
    /* loop through all the BSIM4v6 device models 
       to count the number of instances */
    
    for( ; model != NULL; model = BSIM4v6nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = BSIM4v6instances(model); here != NULL ;
             here=BSIM4v6nextInstance(here)) 
        { 
            InstCount++;
        }
        model->BSIM4v6InstCount = 0;
        model->BSIM4v6InstanceArray = NULL;
    }
    InstArray = TMALLOC(BSIM4v6instance*, InstCount);
    model = (BSIM4v6model*)inModel;
    /* store this in the first model only */
    model->BSIM4v6InstCount = InstCount;
    model->BSIM4v6InstanceArray = InstArray;
    idx = 0;
    for( ; model != NULL; model = BSIM4v6nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = BSIM4v6instances(model); here != NULL ;
             here=BSIM4v6nextInstance(here)) 
        { 
            InstArray[idx] = here;
            idx++;
        }
    }
#endif

    return(OK);
}  

int
BSIM4v6unsetup(GENmodel *inModel, CKTcircuit *ckt)
{
#ifndef HAS_BATCHSIM
    BSIM4v6model *model;
    BSIM4v6instance *here;

#ifdef USE_OMP
    model = (BSIM4v6model*)inModel;
    tfree(model->BSIM4v6InstanceArray);
#endif

    for (model = (BSIM4v6model *)inModel; model != NULL;
            model = BSIM4v6nextModel(model))
    {
        for (here = BSIM4v6instances(model); here != NULL;
                here=BSIM4v6nextInstance(here))
        {
            if (here->BSIM4v6qNode > 0)
                CKTdltNNum(ckt, here->BSIM4v6qNode);
            here->BSIM4v6qNode = 0;

            if (here->BSIM4v6sbNode > 0 &&
                here->BSIM4v6sbNode != here->BSIM4v6bNode)
                CKTdltNNum(ckt, here->BSIM4v6sbNode);
            here->BSIM4v6sbNode = 0;

            if (here->BSIM4v6bNodePrime > 0 &&
                here->BSIM4v6bNodePrime != here->BSIM4v6bNode)
                CKTdltNNum(ckt, here->BSIM4v6bNodePrime);
            here->BSIM4v6bNodePrime = 0;

            if (here->BSIM4v6dbNode > 0 &&
                here->BSIM4v6dbNode != here->BSIM4v6bNode)
                CKTdltNNum(ckt, here->BSIM4v6dbNode);
            here->BSIM4v6dbNode = 0;

            if (here->BSIM4v6gNodeMid > 0 &&
                here->BSIM4v6gNodeMid != here->BSIM4v6gNodeExt)
                CKTdltNNum(ckt, here->BSIM4v6gNodeMid);
            here->BSIM4v6gNodeMid = 0;

            if (here->BSIM4v6gNodePrime > 0 &&
                here->BSIM4v6gNodePrime != here->BSIM4v6gNodeExt)
                CKTdltNNum(ckt, here->BSIM4v6gNodePrime);
            here->BSIM4v6gNodePrime = 0;

            if (here->BSIM4v6sNodePrime > 0
                    && here->BSIM4v6sNodePrime != here->BSIM4v6sNode)
                CKTdltNNum(ckt, here->BSIM4v6sNodePrime);
            here->BSIM4v6sNodePrime = 0;

            if (here->BSIM4v6dNodePrime > 0
                    && here->BSIM4v6dNodePrime != here->BSIM4v6dNode)
                CKTdltNNum(ckt, here->BSIM4v6dNodePrime);
            here->BSIM4v6dNodePrime = 0;
        }
    }
#endif
    return OK;
}
