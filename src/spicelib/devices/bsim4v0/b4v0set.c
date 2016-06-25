/**** BSIM4.0.0, Released by Weidong Liu 3/24/2000 ****/

/**********
 * Copyright 2000 Regents of the University of California. All rights reserved.
 * File: b4set.c of BSIM4.0.0.
 * Authors: Weidong Liu, Xiaodong Jin, Kanyu M. Cao, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/jobdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "bsim4v0def.h"
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
BSIM4v0setup(matrix,inModel,ckt,states)
register SMPmatrix *matrix;
register GENmodel *inModel;
register CKTcircuit *ckt;
int *states;
{
register BSIM4v0model *model = (BSIM4v0model*)inModel;
register BSIM4v0instance *here;
int error;
CKTnode *tmp;

//double tmp1, tmp2;

    /*  loop through all the BSIM4v0 device models */
    for( ; model != NULL; model = model->BSIM4v0nextModel )
    {   /* process defaults of model parameters */
        if (!model->BSIM4v0typeGiven)
            model->BSIM4v0type = NMOS;     

        if (!model->BSIM4v0mobModGiven) 
            model->BSIM4v0mobMod = 0;
	else if ((model->BSIM4v0mobMod != 0) && (model->BSIM4v0mobMod != 1)
	         && (model->BSIM4v0mobMod != 2))
	{   model->BSIM4v0mobMod = 0;
            printf("Warning: mobMod has been set to its default value: 0.\n");
	}

        if (!model->BSIM4v0binUnitGiven) 
            model->BSIM4v0binUnit = 1;
        if (!model->BSIM4v0paramChkGiven) 
            model->BSIM4v0paramChk = 1;

        if (!model->BSIM4v0dioModGiven)
            model->BSIM4v0dioMod = 1;
        else if ((model->BSIM4v0dioMod != 0) && (model->BSIM4v0dioMod != 1)
            && (model->BSIM4v0dioMod != 2))
        {   model->BSIM4v0dioMod = 1;
            printf("Warning: dioMod has been set to its default value: 1.\n");
        }

        if (!model->BSIM4v0capModGiven) 
            model->BSIM4v0capMod = 2;
        else if ((model->BSIM4v0capMod != 0) && (model->BSIM4v0capMod != 1)
            && (model->BSIM4v0capMod != 2))
        {   model->BSIM4v0capMod = 2;
            printf("Warning: capMod has been set to its default value: 2.\n");
        }

        if (!model->BSIM4v0rdsModGiven)
            model->BSIM4v0rdsMod = 0;
	else if ((model->BSIM4v0rdsMod != 0) && (model->BSIM4v0rdsMod != 1))
        {   model->BSIM4v0rdsMod = 0;
	    printf("Warning: rdsMod has been set to its default value: 0.\n");
	}
        if (!model->BSIM4v0rbodyModGiven)
            model->BSIM4v0rbodyMod = 0;
        else if ((model->BSIM4v0rbodyMod != 0) && (model->BSIM4v0rbodyMod != 1))
        {   model->BSIM4v0rbodyMod = 0;
            printf("Warning: rbodyMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v0rgateModGiven)
            model->BSIM4v0rgateMod = 0;
        else if ((model->BSIM4v0rgateMod != 0) && (model->BSIM4v0rgateMod != 1)
            && (model->BSIM4v0rgateMod != 2) && (model->BSIM4v0rgateMod != 3))
        {   model->BSIM4v0rgateMod = 0;
            printf("Warning: rgateMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v0perModGiven)
            model->BSIM4v0perMod = 1;
        else if ((model->BSIM4v0perMod != 0) && (model->BSIM4v0perMod != 1))
        {   model->BSIM4v0perMod = 1;
            printf("Warning: perMod has been set to its default value: 1.\n");
        }

        if (!model->BSIM4v0geoModGiven)
            model->BSIM4v0geoMod = 0;

        if (!model->BSIM4v0fnoiModGiven) 
            model->BSIM4v0fnoiMod = 1;
        else if ((model->BSIM4v0fnoiMod != 0) && (model->BSIM4v0fnoiMod != 1))
        {   model->BSIM4v0fnoiMod = 1;
            printf("Warning: fnoiMod has been set to its default value: 1.\n");
        }
        if (!model->BSIM4v0tnoiModGiven)
            model->BSIM4v0tnoiMod = 0; /* WDLiu: tnoiMod=1 needs to set internal S/D nodes */
        else if ((model->BSIM4v0tnoiMod != 0) && (model->BSIM4v0tnoiMod != 1))
        {   model->BSIM4v0tnoiMod = 0;
            printf("Warning: tnoiMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v0trnqsModGiven)
            model->BSIM4v0trnqsMod = 0; 
        else if ((model->BSIM4v0trnqsMod != 0) && (model->BSIM4v0trnqsMod != 1))
        {   model->BSIM4v0trnqsMod = 0;
            printf("Warning: trnqsMod has been set to its default value: 0.\n");
        }
        if (!model->BSIM4v0acnqsModGiven)
            model->BSIM4v0acnqsMod = 0;
        else if ((model->BSIM4v0acnqsMod != 0) && (model->BSIM4v0acnqsMod != 1))
        {   model->BSIM4v0acnqsMod = 0;
            printf("Warning: acnqsMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v0igcModGiven)
            model->BSIM4v0igcMod = 0;
        else if ((model->BSIM4v0igcMod != 0) && (model->BSIM4v0igcMod != 1))
        {   model->BSIM4v0igcMod = 0;
            printf("Warning: igcMod has been set to its default value: 0.\n");
        }
        if (!model->BSIM4v0igbModGiven)
            model->BSIM4v0igbMod = 0;
        else if ((model->BSIM4v0igbMod != 0) && (model->BSIM4v0igbMod != 1))
        {   model->BSIM4v0igbMod = 0;
            printf("Warning: igbMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v0versionGiven) 
            model->BSIM4v0version = "4.0.0";
        if (!model->BSIM4v0toxrefGiven)
            model->BSIM4v0toxref = 30.0e-10;
        if (!model->BSIM4v0toxeGiven)
            model->BSIM4v0toxe = 30.0e-10;
        if (!model->BSIM4v0toxpGiven)
            model->BSIM4v0toxp = model->BSIM4v0toxe;
        if (!model->BSIM4v0toxmGiven)
            model->BSIM4v0toxm = model->BSIM4v0toxe;
        if (!model->BSIM4v0dtoxGiven)
            model->BSIM4v0dtox = 0.0;
        if (!model->BSIM4v0epsroxGiven)
            model->BSIM4v0epsrox = 3.9;

        if (!model->BSIM4v0cdscGiven)
	    model->BSIM4v0cdsc = 2.4e-4;   /* unit Q/V/m^2  */
        if (!model->BSIM4v0cdscbGiven)
	    model->BSIM4v0cdscb = 0.0;   /* unit Q/V/m^2  */    
	    if (!model->BSIM4v0cdscdGiven)
	    model->BSIM4v0cdscd = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM4v0citGiven)
	    model->BSIM4v0cit = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM4v0nfactorGiven)
	    model->BSIM4v0nfactor = 1.0;
        if (!model->BSIM4v0xjGiven)
            model->BSIM4v0xj = .15e-6;
        if (!model->BSIM4v0vsatGiven)
            model->BSIM4v0vsat = 8.0e4;    /* unit m/s */ 
        if (!model->BSIM4v0atGiven)
            model->BSIM4v0at = 3.3e4;    /* unit m/s */ 
        if (!model->BSIM4v0a0Given)
            model->BSIM4v0a0 = 1.0;  
        if (!model->BSIM4v0agsGiven)
            model->BSIM4v0ags = 0.0;
        if (!model->BSIM4v0a1Given)
            model->BSIM4v0a1 = 0.0;
        if (!model->BSIM4v0a2Given)
            model->BSIM4v0a2 = 1.0;
        if (!model->BSIM4v0ketaGiven)
            model->BSIM4v0keta = -0.047;    /* unit  / V */
        if (!model->BSIM4v0nsubGiven)
            model->BSIM4v0nsub = 6.0e16;   /* unit 1/cm3 */
        if (!model->BSIM4v0ndepGiven)
            model->BSIM4v0ndep = 1.7e17;   /* unit 1/cm3 */
        if (!model->BSIM4v0nsdGiven)
            model->BSIM4v0nsd = 1.0e20;   /* unit 1/cm3 */
        if (!model->BSIM4v0phinGiven)
            model->BSIM4v0phin = 0.0; /* unit V */
        if (!model->BSIM4v0ngateGiven)
            model->BSIM4v0ngate = 0;   /* unit 1/cm3 */
        if (!model->BSIM4v0vbmGiven)
	    model->BSIM4v0vbm = -3.0;
        if (!model->BSIM4v0xtGiven)
	    model->BSIM4v0xt = 1.55e-7;
        if (!model->BSIM4v0kt1Given)
            model->BSIM4v0kt1 = -0.11;      /* unit V */
        if (!model->BSIM4v0kt1lGiven)
            model->BSIM4v0kt1l = 0.0;      /* unit V*m */
        if (!model->BSIM4v0kt2Given)
            model->BSIM4v0kt2 = 0.022;      /* No unit */
        if (!model->BSIM4v0k3Given)
            model->BSIM4v0k3 = 80.0;      
        if (!model->BSIM4v0k3bGiven)
            model->BSIM4v0k3b = 0.0;      
        if (!model->BSIM4v0w0Given)
            model->BSIM4v0w0 = 2.5e-6;    
        if (!model->BSIM4v0lpe0Given)
            model->BSIM4v0lpe0 = 1.74e-7;     
        if (!model->BSIM4v0lpebGiven)
            model->BSIM4v0lpeb = 0.0;
        if (!model->BSIM4v0dvtp0Given)
            model->BSIM4v0dvtp0 = 0.0;
        if (!model->BSIM4v0dvtp1Given)
            model->BSIM4v0dvtp1 = 0.0;
        if (!model->BSIM4v0dvt0Given)
            model->BSIM4v0dvt0 = 2.2;    
        if (!model->BSIM4v0dvt1Given)
            model->BSIM4v0dvt1 = 0.53;      
        if (!model->BSIM4v0dvt2Given)
            model->BSIM4v0dvt2 = -0.032;   /* unit 1 / V */     

        if (!model->BSIM4v0dvt0wGiven)
            model->BSIM4v0dvt0w = 0.0;    
        if (!model->BSIM4v0dvt1wGiven)
            model->BSIM4v0dvt1w = 5.3e6;    
        if (!model->BSIM4v0dvt2wGiven)
            model->BSIM4v0dvt2w = -0.032;   

        if (!model->BSIM4v0droutGiven)
            model->BSIM4v0drout = 0.56;     
        if (!model->BSIM4v0dsubGiven)
            model->BSIM4v0dsub = model->BSIM4v0drout;     
        if (!model->BSIM4v0vth0Given)
            model->BSIM4v0vth0 = (model->BSIM4v0type == NMOS) ? 0.7 : -0.7;
        if (!model->BSIM4v0euGiven)
            model->BSIM4v0eu = (model->BSIM4v0type == NMOS) ? 1.67 : 1.0;;
        if (!model->BSIM4v0uaGiven)
            model->BSIM4v0ua = (model->BSIM4v0mobMod == 2) ? 1.0e-15 : 1.0e-9; /* unit m/V */
        if (!model->BSIM4v0ua1Given)
            model->BSIM4v0ua1 = 1.0e-9;      /* unit m/V */
        if (!model->BSIM4v0ubGiven)
            model->BSIM4v0ub = 1.0e-19;     /* unit (m/V)**2 */
        if (!model->BSIM4v0ub1Given)
            model->BSIM4v0ub1 = -1.0e-18;     /* unit (m/V)**2 */
        if (!model->BSIM4v0ucGiven)
            model->BSIM4v0uc = (model->BSIM4v0mobMod == 1) ? -0.0465 : -0.0465e-9;   
        if (!model->BSIM4v0uc1Given)
            model->BSIM4v0uc1 = (model->BSIM4v0mobMod == 1) ? -0.056 : -0.056e-9;   
        if (!model->BSIM4v0u0Given)
            model->BSIM4v0u0 = (model->BSIM4v0type == NMOS) ? 0.067 : 0.025;
        if (!model->BSIM4v0uteGiven)
	    model->BSIM4v0ute = -1.5;    
        if (!model->BSIM4v0voffGiven)
	    model->BSIM4v0voff = -0.08;
        if (!model->BSIM4v0vofflGiven)
            model->BSIM4v0voffl = 0.0;
        if (!model->BSIM4v0minvGiven)
            model->BSIM4v0minv = 0.0;
        if (!model->BSIM4v0fproutGiven)
            model->BSIM4v0fprout = 0.0;
        if (!model->BSIM4v0pditsGiven)
            model->BSIM4v0pdits = 0.0;
        if (!model->BSIM4v0pditsdGiven)
            model->BSIM4v0pditsd = 0.0;
        if (!model->BSIM4v0pditslGiven)
            model->BSIM4v0pditsl = 0.0;
        if (!model->BSIM4v0deltaGiven)  
           model->BSIM4v0delta = 0.01;
        if (!model->BSIM4v0rdswminGiven)
            model->BSIM4v0rdswmin = 0.0;
        if (!model->BSIM4v0rdwminGiven)
            model->BSIM4v0rdwmin = 0.0;
        if (!model->BSIM4v0rswminGiven)
            model->BSIM4v0rswmin = 0.0;
        if (!model->BSIM4v0rdswGiven)
	    model->BSIM4v0rdsw = 200.0; /* in ohm*um */     
        if (!model->BSIM4v0rdwGiven)
            model->BSIM4v0rdw = 100.0;
        if (!model->BSIM4v0rswGiven)
            model->BSIM4v0rsw = 100.0;
        if (!model->BSIM4v0prwgGiven)
            model->BSIM4v0prwg = 1.0; /* in 1/V */
        if (!model->BSIM4v0prwbGiven)
            model->BSIM4v0prwb = 0.0;      
        if (!model->BSIM4v0prtGiven)
        if (!model->BSIM4v0prtGiven)
            model->BSIM4v0prt = 0.0;      
        if (!model->BSIM4v0eta0Given)
            model->BSIM4v0eta0 = 0.08;      /* no unit  */ 
        if (!model->BSIM4v0etabGiven)
            model->BSIM4v0etab = -0.07;      /* unit  1/V */ 
        if (!model->BSIM4v0pclmGiven)
            model->BSIM4v0pclm = 1.3;      /* no unit  */ 
        if (!model->BSIM4v0pdibl1Given)
            model->BSIM4v0pdibl1 = 0.39;    /* no unit  */
        if (!model->BSIM4v0pdibl2Given)
            model->BSIM4v0pdibl2 = 0.0086;    /* no unit  */ 
        if (!model->BSIM4v0pdiblbGiven)
            model->BSIM4v0pdiblb = 0.0;    /* 1/V  */ 
        if (!model->BSIM4v0pscbe1Given)
            model->BSIM4v0pscbe1 = 4.24e8;     
        if (!model->BSIM4v0pscbe2Given)
            model->BSIM4v0pscbe2 = 1.0e-5;    
        if (!model->BSIM4v0pvagGiven)
            model->BSIM4v0pvag = 0.0;     
        if (!model->BSIM4v0wrGiven)  
            model->BSIM4v0wr = 1.0;
        if (!model->BSIM4v0dwgGiven)  
            model->BSIM4v0dwg = 0.0;
        if (!model->BSIM4v0dwbGiven)  
            model->BSIM4v0dwb = 0.0;
        if (!model->BSIM4v0b0Given)
            model->BSIM4v0b0 = 0.0;
        if (!model->BSIM4v0b1Given)  
            model->BSIM4v0b1 = 0.0;
        if (!model->BSIM4v0alpha0Given)  
            model->BSIM4v0alpha0 = 0.0;
        if (!model->BSIM4v0alpha1Given)
            model->BSIM4v0alpha1 = 0.0;
        if (!model->BSIM4v0beta0Given)  
            model->BSIM4v0beta0 = 30.0;
        if (!model->BSIM4v0agidlGiven)
            model->BSIM4v0agidl = 0.0;
        if (!model->BSIM4v0bgidlGiven)
            model->BSIM4v0bgidl = 2.3e9; /* V/m */
        if (!model->BSIM4v0cgidlGiven)
            model->BSIM4v0cgidl = 0.5; /* V^3 */
        if (!model->BSIM4v0egidlGiven)
            model->BSIM4v0egidl = 0.8; /* V */
        if (!model->BSIM4v0aigcGiven)
            model->BSIM4v0aigc = (model->BSIM4v0type == NMOS) ? 0.43 : 0.31;
        if (!model->BSIM4v0bigcGiven)
            model->BSIM4v0bigc = (model->BSIM4v0type == NMOS) ? 0.054 : 0.024;
        if (!model->BSIM4v0cigcGiven)
            model->BSIM4v0cigc = (model->BSIM4v0type == NMOS) ? 0.075 : 0.03;
        if (!model->BSIM4v0aigsdGiven)
            model->BSIM4v0aigsd = (model->BSIM4v0type == NMOS) ? 0.43 : 0.31;
        if (!model->BSIM4v0bigsdGiven)
            model->BSIM4v0bigsd = (model->BSIM4v0type == NMOS) ? 0.054 : 0.024;
        if (!model->BSIM4v0cigsdGiven)
            model->BSIM4v0cigsd = (model->BSIM4v0type == NMOS) ? 0.075 : 0.03;
        if (!model->BSIM4v0aigbaccGiven)
            model->BSIM4v0aigbacc = 0.43;
        if (!model->BSIM4v0bigbaccGiven)
            model->BSIM4v0bigbacc = 0.054;
        if (!model->BSIM4v0cigbaccGiven)
            model->BSIM4v0cigbacc = 0.075;
        if (!model->BSIM4v0aigbinvGiven)
            model->BSIM4v0aigbinv = 0.35;
        if (!model->BSIM4v0bigbinvGiven)
            model->BSIM4v0bigbinv = 0.03;
        if (!model->BSIM4v0cigbinvGiven)
            model->BSIM4v0cigbinv = 0.006;
        if (!model->BSIM4v0nigcGiven)
            model->BSIM4v0nigc = 1.0;
        if (!model->BSIM4v0nigbinvGiven)
            model->BSIM4v0nigbinv = 3.0;
        if (!model->BSIM4v0nigbaccGiven)
            model->BSIM4v0nigbacc = 1.0;
        if (!model->BSIM4v0ntoxGiven)
            model->BSIM4v0ntox = 1.0;
        if (!model->BSIM4v0eigbinvGiven)
            model->BSIM4v0eigbinv = 1.1;
        if (!model->BSIM4v0pigcdGiven)
            model->BSIM4v0pigcd = 1.0;
        if (!model->BSIM4v0poxedgeGiven)
            model->BSIM4v0poxedge = 1.0;
        if (!model->BSIM4v0xrcrg1Given)
            model->BSIM4v0xrcrg1 = 12.0;
        if (!model->BSIM4v0xrcrg2Given)
            model->BSIM4v0xrcrg2 = 1.0;
        if (!model->BSIM4v0ijthsfwdGiven)
            model->BSIM4v0ijthsfwd = 0.1; /* unit A */
        if (!model->BSIM4v0ijthdfwdGiven)
            model->BSIM4v0ijthdfwd = model->BSIM4v0ijthsfwd;
        if (!model->BSIM4v0ijthsrevGiven)
            model->BSIM4v0ijthsrev = 0.1; /* unit A */
        if (!model->BSIM4v0ijthdrevGiven)
            model->BSIM4v0ijthdrev = model->BSIM4v0ijthsrev;
        if (!model->BSIM4v0tnoiaGiven)
            model->BSIM4v0tnoia = 1.5;
        if (!model->BSIM4v0tnoibGiven)
            model->BSIM4v0tnoib = 3.5;
        if (!model->BSIM4v0ntnoiGiven)
            model->BSIM4v0ntnoi = 1.0;

        if (!model->BSIM4v0xjbvsGiven)
            model->BSIM4v0xjbvs = 1.0; /* no unit */
        if (!model->BSIM4v0xjbvdGiven)
            model->BSIM4v0xjbvd = model->BSIM4v0xjbvs;
        if (!model->BSIM4v0bvsGiven)
            model->BSIM4v0bvs = 10.0; /* V */
        if (!model->BSIM4v0bvdGiven)
            model->BSIM4v0bvd = model->BSIM4v0bvs;
        if (!model->BSIM4v0gbminGiven)
            model->BSIM4v0gbmin = 1.0e-12; /* in mho */
        if (!model->BSIM4v0rbdbGiven)
            model->BSIM4v0rbdb = 50.0; /* in ohm */
        if (!model->BSIM4v0rbpbGiven)
            model->BSIM4v0rbpb = 50.0;
        if (!model->BSIM4v0rbsbGiven)
            model->BSIM4v0rbsb = 50.0;
        if (!model->BSIM4v0rbpsGiven)
            model->BSIM4v0rbps = 50.0;
        if (!model->BSIM4v0rbpdGiven)
            model->BSIM4v0rbpd = 50.0;

        if (!model->BSIM4v0cgslGiven)  
            model->BSIM4v0cgsl = 0.0;
        if (!model->BSIM4v0cgdlGiven)  
            model->BSIM4v0cgdl = 0.0;
        if (!model->BSIM4v0ckappasGiven)  
            model->BSIM4v0ckappas = 0.6;
        if (!model->BSIM4v0ckappadGiven)
            model->BSIM4v0ckappad = model->BSIM4v0ckappas;
        if (!model->BSIM4v0clcGiven)  
            model->BSIM4v0clc = 0.1e-6;
        if (!model->BSIM4v0cleGiven)  
            model->BSIM4v0cle = 0.6;
        if (!model->BSIM4v0vfbcvGiven)  
            model->BSIM4v0vfbcv = -1.0;
        if (!model->BSIM4v0acdeGiven)
            model->BSIM4v0acde = 1.0;
        if (!model->BSIM4v0moinGiven)
            model->BSIM4v0moin = 15.0;
        if (!model->BSIM4v0noffGiven)
            model->BSIM4v0noff = 1.0;
        if (!model->BSIM4v0voffcvGiven)
            model->BSIM4v0voffcv = 0.0;
        if (!model->BSIM4v0dmcgGiven)
            model->BSIM4v0dmcg = 0.0;
        if (!model->BSIM4v0dmciGiven)
            model->BSIM4v0dmci = model->BSIM4v0dmcg;
        if (!model->BSIM4v0dmdgGiven)
            model->BSIM4v0dmdg = 0.0;
        if (!model->BSIM4v0dmcgtGiven)
            model->BSIM4v0dmcgt = 0.0;
        if (!model->BSIM4v0xgwGiven)
            model->BSIM4v0xgw = 0.0;
        if (!model->BSIM4v0xglGiven)
            model->BSIM4v0xgl = 0.0;
        if (!model->BSIM4v0rshgGiven)
            model->BSIM4v0rshg = 0.1;
        if (!model->BSIM4v0ngconGiven)
            model->BSIM4v0ngcon = 1.0;
        if (!model->BSIM4v0tcjGiven)
            model->BSIM4v0tcj = 0.0;
        if (!model->BSIM4v0tpbGiven)
            model->BSIM4v0tpb = 0.0;
        if (!model->BSIM4v0tcjswGiven)
            model->BSIM4v0tcjsw = 0.0;
        if (!model->BSIM4v0tpbswGiven)
            model->BSIM4v0tpbsw = 0.0;
        if (!model->BSIM4v0tcjswgGiven)
            model->BSIM4v0tcjswg = 0.0;
        if (!model->BSIM4v0tpbswgGiven)
            model->BSIM4v0tpbswg = 0.0;

	/* Length dependence */
        if (!model->BSIM4v0lcdscGiven)
	    model->BSIM4v0lcdsc = 0.0;
        if (!model->BSIM4v0lcdscbGiven)
	    model->BSIM4v0lcdscb = 0.0;
	    if (!model->BSIM4v0lcdscdGiven) 
	    model->BSIM4v0lcdscd = 0.0;
        if (!model->BSIM4v0lcitGiven)
	    model->BSIM4v0lcit = 0.0;
        if (!model->BSIM4v0lnfactorGiven)
	    model->BSIM4v0lnfactor = 0.0;
        if (!model->BSIM4v0lxjGiven)
            model->BSIM4v0lxj = 0.0;
        if (!model->BSIM4v0lvsatGiven)
            model->BSIM4v0lvsat = 0.0;
        if (!model->BSIM4v0latGiven)
            model->BSIM4v0lat = 0.0;
        if (!model->BSIM4v0la0Given)
            model->BSIM4v0la0 = 0.0; 
        if (!model->BSIM4v0lagsGiven)
            model->BSIM4v0lags = 0.0;
        if (!model->BSIM4v0la1Given)
            model->BSIM4v0la1 = 0.0;
        if (!model->BSIM4v0la2Given)
            model->BSIM4v0la2 = 0.0;
        if (!model->BSIM4v0lketaGiven)
            model->BSIM4v0lketa = 0.0;
        if (!model->BSIM4v0lnsubGiven)
            model->BSIM4v0lnsub = 0.0;
        if (!model->BSIM4v0lndepGiven)
            model->BSIM4v0lndep = 0.0;
        if (!model->BSIM4v0lnsdGiven)
            model->BSIM4v0lnsd = 0.0;
        if (!model->BSIM4v0lphinGiven)
            model->BSIM4v0lphin = 0.0;
        if (!model->BSIM4v0lngateGiven)
            model->BSIM4v0lngate = 0.0;
        if (!model->BSIM4v0lvbmGiven)
	    model->BSIM4v0lvbm = 0.0;
        if (!model->BSIM4v0lxtGiven)
	    model->BSIM4v0lxt = 0.0;
        if (!model->BSIM4v0lkt1Given)
            model->BSIM4v0lkt1 = 0.0; 
        if (!model->BSIM4v0lkt1lGiven)
            model->BSIM4v0lkt1l = 0.0;
        if (!model->BSIM4v0lkt2Given)
            model->BSIM4v0lkt2 = 0.0;
        if (!model->BSIM4v0lk3Given)
            model->BSIM4v0lk3 = 0.0;      
        if (!model->BSIM4v0lk3bGiven)
            model->BSIM4v0lk3b = 0.0;      
        if (!model->BSIM4v0lw0Given)
            model->BSIM4v0lw0 = 0.0;    
        if (!model->BSIM4v0llpe0Given)
            model->BSIM4v0llpe0 = 0.0;
        if (!model->BSIM4v0llpebGiven)
            model->BSIM4v0llpeb = model->BSIM4v0llpe0;
        if (!model->BSIM4v0ldvtp0Given)
            model->BSIM4v0ldvtp0 = 0.0;
        if (!model->BSIM4v0ldvtp1Given)
            model->BSIM4v0ldvtp1 = 0.0;
        if (!model->BSIM4v0ldvt0Given)
            model->BSIM4v0ldvt0 = 0.0;    
        if (!model->BSIM4v0ldvt1Given)
            model->BSIM4v0ldvt1 = 0.0;      
        if (!model->BSIM4v0ldvt2Given)
            model->BSIM4v0ldvt2 = 0.0;
        if (!model->BSIM4v0ldvt0wGiven)
            model->BSIM4v0ldvt0w = 0.0;    
        if (!model->BSIM4v0ldvt1wGiven)
            model->BSIM4v0ldvt1w = 0.0;      
        if (!model->BSIM4v0ldvt2wGiven)
            model->BSIM4v0ldvt2w = 0.0;
        if (!model->BSIM4v0ldroutGiven)
            model->BSIM4v0ldrout = 0.0;     
        if (!model->BSIM4v0ldsubGiven)
            model->BSIM4v0ldsub = 0.0;
        if (!model->BSIM4v0lvth0Given)
           model->BSIM4v0lvth0 = 0.0;
        if (!model->BSIM4v0luaGiven)
            model->BSIM4v0lua = 0.0;
        if (!model->BSIM4v0lua1Given)
            model->BSIM4v0lua1 = 0.0;
        if (!model->BSIM4v0lubGiven)
            model->BSIM4v0lub = 0.0;
        if (!model->BSIM4v0lub1Given)
            model->BSIM4v0lub1 = 0.0;
        if (!model->BSIM4v0lucGiven)
            model->BSIM4v0luc = 0.0;
        if (!model->BSIM4v0luc1Given)
            model->BSIM4v0luc1 = 0.0;
        if (!model->BSIM4v0lu0Given)
            model->BSIM4v0lu0 = 0.0;
        if (!model->BSIM4v0luteGiven)
	    model->BSIM4v0lute = 0.0;    
        if (!model->BSIM4v0lvoffGiven)
	    model->BSIM4v0lvoff = 0.0;
        if (!model->BSIM4v0lminvGiven)
            model->BSIM4v0lminv = 0.0;
        if (!model->BSIM4v0lfproutGiven)
            model->BSIM4v0lfprout = 0.0;
        if (!model->BSIM4v0lpditsGiven)
            model->BSIM4v0lpdits = 0.0;
        if (!model->BSIM4v0lpditsdGiven)
            model->BSIM4v0lpditsd = 0.0;
        if (!model->BSIM4v0ldeltaGiven)  
            model->BSIM4v0ldelta = 0.0;
        if (!model->BSIM4v0lrdswGiven)
            model->BSIM4v0lrdsw = 0.0;
        if (!model->BSIM4v0lrdwGiven)
            model->BSIM4v0lrdw = 0.0;
        if (!model->BSIM4v0lrswGiven)
            model->BSIM4v0lrsw = 0.0;
        if (!model->BSIM4v0lprwbGiven)
            model->BSIM4v0lprwb = 0.0;
        if (!model->BSIM4v0lprwgGiven)
            model->BSIM4v0lprwg = 0.0;
        if (!model->BSIM4v0lprtGiven)
            model->BSIM4v0lprt = 0.0;
        if (!model->BSIM4v0leta0Given)
            model->BSIM4v0leta0 = 0.0;
        if (!model->BSIM4v0letabGiven)
            model->BSIM4v0letab = -0.0;
        if (!model->BSIM4v0lpclmGiven)
            model->BSIM4v0lpclm = 0.0; 
        if (!model->BSIM4v0lpdibl1Given)
            model->BSIM4v0lpdibl1 = 0.0;
        if (!model->BSIM4v0lpdibl2Given)
            model->BSIM4v0lpdibl2 = 0.0;
        if (!model->BSIM4v0lpdiblbGiven)
            model->BSIM4v0lpdiblb = 0.0;
        if (!model->BSIM4v0lpscbe1Given)
            model->BSIM4v0lpscbe1 = 0.0;
        if (!model->BSIM4v0lpscbe2Given)
            model->BSIM4v0lpscbe2 = 0.0;
        if (!model->BSIM4v0lpvagGiven)
            model->BSIM4v0lpvag = 0.0;     
        if (!model->BSIM4v0lwrGiven)  
            model->BSIM4v0lwr = 0.0;
        if (!model->BSIM4v0ldwgGiven)  
            model->BSIM4v0ldwg = 0.0;
        if (!model->BSIM4v0ldwbGiven)  
            model->BSIM4v0ldwb = 0.0;
        if (!model->BSIM4v0lb0Given)
            model->BSIM4v0lb0 = 0.0;
        if (!model->BSIM4v0lb1Given)  
            model->BSIM4v0lb1 = 0.0;
        if (!model->BSIM4v0lalpha0Given)  
            model->BSIM4v0lalpha0 = 0.0;
        if (!model->BSIM4v0lalpha1Given)
            model->BSIM4v0lalpha1 = 0.0;
        if (!model->BSIM4v0lbeta0Given)  
            model->BSIM4v0lbeta0 = 0.0;
        if (!model->BSIM4v0lagidlGiven)
            model->BSIM4v0lagidl = 0.0;
        if (!model->BSIM4v0lbgidlGiven)
            model->BSIM4v0lbgidl = 0.0;
        if (!model->BSIM4v0lcgidlGiven)
            model->BSIM4v0lcgidl = 0.0;
        if (!model->BSIM4v0legidlGiven)
            model->BSIM4v0legidl = 0.0;
        if (!model->BSIM4v0laigcGiven)
            model->BSIM4v0laigc = 0.0;
        if (!model->BSIM4v0lbigcGiven)
            model->BSIM4v0lbigc = 0.0;
        if (!model->BSIM4v0lcigcGiven)
            model->BSIM4v0lcigc = 0.0;
        if (!model->BSIM4v0laigsdGiven)
            model->BSIM4v0laigsd = 0.0;
        if (!model->BSIM4v0lbigsdGiven)
            model->BSIM4v0lbigsd = 0.0;
        if (!model->BSIM4v0lcigsdGiven)
            model->BSIM4v0lcigsd = 0.0;
        if (!model->BSIM4v0laigbaccGiven)
            model->BSIM4v0laigbacc = 0.0;
        if (!model->BSIM4v0lbigbaccGiven)
            model->BSIM4v0lbigbacc = 0.0;
        if (!model->BSIM4v0lcigbaccGiven)
            model->BSIM4v0lcigbacc = 0.0;
        if (!model->BSIM4v0laigbinvGiven)
            model->BSIM4v0laigbinv = 0.0;
        if (!model->BSIM4v0lbigbinvGiven)
            model->BSIM4v0lbigbinv = 0.0;
        if (!model->BSIM4v0lcigbinvGiven)
            model->BSIM4v0lcigbinv = 0.0;
        if (!model->BSIM4v0lnigcGiven)
            model->BSIM4v0lnigc = 0.0;
        if (!model->BSIM4v0lnigbinvGiven)
            model->BSIM4v0lnigbinv = 0.0;
        if (!model->BSIM4v0lnigbaccGiven)
            model->BSIM4v0lnigbacc = 0.0;
        if (!model->BSIM4v0lntoxGiven)
            model->BSIM4v0lntox = 0.0;
        if (!model->BSIM4v0leigbinvGiven)
            model->BSIM4v0leigbinv = 0.0;
        if (!model->BSIM4v0lpigcdGiven)
            model->BSIM4v0lpigcd = 0.0;
        if (!model->BSIM4v0lpoxedgeGiven)
            model->BSIM4v0lpoxedge = 0.0;
        if (!model->BSIM4v0lxrcrg1Given)
            model->BSIM4v0lxrcrg1 = 0.0;
        if (!model->BSIM4v0lxrcrg2Given)
            model->BSIM4v0lxrcrg2 = 0.0;
        if (!model->BSIM4v0leuGiven)
            model->BSIM4v0leu = 0.0;
        if (!model->BSIM4v0lvfbGiven)
            model->BSIM4v0lvfb = 0.0;

        if (!model->BSIM4v0lcgslGiven)  
            model->BSIM4v0lcgsl = 0.0;
        if (!model->BSIM4v0lcgdlGiven)  
            model->BSIM4v0lcgdl = 0.0;
        if (!model->BSIM4v0lckappasGiven)  
            model->BSIM4v0lckappas = 0.0;
        if (!model->BSIM4v0lckappadGiven)
            model->BSIM4v0lckappad = 0.0;
        if (!model->BSIM4v0lclcGiven)  
            model->BSIM4v0lclc = 0.0;
        if (!model->BSIM4v0lcleGiven)  
            model->BSIM4v0lcle = 0.0;
        if (!model->BSIM4v0lcfGiven)  
            model->BSIM4v0lcf = 0.0;
        if (!model->BSIM4v0lvfbcvGiven)  
            model->BSIM4v0lvfbcv = 0.0;
        if (!model->BSIM4v0lacdeGiven)
            model->BSIM4v0lacde = 0.0;
        if (!model->BSIM4v0lmoinGiven)
            model->BSIM4v0lmoin = 0.0;
        if (!model->BSIM4v0lnoffGiven)
            model->BSIM4v0lnoff = 0.0;
        if (!model->BSIM4v0lvoffcvGiven)
            model->BSIM4v0lvoffcv = 0.0;

	/* Width dependence */
        if (!model->BSIM4v0wcdscGiven)
	    model->BSIM4v0wcdsc = 0.0;
        if (!model->BSIM4v0wcdscbGiven)
	    model->BSIM4v0wcdscb = 0.0;  
	    if (!model->BSIM4v0wcdscdGiven)
	    model->BSIM4v0wcdscd = 0.0;
        if (!model->BSIM4v0wcitGiven)
	    model->BSIM4v0wcit = 0.0;
        if (!model->BSIM4v0wnfactorGiven)
	    model->BSIM4v0wnfactor = 0.0;
        if (!model->BSIM4v0wxjGiven)
            model->BSIM4v0wxj = 0.0;
        if (!model->BSIM4v0wvsatGiven)
            model->BSIM4v0wvsat = 0.0;
        if (!model->BSIM4v0watGiven)
            model->BSIM4v0wat = 0.0;
        if (!model->BSIM4v0wa0Given)
            model->BSIM4v0wa0 = 0.0; 
        if (!model->BSIM4v0wagsGiven)
            model->BSIM4v0wags = 0.0;
        if (!model->BSIM4v0wa1Given)
            model->BSIM4v0wa1 = 0.0;
        if (!model->BSIM4v0wa2Given)
            model->BSIM4v0wa2 = 0.0;
        if (!model->BSIM4v0wketaGiven)
            model->BSIM4v0wketa = 0.0;
        if (!model->BSIM4v0wnsubGiven)
            model->BSIM4v0wnsub = 0.0;
        if (!model->BSIM4v0wndepGiven)
            model->BSIM4v0wndep = 0.0;
        if (!model->BSIM4v0wnsdGiven)
            model->BSIM4v0wnsd = 0.0;
        if (!model->BSIM4v0wphinGiven)
            model->BSIM4v0wphin = 0.0;
        if (!model->BSIM4v0wngateGiven)
            model->BSIM4v0wngate = 0.0;
        if (!model->BSIM4v0wvbmGiven)
	    model->BSIM4v0wvbm = 0.0;
        if (!model->BSIM4v0wxtGiven)
	    model->BSIM4v0wxt = 0.0;
        if (!model->BSIM4v0wkt1Given)
            model->BSIM4v0wkt1 = 0.0; 
        if (!model->BSIM4v0wkt1lGiven)
            model->BSIM4v0wkt1l = 0.0;
        if (!model->BSIM4v0wkt2Given)
            model->BSIM4v0wkt2 = 0.0;
        if (!model->BSIM4v0wk3Given)
            model->BSIM4v0wk3 = 0.0;      
        if (!model->BSIM4v0wk3bGiven)
            model->BSIM4v0wk3b = 0.0;      
        if (!model->BSIM4v0ww0Given)
            model->BSIM4v0ww0 = 0.0;    
        if (!model->BSIM4v0wlpe0Given)
            model->BSIM4v0wlpe0 = 0.0;
        if (!model->BSIM4v0wlpebGiven)
            model->BSIM4v0wlpeb = model->BSIM4v0wlpe0;
        if (!model->BSIM4v0wdvtp0Given)
            model->BSIM4v0wdvtp0 = 0.0;
        if (!model->BSIM4v0wdvtp1Given)
            model->BSIM4v0wdvtp1 = 0.0;
        if (!model->BSIM4v0wdvt0Given)
            model->BSIM4v0wdvt0 = 0.0;    
        if (!model->BSIM4v0wdvt1Given)
            model->BSIM4v0wdvt1 = 0.0;      
        if (!model->BSIM4v0wdvt2Given)
            model->BSIM4v0wdvt2 = 0.0;
        if (!model->BSIM4v0wdvt0wGiven)
            model->BSIM4v0wdvt0w = 0.0;    
        if (!model->BSIM4v0wdvt1wGiven)
            model->BSIM4v0wdvt1w = 0.0;      
        if (!model->BSIM4v0wdvt2wGiven)
            model->BSIM4v0wdvt2w = 0.0;
        if (!model->BSIM4v0wdroutGiven)
            model->BSIM4v0wdrout = 0.0;     
        if (!model->BSIM4v0wdsubGiven)
            model->BSIM4v0wdsub = 0.0;
        if (!model->BSIM4v0wvth0Given)
           model->BSIM4v0wvth0 = 0.0;
        if (!model->BSIM4v0wuaGiven)
            model->BSIM4v0wua = 0.0;
        if (!model->BSIM4v0wua1Given)
            model->BSIM4v0wua1 = 0.0;
        if (!model->BSIM4v0wubGiven)
            model->BSIM4v0wub = 0.0;
        if (!model->BSIM4v0wub1Given)
            model->BSIM4v0wub1 = 0.0;
        if (!model->BSIM4v0wucGiven)
            model->BSIM4v0wuc = 0.0;
        if (!model->BSIM4v0wuc1Given)
            model->BSIM4v0wuc1 = 0.0;
        if (!model->BSIM4v0wu0Given)
            model->BSIM4v0wu0 = 0.0;
        if (!model->BSIM4v0wuteGiven)
	    model->BSIM4v0wute = 0.0;    
        if (!model->BSIM4v0wvoffGiven)
	    model->BSIM4v0wvoff = 0.0;
        if (!model->BSIM4v0wminvGiven)
            model->BSIM4v0wminv = 0.0;
        if (!model->BSIM4v0wfproutGiven)
            model->BSIM4v0wfprout = 0.0;
        if (!model->BSIM4v0wpditsGiven)
            model->BSIM4v0wpdits = 0.0;
        if (!model->BSIM4v0wpditsdGiven)
            model->BSIM4v0wpditsd = 0.0;
        if (!model->BSIM4v0wdeltaGiven)  
            model->BSIM4v0wdelta = 0.0;
        if (!model->BSIM4v0wrdswGiven)
            model->BSIM4v0wrdsw = 0.0;
        if (!model->BSIM4v0wrdwGiven)
            model->BSIM4v0wrdw = 0.0;
        if (!model->BSIM4v0wrswGiven)
            model->BSIM4v0wrsw = 0.0;
        if (!model->BSIM4v0wprwbGiven)
            model->BSIM4v0wprwb = 0.0;
        if (!model->BSIM4v0wprwgGiven)
            model->BSIM4v0wprwg = 0.0;
        if (!model->BSIM4v0wprtGiven)
            model->BSIM4v0wprt = 0.0;
        if (!model->BSIM4v0weta0Given)
            model->BSIM4v0weta0 = 0.0;
        if (!model->BSIM4v0wetabGiven)
            model->BSIM4v0wetab = 0.0;
        if (!model->BSIM4v0wpclmGiven)
            model->BSIM4v0wpclm = 0.0; 
        if (!model->BSIM4v0wpdibl1Given)
            model->BSIM4v0wpdibl1 = 0.0;
        if (!model->BSIM4v0wpdibl2Given)
            model->BSIM4v0wpdibl2 = 0.0;
        if (!model->BSIM4v0wpdiblbGiven)
            model->BSIM4v0wpdiblb = 0.0;
        if (!model->BSIM4v0wpscbe1Given)
            model->BSIM4v0wpscbe1 = 0.0;
        if (!model->BSIM4v0wpscbe2Given)
            model->BSIM4v0wpscbe2 = 0.0;
        if (!model->BSIM4v0wpvagGiven)
            model->BSIM4v0wpvag = 0.0;     
        if (!model->BSIM4v0wwrGiven)  
            model->BSIM4v0wwr = 0.0;
        if (!model->BSIM4v0wdwgGiven)  
            model->BSIM4v0wdwg = 0.0;
        if (!model->BSIM4v0wdwbGiven)  
            model->BSIM4v0wdwb = 0.0;
        if (!model->BSIM4v0wb0Given)
            model->BSIM4v0wb0 = 0.0;
        if (!model->BSIM4v0wb1Given)  
            model->BSIM4v0wb1 = 0.0;
        if (!model->BSIM4v0walpha0Given)  
            model->BSIM4v0walpha0 = 0.0;
        if (!model->BSIM4v0walpha1Given)
            model->BSIM4v0walpha1 = 0.0;
        if (!model->BSIM4v0wbeta0Given)  
            model->BSIM4v0wbeta0 = 0.0;
        if (!model->BSIM4v0wagidlGiven)
            model->BSIM4v0wagidl = 0.0;
        if (!model->BSIM4v0wbgidlGiven)
            model->BSIM4v0wbgidl = 0.0;
        if (!model->BSIM4v0wcgidlGiven)
            model->BSIM4v0wcgidl = 0.0;
        if (!model->BSIM4v0wegidlGiven)
            model->BSIM4v0wegidl = 0.0;
        if (!model->BSIM4v0waigcGiven)
            model->BSIM4v0waigc = 0.0;
        if (!model->BSIM4v0wbigcGiven)
            model->BSIM4v0wbigc = 0.0;
        if (!model->BSIM4v0wcigcGiven)
            model->BSIM4v0wcigc = 0.0;
        if (!model->BSIM4v0waigsdGiven)
            model->BSIM4v0waigsd = 0.0;
        if (!model->BSIM4v0wbigsdGiven)
            model->BSIM4v0wbigsd = 0.0;
        if (!model->BSIM4v0wcigsdGiven)
            model->BSIM4v0wcigsd = 0.0;
        if (!model->BSIM4v0waigbaccGiven)
            model->BSIM4v0waigbacc = 0.0;
        if (!model->BSIM4v0wbigbaccGiven)
            model->BSIM4v0wbigbacc = 0.0;
        if (!model->BSIM4v0wcigbaccGiven)
            model->BSIM4v0wcigbacc = 0.0;
        if (!model->BSIM4v0waigbinvGiven)
            model->BSIM4v0waigbinv = 0.0;
        if (!model->BSIM4v0wbigbinvGiven)
            model->BSIM4v0wbigbinv = 0.0;
        if (!model->BSIM4v0wcigbinvGiven)
            model->BSIM4v0wcigbinv = 0.0;
        if (!model->BSIM4v0wnigcGiven)
            model->BSIM4v0wnigc = 0.0;
        if (!model->BSIM4v0wnigbinvGiven)
            model->BSIM4v0wnigbinv = 0.0;
        if (!model->BSIM4v0wnigbaccGiven)
            model->BSIM4v0wnigbacc = 0.0;
        if (!model->BSIM4v0wntoxGiven)
            model->BSIM4v0wntox = 0.0;
        if (!model->BSIM4v0weigbinvGiven)
            model->BSIM4v0weigbinv = 0.0;
        if (!model->BSIM4v0wpigcdGiven)
            model->BSIM4v0wpigcd = 0.0;
        if (!model->BSIM4v0wpoxedgeGiven)
            model->BSIM4v0wpoxedge = 0.0;
        if (!model->BSIM4v0wxrcrg1Given)
            model->BSIM4v0wxrcrg1 = 0.0;
        if (!model->BSIM4v0wxrcrg2Given)
            model->BSIM4v0wxrcrg2 = 0.0;
        if (!model->BSIM4v0weuGiven)
            model->BSIM4v0weu = 0.0;
        if (!model->BSIM4v0wvfbGiven)
            model->BSIM4v0wvfb = 0.0;

        if (!model->BSIM4v0wcgslGiven)  
            model->BSIM4v0wcgsl = 0.0;
        if (!model->BSIM4v0wcgdlGiven)  
            model->BSIM4v0wcgdl = 0.0;
        if (!model->BSIM4v0wckappasGiven)  
            model->BSIM4v0wckappas = 0.0;
        if (!model->BSIM4v0wckappadGiven)
            model->BSIM4v0wckappad = 0.0;
        if (!model->BSIM4v0wcfGiven)  
            model->BSIM4v0wcf = 0.0;
        if (!model->BSIM4v0wclcGiven)  
            model->BSIM4v0wclc = 0.0;
        if (!model->BSIM4v0wcleGiven)  
            model->BSIM4v0wcle = 0.0;
        if (!model->BSIM4v0wvfbcvGiven)  
            model->BSIM4v0wvfbcv = 0.0;
        if (!model->BSIM4v0wacdeGiven)
            model->BSIM4v0wacde = 0.0;
        if (!model->BSIM4v0wmoinGiven)
            model->BSIM4v0wmoin = 0.0;
        if (!model->BSIM4v0wnoffGiven)
            model->BSIM4v0wnoff = 0.0;
        if (!model->BSIM4v0wvoffcvGiven)
            model->BSIM4v0wvoffcv = 0.0;

	/* Cross-term dependence */
        if (!model->BSIM4v0pcdscGiven)
	    model->BSIM4v0pcdsc = 0.0;
        if (!model->BSIM4v0pcdscbGiven)
	    model->BSIM4v0pcdscb = 0.0;   
	    if (!model->BSIM4v0pcdscdGiven)
	    model->BSIM4v0pcdscd = 0.0;
        if (!model->BSIM4v0pcitGiven)
	    model->BSIM4v0pcit = 0.0;
        if (!model->BSIM4v0pnfactorGiven)
	    model->BSIM4v0pnfactor = 0.0;
        if (!model->BSIM4v0pxjGiven)
            model->BSIM4v0pxj = 0.0;
        if (!model->BSIM4v0pvsatGiven)
            model->BSIM4v0pvsat = 0.0;
        if (!model->BSIM4v0patGiven)
            model->BSIM4v0pat = 0.0;
        if (!model->BSIM4v0pa0Given)
            model->BSIM4v0pa0 = 0.0; 
            
        if (!model->BSIM4v0pagsGiven)
            model->BSIM4v0pags = 0.0;
        if (!model->BSIM4v0pa1Given)
            model->BSIM4v0pa1 = 0.0;
        if (!model->BSIM4v0pa2Given)
            model->BSIM4v0pa2 = 0.0;
        if (!model->BSIM4v0pketaGiven)
            model->BSIM4v0pketa = 0.0;
        if (!model->BSIM4v0pnsubGiven)
            model->BSIM4v0pnsub = 0.0;
        if (!model->BSIM4v0pndepGiven)
            model->BSIM4v0pndep = 0.0;
        if (!model->BSIM4v0pnsdGiven)
            model->BSIM4v0pnsd = 0.0;
        if (!model->BSIM4v0pphinGiven)
            model->BSIM4v0pphin = 0.0;
        if (!model->BSIM4v0pngateGiven)
            model->BSIM4v0pngate = 0.0;
        if (!model->BSIM4v0pvbmGiven)
	    model->BSIM4v0pvbm = 0.0;
        if (!model->BSIM4v0pxtGiven)
	    model->BSIM4v0pxt = 0.0;
        if (!model->BSIM4v0pkt1Given)
            model->BSIM4v0pkt1 = 0.0; 
        if (!model->BSIM4v0pkt1lGiven)
            model->BSIM4v0pkt1l = 0.0;
        if (!model->BSIM4v0pkt2Given)
            model->BSIM4v0pkt2 = 0.0;
        if (!model->BSIM4v0pk3Given)
            model->BSIM4v0pk3 = 0.0;      
        if (!model->BSIM4v0pk3bGiven)
            model->BSIM4v0pk3b = 0.0;      
        if (!model->BSIM4v0pw0Given)
            model->BSIM4v0pw0 = 0.0;    
        if (!model->BSIM4v0plpe0Given)
            model->BSIM4v0plpe0 = 0.0;
        if (!model->BSIM4v0plpebGiven)
            model->BSIM4v0plpeb = model->BSIM4v0plpe0;
        if (!model->BSIM4v0pdvtp0Given)
            model->BSIM4v0pdvtp0 = 0.0;
        if (!model->BSIM4v0pdvtp1Given)
            model->BSIM4v0pdvtp1 = 0.0;
        if (!model->BSIM4v0pdvt0Given)
            model->BSIM4v0pdvt0 = 0.0;    
        if (!model->BSIM4v0pdvt1Given)
            model->BSIM4v0pdvt1 = 0.0;      
        if (!model->BSIM4v0pdvt2Given)
            model->BSIM4v0pdvt2 = 0.0;
        if (!model->BSIM4v0pdvt0wGiven)
            model->BSIM4v0pdvt0w = 0.0;    
        if (!model->BSIM4v0pdvt1wGiven)
            model->BSIM4v0pdvt1w = 0.0;      
        if (!model->BSIM4v0pdvt2wGiven)
            model->BSIM4v0pdvt2w = 0.0;
        if (!model->BSIM4v0pdroutGiven)
            model->BSIM4v0pdrout = 0.0;     
        if (!model->BSIM4v0pdsubGiven)
            model->BSIM4v0pdsub = 0.0;
        if (!model->BSIM4v0pvth0Given)
           model->BSIM4v0pvth0 = 0.0;
        if (!model->BSIM4v0puaGiven)
            model->BSIM4v0pua = 0.0;
        if (!model->BSIM4v0pua1Given)
            model->BSIM4v0pua1 = 0.0;
        if (!model->BSIM4v0pubGiven)
            model->BSIM4v0pub = 0.0;
        if (!model->BSIM4v0pub1Given)
            model->BSIM4v0pub1 = 0.0;
        if (!model->BSIM4v0pucGiven)
            model->BSIM4v0puc = 0.0;
        if (!model->BSIM4v0puc1Given)
            model->BSIM4v0puc1 = 0.0;
        if (!model->BSIM4v0pu0Given)
            model->BSIM4v0pu0 = 0.0;
        if (!model->BSIM4v0puteGiven)
	    model->BSIM4v0pute = 0.0;    
        if (!model->BSIM4v0pvoffGiven)
	    model->BSIM4v0pvoff = 0.0;
        if (!model->BSIM4v0pminvGiven)
            model->BSIM4v0pminv = 0.0;
        if (!model->BSIM4v0pfproutGiven)
            model->BSIM4v0pfprout = 0.0;
        if (!model->BSIM4v0ppditsGiven)
            model->BSIM4v0ppdits = 0.0;
        if (!model->BSIM4v0ppditsdGiven)
            model->BSIM4v0ppditsd = 0.0;
        if (!model->BSIM4v0pdeltaGiven)  
            model->BSIM4v0pdelta = 0.0;
        if (!model->BSIM4v0prdswGiven)
            model->BSIM4v0prdsw = 0.0;
        if (!model->BSIM4v0prdwGiven)
            model->BSIM4v0prdw = 0.0;
        if (!model->BSIM4v0prswGiven)
            model->BSIM4v0prsw = 0.0;
        if (!model->BSIM4v0pprwbGiven)
            model->BSIM4v0pprwb = 0.0;
        if (!model->BSIM4v0pprwgGiven)
            model->BSIM4v0pprwg = 0.0;
        if (!model->BSIM4v0pprtGiven)
            model->BSIM4v0pprt = 0.0;
        if (!model->BSIM4v0peta0Given)
            model->BSIM4v0peta0 = 0.0;
        if (!model->BSIM4v0petabGiven)
            model->BSIM4v0petab = 0.0;
        if (!model->BSIM4v0ppclmGiven)
            model->BSIM4v0ppclm = 0.0; 
        if (!model->BSIM4v0ppdibl1Given)
            model->BSIM4v0ppdibl1 = 0.0;
        if (!model->BSIM4v0ppdibl2Given)
            model->BSIM4v0ppdibl2 = 0.0;
        if (!model->BSIM4v0ppdiblbGiven)
            model->BSIM4v0ppdiblb = 0.0;
        if (!model->BSIM4v0ppscbe1Given)
            model->BSIM4v0ppscbe1 = 0.0;
        if (!model->BSIM4v0ppscbe2Given)
            model->BSIM4v0ppscbe2 = 0.0;
        if (!model->BSIM4v0ppvagGiven)
            model->BSIM4v0ppvag = 0.0;     
        if (!model->BSIM4v0pwrGiven)  
            model->BSIM4v0pwr = 0.0;
        if (!model->BSIM4v0pdwgGiven)  
            model->BSIM4v0pdwg = 0.0;
        if (!model->BSIM4v0pdwbGiven)  
            model->BSIM4v0pdwb = 0.0;
        if (!model->BSIM4v0pb0Given)
            model->BSIM4v0pb0 = 0.0;
        if (!model->BSIM4v0pb1Given)  
            model->BSIM4v0pb1 = 0.0;
        if (!model->BSIM4v0palpha0Given)  
            model->BSIM4v0palpha0 = 0.0;
        if (!model->BSIM4v0palpha1Given)
            model->BSIM4v0palpha1 = 0.0;
        if (!model->BSIM4v0pbeta0Given)  
            model->BSIM4v0pbeta0 = 0.0;
        if (!model->BSIM4v0pagidlGiven)
            model->BSIM4v0pagidl = 0.0;
        if (!model->BSIM4v0pbgidlGiven)
            model->BSIM4v0pbgidl = 0.0;
        if (!model->BSIM4v0pcgidlGiven)
            model->BSIM4v0pcgidl = 0.0;
        if (!model->BSIM4v0pegidlGiven)
            model->BSIM4v0pegidl = 0.0;
        if (!model->BSIM4v0paigcGiven)
            model->BSIM4v0paigc = 0.0;
        if (!model->BSIM4v0pbigcGiven)
            model->BSIM4v0pbigc = 0.0;
        if (!model->BSIM4v0pcigcGiven)
            model->BSIM4v0pcigc = 0.0;
        if (!model->BSIM4v0paigsdGiven)
            model->BSIM4v0paigsd = 0.0;
        if (!model->BSIM4v0pbigsdGiven)
            model->BSIM4v0pbigsd = 0.0;
        if (!model->BSIM4v0pcigsdGiven)
            model->BSIM4v0pcigsd = 0.0;
        if (!model->BSIM4v0paigbaccGiven)
            model->BSIM4v0paigbacc = 0.0;
        if (!model->BSIM4v0pbigbaccGiven)
            model->BSIM4v0pbigbacc = 0.0;
        if (!model->BSIM4v0pcigbaccGiven)
            model->BSIM4v0pcigbacc = 0.0;
        if (!model->BSIM4v0paigbinvGiven)
            model->BSIM4v0paigbinv = 0.0;
        if (!model->BSIM4v0pbigbinvGiven)
            model->BSIM4v0pbigbinv = 0.0;
        if (!model->BSIM4v0pcigbinvGiven)
            model->BSIM4v0pcigbinv = 0.0;
        if (!model->BSIM4v0pnigcGiven)
            model->BSIM4v0pnigc = 0.0;
        if (!model->BSIM4v0pnigbinvGiven)
            model->BSIM4v0pnigbinv = 0.0;
        if (!model->BSIM4v0pnigbaccGiven)
            model->BSIM4v0pnigbacc = 0.0;
        if (!model->BSIM4v0pntoxGiven)
            model->BSIM4v0pntox = 0.0;
        if (!model->BSIM4v0peigbinvGiven)
            model->BSIM4v0peigbinv = 0.0;
        if (!model->BSIM4v0ppigcdGiven)
            model->BSIM4v0ppigcd = 0.0;
        if (!model->BSIM4v0ppoxedgeGiven)
            model->BSIM4v0ppoxedge = 0.0;
        if (!model->BSIM4v0pxrcrg1Given)
            model->BSIM4v0pxrcrg1 = 0.0;
        if (!model->BSIM4v0pxrcrg2Given)
            model->BSIM4v0pxrcrg2 = 0.0;
        if (!model->BSIM4v0peuGiven)
            model->BSIM4v0peu = 0.0;
        if (!model->BSIM4v0pvfbGiven)
            model->BSIM4v0pvfb = 0.0;

        if (!model->BSIM4v0pcgslGiven)  
            model->BSIM4v0pcgsl = 0.0;
        if (!model->BSIM4v0pcgdlGiven)  
            model->BSIM4v0pcgdl = 0.0;
        if (!model->BSIM4v0pckappasGiven)  
            model->BSIM4v0pckappas = 0.0;
        if (!model->BSIM4v0pckappadGiven)
            model->BSIM4v0pckappad = 0.0;
        if (!model->BSIM4v0pcfGiven)  
            model->BSIM4v0pcf = 0.0;
        if (!model->BSIM4v0pclcGiven)  
            model->BSIM4v0pclc = 0.0;
        if (!model->BSIM4v0pcleGiven)  
            model->BSIM4v0pcle = 0.0;
        if (!model->BSIM4v0pvfbcvGiven)  
            model->BSIM4v0pvfbcv = 0.0;
        if (!model->BSIM4v0pacdeGiven)
            model->BSIM4v0pacde = 0.0;
        if (!model->BSIM4v0pmoinGiven)
            model->BSIM4v0pmoin = 0.0;
        if (!model->BSIM4v0pnoffGiven)
            model->BSIM4v0pnoff = 0.0;
        if (!model->BSIM4v0pvoffcvGiven)
            model->BSIM4v0pvoffcv = 0.0;

        /* unit degree celcius */
        if (!model->BSIM4v0tnomGiven)  
	    model->BSIM4v0tnom = ckt->CKTnomTemp; 
        if (!model->BSIM4v0LintGiven)  
           model->BSIM4v0Lint = 0.0;
        if (!model->BSIM4v0LlGiven)  
           model->BSIM4v0Ll = 0.0;
        if (!model->BSIM4v0LlcGiven)
           model->BSIM4v0Llc = model->BSIM4v0Ll;
        if (!model->BSIM4v0LlnGiven)  
           model->BSIM4v0Lln = 1.0;
        if (!model->BSIM4v0LwGiven)  
           model->BSIM4v0Lw = 0.0;
        if (!model->BSIM4v0LwcGiven)
           model->BSIM4v0Lwc = model->BSIM4v0Lw;
        if (!model->BSIM4v0LwnGiven)  
           model->BSIM4v0Lwn = 1.0;
        if (!model->BSIM4v0LwlGiven)  
           model->BSIM4v0Lwl = 0.0;
        if (!model->BSIM4v0LwlcGiven)
           model->BSIM4v0Lwlc = model->BSIM4v0Lwl;
        if (!model->BSIM4v0LminGiven)  
           model->BSIM4v0Lmin = 0.0;
        if (!model->BSIM4v0LmaxGiven)  
           model->BSIM4v0Lmax = 1.0;
        if (!model->BSIM4v0WintGiven)  
           model->BSIM4v0Wint = 0.0;
        if (!model->BSIM4v0WlGiven)  
           model->BSIM4v0Wl = 0.0;
        if (!model->BSIM4v0WlcGiven)
           model->BSIM4v0Wlc = model->BSIM4v0Wl;
        if (!model->BSIM4v0WlnGiven)  
           model->BSIM4v0Wln = 1.0;
        if (!model->BSIM4v0WwGiven)  
           model->BSIM4v0Ww = 0.0;
        if (!model->BSIM4v0WwcGiven)
           model->BSIM4v0Wwc = model->BSIM4v0Ww;
        if (!model->BSIM4v0WwnGiven)  
           model->BSIM4v0Wwn = 1.0;
        if (!model->BSIM4v0WwlGiven)  
           model->BSIM4v0Wwl = 0.0;
        if (!model->BSIM4v0WwlcGiven)
           model->BSIM4v0Wwlc = model->BSIM4v0Wwl;
        if (!model->BSIM4v0WminGiven)  
           model->BSIM4v0Wmin = 0.0;
        if (!model->BSIM4v0WmaxGiven)  
           model->BSIM4v0Wmax = 1.0;
        if (!model->BSIM4v0dwcGiven)  
           model->BSIM4v0dwc = model->BSIM4v0Wint;
        if (!model->BSIM4v0dlcGiven)  
           model->BSIM4v0dlc = model->BSIM4v0Lint;
        if (!model->BSIM4v0dlcigGiven)
           model->BSIM4v0dlcig = model->BSIM4v0Lint;
        if (!model->BSIM4v0dwjGiven)
           model->BSIM4v0dwj = model->BSIM4v0dwc;
	if (!model->BSIM4v0cfGiven)
           model->BSIM4v0cf = 2.0 * model->BSIM4v0epsrox * EPS0 / PI
		          * log(1.0 + 0.4e-6 / model->BSIM4v0toxe);

        if (!model->BSIM4v0xpartGiven)
            model->BSIM4v0xpart = 0.0;
        if (!model->BSIM4v0sheetResistanceGiven)
            model->BSIM4v0sheetResistance = 0.0;

        if (!model->BSIM4v0SunitAreaJctCapGiven)
            model->BSIM4v0SunitAreaJctCap = 5.0E-4;
        if (!model->BSIM4v0DunitAreaJctCapGiven)
            model->BSIM4v0DunitAreaJctCap = model->BSIM4v0SunitAreaJctCap;
        if (!model->BSIM4v0SunitLengthSidewallJctCapGiven)
            model->BSIM4v0SunitLengthSidewallJctCap = 5.0E-10;
        if (!model->BSIM4v0DunitLengthSidewallJctCapGiven)
            model->BSIM4v0DunitLengthSidewallJctCap = model->BSIM4v0SunitLengthSidewallJctCap;
        if (!model->BSIM4v0SunitLengthGateSidewallJctCapGiven)
            model->BSIM4v0SunitLengthGateSidewallJctCap = model->BSIM4v0SunitLengthSidewallJctCap ;
        if (!model->BSIM4v0DunitLengthGateSidewallJctCapGiven)
            model->BSIM4v0DunitLengthGateSidewallJctCap = model->BSIM4v0SunitLengthGateSidewallJctCap;
        if (!model->BSIM4v0SjctSatCurDensityGiven)
            model->BSIM4v0SjctSatCurDensity = 1.0E-4;
        if (!model->BSIM4v0DjctSatCurDensityGiven)
            model->BSIM4v0DjctSatCurDensity = model->BSIM4v0SjctSatCurDensity;
        if (!model->BSIM4v0SjctSidewallSatCurDensityGiven)
            model->BSIM4v0SjctSidewallSatCurDensity = 0.0;
        if (!model->BSIM4v0DjctSidewallSatCurDensityGiven)
            model->BSIM4v0DjctSidewallSatCurDensity = model->BSIM4v0SjctSidewallSatCurDensity;
        if (!model->BSIM4v0SjctGateSidewallSatCurDensityGiven)
            model->BSIM4v0SjctGateSidewallSatCurDensity = 0.0;
        if (!model->BSIM4v0DjctGateSidewallSatCurDensityGiven)
            model->BSIM4v0DjctGateSidewallSatCurDensity = model->BSIM4v0SjctGateSidewallSatCurDensity;
        if (!model->BSIM4v0SbulkJctPotentialGiven)
            model->BSIM4v0SbulkJctPotential = 1.0;
        if (!model->BSIM4v0DbulkJctPotentialGiven)
            model->BSIM4v0DbulkJctPotential = model->BSIM4v0SbulkJctPotential;
        if (!model->BSIM4v0SsidewallJctPotentialGiven)
            model->BSIM4v0SsidewallJctPotential = 1.0;
        if (!model->BSIM4v0DsidewallJctPotentialGiven)
            model->BSIM4v0DsidewallJctPotential = model->BSIM4v0SsidewallJctPotential;
        if (!model->BSIM4v0SGatesidewallJctPotentialGiven)
            model->BSIM4v0SGatesidewallJctPotential = model->BSIM4v0SsidewallJctPotential;
        if (!model->BSIM4v0DGatesidewallJctPotentialGiven)
            model->BSIM4v0DGatesidewallJctPotential = model->BSIM4v0SGatesidewallJctPotential;
        if (!model->BSIM4v0SbulkJctBotGradingCoeffGiven)
            model->BSIM4v0SbulkJctBotGradingCoeff = 0.5;
        if (!model->BSIM4v0DbulkJctBotGradingCoeffGiven)
            model->BSIM4v0DbulkJctBotGradingCoeff = model->BSIM4v0SbulkJctBotGradingCoeff;
        if (!model->BSIM4v0SbulkJctSideGradingCoeffGiven)
            model->BSIM4v0SbulkJctSideGradingCoeff = 0.33;
        if (!model->BSIM4v0DbulkJctSideGradingCoeffGiven)
            model->BSIM4v0DbulkJctSideGradingCoeff = model->BSIM4v0SbulkJctSideGradingCoeff;
        if (!model->BSIM4v0SbulkJctGateSideGradingCoeffGiven)
            model->BSIM4v0SbulkJctGateSideGradingCoeff = model->BSIM4v0SbulkJctSideGradingCoeff;
        if (!model->BSIM4v0DbulkJctGateSideGradingCoeffGiven)
            model->BSIM4v0DbulkJctGateSideGradingCoeff = model->BSIM4v0SbulkJctGateSideGradingCoeff;
        if (!model->BSIM4v0SjctEmissionCoeffGiven)
            model->BSIM4v0SjctEmissionCoeff = 1.0;
        if (!model->BSIM4v0DjctEmissionCoeffGiven)
            model->BSIM4v0DjctEmissionCoeff = model->BSIM4v0SjctEmissionCoeff;
        if (!model->BSIM4v0SjctTempExponentGiven)
            model->BSIM4v0SjctTempExponent = 3.0;
        if (!model->BSIM4v0DjctTempExponentGiven)
            model->BSIM4v0DjctTempExponent = model->BSIM4v0SjctTempExponent;

        if (!model->BSIM4v0oxideTrapDensityAGiven)
	{   if (model->BSIM4v0type == NMOS)
                model->BSIM4v0oxideTrapDensityA = 6.25e41;
            else
                model->BSIM4v0oxideTrapDensityA= 6.188e40;
	}
        if (!model->BSIM4v0oxideTrapDensityBGiven)
	{   if (model->BSIM4v0type == NMOS)
                model->BSIM4v0oxideTrapDensityB = 3.125e26;
            else
                model->BSIM4v0oxideTrapDensityB = 1.5e25;
	}
        if (!model->BSIM4v0oxideTrapDensityCGiven)
            model->BSIM4v0oxideTrapDensityC = 8.75e9;
        if (!model->BSIM4v0emGiven)
            model->BSIM4v0em = 4.1e7; /* V/m */
        if (!model->BSIM4v0efGiven)
            model->BSIM4v0ef = 1.0;
        if (!model->BSIM4v0afGiven)
            model->BSIM4v0af = 1.0;
        if (!model->BSIM4v0kfGiven)
            model->BSIM4v0kf = 0.0;

	/*
         * End processing models and begin to loop
         * through all the instances of the model
         */

        for (here = model->BSIM4v0instances; here != NULL ;
             here=here->BSIM4v0nextInstance) 
	{   /* allocate a chunk of the state vector */
            here->BSIM4v0states = *states;
            *states += BSIM4v0numStates;
            /* perform the parameter defaulting */
            if (!here->BSIM4v0lGiven)
                here->BSIM4v0l = 5.0e-6;
            if (!here->BSIM4v0wGiven)
                here->BSIM4v0w = 5.0e-6;
            if (!here->BSIM4v0nfGiven)
                here->BSIM4v0nf = 1.0;
            if (!here->BSIM4v0minGiven)
                here->BSIM4v0min = 0; /* integer */
            if (!here->BSIM4v0icVDSGiven)
                here->BSIM4v0icVDS = 0.0;
            if (!here->BSIM4v0icVGSGiven)
                here->BSIM4v0icVGS = 0.0;
            if (!here->BSIM4v0icVBSGiven)
                here->BSIM4v0icVBS = 0.0;
            if (!here->BSIM4v0drainAreaGiven)
                here->BSIM4v0drainArea = 0.0;
            if (!here->BSIM4v0drainPerimeterGiven)
                here->BSIM4v0drainPerimeter = 0.0;
            if (!here->BSIM4v0drainSquaresGiven)
                here->BSIM4v0drainSquares = 1.0;
            if (!here->BSIM4v0sourceAreaGiven)
                here->BSIM4v0sourceArea = 0.0;
            if (!here->BSIM4v0sourcePerimeterGiven)
                here->BSIM4v0sourcePerimeter = 0.0;
            if (!here->BSIM4v0sourceSquaresGiven)
                here->BSIM4v0sourceSquares = 1.0;

            if (!here->BSIM4v0rbdbGiven)
                here->BSIM4v0rbdb = model->BSIM4v0rbdb; /* in ohm */
            if (!here->BSIM4v0rbsbGiven)
                here->BSIM4v0rbsb = model->BSIM4v0rbsb;
            if (!here->BSIM4v0rbpbGiven)
                here->BSIM4v0rbpb = model->BSIM4v0rbpb;
            if (!here->BSIM4v0rbpsGiven)
                here->BSIM4v0rbps = model->BSIM4v0rbps;
            if (!here->BSIM4v0rbpdGiven)
                here->BSIM4v0rbpd = model->BSIM4v0rbpd;

                    
            /* Process instance model selectors, some
             * may override their global counterparts
	     */
            if (!here->BSIM4v0rbodyModGiven)
                here->BSIM4v0rbodyMod = model->BSIM4v0rbodyMod;
            else if ((here->BSIM4v0rbodyMod != 0) && (here->BSIM4v0rbodyMod != 1))
            {   here->BSIM4v0rbodyMod = model->BSIM4v0rbodyMod;
                printf("Warning: rbodyMod has been set to its global value %d.\n",
	        model->BSIM4v0rbodyMod);
            }

            if (!here->BSIM4v0rgateModGiven)
                here->BSIM4v0rgateMod = model->BSIM4v0rgateMod;
            else if ((here->BSIM4v0rgateMod != 0) && (here->BSIM4v0rgateMod != 1)
	        && (here->BSIM4v0rgateMod != 2) && (here->BSIM4v0rgateMod != 3))
            {   here->BSIM4v0rgateMod = model->BSIM4v0rgateMod;
                printf("Warning: rgateMod has been set to its global value %d.\n",
                model->BSIM4v0rgateMod);
            }

            if (!here->BSIM4v0geoModGiven)
                here->BSIM4v0geoMod = model->BSIM4v0geoMod;
            if (!here->BSIM4v0rgeoModGiven)
                here->BSIM4v0rgeoMod = 0.0;
            if (!here->BSIM4v0trnqsModGiven)
                here->BSIM4v0trnqsMod = model->BSIM4v0trnqsMod;
            else if ((here->BSIM4v0trnqsMod != 0) && (here->BSIM4v0trnqsMod != 1))
            {   here->BSIM4v0trnqsMod = model->BSIM4v0trnqsMod;
                printf("Warning: trnqsMod has been set to its global value %d.\n",
                model->BSIM4v0trnqsMod);
            }

            if (!here->BSIM4v0acnqsModGiven)
                here->BSIM4v0acnqsMod = model->BSIM4v0acnqsMod;
            else if ((here->BSIM4v0acnqsMod != 0) && (here->BSIM4v0acnqsMod != 1))
            {   here->BSIM4v0acnqsMod = model->BSIM4v0acnqsMod;
                printf("Warning: acnqsMod has been set to its global value %d.\n",
                model->BSIM4v0acnqsMod);
            }

            /* process drain series resistance */
            if (((here->BSIM4v0rgeoMod != 0) || (model->BSIM4v0rdsMod != 0)
                || (model->BSIM4v0tnoiMod != 0)) && (here->BSIM4v0dNodePrime == 0))
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v0name,"drain");
                if(error) return(error);
                here->BSIM4v0dNodePrime = tmp->number;
            }
            else
            {   here->BSIM4v0dNodePrime = here->BSIM4v0dNode;
            }
            /* process source series resistance */
            if (((here->BSIM4v0rgeoMod != 0) || (model->BSIM4v0rdsMod != 0)
		|| (model->BSIM4v0tnoiMod != 0)) && (here->BSIM4v0sNodePrime == 0))
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v0name,"source");
                if(error) return(error);
                here->BSIM4v0sNodePrime = tmp->number;
            }
            else
                here->BSIM4v0sNodePrime = here->BSIM4v0sNode;
            

            if ((here->BSIM4v0rgateMod > 0) && (here->BSIM4v0gNodePrime == 0))
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v0name,"gate");
                if(error) return(error);
                   here->BSIM4v0gNodePrime = tmp->number;
                if (here->BSIM4v0rgateMod == 1)
                {   if (model->BSIM4v0rshg <= 0.0)
	                printf("Warning: rshg should be positive for rgateMod = 1.\n");
		}
                else if (here->BSIM4v0rgateMod == 2)
                {   if (model->BSIM4v0rshg <= 0.0)
                        printf("Warning: rshg <= 0.0 for rgateMod = 2!!!\n");
                    else if (model->BSIM4v0xrcrg1 <= 0.0)
                        printf("Warning: xrcrg1 <= 0.0 for rgateMod = 2!!!\n");
                }
            }
            else
                here->BSIM4v0gNodePrime = here->BSIM4v0gNodeExt;

            if ((here->BSIM4v0rgateMod == 3) && (here->BSIM4v0gNodeMid == 0))
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v0name,"midgate");
                if(error) return(error);
                   here->BSIM4v0gNodeMid = tmp->number;
                if (model->BSIM4v0rshg <= 0.0)
                    printf("Warning: rshg should be positive for rgateMod = 3.\n");
                else if (model->BSIM4v0xrcrg1 <= 0.0)
                    printf("Warning: xrcrg1 should be positive for rgateMod = 3.\n");
            }
            else
                here->BSIM4v0gNodeMid = here->BSIM4v0gNodeExt;
            

            /* internal body nodes for body resistance model */
            if (here->BSIM4v0rbodyMod)
            {   if (here->BSIM4v0dbNode == 0)
		{   error = CKTmkVolt(ckt,&tmp,here->BSIM4v0name,"dbody");
                    if(error) return(error);
                    here->BSIM4v0dbNode = tmp->number;
		}
		if (here->BSIM4v0bNodePrime == 0)
                {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v0name,"body");
                    if(error) return(error);
                    here->BSIM4v0bNodePrime = tmp->number;
                }
		if (here->BSIM4v0sbNode == 0)
                {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v0name,"sbody");
                    if(error) return(error);
                    here->BSIM4v0sbNode = tmp->number;
                }
            }
	    else
	        here->BSIM4v0dbNode = here->BSIM4v0bNodePrime = here->BSIM4v0sbNode
				  = here->BSIM4v0bNode;

            /* NQS node */
            if ((here->BSIM4v0trnqsMod) && (here->BSIM4v0qNode == 0)) 
	    {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v0name,"charge");
                if(error) return(error);
                here->BSIM4v0qNode = tmp->number;
            }
	    else 
	        here->BSIM4v0qNode = 0;

/* set Sparse Matrix Pointers 
 * macro to make elements with built-in out-of-memory test */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==(double *)NULL){\
    return(E_NOMEM);\
}

            TSTALLOC(BSIM4v0DPbpPtr, BSIM4v0dNodePrime, BSIM4v0bNodePrime)
            TSTALLOC(BSIM4v0GPbpPtr, BSIM4v0gNodePrime, BSIM4v0bNodePrime)
            TSTALLOC(BSIM4v0SPbpPtr, BSIM4v0sNodePrime, BSIM4v0bNodePrime)

            TSTALLOC(BSIM4v0BPdpPtr, BSIM4v0bNodePrime, BSIM4v0dNodePrime)
            TSTALLOC(BSIM4v0BPgpPtr, BSIM4v0bNodePrime, BSIM4v0gNodePrime)
            TSTALLOC(BSIM4v0BPspPtr, BSIM4v0bNodePrime, BSIM4v0sNodePrime)
            TSTALLOC(BSIM4v0BPbpPtr, BSIM4v0bNodePrime, BSIM4v0bNodePrime)

            TSTALLOC(BSIM4v0DdPtr, BSIM4v0dNode, BSIM4v0dNode)
            TSTALLOC(BSIM4v0GPgpPtr, BSIM4v0gNodePrime, BSIM4v0gNodePrime)
            TSTALLOC(BSIM4v0SsPtr, BSIM4v0sNode, BSIM4v0sNode)
            TSTALLOC(BSIM4v0DPdpPtr, BSIM4v0dNodePrime, BSIM4v0dNodePrime)
            TSTALLOC(BSIM4v0SPspPtr, BSIM4v0sNodePrime, BSIM4v0sNodePrime)
            TSTALLOC(BSIM4v0DdpPtr, BSIM4v0dNode, BSIM4v0dNodePrime)
            TSTALLOC(BSIM4v0GPdpPtr, BSIM4v0gNodePrime, BSIM4v0dNodePrime)
            TSTALLOC(BSIM4v0GPspPtr, BSIM4v0gNodePrime, BSIM4v0sNodePrime)
            TSTALLOC(BSIM4v0SspPtr, BSIM4v0sNode, BSIM4v0sNodePrime)
            TSTALLOC(BSIM4v0DPspPtr, BSIM4v0dNodePrime, BSIM4v0sNodePrime)
            TSTALLOC(BSIM4v0DPdPtr, BSIM4v0dNodePrime, BSIM4v0dNode)
            TSTALLOC(BSIM4v0DPgpPtr, BSIM4v0dNodePrime, BSIM4v0gNodePrime)
            TSTALLOC(BSIM4v0SPgpPtr, BSIM4v0sNodePrime, BSIM4v0gNodePrime)
            TSTALLOC(BSIM4v0SPsPtr, BSIM4v0sNodePrime, BSIM4v0sNode)
            TSTALLOC(BSIM4v0SPdpPtr, BSIM4v0sNodePrime, BSIM4v0dNodePrime)

            TSTALLOC(BSIM4v0QqPtr, BSIM4v0qNode, BSIM4v0qNode)
            TSTALLOC(BSIM4v0QbpPtr, BSIM4v0qNode, BSIM4v0bNodePrime) 
            TSTALLOC(BSIM4v0QdpPtr, BSIM4v0qNode, BSIM4v0dNodePrime)
            TSTALLOC(BSIM4v0QspPtr, BSIM4v0qNode, BSIM4v0sNodePrime)
            TSTALLOC(BSIM4v0QgpPtr, BSIM4v0qNode, BSIM4v0gNodePrime)
            TSTALLOC(BSIM4v0DPqPtr, BSIM4v0dNodePrime, BSIM4v0qNode)
            TSTALLOC(BSIM4v0SPqPtr, BSIM4v0sNodePrime, BSIM4v0qNode)
            TSTALLOC(BSIM4v0GPqPtr, BSIM4v0gNodePrime, BSIM4v0qNode)

            if (here->BSIM4v0rgateMod != 0)
            {   TSTALLOC(BSIM4v0GEgePtr, BSIM4v0gNodeExt, BSIM4v0gNodeExt)
                TSTALLOC(BSIM4v0GEgpPtr, BSIM4v0gNodeExt, BSIM4v0gNodePrime)
                TSTALLOC(BSIM4v0GPgePtr, BSIM4v0gNodePrime, BSIM4v0gNodeExt)
		TSTALLOC(BSIM4v0GEdpPtr, BSIM4v0gNodeExt, BSIM4v0dNodePrime)
		TSTALLOC(BSIM4v0GEspPtr, BSIM4v0gNodeExt, BSIM4v0sNodePrime)
		TSTALLOC(BSIM4v0GEbpPtr, BSIM4v0gNodeExt, BSIM4v0bNodePrime)

                TSTALLOC(BSIM4v0GMdpPtr, BSIM4v0gNodeMid, BSIM4v0dNodePrime)
                TSTALLOC(BSIM4v0GMgpPtr, BSIM4v0gNodeMid, BSIM4v0gNodePrime)
                TSTALLOC(BSIM4v0GMgmPtr, BSIM4v0gNodeMid, BSIM4v0gNodeMid)
                TSTALLOC(BSIM4v0GMgePtr, BSIM4v0gNodeMid, BSIM4v0gNodeExt)
                TSTALLOC(BSIM4v0GMspPtr, BSIM4v0gNodeMid, BSIM4v0sNodePrime)
                TSTALLOC(BSIM4v0GMbpPtr, BSIM4v0gNodeMid, BSIM4v0bNodePrime)
                TSTALLOC(BSIM4v0DPgmPtr, BSIM4v0dNodePrime, BSIM4v0gNodeMid)
                TSTALLOC(BSIM4v0GPgmPtr, BSIM4v0gNodePrime, BSIM4v0gNodeMid)
                TSTALLOC(BSIM4v0GEgmPtr, BSIM4v0gNodeExt, BSIM4v0gNodeMid)
                TSTALLOC(BSIM4v0SPgmPtr, BSIM4v0sNodePrime, BSIM4v0gNodeMid)
                TSTALLOC(BSIM4v0BPgmPtr, BSIM4v0bNodePrime, BSIM4v0gNodeMid)
            }	

            if (here->BSIM4v0rbodyMod)
            {   TSTALLOC(BSIM4v0DPdbPtr, BSIM4v0dNodePrime, BSIM4v0dbNode)
                TSTALLOC(BSIM4v0SPsbPtr, BSIM4v0sNodePrime, BSIM4v0sbNode)

                TSTALLOC(BSIM4v0DBdpPtr, BSIM4v0dbNode, BSIM4v0dNodePrime)
                TSTALLOC(BSIM4v0DBdbPtr, BSIM4v0dbNode, BSIM4v0dbNode)
                TSTALLOC(BSIM4v0DBbpPtr, BSIM4v0dbNode, BSIM4v0bNodePrime)
                TSTALLOC(BSIM4v0DBbPtr, BSIM4v0dbNode, BSIM4v0bNode)

                TSTALLOC(BSIM4v0BPdbPtr, BSIM4v0bNodePrime, BSIM4v0dbNode)
                TSTALLOC(BSIM4v0BPbPtr, BSIM4v0bNodePrime, BSIM4v0bNode)
                TSTALLOC(BSIM4v0BPsbPtr, BSIM4v0bNodePrime, BSIM4v0sbNode)

                TSTALLOC(BSIM4v0SBspPtr, BSIM4v0sbNode, BSIM4v0sNodePrime)
                TSTALLOC(BSIM4v0SBbpPtr, BSIM4v0sbNode, BSIM4v0bNodePrime)
                TSTALLOC(BSIM4v0SBbPtr, BSIM4v0sbNode, BSIM4v0bNode)
                TSTALLOC(BSIM4v0SBsbPtr, BSIM4v0sbNode, BSIM4v0sbNode)

                TSTALLOC(BSIM4v0BdbPtr, BSIM4v0bNode, BSIM4v0dbNode)
                TSTALLOC(BSIM4v0BbpPtr, BSIM4v0bNode, BSIM4v0bNodePrime)
                TSTALLOC(BSIM4v0BsbPtr, BSIM4v0bNode, BSIM4v0sbNode)
	        TSTALLOC(BSIM4v0BbPtr, BSIM4v0bNode, BSIM4v0bNode)
	    }

            if (model->BSIM4v0rdsMod)
            {   TSTALLOC(BSIM4v0DgpPtr, BSIM4v0dNode, BSIM4v0gNodePrime)
		TSTALLOC(BSIM4v0DspPtr, BSIM4v0dNode, BSIM4v0sNodePrime)
                TSTALLOC(BSIM4v0DbpPtr, BSIM4v0dNode, BSIM4v0bNodePrime)
                TSTALLOC(BSIM4v0SdpPtr, BSIM4v0sNode, BSIM4v0dNodePrime)
                TSTALLOC(BSIM4v0SgpPtr, BSIM4v0sNode, BSIM4v0gNodePrime)
                TSTALLOC(BSIM4v0SbpPtr, BSIM4v0sNode, BSIM4v0bNodePrime)
            }
        }
    }
    return(OK);
}  

int
BSIM4v0unsetup(inModel,ckt)
    GENmodel *inModel;
    CKTcircuit *ckt;
{
#ifndef HAS_BATCHSIM
    BSIM4v0model *model;
    BSIM4v0instance *here;

    for (model = (BSIM4v0model *)inModel; model != NULL;
            model = model->BSIM4v0nextModel)
    {
        for (here = model->BSIM4v0instances; here != NULL;
                here=here->BSIM4v0nextInstance)
        {
            if (here->BSIM4v0dNodePrime
                    && here->BSIM4v0dNodePrime != here->BSIM4v0dNode)
            {
                CKTdltNNum(ckt, here->BSIM4v0dNodePrime);
                here->BSIM4v0dNodePrime = 0;
            }
            if (here->BSIM4v0sNodePrime
                    && here->BSIM4v0sNodePrime != here->BSIM4v0sNode)
            {
                CKTdltNNum(ckt, here->BSIM4v0sNodePrime);
                here->BSIM4v0sNodePrime = 0;
            }
        }
    }
#endif
    return OK;
}
