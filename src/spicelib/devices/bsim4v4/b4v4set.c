/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/
/* ngspice multirevision code extension covering 4.2.1 & 4.3.0 & 4.4.0 */
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

#include "ngspice/ngspice.h"
#include "ngspice/jobdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "bsim4v4def.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"

#define MAX_EXP 5.834617425e14
#define MIN_EXP 1.713908431e-15
#define EXP_THRESHOLD 34.0
#define EPS0 8.85418e-12
#define EPSSI 1.03594e-10
#define PI 3.141592654
#define Charge_q 1.60219e-19


int
BSIM4v4setup(
SMPmatrix *matrix,
GENmodel *inModel,
CKTcircuit *ckt,
int *states)
{
BSIM4v4model *model = (BSIM4v4model*)inModel;
BSIM4v4instance *here;
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

    /*  loop through all the BSIM4v4 device models */
    for( ; model != NULL; model = model->BSIM4v4nextModel )
    {   /* process defaults of model parameters */
        if (!model->BSIM4v4typeGiven)
            model->BSIM4v4type = NMOS;

        if (!model->BSIM4v4mobModGiven)
            model->BSIM4v4mobMod = 0;
        else if ((model->BSIM4v4mobMod != 0) && (model->BSIM4v4mobMod != 1)
                 && (model->BSIM4v4mobMod != 2))
        {   model->BSIM4v4mobMod = 0;
            printf("Warning: mobMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v4binUnitGiven)
            model->BSIM4v4binUnit = 1;
        if (!model->BSIM4v4paramChkGiven)
            model->BSIM4v4paramChk = 1;

        if (!model->BSIM4v4dioModGiven)
            model->BSIM4v4dioMod = 1;
        else if ((model->BSIM4v4dioMod != 0) && (model->BSIM4v4dioMod != 1)
            && (model->BSIM4v4dioMod != 2))
        {   model->BSIM4v4dioMod = 1;
            printf("Warning: dioMod has been set to its default value: 1.\n");
        }

        if (!model->BSIM4v4capModGiven)
            model->BSIM4v4capMod = 2;
        else if ((model->BSIM4v4capMod != 0) && (model->BSIM4v4capMod != 1)
            && (model->BSIM4v4capMod != 2))
        {   model->BSIM4v4capMod = 2;
            printf("Warning: capMod has been set to its default value: 2.\n");
        }

        if (!model->BSIM4v4rdsModGiven)
            model->BSIM4v4rdsMod = 0;
        else if ((model->BSIM4v4rdsMod != 0) && (model->BSIM4v4rdsMod != 1))
        {   model->BSIM4v4rdsMod = 0;
            printf("Warning: rdsMod has been set to its default value: 0.\n");
        }
        if (!model->BSIM4v4rbodyModGiven)
            model->BSIM4v4rbodyMod = 0;
        else if ((model->BSIM4v4rbodyMod != 0) && (model->BSIM4v4rbodyMod != 1))
        {   model->BSIM4v4rbodyMod = 0;
            printf("Warning: rbodyMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v4rgateModGiven)
            model->BSIM4v4rgateMod = 0;
        else if ((model->BSIM4v4rgateMod != 0) && (model->BSIM4v4rgateMod != 1)
            && (model->BSIM4v4rgateMod != 2) && (model->BSIM4v4rgateMod != 3))
        {   model->BSIM4v4rgateMod = 0;
            printf("Warning: rgateMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v4perModGiven)
            model->BSIM4v4perMod = 1;
        else if ((model->BSIM4v4perMod != 0) && (model->BSIM4v4perMod != 1))
        {   model->BSIM4v4perMod = 1;
            printf("Warning: perMod has been set to its default value: 1.\n");
        }

        if (!model->BSIM4v4geoModGiven)
            model->BSIM4v4geoMod = 0;

        if (!model->BSIM4v4fnoiModGiven)
            model->BSIM4v4fnoiMod = 1;
        else if ((model->BSIM4v4fnoiMod != 0) && (model->BSIM4v4fnoiMod != 1))
        {   model->BSIM4v4fnoiMod = 1;
            printf("Warning: fnoiMod has been set to its default value: 1.\n");
        }
        if (!model->BSIM4v4tnoiModGiven)
            model->BSIM4v4tnoiMod = 0; /* WDLiu: tnoiMod=1 needs to set internal S/D nodes */
        else if ((model->BSIM4v4tnoiMod != 0) && (model->BSIM4v4tnoiMod != 1))
        {   model->BSIM4v4tnoiMod = 0;
            printf("Warning: tnoiMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v4trnqsModGiven)
            model->BSIM4v4trnqsMod = 0;
        else if ((model->BSIM4v4trnqsMod != 0) && (model->BSIM4v4trnqsMod != 1))
        {   model->BSIM4v4trnqsMod = 0;
            printf("Warning: trnqsMod has been set to its default value: 0.\n");
        }
        if (!model->BSIM4v4acnqsModGiven)
            model->BSIM4v4acnqsMod = 0;
        else if ((model->BSIM4v4acnqsMod != 0) && (model->BSIM4v4acnqsMod != 1))
        {   model->BSIM4v4acnqsMod = 0;
            printf("Warning: acnqsMod has been set to its default value: 0.\n");
        }

        if (!model->BSIM4v4igcModGiven)
            model->BSIM4v4igcMod = 0;
        else if ((model->BSIM4v4igcMod != 0) && (model->BSIM4v4igcMod != 1))
        {   model->BSIM4v4igcMod = 0;
            printf("Warning: igcMod has been set to its default value: 0.\n");
        }
        if (!model->BSIM4v4igbModGiven)
            model->BSIM4v4igbMod = 0;
        else if ((model->BSIM4v4igbMod != 0) && (model->BSIM4v4igbMod != 1))
        {   model->BSIM4v4igbMod = 0;
            printf("Warning: igbMod has been set to its default value: 0.\n");
        }

        /* If the user does not provide the model revision,
         * we always choose the most recent for this code.
         */
        if (!model->BSIM4v4versionGiven)
            model->BSIM4v4version = "4.4.0";

        if ((!strcmp(model->BSIM4v4version, "4.2.1"))||(!strcmp(model->BSIM4v4version, "4.21")))
            model->BSIM4v4intVersion = BSIM4v21;
        else if ((!strcmp(model->BSIM4v4version, "4.3.0"))||(!strcmp(model->BSIM4v4version, "4.30"))||(!strcmp(model->BSIM4v4version, "4.3")))
            model->BSIM4v4intVersion = BSIM4v30;
        else if ((!strcmp(model->BSIM4v4version, "4.4.0"))||(!strcmp(model->BSIM4v4version, "4.40"))||(!strcmp(model->BSIM4v4version, "4.4")))
            model->BSIM4v4intVersion = BSIM4v40;
        else
            model->BSIM4v4intVersion = BSIM4vOLD;
        /* BSIM4vOLD is a placeholder for pre 2.1 revision
         * This model should not be used for pre 2.1 models.
         */

        switch (model->BSIM4v4intVersion) {
          case BSIM4vOLD: case BSIM4v21:
            break;
          case BSIM4v30: case BSIM4v40:
            if (!model->BSIM4v4tempModGiven)
               model->BSIM4v4tempMod = 0;
            else if ((model->BSIM4v4tempMod != 0) && (model->BSIM4v4tempMod != 1))
            {   model->BSIM4v4tempMod = 0;
                printf("Warning: tempMod has been set to its default value: 0.\n");
            }
            break;
          default: break;
        }

        if (!model->BSIM4v4toxrefGiven)
            model->BSIM4v4toxref = 30.0e-10;
        if (!model->BSIM4v4toxeGiven)
            model->BSIM4v4toxe = 30.0e-10;
        if (!model->BSIM4v4toxpGiven)
            model->BSIM4v4toxp = model->BSIM4v4toxe;
        if (!model->BSIM4v4toxmGiven)
            model->BSIM4v4toxm = model->BSIM4v4toxe;
        if (!model->BSIM4v4dtoxGiven)
            model->BSIM4v4dtox = 0.0;
        if (!model->BSIM4v4epsroxGiven)
            model->BSIM4v4epsrox = 3.9;

        if (!model->BSIM4v4cdscGiven)
            model->BSIM4v4cdsc = 2.4e-4;   /* unit Q/V/m^2  */
        if (!model->BSIM4v4cdscbGiven)
            model->BSIM4v4cdscb = 0.0;   /* unit Q/V/m^2  */
            if (!model->BSIM4v4cdscdGiven)
            model->BSIM4v4cdscd = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM4v4citGiven)
            model->BSIM4v4cit = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM4v4nfactorGiven)
            model->BSIM4v4nfactor = 1.0;
        if (!model->BSIM4v4xjGiven)
            model->BSIM4v4xj = .15e-6;
        if (!model->BSIM4v4vsatGiven)
            model->BSIM4v4vsat = 8.0e4;    /* unit m/s */
        if (!model->BSIM4v4atGiven)
            model->BSIM4v4at = 3.3e4;    /* unit m/s */
        if (!model->BSIM4v4a0Given)
            model->BSIM4v4a0 = 1.0;
        if (!model->BSIM4v4agsGiven)
            model->BSIM4v4ags = 0.0;
        if (!model->BSIM4v4a1Given)
            model->BSIM4v4a1 = 0.0;
        if (!model->BSIM4v4a2Given)
            model->BSIM4v4a2 = 1.0;
        if (!model->BSIM4v4ketaGiven)
            model->BSIM4v4keta = -0.047;    /* unit  / V */
        if (!model->BSIM4v4nsubGiven)
            model->BSIM4v4nsub = 6.0e16;   /* unit 1/cm3 */
        if (!model->BSIM4v4ndepGiven)
            model->BSIM4v4ndep = 1.7e17;   /* unit 1/cm3 */
        if (!model->BSIM4v4nsdGiven)
            model->BSIM4v4nsd = 1.0e20;   /* unit 1/cm3 */
        if (!model->BSIM4v4phinGiven)
            model->BSIM4v4phin = 0.0; /* unit V */
        if (!model->BSIM4v4ngateGiven)
            model->BSIM4v4ngate = 0;   /* unit 1/cm3 */
        if (!model->BSIM4v4vbmGiven)
            model->BSIM4v4vbm = -3.0;
        if (!model->BSIM4v4xtGiven)
            model->BSIM4v4xt = 1.55e-7;
        if (!model->BSIM4v4kt1Given)
            model->BSIM4v4kt1 = -0.11;      /* unit V */
        if (!model->BSIM4v4kt1lGiven)
            model->BSIM4v4kt1l = 0.0;      /* unit V*m */
        if (!model->BSIM4v4kt2Given)
            model->BSIM4v4kt2 = 0.022;      /* No unit */
        if (!model->BSIM4v4k3Given)
            model->BSIM4v4k3 = 80.0;
        if (!model->BSIM4v4k3bGiven)
            model->BSIM4v4k3b = 0.0;
        if (!model->BSIM4v4w0Given)
            model->BSIM4v4w0 = 2.5e-6;
        if (!model->BSIM4v4lpe0Given)
            model->BSIM4v4lpe0 = 1.74e-7;
        if (!model->BSIM4v4lpebGiven)
            model->BSIM4v4lpeb = 0.0;
        if (!model->BSIM4v4dvtp0Given)
            model->BSIM4v4dvtp0 = 0.0;
        if (!model->BSIM4v4dvtp1Given)
            model->BSIM4v4dvtp1 = 0.0;
        if (!model->BSIM4v4dvt0Given)
            model->BSIM4v4dvt0 = 2.2;
        if (!model->BSIM4v4dvt1Given)
            model->BSIM4v4dvt1 = 0.53;
        if (!model->BSIM4v4dvt2Given)
            model->BSIM4v4dvt2 = -0.032;   /* unit 1 / V */

        if (!model->BSIM4v4dvt0wGiven)
            model->BSIM4v4dvt0w = 0.0;
        if (!model->BSIM4v4dvt1wGiven)
            model->BSIM4v4dvt1w = 5.3e6;
        if (!model->BSIM4v4dvt2wGiven)
            model->BSIM4v4dvt2w = -0.032;

        if (!model->BSIM4v4droutGiven)
            model->BSIM4v4drout = 0.56;
        if (!model->BSIM4v4dsubGiven)
            model->BSIM4v4dsub = model->BSIM4v4drout;
        if (!model->BSIM4v4vth0Given)
            model->BSIM4v4vth0 = (model->BSIM4v4type == NMOS) ? 0.7 : -0.7;
        if (!model->BSIM4v4euGiven)
            model->BSIM4v4eu = (model->BSIM4v4type == NMOS) ? 1.67 : 1.0;;
        if (!model->BSIM4v4uaGiven)
            model->BSIM4v4ua = (model->BSIM4v4mobMod == 2) ? 1.0e-15 : 1.0e-9; /* unit m/V */
        if (!model->BSIM4v4ua1Given)
            model->BSIM4v4ua1 = 1.0e-9;      /* unit m/V */
        if (!model->BSIM4v4ubGiven)
            model->BSIM4v4ub = 1.0e-19;     /* unit (m/V)**2 */
        if (!model->BSIM4v4ub1Given)
            model->BSIM4v4ub1 = -1.0e-18;     /* unit (m/V)**2 */
        if (!model->BSIM4v4ucGiven)
            model->BSIM4v4uc = (model->BSIM4v4mobMod == 1) ? -0.0465 : -0.0465e-9;
        if (!model->BSIM4v4uc1Given)
            model->BSIM4v4uc1 = (model->BSIM4v4mobMod == 1) ? -0.056 : -0.056e-9;
        if (!model->BSIM4v4u0Given)
            model->BSIM4v4u0 = (model->BSIM4v4type == NMOS) ? 0.067 : 0.025;
        if (!model->BSIM4v4uteGiven)
            model->BSIM4v4ute = -1.5;
        if (!model->BSIM4v4voffGiven)
            model->BSIM4v4voff = -0.08;
        if (!model->BSIM4v4vofflGiven)
            model->BSIM4v4voffl = 0.0;
        if (!model->BSIM4v4minvGiven)
            model->BSIM4v4minv = 0.0;
        if (!model->BSIM4v4fproutGiven)
            model->BSIM4v4fprout = 0.0;
        if (!model->BSIM4v4pditsGiven)
            model->BSIM4v4pdits = 0.0;
        if (!model->BSIM4v4pditsdGiven)
            model->BSIM4v4pditsd = 0.0;
        if (!model->BSIM4v4pditslGiven)
            model->BSIM4v4pditsl = 0.0;
        if (!model->BSIM4v4deltaGiven)
           model->BSIM4v4delta = 0.01;
        if (!model->BSIM4v4rdswminGiven)
            model->BSIM4v4rdswmin = 0.0;
        if (!model->BSIM4v4rdwminGiven)
            model->BSIM4v4rdwmin = 0.0;
        if (!model->BSIM4v4rswminGiven)
            model->BSIM4v4rswmin = 0.0;
        if (!model->BSIM4v4rdswGiven)
            model->BSIM4v4rdsw = 200.0; /* in ohm*um */
        if (!model->BSIM4v4rdwGiven)
            model->BSIM4v4rdw = 100.0;
        if (!model->BSIM4v4rswGiven)
            model->BSIM4v4rsw = 100.0;
        if (!model->BSIM4v4prwgGiven)
            model->BSIM4v4prwg = 1.0; /* in 1/V */
        if (!model->BSIM4v4prwbGiven)
            model->BSIM4v4prwb = 0.0;
        if (!model->BSIM4v4prtGiven)
        if (!model->BSIM4v4prtGiven)
            model->BSIM4v4prt = 0.0;
        if (!model->BSIM4v4eta0Given)
            model->BSIM4v4eta0 = 0.08;      /* no unit  */
        if (!model->BSIM4v4etabGiven)
            model->BSIM4v4etab = -0.07;      /* unit  1/V */
        if (!model->BSIM4v4pclmGiven)
            model->BSIM4v4pclm = 1.3;      /* no unit  */
        if (!model->BSIM4v4pdibl1Given)
            model->BSIM4v4pdibl1 = 0.39;    /* no unit  */
        if (!model->BSIM4v4pdibl2Given)
            model->BSIM4v4pdibl2 = 0.0086;    /* no unit  */
        if (!model->BSIM4v4pdiblbGiven)
            model->BSIM4v4pdiblb = 0.0;    /* 1/V  */
        if (!model->BSIM4v4pscbe1Given)
            model->BSIM4v4pscbe1 = 4.24e8;
        if (!model->BSIM4v4pscbe2Given)
            model->BSIM4v4pscbe2 = 1.0e-5;
        if (!model->BSIM4v4pvagGiven)
            model->BSIM4v4pvag = 0.0;
        if (!model->BSIM4v4wrGiven)
            model->BSIM4v4wr = 1.0;
        if (!model->BSIM4v4dwgGiven)
            model->BSIM4v4dwg = 0.0;
        if (!model->BSIM4v4dwbGiven)
            model->BSIM4v4dwb = 0.0;
        if (!model->BSIM4v4b0Given)
            model->BSIM4v4b0 = 0.0;
        if (!model->BSIM4v4b1Given)
            model->BSIM4v4b1 = 0.0;
        if (!model->BSIM4v4alpha0Given)
            model->BSIM4v4alpha0 = 0.0;
        if (!model->BSIM4v4alpha1Given)
            model->BSIM4v4alpha1 = 0.0;
        if (!model->BSIM4v4beta0Given)
            model->BSIM4v4beta0 = 30.0;
        if (!model->BSIM4v4agidlGiven)
            model->BSIM4v4agidl = 0.0;
        if (!model->BSIM4v4bgidlGiven)
            model->BSIM4v4bgidl = 2.3e9; /* V/m */
        if (!model->BSIM4v4cgidlGiven)
            model->BSIM4v4cgidl = 0.5; /* V^3 */
        if (!model->BSIM4v4egidlGiven)
            model->BSIM4v4egidl = 0.8; /* V */
        if (!model->BSIM4v4aigcGiven)
            model->BSIM4v4aigc = (model->BSIM4v4type == NMOS) ? 0.43 : 0.31;
        if (!model->BSIM4v4bigcGiven)
            model->BSIM4v4bigc = (model->BSIM4v4type == NMOS) ? 0.054 : 0.024;
        if (!model->BSIM4v4cigcGiven)
            model->BSIM4v4cigc = (model->BSIM4v4type == NMOS) ? 0.075 : 0.03;
        if (!model->BSIM4v4aigsdGiven)
            model->BSIM4v4aigsd = (model->BSIM4v4type == NMOS) ? 0.43 : 0.31;
        if (!model->BSIM4v4bigsdGiven)
            model->BSIM4v4bigsd = (model->BSIM4v4type == NMOS) ? 0.054 : 0.024;
        if (!model->BSIM4v4cigsdGiven)
            model->BSIM4v4cigsd = (model->BSIM4v4type == NMOS) ? 0.075 : 0.03;
        if (!model->BSIM4v4aigbaccGiven)
            model->BSIM4v4aigbacc = 0.43;
        if (!model->BSIM4v4bigbaccGiven)
            model->BSIM4v4bigbacc = 0.054;
        if (!model->BSIM4v4cigbaccGiven)
            model->BSIM4v4cigbacc = 0.075;
        if (!model->BSIM4v4aigbinvGiven)
            model->BSIM4v4aigbinv = 0.35;
        if (!model->BSIM4v4bigbinvGiven)
            model->BSIM4v4bigbinv = 0.03;
        if (!model->BSIM4v4cigbinvGiven)
            model->BSIM4v4cigbinv = 0.006;
        if (!model->BSIM4v4nigcGiven)
            model->BSIM4v4nigc = 1.0;
        if (!model->BSIM4v4nigbinvGiven)
            model->BSIM4v4nigbinv = 3.0;
        if (!model->BSIM4v4nigbaccGiven)
            model->BSIM4v4nigbacc = 1.0;
        if (!model->BSIM4v4ntoxGiven)
            model->BSIM4v4ntox = 1.0;
        if (!model->BSIM4v4eigbinvGiven)
            model->BSIM4v4eigbinv = 1.1;
        if (!model->BSIM4v4pigcdGiven)
            model->BSIM4v4pigcd = 1.0;
        if (!model->BSIM4v4poxedgeGiven)
            model->BSIM4v4poxedge = 1.0;
        if (!model->BSIM4v4xrcrg1Given)
            model->BSIM4v4xrcrg1 = 12.0;
        if (!model->BSIM4v4xrcrg2Given)
            model->BSIM4v4xrcrg2 = 1.0;
        if (!model->BSIM4v4ijthsfwdGiven)
            model->BSIM4v4ijthsfwd = 0.1; /* unit A */
        if (!model->BSIM4v4ijthdfwdGiven)
            model->BSIM4v4ijthdfwd = model->BSIM4v4ijthsfwd;
        if (!model->BSIM4v4ijthsrevGiven)
            model->BSIM4v4ijthsrev = 0.1; /* unit A */
        if (!model->BSIM4v4ijthdrevGiven)
            model->BSIM4v4ijthdrev = model->BSIM4v4ijthsrev;
        if (!model->BSIM4v4tnoiaGiven)
            model->BSIM4v4tnoia = 1.5;
        if (!model->BSIM4v4tnoibGiven)
            model->BSIM4v4tnoib = 3.5;
        if (!model->BSIM4v4ntnoiGiven)
            model->BSIM4v4ntnoi = 1.0;
        switch (model->BSIM4v4intVersion) {
          case BSIM4vOLD: case BSIM4v21:
            break;
          case BSIM4v30:
            if (!model->BSIM4v4rnoibGiven)
                model->BSIM4v4rnoib = 0.37;
            break;
          case BSIM4v40:
            if (!model->BSIM4v4rnoibGiven)
                model->BSIM4v4rnoib = 0.5164;
            break;
          default: break;
        }
        switch (model->BSIM4v4intVersion) {
          case BSIM4vOLD: case BSIM4v21:
            break;
          case BSIM4v30: case BSIM4v40:
            if (!model->BSIM4v4rnoiaGiven)
                model->BSIM4v4rnoia = 0.577;
            if (!model->BSIM4v4lambdaGiven)
                model->BSIM4v4lambda = 0.0;
            if (!model->BSIM4v4vtlGiven)
                model->BSIM4v4vtl = 2.0e5;    /* unit m/s */
            if (!model->BSIM4v4xnGiven)
                model->BSIM4v4xn = 3.0;
            if (!model->BSIM4v4lcGiven)
                model->BSIM4v4lc = 5.0e-9;
            break;
          default: break;
        }
        switch (model->BSIM4v4intVersion) {
          case BSIM4vOLD: case BSIM4v21: case BSIM4v30:
            break;
          case BSIM4v40:
             if (!model->BSIM4v4vfbsdoffGiven)
                 model->BSIM4v4vfbsdoff = 0.0;  /* unit v */
             if (!model->BSIM4v4lintnoiGiven)
                 model->BSIM4v4lintnoi = 0.0;  /* unit m */
             break;
          default: break;
        }
        if (!model->BSIM4v4xjbvsGiven)
            model->BSIM4v4xjbvs = 1.0; /* no unit */
        if (!model->BSIM4v4xjbvdGiven)
            model->BSIM4v4xjbvd = model->BSIM4v4xjbvs;
        if (!model->BSIM4v4bvsGiven)
            model->BSIM4v4bvs = 10.0; /* V */
        if (!model->BSIM4v4bvdGiven)
            model->BSIM4v4bvd = model->BSIM4v4bvs;
        if (!model->BSIM4v4gbminGiven)
            model->BSIM4v4gbmin = 1.0e-12; /* in mho */
        if (!model->BSIM4v4rbdbGiven)
            model->BSIM4v4rbdb = 50.0; /* in ohm */
        if (!model->BSIM4v4rbpbGiven)
            model->BSIM4v4rbpb = 50.0;
        if (!model->BSIM4v4rbsbGiven)
            model->BSIM4v4rbsb = 50.0;
        if (!model->BSIM4v4rbpsGiven)
            model->BSIM4v4rbps = 50.0;
        if (!model->BSIM4v4rbpdGiven)
            model->BSIM4v4rbpd = 50.0;

        if (!model->BSIM4v4cgslGiven)
            model->BSIM4v4cgsl = 0.0;
        if (!model->BSIM4v4cgdlGiven)
            model->BSIM4v4cgdl = 0.0;
        if (!model->BSIM4v4ckappasGiven)
            model->BSIM4v4ckappas = 0.6;
        if (!model->BSIM4v4ckappadGiven)
            model->BSIM4v4ckappad = model->BSIM4v4ckappas;
        if (!model->BSIM4v4clcGiven)
            model->BSIM4v4clc = 0.1e-6;
        if (!model->BSIM4v4cleGiven)
            model->BSIM4v4cle = 0.6;
        if (!model->BSIM4v4vfbcvGiven)
            model->BSIM4v4vfbcv = -1.0;
        if (!model->BSIM4v4acdeGiven)
            model->BSIM4v4acde = 1.0;
        if (!model->BSIM4v4moinGiven)
            model->BSIM4v4moin = 15.0;
        if (!model->BSIM4v4noffGiven)
            model->BSIM4v4noff = 1.0;
        if (!model->BSIM4v4voffcvGiven)
            model->BSIM4v4voffcv = 0.0;
        if (!model->BSIM4v4dmcgGiven)
            model->BSIM4v4dmcg = 0.0;
        if (!model->BSIM4v4dmciGiven)
            model->BSIM4v4dmci = model->BSIM4v4dmcg;
        if (!model->BSIM4v4dmdgGiven)
            model->BSIM4v4dmdg = 0.0;
        if (!model->BSIM4v4dmcgtGiven)
            model->BSIM4v4dmcgt = 0.0;
        if (!model->BSIM4v4xgwGiven)
            model->BSIM4v4xgw = 0.0;
        if (!model->BSIM4v4xglGiven)
            model->BSIM4v4xgl = 0.0;
        if (!model->BSIM4v4rshgGiven)
            model->BSIM4v4rshg = 0.1;
        if (!model->BSIM4v4ngconGiven)
            model->BSIM4v4ngcon = 1.0;
        if (!model->BSIM4v4tcjGiven)
            model->BSIM4v4tcj = 0.0;
        if (!model->BSIM4v4tpbGiven)
            model->BSIM4v4tpb = 0.0;
        if (!model->BSIM4v4tcjswGiven)
            model->BSIM4v4tcjsw = 0.0;
        if (!model->BSIM4v4tpbswGiven)
            model->BSIM4v4tpbsw = 0.0;
        if (!model->BSIM4v4tcjswgGiven)
            model->BSIM4v4tcjswg = 0.0;
        if (!model->BSIM4v4tpbswgGiven)
            model->BSIM4v4tpbswg = 0.0;

        /* Length dependence */
        if (!model->BSIM4v4lcdscGiven)
            model->BSIM4v4lcdsc = 0.0;
        if (!model->BSIM4v4lcdscbGiven)
            model->BSIM4v4lcdscb = 0.0;
        if (!model->BSIM4v4lcdscdGiven)
            model->BSIM4v4lcdscd = 0.0;
        if (!model->BSIM4v4lcitGiven)
            model->BSIM4v4lcit = 0.0;
        if (!model->BSIM4v4lnfactorGiven)
            model->BSIM4v4lnfactor = 0.0;
        if (!model->BSIM4v4lxjGiven)
            model->BSIM4v4lxj = 0.0;
        if (!model->BSIM4v4lvsatGiven)
            model->BSIM4v4lvsat = 0.0;
        if (!model->BSIM4v4latGiven)
            model->BSIM4v4lat = 0.0;
        if (!model->BSIM4v4la0Given)
            model->BSIM4v4la0 = 0.0;
        if (!model->BSIM4v4lagsGiven)
            model->BSIM4v4lags = 0.0;
        if (!model->BSIM4v4la1Given)
            model->BSIM4v4la1 = 0.0;
        if (!model->BSIM4v4la2Given)
            model->BSIM4v4la2 = 0.0;
        if (!model->BSIM4v4lketaGiven)
            model->BSIM4v4lketa = 0.0;
        if (!model->BSIM4v4lnsubGiven)
            model->BSIM4v4lnsub = 0.0;
        if (!model->BSIM4v4lndepGiven)
            model->BSIM4v4lndep = 0.0;
        if (!model->BSIM4v4lnsdGiven)
            model->BSIM4v4lnsd = 0.0;
        if (!model->BSIM4v4lphinGiven)
            model->BSIM4v4lphin = 0.0;
        if (!model->BSIM4v4lngateGiven)
            model->BSIM4v4lngate = 0.0;
        if (!model->BSIM4v4lvbmGiven)
            model->BSIM4v4lvbm = 0.0;
        if (!model->BSIM4v4lxtGiven)
            model->BSIM4v4lxt = 0.0;
        if (!model->BSIM4v4lkt1Given)
            model->BSIM4v4lkt1 = 0.0;
        if (!model->BSIM4v4lkt1lGiven)
            model->BSIM4v4lkt1l = 0.0;
        if (!model->BSIM4v4lkt2Given)
            model->BSIM4v4lkt2 = 0.0;
        if (!model->BSIM4v4lk3Given)
            model->BSIM4v4lk3 = 0.0;
        if (!model->BSIM4v4lk3bGiven)
            model->BSIM4v4lk3b = 0.0;
        if (!model->BSIM4v4lw0Given)
            model->BSIM4v4lw0 = 0.0;
        if (!model->BSIM4v4llpe0Given)
            model->BSIM4v4llpe0 = 0.0;
        switch (model->BSIM4v4intVersion) {
          case BSIM4vOLD: case BSIM4v21: case BSIM4v30:
            if (!model->BSIM4v4llpebGiven)
                model->BSIM4v4llpeb = model->BSIM4v4llpe0;
            break;
          case BSIM4v40:
            if (!model->BSIM4v4llpebGiven)
                model->BSIM4v4llpeb = 0.0;
             break;
          default: break;
        }
        if (!model->BSIM4v4ldvtp0Given)
            model->BSIM4v4ldvtp0 = 0.0;
        if (!model->BSIM4v4ldvtp1Given)
            model->BSIM4v4ldvtp1 = 0.0;
        if (!model->BSIM4v4ldvt0Given)
            model->BSIM4v4ldvt0 = 0.0;
        if (!model->BSIM4v4ldvt1Given)
            model->BSIM4v4ldvt1 = 0.0;
        if (!model->BSIM4v4ldvt2Given)
            model->BSIM4v4ldvt2 = 0.0;
        if (!model->BSIM4v4ldvt0wGiven)
            model->BSIM4v4ldvt0w = 0.0;
        if (!model->BSIM4v4ldvt1wGiven)
            model->BSIM4v4ldvt1w = 0.0;
        if (!model->BSIM4v4ldvt2wGiven)
            model->BSIM4v4ldvt2w = 0.0;
        if (!model->BSIM4v4ldroutGiven)
            model->BSIM4v4ldrout = 0.0;
        if (!model->BSIM4v4ldsubGiven)
            model->BSIM4v4ldsub = 0.0;
        if (!model->BSIM4v4lvth0Given)
           model->BSIM4v4lvth0 = 0.0;
        if (!model->BSIM4v4luaGiven)
            model->BSIM4v4lua = 0.0;
        if (!model->BSIM4v4lua1Given)
            model->BSIM4v4lua1 = 0.0;
        if (!model->BSIM4v4lubGiven)
            model->BSIM4v4lub = 0.0;
        if (!model->BSIM4v4lub1Given)
            model->BSIM4v4lub1 = 0.0;
        if (!model->BSIM4v4lucGiven)
            model->BSIM4v4luc = 0.0;
        if (!model->BSIM4v4luc1Given)
            model->BSIM4v4luc1 = 0.0;
        if (!model->BSIM4v4lu0Given)
            model->BSIM4v4lu0 = 0.0;
        if (!model->BSIM4v4luteGiven)
            model->BSIM4v4lute = 0.0;
        if (!model->BSIM4v4lvoffGiven)
            model->BSIM4v4lvoff = 0.0;
        if (!model->BSIM4v4lminvGiven)
            model->BSIM4v4lminv = 0.0;
        if (!model->BSIM4v4lfproutGiven)
            model->BSIM4v4lfprout = 0.0;
        if (!model->BSIM4v4lpditsGiven)
            model->BSIM4v4lpdits = 0.0;
        if (!model->BSIM4v4lpditsdGiven)
            model->BSIM4v4lpditsd = 0.0;
        if (!model->BSIM4v4ldeltaGiven)
            model->BSIM4v4ldelta = 0.0;
        if (!model->BSIM4v4lrdswGiven)
            model->BSIM4v4lrdsw = 0.0;
        if (!model->BSIM4v4lrdwGiven)
            model->BSIM4v4lrdw = 0.0;
        if (!model->BSIM4v4lrswGiven)
            model->BSIM4v4lrsw = 0.0;
        if (!model->BSIM4v4lprwbGiven)
            model->BSIM4v4lprwb = 0.0;
        if (!model->BSIM4v4lprwgGiven)
            model->BSIM4v4lprwg = 0.0;
        if (!model->BSIM4v4lprtGiven)
            model->BSIM4v4lprt = 0.0;
        if (!model->BSIM4v4leta0Given)
            model->BSIM4v4leta0 = 0.0;
        if (!model->BSIM4v4letabGiven)
            model->BSIM4v4letab = -0.0;
        if (!model->BSIM4v4lpclmGiven)
            model->BSIM4v4lpclm = 0.0;
        if (!model->BSIM4v4lpdibl1Given)
            model->BSIM4v4lpdibl1 = 0.0;
        if (!model->BSIM4v4lpdibl2Given)
            model->BSIM4v4lpdibl2 = 0.0;
        if (!model->BSIM4v4lpdiblbGiven)
            model->BSIM4v4lpdiblb = 0.0;
        if (!model->BSIM4v4lpscbe1Given)
            model->BSIM4v4lpscbe1 = 0.0;
        if (!model->BSIM4v4lpscbe2Given)
            model->BSIM4v4lpscbe2 = 0.0;
        if (!model->BSIM4v4lpvagGiven)
            model->BSIM4v4lpvag = 0.0;
        if (!model->BSIM4v4lwrGiven)
            model->BSIM4v4lwr = 0.0;
        if (!model->BSIM4v4ldwgGiven)
            model->BSIM4v4ldwg = 0.0;
        if (!model->BSIM4v4ldwbGiven)
            model->BSIM4v4ldwb = 0.0;
        if (!model->BSIM4v4lb0Given)
            model->BSIM4v4lb0 = 0.0;
        if (!model->BSIM4v4lb1Given)
            model->BSIM4v4lb1 = 0.0;
        if (!model->BSIM4v4lalpha0Given)
            model->BSIM4v4lalpha0 = 0.0;
        if (!model->BSIM4v4lalpha1Given)
            model->BSIM4v4lalpha1 = 0.0;
        if (!model->BSIM4v4lbeta0Given)
            model->BSIM4v4lbeta0 = 0.0;
        if (!model->BSIM4v4lagidlGiven)
            model->BSIM4v4lagidl = 0.0;
        if (!model->BSIM4v4lbgidlGiven)
            model->BSIM4v4lbgidl = 0.0;
        if (!model->BSIM4v4lcgidlGiven)
            model->BSIM4v4lcgidl = 0.0;
        if (!model->BSIM4v4legidlGiven)
            model->BSIM4v4legidl = 0.0;
        if (!model->BSIM4v4laigcGiven)
            model->BSIM4v4laigc = 0.0;
        if (!model->BSIM4v4lbigcGiven)
            model->BSIM4v4lbigc = 0.0;
        if (!model->BSIM4v4lcigcGiven)
            model->BSIM4v4lcigc = 0.0;
        if (!model->BSIM4v4laigsdGiven)
            model->BSIM4v4laigsd = 0.0;
        if (!model->BSIM4v4lbigsdGiven)
            model->BSIM4v4lbigsd = 0.0;
        if (!model->BSIM4v4lcigsdGiven)
            model->BSIM4v4lcigsd = 0.0;
        if (!model->BSIM4v4laigbaccGiven)
            model->BSIM4v4laigbacc = 0.0;
        if (!model->BSIM4v4lbigbaccGiven)
            model->BSIM4v4lbigbacc = 0.0;
        if (!model->BSIM4v4lcigbaccGiven)
            model->BSIM4v4lcigbacc = 0.0;
        if (!model->BSIM4v4laigbinvGiven)
            model->BSIM4v4laigbinv = 0.0;
        if (!model->BSIM4v4lbigbinvGiven)
            model->BSIM4v4lbigbinv = 0.0;
        if (!model->BSIM4v4lcigbinvGiven)
            model->BSIM4v4lcigbinv = 0.0;
        if (!model->BSIM4v4lnigcGiven)
            model->BSIM4v4lnigc = 0.0;
        if (!model->BSIM4v4lnigbinvGiven)
            model->BSIM4v4lnigbinv = 0.0;
        if (!model->BSIM4v4lnigbaccGiven)
            model->BSIM4v4lnigbacc = 0.0;
        if (!model->BSIM4v4lntoxGiven)
            model->BSIM4v4lntox = 0.0;
        if (!model->BSIM4v4leigbinvGiven)
            model->BSIM4v4leigbinv = 0.0;
        if (!model->BSIM4v4lpigcdGiven)
            model->BSIM4v4lpigcd = 0.0;
        if (!model->BSIM4v4lpoxedgeGiven)
            model->BSIM4v4lpoxedge = 0.0;
        if (!model->BSIM4v4lxrcrg1Given)
            model->BSIM4v4lxrcrg1 = 0.0;
        if (!model->BSIM4v4lxrcrg2Given)
            model->BSIM4v4lxrcrg2 = 0.0;
        if (!model->BSIM4v4leuGiven)
            model->BSIM4v4leu = 0.0;
        if (!model->BSIM4v4lvfbGiven)
            model->BSIM4v4lvfb = 0.0;
        switch (model->BSIM4v4intVersion) {
          case BSIM4vOLD: case BSIM4v21:
            break;
          case BSIM4v30: case BSIM4v40:
            if (!model->BSIM4v4llambdaGiven)
                model->BSIM4v4llambda = 0.0;
            if (!model->BSIM4v4lvtlGiven)
                model->BSIM4v4lvtl = 0.0;
            if (!model->BSIM4v4lxnGiven)
                model->BSIM4v4lxn = 0.0;
            break;
          default: break;
        }
        switch (model->BSIM4v4intVersion) {
          case BSIM4vOLD: case BSIM4v21: case BSIM4v30:
            break;
          case BSIM4v40:
            if (!model->BSIM4v4lvfbsdoffGiven)
                model->BSIM4v4lvfbsdoff = 0.0;
            break;
          default: break;
        }
        if (!model->BSIM4v4lcgslGiven)
            model->BSIM4v4lcgsl = 0.0;
        if (!model->BSIM4v4lcgdlGiven)
            model->BSIM4v4lcgdl = 0.0;
        if (!model->BSIM4v4lckappasGiven)
            model->BSIM4v4lckappas = 0.0;
        if (!model->BSIM4v4lckappadGiven)
            model->BSIM4v4lckappad = 0.0;
        if (!model->BSIM4v4lclcGiven)
            model->BSIM4v4lclc = 0.0;
        if (!model->BSIM4v4lcleGiven)
            model->BSIM4v4lcle = 0.0;
        if (!model->BSIM4v4lcfGiven)
            model->BSIM4v4lcf = 0.0;
        if (!model->BSIM4v4lvfbcvGiven)
            model->BSIM4v4lvfbcv = 0.0;
        if (!model->BSIM4v4lacdeGiven)
            model->BSIM4v4lacde = 0.0;
        if (!model->BSIM4v4lmoinGiven)
            model->BSIM4v4lmoin = 0.0;
        if (!model->BSIM4v4lnoffGiven)
            model->BSIM4v4lnoff = 0.0;
        if (!model->BSIM4v4lvoffcvGiven)
            model->BSIM4v4lvoffcv = 0.0;

        /* Width dependence */
        if (!model->BSIM4v4wcdscGiven)
            model->BSIM4v4wcdsc = 0.0;
        if (!model->BSIM4v4wcdscbGiven)
            model->BSIM4v4wcdscb = 0.0;
        if (!model->BSIM4v4wcdscdGiven)
            model->BSIM4v4wcdscd = 0.0;
        if (!model->BSIM4v4wcitGiven)
            model->BSIM4v4wcit = 0.0;
        if (!model->BSIM4v4wnfactorGiven)
            model->BSIM4v4wnfactor = 0.0;
        if (!model->BSIM4v4wxjGiven)
            model->BSIM4v4wxj = 0.0;
        if (!model->BSIM4v4wvsatGiven)
            model->BSIM4v4wvsat = 0.0;
        if (!model->BSIM4v4watGiven)
            model->BSIM4v4wat = 0.0;
        if (!model->BSIM4v4wa0Given)
            model->BSIM4v4wa0 = 0.0;
        if (!model->BSIM4v4wagsGiven)
            model->BSIM4v4wags = 0.0;
        if (!model->BSIM4v4wa1Given)
            model->BSIM4v4wa1 = 0.0;
        if (!model->BSIM4v4wa2Given)
            model->BSIM4v4wa2 = 0.0;
        if (!model->BSIM4v4wketaGiven)
            model->BSIM4v4wketa = 0.0;
        if (!model->BSIM4v4wnsubGiven)
            model->BSIM4v4wnsub = 0.0;
        if (!model->BSIM4v4wndepGiven)
            model->BSIM4v4wndep = 0.0;
        if (!model->BSIM4v4wnsdGiven)
            model->BSIM4v4wnsd = 0.0;
        if (!model->BSIM4v4wphinGiven)
            model->BSIM4v4wphin = 0.0;
        if (!model->BSIM4v4wngateGiven)
            model->BSIM4v4wngate = 0.0;
        if (!model->BSIM4v4wvbmGiven)
            model->BSIM4v4wvbm = 0.0;
        if (!model->BSIM4v4wxtGiven)
            model->BSIM4v4wxt = 0.0;
        if (!model->BSIM4v4wkt1Given)
            model->BSIM4v4wkt1 = 0.0;
        if (!model->BSIM4v4wkt1lGiven)
            model->BSIM4v4wkt1l = 0.0;
        if (!model->BSIM4v4wkt2Given)
            model->BSIM4v4wkt2 = 0.0;
        if (!model->BSIM4v4wk3Given)
            model->BSIM4v4wk3 = 0.0;
        if (!model->BSIM4v4wk3bGiven)
            model->BSIM4v4wk3b = 0.0;
        if (!model->BSIM4v4ww0Given)
            model->BSIM4v4ww0 = 0.0;
        if (!model->BSIM4v4wlpe0Given)
            model->BSIM4v4wlpe0 = 0.0;
        switch (model->BSIM4v4intVersion) {
          case BSIM4vOLD: case BSIM4v21: case BSIM4v30:
            if (!model->BSIM4v4wlpebGiven)
                model->BSIM4v4wlpeb = model->BSIM4v4wlpe0;
            break;
          case BSIM4v40:
            if (!model->BSIM4v4wlpebGiven)
                model->BSIM4v4wlpeb = 0.0;
            break;
          default: break;
        }
        if (!model->BSIM4v4wdvtp0Given)
            model->BSIM4v4wdvtp0 = 0.0;
        if (!model->BSIM4v4wdvtp1Given)
            model->BSIM4v4wdvtp1 = 0.0;
        if (!model->BSIM4v4wdvt0Given)
            model->BSIM4v4wdvt0 = 0.0;
        if (!model->BSIM4v4wdvt1Given)
            model->BSIM4v4wdvt1 = 0.0;
        if (!model->BSIM4v4wdvt2Given)
            model->BSIM4v4wdvt2 = 0.0;
        if (!model->BSIM4v4wdvt0wGiven)
            model->BSIM4v4wdvt0w = 0.0;
        if (!model->BSIM4v4wdvt1wGiven)
            model->BSIM4v4wdvt1w = 0.0;
        if (!model->BSIM4v4wdvt2wGiven)
            model->BSIM4v4wdvt2w = 0.0;
        if (!model->BSIM4v4wdroutGiven)
            model->BSIM4v4wdrout = 0.0;
        if (!model->BSIM4v4wdsubGiven)
            model->BSIM4v4wdsub = 0.0;
        if (!model->BSIM4v4wvth0Given)
           model->BSIM4v4wvth0 = 0.0;
        if (!model->BSIM4v4wuaGiven)
            model->BSIM4v4wua = 0.0;
        if (!model->BSIM4v4wua1Given)
            model->BSIM4v4wua1 = 0.0;
        if (!model->BSIM4v4wubGiven)
            model->BSIM4v4wub = 0.0;
        if (!model->BSIM4v4wub1Given)
            model->BSIM4v4wub1 = 0.0;
        if (!model->BSIM4v4wucGiven)
            model->BSIM4v4wuc = 0.0;
        if (!model->BSIM4v4wuc1Given)
            model->BSIM4v4wuc1 = 0.0;
        if (!model->BSIM4v4wu0Given)
            model->BSIM4v4wu0 = 0.0;
        if (!model->BSIM4v4wuteGiven)
            model->BSIM4v4wute = 0.0;
        if (!model->BSIM4v4wvoffGiven)
            model->BSIM4v4wvoff = 0.0;
        if (!model->BSIM4v4wminvGiven)
            model->BSIM4v4wminv = 0.0;
        if (!model->BSIM4v4wfproutGiven)
            model->BSIM4v4wfprout = 0.0;
        if (!model->BSIM4v4wpditsGiven)
            model->BSIM4v4wpdits = 0.0;
        if (!model->BSIM4v4wpditsdGiven)
            model->BSIM4v4wpditsd = 0.0;
        if (!model->BSIM4v4wdeltaGiven)
            model->BSIM4v4wdelta = 0.0;
        if (!model->BSIM4v4wrdswGiven)
            model->BSIM4v4wrdsw = 0.0;
        if (!model->BSIM4v4wrdwGiven)
            model->BSIM4v4wrdw = 0.0;
        if (!model->BSIM4v4wrswGiven)
            model->BSIM4v4wrsw = 0.0;
        if (!model->BSIM4v4wprwbGiven)
            model->BSIM4v4wprwb = 0.0;
        if (!model->BSIM4v4wprwgGiven)
            model->BSIM4v4wprwg = 0.0;
        if (!model->BSIM4v4wprtGiven)
            model->BSIM4v4wprt = 0.0;
        if (!model->BSIM4v4weta0Given)
            model->BSIM4v4weta0 = 0.0;
        if (!model->BSIM4v4wetabGiven)
            model->BSIM4v4wetab = 0.0;
        if (!model->BSIM4v4wpclmGiven)
            model->BSIM4v4wpclm = 0.0;
        if (!model->BSIM4v4wpdibl1Given)
            model->BSIM4v4wpdibl1 = 0.0;
        if (!model->BSIM4v4wpdibl2Given)
            model->BSIM4v4wpdibl2 = 0.0;
        if (!model->BSIM4v4wpdiblbGiven)
            model->BSIM4v4wpdiblb = 0.0;
        if (!model->BSIM4v4wpscbe1Given)
            model->BSIM4v4wpscbe1 = 0.0;
        if (!model->BSIM4v4wpscbe2Given)
            model->BSIM4v4wpscbe2 = 0.0;
        if (!model->BSIM4v4wpvagGiven)
            model->BSIM4v4wpvag = 0.0;
        if (!model->BSIM4v4wwrGiven)
            model->BSIM4v4wwr = 0.0;
        if (!model->BSIM4v4wdwgGiven)
            model->BSIM4v4wdwg = 0.0;
        if (!model->BSIM4v4wdwbGiven)
            model->BSIM4v4wdwb = 0.0;
        if (!model->BSIM4v4wb0Given)
            model->BSIM4v4wb0 = 0.0;
        if (!model->BSIM4v4wb1Given)
            model->BSIM4v4wb1 = 0.0;
        if (!model->BSIM4v4walpha0Given)
            model->BSIM4v4walpha0 = 0.0;
        if (!model->BSIM4v4walpha1Given)
            model->BSIM4v4walpha1 = 0.0;
        if (!model->BSIM4v4wbeta0Given)
            model->BSIM4v4wbeta0 = 0.0;
        if (!model->BSIM4v4wagidlGiven)
            model->BSIM4v4wagidl = 0.0;
        if (!model->BSIM4v4wbgidlGiven)
            model->BSIM4v4wbgidl = 0.0;
        if (!model->BSIM4v4wcgidlGiven)
            model->BSIM4v4wcgidl = 0.0;
        if (!model->BSIM4v4wegidlGiven)
            model->BSIM4v4wegidl = 0.0;
        if (!model->BSIM4v4waigcGiven)
            model->BSIM4v4waigc = 0.0;
        if (!model->BSIM4v4wbigcGiven)
            model->BSIM4v4wbigc = 0.0;
        if (!model->BSIM4v4wcigcGiven)
            model->BSIM4v4wcigc = 0.0;
        if (!model->BSIM4v4waigsdGiven)
            model->BSIM4v4waigsd = 0.0;
        if (!model->BSIM4v4wbigsdGiven)
            model->BSIM4v4wbigsd = 0.0;
        if (!model->BSIM4v4wcigsdGiven)
            model->BSIM4v4wcigsd = 0.0;
        if (!model->BSIM4v4waigbaccGiven)
            model->BSIM4v4waigbacc = 0.0;
        if (!model->BSIM4v4wbigbaccGiven)
            model->BSIM4v4wbigbacc = 0.0;
        if (!model->BSIM4v4wcigbaccGiven)
            model->BSIM4v4wcigbacc = 0.0;
        if (!model->BSIM4v4waigbinvGiven)
            model->BSIM4v4waigbinv = 0.0;
        if (!model->BSIM4v4wbigbinvGiven)
            model->BSIM4v4wbigbinv = 0.0;
        if (!model->BSIM4v4wcigbinvGiven)
            model->BSIM4v4wcigbinv = 0.0;
        if (!model->BSIM4v4wnigcGiven)
            model->BSIM4v4wnigc = 0.0;
        if (!model->BSIM4v4wnigbinvGiven)
            model->BSIM4v4wnigbinv = 0.0;
        if (!model->BSIM4v4wnigbaccGiven)
            model->BSIM4v4wnigbacc = 0.0;
        if (!model->BSIM4v4wntoxGiven)
            model->BSIM4v4wntox = 0.0;
        if (!model->BSIM4v4weigbinvGiven)
            model->BSIM4v4weigbinv = 0.0;
        if (!model->BSIM4v4wpigcdGiven)
            model->BSIM4v4wpigcd = 0.0;
        if (!model->BSIM4v4wpoxedgeGiven)
            model->BSIM4v4wpoxedge = 0.0;
        if (!model->BSIM4v4wxrcrg1Given)
            model->BSIM4v4wxrcrg1 = 0.0;
        if (!model->BSIM4v4wxrcrg2Given)
            model->BSIM4v4wxrcrg2 = 0.0;
        if (!model->BSIM4v4weuGiven)
            model->BSIM4v4weu = 0.0;
        if (!model->BSIM4v4wvfbGiven)
            model->BSIM4v4wvfb = 0.0;
        switch (model->BSIM4v4intVersion) {
          case BSIM4vOLD:
          case BSIM4v21:
            break;
          case BSIM4v40:
            if (!model->BSIM4v4wlambdaGiven)
                model->BSIM4v4wlambda = 0.0;
            if (!model->BSIM4v4wvtlGiven)
                model->BSIM4v4wvtl = 0.0;
            if (!model->BSIM4v4wxnGiven)
                model->BSIM4v4wxn = 0.0;
            if (!model->BSIM4v4wvfbsdoffGiven)
                model->BSIM4v4wvfbsdoff = 0.0;   
            break;
          default: break;
        }
        if (!model->BSIM4v4wcgslGiven)
            model->BSIM4v4wcgsl = 0.0;
        if (!model->BSIM4v4wcgdlGiven)
            model->BSIM4v4wcgdl = 0.0;
        if (!model->BSIM4v4wckappasGiven)
            model->BSIM4v4wckappas = 0.0;
        if (!model->BSIM4v4wckappadGiven)
            model->BSIM4v4wckappad = 0.0;
        if (!model->BSIM4v4wcfGiven)
            model->BSIM4v4wcf = 0.0;
        if (!model->BSIM4v4wclcGiven)
            model->BSIM4v4wclc = 0.0;
        if (!model->BSIM4v4wcleGiven)
            model->BSIM4v4wcle = 0.0;
        if (!model->BSIM4v4wvfbcvGiven)
            model->BSIM4v4wvfbcv = 0.0;
        if (!model->BSIM4v4wacdeGiven)
            model->BSIM4v4wacde = 0.0;
        if (!model->BSIM4v4wmoinGiven)
            model->BSIM4v4wmoin = 0.0;
        if (!model->BSIM4v4wnoffGiven)
            model->BSIM4v4wnoff = 0.0;
        if (!model->BSIM4v4wvoffcvGiven)
            model->BSIM4v4wvoffcv = 0.0;

        /* Cross-term dependence */
        if (!model->BSIM4v4pcdscGiven)
            model->BSIM4v4pcdsc = 0.0;
        if (!model->BSIM4v4pcdscbGiven)
            model->BSIM4v4pcdscb = 0.0;
        if (!model->BSIM4v4pcdscdGiven)
            model->BSIM4v4pcdscd = 0.0;
        if (!model->BSIM4v4pcitGiven)
            model->BSIM4v4pcit = 0.0;
        if (!model->BSIM4v4pnfactorGiven)
            model->BSIM4v4pnfactor = 0.0;
        if (!model->BSIM4v4pxjGiven)
            model->BSIM4v4pxj = 0.0;
        if (!model->BSIM4v4pvsatGiven)
            model->BSIM4v4pvsat = 0.0;
        if (!model->BSIM4v4patGiven)
            model->BSIM4v4pat = 0.0;
        if (!model->BSIM4v4pa0Given)
            model->BSIM4v4pa0 = 0.0;

        if (!model->BSIM4v4pagsGiven)
            model->BSIM4v4pags = 0.0;
        if (!model->BSIM4v4pa1Given)
            model->BSIM4v4pa1 = 0.0;
        if (!model->BSIM4v4pa2Given)
            model->BSIM4v4pa2 = 0.0;
        if (!model->BSIM4v4pketaGiven)
            model->BSIM4v4pketa = 0.0;
        if (!model->BSIM4v4pnsubGiven)
            model->BSIM4v4pnsub = 0.0;
        if (!model->BSIM4v4pndepGiven)
            model->BSIM4v4pndep = 0.0;
        if (!model->BSIM4v4pnsdGiven)
            model->BSIM4v4pnsd = 0.0;
        if (!model->BSIM4v4pphinGiven)
            model->BSIM4v4pphin = 0.0;
        if (!model->BSIM4v4pngateGiven)
            model->BSIM4v4pngate = 0.0;
        if (!model->BSIM4v4pvbmGiven)
            model->BSIM4v4pvbm = 0.0;
        if (!model->BSIM4v4pxtGiven)
            model->BSIM4v4pxt = 0.0;
        if (!model->BSIM4v4pkt1Given)
            model->BSIM4v4pkt1 = 0.0;
        if (!model->BSIM4v4pkt1lGiven)
            model->BSIM4v4pkt1l = 0.0;
        if (!model->BSIM4v4pkt2Given)
            model->BSIM4v4pkt2 = 0.0;
        if (!model->BSIM4v4pk3Given)
            model->BSIM4v4pk3 = 0.0;
        if (!model->BSIM4v4pk3bGiven)
            model->BSIM4v4pk3b = 0.0;
        if (!model->BSIM4v4pw0Given)
            model->BSIM4v4pw0 = 0.0;
        if (!model->BSIM4v4plpe0Given)
            model->BSIM4v4plpe0 = 0.0;
        switch (model->BSIM4v4intVersion) {
          case BSIM4vOLD: case BSIM4v21: case BSIM4v30:
            if (!model->BSIM4v4plpebGiven)
                model->BSIM4v4plpeb = model->BSIM4v4plpe0;
            break;
          case BSIM4v40:
            if (!model->BSIM4v4llpebGiven)
                model->BSIM4v4plpeb = 0.0;
             break;
          default: break;
        }
        if (!model->BSIM4v4pdvtp0Given)
            model->BSIM4v4pdvtp0 = 0.0;
        if (!model->BSIM4v4pdvtp1Given)
            model->BSIM4v4pdvtp1 = 0.0;
        if (!model->BSIM4v4pdvt0Given)
            model->BSIM4v4pdvt0 = 0.0;
        if (!model->BSIM4v4pdvt1Given)
            model->BSIM4v4pdvt1 = 0.0;
        if (!model->BSIM4v4pdvt2Given)
            model->BSIM4v4pdvt2 = 0.0;
        if (!model->BSIM4v4pdvt0wGiven)
            model->BSIM4v4pdvt0w = 0.0;
        if (!model->BSIM4v4pdvt1wGiven)
            model->BSIM4v4pdvt1w = 0.0;
        if (!model->BSIM4v4pdvt2wGiven)
            model->BSIM4v4pdvt2w = 0.0;
        if (!model->BSIM4v4pdroutGiven)
            model->BSIM4v4pdrout = 0.0;
        if (!model->BSIM4v4pdsubGiven)
            model->BSIM4v4pdsub = 0.0;
        if (!model->BSIM4v4pvth0Given)
           model->BSIM4v4pvth0 = 0.0;
        if (!model->BSIM4v4puaGiven)
            model->BSIM4v4pua = 0.0;
        if (!model->BSIM4v4pua1Given)
            model->BSIM4v4pua1 = 0.0;
        if (!model->BSIM4v4pubGiven)
            model->BSIM4v4pub = 0.0;
        if (!model->BSIM4v4pub1Given)
            model->BSIM4v4pub1 = 0.0;
        if (!model->BSIM4v4pucGiven)
            model->BSIM4v4puc = 0.0;
        if (!model->BSIM4v4puc1Given)
            model->BSIM4v4puc1 = 0.0;
        if (!model->BSIM4v4pu0Given)
            model->BSIM4v4pu0 = 0.0;
        if (!model->BSIM4v4puteGiven)
            model->BSIM4v4pute = 0.0;
        if (!model->BSIM4v4pvoffGiven)
            model->BSIM4v4pvoff = 0.0;
        if (!model->BSIM4v4pminvGiven)
            model->BSIM4v4pminv = 0.0;
        if (!model->BSIM4v4pfproutGiven)
            model->BSIM4v4pfprout = 0.0;
        if (!model->BSIM4v4ppditsGiven)
            model->BSIM4v4ppdits = 0.0;
        if (!model->BSIM4v4ppditsdGiven)
            model->BSIM4v4ppditsd = 0.0;
        if (!model->BSIM4v4pdeltaGiven)
            model->BSIM4v4pdelta = 0.0;
        if (!model->BSIM4v4prdswGiven)
            model->BSIM4v4prdsw = 0.0;
        if (!model->BSIM4v4prdwGiven)
            model->BSIM4v4prdw = 0.0;
        if (!model->BSIM4v4prswGiven)
            model->BSIM4v4prsw = 0.0;
        if (!model->BSIM4v4pprwbGiven)
            model->BSIM4v4pprwb = 0.0;
        if (!model->BSIM4v4pprwgGiven)
            model->BSIM4v4pprwg = 0.0;
        if (!model->BSIM4v4pprtGiven)
            model->BSIM4v4pprt = 0.0;
        if (!model->BSIM4v4peta0Given)
            model->BSIM4v4peta0 = 0.0;
        if (!model->BSIM4v4petabGiven)
            model->BSIM4v4petab = 0.0;
        if (!model->BSIM4v4ppclmGiven)
            model->BSIM4v4ppclm = 0.0;
        if (!model->BSIM4v4ppdibl1Given)
            model->BSIM4v4ppdibl1 = 0.0;
        if (!model->BSIM4v4ppdibl2Given)
            model->BSIM4v4ppdibl2 = 0.0;
        if (!model->BSIM4v4ppdiblbGiven)
            model->BSIM4v4ppdiblb = 0.0;
        if (!model->BSIM4v4ppscbe1Given)
            model->BSIM4v4ppscbe1 = 0.0;
        if (!model->BSIM4v4ppscbe2Given)
            model->BSIM4v4ppscbe2 = 0.0;
        if (!model->BSIM4v4ppvagGiven)
            model->BSIM4v4ppvag = 0.0;
        if (!model->BSIM4v4pwrGiven)
            model->BSIM4v4pwr = 0.0;
        if (!model->BSIM4v4pdwgGiven)
            model->BSIM4v4pdwg = 0.0;
        if (!model->BSIM4v4pdwbGiven)
            model->BSIM4v4pdwb = 0.0;
        if (!model->BSIM4v4pb0Given)
            model->BSIM4v4pb0 = 0.0;
        if (!model->BSIM4v4pb1Given)
            model->BSIM4v4pb1 = 0.0;
        if (!model->BSIM4v4palpha0Given)
            model->BSIM4v4palpha0 = 0.0;
        if (!model->BSIM4v4palpha1Given)
            model->BSIM4v4palpha1 = 0.0;
        if (!model->BSIM4v4pbeta0Given)
            model->BSIM4v4pbeta0 = 0.0;
        if (!model->BSIM4v4pagidlGiven)
            model->BSIM4v4pagidl = 0.0;
        if (!model->BSIM4v4pbgidlGiven)
            model->BSIM4v4pbgidl = 0.0;
        if (!model->BSIM4v4pcgidlGiven)
            model->BSIM4v4pcgidl = 0.0;
        if (!model->BSIM4v4pegidlGiven)
            model->BSIM4v4pegidl = 0.0;
        if (!model->BSIM4v4paigcGiven)
            model->BSIM4v4paigc = 0.0;
        if (!model->BSIM4v4pbigcGiven)
            model->BSIM4v4pbigc = 0.0;
        if (!model->BSIM4v4pcigcGiven)
            model->BSIM4v4pcigc = 0.0;
        if (!model->BSIM4v4paigsdGiven)
            model->BSIM4v4paigsd = 0.0;
        if (!model->BSIM4v4pbigsdGiven)
            model->BSIM4v4pbigsd = 0.0;
        if (!model->BSIM4v4pcigsdGiven)
            model->BSIM4v4pcigsd = 0.0;
        if (!model->BSIM4v4paigbaccGiven)
            model->BSIM4v4paigbacc = 0.0;
        if (!model->BSIM4v4pbigbaccGiven)
            model->BSIM4v4pbigbacc = 0.0;
        if (!model->BSIM4v4pcigbaccGiven)
            model->BSIM4v4pcigbacc = 0.0;
        if (!model->BSIM4v4paigbinvGiven)
            model->BSIM4v4paigbinv = 0.0;
        if (!model->BSIM4v4pbigbinvGiven)
            model->BSIM4v4pbigbinv = 0.0;
        if (!model->BSIM4v4pcigbinvGiven)
            model->BSIM4v4pcigbinv = 0.0;
        if (!model->BSIM4v4pnigcGiven)
            model->BSIM4v4pnigc = 0.0;
        if (!model->BSIM4v4pnigbinvGiven)
            model->BSIM4v4pnigbinv = 0.0;
        if (!model->BSIM4v4pnigbaccGiven)
            model->BSIM4v4pnigbacc = 0.0;
        if (!model->BSIM4v4pntoxGiven)
            model->BSIM4v4pntox = 0.0;
        if (!model->BSIM4v4peigbinvGiven)
            model->BSIM4v4peigbinv = 0.0;
        if (!model->BSIM4v4ppigcdGiven)
            model->BSIM4v4ppigcd = 0.0;
        if (!model->BSIM4v4ppoxedgeGiven)
            model->BSIM4v4ppoxedge = 0.0;
        if (!model->BSIM4v4pxrcrg1Given)
            model->BSIM4v4pxrcrg1 = 0.0;
        if (!model->BSIM4v4pxrcrg2Given)
            model->BSIM4v4pxrcrg2 = 0.0;
        if (!model->BSIM4v4peuGiven)
            model->BSIM4v4peu = 0.0;
        if (!model->BSIM4v4pvfbGiven)
            model->BSIM4v4pvfb = 0.0;
        switch (model->BSIM4v4intVersion) {
          case BSIM4vOLD: case BSIM4v21:
            break;
          case BSIM4v30: case BSIM4v40:
            if (!model->BSIM4v4plambdaGiven)
                model->BSIM4v4plambda = 0.0;
            if (!model->BSIM4v4pvtlGiven)
                model->BSIM4v4pvtl = 0.0;
            if (!model->BSIM4v4pxnGiven)
                model->BSIM4v4pxn = 0.0;
            break;
          default: break;
        }
        switch (model->BSIM4v4intVersion) {
          case BSIM4vOLD: case BSIM4v21: case BSIM4v30:
            break;
          case BSIM4v40:
            if (!model->BSIM4v4pvfbsdoffGiven)
                model->BSIM4v4pvfbsdoff = 0.0;
             break;
          default: break;
        }

        if (!model->BSIM4v4pcgslGiven)
            model->BSIM4v4pcgsl = 0.0;
        if (!model->BSIM4v4pcgdlGiven)
            model->BSIM4v4pcgdl = 0.0;
        if (!model->BSIM4v4pckappasGiven)
            model->BSIM4v4pckappas = 0.0;
        if (!model->BSIM4v4pckappadGiven)
            model->BSIM4v4pckappad = 0.0;
        if (!model->BSIM4v4pcfGiven)
            model->BSIM4v4pcf = 0.0;
        if (!model->BSIM4v4pclcGiven)
            model->BSIM4v4pclc = 0.0;
        if (!model->BSIM4v4pcleGiven)
            model->BSIM4v4pcle = 0.0;
        if (!model->BSIM4v4pvfbcvGiven)
            model->BSIM4v4pvfbcv = 0.0;
        if (!model->BSIM4v4pacdeGiven)
            model->BSIM4v4pacde = 0.0;
        if (!model->BSIM4v4pmoinGiven)
            model->BSIM4v4pmoin = 0.0;
        if (!model->BSIM4v4pnoffGiven)
            model->BSIM4v4pnoff = 0.0;
        if (!model->BSIM4v4pvoffcvGiven)
            model->BSIM4v4pvoffcv = 0.0;
        switch (model->BSIM4v4intVersion) {
          case BSIM4vOLD: case BSIM4v21: case BSIM4v30:
            break;
          case BSIM4v40:
           if (!model->BSIM4v4gamma1Given)
               model->BSIM4v4gamma1 = 0.0;
           if (!model->BSIM4v4lgamma1Given)
               model->BSIM4v4lgamma1 = 0.0;
           if (!model->BSIM4v4wgamma1Given)
               model->BSIM4v4wgamma1 = 0.0;
           if (!model->BSIM4v4pgamma1Given)
               model->BSIM4v4pgamma1 = 0.0;
           if (!model->BSIM4v4gamma2Given)
               model->BSIM4v4gamma2 = 0.0;
           if (!model->BSIM4v4lgamma2Given)
               model->BSIM4v4lgamma2 = 0.0;
           if (!model->BSIM4v4wgamma2Given)
               model->BSIM4v4wgamma2 = 0.0;
           if (!model->BSIM4v4pgamma2Given)
               model->BSIM4v4pgamma2 = 0.0;
           if (!model->BSIM4v4vbxGiven)
               model->BSIM4v4vbx = 0.0;
           if (!model->BSIM4v4lvbxGiven)
               model->BSIM4v4lvbx = 0.0;
           if (!model->BSIM4v4wvbxGiven)
               model->BSIM4v4wvbx = 0.0;
           if (!model->BSIM4v4pvbxGiven)
               model->BSIM4v4pvbx = 0.0;
             break;
          default: break;
        }

        /* unit degree celcius */
        if (!model->BSIM4v4tnomGiven)
            model->BSIM4v4tnom = ckt->CKTnomTemp;
        if (!model->BSIM4v4LintGiven)
           model->BSIM4v4Lint = 0.0;
        if (!model->BSIM4v4LlGiven)
           model->BSIM4v4Ll = 0.0;
        if (!model->BSIM4v4LlcGiven)
           model->BSIM4v4Llc = model->BSIM4v4Ll;
        if (!model->BSIM4v4LlnGiven)
           model->BSIM4v4Lln = 1.0;
        if (!model->BSIM4v4LwGiven)
           model->BSIM4v4Lw = 0.0;
        if (!model->BSIM4v4LwcGiven)
           model->BSIM4v4Lwc = model->BSIM4v4Lw;
        if (!model->BSIM4v4LwnGiven)
           model->BSIM4v4Lwn = 1.0;
        if (!model->BSIM4v4LwlGiven)
           model->BSIM4v4Lwl = 0.0;
        if (!model->BSIM4v4LwlcGiven)
           model->BSIM4v4Lwlc = model->BSIM4v4Lwl;
        if (!model->BSIM4v4LminGiven)
           model->BSIM4v4Lmin = 0.0;
        if (!model->BSIM4v4LmaxGiven)
           model->BSIM4v4Lmax = 1.0;
        if (!model->BSIM4v4WintGiven)
           model->BSIM4v4Wint = 0.0;
        if (!model->BSIM4v4WlGiven)
           model->BSIM4v4Wl = 0.0;
        if (!model->BSIM4v4WlcGiven)
           model->BSIM4v4Wlc = model->BSIM4v4Wl;
        if (!model->BSIM4v4WlnGiven)
           model->BSIM4v4Wln = 1.0;
        if (!model->BSIM4v4WwGiven)
           model->BSIM4v4Ww = 0.0;
        if (!model->BSIM4v4WwcGiven)
           model->BSIM4v4Wwc = model->BSIM4v4Ww;
        if (!model->BSIM4v4WwnGiven)
           model->BSIM4v4Wwn = 1.0;
        if (!model->BSIM4v4WwlGiven)
           model->BSIM4v4Wwl = 0.0;
        if (!model->BSIM4v4WwlcGiven)
           model->BSIM4v4Wwlc = model->BSIM4v4Wwl;
        if (!model->BSIM4v4WminGiven)
           model->BSIM4v4Wmin = 0.0;
        if (!model->BSIM4v4WmaxGiven)
           model->BSIM4v4Wmax = 1.0;
        if (!model->BSIM4v4dwcGiven)
           model->BSIM4v4dwc = model->BSIM4v4Wint;
        if (!model->BSIM4v4dlcGiven)
           model->BSIM4v4dlc = model->BSIM4v4Lint;
        if (!model->BSIM4v4xlGiven)
           model->BSIM4v4xl = 0.0;
        if (!model->BSIM4v4xwGiven)
           model->BSIM4v4xw = 0.0;
        if (!model->BSIM4v4dlcigGiven)
           model->BSIM4v4dlcig = model->BSIM4v4Lint;
        if (!model->BSIM4v4dwjGiven)
           model->BSIM4v4dwj = model->BSIM4v4dwc;
        if (!model->BSIM4v4cfGiven)
           model->BSIM4v4cf = 2.0 * model->BSIM4v4epsrox * EPS0 / PI
                          * log(1.0 + 0.4e-6 / model->BSIM4v4toxe);

        if (!model->BSIM4v4xpartGiven)
            model->BSIM4v4xpart = 0.0;
        if (!model->BSIM4v4sheetResistanceGiven)
            model->BSIM4v4sheetResistance = 0.0;

        if (!model->BSIM4v4SunitAreaJctCapGiven)
            model->BSIM4v4SunitAreaJctCap = 5.0E-4;
        if (!model->BSIM4v4DunitAreaJctCapGiven)
            model->BSIM4v4DunitAreaJctCap = model->BSIM4v4SunitAreaJctCap;
        if (!model->BSIM4v4SunitLengthSidewallJctCapGiven)
            model->BSIM4v4SunitLengthSidewallJctCap = 5.0E-10;
        if (!model->BSIM4v4DunitLengthSidewallJctCapGiven)
            model->BSIM4v4DunitLengthSidewallJctCap = model->BSIM4v4SunitLengthSidewallJctCap;
        if (!model->BSIM4v4SunitLengthGateSidewallJctCapGiven)
            model->BSIM4v4SunitLengthGateSidewallJctCap = model->BSIM4v4SunitLengthSidewallJctCap ;
        if (!model->BSIM4v4DunitLengthGateSidewallJctCapGiven)
            model->BSIM4v4DunitLengthGateSidewallJctCap = model->BSIM4v4SunitLengthGateSidewallJctCap;
        if (!model->BSIM4v4SjctSatCurDensityGiven)
            model->BSIM4v4SjctSatCurDensity = 1.0E-4;
        if (!model->BSIM4v4DjctSatCurDensityGiven)
            model->BSIM4v4DjctSatCurDensity = model->BSIM4v4SjctSatCurDensity;
        if (!model->BSIM4v4SjctSidewallSatCurDensityGiven)
            model->BSIM4v4SjctSidewallSatCurDensity = 0.0;
        if (!model->BSIM4v4DjctSidewallSatCurDensityGiven)
            model->BSIM4v4DjctSidewallSatCurDensity = model->BSIM4v4SjctSidewallSatCurDensity;
        if (!model->BSIM4v4SjctGateSidewallSatCurDensityGiven)
            model->BSIM4v4SjctGateSidewallSatCurDensity = 0.0;
        if (!model->BSIM4v4DjctGateSidewallSatCurDensityGiven)
            model->BSIM4v4DjctGateSidewallSatCurDensity = model->BSIM4v4SjctGateSidewallSatCurDensity;
        if (!model->BSIM4v4SbulkJctPotentialGiven)
            model->BSIM4v4SbulkJctPotential = 1.0;
        if (!model->BSIM4v4DbulkJctPotentialGiven)
            model->BSIM4v4DbulkJctPotential = model->BSIM4v4SbulkJctPotential;
        if (!model->BSIM4v4SsidewallJctPotentialGiven)
            model->BSIM4v4SsidewallJctPotential = 1.0;
        if (!model->BSIM4v4DsidewallJctPotentialGiven)
            model->BSIM4v4DsidewallJctPotential = model->BSIM4v4SsidewallJctPotential;
        if (!model->BSIM4v4SGatesidewallJctPotentialGiven)
            model->BSIM4v4SGatesidewallJctPotential = model->BSIM4v4SsidewallJctPotential;
        if (!model->BSIM4v4DGatesidewallJctPotentialGiven)
            model->BSIM4v4DGatesidewallJctPotential = model->BSIM4v4SGatesidewallJctPotential;
        if (!model->BSIM4v4SbulkJctBotGradingCoeffGiven)
            model->BSIM4v4SbulkJctBotGradingCoeff = 0.5;
        if (!model->BSIM4v4DbulkJctBotGradingCoeffGiven)
            model->BSIM4v4DbulkJctBotGradingCoeff = model->BSIM4v4SbulkJctBotGradingCoeff;
        if (!model->BSIM4v4SbulkJctSideGradingCoeffGiven)
            model->BSIM4v4SbulkJctSideGradingCoeff = 0.33;
        if (!model->BSIM4v4DbulkJctSideGradingCoeffGiven)
            model->BSIM4v4DbulkJctSideGradingCoeff = model->BSIM4v4SbulkJctSideGradingCoeff;
        if (!model->BSIM4v4SbulkJctGateSideGradingCoeffGiven)
            model->BSIM4v4SbulkJctGateSideGradingCoeff = model->BSIM4v4SbulkJctSideGradingCoeff;
        if (!model->BSIM4v4DbulkJctGateSideGradingCoeffGiven)
            model->BSIM4v4DbulkJctGateSideGradingCoeff = model->BSIM4v4SbulkJctGateSideGradingCoeff;
        if (!model->BSIM4v4SjctEmissionCoeffGiven)
            model->BSIM4v4SjctEmissionCoeff = 1.0;
        if (!model->BSIM4v4DjctEmissionCoeffGiven)
            model->BSIM4v4DjctEmissionCoeff = model->BSIM4v4SjctEmissionCoeff;
        if (!model->BSIM4v4SjctTempExponentGiven)
            model->BSIM4v4SjctTempExponent = 3.0;
        if (!model->BSIM4v4DjctTempExponentGiven)
            model->BSIM4v4DjctTempExponent = model->BSIM4v4SjctTempExponent;
        switch (model->BSIM4v4intVersion) {
          case BSIM4vOLD: case BSIM4v21: case BSIM4v30:
            break;
          case BSIM4v40:
            if (!model->BSIM4v4jtssGiven)
                model->BSIM4v4jtss = 0.0;
            if (!model->BSIM4v4jtsdGiven)
                model->BSIM4v4jtsd = model->BSIM4v4jtss;
            if (!model->BSIM4v4jtsswsGiven)
                model->BSIM4v4jtssws = 0.0;
            if (!model->BSIM4v4jtsswdGiven)
                model->BSIM4v4jtsswd = model->BSIM4v4jtssws;
            if (!model->BSIM4v4jtsswgsGiven)
                model->BSIM4v4jtsswgs = 0.0;
            if (!model->BSIM4v4jtsswgdGiven)
                model->BSIM4v4jtsswgd = model->BSIM4v4jtsswgs;
            if (!model->BSIM4v4njtsGiven)
                model->BSIM4v4njts = 20.0;
            if (!model->BSIM4v4njtsswGiven)
                model->BSIM4v4njtssw = 20.0;
            if (!model->BSIM4v4njtsswgGiven)
                model->BSIM4v4njtsswg = 20.0;
            if (!model->BSIM4v4xtssGiven)
                model->BSIM4v4xtss = 0.02;
            if (!model->BSIM4v4xtsdGiven)
                model->BSIM4v4xtsd = model->BSIM4v4xtss;
            if (!model->BSIM4v4xtsswsGiven)
                model->BSIM4v4xtssws = 0.02;
            if (!model->BSIM4v4jtsswdGiven)
                model->BSIM4v4xtsswd = model->BSIM4v4xtssws;
            if (!model->BSIM4v4xtsswgsGiven)
                model->BSIM4v4xtsswgs = 0.02;
            if (!model->BSIM4v4xtsswgdGiven)
                model->BSIM4v4xtsswgd = model->BSIM4v4xtsswgs;
            if (!model->BSIM4v4tnjtsGiven)
                model->BSIM4v4tnjts = 0.0;
            if (!model->BSIM4v4tnjtsswGiven)
                model->BSIM4v4tnjtssw = 0.0;
            if (!model->BSIM4v4tnjtsswgGiven)
                model->BSIM4v4tnjtsswg = 0.0;
            if (!model->BSIM4v4vtssGiven)
                model->BSIM4v4vtss = 10.0;
            if (!model->BSIM4v4vtsdGiven)
                model->BSIM4v4vtsd = model->BSIM4v4vtss;
            if (!model->BSIM4v4vtsswsGiven)
                model->BSIM4v4vtssws = 10.0;
            if (!model->BSIM4v4vtsswdGiven)
                model->BSIM4v4vtsswd = model->BSIM4v4vtssws;
            if (!model->BSIM4v4vtsswgsGiven)
                model->BSIM4v4vtsswgs = 10.0;
            if (!model->BSIM4v4vtsswgdGiven)
                model->BSIM4v4vtsswgd = model->BSIM4v4vtsswgs;
            break;
          default: break;
        }
        if (!model->BSIM4v4oxideTrapDensityAGiven)
        {   if (model->BSIM4v4type == NMOS)
                model->BSIM4v4oxideTrapDensityA = 6.25e41;
            else
                model->BSIM4v4oxideTrapDensityA= 6.188e40;
        }
        if (!model->BSIM4v4oxideTrapDensityBGiven)
        {   if (model->BSIM4v4type == NMOS)
                model->BSIM4v4oxideTrapDensityB = 3.125e26;
            else
                model->BSIM4v4oxideTrapDensityB = 1.5e25;
        }
        if (!model->BSIM4v4oxideTrapDensityCGiven)
            model->BSIM4v4oxideTrapDensityC = 8.75e9;
        if (!model->BSIM4v4emGiven)
            model->BSIM4v4em = 4.1e7; /* V/m */
        if (!model->BSIM4v4efGiven)
            model->BSIM4v4ef = 1.0;
        if (!model->BSIM4v4afGiven)
            model->BSIM4v4af = 1.0;
        if (!model->BSIM4v4kfGiven)
            model->BSIM4v4kf = 0.0;
        switch (model->BSIM4v4intVersion) {
          case BSIM4vOLD: case BSIM4v21:
            break;
          case BSIM4v30: case BSIM4v40:
            /* stress effect */
            if (!model->BSIM4v4sarefGiven)
                model->BSIM4v4saref = 1e-6; /* m */
            if (!model->BSIM4v4sbrefGiven)
                model->BSIM4v4sbref = 1e-6;  /* m */
            if (!model->BSIM4v4wlodGiven)
                model->BSIM4v4wlod = 0;  /* m */
            if (!model->BSIM4v4ku0Given)
                model->BSIM4v4ku0 = 0; /* 1/m */
            if (!model->BSIM4v4kvsatGiven)
                model->BSIM4v4kvsat = 0;
            if (!model->BSIM4v4kvth0Given) /* m */
                model->BSIM4v4kvth0 = 0;
            if (!model->BSIM4v4tku0Given)
                model->BSIM4v4tku0 = 0;
            if (!model->BSIM4v4llodku0Given)
                model->BSIM4v4llodku0 = 0;
            if (!model->BSIM4v4wlodku0Given)
                model->BSIM4v4wlodku0 = 0;
            if (!model->BSIM4v4llodvthGiven)
                model->BSIM4v4llodvth = 0;
            if (!model->BSIM4v4wlodvthGiven)
                model->BSIM4v4wlodvth = 0;
            if (!model->BSIM4v4lku0Given)
                model->BSIM4v4lku0 = 0;
            if (!model->BSIM4v4wku0Given)
                model->BSIM4v4wku0 = 0;
            if (!model->BSIM4v4pku0Given)
                model->BSIM4v4pku0 = 0;
            if (!model->BSIM4v4lkvth0Given)
                model->BSIM4v4lkvth0 = 0;
            if (!model->BSIM4v4wkvth0Given)
                model->BSIM4v4wkvth0 = 0;
            if (!model->BSIM4v4pkvth0Given)
                model->BSIM4v4pkvth0 = 0;
            if (!model->BSIM4v4stk2Given)
                model->BSIM4v4stk2 = 0;
            if (!model->BSIM4v4lodk2Given)
                model->BSIM4v4lodk2 = 1.0;
            if (!model->BSIM4v4steta0Given)
                model->BSIM4v4steta0 = 0;
            if (!model->BSIM4v4lodeta0Given)
                model->BSIM4v4lodeta0 = 1.0;
            break;
          default: break;
        }
        DMCGeff = model->BSIM4v4dmcg - model->BSIM4v4dmcgt;
        DMCIeff = model->BSIM4v4dmci;
        DMDGeff = model->BSIM4v4dmdg - model->BSIM4v4dmcgt;

        /*
         * End processing models and begin to loop
         * through all the instances of the model
         */

        for (here = model->BSIM4v4instances; here != NULL ;
             here=here->BSIM4v4nextInstance)
        {
            /* allocate a chunk of the state vector */
            here->BSIM4v4states = *states;
            *states += BSIM4v4numStates;

            /* perform the parameter defaulting */
            if (!here->BSIM4v4lGiven)
                here->BSIM4v4l = 5.0e-6;
            if (!here->BSIM4v4wGiven)
                here->BSIM4v4w = 5.0e-6;
            if (!here->BSIM4v4mGiven)
                here->BSIM4v4m = 1.0;
            if (!here->BSIM4v4nfGiven)
                here->BSIM4v4nf = 1.0;
            if (!here->BSIM4v4minGiven)
                here->BSIM4v4min = 0; /* integer */
            if (!here->BSIM4v4icVDSGiven)
                here->BSIM4v4icVDS = 0.0;
            if (!here->BSIM4v4icVGSGiven)
                here->BSIM4v4icVGS = 0.0;
            if (!here->BSIM4v4icVBSGiven)
                here->BSIM4v4icVBS = 0.0;
            if (!here->BSIM4v4drainAreaGiven)
                here->BSIM4v4drainArea = 0.0;
            if (!here->BSIM4v4drainPerimeterGiven)
                here->BSIM4v4drainPerimeter = 0.0;
            if (!here->BSIM4v4drainSquaresGiven)
                here->BSIM4v4drainSquares = 1.0;
            if (!here->BSIM4v4sourceAreaGiven)
                here->BSIM4v4sourceArea = 0.0;
            if (!here->BSIM4v4sourcePerimeterGiven)
                here->BSIM4v4sourcePerimeter = 0.0;
            if (!here->BSIM4v4sourceSquaresGiven)
                here->BSIM4v4sourceSquares = 1.0;
            switch (model->BSIM4v4intVersion) {
              case BSIM4vOLD: case BSIM4v21:
                break;
              case BSIM4v30: case BSIM4v40:
                if (!here->BSIM4v4saGiven)
                    here->BSIM4v4sa = 0.0;
                if (!here->BSIM4v4sbGiven)
                    here->BSIM4v4sb = 0.0;
                if (!here->BSIM4v4sdGiven)
                    here->BSIM4v4sd = 0.0;
                break;
              default: break;
            }
            if (!here->BSIM4v4rbdbGiven)
                here->BSIM4v4rbdb = model->BSIM4v4rbdb; /* in ohm */
            if (!here->BSIM4v4rbsbGiven)
                here->BSIM4v4rbsb = model->BSIM4v4rbsb;
            if (!here->BSIM4v4rbpbGiven)
                here->BSIM4v4rbpb = model->BSIM4v4rbpb;
            if (!here->BSIM4v4rbpsGiven)
                here->BSIM4v4rbps = model->BSIM4v4rbps;
            if (!here->BSIM4v4rbpdGiven)
                here->BSIM4v4rbpd = model->BSIM4v4rbpd;


            /* Process instance model selectors, some
             * may override their global counterparts
             */
            if (!here->BSIM4v4rbodyModGiven)
                here->BSIM4v4rbodyMod = model->BSIM4v4rbodyMod;
            else if ((here->BSIM4v4rbodyMod != 0) && (here->BSIM4v4rbodyMod != 1))
            {   here->BSIM4v4rbodyMod = model->BSIM4v4rbodyMod;
                printf("Warning: rbodyMod has been set to its global value %d.\n",
                model->BSIM4v4rbodyMod);
            }

            if (!here->BSIM4v4rgateModGiven)
                here->BSIM4v4rgateMod = model->BSIM4v4rgateMod;
            else if ((here->BSIM4v4rgateMod != 0) && (here->BSIM4v4rgateMod != 1)
                && (here->BSIM4v4rgateMod != 2) && (here->BSIM4v4rgateMod != 3))
            {   here->BSIM4v4rgateMod = model->BSIM4v4rgateMod;
                printf("Warning: rgateMod has been set to its global value %d.\n",
                model->BSIM4v4rgateMod);
            }

            if (!here->BSIM4v4geoModGiven)
                here->BSIM4v4geoMod = model->BSIM4v4geoMod;
            if (!here->BSIM4v4rgeoModGiven)
                here->BSIM4v4rgeoMod = 0;
            if (!here->BSIM4v4trnqsModGiven)
                here->BSIM4v4trnqsMod = model->BSIM4v4trnqsMod;
            else if ((here->BSIM4v4trnqsMod != 0) && (here->BSIM4v4trnqsMod != 1))
            {   here->BSIM4v4trnqsMod = model->BSIM4v4trnqsMod;
                printf("Warning: trnqsMod has been set to its global value %d.\n",
                model->BSIM4v4trnqsMod);
            }

            if (!here->BSIM4v4acnqsModGiven)
                here->BSIM4v4acnqsMod = model->BSIM4v4acnqsMod;
            else if ((here->BSIM4v4acnqsMod != 0) && (here->BSIM4v4acnqsMod != 1))
            {   here->BSIM4v4acnqsMod = model->BSIM4v4acnqsMod;
                printf("Warning: acnqsMod has been set to its global value %d.\n",
                model->BSIM4v4acnqsMod);
            }

            /* process drain series resistance */
            createNode = 0;
            if ( (model->BSIM4v4rdsMod != 0)
                            || (model->BSIM4v4tnoiMod != 0 && noiseAnalGiven))
            {
               createNode = 1;
            } else if (model->BSIM4v4sheetResistance > 0)
            {
                     if (here->BSIM4v4drainSquaresGiven
                                       && here->BSIM4v4drainSquares > 0)
                     {
                          createNode = 1;
                     } else if (!here->BSIM4v4drainSquaresGiven
                                       && (here->BSIM4v4rgeoMod != 0))
                     {
                          BSIM4v4RdseffGeo(here->BSIM4v4nf, here->BSIM4v4geoMod,
                                  here->BSIM4v4rgeoMod, here->BSIM4v4min,
                                  here->BSIM4v4w, model->BSIM4v4sheetResistance,
                                  DMCGeff, DMCIeff, DMDGeff, 0, &Rtot);
                          if(Rtot > 0)
                             createNode = 1;
                     }
            }
            if ( createNode != 0 )
            {   if (here->BSIM4v4dNodePrime == 0)
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v4name,"drain");
                if(error) return(error);
                here->BSIM4v4dNodePrime = tmp->number;
            }
            }
            else
            {   here->BSIM4v4dNodePrime = here->BSIM4v4dNode;
            }

            /* process source series resistance */
            createNode = 0;
            if ( (model->BSIM4v4rdsMod != 0)
                            || (model->BSIM4v4tnoiMod != 0 && noiseAnalGiven))
            {
               createNode = 1;
            } else if (model->BSIM4v4sheetResistance > 0)
            {
                     if (here->BSIM4v4sourceSquaresGiven
                                        && here->BSIM4v4sourceSquares > 0)
                     {
                          createNode = 1;
                     } else if (!here->BSIM4v4sourceSquaresGiven
                                        && (here->BSIM4v4rgeoMod != 0))
                     {
                          BSIM4v4RdseffGeo(here->BSIM4v4nf, here->BSIM4v4geoMod,
                                  here->BSIM4v4rgeoMod, here->BSIM4v4min,
                                  here->BSIM4v4w, model->BSIM4v4sheetResistance,
                                  DMCGeff, DMCIeff, DMDGeff, 1, &Rtot);
                          if(Rtot > 0)
                             createNode = 1;
                     }
            }
            if ( createNode != 0 )
            {   if (here->BSIM4v4sNodePrime == 0)
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v4name,"source");
                if(error) return(error);
                here->BSIM4v4sNodePrime = tmp->number;
            }
            }
            else
                here->BSIM4v4sNodePrime = here->BSIM4v4sNode;

            if (here->BSIM4v4rgateMod > 0)
            {   if (here->BSIM4v4gNodePrime == 0)
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v4name,"gate");
                if(error) return(error);
                   here->BSIM4v4gNodePrime = tmp->number;
            }
            }
            else
                here->BSIM4v4gNodePrime = here->BSIM4v4gNodeExt;

            if (here->BSIM4v4rgateMod == 3)
            {   if (here->BSIM4v4gNodeMid == 0)
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v4name,"midgate");
                if(error) return(error);
                   here->BSIM4v4gNodeMid = tmp->number;
            }
            }
            else
                here->BSIM4v4gNodeMid = here->BSIM4v4gNodeExt;


            /* internal body nodes for body resistance model */
            if (here->BSIM4v4rbodyMod)
            {   if (here->BSIM4v4dbNode == 0)
                {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v4name,"dbody");
                    if(error) return(error);
                    here->BSIM4v4dbNode = tmp->number;
                }
                if (here->BSIM4v4bNodePrime == 0)
                {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v4name,"body");
                    if(error) return(error);
                    here->BSIM4v4bNodePrime = tmp->number;
                }
                if (here->BSIM4v4sbNode == 0)
                {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v4name,"sbody");
                    if(error) return(error);
                    here->BSIM4v4sbNode = tmp->number;
                }
            }
            else
                here->BSIM4v4dbNode = here->BSIM4v4bNodePrime = here->BSIM4v4sbNode
                                  = here->BSIM4v4bNode;

            /* NQS node */
            if (here->BSIM4v4trnqsMod)
            {   if (here->BSIM4v4qNode == 0)
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM4v4name,"charge");
                if(error) return(error);
                here->BSIM4v4qNode = tmp->number;
            }
            }
            else
                here->BSIM4v4qNode = 0;

/* set Sparse Matrix Pointers
 * macro to make elements with built-in out-of-memory test */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
}

            TSTALLOC(BSIM4v4DPbpPtr, BSIM4v4dNodePrime, BSIM4v4bNodePrime)
            TSTALLOC(BSIM4v4GPbpPtr, BSIM4v4gNodePrime, BSIM4v4bNodePrime)
            TSTALLOC(BSIM4v4SPbpPtr, BSIM4v4sNodePrime, BSIM4v4bNodePrime)

            TSTALLOC(BSIM4v4BPdpPtr, BSIM4v4bNodePrime, BSIM4v4dNodePrime)
            TSTALLOC(BSIM4v4BPgpPtr, BSIM4v4bNodePrime, BSIM4v4gNodePrime)
            TSTALLOC(BSIM4v4BPspPtr, BSIM4v4bNodePrime, BSIM4v4sNodePrime)
            TSTALLOC(BSIM4v4BPbpPtr, BSIM4v4bNodePrime, BSIM4v4bNodePrime)

            TSTALLOC(BSIM4v4DdPtr, BSIM4v4dNode, BSIM4v4dNode)
            TSTALLOC(BSIM4v4GPgpPtr, BSIM4v4gNodePrime, BSIM4v4gNodePrime)
            TSTALLOC(BSIM4v4SsPtr, BSIM4v4sNode, BSIM4v4sNode)
            TSTALLOC(BSIM4v4DPdpPtr, BSIM4v4dNodePrime, BSIM4v4dNodePrime)
            TSTALLOC(BSIM4v4SPspPtr, BSIM4v4sNodePrime, BSIM4v4sNodePrime)
            TSTALLOC(BSIM4v4DdpPtr, BSIM4v4dNode, BSIM4v4dNodePrime)
            TSTALLOC(BSIM4v4GPdpPtr, BSIM4v4gNodePrime, BSIM4v4dNodePrime)
            TSTALLOC(BSIM4v4GPspPtr, BSIM4v4gNodePrime, BSIM4v4sNodePrime)
            TSTALLOC(BSIM4v4SspPtr, BSIM4v4sNode, BSIM4v4sNodePrime)
            TSTALLOC(BSIM4v4DPspPtr, BSIM4v4dNodePrime, BSIM4v4sNodePrime)
            TSTALLOC(BSIM4v4DPdPtr, BSIM4v4dNodePrime, BSIM4v4dNode)
            TSTALLOC(BSIM4v4DPgpPtr, BSIM4v4dNodePrime, BSIM4v4gNodePrime)
            TSTALLOC(BSIM4v4SPgpPtr, BSIM4v4sNodePrime, BSIM4v4gNodePrime)
            TSTALLOC(BSIM4v4SPsPtr, BSIM4v4sNodePrime, BSIM4v4sNode)
            TSTALLOC(BSIM4v4SPdpPtr, BSIM4v4sNodePrime, BSIM4v4dNodePrime)

            TSTALLOC(BSIM4v4QqPtr, BSIM4v4qNode, BSIM4v4qNode)
            TSTALLOC(BSIM4v4QbpPtr, BSIM4v4qNode, BSIM4v4bNodePrime)
            TSTALLOC(BSIM4v4QdpPtr, BSIM4v4qNode, BSIM4v4dNodePrime)
            TSTALLOC(BSIM4v4QspPtr, BSIM4v4qNode, BSIM4v4sNodePrime)
            TSTALLOC(BSIM4v4QgpPtr, BSIM4v4qNode, BSIM4v4gNodePrime)
            TSTALLOC(BSIM4v4DPqPtr, BSIM4v4dNodePrime, BSIM4v4qNode)
            TSTALLOC(BSIM4v4SPqPtr, BSIM4v4sNodePrime, BSIM4v4qNode)
            TSTALLOC(BSIM4v4GPqPtr, BSIM4v4gNodePrime, BSIM4v4qNode)

            if (here->BSIM4v4rgateMod != 0)
            {   TSTALLOC(BSIM4v4GEgePtr, BSIM4v4gNodeExt, BSIM4v4gNodeExt)
                TSTALLOC(BSIM4v4GEgpPtr, BSIM4v4gNodeExt, BSIM4v4gNodePrime)
                TSTALLOC(BSIM4v4GPgePtr, BSIM4v4gNodePrime, BSIM4v4gNodeExt)
                TSTALLOC(BSIM4v4GEdpPtr, BSIM4v4gNodeExt, BSIM4v4dNodePrime)
                TSTALLOC(BSIM4v4GEspPtr, BSIM4v4gNodeExt, BSIM4v4sNodePrime)
                TSTALLOC(BSIM4v4GEbpPtr, BSIM4v4gNodeExt, BSIM4v4bNodePrime)

                TSTALLOC(BSIM4v4GMdpPtr, BSIM4v4gNodeMid, BSIM4v4dNodePrime)
                TSTALLOC(BSIM4v4GMgpPtr, BSIM4v4gNodeMid, BSIM4v4gNodePrime)
                TSTALLOC(BSIM4v4GMgmPtr, BSIM4v4gNodeMid, BSIM4v4gNodeMid)
                TSTALLOC(BSIM4v4GMgePtr, BSIM4v4gNodeMid, BSIM4v4gNodeExt)
                TSTALLOC(BSIM4v4GMspPtr, BSIM4v4gNodeMid, BSIM4v4sNodePrime)
                TSTALLOC(BSIM4v4GMbpPtr, BSIM4v4gNodeMid, BSIM4v4bNodePrime)
                TSTALLOC(BSIM4v4DPgmPtr, BSIM4v4dNodePrime, BSIM4v4gNodeMid)
                TSTALLOC(BSIM4v4GPgmPtr, BSIM4v4gNodePrime, BSIM4v4gNodeMid)
                TSTALLOC(BSIM4v4GEgmPtr, BSIM4v4gNodeExt, BSIM4v4gNodeMid)
                TSTALLOC(BSIM4v4SPgmPtr, BSIM4v4sNodePrime, BSIM4v4gNodeMid)
                TSTALLOC(BSIM4v4BPgmPtr, BSIM4v4bNodePrime, BSIM4v4gNodeMid)
            }

            if (here->BSIM4v4rbodyMod)
            {   TSTALLOC(BSIM4v4DPdbPtr, BSIM4v4dNodePrime, BSIM4v4dbNode)
                TSTALLOC(BSIM4v4SPsbPtr, BSIM4v4sNodePrime, BSIM4v4sbNode)

                TSTALLOC(BSIM4v4DBdpPtr, BSIM4v4dbNode, BSIM4v4dNodePrime)
                TSTALLOC(BSIM4v4DBdbPtr, BSIM4v4dbNode, BSIM4v4dbNode)
                TSTALLOC(BSIM4v4DBbpPtr, BSIM4v4dbNode, BSIM4v4bNodePrime)
                TSTALLOC(BSIM4v4DBbPtr, BSIM4v4dbNode, BSIM4v4bNode)

                TSTALLOC(BSIM4v4BPdbPtr, BSIM4v4bNodePrime, BSIM4v4dbNode)
                TSTALLOC(BSIM4v4BPbPtr, BSIM4v4bNodePrime, BSIM4v4bNode)
                TSTALLOC(BSIM4v4BPsbPtr, BSIM4v4bNodePrime, BSIM4v4sbNode)

                TSTALLOC(BSIM4v4SBspPtr, BSIM4v4sbNode, BSIM4v4sNodePrime)
                TSTALLOC(BSIM4v4SBbpPtr, BSIM4v4sbNode, BSIM4v4bNodePrime)
                TSTALLOC(BSIM4v4SBbPtr, BSIM4v4sbNode, BSIM4v4bNode)
                TSTALLOC(BSIM4v4SBsbPtr, BSIM4v4sbNode, BSIM4v4sbNode)

                TSTALLOC(BSIM4v4BdbPtr, BSIM4v4bNode, BSIM4v4dbNode)
                TSTALLOC(BSIM4v4BbpPtr, BSIM4v4bNode, BSIM4v4bNodePrime)
                TSTALLOC(BSIM4v4BsbPtr, BSIM4v4bNode, BSIM4v4sbNode)
                TSTALLOC(BSIM4v4BbPtr, BSIM4v4bNode, BSIM4v4bNode)
            }

            if (model->BSIM4v4rdsMod)
            {   TSTALLOC(BSIM4v4DgpPtr, BSIM4v4dNode, BSIM4v4gNodePrime)
                TSTALLOC(BSIM4v4DspPtr, BSIM4v4dNode, BSIM4v4sNodePrime)
                TSTALLOC(BSIM4v4DbpPtr, BSIM4v4dNode, BSIM4v4bNodePrime)
                TSTALLOC(BSIM4v4SdpPtr, BSIM4v4sNode, BSIM4v4dNodePrime)
                TSTALLOC(BSIM4v4SgpPtr, BSIM4v4sNode, BSIM4v4gNodePrime)
                TSTALLOC(BSIM4v4SbpPtr, BSIM4v4sNode, BSIM4v4bNodePrime)
            }
        }
    }
    return(OK);
}

int
BSIM4v4unsetup(
    GENmodel *inModel,
    CKTcircuit *ckt)
{
#ifndef HAS_BATCHSIM
    BSIM4v4model *model;
    BSIM4v4instance *here;

    for (model = (BSIM4v4model *)inModel; model != NULL;
            model = model->BSIM4v4nextModel)
    {
        for (here = model->BSIM4v4instances; here != NULL;
                here=here->BSIM4v4nextInstance)
        {
            if (here->BSIM4v4dNodePrime
                    && here->BSIM4v4dNodePrime != here->BSIM4v4dNode)
            {
                CKTdltNNum(ckt, here->BSIM4v4dNodePrime);
                here->BSIM4v4dNodePrime = 0;
            }
            if (here->BSIM4v4sNodePrime
                    && here->BSIM4v4sNodePrime != here->BSIM4v4sNode)
            {
                CKTdltNNum(ckt, here->BSIM4v4sNodePrime);
                here->BSIM4v4sNodePrime = 0;
            }
        }
    }
#endif
    return OK;
}
