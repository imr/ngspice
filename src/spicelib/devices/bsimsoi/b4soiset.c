/***  B4SOI 12/16/2010 Released by Tanvir Morshed   ***/


/**********
 * Copyright 2010 Regents of the University of California.  All rights reserved.
 * Authors: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
 * Authors: 1999-2004 Pin Su, Hui Wan, Wei Jin, b3soiset.c
 * Authors: 2005- Hui Wan, Xuemei Xi, Ali Niknejad, Chenming Hu.
 * Authors: 2009- Wenwei Yang, Chung-Hsun Lin, Ali Niknejad, Chenming Hu.
 * Authors: 2009- Tanvir Morshed, Ali Niknejad, Chenming Hu.
 * File: b4soiset.c
 * Modified by Hui Wan, Xuemei Xi 11/30/2005
 * Modified by Wenwei Yang, Chung-Hsun Lin, Darsen Lu 03/06/2009
 * Modified by Tanvir Morshed 09/22/2009
 * Modified by Tanvir Morshed 12/31/2009
 * Modified by Tanvir Morshed 04/27/2010
 * Modified by Tanvir Morshed 12/16/2010
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "b4soidef.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#define SMOOTHFACTOR 0.1
#define EPSOX 3.453133e-11
#define EPSSI 1.03594e-10
#define PI 3.141592654
#define Charge_q 1.60219e-19
#define Meter2Micron 1.0e6
#define EPS0 8.85418e-12

double epsrox, toxe, epssub;
double NchMax; /* v4.4  */

int
B4SOIsetup(
SMPmatrix *matrix,
GENmodel *inModel,
CKTcircuit *ckt,
int *states)
{
register B4SOImodel *model = (B4SOImodel*)inModel;
register B4SOIinstance *here;
int error;
CKTnode *tmp;

double Cboxt;

/* v3.2 */
double Vbs0t, Qsi;

#ifdef USE_OMP
int idx, InstCount;
B4SOIinstance **InstArray;
#endif

    /*  loop through all the B4SOI device models */
    for( ; model != NULL; model = B4SOInextModel(model))
    {
/* Default value Processing for B4SOI MOSFET Models */

        if (!model->B4SOItypeGiven)
            model->B4SOItype = NMOS;
        if (!model->B4SOImobModGiven)
            model->B4SOImobMod = 1;
        if (!model->B4SOIbinUnitGiven)
            model->B4SOIbinUnit = 1;
        if (!model->B4SOIparamChkGiven)
            model->B4SOIparamChk = 0;
        if (!model->B4SOIcapModGiven)
            model->B4SOIcapMod = 2;
        if (!model->B4SOIiiiModGiven)                                   /* Bug fix #7 Jun 09 'iiimod' with default value added */
            model->B4SOIiiiMod = 0;

        if (!model->B4SOImtrlModGiven)
            model->B4SOImtrlMod = 0; /*4.1*/
        if (!model->B4SOIvgstcvModGiven)
            /*model->B4SOIvgstcvMod = 0;                                v4.2 Bugfix */
            model->B4SOIvgstcvMod = 1;
        if (!model->B4SOIgidlModGiven)
            model->B4SOIgidlMod = 0;
        if (!model->B4SOIeotGiven)
            model->B4SOIeot = 100.0e-10;
        if (!model->B4SOIepsroxGiven)
            model->B4SOIepsrox = 3.9;
        if (!model->B4SOIepsrsubGiven)
            model->B4SOIepsrsub = 11.7;
        if (!model->B4SOIni0subGiven)
            model->B4SOIni0sub = 1.45e10;   /* unit 1/cm3 */
        if (!model->B4SOIbg0subGiven)
            model->B4SOIbg0sub =  1.16;     /* unit eV */
        if (!model->B4SOItbgasubGiven)
            model->B4SOItbgasub = 7.02e-4;
        if (!model->B4SOItbgbsubGiven)
            model->B4SOItbgbsub = 1108.0;
        if (!model->B4SOIleffeotGiven)
            model->B4SOIleffeot = 1.0;
        if (!model->B4SOIweffeotGiven)
            model->B4SOIweffeot = 10.0;
        if (!model->B4SOIvddeotGiven)
            model->B4SOIvddeot = (model->B4SOItype == NMOS) ? 1.5 : -1.5;
        if (!model->B4SOItempeotGiven)
                    model->B4SOItempeot = 300.15;
        if (!model->B4SOIadosGiven)
            model->B4SOIados = 1.0;
        if (!model->B4SOIbdosGiven)
            model->B4SOIbdos = 1.0;
        if (!model->B4SOIepsrgateGiven)
                model->B4SOIepsrgate = 11.7;
        if (!model->B4SOIphigGiven)
                model->B4SOIphig = 4.05;
        if (!model->B4SOIeasubGiven)
            model->B4SOIeasub = 4.05;

/*        if (!model->B4SOInoiModGiven)
            model->B4SOInoiMod = 1;     v3.2 */
        if (!model->B4SOIshModGiven)
            model->B4SOIshMod = 0;
        if (!model->B4SOIversionGiven)
            model->B4SOIversion = 4.4;
        if (!model->B4SOItoxGiven)
            model->B4SOItox = 100.0e-10;
       /*model->B4SOIcox = 3.453133e-11 / model->B4SOItox;*/
           if(model->B4SOImtrlMod)
                 {
             epsrox = 3.9;
             toxe = model->B4SOIeot;
             epssub = EPS0 * model->B4SOIepsrsub;
         /*model->B4SOIcox = 3.453133e-11 / model->B4SOItox;*/
                 model->B4SOIcox = epsrox * EPS0 / toxe;
                 }
              else
                 {
             epsrox = model->B4SOIepsrox;
             toxe = model->B4SOItox;
             epssub = EPSSI;
                 /*model->B4SOIcox = epsrox * EPS0 / toxe;*/
                 model->B4SOIcox = 3.453133e-11 / model->B4SOItox;
                 }


            if (!model->B4SOItoxpGiven)
            model->B4SOItoxp = model->B4SOItox;



        if (!model->B4SOItoxmGiven)
            model->B4SOItoxm = model->B4SOItox; /* v3.2 */

/* v3.2 */
        if (!model->B4SOIsoiModGiven)
            model->B4SOIsoiMod = 0;
        else if ((model->B4SOIsoiMod != 0) && (model->B4SOIsoiMod != 1)
            && (model->B4SOIsoiMod != 2) && (model->B4SOIsoiMod != 3))
        {   model->B4SOIsoiMod = 0;
            printf("Warning: soiMod has been set to its default value: 0.\n");
        }

        /* v3.1 added for RF */
        if (!model->B4SOIrgateModGiven)
            model->B4SOIrgateMod = 0;
        else if ((model->B4SOIrgateMod != 0) && (model->B4SOIrgateMod != 1)
            && (model->B4SOIrgateMod != 2) && (model->B4SOIrgateMod != 3))
        {   model->B4SOIrgateMod = 0;
            printf("Warning: rgateMod has been set to its default value: 0.\n");
        }
        /* v3.1 added for RF end */

        /* v3.2 for noise */
        if (!model->B4SOIfnoiModGiven)
            model->B4SOIfnoiMod = 1;
        else if ((model->B4SOIfnoiMod != 0) && (model->B4SOIfnoiMod != 1))
        {    model->B4SOIfnoiMod = 1;
             printf("Waring: fnoiMod has been set to default value:1.\n");
        }

        if (!model->B4SOItnoiModGiven)
            model->B4SOItnoiMod = 0;
        else if ((model->B4SOItnoiMod != 0) && (model->B4SOItnoiMod != 1)&& (model->B4SOItnoiMod != 2))
        {    model->B4SOItnoiMod = 0;
             printf("Waring: tnoiMod has been set to default value:0.\n");
        }

        if (!model->B4SOItnoiaGiven)
            model->B4SOItnoia = 1.5;
        if (!model->B4SOItnoibGiven)
            model->B4SOItnoib = 3.5;
        if (!model->B4SOIrnoiaGiven)
            model->B4SOIrnoia = 0.577;
        if (!model->B4SOIrnoibGiven)
            model->B4SOIrnoib = 0.37;
        if (!model->B4SOIntnoiGiven)
            model->B4SOIntnoi = 1.0;
        /* v3.2 for noise end */

        /* v4.0 */
        if (!model->B4SOIrdsModGiven)
            model->B4SOIrdsMod = 0;
        else if ((model->B4SOIrdsMod != 0) && (model->B4SOIrdsMod != 1))
        {   model->B4SOIrdsMod = 0;
            printf("Warning: rdsMod has been set to its default value: 0.\n");
        }

        if (!model->B4SOIrbodyModGiven)
            model->B4SOIrbodyMod = 0;
        else if ((model->B4SOIrbodyMod != 0) && (model->B4SOIrbodyMod != 1))
        {   model->B4SOIrbodyMod = 0;
            printf("Warning: rbodyMod has been set to its default value: 0.\n");
        }
        if (!model->B4SOIgbminGiven)
            model->B4SOIgbmin = 1.0e-12; /* in mho */
        if (!model->B4SOIrbdbGiven)
            model->B4SOIrbdb = 50.0; /* in ohm */
        if (!model->B4SOIrbsbGiven)
            model->B4SOIrbsb = 50.0; /* in ohm */

        /* v4.0 end */


/* v2.2.3 */
        if (!model->B4SOIdtoxcvGiven)
            model->B4SOIdtoxcv = 0.0;

        if (!model->B4SOIcdscGiven)
            model->B4SOIcdsc = 2.4e-4;   /* unit Q/V/m^2  */
        if (!model->B4SOIcdscbGiven)
            model->B4SOIcdscb = 0.0;   /* unit Q/V/m^2  */
        if (!model->B4SOIcdscdGiven)
            model->B4SOIcdscd = 0.0;   /* unit Q/V/m^2  */
        if (!model->B4SOIcitGiven)
            model->B4SOIcit = 0.0;   /* unit Q/V/m^2  */
        if (!model->B4SOInfactorGiven)
            model->B4SOInfactor = 1;
        if (!model->B4SOIvsatGiven)
            model->B4SOIvsat = 8.0e4;    /* unit m/s */
        if (!model->B4SOIatGiven)
            model->B4SOIat = 3.3e4;    /* unit m/s */
        if (!model->B4SOIa0Given)
            model->B4SOIa0 = 1.0;
        if (!model->B4SOIagsGiven)
            model->B4SOIags = 0.0;
        if (!model->B4SOIa1Given)
            model->B4SOIa1 = 0.0;
        if (!model->B4SOIa2Given)
            model->B4SOIa2 = 1.0;
        if (!model->B4SOIketaGiven)
            model->B4SOIketa = -0.6;    /* unit  / V */
        if (!model->B4SOInsubGiven)
            model->B4SOInsub = 6.0e16;   /* unit 1/cm3 */
        if (!model->B4SOInpeakGiven)
            model->B4SOInpeak = 1.7e17;   /* unit 1/cm3 */
        if (!model->B4SOIngateGiven)
            model->B4SOIngate = 0;   /* unit 1/cm3 */
        if (!model->B4SOInsdGiven)
            model->B4SOInsd = 1.0e20;
        if (!model->B4SOIvbmGiven)
            model->B4SOIvbm = -3.0;
        if (!model->B4SOIxtGiven)
            model->B4SOIxt = 1.55e-7;
        if (!model->B4SOIkt1Given)
            model->B4SOIkt1 = -0.11;      /* unit V */
        if (!model->B4SOIkt1lGiven)
            model->B4SOIkt1l = 0.0;      /* unit V*m */
        if (!model->B4SOIkt2Given)
            model->B4SOIkt2 = 0.022;      /* No unit */
        if (!model->B4SOIk3Given)
            model->B4SOIk3 = 0.0;
        if (!model->B4SOIk3bGiven)
            model->B4SOIk3b = 0.0;
        if (!model->B4SOIw0Given)
            model->B4SOIw0 = 2.5e-6;
        if (!model->B4SOIlpebGiven)
            model->B4SOIlpeb = 0.0;
        if (!model->B4SOIdvt0Given)
            model->B4SOIdvt0 = 2.2;
        if (!model->B4SOIdvt1Given)
            model->B4SOIdvt1 = 0.53;
        if (!model->B4SOIdvt2Given)
            model->B4SOIdvt2 = -0.032;   /* unit 1 / V */

        if (!model->B4SOIdvt0wGiven)
            model->B4SOIdvt0w = 0.0;
        if (!model->B4SOIdvt1wGiven)
            model->B4SOIdvt1w = 5.3e6;
        if (!model->B4SOIdvt2wGiven)
            model->B4SOIdvt2w = -0.032;

        if (!model->B4SOIdroutGiven)
            model->B4SOIdrout = 0.56;
        if (!model->B4SOIdsubGiven)
            model->B4SOIdsub = model->B4SOIdrout;
        if (!model->B4SOIvth0Given)
            model->B4SOIvth0 = (model->B4SOItype == NMOS) ? 0.7 : -0.7;
        if (!model->B4SOIvfbGiven)
            model->B4SOIvfb = -1.0;       /* v4.1 */
        if (!model->B4SOIuaGiven)
            model->B4SOIua = 2.25e-9;      /* unit m/V */
        if (!model->B4SOIua1Given)
            model->B4SOIua1 = 4.31e-9;      /* unit m/V */
        if (!model->B4SOIubGiven)
            model->B4SOIub = 5.87e-19;     /* unit (m/V)**2 */
        if (!model->B4SOIub1Given)
            model->B4SOIub1 = -7.61e-18;     /* unit (m/V)**2 */
        if (!model->B4SOIucGiven)
            model->B4SOIuc = (model->B4SOImobMod == 3) ? -0.0465 : -0.0465e-9;
        if (!model->B4SOIuc1Given)
            model->B4SOIuc1 = (model->B4SOImobMod == 3) ? -0.056 : -0.056e-9;
        if (!model->B4SOIu0Given)
            model->B4SOIu0 = (model->B4SOItype == NMOS) ? 0.067 : 0.025;
        if (!model->B4SOIuteGiven)
            model->B4SOIute = -1.5;

                /*4.1           mobmod =4       */
         if (!model->B4SOIudGiven)
                    model->B4SOIud = 0.0;
                 if (!model->B4SOIludGiven)
                    model->B4SOIlud = 0.0;
                 if (!model->B4SOIwudGiven)
                    model->B4SOIwud = 0.0;
         if (!model->B4SOIpudGiven)
                  /*  model->B4SOIpud1 = 0.0; */ /*Bug fix # 33 Jul 09 */
                    model->B4SOIpud = 0.0;
         if (!model->B4SOIud1Given)
                    model->B4SOIud1 = 0.0;
                 if (!model->B4SOIlud1Given)
                    model->B4SOIlud1 = 0.0;
                 if (!model->B4SOIwud1Given)
                    model->B4SOIwud1 = 0.0;
         if (!model->B4SOIpud1Given)
                    model->B4SOIpud1 = 0.0;
         if (!model->B4SOIeuGiven)
            model->B4SOIeu = (model->B4SOItype == NMOS) ? 1.67 : 1.0;
         if (!model->B4SOIleuGiven)
            model->B4SOIleu = 0.0;
         if (!model->B4SOIweuGiven)
            model->B4SOIweu = 0.0;
         if (!model->B4SOIpeuGiven)
            model->B4SOIpeu = 0.0;
                if (!model->B4SOIucsGiven)
            model->B4SOIucs = (model->B4SOItype == NMOS) ? 1.67 : 1.0;
                if (!model->B4SOIlucsGiven)
            model->B4SOIlucs =0.0;
                if (!model->B4SOIwucsGiven)
            model->B4SOIwucs =0.0;
                if (!model->B4SOIpucsGiven)
            model->B4SOIpucs =0.0;
            if (!model->B4SOIucsteGiven)
                    model->B4SOIucste = -4.775e-3;
            if (!model->B4SOIlucsteGiven)
                    model->B4SOIlucste = 0.0;
        if (!model->B4SOIwucsteGiven)
                    model->B4SOIwucste = 0.0;
        if (!model->B4SOIpucsteGiven)
                    model->B4SOIpucste = 0.0;

        if (!model->B4SOIvoffGiven)
            model->B4SOIvoff = -0.08;
        if (!model->B4SOIdeltaGiven)
           model->B4SOIdelta = 0.01;
        if (!model->B4SOIrdswGiven)
            model->B4SOIrdsw = 100;
        if (!model->B4SOIrswGiven)      /* v4.0 */
            model->B4SOIrsw = 50;
        if (!model->B4SOIrdwGiven)      /* v4.0 */
            model->B4SOIrdw = 50;
        if (!model->B4SOIrswminGiven)   /* v4.0 */
            model->B4SOIrswmin = 0.0;
        if (!model->B4SOIrdwminGiven)   /* v4.0 */
            model->B4SOIrdwmin = 0.0;
        if (!model->B4SOIprwgGiven)
            model->B4SOIprwg = 0.0;      /* unit 1/V */
        if (!model->B4SOIprwbGiven)
            model->B4SOIprwb = 0.0;
        if (!model->B4SOIprtGiven)
            model->B4SOIprt = 0.0;
        if (!model->B4SOIeta0Given)
            model->B4SOIeta0 = 0.08;      /* no unit  */
        if (!model->B4SOIetabGiven)
            model->B4SOIetab = -0.07;      /* unit  1/V */
        if (!model->B4SOIpclmGiven)
            model->B4SOIpclm = 1.3;      /* no unit  */
        if (!model->B4SOIpdibl1Given)
            model->B4SOIpdibl1 = .39;    /* no unit  */
        if (!model->B4SOIpdibl2Given)
            model->B4SOIpdibl2 = 0.0086;    /* no unit  */
        if (!model->B4SOIpdiblbGiven)
            model->B4SOIpdiblb = 0.0;    /* 1/V  */
        if (!model->B4SOIpvagGiven)
            model->B4SOIpvag = 0.0;
        if (!model->B4SOIwrGiven)
            model->B4SOIwr = 1.0;
        if (!model->B4SOIdwgGiven)
            model->B4SOIdwg = 0.0;
        if (!model->B4SOIdwbGiven)
            model->B4SOIdwb = 0.0;
        if (!model->B4SOIb0Given)
            model->B4SOIb0 = 0.0;
        if (!model->B4SOIb1Given)
            model->B4SOIb1 = 0.0;
        if (!model->B4SOIalpha0Given)
            model->B4SOIalpha0 = 0.0;

        if (!model->B4SOIcgslGiven)
            model->B4SOIcgsl = 0.0;
        if (!model->B4SOIcgdlGiven)
            model->B4SOIcgdl = 0.0;
        if (!model->B4SOIckappaGiven)
            model->B4SOIckappa = 0.6;
        if (!model->B4SOIclcGiven)
            model->B4SOIclc = 0.1e-7;
        if (!model->B4SOIcleGiven)
            model->B4SOIcle = 0.0;
        if (!model->B4SOItboxGiven)
            model->B4SOItbox = 3e-7;
        if (!model->B4SOItsiGiven)
            model->B4SOItsi = 1e-7;
                if (!model->B4SOIetsiGiven)
            model->B4SOIetsi = 1e-7;
        if (!model->B4SOIxjGiven)
            model->B4SOIxj = model->B4SOItsi;
        if (!model->B4SOIrbodyGiven)
            model->B4SOIrbody = 0.0;
        if (!model->B4SOIrbshGiven)
            model->B4SOIrbsh = 0.0;
        if (!model->B4SOIrth0Given)
            model->B4SOIrth0 = 0;

/* v3.0 bug fix */
        if (!model->B4SOIcth0Given)
            model->B4SOIcth0 = 1e-5;
        if (!model->B4SOIcfrcoeffGiven)  /* v4.4 */
            model->B4SOIcfrcoeff = 1.0;

        if (!model->B4SOIagidlGiven)
            model->B4SOIagidl = 0.0;
        if (!model->B4SOIbgidlGiven)
            model->B4SOIbgidl = 2.3e9;  /* v4.0 */
        if (!model->B4SOIcgidlGiven)    /* v4.0 */
            model->B4SOIcgidl = 0.5;    /* v4.2 default value changed from 0 to 0.5 */
                if (!model->B4SOIrgidlGiven)    /* v4.1 */
            model->B4SOIrgidl = 1.0;
            if (!model->B4SOIkgidlGiven)        /* v4.1 */
            model->B4SOIkgidl = 0.0;
                if (!model->B4SOIfgidlGiven)    /* v4.1 */
            model->B4SOIfgidl = 0.0;
                        if (!model->B4SOIagislGiven)
            model->B4SOIagisl = model->B4SOIagidl;
        if (!model->B4SOIbgislGiven)
            model->B4SOIbgisl = model->B4SOIbgidl;      /* v4.0 */
        if (!model->B4SOIcgislGiven)    /* v4.0 */
            model->B4SOIcgisl = model->B4SOIcgidl;
                if (!model->B4SOIrgislGiven)    /* v4.1 */
            model->B4SOIrgisl = model->B4SOIrgidl;
            if (!model->B4SOIkgislGiven)        /* v4.1 */
            model->B4SOIkgisl = model->B4SOIkgidl;
                if (!model->B4SOIfgislGiven)    /* v4.1 */
            model->B4SOIfgisl = model->B4SOIfgidl;

        if (!model->B4SOIndiodeGiven)   /* v4.0 */
            model->B4SOIndiode = 1.0;
        if (!model->B4SOIndiodedGiven)  /* v4.0 */
            model->B4SOIndioded = model->B4SOIndiode;
        if (!model->B4SOIntunGiven)     /* v4.0 */
            model->B4SOIntun = 10.0;
        if (!model->B4SOIntundGiven)    /* v4.0 */
            model->B4SOIntund = model->B4SOIntun;

        if (!model->B4SOInrecf0Given)
            model->B4SOInrecf0 = 2.0;
        if (!model->B4SOInrecf0dGiven)
            model->B4SOInrecf0d = model->B4SOInrecf0;
        if (!model->B4SOInrecr0Given)
            model->B4SOInrecr0 = 10.0;
        if (!model->B4SOInrecr0dGiven)
            model->B4SOInrecr0d = model->B4SOInrecr0;

        if (!model->B4SOIisbjtGiven)
            model->B4SOIisbjt = 1e-6;
        if (!model->B4SOIidbjtGiven)
            model->B4SOIidbjt = model->B4SOIisbjt;
        if (!model->B4SOIisdifGiven)
            model->B4SOIisdif = 0.0;
        if (!model->B4SOIiddifGiven)
            model->B4SOIiddif = model->B4SOIisdif;      /* v4.0 */
        if (!model->B4SOIisrecGiven)
            model->B4SOIisrec = 1e-5;
        if (!model->B4SOIidrecGiven)
            model->B4SOIidrec = model->B4SOIisrec;
        if (!model->B4SOIistunGiven)
            model->B4SOIistun = 0.0;
        if (!model->B4SOIidtunGiven)
            model->B4SOIidtun = model->B4SOIistun;
        if (!model->B4SOIxbjtGiven)
            model->B4SOIxbjt = 1.0;
        if (!model->B4SOIxdifGiven)
            model->B4SOIxdif = model->B4SOIxbjt;
        if (!model->B4SOIxdifdGiven)
            model->B4SOIxdifd = model->B4SOIxdif;
        if (!model->B4SOIxrecGiven)
            model->B4SOIxrec = 1.0;
        if (!model->B4SOIxrecdGiven)
            model->B4SOIxrecd =  model->B4SOIxrec;
        if (!model->B4SOIxtunGiven)
            model->B4SOIxtun = 0.0;
        if (!model->B4SOIxtundGiven)
            model->B4SOIxtund = model->B4SOIxtun;
        if (!model->B4SOIttGiven)
            model->B4SOItt = 1e-12;
        if (!model->B4SOIasdGiven)
            model->B4SOIasd = 0.3;

        /* 4.0 backward compatibility  */
        if (!model->B4SOIlpe0Given) {
           if(!model->B4SOInlxGiven)
               model->B4SOIlpe0 = 1.74e-7;
           else
               model->B4SOIlpe0 =  model->B4SOInlx;
        }
        if(model->B4SOIlpe0Given && model->B4SOInlxGiven)
           printf("Warning: both lpe0 and nlx are given. Lpe0 value is taken \n");
        if (!model->B4SOIllpe0Given) {
           if(!model->B4SOIlnlxGiven)
               model->B4SOIllpe0 = 0.0;
           else
               model->B4SOIllpe0 =  model->B4SOIlnlx;
        }
        if(model->B4SOIllpe0Given && model->B4SOIlnlxGiven)
           printf("Warning: both llpe0 and lnlx are given. Llpe0 value is taken \n");
        if (!model->B4SOIwlpe0Given) {
           if(!model->B4SOIwnlxGiven)
               model->B4SOIwlpe0 = 0.0;
           else
               model->B4SOIwlpe0 =  model->B4SOIwnlx;
        }
        if(model->B4SOIwlpe0Given && model->B4SOIwnlxGiven)
           printf("Warning: both wlpe0 and wnlx are given. Wlpe0 value is taken \n");
        if (!model->B4SOIplpe0Given) {
           if(!model->B4SOIpnlxGiven)
               model->B4SOIplpe0 = 0.0;
           else
               model->B4SOIplpe0 =  model->B4SOIpnlx;
        }
        if(model->B4SOIplpe0Given && model->B4SOIpnlxGiven)
           printf("Warning: both plpe0 and pnlx are given. Plpe0 value is taken \n");

        if (!model->B4SOIegidlGiven) {
           if(!model->B4SOIngidlGiven)
               model->B4SOIegidl = 1.2;
           else
               model->B4SOIegidl =  model->B4SOIngidl;
        }

        if(model->B4SOIegidlGiven && model->B4SOIngidlGiven)
           printf("Warning: both egidl and ngidl are given. Egidl value is taken \n");
        if (!model->B4SOIlegidlGiven) {
           if(!model->B4SOIlngidlGiven)
               model->B4SOIlegidl = 0.0;
           else
               model->B4SOIlegidl =  model->B4SOIlngidl;
        }
        if(model->B4SOIlegidlGiven && model->B4SOIlngidlGiven)
           printf("Warning: both legidl and lngidl are given. Legidl value is taken \n");
        if (!model->B4SOIwegidlGiven) {
           if(!model->B4SOIwngidlGiven)
               model->B4SOIwegidl = 0.0;
           else
               model->B4SOIwegidl =  model->B4SOIwngidl;
        }
        if(model->B4SOIwegidlGiven && model->B4SOIwngidlGiven)
           printf("Warning: both wegidl and wngidl are given. Wegidl value is taken \n");
        if (!model->B4SOIpegidlGiven) {
           if(!model->B4SOIpngidlGiven)
               model->B4SOIpegidl = 0.0;
           else
               model->B4SOIpegidl =  model->B4SOIpngidl;
        }
        if(model->B4SOIpegidlGiven && model->B4SOIpngidlGiven)
           printf("Warning: both pegidl and pngidl are given. Pegidl value is taken \n");

                if (!model->B4SOIegislGiven) {
                model->B4SOIegisl = model->B4SOIegidl;
                   }
            if (!model->B4SOIlegislGiven) {
                model->B4SOIlegisl = model->B4SOIlegidl;
                   }
            if (!model->B4SOIwegislGiven) {
                model->B4SOIwegisl = model->B4SOIwegidl;
                   }
            if (!model->B4SOIpegislGiven) {
                model->B4SOIpegisl = model->B4SOIpegidl;
                   }
        /* unit degree celcius */
        if (!model->B4SOItnomGiven)
            model->B4SOItnom = ckt->CKTnomTemp;
        if (!model->B4SOILintGiven)
           model->B4SOILint = 0.0;
        if (!model->B4SOILlGiven)
           model->B4SOILl = 0.0;
        if (!model->B4SOILlcGiven)
           model->B4SOILlc = 0.0; /* v2.2.3 */
        if (!model->B4SOILlnGiven)
           model->B4SOILln = 1.0;
        if (!model->B4SOILwGiven)
           model->B4SOILw = 0.0;
        if (!model->B4SOILwcGiven)
           model->B4SOILwc = 0.0; /* v2.2.3 */
        if (!model->B4SOILwnGiven)
           model->B4SOILwn = 1.0;
        if (!model->B4SOILwlGiven)
           model->B4SOILwl = 0.0;
        if (!model->B4SOILwlcGiven)
           model->B4SOILwlc = 0.0; /* v2.2.3 */
        if (!model->B4SOILminGiven)
           model->B4SOILmin = 0.0;
        if (!model->B4SOILmaxGiven)
           model->B4SOILmax = 1.0;
        if (!model->B4SOIWintGiven)
           model->B4SOIWint = 0.0;
        if (!model->B4SOIWlGiven)
           model->B4SOIWl = 0.0;
        if (!model->B4SOIWlcGiven)
           model->B4SOIWlc = 0.0; /* v2.2.3 */
        if (!model->B4SOIWlnGiven)
           model->B4SOIWln = 1.0;
        if (!model->B4SOIWwGiven)
           model->B4SOIWw = 0.0;
        if (!model->B4SOIWwcGiven)
           model->B4SOIWwc = 0.0; /* v2.2.3 */
        if (!model->B4SOIWwnGiven)
           model->B4SOIWwn = 1.0;
        if (!model->B4SOIWwlGiven)
           model->B4SOIWwl = 0.0;
        if (!model->B4SOIWwlcGiven)
           model->B4SOIWwlc = 0.0; /* v2.2.3 */
        if (!model->B4SOIWminGiven)
           model->B4SOIWmin = 0.0;
        if (!model->B4SOIWmaxGiven)
           model->B4SOIWmax = 1.0;
        if (!model->B4SOIdwcGiven)
           model->B4SOIdwc = model->B4SOIWint;
        if (!model->B4SOIdlcGiven)
           model->B4SOIdlc = model->B4SOILint;
        if (!model->B4SOIdlcigGiven)
           model->B4SOIdlcig = model->B4SOILint; /* v3.0 */


/* v3.0 */
        if (!model->B4SOIvbs0pdGiven)
            model->B4SOIvbs0pd = 0.0; /* v3.2 */
        if (!model->B4SOIvbs0fdGiven)
            model->B4SOIvbs0fd = 0.5; /* v3.2 */
        if (!model->B4SOIvbsaGiven)
            model->B4SOIvbsa = 0.0;
        if (!model->B4SOInofffdGiven)
            model->B4SOInofffd = 1.0;
        if (!model->B4SOIvofffdGiven)
            model->B4SOIvofffd = 0.0;
        if (!model->B4SOIk1bGiven)
            model->B4SOIk1b = 1.0;
        if (!model->B4SOIk2bGiven)
            model->B4SOIk2b = 0.0;
        if (!model->B4SOIdk2bGiven)
            model->B4SOIdk2b = 0.0;
        if (!model->B4SOIdvbd0Given)
            model->B4SOIdvbd0 = 0.0;
        if (!model->B4SOIdvbd1Given)
            model->B4SOIdvbd1 = 0.0;
        if (!model->B4SOImoinFDGiven)
            model->B4SOImoinFD = 1e3;

/* v3.1 added for RF */
        if (!model->B4SOIxrcrg1Given)
            model->B4SOIxrcrg1 =12.0;
        if (!model->B4SOIxrcrg2Given)
            model->B4SOIxrcrg2 =1.0;
        if (!model->B4SOIrshgGiven)
            model->B4SOIrshg =0.1;
        if (!model->B4SOIngconGiven)
            model->B4SOIngcon =1.0;
        if (!model->B4SOIxgwGiven)
            model->B4SOIxgw = 0.0;
        if (!model->B4SOIxglGiven)
            model->B4SOIxgl = 0.0;
/* v3.1 added for RF end */

/* v2.2 release */
        if (!model->B4SOIwth0Given)
           model->B4SOIwth0 = 0.0;
        if (!model->B4SOIrhaloGiven)
           model->B4SOIrhalo = 1e15;
        if (!model->B4SOIntoxGiven)
           model->B4SOIntox = 1;
        if (!model->B4SOItoxrefGiven)
           model->B4SOItoxref = 2.5e-9;
        if (!model->B4SOIebgGiven)
           model->B4SOIebg = 1.2;
        if (!model->B4SOIvevbGiven)
           model->B4SOIvevb = 0.075;
        if (!model->B4SOIalphaGB1Given)
           model->B4SOIalphaGB1 = 0.35;
        if (!model->B4SOIbetaGB1Given)
           model->B4SOIbetaGB1 = 0.03;
        if (!model->B4SOIvgb1Given)
           model->B4SOIvgb1 = 300;
        if (!model->B4SOIalphaGB2Given)
           model->B4SOIalphaGB2 = 0.43;
        if (!model->B4SOIbetaGB2Given)
           model->B4SOIbetaGB2 = 0.05;
        if (!model->B4SOIvecbGiven)
           model->B4SOIvecb = 0.026;
        if (!model->B4SOIvgb2Given)
           model->B4SOIvgb2 = 17;
        if (!model->B4SOIaigbcp2Given)
            model->B4SOIaigbcp2 = 0.043;
        if (!model->B4SOIbigbcp2Given)
            model->B4SOIbigbcp2 = 0.0054;
        if (!model->B4SOIcigbcp2Given)
            model->B4SOIcigbcp2 = 0.0075;
        if (!model->B4SOItoxqmGiven)
           model->B4SOItoxqm = model->B4SOItox;
        if (!model->B4SOIvoxhGiven)
           model->B4SOIvoxh = 5.0;
        if (!model->B4SOIdeltavoxGiven)
           model->B4SOIdeltavox = 0.005;

/* v3.0 */
        if (!model->B4SOIigbModGiven)
           model->B4SOIigbMod = 0;
        if (!model->B4SOIigcModGiven)
           model->B4SOIigcMod = 0;
        if (!model->B4SOInigcGiven)
            model->B4SOInigc = 1.0;
        if (!model->B4SOIaigcGiven)
            model->B4SOIaigc = (model->B4SOItype == NMOS) ? 0.43 : 0.31;
        if (!model->B4SOIbigcGiven)
            model->B4SOIbigc = (model->B4SOItype == NMOS) ? 0.054 : 0.024;
        if (!model->B4SOIcigcGiven)
            model->B4SOIcigc = (model->B4SOItype == NMOS) ? 0.075 : 0.03;
        if (!model->B4SOIaigsdGiven)
            model->B4SOIaigsd = (model->B4SOItype == NMOS) ? 0.43 : 0.31;
        if (!model->B4SOIbigsdGiven)
            model->B4SOIbigsd = (model->B4SOItype == NMOS) ? 0.054 : 0.024;
        if (!model->B4SOIcigsdGiven)
            model->B4SOIcigsd = (model->B4SOItype == NMOS) ? 0.075 : 0.03;
        if (!model->B4SOIpigcdGiven)
            model->B4SOIpigcd = 1.0;
        if (!model->B4SOIpoxedgeGiven)
            model->B4SOIpoxedge = 1.0;



/* v2.0 release */
        if (!model->B4SOIk1w1Given)
           model->B4SOIk1w1 = 0.0;
        if (!model->B4SOIk1w2Given)
           model->B4SOIk1w2 = 0.0;
        if (!model->B4SOIketasGiven)
           model->B4SOIketas = 0.0;
        if (!model->B4SOIdwbcGiven)
           model->B4SOIdwbc = 0.0;
        if (!model->B4SOIbeta0Given)
           model->B4SOIbeta0 = 0.0;
        if (!model->B4SOIbeta1Given)
           model->B4SOIbeta1 = 0.0;
        if (!model->B4SOIbeta2Given)
           model->B4SOIbeta2 = 0.1;
        if (!model->B4SOIvdsatii0Given)
           model->B4SOIvdsatii0 = 0.9;
        if (!model->B4SOItiiGiven)
           model->B4SOItii = 0.0;
        if (!model->B4SOIliiGiven)
           model->B4SOIlii = 0.0;
        if (!model->B4SOIsii0Given)
           model->B4SOIsii0 = 0.5;
        if (!model->B4SOIsii1Given)
           model->B4SOIsii1 = 0.1;
        if (!model->B4SOIsii2Given)
           model->B4SOIsii2 = 0.0;
        if (!model->B4SOIsiidGiven)
           model->B4SOIsiid = 0.0;
        if (!model->B4SOIfbjtiiGiven)
           model->B4SOIfbjtii = 0.0;
                 /*4.1 Iii model*/
                if (!model->B4SOIebjtiiGiven)
           model->B4SOIebjtii = 0.0;
        if (!model->B4SOIcbjtiiGiven)
           model->B4SOIcbjtii = 0.0;
        if (!model->B4SOIvbciGiven)
           model->B4SOIvbci = 0.0;
            if (!model->B4SOItvbciGiven)
           model->B4SOItvbci = 0.0;
        if (!model->B4SOIabjtiiGiven)
           model->B4SOIabjtii = 0.0;
        if (!model->B4SOImbjtiiGiven)
           model->B4SOImbjtii = 0.4;

        if (!model->B4SOIesatiiGiven)
            model->B4SOIesatii = 1e7;
        if (!model->B4SOIlnGiven)
           model->B4SOIln = 2e-6;
        if (!model->B4SOIvrec0Given)    /* v4.0 */
           model->B4SOIvrec0 = 0;
        if (!model->B4SOIvrec0dGiven)   /* v4.0 */
           model->B4SOIvrec0d = model->B4SOIvrec0;
        if (!model->B4SOIvtun0Given)    /* v4.0 */
           model->B4SOIvtun0 = 0;
        if (!model->B4SOIvtun0dGiven)   /* v4.0 */
           model->B4SOIvtun0d = model->B4SOIvtun0;
        if (!model->B4SOInbjtGiven)
           model->B4SOInbjt = 1.0;
        if (!model->B4SOIlbjt0Given)
           model->B4SOIlbjt0 = 0.20e-6;
        if (!model->B4SOIldif0Given)
           model->B4SOIldif0 = 1.0;
        if (!model->B4SOIvabjtGiven)
           model->B4SOIvabjt = 10.0;
        if (!model->B4SOIaelyGiven)
           model->B4SOIaely = 0;
        if (!model->B4SOIahliGiven)     /* v4.0 */
           model->B4SOIahli = 0;
        if (!model->B4SOIahlidGiven)    /* v4.0 */
           model->B4SOIahlid = model->B4SOIahli;
        if (!model->B4SOIrbodyGiven)
           model->B4SOIrbody = 0.0;
        if (!model->B4SOIrbshGiven)
           model->B4SOIrbsh = 0.0;
        if (!model->B4SOIntrecfGiven)
           model->B4SOIntrecf = 0.0;
        if (!model->B4SOIntrecrGiven)
           model->B4SOIntrecr = 0.0;
        if (!model->B4SOIndifGiven)
           model->B4SOIndif = -1.0;
        if (!model->B4SOIdlcbGiven)
           model->B4SOIdlcb = 0.0;
        if (!model->B4SOIfbodyGiven)
           model->B4SOIfbody = 1.0;
        if (!model->B4SOItcjswgGiven)
           model->B4SOItcjswg = 0.0;
        if (!model->B4SOItpbswgGiven)
           model->B4SOItpbswg = 0.0;
        if (!model->B4SOItcjswgdGiven)
           model->B4SOItcjswgd = model->B4SOItcjswg;
        if (!model->B4SOItpbswgdGiven)
           model->B4SOItpbswgd = model->B4SOItpbswg;
        if (!model->B4SOIacdeGiven)
           model->B4SOIacde = 1.0;
        if (!model->B4SOImoinGiven)
           model->B4SOImoin = 15.0;
        if (!model->B4SOInoffGiven)
            model->B4SOInoff = 1.0; /* v3.2 */
        if (!model->B4SOIdelvtGiven)
           model->B4SOIdelvt = 0.0;
        if (!model->B4SOIkb1Given)
           model->B4SOIkb1 = 1.0;
        if (!model->B4SOIdlbgGiven)
           model->B4SOIdlbg = 0.0;

/* Added for binning - START */
        /* Length dependence */
/* v3.1 */
        if (!model->B4SOIlxjGiven)
            model->B4SOIlxj = 0.0;
        if (!model->B4SOIlalphaGB1Given)
            model->B4SOIlalphaGB1 = 0.0;
        if (!model->B4SOIlalphaGB2Given)
            model->B4SOIlalphaGB2 = 0.0;
        if (!model->B4SOIlbetaGB1Given)
            model->B4SOIlbetaGB1 = 0.0;
        if (!model->B4SOIlbetaGB2Given)
            model->B4SOIlbetaGB2 = 0.0;
        if (!model->B4SOIlaigbcp2Given)
            model->B4SOIlaigbcp2 = 0.0;
        if (!model->B4SOIlbigbcp2Given)
            model->B4SOIlbigbcp2 = 0.0;
        if (!model->B4SOIlcigbcp2Given)
            model->B4SOIlcigbcp2 = 0.0;
        if (!model->B4SOIlndifGiven)
            model->B4SOIlndif = 0.0;
        if (!model->B4SOIlntrecfGiven)
            model->B4SOIlntrecf = 0.0;
        if (!model->B4SOIlntrecrGiven)
            model->B4SOIlntrecr = 0.0;
        if (!model->B4SOIlxbjtGiven)
            model->B4SOIlxbjt = 0.0;
        if (!model->B4SOIlxdifGiven)
            model->B4SOIlxdif = 0.0;
        if (!model->B4SOIlxrecGiven)
            model->B4SOIlxrec = 0.0;
        if (!model->B4SOIlxtunGiven)
            model->B4SOIlxtun = 0.0;
        if (!model->B4SOIlxdifdGiven)
            model->B4SOIlxdifd = model->B4SOIlxdif;
        if (!model->B4SOIlxrecdGiven)
            model->B4SOIlxrecd = model->B4SOIlxrec;
        if (!model->B4SOIlxtundGiven)
            model->B4SOIlxtund = model->B4SOIlxtun;
        if (!model->B4SOIlcgdlGiven)
            model->B4SOIlcgdl = 0.0;
        if (!model->B4SOIlcgslGiven)
            model->B4SOIlcgsl = 0.0;
        if (!model->B4SOIlckappaGiven)
            model->B4SOIlckappa = 0.0;
        if (!model->B4SOIluteGiven)
            model->B4SOIlute = 0.0;
        if (!model->B4SOIlkt1Given)
            model->B4SOIlkt1 = 0.0;
        if (!model->B4SOIlkt2Given)
            model->B4SOIlkt2 = 0.0;
        if (!model->B4SOIlkt1lGiven)
            model->B4SOIlkt1l = 0.0;
        if (!model->B4SOIlua1Given)
            model->B4SOIlua1 = 0.0;
        if (!model->B4SOIlub1Given)
            model->B4SOIlub1 = 0.0;
        if (!model->B4SOIluc1Given)
            model->B4SOIluc1 = 0.0;
        if (!model->B4SOIlatGiven)
            model->B4SOIlat = 0.0;
        if (!model->B4SOIlprtGiven)
            model->B4SOIlprt = 0.0;

/* v3.0 */
        if (!model->B4SOIlnigcGiven)
            model->B4SOIlnigc = 0.0;
        if (!model->B4SOIlpigcdGiven)
            model->B4SOIlpigcd = 0.0;
        if (!model->B4SOIlpoxedgeGiven)
            model->B4SOIlpoxedge = 0.0;
        if (!model->B4SOIlaigcGiven)
            model->B4SOIlaigc = 0.0;
        if (!model->B4SOIlbigcGiven)
            model->B4SOIlbigc = 0.0;
        if (!model->B4SOIlcigcGiven)
            model->B4SOIlcigc = 0.0;
        if (!model->B4SOIlaigsdGiven)
            model->B4SOIlaigsd = 0.0;
        if (!model->B4SOIlbigsdGiven)
            model->B4SOIlbigsd = 0.0;
        if (!model->B4SOIlcigsdGiven)
            model->B4SOIlcigsd = 0.0;

        if (!model->B4SOIlnpeakGiven)
            model->B4SOIlnpeak = 0.0;
        if (!model->B4SOIlnsubGiven)
            model->B4SOIlnsub = 0.0;
        if (!model->B4SOIlngateGiven)
            model->B4SOIlngate = 0.0;
        if (!model->B4SOIlnsdGiven)
            model->B4SOIlnsd = 0.0;
        if (!model->B4SOIlvth0Given)
           model->B4SOIlvth0 = 0.0;
        if (!model->B4SOIlvfbGiven)
            model->B4SOIlvfb = 0.0;   /* v4.1 */
        if (!model->B4SOIlk1Given)
            model->B4SOIlk1 = 0.0;
        if (!model->B4SOIlk1w1Given)
            model->B4SOIlk1w1 = 0.0;
        if (!model->B4SOIlk1w2Given)
            model->B4SOIlk1w2 = 0.0;
        if (!model->B4SOIlk2Given)
            model->B4SOIlk2 = 0.0;
        if (!model->B4SOIlk3Given)
            model->B4SOIlk3 = 0.0;
        if (!model->B4SOIlk3bGiven)
            model->B4SOIlk3b = 0.0;
        if (!model->B4SOIlkb1Given)
           model->B4SOIlkb1 = 0.0;
        if (!model->B4SOIlw0Given)
            model->B4SOIlw0 = 0.0;
        if (!model->B4SOIllpebGiven)
            model->B4SOIllpeb = 0.0;
        if (!model->B4SOIldvt0Given)
            model->B4SOIldvt0 = 0.0;
        if (!model->B4SOIldvt1Given)
            model->B4SOIldvt1 = 0.0;
        if (!model->B4SOIldvt2Given)
            model->B4SOIldvt2 = 0.0;
        if (!model->B4SOIldvt0wGiven)
            model->B4SOIldvt0w = 0.0;
        if (!model->B4SOIldvt1wGiven)
            model->B4SOIldvt1w = 0.0;
        if (!model->B4SOIldvt2wGiven)
            model->B4SOIldvt2w = 0.0;
        if (!model->B4SOIlu0Given)
            model->B4SOIlu0 = 0.0;
        if (!model->B4SOIluaGiven)
            model->B4SOIlua = 0.0;
        if (!model->B4SOIlubGiven)
            model->B4SOIlub = 0.0;
        if (!model->B4SOIlucGiven)
            model->B4SOIluc = 0.0;
        if (!model->B4SOIlvsatGiven)
            model->B4SOIlvsat = 0.0;
        if (!model->B4SOIla0Given)
            model->B4SOIla0 = 0.0;
        if (!model->B4SOIlagsGiven)
            model->B4SOIlags = 0.0;
        if (!model->B4SOIlb0Given)
            model->B4SOIlb0 = 0.0;
        if (!model->B4SOIlb1Given)
            model->B4SOIlb1 = 0.0;
        if (!model->B4SOIlketaGiven)
            model->B4SOIlketa = 0.0;
        if (!model->B4SOIlketasGiven)
            model->B4SOIlketas = 0.0;
        if (!model->B4SOIla1Given)
            model->B4SOIla1 = 0.0;
        if (!model->B4SOIla2Given)
            model->B4SOIla2 = 0.0;
        if (!model->B4SOIlrdswGiven)
            model->B4SOIlrdsw = 0.0;
        if (!model->B4SOIlrswGiven)     /* v4.0 */
            model->B4SOIlrsw = 0.0;
        if (!model->B4SOIlrdwGiven)     /* v4.0 */
            model->B4SOIlrdw = 0.0;
        if (!model->B4SOIlprwbGiven)
            model->B4SOIlprwb = 0.0;
        if (!model->B4SOIlprwgGiven)
            model->B4SOIlprwg = 0.0;
        if (!model->B4SOIlwrGiven)
            model->B4SOIlwr = 0.0;
        if (!model->B4SOIlnfactorGiven)
            model->B4SOIlnfactor = 0.0;
        if (!model->B4SOIldwgGiven)
            model->B4SOIldwg = 0.0;
        if (!model->B4SOIldwbGiven)
            model->B4SOIldwb = 0.0;
        if (!model->B4SOIlvoffGiven)
            model->B4SOIlvoff = 0.0;
        if (!model->B4SOIleta0Given)
            model->B4SOIleta0 = 0.0;
        if (!model->B4SOIletabGiven)
            model->B4SOIletab = 0.0;
        if (!model->B4SOIldsubGiven)
            model->B4SOIldsub = 0.0;
        if (!model->B4SOIlcitGiven)
            model->B4SOIlcit = 0.0;
        if (!model->B4SOIlcdscGiven)
            model->B4SOIlcdsc = 0.0;
        if (!model->B4SOIlcdscbGiven)
            model->B4SOIlcdscb = 0.0;
        if (!model->B4SOIlcdscdGiven)
            model->B4SOIlcdscd = 0.0;
        if (!model->B4SOIlpclmGiven)
            model->B4SOIlpclm = 0.0;
        if (!model->B4SOIlpdibl1Given)
            model->B4SOIlpdibl1 = 0.0;
        if (!model->B4SOIlpdibl2Given)
            model->B4SOIlpdibl2 = 0.0;
        if (!model->B4SOIlpdiblbGiven)
            model->B4SOIlpdiblb = 0.0;
        if (!model->B4SOIldroutGiven)
            model->B4SOIldrout = 0.0;
        if (!model->B4SOIlpvagGiven)
            model->B4SOIlpvag = 0.0;
        if (!model->B4SOIldeltaGiven)
            model->B4SOIldelta = 0.0;
        if (!model->B4SOIlalpha0Given)
            model->B4SOIlalpha0 = 0.0;
        if (!model->B4SOIlfbjtiiGiven)
            model->B4SOIlfbjtii = 0.0;
                        /*4.1 Iii model*/
                if (!model->B4SOIlebjtiiGiven)
           model->B4SOIlebjtii = 0.0;
        if (!model->B4SOIlcbjtiiGiven)
           model->B4SOIlcbjtii = 0.0;
        if (!model->B4SOIlvbciGiven)
           model->B4SOIlvbci = 0.0;
        if (!model->B4SOIlabjtiiGiven)
           model->B4SOIlabjtii = 0.0;
        if (!model->B4SOIlmbjtiiGiven)
           model->B4SOIlmbjtii = 0.0;

        if (!model->B4SOIlbeta0Given)
            model->B4SOIlbeta0 = 0.0;
        if (!model->B4SOIlbeta1Given)
            model->B4SOIlbeta1 = 0.0;
        if (!model->B4SOIlbeta2Given)
            model->B4SOIlbeta2 = 0.0;
        if (!model->B4SOIlvdsatii0Given)
            model->B4SOIlvdsatii0 = 0.0;
        if (!model->B4SOIlliiGiven)
            model->B4SOIllii = 0.0;
        if (!model->B4SOIlesatiiGiven)
            model->B4SOIlesatii = 0.0;
        if (!model->B4SOIlsii0Given)
            model->B4SOIlsii0 = 0.0;
        if (!model->B4SOIlsii1Given)
            model->B4SOIlsii1 = 0.0;
        if (!model->B4SOIlsii2Given)
            model->B4SOIlsii2 = 0.0;
        if (!model->B4SOIlsiidGiven)
            model->B4SOIlsiid = 0.0;
        if (!model->B4SOIlagidlGiven)
            model->B4SOIlagidl = 0.0;
        if (!model->B4SOIlbgidlGiven)
            model->B4SOIlbgidl = 0.0;
        if (!model->B4SOIlcgidlGiven)
            model->B4SOIlcgidl = 0.0;
        if (!model->B4SOIlrgidlGiven)
            model->B4SOIlrgidl = 0.0;
        if (!model->B4SOIlkgidlGiven)
            model->B4SOIlkgidl = 0.0;
        if (!model->B4SOIlfgidlGiven)
            model->B4SOIlfgidl = 0.0;

                        if (!model->B4SOIlagislGiven)
            model->B4SOIlagisl = 0.0;
        if (!model->B4SOIlbgislGiven)
            model->B4SOIlbgisl = 0.0;
        if (!model->B4SOIlcgislGiven)
            model->B4SOIlcgisl = 0.0;
        if (!model->B4SOIlrgislGiven)
            model->B4SOIlrgisl = 0.0;
        if (!model->B4SOIlkgislGiven)
            model->B4SOIlkgisl = 0.0;
        if (!model->B4SOIlfgislGiven)
            model->B4SOIlfgisl = 0.0;
        if (!model->B4SOIlntunGiven)    /* v4.0 */
            model->B4SOIlntun = 0.0;
        if (!model->B4SOIlntundGiven)   /* v4.0 */
            model->B4SOIlntund = model->B4SOIlntun;
        if (!model->B4SOIlndiodeGiven)  /* v4.0 */
            model->B4SOIlndiode = 0.0;
        if (!model->B4SOIlndiodedGiven) /* v4.0 */
            model->B4SOIlndioded = model->B4SOIlndiode;
        if (!model->B4SOIlnrecf0Given)  /* v4.0 */
            model->B4SOIlnrecf0 = 0.0;
        if (!model->B4SOIlnrecf0dGiven) /* v4.0 */
            model->B4SOIlnrecf0d = model->B4SOIlnrecf0;
        if (!model->B4SOIlnrecr0Given)  /* v4.0 */
            model->B4SOIlnrecr0 = 0.0;
        if (!model->B4SOIlnrecr0dGiven) /* v4.0 */
            model->B4SOIlnrecr0d = model->B4SOIlnrecr0;
        if (!model->B4SOIlisbjtGiven)
            model->B4SOIlisbjt = 0.0;
        if (!model->B4SOIlidbjtGiven)
            model->B4SOIlidbjt = model->B4SOIlisbjt;
        if (!model->B4SOIlisdifGiven)
            model->B4SOIlisdif = 0.0;
        if (!model->B4SOIliddifGiven)
            model->B4SOIliddif = model->B4SOIlisdif;    /* v4.0 */
        if (!model->B4SOIlisrecGiven)
            model->B4SOIlisrec = 0.0;
        if (!model->B4SOIlidrecGiven)
            model->B4SOIlidrec = model->B4SOIlisrec;
        if (!model->B4SOIlistunGiven)
            model->B4SOIlistun = 0.0;
        if (!model->B4SOIlidtunGiven)
            model->B4SOIlidtun = model->B4SOIlistun;
        if (!model->B4SOIlvrec0Given)   /* v4.0 */
           model->B4SOIlvrec0 = 0;
        if (!model->B4SOIlvrec0dGiven)   /* v4.0 */
           model->B4SOIlvrec0d = model->B4SOIlvrec0;
        if (!model->B4SOIlvtun0Given)   /* v4.0 */
           model->B4SOIlvtun0 = 0;
        if (!model->B4SOIlvtun0dGiven)   /* v4.0 */
           model->B4SOIlvtun0d = model->B4SOIlvtun0;
        if (!model->B4SOIlnbjtGiven)
            model->B4SOIlnbjt = 0.0;
        if (!model->B4SOIllbjt0Given)
            model->B4SOIllbjt0 = 0.0;
        if (!model->B4SOIlvabjtGiven)
            model->B4SOIlvabjt = 0.0;
        if (!model->B4SOIlaelyGiven)
            model->B4SOIlaely = 0.0;
        if (!model->B4SOIlahliGiven)    /* v4.0 */
            model->B4SOIlahli = 0.0;
        if (!model->B4SOIlahlidGiven)   /* v4.0 */
            model->B4SOIlahlid = model->B4SOIlahli;
        /* CV Model */
        if (!model->B4SOIlvsdfbGiven)
            model->B4SOIlvsdfb = 0.0;
        if (!model->B4SOIlvsdthGiven)
            model->B4SOIlvsdth = 0.0;
        if (!model->B4SOIldelvtGiven)
            model->B4SOIldelvt = 0.0;
        if (!model->B4SOIlacdeGiven)
            model->B4SOIlacde = 0.0;
        if (!model->B4SOIlmoinGiven)
            model->B4SOIlmoin = 0.0;
        if (!model->B4SOIlnoffGiven)
            model->B4SOIlnoff = 0.0; /* v3.2 */

/* v3.1 added for RF */
        if (!model->B4SOIlxrcrg1Given)
            model->B4SOIlxrcrg1 =0.0;
        if (!model->B4SOIlxrcrg2Given)
            model->B4SOIlxrcrg2 =0.0;
/* v3.1 added for RF end */

        /* Width dependence */
/* v3.1 */
        if (!model->B4SOIwxjGiven)
            model->B4SOIwxj = 0.0;
        if (!model->B4SOIwalphaGB1Given)
            model->B4SOIwalphaGB1 = 0.0;
        if (!model->B4SOIwalphaGB2Given)
            model->B4SOIwalphaGB2 = 0.0;
        if (!model->B4SOIwbetaGB1Given)
            model->B4SOIwbetaGB1 = 0.0;
        if (!model->B4SOIwbetaGB2Given)
            model->B4SOIwbetaGB2 = 0.0;
        if (!model->B4SOIwaigbcp2Given)
            model->B4SOIwaigbcp2 = 0.0;
        if (!model->B4SOIwbigbcp2Given)
            model->B4SOIwbigbcp2 = 0.0;
        if (!model->B4SOIwcigbcp2Given)
            model->B4SOIwcigbcp2 = 0.0;
        if (!model->B4SOIwndifGiven)
            model->B4SOIwndif = 0.0;
        if (!model->B4SOIwntrecfGiven)
            model->B4SOIwntrecf = 0.0;
        if (!model->B4SOIwntrecrGiven)
            model->B4SOIwntrecr = 0.0;
        if (!model->B4SOIwxbjtGiven)
            model->B4SOIwxbjt = 0.0;
        if (!model->B4SOIwxdifGiven)
            model->B4SOIwxdif = 0.0;
        if (!model->B4SOIwxrecGiven)
            model->B4SOIwxrec = 0.0;
        if (!model->B4SOIwxtunGiven)
            model->B4SOIwxtun = 0.0;
        if (!model->B4SOIwxdifdGiven)
            model->B4SOIwxdifd = model->B4SOIwxdif;
        if (!model->B4SOIwxrecdGiven)
            model->B4SOIwxrecd = model->B4SOIwxrec;
        if (!model->B4SOIwxtundGiven)
            model->B4SOIwxtund = model->B4SOIwxtun;
        if (!model->B4SOIwcgdlGiven)
            model->B4SOIwcgdl = 0.0;
        if (!model->B4SOIwcgslGiven)
            model->B4SOIwcgsl = 0.0;
        if (!model->B4SOIwckappaGiven)
            model->B4SOIwckappa = 0.0;
        if (!model->B4SOIwuteGiven)
            model->B4SOIwute = 0.0;
        if (!model->B4SOIwkt1Given)
            model->B4SOIwkt1 = 0.0;
        if (!model->B4SOIwkt2Given)
            model->B4SOIwkt2 = 0.0;
        if (!model->B4SOIwkt1lGiven)
            model->B4SOIwkt1l = 0.0;
        if (!model->B4SOIwua1Given)
            model->B4SOIwua1 = 0.0;
        if (!model->B4SOIwub1Given)
            model->B4SOIwub1 = 0.0;
        if (!model->B4SOIwuc1Given)
            model->B4SOIwuc1 = 0.0;
        if (!model->B4SOIwatGiven)
            model->B4SOIwat = 0.0;
        if (!model->B4SOIwprtGiven)
            model->B4SOIwprt = 0.0;


/* v3.0 */
        if (!model->B4SOIwnigcGiven)
            model->B4SOIwnigc = 0.0;
        if (!model->B4SOIwpigcdGiven)
            model->B4SOIwpigcd = 0.0;
        if (!model->B4SOIwpoxedgeGiven)
            model->B4SOIwpoxedge = 0.0;
        if (!model->B4SOIwaigcGiven)
            model->B4SOIwaigc = 0.0;
        if (!model->B4SOIwbigcGiven)
            model->B4SOIwbigc = 0.0;
        if (!model->B4SOIwcigcGiven)
            model->B4SOIwcigc = 0.0;
        if (!model->B4SOIwaigsdGiven)
            model->B4SOIwaigsd = 0.0;
        if (!model->B4SOIwbigsdGiven)
            model->B4SOIwbigsd = 0.0;
        if (!model->B4SOIwcigsdGiven)
            model->B4SOIwcigsd = 0.0;

        if (!model->B4SOIwnpeakGiven)
            model->B4SOIwnpeak = 0.0;
        if (!model->B4SOIwnsubGiven)
            model->B4SOIwnsub = 0.0;
        if (!model->B4SOIwngateGiven)
            model->B4SOIwngate = 0.0;
        if (!model->B4SOIwnsdGiven)
            model->B4SOIwnsd = 0.0;
        if (!model->B4SOIwvth0Given)
           model->B4SOIwvth0 = 0.0;
        if (!model->B4SOIwvfbGiven)
            model->B4SOIwvfb = 0.0;   /* v4.1 */
        if (!model->B4SOIwk1Given)
            model->B4SOIwk1 = 0.0;
        if (!model->B4SOIwk1w1Given)
            model->B4SOIwk1w1 = 0.0;
        if (!model->B4SOIwk1w2Given)
            model->B4SOIwk1w2 = 0.0;
        if (!model->B4SOIwk2Given)
            model->B4SOIwk2 = 0.0;
        if (!model->B4SOIwk3Given)
            model->B4SOIwk3 = 0.0;
        if (!model->B4SOIwk3bGiven)
            model->B4SOIwk3b = 0.0;
        if (!model->B4SOIwkb1Given)
           model->B4SOIwkb1 = 0.0;
        if (!model->B4SOIww0Given)
            model->B4SOIww0 = 0.0;
        if (!model->B4SOIwlpebGiven)
            model->B4SOIwlpeb = 0.0;
        if (!model->B4SOIwdvt0Given)
            model->B4SOIwdvt0 = 0.0;
        if (!model->B4SOIwdvt1Given)
            model->B4SOIwdvt1 = 0.0;
        if (!model->B4SOIwdvt2Given)
            model->B4SOIwdvt2 = 0.0;
        if (!model->B4SOIwdvt0wGiven)
            model->B4SOIwdvt0w = 0.0;
        if (!model->B4SOIwdvt1wGiven)
            model->B4SOIwdvt1w = 0.0;
        if (!model->B4SOIwdvt2wGiven)
            model->B4SOIwdvt2w = 0.0;
        if (!model->B4SOIwu0Given)
            model->B4SOIwu0 = 0.0;
        if (!model->B4SOIwuaGiven)
            model->B4SOIwua = 0.0;
        if (!model->B4SOIwubGiven)
            model->B4SOIwub = 0.0;
        if (!model->B4SOIwucGiven)
            model->B4SOIwuc = 0.0;
        if (!model->B4SOIwvsatGiven)
            model->B4SOIwvsat = 0.0;
        if (!model->B4SOIwa0Given)
            model->B4SOIwa0 = 0.0;
        if (!model->B4SOIwagsGiven)
            model->B4SOIwags = 0.0;
        if (!model->B4SOIwb0Given)
            model->B4SOIwb0 = 0.0;
        if (!model->B4SOIwb1Given)
            model->B4SOIwb1 = 0.0;
        if (!model->B4SOIwketaGiven)
            model->B4SOIwketa = 0.0;
        if (!model->B4SOIwketasGiven)
            model->B4SOIwketas = 0.0;
        if (!model->B4SOIwa1Given)
            model->B4SOIwa1 = 0.0;
        if (!model->B4SOIwa2Given)
            model->B4SOIwa2 = 0.0;
        if (!model->B4SOIwrdswGiven)
            model->B4SOIwrdsw = 0.0;
        if (!model->B4SOIwrswGiven)     /* v4.0 */
            model->B4SOIwrsw = 0.0;
        if (!model->B4SOIwrdwGiven)     /* v4.0 */
            model->B4SOIwrdw = 0.0;
        if (!model->B4SOIwprwbGiven)
            model->B4SOIwprwb = 0.0;
        if (!model->B4SOIwprwgGiven)
            model->B4SOIwprwg = 0.0;
        if (!model->B4SOIwwrGiven)
            model->B4SOIwwr = 0.0;
        if (!model->B4SOIwnfactorGiven)
            model->B4SOIwnfactor = 0.0;
        if (!model->B4SOIwdwgGiven)
            model->B4SOIwdwg = 0.0;
        if (!model->B4SOIwdwbGiven)
            model->B4SOIwdwb = 0.0;
        if (!model->B4SOIwvoffGiven)
            model->B4SOIwvoff = 0.0;
        if (!model->B4SOIweta0Given)
            model->B4SOIweta0 = 0.0;
        if (!model->B4SOIwetabGiven)
            model->B4SOIwetab = 0.0;
        if (!model->B4SOIwdsubGiven)
            model->B4SOIwdsub = 0.0;
        if (!model->B4SOIwcitGiven)
            model->B4SOIwcit = 0.0;
        if (!model->B4SOIwcdscGiven)
            model->B4SOIwcdsc = 0.0;
        if (!model->B4SOIwcdscbGiven)
            model->B4SOIwcdscb = 0.0;
        if (!model->B4SOIwcdscdGiven)
            model->B4SOIwcdscd = 0.0;
        if (!model->B4SOIwpclmGiven)
            model->B4SOIwpclm = 0.0;
        if (!model->B4SOIwpdibl1Given)
            model->B4SOIwpdibl1 = 0.0;
        if (!model->B4SOIwpdibl2Given)
            model->B4SOIwpdibl2 = 0.0;
        if (!model->B4SOIwpdiblbGiven)
            model->B4SOIwpdiblb = 0.0;
        if (!model->B4SOIwdroutGiven)
            model->B4SOIwdrout = 0.0;
        if (!model->B4SOIwpvagGiven)
            model->B4SOIwpvag = 0.0;
        if (!model->B4SOIwdeltaGiven)
            model->B4SOIwdelta = 0.0;
        if (!model->B4SOIwalpha0Given)
            model->B4SOIwalpha0 = 0.0;
        if (!model->B4SOIwfbjtiiGiven)
            model->B4SOIwfbjtii = 0.0;
        /*4.1 Iii model*/
                if (!model->B4SOIwebjtiiGiven)
           model->B4SOIwebjtii = 0.0;
        if (!model->B4SOIwcbjtiiGiven)
           model->B4SOIwcbjtii = 0.0;
        if (!model->B4SOIwvbciGiven)
           model->B4SOIwvbci = 0.0;
        if (!model->B4SOIwabjtiiGiven)
           model->B4SOIwabjtii = 0.0;
        if (!model->B4SOIwmbjtiiGiven)
           model->B4SOIwmbjtii = 0.0;
        if (!model->B4SOIwbeta0Given)
            model->B4SOIwbeta0 = 0.0;
        if (!model->B4SOIwbeta1Given)
            model->B4SOIwbeta1 = 0.0;
        if (!model->B4SOIwbeta2Given)
            model->B4SOIwbeta2 = 0.0;
        if (!model->B4SOIwvdsatii0Given)
            model->B4SOIwvdsatii0 = 0.0;
        if (!model->B4SOIwliiGiven)
            model->B4SOIwlii = 0.0;
        if (!model->B4SOIwesatiiGiven)
            model->B4SOIwesatii = 0.0;
        if (!model->B4SOIwsii0Given)
            model->B4SOIwsii0 = 0.0;
        if (!model->B4SOIwsii1Given)
            model->B4SOIwsii1 = 0.0;
        if (!model->B4SOIwsii2Given)
            model->B4SOIwsii2 = 0.0;
        if (!model->B4SOIwsiidGiven)
            model->B4SOIwsiid = 0.0;
        if (!model->B4SOIwagidlGiven)
            model->B4SOIwagidl = 0.0;
        if (!model->B4SOIwbgidlGiven)
            model->B4SOIwbgidl = 0.0;
        if (!model->B4SOIwcgidlGiven)
            model->B4SOIwcgidl = 0.0;
        if (!model->B4SOIwrgidlGiven)
            model->B4SOIwrgidl = 0.0;
        if (!model->B4SOIwkgidlGiven)
            model->B4SOIwkgidl = 0.0;
        if (!model->B4SOIwfgidlGiven)
            model->B4SOIwfgidl = 0.0;

                if (!model->B4SOIwagislGiven)
            model->B4SOIwagisl = 0.0;
        if (!model->B4SOIwbgislGiven)
            model->B4SOIwbgisl = 0.0;
        if (!model->B4SOIwcgislGiven)
            model->B4SOIwcgisl = 0.0;
        if (!model->B4SOIwrgislGiven)
            model->B4SOIwrgisl = 0.0;
        if (!model->B4SOIwkgislGiven)
            model->B4SOIwkgisl = 0.0;
        if (!model->B4SOIwfgislGiven)
            model->B4SOIwfgisl = 0.0;
        if (!model->B4SOIwntunGiven)    /* v4.0 */
            model->B4SOIwntun = 0.0;
        if (!model->B4SOIwntundGiven)   /* v4.0 */
            model->B4SOIwntund = model->B4SOIwntun;
        if (!model->B4SOIwndiodeGiven)  /* v4.0 */
            model->B4SOIwndiode = 0.0;
        if (!model->B4SOIwndiodedGiven) /* v4.0 */
            model->B4SOIwndioded = model->B4SOIwndiode;
        if (!model->B4SOIwnrecf0Given) /* v4.0 */
            model->B4SOIwnrecf0 = 0.0;
        if (!model->B4SOIwnrecf0dGiven) /* v4.0 */
            model->B4SOIwnrecf0d = model->B4SOIwnrecf0;
        if (!model->B4SOIwnrecr0Given)  /* v4.0 */
            model->B4SOIwnrecr0 = 0.0;
        if (!model->B4SOIwnrecr0dGiven) /* v4.0 */
            model->B4SOIwnrecr0d = model->B4SOIwnrecr0;
        if (!model->B4SOIwisbjtGiven)
            model->B4SOIwisbjt = 0.0;
        if (!model->B4SOIwidbjtGiven)
            model->B4SOIwidbjt = model->B4SOIwisbjt;
        if (!model->B4SOIwisdifGiven)
            model->B4SOIwisdif = 0.0;
        if (!model->B4SOIwiddifGiven)
            model->B4SOIwiddif = model->B4SOIwisdif;    /* v4.0 */
        if (!model->B4SOIwisrecGiven)
            model->B4SOIwisrec = 0.0;
        if (!model->B4SOIwidrecGiven)
            model->B4SOIwidrec = model->B4SOIwisrec;
        if (!model->B4SOIwistunGiven)
            model->B4SOIwistun = 0.0;
        if (!model->B4SOIwidtunGiven)
            model->B4SOIwidtun = model->B4SOIwistun;
        if (!model->B4SOIwvrec0Given)   /* v4.0 */
           model->B4SOIwvrec0 = 0;
        if (!model->B4SOIwvrec0dGiven)   /* v4.0 */
           model->B4SOIwvrec0d = model->B4SOIwvrec0;
        if (!model->B4SOIwvtun0Given)   /* v4.0 */
           model->B4SOIwvtun0 = 0;
        if (!model->B4SOIwvtun0dGiven)   /* v4.0 */
           model->B4SOIwvtun0d = model->B4SOIwvtun0;
        if (!model->B4SOIwnbjtGiven)
            model->B4SOIwnbjt = 0.0;
        if (!model->B4SOIwlbjt0Given)
            model->B4SOIwlbjt0 = 0.0;
        if (!model->B4SOIwvabjtGiven)
            model->B4SOIwvabjt = 0.0;
        if (!model->B4SOIwaelyGiven)
            model->B4SOIwaely = 0.0;
        if (!model->B4SOIwahliGiven)    /* v4.0 */
            model->B4SOIwahli = 0.0;
        if (!model->B4SOIwahlidGiven)   /* v4.0 */
            model->B4SOIwahlid = model->B4SOIwahli;

/* v3.1 added for RF */
        if (!model->B4SOIwxrcrg1Given)
            model->B4SOIwxrcrg1 =0.0;
        if (!model->B4SOIwxrcrg2Given)
            model->B4SOIwxrcrg2 =0.0;
/* v3.1 added for RF end */

        /* CV Model */
        if (!model->B4SOIwvsdfbGiven)
            model->B4SOIwvsdfb = 0.0;
        if (!model->B4SOIwvsdthGiven)
            model->B4SOIwvsdth = 0.0;
        if (!model->B4SOIwdelvtGiven)
            model->B4SOIwdelvt = 0.0;
        if (!model->B4SOIwacdeGiven)
            model->B4SOIwacde = 0.0;
        if (!model->B4SOIwmoinGiven)
            model->B4SOIwmoin = 0.0;
        if (!model->B4SOIwnoffGiven)
            model->B4SOIwnoff = 0.0; /* v3.2 */

        /* Cross-term dependence */
/* v3.1 */
        if (!model->B4SOIpxjGiven)
            model->B4SOIpxj = 0.0;
        if (!model->B4SOIpalphaGB1Given)
            model->B4SOIpalphaGB1 = 0.0;
        if (!model->B4SOIpalphaGB2Given)
            model->B4SOIpalphaGB2 = 0.0;
        if (!model->B4SOIpbetaGB1Given)
            model->B4SOIpbetaGB1 = 0.0;
        if (!model->B4SOIpbetaGB2Given)
            model->B4SOIpbetaGB2 = 0.0;
        if (!model->B4SOIpaigbcp2Given)
            model->B4SOIpaigbcp2 = 0.0;
        if (!model->B4SOIpbigbcp2Given)
            model->B4SOIpbigbcp2 = 0.0;
        if (!model->B4SOIpcigbcp2Given)
            model->B4SOIpcigbcp2 = 0.0;
        if (!model->B4SOIpndifGiven)
            model->B4SOIpndif = 0.0;
        if (!model->B4SOIpntrecfGiven)
            model->B4SOIpntrecf = 0.0;
        if (!model->B4SOIpntrecrGiven)
            model->B4SOIpntrecr = 0.0;
        if (!model->B4SOIpxbjtGiven)
            model->B4SOIpxbjt = 0.0;
        if (!model->B4SOIpxdifGiven)
            model->B4SOIpxdif = 0.0;
        if (!model->B4SOIpxrecGiven)
            model->B4SOIpxrec = 0.0;
        if (!model->B4SOIpxtunGiven)
            model->B4SOIpxtun = 0.0;
        if (!model->B4SOIpxdifdGiven)
            model->B4SOIpxdifd = model->B4SOIpxdif;
        if (!model->B4SOIpxrecdGiven)
            model->B4SOIpxrecd = model->B4SOIpxrec;
        if (!model->B4SOIpxtundGiven)
            model->B4SOIpxtund = model->B4SOIpxtun;
        if (!model->B4SOIpcgdlGiven)
            model->B4SOIpcgdl = 0.0;
        if (!model->B4SOIpcgslGiven)
            model->B4SOIpcgsl = 0.0;
        if (!model->B4SOIpckappaGiven)
            model->B4SOIpckappa = 0.0;
        if (!model->B4SOIputeGiven)
            model->B4SOIpute = 0.0;
        if (!model->B4SOIpkt1Given)
            model->B4SOIpkt1 = 0.0;
        if (!model->B4SOIpkt2Given)
            model->B4SOIpkt2 = 0.0;
        if (!model->B4SOIpkt1lGiven)
            model->B4SOIpkt1l = 0.0;
        if (!model->B4SOIpua1Given)
            model->B4SOIpua1 = 0.0;
        if (!model->B4SOIpub1Given)
            model->B4SOIpub1 = 0.0;
        if (!model->B4SOIpuc1Given)
            model->B4SOIpuc1 = 0.0;
        if (!model->B4SOIpatGiven)
            model->B4SOIpat = 0.0;
        if (!model->B4SOIpprtGiven)
            model->B4SOIpprt = 0.0;


/* v3.0 */
        if (!model->B4SOIpnigcGiven)
            model->B4SOIpnigc = 0.0;
        if (!model->B4SOIppigcdGiven)
            model->B4SOIppigcd = 0.0;
        if (!model->B4SOIppoxedgeGiven)
            model->B4SOIppoxedge = 0.0;
        if (!model->B4SOIpaigcGiven)
            model->B4SOIpaigc = 0.0;
        if (!model->B4SOIpbigcGiven)
            model->B4SOIpbigc = 0.0;
        if (!model->B4SOIpcigcGiven)
            model->B4SOIpcigc = 0.0;
        if (!model->B4SOIpaigsdGiven)
            model->B4SOIpaigsd = 0.0;
        if (!model->B4SOIpbigsdGiven)
            model->B4SOIpbigsd = 0.0;
        if (!model->B4SOIpcigsdGiven)
            model->B4SOIpcigsd = 0.0;

        if (!model->B4SOIpnpeakGiven)
            model->B4SOIpnpeak = 0.0;
        if (!model->B4SOIpnsubGiven)
            model->B4SOIpnsub = 0.0;
        if (!model->B4SOIpngateGiven)
            model->B4SOIpngate = 0.0;
        if (!model->B4SOIpnsdGiven)
            model->B4SOIpnsd = 0.0;
        if (!model->B4SOIpvth0Given)
           model->B4SOIpvth0 = 0.0;
        if (!model->B4SOIpvfbGiven)
            model->B4SOIpvfb = 0.0;   /* v4.1 */
        if (!model->B4SOIpk1Given)
            model->B4SOIpk1 = 0.0;
        if (!model->B4SOIpk1w1Given)
            model->B4SOIpk1w1 = 0.0;
        if (!model->B4SOIpk1w2Given)
            model->B4SOIpk1w2 = 0.0;
        if (!model->B4SOIpk2Given)
            model->B4SOIpk2 = 0.0;
        if (!model->B4SOIpk3Given)
            model->B4SOIpk3 = 0.0;
        if (!model->B4SOIpk3bGiven)
            model->B4SOIpk3b = 0.0;
        if (!model->B4SOIpkb1Given)
           model->B4SOIpkb1 = 0.0;
        if (!model->B4SOIpw0Given)
            model->B4SOIpw0 = 0.0;
        if (!model->B4SOIplpebGiven)
            model->B4SOIplpeb = 0.0;
        if (!model->B4SOIpdvt0Given)
            model->B4SOIpdvt0 = 0.0;
        if (!model->B4SOIpdvt1Given)
            model->B4SOIpdvt1 = 0.0;
        if (!model->B4SOIpdvt2Given)
            model->B4SOIpdvt2 = 0.0;
        if (!model->B4SOIpdvt0wGiven)
            model->B4SOIpdvt0w = 0.0;
        if (!model->B4SOIpdvt1wGiven)
            model->B4SOIpdvt1w = 0.0;
        if (!model->B4SOIpdvt2wGiven)
            model->B4SOIpdvt2w = 0.0;
        if (!model->B4SOIpu0Given)
            model->B4SOIpu0 = 0.0;
        if (!model->B4SOIpuaGiven)
            model->B4SOIpua = 0.0;
        if (!model->B4SOIpubGiven)
            model->B4SOIpub = 0.0;
        if (!model->B4SOIpucGiven)
            model->B4SOIpuc = 0.0;
        if (!model->B4SOIpvsatGiven)
            model->B4SOIpvsat = 0.0;
        if (!model->B4SOIpa0Given)
            model->B4SOIpa0 = 0.0;
        if (!model->B4SOIpagsGiven)
            model->B4SOIpags = 0.0;
        if (!model->B4SOIpb0Given)
            model->B4SOIpb0 = 0.0;
        if (!model->B4SOIpb1Given)
            model->B4SOIpb1 = 0.0;
        if (!model->B4SOIpketaGiven)
            model->B4SOIpketa = 0.0;
        if (!model->B4SOIpketasGiven)
            model->B4SOIpketas = 0.0;
        if (!model->B4SOIpa1Given)
            model->B4SOIpa1 = 0.0;
        if (!model->B4SOIpa2Given)
            model->B4SOIpa2 = 0.0;
        if (!model->B4SOIprdswGiven)
            model->B4SOIprdsw = 0.0;
        if (!model->B4SOIprswGiven)     /* v4.0 */
            model->B4SOIprsw = 0.0;
        if (!model->B4SOIprdwGiven)     /* v4.0 */
            model->B4SOIprdw = 0.0;
        if (!model->B4SOIpprwbGiven)
            model->B4SOIpprwb = 0.0;
        if (!model->B4SOIpprwgGiven)
            model->B4SOIpprwg = 0.0;
        if (!model->B4SOIpwrGiven)
            model->B4SOIpwr = 0.0;
        if (!model->B4SOIpnfactorGiven)
            model->B4SOIpnfactor = 0.0;
        if (!model->B4SOIpdwgGiven)
            model->B4SOIpdwg = 0.0;
        if (!model->B4SOIpdwbGiven)
            model->B4SOIpdwb = 0.0;
        if (!model->B4SOIpvoffGiven)
            model->B4SOIpvoff = 0.0;
        if (!model->B4SOIpeta0Given)
            model->B4SOIpeta0 = 0.0;
        if (!model->B4SOIpetabGiven)
            model->B4SOIpetab = 0.0;
        if (!model->B4SOIpdsubGiven)
            model->B4SOIpdsub = 0.0;
        if (!model->B4SOIpcitGiven)
            model->B4SOIpcit = 0.0;
        if (!model->B4SOIpcdscGiven)
            model->B4SOIpcdsc = 0.0;
        if (!model->B4SOIpcdscbGiven)
            model->B4SOIpcdscb = 0.0;
        if (!model->B4SOIpcdscdGiven)
            model->B4SOIpcdscd = 0.0;
        if (!model->B4SOIppclmGiven)
            model->B4SOIppclm = 0.0;
        if (!model->B4SOIppdibl1Given)
            model->B4SOIppdibl1 = 0.0;
        if (!model->B4SOIppdibl2Given)
            model->B4SOIppdibl2 = 0.0;
        if (!model->B4SOIppdiblbGiven)
            model->B4SOIppdiblb = 0.0;
        if (!model->B4SOIpdroutGiven)
            model->B4SOIpdrout = 0.0;
        if (!model->B4SOIppvagGiven)
            model->B4SOIppvag = 0.0;
        if (!model->B4SOIpdeltaGiven)
            model->B4SOIpdelta = 0.0;
        if (!model->B4SOIpalpha0Given)
            model->B4SOIpalpha0 = 0.0;
        if (!model->B4SOIpfbjtiiGiven)
            model->B4SOIpfbjtii = 0.0;
        /*4.1 Iii model*/
            if (!model->B4SOIpebjtiiGiven)
           model->B4SOIpebjtii = 0.0;
        if (!model->B4SOIpcbjtiiGiven)
           model->B4SOIpcbjtii = 0.0;
        if (!model->B4SOIpvbciGiven)
           model->B4SOIpvbci = 0.0;
        if (!model->B4SOIpabjtiiGiven)
           model->B4SOIpabjtii = 0.0;
        if (!model->B4SOIpmbjtiiGiven)
           model->B4SOIpmbjtii = 0.0;
        if (!model->B4SOIpbeta0Given)
            model->B4SOIpbeta0 = 0.0;
        if (!model->B4SOIpbeta1Given)
            model->B4SOIpbeta1 = 0.0;
        if (!model->B4SOIpbeta2Given)
            model->B4SOIpbeta2 = 0.0;
        if (!model->B4SOIpvdsatii0Given)
            model->B4SOIpvdsatii0 = 0.0;
        if (!model->B4SOIpliiGiven)
            model->B4SOIplii = 0.0;
        if (!model->B4SOIpesatiiGiven)
            model->B4SOIpesatii = 0.0;
        if (!model->B4SOIpsii0Given)
            model->B4SOIpsii0 = 0.0;
        if (!model->B4SOIpsii1Given)
            model->B4SOIpsii1 = 0.0;
        if (!model->B4SOIpsii2Given)
            model->B4SOIpsii2 = 0.0;
        if (!model->B4SOIpsiidGiven)
            model->B4SOIpsiid = 0.0;
        if (!model->B4SOIpagidlGiven)
            model->B4SOIpagidl = 0.0;
        if (!model->B4SOIpbgidlGiven)
            model->B4SOIpbgidl = 0.0;
        if (!model->B4SOIpcgidlGiven)
            model->B4SOIpcgidl = 0.0;
        if (!model->B4SOIprgidlGiven)
            model->B4SOIprgidl = 0.0;
        if (!model->B4SOIpkgidlGiven)
            model->B4SOIpkgidl = 0.0;
        if (!model->B4SOIpfgidlGiven)
            model->B4SOIpfgidl = 0.0;

                if (!model->B4SOIpagislGiven)
            model->B4SOIpagisl = 0.0;
        if (!model->B4SOIpbgislGiven)
            model->B4SOIpbgisl = 0.0;
        if (!model->B4SOIpcgislGiven)
            model->B4SOIpcgisl = 0.0;
        if (!model->B4SOIprgislGiven)
            model->B4SOIprgisl = 0.0;
        if (!model->B4SOIpkgislGiven)
            model->B4SOIpkgisl = 0.0;
        if (!model->B4SOIpfgislGiven)
            model->B4SOIpfgisl = 0.0;
        if (!model->B4SOIpntunGiven)    /* v4.0 */
            model->B4SOIpntun = 0.0;
        if (!model->B4SOIpntundGiven)   /* v4.0 */
            model->B4SOIpntund = model->B4SOIpntun;
        if (!model->B4SOIpndiodeGiven)  /* v4.0 */
            model->B4SOIpndiode = 0.0;
        if (!model->B4SOIpndiodedGiven) /* v4.0 */
            model->B4SOIpndioded = model->B4SOIpndiode;
        if (!model->B4SOIpnrecf0Given)  /* v4.0 */
            model->B4SOIpnrecf0 = 0.0;
        if (!model->B4SOIpnrecf0dGiven) /* v4.0 */
            model->B4SOIpnrecf0d = model->B4SOIpnrecf0;
        if (!model->B4SOIpnrecr0Given)  /* v4.0 */
            model->B4SOIpnrecr0 = 0.0;
        if (!model->B4SOIpnrecr0dGiven) /* v4.0 */
            model->B4SOIpnrecr0d = model->B4SOIpnrecr0;
        if (!model->B4SOIpisbjtGiven)
            model->B4SOIpisbjt = 0.0;
        if (!model->B4SOIpidbjtGiven)
            model->B4SOIpidbjt = model->B4SOIpisbjt;
        if (!model->B4SOIpisdifGiven)
            model->B4SOIpisdif = 0.0;
        if (!model->B4SOIpiddifGiven)
            model->B4SOIpiddif = model->B4SOIpisdif;  /* v4.0 */
        if (!model->B4SOIpisrecGiven)
            model->B4SOIpisrec = 0.0;
        if (!model->B4SOIpidrecGiven)
            model->B4SOIpidrec = model->B4SOIpisrec;
        if (!model->B4SOIpistunGiven)
            model->B4SOIpistun = 0.0;
        if (!model->B4SOIpidtunGiven)
            model->B4SOIpidtun = model->B4SOIpistun;
        if (!model->B4SOIpvrec0Given)   /* v4.0 */
            model->B4SOIpvrec0 = 0.0;
        if (!model->B4SOIpvrec0dGiven)  /* v4.0 */
            model->B4SOIpvrec0d = model->B4SOIpvrec0;
        if (!model->B4SOIpvtun0Given)   /* v4.0 */
            model->B4SOIpvtun0 = 0.0;
        if (!model->B4SOIpvtun0dGiven)  /* v4.0 */
            model->B4SOIpvtun0d = model->B4SOIpvtun0;
        if (!model->B4SOIpnbjtGiven)
            model->B4SOIpnbjt = 0.0;
        if (!model->B4SOIplbjt0Given)
            model->B4SOIplbjt0 = 0.0;
        if (!model->B4SOIpvabjtGiven)
            model->B4SOIpvabjt = 0.0;
        if (!model->B4SOIpaelyGiven)
            model->B4SOIpaely = 0.0;
        if (!model->B4SOIpahliGiven)    /* v4.0 */
            model->B4SOIpahli = 0.0;
        if (!model->B4SOIpahlidGiven)   /* v4.0 */
            model->B4SOIpahlid = model->B4SOIpahli;

/* v3.1 for RF */
        if (!model->B4SOIpxrcrg1Given)
            model->B4SOIpxrcrg1 =0.0;
        if (!model->B4SOIpxrcrg2Given)
            model->B4SOIpxrcrg2 =0.0;
/* v3.1 for RF end */

        /* CV Model */
        if (!model->B4SOIpvsdfbGiven)
            model->B4SOIpvsdfb = 0.0;
        if (!model->B4SOIpvsdthGiven)
            model->B4SOIpvsdth = 0.0;
        if (!model->B4SOIpdelvtGiven)
            model->B4SOIpdelvt = 0.0;
        if (!model->B4SOIpacdeGiven)
            model->B4SOIpacde = 0.0;
        if (!model->B4SOIpmoinGiven)
            model->B4SOIpmoin = 0.0;
        if (!model->B4SOIpnoffGiven)
            model->B4SOIpnoff = 0.0; /* v3.2 */
/* Added for binning - END */

        if (!model->B4SOIcfGiven)
            model->B4SOIcf = 2.0 * EPSOX / PI
                           * log(1.0 + 0.4e-6 / model->B4SOItox);
        if (!model->B4SOIcgdoGiven)
        {   if (model->B4SOIdlcGiven && (model->B4SOIdlc > 0.0))
            {   model->B4SOIcgdo = model->B4SOIdlc * model->B4SOIcox
                                 - model->B4SOIcgdl ;
            }
            else
                model->B4SOIcgdo = 0.6 * model->B4SOIxj * model->B4SOIcox;
        }
        if (!model->B4SOIcgsoGiven)
        {   if (model->B4SOIdlcGiven && (model->B4SOIdlc > 0.0))
            {   model->B4SOIcgso = model->B4SOIdlc * model->B4SOIcox
                                 - model->B4SOIcgsl ;
            }
            else
                model->B4SOIcgso = 0.6 * model->B4SOIxj * model->B4SOIcox;
        }

        if (!model->B4SOIcgeoGiven)
        {   model->B4SOIcgeo = 0.0;
        }
        if (!model->B4SOIxpartGiven)
            model->B4SOIxpart = 0.0;
        if (!model->B4SOIsheetResistanceGiven)
            model->B4SOIsheetResistance = 0.0;
        if (!model->B4SOIcsdeswGiven)
            model->B4SOIcsdesw = 0.0;
        if (!model->B4SOIunitLengthGateSidewallJctCapSGiven)    /* v4.0 */
            model->B4SOIunitLengthGateSidewallJctCapS = 1e-10;
        if (!model->B4SOIunitLengthGateSidewallJctCapDGiven)    /* v4.0 */
            model->B4SOIunitLengthGateSidewallJctCapD = model->B4SOIunitLengthGateSidewallJctCapS;
        if (!model->B4SOIGatesidewallJctSPotentialGiven)        /* v4.0 */
            model->B4SOIGatesidewallJctSPotential = 0.7;
        if (!model->B4SOIGatesidewallJctDPotentialGiven)        /* v4.0 */
            model->B4SOIGatesidewallJctDPotential = model->B4SOIGatesidewallJctSPotential;
        if (!model->B4SOIbodyJctGateSideSGradingCoeffGiven)     /* v4.0 */
            model->B4SOIbodyJctGateSideSGradingCoeff = 0.5;
        if (!model->B4SOIbodyJctGateSideDGradingCoeffGiven)     /* v4.0 */
            model->B4SOIbodyJctGateSideDGradingCoeff = model->B4SOIbodyJctGateSideSGradingCoeff;
        if (!model->B4SOIoxideTrapDensityAGiven)
        {   if (model->B4SOItype == NMOS)
                model->B4SOIoxideTrapDensityA = 6.25e41;
            else
                model->B4SOIoxideTrapDensityA=6.188e40;
        }
        if (!model->B4SOIoxideTrapDensityBGiven)
        {   if (model->B4SOItype == NMOS)
                model->B4SOIoxideTrapDensityB = 3.125e26;
            else
                model->B4SOIoxideTrapDensityB = 1.5e25;
        }
        if (!model->B4SOIoxideTrapDensityCGiven)
            model->B4SOIoxideTrapDensityC = 8.75e9;

        if (!model->B4SOIemGiven)
            model->B4SOIem = 4.1e7; /* V/m */
        if (!model->B4SOIefGiven)
            model->B4SOIef = 1.0;
        if (!model->B4SOIafGiven)
            model->B4SOIaf = 1.0;
        if (!model->B4SOIkfGiven)
            model->B4SOIkf = 0.0;
        if (!model->B4SOInoifGiven)
            model->B4SOInoif = 1.0;
        if (!model->B4SOIbfGiven)       /* v4.0 */
            model->B4SOIbf = 2.0;
        if (!model->B4SOIw0flkGiven)    /* v4.0 */
            model->B4SOIw0flk = 10.0e-6;
        if (!model->B4SOIfrbodyGiven)   /* v4.0 */
            model->B4SOIfrbody = 1;
        if (!model->B4SOIdvtp0Given)    /* v4.0 for Vth */
            model->B4SOIdvtp0 = 0.0;
        if (!model->B4SOIdvtp1Given)    /* v4.0 for Vth */
            model->B4SOIdvtp1 = 0.0;
        if (!model->B4SOIdvtp2Given)    /* v4.1 for Vth */
            model->B4SOIdvtp2 = 0.0;
        if (!model->B4SOIdvtp3Given)    /* v4.1 for Vth */
            model->B4SOIdvtp3 = 0.0;
        if (!model->B4SOIdvtp4Given)    /* v4.1 for Vth */
            model->B4SOIdvtp4 = 0.0;
        if (!model->B4SOIldvtp0Given)   /* v4.0 for Vth */
            model->B4SOIldvtp0 = 0.0;
        if (!model->B4SOIldvtp1Given)   /* v4.0 for Vth */
            model->B4SOIldvtp1 = 0.0;
        if (!model->B4SOIldvtp2Given)   /* v4.1 for Vth */
            model->B4SOIldvtp2 = 0.0;
        if (!model->B4SOIldvtp3Given)   /* v4.1 for Vth */
            model->B4SOIldvtp3 = 0.0;
        if (!model->B4SOIldvtp4Given)   /* v4.1 for Vth */
            model->B4SOIldvtp4 = 0.0;
        if (!model->B4SOIwdvtp0Given)   /* v4.0 for Vth */
            model->B4SOIwdvtp0 = 0.0;
        if (!model->B4SOIwdvtp1Given)   /* v4.0 for Vth */
            model->B4SOIwdvtp1 = 0.0;
        if (!model->B4SOIwdvtp2Given)   /* v4.1 for Vth */
            model->B4SOIwdvtp2 = 0.0;
        if (!model->B4SOIwdvtp3Given)   /* v4.1 for Vth */
            model->B4SOIwdvtp3 = 0.0;
        if (!model->B4SOIwdvtp4Given)   /* v4.1 for Vth */
            model->B4SOIwdvtp4 = 0.0;
        if (!model->B4SOIpdvtp0Given)   /* v4.0 for Vth */
            model->B4SOIpdvtp0 = 0.0;
        if (!model->B4SOIpdvtp1Given)   /* v4.0 for Vth */
            model->B4SOIpdvtp1 = 0.0;
        if (!model->B4SOIpdvtp2Given)   /* v4.1 for Vth */
            model->B4SOIpdvtp2 = 0.0;
        if (!model->B4SOIpdvtp3Given)   /* v4.1 for Vth */
            model->B4SOIpdvtp3 = 0.0;
        if (!model->B4SOIpdvtp4Given)   /* v4.1 for Vth */
            model->B4SOIpdvtp4 = 0.0;
        if (!model->B4SOIminvGiven)     /* v4.0 for Vgsteff */
            model->B4SOIminv = 0.0;
        if (!model->B4SOIlminvGiven)    /* v4.0 for Vgsteff */
            model->B4SOIlminv = 0.0;
        if (!model->B4SOIwminvGiven)    /* v4.0 for Vgsteff */
            model->B4SOIwminv = 0.0;
        if (!model->B4SOIpminvGiven)    /* v4.0 for Vgsteff */
            model->B4SOIpminv = 0.0;
        if (!model->B4SOIfproutGiven)   /* v4.0 for DITS in Id */
            model->B4SOIfprout = 0.0;
        if (!model->B4SOIpditsGiven)
            model->B4SOIpdits = 1e-20;
        if (!model->B4SOIpditsdGiven)
            model->B4SOIpditsd = 0.0;
        if (!model->B4SOIpditslGiven)
            model->B4SOIpditsl = 0.0;
        if (!model->B4SOIlfproutGiven)
            model->B4SOIlfprout = 0.0;
        if (!model->B4SOIlpditsGiven)
            model->B4SOIlpdits = 0.0;
        if (!model->B4SOIlpditsdGiven)
            model->B4SOIlpditsd = 0.0;
        if (!model->B4SOIwfproutGiven)
            model->B4SOIwfprout = 0.0;
        if (!model->B4SOIwpditsGiven)
            model->B4SOIwpdits = 0.0;
        if (!model->B4SOIwpditsdGiven)
            model->B4SOIwpditsd = 0.0;
        if (!model->B4SOIpfproutGiven)
            model->B4SOIpfprout = 0.0;
        if (!model->B4SOIppditsGiven)
            model->B4SOIppdits = 0.0;
        if (!model->B4SOIppditsdGiven)
            model->B4SOIppditsd = 0.0;

/* v4.0 stress effect */
        if (!model->B4SOIsarefGiven)
            model->B4SOIsaref = 1e-6; /* m */
        if (!model->B4SOIsbrefGiven)
            model->B4SOIsbref = 1e-6;  /* m */
        if (!model->B4SOIwlodGiven)
            model->B4SOIwlod = 0;  /* m */
        if (!model->B4SOIku0Given)
            model->B4SOIku0 = 0; /* 1/m */
        if (!model->B4SOIkvsatGiven)
            model->B4SOIkvsat = 0;
        if (!model->B4SOIkvth0Given) /* m */
            model->B4SOIkvth0 = 0;
        if (!model->B4SOItku0Given)
            model->B4SOItku0 = 0;
        if (!model->B4SOIllodku0Given)
            model->B4SOIllodku0 = 0;
        if (!model->B4SOIwlodku0Given)
            model->B4SOIwlodku0 = 0;
        if (!model->B4SOIllodvthGiven)
            model->B4SOIllodvth = 0;
        if (!model->B4SOIwlodvthGiven)
            model->B4SOIwlodvth = 0;
        if (!model->B4SOIlku0Given)
            model->B4SOIlku0 = 0;
        if (!model->B4SOIwku0Given)
            model->B4SOIwku0 = 0;
        if (!model->B4SOIpku0Given)
            model->B4SOIpku0 = 0;
        if (!model->B4SOIlkvth0Given)
            model->B4SOIlkvth0 = 0;
        if (!model->B4SOIwkvth0Given)
            model->B4SOIwkvth0 = 0;
        if (!model->B4SOIpkvth0Given)
            model->B4SOIpkvth0 = 0;
        if (!model->B4SOIstk2Given)
            model->B4SOIstk2 = 0;
        if (!model->B4SOIlodk2Given)
            model->B4SOIlodk2 = 1.0;
        if (!model->B4SOIsteta0Given)
            model->B4SOIsteta0 = 0;
        if (!model->B4SOIlodeta0Given)
            model->B4SOIlodeta0 = 1.0;
/* stress effect end */

        if (!model->B4SOIvgsMaxGiven)
            model->B4SOIvgsMax = 1e99;
        if (!model->B4SOIvgdMaxGiven)
            model->B4SOIvgdMax = 1e99;
        if (!model->B4SOIvgbMaxGiven)
            model->B4SOIvgbMax = 1e99;
        if (!model->B4SOIvdsMaxGiven)
            model->B4SOIvdsMax = 1e99;
        if (!model->B4SOIvbsMaxGiven)
            model->B4SOIvbsMax = 1e99;
        if (!model->B4SOIvbdMaxGiven)
            model->B4SOIvbdMax = 1e99;
        if (!model->B4SOIvgsrMaxGiven)
            model->B4SOIvgsrMax = 1e99;
        if (!model->B4SOIvgdrMaxGiven)
            model->B4SOIvgdrMax = 1e99;
        if (!model->B4SOIvgbrMaxGiven)
            model->B4SOIvgbrMax = 1e99;
        if (!model->B4SOIvbsrMaxGiven)
            model->B4SOIvbsrMax = 1e99;
        if (!model->B4SOIvbdrMaxGiven)
            model->B4SOIvbdrMax = 1e99;

        if (!model->B4SOIfdModGiven)
            model->B4SOIfdMod = 0;
        if (!model->B4SOIvsceGiven)
            model->B4SOIvsce = 0.0;
        if (!model->B4SOIcdsbsGiven)
            model->B4SOIcdsbs = 0.0;
        if (!model->B4SOIminvcvGiven)     /* v4.1 for Vgsteffcv */
            model->B4SOIminvcv = 0.0;
        if (!model->B4SOIlminvcvGiven)    /* v4.1 for Vgsteffcv */
            model->B4SOIlminvcv = 0.0;
        if (!model->B4SOIwminvcvGiven)    /* v4.1 for Vgsteffcv */
            model->B4SOIwminvcv = 0.0;
        if (!model->B4SOIpminvcvGiven)    /* v4.1 for Vgsteffcv */
            model->B4SOIpminvcv = 0.0;
        if (!model->B4SOIvoffcvGiven)
                        /*model->B4SOIvoffcv = -0.08;   v4.2 */
            model->B4SOIvoffcv = 0.0;
        if (!model->B4SOIlvoffcvGiven)
            model->B4SOIlvoffcv = 0.0;
        if (!model->B4SOIwvoffcvGiven)
            model->B4SOIwvoffcv = 0.0;
        if (!model->B4SOIpvoffcvGiven)
            model->B4SOIpvoffcv = 0.0;
        /* loop through all the instances of the model */
        for (here = B4SOIinstances(model); here != NULL ;
             here=B4SOInextInstance(here))
        {   /* allocate a chunk of the state vector */
            here->B4SOIstates = *states;
            *states += B4SOInumStates;
            /* perform the parameter defaulting */
            if (!here->B4SOIdrainAreaGiven)
                here->B4SOIdrainArea = 0.0;
            if (!here->B4SOIdrainPerimeterGiven)
                here->B4SOIdrainPerimeter = 0.0;
            if (!here->B4SOIdrainSquaresGiven)
                here->B4SOIdrainSquares = 1.0;
            if (!here->B4SOIicVBSGiven)
                here->B4SOIicVBS = 0;
            if (!here->B4SOIicVDSGiven)
                here->B4SOIicVDS = 0;
            if (!here->B4SOIicVGSGiven)
                here->B4SOIicVGS = 0;
            if (!here->B4SOIicVESGiven)
                here->B4SOIicVES = 0;
            if (!here->B4SOIicVPSGiven)
                here->B4SOIicVPS = 0;
            if (!here->B4SOIbjtoffGiven)
                here->B4SOIbjtoff = 0;
            if (!here->B4SOIdebugModGiven)
                here->B4SOIdebugMod = 0;
            if (!here->B4SOIrth0Given)
                here->B4SOIrth0 = model->B4SOIrth0;
            if (!here->B4SOIcth0Given)
                here->B4SOIcth0 = model->B4SOIcth0;
            if (!here->B4SOIbodySquaresGiven)
                here->B4SOIbodySquares = 1.0;
            if (!here->B4SOIfrbodyGiven)
                here->B4SOIfrbody = model->B4SOIfrbody; /* v4.0 */
            if (!here->B4SOIlGiven)
                here->B4SOIl = 5e-6;
            if (!here->B4SOIsourceAreaGiven)
                here->B4SOIsourceArea = 0;
            if (!here->B4SOIsourcePerimeterGiven)
                here->B4SOIsourcePerimeter = 0;
            if (!here->B4SOIsourceSquaresGiven)
                here->B4SOIsourceSquares = 1;
            if (!here->B4SOIwGiven)
                here->B4SOIw = 5e-6;
            if (!here->B4SOImGiven)
                here->B4SOIm = 1;  
/* v2.0 release */
            if (!here->B4SOInbcGiven)
                here->B4SOInbc = 0;
            if (!here->B4SOInsegGiven)
                here->B4SOInseg = 1;
            if (!here->B4SOIpdbcpGiven)
                here->B4SOIpdbcp = 0;
            if (!here->B4SOIpsbcpGiven)
                here->B4SOIpsbcp = 0;
            if (!here->B4SOIagbcpGiven)
                here->B4SOIagbcp = 0;
                        if (!here->B4SOIagbcp2Given)
                here->B4SOIagbcp2 = 0;   /* v4.1 */
            if (!here->B4SOIagbcpdGiven)
                here->B4SOIagbcpd = here->B4SOIagbcp;
            if (!here->B4SOIaebcpGiven)
                here->B4SOIaebcp = 0;

            if (!here->B4SOIoffGiven)
                here->B4SOIoff = 0;

            /* process drain series resistance */

            if ( ((model->B4SOIsheetResistance > 0.0) &&
                (here->B4SOIdrainSquares > 0.0 ) &&
                (here->B4SOIdNodePrime == 0)) )
            {   error = CKTmkVolt(ckt,&tmp,here->B4SOIname,"drain");
                if(error) return(error);
                here->B4SOIdNodePrime = tmp->number;
            }
            else
            {   here->B4SOIdNodePrime = here->B4SOIdNode;
            }

            /* process source series resistance */
            if ( ((model->B4SOIsheetResistance > 0.0) &&
                (here->B4SOIsourceSquares > 0.0 ) &&
                (here->B4SOIsNodePrime == 0)) )
            {   error = CKTmkVolt(ckt,&tmp,here->B4SOIname,"source");
                if(error) return(error);
                here->B4SOIsNodePrime = tmp->number;
            }
            else
            {   here->B4SOIsNodePrime = here->B4SOIsNode;
            }

/* v4.4 -- Check of TsiMax for SOIMOD = 2 */
    if (model->B4SOIsoiMod == 2){
        if (model->B4SOImtrlMod){
            NchMax = (model->B4SOIbg0sub - 0.1) / Charge_q * 2.0e-6 * epssub / (model->B4SOIetsi * model->B4SOIetsi);
           if (model->B4SOInpeak > NchMax ){
               printf("Warning: SOIMOD=2 can not support given Nch=%g cm^-3 and Etsi=%g m. \n ", model->B4SOInpeak, model->B4SOIetsi);
               printf("Exceeds maximum allowed band bending of (Eg-0.1)eV. \n");
               printf("Nch is set to  %g cm^-3. \n",NchMax);
               model->B4SOInpeak = NchMax;
           }
        } else {
            NchMax = (1.12 - 0.1) / Charge_q * 2.0e-6 * epssub / (model->B4SOItsi * model->B4SOItsi);
            if (model->B4SOInpeak > NchMax ) {
               printf("Warning: SOIMOD=2 can not support given Nch=%g cm^-3 and Tsi=%g m. \n", model->B4SOInpeak, model->B4SOItsi);
               printf("Exceeds maximum allowed band bending of (Eg-0.1)eV. \n");
               printf("Nch is set to  %g cm^-3. \n",NchMax);
               model->B4SOInpeak = NchMax;
            }
        }
    }

            /* process effective silicon film thickness */
            model->B4SOIcbox = 3.453133e-11 / model->B4SOItbox;
                        if(model->B4SOImtrlMod)
                        {
            model->B4SOIcsi = 1.03594e-10 / model->B4SOIetsi;
                        }
                        else
                        {
                        model->B4SOIcsi = 1.03594e-10 / model->B4SOItsi;
                        }
            Cboxt = model->B4SOIcbox * model->B4SOIcsi / (model->B4SOIcbox + model->B4SOIcsi);


/* v3.2 */
            if(model->B4SOImtrlMod)
                        {
            Qsi = Charge_q * model->B4SOInpeak
                * (1.0 + model->B4SOIlpe0 / here->B4SOIl) * 1e6 * model->B4SOIetsi;
                    }
                        else
                        {
                        Qsi = Charge_q * model->B4SOInpeak
                * (1.0 + model->B4SOIlpe0 / here->B4SOIl) * 1e6 * model->B4SOItsi;
                        }
            Vbs0t = 0.8 - 0.5 * Qsi / model->B4SOIcsi + model->B4SOIvbsa;

            if (!here->B4SOIsoiModGiven)
                here->B4SOIsoiMod = model->B4SOIsoiMod;
            else if ((here->B4SOIsoiMod != 0) && (here->B4SOIsoiMod != 1)
                 && (here->B4SOIsoiMod != 2) && (here->B4SOIsoiMod != 3))
            {   here->B4SOIsoiMod = model->B4SOIsoiMod;
                printf("Warning: soiMod has been set to its global value %d.\n", model->B4SOIsoiMod);
            }

            if (here->B4SOIsoiMod == 3) { /* auto selection */
               if (Vbs0t > model->B4SOIvbs0fd)
                  here->B4SOIsoiMod = 2; /* ideal FD mode */
               else {
                  if (Vbs0t < model->B4SOIvbs0pd)
                     here->B4SOIsoiMod = 0; /* BSIMPD */
                  else
                     here->B4SOIsoiMod = 1;
               }
            }

            here->B4SOIpNode = here->B4SOIpNodeExt;
            here->B4SOIbNode = here->B4SOIbNodeExt;
            here->B4SOItempNode = here->B4SOItempNodeExt;

            here->B4SOIfloat = 0;
            if (here->B4SOIsoiMod == 2) /* v3.2 */
            {
               here->B4SOIbNode = here->B4SOIpNode = 0;
               here->B4SOIbodyMod = 0;
            } /* For ideal FD, body contact is disabled and no body node */
            else
            {
               if (here->B4SOIpNode == -1) {  /* floating body case -- 4-node */
                  error = CKTmkVolt(ckt,&tmp,here->B4SOIname,"Body");
                  if (error) return(error);
                  here->B4SOIbNode = tmp->number;
                  here->B4SOIpNode = 0;
                  here->B4SOIfloat = 1;
                  here->B4SOIbodyMod = 0;

               }
               else /* the 5th Node has been assigned */
               {
                 if (!here->B4SOItnodeoutGiven) { /* if t-node not assigned  */
                    if (here->B4SOIbNode == -1)
                    { /* 5-node body tie, bNode has not been assigned */
                       if ((model->B4SOIrbody == 0.0) && (model->B4SOIrbsh == 0.0))
                       { /* ideal body tie, pNode is not used */
                          here->B4SOIbNode = here->B4SOIpNode;
                          here->B4SOIbodyMod = 2;
                       }
                       else { /* nonideal body tie */
                         error = CKTmkVolt(ckt, &tmp, here->B4SOIname, "Body");
                         if (error) return(error);
                         here->B4SOIbNode = tmp->number;
                         here->B4SOIbodyMod = 1;
                       }
                    }
                    else { /* 6-node body tie, bNode has been assigned */
                       if ((model->B4SOIrbody == 0.0) && (model->B4SOIrbsh == 0.0))
                       {
                          printf("\n      Warning: model parameter rbody=0!\n");
                          model->B4SOIrbody = 1e0;
                          here->B4SOIbodyMod = 1;
                       }
                       else { /* nonideal body tie */
                           here->B4SOIbodyMod = 1;
                       }
                    }
                 }
                 else {    /*  t-node assigned   */
                    if (here->B4SOIbNode == -1)
                    { /* 4 nodes & t-node, floating body */
                       error = CKTmkVolt(ckt, &tmp, here->B4SOIname, "Body");
                       if (error) return(error);
                       here->B4SOIbNode = tmp->number;
                       here->B4SOItempNode = here->B4SOIpNode;
                       here->B4SOIpNode = 0;
                       here->B4SOIfloat = 1;
                       here->B4SOIbodyMod = 0;
                    }
                    else { /* 5 or 6 nodes & t-node, body-contact device */
                      if (here->B4SOItempNode == -1) { /* 5 nodes & tnode */
                        if ((model->B4SOIrbody == 0.0) && (model->B4SOIrbsh == 0.0))
                        { /* ideal body tie, pNode is not used */
                           here->B4SOItempNode = here->B4SOIbNode;
                           here->B4SOIbNode = here->B4SOIpNode;
                           here->B4SOIbodyMod = 2;
                        }
                        else { /* nonideal body tie */
                          here->B4SOItempNode = here->B4SOIbNode;
                          error = CKTmkVolt(ckt, &tmp, here->B4SOIname, "Body");
                          if (error) return(error);
                          here->B4SOIbNode = tmp->number;
                          here->B4SOIbodyMod = 1;
                        }
                      }
                      else {  /* 6 nodes & t-node */
                        if ((model->B4SOIrbody == 0.0) && (model->B4SOIrbsh == 0.0))
                        {
                           printf("\n      Warning: model parameter rbody=0!\n");
                           model->B4SOIrbody = 1e0;
                           here->B4SOIbodyMod = 1;
                        }
                        else { /* nonideal body tie */
                           here->B4SOIbodyMod = 1;
                        }
                      }
                    }
                 }
               }
            }


            if ((model->B4SOIshMod == 1) && (here->B4SOIrth0!=0))
            {
               if (here->B4SOItempNode == -1) {
                  error = CKTmkVolt(ckt,&tmp,here->B4SOIname,"Temp");
                  if (error) return(error);
                     here->B4SOItempNode = tmp->number;
               }

            } else {
                here->B4SOItempNode = 0;
            }

/* v3.1 added for RF */
            if (!here->B4SOIrgateModGiven)
                here->B4SOIrgateMod = model->B4SOIrgateMod;
            else if ((here->B4SOIrgateMod != 0) && (here->B4SOIrgateMod != 1)
                && (here->B4SOIrgateMod != 2) && (here->B4SOIrgateMod != 3))
            {   here->B4SOIrgateMod = model->B4SOIrgateMod;
                printf("Warning: rgateMod has been set to its global value %d.\n", model->B4SOIrgateMod);
            }

            if (here->B4SOIrgateMod > 0)
            {   error = CKTmkVolt(ckt,&tmp,here->B4SOIname,"gate");
                if(error) return(error);
                   here->B4SOIgNode = tmp->number;
                if (here->B4SOIrgateMod == 1)
                {   if (model->B4SOIrshg <= 0.0)
                        printf("Warning: rshg should be positive for rgateMod = 1.\n");
                }
                else if (here->B4SOIrgateMod == 2)
                {   if (model->B4SOIrshg <= 0.0)
                        printf("Warning: rshg <= 0.0 for rgateMod = 2!!!\n");
                    else if (model->B4SOIxrcrg1 <= 0.0)
                        printf("Warning: xrcrg1 <= 0.0 for rgateMod = 2!!!\n");
                }
            }
            else
            {
                here->B4SOIgNode = here->B4SOIgNodeExt;
            }

            if (here->B4SOIrgateMod == 3)
            {   error = CKTmkVolt(ckt,&tmp,here->B4SOIname,"midgate");
                if(error) return(error);
                   here->B4SOIgNodeMid = tmp->number;
                if (model->B4SOIrshg <= 0.0)
                    printf("Warning: rshg should be positive for rgateMod = 3.\n");
                else if (model->B4SOIxrcrg1 <= 0.0)
                    printf("Warning: xrcrg1 should be positive for rgateMod = 3. \n");
            }
            else
                here->B4SOIgNodeMid = here->B4SOIgNodeExt;
/* v3.1 added for RF end */

/* v4.0 */
            if (!here->B4SOIrbodyModGiven)
                here->B4SOIrbodyMod = model->B4SOIrbodyMod;
            else if ((here->B4SOIrbodyMod != 0) && (here->B4SOIrbodyMod != 1))
            {   here->B4SOIrbodyMod = model->B4SOIrbodyMod;
                printf("Warning: rbodyMod has been set to its global value %d.  \n", model->B4SOIrbodyMod);
            }

            if (here->B4SOIrbodyMod ==1 && here->B4SOIsoiMod == 2) /* v4.0 */
                here->B4SOIrbodyMod = 0;

            if (here->B4SOIrbodyMod == 1)
            {   if (here->B4SOIdbNode == 0)
                {   error = CKTmkVolt(ckt,&tmp,here->B4SOIname,"dbody");
                    if(error) return(error);
                    here->B4SOIdbNode = tmp->number;
                }
                if (here->B4SOIsbNode == 0)
                {   error = CKTmkVolt(ckt,&tmp,here->B4SOIname,"sbody");
                    if(error) return(error);
                    here->B4SOIsbNode = tmp->number;
                }
            }
            else
                here->B4SOIdbNode = here->B4SOIsbNode
                                  = here->B4SOIbNode;

/* v4.0 end */

            /* v4.0 added */
            if (!here->B4SOIsaGiven) /* stress */
                here->B4SOIsa = 0.0;
            if (!here->B4SOIsbGiven)
                here->B4SOIsb = 0.0;
            if (!here->B4SOIsdGiven)
                here->B4SOIsd = 0.0;
            if (!here->B4SOInfGiven)
                here->B4SOInf = 1.0;
            if (!here->B4SOIrbdbGiven)
                here->B4SOIrbdb = model->B4SOIrbdb; /* in ohm */
            if (!here->B4SOIrbsbGiven)
                here->B4SOIrbsb = model->B4SOIrbsb;
            if (!here->B4SOIdelvtoGiven)
                here->B4SOIdelvto = 0.0;

            /* v4.0 added end */


/* here for debugging purpose only */
            if (here->B4SOIdebugMod != 0)
            {
               /* The real Vbs value */
               error = CKTmkVolt(ckt, &tmp, here->B4SOIname, "Vbs");
               if(error) return(error);
               here->B4SOIvbsNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B4SOIname, "Ids");
               if(error) return(error);
               here->B4SOIidsNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B4SOIname, "Ic");
               if(error) return(error);
               here->B4SOIicNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B4SOIname, "Ibs");
               if(error) return(error);
               here->B4SOIibsNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B4SOIname, "Ibd");
               if(error) return(error);
               here->B4SOIibdNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B4SOIname, "Iii");
               if(error) return(error);
               here->B4SOIiiiNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B4SOIname, "Ig");
               if(error) return(error);
               here->B4SOIigNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B4SOIname, "Gigg");
               if(error) return(error);
               here->B4SOIgiggNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B4SOIname, "Gigd");
               if(error) return(error);
               here->B4SOIgigdNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B4SOIname, "Gigb");
               if(error) return(error);
               here->B4SOIgigbNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B4SOIname, "Igidl");
               if(error) return(error);
               here->B4SOIigidlNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B4SOIname, "Itun");
               if(error) return(error);
               here->B4SOIitunNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B4SOIname, "Ibp");
               if(error) return(error);
               here->B4SOIibpNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B4SOIname, "Cbb");
               if(error) return(error);
               here->B4SOIcbbNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B4SOIname, "Cbd");
               if(error) return(error);
               here->B4SOIcbdNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B4SOIname, "Cbg");
               if(error) return(error);
               here->B4SOIcbgNode = tmp->number;


               error = CKTmkVolt(ckt, &tmp, here->B4SOIname, "Qbf");
               if(error) return(error);
               here->B4SOIqbfNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B4SOIname, "Qjs");
               if(error) return(error);
               here->B4SOIqjsNode = tmp->number;

               error = CKTmkVolt(ckt, &tmp, here->B4SOIname, "Qjd");
               if(error) return(error);
               here->B4SOIqjdNode = tmp->number;


           }

        /* set Sparse Matrix Pointers */

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==(double *)NULL){\
    return(E_NOMEM);\
} } while(0)


        if ((model->B4SOIshMod == 1) && (here->B4SOIrth0!=0.0)) {
            TSTALLOC(B4SOITemptempPtr, B4SOItempNode, B4SOItempNode);
            TSTALLOC(B4SOITempdpPtr, B4SOItempNode, B4SOIdNodePrime);
            TSTALLOC(B4SOITempspPtr, B4SOItempNode, B4SOIsNodePrime);
            TSTALLOC(B4SOITempgPtr, B4SOItempNode, B4SOIgNode);
            TSTALLOC(B4SOITempbPtr, B4SOItempNode, B4SOIbNode);

            TSTALLOC(B4SOIGtempPtr, B4SOIgNode, B4SOItempNode);

            TSTALLOC(B4SOIDPtempPtr, B4SOIdNodePrime, B4SOItempNode);
            TSTALLOC(B4SOISPtempPtr, B4SOIsNodePrime, B4SOItempNode);
            TSTALLOC(B4SOIEtempPtr, B4SOIeNode, B4SOItempNode);
            TSTALLOC(B4SOIBtempPtr, B4SOIbNode, B4SOItempNode);

            if (here->B4SOIbodyMod == 1) {
                TSTALLOC(B4SOIPtempPtr, B4SOIpNode, B4SOItempNode);
            }

/* v3.0 */
            if (here->B4SOIsoiMod != 0) { /* v3.2 */
               TSTALLOC(B4SOITempePtr, B4SOItempNode, B4SOIeNode);
            }

        }
        if (here->B4SOIbodyMod == 2) {
            /* Don't create any Jacobian entry for pNode */
        }
        else if (here->B4SOIbodyMod == 1) {
            TSTALLOC(B4SOIBpPtr, B4SOIbNode, B4SOIpNode);
            TSTALLOC(B4SOIPbPtr, B4SOIpNode, B4SOIbNode);
            TSTALLOC(B4SOIPpPtr, B4SOIpNode, B4SOIpNode);

            /* 4.1 for Igb2_agbcp2 */
            TSTALLOC(B4SOIPgPtr , B4SOIpNode, B4SOIgNode);
            TSTALLOC(B4SOIGpPtr , B4SOIgNode, B4SOIpNode);
        }


/* v3.1 added for RF */
            if (here->B4SOIrgateMod != 0)
            {   TSTALLOC(B4SOIGEgePtr, B4SOIgNodeExt, B4SOIgNodeExt);
                TSTALLOC(B4SOIGEgPtr, B4SOIgNodeExt, B4SOIgNode);
                TSTALLOC(B4SOIGgePtr, B4SOIgNode, B4SOIgNodeExt);
                TSTALLOC(B4SOIGEdpPtr, B4SOIgNodeExt, B4SOIdNodePrime);
                TSTALLOC(B4SOIGEspPtr, B4SOIgNodeExt, B4SOIsNodePrime);
                if (here->B4SOIsoiMod !=2) /* v3.2 */
                   TSTALLOC(B4SOIGEbPtr, B4SOIgNodeExt, B4SOIbNode);

                TSTALLOC(B4SOIGMdpPtr, B4SOIgNodeMid, B4SOIdNodePrime);
                TSTALLOC(B4SOIGMgPtr, B4SOIgNodeMid, B4SOIgNode);
                TSTALLOC(B4SOIGMgmPtr, B4SOIgNodeMid, B4SOIgNodeMid);
                TSTALLOC(B4SOIGMgePtr, B4SOIgNodeMid, B4SOIgNodeExt);
                TSTALLOC(B4SOIGMspPtr, B4SOIgNodeMid, B4SOIsNodePrime);
                if (here->B4SOIsoiMod !=2) /* v3.2 */
                   TSTALLOC(B4SOIGMbPtr, B4SOIgNodeMid, B4SOIbNode);

                TSTALLOC(B4SOIGMePtr, B4SOIgNodeMid, B4SOIeNode);
                TSTALLOC(B4SOIDPgmPtr, B4SOIdNodePrime, B4SOIgNodeMid);
                TSTALLOC(B4SOIGgmPtr, B4SOIgNode, B4SOIgNodeMid);
                TSTALLOC(B4SOIGEgmPtr, B4SOIgNodeExt, B4SOIgNodeMid);
                TSTALLOC(B4SOISPgmPtr, B4SOIsNodePrime, B4SOIgNodeMid);
                TSTALLOC(B4SOIEgmPtr, B4SOIeNode, B4SOIgNodeMid);
            }
/* v3.1 added for RF end */


/* v3.1 */
        if (here->B4SOIsoiMod != 2) /* v3.2 */
        {
           TSTALLOC(B4SOIEbPtr, B4SOIeNode, B4SOIbNode);
           TSTALLOC(B4SOIGbPtr, B4SOIgNode, B4SOIbNode);
           TSTALLOC(B4SOIDPbPtr, B4SOIdNodePrime, B4SOIbNode);
           TSTALLOC(B4SOISPbPtr, B4SOIsNodePrime, B4SOIbNode);
           TSTALLOC(B4SOIBePtr, B4SOIbNode, B4SOIeNode);
           TSTALLOC(B4SOIBgPtr, B4SOIbNode, B4SOIgNode);
           TSTALLOC(B4SOIBdpPtr, B4SOIbNode, B4SOIdNodePrime);
           TSTALLOC(B4SOIBspPtr, B4SOIbNode, B4SOIsNodePrime);
           TSTALLOC(B4SOIBbPtr, B4SOIbNode, B4SOIbNode);
        }
/* v3.1 */


        TSTALLOC(B4SOIEgPtr, B4SOIeNode, B4SOIgNode);
        TSTALLOC(B4SOIEdpPtr, B4SOIeNode, B4SOIdNodePrime);
        TSTALLOC(B4SOIEspPtr, B4SOIeNode, B4SOIsNodePrime);
        TSTALLOC(B4SOIGePtr, B4SOIgNode, B4SOIeNode);
        TSTALLOC(B4SOIDPePtr, B4SOIdNodePrime, B4SOIeNode);
        TSTALLOC(B4SOISPePtr, B4SOIsNodePrime, B4SOIeNode);

        TSTALLOC(B4SOIEePtr, B4SOIeNode, B4SOIeNode);

        TSTALLOC(B4SOIGgPtr, B4SOIgNode, B4SOIgNode);
        TSTALLOC(B4SOIGdpPtr, B4SOIgNode, B4SOIdNodePrime);
        TSTALLOC(B4SOIGspPtr, B4SOIgNode, B4SOIsNodePrime);

        TSTALLOC(B4SOIDPgPtr, B4SOIdNodePrime, B4SOIgNode);
        TSTALLOC(B4SOIDPdpPtr, B4SOIdNodePrime, B4SOIdNodePrime);
        TSTALLOC(B4SOIDPspPtr, B4SOIdNodePrime, B4SOIsNodePrime);
        TSTALLOC(B4SOIDPdPtr, B4SOIdNodePrime, B4SOIdNode);

        TSTALLOC(B4SOISPgPtr, B4SOIsNodePrime, B4SOIgNode);
        TSTALLOC(B4SOISPdpPtr, B4SOIsNodePrime, B4SOIdNodePrime);
        TSTALLOC(B4SOISPspPtr, B4SOIsNodePrime, B4SOIsNodePrime);
        TSTALLOC(B4SOISPsPtr, B4SOIsNodePrime, B4SOIsNode);

        TSTALLOC(B4SOIDdPtr, B4SOIdNode, B4SOIdNode);
        TSTALLOC(B4SOIDdpPtr, B4SOIdNode, B4SOIdNodePrime);

        TSTALLOC(B4SOISsPtr, B4SOIsNode, B4SOIsNode);
        TSTALLOC(B4SOISspPtr, B4SOIsNode, B4SOIsNodePrime);

/* v4.0 */
            if (here->B4SOIrbodyMod == 1)
            {   TSTALLOC(B4SOIDPdbPtr, B4SOIdNodePrime, B4SOIdbNode);
                TSTALLOC(B4SOISPsbPtr, B4SOIsNodePrime, B4SOIsbNode);

                TSTALLOC(B4SOIDBdpPtr, B4SOIdbNode, B4SOIdNodePrime);
                TSTALLOC(B4SOIDBdbPtr, B4SOIdbNode, B4SOIdbNode);
                TSTALLOC(B4SOIDBbPtr, B4SOIdbNode, B4SOIbNode);

                TSTALLOC(B4SOISBspPtr, B4SOIsbNode, B4SOIsNodePrime);
                TSTALLOC(B4SOISBsbPtr, B4SOIsbNode, B4SOIsbNode);
                TSTALLOC(B4SOISBbPtr, B4SOIsbNode, B4SOIbNode);

                TSTALLOC(B4SOIBdbPtr, B4SOIbNode, B4SOIdbNode);
                TSTALLOC(B4SOIBsbPtr, B4SOIbNode, B4SOIsbNode);
            }

            if (model->B4SOIrdsMod)
            {   TSTALLOC(B4SOIDgPtr,  B4SOIdNode, B4SOIgNode);
                TSTALLOC(B4SOIDspPtr, B4SOIdNode, B4SOIsNodePrime);
                TSTALLOC(B4SOISdpPtr, B4SOIsNode, B4SOIdNodePrime);
                TSTALLOC(B4SOISgPtr,  B4SOIsNode, B4SOIgNode);
                if (model->B4SOIsoiMod != 2)  {
                        TSTALLOC(B4SOIDbPtr, B4SOIdNode, B4SOIbNode);
                        TSTALLOC(B4SOISbPtr, B4SOIsNode, B4SOIbNode);
                }
            }


/* v4.0 end*/

/* here for debugging purpose only */
         if (here->B4SOIdebugMod != 0)
         {
            TSTALLOC(B4SOIVbsPtr, B4SOIvbsNode, B4SOIvbsNode);
            TSTALLOC(B4SOIIdsPtr, B4SOIidsNode, B4SOIidsNode);
            TSTALLOC(B4SOIIcPtr, B4SOIicNode, B4SOIicNode);
            TSTALLOC(B4SOIIbsPtr, B4SOIibsNode, B4SOIibsNode);
            TSTALLOC(B4SOIIbdPtr, B4SOIibdNode, B4SOIibdNode);
            TSTALLOC(B4SOIIiiPtr, B4SOIiiiNode, B4SOIiiiNode);
            TSTALLOC(B4SOIIgPtr, B4SOIigNode, B4SOIigNode);
            TSTALLOC(B4SOIGiggPtr, B4SOIgiggNode, B4SOIgiggNode);
            TSTALLOC(B4SOIGigdPtr, B4SOIgigdNode, B4SOIgigdNode);
            TSTALLOC(B4SOIGigbPtr, B4SOIgigbNode, B4SOIgigbNode);
            TSTALLOC(B4SOIIgidlPtr, B4SOIigidlNode, B4SOIigidlNode);
            TSTALLOC(B4SOIItunPtr, B4SOIitunNode, B4SOIitunNode);
            TSTALLOC(B4SOIIbpPtr, B4SOIibpNode, B4SOIibpNode);
            TSTALLOC(B4SOICbbPtr, B4SOIcbbNode, B4SOIcbbNode);
            TSTALLOC(B4SOICbdPtr, B4SOIcbdNode, B4SOIcbdNode);
            TSTALLOC(B4SOICbgPtr, B4SOIcbgNode, B4SOIcbgNode);
            TSTALLOC(B4SOIQbfPtr, B4SOIqbfNode, B4SOIqbfNode);
            TSTALLOC(B4SOIQjsPtr, B4SOIqjsNode, B4SOIqjsNode);
            TSTALLOC(B4SOIQjdPtr, B4SOIqjdNode, B4SOIqjdNode);


         }

        }
    }

#ifdef USE_OMP
    InstCount = 0;
    model = (B4SOImodel*)inModel;
    /* loop through all the B4SOI device models 
       to count the number of instances */
    
    for( ; model != NULL; model = B4SOInextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = B4SOIinstances(model); here != NULL ;
             here=B4SOInextInstance(here)) 
        { 
            InstCount++;
        }
        model->B4SOIInstCount = 0;
        model->B4SOIInstanceArray = NULL;
    }
    InstArray = TMALLOC(B4SOIinstance*, InstCount);
    model = (B4SOImodel*)inModel;
    /* store this in the first model only */
    model->B4SOIInstCount = InstCount;
    model->B4SOIInstanceArray = InstArray;
    idx = 0;
    for( ; model != NULL; model = B4SOInextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = B4SOIinstances(model); here != NULL ;
             here=B4SOInextInstance(here)) 
        { 
            InstArray[idx] = here;
            idx++;
        }
    }
#endif

    return(OK);
}

int
B4SOIunsetup(
    GENmodel *inModel,
    CKTcircuit *ckt)
{
#ifndef HAS_BATCHSIM
    B4SOImodel *model;
    B4SOIinstance *here;

#ifdef USE_OMP
    model = (B4SOImodel*)inModel;
    tfree(model->B4SOIInstanceArray);
#endif

    for (model = (B4SOImodel *)inModel; model != NULL;
            model = B4SOInextModel(model))
    {
        for (here = B4SOIinstances(model); here != NULL;
                here=B4SOInextInstance(here))
        {
            /* here for debugging purpose only */
            if (here->B4SOIqjdNode > 0)
                CKTdltNNum(ckt, here->B4SOIqjdNode);
            here->B4SOIqjdNode = 0;

            if (here->B4SOIqjsNode > 0)
                CKTdltNNum(ckt, here->B4SOIqjsNode);
            here->B4SOIqjsNode = 0;

            if (here->B4SOIqbfNode > 0)
                CKTdltNNum(ckt, here->B4SOIqbfNode);
            here->B4SOIqbfNode = 0;

            if (here->B4SOIcbgNode > 0)
                CKTdltNNum(ckt, here->B4SOIcbgNode);
            here->B4SOIcbgNode = 0;

            if (here->B4SOIcbdNode > 0)
                CKTdltNNum(ckt, here->B4SOIcbdNode);
            here->B4SOIcbdNode = 0;

            if (here->B4SOIcbbNode > 0)
                CKTdltNNum(ckt, here->B4SOIcbbNode);
            here->B4SOIcbbNode = 0;

            if (here->B4SOIibpNode > 0)
                CKTdltNNum(ckt, here->B4SOIibpNode);
            here->B4SOIibpNode = 0;

            if (here->B4SOIitunNode > 0)
                CKTdltNNum(ckt, here->B4SOIitunNode);
            here->B4SOIitunNode = 0;

            if (here->B4SOIigidlNode > 0)
                CKTdltNNum(ckt, here->B4SOIigidlNode);
            here->B4SOIigidlNode = 0;

            if (here->B4SOIgigbNode > 0)
                CKTdltNNum(ckt, here->B4SOIgigbNode);
            here->B4SOIgigbNode = 0;

            if (here->B4SOIgigdNode > 0)
                CKTdltNNum(ckt, here->B4SOIgigdNode);
            here->B4SOIgigdNode = 0;

            if (here->B4SOIgiggNode > 0)
                CKTdltNNum(ckt, here->B4SOIgiggNode);
            here->B4SOIgiggNode = 0;

            if (here->B4SOIigNode > 0)
                CKTdltNNum(ckt, here->B4SOIigNode);
            here->B4SOIigNode = 0;

            if (here->B4SOIiiiNode > 0)
                CKTdltNNum(ckt, here->B4SOIiiiNode);
            here->B4SOIiiiNode = 0;

            if (here->B4SOIibdNode > 0)
                CKTdltNNum(ckt, here->B4SOIibdNode);
            here->B4SOIibdNode = 0;

            if (here->B4SOIibsNode > 0)
                CKTdltNNum(ckt, here->B4SOIibsNode);
            here->B4SOIibsNode = 0;

            if (here->B4SOIicNode > 0)
                CKTdltNNum(ckt, here->B4SOIicNode);
            here->B4SOIicNode = 0;

            if (here->B4SOIidsNode > 0)
                CKTdltNNum(ckt, here->B4SOIidsNode);
            here->B4SOIidsNode = 0;

            if (here->B4SOIvbsNode > 0)
                CKTdltNNum(ckt, here->B4SOIvbsNode);
            here->B4SOIvbsNode = 0;

            if (here->B4SOIsbNode > 0 &&
                here->B4SOIsbNode != here->B4SOIbNode)
                CKTdltNNum(ckt, here->B4SOIsbNode);
            here->B4SOIsbNode = 0;

            if (here->B4SOIdbNode > 0 &&
                here->B4SOIdbNode != here->B4SOIbNode)
                CKTdltNNum(ckt, here->B4SOIdbNode);
            here->B4SOIdbNode = 0;

            if (here->B4SOIgNodeMid > 0 &&
                here->B4SOIgNodeMid != here->B4SOIgNodeExt)
                CKTdltNNum(ckt, here->B4SOIgNodeMid);
            here->B4SOIgNodeMid = 0;

            if (here->B4SOIgNode > 0 &&
                here->B4SOIgNode != here->B4SOIgNodeExt)
                CKTdltNNum(ckt, here->B4SOIgNode);
            here->B4SOIgNode = 0;

            if (here->B4SOItempNode > 0 &&
                here->B4SOItempNode != here->B4SOItempNodeExt &&
                here->B4SOItempNode != here->B4SOIbNodeExt &&
                here->B4SOItempNode != here->B4SOIpNodeExt)
                CKTdltNNum(ckt, here->B4SOItempNode);
            here->B4SOItempNode = 0;

            if (here->B4SOIbNode > 0 &&
                here->B4SOIbNode != here->B4SOIbNodeExt &&
                here->B4SOIbNode != here->B4SOIpNodeExt)
                CKTdltNNum(ckt, here->B4SOIbNode);
            here->B4SOIbNode = 0;

            here->B4SOIpNode = 0;

            if (here->B4SOIsNodePrime > 0
                    && here->B4SOIsNodePrime != here->B4SOIsNode)
                CKTdltNNum(ckt, here->B4SOIsNodePrime);
            here->B4SOIsNodePrime = 0;

            if (here->B4SOIdNodePrime > 0
                    && here->B4SOIdNodePrime != here->B4SOIdNode)
                CKTdltNNum(ckt, here->B4SOIdNodePrime);
            here->B4SOIdNodePrime = 0;
        }
    }
#endif
    return OK;
}
