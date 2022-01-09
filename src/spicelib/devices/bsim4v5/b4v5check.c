/**** BSIM4.5.0 Released by Xuemei (Jane) Xi 07/29/2005 ****/

/**********
 * Copyright 2005 Regents of the University of California. All rights reserved.
 * File: b4check.c of BSIM4.5.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 * Modified by Xuemei Xi, 03/04/2004.
 * Modified by Xuemei Xi, 07/29/2005.
 * Modified by Holger Vogt, 12/21/2020.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v5def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"
#include "ngspice/wordlist.h"
#include "ngspice/cpextern.h"

/* Check for correctness of the BSIM4.5 parameters:
   If parameter excursions are found, put the warning or error message into a wordlist. 
   Only then open a file bsim4v5.out and print the data into the file. */
int
BSIM4v5checkModel(
BSIM4v5model *model,
BSIM4v5instance *here,
CKTcircuit *ckt)
{
    struct bsim4v5SizeDependParam *pParam;
    int Fatal_Flag = 0;
    FILE *fplog;
    wordlist* wl, *wlstart;

    if (cp_getvar("ng_nomodcheck", CP_BOOL, NULL, 0))
        return(0);

    static char modname[BSIZE_SP];
    size_t mlen = strlen(model->BSIM4v5modName);

    if (mlen < BSIZE_SP) {
        /* Check the model named model->BSIM4v5modName only once,
           because BSIM4v5checkModel() is called for each instance. */
        if (!strncmp(modname, model->BSIM4v5modName, mlen))
            return(0);
        strcpy(modname, model->BSIM4v5modName);
    }

    pParam = here->pParam;

    wl = wlstart = TMALLOC(wordlist, 1);
    wl->wl_prev = NULL;
    wl->wl_next = NULL;
    wl->wl_word = tprintf("\nChecking parameters for BSIM 4.5 model %s\n", model->BSIM4v5modName);

    if ((strcmp(model->BSIM4v5version, "4.5.0")) && (strncmp(model->BSIM4v5version, "4.50", 4)) && (strncmp(model->BSIM4v5version, "4.5", 3))
        && (strcmp(model->BSIM4v5version, "4.4.0")) && (strncmp(model->BSIM4v5version, "4.40", 4)) && (strncmp(model->BSIM4v5version, "4.4", 3))
        && (strcmp(model->BSIM4v5version, "4.3.0")) && (strncmp(model->BSIM4v5version, "4.30", 4)) && (strncmp(model->BSIM4v5version, "4.3", 3))
        && (strcmp(model->BSIM4v5version, "4.2.0")) && (strncmp(model->BSIM4v5version, "4.20", 4)) && (strncmp(model->BSIM4v5version, "4.2", 3))
        && (strcmp(model->BSIM4v5version, "4.1.0")) && (strncmp(model->BSIM4v5version, "4.10", 4)) && (strncmp(model->BSIM4v5version, "4.1", 3))
        && (strcmp(model->BSIM4v5version, "4.0.0")) && (strncmp(model->BSIM4v5version, "4.00", 4)) && (strncmp(model->BSIM4v5version, "4.0", 3)))
    {
        printf("Warning: This model supports BSIM4 versions 4.0, 4.1, 4.2, 4.3, 4.4, 4.5\n");
        printf("You specified a wrong version number. Working now with BSIM4v5\n");
        wl_append_word(&wl, &wl, tprintf("Warning: This model supports BSIM4 versions 4.0, 4.1, 4.2, 4.3, 4.4, 4.5\n"));
        wl_append_word(&wl, &wl, tprintf("You specified a wrong version number. Working now with BSIM4v5.\n"));
    }

    if ((here->BSIM4v5rgateMod == 2) || (here->BSIM4v5rgateMod == 3))
    {   if ((here->BSIM4v5trnqsMod == 1) || (here->BSIM4v5acnqsMod == 1)) {
            wl_append_word(&wl, &wl, tprintf("Warning: You've selected both Rg and charge deficit NQS; select one only.\n"));
        }
    }

    if (model->BSIM4v5toxe <= 0.0)
    {
         wl_append_word(&wl, &wl, tprintf("Fatal: Toxe = %g is not positive.\n",
            model->BSIM4v5toxe));
        Fatal_Flag = 1;
    }

    if (model->BSIM4v5toxp <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Toxp = %g is not positive.\n", model->BSIM4v5toxp));
        Fatal_Flag = 1;
    }

    if (model->BSIM4v5toxm <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Toxm = %g is not positive.\n", model->BSIM4v5toxm));
        Fatal_Flag = 1;
    }

    if (model->BSIM4v5toxref <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Toxref = %g is not positive.\n", model->BSIM4v5toxref));
        Fatal_Flag = 1;
    }



    if (pParam->BSIM4v5lpe0 < -pParam->BSIM4v5leff)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Lpe0 = %g is less than -Leff.\n",
            pParam->BSIM4v5lpe0));
        Fatal_Flag = 1;
    }
    if (model->BSIM4v5lintnoi > pParam->BSIM4v5leff / 2)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Lintnoi = %g is too large - Leff for noise is negative.\n",
            model->BSIM4v5lintnoi));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM4v5lpeb < -pParam->BSIM4v5leff)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Lpeb = %g is less than -Leff.\n",
            pParam->BSIM4v5lpeb));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM4v5ndep <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Ndep = %g is not positive.\n",
            pParam->BSIM4v5ndep));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM4v5phi <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Phi = %g is not positive. Please check Phin and Ndep\n",
            pParam->BSIM4v5phi));
        wl_append_word(&wl, &wl, tprintf("	   Phin = %g  Ndep = %g \n",
            pParam->BSIM4v5phin, pParam->BSIM4v5ndep));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM4v5nsub <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Nsub = %g is not positive.\n",
            pParam->BSIM4v5nsub));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM4v5ngate < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Ngate = %g Ngate is not positive.\n",
            pParam->BSIM4v5ngate));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM4v5ngate > 1.e25)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Ngate = %g Ngate is too high\n",
            pParam->BSIM4v5ngate));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM4v5xj <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Xj = %g is not positive.\n", pParam->BSIM4v5xj));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM4v5dvt1 < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Dvt1 = %g is negative.\n", pParam->BSIM4v5dvt1));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM4v5dvt1w < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Dvt1w = %g is negative.\n", pParam->BSIM4v5dvt1w));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM4v5w0 == -pParam->BSIM4v5weff)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n"));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM4v5dsub < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Dsub = %g is negative.\n", pParam->BSIM4v5dsub));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM4v5b1 == -pParam->BSIM4v5weff)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n"));
        Fatal_Flag = 1;
    }
    if (here->BSIM4v5u0temp <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: u0 at current temperature = %g is not positive.\n",
            here->BSIM4v5u0temp));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM4v5delta < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Delta = %g is less than zero.\n", pParam->BSIM4v5delta));
        Fatal_Flag = 1;
    }

    if (here->BSIM4v5vsattemp <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Vsat at current temperature = %g is not positive.\n",
            here->BSIM4v5vsattemp));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM4v5pclm <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Pclm = %g is not positive.\n", pParam->BSIM4v5pclm));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM4v5drout < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Drout = %g is negative.\n", pParam->BSIM4v5drout));
        Fatal_Flag = 1;
    }

    if (here->BSIM4v5m <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: multiplier = %g is not positive.\n", here->BSIM4v5m));
        Fatal_Flag = 1;
    }
    if (here->BSIM4v5nf < 1.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Number of finger = %g is smaller than one.\n", here->BSIM4v5nf));
        Fatal_Flag = 1;
    }

    if ((here->BSIM4v5sa > 0.0) && (here->BSIM4v5sb > 0.0) &&
        ((here->BSIM4v5nf == 1.0) || ((here->BSIM4v5nf > 1.0) && (here->BSIM4v5sd > 0.0))))
    {
        if (model->BSIM4v5saref <= 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Fatal: SAref = %g is not positive.\n", model->BSIM4v5saref));
            Fatal_Flag = 1;
        }
        if (model->BSIM4v5sbref <= 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Fatal: SBref = %g is not positive.\n", model->BSIM4v5sbref));
            Fatal_Flag = 1;
        }
    }

    if ((here->BSIM4v5l + model->BSIM4v5xl) <= model->BSIM4v5xgl)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: The parameter xgl must be smaller than Ldrawn+XL.\n"));
        Fatal_Flag = 1;
    }
    if (here->BSIM4v5ngcon < 1.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: The parameter ngcon cannot be smaller than one.\n"));
        Fatal_Flag = 1;
    }
    if ((here->BSIM4v5ngcon != 1.0) && (here->BSIM4v5ngcon != 2.0))
    {
        here->BSIM4v5ngcon = 1.0;
        wl_append_word(&wl, &wl, tprintf("Warning: Ngcon must be equal to one or two; reset to 1.0.\n"));
    }

    if (model->BSIM4v5gbmin < 1.0e-20)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Gbmin = %g is too small.\n", model->BSIM4v5gbmin));
    }

    /* Check saturation parameters */
    if (pParam->BSIM4v5fprout < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: fprout = %g is negative.\n", pParam->BSIM4v5fprout));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM4v5pdits < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: pdits = %g is negative.\n", pParam->BSIM4v5pdits));
        Fatal_Flag = 1;
    }
    if (model->BSIM4v5pditsl < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: pditsl = %g is negative.\n", model->BSIM4v5pditsl));
        Fatal_Flag = 1;
    }

    /* Check gate current parameters */
    if (model->BSIM4v5igbMod) {
        if (pParam->BSIM4v5nigbinv <= 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Fatal: nigbinv = %g is non-positive.\n", pParam->BSIM4v5nigbinv));
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v5nigbacc <= 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Fatal: nigbacc = %g is non-positive.\n", pParam->BSIM4v5nigbacc));
            Fatal_Flag = 1;
        }
    }
    if (model->BSIM4v5igcMod) {
        if (pParam->BSIM4v5nigc <= 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Fatal: nigc = %g is non-positive.\n", pParam->BSIM4v5nigc));
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v5poxedge <= 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Fatal: poxedge = %g is non-positive.\n", pParam->BSIM4v5poxedge));
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v5pigcd <= 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Fatal: pigcd = %g is non-positive.\n", pParam->BSIM4v5pigcd));
            Fatal_Flag = 1;
        }
    }

    /* Check capacitance parameters */
    if (pParam->BSIM4v5clc < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Clc = %g is negative.\n", pParam->BSIM4v5clc));
        Fatal_Flag = 1;
    }

    /* Check overlap capacitance parameters */
    if (pParam->BSIM4v5ckappas < 0.02)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: ckappas = %g is too small.\n", pParam->BSIM4v5ckappas));
        pParam->BSIM4v5ckappas = 0.02;
    }
    if (pParam->BSIM4v5ckappad < 0.02)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: ckappad = %g is too small.\n", pParam->BSIM4v5ckappad));
        pParam->BSIM4v5ckappad = 0.02;
    }

    if (model->BSIM4v5vtss < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Vtss = %g is negative.\n",
            model->BSIM4v5vtss));
        Fatal_Flag = 1;
    }
    if (model->BSIM4v5vtsd < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Vtsd = %g is negative.\n",
            model->BSIM4v5vtsd));
        Fatal_Flag = 1;
    }
    if (model->BSIM4v5vtssws < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Vtssws = %g is negative.\n",
            model->BSIM4v5vtssws));
        Fatal_Flag = 1;
    }
    if (model->BSIM4v5vtsswd < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Vtsswd = %g is negative.\n",
            model->BSIM4v5vtsswd));
        Fatal_Flag = 1;
    }
    if (model->BSIM4v5vtsswgs < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Vtsswgs = %g is negative.\n",
            model->BSIM4v5vtsswgs));
        Fatal_Flag = 1;
    }
    if (model->BSIM4v5vtsswgd < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Vtsswgd = %g is negative.\n",
            model->BSIM4v5vtsswgd));
        Fatal_Flag = 1;
    }


    if (model->BSIM4v5paramChk == 1)
    {
        /* Check L and W parameters */
        if (pParam->BSIM4v5leff <= 1.0e-9)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Leff = %g <= 1.0e-9. Recommended Leff >= 1e-8 \n",
                pParam->BSIM4v5leff));
        }

        if (pParam->BSIM4v5leffCV <= 1.0e-9)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Leff for CV = %g <= 1.0e-9. Recommended LeffCV >=1e-8 \n",
                pParam->BSIM4v5leffCV));
        }

        if (pParam->BSIM4v5weff <= 1.0e-9)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Weff = %g <= 1.0e-9. Recommended Weff >=1e-7 \n",
                pParam->BSIM4v5weff));
        }

        if (pParam->BSIM4v5weffCV <= 1.0e-9)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Weff for CV = %g <= 1.0e-9. Recommended WeffCV >= 1e-7 \n",
                pParam->BSIM4v5weffCV));
        }

        /* Check threshold voltage parameters */
        if (model->BSIM4v5toxe < 1.0e-10)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Toxe = %g is less than 1A. Recommended Toxe >= 5A\n", model->BSIM4v5toxe));
        }
        if (model->BSIM4v5toxp < 1.0e-10)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Toxp = %g is less than 1A. Recommended Toxp >= 5A\n", model->BSIM4v5toxp));
        }
        if (model->BSIM4v5toxm < 1.0e-10)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Toxm = %g is less than 1A. Recommended Toxm >= 5A\n", model->BSIM4v5toxm));
        }

        if (pParam->BSIM4v5ndep <= 1.0e12)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Ndep = %g may be too small.\n",
                pParam->BSIM4v5ndep));
        }
        else if (pParam->BSIM4v5ndep >= 1.0e21)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Ndep = %g may be too large.\n",
                pParam->BSIM4v5ndep));
        }

        if (pParam->BSIM4v5nsub <= 1.0e14)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Nsub = %g may be too small.\n",
                pParam->BSIM4v5nsub));
        }
        else if (pParam->BSIM4v5nsub >= 1.0e21)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Nsub = %g may be too large.\n",
                pParam->BSIM4v5nsub));
        }

        if ((pParam->BSIM4v5ngate > 0.0) &&
            (pParam->BSIM4v5ngate <= 1.e18))
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Ngate = %g is less than 1.E18cm^-3.\n",
                pParam->BSIM4v5ngate));
        }

        if (pParam->BSIM4v5dvt0 < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Dvt0 = %g is negative.\n", pParam->BSIM4v5dvt0));
        }

        if (fabs(1.0e-8 / (pParam->BSIM4v5w0 + pParam->BSIM4v5weff)) > 10.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: (W0 + Weff) may be too small.\n"));
        }

        /* Check subthreshold parameters */
        if (pParam->BSIM4v5nfactor < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Nfactor = %g is negative.\n", pParam->BSIM4v5nfactor));
        }
        if (pParam->BSIM4v5cdsc < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Cdsc = %g is negative.\n", pParam->BSIM4v5cdsc));
        }
        if (pParam->BSIM4v5cdscd < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Cdscd = %g is negative.\n", pParam->BSIM4v5cdscd));
        }
        /* Check DIBL parameters */
        if (here->BSIM4v5eta0 < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Eta0 = %g is negative.\n", here->BSIM4v5eta0));
        }

        /* Check Abulk parameters */
        if (fabs(1.0e-8 / (pParam->BSIM4v5b1 + pParam->BSIM4v5weff)) > 10.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: (B1 + Weff) may be too small.\n"));
        }


        /* Check Saturation parameters */
        if (pParam->BSIM4v5a2 < 0.01)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: A2 = %g is too small. Set to 0.01.\n",
                pParam->BSIM4v5a2));
            pParam->BSIM4v5a2 = 0.01;
        }
        else if (pParam->BSIM4v5a2 > 1.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
                pParam->BSIM4v5a2));
            pParam->BSIM4v5a2 = 1.0;
            pParam->BSIM4v5a1 = 0.0;
        }

        if (pParam->BSIM4v5prwg < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Prwg = %g is negative. Set to zero.\n",
                pParam->BSIM4v5prwg));
            pParam->BSIM4v5prwg = 0.0;
        }

        if (pParam->BSIM4v5rdsw < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Rdsw = %g is negative. Set to zero.\n",
                pParam->BSIM4v5rdsw));
            pParam->BSIM4v5rdsw = 0.0;
            pParam->BSIM4v5rds0 = 0.0;
        }

        if (pParam->BSIM4v5rds0 < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Rds at current temperature = %g is negative. Set to zero.\n",
                pParam->BSIM4v5rds0));
            pParam->BSIM4v5rds0 = 0.0;
        }

        if (pParam->BSIM4v5rdswmin < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Rdswmin at current temperature = %g is negative. Set to zero.\n",
                pParam->BSIM4v5rdswmin));
            pParam->BSIM4v5rdswmin = 0.0;
        }

        if (pParam->BSIM4v5pscbe2 <= 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Pscbe2 = %g is not positive.\n", pParam->BSIM4v5pscbe2));
        }

        if (pParam->BSIM4v5vsattemp < 1.0e3)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM4v5vsattemp));
        }

        if ((model->BSIM4v5lambdaGiven) && (pParam->BSIM4v5lambda > 0.0))
        {
            if (pParam->BSIM4v5lambda > 1.0e-9)
            {
                wl_append_word(&wl, &wl, tprintf("Warning: Lambda = %g may be too large.\n", pParam->BSIM4v5lambda));
            }
        }

        if ((model->BSIM4v5vtlGiven) && (pParam->BSIM4v5vtl > 0.0))
        {
            if (pParam->BSIM4v5vtl < 6.0e4)
            {
                wl_append_word(&wl, &wl, tprintf("Warning: Thermal velocity vtl = %g may be too small.\n", pParam->BSIM4v5vtl));
            }

            if (pParam->BSIM4v5xn < 3.0)
            {
                wl_append_word(&wl, &wl, tprintf("Warning: back scattering coeff xn = %g is too small. Reset to 3.0 \n", pParam->BSIM4v5xn));
                pParam->BSIM4v5xn = 3.0;
            }

            if (model->BSIM4v5lc < 0.0)
            {
                wl_append_word(&wl, &wl, tprintf("Warning: back scattering coeff lc = %g is too small. Reset to 0.0\n", model->BSIM4v5lc));
                pParam->BSIM4v5lc = 0.0;
            }
        }

        if (pParam->BSIM4v5pdibl1 < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Pdibl1 = %g is negative.\n", pParam->BSIM4v5pdibl1));
        }
    }

    if (pParam->BSIM4v5pdibl2 < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Pdibl2 = %g is negative.\n", pParam->BSIM4v5pdibl2));
    }

    /* Check stress effect parameters */
    if ((here->BSIM4v5sa > 0.0) && (here->BSIM4v5sb > 0.0) &&
        ((here->BSIM4v5nf == 1.0) || ((here->BSIM4v5nf > 1.0) && (here->BSIM4v5sd > 0.0))))
    {
        if (model->BSIM4v5lodk2 <= 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: LODK2 = %g is not positive.\n", model->BSIM4v5lodk2));
        }
        if (model->BSIM4v5lodeta0 <= 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: LODETA0 = %g is not positive.\n", model->BSIM4v5lodeta0));
        }
    }

    /* Check gate resistance parameters */
    if (here->BSIM4v5rgateMod == 1)
    {
        if (model->BSIM4v5rshg <= 0.0)
            wl_append_word(&wl, &wl, tprintf("Warning: rshg should be positive for rgateMod = 1.\n"));
    }
    else if (here->BSIM4v5rgateMod == 2)
    {
        if (model->BSIM4v5rshg <= 0.0)
            wl_append_word(&wl, &wl, tprintf("Warning: rshg <= 0.0 for rgateMod = 2.\n"));
        else if (pParam->BSIM4v5xrcrg1 <= 0.0)
                wl_append_word(&wl, &wl, tprintf("Warning: xrcrg1 <= 0.0 for rgateMod = 2.\n"));
    }
    if (here->BSIM4v5rgateMod == 3)
    {
        if (model->BSIM4v5rshg <= 0.0)
            wl_append_word(&wl, &wl, tprintf("Warning: rshg should be positive for rgateMod = 3.\n"));
        else if (pParam->BSIM4v5xrcrg1 <= 0.0)
                wl_append_word(&wl, &wl, tprintf("Warning: xrcrg1 should be positive for rgateMod = 3.\n"));
    }

    /* Check capacitance parameters */
    if (pParam->BSIM4v5noff < 0.1)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Noff = %g is too small.\n", pParam->BSIM4v5noff));
    }

    if (pParam->BSIM4v5voffcv < -0.5)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Voffcv = %g is too small.\n", pParam->BSIM4v5voffcv));
    }

    if (pParam->BSIM4v5moin < 5.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Moin = %g is too small.\n", pParam->BSIM4v5moin));
    }
    if (pParam->BSIM4v5moin > 25.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Moin = %g is too large.\n", pParam->BSIM4v5moin));
    }
    if (model->BSIM4v5capMod == 2) {
        if (pParam->BSIM4v5acde < 0.1)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Acde = %g is too small.\n", pParam->BSIM4v5acde));
        }
        if (pParam->BSIM4v5acde > 1.6)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Acde = %g is too large.\n", pParam->BSIM4v5acde));
        }
    }

    /* Check overlap capacitance parameters */
    if (model->BSIM4v5cgdo < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM4v5cgdo));
        model->BSIM4v5cgdo = 0.0;
    }
    if (model->BSIM4v5cgso < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: cgso = %g is negative. Set to zero.\n", model->BSIM4v5cgso));
        model->BSIM4v5cgso = 0.0;
    }
    if (model->BSIM4v5cgbo < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: cgbo = %g is negative. Set to zero.\n", model->BSIM4v5cgbo));
        model->BSIM4v5cgbo = 0.0;
    }
    if (model->BSIM4v5tnoiMod == 1) {
        if (model->BSIM4v5tnoia < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: tnoia = %g is negative. Set to zero.\n", model->BSIM4v5tnoia));
            model->BSIM4v5tnoia = 0.0;
        }
        if (model->BSIM4v5tnoib < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: tnoib = %g is negative. Set to zero.\n", model->BSIM4v5tnoib));
            model->BSIM4v5tnoib = 0.0;
        }

        if (model->BSIM4v5rnoia < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: rnoia = %g is negative. Set to zero.\n", model->BSIM4v5rnoia));
            model->BSIM4v5rnoia = 0.0;
        }
        if (model->BSIM4v5rnoib < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: rnoib = %g is negative. Set to zero.\n", model->BSIM4v5rnoib));
            model->BSIM4v5rnoib = 0.0;
        }
    }

    if (model->BSIM4v5SjctEmissionCoeff < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Njs = %g is negative.\n",
            model->BSIM4v5SjctEmissionCoeff));
    }
    if (model->BSIM4v5DjctEmissionCoeff < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Njd = %g is negative.\n",
            model->BSIM4v5DjctEmissionCoeff));
    }
    if (model->BSIM4v5njtstemp < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Njts = %g is negative at temperature = %g.\n",
            model->BSIM4v5njtstemp, ckt->CKTtemp));
    }
    if (model->BSIM4v5njtsswtemp < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Njtssw = %g is negative at temperature = %g.\n",
            model->BSIM4v5njtsswtemp, ckt->CKTtemp));
    }
    if (model->BSIM4v5njtsswgtemp < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Njtsswg = %g is negative at temperature = %g.\n",
            model->BSIM4v5njtsswgtemp, ckt->CKTtemp));
    }
    if (model->BSIM4v5ntnoi < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: ntnoi = %g is negative. Set to zero.\n", model->BSIM4v5ntnoi));
        model->BSIM4v5ntnoi = 0.0;
    }

    /* diode model */
    if (model->BSIM4v5SbulkJctBotGradingCoeff >= 0.99)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: MJS = %g is too big. Set to 0.99.\n", model->BSIM4v5SbulkJctBotGradingCoeff));
        model->BSIM4v5SbulkJctBotGradingCoeff = 0.99;
    }
    if (model->BSIM4v5SbulkJctSideGradingCoeff >= 0.99)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: MJSWS = %g is too big. Set to 0.99.\n", model->BSIM4v5SbulkJctSideGradingCoeff));
        model->BSIM4v5SbulkJctSideGradingCoeff = 0.99;
    }
    if (model->BSIM4v5SbulkJctGateSideGradingCoeff >= 0.99)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: MJSWGS = %g is too big. Set to 0.99.\n", model->BSIM4v5SbulkJctGateSideGradingCoeff));
        model->BSIM4v5SbulkJctGateSideGradingCoeff = 0.99;
    }

    if (model->BSIM4v5DbulkJctBotGradingCoeff >= 0.99)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: MJD = %g is too big. Set to 0.99.\n", model->BSIM4v5DbulkJctBotGradingCoeff));
        model->BSIM4v5DbulkJctBotGradingCoeff = 0.99;
    }
    if (model->BSIM4v5DbulkJctSideGradingCoeff >= 0.99)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: MJSWD = %g is too big. Set to 0.99.\n", model->BSIM4v5DbulkJctSideGradingCoeff));
        model->BSIM4v5DbulkJctSideGradingCoeff = 0.99;
    }
    if (model->BSIM4v5DbulkJctGateSideGradingCoeff >= 0.99)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: MJSWGD = %g is too big. Set to 0.99.\n", model->BSIM4v5DbulkJctGateSideGradingCoeff));
        model->BSIM4v5DbulkJctGateSideGradingCoeff = 0.99;
    }
    if (model->BSIM4v5wpemod == 1)
    {
        if (model->BSIM4v5scref <= 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: SCREF = %g is not positive. Set to 1e-6.\n", model->BSIM4v5scref));
            model->BSIM4v5scref = 1e-6;
        }
        if (here->BSIM4v5sca < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: SCA = %g is negative. Set to 0.0.\n", here->BSIM4v5sca));
            here->BSIM4v5sca = 0.0;
        }
        if (here->BSIM4v5scb < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: SCB = %g is negative. Set to 0.0.\n", here->BSIM4v5scb));
            here->BSIM4v5scb = 0.0;
        }
        if (here->BSIM4v5scc < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: SCC = %g is negative. Set to 0.0.\n", here->BSIM4v5scc));
            here->BSIM4v5scc = 0.0;
        }
        if (here->BSIM4v5sc < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: SC = %g is negative. Set to 0.0.\n", here->BSIM4v5sc));
            here->BSIM4v5sc = 0.0;
        }
    }

    wordlist* wlfree = wlstart;
    if (wlstart->wl_next) {
        if ((fplog = fopen("bsim4v5.out", "w")) != NULL) {
            while (wlstart) {
                fprintf(fplog, "%s", wlstart->wl_word);
                fprintf(stderr, "%s", wlstart->wl_word);
                wlstart = wlstart->wl_next;
            }
            fclose(fplog);
        }
        else {
            while (wlstart) {
                fprintf(stderr, "%s", wlstart->wl_word);
                wlstart = wlstart->wl_next;
            }
        }
    }

    wl_free(wlfree);

    return(Fatal_Flag);
}

