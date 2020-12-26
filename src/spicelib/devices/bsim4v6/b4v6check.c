/**** BSIM4.6.2 Released by Wenwei Yang 07/31/2008 ****/
/**** BSIM4.6.5 Update ngspice 09/22/2009 ****/
/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4check.c of BSIM4.6.1.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 * Modified by Xuemei Xi, 03/04/2004.
 * Modified by Xuemei Xi, 07/29/2005.
 * Modified by Mohan Dunga, 12/13/2006
 * Modified by Mohan Dunga, Wenwei Yang, 05/18/2007.
 * Modified by  Wenwei Yang, 07/31/2008 .
 * Modified by Holger Vogt, 12/27/2020.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v6def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"
#include "ngspice/wordlist.h"
#include "ngspice/cpextern.h"



/* Check for correctness of the BSIM4.5 parameters:
   If parameter excursions are found, put the warning or error message into a wordlist. 
   Only then open a file bsim4v6.out and print the data into the file. */
int
BSIM4v6checkModel(
BSIM4v6model *model,
BSIM4v6instance *here,
CKTcircuit *ckt)
{
    struct bsim4v6SizeDependParam *pParam;
    int Fatal_Flag = 0;
    FILE *fplog;
    wordlist* wl, *wlstart;

    pParam = here->pParam;

    if (cp_getvar("ng_nomodcheck", CP_BOOL, NULL, 0))
        return(0);

    wl = wlstart = TMALLOC(wordlist, 1);
    wl->wl_prev = NULL;
    wl->wl_next = NULL;
    wl->wl_word = tprintf("\nChecking parameters for BSIM 4.5 model %s\n", model->BSIM4v6modName);

    if ((here->BSIM4v6rgateMod == 2) || (here->BSIM4v6rgateMod == 3))
    {   if ((here->BSIM4v6trnqsMod == 1) || (here->BSIM4v6acnqsMod == 1)) {
            wl_append_word(&wl, &wl, tprintf("Warning: You've selected both Rg and charge deficit NQS; select one only.\n"));
        }
    }

    if (model->BSIM4v6toxe <= 0.0)
    {
         wl_append_word(&wl, &wl, tprintf("Fatal: Toxe = %g is not positive.\n",
            model->BSIM4v6toxe));
        Fatal_Flag = 1;
    }

    if (model->BSIM4v6toxp <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Toxp = %g is not positive.\n", model->BSIM4v6toxp));
        Fatal_Flag = 1;
    }

    if (model->BSIM4v6eot <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: EOT = %g is not positive.\n", model->BSIM4v6eot));
        Fatal_Flag = 1;
    }
    if (model->BSIM4v6epsrgate < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Epsrgate = %g is not positive.\n", model->BSIM4v6epsrgate));
        Fatal_Flag = 1;
    }
    if (model->BSIM4v6epsrsub < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Epsrsub = %g is not positive.\n", model->BSIM4v6epsrsub));
        Fatal_Flag = 1;
    }
    if (model->BSIM4v6easub < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Easub = %g is not positive.\n", model->BSIM4v6easub));
        Fatal_Flag = 1;
    }
    if (model->BSIM4v6ni0sub <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Easub = %g is not positive.\n", model->BSIM4v6ni0sub));
        Fatal_Flag = 1;
    }

    if (model->BSIM4v6toxm <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Toxm = %g is not positive.\n", model->BSIM4v6toxm));
        Fatal_Flag = 1;
    }
    if (model->BSIM4v6toxref <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Toxref = %g is not positive.\n", model->BSIM4v6toxref));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM4v6lpe0 < -pParam->BSIM4v6leff)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Lpe0 = %g is less than -Leff.\n",
            pParam->BSIM4v6lpe0));
        Fatal_Flag = 1;
    }
    if (model->BSIM4v6lintnoi > pParam->BSIM4v6leff / 2)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Lintnoi = %g is too large - Leff for noise is negative.\n",
            model->BSIM4v6lintnoi));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM4v6lpeb < -pParam->BSIM4v6leff)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Lpeb = %g is less than -Leff.\n",
            pParam->BSIM4v6lpeb));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM4v6ndep <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Ndep = %g is not positive.\n",
            pParam->BSIM4v6ndep));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM4v6phi <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Phi = %g is not positive. Please check Phin and Ndep\n",
            pParam->BSIM4v6phi));
        wl_append_word(&wl, &wl, tprintf("	   Phin = %g  Ndep = %g \n",
            pParam->BSIM4v6phin, pParam->BSIM4v6ndep));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM4v6nsub <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Nsub = %g is not positive.\n",
            pParam->BSIM4v6nsub));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM4v6ngate < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Ngate = %g Ngate is not positive.\n",
            pParam->BSIM4v6ngate));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM4v6ngate > 1.e25)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Ngate = %g Ngate is too high\n",
            pParam->BSIM4v6ngate));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM4v6xj <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Xj = %g is not positive.\n", pParam->BSIM4v6xj));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM4v6dvt1 < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Dvt1 = %g is negative.\n", pParam->BSIM4v6dvt1));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM4v6dvt1w < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Dvt1w = %g is negative.\n", pParam->BSIM4v6dvt1w));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM4v6w0 == -pParam->BSIM4v6weff)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n"));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM4v6dsub < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Dsub = %g is negative.\n", pParam->BSIM4v6dsub));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM4v6b1 == -pParam->BSIM4v6weff)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n"));
        Fatal_Flag = 1;
    }
    if (here->BSIM4v6u0temp <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: u0 at current temperature = %g is not positive.\n",
            here->BSIM4v6u0temp));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM4v6delta < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Delta = %g is less than zero.\n", pParam->BSIM4v6delta));
        Fatal_Flag = 1;
    }

    if (here->BSIM4v6vsattemp <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Vsat at current temperature = %g is not positive.\n",
            here->BSIM4v6vsattemp));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM4v6pclm <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Pclm = %g is not positive.\n", pParam->BSIM4v6pclm));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM4v6drout < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Drout = %g is negative.\n", pParam->BSIM4v6drout));
        Fatal_Flag = 1;
    }

    if (here->BSIM4v6m <= 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: multiplier = %g is not positive.\n", here->BSIM4v6m));
        Fatal_Flag = 1;
    }
    if (here->BSIM4v6nf < 1.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Number of finger = %g is smaller than one.\n", here->BSIM4v6nf));
        Fatal_Flag = 1;
    }

    if ((here->BSIM4v6sa > 0.0) && (here->BSIM4v6sb > 0.0) &&
        ((here->BSIM4v6nf == 1.0) || ((here->BSIM4v6nf > 1.0) && (here->BSIM4v6sd > 0.0))))
    {
        if (model->BSIM4v6saref <= 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Fatal: SAref = %g is not positive.\n", model->BSIM4v6saref));
            Fatal_Flag = 1;
        }
        if (model->BSIM4v6sbref <= 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Fatal: SBref = %g is not positive.\n", model->BSIM4v6sbref));
            Fatal_Flag = 1;
        }
    }

    if ((here->BSIM4v6l + model->BSIM4v6xl) <= model->BSIM4v6xgl)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: The parameter xgl must be smaller than Ldrawn+XL.\n"));
        Fatal_Flag = 1;
    }
    if (here->BSIM4v6ngcon < 1.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: The parameter ngcon cannot be smaller than one.\n"));
        Fatal_Flag = 1;
    }
    if ((here->BSIM4v6ngcon != 1.0) && (here->BSIM4v6ngcon != 2.0))
    {
        here->BSIM4v6ngcon = 1.0;
        wl_append_word(&wl, &wl, tprintf("Warning: Ngcon must be equal to one or two; reset to 1.0.\n"));
    }

    if (model->BSIM4v6gbmin < 1.0e-20)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Gbmin = %g is too small.\n", model->BSIM4v6gbmin));
    }

    /* Check saturation parameters */
    if (pParam->BSIM4v6fprout < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: fprout = %g is negative.\n", pParam->BSIM4v6fprout));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM4v6pdits < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: pdits = %g is negative.\n", pParam->BSIM4v6pdits));
        Fatal_Flag = 1;
    }
    if (model->BSIM4v6pditsl < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: pditsl = %g is negative.\n", model->BSIM4v6pditsl));
        Fatal_Flag = 1;
    }

    /* Check gate current parameters */
    if (model->BSIM4v6igbMod) {
        if (pParam->BSIM4v6nigbinv <= 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Fatal: nigbinv = %g is non-positive.\n", pParam->BSIM4v6nigbinv));
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v6nigbacc <= 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Fatal: nigbacc = %g is non-positive.\n", pParam->BSIM4v6nigbacc));
            Fatal_Flag = 1;
        }
    }
    if (model->BSIM4v6igcMod) {
        if (pParam->BSIM4v6nigc <= 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Fatal: nigc = %g is non-positive.\n", pParam->BSIM4v6nigc));
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v6poxedge <= 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Fatal: poxedge = %g is non-positive.\n", pParam->BSIM4v6poxedge));
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v6pigcd <= 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Fatal: pigcd = %g is non-positive.\n", pParam->BSIM4v6pigcd));
            Fatal_Flag = 1;
        }
    }

    /* Check capacitance parameters */
    if (pParam->BSIM4v6clc < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Clc = %g is negative.\n", pParam->BSIM4v6clc));
        Fatal_Flag = 1;
    }

    /* Check overlap capacitance parameters */
    if (pParam->BSIM4v6ckappas < 0.02)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: ckappas = %g is too small.\n", pParam->BSIM4v6ckappas));
        pParam->BSIM4v6ckappas = 0.02;
    }
    if (pParam->BSIM4v6ckappad < 0.02)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: ckappad = %g is too small.\n", pParam->BSIM4v6ckappad));
        pParam->BSIM4v6ckappad = 0.02;
    }

    if (model->BSIM4v6vtss < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Vtss = %g is negative.\n",
            model->BSIM4v6vtss));
        Fatal_Flag = 1;
    }
    if (model->BSIM4v6vtsd < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Vtsd = %g is negative.\n",
            model->BSIM4v6vtsd));
        Fatal_Flag = 1;
    }
    if (model->BSIM4v6vtssws < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Vtssws = %g is negative.\n",
            model->BSIM4v6vtssws));
        Fatal_Flag = 1;
    }
    if (model->BSIM4v6vtsswd < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Vtsswd = %g is negative.\n",
            model->BSIM4v6vtsswd));
        Fatal_Flag = 1;
    }
    if (model->BSIM4v6vtsswgs < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Vtsswgs = %g is negative.\n",
            model->BSIM4v6vtsswgs));
        Fatal_Flag = 1;
    }
    if (model->BSIM4v6vtsswgd < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Fatal: Vtsswgd = %g is negative.\n",
            model->BSIM4v6vtsswgd));
        Fatal_Flag = 1;
    }


    if (model->BSIM4v6paramChk == 1)
    {
        /* Check L and W parameters */
        if (pParam->BSIM4v6leff <= 1.0e-9)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Leff = %g <= 1.0e-9. Recommended Leff >= 1e-8 \n",
                pParam->BSIM4v6leff));
        }

        if (pParam->BSIM4v6leffCV <= 1.0e-9)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Leff for CV = %g <= 1.0e-9. Recommended LeffCV >=1e-8 \n",
                pParam->BSIM4v6leffCV));
        }

        if (pParam->BSIM4v6weff <= 1.0e-9)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Weff = %g <= 1.0e-9. Recommended Weff >=1e-7 \n",
                pParam->BSIM4v6weff));
        }

        if (pParam->BSIM4v6weffCV <= 1.0e-9)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Weff for CV = %g <= 1.0e-9. Recommended WeffCV >= 1e-7 \n",
                pParam->BSIM4v6weffCV));
        }

        /* Check threshold voltage parameters */
        if (model->BSIM4v6toxe < 1.0e-10)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Toxe = %g is less than 1A. Recommended Toxe >= 5A\n", model->BSIM4v6toxe));
        }
        if (model->BSIM4v6toxp < 1.0e-10)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Toxp = %g is less than 1A. Recommended Toxp >= 5A\n", model->BSIM4v6toxp));
        }
        if (model->BSIM4v6toxm < 1.0e-10)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Toxm = %g is less than 1A. Recommended Toxm >= 5A\n", model->BSIM4v6toxm));
        }

        if (pParam->BSIM4v6ndep <= 1.0e12)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Ndep = %g may be too small.\n",
                pParam->BSIM4v6ndep));
        }
        else if (pParam->BSIM4v6ndep >= 1.0e21)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Ndep = %g may be too large.\n",
                pParam->BSIM4v6ndep));
        }

        if (pParam->BSIM4v6nsub <= 1.0e14)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Nsub = %g may be too small.\n",
                pParam->BSIM4v6nsub));
        }
        else if (pParam->BSIM4v6nsub >= 1.0e21)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Nsub = %g may be too large.\n",
                pParam->BSIM4v6nsub));
        }

        if ((pParam->BSIM4v6ngate > 0.0) &&
            (pParam->BSIM4v6ngate <= 1.e18))
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Ngate = %g is less than 1.E18cm^-3.\n",
                pParam->BSIM4v6ngate));
        }

        if (pParam->BSIM4v6dvt0 < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Dvt0 = %g is negative.\n", pParam->BSIM4v6dvt0));
        }

        if (fabs(1.0e-8 / (pParam->BSIM4v6w0 + pParam->BSIM4v6weff)) > 10.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: (W0 + Weff) may be too small.\n"));
        }

        /* Check subthreshold parameters */
        if (pParam->BSIM4v6nfactor < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Nfactor = %g is negative.\n", pParam->BSIM4v6nfactor));
        }
        if (pParam->BSIM4v6cdsc < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Cdsc = %g is negative.\n", pParam->BSIM4v6cdsc));
        }
        if (pParam->BSIM4v6cdscd < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Cdscd = %g is negative.\n", pParam->BSIM4v6cdscd));
        }
        /* Check DIBL parameters */
        if (here->BSIM4v6eta0 < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Eta0 = %g is negative.\n", here->BSIM4v6eta0));
        }

        /* Check Abulk parameters */
        if (fabs(1.0e-8 / (pParam->BSIM4v6b1 + pParam->BSIM4v6weff)) > 10.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: (B1 + Weff) may be too small.\n"));
        }


        /* Check Saturation parameters */
        if (pParam->BSIM4v6a2 < 0.01)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: A2 = %g is too small. Set to 0.01.\n",
                pParam->BSIM4v6a2));
            pParam->BSIM4v6a2 = 0.01;
        }
        else if (pParam->BSIM4v6a2 > 1.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
                pParam->BSIM4v6a2));
            pParam->BSIM4v6a2 = 1.0;
            pParam->BSIM4v6a1 = 0.0;
        }

        if (pParam->BSIM4v6prwg < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Prwg = %g is negative. Set to zero.\n",
                pParam->BSIM4v6prwg));
            pParam->BSIM4v6prwg = 0.0;
        }

        if (pParam->BSIM4v6rdsw < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Rdsw = %g is negative. Set to zero.\n",
                pParam->BSIM4v6rdsw));
            pParam->BSIM4v6rdsw = 0.0;
            pParam->BSIM4v6rds0 = 0.0;
        }

        if (pParam->BSIM4v6rds0 < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Rds at current temperature = %g is negative. Set to zero.\n",
                pParam->BSIM4v6rds0));
            pParam->BSIM4v6rds0 = 0.0;
        }

        if (pParam->BSIM4v6rdswmin < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Rdswmin at current temperature = %g is negative. Set to zero.\n",
                pParam->BSIM4v6rdswmin));
            pParam->BSIM4v6rdswmin = 0.0;
        }

        if (pParam->BSIM4v6pscbe2 <= 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Pscbe2 = %g is not positive.\n", pParam->BSIM4v6pscbe2));
        }

        if (pParam->BSIM4v6vsattemp < 1.0e3)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM4v6vsattemp));
        }

        if ((model->BSIM4v6lambdaGiven) && (pParam->BSIM4v6lambda > 0.0))
        {
            if (pParam->BSIM4v6lambda > 1.0e-9)
            {
                wl_append_word(&wl, &wl, tprintf("Warning: Lambda = %g may be too large.\n", pParam->BSIM4v6lambda));
            }
        }

        if ((model->BSIM4v6vtlGiven) && (pParam->BSIM4v6vtl > 0.0))
        {
            if (pParam->BSIM4v6vtl < 6.0e4)
            {
                wl_append_word(&wl, &wl, tprintf("Warning: Thermal velocity vtl = %g may be too small.\n", pParam->BSIM4v6vtl));
            }

            if (pParam->BSIM4v6xn < 3.0)
            {
                wl_append_word(&wl, &wl, tprintf("Warning: back scattering coeff xn = %g is too small. Reset to 3.0 \n", pParam->BSIM4v6xn));
                pParam->BSIM4v6xn = 3.0;
            }

            if (model->BSIM4v6lc < 0.0)
            {
                wl_append_word(&wl, &wl, tprintf("Warning: back scattering coeff lc = %g is too small. Reset to 0.0\n", model->BSIM4v6lc));
                pParam->BSIM4v6lc = 0.0;
            }
        }

        if (pParam->BSIM4v6pdibl1 < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Pdibl1 = %g is negative.\n", pParam->BSIM4v6pdibl1));
        }
    }

    if (pParam->BSIM4v6pdibl2 < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Pdibl2 = %g is negative.\n", pParam->BSIM4v6pdibl2));
    }

    /* Check stress effect parameters */
    if ((here->BSIM4v6sa > 0.0) && (here->BSIM4v6sb > 0.0) &&
        ((here->BSIM4v6nf == 1.0) || ((here->BSIM4v6nf > 1.0) && (here->BSIM4v6sd > 0.0))))
    {
        if (model->BSIM4v6lodk2 <= 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: LODK2 = %g is not positive.\n", model->BSIM4v6lodk2));
        }
        if (model->BSIM4v6lodeta0 <= 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: LODETA0 = %g is not positive.\n", model->BSIM4v6lodeta0));
        }
    }

    /* Check gate resistance parameters */
    if (here->BSIM4v6rgateMod == 1)
    {
        if (model->BSIM4v6rshg <= 0.0)
            wl_append_word(&wl, &wl, tprintf("Warning: rshg should be positive for rgateMod = 1.\n"));
    }
    else if (here->BSIM4v6rgateMod == 2)
    {
        if (model->BSIM4v6rshg <= 0.0)
            wl_append_word(&wl, &wl, tprintf("Warning: rshg <= 0.0 for rgateMod = 2.\n"));
        else if (pParam->BSIM4v6xrcrg1 <= 0.0)
                wl_append_word(&wl, &wl, tprintf("Warning: xrcrg1 <= 0.0 for rgateMod = 2.\n"));
    }
    if (here->BSIM4v6rgateMod == 3)
    {
        if (model->BSIM4v6rshg <= 0.0)
            wl_append_word(&wl, &wl, tprintf("Warning: rshg should be positive for rgateMod = 3.\n"));
        else if (pParam->BSIM4v6xrcrg1 <= 0.0)
                wl_append_word(&wl, &wl, tprintf("Warning: xrcrg1 should be positive for rgateMod = 3.\n"));
    }

    /* Check capacitance parameters */
    if (pParam->BSIM4v6noff < 0.1)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Noff = %g is too small.\n", pParam->BSIM4v6noff));
    }

    if (pParam->BSIM4v6voffcv < -0.5)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Voffcv = %g is too small.\n", pParam->BSIM4v6voffcv));
    }

    if (pParam->BSIM4v6moin < 5.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Moin = %g is too small.\n", pParam->BSIM4v6moin));
    }
    if (pParam->BSIM4v6moin > 25.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Moin = %g is too large.\n", pParam->BSIM4v6moin));
    }
    if (model->BSIM4v6capMod == 2) {
        if (pParam->BSIM4v6acde < 0.1)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Acde = %g is too small.\n", pParam->BSIM4v6acde));
        }
        if (pParam->BSIM4v6acde > 1.6)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: Acde = %g is too large.\n", pParam->BSIM4v6acde));
        }
    }

    /* Check overlap capacitance parameters */
    if (model->BSIM4v6cgdo < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM4v6cgdo));
        model->BSIM4v6cgdo = 0.0;
    }
    if (model->BSIM4v6cgso < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: cgso = %g is negative. Set to zero.\n", model->BSIM4v6cgso));
        model->BSIM4v6cgso = 0.0;
    }
    if (model->BSIM4v6cgbo < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: cgbo = %g is negative. Set to zero.\n", model->BSIM4v6cgbo));
        model->BSIM4v6cgbo = 0.0;
    }
    if (model->BSIM4v6tnoiMod == 1) {
        if (model->BSIM4v6tnoia < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: tnoia = %g is negative. Set to zero.\n", model->BSIM4v6tnoia));
            model->BSIM4v6tnoia = 0.0;
        }
        if (model->BSIM4v6tnoib < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: tnoib = %g is negative. Set to zero.\n", model->BSIM4v6tnoib));
            model->BSIM4v6tnoib = 0.0;
        }

        if (model->BSIM4v6rnoia < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: rnoia = %g is negative. Set to zero.\n", model->BSIM4v6rnoia));
            model->BSIM4v6rnoia = 0.0;
        }
        if (model->BSIM4v6rnoib < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: rnoib = %g is negative. Set to zero.\n", model->BSIM4v6rnoib));
            model->BSIM4v6rnoib = 0.0;
        }
    }

    if (model->BSIM4v6SjctEmissionCoeff < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Njs = %g is negative.\n",
            model->BSIM4v6SjctEmissionCoeff));
    }
    if (model->BSIM4v6DjctEmissionCoeff < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Njd = %g is negative.\n",
            model->BSIM4v6DjctEmissionCoeff));
    }
    if (model->BSIM4v6njts < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Njts = %g is negative at temperature = %g.\n",
            model->BSIM4v6njts, ckt->CKTtemp));
    }
    if (model->BSIM4v6njtssw < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Njtssw = %g is negative at temperature = %g.\n",
            model->BSIM4v6njtssw, ckt->CKTtemp));
    }
    if (model->BSIM4v6njtsswg < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Njtsswg = %g is negative at temperature = %g.\n",
            model->BSIM4v6njtsswg, ckt->CKTtemp));
    }
    if (model->BSIM4v6njtsdGiven && model->BSIM4v6njtsdtemp < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Njtsd = %g is negative at temperature = %g.\n",
            model->BSIM4v6njtsdtemp, ckt->CKTtemp));
    }
    if (model->BSIM4v6njtsswdGiven && model->BSIM4v6njtsswdtemp < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Njtsswd = %g is negative at temperature = %g.\n",
            model->BSIM4v6njtsswdtemp, ckt->CKTtemp));
    }
    if (model->BSIM4v6njtsswgdGiven && model->BSIM4v6njtsswgdtemp < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: Njtsswgd = %g is negative at temperature = %g.\n",
            model->BSIM4v6njtsswgdtemp, ckt->CKTtemp));
    }

    if (model->BSIM4v6ntnoi < 0.0)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: ntnoi = %g is negative. Set to zero.\n", model->BSIM4v6ntnoi));
        model->BSIM4v6ntnoi = 0.0;
    }

    /* diode model */
    if (model->BSIM4v6SbulkJctBotGradingCoeff >= 0.99)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: MJS = %g is too big. Set to 0.99.\n", model->BSIM4v6SbulkJctBotGradingCoeff));
        model->BSIM4v6SbulkJctBotGradingCoeff = 0.99;
    }
    if (model->BSIM4v6SbulkJctSideGradingCoeff >= 0.99)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: MJSWS = %g is too big. Set to 0.99.\n", model->BSIM4v6SbulkJctSideGradingCoeff));
        model->BSIM4v6SbulkJctSideGradingCoeff = 0.99;
    }
    if (model->BSIM4v6SbulkJctGateSideGradingCoeff >= 0.99)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: MJSWGS = %g is too big. Set to 0.99.\n", model->BSIM4v6SbulkJctGateSideGradingCoeff));
        model->BSIM4v6SbulkJctGateSideGradingCoeff = 0.99;
    }

    if (model->BSIM4v6DbulkJctBotGradingCoeff >= 0.99)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: MJD = %g is too big. Set to 0.99.\n", model->BSIM4v6DbulkJctBotGradingCoeff));
        model->BSIM4v6DbulkJctBotGradingCoeff = 0.99;
    }
    if (model->BSIM4v6DbulkJctSideGradingCoeff >= 0.99)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: MJSWD = %g is too big. Set to 0.99.\n", model->BSIM4v6DbulkJctSideGradingCoeff));
        model->BSIM4v6DbulkJctSideGradingCoeff = 0.99;
    }
    if (model->BSIM4v6DbulkJctGateSideGradingCoeff >= 0.99)
    {
        wl_append_word(&wl, &wl, tprintf("Warning: MJSWGD = %g is too big. Set to 0.99.\n", model->BSIM4v6DbulkJctGateSideGradingCoeff));
        model->BSIM4v6DbulkJctGateSideGradingCoeff = 0.99;
    }
    if (model->BSIM4v6wpemod == 1)
    {
        if (model->BSIM4v6scref <= 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: SCREF = %g is not positive. Set to 1e-6.\n", model->BSIM4v6scref));
            model->BSIM4v6scref = 1e-6;
        }
        if (here->BSIM4v6sca < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: SCA = %g is negative. Set to 0.0.\n", here->BSIM4v6sca));
            here->BSIM4v6sca = 0.0;
        }
        if (here->BSIM4v6scb < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: SCB = %g is negative. Set to 0.0.\n", here->BSIM4v6scb));
            here->BSIM4v6scb = 0.0;
        }
        if (here->BSIM4v6scc < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: SCC = %g is negative. Set to 0.0.\n", here->BSIM4v6scc));
            here->BSIM4v6scc = 0.0;
        }
        if (here->BSIM4v6sc < 0.0)
        {
            wl_append_word(&wl, &wl, tprintf("Warning: SC = %g is negative. Set to 0.0.\n", here->BSIM4v6sc));
            here->BSIM4v6sc = 0.0;
        }
    }

    if (wlstart->wl_next) {
        if ((fplog = fopen("bsim4v6.out", "w")) != NULL) {
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

    wl_free(wlstart);

    if ((strcmp(model->BSIM4v6version, "4.6.5")) && (strncmp(model->BSIM4v6version, "4.60", 4)) && (strncmp(model->BSIM4v6version, "4.6", 3)))
    {
        printf("Warning: This model is BSIM4.6.5; you specified a wrong version number '%s'.\n", model->BSIM4v6version);
    }

    return(Fatal_Flag);
}

