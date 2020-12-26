/**** BSIM3v3.2.4, Released by Xuemei Xi 12/14/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b3check.c of BSIM3v3.2.4
 * Author: 1995 Min-Chie Jeng
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001 Xuemei Xi
 * Modified by Xuemei Xi, 10/05, 12/14, 2001.
 * Modified by Paolo Nenzi 2002 and Dietmar Warning 2003
 * Modified by Dietmar Warning, 12/21/2020.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v32def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"
#include "ngspice/wordlist.h"
#include "ngspice/cpextern.h"

/* Check for correctness of the BSIM3.2 parameters:
   If parameter excursions are found, put the warning or error message into a wordlist.
   Only then open a file b3v32check.log and print the data into the file. */

int
BSIM3v32checkModel (BSIM3v32model *model, BSIM3v32instance *here, CKTcircuit *ckt)
{
    struct bsim3v32SizeDependParam *pParam;
    int Fatal_Flag = 0;
    FILE *fplog;
    wordlist* wl, *wlstart;

    NG_IGNORE(ckt);

    pParam = here->pParam;

    if (cp_getvar("ng_nomodcheck", CP_BOOL, NULL, 0))
        return(0);

    wl = wlstart = TMALLOC(wordlist, 1);
    wl->wl_prev = NULL;
    wl->wl_next = NULL;
    wl->wl_word = tprintf("\nChecking parameters for BSIM 3.2 model %s\n", model->BSIM3v32modName);

    if ((strcmp(model->BSIM3v32version, "3.2.4")) && (strncmp(model->BSIM3v32version, "3.24", 4))
     && (strcmp(model->BSIM3v32version, "3.2.3")) && (strncmp(model->BSIM3v32version, "3.23", 4))
     && (strcmp(model->BSIM3v32version, "3.2.2")) && (strncmp(model->BSIM3v32version, "3.22", 4))
     && (strncmp(model->BSIM3v32version, "3.2", 3)) && (strncmp(model->BSIM3v32version, "3.20", 4)))
    {
        printf("Warning: This model supports BSIM3v3.2, BSIM3v3.2.2, BSIM3v3.2.3, BSIM3v3.2.4\n");
        printf("You specified a wrong version number. Working now with BSIM3v3.2.4.\n");
        wl_append_word(&wl, &wl, tprintf("Warning: This model supports BSIM3v3.2, BSIM3v3.2.2, BSIM3v3.2.3, BSIM3v3.2.4\n"));
        wl_append_word(&wl, &wl, tprintf("You specified a wrong version number. Working now with BSIM3v3.2.4.\n"));
    }

    if (pParam->BSIM3v32nlx < -pParam->BSIM3v32leff)
    {   wl_append_word(&wl, &wl, tprintf("Fatal: Nlx = %g is less than -Leff.\n",
                    pParam->BSIM3v32nlx));
        Fatal_Flag = 1;
    }

    if (model->BSIM3v32tox <= 0.0)
    {   wl_append_word(&wl, &wl, tprintf("Fatal: Tox = %g is not positive.\n",
                model->BSIM3v32tox));
        Fatal_Flag = 1;
    }

    if (model->BSIM3v32toxm <= 0.0)
    {   wl_append_word(&wl, &wl, tprintf("Fatal: Toxm = %g is not positive.\n",
                model->BSIM3v32toxm));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM3v32npeak <= 0.0)
    {   wl_append_word(&wl, &wl, tprintf("Fatal: Nch = %g is not positive.\n",
                pParam->BSIM3v32npeak));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM3v32nsub <= 0.0)
    {   wl_append_word(&wl, &wl, tprintf("Fatal: Nsub = %g is not positive.\n",
                pParam->BSIM3v32nsub));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM3v32ngate < 0.0)
    {   wl_append_word(&wl, &wl, tprintf("Fatal: Ngate = %g is not positive.\n",
                pParam->BSIM3v32ngate));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM3v32ngate > 1.e25)
    {   wl_append_word(&wl, &wl, tprintf("Fatal: Ngate = %g is too high.\n",
                pParam->BSIM3v32ngate));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM3v32xj <= 0.0)
    {   wl_append_word(&wl, &wl, tprintf("Fatal: Xj = %g is not positive.\n",
                pParam->BSIM3v32xj));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM3v32dvt1 < 0.0)
    {   wl_append_word(&wl, &wl, tprintf("Fatal: Dvt1 = %g is negative.\n",
                pParam->BSIM3v32dvt1));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM3v32dvt1w < 0.0)
    {   wl_append_word(&wl, &wl, tprintf("Fatal: Dvt1w = %g is negative.\n",
                pParam->BSIM3v32dvt1w));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM3v32w0 == -pParam->BSIM3v32weff)
    {   wl_append_word(&wl, &wl, tprintf("Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n"));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM3v32dsub < 0.0)
    {   wl_append_word(&wl, &wl, tprintf("Fatal: Dsub = %g is negative.\n", pParam->BSIM3v32dsub));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM3v32b1 == -pParam->BSIM3v32weff)
    {   wl_append_word(&wl, &wl, tprintf("Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n"));
        Fatal_Flag = 1;
    }
    if (pParam->BSIM3v32u0temp <= 0.0)
    {   wl_append_word(&wl, &wl, tprintf("Fatal: u0 at current temperature = %g is not positive.\n", pParam->BSIM3v32u0temp));
        Fatal_Flag = 1;
    }

/* Check delta parameter */
    if (pParam->BSIM3v32delta < 0.0)
    {   wl_append_word(&wl, &wl, tprintf("Fatal: Delta = %g is less than zero.\n",
                pParam->BSIM3v32delta));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM3v32vsattemp <= 0.0)
    {   wl_append_word(&wl, &wl, tprintf("Fatal: Vsat at current temperature = %g is not positive.\n", pParam->BSIM3v32vsattemp));
        Fatal_Flag = 1;
    }
/* Check Rout parameters */
    if (pParam->BSIM3v32pclm <= 0.0)
    {   wl_append_word(&wl, &wl, tprintf("Fatal: Pclm = %g is not positive.\n", pParam->BSIM3v32pclm));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM3v32drout < 0.0)
    {   wl_append_word(&wl, &wl, tprintf("Fatal: Drout = %g is negative.\n", pParam->BSIM3v32drout));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM3v32pscbe2 <= 0.0)
    {   wl_append_word(&wl, &wl, tprintf("Warning: Pscbe2 = %g is not positive.\n",
                pParam->BSIM3v32pscbe2));
    }

    /* ACM model */
    if (model->BSIM3v32acmMod == 0) {
        if (model->BSIM3v32unitLengthSidewallJctCap > 0.0 ||
              model->BSIM3v32unitLengthGateSidewallJctCap > 0.0)
        {
            if (here->BSIM3v32drainPerimeter < pParam->BSIM3v32weff)
            {   wl_append_word(&wl, &wl, tprintf("Warning: Pd = %g is less than W.\n",
                     here->BSIM3v32drainPerimeter));
            }
            if (here->BSIM3v32sourcePerimeter < pParam->BSIM3v32weff)
            {   wl_append_word(&wl, &wl, tprintf("Warning: Ps = %g is less than W.\n",
                     here->BSIM3v32sourcePerimeter));
            }
        }
    }

    if ((model->BSIM3v32calcacm > 0) && (model->BSIM3v32acmMod != 12))
    {   wl_append_word(&wl, &wl, tprintf("Warning: CALCACM = %d is wrong. Set back to 0.\n",
            model->BSIM3v32calcacm));
        model->BSIM3v32calcacm = 0;
    }

    if (pParam->BSIM3v32noff < 0.1)
    {   wl_append_word(&wl, &wl, tprintf("Warning: Noff = %g is too small.\n",
                pParam->BSIM3v32noff));
    }
    if (pParam->BSIM3v32noff > 4.0)
    {   wl_append_word(&wl, &wl, tprintf("Warning: Noff = %g is too large.\n",
                pParam->BSIM3v32noff));
    }

    if (pParam->BSIM3v32voffcv < -0.5)
    {   wl_append_word(&wl, &wl, tprintf("Warning: Voffcv = %g is too small.\n",
                pParam->BSIM3v32voffcv));
    }
    if (pParam->BSIM3v32voffcv > 0.5)
    {   wl_append_word(&wl, &wl, tprintf("Warning: Voffcv = %g is too large.\n",
                pParam->BSIM3v32voffcv));
    }

    if (model->BSIM3v32ijth < 0.0)
    {   wl_append_word(&wl, &wl, tprintf("Fatal: Ijth = %g cannot be negative.\n",
                model->BSIM3v32ijth));
        Fatal_Flag = 1;
    }

/* Check capacitance parameters */
    if (pParam->BSIM3v32clc < 0.0)
    {   wl_append_word(&wl, &wl, tprintf("Fatal: Clc = %g is negative.\n", pParam->BSIM3v32clc));
        Fatal_Flag = 1;
    }

    if (pParam->BSIM3v32moin < 5.0)
    {   wl_append_word(&wl, &wl, tprintf("Warning: Moin = %g is too small.\n",
                pParam->BSIM3v32moin));
    }
    if (pParam->BSIM3v32moin > 25.0)
    {   wl_append_word(&wl, &wl, tprintf("Warning: Moin = %g is too large.\n",
                pParam->BSIM3v32moin));
    }

    if(model->BSIM3v32capMod ==3) {
        if (pParam->BSIM3v32acde < 0.4)
        {   wl_append_word(&wl, &wl, tprintf("Warning:  Acde = %g is too small.\n",
                    pParam->BSIM3v32acde));
        }
        if (pParam->BSIM3v32acde > 1.6)
        {   wl_append_word(&wl, &wl, tprintf("Warning:  Acde = %g is too large.\n",
                    pParam->BSIM3v32acde));
        }
    }

    if (model->BSIM3v32paramChk ==1)
    {
    /* Check L and W parameters */
        if (pParam->BSIM3v32leff <= 5.0e-8)
        {   wl_append_word(&wl, &wl, tprintf("Warning: Leff = %g may be too small.\n",
                    pParam->BSIM3v32leff));
        }

        if (pParam->BSIM3v32leffCV <= 5.0e-8)
        {   wl_append_word(&wl, &wl, tprintf("Warning: Leff for CV = %g may be too small.\n",
                    pParam->BSIM3v32leffCV));
        }

        if (pParam->BSIM3v32weff <= 1.0e-7)
        {   wl_append_word(&wl, &wl, tprintf("Warning: Weff = %g may be too small.\n",
                    pParam->BSIM3v32weff));
        }

        if (pParam->BSIM3v32weffCV <= 1.0e-7)
        {   wl_append_word(&wl, &wl, tprintf("Warning: Weff for CV = %g may be too small.\n",
                    pParam->BSIM3v32weffCV));
        }

    /* Check threshold voltage parameters */
        if (pParam->BSIM3v32nlx < 0.0)
        {   wl_append_word(&wl, &wl, tprintf("Warning: Nlx = %g is negative.\n", pParam->BSIM3v32nlx));
        }
         if (model->BSIM3v32tox < 1.0e-9)
        {   wl_append_word(&wl, &wl, tprintf("Warning: Tox = %g is less than 10A.\n",
                    model->BSIM3v32tox));
        }

        if (pParam->BSIM3v32npeak <= 1.0e15)
        {   wl_append_word(&wl, &wl, tprintf("Warning: Nch = %g may be too small.\n",
                    pParam->BSIM3v32npeak));
        }
        else if (pParam->BSIM3v32npeak >= 1.0e21)
        {   wl_append_word(&wl, &wl, tprintf("Warning: Nch = %g may be too large.\n",
                    pParam->BSIM3v32npeak));
        }

        if (pParam->BSIM3v32nsub <= 1.0e14)
        {   wl_append_word(&wl, &wl, tprintf("Warning: Nsub = %g may be too small.\n",
                    pParam->BSIM3v32nsub));
        }
        else if (pParam->BSIM3v32nsub >= 1.0e21)
        {   wl_append_word(&wl, &wl, tprintf("Warning: Nsub = %g may be too large.\n",
                    pParam->BSIM3v32nsub));
        }

        if ((pParam->BSIM3v32ngate > 0.0) &&
            (pParam->BSIM3v32ngate <= 1.e18))
        {   wl_append_word(&wl, &wl, tprintf("Warning: Ngate = %g is less than 1.E18cm^-3.\n",
                    pParam->BSIM3v32ngate));
        }

        if (pParam->BSIM3v32dvt0 < 0.0)
        {   wl_append_word(&wl, &wl, tprintf("Warning: Dvt0 = %g is negative.\n",
                    pParam->BSIM3v32dvt0));
        }

        if (fabs(1.0e-6 / (pParam->BSIM3v32w0 + pParam->BSIM3v32weff)) > 10.0)
        {   wl_append_word(&wl, &wl, tprintf("Warning: (W0 + Weff) may be too small.\n"));
        }

    /* Check subthreshold parameters */
        if (pParam->BSIM3v32nfactor < 0.0)
        {   wl_append_word(&wl, &wl, tprintf("Warning: Nfactor = %g is negative.\n",
                    pParam->BSIM3v32nfactor));
        }
        if (pParam->BSIM3v32cdsc < 0.0)
        {   wl_append_word(&wl, &wl, tprintf("Warning: Cdsc = %g is negative.\n",
                    pParam->BSIM3v32cdsc));
        }
        if (pParam->BSIM3v32cdscd < 0.0)
        {   wl_append_word(&wl, &wl, tprintf("Warning: Cdscd = %g is negative.\n",
                    pParam->BSIM3v32cdscd));
        }
    /* Check DIBL parameters */
        if (pParam->BSIM3v32eta0 < 0.0)
        {   wl_append_word(&wl, &wl, tprintf("Warning: Eta0 = %g is negative.\n",
                    pParam->BSIM3v32eta0));
        }

    /* Check Abulk parameters */
        if (fabs(1.0e-6 / (pParam->BSIM3v32b1 + pParam->BSIM3v32weff)) > 10.0)
               {   wl_append_word(&wl, &wl, tprintf("Warning: (B1 + Weff) may be too small.\n"));
        }

    /* Check Saturation parameters */
        if (pParam->BSIM3v32a2 < 0.01)
        {   wl_append_word(&wl, &wl, tprintf("Warning: A2 = %g is too small. Set to 0.01.\n", pParam->BSIM3v32a2));
            pParam->BSIM3v32a2 = 0.01;
        }
        else if (pParam->BSIM3v32a2 > 1.0)
        {   wl_append_word(&wl, &wl, tprintf("Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
                    pParam->BSIM3v32a2));
            pParam->BSIM3v32a2 = 1.0;
            pParam->BSIM3v32a1 = 0.0;

        }

        if (pParam->BSIM3v32rdsw < 0.0)
        {   wl_append_word(&wl, &wl, tprintf("Warning: Rdsw = %g is negative. Set to zero.\n",
                    pParam->BSIM3v32rdsw));
            pParam->BSIM3v32rdsw = 0.0;
            pParam->BSIM3v32rds0 = 0.0;
        }
        else if ((pParam->BSIM3v32rds0 > 0.0) && (pParam->BSIM3v32rds0 < 0.001))
        {   wl_append_word(&wl, &wl, tprintf("Warning: Rds at current temperature = %g is less than 0.001 ohm. Set to zero.\n",
                    pParam->BSIM3v32rds0));
            pParam->BSIM3v32rds0 = 0.0;
        }
        if (pParam->BSIM3v32vsattemp < 1.0e3)
        {   wl_append_word(&wl, &wl, tprintf("Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM3v32vsattemp));
        }

        if (pParam->BSIM3v32pdibl1 < 0.0)
        {   wl_append_word(&wl, &wl, tprintf("Warning: Pdibl1 = %g is negative.\n",
                    pParam->BSIM3v32pdibl1));
        }
        if (pParam->BSIM3v32pdibl2 < 0.0)
        {   wl_append_word(&wl, &wl, tprintf("Warning: Pdibl2 = %g is negative.\n",
                    pParam->BSIM3v32pdibl2));
        }
    /* Check overlap capacitance parameters */
        if (model->BSIM3v32cgdo < 0.0)
        {   wl_append_word(&wl, &wl, tprintf("Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM3v32cgdo));
            model->BSIM3v32cgdo = 0.0;
        }
        if (model->BSIM3v32cgso < 0.0)
        {   wl_append_word(&wl, &wl, tprintf("Warning: cgso = %g is negative. Set to zero.\n", model->BSIM3v32cgso));
            model->BSIM3v32cgso = 0.0;
        }
        if (model->BSIM3v32cgbo < 0.0)
        {   wl_append_word(&wl, &wl, tprintf("Warning: cgbo = %g is negative. Set to zero.\n", model->BSIM3v32cgbo));
            model->BSIM3v32cgbo = 0.0;
        }

    }

    if (wlstart->wl_next) {
        if ((fplog = fopen("b3v32check.log", "w")) != NULL) {
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

    return(Fatal_Flag);
}

