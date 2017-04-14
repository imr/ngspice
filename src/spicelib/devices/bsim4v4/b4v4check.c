/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4check.c of BSIM4.4.0.
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
#include "ngspice/cktdefs.h"
#include "bsim4v4def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"

int
BSIM4v4checkModel(model, here, ckt)
register BSIM4v4model *model;
register BSIM4v4instance *here;
CKTcircuit *ckt;
{
struct bsim4v4SizeDependParam *pParam;
int Fatal_Flag = 0;
FILE *fplog;

    if ((fplog = fopen("bsim4v4.out", "w")) != NULL)
    {   pParam = here->pParam;
        fprintf(fplog, "BSIM4v4: Berkeley Short Channel IGFET Model-4\n");
        fprintf(fplog, "Developed by Xuemei (Jane) Xi, Jin He, Mohan Dunga, Prof. Ali Niknejad and Prof. Chenming Hu in 2003.\n");
        fprintf(fplog, "\n");
	fprintf(fplog, "++++++++++ BSIM4v4 PARAMETER CHECKING BELOW ++++++++++\n");

        if (strcmp(model->BSIM4v4version, "4.4.0") != 0)
        {  fprintf(fplog, "Warning: This model is BSIM4.4.0; you specified a wrong version number.\n");
           printf("Warning: This model is BSIM4.4.0; you specified a wrong version number.\n");
        }
	fprintf(fplog, "Model = %s\n", model->BSIM4v4modName);


        if ((here->BSIM4v4rgateMod == 2) || (here->BSIM4v4rgateMod == 3))
        {   if ((here->BSIM4v4trnqsMod == 1) || (here->BSIM4v4acnqsMod == 1))
            {   fprintf(fplog, "Warning: You've selected both Rg and charge deficit NQS; select one only.\n");
                printf("Warning: You've selected both Rg and charge deficit NQS; select one only.\n");
            }
        }

	if (model->BSIM4v4toxe <= 0.0)
	{   fprintf(fplog, "Fatal: Toxe = %g is not positive.\n",
		    model->BSIM4v4toxe);
	    printf("Fatal: Toxe = %g is not positive.\n", model->BSIM4v4toxe);
	    Fatal_Flag = 1;
	}
        if (model->BSIM4v4toxp <= 0.0)
        {   fprintf(fplog, "Fatal: Toxp = %g is not positive.\n",
                    model->BSIM4v4toxp);
            printf("Fatal: Toxp = %g is not positive.\n", model->BSIM4v4toxp);
            Fatal_Flag = 1;
        }

        if (model->BSIM4v4toxm <= 0.0)
        {   fprintf(fplog, "Fatal: Toxm = %g is not positive.\n",
                    model->BSIM4v4toxm);
            printf("Fatal: Toxm = %g is not positive.\n", model->BSIM4v4toxm);
            Fatal_Flag = 1;
        }

        if (model->BSIM4v4toxref <= 0.0)
        {   fprintf(fplog, "Fatal: Toxref = %g is not positive.\n",
                    model->BSIM4v4toxref);
            printf("Fatal: Toxref = %g is not positive.\n", model->BSIM4v4toxref);
            Fatal_Flag = 1;
        }

        if (pParam->BSIM4v4lpe0 < -pParam->BSIM4v4leff)
        {   fprintf(fplog, "Fatal: Lpe0 = %g is less than -Leff.\n",
                    pParam->BSIM4v4lpe0);
            printf("Fatal: Lpe0 = %g is less than -Leff.\n",
                        pParam->BSIM4v4lpe0);
            Fatal_Flag = 1;
        }
        if (model->BSIM4v4lintnoi > pParam->BSIM4v4leff/2)
        {   fprintf(fplog, "Fatal: Lintnoi = %g is too large - Leff for noise is negative.\n",
                    model->BSIM4v4lintnoi);
            printf("Fatal: Lintnoi = %g is too large - Leff for noise is negative.\n",
                    model->BSIM4v4lintnoi);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v4lpeb < -pParam->BSIM4v4leff)
        {   fprintf(fplog, "Fatal: Lpeb = %g is less than -Leff.\n",
                    pParam->BSIM4v4lpeb);
            printf("Fatal: Lpeb = %g is less than -Leff.\n",
                        pParam->BSIM4v4lpeb);
            Fatal_Flag = 1;
        }
	if (pParam->BSIM4v4ndep <= 0.0)
	{   fprintf(fplog, "Fatal: Ndep = %g is not positive.\n",
		    pParam->BSIM4v4ndep);
	    printf("Fatal: Ndep = %g is not positive.\n",
		   pParam->BSIM4v4ndep);
	    Fatal_Flag = 1;
	}
        if (pParam->BSIM4v4phi <= 0.0)
        {   fprintf(fplog, "Fatal: Phi = %g is not positive. Please check Phin and Ndep\n",
                    pParam->BSIM4v4phi);
            fprintf(fplog, "	   Phin = %g  Ndep = %g \n",
            	    pParam->BSIM4v4phin, pParam->BSIM4v4ndep);
            printf("Fatal: Phi = %g is not positive. Please check Phin and Ndep\n",
                    pParam->BSIM4v4phi);
            printf("	   Phin = %g  Ndep = %g \n",
            	    pParam->BSIM4v4phin, pParam->BSIM4v4ndep);
            Fatal_Flag = 1;
        }
	if (pParam->BSIM4v4nsub <= 0.0)
	{   fprintf(fplog, "Fatal: Nsub = %g is not positive.\n",
		    pParam->BSIM4v4nsub);
	    printf("Fatal: Nsub = %g is not positive.\n",
		   pParam->BSIM4v4nsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v4ngate < 0.0)
	{   fprintf(fplog, "Fatal: Ngate = %g is not positive.\n",
		    pParam->BSIM4v4ngate);
	    printf("Fatal: Ngate = %g Ngate is not positive.\n",
		   pParam->BSIM4v4ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v4ngate > 1.e25)
	{   fprintf(fplog, "Fatal: Ngate = %g is too high.\n",
		    pParam->BSIM4v4ngate);
	    printf("Fatal: Ngate = %g Ngate is too high\n",
		   pParam->BSIM4v4ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v4xj <= 0.0)
	{   fprintf(fplog, "Fatal: Xj = %g is not positive.\n",
		    pParam->BSIM4v4xj);
	    printf("Fatal: Xj = %g is not positive.\n", pParam->BSIM4v4xj);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4v4dvt1 < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1 = %g is negative.\n",
		    pParam->BSIM4v4dvt1);
	    printf("Fatal: Dvt1 = %g is negative.\n", pParam->BSIM4v4dvt1);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4v4dvt1w < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1w = %g is negative.\n",
		    pParam->BSIM4v4dvt1w);
	    printf("Fatal: Dvt1w = %g is negative.\n", pParam->BSIM4v4dvt1w);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4v4w0 == -pParam->BSIM4v4weff)
	{   fprintf(fplog, "Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }

	if (pParam->BSIM4v4dsub < 0.0)
	{   fprintf(fplog, "Fatal: Dsub = %g is negative.\n", pParam->BSIM4v4dsub);
	    printf("Fatal: Dsub = %g is negative.\n", pParam->BSIM4v4dsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v4b1 == -pParam->BSIM4v4weff)
	{   fprintf(fplog, "Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }
        if (here->BSIM4v4u0temp <= 0.0)
	{   fprintf(fplog, "Fatal: u0 at current temperature = %g is not positive.\n", here->BSIM4v4u0temp);
	    printf("Fatal: u0 at current temperature = %g is not positive.\n",
		   here->BSIM4v4u0temp);
	    Fatal_Flag = 1;
        }

        if (pParam->BSIM4v4delta < 0.0)
	{   fprintf(fplog, "Fatal: Delta = %g is less than zero.\n",
		    pParam->BSIM4v4delta);
	    printf("Fatal: Delta = %g is less than zero.\n", pParam->BSIM4v4delta);
	    Fatal_Flag = 1;
        }

	if (here->BSIM4v4vsattemp <= 0.0)
	{   fprintf(fplog, "Fatal: Vsat at current temperature = %g is not positive.\n", here->BSIM4v4vsattemp);
	    printf("Fatal: Vsat at current temperature = %g is not positive.\n",
		   here->BSIM4v4vsattemp);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4v4pclm <= 0.0)
	{   fprintf(fplog, "Fatal: Pclm = %g is not positive.\n", pParam->BSIM4v4pclm);
	    printf("Fatal: Pclm = %g is not positive.\n", pParam->BSIM4v4pclm);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4v4drout < 0.0)
	{   fprintf(fplog, "Fatal: Drout = %g is negative.\n", pParam->BSIM4v4drout);
	    printf("Fatal: Drout = %g is negative.\n", pParam->BSIM4v4drout);
	    Fatal_Flag = 1;
	}

        if (here->BSIM4v4nf < 1.0)
        {   fprintf(fplog, "Fatal: Number of finger = %g is smaller than one.\n", here->BSIM4v4nf);
            printf("Fatal: Number of finger = %g is smaller than one.\n", here->BSIM4v4nf);
            Fatal_Flag = 1;
        }

        if((here->BSIM4v4sa > 0.0) && (here->BSIM4v4sb > 0.0) &&
       	((here->BSIM4v4nf == 1.0) || ((here->BSIM4v4nf > 1.0) && (here->BSIM4v4sd > 0.0))) )
        {   if (model->BSIM4v4saref <= 0.0)
            {   fprintf(fplog, "Fatal: SAref = %g is not positive.\n",model->BSIM4v4saref);
             	printf("Fatal: SAref = %g is not positive.\n",model->BSIM4v4saref);
             	Fatal_Flag = 1;
            }
            if (model->BSIM4v4sbref <= 0.0)
            {   fprintf(fplog, "Fatal: SBref = %g is not positive.\n",model->BSIM4v4sbref);
            	printf("Fatal: SBref = %g is not positive.\n",model->BSIM4v4sbref);
            	Fatal_Flag = 1;
            }
 	}

        if ((here->BSIM4v4l + model->BSIM4v4xl) <= model->BSIM4v4xgl)
        {   fprintf(fplog, "Fatal: The parameter xgl must be smaller than Ldrawn+XL.\n");
            printf("Fatal: The parameter xgl must be smaller than Ldrawn+XL.\n");
            Fatal_Flag = 1;
        }
        if (model->BSIM4v4ngcon < 1.0)
        {   fprintf(fplog, "Fatal: The parameter ngcon cannot be smaller than one.\n");
            printf("Fatal: The parameter ngcon cannot be smaller than one.\n");
            Fatal_Flag = 1;
        }
        if ((model->BSIM4v4ngcon != 1.0) && (model->BSIM4v4ngcon != 2.0))
        {   model->BSIM4v4ngcon = 1.0;
            fprintf(fplog, "Warning: Ngcon must be equal to one or two; reset to 1.0.\n");
            printf("Warning: Ngcon must be equal to one or two; reset to 1.0.\n");
        }

        if (model->BSIM4v4gbmin < 1.0e-20)
        {   fprintf(fplog, "Warning: Gbmin = %g is too small.\n",
                    model->BSIM4v4gbmin);
            printf("Warning: Gbmin = %g is too small.\n", model->BSIM4v4gbmin);
        }

        /* Check saturation parameters */
        if (pParam->BSIM4v4fprout < 0.0)
        {   fprintf(fplog, "Fatal: fprout = %g is negative.\n",
                    pParam->BSIM4v4fprout);
            printf("Fatal: fprout = %g is negative.\n", pParam->BSIM4v4fprout);
	    Fatal_Flag = 1;
        }
        if (pParam->BSIM4v4pdits < 0.0)
        {   fprintf(fplog, "Fatal: pdits = %g is negative.\n",
                    pParam->BSIM4v4pdits);
            printf("Fatal: pdits = %g is negative.\n", pParam->BSIM4v4pdits);
            Fatal_Flag = 1;
        }
        if (model->BSIM4v4pditsl < 0.0)
        {   fprintf(fplog, "Fatal: pditsl = %g is negative.\n",
                    model->BSIM4v4pditsl);
            printf("Fatal: pditsl = %g is negative.\n", model->BSIM4v4pditsl);
            Fatal_Flag = 1;
        }

        /* Check gate current parameters */
      if (model->BSIM4v4igbMod) {
        if (pParam->BSIM4v4nigbinv <= 0.0)
        {   fprintf(fplog, "Fatal: nigbinv = %g is non-positive.\n",
                    pParam->BSIM4v4nigbinv);
            printf("Fatal: nigbinv = %g is non-positive.\n", pParam->BSIM4v4nigbinv);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v4nigbacc <= 0.0)
        {   fprintf(fplog, "Fatal: nigbacc = %g is non-positive.\n",
                    pParam->BSIM4v4nigbacc);
            printf("Fatal: nigbacc = %g is non-positive.\n", pParam->BSIM4v4nigbacc);
            Fatal_Flag = 1;
        }
      }
      if (model->BSIM4v4igcMod) {
        if (pParam->BSIM4v4nigc <= 0.0)
        {   fprintf(fplog, "Fatal: nigc = %g is non-positive.\n",
                    pParam->BSIM4v4nigc);
            printf("Fatal: nigc = %g is non-positive.\n", pParam->BSIM4v4nigc);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v4poxedge <= 0.0)
        {   fprintf(fplog, "Fatal: poxedge = %g is non-positive.\n",
                    pParam->BSIM4v4poxedge);
            printf("Fatal: poxedge = %g is non-positive.\n", pParam->BSIM4v4poxedge);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v4pigcd <= 0.0)
        {   fprintf(fplog, "Fatal: pigcd = %g is non-positive.\n",
                    pParam->BSIM4v4pigcd);
            printf("Fatal: pigcd = %g is non-positive.\n", pParam->BSIM4v4pigcd);
            Fatal_Flag = 1;
        }
      }

        /* Check capacitance parameters */
        if (pParam->BSIM4v4clc < 0.0)
	{   fprintf(fplog, "Fatal: Clc = %g is negative.\n", pParam->BSIM4v4clc);
	    printf("Fatal: Clc = %g is negative.\n", pParam->BSIM4v4clc);
	    Fatal_Flag = 1;
        }

        /* Check overlap capacitance parameters */
        if (pParam->BSIM4v4ckappas < 0.02)
        {   fprintf(fplog, "Warning: ckappas = %g is too small. Set to 0.02\n",
                    pParam->BSIM4v4ckappas);
            printf("Warning: ckappas = %g is too small.\n", pParam->BSIM4v4ckappas);
            pParam->BSIM4v4ckappas = 0.02;
       }
        if (pParam->BSIM4v4ckappad < 0.02)
        {   fprintf(fplog, "Warning: ckappad = %g is too small. Set to 0.02\n",
                    pParam->BSIM4v4ckappad);
            printf("Warning: ckappad = %g is too small.\n", pParam->BSIM4v4ckappad);
            pParam->BSIM4v4ckappad = 0.02;
        }

      if (model->BSIM4v4paramChk ==1)
      {
/* Check L and W parameters */
	if (pParam->BSIM4v4leff <= 1.0e-9)
	{   fprintf(fplog, "Warning: Leff = %g <= 1.0e-9. Recommended Leff >= 1e-8 \n",
		    pParam->BSIM4v4leff);
	    printf("Warning: Leff = %g <= 1.0e-9. Recommended Leff >= 1e-8 \n",
		    pParam->BSIM4v4leff);
	}

	if (pParam->BSIM4v4leffCV <= 1.0e-9)
	{   fprintf(fplog, "Warning: Leff for CV = %g <= 1.0e-9. Recommended LeffCV >=1e-8 \n",
		    pParam->BSIM4v4leffCV);
	    printf("Warning: Leff for CV = %g <= 1.0e-9. Recommended LeffCV >=1e-8 \n",
		    pParam->BSIM4v4leffCV);
	}

        if (pParam->BSIM4v4weff <= 1.0e-9)
	{   fprintf(fplog, "Warning: Weff = %g <= 1.0e-9. Recommended Weff >=1e-7 \n",
		    pParam->BSIM4v4weff);
	    printf("Warning: Weff = %g <= 1.0e-9. Recommended Weff >=1e-7 \n",
		   pParam->BSIM4v4weff);
	}

	if (pParam->BSIM4v4weffCV <= 1.0e-9)
	{   fprintf(fplog, "Warning: Weff for CV = %g <= 1.0e-9. Recommended WeffCV >= 1e-7 \n",
		    pParam->BSIM4v4weffCV);
	    printf("Warning: Weff for CV = %g <= 1.0e-9. Recommended WeffCV >= 1e-7 \n",
		    pParam->BSIM4v4weffCV);
	}

        /* Check threshold voltage parameters */
	if (model->BSIM4v4toxe < 1.0e-10)
	{   fprintf(fplog, "Warning: Toxe = %g is less than 1A. Recommended Toxe >= 5A\n",
	            model->BSIM4v4toxe);
	    printf("Warning: Toxe = %g is less than 1A. Recommended Toxe >= 5A\n", model->BSIM4v4toxe);
        }
        if (model->BSIM4v4toxp < 1.0e-10)
        {   fprintf(fplog, "Warning: Toxp = %g is less than 1A. Recommended Toxp >= 5A\n",
                    model->BSIM4v4toxp);
            printf("Warning: Toxp = %g is less than 1A. Recommended Toxp >= 5A\n", model->BSIM4v4toxp);
        }
        if (model->BSIM4v4toxm < 1.0e-10)
        {   fprintf(fplog, "Warning: Toxm = %g is less than 1A. Recommended Toxm >= 5A\n",
                    model->BSIM4v4toxm);
            printf("Warning: Toxm = %g is less than 1A. Recommended Toxm >= 5A\n", model->BSIM4v4toxm);
        }

        if (pParam->BSIM4v4ndep <= 1.0e12)
	{   fprintf(fplog, "Warning: Ndep = %g may be too small.\n",
	            pParam->BSIM4v4ndep);
	    printf("Warning: Ndep = %g may be too small.\n",
	           pParam->BSIM4v4ndep);
	}
	else if (pParam->BSIM4v4ndep >= 1.0e21)
	{   fprintf(fplog, "Warning: Ndep = %g may be too large.\n",
	            pParam->BSIM4v4ndep);
	    printf("Warning: Ndep = %g may be too large.\n",
	           pParam->BSIM4v4ndep);
	}

	 if (pParam->BSIM4v4nsub <= 1.0e14)
	{   fprintf(fplog, "Warning: Nsub = %g may be too small.\n",
	            pParam->BSIM4v4nsub);
	    printf("Warning: Nsub = %g may be too small.\n",
	           pParam->BSIM4v4nsub);
	}
	else if (pParam->BSIM4v4nsub >= 1.0e21)
	{   fprintf(fplog, "Warning: Nsub = %g may be too large.\n",
	            pParam->BSIM4v4nsub);
	    printf("Warning: Nsub = %g may be too large.\n",
	           pParam->BSIM4v4nsub);
	}

	if ((pParam->BSIM4v4ngate > 0.0) &&
	    (pParam->BSIM4v4ngate <= 1.e18))
	{   fprintf(fplog, "Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	            pParam->BSIM4v4ngate);
	    printf("Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	           pParam->BSIM4v4ngate);
	}

        if (pParam->BSIM4v4dvt0 < 0.0)
	{   fprintf(fplog, "Warning: Dvt0 = %g is negative.\n",
		    pParam->BSIM4v4dvt0);
	    printf("Warning: Dvt0 = %g is negative.\n", pParam->BSIM4v4dvt0);
	}

	if (fabs(1.0e-8 / (pParam->BSIM4v4w0 + pParam->BSIM4v4weff)) > 10.0)
	{   fprintf(fplog, "Warning: (W0 + Weff) may be too small.\n");
	    printf("Warning: (W0 + Weff) may be too small.\n");
        }

/* Check subthreshold parameters */
	if (pParam->BSIM4v4nfactor < 0.0)
	{   fprintf(fplog, "Warning: Nfactor = %g is negative.\n",
		    pParam->BSIM4v4nfactor);
	    printf("Warning: Nfactor = %g is negative.\n", pParam->BSIM4v4nfactor);
	}
	if (pParam->BSIM4v4cdsc < 0.0)
	{   fprintf(fplog, "Warning: Cdsc = %g is negative.\n",
		    pParam->BSIM4v4cdsc);
	    printf("Warning: Cdsc = %g is negative.\n", pParam->BSIM4v4cdsc);
	}
	if (pParam->BSIM4v4cdscd < 0.0)
	{   fprintf(fplog, "Warning: Cdscd = %g is negative.\n",
		    pParam->BSIM4v4cdscd);
	    printf("Warning: Cdscd = %g is negative.\n", pParam->BSIM4v4cdscd);
	}
/* Check DIBL parameters */
	if (here->BSIM4v4eta0 < 0.0)
	{   fprintf(fplog, "Warning: Eta0 = %g is negative.\n",
		    here->BSIM4v4eta0);
	    printf("Warning: Eta0 = %g is negative.\n", here->BSIM4v4eta0);
	}

/* Check Abulk parameters */
	 if (fabs(1.0e-8 / (pParam->BSIM4v4b1 + pParam->BSIM4v4weff)) > 10.0)
       	{   fprintf(fplog, "Warning: (B1 + Weff) may be too small.\n");
       	    printf("Warning: (B1 + Weff) may be too small.\n");
        }


/* Check Saturation parameters */
     	if (pParam->BSIM4v4a2 < 0.01)
	{   fprintf(fplog, "Warning: A2 = %g is too small. Set to 0.01.\n", pParam->BSIM4v4a2);
	    printf("Warning: A2 = %g is too small. Set to 0.01.\n",
		   pParam->BSIM4v4a2);
	    pParam->BSIM4v4a2 = 0.01;
	}
	else if (pParam->BSIM4v4a2 > 1.0)
	{   fprintf(fplog, "Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		    pParam->BSIM4v4a2);
	    printf("Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		   pParam->BSIM4v4a2);
	    pParam->BSIM4v4a2 = 1.0;
	    pParam->BSIM4v4a1 = 0.0;
	}

        if (pParam->BSIM4v4prwg < 0.0)
        {   fprintf(fplog, "Warning: Prwg = %g is negative. Set to zero.\n",
                    pParam->BSIM4v4prwg);
            printf("Warning: Prwg = %g is negative. Set to zero.\n",
                   pParam->BSIM4v4prwg);
            pParam->BSIM4v4prwg = 0.0;
        }

	if (pParam->BSIM4v4rdsw < 0.0)
	{   fprintf(fplog, "Warning: Rdsw = %g is negative. Set to zero.\n",
		    pParam->BSIM4v4rdsw);
	    printf("Warning: Rdsw = %g is negative. Set to zero.\n",
		   pParam->BSIM4v4rdsw);
	    pParam->BSIM4v4rdsw = 0.0;
	    pParam->BSIM4v4rds0 = 0.0;
	}

	if (pParam->BSIM4v4rds0 < 0.0)
	{   fprintf(fplog, "Warning: Rds at current temperature = %g is negative. Set to zero.\n",
		    pParam->BSIM4v4rds0);
	    printf("Warning: Rds at current temperature = %g is negative. Set to zero.\n",
		   pParam->BSIM4v4rds0);
	    pParam->BSIM4v4rds0 = 0.0;
	}

        if (pParam->BSIM4v4rdswmin < 0.0)
        {   fprintf(fplog, "Warning: Rdswmin at current temperature = %g is negative. Set to zero.\n",
                    pParam->BSIM4v4rdswmin);
            printf("Warning: Rdswmin at current temperature = %g is negative. Set to zero.\n",
                   pParam->BSIM4v4rdswmin);
            pParam->BSIM4v4rdswmin = 0.0;
        }

        if (pParam->BSIM4v4pscbe2 <= 0.0)
        {   fprintf(fplog, "Warning: Pscbe2 = %g is not positive.\n",
                    pParam->BSIM4v4pscbe2);
            printf("Warning: Pscbe2 = %g is not positive.\n", pParam->BSIM4v4pscbe2);
        }

        if (pParam->BSIM4v4vsattemp < 1.0e3)
	{   fprintf(fplog, "Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM4v4vsattemp);
	   printf("Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM4v4vsattemp);
	}

      if((model->BSIM4v4lambdaGiven) && (pParam->BSIM4v4lambda > 0.0) )
      {
        if (pParam->BSIM4v4lambda > 1.0e-9)
	{   fprintf(fplog, "Warning: Lambda = %g may be too large.\n", pParam->BSIM4v4lambda);
	   printf("Warning: Lambda = %g may be too large.\n", pParam->BSIM4v4lambda);
	}
      }

      if((model->BSIM4v4vtlGiven) && (pParam->BSIM4v4vtl > 0.0) )
      {
        if (pParam->BSIM4v4vtl < 6.0e4)
	{   fprintf(fplog, "Warning: Thermal velocity vtl = %g may be too small.\n", pParam->BSIM4v4vtl);
	   printf("Warning: Thermal velocity vtl = %g may be too small.\n", pParam->BSIM4v4vtl);
	}

        if (pParam->BSIM4v4xn < 3.0)
	{   fprintf(fplog, "Warning: back scattering coeff xn = %g is too small.\n", pParam->BSIM4v4xn);
	    printf("Warning: back scattering coeff xn = %g is too small. Reset to 3.0 \n", pParam->BSIM4v4xn);
	    pParam->BSIM4v4xn = 3.0;
	}

        if (model->BSIM4v4lc < 0.0)
	{   fprintf(fplog, "Warning: back scattering coeff lc = %g is too small.\n", model->BSIM4v4lc);
	    printf("Warning: back scattering coeff lc = %g is too small. Reset to 0.0\n", model->BSIM4v4lc);
	    pParam->BSIM4v4lc = 0.0;
	}
      }

	if (pParam->BSIM4v4pdibl1 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl1 = %g is negative.\n",
		    pParam->BSIM4v4pdibl1);
	    printf("Warning: Pdibl1 = %g is negative.\n", pParam->BSIM4v4pdibl1);
	}
	if (pParam->BSIM4v4pdibl2 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl2 = %g is negative.\n",
		    pParam->BSIM4v4pdibl2);
	    printf("Warning: Pdibl2 = %g is negative.\n", pParam->BSIM4v4pdibl2);
	}

/* Check stress effect parameters */
        if((here->BSIM4v4sa > 0.0) && (here->BSIM4v4sb > 0.0) &&
       	((here->BSIM4v4nf == 1.0) || ((here->BSIM4v4nf > 1.0) && (here->BSIM4v4sd > 0.0))) )
        {   if (model->BSIM4v4lodk2 <= 0.0)
            {   fprintf(fplog, "Warning: LODK2 = %g is not positive.\n",model->BSIM4v4lodk2);
                printf("Warning: LODK2 = %g is not positive.\n",model->BSIM4v4lodk2);
            }
            if (model->BSIM4v4lodeta0 <= 0.0)
            {   fprintf(fplog, "Warning: LODETA0 = %g is not positive.\n",model->BSIM4v4lodeta0);
            	printf("Warning: LODETA0 = %g is not positive.\n",model->BSIM4v4lodeta0);
            }
 	}

/* Check gate resistance parameters */
        if (here->BSIM4v4rgateMod == 1)
        {   if (model->BSIM4v4rshg <= 0.0)
	        printf("Warning: rshg should be positive for rgateMod = 1.\n");
	}
        else if (here->BSIM4v4rgateMod == 2)
        {   if (model->BSIM4v4rshg <= 0.0)
                printf("Warning: rshg <= 0.0 for rgateMod = 2.\n");
            else if (pParam->BSIM4v4xrcrg1 <= 0.0)
                     printf("Warning: xrcrg1 <= 0.0 for rgateMod = 2.\n");
         }
        if (here->BSIM4v4rgateMod == 3)
        {   if (model->BSIM4v4rshg <= 0.0)
                printf("Warning: rshg should be positive for rgateMod = 3.\n");
            else if (pParam->BSIM4v4xrcrg1 <= 0.0)
                     printf("Warning: xrcrg1 should be positive for rgateMod = 3.\n");
         }

/* Check capacitance parameters */
        if (pParam->BSIM4v4noff < 0.1)
        {   fprintf(fplog, "Warning: Noff = %g is too small.\n",
                    pParam->BSIM4v4noff);
            printf("Warning: Noff = %g is too small.\n", pParam->BSIM4v4noff);
        }

        if (pParam->BSIM4v4voffcv < -0.5)
        {   fprintf(fplog, "Warning: Voffcv = %g is too small.\n",
                    pParam->BSIM4v4voffcv);
            printf("Warning: Voffcv = %g is too small.\n", pParam->BSIM4v4voffcv);
        }

        if (pParam->BSIM4v4moin < 5.0)
        {   fprintf(fplog, "Warning: Moin = %g is too small.\n",
                    pParam->BSIM4v4moin);
            printf("Warning: Moin = %g is too small.\n", pParam->BSIM4v4moin);
        }
        if (pParam->BSIM4v4moin > 25.0)
        {   fprintf(fplog, "Warning: Moin = %g is too large.\n",
                    pParam->BSIM4v4moin);
            printf("Warning: Moin = %g is too large.\n", pParam->BSIM4v4moin);
        }
	if(model->BSIM4v4capMod ==2) {
        	if (pParam->BSIM4v4acde < 0.1)
        	{   fprintf(fplog, "Warning:  Acde = %g is too small.\n",
                    	pParam->BSIM4v4acde);
            	printf("Warning: Acde = %g is too small.\n", pParam->BSIM4v4acde);
        	}
        	if (pParam->BSIM4v4acde > 1.6)
        	{   fprintf(fplog, "Warning:  Acde = %g is too large.\n",
                    	pParam->BSIM4v4acde);
            	printf("Warning: Acde = %g is too large.\n", pParam->BSIM4v4acde);
        	}
	}

/* Check overlap capacitance parameters */
        if (model->BSIM4v4cgdo < 0.0)
	{   fprintf(fplog, "Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM4v4cgdo);
	    printf("Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM4v4cgdo);
	    model->BSIM4v4cgdo = 0.0;
        }
        if (model->BSIM4v4cgso < 0.0)
	{   fprintf(fplog, "Warning: cgso = %g is negative. Set to zero.\n", model->BSIM4v4cgso);
	    printf("Warning: cgso = %g is negative. Set to zero.\n", model->BSIM4v4cgso);
	    model->BSIM4v4cgso = 0.0;
        }

      if (model->BSIM4v4tnoiMod == 1) {
        if (model->BSIM4v4tnoia < 0.0)
        {   fprintf(fplog, "Warning: tnoia = %g is negative. Set to zero.\n", model->BSIM4v4tnoia);
            printf("Warning: tnoia = %g is negative. Set to zero.\n", model->BSIM4v4tnoia);
            model->BSIM4v4tnoia = 0.0;
        }
        if (model->BSIM4v4tnoib < 0.0)
        {   fprintf(fplog, "Warning: tnoib = %g is negative. Set to zero.\n", model->BSIM4v4tnoib);
            printf("Warning: tnoib = %g is negative. Set to zero.\n", model->BSIM4v4tnoib);
            model->BSIM4v4tnoib = 0.0;
        }

         if (model->BSIM4v4rnoia < 0.0)
        {   fprintf(fplog, "Warning: rnoia = %g is negative. Set to zero.\n", model->BSIM4v4rnoia);
            printf("Warning: rnoia = %g is negative. Set to zero.\n", model->BSIM4v4rnoia);
            model->BSIM4v4rnoia = 0.0;
        }
        if (model->BSIM4v4rnoib < 0.0)
        {   fprintf(fplog, "Warning: rnoib = %g is negative. Set to zero.\n", model->BSIM4v4rnoib);
            printf("Warning: rnoib = %g is negative. Set to zero.\n", model->BSIM4v4rnoib);
            model->BSIM4v4rnoib = 0.0;
        }
      }

	if (model->BSIM4v4SjctEmissionCoeff < 0.0)
	{   fprintf(fplog, "Warning: Njs = %g is negative.\n",
		    model->BSIM4v4SjctEmissionCoeff);
	    printf("Warning: Njs = %g is negative.\n",
		    model->BSIM4v4SjctEmissionCoeff);
	}
	if (model->BSIM4v4DjctEmissionCoeff < 0.0)
	{   fprintf(fplog, "Warning: Njd = %g is negative.\n",
		    model->BSIM4v4DjctEmissionCoeff);
	    printf("Warning: Njd = %g is negative.\n",
		    model->BSIM4v4DjctEmissionCoeff);
	}
	if (model->BSIM4v4njtstemp < 0.0)
	{   fprintf(fplog, "Warning: Njts = %g is negative at temperature = %g.\n",
		    model->BSIM4v4njtstemp, ckt->CKTtemp);
	    printf("Warning: Njts = %g is negative at temperature = %g.\n",
		    model->BSIM4v4njtstemp, ckt->CKTtemp);
	}
	if (model->BSIM4v4njtsswtemp < 0.0)
	{   fprintf(fplog, "Warning: Njtssw = %g is negative at temperature = %g.\n",
		    model->BSIM4v4njtsswtemp, ckt->CKTtemp);
	    printf("Warning: Njtssw = %g is negative at temperature = %g.\n",
		    model->BSIM4v4njtsswtemp, ckt->CKTtemp);
	}
	if (model->BSIM4v4njtsswgtemp < 0.0)
	{   fprintf(fplog, "Warning: Njtsswg = %g is negative at temperature = %g.\n",
		    model->BSIM4v4njtsswgtemp, ckt->CKTtemp);
	    printf("Warning: Njtsswg = %g is negative at temperature = %g.\n",
		    model->BSIM4v4njtsswgtemp, ckt->CKTtemp);
	}
	if (model->BSIM4v4vtss < 0.0)
	{   fprintf(fplog, "Warning: Vtss = %g is negative.\n",
		    model->BSIM4v4vtss);
	    printf("Warning: Vtss = %g is negative.\n",
		    model->BSIM4v4vtss);
	}
	if (model->BSIM4v4vtsd < 0.0)
	{   fprintf(fplog, "Warning: Vtsd = %g is negative.\n",
		    model->BSIM4v4vtsd);
	    printf("Warning: Vtsd = %g is negative.\n",
		    model->BSIM4v4vtsd);
	}
	if (model->BSIM4v4vtssws < 0.0)
	{   fprintf(fplog, "Warning: Vtssws = %g is negative.\n",
		    model->BSIM4v4vtssws);
	    printf("Warning: Vtssws = %g is negative.\n",
		    model->BSIM4v4vtssws);
	}
	if (model->BSIM4v4vtsswd < 0.0)
	{   fprintf(fplog, "Warning: Vtsswd = %g is negative.\n",
		    model->BSIM4v4vtsswd);
	    printf("Warning: Vtsswd = %g is negative.\n",
		    model->BSIM4v4vtsswd);
	}
	if (model->BSIM4v4vtsswgs < 0.0)
	{   fprintf(fplog, "Warning: Vtsswgs = %g is negative.\n",
		    model->BSIM4v4vtsswgs);
	    printf("Warning: Vtsswgs = %g is negative.\n",
		    model->BSIM4v4vtsswgs);
	}
	if (model->BSIM4v4vtsswgd < 0.0)
	{   fprintf(fplog, "Warning: Vtsswgd = %g is negative.\n",
		    model->BSIM4v4vtsswgd);
	    printf("Warning: Vtsswgd = %g is negative.\n",
		    model->BSIM4v4vtsswgd);
	}

       if (model->BSIM4v4ntnoi < 0.0)
        {   fprintf(fplog, "Warning: ntnoi = %g is negative. Set to zero.\n", model->BSIM4v4ntnoi);
            printf("Warning: ntnoi = %g is negative. Set to zero.\n", model->BSIM4v4ntnoi);
            model->BSIM4v4ntnoi = 0.0;
        }

     }/* loop for the parameter check for warning messages */
	fclose(fplog);
    }
    else
    {   fprintf(stderr, "Warning: Can't open log file. Parameter checking skipped.\n");
    }

    return(Fatal_Flag);
}

