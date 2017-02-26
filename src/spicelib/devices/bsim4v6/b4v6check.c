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
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v6def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"
#include "ngspice/fteext.h"

int
BSIM4v6checkModel(
BSIM4v6model *model,
BSIM4v6instance *here,
CKTcircuit *ckt)
{
struct bsim4v6SizeDependParam *pParam;
int Fatal_Flag = 0;
FILE *fplog;
    /* add user defined path (nname has to be freed after usage) */
    char *nname = set_output_path("bsim4v6.out");
    if ((fplog = fopen(nname, "w")) != NULL)
    {   pParam = here->pParam;
        fprintf(fplog, "BSIM4v6: Berkeley Short Channel IGFET Model-4\n");
        fprintf(fplog, "Developed by Xuemei (Jane) Xi, Mohan Dunga, Prof. Ali Niknejad and Prof. Chenming Hu in 2003.\n");
        fprintf(fplog, "\n");
	fprintf(fplog, "++++++++++ BSIM4v6 PARAMETER CHECKING BELOW ++++++++++\n");

        if (strcmp(model->BSIM4v6version, "4.6.5") != 0)
        {  fprintf(fplog, "Warning: This model is BSIM4.6.5; you specified a wrong version number.\n");
           printf("Warning: This model is BSIM4.6.5; you specified a wrong version number.\n");
        }
	fprintf(fplog, "Model = %s\n", model->BSIM4v6modName);


        if ((here->BSIM4v6rgateMod == 2) || (here->BSIM4v6rgateMod == 3))
        {   if ((here->BSIM4v6trnqsMod == 1) || (here->BSIM4v6acnqsMod == 1))
            {   fprintf(fplog, "Warning: You've selected both Rg and charge deficit NQS; select one only.\n");
                printf("Warning: You've selected both Rg and charge deficit NQS; select one only.\n");
            }
        }

	if (model->BSIM4v6toxe <= 0.0)
	{   fprintf(fplog, "Fatal: Toxe = %g is not positive.\n",
		    model->BSIM4v6toxe);
	    printf("Fatal: Toxe = %g is not positive.\n", model->BSIM4v6toxe);
	    Fatal_Flag = 1;
	}
        if (model->BSIM4v6toxp <= 0.0)
        {   fprintf(fplog, "Fatal: Toxp = %g is not positive.\n",
                    model->BSIM4v6toxp);
            printf("Fatal: Toxp = %g is not positive.\n", model->BSIM4v6toxp);
            Fatal_Flag = 1;
        }
        if (model->BSIM4v6eot <= 0.0)
        {   fprintf(fplog, "Fatal: EOT = %g is not positive.\n",
                    model->BSIM4v6eot);
            printf("Fatal: EOT = %g is not positive.\n", model->BSIM4v6eot);
            Fatal_Flag = 1;
        }
        if (model->BSIM4v6epsrgate < 0.0)
        {   fprintf(fplog, "Fatal: Epsrgate = %g is not positive.\n",
                    model->BSIM4v6epsrgate);
            printf("Fatal: Epsrgate = %g is not positive.\n", model->BSIM4v6epsrgate);
            Fatal_Flag = 1;
        }
        if (model->BSIM4v6epsrsub < 0.0)
        {   fprintf(fplog, "Fatal: Epsrsub = %g is not positive.\n",
                    model->BSIM4v6epsrsub);
            printf("Fatal: Epsrsub = %g is not positive.\n", model->BSIM4v6epsrsub);
            Fatal_Flag = 1;
        }
        if (model->BSIM4v6easub < 0.0)
        {   fprintf(fplog, "Fatal: Easub = %g is not positive.\n",
                    model->BSIM4v6easub);
            printf("Fatal: Easub = %g is not positive.\n", model->BSIM4v6easub);
            Fatal_Flag = 1;
        }
        if (model->BSIM4v6ni0sub <= 0.0)
        {   fprintf(fplog, "Fatal: Ni0sub = %g is not positive.\n",
                    model->BSIM4v6ni0sub);
            printf("Fatal: Easub = %g is not positive.\n", model->BSIM4v6ni0sub);
            Fatal_Flag = 1;
        }

        if (model->BSIM4v6toxm <= 0.0)
        {   fprintf(fplog, "Fatal: Toxm = %g is not positive.\n",
                    model->BSIM4v6toxm);
            printf("Fatal: Toxm = %g is not positive.\n", model->BSIM4v6toxm);
            Fatal_Flag = 1;
        }

        if (model->BSIM4v6toxref <= 0.0)
        {   fprintf(fplog, "Fatal: Toxref = %g is not positive.\n",
                    model->BSIM4v6toxref);
            printf("Fatal: Toxref = %g is not positive.\n", model->BSIM4v6toxref);
            Fatal_Flag = 1;
        }

        if (pParam->BSIM4v6lpe0 < -pParam->BSIM4v6leff)
        {   fprintf(fplog, "Fatal: Lpe0 = %g is less than -Leff.\n",
                    pParam->BSIM4v6lpe0);
            printf("Fatal: Lpe0 = %g is less than -Leff.\n",
                        pParam->BSIM4v6lpe0);
            Fatal_Flag = 1;
        }
        if (model->BSIM4v6lintnoi > pParam->BSIM4v6leff/2)
        {   fprintf(fplog, "Fatal: Lintnoi = %g is too large - Leff for noise is negative.\n",
                    model->BSIM4v6lintnoi);
            printf("Fatal: Lintnoi = %g is too large - Leff for noise is negative.\n",
                    model->BSIM4v6lintnoi);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v6lpeb < -pParam->BSIM4v6leff)
        {   fprintf(fplog, "Fatal: Lpeb = %g is less than -Leff.\n",
                    pParam->BSIM4v6lpeb);
            printf("Fatal: Lpeb = %g is less than -Leff.\n",
                        pParam->BSIM4v6lpeb);
            Fatal_Flag = 1;
        }
	if (pParam->BSIM4v6ndep <= 0.0)
	{   fprintf(fplog, "Fatal: Ndep = %g is not positive.\n",
		    pParam->BSIM4v6ndep);
	    printf("Fatal: Ndep = %g is not positive.\n",
		   pParam->BSIM4v6ndep);
	    Fatal_Flag = 1;
	}
        if (pParam->BSIM4v6phi <= 0.0)
        {   fprintf(fplog, "Fatal: Phi = %g is not positive. Please check Phin and Ndep\n",
                    pParam->BSIM4v6phi);
            fprintf(fplog, "	   Phin = %g  Ndep = %g \n", 
            	    pParam->BSIM4v6phin, pParam->BSIM4v6ndep);
            printf("Fatal: Phi = %g is not positive. Please check Phin and Ndep\n",
                    pParam->BSIM4v6phi);
            printf("	   Phin = %g  Ndep = %g \n", 
            	    pParam->BSIM4v6phin, pParam->BSIM4v6ndep);
            Fatal_Flag = 1;
        }
	if (pParam->BSIM4v6nsub <= 0.0)
	{   fprintf(fplog, "Fatal: Nsub = %g is not positive.\n",
		    pParam->BSIM4v6nsub);
	    printf("Fatal: Nsub = %g is not positive.\n",
		   pParam->BSIM4v6nsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v6ngate < 0.0)
	{   fprintf(fplog, "Fatal: Ngate = %g is not positive.\n",
		    pParam->BSIM4v6ngate);
	    printf("Fatal: Ngate = %g Ngate is not positive.\n",
		   pParam->BSIM4v6ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v6ngate > 1.e25)
	{   fprintf(fplog, "Fatal: Ngate = %g is too high.\n",
		    pParam->BSIM4v6ngate);
	    printf("Fatal: Ngate = %g Ngate is too high\n",
		   pParam->BSIM4v6ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v6xj <= 0.0)
	{   fprintf(fplog, "Fatal: Xj = %g is not positive.\n",
		    pParam->BSIM4v6xj);
	    printf("Fatal: Xj = %g is not positive.\n", pParam->BSIM4v6xj);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4v6dvt1 < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1 = %g is negative.\n",
		    pParam->BSIM4v6dvt1);   
	    printf("Fatal: Dvt1 = %g is negative.\n", pParam->BSIM4v6dvt1);   
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM4v6dvt1w < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1w = %g is negative.\n",
		    pParam->BSIM4v6dvt1w);
	    printf("Fatal: Dvt1w = %g is negative.\n", pParam->BSIM4v6dvt1w);
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM4v6w0 == -pParam->BSIM4v6weff)
	{   fprintf(fplog, "Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }   

	if (pParam->BSIM4v6dsub < 0.0)
	{   fprintf(fplog, "Fatal: Dsub = %g is negative.\n", pParam->BSIM4v6dsub);
	    printf("Fatal: Dsub = %g is negative.\n", pParam->BSIM4v6dsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v6b1 == -pParam->BSIM4v6weff)
	{   fprintf(fplog, "Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }  
        if (here->BSIM4v6u0temp <= 0.0)
	{   fprintf(fplog, "Fatal: u0 at current temperature = %g is not positive.\n", here->BSIM4v6u0temp);
	    printf("Fatal: u0 at current temperature = %g is not positive.\n",
		   here->BSIM4v6u0temp);
	    Fatal_Flag = 1;
        }
    
        if (pParam->BSIM4v6delta < 0.0)
	{   fprintf(fplog, "Fatal: Delta = %g is less than zero.\n",
		    pParam->BSIM4v6delta);
	    printf("Fatal: Delta = %g is less than zero.\n", pParam->BSIM4v6delta);
	    Fatal_Flag = 1;
        }      

	if (here->BSIM4v6vsattemp <= 0.0)
	{   fprintf(fplog, "Fatal: Vsat at current temperature = %g is not positive.\n", here->BSIM4v6vsattemp);
	    printf("Fatal: Vsat at current temperature = %g is not positive.\n",
		   here->BSIM4v6vsattemp);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4v6pclm <= 0.0)
	{   fprintf(fplog, "Fatal: Pclm = %g is not positive.\n", pParam->BSIM4v6pclm);
	    printf("Fatal: Pclm = %g is not positive.\n", pParam->BSIM4v6pclm);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4v6drout < 0.0)
	{   fprintf(fplog, "Fatal: Drout = %g is negative.\n", pParam->BSIM4v6drout);
	    printf("Fatal: Drout = %g is negative.\n", pParam->BSIM4v6drout);
	    Fatal_Flag = 1;
	}

        if (here->BSIM4v6m < 1.0)
        {   fprintf(fplog, "Fatal: Number of multiplier = %g is smaller than one.\n", here->BSIM4v6m);
            printf("Fatal: Number of multiplier = %g is smaller than one.\n", here->BSIM4v6m);
            Fatal_Flag = 1;
        }

        if (here->BSIM4v6nf < 1.0)
        {   fprintf(fplog, "Fatal: Number of finger = %g is smaller than one.\n", here->BSIM4v6nf);
            printf("Fatal: Number of finger = %g is smaller than one.\n", here->BSIM4v6nf);
            Fatal_Flag = 1;
        }

        if((here->BSIM4v6sa > 0.0) && (here->BSIM4v6sb > 0.0) && 
       	((here->BSIM4v6nf == 1.0) || ((here->BSIM4v6nf > 1.0) && (here->BSIM4v6sd > 0.0))) )
        {   if (model->BSIM4v6saref <= 0.0)
            {   fprintf(fplog, "Fatal: SAref = %g is not positive.\n",model->BSIM4v6saref);
             	printf("Fatal: SAref = %g is not positive.\n",model->BSIM4v6saref);
             	Fatal_Flag = 1;
            }
            if (model->BSIM4v6sbref <= 0.0)
            {   fprintf(fplog, "Fatal: SBref = %g is not positive.\n",model->BSIM4v6sbref);
            	printf("Fatal: SBref = %g is not positive.\n",model->BSIM4v6sbref);
            	Fatal_Flag = 1;
            }
 	}

        if ((here->BSIM4v6l + model->BSIM4v6xl) <= model->BSIM4v6xgl)
        {   fprintf(fplog, "Fatal: The parameter xgl must be smaller than Ldrawn+XL.\n");
            printf("Fatal: The parameter xgl must be smaller than Ldrawn+XL.\n");
            Fatal_Flag = 1;
        }
        if (here->BSIM4v6ngcon < 1.0)
        {   fprintf(fplog, "Fatal: The parameter ngcon cannot be smaller than one.\n");
            printf("Fatal: The parameter ngcon cannot be smaller than one.\n");
            Fatal_Flag = 1;
        }
        if ((here->BSIM4v6ngcon != 1.0) && (here->BSIM4v6ngcon != 2.0))
        {   here->BSIM4v6ngcon = 1.0;
            fprintf(fplog, "Warning: Ngcon must be equal to one or two; reset to 1.0.\n");
            printf("Warning: Ngcon must be equal to one or two; reset to 1.0.\n");
        }

        if (model->BSIM4v6gbmin < 1.0e-20)
        {   fprintf(fplog, "Warning: Gbmin = %g is too small.\n",
                    model->BSIM4v6gbmin);
            printf("Warning: Gbmin = %g is too small.\n", model->BSIM4v6gbmin);
        }

        /* Check saturation parameters */
        if (pParam->BSIM4v6fprout < 0.0)
        {   fprintf(fplog, "Fatal: fprout = %g is negative.\n",
                    pParam->BSIM4v6fprout);
            printf("Fatal: fprout = %g is negative.\n", pParam->BSIM4v6fprout);
	    Fatal_Flag = 1;
        }
        if (pParam->BSIM4v6pdits < 0.0)
        {   fprintf(fplog, "Fatal: pdits = %g is negative.\n",
                    pParam->BSIM4v6pdits);
            printf("Fatal: pdits = %g is negative.\n", pParam->BSIM4v6pdits);
            Fatal_Flag = 1;
        }
        if (model->BSIM4v6pditsl < 0.0)
        {   fprintf(fplog, "Fatal: pditsl = %g is negative.\n",
                    model->BSIM4v6pditsl);
            printf("Fatal: pditsl = %g is negative.\n", model->BSIM4v6pditsl);
            Fatal_Flag = 1;
        }

        /* Check gate current parameters */
      if (model->BSIM4v6igbMod) {
        if (pParam->BSIM4v6nigbinv <= 0.0)
        {   fprintf(fplog, "Fatal: nigbinv = %g is non-positive.\n",
                    pParam->BSIM4v6nigbinv);
            printf("Fatal: nigbinv = %g is non-positive.\n", pParam->BSIM4v6nigbinv);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v6nigbacc <= 0.0)
        {   fprintf(fplog, "Fatal: nigbacc = %g is non-positive.\n",
                    pParam->BSIM4v6nigbacc);
            printf("Fatal: nigbacc = %g is non-positive.\n", pParam->BSIM4v6nigbacc);
            Fatal_Flag = 1;
        }
      }
      if (model->BSIM4v6igcMod) {
        if (pParam->BSIM4v6nigc <= 0.0)
        {   fprintf(fplog, "Fatal: nigc = %g is non-positive.\n",
                    pParam->BSIM4v6nigc);
            printf("Fatal: nigc = %g is non-positive.\n", pParam->BSIM4v6nigc);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v6poxedge <= 0.0)
        {   fprintf(fplog, "Fatal: poxedge = %g is non-positive.\n",
                    pParam->BSIM4v6poxedge);
            printf("Fatal: poxedge = %g is non-positive.\n", pParam->BSIM4v6poxedge);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v6pigcd <= 0.0)
        {   fprintf(fplog, "Fatal: pigcd = %g is non-positive.\n",
                    pParam->BSIM4v6pigcd);
            printf("Fatal: pigcd = %g is non-positive.\n", pParam->BSIM4v6pigcd);
            Fatal_Flag = 1;
        }
      }

        /* Check capacitance parameters */
        if (pParam->BSIM4v6clc < 0.0)
	{   fprintf(fplog, "Fatal: Clc = %g is negative.\n", pParam->BSIM4v6clc);
	    printf("Fatal: Clc = %g is negative.\n", pParam->BSIM4v6clc);
	    Fatal_Flag = 1;
        }      

        /* Check overlap capacitance parameters */
        if (pParam->BSIM4v6ckappas < 0.02)
        {   fprintf(fplog, "Warning: ckappas = %g is too small. Set to 0.02\n",
                    pParam->BSIM4v6ckappas);
            printf("Warning: ckappas = %g is too small.\n", pParam->BSIM4v6ckappas);
            pParam->BSIM4v6ckappas = 0.02;
       }
        if (pParam->BSIM4v6ckappad < 0.02)
        {   fprintf(fplog, "Warning: ckappad = %g is too small. Set to 0.02\n",
                    pParam->BSIM4v6ckappad);
            printf("Warning: ckappad = %g is too small.\n", pParam->BSIM4v6ckappad);
            pParam->BSIM4v6ckappad = 0.02;
        }

	if (model->BSIM4v6vtss < 0.0)
	{   fprintf(fplog, "Fatal: Vtss = %g is negative.\n",
			model->BSIM4v6vtss);
	    printf("Fatal: Vtss = %g is negative.\n",
			model->BSIM4v6vtss);
	    Fatal_Flag = 1;
	}
	if (model->BSIM4v6vtsd < 0.0)
	{   fprintf(fplog, "Fatal: Vtsd = %g is negative.\n",
			model->BSIM4v6vtsd);
	    printf("Fatal: Vtsd = %g is negative.\n",
		    model->BSIM4v6vtsd);
	    Fatal_Flag = 1;
	}
	if (model->BSIM4v6vtssws < 0.0)
	{   fprintf(fplog, "Fatal: Vtssws = %g is negative.\n",
		    model->BSIM4v6vtssws);
	    printf("Fatal: Vtssws = %g is negative.\n",
		    model->BSIM4v6vtssws);
	    Fatal_Flag = 1;
	}
	if (model->BSIM4v6vtsswd < 0.0)
	{   fprintf(fplog, "Fatal: Vtsswd = %g is negative.\n",
		    model->BSIM4v6vtsswd);
	    printf("Fatal: Vtsswd = %g is negative.\n",
		    model->BSIM4v6vtsswd);
	    Fatal_Flag = 1;
	}
	if (model->BSIM4v6vtsswgs < 0.0)
	{   fprintf(fplog, "Fatal: Vtsswgs = %g is negative.\n",
		    model->BSIM4v6vtsswgs);
	    printf("Fatal: Vtsswgs = %g is negative.\n",
		    model->BSIM4v6vtsswgs);
	    Fatal_Flag = 1;
	}
	if (model->BSIM4v6vtsswgd < 0.0)
	{   fprintf(fplog, "Fatal: Vtsswgd = %g is negative.\n",
		    model->BSIM4v6vtsswgd);
	    printf("Fatal: Vtsswgd = %g is negative.\n",
			model->BSIM4v6vtsswgd);
	    Fatal_Flag = 1;
	}


      if (model->BSIM4v6paramChk ==1)
      {
/* Check L and W parameters */ 
	if (pParam->BSIM4v6leff <= 1.0e-9)
	{   fprintf(fplog, "Warning: Leff = %g <= 1.0e-9. Recommended Leff >= 1e-8 \n", 
		    pParam->BSIM4v6leff);
	    printf("Warning: Leff = %g <= 1.0e-9. Recommended Leff >= 1e-8 \n",
		    pParam->BSIM4v6leff);
	}    
	
	if (pParam->BSIM4v6leffCV <= 1.0e-9)
	{   fprintf(fplog, "Warning: Leff for CV = %g <= 1.0e-9. Recommended LeffCV >=1e-8 \n",
		    pParam->BSIM4v6leffCV);
	    printf("Warning: Leff for CV = %g <= 1.0e-9. Recommended LeffCV >=1e-8 \n",
		    pParam->BSIM4v6leffCV);
	}  
	
        if (pParam->BSIM4v6weff <= 1.0e-9)
	{   fprintf(fplog, "Warning: Weff = %g <= 1.0e-9. Recommended Weff >=1e-7 \n",
		    pParam->BSIM4v6weff);
	    printf("Warning: Weff = %g <= 1.0e-9. Recommended Weff >=1e-7 \n",
		   pParam->BSIM4v6weff);
	}             
	
	if (pParam->BSIM4v6weffCV <= 1.0e-9)
	{   fprintf(fplog, "Warning: Weff for CV = %g <= 1.0e-9. Recommended WeffCV >= 1e-7 \n",
		    pParam->BSIM4v6weffCV);
	    printf("Warning: Weff for CV = %g <= 1.0e-9. Recommended WeffCV >= 1e-7 \n",
		    pParam->BSIM4v6weffCV);
	}        
	
        /* Check threshold voltage parameters */
	if (model->BSIM4v6toxe < 1.0e-10)
	{   fprintf(fplog, "Warning: Toxe = %g is less than 1A. Recommended Toxe >= 5A\n",
	            model->BSIM4v6toxe);
	    printf("Warning: Toxe = %g is less than 1A. Recommended Toxe >= 5A\n", model->BSIM4v6toxe);
        }
        if (model->BSIM4v6toxp < 1.0e-10)
        {   fprintf(fplog, "Warning: Toxp = %g is less than 1A. Recommended Toxp >= 5A\n",
                    model->BSIM4v6toxp);
            printf("Warning: Toxp = %g is less than 1A. Recommended Toxp >= 5A\n", model->BSIM4v6toxp);
        }
        if (model->BSIM4v6toxm < 1.0e-10)
        {   fprintf(fplog, "Warning: Toxm = %g is less than 1A. Recommended Toxm >= 5A\n",
                    model->BSIM4v6toxm);
            printf("Warning: Toxm = %g is less than 1A. Recommended Toxm >= 5A\n", model->BSIM4v6toxm);
        }

        if (pParam->BSIM4v6ndep <= 1.0e12)
	{   fprintf(fplog, "Warning: Ndep = %g may be too small.\n",
	            pParam->BSIM4v6ndep);
	    printf("Warning: Ndep = %g may be too small.\n",
	           pParam->BSIM4v6ndep);
	}
	else if (pParam->BSIM4v6ndep >= 1.0e21)
	{   fprintf(fplog, "Warning: Ndep = %g may be too large.\n",
	            pParam->BSIM4v6ndep);
	    printf("Warning: Ndep = %g may be too large.\n",
	           pParam->BSIM4v6ndep);
	}

	 if (pParam->BSIM4v6nsub <= 1.0e14)
	{   fprintf(fplog, "Warning: Nsub = %g may be too small.\n",
	            pParam->BSIM4v6nsub);
	    printf("Warning: Nsub = %g may be too small.\n",
	           pParam->BSIM4v6nsub);
	}
	else if (pParam->BSIM4v6nsub >= 1.0e21)
	{   fprintf(fplog, "Warning: Nsub = %g may be too large.\n",
	            pParam->BSIM4v6nsub);
	    printf("Warning: Nsub = %g may be too large.\n",
	           pParam->BSIM4v6nsub);
	}

	if ((pParam->BSIM4v6ngate > 0.0) &&
	    (pParam->BSIM4v6ngate <= 1.e18))
	{   fprintf(fplog, "Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	            pParam->BSIM4v6ngate);
	    printf("Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	           pParam->BSIM4v6ngate);
	}
       
        if (pParam->BSIM4v6dvt0 < 0.0)
	{   fprintf(fplog, "Warning: Dvt0 = %g is negative.\n",
		    pParam->BSIM4v6dvt0);   
	    printf("Warning: Dvt0 = %g is negative.\n", pParam->BSIM4v6dvt0);   
	}
	    
	if (fabs(1.0e-8 / (pParam->BSIM4v6w0 + pParam->BSIM4v6weff)) > 10.0)
	{   fprintf(fplog, "Warning: (W0 + Weff) may be too small.\n");
	    printf("Warning: (W0 + Weff) may be too small.\n");
        }

/* Check subthreshold parameters */
	if (pParam->BSIM4v6nfactor < 0.0)
	{   fprintf(fplog, "Warning: Nfactor = %g is negative.\n",
		    pParam->BSIM4v6nfactor);
	    printf("Warning: Nfactor = %g is negative.\n", pParam->BSIM4v6nfactor);
	}
	if (pParam->BSIM4v6cdsc < 0.0)
	{   fprintf(fplog, "Warning: Cdsc = %g is negative.\n",
		    pParam->BSIM4v6cdsc);
	    printf("Warning: Cdsc = %g is negative.\n", pParam->BSIM4v6cdsc);
	}
	if (pParam->BSIM4v6cdscd < 0.0)
	{   fprintf(fplog, "Warning: Cdscd = %g is negative.\n",
		    pParam->BSIM4v6cdscd);
	    printf("Warning: Cdscd = %g is negative.\n", pParam->BSIM4v6cdscd);
	}
/* Check DIBL parameters */
	if (here->BSIM4v6eta0 < 0.0)
	{   fprintf(fplog, "Warning: Eta0 = %g is negative.\n",
		    here->BSIM4v6eta0); 
	    printf("Warning: Eta0 = %g is negative.\n", here->BSIM4v6eta0); 
	}
	      
/* Check Abulk parameters */	    
	 if (fabs(1.0e-8 / (pParam->BSIM4v6b1 + pParam->BSIM4v6weff)) > 10.0)
       	{   fprintf(fplog, "Warning: (B1 + Weff) may be too small.\n");
       	    printf("Warning: (B1 + Weff) may be too small.\n");
        }    
    

/* Check Saturation parameters */
     	if (pParam->BSIM4v6a2 < 0.01)
	{   fprintf(fplog, "Warning: A2 = %g is too small. Set to 0.01.\n", pParam->BSIM4v6a2);
	    printf("Warning: A2 = %g is too small. Set to 0.01.\n",
		   pParam->BSIM4v6a2);
	    pParam->BSIM4v6a2 = 0.01;
	}
	else if (pParam->BSIM4v6a2 > 1.0)
	{   fprintf(fplog, "Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		    pParam->BSIM4v6a2);
	    printf("Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		   pParam->BSIM4v6a2);
	    pParam->BSIM4v6a2 = 1.0;
	    pParam->BSIM4v6a1 = 0.0;
	}

        if (pParam->BSIM4v6prwg < 0.0)
        {   fprintf(fplog, "Warning: Prwg = %g is negative. Set to zero.\n",
                    pParam->BSIM4v6prwg);
            printf("Warning: Prwg = %g is negative. Set to zero.\n",
                   pParam->BSIM4v6prwg);
            pParam->BSIM4v6prwg = 0.0;
        }

	if (pParam->BSIM4v6rdsw < 0.0)
	{   fprintf(fplog, "Warning: Rdsw = %g is negative. Set to zero.\n",
		    pParam->BSIM4v6rdsw);
	    printf("Warning: Rdsw = %g is negative. Set to zero.\n",
		   pParam->BSIM4v6rdsw);
	    pParam->BSIM4v6rdsw = 0.0;
	    pParam->BSIM4v6rds0 = 0.0;
	}

	if (pParam->BSIM4v6rds0 < 0.0)
	{   fprintf(fplog, "Warning: Rds at current temperature = %g is negative. Set to zero.\n",
		    pParam->BSIM4v6rds0);
	    printf("Warning: Rds at current temperature = %g is negative. Set to zero.\n",
		   pParam->BSIM4v6rds0);
	    pParam->BSIM4v6rds0 = 0.0;
	}

        if (pParam->BSIM4v6rdswmin < 0.0)
        {   fprintf(fplog, "Warning: Rdswmin at current temperature = %g is negative. Set to zero.\n",
                    pParam->BSIM4v6rdswmin);
            printf("Warning: Rdswmin at current temperature = %g is negative. Set to zero.\n",
                   pParam->BSIM4v6rdswmin);
            pParam->BSIM4v6rdswmin = 0.0;
        }

        if (pParam->BSIM4v6pscbe2 <= 0.0)
        {   fprintf(fplog, "Warning: Pscbe2 = %g is not positive.\n",
                    pParam->BSIM4v6pscbe2);
            printf("Warning: Pscbe2 = %g is not positive.\n", pParam->BSIM4v6pscbe2);
        }

        if (pParam->BSIM4v6vsattemp < 1.0e3)
	{   fprintf(fplog, "Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM4v6vsattemp);
	   printf("Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM4v6vsattemp);
	}

      if((model->BSIM4v6lambdaGiven) && (pParam->BSIM4v6lambda > 0.0) ) 
      {
        if (pParam->BSIM4v6lambda > 1.0e-9)
	{   fprintf(fplog, "Warning: Lambda = %g may be too large.\n", pParam->BSIM4v6lambda);
	   printf("Warning: Lambda = %g may be too large.\n", pParam->BSIM4v6lambda);
	}
      }

      if((model->BSIM4v6vtlGiven) && (pParam->BSIM4v6vtl > 0.0) )
      {  
        if (pParam->BSIM4v6vtl < 6.0e4)
	{   fprintf(fplog, "Warning: Thermal velocity vtl = %g may be too small.\n", pParam->BSIM4v6vtl);
	   printf("Warning: Thermal velocity vtl = %g may be too small.\n", pParam->BSIM4v6vtl);
	}

        if (pParam->BSIM4v6xn < 3.0)
	{   fprintf(fplog, "Warning: back scattering coeff xn = %g is too small.\n", pParam->BSIM4v6xn);
	    printf("Warning: back scattering coeff xn = %g is too small. Reset to 3.0 \n", pParam->BSIM4v6xn);
	    pParam->BSIM4v6xn = 3.0;
	}

        if (model->BSIM4v6lc < 0.0)
	{   fprintf(fplog, "Warning: back scattering coeff lc = %g is too small.\n", model->BSIM4v6lc);
	    printf("Warning: back scattering coeff lc = %g is too small. Reset to 0.0\n", model->BSIM4v6lc);
	    pParam->BSIM4v6lc = 0.0;
	}
      }

	if (pParam->BSIM4v6pdibl1 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl1 = %g is negative.\n",
		    pParam->BSIM4v6pdibl1);
	    printf("Warning: Pdibl1 = %g is negative.\n", pParam->BSIM4v6pdibl1);
	}
	if (pParam->BSIM4v6pdibl2 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl2 = %g is negative.\n",
		    pParam->BSIM4v6pdibl2);
	    printf("Warning: Pdibl2 = %g is negative.\n", pParam->BSIM4v6pdibl2);
	}

/* Check stress effect parameters */        
        if((here->BSIM4v6sa > 0.0) && (here->BSIM4v6sb > 0.0) && 
       	((here->BSIM4v6nf == 1.0) || ((here->BSIM4v6nf > 1.0) && (here->BSIM4v6sd > 0.0))) )
        {   if (model->BSIM4v6lodk2 <= 0.0)
            {   fprintf(fplog, "Warning: LODK2 = %g is not positive.\n",model->BSIM4v6lodk2);
                printf("Warning: LODK2 = %g is not positive.\n",model->BSIM4v6lodk2);
            }
            if (model->BSIM4v6lodeta0 <= 0.0)
            {   fprintf(fplog, "Warning: LODETA0 = %g is not positive.\n",model->BSIM4v6lodeta0);
            	printf("Warning: LODETA0 = %g is not positive.\n",model->BSIM4v6lodeta0);
            }
 	}

/* Check gate resistance parameters */        
        if (here->BSIM4v6rgateMod == 1)
        {   if (model->BSIM4v6rshg <= 0.0)
	        printf("Warning: rshg should be positive for rgateMod = 1.\n");
	}
        else if (here->BSIM4v6rgateMod == 2)
        {   if (model->BSIM4v6rshg <= 0.0)
                printf("Warning: rshg <= 0.0 for rgateMod = 2.\n");
            else if (pParam->BSIM4v6xrcrg1 <= 0.0)
                     printf("Warning: xrcrg1 <= 0.0 for rgateMod = 2.\n");
         }
        if (here->BSIM4v6rgateMod == 3)
        {   if (model->BSIM4v6rshg <= 0.0)
                printf("Warning: rshg should be positive for rgateMod = 3.\n");
            else if (pParam->BSIM4v6xrcrg1 <= 0.0)
                     printf("Warning: xrcrg1 should be positive for rgateMod = 3.\n");
         }
         
/* Check capacitance parameters */
        if (pParam->BSIM4v6noff < 0.1)
        {   fprintf(fplog, "Warning: Noff = %g is too small.\n",
                    pParam->BSIM4v6noff);
            printf("Warning: Noff = %g is too small.\n", pParam->BSIM4v6noff);
        }

        if (pParam->BSIM4v6voffcv < -0.5)
        {   fprintf(fplog, "Warning: Voffcv = %g is too small.\n",
                    pParam->BSIM4v6voffcv);
            printf("Warning: Voffcv = %g is too small.\n", pParam->BSIM4v6voffcv);
        }
        if (pParam->BSIM4v6moin < 5.0)
        {   fprintf(fplog, "Warning: Moin = %g is too small.\n",
                    pParam->BSIM4v6moin);
            printf("Warning: Moin = %g is too small.\n", pParam->BSIM4v6moin);
        }
        if (pParam->BSIM4v6moin > 25.0)
        {   fprintf(fplog, "Warning: Moin = %g is too large.\n",
                    pParam->BSIM4v6moin);
            printf("Warning: Moin = %g is too large.\n", pParam->BSIM4v6moin);
        }
	if(model->BSIM4v6capMod ==2) {
        	if (pParam->BSIM4v6acde < 0.1)
        	{   fprintf(fplog, "Warning:  Acde = %g is too small.\n",
                    	pParam->BSIM4v6acde);
            	printf("Warning: Acde = %g is too small.\n", pParam->BSIM4v6acde);
        	}
        	if (pParam->BSIM4v6acde > 1.6)
        	{   fprintf(fplog, "Warning:  Acde = %g is too large.\n",
                    	pParam->BSIM4v6acde);
            	printf("Warning: Acde = %g is too large.\n", pParam->BSIM4v6acde);
        	}
	}

/* Check overlap capacitance parameters */
        if (model->BSIM4v6cgdo < 0.0)
	{   fprintf(fplog, "Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM4v6cgdo);
	    printf("Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM4v6cgdo);
	    model->BSIM4v6cgdo = 0.0;
        }      
        if (model->BSIM4v6cgso < 0.0)
	{   fprintf(fplog, "Warning: cgso = %g is negative. Set to zero.\n", model->BSIM4v6cgso);
	    printf("Warning: cgso = %g is negative. Set to zero.\n", model->BSIM4v6cgso);
	    model->BSIM4v6cgso = 0.0;
        }      
        if (model->BSIM4v6cgbo < 0.0)
	{   fprintf(fplog, "Warning: cgbo = %g is negative. Set to zero.\n", model->BSIM4v6cgbo);
	    printf("Warning: cgbo = %g is negative. Set to zero.\n", model->BSIM4v6cgbo);
	    model->BSIM4v6cgbo = 0.0;
        }      
      if (model->BSIM4v6tnoiMod == 1) { 
        if (model->BSIM4v6tnoia < 0.0)
        {   fprintf(fplog, "Warning: tnoia = %g is negative. Set to zero.\n", model->BSIM4v6tnoia);
            printf("Warning: tnoia = %g is negative. Set to zero.\n", model->BSIM4v6tnoia);
            model->BSIM4v6tnoia = 0.0;
        }
        if (model->BSIM4v6tnoib < 0.0)
        {   fprintf(fplog, "Warning: tnoib = %g is negative. Set to zero.\n", model->BSIM4v6tnoib);
            printf("Warning: tnoib = %g is negative. Set to zero.\n", model->BSIM4v6tnoib);
            model->BSIM4v6tnoib = 0.0;
        }

         if (model->BSIM4v6rnoia < 0.0)
        {   fprintf(fplog, "Warning: rnoia = %g is negative. Set to zero.\n", model->BSIM4v6rnoia);
            printf("Warning: rnoia = %g is negative. Set to zero.\n", model->BSIM4v6rnoia);
            model->BSIM4v6rnoia = 0.0;
        }
        if (model->BSIM4v6rnoib < 0.0)
        {   fprintf(fplog, "Warning: rnoib = %g is negative. Set to zero.\n", model->BSIM4v6rnoib);
            printf("Warning: rnoib = %g is negative. Set to zero.\n", model->BSIM4v6rnoib);
            model->BSIM4v6rnoib = 0.0;
        }
      }

	if (model->BSIM4v6SjctEmissionCoeff < 0.0)
	{   fprintf(fplog, "Warning: Njs = %g is negative.\n",
		    model->BSIM4v6SjctEmissionCoeff);
	    printf("Warning: Njs = %g is negative.\n",
		    model->BSIM4v6SjctEmissionCoeff);
	}
	if (model->BSIM4v6DjctEmissionCoeff < 0.0)
	{   fprintf(fplog, "Warning: Njd = %g is negative.\n",
		    model->BSIM4v6DjctEmissionCoeff);
	    printf("Warning: Njd = %g is negative.\n",
		    model->BSIM4v6DjctEmissionCoeff);
	}
        
	if (model->BSIM4v6njtsstemp < 0.0)
	{   fprintf(fplog, "Warning: Njts = %g is negative at temperature = %g.\n",
                    model->BSIM4v6njtsstemp, ckt->CKTtemp);
	    printf("Warning: Njts = %g is negative at temperature = %g.\n",
                   model->BSIM4v6njtsstemp, ckt->CKTtemp);
	}
	if (model->BSIM4v6njtsswstemp < 0.0)
	{   fprintf(fplog, "Warning: Njtssw = %g is negative at temperature = %g.\n",
                    model->BSIM4v6njtsswstemp, ckt->CKTtemp);
	    printf("Warning: Njtssw = %g is negative at temperature = %g.\n",
		    model->BSIM4v6njtsswstemp, ckt->CKTtemp);
	}
	if (model->BSIM4v6njtsswgstemp < 0.0)
	{   fprintf(fplog, "Warning: Njtsswg = %g is negative at temperature = %g.\n",
		    model->BSIM4v6njtsswgstemp, ckt->CKTtemp);
	    printf("Warning: Njtsswg = %g is negative at temperature = %g.\n",
		    model->BSIM4v6njtsswgstemp, ckt->CKTtemp);
	}

	if (model->BSIM4v6njtsdGiven && model->BSIM4v6njtsdtemp < 0.0)
	{   fprintf(fplog, "Warning: Njtsd = %g is negative at temperature = %g.\n",
		    model->BSIM4v6njtsdtemp, ckt->CKTtemp);
	    printf("Warning: Njtsd = %g is negative at temperature = %g.\n",
		    model->BSIM4v6njtsdtemp, ckt->CKTtemp);
	}
	if (model->BSIM4v6njtsswdGiven && model->BSIM4v6njtsswdtemp < 0.0)
	{   fprintf(fplog, "Warning: Njtsswd = %g is negative at temperature = %g.\n",
		    model->BSIM4v6njtsswdtemp, ckt->CKTtemp);
	    printf("Warning: Njtsswd = %g is negative at temperature = %g.\n",
		    model->BSIM4v6njtsswdtemp, ckt->CKTtemp);
	}
	if (model->BSIM4v6njtsswgdGiven && model->BSIM4v6njtsswgdtemp < 0.0)
	{   fprintf(fplog, "Warning: Njtsswgd = %g is negative at temperature = %g.\n",
		    model->BSIM4v6njtsswgdtemp, ckt->CKTtemp);
	    printf("Warning: Njtsswgd = %g is negative at temperature = %g.\n",
		    model->BSIM4v6njtsswgdtemp, ckt->CKTtemp);
	}

       if (model->BSIM4v6ntnoi < 0.0)
        {   fprintf(fplog, "Warning: ntnoi = %g is negative. Set to zero.\n", model->BSIM4v6ntnoi);
            printf("Warning: ntnoi = %g is negative. Set to zero.\n", model->BSIM4v6ntnoi);
            model->BSIM4v6ntnoi = 0.0;
        }

        /* diode model */
       if (model->BSIM4v6SbulkJctBotGradingCoeff >= 0.99)
        {   fprintf(fplog, "Warning: MJS = %g is too big. Set to 0.99.\n", model->BSIM4v6SbulkJctBotGradingCoeff);
            printf("Warning: MJS = %g is too big. Set to 0.99.\n", model->BSIM4v6SbulkJctBotGradingCoeff);
            model->BSIM4v6SbulkJctBotGradingCoeff = 0.99;
        }
       if (model->BSIM4v6SbulkJctSideGradingCoeff >= 0.99)
        {   fprintf(fplog, "Warning: MJSWS = %g is too big. Set to 0.99.\n", model->BSIM4v6SbulkJctSideGradingCoeff);
            printf("Warning: MJSWS = %g is too big. Set to 0.99.\n", model->BSIM4v6SbulkJctSideGradingCoeff);
            model->BSIM4v6SbulkJctSideGradingCoeff = 0.99;
        }
       if (model->BSIM4v6SbulkJctGateSideGradingCoeff >= 0.99)
        {   fprintf(fplog, "Warning: MJSWGS = %g is too big. Set to 0.99.\n", model->BSIM4v6SbulkJctGateSideGradingCoeff);
            printf("Warning: MJSWGS = %g is too big. Set to 0.99.\n", model->BSIM4v6SbulkJctGateSideGradingCoeff);
            model->BSIM4v6SbulkJctGateSideGradingCoeff = 0.99;
        }

       if (model->BSIM4v6DbulkJctBotGradingCoeff >= 0.99)
        {   fprintf(fplog, "Warning: MJD = %g is too big. Set to 0.99.\n", model->BSIM4v6DbulkJctBotGradingCoeff);
            printf("Warning: MJD = %g is too big. Set to 0.99.\n", model->BSIM4v6DbulkJctBotGradingCoeff);
            model->BSIM4v6DbulkJctBotGradingCoeff = 0.99;
        }
       if (model->BSIM4v6DbulkJctSideGradingCoeff >= 0.99)
        {   fprintf(fplog, "Warning: MJSWD = %g is too big. Set to 0.99.\n", model->BSIM4v6DbulkJctSideGradingCoeff);
            printf("Warning: MJSWD = %g is too big. Set to 0.99.\n", model->BSIM4v6DbulkJctSideGradingCoeff);
            model->BSIM4v6DbulkJctSideGradingCoeff = 0.99;
        }
       if (model->BSIM4v6DbulkJctGateSideGradingCoeff >= 0.99)
        {   fprintf(fplog, "Warning: MJSWGD = %g is too big. Set to 0.99.\n", model->BSIM4v6DbulkJctGateSideGradingCoeff);
            printf("Warning: MJSWGD = %g is too big. Set to 0.99.\n", model->BSIM4v6DbulkJctGateSideGradingCoeff);
            model->BSIM4v6DbulkJctGateSideGradingCoeff = 0.99;
        }
	if (model->BSIM4v6wpemod == 1)
	{
		if (model->BSIM4v6scref <= 0.0)
		{   fprintf(fplog, "Warning: SCREF = %g is not positive. Set to 1e-6.\n", model->BSIM4v6scref);
		    printf("Warning: SCREF = %g is not positive. Set to 1e-6.\n", model->BSIM4v6scref);
		    model->BSIM4v6scref = 1e-6;
		}
	        /*Move these checks to temp.c for sceff calculation*/
			/*
			if (here->BSIM4v6sca < 0.0)
                {   fprintf(fplog, "Warning: SCA = %g is negative. Set to 0.0.\n", here->BSIM4v6sca);
                    printf("Warning: SCA = %g is negative. Set to 0.0.\n", here->BSIM4v6sca);
                    here->BSIM4v6sca = 0.0;
                }
                if (here->BSIM4v6scb < 0.0)
                {   fprintf(fplog, "Warning: SCB = %g is negative. Set to 0.0.\n", here->BSIM4v6scb);
                    printf("Warning: SCB = %g is negative. Set to 0.0.\n", here->BSIM4v6scb);
                    here->BSIM4v6scb = 0.0;
                }
                if (here->BSIM4v6scc < 0.0)
                {   fprintf(fplog, "Warning: SCC = %g is negative. Set to 0.0.\n", here->BSIM4v6scc);
                    printf("Warning: SCC = %g is negative. Set to 0.0.\n", here->BSIM4v6scc);
                    here->BSIM4v6scc = 0.0;
                }
                if (here->BSIM4v6sc < 0.0)
                {   fprintf(fplog, "Warning: SC = %g is negative. Set to 0.0.\n", here->BSIM4v6sc);
                    printf("Warning: SC = %g is negative. Set to 0.0.\n", here->BSIM4v6sc);
                    here->BSIM4v6sc = 0.0;
                }
				*/
				
	}
     }/* loop for the parameter check for warning messages */      
	fclose(fplog);
    }
    else
    {   fprintf(stderr, "Warning: Can't open log file. Parameter checking skipped.\n");
    }
    tfree(nname);
    return(Fatal_Flag);
}

