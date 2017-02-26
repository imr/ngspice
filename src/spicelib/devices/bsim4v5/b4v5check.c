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
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v5def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"
#include "ngspice/fteext.h"

int
BSIM4v5checkModel(
BSIM4v5model *model,
BSIM4v5instance *here,
CKTcircuit *ckt)
{
struct bsim4v5SizeDependParam *pParam;
int Fatal_Flag = 0;
FILE *fplog;
    /* add user defined path (nname has to be freed after usage) */
    char *nname = set_output_path("bsim4v5.out");
    if ((fplog = fopen(nname, "w")) != NULL)
    {   pParam = here->pParam;
        fprintf(fplog, "BSIM4v5: Berkeley Short Channel IGFET Model-4\n");
        fprintf(fplog, "Developed by Xuemei (Jane) Xi, Mohan Dunga, Prof. Ali Niknejad and Prof. Chenming Hu in 2003.\n");
        fprintf(fplog, "\n");
        fprintf(fplog, "++++++++++ BSIM4v5 PARAMETER CHECKING BELOW ++++++++++\n");

        if ((strcmp(model->BSIM4v5version, "4.5.0")) && (strncmp(model->BSIM4v5version, "4.50", 4)) && (strncmp(model->BSIM4v5version, "4.5", 3)))
        {  fprintf(fplog, "Warning: This model is BSIM4.5.0; you specified a wrong version number '%s'.\n", model->BSIM4v5version);
           printf("Warning: This model is BSIM4.5.0; you specified a wrong version number '%s'.\n", model->BSIM4v5version);
        }
        fprintf(fplog, "Model = %s\n", model->BSIM4v5modName);


        if ((here->BSIM4v5rgateMod == 2) || (here->BSIM4v5rgateMod == 3))
        {   if ((here->BSIM4v5trnqsMod == 1) || (here->BSIM4v5acnqsMod == 1))
            {   fprintf(fplog, "Warning: You've selected both Rg and charge deficit NQS; select one only.\n");
                printf("Warning: You've selected both Rg and charge deficit NQS; select one only.\n");
            }
        }

	if (model->BSIM4v5toxe <= 0.0)
	{   fprintf(fplog, "Fatal: Toxe = %g is not positive.\n",
		    model->BSIM4v5toxe);
	    printf("Fatal: Toxe = %g is not positive.\n", model->BSIM4v5toxe);
	    Fatal_Flag = 1;
	}
        if (model->BSIM4v5toxp <= 0.0)
        {   fprintf(fplog, "Fatal: Toxp = %g is not positive.\n",
                    model->BSIM4v5toxp);
            printf("Fatal: Toxp = %g is not positive.\n", model->BSIM4v5toxp);
            Fatal_Flag = 1;
        }

        if (model->BSIM4v5toxm <= 0.0)
        {   fprintf(fplog, "Fatal: Toxm = %g is not positive.\n",
                    model->BSIM4v5toxm);
            printf("Fatal: Toxm = %g is not positive.\n", model->BSIM4v5toxm);
            Fatal_Flag = 1;
        }

        if (model->BSIM4v5toxref <= 0.0)
        {   fprintf(fplog, "Fatal: Toxref = %g is not positive.\n",
                    model->BSIM4v5toxref);
            printf("Fatal: Toxref = %g is not positive.\n", model->BSIM4v5toxref);
            Fatal_Flag = 1;
        }

        if (pParam->BSIM4v5lpe0 < -pParam->BSIM4v5leff)
        {   fprintf(fplog, "Fatal: Lpe0 = %g is less than -Leff.\n",
                    pParam->BSIM4v5lpe0);
            printf("Fatal: Lpe0 = %g is less than -Leff.\n",
                        pParam->BSIM4v5lpe0);
            Fatal_Flag = 1;
        }
        if (model->BSIM4v5lintnoi > pParam->BSIM4v5leff/2)
        {   fprintf(fplog, "Fatal: Lintnoi = %g is too large - Leff for noise is negative.\n",
                    model->BSIM4v5lintnoi);
            printf("Fatal: Lintnoi = %g is too large - Leff for noise is negative.\n",
                    model->BSIM4v5lintnoi);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v5lpeb < -pParam->BSIM4v5leff)
        {   fprintf(fplog, "Fatal: Lpeb = %g is less than -Leff.\n",
                    pParam->BSIM4v5lpeb);
            printf("Fatal: Lpeb = %g is less than -Leff.\n",
                        pParam->BSIM4v5lpeb);
            Fatal_Flag = 1;
        }
	if (pParam->BSIM4v5ndep <= 0.0)
	{   fprintf(fplog, "Fatal: Ndep = %g is not positive.\n",
		    pParam->BSIM4v5ndep);
	    printf("Fatal: Ndep = %g is not positive.\n",
		   pParam->BSIM4v5ndep);
	    Fatal_Flag = 1;
	}
        if (pParam->BSIM4v5phi <= 0.0)
        {   fprintf(fplog, "Fatal: Phi = %g is not positive. Please check Phin and Ndep\n",
                    pParam->BSIM4v5phi);
            fprintf(fplog, "	   Phin = %g  Ndep = %g \n", 
            	    pParam->BSIM4v5phin, pParam->BSIM4v5ndep);
            printf("Fatal: Phi = %g is not positive. Please check Phin and Ndep\n",
                    pParam->BSIM4v5phi);
            printf("	   Phin = %g  Ndep = %g \n", 
            	    pParam->BSIM4v5phin, pParam->BSIM4v5ndep);
            Fatal_Flag = 1;
        }
	if (pParam->BSIM4v5nsub <= 0.0)
	{   fprintf(fplog, "Fatal: Nsub = %g is not positive.\n",
		    pParam->BSIM4v5nsub);
	    printf("Fatal: Nsub = %g is not positive.\n",
		   pParam->BSIM4v5nsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v5ngate < 0.0)
	{   fprintf(fplog, "Fatal: Ngate = %g is not positive.\n",
		    pParam->BSIM4v5ngate);
	    printf("Fatal: Ngate = %g Ngate is not positive.\n",
		   pParam->BSIM4v5ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v5ngate > 1.e25)
	{   fprintf(fplog, "Fatal: Ngate = %g is too high.\n",
		    pParam->BSIM4v5ngate);
	    printf("Fatal: Ngate = %g Ngate is too high\n",
		   pParam->BSIM4v5ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v5xj <= 0.0)
	{   fprintf(fplog, "Fatal: Xj = %g is not positive.\n",
		    pParam->BSIM4v5xj);
	    printf("Fatal: Xj = %g is not positive.\n", pParam->BSIM4v5xj);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4v5dvt1 < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1 = %g is negative.\n",
		    pParam->BSIM4v5dvt1);   
	    printf("Fatal: Dvt1 = %g is negative.\n", pParam->BSIM4v5dvt1);   
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM4v5dvt1w < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1w = %g is negative.\n",
		    pParam->BSIM4v5dvt1w);
	    printf("Fatal: Dvt1w = %g is negative.\n", pParam->BSIM4v5dvt1w);
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM4v5w0 == -pParam->BSIM4v5weff)
	{   fprintf(fplog, "Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }   

	if (pParam->BSIM4v5dsub < 0.0)
	{   fprintf(fplog, "Fatal: Dsub = %g is negative.\n", pParam->BSIM4v5dsub);
	    printf("Fatal: Dsub = %g is negative.\n", pParam->BSIM4v5dsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v5b1 == -pParam->BSIM4v5weff)
	{   fprintf(fplog, "Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }  
        if (here->BSIM4v5u0temp <= 0.0)
	{   fprintf(fplog, "Fatal: u0 at current temperature = %g is not positive.\n", here->BSIM4v5u0temp);
	    printf("Fatal: u0 at current temperature = %g is not positive.\n",
		   here->BSIM4v5u0temp);
	    Fatal_Flag = 1;
        }
    
        if (pParam->BSIM4v5delta < 0.0)
	{   fprintf(fplog, "Fatal: Delta = %g is less than zero.\n",
		    pParam->BSIM4v5delta);
	    printf("Fatal: Delta = %g is less than zero.\n", pParam->BSIM4v5delta);
	    Fatal_Flag = 1;
        }      

	if (here->BSIM4v5vsattemp <= 0.0)
	{   fprintf(fplog, "Fatal: Vsat at current temperature = %g is not positive.\n", here->BSIM4v5vsattemp);
	    printf("Fatal: Vsat at current temperature = %g is not positive.\n",
		   here->BSIM4v5vsattemp);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4v5pclm <= 0.0)
	{   fprintf(fplog, "Fatal: Pclm = %g is not positive.\n", pParam->BSIM4v5pclm);
	    printf("Fatal: Pclm = %g is not positive.\n", pParam->BSIM4v5pclm);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4v5drout < 0.0)
	{   fprintf(fplog, "Fatal: Drout = %g is negative.\n", pParam->BSIM4v5drout);
	    printf("Fatal: Drout = %g is negative.\n", pParam->BSIM4v5drout);
	    Fatal_Flag = 1;
	}

        if (here->BSIM4v5m <= 0.0)
        {   fprintf(fplog, "Fatal: multiplier = %g is not positive.\n", here->BSIM4v5m);
            printf("Fatal: multiplier = %g is not positive.\n", here->BSIM4v5m);
            Fatal_Flag = 1;
        }
        if (here->BSIM4v5nf < 1.0)
        {   fprintf(fplog, "Fatal: Number of finger = %g is smaller than one.\n", here->BSIM4v5nf);
            printf("Fatal: Number of finger = %g is smaller than one.\n", here->BSIM4v5nf);
            Fatal_Flag = 1;
        }

        if((here->BSIM4v5sa > 0.0) && (here->BSIM4v5sb > 0.0) && 
       	((here->BSIM4v5nf == 1.0) || ((here->BSIM4v5nf > 1.0) && (here->BSIM4v5sd > 0.0))) )
        {   if (model->BSIM4v5saref <= 0.0)
            {   fprintf(fplog, "Fatal: SAref = %g is not positive.\n",model->BSIM4v5saref);
             	printf("Fatal: SAref = %g is not positive.\n",model->BSIM4v5saref);
             	Fatal_Flag = 1;
            }
            if (model->BSIM4v5sbref <= 0.0)
            {   fprintf(fplog, "Fatal: SBref = %g is not positive.\n",model->BSIM4v5sbref);
            	printf("Fatal: SBref = %g is not positive.\n",model->BSIM4v5sbref);
            	Fatal_Flag = 1;
            }
 	}

        if ((here->BSIM4v5l + model->BSIM4v5xl) <= model->BSIM4v5xgl)
        {   fprintf(fplog, "Fatal: The parameter xgl must be smaller than Ldrawn+XL.\n");
            printf("Fatal: The parameter xgl must be smaller than Ldrawn+XL.\n");
            Fatal_Flag = 1;
        }
        if (here->BSIM4v5ngcon < 1.0)
        {   fprintf(fplog, "Fatal: The parameter ngcon cannot be smaller than one.\n");
            printf("Fatal: The parameter ngcon cannot be smaller than one.\n");
            Fatal_Flag = 1;
        }
        if ((here->BSIM4v5ngcon != 1.0) && (here->BSIM4v5ngcon != 2.0))
        {   here->BSIM4v5ngcon = 1.0;
            fprintf(fplog, "Warning: Ngcon must be equal to one or two; reset to 1.0.\n");
            printf("Warning: Ngcon must be equal to one or two; reset to 1.0.\n");
        }

        if (model->BSIM4v5gbmin < 1.0e-20)
        {   fprintf(fplog, "Warning: Gbmin = %g is too small.\n",
                    model->BSIM4v5gbmin);
            printf("Warning: Gbmin = %g is too small.\n", model->BSIM4v5gbmin);
        }

        /* Check saturation parameters */
        if (pParam->BSIM4v5fprout < 0.0)
        {   fprintf(fplog, "Fatal: fprout = %g is negative.\n",
                    pParam->BSIM4v5fprout);
            printf("Fatal: fprout = %g is negative.\n", pParam->BSIM4v5fprout);
	    Fatal_Flag = 1;
        }
        if (pParam->BSIM4v5pdits < 0.0)
        {   fprintf(fplog, "Fatal: pdits = %g is negative.\n",
                    pParam->BSIM4v5pdits);
            printf("Fatal: pdits = %g is negative.\n", pParam->BSIM4v5pdits);
            Fatal_Flag = 1;
        }
        if (model->BSIM4v5pditsl < 0.0)
        {   fprintf(fplog, "Fatal: pditsl = %g is negative.\n",
                    model->BSIM4v5pditsl);
            printf("Fatal: pditsl = %g is negative.\n", model->BSIM4v5pditsl);
            Fatal_Flag = 1;
        }

        /* Check gate current parameters */
      if (model->BSIM4v5igbMod) {
        if (pParam->BSIM4v5nigbinv <= 0.0)
        {   fprintf(fplog, "Fatal: nigbinv = %g is non-positive.\n",
                    pParam->BSIM4v5nigbinv);
            printf("Fatal: nigbinv = %g is non-positive.\n", pParam->BSIM4v5nigbinv);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v5nigbacc <= 0.0)
        {   fprintf(fplog, "Fatal: nigbacc = %g is non-positive.\n",
                    pParam->BSIM4v5nigbacc);
            printf("Fatal: nigbacc = %g is non-positive.\n", pParam->BSIM4v5nigbacc);
            Fatal_Flag = 1;
        }
      }
      if (model->BSIM4v5igcMod) {
        if (pParam->BSIM4v5nigc <= 0.0)
        {   fprintf(fplog, "Fatal: nigc = %g is non-positive.\n",
                    pParam->BSIM4v5nigc);
            printf("Fatal: nigc = %g is non-positive.\n", pParam->BSIM4v5nigc);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v5poxedge <= 0.0)
        {   fprintf(fplog, "Fatal: poxedge = %g is non-positive.\n",
                    pParam->BSIM4v5poxedge);
            printf("Fatal: poxedge = %g is non-positive.\n", pParam->BSIM4v5poxedge);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v5pigcd <= 0.0)
        {   fprintf(fplog, "Fatal: pigcd = %g is non-positive.\n",
                    pParam->BSIM4v5pigcd);
            printf("Fatal: pigcd = %g is non-positive.\n", pParam->BSIM4v5pigcd);
            Fatal_Flag = 1;
        }
      }

        /* Check capacitance parameters */
        if (pParam->BSIM4v5clc < 0.0)
	{   fprintf(fplog, "Fatal: Clc = %g is negative.\n", pParam->BSIM4v5clc);
	    printf("Fatal: Clc = %g is negative.\n", pParam->BSIM4v5clc);
	    Fatal_Flag = 1;
        }      

        /* Check overlap capacitance parameters */
        if (pParam->BSIM4v5ckappas < 0.02)
        {   fprintf(fplog, "Warning: ckappas = %g is too small. Set to 0.02\n",
                    pParam->BSIM4v5ckappas);
            printf("Warning: ckappas = %g is too small.\n", pParam->BSIM4v5ckappas);
            pParam->BSIM4v5ckappas = 0.02;
       }
        if (pParam->BSIM4v5ckappad < 0.02)
        {   fprintf(fplog, "Warning: ckappad = %g is too small. Set to 0.02\n",
                    pParam->BSIM4v5ckappad);
            printf("Warning: ckappad = %g is too small.\n", pParam->BSIM4v5ckappad);
            pParam->BSIM4v5ckappad = 0.02;
        }

	if (model->BSIM4v5vtss < 0.0)
	{   fprintf(fplog, "Fatal: Vtss = %g is negative.\n",
			model->BSIM4v5vtss);
	    printf("Fatal: Vtss = %g is negative.\n",
			model->BSIM4v5vtss);
	    Fatal_Flag = 1;
	}
	if (model->BSIM4v5vtsd < 0.0)
	{   fprintf(fplog, "Fatal: Vtsd = %g is negative.\n",
			model->BSIM4v5vtsd);
	    printf("Fatal: Vtsd = %g is negative.\n",
		    model->BSIM4v5vtsd);
	    Fatal_Flag = 1;
	}
	if (model->BSIM4v5vtssws < 0.0)
	{   fprintf(fplog, "Fatal: Vtssws = %g is negative.\n",
		    model->BSIM4v5vtssws);
	    printf("Fatal: Vtssws = %g is negative.\n",
		    model->BSIM4v5vtssws);
	    Fatal_Flag = 1;
	}
	if (model->BSIM4v5vtsswd < 0.0)
	{   fprintf(fplog, "Fatal: Vtsswd = %g is negative.\n",
		    model->BSIM4v5vtsswd);
	    printf("Fatal: Vtsswd = %g is negative.\n",
		    model->BSIM4v5vtsswd);
	    Fatal_Flag = 1;
	}
	if (model->BSIM4v5vtsswgs < 0.0)
	{   fprintf(fplog, "Fatal: Vtsswgs = %g is negative.\n",
		    model->BSIM4v5vtsswgs);
	    printf("Fatal: Vtsswgs = %g is negative.\n",
		    model->BSIM4v5vtsswgs);
	    Fatal_Flag = 1;
	}
	if (model->BSIM4v5vtsswgd < 0.0)
	{   fprintf(fplog, "Fatal: Vtsswgd = %g is negative.\n",
		    model->BSIM4v5vtsswgd);
	    printf("Fatal: Vtsswgd = %g is negative.\n",
			model->BSIM4v5vtsswgd);
	    Fatal_Flag = 1;
	}


      if (model->BSIM4v5paramChk ==1)
      {
/* Check L and W parameters */ 
	if (pParam->BSIM4v5leff <= 1.0e-9)
	{   fprintf(fplog, "Warning: Leff = %g <= 1.0e-9. Recommended Leff >= 1e-8 \n", 
		    pParam->BSIM4v5leff);
	    printf("Warning: Leff = %g <= 1.0e-9. Recommended Leff >= 1e-8 \n",
		    pParam->BSIM4v5leff);
	}    
	
	if (pParam->BSIM4v5leffCV <= 1.0e-9)
	{   fprintf(fplog, "Warning: Leff for CV = %g <= 1.0e-9. Recommended LeffCV >=1e-8 \n",
		    pParam->BSIM4v5leffCV);
	    printf("Warning: Leff for CV = %g <= 1.0e-9. Recommended LeffCV >=1e-8 \n",
		    pParam->BSIM4v5leffCV);
	}  
	
        if (pParam->BSIM4v5weff <= 1.0e-9)
	{   fprintf(fplog, "Warning: Weff = %g <= 1.0e-9. Recommended Weff >=1e-7 \n",
		    pParam->BSIM4v5weff);
	    printf("Warning: Weff = %g <= 1.0e-9. Recommended Weff >=1e-7 \n",
		   pParam->BSIM4v5weff);
	}             
	
	if (pParam->BSIM4v5weffCV <= 1.0e-9)
	{   fprintf(fplog, "Warning: Weff for CV = %g <= 1.0e-9. Recommended WeffCV >= 1e-7 \n",
		    pParam->BSIM4v5weffCV);
	    printf("Warning: Weff for CV = %g <= 1.0e-9. Recommended WeffCV >= 1e-7 \n",
		    pParam->BSIM4v5weffCV);
	}        
	
        /* Check threshold voltage parameters */
	if (model->BSIM4v5toxe < 1.0e-10)
	{   fprintf(fplog, "Warning: Toxe = %g is less than 1A. Recommended Toxe >= 5A\n",
	            model->BSIM4v5toxe);
	    printf("Warning: Toxe = %g is less than 1A. Recommended Toxe >= 5A\n", model->BSIM4v5toxe);
        }
        if (model->BSIM4v5toxp < 1.0e-10)
        {   fprintf(fplog, "Warning: Toxp = %g is less than 1A. Recommended Toxp >= 5A\n",
                    model->BSIM4v5toxp);
            printf("Warning: Toxp = %g is less than 1A. Recommended Toxp >= 5A\n", model->BSIM4v5toxp);
        }
        if (model->BSIM4v5toxm < 1.0e-10)
        {   fprintf(fplog, "Warning: Toxm = %g is less than 1A. Recommended Toxm >= 5A\n",
                    model->BSIM4v5toxm);
            printf("Warning: Toxm = %g is less than 1A. Recommended Toxm >= 5A\n", model->BSIM4v5toxm);
        }

        if (pParam->BSIM4v5ndep <= 1.0e12)
	{   fprintf(fplog, "Warning: Ndep = %g may be too small.\n",
	            pParam->BSIM4v5ndep);
	    printf("Warning: Ndep = %g may be too small.\n",
	           pParam->BSIM4v5ndep);
	}
	else if (pParam->BSIM4v5ndep >= 1.0e21)
	{   fprintf(fplog, "Warning: Ndep = %g may be too large.\n",
	            pParam->BSIM4v5ndep);
	    printf("Warning: Ndep = %g may be too large.\n",
	           pParam->BSIM4v5ndep);
	}

	 if (pParam->BSIM4v5nsub <= 1.0e14)
	{   fprintf(fplog, "Warning: Nsub = %g may be too small.\n",
	            pParam->BSIM4v5nsub);
	    printf("Warning: Nsub = %g may be too small.\n",
	           pParam->BSIM4v5nsub);
	}
	else if (pParam->BSIM4v5nsub >= 1.0e21)
	{   fprintf(fplog, "Warning: Nsub = %g may be too large.\n",
	            pParam->BSIM4v5nsub);
	    printf("Warning: Nsub = %g may be too large.\n",
	           pParam->BSIM4v5nsub);
	}

	if ((pParam->BSIM4v5ngate > 0.0) &&
	    (pParam->BSIM4v5ngate <= 1.e18))
	{   fprintf(fplog, "Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	            pParam->BSIM4v5ngate);
	    printf("Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	           pParam->BSIM4v5ngate);
	}
       
        if (pParam->BSIM4v5dvt0 < 0.0)
	{   fprintf(fplog, "Warning: Dvt0 = %g is negative.\n",
		    pParam->BSIM4v5dvt0);   
	    printf("Warning: Dvt0 = %g is negative.\n", pParam->BSIM4v5dvt0);   
	}
	    
	if (fabs(1.0e-8 / (pParam->BSIM4v5w0 + pParam->BSIM4v5weff)) > 10.0)
	{   fprintf(fplog, "Warning: (W0 + Weff) may be too small.\n");
	    printf("Warning: (W0 + Weff) may be too small.\n");
        }

/* Check subthreshold parameters */
	if (pParam->BSIM4v5nfactor < 0.0)
	{   fprintf(fplog, "Warning: Nfactor = %g is negative.\n",
		    pParam->BSIM4v5nfactor);
	    printf("Warning: Nfactor = %g is negative.\n", pParam->BSIM4v5nfactor);
	}
	if (pParam->BSIM4v5cdsc < 0.0)
	{   fprintf(fplog, "Warning: Cdsc = %g is negative.\n",
		    pParam->BSIM4v5cdsc);
	    printf("Warning: Cdsc = %g is negative.\n", pParam->BSIM4v5cdsc);
	}
	if (pParam->BSIM4v5cdscd < 0.0)
	{   fprintf(fplog, "Warning: Cdscd = %g is negative.\n",
		    pParam->BSIM4v5cdscd);
	    printf("Warning: Cdscd = %g is negative.\n", pParam->BSIM4v5cdscd);
	}
/* Check DIBL parameters */
	if (here->BSIM4v5eta0 < 0.0)
	{   fprintf(fplog, "Warning: Eta0 = %g is negative.\n",
		    here->BSIM4v5eta0); 
	    printf("Warning: Eta0 = %g is negative.\n", here->BSIM4v5eta0); 
	}
	      
/* Check Abulk parameters */	    
	 if (fabs(1.0e-8 / (pParam->BSIM4v5b1 + pParam->BSIM4v5weff)) > 10.0)
       	{   fprintf(fplog, "Warning: (B1 + Weff) may be too small.\n");
       	    printf("Warning: (B1 + Weff) may be too small.\n");
        }    
    

/* Check Saturation parameters */
     	if (pParam->BSIM4v5a2 < 0.01)
	{   fprintf(fplog, "Warning: A2 = %g is too small. Set to 0.01.\n", pParam->BSIM4v5a2);
	    printf("Warning: A2 = %g is too small. Set to 0.01.\n",
		   pParam->BSIM4v5a2);
	    pParam->BSIM4v5a2 = 0.01;
	}
	else if (pParam->BSIM4v5a2 > 1.0)
	{   fprintf(fplog, "Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		    pParam->BSIM4v5a2);
	    printf("Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		   pParam->BSIM4v5a2);
	    pParam->BSIM4v5a2 = 1.0;
	    pParam->BSIM4v5a1 = 0.0;
	}

        if (pParam->BSIM4v5prwg < 0.0)
        {   fprintf(fplog, "Warning: Prwg = %g is negative. Set to zero.\n",
                    pParam->BSIM4v5prwg);
            printf("Warning: Prwg = %g is negative. Set to zero.\n",
                   pParam->BSIM4v5prwg);
            pParam->BSIM4v5prwg = 0.0;
        }

	if (pParam->BSIM4v5rdsw < 0.0)
	{   fprintf(fplog, "Warning: Rdsw = %g is negative. Set to zero.\n",
		    pParam->BSIM4v5rdsw);
	    printf("Warning: Rdsw = %g is negative. Set to zero.\n",
		   pParam->BSIM4v5rdsw);
	    pParam->BSIM4v5rdsw = 0.0;
	    pParam->BSIM4v5rds0 = 0.0;
	}

	if (pParam->BSIM4v5rds0 < 0.0)
	{   fprintf(fplog, "Warning: Rds at current temperature = %g is negative. Set to zero.\n",
		    pParam->BSIM4v5rds0);
	    printf("Warning: Rds at current temperature = %g is negative. Set to zero.\n",
		   pParam->BSIM4v5rds0);
	    pParam->BSIM4v5rds0 = 0.0;
	}

        if (pParam->BSIM4v5rdswmin < 0.0)
        {   fprintf(fplog, "Warning: Rdswmin at current temperature = %g is negative. Set to zero.\n",
                    pParam->BSIM4v5rdswmin);
            printf("Warning: Rdswmin at current temperature = %g is negative. Set to zero.\n",
                   pParam->BSIM4v5rdswmin);
            pParam->BSIM4v5rdswmin = 0.0;
        }

        if (pParam->BSIM4v5pscbe2 <= 0.0)
        {   fprintf(fplog, "Warning: Pscbe2 = %g is not positive.\n",
                    pParam->BSIM4v5pscbe2);
            printf("Warning: Pscbe2 = %g is not positive.\n", pParam->BSIM4v5pscbe2);
        }

        if (pParam->BSIM4v5vsattemp < 1.0e3)
	{   fprintf(fplog, "Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM4v5vsattemp);
	   printf("Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM4v5vsattemp);
	}

      if((model->BSIM4v5lambdaGiven) && (pParam->BSIM4v5lambda > 0.0) ) 
      {
        if (pParam->BSIM4v5lambda > 1.0e-9)
	{   fprintf(fplog, "Warning: Lambda = %g may be too large.\n", pParam->BSIM4v5lambda);
	   printf("Warning: Lambda = %g may be too large.\n", pParam->BSIM4v5lambda);
	}
      }

      if((model->BSIM4v5vtlGiven) && (pParam->BSIM4v5vtl > 0.0) )
      {  
        if (pParam->BSIM4v5vtl < 6.0e4)
	{   fprintf(fplog, "Warning: Thermal velocity vtl = %g may be too small.\n", pParam->BSIM4v5vtl);
	   printf("Warning: Thermal velocity vtl = %g may be too small.\n", pParam->BSIM4v5vtl);
	}

        if (pParam->BSIM4v5xn < 3.0)
	{   fprintf(fplog, "Warning: back scattering coeff xn = %g is too small.\n", pParam->BSIM4v5xn);
	    printf("Warning: back scattering coeff xn = %g is too small. Reset to 3.0 \n", pParam->BSIM4v5xn);
	    pParam->BSIM4v5xn = 3.0;
	}

        if (model->BSIM4v5lc < 0.0)
	{   fprintf(fplog, "Warning: back scattering coeff lc = %g is too small.\n", model->BSIM4v5lc);
	    printf("Warning: back scattering coeff lc = %g is too small. Reset to 0.0\n", model->BSIM4v5lc);
	    pParam->BSIM4v5lc = 0.0;
	}
      }

	if (pParam->BSIM4v5pdibl1 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl1 = %g is negative.\n",
		    pParam->BSIM4v5pdibl1);
	    printf("Warning: Pdibl1 = %g is negative.\n", pParam->BSIM4v5pdibl1);
	}
	if (pParam->BSIM4v5pdibl2 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl2 = %g is negative.\n",
		    pParam->BSIM4v5pdibl2);
	    printf("Warning: Pdibl2 = %g is negative.\n", pParam->BSIM4v5pdibl2);
	}

/* Check stress effect parameters */        
        if((here->BSIM4v5sa > 0.0) && (here->BSIM4v5sb > 0.0) && 
       	((here->BSIM4v5nf == 1.0) || ((here->BSIM4v5nf > 1.0) && (here->BSIM4v5sd > 0.0))) )
        {   if (model->BSIM4v5lodk2 <= 0.0)
            {   fprintf(fplog, "Warning: LODK2 = %g is not positive.\n",model->BSIM4v5lodk2);
                printf("Warning: LODK2 = %g is not positive.\n",model->BSIM4v5lodk2);
            }
            if (model->BSIM4v5lodeta0 <= 0.0)
            {   fprintf(fplog, "Warning: LODETA0 = %g is not positive.\n",model->BSIM4v5lodeta0);
            	printf("Warning: LODETA0 = %g is not positive.\n",model->BSIM4v5lodeta0);
            }
 	}

/* Check gate resistance parameters */        
        if (here->BSIM4v5rgateMod == 1)
        {   if (model->BSIM4v5rshg <= 0.0)
	        printf("Warning: rshg should be positive for rgateMod = 1.\n");
	}
        else if (here->BSIM4v5rgateMod == 2)
        {   if (model->BSIM4v5rshg <= 0.0)
                printf("Warning: rshg <= 0.0 for rgateMod = 2.\n");
            else if (pParam->BSIM4v5xrcrg1 <= 0.0)
                     printf("Warning: xrcrg1 <= 0.0 for rgateMod = 2.\n");
         }
        if (here->BSIM4v5rgateMod == 3)
        {   if (model->BSIM4v5rshg <= 0.0)
                printf("Warning: rshg should be positive for rgateMod = 3.\n");
            else if (pParam->BSIM4v5xrcrg1 <= 0.0)
                     printf("Warning: xrcrg1 should be positive for rgateMod = 3.\n");
         }
         
/* Check capacitance parameters */
        if (pParam->BSIM4v5noff < 0.1)
        {   fprintf(fplog, "Warning: Noff = %g is too small.\n",
                    pParam->BSIM4v5noff);
            printf("Warning: Noff = %g is too small.\n", pParam->BSIM4v5noff);
        }

        if (pParam->BSIM4v5voffcv < -0.5)
        {   fprintf(fplog, "Warning: Voffcv = %g is too small.\n",
                    pParam->BSIM4v5voffcv);
            printf("Warning: Voffcv = %g is too small.\n", pParam->BSIM4v5voffcv);
        }

        if (pParam->BSIM4v5moin < 5.0)
        {   fprintf(fplog, "Warning: Moin = %g is too small.\n",
                    pParam->BSIM4v5moin);
            printf("Warning: Moin = %g is too small.\n", pParam->BSIM4v5moin);
        }
        if (pParam->BSIM4v5moin > 25.0)
        {   fprintf(fplog, "Warning: Moin = %g is too large.\n",
                    pParam->BSIM4v5moin);
            printf("Warning: Moin = %g is too large.\n", pParam->BSIM4v5moin);
        }
	if(model->BSIM4v5capMod ==2) {
        	if (pParam->BSIM4v5acde < 0.1)
        	{   fprintf(fplog, "Warning:  Acde = %g is too small.\n",
                    	pParam->BSIM4v5acde);
            	printf("Warning: Acde = %g is too small.\n", pParam->BSIM4v5acde);
        	}
        	if (pParam->BSIM4v5acde > 1.6)
        	{   fprintf(fplog, "Warning:  Acde = %g is too large.\n",
                    	pParam->BSIM4v5acde);
            	printf("Warning: Acde = %g is too large.\n", pParam->BSIM4v5acde);
        	}
	}

/* Check overlap capacitance parameters */
        if (model->BSIM4v5cgdo < 0.0)
	{   fprintf(fplog, "Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM4v5cgdo);
	    printf("Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM4v5cgdo);
	    model->BSIM4v5cgdo = 0.0;
        }      
        if (model->BSIM4v5cgso < 0.0)
	{   fprintf(fplog, "Warning: cgso = %g is negative. Set to zero.\n", model->BSIM4v5cgso);
	    printf("Warning: cgso = %g is negative. Set to zero.\n", model->BSIM4v5cgso);
	    model->BSIM4v5cgso = 0.0;
        }      
        if (model->BSIM4v5cgbo < 0.0)
	{   fprintf(fplog, "Warning: cgbo = %g is negative. Set to zero.\n", model->BSIM4v5cgbo);
	    printf("Warning: cgbo = %g is negative. Set to zero.\n", model->BSIM4v5cgbo);
	    model->BSIM4v5cgbo = 0.0;
        }      
      if (model->BSIM4v5tnoiMod == 1) { 
        if (model->BSIM4v5tnoia < 0.0)
        {   fprintf(fplog, "Warning: tnoia = %g is negative. Set to zero.\n", model->BSIM4v5tnoia);
            printf("Warning: tnoia = %g is negative. Set to zero.\n", model->BSIM4v5tnoia);
            model->BSIM4v5tnoia = 0.0;
        }
        if (model->BSIM4v5tnoib < 0.0)
        {   fprintf(fplog, "Warning: tnoib = %g is negative. Set to zero.\n", model->BSIM4v5tnoib);
            printf("Warning: tnoib = %g is negative. Set to zero.\n", model->BSIM4v5tnoib);
            model->BSIM4v5tnoib = 0.0;
        }

         if (model->BSIM4v5rnoia < 0.0)
        {   fprintf(fplog, "Warning: rnoia = %g is negative. Set to zero.\n", model->BSIM4v5rnoia);
            printf("Warning: rnoia = %g is negative. Set to zero.\n", model->BSIM4v5rnoia);
            model->BSIM4v5rnoia = 0.0;
        }
        if (model->BSIM4v5rnoib < 0.0)
        {   fprintf(fplog, "Warning: rnoib = %g is negative. Set to zero.\n", model->BSIM4v5rnoib);
            printf("Warning: rnoib = %g is negative. Set to zero.\n", model->BSIM4v5rnoib);
            model->BSIM4v5rnoib = 0.0;
        }
      }

	if (model->BSIM4v5SjctEmissionCoeff < 0.0)
	{   fprintf(fplog, "Warning: Njs = %g is negative.\n",
		    model->BSIM4v5SjctEmissionCoeff);
	    printf("Warning: Njs = %g is negative.\n",
		    model->BSIM4v5SjctEmissionCoeff);
	}
	if (model->BSIM4v5DjctEmissionCoeff < 0.0)
	{   fprintf(fplog, "Warning: Njd = %g is negative.\n",
		    model->BSIM4v5DjctEmissionCoeff);
	    printf("Warning: Njd = %g is negative.\n",
		    model->BSIM4v5DjctEmissionCoeff);
	}
	if (model->BSIM4v5njtstemp < 0.0)
	{   fprintf(fplog, "Warning: Njts = %g is negative at temperature = %g.\n",
		    model->BSIM4v5njtstemp, ckt->CKTtemp);
	    printf("Warning: Njts = %g is negative at temperature = %g.\n",
		    model->BSIM4v5njtstemp, ckt->CKTtemp);
	}
	if (model->BSIM4v5njtsswtemp < 0.0)
	{   fprintf(fplog, "Warning: Njtssw = %g is negative at temperature = %g.\n",
		    model->BSIM4v5njtsswtemp, ckt->CKTtemp);
	    printf("Warning: Njtssw = %g is negative at temperature = %g.\n",
		    model->BSIM4v5njtsswtemp, ckt->CKTtemp);
	}
	if (model->BSIM4v5njtsswgtemp < 0.0)
	{   fprintf(fplog, "Warning: Njtsswg = %g is negative at temperature = %g.\n",
		    model->BSIM4v5njtsswgtemp, ckt->CKTtemp);
	    printf("Warning: Njtsswg = %g is negative at temperature = %g.\n",
		    model->BSIM4v5njtsswgtemp, ckt->CKTtemp);
	}
       if (model->BSIM4v5ntnoi < 0.0)
        {   fprintf(fplog, "Warning: ntnoi = %g is negative. Set to zero.\n", model->BSIM4v5ntnoi);
            printf("Warning: ntnoi = %g is negative. Set to zero.\n", model->BSIM4v5ntnoi);
            model->BSIM4v5ntnoi = 0.0;
        }

        /* diode model */
       if (model->BSIM4v5SbulkJctBotGradingCoeff >= 0.99)
        {   fprintf(fplog, "Warning: MJS = %g is too big. Set to 0.99.\n", model->BSIM4v5SbulkJctBotGradingCoeff);
            printf("Warning: MJS = %g is too big. Set to 0.99.\n", model->BSIM4v5SbulkJctBotGradingCoeff);
            model->BSIM4v5SbulkJctBotGradingCoeff = 0.99;
        }
       if (model->BSIM4v5SbulkJctSideGradingCoeff >= 0.99)
        {   fprintf(fplog, "Warning: MJSWS = %g is too big. Set to 0.99.\n", model->BSIM4v5SbulkJctSideGradingCoeff);
            printf("Warning: MJSWS = %g is too big. Set to 0.99.\n", model->BSIM4v5SbulkJctSideGradingCoeff);
            model->BSIM4v5SbulkJctSideGradingCoeff = 0.99;
        }
       if (model->BSIM4v5SbulkJctGateSideGradingCoeff >= 0.99)
        {   fprintf(fplog, "Warning: MJSWGS = %g is too big. Set to 0.99.\n", model->BSIM4v5SbulkJctGateSideGradingCoeff);
            printf("Warning: MJSWGS = %g is too big. Set to 0.99.\n", model->BSIM4v5SbulkJctGateSideGradingCoeff);
            model->BSIM4v5SbulkJctGateSideGradingCoeff = 0.99;
        }

       if (model->BSIM4v5DbulkJctBotGradingCoeff >= 0.99)
        {   fprintf(fplog, "Warning: MJD = %g is too big. Set to 0.99.\n", model->BSIM4v5DbulkJctBotGradingCoeff);
            printf("Warning: MJD = %g is too big. Set to 0.99.\n", model->BSIM4v5DbulkJctBotGradingCoeff);
            model->BSIM4v5DbulkJctBotGradingCoeff = 0.99;
        }
       if (model->BSIM4v5DbulkJctSideGradingCoeff >= 0.99)
        {   fprintf(fplog, "Warning: MJSWD = %g is too big. Set to 0.99.\n", model->BSIM4v5DbulkJctSideGradingCoeff);
            printf("Warning: MJSWD = %g is too big. Set to 0.99.\n", model->BSIM4v5DbulkJctSideGradingCoeff);
            model->BSIM4v5DbulkJctSideGradingCoeff = 0.99;
        }
       if (model->BSIM4v5DbulkJctGateSideGradingCoeff >= 0.99)
        {   fprintf(fplog, "Warning: MJSWGD = %g is too big. Set to 0.99.\n", model->BSIM4v5DbulkJctGateSideGradingCoeff);
            printf("Warning: MJSWGD = %g is too big. Set to 0.99.\n", model->BSIM4v5DbulkJctGateSideGradingCoeff);
            model->BSIM4v5DbulkJctGateSideGradingCoeff = 0.99;
        }
	if (model->BSIM4v5wpemod == 1)
	{
		if (model->BSIM4v5scref <= 0.0)
		{   fprintf(fplog, "Warning: SCREF = %g is not positive. Set to 1e-6.\n", model->BSIM4v5scref);
		    printf("Warning: SCREF = %g is not positive. Set to 1e-6.\n", model->BSIM4v5scref);
		    model->BSIM4v5scref = 1e-6;
		}
	        if (here->BSIM4v5sca < 0.0)
		{   fprintf(fplog, "Warning: SCA = %g is negative. Set to 0.0.\n", here->BSIM4v5sca);
                    printf("Warning: SCA = %g is negative. Set to 0.0.\n", here->BSIM4v5sca);
                    here->BSIM4v5sca = 0.0;
                }
                if (here->BSIM4v5scb < 0.0)
                {   fprintf(fplog, "Warning: SCB = %g is negative. Set to 0.0.\n", here->BSIM4v5scb);
                    printf("Warning: SCB = %g is negative. Set to 0.0.\n", here->BSIM4v5scb);
                    here->BSIM4v5scb = 0.0;
                }
                if (here->BSIM4v5scc < 0.0)
                {   fprintf(fplog, "Warning: SCC = %g is negative. Set to 0.0.\n", here->BSIM4v5scc);
                    printf("Warning: SCC = %g is negative. Set to 0.0.\n", here->BSIM4v5scc);
                    here->BSIM4v5scc = 0.0;
                }
                if (here->BSIM4v5sc < 0.0)
                {   fprintf(fplog, "Warning: SC = %g is negative. Set to 0.0.\n", here->BSIM4v5sc);
                    printf("Warning: SC = %g is negative. Set to 0.0.\n", here->BSIM4v5sc);
                    here->BSIM4v5sc = 0.0;
                }
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

