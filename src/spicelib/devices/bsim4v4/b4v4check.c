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

#include "ngspice.h"
#include "cktdefs.h"
#include "bsim4v4def.h"
#include "trandefs.h"
#include "const.h"
#include "sperror.h"
#include "devdefs.h"
#include "suffix.h"

int
BSIM4V4checkModel(model, here, ckt)
BSIM4V4model *model;
BSIM4V4instance *here;
CKTcircuit *ckt;
{
struct bsim4SizeDependParam *pParam;
int Fatal_Flag = 0;
FILE *fplog;
double model_version = atof(model->BSIM4V4version);
    
    if ((fplog = fopen("bsim4.out", "w")) != NULL)
    {   pParam = here->pParam;
        fprintf(fplog, "BSIM4V4: Berkeley Short Channel IGFET Model-4\n");
        fprintf(fplog, "Developed by Xuemei (Jane) Xi, Jin He, Mohan Dunga, Prof. Ali Niknejad and Prof. Chenming Hu in 2003.\n");
        fprintf(fplog, "\n");
	fprintf(fplog, "++++++++++ BSIM4V4 PARAMETER CHECKING BELOW ++++++++++\n");

	if ( !strstr( model->BSIM4V4version, "4.4" ) )
        {  fprintf(fplog, "Warning: This model is BSIM4V4.4.0; you specified a wrong version number '%s'.\n",model->BSIM4V4version);
           printf("Warning: This model is BSIM4V4.4.0; you specified a wrong version number '%s'.\n",model->BSIM4V4version);
        }
	fprintf(fplog, "Model = %s\n", model->BSIM4V4modName);


        if ((here->BSIM4V4rgateMod == 2) || (here->BSIM4V4rgateMod == 3))
        {   if ((here->BSIM4V4trnqsMod == 1) || (here->BSIM4V4acnqsMod == 1))
            {   fprintf(fplog, "Warning: You've selected both Rg and charge deficit NQS; select one only.\n");
                printf("Warning: You've selected both Rg and charge deficit NQS; select one only.\n");
            }
        }

	if (model->BSIM4V4toxe <= 0.0)
	{   fprintf(fplog, "Fatal: Toxe = %g is not positive.\n",
		    model->BSIM4V4toxe);
	    printf("Fatal: Toxe = %g is not positive.\n", model->BSIM4V4toxe);
	    Fatal_Flag = 1;
	}
        if (model->BSIM4V4toxp <= 0.0)
        {   fprintf(fplog, "Fatal: Toxp = %g is not positive.\n",
                    model->BSIM4V4toxp);
            printf("Fatal: Toxp = %g is not positive.\n", model->BSIM4V4toxp);
            Fatal_Flag = 1;
        }

        if (model->BSIM4V4toxm <= 0.0)
        {   fprintf(fplog, "Fatal: Toxm = %g is not positive.\n",
                    model->BSIM4V4toxm);
            printf("Fatal: Toxm = %g is not positive.\n", model->BSIM4V4toxm);
            Fatal_Flag = 1;
        }

        if (model->BSIM4V4toxref <= 0.0)
        {   fprintf(fplog, "Fatal: Toxref = %g is not positive.\n",
                    model->BSIM4V4toxref);
            printf("Fatal: Toxref = %g is not positive.\n", model->BSIM4V4toxref);
            Fatal_Flag = 1;
        }

        if (pParam->BSIM4V4lpe0 < -pParam->BSIM4V4leff)
        {   fprintf(fplog, "Fatal: Lpe0 = %g is less than -Leff.\n",
                    pParam->BSIM4V4lpe0);
            printf("Fatal: Lpe0 = %g is less than -Leff.\n",
                        pParam->BSIM4V4lpe0);
            Fatal_Flag = 1;
        }
        if (model->BSIM4V4lintnoi > pParam->BSIM4V4leff/2)
        {   fprintf(fplog, "Fatal: Lintnoi = %g is too large - Leff for noise is negative.\n",
                    model->BSIM4V4lintnoi);
            printf("Fatal: Lintnoi = %g is too large - Leff for noise is negative.\n",
                    model->BSIM4V4lintnoi);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4V4lpeb < -pParam->BSIM4V4leff)
        {   fprintf(fplog, "Fatal: Lpeb = %g is less than -Leff.\n",
                    pParam->BSIM4V4lpeb);
            printf("Fatal: Lpeb = %g is less than -Leff.\n",
                        pParam->BSIM4V4lpeb);
            Fatal_Flag = 1;
        }
	if (pParam->BSIM4V4ndep <= 0.0)
	{   fprintf(fplog, "Fatal: Ndep = %g is not positive.\n",
		    pParam->BSIM4V4ndep);
	    printf("Fatal: Ndep = %g is not positive.\n",
		   pParam->BSIM4V4ndep);
	    Fatal_Flag = 1;
	}
        if (pParam->BSIM4V4phi <= 0.0)
        {   fprintf(fplog, "Fatal: Phi = %g is not positive. Please check Phin and Ndep\n",
                    pParam->BSIM4V4phi);
            fprintf(fplog, "	   Phin = %g  Ndep = %g \n", 
            	    pParam->BSIM4V4phin, pParam->BSIM4V4ndep);
            printf("Fatal: Phi = %g is not positive. Please check Phin and Ndep\n",
                    pParam->BSIM4V4phi);
            printf("	   Phin = %g  Ndep = %g \n", 
            	    pParam->BSIM4V4phin, pParam->BSIM4V4ndep);
            Fatal_Flag = 1;
        }
	if (pParam->BSIM4V4nsub <= 0.0)
	{   fprintf(fplog, "Fatal: Nsub = %g is not positive.\n",
		    pParam->BSIM4V4nsub);
	    printf("Fatal: Nsub = %g is not positive.\n",
		   pParam->BSIM4V4nsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4V4ngate < 0.0)
	{   fprintf(fplog, "Fatal: Ngate = %g is not positive.\n",
		    pParam->BSIM4V4ngate);
	    printf("Fatal: Ngate = %g Ngate is not positive.\n",
		   pParam->BSIM4V4ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4V4ngate > 1.e25)
	{   fprintf(fplog, "Fatal: Ngate = %g is too high.\n",
		    pParam->BSIM4V4ngate);
	    printf("Fatal: Ngate = %g Ngate is too high\n",
		   pParam->BSIM4V4ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4V4xj <= 0.0)
	{   fprintf(fplog, "Fatal: Xj = %g is not positive.\n",
		    pParam->BSIM4V4xj);
	    printf("Fatal: Xj = %g is not positive.\n", pParam->BSIM4V4xj);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4V4dvt1 < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1 = %g is negative.\n",
		    pParam->BSIM4V4dvt1);   
	    printf("Fatal: Dvt1 = %g is negative.\n", pParam->BSIM4V4dvt1);   
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM4V4dvt1w < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1w = %g is negative.\n",
		    pParam->BSIM4V4dvt1w);
	    printf("Fatal: Dvt1w = %g is negative.\n", pParam->BSIM4V4dvt1w);
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM4V4w0 == -pParam->BSIM4V4weff)
	{   fprintf(fplog, "Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }   

	if (pParam->BSIM4V4dsub < 0.0)
	{   fprintf(fplog, "Fatal: Dsub = %g is negative.\n", pParam->BSIM4V4dsub);
	    printf("Fatal: Dsub = %g is negative.\n", pParam->BSIM4V4dsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4V4b1 == -pParam->BSIM4V4weff)
	{   fprintf(fplog, "Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }  
        if (here->BSIM4V4u0temp <= 0.0)
	{   fprintf(fplog, "Fatal: u0 at current temperature = %g is not positive.\n", here->BSIM4V4u0temp);
	    printf("Fatal: u0 at current temperature = %g is not positive.\n",
		   here->BSIM4V4u0temp);
	    Fatal_Flag = 1;
        }
    
        if (pParam->BSIM4V4delta < 0.0)
	{   fprintf(fplog, "Fatal: Delta = %g is less than zero.\n",
		    pParam->BSIM4V4delta);
	    printf("Fatal: Delta = %g is less than zero.\n", pParam->BSIM4V4delta);
	    Fatal_Flag = 1;
        }      

	if (here->BSIM4V4vsattemp <= 0.0)
	{   fprintf(fplog, "Fatal: Vsat at current temperature = %g is not positive.\n", here->BSIM4V4vsattemp);
	    printf("Fatal: Vsat at current temperature = %g is not positive.\n",
		   here->BSIM4V4vsattemp);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4V4pclm <= 0.0)
	{   fprintf(fplog, "Fatal: Pclm = %g is not positive.\n", pParam->BSIM4V4pclm);
	    printf("Fatal: Pclm = %g is not positive.\n", pParam->BSIM4V4pclm);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4V4drout < 0.0)
	{   fprintf(fplog, "Fatal: Drout = %g is negative.\n", pParam->BSIM4V4drout);
	    printf("Fatal: Drout = %g is negative.\n", pParam->BSIM4V4drout);
	    Fatal_Flag = 1;
	}

        if (here->BSIM4V4m <= 0.0)
        {   fprintf(fplog, "Fatal: multiplier = %g is not positive.\n", here->BSIM4V4m);
            printf("Fatal: multiplier = %g is not positive.\n", here->BSIM4V4m);
            Fatal_Flag = 1;
        }
        if (here->BSIM4V4nf < 1.0)
        {   fprintf(fplog, "Fatal: Number of finger = %g is smaller than one.\n", here->BSIM4V4nf);
            printf("Fatal: Number of finger = %g is smaller than one.\n", here->BSIM4V4nf);
            Fatal_Flag = 1;
        }

        if((here->BSIM4V4sa > 0.0) && (here->BSIM4V4sb > 0.0) && 
       	((here->BSIM4V4nf == 1.0) || ((here->BSIM4V4nf > 1.0) && (here->BSIM4V4sd > 0.0))) )
        {   if (model->BSIM4V4saref <= 0.0)
            {   fprintf(fplog, "Fatal: SAref = %g is not positive.\n",model->BSIM4V4saref);
             	printf("Fatal: SAref = %g is not positive.\n",model->BSIM4V4saref);
             	Fatal_Flag = 1;
            }
            if (model->BSIM4V4sbref <= 0.0)
            {   fprintf(fplog, "Fatal: SBref = %g is not positive.\n",model->BSIM4V4sbref);
            	printf("Fatal: SBref = %g is not positive.\n",model->BSIM4V4sbref);
            	Fatal_Flag = 1;
            }
 	}

        if ((here->BSIM4V4l + model->BSIM4V4xl) <= model->BSIM4V4xgl)
        {   fprintf(fplog, "Fatal: The parameter xgl must be smaller than Ldrawn+XL.\n");
            printf("Fatal: The parameter xgl must be smaller than Ldrawn+XL.\n");
            Fatal_Flag = 1;
        }
        if (model->BSIM4V4ngcon < 1.0)
        {   fprintf(fplog, "Fatal: The parameter ngcon cannot be smaller than one.\n");
            printf("Fatal: The parameter ngcon cannot be smaller than one.\n");
            Fatal_Flag = 1;
        }
        if ((model->BSIM4V4ngcon != 1.0) && (model->BSIM4V4ngcon != 2.0))
        {   model->BSIM4V4ngcon = 1.0;
            fprintf(fplog, "Warning: Ngcon must be equal to one or two; reset to 1.0.\n");
            printf("Warning: Ngcon must be equal to one or two; reset to 1.0.\n");
        }

        if (model->BSIM4V4gbmin < 1.0e-20)
        {   fprintf(fplog, "Warning: Gbmin = %g is too small.\n",
                    model->BSIM4V4gbmin);
            printf("Warning: Gbmin = %g is too small.\n", model->BSIM4V4gbmin);
        }

        /* Check saturation parameters */
        if (pParam->BSIM4V4fprout < 0.0)
        {   fprintf(fplog, "Fatal: fprout = %g is negative.\n",
                    pParam->BSIM4V4fprout);
            printf("Fatal: fprout = %g is negative.\n", pParam->BSIM4V4fprout);
	    Fatal_Flag = 1;
        }
        if (pParam->BSIM4V4pdits < 0.0)
        {   fprintf(fplog, "Fatal: pdits = %g is negative.\n",
                    pParam->BSIM4V4pdits);
            printf("Fatal: pdits = %g is negative.\n", pParam->BSIM4V4pdits);
            Fatal_Flag = 1;
        }
        if (model->BSIM4V4pditsl < 0.0)
        {   fprintf(fplog, "Fatal: pditsl = %g is negative.\n",
                    model->BSIM4V4pditsl);
            printf("Fatal: pditsl = %g is negative.\n", model->BSIM4V4pditsl);
            Fatal_Flag = 1;
        }

        /* Check gate current parameters */
      if (model->BSIM4V4igbMod) {
        if (pParam->BSIM4V4nigbinv <= 0.0)
        {   fprintf(fplog, "Fatal: nigbinv = %g is non-positive.\n",
                    pParam->BSIM4V4nigbinv);
            printf("Fatal: nigbinv = %g is non-positive.\n", pParam->BSIM4V4nigbinv);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4V4nigbacc <= 0.0)
        {   fprintf(fplog, "Fatal: nigbacc = %g is non-positive.\n",
                    pParam->BSIM4V4nigbacc);
            printf("Fatal: nigbacc = %g is non-positive.\n", pParam->BSIM4V4nigbacc);
            Fatal_Flag = 1;
        }
      }
      if (model->BSIM4V4igcMod) {
        if (pParam->BSIM4V4nigc <= 0.0)
        {   fprintf(fplog, "Fatal: nigc = %g is non-positive.\n",
                    pParam->BSIM4V4nigc);
            printf("Fatal: nigc = %g is non-positive.\n", pParam->BSIM4V4nigc);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4V4poxedge <= 0.0)
        {   fprintf(fplog, "Fatal: poxedge = %g is non-positive.\n",
                    pParam->BSIM4V4poxedge);
            printf("Fatal: poxedge = %g is non-positive.\n", pParam->BSIM4V4poxedge);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4V4pigcd <= 0.0)
        {   fprintf(fplog, "Fatal: pigcd = %g is non-positive.\n",
                    pParam->BSIM4V4pigcd);
            printf("Fatal: pigcd = %g is non-positive.\n", pParam->BSIM4V4pigcd);
            Fatal_Flag = 1;
        }
      }

        /* Check capacitance parameters */
        if (pParam->BSIM4V4clc < 0.0)
	{   fprintf(fplog, "Fatal: Clc = %g is negative.\n", pParam->BSIM4V4clc);
	    printf("Fatal: Clc = %g is negative.\n", pParam->BSIM4V4clc);
	    Fatal_Flag = 1;
        }      

        /* Check overlap capacitance parameters */
        if (pParam->BSIM4V4ckappas < 0.02)
        {   fprintf(fplog, "Warning: ckappas = %g is too small. Set to 0.02\n",
                    pParam->BSIM4V4ckappas);
            printf("Warning: ckappas = %g is too small.\n", pParam->BSIM4V4ckappas);
            pParam->BSIM4V4ckappas = 0.02;
       }
        if (pParam->BSIM4V4ckappad < 0.02)
        {   fprintf(fplog, "Warning: ckappad = %g is too small. Set to 0.02\n",
                    pParam->BSIM4V4ckappad);
            printf("Warning: ckappad = %g is too small.\n", pParam->BSIM4V4ckappad);
            pParam->BSIM4V4ckappad = 0.02;
        }

      if (model->BSIM4V4paramChk ==1)
      {
/* Check L and W parameters */ 
	if (pParam->BSIM4V4leff <= 1.0e-9)
	{   fprintf(fplog, "Warning: Leff = %g <= 1.0e-9. Recommended Leff >= 1e-8 \n", 
		    pParam->BSIM4V4leff);
	    printf("Warning: Leff = %g <= 1.0e-9. Recommended Leff >= 1e-8 \n",
		    pParam->BSIM4V4leff);
	}    
	
	if (pParam->BSIM4V4leffCV <= 1.0e-9)
	{   fprintf(fplog, "Warning: Leff for CV = %g <= 1.0e-9. Recommended LeffCV >=1e-8 \n",
		    pParam->BSIM4V4leffCV);
	    printf("Warning: Leff for CV = %g <= 1.0e-9. Recommended LeffCV >=1e-8 \n",
		    pParam->BSIM4V4leffCV);
	}  
	
        if (pParam->BSIM4V4weff <= 1.0e-9)
	{   fprintf(fplog, "Warning: Weff = %g <= 1.0e-9. Recommended Weff >=1e-7 \n",
		    pParam->BSIM4V4weff);
	    printf("Warning: Weff = %g <= 1.0e-9. Recommended Weff >=1e-7 \n",
		   pParam->BSIM4V4weff);
	}             
	
	if (pParam->BSIM4V4weffCV <= 1.0e-9)
	{   fprintf(fplog, "Warning: Weff for CV = %g <= 1.0e-9. Recommended WeffCV >= 1e-7 \n",
		    pParam->BSIM4V4weffCV);
	    printf("Warning: Weff for CV = %g <= 1.0e-9. Recommended WeffCV >= 1e-7 \n",
		    pParam->BSIM4V4weffCV);
	}        
	
        /* Check threshold voltage parameters */
	if (model->BSIM4V4toxe < 1.0e-10)
	{   fprintf(fplog, "Warning: Toxe = %g is less than 1A. Recommended Toxe >= 5A\n",
	            model->BSIM4V4toxe);
	    printf("Warning: Toxe = %g is less than 1A. Recommended Toxe >= 5A\n", model->BSIM4V4toxe);
        }
        if (model->BSIM4V4toxp < 1.0e-10)
        {   fprintf(fplog, "Warning: Toxp = %g is less than 1A. Recommended Toxp >= 5A\n",
                    model->BSIM4V4toxp);
            printf("Warning: Toxp = %g is less than 1A. Recommended Toxp >= 5A\n", model->BSIM4V4toxp);
        }
        if (model->BSIM4V4toxm < 1.0e-10)
        {   fprintf(fplog, "Warning: Toxm = %g is less than 1A. Recommended Toxm >= 5A\n",
                    model->BSIM4V4toxm);
            printf("Warning: Toxm = %g is less than 1A. Recommended Toxm >= 5A\n", model->BSIM4V4toxm);
        }

        if (pParam->BSIM4V4ndep <= 1.0e12)
	{   fprintf(fplog, "Warning: Ndep = %g may be too small.\n",
	            pParam->BSIM4V4ndep);
	    printf("Warning: Ndep = %g may be too small.\n",
	           pParam->BSIM4V4ndep);
	}
	else if (pParam->BSIM4V4ndep >= 1.0e21)
	{   fprintf(fplog, "Warning: Ndep = %g may be too large.\n",
	            pParam->BSIM4V4ndep);
	    printf("Warning: Ndep = %g may be too large.\n",
	           pParam->BSIM4V4ndep);
	}

	 if (pParam->BSIM4V4nsub <= 1.0e14)
	{   fprintf(fplog, "Warning: Nsub = %g may be too small.\n",
	            pParam->BSIM4V4nsub);
	    printf("Warning: Nsub = %g may be too small.\n",
	           pParam->BSIM4V4nsub);
	}
	else if (pParam->BSIM4V4nsub >= 1.0e21)
	{   fprintf(fplog, "Warning: Nsub = %g may be too large.\n",
	            pParam->BSIM4V4nsub);
	    printf("Warning: Nsub = %g may be too large.\n",
	           pParam->BSIM4V4nsub);
	}

	if ((pParam->BSIM4V4ngate > 0.0) &&
	    (pParam->BSIM4V4ngate <= 1.e18))
	{   fprintf(fplog, "Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	            pParam->BSIM4V4ngate);
	    printf("Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	           pParam->BSIM4V4ngate);
	}
       
        if (pParam->BSIM4V4dvt0 < 0.0)
	{   fprintf(fplog, "Warning: Dvt0 = %g is negative.\n",
		    pParam->BSIM4V4dvt0);   
	    printf("Warning: Dvt0 = %g is negative.\n", pParam->BSIM4V4dvt0);   
	}
	    
	if (fabs(1.0e-8 / (pParam->BSIM4V4w0 + pParam->BSIM4V4weff)) > 10.0)
	{   fprintf(fplog, "Warning: (W0 + Weff) may be too small.\n");
	    printf("Warning: (W0 + Weff) may be too small.\n");
        }

/* Check subthreshold parameters */
	if (pParam->BSIM4V4nfactor < 0.0)
	{   fprintf(fplog, "Warning: Nfactor = %g is negative.\n",
		    pParam->BSIM4V4nfactor);
	    printf("Warning: Nfactor = %g is negative.\n", pParam->BSIM4V4nfactor);
	}
	if (pParam->BSIM4V4cdsc < 0.0)
	{   fprintf(fplog, "Warning: Cdsc = %g is negative.\n",
		    pParam->BSIM4V4cdsc);
	    printf("Warning: Cdsc = %g is negative.\n", pParam->BSIM4V4cdsc);
	}
	if (pParam->BSIM4V4cdscd < 0.0)
	{   fprintf(fplog, "Warning: Cdscd = %g is negative.\n",
		    pParam->BSIM4V4cdscd);
	    printf("Warning: Cdscd = %g is negative.\n", pParam->BSIM4V4cdscd);
	}
/* Check DIBL parameters */
/*	if (here->BSIM4V4eta0 < 0.0)
	{   fprintf(fplog, "Warning: Eta0 = %g is negative.\n",
		    here->BSIM4V4eta0); 
	    printf("Warning: Eta0 = %g is negative.\n", here->BSIM4V4eta0); 
	}
*/
	      
/* Check Abulk parameters */	    
	 if (fabs(1.0e-8 / (pParam->BSIM4V4b1 + pParam->BSIM4V4weff)) > 10.0)
       	{   fprintf(fplog, "Warning: (B1 + Weff) may be too small.\n");
       	    printf("Warning: (B1 + Weff) may be too small.\n");
        }    
    

/* Check Saturation parameters */
     	if (pParam->BSIM4V4a2 < 0.01)
	{   fprintf(fplog, "Warning: A2 = %g is too small. Set to 0.01.\n", pParam->BSIM4V4a2);
	    printf("Warning: A2 = %g is too small. Set to 0.01.\n",
		   pParam->BSIM4V4a2);
	    pParam->BSIM4V4a2 = 0.01;
	}
	else if (pParam->BSIM4V4a2 > 1.0)
	{   fprintf(fplog, "Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		    pParam->BSIM4V4a2);
	    printf("Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		   pParam->BSIM4V4a2);
	    pParam->BSIM4V4a2 = 1.0;
	    pParam->BSIM4V4a1 = 0.0;
	}

        if (pParam->BSIM4V4prwg < 0.0)
        {   fprintf(fplog, "Warning: Prwg = %g is negative. Set to zero.\n",
                    pParam->BSIM4V4prwg);
            printf("Warning: Prwg = %g is negative. Set to zero.\n",
                   pParam->BSIM4V4prwg);
            pParam->BSIM4V4prwg = 0.0;
        }

	if (pParam->BSIM4V4rdsw < 0.0)
	{   fprintf(fplog, "Warning: Rdsw = %g is negative. Set to zero.\n",
		    pParam->BSIM4V4rdsw);
	    printf("Warning: Rdsw = %g is negative. Set to zero.\n",
		   pParam->BSIM4V4rdsw);
	    pParam->BSIM4V4rdsw = 0.0;
	    pParam->BSIM4V4rds0 = 0.0;
	}

	if (pParam->BSIM4V4rds0 < 0.0)
	{   fprintf(fplog, "Warning: Rds at current temperature = %g is negative. Set to zero.\n",
		    pParam->BSIM4V4rds0);
	    printf("Warning: Rds at current temperature = %g is negative. Set to zero.\n",
		   pParam->BSIM4V4rds0);
	    pParam->BSIM4V4rds0 = 0.0;
	}

        if (pParam->BSIM4V4rdswmin < 0.0)
        {   fprintf(fplog, "Warning: Rdswmin at current temperature = %g is negative. Set to zero.\n",
                    pParam->BSIM4V4rdswmin);
            printf("Warning: Rdswmin at current temperature = %g is negative. Set to zero.\n",
                   pParam->BSIM4V4rdswmin);
            pParam->BSIM4V4rdswmin = 0.0;
        }

        if (pParam->BSIM4V4pscbe2 <= 0.0)
        {   fprintf(fplog, "Warning: Pscbe2 = %g is not positive.\n",
                    pParam->BSIM4V4pscbe2);
            printf("Warning: Pscbe2 = %g is not positive.\n", pParam->BSIM4V4pscbe2);
        }

        if (pParam->BSIM4V4vsattemp < 1.0e3)
	{   fprintf(fplog, "Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM4V4vsattemp);
	   printf("Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM4V4vsattemp);
	}

      if((model->BSIM4V4lambdaGiven) && (pParam->BSIM4V4lambda > 0.0) ) 
      {
        if (pParam->BSIM4V4lambda > 1.0e-9)
	{   fprintf(fplog, "Warning: Lambda = %g may be too large.\n", pParam->BSIM4V4lambda);
	   printf("Warning: Lambda = %g may be too large.\n", pParam->BSIM4V4lambda);
	}
      }

      if((model->BSIM4V4vtlGiven) && (pParam->BSIM4V4vtl > 0.0) )
      {  
        if (pParam->BSIM4V4vtl < 6.0e4)
	{   fprintf(fplog, "Warning: Thermal velocity vtl = %g may be too small.\n", pParam->BSIM4V4vtl);
	   printf("Warning: Thermal velocity vtl = %g may be too small.\n", pParam->BSIM4V4vtl);
	}

        if (pParam->BSIM4V4xn < 3.0)
	{   fprintf(fplog, "Warning: back scattering coeff xn = %g is too small.\n", pParam->BSIM4V4xn);
	    printf("Warning: back scattering coeff xn = %g is too small. Reset to 3.0 \n", pParam->BSIM4V4xn);
	    pParam->BSIM4V4xn = 3.0;
	}

        if (model->BSIM4V4lc < 0.0)
	{   fprintf(fplog, "Warning: back scattering coeff lc = %g is too small.\n", model->BSIM4V4lc);
	    printf("Warning: back scattering coeff lc = %g is too small. Reset to 0.0\n", model->BSIM4V4lc);
	    pParam->BSIM4V4lc = 0.0;
	}
      }

	if (pParam->BSIM4V4pdibl1 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl1 = %g is negative.\n",
		    pParam->BSIM4V4pdibl1);
	    printf("Warning: Pdibl1 = %g is negative.\n", pParam->BSIM4V4pdibl1);
	}
/*	if (pParam->BSIM4V4pdibl2 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl2 = %g is negative.\n",
		    pParam->BSIM4V4pdibl2);
	    printf("Warning: Pdibl2 = %g is negative.\n", pParam->BSIM4V4pdibl2);
	}
*/

/* Check stress effect parameters */        
        if((here->BSIM4V4sa > 0.0) && (here->BSIM4V4sb > 0.0) && 
       	((here->BSIM4V4nf == 1.0) || ((here->BSIM4V4nf > 1.0) && (here->BSIM4V4sd > 0.0))) )
        {   if (model->BSIM4V4lodk2 <= 0.0)
            {   fprintf(fplog, "Warning: LODK2 = %g is not positive.\n",model->BSIM4V4lodk2);
                printf("Warning: LODK2 = %g is not positive.\n",model->BSIM4V4lodk2);
            }
            if (model->BSIM4V4lodeta0 <= 0.0)
            {   fprintf(fplog, "Warning: LODETA0 = %g is not positive.\n",model->BSIM4V4lodeta0);
            	printf("Warning: LODETA0 = %g is not positive.\n",model->BSIM4V4lodeta0);
            }
 	}

/* Check gate resistance parameters */        
        if (here->BSIM4V4rgateMod == 1)
        {   if (model->BSIM4V4rshg <= 0.0)
	        printf("Warning: rshg should be positive for rgateMod = 1.\n");
	}
        else if (here->BSIM4V4rgateMod == 2)
        {   if (model->BSIM4V4rshg <= 0.0)
                printf("Warning: rshg <= 0.0 for rgateMod = 2.\n");
            else if (pParam->BSIM4V4xrcrg1 <= 0.0)
                     printf("Warning: xrcrg1 <= 0.0 for rgateMod = 2.\n");
         }
        if (here->BSIM4V4rgateMod == 3)
        {   if (model->BSIM4V4rshg <= 0.0)
                printf("Warning: rshg should be positive for rgateMod = 3.\n");
            else if (pParam->BSIM4V4xrcrg1 <= 0.0)
                     printf("Warning: xrcrg1 should be positive for rgateMod = 3.\n");
         }
         
/* Check capacitance parameters */
        if (pParam->BSIM4V4noff < 0.1)
        {   fprintf(fplog, "Warning: Noff = %g is too small.\n",
                    pParam->BSIM4V4noff);
            printf("Warning: Noff = %g is too small.\n", pParam->BSIM4V4noff);
        }

        if (pParam->BSIM4V4voffcv < -0.5)
        {   fprintf(fplog, "Warning: Voffcv = %g is too small.\n",
                    pParam->BSIM4V4voffcv);
            printf("Warning: Voffcv = %g is too small.\n", pParam->BSIM4V4voffcv);
        }

        if (pParam->BSIM4V4moin < 5.0)
        {   fprintf(fplog, "Warning: Moin = %g is too small.\n",
                    pParam->BSIM4V4moin);
            printf("Warning: Moin = %g is too small.\n", pParam->BSIM4V4moin);
        }
        if (pParam->BSIM4V4moin > 25.0)
        {   fprintf(fplog, "Warning: Moin = %g is too large.\n",
                    pParam->BSIM4V4moin);
            printf("Warning: Moin = %g is too large.\n", pParam->BSIM4V4moin);
        }
	if(model->BSIM4V4capMod ==2) {
        	if (pParam->BSIM4V4acde < 0.1)
        	{   fprintf(fplog, "Warning:  Acde = %g is too small.\n",
                    	pParam->BSIM4V4acde);
            	printf("Warning: Acde = %g is too small.\n", pParam->BSIM4V4acde);
        	}
        	if (pParam->BSIM4V4acde > 1.6)
        	{   fprintf(fplog, "Warning:  Acde = %g is too large.\n",
                    	pParam->BSIM4V4acde);
            	printf("Warning: Acde = %g is too large.\n", pParam->BSIM4V4acde);
        	}
	}

/* Check overlap capacitance parameters */
        if (model->BSIM4V4cgdo < 0.0)
	{   fprintf(fplog, "Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM4V4cgdo);
	    printf("Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM4V4cgdo);
	    model->BSIM4V4cgdo = 0.0;
        }      
        if (model->BSIM4V4cgso < 0.0)
	{   fprintf(fplog, "Warning: cgso = %g is negative. Set to zero.\n", model->BSIM4V4cgso);
	    printf("Warning: cgso = %g is negative. Set to zero.\n", model->BSIM4V4cgso);
	    model->BSIM4V4cgso = 0.0;
        }      

      if (model->BSIM4V4tnoiMod == 1) { 
        if (model->BSIM4V4tnoia < 0.0)
        {   fprintf(fplog, "Warning: tnoia = %g is negative. Set to zero.\n", model->BSIM4V4tnoia);
            printf("Warning: tnoia = %g is negative. Set to zero.\n", model->BSIM4V4tnoia);
            model->BSIM4V4tnoia = 0.0;
        }
        if (model->BSIM4V4tnoib < 0.0)
        {   fprintf(fplog, "Warning: tnoib = %g is negative. Set to zero.\n", model->BSIM4V4tnoib);
            printf("Warning: tnoib = %g is negative. Set to zero.\n", model->BSIM4V4tnoib);
            model->BSIM4V4tnoib = 0.0;
        }

         if (model->BSIM4V4rnoia < 0.0)
        {   fprintf(fplog, "Warning: rnoia = %g is negative. Set to zero.\n", model->BSIM4V4rnoia);
            printf("Warning: rnoia = %g is negative. Set to zero.\n", model->BSIM4V4rnoia);
            model->BSIM4V4rnoia = 0.0;
        }
        if (model->BSIM4V4rnoib < 0.0)
        {   fprintf(fplog, "Warning: rnoib = %g is negative. Set to zero.\n", model->BSIM4V4rnoib);
            printf("Warning: rnoib = %g is negative. Set to zero.\n", model->BSIM4V4rnoib);
            model->BSIM4V4rnoib = 0.0;
        }
      }

	if (model->BSIM4V4SjctEmissionCoeff < 0.0)
	{   fprintf(fplog, "Warning: Njs = %g is negative.\n",
		    model->BSIM4V4SjctEmissionCoeff);
	    printf("Warning: Njs = %g is negative.\n",
		    model->BSIM4V4SjctEmissionCoeff);
	}
	if (model->BSIM4V4DjctEmissionCoeff < 0.0)
	{   fprintf(fplog, "Warning: Njd = %g is negative.\n",
		    model->BSIM4V4DjctEmissionCoeff);
	    printf("Warning: Njd = %g is negative.\n",
		    model->BSIM4V4DjctEmissionCoeff);
	}
	if (model->BSIM4V4njtstemp < 0.0)
	{   fprintf(fplog, "Warning: Njts = %g is negative at temperature = %g.\n",
		    model->BSIM4V4njtstemp, ckt->CKTtemp);
	    printf("Warning: Njts = %g is negative at temperature = %g.\n",
		    model->BSIM4V4njtstemp, ckt->CKTtemp);
	}
	if (model->BSIM4V4njtsswtemp < 0.0)
	{   fprintf(fplog, "Warning: Njtssw = %g is negative at temperature = %g.\n",
		    model->BSIM4V4njtsswtemp, ckt->CKTtemp);
	    printf("Warning: Njtssw = %g is negative at temperature = %g.\n",
		    model->BSIM4V4njtsswtemp, ckt->CKTtemp);
	}
	if (model->BSIM4V4njtsswgtemp < 0.0)
	{   fprintf(fplog, "Warning: Njtsswg = %g is negative at temperature = %g.\n",
		    model->BSIM4V4njtsswgtemp, ckt->CKTtemp);
	    printf("Warning: Njtsswg = %g is negative at temperature = %g.\n",
		    model->BSIM4V4njtsswgtemp, ckt->CKTtemp);
	}
	if (model->BSIM4V4vtss < 0.0)
	{   fprintf(fplog, "Warning: Vtss = %g is negative.\n",
		    model->BSIM4V4vtss);
	    printf("Warning: Vtss = %g is negative.\n",
		    model->BSIM4V4vtss);
	}
	if (model->BSIM4V4vtsd < 0.0)
	{   fprintf(fplog, "Warning: Vtsd = %g is negative.\n",
		    model->BSIM4V4vtsd);
	    printf("Warning: Vtsd = %g is negative.\n",
		    model->BSIM4V4vtsd);
	}
	if (model->BSIM4V4vtssws < 0.0)
	{   fprintf(fplog, "Warning: Vtssws = %g is negative.\n",
		    model->BSIM4V4vtssws);
	    printf("Warning: Vtssws = %g is negative.\n",
		    model->BSIM4V4vtssws);
	}
	if (model->BSIM4V4vtsswd < 0.0)
	{   fprintf(fplog, "Warning: Vtsswd = %g is negative.\n",
		    model->BSIM4V4vtsswd);
	    printf("Warning: Vtsswd = %g is negative.\n",
		    model->BSIM4V4vtsswd);
	}
	if (model->BSIM4V4vtsswgs < 0.0)
	{   fprintf(fplog, "Warning: Vtsswgs = %g is negative.\n",
		    model->BSIM4V4vtsswgs);
	    printf("Warning: Vtsswgs = %g is negative.\n",
		    model->BSIM4V4vtsswgs);
	}
	if (model->BSIM4V4vtsswgd < 0.0)
	{   fprintf(fplog, "Warning: Vtsswgd = %g is negative.\n",
		    model->BSIM4V4vtsswgd);
	    printf("Warning: Vtsswgd = %g is negative.\n",
		    model->BSIM4V4vtsswgd);
	}

       if (model->BSIM4V4ntnoi < 0.0)
        {   fprintf(fplog, "Warning: ntnoi = %g is negative. Set to zero.\n", model->BSIM4V4ntnoi);
            printf("Warning: ntnoi = %g is negative. Set to zero.\n", model->BSIM4V4ntnoi);
            model->BSIM4V4ntnoi = 0.0;
        }

     }/* loop for the parameter check for warning messages */      
	fclose(fplog);
    }
    else
    {   fprintf(stderr, "Warning: Can't open log file. Parameter checking skipped.\n");
    }

    return(Fatal_Flag);
}

