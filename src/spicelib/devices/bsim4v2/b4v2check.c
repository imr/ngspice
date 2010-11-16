/**** BSIM4.2.1, Released by Xuemei Xi 10/05/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b4check.c of BSIM4.2.1.
 * Author: 2000 Weidong Liu
 * Authors: Xuemei Xi, Kanyu M. Cao, Hui Wan, Mansun Chan, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "cktdefs.h"
#include "bsim4v2def.h"
#include "trandefs.h"
#include "const.h"
#include "sperror.h"
#include "devdefs.h"

int
BSIM4v2checkModel(
BSIM4v2model *model,
BSIM4v2instance *here,
CKTcircuit *ckt)
{
struct bsim4SizeDependParam *pParam;
int Fatal_Flag = 0;
FILE *fplog;

    IGNORE(ckt);
    
    if ((fplog = fopen("bsim4.out", "w")) != NULL)
    {   pParam = here->pParam;
        fprintf(fplog, "BSIM4v2: Berkeley Short Channel IGFET Model-4\n");
        fprintf(fplog, "Developed by Weidong Liu, Xuemei Xi , Xiaodong Jin, Kanyu M. Cao and Prof. Chenming Hu in 2001.\n");
        fprintf(fplog, "\n");
	fprintf(fplog, "++++++++++ BSIM4v2 PARAMETER CHECKING BELOW ++++++++++\n");

        if (strcmp(model->BSIM4v2version, "4.2.1") != 0)
        {  fprintf(fplog, "Warning: This model is BSIM4.2.1; you specified a wrong version number.\n");
           printf("Warning: This model is BSIM4.2.1; you specified a wrong version number.\n");
        }
	fprintf(fplog, "Model = %s\n", model->BSIM4v2modName);


        if ((here->BSIM4v2rgateMod == 2) || (here->BSIM4v2rgateMod == 3))
        {   if ((here->BSIM4v2trnqsMod == 1) || (here->BSIM4v2acnqsMod == 1))
            {   fprintf(fplog, "Warning: You've selected both Rg and charge deficit NQS; select one only.\n");
                printf("Warning: You've selected both Rg and charge deficit NQS; select one only.\n");
            }
        }


	if (model->BSIM4v2toxe <= 0.0)
	{   fprintf(fplog, "Fatal: Toxe = %g is not positive.\n",
		    model->BSIM4v2toxe);
	    printf("Fatal: Toxe = %g is not positive.\n", model->BSIM4v2toxe);
	    Fatal_Flag = 1;
	}
        if (model->BSIM4v2toxp <= 0.0)
        {   fprintf(fplog, "Fatal: Toxp = %g is not positive.\n",
                    model->BSIM4v2toxp);
            printf("Fatal: Toxp = %g is not positive.\n", model->BSIM4v2toxp);
            Fatal_Flag = 1;
        }

        if (model->BSIM4v2toxm <= 0.0)
        {   fprintf(fplog, "Fatal: Toxm = %g is not positive.\n",
                    model->BSIM4v2toxm);
            printf("Fatal: Toxm = %g is not positive.\n", model->BSIM4v2toxm);
            Fatal_Flag = 1;
        }

        if (model->BSIM4v2toxref <= 0.0)
        {   fprintf(fplog, "Fatal: Toxref = %g is not positive.\n",
                    model->BSIM4v2toxref);
            printf("Fatal: Toxref = %g is not positive.\n", model->BSIM4v2toxref);
            Fatal_Flag = 1;
        }

        if (pParam->BSIM4v2lpe0 < -pParam->BSIM4v2leff)
        {   fprintf(fplog, "Fatal: Lpe0 = %g is less than -Leff.\n",
                    pParam->BSIM4v2lpe0);
            printf("Fatal: Lpe0 = %g is less than -Leff.\n",
                        pParam->BSIM4v2lpe0);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v2lpeb < -pParam->BSIM4v2leff)
        {   fprintf(fplog, "Fatal: Lpeb = %g is less than -Leff.\n",
                    pParam->BSIM4v2lpeb);
            printf("Fatal: Lpeb = %g is less than -Leff.\n",
                        pParam->BSIM4v2lpeb);
            Fatal_Flag = 1;
        }

        if (pParam->BSIM4v2phin < -0.4)
        {   fprintf(fplog, "Fatal: Phin = %g is less than -0.4.\n",
                    pParam->BSIM4v2phin);
            printf("Fatal: Phin = %g is less than -0.4.\n",
                   pParam->BSIM4v2phin);
            Fatal_Flag = 1;
        }
	if (pParam->BSIM4v2ndep <= 0.0)
	{   fprintf(fplog, "Fatal: Ndep = %g is not positive.\n",
		    pParam->BSIM4v2ndep);
	    printf("Fatal: Ndep = %g is not positive.\n",
		   pParam->BSIM4v2ndep);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v2nsub <= 0.0)
	{   fprintf(fplog, "Fatal: Nsub = %g is not positive.\n",
		    pParam->BSIM4v2nsub);
	    printf("Fatal: Nsub = %g is not positive.\n",
		   pParam->BSIM4v2nsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v2ngate < 0.0)
	{   fprintf(fplog, "Fatal: Ngate = %g is not positive.\n",
		    pParam->BSIM4v2ngate);
	    printf("Fatal: Ngate = %g Ngate is not positive.\n",
		   pParam->BSIM4v2ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v2ngate > 1.e25)
	{   fprintf(fplog, "Fatal: Ngate = %g is too high.\n",
		    pParam->BSIM4v2ngate);
	    printf("Fatal: Ngate = %g Ngate is too high\n",
		   pParam->BSIM4v2ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v2xj <= 0.0)
	{   fprintf(fplog, "Fatal: Xj = %g is not positive.\n",
		    pParam->BSIM4v2xj);
	    printf("Fatal: Xj = %g is not positive.\n", pParam->BSIM4v2xj);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4v2dvt1 < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1 = %g is negative.\n",
		    pParam->BSIM4v2dvt1);   
	    printf("Fatal: Dvt1 = %g is negative.\n", pParam->BSIM4v2dvt1);   
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM4v2dvt1w < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1w = %g is negative.\n",
		    pParam->BSIM4v2dvt1w);
	    printf("Fatal: Dvt1w = %g is negative.\n", pParam->BSIM4v2dvt1w);
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM4v2w0 == -pParam->BSIM4v2weff)
	{   fprintf(fplog, "Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }   

	if (pParam->BSIM4v2dsub < 0.0)
	{   fprintf(fplog, "Fatal: Dsub = %g is negative.\n", pParam->BSIM4v2dsub);
	    printf("Fatal: Dsub = %g is negative.\n", pParam->BSIM4v2dsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v2b1 == -pParam->BSIM4v2weff)
	{   fprintf(fplog, "Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }  
        if (pParam->BSIM4v2u0temp <= 0.0)
	{   fprintf(fplog, "Fatal: u0 at current temperature = %g is not positive.\n", pParam->BSIM4v2u0temp);
	    printf("Fatal: u0 at current temperature = %g is not positive.\n",
		   pParam->BSIM4v2u0temp);
	    Fatal_Flag = 1;
        }
    
        if (pParam->BSIM4v2delta < 0.0)
	{   fprintf(fplog, "Fatal: Delta = %g is less than zero.\n",
		    pParam->BSIM4v2delta);
	    printf("Fatal: Delta = %g is less than zero.\n", pParam->BSIM4v2delta);
	    Fatal_Flag = 1;
        }      

	if (pParam->BSIM4v2vsattemp <= 0.0)
	{   fprintf(fplog, "Fatal: Vsat at current temperature = %g is not positive.\n", pParam->BSIM4v2vsattemp);
	    printf("Fatal: Vsat at current temperature = %g is not positive.\n",
		   pParam->BSIM4v2vsattemp);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4v2pclm <= 0.0)
	{   fprintf(fplog, "Fatal: Pclm = %g is not positive.\n", pParam->BSIM4v2pclm);
	    printf("Fatal: Pclm = %g is not positive.\n", pParam->BSIM4v2pclm);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4v2drout < 0.0)
	{   fprintf(fplog, "Fatal: Drout = %g is negative.\n", pParam->BSIM4v2drout);
	    printf("Fatal: Drout = %g is negative.\n", pParam->BSIM4v2drout);
	    Fatal_Flag = 1;
	}

        if (pParam->BSIM4v2pscbe2 <= 0.0)
        {   fprintf(fplog, "Warning: Pscbe2 = %g is not positive.\n",
                    pParam->BSIM4v2pscbe2);
            printf("Warning: Pscbe2 = %g is not positive.\n", pParam->BSIM4v2pscbe2);
        }

        if (here->BSIM4v2nf < 1.0)
        {   fprintf(fplog, "Fatal: Number of finger = %g is smaller than one.\n", here->BSIM4v2nf);
            printf("Fatal: Number of finger = %g is smaller than one.\n", here->BSIM4v2nf);
            Fatal_Flag = 1;
        }

        if ((here->BSIM4v2l + model->BSIM4v2xl) <= model->BSIM4v2xgl)
        {   fprintf(fplog, "Fatal: The parameter xgl must be smaller than Ldrawn+XL.\n");
            printf("Fatal: The parameter xgl must be smaller than Ldrawn+XL.\n");
            Fatal_Flag = 1;
        }
        if (model->BSIM4v2ngcon < 1.0)
        {   fprintf(fplog, "Fatal: The parameter ngcon cannot be smaller than one.\n");
            printf("Fatal: The parameter ngcon cannot be smaller than one.\n");
            Fatal_Flag = 1;
        }
        if ((model->BSIM4v2ngcon != 1.0) && (model->BSIM4v2ngcon != 2.0))
        {   model->BSIM4v2ngcon = 1.0;
            fprintf(fplog, "Warning: Ngcon must be equal to one or two; reset to 1.0.\n");
            printf("Warning: Ngcon must be equal to one or two; reset to 1.0.\n");
        }

        if (model->BSIM4v2gbmin < 1.0e-20)
        {   fprintf(fplog, "Warning: Gbmin = %g is too small.\n",
                    model->BSIM4v2gbmin);
            printf("Warning: Gbmin = %g is too small.\n", model->BSIM4v2gbmin);
        }

        if (pParam->BSIM4v2noff < 0.1)
        {   fprintf(fplog, "Warning: Noff = %g is too small.\n",
                    pParam->BSIM4v2noff);
            printf("Warning: Noff = %g is too small.\n", pParam->BSIM4v2noff);
        }
        if (pParam->BSIM4v2noff > 4.0)
        {   fprintf(fplog, "Warning: Noff = %g is too large.\n",
                    pParam->BSIM4v2noff);
            printf("Warning: Noff = %g is too large.\n", pParam->BSIM4v2noff);
        }

        if (pParam->BSIM4v2voffcv < -0.5)
        {   fprintf(fplog, "Warning: Voffcv = %g is too small.\n",
                    pParam->BSIM4v2voffcv);
            printf("Warning: Voffcv = %g is too small.\n", pParam->BSIM4v2voffcv);
        }
        if (pParam->BSIM4v2voffcv > 0.5)
        {   fprintf(fplog, "Warning: Voffcv = %g is too large.\n",
                    pParam->BSIM4v2voffcv);
            printf("Warning: Voffcv = %g is too large.\n", pParam->BSIM4v2voffcv);
        }

        /* Check capacitance parameters */
        if (pParam->BSIM4v2clc < 0.0)
	{   fprintf(fplog, "Fatal: Clc = %g is negative.\n", pParam->BSIM4v2clc);
	    printf("Fatal: Clc = %g is negative.\n", pParam->BSIM4v2clc);
	    Fatal_Flag = 1;
        }      

        if (pParam->BSIM4v2moin < 5.0)
        {   fprintf(fplog, "Warning: Moin = %g is too small.\n",
                    pParam->BSIM4v2moin);
            printf("Warning: Moin = %g is too small.\n", pParam->BSIM4v2moin);
        }
        if (pParam->BSIM4v2moin > 25.0)
        {   fprintf(fplog, "Warning: Moin = %g is too large.\n",
                    pParam->BSIM4v2moin);
            printf("Warning: Moin = %g is too large.\n", pParam->BSIM4v2moin);
        }
	if(model->BSIM4v2capMod ==2) {
        	if (pParam->BSIM4v2acde < 0.4)
        	{   fprintf(fplog, "Warning:  Acde = %g is too small.\n",
                    	pParam->BSIM4v2acde);
            	printf("Warning: Acde = %g is too small.\n", pParam->BSIM4v2acde);
        	}
        	if (pParam->BSIM4v2acde > 1.6)
        	{   fprintf(fplog, "Warning:  Acde = %g is too large.\n",
                    	pParam->BSIM4v2acde);
            	printf("Warning: Acde = %g is too large.\n", pParam->BSIM4v2acde);
        	}
	}

      if (model->BSIM4v2paramChk ==1)
      {
/* Check L and W parameters */ 
	if (pParam->BSIM4v2leff <= 1.0e-9)
	{   fprintf(fplog, "Warning: Leff = %g <= 1.0e-9. Recommended Leff >= 1e-8 \n", 
		    pParam->BSIM4v2leff);
	    printf("Warning: Leff = %g <= 1.0e-9. Recommended Leff >= 1e-8 \n",
		    pParam->BSIM4v2leff);
	}    
	
	if (pParam->BSIM4v2leffCV <= 1.0e-9)
	{   fprintf(fplog, "Warning: Leff for CV = %g <= 1.0e-9. Recommended LeffCV >=1e-8 \n",
		    pParam->BSIM4v2leffCV);
	    printf("Warning: Leff for CV = %g <= 1.0e-9. Recommended LeffCV >=1e-8 \n",
		    pParam->BSIM4v2leffCV);
	}  
	
        if (pParam->BSIM4v2weff <= 1.0e-9)
	{   fprintf(fplog, "Warning: Weff = %g <= 1.0e-9. Recommended Weff >=1e-7 \n",
		    pParam->BSIM4v2weff);
	    printf("Warning: Weff = %g <= 1.0e-9. Recommended Weff >=1e-7 \n",
		   pParam->BSIM4v2weff);
	}             
	
	if (pParam->BSIM4v2weffCV <= 1.0e-9)
	{   fprintf(fplog, "Warning: Weff for CV = %g <= 1.0e-9. Recommended WeffCV >= 1e-7 \n",
		    pParam->BSIM4v2weffCV);
	    printf("Warning: Weff for CV = %g <= 1.0e-9. Recommended WeffCV >= 1e-7 \n",
		    pParam->BSIM4v2weffCV);
	}        
	
        /* Check threshold voltage parameters */
	if (model->BSIM4v2toxe < 1.0e-10)
	{   fprintf(fplog, "Warning: Toxe = %g is less than 1A. Recommended Toxe >= 5A\n",
	            model->BSIM4v2toxe);
	    printf("Warning: Toxe = %g is less than 1A. Recommended Toxe >= 5A\n", model->BSIM4v2toxe);
        }
        if (model->BSIM4v2toxp < 1.0e-10)
        {   fprintf(fplog, "Warning: Toxp = %g is less than 1A. Recommended Toxp >= 5A\n",
                    model->BSIM4v2toxp);
            printf("Warning: Toxp = %g is less than 1A. Recommended Toxp >= 5A\n", model->BSIM4v2toxp);
        }
        if (model->BSIM4v2toxm < 1.0e-10)
        {   fprintf(fplog, "Warning: Toxm = %g is less than 1A. Recommended Toxm >= 5A\n",
                    model->BSIM4v2toxm);
            printf("Warning: Toxm = %g is less than 1A. Recommended Toxm >= 5A\n", model->BSIM4v2toxm);
        }

        if (pParam->BSIM4v2ndep <= 1.0e12)
	{   fprintf(fplog, "Warning: Ndep = %g may be too small.\n",
	            pParam->BSIM4v2ndep);
	    printf("Warning: Ndep = %g may be too small.\n",
	           pParam->BSIM4v2ndep);
	}
	else if (pParam->BSIM4v2ndep >= 1.0e21)
	{   fprintf(fplog, "Warning: Ndep = %g may be too large.\n",
	            pParam->BSIM4v2ndep);
	    printf("Warning: Ndep = %g may be too large.\n",
	           pParam->BSIM4v2ndep);
	}

	 if (pParam->BSIM4v2nsub <= 1.0e14)
	{   fprintf(fplog, "Warning: Nsub = %g may be too small.\n",
	            pParam->BSIM4v2nsub);
	    printf("Warning: Nsub = %g may be too small.\n",
	           pParam->BSIM4v2nsub);
	}
	else if (pParam->BSIM4v2nsub >= 1.0e21)
	{   fprintf(fplog, "Warning: Nsub = %g may be too large.\n",
	            pParam->BSIM4v2nsub);
	    printf("Warning: Nsub = %g may be too large.\n",
	           pParam->BSIM4v2nsub);
	}

	if ((pParam->BSIM4v2ngate > 0.0) &&
	    (pParam->BSIM4v2ngate <= 1.e18))
	{   fprintf(fplog, "Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	            pParam->BSIM4v2ngate);
	    printf("Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	           pParam->BSIM4v2ngate);
	}
       
        if (pParam->BSIM4v2dvt0 < 0.0)
	{   fprintf(fplog, "Warning: Dvt0 = %g is negative.\n",
		    pParam->BSIM4v2dvt0);   
	    printf("Warning: Dvt0 = %g is negative.\n", pParam->BSIM4v2dvt0);   
	}
	    
	if (fabs(1.0e-6 / (pParam->BSIM4v2w0 + pParam->BSIM4v2weff)) > 10.0)
	{   fprintf(fplog, "Warning: (W0 + Weff) may be too small.\n");
	    printf("Warning: (W0 + Weff) may be too small.\n");
        }

/* Check subthreshold parameters */
	if (pParam->BSIM4v2nfactor < 0.0)
	{   fprintf(fplog, "Warning: Nfactor = %g is negative.\n",
		    pParam->BSIM4v2nfactor);
	    printf("Warning: Nfactor = %g is negative.\n", pParam->BSIM4v2nfactor);
	}
	if (pParam->BSIM4v2cdsc < 0.0)
	{   fprintf(fplog, "Warning: Cdsc = %g is negative.\n",
		    pParam->BSIM4v2cdsc);
	    printf("Warning: Cdsc = %g is negative.\n", pParam->BSIM4v2cdsc);
	}
	if (pParam->BSIM4v2cdscd < 0.0)
	{   fprintf(fplog, "Warning: Cdscd = %g is negative.\n",
		    pParam->BSIM4v2cdscd);
	    printf("Warning: Cdscd = %g is negative.\n", pParam->BSIM4v2cdscd);
	}
/* Check DIBL parameters */
	if (pParam->BSIM4v2eta0 < 0.0)
	{   fprintf(fplog, "Warning: Eta0 = %g is negative.\n",
		    pParam->BSIM4v2eta0); 
	    printf("Warning: Eta0 = %g is negative.\n", pParam->BSIM4v2eta0); 
	}
	      
/* Check Abulk parameters */	    
	 if (fabs(1.0e-6 / (pParam->BSIM4v2b1 + pParam->BSIM4v2weff)) > 10.0)
       	{   fprintf(fplog, "Warning: (B1 + Weff) may be too small.\n");
       	    printf("Warning: (B1 + Weff) may be too small.\n");
        }    
    

/* Check Saturation parameters */
     	if (pParam->BSIM4v2a2 < 0.01)
	{   fprintf(fplog, "Warning: A2 = %g is too small. Set to 0.01.\n", pParam->BSIM4v2a2);
	    printf("Warning: A2 = %g is too small. Set to 0.01.\n",
		   pParam->BSIM4v2a2);
	    pParam->BSIM4v2a2 = 0.01;
	}
	else if (pParam->BSIM4v2a2 > 1.0)
	{   fprintf(fplog, "Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		    pParam->BSIM4v2a2);
	    printf("Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		   pParam->BSIM4v2a2);
	    pParam->BSIM4v2a2 = 1.0;
	    pParam->BSIM4v2a1 = 0.0;

	}

        if (pParam->BSIM4v2prwg < 0.0)
        {   fprintf(fplog, "Warning: Prwg = %g is negative. Set to zero.\n",
                    pParam->BSIM4v2prwg);
            printf("Warning: Prwg = %g is negative. Set to zero.\n",
                   pParam->BSIM4v2prwg);
            pParam->BSIM4v2prwg = 0.0;
        }
	if (pParam->BSIM4v2rdsw < 0.0)
	{   fprintf(fplog, "Warning: Rdsw = %g is negative. Set to zero.\n",
		    pParam->BSIM4v2rdsw);
	    printf("Warning: Rdsw = %g is negative. Set to zero.\n",
		   pParam->BSIM4v2rdsw);
	    pParam->BSIM4v2rdsw = 0.0;
	    pParam->BSIM4v2rds0 = 0.0;
	}
	if (pParam->BSIM4v2rds0 < 0.0)
	{   fprintf(fplog, "Warning: Rds at current temperature = %g is negative. Set to zero.\n",
		    pParam->BSIM4v2rds0);
	    printf("Warning: Rds at current temperature = %g is negative. Set to zero.\n",
		   pParam->BSIM4v2rds0);
	    pParam->BSIM4v2rds0 = 0.0;
	}

        if (pParam->BSIM4v2rdswmin < 0.0)
        {   fprintf(fplog, "Warning: Rdswmin at current temperature = %g is negative. Set to zero.\n",
                    pParam->BSIM4v2rdswmin);
            printf("Warning: Rdswmin at current temperature = %g is negative. Set to zero.\n",
                   pParam->BSIM4v2rdswmin);
            pParam->BSIM4v2rdswmin = 0.0;
        }

        if (pParam->BSIM4v2vsattemp < 1.0e3)
	{   fprintf(fplog, "Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM4v2vsattemp);
	   printf("Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM4v2vsattemp);
	}

        if (pParam->BSIM4v2fprout < 0.0)
        {   fprintf(fplog, "Fatal: fprout = %g is negative.\n",
                    pParam->BSIM4v2fprout);
            printf("Fatal: fprout = %g is negative.\n", pParam->BSIM4v2fprout);
	    Fatal_Flag = 1;
        }
        if (pParam->BSIM4v2pdits < 0.0)
        {   fprintf(fplog, "Fatal: pdits = %g is negative.\n",
                    pParam->BSIM4v2pdits);
            printf("Fatal: pdits = %g is negative.\n", pParam->BSIM4v2pdits);
            Fatal_Flag = 1;
        }
        if (model->BSIM4v2pditsl < 0.0)
        {   fprintf(fplog, "Fatal: pditsl = %g is negative.\n",
                    model->BSIM4v2pditsl);
            printf("Fatal: pditsl = %g is negative.\n", model->BSIM4v2pditsl);
            Fatal_Flag = 1;
        }

	if (pParam->BSIM4v2pdibl1 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl1 = %g is negative.\n",
		    pParam->BSIM4v2pdibl1);
	    printf("Warning: Pdibl1 = %g is negative.\n", pParam->BSIM4v2pdibl1);
	}
	if (pParam->BSIM4v2pdibl2 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl2 = %g is negative.\n",
		    pParam->BSIM4v2pdibl2);
	    printf("Warning: Pdibl2 = %g is negative.\n", pParam->BSIM4v2pdibl2);
	}

        if (pParam->BSIM4v2nigbinv <= 0.0)
        {   fprintf(fplog, "Fatal: nigbinv = %g is non-positive.\n",
                    pParam->BSIM4v2nigbinv);
            printf("Fatal: nigbinv = %g is non-positive.\n", pParam->BSIM4v2nigbinv);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v2nigbacc <= 0.0)
        {   fprintf(fplog, "Fatal: nigbacc = %g is non-positive.\n",
                    pParam->BSIM4v2nigbacc);
            printf("Fatal: nigbacc = %g is non-positive.\n", pParam->BSIM4v2nigbacc);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v2nigc <= 0.0)
        {   fprintf(fplog, "Fatal: nigc = %g is non-positive.\n",
                    pParam->BSIM4v2nigc);
            printf("Fatal: nigc = %g is non-positive.\n", pParam->BSIM4v2nigc);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v2poxedge <= 0.0)
        {   fprintf(fplog, "Fatal: poxedge = %g is non-positive.\n",
                    pParam->BSIM4v2poxedge);
            printf("Fatal: poxedge = %g is non-positive.\n", pParam->BSIM4v2poxedge);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v2pigcd <= 0.0)
        {   fprintf(fplog, "Fatal: pigcd = %g is non-positive.\n",
                    pParam->BSIM4v2pigcd);
            printf("Fatal: pigcd = %g is non-positive.\n", pParam->BSIM4v2pigcd);
            Fatal_Flag = 1;
        }


/* Check overlap capacitance parameters */
        if (model->BSIM4v2cgdo < 0.0)
	{   fprintf(fplog, "Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM4v2cgdo);
	    printf("Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM4v2cgdo);
	    model->BSIM4v2cgdo = 0.0;
        }      
        if (model->BSIM4v2cgso < 0.0)
	{   fprintf(fplog, "Warning: cgso = %g is negative. Set to zero.\n", model->BSIM4v2cgso);
	    printf("Warning: cgso = %g is negative. Set to zero.\n", model->BSIM4v2cgso);
	    model->BSIM4v2cgso = 0.0;
        }      

        if (model->BSIM4v2tnoia < 0.0)
        {   fprintf(fplog, "Warning: tnoia = %g is negative. Set to zero.\n", model->BSIM4v2tnoia);
            printf("Warning: tnoia = %g is negative. Set to zero.\n", model->BSIM4v2tnoia);
            model->BSIM4v2tnoia = 0.0;
        }
        if (model->BSIM4v2tnoib < 0.0)
        {   fprintf(fplog, "Warning: tnoib = %g is negative. Set to zero.\n", model->BSIM4v2tnoib);
            printf("Warning: tnoib = %g is negative. Set to zero.\n", model->BSIM4v2tnoib);
            model->BSIM4v2tnoib = 0.0;
        }

        if (model->BSIM4v2ntnoi < 0.0)
        {   fprintf(fplog, "Warning: ntnoi = %g is negative. Set to zero.\n", model->BSIM4v2ntnoi);
            printf("Warning: ntnoi = %g is negative. Set to zero.\n", model->BSIM4v2ntnoi);
            model->BSIM4v2ntnoi = 0.0;
        }

     }/* loop for the parameter check for warning messages */      
	fclose(fplog);
    }
    else
    {   fprintf(stderr, "Warning: Can't open log file. Parameter checking skipped.\n");
    }

    return(Fatal_Flag);
}

