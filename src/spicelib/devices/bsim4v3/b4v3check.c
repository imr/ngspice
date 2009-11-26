/**** BSIM4.3.0 Released by Xuemei (Jane) Xi 05/09/2003 ****/

/**********
 * Copyright 2003 Regents of the University of California. All rights reserved.
 * File: b4v3check.c of BSIM4.3.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "cktdefs.h"
#include "bsim4v3def.h"
#include "trandefs.h"
#include "const.h"
#include "sperror.h"
#include "devdefs.h"

int
BSIM4v3checkModel(
BSIM4v3model *model,
BSIM4v3instance *here,
CKTcircuit *ckt)
{
struct bsim4v3SizeDependParam *pParam;
int Fatal_Flag = 0;
FILE *fplog;
    
    if ((fplog = fopen("bsim4v3.out", "w")) != NULL)
    {   pParam = here->pParam;
        fprintf(fplog, "BSIM4v3: Berkeley Short Channel IGFET Model-4\n");
        fprintf(fplog, "Developed by Xuemei (Jane) Xi, Jin He, Mohan Dunga, Prof. Ali Niknejad and Prof. Chenming Hu in 2003.\n");
        fprintf(fplog, "\n");
	fprintf(fplog, "++++++++++ BSIM4v3 PARAMETER CHECKING BELOW ++++++++++\n");

        if (strcmp(model->BSIM4v3version, "4.3.0") != 0)
        {  fprintf(fplog, "Warning: This model is BSIM4.3.0; you specified a wrong version number.\n");
           printf("Warning: This model is BSIM4.3.0; you specified a wrong version number.\n");
        }
	fprintf(fplog, "Model = %s\n", model->BSIM4v3modName);


        if ((here->BSIM4v3rgateMod == 2) || (here->BSIM4v3rgateMod == 3))
        {   if ((here->BSIM4v3trnqsMod == 1) || (here->BSIM4v3acnqsMod == 1))
            {   fprintf(fplog, "Warning: You've selected both Rg and charge deficit NQS; select one only.\n");
                printf("Warning: You've selected both Rg and charge deficit NQS; select one only.\n");
            }
        }


	if (model->BSIM4v3toxe <= 0.0)
	{   fprintf(fplog, "Fatal: Toxe = %g is not positive.\n",
		    model->BSIM4v3toxe);
	    printf("Fatal: Toxe = %g is not positive.\n", model->BSIM4v3toxe);
	    Fatal_Flag = 1;
	}
        if (model->BSIM4v3toxp <= 0.0)
        {   fprintf(fplog, "Fatal: Toxp = %g is not positive.\n",
                    model->BSIM4v3toxp);
            printf("Fatal: Toxp = %g is not positive.\n", model->BSIM4v3toxp);
            Fatal_Flag = 1;
        }

        if (model->BSIM4v3toxm <= 0.0)
        {   fprintf(fplog, "Fatal: Toxm = %g is not positive.\n",
                    model->BSIM4v3toxm);
            printf("Fatal: Toxm = %g is not positive.\n", model->BSIM4v3toxm);
            Fatal_Flag = 1;
        }

        if (model->BSIM4v3toxref <= 0.0)
        {   fprintf(fplog, "Fatal: Toxref = %g is not positive.\n",
                    model->BSIM4v3toxref);
            printf("Fatal: Toxref = %g is not positive.\n", model->BSIM4v3toxref);
            Fatal_Flag = 1;
        }

        if (pParam->BSIM4v3lpe0 < -pParam->BSIM4v3leff)
        {   fprintf(fplog, "Fatal: Lpe0 = %g is less than -Leff.\n",
                    pParam->BSIM4v3lpe0);
            printf("Fatal: Lpe0 = %g is less than -Leff.\n",
                        pParam->BSIM4v3lpe0);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v3lpeb < -pParam->BSIM4v3leff)
        {   fprintf(fplog, "Fatal: Lpeb = %g is less than -Leff.\n",
                    pParam->BSIM4v3lpeb);
            printf("Fatal: Lpeb = %g is less than -Leff.\n",
                        pParam->BSIM4v3lpeb);
            Fatal_Flag = 1;
        }

        if (pParam->BSIM4v3phin < -0.4)
        {   fprintf(fplog, "Fatal: Phin = %g is less than -0.4.\n",
                    pParam->BSIM4v3phin);
            printf("Fatal: Phin = %g is less than -0.4.\n",
                   pParam->BSIM4v3phin);
            Fatal_Flag = 1;
        }
	if (pParam->BSIM4v3ndep <= 0.0)
	{   fprintf(fplog, "Fatal: Ndep = %g is not positive.\n",
		    pParam->BSIM4v3ndep);
	    printf("Fatal: Ndep = %g is not positive.\n",
		   pParam->BSIM4v3ndep);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v3nsub <= 0.0)
	{   fprintf(fplog, "Fatal: Nsub = %g is not positive.\n",
		    pParam->BSIM4v3nsub);
	    printf("Fatal: Nsub = %g is not positive.\n",
		   pParam->BSIM4v3nsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v3ngate < 0.0)
	{   fprintf(fplog, "Fatal: Ngate = %g is not positive.\n",
		    pParam->BSIM4v3ngate);
	    printf("Fatal: Ngate = %g Ngate is not positive.\n",
		   pParam->BSIM4v3ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v3ngate > 1.e25)
	{   fprintf(fplog, "Fatal: Ngate = %g is too high.\n",
		    pParam->BSIM4v3ngate);
	    printf("Fatal: Ngate = %g Ngate is too high\n",
		   pParam->BSIM4v3ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v3xj <= 0.0)
	{   fprintf(fplog, "Fatal: Xj = %g is not positive.\n",
		    pParam->BSIM4v3xj);
	    printf("Fatal: Xj = %g is not positive.\n", pParam->BSIM4v3xj);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4v3dvt1 < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1 = %g is negative.\n",
		    pParam->BSIM4v3dvt1);   
	    printf("Fatal: Dvt1 = %g is negative.\n", pParam->BSIM4v3dvt1);   
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM4v3dvt1w < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1w = %g is negative.\n",
		    pParam->BSIM4v3dvt1w);
	    printf("Fatal: Dvt1w = %g is negative.\n", pParam->BSIM4v3dvt1w);
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM4v3w0 == -pParam->BSIM4v3weff)
	{   fprintf(fplog, "Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }   

	if (pParam->BSIM4v3dsub < 0.0)
	{   fprintf(fplog, "Fatal: Dsub = %g is negative.\n", pParam->BSIM4v3dsub);
	    printf("Fatal: Dsub = %g is negative.\n", pParam->BSIM4v3dsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v3b1 == -pParam->BSIM4v3weff)
	{   fprintf(fplog, "Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }  
        if (pParam->BSIM4v3u0temp <= 0.0)
	{   fprintf(fplog, "Fatal: u0 at current temperature = %g is not positive.\n", pParam->BSIM4v3u0temp);
	    printf("Fatal: u0 at current temperature = %g is not positive.\n",
		   pParam->BSIM4v3u0temp);
	    Fatal_Flag = 1;
        }
    
        if (pParam->BSIM4v3delta < 0.0)
	{   fprintf(fplog, "Fatal: Delta = %g is less than zero.\n",
		    pParam->BSIM4v3delta);
	    printf("Fatal: Delta = %g is less than zero.\n", pParam->BSIM4v3delta);
	    Fatal_Flag = 1;
        }      

	if (pParam->BSIM4v3vsattemp <= 0.0)
	{   fprintf(fplog, "Fatal: Vsat at current temperature = %g is not positive.\n", pParam->BSIM4v3vsattemp);
	    printf("Fatal: Vsat at current temperature = %g is not positive.\n",
		   pParam->BSIM4v3vsattemp);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4v3pclm <= 0.0)
	{   fprintf(fplog, "Fatal: Pclm = %g is not positive.\n", pParam->BSIM4v3pclm);
	    printf("Fatal: Pclm = %g is not positive.\n", pParam->BSIM4v3pclm);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4v3drout < 0.0)
	{   fprintf(fplog, "Fatal: Drout = %g is negative.\n", pParam->BSIM4v3drout);
	    printf("Fatal: Drout = %g is negative.\n", pParam->BSIM4v3drout);
	    Fatal_Flag = 1;
	}

        if (pParam->BSIM4v3pscbe2 <= 0.0)
        {   fprintf(fplog, "Warning: Pscbe2 = %g is not positive.\n",
                    pParam->BSIM4v3pscbe2);
            printf("Warning: Pscbe2 = %g is not positive.\n", pParam->BSIM4v3pscbe2);
        }

        if (here->BSIM4v3nf < 1.0)
        {   fprintf(fplog, "Fatal: Number of finger = %g is smaller than one.\n", here->BSIM4v3nf);
            printf("Fatal: Number of finger = %g is smaller than one.\n", here->BSIM4v3nf);
            Fatal_Flag = 1;
        }

        if((here->BSIM4v3sa > 0.0) && (here->BSIM4v3sb > 0.0) && 
       	((here->BSIM4v3nf == 1.0) || ((here->BSIM4v3nf > 1.0) && (here->BSIM4v3sd > 0.0))) )
        {   if (model->BSIM4v3saref <= 0.0)
            {   fprintf(fplog, "Fatal: SAref = %g is not positive.\n",model->BSIM4v3saref);
             	printf("Fatal: SAref = %g is not positive.\n",model->BSIM4v3saref);
             	Fatal_Flag = 1;
            }
            if (model->BSIM4v3sbref <= 0.0)
            {   fprintf(fplog, "Fatal: SBref = %g is not positive.\n",model->BSIM4v3sbref);
            	printf("Fatal: SBref = %g is not positive.\n",model->BSIM4v3sbref);
            	Fatal_Flag = 1;
            }
	    if (model->BSIM4v3wlod < 0.0)
            {   fprintf(fplog, "Warning: WLOD = %g is less than 0. Set to 0.0\n",model->BSIM4v3wlod);
                printf("Warning: WLOD = %g is less than 0.\n",model->BSIM4v3wlod);
                model->BSIM4v3wlod = 0.0;
            }
            if (model->BSIM4v3kvsat < -1.0 )
            {   fprintf(fplog, "Warning: KVSAT = %g is too small; Reset to -1.0.\n",model->BSIM4v3kvsat);
            	printf("Warning: KVSAT = %g is is too small; Reset to -1.0.\n",model->BSIM4v3kvsat);
            	model->BSIM4v3kvsat = -1.0;
            }
            if (model->BSIM4v3kvsat > 1.0)
            {   fprintf(fplog, "Warning: KVSAT = %g is too big; Reset to 1.0.\n",model->BSIM4v3kvsat);
            	printf("Warning: KVSAT = %g is too big; Reset to 1.0.\n",model->BSIM4v3kvsat);
            	model->BSIM4v3kvsat = 1.0;
            }
            if (model->BSIM4v3lodk2 <= 0.0)
            {   fprintf(fplog, "Warning: LODK2 = %g is not positive.\n",model->BSIM4v3lodk2);
                printf("Warning: LODK2 = %g is not positive.\n",model->BSIM4v3lodk2);
            }
            if (model->BSIM4v3lodeta0 <= 0.0)
            {   fprintf(fplog, "Warning: LODETA0 = %g is not positive.\n",model->BSIM4v3lodeta0);
            	printf("Warning: LODETA0 = %g is not positive.\n",model->BSIM4v3lodeta0);
            }
 	}

        if ((here->BSIM4v3l + model->BSIM4v3xl) <= model->BSIM4v3xgl)
        {   fprintf(fplog, "Fatal: The parameter xgl must be smaller than Ldrawn+XL.\n");
            printf("Fatal: The parameter xgl must be smaller than Ldrawn+XL.\n");
            Fatal_Flag = 1;
        }
        if (model->BSIM4v3ngcon < 1.0)
        {   fprintf(fplog, "Fatal: The parameter ngcon cannot be smaller than one.\n");
            printf("Fatal: The parameter ngcon cannot be smaller than one.\n");
            Fatal_Flag = 1;
        }
        if ((model->BSIM4v3ngcon != 1.0) && (model->BSIM4v3ngcon != 2.0))
        {   model->BSIM4v3ngcon = 1.0;
            fprintf(fplog, "Warning: Ngcon must be equal to one or two; reset to 1.0.\n");
            printf("Warning: Ngcon must be equal to one or two; reset to 1.0.\n");
        }

        if (model->BSIM4v3gbmin < 1.0e-20)
        {   fprintf(fplog, "Warning: Gbmin = %g is too small.\n",
                    model->BSIM4v3gbmin);
            printf("Warning: Gbmin = %g is too small.\n", model->BSIM4v3gbmin);
        }

        if (pParam->BSIM4v3noff < 0.1)
        {   fprintf(fplog, "Warning: Noff = %g is too small.\n",
                    pParam->BSIM4v3noff);
            printf("Warning: Noff = %g is too small.\n", pParam->BSIM4v3noff);
        }

        if (pParam->BSIM4v3voffcv < -0.5)
        {   fprintf(fplog, "Warning: Voffcv = %g is too small.\n",
                    pParam->BSIM4v3voffcv);
            printf("Warning: Voffcv = %g is too small.\n", pParam->BSIM4v3voffcv);
        }

        /* Check capacitance parameters */
        if (pParam->BSIM4v3clc < 0.0)
	{   fprintf(fplog, "Fatal: Clc = %g is negative.\n", pParam->BSIM4v3clc);
	    printf("Fatal: Clc = %g is negative.\n", pParam->BSIM4v3clc);
	    Fatal_Flag = 1;
        }      

        if (pParam->BSIM4v3moin < 5.0)
        {   fprintf(fplog, "Warning: Moin = %g is too small.\n",
                    pParam->BSIM4v3moin);
            printf("Warning: Moin = %g is too small.\n", pParam->BSIM4v3moin);
        }
        if (pParam->BSIM4v3moin > 25.0)
        {   fprintf(fplog, "Warning: Moin = %g is too large.\n",
                    pParam->BSIM4v3moin);
            printf("Warning: Moin = %g is too large.\n", pParam->BSIM4v3moin);
        }
	if(model->BSIM4v3capMod ==2) {
        	if (pParam->BSIM4v3acde < 0.1)
        	{   fprintf(fplog, "Warning:  Acde = %g is too small.\n",
                    	pParam->BSIM4v3acde);
            	printf("Warning: Acde = %g is too small.\n", pParam->BSIM4v3acde);
        	}
        	if (pParam->BSIM4v3acde > 1.6)
        	{   fprintf(fplog, "Warning:  Acde = %g is too large.\n",
                    	pParam->BSIM4v3acde);
            	printf("Warning: Acde = %g is too large.\n", pParam->BSIM4v3acde);
        	}
	}

        /* Check overlap capacitance parameters */
        if (pParam->BSIM4v3ckappas < 0.02)
        {   fprintf(fplog, "Warning: ckappas = %g is too small. Set to 0.02\n",
                    pParam->BSIM4v3ckappas);
            printf("Warning: ckappas = %g is too small.\n", pParam->BSIM4v3ckappas);
            pParam->BSIM4v3ckappas = 0.02;
       }
        if (pParam->BSIM4v3ckappad < 0.02)
        {   fprintf(fplog, "Warning: ckappad = %g is too small. Set to 0.02\n",
                    pParam->BSIM4v3ckappad);
            printf("Warning: ckappad = %g is too small.\n", pParam->BSIM4v3ckappad);
            pParam->BSIM4v3ckappad = 0.02;
        }

      if (model->BSIM4v3paramChk ==1)
      {
/* Check L and W parameters */ 
	if (pParam->BSIM4v3leff <= 1.0e-9)
	{   fprintf(fplog, "Warning: Leff = %g <= 1.0e-9. Recommended Leff >= 1e-8 \n", 
		    pParam->BSIM4v3leff);
	    printf("Warning: Leff = %g <= 1.0e-9. Recommended Leff >= 1e-8 \n",
		    pParam->BSIM4v3leff);
	}    
	
	if (pParam->BSIM4v3leffCV <= 1.0e-9)
	{   fprintf(fplog, "Warning: Leff for CV = %g <= 1.0e-9. Recommended LeffCV >=1e-8 \n",
		    pParam->BSIM4v3leffCV);
	    printf("Warning: Leff for CV = %g <= 1.0e-9. Recommended LeffCV >=1e-8 \n",
		    pParam->BSIM4v3leffCV);
	}  
	
        if (pParam->BSIM4v3weff <= 1.0e-9)
	{   fprintf(fplog, "Warning: Weff = %g <= 1.0e-9. Recommended Weff >=1e-7 \n",
		    pParam->BSIM4v3weff);
	    printf("Warning: Weff = %g <= 1.0e-9. Recommended Weff >=1e-7 \n",
		   pParam->BSIM4v3weff);
	}             
	
	if (pParam->BSIM4v3weffCV <= 1.0e-9)
	{   fprintf(fplog, "Warning: Weff for CV = %g <= 1.0e-9. Recommended WeffCV >= 1e-7 \n",
		    pParam->BSIM4v3weffCV);
	    printf("Warning: Weff for CV = %g <= 1.0e-9. Recommended WeffCV >= 1e-7 \n",
		    pParam->BSIM4v3weffCV);
	}        
	
        /* Check threshold voltage parameters */
	if (model->BSIM4v3toxe < 1.0e-10)
	{   fprintf(fplog, "Warning: Toxe = %g is less than 1A. Recommended Toxe >= 5A\n",
	            model->BSIM4v3toxe);
	    printf("Warning: Toxe = %g is less than 1A. Recommended Toxe >= 5A\n", model->BSIM4v3toxe);
        }
        if (model->BSIM4v3toxp < 1.0e-10)
        {   fprintf(fplog, "Warning: Toxp = %g is less than 1A. Recommended Toxp >= 5A\n",
                    model->BSIM4v3toxp);
            printf("Warning: Toxp = %g is less than 1A. Recommended Toxp >= 5A\n", model->BSIM4v3toxp);
        }
        if (model->BSIM4v3toxm < 1.0e-10)
        {   fprintf(fplog, "Warning: Toxm = %g is less than 1A. Recommended Toxm >= 5A\n",
                    model->BSIM4v3toxm);
            printf("Warning: Toxm = %g is less than 1A. Recommended Toxm >= 5A\n", model->BSIM4v3toxm);
        }

        if (pParam->BSIM4v3ndep <= 1.0e12)
	{   fprintf(fplog, "Warning: Ndep = %g may be too small.\n",
	            pParam->BSIM4v3ndep);
	    printf("Warning: Ndep = %g may be too small.\n",
	           pParam->BSIM4v3ndep);
	}
	else if (pParam->BSIM4v3ndep >= 1.0e21)
	{   fprintf(fplog, "Warning: Ndep = %g may be too large.\n",
	            pParam->BSIM4v3ndep);
	    printf("Warning: Ndep = %g may be too large.\n",
	           pParam->BSIM4v3ndep);
	}

	 if (pParam->BSIM4v3nsub <= 1.0e14)
	{   fprintf(fplog, "Warning: Nsub = %g may be too small.\n",
	            pParam->BSIM4v3nsub);
	    printf("Warning: Nsub = %g may be too small.\n",
	           pParam->BSIM4v3nsub);
	}
	else if (pParam->BSIM4v3nsub >= 1.0e21)
	{   fprintf(fplog, "Warning: Nsub = %g may be too large.\n",
	            pParam->BSIM4v3nsub);
	    printf("Warning: Nsub = %g may be too large.\n",
	           pParam->BSIM4v3nsub);
	}

	if ((pParam->BSIM4v3ngate > 0.0) &&
	    (pParam->BSIM4v3ngate <= 1.e18))
	{   fprintf(fplog, "Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	            pParam->BSIM4v3ngate);
	    printf("Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	           pParam->BSIM4v3ngate);
	}
       
        if (pParam->BSIM4v3dvt0 < 0.0)
	{   fprintf(fplog, "Warning: Dvt0 = %g is negative.\n",
		    pParam->BSIM4v3dvt0);   
	    printf("Warning: Dvt0 = %g is negative.\n", pParam->BSIM4v3dvt0);   
	}
	    
	if (fabs(1.0e-6 / (pParam->BSIM4v3w0 + pParam->BSIM4v3weff)) > 10.0)
	{   fprintf(fplog, "Warning: (W0 + Weff) may be too small.\n");
	    printf("Warning: (W0 + Weff) may be too small.\n");
        }

/* Check subthreshold parameters */
	if (pParam->BSIM4v3nfactor < 0.0)
	{   fprintf(fplog, "Warning: Nfactor = %g is negative.\n",
		    pParam->BSIM4v3nfactor);
	    printf("Warning: Nfactor = %g is negative.\n", pParam->BSIM4v3nfactor);
	}
	if (pParam->BSIM4v3cdsc < 0.0)
	{   fprintf(fplog, "Warning: Cdsc = %g is negative.\n",
		    pParam->BSIM4v3cdsc);
	    printf("Warning: Cdsc = %g is negative.\n", pParam->BSIM4v3cdsc);
	}
	if (pParam->BSIM4v3cdscd < 0.0)
	{   fprintf(fplog, "Warning: Cdscd = %g is negative.\n",
		    pParam->BSIM4v3cdscd);
	    printf("Warning: Cdscd = %g is negative.\n", pParam->BSIM4v3cdscd);
	}
/* Check DIBL parameters */
	if (pParam->BSIM4v3eta0 < 0.0)
	{   fprintf(fplog, "Warning: Eta0 = %g is negative.\n",
		    pParam->BSIM4v3eta0); 
	    printf("Warning: Eta0 = %g is negative.\n", pParam->BSIM4v3eta0); 
	}
	      
/* Check Abulk parameters */	    
	 if (fabs(1.0e-6 / (pParam->BSIM4v3b1 + pParam->BSIM4v3weff)) > 10.0)
       	{   fprintf(fplog, "Warning: (B1 + Weff) may be too small.\n");
       	    printf("Warning: (B1 + Weff) may be too small.\n");
        }    
    

/* Check Saturation parameters */
     	if (pParam->BSIM4v3a2 < 0.01)
	{   fprintf(fplog, "Warning: A2 = %g is too small. Set to 0.01.\n", pParam->BSIM4v3a2);
	    printf("Warning: A2 = %g is too small. Set to 0.01.\n",
		   pParam->BSIM4v3a2);
	    pParam->BSIM4v3a2 = 0.01;
	}
	else if (pParam->BSIM4v3a2 > 1.0)
	{   fprintf(fplog, "Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		    pParam->BSIM4v3a2);
	    printf("Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		   pParam->BSIM4v3a2);
	    pParam->BSIM4v3a2 = 1.0;
	    pParam->BSIM4v3a1 = 0.0;

	}

        if (pParam->BSIM4v3prwg < 0.0)
        {   fprintf(fplog, "Warning: Prwg = %g is negative. Set to zero.\n",
                    pParam->BSIM4v3prwg);
            printf("Warning: Prwg = %g is negative. Set to zero.\n",
                   pParam->BSIM4v3prwg);
            pParam->BSIM4v3prwg = 0.0;
        }
	if (pParam->BSIM4v3rdsw < 0.0)
	{   fprintf(fplog, "Warning: Rdsw = %g is negative. Set to zero.\n",
		    pParam->BSIM4v3rdsw);
	    printf("Warning: Rdsw = %g is negative. Set to zero.\n",
		   pParam->BSIM4v3rdsw);
	    pParam->BSIM4v3rdsw = 0.0;
	    pParam->BSIM4v3rds0 = 0.0;
	}
	if (pParam->BSIM4v3rds0 < 0.0)
	{   fprintf(fplog, "Warning: Rds at current temperature = %g is negative. Set to zero.\n",
		    pParam->BSIM4v3rds0);
	    printf("Warning: Rds at current temperature = %g is negative. Set to zero.\n",
		   pParam->BSIM4v3rds0);
	    pParam->BSIM4v3rds0 = 0.0;
	}

        if (pParam->BSIM4v3rdswmin < 0.0)
        {   fprintf(fplog, "Warning: Rdswmin at current temperature = %g is negative. Set to zero.\n",
                    pParam->BSIM4v3rdswmin);
            printf("Warning: Rdswmin at current temperature = %g is negative. Set to zero.\n",
                   pParam->BSIM4v3rdswmin);
            pParam->BSIM4v3rdswmin = 0.0;
        }

        if (pParam->BSIM4v3vsattemp < 1.0e3)
	{   fprintf(fplog, "Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM4v3vsattemp);
	   printf("Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM4v3vsattemp);
	}

      if((model->BSIM4v3lambdaGiven) && (pParam->BSIM4v3lambda > 0.0) ) 
      {
        if (pParam->BSIM4v3lambda > 1.0e-9)
	{   fprintf(fplog, "Warning: Lambda = %g may be too large.\n", pParam->BSIM4v3lambda);
	   printf("Warning: Lambda = %g may be too large.\n", pParam->BSIM4v3lambda);
	}
      }

      if((model->BSIM4v3vtlGiven) && (pParam->BSIM4v3vtl > 0.0) )
      {  
        if (pParam->BSIM4v3vtl < 6.0e4)
	{   fprintf(fplog, "Warning: Thermal velocity vtl = %g may be too small.\n", pParam->BSIM4v3vtl);
	   printf("Warning: Thermal velocity vtl = %g may be too small.\n", pParam->BSIM4v3vtl);
	}

        if (pParam->BSIM4v3xn < 3.0)
	{   fprintf(fplog, "Warning: back scattering coeff xn = %g is too small.\n", pParam->BSIM4v3xn);
	    printf("Warning: back scattering coeff xn = %g is too small. Reset to 3.0 \n", pParam->BSIM4v3xn);
	    pParam->BSIM4v3xn = 3.0;
	   
	}
        if (model->BSIM4v3lc < 0.0)
	{   fprintf(fplog, "Warning: back scattering coeff lc = %g is too small.\n", model->BSIM4v3lc);
	    printf("Warning: back scattering coeff lc = %g is too small. Reset to 0.0\n", model->BSIM4v3lc);
	    pParam->BSIM4v3lc = 0.0;
	}
      }

        if (pParam->BSIM4v3fprout < 0.0)
        {   fprintf(fplog, "Fatal: fprout = %g is negative.\n",
                    pParam->BSIM4v3fprout);
            printf("Fatal: fprout = %g is negative.\n", pParam->BSIM4v3fprout);
	    Fatal_Flag = 1;
        }
        if (pParam->BSIM4v3pdits < 0.0)
        {   fprintf(fplog, "Fatal: pdits = %g is negative.\n",
                    pParam->BSIM4v3pdits);
            printf("Fatal: pdits = %g is negative.\n", pParam->BSIM4v3pdits);
            Fatal_Flag = 1;
        }
        if (model->BSIM4v3pditsl < 0.0)
        {   fprintf(fplog, "Fatal: pditsl = %g is negative.\n",
                    model->BSIM4v3pditsl);
            printf("Fatal: pditsl = %g is negative.\n", model->BSIM4v3pditsl);
            Fatal_Flag = 1;
        }

	if (pParam->BSIM4v3pdibl1 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl1 = %g is negative.\n",
		    pParam->BSIM4v3pdibl1);
	    printf("Warning: Pdibl1 = %g is negative.\n", pParam->BSIM4v3pdibl1);
	}
	if (pParam->BSIM4v3pdibl2 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl2 = %g is negative.\n",
		    pParam->BSIM4v3pdibl2);
	    printf("Warning: Pdibl2 = %g is negative.\n", pParam->BSIM4v3pdibl2);
	}

        if (pParam->BSIM4v3nigbinv <= 0.0)
        {   fprintf(fplog, "Fatal: nigbinv = %g is non-positive.\n",
                    pParam->BSIM4v3nigbinv);
            printf("Fatal: nigbinv = %g is non-positive.\n", pParam->BSIM4v3nigbinv);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v3nigbacc <= 0.0)
        {   fprintf(fplog, "Fatal: nigbacc = %g is non-positive.\n",
                    pParam->BSIM4v3nigbacc);
            printf("Fatal: nigbacc = %g is non-positive.\n", pParam->BSIM4v3nigbacc);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v3nigc <= 0.0)
        {   fprintf(fplog, "Fatal: nigc = %g is non-positive.\n",
                    pParam->BSIM4v3nigc);
            printf("Fatal: nigc = %g is non-positive.\n", pParam->BSIM4v3nigc);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v3poxedge <= 0.0)
        {   fprintf(fplog, "Fatal: poxedge = %g is non-positive.\n",
                    pParam->BSIM4v3poxedge);
            printf("Fatal: poxedge = %g is non-positive.\n", pParam->BSIM4v3poxedge);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v3pigcd <= 0.0)
        {   fprintf(fplog, "Fatal: pigcd = %g is non-positive.\n",
                    pParam->BSIM4v3pigcd);
            printf("Fatal: pigcd = %g is non-positive.\n", pParam->BSIM4v3pigcd);
            Fatal_Flag = 1;
        }
        
/* Check gate resistance parameters */        
        if (here->BSIM4v3rgateMod == 1)
        {   if (model->BSIM4v3rshg <= 0.0)
	        printf("Warning: rshg should be positive for rgateMod = 1.\n");
	}
        else if (here->BSIM4v3rgateMod == 2)
        {   if (model->BSIM4v3rshg <= 0.0)
                printf("Warning: rshg <= 0.0 for rgateMod = 2.\n");
            else if (pParam->BSIM4v3xrcrg1 <= 0.0)
                     printf("Warning: xrcrg1 <= 0.0 for rgateMod = 2.\n");
         }
        if (here->BSIM4v3rgateMod == 3)
        {   if (model->BSIM4v3rshg <= 0.0)
                printf("Warning: rshg should be positive for rgateMod = 3.\n");
            else if (pParam->BSIM4v3xrcrg1 <= 0.0)
                     printf("Warning: xrcrg1 should be positive for rgateMod = 3.\n");
         }
         
/* Check overlap capacitance parameters */
        if (model->BSIM4v3cgdo < 0.0)
	{   fprintf(fplog, "Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM4v3cgdo);
	    printf("Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM4v3cgdo);
	    model->BSIM4v3cgdo = 0.0;
        }      
        if (model->BSIM4v3cgso < 0.0)
	{   fprintf(fplog, "Warning: cgso = %g is negative. Set to zero.\n", model->BSIM4v3cgso);
	    printf("Warning: cgso = %g is negative. Set to zero.\n", model->BSIM4v3cgso);
	    model->BSIM4v3cgso = 0.0;
        }      

        if (model->BSIM4v3tnoia < 0.0)
        {   fprintf(fplog, "Warning: tnoia = %g is negative. Set to zero.\n", model->BSIM4v3tnoia);
            printf("Warning: tnoia = %g is negative. Set to zero.\n", model->BSIM4v3tnoia);
            model->BSIM4v3tnoia = 0.0;
        }
        if (model->BSIM4v3tnoib < 0.0)
        {   fprintf(fplog, "Warning: tnoib = %g is negative. Set to zero.\n", model->BSIM4v3tnoib);
            printf("Warning: tnoib = %g is negative. Set to zero.\n", model->BSIM4v3tnoib);
            model->BSIM4v3tnoib = 0.0;
        }

         if (model->BSIM4v3rnoia < 0.0)
        {   fprintf(fplog, "Warning: rnoia = %g is negative. Set to zero.\n", model->BSIM4v3rnoia);
            printf("Warning: rnoia = %g is negative. Set to zero.\n", model->BSIM4v3rnoia);
            model->BSIM4v3rnoia = 0.0;
        }
        if (model->BSIM4v3rnoib < 0.0)
        {   fprintf(fplog, "Warning: rnoib = %g is negative. Set to zero.\n", model->BSIM4v3rnoib);
            printf("Warning: rnoib = %g is negative. Set to zero.\n", model->BSIM4v3rnoib);
            model->BSIM4v3rnoib = 0.0;
        }

       if (model->BSIM4v3ntnoi < 0.0)
        {   fprintf(fplog, "Warning: ntnoi = %g is negative. Set to zero.\n", model->BSIM4v3ntnoi);
            printf("Warning: ntnoi = %g is negative. Set to zero.\n", model->BSIM4v3ntnoi);
            model->BSIM4v3ntnoi = 0.0;
        }

     }/* loop for the parameter check for warning messages */      
	fclose(fplog);
    }
    else
    {   fprintf(stderr, "Warning: Can't open log file. Parameter checking skipped.\n");
    }

    return(Fatal_Flag);
}

