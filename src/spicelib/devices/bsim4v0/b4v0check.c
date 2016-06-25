/**** BSIM4.0.0, Released by Weidong Liu 3/24/2000 ****/

/**********
 * Copyright 2000 Regents of the University of California. All rights reserved.
 * File: b4check.c of BSIM4.0.0.
 * Authors: Weidong Liu, Kanyu M. Cao, Xiaodong Jin, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v0def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"

int
BSIM4v0checkModel(model, here, ckt)
register BSIM4v0model *model;
register BSIM4v0instance *here;
CKTcircuit *ckt;
{
struct bsim4v0SizeDependParam *pParam;
int Fatal_Flag = 0;
FILE *fplog;
    
    if ((fplog = fopen("bsim4v0.out", "w")) != NULL)
    {   pParam = here->pParam;
        fprintf(fplog, "BSIM4v0: Berkeley Short Channel IGFET Model-4\n");
        fprintf(fplog, "Developed by Dr. Weidong Liu, Xiaodong Jin, Kanyu M. Cao and Prof. Chenming Hu in 2000.\n");
        fprintf(fplog, "\n");
	fprintf(fplog, "++++++++++ BSIM4v0 PARAMETER CHECKING BELOW ++++++++++\n");

        if ((strcmp(model->BSIM4v0version, "4.0.0")) && (strcmp(model->BSIM4v0version, "4.00")) && (strcmp(model->BSIM4v0version, "4.0")))
        {  fprintf(fplog, "Warning: This model is BSIM4.0.0; you specified a wrong version number '%s'.\n", model->BSIM4v0version);
           printf("Warning: This model is BSIM4.0.0; you specified a wrong version number '%s'.\n", model->BSIM4v0version);
        }
	fprintf(fplog, "Model = %s\n", model->BSIM4v0modName);


        if ((here->BSIM4v0rgateMod == 2) || (here->BSIM4v0rgateMod == 3))
        {   if ((here->BSIM4v0trnqsMod == 1) || (here->BSIM4v0acnqsMod == 1))
            fprintf(fplog, "Warning: You've selected both Rg and charge deficit NQS; select one only.\n");
            printf("Warning: You've selected both Rg and charge deficit NQS; select one only.\n");
        }


	if (model->BSIM4v0toxe <= 0.0)
	{   fprintf(fplog, "Fatal: Toxe = %g is not positive.\n",
		    model->BSIM4v0toxe);
	    printf("Fatal: Toxe = %g is not positive.\n", model->BSIM4v0toxe);
	    Fatal_Flag = 1;
	}
        if (model->BSIM4v0toxp <= 0.0)
        {   fprintf(fplog, "Fatal: Toxp = %g is not positive.\n",
                    model->BSIM4v0toxp);
            printf("Fatal: Toxp = %g is not positive.\n", model->BSIM4v0toxp);
            Fatal_Flag = 1;
        }

        if (model->BSIM4v0toxm <= 0.0)
        {   fprintf(fplog, "Fatal: Toxm = %g is not positive.\n",
                    model->BSIM4v0toxm);
            printf("Fatal: Toxm = %g is not positive.\n", model->BSIM4v0toxm);
            Fatal_Flag = 1;
        }

        if (model->BSIM4v0toxref <= 0.0)
        {   fprintf(fplog, "Fatal: Toxref = %g is not positive.\n",
                    model->BSIM4v0toxref);
            printf("Fatal: Toxref = %g is not positive.\n", model->BSIM4v0toxref);
            Fatal_Flag = 1;
        }

        if (pParam->BSIM4v0lpe0 < -pParam->BSIM4v0leff)
        {   fprintf(fplog, "Fatal: Lpe0 = %g is less than -Leff.\n",
                    pParam->BSIM4v0lpe0);
            printf("Fatal: Lpe0 = %g is less than -Leff.\n",
                        pParam->BSIM4v0lpe0);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v0lpeb < -pParam->BSIM4v0leff)
        {   fprintf(fplog, "Fatal: Lpeb = %g is less than -Leff.\n",
                    pParam->BSIM4v0lpeb);
            printf("Fatal: Lpeb = %g is less than -Leff.\n",
                        pParam->BSIM4v0lpeb);
            Fatal_Flag = 1;
        }

        if (pParam->BSIM4v0phin < -0.4)
        {   fprintf(fplog, "Fatal: Phin = %g is less than -0.4.\n",
                    pParam->BSIM4v0phin);
            printf("Fatal: Phin = %g is less than -0.4.\n",
                   pParam->BSIM4v0phin);
            Fatal_Flag = 1;
        }
	if (pParam->BSIM4v0ndep <= 0.0)
	{   fprintf(fplog, "Fatal: Ndep = %g is not positive.\n",
		    pParam->BSIM4v0ndep);
	    printf("Fatal: Ndep = %g is not positive.\n",
		   pParam->BSIM4v0ndep);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v0nsub <= 0.0)
	{   fprintf(fplog, "Fatal: Nsub = %g is not positive.\n",
		    pParam->BSIM4v0nsub);
	    printf("Fatal: Nsub = %g is not positive.\n",
		   pParam->BSIM4v0nsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v0ngate < 0.0)
	{   fprintf(fplog, "Fatal: Ngate = %g is not positive.\n",
		    pParam->BSIM4v0ngate);
	    printf("Fatal: Ngate = %g Ngate is not positive.\n",
		   pParam->BSIM4v0ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v0ngate > 1.e25)
	{   fprintf(fplog, "Fatal: Ngate = %g is too high.\n",
		    pParam->BSIM4v0ngate);
	    printf("Fatal: Ngate = %g Ngate is too high\n",
		   pParam->BSIM4v0ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v0xj <= 0.0)
	{   fprintf(fplog, "Fatal: Xj = %g is not positive.\n",
		    pParam->BSIM4v0xj);
	    printf("Fatal: Xj = %g is not positive.\n", pParam->BSIM4v0xj);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4v0dvt1 < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1 = %g is negative.\n",
		    pParam->BSIM4v0dvt1);   
	    printf("Fatal: Dvt1 = %g is negative.\n", pParam->BSIM4v0dvt1);   
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM4v0dvt1w < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1w = %g is negative.\n",
		    pParam->BSIM4v0dvt1w);
	    printf("Fatal: Dvt1w = %g is negative.\n", pParam->BSIM4v0dvt1w);
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM4v0w0 == -pParam->BSIM4v0weff)
	{   fprintf(fplog, "Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }   

	if (pParam->BSIM4v0dsub < 0.0)
	{   fprintf(fplog, "Fatal: Dsub = %g is negative.\n", pParam->BSIM4v0dsub);
	    printf("Fatal: Dsub = %g is negative.\n", pParam->BSIM4v0dsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4v0b1 == -pParam->BSIM4v0weff)
	{   fprintf(fplog, "Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }  
        if (pParam->BSIM4v0u0temp <= 0.0)
	{   fprintf(fplog, "Fatal: u0 at current temperature = %g is not positive.\n", pParam->BSIM4v0u0temp);
	    printf("Fatal: u0 at current temperature = %g is not positive.\n",
		   pParam->BSIM4v0u0temp);
	    Fatal_Flag = 1;
        }
    
        if (pParam->BSIM4v0delta < 0.0)
	{   fprintf(fplog, "Fatal: Delta = %g is less than zero.\n",
		    pParam->BSIM4v0delta);
	    printf("Fatal: Delta = %g is less than zero.\n", pParam->BSIM4v0delta);
	    Fatal_Flag = 1;
        }      

	if (pParam->BSIM4v0vsattemp <= 0.0)
	{   fprintf(fplog, "Fatal: Vsat at current temperature = %g is not positive.\n", pParam->BSIM4v0vsattemp);
	    printf("Fatal: Vsat at current temperature = %g is not positive.\n",
		   pParam->BSIM4v0vsattemp);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4v0pclm <= 0.0)
	{   fprintf(fplog, "Fatal: Pclm = %g is not positive.\n", pParam->BSIM4v0pclm);
	    printf("Fatal: Pclm = %g is not positive.\n", pParam->BSIM4v0pclm);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4v0drout < 0.0)
	{   fprintf(fplog, "Fatal: Drout = %g is negative.\n", pParam->BSIM4v0drout);
	    printf("Fatal: Drout = %g is negative.\n", pParam->BSIM4v0drout);
	    Fatal_Flag = 1;
	}

        if (pParam->BSIM4v0pscbe2 <= 0.0)
        {   fprintf(fplog, "Warning: Pscbe2 = %g is not positive.\n",
                    pParam->BSIM4v0pscbe2);
            printf("Warning: Pscbe2 = %g is not positive.\n", pParam->BSIM4v0pscbe2);
        }

        if (here->BSIM4v0nf < 1.0)
        {   fprintf(fplog, "Fatal: Number of finger = %g is smaller than one.\n", here->BSIM4v0nf);
            printf("Fatal: Number of finger = %g is smaller than one.\n", here->BSIM4v0nf);
            Fatal_Flag = 1;
        }
        if (here->BSIM4v0nf > 500.0)
        {   here->BSIM4v0nf = 20.0;
            fprintf(fplog, "Warning: Nf = %g is too large; reset to 20.0.\n", here->BSIM4v0nf);
            printf("Warning: Nf = %g is too large; reset to 20.0.\n", here->BSIM4v0nf);
        }

        if (here->BSIM4v0l <= model->BSIM4v0xgl)
        {   fprintf(fplog, "Fatal: The parameter xgl must be smaller than Ldrawn.\n");
            printf("Fatal: The parameter xgl must be smaller than Ldrawn.\n");
            Fatal_Flag = 1;
        }
        if (model->BSIM4v0ngcon < 1.0)
        {   fprintf(fplog, "Fatal: The parameter ngcon cannot be smaller than one.\n");
            printf("Fatal: The parameter ngcon cannot be smaller than one.\n");
            Fatal_Flag = 1;
        }
        if ((model->BSIM4v0ngcon != 1.0) && (model->BSIM4v0ngcon != 2.0))
        {   model->BSIM4v0ngcon = 1.0;
            fprintf(fplog, "Warning: Ngcon must be equal to one or two; reset to 1.0.\n");
            printf("Warning: Ngcon must be equal to one or two; reset to 1.0.\n");
        }

        if (model->BSIM4v0gbmin < 1.0e-20)
        {   fprintf(fplog, "Warning: Gbmin = %g is too small.\n",
                    model->BSIM4v0gbmin);
            printf("Warning: Gbmin = %g is too small.\n", model->BSIM4v0gbmin);
        }

        if (pParam->BSIM4v0noff < 0.1)
        {   fprintf(fplog, "Warning: Noff = %g is too small.\n",
                    pParam->BSIM4v0noff);
            printf("Warning: Noff = %g is too small.\n", pParam->BSIM4v0noff);
        }
        if (pParam->BSIM4v0noff > 4.0)
        {   fprintf(fplog, "Warning: Noff = %g is too large.\n",
                    pParam->BSIM4v0noff);
            printf("Warning: Noff = %g is too large.\n", pParam->BSIM4v0noff);
        }

        if (pParam->BSIM4v0voffcv < -0.5)
        {   fprintf(fplog, "Warning: Voffcv = %g is too small.\n",
                    pParam->BSIM4v0voffcv);
            printf("Warning: Voffcv = %g is too small.\n", pParam->BSIM4v0voffcv);
        }
        if (pParam->BSIM4v0voffcv > 0.5)
        {   fprintf(fplog, "Warning: Voffcv = %g is too large.\n",
                    pParam->BSIM4v0voffcv);
            printf("Warning: Voffcv = %g is too large.\n", pParam->BSIM4v0voffcv);
        }

        /* Check capacitance parameters */
        if (pParam->BSIM4v0clc < 0.0)
	{   fprintf(fplog, "Fatal: Clc = %g is negative.\n", pParam->BSIM4v0clc);
	    printf("Fatal: Clc = %g is negative.\n", pParam->BSIM4v0clc);
	    Fatal_Flag = 1;
        }      

        if (pParam->BSIM4v0moin < 5.0)
        {   fprintf(fplog, "Warning: Moin = %g is too small.\n",
                    pParam->BSIM4v0moin);
            printf("Warning: Moin = %g is too small.\n", pParam->BSIM4v0moin);
        }
        if (pParam->BSIM4v0moin > 25.0)
        {   fprintf(fplog, "Warning: Moin = %g is too large.\n",
                    pParam->BSIM4v0moin);
            printf("Warning: Moin = %g is too large.\n", pParam->BSIM4v0moin);
        }

        if (pParam->BSIM4v0acde < 0.4)
        {   fprintf(fplog, "Warning:  Acde = %g is too small.\n",
                    pParam->BSIM4v0acde);
            printf("Warning: Acde = %g is too small.\n", pParam->BSIM4v0acde);
        }
        if (pParam->BSIM4v0acde > 1.6)
        {   fprintf(fplog, "Warning:  Acde = %g is too large.\n",
                    pParam->BSIM4v0acde);
            printf("Warning: Acde = %g is too large.\n", pParam->BSIM4v0acde);
        }

      if (model->BSIM4v0paramChk ==1)
      {
/* Check L and W parameters */ 
	if (pParam->BSIM4v0leff <= 5.0e-8)
	{   fprintf(fplog, "Warning: Leff = %g may be too small.\n",
	            pParam->BSIM4v0leff);
	    printf("Warning: Leff = %g may be too small.\n",
		    pParam->BSIM4v0leff);
	}    
	
	if (pParam->BSIM4v0leffCV <= 5.0e-8)
	{   fprintf(fplog, "Warning: Leff for CV = %g may be too small.\n",
		    pParam->BSIM4v0leffCV);
	    printf("Warning: Leff for CV = %g may be too small.\n",
		   pParam->BSIM4v0leffCV);
	}  
	
        if (pParam->BSIM4v0weff <= 1.0e-7)
	{   fprintf(fplog, "Warning: Weff = %g may be too small.\n",
		    pParam->BSIM4v0weff);
	    printf("Warning: Weff = %g may be too small.\n",
		   pParam->BSIM4v0weff);
	}             
	
	if (pParam->BSIM4v0weffCV <= 1.0e-7)
	{   fprintf(fplog, "Warning: Weff for CV = %g may be too small.\n",
		    pParam->BSIM4v0weffCV);
	    printf("Warning: Weff for CV = %g may be too small.\n",
		   pParam->BSIM4v0weffCV);
	}        
	
        /* Check threshold voltage parameters */
	if (model->BSIM4v0toxe < 1.0e-9)
	{   fprintf(fplog, "Warning: Toxe = %g is less than 10A.\n",
	            model->BSIM4v0toxe);
	    printf("Warning: Toxe = %g is less than 10A.\n", model->BSIM4v0toxe);
        }
        if (model->BSIM4v0toxp < 1.0e-9)
        {   fprintf(fplog, "Warning: Toxp = %g is less than 10A.\n",
                    model->BSIM4v0toxp);
            printf("Warning: Toxp = %g is less than 10A.\n", model->BSIM4v0toxp);
        }
        if (model->BSIM4v0toxm < 1.0e-9)
        {   fprintf(fplog, "Warning: Toxm = %g is less than 10A.\n",
                    model->BSIM4v0toxm);
            printf("Warning: Toxm = %g is less than 10A.\n", model->BSIM4v0toxm);
        }

        if (pParam->BSIM4v0ndep <= 1.0e12)
	{   fprintf(fplog, "Warning: Ndep = %g may be too small.\n",
	            pParam->BSIM4v0ndep);
	    printf("Warning: Ndep = %g may be too small.\n",
	           pParam->BSIM4v0ndep);
	}
	else if (pParam->BSIM4v0ndep >= 1.0e21)
	{   fprintf(fplog, "Warning: Ndep = %g may be too large.\n",
	            pParam->BSIM4v0ndep);
	    printf("Warning: Ndep = %g may be too large.\n",
	           pParam->BSIM4v0ndep);
	}

	 if (pParam->BSIM4v0nsub <= 1.0e14)
	{   fprintf(fplog, "Warning: Nsub = %g may be too small.\n",
	            pParam->BSIM4v0nsub);
	    printf("Warning: Nsub = %g may be too small.\n",
	           pParam->BSIM4v0nsub);
	}
	else if (pParam->BSIM4v0nsub >= 1.0e21)
	{   fprintf(fplog, "Warning: Nsub = %g may be too large.\n",
	            pParam->BSIM4v0nsub);
	    printf("Warning: Nsub = %g may be too large.\n",
	           pParam->BSIM4v0nsub);
	}

	if ((pParam->BSIM4v0ngate > 0.0) &&
	    (pParam->BSIM4v0ngate <= 1.e18))
	{   fprintf(fplog, "Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	            pParam->BSIM4v0ngate);
	    printf("Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	           pParam->BSIM4v0ngate);
	}
       
        if (pParam->BSIM4v0dvt0 < 0.0)
	{   fprintf(fplog, "Warning: Dvt0 = %g is negative.\n",
		    pParam->BSIM4v0dvt0);   
	    printf("Warning: Dvt0 = %g is negative.\n", pParam->BSIM4v0dvt0);   
	}
	    
	if (fabs(1.0e-6 / (pParam->BSIM4v0w0 + pParam->BSIM4v0weff)) > 10.0)
	{   fprintf(fplog, "Warning: (W0 + Weff) may be too small.\n");
	    printf("Warning: (W0 + Weff) may be too small.\n");
        }

/* Check subthreshold parameters */
	if (pParam->BSIM4v0nfactor < 0.0)
	{   fprintf(fplog, "Warning: Nfactor = %g is negative.\n",
		    pParam->BSIM4v0nfactor);
	    printf("Warning: Nfactor = %g is negative.\n", pParam->BSIM4v0nfactor);
	}
	if (pParam->BSIM4v0cdsc < 0.0)
	{   fprintf(fplog, "Warning: Cdsc = %g is negative.\n",
		    pParam->BSIM4v0cdsc);
	    printf("Warning: Cdsc = %g is negative.\n", pParam->BSIM4v0cdsc);
	}
	if (pParam->BSIM4v0cdscd < 0.0)
	{   fprintf(fplog, "Warning: Cdscd = %g is negative.\n",
		    pParam->BSIM4v0cdscd);
	    printf("Warning: Cdscd = %g is negative.\n", pParam->BSIM4v0cdscd);
	}
/* Check DIBL parameters */
	if (pParam->BSIM4v0eta0 < 0.0)
	{   fprintf(fplog, "Warning: Eta0 = %g is negative.\n",
		    pParam->BSIM4v0eta0); 
	    printf("Warning: Eta0 = %g is negative.\n", pParam->BSIM4v0eta0); 
	}
	      
/* Check Abulk parameters */	    
	 if (fabs(1.0e-6 / (pParam->BSIM4v0b1 + pParam->BSIM4v0weff)) > 10.0)
       	{   fprintf(fplog, "Warning: (B1 + Weff) may be too small.\n");
       	    printf("Warning: (B1 + Weff) may be too small.\n");
        }    
    

/* Check Saturation parameters */
     	if (pParam->BSIM4v0a2 < 0.01)
	{   fprintf(fplog, "Warning: A2 = %g is too small. Set to 0.01.\n", pParam->BSIM4v0a2);
	    printf("Warning: A2 = %g is too small. Set to 0.01.\n",
		   pParam->BSIM4v0a2);
	    pParam->BSIM4v0a2 = 0.01;
	}
	else if (pParam->BSIM4v0a2 > 1.0)
	{   fprintf(fplog, "Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		    pParam->BSIM4v0a2);
	    printf("Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		   pParam->BSIM4v0a2);
	    pParam->BSIM4v0a2 = 1.0;
	    pParam->BSIM4v0a1 = 0.0;

	}

        if (pParam->BSIM4v0prwg < 0.0)
        {   fprintf(fplog, "Warning: Prwg = %g is negative. Set to zero.\n",
                    pParam->BSIM4v0prwg);
            printf("Warning: Prwg = %g is negative. Set to zero.\n",
                   pParam->BSIM4v0prwg);
            pParam->BSIM4v0prwg = 0.0;
        }
	if (pParam->BSIM4v0rdsw < 0.0)
	{   fprintf(fplog, "Warning: Rdsw = %g is negative. Set to zero.\n",
		    pParam->BSIM4v0rdsw);
	    printf("Warning: Rdsw = %g is negative. Set to zero.\n",
		   pParam->BSIM4v0rdsw);
	    pParam->BSIM4v0rdsw = 0.0;
	    pParam->BSIM4v0rds0 = 0.0;
	}
	if (pParam->BSIM4v0rds0 < 0.0)
	{   fprintf(fplog, "Warning: Rds at current temperature = %g is negative. Set to zero.\n",
		    pParam->BSIM4v0rds0);
	    printf("Warning: Rds at current temperature = %g is negative. Set to zero.\n",
		   pParam->BSIM4v0rds0);
	    pParam->BSIM4v0rds0 = 0.0;
	}

        if (pParam->BSIM4v0rdswmin < 0.0)
        {   fprintf(fplog, "Warning: Rdswmin at current temperature = %g is negative. Set to zero.\n",
                    pParam->BSIM4v0rdswmin);
            printf("Warning: Rdswmin at current temperature = %g is negative. Set to zero.\n",
                   pParam->BSIM4v0rdswmin);
            pParam->BSIM4v0rdswmin = 0.0;
        }

        if (pParam->BSIM4v0vsattemp < 1.0e3)
	{   fprintf(fplog, "Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM4v0vsattemp);
	   printf("Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM4v0vsattemp);
	}

        if (pParam->BSIM4v0fprout < 0.0)
        {   fprintf(fplog, "Fatal: fprout = %g is negative.\n",
                    pParam->BSIM4v0fprout);
            printf("Fatal: fprout = %g is negative.\n", pParam->BSIM4v0fprout);
	    Fatal_Flag = 1;
        }
        if (pParam->BSIM4v0pdits < 0.0)
        {   fprintf(fplog, "Fatal: pdits = %g is negative.\n",
                    pParam->BSIM4v0pdits);
            printf("Fatal: pdits = %g is negative.\n", pParam->BSIM4v0pdits);
            Fatal_Flag = 1;
        }
        if (model->BSIM4v0pditsl < 0.0)
        {   fprintf(fplog, "Fatal: pditsl = %g is negative.\n",
                    model->BSIM4v0pditsl);
            printf("Fatal: pditsl = %g is negative.\n", model->BSIM4v0pditsl);
            Fatal_Flag = 1;
        }

	if (pParam->BSIM4v0pdibl1 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl1 = %g is negative.\n",
		    pParam->BSIM4v0pdibl1);
	    printf("Warning: Pdibl1 = %g is negative.\n", pParam->BSIM4v0pdibl1);
	}
	if (pParam->BSIM4v0pdibl2 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl2 = %g is negative.\n",
		    pParam->BSIM4v0pdibl2);
	    printf("Warning: Pdibl2 = %g is negative.\n", pParam->BSIM4v0pdibl2);
	}

        if (pParam->BSIM4v0nigbinv <= 0.0)
        {   fprintf(fplog, "Fatal: nigbinv = %g is non-positive.\n",
                    pParam->BSIM4v0nigbinv);
            printf("Fatal: nigbinv = %g is non-positive.\n", pParam->BSIM4v0nigbinv);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v0nigbacc <= 0.0)
        {   fprintf(fplog, "Fatal: nigbacc = %g is non-positive.\n",
                    pParam->BSIM4v0nigbacc);
            printf("Fatal: nigbacc = %g is non-positive.\n", pParam->BSIM4v0nigbacc);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v0nigc <= 0.0)
        {   fprintf(fplog, "Fatal: nigc = %g is non-positive.\n",
                    pParam->BSIM4v0nigc);
            printf("Fatal: nigc = %g is non-positive.\n", pParam->BSIM4v0nigc);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v0poxedge <= 0.0)
        {   fprintf(fplog, "Fatal: poxedge = %g is non-positive.\n",
                    pParam->BSIM4v0poxedge);
            printf("Fatal: poxedge = %g is non-positive.\n", pParam->BSIM4v0poxedge);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4v0pigcd <= 0.0)
        {   fprintf(fplog, "Fatal: pigcd = %g is non-positive.\n",
                    pParam->BSIM4v0pigcd);
            printf("Fatal: pigcd = %g is non-positive.\n", pParam->BSIM4v0pigcd);
            Fatal_Flag = 1;
        }


/* Check overlap capacitance parameters */
        if (model->BSIM4v0cgdo < 0.0)
	{   fprintf(fplog, "Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM4v0cgdo);
	    printf("Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM4v0cgdo);
	    model->BSIM4v0cgdo = 0.0;
        }      
        if (model->BSIM4v0cgso < 0.0)
	{   fprintf(fplog, "Warning: cgso = %g is negative. Set to zero.\n", model->BSIM4v0cgso);
	    printf("Warning: cgso = %g is negative. Set to zero.\n", model->BSIM4v0cgso);
	    model->BSIM4v0cgso = 0.0;
        }      

        if (model->BSIM4v0tnoia < 0.0)
        {   fprintf(fplog, "Warning: tnoia = %g is negative. Set to zero.\n", model->BSIM4v0tnoia);
            printf("Warning: tnoia = %g is negative. Set to zero.\n", model->BSIM4v0tnoia);
            model->BSIM4v0tnoia = 0.0;
        }
        if (model->BSIM4v0tnoib < 0.0)
        {   fprintf(fplog, "Warning: tnoib = %g is negative. Set to zero.\n", model->BSIM4v0tnoib);
            printf("Warning: tnoib = %g is negative. Set to zero.\n", model->BSIM4v0tnoib);
            model->BSIM4v0tnoib = 0.0;
        }

        if (model->BSIM4v0ntnoi < 0.0)
        {   fprintf(fplog, "Warning: ntnoi = %g is negative. Set to zero.\n", model->BSIM4v0ntnoi);
            printf("Warning: ntnoi = %g is negative. Set to zero.\n", model->BSIM4v0ntnoi);
            model->BSIM4v0ntnoi = 0.0;
        }

     }/* loop for the parameter check for warning messages */      
	fclose(fplog);
    }
    else
    {   fprintf(stderr, "Warning: Can't open log file. Parameter checking skipped.\n");
    }

    return(Fatal_Flag);
}

