/**** BSIM4.0.0, Released by Weidong Liu 3/24/2000 ****/

/**********
 * Copyright 2000 Regents of the University of California. All rights reserved.
 * File: b4check.c of BSIM4.0.0.
 * Authors: Weidong Liu, Kanyu M. Cao, Xiaodong Jin, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "cktdefs.h"
#include "bsim4def.h"
#include "trandefs.h"
#include "const.h"
#include "sperror.h"
#include "devdefs.h"


int
BSIM4checkModel(model, here, ckt)
register BSIM4model *model;
register BSIM4instance *here;
CKTcircuit *ckt;
{
struct bsim4SizeDependParam *pParam;
int Fatal_Flag = 0;
FILE *fplog;
    
    if ((fplog = fopen("bsim4.out", "w")) != NULL)
    {   pParam = here->pParam;
        fprintf(fplog, "BSIM4: Berkeley Short Channel IGFET Model-4\n");
        fprintf(fplog, "Developed by Dr. Weidong Liu, Xiaodong Jin, Kanyu M. Cao and Prof. Chenming Hu in 2000.\n");
        fprintf(fplog, "\n");
	fprintf(fplog, "++++++++++ BSIM4 PARAMETER CHECKING BELOW ++++++++++\n");

        if (strcmp(model->BSIM4version, "4.0.0") != 0)
        {  fprintf(fplog, "Warning: This model is BSIM4.0.0; you specified a wrong version number.\n");
           printf("Warning: This model is BSIM4.0.0; you specified a wrong version number.\n");
        }
	fprintf(fplog, "Model = %s\n", model->BSIM4modName);


        if ((here->BSIM4rgateMod == 2) || (here->BSIM4rgateMod == 3))
        {   if ((here->BSIM4trnqsMod == 1) || (here->BSIM4acnqsMod == 1))
            fprintf(fplog, "Warning: You've selected both Rg and charge deficit NQS; select one only.\n");
            printf("Warning: You've selected both Rg and charge deficit NQS; select one only.\n");
        }


	if (model->BSIM4toxe <= 0.0)
	{   fprintf(fplog, "Fatal: Toxe = %g is not positive.\n",
		    model->BSIM4toxe);
	    printf("Fatal: Toxe = %g is not positive.\n", model->BSIM4toxe);
	    Fatal_Flag = 1;
	}
        if (model->BSIM4toxp <= 0.0)
        {   fprintf(fplog, "Fatal: Toxp = %g is not positive.\n",
                    model->BSIM4toxp);
            printf("Fatal: Toxp = %g is not positive.\n", model->BSIM4toxp);
            Fatal_Flag = 1;
        }

        if (model->BSIM4toxm <= 0.0)
        {   fprintf(fplog, "Fatal: Toxm = %g is not positive.\n",
                    model->BSIM4toxm);
            printf("Fatal: Toxm = %g is not positive.\n", model->BSIM4toxm);
            Fatal_Flag = 1;
        }

        if (model->BSIM4toxref <= 0.0)
        {   fprintf(fplog, "Fatal: Toxref = %g is not positive.\n",
                    model->BSIM4toxref);
            printf("Fatal: Toxref = %g is not positive.\n", model->BSIM4toxref);
            Fatal_Flag = 1;
        }

        if (pParam->BSIM4lpe0 < -pParam->BSIM4leff)
        {   fprintf(fplog, "Fatal: Lpe0 = %g is less than -Leff.\n",
                    pParam->BSIM4lpe0);
            printf("Fatal: Lpe0 = %g is less than -Leff.\n",
                        pParam->BSIM4lpe0);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4lpeb < -pParam->BSIM4leff)
        {   fprintf(fplog, "Fatal: Lpeb = %g is less than -Leff.\n",
                    pParam->BSIM4lpeb);
            printf("Fatal: Lpeb = %g is less than -Leff.\n",
                        pParam->BSIM4lpeb);
            Fatal_Flag = 1;
        }

        if (pParam->BSIM4phin < -0.4)
        {   fprintf(fplog, "Fatal: Phin = %g is less than -0.4.\n",
                    pParam->BSIM4phin);
            printf("Fatal: Phin = %g is less than -0.4.\n",
                   pParam->BSIM4phin);
            Fatal_Flag = 1;
        }
	if (pParam->BSIM4ndep <= 0.0)
	{   fprintf(fplog, "Fatal: Ndep = %g is not positive.\n",
		    pParam->BSIM4ndep);
	    printf("Fatal: Ndep = %g is not positive.\n",
		   pParam->BSIM4ndep);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4nsub <= 0.0)
	{   fprintf(fplog, "Fatal: Nsub = %g is not positive.\n",
		    pParam->BSIM4nsub);
	    printf("Fatal: Nsub = %g is not positive.\n",
		   pParam->BSIM4nsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4ngate < 0.0)
	{   fprintf(fplog, "Fatal: Ngate = %g is not positive.\n",
		    pParam->BSIM4ngate);
	    printf("Fatal: Ngate = %g Ngate is not positive.\n",
		   pParam->BSIM4ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4ngate > 1.e25)
	{   fprintf(fplog, "Fatal: Ngate = %g is too high.\n",
		    pParam->BSIM4ngate);
	    printf("Fatal: Ngate = %g Ngate is too high\n",
		   pParam->BSIM4ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4xj <= 0.0)
	{   fprintf(fplog, "Fatal: Xj = %g is not positive.\n",
		    pParam->BSIM4xj);
	    printf("Fatal: Xj = %g is not positive.\n", pParam->BSIM4xj);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4dvt1 < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1 = %g is negative.\n",
		    pParam->BSIM4dvt1);   
	    printf("Fatal: Dvt1 = %g is negative.\n", pParam->BSIM4dvt1);   
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM4dvt1w < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1w = %g is negative.\n",
		    pParam->BSIM4dvt1w);
	    printf("Fatal: Dvt1w = %g is negative.\n", pParam->BSIM4dvt1w);
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM4w0 == -pParam->BSIM4weff)
	{   fprintf(fplog, "Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }   

	if (pParam->BSIM4dsub < 0.0)
	{   fprintf(fplog, "Fatal: Dsub = %g is negative.\n", pParam->BSIM4dsub);
	    printf("Fatal: Dsub = %g is negative.\n", pParam->BSIM4dsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM4b1 == -pParam->BSIM4weff)
	{   fprintf(fplog, "Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }  
        if (pParam->BSIM4u0temp <= 0.0)
	{   fprintf(fplog, "Fatal: u0 at current temperature = %g is not positive.\n", pParam->BSIM4u0temp);
	    printf("Fatal: u0 at current temperature = %g is not positive.\n",
		   pParam->BSIM4u0temp);
	    Fatal_Flag = 1;
        }
    
        if (pParam->BSIM4delta < 0.0)
	{   fprintf(fplog, "Fatal: Delta = %g is less than zero.\n",
		    pParam->BSIM4delta);
	    printf("Fatal: Delta = %g is less than zero.\n", pParam->BSIM4delta);
	    Fatal_Flag = 1;
        }      

	if (pParam->BSIM4vsattemp <= 0.0)
	{   fprintf(fplog, "Fatal: Vsat at current temperature = %g is not positive.\n", pParam->BSIM4vsattemp);
	    printf("Fatal: Vsat at current temperature = %g is not positive.\n",
		   pParam->BSIM4vsattemp);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4pclm <= 0.0)
	{   fprintf(fplog, "Fatal: Pclm = %g is not positive.\n", pParam->BSIM4pclm);
	    printf("Fatal: Pclm = %g is not positive.\n", pParam->BSIM4pclm);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM4drout < 0.0)
	{   fprintf(fplog, "Fatal: Drout = %g is negative.\n", pParam->BSIM4drout);
	    printf("Fatal: Drout = %g is negative.\n", pParam->BSIM4drout);
	    Fatal_Flag = 1;
	}

        if (pParam->BSIM4pscbe2 <= 0.0)
        {   fprintf(fplog, "Warning: Pscbe2 = %g is not positive.\n",
                    pParam->BSIM4pscbe2);
            printf("Warning: Pscbe2 = %g is not positive.\n", pParam->BSIM4pscbe2);
        }

        if (here->BSIM4nf < 1.0)
        {   fprintf(fplog, "Fatal: Number of finger = %g is smaller than one.\n", here->BSIM4nf);
            printf("Fatal: Number of finger = %g is smaller than one.\n", here->BSIM4nf);
            Fatal_Flag = 1;
        }
        if (here->BSIM4nf > 500.0)
        {   here->BSIM4nf = 20.0;
            fprintf(fplog, "Warning: Nf = %g is too large; reset to 20.0.\n", here->BSIM4nf);
            printf("Warning: Nf = %g is too large; reset to 20.0.\n", here->BSIM4nf);
        }

        if (here->BSIM4l <= model->BSIM4xgl)
        {   fprintf(fplog, "Fatal: The parameter xgl must be smaller than Ldrawn.\n");
            printf("Fatal: The parameter xgl must be smaller than Ldrawn.\n");
            Fatal_Flag = 1;
        }
        if (model->BSIM4ngcon < 1.0)
        {   fprintf(fplog, "Fatal: The parameter ngcon cannot be smaller than one.\n");
            printf("Fatal: The parameter ngcon cannot be smaller than one.\n");
            Fatal_Flag = 1;
        }
        if ((model->BSIM4ngcon != 1.0) && (model->BSIM4ngcon != 2.0))
        {   model->BSIM4ngcon = 1.0;
            fprintf(fplog, "Warning: Ngcon must be equal to one or two; reset to 1.0.\n");
            printf("Warning: Ngcon must be equal to one or two; reset to 1.0.\n");
        }

        if (model->BSIM4gbmin < 1.0e-20)
        {   fprintf(fplog, "Warning: Gbmin = %g is too small.\n",
                    model->BSIM4gbmin);
            printf("Warning: Gbmin = %g is too small.\n", model->BSIM4gbmin);
        }

        if (pParam->BSIM4noff < 0.1)
        {   fprintf(fplog, "Warning: Noff = %g is too small.\n",
                    pParam->BSIM4noff);
            printf("Warning: Noff = %g is too small.\n", pParam->BSIM4noff);
        }
        if (pParam->BSIM4noff > 4.0)
        {   fprintf(fplog, "Warning: Noff = %g is too large.\n",
                    pParam->BSIM4noff);
            printf("Warning: Noff = %g is too large.\n", pParam->BSIM4noff);
        }

        if (pParam->BSIM4voffcv < -0.5)
        {   fprintf(fplog, "Warning: Voffcv = %g is too small.\n",
                    pParam->BSIM4voffcv);
            printf("Warning: Voffcv = %g is too small.\n", pParam->BSIM4voffcv);
        }
        if (pParam->BSIM4voffcv > 0.5)
        {   fprintf(fplog, "Warning: Voffcv = %g is too large.\n",
                    pParam->BSIM4voffcv);
            printf("Warning: Voffcv = %g is too large.\n", pParam->BSIM4voffcv);
        }

        /* Check capacitance parameters */
        if (pParam->BSIM4clc < 0.0)
	{   fprintf(fplog, "Fatal: Clc = %g is negative.\n", pParam->BSIM4clc);
	    printf("Fatal: Clc = %g is negative.\n", pParam->BSIM4clc);
	    Fatal_Flag = 1;
        }      

        if (pParam->BSIM4moin < 5.0)
        {   fprintf(fplog, "Warning: Moin = %g is too small.\n",
                    pParam->BSIM4moin);
            printf("Warning: Moin = %g is too small.\n", pParam->BSIM4moin);
        }
        if (pParam->BSIM4moin > 25.0)
        {   fprintf(fplog, "Warning: Moin = %g is too large.\n",
                    pParam->BSIM4moin);
            printf("Warning: Moin = %g is too large.\n", pParam->BSIM4moin);
        }

        if (pParam->BSIM4acde < 0.4)
        {   fprintf(fplog, "Warning:  Acde = %g is too small.\n",
                    pParam->BSIM4acde);
            printf("Warning: Acde = %g is too small.\n", pParam->BSIM4acde);
        }
        if (pParam->BSIM4acde > 1.6)
        {   fprintf(fplog, "Warning:  Acde = %g is too large.\n",
                    pParam->BSIM4acde);
            printf("Warning: Acde = %g is too large.\n", pParam->BSIM4acde);
        }

      if (model->BSIM4paramChk ==1)
      {
/* Check L and W parameters */ 
	if (pParam->BSIM4leff <= 5.0e-8)
	{   fprintf(fplog, "Warning: Leff = %g may be too small.\n",
	            pParam->BSIM4leff);
	    printf("Warning: Leff = %g may be too small.\n",
		    pParam->BSIM4leff);
	}    
	
	if (pParam->BSIM4leffCV <= 5.0e-8)
	{   fprintf(fplog, "Warning: Leff for CV = %g may be too small.\n",
		    pParam->BSIM4leffCV);
	    printf("Warning: Leff for CV = %g may be too small.\n",
		   pParam->BSIM4leffCV);
	}  
	
        if (pParam->BSIM4weff <= 1.0e-7)
	{   fprintf(fplog, "Warning: Weff = %g may be too small.\n",
		    pParam->BSIM4weff);
	    printf("Warning: Weff = %g may be too small.\n",
		   pParam->BSIM4weff);
	}             
	
	if (pParam->BSIM4weffCV <= 1.0e-7)
	{   fprintf(fplog, "Warning: Weff for CV = %g may be too small.\n",
		    pParam->BSIM4weffCV);
	    printf("Warning: Weff for CV = %g may be too small.\n",
		   pParam->BSIM4weffCV);
	}        
	
        /* Check threshold voltage parameters */
	if (model->BSIM4toxe < 1.0e-9)
	{   fprintf(fplog, "Warning: Toxe = %g is less than 10A.\n",
	            model->BSIM4toxe);
	    printf("Warning: Toxe = %g is less than 10A.\n", model->BSIM4toxe);
        }
        if (model->BSIM4toxp < 1.0e-9)
        {   fprintf(fplog, "Warning: Toxp = %g is less than 10A.\n",
                    model->BSIM4toxp);
            printf("Warning: Toxp = %g is less than 10A.\n", model->BSIM4toxp);
        }
        if (model->BSIM4toxm < 1.0e-9)
        {   fprintf(fplog, "Warning: Toxm = %g is less than 10A.\n",
                    model->BSIM4toxm);
            printf("Warning: Toxm = %g is less than 10A.\n", model->BSIM4toxm);
        }

        if (pParam->BSIM4ndep <= 1.0e12)
	{   fprintf(fplog, "Warning: Ndep = %g may be too small.\n",
	            pParam->BSIM4ndep);
	    printf("Warning: Ndep = %g may be too small.\n",
	           pParam->BSIM4ndep);
	}
	else if (pParam->BSIM4ndep >= 1.0e21)
	{   fprintf(fplog, "Warning: Ndep = %g may be too large.\n",
	            pParam->BSIM4ndep);
	    printf("Warning: Ndep = %g may be too large.\n",
	           pParam->BSIM4ndep);
	}

	 if (pParam->BSIM4nsub <= 1.0e14)
	{   fprintf(fplog, "Warning: Nsub = %g may be too small.\n",
	            pParam->BSIM4nsub);
	    printf("Warning: Nsub = %g may be too small.\n",
	           pParam->BSIM4nsub);
	}
	else if (pParam->BSIM4nsub >= 1.0e21)
	{   fprintf(fplog, "Warning: Nsub = %g may be too large.\n",
	            pParam->BSIM4nsub);
	    printf("Warning: Nsub = %g may be too large.\n",
	           pParam->BSIM4nsub);
	}

	if ((pParam->BSIM4ngate > 0.0) &&
	    (pParam->BSIM4ngate <= 1.e18))
	{   fprintf(fplog, "Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	            pParam->BSIM4ngate);
	    printf("Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	           pParam->BSIM4ngate);
	}
       
        if (pParam->BSIM4dvt0 < 0.0)
	{   fprintf(fplog, "Warning: Dvt0 = %g is negative.\n",
		    pParam->BSIM4dvt0);   
	    printf("Warning: Dvt0 = %g is negative.\n", pParam->BSIM4dvt0);   
	}
	    
	if (fabs(1.0e-6 / (pParam->BSIM4w0 + pParam->BSIM4weff)) > 10.0)
	{   fprintf(fplog, "Warning: (W0 + Weff) may be too small.\n");
	    printf("Warning: (W0 + Weff) may be too small.\n");
        }

/* Check subthreshold parameters */
	if (pParam->BSIM4nfactor < 0.0)
	{   fprintf(fplog, "Warning: Nfactor = %g is negative.\n",
		    pParam->BSIM4nfactor);
	    printf("Warning: Nfactor = %g is negative.\n", pParam->BSIM4nfactor);
	}
	if (pParam->BSIM4cdsc < 0.0)
	{   fprintf(fplog, "Warning: Cdsc = %g is negative.\n",
		    pParam->BSIM4cdsc);
	    printf("Warning: Cdsc = %g is negative.\n", pParam->BSIM4cdsc);
	}
	if (pParam->BSIM4cdscd < 0.0)
	{   fprintf(fplog, "Warning: Cdscd = %g is negative.\n",
		    pParam->BSIM4cdscd);
	    printf("Warning: Cdscd = %g is negative.\n", pParam->BSIM4cdscd);
	}
/* Check DIBL parameters */
	if (pParam->BSIM4eta0 < 0.0)
	{   fprintf(fplog, "Warning: Eta0 = %g is negative.\n",
		    pParam->BSIM4eta0); 
	    printf("Warning: Eta0 = %g is negative.\n", pParam->BSIM4eta0); 
	}
	      
/* Check Abulk parameters */	    
	 if (fabs(1.0e-6 / (pParam->BSIM4b1 + pParam->BSIM4weff)) > 10.0)
       	{   fprintf(fplog, "Warning: (B1 + Weff) may be too small.\n");
       	    printf("Warning: (B1 + Weff) may be too small.\n");
        }    
    

/* Check Saturation parameters */
     	if (pParam->BSIM4a2 < 0.01)
	{   fprintf(fplog, "Warning: A2 = %g is too small. Set to 0.01.\n", pParam->BSIM4a2);
	    printf("Warning: A2 = %g is too small. Set to 0.01.\n",
		   pParam->BSIM4a2);
	    pParam->BSIM4a2 = 0.01;
	}
	else if (pParam->BSIM4a2 > 1.0)
	{   fprintf(fplog, "Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		    pParam->BSIM4a2);
	    printf("Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		   pParam->BSIM4a2);
	    pParam->BSIM4a2 = 1.0;
	    pParam->BSIM4a1 = 0.0;

	}

        if (pParam->BSIM4prwg < 0.0)
        {   fprintf(fplog, "Warning: Prwg = %g is negative. Set to zero.\n",
                    pParam->BSIM4prwg);
            printf("Warning: Prwg = %g is negative. Set to zero.\n",
                   pParam->BSIM4prwg);
            pParam->BSIM4prwg = 0.0;
        }
	if (pParam->BSIM4rdsw < 0.0)
	{   fprintf(fplog, "Warning: Rdsw = %g is negative. Set to zero.\n",
		    pParam->BSIM4rdsw);
	    printf("Warning: Rdsw = %g is negative. Set to zero.\n",
		   pParam->BSIM4rdsw);
	    pParam->BSIM4rdsw = 0.0;
	    pParam->BSIM4rds0 = 0.0;
	}
	if (pParam->BSIM4rds0 < 0.0)
	{   fprintf(fplog, "Warning: Rds at current temperature = %g is negative. Set to zero.\n",
		    pParam->BSIM4rds0);
	    printf("Warning: Rds at current temperature = %g is negative. Set to zero.\n",
		   pParam->BSIM4rds0);
	    pParam->BSIM4rds0 = 0.0;
	}

        if (pParam->BSIM4rdswmin < 0.0)
        {   fprintf(fplog, "Warning: Rdswmin at current temperature = %g is negative. Set to zero.\n",
                    pParam->BSIM4rdswmin);
            printf("Warning: Rdswmin at current temperature = %g is negative. Set to zero.\n",
                   pParam->BSIM4rdswmin);
            pParam->BSIM4rdswmin = 0.0;
        }

        if (pParam->BSIM4vsattemp < 1.0e3)
	{   fprintf(fplog, "Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM4vsattemp);
	   printf("Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM4vsattemp);
	}

        if (pParam->BSIM4fprout < 0.0)
        {   fprintf(fplog, "Fatal: fprout = %g is negative.\n",
                    pParam->BSIM4fprout);
            printf("Fatal: fprout = %g is negative.\n", pParam->BSIM4fprout);
	    Fatal_Flag = 1;
        }
        if (pParam->BSIM4pdits < 0.0)
        {   fprintf(fplog, "Fatal: pdits = %g is negative.\n",
                    pParam->BSIM4pdits);
            printf("Fatal: pdits = %g is negative.\n", pParam->BSIM4pdits);
            Fatal_Flag = 1;
        }
        if (model->BSIM4pditsl < 0.0)
        {   fprintf(fplog, "Fatal: pditsl = %g is negative.\n",
                    model->BSIM4pditsl);
            printf("Fatal: pditsl = %g is negative.\n", model->BSIM4pditsl);
            Fatal_Flag = 1;
        }

	if (pParam->BSIM4pdibl1 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl1 = %g is negative.\n",
		    pParam->BSIM4pdibl1);
	    printf("Warning: Pdibl1 = %g is negative.\n", pParam->BSIM4pdibl1);
	}
	if (pParam->BSIM4pdibl2 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl2 = %g is negative.\n",
		    pParam->BSIM4pdibl2);
	    printf("Warning: Pdibl2 = %g is negative.\n", pParam->BSIM4pdibl2);
	}

        if (pParam->BSIM4nigbinv <= 0.0)
        {   fprintf(fplog, "Fatal: nigbinv = %g is non-positive.\n",
                    pParam->BSIM4nigbinv);
            printf("Fatal: nigbinv = %g is non-positive.\n", pParam->BSIM4nigbinv);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4nigbacc <= 0.0)
        {   fprintf(fplog, "Fatal: nigbacc = %g is non-positive.\n",
                    pParam->BSIM4nigbacc);
            printf("Fatal: nigbacc = %g is non-positive.\n", pParam->BSIM4nigbacc);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4nigc <= 0.0)
        {   fprintf(fplog, "Fatal: nigc = %g is non-positive.\n",
                    pParam->BSIM4nigc);
            printf("Fatal: nigc = %g is non-positive.\n", pParam->BSIM4nigc);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4poxedge <= 0.0)
        {   fprintf(fplog, "Fatal: poxedge = %g is non-positive.\n",
                    pParam->BSIM4poxedge);
            printf("Fatal: poxedge = %g is non-positive.\n", pParam->BSIM4poxedge);
            Fatal_Flag = 1;
        }
        if (pParam->BSIM4pigcd <= 0.0)
        {   fprintf(fplog, "Fatal: pigcd = %g is non-positive.\n",
                    pParam->BSIM4pigcd);
            printf("Fatal: pigcd = %g is non-positive.\n", pParam->BSIM4pigcd);
            Fatal_Flag = 1;
        }


/* Check overlap capacitance parameters */
        if (model->BSIM4cgdo < 0.0)
	{   fprintf(fplog, "Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM4cgdo);
	    printf("Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM4cgdo);
	    model->BSIM4cgdo = 0.0;
        }      
        if (model->BSIM4cgso < 0.0)
	{   fprintf(fplog, "Warning: cgso = %g is negative. Set to zero.\n", model->BSIM4cgso);
	    printf("Warning: cgso = %g is negative. Set to zero.\n", model->BSIM4cgso);
	    model->BSIM4cgso = 0.0;
        }      

        if (model->BSIM4tnoia < 0.0)
        {   fprintf(fplog, "Warning: tnoia = %g is negative. Set to zero.\n", model->BSIM4tnoia);
            printf("Warning: tnoia = %g is negative. Set to zero.\n", model->BSIM4tnoia);
            model->BSIM4tnoia = 0.0;
        }
        if (model->BSIM4tnoib < 0.0)
        {   fprintf(fplog, "Warning: tnoib = %g is negative. Set to zero.\n", model->BSIM4tnoib);
            printf("Warning: tnoib = %g is negative. Set to zero.\n", model->BSIM4tnoib);
            model->BSIM4tnoib = 0.0;
        }

        if (model->BSIM4ntnoi < 0.0)
        {   fprintf(fplog, "Warning: ntnoi = %g is negative. Set to zero.\n", model->BSIM4ntnoi);
            printf("Warning: ntnoi = %g is negative. Set to zero.\n", model->BSIM4ntnoi);
            model->BSIM4ntnoi = 0.0;
        }

     }/* loop for the parameter check for warning messages */      
	fclose(fplog);
    }
    else
    {   fprintf(stderr, "Warning: Can't open log file. Parameter checking skipped.\n");
    }

    return(Fatal_Flag);
}

