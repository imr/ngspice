/**** BSIM3v3.2.4, Released by Xuemei Xi 12/14/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b3check.c of BSIM3v3.2.4
 * Author: 1995 Min-Chie Jeng 
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001 Xuemei Xi
 * Modified by Xuemei Xi, 10/05, 12/14, 2001.
 * Modified by Paolo Nenzi 2002 and Dietmar Warning 2003
 **********/

#include "ngspice.h"
#include "cktdefs.h"
#include "bsim3def.h"
#include "trandefs.h"
#include "const.h"
#include "sperror.h"
#include "devdefs.h"
#include "suffix.h"

int
BSIM3checkModel (BSIM3model *model, BSIM3instance *here, CKTcircuit *ckt)
{
struct bsim3SizeDependParam *pParam;
int Fatal_Flag = 0;
FILE *fplog;
    
    if ((fplog = fopen("b3v3check.log", "w")) != NULL)
    {   pParam = here->pParam;

	fprintf (fplog,
		 "BSIM3v3.2 Model (Supports: v3.2 and v3.2.{2,3,4}).\n");
	fprintf (fplog, "Parameter Checking.\n");
	fprintf (fplog, "Model = %s\n", model->BSIM3modName);
	fprintf (fplog, "W = %g, L = %g, M = %g\n", here->BSIM3w,
		 here->BSIM3l, here->BSIM3m);

	if ((strcmp (model->BSIM3version, "3.2.4"))
	    && (strcmp (model->BSIM3version, "3.2.3"))
	    && (strcmp (model->BSIM3version, "3.2.2"))
	    && (strcmp (model->BSIM3version, "3.2")))
	{
		fprintf (fplog,
			 "Warning: This model supports BSIM3v3.2 and BSIM3v3.2.{2,3,4}\n");
		fprintf (fplog,
			 "You specified a wrong version number.\n");
		printf ("Warning: This model supports BSIM3v3.2 and BSIM3v3.2.{2,3,4}\n");
		printf ("You specified a wrong version number.\n");
	}

            if (pParam->BSIM3nlx < -pParam->BSIM3leff)
	    {   fprintf(fplog, "Fatal: Nlx = %g is less than -Leff.\n",
			pParam->BSIM3nlx);
	        printf("Fatal: Nlx = %g is less than -Leff.\n",
			pParam->BSIM3nlx);
		Fatal_Flag = 1;
            }

	if (model->BSIM3tox <= 0.0)
	{   fprintf(fplog, "Fatal: Tox = %g is not positive.\n",
		    model->BSIM3tox);
	    printf("Fatal: Tox = %g is not positive.\n", model->BSIM3tox);
	    Fatal_Flag = 1;
	}

        if (model->BSIM3toxm <= 0.0)
        {   fprintf(fplog, "Fatal: Toxm = %g is not positive.\n",
                    model->BSIM3toxm);
            printf("Fatal: Toxm = %g is not positive.\n", model->BSIM3toxm);
            Fatal_Flag = 1;
        }

	if (pParam->BSIM3npeak <= 0.0)
	{   fprintf(fplog, "Fatal: Nch = %g is not positive.\n",
		    pParam->BSIM3npeak);
	    printf("Fatal: Nch = %g is not positive.\n",
		   pParam->BSIM3npeak);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3nsub <= 0.0)
	{   fprintf(fplog, "Fatal: Nsub = %g is not positive.\n",
		    pParam->BSIM3nsub);
	    printf("Fatal: Nsub = %g is not positive.\n",
		   pParam->BSIM3nsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3ngate < 0.0)
	{   fprintf(fplog, "Fatal: Ngate = %g is not positive.\n",
		    pParam->BSIM3ngate);
	    printf("Fatal: Ngate = %g Ngate is not positive.\n",
		   pParam->BSIM3ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3ngate > 1.e25)
	{   fprintf(fplog, "Fatal: Ngate = %g is too high.\n",
		    pParam->BSIM3ngate);
	    printf("Fatal: Ngate = %g Ngate is too high\n",
		   pParam->BSIM3ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3xj <= 0.0)
	{   fprintf(fplog, "Fatal: Xj = %g is not positive.\n",
		    pParam->BSIM3xj);
	    printf("Fatal: Xj = %g is not positive.\n", pParam->BSIM3xj);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM3dvt1 < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1 = %g is negative.\n",
		    pParam->BSIM3dvt1);   
	    printf("Fatal: Dvt1 = %g is negative.\n", pParam->BSIM3dvt1);   
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM3dvt1w < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1w = %g is negative.\n",
		    pParam->BSIM3dvt1w);
	    printf("Fatal: Dvt1w = %g is negative.\n", pParam->BSIM3dvt1w);
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM3w0 == -pParam->BSIM3weff)
	{   fprintf(fplog, "Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }   

	if (pParam->BSIM3dsub < 0.0)
	{   fprintf(fplog, "Fatal: Dsub = %g is negative.\n", pParam->BSIM3dsub);
	    printf("Fatal: Dsub = %g is negative.\n", pParam->BSIM3dsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3b1 == -pParam->BSIM3weff)
	{   fprintf(fplog, "Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }  
        if (pParam->BSIM3u0temp <= 0.0)
	{   fprintf(fplog, "Fatal: u0 at current temperature = %g is not positive.\n", pParam->BSIM3u0temp);
	    printf("Fatal: u0 at current temperature = %g is not positive.\n",
		   pParam->BSIM3u0temp);
	    Fatal_Flag = 1;
        }
    
/* Check delta parameter */      
        if (pParam->BSIM3delta < 0.0)
	{   fprintf(fplog, "Fatal: Delta = %g is less than zero.\n",
		    pParam->BSIM3delta);
	    printf("Fatal: Delta = %g is less than zero.\n", pParam->BSIM3delta);
	    Fatal_Flag = 1;
        }      

	if (pParam->BSIM3vsattemp <= 0.0)
	{   fprintf(fplog, "Fatal: Vsat at current temperature = %g is not positive.\n", pParam->BSIM3vsattemp);
	    printf("Fatal: Vsat at current temperature = %g is not positive.\n",
		   pParam->BSIM3vsattemp);
	    Fatal_Flag = 1;
	}
/* Check Rout parameters */
	if (pParam->BSIM3pclm <= 0.0)
	{   fprintf(fplog, "Fatal: Pclm = %g is not positive.\n", pParam->BSIM3pclm);
	    printf("Fatal: Pclm = %g is not positive.\n", pParam->BSIM3pclm);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM3drout < 0.0)
	{   fprintf(fplog, "Fatal: Drout = %g is negative.\n", pParam->BSIM3drout);
	    printf("Fatal: Drout = %g is negative.\n", pParam->BSIM3drout);
	    Fatal_Flag = 1;
	}

        if (pParam->BSIM3pscbe2 <= 0.0)
        {   fprintf(fplog, "Warning: Pscbe2 = %g is not positive.\n",
                    pParam->BSIM3pscbe2);
            printf("Warning: Pscbe2 = %g is not positive.\n", pParam->BSIM3pscbe2);
        }

       /* acm model */
       if (model->BSIM3acmMod == 0) {
         if (model->BSIM3unitLengthSidewallJctCap > 0.0 || 
               model->BSIM3unitLengthGateSidewallJctCap > 0.0)
         {
           if (here->BSIM3drainPerimeter < pParam->BSIM3weff)
           {   fprintf(fplog, "Warning: Pd = %g is less than W.\n",
                    here->BSIM3drainPerimeter);
               printf("Warning: Pd = %g is less than W.\n",
                    here->BSIM3drainPerimeter);
           }
           if (here->BSIM3sourcePerimeter < pParam->BSIM3weff)
           {   fprintf(fplog, "Warning: Ps = %g is less than W.\n",
                    here->BSIM3sourcePerimeter);
               printf("Warning: Ps = %g is less than W.\n",
                    here->BSIM3sourcePerimeter);
           }
         }
       }

        if (pParam->BSIM3noff < 0.1)
        {   fprintf(fplog, "Warning: Noff = %g is too small.\n",
                    pParam->BSIM3noff);
            printf("Warning: Noff = %g is too small.\n", pParam->BSIM3noff);
        }
        if (pParam->BSIM3noff > 4.0)
        {   fprintf(fplog, "Warning: Noff = %g is too large.\n",
                    pParam->BSIM3noff);
            printf("Warning: Noff = %g is too large.\n", pParam->BSIM3noff);
        }

        if (pParam->BSIM3voffcv < -0.5)
        {   fprintf(fplog, "Warning: Voffcv = %g is too small.\n",
                    pParam->BSIM3voffcv);
            printf("Warning: Voffcv = %g is too small.\n", pParam->BSIM3voffcv);
        }
        if (pParam->BSIM3voffcv > 0.5)
        {   fprintf(fplog, "Warning: Voffcv = %g is too large.\n",
                    pParam->BSIM3voffcv);
            printf("Warning: Voffcv = %g is too large.\n", pParam->BSIM3voffcv);
        }

        if (model->BSIM3ijth < 0.0)
        {   fprintf(fplog, "Fatal: Ijth = %g cannot be negative.\n",
                    model->BSIM3ijth);
            printf("Fatal: Ijth = %g cannot be negative.\n", model->BSIM3ijth);
            Fatal_Flag = 1;
        }

/* Check capacitance parameters */
        if (pParam->BSIM3clc < 0.0)
	{   fprintf(fplog, "Fatal: Clc = %g is negative.\n", pParam->BSIM3clc);
	    printf("Fatal: Clc = %g is negative.\n", pParam->BSIM3clc);
	    Fatal_Flag = 1;
        }      

        if (pParam->BSIM3moin < 5.0)
        {   fprintf(fplog, "Warning: Moin = %g is too small.\n",
                    pParam->BSIM3moin);
            printf("Warning: Moin = %g is too small.\n", pParam->BSIM3moin);
        }
        if (pParam->BSIM3moin > 25.0)
        {   fprintf(fplog, "Warning: Moin = %g is too large.\n",
                    pParam->BSIM3moin);
            printf("Warning: Moin = %g is too large.\n", pParam->BSIM3moin);
        }

	if(model->BSIM3capMod ==3) {
        	if (pParam->BSIM3acde < 0.4)
        	{   fprintf(fplog, "Warning:  Acde = %g is too small.\n",
                    	pParam->BSIM3acde);
            	printf("Warning: Acde = %g is too small.\n", pParam->BSIM3acde);
        	}
        	if (pParam->BSIM3acde > 1.6)
        	{   fprintf(fplog, "Warning:  Acde = %g is too large.\n",
                    	pParam->BSIM3acde);
            	printf("Warning: Acde = %g is too large.\n", pParam->BSIM3acde);
        	}
	}

      if (model->BSIM3paramChk ==1)
      {
/* Check L and W parameters */ 
	if (pParam->BSIM3leff <= 5.0e-8)
	{   fprintf(fplog, "Warning: Leff = %g may be too small.\n",
	            pParam->BSIM3leff);
	    printf("Warning: Leff = %g may be too small.\n",
		    pParam->BSIM3leff);
	}    
	
	if (pParam->BSIM3leffCV <= 5.0e-8)
	{   fprintf(fplog, "Warning: Leff for CV = %g may be too small.\n",
		    pParam->BSIM3leffCV);
	    printf("Warning: Leff for CV = %g may be too small.\n",
		   pParam->BSIM3leffCV);
	}  
	
        if (pParam->BSIM3weff <= 1.0e-7)
	{   fprintf(fplog, "Warning: Weff = %g may be too small.\n",
		    pParam->BSIM3weff);
	    printf("Warning: Weff = %g may be too small.\n",
		   pParam->BSIM3weff);
	}             
	
	if (pParam->BSIM3weffCV <= 1.0e-7)
	{   fprintf(fplog, "Warning: Weff for CV = %g may be too small.\n",
		    pParam->BSIM3weffCV);
	    printf("Warning: Weff for CV = %g may be too small.\n",
		   pParam->BSIM3weffCV);
	}        
	
/* Check threshold voltage parameters */
	if (pParam->BSIM3nlx < 0.0)
	{   fprintf(fplog, "Warning: Nlx = %g is negative.\n", pParam->BSIM3nlx);
	   printf("Warning: Nlx = %g is negative.\n", pParam->BSIM3nlx);
        }
	 if (model->BSIM3tox < 1.0e-9)
	{   fprintf(fplog, "Warning: Tox = %g is less than 10A.\n",
	            model->BSIM3tox);
	    printf("Warning: Tox = %g is less than 10A.\n", model->BSIM3tox);
        }

        if (pParam->BSIM3npeak <= 1.0e15)
	{   fprintf(fplog, "Warning: Nch = %g may be too small.\n",
	            pParam->BSIM3npeak);
	    printf("Warning: Nch = %g may be too small.\n",
	           pParam->BSIM3npeak);
	}
	else if (pParam->BSIM3npeak >= 1.0e21)
	{   fprintf(fplog, "Warning: Nch = %g may be too large.\n",
	            pParam->BSIM3npeak);
	    printf("Warning: Nch = %g may be too large.\n",
	           pParam->BSIM3npeak);
	}

	 if (pParam->BSIM3nsub <= 1.0e14)
	{   fprintf(fplog, "Warning: Nsub = %g may be too small.\n",
	            pParam->BSIM3nsub);
	    printf("Warning: Nsub = %g may be too small.\n",
	           pParam->BSIM3nsub);
	}
	else if (pParam->BSIM3nsub >= 1.0e21)
	{   fprintf(fplog, "Warning: Nsub = %g may be too large.\n",
	            pParam->BSIM3nsub);
	    printf("Warning: Nsub = %g may be too large.\n",
	           pParam->BSIM3nsub);
	}

	if ((pParam->BSIM3ngate > 0.0) &&
	    (pParam->BSIM3ngate <= 1.e18))
	{   fprintf(fplog, "Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	            pParam->BSIM3ngate);
	    printf("Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	           pParam->BSIM3ngate);
	}
       
        if (pParam->BSIM3dvt0 < 0.0)
	{   fprintf(fplog, "Warning: Dvt0 = %g is negative.\n",
		    pParam->BSIM3dvt0);   
	    printf("Warning: Dvt0 = %g is negative.\n", pParam->BSIM3dvt0);   
	}
	    
	if (fabs(1.0e-6 / (pParam->BSIM3w0 + pParam->BSIM3weff)) > 10.0)
	{   fprintf(fplog, "Warning: (W0 + Weff) may be too small.\n");
	    printf("Warning: (W0 + Weff) may be too small.\n");
        }

/* Check subthreshold parameters */
	if (pParam->BSIM3nfactor < 0.0)
	{   fprintf(fplog, "Warning: Nfactor = %g is negative.\n",
		    pParam->BSIM3nfactor);
	    printf("Warning: Nfactor = %g is negative.\n", pParam->BSIM3nfactor);
	}
	if (pParam->BSIM3cdsc < 0.0)
	{   fprintf(fplog, "Warning: Cdsc = %g is negative.\n",
		    pParam->BSIM3cdsc);
	    printf("Warning: Cdsc = %g is negative.\n", pParam->BSIM3cdsc);
	}
	if (pParam->BSIM3cdscd < 0.0)
	{   fprintf(fplog, "Warning: Cdscd = %g is negative.\n",
		    pParam->BSIM3cdscd);
	    printf("Warning: Cdscd = %g is negative.\n", pParam->BSIM3cdscd);
	}
/* Check DIBL parameters */
	if (pParam->BSIM3eta0 < 0.0)
	{   fprintf(fplog, "Warning: Eta0 = %g is negative.\n",
		    pParam->BSIM3eta0); 
	    printf("Warning: Eta0 = %g is negative.\n", pParam->BSIM3eta0); 
	}
	      
/* Check Abulk parameters */	    
	 if (fabs(1.0e-6 / (pParam->BSIM3b1 + pParam->BSIM3weff)) > 10.0)
       	{   fprintf(fplog, "Warning: (B1 + Weff) may be too small.\n");
       	    printf("Warning: (B1 + Weff) may be too small.\n");
        }    
    

/* Check Saturation parameters */
     	if (pParam->BSIM3a2 < 0.01)
	{   fprintf(fplog, "Warning: A2 = %g is too small. Set to 0.01.\n", pParam->BSIM3a2);
	    printf("Warning: A2 = %g is too small. Set to 0.01.\n",
		   pParam->BSIM3a2);
	    pParam->BSIM3a2 = 0.01;
	}
	else if (pParam->BSIM3a2 > 1.0)
	{   fprintf(fplog, "Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		    pParam->BSIM3a2);
	    printf("Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		   pParam->BSIM3a2);
	    pParam->BSIM3a2 = 1.0;
	    pParam->BSIM3a1 = 0.0;

	}

	if (pParam->BSIM3rdsw < 0.0)
	{   fprintf(fplog, "Warning: Rdsw = %g is negative. Set to zero.\n",
		    pParam->BSIM3rdsw);
	    printf("Warning: Rdsw = %g is negative. Set to zero.\n",
		   pParam->BSIM3rdsw);
	    pParam->BSIM3rdsw = 0.0;
	    pParam->BSIM3rds0 = 0.0;
	}
	else if ((pParam->BSIM3rds0 > 0.0) && (pParam->BSIM3rds0 < 0.001))
	{   fprintf(fplog, "Warning: Rds at current temperature = %g is less than 0.001 ohm. Set to zero.\n",
		    pParam->BSIM3rds0);
	    printf("Warning: Rds at current temperature = %g is less than 0.001 ohm. Set to zero.\n",
		   pParam->BSIM3rds0);
	    pParam->BSIM3rds0 = 0.0;
	}
	 if (pParam->BSIM3vsattemp < 1.0e3)
	{   fprintf(fplog, "Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM3vsattemp);
	   printf("Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM3vsattemp);
	}

	if (pParam->BSIM3pdibl1 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl1 = %g is negative.\n",
		    pParam->BSIM3pdibl1);
	    printf("Warning: Pdibl1 = %g is negative.\n", pParam->BSIM3pdibl1);
	}
	if (pParam->BSIM3pdibl2 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl2 = %g is negative.\n",
		    pParam->BSIM3pdibl2);
	    printf("Warning: Pdibl2 = %g is negative.\n", pParam->BSIM3pdibl2);
	}
/* Check overlap capacitance parameters */
        if (model->BSIM3cgdo < 0.0)
	{   fprintf(fplog, "Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM3cgdo);
	    printf("Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM3cgdo);
	    model->BSIM3cgdo = 0.0;
        }      
        if (model->BSIM3cgso < 0.0)
	{   fprintf(fplog, "Warning: cgso = %g is negative. Set to zero.\n", model->BSIM3cgso);
	    printf("Warning: cgso = %g is negative. Set to zero.\n", model->BSIM3cgso);
	    model->BSIM3cgso = 0.0;
        }      
        if (model->BSIM3cgbo < 0.0)
	{   fprintf(fplog, "Warning: cgbo = %g is negative. Set to zero.\n", model->BSIM3cgbo);
	    printf("Warning: cgbo = %g is negative. Set to zero.\n", model->BSIM3cgbo);
	    model->BSIM3cgbo = 0.0;
        }

     }/* loop for the parameter check for warning messages */      
	fclose(fplog);
    }
    else
    {   fprintf(stderr, "Warning: Can't open log file. Parameter checking skipped.\n");
    }

    return(Fatal_Flag);
}

