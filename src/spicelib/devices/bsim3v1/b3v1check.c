/* $Id$  */
/* 
$Log$
Revision 1.1.1.1  2000-04-27 20:03:59  pnenzi
Imported sources

 * Revision 3.1  96/12/08  19:52:18  yuhua
 * BSIM3v3.1 release
 * 
*/
static char rcsid[] = "$Id$";

/*************************************/

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: Min-Chie Jeng.
File: b3v1check.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "cktdefs.h"
#include "bsim3v1def.h"
#include "trandefs.h"
#include "const.h"
#include "sperror.h"
#include "devdefs.h"
#include "suffix.h"

int
BSIM3V1checkModel(model, here, ckt)
register BSIM3V1model *model;
register BSIM3V1instance *here;
CKTcircuit *ckt;
{
struct bsim3v1SizeDependParam *pParam;
int Fatal_Flag = 0;
FILE *fplog;
    
    if ((fplog = fopen("BSIM3V3_1_check.log", "w")) != NULL)
    {   pParam = here->pParam;
	fprintf(fplog, "BSIM3V3.1 Parameter Check\n");
	fprintf(fplog, "Model = %s\n", model->BSIM3V1modName);
	fprintf(fplog, "W = %g, L = %g\n", here->BSIM3V1w, here->BSIM3V1l);
   	    

            if (pParam->BSIM3V1nlx < -pParam->BSIM3V1leff)
	    {   fprintf(fplog, "Fatal: Nlx = %g is less than -Leff.\n",
			pParam->BSIM3V1nlx);
	        printf("Fatal: Nlx = %g is less than -Leff.\n",
			pParam->BSIM3V1nlx);
		Fatal_Flag = 1;
            }

	if (model->BSIM3V1tox <= 0.0)
	{   fprintf(fplog, "Fatal: Tox = %g is not positive.\n",
		    model->BSIM3V1tox);
	    printf("Fatal: Tox = %g is not positive.\n", model->BSIM3V1tox);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM3V1npeak <= 0.0)
	{   fprintf(fplog, "Fatal: Nch = %g is not positive.\n",
		    pParam->BSIM3V1npeak);
	    printf("Fatal: Nch = %g is not positive.\n",
		   pParam->BSIM3V1npeak);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3V1nsub <= 0.0)
	{   fprintf(fplog, "Fatal: Nsub = %g is not positive.\n",
		    pParam->BSIM3V1nsub);
	    printf("Fatal: Nsub = %g is not positive.\n",
		   pParam->BSIM3V1nsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3V1ngate < 0.0)
	{   fprintf(fplog, "Fatal: Ngate = %g is not positive.\n",
		    pParam->BSIM3V1ngate);
	    printf("Fatal: Ngate = %g Ngate is not positive.\n",
		   pParam->BSIM3V1ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3V1ngate > 1.e25)
	{   fprintf(fplog, "Fatal: Ngate = %g is too high.\n",
		    pParam->BSIM3V1ngate);
	    printf("Fatal: Ngate = %g Ngate is too high\n",
		   pParam->BSIM3V1ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3V1xj <= 0.0)
	{   fprintf(fplog, "Fatal: Xj = %g is not positive.\n",
		    pParam->BSIM3V1xj);
	    printf("Fatal: Xj = %g is not positive.\n", pParam->BSIM3V1xj);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM3V1dvt1 < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1 = %g is negative.\n",
		    pParam->BSIM3V1dvt1);   
	    printf("Fatal: Dvt1 = %g is negative.\n", pParam->BSIM3V1dvt1);   
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM3V1dvt1w < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1w = %g is negative.\n",
		    pParam->BSIM3V1dvt1w);
	    printf("Fatal: Dvt1w = %g is negative.\n", pParam->BSIM3V1dvt1w);
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM3V1w0 == -pParam->BSIM3V1weff)
	{   fprintf(fplog, "Fatal: (W0 + Weff) = 0 cauing divided-by-zero.\n");
	    printf("Fatal: (W0 + Weff) = 0 cauing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }   

	if (pParam->BSIM3V1dsub < 0.0)
	{   fprintf(fplog, "Fatal: Dsub = %g is negative.\n", pParam->BSIM3V1dsub);
	    printf("Fatal: Dsub = %g is negative.\n", pParam->BSIM3V1dsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3V1b1 == -pParam->BSIM3V1weff)
	{   fprintf(fplog, "Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }  
        if (pParam->BSIM3V1u0temp <= 0.0)
	{   fprintf(fplog, "Fatal: u0 at current temperature = %g is not positive.\n", pParam->BSIM3V1u0temp);
	    printf("Fatal: u0 at current temperature = %g is not positive.\n",
		   pParam->BSIM3V1u0temp);
	    Fatal_Flag = 1;
        }
    
/* Check delta parameter */      
        if (pParam->BSIM3V1delta < 0.0)
	{   fprintf(fplog, "Fatal: Delta = %g is less than zero.\n",
		    pParam->BSIM3V1delta);
	    printf("Fatal: Delta = %g is less than zero.\n", pParam->BSIM3V1delta);
	    Fatal_Flag = 1;
        }      

	if (pParam->BSIM3V1vsattemp <= 0.0)
	{   fprintf(fplog, "Fatal: Vsat at current temperature = %g is not positive.\n", pParam->BSIM3V1vsattemp);
	    printf("Fatal: Vsat at current temperature = %g is not positive.\n",
		   pParam->BSIM3V1vsattemp);
	    Fatal_Flag = 1;
	}
/* Check Rout parameters */
	if (pParam->BSIM3V1pclm <= 0.0)
	{   fprintf(fplog, "Fatal: Pclm = %g is not positive.\n", pParam->BSIM3V1pclm);
	    printf("Fatal: Pclm = %g is not positive.\n", pParam->BSIM3V1pclm);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM3V1drout < 0.0)
	{   fprintf(fplog, "Fatal: Drout = %g is negative.\n", pParam->BSIM3V1drout);
	    printf("Fatal: Drout = %g is negative.\n", pParam->BSIM3V1drout);
	    Fatal_Flag = 1;
	}
      if (model->BSIM3V1unitLengthSidewallJctCap > 0.0 || 
            model->BSIM3V1unitLengthGateSidewallJctCap > 0.0)
      {
	if (here->BSIM3V1drainPerimeter < pParam->BSIM3V1weff)
	{   fprintf(fplog, "Warning: Pd = %g is less than W.\n",
		    here->BSIM3V1drainPerimeter);
	   printf("Warning: Pd = %g is less than W.\n",
		    here->BSIM3V1drainPerimeter);
           here->BSIM3V1drainPerimeter =pParam->BSIM3V1weff;
	}
	if (here->BSIM3V1sourcePerimeter < pParam->BSIM3V1weff)
	{   fprintf(fplog, "Warning: Ps = %g is less than W.\n",
		    here->BSIM3V1sourcePerimeter);
	   printf("Warning: Ps = %g is less than W.\n",
		    here->BSIM3V1sourcePerimeter);
           here->BSIM3V1sourcePerimeter =pParam->BSIM3V1weff;
	}
      }
/* Check capacitance parameters */
        if (pParam->BSIM3V1clc < 0.0)
	{   fprintf(fplog, "Fatal: Clc = %g is negative.\n", pParam->BSIM3V1clc);
	    printf("Fatal: Clc = %g is negative.\n", pParam->BSIM3V1clc);
	    Fatal_Flag = 1;
        }      
      if (model->BSIM3V1paramChk ==1)
      {
/* Check L and W parameters */ 
	if (pParam->BSIM3V1leff <= 5.0e-8)
	{   fprintf(fplog, "Warning: Leff = %g may be too small.\n",
	            pParam->BSIM3V1leff);
	    printf("Warning: Leff = %g may be too small.\n",
		    pParam->BSIM3V1leff);
	}    
	
	if (pParam->BSIM3V1leffCV <= 5.0e-8)
	{   fprintf(fplog, "Warning: Leff for CV = %g may be too small.\n",
		    pParam->BSIM3V1leffCV);
	    printf("Warning: Leff for CV = %g may be too small.\n",
		   pParam->BSIM3V1leffCV);
	}  
	
        if (pParam->BSIM3V1weff <= 1.0e-7)
	{   fprintf(fplog, "Warning: Weff = %g may be too small.\n",
		    pParam->BSIM3V1weff);
	    printf("Warning: Weff = %g may be too small.\n",
		   pParam->BSIM3V1weff);
	}             
	
	if (pParam->BSIM3V1weffCV <= 1.0e-7)
	{   fprintf(fplog, "Warning: Weff for CV = %g may be too small.\n",
		    pParam->BSIM3V1weffCV);
	    printf("Warning: Weff for CV = %g may be too small.\n",
		   pParam->BSIM3V1weffCV);
	}        
	
/* Check threshold voltage parameters */
	if (pParam->BSIM3V1nlx < 0.0)
	{   fprintf(fplog, "Warning: Nlx = %g is negative.\n", pParam->BSIM3V1nlx);
	   printf("Warning: Nlx = %g is negative.\n", pParam->BSIM3V1nlx);
        }
	 if (model->BSIM3V1tox < 1.0e-9)
	{   fprintf(fplog, "Warning: Tox = %g is less than 10A.\n",
	            model->BSIM3V1tox);
	    printf("Warning: Tox = %g is less than 10A.\n", model->BSIM3V1tox);
        }

        if (pParam->BSIM3V1npeak <= 1.0e15)
	{   fprintf(fplog, "Warning: Nch = %g may be too small.\n",
	            pParam->BSIM3V1npeak);
	    printf("Warning: Nch = %g may be too small.\n",
	           pParam->BSIM3V1npeak);
	}
	else if (pParam->BSIM3V1npeak >= 1.0e21)
	{   fprintf(fplog, "Warning: Nch = %g may be too large.\n",
	            pParam->BSIM3V1npeak);
	    printf("Warning: Nch = %g may be too large.\n",
	           pParam->BSIM3V1npeak);
	}

	 if (pParam->BSIM3V1nsub <= 1.0e14)
	{   fprintf(fplog, "Warning: Nsub = %g may be too small.\n",
	            pParam->BSIM3V1nsub);
	    printf("Warning: Nsub = %g may be too small.\n",
	           pParam->BSIM3V1nsub);
	}
	else if (pParam->BSIM3V1nsub >= 1.0e21)
	{   fprintf(fplog, "Warning: Nsub = %g may be too large.\n",
	            pParam->BSIM3V1nsub);
	    printf("Warning: Nsub = %g may be too large.\n",
	           pParam->BSIM3V1nsub);
	}

	if ((pParam->BSIM3V1ngate > 0.0) &&
	    (pParam->BSIM3V1ngate <= 1.e18))
	{   fprintf(fplog, "Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	            pParam->BSIM3V1ngate);
	    printf("Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	           pParam->BSIM3V1ngate);
	}
       
        if (pParam->BSIM3V1dvt0 < 0.0)
	{   fprintf(fplog, "Warning: Dvt0 = %g is negative.\n",
		    pParam->BSIM3V1dvt0);   
	    printf("Warning: Dvt0 = %g is negative.\n", pParam->BSIM3V1dvt0);   
	}
	    
	if (fabs(1.0e-6 / (pParam->BSIM3V1w0 + pParam->BSIM3V1weff)) > 10.0)
	{   fprintf(fplog, "Warning: (W0 + Weff) may be too small.\n");
	    printf("Warning: (W0 + Weff) may be too small.\n");
        }

/* Check subthreshold parameters */
	if (pParam->BSIM3V1nfactor < 0.0)
	{   fprintf(fplog, "Warning: Nfactor = %g is negative.\n",
		    pParam->BSIM3V1nfactor);
	    printf("Warning: Nfactor = %g is negative.\n", pParam->BSIM3V1nfactor);
	}
	if (pParam->BSIM3V1cdsc < 0.0)
	{   fprintf(fplog, "Warning: Cdsc = %g is negative.\n",
		    pParam->BSIM3V1cdsc);
	    printf("Warning: Cdsc = %g is negative.\n", pParam->BSIM3V1cdsc);
	}
	if (pParam->BSIM3V1cdscd < 0.0)
	{   fprintf(fplog, "Warning: Cdscd = %g is negative.\n",
		    pParam->BSIM3V1cdscd);
	    printf("Warning: Cdscd = %g is negative.\n", pParam->BSIM3V1cdscd);
	}
/* Check DIBL parameters */
	if (pParam->BSIM3V1eta0 < 0.0)
	{   fprintf(fplog, "Warning: Eta0 = %g is negative.\n",
		    pParam->BSIM3V1eta0); 
	    printf("Warning: Eta0 = %g is negative.\n", pParam->BSIM3V1eta0); 
	}
	      
/* Check Abulk parameters */	    
	 if (fabs(1.0e-6 / (pParam->BSIM3V1b1 + pParam->BSIM3V1weff)) > 10.0)
       	{   fprintf(fplog, "Warning: (B1 + Weff) may be too small.\n");
       	    printf("Warning: (B1 + Weff) may be too small.\n");
        }    
    

/* Check Saturation parameters */
     	if (pParam->BSIM3V1a2 < 0.01)
	{   fprintf(fplog, "Warning: A2 = %g is too small. Set to 0.01.\n", pParam->BSIM3V1a2);
	    printf("Warning: A2 = %g is too small. Set to 0.01.\n",
		   pParam->BSIM3V1a2);
	    pParam->BSIM3V1a2 = 0.01;
	}
	else if (pParam->BSIM3V1a2 > 1.0)
	{   fprintf(fplog, "Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		    pParam->BSIM3V1a2);
	    printf("Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		   pParam->BSIM3V1a2);
	    pParam->BSIM3V1a2 = 1.0;
	    pParam->BSIM3V1a1 = 0.0;

	}

	if (pParam->BSIM3V1rdsw < 0.0)
	{   fprintf(fplog, "Warning: Rdsw = %g is negative. Set to zero.\n",
		    pParam->BSIM3V1rdsw);
	    printf("Warning: Rdsw = %g is negative. Set to zero.\n",
		   pParam->BSIM3V1rdsw);
	    pParam->BSIM3V1rdsw = 0.0;
	    pParam->BSIM3V1rds0 = 0.0;
	}
	else if ((pParam->BSIM3V1rds0 > 0.0) && (pParam->BSIM3V1rds0 < 0.001))
	{   fprintf(fplog, "Warning: Rds at current temperature = %g is less than 0.001 ohm. Set to zero.\n",
		    pParam->BSIM3V1rds0);
	    printf("Warning: Rds at current temperature = %g is less than 0.001 ohm. Set to zero.\n",
		   pParam->BSIM3V1rds0);
	    pParam->BSIM3V1rds0 = 0.0;
	}
	 if (pParam->BSIM3V1vsattemp < 1.0e3)
	{   fprintf(fplog, "Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM3V1vsattemp);
	   printf("Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM3V1vsattemp);
	}

	if (pParam->BSIM3V1pdibl1 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl1 = %g is negative.\n",
		    pParam->BSIM3V1pdibl1);
	    printf("Warning: Pdibl1 = %g is negative.\n", pParam->BSIM3V1pdibl1);
	}
	if (pParam->BSIM3V1pdibl2 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl2 = %g is negative.\n",
		    pParam->BSIM3V1pdibl2);
	    printf("Warning: Pdibl2 = %g is negative.\n", pParam->BSIM3V1pdibl2);
	}
/* Check overlap capacitance parameters */
        if (model->BSIM3V1cgdo < 0.0)
	{   fprintf(fplog, "Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM3V1cgdo);
	    printf("Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM3V1cgdo);
	    model->BSIM3V1cgdo = 0.0;
        }      
        if (model->BSIM3V1cgso < 0.0)
	{   fprintf(fplog, "Warning: cgso = %g is negative. Set to zero.\n", model->BSIM3V1cgso);
	    printf("Warning: cgso = %g is negative. Set to zero.\n", model->BSIM3V1cgso);
	    model->BSIM3V1cgso = 0.0;
        }      
        if (model->BSIM3V1cgbo < 0.0)
	{   fprintf(fplog, "Warning: cgbo = %g is negative. Set to zero.\n", model->BSIM3V1cgbo);
	    printf("Warning: cgbo = %g is negative. Set to zero.\n", model->BSIM3V1cgbo);
	    model->BSIM3V1cgbo = 0.0;
        }

     }/* loop for the parameter check for warning messages */      
	fclose(fplog);
    }
    else
    {   fprintf(stderr, "Warning: Can't open log file. Parameter checking skipped.\n");
    }

    return(Fatal_Flag);
}

