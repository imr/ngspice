/**********
 * Copyright 1990 Regents of the University of California. All rights reserved.
 * File: b3v1check.c
 * Author: 1995 Min-Chie Jeng and Mansun Chan. 
 * Modified by Paolo Nenzi 2002
 **********/
 
/* 
 * Release Notes: 
 * BSIM3v3.1,   Released by yuhua  96/12/08
 */


#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v1def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"

int
BSIM3v1checkModel(BSIM3v1model *model, BSIM3v1instance *here, CKTcircuit *ckt)
{
struct bsim3v1SizeDependParam *pParam;
int Fatal_Flag = 0;
FILE *fplog;

    NG_IGNORE(ckt);
    
    if ((fplog = fopen("b3v3_1check.log", "w")) != NULL)
    {   pParam = here->pParam;
	fprintf(fplog, "BSIM3V3.1 Parameter Check\n");
	fprintf(fplog, "Model = %s\n", model->BSIM3v1modName);
	fprintf(fplog, "W = %g, L = %g\n", here->BSIM3v1w, here->BSIM3v1l);
   	    

            if (pParam->BSIM3v1nlx < -pParam->BSIM3v1leff)
	    {   fprintf(fplog, "Fatal: Nlx = %g is less than -Leff.\n",
			pParam->BSIM3v1nlx);
	        printf("Fatal: Nlx = %g is less than -Leff.\n",
			pParam->BSIM3v1nlx);
		Fatal_Flag = 1;
            }

	if (model->BSIM3v1tox <= 0.0)
	{   fprintf(fplog, "Fatal: Tox = %g is not positive.\n",
		    model->BSIM3v1tox);
	    printf("Fatal: Tox = %g is not positive.\n", model->BSIM3v1tox);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM3v1npeak <= 0.0)
	{   fprintf(fplog, "Fatal: Nch = %g is not positive.\n",
		    pParam->BSIM3v1npeak);
	    printf("Fatal: Nch = %g is not positive.\n",
		   pParam->BSIM3v1npeak);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3v1nsub <= 0.0)
	{   fprintf(fplog, "Fatal: Nsub = %g is not positive.\n",
		    pParam->BSIM3v1nsub);
	    printf("Fatal: Nsub = %g is not positive.\n",
		   pParam->BSIM3v1nsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3v1ngate < 0.0)
	{   fprintf(fplog, "Fatal: Ngate = %g is not positive.\n",
		    pParam->BSIM3v1ngate);
	    printf("Fatal: Ngate = %g Ngate is not positive.\n",
		   pParam->BSIM3v1ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3v1ngate > 1.e25)
	{   fprintf(fplog, "Fatal: Ngate = %g is too high.\n",
		    pParam->BSIM3v1ngate);
	    printf("Fatal: Ngate = %g Ngate is too high\n",
		   pParam->BSIM3v1ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3v1xj <= 0.0)
	{   fprintf(fplog, "Fatal: Xj = %g is not positive.\n",
		    pParam->BSIM3v1xj);
	    printf("Fatal: Xj = %g is not positive.\n", pParam->BSIM3v1xj);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM3v1dvt1 < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1 = %g is negative.\n",
		    pParam->BSIM3v1dvt1);   
	    printf("Fatal: Dvt1 = %g is negative.\n", pParam->BSIM3v1dvt1);   
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM3v1dvt1w < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1w = %g is negative.\n",
		    pParam->BSIM3v1dvt1w);
	    printf("Fatal: Dvt1w = %g is negative.\n", pParam->BSIM3v1dvt1w);
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM3v1w0 == -pParam->BSIM3v1weff)
	{   fprintf(fplog, "Fatal: (W0 + Weff) = 0 cauing divided-by-zero.\n");
	    printf("Fatal: (W0 + Weff) = 0 cauing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }   

	if (pParam->BSIM3v1dsub < 0.0)
	{   fprintf(fplog, "Fatal: Dsub = %g is negative.\n", pParam->BSIM3v1dsub);
	    printf("Fatal: Dsub = %g is negative.\n", pParam->BSIM3v1dsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3v1b1 == -pParam->BSIM3v1weff)
	{   fprintf(fplog, "Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }  
        if (pParam->BSIM3v1u0temp <= 0.0)
	{   fprintf(fplog, "Fatal: u0 at current temperature = %g is not positive.\n", pParam->BSIM3v1u0temp);
	    printf("Fatal: u0 at current temperature = %g is not positive.\n",
		   pParam->BSIM3v1u0temp);
	    Fatal_Flag = 1;
        }
    
/* Check delta parameter */      
        if (pParam->BSIM3v1delta < 0.0)
	{   fprintf(fplog, "Fatal: Delta = %g is less than zero.\n",
		    pParam->BSIM3v1delta);
	    printf("Fatal: Delta = %g is less than zero.\n", pParam->BSIM3v1delta);
	    Fatal_Flag = 1;
        }      

	if (pParam->BSIM3v1vsattemp <= 0.0)
	{   fprintf(fplog, "Fatal: Vsat at current temperature = %g is not positive.\n", pParam->BSIM3v1vsattemp);
	    printf("Fatal: Vsat at current temperature = %g is not positive.\n",
		   pParam->BSIM3v1vsattemp);
	    Fatal_Flag = 1;
	}
/* Check Rout parameters */
	if (pParam->BSIM3v1pclm <= 0.0)
	{   fprintf(fplog, "Fatal: Pclm = %g is not positive.\n", pParam->BSIM3v1pclm);
	    printf("Fatal: Pclm = %g is not positive.\n", pParam->BSIM3v1pclm);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM3v1drout < 0.0)
	{   fprintf(fplog, "Fatal: Drout = %g is negative.\n", pParam->BSIM3v1drout);
	    printf("Fatal: Drout = %g is negative.\n", pParam->BSIM3v1drout);
	    Fatal_Flag = 1;
	}
      if (model->BSIM3v1unitLengthSidewallJctCap > 0.0 || 
            model->BSIM3v1unitLengthGateSidewallJctCap > 0.0)
      {
	if (here->BSIM3v1drainPerimeter < pParam->BSIM3v1weff)
	{   fprintf(fplog, "Warning: Pd = %g is less than W.\n",
		    here->BSIM3v1drainPerimeter);
	   printf("Warning: Pd = %g is less than W.\n",
		    here->BSIM3v1drainPerimeter);
           here->BSIM3v1drainPerimeter =pParam->BSIM3v1weff;
	}
	if (here->BSIM3v1sourcePerimeter < pParam->BSIM3v1weff)
	{   fprintf(fplog, "Warning: Ps = %g is less than W.\n",
		    here->BSIM3v1sourcePerimeter);
	   printf("Warning: Ps = %g is less than W.\n",
		    here->BSIM3v1sourcePerimeter);
           here->BSIM3v1sourcePerimeter =pParam->BSIM3v1weff;
	}
      }
/* Check capacitance parameters */
        if (pParam->BSIM3v1clc < 0.0)
	{   fprintf(fplog, "Fatal: Clc = %g is negative.\n", pParam->BSIM3v1clc);
	    printf("Fatal: Clc = %g is negative.\n", pParam->BSIM3v1clc);
	    Fatal_Flag = 1;
        }      
      if (model->BSIM3v1paramChk ==1)
      {
/* Check L and W parameters */ 
	if (pParam->BSIM3v1leff <= 5.0e-8)
	{   fprintf(fplog, "Warning: Leff = %g may be too small.\n",
	            pParam->BSIM3v1leff);
	    printf("Warning: Leff = %g may be too small.\n",
		    pParam->BSIM3v1leff);
	}    
	
	if (pParam->BSIM3v1leffCV <= 5.0e-8)
	{   fprintf(fplog, "Warning: Leff for CV = %g may be too small.\n",
		    pParam->BSIM3v1leffCV);
	    printf("Warning: Leff for CV = %g may be too small.\n",
		   pParam->BSIM3v1leffCV);
	}  
	
        if (pParam->BSIM3v1weff <= 1.0e-7)
	{   fprintf(fplog, "Warning: Weff = %g may be too small.\n",
		    pParam->BSIM3v1weff);
	    printf("Warning: Weff = %g may be too small.\n",
		   pParam->BSIM3v1weff);
	}             
	
	if (pParam->BSIM3v1weffCV <= 1.0e-7)
	{   fprintf(fplog, "Warning: Weff for CV = %g may be too small.\n",
		    pParam->BSIM3v1weffCV);
	    printf("Warning: Weff for CV = %g may be too small.\n",
		   pParam->BSIM3v1weffCV);
	}        
	
/* Check threshold voltage parameters */
	if (pParam->BSIM3v1nlx < 0.0)
	{   fprintf(fplog, "Warning: Nlx = %g is negative.\n", pParam->BSIM3v1nlx);
	   printf("Warning: Nlx = %g is negative.\n", pParam->BSIM3v1nlx);
        }
	 if (model->BSIM3v1tox < 1.0e-9)
	{   fprintf(fplog, "Warning: Tox = %g is less than 10A.\n",
	            model->BSIM3v1tox);
	    printf("Warning: Tox = %g is less than 10A.\n", model->BSIM3v1tox);
        }

        if (pParam->BSIM3v1npeak <= 1.0e15)
	{   fprintf(fplog, "Warning: Nch = %g may be too small.\n",
	            pParam->BSIM3v1npeak);
	    printf("Warning: Nch = %g may be too small.\n",
	           pParam->BSIM3v1npeak);
	}
	else if (pParam->BSIM3v1npeak >= 1.0e21)
	{   fprintf(fplog, "Warning: Nch = %g may be too large.\n",
	            pParam->BSIM3v1npeak);
	    printf("Warning: Nch = %g may be too large.\n",
	           pParam->BSIM3v1npeak);
	}

	 if (pParam->BSIM3v1nsub <= 1.0e14)
	{   fprintf(fplog, "Warning: Nsub = %g may be too small.\n",
	            pParam->BSIM3v1nsub);
	    printf("Warning: Nsub = %g may be too small.\n",
	           pParam->BSIM3v1nsub);
	}
	else if (pParam->BSIM3v1nsub >= 1.0e21)
	{   fprintf(fplog, "Warning: Nsub = %g may be too large.\n",
	            pParam->BSIM3v1nsub);
	    printf("Warning: Nsub = %g may be too large.\n",
	           pParam->BSIM3v1nsub);
	}

	if ((pParam->BSIM3v1ngate > 0.0) &&
	    (pParam->BSIM3v1ngate <= 1.e18))
	{   fprintf(fplog, "Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	            pParam->BSIM3v1ngate);
	    printf("Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	           pParam->BSIM3v1ngate);
	}
       
        if (pParam->BSIM3v1dvt0 < 0.0)
	{   fprintf(fplog, "Warning: Dvt0 = %g is negative.\n",
		    pParam->BSIM3v1dvt0);   
	    printf("Warning: Dvt0 = %g is negative.\n", pParam->BSIM3v1dvt0);   
	}
	    
	if (fabs(1.0e-6 / (pParam->BSIM3v1w0 + pParam->BSIM3v1weff)) > 10.0)
	{   fprintf(fplog, "Warning: (W0 + Weff) may be too small.\n");
	    printf("Warning: (W0 + Weff) may be too small.\n");
        }

/* Check subthreshold parameters */
	if (pParam->BSIM3v1nfactor < 0.0)
	{   fprintf(fplog, "Warning: Nfactor = %g is negative.\n",
		    pParam->BSIM3v1nfactor);
	    printf("Warning: Nfactor = %g is negative.\n", pParam->BSIM3v1nfactor);
	}
	if (pParam->BSIM3v1cdsc < 0.0)
	{   fprintf(fplog, "Warning: Cdsc = %g is negative.\n",
		    pParam->BSIM3v1cdsc);
	    printf("Warning: Cdsc = %g is negative.\n", pParam->BSIM3v1cdsc);
	}
	if (pParam->BSIM3v1cdscd < 0.0)
	{   fprintf(fplog, "Warning: Cdscd = %g is negative.\n",
		    pParam->BSIM3v1cdscd);
	    printf("Warning: Cdscd = %g is negative.\n", pParam->BSIM3v1cdscd);
	}
/* Check DIBL parameters */
	if (pParam->BSIM3v1eta0 < 0.0)
	{   fprintf(fplog, "Warning: Eta0 = %g is negative.\n",
		    pParam->BSIM3v1eta0); 
	    printf("Warning: Eta0 = %g is negative.\n", pParam->BSIM3v1eta0); 
	}
	      
/* Check Abulk parameters */	    
	 if (fabs(1.0e-6 / (pParam->BSIM3v1b1 + pParam->BSIM3v1weff)) > 10.0)
       	{   fprintf(fplog, "Warning: (B1 + Weff) may be too small.\n");
       	    printf("Warning: (B1 + Weff) may be too small.\n");
        }    
    

/* Check Saturation parameters */
     	if (pParam->BSIM3v1a2 < 0.01)
	{   fprintf(fplog, "Warning: A2 = %g is too small. Set to 0.01.\n", pParam->BSIM3v1a2);
	    printf("Warning: A2 = %g is too small. Set to 0.01.\n",
		   pParam->BSIM3v1a2);
	    pParam->BSIM3v1a2 = 0.01;
	}
	else if (pParam->BSIM3v1a2 > 1.0)
	{   fprintf(fplog, "Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		    pParam->BSIM3v1a2);
	    printf("Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		   pParam->BSIM3v1a2);
	    pParam->BSIM3v1a2 = 1.0;
	    pParam->BSIM3v1a1 = 0.0;

	}

	if (pParam->BSIM3v1rdsw < 0.0)
	{   fprintf(fplog, "Warning: Rdsw = %g is negative. Set to zero.\n",
		    pParam->BSIM3v1rdsw);
	    printf("Warning: Rdsw = %g is negative. Set to zero.\n",
		   pParam->BSIM3v1rdsw);
	    pParam->BSIM3v1rdsw = 0.0;
	    pParam->BSIM3v1rds0 = 0.0;
	}
	else if ((pParam->BSIM3v1rds0 > 0.0) && (pParam->BSIM3v1rds0 < 0.001))
	{   fprintf(fplog, "Warning: Rds at current temperature = %g is less than 0.001 ohm. Set to zero.\n",
		    pParam->BSIM3v1rds0);
	    printf("Warning: Rds at current temperature = %g is less than 0.001 ohm. Set to zero.\n",
		   pParam->BSIM3v1rds0);
	    pParam->BSIM3v1rds0 = 0.0;
	}
	 if (pParam->BSIM3v1vsattemp < 1.0e3)
	{   fprintf(fplog, "Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM3v1vsattemp);
	   printf("Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM3v1vsattemp);
	}

	if (pParam->BSIM3v1pdibl1 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl1 = %g is negative.\n",
		    pParam->BSIM3v1pdibl1);
	    printf("Warning: Pdibl1 = %g is negative.\n", pParam->BSIM3v1pdibl1);
	}
	if (pParam->BSIM3v1pdibl2 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl2 = %g is negative.\n",
		    pParam->BSIM3v1pdibl2);
	    printf("Warning: Pdibl2 = %g is negative.\n", pParam->BSIM3v1pdibl2);
	}
/* Check overlap capacitance parameters */
        if (model->BSIM3v1cgdo < 0.0)
	{   fprintf(fplog, "Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM3v1cgdo);
	    printf("Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM3v1cgdo);
	    model->BSIM3v1cgdo = 0.0;
        }      
        if (model->BSIM3v1cgso < 0.0)
	{   fprintf(fplog, "Warning: cgso = %g is negative. Set to zero.\n", model->BSIM3v1cgso);
	    printf("Warning: cgso = %g is negative. Set to zero.\n", model->BSIM3v1cgso);
	    model->BSIM3v1cgso = 0.0;
        }      
        if (model->BSIM3v1cgbo < 0.0)
	{   fprintf(fplog, "Warning: cgbo = %g is negative. Set to zero.\n", model->BSIM3v1cgbo);
	    printf("Warning: cgbo = %g is negative. Set to zero.\n", model->BSIM3v1cgbo);
	    model->BSIM3v1cgbo = 0.0;
        }

     }/* loop for the parameter check for warning messages */      
	fclose(fplog);
    }
    else
    {   fprintf(stderr, "Warning: Can't open log file. Parameter checking skipped.\n");
    }

    return(Fatal_Flag);
}

