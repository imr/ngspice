/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: Min-Chie Jeng.
Modified by Paolo Nenzi 2002
File: b3v1scheck.c
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "bsim3v1sdef.h"
#include "trandefs.h"
#include "const.h"
#include "sperror.h"
#include "devdefs.h"
#include "suffix.h"

int
BSIM3v1ScheckModel(BSIM3v1Smodel *model, BSIM3v1Sinstance *here,
                   CKTcircuit *ckt)
{
struct bsim3v1sSizeDependParam *pParam;
int Fatal_Flag = 0;
FILE *fplog;
    
    if ((fplog = fopen("BSIM3V3_1S_check.log", "w")) != NULL)
    {   pParam = here->pParam;
	fprintf(fplog, "BSIM3V3.1 (Serban) Parameter Check\n");
	fprintf(fplog, "Model = %s\n", model->BSIM3v1SmodName);
	fprintf(fplog, "W = %g, L = %g\n", here->BSIM3v1Sw, here->BSIM3v1Sl);
   	    

            if (pParam->BSIM3v1Snlx < -pParam->BSIM3v1Sleff)
	    {   fprintf(fplog, "Fatal: Nlx = %g is less than -Leff.\n",
			pParam->BSIM3v1Snlx);
	        printf("Fatal: Nlx = %g is less than -Leff.\n",
			pParam->BSIM3v1Snlx);
		Fatal_Flag = 1;
            }

	if (model->BSIM3v1Stox <= 0.0)
	{   fprintf(fplog, "Fatal: Tox = %g is not positive.\n",
		    model->BSIM3v1Stox);
	    printf("Fatal: Tox = %g is not positive.\n", model->BSIM3v1Stox);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM3v1Snpeak <= 0.0)
	{   fprintf(fplog, "Fatal: Nch = %g is not positive.\n",
		    pParam->BSIM3v1Snpeak);
	    printf("Fatal: Nch = %g is not positive.\n",
		   pParam->BSIM3v1Snpeak);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3v1Snsub <= 0.0)
	{   fprintf(fplog, "Fatal: Nsub = %g is not positive.\n",
		    pParam->BSIM3v1Snsub);
	    printf("Fatal: Nsub = %g is not positive.\n",
		   pParam->BSIM3v1Snsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3v1Sngate < 0.0)
	{   fprintf(fplog, "Fatal: Ngate = %g is not positive.\n",
		    pParam->BSIM3v1Sngate);
	    printf("Fatal: Ngate = %g Ngate is not positive.\n",
		   pParam->BSIM3v1Sngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3v1Sngate > 1.e25)
	{   fprintf(fplog, "Fatal: Ngate = %g is too high.\n",
		    pParam->BSIM3v1Sngate);
	    printf("Fatal: Ngate = %g Ngate is too high\n",
		   pParam->BSIM3v1Sngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3v1Sxj <= 0.0)
	{   fprintf(fplog, "Fatal: Xj = %g is not positive.\n",
		    pParam->BSIM3v1Sxj);
	    printf("Fatal: Xj = %g is not positive.\n", pParam->BSIM3v1Sxj);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM3v1Sdvt1 < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1 = %g is negative.\n",
		    pParam->BSIM3v1Sdvt1);   
	    printf("Fatal: Dvt1 = %g is negative.\n", pParam->BSIM3v1Sdvt1);   
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM3v1Sdvt1w < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1w = %g is negative.\n",
		    pParam->BSIM3v1Sdvt1w);
	    printf("Fatal: Dvt1w = %g is negative.\n", pParam->BSIM3v1Sdvt1w);
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM3v1Sw0 == -pParam->BSIM3v1Sweff)
	{   fprintf(fplog, "Fatal: (W0 + Weff) = 0 cauing divided-by-zero.\n");
	    printf("Fatal: (W0 + Weff) = 0 cauing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }   

	if (pParam->BSIM3v1Sdsub < 0.0)
	{   fprintf(fplog, "Fatal: Dsub = %g is negative.\n", pParam->BSIM3v1Sdsub);
	    printf("Fatal: Dsub = %g is negative.\n", pParam->BSIM3v1Sdsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3v1Sb1 == -pParam->BSIM3v1Sweff)
	{   fprintf(fplog, "Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }  
        if (pParam->BSIM3v1Su0temp <= 0.0)
	{   fprintf(fplog, "Fatal: u0 at current temperature = %g is not positive.\n", pParam->BSIM3v1Su0temp);
	    printf("Fatal: u0 at current temperature = %g is not positive.\n",
		   pParam->BSIM3v1Su0temp);
	    Fatal_Flag = 1;
        }
    
/* Check delta parameter */      
        if (pParam->BSIM3v1Sdelta < 0.0)
	{   fprintf(fplog, "Fatal: Delta = %g is less than zero.\n",
		    pParam->BSIM3v1Sdelta);
	    printf("Fatal: Delta = %g is less than zero.\n", pParam->BSIM3v1Sdelta);
	    Fatal_Flag = 1;
        }      

	if (pParam->BSIM3v1Svsattemp <= 0.0)
	{   fprintf(fplog, "Fatal: Vsat at current temperature = %g is not positive.\n", pParam->BSIM3v1Svsattemp);
	    printf("Fatal: Vsat at current temperature = %g is not positive.\n",
		   pParam->BSIM3v1Svsattemp);
	    Fatal_Flag = 1;
	}
/* Check Rout parameters */
	if (pParam->BSIM3v1Spclm <= 0.0)
	{   fprintf(fplog, "Fatal: Pclm = %g is not positive.\n", pParam->BSIM3v1Spclm);
	    printf("Fatal: Pclm = %g is not positive.\n", pParam->BSIM3v1Spclm);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM3v1Sdrout < 0.0)
	{   fprintf(fplog, "Fatal: Drout = %g is negative.\n", pParam->BSIM3v1Sdrout);
	    printf("Fatal: Drout = %g is negative.\n", pParam->BSIM3v1Sdrout);
	    Fatal_Flag = 1;
	}
      if (model->BSIM3v1SunitLengthSidewallJctCap > 0.0 || 
            model->BSIM3v1SunitLengthGateSidewallJctCap > 0.0)
      {
	if (here->BSIM3v1SdrainPerimeter < pParam->BSIM3v1Sweff)
	{   fprintf(fplog, "Warning: Pd = %g is less than W.\n",
		    here->BSIM3v1SdrainPerimeter);
	   printf("Warning: Pd = %g is less than W.\n",
		    here->BSIM3v1SdrainPerimeter);
           here->BSIM3v1SdrainPerimeter =pParam->BSIM3v1Sweff;
	}
	if (here->BSIM3v1SsourcePerimeter < pParam->BSIM3v1Sweff)
	{   fprintf(fplog, "Warning: Ps = %g is less than W.\n",
		    here->BSIM3v1SsourcePerimeter);
	   printf("Warning: Ps = %g is less than W.\n",
		    here->BSIM3v1SsourcePerimeter);
           here->BSIM3v1SsourcePerimeter =pParam->BSIM3v1Sweff;
	}
      }
/* Check capacitance parameters */
        if (pParam->BSIM3v1Sclc < 0.0)
	{   fprintf(fplog, "Fatal: Clc = %g is negative.\n", pParam->BSIM3v1Sclc);
	    printf("Fatal: Clc = %g is negative.\n", pParam->BSIM3v1Sclc);
	    Fatal_Flag = 1;
        }      
      if (model->BSIM3v1SparamChk ==1)
      {
/* Check L and W parameters */ 
	if (pParam->BSIM3v1Sleff <= 5.0e-8)
	{   fprintf(fplog, "Warning: Leff = %g may be too small.\n",
	            pParam->BSIM3v1Sleff);
	    printf("Warning: Leff = %g may be too small.\n",
		    pParam->BSIM3v1Sleff);
	}    
	
	if (pParam->BSIM3v1SleffCV <= 5.0e-8)
	{   fprintf(fplog, "Warning: Leff for CV = %g may be too small.\n",
		    pParam->BSIM3v1SleffCV);
	    printf("Warning: Leff for CV = %g may be too small.\n",
		   pParam->BSIM3v1SleffCV);
	}  
	
        if (pParam->BSIM3v1Sweff <= 1.0e-7)
	{   fprintf(fplog, "Warning: Weff = %g may be too small.\n",
		    pParam->BSIM3v1Sweff);
	    printf("Warning: Weff = %g may be too small.\n",
		   pParam->BSIM3v1Sweff);
	}             
	
	if (pParam->BSIM3v1SweffCV <= 1.0e-7)
	{   fprintf(fplog, "Warning: Weff for CV = %g may be too small.\n",
		    pParam->BSIM3v1SweffCV);
	    printf("Warning: Weff for CV = %g may be too small.\n",
		   pParam->BSIM3v1SweffCV);
	}        
	
/* Check threshold voltage parameters */
	if (pParam->BSIM3v1Snlx < 0.0)
	{   fprintf(fplog, "Warning: Nlx = %g is negative.\n", pParam->BSIM3v1Snlx);
	   printf("Warning: Nlx = %g is negative.\n", pParam->BSIM3v1Snlx);
        }
	 if (model->BSIM3v1Stox < 1.0e-9)
	{   fprintf(fplog, "Warning: Tox = %g is less than 10A.\n",
	            model->BSIM3v1Stox);
	    printf("Warning: Tox = %g is less than 10A.\n", model->BSIM3v1Stox);
        }

        if (pParam->BSIM3v1Snpeak <= 1.0e15)
	{   fprintf(fplog, "Warning: Nch = %g may be too small.\n",
	            pParam->BSIM3v1Snpeak);
	    printf("Warning: Nch = %g may be too small.\n",
	           pParam->BSIM3v1Snpeak);
	}
	else if (pParam->BSIM3v1Snpeak >= 1.0e21)
	{   fprintf(fplog, "Warning: Nch = %g may be too large.\n",
	            pParam->BSIM3v1Snpeak);
	    printf("Warning: Nch = %g may be too large.\n",
	           pParam->BSIM3v1Snpeak);
	}

	 if (pParam->BSIM3v1Snsub <= 1.0e14)
	{   fprintf(fplog, "Warning: Nsub = %g may be too small.\n",
	            pParam->BSIM3v1Snsub);
	    printf("Warning: Nsub = %g may be too small.\n",
	           pParam->BSIM3v1Snsub);
	}
	else if (pParam->BSIM3v1Snsub >= 1.0e21)
	{   fprintf(fplog, "Warning: Nsub = %g may be too large.\n",
	            pParam->BSIM3v1Snsub);
	    printf("Warning: Nsub = %g may be too large.\n",
	           pParam->BSIM3v1Snsub);
	}

	if ((pParam->BSIM3v1Sngate > 0.0) &&
	    (pParam->BSIM3v1Sngate <= 1.e18))
	{   fprintf(fplog, "Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	            pParam->BSIM3v1Sngate);
	    printf("Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	           pParam->BSIM3v1Sngate);
	}
       
        if (pParam->BSIM3v1Sdvt0 < 0.0)
	{   fprintf(fplog, "Warning: Dvt0 = %g is negative.\n",
		    pParam->BSIM3v1Sdvt0);   
	    printf("Warning: Dvt0 = %g is negative.\n", pParam->BSIM3v1Sdvt0);   
	}
	    
	if (fabs(1.0e-6 / (pParam->BSIM3v1Sw0 + pParam->BSIM3v1Sweff)) > 10.0)
	{   fprintf(fplog, "Warning: (W0 + Weff) may be too small.\n");
	    printf("Warning: (W0 + Weff) may be too small.\n");
        }

/* Check subthreshold parameters */
	if (pParam->BSIM3v1Snfactor < 0.0)
	{   fprintf(fplog, "Warning: Nfactor = %g is negative.\n",
		    pParam->BSIM3v1Snfactor);
	    printf("Warning: Nfactor = %g is negative.\n", pParam->BSIM3v1Snfactor);
	}
	if (pParam->BSIM3v1Scdsc < 0.0)
	{   fprintf(fplog, "Warning: Cdsc = %g is negative.\n",
		    pParam->BSIM3v1Scdsc);
	    printf("Warning: Cdsc = %g is negative.\n", pParam->BSIM3v1Scdsc);
	}
	if (pParam->BSIM3v1Scdscd < 0.0)
	{   fprintf(fplog, "Warning: Cdscd = %g is negative.\n",
		    pParam->BSIM3v1Scdscd);
	    printf("Warning: Cdscd = %g is negative.\n", pParam->BSIM3v1Scdscd);
	}
/* Check DIBL parameters */
	if (pParam->BSIM3v1Seta0 < 0.0)
	{   fprintf(fplog, "Warning: Eta0 = %g is negative.\n",
		    pParam->BSIM3v1Seta0); 
	    printf("Warning: Eta0 = %g is negative.\n", pParam->BSIM3v1Seta0); 
	}
	      
/* Check Abulk parameters */	    
	 if (fabs(1.0e-6 / (pParam->BSIM3v1Sb1 + pParam->BSIM3v1Sweff)) > 10.0)
       	{   fprintf(fplog, "Warning: (B1 + Weff) may be too small.\n");
       	    printf("Warning: (B1 + Weff) may be too small.\n");
        }    
    

/* Check Saturation parameters */
     	if (pParam->BSIM3v1Sa2 < 0.01)
	{   fprintf(fplog, "Warning: A2 = %g is too small. Set to 0.01.\n", pParam->BSIM3v1Sa2);
	    printf("Warning: A2 = %g is too small. Set to 0.01.\n",
		   pParam->BSIM3v1Sa2);
	    pParam->BSIM3v1Sa2 = 0.01;
	}
	else if (pParam->BSIM3v1Sa2 > 1.0)
	{   fprintf(fplog, "Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		    pParam->BSIM3v1Sa2);
	    printf("Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		   pParam->BSIM3v1Sa2);
	    pParam->BSIM3v1Sa2 = 1.0;
	    pParam->BSIM3v1Sa1 = 0.0;

	}

	if (pParam->BSIM3v1Srdsw < 0.0)
	{   fprintf(fplog, "Warning: Rdsw = %g is negative. Set to zero.\n",
		    pParam->BSIM3v1Srdsw);
	    printf("Warning: Rdsw = %g is negative. Set to zero.\n",
		   pParam->BSIM3v1Srdsw);
	    pParam->BSIM3v1Srdsw = 0.0;
	    pParam->BSIM3v1Srds0 = 0.0;
	}
	else if ((pParam->BSIM3v1Srds0 > 0.0) && (pParam->BSIM3v1Srds0 < 0.001))
	{   fprintf(fplog, "Warning: Rds at current temperature = %g is less than 0.001 ohm. Set to zero.\n",
		    pParam->BSIM3v1Srds0);
	    printf("Warning: Rds at current temperature = %g is less than 0.001 ohm. Set to zero.\n",
		   pParam->BSIM3v1Srds0);
	    pParam->BSIM3v1Srds0 = 0.0;
	}
	 if (pParam->BSIM3v1Svsattemp < 1.0e3)
	{   fprintf(fplog, "Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM3v1Svsattemp);
	   printf("Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM3v1Svsattemp);
	}

	if (pParam->BSIM3v1Spdibl1 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl1 = %g is negative.\n",
		    pParam->BSIM3v1Spdibl1);
	    printf("Warning: Pdibl1 = %g is negative.\n", pParam->BSIM3v1Spdibl1);
	}
	if (pParam->BSIM3v1Spdibl2 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl2 = %g is negative.\n",
		    pParam->BSIM3v1Spdibl2);
	    printf("Warning: Pdibl2 = %g is negative.\n", pParam->BSIM3v1Spdibl2);
	}
/* Check overlap capacitance parameters */
        if (model->BSIM3v1Scgdo < 0.0)
	{   fprintf(fplog, "Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM3v1Scgdo);
	    printf("Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM3v1Scgdo);
	    model->BSIM3v1Scgdo = 0.0;
        }      
        if (model->BSIM3v1Scgso < 0.0)
	{   fprintf(fplog, "Warning: cgso = %g is negative. Set to zero.\n", model->BSIM3v1Scgso);
	    printf("Warning: cgso = %g is negative. Set to zero.\n", model->BSIM3v1Scgso);
	    model->BSIM3v1Scgso = 0.0;
        }      
        if (model->BSIM3v1Scgbo < 0.0)
	{   fprintf(fplog, "Warning: cgbo = %g is negative. Set to zero.\n", model->BSIM3v1Scgbo);
	    printf("Warning: cgbo = %g is negative. Set to zero.\n", model->BSIM3v1Scgbo);
	    model->BSIM3v1Scgbo = 0.0;
        }

     }/* loop for the parameter check for warning messages */      
	fclose(fplog);
    }
    else
    {   fprintf(stderr, "Warning: Can't open log file. Parameter checking skipped.\n");
    }

    return(Fatal_Flag);
}

