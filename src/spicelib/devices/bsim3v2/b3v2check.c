/* $Id$  */
/*
 $Log$
 Revision 1.1  2000-04-27 20:03:59  pnenzi
 Initial revision

 * Revision 3.2 1998/6/16  18:00:00  Weidong 
 * BSIM3v3.2 release
 *
*/
static char rcsid[] = "$Id$";

/*************************************/

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: Min-Chie Jeng.
Modified by Weidong Liu (1997-1998).
File: b3v2check.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "cktdefs.h"
#include "bsim3v2def.h"
#include "trandefs.h"
#include "const.h"
#include "sperror.h"
#include "devdefs.h"
#include "suffix.h"

int
BSIM3V2checkModel(model, here, ckt)
register BSIM3V2model *model;
register BSIM3V2instance *here;
CKTcircuit *ckt;
{
struct BSIM3V2SizeDependParam *pParam;
int Fatal_Flag = 0;
FILE *fplog;
    
    if ((fplog = fopen("b3v3check.log", "w")) != NULL)
    {   pParam = here->pParam;
	fprintf(fplog, "BSIM3V3.2 Parameter Check\n");
	fprintf(fplog, "Model = %s\n", model->BSIM3V2modName);
	fprintf(fplog, "W = %g, L = %g\n", here->BSIM3V2w, here->BSIM3V2l);
   	    

            if (pParam->BSIM3V2nlx < -pParam->BSIM3V2leff)
	    {   fprintf(fplog, "Fatal: Nlx = %g is less than -Leff.\n",
			pParam->BSIM3V2nlx);
	        printf("Fatal: Nlx = %g is less than -Leff.\n",
			pParam->BSIM3V2nlx);
		Fatal_Flag = 1;
            }

	if (model->BSIM3V2tox <= 0.0)
	{   fprintf(fplog, "Fatal: Tox = %g is not positive.\n",
		    model->BSIM3V2tox);
	    printf("Fatal: Tox = %g is not positive.\n", model->BSIM3V2tox);
	    Fatal_Flag = 1;
	}

        if (model->BSIM3V2toxm <= 0.0)
        {   fprintf(fplog, "Fatal: Toxm = %g is not positive.\n",
                    model->BSIM3V2toxm);
            printf("Fatal: Toxm = %g is not positive.\n", model->BSIM3V2toxm);
            Fatal_Flag = 1;
        }

	if (pParam->BSIM3V2npeak <= 0.0)
	{   fprintf(fplog, "Fatal: Nch = %g is not positive.\n",
		    pParam->BSIM3V2npeak);
	    printf("Fatal: Nch = %g is not positive.\n",
		   pParam->BSIM3V2npeak);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3V2nsub <= 0.0)
	{   fprintf(fplog, "Fatal: Nsub = %g is not positive.\n",
		    pParam->BSIM3V2nsub);
	    printf("Fatal: Nsub = %g is not positive.\n",
		   pParam->BSIM3V2nsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3V2ngate < 0.0)
	{   fprintf(fplog, "Fatal: Ngate = %g is not positive.\n",
		    pParam->BSIM3V2ngate);
	    printf("Fatal: Ngate = %g Ngate is not positive.\n",
		   pParam->BSIM3V2ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3V2ngate > 1.e25)
	{   fprintf(fplog, "Fatal: Ngate = %g is too high.\n",
		    pParam->BSIM3V2ngate);
	    printf("Fatal: Ngate = %g Ngate is too high\n",
		   pParam->BSIM3V2ngate);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3V2xj <= 0.0)
	{   fprintf(fplog, "Fatal: Xj = %g is not positive.\n",
		    pParam->BSIM3V2xj);
	    printf("Fatal: Xj = %g is not positive.\n", pParam->BSIM3V2xj);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM3V2dvt1 < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1 = %g is negative.\n",
		    pParam->BSIM3V2dvt1);   
	    printf("Fatal: Dvt1 = %g is negative.\n", pParam->BSIM3V2dvt1);   
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM3V2dvt1w < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1w = %g is negative.\n",
		    pParam->BSIM3V2dvt1w);
	    printf("Fatal: Dvt1w = %g is negative.\n", pParam->BSIM3V2dvt1w);
	    Fatal_Flag = 1;
	}
	    
	if (pParam->BSIM3V2w0 == -pParam->BSIM3V2weff)
	{   fprintf(fplog, "Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (W0 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }   

	if (pParam->BSIM3V2dsub < 0.0)
	{   fprintf(fplog, "Fatal: Dsub = %g is negative.\n", pParam->BSIM3V2dsub);
	    printf("Fatal: Dsub = %g is negative.\n", pParam->BSIM3V2dsub);
	    Fatal_Flag = 1;
	}
	if (pParam->BSIM3V2b1 == -pParam->BSIM3V2weff)
	{   fprintf(fplog, "Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }  
        if (pParam->BSIM3V2u0temp <= 0.0)
	{   fprintf(fplog, "Fatal: u0 at current temperature = %g is not positive.\n", pParam->BSIM3V2u0temp);
	    printf("Fatal: u0 at current temperature = %g is not positive.\n",
		   pParam->BSIM3V2u0temp);
	    Fatal_Flag = 1;
        }
    
/* Check delta parameter */      
        if (pParam->BSIM3V2delta < 0.0)
	{   fprintf(fplog, "Fatal: Delta = %g is less than zero.\n",
		    pParam->BSIM3V2delta);
	    printf("Fatal: Delta = %g is less than zero.\n", pParam->BSIM3V2delta);
	    Fatal_Flag = 1;
        }      

	if (pParam->BSIM3V2vsattemp <= 0.0)
	{   fprintf(fplog, "Fatal: Vsat at current temperature = %g is not positive.\n", pParam->BSIM3V2vsattemp);
	    printf("Fatal: Vsat at current temperature = %g is not positive.\n",
		   pParam->BSIM3V2vsattemp);
	    Fatal_Flag = 1;
	}
/* Check Rout parameters */
	if (pParam->BSIM3V2pclm <= 0.0)
	{   fprintf(fplog, "Fatal: Pclm = %g is not positive.\n", pParam->BSIM3V2pclm);
	    printf("Fatal: Pclm = %g is not positive.\n", pParam->BSIM3V2pclm);
	    Fatal_Flag = 1;
	}

	if (pParam->BSIM3V2drout < 0.0)
	{   fprintf(fplog, "Fatal: Drout = %g is negative.\n", pParam->BSIM3V2drout);
	    printf("Fatal: Drout = %g is negative.\n", pParam->BSIM3V2drout);
	    Fatal_Flag = 1;
	}

        if (pParam->BSIM3V2pscbe2 <= 0.0)
        {   fprintf(fplog, "Warning: Pscbe2 = %g is not positive.\n",
                    pParam->BSIM3V2pscbe2);
            printf("Warning: Pscbe2 = %g is not positive.\n", pParam->BSIM3V2pscbe2);
        }

      if (model->BSIM3V2unitLengthSidewallJctCap > 0.0 || 
            model->BSIM3V2unitLengthGateSidewallJctCap > 0.0)
      {
	if (here->BSIM3V2drainPerimeter < pParam->BSIM3V2weff)
	{   fprintf(fplog, "Warning: Pd = %g is less than W.\n",
		    here->BSIM3V2drainPerimeter);
	   printf("Warning: Pd = %g is less than W.\n",
		    here->BSIM3V2drainPerimeter);
	}
	if (here->BSIM3V2sourcePerimeter < pParam->BSIM3V2weff)
	{   fprintf(fplog, "Warning: Ps = %g is less than W.\n",
		    here->BSIM3V2sourcePerimeter);
	   printf("Warning: Ps = %g is less than W.\n",
		    here->BSIM3V2sourcePerimeter);
	}
      }

        if (pParam->BSIM3V2noff < 0.1)
        {   fprintf(fplog, "Warning: Noff = %g is too small.\n",
                    pParam->BSIM3V2noff);
            printf("Warning: Noff = %g is too small.\n", pParam->BSIM3V2noff);
        }
        if (pParam->BSIM3V2noff > 4.0)
        {   fprintf(fplog, "Warning: Noff = %g is too large.\n",
                    pParam->BSIM3V2noff);
            printf("Warning: Noff = %g is too large.\n", pParam->BSIM3V2noff);
        }

        if (pParam->BSIM3V2voffcv < -0.5)
        {   fprintf(fplog, "Warning: Voffcv = %g is too small.\n",
                    pParam->BSIM3V2voffcv);
            printf("Warning: Voffcv = %g is too small.\n", pParam->BSIM3V2voffcv);
        }
        if (pParam->BSIM3V2voffcv > 0.5)
        {   fprintf(fplog, "Warning: Voffcv = %g is too large.\n",
                    pParam->BSIM3V2voffcv);
            printf("Warning: Voffcv = %g is too large.\n", pParam->BSIM3V2voffcv);
        }

        if (model->BSIM3V2ijth < 0.0)
        {   fprintf(fplog, "Fatal: Ijth = %g cannot be negative.\n",
                    model->BSIM3V2ijth);
            printf("Fatal: Ijth = %g cannot be negative.\n", model->BSIM3V2ijth);
            Fatal_Flag = 1;
        }

/* Check capacitance parameters */
        if (pParam->BSIM3V2clc < 0.0)
	{   fprintf(fplog, "Fatal: Clc = %g is negative.\n", pParam->BSIM3V2clc);
	    printf("Fatal: Clc = %g is negative.\n", pParam->BSIM3V2clc);
	    Fatal_Flag = 1;
        }      

        if (pParam->BSIM3V2moin < 5.0)
        {   fprintf(fplog, "Warning: Moin = %g is too small.\n",
                    pParam->BSIM3V2moin);
            printf("Warning: Moin = %g is too small.\n", pParam->BSIM3V2moin);
        }
        if (pParam->BSIM3V2moin > 25.0)
        {   fprintf(fplog, "Warning: Moin = %g is too large.\n",
                    pParam->BSIM3V2moin);
            printf("Warning: Moin = %g is too large.\n", pParam->BSIM3V2moin);
        }

        if (pParam->BSIM3V2acde < 0.4)
        {   fprintf(fplog, "Warning:  Acde = %g is too small.\n",
                    pParam->BSIM3V2acde);
            printf("Warning: Acde = %g is too small.\n", pParam->BSIM3V2acde);
        }
        if (pParam->BSIM3V2acde > 1.6)
        {   fprintf(fplog, "Warning:  Acde = %g is too large.\n",
                    pParam->BSIM3V2acde);
            printf("Warning: Acde = %g is too large.\n", pParam->BSIM3V2acde);
        }

      if (model->BSIM3V2paramChk ==1)
      {
/* Check L and W parameters */ 
	if (pParam->BSIM3V2leff <= 5.0e-8)
	{   fprintf(fplog, "Warning: Leff = %g may be too small.\n",
	            pParam->BSIM3V2leff);
	    printf("Warning: Leff = %g may be too small.\n",
		    pParam->BSIM3V2leff);
	}    
	
	if (pParam->BSIM3V2leffCV <= 5.0e-8)
	{   fprintf(fplog, "Warning: Leff for CV = %g may be too small.\n",
		    pParam->BSIM3V2leffCV);
	    printf("Warning: Leff for CV = %g may be too small.\n",
		   pParam->BSIM3V2leffCV);
	}  
	
        if (pParam->BSIM3V2weff <= 1.0e-7)
	{   fprintf(fplog, "Warning: Weff = %g may be too small.\n",
		    pParam->BSIM3V2weff);
	    printf("Warning: Weff = %g may be too small.\n",
		   pParam->BSIM3V2weff);
	}             
	
	if (pParam->BSIM3V2weffCV <= 1.0e-7)
	{   fprintf(fplog, "Warning: Weff for CV = %g may be too small.\n",
		    pParam->BSIM3V2weffCV);
	    printf("Warning: Weff for CV = %g may be too small.\n",
		   pParam->BSIM3V2weffCV);
	}        
	
/* Check threshold voltage parameters */
	if (pParam->BSIM3V2nlx < 0.0)
	{   fprintf(fplog, "Warning: Nlx = %g is negative.\n", pParam->BSIM3V2nlx);
	   printf("Warning: Nlx = %g is negative.\n", pParam->BSIM3V2nlx);
        }
	 if (model->BSIM3V2tox < 1.0e-9)
	{   fprintf(fplog, "Warning: Tox = %g is less than 10A.\n",
	            model->BSIM3V2tox);
	    printf("Warning: Tox = %g is less than 10A.\n", model->BSIM3V2tox);
        }

        if (pParam->BSIM3V2npeak <= 1.0e15)
	{   fprintf(fplog, "Warning: Nch = %g may be too small.\n",
	            pParam->BSIM3V2npeak);
	    printf("Warning: Nch = %g may be too small.\n",
	           pParam->BSIM3V2npeak);
	}
	else if (pParam->BSIM3V2npeak >= 1.0e21)
	{   fprintf(fplog, "Warning: Nch = %g may be too large.\n",
	            pParam->BSIM3V2npeak);
	    printf("Warning: Nch = %g may be too large.\n",
	           pParam->BSIM3V2npeak);
	}

	 if (pParam->BSIM3V2nsub <= 1.0e14)
	{   fprintf(fplog, "Warning: Nsub = %g may be too small.\n",
	            pParam->BSIM3V2nsub);
	    printf("Warning: Nsub = %g may be too small.\n",
	           pParam->BSIM3V2nsub);
	}
	else if (pParam->BSIM3V2nsub >= 1.0e21)
	{   fprintf(fplog, "Warning: Nsub = %g may be too large.\n",
	            pParam->BSIM3V2nsub);
	    printf("Warning: Nsub = %g may be too large.\n",
	           pParam->BSIM3V2nsub);
	}

	if ((pParam->BSIM3V2ngate > 0.0) &&
	    (pParam->BSIM3V2ngate <= 1.e18))
	{   fprintf(fplog, "Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	            pParam->BSIM3V2ngate);
	    printf("Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	           pParam->BSIM3V2ngate);
	}
       
        if (pParam->BSIM3V2dvt0 < 0.0)
	{   fprintf(fplog, "Warning: Dvt0 = %g is negative.\n",
		    pParam->BSIM3V2dvt0);   
	    printf("Warning: Dvt0 = %g is negative.\n", pParam->BSIM3V2dvt0);   
	}
	    
	if (fabs(1.0e-6 / (pParam->BSIM3V2w0 + pParam->BSIM3V2weff)) > 10.0)
	{   fprintf(fplog, "Warning: (W0 + Weff) may be too small.\n");
	    printf("Warning: (W0 + Weff) may be too small.\n");
        }

/* Check subthreshold parameters */
	if (pParam->BSIM3V2nfactor < 0.0)
	{   fprintf(fplog, "Warning: Nfactor = %g is negative.\n",
		    pParam->BSIM3V2nfactor);
	    printf("Warning: Nfactor = %g is negative.\n", pParam->BSIM3V2nfactor);
	}
	if (pParam->BSIM3V2cdsc < 0.0)
	{   fprintf(fplog, "Warning: Cdsc = %g is negative.\n",
		    pParam->BSIM3V2cdsc);
	    printf("Warning: Cdsc = %g is negative.\n", pParam->BSIM3V2cdsc);
	}
	if (pParam->BSIM3V2cdscd < 0.0)
	{   fprintf(fplog, "Warning: Cdscd = %g is negative.\n",
		    pParam->BSIM3V2cdscd);
	    printf("Warning: Cdscd = %g is negative.\n", pParam->BSIM3V2cdscd);
	}
/* Check DIBL parameters */
	if (pParam->BSIM3V2eta0 < 0.0)
	{   fprintf(fplog, "Warning: Eta0 = %g is negative.\n",
		    pParam->BSIM3V2eta0); 
	    printf("Warning: Eta0 = %g is negative.\n", pParam->BSIM3V2eta0); 
	}
	      
/* Check Abulk parameters */	    
	 if (fabs(1.0e-6 / (pParam->BSIM3V2b1 + pParam->BSIM3V2weff)) > 10.0)
       	{   fprintf(fplog, "Warning: (B1 + Weff) may be too small.\n");
       	    printf("Warning: (B1 + Weff) may be too small.\n");
        }    
    

/* Check Saturation parameters */
     	if (pParam->BSIM3V2a2 < 0.01)
	{   fprintf(fplog, "Warning: A2 = %g is too small. Set to 0.01.\n", pParam->BSIM3V2a2);
	    printf("Warning: A2 = %g is too small. Set to 0.01.\n",
		   pParam->BSIM3V2a2);
	    pParam->BSIM3V2a2 = 0.01;
	}
	else if (pParam->BSIM3V2a2 > 1.0)
	{   fprintf(fplog, "Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		    pParam->BSIM3V2a2);
	    printf("Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		   pParam->BSIM3V2a2);
	    pParam->BSIM3V2a2 = 1.0;
	    pParam->BSIM3V2a1 = 0.0;

	}

	if (pParam->BSIM3V2rdsw < 0.0)
	{   fprintf(fplog, "Warning: Rdsw = %g is negative. Set to zero.\n",
		    pParam->BSIM3V2rdsw);
	    printf("Warning: Rdsw = %g is negative. Set to zero.\n",
		   pParam->BSIM3V2rdsw);
	    pParam->BSIM3V2rdsw = 0.0;
	    pParam->BSIM3V2rds0 = 0.0;
	}
	else if ((pParam->BSIM3V2rds0 > 0.0) && (pParam->BSIM3V2rds0 < 0.001))
	{   fprintf(fplog, "Warning: Rds at current temperature = %g is less than 0.001 ohm. Set to zero.\n",
		    pParam->BSIM3V2rds0);
	    printf("Warning: Rds at current temperature = %g is less than 0.001 ohm. Set to zero.\n",
		   pParam->BSIM3V2rds0);
	    pParam->BSIM3V2rds0 = 0.0;
	}
	 if (pParam->BSIM3V2vsattemp < 1.0e3)
	{   fprintf(fplog, "Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM3V2vsattemp);
	   printf("Warning: Vsat at current temperature = %g may be too small.\n", pParam->BSIM3V2vsattemp);
	}

	if (pParam->BSIM3V2pdibl1 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl1 = %g is negative.\n",
		    pParam->BSIM3V2pdibl1);
	    printf("Warning: Pdibl1 = %g is negative.\n", pParam->BSIM3V2pdibl1);
	}
	if (pParam->BSIM3V2pdibl2 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl2 = %g is negative.\n",
		    pParam->BSIM3V2pdibl2);
	    printf("Warning: Pdibl2 = %g is negative.\n", pParam->BSIM3V2pdibl2);
	}
/* Check overlap capacitance parameters */
        if (model->BSIM3V2cgdo < 0.0)
	{   fprintf(fplog, "Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM3V2cgdo);
	    printf("Warning: cgdo = %g is negative. Set to zero.\n", model->BSIM3V2cgdo);
	    model->BSIM3V2cgdo = 0.0;
        }      
        if (model->BSIM3V2cgso < 0.0)
	{   fprintf(fplog, "Warning: cgso = %g is negative. Set to zero.\n", model->BSIM3V2cgso);
	    printf("Warning: cgso = %g is negative. Set to zero.\n", model->BSIM3V2cgso);
	    model->BSIM3V2cgso = 0.0;
        }      
        if (model->BSIM3V2cgbo < 0.0)
	{   fprintf(fplog, "Warning: cgbo = %g is negative. Set to zero.\n", model->BSIM3V2cgbo);
	    printf("Warning: cgbo = %g is negative. Set to zero.\n", model->BSIM3V2cgbo);
	    model->BSIM3V2cgbo = 0.0;
        }

     }/* loop for the parameter check for warning messages */      
	fclose(fplog);
    }
    else
    {   fprintf(stderr, "Warning: Can't open log file. Parameter checking skipped.\n");
    }

    return(Fatal_Flag);
}

