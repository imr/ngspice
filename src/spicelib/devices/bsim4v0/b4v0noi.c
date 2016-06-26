/**** BSIM4.0.0, Released by Weidong Liu 3/24/2000 ****/

/**********
 * Copyright 2000 Regents of the University of California. All rights reserved.
 * File: b4noi.c of BSIM4.0.0.
 * Authors: Weidong Liu, Xiaodong Jin, Kanyu M. Cao, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v0def.h"
#include "ngspice/cktdefs.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/noisedef.h"
#include "ngspice/suffix.h"
#include "ngspice/const.h"


/*
 * WDL: 1/f noise model has been smoothed out and enhanced with
 * bulk charge effect as well as physical N* equ. and necessary
 * conversion into the SI unit system.
 */

double
Eval1ovFNoise(Vds, model, here, freq, temp)
double Vds, freq, temp;
BSIM4v0model *model;
BSIM4v0instance *here;
{
struct bsim4v0SizeDependParam *pParam;
double cd, esat, DelClm, EffFreq, N0, Nl;
double T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, Ssi;

    pParam = here->pParam;
    cd = fabs(here->BSIM4v0cd);
    esat = 2.0 * pParam->BSIM4v0vsattemp / here->BSIM4v0ueff;
    T0 = ((((Vds - here->BSIM4v0Vdseff) / pParam->BSIM4v0litl)
       + model->BSIM4v0em) / esat);
    DelClm = pParam->BSIM4v0litl * log (MAX(T0, N_MINLOG));
    EffFreq = pow(freq, model->BSIM4v0ef);
    T1 = CHARGE * CHARGE * CONSTboltz * cd * temp * here->BSIM4v0ueff;
    T2 = 1.0e10 * EffFreq * here->BSIM4v0Abulk * model->BSIM4v0coxe
       * pParam->BSIM4v0leff * pParam->BSIM4v0leff;
    N0 = model->BSIM4v0coxe * here->BSIM4v0Vgsteff / CHARGE;
    Nl = model->BSIM4v0coxe * here->BSIM4v0Vgsteff
       * (1.0 - here->BSIM4v0AbovVgst2Vtm * here->BSIM4v0Vdseff) / CHARGE;

    T3 = model->BSIM4v0oxideTrapDensityA
       * log(MAX(((N0 + here->BSIM4v0nstar) / (Nl + here->BSIM4v0nstar)), N_MINLOG));
    T4 = model->BSIM4v0oxideTrapDensityB * (N0 - Nl);
    T5 = model->BSIM4v0oxideTrapDensityC * 0.5 * (N0 * N0 - Nl * Nl);

    T6 = CONSTboltz * temp * cd * cd;
    T7 = 1.0e10 * EffFreq * pParam->BSIM4v0leff
       * pParam->BSIM4v0leff * pParam->BSIM4v0weff;
    T8 = model->BSIM4v0oxideTrapDensityA + model->BSIM4v0oxideTrapDensityB * Nl
       + model->BSIM4v0oxideTrapDensityC * Nl * Nl;
    T9 = (Nl + here->BSIM4v0nstar) * (Nl + here->BSIM4v0nstar);
    Ssi = T1 / T2 * (T3 + T4 + T5) + T6 / T7 * DelClm * T8 / T9;
    return Ssi;
}


int
BSIM4v0noise (mode, operation, inModel, ckt, data, OnDens)
int mode, operation;
GENmodel *inModel;
CKTcircuit *ckt;
register Ndata *data;
double *OnDens;
{
register BSIM4v0model *model = (BSIM4v0model *)inModel;
register BSIM4v0instance *here;
struct bsim4v0SizeDependParam *pParam;
char name[N_MXVLNTH];
double tempOnoise;
double tempInoise;
double noizDens[BSIM4v0NSRCS];
double lnNdens[BSIM4v0NSRCS];

double N0, Nl;
double T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13;
double Vds, n, ExpArg, Ssi, Swi;
double tmp, gdpr, gspr, npart_theta, npart_beta, igsquare;

int error, i;

    /* define the names of the noise sources */
    static char *BSIM4v0nNames[BSIM4v0NSRCS] =
    {   /* Note that we have to keep the order */
	".rd",              /* noise due to rd */
	".rs",              /* noise due to rs */
        ".rg",              /* noise due to rgeltd */
        ".rbps",            /* noise due to rbps */
        ".rbpd",            /* noise due to rbpd */
        ".rbpb",            /* noise due to rbpb */
        ".rbsb",            /* noise due to rbsb */
        ".rbdb",            /* noise due to rbdb */
	".id",              /* noise due to id */
	".1overf",          /* flicker (1/f) noise */
        ".igs",             /* shot noise due to IGS */
        ".igd",             /* shot noise due to IGD */
        ".igb",             /* shot noise due to IGB */
	""                  /* total transistor noise */
    };

    for (; model != NULL; model = model->BSIM4v0nextModel)
    {    for (here = model->BSIM4v0instances; here != NULL;
	      here = here->BSIM4v0nextInstance)
	 {    pParam = here->pParam;
	      switch (operation)
	      {  case N_OPEN:
		     /* see if we have to to produce a summary report */
		     /* if so, name all the noise generators */

		      if (((NOISEAN*)ckt->CKTcurJob)->NStpsSm != 0)
		      {   switch (mode)
			  {  case N_DENS:
			          for (i = 0; i < BSIM4v0NSRCS; i++)
				  {    (void) sprintf(name, "onoise.%s%s",
					              here->BSIM4v0name,
						      BSIM4v0nNames[i]);
                                       data->namelist = (IFuid *) trealloc(
					     (char *) data->namelist,
					     (data->numPlots + 1)
					     * sizeof(IFuid));
                                       if (!data->namelist)
					   return(E_NOMEM);
		                       (*(SPfrontEnd->IFnewUid)) (ckt,
			                  &(data->namelist[data->numPlots++]),
			                  (IFuid) NULL, name, UID_OTHER,
					  NULL);
				       /* we've added one more plot */
			          }
			          break;
		             case INT_NOIZ:
			          for (i = 0; i < BSIM4v0NSRCS; i++)
				  {    (void) sprintf(name, "onoise_total.%s%s",
						      here->BSIM4v0name,
						      BSIM4v0nNames[i]);
                                       data->namelist = (IFuid *) trealloc(
					     (char *) data->namelist,
					     (data->numPlots + 1)
					     * sizeof(IFuid));
                                       if (!data->namelist)
					   return(E_NOMEM);
		                       (*(SPfrontEnd->IFnewUid)) (ckt,
			                  &(data->namelist[data->numPlots++]),
			                  (IFuid) NULL, name, UID_OTHER,
					  NULL);
				       /* we've added one more plot */

			               (void) sprintf(name, "inoise_total.%s%s",
						      here->BSIM4v0name,
						      BSIM4v0nNames[i]);
                                       data->namelist = (IFuid *) trealloc(
					     (char *) data->namelist,
					     (data->numPlots + 1)
					     * sizeof(IFuid));
                                       if (!data->namelist)
					   return(E_NOMEM);
		                       (*(SPfrontEnd->IFnewUid)) (ckt,
			                  &(data->namelist[data->numPlots++]),
			                  (IFuid) NULL, name, UID_OTHER,
					  NULL);
				       /* we've added one more plot */
			          }
			          break;
		          }
		      }
		      break;
	         case N_CALC:
		      switch (mode)
		      {  case N_DENS:
			      if (model->BSIM4v0tnoiMod == 0)
			      {   if (model->BSIM4v0rdsMod == 0)
				  {   gspr = here->BSIM4v0sourceConductance;
                                      gdpr = here->BSIM4v0drainConductance;
				      if (here->BSIM4v0grdsw > 0.0)
				          tmp = 1.0 / here->BSIM4v0grdsw; /* tmp used below */ 
				      else
					  tmp = 0.0;
				  }
				  else
				  {   gspr = here->BSIM4v0gstot;
                                      gdpr = here->BSIM4v0gdtot;
                                      tmp = 0.0;
				  }
			      }
			      else
			      {   T5 = here->BSIM4v0Vgsteff / here->BSIM4v0EsatL;
				  T5 *= T5;
				  npart_beta = 0.577 * (1.0 + T5
					     * model->BSIM4v0tnoia * pParam->BSIM4v0leff);
				  npart_theta = 0.37 * (1.0 + T5
                                              * model->BSIM4v0tnoib * pParam->BSIM4v0leff);

				  if (model->BSIM4v0rdsMod == 0)
                                  {   gspr = here->BSIM4v0sourceConductance;
                                      gdpr = here->BSIM4v0drainConductance;
                                  }
                                  else
                                  {   gspr = here->BSIM4v0gstot;
                                      gdpr = here->BSIM4v0gdtot;
                                  }

				  if ((*(ckt->CKTstates[0] + here->BSIM4v0vds)) >= 0.0)
			              gspr = gspr * (1.0 + npart_theta * npart_theta * gspr
				 	   / here->BSIM4v0IdovVds);
				  else
				      gdpr = gdpr * (1.0 + npart_theta * npart_theta * gdpr
                                           / here->BSIM4v0IdovVds);
			      } 

		              NevalSrc(&noizDens[BSIM4v0RDNOIZ],
				       &lnNdens[BSIM4v0RDNOIZ], ckt, THERMNOISE,
				       here->BSIM4v0dNodePrime, here->BSIM4v0dNode,
				       gdpr);

		              NevalSrc(&noizDens[BSIM4v0RSNOIZ],
				       &lnNdens[BSIM4v0RSNOIZ], ckt, THERMNOISE,
				       here->BSIM4v0sNodePrime, here->BSIM4v0sNode,
				       gspr);


			      if ((here->BSIM4v0rgateMod == 1) || (here->BSIM4v0rgateMod == 2))
			      {   NevalSrc(&noizDens[BSIM4v0RGNOIZ],
                                       &lnNdens[BSIM4v0RGNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v0gNodePrime, here->BSIM4v0gNodeExt,
                                       here->BSIM4v0grgeltd);
			      }
			      else if (here->BSIM4v0rgateMod == 3)
			      {   NevalSrc(&noizDens[BSIM4v0RGNOIZ],
                                       &lnNdens[BSIM4v0RGNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v0gNodeMid, here->BSIM4v0gNodeExt,
                                       here->BSIM4v0grgeltd);
			      }
			      else
			      {    noizDens[BSIM4v0RGNOIZ] = 0.0;
                                   lnNdens[BSIM4v0RGNOIZ] =
                                          log(MAX(noizDens[BSIM4v0RGNOIZ], N_MINLOG));
			      }


                              if (here->BSIM4v0rbodyMod)
                              {   NevalSrc(&noizDens[BSIM4v0RBPSNOIZ],
                                       &lnNdens[BSIM4v0RBPSNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v0bNodePrime, here->BSIM4v0sbNode,
                                       here->BSIM4v0grbps);
                                  NevalSrc(&noizDens[BSIM4v0RBPDNOIZ],
                                       &lnNdens[BSIM4v0RBPDNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v0bNodePrime, here->BSIM4v0dbNode,
                                       here->BSIM4v0grbpd);
                                  NevalSrc(&noizDens[BSIM4v0RBPBNOIZ],
                                       &lnNdens[BSIM4v0RBPBNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v0bNodePrime, here->BSIM4v0bNode,
                                       here->BSIM4v0grbpb);
                                  NevalSrc(&noizDens[BSIM4v0RBSBNOIZ],
                                       &lnNdens[BSIM4v0RBSBNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v0bNode, here->BSIM4v0sbNode,
                                       here->BSIM4v0grbsb);
                                  NevalSrc(&noizDens[BSIM4v0RBDBNOIZ],
                                       &lnNdens[BSIM4v0RBDBNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v0bNode, here->BSIM4v0dbNode,
                                       here->BSIM4v0grbdb);
                              }
                              else
                              {   noizDens[BSIM4v0RBPSNOIZ] = noizDens[BSIM4v0RBPDNOIZ] = 0.0;   
                                  noizDens[BSIM4v0RBPBNOIZ] = 0.0;
                                  noizDens[BSIM4v0RBSBNOIZ] = noizDens[BSIM4v0RBDBNOIZ] = 0.0;
                                  lnNdens[BSIM4v0RBPSNOIZ] =
                                          log(MAX(noizDens[BSIM4v0RBPSNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v0RBPDNOIZ] =
                                          log(MAX(noizDens[BSIM4v0RBPDNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v0RBPBNOIZ] =
                                          log(MAX(noizDens[BSIM4v0RBPBNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v0RBSBNOIZ] =
                                          log(MAX(noizDens[BSIM4v0RBSBNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v0RBDBNOIZ] =
                                          log(MAX(noizDens[BSIM4v0RBDBNOIZ], N_MINLOG));
                              }


                              switch(model->BSIM4v0tnoiMod)
			      {  case 0:
				      T0 = here->BSIM4v0ueff * fabs(here->BSIM4v0qinv);
				      T1 = T0 * tmp + pParam->BSIM4v0leff
                                         * pParam->BSIM4v0leff;
		                      NevalSrc(&noizDens[BSIM4v0IDNOIZ],
				               &lnNdens[BSIM4v0IDNOIZ], ckt,
					       THERMNOISE, here->BSIM4v0dNodePrime,
                                               here->BSIM4v0sNodePrime,
					       (T0 / T1) * model->BSIM4v0ntnoi);
				      break;
				 case 1:
				      T0 = here->BSIM4v0gm + here->BSIM4v0gmbs + here->BSIM4v0gds;
				      T0 *= T0;
				      igsquare = npart_theta * npart_theta * T0 / here->BSIM4v0IdovVds;
				      T1 = npart_beta * (here->BSIM4v0gm
					 + here->BSIM4v0gmbs) + here->BSIM4v0gds;
				      T2 = T1 * T1 / here->BSIM4v0IdovVds;
                                      NevalSrc(&noizDens[BSIM4v0IDNOIZ],
                                               &lnNdens[BSIM4v0IDNOIZ], ckt,
                                               THERMNOISE, here->BSIM4v0dNodePrime,
                                               here->BSIM4v0sNodePrime, (T2 - igsquare));
                                      break;
			      }

		              NevalSrc(&noizDens[BSIM4v0FLNOIZ], (double*) NULL,
				       ckt, N_GAIN, here->BSIM4v0dNodePrime,
				       here->BSIM4v0sNodePrime, (double) 0.0);

                              switch(model->BSIM4v0fnoiMod)
			      {  case 0:
			              noizDens[BSIM4v0FLNOIZ] *= model->BSIM4v0kf
					    * exp(model->BSIM4v0af
					    * log(MAX(fabs(here->BSIM4v0cd),
					    N_MINLOG)))
					    / (pow(data->freq, model->BSIM4v0ef)
					    * pParam->BSIM4v0leff
				            * pParam->BSIM4v0leff
					    * model->BSIM4v0coxe);
				      break;
			         case 1:
		                      Vds = *(ckt->CKTstates[0] + here->BSIM4v0vds);
			              if (Vds < 0.0)
			                  Vds = -Vds;

                                      Ssi = Eval1ovFNoise(Vds, model, here,
                                          data->freq, ckt->CKTtemp);
                                      T10 = model->BSIM4v0oxideTrapDensityA
                                          * CONSTboltz * ckt->CKTtemp;
                                      T11 = pParam->BSIM4v0weff * pParam->BSIM4v0leff
                                          * pow(data->freq, model->BSIM4v0ef) * 1.0e10
					  * here->BSIM4v0nstar * here->BSIM4v0nstar;
                                      Swi = T10 / T11 * here->BSIM4v0cd
                                          * here->BSIM4v0cd;
                                      T1 = Swi + Ssi;
                                      if (T1 > 0.0)
                                          noizDens[BSIM4v0FLNOIZ] *= (Ssi * Swi) / T1;
                                      else
                                          noizDens[BSIM4v0FLNOIZ] *= 0.0;
				      break;
			      }

		              lnNdens[BSIM4v0FLNOIZ] =
				     log(MAX(noizDens[BSIM4v0FLNOIZ], N_MINLOG));


                              NevalSrc(&noizDens[BSIM4v0IGSNOIZ],
                                   &lnNdens[BSIM4v0IGSNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v0gNodePrime, here->BSIM4v0sNodePrime,
                                   (here->BSIM4v0Igs + here->BSIM4v0Igcs));
                              NevalSrc(&noizDens[BSIM4v0IGDNOIZ],
                                   &lnNdens[BSIM4v0IGDNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v0gNodePrime, here->BSIM4v0dNodePrime,
                                   (here->BSIM4v0Igd + here->BSIM4v0Igcd));

                              NevalSrc(&noizDens[BSIM4v0IGBNOIZ],
                                   &lnNdens[BSIM4v0IGBNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v0gNodePrime, here->BSIM4v0bNodePrime,
                                   here->BSIM4v0Igb);


		              noizDens[BSIM4v0TOTNOIZ] = noizDens[BSIM4v0RDNOIZ]
				     + noizDens[BSIM4v0RSNOIZ] + noizDens[BSIM4v0RGNOIZ]
				     + noizDens[BSIM4v0RBPSNOIZ] + noizDens[BSIM4v0RBPDNOIZ]
				     + noizDens[BSIM4v0RBPBNOIZ]
				     + noizDens[BSIM4v0RBSBNOIZ] + noizDens[BSIM4v0RBDBNOIZ]
				     + noizDens[BSIM4v0IDNOIZ] + noizDens[BSIM4v0FLNOIZ]
                                     + noizDens[BSIM4v0IGSNOIZ] + noizDens[BSIM4v0IGDNOIZ]
                                     + noizDens[BSIM4v0IGBNOIZ];
		              lnNdens[BSIM4v0TOTNOIZ] = 
				     log(MAX(noizDens[BSIM4v0TOTNOIZ], N_MINLOG));

		              *OnDens += noizDens[BSIM4v0TOTNOIZ];

		              if (data->delFreq == 0.0)
			      {   /* if we haven't done any previous 
				     integration, we need to initialize our
				     "history" variables.
				    */

			          for (i = 0; i < BSIM4v0NSRCS; i++)
				  {    here->BSIM4v0nVar[LNLSTDENS][i] =
					     lnNdens[i];
			          }

			          /* clear out our integration variables
				     if it's the first pass
				   */
			          if (data->freq ==
				      ((NOISEAN*) ckt->CKTcurJob)->NstartFreq)
				  {   for (i = 0; i < BSIM4v0NSRCS; i++)
				      {    here->BSIM4v0nVar[OUTNOIZ][i] = 0.0;
				           here->BSIM4v0nVar[INNOIZ][i] = 0.0;
			              }
			          }
		              }
			      else
			      {   /* data->delFreq != 0.0,
				     we have to integrate.
				   */
			          for (i = 0; i < BSIM4v0NSRCS; i++)
				  {    if (i != BSIM4v0TOTNOIZ)
				       {   tempOnoise = Nintegrate(noizDens[i],
						lnNdens[i],
				                here->BSIM4v0nVar[LNLSTDENS][i],
						data);
				           tempInoise = Nintegrate(noizDens[i]
						* data->GainSqInv, lnNdens[i]
						+ data->lnGainInv,
				                here->BSIM4v0nVar[LNLSTDENS][i]
						+ data->lnGainInv, data);
				           here->BSIM4v0nVar[LNLSTDENS][i] =
						lnNdens[i];
				           data->outNoiz += tempOnoise;
				           data->inNoise += tempInoise;
				           if (((NOISEAN*)
					       ckt->CKTcurJob)->NStpsSm != 0)
					   {   here->BSIM4v0nVar[OUTNOIZ][i]
						     += tempOnoise;
				               here->BSIM4v0nVar[OUTNOIZ][BSIM4v0TOTNOIZ]
						     += tempOnoise;
				               here->BSIM4v0nVar[INNOIZ][i]
						     += tempInoise;
				               here->BSIM4v0nVar[INNOIZ][BSIM4v0TOTNOIZ]
						     += tempInoise;
                                           }
			               }
			          }
		              }
		              if (data->prtSummary)
			      {   for (i = 0; i < BSIM4v0NSRCS; i++)
				  {    /* print a summary report */
			               data->outpVector[data->outNumber++]
					     = noizDens[i];
			          }
		              }
		              break;
		         case INT_NOIZ:
			      /* already calculated, just output */
		              if (((NOISEAN*)ckt->CKTcurJob)->NStpsSm != 0)
			      {   for (i = 0; i < BSIM4v0NSRCS; i++)
				  {    data->outpVector[data->outNumber++]
					     = here->BSIM4v0nVar[OUTNOIZ][i];
			               data->outpVector[data->outNumber++]
					     = here->BSIM4v0nVar[INNOIZ][i];
			          }
		              }
		              break;
		      }
		      break;
	         case N_CLOSE:
		      /* do nothing, the main calling routine will close */
		      return (OK);
		      break;   /* the plots */
	      }       /* switch (operation) */
	 }    /* for here */
    }    /* for model */

    return(OK);
}
