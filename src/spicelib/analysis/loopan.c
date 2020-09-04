/**********
Copyright 2020 Anamosic Ballenegger Design.  All rights reserved.
Author: 2020 Florian Ballenegger
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/acdefs.h"
#include "ngspice/loopdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"

#ifdef XSPICE
#include "ngspice/evt.h"
#include "ngspice/enh.h"
/* gtri - add - wbk - 12/19/90 - Add headers */ 
#include "ngspice/mif.h"
#include "ngspice/evtproto.h"
#include "ngspice/ipctiein.h"
/* gtri - end - wbk */
#endif


#define INIT_STATS() \
do { \
    startTime  = SPfrontEnd->IFseconds();       \
    startdTime = ckt->CKTstat->STATdecompTime;  \
    startsTime = ckt->CKTstat->STATsolveTime;   \
    startlTime = ckt->CKTstat->STATloadTime;    \
    startkTime = ckt->CKTstat->STATsyncTime;    \
} while(0)

#define UPDATE_STATS(analysis) \
do { \
    ckt->CKTcurrentAnalysis = analysis; \
    ckt->CKTstat->STATacTime += SPfrontEnd->IFseconds() - startTime; \
    ckt->CKTstat->STATacDecompTime += ckt->CKTstat->STATdecompTime - startdTime; \
    ckt->CKTstat->STATacSolveTime += ckt->CKTstat->STATsolveTime - startsTime; \
    ckt->CKTstat->STATacLoadTime += ckt->CKTstat->STATloadTime - startlTime; \
    ckt->CKTstat->STATacSyncTime += ckt->CKTstat->STATsyncTime - startkTime; \
} while(0)

extern SPICEdev **DEVices;

int
LOOPpreset(CKTcircuit *ckt, JOB *anal)
{
    LOOPAN *job = (LOOPAN *) anal;
    GENinstance* inst;
    GENinstance* probesrc;
    GENmodel*    modvsrc;
    CKTnode *nodeinj, *node;
    IFdevice* dev;
    IFuid    eltUid;
    IFuid    modUid;
    IFvalue  ptemp;
    int      termidx;
    int      error;
    int      vtype;
    
    printf("LOOPpreset with direction %d!\n", job->LOOPdirection);
    
    if(job->LOOPportnameGiven || job->LOOPportnumGiven)
    {
       inst = CKTfndDev(ckt, job->LOOPprobeSrc);
       if (!inst || inst->GENmodPtr->GENmodType < 0) {
        SPfrontEnd->IFerrorf (ERR_WARNING,
                             "Loop probe source %s not in circuit",
                             job->LOOPprobeSrc);
        return E_NOTFOUND;
       }
       dev = &DEVices[inst->GENmodPtr->GENmodType]->DEVpublic;
       if (inst->GENmodPtr->GENmodType == CKTtypelook("Vsource"))
          printf("Consider to specify the whole Vsource %s as loop probe\n", job->LOOPprobeSrc);
			     
       if(job->LOOPportnameGiven) {
	   for(termidx=0;termidx<(*(dev->numNames));termidx++) {
	     if(0) printf("cmp %s with %s\n", job->LOOPportname, dev->termNames[termidx]);
	     if(strcasecmp(job->LOOPportname, dev->termNames[termidx])==0)
	       break;
	   }
       } else
	   termidx = job->LOOPportnum-1; /* LOOPportnum counts from 1, termidx counts from 0 */
       if(termidx<0 || termidx>=*(dev->numNames)) {
           SPfrontEnd->IFerrorf (ERR_WARNING, "No such terminal %d", termidx);
           return E_NOTFOUND;
       }
       /* now break the loop at terminal termidx */
       printf("Break the loop at device %s terminal %s\n", inst->GENname, dev->termNames[termidx]);
       node = CKTnum2nod(ckt, GENnode(inst)[termidx]);
       error = CKTmkVolt(ckt, &nodeinj, inst->GENname, "loopinj");
       if(error) return(error);
       error = SPfrontEnd->IFnewUid (ckt, &eltUid, inst->GENname,
                        "probe", UID_INSTANCE, NULL);
       if(error) return(error);
       error = SPfrontEnd->IFnewUid (ckt, &modUid, inst->GENname,
                        "probemod", UID_MODEL, NULL);
       if(error) return(error);
       vtype = CKTtypelook("Vsource");
       modvsrc = NULL;
       error = CKTmodCrt(ckt,vtype,&modvsrc, modUid);
       if(error) return(error);
       error = CKTcrtElt(ckt, modvsrc, &probesrc, eltUid);
       if(error) return(error);
       ptemp.rValue = 0;
       error = CKTpName("dc",&ptemp,ckt,vtype,"probe",&probesrc);
       if(error) return(error);
       error = CKTbindNode(ckt,probesrc,job->LOOPdirection==2 ? 2 : 1,nodeinj);
       if(error) return(error);
       error = CKTbindNode(ckt,probesrc,job->LOOPdirection==2 ? 1 : 2 ,node);
       if(error) return(error);
       error = CKTbindNode(ckt,inst,termidx+1,nodeinj); /* bindNode counts from 1 ! */
       ptemp.rValue = 0;
       job->LOOPprobeSrc = probesrc->GENname;
       
    }
    return OK;
}

int
LOOPan(CKTcircuit *ckt, int restart)
{
    LOOPAN *job = (LOOPAN *) ckt->CKTcurJob;
    
    double freq;
    double freqTol; /* tolerence parameter for finding final frequency */
    double startdTime;
    double startsTime;
    double startlTime;
    double startkTime;
    double startTime;
    
    double Vy[3],Iy[3];
    double iVy[3],iIy[3];
    double phasemargin;
    double gainmargin;
    double gainsq, phase, curTr;
    
    int error;
    int numNames;
    IFuid *nameList;  /* va: tmalloc'ed list of names */
    IFuid freqUid;
    static runDesc *acPlot = NULL;
    runDesc *plot = NULL;
    GENinstance *inst = NULL;
    int size, i;
    int probe_brnum;



    inst = CKTfndDev(ckt, job->LOOPprobeSrc);
    
    if (!inst || inst->GENmodPtr->GENmodType < 0) {
        SPfrontEnd->IFerrorf (ERR_WARNING,
                             "Loop probe source %s not in circuit",
                             job->LOOPprobeSrc);
        return E_NOTFOUND;
    }
    if (inst->GENmodPtr->GENmodType != CKTtypelook("Vsource"))
    {
        SPfrontEnd->IFerrorf (ERR_WARNING,
                             "Loop probe source %s not of proper type",
                             job->LOOPprobeSrc);
        return E_NOTFOUND;
    }
    printf("LOOPan Vy at %s\n", CKTnodName(ckt,GENnode(inst)[0]));
    probe_brnum = CKTfndBranch(ckt, job->LOOPprobeSrc);
    phasemargin = 180;
    gainmargin = strtod("NaN", NULL);
    
#ifdef XSPICE
/* gtri - add - wbk - 12/19/90 - Add IPC stuff and anal_init and anal_type */

    /* Tell the beginPlot routine what mode we're in */
    g_ipc.anal_type = IPC_ANAL_AC;

    /* Tell the code models what mode we're in */
    g_mif_info.circuit.anal_type = MIF_DC;
    g_mif_info.circuit.anal_init = MIF_TRUE;

/* gtri - end - wbk */
#endif

    /* start at beginning */
    if (job->LOOPsaveFreq == 0 || restart) {
        if (job->LOOPnumSteps < 1)
            job->LOOPnumSteps = 1;

        switch (job->LOOPstepType) {

        case DECADE:
            if (job->LOOPstartFreq <= 0) {
                fprintf(stderr, "ERROR: LOOP startfreq <= 0\n");
                return E_PARMVAL;
            }
            job->LOOPfreqDelta =
                exp(log(10.0)/job->LOOPnumSteps);
            break;
        case OCTAVE:
            if (job->LOOPstartFreq <= 0) {
                fprintf(stderr, "ERROR: LOOP startfreq <= 0\n");
                return E_PARMVAL;
            }
            job->LOOPfreqDelta =
                exp(log(2.0)/job->LOOPnumSteps);
            break;
        case LINEAR:
            if (job->LOOPnumSteps-1 > 1)
                job->LOOPfreqDelta =
                    (job->LOOPstopFreq -
                     job->LOOPstartFreq) /
                    (job->LOOPnumSteps - 1);
            else
            /* Patch from: Richard McRoberts
            * This patch is for a rather pathological case:
            * a linear step with only one point */
                job->LOOPfreqDelta = 0;
            break;
        default:
            return(E_BADPARM);
    }
#ifdef XSPICE
/* gtri - begin - wbk - Call EVTop if event-driven instances exist */

    if(ckt->evt->counts.num_insts != 0) {
        error = EVTop(ckt,
            (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
            (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
            ckt->CKTdcMaxIter,
            MIF_TRUE);
        EVTdump(ckt, IPC_ANAL_DCOP, 0.0);
        EVTop_save(ckt, MIF_TRUE, 0.0);
    }
    else 
#endif 
    /* If no event-driven instances, do what SPICE normally does */
    if (!ckt->CKTnoopac) { /* skip OP if option NOOPAC is set and circuit is linear */
        error = CKTop(ckt,
            (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
            (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
            ckt->CKTdcMaxIter);

        if(error){
            fprintf(stdout,"\nLOOP operating point failed -\n");
            CKTncDump(ckt);
            return(error);
        }
    }
    else
        fprintf(stdout,"\n Linear circuit, option noopac given: no OP analysis\n");
		
#ifdef XSPICE
/* gtri - add - wbk - 12/19/90 - Add IPC stuff */

    /* Send the operating point results for Mspice compatibility */
    if(g_ipc.enabled) 
    {
        /* Call CKTnames to get names of nodes/branches used by 
            BeginPlot */
        /* Probably should free nameList after this block since 
            called again... */
        error = CKTnames(ckt,&numNames,&nameList);
        if(error) return(error);

        /* We have to do a beginPlot here since the data to return is
         * different for the DCOP than it is for the AC analysis.  
         * Moreover the begin plot has not even been done yet at this 
         * point... 
         */
        SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob,
                                   ckt->CKTcurJob->JOBname,
                                   NULL, IF_REAL,
                                   numNames, nameList, IF_REAL,
                                   &acPlot);
        txfree(nameList);

        ipc_send_dcop_prefix();
        CKTdump(ckt, 0.0, acPlot);
        ipc_send_dcop_suffix();

        SPfrontEnd->OUTendPlot (acPlot);
    }
/* gtri - end - wbk */
#endif

        ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
        error = CKTload(ckt);
        if(error) return(error);

        error = CKTnames(ckt,&numNames,&nameList);
        if(error) return(error);

	if (ckt->CKTkeepOpInfo) {
	    /* Dump operating point. */
            error = SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob,
                                               "LOOP Operating Point",
                                               NULL, IF_REAL,
                                               numNames, nameList, IF_REAL,
                                               &plot);
	    if(error) return(error);
	    CKTdump(ckt, 0.0, plot);
	    SPfrontEnd->OUTendPlot (plot);
	    plot = NULL;
	}

        SPfrontEnd->IFnewUid (ckt, &freqUid, NULL, "frequency", UID_OTHER, NULL);
        error = SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob,
                                           ckt->CKTcurJob->JOBname,
                                           freqUid, IF_REAL,
                                           numNames, nameList, IF_COMPLEX,
                                           &acPlot);
	tfree(nameList);		
	if(error) return(error);

        if (job->LOOPstepType != LINEAR) {
	    SPfrontEnd->OUTattributes (acPlot, NULL, OUT_SCALE_LOG, NULL);
	}
        freq = job->LOOPstartFreq;

    } else {    /* continue previous analysis */
        freq = job->LOOPsaveFreq;
        job->LOOPsaveFreq = 0; /* clear the 'old' frequency */
	/* fix resume? saj, indeed !*/
        error = SPfrontEnd->OUTpBeginPlot (NULL, NULL,
                                           NULL,
                                           NULL, 0,
                                           666, NULL, 666,
                                           &acPlot);
	/* saj*/    
    }
        
    switch (job->LOOPstepType) {
    case DECADE:
    case OCTAVE:
        freqTol = job->LOOPfreqDelta *
            job->LOOPstopFreq * ckt->CKTreltol;
        break;
    case LINEAR:
        freqTol = job->LOOPfreqDelta * ckt->CKTreltol;
        break;
    default:
        return(E_BADPARM);
    }


#ifdef XSPICE
/* gtri - add - wbk - 12/19/90 - Set anal_init and anal_type */

    g_mif_info.circuit.anal_init = MIF_TRUE;

    /* Tell the code models what mode we're in */
    g_mif_info.circuit.anal_type = MIF_AC;

/* gtri - end - wbk */
#endif

    INIT_STATS();

    ckt->CKTcurrentAnalysis = DOING_AC;

    /* main loop through all scheduled frequencies */
    while (freq <= job->LOOPstopFreq + freqTol) {
        if(SPfrontEnd->IFpauseTest()) {
            /* user asked us to pause via an interrupt */
            job->LOOPsaveFreq = freq;
            return(E_PAUSE);
        }
        ckt->CKTomega = 2.0 * M_PI *freq;

        /* Update opertating point, if variable 'hertz' is given */
        if (ckt->CKTvarHertz) {
#ifdef XSPICE
            /* Call EVTop if event-driven instances exist */

            if(ckt->evt->counts.num_insts != 0) {
                error = EVTop(ckt,
                    (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
                    (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
                    ckt->CKTdcMaxIter,
                    MIF_TRUE);
                EVTdump(ckt, IPC_ANAL_DCOP, 0.0);
                EVTop_save(ckt, MIF_TRUE, 0.0);
            }
            else 
#endif 
// If no event-driven instances, do what SPICE normally does
                error = CKTop(ckt,
                    (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
                    (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
                    ckt->CKTdcMaxIter);

            if(error){
                fprintf(stdout,"\nLOOP operating point failed -\n");
                CKTncDump(ckt);
                return(error);
            } 
            ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
            error = CKTload(ckt);
            if(error) return(error);
        }

        ckt->CKTmode = (ckt->CKTmode&MODEUIC) | MODEAC;
	
	/************* Main part starts here ********************/
	size = SMPmatSize(ckt->CKTmatrix);
	for(int simno=0;simno<2;simno++)
	{
	  for (i=0; i<=size; i++)
	  {
	    ckt->CKTrhs[i] = 0.0;
	    ckt->CKTirhs[i] = 0.0;
	  }
	  error = CKTacLoad(ckt);
	  /* clear any AC stimulus set by acLoad */
	  for (i=1; i<=size; i++)
	  {
	    ckt->CKTrhs[i] = 0.0;
	    ckt->CKTirhs[i] = 0.0;
	  }
	  
	  switch(simno)
	  {
	  case 1:
	    /* AC current src magn=1 from ground to Vx */
	    ckt->CKTrhs[GENnode(inst)[1]] -= 1;
            ckt->CKTrhs[0] += 1;
	    break;
	  case 0: 
	    /* AC voltage src magn=1 accross probe */
	    ckt->CKTrhs[probe_brnum] += 1;
	    break;
	  default:
	    return(E_BADPARM);
	  }
	
	
	  /*error = NIacIter(ckt);*/
	  error = NIdIter(ckt);
          if (error) {
            UPDATE_STATS(DOING_AC);
            return(error);
          }
	  Vy[simno] = ckt->CKTrhsOld[GENnode(inst)[0]];
	  iVy[simno] = ckt->CKTirhsOld[GENnode(inst)[0]];
	  Iy[simno] = ckt->CKTrhsOld[probe_brnum];
	  iIy[simno] = ckt->CKTirhsOld[probe_brnum];
	  if(0) printf("|Vy| = %g, |Iy|=%g\n", sqrt(Vy[simno]*Vy[simno]+iVy[simno]*iVy[simno]), sqrt(Iy[simno]*Iy[simno]+iIy[simno]*iIy[simno]));
	}
	
	/* V(y)@1*I(Viy)@3-V(y)@3*I(Viy)@1 */
	{
	double D, iD, T, iT, ngainsq, nphase;
	
	D = Vy[0]*Iy[1] - iVy[0]*iIy[1] - Vy[1]*Iy[0] + iVy[1]*iIy[0];
	iD = Vy[0]*iIy[1] + iVy[0]*Iy[1] - Vy[1]*iIy[0] - iVy[1]*Iy[0];
	/*T = (D-iD*iD/(1-D))/(1-D+iD/(1-D));*/
	T = (D*(1-D)-iD*iD)/((1-D)*(1-D) + iD*iD);
	iT = iD/((1-D)*(1-D) + iD*iD);
	nphase = 180*atan2(-iD,-D)/M_PI;
	ngainsq = T*T + iT*iT;
	if((ngainsq-1)*(gainsq-1)<0)
	{ /* gain cross 1 - interpolate lin/log for more accurate results */
		double crossphase = (log(gainsq)*nphase -  
		  log(ngainsq)*phase)/(log(gainsq)-log(ngainsq));
		if(fabs(crossphase)<phasemargin)
		    phasemargin = fabs(crossphase);
		/* idea: binary search/sim + for frequency where gain=1 ? */
        }
	
	if(ngainsq>=1)
	if(fabs(nphase)<phasemargin)
		    phasemargin = fabs(nphase);
		    
        if(T>0 && curTr>0)
	if(nphase * phase <= 0)
	{
	     /* right quandrant and cross phase=0 */
	     double gmsqlog;
	     gmsqlog = (nphase*log(gainsq)-phase*log(ngainsq))/(nphase-phase);
	     gainmargin = 10*gmsqlog;
	}
	
	gainsq=ngainsq;
	phase=nphase;
	curTr = T; /* remember real part only */
		
	if(1) printf("f=%g |T| = %g, |D|=%g, phase=%g\n", freq, sqrt(T*T+iT*iT), sqrt(D*D+iD*iD), 180*atan2(-iD,-D)/M_PI);
	}
	
	
	/************* End of main part ********************/

#ifdef WANT_SENSE2
        if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode&ACSEN) ){
            long save;
            int save1;

            save = ckt->CKTmode;
            ckt->CKTmode=(ckt->CKTmode&MODEUIC)|MODEDCOP|MODEINITSMSIG;
            save1 = ckt->CKTsenInfo->SENmode;
            ckt->CKTsenInfo->SENmode = ACSEN;
            if (freq == job->LOOPstartFreq) {
                ckt->CKTsenInfo->SENacpertflag = 1;
            }
            else{
                ckt->CKTsenInfo->SENacpertflag = 0;
            }
            error = CKTsenAC(ckt);
            if (error)
                return (error);
            ckt->CKTmode = save;
            ckt->CKTsenInfo->SENmode = save1;
        }
#endif

#ifdef XSPICE
/* gtri - modify - wbk - 12/19/90 - Send IPC stuff */

        if(g_ipc.enabled)
            ipc_send_data_prefix(freq);

        error = CKTacDump(ckt,freq,acPlot);

        if(g_ipc.enabled)
            ipc_send_data_suffix();

/* gtri - modify - wbk - 12/19/90 - Send IPC stuff */
#else
        error = CKTacDump(ckt,freq,acPlot);
#endif	
        if (error) {
	    UPDATE_STATS(DOING_AC);
 	    return(error);
 	}

        /*  increment frequency */

        switch (job->LOOPstepType) {
        case DECADE:
        case OCTAVE:

/* inserted again 14.12.2001  */
#ifdef HAS_PROGREP
            {
                double endfreq   = job->LOOPstopFreq;
                double startfreq = job->LOOPstartFreq;
                endfreq   = log(endfreq);
                if (startfreq == 0.0)
                    startfreq = 1e-12;
                startfreq = log(startfreq);

                if (freq > 0.0)
                    SetAnalyse( "loop", (int)((log(freq)-startfreq) * 1000.0 / (endfreq-startfreq)));
            }
#endif

            freq *= job->LOOPfreqDelta;
            if (job->LOOPfreqDelta == 1) goto endsweep;
        break;
        case LINEAR:

#ifdef HAS_PROGREP
			 {
				 double endfreq   = job->LOOPstopFreq;
				 double startfreq = job->LOOPstartFreq;
				 SetAnalyse( "loop", (int)((freq - startfreq)* 1000.0 / (endfreq-startfreq)));
			 }
#endif

            freq += job->LOOPfreqDelta;
            if (job->LOOPfreqDelta == 0) goto endsweep;
            break;
        default:
            return(E_INTERN);
    
        }

    }
endsweep:
    printf("Loop analysis '%s': phase margin is %g [degrees], gain margin is %g [dB]\n", job->LOOPname ? job->LOOPname : "anon", phasemargin, gainmargin);
        
    SPfrontEnd->OUTendPlot (acPlot);
    acPlot = NULL;
    UPDATE_STATS(0);
    return(0);
}
