/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

/* Do a run of the circuit, of the given type. Type "resume" is special --
 * it means to resume whatever simulation that was in progress. The
 * return value of this routine is 0 if the exit was ok, and 1 if there was
 * a reason to interrupt the circuit (interrupt typed at the keyboard,
 * error in the simulation, etc). args should be the entire command line,
 * e.g. "tran 1 10 20 uic"
 */

#include "ngspice.h"
#include "cpdefs.h"
#include "ftedefs.h"
#include "fteinp.h"
#include "sim.h"
#include "devdefs.h"
#include "inpdefs.h"
#include "iferrmsg.h"
#include "ifsim.h"

#include "circuits.h"
#include "shyu.h"


int
if_sens_run(char *t, wordlist *args, INPtables *tab)
{
    void *ckt = (void *) t;
    void *senseJob;
    void *acJob;
    void *opJob;
    void *dcJob;
    void *tranJob;
    card *current;
    IFvalue ptemp;
    IFvalue *parm;
    char buf[BSIZE_SP];
    int err;
    char *token;
    char *steptype;
    char *name;
    char *line;
    struct line deck;
    int i;
    int j;
    int error;
    int save ;
    int flag = 0;
    int which = -1;

    (void) sprintf(buf, ".%s", wl_flatten(args));
    deck.li_next = deck.li_actual = NULL;
    deck.li_error = NULL;
    deck.li_linenum = 0;
    deck.li_line = buf;

    current = (card *) &deck;
    line = current->line;
    INPgetTok(&line,&token,1);

    if(ft_curckt->ci_specTask) {
        err=(*(ft_sim->deleteTask))(ft_curckt->ci_ckt,
        ft_curckt->ci_specTask);
        if(err) {
            ft_sperror(err,"deleteTask");
            return(0); /* temporary */
        }
    }
    err = (*(ft_sim->newTask))(ft_curckt->ci_ckt,
	    (void **) &(ft_curckt->ci_specTask),"special",
	    (void**)&(ft_curckt->ci_defTask));
    if(err) {
        ft_sperror(err,"newTask");
        return(0); /* temporary */
    }
    for(j=0;j<ft_sim->numAnalyses;j++) {
        if(strcmp(ft_sim->analyses[j]->name,"options")==0) {
            which = j;
            break;
        }
    } 
    if(which != -1) {
        err = (*(ft_sim->newAnalysis))(ft_curckt->ci_ckt,which,"options",
		(void **) &(ft_curckt->ci_specOpt),ft_curckt->ci_specTask);
        if(err) {
            ft_sperror(err,"createOptions");
            return(0);/* temporary */
        }
        ft_curckt->ci_curOpt  = ft_curckt->ci_specOpt;
    } 
    else ;
    { /* in DEEP trouble */
    }
    ft_curckt->ci_curTask = ft_curckt->ci_specTask;
    which = -1;
    for(j=0;j<ft_sim->numAnalyses;j++) {
        if(strcmp(ft_sim->analyses[j]->name,"SEN")==0) {
            which = j;
            break;
        }
    } 
    if(which != -1) {
        err = (*(ft_sim->newAnalysis))(ft_curckt->ci_ckt,which,"sense",
        &(senseJob),ft_curckt->ci_specTask);
        if(err) {
            ft_sperror(err,"createSense");
            return(0);/* temporary */
        }
    } 
    else{ 
        current->error = INPerrCat(current->error,INPmkTemp(
        "sensetivity analysis unsupported\n"));
        return(0);
    }
    save = which;
    INPgetTok(&line,&token,1);
    if(strcmp(token ,"ac")==0){
        which = -1;
        for(j=0;j<ft_sim->numAnalyses;j++) {
            if(strcmp(ft_sim->analyses[j]->name,"AC")==0) {
                which = j;
                break;
            }
        } 
        if(which != -1) {
            err = (*(ft_sim->newAnalysis))(ft_curckt->ci_ckt,which,"acan",
            &(acJob),ft_curckt->ci_specTask);
            if(err) {
                ft_sperror(err,"createAC"); /* or similar error message */
                return(0);
            }
        } 
        else{ 
            current->error = INPerrCat(current->error,INPmkTemp(
            "ac analysis unsupported\n"));
        }

        INPgetTok(&line,&steptype,1); /* get DEC, OCT, or LIN */
        ptemp.iValue=1;
        error = INPapName(ckt,which,acJob,steptype,&ptemp);
        if(error) current->error = INPerrCat(current->error,
        INPerror(error));
        parm=INPgetValue(ckt,&line,IF_INTEGER,tab);/* number of points*/
        error = INPapName(ckt,which,acJob,"numsteps",parm);
        if(error) current->error = INPerrCat(current->error,
        INPerror(error));
        parm = INPgetValue(ckt,&line,IF_REAL,tab); /* fstart */
        error = INPapName(ckt,which,acJob,"start",parm);
        if(error) current->error = INPerrCat(current->error,
        INPerror(error));
        parm = INPgetValue(ckt,&line,IF_REAL,tab); /* fstop */
        error = INPapName(ckt,which,acJob,"stop",parm);
        if(error) current->error = INPerrCat(current->error,
        INPerror(error));

    }
    if(strcmp(token ,"op")==0){
        which = -1;
        for(i=0;i<ft_sim->numAnalyses;i++) {
            if(strcmp(ft_sim->analyses[i]->name,"DCOP")==0) {
                which=i;
                break;
            }
        }
        if(which == -1) {
            current->error = INPerrCat(current->error,INPmkTemp(
            "DC operating point analysis unsupported\n"));
        }
        else {
            err = (*(ft_sim->newAnalysis))(ft_curckt->ci_ckt,which,"dcop", 
                    &(opJob),ft_curckt->ci_specTask);
            if(err) {
                ft_sperror(err,"createOP"); /* or similar error message */
                return(0);
            }
        }
    }
    if(strcmp(token ,"dc")==0){
        /* .dc SRC1NAME Vstart1 Vstop1 Vinc1 [SRC2NAME Vstart2 */
        /*        Vstop2 Vinc2 */
        which = -1;
        for(i=0;i<ft_sim->numAnalyses;i++) {
            if(strcmp(ft_sim->analyses[i]->name,"DCTransfer")==0) {
                which=i;
                break;
            }
        }
        if(which==-1) {
            current->error = INPerrCat(current->error,INPmkTemp(
            "DC transfer curve analysis unsupported\n"));
        }
        err = (*(ft_sim->newAnalysis))(ft_curckt->ci_ckt,which,"DCtransfer", 
                &(dcJob),ft_curckt->ci_specTask);
        if(err) {
            ft_sperror(err,"createOP"); /* or similar error message */
            return(0);
        }
        INPgetTok(&line,&name,1);
        INPinsert(&name,tab);
        ptemp.uValue=name;
        error = INPapName(ckt,which,dcJob,"name1",&ptemp);
        if(error) current->error = INPerrCat(current->error,INPerror(error));
        parm = INPgetValue(ckt,&line,IF_REAL,tab); /* vstart1 */
        error = INPapName(ckt,which,dcJob,"start1",parm);
        if(error) current->error = INPerrCat(current->error,INPerror(error));
        parm = INPgetValue(ckt,&line,IF_REAL,tab); /* vstop1 */
        error = INPapName(ckt,which,dcJob,"stop1",parm);
        if(error) current->error = INPerrCat(current->error,INPerror(error));
        parm = INPgetValue(ckt,&line,IF_REAL,tab); /* vinc1 */
        error = INPapName(ckt,which,dcJob,"step1",parm);
        if(error) current->error = INPerrCat(current->error,INPerror(error));
        if(*line) {
            if(*line == 'd') goto next;
            INPgetTok(&line,&name,1);
            INPinsert(&name,tab);
            ptemp.uValue=name;
            error = INPapName(ckt,which,dcJob,"name2",&ptemp);
            if(error) current->error= INPerrCat(current->error,INPerror(error));
            parm = INPgetValue(ckt,&line,IF_REAL,tab); /* vstart1 */
            error = INPapName(ckt,which,dcJob,"start2",parm);
            if(error) current->error= INPerrCat(current->error,INPerror(error));
            parm = INPgetValue(ckt,&line,IF_REAL,tab); /* vstop1 */
            error = INPapName(ckt,which,dcJob,"stop2",parm);
            if(error) current->error= INPerrCat(current->error,INPerror(error));
            parm = INPgetValue(ckt,&line,IF_REAL,tab); /* vinc1 */
            error = INPapName(ckt,which,dcJob,"step2",parm);
            if(error) current->error= INPerrCat(current->error,INPerror(error));
        }
    }
    if(strcmp(token ,"tran")==0){
        which = -1;
        for(j=0;j<ft_sim->numAnalyses;j++) {
            if(strcmp(ft_sim->analyses[j]->name,"TRAN")==0) {
                which = j;
                break;
            }
        } 
        if(which != -1) {
            err = (*(ft_sim->newAnalysis))(ft_curckt->ci_ckt,which,"tranan",
            &(tranJob),ft_curckt->ci_specTask);
            if(err) {
                ft_sperror(err,"createTRAN"); 
                return(0);
            }
        } 
        else{ 
            current->error = INPerrCat(current->error,INPmkTemp(
            "transient analysis unsupported\n"));
        }

        parm=INPgetValue(ckt,&line,IF_REAL,tab);/* Tstep */
        error = INPapName(ckt,which,tranJob,"tstep",parm);
        if(error) current->error = INPerrCat(current->error,
        INPerror(error));
        parm = INPgetValue(ckt,&line,IF_REAL,tab); /* Tstop*/
        error = INPapName(ckt,which,tranJob,"tstop",parm);
        if(error) current->error = INPerrCat(current->error,
        INPerror(error));
        if(*line){
            if(*line == 'd') goto next;
            if(*line == 'u') goto uic;
            parm=INPgetValue(ckt,&line,IF_REAL,tab);/* Tstart */
            error = INPapName(ckt,which,tranJob,"tstart",parm);
            if(error) current->error = INPerrCat(current->error,
            INPerror(error));
            if(*line == 'u') goto uic;
            parm=INPgetValue(ckt,&line,IF_REAL,tab);/* Tmax */
            error = INPapName(ckt,which,tranJob,"tmax",parm);
            if(error) current->error = INPerrCat(current->error,
            INPerror(error));
uic:            
            if(*line == 'u') {
                INPgetTok(&line,&name,1);
                if(strcmp(name,"uic")==0) {
                    ptemp.iValue = 1;
                    error = INPapName(ckt,which,tranJob,"tstart",&ptemp);
                    if(error) current->error = INPerrCat(current->error,
                    INPerror(error));

                }
            }

        }
    }

next:          
    while(*line) { /* read the entire line */
        if(flag){
            INPgetTok(&line,&token,1);
        }
        else{
            flag = 1;
        }
        for(i=0;i<ft_sim->analyses[save]->numParms;i++) {
            /* find the parameter */
            if(0==strcmp(token ,
            ft_sim->analyses[save]->analysisParms[i].
                keyword) ){
                /* found it, analysis which, parameter i */
                if(ft_sim->analyses[save]->analysisParms[i].
                    dataType & IF_FLAG) {
                    /* one of the keywords! */
                    ptemp.iValue = 1;
                    error = (*(ft_sim->setAnalysisParm))(ckt,
                        senseJob, ft_sim->analyses[save]->
                        analysisParms[i].id,&ptemp,(IFvalue*)NULL);
                    if(error) current->error = INPerrCat(
                    current->error, INPerror(error));
                } 
                else {
                    parm = INPgetValue(ckt,&line,ft_sim->
                        analyses[save]->analysisParms[i].
                        dataType,tab);
                    error = (*(ft_sim->setAnalysisParm))(ckt,
                        senseJob, ft_sim->analyses[save]->
                        analysisParms[i].id,parm,(IFvalue*)NULL);
                    if(error) current->error = INPerrCat(
                    current->error, INPerror(error));

                }
                break;
            }
        }
        if(i==ft_sim->analyses[save]->numParms) {
            /* didn't find it! */
            current->error = INPerrCat(current->error,INPmkTemp(
            " Error: unknown parameter on .sens - ignored \n"));
        }
    }



    if((err = (*(ft_sim->doAnalyses))(ckt, 1, ft_curckt->ci_curTask))!=OK){
        ft_sperror(err, "doAnalyses");
        return(0);/* temporary */
    }
    return(0);
}
