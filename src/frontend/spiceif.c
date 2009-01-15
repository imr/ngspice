/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
Modified: 2000 AlansFixes
$Id$
**********/

/*
 * Interface routines. These are specific to spice. The only changes to FTE
 * that should be needed to make FTE work with a different simulator is
 * to rewrite this file. What each routine is expected to do can be
 * found in the programmer's manual. This file should be the only one
 * that includes ngspice.header files.
 */

/*CDHW Notes:

I have never really understood the way Berkeley intended the six pointers
to default values (ci_defOpt/Task  ci_specOpt/Task ci_curOpt/Task) to work,
as there only see to be two data blocks to point at, or I've missed something
clever elsewhere.

Anyway, in the original 3f4 the interactive command 'set temp = 10'
set temp for its current task and clobbered the default values as a side
effect. When an interactive is run it created specTask using the spice
application default values, not the circuit defaults affected
by 'set temp = 10'.

The fix involves two changes

  1. Make 'set temp = 10' change the values in the 'default' block, not whatever
     the 'current' pointer happens to be pointing at (which is usually the
     default block except when one interactive is run immediately 
after another).

  2. Hack CKTnewTask() so that it looks to see whether it is creating 
a 'special'
     task, in which case it copies the values from 
ft_curckt->ci_defTask providing
     everything looks sane, otherwise it uses the hard-coded 
'application defaults'.

These are fairly minor changes, and as they don't change the data structures
they should be fairly 'safe'. However, ...


CDHW*/

#include "ngspice.h"
#include "cktdefs.h"
#include "cpdefs.h"
#include "tskdefs.h" /* Is really needed ? */
#include "ftedefs.h"
#include "fteinp.h"
#include "inpdefs.h"
#include "iferrmsg.h"
#include "ifsim.h"

#include "circuits.h"
#include "spiceif.h"
#include "variable.h"

#ifdef XSPICE
/* gtri - add - wbk - 11/9/90 - include MIF function prototypes */
#include "mifproto.h"
/* gtri - end - wbk - 11/9/90 */

/* gtri - evt - wbk - 5/20/91 - Add stuff for user-defined nodes */
#include "evtproto.h"
#include "evtudn.h"
/* gtri - end - wbk - 5/20/91 - Add stuff for user-defined nodes */
#endif

/* static declarations */
static struct variable * parmtovar(IFvalue *pv, IFparm *opt);
static IFparm * parmlookup(IFdevice *dev, GENinstance **inptr, char *param, 
			   int do_model, int inout);
static IFvalue * doask(void *ckt, int typecode, GENinstance *dev, GENmodel *mod, 
		       IFparm *opt, int ind);
static int doset(void *ckt, int typecode, GENinstance *dev, GENmodel *mod, 
		 IFparm *opt, struct dvec *val);
static int finddev(void *ck, char *name, void **devptr, void **modptr);

/*espice fix integration */
static int finddev_special(char *ck, char *name, void **devptr, void **modptr, int *device_or_model);

/* Input a single deck, and return a pointer to the circuit. */

char *
if_inpdeck(struct line *deck, INPtables **tab)
{
    void *ckt;
    int err, i, j;
    struct line *ll;
    IFuid taskUid;
    IFuid optUid;
    int which = -1;

    for (i = 0, ll = deck; ll; ll = ll->li_next)
        i++;
    *tab = INPtabInit(i);
    ft_curckt->ci_symtab = *tab;

    err = (*(ft_sim->newCircuit))(&ckt);
    if (err != OK) {
        ft_sperror(err, "CKTinit");
        return (NULL);
    }

/*CDHW Create a task DDD with a new UID. ci_defTask will point to it CDHW*/

    err = IFnewUid(ckt,&taskUid,(IFuid)NULL,"default",UID_TASK,(void**)NULL);
    if(err) {
        ft_sperror(err,"newUid");
        return(NULL);
    }
#if (0)
       err = 
     (*(ft_sim->newTask))(ckt,(void**)&(ft_curckt->ci_defTask),taskUid);
#else /*CDHW*/
       err = 
       (*(ft_sim->newTask))(ckt,(void**)&(ft_curckt->ci_defTask),taskUid,
       (void**)NULL);
#endif
    if(err) {
        ft_sperror(err,"newTask");
        return(NULL);
    }

/*CDHW which options available for this simulator? CDHW*/

    for(j=0;j<ft_sim->numAnalyses;j++) {
        if(strcmp(ft_sim->analyses[j]->name,"options")==0) {
            which = j;
            break;
        }
    } 

    if(which != -1) {
        err = IFnewUid(ckt,&optUid,(IFuid)NULL,"options",UID_ANALYSIS,
                (void**)NULL);
        if(err) {
            ft_sperror(err,"newUid");
            return(NULL);
        }

        err = (*(ft_sim->newAnalysis))(ft_curckt->ci_ckt,which,optUid,
                (void**)&(ft_curckt->ci_defOpt),
                (void*)ft_curckt->ci_defTask);
		
/*CDHW ci_defTask and ci_defOpt point to parameters DDD CDHW*/		
		
        if(err) {
            ft_sperror(err,"createOptions");
            return(NULL);
        }

        ft_curckt->ci_curOpt  = ft_curckt->ci_defOpt;
/*CDHW ci_curOpt and ci_defOpt point to DDD CDHW*/
    }

    ft_curckt->ci_curTask = ft_curckt->ci_defTask;
    
    INPpas1((void *) ckt, (card *) deck->li_next,(INPtables *)*tab);
    INPpas2((void *) ckt, (card *) deck->li_next,
            (INPtables *) *tab,ft_curckt->ci_defTask);
    INPkillMods();

    /* INPpas2 has been modified to ignore .NODESET and .IC
     * cards. These are left till INPpas3 so that we can check for
     * nodeset/ic of non-existant nodes.  */

    INPpas3((void *) ckt, (card *) deck->li_next,
            (INPtables *) *tab,ft_curckt->ci_defTask, ft_sim->nodeParms,
	    ft_sim->numNodeParms);

#ifdef XSPICE
/* gtri - begin - wbk - 6/6/91 - Finish initialization of event driven structures */
    err = EVTinit((void *) ckt);
    if(err) {
        ft_sperror(err,"EVTinit");
        return(NULL);
    }
/* gtri - end - wbk - 6/6/91 - Finish initialization of event driven structures */
#endif

    return (ckt);
}


/* Do a run of the circuit, of the given type. Type "resume" is
 * special -- it means to resume whatever simulation that was in
 * progress. The return value of this routine is 0 if the exit was ok,
 * and 1 if there was a reason to interrupt the circuit (interrupt
 * typed at the keyboard, error in the simulation, etc). args should
 * be the entire command line, e.g. "tran 1 10 20 uic" */
int
if_run(char *t, char *what, wordlist *args, char *tab)
{
    void *ckt = (void *) t;
    int err;
    struct line deck;
    char buf[BSIZE_SP];
    int j;
    int which = -1;
    IFuid specUid,optUid;
    char *s;
    
    
    /* First parse the line... */
    /*CDHW Look for an interactive task CDHW*/
    if (eq(what, "tran") 
    || eq(what, "ac") 
    || eq(what, "dc")
    || eq(what, "op") 
    || eq(what, "pz") 
    || eq(what,"disto")
    || eq(what, "adjsen") 
    || eq(what, "sens") 
    || eq(what,"tf")
    || eq(what, "noise")) 
    {
    	s = wl_flatten(args); /* va: tfree char's tmalloc'ed in wl_flatten */
        (void) sprintf(buf, ".%s", s);
        tfree(s);      
        deck.li_next = deck.li_actual = NULL;
        deck.li_error = NULL;
        deck.li_linenum = 0;
        deck.li_line = buf;
        
/*CDHW Delete any previous special task CDHW*/	

        if(ft_curckt->ci_specTask) {
	   if (ft_curckt->ci_specTask == ft_curckt->ci_defTask) { /*CDHW*/
              printf("Oh dear...something bad has happened to the options.\n");
            }
            err=(*(ft_sim->deleteTask))(ft_curckt->ci_ckt,
                    ft_curckt->ci_specTask);
            if(err) {
                ft_sperror(err,"deleteTask");
                return(2);
            }
	    ft_curckt->ci_specTask = ft_curckt->ci_specOpt = NULL; /*CDHW*/
        }
       /*CDHW Create an interactive task AAA with a new UID.  
ci_specTask will point to it CDHW*/
	
        err = IFnewUid(ft_curckt->ci_ckt,&specUid,(IFuid)NULL,"special",
                UID_TASK,(void**)NULL);
        if(err) {
            ft_sperror(err,"newUid");
            return(2);
        }
#if (0)
        err = (*(ft_sim->newTask))(ft_curckt->ci_ckt, 
                (void**)&(ft_curckt->ci_specTask),specUid);
#else /*CDHW*/
        err = (*(ft_sim->newTask))(ft_curckt->ci_ckt, 
                 (void**)&(ft_curckt->ci_specTask),
                 specUid,(void**)&(ft_curckt->ci_defTask));
#endif        
        if(err) {
            ft_sperror(err,"newTask");
            return(2);
        }

/*CDHW which options available for this simulator? CDHW*/   
        
        for(j=0;j<ft_sim->numAnalyses;j++) {
            if(strcmp(ft_sim->analyses[j]->name,"options")==0) {
                which = j;
                break;
            }
        } 
        if(which != -1) { /*CDHW options are available CDHW*/
            err = IFnewUid(ft_curckt->ci_ckt,&optUid,(IFuid)NULL,"options",
                    UID_ANALYSIS,(void**)NULL);
            if(err) {
                ft_sperror(err,"newUid");
                return(2);
            }
            err = (*(ft_sim->newAnalysis))(ft_curckt->ci_ckt,which,optUid,
                    (void**)&(ft_curckt->ci_specOpt),
                    (void*)ft_curckt->ci_specTask);
		    
/*CDHW 'options' ci_specOpt points to AAA in this case CDHW*/		    
		    
            if(err) {
                ft_sperror(err,"createOptions");
                return(2);
            }
	    
            ft_curckt->ci_curOpt  = ft_curckt->ci_specOpt;

/*CDHW ci_specTask ci_specOpt and ci_curOpt all point to AAA CDHW*/		
        
	}
	
        ft_curckt->ci_curTask = ft_curckt->ci_specTask;

/*CDHW ci_curTask and ci_specTask point to the interactive task AAA CDHW*/  
        
      INPpas2(ckt, (card *) &deck, (INPtables *)tab, ft_curckt->ci_specTask);
        
        if (deck.li_error) {
            fprintf(cp_err, "Warning: %s\n", deck.li_error);
	    return 2;
        }
    }

     /*CDHW
     ** if the task is to 'run' the deck, change ci_curTask and     
     ** ci_curOpt to point to DDD
     ** created by if_inpdeck(), otherwise they point to AAA.
     CDHW*/
    
    if( eq(what,"run") ) {
        ft_curckt->ci_curTask = ft_curckt->ci_defTask;
        ft_curckt->ci_curOpt = ft_curckt->ci_defOpt;
    }

/* -- Find out what we are supposed to do.              */

    if (  (eq(what, "tran"))
        ||(eq(what, "ac"))
        ||(eq(what, "dc"))
        ||(eq(what, "op"))
        ||(eq(what, "pz"))
        ||(eq(what, "disto"))
        ||(eq(what, "noise"))
        ||(eq(what, "adjsen")) 
        ||(eq(what, "sens")) 
        ||(eq(what,"tf"))
        ||(eq(what, "run"))  )  {
	
/*CDHW Run the analysis pointed to by ci_curTask CDHW*/

        ft_curckt->ci_curOpt = ft_curckt->ci_defOpt;	
        if ((err = (*(ft_sim->doAnalyses))(ckt, 1, ft_curckt->ci_curTask))!=OK){
            ft_sperror(err, "doAnalyses");
            /* wrd_end(); */
	    if (err == E_PAUSE)
		return (1);
	    else
		return (2);
        }
    } else if (eq(what, "resume")) {
        if ((err = (*(ft_sim->doAnalyses))(ckt, 0, ft_curckt->ci_curTask))!=OK){
            ft_sperror(err, "doAnalyses");
            /* wrd_end(); */
	    if (err == E_PAUSE)
		return (1);
	    else
		return (2);
        }
    } else {
        fprintf(cp_err, "if_run: Internal Error: bad run type %s\n",
                what);
	return (2);
    }
    return (0);
}

/* Set an option in the circuit. Arguments are option name, type, and
 * value (the last a char *), suitable for casting to whatever needed...
 */

static char *unsupported[] = {
    "itl3",
    "itl5",
    "lvltim",
    "maxord",
    "method",
    NULL
} ;

static char *obsolete[] = {
    "limpts",
    "limtim",
    "lvlcod",
    NULL
} ;

int
if_option(void *ckt, char *name, int type, char *value)
{
    IFvalue pval;
    int err, i;
    void *cc = (void *) ckt;
    char **vv;
    int which = -1;

    if (eq(name, "acct")) {
        ft_acctprint = TRUE;
	return 0;
    } else if (eq(name, "list")) {
        ft_listprint = TRUE;
	return 0;
    } else if (eq(name, "node")) {
        ft_nodesprint = TRUE;
	return 0;
    } else if (eq(name, "opts")) {
        ft_optsprint = TRUE;
	return 0;
    } else if (eq(name, "nopage")) {
	ft_nopage = TRUE;
	return 0;
    } else if (eq(name, "nomod")) {
	ft_nomod = TRUE;
	return 0;
    }

    for(i=0;i<ft_sim->numAnalyses;i++) {
        if(strcmp(ft_sim->analyses[i]->name,"options")==0) {
            which = i;
            break;
        }
    }
    if(which==-1) {
        fprintf(cp_err,"Warning:  .options line unsupported\n");
        return 0;
    }

    for (i = 0; i < ft_sim->analyses[which]->numParms; i++)
        if (eq(ft_sim->analyses[which]->analysisParms[i].keyword, name) &&
                (ft_sim->analyses[which]->analysisParms[i].dataType & IF_SET))
            break;
    if (i == ft_sim->analyses[which]->numParms) {
        /* See if this is unsupported or obsolete. */
        for (vv = unsupported; *vv; vv++)
            if (eq(name, *vv)) {
                fprintf(cp_err, 
            "Warning: option %s is currently unsupported.\n", name);
                return 1;
            }
        for (vv = obsolete; *vv; vv++)
            if (eq(name, *vv)) {
                fprintf(cp_err, 
                "Warning: option %s is obsolete.\n", name);
                return 1;
            }
        return 0;
    }

    switch (ft_sim->analyses[which]->analysisParms[i].dataType & IF_VARTYPES) {
        case IF_REAL:
            if (type == VT_REAL)
                pval.rValue = *((double *) value);
            else if (type == VT_NUM)
                pval.rValue = *((int *) value);
            else
                goto badtype;
            break;
        case IF_INTEGER:
            if (type == VT_NUM)
                pval.iValue = *((int *) value);
            else if (type == VT_REAL)
                pval.iValue = *((double *) value);
            else
                goto badtype;
            break;
        case IF_STRING:
            if (type == VT_STRING)
                pval.sValue = copy(value);
            else
                goto badtype;
            break;
        case IF_FLAG:
            /* Do nothing. */
            pval.iValue = *((int *) value);
            break;
        default:
            fprintf(cp_err, 
            "if_option: Internal Error: bad option type %d.\n",
                    ft_sim->analyses[which]->analysisParms[i].dataType);
    }

    if (!ckt) {
	/* XXX No circuit loaded */
	fprintf(cp_err, "Simulation parameter \"%s\" can't be set until\n",
		name);
	fprintf(cp_err, "a circuit has been loaded.\n");
	return 1;
    }

#if (0)
     if ((err = (*(ft_sim->setAnalysisParm))(cc, (void *)ft_curckt->ci_curOpt,
             ft_sim->analyses[which]->analysisParms[i].id, &pval,
             (IFvalue *)NULL)) != OK)
         ft_sperror(err, "setAnalysisParm(options) ci_curOpt");
#else /*CDHW*/
     if ((err = (*(ft_sim->setAnalysisParm))(cc, (void *)ft_curckt->ci_defOpt,
             ft_sim->analyses[which]->analysisParms[i].id, &pval,
             (IFvalue *)NULL)) != OK)
         ft_sperror(err, "setAnalysisParm(options) ci_curOpt");
     return 1;
#endif

badtype:
    fprintf(cp_err, "Error: bad type given for option %s --\n", name);
    fprintf(cp_err, "\ttype given was ");
    switch (type) {
        case VT_BOOL:   fputs("boolean", cp_err); break;
        case VT_NUM:    fputs("integer", cp_err); break;
        case VT_REAL:   fputs("real", cp_err); break;
        case VT_STRING: fputs("string", cp_err); break;
        case VT_LIST:   fputs("list", cp_err); break;
        default:    fputs("something strange", cp_err); break;
    }
    fprintf(cp_err, ", type expected was ");
    switch(ft_sim->analyses[which]->analysisParms[i].dataType & IF_VARTYPES) {
        case IF_REAL:   fputs("real.\n", cp_err); break;
        case IF_INTEGER:fputs("integer.\n", cp_err); break;
        case IF_STRING: fputs("string.\n", cp_err); break;
        case IF_FLAG:   fputs("flag.\n", cp_err); break;
        default:    fputs("something strange.\n", cp_err); break;
    }
    if (type == VT_BOOL)
fputs("\t(Note that you must use an = to separate option name and value.)\n", 
                    cp_err); 
    return 0;
}


void
if_dump(void *ckt, FILE *file)
{
    /*void *cc = (void *) ckt;*/

    fprintf(file,"diagnostic output dump unavailable.");
    return;
}

void
if_cktfree(void *ckt, char *tab)
{
    void *cc = (void *) ckt;

    (*(ft_sim->deleteCircuit))(cc);
    INPtabEnd((INPtables *) tab);
    return;
}

/* Return a string describing an error code. */


/* BLOW THIS AWAY.... */

char *
if_errstring(int code)
{
    return (INPerror(code));
}

/* Get pointers to a device, its model, and its type number given the name. If
 * there is no such device, try to find a model with that name
 * device_or_model says if we are referencing a device or a model.
 *  finddev_special(ck, name, devptr, modptr,device_or_model):
 *  Introduced to look for correct reference in expression like  print @BC107 [is] 
 * and find out  whether a model or a device parameter is referenced and properly 
 * call the spif_getparam_special (ckt, name, param, ind, do_model) function in
 * vector.c - A. Roldan (espice).
 */
static int 
finddev_special(
    char *ck,
    char *name,
    void **devptr,
    void **modptr,
    int *device_or_model)
{
    int err;
    int type = -1;

    err = (*(ft_sim->findInstance))((void *)ck,&type,devptr,name,NULL,NULL);
    if(err == OK)
    {
     *device_or_model=0;
     return(type);
    }
    type = -1;
    *devptr = (void *)NULL;
    err = (*(ft_sim->findModel))((void *)ck,&type,modptr,name);
    if(err == OK)
    {
     *device_or_model=1;
     return(type);
    }
    *modptr = (void *)NULL;
    *device_or_model=2;
    return(-1);

}

/* Get a parameter value from the circuit. If name is left unspecified,
 * we want a circuit parameter. Now works both for devices and models.
 * A.Roldan (espice)
 */
struct variable *
spif_getparam_special(void *ckt,char **name,char *param,int ind,int do_model)    
{
    struct variable *vv = NULL, *tv;
    IFvalue *pv;
    IFparm *opt;
    int typecode, i, modelo_dispositivo;
    GENinstance *dev=(GENinstance *)NULL;
    GENmodel *mod=(GENmodel *)NULL;
    IFdevice *device;

    /* fprintf(cp_err, "Calling if_getparam(%s, %s)\n", *name, param); */

    if (!param || (param && eq(param, "all")))
     {
        INPretrieve(name,(INPtables *)ft_curckt->ci_symtab);
        typecode = finddev_special(ckt, *name, (void**)&dev, (void**)&mod,&modelo_dispositivo);
        if (typecode == -1)
        {
            fprintf(cp_err,"Error: no such device or model name %s\n",*name);
            return (NULL);
        }
        device = ft_sim->devices[typecode];
        if(!modelo_dispositivo)
        {
          /* It is a device */
          for (i = 0; i < *(device->numInstanceParms); i++)
          {
            opt = &device->instanceParms[i];
            if(opt->dataType & IF_REDUNDANT || !opt->description) continue;
            if(!(opt->dataType & IF_ASK)) continue;
            pv = doask(ckt, typecode, dev, mod, opt, ind);
            if (pv)
            {
             tv = parmtovar(pv, opt);

	     /* With the following we pack the name and the acronym of the parameter */
	     {
		char auxiliar[70],*aux_pointer;
		sprintf(auxiliar,"%s [%s]",tv->va_name, device->instanceParms[i].keyword);
		aux_pointer=tv->va_name;
		free(aux_pointer);
		tv->va_name = copy(auxiliar);
	     }
             if (vv) tv->va_next = vv;
             vv = tv;
          }
          else
           fprintf(cp_err,"Internal Error: no parameter '%s' on device '%s'\n",
			   device->instanceParms[i].keyword,device->name);
         }
         return (vv);
        }
        else  /* Is it a model or a device ? */
        {
         /* It is a model */
         for (i = 0; i < *(device->numModelParms); i++)
         {
            opt = &device->modelParms[i];
            if(opt->dataType & IF_REDUNDANT || !opt->description) continue;
        
            /* We check that the parameter is interesting and therefore is 
             * implemented in the corresponding function ModelAsk. Originally
             * the argument of "if" was: || (opt->dataType & IF_STRING)) continue;
             * so, a model parameter defined like  OP("type",   MOS_SGT_MOD_TYPE,  
             * IF_STRING, N-channel or P-channel MOS") would not be printed.
             */

        /* if(!(opt->dataType & IF_ASK ) || (opt->dataType & IF_UNINTERESTING ) || (opt->dataType & IF_STRING)) continue; */
	    if(!(opt->dataType & IF_ASK ) || (opt->dataType & IF_UNINTERESTING )) continue;
            pv = doask(ckt, typecode, dev, mod, opt, ind);
            if (pv)
            {
                tv = parmtovar(pv, opt);
		/* Inside parmtovar:
		 * 1. tv->va_name = copy(opt->description);
		 * 2. Copy the type of variable of IFparm into a variable (thus parm-to-var)
		 * vv->va_type = opt->dataType
		 * The long description of the parameter:
		 * IFparm MOS_SGTmPTable[] = { // model parameters //
		 * OP("type",   MOS_SGT_MOD_TYPE,  IF_STRING, "N-channel or P-channel MOS") 
         * goes into tv->va_name to put braces around the parameter of the model
		 * tv->va_name += device->modelParms[i].keyword;
         */
		{
		 char auxiliar[70],*aux_pointer;
		 sprintf(auxiliar,"%s [%s]",tv->va_name,device->modelParms[i].keyword);
		 aux_pointer=tv->va_name;
		 free(aux_pointer);
		 tv->va_name = copy(auxiliar);
		/* strcpy(aux_pointer,auxiliar); */
		}
		/* tv->va_string=device->modelParms[i].keyword;	Put the name of the variable */
                if (vv)
                {
                  tv->va_next = vv;
                }
                vv = tv;
            }
            else
                fprintf(cp_err,"Internal Error: no parameter '%s' on device '%s'\n",device->modelParms[i].keyword,device->name);
        }
        return (vv);
        } 
    }
    else if (param)
    {
        INPretrieve(name,(INPtables *)ft_curckt->ci_symtab);
        typecode = finddev_special(ckt, *name, (void**)&dev, (void**)&mod,&modelo_dispositivo);
        if (typecode == -1)
        {
            fprintf(cp_err,"Error: no such device or model name %s\n",*name);
            return (NULL);
        }
        device = ft_sim->devices[typecode];
        opt = parmlookup(device, &dev, param, modelo_dispositivo, 0);
        if (!opt)
        {
            fprintf(cp_err, "Error: no such parameter %s.\n",param);
            return (NULL);
        }
        pv = doask(ckt, typecode, dev, mod, opt, ind);
        if (pv)
            vv = parmtovar(pv, opt);
        return (vv);
    } else
        return (if_getstat(ckt, *name));
}




/* Get a parameter value from the circuit. If name is left unspecified,
 * we want a circuit parameter.
 */

struct variable *
spif_getparam(void *ckt, char **name, char *param, int ind, int do_model)
{
    struct variable *vv = NULL, *tv;
    IFvalue *pv;
    IFparm *opt;
    int typecode, i;
    GENinstance *dev=(GENinstance *)NULL;
    GENmodel *mod=(GENmodel *)NULL;
    IFdevice *device;

    /* fprintf(cp_err, "Calling if_getparam(%s, %s)\n", *name, param); */

    if (param && eq(param, "all")) {
    
    	/* MW. My "special routine here" */
        INPretrieve(name,(INPtables *)ft_curckt->ci_symtab);
        
        typecode = finddev(ckt, *name,(void**) &dev,(void **) &mod);
        if (typecode == -1) {
            fprintf(cp_err,
                "Error: no such device or model name %s\n",
                    *name);
            return (NULL);
        }
        device = ft_sim->devices[typecode];
        for (i = 0; i < *(device->numInstanceParms); i++) {
            opt = &device->instanceParms[i];
            if(opt->dataType & IF_REDUNDANT || !opt->description)
		    continue;
            if(!(opt->dataType & IF_ASK)) continue;
            pv = doask(ckt, typecode, dev, mod, opt, ind);
            if (pv) {
                tv = parmtovar(pv, opt);
                if (vv)
                    tv->va_next = vv;
                vv = tv;
            } else
                fprintf(cp_err,
            "Internal Error: no parameter '%s' on device '%s'\n",
                    device->instanceParms[i].keyword,
                    device->name);
        }
        return (vv);
    } else if (param) {
    
    	/* MW.  */
        INPretrieve(name,(INPtables *)ft_curckt->ci_symtab);
        typecode = finddev(ckt, *name, (void**)&dev, (void **)&mod);
        if (typecode == -1) {
            fprintf(cp_err,
                "Error: no such device or model name %s\n",
                    *name);
            return (NULL);
        }
        device = ft_sim->devices[typecode];
        opt = parmlookup(device, &dev, param, do_model, 0);
        if (!opt) {
            fprintf(cp_err, "Error: no such parameter %s.\n",
                    param);
            return (NULL);
        }
        pv = doask(ckt, typecode, dev, mod, opt, ind);
        if (pv)
            vv = parmtovar(pv, opt);
        return (vv);
    } else
        return (if_getstat(ckt, *name));
}

/* 9/26/03 PJB : function to allow setting model of device */
void
if_setparam_model( void *ckt, char **name, char *val )
{
  GENinstance *dev     = (GENinstance *)NULL;
  GENinstance *prevDev = (GENinstance *)NULL;
  GENmodel    *curMod  = (GENmodel *)   NULL;
  GENmodel    *newMod  = (GENmodel *)   NULL;
  INPmodel    *inpmod  = (INPmodel *)   NULL;
  GENinstance *iter;
  GENmodel    *mods, *prevMod;
  int         typecode;

  /* retrieve device name from symbol table */
  INPretrieve(name,(INPtables *)ft_curckt->ci_symtab);
  /* find the specified device */
  typecode = finddev(ckt, *name, (void**)&dev, (void **)&curMod);
  if (typecode == -1) {
    fprintf(cp_err, "Error: no such device or model name %s\n", *name);
    return;
  }
  curMod = dev->GENmodPtr;
  /* 
     retrieve the model from the global model table; also add the model to 'ckt'
     and indicate model is being used
  */
  INPgetMod( ckt, val, &inpmod, (INPtables *)ft_curckt->ci_symtab );
  if ( inpmod == NULL ) {
    fprintf(cp_err, "Error: no such model %s.\n", val);
    return;
  }
  newMod = (GENmodel*)(inpmod->INPmodfast);

  /* see if new model name same as current model name */
  if ( newMod->GENmodName == curMod->GENmodName ) {
    fprintf(cp_err, "Warning: new model same as current model; nothing changed.\n");
    return;
  }
  if ( newMod->GENmodType != curMod->GENmodType ) {
    fprintf(cp_err, "Error: new model %s must be same type as current model.\n", val); 
    return;
  }

  /* fix current model linked list */
  prevDev = NULL;
  for( iter = curMod->GENinstances; iter != NULL; iter = iter->GENnextInstance ) {
    if ( iter->GENname == dev->GENname ) {

      /* see if at beginning of linked list */
      if ( prevDev == NULL ) curMod->GENinstances     = iter->GENnextInstance;
      else                   prevDev->GENnextInstance = iter->GENnextInstance;

      /* update model for device */
      dev->GENmodPtr       = newMod;
      dev->GENnextInstance = newMod->GENinstances;
      newMod->GENinstances = dev;
      break;
    }
    prevDev = iter;
  }
  /* see if any devices remaining that reference current model */
  if ( curMod->GENinstances == NULL ) {
    prevMod = NULL;
    for( mods = ((CKTcircuit *)ckt)->CKThead[typecode]; mods != NULL; mods = mods->GENnextModel ) {
      if ( mods->GENmodName == curMod->GENmodName ) {

	/* see if at beginning of linked list */
	if ( prevMod == NULL ) ((CKTcircuit *)ckt)->CKThead[typecode] = mods->GENnextModel;
	else 	               prevMod->GENnextModel                  = mods->GENnextModel;

	INPgetMod( ckt, (char *)mods->GENmodName, &inpmod, (INPtables *)ft_curckt->ci_symtab );
	inpmod->INPmodUsed = 0;
	FREE(mods);

	break;
      }
      prevMod = mods;
    }
  }
}

void
if_setparam(void *ckt, char **name, char *param, struct dvec *val, int do_model)
{
    IFparm *opt;
    IFdevice *device;
    GENmodel *mod=(GENmodel *)NULL;
    GENinstance *dev=(GENinstance *)NULL;
    int typecode;

	/* PN  */
    INPretrieve(name,(INPtables *)ft_curckt->ci_symtab);
    typecode = finddev(ckt, *name, (void**)&dev, (void **)&mod);
    if (typecode == -1) {
	fprintf(cp_err, "Error: no such device or model name %s\n", *name);
	return;
    }
    device = ft_sim->devices[typecode];
    opt = parmlookup(device, &dev, param, do_model, 1);
    if (!opt) {
	if (param)
		fprintf(cp_err, "Error: no such parameter %s.\n", param);
	else
		fprintf(cp_err, "Error: no default parameter.\n");
	return;
    }
    if (do_model && !mod) {
	mod = dev->GENmodPtr;
	dev = (GENinstance *)NULL;
    }
    doset(ckt, typecode, dev, mod, opt, val);
}

static struct variable *
parmtovar(IFvalue *pv, IFparm *opt)
{
    struct variable *vv = alloc(struct variable);
    struct variable *nv;
    int i = 0;

    switch (opt->dataType & IF_VARTYPES) {
        case IF_INTEGER:
            vv->va_type = VT_NUM;
            vv->va_num = pv->iValue;
            break;
        case IF_REAL:
        case IF_COMPLEX:
            vv->va_type = VT_REAL;
            vv->va_real = pv->rValue;
            break;
        case IF_STRING:
            vv->va_type = VT_STRING;
            vv->va_string = pv->sValue;
            break;
        case IF_FLAG:
            vv->va_type = VT_BOOL;
            vv->va_bool = pv->iValue ? TRUE : FALSE;
            break;
        case IF_REALVEC:
            vv->va_type = VT_LIST;
            for (i = 0; i < pv->v.numValue; i++) 
       {
                nv = alloc(struct variable);
                nv->va_next = vv->va_vlist;
                vv->va_vlist = nv;
                nv->va_type = VT_REAL;
                /* Change this so that the values are printed in order and
                 * not in inverted order as happens in the conversion process.
                 * Originally was  nv->va_real = pv->v.vec.rVec[i];
                 */
                nv->va_real = pv->v.vec.rVec[pv->v.numValue-i-1];
      }
            /* It is a linked list where the first node is a variable
             * pointing to the different values of the variables.
             *
             * To access the values of the real variable vector must be
             * vv->va_V.vV_real=valor node ppal that is of no use.
             *
             * In the case of Vin_sin 1 0 sin (0 2 2000)
             * and of print @vin_sin[sin]
             *
             * vv->va_V.vV_list->va_V.vV_real=2000
             * vv->va_V.vV_list->va_next->va_V.vV_real=2
             * vv->va_V.vV_list->va_next->va_next->va_V.vV_real=0
             * So the list is starting from behind, but no problem
             * This works fine
             */        
       
       break;
        default:
            fprintf(cp_err,  
            "parmtovar: Internal Error: bad PARM type %d.\n",
                    opt->dataType);
            return (NULL);
    }

    /* It's not clear whether we want the keyword or the desc here... */
    vv->va_name = copy(opt->description);
    vv->va_next = NULL;
    return (vv);
}

/* Extract the parameter (IFparm structure) from the device or device's model.
 * If do_mode is TRUE then look in the device's parameters
 * If do_mode is FALSE then look in the device model's parameters
 * If inout equals 1 then look only for parameters with the IF_SET type flag
 * if inout equals 0 then look only for parameters with the IF_ASK type flag
 */

static IFparm *
parmlookup(IFdevice *dev, GENinstance **inptr, char *param, int do_model, int inout)
{
    int i;

    /* First try the device questions... */
    if (!do_model && dev->numInstanceParms) {
        for (i = 0; i < *(dev->numInstanceParms); i++) {
            if (!param && (dev->instanceParms[i].dataType & IF_PRINCIPAL))
                return (&dev->instanceParms[i]);
	    else if (!param)
		continue;
            else if ((((dev->instanceParms[i].dataType & IF_SET) && inout == 1)
	    	|| ((dev->instanceParms[i].dataType & IF_ASK) && inout == 0))
	    	&& cieq(dev->instanceParms[i].keyword, param))
	    {
		if (dev->instanceParms[i].dataType & IF_REDUNDANT)
		    i -= 1;
                return (&dev->instanceParms[i]);
	    }
        }
	return NULL;
    }

    if (dev->numModelParms) {
	for (i = 0; i < *(dev->numModelParms); i++)
	    if ((((dev->modelParms[i].dataType & IF_SET) && inout == 1)
		    || ((dev->modelParms[i].dataType & IF_ASK) && inout == 0))
		    && eq(dev->modelParms[i].keyword, param))
	    {
		if (dev->modelParms[i].dataType & IF_REDUNDANT)
		    i -= 1;
		return (&dev->modelParms[i]);
	    }
    }

    return (NULL);
}

/* Perform the CKTask call. We have both 'fast' and 'modfast', so the other
 * parameters aren't necessary.
 */


static IFvalue *
doask(void *ckt, int typecode, GENinstance *dev, GENmodel *mod, IFparm *opt, int ind)
{
    static IFvalue pv;
    int err;

    pv.iValue = ind;    /* Sometimes this will be junk and ignored... */

    /* fprintf(cp_err, "Calling doask(%d, %x, %x, %x)\n", 
            typecode, dev, mod, opt); */
    if (dev)
        err = (*(ft_sim->askInstanceQuest))((void *)ckt, (void *)dev, 
                opt->id, &pv, (IFvalue *)NULL);
    else
        err = (*(ft_sim->askModelQuest))((void*)ckt, (void *) mod, 
                opt->id, &pv, (IFvalue *)NULL);
    if (err != OK) {
        ft_sperror(err, "if_getparam");
        return (NULL);
    }

    return (&pv);
}

/* Perform the CKTset call. We have both 'fast' and 'modfast', so the other
 * parameters aren't necessary.
 */


static int
doset(void *ckt, int typecode, GENinstance *dev, GENmodel *mod, IFparm *opt, struct dvec *val)
{
    IFvalue nval;
    int err;
    int n;
    int *iptr;
    double *dptr;
    int i;

    /* Count items */
    if (opt->dataType & IF_VECTOR) {
	n = nval.v.numValue = val->v_length;

	dptr = val->v_realdata;
	/* XXXX compdata!!! */

	switch (opt->dataType & (IF_VARTYPES & ~IF_VECTOR)) {
	    case IF_FLAG:
	    case IF_INTEGER:
		iptr = nval.v.vec.iVec = NEWN(int, n);

		for (i = 0; i < n; i++)
		    *iptr++ = *dptr++;
		break;

	    case IF_REAL:
		nval.v.vec.rVec = val->v_realdata;
		break;

	    default:
		fprintf(cp_err,
		    "Can't assign value to \"%s\" (unsupported vector type)\n",
		    opt->keyword);
		return E_UNSUPP;
	}
    } else {
	switch (opt->dataType & IF_VARTYPES) {
	    case IF_FLAG:
	    case IF_INTEGER:
		nval.iValue = *val->v_realdata;
		break;

	    case IF_REAL:
/*kensmith don't blow up with NULL dereference*/
		if (!val->v_realdata) {
			fprintf(cp_err,"Unable to determine the value\n");
			return E_UNSUPP;
			}
	    
		nval.rValue = *val->v_realdata;
		break;

	    default:
		fprintf(cp_err,
		    "Can't assign value to \"%s\" (unsupported type)\n",
		    opt->keyword);
		return E_UNSUPP;
	}
    }

    /* fprintf(cp_err, "Calling doask(%d, %x, %x, %x)\n", 
            typecode, dev, mod, opt); */

    if (dev)
        err = (*(ft_sim->setInstanceParm))((void *)ckt, (void *)dev, 
                opt->id, &nval, (IFvalue *)NULL);
    else
        err = (*(ft_sim->setModelParm))((void*)ckt, (void *) mod, 
                opt->id, &nval, (IFvalue *)NULL);

    return err;
}



/* Get pointers to a device, its model, and its type number given the name. If
 * there is no such device, try to find a model with that name.
 */

static int
finddev(void *ck, char *name, void **devptr, void **modptr)
{
    int err;
    int type = -1;

    err = (*(ft_sim->findInstance))((void *)ck,&type,devptr,name,NULL,NULL);
    if(err == OK) return(type);
    type = -1;
    *devptr = (void *)NULL;
    err = (*(ft_sim->findModel))((void *)ck,&type,modptr,name);
    if(err == OK) return(type);
    *modptr = (void *)NULL;
    return(-1);
}

/* get an analysis parameter by name instead of id */

int 
if_analQbyName(void *ckt, int which, void *anal, char *name, IFvalue *parm)
{
    int i;
    for(i=0;i<ft_sim->analyses[which]->numParms;i++) {
        if(strcmp(ft_sim->analyses[which]->analysisParms[i].keyword,name)==0) {
            return( (*(ft_sim->askAnalysisQuest))(ckt,anal,
                    ft_sim->analyses[which]->analysisParms[i].id,parm,
                    (IFvalue *)NULL) );
        }
    }
    return(E_BADPARM);
}

/* Get the parameters tstart, tstop, and tstep from the CKT struct. */

/* BLOW THIS AWAY TOO */

bool
if_tranparams(struct circ *ci, double *start, double *stop, double *step)
{
    IFvalue tmp;
    int err;
    int which = -1;
    int i;
    void *anal;
    IFuid tranUid;

    if(!ci->ci_curTask) return(FALSE);
    for(i=0;i<ft_sim->numAnalyses;i++) {
        if(strcmp(ft_sim->analyses[i]->name,"TRAN")==0){
            which = i;
            break;
        }
    }
    if(which == -1) return(FALSE);

    err = IFnewUid(ci->ci_ckt,&tranUid,(IFuid)NULL,"Transient Analysis",
	UID_ANALYSIS, (void**)NULL);
    if(err != OK) return(FALSE);
    err =(*(ft_sim->findAnalysis))(ci->ci_ckt,&which, &anal,tranUid,
            ci->ci_curTask,(IFuid )NULL);
    if(err != OK) return(FALSE);
    err = if_analQbyName(ci->ci_ckt,which,anal,"tstart",&tmp);
    if(err != OK) return(FALSE);
    *start = tmp.rValue;
    err = if_analQbyName(ci->ci_ckt,which,anal,"tstop",&tmp);
    if(err != OK) return(FALSE);
    *stop = tmp.rValue;
    err = if_analQbyName(ci->ci_ckt,which,anal,"tstep",&tmp);
    if(err != OK) return(FALSE);
    *step = tmp.rValue;
    return (TRUE);
}

/* Get the statistic called 'name'.  If this is NULL get all statistics
 * available.
 */

struct variable *
if_getstat(void *ckt, char *name)
{
    int i;
    struct variable *v, *vars;
    IFvalue parm;
    int which = -1;

    for(i=0;i<ft_sim->numAnalyses;i++) {
        if(strcmp(ft_sim->analyses[i]->name,"options")==0) {
            which = i;
            break;
        }
    }
    if(which==-1) {
        fprintf(cp_err,"Warning:  statistics unsupported\n");
        return(NULL);
    }

    if (name) {
        for (i = 0; i < ft_sim->analyses[which]->numParms; i++)
            if (eq(ft_sim->analyses[which]->analysisParms[i].keyword, name))
                break;
        if (i == ft_sim->analyses[which]->numParms)
            return (NULL);
        if ((*(ft_sim->askAnalysisQuest))(ckt, ft_curckt->ci_curTask,
                ft_sim->analyses[which]->analysisParms[i].id, &parm, 
                (IFvalue *)NULL) == -1) {
            fprintf(cp_err, 
                "if_getstat: Internal Error: can't get %s\n",
                name);
            return (NULL);
        }
        return (parmtovar(&parm, &(ft_sim->analyses[which]->analysisParms[i])));
    } else {
        for (i = 0, vars = v = NULL; i<ft_sim->analyses[which]->numParms; i++) {
            if(!(ft_sim->analyses[which]->analysisParms[i].dataType&IF_ASK)) {
                continue;
            }
            if ((*(ft_sim->askAnalysisQuest))(ckt, ft_curckt->ci_curTask, 
                    ft_sim->analyses[which]->analysisParms[i].id, 
                    &parm, (IFvalue *)NULL) == -1) {
                fprintf(cp_err, 
                "if_getstat: Internal Error: can't get %s\n",
                    name);
                return (NULL);
            }
            if (v) {
                v->va_next = parmtovar(&parm, 
                        &(ft_sim->analyses[which]->analysisParms[i]));
                v = v->va_next;
            } else {
                vars = v = parmtovar(&parm, 
                        &(ft_sim->analyses[which]->analysisParms[i])); 
            }
        }
        return (vars);
    }
}

#ifdef EXPERIMENTAL_CODE

#include <cktdefs.h>
#include <trandefs.h>

/* arg0: circuit file, arg1: data file */
void com_loadsnap(wordlist *wl) {
  int error = 0;
  FILE *file;
  int tmpI, i, size;
  CKTcircuit *my_ckt, *ckt;
  
  /*
    Phesudo code:
  
    source(file_name);
    This should setup all the device structs, voltage nodes, etc.

    call cktsetup;
    This is needed to setup vector mamory allocation for vectors and branch nodes
  
    load_binary_data(info);
    Overwrite the allocated numbers, rhs etc, with saved data
  */
 

  if (ft_curckt) {
    fprintf(cp_err, "Error: there is already a circuit loaded.\n");
    return;
  }
  
  /* source the circuit */
  inp_source(wl->wl_word);
  if (!ft_curckt) {
    return;
  }
  
  /* allocate all the vectors, with luck!  */
  if (!error)
    error = CKTsetup((CKTcircuit *)ft_curckt->ci_ckt);
  if (!error)
    error = CKTtemp((CKTcircuit *)ft_curckt->ci_ckt);
  
  if(error) {
    fprintf(cp_err,"Some error in the CKT setup fncts!\n");
    return;
  }

  /* so it resumes ... */
  ft_curckt->ci_inprogress = TRUE;


  /* now load the binary file */
  
    
  ckt = (CKTcircuit *)ft_curckt->ci_ckt;

  file = fopen(wl->wl_next->wl_word,"rb");
    
  if(!file) {
    fprintf(cp_err, 
	    "Error: Couldn't open \"%s\" for reading\n",
	    wl->wl_next->wl_word);
    return;
  }
    
  fread(&tmpI,sizeof(int),1,file);
  if(tmpI != sizeof(CKTcircuit) ) {
    fprintf(cp_err,"loaded num: %d, expected num: %ld\n",tmpI,(long)sizeof(CKTcircuit));
    fprintf(cp_err, 
	    "Error: snapshot saved with different version of spice\n");
    fclose(file);
    return;
  }
    
  my_ckt = (CKTcircuit *)tmalloc(sizeof(CKTcircuit));

  fread(my_ckt,sizeof(CKTcircuit),1,file);

#define _t(name) ckt->name = my_ckt->name
#define _ta(name,size)\
 do{ int __i; for(__i=0;__i<size;__i++) _t(name[__i]); } while(0)

  _t(CKTtime);
  _t(CKTdelta);
  _ta(CKTdeltaOld,7);
  _t(CKTtemp);
  _t(CKTnomTemp);
  _t(CKTvt);
  _ta(CKTag,7);

  _t(CKTorder);
  _t(CKTmaxOrder);
  _t(CKTintegrateMethod);

  _t(CKTniState);

  _t(CKTmaxEqNum);
  _t(CKTcurrentAnalysis);

  _t(CKTnumStates);
  _t(CKTmode);

  _t(CKTbypass);
  _t(CKTdcMaxIter);
  _t(CKTdcTrcvMaxIter);
  _t(CKTtranMaxIter);
  _t(CKTbreakSize);
  _t(CKTbreak);
  _t(CKTsaveDelta);
  _t(CKTminBreak);
  _t(CKTabstol);
  _t(CKTpivotAbsTol);
  _t(CKTpivotRelTol);
  _t(CKTreltol);
  _t(CKTchgtol);
  _t(CKTvoltTol);

  _t(CKTgmin);
  _t(CKTgshunt);
  _t(CKTdelmin);
  _t(CKTtrtol);
  _t(CKTfinalTime);
  _t(CKTstep);
  _t(CKTmaxStep);
  _t(CKTinitTime);
  _t(CKTomega);
  _t(CKTsrcFact);
  _t(CKTdiagGmin);
  _t(CKTnumSrcSteps);
  _t(CKTnumGminSteps);
  _t(CKTgminFactor);
  _t(CKTnoncon);
  _t(CKTdefaultMosM);
  _t(CKTdefaultMosL);
  _t(CKTdefaultMosW);
  _t(CKTdefaultMosAD);
  _t(CKTdefaultMosAS);
  _t(CKThadNodeset);
  _t(CKTfixLimit);
  _t(CKTnoOpIter);
  _t(CKTisSetup);

  _t(CKTtimeListSize);
  _t(CKTtimeIndex);
  _t(CKTsizeIncr);

  _t(CKTtryToCompact);
  _t(CKTbadMos3);
  _t(CKTkeepOpInfo);
  _t(CKTcopyNodesets);
  _t(CKTnodeDamping);
  _t(CKTabsDv);
  _t(CKTrelDv);
  _t(CKTtroubleNode);
 
#undef _foo
#define _foo(name,type,_size)\
do {\
    int __i;\
    fread(&__i,sizeof(int),1,file);\
    if(__i) {\
      if(name)\
	tfree(name);\
      name = (type *)tmalloc(__i);\
      fread(name,1,__i,file);\
    } else {\
      fprintf(cp_err, "size for vector " #name " is 0\n");\
    }\
    if((_size) != -1 && __i != (_size) * sizeof(type)) {\
      fprintf(cp_err,"expected %ld, but got %d for "#name"\n",(long)(_size)*sizeof(type),__i);\
    }\
  } while(0)

    
    for(i=0;i<=ckt->CKTmaxOrder+1;i++) {
      _foo(ckt->CKTstates[i],double,ckt->CKTnumStates);
    }
 
    size = SMPmatSize(ckt->CKTmatrix) + 1;
    _foo(ckt->CKTrhs, double,size);
    _foo(ckt->CKTrhsOld, double,size);
    _foo(ckt->CKTrhsSpare, double,size);
    _foo(ckt->CKTirhs, double,size);	
    _foo(ckt->CKTirhsOld, double,size);		
    _foo(ckt->CKTirhsSpare, double,size);	
    _foo(ckt->CKTrhsOp, double,size);		
    _foo(ckt->CKTsenRhs, double,size);		    
    _foo(ckt->CKTseniRhs, double,size);
      
    _foo(ckt->CKTtimePoints,double,-1);
    _foo(ckt->CKTdeltaList,double,-1);

    _foo(ckt->CKTbreaks,double,ckt->CKTbreakSize);
    
    {	/* avoid invalid lvalue assignment errors in the macro _foo() */
    	TSKtask * lname = (TSKtask *)ft_curckt->ci_curTask;
    	_foo(lname,TSKtask,1);
    }

    /* To stop the Free */
    ((TSKtask *)ft_curckt->ci_curTask)->TSKname = NULL;
    ((TSKtask *)ft_curckt->ci_curTask)->jobs = NULL;

    _foo(((TSKtask *)ft_curckt->ci_curTask)->TSKname,char,-1);
    
    {	/* avoid invalid lvalue assignment errors in the macro _foo() */
    	TRANan * lname = (TRANan *)((TSKtask *)ft_curckt->ci_curTask)->jobs;
    	_foo(lname,TRANan,1);
    }
    ((TSKtask *)ft_curckt->ci_curTask)->jobs->JOBname = NULL;
    ckt->CKTcurJob = (JOB *)((TSKtask *)ft_curckt->ci_curTask)->jobs;
    
    _foo(((TSKtask *)ft_curckt->ci_curTask)->jobs->JOBname,char,-1);

    ((TSKtask *)ft_curckt->ci_curTask)->jobs->JOBnextJob = NULL;
    
    ((TRANan *)((TSKtask *)ft_curckt->ci_curTask)->jobs)->TRANplot = NULL;

    _foo(ckt->CKTstat,STATistics,1);

    tfree(my_ckt);
    fclose(file);
    
    /* Finally to resume the plot in some fashion */

    /* a worked out version of this should be enough */
    {
      IFuid *nameList;
      int numNames;
      IFuid timeUid;
 
      error = CKTnames(ckt,&numNames,&nameList);
      if(error){
	fprintf(cp_err,"error in CKTnames\n");
	return;
      }
      (*(SPfrontEnd->IFnewUid))((void *)ckt,&timeUid,(IFuid)NULL,
				"time", UID_OTHER, (void **)NULL);
      error = (*(SPfrontEnd->OUTpBeginPlot))((void *)ckt,
	     (void*)ckt->CKTcurJob,
	     ckt->CKTcurJob->JOBname,timeUid,IF_REAL,numNames,nameList,
	     IF_REAL,&(((TRANan*)ckt->CKTcurJob)->TRANplot));
      if(error) {
	fprintf(cp_err,"error in CKTnames\n");
	return;
      }	
    }
    
    
    
    return ;
}

void com_savesnap(wordlist *wl) {
  FILE *file;
  int i, size;
  CKTcircuit *ckt;
  TSKtask *task;
  
  if (!ft_curckt) {
    fprintf(cp_err, "Error: there is no circuit loaded.\n");
    return;
  } else if (ft_curckt->ci_ckt == NULL) { /* Set noparse? */
    fprintf(cp_err, "Error: circuit not parsed.\n");
    return;
  }

  /* save the data */
  
  ckt = (CKTcircuit *)ft_curckt->ci_ckt;
    
  task = (TSKtask *)ft_curckt->ci_curTask;

  if(task->jobs->JOBtype != 4) {
    fprintf(cp_err,"Only saving of tran analysis is implemented\n");
    return;
  }
  
  file = fopen(wl->wl_word,"wb");
 
  if(!file) {
    fprintf(cp_err, 
	    "Error: Couldn't open \"%s\" for writing\n",wl->wl_word);
    return;
  }

#undef _foo
#define _foo(name,type,num)\
 do {\
      int __i;\
      if(name) {\
	__i = (num) * sizeof(type); fwrite(&__i,sizeof(int),1,file);\
	if((num))\
	  fwrite(name,sizeof(type),(num),file);\
      } else {\
	__i = 0;\
	fprintf(cp_err,#name " is NULL, zero written\n");\
	fwrite(&__i,sizeof(int),1,file);\
      }\
    } while(0)
    

  _foo(ckt,CKTcircuit,1);

  /* To save list
       
  double *(CKTstates[8]);
  double *CKTrhs;		
  double *CKTrhsOld;		                            
  double *CKTrhsSpare;	
  double *CKTirhs;		
  double *CKTirhsOld;		
  double *CKTirhsSpare;	
  double *CKTrhsOp;		
  double *CKTsenRhs;		    
  double *CKTseniRhs;	       
  double *CKTtimePoints;	 list of all accepted timepoints in
  the current transient simulation 
  double *CKTdeltaList;	 list of all timesteps in the
  current transient simulation 

  */
    

  for(i=0;i<=ckt->CKTmaxOrder+1;i++) {
    _foo(ckt->CKTstates[i],double,ckt->CKTnumStates);
  }
    
    
      
  size = SMPmatSize(ckt->CKTmatrix) + 1;
      
  _foo(ckt->CKTrhs,double,size);
  _foo(ckt->CKTrhsOld,double,size);		                            
  _foo(ckt->CKTrhsSpare,double,size);	
  _foo(ckt->CKTirhs,double,size);		
  _foo(ckt->CKTirhsOld,double,size);		
  _foo(ckt->CKTirhsSpare,double,size);	
  _foo(ckt->CKTrhsOp,double,size);		
  _foo(ckt->CKTsenRhs,double,size);		    
  _foo(ckt->CKTseniRhs,double,size);

  _foo(ckt->CKTtimePoints,double,ckt->CKTtimeListSize);
  _foo(ckt->CKTdeltaList,double,ckt->CKTtimeListSize);	       
    
  /* need to save the breakpoints, or something */

  _foo(ckt->CKTbreaks,double,ckt->CKTbreakSize);


  /* now save the TSK struct, ft_curckt->ci_curTask*/
  
  _foo(task,TSKtask,1);
  _foo(task->TSKname,char,(strlen(task->TSKname)+1));

  /* now save the JOB struct task->jobs */
  /* lol, only allow one job, tough! */
  /* Note that JOB is a base class, need to save actual type!! */
  
  _foo(task->jobs,TRANan,1);
  
  _foo(task->jobs->JOBname,char,(strlen(task->jobs->JOBname)+1));


  /* Finally the stats */

  _foo(ckt->CKTstat,STATistics,1);

  
  fclose(file);
  
  return;

}

#endif /* EXPERIMENTAL_CODE */
