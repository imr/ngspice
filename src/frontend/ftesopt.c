/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 2010 Paolo Nenzi
**********/


#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ftedefs.h"
#include "ngspice/wordlist.h"
#include "variable.h"
#include "ngspice/sperror.h"


struct FTEparm {
    char *keyword;
    int id;
    enum cp_types dataType;
    char *description;
};


static struct FTEparm FTEOPTtbl[] = {
    { "decklineno",   FTEOPT_NLDECK, CP_NUM,  "Number of lines in the deck" },
    { "netloadtime",  FTEOPT_NLT,    CP_REAL, "Netlist loading time"        },
    { "netparsetime", FTEOPT_NPT,    CP_REAL, "Netlist parsing time"        }
};

static const int FTEOPTcount = sizeof(FTEOPTtbl)/sizeof(*FTEOPTtbl);

static struct variable *getFTEstat(struct circ *, int);


struct variable *
ft_getstat(struct circ *ft_curckt, char *name)
{
    int i;
    struct variable *v, *vars , *vv = NULL;

    if (name) {
        for (i = 0; i < FTEOPTcount; i++)
            if (eq(name, FTEOPTtbl[i].keyword)) {
                vv = getFTEstat(ft_curckt, FTEOPTtbl[i].id);
                if (vv) {
                    vv->va_type = FTEOPTtbl[i].dataType;
                    vv->va_name = copy(FTEOPTtbl[i].description);
                    vv->va_next = NULL;
                    return (vv);
                } else {
                    return (NULL);
                }
            }
        return (NULL);
    } else {
        for (i = 0, v = vars = NULL; i < FTEOPTcount; i++) {
            if (v) {
                v->va_next = getFTEstat(ft_curckt, FTEOPTtbl[i].id);
                v = v->va_next;
            } else {
                vars = v = getFTEstat(ft_curckt, FTEOPTtbl[i].id);
            }

            v->va_type = FTEOPTtbl[i].dataType;
            v->va_name = copy(FTEOPTtbl[i].description);

        }
        return vars;
    }
}


/* This function fill the value field of the variable */

static struct variable *
getFTEstat(struct circ *ft_curckt, int id)
{

    struct variable *v = TMALLOC(struct variable, 1);

    switch (id) {
    case FTEOPT_NLDECK:
        v->va_num = ft_curckt->FTEstats->FTESTATdeckNumLines;
        break;
    case FTEOPT_NLT:
        v->va_real = ft_curckt->FTEstats->FTESTATnetLoadTime;
        break;
    case FTEOPT_NPT:
        v->va_real = ft_curckt->FTEstats->FTESTATnetParseTime;
        break;
    default:
        tfree(v);
        return (NULL);
    }

    return (v);
}
