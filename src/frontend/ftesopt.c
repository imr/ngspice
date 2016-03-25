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

static struct variable *getFTEstat(struct FTEparm *, FTESTATistics *, struct variable *);


struct variable *
ft_getstat(struct circ *ci, char *name)
{
    int i;

    if (name) {
        for (i = 0; i < FTEOPTcount; i++)
            if (eq(name, FTEOPTtbl[i].keyword))
                return getFTEstat(FTEOPTtbl + i, ci->FTEstats, NULL);
        return (NULL);
    } else {
        struct variable *vars = NULL;
        for (i = FTEOPTcount; --i >= 0;)
            vars = getFTEstat(FTEOPTtbl + i, ci->FTEstats, vars);
        return vars;
    }
}


static struct variable *
getFTEstat(struct FTEparm *p, FTESTATistics *stat, struct variable *next)
{

    struct variable *v = TMALLOC(struct variable, 1);

    switch (p->id) {
    case FTEOPT_NLDECK:
        v->va_name = copy(p->description);
        v->va_next = next;
        v->va_type = p->dataType;
        v->va_num = stat->FTESTATdeckNumLines;
        return v;
    case FTEOPT_NLT:
        v->va_name = copy(p->description);
        v->va_next = next;
        v->va_type = p->dataType;
        v->va_real = stat->FTESTATnetLoadTime;
        return v;
    case FTEOPT_NPT:
        v->va_name = copy(p->description);
        v->va_next = next;
        v->va_type = p->dataType;
        v->va_real = stat->FTESTATnetParseTime;
        return v;
    default:
        tfree(v);
        return (NULL);
    }
}
