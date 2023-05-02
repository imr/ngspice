/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/ifsim.h"
#include "ngspice/inpdefs.h"
#include "ngspice/inpptree.h"
#include "inpxx.h"

extern bool ft_ngdebug;

IFvalue *
INPgetValue(CKTcircuit *ckt, char **line, int type, INPtables *tab)
{
    double *list;
    int *ilist;
    double tmp;
    char *word;
    int error;
    static IFvalue temp;
    INPparseTree *pt;
    char *compline = *line;

    /* make sure we get rid of extra bits in type */
    type &= IF_VARTYPES;
    if (type == IF_INTEGER) {
        tmp = INPevaluate(line, &error, 1);
        temp.iValue = (int) floor(0.5 + tmp);
        /* printf(" returning integer value %d\n",temp.iValue); */
    } else if (type == IF_REAL) {
        temp.rValue = INPevaluate(line, &error, 1);
        /* printf(" returning real value %e\n",temp.rValue); */
    } else if (type == IF_REALVEC) {
        /* read until error occurs. If error, and first
           character of remaining line is ')', everything is o.k.
           If first token is already in error, return NULL.*/
        temp.v.numValue = 0;
        list = TMALLOC(double, 1);
        tmp = INPevaluate(line, &error, 1);
        if (error) {
            if(ft_ngdebug)
                fprintf(stderr, "\nError: Could not read parameter in front of\n    %s\n", *line);
            tfree(list);
            return NULL;
        }
        while (error == 0) {
            /* printf(" returning vector value %g\n",tmp); */
            temp.v.numValue++;
            list = TREALLOC(double, list, temp.v.numValue);
            list[temp.v.numValue - 1] = tmp;
            tmp = INPevaluate(line, &error, 1);
        }
        if (error && ft_ngdebug && !eq(*line, "") && !prefix(")", *line) &&
            temp.v.numValue > 1) {
            fprintf(stderr, "\nWarning: Reading a vector without limiting parens may be dangerous\n%s\nat\n", compline);
            fprintf(stderr, "%*s%s\n", (int)(*line - compline)," ", *line);
        }
        temp.v.vec.rVec = list;
    } else if (type == IF_INTVEC) {
        /* read until error occurs. If error, and first 
           character of remaining line is ')', everything is o.k. 
           If first token is already in error, return NULL.*/
        temp.v.numValue = 0;
        ilist = TMALLOC(int, 1);
        tmp = INPevaluate(line, &error, 1);
        if (error) {
            tfree(ilist);
            return NULL;
        }
        while (error == 0) {
            /* printf(" returning vector value %g\n",tmp); */
            temp.v.numValue++;
            ilist = TREALLOC(int, ilist, temp.v.numValue);
            ilist[temp.v.numValue - 1] = (int) floor(0.5 + tmp);
            tmp = INPevaluate(line, &error, 1);
        }
        if (error && ft_ngdebug && !eq(*line, "") && !prefix(")", *line) &&
            temp.v.numValue > 1) {
            fprintf(stderr, "\nWarning: Reading a vector without limiting parens may be dangerous\n%s\nat\n", compline);
            fprintf(stderr, "%*s%s\n", (int)(*line - compline), " ", *line);
        }
        temp.v.vec.iVec = ilist;
    } else if (type == IF_FLAG) {
        temp.iValue = 1;
    } else if (type == IF_NODE) {
        INPgetNetTok(line, &word, 1);
        INPtermInsert(ckt, &word, tab, &(temp.nValue));
    } else if (type == IF_INSTANCE) {
        INPgetTok(line, &word, 1);
        INPinsert(&word, tab);
        temp.uValue = word;
    } else if (type == IF_STRING) {
        INPgetStr(line, &word, 1);
        temp.sValue = word;
    } else if (type == IF_PARSETREE) {
        INPgetTree(line, &pt, ckt, tab);
        if (!pt)
            return NULL;
        temp.tValue = (IFparseTree *) pt;
        /* INPptPrint("Parse tree is: ", temp.tValue); */
    } else {
        /* don't know what type of parameter caller is talking about! */
        return NULL;
    }

    return &temp;
}
