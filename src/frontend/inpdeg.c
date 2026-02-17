/* inpdeg.c
Read and parse the .agemodel parameters of the ngspice netlist
Store them in a hash table ageparams

Copyright Holger Vogt 2025
License: Modified BSD
*/

#include <stdio.h>

#include "ngspice/ngspice.h"
#include "ngspice/inpdefs.h"
#include "ngspice/hash.h"
#include "ngspice/compatmode.h"
#include "ngspice/fteext.h"

#include "inpcom.h"

int prepare_degsim(struct card* deck);
int prepare_plainsim(struct card* deck);
int clear_degsim(void);
static int add_degmodel(struct card* deck, double* result);

/* maximum number of model parameters */
#define DEGPARAMAX 64
/* maximum number of models */
#define DEGMODMAX 64

/* global pointer: results from first tran run */
NGHASHPTR degdatahash = NULL;

struct agemod {
    char* devmodel;
    char* simmodel;
    int type;
    int numparams;
    char *paramnames[DEGPARAMAX];
    char *paramvalstr[DEGPARAMAX];
    double paramvals[DEGPARAMAX];
    bool paramread[DEGPARAMAX];
    NGHASHPTR paramhash;
} agemods[DEGMODMAX];

int readdegparams (struct card *deck) {
    struct card* card;
    int ageindex = 0;

    for (card = deck; card; card = card->nextcard) {
        if (ciprefix(".agemodel", card->line)) {
            /* comment out .agemodel, if compatmode is not set */
            if (!newcompat.de) {
                *card->line = '*';
                continue;
            }
            int parno = 0;

            if (ageindex == DEGMODMAX) {
                fprintf(stderr, "Error: Too many agemodels ( > %d)\n", DEGMODMAX);
                *card->line = '*';
                continue;
            }

            agemods[ageindex].paramhash = nghash_init(64);

            card->line = inp_remove_ws(card->line);

            char* cut_line = card->line;
            cut_line = nexttok(cut_line); // skip *agemodel
            char* ftok, *dftok, *f1;
            ftok = dftok = gettok(&cut_line);
            f1 = gettok_char(&ftok, '=', TRUE, FALSE);
            if (f1 && ciprefix("devmodel=", f1))
                agemods[ageindex].devmodel = copy(ftok);
            else {
                fprintf(stderr, "Error: bad .agemodel syntax in line\n    %s", card->line);
                continue;
            }
            tfree(dftok);
            tfree(f1);
            ftok = dftok = gettok(&cut_line);
            f1 = gettok_char(&ftok, '=', TRUE, FALSE);
            if (f1 && ciprefix("simmodel=", f1))
                agemods[ageindex].simmodel = copy(ftok);
            else {
                fprintf(stderr, "Error: bad .agemodel syntax in line\n    %s", card->line);
                continue;
            }
            tfree(dftok);
            tfree(f1);
            ftok = dftok = gettok(&cut_line);
            f1 = gettok_char(&ftok, '=', TRUE, FALSE);
            if (f1 && ciprefix("type=", f1))
                agemods[ageindex].type = atoi(ftok);
            else {
                fprintf(stderr, "Error: bad .agemodel syntax in line\n    %s", card->line);
                continue;
            }
            tfree(dftok);
            tfree(f1);

            /* now read all other parameters */
            while (cut_line && *cut_line) {
                if (parno == DEGPARAMAX) {
                    fprintf(stderr, "Error: Too many model parameters (> %d) in line\n", DEGPARAMAX);
                    fprintf(stderr, "    %s\n", card->line);
                    *card->line = '*';
                    break;
                }
                char* f2 = NULL;
                int err = 0;
                ftok = dftok = gettok(&cut_line);
                if (!dftok) {
                    fprintf(stderr, "Error: bad .agemodel syntax in line\n % s", card->line);
                    continue;
                }
                /* parameter name */
                f1 = gettok_char(&ftok, '=', FALSE, FALSE);
                if (!f1) {
                    fprintf(stderr, "Error: bad .agemodel syntax in line\n % s", card->line);
                    tfree(dftok);
                    continue;
                }
                /* parameter value */
                f2 = copy(ftok + 1);
                agemods[ageindex].paramnames[parno] = f1;
                agemods[ageindex].paramvalstr[parno] = f2;
                char *fp = f2;
                agemods[ageindex].paramvals[parno] = INPevaluate(&fp, &err, 1);
                if (err != 0)
                    fprintf(stderr, "\nError: Could not evaluate parameter %s\n", f2);
                else {
                    agemods[ageindex].paramread[parno] = FALSE;
                    nghash_insert(agemods[ageindex].paramhash, f1, &(agemods[ageindex].paramvals[parno]));
                    parno++;
                    agemods[ageindex].numparams = parno;
                }

                tfree(dftok);
            }
            ageindex++;
            *card->line = '*';
        }
    }

    return ageindex;
}

/* Look for an X line.
   Check if the model in the x line is found in the model list agemodds.
   Create a degradation monitor for each x line if model found.
   Add the x instance name to the degmod name.
   Add degmon line and its model line to the netlist.
   Return number of degradation monitors, or 0 in case of error. */
int adddegmonitors(struct card* deck) {
    static int degmonno;
    double tfuture = 315336e3; /* 10 years */
    int nodes = 4;
    if (agemods[0].paramhash == NULL)
        return 1;
    for (; deck; deck = deck->nextcard) {
        int skip_control = 0;

        char* line = deck->line;

        if (*line == '*') {
            continue;
        }
        /* there is no e source inside .control ... .endc */
        if (ciprefix(".control", line)) {
            skip_control++;
            continue;
        }
        else if (ciprefix(".endc", line)) {
            skip_control--;
            continue;
        }
        else if (skip_control > 0) {
            continue;
        }
        if (*line == 'x') {
            char *modname, *fournodes, *instname;
            int ii;
//            fprintf(stdout, "%.80s\n", line);
            /* x  instance model in subcircuit */
            /* get instance name */
            instname = gettok_instance(&line);
            fournodes = line;
            /* and 4 nodes */
            for (ii = 0; ii < nodes; ii++)
                line = nexttok(line);
            if (!line) {
                /* Must be something else, not a 4-node MOS */
                continue;
            }
            fournodes = copy_substring(fournodes, line);
            modname = gettok(&line);

            /*check if model is available in agemods */
            for (ii = 0; ii < DEGMODMAX; ii++) {
                if (agemods[ii].devmodel) {
                    if (cieq(modname, agemods[ii].devmodel)) {
//                        fprintf(stdout, "device model %s found as no. %d\n\n", modname, ii);
                        /* get the channel length */
                        char* lpos = strstr(line, "l=");
                        if (!lpos) {
                             fprintf(stderr, "Error, channel length l not found in device %s \n\n", deck->line);
                             return 0;
                        }
                        /* get l=val [m] */
                        char* clength = gettok(&lpos);
                        /* Now add a degradation monitor like
                           adegmon1 %v([z a vss vss]) mon degmon1
                          .model degmon1 degmon (tfuture=3153360000 l=0.15e-6 devmod="sg13_lv_nmos")
                         */
                        char* aline = tprintf("adegmon%d_%s %%v([%s]) mon%d degmon%d\n", 
                            degmonno, instname, fournodes, degmonno, degmonno);
                        char* mline = tprintf(".model degmon%d degmon (tfuture=%e %s devmod=\"%s\" instname=\"%s\"\n",
                            degmonno, tfuture, clength, modname, instname);
                        tfree(clength);
                        insert_new_line(deck, aline, 0, deck->linenum_orig, deck->linesource);
                        insert_new_line(deck, mline, 0, deck->linenum_orig, deck->linesource);
                        degmonno++;
                        break;
                    }
                }
                else {
//                    fprintf(stderr, "No model found for device %.80s \n\n", deck->line);
                    break;
                }
            }
            tfree(fournodes);
            tfree(modname);
            tfree(instname);
        }
    }
    /* initialze the result data storage */
    degdatahash = nghash_init(64);

    return degmonno;
}

int quote_degmons(struct card* deck) {
    for (; deck; deck = deck->nextcard) {
        int skip_control = 0;

        char* line = deck->line;

        if (*line == '*') {
            continue;
        }
        /* there is no e source inside .control ... .endc */
        if (ciprefix(".control", line)) {
            skip_control++;
            continue;
        }
        else if (ciprefix(".endc", line)) {
            skip_control--;
            continue;
        }
        else if (skip_control > 0) {
            continue;
        }
        if (*line == 'a' && strstr(line, "adegmon")) {
            char allnodes[1024];
            allnodes[0] = '\0';
            int ii, nodes = 4;
            char* newnodes, *instname;
            instname = gettok_instance(&line);
            /* skip %v */
            line = nexttok(line);
            char* deltoken;
            char* nodetoken = deltoken = gettok_char(&line, ']', false, true);
            if (!nodetoken)
                break;
            /* go beyond '[' */
            nodetoken++;
            for (ii = 0; ii < nodes; ii++) {
                char* nexttoken = gettok(&nodetoken);
                sprintf(allnodes, "%s \"%s\"", allnodes, nexttoken);
                if (!nexttoken)
                    break;
                tfree(nexttoken);
            }
            if (!line || eq(line, "")) {
                /* Must be something else, not a 4-node MOS */
                continue;
            }
            newnodes = tprintf("%s %%v [ %s %s", instname, allnodes, line);

            tfree(deltoken);
            tfree(instname);
            tfree(deck->line);
            deck->line = newnodes;
        }
    }
    return 0;
}

/* Replace '[' and ']' by '@'
   Required by code model parsing, when [ or ] are part of instance or node name. */
int remsqrbra(struct card* deck) {

    for (; deck; deck = deck->nextcard) {
        char* line = deck->line;

        if (*line == '*') {
            continue;
        }
        if (*line == 'a') {
            continue;
        }

        while (*line) {
            if (*line == '[' || *line == ']')
                *line = '@';
            line++;
        }
    }
    return 0;
}

/* Remove the degradation monitors. */
int prepare_plainsim(struct card* deck) {
    struct card* ldeck;
    /* skip the title line */
    for (ldeck = deck->nextcard; ldeck; ldeck = ldeck->nextcard) {
        char* line = ldeck->line;

        if (*line == '*') {
            continue;
        }

        /* remove the remnants of the first run */
        if (ciprefix(".model", line) && search_plain_identifier(line, "degmon")) {
            struct card* nextdeck = ldeck->nextcard;
            if (nextdeck) {
                char* nextline = nextdeck->line;
                if (*nextline == 'a' && strstr(nextline, "degmon")) {
                    *nextline = '*';
                }
            }
        }
    }
    return 0;
}

/* Remove the degradation monitors.
   Add instance parameters delvto and factuo.
   Use the data retrieved from degdatahash */
int prepare_degsim(struct card* deck) {
    struct card* prevcard = deck, *ldeck;
    int no_devs = 0;

    if (!deck)
        return 1;

    /* skip the title line */
    for (ldeck = deck->nextcard; ldeck; ldeck = ldeck->nextcard) {
        char* line = ldeck->line;

        if (*line == '*') {
            continue;
        }

        /* remove the remnants of the first run */
        if (ciprefix(".model", line) && search_plain_identifier(line, "degmon")) {
            double *result;
            char* insttoken;
            struct card* nextdeck = ldeck->nextcard;
            char* prevline = prevcard->line;
            if (nextdeck) {
                char* nextline = nextdeck->line;
                if (*nextline == 'a' && strstr(nextline, "degmon")) {
                    *nextline = '*';
                }
            }
            /* get the device instance line */
            insttoken = gettok_instance(&prevline);
            if (*insttoken != 'n' && *insttoken != 'm'){
                fprintf(stderr, "Error: expected N or M device, but found %s\n", insttoken);
                continue;
            }
            result = (double*)nghash_find(degdatahash, insttoken);
            /* only if significant degradation is measured */
            if (result && (result[0] != 0 || result[1] != 0 || result[2] != 0)) {
                fprintf(stdout, "Instance %s, Result: %e, %e, %e\n", insttoken, result[0], result[1], result[2]);
                /* this will add the necessary devices to
                   the netlist according to the deg model */
                add_degmodel(prevcard, result);
                no_devs++;
            }
            else if (ft_ngdebug) {
                fprintf(stdout, "Instance %s\n", insttoken);
                fprintf(stdout, "     degradation not significant, no data available\n", insttoken);
            }

            tfree(insttoken);

            *line = '*';
        }
        prevcard = ldeck;
    }
    if (no_devs == 1)
        fprintf(stdout, "Note: degradation simulation prepared, 1 device degrades.\n");
    else
        fprintf(stdout, "Note: degradation simulation prepared, %d devices degrade.\n", no_devs);
    return 0;
}

/* user defined delete function */
static void
del_data(void* rdata)
{
    double* data = (double*)rdata;
    if (data) {
        tfree(data);
    }
}

/* clear memory */
int clear_degsim (void){
    /* delete the result hashtable */
    if(degdatahash)
        nghash_free(degdatahash, del_data, NULL);
    return 0;
}

/* Use the mean of d_idlin (result[1]) and d_idsat (result[2]) plus 1 as instance parameter factuo.
   Use dlt_vth shift from result[0] as instance parameter delvto. */
static int add_degmodel(struct card* deck, double* result) {

    char* curr_line = deck->line;
    double currdeg = (result[1] + result[2]) / 2.;
    bool currd = FALSE;
    bool vts = FALSE;

    if (fabs(currdeg) >= 1.) {
        fprintf(stderr, "Warning: drain current degradation greater than 100%%\n");
    }
    else if (currdeg > 0.) {
        fprintf(stderr, "Warning: drain current increases\n");
        currd = TRUE;
    }
    else {
        /* parallel drain current */
        currd = TRUE;
    }
    /* gate voltage shift */
    if (result[0] != 0.) {
        vts = TRUE;
    }

    /* modify the instance line */
    char* instline = NULL;
    if (vts && currd) {
        instline = tprintf("%s delvto=%e factuo=%e\n", 
            curr_line, result[0], 1.+ currdeg);
        if (ft_ngdebug) {
            fprintf(stdout, "Instance now has extra delvto=%e factuo=%e\n", result[0], 1. + currdeg);
        }
    }
    else if (vts && !currd) {
        instline = tprintf("%s delvto=%e\n",
            curr_line, result[0]);
        if (ft_ngdebug) {
            fprintf(stdout, "Instance now has extra delvto=%e\n", result[0]);
        }
    }
    else if (!vts && currd) {
        instline = tprintf("%s factuo=%e\n",
            curr_line, 1. + currdeg);
        if (ft_ngdebug) {
            fprintf(stdout, "Instance now has extra factuo=%e\n", 1. + currdeg);
        }
    }

    if (instline) {
        tfree(deck->line);
        deck->line = instline;
    }
    return 0;
}
