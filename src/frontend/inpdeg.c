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

#include "inpcom.h"

/* maximum number of model parameters */
#define DEGPARAMAX 64
/* maximum number of models */
#define DEGMODMAX 64

struct agemod {
    char* devmodel;
    char* simmodel;
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
                controlled_exit(1);
            }
            tfree(dftok);
            tfree(f1);
            ftok = dftok = gettok(&cut_line);
            f1 = gettok_char(&ftok, '=', TRUE, FALSE);
            if (f1 && ciprefix("simmodel=", f1))
                agemods[ageindex].simmodel = copy(ftok);
            else {
                fprintf(stderr, "Error: bad .agemodel syntax in line\n    %s", card->line);
                controlled_exit(1);
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
                /* parameter name */
                f1 = gettok_char(&ftok, '=', FALSE, FALSE);
                if (!f1) {
                    fprintf(stderr, "Error: bad .agemodel syntax in line\n % s", card->line);
                    controlled_exit(1);
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

    return 0;
}

/* Look for an X line.
   Check if the model in the x line is found in the model list agemodds.
   Create a degradation monitor for each x line if model found.
   Add the x instance name to the degmod name.
   Add degmon line and its model line to the netlist.*/
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
                             fprintf(stderr, "Error, l not found in device %s \n\n", deck->line);
                             return 1;
                        }
                        char* clength = gettok(&lpos);
                        /* Now add a degradation monitor like
                           adegmon1 %v([z a vss vss]) mon degmon1
                          .model degmon1 degmon (tfuture=3153360000 l=0.15e-6 devmod="sg13_lv_nmos")
                         */
                        char* aline = tprintf("adegmon%d_%s %%v([%s]) mon%d degmon%d\n", 
                            degmonno, instname, fournodes, degmonno, degmonno);
                        char* mline = tprintf(".model degmon%d degmon (tfuture=%e %s devmod=\"%s\"\n",
                            degmonno, tfuture, clength, modname);
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
    return 0;
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
