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


