/* Routines to evaluate the .measure cards.
   Entry point is function do_measure(), called by fcn dosim()
   from runcoms.c:335, after simulation is finished.

   In addition it contains the fcn com_meas(), which provide the
   interactive 'meas' command.
*/

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"

#include "rawfile.h"
#include "variable.h"
#include "numparam/numpaif.h"
#include "ngspice/missing_math.h"
#include "com_measure2.h"
#include "com_let.h"
#include "com_commands.h"
#include "com_display.h"


static wordlist *measure_parse_line(char *line);

extern bool ft_batchmode;
extern bool rflag;


/* measure in interactive mode:
   meas command inside .control ... .endc loop or manually entered.
   meas has to be followed by the standard tokens (see measure_extract_variables()).
   The result is put into a vector with name "result"
*/

void
com_meas(wordlist *wl)
{
    /* wl: in, input line of meas command */
    char *line_in, *outvar;
    wordlist *wl_count, *wl_let;

    char *vec_found, *token, *equal_ptr;
    wordlist *wl_index;
    struct dvec *d;
    int err = 0;

    int fail;
    double result = 0;

    if (!wl) {
        com_display(NULL);
        return;
    }
    wl_count = wl;

    /* check each wl entry, if it contain '=' and if the following token is
       a single valued vector. If yes, replace this vector by its value.
       Vectors may stem from other meas commands, or be generated elsewhere
       within the .control .endc script. All other right hand side vectors are
       treated in com_measure2.c. */
    wl_index = wl;

    while (wl_index) {
        token = wl_index->wl_word;
        /* find the vector vec_found, next token after each '=' sign.
           May be in the next wl_word */
        if (token[strlen(token) - 1] == '=') {
            wl_index = wl_index->wl_next;
            if (wl_index == NULL) {
                line_in = wl_flatten(wl);
                fprintf(stderr, "\nError: meas failed due to missing token in \n    meas %s \n\n", line_in);
                tfree(line_in);
                return;
            }
            vec_found = wl_index->wl_word;
            /* token may be already a value, maybe 'LAST', which we have to keep, or maybe a vector */
            if (!cieq(vec_found, "LAST")) {
                INPevaluate(&vec_found, &err, 1);
                /* if not a valid number */
                if (err) {
                    /* check if vec_found is a valid vector */
                    d = vec_get(vec_found);
                    /* Only if we have a single valued vector, replacing
                       of the rigt hand side does make sense */
                    if (d && (d->v_length == 1) && (d->v_numdims == 1)) {
                        /* get its value */
                        wl_index->wl_word = tprintf("%e", d->v_realdata[0]);
                        tfree(vec_found);
                    }
                }
            }
        }
        /* may be inside the same wl_word */
        else if ((equal_ptr = strchr(token, '=')) != NULL) {
            vec_found = equal_ptr + 1;
            if (!cieq(vec_found, "LAST")) {
                INPevaluate(&vec_found, &err, 1);
                if (err) {
                    d = vec_get(vec_found);
                    /* Only if we have a single valued vector, replacing
                    of the rigt hand side does make sense */
                    if (d && (d->v_length == 1) && (d->v_numdims == 1)) {
                        int lhs_len = (int)(equal_ptr - token);
                        wl_index->wl_word =
                            tprintf("%.*s=%e", lhs_len, token, d->v_realdata[0]);
                        tfree(token);
                    }
                }
            }
        } else {
            ;                   // nothing
        }
        wl_index = wl_index->wl_next;
    }

    line_in = wl_flatten(wl);

    /* get output var name */
    wl_count = wl_count->wl_next;
    if (!wl_count) {
        fprintf(stdout,
                " meas %s failed!\n"
                "   unspecified output var name\n\n", line_in);
        tfree(line_in);
        return;
    }
    outvar = wl_count->wl_word;

    fail = get_measure2(wl, &result, NULL, FALSE);

    if (fail) {
        fprintf(stdout, " meas %s failed!\n\n", line_in);
        tfree(line_in);
        return;
    }

    wl_let = wl_cons(tprintf("%s = %e", outvar, result), NULL);
    com_let(wl_let);
    wl_free(wl_let);
    tfree(line_in);
}


static bool
chkAnalysisType(char *an_type)
{
    /* only support tran, dc, ac, sp analysis type for now */
    if (strcmp(an_type, "tran") != 0 && strcmp(an_type, "ac") != 0 &&
            strcmp(an_type, "dc") != 0 && strcmp(an_type, "sp") != 0)
        return FALSE;
    else
        return TRUE;
}


/* Gets pointer to double value after 'xxx=' and advances pointer of *line.
   On error returns FALSE. */
static bool
get_double_value(
    char **line,   /*in|out: pointer to line to be parsed */
    char *name,    /*in: xxx e.g. 'val' from 'val=0.5' */
    double *value, /*out: return value (e.g. 0.5) from 'val=0.5'*/
    bool just_chk_meas /* in: just check measurement if true */
)
{
    char *token     = gettok(line);
    bool return_val = TRUE;
    char *equal_ptr, *junk;
    int  err = 0;

    if (name && (strncmp(token, name, strlen(name)) != 0)) {
        if (just_chk_meas != TRUE) fprintf(cp_err, "Error: syntax error for measure statement; expecting next field to be '%s'.\n", name);
        return_val = FALSE;
    } else {
        /* see if '=' is last char of current token -- implies we need to read value in next token */
        if (token[strlen(token) - 1] == '=') {
            txfree(token);
            junk = token = gettok(line);

            *value = INPevaluate(&junk, &err, 1);
        } else {
            if ((equal_ptr = strchr(token, '=')) != NULL) {
                equal_ptr += 1;
                *value = INPevaluate(&equal_ptr, &err, 1);
            } else {
                if (just_chk_meas != TRUE)
                    fprintf(cp_err, "Error: syntax error for measure statement; missing '='!\n");
                return_val = FALSE;
            }
        }
        if (err) {
            if (just_chk_meas != TRUE)
                fprintf(cp_err, "Error: Bad value.\n");
            return_val = FALSE;
        }
    }
    txfree(token);

    return return_val;
}


/* Entry point for .meas evaluation.
   Called in fcn dosim() from runcoms.c:335, after simulation is finished
   with chk_only set to FALSE.
   Called from fcn check_autostop(),
   with chk_only set to TRUE (no printouts, no params set).
   This function returns TRUE if all measurements are ready and complete;
   FALSE otherwise.  If called with chk_only, we can exit early if we
   fail a test in order to reduce execution time.  */
bool
do_measure(
    char *what,   /*in: analysis type*/
    bool chk_only /*in: TRUE if checking for "autostop", FALSE otherwise*/
)
{
    struct card *meas_card, *meas_results = NULL, *end = NULL, *newcard;
    char        *line, *an_name, *an_type, *resname, *meastype, *str_ptr, out_line[1000], out_file[1000];
    int         ok = 0;
    int         fail;
    int         num_failed = 0;
    double      result = 0;
    bool        first_time = TRUE;
    bool        measures_passed;
    wordlist    *measure_word_list;
    int         precision = measure_get_precision();
    FILE       *measout = NULL;

#ifdef HAS_PROGREP
    if (!chk_only)
        SetAnalyse("meas", 0);
#endif

    an_name = copy(what); /* analysis type, e.g. "tran" */
    strtolower(an_name);
    measure_word_list = NULL;
    measures_passed = TRUE;

    /* don't allow .meas if batchmode is set by -b and -r rawfile given */
    if (ft_batchmode && rflag) {
        fprintf(cp_err, "\nNo .measure possible in batch mode (-b) with -r rawfile set!\n");
        fprintf(cp_err, "Remove rawfile and use .print or .plot or\n");
        fprintf(cp_err, "select interactive mode (optionally with .control section) instead.\n\n");
        return (measures_passed);
    }

    /* don't allow autostop if no .meas commands are given in the input file */
    if ((cp_getvar("autostop", CP_BOOL, NULL, 0)) && (ft_curckt->ci_meas == NULL)) {
        fprintf(cp_err, "\nWarning: No .meas commands found!\n");
        fprintf(cp_err, "  Option autostop is not available, ignored!\n\n");
        cp_remvar("autostop");
        return (FALSE);
    }

    if (cp_getvar("measoutfile", CP_STRING, out_file, sizeof(out_file))) {
        measout = fopen(out_file, "w");
        if (!measout)
            fprintf(stderr, " Warning: Could not open file %s\n", out_file);
    }

    /* Evaluating the linked list of .meas cards, assembled from the input deck
       by fcn inp_spsource() in inp.c:575.
       A typical .meas card will contain:
       parameter        value
       nameof card      .meas(ure)
       analysis type    tran        only tran available currently
       result name      myout       defined by user
       measurement type trig|delay|param|expr|avg|mean|max|min|rms|integ(ral)|when

       The measurement type determines how to continue the .meas card.
       param|expr are skipped in first pass through .meas cards and are treated in second pass,
       all others are treated in fcn get_measure2() (com_measure2.c).
       */

    /* first pass through .meas cards: evaluate everything except param|expr */
    for (meas_card = ft_curckt->ci_meas; meas_card != NULL; meas_card = meas_card->nextcard) {
        line = meas_card->line;

        line = nexttok(line); /* discard .meas */

        an_type = gettok(&line);
        resname = gettok(&line);
        meastype = gettok(&line);

        if (!an_type){
            fprintf(cp_err, "\nWarning: Incomplete measurement statement in line\n    %s\nignored!\n", meas_card->line);
            continue;
        }
        if (!resname){
            fprintf(cp_err, "\nWarning: Incomplete measurement statement in line\n    %s\nignored!\n", meas_card->line);
            tfree(an_type);
            continue;
        }
        if (!meastype) {
            fprintf(cp_err, "\nWarning: Incomplete measurement statement in line\n    %s\nignored!\n", meas_card->line);
            tfree(an_type);
            tfree(resname);
            continue;
        }

        if (chkAnalysisType(an_type) != TRUE) {
            if (!chk_only) {
                fprintf(cp_err, "Error: unrecognized analysis type '%s' for the following .meas statement on line %d:\n", an_type, meas_card->linenum);
                fprintf(cp_err, "       %s\n", meas_card->line);
            }

            txfree(an_type);
            txfree(resname);
            txfree(meastype);
            continue;
        }
        /* print header before evaluating first .meas line */
        else if (first_time) {
            first_time = FALSE;

            if (!chk_only && strcmp(an_type, "tran") == 0) {
                fprintf(stdout, "\n  Measurements for Transient Analysis\n\n");
                if (measout)
                    fprintf(measout, "\n  Measurements for Transient Analysis\n\n");
            }
            else if (!chk_only && strcmp(an_type, "dc") == 0) {
                fprintf(stdout, "\n  Measurements for DC Analysis\n\n");
                if (measout)
                    fprintf(measout, "\n  Measurements for DC Analysis\n\n");
            }
            else if (!chk_only && strcmp(an_type, "ac") == 0) {
                fprintf(stdout, "\n  Measurements for AC Analysis\n\n");
                if (measout)
                    fprintf(measout, "\n  Measurements for AC Analysis\n\n");
            }
            else if (!chk_only && strcmp(an_type, "sp") == 0) {
                fprintf(stdout, "\n  Measurements for SP Analysis\n\n");
                if (measout)
                    fprintf(measout, "\n  Measurements for SP Analysis\n\n");
            }
        }

        /* skip param|expr measurement types for now -- will be done after other measurements */
        if (strncmp(meastype, "param", 5) == 0 || strncmp(meastype, "expr", 4) == 0) {
            txfree(an_type);
            txfree(resname);
            txfree(meastype);
            continue;
        }

        /* skip .meas line, if analysis type from line and name of analysis performed differ */
        if (strcmp(an_name, an_type) != 0) {
            txfree(an_type);
            txfree(resname);
            txfree(meastype);
            continue;
        }

        /* New way of processing measure statements using common code
           in fcn get_measure2() (com_measure2.c)*/
        out_line[0] = '\0';
        measure_word_list = measure_parse_line(meas_card->line);
        if (measure_word_list) {
            fail = get_measure2(measure_word_list, &result, out_line, chk_only);
            if (fail) {
                measures_passed = FALSE;
                if (!chk_only)
                    fprintf(stderr, " %s failed!\n\n", meas_card->line);
                num_failed++;
                if (chk_only) {
                    /* added for speed - cleanup last parse and break */
                    txfree(an_type);
                    txfree(resname);
                    txfree(meastype);
                    wl_free(measure_word_list);
                    break;
                }
            } else {
                if (!chk_only)
                    nupa_add_param(resname, result);
            }
            wl_free(measure_word_list);
        } else {
            measures_passed = FALSE;
            num_failed++;
        }

        if (!chk_only) {
            newcard          = TMALLOC(struct card, 1);
            newcard->line = copy(out_line);
            newcard->nextcard = NULL;

            if (meas_results == NULL) {
                meas_results = end = newcard;
            } else {
                end->nextcard = newcard;
                end          = newcard;
            }
        }

        txfree(an_type);
        txfree(resname);
        txfree(meastype);

    } /* end of for loop (first pass through .meas lines) */

    if (chk_only) {
        tfree(an_name);
        return (measures_passed);
    }
    /* second pass through .meas cards: now do param|expr .meas statements */
    newcard = meas_results;
    for (meas_card = ft_curckt->ci_meas; meas_card != NULL; meas_card = meas_card->nextcard) {
        line = meas_card->line;

        line = nexttok(line); /* discard .meas */

        an_type = gettok(&line);
        resname = gettok(&line);
        meastype = gettok(&line);

        if (!an_type) {
            /* Warnings have already been issued in first pass */
            continue;
        }
        if (!resname) {
            tfree(an_type);
            continue;
        }
        if (!meastype) {
            tfree(an_type);
            tfree(resname);
            continue;
        }

        if (chkAnalysisType(an_type) != TRUE) {
            if (!chk_only) {
                fprintf(cp_err, "Error: unrecognized analysis type '%s' for the following .meas statement on line %d:\n", an_type, meas_card->linenum);
                fprintf(cp_err, "       %s\n", meas_card->line);
            }

            txfree(an_type);
            txfree(resname);
            txfree(meastype);
            continue;
        }
        if (strcmp(an_name, an_type) != 0) {
            txfree(an_type);
            txfree(resname);
            txfree(meastype);
            continue;
        }

        if (strncmp(meastype, "param", 5) != 0 && strncmp(meastype, "expr", 4) != 0) {

            if (!chk_only) {
                fprintf(stdout, "%s", newcard->line);
                if (measout)
                    fprintf(measout, "%s", newcard->line);
            }
            end     = newcard;
            newcard = newcard->nextcard;

            txfree(end->line);
            txfree(end);

            txfree(an_type);
            txfree(resname);
            txfree(meastype);
            continue;
        }

        if (!chk_only) {
            fprintf(stdout, "%-20s=", resname);
            if (measout)
                fprintf(measout, "%-20s=", resname);
        }

        if (!chk_only) {
            ok = nupa_eval(meas_card);

            if (ok) {
                str_ptr = strstr(meas_card->line, meastype);
                if (!get_double_value(&str_ptr, meastype, &result, chk_only)) {
                    if (!chk_only) {
                        fprintf(stdout, "   failed\n");
                        if (measout)
                            fprintf(measout, "   failed\n");
                    }
                } else {
                    if (!chk_only) {
                        fprintf(stdout, "  %.*e\n", precision, result);
                        if (measout)
                            fprintf(measout, "  %.*e\n", precision, result);
                    }
                    nupa_add_param(resname, result);
                }
            } else {
                if (!chk_only) {
                    fprintf(stdout, "   failed\n");
                    if (measout)
                        fprintf(measout, "   failed\n");
                }
            }
        }
        txfree(an_type);
        txfree(resname);
        txfree(meastype);
    }

    if (!chk_only) {
        fprintf(stdout, "\n");
        if (measout)
            fprintf(measout, "\n");
    }

    txfree(an_name);

    fflush(stdout);
    if (measout) {
        fclose(measout);
        measout = NULL;
    }
    return(measures_passed);
}


/* called from dctran.c:470, if timepoint is accepted.
   Returns TRUE if measurement (just a check, no output) has been successful.
   If TRUE is returned, transient simulation is stopped.
   Returns TRUE if "autostop" has been set as an option and if do_measure
   passes all tests and thereby returns TRUE.  'what' is set to "tran". */

bool
check_autostop(char* what)
{
    bool flag = FALSE;

    if (cp_getvar("autostop", CP_BOOL, NULL, 0))
        flag = do_measure(what, TRUE);

    return flag;
}


/* parses the .meas line into a wordlist (without leading .meas) */
static wordlist *
measure_parse_line(char *line)
{
    size_t len;                         /* length of string */
    wordlist *wl;                       /* build a word list - head of list */
    wordlist *new_item;                 /* single item of a list */
    char *item;                         /* parsed item */
    char *long_str;                     /* concatenated string */
    char *extra_item;                   /* extra item */

    wl = NULL;
    line = nexttok(line);
    do {
        item = gettok(&line);
        if (!(item))
            break;

        len = strlen(item);
        if (item[len-1] == '=') {
            /* We can't end on an equal append the next piece */
            extra_item = gettok(&line);
            if (!(extra_item))
                break;

            len += strlen(extra_item) + 2;
            long_str = TMALLOC(char, len);
            sprintf(long_str, "%s%s", item, extra_item);
            txfree(item);
            txfree(extra_item);
            item = long_str;
        }
        new_item = wl_cons(item, NULL);
        wl = wl_append(wl, new_item);
    } while (line && *line);

    return (wl);
}
