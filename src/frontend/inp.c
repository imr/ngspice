/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher
**********/

/*
 * Stuff for dealing with spice input decks and command scripts, and
 * the listing routines.
 */

#include "ngspice/ngspice.h"

#include "ngspice/cpdefs.h"
#include "ngspice/inpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/fteinp.h"
#include "inp.h"

#include "runcoms.h"
#include "inpcom.h"
#include "circuits.h"
#include "completion.h"
#include "variable.h"
#include "breakp2.h"
#include "../misc/util.h" /* ngdirname() */
#include "../misc/mktemp.h"
#include "../misc/misc_time.h"
#include "subckt.h"
#include "spiceif.h"
#include "com_let.h"

#ifdef XSPICE
/* include new stuff */
#include "ngspice/ipctiein.h"
#include "ngspice/enh.h"
/* */
#endif

#include "numparam/numpaif.h"

#define line_free(line,flag)	{ line_free_x(line,flag); line = NULL; }

/* static declarations */
static char * upper(register char *string);
static bool doedit(char *filename);
static struct line *com_options = NULL;

void line_free_x(struct line * deck, bool recurse);

#ifdef HAS_WINDOWS
void SetAnalyse( char * Analyse, int Percent);
#endif

/* Do a listing. Use is listing [expanded] [logical] [physical] [deck] */
void
com_listing(wordlist *wl)
{
    int type = LS_LOGICAL;
    bool expand = FALSE, do_param_listing = FALSE;
    char *s;

    if (ft_curckt) {  /* if there is a current circuit . . . .  */
        while (wl) {
            s = wl->wl_word;
            if ( strcmp( s, "param" ) == 0 ) {
                do_param_listing = TRUE;
            } else {
                switch (*s) {
                case 'l':
                case 'L':
                    type = LS_LOGICAL;
                    break;
                case 'p':
                case 'P':
                    type = LS_PHYSICAL;
                    break;
                case 'd':
                case 'D':
                    type = LS_DECK;
                    break;
                case 'e':
                case 'E':
                    expand = TRUE;
                    break;
                default:
                    fprintf(cp_err,
                            "Error: bad listing type %s\n", s);
                    return; /* SJB - don't go on after an error */
                }
            }
            wl = wl->wl_next;
        }
        if ( do_param_listing ) {
            nupa_list_params(cp_out);
        } else {
            if (type != LS_DECK)
                fprintf(cp_out, "\t%s\n\n", ft_curckt->ci_name);
            inp_list(cp_out, expand ? ft_curckt->ci_deck :
                     ft_curckt->ci_origdeck, ft_curckt->ci_options,
                     type);
        }
    } else
        fprintf(cp_err, "Error: no circuit loaded.\n");
    return;
}

/* returns inp_casefix() or NULL */
static char *
upper(char *string)
{
    static char buf[BSIZE_SP];

    if (string) {
        strncpy(buf, string, BSIZE_SP - 1);
        buf[BSIZE_SP - 1] = 0;
        inp_casefix(buf);
    } else {
        strcpy(buf, "<null>");
    }
    return buf;
}


/* Provide an input listing on the specified file of the given card
 * deck.  The listing should be of either LS_PHYSICAL or LS_LOGICAL or
 * LS_DECK lines as specified by the type parameter.  */
void
inp_list(FILE *file, struct line *deck, struct line *extras, int type)
{
    struct line *here;
    struct line *there;
    bool renumber;
    bool useout = (file == cp_out);
    int i = 1;
    /* gtri - wbk - 03/07/91 - Don't use 'more' type output if ipc enabled */
#ifdef XSPICE
    if(g_ipc.enabled) {
        useout = FALSE;
    }
#endif
    /* gtri - end - 03/07/91 */

    if (useout)
        out_init();
    renumber = cp_getvar("renumber", CP_BOOL, NULL);
    if (type == LS_LOGICAL) {
top1:
        for (here = deck; here; here = here->li_next) {
            if (renumber)
                here->li_linenum = i;
            if (ciprefix(".end", here->li_line) &&
                    !isalpha(here->li_line[4]))
                continue;
            if (*here->li_line != '*') {
                if (useout) {
                    sprintf(out_pbuf, "%6d : %s\n",
                            here->li_linenum,
                            upper(here->li_line));
                    out_send(out_pbuf);
                } else
                    fprintf(file, "%6d : %s\n",
                            here->li_linenum,
                            upper(here->li_line));
                if (here->li_error) {
                    if (useout) {
                        out_printf("%s\n", here->li_error);
                    } else
                        fprintf(file, "%s\n", here->li_error);
                }
            }
            i++;
        }
        if (extras) {
            deck = extras;
            extras = NULL;
            goto top1;
        }
        if (useout) {
            sprintf(out_pbuf, "%6d : .end\n", i);
            out_send(out_pbuf);
        } else
            fprintf(file, "%6d : .end\n", i);
    } else if ((type == LS_PHYSICAL) || (type == LS_DECK)) {
top2:
        for (here = deck; here; here = here->li_next) {
            if ((here->li_actual == NULL) || (here == deck)) {
                if (renumber)
                    here->li_linenum = i;
                if (ciprefix(".end", here->li_line) &&
                        !isalpha(here->li_line[4]))
                    continue;
                if (type == LS_PHYSICAL) {
                    if (useout) {
                        sprintf(out_pbuf, "%6d : %s\n",
                                here->li_linenum,
                                upper(here->li_line));
                        out_send(out_pbuf);
                    } else
                        fprintf(file, "%6d : %s\n",
                                here->li_linenum,
                                upper(here->li_line));
                } else {
                    if (useout)
                        out_printf("%s\n",
                                   upper(here->li_line));
                    else
                        fprintf(file, "%s\n",
                                upper(here->li_line));
                }
                if (here->li_error && (type == LS_PHYSICAL)) {
                    if (useout)
                        out_printf("%s\n",
                                   here->li_error);
                    else
                        fprintf(file, "%s\n",
                                here->li_error);
                }
            } else {
                for (there = here->li_actual; there;
                        there = there->li_next) {
                    there->li_linenum = i++;
                    if (ciprefix(".end", here->li_line) &&
                            isalpha(here->li_line[4]))
                        continue;
                    if (type == LS_PHYSICAL) {
                        if (useout) {
                            sprintf(out_pbuf, "%6d : %s\n",
                                    there->li_linenum,
                                    upper(there->li_line));
                            out_send(out_pbuf);
                        } else
                            fprintf(file, "%6d : %s\n",
                                    there->li_linenum,
                                    upper(there->li_line));
                    } else {
                        if (useout)
                            out_printf("%s\n",
                                       upper(there->li_line));
                        else
                            fprintf(file, "%s\n",
                                    upper(there->li_line));
                    }
                    if (there->li_error &&
                            (type == LS_PHYSICAL)) {
                        if (useout)
                            out_printf("%s\n",
                                       there->li_error);
                        else
                            fprintf(file, "%s\n",
                                    there->li_error);
                    }
                }
                here->li_linenum = i;
            }
            i++;
        }
        if (extras) {
            deck = extras;
            extras = NULL;
            goto top2;
        }
        if (type == LS_PHYSICAL) {
            if (useout) {
                sprintf(out_pbuf, "%6d : .end\n", i);
                out_send(out_pbuf);
            } else
                fprintf(file, "%6d : .end\n", i);
        } else {
            if (useout)
                out_printf(".end\n");
            else
                fprintf(file, ".end\n");
        }
    } else
        fprintf(cp_err, "inp_list: Internal Error: bad type %d\n",
                type);
    return;
}

/*
 * Free memory used by a line.
 * If recurse is TRUE then recursively free all lines linked via the li_next field.
 * If recurse is FALSE free only this line.
 * All lines linked via the li_actual field are always recursivly freed.
 * SJB - 22nd May 2001
 */
void
line_free_x(struct line * deck, bool recurse)
{
    if(!deck)
        return;
    tfree(deck->li_line);
    tfree(deck->li_error);
    if(recurse)
        line_free(deck->li_next,TRUE);
    line_free(deck->li_actual,TRUE);
    tfree(deck);
}


/* The routine to source a spice input deck. We read the deck in, take
 * out the front-end commands, and create a CKT structure. Also we
 * filter out the following cards: .save, .width, .four, .print, and
 * .plot, to perform after the run is over.
 * Then, we run dodeck, which parses up the deck.             */
void
inp_spsource(FILE *fp, bool comfile, char *filename)
/* arguments:
 *  *fp = pointer to the input file
 *  comfile = whether it is a command file.  Values are TRUE/FALSE
 *  *filename = name of input file
 */
{
    struct line *deck, *dd, *ld, *prev_param = NULL, *prev_card = NULL;
    struct line *realdeck = NULL, *options = NULL, *curr_meas = NULL;
    char *tt = NULL, name[BSIZE_SP], *s, *t, *temperature = NULL;
    double testemp = 0.0;
    bool commands = FALSE;
    wordlist *wl = NULL, *end = NULL, *wl_first = NULL;
    wordlist *controls = NULL;
    FILE *lastin, *lastout, *lasterr;
    double temperature_value;

    double startTime, endTime;

    /* read in the deck from a file */
    char *filename_dup = ( filename == NULL ) ? strdup(".") : strdup(filename);

    startTime = seconds();
    inp_readall(fp, &deck, 0, ngdirname(filename_dup), comfile);
    endTime = seconds();
    tfree(filename_dup);

    /* if nothing came back from inp_readall, just close fp and return to caller */
    if (!deck) {	/* MW. We must close fp always when returning */
        fclose(fp);
        return;
    }

    if (!comfile) {
        options = inp_getopts(deck);

        realdeck = inp_deckcopy(deck);

        /* Save the title before INPgetTitle gets it. */
        tt = copy(deck->li_line);
        if (!deck->li_next)
            fprintf(cp_err, "Warning: no lines in input\n");
    }
    fclose(fp);

    /* Now save the IO context and start a new control set.  After we
       are done with the source we'll put the old file descriptors
       back.  I guess we could use a FILE stack, but since this
       routine is recursive anyway.  */
    lastin = cp_curin;
    lastout = cp_curout;
    lasterr = cp_curerr;
    cp_curin = cp_in;
    cp_curout = cp_out;
    cp_curerr = cp_err;

    cp_pushcontrol();

    /* We should now go through the deck and execute front-end
     * commands and remove them. Front-end commands are enclosed by
     * the cards .control and .endc, unless comfile is TRUE, in which
     * case every line must be a front-end command.  There are too
     * many problems with matching the first word on the line.  */
    ld = deck;
    if (comfile) {
        /* Process each command, except 'option' which is assembled
           in a list and ingnored here */
        for (dd = deck; dd; dd = ld) {
            ld = dd->li_next;
            if ((dd->li_line[0] == '*') && (dd->li_line[1] != '#'))
                continue;
            if (!ciprefix(".control", dd->li_line) && !ciprefix(".endc", dd->li_line)) {
                if (dd->li_line[0] == '*')
                    cp_evloop(dd->li_line + 2);
                /* option line stored but not processed */
                else if (ciprefix("option", dd->li_line)) {
                    com_options = inp_getoptsc(dd->li_line, com_options);
                } else
                    cp_evloop(dd->li_line);
            }
        }
        /* free the control deck */
        line_free(deck,TRUE);
        /*         printf("Command deck freed\n"); */
    } /* end if(comfile) */

    else {    /* must be regular deck . . . . */
        /* loop through deck and handle control cards */
        for (dd = deck->li_next; dd; dd = ld->li_next) {
            /* get temp from deck */
            if ( ciprefix(".temp", dd->li_line) ) {
                s = dd->li_line + 5;
                while ( isspace(*s) ) s++;
                if ( temperature != NULL ) {
                    txfree(temperature);
                }
                temperature = strdup(s);
            }
            /* Ignore comment lines, but not lines begining with '*#',
               but remove them, if they are in a .control ... .endc section */
            s = dd->li_line;
            while(isspace(*s)) s++;
            if ( (*s == '*') && ( (s != dd->li_line) || (s[1] != '#'))) {
                if (commands) {
                    /* Remove comment lines in control sections, so they  don't
                    * get considered as circuits.  */
                    ld->li_next = dd->li_next;
                    line_free(dd,FALSE);
                    continue;
                }
                ld = dd;
                continue;
            }

            /* Put the first token from line into s */
            strncpy(name, dd->li_line, BSIZE_SP);
            for (s = name; *s && isspace(*s); s++)
                ;
            for (t = s; *t && !isspace(*t); t++)
                ;
            *t = '\0';

            if (ciprefix(".control", dd->li_line)) {
                ld->li_next = dd->li_next;
                line_free(dd,FALSE); /* SJB - free this line's memory */
                if (commands)
                    fprintf(cp_err, "Warning: redundant .control card\n");
                else
                    commands = TRUE;
            } else if (ciprefix(".endc", dd->li_line)) {
                ld->li_next = dd->li_next;
                line_free(dd,FALSE); /* SJB - free this line's memory */
                if (commands)
                    commands = FALSE;
                else
                    fprintf(cp_err, "Warning: misplaced .endc card\n");
            } else if (commands || prefix("*#", dd->li_line)) {
                wl = alloc(struct wordlist);
                if (controls) {
                    wl->wl_next = controls;
                    controls->wl_prev = wl;
                    controls = wl;
                } else
                    controls = wl;
                /* more control lines */
                if (prefix("*#", dd->li_line))
                    wl->wl_word = copy(dd->li_line + 2);
                else {
                    wl->wl_word = dd->li_line;
                    dd->li_line = 0; /* SJB - prevent line_free() freeing the string (now pointed at by wl->wl_word) */
                }
                /* Look for set or unset numparams.
                   If either are found then we evaluate these lines immediately
                   so they take effect before netlist parsing */
                s = wl->wl_word;
                while(isspace(*s)) s++;	/* step past any white space */
                if(ciprefix("set", s)) {
                    s+=3;
                } else if(ciprefix("unset", s)) {
                    s+=5;
                }
                if(s!=dd->li_line) {	/* one of the above must have matched */
                    while(isspace(*s)) s++;	/* step past white space */
                    if(ciprefix("numparams", s)) {
                        cp_evloop(wl->wl_word);
                    }
                }
                ld->li_next = dd->li_next;
                line_free(dd,FALSE);
            } else if (!*dd->li_line) {
                /* So blank lines in com files don't get considered as
                 * circuits.  */
                ld->li_next = dd->li_next;
                line_free(dd,FALSE);
            } else {
                /* lines .width, .four, .plot, .print,. save added to wl_first, removed from deck */
                /* lines .op, .meas, .tf added to wl_first */
                inp_casefix(s); /* s: first token from line */
                inp_casefix(dd->li_line);
                if (eq(s, ".width")
                        || ciprefix(".four", s)
                        || eq(s, ".plot")
                        || eq(s, ".print")
                        || eq(s, ".save")
                        || eq(s, ".op")
                        || ciprefix(".meas", s)
                        || eq(s, ".tf")) {
                    if (end) {
                        end->wl_next = alloc(struct wordlist);
                        end->wl_next->wl_prev = end;
                        end = end->wl_next;
                    } else
                        wl_first = end = alloc(struct wordlist);
                    end->wl_word = copy(dd->li_line);

                    if (!eq(s, ".op") && !eq(s, ".tf") && !ciprefix(".meas", s)) {
                        ld->li_next = dd->li_next;
                        line_free(dd,FALSE);
                    } else
                        ld = dd;
                } else
                    ld = dd;
            }
        }  /* end for(dd=deck->li_next . . . .  */

        /* set temperature if defined to a preliminary variable which may be used
           in numparam evaluation */
        if ( temperature != NULL ) {
            temperature_value = atof(temperature);
            cp_vset("pretemp", CP_REAL, &temperature_value );
        }
        if (ft_ngdebug) {
            cp_getvar("pretemp", CP_REAL, &testemp);
            printf("test temperature %f\n", testemp);
        }
        /* We are done handling the control stuff.  Now process remainder of deck.
           Go on if there is something left after the controls.*/
        if (deck->li_next) {
            fprintf(cp_out, "\nCircuit: %s\n\n", tt);
#ifdef HAS_WINDOWS
            SetAnalyse( "Prepare Deck", 0);
#endif
            /* Now expand subcircuit macros and substitute numparams.*/
            if (!cp_getvar("nosubckt", CP_BOOL, NULL))
                if( (deck->li_next = inp_subcktexpand(deck->li_next)) == NULL ) {
                    line_free(realdeck,TRUE);
                    line_free(deck->li_actual, TRUE);
                    return;
                }

            /* Now handle translation of spice2c6 POLYs. */
#ifdef XSPICE
            /* Translate all SPICE 2G6 polynomial type sources */
            deck->li_next = ENHtranslate_poly(deck->li_next);

#endif

            line_free(deck->li_actual,FALSE);
            deck->li_actual = realdeck;

            /* print out the expanded deck into debug-out2.txt */
            if (ft_ngdebug) {
                FILE *fdo;
                struct line *tmp_ptr1 = NULL;
                /*debug: print into file*/
                fdo = fopen("debug-out2.txt", "w");
                for(tmp_ptr1 = deck; tmp_ptr1 != NULL; tmp_ptr1 = tmp_ptr1->li_next)
                    fprintf(fdo, "%s\n", tmp_ptr1->li_line);
                ;
                (void) fclose(fdo);
            }
            for(dd = deck; dd != NULL; dd = dd->li_next) {
                /* get csparams and create vectors */
                if ( ciprefix(".csparam", dd->li_line) ) {
                    wordlist *wlist = NULL;
                    wordlist *wl = NULL;
                    wordlist *cwl;
                    char *cstoken[3];
                    int i;
                    s = dd->li_line;
                    *s='*';
                    s = dd->li_line + 8;
                    while ( isspace(*s) ) s++;
                    cstoken[0]=gettok_char(&s, '=', FALSE);                   
                    cstoken[1]=gettok_char(&s, '=', TRUE);
                    cstoken[2]=gettok(&s);
                    for (i=0; i<3;i++) {
                        cwl = alloc(struct wordlist);
                        cwl->wl_prev = wl;
                        if (wl)
                            wl->wl_next = cwl;
                        else {
                            wlist = cwl;
                            cwl->wl_next = NULL;
                        }
                        cwl->wl_word = cstoken[i];
                        wl = cwl;
                    }
                    com_let(wlist);
                    wl_free(wlist);
                }
            }

            /*merge the two option line structs*/
            if (!options && com_options)
                options = com_options;
            else if (options && com_options) {
                /* move to end of options
                    struct line* tmp_options = options;
                    while (tmp_options) {
                       if (!tmp_options->li_next) break;
                       tmp_options = tmp_options->li_next;
                    }
                    tmp_options->li_next = com_options;*/
                /* move to end of com_options */
                struct line* tmp_options = com_options;
                while (tmp_options) {
                    if (!tmp_options->li_next) break;
                    tmp_options = tmp_options->li_next;
                }
                tmp_options->li_next = options;
            }

            /* now load deck into ft_curckt -- the current circuit. */
            inp_dodeck(deck, tt, wl_first, FALSE, options, filename);

        }     /*  if (deck->li_next) */

        /* look for and set temperature; also store param and .meas statements in circuit struct */
        if (ft_curckt) {
            ft_curckt->ci_param = NULL;
            ft_curckt->ci_meas  = NULL;
            /* PN add here stats*/
            ft_curckt->FTEstats->FTESTATnetLoadTime = endTime - startTime;
        }

        for (dd = deck; dd; dd = dd->li_next) {
            /* all parameter lines should be sequentially ordered and placed at
             beginning of deck */
            if ( ciprefix( ".param", dd->li_line ) ) {
                ft_curckt->ci_param = dd;
                /* find end of .param statements */
                while ( ciprefix( ".param", dd->li_line ) ) {
                    prev_param = dd;
                    dd = dd->li_next;
                    if (dd == NULL) break; // no line after .param line
                }
                prev_card->li_next  = dd;
                prev_param->li_next = NULL;
                if (dd == NULL) {
                    fprintf(cp_err, "Warning: Missing .end card!\n");
                    break; // no line after .param line
                }
            }

            if ( ciprefix( ".meas", dd->li_line ) ) {
                if ( cp_getvar( "autostop", CP_BOOL, NULL) ) {
                    if ( strstr( dd->li_line, " max " ) || strstr( dd->li_line, " min " ) || strstr( dd->li_line, " avg " ) ||
                            strstr( dd->li_line, " rms " ) || strstr( dd->li_line, " integ " ) ) {
                        printf( "Warning: .OPTION AUTOSTOP will not be effective because one of 'max|min|avg|rms|integ' is used in .meas\n" );
                        printf( "         AUTOSTOP being disabled...\n" );
                        cp_remvar( "autostop" );
                    }
                }

                if ( curr_meas == NULL ) {
                    curr_meas = ft_curckt->ci_meas = dd;
                } else {
                    curr_meas->li_next = dd;
                    curr_meas = dd;
                }
                prev_card->li_next = dd->li_next;
                curr_meas->li_next = NULL;
                dd                 = prev_card;
            }
            prev_card = dd;
        }  //end of for-loop

        /* set temperature, if defined, to new value.
           cp_vset will set the variable "temp" and also set CKTtemp,
           so we can do it only here because the circuit has to be already there */
        if ( temperature != NULL ) {
            temperature_value = atof(temperature);
            cp_vset("temp", CP_REAL, &temperature_value );
            txfree(temperature);
        }

#ifdef TRACE
        /* SDB debug statement */
        printf("In inp_spsource, done with dodeck.\n");
#endif

        /* print out the expanded deck into debug-out3.txt */
        if (ft_ngdebug) {
            FILE *fdo;
            struct line *tmp_ptr1 = NULL;
            /*debug: print into file*/
            fdo = fopen("debug-out3.txt", "w");
            for(tmp_ptr1 = deck; tmp_ptr1 != NULL; tmp_ptr1 = tmp_ptr1->li_next)
                fprintf(fdo, "%s\n", tmp_ptr1->li_line);
            ;
            (void) fclose(fdo);
        }

        /* Now that the deck is loaded, do the commands, if there are any */
        if (controls) {
            for (end = wl = wl_reverse(controls); wl; wl = wl->wl_next)
                cp_evloop(wl->wl_word);
            wl_free(end);
        }
    }

    /* linked list dbs is used to store the "save" or .save data (defined in breakp2.c),
       breakpoint and iplot data, will be renewed in ft_dotsaves(). */
    if(dbs) tfree(dbs);

    /*saj, to process save commands always, not just in batch mode
      *(breaks encapsulation of frontend and parsing commands slightly)*/
    ft_dotsaves();

    /* Now reset everything.  Pop the control stack, and fix up the IO
     * as it was before the source.  */
    cp_popcontrol();

    cp_curin = lastin;
    cp_curout = lastout;
    cp_curerr = lasterr;

    return;
}


/* This routine is cut in half here because com_rset has to do what
 * follows also. End is the list of commands we execute when the job
 * is finished: we only bother with this if we might be running in
 * batch mode, since it isn't much use otherwise.  */
/*------------------------------------------------------------------
 * It appears that inp_dodeck adds the circuit described by *deck
 * to the current circuit (ft_curckt).
 *-----------------------------------------------------------------*/
void
inp_dodeck(
    struct line *deck,     /*in: the spice deck */
    char *tt,              /*in: the title of the deck */
    wordlist *end,         /*in: all lines with .width, .plot, .print, .save, .op, .meas, .tf */
    bool reuse,            /*in: TRUE if called from runcoms2.c com_rset,
                               FALSE if called from inp_spsource() */
    struct line *options,  /*in: all .option lines from deck */
    char *filename         /*in: input file of deck */
)
{
    struct circ *ct;
    struct line *dd;
    CKTcircuit *ckt;
    char *s;
    INPtables *tab = NULL;
    struct variable *eev = NULL;
    wordlist *wl;
    bool noparse, ii;
    int print_listing;

    double startTime;

    /* First throw away any old error messages there might be and fix
       the case of the lines.  */
    for (dd = deck; dd; dd = dd->li_next) {
        if (dd->li_error) {
            tfree(dd->li_error);
            dd->li_error = NULL;
        }
    }
    if (reuse) {
        ct = ft_curckt;
    } else {
        if (ft_curckt) {
            ft_curckt->ci_devices = cp_kwswitch(CT_DEVNAMES,
                                                NULL);
            ft_curckt->ci_nodes = cp_kwswitch(CT_NODENAMES,
                                              NULL);
        }
        ft_curckt = ct = alloc(struct circ);

        /*PN FTESTATS*/
        ft_curckt->FTEstats = TMALLOC(FTESTATistics, 1);
    }
    noparse = cp_getvar("noparse", CP_BOOL, NULL);


    /* We check preliminary for the scale option. This special processing
       is needed because we need the scale info BEFORE building the circuit
       and seems there is no other way to do this. */
    if (!noparse) {
        struct line* opt_beg = options;
        for (; options; options = options->li_next) {
            for (s = options->li_line; *s && !isspace(*s); s++)
                ;

            ii = cp_interactive;
            cp_interactive = FALSE;
            wl = cp_lexer(s);
            cp_interactive = ii;
            if (!wl || !wl->wl_word || !*wl->wl_word)
                continue;
            if (eev)
                eev->va_next = cp_setparse(wl);
            else
                ct->ci_vars = eev = cp_setparse(wl);
            wl_free(wl);
            while (eev->va_next)
                eev = eev->va_next;
        }
        for (eev = ct->ci_vars; eev; eev = eev->va_next) {
            switch (eev->va_type) {
            case CP_BOOL:
                break;
            case CP_NUM:
                break;
            case CP_REAL:
                if ( strcmp("scale",eev->va_name)==0 ) {
                    cp_vset("scale", CP_REAL, &eev->va_real );
                    printf("Scale set\n");
                }
                break;
            case CP_STRING:
                break;
            default: {
                fprintf(stderr, "ERROR: enumeration value `CP_LIST' not handled in inp_dodeck\nAborting...\n" );
                controlled_exit(EXIT_FAILURE);
            }
            } /* switch  . . . */
        }
        options = opt_beg; // back to the beginning
    } /* if (!noparse)  . . . */

    /*----------------------------------------------------
     * Now assuming that we wanna parse this deck, we call
     * if_inpdeck which takes the deck and returns a
     * a pointer to the circuit ckt.
     *---------------------------------------------------*/
    if (!noparse) {
        startTime = seconds();
        ckt = if_inpdeck(deck, &tab);
        ft_curckt->FTEstats->FTESTATnetParseTime = seconds() - startTime;
    } else
        ckt = NULL;

    out_init();

    ft_curckt->FTEstats->FTESTATdeckNumLines = 0;
    /*----------------------------------------------------
     * Now run through the deck and look to see if there are
     * errors on any line.
     *---------------------------------------------------*/
    for (dd = deck; dd; dd = dd->li_next) {

        ft_curckt->FTEstats->FTESTATdeckNumLines += 1;

#ifdef TRACE
        /* SDB debug statement */
        printf("In inp_dodeck, looking for errors and examining line %s . . . \n", dd->li_line);
#endif

        if (dd->li_error) {
            char *p, *q;

#ifdef XSPICE
            /* add setting of ipc syntax error flag */
            g_ipc.syntax_error = IPC_TRUE;
#endif
            p = dd->li_error;
            do {
                q =strchr(p, '\n');
                if (q)
                    *q = 0;

                if (p == dd->li_error) {
                    if (strstr(dd->li_line, ".model"))
                        out_printf("Model issue on line %d : %.*s ...\n%s\n",
                                   dd->li_linenum_orig, 56, dd->li_line, dd->li_error);
                    else
                        out_printf("Error on line %d : %s\n%s\n",
                                   dd->li_linenum_orig, dd->li_line, dd->li_error);
                    if (ft_stricterror)
                        controlled_exit(EXIT_BAD);
                }
                else
                    out_printf("%s\n", p);

                if (q)
                    *q++ = '\n';
                p = q;
            } while (p && *p);
        }  /* end  if (dd->li_error) */

    }   /* for (dd = deck; dd; dd = dd->li_next) */

    /* Only print out netlist if brief is FALSE */

    if(!cp_getvar( "brief", CP_BOOL, NULL )) {
        /* output deck */
        out_printf( "\nProcessed Netlist\n" );
        out_printf( "=================\n" );
        print_listing = 1;
        for (dd = deck; dd; dd = dd->li_next) {
            if ( ciprefix(".prot", dd->li_line) ) print_listing = 0;
            if ( print_listing == 1 ) out_printf( "%s\n", dd->li_line );
            if ( ciprefix(".unprot", dd->li_line) ) print_listing = 1;
        }
        out_printf( "\n" );
    }

    /* Add this circuit to the circuit list. If reuse is TRUE then use
     * the ft_curckt structure.  */
    if (!reuse) {
        /* Be sure that ci_devices and ci_nodes are valid */
        ft_curckt->ci_devices = cp_kwswitch(CT_DEVNAMES,
                                            NULL);
        cp_kwswitch(CT_DEVNAMES, ft_curckt->ci_devices);
        ft_curckt->ci_nodes = cp_kwswitch(CT_NODENAMES, NULL);
        cp_kwswitch(CT_NODENAMES, ft_curckt->ci_nodes);
        ft_newcirc(ct);
        /* Assign current circuit */
        ft_curckt = ct;
    }
    ct->ci_name = tt;
    ct->ci_deck = deck;
    ct->ci_options = options;
    if (deck->li_actual)
        ct->ci_origdeck = deck->li_actual;
    else
        ct->ci_origdeck = ct->ci_deck;
    ct->ci_ckt = ckt;             /* attach the input ckt to the list of circuits */
    ct->ci_symtab = tab;
    ct->ci_inprogress = FALSE;
    ct->ci_runonce = FALSE;
    ct->ci_commands = end;
    if (filename)
        ct->ci_filename = copy(filename);
    else
        ct->ci_filename = NULL;

    if (!noparse) {
        /*        for (; options; options = options->li_next) {
                    for (s = options->li_line; *s && !isspace(*s); s++)
                        ;

                    ii = cp_interactive;
                    cp_interactive = FALSE;
                    wl = cp_lexer(s);
                    cp_interactive = ii;
                    if (!wl || !wl->wl_word || !*wl->wl_word)
                        continue;
                    if (eev)
                        eev->va_next = cp_setparse(wl);
                    else
                        ct->ci_vars = eev = cp_setparse(wl);
                    while (eev->va_next)
                        eev = eev->va_next;
                }
        */
        for (eev = ct->ci_vars; eev; eev = eev->va_next) {
            bool one = TRUE;   /* FIXME, actually eev->va_bool should be TRUE anyway */
            switch (eev->va_type) {
            case CP_BOOL:
                if_option(ct->ci_ckt, eev->va_name,
                          eev->va_type, &one);
                break;
            case CP_NUM:
                if_option(ct->ci_ckt, eev->va_name,
                          eev->va_type, &eev->va_num);
                break;
            case CP_REAL:
                if_option(ct->ci_ckt, eev->va_name,
                          eev->va_type, &eev->va_real);
                break;
            case CP_STRING:
                if_option(ct->ci_ckt, eev->va_name,
                          eev->va_type, eev->va_string);
                break;
            default: {
                fprintf(stderr, "ERROR: enumeration value `CP_LIST' not handled in inp_dodeck\nAborting...\n" );
                controlled_exit(EXIT_FAILURE);
            }
            } // switch  . . .
        }
    } // if (!noparse)  . . .

    /* add title of deck to data base */
    cp_addkword(CT_CKTNAMES, tt);
    return;
}


/* Edit and re-load the current input deck.  Note that if these
 * commands are used on a non-unix machine, they will leave spice.tmp
 * junk files lying around.  */
void
com_edit(wordlist *wl)
{
    char *filename;
    FILE *fp;
    bool inter, permfile;
    char buf[BSIZE_SP];

    inter = cp_interactive;
    cp_interactive = FALSE;
    if (wl) {
        if (!doedit(wl->wl_word)) {
            cp_interactive = inter;
            return;
        }
        if ((fp = inp_pathopen(wl->wl_word, "r")) == NULL) {
            perror(wl->wl_word);
            cp_interactive = inter;
            return;
        }
        inp_spsource(fp, FALSE, wl->wl_word);
    } else {
        /* If there is no circuit yet, then create one */
        if (ft_curckt && ft_curckt->ci_filename) {
            filename = ft_curckt->ci_filename;
            permfile = TRUE;
        } else {
            filename = smktemp("sp");
            permfile = FALSE;
        }
        if (ft_curckt && !ft_curckt->ci_filename) {
            if ((fp = fopen(filename, "w")) == NULL) {
                perror(filename);
                cp_interactive = inter;
                return;
            }
            inp_list(fp, ft_curckt->ci_deck, ft_curckt->ci_options,
                     LS_DECK);
            fprintf(cp_err,
                    "Warning: editing a temporary file -- "
                    "circuit not saved\n");
            fclose(fp);
        } else if (!ft_curckt) {
            if ((fp = fopen(filename, "w")) == NULL) {
                perror(filename);
                cp_interactive = inter;
                return;
            }
            fprintf(fp, "SPICE 3 test deck\n");
            fclose(fp);
        }
        if (!doedit(filename)) {
            cp_interactive = inter;
            return;
        }

        if ((fp = fopen(filename, "r")) == NULL) {
            perror(filename);
            cp_interactive = inter;
            return;
        }
        inp_spsource(fp, FALSE, permfile ? filename : NULL);

        /* fclose(fp);  */
        /*	MW. inp_spsource already closed fp */

        if (ft_curckt && !ft_curckt->ci_filename)
            unlink(filename);
    }

    cp_interactive = inter;

    /* note: default is to run circuit after successful edit */

    fprintf(cp_out, "run circuit? ");
    fflush(cp_out);
    fgets(buf, BSIZE_SP, stdin);
    if (buf[0] != 'n') {
        fprintf(cp_out, "running circuit\n");
        com_run(NULL);
    }

    return;
}

static bool
doedit(char *filename)
{
    char buf[BSIZE_SP], buf2[BSIZE_SP], *editor;

    if (cp_getvar("editor", CP_STRING, buf2)) {
        editor = buf2;
    } else {
        if ((editor = getenv("EDITOR")) == NULL) {
            if (Def_Editor && *Def_Editor)
                editor = Def_Editor;
            else
                editor = "/usr/bin/vi";
        }
    }
    sprintf(buf, "%s %s", editor, filename);
    return (system(buf) ? FALSE : TRUE);
}

void
com_source(wordlist *wl)
{
    FILE *fp, *tp;
    char buf[BSIZE_SP];
    bool inter;
    char *tempfile = NULL;

    wordlist *owl = wl;
    size_t n;

    inter = cp_interactive;
    cp_interactive = FALSE;
    if (wl->wl_next) {
        /* There are several files -- put them into a temp file  */
        tempfile = smktemp("sp");
        if ((fp = inp_pathopen(tempfile, "w+")) == NULL) {
            perror(tempfile);
            cp_interactive = TRUE;
            return;
        }
        while (wl) {
            if ((tp = inp_pathopen(wl->wl_word, "r")) == NULL) {
                perror(wl->wl_word);
                fclose(fp);
                cp_interactive = TRUE;
                unlink(tempfile);
                return;
            }
            while ((n = fread(buf, 1, BSIZE_SP, tp)) > 0)
                fwrite(buf, 1, n, fp);
            fclose(tp);
            wl = wl->wl_next;
        }
        fseek(fp, 0L, SEEK_SET);
    } else
        fp = inp_pathopen(wl->wl_word, "r");
    if (fp == NULL) {
        perror(wl->wl_word);
        cp_interactive = TRUE;
        return;
    }

    /* Don't print the title if this is a spice initialisation file. */
    if (ft_nutmeg || substring(INITSTR, owl->wl_word) || substring(ALT_INITSTR, owl->wl_word)) {
        inp_spsource(fp, TRUE, tempfile ? NULL : wl->wl_word);
    } else {
        inp_spsource(fp, FALSE, tempfile ? NULL : wl->wl_word);
    }
    cp_interactive = inter;
    if (tempfile)
        unlink(tempfile);
    return;
}

void
inp_source(char *file)
{
    static struct wordlist wl = { NULL, NULL, NULL } ;
    wl.wl_word = file;
    com_source(&wl);
    return;
}
