/**********
Copyright 2023 The ngspice team.  All rights reserved.
License: Three-clause BCD
Author: 2023 Holger Vogt
**********/

/*
  For dealing with compatibility transformations

  PSPICE, LTSPICE and others
*/

#include "ngspice/ngspice.h"

#include "ngspice/compatmode.h"
#include "ngspice/cpdefs.h"
#include "ngspice/dstring.h"
#include "ngspice/dvec.h"
#include "ngspice/ftedefs.h"
#include "ngspice/fteext.h"
#include "ngspice/fteinp.h"
#include "numparam/general.h"

#include <limits.h>
#include <stdlib.h>

#include <sys/stat.h>
#include <sys/types.h>

#if !defined(__MINGW32__) && !defined(_MSC_VER)
#include <unistd.h>
#endif

#include "../misc/util.h" /* ngdirname() */
#include "inpcom.h"
#include "ngspice/stringskip.h"
#include "ngspice/stringutil.h"
#include "ngspice/wordlist.h"
#include "subckt.h"
#include "variable.h"

#define INTEGRATE_UDEVICES
#ifdef INTEGRATE_UDEVICES
#include "ngspice/udevices.h"
#endif

void print_compat_mode(void);
void set_compat_mode(void);
struct card* pspice_compat(struct card* newcard);
void pspice_compat_a(struct card* oldcard);
struct card* ltspice_compat(struct card* oldcard);
void ltspice_compat_a(struct card* oldcard);



/* Set a compatibility flag.
Currently available are flags for:
- LTSPICE, HSPICE, Spice3, PSPICE, KiCad, Spectre, XSPICE
*/
struct compat newcompat;
void set_compat_mode(void)
{
    char behaviour[80];
    newcompat.hs = FALSE;
    newcompat.ps = FALSE;
    newcompat.xs = FALSE;
    newcompat.lt = FALSE;
    newcompat.ki = FALSE;
    newcompat.a = FALSE;
    newcompat.spe = FALSE;
    newcompat.isset = FALSE;
    newcompat.s3 = FALSE;
    newcompat.mc = FALSE;
    if (cp_getvar("ngbehavior", CP_STRING, behaviour, sizeof(behaviour))) {
        if (strstr(behaviour, "hs"))
            newcompat.isset = newcompat.hs = TRUE; /*HSPICE*/
        if (strstr(behaviour, "ps"))
            newcompat.isset = newcompat.ps = TRUE; /*PSPICE*/
        if (strstr(behaviour, "xs"))
            newcompat.isset = newcompat.xs = TRUE; /*XSPICE*/
        if (strstr(behaviour, "lt"))
            newcompat.isset = newcompat.lt = TRUE; /*LTSPICE*/
        if (strstr(behaviour, "ki"))
            newcompat.isset = newcompat.ki = TRUE; /*KiCad*/
        if (strstr(behaviour, "a"))
            newcompat.isset = newcompat.a = TRUE; /*complete netlist, used in conjuntion with other mode*/
        if (strstr(behaviour, "ll"))
            newcompat.isset = newcompat.ll = TRUE; /*all (currently not used)*/
        if (strstr(behaviour, "s3"))
            newcompat.isset = newcompat.s3 = TRUE; /*spice3 only*/
        if (strstr(behaviour, "eg"))
            newcompat.isset = newcompat.eg = TRUE; /*EAGLE*/
        if (strstr(behaviour, "spe")) {
            newcompat.isset = newcompat.spe = TRUE; /*Spectre*/
            newcompat.ps = newcompat.lt = newcompat.ki = newcompat.eg = FALSE;
        }
        if (strstr(behaviour, "mc")) {
            newcompat.isset = FALSE;
            newcompat.mc = TRUE; /*make check*/
        }
    }
    if (newcompat.hs && newcompat.ps) {
        fprintf(stderr, "Warning: hs and ps compatibility are mutually exclusive, switch to ps!\n");
        newcompat.hs = FALSE;
    }
    /* reset everything for 'make check' */
    if (newcompat.mc)
        newcompat.eg = newcompat.hs = newcompat.spe = newcompat.ps = newcompat.xs =
        newcompat.ll = newcompat.lt = newcompat.ki = newcompat.a = FALSE;
}

/* Print the compatibility flags */
void print_compat_mode(void) {
    if (newcompat.mc) /* make check */
        return;
    if (newcompat.isset) {
        fprintf(stdout, "\n");
        fprintf(stdout, "Note: Compatibility modes selected:");
        if (newcompat.hs)
            fprintf(stdout, " hs");
        if (newcompat.ps)
            fprintf(stdout, " ps");
        if (newcompat.xs)
            fprintf(stdout, " xs");
        if (newcompat.lt)
            fprintf(stdout, " lt");
        if (newcompat.ki)
            fprintf(stdout, " ki");
        if (newcompat.ll)
            fprintf(stdout, " ll");
        if (newcompat.s3)
            fprintf(stdout, " s3");
        if (newcompat.eg)
            fprintf(stdout, " eg");
        if (newcompat.spe)
            fprintf(stdout, " spe");
        if (newcompat.a)
            fprintf(stdout, " a");
        fprintf(stdout, "\n\n");
    }
    else {
        fprintf(stdout, "\n");
        fprintf(stdout, "Note: No compatibility mode selected!\n\n");
    }
}


/* replace the E and G source TABLE function by a B source pwl
 * (used by ST OpAmps and comparators of Infineon models).
 * E_RO_3 VB_3 VB_4  VALUE={ TABLE( V(VCCP,VCCN), 2 , 35 , 3.3 , 15 , 5 , 10
 *         )*I(VreadIo)}
 * will become
 * BE_RO_3_1 TABLE_NEW_1 0 v = pwl( V(VCCP,VCCN), 2 , 35 , 3.3 , 15 , 5 , 10) 
 * E_RO_3 VB_3 VB_4  VALUE={ V(TABLE_NEW_1)*I(VreadIo)}
 */
static void replace_table(struct card *startcard)
{
    struct card *card;
    static int numb = 0;
    for (card = startcard; card; card = card->nextcard) {
        char *cut_line = card->line;
        if (*cut_line == 'e' || *cut_line == 'g') {
            char *valp = search_plain_identifier(cut_line, "value");
            char *valp2 = search_plain_identifier(cut_line, "cur");
            if (valp || (valp2 && *cut_line == 'g')) {
                char *ftablebeg = strstr(cut_line, "table(");
                while (ftablebeg) {
                    /* get the beginning of the line */
                    char *begline = copy_substring(cut_line, ftablebeg);
                    /* get the table function */
                    char *tabfun = gettok_char(&ftablebeg, ')', TRUE, TRUE);
                    /* the new e, g line */
                    char *neweline = tprintf("%s v(table_new_%d)%s",
                            begline, numb, ftablebeg);
                    char *newbline =
                            tprintf("btable_new_%d table_new_%d 0 v=pwl%s",
                                    numb, numb, tabfun + 5);
                    numb++;
                    tfree(tabfun);
                    tfree(begline);
                    tfree(card->line);
                    card->line = cut_line = neweline;
                    insert_new_line(card, newbline, 0, card->linenum_orig, card->linesource);
                    /* read next TABLE function in cut_line */
                    ftablebeg = strstr(cut_line, "table(");
                }
                continue;
            }
        }
    }
}

/* find the model requested by ako:model and do the replacement */
static struct card *find_model(struct card *startcard,
        struct card *changecard, char *searchname, char *newmname,
        char *newmtype, char *endstr)
{
    struct card *nomod, *returncard = changecard;
    char *origmname, *origmtype;
    char *beginline = startcard->line;
    if (ciprefix(".subckt", beginline))
        startcard = startcard->nextcard;

    int nesting2 = 0;
    for (nomod = startcard; nomod; nomod = nomod->nextcard) {
        char *origmodline = nomod->line;
        if (ciprefix(".subckt", origmodline))
            nesting2++;
        if (ciprefix(".ends", origmodline))
            nesting2--;
        /* skip any subcircuit */
        if (nesting2 > 0)
            continue;
        if (nesting2 == -1) {
            returncard = changecard;
            break;
        }
        if (ciprefix(".model", origmodline)) {
            origmodline = nexttok(origmodline);
            origmname = gettok(&origmodline);
            origmtype = gettok_noparens(&origmodline);
            if (cieq(origmname, searchname)) {
                if (!eq(origmtype, newmtype)) {
                    fprintf(stderr,
                            "Error: Original (%s) and new (%s) type for AKO "
                            "model disagree\n",
                            origmtype, newmtype);
                    controlled_exit(1);
                }
                /* we have got it */
                char *newmodcard = tprintf(".model %s %s %s%s",
                        newmname, newmtype, origmodline, endstr);
                char *tmpstr = strstr(newmodcard, ")(");
                if (tmpstr) {
                    tmpstr[0] = ' ';
                    tmpstr[1] = ' ';
                }
                tfree(changecard->line);
                changecard->line = newmodcard;
                tfree(origmname);
                tfree(origmtype);
                returncard = NULL;
                break;
            }
            tfree(origmname);
            tfree(origmtype);
        }
        else
            returncard = changecard;
    }
    return returncard;
}

/* Process any .distribution cards for PSPICE's Monte-Carlo feature.
 * A .distribution card defines a probability distribution by a PWL
 * density function.  This could be rewritten as a function that
 * returns a random value following that distribution.
 * For now, just comment it away.
 */
static void do_distribution(struct card *oldcard) {
    while (oldcard) {
        char *line = oldcard->line;

        if (line && ciprefix(".distribution", line))
            *line = '*';
        oldcard = oldcard->nextcard;
    }
}

/* Do the .model replacement required by ako (a kind of)
 * PSPICE does not support nested .subckt definitions, so
 * a simple structure is needed: search for ako:modelname,
 * then for modelname in the subcircuit or in the top level.
 * .model qorig npn (BF=48 IS=2e-7)
 * .model qbip1 ako:qorig NPN (BF=60 IKF=45m)
 * after the replacement we have
 * .model qbip1 NPN (BF=48 IS=2e-7 BF=60 IKF=45m)
 * and we benefit from the fact that if parameters have
 * doubled, the last entry of a parameter (e.g. BF=60)
 * overwrites the previous one (BF=48).
 */
static struct card *ako_model(struct card *startcard)
{
    char *newmname, *newmtype;
    struct card *card, *returncard = NULL, *subcktcard = NULL;
    for (card = startcard; card; card = card->nextcard) {
        char *akostr, *searchname;
        char *cut_line = card->line;

        if (ciprefix(".subckt", cut_line))
            subcktcard = card;
        else if (ciprefix(".ends", cut_line))
            subcktcard = NULL;
        if (ciprefix(".model", cut_line)) {
            if ((akostr = strstr(cut_line, "ako:")) != NULL &&
                isspace_c(akostr[-1])) {
                akostr += 4;
                searchname = gettok(&akostr);
                cut_line = nexttok(cut_line);
                newmname = gettok(&cut_line);
                newmtype = gettok_noparens(&akostr);

                /* Find the model and do the replacement. */

                if (subcktcard)
                    returncard = find_model(subcktcard, card, searchname,
                                            newmname, newmtype, akostr);
                if (returncard || !subcktcard)
                    returncard = find_model(startcard, card, searchname,
                                            newmname, newmtype, akostr);
                tfree(searchname);
                tfree(newmname);
                tfree(newmtype);

                /* Replacement not possible, bail out. */

                if (returncard)
                    break;
            }
        }
    }
    return returncard;
}

struct vsmodels {
    char *modelname;
    char *subcktline;
    struct vsmodels *nextmodel;
};

/* insert a new model, just behind the given model */
static struct vsmodels *insert_new_model(
        struct vsmodels *vsmodel, char *name, char *subcktline)
{
    struct vsmodels *x = TMALLOC(struct vsmodels, 1);

    x->nextmodel = vsmodel ? vsmodel->nextmodel : NULL;
    x->modelname = copy(name);
    x->subcktline = copy(subcktline);
    if (vsmodel)
        vsmodel->nextmodel = x;
    else
        vsmodel = x;

    return vsmodel;
}

/* find the model */
static bool find_a_model(
        struct vsmodels *vsmodel, char *name, char *subcktline)
{
    struct vsmodels *x;
    for (x = vsmodel; vsmodel; vsmodel = vsmodel->nextmodel)
        if (eq(vsmodel->modelname, name) &&
                eq(vsmodel->subcktline, subcktline))
            return TRUE;
    return FALSE;
}

/* delete the vsmodels list */
static bool del_models(struct vsmodels *vsmodel)
{
    struct vsmodels *x;

    if (!vsmodel)
        return FALSE;

    while (vsmodel) {
        x = vsmodel->nextmodel;
        tfree(vsmodel->modelname);
        tfree(vsmodel->subcktline);
        tfree(vsmodel);
        vsmodel = x;
    }

    return TRUE;
}

/* Check for double '{', replace the inner '{', '}' by '(', ')'
   in .subckt, .model, or .param (which all three may stem from external sources) */
static void rem_double_braces(struct card* newcard)
{
    struct card* card;
    int slevel = 0;

    for (card = newcard; card; card = card->nextcard) {
        char* cut_line = card->line;
        if (ciprefix(".subckt", cut_line))
            slevel++;
        else if (ciprefix(".ends", cut_line))
            slevel--;
        if (ciprefix(".model", cut_line) || slevel > 0 || ciprefix(".param", cut_line)) {
            cut_line = strchr(cut_line, '{');
            if (cut_line) {
                int level = 1;
                cut_line++;
                while (*cut_line != '\0') {
                    if (*cut_line == '{') {
                        level++;
                        if (level > 1)
                            *cut_line = '(';
                    }
                    else if (*cut_line == '}') {
                        if (level > 1)
                            *cut_line = ')';
                        level--;
                    }
                    cut_line++;
                }
            }
        }
    }
}

#ifdef INTEGRATE_UDEVICES
static void list_the_cards(struct card *startcard, char *prefix)
{
    struct card *card;
    if (!startcard) { return; }
    for (card = startcard; card; card = card->nextcard) {
        char* cut_line = card->line;
        printf("%s %s\n", prefix, cut_line);
    }
}

static struct card *the_last_card(struct card *startcard)
{
    struct card *card, *lastcard = NULL;
    if (!startcard) { return NULL; }
    for (card = startcard; card; card = card->nextcard) {
        lastcard = card;
    }
    return lastcard;
}
 static void remove_old_cards(struct card *first, struct card *stop)
{
    struct card *x, *y, *next = NULL, *nexta = NULL;
    if (!first || !stop || (first == stop)) { return; }
    for (x = first; (x && (x != stop)); x = next) {
        if (x->line) { tfree(x->line); }
        if (x->error) { tfree(x->error); }
        for (y = x->actualLine; y; y = nexta) {
            if (y->line) { tfree(y->line); }
            if (y->error) { tfree(y->error); }
            nexta = y->nextcard;
            tfree(y);
        }
        next = x->nextcard;
        tfree(x);
    }

}

static struct card *u_instances(struct card *startcard)
{
    struct card *card, *returncard = NULL, *subcktcard = NULL;
    struct card *newcard = NULL, *last_newcard = NULL;
    int models_ok = 0, models_not_ok = 0;
    int udev_ok = 0, udev_not_ok = 0;
    BOOL create_called = FALSE, repeat_pass = FALSE;
    BOOL skip_next = FALSE;
    struct card *c = startcard;
    BOOL insub = FALSE;
    int ps_global_tmodels = 0;

    if (!cp_getvar("ps_global_tmodels", CP_NUM, &ps_global_tmodels, 0)) {
        ps_global_tmodels = 0;
    }
    if (ps_global_tmodels) {
        initialize_udevice(NULL);
        /* First scan for global timing models */
        while (c) {
            char *line = c->line;
            if (ciprefix(".subckt", line)) {
                insub = TRUE;
            } else if (ciprefix(".ends", line)) {
                insub = FALSE;
            }
            if (!insub && ciprefix(".model", line)) {
                (void) u_process_model_line(line, TRUE);
            }
            c = c->nextcard;
        }
    }

    /* Now scan for subckts containing U* instances and local timing models */
    card = startcard;
    while (card) {
        char *cut_line = card->line;

        skip_next = FALSE;
        if (ciprefix(".subckt", cut_line)) {
            models_ok = models_not_ok = 0;
            udev_ok = udev_not_ok = 0;
            subcktcard = card;
            if (!repeat_pass) {
                if (create_called) {
                    cleanup_udevice(FALSE);
                }
                initialize_udevice(subcktcard->line);
                create_called = TRUE;
            }
        } else if (ciprefix(".ends", cut_line)) {
            if (repeat_pass) {
                newcard = replacement_udevice_cards();
                if (newcard) {
                    char *tmp = NULL, *pos, *posp, *new_str = NULL, *cl;
                    struct card* tmpc;
                    /* replace linenum_orig and linesource */
                    for (tmpc = newcard; tmpc; tmpc = tmpc->nextcard) {
                        tmpc->linenum_orig = subcktcard->linenum_orig;
                        tmpc->linesource = subcktcard->linesource;
                    }
                    DS_CREATE(ds_tmp, 128);
                    /* Pspice definition of .subckt card:
                       .SUBCKT <name> [node]*
                       + [OPTIONAL: < <interface node> = <default value> >*]
                       + [PARAMS: < <name> = <value> >* ]
                       + [TEXT: < <name> = <text value> >* ]
                       ...
                       .ENDS
                    */
                    cl = subcktcard->line;
                    tmp = TMALLOC(char, strlen(cl) + 1);
                    (void) memcpy(tmp, cl, strlen(cl) + 1);
                    pos = strstr(tmp, "optional:");
                    posp = strstr(tmp, "params:");
                    ds_clear(&ds_tmp);
                    /* If there is an optional: and a param: then posp > pos */
                    if (pos) {
                        /* Remove the optional: section if present */
                        *pos = '\0';
                        if (posp) {
                            ds_cat_str(&ds_tmp, tmp);
                            ds_cat_str(&ds_tmp, posp);
                            new_str = copy(ds_get_buf(&ds_tmp));
                        } else {
                            new_str = copy(tmp);
                        }
                    } else {
                        new_str = copy(tmp);
                    }
                    ds_free(&ds_tmp);
                    tfree(tmp);
                    remove_old_cards(subcktcard->nextcard, card);
                    subcktcard->nextcard = newcard;
                    tfree(subcktcard->line);
                    subcktcard->line = new_str;
                    if (ft_ngdebug) {
                        printf("%s\n", new_str);
                        list_the_cards(newcard, "Replacement:");
                    }
                    last_newcard = the_last_card(newcard);
                    if (last_newcard) {
                        last_newcard->nextcard = card; // the .ends card
                    }
                } else {
                    models_ok = models_not_ok = 0;
                    udev_ok = udev_not_ok = 0;
                }
            }
            if (models_not_ok > 0 || udev_not_ok > 0) {
                repeat_pass = FALSE;
                cleanup_udevice(FALSE);
                create_called = FALSE;
            } else if (udev_ok > 0) {
                repeat_pass = TRUE;
                card = subcktcard;
                skip_next = TRUE;
            } else {
                repeat_pass = FALSE;
                cleanup_udevice(FALSE);
                create_called = FALSE;
            }
            subcktcard = NULL;
        } else if (ciprefix(".model", cut_line)) {
            if (subcktcard && !repeat_pass) {
                // Add .model local to subckt
                if (!u_process_model_line(cut_line, FALSE)) {
                    models_not_ok++;
                } else {
                    models_ok++;
                }
            }
        } else if (ciprefix("u", cut_line) || ciprefix("x", cut_line)) {
            /* U* device instance or X* instance of a subckt */
            if (subcktcard) {
                if (repeat_pass) {
                    if (!u_process_instance(cut_line)) {
                        repeat_pass = FALSE;
                        cleanup_udevice(FALSE);
                        create_called = FALSE;
                        subcktcard = NULL;
                        models_ok = models_not_ok = 0;
                        udev_ok = udev_not_ok = 0;
                        skip_next = FALSE;
                    }
                } else {
                    if (u_check_instance(cut_line)) {
                        udev_ok++;
                    } else {
                        udev_not_ok++;
                    }
                }
            }
        } else {
            if (!ciprefix("*", cut_line)) {
                udev_not_ok++;
            }
        }

        if (!skip_next) {
            card = card->nextcard;
        }
    }
    if (create_called) {
        cleanup_udevice(FALSE);
    }
    cleanup_udevice(TRUE);
    return returncard;
}
#endif

/**** PSPICE to ngspice **************
* .model replacement in ako (a kind of) model descriptions
* replace the E source TABLE function by a B source pwl
* add predefined params TEMP, VT, GMIN to beginning of deck
* add predefined params TEMP, VT, GMIN to beginning of each .subckt call
* add .functions limit, pwr, pwrs, stp, if, int
* replace vswitch part S
  S1 D S DG GND SWN
 .MODEL SWN VSWITCH(VON = { 0.55 } VOFF = { 0.49 }
     RON = { 1 / (2 * M*(W / LE)*(KPN / 2) * 10) }  ROFF = { 1G })
* by
  as1 %vd(DG GND) % gd(D S) aswn
  .model aswn aswitch(cntl_off={0.49} cntl_on={0.55} r_off={1G}
  + r_on={ 1 / (2 * M*(W / LE)*(KPN / 2) * 10) } log = TRUE)
* replace vswitch part S_ST
  S1 D S DG GND S_ST
 .MODEL S_ST VSWITCH(VT = { 1.5 } VH = { 0.3 }
     RON = { 1 / (2 * M*(W / LE)*(KPN / 2) * 10) }  ROFF = { 1G })
* by the classical voltage controlled ngspice switch
  S1 D S DG GND SWN
 .MODEL S_ST SW(VT = { 1.5 } VH = { 0.3 }
     RON = { 1 / (2 * M*(W / LE)*(KPN / 2) * 10) }  ROFF = { 1G })
  switch parameter td is not yet supported
* replace & by &&
* replace | by ||
* in R instance, replace TC = xx1, xx2 by TC1=xx1 TC2=xx2
* replace T_ABS by temp and T_REL_GLOBAL by dtemp in .model cards
* get the area factor for diodes and bipolar devices
* in subcircuit .subckt and X lines with 'params:' statement
  replace comma separator by space. Do nothing if comma is inside of {}.
* in .model, if double curly braces {{}}, replace the inner by {()}  */
struct card *pspice_compat(struct card *oldcard)
{
    struct card *card, *newcard, *nextcard;
    struct vsmodels *modelsfound = NULL;
    int skip_control = 0;

    /* .model replacement in ako (a kind of) model descriptions
     * in first .subckt and top level only */
    struct card *errcard;
    if ((errcard = ako_model(oldcard)) != NULL) {
        fprintf(stderr, "Error: no model found for %s\n", errcard->line);
        controlled_exit(1);
    }

    /* Process .distribution cards. */
    do_distribution(oldcard);

    /* replace TABLE function in E source */
    replace_table(oldcard);

    /* remove double braces */
    rem_double_braces(oldcard);

    /* add predefined params TEMP, VT, GMIN to beginning of deck */
    char *new_str = copy(".param temp = 'temper'");
    newcard = insert_new_line(NULL, new_str, 1, 0, "internal");
    new_str = copy(".param vt = '(temper + 273.15) * 8.6173303e-5'");
    nextcard = insert_new_line(newcard, new_str, 2, 0, "internal");
    new_str = copy(".param gmin = 1e-12");
    nextcard = insert_new_line(nextcard, new_str, 3, 0, "internal");
    /* add funcs limit, pwr, pwrs, stp, if, int */
    /* LIMIT( Output Expression, Limit1, Limit2)
       Output will stay between the two limits given. */
    new_str = copy(".func limit(x, a, b) { ternary_fcn(a > b, max(min(x, a), b), max(min(x, b), a)) }");
    nextcard = insert_new_line(nextcard, new_str, 4, 0, "internal");
    new_str = copy(".func pwr(x, a) { pow(x, a) }");
    nextcard = insert_new_line(nextcard, new_str, 5, 0, "internal");
    new_str = copy(".func pwrs(x, a) { sgn(x) * pow(x, a) }");
    nextcard = insert_new_line(nextcard, new_str, 6, 0, "internal");
    new_str = copy(".func stp(x) { u(x) }");
    nextcard = insert_new_line(nextcard, new_str, 7, 0, "internal");
    new_str = copy(".func if(a, b, c) {ternary_fcn( a , b , c )}");
    nextcard = insert_new_line(nextcard, new_str, 8, 0, "internal");
    new_str = copy(".func int(x) { sign(x)*floor(abs(x)) }");
    nextcard = insert_new_line(nextcard, new_str, 9, 0, "internal");
    nextcard->nextcard = oldcard;

#ifdef INTEGRATE_UDEVICES
    {
        struct card *ucard;
#ifdef TRACE
        list_the_cards(newcard, "Before udevices");
#endif
        ucard = u_instances(newcard);
#ifdef TRACE
        list_the_cards(newcard, "After udevices");
#endif
    }
#endif


    /* add predefined parameters TEMP, VT after each subckt call */
    /* FIXME: This should not be necessary if we had a better sense of
    hierarchy during the evaluation of TEMPER */
    for (card = newcard; card; card = card->nextcard) {
        char *cut_line = card->line;
        if (ciprefix(".subckt", cut_line)) {
            new_str = copy(".param temp = 'temper'");
            nextcard = insert_new_line(card, new_str, 0, card->linenum_orig, card->linesource);
            new_str = copy(".param vt = '(temper + 273.15) * 8.6173303e-5'");
            nextcard = insert_new_line(nextcard, new_str, 1, card->linenum_orig, card->linesource);
            /* params: replace comma separator by space.
               Do nothing if you are inside of { }. */
            char* parastr = strstr(cut_line, "params:");
            int brace = 0;
            if (parastr) {
                parastr += 8;
                while (*parastr) {
                    if (*parastr == '{')
                        brace++;
                    else if (*parastr == '}')
                        brace--;
                    if (brace == 0 && *parastr == ',')
                        *parastr = ' ';
                    parastr++;
                }
            }
        }
    }

    /* .model xxx NMOS/PMOS level=6 --> level = 8,  version=3.2.4
       .model xxx NMOS/PMOS level=7 --> level = 8,  version=3.2.4
       .model xxx NMOS/PMOS level=5 --> level = 44
       .model xxx NMOS/PMOS level=8 --> level = 14, version=4.5.0
       .model xxx NPN/PNP   level=2 --> level = 6
       .model xxx LPNP      level=n --> level = 1 subs=-1
       Remove any Monte - Carlo variation parameters from .model cards.*/
    for (card = newcard; card; card = card->nextcard) {
        char* cut_line = card->line;
        if (ciprefix(".model", cut_line)) {
            char* modname, *modtype, *curr_line;
            int i;
            char *cut_del = curr_line = cut_line = inp_remove_ws(copy(cut_line));
            cut_line = nexttok(cut_line); /* skip .model */
            modname = gettok(&cut_line); /* save model name */
            if (!modname) {
                fprintf(stderr, "Error: No model name given for %s\n", curr_line);
                controlled_exit(EXIT_BAD);
            }
            modtype = gettok_noparens(&cut_line); /* save model type */
            if (!modtype) {
                fprintf(stderr, "Error: No model type given for %s\n", curr_line);
                controlled_exit(EXIT_BAD);
            }
            if (cieq(modtype, "NMOS") || cieq(modtype, "PMOS")) {
                char* lv = strstr(cut_line, "level=");
                if (lv) {
                    int ll;
                    lv = lv + 6;
                    char* ntok = gettok(&lv);
                    ll = atoi(ntok);
                    switch (ll) {
                    case 5:
                        {
                        /* EKV 2.6 in the adms branch */
                        char* newline = tprintf(".model %s %s level=44 %s", modname, modtype, lv);
                        tfree(card->line);
                        card->line = curr_line = newline;
                        }
                        break;
                    case 6:
                    case 7:
                        {
                        /* BSIM3 version 3.2.4 */
                        char* newline = tprintf(".model %s %s level=8 version=3.2.4 %s", modname, modtype, lv);
                        tfree(card->line);
                        card->line = curr_line = newline;
                        }
                        break;
                    case 8:
                        {
                        /* BSIM4 version 4.5.0 */
                        char* newline = tprintf(".model %s %s level=14 version=4.5.0 %s", modname, modtype, lv);
                        tfree(card->line);
                        card->line = curr_line = newline;
                        }
                        break;
                    default:
                        break;
                    }
                    tfree(ntok);
                }
            }
            else if (cieq(modtype, "NPN") || cieq(modtype, "PNP")) {
                char* lv = strstr(cut_line, "level=");
                if (lv) {
                    int ll;
                    lv = lv + 6;
                    char* ntok = gettok(&lv);
                    ll = atoi(ntok);
                    switch (ll) {
                    case 2:
                        {
                        /* MEXTRAM 504.12.1 in the adms branch */
                        char* newline = tprintf(".model %s %s level=6 %s", modname, modtype, lv);
                        tfree(card->line);
                        card->line = curr_line = newline;
                        }
                        break;
                    default:
                        break;
                    }
                    tfree(ntok);
                }
            }
            else if (cieq(modtype, "LPNP")) {
                /* lateral PNP enabled */
                char* newline = tprintf(".model %s PNP level=1 subs=-1 %s", modname, cut_line);
                tfree(card->line);
                card->line = curr_line = newline;
            }
            tfree(modname);
            tfree(modtype);

            /* Remove any Monte-Carlo variation parameters. They qualify
             * a previous parameter, so there must be at least 3 tokens.
             * There are two keywords "dev" (different values for each device),
             * and "lot" (all devices of this model share a value).
             * The keyword may be optionally followed by '/' and
             * a probability distribution name, then there must be '=' and
             * a value, then an optional '%' indicating relative rather than
             * absolute variation. Allow muliple lot and dev on a single .model line.
             */
            bool remdevlot = FALSE;
            cut_line = curr_line;
            for (i = 0; i < 3; i++)
                cut_line = nexttok(cut_line);
            while (cut_line) {
                if (!strncmp(cut_line, "dev=", 4) ||
                    !strncmp(cut_line, "lot=", 4)) {
                    while (*cut_line && !isspace_c(*cut_line)) {
                        *cut_line++ = ' ';
                    }
                    remdevlot = TRUE;
                    cut_line = skip_ws(cut_line);
                    continue;
                }
                cut_line = nexttok(cut_line);
            }
            if (remdevlot) {
                tfree(card->line);
                card->line = curr_line;
            }
            else
                tfree(cut_del);
        } // if .model
    } // for loop through all cards

    /* x ... params: p1=val1, p2=val2 replace comma separator by space.
       Do nothing if you are inside of { }. */
    for (card = newcard; card; card = card->nextcard) {
        char* cut_line = card->line;
        if (ciprefix("x", cut_line)) {
            char* parastr = strstr(cut_line, "params:");
            int brace = 0;
            if (parastr) {
                parastr += 8;
                while (*parastr) {
                    if (*parastr == '{')
                        brace++;
                    else if (*parastr == '}')
                        brace--;
                    if (brace == 0 && *parastr == ',')
                        *parastr = ' ';
                    parastr++;
                }
            }
        }
    }

    /* in R instance, replace TC = xx1, xx2 by TC1=xx1 TC2=xx2 */
    for (card = newcard; card; card = card->nextcard) {
        char *cut_line = card->line;

        /* exclude any command inside .control ... .endc */
        if (ciprefix(".control", cut_line)) {
            skip_control++;
            continue;
        }
        else if (ciprefix(".endc", cut_line)) {
            skip_control--;
            continue;
        }
        else if (skip_control > 0) {
            continue;
        }

        if (*cut_line == 'r' || *cut_line == 'l' || *cut_line == 'c') {
            /* Skip name and two nodes */
            char *ntok = nexttok(cut_line);
            ntok = nexttok(ntok);
            ntok = nexttok(ntok);
            if (!ntok || *ntok == '\0') {
                fprintf(stderr, "Error: Missing token in line %d:\n%s\n",
                        card->linenum, cut_line);
                fprintf(stderr, "    Please correct the input file\n");
                controlled_exit(1);
            }
            char *tctok = search_plain_identifier(ntok, "tc");
            if (tctok) {
                char *tc1, *tc2;
                char *tctok1 = strchr(tctok, '=');
                if (tctok1)
                    /* skip '=' */
                    tctok1 += 1;
                else
                    /* no '=' found, skip 'tc' */
                    tctok1 = tctok + 2;
                /* tc1 may be an expression, enclosed in {} */
                if (*tctok1 == '{') {
                    tc1 = gettok_char(&tctok1, '}', TRUE, TRUE);
                }
                else {
                    tc1 = gettok_node(&tctok1);
                }
                /* skip spaces and commas */
                while (isspace_c(*tctok1) || (*tctok1 == ','))
                   tctok1++;
                /* tc2 may be an expression, enclosed in {} */
                if (*tctok1 == '{') {
                    tc2 = gettok_char(&tctok1, '}', TRUE, TRUE);
                }
                else {
                    tc2 = gettok_node(&tctok1);
                }
                tctok[-1] = '\0';
                char *newstring;
                if (tc1 && tc2)
                    newstring = tprintf("%s tc1=%s tc2=%s",
                            cut_line, tc1, tc2);
                else if (tc1)
                    newstring = tprintf("%s tc1=%s", cut_line, tc1);
                else {
                    fprintf(stderr,
                            "Warning: tc without parameters removed in line "
                            "\n   %s\n",
                            cut_line);
                    continue;
                }
                tfree(card->line);
                card->line = newstring;
                tfree(tc1);
                tfree(tc2);
            }
        }
    }

    /* replace & with && , | with || , *# with * # , and ~ with ! */
    for (card = newcard; card; card = card->nextcard) {
        char *t;
        char *cut_line = card->line;

        /* we don't have command lines in a PSPICE model */
        if (ciprefix("*#", cut_line)) {
            char *tmpstr = tprintf("* #%s", cut_line + 2);
            tfree(card->line);
            card->line = tmpstr;
            continue;
        }

        if (*cut_line == '*')
            continue;

        if (*cut_line == '\0')
            continue;

        /* exclude any command inside .control ... .endc */
        if (ciprefix(".control", cut_line)) {
            skip_control++;
            continue;
        }
        else if (ciprefix(".endc", cut_line)) {
            skip_control--;
            continue;
        }
        else if (skip_control > 0) {
            continue;
        }
        if ((t = strstr(card->line, "&")) != NULL) {
            while (t && (t[1] != '&')) {
                char *tt = NULL;
                char *tn = copy(t + 1); /*skip |*/
                char *strbeg = copy_substring(card->line, t);
                tfree(card->line);
                card->line = tprintf("%s&&%s", strbeg, tn);
                tfree(strbeg);
                tfree(tn);
                t = card->line;
                while ((t = strstr(t, "&&")) != NULL)
                    tt = t = t + 2;
                if (!tt)
                    break;
                else
                    t = strstr(tt, "&");
            }
        }
        if ((t = strstr(card->line, "|")) != NULL) {
            while (t && (t[1] != '|')) {
                char *tt = NULL;
                char *tn = copy(t + 1); /*skip |*/
                char *strbeg = copy_substring(card->line, t);
                tfree(card->line);
                card->line = tprintf("%s||%s", strbeg, tn);
                tfree(strbeg);
                tfree(tn);
                t = card->line;
                while ((t = strstr(t, "||")) != NULL)
                    tt = t = t + 2;
                if (!tt)
                    break;
                else
                    t = strstr(tt, "|");
            }
        }
        /* We may have '~' in path names or A devices */
        if (ciprefix(".inc", card->line) || ciprefix(".lib", card->line) ||
                ciprefix("A", card->line))
            continue;

        if ((t = strstr(card->line, "~")) != NULL) {
            while (t) {
                *t = '!';
                t = strstr(t, "~");
            }
        }
    }

    /* replace T_ABS by temp, T_REL_GLOBAL by dtemp, and T_MEASURED by TNOM
    in .model cards. What about T_REL_LOCAL ? T_REL_LOCAL is used in
    conjunction with AKO and is not yet implemented.  */
    for (card = newcard; card; card = card->nextcard) {
        char *cut_line = card->line;
        if (ciprefix(".model", cut_line)) {
            char *t_str;
            if ((t_str = strstr(cut_line, "t_abs")) != NULL)
                memcpy(t_str, " temp", 5);
            else if ((t_str = strstr(cut_line, "t_rel_global")) != NULL)
                memcpy(t_str, "       dtemp", 12);
            else if ((t_str = strstr(cut_line, "t_measured")) != NULL)
                memcpy(t_str, "      tnom", 10);
        }
    }

    /* get the area factor for diodes and bipolar devices
    d1 n1 n2 dmod 7 --> d1 n1 n2 dmod area=7
    q2 n1 n2 n3 [n4] bjtmod 1.35 --> q2 n1 n2 n3 n4 bjtmod area=1.35
    q3 1 2 3 4 bjtmod 1.45 --> q2 1 2 3 4 bjtmod area=1.45
    */
    for (card = newcard; card; card = card->nextcard) {
        char *cut_line = card->line;
        if (*cut_line == '*')
            continue;
        // exclude any command inside .control ... .endc
        if (ciprefix(".control", cut_line)) {
            skip_control++;
            continue;
        }
        else if (ciprefix(".endc", cut_line)) {
            skip_control--;
            continue;
        }
        else if (skip_control > 0) {
            continue;
        }
        if (*cut_line == 'q') {
            /* According to PSPICE Reference Guide the fourth (substrate) node
            has to be put into [] if it is not just a number */
            cut_line = nexttok(cut_line); //.model
            cut_line = nexttok(cut_line); // node1
            cut_line = nexttok(cut_line); // node2
            cut_line = nexttok(cut_line); // node3
            if (!cut_line || *cut_line == '\0') {
                fprintf(stderr, "Line no. %d, %s, missing tokens\n",
                        card->linenum_orig, card->line);
                if (ft_stricterror)
                    controlled_exit(1);
                else
                    continue;
            }
            if (*cut_line == '[') { // node4 not a number
                *cut_line = ' ';
                cut_line = strchr(cut_line, ']');
                *cut_line = ' ';
                cut_line = skip_ws(cut_line);
                cut_line = nexttok(cut_line); // model name
            }
            else { // if an integer number, it is node4
                bool is_node4 = TRUE;
                while (*cut_line && !isspace_c(*cut_line))
                    if (!isdigit_c(*cut_line++))
                        is_node4 = FALSE; // already model name
                if (is_node4) {
                    cut_line = nexttok(cut_line); // model name
                }
            }
            if (cut_line && *cut_line &&
                    atof(cut_line) > 0.0) { // size of area is a real number
                char *tmpstr1 = copy_substring(card->line, cut_line);
                char *tmpstr2 = tprintf("%s area=%s", tmpstr1, cut_line);
                tfree(tmpstr1);
                tfree(card->line);
                card->line = tmpstr2;
            }
            else if (cut_line && *cut_line &&
                    *(skip_ws(cut_line)) ==
                            '{') { // size of area is parametrized inside {}
                char *tmpstr1 = copy_substring(card->line, cut_line);
                char *tmpstr2 = gettok_char(&cut_line, '}', TRUE, TRUE);
                char *tmpstr3 =
                        tprintf("%s area=%s %s", tmpstr1, tmpstr2, cut_line);
                tfree(tmpstr1);
                tfree(tmpstr2);
                tfree(card->line);
                card->line = tmpstr3;
            }
        }
        else if (*cut_line == 'd') {
            cut_line = nexttok(cut_line); //.model
            cut_line = nexttok(cut_line); // node1
            cut_line = nexttok(cut_line); // node2
            if (!cut_line || *cut_line == '\0') {
                fprintf(stderr, "Line no. %d, %s, missing tokens\n",
                        card->linenum_orig, card->line);
                if (ft_stricterror)
                    controlled_exit(1);
                else
                    continue;
            }
            cut_line = nexttok(cut_line); // model name
            if (cut_line && *cut_line &&
                    atof(cut_line) > 0.0) { // size of area
                char *tmpstr1 = copy_substring(card->line, cut_line);
                char *tmpstr2 = tprintf("%s area=%s", tmpstr1, cut_line);
                tfree(tmpstr1);
                tfree(card->line);
                card->line = tmpstr2;
            }
        }
    }

    /* if vswitch part s, replace
     * S1 D S DG GND SWN
     * .MODEL SWN VSWITCH ( VON = {0.55} VOFF = {0.49}
     *         RON={1/(2*M*(W/LE)*(KPN/2)*10)}  ROFF={1G} )
     * by
     * a1 %v(DG) %gd(D S) swa
     * .MODEL SWA aswitch(cntl_off=0.49 cntl_on=0.55 r_off=1G
     *         r_on={1/(2*M*(W/LE)*(KPN/2)*10)} log=TRUE)
     *
     * if vswitch part s_st, don't replace instance, only model
     * replace
     * S1 D S DG GND S_ST
     * .MODEL S_ST VSWITCH(VT = { 1.5 } VH = { 0.s }
           RON = { 1 / (2 * M*(W / LE)*(KPN / 2) * 10) }  ROFF = { 1G })
     * by the classical voltage controlled ngspice switch
     * S1 D S DG GND S_ST
     * .MODEL S_ST SW(VT = { 1.5 } VH = { 0.s }
             RON = { 1 / (2 * M*(W / LE)*(KPN / 2) * 10) }  ROFF = { 1G })
     * vswitch delay parameter td is not yet supported

     * simple hierachy, as nested subcircuits are not allowed in PSPICE */

    /* first scan: find the vswitch models, transform them and put the S models
       into a list */
    for (card = newcard; card; card = card->nextcard) {
        char *str;
        static struct card *subcktline = NULL;
        static int nesting = 0;
        char *cut_line = card->line;
        if (ciprefix(".subckt", cut_line)) {
            subcktline = card;
            nesting++;
        }
        if (ciprefix(".ends", cut_line))
            nesting--;

        if (ciprefix(".model", card->line) && strstr(card->line, "vswitch")) {
            char *modname;

            str = card->line = inp_remove_ws(card->line);
            str = nexttok(str); /* throw away '.model' */
            INPgetNetTok(&str, &modname, 0); /* model name */
            if (!ciprefix("vswitch", str)) {
                tfree(modname);
                continue;
            }
            str = nexttok_noparens(str); /* throw away 'vswitch' */
            /* S_ST switch (parameters ron, roff, vt, vh)
             * we have to find 0 to 4 parameters, identified by 'vh=' etc.
             * Parameters not found have to be replaced by their default values. */
            if (strstr(str, "vt=") || strstr(str, "vh=")) {
                char* newstr;
                char* lstr = copy(str);
                char* partstr = strstr(lstr, "ron=");
                if (!partstr) {
                    newstr = tprintf("%s %s", "ron=1.0", lstr);  //default value
                    tfree(lstr);
                    lstr = newstr;
                }
                partstr = strstr(lstr, "roff=");
                if (!partstr) {
                    newstr = tprintf("%s %s", "roff=1.0e12", lstr);  //default value
                    tfree(lstr);
                    lstr = newstr;
                }
                partstr = strstr(lstr, "vt=");
                if (!partstr) {
                    newstr = tprintf("%s %s", "vt=0", lstr);  //default value
                    tfree(lstr);
                    lstr = newstr;
                }
                partstr = strstr(lstr, "vh=");
                if (!partstr) {
                    newstr = tprintf("%s %s", "vh=0", lstr);  //default value
                    tfree(lstr);
                    lstr = newstr;
                }
                tfree(card->line);
                if (lstr[strlen(lstr) - 1] == ')')
                    card->line = tprintf(".model %s sw ( %s", modname, lstr);
                else
                    card->line = tprintf(".model %s sw %s", modname, lstr);
                tfree(lstr);
                tfree(modname);
            }
            /* S vswitch  (parameters ron, roff, von, voff) */
            /* We have to find 0 to 4 parameters, identified by 'von=' etc. and
             * replace them by the pswitch code model parameters
             * replace VON by cntl_on, VOFF by cntl_off, RON by r_on, and ROFF by r_off.
             * Parameters not found have to be replaced by their default values. */
            else if (strstr(str, "von=") || strstr(str, "voff=")) {
                char* newstr, *begstr;
                char* lstr = copy(str);
                /* ron */
                char* partstr = strstr(lstr, "ron=");
                if (!partstr) {
                    newstr = tprintf("%s %s", "r_on=1.0", lstr);  //default value
                }
                else {
                    begstr = copy_substring(lstr, partstr);
                    newstr = tprintf("%s r_on%s", begstr, partstr + 3);
                    tfree(begstr);
                }
                tfree(lstr);
                lstr = newstr;
                /* roff */
                partstr = strstr(lstr, "roff=");
                if (!partstr) {
                    newstr = tprintf("%s %s", "r_off=1.0e6", lstr);  //default value
                }
                else {
                    begstr = copy_substring(lstr, partstr);
                    newstr = tprintf("%s r_off%s", begstr, partstr + 4);
                    tfree(begstr);
                }
                tfree(lstr);
                lstr = newstr;
                /* von */
                partstr = strstr(lstr, "von=");
                if (!partstr) {
                    newstr = tprintf("%s %s", "cntl_on=1", lstr);  //default value
                    tfree(lstr);
                    lstr = newstr;
                }
                else {
                    begstr = copy_substring(lstr, partstr);
                    newstr = tprintf("%s cntl_on%s", begstr, partstr + 3);
                    tfree(begstr);
                }
                tfree(lstr);
                lstr = newstr;
                /* voff */
                partstr = strstr(lstr, "voff=");
                if (!partstr) {
                    newstr = tprintf("%s %s", "cntl_off=0", lstr);  //default value
                    tfree(lstr);
                    lstr = newstr;
                }
                else {
                    begstr = copy_substring(lstr, partstr);
                    newstr = tprintf("%s cntl_off%s", begstr, partstr + 4);
                    tfree(begstr);
                }
                tfree(lstr);
                lstr = newstr;
                tfree(card->line);
                if (lstr[strlen(lstr) - 1] == ')')
                    card->line = tprintf(".model a%s pswitch( log=TRUE %s", modname, lstr);
                else
                    card->line = tprintf(".model a%s pswitch(%s log=TRUE)", modname, lstr);
                tfree(lstr);
                /* add to list, to change vswitch instance to code model line */
                if (nesting > 0)
                    modelsfound = insert_new_model(
                        modelsfound, modname, subcktline->line);
                else
                    modelsfound = insert_new_model(modelsfound, modname, "top");
                tfree(modname);
            }
            else {
                fprintf(stderr, "Error: Bad switch model in line %s\n", card->line);
            }
        }
    }

    /* no need to continue if no vswitch is found */
    if (!modelsfound)
        goto iswi;

    /* second scan: find the switch instances s calling a vswitch model and
     * transform them */
    for (card = newcard; card; card = card->nextcard) {
        static struct card *subcktline = NULL;
        static int nesting = 0;
        char *cut_line = card->line;
        if (*cut_line == '*')
            continue;
        // exclude any command inside .control ... .endc
        if (ciprefix(".control", cut_line)) {
            skip_control++;
            continue;
        }
        else if (ciprefix(".endc", cut_line)) {
            skip_control--;
            continue;
        }
        else if (skip_control > 0) {
            continue;
        }
        if (ciprefix(".subckt", cut_line)) {
            subcktline = card;
            nesting++;
        }
        if (ciprefix(".ends", cut_line))
            nesting--;

        if (ciprefix("s", cut_line)) {
            /* check for the model name */
            int i;
            bool good = TRUE;
            char *stoks[6];
            for (i = 0; i < 6; i++) {
                stoks[i] = gettok_node(&cut_line);
                if (!stoks[i]) {
                    fprintf(stderr,
                        "Error: bad syntax in line %d\n  %s\n"
                        "from file\n"
                        "  %s\n",
                        card->linenum_orig, card->line, card->linesource);
                    good = FALSE;
                    break;
                }
            }
            if (!good) {
                for (i = 0; i < 6; i++)
                    tfree(stoks[i]);
                continue;
            }
            /* rewrite s line and replace it if a model is found */
            if ((nesting > 0) &&
                    find_a_model(modelsfound, stoks[5], subcktline->line)) {
                tfree(card->line);
                card->line = tprintf("a%s %%gd(%s %s) %%gd(%s %s) a%s",
                        stoks[0], stoks[3], stoks[4], stoks[1], stoks[2],
                        stoks[5]);
            }
            /* if model is not within same subcircuit, search at top level */
            else if (find_a_model(modelsfound, stoks[5], "top")) {
                tfree(card->line);
                card->line = tprintf("a%s %%gd(%s %s) %%gd(%s %s) a%s",
                        stoks[0], stoks[3], stoks[4], stoks[1], stoks[2],
                        stoks[5]);
            }
            for (i = 0; i < 6; i++)
                tfree(stoks[i]);
        }
    }
    del_models(modelsfound);
    modelsfound = NULL;

iswi:;

    /* if iswitch part s, replace
     * W1 D S VC SWN
     * .MODEL SWN ISWITCH ( ION = {0.55} IOFF = {0.49}
     *         RON={1/(2*M*(W/LE)*(KPN/2)*10)}  ROFF={1G} )
     * by
     * a1 %v(DG) %gd(D S) swa
     * .MODEL SWA aswitch(cntl_off=0.49 cntl_on=0.55 r_off=1G
     *         r_on={1/(2*M*(W/LE)*(KPN/2)*10)} log=TRUE)
     *
     * if iswitch part s_st (short transition), don't replace instance, but only model
     * replace
     * W1 D S VC S_ST
     * .MODEL S_ST ISWITCH(IT = { 1.5 } IH = { 0.2 }
           RON = { 1 / (2 * M*(W / LE)*(KPN / 2) * 10) }  ROFF = { 1G })
     * by the classical current controlled ngspice switch
     * W1 D S DG GND S_ST
     * .MODEL S_ST CSW(IT = { 1.5 } IH = { 0.2 }
             RON = { 1 / (2 * M*(W / LE)*(KPN / 2) * 10) }  ROFF = { 1G })
     * iswitch delay parameter td is not yet supported

     * simple hierachy, as nested subcircuits are not allowed in PSPICE */

     /* first scan: find the iswitch models, transform them and put them into a
      * list */
    for (card = newcard; card; card = card->nextcard) {
        char* str;
        static struct card* subcktline = NULL;
        static int nesting = 0;
        char* cut_line = card->line;
        if (ciprefix(".subckt", cut_line)) {
            subcktline = card;
            nesting++;
        }
        if (ciprefix(".ends", cut_line))
            nesting--;

        if (ciprefix(".model", card->line) && strstr(card->line, "iswitch")) {
            char* modname;

            card->line = str = inp_remove_ws(card->line);
            str = nexttok(str); /* throw away '.model' */
            INPgetNetTok(&str, &modname, 0); /* model name */
            if (!ciprefix("iswitch", str)) {
                tfree(modname);
                continue;
            }
            str = nexttok_noparens(str); /* throw away 'iswitch' */
            /* S_ST switch (parameters ron, roff, it, ih)
             * we have to find 0 to 4 parameters, identified by 'ih=' etc.
             * Parameters not found have to be replaced by their default values. */
            if (strstr(str, "it=") || strstr(str, "ih=")) {
                char* newstr;
                char* lstr = copy(str);
                char* partstr = strstr(lstr, "ron=");
                if (!partstr) {
                    newstr = tprintf("%s %s", "ron=1.0", lstr);  //default value
                    tfree(lstr);
                    lstr = newstr;
                }
                partstr = strstr(lstr, "roff=");
                if (!partstr) {
                    newstr = tprintf("%s %s", "roff=1.0e12", lstr);  //default value
                    tfree(lstr);
                    lstr = newstr;
                }
                partstr = strstr(lstr, "it=");
                if (!partstr) {
                    newstr = tprintf("%s %s", "it=0", lstr);  //default value
                    tfree(lstr);
                    lstr = newstr;
                }
                partstr = strstr(lstr, "ih=");
                if (!partstr) {
                    newstr = tprintf("%s %s", "ih=0", lstr);  //default value
                    tfree(lstr);
                    lstr = newstr;
                }
                tfree(card->line);
                if (lstr[strlen(lstr) - 1] == ')')
                    card->line = tprintf(".model %s csw ( %s", modname, lstr);
                else
                    card->line = tprintf(".model %s csw %s", modname, lstr);
                tfree(lstr);
                tfree(modname);
            }
            /* S vswitch  (parameters ron, roff, ion, ioff) */
            /* We have to find 0 to 4 parameters, identified by 'ion=' etc. and
             * replace them by the pswitch code model parameters
             * replace VON by cntl_on, VOFF by cntl_off, RON by r_on, and ROFF by r_off.
             * Parameters not found have to be replaced by their default values. */
            else if (strstr(str, "ion=") || strstr(str, "ioff=")) {
                char* newstr, * begstr;
                char* lstr = copy(str);
                /* ron */
                char* partstr = strstr(lstr, "ron=");
                if (!partstr) {
                    newstr = tprintf("%s %s", "r_on=1.0", lstr);  //default value
                }
                else {
                    begstr = copy_substring(lstr, partstr);
                    newstr = tprintf("%s r_on%s", begstr, partstr + 3);
                }
                tfree(lstr);
                lstr = newstr;
                /* roff */
                partstr = strstr(lstr, "roff=");
                if (!partstr) {
                    newstr = tprintf("%s %s", "r_off=1.0e6", lstr);  //default value
                }
                else {
                    begstr = copy_substring(lstr, partstr);
                    newstr = tprintf("%s r_off%s", begstr, partstr + 4);
                }
                tfree(lstr);
                lstr = newstr;
                /* von */
                partstr = strstr(lstr, "ion=");
                if (!partstr) {
                    newstr = tprintf("%s %s", "cntl_on=1", lstr);  //default value
                    tfree(lstr);
                    lstr = newstr;
                }
                else {
                    begstr = copy_substring(lstr, partstr);
                    newstr = tprintf("%s cntl_on%s", begstr, partstr + 3);
                }
                tfree(lstr);
                lstr = newstr;
                /* voff */
                partstr = strstr(lstr, "ioff=");
                if (!partstr) {
                    newstr = tprintf("%s %s", "cntl_off=0", lstr);  //default value
                    tfree(lstr);
                    lstr = newstr;
                }
                else {
                    begstr = copy_substring(lstr, partstr);
                    newstr = tprintf("%s cntl_off%s", begstr, partstr + 4);
                }
                tfree(lstr);
                lstr = newstr;
                tfree(card->line);
                if (lstr[strlen(lstr) - 1] == ')')
                    card->line = tprintf(".model a%s aswitch( log=TRUE limit=TRUE %s", modname, lstr);
                else
                    card->line = tprintf(".model a%s aswitch(%s log=TRUE limit=TRUE)", modname, lstr);
                tfree(lstr);
                /* add to list, to change vswitch instance to code model line */
                if (nesting > 0)
                    modelsfound = insert_new_model(
                        modelsfound, modname, subcktline->line);
                else
                    modelsfound = insert_new_model(modelsfound, modname, "top");
                tfree(modname);
            }
            else {
                fprintf(stderr, "Error: Bad switch model in line %s\n", card->line);
            }
        }
    }

#if(0)
            /* we have to find 4 parameters, identified by '=', separated by
             * spaces */
            char* equalptr[4];
            equalptr[0] = strstr(str, "=");
            if (!equalptr[0]) {
                fprintf(stderr,
                    "Error: not enough parameters in iswitch model\n   "
                    "%s\n",
                    card->line);
                controlled_exit(1);
            }
            for (i = 1; i < 4; i++) {
                equalptr[i] = strstr(equalptr[i - 1] + 1, "=");
                if (!equalptr[i]) {
                    fprintf(stderr,
                        "Error: not enough parameters in iswitch model\n "
                        "  %s\n",
                        card->line);
                    controlled_exit(1);
                }
            }
            for (i = 0; i < 4; i++) {
                equalptr[i] = skip_back_ws(equalptr[i], str);
                while (*(equalptr[i]) != '(' && !isspace_c(*(equalptr[i])) &&
                    *(equalptr[i]) != ',')
                    (equalptr[i])--;
                (equalptr[i])++;
            }
            for (i = 0; i < 3; i++)
                modpar[i] = copy_substring(equalptr[i], equalptr[i + 1] - 1);
            if (strrchr(equalptr[3], ')'))
                modpar[3] = copy_substring(
                    equalptr[3], strrchr(equalptr[3], ')'));
            else
                /* iswitch defined without parens */
                modpar[3] = copy(equalptr[3]);

            /* check if we have parameters IT and IH */
            for (i = 0; i < 4; i++) {
                if (ciprefix("ih", modpar[i]))
                    have_ih = TRUE;
                if (ciprefix("it", modpar[i]))
                    have_it = TRUE;
            }
            if (have_ih && have_it) {
                /* replace iswitch by csw */
                char* vs = strstr(card->line, "iswitch");
                memmove(vs, "    csw", 7);
            }
            else {
                /* replace ION by cntl_on, IOFF by cntl_off, RON by r_on, and
                 * ROFF by r_off */
                tfree(card->line);
                rep_spar(modpar);
                card->line = tprintf(
                    /* FIXME: a new switch derived from pswitch with vnam input is due */
                    ".model a%s aswitch(%s %s %s %s  log=TRUE  limit=TRUE)", modname,
                   modpar[0], modpar[1], modpar[2], modpar[3]);
            }
            for (i = 0; i < 4; i++)
                tfree(modpar[i]);
            if (nesting > 0)
                modelsfound = insert_new_model(
                    modelsfound, modname, subcktline->line);
            else
                modelsfound = insert_new_model(modelsfound, modname, "top");
            tfree(modname);
        }
    }
#endif
    /* no need to continue if no iswitch is found */
    if (!modelsfound)
        return newcard;

    /* second scan: find the switch instances s calling an iswitch model and
     * transform them */
    for (card = newcard; card; card = card->nextcard) {
        static struct card* subcktline = NULL;
        static int nesting = 0;
        char* cut_line = card->line;
        if (*cut_line == '*')
            continue;
        // exclude any command inside .control ... .endc
        if (ciprefix(".control", cut_line)) {
            skip_control++;
            continue;
        }
        else if (ciprefix(".endc", cut_line)) {
            skip_control--;
            continue;
        }
        else if (skip_control > 0) {
            continue;
        }
        if (ciprefix(".subckt", cut_line)) {
            subcktline = card;
            nesting++;
        }
        if (ciprefix(".ends", cut_line))
            nesting--;

        if (ciprefix("w", cut_line)) {
            /* check for the model name */
            int i;
            char* stoks[5];
            for (i = 0; i < 5; i++)
                stoks[i] = gettok_node(&cut_line);
            /* rewrite w line and replace it if a model is found */
            if ((nesting > 0) &&
                find_a_model(modelsfound, stoks[4], subcktline->line)) {
                tfree(card->line);
                card->line = tprintf("a%s %%vnam(%s) %%gd(%s %s) a%s",
                    stoks[0], stoks[3], stoks[1], stoks[2],
                    stoks[4]);
            }
            /* if model is not within same subcircuit, search at top level */
            else if (find_a_model(modelsfound, stoks[4], "top")) {
                tfree(card->line);
                card->line = tprintf("a%s %%vnam(%s) %%gd(%s %s) a%s",
                    stoks[0], stoks[3], stoks[1], stoks[2],
                    stoks[4]);
            }
            for (i = 0; i < 5; i++)
                tfree(stoks[i]);
        }
    }
    del_models(modelsfound);

    return newcard;
}



/* do not modify oldcard address, insert everything after first line only */
void pspice_compat_a(struct card *oldcard)
{
    oldcard->nextcard = pspice_compat(oldcard->nextcard);
}


/**** LTSPICE to ngspice **************
 * add functions uplim, dnlim
 * Replace
 * D1 A K SDMOD
 * .MODEL SDMOD D (Roff=1000 Ron=0.7  Rrev=0.2  Vfwd=1  Vrev=10 Revepsilon=0.2
 *         Epsilon=0.2 Ilimit=7 Revilimit=7)
 * by
 * ad1 a k asdmod
 * .model asdmod sidiode(Roff=1000 Ron=0.7  Rrev=0.2  Vfwd=1  Vrev=10
 *         Revepsilon=0.2 Epsilon=0.2 Ilimit=7 Revilimit=7)
 * Remove '.backanno'
 */
struct card *ltspice_compat(struct card *oldcard)
{
    struct card *card, *newcard, *nextcard;
    struct vsmodels *modelsfound = NULL;
    int skip_control = 0;


    /* remove double braces only if not yet done in pspice_compat() */
    if (!newcompat.ps)
        rem_double_braces(oldcard);

    /* add funcs uplim, dnlim to beginning of deck */
    char *new_str =
            copy(".func uplim(x, pos, z) { min(x, pos - z) + (1 - "
                 "(min(max(0, x - pos + z), 2 * z) / 2 / z - 1)**2)*z }");
    newcard = insert_new_line(NULL, new_str, 1, 0, "internal");
    new_str = copy(".func dnlim(x, neg, z) { max(x, neg + z) - (1 - "
                   "(min(max(0, -x + neg + z), 2 * z) / 2 / z - 1)**2)*z }");
    nextcard = insert_new_line(newcard, new_str, 2, 0, "internal");
    new_str = copy(".func uplim_tanh(x, pos, z) { min(x, pos - z) + "
                   "tanh(max(0, x - pos + z) / z)*z }");
    nextcard = insert_new_line(nextcard, new_str, 3, 0, "internal");
    new_str = copy(".func dnlim_tanh(x, neg, z) { max(x, neg + z) - "
                   "tanh(max(0, neg + z - x) / z)*z }");
    nextcard = insert_new_line(nextcard, new_str, 4, 0, "internal");
    nextcard->nextcard = oldcard;

    /* remove .backanno, replace 'noiseless' by 'moisy=0' */
    for (card = nextcard; card; card = card->nextcard) {
        char* cut_line = card->line;
        if (ciprefix(".backanno", cut_line)) {
            *cut_line = '*';
        }
        else if (*cut_line == 'r') {
            char* noi = strstr(cut_line, "noiseless");
            /* only if 'noiseless' is an unconnected token */
            if (noi && isspace_c(noi[-1]) && (isspace_c(noi[9]) || !isprint_c(noi[9]))) {
                memcpy(noi, "noisy=0  ", 9);
            }
        }
    }

    /* replace
   * D1 A K SDMOD
   * .MODEL SDMOD D (Roff=1000 Ron=0.7  Rrev=0.2  Vfwd=1  Vrev=10
   *          Revepsilon=0.2 Epsilon=0.2 Ilimit=7 Revilimit=7)
   * by
   * a1 a k SDMOD
   * .model SDMOD sidiode(Roff=1000 Ron=0.7  Rrev=0.2  Vfwd=1  Vrev=10
   *            Revepsilon=0.2 Epsilon=0.2 Ilimit=7 Revilimit=7)
   * Do this if one of the parameters, which are uncommon to standard diode
   * model, has been found.

   * simple hierachy, as nested subcircuits are not allowed in PSPICE */

    /* first scan: find the d models, transform them and put them into a list
     */
    for (card = nextcard; card; card = card->nextcard) {
        char *str;
        static struct card *subcktline = NULL;
        static int nesting = 0;
        char *cut_line = card->line;
        if (*cut_line == '*' || *cut_line == '\0')
            continue;
        else if (ciprefix(".subckt", cut_line)) {
            subcktline = card;
            nesting++;
        }
        else if (ciprefix(".ends", cut_line))
            nesting--;

        else if (ciprefix(".model", card->line) &&
                search_plain_identifier(card->line, "d")) {
            if (search_plain_identifier(card->line, "roff") ||
                    search_plain_identifier(card->line, "ron") ||
                    search_plain_identifier(card->line, "rrev") ||
                    search_plain_identifier(card->line, "vfwd") ||
                    search_plain_identifier(card->line, "vrev") ||
                    search_plain_identifier(card->line, "revepsilon") ||
                    search_plain_identifier(card->line, "epsilon") ||
                    search_plain_identifier(card->line, "revilimit") ||
                    search_plain_identifier(card->line, "ilimit")) {
                char *modname;

                /* remove parameter 'noiseless' (the model is noiseless anyway) */
                char *nonoise = search_plain_identifier(card->line, "noiseless");
                if (nonoise) {
                    size_t iii;
                    for (iii = 0; iii < 9; iii++)
                        nonoise[iii] = ' ';
                }
                card->line = str = inp_remove_ws(card->line);
                str = nexttok(str); /* throw away '.model' */
                INPgetNetTok(&str, &modname, 0); /* model name */
                if (!ciprefix("d", str)) {
                    tfree(modname);
                    continue;
                }
                /* skip d */
                str++;
                /* we take all the existing parameters */
                char *newstr = copy(str);
                tfree(card->line);
                card->line = tprintf(".model a%s sidiode%s", modname, newstr);
                if (nesting > 0)
                    modelsfound = insert_new_model(
                            modelsfound, modname, subcktline->line);
                else
                    modelsfound =
                            insert_new_model(modelsfound, modname, "top");
                tfree(modname);
                tfree(newstr);
            }
        }
        else
            continue;
    }

    /* no need to continue if no d is found */
    if (!modelsfound)
        return newcard;

    /* second scan: find the diode instances d calling a simple diode model
     * and transform them */
    for (card = nextcard; card; card = card->nextcard) {
        static struct card *subcktline = NULL;
        static int nesting = 0;
        char *cut_line = card->line;
        if (*cut_line == '*')
            continue;
        if (*cut_line == '\0')
            continue;
        // exclude any command inside .control ... .endc
        if (ciprefix(".control", cut_line)) {
            skip_control++;
            continue;
        }
        else if (ciprefix(".endc", cut_line)) {
            skip_control--;
            continue;
        }
        else if (skip_control > 0) {
            continue;
        }
        if (ciprefix(".subckt", cut_line)) {
            subcktline = card;
            nesting++;
        }
        if (ciprefix(".ends", cut_line))
            nesting--;

        if (ciprefix("d", cut_line)) {
            /* check for the model name */
            int i;
            char *stoks[4];
            for (i = 0; i < 4; i++) {
                stoks[i] = gettok_node(&cut_line);
                if (stoks[i] == NULL) {
                    fprintf(stderr, "Error in line %d: buggy diode instance line\n    %s\n", card->linenum_orig, card->linesource);
                    fprintf(stderr, "At least 'Dxx n1 n2 d' is required.\n");
                    controlled_exit(EXIT_BAD);
                }
            }
            /* rewrite d line and replace it if a model is found */
            if ((nesting > 0) &&
                    find_a_model(modelsfound, stoks[3], subcktline->line)) {
                tfree(card->line);
                card->line = tprintf("a%s %s %s a%s",
                    stoks[0], stoks[1], stoks[2], stoks[3]);
            }
            /* if model is not within same subcircuit, search at top level */
            else if (find_a_model(modelsfound, stoks[3], "top")) {
                tfree(card->line);
                card->line = tprintf("a%s %s %s a%s",
                        stoks[0], stoks[1], stoks[2], stoks[3]);
            }
            for (i = 0; i < 4; i++)
                tfree(stoks[i]);
        }
    }
    del_models(modelsfound);

    return newcard;
}

/* do not modify oldcard address, insert everything after first line only */
void ltspice_compat_a(struct card *oldcard)
{
    oldcard->nextcard = ltspice_compat(oldcard->nextcard);
}
