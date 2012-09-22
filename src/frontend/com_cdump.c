/* Command cdump: dump the control structure to the console output */

#include "ngspice/ngspice.h"
#include <stdio.h>

#include "ngspice/wordlist.h"
#include "control.h"
#include "ngspice/cpextern.h"
#include "ngspice/fteext.h"
#include "ngspice/cktdefs.h"
#include "com_cdump.h"

static int indent;


static void
tab(int num)
{
    int i;

    for (i = 0; i < num; i++)
        putc(' ', cp_out);
}


static void
dodump(struct control *cc)
{
    struct control *tc;

    switch (cc->co_type) {
    case CO_UNFILLED:
        tab(indent);
        fprintf(cp_out, "(unfilled)\n");
        break;
    case CO_STATEMENT:
        tab(indent);
        wl_print(cc->co_text, cp_out);
        putc('\n', cp_out);
        break;
    case CO_WHILE:
        tab(indent);
        fprintf(cp_out, "while ");
        wl_print(cc->co_cond, cp_out);
        putc('\n', cp_out);
        indent += TABINDENT;
        for (tc = cc->co_children; tc; tc = tc->co_next)
            dodump(tc);
        indent -= TABINDENT;
        tab(indent);
        fprintf(cp_out, "end\n");
        break;
    case CO_REPEAT:
        tab(indent);
        fprintf(cp_out, "repeat ");
        if (cc->co_numtimes != -1)
            fprintf(cp_out, "%d (%d left to do)\n", cc->co_numtimes, cc->co_timestodo); /* CDHW */
        else
            putc('\n', cp_out);
        indent += TABINDENT;
        for (tc = cc->co_children; tc; tc = tc->co_next)
            dodump(tc);
        indent -= TABINDENT;
        tab(indent);
        fprintf(cp_out, "end\n");
        break;
    case CO_DOWHILE:
        tab(indent);
        fprintf(cp_out, "dowhile ");
        wl_print(cc->co_cond, cp_out);
        putc('\n', cp_out);
        indent += TABINDENT;
        for (tc = cc->co_children; tc; tc = tc->co_next)
            dodump(tc);
        indent -= TABINDENT;
        tab(indent);
        fprintf(cp_out, "end\n");
        break;
    case CO_IF:
        tab(indent);
        fprintf(cp_out, "if ");
        wl_print(cc->co_cond, cp_out);
        putc('\n', cp_out);
        indent += TABINDENT;
        for (tc = cc->co_children; tc; tc = tc->co_next)
            dodump(tc);
        indent -= TABINDENT;
        tab(indent);
        fprintf(cp_out, "end\n");
        break;
    case CO_FOREACH:
        tab(indent);
        fprintf(cp_out, "foreach %s ", cc->co_foreachvar);
        wl_print(cc->co_text, cp_out);
        putc('\n', cp_out);
        indent += TABINDENT;
        for (tc = cc->co_children; tc; tc = tc->co_next)
            dodump(tc);
        indent -= TABINDENT;
        tab(indent);
        fprintf(cp_out, "end\n");
        break;
    case CO_BREAK:
        tab(indent);
        if (cc->co_numtimes != 1)
            fprintf(cp_out, "break %d\n", cc->co_numtimes);
        else
            fprintf(cp_out, "break\n");
        break;
    case CO_CONTINUE:
        tab(indent);
        if (cc->co_numtimes != 1)
            fprintf(cp_out, "continue %d\n",
                    cc->co_numtimes);
        else
            fprintf(cp_out, "continue\n");
        break;
    case CO_LABEL:
        tab(indent);
        fprintf(cp_out, "label %s\n", cc->co_text->wl_word);
        break;
    case CO_GOTO:
        tab(indent);
        fprintf(cp_out, "goto %s\n", cc->co_text->wl_word);
        break;
    default:
        tab(indent);
        fprintf(cp_out, "bad type %d\n", cc->co_type);
        break;
    }
}


void
com_cdump(wordlist *wl)
{
    struct control *c;
    NG_IGNORE(wl);

    indent = 0;
    for (c = control[stackp]; c; c = c->co_next)
        dodump(c);
}


/* dump circuit matrix to stdout or file */
void
com_mdump(wordlist *wl)
{
    CKTcircuit *ckt = NULL;
    char *s;

    if (!ft_curckt || !ft_curckt->ci_ckt) {
        fprintf(cp_err, "Error: no circuit loaded.\n");
        return;
    }

    ckt = ft_curckt->ci_ckt;

    if (ckt->CKTmatrix)
        if (wl == NULL) {
            SMPprint(ckt->CKTmatrix , NULL);
        } else {
            s = cp_unquote(wl->wl_word);
            SMPprint(ckt->CKTmatrix , s);
        }
    else
        fprintf(cp_err, "Error: no matrix available.\n");
}


/* dump circuit matrix RHS to stdout or file */
void
com_rdump(wordlist *wl)
{
    CKTcircuit *ckt = NULL;
    char *s;

    if (!ft_curckt || !ft_curckt->ci_ckt) {
        fprintf(cp_err, "Error: no circuit loaded.\n");
        return;
    }

    ckt = ft_curckt->ci_ckt;

    if ((ckt->CKTmatrix) && (ckt->CKTrhs))
        if (wl == NULL) {
            SMPprintRHS(ckt->CKTmatrix , NULL, ckt->CKTrhs, ckt->CKTirhs);
        } else {
            s = cp_unquote(wl->wl_word);
            SMPprintRHS(ckt->CKTmatrix , s, ckt->CKTrhs, ckt->CKTirhs);
        }
    else
        fprintf(cp_err, "Error: no matrix or RHS available.\n");
}
