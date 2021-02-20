/**********
Copyright 2021 The ngspice team.  All rights reserved.
Author: 2021 Holger Vogt
3-clause BSD
**********/

/* return the number of nodes */

#include "ngspice/ngspice.h"
#include "ngspice/inpdefs.h"
#include "ngspice/compatmode.h"

extern char* inp_remove_ws(char* s);

int getnumnodes(struct card *deck);


int
getnumnodes(struct card* deck)
{
    int skip_control = 0;
    struct card* card;

    for (card = deck; card; card = card->nextcard) {

        int i, j, k;
        char* name[12];
        char nam_buf[128];
        bool area_found = FALSE;
        char* curr_line = card->line;

        /* exclude any command inside .control ... .endc */
        if (ciprefix(".control", curr_line)) {
            skip_control++;
            continue;
        }
        else if (ciprefix(".endc", curr_line)) {
            skip_control--;
            continue;
        }
        else if (skip_control > 0) {
            continue;
        }

        switch (*curr_line) {
        case '*':
        case ' ':
        case '\t':
        case 'x':
        case '$':
            continue;
            break;
        }

        /* catch all models, including their scope */
        if (ciprefix(".model", curr_line))
        { }

        switch (*curr_line) {
        case '.':
            continue;
            break;

        case 'b':
        case 'c':
        case 'd':
        case 'e':
        case 'f':
        case 'g':
        case 'h':
        case 'i':
        case 'l':
        case 'r':
        case 'v':
        case 'w':
            card->nunodes = 2;
            break;

        case 'j':
        case 'u':
        case 'z':
            card->nunodes = 3;
            break;

        case 'o':
        case 's':
        case 't':
        case 'y':
            card->nunodes = 4;
            break;

        case 'm': /* recognition of 4, 5, 6, or 7 nodes for SOI devices needed
                   */
        {
            i = 0;
            char* cc, * ccfree;
            /* we have 3 or 4 terminals with compatmode ps */
            if (newcompat.ps) {
                char* token = curr_line;
                token = nexttok(token); /* skip device name */
                token = nexttok(token); /* skip drain node */
                token = nexttok(token); /* skip gate node */
                token = nexttok(token); /* skip source node */
                if (token && *token == '[') /* bulk node starts with [ */
                    card->nunodes = 4;
                else
                    card->nunodes = 3;
                break;
            }
            cc = copy(curr_line);
            /* required to make m= 1 a single token m=1 */
            ccfree = cc = inp_remove_ws(cc);
            /* find the first token with "off" or "=" in the line*/
            while ((i < 20) && (*cc != '\0')) {
                char* inst = gettok_instance(&cc);
                strncpy(nam_buf, inst, sizeof(nam_buf) - 1);
                txfree(inst);
                if (strstr(nam_buf, "off") || strchr(nam_buf, '=') || strstr(nam_buf, "tnodeout") || strstr(nam_buf, "thermal"))
                    break;
                i++;
            }
            tfree(ccfree);
            card->nunodes = i - 2;
            break;
        }
        case 'p': /* recognition of up to 100 cpl nodes */
            i = j = 0;
            /* find the last token in the line*/
            while ((i < 100) && (*curr_line != '\0')) {
                char* tmp_inst = gettok_instance(&curr_line);
                strncpy(nam_buf, tmp_inst, 32);
                tfree(tmp_inst);
                if (strchr(nam_buf, '='))
                    j++;
                i++;
            }
            if (i == 100)
                return 0;
            card->nunodes = i - j - 2;
            break;
        case 'q': /* recognition of 3, 4 or 5 terminal bjt's needed */
            /* QXXXXXXX NC NB NE <NS> <NT> MNAME <AREA> <OFF> <IC=VBE, VCE>
             * <TEMP=T> */
             /* 12 tokens maximum */
        {
            char* cc, * ccfree;
            i = j = 0;
            cc = copy(curr_line);
            /* required to make m= 1 a single token m=1 */
            ccfree = cc = inp_remove_ws(cc);
            while ((i < 12) && (*cc != '\0')) {
                char* comma;
                name[i] = gettok_instance(&cc);
                if (strstr(name[i], "off") || strchr(name[i], '='))
                    j++;
#ifdef CIDER
                if (strstr(name[i], "save") || strstr(name[i], "print"))
                    j++;
#endif
                /* If we have IC=VBE, VCE instead of IC=VBE,VCE we need to inc
                 * j */
                if ((comma = strchr(name[i], ',')) != NULL &&
                    (*(++comma) == '\0'))
                    j++;
                /* If we have IC=VBE , VCE ("," is a token) we need to inc j
                 */
                if (eq(name[i], ","))
                    j++;
                i++;
            }
            tfree(ccfree);
            i--;
            area_found = FALSE;
            for (k = i; k > i - j - 1; k--) {
                bool only_digits = TRUE;
                char* nametmp = name[k];
                /* MNAME has to contain at least one alpha character. AREA may
                   be assumed if we have a token with only digits, and where
                   the previous token does not end with a ',' */
                while (*nametmp) {
                    if (isalpha_c(*nametmp) || (*nametmp == ','))
                        only_digits = FALSE;
                    nametmp++;
                }
                if (only_digits && (strchr(name[k - 1], ',') == NULL))
                    area_found = TRUE;
            }
            for (k = i; k >= 0; k--)
                tfree(name[k]);
            if (area_found) {
                card->nunodes = i - j - 2;
            }
            else {
                card->nunodes = i - j - 1;
            }
            break;
        }

        case 'k':
            card->nunodes = 0;
            break;

        default:
            card->nunodes = 0;
            break;
        }
    }
	return 0;
}

