/* ngspice file inpc_probe.c
   Copyright Holger Vogt 2021
   License: BSD 3-clause
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpextern.h"
#include "ngspice/dstring.h"
#include "numparam/general.h"
#include "ngspice/hash.h"
#include "ngspice/inpdefs.h"
#include "ngspice/wordlist.h"
#include "ngspice/stringskip.h"

void inp_probe(struct card* card);
void modprobenames(INPtables* tab);

extern struct card* insert_new_line(
    struct card* card, char* line, int linenum, int linenum_orig);
extern int get_number_terminals(char* c);
extern char* search_plain_identifier(char* str, const char* identifier);

static char* get_terminal_name(char* element, char* numberstr, NGHASHPTR instances);
static char* get_terminal_number(char* element, char* numberstr);
static int setallvsources(struct card* tmpcard, NGHASHPTR instances, char* instname, int numnodes, bool haveall, bool power);


/* Find any line starting with .probe: assemble all parameters like
   <empty>     add V(0) current measure sources to all device nodes in addition to .save all
   alli        add V(0) current measure sources to all device nodes in addition to .save all
   I(R1)       add V(0) measure source to node 1 of a two-terminal device R1
   I(Q1)       add V(0) measure sources to all nodes of a multi-terminal device Q1
   I(M4,3)     add V(0) measure source to node 3 of a multi-terminal device M4
   Vd(R1)     add E source inputs to measure voltage difference to both terminals of a two-terminal device R1
   Vd(X1:2:3) add E source inputs to terminals 2 and 3 of a multi-terminal device X1
   Vd(X1:2,X2:3) add E source inputs to terminal 2 of a multi-terminal device X1 and to terminal 3 of X2

   Seach the netlist for the devices found in the .probe parameters.
   Check the number of terminals for each device. Add 0V voltage sources
   in series to each of the named terminals. Add E sources to the differential
   voltage probes.

   */
void inp_probe(struct card* deck)
{
    struct card *card;
    int skip_control = 0;
    int skip_subckt = 0;
    wordlist* probes = NULL, *probeparams = NULL, *wltmp, *allsaves = NULL;
    bool haveall = FALSE, havedifferential = FALSE, t = TRUE, havesave = FALSE;
    NGHASHPTR instances;   /* instance hash table */
    int ee = 0; /* serial number for sources */

    for (card = deck; card; card = card->nextcard) {
        /* get the .probe netlist lines, comment them out */
        if (ciprefix(".probe", card->line)) {
            probes = wl_cons(card->line, probes);
            *(card->line) = '*';
        }
    }
    /* no .probe command */
    if (probes == NULL)
        return;

    /* check for '.save' and (in a .control section) 'save'.
       If not found, add '.save all' */
    for (card = deck; card; card = card->nextcard) {
        /* find .save */
        if (ciprefix(".save", card->line)) {
            havesave = TRUE;
            break;
        }
        /* exclude any command inside .control ... .endc */
        else if (ciprefix(".control", card->line)) {
            skip_control++;
            continue;
        }
        else if (ciprefix(".endc", card->line)) {
            skip_control--;
            continue;
        }
        else if (skip_control > 0) {
            if (ciprefix("save ", card->line)) {
                havesave = TRUE;
                break;
            }
        }
    }
    skip_control = 0;

    if (!havesave) {
        char* vline = copy(".save all");
        deck = insert_new_line(deck, vline, 0, 0);
    }

    /* set a variable if .probe command is given */
    cp_vset("probe_is_given", CP_BOOL, &t);

    /* Assemble all .probe parameters in a wordlist 'probeparams' */
    for (wltmp = probes; wltmp; wltmp = wltmp->wl_next) {
        char* nextnode;
        char* tmpstr = wltmp->wl_word;
        /* skip *probe */
        tmpstr = nexttok(tmpstr);

        if (*tmpstr == '\0') {
            fprintf(stderr, "Note: Empty .probe command, treated as .probe alli\n");
            haveall = TRUE;
            continue;
        }

        /* set haveall and remove 'alli' token */
        char *allistr = search_plain_identifier(tmpstr, "alli");
        if (allistr) {
            haveall = TRUE;
            memcpy(allistr, "    ", 4);
        }

        tmpstr = skip_ws(tmpstr);
        if (!strchr("vipVIP", *tmpstr)) {
            fprintf(stderr, "Warning: Strange parameter in line %s, ingnored\n", wltmp->wl_word);
            tmpstr = nexttok(tmpstr);
        }
        nextnode = gettok_char(&tmpstr, ')', TRUE, FALSE);

        if (haveall == FALSE && !nextnode) {
            fprintf(stderr, "Warning: Strange parameter in line %s, ingnored\n", wltmp->wl_word);
            continue;
        }

        while (nextnode && (*nextnode != '\0')) {
            probeparams = wl_cons(nextnode, probeparams);

            if (ciprefix("vd(", nextnode)) {
                havedifferential = TRUE;
            }

            nextnode = gettok_char(&tmpstr, ')', TRUE, FALSE);
        }
    }
    /* don't free the wl_word, they belong to the cards */
    tfree(probes);

    /* Set up the hash table for all instances (instance name is key, data
       is the storage location of the card) */
    instances = nghash_init(100);
    nghash_unique(instances, TRUE);

    for (card = deck; card; card = card->nextcard) {
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
        /* exclude any device or command inside .subckt ... .ends */
        if (ciprefix(".subckt", curr_line)) {
            skip_subckt++;
            continue;
        }
        else if (ciprefix(".ends", curr_line)) {
            skip_subckt--;
            continue;
        }
        else if (skip_subckt > 0) {
            continue;
        }
        if (*curr_line == '*')
            continue;
        if (*curr_line == '.')
            continue;
        if (*curr_line == '\0')
            continue;

        /* here we should go on with only true device instances at top level.
           Put all instance names as key into a hash table, with the address as parameter. */
           /* Get the instance name as key */
        char* instname = gettok_instance(&curr_line);
        if (!instname)
            continue;
        nghash_insert(instances, instname, card);
    }

    if (haveall || probeparams == NULL) {
        /* Either we have 'alli' among the .probe parameters, or we have a single .probe command without parameters:
           Add current measure voltage sources for all devices, add differential E sources only for selected devices. */
        int numnodes, i;

        for (card = deck; card; card = card->nextcard) {

            char* curr_line = card->line;
            struct card* prevcard = NULL;

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
            /* exclude any device or command inside .subckt ... .ends */
            if (ciprefix(".subckt", curr_line)) {
                skip_subckt++;
                continue;
            }
            else if (ciprefix(".ends", curr_line)) {
                skip_subckt--;
                continue;
            }
            else if (skip_subckt > 0) {
                continue;
            }
            if (*curr_line == '*')
                continue;
            if (*curr_line == '.')
                continue;
            if (*curr_line == '\0')
                continue;

            char* instname = gettok_instance(&curr_line);
            if (!instname)
                continue;

            /* select elements not in need of a measure Vsource */
            if (strchr("ehvk", *instname))
                continue;

            /* exclude a devices (code models may have special characters in their instance line.
            digital nodes should not get V sources in series anyway.) */
            if ('a' == *instname)
                continue;

            /* exclude x devices (Subcircuits may contain digital a devices,
            and digital nodes should not get V sources in series anyway.),
            when probe_alli_nox is set in .spiceinit. */
            if ('x' == *instname && cp_getvar("probe_alli_nox", CP_BOOL, NULL, 0))
                continue;

            /* special treatment for controlled current sources and switches:
               We have three or four tokens until model name, but only the first 2 are relevant nodes. */
            if (strchr("fgsw", *instname))
                numnodes = 2;
            else
                numnodes = get_number_terminals(card->line);

            char* thisline = curr_line;
            prevcard = card;
            /* all elements with 2 nodes: add a voltage source to the second node in the elements line */
            if (numnodes == 2) {
                char *strnode1, *strnode2, *nodename2;
                strnode1 = gettok(&thisline);
                strnode2 = gettok(&thisline);

                if (!strnode2 || *strnode2 == '\0') {
                    fprintf(stderr, "Warning: Cannot read 2 nodes in line %s\n", curr_line);
                    fprintf(stderr, "    Instance not ready for .probe command\n");
                    tfree(strnode1);
                    tfree(strnode2);
                    continue;
                }

                nodename2 = get_terminal_name(instname, "2", instances);

                char* newnode = tprintf("probe_int_%s_%s", strnode2, instname);
                char* vline = tprintf("vcurr_%s:%s_%s %s %s 0", instname, nodename2, strnode2, newnode, strnode2);
                char *newline = tprintf("%s %s %s %s", instname, strnode1, newnode, thisline);

                char* nodesaves = tprintf("%s#branch", instname);
                allsaves = wl_cons(nodesaves, allsaves);

                tfree(card->line);
                card->line = newline;

                card = insert_new_line(card, vline, 0, 0);

                tfree(strnode1);
                tfree(strnode2);
                tfree(newnode);
                tfree(nodename2);

            }
            else {
                char* nodename;
                DS_CREATE(dnewline, 200);
                sadd(&dnewline, instname);
                cadd(&dnewline, ' ');
                for (i = 1; i <= numnodes; i++) {
                    char* thisnode;
                    char nodebuf[20];
                    thisnode = gettok(&thisline);
                    if (!thisnode || *thisnode == '\0') {
                        fprintf(stderr, "Warning: Cannot read node %d in line %s\n", i, curr_line);
                        fprintf(stderr, "    Instance not ready for .probe command\n");
                        tfree(thisnode);
                        continue;
                    }
                    char* newnode = tprintf("probe_int_%s_%s_%d", thisnode, instname, i);
                    sadd(&dnewline, newnode);
                    cadd(&dnewline, ' ');
                    /* to make the nodes unique */
                    snprintf(nodebuf, 12, "%d", i);
                    nodename = get_terminal_name(instname, nodebuf, instances);
                    if (!nodename || *nodename == '\0') {
                        fprintf(stderr, "Warning: Cannot find node name %d in line %s\n", i, curr_line);
                        fprintf(stderr, "    Instance not ready for .probe command\n");
                        tfree(thisnode);
                        continue;
                    }
                    char* vline = tprintf("vcurr_%s:%s:%s_%s %s %s 0", instname, nodename, thisnode, nodebuf, thisnode, newnode);
                    card = insert_new_line(card, vline, 0, 0);
                    /* special for KiCad: add shunt resistor if thisnode contains 'unconnected' */
                    if (*instname == 'x' && strstr(thisnode, "unconnected")) {
                        char *rline = tprintf("R%s %s 0 1e15", thisnode, thisnode);
                        card = insert_new_line(card, rline, 0, 0);
                    }
                    char* nodesaves = tprintf("%s:%s#branch", instname, nodename);
                    allsaves = wl_cons(nodesaves, allsaves);

                    tfree(newnode);
                    tfree(nodename);
                }
                sadd(&dnewline, thisline);
                tfree(prevcard->line);
                prevcard->line = copy(ds_get_buf(&dnewline));
                ds_free(&dnewline);
            }
            if (allsaves) {
                allsaves = wl_cons(copy(".save"), allsaves);
                char* newline = wl_flatten(allsaves);
                wl_free(allsaves);
                allsaves = NULL;
                card = insert_new_line(card, newline, 0, 0);
            }
        }
    }

    if (probeparams) {
        /* There are .probe with parameters:
           Add current measure voltage sources only for the selected devices.
           Add differential probes only if 'all' had been found. */
        for (wltmp = probeparams; wltmp; wltmp = wltmp->wl_next) {
            char *tmpstr = wltmp->wl_word;
            ee++;
            /* check for differential voltage probes:
               v(nR1) voltage at node named nR1
               vd(R1) voltage across a two-terminal device named R1
               vd(m4:1:0) voltage at instance node 1 of device m4
               vd(m4:1:3) voltage between instance nodes 1 and 3 of device m4
               vd(m4:1, m5:3) voltage between instance node 1 of device m4 and node 3 of device m5 */
               /* no nodes after first token: must be a node itself */

            /* v(nodename), voltage at node named nodename */
            if (ciprefix("v(", tmpstr)) {
                char* instname1 = gettok_char(&tmpstr, ')', TRUE, FALSE);
                allsaves = wl_cons(copy(instname1), allsaves);
                continue;
            }
            /* vd(R1), vd(R1:1,R2:1), vd(MN4:d:s), vd(QN4:1:3), vd(nodename,MN5:d), vd(nodename1,nodename2) */
            else  if (ciprefix("vd(", tmpstr)) {
                char* instname1, *instname2;
                int numnodes1, numnodes2;
                struct card* tmpcard1;

                /* skip vd_ */
                tmpstr += 3;

                /* vd(R1)
                   vd(nodename1,nodename2) */
                if (!strchr(tmpstr, ':')) {
                    char* newline = NULL, *strnode1, *strnode2, *tmpstr2;
                    tmpstr2 = tmpstr;
                    strnode1 = gettok_char(&tmpstr2, ',', FALSE, FALSE);
                    if (strnode1) {
                        tmpstr2++; /* beyond ',' */
                        strnode2 = gettok_char(&tmpstr2, ')', FALSE, FALSE);
                        if (!strnode2) {
                        }
                        else {
                            newline = tprintf("Ediff%d_nodes vd_%s:%s 0 %s %s 1", ee, strnode1, strnode2, strnode1, strnode2);

                            char* nodesaves = tprintf("vd_%s:%s", strnode1, strnode2);
                            allsaves = wl_cons(nodesaves, allsaves);
                            tfree(strnode1);
                            tfree(strnode2);
                            tmpcard1 = deck->nextcard;
                            tmpcard1 = insert_new_line(tmpcard1, newline, 0, 0);
                        }
                        continue;
                    }

                    instname1 = gettok_char(&tmpstr, ')', FALSE, FALSE);
                    tmpcard1 = nghash_find(instances, instname1);
                    if (!tmpcard1) {
                        fprintf(stderr, "Warning: Could not find the instance line for %s,\n   .probe %s will be ignored\n", instname1, wltmp->wl_word);
                        tfree(instname1);
                        continue;
                    }
                    char* thisline = tmpcard1->line;
                    numnodes1 = get_number_terminals(thisline);
                    if (numnodes1 != 2) {
                        fprintf(stderr, "Warning: Instance %s has more than 2 nodes,\n   .probe %s will be ignored\n", instname1, wltmp->wl_word);
                        tfree(instname1);
                        continue;
                    }
                    thisline = nexttok(thisline); /* skip instance name */
                    strnode1 = gettok(&thisline);
                    strnode2 = gettok(&thisline);
                    if (!strnode2 || *strnode2 == '\0') {
                        fprintf(stderr, "Warning: Cannot read 2 nodes in line %s\n", tmpcard1->line);
                        fprintf(stderr, "    Instance not ready for .probe command\n");
                        tfree(strnode1);
                        tfree(strnode2);
                        continue;
                    }
                    newline = tprintf("Ediff%d_%s vd_%s 0 %s %s 1", ee, instname1, instname1, strnode1, strnode2);

                    char* nodesaves = tprintf("vd_%s", instname1);
                    allsaves = wl_cons(nodesaves, allsaves);
                    tfree(strnode1);
                    tfree(strnode2);
                    tmpcard1 = insert_new_line(tmpcard1, newline, 0, 0);
                    continue;
                }
                /* node containing ':' 
                vd(R1:1,R2:2)
                vd(M4:1:3)
                vd(m5:d:s)*/
                else {
                    char* tmpstr2, *nodename1, *nodename2;
                    struct card* tmpcard2;
                    tmpstr2 = tmpstr;
                    instname1 = gettok_char(&tmpstr, ':', FALSE, FALSE);
                    if (!instname1) {
                        fprintf(stderr, "Warning: Cannot read instance name in %s, ignored\n", tmpstr);
                        continue;
                    }
                    tmpcard1 = nghash_find(instances, instname1);
                    if (!tmpcard1) {
                        fprintf(stderr, "Warning: Could not find the instance line for %s,\n   .probe %s will be ignored\n", instname1, wltmp->wl_word);
                        tfree(instname1);
                        continue;
                    }
                    char* thisline = tmpcard1->line;
                    numnodes1 = get_number_terminals(thisline);
                    tmpstr++;
                    tmpstr2 = tmpstr;
                    nodename1 =  gettok_char(&tmpstr2, ',', FALSE, FALSE);
                    if (nodename1) {
                        /* vd(R1:1,R2:2) */
                        int nodenum1, nodenum2, i;
                        char* ptr, *node1, *node2, *strnode1, *strnode2;
                        bool err = FALSE;

                        tmpstr2++; /* beyond ',' */
                        instname2 = gettok_char(&tmpstr2, ':', FALSE, FALSE);
                        if (!instname2) {
                            fprintf(stderr, "Warning: Cannot read instance name in %s, ignored\n", tmpstr);
                            tfree(nodename1);
                            continue;
                        }
                        tmpstr2++; /* beyond ':' */
                        tmpcard2 = nghash_find(instances, instname2);
                        if (!tmpcard2) {
                            fprintf(stderr, "Warning: Could not find the instance line for %s,\n   .probe %s will be ignored\n", instname2, wltmp->wl_word);
                            tfree(instname2);
                            tfree(nodename1);
                            continue;
                        }
                        char* thisline2 = tmpcard2->line;
                        numnodes2 = get_number_terminals(thisline2);
                        nodename2 = gettok_char(&tmpstr2, ')', FALSE, FALSE);
                        if (!nodename2) {
                            fprintf(stderr, "Warning: Could not find the second node name for %s,\n   .probe %s will be ignored\n", instname2, wltmp->wl_word);
                            tfree(instname2);
                            tfree(nodename1);
                            continue;
                        }
                        /* nodenames may be numbers or characters, we always need the numbers */
                        node1 = get_terminal_number(instname1, nodename1);
                        if (eq(node1, "0")) {
                            fprintf(stderr, "Warning: Node %s is not available for device %s,\n   .probe %s will be ignored\n", node1, instname1, wltmp->wl_word);
                            continue;
                        }
                        node2 = get_terminal_number(instname2, nodename2);
                        if (eq(node2, "0")) {
                            fprintf(stderr, "Warning: Node %s is not available for device %s,\n   .probe %s will be ignored\n", node1, instname2, wltmp->wl_word);
                            continue;
                        }

                        /* nodes are numbered 1, 2, 3, ... */
                        nodenum1 = (int)strtol(node1, &ptr, 10);
                        nodenum2 = (int)strtol(node2, &ptr, 10);

                        if (nodenum1 > numnodes1) {
                            fprintf(stderr, "Warning: There are only %d nodes available for %s,\n   .probe %s will be ignored!\n", numnodes1, instname1, wltmp->wl_word);
                            continue;
                        }
                        if (nodenum2 > numnodes2) {
                            fprintf(stderr, "Warning: There are only %d nodes available for %s,\n   .probe %s will be ignored!\n", numnodes2, instname2, wltmp->wl_word);
                            continue;
                        }
                        if (nodenum1 == nodenum2 && eq(instname1, instname2)) {
                            fprintf(stderr, "Warning: Duplicate node numbers and instances,\n   .probe %s will be ignored!\n", wltmp->wl_word);
                            continue;
                        }
                        /* if node1 is the 0 node*/
                        if (nodenum1 == 0) {
                            strnode1 = copy("0");
                        }
                        else {
                            /* skip instance and leading nodes not wanted */
                            for (i = 0; i < nodenum1; i++) {
                                thisline = nexttok(thisline);
                                if (*thisline == '\0') {
                                    fprintf(stderr, "Warning: node number %d not available for instance %s, ignored!\n", nodenum1, instname1);
                                    err = TRUE;
                                    break;
                                }
                            }
                            if (err)
                                continue;

                            strnode1 = gettok(&thisline);
                        }

                        /* if node2 is the 0 node*/
                        if (nodenum2 == 0) {
                            strnode2 = copy("0");
                        }
                        else {
                            /* skip instance and leading nodes not wanted */
                            for (i = 0; i < nodenum2; i++) {
                                thisline2 = nexttok(thisline2);
                                if (*thisline2 == '\0') {
                                    fprintf(stderr, "Warning: node number %d not available for instance %s, ignored!\n", nodenum2, instname2);
                                    err = TRUE;
                                    break;
                                }
                            }
                            if (err)
                                continue;

                            strnode2 = gettok(&thisline2);
                        }

                        /* preserve the 0 node */
                        if (*node1 == '0') {
                            nodename1 = copy("0");
                        }
                        else {
                            if (*node1 != '\0' && atoi(node1) == 0) {
                                char* nn = get_terminal_number(instname1, node1);
                                tfree(node1);
                                node1 = copy(nn);
                            }
                            nodename1 = get_terminal_name(instname1, node1, instances);
                        }

                        /* preserve the 0 node */
                        if (*node2 == '0') {
                            nodename2 = copy("0");
                        }
                        else {
                            if (*node2 != '\0' && atoi(node2) == 0) {
                                char* nn = get_terminal_number(instname2, node2);
                                tfree(node2);
                                node2 = copy(nn);
                            }
                            nodename2 = get_terminal_name(instname2, node2, instances);
                        }
                        char *newline = tprintf("Ediff%d_%s_%s vd_%s:%s_%s:%s 0 %s %s 1", ee, instname1, instname2, instname1, nodename1, instname2, nodename2, strnode1, strnode2);
                        char* nodesaves = tprintf("vd_%s:%s_%s:%s", instname1, nodename1, instname2, nodename2);
                        allsaves = wl_cons(nodesaves, allsaves);
                        tmpcard1 = insert_new_line(tmpcard1, newline, 0, 0);
                        tfree(strnode1);
                        tfree(strnode2);
                        tfree(nodename1);
                        tfree(nodename2);

                    }
                    else {
                        /* vd(M4:1:3) */
                        int nodenum1, nodenum2, i;
                        char* ptr, * node1, * node2, * strnode1, * strnode2;
                        bool err = FALSE;
                        char* thisline2 = thisline;

                        tmpstr2 = tmpstr;
                        nodename1 =  gettok_char(&tmpstr2, ':', FALSE, FALSE);
                        if (!nodename1) {
                            fprintf(stderr, "Warning: Could not find the first node name for %s,\n   .probe %s will be ignored\n", instname1, wltmp->wl_word);
                            tfree(instname1);
                            continue;
                        }
                        tmpstr2++;
                        nodename2 =  gettok_char(&tmpstr2, ')', FALSE, FALSE);
                        if (!nodename1 || !nodename2) {
                            fprintf(stderr, "Warning: Could not find the second node name for %s,\n   .probe %s will be ignored\n", instname1, wltmp->wl_word);
                            tfree(instname1);
                            tfree(nodename1);
                            continue;
                        }
                        /* nodenames may be numbers or characters, we always need the numbers */
                        node1 = get_terminal_number(instname1, nodename1);
                        node2 = get_terminal_number(instname1, nodename2);
                        if (eq(node1, "0") && eq(node2, "0")) {
                            fprintf(stderr, "Warning: Either first or second node have to be non-zero,\n   .probe %s will be ignored\n", wltmp->wl_word);
                            continue;
                        }

                        /* nodes are numbered 1, 2, 3, ... */
                        nodenum1 = (int)strtol(node1, &ptr, 10);
                        nodenum2 = (int)strtol(node2, &ptr, 10);

                        if (nodenum1 > numnodes1) {
                            fprintf(stderr, "Warning: There are only %d nodes available for %s,\n   .probe %s will be ignored!\n", numnodes1, instname1, wltmp->wl_word);
                            continue;
                        }
                        if (nodenum2 > numnodes1) {
                            fprintf(stderr, "Warning: There are only %d nodes available for %s,\n   .probe %s will be ignored!\n", numnodes1, instname1, wltmp->wl_word);
                            continue;
                        }
                        if (nodenum1 == nodenum2) {
                            fprintf(stderr, "Warning: Duplicate node numbers,\n   .probe %s will be ignored!\n", wltmp->wl_word);
                            continue;
                        }
                        /* if node1 is the 0 node*/
                        if (nodenum1 == 0) {
                            strnode1 = copy("0");
                        }
                        else {
                            /* skip instance and leading nodes not wanted */
                            for (i = 0; i < nodenum1; i++) {
                                thisline2 = nexttok(thisline2);
                                if (*thisline2 == '\0') {
                                    fprintf(stderr, "Warning: node number %d not available for instance %s, ignored!\n", nodenum1, instname1);
                                    err = TRUE;
                                    break;
                                }
                            }
                            if (err)
                                continue;

                            strnode1 = gettok(&thisline2);
                        }

                        thisline2 = thisline;
                        /* if node2 is the 0 node*/
                        if (nodenum2 == 0) {
                            strnode2 = copy("0");
                        }
                        else {
                            /* skip instance and leading nodes not wanted */
                            for (i = 0; i < nodenum2; i++) {
                                thisline2 = nexttok(thisline2);
                                if (*thisline2 == '\0') {
                                    fprintf(stderr, "Warning: node number %d not available for instance %s, ignored!\n", nodenum2, instname1);
                                    err = TRUE;
                                    break;
                                }
                            }
                            if (err)
                                continue;

                            strnode2 = gettok(&thisline2);
                        }

                        /* preserve the 0 node */
                        if (*node1 == '0') {
                            nodename1 = copy("0");
                        }
                        else {
                            if (*node1 != '\0' && atoi(node1) == 0) {
                                char* nn = get_terminal_number(instname1, node1);
                                tfree(node1);
                                node1 = copy(nn);
                            }
                            nodename1 = get_terminal_name(instname1, node1, instances);
                        }

                        /* preserve the 0 node */
                        if (*node2 == '0') {
                            nodename2 = copy("0");
                        }
                        else {
                            if (*node2 != '\0' && atoi(node2) == 0) {
                                char* nn = get_terminal_number(instname1, node2);
                                tfree(node2);
                                node2 = copy(nn);
                            }
                            nodename2 = get_terminal_name(instname1, node2, instances);
                        }
                        char* newline = tprintf("Ediff%d_%s vd_%s:%s:%s 0 %s %s 1", ee, instname1, instname1, nodename1, nodename2, strnode1, strnode2);
                        char* nodesaves = tprintf("vd_%s:%s:%s", instname1, nodename1, nodename2);
                        allsaves = wl_cons(nodesaves, allsaves);
                        tmpcard1 = insert_new_line(tmpcard1, newline, 0, 0);
                        tfree(strnode1);
                        tfree(strnode2);
                        tfree(nodename1);
                        tfree(nodename2);
                    }
                }
            }

            /* No .probe parameter 'alli' (has been treated already), but dedicated current probes requested */
            else if (!haveall && ciprefix("i(", tmpstr)) {
                char* instname, * node1 = NULL, *nodename1;
                struct card* tmpcard;
                int numnodes;

                tmpstr += 2;

                /* Replace a : by , to enable i(mn1:s) equivalent to i(mn1,s) */
                char* co = strchr(tmpstr, ':');
                if (co) {
                    *co = ',';
                }

                instname = gettok_noparens(&tmpstr);
                tmpcard = nghash_find(instances, instname);
                if (!tmpcard) {
                    fprintf(stderr, "Warning: Could not find the instance line for %s,\n   .probe %s will be ignored\n", instname, wltmp->wl_word);
                    continue;
                }
                char* thisline = tmpcard->line;

                /* special treatment for controlled current sources and switches:
                   We have three or four tokens until model name, but only the first 2 are relevant nodes. */
                if (strchr("fgsw", *instname))
                    numnodes = 2;
                else
                    numnodes = get_number_terminals(thisline);

                /* skip ',' */
                if (*tmpstr == ',')
                    tmpstr++;

                /* read the input for node1: either a number or a (device dependent) name */
                node1 = gettok_noparens(&tmpstr);
                if (*node1 != '\0' && atoi(node1) == 0) {
                    char *nn = get_terminal_number(instname, node1);
                    if (eq(nn, "0")) {
                        fprintf(stderr, "Warning: Node %s is not available for device %s,\n   .probe %s will be ignored\n", node1, instname, wltmp->wl_word);
                        tfree(node1);
                        continue;
                    }

                    tfree(node1);
                    node1 = copy(nn);
                }

                if (node1 && *node1 == '\0') {
                    node1 = NULL;
                    nodename1 = copy("nn");
                }
                else
                    nodename1 = get_terminal_name(instname, node1, instances);

                /* i(R3): add voltage source always to second node */
                if (!node1 && numnodes == 2) {
                    char* newline, *strnode2, *nodename2;
                    /* skip instance */
                    thisline = nexttok(thisline);
                    /* skip first node */
                    thisline = nexttok(thisline);
                    char* begstr = copy_substring(tmpcard->line, thisline);
                    strnode2 = gettok(&thisline);

                    nodename2 = get_terminal_name(instname, "2", instances);

                    char* newnode = tprintf("probe_int_%s_%s_2", strnode2, instname);
                    char* vline = tprintf("vcurr_%s:%s_%s %s %s 0", instname, nodename2, strnode2, newnode, strnode2);
                    newline = tprintf("%s %s %s", begstr, newnode, thisline);

                    char* nodesaves = tprintf("%s#branch", instname);
                    allsaves = wl_cons(nodesaves, allsaves);

                    tfree(tmpcard->line);
                    tmpcard->line = newline;

                    tmpcard = insert_new_line(tmpcard, vline, 0, 0);

                    tfree(strnode2);
                    tfree(newnode);
                    tfree(begstr);
                    tfree(nodename1);
                    tfree(nodename2);
                }
                else if (!node1 && numnodes > 2) {
                    int err = 0;
                    tfree(nodename1);
                    err = setallvsources(tmpcard, instances, instname, numnodes, haveall, FALSE);
                    if (err) {
                        fprintf(stderr, "Warning: Cannot set zero voltage sources,\n   .probe %s will be ignored\n", wltmp->wl_word);
                    }
                    continue;
                }
                /* i(X1, 2): add voltage source to user defined node */
                else if (node1 && *node1 != '\0') {
                    char* newline, * ptr;
                    int nodenum;
                    int i;
                    bool err = FALSE;
                    /* nodes are numbered 1, 2, 3, ... */
                    nodenum = (int)strtol(node1, &ptr, 10);
                    if (nodenum > numnodes) {
                        fprintf(stderr, "Warning: There are only %d nodes available for %s,\n   .probe %s will be ignored\n", numnodes, instname, wltmp->wl_word);
                        continue;
                    }

                    /* skip instance and leading nodes not wanted */
                    for (i = 0; i < nodenum; i++) {
                        thisline = nexttok(thisline);
                        if (*thisline == '\0') {
                            fprintf(stderr, "Warning: node number %d not available for instance %s!\n", nodenum, instname);
                            err = TRUE;
                            break;
                        }
                    }
                    if (err)
                        continue;

                    char* begstr = copy_substring(tmpcard->line, thisline);

                    char* strnode1 = gettok(&thisline);

                    char* newnode = tprintf("probe_int_%s_%s_%d", strnode1, instname, nodenum);

                    newline = tprintf("%s %s %s", begstr, newnode, thisline);

                    char* vline = tprintf("vcurr_%s:%s:%s_%s %s %s 0", instname, nodename1, node1,  strnode1, strnode1, newnode);

                    tfree(tmpcard->line);
                    tmpcard->line = newline;

                    tmpcard = insert_new_line(tmpcard, vline, 0, 0);

                    char* nodesaves = tprintf("%s:%s#branch", instname, nodename1);
                    allsaves = wl_cons(nodesaves, allsaves);

                    tfree(begstr);
                    tfree(strnode1);
                    tfree(newnode);
                    tfree(nodename1);
                }
            }
            /* measure the power of a device */
            else if (ciprefix("p(", tmpstr))
            {
                char* instname;
                struct card* tmpcard;
                int numnodes;

                tmpstr += 2;
                instname = gettok_noparens(&tmpstr);
                tmpcard = nghash_find(instances, instname);
                if (!tmpcard) {
                    fprintf(stderr, "Warning: Could not find the instance line for %s,\n   .probe %s will be ignored\n", instname, wltmp->wl_word);
                    continue;
                }
                char* thisline = tmpcard->line;

                /* special treatment for controlled current sources and switches:
                   We have three or four tokens until model name, but only the first 2 are relevant nodes. */
                if (strchr("fgsw", *instname))
                    numnodes = 2;
                else
                    numnodes = get_number_terminals(thisline);

                if (numnodes < 2) {
                    fprintf(stderr, "Warning: Power mesasurement not available,\n   .probe %s will be ignored\n", wltmp->wl_word);
                    tfree(instname);
                    continue;
                }

                int err = 0;
                /* call fcn with power requested */
                err = setallvsources(tmpcard, instances, instname, numnodes, haveall, TRUE);
                if (err == 1) {
                    fprintf(stderr, "Warning: Cannot set zero voltage sources,\n   .probe %s will be ignored\n", wltmp->wl_word);
                }
                else if (err == 2) {
                    fprintf(stderr, "Warning: Zero voltage sources already set,\n   .probe %s will be ignored\n", wltmp->wl_word);
                }
                else if (err == 3) {
                    fprintf(stderr, "Warning: Number of nodes mismatch,\n   .probe %s will be ignored\n", wltmp->wl_word);
                }
                continue;
            }
            else if (!haveall) {
                fprintf(stderr, "Warning: unknown .probe parameter %s,\n   .probe %s will be ignored!\n", tmpstr, wltmp->wl_word);
                continue;
            }
        }
        if (allsaves) {
            allsaves = wl_cons(copy(".save"), allsaves);
            char* newline = wl_flatten(allsaves);
            wl_free(allsaves);
            allsaves = NULL;
            card = deck->nextcard;
            card = insert_new_line(card, newline, 0, 0);
        }

    }
    nghash_free(instances, NULL, NULL);
}

/* enter the element (instance) line and the node number (as string),
   get the node name, if defined (e.g. a for anode, c for cathode of a diode).
   If not (yet) defined, return "nx" (x is the node number) or "nn" */
static char *get_terminal_name(char* element, char *numberstr, NGHASHPTR instances)
{
    switch (*element) {
    case 'r':
    case 'c':
    case 'l':
    case 'k':
    case 'f':
    case 'h':
    case 'b':
    case 'v':
    case 'i':
        return tprintf("n%s", numberstr);
        break;
    case 'd':
        switch (*numberstr) {
        case 'a':
        case '1':
            return copy("a");
            break;
        case 'c':
        case 'k':
        case '2':
            return copy("c");
            break;
        default:
            return copy("nn");
            break;
        }
        break;
    case 'j':
    case 'z':
        switch (*numberstr) {
        case 'd':
        case '1':
            return copy("d");
            break;
        case 'g':
        case '2':
            return copy("g");
            break;
        case 's':
        case '3':
            return copy("s");
            break;
        default:
            return copy("nn");
            break;
        }
    case 'm':
        switch (*numberstr) {
        case 'd':
        case '1':
            return copy("d");
            break;
        case 'g':
        case '2':
            return copy("g");
            break;
        case 's':
        case '3':
            return copy("s");
            break;
        case 'b':
        case '4':
            return copy("b_tj");
            break;
        case '5':
            return copy("tc");
            break;
        case '6':
            return copy("n6");
            break;
        case '7':
            return copy("n7");
            break;
        default:
            return copy("nn");
            break;
        }

    case 'q':
        switch (*numberstr) {
        case 'c':
        case '1':
            return copy("c");
            break;
        case 'b':
        case '2':
            return copy("b");
            break;
        case 'e':
        case '3':
            return copy("e");
            break;
        case 's':
        case '4':
            return copy("s");
            break;
        case '5':
            return copy("t");
            break;
        default:
            return copy("nn");
            break;
        }

    case 'x':
        /* This should be the names of the corresponding subcircuit:
           Get the subckt name from the x line
           Search for the corresponding .subckt line
           Find the numberstr node name of the .subckt
           */
    {
        int i;
        char* subcktname, * ptr, * xcardsubsline = NULL, * subsnodestr;
        struct card* xcard = nghash_find(instances, element);
        char* thisline = xcard->line;
        int numnodes = get_number_terminals(thisline);
        int nodenumber = (int)strtol(numberstr, &ptr, 10);
        /*Get the subckt name from the x line*/
        for (i = 0; i <= numnodes; i++)
            thisline = nexttok(thisline);
        subcktname = gettok(&thisline);

        /*Search for the corresponding .subckt line*/
        struct card_assoc* allsubs = xcard->level->subckts;

        if (!allsubs) {
            char* instline = xcard->line;
            char* inst = gettok(&instline);
            fprintf(stderr, "Warning: No .subckt line found during evaluating command .probe (...)!\n");
            fprintf(stderr, "    failing instance: %s\n", inst);
            tfree(subcktname);
            tfree(inst);
            return tprintf("n%s", numberstr);
        }

        while (allsubs) {
            xcardsubsline = allsubs->line->line;
            /* safeguard against NULL pointers) */
            if (!subcktname || !allsubs->name) {
                tfree(subcktname);
                return tprintf("n%s", numberstr);
            }
            if (cieq(subcktname, allsubs->name))
                break;
            allsubs = allsubs->next;
        }
        /*Find the numberstr node name of the .subckt*/
        for (i = 1; i < nodenumber + 2; i++) {
            xcardsubsline = nexttok(xcardsubsline);
        }
        subsnodestr = gettok(&xcardsubsline);
        tfree(subcktname);
        return subsnodestr;
        break;
    }
/* the following are not (yet) supported */
    case 'u':
    case 'w':
    case 't':
    case 'o':
    case 'g':
    case 'e':
    case 's':
    case 'y':
    case 'p':
        return tprintf("n%s", numberstr);
        break;

    default:
        return copy("nn");
        break;
    }
}

/* enter the element (instance) line and the node name,
   if defined (e.g. a for anode, c for cathode of a diode)
   return the node number. If there is no regular node
   name, return "0". */
static char* get_terminal_number(char* element, char* namestr)
{
    switch (*element) {
    case 'r':
    case 'c':
    case 'l':
    case 'k':
    case 'f':
    case 'h':
    case 'b':
    case 'v':
    case 'i':
        return "0";
        break;
    case 'd':
        switch (*namestr) {
        case 'a':
        case '1':
            return "1";
            break;
        case 'c':
        case 'k':
        case '2':
            return "2";
            break;
        default:
            return "0";
            break;
        }
        break;
    case 'j':
    case 'z':
        switch (*namestr) {
        case 'd':
        case '1':
            return "1";
            break;
        case 'g':
        case '2':
            return "2";
            break;
        case 's':
        case '3':
            return "3";
            break;
        default:
            return "0";
            break;
        }
    case 'm':
        switch (*namestr) {
        case 'd':
        case '1':
            return "1";
            break;
        case 'g':
        case '2':
            return "2";
            break;
        case 's':
        case '3':
            return "3";
            break;
        case 'b':
        case '4':
            return "4";
            break;
        case 't':
            switch (namestr[1]) {
            case 'j':
                return  "4";
                break;
            case 'c':
                return  "5";
                break;
            default:
                return "0";
            }
        case '5':
            return "5";
            break;
        case '6':
            return "6";
            break;
        case '7':
            return "7";
            break;
        default:
            return "0";
            break;
        }

    case 'q':
        switch (*namestr) {
        case 'c':
        case '1':
            return "1";
            break;
        case 'b':
        case '2':
            return "2";
            break;
        case 'e':
        case '3':
            return "3";
            break;
        case 's':
        case '4':
            return "4";
            break;
        case 't':
            return "5";
            break;
        default:
            return "nn";
            break;
        }

        /* the following are not (yet) supported */
    case 'x':
        if (isdigit_c(*namestr))
            return namestr;
        else
            return "0";
        break;

    case 'u':
    case 'w':
        //        return 3;
        //        break;
    case 't':
    case 'o':
    case 'g':
    case 'e':
    case 's':
    case 'y':
        //        return 4;
        if (isdigit_c(*namestr))
            return namestr;
        else
            return "0";
        break;

    case 'p':
        if (isdigit_c(*namestr))
            return namestr;
        else
            return "0";
        break;

    default:
        return "0";
        break;
    }
}

/* get new .save names from V instances from instance table.
   Called from inp.c*/
void modprobenames(INPtables* tab) {
    GENinstance* GENinst;
    if (tab && tab->defVmod && tab->defVmod->GENinstances) {
        for (GENinst = tab->defVmod->GENinstances; GENinst; GENinst = GENinst->GENnextInstance) {
            char* name = GENinst->GENname;
            if (prefix("vcurr_", name)) {
                /* copy from char no. 6 to (and excluding) second colon */
                char* endname2;
                char* endname = strchr(name, ':');
                if (endname)
                    endname2 = strchr(endname + 1, ':');
                else /* not a single colon, something different? */
                    continue;
                /* two-terminal device, one colon, copy all from char no. 6 to (and excluding) colon */
                if (!endname2) {
                    char* newname = copy_substring(name + 6, endname);
                    memcpy(name, newname, strlen(newname) + 1);
                    tfree(newname);
                }
                /* copy from char no. 6 to (and excluding) second colon */
                else {
                    char* newname = copy_substring(name + 6, endname2);
                    memcpy(name, newname, strlen(newname) + 1);
                    tfree(newname);
                }
            }
        }
    }
}

/* If a command like .probe i(Q1) is found, set 0 VSRC (voltage source) in series to all nodes of the device Q1.
   Polarity of VSRC is that its node 2 is directed towards the device, its node 1 towards the outer net connection. 
   If .probe p(Q1) is found, flag power is true, then do additional power calculations:
   Define a reference voltage of an n-terminal device as Vref = (V(1) + V(2) +...+ V(n)) / n  with terminal (node) voltages V(n).
   Calculate power PQ1 = (v(1) - Vref) * i1 + (V(2) - Vref) * i2 + ... + (V(n) - Vref) * in) with terminal currents in.
   See "Quantities of a Multiterminal Circuit Determined on the Basis of Kirchhoff�s Laws", M. Depenbrock, 
   ETEP Vol. 8, No. 4, July/August 1998.
   probe_int_ is used to trigger supressing the vectors when saving the results. Internal vectors thus are
   not saved. */
static int setallvsources(struct card *tmpcard, NGHASHPTR instances, char *instname, int numnodes, bool haveall, bool power)
{
    
    struct card* card;
    char* newline;
    int nodenum;
    int err = 0;
    wordlist * allsaves = NULL;

    if (haveall && !power)
        return 2;

    DS_CREATE(BVrefline, 200);
    DS_CREATE(Bpowerline, 200);
    DS_CREATE(Bpowersave, 200);

    if (power) {
        /* For example: Bq1Vref q1Vref 0 V = 1/3*( */
        char numbuf[3];
        sadd(&BVrefline, "Bprobe_int_");
        sadd(&BVrefline, instname);
        sadd(&BVrefline, "Vref ");
        sadd(&BVrefline, instname);
        sadd(&BVrefline, "probe_int_Vref 0 V = 1/");
        sadd(&BVrefline, itoa10(numnodes, numbuf));
        sadd(&BVrefline, "*(");
        /* For example: Bq1power q1:power 0 V = */
        sadd(&Bpowerline, "Bprobe_int_");
        sadd(&Bpowerline, instname);
        sadd(&Bpowerline, "power ");
        sadd(&Bpowerline, instname);
        cadd(&Bpowerline, ':');
        sadd(&Bpowerline, "power 0 V = 0+"); /*FIXME 0+ required to suppress adding {} and numparam failure*/
        /* For example: q1:power */
        sadd(&Bpowersave, instname);
        cadd(&Bpowersave, ':');
        sadd(&Bpowersave, "power");

        /* special for VDMOS: exclude thermal nodes */
        if (*instname == 'm' && strstr(tmpcard->line, "thermal"))
            numnodes = 3;
        /* special for MOS, exclude temp nodes (not always possible) */
        if (*instname == 'm' && numnodes > 5)
            numnodes = 5;
        /* special for diodes, they have 2 terminals, so exclude thermal nodes */
        if (*instname == 'd')
            numnodes = 2;
    }

    /* Scan through all nodes of the device */
    for (nodenum = 1; nodenum <= numnodes; nodenum++) {
        int i;
        char* instline = tmpcard->line;
        for (i = nodenum; i > 0; i--) {
            instline = nexttok(instline);
        }
        char* begstr = copy_substring(tmpcard->line, instline);
        char* strnode1 = gettok(&instline);
        char* newnode = tprintf("probe_int_%s_%s_%d", strnode1, instname, nodenum);
        char nodenumstr[3];
        char *nodename1 = get_terminal_name(instname, itoa10(nodenum, nodenumstr), instances);

        if (!nodename1) {
            tfree(begstr);
            tfree(strnode1);
            ds_free(&BVrefline);
            ds_free(&Bpowerline);
            ds_free(&Bpowersave);
            return 3;
        }

        newline = tprintf("%s %s %s", begstr, newnode, instline);

        char* vline = tprintf("vcurr_%s:probe_int_%s:%s_%s %s %s 0", instname, nodename1, nodenumstr, strnode1, strnode1, newnode);

        tfree(tmpcard->line);
        tmpcard->line = newline;

        card = tmpcard->nextcard;

        card = insert_new_line(card, vline, 0, 0);

        if (power) {
            /* For example V(1)+V(2)+V(3)*/
            if (nodenum == 1)
               sadd(&BVrefline, "V(");
            else
               sadd(&BVrefline, "+V(");
            sadd(&BVrefline, newnode);
            cadd(&BVrefline, ')');
            /*For example: (V(node1)-V(q1probe_int_Vref))*node1#branch+(V(node2)-V(q1Vref))*node2#branch */
            if (nodenum == 1)
               sadd(&Bpowerline, "(V(");
            else
               sadd(&Bpowerline, "+(V(");
            sadd(&Bpowerline, newnode);
            sadd(&Bpowerline, ")-V(");
            sadd(&Bpowerline, instname);
            sadd(&Bpowerline, "probe_int_Vref))*i(vcurr_");
            sadd(&Bpowerline, instname);
            sadd(&Bpowerline, ":probe_int_");
            sadd(&Bpowerline, nodename1);
            cadd(&Bpowerline, ':');
            sadd(&Bpowerline, nodenumstr);
            cadd(&Bpowerline, '_');
            sadd(&Bpowerline, strnode1);
            cadd(&Bpowerline, ')');

            allsaves = wl_cons(copy(ds_get_buf(&Bpowersave)), allsaves);
        }

        tfree(begstr);
        tfree(strnode1);
        tfree(newnode);
        tfree(nodename1);
    }

    if (allsaves) {
        allsaves = wl_cons(copy(".save"), allsaves);
        char* newsaveline = wl_flatten(allsaves);
        wl_free(allsaves);
        allsaves = NULL;
        card = tmpcard->nextcard;
        card = insert_new_line(card, newsaveline, 0, 0);
    }

    if (power) {
        cadd(&BVrefline, ')');
        card = tmpcard->nextcard;
        card = insert_new_line(card, copy(ds_get_buf(&BVrefline)), 0, 0);
        card = insert_new_line(card, copy(ds_get_buf(&Bpowerline)), 0, 0);
    }

    ds_free(&BVrefline);
    ds_free(&Bpowerline);
    ds_free(&Bpowersave);
    return err;
}
