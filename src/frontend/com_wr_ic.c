/* ngspice file
   Copyright Holger Vogt 2021
   License: BSD 3-clause 
 */

 /* Print the current node status to a file with format
    .ic V(node) = value 
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/ftedev.h"
#include "ngspice/ftedebug.h"
#include "ngspice/cktdefs.h"

 /* Print the current node status to a file with format
    .ic V(node) = value 
 */
void
com_wric(wordlist* wl) {

    CKTnode* node;
    CKTcircuit* ckt = NULL;
    FILE* fp;
    char* file;

    if (wl)
        file = wl->wl_word;
    else
        file = "dot_ic_out.txt";

    if ((fp = fopen(file, "w")) == NULL) {
        perror(file);
        return;
    }

    if (!ft_curckt) {
        fprintf(cp_err, "Error: there aren't any circuits loaded.\n");
        return;
    }
    else if (ft_curckt->ci_ckt == NULL) { /* Set noparse? */
        fprintf(cp_err, "Error: circuit not parsed.\n");
        return;
    }

    ckt = ft_curckt->ci_ckt;

    fprintf(fp, "* Intermediate Transient Solution\n");
    fprintf(fp, "* Circuit: %s\n", ft_curckt->ci_name);
    fprintf(fp, "* Recorded at simulation time: %g\n", ckt->CKTtime);
    for (node = ckt->CKTnodes->next; node; node = node->next) {
        if (!strstr(node->name, "#branch") && !strchr(node->name, '#'))
            fprintf(fp, ".ic v(%s) = %g\n", node->name,
                ckt->CKTrhsOld[node->number]);
    }

    fprintf(stdout, "\nNode data saved to file %s\n", file);

    fclose(fp);
}
