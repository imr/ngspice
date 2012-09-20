#include <stdio.h>

#include "commands.h"


void
print_struct_comm(struct comm coms[])
{
    int i;

    for (i = 0; coms[i].co_comname != NULL; i++) {
        printf("Command: %s\n"
               "help: %s\n\n",
               coms[i].co_comname,
               coms[i].co_help);
    }
}


int
main(void)
{
    print_struct_comm(nutcp_coms);
    print_struct_comm(spcp_coms);
    return 0;
}
