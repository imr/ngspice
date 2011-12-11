/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

/* from FILENAME.txt, make FILENAME.idx */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ngspice/cpdefs.h"
#include "ngspice/hlpdefs.h"
#include "ngspice/suffix.h"

#define BSIZE_SP  512


static void
makeidx(const char *dst, const char *src)
{
    FILE *fp;
    FILE *wfp;
    char buf[BSIZE_SP];
    long fpos;
    char subject[BSIZE_SP];
    struct hlp_index indexitem;

    if (!(fp = fopen(src, "r"))) {
        perror(src);
        return;
    }

    if (!(wfp = fopen(dst, "wb"))) {
        perror(dst);
        return;
    }

    fpos = 0;
    while (fgets(buf, sizeof(buf), fp)) {
        if (!strncmp(buf, "SUBJECT: ", 9)) {
            strcpy(subject, &buf[9]);
            subject[strlen(subject) - 1] = '\0';  /* get rid of '\n' */
            strncpy(indexitem.subject, subject, 64);  /* zero out end */
            indexitem.fpos = fpos;
            fwrite(&indexitem, sizeof(struct hlp_index), 1, wfp);
        }
        fpos = ftell(fp);
    }
}


int
main(int argc, char **argv)
{
    if(argc == 4 && !strcmp(argv[1], "-o")) {
        makeidx(argv[2], argv[3]);
        exit(0);
    }

    while (--argc) {

        char buf[BSIZE_SP];
        char *pos;

        strcpy(buf, argv[argc]);
        if (!(pos = strrchr(buf, '.')) || strcmp(pos, ".txt")) {
            fprintf(stderr, "%s does not end in .txt\n", buf);
            continue;
        }
        *++pos = 'i'; *++pos = 'd'; *++pos = 'x';
        makeidx(buf, argv[argc]);
    }

    exit(0);
}
