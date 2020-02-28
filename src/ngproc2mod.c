/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
/* convert .process file to set of .model cards */

#include "ngspice/ngspice.h"
#include <stdio.h>
#include <stdlib.h>
#include "ngspice/inpdefs.h"
#include "ngspice/suffix.h"


void getdata(double*,int,int);

typedef struct snmod {
    struct snmod *nnext;
    char *nname;
    double nparms[69];
} nmod;
typedef struct spmod {
    struct spmod *pnext;
    char *pname;
    double pparms[69];
} pmod;
typedef struct sdmod {
    struct sdmod *dnext;
    char *dname;
    double dparms[10];
} dmod;
typedef struct symod {
    struct symod *ynext;
    char *yname;
    double yparms[10];
} ymod;
typedef struct smmod {
    struct smmod *mnext;
    char *mname;
    double mparms[10];
} mmod;

FILE *m = NULL;
FILE *p = NULL;
char *dataline;


int
main(void) {
    char *typeline;
    char *prname;
    nmod *nlist=NULL,*ncur;
    pmod *plist=NULL,*pcur;
    dmod *dlist=NULL,*dcur;
    ymod *ylist=NULL,*ycur;
    mmod *mlist=NULL,*mcur;
    char *filename;


    filename = TMALLOC(char, 1024);
    typeline = TMALLOC(char, 1024);
    dataline = TMALLOC(char, 1024);

    while(p == NULL) {
        printf("name of process file (input): ");
        if(scanf("%s",filename)!=1) {
            printf("error reading process file name\n");
            exit(1);
        }
        p = fopen(filename,"r");
        if(p==NULL) {
            printf("can't open %s:",filename);
            perror("");
        }
    }
    while(m == NULL) {
        printf("name of .model file (output): ");
        if(scanf("%s",filename)!=1) {
            printf("error reading model file name\n");
            exit(1);
        }
        m = fopen(filename,"w");
        if(m==NULL) {
            printf("can't open %s:",filename);
            perror("");
        }
    }
    printf("process name : ");
    if(scanf("%s",filename)!=1) {
        printf("error reading process name\n");
        exit(1);
    }
    prname = filename;
    if(fgets(typeline,1023,p)==NULL) {
        printf("error reading input description line\n");
        exit(1);
    }
    INPcaseFix(typeline);
    for (;;) {
        while(*typeline == ' ' || *typeline == '\t' || *typeline == ',' ||
                *typeline == '\n' ) {
            typeline ++;
        }
        if(*typeline == '\0') break;
        if(strncmp("nm",typeline,2) == 0) {
            ncur = TMALLOC(nmod, 1);
            ncur->nnext = NULL;
            ncur->nname = typeline;
            typeline[3] = '\0';
            typeline += 4;
            getdata(ncur->nparms,69,3);
            ncur->nnext = nlist;
            nlist = ncur;
        } else if(strncmp("pm",typeline,2) == 0) {
            pcur = TMALLOC(pmod, 1);
            pcur->pnext = NULL;
            pcur->pname = typeline;
            typeline[3] = '\0';
            typeline += 4;
            getdata(pcur->pparms,69,3);
            pcur->pnext = plist;
            plist = pcur;
        } else if(strncmp("py",typeline,2) == 0) {
            ycur = TMALLOC(ymod, 1);
            ycur->ynext = NULL;
            ycur->yname = typeline;
            typeline[3] = '\0';
            typeline += 4;
            getdata(ycur->yparms,10,5);
            ycur->ynext = ylist;
            ylist = ycur;
        } else if(strncmp("du",typeline,2) == 0) {
            dcur = TMALLOC(dmod, 1);
            dcur->dnext = NULL;
            dcur->dname = typeline;
            typeline[3] = '\0';
            typeline += 4;
            getdata(dcur->dparms,10,5);
            dcur->dnext = dlist;
            dlist = dcur;
        } else if(strncmp("ml",typeline,2) == 0) {
            mcur = TMALLOC(mmod, 1);
            mcur->mnext = NULL;
            mcur->mname = typeline;
            typeline[3] = '\0';
            typeline += 4;
            getdata(mcur->mparms,10,5);
            mcur->mnext = mlist;
            mlist = mcur;
        } else {
            printf(" illegal header line in process file:  run terminated\n");
            printf(" error occurred while parsing %s\n",typeline);
            exit(1);
        }
    }
    for(dcur=dlist;dcur;dcur=dcur->dnext) {
        fprintf(m,".model %s_%s r rsh = %g defw = %g narrow = %g\n",
            prname,dcur->dname,dcur->dparms[0],dcur->dparms[8],dcur->dparms[9]);
        fprintf(m,".model %s_%s c cj = %g cjsw = %g defw = %g narrow = %g\n",
            prname,dcur->dname,dcur->dparms[1],dcur->dparms[2],dcur->dparms[8],
            dcur->dparms[9]);
    }
    for(ycur=ylist;ycur;ycur=ycur->ynext) {
        fprintf(m,".model %s_%s r rsh = %g defw = %g narrow = %g\n",
            prname,ycur->yname,ycur->yparms[0],ycur->yparms[8],ycur->yparms[9]);
        fprintf(m,".model %s_%s c cj = %g cjsw = %g defw = %g narrow = %g\n",
            prname,ycur->yname,ycur->yparms[1],ycur->yparms[2],ycur->yparms[8],
            ycur->yparms[9]);
    }
    for(mcur=mlist;mcur;mcur=mcur->mnext) {
        fprintf(m,".model %s_%s r rsh = %g defw = %g narrow = %g\n",
            prname,mcur->mname,mcur->mparms[0],mcur->mparms[8],mcur->mparms[9]);
        fprintf(m,".model %s_%s c cj = %g cjsw = %g defw = %g narrow = %g\n",
            prname,mcur->mname,mcur->mparms[1],mcur->mparms[2],mcur->mparms[8],
            mcur->mparms[9]);
    }
    for(pcur=plist;pcur;pcur=pcur->pnext) {
        for(dcur=dlist;dcur;dcur=dcur->dnext) {
            fprintf(m,".model %s_%s_%s pmos level=4\n",prname,pcur->pname,
                    dcur->dname);
            fprintf(m,"+ vfb = %g lvfb = %g wvfb = %g\n",
                    pcur->pparms[0],pcur->pparms[1],pcur->pparms[2]);
            fprintf(m,"+ phi = %g lphi = %g wphi = %g\n",
                    pcur->pparms[3],pcur->pparms[4],pcur->pparms[5]);
            fprintf(m,"+ k1 = %g lk1 = %g wk1 = %g\n",
                    pcur->pparms[6],pcur->pparms[7],pcur->pparms[8]);
            fprintf(m,"+ k2 = %g lk2 = %g wk2 = %g\n",
                    pcur->pparms[9],pcur->pparms[10],pcur->pparms[11]);
            fprintf(m,"+ eta = %g leta = %g weta = %g\n",
                    pcur->pparms[12],pcur->pparms[13],pcur->pparms[14]);
            fprintf(m,"+ muz = %g dl = %g dw = %g\n",
                    pcur->pparms[15],pcur->pparms[16],pcur->pparms[17]);
            fprintf(m,"+ u0 = %g lu0 = %g wu0 = %g\n",
                    pcur->pparms[18],pcur->pparms[19],pcur->pparms[20]);
            fprintf(m,"+ u1 = %g lu1 = %g wu1 = %g\n",
                    pcur->pparms[21],pcur->pparms[22],pcur->pparms[23]);
            fprintf(m,"+ x2mz = %g lx2mz = %g wx2mz = %g\n",
                    pcur->pparms[24],pcur->pparms[25],pcur->pparms[26]);
            fprintf(m,"+ x2e = %g lx2e = %g wx2e = %g\n",
                    pcur->pparms[27],pcur->pparms[28],pcur->pparms[29]);
            fprintf(m,"+ x3e = %g lx3e = %g wx3e = %g\n",
                    pcur->pparms[30],pcur->pparms[31],pcur->pparms[32]);
            fprintf(m,"+ x2u0 = %g lx2u0 = %g wx2u0 = %g\n",
                    pcur->pparms[33],pcur->pparms[34],pcur->pparms[35]);
            fprintf(m,"+ x2u1 = %g lx2u1 = %g wx2u1 = %g\n",
                    pcur->pparms[36],pcur->pparms[37],pcur->pparms[38]);
            fprintf(m,"+ mus = %g lmus = %g wmus = %g\n",
                    pcur->pparms[39],pcur->pparms[40],pcur->pparms[41]);
            fprintf(m,"+ x2ms = %g lx2ms = %g wx2ms = %g\n",
                    pcur->pparms[42],pcur->pparms[43],pcur->pparms[44]);
            fprintf(m,"+ x3ms = %g lx3ms = %g wx3ms = %g\n",
                    pcur->pparms[45],pcur->pparms[46],pcur->pparms[47]);
            fprintf(m,"+ x3u1 = %g lx3u1 = %g wx3u1 = %g\n",
                    pcur->pparms[48],pcur->pparms[49],pcur->pparms[50]);
            fprintf(m,"+ tox = %g temp = %g vdd = %g\n",
                    pcur->pparms[51],pcur->pparms[52],pcur->pparms[53]);
            fprintf(m,"+ cgdo = %g cgso = %g cgbo = %g\n",
                    pcur->pparms[54],pcur->pparms[55],pcur->pparms[56]);
            fprintf(m,"+ xpart = %g \n",
                    pcur->pparms[57]);
            fprintf(m,"+ n0 = %g ln0 = %g wn0 = %g\n",
                    pcur->pparms[60],pcur->pparms[61],pcur->pparms[62]);
            fprintf(m,"+ nb = %g lnb = %g wnb = %g\n",
                    pcur->pparms[63],pcur->pparms[64],pcur->pparms[65]);
            fprintf(m,"+ nd = %g lnd = %g wnd = %g\n",
                    pcur->pparms[66],pcur->pparms[67],pcur->pparms[68]);
            fprintf(m,"+ rsh = %g cj = %g cjsw = %g\n",
                dcur->dparms[0], dcur->dparms[1], dcur->dparms[2]);
            fprintf(m,"+ js = %g pb = %g pbsw = %g\n",
                dcur->dparms[3], dcur->dparms[4], dcur->dparms[5]);
            fprintf(m,"+ mj = %g mjsw = %g wdf = %g\n",
                dcur->dparms[6], dcur->dparms[7], dcur->dparms[8]);
            fprintf(m,"+ dell = %g\n",
                dcur->dparms[9]);
        }
    }
    for(ncur=nlist;ncur;ncur=ncur->nnext) {
        for(dcur=dlist;dcur;dcur=dcur->dnext) {
            fprintf(m,".model %s_%s_%s nmos level=4\n",prname,ncur->nname,
                    dcur->dname);
            fprintf(m,"+ vfb = %g lvfb = %g wvfb = %g\n",
                    ncur->nparms[0],ncur->nparms[1],ncur->nparms[2]);
            fprintf(m,"+ phi = %g lphi = %g wphi = %g\n",
                    ncur->nparms[3],ncur->nparms[4],ncur->nparms[5]);
            fprintf(m,"+ k1 = %g lk1 = %g wk1 = %g\n",
                    ncur->nparms[6],ncur->nparms[7],ncur->nparms[8]);
            fprintf(m,"+ k2 = %g lk2 = %g wk2 = %g\n",
                    ncur->nparms[9],ncur->nparms[10],ncur->nparms[11]);
            fprintf(m,"+ eta = %g leta = %g weta = %g\n",
                    ncur->nparms[12],ncur->nparms[13],ncur->nparms[14]);
            fprintf(m,"+ muz = %g dl = %g dw = %g\n",
                    ncur->nparms[15],ncur->nparms[16],ncur->nparms[17]);
            fprintf(m,"+ u0 = %g lu0 = %g wu0 = %g\n",
                    ncur->nparms[18],ncur->nparms[19],ncur->nparms[20]);
            fprintf(m,"+ u1 = %g lu1 = %g wu1 = %g\n",
                    ncur->nparms[21],ncur->nparms[22],ncur->nparms[23]);
            fprintf(m,"+ x2mz = %g lx2mz = %g wx2mz = %g\n",
                    ncur->nparms[24],ncur->nparms[25],ncur->nparms[26]);
            fprintf(m,"+ x2e = %g lx2e = %g wx2e = %g\n",
                    ncur->nparms[27],ncur->nparms[28],ncur->nparms[29]);
            fprintf(m,"+ x3e = %g lx3e = %g wx3e = %g\n",
                    ncur->nparms[30],ncur->nparms[31],ncur->nparms[32]);
            fprintf(m,"+ x2u0 = %g lx2u0 = %g wx2u0 = %g\n",
                    ncur->nparms[33],ncur->nparms[34],ncur->nparms[35]);
            fprintf(m,"+ x2u1 = %g lx2u1 = %g wx2u1 = %g\n",
                    ncur->nparms[36],ncur->nparms[37],ncur->nparms[38]);
            fprintf(m,"+ mus = %g lmus = %g wmus = %g\n",
                    ncur->nparms[39],ncur->nparms[40],ncur->nparms[41]);
            fprintf(m,"+ x2ms = %g lx2ms = %g wx2ms = %g\n",
                    ncur->nparms[42],ncur->nparms[43],ncur->nparms[44]);
            fprintf(m,"+ x3ms = %g lx3ms = %g wx3ms = %g\n",
                    ncur->nparms[45],ncur->nparms[46],ncur->nparms[47]);
            fprintf(m,"+ x3u1 = %g lx3u1 = %g wx3u1 = %g\n",
                    ncur->nparms[48],ncur->nparms[49],ncur->nparms[50]);
            fprintf(m,"+ tox = %g temp = %g vdd = %g\n",
                    ncur->nparms[51],ncur->nparms[52],ncur->nparms[53]);
            fprintf(m,"+ cgdo = %g cgso = %g cgbo = %g\n",
                    ncur->nparms[54],ncur->nparms[55],ncur->nparms[56]);
            fprintf(m,"+ xpart = %g \n",
                    ncur->nparms[57]);
            fprintf(m,"+ n0 = %g ln0 = %g wn0 = %g\n",
                    ncur->nparms[60],ncur->nparms[61],ncur->nparms[62]);
            fprintf(m,"+ nb = %g lnb = %g wnb = %g\n",
                    ncur->nparms[63],ncur->nparms[64],ncur->nparms[65]);
            fprintf(m,"+ nd = %g lnd = %g wnd = %g\n",
                    ncur->nparms[66],ncur->nparms[67],ncur->nparms[68]);
            fprintf(m,"+ rsh = %g cj = %g cjsw = %g\n",
                dcur->dparms[0], dcur->dparms[1], dcur->dparms[2]);
            fprintf(m,"+ js = %g pb = %g pbsw = %g\n",
                dcur->dparms[3], dcur->dparms[4], dcur->dparms[5]);
            fprintf(m,"+ mj = %g mjsw = %g wdf = %g\n",
                dcur->dparms[6], dcur->dparms[7], dcur->dparms[8]);
            fprintf(m,"+ dell = %g\n",
                dcur->dparms[9]);
        }
    }
    return EXIT_NORMAL;
}

void
getdata(double *vals, int count, int width) 
 /* width: maximum number of values to accept per line */
{
    int i;
    int error;
    int start;
    char *c;

    do {
        if(fgets(dataline,1023,p)==NULL) {
            printf("premature end of file getting input data line\n");
            exit(1);
        }
        start=0;
    } while (*dataline == '*') ;
    c = dataline;
    for(i=0;i<count;i++) {
retry:
        vals[i] = INPevaluate(&c,&error,1);
        start++;
        if(error || (start>width)) { /* end of line, so read another one */
            do {
                if(fgets(dataline,1023,p)==NULL) {
                    printf("premature end of file reading input data line \n");
                    exit(1);
                }
                start=0;
            } while (*dataline == '*') ;
            c = dataline;
            goto retry;
        }
    }
}

void
controlled_exit(int status) { exit(status); }
