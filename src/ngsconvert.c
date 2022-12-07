/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Main routine for sconvert.
 */

#include "ngspice/ngspice.h"
#include <stdio.h>

#include "ngspice/fteinput.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/sim.h"
#include "ngspice/suffix.h"
#include "ngspice/compatmode.h"
#include "frontend/display.h"
#include "../misc/mktemp.h"

#include <errno.h>


FILE *cp_in = NULL;
FILE *cp_out = NULL;
FILE *cp_err = NULL;
FILE *cp_curin = NULL;
FILE *cp_curout = NULL;
FILE *cp_curerr = NULL;
bool cp_debug = FALSE;
char cp_chars[128];
bool cp_nocc = TRUE;
bool ft_stricterror = FALSE;
bool ft_parsedb = FALSE;
struct circ *ft_curckt = NULL;
struct plot *plot_cur = NULL;
int  cp_maxhistlength = 0;
bool cp_no_histsubst = FALSE;
struct compat newcompat;
bool cx_degrees = FALSE;

char *cp_program = "sconvert";


#define tfread(ptr, siz, nit, fp)   if (fread((ptr), (siz), \
                        (nit), (fp)) != (nit)) { \
                fprintf(cp_err, "Error: unexpected EOF\n"); \
                    return (NULL); }

#define tfwrite(ptr, siz, nit, fp)  if (fwrite((ptr), (siz), \
                        (nit), (fp)) != (nit)) { \
                    fprintf(cp_err, "Write error\n"); \
                    return; }

#define TMALLOC(t, n)       (t*) tmalloc(sizeof(t) * (size_t)(n))
#define TREALLOC(t, p, n)   (t*) trealloc(p, sizeof(t) * (size_t)(n))


char *
smktemp(char *id)
{
    if (!id)
        id = "sp";
    const char* const home = getenv("HOME");
    if (home) {
        return tprintf("%s/"TEMPFORMAT, home, id, getpid());
    }
    const char* const usr = getenv("USERPROFILE");
    if (usr) {
        return tprintf("%s\\"TEMPFORMAT, usr, id, getpid());
    }
    return tprintf(TEMPFORMAT, id, getpid());
}

int
inchar(FILE *fp)
{

#if !defined(__MINGW32__)
    char c;
    ssize_t i;

    do
        i = read(fileno(fp), &c, 1);
    while (i == -1 && errno == EINTR);

    if (i == 0 || c == '\004')
        return EOF;

    if (i == -1) {
        perror("read");
        return EOF;
    }

    return (int) c;
#elif

    return getc(fp);
#endif    
}

int
input(FILE *fp)
{
    REQUEST request;
    RESPONSE response;

    request.option = char_option;
    request.fp = fp;

    Input(&request, &response);

    return (inchar(fp));
}


void
Input(REQUEST *request, RESPONSE *response)
{
    switch (request->option) {
    case char_option:
	response->reply.ch = inchar(request->fp);
	response->option = request->option;
	break;
    default:
	/* just ignore, since we don't want a million error messages */
	response->option = error_option;
	break;
    }
    return;
}


static char *
fixdate(char *date)
{
    char buf[20];
    int i;

    (void) strcpy(buf, date);
    for (i = 17; i > 8; i--)
        buf[i] = buf[i - 1];
    buf[8] = ' ';
    buf[18] = '\0';
    return (strdup(buf));
}


static struct plot *
oldread(char *name)
{
    struct plot *pl;
    char buf[BSIZE_SP];
    struct dvec *v, *end = NULL;
    short nv;       /* # vars */
    long np;        /* # points/var. */
    long i, j;
    short a;        /* The magic number. */
    float f1, f2;
    FILE *fp;

    if (!(fp = fopen(name, "r"))) {
        perror(name);
        return (NULL);
    }
    pl = TMALLOC(struct plot, 1);
    tfread(buf, 1, 80, fp);
    buf[80] = '\0';
    for (i = (int) strlen(buf) - 1; (i > 1) && (buf[i] == ' '); i--)
        ;
    buf[i + 1] = '\0';
    pl->pl_title = strdup(buf);

    tfread(buf, 1, 16, fp);
    buf[16] = '\0';
    pl->pl_date = strdup(fixdate(buf));

    tfread(&nv, sizeof (short), 1, fp);

    tfread(&a, sizeof (short), 1, fp);
    if (a != 4)
        fprintf(cp_err, "Warning: magic number 4 is wrong...\n");

    for (i = 0; i < nv; i++) {
        v = dvec_alloc(NULL,
                       SV_NOTYPE, 0,
                       0, NULL);
        if (end)
            end->v_next = v;
        else
            pl->pl_scale = pl->pl_dvecs = v;
        end = v;
        tfread(buf, 1, 8, fp);
        buf[8] = '\0';
        v->v_name = strdup(buf);
    }
    for (v = pl->pl_dvecs; v; v = v->v_next) {
        tfread(&a, sizeof (short), 1, fp);
        v->v_type = a;
    }

    /* If the first output variable is type FREQ then there is complex
     * data, otherwise the data is real.
     */
    i = pl->pl_dvecs->v_type;
    if ((i == SV_FREQUENCY) || (i == SV_POLE) || (i == SV_ZERO))
        for (v = pl->pl_dvecs; v; v = v->v_next)
            v->v_flags |= VF_COMPLEX;
    else 
        for (v = pl->pl_dvecs; v; v = v->v_next)
            v->v_flags |= VF_REAL;

    /* Check the node indices -- this shouldn't be a problem ever. */
    for (i = 0; i < nv; i++) {
        tfread(&a, sizeof(short), 1, fp);
        if (a != i + 1) 
	  fprintf(cp_err, "Warning: output %d should be %ld\n",
		  a, i);
    }
    tfread(buf, 1, 24, fp);
    buf[24] = '\0';
    pl->pl_name = strdup(buf);
    /* Now to figure out how many points of data there are left in
     * the file. 
     */
    i = ftell(fp);
    (void) fseek(fp, 0L, SEEK_END);
    j = ftell(fp);
    (void) fseek(fp, i, SEEK_SET);
    i = j - i;
    if (i % 8) {    /* Data points are always 8 bytes... */
        fprintf(cp_err, "Error: alignment error in data\n");
        (void) fclose(fp);
        return (NULL);
    }
    i = i / 8;
    if (i % nv) {
        fprintf(cp_err, "Error: alignment error in data\n");
        (void) fclose(fp);
        return (NULL);
    }
    np = i / nv;

    for (v = pl->pl_dvecs; v; v = v->v_next) {
        dvec_realloc(v, (int) np, NULL);
    }
    for (i = 0; i < np; i++) {
        /* Read in the output vector for point i. If the type is
         * complex it will be float and we want double.
         */
        for (v = pl->pl_dvecs; v; v = v->v_next) {
            if (v->v_flags & VF_REAL) {
                tfread(&v->v_realdata[i], sizeof (double),
                        1, fp);
            } else {
                tfread(&f1, sizeof (float), 1, fp);
                tfread(&f2, sizeof (float), 1, fp);
                realpart(v->v_compdata[i]) = f1;
                imagpart(v->v_compdata[i]) = f2;
            }
        }
    }
    (void) fclose(fp);
    return (pl);
}


static void
oldwrite(char *name, bool app, struct plot *pl)
{
    short four = 4, k;
    struct dvec *v;
    float f1, f2, zero = 0.0;
    char buf[80];
    int i, j, tp = VF_REAL, numpts = 0, numvecs = 0;
    FILE *fp;

    if (!(fp = fopen(name, app ? "a" : "w"))) {
        perror(name);
        return;
    }

    for (v = pl->pl_dvecs; v; v = v->v_next) {
        if (v->v_length > numpts)
            numpts = v->v_length;
        numvecs++;
        if (iscomplex(v))
            tp = VF_COMPLEX;
    }

    /* This may not be a good idea... */
    if (tp == VF_COMPLEX)
        pl->pl_scale->v_type = SV_FREQUENCY;

    for (i = 0; i < 80; i++)
        buf[i] = ' ';
    for (i = 0; i < 80; i++)
        if (pl->pl_title[i] == '\0')
            break;
        else
            buf[i] = pl->pl_title[i];
    tfwrite(buf, 1, 80, fp);

    for (i = 0; i < 80; i++)
        buf[i] = ' ';
    for (i = 0; i < 16; i++)
        if (pl->pl_date[i] == '\0')
            break;
        else
            buf[i] = pl->pl_date[i];
    tfwrite(buf, 1, 16, fp);

    tfwrite(&numvecs, sizeof (short), 1, fp);
    tfwrite(&four, sizeof (short), 1, fp);

    for (v = pl->pl_dvecs; v; v = v->v_next) {
        for (j = 0; j < 80; j++)
            buf[j] = ' ';
        for (j = 0; j < 8; j++)
            if (v->v_name[j] == '\0')
                break;
            else
                buf[j] = v->v_name[j];
        tfwrite(buf, 1, 8, fp);
    }

    for (v = pl->pl_dvecs; v; v = v->v_next) {
        j = (short) v->v_type;
        tfwrite(&j, sizeof (short), 1, fp);
    }

    for (k = 1; k < numvecs + 1; k++)
        tfwrite(&k, sizeof (short), 1, fp);
    for (j = 0; j < 80; j++)
        buf[j] = ' ';
    for (j = 0; j < 24; j++)
        if (pl->pl_name[j] == '\0')
            break;
        else
            buf[j] = pl->pl_name[j];
    tfwrite(buf, 1, 24, fp);
    for (i = 0; i < numpts; i++) {
        for (v = pl->pl_dvecs; v; v = v->v_next) {
            if ((tp == VF_REAL) && isreal(v)) {
                if (i < v->v_length) {
            tfwrite(&v->v_realdata[i], sizeof (double), 1, fp);
                } else {
    tfwrite(&v->v_realdata[v->v_length - 1], sizeof (double), 1, fp);
                }
            } else if ((tp == VF_REAL) && iscomplex(v)) {
                fprintf(cp_err, "internal error, everything real, yet complex ...\n");
                exit(1);
            } else if ((tp == VF_COMPLEX) && isreal(v)) {
                if (i < v->v_length)
                    f1 = (float) v->v_realdata[i];
                else
                    f1 = (float) v->v_realdata[v->v_length - 1];
                tfwrite(&f1, sizeof (float), 1, fp);
                tfwrite(&zero, sizeof (float), 1, fp);
            } else if ((tp == VF_COMPLEX) && iscomplex(v)) {
                if (i < v->v_length) {
                    f1 = (float) realpart(v->v_compdata[i]);
                    f2 = (float) imagpart(v->v_compdata[i]);
                } else {
                    f1 = (float) realpart(v->v_compdata[v-> v_length - 1]);
                    f2 = (float) imagpart(v->v_compdata[v-> v_length - 1]);
                }
                tfwrite(&f1, sizeof (float), 1, fp);
                tfwrite(&f2, sizeof (float), 1, fp);
            }
        }
    }

    (void) fclose(fp);
    return;
}


int
main(int ac, char **av)
{
    char *sf, *af;
    char buf[BSIZE_SP];
    char t, f;
    struct plot *pl;
    size_t n;
    char *infile = NULL;
    char *outfile = NULL;
    FILE *fp;

    switch (ac) {
        case 5: 
            sf = av[2];
            af = av[4];
            f = *av[1];
            t = *av[3];
            break;

        case 3:
            f = *av[1];
            t = *av[2];
            /* This is a pain, but there is no choice */
            sf = infile = smktemp("scin");
            af = outfile = smktemp("scout");
            if (!(fp = fopen(infile, "w"))) {
                perror(infile);
                exit(EXIT_BAD);
            }
            while ((n = fread(buf, 1, sizeof(buf), stdin)) != 0)
                (void) fwrite(buf, 1, n, fp);
            (void) fclose(fp);
            break;

        case 1: printf("Input file: ");
            (void) fflush(stdout);
            (void) fgets(buf, BSIZE_SP, stdin);
            sf = strdup(buf);
            printf("Input type: ");
            (void) fflush(stdout);
            (void) fgets(buf, BSIZE_SP, stdin);
            f = buf[0];
            printf("Output file: ");
            (void) fflush(stdout);
            (void) fgets(buf, BSIZE_SP, stdin);
            af = strdup(buf);
            printf("Output type: ");
            (void) fflush(stdout);
            (void) fgets(buf, BSIZE_SP, stdin);
            t = buf[0];
            break;
        default:
            fprintf(cp_err, 
                "Usage: %s fromtype fromfile totype tofile,\n",
                cp_program);
            fprintf(cp_err, "\twhere types are o, b, or a\n");
            fprintf(cp_err, 
                "\tor, %s fromtype totype, used as a filter.\n",
                cp_program);
            exit(EXIT_BAD);
    }
    switch(f) {
        case 'o' :
        pl = oldread(sf);
        break;

        case 'b' :
        case 'a' :
        pl = raw_read(sf);
        break;

        default:
        fprintf(cp_err, "Types are o, a, or b\n");
        exit(EXIT_BAD);
    }
    if (!pl)
        exit(EXIT_BAD);

    switch(t) {
        case 'o' :
        oldwrite(af, FALSE, pl);
        break;

        case 'b' :
        raw_write(af, pl, FALSE, TRUE);
        break;

        case 'a' :
        raw_write(af, pl, FALSE, FALSE);
        break;

        default:
        fprintf(cp_err, "Types are o, a, or b\n");
        exit(EXIT_BAD);
    }
    if (ac == 3) {
        /* Gotta finish this stuff up */
        if (!(fp = fopen(outfile, "r"))) {
            perror(outfile);
            exit(EXIT_BAD);
        }
        while ((n = fread(buf, 1, sizeof(buf), fp)) != 0)
            (void) fwrite(buf, 1, n, stdout);
        (void) fclose(fp);
        (void) unlink(infile);
        (void) unlink(outfile);
    }
    exit(EXIT_NORMAL);
}


void cp_pushcontrol(void) { }
void cp_popcontrol(void) { }
void out_init(void) { }
void cp_doquit(void) { exit(0); }
struct variable *cp_usrvars(void) { return NULL; }
int cp_evloop(char *s) { NG_IGNORE(s); return (0); }
void cp_ccon(bool o) { NG_IGNORE(o); }
char*if_errstring(int c) { NG_IGNORE(c); return strdup("error"); }
void out_printf(char *fmt, ...) { NG_IGNORE(fmt); }
void out_send(char *string) { NG_IGNORE(string); }
struct variable * cp_enqvar(const char *word, int *tbfreed) { NG_IGNORE(word); NG_IGNORE(*tbfreed); return (NULL); }
struct dvec *vec_get(const char *word) { NG_IGNORE(word); return (NULL); }
void cp_ccom(wordlist *w, char *b, bool e) {
  NG_IGNORE(e);
  NG_IGNORE(b);
  NG_IGNORE(w); return; }
int cp_usrset(struct variable *v, bool i) {
    NG_IGNORE(i);
    NG_IGNORE(v); return(US_OK); }
wordlist * cp_doalias(wordlist *wlist) {NG_IGNORE(wlist); return NULL;}

void controlled_exit(int no){exit(no);}

int disptype;

