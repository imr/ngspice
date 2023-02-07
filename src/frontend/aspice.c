/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Stuff for asynchronous spice runs, and also rspice.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"

#include "aspice.h"
#include "variable.h"
#include "circuits.h"


#  ifdef HAVE_SYS_WAIT_H
/* should be more tests here I think */
#    define OK_ASPICE
#  endif

#ifdef OK_ASPICE

#ifdef HAVE_PWD_H
#  include <pwd.h>
#endif

#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/wait.h>

#include <signal.h>

#include "ngspice/fteinp.h"
#include "ngspice/dvec.h"

#include "../misc/mktemp.h"

/*
  This is required for the GCC pre-processor and might be needed for others
  Added to resolve ngspice bug 1293746
  http://sourceforge.net/tracker/index.php?func=detail&aid=1293746&group_id=38962&atid=423915
*/
#if !defined(SOLARIS) && defined(__SVR4) && defined(__sun)
#  define SOLARIS
#endif


#ifndef SEEK_SET
#  define SEEK_SET 0
#endif

static void sigchild(void);

struct proc {
    int pr_pid;                 /* The pid of the spice job. */
    char *pr_rawfile;           /* The temporary raw file. */
    char *pr_name;              /* The name of the spice run. */
    char *pr_inpfile;           /* The name of the input file. */
    char *pr_outfile;           /* The name of the (tmp) output file. */
    bool pr_saveout;            /* Don't (void) unlink the output file */
    struct proc *pr_next;       /* Link. */
};

static struct proc *running = NULL;
static int numchanged = 0;      /* How many children have changed in state. */


void
com_aspice(wordlist *wl)
{
    char *deck, *output = NULL, spicepath[BSIZE_SP], s[BSIZE_SP];
    char *raw, *t;
    FILE *inp;
    struct proc *p;
    int pid;
    bool saveout = FALSE;

    deck = wl->wl_word;
    if (!cp_getvar("spicepath", CP_STRING, spicepath, sizeof(spicepath))) {
        if (!Spice_Path || !*Spice_Path) {
            fprintf(cp_err,
                    "No spice-3 binary is available for the aspice command.\n");
            return;
        }
        (void) strcpy(spicepath, Spice_Path);
    }

    if (wl->wl_next) {
        output = wl->wl_next->wl_word;
        saveout = TRUE;
    } else {
        output = smktemp("spout");
    }

    if ((inp = fopen(deck, "r")) == NULL) {
        perror(deck);
        return;
    }
    if (!fgets(s, BSIZE_SP, inp)) {
        fprintf(cp_err, "Error: bad deck %s\n", deck);
        (void) fclose(inp);
        return;
    }
    for (t = s; *t && (*t != '\n'); t++)
        ;
    *t = '\0';
    fprintf(cp_out, "Starting spice run for:\n%s\n", s);
    (void) fclose(inp);
    raw = smktemp("raw");
    (void) fclose(fopen(raw, "w")); /* So there isn't a race condition. */
    pid = fork();
    if (pid == 0) {
        if (!(freopen(deck, "r", stdin))) {
            perror(deck);
            exit(EXIT_BAD);
        }
        if (!(freopen(output, "w", stdout))) {
            perror(output);
            exit(EXIT_BAD);
        }
        (void) dup2(fileno(stdout), fileno(stderr));

        (void) execl(spicepath, spicepath, "-r", raw, NULL);

        /* Screwed up. */
        perror(spicepath);
        exit(EXIT_BAD);
    }

    /* Add this one to the job list. */
    p = TMALLOC(struct proc, 1);
    p->pr_pid = pid;
    p->pr_name = copy(s);
    p->pr_rawfile = copy(raw);
    p->pr_inpfile = copy(deck);
    p->pr_outfile = copy(output);
    p->pr_saveout = saveout;
    if (running)
        p->pr_next = running;
    running = p;
#  ifdef SIGCHLD
    (void) signal(SIGCHLD, (SIGNAL_FUNCTION) sigchild);
#  else
#    ifdef SIGCLD
    (void) signal(SIGCLD, (SIGNAL_FUNCTION) sigchild);
#    endif
#  endif
}


void
com_jobs(wordlist *wl)
{
    struct proc *p;

    NG_IGNORE(wl);

    for (p = running; p; p = p->pr_next)
        fprintf(cp_out, "%d\t%.70s\n", p->pr_pid, p->pr_name);
}


static void
sigchild(void)
{
    numchanged++;
    if (ft_asyncdb)
        fprintf(cp_err, "%d jobs done now\n", numchanged);
    if (cp_cwait)
        ft_checkkids();
}


/* This gets called every once in a while, and checks to see if any
 * jobs have finished. If they have it gets the data.  The problem is
 * that wait(0) is probably more portable, but it can't tell
 * whether the exit was normal or not.
 */

/*
 * On posix systems, wait() is:
 * pid_t wait(int *status);
 */

int status;


void
ft_checkkids(void)
{
    struct proc *p = NULL, *lp = NULL;
    char buf[BSIZE_SP];
    FILE *fp;
    pid_t pid = 0;
    static bool here = FALSE;   /* Don't want to be re-entrant. */

    if (!numchanged || here)
        return;

    here = TRUE;

    while (numchanged > 0) {
        pid = wait(&status);
        if (pid == -1) {
            fprintf(cp_err,
                    "ft_checkkids: Internal Error: should be %d jobs done but there aren't any.\n",
                    numchanged);
            numchanged = 0;
            running = NULL;
            here = FALSE;
            return;
        }

        for (p = running; p; p = p->pr_next) {
            if (p->pr_pid == pid)
                break;
            lp = p;
        }

        if (p == NULL) {
            fprintf(cp_err,
                    "ft_checkkids: Internal Error: Process %d not a job!\n",
                    (int) pid);
            here = FALSE;
            return;
        }

        if (p == running)
            running = p->pr_next;
        else
            lp->pr_next = p->pr_next;

        fprintf(cp_out, "Job finished: %.60s\n", p->pr_name);
        numchanged--;
        ft_loadfile(p->pr_rawfile);
        (void) unlink(p->pr_rawfile);
        out_init();

        if (!(fp = fopen(p->pr_outfile, "r"))) {
            perror(p->pr_outfile);
            here = FALSE;
            return;
        }

        while (fgets(buf, BSIZE_SP, fp))
            out_send(buf);

        (void) fclose(fp);
        if (!p->pr_saveout)
            (void) unlink(p->pr_outfile);
        printf("\n-----\n");
    }

    printf("\n");
#ifdef TIOCSTI
    (void) ioctl(0, TIOCSTI, "\022");   /* Reprint the line. */
#endif
    here = FALSE;
}


/* Run a spice job remotely. See the description of the spice daemon for
 * the protocol. This is no longer 4.2 specific.
 */

void
com_rspice(wordlist *wl)
{
    char rhost[64], program[128], buf[BSIZE_SP];
    char remote_shell[513];
    char *outfile;
    FILE *inp, *serv, *out, *srv_input, *err_outp;
    struct plot *pl;
    size_t n;
    int to_serv[2], from_serv[2], err_serv[2];
    int pid;
    long        pos;
    int num;
    char *p;

    /* Figure out where the spicedaemon is and connect to it. */
    if (!cp_getvar("rhost", CP_STRING, rhost, sizeof(rhost)))
        (void) strcpy(rhost, Spice_Host);
    if (!cp_getvar("rprogram", CP_STRING, program, sizeof(program)))
        *program = '\0';
    if (!cp_getvar("remote_shell", CP_STRING, remote_shell, sizeof(remote_shell)))
        strcpy(remote_shell, "rsh");

    if (*rhost == '\0') {
        fprintf(cp_err,
                "Error: there is no remote ngspice.host for this site -- set \"rhost\".\n");
        return;
    }
    if (*program == '\0') {
        fprintf(cp_err,
                "Error: there is no remote spice program for this site -- set \"rprogram\".\n");
        return;
    }

    if (pipe(to_serv) < 0) {
        perror("pipe to server");
        return;
    }
    if (pipe(from_serv) < 0) {
        perror("pipe from server");
        return;
    }
    if (pipe(err_serv) < 0) {
        perror("2nd pipe from server");
        return;
    }

    pid = fork();
    if (pid == 0) {
        /* I am the "server" process */
        close(to_serv[1]);
        close(from_serv[0]);
        close(err_serv[0]);

        fclose(stdin);
        fclose(stdout);
        fclose(stderr);

        dup2(to_serv[0], 0);    /* stdin */
        dup2(from_serv[1], 1);  /* stdout */
        dup2(err_serv[1], 2);   /* stderr */

        execlp(remote_shell, remote_shell, rhost, program, "-s", NULL);
        /* system(com_buf); */
        perror(remote_shell);
        exit(-1);
    } else if (pid == -1) {
        perror("fork");
        return;
    }

    /* I am the "client" side */
    close(to_serv[0]);
    close(from_serv[1]);
    close(err_serv[1]);
    srv_input = fdopen(to_serv[1], "w");
    serv = fdopen(from_serv[0], "r");
    err_outp = fdopen(err_serv[0], "r");

    /* Send the circuit over. */
    if (wl) {
        while (wl) {
            if (!(inp = fopen(wl->wl_word, "r"))) {
                perror(wl->wl_word);
                wl = wl->wl_next;
                continue;   /* Should be careful */
            }
            while ((n = fread(buf, 1, BSIZE_SP, inp)) > 0)
                (void) fwrite(buf, 1, strlen(buf), srv_input);
            /* (void) write(s, buf, n); */
            wl = wl->wl_next;
            fclose(inp);
        }
        /* (void) write(s, "@\n", 3);*/
    } else {
        if (ft_nutmeg || !ft_curckt) {
            fprintf(cp_err, "Error: no circuits loaded\n");
            fclose(srv_input);
            fclose(serv);
            return;
        }

        inp_list(srv_input, ft_curckt->ci_deck, ft_curckt->ci_options, LS_DECK);
    }

    fclose(srv_input);

    /* Now wait for things to come through */
    while ((p = fgets(buf, BSIZE_SP, serv)) != NULL) {
        if (!strncmp(buf, "Title:", 6))
            break;
        fputs(buf, cp_out);
    }

    outfile = smktemp("rsp");
    if ((out = fopen(outfile, "w+")) == NULL) {
        perror(outfile);
        (void) fclose(serv);
        return;
    }

    if (p)
        fputs(buf, out);

    while ((n = fread(buf, 1, BSIZE_SP, serv)) != 0)
        (void) fwrite(buf, 1, n, out);

    /* We hope that positioning info + error messages < pipe size */
    while (fgets(buf, BSIZE_SP, err_outp))
        if (!strncmp("@@@", buf, 3)) {
            if (sscanf(buf, "@@@ %ld %d", &pos, &num) != 2) {
                fprintf(stderr, "Error reading rawdata: %s\n", buf);
                continue;
            }
            if (fseek(out, pos, SEEK_SET))
                fprintf(stderr,
                        "Error adjusting rawfile: write \"%d\" at %ld\n",
                        num, pos);
            else
                fprintf(out, "%d", num);
        } else {
            fprintf(stderr, "%s", buf);
        }

    (void) fclose(out);
    (void) fclose(serv);
    (void) fclose(err_outp);

    pl = raw_read(outfile);
    if (pl)
        plot_add(pl);
    (void) unlink(outfile);
    fprintf(stderr, "done.\n");
}

#else

void
com_aspice(wordlist *wl)
{
    NG_IGNORE(wl);
    fprintf(cp_err, "Asynchronous spice jobs are not available.\n");
}


void
com_jobs(wordlist *wl)
{
    NG_IGNORE(wl);
    fprintf(cp_err, "Asynchronous spice jobs are not available.\n");
}


void
ft_checkkids(void)
{
}


void
com_rspice(wordlist *wl)
{
    NG_IGNORE(wl);
    fprintf(cp_err, "Remote spice jobs are not available.\n");
}

#endif
