/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Routines to do execution of unix commands.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "unixcom.h"
#include "../frontend/streams.h"

#ifdef HAVE_VFORK_H

/* The only reason this exists is efficiency */

#  ifdef HAVE_SYS_DIR_H
#    include <sys/types.h>
#    include <sys/dir.h>
#  else

#    ifdef HAVE_DIRENT_H
#      include <sys/types.h>
#      include <dirent.h>
#      ifndef direct
#        define direct dirent
#      endif
#    endif
#  endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include <sys/file.h>
#include <sys/wait.h>
#include <signal.h>


static bool tryexec(char *name, char *argv[]);
static int hash(register char *str);


struct hashent {
    char *h_name;
    char *h_path;
    struct hashent *h_next;
};


#define HASHSIZE 256

static struct hashent *hashtab[HASHSIZE];
static char *dirbuffer;
static int dirlength, dirpos;


/* Create the hash table for the given search path. pathlist is a : seperated
 * list of directories. If docc is TRUE, then all the commands found are
 * added to the command completion lists.
 */

void
cp_rehash(char *pathlist, bool docc)
{
    register int i;
    struct hashent *hh, *ht;
    char buf[BSIZE_SP], pbuf[BSIZE_SP], *curpath;
    DIR *pdir;
    struct direct *entry;

    /* First clear out the old hash table. */
    for (i = 0; i < HASHSIZE; i++) {
        for (hh = hashtab[i]; hh; hh = ht) {
            ht = hh->h_next;
            /* Don't free any of the other stuff -- it is too
             * strange.
             */
            tfree(hh);
        }
        hashtab[i] = NULL;
    }

    while (pathlist && *pathlist) {
        /* Copy one path to buf. We have to make sure that the path
         * is a full path name.
         */
        if (*pathlist == '/') {
            i = 0;
        } else {
#ifdef HAVE_GETWD
            (void) getwd(buf);
#else
#  ifdef HAVE_GETCWD
            (void) getcwd(buf, sizeof(buf));
#  else
            *buf = '\0';
#  endif
#endif
            i = strlen(buf);
        }
        while (*pathlist && (*pathlist != ':'))
            buf[i++] = *pathlist++;
        while (*pathlist == ':')
            pathlist++;
        buf[i] = '\0';

        curpath = copy(buf);
        if (!(pdir = opendir(curpath)))
            continue;
        while (entry = readdir(pdir)) {
            (void) strcpy(pbuf, curpath);
            (void) strcat(pbuf, "/");
            (void) strcat(pbuf, entry->d_name);
            /* Now we could make sure that it is really an
             * executable, but that is too slow
             * (as if "we" really cared).
             */
            hh = TMALLOC(struct hashent, 1);
            hh->h_name = copy(entry->d_name);
            hh->h_path = curpath;
            i = hash(entry->d_name);
            /* Make sure this goes at the end, with
             * possible duplications of names.
             */
            if (hashtab[i]) {
                ht = hashtab[i];
                while (ht->h_next)
                    ht = ht->h_next;
                ht->h_next = hh;
            } else {
                hashtab[i] = hh;
            }

            if (docc) {
                /* Add to completion hash table. */
                cp_addcomm(entry->d_name, (long) 0, (long) 0, (long) 0, (long) 0);
            }
        }
        closedir(pdir);
    }
}


/* The return value is FALSE if no command was found, and TRUE if it was. */

bool
cp_unixcom(wordlist *wl)
{
    int i;
    register struct hashent *hh;
    register char *name;
    char **argv;
    char buf[BSIZE_SP];

    if (!wl)
        return (FALSE);
    name = wl->wl_word;
    argv = wl_mkvec(wl);
    if (cp_debug) {
        printf("name: %s, argv: ", name);
        wl_print(wl, stdout);
        printf(".\n");
    }
    if (strchr(name, '/'))
        return (tryexec(name, argv));
    i = hash(name);
    for (hh = hashtab[i]; hh; hh = hh->h_next)
        if (eq(name, hh->h_name)) {
            (void) sprintf(buf, "%s/%s", hh->h_path, hh->h_name);
            if (tryexec(buf, argv))
                return (TRUE);
        }
    return (FALSE);
}


static bool
tryexec(char *name, char *argv[])
{
#  ifdef HAVE_SYS_WAIT_H
    int status;
#  else
    union wait status;
#  endif

    int pid, j;
    void (*svint)(), (*svquit)(), (*svtstp)();

    pid = vfork();
    if (pid == 0) {
        fixdescriptors();
        (void) execv(name, argv);
        (void) _exit(120);  /* A random value. */
        /* NOTREACHED */
    } else {
        svint = signal(SIGINT, SIG_DFL);
        svquit = signal(SIGQUIT, SIG_DFL);
        svtstp = signal(SIGTSTP, SIG_DFL);
        do {
            j = wait(&status);
        } while (j != pid);
        (void) signal(SIGINT, (SIGNAL_FUNCTION) svint);
        (void) signal(SIGQUIT, (SIGNAL_FUNCTION) svquit);
        (void) signal(SIGTSTP, (SIGNAL_FUNCTION) svtstp);
    }

    if (WTERMSIG(status) == 0 && WEXITSTATUS(status) == 120)
        /*if ((status.w_termsig == 0) && (status.w_retcode == 120)) */
        return (FALSE);
    else
        return (TRUE);
}


static int
hash(register char *str)
{
    register int i = 0;

    while (*str)
        i += *str++;

    return (i % HASHSIZE);
}


/* Debugging. */

void
cp_hstat(void)
{
    struct hashent *hh;
    int i;

    for (i = 0; i < HASHSIZE; i++)
        for (hh = hashtab[i]; hh; hh = hh->h_next)
            fprintf(cp_err, "i = %d, name = %s, path = %s\n",
                    i, hh->h_name, hh->h_path);
}


#else


void
cp_rehash(char *pathlist, bool docc)
{
    NG_IGNORE(docc);
    NG_IGNORE(pathlist);
}


bool
cp_unixcom(wordlist *wl)
{
    char *s = wl_flatten(wl);

    if (system(s))
        return (FALSE);
    else
        return (TRUE);
}

#endif
