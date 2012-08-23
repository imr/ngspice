/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
**********/

#include "ngspice/ngspice.h"
#include "ivars.h"
#include "../misc/util.h" /* ngdirname() */

#ifdef HAVE_ASPRINTF
#ifdef HAVE_LIBIBERTY_H /* asprintf */
#include <libiberty.h>
#elif defined(__MINGW32__) || defined(__SUNPRO_C) /* we have asprintf, but not libiberty.h */
#include <stdarg.h>
extern int asprintf(char **out, const char *fmt, ...);
extern int vasprintf(char **out, const char *fmt, va_list ap);
#endif
#endif

char *Spice_Path;
char *News_File;
char *Default_MFB_Cap;
char *Help_Path;
char *Lib_Path;
char *Inp_Path;


static void
env_overr(char **v, char *e)
{
    char *p;
    if (v && e && (p = getenv(e)) != NULL)
        *v = p;
}

static void
mkvar(char **p, char *path_prefix, char *var_dir, char *env_var)
{
    char *buffer;

    /* Override by environment variables */
    buffer = getenv(env_var);

#ifdef HAVE_ASPRINTF
    if (buffer)
        asprintf(p, "%s", buffer);
    else
        asprintf(p, "%s%s%s", path_prefix, DIR_PATHSEP, var_dir);
#else /* ~ HAVE_ASPRINTF */
    if (buffer){
        *p = TMALLOC(char, strlen(buffer) + 1);
        sprintf(*p,"%s",buffer);
        /* asprintf(p, "%s", buffer); */
    }
    else{
        *p = TMALLOC(char, strlen(path_prefix) + strlen(DIR_PATHSEP) + strlen(var_dir) + 1);
        sprintf(*p, "%s%s%s", path_prefix, DIR_PATHSEP, var_dir); 
        /* asprintf(p, "%s%s%s", path_prefix, DIR_PATHSEP, var_dir); */
    }
#endif /* HAVE_ASPRINTF */
}

void
ivars(char *argv0)
{
    char *temp=NULL;
	 /* $dprefix has been set to /usr/local or C:/Spice (Windows) in configure.ac,
    NGSPICEBINDIR has been set to $dprefix/bin in configure.ac, 
    Spice_Exec_Dir has been set to NGSPICEBINDIR in conf.c,
    may be overridden here by environmental variable SPICE_EXEC_DIR */
    env_overr(&Spice_Exec_Dir, "SPICE_EXEC_DIR");
    env_overr(&Spice_Lib_Dir, "SPICE_LIB_DIR");

    /* for printing a news file */
    mkvar(&News_File, Spice_Lib_Dir, "news", "SPICE_NEWS");
    /* not used in ngspice */
    mkvar(&Default_MFB_Cap, Spice_Lib_Dir, "mfbcap", "SPICE_MFBCAP");
    /* help directory, not used in Windows mode */
    mkvar(&Help_Path, Spice_Lib_Dir, "helpdir", "SPICE_HELP_DIR");
    /* where spinit is found */
    mkvar(&Lib_Path, Spice_Lib_Dir, "scripts", "SPICE_SCRIPTS");
    /* used to call ngspice with aspice command, not used in Windows mode */
    mkvar(&Spice_Path, Spice_Exec_Dir, "ngspice", "SPICE_PATH");
    /* may be used to store input files (*.lib, *.include, ...) */
    /* get directory where ngspice resides */
#if defined (HAS_WINDOWS) || defined (__MINGW32__) || defined (_MSC_VER)
    {
        char *ngpath = ngdirname(argv0);
        /* set path either to <ngspice-bin-directory>/input or,
        if set, to environment variable NGSPICE_INPUT_DIR */
        mkvar(&Inp_Path, ngpath, "input", "NGSPICE_INPUT_DIR");
    }
#else
    NG_IGNORE(argv0);
    /* set path either to environment variable NGSPICE_INPUT_DIR
    (if given) or to NULL */
    env_overr(&Inp_Path, "NGSPICE_INPUT_DIR");
#endif
    env_overr(&Spice_Host, "SPICE_HOST"); /* aspice */
    env_overr(&Bug_Addr, "SPICE_BUGADDR");
    env_overr(&Def_Editor, "SPICE_EDITOR");
    
    /* Set raw file mode, 0 by default (binary) set in conf.c, 
    may be overridden by environmental
    variable, not sure if acknowledged everywhere in ngspice */
    env_overr(&temp, "SPICE_ASCIIRAWFILE");
    if(temp)
       AsciiRawFile = atoi(temp);
    
}

void
destroy_ivars(void)
{
    tfree(News_File);
    tfree(Default_MFB_Cap);
    tfree(Help_Path);
    tfree(Lib_Path);
    tfree(Spice_Path);
    tfree(Inp_Path);
}
