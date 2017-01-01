/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
**********/

#if defined(__MINGW32__) || defined(_MSC_VER)
#include <Windows.h>
#endif

#include "ngspice/ngspice.h"
#include "ivars.h"
#include "../misc/util.h" /* ngdirname() */

char *Spice_Path;
char *News_File;
char *Help_Path;
char *Lib_Path;
char *Inp_Path;
char *Outp_Path;

#if defined (SHARED_MODULE) && defined (HAS_RELPATH)
#if defined(__MINGW32__) || defined(_MSC_VER)

static char *
get_abs_path(void)
{
    char path[MAX_PATH];
    HMODULE hm = NULL;

    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCTSTR)get_abs_path,
                            &hm))
    {
        int ret = GetLastError();
        fprintf(stderr, "GetModuleHandle returned %d\n", ret);
    }
    GetModuleFileNameA(hm, path, sizeof(path));
    return ngdirname(path);
}

#else

#include <dlfcn.h>

static char *
get_abs_path(void)
{
    Dl_info info;

    // Lookup the path of the library given the pointer to a function from within the library
    if (dladdr(get_abs_path, &info))
        return ngdirname(info.dli_fname);
    return NULL;
}

#endif
#endif


static bool
env_overr(char **v, char *e)
{
    char *p;
    if (v && e && (p = getenv(e)) != NULL) {
        *v = p;
        return (TRUE);
    }
    else
        return (FALSE);
}

static void
mkvar(char **p, char *path_prefix, char *var_dir, char *env_var)
{
    /* Override by environment variables */
    char *buffer = getenv(env_var);

    if (buffer)
        *p = tprintf("%s",buffer);
    else
        *p = tprintf("%s%s%s", path_prefix, DIR_PATHSEP, var_dir);
}

void
ivars(char *argv0)
{
    char *temp=NULL;
    /* $dprefix has been set to /usr/local or C:/Spice (Windows) in configure.ac,
    NGSPICEBINDIR has been set to $dprefix/bin in config.h,
    NGSPICEDATADIR has been set to $dprefix/share/ngspice in config.h,
    Spice_Exec_Dir has been set to NGSPICEBINDIR in conf.c,
    may be overridden here by environmental variable SPICE_EXEC_DIR
    Spice_Lib_Dir has been set to NGSPICEDATADIR in conf.c;
    may be overridden here by absolute path to library location if shared library
    may be overridden here by environmental variable SPICE_LIB_DIR */

#if defined (SHARED_MODULE) && defined (HAS_RELPATH)
    if (!env_overr(&Spice_Lib_Dir, "SPICE_LIB_DIR")) {
        /* get the absolute path of the ngspice library as reference to other paths */
        char *Shared_Dir = get_abs_path();
        /* Here we determine the path relative to the library directory */
        Spice_Lib_Dir = temp = tprintf("%s/../share/ngspice", Shared_Dir);
#ifdef OUTDEBUG
        printf("Shared lib is located in %s\n", Shared_Dir);
#endif
        tfree(Shared_Dir);
    }
#else
    env_overr(&Spice_Lib_Dir, "SPICE_LIB_DIR");
#endif
    env_overr(&Spice_Exec_Dir, "SPICE_EXEC_DIR");

    /* for printing a news file */
    mkvar(&News_File, Spice_Lib_Dir, "news", "SPICE_NEWS");
    /* help directory, not used in Windows mode */
    mkvar(&Help_Path, Spice_Lib_Dir, "helpdir", "SPICE_HELP_DIR");
    /* where spinit is found */
    mkvar(&Lib_Path, Spice_Lib_Dir, "scripts", "SPICE_SCRIPTS");
    /* used to call ngspice with aspice command, not used in Windows mode */
    mkvar(&Spice_Path, Spice_Exec_Dir, "ngspice", "SPICE_PATH");
    tfree(temp);
    /* may be used to store input files (*.lib, *.include, ...) */
    /* get directory where ngspice resides */
#if defined (HAS_WINGUI) || defined (__MINGW32__) || defined (_MSC_VER)
    {
        char *ngpath = ngdirname(argv0);
        /* set path either to <ngspice-bin-directory>/input or,
        if set, to environment variable NGSPICE_INPUT_DIR */
        mkvar(&Inp_Path, ngpath, "input", "NGSPICE_INPUT_DIR");
        tfree(ngpath);
    }
#else
    NG_IGNORE(argv0);
    /* set path either to environment variable NGSPICE_INPUT_DIR
    (if given) or to NULL */
    env_overr(&Inp_Path, "NGSPICE_INPUT_DIR");
    Inp_Path = copy(Inp_Path); /* allow tfree */
#endif
    env_overr(&Outp_Path, "NGSPICE_OUTPUT_DIR");
    Outp_Path = copy(Outp_Path); /* allow tfree */
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
    tfree(Help_Path);
    tfree(Lib_Path);
    tfree(Spice_Path);
    tfree(Inp_Path);
    tfree(Outp_Path);
}
