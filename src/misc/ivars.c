/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
**********/

#include "ngspice.h"
#include "ivars.h"

#ifdef HAVE_STRING_H
#include <string.h>
#endif /* HAVE_STRING_H */

#include <stdlib.h>
#include <stdio.h>

char *Spice_Path;
char *News_File;
char *Default_MFB_Cap;
char *Help_Path;
char *Lib_Path;


static void
env_overr(char **v, char *e)
{
    char *p;
    if (v && e && (p = getenv(e)))
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
	*p = (char *) tmalloc(strlen(buffer)+1);
	sprintf(*p,"%s",buffer);
	/* asprintf(p, "%s", buffer); */
    }
    else{
	*p = (char *) tmalloc(strlen(path_prefix) + 
			strlen(DIR_PATHSEP) + strlen(var_dir) + 1);
	sprintf(*p, "%s%s%s", path_prefix, DIR_PATHSEP, var_dir); 
	/* asprintf(p, "%s%s%s", path_prefix, DIR_PATHSEP, var_dir); */
    }
#endif /* HAVE_ASPRINTF */
}

void
ivars(void)
{
	
    char *temp=NULL;
	
    env_overr(&Spice_Exec_Dir, "SPICE_EXEC_DIR");
    env_overr(&Spice_Lib_Dir, "SPICE_LIB_DIR");


    mkvar(&News_File, Spice_Lib_Dir, "news", "SPICE_NEWS");
    mkvar(&Default_MFB_Cap, Spice_Lib_Dir, "mfbcap", "SPICE_MFBCAP");
    mkvar(&Help_Path, Spice_Lib_Dir, "helpdir", "SPICE_HELP_DIR");
    mkvar(&Lib_Path, Spice_Lib_Dir, "scripts", "SPICE_SCRIPTS");
    mkvar(&Spice_Path, Spice_Exec_Dir, "ngspice", "SPICE_PATH");

    env_overr(&Spice_Host, "SPICE_HOST");
    env_overr(&Bug_Addr, "SPICE_BUGADDR");
    env_overr(&Def_Editor, "SPICE_EDITOR");
    env_overr(&temp, "SPICE_ASCIIRAWFILE");
    
    if(temp)
       AsciiRawFile = atoi(temp);
    
}

void
cleanvars(void)
{
    tfree(News_File);
    tfree(Default_MFB_Cap);
    tfree(Help_Path);
    tfree(Lib_Path);
    tfree(Spice_Path);
}
