/* Common configuration file for ng-spice and nutmeg */

#include <config.h>

#include "conf.h"

char	Spice_Version[] = VERSION;
char	Spice_Notice[] = "Please submit bug-reports to: ngspice-bugs@lists.sourceforge.net";
char	Spice_Build_Date[] = NGSPICEBUILDDATE;
/*
char	*Spice_Exec_Dir	= "/spice_win/bin";
char	*Spice_Lib_Dir	= "/spice_win/lib";
*/
char	*Spice_Exec_Dir	= NGSPICEBINDIR;
char	*Spice_Lib_Dir	= NGSPICEDATADIR;

#ifdef __MINGW32__
char	*Def_Editor	= "notepad.exe";
int	AsciiRawFile	= 1;
#else
char	*Def_Editor	= "vi";
int	AsciiRawFile	= 0;
#endif


char	*Bug_Addr	= "ngspice-bugs@lists.sourceforge.net";
char	*Spice_Host	= "";
char	*Spiced_Log	= "";

