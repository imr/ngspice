/* Common configuration file for ng-spice and nutmeg */

#include <config.h>
#include "conf.h"

/*  The GNU configure system should set PACKAGE_BUGREPORT
	use this if it does, otherwise fall back to a hardcoded address */
#ifdef PACKAGE_BUGREPORT
#define BUG_ADDRESS		PACKAGE_BUGREPORT
#else
#define BUG_ADDRESS		"ngspice-bugs@lists.sourceforge.net"
#endif

char	Spice_Version[] = PACKAGE_VERSION;
char	Spice_Notice[] = "Please submit bug-reports to: " BUG_ADDRESS;
char	Spice_Build_Date[] = NGSPICEBUILDDATE;

char	*Spice_Exec_Dir	= NGSPICEBINDIR;
char	*Spice_Lib_Dir	= NGSPICEDATADIR;

#ifdef __MINGW32__
char	*Def_Editor	= "notepad.exe";
int	AsciiRawFile	= 1;
#else
char	*Def_Editor	= "vi";
int	AsciiRawFile	= 0;
#endif

char	*Bug_Addr	= BUG_ADDRESS;
char	*Spice_Host	= "";
char	*Spiced_Log	= "";
