/* Common configuration file for ng-spice and nutmeg */

#include "ngspice/config.h"
#include "conf.h"

/*  The GNU configure system should set PACKAGE_BUGREPORT
	use this if it does, otherwise fall back to a hardcoded address */
#ifdef PACKAGE_BUGREPORT
#define BUG_ADDRESS		PACKAGE_BUGREPORT
#else
#define BUG_ADDRESS		"https://ngspice.sourceforge.io/bugrep.html"
#endif

char	Spice_Version[] = PACKAGE_VERSION;
char	Spice_Notice[] = "Please file your bug-reports at " BUG_ADDRESS;
#ifndef _MSC_VER
char	Spice_Build_Date[] = NGSPICEBUILDDATE;
#else
char	Spice_Build_Date[] = __DATE__"   "__TIME__;
#endif

char	Spice_Manual[] = "Please get your ngspice manual from https://ngspice.sourceforge.io/docs.html";

char	*Spice_Exec_Dir	= NGSPICEBINDIR;
char	*Spice_Lib_Dir	= NGSPICEDATADIR;

#if defined (__MINGW32__) || defined (_MSC_VER)
char	*Def_Editor	= "notepad.exe";
#else
char	*Def_Editor	= "vi";
#endif
int	AsciiRawFile	= 0;

char	*Bug_Addr	= BUG_ADDRESS;
char	*Spice_Host	= "";
char	*Spiced_Log	= "";
