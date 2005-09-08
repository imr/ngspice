#ifndef __CONF_H
#define __CONF_H

#ifdef __MINGW32__
#define NGSPICEBINDIR "C:\\msys\\1.0\\local\\bin"
#define NGSPICEDATADIR "C:\\msys\\1.0\\local\\share\\ng-spice-rework"
#endif

char Spice_Version[];
char Spice_Notice[];
char Spice_Build_Date[];
char *Spice_Exec_Dir;
char *Spice_Lib_Dir;
char *Def_Editor;
int AsciiRawFile;

char *Bug_Addr;
char *Spice_Host;
char *Spiced_Log;

#endif
