#ifndef ngspice_COM_COMMANDS_H
#define ngspice_COM_COMMANDS_H

void com_showmod(wordlist *wl);
void com_show(wordlist *wl);
void com_alter(wordlist *wl);
void com_altermod(wordlist *wl);
void com_alterparam(wordlist *wl);
void com_meas(wordlist *wl);
void com_sysinfo(wordlist *wl);
void com_check_ifparm(wordlist *wl);

#endif
