#ifndef ngspice_COM_CHECKPOINT_H
#define ngspice_COM_CHECKPOINT_H

/* Enhancement-131: transient checkpoint / restart commands. */
void com_savestate(wordlist *wl);
void com_loadstate(wordlist *wl);

#endif
