# Process this file with automake

EXTRA_DIST = README examples icm

## This is removed because icm relies upon the existance of all other
## libs.  It is currently compiled manually, last.
##SUBDIRS = mif cm enh evt ipc idn icm

SUBDIRS = mif cm enh evt ipc idn cmpp icm


dist-hook:
	rm -rf `find $(distdir)/examples -name CVS`
	rm -rf `find $(distdir)/icm -name CVS`
	rm -rf `find $(distdir)/icm -name .cvsignore`
	rm -rf `find $(distdir)/icm -name .deps`
	rm -rf `find $(distdir)/icm -name *.o`


MAINTAINERCLEANFILES = Makefile.in

