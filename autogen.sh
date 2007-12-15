#!/bin/sh
# Configuration script for ngspice. 
#
# This script performs initial configuration of ngspice source 
# package.
#
#
# $Id$
#

PROJECT=ngspice
TEST_TYPE=-f
FILE=DEVICES

# ADMS variables

ADMSDIR=src/spicelib/devices/adms
XMLPATH=src/spicelib/devices/adms/admst
ADMSXML=admsXml
ADMS=0

# Exit variable
DIE=0



help()
{
 echo
 echo "$PROJECT autogen.sh help"
 echo
 echo "--adms     -a: enables adms feature"
 echo "--help     -h: print this file"
 echo "--version  -v: print version"
 echo
}

version()
{
echo
echo "$PROJECT autogen.sh 1.0"
echo
}


check_autoconf()
{
(autoconf --version) < /dev/null > /dev/null 2>&1 || {
	echo
	echo "You must have autoconf installed to compile $PROJECT."
	echo "See http://www.gnu.org/software/automake/"
	echo "(newest stable release is recommended)"
	DIE=1
        }

(libtoolize --version) < /dev/null > /dev/null 2>&1 || {
	echo
	echo "You must have libtool installed to compile $PROJECT."
	echo "See http://www.gnu.org/software/libtool/"
	echo "(newest stable release is recommended)"
	DIE=1
        }

(automake --version) < /dev/null > /dev/null 2>&1 || {
	echo
	echo "You must have automake installed to compile $PROJECT."
	echo "See http://www.gnu.org/software/automake/"
	echo "(newest stable release is recommended)"
	DIE=1
        }
}


check_adms()
{
(admsXml --version)   < /dev/null > /dev/null 2>&1 || {
        echo
	echo "You must have admsXml installed to compile adms models."
	echo "See http://mot-adms.sourceforge.net"
	echo "(newest stable release is recommended)"
        DIE=1
}
}

case "$1" in
    "--adms" | "-a")
    check_adms 
    ADMS=1
    ;;
    
    "--help" | "-h")
    help
    exit 0
    ;;
    
    "--version" | "-v")
    version
    exit 0
    ;;
    
    *)
    ;;
    esac


check_autoconf

if test "$DIE" -eq 1; then
	exit 1
fi

test $TEST_TYPE $FILE || {
	echo "You must run this script in the top-level $PROJECT directory"
	exit 1
}



if test "$ADMS" -eq 1; then

# Build admsXml arguments list
#    for xml in `ls $XMLPATH | grep .xml`; do
#       if [ "$xml" != "ngspiceVersion.xml" ]; then
#	    XMLARG="$XMLARG -e ../admst/$xml"
#	    fi
#    done 

# Prepend ngspiceVersion.xml    
#    XMLARG="-e ../admst/ngspiceVersion.xml $XMLARG"

currentdir=`pwd`

for adms_dir in `ls $ADMSDIR`
do
  if [ -d "$ADMSDIR/$adms_dir" ]; then
   
   case "$adms_dir" in
      "CVS")
      echo "Skipping CVS"
      ;;
      
      "admst")
      echo "Skipping scripts dir"
      
      ;;
      
      *)
      echo "Entering into directory: $adms_dir"
      echo "-->"$ADMSDIR/$adms_dir
      cd $ADMSDIR/$adms_dir
      file=`ls admsva/*.va`
      $ADMSXML $file -Iadmsva -e ../admst/ngspiceVersion.xml \
      -e ../admst/ngspiceMakefile.am.xml
      
      cd $currentdir
      ;;
   esac
  fi 
done
fi

echo "Running libtoolize"
libtoolize --copy --force
if [ $? -ne 0 ];then  echo "libtoolize failed"; exit 1 ; fi

echo "Running aclocal $ACLOCAL_FLAGS"
aclocal $ACLOCAL_FLAGS
if [ $? -ne 0 ]; then  echo "aclocal failed"; exit 1 ; fi

# optional feature: autoheader
(autoheader --version)  < /dev/null > /dev/null 2>&1
if [ $? -eq 0 ]; then
  echo "Running autoheader"
  autoheader
  if [ $? -ne 0 ]; then  echo "autoheader failed"; exit 1 ; fi
fi

echo "Running automake -Wall --copy --add-missing"
automake -Wall --copy --add-missing $am_opt
if [ $? -ne 0 ]; then  echo "automake failed"; exit 1 ; fi

echo "Running autoconf"
autoconf
if [ $? -ne 0 ]; then  echo "autoconf failed"; exit 1 ; fi

echo "Success."
