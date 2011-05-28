
#
#	Example Makefile to run tests and check results.
#
#	This is an example makefile for running QA tests on a
#	model and then checking the simulated results against
#	reference results. A separate target is defined for each
#	variant of the model. The program runQaTests.pl runs the
#	tests, and that program expects a perl module SIMULATOR.pm
#	to be provided for each simulator that is tested.
#	Examples of these are provided.
#

qaSpecFile		=	qaSpec
qaResultsDirectory	=	results
testProgramName		=	../../bin/runQaTests.pl
testProgramFlags        =       -nwV
#testProgramFlags        =       -d

help:
	@echo "" ; \
	echo "Valid targets are:" ; \
	echo "" ; \
	echo "all                run tests and compare results for all simulators" ; \
	echo "" ; \
	echo "spice              run tests and compare results spice" ; \
	echo "ngspice            run tests and compare results ngspice" ; \
	echo "" ; \
	echo "clean              remove all previously generated simulation results"; \
	echo "" ; \
	echo "NOTE: if test results exist they are not resimulated" ; \
	echo "NOTE: to force resimulation run \"make clean\" first" ; \
	echo ""

all:	spice ngspice

#####
##### spice tests
#####

spice:
	@-echo ""; \
	localPlatform=`$(testProgramName) -platform` ; \
	localVersion=`$(testProgramName) -sv -s spice $(qaSpecFile)` ; \
	localVersionAndPlatform=$$localVersion._.$$localPlatform ; \
	echo "******"; \
	echo "****** $(qaSpecFile) tests for spice"; \
	echo "****** (for version $$localVersion on platform $$localPlatform)"; \
	echo "******"; \
	for test in `$(testProgramName) -lt -s spice $(qaSpecFile)` ; \
	do \
	    echo ""; \
	    echo "****** Checking test (spice): $$test" ; \
	    for variant in `$(testProgramName) -lv -s spice $(qaSpecFile)` ; \
	    do \
		$(testProgramName) $(testProgramFlags) -s spice -r -t $$test -var $$variant $(qaSpecFile) ; \
	    done ; \
	done ; \
	for version in `ls -C1 $(qaResultsDirectory)/spice` ; \
	do \
	    for platform in `ls -C1 $(qaResultsDirectory)/spice/$$version` ; \
	    do \
		versionAndPlatform=$$version._.$$platform ; \
		if [ $$versionAndPlatform = $$localVersionAndPlatform ] ; \
		then \
		    break ; \
		fi ; \
		echo "" ; \
		echo "******"; \
		echo "****** Comparing previously run $(qaSpecFile) tests for spice"; \
		echo "****** (for version $$version on platform $$platform)"; \
		echo "******"; \
		for test in `$(testProgramName) -lt -s spice $(qaSpecFile)` ; \
		do \
		    echo ""; \
		    echo "****** Checking test (spice): $$test" ; \
		    for variant in `$(testProgramName) -lv -s spice $(qaSpecFile)` ; \
		    do \
			$(testProgramName) -c $$version $$platform -s spice -t $$test -var $$variant $(qaSpecFile) ; \
		    done ; \
		done ; \
	    done ; \
	done

#####
##### ngspice tests
#####

ngspice:
	@-echo ""; \
	localPlatform=`$(testProgramName) -platform` ; \
	localVersion=`$(testProgramName) -sv -s ngspice $(qaSpecFile)` ; \
	localVersionAndPlatform=$$localVersion._.$$localPlatform ; \
	echo "******"; \
	echo "****** $(qaSpecFile) tests for ngspice"; \
	echo "****** (for version $$localVersion on platform $$localPlatform)"; \
	echo "******"; \
	for test in `$(testProgramName) -lt -s ngspice $(qaSpecFile)` ; \
	do \
	    echo ""; \
	    echo "****** Checking test (ngspice): $$test" ; \
	    for variant in `$(testProgramName) -lv -s ngspice $(qaSpecFile)` ; \
	    do \
		$(testProgramName) $(testProgramFlags) -s ngspice -r -t $$test -var $$variant $(qaSpecFile) ; \
	    done ; \
	done ; \
	for version in `ls -C1 $(qaResultsDirectory)/ngspice` ; \
	do \
	    for platform in `ls -C1 $(qaResultsDirectory)/ngspice/$$version` ; \
	    do \
		versionAndPlatform=$$version._.$$platform ; \
		if [ $$versionAndPlatform = $$localVersionAndPlatform ] ; \
		then \
		    break ; \
		fi ; \
		echo "" ; \
		echo "******"; \
		echo "****** Comparing previously run $(qaSpecFile) tests for ngspice"; \
		echo "****** (for version $$version on platform $$platform)"; \
		echo "******"; \
		for test in `$(testProgramName) -lt -s ngspice $(qaSpecFile)` ; \
		do \
		    echo ""; \
		    echo "****** Checking test (ngspice): $$test" ; \
		    for variant in `$(testProgramName) -lv -s ngspice $(qaSpecFile)` ; \
		    do \
			$(testProgramName) -c $$version $$platform -s ngspice -t $$test -var $$variant $(qaSpecFile) ; \
		    done ; \
		done ; \
	    done ; \
	done

clean:
	@/bin/rm -rf $(qaResultsDirectory)/spice b3v3check.log
	@/bin/rm -rf $(qaResultsDirectory)/ngspice ngspiceCkt*
