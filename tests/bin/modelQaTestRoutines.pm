
#
# perl module with subroutines used as part of compact model QA (runQaTests.pl)
#

#
#  Rel  Date            Who             Comments
# ====  ==========      =============   ========
#  1.4                  Colin McAndrew  Q added as AC output option
#                                       Unused AC simulations skipped
#  1.2  06/30/06        Colin McAndrew  Floating node support added
#                                       Noise simulation added
#                                       Other general cleanup
#  1.0  04/13/06        Colin McAndrew  Initial version
#

package modelQa;
#use strict; # hirearchical read cannot be done while "strict refs" is in use

#
#   This subroutine processes the generic (not test specific) setup information.
#   It sets the information in global variables.
#

sub processSetup {
    my(@Setup)=@_;
    my(@Field,$pin,$temperature);

    undef(%main::isLinearScale);
    undef(%main::isAreaScale);
    undef(%main::DefaultTemperature);
    $main::doMfactorTest=0;
    $main::doScaleTest  =0;
    $main::doShrinkTest =0;
    $main::doPinFlipTest=0;
    $main::doPNFlipTest =0;
    @main::Pin=();
    foreach (@Setup) {
        @Field=split(/[\s,]+/,$_);
        if (s/^keyLetter\s+//i) {
            if ($_ !~ /^[a-zA-Z]$/) {
                die("ERROR: bad keyLetter specification, stopped");
            }
            $main::keyLetter=$_;
            next;
        }
        if (s/^verilogaFile\s+//i) {
            $main::verilogaFile=$_;
            if (! -f $main::verilogaFile) {
                die("ERROR: cannot find file $main::verilogaFile, stopped");
            }
            next;
        }
        if (s/^(pins|terminals)\s+//i) {
            push(@main::Pin,@Field[1..$#Field]);
            foreach $pin (@main::Pin) {
                $main::isPin{$pin}=1;
                if ($pin !~ /^[a-zA-Z][a-zA-Z0-9]*$/) { # underscores are not allowed
                    die("ERROR: bad pin name specification $pin, stopped");
                }
            }
            next;
        }
        if (/^symmetric(Pins|Terminals)/i) {
            if ($#Field != 2) {
                die("ERROR: bad symmetricPins specification, stopped");
            }
            $main::isSymmetryPin{$Field[1]}=1;
            $main::isSymmetryPin{$Field[2]}=1;
            $main::flipPin{$Field[1]}=$Field[2];
            $main::flipPin{$Field[2]}=$Field[1];
            $main::doPinFlipTest=1;
            next;
        }
        if (s/^pTypeSelectionArguments\s+//i) {
            s/\s*=\s*/=/g;
            $main::pTypeSelectionArguments=$_;
            next;
        }
        if (s/^(nType|type|model)SelectionArguments\s+//i) {
            s/\s*=\s*/=/g;
            $main::nTypeSelectionArguments=$_;
            next;
        }
        if (/^checkPolarity/i) {
            if ($#Field<1 || ($Field[1] !~ /^[yn01]/i)) {
                die("ERROR: bad checkPolarity specification, stopped");
            }
            if ($Field[1] =~ /^[y1]/i) {
                $main::doPNFlipTest=1;
            } else {
                $main::doPNFlipTest=0;
            }
            next;
        }
        if (/^scaleParameters/i) {
            foreach (@Field[1..$#Field]) {
                if (/^m$/i)      {$main::doMfactorTest=1}
                if (/^scale$/i)  {$main::doScaleTest  =1}
                if (/^shrink$/i) {$main::doShrinkTest =1}
            }
            next;
        }
        if (/^linearScale/i) {
            foreach (@Field[1..$#Field]) {
                $main::isLinearScale{$_}=1;
            }
            next;
        }
        if (/^areaScale/i) {
            foreach (@Field[1..$#Field]) {
                $main::isAreaScale{$_}=1;
            }
            next;
        }
        if (/^temperature/i) {
            push(@main::DefaultTemperature,@Field[1..$#Field]);
            next;
        }
        if (/^float/i) {
            foreach (@Field[1..$#Field]) {
                $main::isGeneralFloatingPin{$_}=1;
            }
            next;
        }
        die("ERROR: unknown setup directive $Field[0], stopped");
    }
    if ($#main::Pin < 1) {
        die("ERROR: there must be two or more device pins, stopped");
    }
    foreach $pin (keys(%main::isSymmetryPin)) {
        if (!$main::isPin{$pin}) {
            die("ERROR: symmetry pin $pin is not a specified device pin, stopped");
        }
    }
    foreach $pin (keys(%main::isGeneralFloatingPin)) {
        if (!$main::isPin{$pin}) {
            die("ERROR: floating pin $pin is not a specified device pin, stopped");
        }
    }
    unless (@main::DefaultTemperature) {
        @main::DefaultTemperature=(27);
    }
    foreach $temperature (@main::DefaultTemperature) {
        if ($temperature!~/^$main::number$/) {
            die("ERROR: bad temperature value specified, stopped");
        }
    }
    if ($main::simulatorName =~ /mica|ads/i && defined($main::verilogaFile)) {
        $main::keyLetter="a";
    }
    if ($main::simulatorName =~ /hspice/i && defined($main::verilogaFile)) {
        $main::keyLetter="X";
    }
    if ($main::simulatorName =~ /spectre/i && !defined($main::keyLetter)) {
        $main::keyLetter="x";
    }
    if (!defined($main::keyLetter)) {
        die("ERROR: no keyLetter specified, stopped");
    }
    # if (!defined($main::nTypeSelectionArguments)) {
    #    die("ERROR: no model selection arguments specified, stopped");
    #}
    @main::Variants=("standard");
    if ($main::doPinFlipTest&&$main::doPNFlipTest) {
        push(@main::Variants,"Flip_N");
        push(@main::Variants,"noFlip_P");
        push(@main::Variants,"Flip_P");
        if (!defined($main::pTypeSelectionArguments)) {
            die("ERROR: no pType model selection arguments specified, stopped");
        }
    } elsif ($main::doPinFlipTest) {
        push(@main::Variants,"Flip_N");
    } elsif ($main::doPNFlipTest) {
        push(@main::Variants,"noFlip_P");
        if (!defined($main::pTypeSelectionArguments)) {
            die("ERROR: no pType model selection arguments specified, stopped");
        }
    }
    if ($main::doShrinkTest ) {push(@main::Variants,"shrink")}
    if ($main::doScaleTest  ) {push(@main::Variants,"scale")}
    if ($main::doMfactorTest) {push(@main::Variants,"m")}
}

#
#   This subroutine processes test specific setup information.
#   It sets the information in global variables.
#

sub processTestSpec {
    my(@Spec)=@_;
    my($i,$arg,$temperature,$bias,$pin,@Field,$oneOverTwoPi,$name,$value,%isAnalysisPin,%AlreadyHave,%IndexFor);

    $main::outputDc=0;$main::outputAc=0;$main::outputNoise=0;
    undef($main::biasSweepPin);undef($main::biasSweepSpec);undef(@main::BiasSweepList);
    undef($main::biasListPin);undef($main::biasListSpec);
    undef($main::frequencySpec);
    undef($main::Temperature);
    @main::InstanceParameters=();
    @main::Outputs=();
    @main::ModelParameters=();
    @main::TestVariants=@main::Variants;
    undef(%main::BiasFor);
    if (%main::isGeneralFloatingPin) {
        %main::isFloatingPin=%main::isGeneralFloatingPin;
    } else {
        undef(%main::isFloatingPin);
    }
    undef(%isAnalysisPin);
    undef(@main::Temperature);
    undef(%main::referencePinFor);
    foreach $pin (@main::Pin) {
        $main::needAcStimulusFor{$pin}=0;
    }
    foreach (@Spec) {
        if (s/^output[s]?\s+//i) {
            s/\(/ /g;s/\)//g;
            @Field=split(/[\s,]+/,$_);
            for ($i=0;$i<=$#Field;++$i) {
                if ($Field[$i] =~ /^[IV]$/) {
                    $main::outputDc=1;
                    ++$i;
                    if (!$main::isPin{$Field[$i]}) {
                        die("ERROR: pin $Field[$i] listed for DC output is not a specified pin, stopped");
                    }
                    push(@main::Outputs,$Field[$i]);
                }
                if ($Field[$i] =~ /^[CGQ]$/) {
                    $main::outputAc=1;
                    push(@main::Outputs,lc($Field[$i]));
                    ++$i;
                    if (!$main::isPin{$Field[$i]}) {
                        die("ERROR: pin $Field[$i] listed for AC output is not a specified pin, stopped");
                    }
                    $main::Outputs[$#main::Outputs].=" $Field[$i]";
                    $isAnalysisPin{$Field[$i]}=1;
                    ++$i;
                    if (!$main::isPin{$Field[$i]}) {
                        die("ERROR: pin $Field[$i] listed for AC output is not a specified pin, stopped");
                    }
                    $main::Outputs[$#main::Outputs].=" $Field[$i]";
                    $main::needAcStimulusFor{$Field[$i]}=1;
                    $isAnalysisPin{$Field[$i]}=1;
                }
                if ($Field[$i] =~ /^N$/) {
                    my($noi1,$noi2);
                    $main::outputNoise=1;
                    ++$i;
                    $noi1 = $Field[$i];
                    $isAnalysisPin{$noi1}=1;
                    if (!$main::isPin{$noi1}) {
                        die("ERROR: pin $noi1 listed for noise output is not a specified pin, stopped");
                    }
                    if ($#main::Outputs==0) {
                        die("ERROR: can only specify one pin for noise output, stopped");
                    }
                    push(@main::Outputs,$noi1);
                    ++$i;
                    if ($i <= $#Field) {
                        $noi2 = $Field[$i];
                        $isAnalysisPin{$noi2}=1;
                        if (!$main::isPin{$noi2}) {
                            die("ERROR: pin $noi2 listed for noise output is not a specified pin, stopped");
                        }
                        $main::outputNoise=2;
                        push(@main::Outputs,$noi2);
                    }
                }
            }
            next;
        }
        if (/^biases\s+/i) {
            s/V\s*\(\s*/V(/;s/\s*\)/)/;
            s/V\(([a-zA-Z0-9]+),([a-zA-Z0-9]+)\)/V($1_$2)/; # convert V(n1,n2) to V(n1_n2)
            @Field=split(/[\s,]+/,$_);
            for ($i=1;$i<=$#Field;++$i) {
                if ($Field[$i] !~ /=/) {
                    die("ERROR: biases specifications must be V(pin)=number, stopped");
                }
                $Field[$i]=~s/V\s*\(\s*//;$Field[$i]=~s/\s*\)//;
                ($pin,$bias)=split("=",$Field[$i]);
                if ($bias !~ /^$main::number$/) {
                    die("ERROR: biases specifications must be V(pin)=number, stopped");
                }
                if ($pin =~ s/_([a-zA-Z0-9]+)$//) { # this a V(n1,n2) pin (not ground) referenced bias
                    $main::referencePinFor{$pin}=$1;
                }
                $main::BiasFor{$pin}=$bias;
            }
            next;
        }
        if (s/^(biasSweep|sweepBias)\s+//i) {
            if (defined($main::biasSweepSpec)) {
                die("ERROR: can only have one biasSweep specification, stopped");
            }
            s/V\s*\(\s*//i;s/\s*\)//;
            @Field=split(/[=,\s]+/,$_);
            if ($#Field!=3) {
                die("ERROR: biasSweep specification must be V(pin)=start,stop,step, stopped");
            }
            $main::biasSweepPin=$Field[0];
            if (($Field[1] !~ /^$main::number$/) || ($Field[2] !~ /^$main::number$/) || ($Field[3] !~ /^$main::number$/)) {
                die("ERROR: biasSweep start,stop,step must be numbers, stopped");
            }
            if ($Field[1] == $Field[2]) {
                die("ERROR: biasSweep start and stop must be different, stopped");
            }
            if ($Field[3] == 0.0) {
                die("ERROR: biasStep must be non-zero, stopped");
            }
            @main::BiasSweepList=();
            if ($Field[2] > $Field[1]) {
                $Field[3]=abs($Field[3]);
                for ($bias=$Field[1];$bias<=$Field[2]+0.1*$Field[3];$bias+=$Field[3]) {
                    push(@main::BiasSweepList,$bias);
                }
            } else {
                $Field[3]=-1.0*abs($Field[3]);
                for ($bias=$Field[1];$bias>=$Field[2]+0.1*$Field[3];$bias+=$Field[3]) {
                    push(@main::BiasSweepList,$bias);
                }
            }
            $main::biasSweepSpec=join(" ",@Field[1..3]);
            $main::BiasFor{$main::biasSweepPin}=$Field[1];
            next;
        }
        if (s/^(biasList|listBias)\s+//i) {
            if (defined($main::biasListSpec)) {
                die("ERROR: can only have one biasList specification, stopped");
            }
            s/V\s*\(\s*//i;s/\s*\)//;
            @Field=split(/[=,\s]+/,$_);
            if ($#Field < 2) {
                die("ERROR: biasList specification must be V(pin)=val1,val2,..., stopped");
            }
            $main::biasListPin=$Field[0];
            for ($i=1;$i<=$#Field;++$i) {
                if ($Field[$i] !~ /^$main::number$/) {
                    die("ERROR: biasList values must be numbers, stopped");
                }
            }
            $main::biasListSpec=join(" ",@Field[1..$#Field]);
            $main::BiasFor{$main::biasListPin}=$Field[1];
            next;
        }
        if (s/^verilogAfile\s+//i) {
            $main::verilogaFile=$_;
            if (! -f $main::verilogaFile) {
                die("ERROR: cannot find file $main::verilogaFile, stopped");
            }
            next;
        }
        if (s/^(pins|terminals)\s+//i) {
            foreach $pin (@main::Pin) {
                $main::isPin{$pin}=0;
            }
            @main::Pin = split(/[\s,]+/,$_);
            foreach $pin (@main::Pin) {
                $main::isPin{$pin}=1;
                if ($pin !~ /^[a-zA-Z][a-zA-Z0-9]*$/) { # underscores are not allowed
                    die("ERROR: bad pin name specification $pin, stopped");
                }
            }
            next;
        }
        if (s/^instanceParameters\s+//i) {
            foreach $arg (split(/\s+/,$_)) {
                if ($arg !~ /.=./) {
                    die("ERROR: instance parameters must be name=value pairs, stopped");
                }
                ($name,$value)=split(/=/,$arg);
                $value=~s/\(//;$value=~s/\)//; # get rid of possible parens
                if ($value !~ /^$main::number$/) {
                    die("ERROR: instance parameter value in $arg is not a number, stopped");
                }
                push(@main::InstanceParameters,$arg);
            }
            next;
        }
        if (s/^modelParameters\s+//i) {
            foreach $arg (split(/\s+/,$_)) {
                $arg_file = $main::srcdir . $arg;
                if ($arg !~ /.=./ && ! -r $arg_file) {
                    die("ERROR: model parameters must be name=value pairs or a file name, stopped");
                }
                if (-r $arg_file) {
                    if (!open(IF,"$arg_file")) {
                        die("ERROR: cannot open file $arg_file, stopped");
                    }
                    while (<IF>) {
                        chomp;s/\s*=\s*/=/g;
                        s/^\+\s*//;s/^\s+//;s/\s+$//;
                        ($name,$value)=split(/=/,$_);
                        $value=~s/\(//;$value=~s/\)//; # get rid of possible parens
                        if ($value !~ /^$main::number$/) {
                            die("ERROR: model parameter value in $_ is not a number, stopped");
                        }
                        if (!defined($AlreadyHave{$name})) {
                            push(@main::ModelParameters,"$name=$value");
                            $AlreadyHave{$name}=1;
                            $IndexFor{$name}=$#main::ModelParameters;
                        } else {
                            if ($AlreadyHave{$name} == 1 && $main::printWarnings) {
                                printf("WARNING: parameter $name defined more than once, last value specified will be used\n");
                            }
                            $AlreadyHave{$name}=2;
                            $main::ModelParameters[$IndexFor{$name}]="$name=$value";
                        }
                    }
                    close(IF);
                } else {
                    ($name,$value)=split(/=/,$arg);
                    $value=~s/\(//;$value=~s/\)//; # get rid of possible parens
                    if ($value !~ /^$main::number$/) {
                        die("ERROR: model parameter value in $arg is not a number, stopped");
                    }
                    if (!defined($AlreadyHave{$name})) {
                        push(@main::ModelParameters,"$name=$value");
                        $AlreadyHave{$name}=1;
                        $IndexFor{$name}=$#main::ModelParameters;
                    } else {
                        if ($AlreadyHave{$name} == 1 && $main::printWarnings) {
                            printf("WARNING: parameter $name defined more than once, last value specified will be used\n");
                        }
                        $AlreadyHave{$name}=2;
                        $main::ModelParameters[$IndexFor{$name}]="$name=$value";
                    }
                }
            }
            next;
        }
        if (s/^freq[uency]*\s+//i) {
            if (!/^(lin|oct|dec)\s+(\d+)\s+($main::number)\s+($main::number)$/) {
                die("ERROR: bad frequency sweep specification, stopped");
            }
            $main::fType=$1;$main::fSteps=$2;$main::fMin=$3;$main::fMax=$4;
            $main::frequencySpec=$_;
            next;
        }
        if (s/^temperature\s+//i) {
            push(@main::Temperature,split(/[,\s]+/,$_));
            foreach $temperature (@main::Temperature) {
                if ($temperature !~ /^$main::number$/) {
                    die("ERROR: bad temperature value specified, stopped");
                }
            }
            next;
        }
        if (s/^float[ingpinode]*\s+//i) {
            @Field=split(/[\s,]+/,$_);
            if ($#Field < 0) {
                die("ERROR: bad floating pin specification, stopped");
            }
            foreach (@Field) {
                if (!$main::isPin{$_}) {
                    die("ERROR: floating pin $_ is not a specified device pin, stopped");
                }
                $main::isFloatingPin{$_}=1;
            }
            next;
        }
        if (/^symmetric(Pins|Terminals)/i) {
            @Field=split(/[\s,]+/,$_);
            if ($#Field == 2) {
                die("ERROR: cannot add symmetric pin specification per test, stopped");
            } else {
                @main::TestVariants=();
                foreach $variant (@main::Variants) {
                    if ($variant ne "Flip_N" && $variant ne "Flip_P") {
                        push(@main::TestVariants,$variant);
                    }
                }
            }
            next;
        }
        die("ERROR: unknown test directive\n$_\nstopped");
    }
    unless (@main::Temperature) {
        @main::Temperature=@main::DefaultTemperature;
    }
    if (abs($main::outputDc+$main::outputAc+($main::outputNoise>0)-1) > 0.001) {
        die("ERROR: outputs specified must be one of DC, AC or noise, stopped");
    }
    if ($main::outputDc && !defined($main::biasSweepSpec)) {
        die("ERROR: no bias sweep spec defined for DC testing, stopped");
    }
    if ($main::outputNoise && !defined($main::frequencySpec)) { # default for noise is f=1
        $main::frequencySpec="lin 1 1 1";
        $main::fType="lin";$main::fSteps=1;$main::fMin=1;$main::fMax=1;
    }
    if ($main::outputAc && !defined($main::frequencySpec)) { # default for AC is omega=1
        $oneOverTwoPi=1.0/(8.0*atan2(1.0,1.0));
        $main::frequencySpec="lin 1 $oneOverTwoPi $oneOverTwoPi";
        $main::fType="lin";$main::fSteps=1;$main::fMin=$oneOverTwoPi;$main::fMax=$oneOverTwoPi;
    }
    if ($main::outputAc && ($main::frequencySpec eq "lin") && ($main::simulatorName =~ /hspice|ads/i)) {
        # AC spec is number of points, not number of steps, for hspice and ads
        ++$main::fSteps if ($main::fMin != $main::fMax);
        $main::frequencySpec="$main::fType $main::fSteps $main::fMin $main::fMax";
    }
    foreach $pin (@main::Pin) {
        if (!defined($main::BiasFor{$pin}) && !defined($main::isFloatingPin{$pin})) {
            die("ERROR: a bias must be specified for all non-floating pins, stopped");
        }
        if ($main::isSymmetryPin{$pin} && $main::isFloatingPin{$pin}) {
            die("ERROR: a floating pin cannot be specified as a symmetry pin, stopped");
        }
        if ($isAnalysisPin{$pin} && $main::isFloatingPin{$pin} && !main::outputNoise) {
            die("ERROR: a floating pin can only have its voltage measured in DC analyses, stopped");
        }
    }
    if (!defined($main::biasListPin)) { # if not specified make a dummy bias list, to simplify processing later
        $main::biasListPin="dummyPinNameThatIsNeverUsed";
        $main::biasListSpec="0";
    } elsif (defined($main::isFloatingPin{$main::biasListPin})) {
        die("ERROR: a bias list cannot be specified for a floating pin, stopped");
    }
    if (!defined($main::biasSweepPin)) { # if not specified make a dummy bias sweep, to simplify processing later
        if($main::biasListPin eq $main::Pin[0]) {
            $main::biasSweepPin=$main::Pin[1];
        } else {
            $main::biasSweepPin=$main::Pin[0];
        }
        if (!defined($main::BiasFor{$main::biasSweepPin})) {
            $main::biasSweepSpec="0 0 0";
            @main::BiasSweepList="vin";
        } else {
            $main::biasSweepSpec="$main::BiasFor{$main::biasSweepPin} $main::BiasFor{$main::biasSweepPin} 0";
            @main::BiasSweepList=($main::BiasFor{$main::biasSweepPin});
        }
    } elsif (defined($main::isFloatingPin{$main::biasSweepPin})) {
        die("ERROR: a bias sweep cannot be specified for a floating pin, stopped");
    }
}

#
#   This subroutine reads in a test specification file.
#   It cleans up the syntax by getting rid of comments
#   and continutation lines, processing conditionals,
#   and splitting up the contents of the file into
#   global specifications and individual test specifications.
#
#   On call:
#       $main::qaSpecFile must be set to the name of the file that
#                         contains the qaSpec information
#   On return:
#       @main::Setup      contains general, test-nonspecific information
#       @main::Test       contains a list of the test defined in the qaSpec file
#       %main::TestSpec   contains each test specification, hash keys are @main::Test elements
#

sub readQaSpecFile {
    my(@File,@RawFile,@Field);

    @RawFile=&readHierarchicalFile($main::qaSpecFile);
    foreach (@RawFile) {
        s%\s*//.*%%;                              # eliminate C++ style comments
        s/^\s+//;s/\s+$//;                        # eliminate leading and trailing white space
        next if (/^$/);                           # ignore blank lines
        s/\s*=\s*/=/g;                            # eliminate space around "=" in name=value pairs
        if (/^\+/) {                              # process a continuation line
            s/^\+\s*//;                           # eliminate continuation "+" and any following whitespace
            $File[$#File]=~s/\s*\\$//;            # get rid of possible additional continuation on previous line
            $File[$#File].=" $_";                 # add to previous line
            next;
        }
        if (($#File >= 0) && ($File[$#File] =~ /\\$/)) {  # add to previous line if that had an end-of-line continuation
            $File[$#File]=~s/\s*\\$//;
            $File[$#File].=" $_";
        } else {
            push(@File,$_);
        }
    }
    @File=&processIfdefs(\%main::Defined,@File);  # process ifdef's to get conditional-free qaSpec
    @main::Test=();@main::Setup=();
    foreach (@File) {                             # process the qaSpec
        if (/^test(name)?\s+/i) {
            @Field=split;
            if ($#Field < 1) {
                die("ERROR: no test name specified for a test directive, stopped");
            }
            push(@main::Test,$Field[1]);
            @{$main::TestSpec{$main::Test[$#main::Test]}}=();
            next;
        }
        if ($#main::Test >= 0) {
            push(@{$main::TestSpec{$main::Test[$#main::Test]}},$_);
        } else {
            push(@main::Setup,$_);
        }
    }
}

sub readHierarchicalFile {
    my($fileName,$hierarchyLevel)=@_;
    my(@File,$FH,$includeFileName);

    if (!defined($hierarchyLevel)) {
        $hierarchyLevel=0;
    }
    $FH="file".$hierarchyLevel;
    if (!open($FH,$fileName)) {
        die("ERROR: cannot open file $fileName, stopped");
    }
    @File=();
    while (<$FH>) {
        chomp;
        if (/^\s*\`include\s+/) {
            ($includeFileName=$')=~s/"//g;
            ++$hierarchyLevel;
            push(@File,&readHierarchicalFile($includeFileName,$hierarchyLevel));
            --$hierarchyLevel;
        } else {
            push(@File,$_);
        }
    }
    close($FH);
    return(@File);
}

#
#   This subroutine processes the `ifdef statements (recursively, so nested `ifdef's are handled)
#   and returns the test specification with the appropriate blocks included and excluded.
#   Note that the simulator name is defined, so that
#   simulator specific directives are automatically included.
#

sub processIfdefs {
    my($defRef,@Input)=@_;
    my(%Defined,$i,$block,@Field,$start,$middle,$end,$ifdefLevel,$maxIfdefLevel);
    my(@Insert);

    %Defined=%$defRef;
    for ($i=0;$i<=$#Input;++$i) {
        if ($Input[$i] =~ /^`(define|undef)/) {
            @Field=split(/\s+/,$Input[$i]);
            if ($#Field > 0) {
                if ($Input[$i] =~ /^`define/) {
                    $Defined{$Field[1]}=1;
                } else {
                    $Defined{$Field[1]}=0;
                }
            }
            splice(@Input,$i,1);
            --$i;
            next;
        }
        if ($Input[$i] =~ /^`ifdef\s+/) {
            $start=$i;
            $ifdefLevel=1;$maxIfdefLevel=1;
            undef($middle);
            for ($end=$start+1;$end<=$#Input;++$end) {
                if ($Input[$end] =~ /^`ifdef/) {
                    ++$ifdefLevel;
                    if ($ifdefLevel > $maxIfdefLevel) {$maxIfdefLevel=$ifdefLevel}
                }
                if ($Input[$end] =~ /^`end/)  {--$ifdefLevel}
                if ($Input[$end] =~ /^`else/ && $ifdefLevel == 1) {$middle=$end}
                last if ($ifdefLevel == 0);
            }
            if (($end > $#Input) && ($ifdefLevel > 0)) {
                die("ERROR: `ifdef not terminated, stopped");
            }
            ($block=$Input[$i])=~s/^`ifdef\s+//;
            if ($maxIfdefLevel > 1) {
                if (!defined($middle)) {$middle=$end}
                @Insert=();
                if ($Defined{$block}) {
                    if ($start+1 <= $middle-1) {
                        @Insert=&processIfdefs(\%Defined,@Input[$start+1..$middle-1]);
                    }
                } else {
                    if ($middle+1 <= $end-1) {
                        @Insert=&processIfdefs(\%Defined,@Input[$middle+1..$end-1]);
                    }
                }
                splice(@Input,$start,$end-$start+1,@Insert);
            } else {
                if (!defined($middle)) {$middle=$end}
                @Insert=();
                if ($Defined{$block}) {
                    if ($start+1 <= $middle-1) {
                        @Insert=@Input[$start+1..$middle-1]
                    }
                } else {
                    if ($middle+1 <= $end-1) {
                        @Insert=@Input[$middle+1..$end-1];
                    }
                }
                splice(@Input,$start,$end-$start+1,@Insert);
            }
            --$i;
            next;
        }
        if ($Input[$i] =~ /^`/) {die("ERROR: bad directive\n$Input[$i]\nstopped")}
    }
    return(@Input);
}

sub unScale {

#
#   call: $Result=&unScale($Scalar);
#
#   If $Scalar is a SPICE-like scaled number then $Result is the value
#   of that number, else $Result is just $Scalar.
#
    my($String)=@_;
    my($Result);

    $Result=$String;
    if ($String =~ /^([+-]?[0-9]+[.]?[0-9]*|[+-]?[.][0-9]+)T/i) {
        $Result=$1*1e12;
    } elsif ($String =~ /^([+-]?[0-9]+[.]?[0-9]*|[+-]?[.][0-9]+)G/i) {
        $Result=$1*1e9;
    } elsif ($String =~ /^([+-]?[0-9]+[.]?[0-9]*|[+-]?[.][0-9]+)(M|meg|x)/) {
        $Result=$1*1e6;
    } elsif ($String =~ /^([+-]?[0-9]+[.]?[0-9]*|[+-]?[.][0-9]+)K/i) {
        $Result=$1*1e3;
    } elsif ($String =~ /^([+-]?[0-9]+[.]?[0-9]*|[+-]?[.][0-9]+)m/) {
        $Result=$1*1e-3;
    } if ($String =~ /^([+-]?[0-9]+[.]?[0-9]*|[+-]?[.][0-9]+)u/) {
        $Result=$1*1e-6;
    } elsif ($String =~ /^([+-]?[0-9]+[.]?[0-9]*|[+-]?[.][0-9]+)n/) {
        $Result=$1*1e-9;
    } elsif ($String =~ /^([+-]?[0-9]+[.]?[0-9]*|[+-]?[.][0-9]+)p/) {
        $Result=$1*1e-12;
    } elsif ($String =~ /^([+-]?[0-9]+[.]?[0-9]*|[+-]?[.][0-9]+)f/) {
        $Result=$1*1e-15;
    } elsif ($String =~ /^([+-]?[0-9]+[.]?[0-9]*|[+-]?[.][0-9]+)a/) {
        $Result=$1*1e-18;
    }
    return($Result);
}

sub platform {

#
#   This subroutines returns a string that includes the processor
#   type, OS name, and OS version. This string is used as one level
#   of the directory hierarchy for storing test results, because
#   simulation results can vary with processor and OS.
#
#   The UNIX uname command is used to get the appropriate information.
#   If the system appears to be Windows, then the perl Config module
#   information is used instead. However this information is generated
#   as part of the Perl build, and so may not relate to the machine
#   on which it is being run.
#

    use Config;
    my($osName,$osVer,$archName)=($modelQa::Config{osname},$modelQa::Config{osvers},$modelQa::Config{archname});
    my($platform);

    if ($osName !~ /win/i) {
        open(UNAME,"uname -p|") or die("ERROR: cannot determine processore and OS information, stopped");
        chomp($archName=<UNAME>);close(UNAME);
        if ($archName eq "unknown") {
            open(UNAME,"uname -m|");chomp($archName=<UNAME>);close(UNAME);
        }
        open(UNAME,"uname -s|");chomp($osName=<UNAME>);close(UNAME);
        open(UNAME,"uname -r|");chomp($osVer =<UNAME>);close(UNAME);
    }
    $platform = "${archName}_${osName}_${osVer}";
    $platform =~ s/\s+/-/g;
    $platform =~ s|[()/]||g;
    return($platform);
}

1;
