#!/usr/bin/env bash

#
# run benchmarks -
#

CMDLINE="$0 $@"
SCRIPT=$(basename "$0")
SRCDIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OnError() {
    echo "$SCRIPT: Error on line ${BASH_LINENO[0]}, exiting."
    exit 1
}
trap OnError ERR

HOSTTAG=$(hostname | cut -d . -f 1)
VERBOSE=1
GOLDBASE="$SRCDIR/gold-${HOSTTAG}"
BENCH_LIST=
BENCH_ARGS=
MAKE_ARGS=
TAG=

usage () {
    if [ -t 3 ]; then echo "$SCRIPT failed, see log" >&3 ; fi
    cat <<EOF
usage:
 ${SCRIPT} <benchmarks> [options...]

where <benchmarks> are [bench_eigen|bench_dual]

where [options...] are:
      -g|--makegold           - save ouput to the 'gold' file standard location
      -T|--tag <tag>          - add tag to results file [$TAG]
      -t|--test <test>        - filter benchmark tests against <test> [*]
      -o|--output <base>      - output file name base [$OUTFILE]
      -f|--fast               - run fast, at the great expense of accurate measurement
      -m|--medium             - run medium fast, at the medium expense of accurate measurement
      -v|--verbose            - be more verbose (can add multiple times)
      -q|--quiet              - be more quiet   (can add multiple times)
      -h|--help               - print this usage message and exit

current bench1 args: ${BENCH_ARGS}
EOF
    exit 1
}

# process command line args
while [ $# -gt 0 ]; do
    key="$1"
    case $key in
        -g|--makegold)         OUTFILE="$GOLDBASE"   ;;
        -T|--tag)              TAG="-$2" ;    shift   ;;
        -o|--output)           OUTFILE="$2"; shift   ;;
        -t|--test)
            BENCH_ARGS="${BENCH_ARGS} --benchmark_filter=$2"
            shift
            ;;
        -f|--fast)
            BENCH_ARGS="${BENCH_ARGS} --benchmark_min_time=0.001"
            ;;
        -m|--medium)
            BENCH_ARGS="${BENCH_ARGS} --benchmark_min_time=0.05"
            ;;
        -h|help)               usage                 ;;
        -v|--verbose)
            VERBOSE=$(( VERBOSE + 1 ))
            MAKE_ARGS="${MAKE_ARGS} VERBOSE=1"
            ;;
        -q|--quiet)            VERBOSE=$(( VERBOSE - 1 )) ;;
        *)
            if [ -f "${SRCDIR}/${key}.cpp" ]; then
                BENCH_LIST="${BENCH_LIST} ${key}"
                shift
                continue
            fi
            # unknown option
            echo "unknown option $key"
            usage
            ;;
    esac
    shift
done

# default
if [ -z "${BENCH_LIST}" ]; then
    BENCH_LIST=bench_dual
fi

BENCH_ARGS="${BENCH_ARGS} --benchmark_out_format=json"
ret=0
for NAME in ${BENCH_LIST}; do
    GOLDFILE=${GOLDBASE}.${NAME}${TAG}.json
    TMPFILE=$(mktemp ./bench.XXXXXXXXX)

    # if there's no gold reference yet, by default create one
    if [ -z "$OUTFILE" ]; then
        if [ ! -f "$GOLDFILE" ]; then
            echo "No gold file found, creating one..."
            REALOUT=$GOLDFILE
        else
            REALOUT=bo-${NAME}${TAG}.json
        fi
    else
        REALOUT=${OUTFILE}.${NAME}${TAG}.json
    fi


    BENCH_ARGS="${BENCH_ARGS} --benchmark_out=${TMPFILE}"
    #BENCH_ARGS="${BENCH_ARGS} --benchmark_counters_tabular=true"
    #BENCH_ARGS="${BENCH_ARGS} --benchmark_repetitions=1000"

    echo "GOLDFILE=$GOLDFILE"
    echo "TMPFILE =$TMPFILE"
    echo "OUTFILE =$OUTFILE"
    echo "REALOUT =$REALOUT"
    echo "Making ${NAME}${TAG} $MAKE_ARGS"
    make -j4 "${NAME}" $MAKE_ARGS

    echo "Running benchmarks: ./tests/${NAME} $BENCH_ARGS"
    #echo " : output in ${NAME}.out"
    if ! ./tests/${NAME} $BENCH_ARGS ; then
        echo "failed $NAME"
        ret=1
        continue
    fi

    if [ "${REALOUT}" != "$TMPFILE" ]; then
        mv "$TMPFILE" "${REALOUT}"
    fi

    if [ "${REALOUT}" != "$GOLDFILE" ]; then
        echo "comparing with $GOLDFILE"
        ./thirdparty/benchmarkX/src/benchmarkX/tools/compare.py benchmarks "$GOLDFILE" "${REALOUT}"
    fi

    #./thirdparty/benchmarkX/src/benchmarkX/tools/compare.py filters ./bench_eigen B_Pade B_Expm

    echo "Benchmark ouput in '${REALOUT}'"
done

exit $ret
