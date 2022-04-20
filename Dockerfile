FROM python:3.10.4-bullseye

# python installation
RUN apt-get update && apt-get -y install bc bison flex libxaw7 libxaw7-dev libx11-6 libx11-dev libreadline8 libxmu6
RUN apt-get update && apt-get -y install build-essential libtool gperf libxml2 libxml2-dev libxml-libxml-perl libgd-perl
RUN apt-get update && apt-get -y install g++ gfortran make cmake libfl-dev libfftw3-dev 

RUN pip install pytest numpy pandas
