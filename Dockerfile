FROM ubuntu
RUN apt-get update
RUN apt-get install bzip2
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh tmp/Miniconda3-latest-Linux-x86_64.sh
RUN bash tmp/Miniconda3-latest-Linux-x86_64.sh -b
ENV PATH $PATH:/root/miniconda3/bin/
ADD https://raw.githubusercontent.com/udacity/CarND-TensorFlow-L2/master/environment.yml tmp/environment.yml
RUN conda env create -f tmp/environment.yml
RUN source activate CarND-TensorFlow-L2
RUN conda install -c conda-forge tensorflow
EXPOSE 8888
