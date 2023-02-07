FROM nvidia/cudagl:11.1.1-devel-ubuntu18.04

ARG USER_NAME
ARG USER_PASSWORD
ARG USER_ID
ARG USER_GID

RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update
RUN apt install sudo
RUN useradd -ms /bin/bash $USER_NAME
RUN usermod -aG sudo $USER_NAME
RUN yes $USER_PASSWORD | passwd $USER_NAME

# set uid and gid to match those outside the container
RUN usermod -u $USER_ID $USER_NAME
RUN groupmod -g $USER_GID $USER_NAME

# work directory
WORKDIR /home/$USER_NAME

# install system dependencies
COPY ./scripts/install_deps.sh /tmp/install_deps.sh
RUN yes "Y" | /tmp/install_deps.sh

# setup python environment
RUN cd $WORKDIR

# install python requirements
# RUN sudo python3 -m pip install --upgrade pip && \ 
#     sudo python3 -m pip install --upgrade

# install pip3
RUN apt-get -y install python3-pip
RUN sudo python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade Pillow

# install pytorch

# install GLX-Gears (for debugging)
RUN apt-get update && apt-get install -y \
   mesa-utils \
   python3-setuptools \
   && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade setuptools wheel
RUN sudo apt-get install build-essential
RUN wget -q -O- https://data.kitware.com/api/v1/file/5e8b740d2660cbefba944189/download | tar zxf - -C ${HOME}
RUN export PATH=${HOME}/castxml/bin:${PATH}
RUN pip3 install \
    numpy\
    scipy\
    pybullet\
    imageio\

    transform3d
RUN sudo pip3 install nvisii 
RUN python3 -m pip install --upgrade Pillow
RUN pip3 install -vU https://github.com/CastXML/pygccxml/archive/develop.zip pyplusplus


# change ownership of everything to our user
RUN mkdir /home/$USER_NAME/tabletopGym
RUN cd /home/$USER_NAME/tabletopGym && echo $(pwd) && chown $USER_NAME:$USER_NAME -R .