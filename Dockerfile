FROM tensorflow/tensorflow:latest-gpu-py3
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install Python packages
RUN pip --no-cache-dir install --upgrade \
        keras \
		pygments \
		whatthepatch

# Open Ports for TensorBoard, Jupyter, SSH and Jupyter
EXPOSE 6006
EXPOSE 7654
EXPOSE 22
EXPOSE 8888

#Setup File System
RUN mkdir ds
ENV HOME=/ds
ENV SHELL=/bin/bash
VOLUME /ds
WORKDIR /ds

# Run the shell
CMD [ "/bin/bash"  ]
