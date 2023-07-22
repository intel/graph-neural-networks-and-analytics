FROM ubuntu:22.04

#Install necessary packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    openssh-server \
    expect \
    net-tools \
    python3 \
    python3-pip && \
    apt-get install -y --no-install-recommends --fix-missing numactl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf $(which python3) /usr/local/bin/python && \
    ln -sf $(which python3) /usr/local/bin/python3 && \
    ln -sf $(which python3) /usr/bin/python

##Install packages
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt && \
rm -rf requirements.txt

SHELL ["/bin/bash", "-c"]
RUN sed -i 's/#Port 22/Port 12347/g' /etc/ssh/sshd_config
RUN sed -i 's/#   Port 22/    Port 12347/g' /etc/ssh/ssh_config
RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config
RUN echo "root:docker" | chpasswd

CMD [ "sh", "-c", "service ssh start; bash"]
