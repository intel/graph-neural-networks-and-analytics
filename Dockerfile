FROM ubuntu:20.04

#Install necessary packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
	 wget \
	 net-tools \
	 python3 \
	 python3-pip && \
	 apt-get clean && \
	 rm -rf /var/lib/apt/lists/*

RUN ln -sf $(which python3) /usr/local/bin/python && \
    ln -sf $(which python3) /usr/local/bin/python3 && \
    ln -sf $(which python3) /usr/bin/python

#Install packages
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu \
    torchmetrics tqdm cmake numpy pyyaml \
	scikit-learn ogb chardet && \
    python3 -m pip install intel_extension_for_pytorch -f https://software.intel.com/ipex-whl-stable

RUN pip install --pre dgl -f https://data.dgl.ai/wheels/repo.html && \
    pip install --pre dglgo -f https://data.dgl.ai/wheels-test/repo.html

RUN pip install --no-cache-dir certifi==2022.12.7
RUN pip install --no-cache-dir urllib3==1.26.5
RUN pip install --no-cache-dir setuptools==65.5.1

RUN apt-get update && \
     apt-get install -y --no-install-recommends --fix-missing numactl && \
     apt-get clean && \
     rm -rf /var/lib/apt/lists/*
