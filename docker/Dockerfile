FROM julia:1.9.3

WORKDIR /home

# Utils
RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y python3-pip

# Install jupyterlab for notebook capabilities
RUN pip install --upgrade pip && \
    pip install --no-cache-dir jupyterlab

# Clone ADDM.jl from Github
RUN git clone https://github.com/aDDM-Toolbox/ADDM.jl.git

# Change directory into julia project
WORKDIR /home/ADDM.jl

# Instantiate environment
RUN julia --project -e 'import Pkg; Pkg.instantiate()'

# Change 
CMD ["julia", "--project"]
