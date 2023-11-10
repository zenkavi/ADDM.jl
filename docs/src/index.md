# Introduction

**[Add build status badges]**

Welcome to [ADDM.jl](https://github.com/aDDM-Toolbox/ADDM.jl), a package for 
joint modeling of response times, eyetracking data and choice behavior using
evidence accummulations models with time-varying drift rates. 

## Installation

### Currently

#### Docker

**If you don't want to deal with installing any dependencies**

- Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Start Docker Desktop.
- Pull Docker image

From Terminal: 
```
docker pull zenkavi/addm.jl:0.0.1
```
- Run a container using this image. This will start in a Julia kernel with all environment dependencies installed.

```
docker run --rm -it zenkavi/addm.jl:0.0.1
```
- If you prefer an image with jupyter notebooks

```
docker pull zenkavi/addm.jl.nb:0.0.1
```

From Docker Desktop (not recommended):
  - `Cmd+K` to search hub images
  - Type `zenkav/addm` and it should list images. Select either `zenkavi/addm.jl` or `zenkavi/addm.jl.nb` depending on whether you want to run a notebook
  - Click `Pull` to pull the image


- Start a notebook from a terminal mounting your current local directory onto the containers work directory

```
docker run -it --rm \
-v $(pwd):/home/jovyan/work \
-p 8888:8888 zenkavi/addm.jl.nb:0.0.1 jupyter-lab
```

#### Github

**If you have Julia and Git installed and want a local copy of the toolbox on your machine. This will require you to install all Julia dependencies**

- Clone the Github repo

```
git clone https://github.com/aDDM-Toolbox/ADDM.jl.git
```

or

```
git clone git@github.com:aDDM-Toolbox/ADDM.jl.git
```

- Navigate to the ADDM.jl directory

```
cd ADDM.jl
```

- Set up the Julia environment (might take a few minutes)

```
julia --project -e 'import Pkg; Pkg.instantiate()'
```

- Start up a Julia kernel using the project's environment

```
julia --project
```

### Once `ADDM.jl` is on the Julia Registry

To install `ADDM.jl` from the Julia Registry

```julia
julia> import Pkg

julia> Pkg.add("ADDM")
```

## License

`ADDM.jl` is licensed under the [GNU General Public License v3.0](https://github.com/aDDM-Toolbox/ADDM.jl/blob/main/LICENSE).

## Resources for getting started

There are a few ways to get started with ADDM.jl:

 * Become familiar with the modeling framework described in [Krajbich et al. (2010)](https://www.nature.com/articles/nn.2635)  
 * Become familiar with algorithm used for parameter estimation [Tavares et al. (2017)](https://www.frontiersin.org/articles/10.3389/fnins.2017.00468/full)  
 * Read the [introductory tutorial](https://addm-toolbox.github.io/ADDM.jl/dev/tutorials/getting_started/)

## Getting help

If you need help, please [open a GitHub issue](https://github.com/aDDM-Toolbox/ADDM.jl/issues/new).

## Citing `ADDM.jl`

If you use `ADDM.jl`, we ask that you please cite the following:

