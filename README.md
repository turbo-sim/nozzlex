# Nozzle-hyperbolic


## Cloning git repository and installing dependencies

Commands to clone repository, create Conda environment, and install all required packages:

```bash
git clone https://github.com/AndCiof/nozzlex.git
cd nozzlex
conda env create -f environment.yaml
conda activate nozzlex_env
poetry install
pip install barotropy # There are some conflicting package, so far it is better to do like this
```


## Pushing to git repository

Commands to add changes and push to the remote repository

```bash
git add <filename>
git commit -m <commit message>
git push origin
```