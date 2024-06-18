# Pollinator NSGA2
## NSGA 2 customization for createing a Pareto front with objectives "population of bumblebees" and "park livability"

### Requirements
0. We assume Anaconda is installed. One can install it according to its [installation page](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
1. Clone this repo:
```
git clone https://github.com/chiara-camilla-rambaldi-migliore/pollinator-nsga2.git
cd pollinator-nsga2
```
2. Create a virtual environment `pollinator_nsga2`. 
```
conda create --name pollinator_nsga2
conda activate pollinator_nsga2
pip install -r requirements.txt
```

## Run
```
python3 main.py -a [gray|gray_doe|random|random_doe] -c ["max cpu cores (number)"] -g ["generations (number)"] -n ["individuals (number)" -s ["seed (number)"]
```
