[![on push](https://github.com/rakki-18/Epidemiology/actions/workflows/main.yml/badge.svg?branch=master)](https://github.com/rakki-18/Epidemiology/actions/workflows/main.yml)
# Epidemiology

## Setting Up
- When in the root directory inside your terminal, change directory to the subfolder `src/model` using `cd`
- Setup a virtual environment to install dependencies. Run the following commands to setup a virtualenv and install dependencies
### Windows
```cmd
python -m venv venv
venv/Scripts/activate
pip install -r ./requirements.txt
```
- Execute `python ./setup.py develop` to install the package
- Execute `pytest` to check if everything is working ok.

## Contribution guidelines
- Fork the project
- Make a new branch
- Make the changes in that particular branch
- Create a pull request
Check out this [link](https://cbea.ms/git-commit/) for commit message guidelines.

## References
- http://homepage.cs.uiowa.edu/~sriram/4980/spring20/notes/slides.1.21.pdf
- http://homepage.cs.uiowa.edu/~sriram/4980/spring20/notes/Jan28-30.pdf
- http://www.bio.utexas.edu/research/meyers/_docs/publications/DimitrovINFORMS10.pdf
