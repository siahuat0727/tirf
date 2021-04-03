# Tool for analysis TIRF (Total Internal Reflection Fluorescence)

## Installation

Optional

```
$ python3 -m venv env
$ source env/bin/activate
```

Install requirements

```
$ python3 -m pip install -r requirements.txt
```

## Execute examples

### Play

```
$ python3 main.py --task play --x 50 150 --y 50 200 --video xx.avi
```

### Generate data

```
$ python3 main.py --task gen --video xx.avi
```
