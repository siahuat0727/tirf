# Tool for analysis TIRF (Total Internal Reflection Fluorescence)

### To be continued...

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
$ python3 main.py --task play --input path/to/input --input_type video --show --fps 10 --reverse
```

### Generate data

```
$ python3 main.py --task gen --input path/to/input --input_type video --fps 10 --pkl info.pkl
```
