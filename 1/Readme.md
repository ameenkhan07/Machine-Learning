
### Setup

```{sh}
# Setup venv
python3 -m venv virtualenv
source virtualenv/bin/activate

# Install all libraries
pip install jupyter --user
pip install jupyterthemes
jt -t grade3 -fs 95 -altp -tfs 11 -nfs 115 -cellw 88% -T

pip3 install --upgrade tensorflow
pip3 install keras
pip3 install matplotlib
pip3 install pandas
```

### Files

- data (Directory for storing data being generated)
    - testing.csv
    - training.csv
    - output.csv
- main.py
- FizzBuzz.py
- create_dataset.py
- process_dataset.py
- sequential_neural_network.py

- Fizzing and Buzzing.ipynb