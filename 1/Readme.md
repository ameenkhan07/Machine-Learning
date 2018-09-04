
### Setup

```{sh}
# Setup virtualenv
virtualenv --system-site-packages -p python3 env
source venv/bin/activate

# Install all libraries
pip3 install jupyter --user
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