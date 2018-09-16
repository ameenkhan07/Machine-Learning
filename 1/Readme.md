
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

### Code Files


- main.py
- utils.py
- FizzBuzz.py
- create_dataset.py
- process_dataset.py
- sequential_neural_network.py

- Fizzing and Buzzing Keras.ipynb
- Fizzing and Buzzing Tensorflow.ipynb


- data (Directory for storing data being generated)
    - testing.csv
    - training.csv
- outputs
    - keras.png
    - output.csv


### Additional Notes

- Reproducible Results in Keras needed for every set of hyperparameter, details [here](https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-developmen) 