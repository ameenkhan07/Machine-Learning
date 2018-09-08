import os

from create_dataset import createInputCSV
from utils import save_plot
from sequential_neural_network import (get_model, run_model, test_model)

print('-------------------')
print('UBitName = ameenmoh')
print('personNumber = 50288968')
print('-------------------')

# Create dataset for training and testing the model
createInputCSV(101, 1001, 'training.csv')
createInputCSV(1, 101, 'testing.csv')

# Initialize and run the model
model = get_model()
history = run_model(model)
save_plot(history.history, 'tf.png')

# Test Accuracy
test_model(model)
