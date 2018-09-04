from create_dataset import createInputCSV

from sequential_neural_network import (
    get_model, 
    run_model, 
    test_model
)

print('-------------------')
print('UBitName = ameenmoh')
print('personNumber = 50288968')
print('-------------------')


TRAINING_DATA = "./data/training.csv"
TESTING_DATA = "./data/testing.csv"

# Create dataset for training and testing the model
createInputCSV(101, 1001, TRAINING_DATA)
createInputCSV(1, 101, TESTING_DATA)

# Initialize and run the model
model = get_model()
history = run_model(model)

# Training and Validation Graph
# %matplotlib inline
# df = pd.DataFrame(history.history)
# df.plot(subplots=True, grid=True, figsize=(10,15))

# Test Accuracy
test_model(model)
