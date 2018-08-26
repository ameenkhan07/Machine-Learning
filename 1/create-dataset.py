# Script : Creates Training Data Set

from FizzBuzz import FizzBuzz
import json

training_data_file = 'TrainingData.json'

dict_ = {}

for ele in range(101, 1001):
    dict_[ele]=FizzBuzz().func(ele)
    
outfile = open(training_data_file, 'w')
json.dump(dict_, outfile)
