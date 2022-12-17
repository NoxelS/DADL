# How to use
- nn.py: Class for the neural network
- char_input_50_50.py: Deprecated tool to save characters as 50x50 image
  - Usage: ```char_input_50_50.py <path-to-save-to>```
- training.py: Used to train networks and find the one with the best accuracy. Change ```identifier_to_vector()``` to map a character to your output vector fitting your own character set.
- live_test.py: Test your network in a live preview
  - Usage: ```live_test.py <path-to-network-json-dump>```