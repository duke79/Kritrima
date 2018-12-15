import os

output_dir = os.path.abspath("output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)