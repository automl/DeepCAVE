# Instantiate your run
run = SMACRun.from_path(...)

# Check which inputs are needed inside the process method
inputs = dict(num_trees=dict(value=16))

# Also works for grouped run
data = fANOVA.process(run, inputs)