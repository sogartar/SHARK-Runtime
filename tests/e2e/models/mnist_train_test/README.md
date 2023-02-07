Test that IREE has the correct model and optimizer state
after doing one train step and after initialization of parameters.
The ground truth is extracted from a JAX model.
The MLIR model is generated with IREE JAX.

To regenerate the model together with the test data use
```
python ./generate_test_data.py
```
