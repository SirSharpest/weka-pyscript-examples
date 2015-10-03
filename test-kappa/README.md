```

.PyScriptClassifier -script train.py -args "a=1;b=0;logistic=True;num_hidden_units=10;alpha=0.01;lambda=0;epochs=500;rmsprop=True" -impute -binarize -standardize

.PyScriptClassifier -script train.py -args "expectation=True;a=1;b=0;logistic=True;num_hidden_units=10;alpha=0.01;lambda=0;epochs=500;rmsprop=True" -impute -binarize -standardize

.PyScriptClassifier -script train.py -args "expectation=True;a=1;b=0.5;logistic=True;num_hidden_units=10;alpha=0.01;lambda=0;epochs=500;rmsprop=True" -impute -binarize -standardize

.PyScriptClassifier -script train.py -args "expectation=True;a=1;b=1;logistic=True;num_hidden_units=10;alpha=0.01;lambda=0;epochs=500;rmsprop=True" -impute -binarize -standardize

.PyScriptClassifier -script train.py -args "expectation=True;a=1;b=1.5;logistic=True;num_hidden_units=10;alpha=0.01;lambda=0;epochs=500;rmsprop=True" -impute -binarize -standardize

.PyScriptClassifier -script train.py -args "expectation=True;a=1;b=2.0;logistic=True;num_hidden_units=10;alpha=0.01;lambda=0;epochs=500;rmsprop=True" -impute -binarize -standardize

.PyScriptClassifier -script train.py -args "expectation=True;a=0;b=1;logistic=True;num_hidden_units=10;alpha=0.01;lambda=0;epochs=500;rmsprop=True" -impute -binarize -standardize

# ??
.PyScriptClassifier -script train-regression.py -args "alpha=0.01;lambda=0;epochs=500;rmsprop=True" -impute -binarize -standardize

.PyScriptClassifier -script train-regression.py -args "adaptive=True;alpha=0.01;lambda=0;epochs=500;rmsprop=True" -impute -binarize -standardize

```
