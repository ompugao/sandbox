# Linear MPC
- linear model-predictive control using ceres, controling 3 omni-wheels robot

[![Linear Model-Predictive Control](http://img.youtube.com/vi/sqt6TSKI2EQ/0.jpg)](http://www.youtube.com/watch?v=sqt6TSKI2EQ "Linear Model-Predictive Control")

# dependencies
- [ceres-solver](https://ceres-solver.googlesource.com/ceres-solver)
- matplotlib
- picojson

# Usage
```sh
$ # install python and matplotlib, and ceres. and then,
$ make
$ ./build/main --parameters_path ./params.json --logtostderr=1
```

