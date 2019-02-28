# Linear MPC
- linear model-predictive control using ceres, controling 3 omni-wheels robot

[![youtube_video](http://img.youtube.com/vi/7nh470VFEvE/0.jpg)](http://www.youtube.com/watch?v=7nh470VFEvE "linear_mpc_wo_obstacles")

[![youtube_video](http://img.youtube.com/vi/31WVlmaRspQ/0.jpg)](http://www.youtube.com/watch?v=31WVlmaRspQ "linear_mpc_w_obstacles")

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

