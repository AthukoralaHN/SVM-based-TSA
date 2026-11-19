"""Transient-stability classification of the IEEE 39-bus system with an SVM.

Modules
  config    file locations, feature names, split sizes, tuning grid
  data      load, describe and split the dataset (train / validation / test)
  model     SVM pipeline, tuning on the validation set, save / load
  evaluate  metrics, breakdowns, feature importance, learning curve
  plots     all charts
  predict   predictions for new cases
"""
