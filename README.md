# CombinatoricDegeneracies
a playground for defeating MCMC and evidence integrations

## Authors
- David W. Hogg (SCDA)
- Daniel Foreman-Mackey (UW)

## Projects
- Construct a family of trivial inference problems in which there are combinatoric degeneracies (labeling degeneracies or non-identifiability), but in which the Bayesian evidence integral can be computed analytically.
- Show that most methods that claim to measure evidence integrals are not succeeding, or delivering answers that depart from the true value by more than any reported uncertainty.
- Compare methods for sampling or combining models that are of different complexity but that "nest" naturally (for example, a 5-planet model for some astronomical data will nest inside a 6-planet model for those same data).  Some methods sample each different complexity separately and combine the separate models *ex post facto* using the evidence integral values; other methods sample all complexities simultaneously.
