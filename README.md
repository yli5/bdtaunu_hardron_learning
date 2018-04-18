# Ideas to improve the measurement

The main bottleneck in getting a good measurement is the large systematic uncertainties on detector data.  Systematic uncertainty when evaluated on MC test set is large but reasonable, but when evaluated on the detector data, the systematic uncertainty increases by a factor of 3-5.

We can attribute this to the differences between MC (specifically mc.central) and real data, as had our simulation been perfect, we should be extracted the exact same numbers for both data sets (within statistical uncertainty).

## Idea 1: [Learning to pivot](https://arxiv.org/abs/1611.01046)

The basic idea is to learn a classifier that is robust against the changes of the nuisance parameters *z*.  In our case, the nuisance parameters would be the parameters used in the generation of MC samples, as described in DECAY.DEC.  

The hope is that the score we learn, which can be thought of as a mapping from R<sup>n</sup> to R<sup>1</sup>, provides a "summary" feature of the data that is invariant to the changes of the the assumed values of SM parameters.  This would in turn reduce our systematic uncertainty.  The caveat being that there is no such guarentee if *z*<sub>data</sub> is in the region far outside of the assumed uncertainties of *z*.  Nonetheless, we hope that this method would somewhat alleviate the large systematic uncertainties.

In practice, the "pivot" is learned by feeding the output of a classifier into an adversarial network, which tries to guess the instance of *z* that generated the particular data.  In effect, the adversarial network acts as a regularizer to the classifier; the trading off accuracy in favor of invariance to *z*.  This tradeoff can be controlled by the parameter &lambda; as described in the paper.

### Applying pivoting to our analysis

We can directly apply the above procedure to our analysis simply by introducing noise into our training samples.  The noise corresponds to the uncertainties on the physics parameters, which we assume to be Gaussian with its &mu; and &sigma; given by the world averages.  The change of the parameters manifests in our data as a change in its weight.  This approach makes sense given that the parameters control the cross section and decay rate of various interactions, which in turn affects the frequency of such interactions showing up in our sample.

The nuisance parameter *z* in this case consists of 10 continuous variables (model parameters) and 4 binary variables (model choice).  The re-weighing procedure would pass over our training data and for each point generate a *re-weight factor* according to the Gaussian distributions and binary choices.  The computational cost need not be optimized as this is a one-time effort over the training data.

After the training data has been re-weighed, there is still a matter of how to present the data in the form of mini-batches for performing the batch gradient descent.  

We would like to present our data to deep learning frameworks as if our data is *i.i.d.*.  The most straightforward (and true to the sprit of batch GD) would be to create each mini-batch by randomly sampling our training data according to the weight of each point.  

This is a very expensive operation (*n* choose 64 per mini-batch for batch size of 64; there are *n* / 64 batches per epoch).  In our benchmarks, this method is about 1000 times slower than the method of creating mini-batches by partitioning the data. 

There are two proposed method of bypassing the expensive sampling process:
1. For each epoch, generate a bootstrap sample of the entire training data.  Within an epoch, create mini-batches by partitioning.
2. Generate mini-batches by partitioning without considering weights. During training, for each mini-batch, generate a small number of bootstrap samples (cheap; 10 * (64 choose 64)), calculate the gridient of each bootstrap sample, and update the weights using the mean of the bootstrap gradients.

## Idea 2: [Unsupervised domain adaptation](http://sites.skoltech.ru/compvision/projects/grl/)
This method takes a different approach to building a more robust classifier.  Rather than having to assume the distributions of the nuisance variables *z*, we can incorporate unsupervised learning by using a sample of the detector data (which is generated using some true instance of the random variable *z*).

This method also introduces an adversarial network along with the base classifier for discriminating labels.  When presented with a data point, the adversarial network is train to discriminate *domain* of the point (*i.e.* does it come from the *labeled* source distribution or the *unlabeled* target distribution).  The input to both models are the output of a *feature learner*, which can be thought of as an autoencoder.  The job of the feature learner is to transform the input data as to minimize the loss of the label classifier while maximizing the loss of the domain classifier.

By incorporating the real data into our training process, we hope to learn a classifier that performs just as well on the unknown detector data; reducing the data-MC discrepancy.

## Measuring the performance of these models
TODO.
