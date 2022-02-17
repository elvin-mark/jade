package com.nn;

import com.data.Tensor;
import com.functions.F;

public class BatchNorm2d extends NNModule {
    int numFeatures;
    float momentum;

    public BatchNorm2d(int numFeatures, float momentum) {
        super();
        this.numFeatures = numFeatures;
        this.momentum = momentum;

        this.moduleName = "BatchNorm2d";

        Tensor runningMean = new Tensor(new int[] { 1, numFeatures, 1, 1 });
        Tensor runningVar = new Tensor(new int[] { 1, numFeatures, 1, 1 });
        Tensor gamma = new Tensor(new int[] { 1, numFeatures, 1, 1 });
        Tensor beta = new Tensor(new int[] { 1, numFeatures, 1, 1 });

        runningMean.zeros();
        runningVar.ones();
        gamma.ones();
        beta.zeros();

        runningMean.requires_grad(false);
        runningVar.requires_grad(false);
        gamma.requires_grad(true);
        beta.requires_grad(true);

        this.params.add(runningMean);
        this.params.add(runningVar);
        this.params.add(gamma);
        this.params.add(beta);
    }

    public BatchNorm2d(int numFeatures) {
        this(numFeatures, 0.1f);
    }

    public Tensor forward(Tensor input) {
        Tensor runningMean = this.params.get(0);
        Tensor runningVar = this.params.get(1);
        Tensor gamma = this.params.get(2);
        Tensor beta = this.params.get(3);

        if (this.training)
            return F.batchnorm2d(input, runningMean, runningVar, gamma, beta, momentum);
        else
            return F.batchnorm2d(input, runningMean, runningVar, gamma, beta);
    }
}
