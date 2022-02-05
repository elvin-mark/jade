package com.nn;

import com.data.Tensor;
import com.functions.F;

public class BatchNorm2d extends NNModule {
    int numFeatures;

    public BatchNorm2d(int numFeatures) {
        super();
        this.numFeatures = numFeatures;
        Tensor runningMean = new Tensor(new int[] { numFeatures });
        Tensor runningVar = new Tensor(new int[] { numFeatures });
        Tensor gamma = new Tensor(new int[] { numFeatures });
        Tensor beta = new Tensor(new int[] { numFeatures });

        runningMean.zeros();
        runningVar.ones();
        gamma.ones();
        beta.zeros();

        runningMean.requires_grad(true);
        runningVar.requires_grad(true);
        gamma.requires_grad(true);
        beta.requires_grad(true);

        this.params.add(runningMean);
        this.params.add(runningVar);
        this.params.add(gamma);
        this.params.add(beta);
    }

    public Tensor forward(Tensor input) {
        Tensor runningMean = this.params.get(0);
        Tensor runningVar = this.params.get(1);
        Tensor gamma = this.params.get(2);
        Tensor beta = this.params.get(3);

        return F.batchnorm2d(input, runningMean, runningVar, gamma, beta);
    }
}
