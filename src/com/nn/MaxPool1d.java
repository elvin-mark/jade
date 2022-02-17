package com.nn;

import com.data.Tensor;
import com.functions.F;

public class MaxPool1d extends NNModule {
    int kernel;

    public MaxPool1d(int kernel) {
        super();
        this.kernel = kernel;
    }

    public Tensor forward(Tensor input) {
        return F.maxpool1d(input, this.kernel);
    }
}
