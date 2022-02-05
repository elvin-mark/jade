package com.nn;

import com.data.Tensor;
import com.functions.F;

public class MaxPool2d extends NNModule {
    int[] kernel;

    public MaxPool2d(int[] kernel) {
        super();
        this.kernel = kernel;
    }

    public Tensor forward(Tensor input) {
        return F.maxpool2d(input, this.kernel);
    }
}
