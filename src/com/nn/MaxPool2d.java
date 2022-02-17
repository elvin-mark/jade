package com.nn;

import com.data.Tensor;
import com.functions.F;

public class MaxPool2d extends NNModule {
    int[] kernel;

    public MaxPool2d(int kernel_size) {
        this(new int[] { kernel_size, kernel_size });
    }

    public MaxPool2d(int[] kernel) {
        super();
        this.kernel = kernel;
        this.moduleName = "MaxPool2d";
    }

    public Tensor forward(Tensor input) {
        return F.maxpool2d(input, this.kernel);
    }
}
