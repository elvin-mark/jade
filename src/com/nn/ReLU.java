package com.nn;

import com.data.Tensor;
import com.functions.F;

public class ReLU extends NNModule {

    public ReLU() {
        super();
        this.moduleName = "ReLU";
    }

    public Tensor forward(Tensor input) {
        return F.relu(input);
    }

}
