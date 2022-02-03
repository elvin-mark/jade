package com.nn;

import com.data.Tensor;
import com.functions.F;

public class Tanh extends NNModule {

    public Tanh() {
        super();
        this.moduleName = "Tanh";
    }

    public Tensor forward(Tensor input) {
        return F.tanh(input);
    }

}
