package com.nn;

import com.data.Tensor;
import com.functions.F;

public class Softmax extends NNModule {
    public Softmax() {
        super();
        this.moduleName = "Softmax";
    }

    public Tensor forward(Tensor input) {
        return F.softmax(input);
    }
}
