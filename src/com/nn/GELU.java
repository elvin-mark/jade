package com.nn;

import com.data.Tensor;
import com.functions.F;

public class GELU extends NNModule {
    public GELU() {
        super();
        this.moduleName = "GELU";
    }

    public Tensor forward(Tensor input) {
        return F.gelu(input);
    }
}
