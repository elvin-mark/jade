package com.optim;

import com.data.Tensor;
import java.util.*;

public class Optimizer {
    public Tensor[] params;
    public Map<String, Float> hyperParameters;

    public Optimizer(Tensor[] params, Map<String, Float> hyperParameters) {
        this.params = params;
        this.hyperParameters = hyperParameters;
    }

    public void step() {
    }
}
