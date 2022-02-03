package com.optim;

import com.data.Tensor;
import java.util.*;

public class Optimizer {
    public ArrayList<Tensor> params;
    public Map<String, Float> hyperParameters;

    public Optimizer(ArrayList<Tensor> params, Map<String, Float> hyperParameters) {
        this.params = params;
        this.hyperParameters = hyperParameters;
    }

    public void zero_grad() {
        for (Tensor param : params) {
            if (param.requires_grad_) {
                param.zero_grad();
            }
        }
    }

    public void step() {
    }
}
