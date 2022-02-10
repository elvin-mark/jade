package com.optim;

import com.data.Tensor;
import java.util.*;

public class SGD extends Optimizer {
    public Tensor[] m = null;
    public Tensor lr = null;
    public Tensor momentum = null;

    public SGD(ArrayList<Tensor> params, Map<String, Float> hyperParameters) {
        super(params, hyperParameters);
        if (hyperParameters.containsKey("lr")) {
            lr = new Tensor(hyperParameters.get("lr"));
        } else {
            lr = new Tensor(0.01f);
        }

        if (hyperParameters.containsKey("momentum")) {
            momentum = new Tensor(hyperParameters.get("momentum"));
            m = new Tensor[params.size()];
            for (int i = 0; i < params.size(); i++) {
                m[i] = new Tensor(params.get(i).shape);
                m[i].zeros();
            }
        }
    }

    public void step() {
        for (int i = 0; i < params.size(); i++) {
            Tensor p = params.get(i);
            if (p.requires_grad_) {
                Tensor g = p.grad;
                if (momentum != null) {
                    Tensor m = this.m[i];
                    m.data = m.mul(momentum).add(g).data;
                    p.data = p.sub(m.mul(lr)).data;
                } else {
                    p.data = p.sub(g.mul(lr)).data;
                }
            }
        }
    }
}
