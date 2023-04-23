package com.models;

import java.util.*;

import com.nn.Linear;
import com.nn.NNModule;
import com.nn.Sigmoid;
import com.data.Tensor;

public class MLP extends NNModule {
    public MLP(int inp, int hidden, int out) {
        super();
        this.moduleName = "MLP";
        this.modules = new ArrayList<NNModule>();
        this.add_module(new Linear(inp, hidden, true));
        this.add_module(new Sigmoid());
        this.add_module(new Linear(hidden, out, true));
    }

    public Tensor forward(Tensor input) {
        Tensor o = input;
        o = this.modules.get(0).forward(o);
        o = this.modules.get(1).forward(o);
        o = this.modules.get(2).forward(o);
        return o;
    }

    public String toString() {
        String s = "";
        s += this.modules.get(0).toString() + "\n";
        s += this.modules.get(1).toString() + "\n";
        s += this.modules.get(2).toString() + "\n";
        return s;
    }
}
