package com.nn;

import java.util.*;
import com.data.Tensor;

public class Sequential extends NNModule {
    ArrayList<NNModule> modules;

    public Sequential() {

        super();
        this.moduleName = "Sequential";
        this.modules = new ArrayList<NNModule>();
    }

    public void add_module(NNModule module) {
        this.modules.add(module);
        for (Tensor t : module.params)
            this.params.add(t);
    }

    public Tensor forward(Tensor input) {
        for (NNModule m : this.modules)
            input = m.forward(input);
        return input;
    }

    public String toString() {
        String s = "";
        for (NNModule m : this.modules)
            s += m.toString() + "\n";
        return s;
    }
}
