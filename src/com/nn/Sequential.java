package com.nn;

import java.util.*;
import com.data.Tensor;

public class Sequential extends NNModule {
    public Sequential() {

        super();
        this.moduleName = "Sequential";
        this.modules = new ArrayList<NNModule>();
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
