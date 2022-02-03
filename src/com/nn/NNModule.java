package com.nn;

import java.util.ArrayList;
import com.data.Tensor;

public class NNModule {
    ArrayList<Tensor> params;
    String moduleName;

    public NNModule() {
        this.moduleName = "NNModule";
        this.params = new ArrayList<Tensor>();
    }

    public Tensor forward(Tensor input) {
        return input;
    }

    public ArrayList<Tensor> parameters() {
        return this.params;
    }

    public String name() {
        return this.moduleName;
    }

    public void add_module(NNModule nnModule) {
        // Implement this method
    }
}
