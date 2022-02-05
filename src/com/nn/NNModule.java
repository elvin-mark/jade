package com.nn;

import java.util.ArrayList;
import com.data.Tensor;
import com.utils.Misc;

public class NNModule {
    ArrayList<Tensor> params;
    String moduleName;
    boolean training;

    public NNModule() {
        this.moduleName = "NNModule";
        this.params = new ArrayList<Tensor>();
        this.training = true;
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

    public void train() {
        // IMPROVE THIS: maybe add this to each parameter (Tensor)?
        this.training = true;
    }

    public void eval() {
        // IMPROVE THIS: maybe add this to each parameter (Tensor)?
        this.training = false;
    }

    public void save_parameters(String filename) {
        // Implement this method

    }

    public void load_parameters(String filename) {
        // Implement this method

    }
}
