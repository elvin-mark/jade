package com.nn;

import java.util.ArrayList;
import com.data.Tensor;
import com.utils.Misc;
import java.io.File;

public class NNModule {
    ArrayList<Tensor> params;
    ArrayList<NNModule> modules;
    String moduleName;
    boolean training;

    public NNModule() {
        this.moduleName = "NNModule";
        this.params = new ArrayList<Tensor>();
        this.modules = new ArrayList<NNModule>();
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

    public void add_module(NNModule module) {
        this.modules.add(module);
        for (Tensor t : module.params)
            this.params.add(t);
    }

    public void train() {
        // IMPROVE THIS: maybe add this to each parameter (Tensor)?
        this.training = true;
        for (NNModule m : this.modules)
            m.train();
    }

    public void eval() {
        // IMPROVE THIS: maybe add this to each parameter (Tensor)?
        this.training = false;
        for (NNModule m : this.modules)
            m.eval();
    }

    public void save_parameters(String path) {
        // Implement this method
        if (!new File(path).exists()) {
            new File(path).mkdir();
        }
        for (int i = 0; i < this.params.size(); i++) {
            Misc.saveTensor(this.params.get(i), path + "/" + this.name() + "_" + i + ".bin");
        }
    }

    public void load_parameters(String path) {
        // Implement this method
        String file_path;
        for (int i = 0; i < this.params.size(); i++) {
            file_path = path + "/" + this.name() + "_" + i + ".bin";
            if (!new File(file_path).exists()) {
                throw new RuntimeException("File " + file_path + " does not exist");
            }
            this.params.get(i).data = Misc.loadTensor(file_path).data;
        }
    }
}
