package com.test;

import com.data.Tensor;
import com.functions.F;
import com.nn.*;
import com.utils.*;
import com.optim.*;

import java.util.HashMap;

public class Test {
        public static void testDraft() {
                // Some random tests
                Sequential model = new Sequential();

                model.add_module((NNModule) new Conv2d(1, 16, 3, 1, 1, true));
                model.add_module((NNModule) new Sigmoid());
                model.add_module((NNModule) new Conv2d(16, 16, 3, 1, 1, true));
                model.add_module((NNModule) new Sigmoid());
                model.add_module((NNModule) new MaxPool2d(2));
                model.add_module((NNModule) new Conv2d(16, 32, 3, 1, 1, true));
                model.add_module((NNModule) new Sigmoid());
                model.add_module((NNModule) new Conv2d(32, 32, 3, 1, 1, true));
                model.add_module((NNModule) new Sigmoid());
                model.add_module((NNModule) new Flatten());
                model.add_module((NNModule) new Linear(32 * 4 * 4, 10, true));

                // model.load_parameters("./my_model");

                // Tensor x = Misc.loadTensor("./x.bin");
                // Tensor y = Misc.loadTensor("./y.bin");

                // System.out.println(model.parameters().get(8));

                // Loss loss_fn = new CrossEntropyLoss();
                // Tensor o = model.forward(x);
                // System.out.println(o);
                // System.out.println(loss_fn.criterion(o, y));

                String path_to_digits = "./data/DIGITS/";
                DataLoader[] dl = com.vision.Datasets.loadDIGITS(path_to_digits, 32);

                HashMap<String, Float> hyperparams = new HashMap<String, Float>();
                hyperparams.put("lr", 0.001f);
                Optimizer optim = new SGD(model.parameters(), hyperparams);
                Loss loss_fn = new CrossEntropyLoss();

                Misc.train(model, dl[0], dl[1], optim, loss_fn, 20);
                System.out.println(Misc.eval(model, dl[0], loss_fn)[0]);
                System.out.println(Misc.eval(model, dl[1], loss_fn)[0]);
        }

        public static void main(String args[]) {
                testDraft();
                // testRandom();
        }
}
