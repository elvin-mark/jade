package com.test;

import com.data.Tensor;
import com.nn.*;

public class SampleConvClassification {
    public static void main(String args[]) {
        Sequential model = new Sequential();
        Tensor x = new Tensor(new int[] { 2, 1, 28, 28 });
        model.add_module((NNModule) new Conv2d(1, 8, 3, true));
        model.add_module((NNModule) new Tanh());
        model.add_module((NNModule) new Conv2d(8, 16, 3, true));
        model.add_module((NNModule) new Tanh());
        model.add_module((NNModule) new Flatten());
        model.add_module((NNModule) new Linear(16 * 4 * 4, 10, true));

        Tensor output = model.forward(x);
        System.out.println(output);
        for (int i : output.shape) {
            System.out.print(i + " ");
        }
    }

}
