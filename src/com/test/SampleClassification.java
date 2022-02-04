package com.test;

import com.nn.*;
import com.optim.*;
import com.data.*;
import java.util.*;

public class SampleClassification {
    public static void main(String args[]) {
        Tensor x_train = new Tensor(new int[] { 4, 2 }, new float[] { 1.0f, 5.0f, 2.0f, 4.0f, -1.f, -4.f, -2.f, -3.f });
        Tensor y_train = new Tensor(new int[] { 4, 1 }, new float[] { 0.0f, 0.0f, 1.0f, 1.0f });

        NNModule seq = new Sequential();
        seq.add_module((NNModule) new Linear(2, 5, true));
        seq.add_module((NNModule) new Sigmoid());
        seq.add_module((NNModule) new Linear(5, 2, true));

        Loss loss_fn = new CrossEntropyLoss();
        Map<String, Float> optim_params = new HashMap<String, Float>();
        optim_params.put("lr", 0.01f);
        Optimizer optim = new SGD(seq.parameters(), optim_params);

        for (int epoch = 0; epoch < 1000; epoch++) {
            Tensor o = seq.forward(x_train);
            Tensor loss = loss_fn.criterion(o, y_train);
            if (epoch % 100 == 0) {
                System.out.println(epoch + ": " + loss);
            }
            optim.zero_grad();
            loss.backward();
            optim.step();
        }
        System.out.println(seq.forward(x_train));
        for (Tensor param : seq.parameters()) {
            param.print();
        }
    }
}
