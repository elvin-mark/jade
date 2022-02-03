package com.test;

import com.nn.*;
import com.optim.*;
import com.data.*;
import java.util.*;

public class SampleLinearRegression {
    public static void main(String args[]) {
        float[] X = new float[] { 1, 2, 3, 4, 5 };
        float[] y = new float[] { 7, 9, 11, 13, 15 };

        Tensor x_train = new Tensor(new int[] { 5, 1 }, X);
        Tensor y_train = new Tensor(new int[] { 5, 1 }, y);

        NNModule seq = new Sequential();
        seq.add_module((NNModule) new Linear(1, 1, true));

        Loss loss_fn = new MSELoss();
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

        for (Tensor param : seq.parameters()) {
            param.print();
        }
    }
}
