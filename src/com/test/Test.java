package com.test;

import com.data.Tensor;
import com.functions.F;
import com.nn.*;

public class Test {
    public static void testTensors() {
        int[] shape1 = { 2, 2 };
        int[] shape2 = { 2, 2 };
        int[] shape3 = { 2 };

        float[] data1 = { 1.0f, 2.0f, 3.0f, 4.0f };
        float[] data2 = { 2.0f, 3.0f, 4.0f, 5.0f };
        float[] data3 = { 8.0f, 9.0f };

        Tensor t1 = new Tensor(data1);
        Tensor t2 = new Tensor(data2);
        Tensor b = new Tensor(data3);
        Tensor c = new Tensor(2.3f);

        t1.requires_grad(true);

        t1.reshape(new int[] { 2, 2 });
        t2.reshape(new int[] { 2, 2 });

        Tensor t3 = F.linear(t1, t2, b);
        t1.print();
        t2.print();
        t3.print();
        System.out.println(t3.requires_grad_);
        System.out.println(t3);

        System.out.println(t1.mul(c));
    }

    public static void testModule() {
        NNModule seq = new Sequential();
        seq.add_module(new NNModule());

        seq.add_module((NNModule) new Linear(2, 2, true));
        seq.add_module((NNModule) new Sigmoid());
        seq.add_module((NNModule) new Linear(2, 1, true));
        seq.add_module((NNModule) new Sigmoid());

        Tensor input = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        Tensor target = new Tensor(new float[] { 0.0f, 1.0f });

        input.reshape(new int[] { 2, 2 });
        target.reshape(new int[] { 2, 1 });

        Tensor output = seq.forward(input);

        System.out.println(output);
        System.out.println(output.requires_grad_);

        for (Tensor param : seq.parameters()) {
            param.print();
        }

        Loss loss_fn = new MSELoss();
        Tensor loss = loss_fn.criterion(output, target);
        loss.backward();
        System.out.println(loss);
    }

    public static void main(String args[]) {
        testModule();
    }
}
