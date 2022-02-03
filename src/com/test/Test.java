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

        Tensor tmp = input.transpose();
        System.out.println(tmp.at(new int[] { 0, 1 }));

        System.out.println(input.transpose().mm(input));
        System.out.println(input.mm(input));

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

        for (Tensor param : seq.parameters()) {
            System.out.println("param: " + param);
            System.out.println("grad: " + param.grad);
        }
    }

    public static void testBackward() {
        Tensor t1 = new Tensor(1.2f);
        Tensor t2 = new Tensor(2.3f);
        Tensor t3 = new Tensor(3.4f);
        Tensor t4 = new Tensor(4.5f);
        Tensor A = new Tensor(new int[] { 2, 2 });
        A.random(0.0f, 1.0f);
        A.requires_grad(true);
        Tensor tmp = A.mean();
        tmp.backward();
        System.out.println(A);
        System.out.println(tmp);
        System.out.println(A.grad);

        t1.requires_grad(true);
        t2.requires_grad(true);
        t3.requires_grad(true);
        t4.requires_grad(true);

        Tensor t5 = t4.div(t1.mul(t2).add(t3));
        t5.backward();
        System.out.println(t5);
        System.out.println(t1.grad);
        System.out.println(t2.grad);
        System.out.println(t3.grad);
        System.out.println(t4.grad);
    }

    public static void main(String args[]) {
        testModule();
        // testBackward();
    }
}
