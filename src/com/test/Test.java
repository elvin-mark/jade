package com.test;

import com.data.Tensor;
import com.functions.F;
import com.nn.*;
import com.utils.Misc;

public class Test {
    public static void testDraft() {
        Tensor x_train = Misc.loadTensor("/home/elvin/Downloads/x_train_digits.bin");
        Tensor y_train = Misc.loadTensor("/home/elvin/Downloads/y_train_digits.bin");
        System.out.println(x_train.data[0]);
        System.out.println(y_train.data[5]);
    }

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

    public static void testLoss() {
        Tensor t1_ = new Tensor(new int[] { 3, 2, 2, 3 },
                new float[] { 0.5289f, 0.4626f, 0.5518f, 0.3476f, 0.5939f, 0.8576f, 0.4022f, 0.0636f, 0.8197f,
                        0.3296f, 0.8923f, 0.9391f, 0.6479f, 0.0469f, 0.9633f, 0.4420f, 0.1681f, 0.2387f,
                        0.0612f, 0.5698f, 0.6793f, 0.6686f, 0.5192f, 0.6727f, 0.4391f, 0.2874f, 0.0601f,
                        0.0318f, 0.4551f, 0.5309f, 0.1438f, 0.5036f, 0.6353f, 0.2046f, 0.4513f, 0.5982f });

        t1_.requires_grad(true);
        Tensor tmp = t1_.logsoftmax();
        Tensor t2_ = new Tensor(new int[] { 3, 1, 2, 3 },
                new float[] { 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1 });
        CrossEntropyLoss l_ = new CrossEntropyLoss();
        Tensor loss_ = l_.criterion(tmp, t2_);
        System.out.println(loss_);
        System.out.println(loss_.requires_grad_);
        loss_.backward();
        System.out.println(t1_.grad);
        Tensor t = new Tensor(new int[] { 4, 2 }, new float[] { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f });
        System.out.println(t.softmax());
        Tensor t1 = t.logsoftmax();
        Tensor t2 = new Tensor(new int[] { 4, 1 }, new float[] { 1, 0, 0, 1 });

        NLLLoss l = new NLLLoss();
        Tensor o = l.criterion(t1, t2);
        System.out.println(o.item());
    }

    public static void main(String args[]) {
        // testModule();
        // testBackward();
        // testLoss();
        testDraft();
    }
}
