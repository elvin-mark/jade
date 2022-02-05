package com.test;

import com.data.Tensor;
import com.functions.F;
import com.nn.*;

public class Test {
    public static void testDraft() {
        // Sequential model = new Sequential();
        // model.add_module((NNModule) new Conv2d(1, 4, 2, true));
        // model.add_module((NNModule) new ReLU());
        // model.add_module((NNModule) new Conv2d(4, 8, 2, true));
        // model.add_module((NNModule) new ReLU());
        // model.add_module((NNModule) new Conv2d(8, 16, 2, true));
        // Tensor x = new Tensor(new int[] { 2, 1, 10, 10 });
        // x.randn(0.0f, 1.0f);
        // Tensor out = model.forward(x);
        // for (int i : out.shape) {
        // System.out.print(i + " ");
        // }
        // System.out.println();
        Tensor t = new Tensor(new int[] { 2, 2, 5, 5 }, new float[] { 0.4962566f, 0.7682218f, 0.08847743f, 0.13203049f,
                0.30742282f, 0.6340787f, 0.4900934f, 0.89644474f, 0.45562798f, 0.6323063f, 0.34889346f, 0.4017173f,
                0.022325754f, 0.16885895f, 0.29388845f, 0.5185218f, 0.6976676f, 0.8000114f, 0.16102946f, 0.28226858f,
                0.68160856f, 0.915194f, 0.3970999f, 0.8741559f, 0.41940832f, 0.55290705f, 0.9527381f, 0.03616482f,
                0.18523103f, 0.37341738f, 0.30510002f, 0.9320004f, 0.17591017f, 0.26983356f, 0.15067977f, 0.031719506f,
                0.20812976f, 0.929799f, 0.7231092f, 0.7423363f, 0.5262958f, 0.24365824f, 0.58459234f, 0.03315264f,
                0.13871688f, 0.242235f, 0.81546897f, 0.7931606f, 0.27825248f, 0.4819588f, 0.81978035f, 0.99706656f,
                0.6984411f, 0.5675464f, 0.83524317f, 0.20559883f, 0.593172f, 0.112347245f, 0.15345693f, 0.24170822f,
                0.7262365f, 0.7010802f, 0.20382375f, 0.65105355f, 0.774486f, 0.43689132f, 0.5190908f, 0.61585236f,
                0.8101883f, 0.98009706f, 0.11468822f, 0.31676513f, 0.69650495f, 0.9142747f, 0.93510365f, 0.9411784f,
                0.5995073f, 0.06520867f, 0.54599625f, 0.18719733f, 0.034022927f, 0.94424623f, 0.8801799f, 0.0012360215f,
                0.593586f, 0.41577f, 0.41771942f, 0.27112156f, 0.6922781f, 0.20384824f, 0.68329567f, 0.75285405f,
                0.8579358f, 0.6869556f, 0.005132377f, 0.17565155f, 0.7496575f, 0.6046507f, 0.10995799f, 0.21209025f

        });
        // t.randn(0.0f, 1.0f);
        t.ones();
        t.requires_grad(true);
        // System.out.println(t);
        // Tensor o = t.maxpool2d(new int[] { 2, 2 });
        Tensor o = t.dropout2d(0.5f);
        System.out.println(o);
        Tensor l = o.norm();
        l.backward();
        System.out.println(l);
        System.out.println(t.grad);
        System.out.println();
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
