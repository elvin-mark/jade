package com.test;

import com.data.Tensor;
import com.functions.F;
import com.nn.*;

public class Test {
    public static void testDraft() {
        Tensor t = new Tensor(new int[] { 2, 2, 5 },
                new float[] { 0.9712881f, 0.9923937f, 0.8624685f, 0.5829317f, 0.8995149f,
                        0.72404754f, 0.46881002f,
                        0.68940943f, 0.21434641f, 0.6536304f, 0.004905939f, 0.6508469f, 0.7345901f,
                        0.44306612f,
                        0.4500317f, 0.24441409f, 0.084246516f, 0.94214594f, 0.93758404f, 0.39994133f
                });
        Tensor f = new Tensor(new int[] { 3, 2, 2 }, new float[] { 0.41704082f,
                0.82422876f, 0.7174187f, 0.4910134f,
                0.008470118f, 0.20370054f, 0.36883116f, 0.8914987f, 0.29231638f, 0.3895387f,
                0.0074648857f,
                0.57268023f });

        t.requires_grad(true);
        f.requires_grad(true);

        Tensor o = t.conv1d(f, 1, 0);
        Tensor l = o.norm();

        l.backward();

        System.out.println(l);
        System.out.println(t.grad);
        System.out.println(f.grad);

        for (int i : o.shape()) {
            System.out.println(i);
        }
        System.out.println(o);

        // Tensor t = new Tensor(new int[] { 2, 2, 5, 5 },
        // new float[] { 0.029287994f, 0.68998295f, 0.5212799f, 0.8255198f, 0.4968027f,
        // 0.6793015f, 0.7112422f,
        // 0.65369105f, 0.916024f, 0.3646918f, 0.5418354f, 0.7812271f, 0.14167666f,
        // 0.22757393f,
        // 0.07110268f, 0.67199457f, 0.40220606f, 0.43592763f, 0.0091763735f,
        // 0.35617954f, 0.47424626f,
        // 0.14259946f, 0.6191177f, 0.1728484f, 0.69607437f, 0.48074144f, 0.2583788f,
        // 0.5505779f,
        // 0.6777148f, 0.044979215f, 0.44581264f, 0.32109624f, 0.9555772f, 0.02372849f,
        // 0.82700837f,
        // 0.831391f, 0.7557432f, 0.037344694f, 0.81794053f, 0.56090915f, 0.9799189f,
        // 0.71223277f,
        // 0.9229185f, 0.36649823f, 0.612735f, 0.15279317f, 0.5656025f, 0.45752758f,
        // 0.9684501f,
        // 0.28292632f, 0.55830604f, 0.17242742f, 0.0064552426f, 0.029224694f,
        // 0.057399154f, 0.20213443f,
        // 0.28989154f, 0.49094254f, 0.87080926f, 0.055322707f, 0.6113274f, 0.8396533f,
        // 0.16460228f,
        // 0.3232631f, 0.2713793f, 0.6608822f, 0.79126734f, 0.5430018f, 0.31630087f,
        // 0.017933011f,
        // 0.37349313f, 0.69943386f, 0.38717318f, 0.33282256f, 0.7931707f, 0.5468885f,
        // 0.48154503f,
        // 0.60112995f, 0.621f, 0.4747839f, 0.96085536f, 0.27715355f, 0.08662361f,
        // 0.73021835f,
        // 0.89718604f, 0.28494453f, 0.81858164f, 0.6801523f, 0.38913584f, 0.343027f,
        // 0.36820477f,
        // 0.032734454f, 0.39637905f, 0.39350814f, 0.29886866f, 0.49656576f, 0.741266f,
        // 0.33732837f,
        // 0.67659086f, 0.84850794f });
        // Tensor f = new Tensor(new int[] { 3, 2, 2, 2 }, new float[] { 0.2728427f,
        // 0.24315017f, 0.8356812f, 0.11708015f,
        // 0.02459389f, 0.536625f, 0.848438f, 0.036521852f, 0.83922255f, 0.24925536f,
        // 0.48304248f, 0.2815041f,
        // 0.39941067f, 0.802756f, 0.26303077f, 0.06542307f, 0.60303646f, 0.5547626f,
        // 0.893597f, 0.37561542f,
        // 0.28431123f, 0.04780817f, 0.9418152f, 0.034682572f });

        // Tensor o = t.conv2d(f, new int[] { 2, 2 }, new int[] { 1, 1 });

        // for (int i : o.shape()) {
        // System.out.println(i);
        // }
        // System.out.println(o);
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
