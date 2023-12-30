# Result of Scripts Running

1. LeNet-5 with MNIST

    ```[Bach]
    ../data/mnist/
    mnist train number: 60000
    mnist test number: 10000
    weight: 25 x 6
    weight: 150 x 16
    0-th batch, loss: 2.30106
    50-th batch, loss: 2.30161
    100-th batch, loss: 2.30255
    150-th batch, loss: 2.30216
    200-th batch, loss: 2.30364
    250-th batch, loss: 2.30169
    300-th batch, loss: 2.29974
    350-th batch, loss: 2.3047
    400-th batch, loss: 2.29766
    450-th batch, loss: 2.30504

    1-th epoch, test acc: 0.1135

    0-th batch, loss: 2.29936
    50-th batch, loss: 2.30154
    100-th batch, loss: 2.29908
    150-th batch, loss: 2.30656
    200-th batch, loss: 2.29897
    250-th batch, loss: 2.29905
    300-th batch, loss: 2.29567
    350-th batch, loss: 2.29643
    400-th batch, loss: 2.29737
    450-th batch, loss: 2.29367

    2-th epoch, test acc: 0.2945

    0-th batch, loss: 2.29268
    50-th batch, loss: 2.25947
    100-th batch, loss: 1.4935
    150-th batch, loss: 0.656153
    200-th batch, loss: 0.484903
    250-th batch, loss: 0.371444
    300-th batch, loss: 0.394418
    350-th batch, loss: 0.344895
    400-th batch, loss: 0.147748
    450-th batch, loss: 0.207219

    3-th epoch, test acc: 0.946

    0-th batch, loss: 0.309088
    50-th batch, loss: 0.155788
    100-th batch, loss: 0.126408
    150-th batch, loss: 0.0778514
    200-th batch, loss: 0.0862488
    250-th batch, loss: 0.109127
    300-th batch, loss: 0.187663
    350-th batch, loss: 0.307188
    400-th batch, loss: 0.0749426
    450-th batch, loss: 0.233346

    4-th epoch, test acc: 0.9712

    0-th batch, loss: 0.106017
    50-th batch, loss: 0.225058
    100-th batch, loss: 0.1098
    150-th batch, loss: 0.0508292
    200-th batch, loss: 0.161406
    250-th batch, loss: 0.0563152
    300-th batch, loss: 0.0594433
    350-th batch, loss: 0.0520018
    400-th batch, loss: 0.0731089
    450-th batch, loss: 0.13004

    5-th epoch, test acc: 0.9737
    ```

2. LeNet-5 with Fashion-MNIST

    ```[Bach]
        ../data/fashion-mnist/
    mnist train number: 60000
    mnist test number: 10000
    weight: 25 x 6
    weight: 150 x 16
    0-th batch, loss: 2.30187
    50-th batch, loss: 2.30262
    100-th batch, loss: 2.30198
    150-th batch, loss: 2.30255
    200-th batch, loss: 2.30246
    250-th batch, loss: 2.30161
    300-th batch, loss: 2.30366
    350-th batch, loss: 2.30217
    400-th batch, loss: 2.30177
    450-th batch, loss: 2.30305

    1-th epoch, test acc: 0.1

    0-th batch, loss: 2.30196
    50-th batch, loss: 2.30234
    100-th batch, loss: 2.302
    150-th batch, loss: 2.30173
    200-th batch, loss: 2.29918
    250-th batch, loss: 2.2945
    300-th batch, loss: 2.25686
    350-th batch, loss: 1.64056
    400-th batch, loss: 1.09968
    450-th batch, loss: 0.895271

    2-th epoch, test acc: 0.6223

    0-th batch, loss: 0.971867
    50-th batch, loss: 0.85752
    100-th batch, loss: 0.882794
    150-th batch, loss: 0.819006
    200-th batch, loss: 0.613208
    250-th batch, loss: 0.582175
    300-th batch, loss: 0.819144
    350-th batch, loss: 0.564239
    400-th batch, loss: 0.732092
    450-th batch, loss: 0.458242

    3-th epoch, test acc: 0.7613

    0-th batch, loss: 0.630592
    50-th batch, loss: 0.576978
    100-th batch, loss: 0.603608
    150-th batch, loss: 0.73861
    200-th batch, loss: 0.670806
    250-th batch, loss: 0.409765
    300-th batch, loss: 0.630083
    350-th batch, loss: 0.487741
    400-th batch, loss: 0.57684
    450-th batch, loss: 0.591467

    4-th epoch, test acc: 0.7909

    0-th batch, loss: 0.550854
    50-th batch, loss: 0.46269
    100-th batch, loss: 0.406975
    150-th batch, loss: 0.548449
    200-th batch, loss: 0.416689
    250-th batch, loss: 0.460913
    300-th batch, loss: 0.708948
    350-th batch, loss: 0.484831
    400-th batch, loss: 0.531239
    450-th batch, loss: 0.476569

    5-th epoch, test acc: 0.8207

    ```

3. LeNet-5 with Fashion-MNIST; Transpose Matrix Multiplication

    ```[Bach]
    0-th batch, loss: 2.30187
    50-th batch, loss: 2.30262
    100-th batch, loss: 2.30198
    150-th batch, loss: 2.30255
    200-th batch, loss: 2.30246
    250-th batch, loss: 2.30161
    300-th batch, loss: 2.30366
    350-th batch, loss: 2.30217
    400-th batch, loss: 2.30177
    450-th batch, loss: 2.30305

    1-th epoch, test acc: 0.1

    0-th batch, loss: 2.30196
    50-th batch, loss: 2.30234
    100-th batch, loss: 2.302
    150-th batch, loss: 2.30173
    200-th batch, loss: 2.29919
    250-th batch, loss: 2.29452
    300-th batch, loss: 2.25696
    350-th batch, loss: 1.64363
    400-th batch, loss: 1.10052
    450-th batch, loss: 0.890389

    2-th epoch, test acc: 0.6231

    0-th batch, loss: 0.982203
    50-th batch, loss: 0.8619
    100-th batch, loss: 0.883574
    150-th batch, loss: 0.814321
    200-th batch, loss: 0.615839
    250-th batch, loss: 0.580562
    300-th batch, loss: 0.838665
    350-th batch, loss: 0.565908
    400-th batch, loss: 0.728824
    450-th batch, loss: 0.458302

    3-th epoch, test acc: 0.7602

    0-th batch, loss: 0.633762
    50-th batch, loss: 0.577861
    100-th batch, loss: 0.612528
    150-th batch, loss: 0.725019
    200-th batch, loss: 0.652117
    250-th batch, loss: 0.411808
    300-th batch, loss: 0.626963
    350-th batch, loss: 0.49442
    400-th batch, loss: 0.56591
    450-th batch, loss: 0.585407

    4-th epoch, test acc: 0.7898

    0-th batch, loss: 0.556282
    50-th batch, loss: 0.47116
    100-th batch, loss: 0.391514
    150-th batch, loss: 0.543599
    200-th batch, loss: 0.431745
    250-th batch, loss: 0.475138
    300-th batch, loss: 0.696249
    350-th batch, loss: 0.491586
    400-th batch, loss: 0.538841
    450-th batch, loss: 0.493691

    5-th epoch, test acc: 0.8194
    ```
