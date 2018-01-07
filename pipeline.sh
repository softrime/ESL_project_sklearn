#!/bin/bash

model=KNN
for  ((k=1;k<50;k=k+1));do
  python train.py ${model} ${k}
done

model=LR
for  ((C=1;C<100;C=C+5));do
  python train.py ${model} ${C}
done

model=SVM
kernel=rbf
for ((C=1;C<100;C=C+5));do
  python train.py ${model} ${kernel} ${C}
done

kernel=linear
for ((C=1;C<100;C=C+5));do
  python train.py ${model} ${kernel} ${C}
done

kernel=poly
for ((C=1;C<100;C=C+5));do
  python train.py ${model} ${kernel} ${C}
done

kernel=sigmoid
for ((C=1;C<100;C=C+5));do
  python train.py ${model} ${kernel} ${C}
done
