#!/bin/bash

datasets=("Waveform" "InternetAds" "Arrhythmia" "landsat" "Hepatitis" "satimage-2" "annthyroid" "cardio" "Cardiotocography" "fault" "letter" "Ionosphere" "WPBC" "satellite" "SpamBase" "optdigits" "mnist" "musk" "speech" "CIFAR10_0" "FashionMNIST_0" "SVHN_0" "20news_0" "MVTec-AD_cable")
integers=(10 130 50 60 5 50 150 2000 62 15 400 300 50 5 62 10 10 100 100 17 100 23 200 200)

for i in "${!datasets[@]}"
do
    dataset="${datasets[$i]}"
    integer="${integers[$i]}"

    echo "Processing dataset: $dataset with integer: $integer"
    python Novelty.py --data "$dataset" --stop_epochs "$integer"
done
