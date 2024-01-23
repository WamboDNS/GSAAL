echo "Running all datasets for the MV experiments" 
#echo "Running spiral"
#python Gridding.py --data spiral --stop_epoch 100 --lr_g 0.005
#echo "Running sinusoidal_wave"
#python Gridding.py --data sinusoidal_wave --stop_epoch 200 --lr_g 0.005 --gpu 1
echo "Running banana" 
python Gridding.py --data banana --stop_epochs 200 --batch_size  100
#echo "Runnin X"
#python Gridding.py --data X --stop_epochs 500 --gpu 1
#echo "Running L"
#python Gridding.py --data L --stop_epoch 200 --lr_g 0.005 --gpu 1
#echo "Running star" *Done*
#python Gridding.py --data star --stop_epoch 300
#echo "Running circle"
#python Gridding.py --data circle --stop_epoch 200 --lr_g 0.05 --gpu 1

