
for bs in 9 18 36 72; do
    echo "Rodando experimento com batch size = $bs"
    python experiment.005.py -d obqa --batch_size $bs
done



# for bs in 9 18 36 72 144; do
#     for lr in 2e-3 5e-3 2e-4 5e-4 2e-5 5e-5; do
#         echo "Rodando experimento com batch size = $bs and learning rate = $lr"
#         python experiment.004.py -d obqa --batch_size $bs --lr $lr
# done