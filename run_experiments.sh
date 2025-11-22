for ds in obqa arc piqa riddle bioasq pubmedqa; do
        python main.py -d $ds --lr 5e-3 --run 88
    done
done




# for ds in obqa arc piqa riddle bioasq pubmedqa; do
#     for tw in 0.01 0.1 0.5 1 2 3; do
#         python main.py -d $ds -t $tw $tw --run 42
#     done
# done

# 2e-2 5e-2 8e-2 2e-3 5e-3 8e-3 2e-4 5e-4 8e-4 2e-5 5e-5 8e-5
# for ds in obqa arc piqa riddle bioasq pubmedqa; do
#     for lr in 5e-2 5e-3 5e-4 5e-6 5e-7 5e-8; do
#         python main.py -d $ds --lr $lr --run 72
#     done
# done

# for ds in obqa arc piqa riddle bioasq pubmedqa; do
#     for tw in 0.01 0.1 0.5 1 2 3; do
#         python main.py -d $ds -t $tw $tw --lr 5e-4 --run 80
#     done
# done

# for ds in obqa arc piqa riddle bioasq pubmedqa; do
#     for bs in 4 16 32 64; do
#         python main.py -d $ds --batch_size $bs --lr 5e-4 --run 90
#     done
# done

##############################################################################

# for bs in 9 18 21 36 72; do
#     echo "Rodando experimento com batch size = $bs"
#     python experiment.005.py -d obqa --batch_size $bs
# done

# for bs in 9 18 36 72; do
#     echo "Rodando experimento com batch size = $bs"
#     python experiment.005.py -d arc --batch_size $bs
# done

# for bs in 9 18 36 72; do
#     echo "Rodando experimento com batch size = $bs"
#     python experiment.005.py -d piqa --batch_size $bs
# done

# for bs in 9 18 36 72; do
#     echo "Rodando experimento com batch size = $bs"
#     python experiment.005.py -d riddle --batch_size $bs
# done

# for bs in 9 18 36 72; do
#     echo "Rodando experimento com batch size = $bs"
#     python experiment.005.py -d bioasq --batch_size $bs
# done

# for bs in 9 18 36 72; do
#     echo "Rodando experimento com batch size = $bs"
#     python experiment.005.py -d pubmedqa --batch_size $bs
# done
