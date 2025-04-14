@echo off
setlocal

set TRAIN=train.json
set VAL=val.json

(for %%H in (64 128 256) do (
    for %%E in (3 5 10) do (
        echo Running hidden_dim=%%H, epochs=%%E
        python rnn.py --hidden_dim %%H --epochs %%E --train_data %TRAIN% --val_data %VAL% --do_train > resultsRNN\results_%%H_%%E.txt
    )
)) 
