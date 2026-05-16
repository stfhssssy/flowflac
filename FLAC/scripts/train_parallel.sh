
#!/bin/bash

# LAC hyperparameter search for Humanoid Bench
log_dir="hb_opensource"
mkdir -p $log_dir

# Humanoid Bench tasks
tasks=(h1-walk-v0
    h1-crawl-v0
    h1-slide-v0
    h1-stand-v0
    h1-run-v0
    h1-pole-v0
    h1-sit_hard-v0
    h1-balance_simple-v0
    h1-maze-v0
    h1-sit_simple-v0
    h1-sit_hard-v0
    h1-stair-v0
    h1-hurdle-v0
    )


seeds=(1 2 3 4 5)

# LAC: Target kinetic energy coefficient search
target_kinetic_coefs=(2.5)

# LAC: Initial log_alpha values
init_log_alphas=(0)


NUM_GPUS=8
MAX_TASKS_PER_GPU=2
MAX_CONCURRENT=$((NUM_GPUS * MAX_TASKS_PER_GPU))

function current_jobs {
    jobs -rp | wc -l
}

task_id=0
for task in "${tasks[@]}"; do
    for seed in "${seeds[@]}"; do
        for coef in "${target_kinetic_coefs[@]}"; do
            for init_alpha in "${init_log_alphas[@]}"; do
                gpu=$(( (task_id / MAX_TASKS_PER_GPU) % NUM_GPUS ))
                while [ $(current_jobs) -ge $MAX_CONCURRENT ]; do
                    sleep 1
                done
                echo "Launching LAC: seed=$seed, task=$task, target_kinetic_coef=$coef, init_log_alpha=$init_alpha on GPU $gpu"
                CUDA_VISIBLE_DEVICES=$gpu python3 mainhb.py \
                    --task $task \
                    --seed $seed \
                    --target_kinetic_coef $coef \
                    --init_log_alpha $init_alpha \
                    > "${log_dir}/lac_${task}_seed${seed}_coef${coef}_alpha${init_alpha}.log" 2>&1 &
                ((task_id++))
            done
        done
    done
done

wait
echo "LAC Humanoid Bench hyperparameter search completed"
